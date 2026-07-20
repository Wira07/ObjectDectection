"""
recognize_face.py - Face recognition realtime pakai InsightFace,
berbasis dataset dari register_face.py (face_dataset/<Nama>/*.jpg).
Tidak menyentuh file YOLO utama.

Dipakai sebagai modul dari file lain:
    from recognize_face import recognize_face
    person_name, confidence = recognize_face(frame)
    # None        -> tidak ada wajah sama sekali
    # "Unknown"   -> ada wajah, tapi tidak dikenali
    # nama_orang  -> wajah dikenali
Model & database wajah otomatis dimuat sekali saat modul di-import.

Dipakai mandiri (testing):
    python recognize_face.py
    python recognize_face.py --rebuild   (paksa build ulang database)

pip install insightface onnxruntime opencv-python numpy
"""

import hashlib
import os
import pickle
import sys
import threading
import time

import cv2
import numpy as np

from insightface.app import FaceAnalysis


# ================= KONFIGURASI =================

CAMERA_URL = "http://192.168.1.9:8080/video"
FACE_DATASET_DIR = "face_dataset"
EMBEDDINGS_CACHE_PATH = os.path.join(FACE_DATASET_DIR, "embeddings_cache.pkl")
DET_SCORE_THRESHOLD = 0.55
RECOGNITION_THRESHOLD = 0.45   # makin tinggi = makin ketat, sesuaikan sendiri
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540


# ================= CAMERA (thread, anti-delay) =================

class CameraStream:
    """
    Baca kamera di thread terpisah & selalu simpan frame TERBARU saja.
    Mencegah delay akibat buffer numpuk saat proses recognition sibuk.
    """

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.lock = threading.Lock()
        self.ret, self.frame, self.stopped = False, None, False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


# ================= INIT MODEL =================

print("[recognize_face] Memuat model InsightFace...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))  # 320 = lebih cepat dari 640
print("[recognize_face] Model siap.")


# ================= DATABASE EMBEDDING =================

def _list_dataset_images(dataset_dir):
    """Kumpulkan semua foto di face_dataset -> {nama_orang: [path, ...]}."""
    result = {}
    if not os.path.isdir(dataset_dir):
        return result

    for person_name in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        paths = [os.path.join(person_dir, f) for f in sorted(os.listdir(person_dir))
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if paths:
            result[person_name] = paths
    return result


def _compute_dataset_signature(dataset_map):
    """Hash nama file + mtime, dipakai buat deteksi apakah dataset berubah."""
    parts = [f"{name}|{p}|{os.path.getmtime(p)}" for name, paths in dataset_map.items() for p in paths]
    return hashlib.md5("\n".join(sorted(parts)).encode("utf-8")).hexdigest()


def _extract_embedding(image_bgr):
    """Ambil embedding wajah terbesar dari 1 gambar. None kalau tidak ada wajah."""
    faces = face_app.get(image_bgr)
    if not faces:
        return None
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return best_face.normed_embedding


def _save_cache(signature, database):
    try:
        os.makedirs(FACE_DATASET_DIR, exist_ok=True)
        with open(EMBEDDINGS_CACHE_PATH, "wb") as f:
            pickle.dump({"signature": signature, "database": database}, f)
    except OSError as e:
        print(f"[recognize_face] Gagal menyimpan cache: {e}")


def build_face_database(force_rebuild=False):
    """
    Bangun database embedding dari face_dataset. Pakai cache kalau isi
    dataset belum berubah sejak terakhir build (lebih cepat).
    Return: {nama_orang: [embedding, ...]}
    """
    dataset_map = _list_dataset_images(FACE_DATASET_DIR)
    signature = _compute_dataset_signature(dataset_map)

    if not force_rebuild and os.path.exists(EMBEDDINGS_CACHE_PATH):
        try:
            with open(EMBEDDINGS_CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            if cache.get("signature") == signature:
                print("[recognize_face] Database dimuat dari cache.")
                return cache.get("database", {})
        except (pickle.PickleError, EOFError, KeyError):
            pass  # cache rusak/lama -> build ulang di bawah

    if not dataset_map:
        print("[recognize_face] PERINGATAN: face_dataset kosong. Semua wajah = 'Unknown'.")
        _save_cache(signature, {})
        return {}

    print("[recognize_face] Membangun database embedding...")
    database = {}
    for person_name, paths in dataset_map.items():
        embeddings = []
        for path in paths:
            image = cv2.imread(path)
            if image is None:
                print(f"  - Lewati (gagal dibaca): {path}")
                continue
            emb = _extract_embedding(image)
            if emb is None:
                print(f"  - Lewati (wajah tidak terdeteksi): {path}")
                continue
            embeddings.append(emb)

        if embeddings:
            database[person_name] = embeddings
            print(f"  - {person_name}: {len(embeddings)} foto diproses")
        else:
            print(f"  - {person_name}: tidak ada foto valid, dilewati")

    print(f"[recognize_face] Database selesai: {len(database)} orang terdaftar")
    _save_cache(signature, database)
    return database


FACE_DATABASE = build_face_database()  # dimuat sekali saat modul di-import


# ================= PENCOCOKAN WAJAH =================

def _cosine_similarity(a, b):
    """normed_embedding sudah dinormalisasi -> cosine similarity = dot product."""
    return float(np.dot(a, b))


def _match_embedding(embedding, database):
    """Bandingkan 1 embedding ke seluruh database. Return (nama, skor_terbaik)."""
    best_name, best_score = "Unknown", 0.0
    for person_name, ref_embeddings in database.items():
        for ref in ref_embeddings:
            score = _cosine_similarity(embedding, ref)
            if score > best_score:
                best_name, best_score = person_name, score

    if best_score < RECOGNITION_THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score


# ================= FUNGSI UTAMA (dipanggil dari file lain) =================

def recognize_faces_multi(frame):
    """Deteksi & kenali SEMUA wajah di 1 frame -> list {bbox, name, confidence}."""
    results = []
    for face in face_app.get(frame):
        if face.det_score < DET_SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, face.bbox)
        name, confidence = _match_embedding(face.normed_embedding, FACE_DATABASE)
        results.append({"bbox": (x1, y1, x2, y2), "name": name, "confidence": round(confidence, 4)})
    return results


def recognize_face(frame):
    """
    Fungsi modular utama - kenali wajah PALING DOMINAN (box terbesar) di frame.
    Return: (person_name, confidence)
        (None, 0.0)        -> tidak ada wajah
        ("Unknown", skor)  -> ada wajah, tidak dikenali
        (nama, skor)       -> wajah dikenali
    """
    all_faces = recognize_faces_multi(frame)
    if not all_faces:
        return None, 0.0
    primary = max(all_faces, key=lambda it: (it["bbox"][2] - it["bbox"][0]) * (it["bbox"][3] - it["bbox"][1]))
    return primary["name"], primary["confidence"]


def draw_face_box(frame, bbox, name, confidence):
    """Gambar bounding box + label. Hijau = dikenali, merah = Unknown. Opsional dipakai."""
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, max(0, y1 - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


# ================= MODE MANDIRI (testing) =================

if __name__ == "__main__":

    if "--rebuild" in sys.argv:
        print("[recognize_face] Paksa build ulang database (--rebuild)...")
        FACE_DATABASE = build_face_database(force_rebuild=True)

    print("Membuka kamera:", CAMERA_URL)
    cap = CameraStream(CAMERA_URL).start()

    if not cap.isOpened():
        print("Gagal membuka kamera. Cek URL IP Webcam.")
        sys.exit(1)

    # Tunggu frame pertama siap dari thread (maks ~5 detik)
    for _ in range(50):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.1)
    else:
        print("Timeout: tidak ada frame dari kamera.")
        cap.release()
        sys.exit(1)

    window_title = "Face Recognition - Testing"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, WINDOW_WIDTH, WINDOW_HEIGHT)

    print("Face Recognition berjalan. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        display_frame = frame.copy()
        for face_info in recognize_faces_multi(frame):
            draw_face_box(display_frame, face_info["bbox"], face_info["name"], face_info["confidence"])

        display_frame = cv2.resize(display_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow(window_title, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Dihentikan oleh user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program recognize_face.py selesai.")