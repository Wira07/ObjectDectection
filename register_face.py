"""
register_face.py - Registrasi dataset wajah pakai InsightFace + IP Webcam.
Tidak menyentuh file YOLO utama. Jalankan terpisah: python register_face.py

Nama yang SAMA -> foto baru ditambahkan ke folder yang sama (lanjut nomor).
Nama BEDA -> folder baru otomatis dibuat.

pip install insightface onnxruntime opencv-python numpy
"""

import os
import re
import sys
import threading
import time

import cv2

from insightface.app import FaceAnalysis


# ================= KONFIGURASI =================

CAMERA_URL = "http://192.168.1.11:8080/video"
DATASET_DIR = "face_dataset"
TARGET_IMAGES = 1          # jumlah foto BARU per sesi run (bukan total)
DET_SCORE_THRESHOLD = 0.55
CROP_MARGIN = 0.35         # padding di sekitar wajah saat crop
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540


# ================= FUNGSI BANTU =================

def sanitize_folder_name(name: str) -> str:
    """Bersihkan nama biar aman dipakai sebagai nama folder."""
    name = re.sub(r'[\\/:*?"<>|]', "", name.strip())
    return re.sub(r"\s+", " ", name)


def get_existing_count(folder_path: str) -> int:
    """Hitung jumlah foto .jpg yang sudah ada di folder (untuk lanjut nomor)."""
    if not os.path.exists(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])


def crop_face_with_margin(frame, bbox, margin_ratio):
    """Crop area wajah dari frame + margin, dibatasi ukuran frame."""
    h, w, _ = frame.shape
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin_ratio), int((y2 - y1) * margin_ratio)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    return frame[y1:y2, x1:x2]


class CameraStream:
    """
    Baca kamera di thread terpisah & selalu simpan frame TERBARU saja.
    Ini mencegah delay: tanpa ini, frame lama numpuk di buffer saat
    program sibuk memproses (mis. deteksi wajah), jadi kamera terasa
    ketinggalan dari kondisi aslinya.
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

print("Memuat model InsightFace...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))  # 320 = lebih cepat dari 640
print("Model siap.")


# ================= INPUT NAMA =================

person_name = sanitize_folder_name(input("Masukkan nama orang yang akan didaftarkan: "))
if person_name == "":
    print("Nama tidak boleh kosong. Program dihentikan.")
    sys.exit(1)

person_dir = os.path.join(DATASET_DIR, person_name)
os.makedirs(person_dir, exist_ok=True)

existing_count = get_existing_count(person_dir)
session_target = existing_count + TARGET_IMAGES  # target total setelah sesi ini
image_count = existing_count

if existing_count > 0:
    print(f"Folder '{person_name}' sudah ada {existing_count} foto. Menambahkan {TARGET_IMAGES} foto baru.")


# ================= SETUP KAMERA =================

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

window_title = f"Registrasi Wajah - {person_name}"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, WINDOW_WIDTH, WINDOW_HEIGHT)


# ================= LOOP UTAMA =================

print("Registrasi dimulai. Kamera tidak auto-close. Tekan 'q' untuk keluar kapan saja.")
target_reached = image_count >= session_target

while True:

    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    clean_frame = frame.copy()      # dipakai untuk disimpan (tanpa box)
    display_frame = frame.copy()    # dipakai untuk ditampilkan (dengan box)

    faces = face_app.get(frame)
    valid_faces = [f for f in faces if f.det_score >= DET_SCORE_THRESHOLD]

    session_progress = image_count - existing_count
    status_text = f"Sesi ini: {session_progress}/{TARGET_IMAGES} | Total: {image_count}"
    if target_reached:
        status_text += " (Selesai)"
    status_color = (0, 255, 0)

    if len(valid_faces) == 0:
        info_text = "Wajah tidak terdeteksi"
        status_color = (0, 0, 255)

    elif len(valid_faces) > 1:
        info_text = "Terdeteksi lebih dari 1 wajah! Pastikan hanya 1 orang."
        status_color = (0, 165, 255)
        for f in valid_faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

    else:
        face = valid_faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(display_frame, f"{face.det_score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if target_reached:
            # Sudah selesai -> box tetap tampil, tapi tidak disimpan lagi
            info_text = "Registrasi selesai. Foto tidak disimpan lagi."
        else:
            info_text = "Wajah terdeteksi, mengambil foto..."
            face_crop = crop_face_with_margin(clean_frame, (x1, y1, x2, y2), CROP_MARGIN)

            if face_crop.size > 0:
                image_count += 1
                filepath = os.path.join(person_dir, f"{image_count:03d}.jpg")
                cv2.imwrite(filepath, face_crop)

                session_progress = image_count - existing_count
                print(f"[Sesi {session_progress}/{TARGET_IMAGES} | Total {image_count}] Tersimpan -> {filepath}")
                status_text = f"Sesi ini: {session_progress}/{TARGET_IMAGES} | Total: {image_count}"

                if image_count >= session_target:
                    target_reached = True
                    print("Target sesi tercapai. Kamera tetap terbuka, tekan 'q' untuk menutup.")

    cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(display_frame, info_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    display_frame = cv2.resize(display_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    cv2.imshow(window_title, display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        reason = "setelah selesai" if target_reached else "sebelum target tercapai"
        print(f"Kamera ditutup manual oleh user ({reason}).")
        break


# ================= SELESAI =================

cap.release()
cv2.destroyAllWindows()

final_progress = image_count - existing_count
print("======================================")
print(f"Registrasi '{person_name}' {'SELESAI' if image_count >= session_target else 'berhenti di tengah jalan'}.")
print(f"Foto baru sesi ini : {final_progress}/{TARGET_IMAGES}")
print(f"Total foto di folder : {image_count}")
print(f"Lokasi folder : {os.path.abspath(person_dir)}")
print("======================================")