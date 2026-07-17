"""
=========================================================
 REGISTER FACE - Modul Registrasi Dataset Wajah
=========================================================
Modul terpisah untuk membuat dataset wajah menggunakan
InsightFace + IP Webcam.

Modul ini TIDAK berhubungan dan TIDAK mengubah program
utama YOLO Object Detection (app.py / ip_camera_yolo.py).

Fungsi:
    1. User memasukkan nama orang yang akan didaftarkan.
    2. Kamera IP Webcam dibuka.
    3. Wajah dideteksi menggunakan InsightFace.
    4. Otomatis mengambil +/- 30 foto wajah (crop + margin).
    5. Foto disimpan ke folder:
         face_dataset/<Nama Orang>/001.jpg ... 030.jpg
    6. Program berhenti otomatis saat 30 foto tercapai.

Instalasi library yang dibutuhkan (jika belum ada):
    pip install insightface onnxruntime opencv-python numpy

Catatan:
    - Model InsightFace ("buffalo_l") akan otomatis
      terunduh ke ~/.insightface/models saat pertama kali
      dijalankan (butuh koneksi internet sekali saja).
    - Jalankan file ini terpisah dari program YOLO:
          python register_face.py
=========================================================
"""

import os
import re
import sys
import time

import cv2
import numpy as np

from insightface.app import FaceAnalysis


# =====================================================
# KONFIGURASI
# =====================================================

# Alamat kamera IP Webcam (samakan dengan project YOLO)
CAMERA_URL = "http://192.168.1.9:8080/video"

# Folder utama tempat dataset wajah disimpan
DATASET_DIR = "face_dataset"

# Jumlah foto target per orang
TARGET_IMAGES = 30

# Jarak minimum antar pengambilan foto (detik)
# Supaya foto tidak diambil terlalu rapat / mirip semua
CAPTURE_INTERVAL = 0.35

# Minimal skor deteksi wajah agar dianggap valid
DET_SCORE_THRESHOLD = 0.55

# Margin tambahan di sekitar bounding box wajah saat crop
# (dalam persen dari ukuran box), agar hasil crop tidak
# terlalu ketat / mepet ke wajah
CROP_MARGIN = 0.35

# Ukuran window preview
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540


# =====================================================
# FUNGSI BANTU
# =====================================================

def sanitize_folder_name(name: str) -> str:
    """
    Membersihkan input nama agar aman dipakai sebagai
    nama folder (menghapus karakter terlarang di Windows/Linux).
    """
    name = name.strip()
    # Hapus karakter yang tidak boleh ada di nama folder
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    # Rapikan spasi berlebih
    name = re.sub(r"\s+", " ", name)
    return name


def get_existing_count(folder_path: str) -> int:
    """
    Menghitung jumlah foto (.jpg) yang sudah ada di folder,
    agar proses bisa dilanjutkan (resume) jika sebelumnya
    sempat berhenti di tengah jalan.
    """
    if not os.path.exists(folder_path):
        return 0

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    return len(files)


def crop_face_with_margin(frame, bbox, margin_ratio):
    """
    Melakukan crop wajah dari frame berdasarkan bounding box,
    ditambah margin di sekeliling wajah supaya hasil crop
    tidak terlalu mepet.
    """
    h, w, _ = frame.shape

    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    margin_x = int(box_w * margin_ratio)
    margin_y = int(box_h * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    return frame[y1:y2, x1:x2]


# =====================================================
# INISIALISASI MODEL INSIGHTFACE
# =====================================================

print("======================================")
print("Memuat Model InsightFace...")
print("======================================")

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

print("======================================")
print("Model InsightFace Berhasil Dimuat")
print("======================================")


# =====================================================
# INPUT NAMA USER
# =====================================================

raw_name = input("Masukkan nama orang yang akan didaftarkan: ")
person_name = sanitize_folder_name(raw_name)

if person_name == "":
    print("Nama tidak boleh kosong. Program dihentikan.")
    sys.exit(1)

# Folder khusus untuk orang ini
person_dir = os.path.join(DATASET_DIR, person_name)
os.makedirs(person_dir, exist_ok=True)

# Cek apakah sudah ada foto sebelumnya (untuk resume)
existing_count = get_existing_count(person_dir)

if existing_count >= TARGET_IMAGES:
    print("======================================")
    print(f"Dataset untuk '{person_name}' sudah lengkap ({existing_count} foto).")
    print("Hapus folder tersebut atau gunakan nama lain jika ingin mengulang.")
    print("======================================")
    sys.exit(0)

if existing_count > 0:
    print("======================================")
    print(f"Melanjutkan pengambilan foto untuk '{person_name}'.")
    print(f"Sudah ada {existing_count} foto, target {TARGET_IMAGES} foto.")
    print("======================================")

image_count = existing_count


# =====================================================
# SETUP KAMERA
# =====================================================

print("======================================")
print("Membuka Kamera IP Webcam...")
print("URL   :", CAMERA_URL)
print("======================================")

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("Gagal membuka kamera. Pastikan URL IP Webcam benar.")
    sys.exit(1)

window_title = f"Registrasi Wajah - {person_name}"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, WINDOW_WIDTH, WINDOW_HEIGHT)


# =====================================================
# LOOP CAPTURE WAJAH
# =====================================================

last_capture_time = 0

print("======================================")
print("Registrasi dimulai. Tekan 'q' untuk berhenti kapan saja.")
print("======================================")

while image_count < TARGET_IMAGES:

    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Frame bersih untuk disimpan (tanpa bounding box)
    clean_frame = frame.copy()

    # Frame untuk ditampilkan (dengan bounding box + info)
    display_frame = frame.copy()

    # Deteksi wajah menggunakan InsightFace
    faces = face_app.get(frame)

    # Filter hanya wajah dengan skor deteksi yang cukup tinggi
    valid_faces = [f for f in faces if f.det_score >= DET_SCORE_THRESHOLD]

    status_text = f"Foto: {image_count}/{TARGET_IMAGES}"
    status_color = (0, 255, 0)

    if len(valid_faces) == 0:
        info_text = "Wajah tidak terdeteksi"
        status_color = (0, 0, 255)

    elif len(valid_faces) > 1:
        info_text = "Terdeteksi lebih dari 1 wajah! Pastikan hanya 1 orang."
        status_color = (0, 165, 255)

        # Tetap gambar semua box yang terdeteksi sebagai peringatan visual
        for f in valid_faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

    else:
        # Tepat 1 wajah valid -> boleh diambil fotonya
        face = valid_faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)

        # Gambar bounding box pada frame tampilan
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            display_frame,
            f"{face.det_score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        info_text = "Wajah terdeteksi, mengambil foto..."

        current_time = time.time()

        # Ambil foto hanya jika sudah melewati interval minimum
        if current_time - last_capture_time >= CAPTURE_INTERVAL:

            face_crop = crop_face_with_margin(clean_frame, (x1, y1, x2, y2), CROP_MARGIN)

            # Lewati jika hasil crop tidak valid (terlalu di pinggir frame)
            if face_crop.size > 0:

                image_count += 1
                filename = f"{image_count:03d}.jpg"
                filepath = os.path.join(person_dir, filename)

                cv2.imwrite(filepath, face_crop)

                last_capture_time = current_time

                print(f"[{image_count}/{TARGET_IMAGES}] Tersimpan -> {filepath}")

                status_text = f"Foto: {image_count}/{TARGET_IMAGES}"

    # =====================================================
    # TAMPILKAN INFORMASI DI LAYAR
    # =====================================================

    cv2.putText(
        display_frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        display_frame,
        info_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )

    display_frame = cv2.resize(display_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    cv2.imshow(window_title, display_frame)

    # Tekan 'q' untuk berhenti lebih awal
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Dihentikan oleh user sebelum mencapai target.")
        break


# =====================================================
# SELESAI
# =====================================================

cap.release()
cv2.destroyAllWindows()

print("======================================")
if image_count >= TARGET_IMAGES:
    print(f"Registrasi wajah '{person_name}' SELESAI.")
else:
    print(f"Registrasi wajah '{person_name}' berhenti di tengah jalan.")
print(f"Total foto tersimpan : {image_count}/{TARGET_IMAGES}")
print(f"Lokasi folder        : {os.path.abspath(person_dir)}")
print("======================================")