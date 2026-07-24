from ultralytics import YOLO
import cv2
import os

# ================= KONFIGURASI =================

MODEL_PATH = "models/best.pt"


VIDEO_FOLDER = "test_videos"    # taruh video yang mau ditest di sini
OUTPUT_FOLDER = "test_results"  # video hasil (dengan bounding box) disimpan di sini
CONF_THRESHOLD = 0.05           # rendah dulu, biar semua kandidat kelihatan
IMG_SIZE = 640

# ================= LOAD MODEL =================

model = YOLO(MODEL_PATH)
print("Class yang dikenali model:", model.names)

# ================= BACA FOLDER VIDEO =================

valid_ext = (".mp4", ".avi", ".mov", ".mkv")

if not os.path.isdir(VIDEO_FOLDER):
    print(f"Folder '{VIDEO_FOLDER}' tidak ditemukan. Buat folder itu dan isi dengan video tes.")
    exit(1)

video_files = [f for f in sorted(os.listdir(VIDEO_FOLDER)) if f.lower().endswith(valid_ext)]

if not video_files:
    print(f"Tidak ada video (.mp4/.avi/.mov/.mkv) di folder '{VIDEO_FOLDER}'.")
    exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Ditemukan {len(video_files)} video. Memproses...\n")

# ================= PROSES SATU-PER-SATU =================

for filename in video_files:
    path = os.path.join(VIDEO_FOLDER, filename)
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f"[{filename}] Gagal dibuka, dilewati.")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(OUTPUT_FOLDER, filename)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_index = 0
    frames_with_detection = 0
    best_conf = 0.0

    print(f"[{filename}] Memproses...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            frames_with_detection += 1
            best_conf = max(best_conf, max(float(b.conf) for b in boxes))

        writer.write(results[0].plot())

        if frame_index % 100 == 0:
            print(f"  ...{frame_index} frame diproses")

    cap.release()
    writer.release()

    print(f"[{filename}] Selesai -> {frame_index} frame, {frames_with_detection} frame ada deteksi, confidence tertinggi {best_conf:.2f}")
    print(f"  Hasil disimpan: {output_path}\n")

print(f"Semua video selesai. Cek folder '{OUTPUT_FOLDER}/' untuk hasilnya.")
