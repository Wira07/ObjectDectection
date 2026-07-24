from ultralytics import YOLO
import os

# ================= KONFIGURASI =================

MODEL_PATH = "models/best.pt"
IMAGE_FOLDER = "test_images"    # taruh semua foto yang mau ditest di sini
OUTPUT_FOLDER = "test_results"  # hasil (dengan bounding box) disimpan di sini
CONF_THRESHOLD = 0.05           # rendah dulu, biar semua kandidat kelihatan
IMG_SIZE = 640

# ================= LOAD MODEL =================

model = YOLO(MODEL_PATH)
print("Class yang dikenali model:", model.names)

# ================= BACA FOLDER GAMBAR =================

valid_ext = (".jpg", ".jpeg", ".png")

if not os.path.isdir(IMAGE_FOLDER):
    print(f"Folder '{IMAGE_FOLDER}' tidak ditemukan. Buat folder itu dan isi dengan foto tes.")
    exit(1)

image_files = [f for f in sorted(os.listdir(IMAGE_FOLDER)) if f.lower().endswith(valid_ext)]

if not image_files:
    print(f"Tidak ada gambar (.jpg/.jpeg/.png) di folder '{IMAGE_FOLDER}'.")
    exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Ditemukan {len(image_files)} gambar. Memproses...\n")

# ================= PROSES SATU-PER-SATU =================

for filename in image_files:
    path = os.path.join(IMAGE_FOLDER, filename)
    results = model(path, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print(f"[{filename}] Tidak ada kandidat terdeteksi.")
    else:
        scores = [f"{model.names[int(b.cls)]}:{float(b.conf):.2f}" for b in boxes]
        print(f"[{filename}] Terdeteksi -> {', '.join(scores)}")

    results[0].save(os.path.join(OUTPUT_FOLDER, filename))

print(f"\nSelesai. Hasil (dengan bounding box kalau ada) disimpan di folder '{OUTPUT_FOLDER}/'.")