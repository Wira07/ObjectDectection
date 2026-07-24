from ultralytics import YOLO
import os
import sys

MODEL_PATH = "models/best.pt"

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_images/smoke_2.png"

# ==========================
# Cek model
# ==========================

if not os.path.exists(MODEL_PATH):
    print(f"Model tidak ditemukan: {MODEL_PATH}")
    exit()

# ==========================
# Cek gambar
# ==========================

if not os.path.exists(IMAGE_PATH):
    print(f"Gambar tidak ditemukan: {IMAGE_PATH}")
    print("\nContoh penggunaan:")
    print("python test_smoke_model.py gambar.jpg")
    exit()

# ==========================
# Load model
# ==========================

model = YOLO(MODEL_PATH)

print("Class yang dikenali model:")
print(model.names)

# ==========================
# Predict
# ==========================

results = model.predict(
    source=IMAGE_PATH,
    conf=0.05,
    imgsz=1280,
    verbose=True
)

boxes = results[0].boxes

print("\n============================")
print("Jumlah deteksi:", len(boxes))
print("============================")

for box in boxes:
    cls = int(box.cls)
    conf = float(box.conf)

    print(
        f"{model.names[cls]} | confidence = {conf:.3f}"
    )

# ==========================
# Simpan hasil
# ==========================

output = "hasil_test.jpg"

results[0].save(output)

print(f"\nHasil disimpan ke {output}")