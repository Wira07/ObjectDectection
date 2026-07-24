from ultralytics import YOLO
import cv2
import os

# ================= KONFIGURASI =================

MODEL_SMOKE_PATH = "models/best.pt"        # model custom kamu -> Smoke/Fire
MODEL_PERSON_PATH = "models/yolo11n.pt"    # model bawaan (COCO) -> Person & Animal

ANIMAL_NAMES = ["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]

VIDEO_FOLDER = "gabungan_model_person_animal_smoke"    # taruh video yang mau ditest di sini
OUTPUT_FOLDER = "hasil_gabungan_model_person_animal_smoke"  # video hasil (dengan bounding box) disimpan di sini
CONF_SMOKE = 0.01                # dibuat seminim mungkin dulu, biar threshold pasti bukan penyebabnya
CONF_PERSON_ANIMAL = 0.4
IMG_SIZE = 640

# ================= LOAD MODEL =================

model_smoke = YOLO(MODEL_SMOKE_PATH)
model_person = YOLO(MODEL_PERSON_PATH)

# Buang class "other" dari model smoke -> nggak relevan, person/animal sudah ditangani model_person
smoke_class_ids = [i for i, name in model_smoke.names.items() if name.strip().lower() != "other"]
person_animal_ids = [i for i, name in model_person.names.items() if name in ANIMAL_NAMES]

print("Model Smoke/Fire   :", {i: n for i, n in model_smoke.names.items() if i in smoke_class_ids})
print("Model Person/Animal:", model_person.names)

# ================= FUNGSI GAMBAR BOX =================

def draw_boxes(frame, boxes, names, color_map, default_color=(255, 255, 255), thickness=2):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        name = names[int(b.cls)]
        color = color_map.get(name.strip().lower(), default_color)
        label = f"{name} {float(b.conf):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)


# Warna per class (format OpenCV = BGR, bukan RGB)
SMOKE_FIRE_COLORS = {
    "smoke": (255, 255, 255),  # putih
    "fire": (0, 255, 255),     # kuning
}
PERSON_ANIMAL_COLOR = (255, 0, 0)  # biru

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
    frames_with_smoke = 0
    frames_with_person_animal = 0
    best_conf_smoke = 0.0

    print(f"[{filename}] Memproses...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        results_smoke = model_smoke(frame, conf=CONF_SMOKE, imgsz=IMG_SIZE,
                                     classes=smoke_class_ids, verbose=False)
        results_person = model_person(frame, conf=CONF_PERSON_ANIMAL, imgsz=IMG_SIZE,
                                       classes=person_animal_ids, verbose=False)

        boxes_smoke = results_smoke[0].boxes
        boxes_person = results_person[0].boxes

        if len(boxes_smoke) > 0:
            frames_with_smoke += 1
            confs = [float(b.conf) for b in boxes_smoke]
            best_conf_smoke = max(best_conf_smoke, max(confs))
            if frame_index % 20 == 0:  # jangan print tiap frame, cukup sesekali
                print(f"  [frame {frame_index}] smoke/fire conf: {[round(c, 2) for c in confs]}")
        if len(boxes_person) > 0:
            frames_with_person_animal += 1

        annotated = frame.copy()
        draw_boxes(annotated, boxes_smoke, model_smoke.names, SMOKE_FIRE_COLORS, thickness=3)
        draw_boxes(annotated, boxes_person, model_person.names, {}, default_color=PERSON_ANIMAL_COLOR)
        writer.write(annotated)

        if frame_index % 100 == 0:
            print(f"  ...{frame_index} frame diproses")

    cap.release()
    writer.release()

    print(f"[{filename}] Selesai -> {frame_index} frame")
    print(f"  Smoke/Fire terdeteksi di {frames_with_smoke} frame (confidence tertinggi {best_conf_smoke:.2f})")
    print(f"  Person/Animal terdeteksi di {frames_with_person_animal} frame")
    print(f"  Hasil disimpan: {output_path}\n")

print(f"Semua video selesai. Cek folder '{OUTPUT_FOLDER}/' untuk hasilnya.")