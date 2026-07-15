from ultralytics import YOLO
import cv2
import mysql.connector
import os
from datetime import datetime
import time

# =====================================================
# KONEKSI DATABASE
# =====================================================

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="object_detection"
)

cursor = db.cursor()

# =====================================================
# LOAD MODEL
# =====================================================

# Folder project (folder tempat file Python ini berada)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lokasi model YOLO
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")

# Cek apakah model ada
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

print("======================================")
print("Loading YOLO Model...")
print("Model :", MODEL_PATH)
print("======================================")

# Load model
model = YOLO(MODEL_PATH)

print("======================================")
print("YOLO Model Berhasil Dimuat")
print("======================================")

# =====================================================
# URL KAMERA HP
# =====================================================

url = "http://192.168.1.19:8080/video"

cap = cv2.VideoCapture(url)

# =====================================================
# BUAT FOLDER SCREENSHOT
# =====================================================

os.makedirs("captures", exist_ok=True)

# =====================================================
# DAFTAR HEWAN (COCO)
# =====================================================

animals = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
]

# =====================================================
# AGAR DATABASE TIDAK SPAM
# =====================================================

last_save = {}

SAVE_INTERVAL = 5      # detik

# =====================================================
# WINDOW
# =====================================================

cv2.namedWindow("YOLO HP Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO HP Camera", 960, 540)

# =====================================================
# LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca kamera")
        break

    # Confidence minimal 70%
    results = model(frame, conf=0.70)

    annotated = frame.copy()

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[cls]

            object_type_id = None
            label = None

            # =====================================
            # PERSON
            # =====================================

            if class_name == "person":

                object_type_id = 1
                label = f"Person {conf:.2f}"

            # =====================================
            # ANIMAL
            # =====================================

            elif class_name in animals:

                object_type_id = 2
                label = f"Animal {conf:.2f}"

            else:
                continue

            # =====================================
            # KOORDINAT BOX
            # =====================================

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # =====================================
            # BOUNDING BOX
            # =====================================

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

            cv2.putText(
                annotated,
                label,
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3
            )

            # =====================================
            # SIMPAN KE DATABASE
            # =====================================

            current_time = time.time()

            if label not in last_save or current_time - last_save[label] > SAVE_INTERVAL:

                filename = f"{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

                image_path = os.path.join("captures", filename)

                cv2.imwrite(image_path, frame)

                sql = """
                INSERT INTO detections
                (
                    camera_id,
                    object_type_id,
                    confidence,
                    image_path,
                    detected_at
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """

                value = (
                    1,
                    object_type_id,
                    round(conf, 2),
                    image_path,
                    datetime.now()
                )

                cursor.execute(sql, value)
                db.commit()

                print("======================================")
                print("DATA TERSIMPAN")
                print("Object :", class_name)
                print("Confidence :", round(conf,2))
                print("Image :", image_path)
                print("======================================")

                last_save[label] = current_time

    annotated = cv2.resize(annotated, (960, 540))

    cv2.imshow("YOLO HP Camera", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================================================
# SELESAI
# =====================================================

cursor.close()
db.close()

cap.release()

cv2.destroyAllWindows()