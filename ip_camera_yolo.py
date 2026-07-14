from ultralytics import YOLO
import cv2

# Load model YOLO
model = YOLO("yolo11n.pt")

# URL Kamera HP
url = "http://192.168.1.14:8080/video"

cap = cv2.VideoCapture(url)

# Ukuran window
cv2.namedWindow("YOLO HP Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO HP Camera", 960, 540)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca kamera")
        break

    # =============================
    # DETEKSI YOLO
    # =============================
    results = model(frame)

    # Gambar bounding box
    annotated_frame = results[0].plot()

    # Resize agar tidak terlalu besar
    annotated_frame = cv2.resize(annotated_frame, (960,540))

    cv2.imshow("YOLO HP Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 