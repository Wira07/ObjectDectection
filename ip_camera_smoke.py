from ultralytics import YOLO
import cv2
import threading
import time

# ================= KONFIGURASI =================

MODEL_PATH = "models/best.pt"   # sesuaikan kalau lokasi filenya beda
CAMERA_URL = "http://192.168.1.16:8080/video"
CONF_THRESHOLD = 0.25             # sementara diturunkan buat debugging, naikkan lagi setelah ketemu penyebabnya
IMG_SIZE = 640                     # samakan dengan resolusi training kalau tahu; 640 = default Ultralytics, lebih aman utk tes awal
DEBUG = True                       # cetak semua deteksi mentah (termasuk yang di bawah threshold) ke terminal
SKIP_FRAMES = 2                   # proses 1 dari N frame (CameraStream sudah menghilangkan delay, ini cuma hemat CPU)
WINDOW_WIDTH, WINDOW_HEIGHT = 960, 540


# ================= CAMERA (thread, anti-delay) =================

class CameraStream:
    """Baca kamera di thread terpisah & selalu simpan frame TERBARU saja.
    Mencegah delay akibat buffer numpuk saat model sedang sibuk inferensi."""

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


# ================= LOAD MODEL =================

print("Memuat model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("Model siap. Class yang dikenali:", model.names)

# ================= SETUP KAMERA =================

print("Membuka kamera:", CAMERA_URL)
cap = CameraStream(CAMERA_URL).start()

if not cap.isOpened():
    print("Gagal membuka kamera. Cek URL IP Webcam.")
    exit(1)

# Tunggu frame pertama siap dari thread (maks ~5 detik)
for _ in range(50):
    ret, _ = cap.read()
    if ret:
        break
    time.sleep(0.1)
else:
    print("Timeout: tidak ada frame dari kamera.")
    cap.release()
    exit(1)

cv2.namedWindow("Smoke Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smoke Detection", WINDOW_WIDTH, WINDOW_HEIGHT)

# ================= LOOP UTAMA =================

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca kamera.")
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # DEBUG: jalankan dengan conf sangat rendah supaya semua kandidat box ikut kebaca,
    # lalu kita lihat sendiri skornya di terminal sebelum diputuskan mana yang ditampilkan.
    raw_results = model(frame, conf=0.05, imgsz=IMG_SIZE, verbose=False)
    boxes = raw_results[0].boxes

    if DEBUG:
        if len(boxes) == 0:
            print("Tidak ada kandidat box sama sekali di frame ini.")
        else:
            scores = [f"{model.names[int(b.cls)]}:{float(b.conf):.2f}" for b in boxes]
            print("Kandidat terdeteksi ->", ", ".join(scores))

    # Yang benar-benar ditampilkan cuma yang lolos CONF_THRESHOLD asli
    results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
    annotated = results[0].plot()
    annotated = cv2.resize(annotated, (WINDOW_WIDTH, WINDOW_HEIGHT))
    cv2.imshow("Smoke Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()