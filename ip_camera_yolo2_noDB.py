from ultralytics import YOLO
import cv2

# 1. Load Model YOLO (Menggunakan versi Nano yang paling ringan)
model = YOLO("yolo11n.pt")

# 2. Setup URL IP Webcam 
url = "http://192.168.1.9:8080/video" 
cap = cv2.VideoCapture(url)

# Setup Jendela Tampilan
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("YOLO Detection", 960, 540)

# Variabel untuk Frame Skipping
frame_count = 0
skip_frames = 3 # Hanya proses 1 dari setiap 3 frame yang masuk

while True: 
    ret, frame = cap.read() 
    
    if not ret: 
        print("Gagal membaca kamera. Cek koneksi IP Webcam Anda.") 
        break 
    
    frame_count += 1
    
    # 3. Optimasi: Lompat frame untuk menghemat CPU
    if frame_count % skip_frames == 0:
        
        # 4. Object Detection (Sudah Dioptimasi)
        # - classes=[0] : Hanya mencari class ke-0 (Person)
        # - verbose=False : Mematikan log teks di console
        # - imgsz=320 : Menurunkan resolusi perhitungan AI menjadi 320x320
        results = model(frame, classes=[0], verbose=False, imgsz=320) 
        
        # 5. Gambar Bounding Box hasil deteksi
        annotated_frame = results[0].plot() 
        
        # 6. Tampilkan ke layar (Resize hanya untuk visual GUI, tidak membebani model)
        annotated_frame = cv2.resize(annotated_frame, (960, 540)) 
        cv2.imshow("YOLO Detection", annotated_frame) 
    
    # Keluar jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 

# Bersihkan resource saat selesai
cap.release() 
cv2.destroyAllWindows()