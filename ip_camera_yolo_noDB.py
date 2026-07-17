from ultralytics import YOLO
import cv2

# Load Model YOLO 
model = YOLO("yolo11n.pt")

# URL IP Webcam 
url = "http://192.168.1.13:8080/video" 
cap = cv2.VideoCapture(url)
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("YOLO Detection", 960, 540)

while True: 
    ret, frame = cap.read() 
    
    if not ret: 
        print("Gagal membaca kamera") 
        break 
    
    # Object Detection 
    results = model(frame) 
    
    # Gambar Bounding Box 
    annotated_frame = results[0].plot() 
    annotated_frame = cv2.resize( annotated_frame, (960, 540) ) 
    
    cv2.imshow("YOLO Detection", annotated_frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 

cap.release() 
cv2.destroyAllWindows()    