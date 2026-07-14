import cv2

# URL kamera HP
url = "http://192.168.1.14:8080/video"

cap = cv2.VideoCapture(url)

# Membuat window yang bisa di-resize
cv2.namedWindow("HP Camera", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("HP Camera", 960, 540)
cv2.resizeWindow("HP Camera", 800, 450)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca kamera")
        break

    # Resize frame agar tampilannya tidak terlalu besar
    frame = cv2.resize(frame, (960, 540))

    cv2.imshow("HP Camera", frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()