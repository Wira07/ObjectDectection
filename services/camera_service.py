import cv2
from config.config import CAMERA_URL, WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT


class CameraService:

    def __init__(self, url=CAMERA_URL):
        self.url = url
        self.cap = cv2.VideoCapture(url)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def show(self, frame):
        resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow(WINDOW_NAME, resized)

    def is_quit_pressed(self):
        return cv2.waitKey(1) & 0xFF == ord("q")

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
