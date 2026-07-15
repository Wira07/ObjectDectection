import os
import cv2
from datetime import datetime
from config.config import CAPTURES_DIR


class ImageService:

    def __init__(self):
        os.makedirs(CAPTURES_DIR, exist_ok=True)

    def save_capture(self, frame, class_name):
        """
        Simpan frame asli (tanpa bounding box) ke folder captures/.
        Return path relative-nya buat disimpan ke database.
        """
        filename = f"{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(CAPTURES_DIR, filename)
        cv2.imwrite(image_path, frame)
        return image_path
