import os
import cv2
from ultralytics import YOLO
from config.config import MODEL_PATH, CONFIDENCE_THRESHOLD, ANIMALS


class DetectionService:

    def __init__(self, model_path=MODEL_PATH, confidence=CONFIDENCE_THRESHOLD):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

        print("======================================")
        print("Loading YOLO Model...")
        print("Model :", model_path)
        print("======================================")

        self.model = YOLO(model_path)
        self.confidence = confidence

        print("======================================")
        print("YOLO Model Berhasil Dimuat")
        print("======================================")

    def detect(self, frame):
        """
        Jalankan deteksi + TRACKING pada 1 frame.

        Bedanya sama sebelumnya (self.model(frame)):
        - Sekarang pakai self.model.track(frame, persist=True)
        - Tiap objek yang kedeteksi dikasih "track_id" yang KONSISTEN
          selama objek itu masih terus kebaca kamera.
        - Ini yang dipakai di app.py buat nentuin "orang ini udah
          pernah disave belum" tanpa harus interval waktu.

        Return list of dict:
        [{class_name, object_type_id, label, confidence, box, track_id}, ...]
        """
        results = self.model.track(
            frame,
            conf=self.confidence,
            persist=True,
            verbose=False
        )

        detections = []

        for result in results:
            # Kalau belum ada track ID sama sekali di frame ini, skip.
            # Ini bisa kejadian di frame pertama sebelum tracker "warm up".
            if result.boxes.id is None:
                continue

            for box, track_id in zip(result.boxes, result.boxes.id):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]

                object_type_id, label = self._classify(class_name, conf)

                if object_type_id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class_name": class_name,
                    "object_type_id": object_type_id,
                    "label": label,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "track_id": int(track_id)
                })

        return detections

    def _classify(self, class_name, conf):
        if class_name == "person":
            return 1, f"Person {conf:.2f}"
        elif class_name in ANIMALS:
            return 2, f"Animal {conf:.2f}"
        return None, None

    @staticmethod
    def draw_box(frame, box, label):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )