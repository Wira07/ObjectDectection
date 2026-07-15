from datetime import datetime
from database.database import Database


class DetectionRepository:

    @staticmethod
    def save_detection(camera_id, object_type_id, confidence, image_path):
        conn = Database.get_connection()
        cursor = conn.cursor()

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

        values = (
            camera_id,
            object_type_id,
            round(confidence, 2),
            image_path,
            datetime.now()
        )

        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
