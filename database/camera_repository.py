from database.database import Database


class CameraRepository:
    """
    Repository untuk data kamera.

    CATATAN:
    Script utama (app.py) saat ini masih pakai 1 kamera default
    (DEFAULT_CAMERA_ID di config.py) dan TIDAK butuh tabel `cameras`
    supaya program tetap bisa langsung jalan seperti sebelumnya.

    Method di bawah ini disiapkan untuk pengembangan ke depan
    (misal nanti mau multi-kamera dari database). Baru dipakai
    kalau kamu sudah buat tabel `cameras` di database.
    """

    @staticmethod
    def get_camera_by_id(camera_id):
        conn = Database.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM cameras WHERE id = %s", (camera_id,))
        result = cursor.fetchone()
        cursor.close()
        return result

    @staticmethod
    def get_all_cameras():
        conn = Database.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM cameras")
        result = cursor.fetchall()
        cursor.close()
        return result
