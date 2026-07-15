import time

from config.config import DEFAULT_CAMERA_ID
from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.image_service import ImageService
from database.detection_repository import DetectionRepository


def main():
    camera_service = CameraService()
    detection_service = DetectionService()
    image_service = ImageService()

    # -----------------------------------------------------
    # INI KUNCI UTAMANYA:
    # saved_track_ids nyimpen track_id yang UDAH pernah disave.
    # Selama track_id-nya sama (artinya orang/objek yang sama,
    # terus-terusan kebaca kamera), dia gak akan disave lagi.
    #
    # last_seen dipakai buat "beres-beres" track_id yang udah lama
    # gak muncul lagi di kamera, biar dictionary-nya gak numpuk terus
    # selama program jalan lama.
    # -----------------------------------------------------
    saved_track_ids = set()
    last_seen = {}  # track_id -> timestamp terakhir kelihatan

    TRACK_FORGET_AFTER = 60  # detik, track_id lama dibuang dari memori

    print("======================================")
    print("Program jalan. Tekan 'q' di window kamera buat keluar.")
    print("======================================")

    while True:
        ret, frame = camera_service.read_frame()
        if not ret:
            print("Gagal ambil frame dari kamera, cek CAMERA_URL.")
            break

        detections = detection_service.detect(frame)
        now = time.time()

        for det in detections:
            track_id = det["track_id"]
            last_seen[track_id] = now

            # Gambar box tetap jalan tiap frame, siapapun yang kedeteksi
            DetectionService.draw_box(frame, det["box"], det["label"])

            # -------------------------------------------------
            # INI BAGIAN YANG NENTUIN "SAVE ATAU ENGGAK"
            # Kalau track_id ini BELUM pernah disave -> save sekarang,
            # lalu tandai udah disave. Kalau track_id-nya sama kayak
            # sebelumnya (orangnya masih itu-itu aja) -> skip, gak save.
            # -------------------------------------------------
            if track_id not in saved_track_ids:
                saved_track_ids.add(track_id)

                image_path = image_service.save_capture(frame, det["class_name"])

                DetectionRepository.save_detection(
                    camera_id=DEFAULT_CAMERA_ID,
                    object_type_id=det["object_type_id"],
                    confidence=det["confidence"],
                    image_path=image_path
                )

                print(f"[SAVED] track_id={track_id} | {det['label']} | {image_path}")

        # Beres-beres track_id yang udah lama gak kelihatan lagi,
        # biar kalau ID itu tiba-tiba muncul lagi (jarang, tapi bisa
        # kejadian tergantung tracker), memori gak numpuk sia-sia.
        expired = [
            tid for tid, ts in last_seen.items()
            if now - ts > TRACK_FORGET_AFTER
        ]
        for tid in expired:
            last_seen.pop(tid, None)
            saved_track_ids.discard(tid)

        camera_service.show(frame)

        if camera_service.is_quit_pressed():
            break

    camera_service.release()


if __name__ == "__main__":
    main()