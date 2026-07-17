import os

# =====================================================
# BASE DIR (folder root project)
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# DATABASE
# =====================================================

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "object_detection"
}

# =====================================================
# MODEL YOLO
# =====================================================

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")

CONFIDENCE_THRESHOLD = 0.70

# =====================================================
# KAMERA
# =====================================================

CAMERA_URL = "http://192.168.1.11:8080/video"

DEFAULT_CAMERA_ID = 1

WINDOW_NAME = "YOLO HP Camera"
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540

# =====================================================
# FOLDER CAPTURES
# =====================================================

CAPTURES_DIR = os.path.join(BASE_DIR, "captures")

# =====================================================
# DAFTAR HEWAN (COCO)
# =====================================================

ANIMALS = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
]

# =====================================================
# ANTI SPAM SAVE KE DATABASE
# =====================================================
# CATATAN: dengan tracking (track_id), variabel ini SUDAH TIDAK
# dipakai untuk logic "1 orang = 1x save". Sekarang cuma dipakai
# sebagai jaga-jaga tambahan kalau nanti mau nge-refresh save
# buat orang yang sama tapi udah lama banget di frame.
SAVE_INTERVAL = 5  # detik

# Berapa lama (detik) sebuah track_id dianggap "hilang" dan boleh
# dianggap objek baru kalau muncul lagi. Ini dipakai supaya kalau
# orang cuma kelewat kehalang 1-2 frame doang, gak langsung dianggap
# orang baru.
TRACK_LOST_TIMEOUT = 5  # detik