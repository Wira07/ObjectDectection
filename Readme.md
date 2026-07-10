# 🚀 Object Detection dengan YOLO11 (VS Code + Python)

Project pembelajaran **Object Detection** menggunakan **YOLO11 (Ultralytics)**, sebagai persiapan untuk membangun sistem AI yang mampu mendeteksi:

| Target | Status |
|---|---|
| 👤 Person | ✅ |
| 🐶 Animal | ✅ |
| 💨 Smoke (Custom Training) | 🔄 Belum |
| 🎥 CCTV (RTSP) | 🔄 Belum |
| 🌐 Web Dashboard | 🔄 Belum |

---

## 📋 Daftar Isi

1. [Requirement](#-requirement)
2. [Setup Project](#-setup-project)
3. [Struktur Folder](#-struktur-folder)
4. [Menjalankan Deteksi](#-menjalankan-deteksi)
5. [Perintah yang Sering Digunakan](#-perintah-yang-sering-digunakan)
6. [Troubleshooting](#-troubleshooting)
7. [Progress Belajar](#-progress-belajar)
8. [Tujuan Akhir](#-tujuan-akhir)

---

## 🧩 Requirement

- Windows 10 / 11
- Python 3.10+ (disarankan 3.10 – 3.12)
- Visual Studio Code
- Koneksi internet (untuk download model saat pertama kali dijalankan)

---

## ⚙️ Setup Project

### 1. Buat Folder Project

```
ObjectDetection/
```

Buka folder tersebut menggunakan VS Code.

### 2. Buat Virtual Environment

```bash
py -m venv .venv
```

atau

```bash
python -m venv .venv
```

Akan muncul folder baru:

```
ObjectDetection/
└── .venv/
```

### 3. Aktifkan Virtual Environment (Windows)

```bash
.venv\Scripts\activate
```

Jika berhasil, akan muncul `(.venv)` di depan terminal, contoh:

```bash
(.venv) C:\laragon\www\ObjectDetection>
```

### 4. Cek Versi Python

```bash
python --version
```

Contoh output:

```text
Python 3.14.0
```

### 5. Install YOLO (Ultralytics)

```bash
pip install ultralytics
```

Perintah ini otomatis menginstall dependency berikut:

- `ultralytics`
- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `matplotlib`
- `pillow`
- dan dependency lainnya

### 6. Cek Versi Torch

Masuk ke Python:

```bash
python
```

Lalu jalankan:

```python
import torch
print(torch.__version__)
```

Contoh output:

```text
2.13.0+cpu
```

Keluar dari Python:

```python
exit()
```

---

## 📁 Struktur Folder

```
ObjectDetection/
├── .venv/
├── images/     # gambar testing
├── videos/     # video testing
├── dataset/    # dataset training
└── models/     # model hasil training
```

---

## 🎯 Menjalankan Deteksi

### Detect Gambar

Misalnya ada file `images/bahlil.jpg`:

```bash
yolo task=detect mode=predict model=yolo11n.pt source=images/bahlil.jpg
```

Saat pertama kali dijalankan, YOLO otomatis mendownload `yolo11n.pt`.

Hasil deteksi otomatis tersimpan di:

```
runs/detect/predict/bahlil.jpg
```

### Detect Video

Misalnya ada file `videos/test.mp4`:

```bash
yolo task=detect mode=predict model=yolo11n.pt source=videos/test.mp4
```

Hasil tersimpan di `runs/detect/predict2/`, lalu `predict3/`, `predict4/`, dan seterusnya setiap kali dijalankan ulang.

### Detect Webcam

```bash
yolo task=detect mode=predict model=yolo11n.pt source=0
```

---

## 🔁 Perintah yang Sering Digunakan

| Fungsi | Perintah |
|---|---|
| Aktifkan virtual environment | `.venv\Scripts\activate` |
| Cek versi Python | `python --version` |
| Masuk ke Python | `python` |
| Keluar dari Python | `exit()` |
| Install YOLO | `pip install ultralytics` |
| Detect image | `yolo task=detect mode=predict model=yolo11n.pt source=images/bahlil.jpg` |
| Detect video | `yolo task=detect mode=predict model=yolo11n.pt source=videos/test.mp4` |
| Detect webcam | `yolo task=detect mode=predict model=yolo11n.pt source=0` |

---

## 🛠️ Troubleshooting

### `FileNotFoundError: images/image.jpg does not exist`

**Penyebab:** nama file yang dipanggil salah/tidak sesuai dengan file yang ada.

**Contoh kasus:**

File yang benar-benar ada:
```
images/bahlil.jpg
```

Tapi perintah yang dijalankan:
```bash
source=images/image.jpg
```

**Solusi:** gunakan nama file yang sesuai dengan yang ada di folder:
```bash
source=images/bahlil.jpg
```

### Stuck di prompt `>>>` (masuk mode Python)

Keluar dengan salah satu cara berikut:

```python
exit()
```

atau

```
Ctrl + Z lalu Enter
```

---

## ✅ Progress Belajar

- [x] Install Python
- [x] Virtual Environment
- [x] Install Ultralytics
- [x] Install Torch
- [x] Download Model YOLO11
- [x] Detect Image
- [x] Detect Video
- [ ] Detect Webcam
- [ ] Custom Dataset
- [ ] LabelImg / Label Studio
- [ ] Training Model Smoke
- [ ] Menghasilkan `best.pt`
- [ ] OpenCV
- [ ] CCTV (RTSP)
- [ ] Web Dashboard

---

## 🎯 Tujuan Akhir

```
CCTV
 │
 ▼
OpenCV
 │
 ▼
YOLO11 ──▶ Person / Animal / Smoke
 │
 ▼
Python
 │
 ▼
API
 │
 ▼
Web Dashboard
```