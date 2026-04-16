# AI Real-Time Face Swap (Multi-backend)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Auto_backend-blue)
![Status](https://img.shields.io/badge/Status-Experimental-warning)

Language:
- [English Documentation](#english-documentation)
- [Tai lieu Tieng Viet](#tai-lieu-tieng-viet)

---

## English Documentation

### 1. Overview

This project is a real-time face swap and whole-head synthesis toolkit built with Python, InsightFace, OpenCV, and ONNX Runtime.

It supports multiple acceleration backends:
- CUDA (Linux/NVIDIA)
- DirectML (Windows/AMD)
- CPU fallback

It includes a hardware-aware model setup script that automatically selects a model package profile based on your machine capability.

### 2. Main Features

- Real-time face swap from webcam feed
- Whole-head synthesis mode with LivePortrait models
- Video-to-dataset extraction pipeline
- Dataset-to-embedding pipeline
- Auto backend selection for inference providers
- Auto model setup based on machine tier (weak, balanced, strong)
- User-friendly launcher UI with language selection (English and Vietnamese)
- CI workflows to auto-build `.exe`, `.deb`, and `.dmg` packages

### 2.1 Bilingual User Interface

The launcher now prompts for interface language at startup:
- English
- Vietnamese

The selected language is propagated to child scripts through `DEEPFAKE_LANG`, so model setup and launcher messages stay consistent.

### 3. Model Auto Setup (New)

Run:

```bash
python 06_setup_models.py
```

Or from launcher menu:

```bash
python app.py
# Choose option 7 [MODEL]
```

What it does:
- Detects CPU cores, RAM, free disk space, NVIDIA availability, NVIDIA VRAM, and available ONNX providers
- Classifies your machine as weak / balanced / strong
- Prints recommendation reasons and machine score
- Supports manual tier override before downloading
- Writes full model metadata to `models/model_catalog.json`
- Downloads available models for the detected tier
- Skips files that already exist

Machine tier policy:

| Tier | Detection Rule | Auto-selected Models |
| :--- | :--- | :--- |
| weak | Low RAM/CPU | `inswapper_128.onnx` |
| balanced | Mid-range machine | `inswapper_128.onnx` + LivePortrait ONNX set |
| strong | Strong CPU/RAM + NVIDIA | `inswapper_128.onnx` + LivePortrait ONNX set + `GPEN-BFR-512.onnx` |

LivePortrait download source:
- By default, only known public model URLs are downloaded automatically.
- For LivePortrait auto-download, set:

```bash
export DEEPFAKE_LIVEPORTRAIT_BASE_URL="https://your-host/path/to/liveportrait"
```

Expected file names at that URL:
- `appearance_feature_extractor.onnx`
- `motion_extractor.onnx`
- `stitching_retargeting.onnx`
- `warping_spatially_adaptive_network.onnx`

### 4. Detailed Model Catalog (Purpose and Usage)

| Model File | Required | Purpose | Used By |
| :--- | :---: | :--- | :--- |
| `models/inswapper_128.onnx` | Yes | Core identity swap model for face replacement | `03_run_webcam.py`, `05_run_video_file.py` |
| `models/GPEN-BFR-512.onnx` | Optional | Face restoration/enhancement for sharper details | Enhancement workflows (quality over FPS) |
| `models/liveportrait/appearance_feature_extractor.onnx` | Whole-head mode | Extracts source appearance tensor | `07_head_stitcher.py` |
| `models/liveportrait/motion_extractor.onnx` | Whole-head mode | Extracts driving motion/keypoints from incoming frames | `07_head_stitcher.py` |
| `models/liveportrait/stitching_retargeting.onnx` | Whole-head mode | Retargets and stitches keypoint motions | `07_head_stitcher.py` |
| `models/liveportrait/warping_spatially_adaptive_network.onnx` | Whole-head mode | Final renderer for full-head output | `07_head_stitcher.py` |

### 5. Installation

```bash
git clone https://github.com/manhtuan28/DeepFake-RealTime.git
cd DeepFake-RealTime
pip install -r requirements.txt
```

GPU runtime package (choose one):

```bash
# Linux / NVIDIA
pip install onnxruntime-gpu

# Windows / AMD
pip install onnxruntime-directml
```

Recommended compatibility pin:

```bash
pip install "numpy<2.0"
```

### 6. Usage Workflow

Run launcher:

```bash
python app.py
```

Menu options:
- 1: Build dataset from raw images
- 2: Generate embeddings
- 3: Real-time webcam face swap
- 4: Extract faces from videos into dataset
- 5: Render swap into a video file
- 6: Whole-head LivePortrait mode
- 7: Auto model setup by machine tier

### 7. Performance Tips

For low FPS in whole-head mode:
- Increase `DEEPFAKE_HEAD_FRAME_SKIP` (higher = faster, less smooth)
- Reduce `DEEPFAKE_HEAD_DRIVING_SIZE`
- Keep enhancer model optional for real-time workflows

Force CPU-only mode:

```bash
export DEEPFAKE_DISABLE_GPU=1
```

Force custom provider order:

```bash
export DEEPFAKE_ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
```

### 8. Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── build-distributions.yml
├── app.py
├── 01_data_processing.py
├── 02_create_embeddings.py
├── 03_run_webcam.py
├── 04_video_to_dataset.py
├── 05_run_video_file.py
├── 06_setup_models.py
├── 07_head_stitcher.py
├── runtime_utils.py
├── models/
│   ├── inswapper_128.onnx
│   ├── GPEN-BFR-512.onnx
│   └── liveportrait/
│       ├── appearance_feature_extractor.onnx
│       ├── motion_extractor.onnx
│       ├── stitching_retargeting.onnx
│       └── warping_spatially_adaptive_network.onnx
├── raw_data/
├── dataset/
├── embeddings/
├── video_data/
└── README.md
```

### 9. Automated Distribution Build Workflows

GitHub Actions workflow file:
- `.github/workflows/build-distributions.yml`

It automatically builds artifacts for:
- Windows: `DeepFakeRT.exe` (zipped)
- Linux: `deepfake-rt_0.1.0_amd64.deb`
- macOS: `DeepFakeRT-macos.dmg`

Trigger methods:
- Push to `main`
- Manual trigger via `workflow_dispatch`

### 10. Responsible Use

This project is for education and research only. Do not use it for impersonation, fraud, harassment, or any illegal activity. Always obtain consent for face data usage.

---

## Tai lieu Tieng Viet

### 1. Tong quan

Day la bo cong cu fake mat theo thoi gian thuc va fake toan bo dau, duoc xay dung bang Python, InsightFace, OpenCV va ONNX Runtime.

Ho tro nhieu backend tang toc:
- CUDA (Linux/NVIDIA)
- DirectML (Windows/AMD)
- CPU (du phong)

Du an da bo sung script tu dong cai model dua tren cau hinh may (yeu, trung binh, manh).

### 2. Tinh nang chinh

- Fake mat realtime tu webcam
- Che do fake toan bo dau bang LivePortrait
- Trich xuat du lieu khuon mat tu video
- Tao embedding tu dataset
- Tu dong chon backend suy luan
- Tu dong cai model theo do manh yeu cua may
- Launcher than thien, cho phep chon ngon ngu Anh/Viet
- Workflow CI tu dong build goi `.exe`, `.deb`, `.dmg`

### 2.1 Giao dien song ngu

Khi khoi dong launcher, ban co the chon ngon ngu:
- English
- Tieng Viet

Lua chon nay duoc truyen cho script con qua bien moi truong `DEEPFAKE_LANG` de giao dien dong nhat.

### 3. Tu dong cai model theo cau hinh may (Moi)

Chay truc tiep:

```bash
python 06_setup_models.py
```

Hoac tu menu:

```bash
python app.py
# Chon muc 7 [MODEL]
```

Script se:
- Doc so nhan CPU, dung luong RAM, dung luong o dia con trong, kiem tra NVIDIA GPU, VRAM va ONNX providers
- Phan loai may thanh weak / balanced / strong
- In ly do de xuat va diem danh gia cau hinh
- Cho phep ban ghi de goi model truoc khi tai
- Ghi day du thong tin model vao `models/model_catalog.json`
- Tu dong tai nhung model phu hop voi cau hinh
- Bo qua model da co san

Chinh sach chon goi model:

| Cap may | Quy tac phan loai | Model se cai |
| :--- | :--- | :--- |
| weak | RAM/CPU thap | `inswapper_128.onnx` |
| balanced | Cau hinh trung binh | `inswapper_128.onnx` + bo LivePortrait |
| strong | CPU/RAM cao + NVIDIA | `inswapper_128.onnx` + bo LivePortrait + `GPEN-BFR-512.onnx` |

Nguon tai model LivePortrait:
- Mac dinh chi auto tai duoc cac model co link cong khai san.
- Neu ban co host rieng, dat bien:

```bash
export DEEPFAKE_LIVEPORTRAIT_BASE_URL="https://your-host/path/to/liveportrait"
```

Ten file script se tim:
- `appearance_feature_extractor.onnx`
- `motion_extractor.onnx`
- `stitching_retargeting.onnx`
- `warping_spatially_adaptive_network.onnx`

### 4. Bang model chi tiet (Cong dung va muc dich)

| File model | Bat buoc | Cong dung | Script su dung |
| :--- | :---: | :--- | :--- |
| `models/inswapper_128.onnx` | Co | Model chinh de doi danh tinh khuon mat | `03_run_webcam.py`, `05_run_video_file.py` |
| `models/GPEN-BFR-512.onnx` | Tuy chon | Nang chat luong/lam net khuon mat | Che do uu tien chat luong hon FPS |
| `models/liveportrait/appearance_feature_extractor.onnx` | Che do dau | Trich xuat dac trung hinh dang goc | `07_head_stitcher.py` |
| `models/liveportrait/motion_extractor.onnx` | Che do dau | Trich xuat chuyen dong va keypoint | `07_head_stitcher.py` |
| `models/liveportrait/stitching_retargeting.onnx` | Che do dau | Ghep va retarget keypoint | `07_head_stitcher.py` |
| `models/liveportrait/warping_spatially_adaptive_network.onnx` | Che do dau | Render khung hinh dau cuoi cung | `07_head_stitcher.py` |

### 5. Cai dat

```bash
git clone https://github.com/manhtuan28/DeepFake-RealTime.git
cd DeepFake-RealTime
pip install -r requirements.txt
```

Chon runtime GPU phu hop (chi chon 1):

```bash
# Linux / NVIDIA
pip install onnxruntime-gpu

# Windows / AMD
pip install onnxruntime-directml
```

Khuyen nghi de tranh loi tuong thich:

```bash
pip install "numpy<2.0"
```

### 6. Quy trinh su dung

Chay launcher:

```bash
python app.py
```

Y nghia menu:
- 1: Xu ly du lieu anh tho thanh dataset
- 2: Tao embedding
- 3: Fake mat realtime webcam
- 4: Cat du lieu mat tu video
- 5: Render fake vao file video
- 6: Fake toan bo dau bang LivePortrait
- 7: Tu dong cai model theo cau hinh may

### 7. Toi uu hieu nang

Neu FPS thap trong che do fake dau:
- Tang `DEEPFAKE_HEAD_FRAME_SKIP` (cao hon thi nhanh hon, giam do muot)
- Giam `DEEPFAKE_HEAD_DRIVING_SIZE`
- Khong bat enhancer neu uu tien toc do

Bat che do CPU-only:

```bash
export DEEPFAKE_DISABLE_GPU=1
```

Ep thu tu provider:

```bash
export DEEPFAKE_ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
```

### 8. Cau truc du an

```text
.
├── .github/
│   └── workflows/
│       └── build-distributions.yml
├── app.py
├── 01_data_processing.py
├── 02_create_embeddings.py
├── 03_run_webcam.py
├── 04_video_to_dataset.py
├── 05_run_video_file.py
├── 06_setup_models.py
├── 07_head_stitcher.py
├── runtime_utils.py
├── models/
│   ├── inswapper_128.onnx
│   ├── GPEN-BFR-512.onnx
│   └── liveportrait/
│       ├── appearance_feature_extractor.onnx
│       ├── motion_extractor.onnx
│       ├── stitching_retargeting.onnx
│       └── warping_spatially_adaptive_network.onnx
├── raw_data/
├── dataset/
├── embeddings/
├── video_data/
└── README.md
```

### 9. Workflow build bo cai dat tu dong

File workflow GitHub Actions:
- `.github/workflows/build-distributions.yml`

Workflow se tu dong tao artifact:
- Windows: `DeepFakeRT.exe` (nen zip)
- Linux: `deepfake-rt_0.1.0_amd64.deb`
- macOS: `DeepFakeRT-macos.dmg`

Cach kich hoat:
- Push len nhanh `main`
- Chay thu cong qua `workflow_dispatch`

### 10. Luu y su dung

Du an chi phuc vu hoc tap va nghien cuu. Khong duoc dung cho hanh vi gia mao, lua dao, quay roi hay bat ky muc dich trai phap luat nao. Luon xin su dong y khi su dung du lieu khuon mat.

---

Created by ManhTuan28
