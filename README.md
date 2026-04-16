# AI Real-Time Face Swap (Multi-backend)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Auto_backend-blue)
![Status](https://img.shields.io/badge/Status-Experimental-warning)

Language:
- [English Documentation](#english-documentation)
- [Tài liệu Tiếng Việt](#tài-liệu-tiếng-việt)

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

## Tài Liệu Tiếng Việt

### 1. Tổng Quan

Đây là bộ công cụ fake mặt theo thời gian thực và fake toàn bộ đầu, được xây dựng bằng Python, InsightFace, OpenCV và ONNX Runtime.

Hỗ trợ nhiều backend tăng tốc:
- CUDA (Linux/NVIDIA)
- DirectML (Windows/AMD)
- CPU (dự phòng)

Dự án đã bổ sung script tự động cài model dựa trên cấu hình máy (yếu, trung bình, mạnh).

### 2. Tính Năng Chính

- Fake mặt realtime từ webcam
- Chế độ fake toàn bộ đầu bằng LivePortrait
- Trích xuất dữ liệu khuôn mặt từ video
- Tạo embedding từ dataset
- Tự động chọn backend suy luận
- Tự động cài model theo độ mạnh yếu của máy
- Launcher thân thiện, cho phép chọn ngôn ngữ Anh/Việt
- Workflow CI tự động build gói `.exe`, `.deb`, `.dmg`

### 2.1 Giao Diện Song Ngữ

Khi khởi động launcher, bạn có thể chọn ngôn ngữ:
- English
- Tiếng Việt

Lựa chọn này được truyền cho script con qua biến môi trường `DEEPFAKE_LANG` để giao diện đồng nhất.

### 3. Tự Động Cài Model Theo Cấu Hình Máy (Mới)

Chạy trực tiếp:

```bash
python 06_setup_models.py
```

Hoặc từ menu:

```bash
python app.py
# Chọn mục 7 [MODEL]
```

Script sẽ:
- Đọc số nhân CPU, dung lượng RAM, dung lượng ổ đĩa còn trống, kiểm tra NVIDIA GPU, VRAM và ONNX providers
- Phân loại máy thành weak / balanced / strong
- In lý do đề xuất và điểm đánh giá cấu hình
- Cho phép bạn ghi đè gói model trước khi tải
- Ghi đầy đủ thông tin model vào `models/model_catalog.json`
- Tự động tải những model phù hợp với cấu hình
- Bỏ qua model đã có sẵn

Chính sách chọn gói model:

| Cấp Máy | Quy Tắc Phân Loại | Model Sẽ Cài |
| :--- | :--- | :--- |
| weak | RAM/CPU thấp | `inswapper_128.onnx` |
| balanced | Cấu hình trung bình | `inswapper_128.onnx` + bộ LivePortrait |
| strong | CPU/RAM cao + NVIDIA | `inswapper_128.onnx` + bộ LivePortrait + `GPEN-BFR-512.onnx` |

Nguồn tải model LivePortrait:
- Mặc định chỉ auto tải được các model có link công khai sẵn.
- Nếu bạn có host riêng, đặt biến:

```bash
export DEEPFAKE_LIVEPORTRAIT_BASE_URL="https://your-host/path/to/liveportrait"
```

Tên file script sẽ tìm:
- `appearance_feature_extractor.onnx`
- `motion_extractor.onnx`
- `stitching_retargeting.onnx`
- `warping_spatially_adaptive_network.onnx`

### 4. Bảng Model Chi Tiết (Công Dụng Và Mục Đích)

| File Model | Bắt Buộc | Công Dụng | Script Sử Dụng |
| :--- | :---: | :--- | :--- |
| `models/inswapper_128.onnx` | Có | Model chính để đổi danh tính khuôn mặt | `03_run_webcam.py`, `05_run_video_file.py` |
| `models/GPEN-BFR-512.onnx` | Tùy chọn | Nâng chất lượng/làm nét khuôn mặt | Chế độ ưu tiên chất lượng hơn FPS |
| `models/liveportrait/appearance_feature_extractor.onnx` | Chế độ đầu | Trích xuất đặc trưng hình dáng gốc | `07_head_stitcher.py` |
| `models/liveportrait/motion_extractor.onnx` | Chế độ đầu | Trích xuất chuyển động và keypoint | `07_head_stitcher.py` |
| `models/liveportrait/stitching_retargeting.onnx` | Chế độ đầu | Ghép và retarget keypoint | `07_head_stitcher.py` |
| `models/liveportrait/warping_spatially_adaptive_network.onnx` | Chế độ đầu | Render khung hình đầu cuối cùng | `07_head_stitcher.py` |

### 5. Cài Đặt

```bash
git clone https://github.com/manhtuan28/DeepFake-RealTime.git
cd DeepFake-RealTime
pip install -r requirements.txt
```

Chọn runtime GPU phù hợp (chỉ chọn 1):

```bash
# Linux / NVIDIA
pip install onnxruntime-gpu

# Windows / AMD
pip install onnxruntime-directml
```

Khuyến nghị để tránh lỗi tương thích:

```bash
pip install "numpy<2.0"
```

### 6. Quy Trình Sử Dụng

Chạy launcher:

```bash
python app.py
```

Ý nghĩa menu:
- 1: Xử lý dữ liệu ảnh thô thành dataset
- 2: Tạo embedding
- 3: Fake mặt realtime webcam
- 4: Cắt dữ liệu mặt từ video
- 5: Render fake vào file video
- 6: Fake toàn bộ đầu bằng LivePortrait
- 7: Tự động cài model theo cấu hình máy

### 7. Tối Ưu Hiệu Năng

Nếu FPS thấp trong chế độ fake đầu:
- Tăng `DEEPFAKE_HEAD_FRAME_SKIP` (cao hơn thì nhanh hơn, giảm độ mượt)
- Giảm `DEEPFAKE_HEAD_DRIVING_SIZE`
- Không bật enhancer nếu ưu tiên tốc độ

Bật chế độ CPU-only:

```bash
export DEEPFAKE_DISABLE_GPU=1
```

Ép thứ tự provider:

```bash
export DEEPFAKE_ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
```

### 8. Cấu Trúc Dự Án

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

### 9. Workflow Build Bộ Cài Đặt Tự Động

File workflow GitHub Actions:
- `.github/workflows/build-distributions.yml`

Workflow sẽ tự động tạo artifact:
- Windows: `DeepFakeRT.exe` (nén zip)
- Linux: `deepfake-rt_0.1.0_amd64.deb`
- macOS: `DeepFakeRT-macos.dmg`

Cách kích hoạt:
- Push lên nhánh `main`
- Chạy thủ công qua `workflow_dispatch`

### 10. Lưu Ý Sử Dụng

Dự án chỉ phục vụ học tập và nghiên cứu. Không được dùng cho hành vi giả mạo, lừa đảo, qu騷rào hay bất kỳ mục đích trái pháp luật nào. Luôn xin sự đồng ý khi sử dụng dữ liệu khuôn mặt.

---

Created by ManhTuan28
