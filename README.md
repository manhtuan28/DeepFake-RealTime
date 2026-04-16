# 🎭 AI Real-Time Face Swap (Multi-backend)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Auto_backend-blue)
![Status](https://img.shields.io/badge/Status-Experimental-warning)

A lightweight, real-time face swapping application built with Python, InsightFace, and ONNX Runtime. 

**Key Highlight:** This project now auto-selects the best available inference backend on the current machine: **CUDA** on Linux/Nvidia, **DirectML** on Windows/AMD, and **CPU** as a fallback. The default pipeline stays lightweight for real-time use and only loads the enhancer when you choose to use it.

**Language / Ngôn ngữ:** English | [Tiếng Việt](#phiên-bản-tiếng-việt)

---

## Phiên bản tiếng Việt

Đây là công cụ swap khuôn mặt theo thời gian thực được viết bằng Python, InsightFace và ONNX Runtime.

**Điểm chính:** Dự án tự chọn backend phù hợp trên máy hiện tại: **CUDA** trên Linux/Nvidia, **DirectML** trên Windows/AMD và **CPU** làm phương án dự phòng. Luồng mặc định được giữ nhẹ để chạy realtime, còn chế độ tăng chất lượng chỉ tải khi bạn bật.

### Cài đặt nhanh

```bash
pip install -r requirements.txt
```

Nếu muốn tăng tốc bằng GPU:

```bash
# Linux / NVIDIA
pip install onnxruntime-gpu

# Windows / AMD
pip install onnxruntime-directml
```

### Cách dùng nhanh

1. Chạy `python app.py`.
2. Chọn xử lý dữ liệu, tạo embeddings, chạy webcam, xuất video hoặc chế độ fake toàn bộ đầu.
3. Nếu FPS thấp, bật CPU-only bằng `DEEPFAKE_DISABLE_GPU=1` hoặc giảm `DEEPFAKE_HEAD_FRAME_SKIP` trong chế độ head swap.

### Lưu ý

- `inswapper_128.onnx` là model bắt buộc cho face swap.
- `GPEN-BFR-512.onnx` là tùy chọn nếu bạn muốn tăng chất lượng ảnh.
- Chế độ fake toàn bộ đầu nằm ở option 6 trong menu.

---

## ✨ Features

- **🚀 Real-time Performance:** Swaps faces directly on webcam feed with low latency.
- **🔴 Cross-platform acceleration:** Auto-detects `CUDAExecutionProvider`, `DmlExecutionProvider`, or CPU depending on what is installed.
- **💎 HD Mode (Face Enhancement):** Integrated **GPEN-BFR-512** to restore facial details (skin texture, eyes) up to 512px resolution.
- **🧠 Whole-head mode:** Separate LivePortrait-based path for faking the full head instead of only the face.
- **🎬 Smart Video Extraction:** Automatically extracts high-quality training images from video files (e.g., YouTube interviews), filtering out blurry frames and unwanted people.
- **⚡ Incremental Processing:** Smart data pipeline that skips already processed images/embeddings to save time.
- **🖥️ Split-View UI:** Real-time comparison interface (Original vs. Fake) with FPS counter and active model status.

---

## 🛠️ Tech Stack

- **Core:** Python 3.x
- **Computer Vision:** OpenCV, InsightFace
- **Inference Engine:** ONNX Runtime (auto provider selection)
- **Models:**
  - `inswapper_128.onnx` (Face Swapping)
  - `GPEN-BFR-512.onnx` (Face Restoration/Enhancement)

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/manhtuan28/DeepFake-RealTime.git
cd DeepFake-RealTime
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
*Note: `requirements.txt` installs the CPU-safe base package. For GPU acceleration, install one accelerator package that matches your platform:*

```bash
# Linux / NVIDIA
pip install onnxruntime-gpu

# Windows / AMD
pip install onnxruntime-directml
```

*Keep `numpy<2.0` to avoid compatibility issues with ONNX Runtime builds.*

### 3. Download Models
Due to GitHub's file size limits, you must download the models manually and place them in the **models/** directory of the project:

| Model | Description | Required? | Link |
| :--- | :--- | :---: | :--- |
| **inswapper_128.onnx** | The core face swap model. | ✅ Yes | [Download here](https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx) |
| **GPEN-BFR-512.onnx** | Enhancer for HD details. | ❌ Optional | [Download here](https://huggingface.co/nguyenvando/GPEN-BFR-512/resolve/main/GPEN-BFR-512.onnx) |

Place them as:

- `models/inswapper_128.onnx`
- `models/GPEN-BFR-512.onnx`

---

## 🚀 Usage

Run the master controller script to access all features:

```bash
python app.py
```

### The Workflow:

1.  **Option 1: [DATASET] Processing**
    * Place raw images or videos of the target person in `raw_data/PersonName/`.
    * Run Option 1. The script will detect, align, crop faces, and save them to `dataset/`.
    
2.  **Option 2: [VECTOR] Embedding Generation**
    * Run Option 2. It scans the `dataset/` folder and creates a `.npy` file (feature vector) in `embeddings/`.
    * *Why?* This allows the AI to understand the facial structure once, saving processing power during live swapping.

3.  **Option 3: [START] Real-time Webcam**
    * Opens the webcam feed.
    * **Controls:**
        * `N`: Switch to the next face in your embeddings bank.
        * `Q`: Quit the application.

4.  **Option 4: [VIDEO] Extraction (Recommended Source)**
    * Opens the webcam feed.
    * **Controls:**
        * Place .mp4 video files (e.g., 4K interviews, fancams) into video_data/.
        * (Optional) Place a sample image (e.g., sample.jpg) of the target person in the same folder to activate Smart Filter (removes interviewers/audience automatically).
        * The script extracts clean, sharp faces into dataset/.

5.  **Option 6: [HEAD] Whole-head LivePortrait**
    * Uses the LivePortrait ONNX models in `models/liveportrait/`.
    * Produces a full-head transformation rather than a face-only swap.
    * If it feels laggy, lower `DEEPFAKE_HEAD_FRAME_SKIP` or `DEEPFAKE_HEAD_DRIVING_SIZE` only as needed; higher values are faster.
---

## 📂 Project Structure

```text
├── video_data/         # Put MP4 videos here for extraction
├── raw_data/           # Put raw images here (Alternative source)
├── dataset/            # Processed 512x512 aligned faces
├── embeddings/         # Extracted feature vectors (.npy)
├── models/
│   ├── inswapper_128.onnx  # [Required] Swap Model
│   ├── GPEN-BFR-512.onnx   # [Optional] Enhance Model
│   └── liveportrait/
│       ├── appearance_feature_extractor.onnx
│       ├── motion_extractor.onnx
│       ├── stitching_retargeting.onnx
│       └── warping_spatially_adaptive_network.onnx
├── app.py              # Main launcher
├── 01_data_processing.py
├── 02_create_embeddings.py
├── 03_run_webcam.py
├── 04_video_to_dataset.py
├── requirements.txt
└── README.md
```

---

## 🔧 Troubleshooting

**Linux / NVIDIA GPU is not being used**
* **Cause:** The CPU-only ONNX Runtime package is installed, or CUDA libraries are missing.
* **Fix:** Install the GPU wheel and confirm CUDA is available:
    ```bash
    pip install onnxruntime-gpu
    python -c "import onnxruntime as ort; print(ort.get_available_providers())"
    ```

**Windows / AMD GPU is not being used**
* **Cause:** The DirectML package is not installed.
* **Fix:**
    ```bash
    pip install onnxruntime-directml
    ```

**Error: `Access Violation (0xC0000005)` or Crash on Start**
* **Cause:** Conflict between NumPy 2.0+ and an ONNX Runtime build, or driver issues.
* **Fix:** Downgrade NumPy:
    ```bash
    pip install "numpy<2.0"
    ```

**Low FPS in HD Mode**
* **Cause:** GPEN Enhancer is computationally expensive.
* **Fix:** Keep the enhancer optional. If you do not need HD details, rename/remove `models/GPEN-BFR-512.onnx` to switch back to SD mode (higher FPS).

## ⚙️ Backend Override

If you want to force a backend, set `DEEPFAKE_ORT_PROVIDERS` before running the scripts:

```bash
export DEEPFAKE_ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
python app.py
```

To force CPU-only execution, set `DEEPFAKE_DISABLE_GPU=1`.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. 
* Do not use this software for illegal acts, including but not limited to non-consensual deepfake pornography, fraud, or impersonation.
* Always obtain consent from the person whose face you are using in the dataset.
* The developers are not responsible for any misuse of this tool.

---

*Created by ManhTuan28*