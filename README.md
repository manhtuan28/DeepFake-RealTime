# 🎭 AI Real-Time Face Swap (AMD DirectML Optimized)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-DirectML-red)
![Status](https://img.shields.io/badge/Status-Experimental-warning)

A lightweight, real-time face swapping application built with Python, InsightFace, and ONNX Runtime. 

**Key Highlight:** This project is specifically optimized for **AMD GPUs** (using DirectML) and low-end hardware, featuring a custom pipeline that balances performance (FPS) and visual quality (GPEN Enhancement).

---

## ✨ Features

- **🚀 Real-time Performance:** Swaps faces directly on webcam feed with low latency.
- **🔴 AMD Optimization:** Uses `onnxruntime-directml` to leverage AMD Radeon GPUs (which are often unsupported by standard CUDA builds).
- **💎 HD Mode (Face Enhancement):** Integrated **GPEN-BFR-512** to restore facial details (skin texture, eyes) up to 512px resolution.
- **⚡ Incremental Processing:** Smart data pipeline that skips already processed images/embeddings to save time.
- **🖥️ Split-View UI:** Real-time comparison interface (Original vs. Fake) with FPS counter and active model status.
- **🎛️ Master Controller:** Centralized `app.py` launcher to manage the entire workflow.

---

## 🛠️ Tech Stack

- **Core:** Python 3.x
- **Computer Vision:** OpenCV, InsightFace
- **Inference Engine:** ONNX Runtime (DirectML)
- **Models:**
  - `inswapper_128.onnx` (Face Swapping)
  - `GPEN-BFR-512.onnx` (Face Restoration/Enhancement)

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/manhtuan28/DeepFake-RealTime.git
cd DeepFake-RealTime.git
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
*Note: Ensure you have `numpy<2.0` installed to avoid compatibility issues with onnxruntime.*

### 3. Download Models
Due to GitHub's file size limits, you must download the models manually and place them in the **root directory** of the project:

| Model | Description | Required? | Link |
| :--- | :--- | :---: | :--- |
| **inswapper_128.onnx** | The core face swap model. | ✅ Yes | [Download here](https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx) |
| **GPEN-BFR-512.onnx** | Enhancer for HD details. | ❌ Optional | [Download here](https://huggingface.co/nguyenvando/GPEN-BFR-512/resolve/main/GPEN-BFR-512.onnx) |

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

---

## 📂 Project Structure

```text
├── raw_data/           # Put your raw images/videos here
├── dataset/            # Processed 512x512 aligned faces
├── embeddings/         # Extracted feature vectors (.npy)
├── inswapper_128.onnx  # [Required] Swap Model
├── GPEN-BFR-512.onnx   # [Optional] Enhance Model
├── app.py              # Main launcher
├── 01_data_processing.py
├── 02_create_embeddings.py
├── 03_run_webcam.py
├── requirements.txt
└── README.md
```

---

## 🔧 Troubleshooting (AMD/DirectML)

**Error: `Access Violation (0xC0000005)` or Crash on Start**
* **Cause:** Conflict between NumPy 2.0+ and OnnxRuntime-DirectML, or Driver issues.
* **Fix:** Downgrade NumPy:
    ```bash
    pip install "numpy<2.0"
    ```

**Low FPS in HD Mode**
* **Cause:** GPEN Enhancer is computationally expensive.
* **Fix:** In `03_run_webcam.py`, verify `PROCESS_WIDTH` is set to `640` or lower. If you don't need HD details, rename/remove `GPEN-BFR-512.onnx` to switch back to SD mode (High FPS).

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. 
* Do not use this software for illegal acts, including but not limited to non-consensual deepfake pornography, fraud, or impersonation.
* Always obtain consent from the person whose face you are using in the dataset.
* The developers are not responsible for any misuse of this tool.

---

*Created by ManhTuan28*