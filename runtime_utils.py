import os
import platform
from typing import List, Optional, Sequence


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _detect_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            print(f"[GPU] Detected NVIDIA GPU: {gpu_name}")
            return True
    except Exception:
        pass
    return False


def _detect_amd_gpu() -> bool:
    """Check if AMD GPU (ROCm) is available."""
    try:
        import subprocess
        result = subprocess.run(["rocm-smi", "--showid"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "GPU" in result.stdout:
            print(f"[GPU] Detected AMD GPU with ROCm")
            return True
    except Exception:
        pass
    
    # Fallback: check ROCm environment
    if os.getenv("ROCM_HOME") or os.path.exists("/opt/rocm"):
        print(f"[GPU] Detected ROCm installation")
        return True
    
    return False


def _detect_intel_gpu() -> bool:
    """Check if Intel GPU is available."""
    try:
        import subprocess
        result = subprocess.run(["clinfo"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "Intel" in result.stdout:
            print(f"[GPU] Detected Intel GPU")
            return True
    except Exception:
        pass
    return False


def get_onnxruntime_providers() -> List[str]:
    disable_gpu = os.getenv("DEEPFAKE_DISABLE_GPU", "0").strip() == "1"
    provider_override = os.getenv("DEEPFAKE_ORT_PROVIDERS", "").strip()

    try:
        import onnxruntime as ort
    except Exception:
        return ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())

    if provider_override:
        requested = [item.strip() for item in provider_override.split(",") if item.strip()]
        providers = [provider for provider in requested if provider in available]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        print(f"[PROVIDERS] Using override: {providers}")
        return _dedupe(providers)

    if disable_gpu:
        print(f"[PROVIDERS] GPU disabled via DEEPFAKE_DISABLE_GPU, using CPU only")
        return ["CPUExecutionProvider"]

    system = platform.system().lower()
    preferred = []

    # Auto-detect GPU and prioritize accordingly
    nvidia_detected = "CUDAExecutionProvider" in available and _detect_nvidia_gpu()
    amd_detected = "ROCMExecutionProvider" in available and _detect_amd_gpu()
    intel_detected = "DmlExecutionProvider" in available and system == "windows" and _detect_intel_gpu()

    # Build provider list based on detection
    if system == "windows":
        if nvidia_detected:
            preferred.append("CUDAExecutionProvider")
        if intel_detected or "DmlExecutionProvider" in available:
            preferred.append("DmlExecutionProvider")
        if amd_detected:
            preferred.append("ROCMExecutionProvider")
    else:  # Linux/macOS
        if nvidia_detected:
            preferred.append("CUDAExecutionProvider")
        if amd_detected:
            preferred.append("ROCMExecutionProvider")
        if "TensorrtExecutionProvider" in available:
            preferred.append("TensorrtExecutionProvider")

    # Fallback providers
    if "DmlExecutionProvider" in available and "DmlExecutionProvider" not in preferred:
        preferred.append("DmlExecutionProvider")
    if "DmlExecutionProvider" not in available or system != "windows":
        preferred.append("CPUExecutionProvider")

    providers = [provider for provider in preferred if provider in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    elif "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    result = _dedupe(providers)
    print(f"[PROVIDERS] Selected: {result}")
    return result


def get_insightface_kwargs(model_name: Optional[str] = None, allowed_modules=None) -> dict:
    kwargs = {"providers": get_onnxruntime_providers()}
    if model_name is not None:
        kwargs["name"] = model_name
    if allowed_modules is not None:
        kwargs["allowed_modules"] = allowed_modules
    return kwargs


def create_face_analysis(model_name: str = "buffalo_l", allowed_modules=None):
    from insightface.app import FaceAnalysis

    return FaceAnalysis(**get_insightface_kwargs(model_name=model_name, allowed_modules=allowed_modules))


def create_onnx_session(model_path: str):
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=session_options, providers=get_onnxruntime_providers())


def open_video_capture(index: int = 0):
    import cv2

    if platform.system().lower() == "linux":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(index)
