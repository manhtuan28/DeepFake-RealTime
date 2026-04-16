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
        return _dedupe(providers)

    if disable_gpu:
        return ["CPUExecutionProvider"]

    system = platform.system().lower()
    preferred = []

    if system == "windows":
        preferred.extend(["DmlExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"])
    else:
        preferred.extend(["CUDAExecutionProvider", "TensorrtExecutionProvider", "ROCMExecutionProvider"])

    preferred.extend(["DmlExecutionProvider", "CPUExecutionProvider"])

    providers = [provider for provider in preferred if provider in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    elif "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    return _dedupe(providers)


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
