"""
Runtime utilities for GPU detection, ONNX provider selection, and session management.

Key features:
- Session-level caching (nvidia-smi / rocm-smi called once, not per-call)
- Auto-selects GPU with most FREE VRAM (avoids busy GPUs)
- Supports NVIDIA (CUDA/TensorRT), AMD (ROCm), Intel (DirectML/SYCL), Apple Silicon (CoreML)
- Environment overrides: DEEPFAKE_GPU_ID, DEEPFAKE_DISABLE_GPU, DEEPFAKE_ORT_PROVIDERS
"""

import os
import platform
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    vendor: str  # "nvidia", "amd", "intel", "apple"
    total_vram_mb: float = 0.0
    free_vram_mb: float = 0.0
    utilization_pct: float = 0.0  # GPU core utilization %

    @property
    def total_vram_gb(self) -> float:
        return round(self.total_vram_mb / 1024.0, 1)

    @property
    def free_vram_gb(self) -> float:
        return round(self.free_vram_mb / 1024.0, 1)

    @property
    def used_vram_mb(self) -> float:
        return self.total_vram_mb - self.free_vram_mb

    @property
    def used_vram_gb(self) -> float:
        return round(self.used_vram_mb / 1024.0, 1)


@dataclass
class GPUSummary:
    """Aggregated GPU information for the whole system."""
    nvidia_gpus: List[GPUInfo] = field(default_factory=list)
    amd_gpus: List[GPUInfo] = field(default_factory=list)
    intel_detected: bool = False
    apple_silicon: bool = False
    selected_gpu: Optional[GPUInfo] = None
    onnx_providers_available: List[str] = field(default_factory=list)
    onnx_providers_selected: List[str] = field(default_factory=list)

    @property
    def has_any_gpu(self) -> bool:
        return bool(self.nvidia_gpus or self.amd_gpus or self.intel_detected or self.apple_silicon)

    @property
    def best_vram_gb(self) -> float:
        """Best total VRAM across all GPUs."""
        vrams = [g.total_vram_gb for g in self.nvidia_gpus + self.amd_gpus]
        return max(vrams) if vrams else 0.0

    @property
    def best_free_vram_gb(self) -> float:
        """Best free VRAM across all GPUs."""
        vrams = [g.free_vram_gb for g in self.nvidia_gpus + self.amd_gpus]
        return max(vrams) if vrams else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _run_cmd(args: List[str], timeout: int = 5) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# NVIDIA Detection (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _detect_nvidia_gpus() -> List[GPUInfo]:
    """Detect all NVIDIA GPUs with total/free VRAM and utilization. Cached."""
    output = _run_cmd([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"
    ])
    if not output:
        return []

    gpus = []
    for line in output.split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4:
            try:
                gpu = GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vendor="nvidia",
                    total_vram_mb=float(parts[2]),
                    free_vram_mb=float(parts[3]),
                    utilization_pct=float(parts[4]) if len(parts) >= 5 else 0.0,
                )
                gpus.append(gpu)
            except (ValueError, IndexError):
                continue
    return gpus


def _select_best_nvidia_gpu(gpus: List[GPUInfo]) -> Optional[GPUInfo]:
    """Select NVIDIA GPU with most FREE VRAM. Respects user overrides."""
    # User override
    override = os.getenv("DEEPFAKE_GPU_ID") or os.getenv("CUDA_VISIBLE_DEVICES")
    if override and override.strip():
        gpu_id = override.strip().split(",")[0]  # Take first if comma-separated
        try:
            idx = int(gpu_id)
            match = next((g for g in gpus if g.index == idx), None)
            if match:
                print(f"[GPU] ✅ Using manual override: GPU {idx} ({match.name})")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
                return match
        except ValueError:
            pass

    if not gpus:
        return None

    # Sort by free VRAM descending → picks the least-busy GPU
    ranked = sorted(gpus, key=lambda g: g.free_vram_mb, reverse=True)
    best = ranked[0]

    if len(gpus) > 1:
        print(f"[GPU] 🔍 Found {len(gpus)} NVIDIA GPU(s). Ranking by free VRAM:")
        for i, g in enumerate(ranked):
            marker = "→" if g is best else " "
            print(f"  {marker} GPU {g.index}: {g.name}  |  "
                  f"Free: {g.free_vram_gb:.1f}/{g.total_vram_gb:.1f} GB  |  "
                  f"Util: {g.utilization_pct:.0f}%")
    else:
        print(f"[GPU] ✅ NVIDIA GPU {best.index}: {best.name} "
              f"(Free: {best.free_vram_gb:.1f}/{best.total_vram_gb:.1f} GB)")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(best.index)
    return best


# ---------------------------------------------------------------------------
# AMD Detection (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _detect_amd_gpus() -> List[GPUInfo]:
    """Detect AMD GPUs via rocm-smi or sysfs fallback. Cached.
    
    Detection order:
    1. rocm-smi (if installed) — best for ROCm-enabled GPUs
    2. Linux sysfs /sys/class/drm/card*/device/ — works without ROCm
    3. ROCm environment variables — minimal fallback
    """
    import shutil

    # ── Method 1: rocm-smi ────────────────────────────────────
    if shutil.which("rocm-smi"):
        gpus = _detect_amd_gpus_rocmsmi()
        if gpus:
            return gpus

    # ── Method 2: Linux sysfs (works without ROCm) ────────────
    gpus = _detect_amd_gpus_sysfs()
    if gpus:
        return gpus

    # ── Method 3: ROCm environment only ───────────────────────
    if os.getenv("ROCM_HOME") or os.path.exists("/opt/rocm"):
        return [GPUInfo(index=0, name="AMD GPU (ROCm env detected)", vendor="amd")]

    return []


# Known AMD PCI device IDs → friendly names (common discrete GPUs)
_AMD_PCI_NAMES = {
    "7340": "Radeon RX 5500/5500M",
    "7341": "Radeon RX 5500 XT",
    "731f": "Radeon RX 5600/5700",
    "73bf": "Radeon RX 6900 XT",
    "73df": "Radeon RX 6700 XT",
    "73ff": "Radeon RX 6600/6600 XT",
    "744c": "Radeon RX 7900 XTX",
    "7480": "Radeon RX 7600",
    "15e7": "Radeon Vega (Ryzen iGPU)",
    "1636": "Radeon Vega 6 (Ryzen 4000/5000)",
    "1638": "Radeon Vega 7 (Ryzen 4000/5000)",
    "164c": "Radeon Vega (Ryzen 6000)",
    "15bf": "Radeon Vega 8 (Ryzen 3000)",
}


def _detect_amd_gpus_sysfs() -> List[GPUInfo]:
    """Detect AMD GPUs via Linux sysfs — works without ROCm installed."""
    import glob

    AMD_VENDOR_ID = "0x1002"
    gpus = []
    card_dirs = sorted(glob.glob("/sys/class/drm/card[0-9]*/device"))

    for idx, card_dev in enumerate(card_dirs):
        # Check vendor
        vendor_path = os.path.join(card_dev, "vendor")
        if not os.path.exists(vendor_path):
            continue
        try:
            vendor = open(vendor_path).read().strip()
        except OSError:
            continue
        if vendor != AMD_VENDOR_ID:
            continue

        # Get device ID for name lookup
        device_id = ""
        device_path = os.path.join(card_dev, "device")
        if os.path.exists(device_path):
            try:
                device_id = open(device_path).read().strip().replace("0x", "")
            except OSError:
                pass

        gpu_name = _AMD_PCI_NAMES.get(device_id, f"AMD GPU (PCI 0x{device_id})")

        # Read VRAM from sysfs (amdgpu driver exposes this)
        total_vram_bytes = 0
        used_vram_bytes = 0

        vram_total_path = os.path.join(card_dev, "mem_info_vram_total")
        vram_used_path = os.path.join(card_dev, "mem_info_vram_used")

        if os.path.exists(vram_total_path):
            try:
                total_vram_bytes = int(open(vram_total_path).read().strip())
            except (OSError, ValueError):
                pass
        if os.path.exists(vram_used_path):
            try:
                used_vram_bytes = int(open(vram_used_path).read().strip())
            except (OSError, ValueError):
                pass

        total_mb = total_vram_bytes / (1024 * 1024)
        used_mb = used_vram_bytes / (1024 * 1024)

        gpus.append(GPUInfo(
            index=len(gpus),
            name=gpu_name,
            vendor="amd",
            total_vram_mb=total_mb,
            free_vram_mb=max(0, total_mb - used_mb),
        ))

    if gpus:
        print(f"[GPU] 🔍 Detected {len(gpus)} AMD GPU(s) via sysfs (no ROCm required)")
        for g in gpus:
            label = "dedicated" if g.total_vram_gb >= 1.0 else "integrated"
            print(f"  GPU {g.index}: {g.name} ({label})  |  "
                  f"VRAM: {g.free_vram_gb:.1f} free / {g.total_vram_gb:.1f} GB")

    return gpus


def _detect_amd_gpus_rocmsmi() -> List[GPUInfo]:
    """Detect AMD GPUs via rocm-smi."""
    gpus = []

    id_output = _run_cmd(["rocm-smi", "--showid"])
    if not id_output or "GPU" not in id_output:
        return []

    mem_output = _run_cmd(["rocm-smi", "--showmeminfo", "vram"])

    gpu_indices = []
    for line in id_output.split('\n'):
        line = line.strip()
        if line and line.startswith("GPU["):
            try:
                idx = int(line.split("[")[1].split("]")[0])
                gpu_indices.append(idx)
            except (ValueError, IndexError):
                continue

    if not gpu_indices:
        gpu_indices = [0]

    vram_data: Dict[int, Tuple[float, float]] = {}
    if mem_output:
        current_gpu = -1
        total_bytes = 0
        used_bytes = 0
        for line in mem_output.split('\n'):
            line_lower = line.lower().strip()
            if line_lower.startswith("gpu["):
                try:
                    current_gpu = int(line_lower.split("[")[1].split("]")[0])
                except (ValueError, IndexError):
                    current_gpu = -1
            elif "total" in line_lower and "memory" in line_lower:
                parts = line.split()
                for p in parts:
                    try:
                        val = float(p.replace(",", ""))
                        if val > 1000:
                            total_bytes = val
                    except ValueError:
                        continue
            elif "used" in line_lower and "memory" in line_lower:
                parts = line.split()
                for p in parts:
                    try:
                        val = float(p.replace(",", ""))
                        if val >= 0:
                            used_bytes = val
                    except ValueError:
                        continue
                if current_gpu >= 0:
                    total_mb = total_bytes / (1024 * 1024) if total_bytes > 1_000_000 else total_bytes
                    used_mb = used_bytes / (1024 * 1024) if used_bytes > 1_000_000 else used_bytes
                    vram_data[current_gpu] = (total_mb, used_mb)

    for idx in gpu_indices:
        total_mb, used_mb = vram_data.get(idx, (0.0, 0.0))
        gpus.append(GPUInfo(
            index=idx,
            name=f"AMD GPU {idx}",
            vendor="amd",
            total_vram_mb=total_mb,
            free_vram_mb=max(0, total_mb - used_mb),
        ))

    return gpus


def _select_best_amd_gpu(gpus: List[GPUInfo]) -> Optional[GPUInfo]:
    """Select AMD GPU with most free VRAM. Respects user overrides."""
    override = os.getenv("DEEPFAKE_GPU_ID") or os.getenv("HIP_VISIBLE_DEVICES")
    if override and override.strip():
        gpu_id = override.strip().split(",")[0]
        try:
            idx = int(gpu_id)
            match = next((g for g in gpus if g.index == idx), None)
            if match:
                print(f"[GPU] ✅ Using manual override: AMD GPU {idx}")
                os.environ["HIP_VISIBLE_DEVICES"] = str(idx)
                return match
        except ValueError:
            pass

    if not gpus:
        return None

    ranked = sorted(gpus, key=lambda g: g.free_vram_mb, reverse=True)
    best = ranked[0]
    print(f"[GPU] ✅ AMD (ROCm) GPU {best.index}: {best.name} "
          f"(Free: {best.free_vram_gb:.1f}/{best.total_vram_gb:.1f} GB)")
    os.environ["HIP_VISIBLE_DEVICES"] = str(best.index)
    return best


# ---------------------------------------------------------------------------
# Intel Detection (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _detect_intel_gpu() -> bool:
    """Check if Intel GPU is available via clinfo or sycl-ls. Cached."""
    # Check clinfo
    clinfo = _run_cmd(["clinfo"])
    if clinfo and "Intel" in clinfo:
        print("[GPU] ✅ Intel GPU detected via OpenCL")
        return True

    # Check sycl-ls (Intel oneAPI)
    sycl = _run_cmd(["sycl-ls"])
    if sycl and "Intel" in sycl:
        print("[GPU] ✅ Intel GPU detected via SYCL (oneAPI)")
        return True

    return False


# ---------------------------------------------------------------------------
# Apple Silicon Detection (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _detect_apple_silicon() -> bool:
    """Check if running on Apple Silicon (macOS arm64). Cached."""
    if platform.system() != "Darwin":
        return False
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        print("[GPU] ✅ Apple Silicon detected (CoreML available)")
        return True
    return False


# ---------------------------------------------------------------------------
# GPU Summary (cached, single entry point)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_gpu_summary() -> GPUSummary:
    """
    Detect all GPUs and select the best one. Cached for the session.
    
    This is the SINGLE SOURCE OF TRUTH for GPU info — used by check_gpus.py,
    06_setup_models.py, and all consumer scripts.
    """
    summary = GPUSummary()

    # Detect all GPU types
    summary.nvidia_gpus = list(_detect_nvidia_gpus())
    summary.amd_gpus = list(_detect_amd_gpus())
    summary.intel_detected = _detect_intel_gpu()
    summary.apple_silicon = _detect_apple_silicon()

    # Select best GPU (NVIDIA preferred, then AMD)
    if summary.nvidia_gpus:
        summary.selected_gpu = _select_best_nvidia_gpu(summary.nvidia_gpus)
    elif summary.amd_gpus:
        summary.selected_gpu = _select_best_amd_gpu(summary.amd_gpus)

    # Get ONNX providers
    try:
        import onnxruntime as ort
        summary.onnx_providers_available = list(ort.get_available_providers())
    except Exception:
        summary.onnx_providers_available = ["CPUExecutionProvider"]

    summary.onnx_providers_selected = _build_provider_list(summary)

    return summary


# ---------------------------------------------------------------------------
# ONNX Provider Selection
# ---------------------------------------------------------------------------

def _build_provider_list(summary: GPUSummary) -> List[str]:
    """Build prioritized provider list based on detected hardware."""
    disable_gpu = os.getenv("DEEPFAKE_DISABLE_GPU", "0").strip() == "1"
    provider_override = os.getenv("DEEPFAKE_ORT_PROVIDERS", "").strip()
    available = set(summary.onnx_providers_available)

    # Manual override
    if provider_override:
        requested = [p.strip() for p in provider_override.split(",") if p.strip()]
        providers = [p for p in requested if p in available]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        print(f"[PROVIDERS] ⚙️ Using manual override: {providers}")
        return _dedupe(providers)

    # GPU disabled
    if disable_gpu:
        print("[PROVIDERS] ⚠️ GPU disabled via DEEPFAKE_DISABLE_GPU — CPU only")
        return ["CPUExecutionProvider"]

    system = platform.system().lower()
    preferred = []

    # Priority: TensorRT > CUDA > ROCm > CoreML > DirectML > CPU
    if "TensorrtExecutionProvider" in available and summary.nvidia_gpus:
        preferred.append("TensorrtExecutionProvider")

    if "CUDAExecutionProvider" in available and summary.nvidia_gpus:
        preferred.append("CUDAExecutionProvider")

    if "ROCMExecutionProvider" in available and summary.amd_gpus:
        preferred.append("ROCMExecutionProvider")

    if "CoreMLExecutionProvider" in available and summary.apple_silicon:
        preferred.append("CoreMLExecutionProvider")

    if "DmlExecutionProvider" in available:
        # DirectML: useful on Windows for AMD/Intel without ROCm
        if system == "windows" or (not summary.nvidia_gpus and not summary.amd_gpus):
            preferred.append("DmlExecutionProvider")

    # Always have CPU as fallback
    preferred.append("CPUExecutionProvider")

    providers = [p for p in preferred if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    result = _dedupe(providers)
    gpu_label = "CPU only" if result == ["CPUExecutionProvider"] else result[0].replace("ExecutionProvider", "")
    print(f"[PROVIDERS] 🚀 Selected: {result}  (primary: {gpu_label})")
    return result


def get_onnxruntime_providers() -> List[str]:
    """Get the selected ONNX Runtime providers. Uses cached GPU summary."""
    return get_gpu_summary().onnx_providers_selected


# ---------------------------------------------------------------------------
# ONNX Session / InsightFace helpers
# ---------------------------------------------------------------------------

def _optimal_thread_count() -> int:
    """Determine optimal thread count for ONNX Runtime."""
    cores = os.cpu_count() or 4
    # Use half the cores to leave room for other tasks (video capture, UI, etc.)
    return max(1, min(cores // 2, 8))


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

    threads = _optimal_thread_count()
    session_options.intra_op_num_threads = threads
    session_options.inter_op_num_threads = max(1, threads // 2)

    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=get_onnxruntime_providers(),
    )


def open_video_capture(index: int = 0):
    import cv2

    if platform.system().lower() == "linux":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(index)
