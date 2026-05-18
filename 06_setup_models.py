import json
import os
import platform
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from runtime_utils import get_gpu_summary


MODEL_DIR = "models"
LIVEPORTRAIT_DIR = os.path.join(MODEL_DIR, "liveportrait")


I18N = {
    "en": {
        "header": "AUTO MODEL SETUP",
        "profile": "Detected machine profile",
        "os": "OS",
        "cpu": "CPU cores",
        "ram": "RAM",
        "disk": "Free disk for models",
        "gpu": "NVIDIA GPU",
        "vram": "NVIDIA VRAM",
        "providers": "ONNX providers",
        "recommended": "Recommended tier",
        "reasons": "Why this tier",
        "catalog_saved": "Model catalog saved",
        "selected_models": "Models selected for download/check",
        "starting": "Starting",
        "skip": "Exists",
        "warn_missing_url": "Missing URL source for",
        "warn_set_env": "Set DEEPFAKE_LIVEPORTRAIT_BASE_URL to auto-download LivePortrait models.",
        "download": "Download",
        "downloaded": "Downloaded",
        "completed_warn": "Completed with warnings. Missing models",
        "completed_ok": "All required models for this machine tier are ready.",
        "prompt_override": "Use recommended tier? [Enter=yes, w=weak, b=balanced, s=strong]: ",
        "tier_changed": "Using manually selected tier",
        "yes": "yes",
        "no": "no",
    },
    "vi": {
        "header": "CÀI ĐẶT MODEL TỰ ĐỘNG",
        "profile": "Thông tin cấu hình máy",
        "os": "Hệ điều hành",
        "cpu": "Số nhân CPU",
        "ram": "RAM",
        "disk": "Dung lượng trong cho models",
        "gpu": "NVIDIA GPU",
        "vram": "VRAM NVIDIA",
        "providers": "ONNX providers",
        "recommended": "Gói model đề xuất",
        "reasons": "Lý do chọn gói",
        "catalog_saved": "Đã lưu thông tin model",
        "selected_models": "Danh sách model sẽ kiểm tra/tải",
        "starting": "Bắt đầu",
        "skip": "Đã có",
        "warn_missing_url": "Chưa có URL để tải",
        "warn_set_env": "Đặt DEEPFAKE_LIVEPORTRAIT_BASE_URL để tự động tải bộ LivePortrait.",
        "download": "Đang tải",
        "downloaded": "Đã tải xong",
        "completed_warn": "Hoàn tất có cảnh báo. Model còn thiếu",
        "completed_ok": "Tất cả model cần thiết cho máy này đã sẵn sàng.",
        "prompt_override": "Dùng gói đề xuất? [Enter=có, w=yếu, b=trung bình, s=mạnh]: ",
        "tier_changed": "Đã dùng gói do bạn chọn",
        "yes": "có",
        "no": "không",
    },
}


@dataclass
class ModelInfo:
    key: str
    path: str
    purpose: str
    required_for: str
    size_hint: str
    url: Optional[str] = None


def get_lang() -> str:
    raw = os.getenv("DEEPFAKE_LANG", "en").strip().lower()
    return "vi" if raw.startswith("vi") else "en"


def tr(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["en"]).get(key, key)


def get_total_ram_gb() -> float:
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
    return 0.0


def get_free_disk_gb(path: str = MODEL_DIR) -> float:
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    return round(free / (1024 ** 3), 1)


def classify_machine() -> Tuple[str, Dict[str, object], List[str]]:
    """Classify machine using centralized GPU detection from runtime_utils."""
    summary = get_gpu_summary()

    cpu_cores = os.cpu_count() or 1
    ram_gb = get_total_ram_gb()
    disk_gb = get_free_disk_gb()

    # GPU info from centralized detection
    has_nvidia = bool(summary.nvidia_gpus)
    has_amd = bool(summary.amd_gpus)
    nvidia_vram_gb = max((g.total_vram_gb for g in summary.nvidia_gpus), default=0.0)
    amd_vram_gb = max((g.total_vram_gb for g in summary.amd_gpus), default=0.0)
    nvidia_name = summary.nvidia_gpus[0].name if summary.nvidia_gpus else ""
    amd_name = summary.amd_gpus[0].name if summary.amd_gpus else ""
    providers = summary.onnx_providers_available

    score = 0
    reasons = []

    # CPU scoring
    if cpu_cores >= 12:
        score += 3
        reasons.append(f"high CPU core count ({cpu_cores})")
    elif cpu_cores >= 8:
        score += 2
        reasons.append(f"good CPU core count ({cpu_cores})")
    elif cpu_cores >= 4:
        score += 1
        reasons.append(f"baseline CPU core count ({cpu_cores})")

    # RAM scoring
    if ram_gb >= 24:
        score += 3
        reasons.append(f"high RAM ({ram_gb:.0f} GB)")
    elif ram_gb >= 16:
        score += 2
        reasons.append(f"good RAM ({ram_gb:.0f} GB)")
    elif ram_gb >= 8:
        score += 1
        reasons.append(f"minimum recommended RAM ({ram_gb:.0f} GB)")

    # GPU detection scoring (don't double-count if both NVIDIA and AMD)
    if has_nvidia:
        score += 2
        reasons.append(f"NVIDIA GPU ({nvidia_name})")
    elif has_amd:
        score += 2
        reasons.append(f"AMD GPU ({amd_name})")
    elif summary.apple_silicon:
        score += 2
        reasons.append("Apple Silicon (CoreML)")

    # VRAM scoring — use the BEST GPU, not sum of all
    best_vram = max(nvidia_vram_gb, amd_vram_gb)
    if best_vram >= 10:
        score += 3
        reasons.append(f"high GPU VRAM ({best_vram:.0f} GB)")
    elif best_vram >= 6:
        score += 2
        reasons.append(f"usable GPU VRAM ({best_vram:.0f} GB)")
    elif best_vram >= 2:
        score += 1
        reasons.append(f"low GPU VRAM ({best_vram:.0f} GB)")

    # Disk scoring
    if disk_gb < 8:
        score -= 2
        reasons.append("very low free disk")
    elif disk_gb < 15:
        score -= 1
        reasons.append("limited free disk")
    else:
        reasons.append(f"enough free disk ({disk_gb:.0f} GB)")

    # Tier classification
    if score >= 8:
        tier = "strong"
    elif score >= 4:
        tier = "balanced"
    else:
        tier = "weak"

    profile = {
        "cpu_cores": cpu_cores,
        "ram_gb": round(ram_gb, 1),
        "nvidia_gpu": has_nvidia,
        "nvidia_gpu_name": nvidia_name,
        "nvidia_vram_gb": nvidia_vram_gb,
        "amd_gpu": has_amd,
        "amd_gpu_name": amd_name,
        "amd_vram_gb": amd_vram_gb,
        "apple_silicon": summary.apple_silicon,
        "disk_free_gb": disk_gb,
        "onnx_providers": providers,
        "score": score,
    }
    return tier, profile, reasons


def build_catalog() -> Dict[str, ModelInfo]:
    liveportrait_base = os.getenv("DEEPFAKE_LIVEPORTRAIT_BASE_URL", "").rstrip("/")

    def lp_url(filename: str) -> Optional[str]:
        if not liveportrait_base:
            return None
        return f"{liveportrait_base}/{filename}"

    return {
        "inswapper": ModelInfo(
            key="inswapper",
            path=os.path.join(MODEL_DIR, "inswapper_128.onnx"),
            purpose="Core face swap inference model.",
            required_for="03_run_webcam.py and 05_run_video_file.py",
            size_hint="~529 MB",
            url="https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        ),
        "gpen": ModelInfo(
            key="gpen",
            path=os.path.join(MODEL_DIR, "GPEN-BFR-512.onnx"),
            purpose="Optional face restoration/enhancement model.",
            required_for="High-detail enhancement workflows",
            size_hint="~272 MB",
            url="https://huggingface.co/nguyenvando/GPEN-BFR-512/resolve/main/GPEN-BFR-512.onnx",
        ),
        "lp_appearance": ModelInfo(
            key="lp_appearance",
            path=os.path.join(LIVEPORTRAIT_DIR, "appearance_feature_extractor.onnx"),
            purpose="Extracts appearance feature tensor for whole-head synthesis.",
            required_for="07_head_stitcher.py",
            size_hint="~3 MB",
            url=lp_url("appearance_feature_extractor.onnx"),
        ),
        "lp_motion": ModelInfo(
            key="lp_motion",
            path=os.path.join(LIVEPORTRAIT_DIR, "motion_extractor.onnx"),
            purpose="Extracts driving motion/keypoints from input frames.",
            required_for="07_head_stitcher.py",
            size_hint="~108 MB",
            url=lp_url("motion_extractor.onnx"),
        ),
        "lp_stitch": ModelInfo(
            key="lp_stitch",
            path=os.path.join(LIVEPORTRAIT_DIR, "stitching_retargeting.onnx"),
            purpose="Blends source/driving keypoints before rendering.",
            required_for="07_head_stitcher.py",
            size_hint="<1 MB",
            url=lp_url("stitching_retargeting.onnx"),
        ),
        "lp_warp": ModelInfo(
            key="lp_warp",
            path=os.path.join(LIVEPORTRAIT_DIR, "warping_spatially_adaptive_network.onnx"),
            purpose="Main warping renderer for full head output.",
            required_for="07_head_stitcher.py",
            size_hint="~174 MB",
            url=lp_url("warping_spatially_adaptive_network.onnx"),
        ),
    }


def tier_plan(tier: str) -> List[str]:
    plans = {
        "weak": ["inswapper"],
        "balanced": ["inswapper", "lp_appearance", "lp_motion", "lp_stitch", "lp_warp"],
        "strong": ["inswapper", "gpen", "lp_appearance", "lp_motion", "lp_stitch", "lp_warp"],
    }
    return plans[tier]


def download_with_progress(url: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def reporthook(block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        pct = downloaded * 100.0 / total_size
        print(f"\r    {pct:5.1f}% ({downloaded // (1024 * 1024)} / {total_size // (1024 * 1024)} MB)", end="")

    urllib.request.urlretrieve(url, output_path, reporthook)
    print()


def write_catalog(catalog: Dict[str, ModelInfo], profile: Dict[str, object], tier: str, reasons: List[str]) -> str:
    output = {
        "machine_profile": profile,
        "recommended_tier": tier,
        "recommendation_reasons": reasons,
        "models": {k: asdict(v) for k, v in catalog.items()},
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, "model_catalog.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return out_path


def run_setup() -> int:
    lang = get_lang()
    tier, profile, reasons = classify_machine()
    catalog = build_catalog()
    if os.isatty(0):
        override = input(tr(lang, "prompt_override")).strip().lower()
        if override in {"w", "b", "s"}:
            tier = {"w": "weak", "b": "balanced", "s": "strong"}[override]
            print(f"{tr(lang, 'tier_changed')}: {tier}")
    plan = tier_plan(tier)
    catalog_path = write_catalog(catalog, profile, tier, reasons)

    print("=" * 60)
    print(tr(lang, "header"))
    print("=" * 60)
    print(f"{tr(lang, 'profile')}")
    print(f"- {tr(lang, 'os')}: {platform.system()} {platform.release()}")
    print(f"- {tr(lang, 'cpu')}: {profile['cpu_cores']}")
    print(f"- {tr(lang, 'ram')}: {profile['ram_gb']} GB")
    print(f"- {tr(lang, 'disk')}: {profile['disk_free_gb']} GB")
    gpu_label = profile.get('nvidia_gpu_name') or profile.get('amd_gpu_name') or ("Apple Silicon" if profile.get('apple_silicon') else tr(lang, 'no'))
    print(f"- {tr(lang, 'gpu')}: {gpu_label}")
    best_vram = max(profile.get('nvidia_vram_gb', 0), profile.get('amd_vram_gb', 0))
    print(f"- {tr(lang, 'vram')}: {best_vram} GB")
    print(f"- {tr(lang, 'providers')}: {', '.join(profile['onnx_providers']) if profile['onnx_providers'] else 'n/a'}")
    print(f"- {tr(lang, 'recommended')}: {tier} (score: {profile.get('score', '?')})")
    print(f"- {tr(lang, 'reasons')}: {', '.join(reasons)}")
    print(f"- {tr(lang, 'catalog_saved')}: {catalog_path}")
    print(f"\n{tr(lang, 'selected_models')}:")
    for key in plan:
        model = catalog[key]
        print(f"- {model.path} ({model.size_hint})")

    print(f"\n{tr(lang, 'starting')}...\n")
    failures = []

    for key in plan:
        model = catalog[key]
        if os.path.exists(model.path):
            print(f"[SKIP] {tr(lang, 'skip')}: {model.path}")
            continue

        if not model.url:
            print(f"[WARN] {tr(lang, 'warn_missing_url')}: {model.path}")
            print(f"       {tr(lang, 'warn_set_env')}")
            failures.append(model.path)
            continue

        print(f"[DL] {tr(lang, 'download')}: {model.path}")
        try:
            download_with_progress(model.url, model.path)
            print(f"[OK] {tr(lang, 'downloaded')}: {model.path}")
        except Exception as exc:
            print(f"[ERR] {model.path}: {exc}")
            failures.append(model.path)

    if failures:
        print(f"\n{tr(lang, 'completed_warn')}:")
        for item in failures:
            print(f"- {item}")
        return 1

    print(f"\n{tr(lang, 'completed_ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_setup())