import json
import os
import platform
import shutil
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


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
        "header": "CAI DAT MODEL TU DONG",
        "profile": "Thong tin cau hinh may",
        "os": "He dieu hanh",
        "cpu": "So nhan CPU",
        "ram": "RAM",
        "disk": "Dung luong trong cho models",
        "gpu": "NVIDIA GPU",
        "vram": "VRAM NVIDIA",
        "providers": "ONNX providers",
        "recommended": "Goi model de xuat",
        "reasons": "Ly do chon goi",
        "catalog_saved": "Da luu thong tin model",
        "selected_models": "Danh sach model se kiem tra/tai",
        "starting": "Bat dau",
        "skip": "Da co",
        "warn_missing_url": "Chua co URL de tai",
        "warn_set_env": "Dat DEEPFAKE_LIVEPORTRAIT_BASE_URL de tu dong tai bo LivePortrait.",
        "download": "Dang tai",
        "downloaded": "Da tai xong",
        "completed_warn": "Hoan tat co canh bao. Model con thieu",
        "completed_ok": "Tat ca model can thiet cho may nay da san sang.",
        "prompt_override": "Dung goi de xuat? [Enter=co, w=yeu, b=trung binh, s=manh]: ",
        "tier_changed": "Da dung goi do ban chon",
        "yes": "co",
        "no": "khong",
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


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def get_nvidia_vram_gb() -> float:
    if not has_nvidia_gpu():
        return 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        values = [float(line.strip()) for line in out.splitlines() if line.strip()]
        if not values:
            return 0.0
        return round(max(values) / 1024.0, 1)
    except Exception:
        return 0.0


def get_free_disk_gb(path: str = MODEL_DIR) -> float:
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    return round(free / (1024 ** 3), 1)


def get_onnx_providers() -> List[str]:
    try:
        import onnxruntime as ort

        return ort.get_available_providers()
    except Exception:
        return []


def classify_machine() -> Tuple[str, Dict[str, object], List[str]]:
    cpu_cores = os.cpu_count() or 1
    ram_gb = get_total_ram_gb()
    nvidia = has_nvidia_gpu()
    vram_gb = get_nvidia_vram_gb()
    disk_gb = get_free_disk_gb()
    providers = get_onnx_providers()

    score = 0
    reasons = []

    if cpu_cores >= 12:
        score += 3
        reasons.append("high CPU core count")
    elif cpu_cores >= 8:
        score += 2
        reasons.append("good CPU core count")
    elif cpu_cores >= 4:
        score += 1
        reasons.append("baseline CPU core count")

    if ram_gb >= 24:
        score += 3
        reasons.append("high RAM")
    elif ram_gb >= 16:
        score += 2
        reasons.append("good RAM")
    elif ram_gb >= 8:
        score += 1
        reasons.append("minimum recommended RAM")

    if nvidia:
        score += 2
        reasons.append("NVIDIA GPU detected")

    if vram_gb >= 10:
        score += 2
        reasons.append("high GPU VRAM")
    elif vram_gb >= 6:
        score += 1
        reasons.append("usable GPU VRAM")

    if disk_gb < 8:
        score -= 2
        reasons.append("very low free disk")
    elif disk_gb < 15:
        score -= 1
        reasons.append("limited free disk")
    else:
        reasons.append("enough free disk")

    if score >= 7:
        tier = "strong"
    elif score >= 3:
        tier = "balanced"
    else:
        tier = "weak"

    profile = {
        "cpu_cores": cpu_cores,
        "ram_gb": round(ram_gb, 1),
        "nvidia_gpu": nvidia,
        "nvidia_vram_gb": vram_gb,
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
    print(f"- {tr(lang, 'gpu')}: {tr(lang, 'yes') if profile['nvidia_gpu'] else tr(lang, 'no')}")
    print(f"- {tr(lang, 'vram')}: {profile['nvidia_vram_gb']} GB")
    print(f"- {tr(lang, 'providers')}: {', '.join(profile['onnx_providers']) if profile['onnx_providers'] else 'n/a'}")
    print(f"- {tr(lang, 'recommended')}: {tier}")
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