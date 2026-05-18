#!/usr/bin/env python3
"""
GPU Detection & Diagnostics Tool.

Shows all available GPUs with live VRAM usage, detected ONNX providers,
machine tier classification, and environment variable reference.

Usage:
    python check_gpus.py              # Show full GPU diagnostics
    DEEPFAKE_GPU_ID=0 python app.py   # Force use GPU 0
"""

import sys

# ANSI color helpers
class C:
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    CYAN  = "\033[96m"
    YELLOW= "\033[93m"
    RED   = "\033[91m"
    DIM   = "\033[90m"
    RESET = "\033[0m"


def _bar(used: float, total: float, width: int = 20) -> str:
    """Draw a simple VRAM usage bar."""
    if total <= 0:
        return "[" + "?" * width + "]"
    pct = min(used / total, 1.0)
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:.0f}%"


def main() -> int:
    # Import here so any install issue is caught cleanly
    from runtime_utils import get_gpu_summary

    summary = get_gpu_summary()

    print()
    print(f"{C.BOLD}{'='*65}{C.RESET}")
    print(f"{C.BOLD}  🖥️  GPU DIAGNOSTICS — DeepFake RealTime{C.RESET}")
    print(f"{C.BOLD}{'='*65}{C.RESET}")

    # ── NVIDIA GPUs ──────────────────────────────────────────────
    if summary.nvidia_gpus:
        print(f"\n{C.GREEN}▸ NVIDIA GPUs ({len(summary.nvidia_gpus)} detected){C.RESET}")
        for g in summary.nvidia_gpus:
            selected = " ← selected" if summary.selected_gpu and g.index == summary.selected_gpu.index else ""
            print(f"  GPU {g.index}: {C.BOLD}{g.name}{C.RESET}{C.CYAN}{selected}{C.RESET}")
            print(f"    VRAM:  {_bar(g.used_vram_mb, g.total_vram_mb)}  "
                  f"{g.free_vram_gb:.1f} free / {g.total_vram_gb:.1f} GB total")
            if g.utilization_pct > 0:
                print(f"    Util:  {g.utilization_pct:.0f}%")
    else:
        print(f"\n{C.DIM}▸ NVIDIA: not detected{C.RESET}")

    # ── AMD GPUs ─────────────────────────────────────────────────
    if summary.amd_gpus:
        print(f"\n{C.GREEN}▸ AMD GPUs ({len(summary.amd_gpus)} detected — ROCm){C.RESET}")
        for g in summary.amd_gpus:
            selected = " ← selected" if summary.selected_gpu and summary.selected_gpu.vendor == "amd" and g.index == summary.selected_gpu.index else ""
            print(f"  GPU {g.index}: {C.BOLD}{g.name}{C.RESET}{C.CYAN}{selected}{C.RESET}")
            if g.total_vram_mb > 0:
                print(f"    VRAM:  {_bar(g.used_vram_mb, g.total_vram_mb)}  "
                      f"{g.free_vram_gb:.1f} free / {g.total_vram_gb:.1f} GB total")
    else:
        print(f"\n{C.DIM}▸ AMD (ROCm): not detected{C.RESET}")

    # ── Intel / Apple ────────────────────────────────────────────
    if summary.intel_detected:
        print(f"\n{C.GREEN}▸ Intel GPU: detected{C.RESET}")
    if summary.apple_silicon:
        print(f"\n{C.GREEN}▸ Apple Silicon: detected (CoreML){C.RESET}")

    # ── ONNX Runtime Providers ───────────────────────────────────
    print(f"\n{C.BOLD}{'─'*65}{C.RESET}")
    print(f"{C.BOLD}  ONNX Runtime Providers{C.RESET}")
    print(f"{C.BOLD}{'─'*65}{C.RESET}")
    print(f"  Available : {', '.join(summary.onnx_providers_available) or 'none'}")
    print(f"  {C.CYAN}Selected  : {', '.join(summary.onnx_providers_selected)}{C.RESET}")

    primary = summary.onnx_providers_selected[0] if summary.onnx_providers_selected else "CPU"
    if "CUDA" in primary or "Tensorrt" in primary:
        print(f"  {C.GREEN}⚡ GPU acceleration active (NVIDIA){C.RESET}")
    elif "ROCM" in primary:
        print(f"  {C.GREEN}⚡ GPU acceleration active (AMD ROCm){C.RESET}")
    elif "CoreML" in primary:
        print(f"  {C.GREEN}⚡ GPU acceleration active (Apple CoreML){C.RESET}")
    elif "Dml" in primary:
        print(f"  {C.GREEN}⚡ GPU acceleration active (DirectML){C.RESET}")
    else:
        print(f"  {C.YELLOW}⚠️  Running on CPU — consider installing onnxruntime-gpu{C.RESET}")

    # ── No GPU warning ───────────────────────────────────────────
    if not summary.has_any_gpu:
        print(f"\n{C.RED}{'='*65}{C.RESET}")
        print(f"{C.RED}  ⚠️  No GPU detected — all processing will use CPU{C.RESET}")
        print(f"{C.RED}{'='*65}{C.RESET}")

    # ── Environment variables reference ──────────────────────────
    print(f"\n{C.BOLD}{'─'*65}{C.RESET}")
    print(f"{C.BOLD}  Environment Variables{C.RESET}")
    print(f"{C.BOLD}{'─'*65}{C.RESET}")
    print(f"  {C.CYAN}DEEPFAKE_GPU_ID=0{C.RESET}          Force specific GPU by index")
    print(f"  {C.CYAN}DEEPFAKE_DISABLE_GPU=1{C.RESET}     Disable GPU, use CPU only")
    print(f"  {C.CYAN}DEEPFAKE_ORT_PROVIDERS=...{C.RESET} Override ONNX providers (comma-separated)")
    print(f"  {C.CYAN}CUDA_VISIBLE_DEVICES=0{C.RESET}     NVIDIA: limit visible GPUs")
    print(f"  {C.CYAN}HIP_VISIBLE_DEVICES=0{C.RESET}      AMD: limit visible GPUs")

    print(f"\n{C.BOLD}{'='*65}{C.RESET}\n")
    return 0 if summary.has_any_gpu else 1


if __name__ == "__main__":
    sys.exit(main())
