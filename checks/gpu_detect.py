"""GPU platform detection module (CUDA-only branch)."""

import subprocess
from enum import Enum
from typing import Tuple


class GpuPlatform(Enum):
    NVIDIA = "nvidia"
    NONE = "none"


def detect_gpu_platform() -> Tuple[GpuPlatform, str]:
    """Auto-detect whether an NVIDIA GPU is present on this system."""
    if _check_nvidia_present():
        return GpuPlatform.NVIDIA, "NVIDIA GPU detected"
    return GpuPlatform.NONE, "No GPU detected"


def _check_nvidia_present() -> bool:
    """Check if NVIDIA GPU hardware is present via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
