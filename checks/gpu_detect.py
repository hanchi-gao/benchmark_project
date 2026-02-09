"""GPU platform detection module."""

import subprocess
from enum import Enum
from typing import Tuple


class GpuPlatform(Enum):
    AMD = "amd"
    NVIDIA = "nvidia"
    NONE = "none"


def detect_gpu_platform() -> Tuple[GpuPlatform, str]:
    """
    Auto-detect the GPU platform available on this system.

    Returns:
        Tuple of (GpuPlatform, description string)
    """
    amd_found = _check_amd_present()
    nvidia_found = _check_nvidia_present()

    if amd_found and nvidia_found:
        return GpuPlatform.AMD, "Both AMD and NVIDIA GPUs detected (using AMD)"
    elif amd_found:
        return GpuPlatform.AMD, "AMD GPU detected"
    elif nvidia_found:
        return GpuPlatform.NVIDIA, "NVIDIA GPU detected"
    else:
        return GpuPlatform.NONE, "No GPU detected"


def _check_amd_present() -> bool:
    """Check if AMD GPU hardware is present."""
    try:
        result = subprocess.run(
            ["lsmod"], capture_output=True, text=True, timeout=10
        )
        if "amdgpu" in result.stdout:
            return True
    except Exception:
        pass

    try:
        from pathlib import Path
        if Path("/dev/kfd").exists():
            return True
    except Exception:
        pass

    return False


def _check_nvidia_present() -> bool:
    """Check if NVIDIA GPU hardware is present."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
