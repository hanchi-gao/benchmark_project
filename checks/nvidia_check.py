"""NVIDIA GPU validation module."""

import subprocess
from typing import Tuple


def check_nvidia_driver() -> Tuple[bool, str]:
    """Check if NVIDIA driver is loaded."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip().split("\n")[0]
            return True, f"NVIDIA driver version: {version}"
        return False, "nvidia-smi failed"
    except FileNotFoundError:
        return False, "nvidia-smi not found - NVIDIA driver may not be installed"
    except subprocess.TimeoutExpired:
        return False, "Timeout checking NVIDIA driver"
    except Exception as e:
        return False, f"Error checking NVIDIA driver: {e}"


def check_nvidia_gpu_detection() -> Tuple[bool, list]:
    """Check if NVIDIA GPUs are detected."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            gpus = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            if gpus:
                return True, gpus
        return False, []
    except Exception:
        return False, []


def get_nvidia_gpu_count() -> int:
    """Get the number of NVIDIA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().split("\n") if l.strip()])
        return 0
    except Exception:
        return 0


def check_nvidia() -> dict:
    """
    Run all NVIDIA GPU checks.

    Returns:
        dict with check results
    """
    checks = []
    all_passed = True

    passed, message = check_nvidia_driver()
    checks.append({"name": "NVIDIA Driver", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    passed, gpus = check_nvidia_gpu_detection()
    if passed:
        message = f"Detected GPUs: {len(gpus)}"
        for gpu in gpus[:4]:
            message += f"\n  - {gpu}"
        if len(gpus) > 4:
            message += f"\n  ... and {len(gpus) - 4} more"
    else:
        message = "No NVIDIA GPUs detected"
    checks.append({"name": "GPU Detection", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    gpu_count = get_nvidia_gpu_count()

    return {
        "success": all_passed,
        "checks": checks,
        "gpu_count": gpu_count,
    }
