"""Intel GPU (Arc / Xe) validation module."""

import subprocess
from pathlib import Path
from typing import Tuple


def check_gpu_driver() -> Tuple[bool, str]:
    """Check whether the Intel GPU kernel module is loaded."""
    try:
        result = subprocess.run(
            ["lsmod"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            modules = result.stdout.split()
            if "xe" in modules:
                return True, "Intel Xe GPU driver loaded"
            if "i915" in modules:
                return True, "Intel i915 GPU driver loaded"
        return False, "Intel GPU kernel module (xe or i915) not found in lsmod"
    except FileNotFoundError:
        return False, "lsmod not found"
    except subprocess.TimeoutExpired:
        return False, "Timeout checking kernel modules"
    except Exception as e:
        return False, f"Error checking Intel driver: {e}"


def check_dri_devices() -> Tuple[bool, list]:
    """List /dev/dri/renderD* devices."""
    try:
        render_devices = sorted(Path("/dev/dri").glob("renderD*"))
        if render_devices:
            return True, [str(d) for d in render_devices]
        return False, []
    except Exception:
        return False, []


def get_gpu_count() -> int:
    """Count /dev/dri/renderD* nodes."""
    try:
        return len(list(Path("/dev/dri").glob("renderD*")))
    except Exception:
        return 0


def check_gpu() -> dict:
    """Run all Intel GPU checks and return a structured result."""
    checks = []
    all_passed = True

    passed, message = check_gpu_driver()
    checks.append({"name": "Intel GPU Driver", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    passed, devices = check_dri_devices()
    if passed:
        message = f"Detected {len(devices)} DRI render device(s)"
        for dev in devices[:4]:
            message += f"\n  - {dev}"
        if len(devices) > 4:
            message += f"\n  ... and {len(devices) - 4} more"
    else:
        message = "No /dev/dri/renderD* devices found"
    checks.append({"name": "DRI Render Devices", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    return {
        "success": all_passed,
        "checks": checks,
        "gpu_count": get_gpu_count(),
    }
