"""Intel GPU (Arc/Xe) validation module."""

import subprocess
from pathlib import Path
from typing import Tuple


def check_intel_driver() -> Tuple[bool, str]:
    """Check if Intel GPU kernel module is loaded."""
    try:
        result = subprocess.run(
            ["lsmod"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            if "xe" in result.stdout.split():
                return True, "Intel Xe GPU driver loaded"
            if "i915" in result.stdout.split():
                return True, "Intel i915 GPU driver loaded"
        return False, "Intel GPU kernel module (xe or i915) not found in lsmod"
    except FileNotFoundError:
        return False, "lsmod not found"
    except subprocess.TimeoutExpired:
        return False, "Timeout checking kernel modules"
    except Exception as e:
        return False, f"Error checking Intel driver: {e}"


def check_intel_dri_devices() -> Tuple[bool, list]:
    """Check if Intel DRI render devices are present."""
    try:
        render_devices = sorted(Path("/dev/dri").glob("renderD*"))
        if render_devices:
            return True, [str(d) for d in render_devices]
        return False, []
    except Exception:
        return False, []


def get_intel_gpu_count() -> int:
    """Get number of Intel GPU render nodes."""
    try:
        render_devices = list(Path("/dev/dri").glob("renderD*"))
        return len(render_devices)
    except Exception:
        return 0


def check_intel() -> dict:
    """
    Run all Intel GPU checks.

    Returns:
        dict with check results
    """
    checks = []
    all_passed = True

    passed, message = check_intel_driver()
    checks.append({"name": "Intel GPU Driver", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    passed, devices = check_intel_dri_devices()
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

    gpu_count = get_intel_gpu_count()

    return {
        "success": all_passed,
        "checks": checks,
        "gpu_count": gpu_count,
    }
