"""AMD GPU Driver validation module."""

import subprocess
from pathlib import Path
from typing import List, Tuple


def check_amdgpu_module() -> Tuple[bool, str]:
    """Check if amdgpu kernel module is loaded."""
    try:
        result = subprocess.run(
            ["lsmod"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "amdgpu" in result.stdout:
            return True, "amdgpu kernel module is loaded"
        return False, "amdgpu kernel module is not loaded"
    except subprocess.TimeoutExpired:
        return False, "Timeout checking kernel modules"
    except FileNotFoundError:
        return False, "lsmod command not found"
    except Exception as e:
        return False, f"Error checking kernel modules: {e}"


def check_kfd_device() -> Tuple[bool, str]:
    """Check if /dev/kfd exists (AMD KFD - Kernel Fusion Driver)."""
    kfd_path = Path("/dev/kfd")
    if kfd_path.exists():
        return True, "/dev/kfd device exists"
    return False, "/dev/kfd device not found - ROCm may not work properly"


def check_render_devices() -> Tuple[bool, List[str]]:
    """Check for render devices in /dev/dri/."""
    dri_path = Path("/dev/dri")
    if not dri_path.exists():
        return False, []

    render_devices = list(dri_path.glob("renderD*"))
    if render_devices:
        device_names = [d.name for d in render_devices]
        return True, device_names
    return False, []


def check_amd_driver() -> dict:
    """
    Run all AMD driver checks.

    Returns:
        dict with check results:
        {
            "success": bool,
            "checks": [
                {"name": str, "passed": bool, "message": str},
                ...
            ]
        }
    """
    checks = []
    all_passed = True

    # Check amdgpu module
    passed, message = check_amdgpu_module()
    checks.append({"name": "AMD GPU Kernel Module", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    # Check /dev/kfd
    passed, message = check_kfd_device()
    checks.append({"name": "KFD Device", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    # Check render devices
    passed, devices = check_render_devices()
    if passed:
        message = f"Found render devices: {', '.join(devices)}"
    else:
        message = "No render devices found in /dev/dri/"
    checks.append({"name": "Render Devices", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    return {
        "success": all_passed,
        "checks": checks
    }
