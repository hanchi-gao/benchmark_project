"""ROCm installation validation module."""

import subprocess
from pathlib import Path
from typing import Tuple, Optional


def check_rocm_path() -> Tuple[bool, str]:
    """Check if /opt/rocm exists."""
    rocm_path = Path("/opt/rocm")
    if rocm_path.exists():
        # Try to find version
        version_file = rocm_path / ".info" / "version"
        if version_file.exists():
            try:
                version = version_file.read_text().strip()
                return True, f"/opt/rocm exists (version: {version})"
            except Exception:
                pass
        return True, "/opt/rocm exists"
    return False, "/opt/rocm not found - ROCm may not be installed"


def check_rocm_smi_version() -> Tuple[bool, Optional[str]]:
    """Check rocm-smi version."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
        return False, None
    except FileNotFoundError:
        return False, None
    except subprocess.TimeoutExpired:
        return False, None
    except Exception:
        return False, None


def check_gpu_detection() -> Tuple[bool, list]:
    """Check if GPUs are detected using rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            # Parse output to find GPU names
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                if "GPU" in line and "Card" in line:
                    gpus.append(line.strip())
                elif ":" in line and ("Radeon" in line or "Instinct" in line or "AMD" in line):
                    gpus.append(line.strip())
            if gpus:
                return True, gpus
            # If no specific GPUs found but command succeeded, try basic detection
            return True, ["GPU detected (see rocm-smi for details)"]
        return False, []
    except FileNotFoundError:
        return False, []
    except subprocess.TimeoutExpired:
        return False, []
    except Exception:
        return False, []


def get_gpu_count() -> int:
    """Get the number of AMD GPUs detected."""
    try:
        result = subprocess.run(
            ["rocm-smi", "-i"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            # Count GPU entries
            count = result.stdout.count("GPU[")
            if count > 0:
                return count
            # Alternative counting
            lines = result.stdout.strip().split('\n')
            gpu_lines = [l for l in lines if l.strip().startswith("GPU")]
            return len(gpu_lines)
        return 0
    except Exception:
        return 0


def check_rocm() -> dict:
    """
    Run all ROCm checks.

    Returns:
        dict with check results:
        {
            "success": bool,
            "checks": [
                {"name": str, "passed": bool, "message": str},
                ...
            ],
            "gpu_count": int
        }
    """
    checks = []
    all_passed = True

    # Check /opt/rocm
    passed, message = check_rocm_path()
    checks.append({"name": "ROCm Installation Path", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    # Check rocm-smi version
    passed, version = check_rocm_smi_version()
    if passed:
        message = f"rocm-smi version: {version}"
    else:
        message = "rocm-smi not found or not working"
    checks.append({"name": "ROCm SMI Tool", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    # Check GPU detection
    passed, gpus = check_gpu_detection()
    if passed:
        message = f"Detected GPUs: {len(gpus)}"
        for gpu in gpus[:4]:  # Show first 4 GPUs
            message += f"\n  - {gpu}"
        if len(gpus) > 4:
            message += f"\n  ... and {len(gpus) - 4} more"
    else:
        message = "No AMD GPUs detected"
    checks.append({"name": "GPU Detection", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    gpu_count = get_gpu_count()

    return {
        "success": all_passed,
        "checks": checks,
        "gpu_count": gpu_count
    }
