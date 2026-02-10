"""Docker readiness validation module."""

import subprocess
from typing import List, Optional, Tuple


def check_docker_daemon() -> Tuple[bool, str]:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, "Docker daemon is running"
        return False, f"Docker daemon not responding: {result.stderr.strip()}"
    except FileNotFoundError:
        return False, "Docker command not found - Docker may not be installed"
    except subprocess.TimeoutExpired:
        return False, "Timeout connecting to Docker daemon"
    except Exception as e:
        return False, f"Error checking Docker: {e}"


def check_docker_compose() -> Tuple[bool, str]:
    """Check if docker compose is available."""
    # Try 'docker compose' (v2) first
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Docker Compose available: {version}"
    except Exception:
        pass

    # Try 'docker-compose' (v1) as fallback
    try:
        result = subprocess.run(
            ["docker-compose", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Docker Compose (v1) available: {version}"
    except Exception:
        pass

    return False, "Docker Compose not found"


def check_vllm_image(image_name: str = "vllm-rocm71:latest") -> Tuple[bool, Optional[str]]:
    """Check if vLLM Docker image is available."""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", image_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and image_name in result.stdout:
            return True, image_name

        # Try to find any vllm image
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            images = result.stdout.strip().split('\n')
            vllm_images = [img for img in images if "vllm" in img.lower()]
            if vllm_images:
                return True, vllm_images[0]

        return False, None
    except Exception:
        return False, None


def list_vllm_images() -> List[str]:
    """List all vLLM-related Docker images."""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            images = result.stdout.strip().split('\n')
            vllm_images = [img for img in images if "vllm" in img.lower() and img != "<none>:<none>"]
            return vllm_images
        return []
    except Exception:
        return []


def check_docker(image_name: str = "vllm-rocm71:latest") -> dict:
    """
    Run all Docker checks.

    Args:
        image_name: Expected vLLM Docker image name

    Returns:
        dict with check results:
        {
            "success": bool,
            "checks": [
                {"name": str, "passed": bool, "message": str},
                ...
            ],
            "available_images": list
        }
    """
    checks = []
    all_passed = True

    # Check Docker daemon
    passed, message = check_docker_daemon()
    checks.append({"name": "Docker Daemon", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    # Check Docker Compose
    passed, message = check_docker_compose()
    checks.append({"name": "Docker Compose", "passed": passed, "message": message})
    if not passed:
        all_passed = False

    # Check vLLM image
    passed, found_image = check_vllm_image(image_name)
    if passed:
        message = f"vLLM image found: {found_image}"
    else:
        message = f"vLLM image '{image_name}' not found"
    checks.append({"name": "vLLM Docker Image", "passed": passed, "message": message})
    # Don't fail overall check if image is missing - user can pull it

    available_images = list_vllm_images()

    return {
        "success": all_passed,
        "checks": checks,
        "available_images": available_images
    }
