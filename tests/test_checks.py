"""Tests for system check modules."""

from unittest.mock import MagicMock, patch

from checks.docker_check import check_docker_daemon
from checks.gpu_detect import GpuPlatform, detect_gpu_platform
from checks.intel_check import check_intel, check_intel_driver
from checks.nvidia_check import check_nvidia_driver


class TestGpuDetection:
    """Tests for GPU platform detection."""

    @patch("checks.gpu_detect._check_amd_present", return_value=True)
    @patch("checks.gpu_detect._check_nvidia_present", return_value=False)
    @patch("checks.gpu_detect._check_intel_present", return_value=False)
    def test_detect_amd(self, mock_intel, mock_nvidia, mock_amd):
        platform, msg = detect_gpu_platform()
        assert platform == GpuPlatform.AMD
        assert "AMD" in msg

    @patch("checks.gpu_detect._check_amd_present", return_value=False)
    @patch("checks.gpu_detect._check_nvidia_present", return_value=True)
    @patch("checks.gpu_detect._check_intel_present", return_value=False)
    def test_detect_nvidia(self, mock_intel, mock_nvidia, mock_amd):
        platform, msg = detect_gpu_platform()
        assert platform == GpuPlatform.NVIDIA
        assert "NVIDIA" in msg

    @patch("checks.gpu_detect._check_amd_present", return_value=False)
    @patch("checks.gpu_detect._check_nvidia_present", return_value=False)
    @patch("checks.gpu_detect._check_intel_present", return_value=True)
    def test_detect_intel(self, mock_intel, mock_nvidia, mock_amd):
        platform, msg = detect_gpu_platform()
        assert platform == GpuPlatform.INTEL
        assert "Intel" in msg

    @patch("checks.gpu_detect._check_amd_present", return_value=False)
    @patch("checks.gpu_detect._check_nvidia_present", return_value=False)
    @patch("checks.gpu_detect._check_intel_present", return_value=False)
    def test_detect_none(self, mock_intel, mock_nvidia, mock_amd):
        platform, msg = detect_gpu_platform()
        assert platform == GpuPlatform.NONE
        assert "No GPU" in msg


class TestDockerChecks:
    """Tests for Docker check module."""

    @patch("subprocess.run")
    def test_docker_daemon_running(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        passed, msg = check_docker_daemon()
        assert passed is True
        assert "running" in msg

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_docker_not_installed(self, mock_run):
        passed, msg = check_docker_daemon()
        assert passed is False
        assert "not found" in msg


class TestNvidiaChecks:
    """Tests for NVIDIA check module."""

    @patch("subprocess.run")
    def test_nvidia_driver_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="535.129.03\n", stderr="")
        passed, msg = check_nvidia_driver()
        assert passed is True
        assert "535.129.03" in msg

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_nvidia_not_installed(self, mock_run):
        passed, msg = check_nvidia_driver()
        assert passed is False


class TestIntelChecks:
    """Tests for Intel GPU check module."""

    @patch("subprocess.run")
    def test_intel_driver_xe_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Module Size Used by\nxe 1234 0\n", stderr="")
        passed, msg = check_intel_driver()
        assert passed is True
        assert "xe" in msg.lower() or "intel" in msg.lower()

    @patch("subprocess.run")
    def test_intel_driver_i915_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Module Size Used by\ni915 5678 1\n", stderr="")
        passed, msg = check_intel_driver()
        assert passed is True

    @patch("subprocess.run")
    def test_intel_driver_not_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Module Size Used by\namdgpu 999 0\n", stderr="")
        passed, msg = check_intel_driver()
        assert passed is False

    @patch("checks.intel_check.check_intel_driver", return_value=(True, "Intel Xe GPU driver loaded"))
    @patch("checks.intel_check.check_intel_dri_devices", return_value=(True, ["/dev/dri/renderD128"]))
    @patch("checks.intel_check.get_intel_gpu_count", return_value=1)
    def test_check_intel_success(self, mock_count, mock_devices, mock_driver):
        result = check_intel()
        assert result["success"] is True
        assert result["gpu_count"] == 1
        assert len(result["checks"]) == 2

    @patch("checks.intel_check.check_intel_driver", return_value=(False, "not found"))
    @patch("checks.intel_check.check_intel_dri_devices", return_value=(False, []))
    @patch("checks.intel_check.get_intel_gpu_count", return_value=0)
    def test_check_intel_failure(self, mock_count, mock_devices, mock_driver):
        result = check_intel()
        assert result["success"] is False
        assert result["gpu_count"] == 0
