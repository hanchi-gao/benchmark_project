"""Tests for system check modules."""

from unittest.mock import MagicMock, patch

from checks.docker_check import check_docker_daemon
from checks.gpu_detect import GpuPlatform, detect_gpu_platform
from checks.nvidia_check import check_nvidia_driver


class TestGpuDetection:
    """Tests for GPU platform detection."""

    @patch("checks.gpu_detect._check_amd_present", return_value=True)
    @patch("checks.gpu_detect._check_nvidia_present", return_value=False)
    def test_detect_amd(self, mock_nvidia, mock_amd):
        platform, msg = detect_gpu_platform()
        assert platform == GpuPlatform.AMD
        assert "AMD" in msg

    @patch("checks.gpu_detect._check_amd_present", return_value=False)
    @patch("checks.gpu_detect._check_nvidia_present", return_value=True)
    def test_detect_nvidia(self, mock_nvidia, mock_amd):
        platform, msg = detect_gpu_platform()
        assert platform == GpuPlatform.NVIDIA
        assert "NVIDIA" in msg

    @patch("checks.gpu_detect._check_amd_present", return_value=False)
    @patch("checks.gpu_detect._check_nvidia_present", return_value=False)
    def test_detect_none(self, mock_nvidia, mock_amd):
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
