"""Tests for system check modules (Intel-only)."""

from unittest.mock import MagicMock, patch

from checks.docker_check import check_docker_daemon
from checks.gpu_check import check_gpu, check_gpu_driver


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


class TestGpuChecks:
    """Tests for Intel GPU check module."""

    @patch("subprocess.run")
    def test_driver_xe_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Module Size Used by\nxe 1234 0\n", stderr="")
        passed, msg = check_gpu_driver()
        assert passed is True
        assert "xe" in msg.lower() or "intel" in msg.lower()

    @patch("subprocess.run")
    def test_driver_i915_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Module Size Used by\ni915 5678 1\n", stderr="")
        passed, msg = check_gpu_driver()
        assert passed is True

    @patch("subprocess.run")
    def test_driver_not_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Module Size Used by\namdgpu 999 0\n", stderr="")
        passed, msg = check_gpu_driver()
        assert passed is False

    @patch("checks.gpu_check.check_gpu_driver", return_value=(True, "Intel Xe GPU driver loaded"))
    @patch("checks.gpu_check.check_dri_devices", return_value=(True, ["/dev/dri/renderD128"]))
    @patch("checks.gpu_check.get_gpu_count", return_value=1)
    def test_check_gpu_success(self, mock_count, mock_devices, mock_driver):
        result = check_gpu()
        assert result["success"] is True
        assert result["gpu_count"] == 1
        assert len(result["checks"]) == 2

    @patch("checks.gpu_check.check_gpu_driver", return_value=(False, "not found"))
    @patch("checks.gpu_check.check_dri_devices", return_value=(False, []))
    @patch("checks.gpu_check.get_gpu_count", return_value=0)
    def test_check_gpu_failure(self, mock_count, mock_devices, mock_driver):
        result = check_gpu()
        assert result["success"] is False
        assert result["gpu_count"] == 0
