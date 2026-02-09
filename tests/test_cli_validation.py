"""Tests for CLI input validation."""

from unittest.mock import patch
from click.testing import CliRunner
from main import cli


class TestBenchmarkValidation:
    """Tests for benchmark command validation."""

    def test_gpu_memory_utilization_too_high(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            'run', 'benchmark',
            '--model', 'test-model',
            '--experiment-name', 'test',
            '--gpu-memory-utilization', '1.5',
        ])
        assert result.exit_code != 0
        assert "between 0.0 and 1.0" in result.output

    def test_gpu_memory_utilization_negative(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            'run', 'benchmark',
            '--model', 'test-model',
            '--experiment-name', 'test',
            '--gpu-memory-utilization', '-0.1',
        ])
        assert result.exit_code != 0
        assert "between 0.0 and 1.0" in result.output

    @patch("main.DockerManager")
    def test_gpu_ids_valid_format(self, mock_docker_cls):
        """gpu-ids should accept comma-separated integers."""
        mock_docker_cls.return_value.is_running.return_value = False
        runner = CliRunner()
        result = runner.invoke(cli, [
            'run', 'benchmark',
            '--model', 'test-model',
            '--experiment-name', 'test',
            '--gpu-ids', '0,1,2',
        ])
        # Should get past validation (fail at container check instead)
        assert "Invalid GPU IDs" not in result.output

    def test_gpu_ids_invalid_format(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            'run', 'benchmark',
            '--model', 'test-model',
            '--experiment-name', 'test',
            '--gpu-ids', 'abc',
        ])
        assert result.exit_code != 0
        assert "Invalid GPU IDs" in result.output

    def test_num_prompts_start_greater_than_end(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            'run', 'benchmark',
            '--model', 'test-model',
            '--experiment-name', 'test',
            '--num-prompts-start', '100',
            '--num-prompts', '10',
        ])
        assert result.exit_code != 0
        assert "start" in result.output.lower() and "end" in result.output.lower()
