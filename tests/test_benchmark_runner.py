"""Tests for benchmark runner module."""

from benchmarks.benchmark_runner import BenchmarkConfig, BenchmarkRunner


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig defaults."""

    def test_default_values(self):
        config = BenchmarkConfig()
        assert config.gpu_count == 1
        assert config.gpu_ids == "0"
        assert config.gpu_memory_utilization == 0.9
        assert config.input_len == 1024
        assert config.output_len == 128
        assert config.num_prompts == 100
        assert config.num_prompts_start == 1

    def test_custom_values(self):
        config = BenchmarkConfig(gpu_count=4, model="test-model", num_prompts=50)
        assert config.gpu_count == 4
        assert config.model == "test-model"
        assert config.num_prompts == 50


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_build_benchmark_command(self):
        config = BenchmarkConfig(model="meta-llama/Llama-3.1-8B")
        runner = BenchmarkRunner(config)
        cmd = runner.build_benchmark_command(10, "result.json")
        assert "--model" in cmd
        assert "meta-llama/Llama-3.1-8B" in cmd
        assert "--num-prompts" in cmd
        assert "10" in cmd

    def test_build_server_command(self):
        config = BenchmarkConfig(gpu_count=2, model="test-model")
        runner = BenchmarkRunner(config)
        cmd = runner.build_server_command()
        assert "vllm" in cmd
        assert "serve" in cmd
        assert "--tensor-parallel-size" in cmd
        assert "2" in cmd

    def test_generate_shell_script(self):
        config = BenchmarkConfig(
            model="test-model",
            experiment_name="test_exp",
            num_prompts_start=1,
            num_prompts_end=10,
        )
        runner = BenchmarkRunner(config)
        script = runner.generate_shell_script()
        assert "#!/bin/bash" in script
        assert "test-model" in script
        assert "test_exp" in script

    def test_get_max_model_len_default(self):
        config = BenchmarkConfig(input_len=1024, output_len=128)
        runner = BenchmarkRunner(config)
        assert runner.get_max_model_len() == 1024 + 128 + 128

    def test_get_max_model_len_explicit(self):
        config = BenchmarkConfig(max_model_len=4096)
        runner = BenchmarkRunner(config)
        assert runner.get_max_model_len() == 4096

    def test_build_docker_benchmark_command(self):
        config = BenchmarkConfig(
            model="test-model",
            input_len=512,
            output_len=64,
            vllm_server_url="http://vllm-server:8000",
        )
        runner = BenchmarkRunner(config)
        cmd = runner.build_docker_benchmark_command(
            num_prompts=5,
            experiment_dir="/root/output/test_exp",
        )
        assert "vllm bench serve" in cmd
        assert "--num-prompts 5" in cmd
        assert "test-model" in cmd
        assert "/root/output/test_exp" in cmd
