"""Benchmark runner with customizable GPU configuration."""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    gpu_count: int = 1
    gpu_ids: str = "0"
    model: str = "meta-llama/Llama-3.1-8B"
    gpu_memory_utilization: float = 0.9
    input_len: int = 1024
    output_len: int = 128
    num_prompts: int = 100
    num_prompts_start: int = 1
    num_prompts_end: Optional[int] = None
    experiment_name: str = "benchmark"
    max_model_len: Optional[int] = None
    enforce_eager: bool = True
    vllm_server_url: str = "http://localhost:8001"


class BenchmarkRunner:
    """Runs vLLM benchmarks with customizable GPU configuration."""

    def __init__(self, config: BenchmarkConfig, output_dir: Optional[Path] = None):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
            output_dir: Directory to store results
        """
        self.config = config
        self.output_dir = output_dir or Path(__file__).parent.parent / "output"
        self.experiment_dir = self.output_dir / config.experiment_name

    def setup_environment(self) -> dict:
        """Set up environment variables for GPU selection."""
        env = os.environ.copy()
        # AMD ROCm GPU selection
        env["HIP_VISIBLE_DEVICES"] = self.config.gpu_ids
        # Also set CUDA_VISIBLE_DEVICES for compatibility
        env["CUDA_VISIBLE_DEVICES"] = self.config.gpu_ids
        return env

    def get_max_model_len(self) -> int:
        """Calculate max model length."""
        if self.config.max_model_len:
            return self.config.max_model_len
        # Default: input + output + some buffer
        return self.config.input_len + self.config.output_len + 128

    def build_server_command(self) -> List[str]:
        """Build vLLM serve command."""
        cmd = [
            "vllm", "serve", self.config.model,
            "--tensor-parallel-size", str(self.config.gpu_count),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-model-len", str(self.get_max_model_len()),
        ]
        if self.config.enforce_eager:
            cmd.append("--enforce-eager")
        return cmd

    def build_benchmark_command(self, num_prompts: int, result_filename: str) -> List[str]:
        """Build vLLM bench serve command."""
        cmd = [
            "vllm", "bench", "serve",
            "--model", self.config.model,
            "--backend", "openai",
            "--endpoint", "/v1/completions",
            "--base-url", self.config.vllm_server_url,
            "--dataset-name", "random",
            "--random-input-len", str(self.config.input_len),
            "--random-output-len", str(self.config.output_len),
            "--num-prompts", str(num_prompts),
            "--ignore-eos",
            "--save-result",
            "--result-filename", result_filename,
        ]
        return cmd

    def build_docker_benchmark_command(self, num_prompts: int, experiment_dir: str) -> str:
        """
        Build a shell command string for running a benchmark inside a Docker container.

        Args:
            num_prompts: Number of prompts to test
            experiment_dir: Path inside the container for results

        Returns:
            Shell command string ready for docker exec
        """
        return (
            f'cd {experiment_dir} && '
            f'vllm bench serve '
            f'--model "{self.config.model}" '
            f'--backend openai '
            f'--endpoint /v1/completions '
            f'--base-url {self.config.vllm_server_url} '
            f'--dataset-name random '
            f'--random-input-len {self.config.input_len} '
            f'--random-output-len {self.config.output_len} '
            f'--num-prompts {num_prompts} '
            f'--ignore-eos '
            f'--save-result '
            f'--result-filename "np{num_prompts}_$(date +%Y%m%d_%H%M%S).json"'
        )

    def wait_for_server(self, timeout: int = 300) -> bool:
        """Wait for vLLM server to be ready."""
        import urllib.error
        import urllib.request

        health_url = f"{self.config.vllm_server_url}/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                req = urllib.request.urlopen(health_url, timeout=5)
                if req.status == 200:
                    return True
            except (urllib.error.URLError, urllib.error.HTTPError):
                pass
            time.sleep(2)

        return False

    def check_server_model(self) -> Tuple[bool, Optional[str]]:
        """Check which model is loaded on the server."""
        import urllib.request

        try:
            models_url = f"{self.config.vllm_server_url}/v1/models"
            req = urllib.request.urlopen(models_url, timeout=10)
            data = json.loads(req.read().decode())
            if data.get("data"):
                model_id = data["data"][0].get("id")
                return model_id == self.config.model, model_id
            return False, None
        except Exception:
            return False, None

    def run_single_benchmark(self, num_prompts: int) -> Tuple[bool, Optional[str]]:
        """
        Run a single benchmark with specified num_prompts.

        Args:
            num_prompts: Number of prompts to test

        Returns:
            Tuple of (success, result_file_path)
        """
        # Ensure experiment directory exists
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"np{num_prompts}_{timestamp}.json"
        result_path = self.experiment_dir / result_filename
        temp_result = f"temp_result_{num_prompts}.json"

        env = self.setup_environment()
        cmd = self.build_benchmark_command(num_prompts, temp_result)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.experiment_dir),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes per benchmark
                env=env
            )

            temp_path = self.experiment_dir / temp_result
            if result.returncode == 0 and temp_path.exists():
                # Move temp file to final location
                temp_path.rename(result_path)
                return True, str(result_path)
            else:
                # Clean up temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()
                return False, None

        except subprocess.TimeoutExpired:
            return False, None
        except Exception:
            return False, None

    def run_benchmark_suite(self, callback=None) -> dict:
        """
        Run complete benchmark suite with varying num_prompts.

        Args:
            callback: Optional callback function(num_prompts, success, result_file)

        Returns:
            dict with results summary
        """
        start = self.config.num_prompts_start
        end = self.config.num_prompts_end or self.config.num_prompts

        total = end - start + 1
        completed = 0
        failed = 0
        results = []

        for num_prompts in range(start, end + 1):
            success, result_file = self.run_single_benchmark(num_prompts)

            if success:
                completed += 1
                results.append(result_file)
            else:
                failed += 1

            if callback:
                callback(num_prompts, success, result_file)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "experiment_dir": str(self.experiment_dir),
            "results": results
        }

    def generate_shell_script(self) -> str:
        """Generate a shell script for running the benchmark."""
        script = f"""#!/bin/bash
# Auto-generated benchmark script
# Experiment: {self.config.experiment_name}
# Generated: {datetime.now().isoformat()}

set -e

# GPU Configuration
export HIP_VISIBLE_DEVICES="{self.config.gpu_ids}"

# Results directory
RESULTS_DIR="/root/output/{self.config.experiment_name}"
mkdir -p "$RESULTS_DIR"

# Benchmark parameters
MODEL="{self.config.model}"
TP_SIZE={self.config.gpu_count}
GPU_MEM_UTIL={self.config.gpu_memory_utilization}
INPUT_LEN={self.config.input_len}
OUTPUT_LEN={self.config.output_len}
MAX_MODEL_LEN={self.get_max_model_len()}
VLLM_SERVER_URL="{self.config.vllm_server_url}"

NUM_PROMPTS_START={self.config.num_prompts_start}
NUM_PROMPTS_END={self.config.num_prompts_end or self.config.num_prompts}

echo "Starting benchmark suite..."
echo "  Model: $MODEL"
echo "  TP Size: $TP_SIZE"
echo "  GPU Memory Utilization: $GPU_MEM_UTIL"
echo "  Input Length: $INPUT_LEN"
echo "  Output Length: $OUTPUT_LEN"
echo "  Prompts Range: $NUM_PROMPTS_START - $NUM_PROMPTS_END"

# Run benchmarks
for NUM_PROMPTS in $(seq $NUM_PROMPTS_START $NUM_PROMPTS_END); do
    echo "[$(date +%H:%M:%S)] Testing num_prompts: $NUM_PROMPTS"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_FILE="$RESULTS_DIR/np${{NUM_PROMPTS}}_${{TIMESTAMP}}.json"

    vllm bench serve \\
        --model "$MODEL" \\
        --backend openai \\
        --endpoint /v1/completions \\
        --base-url "$VLLM_SERVER_URL" \\
        --dataset-name random \\
        --random-input-len "$INPUT_LEN" \\
        --random-output-len "$OUTPUT_LEN" \\
        --num-prompts "$NUM_PROMPTS" \\
        --ignore-eos \\
        --save-result \\
        --result-filename "$RESULT_FILE"

    if [ $? -eq 0 ]; then
        echo "  OK: $RESULT_FILE"
    else
        echo "  FAILED"
    fi
done

echo ""
echo "Benchmark suite completed!"
echo "Results saved to: $RESULTS_DIR"
"""
        return script
