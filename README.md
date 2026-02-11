# vLLM Benchmark Tool

A unified benchmarking and visualization tool for [vLLM](https://github.com/vllm-project/vllm) on GPUs.

## Features

- **Docker-based**: Run vLLM in containers with interactive image selection
- **Interactive CLI**: Simple commands for complete benchmarking workflow
- **Real-time Output**: Stream benchmark progress as it runs
- **Web Dashboard**: Visualize and compare benchmark results across experiments
- **Config Recording**: Automatically saves GPU info, vLLM version, and test parameters

## Installation

### Prerequisites

- **GPU** (one of):
  - AMD GPU with [ROCm](https://rocm.docs.amd.com/) support
  - NVIDIA GPU with [CUDA](https://developer.nvidia.com/cuda-toolkit) drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker with GPU access configured
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

> **No GPU?** You can still use the Web UI to view existing benchmark results.

### Setup

```bash
git clone <repository-url>
cd vllm-benchmark
uv sync  # or: pip install -e .
```

## Quick Start

### Two-Terminal Workflow

**Terminal 1: Host machine**
```bash
# Step 1: Check system readiness
python3 main.py check

# Step 2: Start Docker containers (interactive image selection)
python3 main.py docker start

# Step 5: Run benchmark (after vLLM server is ready)
python3 main.py run benchmark \
    --model meta-llama/Llama-3.1-8B \
    --experiment-name my_test

# Step 6: View results
uv run python3 main.py webui
# Open http://localhost:8050
```

**Terminal 2: Inside container**
```bash
# Step 3: Enter the vLLM server container
docker exec -it vllm-server bash

# Step 4: Start vLLM server
vllm serve meta-llama/Llama-3.1-8B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9

# Wait for "Uvicorn running on http://0.0.0.0:8000" before running benchmark
```

### Stop Environment

```bash
python3 main.py docker stop
```

## CLI Reference

### System Check

```bash
python3 main.py check [--verbose]
```

### Docker Commands

```bash
python3 main.py docker start [--image IMAGE]
python3 main.py docker stop [--volumes]
python3 main.py docker status
python3 main.py docker logs [--service SERVICE] [--tail N]
```

### Benchmark Commands

```bash
python3 main.py run benchmark \
  --model MODEL                    # Required: model name/path
  --experiment-name NAME           # Required: experiment folder name
  --gpu-count N                    # Number of GPUs (default: 1)
  --gpu-ids "0,1"                  # GPU IDs (default: "0")
  --gpu-memory-utilization 0.9    # Memory utilization (default: 0.9)
  --input-len 1024                # Input tokens (default: 1024)
  --output-len 128                # Output tokens (default: 128)
  --num-prompts 200               # End of prompts range (default: 200)
  --num-prompts-start 1           # Start of range (default: 1)
  --generate-script               # Generate shell script instead of running
```

### Web UI

```bash
uv run python3 main.py webui                 # Start on default port 8050
uv run python3 main.py webui --port 8080     # Custom port
uv run python3 main.py webui --output-dir /path/to/results  # Custom results dir
```

Access from other machines on the same LAN: `http://<YOUR_IP>:8050`

## Project Structure

```
vllm-benchmark/
├── main.py                      # Main CLI orchestrator
├── pyproject.toml               # Dependencies
├── docker-compose.yml           # Docker Compose configuration (AMD default)
├── docker-compose.nvidia.yml    # NVIDIA GPU override
├── README.md
│
├── checks/                      # System validation
│   ├── __init__.py
│   ├── gpu_detect.py           # GPU platform auto-detection
│   ├── amd_driver.py           # AMD driver checks
│   ├── nvidia_check.py         # NVIDIA driver checks
│   ├── rocm_check.py           # ROCm installation checks
│   └── docker_check.py         # Docker readiness checks
│
├── docker/                      # Docker management
│   ├── __init__.py
│   └── manager.py              # Container lifecycle management
│
├── benchmarks/                  # Benchmark runners
│   ├── __init__.py
│   └── benchmark_runner.py     # Customizable benchmark runner
│
├── webui/                       # Web UI
│   ├── __init__.py
│   ├── app.py                  # Dash application
│   ├── config.py               # Chart configurations
│   └── data_loader.py          # JSON data loading
│
├── tests/                       # Test suite
│
├── output/                      # Results directory
│   └── {experiment_name}/      # Per-experiment folders
│
└── _sources/                    # Source repositories
    └── vllm_t/                 # Cloned from GitHub
```

## Output Structure

```
output/
└── experiment_name/
    ├── config.json              # GPU, vLLM version, test parameters
    ├── np1_20260130_*.json      # Result for num_prompts=1
    ├── np2_20260130_*.json      # Result for num_prompts=2
    └── ...
```

### config.json Example

```json
{
  "experiment_name": "test1",
  "timestamp": "2026-01-30T14:13:31.081788",
  "model": {
    "name": "meta-llama/Llama-3.1-8B",
    "max_model_len": null
  },
  "gpu": {
    "count": 2,
    "ids": "0,1",
    "memory_utilization": 0.9,
    "info_raw": "..."
  },
  "benchmark": {
    "input_len": 1024,
    "output_len": 128,
    "num_prompts_start": 1,
    "num_prompts_end": 200
  },
  "environment": {
    "vllm_version": "0.13.0+rocm711",
    "rocm_version": "7.1.1"
  }
}
```

## Web UI Features

### Filters

- **Software Version**: Filter by vLLM/ROCm version
- **TP Size**: Tensor parallelism size
- **GPUs**: Number of GPUs
- **Model**: Model name

### Chart Types

- **Overview**: 2x3 grid showing all key metrics
- **Individual Tabs**: Detailed view of each metric
  - System Output Throughput
  - System Output Throughput per Query
  - Time to First Token (TTFT)
  - Time per Output Token (TPOT)
  - Inter-Token Latency (ITL)

## Docker Configuration

The `docker-compose.yml` supports configurable images via environment variable:

```bash
# Using environment variable
VLLM_IMAGE=my-custom-image:v1 docker compose up -d

# Or via CLI
python3 main.py docker start --image my-custom-image:v1
```

The tool auto-detects your GPU platform. For NVIDIA GPUs, it automatically applies `docker-compose.nvidia.yml` as an override to use the NVIDIA Container Toolkit.

Default image: `vllm-rocm71:latest`

## Benchmark Results Format

Results use vLLM's native JSON output format. Each benchmark run produces one JSON file per `num_prompts` value containing metrics like:

- `output_throughput`: System output throughput (tokens/s)
- `request_throughput`: Request throughput (requests/s)
- `mean_ttft_ms`: Mean time to first token (ms)
- `mean_tpot_ms`: Mean time per output token (ms)
- `mean_itl_ms`: Mean inter-token latency (ms)

## Workflows

### Workflow A: Full Benchmark Cycle

See [Quick Start](#quick-start) for the complete two-terminal workflow.

### Workflow B: View Existing Results Only

```bash
# Just launch Web UI (no GPU/Docker required)
uv run python3 main.py webui

# Or with custom results directory
uv run python3 main.py webui --output-dir /path/to/results
```

### Workflow C: Share Web UI on LAN

```bash
# Start webui (binds to 0.0.0.0 by default)
uv run python3 main.py webui

# Find your IP
ip addr | grep "inet " | grep -v 127.0.0.1

# Share with coworkers: http://<YOUR_IP>:8050
```

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Linting

```bash
uv run ruff check .
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
