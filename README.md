# vLLM Benchmark Tool — Intel GPU

A benchmarking and visualization tool for [vLLM](https://github.com/vllm-project/vllm) on **Intel Arc / Xe GPUs**.

This branch is **Intel-only**. AMD / NVIDIA variants live on separate branches (or will be split into their own projects). See [`ENVIRONMENT.md`](./ENVIRONMENT.md) for the exact image, driver, and IPEX versions this branch was validated against.

## Features

- **Docker-based**: runs vLLM inside [`intel/llm-scaler-vllm`](https://hub.docker.com/r/intel/llm-scaler-vllm) with one `docker-compose.yml`
- **Interactive CLI**: single-command check / start / benchmark / stop workflow
- **Real-time output**: streams benchmark progress as it runs
- **Web dashboard**: compare benchmark runs across experiments
- **Config recording**: auto-saves GPU info, vLLM version, IPEX version, and test parameters

## Installation

### Prerequisites

- Intel Arc / Xe GPU with the `xe` or `i915` kernel driver loaded
- Docker v20.10+ with `docker compose` v2 (or `docker-compose` v1)
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

### Two-terminal workflow

**Terminal 1: host**
```bash
# 1. Check system readiness
python3 main.py check

# 2. Start Docker containers (interactive image selection, or pass --image)
python3 main.py docker start

# 5. Run benchmark (after vLLM server is ready)
python3 main.py run benchmark \
    --model Qwen/Qwen3-0.6B \
    --experiment-name 1xB580_x16_qwen3-0.6b_xpu0.14_1-200_TP1

# 6. View results
uv run python3 main.py webui
# Open http://localhost:8050
```

**Terminal 2: inside container**
```bash
# 3. Enter the vLLM server container
docker exec -it vllm-server bash

# 4. Start vLLM server
vllm serve Qwen/Qwen3-0.6B \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9

# Wait for "Uvicorn running on http://0.0.0.0:8000" before running the benchmark.
```

### Stop environment

```bash
python3 main.py docker stop
```

## CLI Reference

### System check

```bash
python3 main.py check [--verbose]
```

Runs the Intel GPU driver check (looks for the `xe` or `i915` kernel module and
`/dev/dri/renderD*` devices) and the Docker daemon check.

Example output:
```
1. Intel GPU Check
   Intel GPU Driver: OK
      Intel Xe GPU driver loaded
   DRI Render Devices: OK
      Detected 2 DRI render device(s)
        - /dev/dri/renderD128
        - /dev/dri/renderD129
   GPU Count: 2

2. Docker Check
   Docker Daemon: OK
   ...
```

### Docker commands

```bash
python3 main.py docker start [--image IMAGE]
python3 main.py docker stop [--volumes]
python3 main.py docker status
python3 main.py docker logs [--service SERVICE] [--tail N]
python3 main.py docker reset
```

### Benchmark command

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
  --server-url http://localhost:8000
                                  # Defaults to localhost because the compose
                                  # file uses network_mode: host
  --generate-script               # Generate shell script instead of running
```

### Web UI

```bash
uv run python3 main.py webui                 # Start on default port 8050
uv run python3 main.py webui --port 8080     # Custom port
uv run python3 main.py webui --output-dir /path/to/results
```

Access from another machine on the same LAN: `http://<YOUR_IP>:8050`.

## Project Structure

```
vllm-benchmark/
├── main.py                      # CLI orchestrator (Intel only)
├── pyproject.toml               # Dependencies
├── docker-compose.yml           # Intel compose (xe/i915, /dev/dri, host net)
├── README.md
├── ENVIRONMENT.md               # Pinned image + driver + IPEX versions
│
├── checks/                      # System validation
│   ├── __init__.py
│   ├── gpu_check.py             # Intel driver + DRI render device checks
│   └── docker_check.py          # Docker readiness checks
│
├── docker/                      # Docker management
│   ├── __init__.py
│   └── manager.py               # Container lifecycle
│
├── benchmarks/                  # Benchmark runner
│   ├── __init__.py
│   └── benchmark_runner.py      # xpu-only runner
│
├── webui/                       # Dash web UI
│   ├── __init__.py
│   ├── app.py                   # Dash app
│   ├── config.py                # Chart configurations
│   └── data_loader.py           # JSON data loading + folder metadata parse
│
├── tests/                       # Test suite (Intel-only)
│
├── output/                      # Results directory
│   └── {experiment_name}/       # Per-experiment folders
│
└── _sources/                    # Source repositories (optional)
```

## Output structure

```
output/
└── experiment_name/
    ├── config.json              # GPU, vLLM version, IPEX version, test parameters
    ├── np1_20260420_*.json      # Result for num_prompts=1
    ├── np2_20260420_*.json      # Result for num_prompts=2
    └── ...
```

### `config.json` example

```json
{
  "experiment_name": "1xB580_x16_qwen3-0.6b_xpu0.14_1-200_TP1",
  "timestamp": "2026-04-20T14:13:31.081788",
  "model": {
    "name": "Qwen/Qwen3-0.6B",
    "max_model_len": null
  },
  "gpu": {
    "count": 1,
    "ids": "0",
    "memory_utilization": 0.9,
    "info_raw": "..."
  },
  "benchmark": {
    "input_len": 1024,
    "output_len": 128,
    "num_prompts_start": 1,
    "num_prompts_end": 200,
    "server_url": "http://localhost:8000"
  },
  "environment": {
    "platform": "xpu",
    "vllm_version": "0.14.1.dev0+gb17039bcc.d20260311.xpu",
    "runtime_version": "2.10.10.post1+xpu"
  }
}
```

## Web UI filters and charts

**Filters**: Runtime Version, TP Size, GPU Count, Model.

**Charts**:
- Overview (2x3 grid)
- System Output Throughput
- System Output Throughput per Query
- Time to First Token (TTFT)
- Time per Output Token (TPOT)
- Inter-Token Latency (ITL)

## Experiment folder naming

```
{gpu_count}x{gpu_model}_{pcie}_{model}_xpu{ver}_{range}_TP{tp}
```

Example: `1xArcB580_x16_llama-3.1-8b_xpu0.14_1-200_TP1`

The Web UI parses `xpu`, `ipex`, and `oneapi` prefixes in the version tag.

## Docker configuration

The compose file reads `VLLM_IMAGE` from the environment:

```bash
# Use the default:
python3 main.py docker start

# Override via env var:
VLLM_IMAGE=intel/llm-scaler-vllm:0.14.0-b8.1 docker compose up -d

# Override via CLI:
python3 main.py docker start --image my-custom-image:v1
```

**Default image**: `intel/llm-scaler-vllm:0.14.0-b8.1` (see [`ENVIRONMENT.md`](./ENVIRONMENT.md)).

**`network_mode: host` caveat**: both `vllm-server` and `vllm-bench-client` share
the host's network namespace. That means the bench client **cannot** resolve
`vllm-server:8000` by container DNS — you must talk to `http://localhost:8000`.
The CLI's `--server-url` default already points there; only change it if you
rewrite the compose file to use a bridge network.

**Upstream vLLM on Intel**: upstream supports the `xpu` target device
(IPEX + oneAPI) but publishes no prebuilt XPU wheel or image. If you build your
own (`VLLM_TARGET_DEVICE=xpu pip install -e .`), pass `--image <your-image>` —
the compose file honours it via `VLLM_IMAGE`.

## Benchmark result JSON fields

Each run produces one JSON file per `num_prompts` value. Key fields:

- `output_throughput` — system output throughput (tokens/s)
- `request_throughput` — request throughput (requests/s)
- `mean_ttft_ms` — mean time to first token (ms)
- `mean_tpot_ms` — mean time per output token (ms)
- `mean_itl_ms` — mean inter-token latency (ms)

## Development

### Running tests

```bash
uv run pytest tests/ -v
```

### Linting

```bash
uv run ruff check .
```

## License

MIT License.
