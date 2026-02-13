# vLLM Benchmark Tool

Unified benchmarking and visualization platform for vLLM across AMD (ROCm) and NVIDIA (CUDA) GPUs.
Python 3.10+ | CLI via Click | Web dashboard via Dash/Plotly | Docker-based execution

## Commands

- Install: `uv sync`
- Test: `uv run pytest tests/ -v`
- Lint: `uv run ruff check .`
- Lint fix: `uv run ruff check --fix .`
- Format: `uv run black .`
- Run CLI: `python3 main.py <command>`
- Web UI: `uv run python3 main.py webui`

## Architecture

- Entry point: `main.py` (Click CLI with command groups: check, docker, run, webui)
- GPU detection: `checks/gpu_detect.py` - auto-detects AMD vs NVIDIA, do not hardcode platform assumptions
- Docker: two-container model (vllm-server + vllm-bench-client) with compose override for NVIDIA (`docker-compose.nvidia.yml`)
- Benchmarks: always use `BenchmarkRunner` class, never inline benchmark logic in main.py
- Web UI: Dash app in `webui/`, chart configs in `webui/config.py`, data loading in `webui/data_loader.py`
- Output folder naming: `{gpu_count}x{gpu_model}_{pcie}_{model}_{runtime}{ver}_{range}_TP{tp}`
  - Example: `2xR9700_x16_llama-3.1-8b_rocm7.2_1-200_TP2`
  - Parsed by regex in `webui/data_loader.py:extract_metadata_from_folder_name()`. Do not change the naming convention without updating the parser.

## Conventions

- Use `uv` for package management (not pip directly)
- Use `ruff` for linting (rules: E, F, W, I; line length: 120)
- Use `pytest` for tests
- Use `black` for formatting
- Do not introduce new tools or dependencies without asking first
- Python 3.10+ - use modern syntax (match statements, type hints with `|` union)
- Use `rich` for terminal output formatting
- Keep Click command signatures consistent with existing patterns in main.py

## Testing

- Tests live in `tests/` directory
- Run full suite before claiming work is done: `uv run pytest tests/ -v`
- Test files follow pattern: `test_<module>.py`
- Use mocks for GPU/Docker checks (no real hardware needed for unit tests)
- When modifying checks/, benchmarks/, or webui/, update corresponding tests

## Prerequisites

- **OS**: Linux with GPU kernel module support (amdgpu or nvidia)
- **Python**: 3.10+
- **Docker**: v20.10+ with `docker compose` v2 (or docker-compose v1)
- **AMD**: ROCm 7.0+, `/opt/rocm` installed, `/dev/kfd` and `/dev/dri` devices, `amdgpu` kernel module loaded
- **NVIDIA**: nvidia-smi available, NVIDIA Container Toolkit installed
- **Default Docker image**: `vllm-rocm71:latest` (override with `VLLM_IMAGE` env var)

## Config Locations

- `docker-compose.yml` — main compose config (AMD/ROCm devices, volumes, network)
- `docker-compose.nvidia.yml` — NVIDIA override (replaces AMD devices with nvidia driver)
- `.claude/settings.local.json` — Claude Code tool permissions
- No `.env` file — config is in compose files or passed via CLI
- HuggingFace cache mounted from `~/.cache/huggingface`

## Workflows

### Adding a new benchmark test

1. Start Docker: `python3 main.py docker start`
2. Run benchmark:
   ```
   python3 main.py run benchmark \
     --model meta-llama/Llama-3.1-8B \
     --experiment-name 1xR9700_x16_llama-3.1-8b_rocm7.2_1-200_TP1 \
     --gpu-count 1 --gpu-ids "0" \
     --input-len 1024 --output-len 128 \
     --num-prompts 200 --num-prompts-start 1
   ```
3. Results saved to `output/{experiment-name}/np{N}_{timestamp}.json`
4. Or use `--generate-script` to create a shell script instead of running directly

### Adding a new Web UI chart

1. Add entry to `CHART_TABS` in `webui/config.py`:
   ```python
   "My New Metric": {
       "type": "single",
       "y_field": "json_field_name",
       "y_label": "Label (units)",
       "title": "Metric vs. Concurrency"
   }
   ```
2. Ensure the field exists in benchmark result JSON files
3. No callback changes needed — the Dash app auto-creates tabs from `CHART_TABS`
4. For new filters: update `METADATA_FIELDS` in config.py, add dropdown in `app.py` layout, update `update_folder_options()` callback
