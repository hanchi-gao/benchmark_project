#!/usr/bin/env python3
"""vLLM Benchmark Tool - Main CLI orchestrator."""

import json
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from checks import check_amd_driver, check_rocm, check_docker, detect_gpu_platform, GpuPlatform, check_nvidia
from docker.manager import DockerManager
from benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig

console = Console()

DEFAULT_IMAGE = "vllm-rocm71:latest"


@click.group()
@click.version_option(version="1.0.0", prog_name="amd-bench")
def cli():
    """vLLM Benchmark Tool - Unified benchmarking for vLLM on GPUs."""
    pass


# =============================================================================
# CHECK COMMANDS
# =============================================================================

@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
def check(verbose):
    """Run all system checks (GPU driver, runtime, Docker)."""
    console.print(Panel.fit(
        "[bold blue]vLLM Benchmark - System Check[/bold blue]",
        border_style="blue"
    ))

    all_passed = True

    # GPU Platform Detection
    platform, platform_msg = detect_gpu_platform()
    console.print(f"\n[bold]GPU Platform:[/bold] {platform_msg}")

    if platform == GpuPlatform.AMD:
        # AMD Driver Check
        console.print("\n[bold]1. AMD Driver Check[/bold]")
        driver_result = check_amd_driver()
        for check_item in driver_result["checks"]:
            status = "[green]OK[/green]" if check_item["passed"] else "[red]FAIL[/red]"
            console.print(f"   {check_item['name']}: {status}")
            if verbose or not check_item["passed"]:
                console.print(f"      {check_item['message']}")
        if not driver_result["success"]:
            all_passed = False

        # ROCm Check
        console.print("\n[bold]2. ROCm Installation Check[/bold]")
        rocm_result = check_rocm()
        for check_item in rocm_result["checks"]:
            status = "[green]OK[/green]" if check_item["passed"] else "[red]FAIL[/red]"
            console.print(f"   {check_item['name']}: {status}")
            if verbose or not check_item["passed"]:
                console.print(f"      {check_item['message']}")
        if rocm_result.get("gpu_count", 0) > 0:
            console.print(f"   GPU Count: [cyan]{rocm_result['gpu_count']}[/cyan]")
        if not rocm_result["success"]:
            all_passed = False

    elif platform == GpuPlatform.NVIDIA:
        console.print("\n[bold]1. NVIDIA GPU Check[/bold]")
        nvidia_result = check_nvidia()
        for check_item in nvidia_result["checks"]:
            status = "[green]OK[/green]" if check_item["passed"] else "[red]FAIL[/red]"
            console.print(f"   {check_item['name']}: {status}")
            if verbose or not check_item["passed"]:
                console.print(f"      {check_item['message']}")
        if nvidia_result.get("gpu_count", 0) > 0:
            console.print(f"   GPU Count: [cyan]{nvidia_result['gpu_count']}[/cyan]")
        if not nvidia_result["success"]:
            all_passed = False

    else:
        console.print("\n[yellow]No GPU detected. Benchmark execution requires a GPU,[/yellow]")
        console.print("[yellow]but the Web UI can still be used to view existing results.[/yellow]")

    # Docker Check (always run)
    step_num = 3 if platform == GpuPlatform.AMD else 2
    console.print(f"\n[bold]{step_num}. Docker Check[/bold]")
    docker_result = check_docker()
    for check_item in docker_result["checks"]:
        status = "[green]OK[/green]" if check_item["passed"] else "[red]FAIL[/red]"
        console.print(f"   {check_item['name']}: {status}")
        if verbose or not check_item["passed"]:
            console.print(f"      {check_item['message']}")
    if docker_result.get("available_images"):
        console.print("   Available vLLM images:")
        for img in docker_result["available_images"][:5]:
            console.print(f"      - {img}")
    if not docker_result["success"]:
        all_passed = False

    # Summary
    console.print()
    if all_passed:
        console.print(Panel.fit(
            "[bold green]All checks passed! System is ready for benchmarking.[/bold green]",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]Some checks failed. Please resolve issues before benchmarking.[/bold red]",
            border_style="red"
        ))
        sys.exit(1)


# =============================================================================
# DOCKER COMMANDS
# =============================================================================

@cli.group()
def docker():
    """Docker container management commands."""
    pass


@docker.command()
@click.option('--image', default=DEFAULT_IMAGE, help='Docker image to pull')
def pull(image):
    """Pull vLLM Docker image."""
    console.print(f"[bold]Pulling Docker image:[/bold] {image}")

    manager = DockerManager(PROJECT_ROOT, image=image)
    success, message = manager.pull()

    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]Failed to pull image: {message}[/red]")
        sys.exit(1)


@docker.command()
@click.option('--image', default=None, help='Docker image to use (interactive if not specified)')
def start(image):
    """Start Docker containers."""
    manager = DockerManager(PROJECT_ROOT)

    # Interactive image selection if not specified
    if image is None:
        images = manager.list_available_images()

        if not images:
            console.print("[red]No vLLM/ROCm Docker images found.[/red]")
            console.print("Please pull an image first: python3 main.py docker pull --image <image>")
            sys.exit(1)

        console.print("\n[bold]Available vLLM Images:[/bold]")
        for i, img in enumerate(images, 1):
            console.print(f"  {i}. {img['name']} ({img['size']}, {img['created']})")

        console.print()
        while True:
            try:
                choice = click.prompt(
                    f"Select image [1-{len(images)}]",
                    type=int,
                    default=1
                )
                if 1 <= choice <= len(images):
                    image = images[choice - 1]['name']
                    break
                console.print(f"[red]Please enter a number between 1 and {len(images)}[/red]")
            except (ValueError, click.Abort):
                console.print("[yellow]Cancelled[/yellow]")
                sys.exit(0)

    console.print(f"\n[bold]Starting Docker containers with image:[/bold] {image}")

    manager = DockerManager(PROJECT_ROOT, image=image)
    success, message = manager.start()

    if success:
        console.print("[green]Containers started successfully[/green]")
        # Show status
        _, status = manager.status()
        console.print(status)

        # Print next steps instructions
        console.print(Panel(
            "[bold]Containers started![/bold]\n\n"
            "[cyan]Step 1:[/cyan] Open another terminal and run:\n"
            "  [green]docker exec -it vllm-server bash[/green]\n\n"
            "[cyan]Step 2:[/cyan] Inside the container, start vLLM server:\n"
            "  [green]vllm serve <model> --tensor-parallel-size <N> --gpu-memory-utilization 0.9[/green]\n\n"
            "  [dim]Example:[/dim]\n"
            "  vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 2 --gpu-memory-utilization 0.9\n\n"
            "[cyan]Step 3:[/cyan] Wait until you see [yellow]\"Uvicorn running on http://0.0.0.0:8000\"[/yellow]\n\n"
            "[cyan]Step 4:[/cyan] Back in this terminal, run the benchmark:\n"
            "  [green]python3 main.py run benchmark --model <model> --experiment-name <name>[/green]",
            title="[bold yellow]Next Steps[/bold yellow]",
            border_style="yellow"
        ))
    else:
        console.print(f"[red]Failed to start containers: {message}[/red]")
        sys.exit(1)


@docker.command()
def stop():
    """Stop Docker containers (keeps containers)."""
    console.print("[bold]Stopping Docker containers...[/bold]")

    manager = DockerManager(PROJECT_ROOT)
    success, message = manager.stop()

    if success:
        console.print("[green]Containers stopped successfully[/green]")
    else:
        console.print(f"[red]Failed to stop containers: {message}[/red]")
        sys.exit(1)


@docker.command()
@click.option('--volumes', '-v', is_flag=True, help='Also remove volumes')
def down(volumes):
    """Stop and remove Docker containers."""
    console.print("[bold]Stopping and removing Docker containers...[/bold]")

    manager = DockerManager(PROJECT_ROOT)
    success, message = manager.down(volumes=volumes)

    if success:
        console.print("[green]Containers removed successfully[/green]")
    else:
        console.print(f"[red]Failed to remove containers: {message}[/red]")
        sys.exit(1)


@docker.command()
def reset():
    """Force reset: stop and remove ALL vllm containers."""
    console.print("[bold yellow]Resetting Docker environment...[/bold yellow]")
    console.print("This will remove ALL vllm-related containers.\n")

    manager = DockerManager(PROJECT_ROOT)
    success, message = manager.reset()

    console.print(message)
    if success:
        console.print("\n[green]Docker environment reset successfully[/green]")
    else:
        console.print(f"\n[red]Reset completed with warnings[/red]")


@docker.command()
def status():
    """Show Docker container status."""
    manager = DockerManager(PROJECT_ROOT)
    success, output = manager.status()

    if success:
        console.print(output)
    else:
        console.print(f"[red]Failed to get status: {output}[/red]")


@docker.command()
@click.option('--service', default=None, help='Service name (vllm-server or vllm-bench-client)')
@click.option('--tail', default=100, help='Number of log lines to show')
def logs(service, tail):
    """Show Docker container logs."""
    manager = DockerManager(PROJECT_ROOT)
    success, output = manager.logs(service=service, tail=tail)

    if success:
        console.print(output)
    else:
        console.print(f"[red]Failed to get logs: {output}[/red]")


# =============================================================================
# BENCHMARK COMMANDS
# =============================================================================

@cli.group()
def run():
    """Run benchmarks and tests."""
    pass


@run.command()
@click.option('--gpu-count', default=1, help='Number of GPUs to use')
@click.option('--gpu-ids', default="0", help='Comma-separated GPU IDs (e.g., "0,1")')
@click.option('--model', required=True, help='Model name or path')
@click.option('--gpu-memory-utilization', default=0.9, help='GPU memory utilization (0.0-1.0)')
@click.option('--input-len', default=1024, help='Input length in tokens')
@click.option('--output-len', default=128, help='Output length in tokens')
@click.option('--num-prompts', default=200, help='Number of prompts (or end of range)')
@click.option('--num-prompts-start', default=1, help='Start of prompts range')
@click.option('--experiment-name', required=True, help='Name for the experiment folder')
@click.option('--max-model-len', default=None, type=int, help='Max model length')
@click.option('--server-url', default="http://vllm-server:8000", help='vLLM server URL (internal container URL)')
@click.option('--generate-script', is_flag=True, help='Generate shell script instead of running')
def benchmark(gpu_count, gpu_ids, model, gpu_memory_utilization, input_len, output_len,
              num_prompts, num_prompts_start, experiment_name, max_model_len, server_url,
              generate_script):
    """Run benchmark with custom GPU configuration.

    Benchmarks are executed INSIDE the vllm-bench-client container via docker exec.
    Make sure containers are running (python3 main.py docker start) and vLLM server
    is started inside the vllm-server container.
    """
    # Input validation
    import re as _re

    if not (0.0 < gpu_memory_utilization <= 1.0):
        console.print("[red]Error: --gpu-memory-utilization must be between 0.0 and 1.0 (exclusive/inclusive)[/red]")
        sys.exit(1)

    if not _re.match(r'^\d+(,\d+)*$', gpu_ids):
        console.print("[red]Error: Invalid GPU IDs format. Use comma-separated integers (e.g., '0,1,2')[/red]")
        sys.exit(1)

    if num_prompts_start > num_prompts:
        console.print("[red]Error: --num-prompts-start cannot be greater than --num-prompts (start > end)[/red]")
        sys.exit(1)

    manager = DockerManager(PROJECT_ROOT)

    # Check if containers are running
    if not manager.is_running("vllm-bench-client"):
        console.print("[red]Error: vllm-bench-client container is not running[/red]")
        console.print("\nPlease start the containers first:")
        console.print("  python3 main.py docker start")
        sys.exit(1)

    if generate_script:
        # Generate shell script (still uses BenchmarkRunner for this)
        config = BenchmarkConfig(
            gpu_count=gpu_count,
            gpu_ids=gpu_ids,
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            input_len=input_len,
            output_len=output_len,
            num_prompts=num_prompts,
            num_prompts_start=num_prompts_start,
            num_prompts_end=num_prompts,
            experiment_name=experiment_name,
            max_model_len=max_model_len,
            vllm_server_url=server_url,
        )
        runner = BenchmarkRunner(config, output_dir=PROJECT_ROOT / "output")
        script = runner.generate_shell_script()
        script_path = PROJECT_ROOT / "scripts" / f"run_{experiment_name}.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        console.print(f"[green]Generated script: {script_path}[/green]")
        return

    # Display configuration
    console.print(Panel.fit(
        "[bold blue]vLLM Benchmark[/bold blue]",
        border_style="blue"
    ))

    table = Table(title="Benchmark Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Experiment Name", experiment_name)
    table.add_row("Model", model)
    table.add_row("GPU Count", str(gpu_count))
    table.add_row("GPU IDs", gpu_ids)
    table.add_row("GPU Memory Utilization", str(gpu_memory_utilization))
    table.add_row("Input Length", str(input_len))
    table.add_row("Output Length", str(output_len))
    table.add_row("Prompts Range", f"{num_prompts_start} - {num_prompts}")
    table.add_row("Server URL", server_url)
    table.add_row("Execution", "Inside vllm-bench-client container")

    console.print(table)

    # Build the benchmark command to run inside the container
    experiment_dir = f"/root/output/{experiment_name}"

    console.print("\n[bold]Running benchmark inside container...[/bold]")
    console.print(f"[dim]Output directory: {experiment_dir}[/dim]\n")

    # Create experiment directory inside container
    mkdir_cmd = f"mkdir -p {experiment_dir}"
    manager.exec_benchmark(mkdir_cmd, stream_output=False)

    # Collect and save experiment configuration
    console.print("[bold]Collecting system information...[/bold]")

    # Get GPU info from container
    gpu_info_cmd = "rocm-smi --showproductname --showmeminfo vram --json 2>/dev/null || echo '{}'"
    _, gpu_info_raw = manager.exec_benchmark(gpu_info_cmd, stream_output=False)

    # Get vLLM version
    vllm_version_cmd = "pip show vllm 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'unknown'"
    _, vllm_version = manager.exec_benchmark(vllm_version_cmd, stream_output=False)

    # Get ROCm version
    rocm_version_cmd = "cat /opt/rocm/.info/version 2>/dev/null || echo 'unknown'"
    _, rocm_version = manager.exec_benchmark(rocm_version_cmd, stream_output=False)

    # Build config JSON
    config = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": model,
            "max_model_len": max_model_len
        },
        "gpu": {
            "count": gpu_count,
            "ids": gpu_ids,
            "memory_utilization": gpu_memory_utilization,
            "info_raw": gpu_info_raw.strip()
        },
        "benchmark": {
            "input_len": input_len,
            "output_len": output_len,
            "num_prompts_start": num_prompts_start,
            "num_prompts_end": num_prompts,
            "server_url": server_url
        },
        "environment": {
            "vllm_version": vllm_version.strip(),
            "rocm_version": rocm_version.strip()
        }
    }

    # Save config file inside container (will be synced to host via volume)
    config_json = json.dumps(config, indent=2)
    save_config_cmd = f"cat > {experiment_dir}/config.json << 'EOF'\n{config_json}\nEOF"
    manager.exec_benchmark(save_config_cmd, stream_output=False)

    console.print(f"[green]✓ Configuration saved to {experiment_dir}/config.json[/green]\n")

    # Run benchmarks for each num_prompts value
    total = num_prompts - num_prompts_start + 1
    completed = 0
    failed = 0

    for np in range(num_prompts_start, num_prompts + 1):
        completed += 1
        result_file = f"{experiment_dir}/np{np}_$(date +%Y%m%d_%H%M%S).json"

        bench_cmd = f"""
cd {experiment_dir} && \\
vllm bench serve \\
    --model "{model}" \\
    --backend openai \\
    --endpoint /v1/completions \\
    --base-url {server_url} \\
    --dataset-name random \\
    --random-input-len {input_len} \\
    --random-output-len {output_len} \\
    --num-prompts {np} \\
    --ignore-eos \\
    --save-result \\
    --result-filename "np{np}_$(date +%Y%m%d_%H%M%S).json"
"""
        console.print(f"[cyan][{completed}/{total}] Running num_prompts={np}...[/cyan]")

        success, output = manager.exec_benchmark(bench_cmd.strip(), stream_output=True)

        if success:
            console.print(f"[green]  ✓ num_prompts={np} completed[/green]")
        else:
            console.print(f"[red]  ✗ num_prompts={np} failed[/red]")
            failed += 1

    # Summary
    console.print()
    local_output_dir = PROJECT_ROOT / "output" / experiment_name
    console.print(Panel.fit(
        f"[bold]Benchmark Complete[/bold]\n\n"
        f"Total: {total}\n"
        f"Completed: [green]{total - failed}[/green]\n"
        f"Failed: [red]{failed}[/red]\n\n"
        f"Results: {local_output_dir}",
        border_style="green" if failed == 0 else "yellow"
    ))


# =============================================================================
# WEB UI COMMAND
# =============================================================================

@cli.command()
@click.option('--output-dir', default=None, help='Directory containing benchmark results')
@click.option('--host', default="0.0.0.0", help='Host address to bind')
@click.option('--port', default=8050, help='Port number')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def webui(output_dir, host, port, debug):
    """Launch web UI for result visualization."""
    from webui.app import run_app

    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "output")

    console.print(Panel.fit(
        "[bold blue]vLLM Benchmark - Web UI[/bold blue]",
        border_style="blue"
    ))

    console.print(f"Output directory: {output_dir}")
    console.print(f"Starting web UI at http://{host}:{port}")
    console.print(f"Press Ctrl+C to stop\n")

    run_app(output_dir=output_dir, host=host, port=port, debug=debug)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    cli()
