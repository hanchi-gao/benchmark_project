"""Docker container management for vLLM benchmarking."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class DockerManager:
    """Manages Docker containers for vLLM benchmarking."""

    DEFAULT_IMAGE = "vllm-rocm71:latest"

    def __init__(self, project_dir: Optional[Path] = None, image: Optional[str] = None):
        """
        Initialize Docker manager.

        Args:
            project_dir: Project directory containing docker-compose.yml
            image: Docker image to use (default: vllm-rocm71:latest)
        """
        self.project_dir = project_dir or Path(__file__).parent.parent
        self.image = image or self.DEFAULT_IMAGE
        self.compose_file = self.project_dir / "docker-compose.yml"

    def _run_compose(self, *args, env: Optional[dict] = None) -> Tuple[bool, str]:
        """Run docker compose command."""
        cmd = ["docker", "compose", "-f", str(self.compose_file)] + list(args)

        # Set up environment with image override
        run_env = os.environ.copy()
        run_env["VLLM_IMAGE"] = self.image
        if env:
            run_env.update(env)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=300,
                env=run_env
            )
            if result.returncode == 0:
                return True, result.stdout
            return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def pull(self) -> Tuple[bool, str]:
        """Pull the vLLM Docker image."""
        try:
            result = subprocess.run(
                ["docker", "pull", self.image],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for pull
            )
            if result.returncode == 0:
                return True, f"Successfully pulled {self.image}"
            return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Pull timed out"
        except Exception as e:
            return False, str(e)

    def start(self, detach: bool = True) -> Tuple[bool, str]:
        """Start Docker containers."""
        args = ["up"]
        if detach:
            args.append("-d")
        return self._run_compose(*args)

    def stop(self) -> Tuple[bool, str]:
        """Stop Docker containers (keeps containers)."""
        return self._run_compose("stop")

    def down(self, volumes: bool = False) -> Tuple[bool, str]:
        """Stop and remove Docker containers."""
        args = ["down"]
        if volumes:
            args.append("-v")
        return self._run_compose(*args)

    def reset(self) -> Tuple[bool, str]:
        """Force stop and remove all vllm containers (including orphans)."""
        messages = []

        # First, try docker compose down
        success, msg = self._run_compose("down", "--remove-orphans")
        messages.append(f"Compose down: {'OK' if success else msg}")

        # Force remove any remaining vllm containers
        try:
            # Get all vllm-related containers
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=vllm", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=30
            )
            containers = [c.strip() for c in result.stdout.strip().split('\n') if c.strip()]

            if containers:
                # Stop containers first
                subprocess.run(
                    ["docker", "stop"] + containers,
                    capture_output=True,
                    timeout=60
                )
                # Remove containers
                result = subprocess.run(
                    ["docker", "rm", "-f"] + containers,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                messages.append(f"Removed containers: {', '.join(containers)}")
            else:
                messages.append("No vllm containers to remove")

        except Exception as e:
            messages.append(f"Error removing containers: {e}")

        return True, "\n".join(messages)

    def remove_container(self, name: str) -> Tuple[bool, str]:
        """Remove a specific container by name."""
        try:
            # Stop first
            subprocess.run(
                ["docker", "stop", name],
                capture_output=True,
                timeout=30
            )
            # Then remove
            result = subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return True, f"Removed container: {name}"
            return False, result.stderr
        except Exception as e:
            return False, str(e)

    def status(self) -> Tuple[bool, str]:
        """Get status of Docker containers."""
        return self._run_compose("ps")

    def logs(self, service: Optional[str] = None, tail: int = 100) -> Tuple[bool, str]:
        """Get logs from Docker containers."""
        args = ["logs", f"--tail={tail}"]
        if service:
            args.append(service)
        return self._run_compose(*args)

    def exec_server(self, command: str) -> Tuple[bool, str]:
        """Execute command in vllm-server container."""
        try:
            result = subprocess.run(
                ["docker", "exec", "vllm-server", "bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour for long-running commands
            )
            if result.returncode == 0:
                return True, result.stdout
            return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def exec_client(self, command: str) -> Tuple[bool, str]:
        """Execute command in vllm-bench-client container."""
        try:
            result = subprocess.run(
                ["docker", "exec", "vllm-bench-client", "bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour for long-running commands
            )
            if result.returncode == 0:
                return True, result.stdout
            return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def is_running(self, service: str = "vllm-server") -> bool:
        """Check if a service is running."""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={service}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return service in result.stdout
        except Exception:
            return False

    def list_available_images(self) -> List[dict]:
        """
        List all vLLM-related Docker images with details.

        Returns:
            List of dicts with image info: name, size, created
        """
        try:
            # Get images with vllm or rocm in the name
            result = subprocess.run(
                [
                    "docker", "images",
                    "--format", "{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return []

            images = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    name = parts[0]
                    # Filter for vllm or rocm related images
                    if 'vllm' in name.lower() or 'rocm' in name.lower():
                        images.append({
                            "name": name,
                            "size": parts[1],
                            "created": parts[2]
                        })

            return images

        except Exception:
            return []

    def exec_benchmark(
        self,
        command: str,
        stream_output: bool = True,
        container: str = "vllm-bench-client"
    ) -> Tuple[bool, str]:
        """
        Execute benchmark command inside container with streaming output.

        Args:
            command: The command to execute
            stream_output: If True, stream stdout/stderr to console in real-time
            container: Container name to execute in

        Returns:
            Tuple of (success, output)
        """
        docker_cmd = ["docker", "exec", container, "bash", "-c", command]

        try:
            if stream_output:
                # Use Popen for real-time output streaming
                process = subprocess.Popen(
                    docker_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                output_lines = []
                # Stream output line by line
                for line in process.stdout:
                    print(line, end='', flush=True)
                    output_lines.append(line)

                process.wait()
                full_output = ''.join(output_lines)

                return process.returncode == 0, full_output
            else:
                # Capture output without streaming
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600
                )
                if result.returncode == 0:
                    return True, result.stdout
                return False, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
