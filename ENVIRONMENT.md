# Verified Environment

This file records the Intel GPU stack that this branch was validated against.
When the prebuilt Intel container, kernel driver, or IPEX version changes,
update the values below so future readers know what was actually tested.

## Last verified: 2026-04-20

### Container image

| Field | Value |
|---|---|
| Repository | `intel/llm-scaler-vllm` |
| Tag | `0.14.0-b8.1` |
| Image digest | `sha256:97a2f4e31d47228ea235e23c0e44bfe56971985fb0bb2e6f9f28ff12b6b549bf` |
| Built | 2026-03-11 |
| Architecture / OS | amd64 / linux |
| Compressed / on-disk size | 13.5 GB / 48 GB |

### Host kernel and driver

| Field | Value |
|---|---|
| Kernel | Linux 6.17.0-22-generic (Ubuntu PREEMPT_DYNAMIC, 2026-03-13) |
| GPU kernel module | `xe` |
| DRI render devices | `/dev/dri/renderD128`, `renderD129`, `renderD130` |
| `/dev/dri/card2` | AMD iGPU (via `amdgpu` kernel module; not used for vLLM) |

### Intel GPUs under test

Two discrete Intel Arc Battlemage GPUs (device ID `0xe223`), detected via
`xpu-smi discovery` inside the container:

```
+-----------+--------------------------------------------------------------------------------------+
| Device ID | Device Information                                                                   |
+-----------+--------------------------------------------------------------------------------------+
| 0         | Device Name: Intel(R) Graphics [0xe223]                                              |
|           | Vendor Name: Intel(R) Corporation                                                    |
|           | SOC UUID: 00000000-0000-0003-0000-0000e2238086                                       |
|           | PCI BDF Address: 0000:03:00.0                                                        |
|           | DRM Device: /dev/dri/card0                                                           |
|           | Function Type: physical                                                              |
+-----------+--------------------------------------------------------------------------------------+
| 1         | Device Name: Intel(R) Graphics [0xe223]                                              |
|           | Vendor Name: Intel(R) Corporation                                                    |
|           | SOC UUID: 00000000-0000-0008-0000-0000e2238086                                       |
|           | PCI BDF Address: 0000:08:00.0                                                        |
|           | DRM Device: /dev/dri/card1                                                           |
|           | Function Type: physical                                                              |
+-----------+--------------------------------------------------------------------------------------+
```

### Runtime versions (inside the container)

| Component | Version |
|---|---|
| vLLM | `0.14.1.dev0+gb17039bcc.d20260311.xpu` |
| IPEX (`intel_extension_for_pytorch`) | `2.10.10.post1+xpu` |
| PyTorch | `2.10.0+xpu` |
| oneAPI DPC++/C++ Compiler | `2025.3.2 (2025.3.2.20260112)` |
| `xpu-smi` | `1.3.5.20251217` |

### Notes

- Upstream vLLM does support the `xpu` target device via IPEX + oneAPI Level Zero,
  but there are no official prebuilt XPU wheels/images. This branch defaults to
  Intel's prebuilt `intel/llm-scaler-vllm` for ergonomics. If you build your own
  upstream XPU image (`VLLM_TARGET_DEVICE=xpu pip install -e .`), pass
  `--image <your-image>` to `docker start` — the compose file reads `VLLM_IMAGE`.
- The compose file uses `network_mode: host`, so the benchmark client talks to
  `http://localhost:8000` (not `http://vllm-server:8000`). Don't change this
  without also switching to a bridge network.
- GPU selection inside the container uses `ONEAPI_DEVICE_SELECTOR=level_zero:<ids>`.
