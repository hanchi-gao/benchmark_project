"""Data loading module for benchmark results."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def get_experiment_folders(output_dir: str = "output") -> List[str]:
    """Return experiment folder names in output_dir, sorted."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    folders = [f.name for f in output_path.iterdir() if f.is_dir()]
    return sorted(folders)


def load_experiment_data(folder_name: str, output_dir: str = "output") -> pd.DataFrame:
    """Load all JSON files from a single experiment folder into a DataFrame."""
    folder_path = Path(output_dir) / folder_name
    if not folder_path.exists():
        return pd.DataFrame()

    data_list = []
    json_files = sorted(folder_path.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_list.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    if not data_list:
        return pd.DataFrame()

    df = pd.DataFrame(data_list)

    if 'max_concurrent_requests' not in df.columns and 'num_prompts' in df.columns:
        df['max_concurrent_requests'] = df['num_prompts']

    if 'output_speed_per_query' not in df.columns:
        if 'output_throughput' in df.columns and 'max_concurrent_requests' in df.columns:
            df['output_speed_per_query'] = df['output_throughput'] / df['max_concurrent_requests']

    return df


def load_multiple_experiments(folder_names: List[str], output_dir: str = "output") -> Dict[str, pd.DataFrame]:
    """Load several experiment folders into a dict of DataFrames."""
    result = {}
    for folder_name in folder_names:
        df = load_experiment_data(folder_name, output_dir)
        if not df.empty:
            result[folder_name] = df
    return result


def extract_metadata_from_folder_name(folder_name: str) -> dict:
    """
    Extract metadata from an experiment folder name.

    Expected pattern:
        {gpu_count}x{gpu_model}_{pcie}_{model}_xpu{ver}_{range}_TP{tp}
        e.g. 1xArcB580_x16_llama-3.1-8b_xpu0.14_1-200_TP1
    """
    metadata = {
        "gpu_count": None,
        "gpu_model": None,
        "pcie_config": None,
        "model_name": None,
        "runtime_version": None,
        "tensor_parallel_size": None,
    }

    gpu_count_match = re.search(r'^(\d+)x', folder_name)
    if gpu_count_match:
        metadata["gpu_count"] = int(gpu_count_match.group(1))

    tp_match = re.search(r'TP(\d+)', folder_name, re.IGNORECASE)
    if tp_match:
        metadata["tensor_parallel_size"] = int(tp_match.group(1))

    xpu_match = re.search(r'(?:xpu|ipex|oneapi)([\d.]+)', folder_name, re.IGNORECASE)
    if xpu_match:
        metadata["runtime_version"] = f"XPU {xpu_match.group(1)}"

    model_patterns = [
        r'(llama-[\d.]+-\d+b)',
        r'(gpt-oss-\d+b)',
        r'(qwen[\d]+-[\d.]+b)',
        r'(phi-\d+-\d+b)',
    ]
    for pattern in model_patterns:
        model_match = re.search(pattern, folder_name, re.IGNORECASE)
        if model_match:
            metadata["model_name"] = model_match.group(1).lower()
            break

    pcie_match = re.search(r'_x(\d+)_', folder_name)
    if pcie_match:
        metadata["pcie_config"] = f"x{pcie_match.group(1)}"

    return metadata


def get_all_metadata_values(output_dir: str = "output") -> dict:
    """Collect unique metadata values across all experiment folders."""
    folders = get_experiment_folders(output_dir)

    values = {
        "runtime_version": set(),
        "tensor_parallel_size": set(),
        "gpu_count": set(),
        "model_name": set(),
    }

    for folder in folders:
        metadata = extract_metadata_from_folder_name(folder)
        for key, value in metadata.items():
            if value is not None and key in values:
                values[key].add(value)

    return {
        key: sorted(list(v), key=lambda x: (isinstance(x, str), x))
        for key, v in values.items()
    }


def filter_folders_by_metadata(
    folders: List[str],
    runtime_version: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    gpu_count: Optional[int] = None,
    model_name: Optional[str] = None,
) -> List[str]:
    """Filter experiment folders by metadata criteria."""
    filtered = []

    for folder in folders:
        metadata = extract_metadata_from_folder_name(folder)

        if runtime_version and metadata["runtime_version"] != runtime_version:
            continue
        if tensor_parallel_size and metadata["tensor_parallel_size"] != tensor_parallel_size:
            continue
        if gpu_count and metadata["gpu_count"] != gpu_count:
            continue
        if model_name and metadata["model_name"] != model_name:
            continue

        filtered.append(folder)

    return filtered
