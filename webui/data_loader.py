"""Data loading module for benchmark results."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def get_experiment_folders(output_dir: str = "output") -> List[str]:
    """
    Get all experiment folder names from the output directory.

    Args:
        output_dir: Path to output directory

    Returns:
        Sorted list of folder names
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    folders = [f.name for f in output_path.iterdir() if f.is_dir()]
    return sorted(folders)


def load_experiment_data(folder_name: str, output_dir: str = "output") -> pd.DataFrame:
    """
    Load all JSON files from an experiment folder.

    Args:
        folder_name: Experiment folder name
        output_dir: Path to output directory

    Returns:
        DataFrame containing all benchmark data
    """
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

    # Map num_prompts to max_concurrent_requests if needed
    if 'max_concurrent_requests' not in df.columns and 'num_prompts' in df.columns:
        df['max_concurrent_requests'] = df['num_prompts']

    # Calculate output_speed_per_query if not present
    if 'output_speed_per_query' not in df.columns:
        if 'output_throughput' in df.columns and 'max_concurrent_requests' in df.columns:
            df['output_speed_per_query'] = df['output_throughput'] / df['max_concurrent_requests']

    return df


def load_multiple_experiments(folder_names: List[str], output_dir: str = "output") -> Dict[str, pd.DataFrame]:
    """
    Load data from multiple experiment folders.

    Args:
        folder_names: List of experiment folder names
        output_dir: Path to output directory

    Returns:
        Dictionary mapping folder names to DataFrames
    """
    result = {}
    for folder_name in folder_names:
        df = load_experiment_data(folder_name, output_dir)
        if not df.empty:
            result[folder_name] = df
    return result


def extract_metadata_from_folder_name(folder_name: str) -> dict:
    """
    Extract metadata from experiment folder name.

    Common patterns:
    - 1xR9700_x16_llama-3.1-8b_rocm7.0_1-200_TP1
    - 2xpro4500_x16_llama-3.1-8b_cuda13.0_1-200_TP2

    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        "gpu_count": None,
        "gpu_model": None,
        "pcie_config": None,
        "model_name": None,
        "rocm_version": None,
        "tensor_parallel_size": None,
    }

    # Try to extract GPU count (e.g., "1x", "2x", "4x", "8x")
    gpu_count_match = re.search(r'^(\d+)x', folder_name)
    if gpu_count_match:
        metadata["gpu_count"] = int(gpu_count_match.group(1))

    # Try to extract TP size
    tp_match = re.search(r'TP(\d+)', folder_name, re.IGNORECASE)
    if tp_match:
        metadata["tensor_parallel_size"] = int(tp_match.group(1))

    # Try to extract ROCm version
    rocm_match = re.search(r'rocm([\d.]+)', folder_name, re.IGNORECASE)
    if rocm_match:
        metadata["rocm_version"] = rocm_match.group(1)

    # Try to extract CUDA version (for NVIDIA comparison)
    cuda_match = re.search(r'cuda([\d.]+)', folder_name, re.IGNORECASE)
    if cuda_match:
        metadata["rocm_version"] = f"CUDA {cuda_match.group(1)}"

    # Try to extract model name
    model_patterns = [
        r'(llama-[\d.]+-\d+b)',
        r'(gpt-oss-\d+b)',
        r'(qwen[\d]+-\d+b)',
    ]
    for pattern in model_patterns:
        model_match = re.search(pattern, folder_name, re.IGNORECASE)
        if model_match:
            metadata["model_name"] = model_match.group(1).lower()
            break

    # Try to extract PCIe config
    pcie_match = re.search(r'_x(\d+)_', folder_name)
    if pcie_match:
        metadata["pcie_config"] = f"x{pcie_match.group(1)}"

    return metadata


def get_all_metadata_values(output_dir: str = "output") -> dict:
    """
    Get all unique metadata values from experiment folders.

    Returns:
        Dictionary with lists of unique values for each metadata field
    """
    folders = get_experiment_folders(output_dir)

    values = {
        "rocm_version": set(),
        "tensor_parallel_size": set(),
        "gpu_count": set(),
        "model_name": set(),
    }

    for folder in folders:
        metadata = extract_metadata_from_folder_name(folder)
        for key, value in metadata.items():
            if value is not None and key in values:
                values[key].add(value)

    # Convert sets to sorted lists
    return {
        key: sorted(list(v), key=lambda x: (isinstance(x, str), x))
        for key, v in values.items()
    }


def filter_folders_by_metadata(
    folders: List[str],
    rocm_version: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    gpu_count: Optional[int] = None,
    model_name: Optional[str] = None,
) -> List[str]:
    """
    Filter experiment folders by metadata criteria.

    Args:
        folders: List of folder names
        rocm_version: ROCm version to filter by
        tensor_parallel_size: TP size to filter by
        gpu_count: GPU count to filter by
        model_name: Model name to filter by

    Returns:
        Filtered list of folder names
    """
    filtered = []

    for folder in folders:
        metadata = extract_metadata_from_folder_name(folder)

        if rocm_version and metadata["rocm_version"] != rocm_version:
            continue
        if tensor_parallel_size and metadata["tensor_parallel_size"] != tensor_parallel_size:
            continue
        if gpu_count and metadata["gpu_count"] != gpu_count:
            continue
        if model_name and metadata["model_name"] != model_name:
            continue

        filtered.append(folder)

    return filtered
