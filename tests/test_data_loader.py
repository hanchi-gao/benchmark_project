"""Tests for data loader module."""

import json
import tempfile
from pathlib import Path

from webui.data_loader import (
    extract_metadata_from_folder_name,
    filter_folders_by_metadata,
    get_experiment_folders,
    load_experiment_data,
)


class TestExtractMetadata:
    """Tests for folder name metadata extraction."""

    def test_standard_amd_folder(self):
        meta = extract_metadata_from_folder_name("1xR9700_x16_llama-3.1-8b_rocm7.0_1-200_TP1")
        assert meta["gpu_count"] == 1
        assert meta["pcie_config"] == "x16"
        assert meta["model_name"] == "llama-3.1-8b"
        assert meta["rocm_version"] == "7.0"
        assert meta["tensor_parallel_size"] == 1

    def test_multi_gpu_folder(self):
        meta = extract_metadata_from_folder_name("4xR9700_x8_llama-3.1-8b_rocm7.2_1-200_TP4")
        assert meta["gpu_count"] == 4
        assert meta["tensor_parallel_size"] == 4
        assert meta["rocm_version"] == "7.2"

    def test_cuda_folder(self):
        meta = extract_metadata_from_folder_name("2xpro4500_x16_llama-3.1-8b_cuda13.0_1-200_TP2")
        assert meta["gpu_count"] == 2
        assert meta["rocm_version"] == "CUDA 13.0"
        assert meta["tensor_parallel_size"] == 2

    def test_nvidia_comparison_folder(self):
        meta = extract_metadata_from_folder_name("8xH100-SXM_gpt-oss-120b_1-32768")
        assert meta["gpu_count"] == 8
        assert meta["model_name"] == "gpt-oss-120b"

    def test_qwen_model(self):
        meta = extract_metadata_from_folder_name("4xMI350_qwen3-14b_rocm7.2_1-200_TP4")
        assert meta["model_name"] == "qwen3-14b"
        assert meta["gpu_count"] == 4

    def test_no_metadata_folder(self):
        meta = extract_metadata_from_folder_name("random_folder_name")
        assert meta["gpu_count"] is None
        assert meta["model_name"] is None


class TestGetExperimentFolders:
    """Tests for experiment folder listing."""

    def test_returns_sorted_folders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "b_experiment").mkdir()
            Path(tmpdir, "a_experiment").mkdir()
            # Create a file (should be excluded)
            Path(tmpdir, "not_a_folder.json").write_text("{}")

            folders = get_experiment_folders(tmpdir)
            assert folders == ["a_experiment", "b_experiment"]

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folders = get_experiment_folders(tmpdir)
            assert folders == []

    def test_nonexistent_directory(self):
        folders = get_experiment_folders("/nonexistent/path")
        assert folders == []


class TestLoadExperimentData:
    """Tests for loading experiment JSON data."""

    def test_load_valid_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir, "test_exp")
            exp_dir.mkdir()

            data = {
                "num_prompts": 10,
                "max_concurrent_requests": 10,
                "output_throughput": 100.0,
                "mean_ttft_ms": 50.0,
                "mean_tpot_ms": 10.0,
                "mean_itl_ms": 10.0,
                "request_throughput": 5.0,
            }
            (exp_dir / "np10_test.json").write_text(json.dumps(data))

            df = load_experiment_data("test_exp", tmpdir)
            assert len(df) == 1
            assert df.iloc[0]["output_throughput"] == 100.0

    def test_calculates_output_speed_per_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir, "test_exp")
            exp_dir.mkdir()

            data = {
                "num_prompts": 10,
                "output_throughput": 100.0,
            }
            (exp_dir / "np10.json").write_text(json.dumps(data))

            df = load_experiment_data("test_exp", tmpdir)
            assert "output_speed_per_query" in df.columns
            assert df.iloc[0]["output_speed_per_query"] == 10.0


class TestFilterFolders:
    """Tests for metadata-based folder filtering."""

    def test_filter_by_rocm_version(self):
        folders = [
            "1xR9700_x16_llama-3.1-8b_rocm7.0_1-200_TP1",
            "1xR9700_x16_llama-3.1-8b_rocm7.2_1-200_TP1",
        ]
        filtered = filter_folders_by_metadata(folders, rocm_version="7.0")
        assert len(filtered) == 1
        assert "rocm7.0" in filtered[0]

    def test_filter_by_gpu_count(self):
        folders = [
            "1xR9700_x16_llama-3.1-8b_rocm7.0_1-200_TP1",
            "4xR9700_x16_llama-3.1-8b_rocm7.0_1-200_TP4",
        ]
        filtered = filter_folders_by_metadata(folders, gpu_count=4)
        assert len(filtered) == 1
        assert "4x" in filtered[0]

    def test_no_filters_returns_all(self):
        folders = ["a", "b", "c"]
        filtered = filter_folders_by_metadata(folders)
        assert len(filtered) == 3
