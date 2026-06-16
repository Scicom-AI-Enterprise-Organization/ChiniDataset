"""Shared fixtures for ChiniDataset tests."""

import json
import numpy as np
import pytest

from chinidataset.writer.parquet import ParquetWriter


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def simple_samples():
    return [
        {"id": i, "text": f"sample {i}", "score": float(i) * 0.5, "flag": i % 2 == 0}
        for i in range(100)
    ]


@pytest.fixture
def numpy_samples():
    rng = np.random.default_rng(42)
    return [
        {"tokens": rng.integers(0, 30000, size=64, dtype=np.uint32), "label": i}
        for i in range(50)
    ]


@pytest.fixture
def write_shards(tmp_dir):
    """Factory: write_shards(samples, columns, size_limit) -> output_dir."""
    def _write(samples, columns, size_limit="1mb"):
        out = tmp_dir / "shards"
        with ParquetWriter(out=out, columns=columns, size_limit=size_limit) as w:
            for s in samples:
                w.write(s)
        return out
    return _write


@pytest.fixture
def local_dataset_dir(tmp_dir, simple_samples):
    """Pre-built dataset directory with 3 shards of simple_samples."""
    columns = {"id": "int64", "text": "str", "score": "float64", "flag": "bool"}
    out = tmp_dir / "dataset"
    # Use a small size limit so we get multiple shards
    with ParquetWriter(out=out, columns=columns, size_limit="1kb") as w:
        for s in simple_samples:
            w.write(s)
    return out


def make_index_json(path, shards):
    """Helper: write an index.json file at path/index.json."""
    path.mkdir(parents=True, exist_ok=True)
    index = {"version": 2, "shards": shards}
    with open(path / "index.json", "w") as f:
        json.dump(index, f)
