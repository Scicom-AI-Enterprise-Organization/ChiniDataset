"""Tests for chinidataset/dataset/reader.py (ParquetReader)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chinidataset.dataset.reader import ParquetReader


def _write_shard(path: Path, n: int = 10) -> Path:
    """Write a small test shard and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "id": pa.array(list(range(n)), type=pa.int64()),
        "text": pa.array([f"row {i}" for i in range(n)]),
    })
    pq.write_table(table, str(path))
    return path


class TestParquetReaderBasic:
    def test_lazy_load(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet")
        reader = ParquetReader(shard)
        assert not reader.is_loaded

    def test_getitem_triggers_load(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet")
        reader = ParquetReader(shard)
        sample = reader[0]
        assert reader.is_loaded
        assert sample["id"] == 0

    def test_getitem_correct_values(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet", n=5)
        reader = ParquetReader(shard)
        for i in range(5):
            assert reader[i]["id"] == i
            assert reader[i]["text"] == f"row {i}"

    def test_len(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet", n=7)
        reader = ParquetReader(shard)
        assert len(reader) == 7

    def test_load_idempotent(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet", n=3)
        reader = ParquetReader(shard)
        _ = reader[0]
        _ = reader[0]
        assert reader.is_loaded

    def test_negative_index(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet", n=5)
        reader = ParquetReader(shard)
        assert reader[-1]["id"] == 4

    def test_out_of_bounds_raises(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet", n=5)
        reader = ParquetReader(shard)
        with pytest.raises(IndexError):
            _ = reader[10]


class TestParquetReaderUnload:
    def test_unload_clears_data(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet")
        reader = ParquetReader(shard)
        _ = reader[0]
        assert reader.is_loaded
        reader.unload()
        assert not reader.is_loaded

    def test_reload_after_unload(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet", n=4)
        reader = ParquetReader(shard)
        _ = reader[0]
        reader.unload()
        assert reader[2]["id"] == 2

    def test_len_after_reload(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet", n=6)
        reader = ParquetReader(shard)
        _ = reader[0]
        reader.unload()
        assert len(reader) == 6


class TestParquetReaderAsync:
    def test_load_async_returns(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet")
        reader = ParquetReader(shard)
        with ThreadPoolExecutor(max_workers=1) as ex:
            reader.load_async(ex)
            assert reader._future is not None
            reader.wait_loaded()
        assert reader.is_loaded

    def test_getitem_after_async(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet", n=5)
        reader = ParquetReader(shard)
        with ThreadPoolExecutor(max_workers=1) as ex:
            reader.load_async(ex)
            reader.wait_loaded()
        assert reader[3]["id"] == 3

    def test_load_async_noop_if_loaded(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet")
        reader = ParquetReader(shard)
        _ = reader[0]
        with ThreadPoolExecutor(max_workers=1) as ex:
            reader.load_async(ex)  # already loaded; should be no-op
        assert reader._future is None

    def test_wait_loaded_sync_fallback(self, tmp_path):
        shard = _write_shard(tmp_path / "shard.00000.parquet", n=3)
        reader = ParquetReader(shard)
        reader.wait_loaded()  # no async submitted; triggers sync load
        assert reader.is_loaded
