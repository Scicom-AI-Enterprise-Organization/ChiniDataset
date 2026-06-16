"""Tests for chinidataset/dataset/streaming.py (StreamingDataset)."""

import json
import numpy as np
import pytest

from chinidataset.dataset.streaming import StreamingDataset
from chinidataset.writer.parquet import ParquetWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dataset(path, n=100, size_limit="1kb"):
    """Write n simple samples to path and return the directory."""
    columns = {"id": "int64", "value": "float64"}
    path.mkdir(parents=True, exist_ok=True)
    with ParquetWriter(out=path, columns=columns, size_limit=size_limit) as w:
        for i in range(n):
            w.write({"id": i, "value": float(i) * 0.1})
    return path


def _all_ids(dataset):
    """Iterate a StreamingDataset and return a sorted list of 'id' values."""
    return sorted(s["id"] for s in dataset)


# ---------------------------------------------------------------------------
# Basic iteration
# ---------------------------------------------------------------------------

class TestStreamingDatasetIteration:
    def test_iterates_all_samples(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=50)
        ds = StreamingDataset(local=out)
        assert _all_ids(ds) == list(range(50))

    def test_len(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=30)
        ds = StreamingDataset(local=out)
        assert len(ds) == 30

    def test_sample_fields_present(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=5)
        ds = StreamingDataset(local=out)
        for sample in ds:
            assert "id" in sample
            assert "value" in sample

    def test_column_values_correct(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=10)
        ds = StreamingDataset(local=out)
        samples = {s["id"]: s for s in ds}
        for i in range(10):
            assert abs(samples[i]["value"] - i * 0.1) < 1e-9

    def test_multiple_shards(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=80, size_limit="100b")
        ds = StreamingDataset(local=out)
        assert ds.num_shards > 1
        assert _all_ids(ds) == list(range(80))

    def test_restart_iteration(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=20)
        ds = StreamingDataset(local=out)
        first = _all_ids(ds)
        second = _all_ids(ds)
        assert first == second


# ---------------------------------------------------------------------------
# Random access (__getitem__)
# ---------------------------------------------------------------------------

class TestStreamingDatasetGetItem:
    def test_getitem_first(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=10)
        ds = StreamingDataset(local=out)
        sample = ds[0]
        assert sample["id"] == 0

    def test_getitem_last(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=10)
        ds = StreamingDataset(local=out)
        sample = ds[9]
        assert sample["id"] == 9

    def test_getitem_middle(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=20)
        ds = StreamingDataset(local=out)
        sample = ds[10]
        assert sample["id"] == 10

    def test_getitem_out_of_bounds(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=5)
        ds = StreamingDataset(local=out)
        with pytest.raises((IndexError, Exception)):
            _ = ds[100]


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------

class TestStreamingDatasetShuffle:
    def test_shuffle_covers_all_samples(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=60)
        ds = StreamingDataset(local=out, shuffle=True, shuffle_seed=42)
        assert _all_ids(ds) == list(range(60))

    def test_shuffle_deterministic(self, tmp_path):
        # Each __iter__ call increments the epoch, so compare two fresh instances
        # at epoch 0 rather than iterating the same instance twice.
        out = _write_dataset(tmp_path / "ds", n=50)
        ds_a = StreamingDataset(local=out, shuffle=True, shuffle_seed=99)
        ds_b = StreamingDataset(local=out, shuffle=True, shuffle_seed=99)
        ids_a = [s["id"] for s in ds_a]
        ids_b = [s["id"] for s in ds_b]
        assert ids_a == ids_b

    def test_no_shuffle_is_ordered(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=40, size_limit="10kb")
        ds = StreamingDataset(local=out, shuffle=False)
        ids = [s["id"] for s in ds]
        assert ids == list(range(40))


# ---------------------------------------------------------------------------
# Distributed / worker simulation (env var patching)
# ---------------------------------------------------------------------------

class TestStreamingDatasetDistributed:
    def test_two_workers_no_overlap(self, tmp_path, monkeypatch):
        out = _write_dataset(tmp_path / "ds", n=100)

        # Simulate worker 0 of 2
        monkeypatch.delenv("WORLD_SIZE", raising=False)

        import chinidataset.dataset.streaming as _streaming_mod
        import chinidataset.dataset.world as _world_mod
        from chinidataset.dataset.world import World

        def mock_w0():
            return World(num_nodes=1, node=0, num_ranks=1, rank=0,
                         ranks_per_node=1, rank_of_node=0,
                         num_workers=2, worker_of_rank=0)

        def mock_w1():
            return World(num_nodes=1, node=0, num_ranks=1, rank=0,
                         ranks_per_node=1, rank_of_node=0,
                         num_workers=2, worker_of_rank=1)

        ds = StreamingDataset(local=out)

        # Monkey-patch _rank_world.detect_workers on the dataset instance
        ds._rank_world = World(num_nodes=1, node=0, num_ranks=1, rank=0,
                               ranks_per_node=1, rank_of_node=0)

        original_detect = ds._rank_world.detect_workers

        ds._rank_world.detect_workers = mock_w0
        ids_w0 = [s["id"] for s in ds]

        ds._rank_world.detect_workers = mock_w1
        ids_w1 = [s["id"] for s in ds]

        assert len(set(ids_w0) & set(ids_w1)) == 0

    def test_two_workers_full_coverage(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=100)
        from chinidataset.dataset.world import World

        ds = StreamingDataset(local=out)
        ds._rank_world = World(num_nodes=1, node=0, num_ranks=1, rank=0,
                               ranks_per_node=1, rank_of_node=0)

        def mock_w0():
            return World(num_nodes=1, node=0, num_ranks=1, rank=0,
                         ranks_per_node=1, rank_of_node=0,
                         num_workers=2, worker_of_rank=0)

        def mock_w1():
            return World(num_nodes=1, node=0, num_ranks=1, rank=0,
                         ranks_per_node=1, rank_of_node=0,
                         num_workers=2, worker_of_rank=1)

        ds._rank_world.detect_workers = mock_w0
        ids_w0 = [s["id"] for s in ds]

        ds._rank_world.detect_workers = mock_w1
        ids_w1 = [s["id"] for s in ds]

        combined = sorted(ids_w0 + ids_w1)
        assert combined == list(range(100))


# ---------------------------------------------------------------------------
# LRU shard cache
# ---------------------------------------------------------------------------

class TestStreamingDatasetLRU:
    def test_lru_correct_values_with_small_cache(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=60, size_limit="200b")
        ds = StreamingDataset(local=out, max_open_shards=2)
        assert _all_ids(ds) == list(range(60))

    def test_open_shard_count_bounded(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=80, size_limit="200b")
        ds = StreamingDataset(local=out, max_open_shards=3)
        for _ in ds:
            assert len(ds._readers) <= 3


# ---------------------------------------------------------------------------
# Look-ahead prefetching
# ---------------------------------------------------------------------------

class TestStreamingDatasetLookAhead:
    def test_look_ahead_all_samples(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=60, size_limit="200b")
        ds = StreamingDataset(local=out, look_ahead=2, max_open_shards=6)
        assert _all_ids(ds) == list(range(60))

    def test_look_ahead_zero_all_samples(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=40)
        ds = StreamingDataset(local=out, look_ahead=0)
        assert _all_ids(ds) == list(range(40))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestStreamingDatasetEdgeCases:
    def test_single_shard(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=5, size_limit="100mb")
        ds = StreamingDataset(local=out)
        assert ds.num_shards == 1
        assert _all_ids(ds) == list(range(5))

    def test_single_sample(self, tmp_path):
        out = _write_dataset(tmp_path / "ds", n=1)
        ds = StreamingDataset(local=out)
        assert len(ds) == 1
        assert _all_ids(ds) == [0]

    def test_missing_index_raises(self, tmp_path):
        out = tmp_path / "empty"
        out.mkdir()
        with pytest.raises(FileNotFoundError):
            StreamingDataset(local=out)

    def test_split_appended_to_local(self, tmp_path):
        train_dir = tmp_path / "data" / "train"
        _write_dataset(train_dir, n=10)
        ds = StreamingDataset(local=tmp_path / "data", split="train")
        assert len(ds) == 10
