"""Tests for chinidataset/dataset/partition.py."""

import numpy as np
import pytest

from chinidataset.dataset.partition import get_partition
from chinidataset.dataset.world import World


def _world(num_ranks=1, rank=0, num_workers=1, worker_of_rank=0):
    return World(
        num_nodes=1,
        node=0,
        num_ranks=num_ranks,
        rank=rank,
        ranks_per_node=num_ranks,
        rank_of_node=rank,
        num_workers=num_workers,
        worker_of_rank=worker_of_rank,
    )


class TestGetPartition:
    def test_single_worker_returns_all(self):
        ids = np.arange(100, dtype=np.int64)
        result = get_partition(ids, _world())
        np.testing.assert_array_equal(result, ids)

    def test_two_ranks_disjoint(self):
        ids = np.arange(100, dtype=np.int64)
        p0 = get_partition(ids, _world(num_ranks=2, rank=0))
        p1 = get_partition(ids, _world(num_ranks=2, rank=1))
        assert len(np.intersect1d(p0, p1)) == 0

    def test_two_ranks_full_coverage(self):
        ids = np.arange(100, dtype=np.int64)
        p0 = get_partition(ids, _world(num_ranks=2, rank=0))
        p1 = get_partition(ids, _world(num_ranks=2, rank=1))
        combined = np.sort(np.concatenate([p0, p1]))
        np.testing.assert_array_equal(combined, ids)

    def test_two_workers_disjoint(self):
        ids = np.arange(100, dtype=np.int64)
        p0 = get_partition(ids, _world(num_workers=2, worker_of_rank=0))
        p1 = get_partition(ids, _world(num_workers=2, worker_of_rank=1))
        assert len(np.intersect1d(p0, p1)) == 0

    def test_two_workers_full_coverage(self):
        ids = np.arange(100, dtype=np.int64)
        p0 = get_partition(ids, _world(num_workers=2, worker_of_rank=0))
        p1 = get_partition(ids, _world(num_workers=2, worker_of_rank=1))
        combined = np.sort(np.concatenate([p0, p1]))
        np.testing.assert_array_equal(combined, ids)

    def test_full_distributed_no_overlap(self):
        ids = np.arange(200, dtype=np.int64)
        partitions = [
            get_partition(ids, _world(num_ranks=2, rank=r, num_workers=2, worker_of_rank=w))
            for r in range(2) for w in range(2)
        ]
        all_ids = np.concatenate(partitions)
        assert len(all_ids) == len(np.unique(all_ids)), "Duplicate sample IDs across workers"

    def test_full_distributed_full_coverage(self):
        ids = np.arange(200, dtype=np.int64)
        partitions = [
            get_partition(ids, _world(num_ranks=2, rank=r, num_workers=2, worker_of_rank=w))
            for r in range(2) for w in range(2)
        ]
        combined = np.sort(np.concatenate(partitions))
        np.testing.assert_array_equal(combined, ids)

    def test_uneven_sample_count(self):
        ids = np.arange(101, dtype=np.int64)
        p0 = get_partition(ids, _world(num_workers=2, worker_of_rank=0))
        p1 = get_partition(ids, _world(num_workers=2, worker_of_rank=1))
        combined = np.sort(np.concatenate([p0, p1]))
        np.testing.assert_array_equal(combined, ids)

    def test_deterministic(self):
        ids = np.arange(50, dtype=np.int64)
        w = _world(num_ranks=2, rank=0)
        assert np.array_equal(get_partition(ids, w), get_partition(ids, w))

    def test_empty_sample_ids(self):
        ids = np.array([], dtype=np.int64)
        result = get_partition(ids, _world())
        assert len(result) == 0
