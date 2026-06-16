"""Tests for chinidataset/dataset/shuffle.py."""

import numpy as np
import pytest

from chinidataset.dataset.shuffle import no_shuffle, shuffle_samples


class TestShuffleSamples:
    def test_same_seed_same_order(self):
        a = shuffle_samples(1000, block_size=256, seed=42, epoch=0)
        b = shuffle_samples(1000, block_size=256, seed=42, epoch=0)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_different_order(self):
        a = shuffle_samples(1000, block_size=256, seed=0, epoch=0)
        b = shuffle_samples(1000, block_size=256, seed=1, epoch=0)
        assert not np.array_equal(a, b)

    def test_different_epochs_different_order(self):
        a = shuffle_samples(1000, block_size=256, seed=42, epoch=0)
        b = shuffle_samples(1000, block_size=256, seed=42, epoch=1)
        assert not np.array_equal(a, b)

    def test_is_permutation(self):
        result = shuffle_samples(500, block_size=128, seed=7, epoch=0)
        assert sorted(result.tolist()) == list(range(500))

    def test_no_duplicates(self):
        result = shuffle_samples(500, block_size=128, seed=7, epoch=0)
        assert len(result) == len(np.unique(result))

    def test_correct_length(self):
        result = shuffle_samples(300, block_size=100, seed=1, epoch=0)
        assert len(result) == 300

    def test_custom_block_size(self):
        result = shuffle_samples(200, block_size=50, seed=3, epoch=0)
        assert sorted(result.tolist()) == list(range(200))

    def test_block_size_larger_than_samples(self):
        result = shuffle_samples(10, block_size=1000, seed=1, epoch=0)
        assert sorted(result.tolist()) == list(range(10))

    def test_single_sample(self):
        result = shuffle_samples(1, block_size=256, seed=0, epoch=0)
        assert result.tolist() == [0]

    def test_empty(self):
        result = shuffle_samples(0, block_size=256, seed=0, epoch=0)
        assert len(result) == 0

    def test_dtype_is_int64(self):
        result = shuffle_samples(100, block_size=64, seed=0, epoch=0)
        assert result.dtype == np.int64

    def test_epoch_changes_produce_different_orderings(self):
        orderings = [shuffle_samples(200, block_size=64, seed=5, epoch=e) for e in range(5)]
        # All epochs should be distinct permutations
        for i in range(len(orderings)):
            for j in range(i + 1, len(orderings)):
                assert not np.array_equal(orderings[i], orderings[j])


class TestNoShuffle:
    def test_sequential_order(self):
        result = no_shuffle(10)
        np.testing.assert_array_equal(result, np.arange(10, dtype=np.int64))

    def test_empty(self):
        result = no_shuffle(0)
        assert len(result) == 0

    def test_dtype(self):
        result = no_shuffle(5)
        assert result.dtype == np.int64
