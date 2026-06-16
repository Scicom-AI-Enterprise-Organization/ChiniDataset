"""Tests for chinidataset/dataset/cache.py (CacheManager)."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chinidataset.dataset.cache import CacheManager, ShardInfo, ShardState


def _make_shard(shard_id, basename, local_dir, size_bytes=1024, num_samples=10):
    return ShardInfo(
        shard_id=shard_id,
        basename=basename,
        num_samples=num_samples,
        size_bytes=size_bytes,
        local_dir=local_dir,
    )


def _write_file(path: Path, content: bytes = b"x" * 1024):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


class TestCacheManagerInit:
    def test_local_shard_detected_on_init(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path)
        _write_file(shard.local_path)

        cache = CacheManager(local=tmp_path, remote=None, shards=[shard])
        assert cache.is_local(0)

    def test_missing_shard_starts_remote(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path)
        cache = CacheManager(local=tmp_path, remote=None, shards=[shard])
        assert not cache.is_local(0)
        assert cache._states[0] == ShardState.REMOTE

    def test_cache_usage_counts_existing_files(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path, size_bytes=512)
        _write_file(shard.local_path, b"x" * 512)

        cache = CacheManager(local=tmp_path, remote=None, shards=[shard])
        assert cache._cache_usage == 512


class TestEnsureLocal:
    def test_local_path_returned_immediately(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path)
        _write_file(shard.local_path)

        cache = CacheManager(local=tmp_path, remote=None, shards=[shard])
        path = cache.ensure_local(0)
        assert path == shard.local_path

    def test_no_remote_raises_for_missing_shard(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path)
        cache = CacheManager(local=tmp_path, remote=None, shards=[shard])
        with pytest.raises(RuntimeError, match="not found locally"):
            cache.ensure_local(0)

    def test_state_becomes_local_after_download(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path)
        src = tmp_path / "remote" / "shard.00000.parquet"
        _write_file(src)

        cache = CacheManager(local=tmp_path / "cache", remote=str(tmp_path / "remote"), shards=[
            _make_shard(0, "shard.00000.parquet", tmp_path / "cache")
        ])
        cache.ensure_local(0)
        assert cache.is_local(0)

    def test_download_failure_resets_to_remote(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path)
        cache = CacheManager(local=tmp_path, remote="http://does.not.exist.invalid/", shards=[shard])
        with pytest.raises(RuntimeError):
            cache.ensure_local(0)
        assert cache._states[0] == ShardState.REMOTE


class TestLRUEviction:
    def test_evicts_coldest_when_full(self, tmp_path):
        cache_dir = tmp_path / "cache"
        size = 500  # bytes per shard

        shards = [
            _make_shard(i, f"shard.{i:05d}.parquet", cache_dir, size_bytes=size)
            for i in range(3)
        ]
        # Pre-write shards 0 and 1 as if they are already cached
        for s in shards[:2]:
            _write_file(s.local_path, b"x" * size)

        cache = CacheManager(local=cache_dir, remote=None, shards=shards, cache_limit=size * 2)
        assert cache._cache_usage == size * 2

        # Touch shard 1 to make it more recent
        time.sleep(0.01)
        cache.touch(1)

        freed = cache._evict_coldest()
        assert freed == size
        assert not cache.is_local(0)  # shard 0 (coldest) was evicted
        assert cache.is_local(1)      # shard 1 (recently touched) kept

    def test_eviction_removes_file(self, tmp_path):
        cache_dir = tmp_path / "cache"
        shard = _make_shard(0, "shard.00000.parquet", cache_dir, size_bytes=256)
        _write_file(shard.local_path, b"x" * 256)

        cache = CacheManager(local=cache_dir, remote=None, shards=[shard])
        assert shard.local_path.exists()

        cache._evict_coldest()
        assert not shard.local_path.exists()

    def test_evict_nothing_when_no_local_shards(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path, size_bytes=100)
        cache = CacheManager(local=tmp_path, remote=None, shards=[shard])
        freed = cache._evict_coldest()
        assert freed == 0


class TestTouch:
    def test_touch_updates_access_time(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path)
        _write_file(shard.local_path)
        cache = CacheManager(local=tmp_path, remote=None, shards=[shard])

        old_time = cache._access_times[0]
        time.sleep(0.01)
        cache.touch(0)
        assert cache._access_times[0] > old_time


class TestIsLocal:
    def test_is_local_true(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path)
        _write_file(shard.local_path)
        cache = CacheManager(local=tmp_path, remote=None, shards=[shard])
        assert cache.is_local(0) is True

    def test_is_local_false(self, tmp_path):
        shard = _make_shard(0, "shard.00000.parquet", tmp_path)
        cache = CacheManager(local=tmp_path, remote=None, shards=[shard])
        assert cache.is_local(0) is False
