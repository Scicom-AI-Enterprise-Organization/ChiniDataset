"""CacheManager: Shard downloading, caching, and LRU eviction.

Simple single-process implementation using plain Python dicts. Fast, no
SharedMemory or FileLock overhead. Suitable for num_workers=0 (which is
what HuggingFace Trainer uses by default).

For multi-worker support (num_workers > 0), this can be upgraded to use
SharedMemory + FileLock for cross-process coordination.
"""

import logging
import shutil
import time
from enum import IntEnum
from pathlib import Path
from typing import Optional

__all__ = ['CacheManager', 'ShardState']

logger = logging.getLogger(__name__)


class ShardState(IntEnum):
    """Download state of a shard."""
    REMOTE = 0
    DOWNLOADING = 1
    LOCAL = 2


class ShardInfo:
    """Metadata for a single shard."""

    def __init__(
        self,
        shard_id: int,
        basename: str,
        num_samples: int,
        size_bytes: int,
        local_dir: Path,
    ) -> None:
        self.shard_id = shard_id
        self.basename = basename
        self.num_samples = num_samples
        self.size_bytes = size_bytes
        self.local_path = local_dir / basename


class CacheManager:
    """Simple shard cache manager using plain Python dicts.

    No SharedMemory, no FileLock, no cross-process overhead. Works perfectly
    for the common case: num_workers=0 (single process).

    Args:
        local (Path): Local cache directory.
        remote (str, optional): Remote URL prefix for shard downloads.
        shards (list[ShardInfo]): Shard metadata from index.json.
        cache_limit (int, optional): Maximum cache size in bytes.
    """

    def __init__(
        self,
        local: Path,
        remote: Optional[str],
        shards: list[ShardInfo],
        cache_limit: Optional[int] = None,
    ) -> None:
        self.local = local
        self.remote = remote
        self.shards = shards
        self.cache_limit = cache_limit
        self.num_shards = len(shards)

        # Plain Python state -- fast, no IPC overhead
        self._states: list[int] = []
        self._access_times: list[float] = []
        self._cache_usage: int = 0

        # Initialize by scanning local filesystem
        for shard in self.shards:
            if shard.local_path.exists():
                self._states.append(ShardState.LOCAL)
                self._access_times.append(time.time())
                self._cache_usage += shard.size_bytes
            else:
                self._states.append(ShardState.REMOTE)
                self._access_times.append(0.0)

    def ensure_local(self, shard_id: int) -> Path:
        """Ensure a shard is available locally. Downloads if needed.

        Args:
            shard_id (int): Shard index.

        Returns:
            Path: Local path to the shard file.
        """
        shard = self.shards[shard_id]

        if self._states[shard_id] == ShardState.LOCAL:
            self._access_times[shard_id] = time.time()
            return shard.local_path

        # Evict if needed
        if self.cache_limit is not None:
            while self._cache_usage + shard.size_bytes > self.cache_limit:
                freed = self._evict_coldest()
                if freed == 0:
                    break

        # Download
        self._states[shard_id] = ShardState.DOWNLOADING
        try:
            self._download_shard(shard)
        except Exception as e:
            self._states[shard_id] = ShardState.REMOTE
            raise RuntimeError(f'Failed to download shard {shard.basename}: {e}') from e

        self._states[shard_id] = ShardState.LOCAL
        self._access_times[shard_id] = time.time()
        self._cache_usage += shard.size_bytes

        return shard.local_path

    def _download_file(self, remote_url: str, local_path: Path) -> None:
        """Download a single file from remote to local.

        Supports: hf://, http://, https://, local paths.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # HuggingFace Hub
        if remote_url.startswith('hf://'):
            self._download_hf(remote_url, local_path)
            return

        # HTTP/HTTPS
        if remote_url.startswith(('http://', 'https://')):
            import urllib.request
            urllib.request.urlretrieve(remote_url, str(local_path))
            return

        # Local path copy
        src = Path(remote_url)
        if src.exists():
            shutil.copy2(src, local_path)
            return

        raise FileNotFoundError(f'Cannot download: {remote_url}')

    def _download_hf(self, remote_url: str, local_path: Path) -> None:
        """Download from HuggingFace Hub. URL format: hf://user/repo/path/to/file"""
        from huggingface_hub import hf_hub_download

        path = remote_url[len('hf://'):]
        parts = path.split('/', 2)
        if len(parts) < 3:
            raise ValueError(f'Invalid HuggingFace URL: {remote_url}')

        repo_id = f'{parts[0]}/{parts[1]}'
        filename = parts[2]

        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset')
        shutil.copy2(downloaded, local_path)

    def _download_shard(self, shard: ShardInfo) -> None:
        """Download a shard from remote to local."""
        if self.remote is None:
            raise FileNotFoundError(
                f'Shard {shard.basename} not found locally and no remote configured.'
            )
        remote_url = f'{self.remote.rstrip("/")}/{shard.basename}'
        self._download_file(remote_url, shard.local_path)

    def _evict_coldest(self) -> int:
        """Evict the least recently used local shard.

        Returns:
            int: Bytes freed (0 if nothing to evict).
        """
        coldest_id = None
        coldest_time = float('inf')

        for i, (state, atime) in enumerate(zip(self._states, self._access_times)):
            if state == ShardState.LOCAL and atime < coldest_time:
                coldest_time = atime
                coldest_id = i

        if coldest_id is None:
            return 0

        shard = self.shards[coldest_id]
        if shard.local_path.exists():
            shard.local_path.unlink()
        self._states[coldest_id] = ShardState.REMOTE
        self._access_times[coldest_id] = 0.0
        self._cache_usage -= shard.size_bytes
        if self._cache_usage < 0:
            self._cache_usage = 0

        return shard.size_bytes

    def touch(self, shard_id: int) -> None:
        """Update access time for LRU tracking."""
        self._access_times[shard_id] = time.time()

    def is_local(self, shard_id: int) -> bool:
        """Check if shard is local."""
        return self._states[shard_id] == ShardState.LOCAL
