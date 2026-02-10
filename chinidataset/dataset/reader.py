"""ParquetReader: Read samples from Parquet shard files.

Uses pandas as the read backend for fast bulk loading, then converts
to a list of dicts for O(1) per-sample access.
"""

import threading
from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any, Optional

import pandas as pd

__all__ = ['ParquetReader']


class ParquetReader:
    """Reads individual samples from a Parquet shard file.

    Uses ``pd.read_parquet`` + ``to_dict(orient='records')`` for fast
    bulk loading (~30x faster than per-element Arrow access).  Each
    sample is a plain Python dict ready for DataLoader collation.

    Supports background loading via :meth:`load_async` for look-ahead
    caching -- the next N shards can be loaded in a thread pool while
    the current shard is being iterated.

    Args:
        path (Path): Path to the .parquet shard file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: list[dict[str, Any]] = []
        self._num_rows: int = 0
        self._loaded: bool = False
        self._lock = threading.Lock()
        self._future: Optional[Future] = None

    def _load(self) -> None:
        """Lazy-load the Parquet file into a list of record dicts.

        Thread-safe: uses a lock to prevent duplicate loading when called
        from both the main thread and a background prefetch thread.
        """
        with self._lock:
            if self._loaded:
                return

            df = pd.read_parquet(str(self.path))
            self._records = df.to_dict(orient='records')
            self._num_rows = len(self._records)
            del df
            self._loaded = True

    def load_async(self, executor: Executor) -> None:
        """Submit ``_load`` to a background thread pool.

        No-op if the shard is already loaded or a load is in progress.

        Args:
            executor: A :class:`concurrent.futures.Executor` to submit
                the load task to.
        """
        if self._loaded or self._future is not None:
            return
        self._future = executor.submit(self._load)

    def wait_loaded(self) -> None:
        """Block until background loading completes, or load synchronously.

        If :meth:`load_async` was called, waits for the background future.
        Otherwise, triggers a normal synchronous ``_load``.
        """
        if self._future is not None:
            self._future.result()  # block; propagates exceptions
            self._future = None
        else:
            self._load()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample by index.

        Args:
            idx (int): Row index within this shard.

        Returns:
            Dict[str, Any]: Sample dictionary.
        """
        self.wait_loaded()
        if idx < 0:
            idx += self._num_rows
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(f'Index {idx} out of range for shard with {self._num_rows} rows')
        return self._records[idx]

    def __len__(self) -> int:
        """Number of samples in this shard."""
        self.wait_loaded()
        return self._num_rows

    def unload(self) -> None:
        """Release data from memory.

        If a background load is in progress, attempts to cancel it.
        If it cannot be cancelled (already running), waits for it to
        finish before clearing.
        """
        if self._future is not None:
            self._future.cancel()
            if not self._future.cancelled():
                self._future.result()
            self._future = None
        with self._lock:
            self._records.clear()
            self._loaded = False
            self._num_rows = 0

    @property
    def is_loaded(self) -> bool:
        """Whether the shard data is loaded in memory."""
        return self._loaded
