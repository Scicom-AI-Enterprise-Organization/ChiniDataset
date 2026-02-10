"""ParquetReader: Read samples from Parquet shard files.

Uses pandas as the read backend for fast bulk loading, then converts
to a list of dicts for O(1) per-sample access.
"""

from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ['ParquetReader']


class ParquetReader:
    """Reads individual samples from a Parquet shard file.

    Uses ``pd.read_parquet`` + ``to_dict(orient='records')`` for fast
    bulk loading (~30x faster than per-element Arrow access).  Each
    sample is a plain Python dict ready for DataLoader collation.

    Args:
        path (Path): Path to the .parquet shard file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: list[dict[str, Any]] = []
        self._num_rows: int = 0
        self._loaded: bool = False

    def _load(self) -> None:
        """Lazy-load the Parquet file into a list of record dicts."""
        if self._loaded:
            return

        df = pd.read_parquet(str(self.path))
        self._records = df.to_dict(orient='records')
        self._num_rows = len(self._records)
        del df
        self._loaded = True

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample by index.

        Args:
            idx (int): Row index within this shard.

        Returns:
            Dict[str, Any]: Sample dictionary.
        """
        self._load()
        if idx < 0:
            idx += self._num_rows
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(f'Index {idx} out of range for shard with {self._num_rows} rows')
        return self._records[idx]

    def __len__(self) -> int:
        """Number of samples in this shard."""
        self._load()
        return self._num_rows

    def unload(self) -> None:
        """Release data from memory."""
        self._records.clear()
        self._loaded = False
        self._num_rows = 0

    @property
    def is_loaded(self) -> bool:
        """Whether the shard data is loaded in memory."""
        return self._loaded
