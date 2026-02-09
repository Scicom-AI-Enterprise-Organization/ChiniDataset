"""ParquetReader: Read samples from Parquet shard files.

Returns numpy arrays for numeric/array columns (fast IPC serialization)
instead of Python lists (slow per-element serialization).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ['ParquetReader']


class ParquetReader:
    """Reads individual samples from a Parquet shard file.

    Loads the Parquet file into per-column numpy arrays on first access.
    Returns numpy arrays for numeric and array columns, which serialize
    efficiently through DataLoader worker pipes (single buffer copy vs
    per-element Python object serialization).

    Args:
        path (Path): Path to the .parquet shard file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._columns: dict[str, Any] = {}
        self._num_rows: int = 0
        self._loaded: bool = False

    def _load(self) -> None:
        """Lazy-load the Parquet file into per-column numpy/list storage."""
        if self._loaded:
            return

        table = pq.read_table(str(self.path))
        self._num_rows = table.num_rows

        for col_name in table.column_names:
            col = table.column(col_name)
            col_type = col.type

            if pa.types.is_list(col_type):
                # List column: store as list of numpy arrays
                # Each element is a numpy array (fast IPC)
                value_type = col_type.value_type
                np_dtype = self._arrow_to_numpy_dtype(value_type)
                arrays = []
                for i in range(self._num_rows):
                    val = col[i].as_py()
                    if val is not None:
                        arrays.append(np.array(val, dtype=np_dtype))
                    else:
                        arrays.append(np.array([], dtype=np_dtype))
                self._columns[col_name] = arrays

            elif pa.types.is_integer(col_type) or pa.types.is_floating(col_type):
                # Numeric scalar: store as numpy array (bulk access)
                self._columns[col_name] = col.to_numpy(zero_copy_only=False)

            elif pa.types.is_boolean(col_type):
                self._columns[col_name] = col.to_numpy(zero_copy_only=False)

            elif pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
                # String: store as Python list (no numpy equivalent)
                self._columns[col_name] = col.to_pylist()

            elif pa.types.is_binary(col_type) or pa.types.is_large_binary(col_type):
                self._columns[col_name] = col.to_pylist()

            else:
                # Fallback: Python list
                self._columns[col_name] = col.to_pylist()

        # Free the Arrow Table (we've extracted everything into numpy)
        del table
        self._loaded = True

    @staticmethod
    def _arrow_to_numpy_dtype(arrow_type: pa.DataType) -> np.dtype:
        """Map Arrow type to numpy dtype."""
        mapping = {
            pa.int8(): np.int8,
            pa.int16(): np.int16,
            pa.int32(): np.int32,
            pa.int64(): np.int64,
            pa.uint8(): np.uint8,
            pa.uint16(): np.uint16,
            pa.uint32(): np.uint32,
            pa.uint64(): np.uint64,
            pa.float16(): np.float16,
            pa.float32(): np.float32,
            pa.float64(): np.float64,
        }
        return mapping.get(arrow_type, np.float64)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample by index.

        Returns numpy arrays for numeric/array columns (fast IPC).

        Args:
            idx (int): Row index within this shard.

        Returns:
            Dict[str, Any]: Sample dictionary.
        """
        self._load()
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(f'Index {idx} out of range for shard with {self._num_rows} rows')

        sample = {}
        for col_name, col_data in self._columns.items():
            val = col_data[idx]
            # Scalar numpy values: extract as Python scalar for collation
            if isinstance(val, (np.integer, np.floating, np.bool_)):
                sample[col_name] = val.item()
            else:
                sample[col_name] = val
        return sample

    def __len__(self) -> int:
        """Number of samples in this shard."""
        self._load()
        return self._num_rows

    def unload(self) -> None:
        """Release data from memory."""
        self._columns.clear()
        self._loaded = False
        self._num_rows = 0

    @property
    def is_loaded(self) -> bool:
        """Whether the shard data is loaded in memory."""
        return self._loaded
