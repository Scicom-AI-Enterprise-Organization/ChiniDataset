"""ParquetWriter writes samples to .parquet files compatible with MosaicML streaming.

This writer extends Writer directly (not JointWriter) because Parquet is a
columnar format that requires batch conversion -- samples cannot be individually
encoded to bytes like MDS/JSON/CSV.
"""

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chinidataset.hashing import get_hash
from chinidataset.util import merge_index
from chinidataset.writer.base import Writer

__all__ = ['ParquetWriter']

logger = logging.getLogger(__name__)

# Default Parquet settings
DEFAULT_ROW_GROUP_SIZE = 10000
DEFAULT_DATA_PAGE_SIZE = 1024 * 1024  # 1MB


def _write_partition_worker(args: tuple) -> None:
    """Worker function for parallel ``write_dataset``. Runs in a subprocess.

    Each worker creates its own ``ParquetWriter`` for an isolated subdirectory,
    iterates over its partition of the dataset, applies the optional transform,
    and writes samples. The writer's context manager ensures ``finish()`` is
    called to flush remaining samples and write the partition's ``index.json``.

    Args:
        args: Tuple of (sub_dir, dataset, start, end, columns, writer_kwargs, transform).
    """
    sub_dir, dataset, start, end, columns, writer_kwargs, transform = args
    with ParquetWriter(out=sub_dir, columns=columns, **writer_kwargs) as w:
        for i in range(start, end):
            sample = dataset[i]
            if transform is not None:
                sample = transform(sample)
            w.write(sample)


class ParquetWriter(Writer):
    """Writes a streaming Parquet dataset compatible with MosaicML streaming.

    Parquet is a columnar storage format that provides:
    - Efficient compression and encoding
    - Self-describing schema
    - Wide ecosystem support (Spark, Arrow, pandas, etc.)
    - Predicate pushdown for query optimization

    Unlike MDSWriter which creates binary .mds files, ParquetWriter creates
    standard .parquet files that can be read by any Parquet-compatible library.

    Note: ParquetWriter extends Writer directly (not JointWriter or SplitWriter)
    because Parquet is a columnar format that requires batch conversion of all
    buffered samples into an Arrow Table. Individual sample-to-bytes encoding
    (as used by MDS/JSON/CSV) does not apply.

    Args:
        columns (Dict[str, str]): Sample columns mapping column name to type.
            Supported types: int8, int16, int32, int64, uint8, uint16, uint32, uint64,
            float16, float32, float64, str, bytes, bool.
            Array types: int64[], float32[], etc.
        out (str | Path): Output dataset directory to save shard files.
        compression (str, optional): Parquet compression codec. Defaults to ``None``
            (no compression).
        compression_level (int, optional): Compression level. Defaults to codec default.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard
            files. Defaults to ``None``.
        size_limit (Union[int, str], optional): Optional shard size limit.
            Defaults to ``1 << 26`` (64MB).
        row_group_size (int, optional): Number of rows per row group.
            Defaults to ``10000``.
        exist_ok (bool): If the local directory exists and is not empty, whether to overwrite.
            Defaults to ``False``.

    Example:
        >>> from chinidataset import ParquetWriter
        >>> columns = {'tokens': 'int64[]', 'label': 'int32'}
        >>> with ParquetWriter(out='./data', columns=columns) as writer:
        ...     writer.write({'tokens': [1, 2, 3, 4], 'label': 0})
    """

    format = 'parquet'

    def __init__(
        self,
        *,
        columns: dict[str, str],
        out: Union[str, Path],
        compression: Optional[str] = None,
        compression_level: Optional[int] = None,
        hashes: Optional[list[str]] = None,
        size_limit: Optional[Union[int, str]] = 1 << 26,
        row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        exist_ok: bool = False,
    ) -> None:
        # Parquet handles compression internally via pyarrow.
        # Hashing is still supported at the file level.
        super().__init__(
            out=out,
            hashes=hashes,
            size_limit=size_limit,
            exist_ok=exist_ok,
        )

        self.columns = columns
        self.parquet_compression = compression
        self.compression_level = compression_level or self._default_compression_level(compression)
        self.row_group_size = row_group_size

        # Parse column types
        self._col_is_array = {col: typ.endswith('[]') for col, typ in columns.items()}
        self._col_base_type = {
            col: typ[:-2] if typ.endswith('[]') else typ
            for col, typ in columns.items()
        }

        # Pre-build Arrow schema
        self._schema = self._build_schema()

        # Store raw samples for batch conversion to Arrow
        self._raw_samples: list[dict[str, Any]] = []

    def _default_compression_level(self, compression: Optional[str]) -> int:
        """Get default compression level for codec."""
        # Snappy has no compression levels
        return 0

    def _build_schema(self) -> pa.Schema:
        """Build PyArrow schema from column definitions."""
        fields = []
        for col_name, col_type in self.columns.items():
            arrow_type = self._parse_column_type(col_type)
            fields.append(pa.field(col_name, arrow_type))
        return pa.schema(fields)

    def _parse_column_type(self, col_type: str) -> pa.DataType:
        """Parse column type string to PyArrow data type."""
        type_map = {
            'int8': pa.int8(),
            'int16': pa.int16(),
            'int32': pa.int32(),
            'int64': pa.int64(),
            'uint8': pa.uint8(),
            'uint16': pa.uint16(),
            'uint32': pa.uint32(),
            'uint64': pa.uint64(),
            'float16': pa.float16(),
            'float32': pa.float32(),
            'float64': pa.float64(),
            'str': pa.string(),
            'string': pa.string(),
            'bytes': pa.binary(),
            'bool': pa.bool_(),
        }

        # Handle array types (e.g., "int64[]")
        if col_type.endswith('[]'):
            base_type = col_type[:-2]
            if base_type in type_map:
                return pa.list_(type_map[base_type])
            raise ValueError(f'Unknown array element type: {base_type}')

        if col_type in type_map:
            return type_map[col_type]
        raise ValueError(f'Unknown column type: {col_type}')

    def encode_sample(self, sample: dict[str, Any]) -> bytes:
        """No-op for Parquet. Parquet requires batch columnar conversion.

        This method exists only to satisfy the abstract base class contract.
        The actual sample buffering happens in write().

        Args:
            sample (Dict[str, Any]): Sample dict (unused).

        Returns:
            bytes: Empty bytes.
        """
        # Do NOT append to _raw_samples here.
        # Parquet samples are buffered in write() and batch-converted in flush_shard().
        return b''

    def write(self, sample: dict[str, Any]) -> None:
        """Write a sample.

        Overrides base write() because Parquet cannot encode samples to bytes
        individually. Instead, raw sample dicts are buffered and batch-converted
        to an Arrow Table when flushing.

        Args:
            sample (Dict[str, Any]): Sample dict.
        """
        sample_size = self._estimate_sample_size(sample)

        if self.size_limit and self.size_limit < self.new_shard_size + sample_size:
            self.flush_shard()
            self._reset_cache()

        self._raw_samples.append(sample)
        self.new_samples.append(b'')  # Placeholder to track count
        self.new_shard_size += sample_size

    def _estimate_sample_size(self, sample: dict[str, Any]) -> int:
        """Estimate the size of a sample in bytes. Fast path for numpy arrays."""
        size = 0
        for col_name, value in sample.items():
            if isinstance(value, np.ndarray):
                # Fast: numpy knows its own byte size
                size += value.nbytes
            elif isinstance(value, (list, tuple)):
                col_type = self.columns.get(col_name, '')
                elem_size = 4
                if '64' in col_type:
                    elem_size = 8
                elif '16' in col_type:
                    elem_size = 2
                elif '8' in col_type:
                    elem_size = 1
                size += len(value) * elem_size
            elif isinstance(value, str):
                size += len(value) * 4  # estimate, avoid encode() overhead
            elif isinstance(value, bytes):
                size += len(value)
            else:
                size += 8
        return size

    def _samples_to_table(self) -> pa.Table:
        """Convert buffered samples to PyArrow Table."""
        arrays = {}
        for col_name in self.columns:
            values = [sample.get(col_name) for sample in self._raw_samples]
            arrays[col_name] = self._to_arrow_array(col_name, values)
        return pa.table(arrays, schema=self._schema)

    def _to_arrow_array(self, col_name: str, values: list) -> pa.Array:
        """Convert a list of values to a PyArrow Array.

        Handles numpy arrays natively (zero-copy where possible) to avoid
        the expensive .tolist() conversion.
        """
        col_type = self.columns[col_name]
        arrow_type = self._parse_column_type(col_type)

        # Handle array columns (int64[], float32[], etc.)
        if self._col_is_array.get(col_name, False) and values:
            first_val = values[0]
            base_type = self._col_base_type[col_name]
            if isinstance(first_val, np.ndarray):
                return self._build_list_array(values, base_type)
            elif isinstance(first_val, (list, tuple)):
                return self._build_list_array_from_lists(values, base_type)

        # Scalar columns: if all numpy scalars, extract efficiently
        if values and isinstance(values[0], (np.integer, np.floating)):
            return pa.array([v.item() for v in values], type=arrow_type)

        return pa.array(values, type=arrow_type)

    def _build_list_array(self, values: list, base_type: str) -> pa.ListArray:
        """Build ListArray from numpy arrays."""
        n = len(values)
        sizes = [arr.size if hasattr(arr, 'size') else len(arr) for arr in values]
        total_size = sum(sizes)

        offsets = np.empty(n + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum(sizes, out=offsets[1:], dtype=np.int32)

        np_dtype = self._get_numpy_dtype(base_type)
        flat_values = np.empty(total_size, dtype=np_dtype)

        offset = 0
        for arr in values:
            sz = arr.size if hasattr(arr, 'size') else len(arr)
            flat_values[offset:offset + sz] = arr.flat if hasattr(arr, 'flat') else arr
            offset += sz

        value_type = self._parse_column_type(base_type)
        return pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int32()),
            pa.array(flat_values, type=value_type),
        )

    def _build_list_array_from_lists(self, values: list, base_type: str) -> pa.ListArray:
        """Build ListArray from Python lists."""
        offsets = [0]
        for v in values:
            offsets.append(offsets[-1] + len(v))
        flat_values = [item for v in values for item in v]
        value_type = self._parse_column_type(base_type)
        return pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int32()),
            pa.array(flat_values, type=value_type),
        )

    def _get_numpy_dtype(self, type_str: str) -> np.dtype:
        """Get numpy dtype from type string."""
        type_map = {
            'int8': np.int8,
            'int16': np.int16,
            'int32': np.int32,
            'int64': np.int64,
            'uint8': np.uint8,
            'uint16': np.uint16,
            'uint32': np.uint32,
            'uint64': np.uint64,
            'float16': np.float16,
            'float32': np.float32,
            'float64': np.float64,
        }
        return type_map.get(type_str, np.int32)

    def get_config(self) -> dict[str, Any]:
        """Get object describing shard-writing configuration."""
        obj = super().get_config()
        obj.update({
            'column_names': list(self.columns.keys()),
            'column_encodings': [
                f'ndarray:{self._col_base_type[c]}' if self._col_is_array[c] else self.columns[c]
                for c in self.columns
            ],
            'column_sizes': [None] * len(self.columns),
            'compression': self.parquet_compression,
        })
        return obj

    def flush_shard(self) -> None:
        """Flush cached samples to storage, creating a new Parquet shard."""
        if not self._raw_samples:
            return

        # Convert samples to Arrow table
        table = self._samples_to_table()

        # Generate shard filename
        shard_idx = len(self.shards)
        shard_name = f'shard.{shard_idx:05}.parquet'
        shard_path = self.local / shard_name

        # Write Parquet file
        pq.write_table(
            table,
            shard_path,
            compression=self.parquet_compression,
            row_group_size=self.row_group_size,
            data_page_size=DEFAULT_DATA_PAGE_SIZE,
            write_statistics=True,
            use_dictionary=False,
        )

        # Compute file-level hashes if requested
        shard_bytes = shard_path.stat().st_size
        file_hashes = {}
        if self.hashes:
            with open(shard_path, 'rb') as f:
                file_data = f.read()
            for algo in self.hashes:
                file_hashes[algo] = get_hash(algo, file_data)

        # Record shard metadata
        obj = {
            'samples': len(self._raw_samples),
            'raw_data': {
                'basename': shard_name,
                'bytes': shard_bytes,
                'hashes': file_hashes,
            },
            'zip_data': None,
        }
        obj.update(self.get_config())
        self.shards.append(obj)

        # Reset sample buffer
        self._raw_samples = []

    def write_batch(self, data: dict[str, Any]) -> None:
        """Write a batch of samples from columnar data. Much faster than write().

        Accepts a dict of columns where each value is a numpy array or list
        with N elements. Converts directly to Arrow Table with zero-copy for
        numpy arrays, bypassing all per-sample overhead.

        For 100k samples with int64[512] arrays, write_batch() is 5-10x faster
        than calling write() in a loop because it skips:
        - 100k dict creations
        - 100k _estimate_sample_size() calls
        - 100k list appends
        - Column extraction list comprehensions at flush time

        Args:
            data (Dict[str, Any]): Columnar data. Each key is a column name,
                each value is a numpy array or list with N rows.

                For scalar columns: 1D array of shape (N,)
                For array columns: 2D array of shape (N, seq_len) or list of arrays

        Example:
            >>> writer.write_batch({
            ...     "input_ids": np.random.randint(0, 30000, (10000, 512), dtype=np.int64),
            ...     "attention_mask": np.ones((10000, 512), dtype=np.int64),
            ...     "label": np.arange(10000, dtype=np.int32),
            ... })
        """
        # Determine number of samples
        first_col = next(iter(data.values()))
        if isinstance(first_col, np.ndarray):
            n_samples = first_col.shape[0]
        else:
            n_samples = len(first_col)

        if n_samples == 0:
            return

        # Estimate bytes per sample for shard splitting
        # Use first column's nbytes as a fast proxy (avoids iterating all data)
        if isinstance(first_col, np.ndarray):
            total_nbytes = sum(
                v.nbytes if isinstance(v, np.ndarray) else n_samples * 8
                for v in data.values()
            )
        else:
            total_nbytes = n_samples * len(data) * 8  # rough estimate

        bytes_per_sample = total_nbytes / n_samples if n_samples else 0

        # Split into shard-sized chunks if needed
        if self.size_limit and total_nbytes > self.size_limit:
            samples_per_shard = max(1, int(self.size_limit / bytes_per_sample))
        else:
            samples_per_shard = n_samples

        # Write in shard-sized chunks (numpy slicing is zero-copy)
        offset = 0
        while offset < n_samples:
            chunk_end = min(offset + samples_per_shard, n_samples)
            chunk_data = {k: v[offset:chunk_end] for k, v in data.items()}
            self._flush_columnar(chunk_data, chunk_end - offset)
            offset = chunk_end

    def _flush_columnar(self, data: dict[str, Any], n_samples: int) -> None:
        """Convert columnar data directly to Arrow Table and write a shard.

        This is the fast path -- numpy arrays go to pa.array() with zero-copy.
        """
        # Build Arrow arrays from columnar data
        arrow_arrays = {}
        for col_name, col_data in data.items():
            arrow_arrays[col_name] = self._columnar_to_arrow(col_name, col_data)

        table = pa.table(arrow_arrays, schema=self._schema)

        # Write shard
        shard_idx = len(self.shards)
        shard_name = f'shard.{shard_idx:05}.parquet'
        shard_path = self.local / shard_name

        pq.write_table(
            table,
            shard_path,
            compression=self.parquet_compression,
            row_group_size=self.row_group_size,
            data_page_size=DEFAULT_DATA_PAGE_SIZE,
            write_statistics=True,
            use_dictionary=False,
        )

        # Hashes
        shard_bytes = shard_path.stat().st_size
        file_hashes = {}
        if self.hashes:
            with open(shard_path, 'rb') as f:
                file_data = f.read()
            for algo in self.hashes:
                file_hashes[algo] = get_hash(algo, file_data)

        # Record shard metadata
        obj = {
            'samples': n_samples,
            'raw_data': {
                'basename': shard_name,
                'bytes': shard_bytes,
                'hashes': file_hashes,
            },
            'zip_data': None,
        }
        obj.update(self.get_config())
        self.shards.append(obj)

    def _columnar_to_arrow(self, col_name: str, col_data: Any) -> pa.Array:
        """Convert a column of data to Arrow array. Zero-copy for numpy."""
        col_type = self.columns[col_name]
        arrow_type = self._parse_column_type(col_type)

        if self._col_is_array.get(col_name, False):
            base_type = self._col_base_type[col_name]
            value_type = self._parse_column_type(base_type)

            if isinstance(col_data, np.ndarray) and col_data.ndim == 2:
                # Fast path: 2D numpy array (N, seq_len) → ListArray
                # Zero-copy flatten + offsets
                n, seq_len = col_data.shape
                flat = col_data.reshape(-1)
                offsets = np.arange(0, (n + 1) * seq_len, seq_len, dtype=np.int32)
                return pa.ListArray.from_arrays(
                    pa.array(offsets, type=pa.int32()),
                    pa.array(flat, type=value_type),
                )
            elif isinstance(col_data, list) and col_data and isinstance(col_data[0], np.ndarray):
                # List of numpy arrays (variable length)
                return self._build_list_array(col_data, base_type)
            else:
                # List of lists
                return self._build_list_array_from_lists(col_data, base_type)

        # Scalar column
        if isinstance(col_data, np.ndarray):
            return pa.array(col_data, type=arrow_type)

        return pa.array(col_data, type=arrow_type)

    def _get_writer_kwargs(self) -> dict[str, Any]:
        """Extract writer config kwargs to forward to sub-writers in parallel mode.

        Returns a dict of keyword arguments that can be passed to a new
        ``ParquetWriter(columns=..., out=..., **kwargs)`` to replicate this
        writer's configuration (compression, hashes, size_limit, etc.).
        """
        return {
            'compression': self.parquet_compression,
            'compression_level': self.compression_level,
            'hashes': self.hashes or None,
            'size_limit': self.size_limit,
            'row_group_size': self.row_group_size,
            'exist_ok': True,  # sub-dirs are freshly created, but be safe
        }

    def write_mp(
        self,
        dataset: Any,
        *,
        num_workers: int = 4,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        """Write an entire dataset in parallel using multiprocessing.

        The dataset is partitioned into ``num_workers`` chunks. Each chunk is
        processed in a separate subprocess that runs its own iteration loop,
        applies the optional ``transform``, and writes to an isolated
        subdirectory via its own ``ParquetWriter``. After all workers finish,
        ``merge_index`` combines the per-partition ``index.json`` files into a
        single unified index at the root output directory.

        This parallelises **all three stages** — iteration, transformation, and
        writing — which is the key advantage over putting multiprocessing only
        inside ``write()``.

        For single-process writing, use the normal ``write()`` loop instead::

            for row in dataset:
                writer.write(row)

        Args:
            dataset: An indexable dataset that supports ``len(dataset)`` and
                ``dataset[i]``. HuggingFace datasets, Python lists, and any
                object implementing ``__len__`` + ``__getitem__`` work.
                Generators and plain iterables are **not** supported — convert
                them to a list first.
            num_workers (int): Number of parallel worker processes.
                Defaults to ``4``.
            transform (callable, optional): A function applied to each sample dict
                before writing. Signature: ``(sample: dict) -> dict``.
                Must be a picklable function (top-level or named function,
                **not** a lambda), because it is sent to worker processes
                via ``multiprocessing``.

        Raises:
            TypeError: If ``dataset`` does not support ``len()`` and ``__getitem__``.
            TypeError: If ``transform`` is not callable.
            ValueError: If ``num_workers`` is less than 1.

        Example:
            Parallel write with 4 workers::

                with ParquetWriter(out="./output", columns=columns) as writer:
                    writer.write_mp(hf_ds, num_workers=4)

            With a transform function::

                def tokenize(row):
                    ids = tokenizer(row["text"])["input_ids"]
                    return {"input_ids": np.array(ids, dtype=np.uint32)}

                with ParquetWriter(out="./output", columns=columns) as writer:
                    writer.write_mp(hf_ds, num_workers=4, transform=tokenize)
        """
        # --- Input validation ---
        if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
            raise TypeError(
                'dataset must be indexable (support __len__ and __getitem__). '
                'HuggingFace datasets and lists work. If you have a generator, '
                'convert it to a list first.'
            )
        if transform is not None and not callable(transform):
            raise TypeError('transform must be a callable, got '
                            f'{type(transform).__name__}.')
        if num_workers < 1:
            raise ValueError(f'num_workers must be >= 1, got {num_workers}.')

        N = len(dataset)
        if N == 0:
            return

        # --- Parallel path ---
        chunk_size = (N + num_workers - 1) // num_workers
        writer_kwargs = self._get_writer_kwargs()

        partition_args = []
        for part_id in range(num_workers):
            start = part_id * chunk_size
            end = min(start + chunk_size, N)
            if start >= N:
                break  # fewer samples than workers
            sub_dir = str(self.local / f'{part_id:05d}')
            partition_args.append(
                (sub_dir, dataset, start, end, self.columns, writer_kwargs, transform)
            )

        actual_workers = len(partition_args)
        logger.info(
            f'write_mp: {N} samples across {actual_workers} workers'
        )

        with Pool(processes=actual_workers) as pool:
            pool.map(_write_partition_worker, partition_args)

        # Merge per-partition index.json files into a single root index.json
        merge_index(str(self.local))

        # Signal that the index has been written so finish() doesn't overwrite it
        self._index_written = True

        logger.info(
            f'write_mp: done. Merged {actual_workers} partitions into '
            f'{self.local / "index.json"}'
        )

    def _reset_cache(self) -> None:
        """Reset our internal shard-building cache."""
        super()._reset_cache()
        self._raw_samples = []
