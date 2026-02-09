# Copyright 2022-2024 MosaicML Streaming authors
# Copyright 2024 ChiniML Contributors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for ChiniML.

Adapted from: https://github.com/mosaicml/streaming/blob/main/streaming/base/util.py
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Union

__all__ = ['bytes_to_int', 'get_index_basename', 'merge_index']

logger = logging.getLogger(__name__)


def bytes_to_int(size: Union[int, str]) -> int:
    """Convert a human-readable byte string to an integer.

    Supports formats like "100kb", "1mb", "1gb", etc.

    Args:
        size: Size as int or human-readable string.

    Returns:
        int: Size in bytes.
    """
    if isinstance(size, int):
        return size

    size = size.strip().lower()
    
    # Match number and optional unit
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([kmgtpe]?b?)?$', size)
    if not match:
        raise ValueError(f'Invalid size format: {size}')

    value = float(match.group(1))
    unit = match.group(2) or ''

    multipliers = {
        '': 1,
        'b': 1,
        'k': 1024,
        'kb': 1024,
        'm': 1024**2,
        'mb': 1024**2,
        'g': 1024**3,
        'gb': 1024**3,
        't': 1024**4,
        'tb': 1024**4,
        'p': 1024**5,
        'pb': 1024**5,
        'e': 1024**6,
        'eb': 1024**6,
    }

    if unit not in multipliers:
        raise ValueError(f'Unknown size unit: {unit}')

    return int(value * multipliers[unit])


def get_index_basename() -> str:
    """Get the basename of the index file.

    Returns:
        str: Index file basename.
    """
    return 'index.json'


def merge_index(
    out: Union[str, Path],
    index_file_urls: Optional[Sequence[str]] = None,
    keep_local: bool = True,
) -> None:
    """Merge index.json from partitions to form a global index.json.

    This supports two modes:

    1. **Auto-discover**: Pass ``out`` (the root directory). All sub-directories
       containing ``index.json`` will be discovered and merged.

    2. **Explicit list**: Pass ``out`` and ``index_file_urls`` (list of paths to
       individual ``index.json`` files).

    After merging, shard basenames are rewritten to include the relative sub-directory
    path so that the merged index correctly points to shard files inside sub-folders.

    Adapted from:
        https://github.com/mosaicml/streaming/blob/main/streaming/base/util.py
        https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/parallel_dataset_conversion.html

    Args:
        out (str | Path): Root directory that contains MDS partitions (sub-directories).
            The merged ``index.json`` will be written here.
        index_file_urls (Sequence[str], optional): Explicit list of ``index.json`` file
            paths. If ``None``, auto-discovers from sub-directories of ``out``.
        keep_local (bool): Keep local copy of the merged index file. Defaults to ``True``.

    Example:
        Parallel conversion with merge::

            from multiprocessing import Pool
            from chiniml import MDSWriter
            from chiniml.util import merge_index

            def convert_partition(args):
                partition_id, samples = args
                sub_dir = f"./output/{partition_id:05d}"
                columns = {"text": "str", "label": "int32"}
                with MDSWriter(out=sub_dir, columns=columns) as writer:
                    for sample in samples:
                        writer.write(sample)

            # Write partitions in parallel
            with Pool(4) as pool:
                pool.map(convert_partition, enumerate(data_chunks))

            # Merge all partition index files
            merge_index("./output")

        Explicit index file list::

            merge_index(
                out="./output",
                index_file_urls=[
                    "./output/00000/index.json",
                    "./output/00001/index.json",
                    "./output/00002/index.json",
                ],
            )

    Raises:
        FileNotFoundError: If ``out`` does not exist or no ``index.json`` files found.
        ValueError: If an ``index.json`` file cannot be parsed.
    """
    out = Path(out).expanduser().resolve()
    index_basename = get_index_basename()

    if not out.exists():
        raise FileNotFoundError(f'Output directory does not exist: {out}')

    # Collect index files
    if index_file_urls is not None:
        # Explicit list mode
        index_files = [Path(p).expanduser().resolve() for p in index_file_urls]
        for f in index_files:
            if not f.exists():
                raise FileNotFoundError(f'Index file does not exist: {f}')
    else:
        # Auto-discover mode: find index.json in sub-directories
        index_files = []
        for sub_dir in sorted(out.iterdir()):
            if sub_dir.is_dir():
                index_path = sub_dir / index_basename
                if index_path.exists():
                    index_files.append(index_path)

    if not index_files:
        raise FileNotFoundError(
            f'No {index_basename} files found in sub-directories of {out}'
        )

    # Merge shards from all index files
    merged_shards: list[dict[str, Any]] = []

    for index_file in index_files:
        with open(index_file, 'r') as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f'Failed to parse {index_file}: {e}') from e

        # Get the relative path from the root to this partition directory
        partition_dir = index_file.parent
        rel_dir = partition_dir.relative_to(out)

        for shard in obj.get('shards', []):
            # Rewrite basenames to include the sub-directory path
            for key in ('raw_data', 'zip_data', 'raw_meta', 'zip_meta'):
                if shard.get(key) and shard[key].get('basename'):
                    original_basename = shard[key]['basename']
                    shard[key]['basename'] = str(rel_dir / original_basename)
            merged_shards.append(shard)

    # Write merged index
    merged_index = {
        'version': 2,
        'shards': merged_shards,
    }

    merged_index_path = out / index_basename
    with open(merged_index_path, 'w') as f:
        json.dump(merged_index, f, sort_keys=True, indent=2)

    logger.info(
        f'Merged {len(index_files)} partitions with {len(merged_shards)} '
        f'total shards into {merged_index_path}'
    )

    return
