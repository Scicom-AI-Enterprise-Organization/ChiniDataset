# Copyright 2022-2024 MosaicML Streaming authors
# Copyright 2024 ChiniML Contributors
# SPDX-License-Identifier: Apache-2.0

"""Base class for serializing samples into streaming dataset shards.

Adapted from: https://github.com/mosaicml/streaming/blob/main/streaming/base/format/base/writer.py
"""

import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, Union

from chiniml.hashing import get_hash, is_hash
from chiniml.util import bytes_to_int, get_index_basename

__all__ = ['Writer']

logger = logging.getLogger(__name__)


class Writer(ABC):
    """Base class for writing streaming datasets.

    Args:
        out (str | Path): Output dataset directory to save shard files.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (Union[int, str], optional): Optional shard size limit, after which point
            to start a new shard. If ``None``, puts everything in one shard. Can specify bytes
            in human-readable format (e.g., "100kb", "64mb"). Defaults to ``1 << 26`` (64MB).
        extra_bytes_per_shard (int): Extra bytes per serialized shard (for computing shard size
            while writing). Defaults to ``0``.
        extra_bytes_per_sample (int): Extra bytes per serialized sample (for computing shard size
            while writing). Defaults to ``0``.
        exist_ok (bool): If the local directory exists and is not empty, whether to overwrite
            the content or raise an error. Defaults to ``False``.
    """

    format: str = ''

    def __init__(
        self,
        *,
        out: Union[str, Path],
        hashes: Optional[list[str]] = None,
        size_limit: Optional[Union[int, str]] = 1 << 26,
        extra_bytes_per_shard: int = 0,
        extra_bytes_per_sample: int = 0,
        exist_ok: bool = False,
    ) -> None:
        # Validate hashes
        hashes = hashes or []
        if list(hashes) != sorted(hashes):
            raise ValueError('Hashes must be unique and in sorted order.')
        for algo in hashes:
            if not is_hash(algo):
                raise ValueError(f'Invalid hash: {algo}.')

        # Validate size limit
        size_limit_value = None
        if size_limit:
            size_limit_value = bytes_to_int(size_limit)
            if size_limit_value < 0:
                raise ValueError(f'`size_limit` must be greater than zero.')
            if size_limit_value >= 2**32:
                raise ValueError(f'`size_limit` must be less than 2**32.')

        self.hashes = hashes
        self.size_limit = size_limit_value
        self.extra_bytes_per_shard = extra_bytes_per_shard
        self.extra_bytes_per_sample = extra_bytes_per_sample
        self.new_samples: list[bytes]
        self.new_shard_size: int

        self.shards: list[dict[str, Any]] = []

        # Setup local directory
        self.local = Path(out).expanduser().resolve()
        if self.local.exists() and any(self.local.iterdir()):
            if exist_ok:
                logger.warning(f'Directory {self.local} exists; removing contents.')
                shutil.rmtree(self.local)
            else:
                raise ValueError(
                    f'Directory {self.local} exists and is not empty. '
                    f'Set exist_ok=True to overwrite.'
                )
        self.local.mkdir(parents=True, exist_ok=True)

        self._reset_cache()

    def _reset_cache(self) -> None:
        """Reset our internal shard-building cache."""
        self.new_samples = []
        self.new_shard_size = self.extra_bytes_per_shard

    @abstractmethod
    def encode_sample(self, sample: dict[str, Any]) -> bytes:
        """Encode a sample dict to bytes."""
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        """Get object describing shard-writing configuration."""
        return {
            'version': 2,
            'format': self.format,
            'hashes': self.hashes,
            'size_limit': self.size_limit,
        }

    @abstractmethod
    def flush_shard(self) -> None:
        """Flush cached samples to storage, creating a new shard."""
        raise NotImplementedError

    def write(self, sample: dict[str, Any]) -> None:
        """Write a sample."""
        new_sample = self.encode_sample(sample)
        new_sample_size = len(new_sample) + self.extra_bytes_per_sample
        if self.size_limit and self.size_limit < self.new_shard_size + new_sample_size:
            self.flush_shard()
            self._reset_cache()
        self.new_samples.append(new_sample)
        self.new_shard_size += new_sample_size

    def _write_index(self) -> None:
        """Write the index, having written all the shards."""
        basename = get_index_basename()
        filename = self.local / basename
        obj = {
            'version': 2,
            'shards': self.shards,
        }
        with open(filename, 'w') as out:
            json.dump(obj, out, sort_keys=True, indent=2)

    def finish(self) -> None:
        """Finish writing samples."""
        if self.new_samples:
            self.flush_shard()
            self._reset_cache()
        self._write_index()

    def __enter__(self) -> 'Writer':
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.finish()
