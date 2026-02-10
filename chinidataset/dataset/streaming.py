"""StreamingDataset: A PyTorch IterableDataset for streaming Parquet shards.

Adapted from: https://github.com/mosaicml/streaming/blob/main/streaming/base/dataset.py
"""

import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import sleep
from typing import Any, Iterator, Optional, Union

import numpy as np
from torch.utils.data import IterableDataset

from chinidataset.dataset.cache import CacheManager, ShardInfo
from chinidataset.dataset.partition import get_partition
from chinidataset.dataset.reader import ParquetReader
from chinidataset.dataset.shuffle import no_shuffle, shuffle_samples
from chinidataset.dataset.world import World
from chinidataset.util import bytes_to_int, get_index_basename

__all__ = ['StreamingDataset']

logger = logging.getLogger(__name__)

# Tick interval for polling loops
_TICK = 0.007


class StreamingDataset(IterableDataset):
    """A streaming Parquet dataset for PyTorch training.

    Reads sharded Parquet datasets created by ChiniDataset's ParquetWriter.
    Supports shuffling, multi-worker DataLoader, distributed training,
    shard caching, LRU eviction, and mid-epoch resumption.

    Implements both ``__iter__`` (for IterableDataset usage) and ``__getitem__``
    (for random access).

    Args:
        local (str | Path): Local directory containing (or caching) shards.
        remote (str, optional): Remote URL to download shards from.
            Supports S3, GCS, HuggingFace Hub, HTTP. If None, all shards must
            exist locally. Defaults to ``None``.
        split (str, optional): Dataset split subdirectory (e.g., "train", "val").
            If provided, appended to both local and remote paths. Defaults to ``None``.
        batch_size (int, optional): Per-device batch size. Required for optimal
            partitioning. Defaults to ``None``.
        shuffle (bool): Whether to shuffle sample order each epoch. Defaults to
            ``False``.
        shuffle_seed (int): Seed for deterministic shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Block size for block-based shuffling. Larger
            blocks = more randomness. Defaults to ``262144`` (256K).
        cache_limit (Union[int, str], optional): Maximum cache size. Supports
            human-readable format ("10gb"). If None, no eviction. Defaults to ``None``.
        predownload (int, optional): Number of samples to prefetch ahead.
            Defaults to ``8 * batch_size`` or ``64``.
        max_open_shards (int): Maximum number of Parquet shard files to keep
            loaded in memory simultaneously. When shuffle is on, random access
            jumps between shards -- without a cap, all shards get loaded and
            eat all RAM. Oldest readers are evicted when this limit is hit.
            Defaults to ``8``.
        look_ahead (int): Number of upcoming shards to load in the background
            while the current shard is being iterated. When reading partition 0,
            partitions 1 .. ``look_ahead`` are loaded asynchronously so they are
            ready by the time the iterator reaches them. Set to ``0`` to disable.
            Must be less than ``max_open_shards``. Defaults to ``2``.

    Example:
        >>> from chinidataset import StreamingDataset
        >>> from torch.utils.data import DataLoader
        >>>
        >>> dataset = StreamingDataset(local="./data", shuffle=True, batch_size=32)
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
        >>> for batch in loader:
        ...     print(batch["label"])
    """

    def __init__(
        self,
        *,
        local: Union[str, Path],
        remote: Optional[str] = None,
        split: Optional[str] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_seed: int = 9176,
        shuffle_block_size: int = 1 << 18,  # 262144
        cache_limit: Optional[Union[int, str]] = None,
        predownload: Optional[int] = None,
        max_open_shards: int = 8,
        look_ahead: int = 2,
    ) -> None:
        local_path = Path(local).expanduser().resolve()
        if split:
            local_path = local_path / split
        self.local = local_path
        self.split = split
        if remote and split:
            self.remote = f'{remote.rstrip("/")}/{split}'
        else:
            self.remote = remote
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_block_size = shuffle_block_size

        # Parse cache limit
        self.cache_limit: Optional[int] = None
        if cache_limit is not None:
            self.cache_limit = bytes_to_int(cache_limit)

        # Predownload
        if predownload is not None:
            self.predownload = predownload
        elif batch_size is not None:
            self.predownload = 8 * batch_size
        else:
            self.predownload = 64

        # Detect distributed topology (rank-level, workers detected in __iter__)
        self._rank_world = World.detect()

        # Load index.json
        index_path = self.local / get_index_basename()
        if not index_path.exists():
            if self.remote:
                self._download_index()
            else:
                raise FileNotFoundError(f'index.json not found at {index_path}')

        with open(index_path, 'r') as f:
            index = json.load(f)

        if index.get('version') != 2:
            raise ValueError(f'Unsupported index version: {index.get("version")}')

        # Parse shard metadata
        self._shard_infos: list[ShardInfo] = []
        self._samples_per_shard: list[int] = []
        self.num_samples = 0

        for shard_idx, shard_meta in enumerate(index['shards']):
            num_samples = shard_meta['samples']
            raw_data = shard_meta.get('raw_data', {})
            basename = raw_data.get('basename', f'shard.{shard_idx:05}.parquet')
            size_bytes = raw_data.get('bytes', 0)

            info = ShardInfo(
                shard_id=shard_idx,
                basename=basename,
                num_samples=num_samples,
                size_bytes=size_bytes,
                local_dir=self.local,
            )
            self._shard_infos.append(info)
            self._samples_per_shard.append(num_samples)
            self.num_samples += num_samples

        self.num_shards = len(self._shard_infos)

        # Build cumulative sample offsets for sample -> shard mapping
        self._sample_offsets = np.zeros(self.num_shards + 1, dtype=np.int64)
        for i, count in enumerate(self._samples_per_shard):
            self._sample_offsets[i + 1] = self._sample_offsets[i] + count

        # Initialize cache manager
        self._cache = CacheManager(
            local=self.local,
            remote=self.remote,
            shards=self._shard_infos,
            cache_limit=self.cache_limit,
        )

        # LRU reader cache: keep at most max_open_shards loaded in memory.
        # When shuffling, random access jumps between shards -- without a cap,
        # all shards get loaded and eat all RAM. With a cap, oldest readers
        # are evicted (unloaded) when new shards are accessed.
        # Pattern from: https://github.com/malaysia-ai/malaysian-dataset/blob/master/speech-to-text-semisupervised/pseudolabel-whisper/run.py#L60
        self.max_open_shards = max_open_shards
        # OrderedDict gives O(1) move_to_end / popitem for LRU eviction,
        # replacing the previous dict + list approach that had O(n) remove.
        self._readers: OrderedDict[int, ParquetReader] = OrderedDict()

        # Look-ahead caching: load upcoming shards in background threads so
        # the main iterator never blocks on pd.read_parquet for the next shard.
        self.look_ahead = look_ahead
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        if self.look_ahead > 0:
            if self.look_ahead >= self.max_open_shards:
                logger.warning(
                    f'look_ahead ({self.look_ahead}) >= max_open_shards '
                    f'({self.max_open_shards}). Increasing max_open_shards to '
                    f'{self.look_ahead + 2} to avoid evicting prefetched shards.'
                )
                self.max_open_shards = self.look_ahead + 2
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=min(self.look_ahead, 4),
            )

        # Epoch tracking
        self._epoch = 0

        # Resumption state
        self._resume_epoch: Optional[int] = None
        self._resume_sample_in_epoch: int = 0

    def _download_index(self) -> None:
        """Download index.json from remote to local.

        Supports S3, GCS, HuggingFace Hub, HTTP, and local paths.
        Uses the same download logic as CacheManager.
        """
        if not self.remote:
            return

        remote_index = f'{self.remote.rstrip("/")}/{get_index_basename()}'
        local_index = self.local / get_index_basename()
        self.local.mkdir(parents=True, exist_ok=True)

        # Reuse CacheManager's download logic (handles hf://, s3://, etc.)
        from chinidataset.dataset.cache import CacheManager
        # Use a temporary CacheManager just for the download method
        dummy = CacheManager.__new__(CacheManager)
        dummy.remote = self.remote
        dummy._download_file(remote_index, local_index)

    def _sample_to_shard(self, sample_id: int) -> tuple[int, int]:
        """Map global sample ID to (shard_id, local_sample_id).

        Uses binary search on cumulative offsets for O(log n).

        Args:
            sample_id (int): Global sample index.

        Returns:
            Tuple[int, int]: (shard_id, sample_index_within_shard)
        """
        shard_id = int(np.searchsorted(self._sample_offsets[1:], sample_id, side='right'))
        local_id = sample_id - int(self._sample_offsets[shard_id])
        return shard_id, local_id

    def _get_reader(self, shard_id: int) -> ParquetReader:
        """Get or create a ParquetReader for a shard, with LRU eviction.

        Keeps at most max_open_shards readers loaded in memory. When a new
        shard is accessed and the cache is full, the least recently used
        reader is unloaded to free memory.

        This avoids OOM when shuffling causes random access across many shards.

        Args:
            shard_id (int): Shard index.

        Returns:
            ParquetReader: Reader for the shard.
        """
        if shard_id in self._readers:
            # Move to end of access order (most recent) — O(1)
            self._readers.move_to_end(shard_id)
            return self._readers[shard_id]

        # Evict oldest reader if at capacity — O(1) per eviction
        while len(self._readers) >= self.max_open_shards:
            _oldest_id, oldest_reader = self._readers.popitem(last=False)
            oldest_reader.unload()

        # Load new reader (inserted at the end = most recent)
        shard_path = self._cache.ensure_local(shard_id)
        reader = ParquetReader(shard_path)
        self._readers[shard_id] = reader
        return reader

    @staticmethod
    def _build_shard_schedule(
        worker_sample_ids: np.ndarray,
        sample_offsets: np.ndarray,
    ) -> list[int]:
        """Pre-compute the ordered list of distinct shards a worker will visit.

        Called once before the iteration loop so that look-ahead prefetching
        can index into the schedule in O(1) instead of scanning samples.

        Args:
            worker_sample_ids: The ordered array of sample IDs for this worker.
            sample_offsets: Cumulative sample offsets (length = num_shards + 1).

        Returns:
            List of shard IDs in visitation order (consecutive duplicates
            collapsed).
        """
        if len(worker_sample_ids) == 0:
            return []

        shard_ids = np.searchsorted(
            sample_offsets[1:], worker_sample_ids.astype(np.int64), side='right',
        )
        # Collapse consecutive duplicates
        change_mask = np.empty(len(shard_ids), dtype=np.bool_)
        change_mask[0] = True
        change_mask[1:] = shard_ids[1:] != shard_ids[:-1]
        return shard_ids[change_mask].tolist()

    def _prefetch_shards_ahead(
        self, shard_schedule: list[int], schedule_idx: int
    ) -> None:
        """Load the next ``look_ahead`` shards from the pre-computed schedule.

        Because the schedule is already deduplicated, this is O(look_ahead)
        regardless of how many samples each shard contains.

        Args:
            shard_schedule: Ordered list of distinct shard IDs (from
                :meth:`_build_shard_schedule`).
            schedule_idx: Current position in the schedule (the shard being
                iterated right now).  Shards at positions
                ``schedule_idx + 1 .. schedule_idx + look_ahead`` will be
                loaded in the background.
        """
        if self._prefetch_executor is None:
            return

        start = schedule_idx + 1
        end = min(start + self.look_ahead, len(shard_schedule))

        for k in range(start, end):
            shard_id = shard_schedule[k]

            # Already loaded -- nothing to do
            if shard_id in self._readers and self._readers[shard_id].is_loaded:
                continue

            # Only prefetch shards that are already on disk (don't block on download)
            if not self._cache.is_local(shard_id):
                continue

            reader = self._get_reader(shard_id)
            reader.load_async(self._prefetch_executor)

    def __del__(self) -> None:
        """Shut down the background prefetch pool on garbage collection."""
        if hasattr(self, '_prefetch_executor') and self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=False)

    def __getitem__(self, sample_id: int) -> dict[str, Any]:
        """Get a sample by global index.

        Downloads the shard if not present locally. Supports negative indexing
        (e.g., ``ds[-1]`` returns the last sample).

        Args:
            sample_id (int): Global sample index (negative values count from the end).

        Returns:
            Dict[str, Any]: Sample dictionary.
        """
        if sample_id < 0:
            sample_id += self.num_samples
        if sample_id < 0 or sample_id >= self.num_samples:
            raise IndexError(
                f'Index {sample_id} out of range for dataset with {self.num_samples} samples'
            )
        shard_id, local_id = self._sample_to_shard(sample_id)
        self._cache.touch(shard_id)
        reader = self._get_reader(shard_id)
        return reader[local_id]

    def __len__(self) -> int:
        """Total number of samples across all shards."""
        return self.num_samples

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples with shuffling, partitioning, and prefetching.

        Each worker gets a disjoint subset of samples. Samples are optionally
        shuffled with a different ordering each epoch.

        Returns:
            Iterator[Dict[str, Any]]: Each sample as a dictionary.
        """
        # Detect worker context
        world = self._rank_world.detect_workers()

        # Determine epoch
        if self._resume_epoch is not None:
            epoch = self._resume_epoch
            sample_in_epoch = self._resume_sample_in_epoch
            self._resume_epoch = None
            self._resume_sample_in_epoch = 0
        else:
            epoch = self._epoch
            sample_in_epoch = 0

        self._epoch = epoch + 1

        # Generate sample ordering for this epoch
        if self.shuffle:
            all_sample_ids = shuffle_samples(
                num_samples=self.num_samples,
                block_size=self.shuffle_block_size,
                seed=self.shuffle_seed,
                epoch=epoch,
            )
        else:
            all_sample_ids = no_shuffle(self.num_samples)

        # Skip already-processed samples (for resumption)
        if sample_in_epoch > 0:
            all_sample_ids = all_sample_ids[sample_in_epoch:]

        # Partition across workers
        worker_sample_ids = get_partition(all_sample_ids, world)

        if len(worker_sample_ids) == 0:
            return

        # Check if all shards are already local (common case: local-only mode)
        all_local = all(
            self._shard_infos[sid].local_path.exists()
            for sid in range(self.num_shards)
        )

        if all_local:
            # Fast path: no download needed.
            # Look-ahead only helps sequential access where shards are visited
            # in order.  With shuffle the access pattern jumps between shards
            # randomly and the LRU cache already keeps them hot -- scanning
            # ahead just adds overhead on every shard transition.
            use_look_ahead = self._prefetch_executor is not None and not self.shuffle

            # Pre-compute shard schedule for O(1) look-ahead indexing
            shard_schedule: list[int] = []
            schedule_idx = -1
            if use_look_ahead:
                shard_schedule = self._build_shard_schedule(
                    worker_sample_ids, self._sample_offsets,
                )
                schedule_idx = 0
                self._prefetch_shards_ahead(shard_schedule, -1)

            last_shard_id = -1
            for i, sample_id in enumerate(worker_sample_ids):
                shard_id, local_id = self._sample_to_shard(sample_id)
                reader = self._get_reader(shard_id)
                reader.wait_loaded()

                if use_look_ahead and shard_id != last_shard_id:
                    self._prefetch_shards_ahead(shard_schedule, schedule_idx)
                    schedule_idx += 1
                    last_shard_id = shard_id

                yield reader[local_id]
        else:
            # Streaming path: prefetch shards in background, wait for downloads
            yield from self._iter_streaming(worker_sample_ids)

    def _iter_streaming(self, worker_sample_ids: np.ndarray) -> Iterator[dict[str, Any]]:
        """Iterate with background prefetching (for remote/cached datasets).

        The download-prefetch thread fetches shard files from the remote.
        After a shard is confirmed local, look-ahead caching loads the
        Parquet data in the background so the main thread doesn't block.
        """
        event = Event()
        prefetch_index = [0]
        yield_index = [0]

        def prefetch_thread():
            while not event.is_set():
                idx = prefetch_index[0]
                if idx >= len(worker_sample_ids):
                    break

                yield_idx = yield_index[0]
                if idx - yield_idx > self.predownload:
                    sleep(_TICK)
                    continue

                sample_id = worker_sample_ids[idx]
                shard_id, _ = self._sample_to_shard(sample_id)

                try:
                    self._cache.ensure_local(shard_id)
                except Exception as e:
                    logger.error(f'Prefetch failed for shard {shard_id}: {e}')
                    event.set()
                    break

                prefetch_index[0] = idx + 1

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(prefetch_thread)

        use_look_ahead = self._prefetch_executor is not None and not self.shuffle

        # Pre-compute shard schedule for O(1) look-ahead indexing
        shard_schedule: list[int] = []
        schedule_idx = -1
        if use_look_ahead:
            shard_schedule = self._build_shard_schedule(
                worker_sample_ids, self._sample_offsets,
            )
            schedule_idx = 0

        try:
            last_shard_id = -1
            for i, sample_id in enumerate(worker_sample_ids):
                if event.is_set():
                    raise RuntimeError('Prefetch thread failed. Check logs.')

                yield_index[0] = i

                shard_id, local_id = self._sample_to_shard(sample_id)
                while not self._cache.is_local(shard_id):
                    if event.is_set():
                        raise RuntimeError('Prefetch thread failed.')
                    sleep(_TICK)

                self._cache.touch(shard_id)
                reader = self._get_reader(shard_id)
                reader.wait_loaded()

                if use_look_ahead and shard_id != last_shard_id:
                    self._prefetch_shards_ahead(shard_schedule, schedule_idx)
                    schedule_idx += 1
                    last_shard_id = shard_id

                yield reader[local_id]
        finally:
            event.set()
            future.result(timeout=5)
            executor.shutdown(wait=False)

    def state_dict(self, num_samples: int, from_beginning: bool = False) -> dict[str, Any]:
        """Get checkpoint state for mid-epoch resumption.

        Args:
            num_samples (int): Number of samples processed so far.
            from_beginning (bool): Whether counting from epoch start. Defaults to False.

        Returns:
            Dict[str, Any]: State dict for checkpointing.
        """
        epoch = self._epoch - 1  # Current epoch (epoch was pre-incremented)

        if from_beginning:
            sample_in_epoch = num_samples
        else:
            sample_in_epoch = self._resume_sample_in_epoch + num_samples

        return {
            'epoch': epoch,
            'sample_in_epoch': sample_in_epoch,
            'shuffle_seed': self.shuffle_seed,
            'num_samples': self.num_samples,
        }

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        """Load checkpoint state for mid-epoch resumption.

        Args:
            obj (Dict[str, Any]): State dict from a previous checkpoint.
        """
        self._resume_epoch = obj['epoch']
        self._resume_sample_in_epoch = obj['sample_in_epoch']
        self.shuffle_seed = obj.get('shuffle_seed', self.shuffle_seed)
