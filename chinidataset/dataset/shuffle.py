"""Deterministic block-based shuffling for streaming datasets.

Adapted from mosaicml-streaming's py1e shuffle algorithm:
https://github.com/mosaicml/streaming/blob/main/streaming/base/shuffle.py
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ['shuffle_samples']


def shuffle_samples(
    num_samples: int,
    block_size: int,
    seed: int,
    epoch: int,
) -> NDArray[np.int64]:
    """Generate a deterministic shuffled sample ordering.

    Uses block-based shuffling: samples are divided into blocks, blocks are
    shuffled globally, then samples within each block are shuffled. This gives
    near-random ordering while maintaining some shard locality (samples from
    the same shard tend to be in the same block).

    The shuffling is deterministic: same (seed, epoch) always produces the same
    ordering, regardless of the number of nodes/workers.

    Args:
        num_samples (int): Total number of samples in the dataset.
        block_size (int): Size of each shuffle block. Larger blocks = more
            randomness, smaller blocks = better cache locality. Typical: 262144.
        seed (int): Base random seed.
        epoch (int): Epoch number (changes ordering each epoch).

    Returns:
        NDArray[np.int64]: Array of sample IDs in shuffled order.
    """
    if num_samples == 0:
        return np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed + epoch)

    # Create sample IDs
    sample_ids = np.arange(num_samples, dtype=np.int64)

    # Divide into blocks
    num_blocks = (num_samples + block_size - 1) // block_size

    # Create block boundaries
    blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, num_samples)
        blocks.append(sample_ids[start:end])

    # Shuffle block order
    block_order = rng.permutation(num_blocks)

    # Shuffle within each block and concatenate
    result = []
    for block_idx in block_order:
        block = blocks[block_idx].copy()
        rng.shuffle(block)
        result.append(block)

    return np.concatenate(result)


def no_shuffle(num_samples: int) -> NDArray[np.int64]:
    """Return sample IDs in sequential order (no shuffling).

    Args:
        num_samples (int): Total number of samples.

    Returns:
        NDArray[np.int64]: Sequential sample IDs.
    """
    return np.arange(num_samples, dtype=np.int64)
