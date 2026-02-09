# Copyright 2024 ChiniML Contributors
# SPDX-License-Identifier: Apache-2.0

"""Partition sample IDs across distributed workers.

Adapted from: https://github.com/mosaicml/streaming/blob/main/streaming/base/partition.py
"""

import numpy as np
from numpy.typing import NDArray

from chiniml.dataset.world import World

__all__ = ['get_partition']


def get_partition(
    sample_ids: NDArray[np.int64],
    world: World,
) -> NDArray[np.int64]:
    """Get this worker's partition of sample IDs.

    Divides samples evenly across: nodes -> ranks_per_node -> workers_per_rank.

    Each worker gets a disjoint subset. If samples don't divide evenly,
    earlier workers get one extra sample.

    Args:
        sample_ids (NDArray[np.int64]): Global sample ordering (e.g., shuffled).
        world (World): Current distributed topology with worker info.

    Returns:
        NDArray[np.int64]: This worker's partition of sample IDs.
    """
    total = len(sample_ids)
    if total == 0:
        return np.array([], dtype=np.int64)

    # Total number of workers across all nodes, ranks
    total_workers = world.num_ranks * world.num_workers

    # Global worker index
    global_worker_id = world.rank * world.num_workers + world.worker_of_rank

    # Divide samples across workers using interleaving for balance
    # Worker i gets samples at indices [i, i+W, i+2W, ...] where W = total_workers
    worker_indices = np.arange(global_worker_id, total, total_workers, dtype=np.int64)

    return sample_ids[worker_indices]
