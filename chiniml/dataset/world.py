# Copyright 2024 ChiniML Contributors
# SPDX-License-Identifier: Apache-2.0

"""Detect distributed training topology from environment variables.

Adapted from: https://github.com/mosaicml/streaming/blob/main/streaming/base/world.py
"""

import os
from dataclasses import dataclass
from typing import Optional

__all__ = ['World']


@dataclass
class World:
    """Distributed training topology.

    Attributes:
        num_nodes: Total number of nodes.
        node: This node's index.
        num_ranks: Total number of ranks (GPUs).
        rank: This rank's global index.
        ranks_per_node: Number of ranks on each node.
        rank_of_node: This rank's index within its node.
        num_workers: Number of DataLoader workers on this rank.
        worker_of_rank: This worker's index within its rank.
        is_local_leader: Whether this rank is the local leader (rank 0 on this node).
    """

    num_nodes: int
    node: int
    num_ranks: int
    rank: int
    ranks_per_node: int
    rank_of_node: int
    num_workers: int = 1
    worker_of_rank: int = 0

    @property
    def is_local_leader(self) -> bool:
        """Whether this is the local leader (first rank on this node)."""
        return self.rank_of_node == 0

    @property
    def is_global_leader(self) -> bool:
        """Whether this is the global leader (rank 0)."""
        return self.rank == 0

    @staticmethod
    def detect() -> 'World':
        """Detect the distributed training topology from environment variables.

        Supports PyTorch distributed (WORLD_SIZE, RANK, LOCAL_RANK),
        OpenMPI (OMPI_COMM_WORLD_*), and SLURM (SLURM_*).

        Returns:
            World: Detected topology.
        """
        # Try PyTorch distributed env vars first
        world_size = _env_int('WORLD_SIZE')
        rank = _env_int('RANK')
        local_rank = _env_int('LOCAL_RANK')
        local_world_size = _env_int('LOCAL_WORLD_SIZE')

        # Try OpenMPI
        if world_size is None:
            world_size = _env_int('OMPI_COMM_WORLD_SIZE')
            rank = _env_int('OMPI_COMM_WORLD_RANK')
            local_rank = _env_int('OMPI_COMM_WORLD_LOCAL_RANK')
            local_world_size = _env_int('OMPI_COMM_WORLD_LOCAL_SIZE')

        # Try SLURM
        if world_size is None:
            world_size = _env_int('SLURM_NTASKS')
            rank = _env_int('SLURM_PROCID')
            local_rank = _env_int('SLURM_LOCALID')
            local_world_size = _env_int('SLURM_NTASKS_PER_NODE')

        # Defaults: single process
        if world_size is None:
            world_size = 1
        if rank is None:
            rank = 0
        if local_rank is None:
            local_rank = 0
        if local_world_size is None:
            local_world_size = 1

        ranks_per_node = local_world_size
        num_nodes = world_size // ranks_per_node if ranks_per_node else 1
        node = rank // ranks_per_node if ranks_per_node else 0
        rank_of_node = local_rank

        return World(
            num_nodes=num_nodes,
            node=node,
            num_ranks=world_size,
            rank=rank,
            ranks_per_node=ranks_per_node,
            rank_of_node=rank_of_node,
        )

    def detect_workers(self) -> 'World':
        """Update with DataLoader worker info (call inside __iter__).

        Returns:
            World: Updated topology with worker info.
        """
        try:
            import torch.utils.data
            info = torch.utils.data.get_worker_info()
            if info is not None:
                return World(
                    num_nodes=self.num_nodes,
                    node=self.node,
                    num_ranks=self.num_ranks,
                    rank=self.rank,
                    ranks_per_node=self.ranks_per_node,
                    rank_of_node=self.rank_of_node,
                    num_workers=info.num_workers,
                    worker_of_rank=info.id,
                )
        except ImportError:
            pass

        return World(
            num_nodes=self.num_nodes,
            node=self.node,
            num_ranks=self.num_ranks,
            rank=self.rank,
            ranks_per_node=self.ranks_per_node,
            rank_of_node=self.rank_of_node,
            num_workers=1,
            worker_of_rank=0,
        )


def _env_int(key: str) -> Optional[int]:
    """Read an integer from an environment variable."""
    val = os.environ.get(key)
    if val is None:
        return None
    return int(val)
