"""Tests for chinidataset/dataset/world.py."""

import pytest

from chinidataset.dataset.world import World


class TestWorldDetect:
    def test_default_single_process(self, monkeypatch):
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                    "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK",
                    "OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_SIZE",
                    "SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS_PER_NODE"):
            monkeypatch.delenv(key, raising=False)

        w = World.detect()
        assert w.num_ranks == 1
        assert w.rank == 0
        assert w.rank_of_node == 0

    def test_pytorch_env(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("LOCAL_RANK", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        for key in ("OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"):
            monkeypatch.delenv(key, raising=False)

        w = World.detect()
        assert w.num_ranks == 4
        assert w.rank == 2

    def test_slurm_env(self, monkeypatch):
        for key in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("SLURM_NTASKS", "8")
        monkeypatch.setenv("SLURM_PROCID", "3")
        monkeypatch.setenv("SLURM_LOCALID", "3")
        monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "4")

        w = World.detect()
        assert w.num_ranks == 8
        assert w.rank == 3

    def test_openmpi_env(self, monkeypatch):
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
        monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
        monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "2")
        for key in ("SLURM_NTASKS",):
            monkeypatch.delenv(key, raising=False)

        w = World.detect()
        assert w.num_ranks == 2
        assert w.rank == 1

    def test_pytorch_takes_precedence_over_slurm(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
        monkeypatch.setenv("SLURM_NTASKS", "16")
        monkeypatch.setenv("SLURM_PROCID", "5")

        w = World.detect()
        assert w.num_ranks == 2
        assert w.rank == 0


class TestWorldProperties:
    def test_is_global_leader_true(self):
        w = World(num_nodes=1, node=0, num_ranks=4, rank=0,
                  ranks_per_node=4, rank_of_node=0)
        assert w.is_global_leader is True

    def test_is_global_leader_false(self):
        w = World(num_nodes=1, node=0, num_ranks=4, rank=2,
                  ranks_per_node=4, rank_of_node=2)
        assert w.is_global_leader is False

    def test_is_local_leader_true(self):
        w = World(num_nodes=2, node=1, num_ranks=4, rank=2,
                  ranks_per_node=2, rank_of_node=0)
        assert w.is_local_leader is True

    def test_is_local_leader_false(self):
        w = World(num_nodes=2, node=0, num_ranks=4, rank=1,
                  ranks_per_node=2, rank_of_node=1)
        assert w.is_local_leader is False

    def test_default_workers(self):
        w = World(num_nodes=1, node=0, num_ranks=1, rank=0,
                  ranks_per_node=1, rank_of_node=0)
        assert w.num_workers == 1
        assert w.worker_of_rank == 0
