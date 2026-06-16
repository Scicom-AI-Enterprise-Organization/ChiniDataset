"""Tests for chinidataset/util.py."""

import json
import pytest

from chinidataset.util import bytes_to_int, get_index_basename, merge_index


class TestBytesToInt:
    def test_plain_integer_string(self):
        assert bytes_to_int("512") == 512

    def test_kilobytes(self):
        assert bytes_to_int("100kb") == 100 * 1024

    def test_megabytes(self):
        assert bytes_to_int("1mb") == 1024 ** 2

    def test_gigabytes(self):
        assert bytes_to_int("2gb") == 2 * 1024 ** 3

    def test_uppercase(self):
        assert bytes_to_int("1GB") == 1024 ** 3

    def test_mixed_case(self):
        assert bytes_to_int("1Mb") == 1024 ** 2

    def test_integer_passthrough(self):
        assert bytes_to_int(1024) == 1024

    def test_shorthand_k(self):
        assert bytes_to_int("4k") == 4 * 1024

    def test_shorthand_m(self):
        assert bytes_to_int("2m") == 2 * 1024 ** 2

    def test_shorthand_g(self):
        assert bytes_to_int("1g") == 1024 ** 3

    def test_invalid_suffix_raises(self):
        with pytest.raises(ValueError):
            bytes_to_int("5zb")

    def test_zero(self):
        assert bytes_to_int("0") == 0

    def test_zero_int(self):
        assert bytes_to_int(0) == 0


class TestGetIndexBasename:
    def test_returns_index_json(self):
        assert get_index_basename() == "index.json"

    def test_is_string(self):
        assert isinstance(get_index_basename(), str)


class TestMergeIndex:
    def _make_partition(self, root, name, shards):
        """Write a sub-directory with its own index.json."""
        part_dir = root / name
        part_dir.mkdir(parents=True)
        index = {"version": 2, "shards": shards}
        with open(part_dir / "index.json", "w") as f:
            json.dump(index, f)
        return part_dir

    def _shard_entry(self, basename):
        return {
            "samples": 10,
            "raw_data": {"basename": basename, "bytes": 1024, "hashes": {}},
            "zip_data": None,
        }

    def test_basic_merge(self, tmp_path):
        self._make_partition(tmp_path, "00000", [self._shard_entry("shard.00000.parquet")])
        self._make_partition(tmp_path, "00001", [self._shard_entry("shard.00000.parquet")])

        merge_index(tmp_path)

        merged = json.loads((tmp_path / "index.json").read_text())
        assert len(merged["shards"]) == 2

    def test_relative_paths_in_basenames(self, tmp_path):
        self._make_partition(tmp_path, "00000", [self._shard_entry("shard.00000.parquet")])

        merge_index(tmp_path)

        merged = json.loads((tmp_path / "index.json").read_text())
        basename = merged["shards"][0]["raw_data"]["basename"]
        assert "00000" in basename
        assert "shard.00000.parquet" in basename

    def test_single_partition(self, tmp_path):
        self._make_partition(tmp_path, "00000", [self._shard_entry("shard.00000.parquet")])

        merge_index(tmp_path)

        merged = json.loads((tmp_path / "index.json").read_text())
        assert len(merged["shards"]) == 1

    def test_output_file_written(self, tmp_path):
        self._make_partition(tmp_path, "00000", [self._shard_entry("shard.00000.parquet")])
        merge_index(tmp_path)
        assert (tmp_path / "index.json").exists()

    def test_explicit_index_file_list(self, tmp_path):
        p0 = self._make_partition(tmp_path, "00000", [self._shard_entry("shard.00000.parquet")])
        p1 = self._make_partition(tmp_path, "00001", [self._shard_entry("shard.00000.parquet")])

        merge_index(tmp_path, index_file_urls=[
            str(p0 / "index.json"),
            str(p1 / "index.json"),
        ])

        merged = json.loads((tmp_path / "index.json").read_text())
        assert len(merged["shards"]) == 2

    def test_empty_shards_list(self, tmp_path):
        self._make_partition(tmp_path, "00000", [])
        merge_index(tmp_path)
        merged = json.loads((tmp_path / "index.json").read_text())
        assert merged["shards"] == []

    def test_missing_out_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            merge_index(tmp_path / "nonexistent")

    def test_no_index_files_found_raises(self, tmp_path):
        (tmp_path / "empty_dir").mkdir()
        with pytest.raises(FileNotFoundError):
            merge_index(tmp_path)

    def test_multiple_shards_per_partition(self, tmp_path):
        shards = [
            self._shard_entry("shard.00000.parquet"),
            self._shard_entry("shard.00001.parquet"),
        ]
        self._make_partition(tmp_path, "00000", shards)
        merge_index(tmp_path)
        merged = json.loads((tmp_path / "index.json").read_text())
        assert len(merged["shards"]) == 2
