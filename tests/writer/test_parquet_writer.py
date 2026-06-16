"""Tests for chinidataset/writer/parquet.py (ParquetWriter)."""

import json
import numpy as np
import pyarrow.parquet as pq
import pytest

from chinidataset.writer.parquet import ParquetWriter


def _simple_columns():
    return {"id": "int64", "text": "str", "score": "float64", "flag": "bool"}


def _make_samples(n=20):
    return [
        {"id": i, "text": f"sample {i}", "score": float(i) * 0.1, "flag": i % 2 == 0}
        for i in range(n)
    ]


class TestScalarTypes:
    def test_write_int(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"val": "int32"}) as w:
            w.write({"val": 42})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        assert table["val"][0].as_py() == 42

    def test_write_float(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"val": "float32"}) as w:
            w.write({"val": 3.14})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        assert abs(table["val"][0].as_py() - 3.14) < 1e-5

    def test_write_str(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"val": "str"}) as w:
            w.write({"val": "hello"})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        assert table["val"][0].as_py() == "hello"

    def test_write_bytes(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"val": "bytes"}) as w:
            w.write({"val": b"\x01\x02\x03"})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        assert table["val"][0].as_py() == b"\x01\x02\x03"

    def test_write_bool(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"val": "bool"}) as w:
            w.write({"val": True})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        assert table["val"][0].as_py() is True

    def test_write_mixed_columns(self, tmp_path):
        out = tmp_path / "d"
        columns = _simple_columns()
        samples = _make_samples(5)
        with ParquetWriter(out=out, columns=columns) as w:
            for s in samples:
                w.write(s)
        table = pq.read_table(str(out / "shard.00000.parquet"))
        assert table.num_rows == 5
        assert table["id"][2].as_py() == 2
        assert table["text"][0].as_py() == "sample 0"

    def test_shard_file_exists(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"x": "int64"}) as w:
            w.write({"x": 1})
        assert (out / "shard.00000.parquet").exists()

    def test_shard_readable_by_pyarrow(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"x": "int64"}) as w:
            for i in range(10):
                w.write({"x": i})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        assert table.num_rows == 10


class TestNumpyArrayTypes:
    def test_write_numpy_int64_array(self, tmp_path):
        out = tmp_path / "d"
        arr = np.array([10, 20, 30], dtype=np.int64)
        with ParquetWriter(out=out, columns={"tokens": "int64[]"}) as w:
            w.write({"tokens": arr})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        result = table["tokens"][0].as_py()
        assert result == [10, 20, 30]

    def test_write_numpy_float32_array(self, tmp_path):
        out = tmp_path / "d"
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with ParquetWriter(out=out, columns={"emb": "float32[]"}) as w:
            w.write({"emb": arr})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        result = table["emb"][0].as_py()
        assert all(abs(a - b) < 1e-5 for a, b in zip(result, [1.0, 2.0, 3.0]))

    def test_write_numpy_uint32_array(self, tmp_path):
        out = tmp_path / "d"
        arr = np.array([100, 200, 300], dtype=np.uint32)
        with ParquetWriter(out=out, columns={"ids": "uint32[]"}) as w:
            w.write({"ids": arr})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        result = table["ids"][0].as_py()
        assert result == [100, 200, 300]

    def test_variable_length_arrays(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"tokens": "int64[]"}) as w:
            w.write({"tokens": np.array([1, 2], dtype=np.int64)})
            w.write({"tokens": np.array([3, 4, 5, 6], dtype=np.int64)})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        assert table["tokens"][0].as_py() == [1, 2]
        assert table["tokens"][1].as_py() == [3, 4, 5, 6]


class TestWriteBatch:
    def test_write_batch_scalar(self, tmp_path):
        out = tmp_path / "d"
        data = {"id": np.arange(10, dtype=np.int64)}
        with ParquetWriter(out=out, columns={"id": "int64"}) as w:
            w.write_batch(data)
        table = pq.read_table(str(out / "shard.00000.parquet"))
        assert table.num_rows == 10
        assert table["id"][5].as_py() == 5

    def test_write_batch_equivalent_to_write_loop(self, tmp_path):
        cols = {"id": "int64", "score": "float32"}
        ids = np.arange(50, dtype=np.int64)
        scores = np.linspace(0, 1, 50, dtype=np.float32)

        out_loop = tmp_path / "loop"
        with ParquetWriter(out=out_loop, columns=cols) as w:
            for i in range(50):
                w.write({"id": int(ids[i]), "score": float(scores[i])})

        out_batch = tmp_path / "batch"
        with ParquetWriter(out=out_batch, columns=cols) as w:
            w.write_batch({"id": ids, "score": scores})

        t_loop = pq.read_table(str(out_loop / "shard.00000.parquet"))
        t_batch = pq.read_table(str(out_batch / "shard.00000.parquet"))
        assert t_loop["id"].to_pylist() == t_batch["id"].to_pylist()

    def test_write_batch_numpy_2d_array_column(self, tmp_path):
        out = tmp_path / "d"
        tokens = np.arange(30, dtype=np.int64).reshape(5, 6)
        with ParquetWriter(out=out, columns={"tokens": "int64[]"}) as w:
            w.write_batch({"tokens": tokens})
        table = pq.read_table(str(out / "shard.00000.parquet"))
        assert table.num_rows == 5
        assert table["tokens"][0].as_py() == list(range(6))

    def test_write_batch_empty(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"x": "int64"}) as w:
            w.write_batch({"x": np.array([], dtype=np.int64)})
        # No shard should be written
        assert not (out / "shard.00000.parquet").exists()


class TestShardMetadata:
    def test_sample_count_in_metadata(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"x": "int64"}) as w:
            for i in range(10):
                w.write({"x": i})
        index = json.loads((out / "index.json").read_text())
        assert index["shards"][0]["samples"] == 10

    def test_file_size_matches_disk(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"x": "int64"}) as w:
            for i in range(5):
                w.write({"x": i})
        index = json.loads((out / "index.json").read_text())
        actual_size = (out / "shard.00000.parquet").stat().st_size
        assert index["shards"][0]["raw_data"]["bytes"] == actual_size

    def test_hash_in_metadata(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"x": "int64"}, hashes=["sha256"]) as w:
            for i in range(3):
                w.write({"x": i})
        index = json.loads((out / "index.json").read_text())
        assert "sha256" in index["shards"][0]["raw_data"]["hashes"]
        assert len(index["shards"][0]["raw_data"]["hashes"]["sha256"]) == 64


class TestMultipleShards:
    def test_multiple_shards_written(self, tmp_path):
        out = tmp_path / "d"
        # Each sample ~40 bytes; size_limit 100 bytes → multiple shards
        with ParquetWriter(out=out, columns={"x": "int64"}, size_limit=100) as w:
            for i in range(20):
                w.write({"x": i})
        index = json.loads((out / "index.json").read_text())
        assert len(index["shards"]) > 1

    def test_all_samples_present_across_shards(self, tmp_path):
        out = tmp_path / "d"
        n = 50
        with ParquetWriter(out=out, columns={"x": "int64"}, size_limit=200) as w:
            for i in range(n):
                w.write({"x": i})
        index = json.loads((out / "index.json").read_text())
        total = sum(s["samples"] for s in index["shards"])
        assert total == n


@pytest.mark.slow
@pytest.mark.skipif(
    __import__("sys").platform == "darwin",
    reason="write_mp relies on fork semantics; macOS Python 3.12+ uses spawn by default",
)
class TestWriteMp:
    def test_write_mp_produces_all_samples(self, tmp_path):
        n = 100
        dataset = [{"x": i} for i in range(n)]
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"x": "int64"}) as w:
            w.write_mp(dataset, num_workers=2, use_tqdm=False)
        index = json.loads((out / "index.json").read_text())
        total = sum(s["samples"] for s in index["shards"])
        assert total == n

    def test_write_mp_no_duplicate_samples(self, tmp_path):
        n = 60
        dataset = [{"x": i} for i in range(n)]
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"x": "int64"}) as w:
            w.write_mp(dataset, num_workers=3, use_tqdm=False)

        all_values = []
        index = json.loads((out / "index.json").read_text())
        for shard in index["shards"]:
            basename = shard["raw_data"]["basename"]
            table = pq.read_table(str(out / basename))
            all_values.extend(table["x"].to_pylist())
        assert sorted(all_values) == list(range(n))

    def test_write_mp_invalid_dataset_raises(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"x": "int64"}) as w:
            with pytest.raises(TypeError):
                w.write_mp(iter(range(10)), num_workers=2, use_tqdm=False)

    def test_write_mp_invalid_num_workers_raises(self, tmp_path):
        out = tmp_path / "d"
        with ParquetWriter(out=out, columns={"x": "int64"}) as w:
            with pytest.raises(ValueError):
                w.write_mp([{"x": 1}], num_workers=0, use_tqdm=False)
