"""Tests for chinidataset/writer/base.py (Writer ABC)."""

import json
import pytest

from chinidataset.writer.base import Writer


class StubWriter(Writer):
    """Minimal concrete Writer for testing the base class."""

    format = "stub"
    flush_count = 0

    def encode_sample(self, sample):
        # Return fixed-size bytes so size_limit logic is predictable
        return b"x" * 100

    def flush_shard(self):
        self.flush_count += 1
        shard = {
            "samples": len(self.new_samples),
            "raw_data": {
                "basename": f"shard.{len(self.shards):05d}.stub",
                "bytes": self.new_shard_size,
                "hashes": {},
            },
            "zip_data": None,
        }
        shard.update(self.get_config())
        self.shards.append(shard)


class TestWriterConstruction:
    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "new_dir"
        w = StubWriter(out=out)
        assert out.exists()

    def test_invalid_hash_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid hash"):
            StubWriter(out=tmp_path / "w", hashes=["notahash"])

    def test_unsorted_hashes_raises(self, tmp_path):
        with pytest.raises(ValueError, match="sorted"):
            StubWriter(out=tmp_path / "w", hashes=["sha256", "md5"])

    def test_invalid_size_limit_raises(self, tmp_path):
        with pytest.raises(ValueError):
            StubWriter(out=tmp_path / "w", size_limit="9zb")

    def test_size_limit_parsed(self, tmp_path):
        w = StubWriter(out=tmp_path / "w", size_limit="1mb")
        assert w.size_limit == 1024 ** 2

    def test_nonempty_dir_without_exist_ok_raises(self, tmp_path):
        out = tmp_path / "exists"
        out.mkdir()
        (out / "file.txt").write_text("data")
        with pytest.raises(ValueError, match="not empty"):
            StubWriter(out=out)

    def test_nonempty_dir_with_exist_ok_ok(self, tmp_path):
        out = tmp_path / "exists"
        out.mkdir()
        (out / "file.txt").write_text("data")
        w = StubWriter(out=out, exist_ok=True)
        assert w is not None


class TestWriterWrite:
    def test_flush_triggered_at_size_limit(self, tmp_path):
        # Each encoded sample = 100 bytes; size_limit = 250 bytes → flush after 3rd sample
        w = StubWriter(out=tmp_path / "w", size_limit=250)
        w.write({"x": 1})
        w.write({"x": 2})
        assert w.flush_count == 0
        w.write({"x": 3})  # pushes over limit
        assert w.flush_count == 1

    def test_no_flush_below_limit(self, tmp_path):
        w = StubWriter(out=tmp_path / "w", size_limit=10000)
        for _ in range(5):
            w.write({"x": 1})
        assert w.flush_count == 0

    def test_flush_increments_shard_count(self, tmp_path):
        w = StubWriter(out=tmp_path / "w", size_limit=200)
        for _ in range(6):
            w.write({"x": 1})
        assert len(w.shards) == 2


class TestWriterFinishAndIndex:
    def test_finish_writes_index_json(self, tmp_path):
        w = StubWriter(out=tmp_path / "w")
        w.write({"x": 1})
        w.finish()
        assert (tmp_path / "w" / "index.json").exists()

    def test_index_contains_shards(self, tmp_path):
        w = StubWriter(out=tmp_path / "w", size_limit=200)
        for _ in range(4):
            w.write({"x": 1})
        w.finish()
        index = json.loads((tmp_path / "w" / "index.json").read_text())
        assert len(index["shards"]) > 0

    def test_context_manager_calls_finish(self, tmp_path):
        with StubWriter(out=tmp_path / "w") as w:
            w.write({"x": 1})
        assert (tmp_path / "w" / "index.json").exists()

    def test_context_manager_exception_propagates(self, tmp_path):
        with pytest.raises(RuntimeError, match="boom"):
            with StubWriter(out=tmp_path / "w") as w:
                w.write({"x": 1})
                raise RuntimeError("boom")

    def test_finish_idempotent(self, tmp_path):
        w = StubWriter(out=tmp_path / "w")
        w.write({"x": 1})
        w.finish()
        w.finish()  # should not raise or double-write
        index = json.loads((tmp_path / "w" / "index.json").read_text())
        assert isinstance(index, dict)
