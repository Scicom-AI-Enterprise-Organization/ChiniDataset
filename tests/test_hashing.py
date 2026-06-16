"""Tests for chinidataset/hashing.py."""

import hashlib
import pytest

from chinidataset.hashing import get_hash, get_hashes, is_hash


class TestIsHash:
    def test_sha256_supported(self):
        assert is_hash("sha256") is True

    def test_md5_supported(self):
        assert is_hash("md5") is True

    def test_sha1_supported(self):
        assert is_hash("sha1") is True

    def test_invalid_algo(self):
        assert is_hash("fakehash999") is False

    def test_empty_string(self):
        assert is_hash("") is False

    def test_xxhash_supported_if_installed(self):
        try:
            import xxhash
            assert is_hash("xxh64") is True
        except ImportError:
            pytest.skip("xxhash not installed")


class TestGetHashes:
    def test_returns_set(self):
        assert isinstance(get_hashes(), set)

    def test_contains_sha256(self):
        assert "sha256" in get_hashes()

    def test_contains_md5(self):
        assert "md5" in get_hashes()

    def test_non_empty(self):
        assert len(get_hashes()) > 0


class TestGetHash:
    def test_sha256_known_digest(self):
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert get_hash("sha256", data) == expected

    def test_md5_known_digest(self):
        data = b"hello world"
        expected = hashlib.md5(data).hexdigest()
        assert get_hash("md5", data) == expected

    def test_deterministic(self):
        data = b"test data 123"
        assert get_hash("sha256", data) == get_hash("sha256", data)

    def test_different_data_different_digest(self):
        assert get_hash("sha256", b"aaa") != get_hash("sha256", b"bbb")

    def test_empty_bytes(self):
        result = get_hash("sha256", b"")
        assert result == hashlib.sha256(b"").hexdigest()

    def test_invalid_algo_raises(self):
        with pytest.raises(ValueError, match="not a supported hash algorithm"):
            get_hash("notreal", b"data")

    def test_returns_hex_string(self):
        result = get_hash("sha256", b"data")
        assert isinstance(result, str)
        int(result, 16)  # should not raise

    def test_sha1_known_digest(self):
        data = b"chinidataset"
        expected = hashlib.sha1(data).hexdigest()
        assert get_hash("sha1", data) == expected
