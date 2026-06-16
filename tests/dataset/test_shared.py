"""Tests for chinidataset/dataset/shared.py."""

import uuid
import numpy as np
import pytest

from chinidataset.dataset.shared import SharedArray, SharedScalar


def unique_name():
    return f"test_{uuid.uuid4().hex[:8]}"


class TestSharedArray:
    def test_create_and_read(self):
        name = unique_name()
        arr = SharedArray(10, np.int64, name)
        try:
            arr[0] = 42
            assert arr[0] == 42
        finally:
            arr.cleanup()

    def test_dtype_int32(self):
        name = unique_name()
        arr = SharedArray(5, np.int32, name)
        try:
            arr[2] = 999
            assert arr.numpy().dtype == np.int32
            assert arr[2] == 999
        finally:
            arr.cleanup()

    def test_dtype_float64(self):
        name = unique_name()
        arr = SharedArray(4, np.float64, name)
        try:
            arr[1] = 3.14
            assert abs(arr[1] - 3.14) < 1e-10
        finally:
            arr.cleanup()

    def test_values_correct(self):
        name = unique_name()
        n = 20
        arr = SharedArray(n, np.int64, name)
        try:
            for i in range(n):
                arr[i] = i * 2
            for i in range(n):
                assert arr[i] == i * 2
        finally:
            arr.cleanup()

    def test_len(self):
        name = unique_name()
        arr = SharedArray(7, np.int64, name)
        try:
            assert len(arr) == 7
        finally:
            arr.cleanup()

    def test_numpy_view(self):
        name = unique_name()
        arr = SharedArray(5, np.int32, name)
        try:
            view = arr.numpy()
            view[:] = np.arange(5, dtype=np.int32)
            assert arr[3] == 3
        finally:
            arr.cleanup()

    def test_cleanup_does_not_raise(self):
        name = unique_name()
        arr = SharedArray(3, np.uint8, name)
        arr.cleanup()  # should not raise


class TestSharedScalar:
    def test_int_round_trip(self):
        name = unique_name()
        s = SharedScalar(np.int64, name)
        try:
            s.set(12345)
            assert s.get() == 12345
        finally:
            s.arr.cleanup()

    def test_update_value(self):
        name = unique_name()
        s = SharedScalar(np.int64, name)
        try:
            s.set(1)
            assert s.get() == 1
            s.set(99)
            assert s.get() == 99
        finally:
            s.arr.cleanup()

    def test_float_scalar(self):
        name = unique_name()
        s = SharedScalar(np.float32, name)
        try:
            s.set(2.5)
            assert abs(s.get() - 2.5) < 1e-5
        finally:
            s.arr.cleanup()
