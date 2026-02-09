"""Cross-process shared memory arrays using Python's multiprocessing.shared_memory.

Adapted from: https://github.com/mosaicml/streaming/blob/main/streaming/base/shared/

These provide RAM-speed, cross-process shared state for coordinating multiple
DataLoader worker processes. Much faster than FileLock+JSON (nanoseconds vs milliseconds).
"""

import atexit
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from time import sleep
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

__all__ = ['SharedMemory', 'SharedArray', 'SharedScalar']

# Small sleep for retrying shared memory attachment
_TICK = 0.007


class SharedMemory:
    """Cross-process shared memory block with proper lifecycle management.

    Handles the tricky parts of Python's SharedMemory:
    - Creator process is responsible for unlinking (destroying)
    - Attaching processes should NOT unlink
    - Suppresses noisy resource tracker warnings

    Args:
        name (str): Unique name for the shared memory block.
        create (bool, optional): True to create new, False to attach, None to auto-detect.
        size (int): Size in bytes.
        auto_cleanup (bool): Register atexit cleanup handler. Defaults to True.
    """

    def __init__(
        self,
        name: str,
        create: Optional[bool] = None,
        size: int = 0,
        auto_cleanup: bool = True,
    ) -> None:
        self._created: list[BuiltinSharedMemory] = []
        self._opened: list[BuiltinSharedMemory] = []

        # Save and monkey-patch resource tracker to suppress warnings for attached shm
        original_register = resource_tracker.register

        try:
            if create is True:
                shm = BuiltinSharedMemory(name, True, size)
                self._created.append(shm)
            elif create is False:
                resource_tracker.register = self._noop_register
                shm = BuiltinSharedMemory(name, False, size)
                self._opened.append(shm)
            else:
                # Auto-detect: try create, fall back to attach
                try:
                    shm = BuiltinSharedMemory(name, True, size)
                    self._created.append(shm)
                except FileExistsError:
                    sleep(_TICK)
                    resource_tracker.register = self._noop_register
                    shm = BuiltinSharedMemory(name, False, size)
                    self._opened.append(shm)

            self._shm = shm
        finally:
            resource_tracker.register = original_register

        if auto_cleanup:
            atexit.register(self.cleanup)

    @property
    def buf(self) -> memoryview:
        """Shared memory buffer, accessible from any process."""
        return self._shm.buf

    def cleanup(self) -> None:
        """Clean up shared memory resources."""
        original_unregister = resource_tracker.unregister
        try:
            for shm in self._created:
                shm.close()
                shm.unlink()
            for shm in self._opened:
                resource_tracker.unregister = self._noop_unregister
                shm.close()
        except FileNotFoundError:
            pass
        finally:
            resource_tracker.unregister = original_unregister

    @staticmethod
    def _noop_register(name: str, rtype: str) -> None:
        if rtype == 'shared_memory':
            return
        resource_tracker._resource_tracker.register(name, rtype)

    @staticmethod
    def _noop_unregister(name: str, rtype: str) -> None:
        if rtype == 'shared_memory':
            return
        resource_tracker._resource_tracker.unregister(name, rtype)


class SharedArray:
    """A numpy array that lives in shared memory -- visible to all processes.

    RAM-speed reads and writes, no filesystem I/O.

    Args:
        shape (int | tuple): Array shape.
        dtype (type): Numpy dtype (e.g., np.uint8, np.uint64, np.float64).
        name (str): Unique shared memory name.
    """

    def __init__(self, shape: Union[int, tuple[int, ...]], dtype: type, name: str) -> None:
        self.shape = np.empty(shape).shape
        self.dtype = dtype
        self.name = name
        size = int(np.prod(self.shape) * dtype(0).nbytes)
        self.shm = SharedMemory(name=name, size=max(size, 1))

    def numpy(self) -> NDArray:
        """Get as a numpy array backed by shared memory."""
        return np.ndarray(self.shape, buffer=self.shm.buf, dtype=self.dtype)

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, index: Any) -> Any:
        return self.numpy()[index]

    def __setitem__(self, index: Any, value: Any) -> None:
        self.numpy()[index] = value

    def cleanup(self) -> None:
        self.shm.cleanup()


class SharedScalar:
    """A single scalar value in shared memory -- visible to all processes.

    Args:
        dtype (type): Numpy dtype.
        name (str): Unique shared memory name.
    """

    def __init__(self, dtype: type, name: str) -> None:
        self.arr = SharedArray(1, dtype, name)

    def get(self) -> Any:
        return self.arr[0]

    def set(self, value: Any) -> None:
        self.arr[0] = value
