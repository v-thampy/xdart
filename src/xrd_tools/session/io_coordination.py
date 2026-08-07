"""Process-wide, Qt-free HDF5 read-handle coordination."""

from __future__ import annotations

from collections import Counter, OrderedDict
import os
import threading

import h5py


def _pool_key(path):
    """Return the canonical absolute key shared by readers and writers."""
    return os.path.normcase(os.path.abspath(str(path)))


class H5FilePool:
    """Keep a bounded LRU of read-only HDF5 handles.

    Writers call :meth:`pause` before opening a file and :meth:`resume` after
    closing it. Pauses are refcounted so nested or concurrent writers keep the
    path excluded until every pause has a matching resume.
    """

    def __init__(self, max_open=5):
        self._files = OrderedDict()
        self._max = max_open
        self._lock = threading.Lock()
        self._paused = Counter()

    def get(self, path):
        """Return a read-only handle, or ``None`` while the path is paused."""
        key = _pool_key(path)
        with self._lock:
            if self._paused[key]:
                return None
            if key in self._files:
                self._files.move_to_end(key)
                handle = self._files[key]
                if handle.id.valid:
                    return handle
                del self._files[key]
            while len(self._files) >= self._max:
                _, old_handle = self._files.popitem(last=False)
                try:
                    old_handle.close()
                except Exception:
                    pass
            handle = h5py.File(key, "r")
            self._files[key] = handle
            return handle

    def close(self, path):
        """Best-effort close of the cached handle for one path."""
        key = _pool_key(path)
        with self._lock:
            if key in self._files:
                try:
                    self._files.pop(key).close()
                except Exception:
                    pass

    def pause(self, path):
        """Close the cached handle and prevent reopening until resumed."""
        key = _pool_key(path)
        with self._lock:
            self._paused[key] += 1
            if key in self._files:
                try:
                    self._files.pop(key).close()
                except Exception:
                    pass

    def resume(self, path):
        """Drop one pause depth; an unbalanced resume is a safe no-op."""
        key = _pool_key(path)
        with self._lock:
            depth = self._paused.get(key, 0)
            if depth <= 1:
                self._paused.pop(key, None)
            else:
                self._paused[key] = depth - 1

    def close_all(self):
        """Best-effort close of every cached read handle."""
        with self._lock:
            for handle in self._files.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self._files.clear()


# Module import is serialized by Python, so eager construction cannot race
# simultaneous first callers into manufacturing multiple process-wide pools.
_pool = H5FilePool(max_open=5)


def get_pool():
    """Return the one process-wide HDF5 read-handle pool."""
    return _pool


__all__ = ["H5FilePool", "get_pool"]
