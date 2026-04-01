import h5py
import threading
from collections import OrderedDict


# Module-level singleton — import and use this from anywhere that needs
# to coordinate read handles with write operations.
_pool = None


def get_pool():
    """Return the process-wide H5FilePool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = H5FilePool(max_open=5)
    return _pool


class H5FilePool:
    """Keeps HDF5 files open in read-only mode for fast repeated access.

    Thread-safety: a threading.Lock guards all mutations.  Writers should call
    ``pause(path)`` before opening a file for writing and ``resume(path)``
    after closing it.  While paused, ``get(path)`` will **not** reopen the
    file, returning ``None`` instead.
    """

    def __init__(self, max_open=5):
        self._files = OrderedDict()  # {path_str: h5py.File}
        self._max = max_open
        self._lock = threading.Lock()
        self._paused = set()  # paths that should not be (re)opened

    def get(self, path):
        """Return open read-only file handle, or *None* if the path is paused.

        Opens the file if needed, evicts LRU if at capacity.
        """
        key = str(path)
        with self._lock:
            if key in self._paused:
                return None
            if key in self._files:
                self._files.move_to_end(key)
                f = self._files[key]
                if f.id.valid:
                    return f
                del self._files[key]
            while len(self._files) >= self._max:
                _, old_f = self._files.popitem(last=False)
                try:
                    old_f.close()
                except Exception:
                    pass
            f = h5py.File(key, 'r')
            self._files[key] = f
            return f

    def close(self, path):
        """Close a cached read handle (e.g. before writing)."""
        key = str(path)
        with self._lock:
            if key in self._files:
                try:
                    self._files.pop(key).close()
                except Exception:
                    pass

    def pause(self, path):
        """Close the read handle *and* prevent ``get()`` from reopening it.

        Call this before opening a file for writing.
        """
        key = str(path)
        with self._lock:
            self._paused.add(key)
            if key in self._files:
                try:
                    self._files.pop(key).close()
                except Exception:
                    pass

    def resume(self, path):
        """Allow ``get()`` to open the file again after a write is done."""
        key = str(path)
        with self._lock:
            self._paused.discard(key)

    def close_all(self):
        with self._lock:
            for f in self._files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._files.clear()
