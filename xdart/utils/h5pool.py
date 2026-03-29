import h5py
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
    Supports multiple simultaneous open files for cross-scan comparison."""

    def __init__(self, max_open=5):
        self._files = OrderedDict()  # {path_str: h5py.File}
        self._max = max_open

    def get(self, path):
        """Return open read-only file handle. Opens if needed, evicts LRU if at capacity."""
        key = str(path)
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
        key = str(path)
        if key in self._files:
            try:
                self._files.pop(key).close()
            except Exception:
                pass

    def close_all(self):
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()
