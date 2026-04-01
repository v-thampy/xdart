# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Other imports
from pandas import Series

# xdart imports
from xdart.utils import catch_h5py_file as catch
from xdart.utils.h5pool import get_pool

# This module imports
from .arch import EwaldArch


def _ensure_frames_group(h5file):
    """Create ``entry/frames`` group hierarchy if it doesn't exist."""
    entry = h5file.require_group("entry")
    entry.attrs.setdefault("NX_class", "NXentry")
    frames = entry.require_group("frames")
    frames.attrs.setdefault("NX_class", "NXprocess")
    return frames


class ArchSeries():
    """Container for storing EwaldArch objects in a NeXus-formatted HDF5 file.

    Data is stored on disk, not in memory. __getitem__ and __setitem__
    write/read integration results via the NeXus entry/frames/ structure.

    attributes:
        data_file: Path where hdf5 file is stored.
        file_lock: Thread safe lock.
        index: List of all arch id numbers.
    """
    def __init__(self, data_file, file_lock, arches=[],
                 static=False, gi=False, h5file=None):
        self.data_file = data_file
        self.file_lock = file_lock
        self.index = []
        self.static = static
        self.gi = gi
        if arches:
            for a in arches:
                self.__setitem__(a.idx, a, h5file=h5file)
        self._i = 0
        # Frames group is created lazily on first write (__setitem__)
        # to avoid creating empty files at startup.

    def __getitem__(self, idx):
        """Load an EwaldArch from the NeXus entry/frames/ group."""
        if idx in self.index:
            arch = EwaldArch(idx, static=self.static, gi=self.gi)
            with self.file_lock:
                with catch(self.data_file, 'r') as f:
                    frames_grp = f["entry/frames"]
                    arch.load_from_nexus(frames_grp)
            return arch
        else:
            raise KeyError(f"Arch not found with {idx} index")

    def iloc(self, idx):
        """Location based retrieval of arches instead of id based."""
        return self.__getitem__(self.index[idx])

    def __setitem__(self, idx, arch, h5file=None, global_mask=None):
        """Save an EwaldArch to HDF5 in NeXus format."""
        if h5file is not None:
            frames_grp = _ensure_frames_group(h5file)
            if idx != arch.idx:
                arch.idx = idx
            arch.save_to_nexus(frames_grp, global_mask=global_mask)
            if arch.idx not in self.index:
                self.index.append(arch.idx)
        else:
            pool = get_pool()
            pool.pause(self.data_file)
            try:
                with self.file_lock:
                    with catch(self.data_file, 'a') as f:
                        frames_grp = _ensure_frames_group(f)
                        if idx != arch.idx:
                            arch.idx = idx
                        arch.save_to_nexus(frames_grp, global_mask=global_mask)
                        if arch.idx not in self.index:
                            self.index.append(arch.idx)
            finally:
                pool.resume(self.data_file)

    def append(self, arch, h5file=None, global_mask=None):
        """Adds a new arch to the end of the index."""
        arches = ArchSeries(self.data_file, self.file_lock, h5file=h5file)
        arches.index = self.index[:]
        if isinstance(arch, Series):
            _arch = arch.iloc[0]
        else:
            _arch = arch
        arches.__setitem__(_arch.idx, _arch, h5file=h5file,
                           global_mask=global_mask)
        return arches

    def sort_index(self, inplace=False):
        """Sorts the index by idx."""
        if inplace:
            self.index.sort()
        else:
            arches = ArchSeries(self.data_file, self.file_lock)
            arches.index = self.index[:]
            arches.index.sort()
            return arches

    def __next__(self):
        if self._i < len(self.index):
            arch = self.iloc(self._i)
            self._i += 1
            return arch
        else:
            raise StopIteration

    def __iter__(self):
        self._i = 0
        return self
