"""N1 (xdart write side): the processed-.nxs writer stores each frame's raw
``source/path`` RELATIVE to the project root when ``scan.source_base`` is set,
so the file is portable; absolute (back-compat) when no root is set.

Unit-level on ``_write_source_ref`` (the relativization), plus an end-to-end
round-trip: write a relative pointer + ``@source_base`` and resolve the raw back
through the ssrl reader.
"""
import os
from types import SimpleNamespace

import h5py
import numpy as np
import nexusformat.nexus as nx
import pytest

from xdart.modules.ewald.nexus_writer import _write_source_ref


def _path_value(fg):
    return str(fg["source"]["path"].nxvalue)


def test_write_source_ref_relative_under_source_base(tmp_path):
    root = tmp_path / "proj"
    raw = root / "raw" / "scan" / "img_0001.tif"
    raw.parent.mkdir(parents=True)
    raw.touch()
    frame = SimpleNamespace(source_file=str(raw), source_frame_idx=0, idx=1)
    fg = nx.NXcollection()
    _write_source_ref(fg, frame, source_base=str(root))
    assert _path_value(fg) == "raw/scan/img_0001.tif"     # POSIX relpath
    assert int(fg["source"]["frame_index"].nxvalue) == 0


def test_write_source_ref_absolute_without_source_base(tmp_path):
    raw = tmp_path / "raw" / "img.tif"
    raw.parent.mkdir(parents=True)
    raw.touch()
    frame = SimpleNamespace(source_file=str(raw), source_frame_idx=0, idx=2)
    fg = nx.NXcollection()
    _write_source_ref(fg, frame, source_base=None)         # back-compat
    assert os.path.isabs(_path_value(fg))
    assert _path_value(fg).endswith("raw/img.tif")


def test_write_source_ref_outside_root_is_absolute(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    raw = tmp_path / "elsewhere" / "img.tif"               # outside the root
    raw.parent.mkdir(parents=True)
    raw.touch()
    frame = SimpleNamespace(source_file=str(raw), source_frame_idx=0, idx=3)
    fg = nx.NXcollection()
    _write_source_ref(fg, frame, source_base=str(root))
    assert os.path.isabs(_path_value(fg))                  # out-of-tree -> absolute


def test_n1_writer_to_ssrl_reader_roundtrip(tmp_path):
    """End-to-end: a relative pointer written under @source_base resolves back to
    the raw master through the ssrl reader, even with the .nxs in another dir."""
    from ssrl_xrd_tools.io import get_raw_frame

    root = tmp_path / "proj"
    raw_arr = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    master = root / "raw" / "m.h5"
    master.parent.mkdir(parents=True)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw_arr)

    # Build the per-frame source pointer with the production writer helper.
    frame = SimpleNamespace(source_file=str(master), source_frame_idx=1, idx=0)
    fg = nx.NXcollection()
    _write_source_ref(fg, frame, source_base=str(root))
    rel = _path_value(fg)
    assert rel == "raw/m.h5"

    nxs = tmp_path / "processed" / "scan.nxs"
    nxs.parent.mkdir(parents=True)
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        from pathlib import Path
        e.attrs["source_base"] = Path(str(root)).as_posix()
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        s = e.create_group("frames/frame_0000/source")
        s.create_dataset("path", data=np.bytes_(rel.encode()))
        s.create_dataset("frame_index", data=1)

    np.testing.assert_allclose(get_raw_frame(nxs, frame=0), raw_arr[1])
