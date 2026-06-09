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


def test_image_wrangler_project_folder_derives_source_base(tmp_path):
    """N1 GUI wiring: the image wrangler's Project Folder param drives
    ``_compute_source_base`` -- the absolute root when set, None when blank
    (-> the writer stores absolute paths, back-compat)."""
    from types import SimpleNamespace
    from pyqtgraph import QtWidgets
    from pyqtgraph.parametertree import Parameter
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler, params)

    root = Parameter.create(name='p', type='group', children=params)
    holder = SimpleNamespace(parameters=root)
    compute = imageWrangler._compute_source_base.__get__(holder)

    root.child('Project').child('project_folder').setValue('')
    assert compute() is None
    root.child('Project').child('project_folder').setValue('   ')   # whitespace
    assert compute() is None

    folder = str(tmp_path / "proj")
    root.child('Project').child('project_folder').setValue(folder)
    assert compute() == os.path.abspath(folder)
    # And it's session-persisted (so a relaunch restores the portable root).
    assert any(p[0] == 'project_folder' for p in imageWrangler._SESSION_PARAMS)


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


def test_load_frame_v2_resolves_relative_source_against_source_base(tmp_path):
    """N1 reload (xdart-internal): a portable .nxs stores source_file RELATIVE to
    the project root (@source_base), which is NOT the .nxs directory (the default
    output is <root>/xdart_processed_data).  _load_frame_v2 must resolve the
    relative source_file against @source_base, not the .nxs dir -- else lazy raw
    reload / is_reload_only detection silently breaks after a Project-Folder run."""
    from pathlib import Path
    from xdart.modules.ewald.frame_series import _load_frame_v2

    root = tmp_path / "proj"
    raw = root / "raw" / "img_0001.tif"
    raw.parent.mkdir(parents=True)
    raw.touch()
    nxs_dir = root / "xdart_processed_data"      # .nxs lives BELOW the root
    nxs_dir.mkdir()
    nxs = nxs_dir / "scan.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        e.attrs["source_base"] = Path(str(root)).as_posix()
        s = e.create_group("frames/frame_0001/source")
        s.create_dataset("path", data=np.bytes_(b"raw/img_0001.tif"))
        s.create_dataset("frame_index", data=0)

    with h5py.File(nxs, "r") as f:
        # source_root = the .nxs dir (the OLD behavior); @source_base must win.
        frame = _load_frame_v2(f, 1, static=True, gi=False,
                               source_root=str(nxs_dir))
    assert frame._resolved_source_path() == os.path.normpath(str(raw))
    assert frame.is_reload_only is False         # the raw exists under the root


def test_load_frame_v2_absolute_source_back_compat(tmp_path):
    """Old absolute-path .nxs (no @source_base): the absolute source_file is used
    as-is regardless of source_root -- back-compat preserved."""
    from xdart.modules.ewald.frame_series import _load_frame_v2

    raw = tmp_path / "raw" / "img.tif"
    raw.parent.mkdir(parents=True)
    raw.touch()
    nxs = tmp_path / "scan.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")                      # NO source_base
        s = e.create_group("frames/frame_0002/source")
        s.create_dataset("path", data=np.bytes_(str(raw).encode()))
        s.create_dataset("frame_index", data=0)
    with h5py.File(nxs, "r") as f:
        frame = _load_frame_v2(f, 2, static=True, gi=False,
                               source_root=str(tmp_path))
    assert frame._resolved_source_path() == str(raw)
    assert frame.is_reload_only is False
