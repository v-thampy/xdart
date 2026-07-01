"""N1 (xdart write side): the processed-.nxs writer stores each frame's raw
``source/path`` RELATIVE to the project root when ``scan.source_base`` is set,
so the file is portable; absolute (back-compat) when no root is set.

Unit-level on the production seam — xdart's ``_resolved_frame_source`` /
``_frame_source_index`` (LiveFrame attribute extraction) feeding the core
``write_frame_source_ref`` (the relativization, moved to
``xrd_tools.io.nexus_record`` in the 6a monorepo refactor) — plus an
end-to-end round-trip: write a relative pointer + ``@source_base`` and
resolve the raw back through the reader.
"""
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import xdart.gui.gui_utils  # noqa: F401  # registers the 'str_browse' param type (the live GUI imports gui_utils at startup; the wrangler-only import path here does not)

from xdart.modules.ewald.nexus_writer import (
    _frame_source_index,
    _resolved_frame_source,
)
from xrd_tools.io.nexus_record import write_frame_source_ref


def _write_ref(tmp_path, frame, source_base):
    """Drive the production write path for one frame's source pointer."""
    p = tmp_path / "ref.nxs"
    with h5py.File(p, "w") as f:
        fg = f.create_group("entry/frames/frame_0000")
        write_frame_source_ref(
            fg, _resolved_frame_source(frame, None),
            _frame_source_index(frame), source_base=source_base,
        )
        path = fg["source/path"][()]
        return (path.decode() if isinstance(path, bytes) else str(path),
                int(fg["source/frame_index"][()]))


def test_write_source_ref_relative_under_source_base(tmp_path):
    root = tmp_path / "proj"
    raw = root / "raw" / "scan" / "img_0001.tif"
    raw.parent.mkdir(parents=True)
    raw.touch()
    frame = SimpleNamespace(source_file=str(raw), source_frame_idx=0, idx=1)
    path, fi = _write_ref(tmp_path, frame, str(root))
    assert path == "raw/scan/img_0001.tif"                 # POSIX relpath
    assert fi == 0


def test_write_source_ref_absolute_without_source_base(tmp_path):
    raw = tmp_path / "raw" / "img.tif"
    raw.parent.mkdir(parents=True)
    raw.touch()
    frame = SimpleNamespace(source_file=str(raw), source_frame_idx=0, idx=2)
    path, _ = _write_ref(tmp_path, frame, None)            # back-compat
    assert os.path.isabs(path)
    assert path.endswith("raw/img.tif")


def test_write_source_ref_outside_root_is_absolute(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    raw = tmp_path / "elsewhere" / "img.tif"               # outside the root
    raw.parent.mkdir(parents=True)
    raw.touch()
    frame = SimpleNamespace(source_file=str(raw), source_frame_idx=0, idx=3)
    path, _ = _write_ref(tmp_path, frame, str(root))
    assert os.path.isabs(path)                             # out-of-tree -> absolute


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

    proj = tmp_path / "proj"
    proj.mkdir()                       # N1: source_base requires an EXISTING dir
    folder = str(proj)
    root.child('Project').child('project_folder').setValue(folder)
    assert compute() == os.path.abspath(folder)
    # And it's session-persisted (so a relaunch restores the portable root).
    assert any(p[0] == 'project_folder' for p in imageWrangler._SESSION_PARAMS)


def test_n1_writer_to_ssrl_reader_roundtrip(tmp_path):
    """End-to-end: a relative pointer written under @source_base resolves back to
    the raw master through the ssrl reader, even with the .nxs in another dir."""
    from xrd_tools.io import get_raw_frame

    root = tmp_path / "proj"
    raw_arr = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    master = root / "raw" / "m.h5"
    master.parent.mkdir(parents=True)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw_arr)

    # Build the record with the production primitives end-to-end.
    from xrd_tools.io.nexus_record import (
        ensure_frames_container, stamp_source_base, write_frame_record,
    )
    frame = SimpleNamespace(source_file=str(master), source_frame_idx=1, idx=0)

    nxs = tmp_path / "processed" / "scan.nxs"
    nxs.parent.mkdir(parents=True)
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        base = stamp_source_base(e, str(root))
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        write_frame_record(
            ensure_frames_container(e), "frame_0000",
            source_path=_resolved_frame_source(frame, None),
            source_frame_index=_frame_source_index(frame),
            source_base=base,
        )
        stored = e["frames/frame_0000/source/path"][()]
        assert (stored.decode() if isinstance(stored, bytes)
                else str(stored)) == "raw/m.h5"

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


def test_nexus_wrangler_thread_initialize_scan_sets_source_base():
    """N1 (nexus-wrangler portability): nexusThread._initialize_scan stamps
    scan.source_base from the worker, which the writer reads to store relative
    raw paths + @source_base.  None -> absolute (back-compat)."""
    from types import SimpleNamespace, MethodType
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import nexusThread

    scan = SimpleNamespace(name=None, gi=None, static=None)
    t = SimpleNamespace(scan=scan, gi=True, source_base="/proj")
    t._initialize_scan = MethodType(nexusThread._initialize_scan, t)
    assert t._initialize_scan("s").source_base == "/proj"

    # No source_base on the worker -> None (back-compat, absolute paths).
    t2 = SimpleNamespace(scan=SimpleNamespace(name=None, gi=None, static=None), gi=False)
    t2._initialize_scan = MethodType(nexusThread._initialize_scan, t2)
    assert getattr(t2._initialize_scan("s"), "source_base", "X") is None


def test_nexus_wrangler_compute_source_base(tmp_path):
    """N1: the nexus wrangler derives source_base from its Project Folder param
    (Project is the first group), mirroring the image wrangler."""
    from types import SimpleNamespace, MethodType
    from pyqtgraph import QtWidgets
    from pyqtgraph.parametertree import Parameter
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler import nexusWrangler, params

    root = Parameter.create(name="p", type="group", children=params)
    assert root.children()[0].name() == "Project"
    h = SimpleNamespace(parameters=root)
    h._compute_source_base = MethodType(nexusWrangler._compute_source_base, h)
    assert h._compute_source_base() is None
    proj = tmp_path / "p"
    proj.mkdir()                       # N1: source_base requires an EXISTING dir
    root.child("Project").child("project_folder").setValue(str(proj))
    assert h._compute_source_base() == os.path.abspath(str(proj))


def test_writer_rejects_append_with_mismatched_source_base(tmp_path):
    """P2 #4 (codex): appending to a .nxs written under a DIFFERENT Project
    Folder (@source_base) must FAIL LOUD -- one scan-level @source_base governs
    ALL frames' relative paths, so overwriting it on append would silently rebase
    the earlier frames against the new root."""
    import nexusformat.nexus as nx
    from types import SimpleNamespace
    from xdart.modules.ewald.nexus_writer import _write_per_frame_metadata

    nxs = tmp_path / "scan.nxs"
    with h5py.File(nxs, "w") as fh:
        e = fh.create_group("entry")
        e.attrs["source_base"] = "/root/A"          # written under Project Folder A
        e.create_group("frames")

    scan = SimpleNamespace(frames=SimpleNamespace(index=[0]),
                           source_base="/root/B")    # appending under a DIFFERENT root
    with nx.nxopen(str(nxs), "a") as f:
        with pytest.raises(ValueError, match="differs"):
            _write_per_frame_metadata(f, scan, entry="entry")

    # Same root appends fine (no mismatch -> guard passes; the @source_base stays).
    scan_same = SimpleNamespace(frames=SimpleNamespace(index=[]),  # empty -> early return
                                source_base="/root/A")
    with nx.nxopen(str(nxs), "a") as f:
        _write_per_frame_metadata(f, scan_same, entry="entry")      # must not raise
