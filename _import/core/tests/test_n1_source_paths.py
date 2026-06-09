"""N1 — portable raw-source paths: relative ``source/path`` + ``@source_base``,
reader resolution with a ``source_root=`` override, and back-compat.

The processed ``.nxs`` stores each frame's raw pointer RELATIVE to the project
root (``entry/@source_base``) so the file resolves its raw images after the data
moves machines.  ``relative_source_path`` is the write side; ``resolve_source_master``
/ ``get_raw_frame`` are the read side.  Precedence for a relative path:
explicit ``source_root`` > stored ``@source_base`` > the scan file's directory.
Absolute stored paths (old files / out-of-tree raw) keep loading unchanged.
"""
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from ssrl_xrd_tools.io import (
    get_raw_frame,
    relative_source_path,
    resolve_source_master,
)


# ── write-side: relative_source_path ────────────────────────────────────────

def test_relative_source_path_inside_root_is_posix_relpath(tmp_path):
    root = tmp_path / "project"
    src = root / "raw" / "scan" / "img_0001.tif"
    src.parent.mkdir(parents=True)
    src.touch()
    assert relative_source_path(src, root) == "raw/scan/img_0001.tif"


def test_relative_source_path_outside_root_is_absolute_and_warns(tmp_path, caplog):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "img.tif"
    outside.parent.mkdir(parents=True)
    outside.touch()
    with caplog.at_level("WARNING"):
        stored = relative_source_path(outside, root)
    assert os.path.isabs(stored) and stored.endswith("elsewhere/img.tif")
    assert any("outside the project root" in r.message for r in caplog.records)


def test_relative_source_path_no_root_is_absolute(tmp_path):
    src = tmp_path / "raw" / "img.tif"
    src.parent.mkdir(parents=True)
    src.touch()
    assert relative_source_path(src, None) == Path(src).resolve().as_posix()


# ── read-side: resolve_source_master precedence + back-compat ────────────────

def test_resolve_relative_against_source_base(tmp_path):
    root = tmp_path / "proj"
    raw = root / "raw" / "m.h5"
    raw.parent.mkdir(parents=True)
    raw.touch()
    nxs = tmp_path / "other" / "scan.nxs"          # .nxs NOT near the raw
    nxs.parent.mkdir(parents=True)
    got = resolve_source_master("raw/m.h5", scan_file=nxs, source_base=str(root))
    assert got == raw.resolve()


def test_resolve_source_root_overrides_source_base(tmp_path):
    # @source_base points at the OLD (now-missing) tree; source_root repoints.
    moved = tmp_path / "moved" / "raw" / "m.h5"
    moved.parent.mkdir(parents=True)
    moved.touch()
    nxs = tmp_path / "scan.nxs"
    got = resolve_source_master(
        "raw/m.h5", scan_file=nxs,
        source_base="/old/gone/proj", source_root=str(tmp_path / "moved"))
    assert got == moved.resolve()


def test_resolve_absolute_back_compat(tmp_path):
    raw = tmp_path / "m.h5"
    raw.touch()
    nxs = tmp_path / "scan.nxs"
    # No source_base at all (old file): absolute stored path is used as-is.
    got = resolve_source_master(str(raw), scan_file=nxs)
    assert got == raw.resolve()


def test_resolve_falls_back_to_scan_dir_when_no_base(tmp_path):
    raw = tmp_path / "m.h5"
    raw.touch()
    nxs = tmp_path / "scan.nxs"
    got = resolve_source_master("m.h5", scan_file=nxs)   # relative, scan-dir
    assert got == raw.resolve()


def test_resolve_missing_returns_none(tmp_path):
    nxs = tmp_path / "scan.nxs"
    assert resolve_source_master("nope/m.h5", scan_file=nxs,
                                 source_base=str(tmp_path)) is None


# ── end-to-end through get_raw_frame (the real reader) ───────────────────────

def _write_master(path, raw):
    with h5py.File(path, "w") as f:
        f.create_dataset("entry/data/data", data=raw)


def _write_processed(nxs, *, rel_path, source_base=None, frame_label=0,
                     master_frame=1):
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        if source_base is not None:
            e.attrs["source_base"] = source_base
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([frame_label], dtype=np.int64))
        s = e.create_group(f"frames/frame_{frame_label:04d}/source")
        s.create_dataset("path", data=np.bytes_(str(rel_path).encode()))
        s.create_dataset("frame_index", data=master_frame)


def test_get_raw_frame_resolves_relative_via_source_base(tmp_path):
    """The portable round-trip: a RELATIVE source/path + @source_base, with the
    .nxs in a DIFFERENT directory than the raw, still loads the full-res raw."""
    root = tmp_path / "proj"
    raw_arr = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    master = root / "raw" / "m.h5"
    master.parent.mkdir(parents=True)
    _write_master(master, raw_arr)

    nxs = tmp_path / "processed" / "scan.nxs"
    nxs.parent.mkdir(parents=True)
    rel = relative_source_path(master, root)                 # "raw/m.h5"
    assert rel == "raw/m.h5"
    _write_processed(nxs, rel_path=rel, source_base=str(root))

    np.testing.assert_allclose(get_raw_frame(nxs, frame=0), raw_arr[1])


def test_get_raw_frame_source_root_overrides_moved_tree(tmp_path):
    """@source_base points at the original root; after the data moves, the
    stored base is stale -> source_root= repoints and the raw loads again."""
    raw_arr = np.arange(4 * 4, dtype=float).reshape(1, 4, 4)
    moved_root = tmp_path / "moved"
    master = moved_root / "raw" / "m.h5"
    master.parent.mkdir(parents=True)
    _write_master(master, raw_arr)

    nxs = tmp_path / "scan.nxs"
    _write_processed(nxs, rel_path="raw/m.h5",
                     source_base="/original/gone/proj", master_frame=0)

    # Without the override the stale base can't resolve -> thumbnail/KeyError.
    with pytest.raises(KeyError):
        get_raw_frame(nxs, frame=0, allow_thumbnail=False)
    # With it, the raw resolves under the new root.
    np.testing.assert_allclose(
        get_raw_frame(nxs, frame=0, source_root=str(moved_root)), raw_arr[0])
