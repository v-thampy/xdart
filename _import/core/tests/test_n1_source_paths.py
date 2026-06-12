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


# ── §3 read sites: FrameView / FrameSource / Scan resolve the portable path ──

def test_read_frame_view_resolves_source_path_via_source_base(tmp_path):
    """N1 §3: FrameView.source_path is RESOLVED (relative -> absolute master)
    against @source_base, so FrameView consumers (FrameSource/Scan/RSM/stitch/
    notebooks) can locate the raw -- not the bare stored relpath."""
    from ssrl_xrd_tools.io import read_frame_view
    root = tmp_path / "proj"
    master = root / "raw" / "m.h5"
    master.parent.mkdir(parents=True)
    _write_master(master, np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2))
    nxs = tmp_path / "other" / "scan.nxs"          # .nxs NOT near the raw
    nxs.parent.mkdir(parents=True)
    _write_processed(nxs, rel_path="raw/m.h5", source_base=str(root))

    fv = read_frame_view(nxs, 0)
    assert Path(fv.source_path) == master.resolve()


def test_read_frame_view_source_root_override(tmp_path):
    from ssrl_xrd_tools.io import read_frame_view
    moved = tmp_path / "moved"
    master = moved / "raw" / "m.h5"
    master.parent.mkdir(parents=True)
    _write_master(master, np.zeros((1, 2, 2)))
    nxs = tmp_path / "scan.nxs"
    _write_processed(nxs, rel_path="raw/m.h5", source_base="/old/gone/proj")
    fv = read_frame_view(nxs, 0, source_root=str(moved))
    assert Path(fv.source_path) == master.resolve()


def test_read_frame_view_unresolved_keeps_stored_string(tmp_path):
    """When nothing resolves, source_path keeps the stored relpath (provenance
    preserved, never silently blanked)."""
    from ssrl_xrd_tools.io import read_frame_view
    nxs = tmp_path / "scan.nxs"
    _write_processed(nxs, rel_path="raw/missing.h5", source_base=str(tmp_path))
    fv = read_frame_view(nxs, 0)
    assert fv.source_path == "raw/missing.h5"


def test_processed_nexus_source_load_frame_returns_full_res_master(tmp_path):
    """N1 §3 (+ fixes a pre-existing dead branch): ProcessedNexusSource.load_frame
    resolves the per-frame source pointer and returns the FULL-RES master (not the
    downsampled thumbnail) when it resolves."""
    from ssrl_xrd_tools.sources.nexus import ProcessedNexusSource
    root = tmp_path / "proj"
    master = root / "raw" / "m.h5"
    master.parent.mkdir(parents=True)
    raw = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    _write_master(master, raw)
    nxs = tmp_path / "processed" / "scan.nxs"
    nxs.parent.mkdir(parents=True)
    _write_processed(nxs, rel_path="raw/m.h5", source_base=str(root), master_frame=1)

    src = ProcessedNexusSource(nxs)
    img = src.load_frame(0)
    assert img.shape == (4, 4)                     # full-res, not a thumbnail
    np.testing.assert_allclose(img, raw[1])


def test_processed_nexus_source_source_root_repoints_moved_tree(tmp_path):
    from ssrl_xrd_tools.sources.nexus import ProcessedNexusSource
    moved = tmp_path / "moved"
    master = moved / "raw" / "m.h5"
    master.parent.mkdir(parents=True)
    raw = np.arange(4 * 4, dtype=float).reshape(1, 4, 4)
    _write_master(master, raw)
    nxs = tmp_path / "scan.nxs"
    _write_processed(nxs, rel_path="raw/m.h5", source_base="/old/gone", master_frame=0)
    src = ProcessedNexusSource(nxs, source_root=str(moved))
    np.testing.assert_allclose(src.load_frame(0), raw[0])


def test_open_scan_source_root_load_frame(tmp_path):
    """N1 §3: open_scan(nxs, source_root=...).load_frame resolves through the
    moved tree (the notebook sugar override)."""
    from ssrl_xrd_tools.io import open_scan
    moved = tmp_path / "moved"
    master = moved / "raw" / "m.h5"
    master.parent.mkdir(parents=True)
    raw = np.arange(4 * 4, dtype=float).reshape(1, 4, 4)
    _write_master(master, raw)
    nxs = tmp_path / "scan.nxs"
    _write_processed(nxs, rel_path="raw/m.h5", source_base="/old/gone", master_frame=0)
    scan = open_scan(nxs, source_root=str(moved))
    np.testing.assert_allclose(scan.load_frame(0), raw[0])


def test_processed_nexus_source_load_frame_strict_raises_without_master(tmp_path):
    """P1 #3 (codex): a headless FrameSource consumer must NEVER silently get a
    downsampled/mask-baked thumbnail in place of the full-res raw.  When the
    master can't be resolved, ProcessedNexusSource.load_frame RAISES (clean
    error) -- while the separate DISPLAY API still degrades to the thumbnail."""
    from ssrl_xrd_tools.sources.nexus import ProcessedNexusSource
    from ssrl_xrd_tools.io.image_source import load_processed_raw_or_thumbnail

    nxs = tmp_path / "scan.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        s = e.create_group("frames/frame_0000/source")
        s.create_dataset("path", data=np.bytes_(b"/gone/missing_master.h5"))
        s.create_dataset("frame_index", data=0)
        th = e.create_dataset("frames/frame_0000/thumbnail",
                              data=np.ones((4, 4), dtype=np.uint8))
        th.attrs["vmin"] = 0.0
        th.attrs["vmax"] = 1.0
        th.attrs["dtype"] = "uint8"

    src = ProcessedNexusSource(nxs)
    with pytest.raises(KeyError):              # strict: no silent thumbnail
        src.load_frame(0)

    # The display path is a SEPARATE API and still degrades to the thumbnail.
    res = load_processed_raw_or_thumbnail(nxs, 0)
    assert res.source == "thumbnail"
