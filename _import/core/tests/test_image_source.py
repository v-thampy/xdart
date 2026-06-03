"""Tests for the image-source classification + loading boundary
(``io.image_source``) that the xdart Image Viewer consumes.

Hand-builds small files with pure h5py: a processed v2 ``.nxs`` whose
frames point back to a raw master, a thumbnail-only processed ``.nxs``
(source master missing), a raw detector master, and an unknown file.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from ssrl_xrd_tools.io import (
    ImageSourceKind,
    classify_image_source,
    load_image_frame,
    load_processed_raw_or_thumbnail,
)


def _write_thumbnail(group, name, data):
    """Store a uint8-quantized thumbnail with the dequantize attrs."""
    vmin, vmax = float(data.min()), float(data.max())
    span = (vmax - vmin) or 1.0
    q = np.clip((data - vmin) / span, 0, 1) * 255.0
    ds = group.create_dataset(name, data=q.astype(np.uint8))
    ds.attrs["vmin"] = vmin
    ds.attrs["vmax"] = vmax
    ds.attrs["dtype"] = "uint8"


@pytest.fixture
def processed_with_master(tmp_path):
    """Processed .nxs whose frame 0 resolves to a sibling raw master."""
    master = tmp_path / "scan_master.h5"
    raw = np.arange(2 * 8 * 8, dtype=float).reshape(2, 8, 8)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw)

    nxs = tmp_path / "scan.nxs"
    thumb = np.linspace(0, 100, 16 * 16).reshape(16, 16)
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        s = e.create_group("frames/frame_0000/source")
        s.create_dataset("path", data=np.bytes_(b"scan_master.h5"))
        s.create_dataset("frame_index", data=1)   # -> master frame 1
        _write_thumbnail(e["frames/frame_0000"], "thumbnail", thumb)
    return nxs, raw, thumb


@pytest.fixture
def thumbnail_only(tmp_path):
    """Processed .nxs whose source master is missing — only the thumbnail."""
    nxs = tmp_path / "thumb_only.nxs"
    thumb = np.linspace(5, 50, 16 * 16).reshape(16, 16)
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        s = e.create_group("frames/frame_0000/source")
        s.create_dataset("path", data=np.bytes_(b"does_not_exist.h5"))
        s.create_dataset("frame_index", data=0)
        _write_thumbnail(e["frames/frame_0000"], "thumbnail", thumb)
    return nxs, thumb


@pytest.fixture
def raw_master(tmp_path):
    nxs = tmp_path / "raw.h5"
    raw = np.arange(3 * 8 * 8, dtype=float).reshape(3, 8, 8)
    with h5py.File(nxs, "w") as f:
        f.create_dataset("entry/data/data", data=raw)
    return nxs, raw


# ── classify_image_source ─────────────────────────────────────────────

def test_classify_processed_with_master(processed_with_master):
    nxs, _raw, _thumb = processed_with_master
    info = classify_image_source(nxs)
    assert info.kind is ImageSourceKind.PROCESSED_XDART
    assert info.has_raw is True
    assert info.has_thumbnail is True
    assert info.frame_labels == (0,)
    assert info.n_frames == 1


def test_classify_thumbnail_only(thumbnail_only):
    nxs, _thumb = thumbnail_only
    info = classify_image_source(nxs)
    assert info.kind is ImageSourceKind.THUMBNAIL_ONLY
    assert info.has_raw is False
    assert info.has_thumbnail is True


def test_classify_raw_master(raw_master):
    nxs, raw = raw_master
    info = classify_image_source(nxs)
    assert info.kind is ImageSourceKind.RAW_MASTER
    assert info.has_raw is True
    assert info.n_frames == raw.shape[0]


def test_classify_unknown(tmp_path):
    p = tmp_path / "empty.nxs"
    with h5py.File(p, "w") as f:
        f.create_group("entry")          # no integrated/frames/reduction/raw
    info = classify_image_source(p)
    assert info.kind is ImageSourceKind.UNKNOWN


# ── load_processed_raw_or_thumbnail ───────────────────────────────────

def test_load_processed_returns_raw_when_master_resolves(processed_with_master):
    nxs, raw, _thumb = processed_with_master
    res = load_processed_raw_or_thumbnail(nxs, 0)
    assert res.source == "raw"
    assert res.frame == 0
    np.testing.assert_allclose(res.image, raw[1])   # source/frame_index -> 1


def test_load_processed_falls_back_to_thumbnail(thumbnail_only):
    nxs, thumb = thumbnail_only
    res = load_processed_raw_or_thumbnail(nxs, 0)
    assert res.source == "thumbnail"
    # dequantized thumbnail recovers the original range (within quantization)
    assert res.image.shape == thumb.shape
    np.testing.assert_allclose(res.image, thumb, atol=1.0)


def test_load_processed_none_when_nothing_available(tmp_path):
    nxs = tmp_path / "barren.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        e.create_group("frames/frame_0000")     # frame group, no source, no thumbnail
    res = load_processed_raw_or_thumbnail(nxs, 0)
    assert res.source == "none"
    assert res.image is None


def test_load_processed_direct_thumbnail_when_get_raw_frame_errors(
        thumbnail_only, monkeypatch):
    # Belt-and-suspenders: if get_raw_frame itself errors outright (not a
    # clean no-master fallthrough), a stored thumbnail is still read directly.
    nxs, thumb = thumbnail_only
    import ssrl_xrd_tools.io.read as read_mod

    def boom(*a, **k):
        raise RuntimeError("get_raw_frame is broken")

    monkeypatch.setattr(read_mod, "get_raw_frame", boom)
    res = load_processed_raw_or_thumbnail(nxs, 0)
    assert res.source == "thumbnail"
    np.testing.assert_allclose(res.image, thumb, atol=1.0)


# ── frame_labels are the displayable frames-group labels (guard a/b) ───

@pytest.fixture
def processed_gapped(tmp_path):
    """Processed .nxs whose integrated frame_index is a SUPERSET of the
    frame groups that actually carry a thumbnail/source — the eiger-style
    case that blanked the Image Viewer (union labels lack a frame group)."""
    master = tmp_path / "scan_master.h5"
    raw = np.arange(3 * 8 * 8, dtype=float).reshape(3, 8, 8)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw)

    nxs = tmp_path / "gapped.nxs"
    thumb = np.linspace(0, 100, 16 * 16).reshape(16, 16)
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((3, 5)))
        # integrated lists labels 0,1,2 ...
        g.create_dataset("frame_index", data=np.array([0, 1, 2], dtype=np.int64))
        # ... but only frames 1 and 2 have a displayable group.
        for lbl, src_idx in ((1, 0), (2, 1)):
            s = e.create_group(f"frames/frame_{lbl:04d}/source")
            s.create_dataset("path", data=np.bytes_(b"scan_master.h5"))
            s.create_dataset("frame_index", data=src_idx)
            _write_thumbnail(e[f"frames/frame_{lbl:04d}"], "thumbnail", thumb)
    return nxs, raw


def test_frame_labels_are_displayable_group_labels(processed_gapped):
    # guard (a): frame_labels are the frames-group labels (1,2), NOT the
    # integrated union (0,1,2) — label 0 has no group and must be excluded.
    nxs, _raw = processed_gapped
    info = classify_image_source(nxs)
    assert info.frame_labels == (1, 2)


def test_every_frame_label_loads_non_none_source_resolvable(processed_gapped):
    # guard (b): every label classify reports as displayable loads non-None.
    nxs, _raw = processed_gapped
    info = classify_image_source(nxs)
    assert info.frame_labels                       # not empty
    for lbl in info.frame_labels:
        res = load_processed_raw_or_thumbnail(nxs, lbl)
        assert res.image is not None, f"frame {lbl} returned no image"
        assert res.source in ("raw", "thumbnail")


def test_every_frame_label_loads_non_none_thumbnail_only(thumbnail_only):
    nxs, _thumb = thumbnail_only
    info = classify_image_source(nxs)
    assert info.frame_labels                       # the thumbnail frame(s)
    for lbl in info.frame_labels:
        res = load_processed_raw_or_thumbnail(nxs, lbl)
        assert res.image is not None and res.source == "thumbnail"


# ── raw detector data with a NATIVE entry/frames group (regression) ────

def test_eiger_data_with_native_frames_group_is_raw_master(tmp_path):
    """Eiger data files carry a raw ``entry/data/data`` stack AND a native
    ``entry/frames`` group.  ``entry/frames`` alone must NOT mark the file
    processed-xdart — the raw dataset wins → RAW_MASTER (regression: these
    files were misclassified and the Image Viewer refused to show them)."""
    p = tmp_path / "eiger_w2s3_scan001_data_000001.h5"
    raw = np.arange(5 * 8 * 8, dtype=np.float32).reshape(5, 8, 8)
    with h5py.File(p, "w") as f:
        f.create_dataset("entry/data/data", data=raw)
        f.create_group("entry/frames")          # native eiger group, no content
    info = classify_image_source(p)
    assert info.kind is ImageSourceKind.RAW_MASTER
    assert info.has_raw and info.frame_labels == (0, 1, 2, 3, 4)


def test_eiger_master_is_raw_master(tmp_path):
    """A ``*_master.h5`` (data in linked files + a native frames group) is
    raw detector data, classified by the eiger-master name."""
    p = tmp_path / "eiger_w2s3_scan001_master.h5"
    with h5py.File(p, "w") as f:
        f.create_group("entry/frames")
        f.create_group("entry/instrument/detector")
    info = classify_image_source(p)
    assert info.kind is ImageSourceKind.RAW_MASTER


# ── load_image_frame ──────────────────────────────────────────────────

def test_load_image_frame_reads_raw_master(raw_master):
    nxs, raw = raw_master
    img = load_image_frame(nxs, 2)
    np.testing.assert_allclose(img, raw[2])
