"""Source-kind inference and concrete ``open_source`` routing contracts.

Adapter registration, discovery, ownership, and override behavior are covered
in ``test_source_format_adapters.py``.  This module keeps the complementary
kind-inference, pass-through, and real source-opening checks.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources import guess_source_kind, open_source


class _FakeSource:
    """Minimal FrameSource duck for the pass-through contract (no I/O)."""

    def __init__(self, spec):
        self.spec = spec
        self.frame_indices = range(1)

    def load_frame(self, index):
        return np.zeros((2, 2))


# ---- open_source contract --------------------------------------------------

def test_open_source_passes_through_an_existing_framesource():
    fs = _FakeSource(SourceSpec("mem://x", SourceKind.MEMORY))
    assert open_source(fs) is fs


def test_open_source_unknown_kind_raises_cleanly():
    with pytest.raises(ValueError):
        open_source(SourceSpec("/x.weird", SourceKind.UNKNOWN))


# ---- guess_source_kind dispatch table --------------------------------------

def test_guess_source_kind_directory_is_a_tiff_series(tmp_path):
    d = tmp_path / "series"
    d.mkdir()
    assert guess_source_kind(d) is SourceKind.TIFF_SERIES


def test_guess_source_kind_hdf5_family_is_a_nexus_stack(tmp_path):
    # Extension fallback: even a non-HDF5 file with an h5/nexus extension routes
    # to NEXUS_STACK (the opener validates content); pins the container policy.
    for ext in (".nxs", ".h5", ".hdf5", ".cxi"):
        p = tmp_path / f"f{ext}"
        p.write_text("placeholder")
        assert guess_source_kind(p) is SourceKind.NEXUS_STACK, ext


def test_guess_source_kind_tif_is_an_image_file(tmp_path):
    p = tmp_path / "f.tif"
    p.write_bytes(b"II*\x00")  # minimal TIFF-ish header; classification is by ext
    assert guess_source_kind(p) is SourceKind.IMAGE_FILE


# ---- the Bluesky worked example (real file; skip without test data) --------

_REAL = __import__("pathlib").Path(
    __import__("os").environ.get(
        "XDART_TEST_DATA",
        __import__("pathlib").Path(__file__).resolve().parents[2] / "test_data")
) / "nexus" / "LaB6_0710_1025pm_00005.nxs"


@pytest.mark.skipif(not _REAL.exists(), reason=f"real Bluesky file not found: {_REAL}")
def test_guess_source_kind_bluesky_nxs_is_a_nexus_stack():
    """A real Bluesky/NXWriter .nxs classifies as NEXUS_STACK and opens through
    the adapter-backed source-opening seam."""
    assert guess_source_kind(_REAL) is SourceKind.NEXUS_STACK
    src = open_source(SourceSpec(_REAL, SourceKind.NEXUS_STACK, entry="entry"))
    assert len(list(src.frame_indices)) == 3


# ---- F6 (Codex review 2026-07-11): single-frame 2-D detector dataset -------

def test_open_source_single_frame_2d_detector_nxs(tmp_path):
    """A .nxs whose only image dataset is a single 2-D detector frame (the
    Bluesky/NXWriter one-exposure convention: NXdata @signal is a scalar
    counter, pixels flagged @signal_type='detector') must open through the
    headless seam as a one-frame source — classify_image_source and read_image
    already accepted it (ndim >= 2), but open_source used to raise
    ValueError('... is 2-D; NexusImageStack expects 3-D')."""
    import h5py

    from xrd_tools.io import ImageSourceKind, classify_image_source

    H, W = 7, 9
    img = np.arange(H * W, dtype=np.uint32).reshape(H, W)
    p = tmp_path / "single_00001.nxs"
    with h5py.File(p, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "gate"          # scalar counter is the @signal
        data.create_dataset("gate", data=np.array([0.5]))
        det = data.create_dataset("eiger_image", data=img)
        det.attrs["signal_type"] = "detector"

    # The display classifier accepts this file as a one-frame RAW_MASTER…
    info = classify_image_source(p)
    assert info.kind is ImageSourceKind.RAW_MASTER
    assert info.frame_labels == (0,)

    # …and the headless FrameSource contract must agree with it.
    src = open_source(SourceSpec(p, SourceKind.NEXUS_STACK, entry="entry"))
    assert list(src.frame_indices) == [0]
    frame = src.load_frame(0)
    assert frame.shape == (H, W)
    assert np.array_equal(frame, img)
    (block, labels), = list(src.iter_chunks(8))
    assert block.shape == (1, H, W)
    assert list(labels) == [0]
    assert np.array_equal(block[0], img)
