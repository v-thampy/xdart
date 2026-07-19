# -*- coding: utf-8 -*-
"""Post-R2 source-edge contract: entry/link/rank/classifier agreement.

One resolution route (handoff §5): resolve the NXentry NXclass-aware →
classify processed output against THAT entry → resolve detector paths
against the same entry → validate supported rank — projected consistently
through the descriptor, the R1 adapter probe, cursor/public open, and the
legacy lower-level finder/read seams.

Closes (fail-before proven at 837397b9):
- NXS-ENTRY-1: /entry1 raw data probed READY but public open raised KeyError;
- NXS-PROC-1: /entry1 processed output re-ingested as raw (probe READY,
  read_image returned the integrated cake as detector pixels);
- NXS-LINK-1: dangling external links escaped the lower-level finder/read
  seams as untyped KeyError (the descriptor/cursor were already typed);
- NXS-DIM-1: a rank-4 detector signal probed/described READY with a bogus
  3-D frame shape although every reader rejects it.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.adapters import candidate_owner
from xrd_tools.sources.descriptor import describe_container
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.registry import open_source


def _probe(path):
    """The R1 adapter probe route (what DirectoryIndex.record_probe consumes)."""
    owner = candidate_owner(Path(path))
    assert owner is not None
    return owner.probe(Path(path))


# ── fixtures ──────────────────────────────────────────────────────────────

def _raw_entry1(tmp_path, *, rank=3, n=3):
    path = tmp_path / f"alt_entry_rank{rank}.nxs"
    with h5py.File(path, "w") as f:
        e = f.create_group("entry1")
        e.attrs["NX_class"] = "NXentry"
        det = e.create_group("instrument").create_group("detector")
        if rank == 2:
            det.create_dataset(
                "data", data=np.arange(64, dtype=np.uint16).reshape(8, 8))
        else:
            det.create_dataset(
                "data",
                data=np.arange(n * 64, dtype=np.uint16).reshape(n, 8, 8))
    return path


def _processed_entry1(tmp_path):
    path = tmp_path / "processed_entry1.nxs"
    with h5py.File(path, "w") as f:
        e = f.create_group("entry1")
        e.attrs["NX_class"] = "NXentry"
        g = e.create_group("integrated_2d")
        g.create_dataset(
            "intensity", data=np.ones((3, 16, 24), dtype=np.float32))
        g.create_dataset("frame_index", data=np.arange(3, dtype=np.int64))
        g1 = e.create_group("integrated_1d")
        g1.create_dataset("intensity", data=np.ones((3, 16), dtype=np.float32))
        g1.create_dataset("frame_index", data=np.arange(3, dtype=np.int64))
    return path


def _dangling_canonical(tmp_path):
    master = tmp_path / "scan_master.h5"
    target = tmp_path / "scan_data_000001.h5"
    with h5py.File(master, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        data = e.create_group("data")
        data.attrs["NX_class"] = "NXdata"        # as Dectris masters write it
        data["data_000001"] = h5py.ExternalLink(str(target), "/entry/data/data")
        # Every real Dectris FileWriter master carries 2-D auxiliary datasets
        # NEXT TO the not-yet-landed data link.  A finder that rummages for
        # "the largest 2-D+ dataset" would classify the mid-transfer master
        # READY with the flatfield as the image (review-caught).
        det = e.create_group("instrument").create_group("detector")
        det.attrs["NX_class"] = "NXdetector"
        spec = det.create_group("detectorSpecific")
        spec.create_dataset("pixel_mask", data=np.zeros((32, 32), np.uint32))
        spec.create_dataset("flatfield", data=np.ones((32, 32), np.float32))
    return master, target


def _land_canonical(target):
    with h5py.File(target, "w") as f:
        f.create_dataset(
            "entry/data/data",
            data=np.arange(3 * 64, dtype=np.uint16).reshape(3, 8, 8))


def _dangling_detector_group(tmp_path):
    master = tmp_path / "det_master.h5"
    target = tmp_path / "det_data.h5"
    with h5py.File(master, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        det = e.create_group("instrument").create_group("detector")
        det["data"] = h5py.ExternalLink(str(target), "/data")
    return master, target


def _land_detector_group(target):
    with h5py.File(target, "w") as f:
        f.create_dataset(
            "data", data=np.arange(2 * 64, dtype=np.uint16).reshape(2, 8, 8))


def _rank4(tmp_path):
    path = tmp_path / "rank4.nxs"
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        det = e.create_group("instrument").create_group("detector")
        det.create_dataset(
            "data",
            data=np.arange(2 * 3 * 64, dtype=np.uint16).reshape(2, 3, 8, 8))
    return path


# ── NXS-ENTRY-1: alternate raw NXentry ────────────────────────────────────

@pytest.mark.parametrize("rank,expected_frames", [(3, 3), (2, 1)])
def test_alternate_entry_raw_ready_and_opens(tmp_path, rank, expected_frames):
    path = _raw_entry1(tmp_path, rank=rank)

    desc = describe_container(str(path))
    assert desc.state is ProbeState.READY
    assert desc.resolved_entry == "entry1"
    assert desc.frame_count == expected_frames

    result = _probe(path)
    assert result.state is ProbeState.READY

    source = open_source(SourceSpec(str(path), SourceKind.NEXUS_STACK))
    try:
        indices = list(source.frame_indices)
        assert len(indices) == expected_frames
        frame = np.asarray(source.load_frame(indices[0]))
        assert frame.shape == (8, 8)
        assert frame.dtype == np.uint16          # native dtype preserved
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()


def test_alternate_entry_stack_source_default_hint(tmp_path):
    from xrd_tools.sources.nexus import NexusStackSource

    path = _raw_entry1(tmp_path, rank=3)
    source = NexusStackSource(path)              # default entry hint
    assert len(list(source.frame_indices)) == 3


# ── NXS-PROC-1: alternate processed NXentry never re-ingested ─────────────

def test_alternate_entry_processed_never_raw(tmp_path):
    from xrd_tools.io.image import read_image
    from xrd_tools.io.image_source import classify_image_source
    from xrd_tools.io.processed_scan_id import (
        ProcessedXdartInputError,
        is_processed_xdart_file,
        is_processed_xdart_path,
    )

    path = _processed_entry1(tmp_path)

    with h5py.File(path, "r") as f:
        assert is_processed_xdart_file(f) is True, \
            "the open-file classifier must resolve the NXentry (NXS-PROC-1)"
    assert is_processed_xdart_path(path) is True

    desc = describe_container(str(path))
    assert desc.state is ProbeState.PROCESSED_OUTPUT

    result = _probe(path)
    assert result.state is ProbeState.PROCESSED_OUTPUT, \
        "the R1 probe must never call a processed record READY"

    info = classify_image_source(str(path))
    assert getattr(info, "has_raw", True) is False or \
        "processed" in str(getattr(info, "kind", "")).lower()

    with pytest.raises(ProcessedXdartInputError):
        read_image(str(path))                    # never the cake as pixels

    with pytest.raises(ProcessedXdartInputError):
        open_source(SourceSpec(str(path), SourceKind.NEXUS_STACK))


# ── NXS-LINK-1: dangling links are typed provisional, then READY ─────────

@pytest.mark.parametrize("fixture,land", [
    (_dangling_canonical, _land_canonical),
    (_dangling_detector_group, _land_detector_group),
])
def test_dangling_link_typed_provisional_then_ready(tmp_path, fixture, land):
    from xrd_tools.io.image import read_image

    master, target = fixture(tmp_path)

    result = _probe(master)                      # fail-before: bare KeyError
    assert result.state is ProbeState.IN_PROGRESS

    desc = describe_container(str(master))
    assert desc.state is ProbeState.IN_PROGRESS

    # the lower-level finder seam itself (what the R1 probe consumes) must
    # raise the TYPED provisional error — not a bare KeyError, and not a
    # READY-looking auxiliary dataset (pixel_mask/flatfield)
    from xrd_tools.io.image import _find_hdf5_image_dataset
    from xrd_tools.io.nexus import UnresolvedSourceLinkError
    with h5py.File(master, "r") as f:
        with pytest.raises(UnresolvedSourceLinkError):
            _find_hdf5_image_dataset(f)
    # public read route smoke: whatever reader claims the file, the failure
    # is never a bare KeyError/AttributeError
    with pytest.raises(Exception) as exc_info:
        read_image(str(master))
    assert not type(exc_info.value) in (KeyError, AttributeError), \
        "raw KeyError/AttributeError must not be the provisional contract"

    stat_before = master.stat()
    land(target)                                 # target lands, master untouched
    assert master.stat().st_mtime_ns == stat_before.st_mtime_ns

    result = _probe(master)
    assert result.state is ProbeState.READY, \
        "reprobe must succeed without a master byte change"
    desc = describe_container(str(master))
    assert desc.state is ProbeState.READY
    source = open_source(SourceSpec(str(master), SourceKind.NEXUS_STACK))
    try:
        assert len(list(source.frame_indices)) >= 2
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()


# ── NXS-DIM-1: rank 4 is INVALID everywhere ───────────────────────────────

def test_rank4_detector_invalid_everywhere(tmp_path):
    from xrd_tools.sources.cursor import ContainerCursor

    path = _rank4(tmp_path)

    desc = describe_container(str(path))
    assert desc.state is ProbeState.INVALID, \
        "a rank-4 detector signal must never describe READY (NXS-DIM-1)"
    assert "rank" in (desc.reason or "").lower()

    result = _probe(path)
    assert result.state is ProbeState.INVALID

    with pytest.raises(ValueError):
        ContainerCursor(str(path)).open()

    with pytest.raises(ValueError):
        # strictly the typed contract: a bare KeyError regression must FAIL
        source = open_source(SourceSpec(str(path), SourceKind.NEXUS_STACK))
        list(source.frame_indices)


def test_rank2_and_rank3_still_ready(tmp_path):
    """No-regression guard: supported ranks keep their exact behavior."""
    for rank, frames in ((2, 1), (3, 3)):
        path = tmp_path / f"ok_rank{rank}.nxs"
        with h5py.File(path, "w") as f:
            e = f.create_group("entry")
            e.attrs["NX_class"] = "NXentry"
            det = e.create_group("instrument").create_group("detector")
            shape = (8, 8) if rank == 2 else (3, 8, 8)
            det.create_dataset(
                "data",
                data=np.arange(int(np.prod(shape)),
                               dtype=np.uint16).reshape(shape))
        desc = describe_container(str(path))
        assert desc.state is ProbeState.READY
        assert desc.frame_count == frames
