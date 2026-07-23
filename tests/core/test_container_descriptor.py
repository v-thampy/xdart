# -*- coding: utf-8 -*-
"""R2 Commit 1 — ContainerDescriptor built from real temporary HDF5 files.

Covers 2-D, 3-D chunked, 3-D contiguous, imageless, processed-shaped, and
external-link Eiger containers, plus the Bluesky finalized/unfinalized state,
and PROVES no detector pixel is decoded while building a descriptor.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources.descriptor import (
    ContainerDescriptor,
    describe_container,
    describe_container_from_open,
)
from xrd_tools.sources.probe import ProbeState


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _chunked_stack(path, *, shape=(8, 16, 16), chunks=(2, 16, 16),
                   dtype=np.uint16, wavelength=None):
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        det = entry.create_group("instrument/detector")
        det.create_dataset("data", data=np.zeros(shape, dtype=dtype), chunks=chunks)
        if wavelength is not None:
            mono = entry.create_group("instrument/monochromator")
            mono.create_dataset("wavelength", data=np.array(wavelength))
    return path


def _contiguous_stack(path, *, shape=(6, 10, 10), dtype=np.uint32):
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        det = entry.create_group("instrument/detector")
        det.create_dataset("data", data=np.zeros(shape, dtype=dtype))  # no chunks
    return path


def _lone_2d(path, *, shape=(24, 20), dtype=np.uint16):
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        det = entry.create_group("instrument/detector")
        det.create_dataset("data", data=np.zeros(shape, dtype=dtype))
    return path


def _imageless(path):
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("instrument")  # no detector dataset anywhere
    return path


def _processed(path):
    from xrd_tools.core import IntegrationResult1D
    from xrd_tools.io.nexus import write_integrated_stack

    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        r1d = IntegrationResult1D(
            radial=np.linspace(0.5, 2.5, 5), intensity=np.linspace(10.0, 50.0, 5),
            sigma=np.ones(5), unit="q_A^-1")
        write_integrated_stack(entry, frame_indices=[0, 1], results_1d=[r1d, r1d])
    return path


def _eiger_master(master_path, data_paths, *, per_seg=(5, 8, 8), dtype=np.uint16,
                  chunks=(2, 8, 8)):
    """One external-link Eiger master pointing at one or more data files."""
    for dp in data_paths:
        with h5py.File(dp, "w") as d:
            g = d.create_group("entry/data")
            g.create_dataset("data", data=np.zeros(per_seg, dtype=dtype), chunks=chunks)
    with h5py.File(master_path, "w") as m:
        entry = m.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        m.create_group("entry/data")
        for i, dp in enumerate(data_paths, start=1):
            m[f"/entry/data/data_{i:06d}"] = h5py.ExternalLink(str(dp), "/entry/data/data")
    return master_path


def _bluesky(path, *, finalized=True, shape=(4, 8, 8), dtype=np.uint16):
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("instrument/bluesky")  # the Bluesky marker
        det = entry.create_group("instrument/detector")
        det.create_dataset("data", data=np.zeros(shape, dtype=dtype), chunks=(2, 8, 8))
        if finalized:
            entry.create_dataset("end_time", data=np.bytes_("2026-07-15T00:00:00"))
    return path


# --------------------------------------------------------------------------- #
# READY / layout facts
# --------------------------------------------------------------------------- #
def test_chunked_stack_descriptor(tmp_path):
    p = _chunked_stack(tmp_path / "chunked.nxs", wavelength=1.2345)
    d = describe_container(p)
    assert d.state is ProbeState.READY
    assert d.kind is SourceKind.NEXUS_STACK
    assert d.dataset_path == "/entry/instrument/detector/data"
    assert d.frame_count == 8
    assert d.frame_shape == (16, 16)
    assert d.dataset_shape == (8, 16, 16)
    assert d.dtype == np.dtype(np.uint16)
    assert d.chunks == (2, 16, 16)
    assert d.is_2d is False
    assert d.self_contained is True
    assert d.scan_name == "chunked"
    assert d.finalized is True
    assert d.frame_bytes == 16 * 16 * 2
    assert d.wavelength == pytest.approx(1.2345)


def test_contiguous_stack_has_no_chunks(tmp_path):
    p = _contiguous_stack(tmp_path / "contig.nxs")
    d = describe_container(p)
    assert d.state is ProbeState.READY
    assert d.chunks is None
    assert d.frame_count == 6
    assert d.dtype == np.dtype(np.uint32)


def test_lone_2d_is_one_frame(tmp_path):
    p = _lone_2d(tmp_path / "single.nxs")
    d = describe_container(p)
    assert d.state is ProbeState.READY
    assert d.is_2d is True
    assert d.frame_count == 1
    assert d.frame_shape == (24, 20)
    assert d.dataset_shape == (24, 20)


def test_imageless_container(tmp_path):
    p = _imageless(tmp_path / "empty.nxs")
    d = describe_container(p)
    assert d.state is ProbeState.IMAGELESS
    assert d.dataset_path is None
    assert d.frame_count == 0


def test_processed_never_resolves_a_detector_dataset(tmp_path):
    p = _processed(tmp_path / "processed.nxs")
    d = describe_container(p)
    assert d.state is ProbeState.PROCESSED_OUTPUT
    assert d.kind is SourceKind.PROCESSED_NEXUS
    assert d.dataset_path is None
    assert d.frame_count == 0


# --------------------------------------------------------------------------- #
# external-link Eiger master
# --------------------------------------------------------------------------- #
def test_single_link_eiger_master(tmp_path):
    p = _eiger_master(
        tmp_path / "burst_master.h5", [tmp_path / "burst_data_000001.h5"])
    d = describe_container(p)
    assert d.state is ProbeState.READY
    assert d.kind is SourceKind.EIGER_MASTER
    assert d.scan_name == "burst"  # _master stripped
    assert d.frame_count == 5
    assert d.frame_shape == (8, 8)
    assert d.dataset_path == "/entry/data/data_000001"
    assert d.segment_paths == ()  # single segment
    assert d.chunks == (2, 8, 8)
    assert d.self_contained is False


def test_multi_segment_eiger_concatenates_frames(tmp_path):
    p = _eiger_master(
        tmp_path / "multi_master.h5",
        [tmp_path / "multi_data_000001.h5", tmp_path / "multi_data_000002.h5"])
    d = describe_container(p)
    assert d.frame_count == 10  # 5 + 5
    assert d.dataset_shape == (10, 8, 8)
    assert d.segment_paths == (
        "/entry/data/data_000001", "/entry/data/data_000002")


def _missing_link_master(path, target_name="late_data_000001.h5"):
    """An Eiger master published before its external data file arrives."""
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data["data_000001"] = h5py.ExternalLink(target_name, "/entry/data/data")
    return path


def _late_eiger_data(path):
    with h5py.File(path, "w") as h5:
        data = h5.create_group("entry/data")
        data.create_dataset("data", data=np.ones((3, 8, 8), np.uint16),
                            chunks=(1, 8, 8))


def _missing_detector_link_master(path, target_name="late_detector.h5"):
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        detector = entry.create_group("instrument/detector")
        detector["data"] = h5py.ExternalLink(target_name, "/data")
    return path


def _late_detector_data(path):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=np.ones((2, 8, 8), np.uint16),
                          chunks=(1, 8, 8))


def test_missing_eiger_external_link_is_provisional_then_ready(tmp_path):
    """A normal data-file arrival race is retryable, not a descriptor crash."""
    from xrd_tools.sources.cursor import ContainerCursor, ContainerNotReadyError

    master = _missing_link_master(tmp_path / "late_master.h5")
    provisional = describe_container(master)
    assert provisional.state is ProbeState.IN_PROGRESS
    assert provisional.kind is SourceKind.EIGER_MASTER
    assert provisional.finalized is False
    assert "retry later" in provisional.reason

    cursor = ContainerCursor(master)
    with pytest.raises(ContainerNotReadyError, match="not ready"):
        cursor.open()
    assert cursor.closed is True

    _late_eiger_data(tmp_path / "late_data_000001.h5")
    ready = describe_container(master)
    assert ready.state is ProbeState.READY
    assert ready.frame_count == 3
    with ContainerCursor(master) as cursor:
        np.testing.assert_array_equal(cursor.read_frame(0), np.ones((8, 8), np.uint16))


def test_missing_detector_group_external_link_is_provisional_then_ready(tmp_path):
    """The same descriptor boundary covers canonical detector-group links."""
    from xrd_tools.sources.cursor import ContainerCursor, ContainerNotReadyError

    master = _missing_detector_link_master(tmp_path / "detector_master.h5")
    assert describe_container(master).state is ProbeState.IN_PROGRESS
    with pytest.raises(ContainerNotReadyError):
        ContainerCursor(master).open()

    _late_detector_data(tmp_path / "late_detector.h5")
    assert describe_container(master).state is ProbeState.READY
    with ContainerCursor(master) as cursor:
        assert cursor.frame_count == 2


def test_missing_link_candidate_stamp_change_stays_fail_closed(tmp_path):
    """The descriptor correction does not weaken R1 candidate identity checks."""
    from xrd_tools.sources.cursor import ContainerCursor
    from xrd_tools.sources.directory_index import StaleCandidateError
    from xrd_tools.sources.discover import Candidate

    master = _missing_link_master(tmp_path / "stamp_master.h5")
    stat = master.stat()
    candidate = Candidate(master, "nexus_hdf5", stat.st_size, stat.st_mtime_ns)
    master.touch()
    with pytest.raises(StaleCandidateError):
        ContainerCursor(master, candidate=candidate).open()


def test_directory_index_promotes_late_external_link_on_reprobe(tmp_path):
    """The R1 provisional window accepts the same master's READY re-probe."""
    from xrd_tools.sources.directory_index import DirectoryIndex
    from xrd_tools.sources.probe import ProbeResult

    master = _missing_link_master(tmp_path / "index_master.h5")
    index = DirectoryIndex(tmp_path, retry_deadline=30.0)
    candidate = next(c for c in index.poll().candidates if c.path == master)

    first = describe_container(master, candidate=candidate)
    held = index.record_probe(
        candidate, ProbeResult(first.state, reason=first.reason, kind=first.kind))
    assert held.state is ProbeState.IN_PROGRESS
    assert index.retry_state(master) is not None

    _late_eiger_data(tmp_path / "late_data_000001.h5")
    second = describe_container(master, candidate=candidate)
    settled = index.record_probe(
        candidate, ProbeResult(second.state, reason=second.reason, kind=second.kind))
    assert settled.state is ProbeState.READY
    assert index.retry_state(master) is None


# --------------------------------------------------------------------------- #
# Bluesky finalized / unfinalized
# --------------------------------------------------------------------------- #
def test_bluesky_finalized_is_ready(tmp_path):
    p = _bluesky(tmp_path / "bl_final.nxs", finalized=True)
    d = describe_container(p)
    assert d.is_bluesky is True
    assert d.finalized is True
    assert d.state is ProbeState.READY


def test_bluesky_unfinalized_is_in_progress(tmp_path):
    p = _bluesky(tmp_path / "bl_open.nxs", finalized=False)
    d = describe_container(p)
    assert d.is_bluesky is True
    assert d.finalized is False
    assert d.state is ProbeState.IN_PROGRESS
    # layout facts are still reported even while provisional
    assert d.frame_count == 4


# --------------------------------------------------------------------------- #
# no detector decode
# --------------------------------------------------------------------------- #
def test_descriptor_construction_decodes_no_detector_pixels(tmp_path, monkeypatch):
    p = _chunked_stack(tmp_path / "nodecode.nxs", wavelength=1.0)
    sliced: list[str] = []
    orig = h5py.Dataset.__getitem__

    def spy(self, key):
        sliced.append(self.name)
        return orig(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    d = describe_container(p)
    assert d.frame_count == 8
    # the detector dataset itself was never sliced (no pixels decoded)
    assert "/entry/instrument/detector/data" not in sliced


# --------------------------------------------------------------------------- #
# path wrapper delegates to the handle-aware builder
# --------------------------------------------------------------------------- #
def test_path_wrapper_matches_handle_aware_builder(tmp_path):
    p = _chunked_stack(tmp_path / "delegate.nxs", wavelength=0.98)
    from_path = describe_container(p)
    st = p.stat()
    # feed the wrapper's own identity stat so delegation is byte-for-byte equal
    with h5py.File(p, "r") as h5:
        from_open = describe_container_from_open(
            h5, path=p, size=st.st_size, mtime_ns=st.st_mtime_ns)
    assert from_path == from_open


def test_candidate_identity_is_recorded(tmp_path):
    from xrd_tools.sources.discover import Candidate

    p = _chunked_stack(tmp_path / "cand.nxs")
    st = p.stat()
    cand = Candidate(p, "nexus_hdf5", st.st_size, st.st_mtime_ns)
    d = describe_container(p, candidate=cand)
    assert d.adapter_id == "nexus_hdf5"
    assert d.size == st.st_size
    assert d.mtime_ns == st.st_mtime_ns
    assert d.version_stamp == (st.st_size, st.st_mtime_ns)


def test_unreadable_file_is_in_progress(tmp_path):
    p = tmp_path / "half_written.nxs"
    p.write_bytes(b"\x89HDF\r\n\x1a\n garbage not-a-real-hdf5")
    d = describe_container(p)
    assert d.state is ProbeState.IN_PROGRESS
    assert d.finalized is False


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        describe_container(tmp_path / "does_not_exist.nxs")


def test_descriptor_holds_no_open_handle_and_serializes(tmp_path):
    p = _chunked_stack(tmp_path / "serial.nxs", wavelength=1.1)
    d = describe_container(p)
    # frozen value: no live h5py object leaked
    assert not isinstance(d.dtype, h5py.Dataset)
    payload = d.to_dict()
    assert payload["dtype"] == "uint16"
    assert payload["chunks"] == [2, 16, 16]
    assert payload["kind"] == "nexus_stack"
    assert payload["state"] == "ready"
    # fully JSON round-trippable
    assert json.loads(json.dumps(payload))["frame_count"] == 8


def test_descriptor_is_frozen(tmp_path):
    p = _lone_2d(tmp_path / "frozen.nxs")
    d = describe_container(p)
    with pytest.raises(Exception):
        d.frame_count = 999  # type: ignore[misc]
    assert isinstance(d, ContainerDescriptor)


# --------------------------------------------------------------------------- #
# audit fixes: nascent NXWriter shell + non-entry detector location
# --------------------------------------------------------------------------- #
def test_nascent_nxwriter_shell_is_in_progress(tmp_path):
    """A still-writing NXWriter run stamps root creator='NXWriter' before its
    entry tree exists; it must read IN_PROGRESS (agree with R1
    is_unfinalized_nxwriter), never finalized/IMAGELESS."""
    from xrd_tools.io.bluesky_nexus import is_unfinalized_nxwriter

    p = tmp_path / "nascent_00001.nxs"
    with h5py.File(p, "w") as f:
        f.attrs["creator"] = "NXWriter"  # creator stamped, NO entry group yet
    d = describe_container(p)
    assert d.state is ProbeState.IN_PROGRESS
    assert d.is_bluesky is True
    assert d.finalized is False
    assert is_unfinalized_nxwriter(p) is True  # R1 agrees


def test_detector_at_non_entry_location_is_found(tmp_path):
    """A detector dataset at a non-entry location (root /data) is still found
    via R1's whole-file resolver, so readiness agrees with R1 by construction."""
    p = tmp_path / "rootdata.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.zeros((3, 8, 8), np.uint16))  # /data, no entry
    d = describe_container(p)
    assert d.state is ProbeState.READY
    assert d.dataset_path == "/data"
    assert d.frame_count == 3


def test_apstools_flat_nxwriter_uses_direct_detector_contract(
        tmp_path, monkeypatch):
    """The qualified fast path avoids a recursive whole-file detector walk."""
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    from xrd_tools.sources import descriptor as descriptor_module

    path = _write_bluesky_nxwriter(tmp_path / "flat_00001.nxs", n=6)
    monkeypatch.setattr(
        descriptor_module,
        "resolve_stack_paths",
        lambda *_args, **_kwargs: pytest.fail(
            "qualified flat NXWriter fell back to recursive detector search"),
    )

    descriptor = describe_container(path)

    assert descriptor.state is ProbeState.READY
    assert descriptor.dataset_path == "/entry/data/eiger_image"
    assert descriptor.frame_count == 6
    assert descriptor.self_contained is True


def test_apstools_flat_detector_does_not_hide_canonical_rank_error(tmp_path):
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter

    path = _write_bluesky_nxwriter(tmp_path / "bad_rank_00001.nxs", n=2)
    with h5py.File(path, "r+") as handle:
        canonical = handle["entry/instrument"].create_group("detector")
        canonical.create_dataset(
            "data", data=np.zeros((1, 2, 3, 4), dtype=np.uint16))

    descriptor = describe_container(path)

    assert descriptor.state is ProbeState.INVALID
    assert "rank 4" in descriptor.reason
