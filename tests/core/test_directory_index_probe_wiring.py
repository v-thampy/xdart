# -*- coding: utf-8 -*-
"""R1 step 6 — connecting explicit probe results to indexed candidates.

Two things pinned here:

1. ``DirectoryIndex.probe_candidate`` ties a candidate's OWNING adapter probe
   to ``record_probe`` in one call — production-wired against the real
   nexus_hdf5/image_file adapters (no fakes on the seam being fixed).
2. READY / IN_PROGRESS / PROCESSED_OUTPUT / IMAGELESS / INVALID all stay
   distinct and reachable through the real adapters, using real (small)
   HDF5/image fixtures — matching the handoff's required test #9.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from xrd_tools.sources.directory_index import DirectoryIndex
from xrd_tools.sources.discover import enumerate_candidates
from xrd_tools.sources.probe import ProbeState


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _nxs_ready(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        inst = entry.create_group("instrument")
        det = inst.create_group("detector")
        det.create_dataset("data", data=np.zeros((2, 3, 4)))
    return path


def _nxs_imageless(path: Path) -> Path:
    """A readable shell with an entry group but no detector dataset and no
    writer markers -- the "newly readable HDF5 shell" case."""
    with h5py.File(path, "w") as f:
        f.create_group("entry")
    return path


def _nxs_processed_output(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        g1 = entry.create_group("integrated_1d")
        g1.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        g1.create_dataset("q", data=np.linspace(0.1, 1.0, 5))
        g1.create_dataset("intensity", data=np.arange(5, dtype=np.float32)[None, :])
    return path


def _nxs_unfinalized_bluesky(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        f.attrs["creator"] = "NXWriter"
        f.create_group("entry")   # no end_time -> not finalized
    return path


def _nxs_corrupt(path: Path) -> Path:
    path.write_bytes(b"not an hdf5 file at all")
    return path


def _image_corrupt(path: Path) -> Path:
    path.write_bytes(b"not a real tiff")
    return path


def _candidate_for(tmp_path: Path, name: str):
    (candidate,) = [c for c in enumerate_candidates(tmp_path) if c.path.name == name]
    return candidate


# ---- the five typed states, through the REAL adapters -----------------------


def test_ready_via_real_nexus_adapter(tmp_path):
    _nxs_ready(tmp_path / "ready.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()

    result = index.probe_candidate(_candidate_for(tmp_path, "ready.nxs"))
    assert result.state is ProbeState.READY


def test_processed_output_via_real_nexus_adapter(tmp_path):
    _nxs_processed_output(tmp_path / "processed.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()

    result = index.probe_candidate(_candidate_for(tmp_path, "processed.nxs"))
    assert result.state is ProbeState.PROCESSED_OUTPUT


def test_in_progress_via_real_unfinalized_bluesky_fixture(tmp_path):
    _nxs_unfinalized_bluesky(tmp_path / "writing.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock(), retry_deadline=30.0)
    index.poll()

    result = index.probe_candidate(_candidate_for(tmp_path, "writing.nxs"))
    assert result.state is ProbeState.IN_PROGRESS   # inside the bounded window
    assert result.descriptor is not None
    assert result.descriptor.finalized is False
    assert index.retry_state(tmp_path / "writing.nxs") is not None


def test_finalized_imageless_is_terminal_via_real_shell_fixture(tmp_path):
    """A readable finalized detectorless record is ignored immediately."""
    _nxs_imageless(tmp_path / "shell.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()
    candidate = _candidate_for(tmp_path, "shell.nxs")

    first = index.probe_candidate(candidate)
    assert first.state is ProbeState.IMAGELESS
    assert first.descriptor is not None
    assert first.descriptor.finalized is True
    assert index.retry_state(candidate.path) is None
    assert index.probe_candidate(candidate) is first


def test_probe_rejects_file_changed_after_descriptor_inspection(
    tmp_path, monkeypatch,
):
    """A post-open mutation cannot retain READY facts under the old stamp."""
    from dataclasses import replace

    import pytest

    from xrd_tools.sources import adapters as adapter_module
    from xrd_tools.sources.directory_index import StaleCandidateError

    path = _nxs_ready(tmp_path / "racing.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()
    candidate = _candidate_for(tmp_path, path.name)

    entry = adapter_module._ADAPTERS[candidate.adapter_id]
    original_probe = entry.adapter.probe

    def racing_probe(probe_path):
        result = original_probe(probe_path)
        with Path(probe_path).open("ab") as stream:
            stream.write(b"x")
        return result

    monkeypatch.setitem(
        adapter_module._ADAPTERS,
        candidate.adapter_id,
        replace(
            entry,
            adapter=replace(entry.adapter, probe=racing_probe),
        ),
    )

    with pytest.raises(StaleCandidateError, match="changed while"):
        index.probe_candidate(candidate)
    assert index.retry_state(path) is None


def test_invalid_via_real_corrupt_image_file(tmp_path):
    _image_corrupt(tmp_path / "broken.tif")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()

    result = index.probe_candidate(_candidate_for(tmp_path, "broken.tif"))
    assert result.state is ProbeState.INVALID


def test_invalid_via_nexus_retry_exhaustion_on_a_corrupt_file(tmp_path):
    """A genuinely corrupt (never-becomes-readable) NeXus file eventually
    surfaces as INVALID rather than retrying forever."""
    _nxs_corrupt(tmp_path / "corrupt.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=5.0)
    index.poll()
    candidate = _candidate_for(tmp_path, "corrupt.nxs")

    first = index.probe_candidate(candidate)
    assert first.state is ProbeState.IN_PROGRESS

    clock.advance(6.0)
    final = index.probe_candidate(candidate)
    assert final.state is ProbeState.INVALID
    assert "retry window" in final.reason


def test_all_five_states_are_pairwise_distinct():
    assert len({s for s in ProbeState}) == 5
    assert {s.value for s in ProbeState} == {
        "ready", "in_progress", "processed_output", "imageless", "invalid",
    }


# ---- probe_candidate is the only I/O-performing DirectoryIndex method -------


def test_probe_candidate_raises_for_an_unregistered_adapter_id(tmp_path):
    """A FRESH candidate whose owning adapter was unregistered between poll and
    probe raises LookupError.  (A stale candidate absent from the snapshot is a
    different rejection — ValueError — checked in test_directory_index_retry.py;
    the freshness check runs first, so this test keeps the candidate fresh and
    removes only its adapter.)"""
    import pytest

    from xrd_tools.sources import adapters as A

    _nxs_ready(tmp_path / "a.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()
    candidate = _candidate_for(tmp_path, "a.nxs")
    assert candidate.adapter_id == "nexus_hdf5"

    saved = dict(A._ADAPTERS)
    try:
        del A._ADAPTERS["nexus_hdf5"]   # adapter gone, candidate still fresh
        with pytest.raises(LookupError):
            index.probe_candidate(candidate)
    finally:
        A._ADAPTERS.clear()
        A._ADAPTERS.update(saved)
