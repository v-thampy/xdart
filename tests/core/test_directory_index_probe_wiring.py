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


def test_imageless_stays_provisional_then_terminal_via_real_shell_fixture(tmp_path):
    """The exact "newly readable shell... not permanently retired as
    imageless" policy, end-to-end through probe_candidate."""
    _nxs_imageless(tmp_path / "shell.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()
    candidate = _candidate_for(tmp_path, "shell.nxs")

    first = index.probe_candidate(candidate)
    assert first.state is ProbeState.IN_PROGRESS   # NOT trusted as imageless yet

    clock.advance(11.0)   # past the deadline; file never changed
    final = index.probe_candidate(candidate)
    assert final.state is ProbeState.IMAGELESS      # now trusted as terminal


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


def test_probe_candidate_raises_for_an_unknown_adapter_id(tmp_path):
    from xrd_tools.sources.discover import Candidate

    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    ghost = Candidate(tmp_path / "ghost.nxs", "no_such_adapter", 0, 0)
    try:
        index.probe_candidate(ghost)
    except LookupError:
        pass
    else:
        raise AssertionError("expected LookupError for an unregistered adapter id")
