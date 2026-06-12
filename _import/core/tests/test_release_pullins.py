"""Release pull-ins (Jun 2026, CC_postreview_plan): C1 reader-side
schema-version check, S2 product-retention knob, S6 SWMR refusal,
S8 monitor-normalization warning.
"""
from __future__ import annotations

import warnings as _warnings
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import IntegrationResult1D
from ssrl_xrd_tools.io.frame_view import FrameViewReader
from ssrl_xrd_tools.io.nexus import (
    PROCESSED_SCHEMA_VERSION,
    open_nexus_writer,
    read_scan,
    read_scan_metadata,
)
from ssrl_xrd_tools.reduction import (
    Frame,
    MemorySink,
    ReductionPlan,
    ReductionSession,
    Scan,
)
import ssrl_xrd_tools.reduction.core as reduction_core


# ── C1: reader-side schema-version check ────────────────────────────────────

def _entry_file(tmp_path, version):
    p = tmp_path / f"v_{version}.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        if version is not None:
            e.attrs["ssrl_schema_version"] = version
    return p


def test_newer_schema_warns_in_all_readers(tmp_path):
    p = _entry_file(tmp_path, PROCESSED_SCHEMA_VERSION + 1)
    with pytest.warns(RuntimeWarning, match="newer"):
        read_scan_metadata(p)
    with pytest.warns(RuntimeWarning, match="newer"):
        read_scan(p)
    with pytest.warns(RuntimeWarning, match="newer"):
        with FrameViewReader(p):
            pass


@pytest.mark.parametrize("version", [None, PROCESSED_SCHEMA_VERSION, 1])
def test_current_or_older_schema_is_silent(tmp_path, version):
    p = _entry_file(tmp_path, version)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        read_scan_metadata(p)
        with FrameViewReader(p):
            pass


# ── S6: SWMR-write refusal ──────────────────────────────────────────────────

def test_open_nexus_writer_swmr_refused(tmp_path):
    # The flag was advertised but guaranteed a failure on the first frame
    # append (HDF5 forbids object creation in SWMR-write mode); refuse loudly.
    with pytest.raises(NotImplementedError, match="swmr|SWMR"):
        open_nexus_writer(tmp_path / "s.nxs", swmr=True)
    assert not (tmp_path / "s.nxs").exists()


# ── S8: monitor-normalization warning ───────────────────────────────────────

def test_missing_monitor_warns_once_per_key():
    plan = SimpleNamespace(monitor_key="mon_s8_missing")
    with pytest.warns(RuntimeWarning, match="UN-normalized"):
        assert reduction_core._normalization_for(Frame(0), plan) is None
    # Once per key per process — a dead monitor on a 10k-frame scan must not
    # emit 10k warnings.
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        assert reduction_core._normalization_for(Frame(1), plan) is None


def test_zero_monitor_warns_and_valid_monitor_does_not():
    plan = SimpleNamespace(monitor_key="mon_s8_zero")
    with pytest.warns(RuntimeWarning, match="UN-normalized"):
        frame = Frame(0, metadata={"mon_s8_zero": 0.0})
        assert reduction_core._normalization_for(frame, plan) is None
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        frame = Frame(1, metadata={"mon_s8_zero": 2.5})
        assert reduction_core._normalization_for(frame, plan) == 2.5


# ── S2: product-retention knob ──────────────────────────────────────────────

def _r1d(value: float) -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=None,
        unit="q_A^-1",
    )


def _frames(n: int) -> list[Frame]:
    return [Frame(i, image=np.full((2, 2), i, dtype=float)) for i in range(n)]


def test_streaming_retain_products_off(monkeypatch):
    """retain_products=False: nothing accumulates in result.frames (the sink
    owns the data), while progress counting and replace/re-feed detection
    (A1 idempotency, via _seen_idxs) stay correct."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    sink = MemorySink()
    frames = _frames(4)
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("s", frames, integrator=object()),
        sink=sink, execution="streaming", executor=2,
        retain_products=False,
    )
    for fr in frames:
        session.submit(fr)
    assert session.drain(timeout=10)
    session.submit(frames[0])              # re-feed: a REPLACE, not a new frame

    result = session.finish()

    assert result.frames == {}             # retention off
    assert result.n_processed == 4         # replace not double-counted
    assert sorted(sink.frames) == [0, 1, 2, 3]


def test_streaming_retain_products_default_unchanged(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    frames = _frames(3)
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("s", frames, integrator=object()),
        sink=MemorySink(), execution="streaming", executor=1,
    )
    for fr in frames:
        session.submit(fr)
    result = session.finish()
    assert sorted(result.frames) == [0, 1, 2]   # historical contract intact


# ── P1a: run_reduction retention auto-default ───────────────────────────────

class _DuckDurableSink:
    """Minimal duck sink standing in for a durable (disk) sink."""

    def __init__(self):
        self.frames = []

    def begin(self, scan, plan):  # noqa: D401
        pass

    def write(self, frame, reduction):
        self.frames.append(int(frame.index))

    def finish(self, result):
        pass


def test_run_reduction_streaming_durable_sink_drops_retention(monkeypatch):
    """P1a: the public headless wrapper must not silently retain every
    FrameReduction when streaming into a durable sink — notebook users got
    the unbounded default the GUI path had already fixed."""
    from ssrl_xrd_tools.reduction import run_reduction

    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    sink = _DuckDurableSink()
    result = run_reduction(
        ReductionPlan(integration_2d=None),
        Scan("s", _frames(3), integrator=object()),
        sink, execution="streaming", executor=2,
    )
    assert result.frames == {}                # not retained
    assert result.n_processed == 3
    assert sorted(sink.frames) == [0, 1, 2]   # data went to the sink

    # Explicit override wins.
    result = run_reduction(
        ReductionPlan(integration_2d=None),
        Scan("s", _frames(2), integrator=object()),
        _DuckDurableSink(), execution="streaming", executor=1,
        retain_products=True,
    )
    assert sorted(result.frames) == [0, 1]


def test_run_reduction_memory_sink_and_chunked_keep_retention(monkeypatch):
    from ssrl_xrd_tools.reduction import run_reduction

    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    # MemorySink streaming: result.frames is the only product channel.
    result = run_reduction(
        ReductionPlan(integration_2d=None),
        Scan("s", _frames(2), integrator=object()),
        MemorySink(), execution="streaming", executor=1,
    )
    assert sorted(result.frames) == [0, 1]
    # Chunked into a durable sink: historical contract intact.
    result = run_reduction(
        ReductionPlan(integration_2d=None),
        Scan("s", _frames(2), integrator=object()),
        _DuckDurableSink(),
    )
    assert sorted(result.frames) == [0, 1]


def test_release_products_keeps_replace_semantics(monkeypatch):
    """S2 (serial flavor): release_products drops retained reductions but
    keeps _seen_idxs, so a released-then-re-fed index is still a REPLACE
    (n_processed never double-counts)."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    frames = _frames(2)
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("s", frames, integrator=object()),
        sink=MemorySink(),
    )
    session.process(frames)
    assert sorted(session.frames) == [0, 1]

    session.release_products([0, 1])
    assert session.frames == {}

    session.process([frames[0]])               # re-feed after release
    result = session.finish()
    assert result.n_processed == 2             # replace, not a 3rd completion
    assert sorted(session.frames) == [0]       # only the re-fed one retained
