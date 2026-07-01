# -*- coding: utf-8 -*-
"""D7 — StrictPolicy: the headless reduction + reader seams are loud by default.

A scripted/batch reduction RAISES on a per-frame degradation (missing
normalization, an all-dummy 2D integration) instead of silently writing bad
data; the xdart GUI opts into ``StrictPolicy.graceful()`` (never abort a save).
Synthetic fixtures only — no Qt.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import xrd_tools.reduction.core as reduction_core
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.core.strictness import (
    GIAllDummyError,
    MissingNormalizationError,
    StrictnessError,
    StrictPolicy,
)
from xrd_tools.reduction import (
    Frame,
    Integration2DPlan,
    MemorySink,
    ReductionPlan,
    ReductionSession,
    Scan,
)


def _r1d(value, n=4):
    return IntegrationResult1D(radial=np.linspace(0.0, 1.0, n),
                               intensity=np.full(n, float(value)),
                               sigma=None, unit="q_A^-1")


def _frames(n):
    return [Frame(i, image=np.full((2, 2), i + 1.0)) for i in range(n)]


# ── StrictPolicy + error hierarchy ───────────────────────────────────────────

def test_strict_policy_loud_graceful_default():
    assert StrictPolicy() == StrictPolicy.loud()            # bare instance = loud
    loud = StrictPolicy.loud()
    assert (loud.missing_normalization and loud.gi_all_dummy
            and loud.thumbnail_fallback)
    g = StrictPolicy.graceful()
    assert not (g.missing_normalization or g.gi_all_dummy or g.thumbnail_fallback)


def test_strictness_error_hierarchy():
    # ``except ValueError`` still catches them (the GIFreezeError precedent);
    # ``except StrictnessError`` catches the whole family.
    assert issubclass(MissingNormalizationError, StrictnessError)
    assert issubclass(GIAllDummyError, StrictnessError)
    assert issubclass(StrictnessError, ValueError)


# ── missing normalization ────────────────────────────────────────────────────

def test_strict_missing_normalization_loud_raises_graceful_warns():
    plan = SimpleNamespace(monitor_key="mon_missing")
    frame = Frame(0)                                   # no normalization_factor
    with pytest.raises(MissingNormalizationError, match="mon_missing"):
        reduction_core._normalization_for(frame, plan, strict=StrictPolicy.loud())
    with pytest.raises(ValueError):                    # hierarchy
        reduction_core._normalization_for(frame, plan, strict=StrictPolicy.loud())
    with pytest.warns(RuntimeWarning, match="UN-normalized"):
        assert reduction_core._normalization_for(
            frame, plan, strict=StrictPolicy.graceful()) is None


def test_strict_missing_normalization_default_session_is_loud(monkeypatch):
    """A ReductionSession with no explicit policy is LOUD: a dead monitor makes
    the run fail (finish re-raises) rather than persist un-normalized data."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    plan = ReductionPlan(integration_2d=None)
    plan.integration_1d.monitor_key = "mon_dead"
    session = ReductionSession(
        plan, Scan("loud", _frames(2), integrator=object()),
        sink=MemorySink(), execution="streaming")
    for fr in session.scan.frames:
        session.submit(fr)
    with pytest.raises(MissingNormalizationError):
        session.finish()


def test_strict_missing_normalization_graceful_session_warns(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    plan = ReductionPlan(integration_2d=None)
    plan.integration_1d.monitor_key = "mon_dead_g"
    session = ReductionSession(
        plan, Scan("graceful", _frames(2), integrator=object()),
        sink=MemorySink(), execution="streaming",
        strict=StrictPolicy.graceful())
    # the warn-once fires during the reduction (submit), so wrap both.
    with pytest.warns(RuntimeWarning, match="UN-normalized"):
        for fr in session.scan.frames:
            session.submit(fr)
        result = session.finish()                      # completes, un-normalized
    assert result is not None


# ── all-dummy 2D ─────────────────────────────────────────────────────────────

def _dummy_2d():
    return IntegrationResult2D(
        radial=np.linspace(0.0, 1.0, 3), azimuthal=np.linspace(-90, 90, 4),
        intensity=np.full((3, 4), -1.0), sigma=None, unit="q_A^-1")


def test_strict_gi_all_dummy_loud_raises(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    monkeypatch.setattr(reduction_core, "integrate_2d",
                        lambda image, ai, **kw: _dummy_2d())
    plan = ReductionPlan(integration_2d=Integration2DPlan())                             # 1d + 2d
    session = ReductionSession(
        plan, Scan("loud2d", _frames(1), integrator=object()),
        sink=MemorySink(), execution="streaming")
    session.submit(session.scan.frames[0])
    with pytest.raises(GIAllDummyError):
        session.finish()


def test_strict_gi_all_dummy_graceful_keeps(monkeypatch):
    """Graceful returns the all-dummy result (dropped per-frame downstream by
    the publication gate / writer) — never aborting the whole-scan save."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    monkeypatch.setattr(reduction_core, "integrate_2d",
                        lambda image, ai, **kw: _dummy_2d())
    plan = ReductionPlan(integration_2d=Integration2DPlan())
    session = ReductionSession(
        plan, Scan("ok2d", _frames(1), integrator=object()),
        sink=MemorySink(), execution="streaming",
        strict=StrictPolicy.graceful())
    session.submit(session.scan.frames[0])
    result = session.finish()                          # no raise
    assert result is not None


# ── chunked path symmetry (B-2) ──────────────────────────────────────────────

def test_strict_chunked_skips_bad_frame_writes_good_and_raises_at_finish(monkeypatch):
    """B-2: a loud per-frame degradation in the CHUNKED path is recorded + SKIPPED
    (the good frames are still written) with the raise DEFERRED to finish() —
    mirroring the streaming submit path.  Pre-fix, the chunked path caught only
    _ReductionCancelled, so a StrictnessError aborted the whole chunk and wrote
    ZERO good frames, violating 'reject per frame, never abort a whole-scan save'."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    real_reduce = reduction_core._reduce_frame

    def _reduce_with_one_dead(frame, *a, **k):
        if int(frame.index) == 1:                      # frame 1: simulate a loud raise
            raise MissingNormalizationError("dead monitor (frame 1)")
        return real_reduce(frame, *a, **k)

    monkeypatch.setattr(reduction_core, "_reduce_frame", _reduce_with_one_dead)

    plan = ReductionPlan(integration_2d=None)
    sink = MemorySink()
    session = ReductionSession(
        plan, Scan("chunked_mix", _frames(3), integrator=object()),
        sink=sink, execution="chunked")                # serial chunked (no executor)
    session.process()                                  # must NOT abort the chunk
    # the good frames are written; the bad one is skipped — NOT a whole-chunk abort:
    assert set(sink.frames) == {0, 2}
    # ...and it is still fail-loud, deferred to finish():
    with pytest.raises(MissingNormalizationError):
        session.finish()


def test_strict_chunked_executor_honors_loud_policy(monkeypatch):
    """The executor-backed chunked path must pass the session StrictPolicy.

    Without the explicit ``strict=`` keyword, worker frames ran graceful and
    silently wrote un-normalized reductions when a monitor was missing.
    """
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))

    frames = _frames(3)
    frames[0].normalization_factor = 1.0
    frames[2].normalization_factor = 1.0

    plan = ReductionPlan(integration_2d=None)
    plan.integration_1d.monitor_key = "dead_monitor"
    sink = MemorySink()
    session = ReductionSession(
        plan,
        Scan("chunked_executor_mix", frames, integrator=object()),
        sink=sink,
        execution="chunked",
        executor=2,
    )
    session.process()

    assert set(sink.frames) == {0, 2}
    with pytest.raises(MissingNormalizationError):
        session.finish()
