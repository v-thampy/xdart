# -*- coding: utf-8 -*-
"""FlushPolicy — the unified headless save-cadence decision (Phase 4b-1).

The golden anchors encode the truth of the two GUI predicates this policy
replaces (xdart serial ``_save_due`` and streaming ``QtNexusSink._due_to_save``),
so a regression in the merge is caught against their real semantics, not a
restatement of the implementation.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from xrd_tools.reduction import FlushPolicy


# cap=64, margin=8 → hard_threshold (persist-before-evict bound) = 56
LIVE_2D = FlushPolicy(interval=8, cap=64, margin=8)      # 2D live / streaming
LIVE_1D = FlushPolicy(interval=1000, cap=64, margin=8)   # 1D skip_2d


def test_hard_threshold():
    assert LIVE_2D.hard_threshold() == 56
    assert FlushPolicy(cap=12, margin=8).hard_threshold() == 4
    assert FlushPolicy(cap=4, margin=8).hard_threshold() == 1   # clamped >= 1


@pytest.mark.parametrize("policy, frames_since, unsaved, force, expected", [
    # nothing pending short-circuits even under force
    (LIVE_2D, 0, None, True, False),
    (LIVE_2D, 0, 99, False, False),
    # force flushes whenever something is pending
    (LIVE_2D, 1, None, True, True),
    # interval branch (2D: every 8 frames)
    (LIVE_2D, 8, None, False, True),
    (LIVE_2D, 7, None, False, False),
    # pressure branch on the LOCAL counter (streaming, unsaved=None) —
    # but note 56 >= interval(8) so the interval branch already fires
    (LIVE_2D, 56, None, False, True),
    # pressure branch on the LIVE unsaved count (serial)
    (LIVE_2D, 1, 56, False, True),       # interval not hit, but unsaved hits hard bound
    (LIVE_2D, 1, 55, False, False),
    # 1D (interval 1000): no 8-frame saves; only pressure / interval
    (LIVE_1D, 8, None, False, False),
    (LIVE_1D, 56, None, False, True),    # local counter hits hard bound
    (LIVE_1D, 55, 56, False, True),      # live unsaved hits hard bound
    (LIVE_1D, 999, None, False, True),   # local counter 999 >= hard bound 56
    (LIVE_1D, 1000, None, False, True),
])
def test_should_flush_golden(policy, frames_since, unsaved, force, expected):
    assert policy.should_flush(frames_since_flush=frames_since,
                               unsaved_in_memory=unsaved, force=force) is expected


def test_interval_dominates_pressure_correction():
    """frames_since=55 with interval=8 flushes via the INTERVAL branch (55>=8),
    independent of the pressure bound — pinning the branch order."""
    assert LIVE_2D.should_flush(frames_since_flush=55, unsaved_in_memory=None) is True


def _reference(p: FlushPolicy, frames_since, unsaved, force) -> bool:
    """Independent spec restatement (different structure) for a grid cross-check."""
    if frames_since <= 0:
        return False
    if force:
        return True
    hits_interval = frames_since >= p.interval
    pressure = frames_since if unsaved is None else unsaved
    hits_pressure = pressure >= max(1, p.cap - p.margin)
    return hits_interval or hits_pressure


def test_grid_matches_reference():
    for policy in (LIVE_2D, LIVE_1D, FlushPolicy(cap=12, margin=8, interval=8)):
        for frames_since in (0, 1, 3, 4, 7, 8, 55, 56, 57, 1000):
            for unsaved in (None, 3, 4, 55, 56):
                for force in (False, True):
                    got = policy.should_flush(frames_since_flush=frames_since,
                                              unsaved_in_memory=unsaved, force=force)
                    assert got is _reference(policy, frames_since, unsaved, force), (
                        policy, frames_since, unsaved, force)


def test_module_is_pure():
    """cadence.py's OWN imports pull no Qt/h5py/numpy.  Loaded BY FILE PATH
    (not via the package) so the reduction __init__ — which does import the
    heavy core — doesn't mask the check (same trick as the display-logic
    purity guard)."""
    import os
    path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "src", "xrd_tools", "reduction", "cadence.py")
    code = (
        "import sys, importlib.util;"
        f"spec=importlib.util.spec_from_file_location('cadence_isolated', {path!r});"
        "mod=importlib.util.module_from_spec(spec);"
        "sys.modules[spec.name]=mod; spec.loader.exec_module(mod);"
        "bad=[m for m in ('PySide6','pyqtgraph','h5py','numpy','pyFAI','fabio')"
        " if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"cadence pulled heavy deps: {out.stdout!r}"
