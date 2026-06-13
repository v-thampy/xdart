# -*- coding: utf-8 -*-
"""ADR-0006 STEP 1 — headless whole-scan GI prepare (prepare_gi_freeze) +
scan_manifest() capability.  Pure/offscreen; no real detector data, no Qt.

These are the always-on CI gates for the extent-discovery logic that moves out
of xdart.  The real-data byte-equivalence (the chosen extremes produce the
(1,5)-union grid) is exercised by the GI spine file once STEP 2 wires it.
"""
from __future__ import annotations

from xrd_tools.core.scan import SourceCapabilities
from xrd_tools.reduction import (
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    PrepareDiagnostics,
    ReductionPlan,
    prepare_gi_freeze,
)
from xrd_tools.reduction.core import _gi_1d_freeze_key, _gi_2d_freeze_keys
from xrd_tools.sources.base import BaseFrameSource


def _gi_plan(motor="th"):
    return ReductionPlan(
        integration_2d=Integration2DPlan(npt_rad=64, npt_azim=48),
        gi=GIMode(incidence_motor=motor, mode_2d="qip_qoop"),
    )


class _ManifestSource:
    """Duck source exposing scan_manifest() (the getattr-probed capability)."""
    def __init__(self, ths):
        # th values may be None to model an unreadable metadata gap
        self._m = [(i + 1, ({} if t is None else {"th": t}))
                   for i, t in enumerate(ths)]

    def scan_manifest(self):
        return list(self._m)


# ── prepare_gi_freeze outcomes ───────────────────────────────────────────────

def test_prepare_pins_value_extremes_not_positional():
    plan2, diag = prepare_gi_freeze(_ManifestSource([0.15, 0.20, 0.25, 0.30, 0.35]),
                                    _gi_plan())
    assert diag.status == "frozen"
    assert diag.scout_indices == (1, 5)                 # (lo_idx, hi_idx)
    assert plan2.extra["gi_freeze_scout_indices"] == [1, 5]
    assert len(diag.scout_refs) == 2 and diag.scout_refs[0]["th"] == 0.15


def test_prepare_extremes_are_order_independent():
    """Reversed metadata order → SAME extreme FRAMES (chosen by value)."""
    _, asc = prepare_gi_freeze(_ManifestSource([0.15, 0.2, 0.25, 0.3, 0.35]), _gi_plan())
    _, desc = prepare_gi_freeze(_ManifestSource([0.35, 0.3, 0.25, 0.2, 0.15]), _gi_plan())
    assert set(asc.scout_indices) == set(desc.scout_indices) == {1, 5}


def test_prepare_tolerates_metadata_gaps():
    """Extremes come from the READABLE frames; unreadable middles are skipped."""
    _, diag = prepare_gi_freeze(
        _ManifestSource([0.15, None, None, 0.35, None]), _gi_plan())
    assert diag.status == "frozen"
    assert set(diag.scout_indices) == {1, 4}


def test_prepare_skips_non_gi_plan():
    _, diag = prepare_gi_freeze(_ManifestSource([0.1, 0.2]),
                                ReductionPlan(integration_2d=None))
    assert diag.status == "skip"


def test_prepare_skips_fixed_or_manual_incidence():
    _, diag = prepare_gi_freeze(_ManifestSource([0.1, 0.2]), _gi_plan(motor="2.5"))
    assert diag.status == "skip"


def test_prepare_skips_single_frame_and_single_incidence():
    _, one = prepare_gi_freeze(_ManifestSource([0.2]), _gi_plan())
    assert one.status == "skip"
    _, flat = prepare_gi_freeze(_ManifestSource([0.2, 0.2, 0.2]), _gi_plan())
    assert flat.status == "skip"                        # no sweep -> chunk grid is fine


def test_prepare_unverifiable_without_manifest():
    """A source with no scan_manifest() (or fewer than two readable incidences)
    is unverifiable → warn-and-proceed, never an exception, never a pin."""
    class _NoManifest:
        pass
    p, diag = prepare_gi_freeze(_NoManifest(), _gi_plan())
    assert diag.status == "unverifiable"
    assert "gi_freeze_scout_indices" not in p.extra

    _, few = prepare_gi_freeze(_ManifestSource([0.15, None, None]), _gi_plan())
    assert few.status == "unverifiable"                  # <2 readable incidences


def test_prepare_skips_when_ranges_already_pinned():
    """Nothing to freeze (both GI output grids pinned / absent) → skip WITHOUT
    enumerating (the T0-3 silent skip): scan_manifest() is never called."""
    # q_total's 1D output axis is radial_range; pin it + drop 2D ⇒ no freeze key.
    pinned = ReductionPlan(
        integration_1d=Integration1DPlan(radial_range=(0.1, 6.0)),
        integration_2d=None,
        gi=GIMode(incidence_motor="th"),
    )
    assert _gi_1d_freeze_key(pinned) is None and not _gi_2d_freeze_keys(pinned)
    calls = []

    class _Spy:
        def scan_manifest(self):
            calls.append(1)
            return [(1, {"th": 0.1}), (2, {"th": 0.3})]

    _, diag = prepare_gi_freeze(_Spy(), pinned)
    assert diag.status == "skip"
    assert calls == []          # short-circuited BEFORE enumerating


def test_prepare_returns_immutable_diagnostics():
    _, diag = prepare_gi_freeze(_ManifestSource([0.1, 0.3]), _gi_plan())
    assert isinstance(diag, PrepareDiagnostics)
    import dataclasses
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        diag.status = "x"           # type: ignore[misc]


# ── BaseFrameSource.scan_manifest() default + capability gating ──────────────

class _TinySource(BaseFrameSource):
    def __init__(self, metas, *, has_manifest):
        super().__init__(
            name="tiny", frame_indices=range(1, len(metas) + 1),
            capabilities=SourceCapabilities(has_metadata=True,
                                            has_scan_manifest=has_manifest),
        )
        self._metas = {i + 1: m for i, m in enumerate(metas)}

    def load_frame(self, index):
        raise AssertionError("scan_manifest must be image-free")

    def metadata_for(self, index):
        return self._metas[int(index)]


def test_base_scan_manifest_default_gated_on_capability():
    src = _TinySource([{"th": 0.1}, {"th": 0.3}], has_manifest=True)
    manifest = src.scan_manifest()
    assert [i for i, _m in manifest] == [1, 2]           # == frame_indices
    assert manifest[1][1]["th"] == 0.3
    # and it is image-free: load_frame would have raised

    off = _TinySource([{"th": 0.1}], has_manifest=False)
    assert off.scan_manifest() is None                   # None, not []


def test_base_scan_manifest_drives_prepare_end_to_end():
    """A real BaseFrameSource (capability on) feeds prepare_gi_freeze headlessly,
    image-free."""
    src = _TinySource([{"th": 0.15}, {"th": 0.25}, {"th": 0.35}], has_manifest=True)
    _, diag = prepare_gi_freeze(src, _gi_plan())
    assert diag.status == "frozen"
    assert set(diag.scout_indices) == {1, 3}
