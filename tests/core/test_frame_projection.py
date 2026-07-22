# -*- coding: utf-8 -*-
"""X1-1/2/3 headless read-authority projection (``xrd_tools.session.frame_projection``).

Production-wired: real :class:`FrameView` / :class:`FrameRecord` /
:class:`FrameRecordStore` (real thinning + hydration path) and a small
``MetadataProvider``-shaped object with the SAME method surface the real
providers expose.  No fake on the projection under test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from xrd_tools.core import FrameRecord, FrameView
from xrd_tools.core.energy import WavelengthUnit
from xrd_tools.session import FrameRecordStore
from xrd_tools.session.frame_projection import (
    Capability,
    CapabilityDisposition,
    CapabilityState,
    MetadataConflictError,
    MetadataRow,
    WavelengthStatus,
    display_capabilities,
    metadata_row_from_provider,
    metadata_row_from_record,
    metadata_row_from_view,
    normalization_channels,
    normalization_value,
    project_frame,
    wavelength_evidence,
)


# ── builders (real value types) ──────────────────────────────────────────────

def _v1d(label=0, *, meta=None, source=None, resident=True, numeric=None):
    r1 = SimpleNamespace(
        radial=np.linspace(1.0, 5.0, 8), intensity=np.arange(8.0),
        unit="q_A^-1", sigma=None) if resident else None
    v = FrameView.from_results(
        label=label, result_1d=r1, metadata_raw=meta or {},
        metadata_numeric=numeric,
        source_path=(source[0] if source else None),
        source_frame_index=(source[1] if source else None))
    return v


def _v2d(label=0, *, meta=None, source=None):
    r2 = SimpleNamespace(
        radial=np.linspace(1.0, 5.0, 6), azimuthal=np.linspace(-180, 180, 4),
        intensity=np.arange(24.0).reshape(6, 4), unit="q_A^-1",
        azimuthal_unit="deg", sigma=None)
    return FrameView.from_results(
        label=label, result_2d=r2, metadata_raw=meta or {},
        source_path=(source[0] if source else None),
        source_frame_index=(source[1] if source else None))


def _record_1d(label=0, *, meta=None, source=None):
    return FrameRecord.from_view(_v1d(label, meta=meta, source=source))


class _Provider:
    """A MetadataProvider-shaped object (same surface as the real providers)."""

    def __init__(self, *, motors=None, table=None, constants=None,
                 metadata=None, wavelength=None, wavelength_unit=None,
                 frame_count=None):
        self._motors = motors or {}
        self._table = table or {}
        self._constants = constants or {}
        # metadata[i] is the per-frame counter/constant dict (table > constant,
        # scanned motors excluded) — exactly what the real metadata_for returns
        self._metadata = metadata or {}
        self._wavelength = wavelength
        self._wavelength_unit = wavelength_unit
        self._frame_count = frame_count
        self.metadata_for_calls = []

    def frame_count(self):
        return self._frame_count

    def motors(self):
        return {k: np.asarray(v) for k, v in self._motors.items()}

    def scan_table(self):
        return {k: np.asarray(v) for k, v in self._table.items()}

    def constants(self):
        return dict(self._constants)

    def metadata_for(self, i):
        self.metadata_for_calls.append(i)
        return dict(self._metadata.get(i, {}))

    def wavelength(self):
        return self._wavelength

    def wavelength_unit(self):
        return self._wavelength_unit


# ── 1. provider row composition + scanned-motor precedence ───────────────────

def test_provider_scanned_motor_beats_counter_beats_constant():
    prov = _Provider(
        motors={"th": [10.0, 20.0, 30.0]},
        table={"th": [10.0, 20.0, 30.0], "i0": [100.0, 110.0, 120.0]},
        constants={"i0": 999.0, "gain": 5.0},
        # a stale "th" in the per-frame table too: the SCANNED motor must win
        metadata={1: {"i0": 110.0, "gain": 5.0, "th": 99.0}})
    row = metadata_row_from_provider(prov, 1)
    assert row.raw["th"] == 20.0        # scanned motor beats the stale table th=99
    assert row.raw["i0"] == 110.0       # per-frame counter beat the 999 constant
    assert row.raw["gain"] == 5.0       # constant present


# ── 2. counters/constants, mixed scalar/string, numeric filtering, case ──────

def test_mixed_scalar_string_numeric_filter_and_case():
    meta = {"th": 10.5, "sample": "NbN", "profile": [1, 2, 3], "I0": 250.0}
    row = MetadataRow(meta)
    assert row.raw["sample"] == "NbN"                  # heterogeneous preserved
    assert row.raw["profile"] == (1, 2, 3)             # mutable containers frozen
    assert dict(row.numeric) == {"th": 10.5, "I0": 250.0}  # finite scalars only
    # case-insensitive guarded selection (resolve_monitor_norm reused)
    assert normalization_value(row, "i0") == 250.0
    assert normalization_value(row, "I0") == 250.0
    assert normalization_value(row, "sample") is None    # non-numeric channel


# ── 3. negative / out-of-range frame identities never wrap ───────────────────

def test_negative_frame_index_is_rejected_not_wrapped():
    prov = _Provider(motors={"th": [10.0, 20.0, 30.0]},
                     metadata={2: {"i0": 3.0}})
    with pytest.raises(IndexError):
        metadata_row_from_provider(prov, -1)          # must not wrap to th=30
    # sanity: the last real frame is reachable the normal way
    assert metadata_row_from_provider(prov, 2).raw["th"] == 30.0


def test_out_of_range_frame_index_is_rejected():
    prov = _Provider(motors={"th": [10.0, 20.0, 30.0]})
    with pytest.raises(IndexError):
        metadata_row_from_provider(prov, 3)


def test_metadataless_provider_returns_empty_row_for_any_index():
    prov = _Provider()                                # no motors/table/metadata
    assert not metadata_row_from_provider(prov, 0)    # empty, not an error
    assert not metadata_row_from_provider(prov, 7)


# ── 4. active vs explicitly requested multimode record projection ────────────

def test_record_projection_prefers_explicit_then_active():
    v1 = _v1d(3, meta={"th": 1.0, "shared": 9.0})
    v2 = _v2d(3, meta={"chi": 2.0, "shared": 9.0})
    record = FrameRecord(label=3, results_1d={"a": v1}, results_2d={"b": v2},
                         active_mode_1d="a", active_mode_2d="b")
    # default: active 1D preferred
    assert "th" in metadata_row_from_record(record).raw
    # explicit 2D requested
    assert "chi" in metadata_row_from_record(record, mode_2d="b").raw
    # 2D-only record → falls through to the 2D view
    record2d = FrameRecord(label=4, results_2d={"b": _v2d(4, meta={"chi": 7.0})},
                           active_mode_2d="b")
    assert metadata_row_from_record(record2d).raw["chi"] == 7.0


# ── 5. contradictory metadata / source identity fails clearly ────────────────

def test_contradictory_metadata_value_raises():
    v1 = _v1d(0, meta={"temperature": 100.0})
    v2 = _v2d(0, meta={"temperature": 200.0})           # same frame, disagrees
    record = FrameRecord(label=0, results_1d={"a": v1}, results_2d={"b": v2},
                         active_mode_1d="a", active_mode_2d="b")
    with pytest.raises(MetadataConflictError):
        metadata_row_from_record(record)


def test_contradictory_source_identity_raises_and_is_error_capability():
    v1 = _v1d(0, meta={"th": 1.0}, source=("/data/a.h5", 0))
    v2 = _v2d(0, meta={"th": 1.0}, source=("/data/b.h5", 0))
    record = FrameRecord(label=0, results_1d={"a": v1}, results_2d={"b": v2},
                         active_mode_1d="a", active_mode_2d="b")
    with pytest.raises(MetadataConflictError):
        metadata_row_from_record(record)
    caps = display_capabilities(record)
    assert caps.metadata.state is CapabilityState.ERROR
    assert caps.metadata.disposition is CapabilityDisposition.ERROR
    assert caps.integrated_1d.state is CapabilityState.ERROR
    assert caps.integrated_1d.disposition is CapabilityDisposition.ERROR


def test_agreeing_metadata_across_modes_is_not_a_conflict():
    v1 = _v1d(0, meta={"th": 1.0, "shared": 5.0})
    v2 = _v2d(0, meta={"chi": 2.0, "shared": 5.0})       # agrees on shared
    record = FrameRecord(label=0, results_1d={"a": v1}, results_2d={"b": v2},
                         active_mode_1d="a", active_mode_2d="b")
    assert metadata_row_from_record(record).raw["th"] == 1.0   # no raise


# ── 6. normalization channel discovery + guarded selected value ──────────────

def test_normalization_channels_and_guarded_value():
    row = MetadataRow({"i0": 500.0, "i1": 0.0, "ineg": -3.0, "name": "x"})
    assert normalization_channels(row) == ("i0", "i1", "ineg")   # numeric only
    assert normalization_value(row, "i0") == 500.0
    assert normalization_value(row, "i1") is None       # zero rejected (guard)
    assert normalization_value(row, "ineg") is None     # negative rejected
    assert normalization_value(row, "missing") is None
    assert normalization_value(row, None) is None


# ── 7. wavelength evidence: canonical metres, explicit units only (R3-P1) ─────

def test_wavelength_present_absent_conflict(caplog):
    """Canonical contract: every source is canonicalized to METRES before
    comparison; only explicitly unit-declared sources contribute."""
    import logging
    caplog.set_level(logging.WARNING)
    # provider declares 1.54 Å; metadata carries the explicit metre key —
    # cross-unit agreement is real agreement after canonicalization.
    prov = _Provider(wavelength=1.54, wavelength_unit=WavelengthUnit.ANGSTROM)
    row = MetadataRow({"wavelength_m": 1.54e-10, "th": 2.0})
    present = wavelength_evidence(provider=prov, row=row)
    assert present.status is WavelengthStatus.PRESENT
    assert present.value == pytest.approx(1.54e-10)          # canonical metres
    assert present.sources["provider"] == pytest.approx(1.54e-10)
    assert present.sources["wavelength_m"] == pytest.approx(1.54e-10)

    absent = wavelength_evidence(row=MetadataRow({"th": 2.0}))
    assert absent.status is WavelengthStatus.ABSENT and absent.value is None

    # provider Å genuinely disagrees with the explicit metre key → conflict,
    # with each source named and reported in canonical metres.
    conflict = wavelength_evidence(
        provider=_Provider(
            wavelength=1.5406, wavelength_unit=WavelengthUnit.ANGSTROM),
        row=MetadataRow({"wavelength_m": 0.9744e-10}))
    assert conflict.status is WavelengthStatus.CONFLICT and conflict.value is None
    assert set(conflict.sources) == {"provider", "wavelength_m"}
    # a missing wavelength is not an operator warning here
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_wavelength_bare_key_and_unknown_aliases_are_structured_absence():
    """Only ``wavelength_m`` / ``wavelength_A`` are accepted.  A bare
    ``wavelength`` (no enforceable unit) and unknown aliases are ignored —
    neither false presence nor false conflict."""
    ignored = wavelength_evidence(
        row=MetadataRow({"wavelength": 1.5406, "lambda_A": 1.54, "th": 2.0}))
    assert ignored.status is WavelengthStatus.ABSENT
    assert ignored.value is None and not dict(ignored.sources)

    # ... and a bare key cannot manufacture a conflict against an explicit one.
    no_conflict = wavelength_evidence(
        row=MetadataRow({"wavelength": 77.0, "wavelength_A": 1.54}))
    assert no_conflict.status is WavelengthStatus.PRESENT
    assert no_conflict.value == pytest.approx(1.54e-10)


def test_wavelength_undeclared_provider_unit_contributes_no_evidence():
    """A provider value with no declared unit is NOT evidence (no magnitude
    inference); a legacy provider without ``wavelength_unit()`` likewise."""
    undeclared = _Provider(wavelength=1.5406)          # wavelength_unit=None
    assert wavelength_evidence(provider=undeclared).status \
        is WavelengthStatus.ABSENT

    legacy = SimpleNamespace(wavelength=lambda: 1.5406)  # no wavelength_unit()
    assert wavelength_evidence(provider=legacy).status \
        is WavelengthStatus.ABSENT

    # ... and an undeclared provider cannot manufacture a conflict against an
    # explicit metadata key (the old cross-unit false-conflict shape).
    ev = wavelength_evidence(
        provider=undeclared, row=MetadataRow({"wavelength_m": 0.9744e-10}))
    assert ev.status is WavelengthStatus.PRESENT
    assert ev.value == pytest.approx(0.9744e-10)
    assert set(ev.sources) == {"wavelength_m"}


def test_wavelength_explicit_one_angstrom_is_valid_evidence():
    """Explicit 1.0 Å is real physical evidence even though it canonicalizes to
    the historical 1e-10 m constructor sentinel — sentinel rejection is
    provenance-sensitive and never applies to an explicitly declared source."""
    from_key = wavelength_evidence(row=MetadataRow({"wavelength_A": 1.0}))
    assert from_key.status is WavelengthStatus.PRESENT
    assert from_key.value == pytest.approx(1.0e-10)

    from_provider = wavelength_evidence(provider=_Provider(
        wavelength=1.0, wavelength_unit=WavelengthUnit.ANGSTROM))
    assert from_provider.status is WavelengthStatus.PRESENT
    assert from_provider.value == pytest.approx(1.0e-10)


def test_wavelength_explicit_key_disagreement_is_conflict():
    """Both explicit keys present and disagreeing (beyond rtol 1e-3 after
    canonicalization) is a REAL typed conflict, reachable with provider=None."""
    conflict = wavelength_evidence(
        row=MetadataRow({"wavelength_m": 1.54e-10, "wavelength_A": 0.9744}))
    assert conflict.status is WavelengthStatus.CONFLICT
    assert conflict.value is None
    assert set(conflict.sources) == {"wavelength_m", "wavelength_A"}
    assert conflict.sources["wavelength_m"] == pytest.approx(1.54e-10)
    assert conflict.sources["wavelength_A"] == pytest.approx(0.9744e-10)

    # agreement within rtol 1e-3 (rounded header vs full precision) stays
    # present — cross-unit, post-canonicalization.
    agree = wavelength_evidence(
        row=MetadataRow({"wavelength_m": 1.540598e-10, "wavelength_A": 1.5406}))
    assert agree.status is WavelengthStatus.PRESENT
    assert agree.value == pytest.approx(1.540598e-10, rel=1e-3)


def test_wavelength_keys_match_case_insensitively():
    ev = wavelength_evidence(
        row=MetadataRow({"Wavelength_M": 1.54e-10, "WAVELENGTH_A": 1.54}))
    assert ev.status is WavelengthStatus.PRESENT
    assert set(ev.sources) == {"wavelength_m", "wavelength_A"}


# ── 8. capability states: resident / thinned / source-fallback / none / error ─

def test_capabilities_resident_record():
    record = _record_1d(0, meta={"th": 1.0})
    caps = display_capabilities(record)
    assert caps.integrated_1d.state is CapabilityState.AVAILABLE
    assert caps.integrated_1d.disposition is CapabilityDisposition.RESIDENT
    assert caps.metadata.state is CapabilityState.AVAILABLE
    assert caps.metadata.disposition is CapabilityDisposition.RESIDENT
    assert caps.integrated_2d.state is CapabilityState.UNAVAILABLE   # no 2D mode
    assert caps.integrated_2d.disposition is CapabilityDisposition.ABSENT


def test_capabilities_thinned_is_pending():
    store = FrameRecordStore(max_heavy_items=0, require_persisted_for_eviction=False)
    store.upsert(_record_1d(5, meta={"th": 1.0}))       # thinned immediately
    thinned = store.get(5)
    assert store.has_heavy_payload(5) is False
    caps = display_capabilities(thinned)
    assert caps.integrated_1d.state is CapabilityState.PENDING   # hydratable
    # no store context here → best-effort, typed as such (R3-P2)
    assert caps.integrated_1d.disposition is CapabilityDisposition.BEST_EFFORT
    assert caps.metadata.state is CapabilityState.AVAILABLE      # metadata survived


def test_persisted_mode_without_hydrator_is_not_falsely_pending():
    resident = _record_1d(6, meta={"th": 1.0})
    thinner = FrameRecordStore(
        max_heavy_items=0, require_persisted_for_eviction=False)
    thinner.upsert(resident)
    store = FrameRecordStore()
    store.upsert(thinner.get(6), persisted=True)

    assert store.persisted_modes(6) == frozenset({("1d", "default")})
    assert store.hydratable_modes(6) == frozenset()
    projected = project_frame(store, 6)
    assert projected.capabilities.integrated_1d.state \
        is CapabilityState.UNAVAILABLE
    assert projected.capabilities.integrated_1d.disposition \
        is CapabilityDisposition.PERSISTED_NO_HYDRATOR
    assert project_frame(store, 6, hydrate=True).capabilities.integrated_1d.state \
        is CapabilityState.UNAVAILABLE


def test_mode_presence_alone_does_not_make_thumbnail_recoverable():
    store = FrameRecordStore()
    store.upsert(_record_1d(6, meta={"th": 1.0}))
    projected = project_frame(store, 6)
    assert projected.capabilities.thumbnail.state is CapabilityState.UNAVAILABLE


def test_capabilities_source_fallback_eligible_and_unavailable():
    # raw not resident, but a source path makes it source-fallback eligible
    v = _v1d(0, meta={"th": 1.0}, source=("/data/master.h5", 0))
    caps = display_capabilities(FrameRecord.from_view(v))
    assert caps.raw.state is CapabilityState.PENDING            # source-fallback
    assert caps.raw.disposition is CapabilityDisposition.SOURCE_FALLBACK
    assert caps.thumbnail.disposition is CapabilityDisposition.SOURCE_FALLBACK
    # no source, no raw → unavailable
    caps2 = display_capabilities(_record_1d(0, meta={"th": 1.0}))
    assert caps2.raw.state is CapabilityState.UNAVAILABLE
    assert caps2.raw.disposition is CapabilityDisposition.ABSENT


def test_capabilities_empty_record_is_all_unavailable():
    empty = FrameRecord(label=99)                       # a label, no results
    caps = display_capabilities(empty)
    assert caps.integrated_1d.state is CapabilityState.UNAVAILABLE
    assert caps.integrated_2d.state is CapabilityState.UNAVAILABLE
    assert caps.raw.state is CapabilityState.UNAVAILABLE
    assert caps.metadata.state is CapabilityState.UNAVAILABLE
    for fact in (caps.metadata, caps.integrated_1d, caps.integrated_2d,
                 caps.raw, caps.thumbnail):
        assert fact.disposition is CapabilityDisposition.ABSENT


# ── 9. thinned lookup preserves metadata/capabilities without heavy arrays ────

def test_thinned_lookup_preserves_metadata_then_hydrates():
    resident = _record_1d(7, meta={"th": 3.0, "i0": 400.0})
    # produce a GENUINELY thinned record through the real eviction path
    thinner = FrameRecordStore(max_heavy_items=0, require_persisted_for_eviction=False)
    thinner.upsert(resident)
    thinned = thinner.get(7)
    assert thinner.has_heavy_payload(7) is False

    store = FrameRecordStore()                           # default cap: has headroom
    store.set_hydrator(lambda label: resident)           # synchronous hydrator
    store.upsert(thinned, persisted=True)                # thinned AND on disk → hydratable

    # resident-only projection: metadata + capabilities without heavy arrays
    proj = project_frame(store, 7)
    assert proj.present is True
    assert proj.metadata.raw["th"] == 3.0
    assert proj.capabilities.integrated_1d.state is CapabilityState.PENDING  # persisted → hydratable
    assert store.has_heavy_payload(7) is False           # projection did not load arrays

    # explicit synchronous hydration flips it AVAILABLE
    hydrated = project_frame(store, 7, hydrate=True)
    assert hydrated.capabilities.integrated_1d.state is CapabilityState.AVAILABLE
    assert store.has_heavy_payload(7) is True


def test_dropped_mode_is_unavailable_not_falsely_pending():
    """A consciously-dropped mode (mark_dropped / MEM-1b) is thinned but NOT on
    disk and NOT hydratable → UNAVAILABLE, not a false PENDING (fix RP#5)."""
    r2 = SimpleNamespace(
        radial=np.linspace(1.0, 5.0, 6), azimuthal=np.linspace(-180, 180, 4),
        intensity=np.arange(24.0).reshape(6, 4), unit="q_A^-1",
        azimuthal_unit="deg", sigma=None)
    record = FrameRecord.from_view(
        FrameView.from_results(label=8, result_2d=r2, metadata_raw={"th": 1.0}))
    store = FrameRecordStore()
    store.upsert(record)
    store.mark_dropped(8, modes=("2d", "default"))       # husk kept, never persisted
    assert store.has_heavy_payload(8) is False
    assert ("2d", "default") not in store.persisted_modes(8)

    proj = project_frame(store, 8)
    assert proj.capabilities.integrated_2d.state is CapabilityState.UNAVAILABLE
    assert proj.capabilities.integrated_2d.disposition \
        is CapabilityDisposition.DROPPED
    # the pure record-level projection (no store context) is best-effort PENDING
    best_effort = display_capabilities(store.get(8)).integrated_2d
    assert best_effort.state is CapabilityState.PENDING
    assert best_effort.disposition is CapabilityDisposition.BEST_EFFORT


def test_project_frame_absent_record():
    store = FrameRecordStore()
    proj = project_frame(store, 123)
    assert proj.present is False
    assert not proj.metadata
    assert proj.capabilities.integrated_1d.state is CapabilityState.UNAVAILABLE
    assert proj.capabilities.integrated_1d.disposition \
        is CapabilityDisposition.ABSENT
    assert proj.wavelength.status == "absent"
    assert proj.wavelength.status is WavelengthStatus.ABSENT


def test_project_frame_merges_owner_scoped_provider_row():
    store = FrameRecordStore()
    store.upsert(_record_1d(10, meta={}, source=("/data/raw.nxs", 1)))
    provider = _Provider(
        motors={"th": [1.0, 2.0]}, metadata={1: {"i0": 42.0}}, frame_count=2)

    projected = project_frame(store, 10, provider=provider)
    assert dict(projected.metadata.raw) == {"i0": 42.0, "th": 2.0}
    assert projected.normalization_channels == ("i0", "th")
    assert projected.capabilities.metadata.state is CapabilityState.AVAILABLE


def test_project_frame_accepts_explicit_provider_identity_without_source_index():
    store = FrameRecordStore()
    store.upsert(_record_1d("frame-a", meta={}))
    provider = _Provider(metadata={1: {"i0": 12.0}}, frame_count=2)
    projected = project_frame(
        store, "frame-a", provider=provider, provider_frame_index=1)
    assert projected.metadata.raw["i0"] == 12.0


def test_project_frame_reports_stored_provider_metadata_conflict():
    store = FrameRecordStore()
    store.upsert(
        _record_1d(10, meta={"i0": 11.0}, source=("/data/raw.nxs", 1)))
    provider = _Provider(metadata={1: {"i0": 42.0}}, frame_count=2)

    projected = project_frame(store, 10, provider=provider)
    assert projected.capabilities.metadata.state is CapabilityState.ERROR
    assert "stored/provider metadata disagree" in projected.capabilities.metadata.reason
    assert not projected.metadata


def test_store_identity_cannot_override_conflicting_record_identity():
    store = FrameRecordStore()
    store.upsert(
        _record_1d(11, meta={"i0": 1.0}, source=("/actual.h5", 0)),
        source_identity="/other.h5#0",
    )
    projected = project_frame(store, 11)
    assert projected.capabilities.identity == "<conflict>"
    assert projected.capabilities.metadata.state is CapabilityState.ERROR
    assert projected.capabilities.raw.state is CapabilityState.ERROR
    assert not projected.metadata


def test_projection_metadata_array_is_detached_and_read_only():
    values = np.array([1.0, 2.0, 3.0])
    store = FrameRecordStore()
    store.upsert(_record_1d(12, meta={"positions": values}))
    projected = project_frame(store, 12)
    stored = store.get(12).view_1d().metadata_raw["positions"]

    assert projected.metadata.raw["positions"] is not stored
    assert projected.metadata.raw["positions"].flags.writeable is False
    with pytest.raises(ValueError):
        projected.metadata.raw["positions"][0] = 99.0
    np.testing.assert_array_equal(stored, [1.0, 2.0, 3.0])


# ── 10. import purity: no Qt / h5py / fabio / pyFAI / xdart ───────────────────

def test_frame_projection_import_is_headless():
    root = Path(__file__).resolve().parents[2]
    code = textwrap.dedent(
        """
        import sys
        import xrd_tools.session.frame_projection  # noqa: F401
        forbidden = ("xdart", "PySide6", "PySide2", "PyQt5", "PyQt6",
                     "pyqtgraph", "matplotlib", "h5py", "fabio", "pyFAI")
        bad = sorted(r for r in forbidden if any(
            n == r or n.startswith(r + ".") for n in sys.modules))
        if bad:
            print(",".join(bad)); sys.exit(1)
        """)
    env = dict(os.environ)
    src = root / "src"
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"]
                                    if env.get("PYTHONPATH") else "")
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, (
        f"frame_projection import pulled in: {proc.stdout.strip()}\n"
        f"{proc.stderr.strip()}")


# ── 11. single-frame projection does not scale with scan length ──────────────

def test_single_frame_projection_is_o1_in_scan_length():
    def _time_projection(n):
        prov = _Provider(
            motors={"th": np.linspace(0.0, 10.0, n)},
            table={"th": np.linspace(0.0, 10.0, n), "i0": np.arange(float(n))},
            metadata={5: {"i0": 5.0}})
        start = time.perf_counter()
        for _ in range(200):
            metadata_row_from_provider(prov, 5)
        return time.perf_counter() - start

    small = _time_projection(1_000)
    large = _time_projection(1_000_000)               # 1000x longer scan
    # a single-frame projection indexes one element; it must not scale with N
    assert large < small * 20 + 0.05, (
        f"projection scaled with scan length: {small:.4f}s -> {large:.4f}s")


def test_provider_row_reads_only_the_selected_frame():
    prov = _Provider(motors={"th": [1.0, 2.0, 3.0, 4.0, 5.0]},
                     metadata={i: {"i0": float(i)} for i in range(5)})
    metadata_row_from_provider(prov, 2)
    assert prov.metadata_for_calls == [2]             # only frame 2, not all 5


def test_real_bluesky_provider_complete_row_does_not_materialize_full_table(
    tmp_path, monkeypatch,
):
    from xrd_tools.sources.metadata_provider import BlueskyMetadataProvider

    path = tmp_path / "row.nxs"
    with h5py.File(path, "w") as root:
        entry = root.create_group("entry")
        data = entry.create_group("data")
        data.create_dataset("th", data=np.array([1.0, 2.0, 3.0]))
        data.create_dataset("i0", data=np.array([10.0, 20.0, 30.0]))
        data.create_dataset("EPOCH", data=np.array([100.0, 101.0, 102.0]))
        positioners = entry.create_group("instrument/positioners")
        positioners.create_group("th")
        bluesky = entry.create_group("instrument/bluesky/metadata")
        bluesky.create_dataset("motors", data=b"!!python/tuple\n- th\n")
        provider = BlueskyMetadataProvider(entry, frame_count=3)
        monkeypatch.setattr(
            provider, "_ensure_table",
            lambda: pytest.fail("single-row projection materialized the full table"),
        )
        row = metadata_row_from_provider(provider, 1)
        assert row.raw["th"] == pytest.approx(2.0)
        assert row.raw["i0"] == pytest.approx(20.0)
        assert row.raw["EPOCH"] == pytest.approx(101.0)
        assert provider._table is None


# ── review corrections (RP#1–#5 + minors) ────────────────────────────────────

def test_provider_frame_count_is_authoritative_over_array_lengths():
    """RP#1: the frame bound follows the provider's frame_count(), not
    max(array lengths), in BOTH directions."""
    # OVER-SHOOT: a scanned-motor column longer than frame_count (a baseline
    # point) must not make a nonexistent frame readable.
    over = _Provider(motors={"th": [10.0, 20.0, 30.0, 40.0]},   # len 4
                     metadata={0: {"i0": 1.0}, 1: {"i0": 2.0}, 2: {"i0": 3.0}},
                     frame_count=3)
    with pytest.raises(IndexError):
        metadata_row_from_provider(over, 3)              # frame 3 does not exist
    assert metadata_row_from_provider(over, 2).raw["th"] == 30.0

    # UNDER-SHOOT: per-frame arrays shorter than frame_count must not refuse a
    # valid frame; its frame-independent constants still resolve.
    under = _Provider(motors={"th": [10.0, 20.0, 30.0, 40.0, 50.0]},  # len 5
                      metadata={i: {"exposure": 0.5} for i in range(10)},
                      frame_count=10)
    row = metadata_row_from_provider(under, 7)           # valid (7 < 10)
    assert row.raw["exposure"] == 0.5                    # constant resolved
    assert "th" not in row.raw                           # baseline-short motor absent


def test_explicit_absent_mode_falls_back_to_active_metadata():
    """RP#2: an explicitly requested mode that is absent must not silently drop
    the frame's real metadata — it falls through to the active view."""
    record = FrameRecord(
        label=0, results_1d={"norm": _v1d(0, meta={"I0": 100.0})},
        active_mode_1d="norm")
    row = metadata_row_from_record(record, mode_1d="raw")   # 'raw' not present
    assert row.raw["I0"] == 100.0                           # fell back to active


def test_source_index_none_reconciles_with_concrete_index():
    """RP#3: a same-path view with an unknown (None) source index reconciles
    with one that knows the index — not a false identity conflict."""
    v1 = _v1d(0, meta={"I0": 5.0}, source=("/f.h5", 0))
    v2 = _v2d(0, meta={"I0": 5.0}, source=("/f.h5", None))
    record = FrameRecord(label=0, results_1d={"a": v1}, results_2d={"b": v2},
                         active_mode_1d="a", active_mode_2d="b")
    # no raise; the identity resolves to the concrete index
    assert metadata_row_from_record(record).raw["I0"] == 5.0
    caps = display_capabilities(record)
    assert caps.identity == "/f.h5#0"
    assert caps.metadata.state is CapabilityState.AVAILABLE


def test_project_frame_returns_error_projection_not_raise_on_conflict():
    """RP#4: the single lookup boundary returns a typed ERROR projection on a
    conflicting record instead of raising."""
    v1 = _v1d(0, meta={"exposure": 1.0})
    v2 = _v2d(0, meta={"exposure": 2.0})
    record = FrameRecord(label=0, results_1d={"a": v1}, results_2d={"b": v2},
                         active_mode_1d="a", active_mode_2d="b")
    store = FrameRecordStore()
    store.upsert(record)
    proj = project_frame(store, 0)                       # must NOT raise
    assert proj.present is True
    assert proj.capabilities.metadata.state is CapabilityState.ERROR
    assert not proj.metadata                             # empty on conflict


def test_equal_ndarray_metadata_across_modes_is_not_a_conflict():
    """Minor: byte-equal array-valued metadata in two modes is not a conflict."""
    v1 = _v1d(0, meta={"positions": np.array([1.0, 2.0, 3.0])})
    v2 = _v2d(0, meta={"positions": np.array([1.0, 2.0, 3.0])})
    record = FrameRecord(label=0, results_1d={"a": v1}, results_2d={"b": v2},
                         active_mode_1d="a", active_mode_2d="b")
    row = metadata_row_from_record(record)               # no raise
    assert "positions" in row.raw
    # a genuine array disagreement still conflicts
    v3 = _v2d(0, meta={"positions": np.array([9.0, 9.0, 9.0])})
    bad = FrameRecord(label=0, results_1d={"a": v1}, results_2d={"b": v3},
                      active_mode_1d="a", active_mode_2d="b")
    with pytest.raises(MetadataConflictError):
        metadata_row_from_record(bad)


# ── 12. public API use from a synthetic headless consumer ────────────────────

def test_public_api_end_to_end_headless_consumer():
    from xrd_tools.session import (
        project_frame as pf, normalization_value as nv,
        metadata_row_from_view as mrv)
    store = FrameRecordStore()
    store.upsert(_record_1d(0, meta={"th": 12.0, "i0": 800.0}),
                 source_identity="/data/scan.h5#0")
    proj = pf(store, 0)
    assert proj.present
    assert proj.capabilities.identity == "/data/scan.h5#0"
    assert "i0" in proj.normalization_channels
    assert nv(proj.metadata, "i0") == 800.0
    assert mrv(store.get(0).view_1d()).raw["th"] == 12.0


# ── independent-review corrections (IFR-1/2/3) ───────────────────────────────

def test_project_frame_out_of_range_provider_index_is_error_not_raise():
    """IFR-1: an out-of-range provider index (explicit, negative, or inferred
    from a mismatched record source index) becomes a typed metadata ERROR
    projection — it must never leak IndexError from the lookup boundary."""
    prov = _Provider(motors={"th": [10.0, 20.0, 30.0]},
                     metadata={i: {"i0": float(i)} for i in range(3)},
                     frame_count=3)
    store = FrameRecordStore()
    store.upsert(_record_1d(0, meta={"th": 1.0}))

    proj = project_frame(store, 0, provider=prov, provider_frame_index=999)
    assert proj.present is True
    assert proj.capabilities.metadata.state is CapabilityState.ERROR
    assert not proj.metadata

    proj_neg = project_frame(store, 0, provider=prov, provider_frame_index=-1)
    assert proj_neg.capabilities.metadata.state is CapabilityState.ERROR

    # automatic path: a record whose source_frame_index exceeds the provider range
    store.upsert(FrameRecord.from_view(
        _v1d(1, meta={"th": 1.0}, source=("/raw.h5", 999))))
    proj_auto = project_frame(store, 1, provider=prov)      # inferred index = 999
    assert proj_auto.capabilities.metadata.state is CapabilityState.ERROR

    # a VALID index still merges the provider row normally (no regression); the
    # record's stored fields don't collide with the provider's counters/motors
    store.upsert(FrameRecord.from_view(
        _v1d(2, meta={"note": "x"}, source=("/raw.h5", 2))))
    ok = project_frame(store, 2, provider=prov)
    assert ok.capabilities.metadata.state is not CapabilityState.ERROR
    assert ok.metadata.raw["note"] == "x"                   # stored field preserved
    assert ok.metadata.raw["th"] == 30.0                    # scanned motor at frame 2 merged in


def test_store_path_only_identity_reconciles_with_concrete_record_index():
    """IFR-2: a path-only store identity (`/f.h5#`) reconciles with the record's
    concrete index (`/f.h5#0`); conflict only on a different path or two
    different CONCRETE indices."""
    store = FrameRecordStore()
    store.upsert(FrameRecord.from_view(_v1d(0, meta={"th": 1.0}, source=("/f.h5", 0))),
                 source_identity="/f.h5#")                   # path-only stamp
    proj = project_frame(store, 0)
    assert proj.capabilities.metadata.state is not CapabilityState.ERROR
    assert proj.capabilities.identity == "/f.h5#0"           # more-specific reported
    assert proj.metadata.raw["th"] == 1.0

    # two different concrete indices → genuine conflict
    store.upsert(FrameRecord.from_view(_v1d(1, meta={"th": 1.0}, source=("/f.h5", 0))),
                 source_identity="/f.h5#5")
    assert project_frame(store, 1).capabilities.metadata.state is CapabilityState.ERROR

    # different path → genuine conflict
    store.upsert(FrameRecord.from_view(_v1d(2, meta={"th": 1.0}, source=("/f.h5", 0))),
                 source_identity="/other.h5#0")
    assert project_frame(store, 2).capabilities.metadata.state is CapabilityState.ERROR


def test_persisted_but_no_hydrator_is_distinct_from_dropped():
    """IFR-3: a persisted mode with no hydrator registered is distinguished from
    a consciously-dropped mode in the capability reason."""
    resident = _record_1d(9, meta={"th": 1.0})
    thinner = FrameRecordStore(max_heavy_items=0, require_persisted_for_eviction=False)
    thinner.upsert(resident)
    thinned = thinner.get(9)
    store = FrameRecordStore()                               # NO hydrator registered
    store.upsert(thinned, persisted=True)                   # persisted on disk
    assert set(store.persisted_modes(9)) == {("1d", "default")}
    assert set(store.hydratable_modes(9)) == set()

    cap = project_frame(store, 9).capabilities.integrated_1d
    assert cap.state is CapabilityState.UNAVAILABLE
    # R3-P2: the TYPED disposition is the policy key; the prose reason is
    # asserted only as a diagnostic.
    assert cap.disposition is CapabilityDisposition.PERSISTED_NO_HYDRATOR
    assert "no hydrator" in cap.reason.lower()               # distinct reason
    assert "dropped" not in cap.reason.lower()               # NOT mislabeled dropped

    # a genuinely dropped mode is still reported as dropped
    r2 = SimpleNamespace(
        radial=np.linspace(1.0, 5.0, 6), azimuthal=np.linspace(-180, 180, 4),
        intensity=np.arange(24.0).reshape(6, 4), unit="q_A^-1",
        azimuthal_unit="deg", sigma=None)
    rec = FrameRecord.from_view(
        FrameView.from_results(label=10, result_2d=r2, metadata_raw={"th": 1.0}))
    dstore = FrameRecordStore()
    dstore.upsert(rec)
    dstore.mark_dropped(10, modes=("2d", "default"))
    dcap = project_frame(dstore, 10).capabilities.integrated_2d
    assert dcap.state is CapabilityState.UNAVAILABLE
    assert dcap.disposition is CapabilityDisposition.DROPPED
    assert "dropped" in dcap.reason.lower()


# ── R3-P2: typed disposition contract (exact members, fixed state mapping) ────

def test_capability_disposition_members_and_state_mapping():
    """The eight exact members and the FIXED disposition→state mapping from the
    Slice-3 contract.  Downstream policy is keyed by ``disposition``; ``state``
    is the validated presentation category."""
    expected = {
        CapabilityDisposition.RESIDENT: CapabilityState.AVAILABLE,
        CapabilityDisposition.HYDRATABLE: CapabilityState.PENDING,
        CapabilityDisposition.SOURCE_FALLBACK: CapabilityState.PENDING,
        CapabilityDisposition.BEST_EFFORT: CapabilityState.PENDING,
        CapabilityDisposition.PERSISTED_NO_HYDRATOR: CapabilityState.UNAVAILABLE,
        CapabilityDisposition.DROPPED: CapabilityState.UNAVAILABLE,
        CapabilityDisposition.ABSENT: CapabilityState.UNAVAILABLE,
        CapabilityDisposition.ERROR: CapabilityState.ERROR,
    }
    assert set(expected) == set(CapabilityDisposition)       # exactly eight
    for disposition, state in expected.items():
        cap = Capability(state, "any diagnostic prose", disposition=disposition)
        assert cap.state is state and cap.disposition is disposition


def test_capability_rejects_inconsistent_or_missing_disposition():
    with pytest.raises(ValueError):
        Capability(CapabilityState.AVAILABLE, "x",
                   disposition=CapabilityDisposition.DROPPED)
    with pytest.raises(ValueError):
        Capability(CapabilityState.ERROR, "x",
                   disposition=CapabilityDisposition.RESIDENT)
    with pytest.raises(TypeError):
        Capability(CapabilityState.AVAILABLE, "x")           # disposition required


def test_capability_reason_is_diagnostic_only_not_identity():
    """Rewording the reason changes neither the typed pair nor policy-relevant
    equality inputs (R3-P2: prose can never drive behavior)."""
    a = Capability(CapabilityState.UNAVAILABLE, "thinned and not persisted",
                   disposition=CapabilityDisposition.DROPPED)
    b = Capability(CapabilityState.UNAVAILABLE, "completely different words",
                   disposition=CapabilityDisposition.DROPPED)
    assert (a.state, a.disposition) == (b.state, b.disposition)
