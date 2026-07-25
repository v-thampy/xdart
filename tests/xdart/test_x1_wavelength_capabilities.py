# -*- coding: utf-8 -*-
"""X1 GUI-adoption Slice 3 — wavelength authority (3a) + display capabilities (3b).

Production-wired: the real ``displayFrameWidget``/``DisplayDataMixin`` tiers,
real ``PublicationStore``/``FrameRecordStore`` records, and the real
scan-qualified projection pin.  Slice-3a contracts locked here:

* the per-selection GUI-thread HDF5 wavelength tier is DELETED — the standing
  invariant is "no per-selection/render GUI-thread open" (R3-P5);
* the run lifecycle (begin/pause/resume/finish) owns the scan-qualified run
  wavelength cache; pause PRESERVES it; final exit stamps the CAPTURED run
  scan and clears in ``finally`` (R3-P5/P6);
* the pinned projection's canonical-metres evidence serves ONLY opted-in
  current-frame contexts, cross-checked against the persisted value, with
  conflict warnings deduplicated by conflict identity — never by render
  generation (R3-P7).
"""

from __future__ import annotations

import logging
import os
import threading
from types import MethodType, SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets

from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
from xdart.modules.ewald.frame import LiveFrame
from xdart.modules.frame_publication import (
    publication_from_frame_view,
    publication_from_live_frame,
    publication_from_nexus_frame,
)
from xrd_tools.core import FrameView, IntegrationResult1D
from xrd_tools.session import FrameRecordStore, WavelengthStatus, project_frame


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _r1d(scale=1.0):
    radial = np.linspace(0.5, 3.5, 4)
    intensity = scale * np.array([2.0, 4.0, 8.0, 16.0])
    return IntegrationResult1D(
        radial=radial, intensity=intensity, sigma=np.sqrt(intensity),
        unit="q_A^-1")


def _view(label, *, meta=None, source="/data/loaded.nxs"):
    return FrameView.from_results(
        label=label, result_1d=_r1d(), metadata_raw=dict(meta or {}),
        source_path=source, source_frame_index=label)


def _make_widget(monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    return staticWidget()


def _wl_host(*, run_writing=False, scan_name="run_a", mg_args=None):
    """Duck host driving the REAL DisplayDataMixin wavelength tiers."""
    host = SimpleNamespace(
        scan=SimpleNamespace(
            name=scan_name, data_file=None,
            mg_args={"wavelength": 1.0e-10} if mg_args is None else mg_args,
            _persisted_wavelength_m=None),
        _run_writing=run_writing,
    )
    host._get_wavelength = MethodType(DisplayDataMixin._get_wavelength, host)
    return host


def _frame(wl):
    return SimpleNamespace(integrator=SimpleNamespace(wavelength=wl), poni=None)


def _count_h5py_opens(monkeypatch):
    """Spy on h5py.File recording (path, thread) per open."""
    import h5py
    real_file = h5py.File
    opened = []

    def counting_file(*args, **kwargs):
        opened.append((os.fspath(args[0]), threading.current_thread()))
        return real_file(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", counting_file)
    return opened


def _wavelength_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno >= logging.WARNING and "wavelength" in r.getMessage().lower()]


# --------------------------------------------------------------------------- #
# 3a0 shim: xdart.modules.wavelength re-exports the headless owner
# --------------------------------------------------------------------------- #

def test_wavelength_shim_reexports_headless_owner():
    import xdart.modules.wavelength as shim
    import xrd_tools.core.energy as owner
    for name in ("DEFAULT_WAVELENGTH_SENTINEL_M",
                 "is_default_wavelength_sentinel_m",
                 "normalize_wavelength_m",
                 "wavelength_angstrom_to_m",
                 "wavelength_m_to_angstrom"):
        assert getattr(shim, name) is getattr(owner, name)


# --------------------------------------------------------------------------- #
# R3-P5: the per-selection GUI-thread HDF5 tier is DELETED (standing guard)
# --------------------------------------------------------------------------- #

def test_get_wavelength_never_opens_hdf5_on_gui_thread(tmp_path, monkeypatch):
    """Rewritten T4 pins (test_live_refresh:76-217): a data_file carrying a
    real ``wavelength_A`` stamp is NO LONGER read per selection on the GUI
    thread.  The composed value for such a scan comes from the run-end stamp
    or the ``load_from_h5``/``ensure_calibration_loaded`` captures instead."""
    import h5py
    path = tmp_path / "wl.nxs"
    with h5py.File(path, "w") as f:
        f.create_group("entry/instrument/source").create_dataset(
            "wavelength_A", data=0.7293)

    opened = _count_h5py_opens(monkeypatch)
    host = _wl_host()
    host.scan.data_file = str(path)

    for _ in range(5):
        assert host._get_wavelength(None) is None      # no silent file read
        assert host._get_wavelength(None, for_selected_frame=True) is None
    assert opened == []                                 # ZERO opens, any thread

    # the persisted capture (owned by scan load / run-end stamp) still serves
    host.scan._persisted_wavelength_m = 0.7293e-10
    assert host._get_wavelength(None) == pytest.approx(0.7293e-10)
    assert opened == []


# --------------------------------------------------------------------------- #
# R3-P5/P6: run lifecycle owns capture/stamp/clear
# --------------------------------------------------------------------------- #

def test_run_end_stamp_restores_wavelength_for_post_run_selection(monkeypatch):
    """4b: run ends → the CAPTURED run scan is stamped from the run cache, so
    a post-run evicted/hydrated selection still converts — with zero
    GUI-thread opens (the deleted tier's replacement owner)."""
    opened = _count_h5py_opens(monkeypatch)
    host = _wl_host(run_writing=True)
    scan = host.scan

    displayFrameWidget.begin_processing(host, scan, "run_a")
    assert host._run_wavelength_m is None                # prior cache reset

    # First frame-backed row stamps the scan-qualified cache (T1).
    assert host._get_wavelength(_frame(1.5e-10)) == pytest.approx(1.5e-10)
    assert host._run_wavelength_m == pytest.approx(1.5e-10)
    assert host._run_wavelength_scan_key == "run_a"

    # Mid-run hydrated row (no frame): the run cache serves (T2).
    assert host._get_wavelength(None) == pytest.approx(1.5e-10)

    # Run end: stamp A BEFORE the clear; cache cleared in finally.
    assert scan._persisted_wavelength_m is None
    displayFrameWidget.finish_processing(host, scan, "run_a")
    assert scan._persisted_wavelength_m == pytest.approx(1.5e-10)
    assert scan.mg_args["wavelength"] == pytest.approx(1.5e-10)
    assert host._run_wavelength_m is None
    assert host._run_wavelength_scan_key is None

    # Post-run selection (writer idle): persisted truth serves, no file open.
    host._run_writing = False
    assert host._get_wavelength(None) == pytest.approx(1.5e-10)
    assert opened == []


def test_pause_preserves_run_cache_and_browse_uses_browsed_scan(monkeypatch):
    """13 (R3-P6): pause preserves run-A's scan-qualified cache; paused
    browsing of scan B converts with B's persisted value, never A's cache;
    resume serves A again; the final exit stamps A."""
    opened = _count_h5py_opens(monkeypatch)
    host = _wl_host(run_writing=True)
    run_scan = host.scan                                  # the GUI singleton

    displayFrameWidget.begin_processing(host, run_scan, "run_a")
    host._get_wavelength(_frame(1.5e-10))                 # stamp λA
    lam_a = 1.5e-10

    # -- pause: cache PRESERVED (not cleared, not stamped) ------------------- #
    host._run_writing = False                             # writer window shut
    displayFrameWidget.pause_processing(host)
    assert host._run_wavelength_m == pytest.approx(lam_a)  # preserved
    assert run_scan._persisted_wavelength_m is None        # pause didn't stamp

    # -- paused browse of scan B (same singleton, repointed) ----------------- #
    lam_b = 0.9744e-10
    run_scan.name = "scan_b"
    run_scan._persisted_wavelength_m = lam_b
    assert host._get_wavelength(None) == pytest.approx(lam_b)   # B's value

    # -- resume while B is still displayed: A's cache must NOT serve B ------- #
    displayFrameWidget.resume_processing(host, "run_a")
    host._run_writing = True
    assert host._get_wavelength(None) == pytest.approx(lam_b)   # key guard

    # -- display returns to A (frame-driven rescope) ------------------------- #
    run_scan.name = "run_a"
    run_scan._persisted_wavelength_m = None
    assert host._get_wavelength(None) == pytest.approx(lam_a)   # A's cache

    # -- final exit stamps the CAPTURED A scan, then clears ------------------ #
    host._run_writing = False
    displayFrameWidget.finish_processing(host, run_scan, "run_a")
    assert run_scan._persisted_wavelength_m == pytest.approx(lam_a)
    assert host._run_wavelength_m is None
    assert opened == []


def test_finish_processing_fails_closed_when_captured_scan_repointed():
    """Stop while paused-browsing B: the captured singleton now DESCRIBES B, so
    stamping A's λ onto it would poison B — the scan-key guard skips the stamp
    but still clears the cache in ``finally``."""
    host = _wl_host(run_writing=True)
    scan = host.scan
    displayFrameWidget.begin_processing(host, scan, "run_a")
    host._get_wavelength(_frame(1.5e-10))                 # cache bound to A

    scan.name = "scan_b"                                  # browse repointed it
    displayFrameWidget.finish_processing(host, scan, "scan_b")
    assert scan._persisted_wavelength_m is None           # no cross-scan stamp
    assert host._run_wavelength_m is None                 # still cleared


def test_run_lifecycle_wired_through_static_widget(qapp, monkeypatch, tmp_path):
    """The static-widget run lifecycle (the SOLE approved owner, R3-P5) calls
    begin/pause/resume/finish — not ``set_processing_active`` — for the
    wavelength capture, with the run scan + canonical key."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        calls = []
        display = widget.displayframe
        monkeypatch.setattr(display, "begin_processing",
                            lambda scan, key: calls.append(("begin", scan, key)))
        monkeypatch.setattr(display, "pause_processing",
                            lambda: calls.append(("pause",)))
        monkeypatch.setattr(display, "resume_processing",
                            lambda key: calls.append(("resume", key)))
        monkeypatch.setattr(display, "finish_processing",
                            lambda scan, key: calls.append(("finish", scan, key)))

        widget.scan.name = "run_scan"
        widget._enter_run_state()
        assert calls and calls[-1][0] == "begin"
        assert calls[-1][1] is widget.scan and calls[-1][2] == "run_scan"

        widget._on_run_paused()
        assert calls[-1] == ("pause",)
        widget._on_run_resuming()
        assert calls[-1] == ("resume", "run_scan")

        widget._exit_run_state(widget._new_projection_receipt())
        assert calls[-1][0] == "finish"
        assert calls[-1][1] is widget.scan and calls[-1][2] == "run_scan"
    finally:
        widget.close()
        widget.deleteLater()


def test_set_processing_active_no_longer_clears_the_run_cache():
    """R3-P6 fail-before: the pause path (``set_processing_active(False)``)
    used to erase the run cache; the lifecycle owns clearing now."""
    host = _wl_host(run_writing=True)
    displayFrameWidget.begin_processing(host, host.scan, "run_a")
    host._get_wavelength(_frame(1.5e-10))
    host._processing_active = True
    host._aggregate_live_scan = None
    host._wf_last_draw_t = 0.0
    MethodType(displayFrameWidget.set_processing_active, host)(False)
    assert host._run_wavelength_m == pytest.approx(1.5e-10)   # preserved


# --------------------------------------------------------------------------- #
# T3' evidence tier: selected-frame contexts only, cross-checked, deduped
# --------------------------------------------------------------------------- #

def _pin_widget(widget, *, meta, persisted, frame=0):
    display = widget.displayframe
    display.scan.name = "loaded"
    display.publication_store.upsert(
        publication_from_frame_view(_view(frame, meta=meta)))
    display.scan._persisted_wavelength_m = persisted
    display.frame_ids = (frame,)
    display.update()                       # pins the scan-qualified projection
    return display


def test_cross_check_conflict_fails_local_conversion_and_warns_once(
        qapp, monkeypatch, tmp_path, caplog):
    """4c (R3-P7): explicit per-frame evidence disagreeing with the persisted
    run wavelength → the LOCAL conversion fails (None → native axis) and ONE
    actionable warning names the source identity and both roles."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = _pin_widget(
            widget, meta={"wavelength_m": 1.6e-10}, persisted=1.0e-10)
        with caplog.at_level(logging.WARNING):
            for _ in range(4):                     # burst: same conflict
                assert display._get_wavelength(
                    None, for_selected_frame=True) is None
        warnings = _wavelength_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "loaded.nxs" in message             # capabilities.identity path
        assert "persisted" in message.lower()      # the cross-check role
        assert "1.6e-10" in message or "1.6000e-10" in message.lower() \
            or "1.6" in message                    # canonical values named

        # Per-row sites (NO opt-in) keep today's scan-constant persisted value
        # — the pinned frame's evidence never leaks into row math (S2-R3).
        assert display._get_wavelength(None) == pytest.approx(1.0e-10)
    finally:
        widget.close()
        widget.deleteLater()


def test_evidence_agreement_and_no_persisted_paths(qapp, monkeypatch, tmp_path,
                                                   caplog):
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        # agreement (within rtol 1e-3): the persisted value is returned, silent
        display = _pin_widget(
            widget, meta={"wavelength_m": 1.0e-10}, persisted=1.0e-10)
        with caplog.at_level(logging.WARNING):
            assert display._get_wavelength(
                None, for_selected_frame=True) == pytest.approx(1.0e-10)
        assert _wavelength_warnings(caplog) == []

        # present with NO persisted value → the canonical evidence serves
        display.scan._persisted_wavelength_m = None
        assert display._get_wavelength(
            None, for_selected_frame=True) == pytest.approx(1.0e-10)
    finally:
        widget.close()
        widget.deleteLater()


def test_evidence_absence_with_poni_persisted_is_silent(
        qapp, monkeypatch, tmp_path, caplog):
    """4d: structured absence — no explicit wavelength keys in the selected
    frame while the persisted/PONI authority supplies the value: silent."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = _pin_widget(
            widget, meta={"i0": 5.0}, persisted=0.9744e-10)
        with caplog.at_level(logging.WARNING):
            assert display._get_wavelength(
                None, for_selected_frame=True) == pytest.approx(0.9744e-10)
        assert _wavelength_warnings(caplog) == []
    finally:
        widget.close()
        widget.deleteLater()


def test_conflict_warning_rearms_on_resolution_and_scan_change(
        qapp, monkeypatch, tmp_path, caplog):
    """14 (R3-P7): dedup key = conflict identity (source identity + canonical
    values + role) — NEVER the render generation.  Re-arms after resolution
    and after a scan-identity change."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = _pin_widget(
            widget, meta={"wavelength_m": 1.6e-10}, persisted=1.0e-10)
        with caplog.at_level(logging.WARNING):
            # burst across many admitted renders (generation churn)
            for _ in range(5):
                display.update()
                display._get_wavelength(None, for_selected_frame=True)
            assert len(_wavelength_warnings(caplog)) == 1

            # resolution: persisted now agrees → consult clears the key
            display.scan._persisted_wavelength_m = 1.6e-10
            assert display._get_wavelength(
                None, for_selected_frame=True) == pytest.approx(1.6e-10)
            # the same conflict returning warns ONCE more
            display.scan._persisted_wavelength_m = 1.0e-10
            display._get_wavelength(None, for_selected_frame=True)
            display._get_wavelength(None, for_selected_frame=True)
            assert len(_wavelength_warnings(caplog)) == 2

            # scan change: a NEW scan's distinct conflict warns once more
            display.scan.name = "scan_b"
            display.publication_store.upsert(publication_from_frame_view(
                _view(0, meta={"wavelength_m": 1.7e-10},
                      source="/data/scan_b.nxs")))
            display.update()                       # re-pin under scan_b
            display._get_wavelength(None, for_selected_frame=True)
            display._get_wavelength(None, for_selected_frame=True)
            assert len(_wavelength_warnings(caplog)) == 3
    finally:
        widget.close()
        widget.deleteLater()


def test_evidence_tier_requires_current_scan_pin(qapp, monkeypatch, tmp_path):
    """A retained pin from scan A must not serve after the display repoints to
    scan B without a re-pin (fail closed → fall through to persisted)."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = _pin_widget(
            widget, meta={"wavelength_m": 1.6e-10}, persisted=None)
        assert display._get_wavelength(
            None, for_selected_frame=True) == pytest.approx(1.6e-10)

        display.scan.name = "scan_b"               # repointed, NOT re-pinned
        display.scan._persisted_wavelength_m = 0.9744e-10
        assert display._get_wavelength(
            None, for_selected_frame=True) == pytest.approx(0.9744e-10)
    finally:
        widget.close()
        widget.deleteLater()


# --------------------------------------------------------------------------- #
# 4e: live ≡ batch ≡ reload — evidence status/value through production paths
# --------------------------------------------------------------------------- #

def _project_publication(publication):
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(publication.record)
    return project_frame(store, publication.label)


def test_wavelength_evidence_live_batch_reload_equivalence(tmp_path):
    """4e (Standard record): the same explicit ``wavelength_m`` row yields the
    SAME evidence status/value through the live, batch, and reload publication
    constructors, and the same composed ``_get_wavelength`` value through the
    real selected-frame tier.  (Full GI live/batch/reload equivalence rides
    the real-data GI spine — ``test_gi_batch_real_data`` — which pins the
    record layer these projections read.)"""
    import h5py
    from xrd_tools.io.nexus import write_integrated_stack

    lam = 1.033e-10
    meta = {"wavelength_m": lam, "i0": 5.0}
    label = 1

    live_frame = LiveFrame(
        label, np.ones((2, 2), dtype=np.float32), scan_info=dict(meta),
        static=True, integrator=SimpleNamespace())
    live_frame.source_file = "/data/run.nxs"
    live_frame.source_frame_idx = label
    live_frame.int_1d = _r1d()
    live = _project_publication(
        publication_from_live_frame(live_frame, include_2d=False))

    batch_frame = LiveFrame(
        label, np.ones((2, 2), dtype=np.float32), scan_info=dict(meta),
        static=True, integrator=SimpleNamespace())
    batch_frame.source_file = "/data/run.nxs"
    batch_frame.source_frame_idx = label
    batch_frame.int_1d = _r1d()
    batch = _project_publication(
        publication_from_live_frame(batch_frame, include_2d=False))

    processed = tmp_path / "processed.nxs"
    with h5py.File(processed, "w") as h5:
        entry = h5.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[label], results_1d=[live_frame.int_1d])
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([label]))
        for key, value in meta.items():
            scan_data.create_dataset(key, data=np.array([float(value)]))
    reload = _project_publication(
        publication_from_nexus_frame(str(processed), label))

    for leg, projection in (("live", live), ("batch", batch),
                            ("reload", reload)):
        assert projection.wavelength.status is WavelengthStatus.PRESENT, leg
        assert projection.wavelength.value == pytest.approx(lam), leg

        host = _wl_host(scan_name="loaded")
        host._current_frame_projection = projection
        host._current_frame_projection_scan_key = "loaded"
        assert host._get_wavelength(
            None, for_selected_frame=True) == pytest.approx(lam), leg


def test_gi_shaped_record_evidence_status_consistent():
    """4e (GI shape): a GI 2D record with the explicit key projects the same
    PRESENT evidence — the GI mode does not change wavelength authority."""
    r2 = SimpleNamespace(
        radial=np.linspace(0.1, 2.0, 6), azimuthal=np.linspace(-1.0, 1.0, 4),
        intensity=np.arange(24.0).reshape(6, 4), unit="qip_A^-1",
        azimuthal_unit="qoop_A^-1", sigma=None)
    view = FrameView.from_results(
        label=0, result_2d=r2, metadata_raw={"wavelength_A": 1.033},
        source_path="/data/gi.nxs", source_frame_index=0)
    projection = _project_publication(publication_from_frame_view(view))
    assert projection.wavelength.status is WavelengthStatus.PRESENT
    assert projection.wavelength.value == pytest.approx(1.033e-10)



# =========================================================================== #
# Slice 3b — per-panel capability adoption + request-boundary eligibility
# =========================================================================== #

from xrd_tools.core import FrameRecord, IntegrationResult2D  # noqa: E402
from xrd_tools.session import Capability, CapabilityDisposition, CapabilityState  # noqa: E402
from xdart.gui.tabs.static_scan.display_logic import (  # noqa: E402
    DataTier,
    Mode,
    PanelRole,
)
from xdart.gui.tabs.static_scan.display_controllers import (  # noqa: E402
    resolve_frame_data_for_widget,
)
from xdart.gui.tabs.static_scan.frame_projection_adapter import (  # noqa: E402
    STORE_SCAN_KEY_ATTR,
)


def _record(label, *, meta=None, source="/data/loaded.nxs"):
    return FrameRecord.from_view(_view(label, meta=meta, source=source))


def _r2d(unit="q_A^-1", az_unit="deg"):
    return IntegrationResult2D(
        radial=np.linspace(0.5, 3.5, 6), azimuthal=np.linspace(-10, 10, 4),
        intensity=np.arange(24.0).reshape(6, 4), sigma=None,
        unit=unit, azimuthal_unit=az_unit)


def _view2d(label, *, meta=None, source="/data/loaded.nxs"):
    return FrameView.from_results(
        label=label, result_2d=_r2d(), metadata_raw=dict(meta or {}),
        source_path=source, source_frame_index=label)


def _scan_store(*records, scan_key="loaded", hydrator=None, persisted=False):
    store = FrameRecordStore(max_heavy_items=None)
    if hydrator is not None:
        store.set_hydrator(hydrator)
    for record in records:
        store.upsert(record, persisted=persisted)
    setattr(store, STORE_SCAN_KEY_ATTR, scan_key)
    return store


def _thinned(record):
    thinner = FrameRecordStore(
        max_heavy_items=0, require_persisted_for_eviction=False)
    thinner.upsert(record)
    return thinner.get(record.label)


def _hydration_spy(display):
    """Route the REAL ``_request_frame_hydration`` guards into a recorder at
    the worker seam (the boundary under test is UPSTREAM, in
    ``resolve_frame_data``)."""
    requests = []
    display._async_hydration_enabled = True
    worker = SimpleNamespace(
        request=lambda label, generation, **kw:
            requests.append((label, kw.get("purpose", "full"))))
    display._ensure_hydration_worker = lambda: worker
    return requests


def _select(display, *labels):
    display.frame_ids = tuple(labels)
    display.update()


def _reset_hydration_dedupe(display):
    """Clear the widget-level latest-generation dedupe guards so a direct
    request-boundary probe is not absorbed by a request the preceding
    ``update()`` legitimately issued (the guards are NOT the policy owner —
    R3-P8 — the boundary under test is upstream of them)."""
    display._hydration_pending_labels = set()
    display._hydration_success_labels = set()
    display._hydration_failure_counts = {}


def test_five_way_distinction_typed_and_request_gated(qapp, monkeypatch, tmp_path):
    """Test 1: resident / hydratable / persisted-no-hydrator / dropped /
    absent — exact typed (state, disposition) on the pin, and the ``"1d"``
    request issued or suppressed at the resolve_frame_data boundary."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        requests = _hydration_spy(display)

        resident = _record(0)
        cases = {}
        # a. resident
        cases[0] = (_scan_store(_record(0)),
                    CapabilityState.AVAILABLE, CapabilityDisposition.RESIDENT)
        # b. thinned + persisted + hydrator → hydratable
        cases[1] = (_scan_store(_thinned(_record(1)),
                                hydrator=lambda label: resident,
                                persisted=True),
                    CapabilityState.PENDING, CapabilityDisposition.HYDRATABLE)
        # c. thinned + persisted, NO hydrator
        cases[2] = (_scan_store(_thinned(_record(2)), persisted=True),
                    CapabilityState.UNAVAILABLE,
                    CapabilityDisposition.PERSISTED_NO_HYDRATOR)
        # d. consciously dropped
        dropped_store = _scan_store(_record(3))
        dropped_store.mark_dropped(3, modes=("1d", "default"))
        cases[3] = (dropped_store,
                    CapabilityState.UNAVAILABLE, CapabilityDisposition.DROPPED)

        for label, (store, state, disposition) in cases.items():
            display.frame_record_store = store
            _select(display, label)
            fact = display._current_frame_projection.capabilities.integrated_1d
            assert fact.state is state, label
            assert fact.disposition is disposition, label

            _reset_hydration_dedupe(display)
            requests.clear()
            resolve_frame_data_for_widget(
                display, label, Mode.INT_1D, DataTier.ONE_D)
            if disposition in (CapabilityDisposition.DROPPED,
                               CapabilityDisposition.PERSISTED_NO_HYDRATOR):
                assert requests == [], (label, requests)      # suppressed
            elif disposition is CapabilityDisposition.HYDRATABLE:
                assert (label, "1d") in requests, label       # still issued
            else:                                             # resident
                assert requests == [], label

        # e. absent record: an ABSENT pin does NOT suppress (record-not-present
        # is not proof recovery is pointless) — legacy request behavior.
        display.frame_record_store = _scan_store(scan_key="loaded")
        _select(display, 9)
        projection = display._current_frame_projection
        assert projection.present is False
        assert projection.capabilities.integrated_1d.disposition \
            is CapabilityDisposition.ABSENT
        _reset_hydration_dedupe(display)
        requests.clear()
        resolve_frame_data_for_widget(display, 9, Mode.INT_1D, DataTier.ONE_D)
        assert (9, "1d") in requests
    finally:
        widget.close()
        widget.deleteLater()


def test_eligibility_precedes_the_request_boundary(qapp, monkeypatch, tmp_path):
    """Test 15 (R3-P8): drive the ``_data_snapshot → resolve_frame_data`` path
    directly (no ``compute_display_state``) for a DROPPED frame — the worker
    is never reached, proving suppression sits at the request boundary."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        store = _scan_store(_record(4))
        store.mark_dropped(4, modes=("1d", "default"))
        display.frame_record_store = store
        requests = _hydration_spy(display)

        _select(display, 4)                   # pins the DROPPED projection
        _reset_hydration_dedupe(display)
        requests.clear()
        result = resolve_frame_data_for_widget(
            display, 4, Mode.INT_1D, DataTier.ONE_D)
        assert requests == []                  # never queued
        assert result.data is None             # nothing resident either

        # Identity mismatch = unchanged legacy behavior (no false suppression
        # of OTHER frames): a different label still issues its request.
        resolve_frame_data_for_widget(display, 5, Mode.INT_1D, DataTier.ONE_D)
        assert (5, "1d") in requests
    finally:
        widget.close()
        widget.deleteLater()


def test_identity_conflict_blanks_every_panel_never_stale(
        qapp, monkeypatch, tmp_path):
    """Test 3a (R3-P3): a genuine identity conflict sets every fact ERROR —
    every panel blanks explicitly (no exception), and a RESIDENT stale
    publication under the same label must not draw."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        # A resident publication (would draw) ...
        display.publication_store.upsert(
            publication_from_frame_view(_view2d(0)))
        # ... but the store's record identity contradicts the record's own.
        store = FrameRecordStore(max_heavy_items=None)
        store.upsert(_record(0), source_identity="/data/other.nxs#0")
        setattr(store, STORE_SCAN_KEY_ATTR, "loaded")
        display.frame_record_store = store

        _select(display, 0)
        caps = display._current_frame_projection.capabilities
        for fact in (caps.metadata, caps.integrated_1d, caps.integrated_2d,
                     caps.raw, caps.thumbnail):
            assert fact.state is CapabilityState.ERROR
        state = display._live_display_state()
        for role in (PanelRole.RAW_2D, PanelRole.CAKE_2D, PanelRole.PLOT_1D):
            panel = state.panel(role)
            if panel is not None:
                assert panel.has_data is False, role
    finally:
        widget.close()
        widget.deleteLater()


def test_single_fact_error_does_not_blank_other_panels(
        qapp, monkeypatch, tmp_path):
    """Test 3b (R3-P3): a metadata-only ERROR (other facts AVAILABLE — the
    S2-R7 stored/provider-disagree shape) leaves raw/cake/1D drawn — one
    fact's failure never becomes a global error.  The metadata-only ERROR
    caps are composed from the REAL projection of a resident record (a record
    whose own metadata conflicts is all-ERROR by design, so this shape only
    arises from the provider-merge path)."""
    from dataclasses import replace

    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        v2 = _view2d(0, meta={"exposure": 2.0})
        display.frame_record_store = _scan_store(FrameRecord.from_view(v2))
        display.publication_store.upsert(publication_from_frame_view(v2))

        _select(display, 0)
        real = display._current_frame_projection
        assert real.capabilities.integrated_2d.state is CapabilityState.AVAILABLE
        conflicted_meta = replace(
            real.capabilities,
            metadata=Capability(
                CapabilityState.ERROR, "stored/provider metadata disagree",
                disposition=CapabilityDisposition.ERROR))
        display._current_frame_projection = SimpleNamespace(
            label=real.label, present=True, capabilities=conflicted_meta,
            wavelength=real.wavelength,
            metadata=real.metadata,
            normalization_channels=real.normalization_channels)

        state = display._live_display_state()
        cake = state.panel(PanelRole.CAKE_2D)
        assert cake is not None
        assert cake.has_data is True                 # NOT globally blanked
        assert cake.capability is not None
        assert cake.capability.state is CapabilityState.AVAILABLE
        raw_panel = state.panel(PanelRole.RAW_2D)
        assert raw_panel is not None
        assert raw_panel.capability is not None
        assert raw_panel.capability.state is not CapabilityState.ERROR
    finally:
        widget.close()
        widget.deleteLater()


def _assert_selected_panels_blank(state):
    for role in (PanelRole.RAW_2D, PanelRole.CAKE_2D, PanelRole.PLOT_1D):
        panel = state.panel(role)
        if panel is not None:
            assert panel.has_data is False, role
    raw_panel = state.panel(PanelRole.RAW_2D)
    if raw_panel is not None:
        from xdart.gui.tabs.static_scan.display_logic import RawSource
        assert raw_panel.source is RawSource.NONE


def test_r4_stale_cross_scan_publication_pin_fails_closed(
        qapp, monkeypatch, tmp_path):
    """Test 2e RESTORED (S3-OR1 correction): a retained scan-A publication
    with an EXPLICIT owner A, requested by scan B under the reused label,
    projects ``present=False`` — and the absent scan-matching pin carries its
    ABSENT capabilities into the pure layer, so raw, cake, AND the
    selected-frame 1D all fail closed.  No legacy pixel fallback."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "scan_b"                      # browsing scan B
        display.publication_store.upsert(publication_from_frame_view(
            _view2d(0, source="/data/scan_a.nxs"),
            scan_key="scan_a"))                           # stamped A publication

        _select(display, 0)
        projection = display._current_frame_projection
        assert projection is not None and projection.present is False
        assert projection.capabilities.integrated_2d.disposition \
            is CapabilityDisposition.ABSENT
        # Projection consumers fail closed on the absent pin (Slice 2/3a).
        assert displayFrameWidget._projection_norm_channels(display) == ()
        # ... and so does the RENDER: the typed ABSENT facts blank every
        # selected-frame panel — never scan A's pixels under scan B.
        state = display._live_display_state()
        _assert_selected_panels_blank(state)
        cake = state.panel(PanelRole.CAKE_2D)
        assert cake is not None and cake.capability is not None
        assert cake.capability.disposition is CapabilityDisposition.ABSENT
    finally:
        widget.close()
        widget.deleteLater()


# --------------------------------------------------------------------------- #
# Slice 3c (S3-OR1): explicit immutable publication scan ownership
# --------------------------------------------------------------------------- #

def test_stamped_tiff_publication_projects_present_and_serves_consumers(
        qapp, monkeypatch, tmp_path):
    """Required test 1: a same-scan TIFF publication whose source_path is a
    per-frame filename (name-unprovable) but whose explicit owner matches the
    requested scan projects present=True; raw/cake/1D behavior is unchanged
    and metadata/channel consumers are populated (the Slice-2 TIFF gap
    closes)."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "tiff_scan"
        view = FrameView.from_results(
            label=0, result_1d=_r1d(), metadata_raw={"i0": 5.0, "motor_x": 7.0},
            source_path="/data/frames/frame_0000.tif", source_frame_index=0)
        display.publication_store.upsert(
            publication_from_frame_view(view, scan_key="tiff_scan"))

        _select(display, 0)
        projection = display._current_frame_projection
        assert projection is not None and projection.present is True
        assert projection.capabilities.integrated_1d.state \
            is CapabilityState.AVAILABLE
        assert projection.metadata.raw["i0"] == 5.0        # metadata populated
        assert displayFrameWidget._projection_norm_channels(display) \
            == ("i0", "motor_x")                           # channels populated
        state = display._live_display_state()
        plot = state.panel(PanelRole.PLOT_1D)
        assert plot is not None and plot.has_data is True  # normal render
    finally:
        widget.close()
        widget.deleteLater()


def test_owner_mismatch_not_rescued_by_matching_source_filename(
        qapp, monkeypatch, tmp_path):
    """Required test 3: an explicit owner mismatch is FINAL — a coincidentally
    matching source filename must not rescue it (no fallback to source-name
    inference after an explicit mismatch)."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "scan_b"
        # The source FILENAME names scan_b, but the explicit owner is scan_a.
        display.publication_store.upsert(publication_from_frame_view(
            _view2d(0, source="/data/scan_b.nxs"), scan_key="scan_a"))

        _select(display, 0)
        projection = display._current_frame_projection
        assert projection is not None and projection.present is False
        state = display._live_display_state()
        _assert_selected_panels_blank(state)
    finally:
        widget.close()
        widget.deleteLater()


def test_explicit_path_owners_do_not_collapse_across_roots():
    """Two qualified owners remain distinct even when leaf and stem match."""
    from xdart.gui.tabs.static_scan.frame_projection_adapter import (
        _publication_serves_scan,
    )

    publication = SimpleNamespace(scan_key="/root_a/sample.nxs")
    assert not _publication_serves_scan(publication, "/root_b/sample.nxs")
    assert _publication_serves_scan(publication, "/root_a/sample.nxs")

    windows_publication = SimpleNamespace(scan_key=r"C:\Data\sample.nxs")
    assert _publication_serves_scan(
        windows_publication, "c:/data/sample.nxs")

    legacy_publication = SimpleNamespace(
        scan_key=None,
        source_identity="/root_a/sample.nxs",
        view=None,
        record=None,
    )
    assert not _publication_serves_scan(
        legacy_publication, "/root_b/sample.nxs")
    assert _publication_serves_scan(legacy_publication, "sample")


def test_legacy_unstamped_provable_serves_and_unprovable_fails_closed(
        qapp, monkeypatch, tmp_path):
    """Required test 4: a legacy UNSTAMPED NeXus publication whose source
    identity positively proves the scan stays compatible; an unstamped,
    unprovable per-frame source fails closed (production sites stamp, so this
    is the legacy boundary only)."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        # (a) provable: source stem == scan name (the existing R4 rule)
        display.scan.name = "loaded"
        display.publication_store.upsert(
            publication_from_frame_view(_view(0)))         # /data/loaded.nxs
        _select(display, 0)
        assert display._current_frame_projection.present is True

        # (b) unprovable per-frame source, unstamped → fails closed
        display.scan.name = "tiff_scan"
        display.publication_store.upsert(publication_from_frame_view(
            FrameView.from_results(
                label=1, result_1d=_r1d(), metadata_raw={"i0": 1.0},
                source_path="/data/frames/frame_0001.tif",
                source_frame_index=1)))
        _select(display, 1)
        projection = display._current_frame_projection
        assert projection is not None and projection.present is False
        state = display._live_display_state()
        _assert_selected_panels_blank(state)
    finally:
        widget.close()
        widget.deleteLater()


def test_pause_browse_resume_owner_qualified_terminal_coherence(
        qapp, monkeypatch, tmp_path):
    """Required test 6: pause run A → browse B → resume A, plus rapid
    same-label selection — the terminal pin is owner-qualified each time."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "run_a"
        display.publication_store.upsert(publication_from_frame_view(
            FrameView.from_results(
                label=0, result_1d=_r1d(), metadata_raw={"i0": 111.0},
                source_path="/data/frames/frame_0000.tif",
                source_frame_index=0),
            scan_key="run_a"))                             # A-stamped TIFF frame

        _select(display, 0)
        assert display._current_frame_projection.present is True
        assert display._current_frame_projection.metadata.raw["i0"] == 111.0

        # paused browse of scan B: A's retained publication must NOT serve B
        display.scan.name = "scan_b"
        _select(display, 0)
        assert display._current_frame_projection.present is False
        _assert_selected_panels_blank(display._live_display_state())

        # resume A: A's frame serves again — owner-qualified, same label
        display.scan.name = "run_a"
        _select(display, 0)
        assert display._current_frame_projection.present is True
        assert display._current_frame_projection.metadata.raw["i0"] == 111.0

        # rapid same-label flips end owner-coherent at the terminal scan
        for terminal in ("scan_b", "run_a", "scan_b"):
            display.scan.name = terminal
            _select(display, 0)
        assert display._current_frame_projection.present is False  # scan_b
        display.scan.name = "run_a"
        _select(display, 0)
        assert display._current_frame_projection.present is True
    finally:
        widget.close()
        widget.deleteLater()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "R4-D captured live failure: the run capture aliases H5Viewer's "
        "mutable browse LiveScan"
    ),
)
def test_r4_run_capture_survives_h5viewer_browse_mutation(
        qapp, monkeypatch, tmp_path):
    """Pause/Browse/Resume must not mutate the acquisition owner in place."""
    widget = _make_widget(monkeypatch, tmp_path)
    resumed_keys = []
    try:
        widget.scan.name = "run_a"
        widget.scan.gi = False
        widget.displayframe.resume_processing = resumed_keys.append

        # Use the production run-state owner without launching a reduction
        # worker.  The production H5Viewer and display still share the same
        # LiveScan object at the captured failure point.
        widget._enter_run_state()
        captured = widget._x1_run_scan_capture
        captured_scan = getattr(captured, "scan", captured)

        # Drive the real Run-button pause state and worker-boundary signal.
        widget.wrangler.command = "start"
        widget.wrangler.thread.command = "start"
        widget.wrangler._set_action_button("running")
        widget.controls.startButton.click()
        assert widget.wrangler.thread.command == "pause"
        widget.wrangler.thread.sigPaused.emit()
        qapp.processEvents()
        assert widget._run_active is True

        # fileHandlerThread.set_datafile mutates this production owner in
        # place.  Direct field mutation isolates the ownership failure without
        # paying for either processed-file load.
        browse_scan = widget.h5viewer.scan
        browse_scan.name = "scan_b"
        browse_scan.gi = True

        widget.controls.startButton.click()
        qapp.processEvents()

        # Current captured tuple at 41292079 is
        # (False, "scan_b", True, ["scan_b"]).
        assert (
            captured_scan is not browse_scan,
            captured_scan.name,
            captured_scan.gi,
            resumed_keys,
        ) == (True, "run_a", False, ["run_a"])
    finally:
        if widget._run_active:
            widget._exit_run_state(widget._new_projection_receipt())
        widget._controls_v2_refresh_timer.cancel()
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "R4-D captured live failure: hydration round trips carry a display "
        "generation but no acquisition/browse owner"
    ),
)
def test_r4_hydration_same_generation_cross_owner_completion_is_rejected(
        qapp, monkeypatch, tmp_path):
    """Generation equality must not authorize another owner's completion."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        generation = 23
        requested = []
        repaints = []

        class RecordingWorker:
            def request(self, label, request_generation, **kwargs):
                requested.append((label, request_generation, dict(kwargs)))

        worker = RecordingWorker()
        display.scan.name = "scan_b"
        display.frame_ids = ("9",)
        display.overall = False
        display.display_generation = generation
        display._last_selection_sig = display._selection_generation_signature()
        display._async_hydration_enabled = True
        display._hydration_pending_labels = set()
        display._hydration_success_labels = set()
        display._hydration_success_generation = generation
        display._hydration_failure_counts = {}
        display._hydration_failure_logged = set()
        display._pending_hydration_render = False
        display._pending_hydration_generation = None
        display._hydration_purpose_resident = lambda *_args: False
        display._ensure_hydration_worker = lambda: worker

        # Exercise the production request builder under browse owner B.
        display._request_frame_hydration(9, purpose="full")
        assert len(requested) == 1

        # Switch to owner A while deliberately preserving the numeric
        # generation.  This prevents the existing generation guard from
        # masking the missing owner contract.
        display.scan.name = "scan_a"
        display._last_selection_sig = display._selection_generation_signature()
        display._hydration_purpose_resident = lambda *_args: True
        display._hydration_quiet_timer = None
        display.request_current_selection_repaint = (
            lambda *, generation=None, reason=None:
                repaints.append((generation, reason)) or True
        )

        # Echo the legacy completion.  A correct round trip must know this
        # request belonged to B and reject it even with an equal generation.
        display._on_frame_hydrated(9, generation)

        request_kwargs = requested[0][2]
        has_owner = (
            any(
                key in request_kwargs
                for key in ("owner", "owner_token", "request_owner")
            )
            or {"context_id", "scan_key"} <= set(request_kwargs)
        )
        # Current captured tuple at 41292079 is
        # (False, [(23, "hydration")]).
        assert (has_owner, repaints) == (True, [])
    finally:
        widget._controls_v2_refresh_timer.cancel()
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_persist_override_never_retains_unavailable_or_error(
        qapp, monkeypatch, tmp_path):
    """Test 2f: during an active run with PERSIST_2D_DURING_PROCESSING, an
    UNAVAILABLE selected frame must not retain another frame's image (forced
    clear), while a same-scan-qualified PENDING selection may keep it."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        resident2d = _view2d(0)
        display.publication_store.upsert(publication_from_frame_view(resident2d))

        store = _scan_store(FrameRecord.from_view(resident2d))
        dropped = FrameRecord.from_view(_view2d(1))
        store.upsert(dropped)
        store.mark_dropped(1, modes=("2d", "default"))
        pending = _thinned(FrameRecord.from_view(_view2d(2)))
        store.set_hydrator(lambda label: FrameRecord.from_view(_view2d(2)))
        store.upsert(pending, persisted=True)
        display.frame_record_store = store

        cleared = []
        real_clear = display.clear_binned_view
        monkeypatch.setattr(
            display, "clear_binned_view",
            lambda *a, **k: (cleared.append("cake"), real_clear()))

        display.set_processing_active(True)               # persist window on
        _select(display, 0)                               # A drawn
        cleared.clear()

        _select(display, 1)                               # DROPPED selection
        fact = display._current_frame_projection.capabilities.integrated_2d
        assert fact.disposition is CapabilityDisposition.DROPPED
        assert "cake" in cleared                          # forced clear

        cleared.clear()
        _select(display, 2)                               # PENDING selection
        fact = display._current_frame_projection.capabilities.integrated_2d
        assert fact.state is CapabilityState.PENDING
        assert cleared == []                              # persist retains
    finally:
        widget.close()
        widget.deleteLater()


def test_aggregate_probe_keeps_legacy_residency(qapp, monkeypatch, tmp_path):
    """Test 6: Sum/Average probe ≠ pin — the pinned frame's capability is never
    applied to frame [0]; the panels keep legacy residency behavior."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        for label in (0, 1):
            display.publication_store.upsert(
                publication_from_frame_view(_view2d(label)))
        display.frame_record_store = _scan_store(
            FrameRecord.from_view(_view2d(0)),
            FrameRecord.from_view(_view2d(1)))

        display.ui.plotMethod.setCurrentText("Sum")
        _select(display, 0, 1)
        # The pin is the CURRENT display frame (R1: the latest, frame 1);
        # the Sum/Average probe is render_2d[0] (frame 0) — they differ.
        assert display._current_frame_projection.label == 1
        state = display._live_display_state()
        cake = state.panel(PanelRole.CAKE_2D)
        assert cake is not None
        # Probe ≠ pin: the pinned frame's capability is never applied to
        # frame [0] — legacy residency behavior, no typed layer.
        assert cake.capability is None
        assert cake.has_data is True          # legacy residency draw
    finally:
        widget.close()
        widget.deleteLater()


def test_rapid_selection_terminal_generation_owns_state(
        qapp, monkeypatch, tmp_path):
    """Test 7: rapid A→B→C ending on an UNAVAILABLE frame — the terminal
    generation's typed state stands, and a stale hydration completion for A
    cannot change it."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        for label in (0, 1):
            display.publication_store.upsert(
                publication_from_frame_view(_view2d(label)))
        store = _scan_store(FrameRecord.from_view(_view2d(0)),
                            FrameRecord.from_view(_view2d(1)))
        dropped = FrameRecord.from_view(_view2d(2))
        store.upsert(dropped)
        store.mark_dropped(2, modes=("2d", "default"))
        display.frame_record_store = store

        _select(display, 0)
        _select(display, 1)
        _select(display, 2)                                # terminal: dropped
        stale_generation = display.display_generation - 1

        fact = display._current_frame_projection.capabilities.integrated_2d
        assert (fact.state, fact.disposition) == (
            CapabilityState.UNAVAILABLE, CapabilityDisposition.DROPPED)
        state = display._live_display_state()
        cake = state.panel(PanelRole.CAKE_2D)
        assert cake is not None and cake.has_data is False

        # A deliberately-delayed stale completion for A (old generation) must
        # change nothing.
        display._on_frame_hydrated(0, stale_generation)
        after = display._current_frame_projection.capabilities.integrated_2d
        assert (after.state, after.disposition) == (
            CapabilityState.UNAVAILABLE, CapabilityDisposition.DROPPED)
        state2 = display._live_display_state()
        cake2 = state2.panel(PanelRole.CAKE_2D)
        assert cake2 is not None and cake2.has_data is False
    finally:
        widget.close()
        widget.deleteLater()


def test_gi_mode_flip_carries_fresh_mode_capability(qapp, monkeypatch, tmp_path):
    """Test 8: flipping the displayed GI mode re-projects (R3) and the panel
    decision follows the fresh mode's (UNAVAILABLE, ABSENT)."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        display.scan.gi = True
        display.scan.bai_1d_args = {"gi_mode_1d": "q_total"}
        display.scan.bai_2d_args = {"gi_mode_2d": "qip_qoop"}

        gi_r2 = IntegrationResult2D(
            radial=np.linspace(-1.0, 1.0, 6), azimuthal=np.linspace(0, 1, 4),
            intensity=np.arange(24.0).reshape(6, 4), sigma=None,
            unit="qip_A^-1", azimuthal_unit="qoop_A^-1")
        gi_view = FrameView.from_results(
            label=0, result_2d=gi_r2, metadata_raw={"i0": 1.0},
            source_path="/data/loaded.nxs", source_frame_index=0)
        record = FrameRecord(
            label=0, results_2d={"qip_qoop": gi_view},
            active_mode_2d="qip_qoop")
        display.frame_record_store = _scan_store(record)

        _select(display, 0)
        fact = display._current_frame_projection.capabilities.integrated_2d
        assert fact.state is CapabilityState.AVAILABLE
        before = display._frame_projection_adapter.lookup_count

        display.scan.bai_2d_args = {"gi_mode_2d": "chi_tth"}   # mode flip
        display.update()
        assert display._frame_projection_adapter.lookup_count == before + 1
        fact = display._current_frame_projection.capabilities.integrated_2d
        assert (fact.state, fact.disposition) == (
            CapabilityState.UNAVAILABLE, CapabilityDisposition.ABSENT)
        state = display._live_display_state()
        cake = state.panel(PanelRole.CAKE_2D)
        if cake is not None:
            assert cake.has_data is False
    finally:
        widget.close()
        widget.deleteLater()


def test_raw_recovery_survives_the_gate(qapp, monkeypatch, tmp_path):
    """Test 9: raw/thumbnail UNAVAILABLE is NOT proof hydration is pointless
    (findings 8-9) — the "full" purpose stays ungated; browse best-effort
    integrated PENDING still issues its "1d" request."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        requests = _hydration_spy(display)

        # (a) raw facts ABSENT (no raw, no source) → "full" still issued.
        no_source = FrameRecord.from_view(FrameView.from_results(
            label=0, result_1d=_r1d(), metadata_raw={"i0": 1.0}))
        display.frame_record_store = _scan_store(no_source)
        _select(display, 0)
        caps = display._current_frame_projection.capabilities
        assert caps.raw.disposition is CapabilityDisposition.ABSENT
        _reset_hydration_dedupe(display)
        requests.clear()
        resolve_frame_data_for_widget(
            display, 0, Mode.INT_2D, DataTier.RAW_OR_THUMBNAIL)
        assert (0, "full") in requests                    # ungated

        # (b) browse best-effort PENDING (no store mode-context — the real
        # publication-backed browse shape) keeps browse "1d" hydration alive.
        from xrd_tools.session import display_capabilities

        display.frame_record_store = None
        thinned = _thinned(_record(1))
        caps = display_capabilities(thinned)      # no store context (browse)
        assert caps.integrated_1d.disposition \
            is CapabilityDisposition.BEST_EFFORT
        display._current_frame_projection = SimpleNamespace(
            label=1, present=True, capabilities=caps)
        display._current_frame_projection_scan_key = "loaded"
        _reset_hydration_dedupe(display)
        requests.clear()
        resolve_frame_data_for_widget(display, 1, Mode.INT_1D, DataTier.ONE_D)
        assert (1, "1d") in requests                      # not suppressed
    finally:
        widget.close()
        widget.deleteLater()


def test_caps_vs_active_view_divergence_is_pending_equivalent(
        qapp, monkeypatch, tmp_path):
    """Test 10 (finding 7): capabilities aggregate ANY view, the draw decision
    stays with the ACTIVE view — a divergence renders pending-equivalent: no
    draw from a fact the active view cannot serve, typed truth carried,
    recovery allowed."""
    from xrd_tools.session import display_capabilities

    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        # Active 2D view (mode a) has no raw; mode b's view carries raw — the
        # multi-mode GI shape.  The capability derivation is the REAL headless
        # one; the store strips raw payloads on upsert, so the divergence pin
        # is injected value-only at the consumer seam.
        va = _view2d(0)
        vb = FrameView.from_results(
            label=0, result_2d=_r2d(), metadata_raw={},
            raw=np.ones((4, 4), dtype=np.float32),
            source_path="/data/loaded.nxs", source_frame_index=0)
        record = FrameRecord(
            label=0, results_2d={"a": va, "b": vb}, active_mode_2d="a")
        caps = display_capabilities(record, mode_2d="a")
        assert caps.raw.state is CapabilityState.AVAILABLE   # any-view truth

        display.publication_store.upsert(publication_from_frame_view(va))
        _select(display, 0)                       # active-view booleans: no raw
        display._current_frame_projection = SimpleNamespace(
            label=0, present=True, capabilities=caps)
        display._current_frame_projection_scan_key = "loaded"

        state = display._live_display_state()
        raw_panel = state.panel(PanelRole.RAW_2D)
        assert raw_panel is not None
        assert raw_panel.has_data is False        # active view can't serve it
        assert raw_panel.capability is not None   # typed truth carried
        assert raw_panel.capability.state is CapabilityState.AVAILABLE
    finally:
        widget.close()
        widget.deleteLater()


@pytest.mark.parametrize("method", ["Overlay", "Waterfall"])
def test_overlay_waterfall_history_survives_unavailable_selection(
        qapp, monkeypatch, tmp_path, method):
    """Test 12 (R3-P4): after accumulating rows, selecting a DROPPED (and a
    PERSISTED_NO_HYDRATOR) row preserves the accumulated payload
    byte/identity-unchanged, appends nothing, and issues NO 1d hydration."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        requests = _hydration_spy(display)
        display.ui.plotMethod.setCurrentText(method)
        for label in (0, 1):
            display.publication_store.upsert(
                publication_from_frame_view(_view(label)))
            _select(display, label)
        history = display._waterfall_history
        assert history is not None and history.count == 2
        rows_before = [np.asarray(row).tobytes() for row in history.rows]

        store = _scan_store(_record(2))
        store.mark_dropped(2, modes=("1d", "default"))
        thinned3 = _thinned(_record(3))
        store.upsert(thinned3, persisted=True)     # persisted, NO hydrator
        display.frame_record_store = store

        for label, disposition in (
                (2, CapabilityDisposition.DROPPED),
                (3, CapabilityDisposition.PERSISTED_NO_HYDRATOR)):
            requests.clear()
            _select(display, label)
            fact = display._current_frame_projection.capabilities.integrated_1d
            assert fact.disposition is disposition, label
            history = display._waterfall_history
            assert history.count == 2                       # nothing appended
            assert [np.asarray(row).tobytes() for row in history.rows] \
                == rows_before                              # byte-identical
            assert (label, "1d") not in requests            # no futile 1d
            state = display._live_display_state()
            plot = state.panel(PanelRole.PLOT_1D)
            assert plot is not None and plot.has_data is True   # history kept

        # Single mode with the same unavailable selection blanks.
        display.ui.plotMethod.setCurrentText("Single")
        _select(display, 2)
        state = display._live_display_state()
        plot = state.panel(PanelRole.PLOT_1D)
        assert plot is not None and plot.has_data is False
    finally:
        widget.close()
        widget.deleteLater()
