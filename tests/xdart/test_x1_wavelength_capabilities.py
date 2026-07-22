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

        widget._exit_run_state()
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

