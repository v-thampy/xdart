# -*- coding: utf-8 -*-
"""H18 responsiveness / I/O contract for the delegated Controls V2 readiness.

Merge-blocker contract (handoff §6): repeated ``_controls_v2_state()`` calls
for an UNCHANGED source or processed scan perform no repeated HDF5 master
opens; cache identity is SOURCE identity (R1 path + version stamp + adapter
owner), invalidated on any of them changing; true-live refresh performs no
file I/O; an active run (including Pause) is never probed synchronously —
the cached snapshot answers, without flapping.

Production-wired: real ``staticWidget`` refreshes over real temporary HDF5
files; monkeypatching only OBSERVES opens (``h5py.File.__init__`` counter,
``count_frames`` tripwire) — it never replaces the readiness/probe path with
a fake answer.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph import Qt

QtWidgets = Qt.QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _controls_panel_session_isolation():
    path = os.environ.get("XDART_SESSION_FILE")

    def _unlink():
        if not path:
            return
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass

    _unlink()
    yield
    _unlink()


@pytest.fixture(autouse=True)
def _drain_qt_events_after_test(qapp):
    yield
    for _ in range(3):
        qapp.processEvents()
    gc.collect()
    for _ in range(2):
        qapp.processEvents()


@pytest.fixture()
def widget(qapp):
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    w = staticWidget()
    try:
        yield w
    finally:
        w.close()
        w.deleteLater()


def _eiger_master(tmp_path, n=2):
    master = tmp_path / "scan_master.h5"
    raw = np.arange(n * 8 * 8, dtype=np.uint32).reshape(n, 8, 8)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw)
    return master


def _configure_master(w, master):
    w._controls_v2_param(("Signal", "inp_type")).setValue("Image Series")
    w._controls_v2_param(("Signal", "File")).setValue(str(master))
    w._controls_v2_param(("Signal", "img_ext")).setValue("h5")


def _arm_open_counter(monkeypatch):
    opens: list[str] = []
    orig = h5py.File.__init__

    def counting(inst, name, *args, **kwargs):
        try:
            opens.append(os.path.abspath(str(name)))
        except Exception:
            pass
        return orig(inst, name, *args, **kwargs)

    monkeypatch.setattr(h5py.File, "__init__", counting)
    return opens


def _forbid_count_frames(monkeypatch):
    from xrd_tools.io import image as image_io

    monkeypatch.setattr(
        image_io, "count_frames",
        lambda p: pytest.fail(f"count_frames({p!r}) called on a cached refresh"))


def test_unchanged_refresh_performs_no_new_master_opens(
        widget, tmp_path, monkeypatch):
    master = _eiger_master(tmp_path)
    # arm BEFORE configuration so the whole first-probe cost (the param
    # setValue signals drive profile refreshes) is measured, then require
    # ZERO further opens once the source is unchanged — the contract governs
    # repeated refreshes, not the one-time configuration cascade.
    opens = _arm_open_counter(monkeypatch)
    target = os.path.abspath(str(master))
    t0 = time.perf_counter()
    _configure_master(widget, master)
    first = widget._controls_v2_state()
    t_first = time.perf_counter() - t0
    first_opens = opens.count(target)
    assert first.source_caps.has_frames is True
    assert first.source_caps.raw_reachable is True
    assert 1 <= first_opens <= 30, \
        f"first configure+probe must stay bounded, saw {first_opens} opens"

    _forbid_count_frames(monkeypatch)
    t0 = time.perf_counter()
    for _ in range(5):
        state = widget._controls_v2_state()
    t_cached = (time.perf_counter() - t0) / 5
    assert opens.count(target) == first_opens, \
        "unchanged refresh reopened the master (I/O contract merge blocker)"
    assert state.source_caps.has_frames is True
    print(f"\n[H18-timing] configure+first state: {t_first * 1e3:.1f} ms "
          f"({first_opens} opens); cached refresh: {t_cached * 1e3:.2f} ms "
          f"(0 opens)")


def test_refresh_reprobes_after_version_stamp_change(
        widget, tmp_path, monkeypatch):
    master = _eiger_master(tmp_path)
    _configure_master(widget, master)
    opens = _arm_open_counter(monkeypatch)
    target = os.path.abspath(str(master))

    widget._controls_v2_state()
    baseline = opens.count(target)
    widget._controls_v2_state()
    assert opens.count(target) == baseline

    _eiger_master(tmp_path, n=3)          # rewrite: size + mtime change
    state = widget._controls_v2_state()
    assert opens.count(target) > baseline, \
        "version-stamp change must invalidate the readiness cache"
    assert state.source_caps.has_frames is True


def test_adapter_owner_change_invalidates_cache(widget, tmp_path, monkeypatch):
    import xrd_tools.sources.adapters as am
    from xrd_tools.core.scan import SourceKind
    from xrd_tools.sources.adapters import SourceFormatAdapter, register_adapter

    master = _eiger_master(tmp_path)
    _configure_master(widget, master)
    opens = _arm_open_counter(monkeypatch)
    target = os.path.abspath(str(master))

    widget._controls_v2_state()
    baseline = opens.count(target)
    widget._controls_v2_state()
    assert opens.count(target) == baseline

    saved = dict(am._ADAPTERS)  # noqa: SLF001
    try:
        register_adapter(SourceFormatAdapter(
            id="h18_usurper", kinds=(SourceKind.NEXUS_STACK,),
            is_candidate=lambda p: str(p).endswith("_master.h5"),
            scan_name=lambda p: p.stem, probe=lambda p: None,
            open=lambda s: None), builtin=False)   # external outranks built-in
        widget._controls_v2_state()
        assert opens.count(target) > baseline, \
            "adapter-owner change must invalidate the readiness cache (rule 3)"
    finally:
        am._ADAPTERS.clear()  # noqa: SLF001
        am._ADAPTERS.update(saved)  # noqa: SLF001


def test_true_live_refresh_performs_no_file_io(widget, tmp_path, monkeypatch):
    master = _eiger_master(tmp_path)
    _configure_master(widget, master)
    widget.wrangler.ui.liveCheckBox.setChecked(True)
    opens = _arm_open_counter(monkeypatch)
    _forbid_count_frames(monkeypatch)
    target = os.path.abspath(str(master))

    for _ in range(3):
        state = widget._controls_v2_state()
    assert opens.count(target) == 0, \
        "true-live readiness refresh must perform no file I/O (rule 4)"
    assert state.source_caps.has_frames is True     # live escape hatch
    assert state.source_caps.raw_reachable is True


def test_active_run_refresh_reuses_snapshot_without_probe(
        widget, tmp_path, monkeypatch):
    master = _eiger_master(tmp_path)
    opens = _arm_open_counter(monkeypatch)
    target = os.path.abspath(str(master))
    _configure_master(widget, master)

    before = widget._controls_v2_state()            # populate the cache
    baseline = opens.count(target)
    assert before.source_caps.has_frames is True

    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
    _eiger_master(tmp_path, n=4)                    # stamp changes mid-run
    baseline = opens.count(target)                  # the rewrite itself opened
    for _ in range(3):
        state = widget._controls_v2_state()
    assert opens.count(target) == baseline, \
        "an active run must never be probed synchronously (rule 5)"
    # no launcher flap: the cached capability snapshot still answers
    assert state.source_caps.has_frames is True
    assert state.source_caps.raw_reachable is True

    monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: False)
    widget._controls_v2_state()
    assert opens.count(target) > baseline, \
        "after the run, the stamp change must be observed again"


def test_processed_record_caps_read_metadata_once(widget, tmp_path, monkeypatch):
    from tests.xdart.test_h5_readiness_parity import _processed_nxs

    nxs = _processed_nxs(tmp_path, raw_reachable=True)
    opens = _arm_open_counter(monkeypatch)
    target = os.path.abspath(str(nxs))

    caps1 = widget._controls_v2_loaded_result_caps(str(nxs))
    first_opens = opens.count(target)
    assert caps1 is not None and caps1.has_1d is True
    assert first_opens >= 1

    caps2 = widget._controls_v2_loaded_result_caps(str(nxs))
    assert opens.count(target) == first_opens, \
        "unchanged processed record must not be reopened per refresh"
    assert caps2 == caps1


def test_readiness_probing_is_quiet_about_expected_energy_absence(
        widget, tmp_path, monkeypatch, caplog):
    """H18-R7: capability observations of energy-less containers must not
    flood the operator with generic 'Energy not found'/'Wavelength not
    derivable' WARNINGs (missing energy is expected — the run's energy
    authority is the selected PONI); opens stay bounded across the configure
    cascade, and readiness itself is unchanged."""
    import logging

    master = _eiger_master(tmp_path)                 # no energy fields
    opens = _arm_open_counter(monkeypatch)
    target = os.path.abspath(str(master))

    with caplog.at_level(logging.DEBUG):
        _configure_master(widget, master)
        state = widget._controls_v2_state()
        # the processed-result probe leg: an energy-less processed record
        from tests.xdart.test_h5_readiness_parity import _processed_nxs
        processed = _processed_nxs(tmp_path, raw_reachable=True)
        widget.scan.data_file = str(processed)
        widget._controls_v2_state()

    flood = [r for r in caplog.records
             if r.levelno >= logging.WARNING
             and ("Energy not found" in r.getMessage()
                  or "Wavelength not derivable" in r.getMessage())]
    assert not flood, \
        f"expected-absence must be quiet, saw {len(flood)} WARNING(s)"
    # the ONE structured observation line names the exact path + subject
    contextual = [r for r in caplog.records
                  if "readiness observation for" in r.getMessage()]
    assert any(target in r.getMessage()
               and "configured acquisition source" in r.getMessage()
               for r in contextual)
    assert state.source_caps.has_frames is True      # readiness unchanged
    assert 1 <= opens.count(target) <= 30            # bounded cascade


def test_observation_suppression_is_thread_scoped(caplog):
    """H18-R11: while one thread holds a readiness observation open, an
    expected-absence WARNING emitted by an UNRELATED thread must pass
    through unchanged (and never be attributed to the observing path);
    nested observations on the observing thread stay suppressed."""
    import logging
    import threading

    from xrd_tools.sources.readiness import quiet_capability_observation

    reader = logging.getLogger("xrd_tools.io.nexus")
    held = threading.Event()
    release = threading.Event()

    def observer():
        with quiet_capability_observation("/A/master.h5", "configured acquisition source"):
            reader.warning("Energy not found in NeXus file; using NaN")  # A's own: suppressed
            with quiet_capability_observation("/A/nested.h5", "configured acquisition source"):
                reader.warning("Wavelength not derivable; using NaN")    # nested: suppressed
            held.set()
            release.wait(5.0)

    thread = threading.Thread(target=observer, daemon=True)
    with caplog.at_level(logging.DEBUG):
        thread.start()
        assert held.wait(5.0)
        # UNRELATED thread (the test main thread) emits the same warning while
        # A's observation is still open — it must SURVIVE.
        reader.warning("Energy not found in NeXus file; using NaN")
        release.set()
        thread.join(5.0)

    survived = [r for r in caplog.records
                if r.levelno == logging.WARNING
                and "Energy not found" in r.getMessage()]
    assert len(survived) == 1, \
        "the unrelated thread's warning must pass through (H18-R11)"
    # the observing thread's own absences were suppressed and contextualized
    contextual = [r for r in caplog.records
                  if "readiness observation for" in r.getMessage()]
    assert any("/A/master.h5" in r.getMessage() for r in contextual)
    assert any("/A/nested.h5" in r.getMessage() for r in contextual)


def test_transient_source_observation_is_retried_not_cached(
        widget, tmp_path, monkeypatch):
    """A temporary sharing denial may debounce, but it must never strand an
    unchanged source at all-False readiness."""
    from importlib import import_module

    from xrd_tools.session.readiness import SourceCaps
    from xrd_tools.sources.readiness import SourceReadinessObservation

    module = import_module(
        "xdart.gui.tabs.static_scan.static_scan_widget")
    source = _eiger_master(tmp_path)
    calls = {"count": 0}

    def once_transient(value):
        calls["count"] += 1
        if calls["count"] == 1:
            return SourceReadinessObservation(
                SourceCaps(), False, "sharing violation")
        return SourceReadinessObservation(
            SourceCaps(has_frames=True, has_raw=True, raw_reachable=True),
            True,
            "ready",
        )

    monkeypatch.setattr(module, "observe_source_readiness", once_transient)
    widget._v2_source_caps_retry_delay = 0.0

    first = widget._controls_v2_headless_source_caps(str(source), live=False)
    second = widget._controls_v2_headless_source_caps(str(source), live=False)

    assert first == SourceCaps()
    assert second.raw_reachable is True
    assert calls["count"] == 2


def test_source_identity_stamp_has_owner_on_first_call(tmp_path):
    """The registry is bootstrapped before the adapter owner becomes part of
    the cache key, so the first and second keys cannot flap."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    source = _eiger_master(tmp_path)
    first = staticWidget._controls_v2_source_identity_stamp(str(source))
    second = staticWidget._controls_v2_source_identity_stamp(str(source))

    assert first == second
    assert first is not None and first[2] == "nexus_hdf5"
