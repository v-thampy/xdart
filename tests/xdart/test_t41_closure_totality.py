"""O-1a-T4.1 (§34) — finish closure must be TOTAL and FAIL CLOSED.

Three production violations survived T-4 because ordinary exit, abnormal closure,
overlap qualification, and wrangler source release each used a partially
different cleanup policy:

* §34.2 `_exit_run_state()` clears `_run_active` at its first mutable line and
  then runs fallible operations in sequence, so an early raise skipped the
  ordinary tail — and closure's separate, smaller fallback list omitted
  `enable_integration`, the mode-correct projection under the unlock guard, the
  advanced widgets/dialog, and the readiness projection of Start.  Every run-owned
  control stayed disabled while admission read `None`;
* §34.3 the overlap helper was a SECOND activity policy with fail-open booleans,
  so an UNOBSERVABLE other owner was treated as idle and the shared H5 writing
  guard was dropped before admission could refuse anything;
* §34.4 the nested `finally` let a source-cleanup exception REPLACE the rich-body
  primary on unwind.

Families 1-7 are §34.7's frozen correction oracle.  Family 5/6 promote the intent
of Codex's preserved reproducer `~/repos/tmp/test_codex_t4_start_lock_release.py`
(3 failed at `3a22d661`); that module is Codex-owned and is run, never edited.

The accepted T-4 14-case oracle is preserved UNCHANGED in
`test_t4_finish_latch_closure.py`; its preservation helpers are imported here
rather than duplicated (§34.7).
"""

from __future__ import annotations

import logging

import pytest
from pyqtgraph.Qt import QtWidgets

from .test_t4_finish_latch_closure import (
    _boom,
    _idle_owners,
    _lifecycle_truth,
    _TRUTHFUL_IDLE,
)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    value = staticWidget()
    try:
        yield value
    finally:
        try:
            value._exit_run_state(value._new_projection_receipt())
        except Exception:
            pass
        value.close()
        value.deleteLater()
        qapp.processEvents()


#: The run-owned control inventory §34.2's probe found stranded.  Start comes from
#: the readiness owner; the rest are the mode-correct integration projection and
#: the advanced widgets, none of which T-4's fallback list restored.
def _control_inventory(widget):
    ui = getattr(widget.integratorTree, "ui", None)
    itree = widget.integratorTree
    out = {"Start": widget.controls.startButton.isEnabled()}
    for label, owner, name in (
        ("frame1D", ui, "frame1D"),
        ("frame2D", ui, "frame2D"),
        ("Calibrate", ui, "pyfai_calib"),
        ("Mask", ui, "get_mask"),
        ("Reintegrate1D", ui, "reintegrate1D"),
        ("AdvancedWidget1D", itree, "advancedWidget1D"),
        ("AdvancedWidget2D", itree, "advancedWidget2D"),
    ):
        w = getattr(owner, name, None)
        if w is not None and hasattr(w, "isEnabled"):
            out[label] = bool(w.isEnabled())
    return out


def _make_ready(widget, tmp_path, qapp):
    """A canonically READY profile: real PONI + real image source + save path."""
    from xrd_tools.session.readiness import build_control_profile

    # Ensure a real processing mode, but switch ONLY if needed: a mode change
    # re-seeds the wrangler parameters through queued handlers and would clear the
    # PONI/source written below.  `_on_mode_changed` persists the mode, and
    # conftest isolates the session PER PROCESS, so a viewer-mode test earlier in
    # this process can leave the next widget in Image Viewer — hence the check
    # (and the restore in the unready case).
    if widget.controls.modeCombo.currentText() != "Int 2D":
        idx = widget.controls.modeCombo.findText("Int 2D")
        assert idx >= 0
        widget.controls.modeCombo.setCurrentIndex(idx)
        qapp.processEvents()
        widget._refresh_controls_v2_profile_now()
        qapp.processEvents()
    poni = tmp_path / "cal.poni"
    poni.write_text("Distance: 0.1\nPoni1: 0.01\nPoni2: 0.02\n"
                    "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n")
    raw = tmp_path / "scan_0001.tif"
    raw.write_bytes(b"")
    # Order matters and is production behaviour, not a workaround:
    # `_on_project_folder_changed` deliberately CLEARS the PONI and the source
    # paths (they were relative to the old root), so the project root must be set
    # BEFORE the calibration and source.
    widget.wrangler.parameters.child(
        "Project", "project_folder").setValue(str(tmp_path))
    widget.wrangler.parameters.child("Project", "h5_dir").setValue(str(tmp_path))
    widget._set_poni_field(str(poni))
    widget.wrangler.img_file = str(raw)
    widget.wrangler.parameters.child("Signal", "File").setValue(str(raw))
    widget._refresh_controls_v2_profile_now()
    profile = build_control_profile(widget._controls_v2_state())
    assert profile.can_run is True, f"fixture is not ready: {profile.run_blockers}"
    return profile


def _make_unready(widget, qapp):
    """A canonically UNREADY profile: a viewer processing page (can_run False)."""
    from xrd_tools.session.readiness import build_control_profile

    idx = widget.controls.modeCombo.findText("Image Viewer")
    assert idx >= 0, "Image Viewer mode is not offered"
    widget.controls.modeCombo.setCurrentIndex(idx)
    qapp.processEvents()
    profile = build_control_profile(widget._controls_v2_state())
    assert profile.can_run is False, "fixture is unexpectedly ready"
    return profile


def _enter_run_with_integrator_lock(widget, monkeypatch):
    """§34.2's exact probe shape: the REAL integrator owner is active at
    `_enter_run_state()` (which is the only thing that locks Start — a wrangler
    run morphs Start to Pause instead), then goes idle for the real finish
    delivery."""
    integrator = widget.integratorTree.integrator_thread
    monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: False)
    monkeypatch.setattr(widget.stitch_thread, "isRunning", lambda: False)
    monkeypatch.setattr(integrator, "isRunning", lambda: True)
    widget._enter_run_state()
    monkeypatch.setattr(integrator, "isRunning", lambda: False)


def _finish_integrator_with_injection(widget, monkeypatch, seam_owner, seam):
    """Drive the real integrator finish slot with one exit substep raising."""
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    monkeypatch.setattr(seam_owner, seam, _boom)
    with pytest.raises(RuntimeError, match="injected finish-tail failure"):
        widget.integrator_thread_finished()


# --------------------------------------------------------------------------- #
# Family 1 — ready profile, EARLY exit failure (§34.2's exact probe shape).
# --------------------------------------------------------------------------- #

def test_ready_profile_recovers_every_control_after_early_exit_failure(
        widget, monkeypatch, tmp_path, qapp):
    """Family 1. Inject at `finish_processing` — the FIRST fallible operation
    inside `_exit_run_state()`, before every restoration step.  Start and the
    whole mode-correct integration/advanced inventory must still recover."""
    _make_ready(widget, tmp_path, qapp)
    before = _control_inventory(widget)
    assert all(before.values()), f"fixture must start fully enabled: {before}"

    _enter_run_with_integrator_lock(widget, monkeypatch)
    during = _control_inventory(widget)
    assert not any(during.values()), f"run must lock every control: {during}"

    _finish_integrator_with_injection(
        widget, monkeypatch, widget.displayframe, "finish_processing")

    after = _control_inventory(widget)
    assert after == before, (
        f"controls stranded after an early exit failure: {after} != {before}")
    assert _lifecycle_truth(widget) == _TRUTHFUL_IDLE
    assert widget._controls_v2_active_run_owner() is None


# --------------------------------------------------------------------------- #
# Family 2 — ready profile, MID exit failure; primary stays primary.
# --------------------------------------------------------------------------- #

def test_ready_profile_recovers_after_mid_exit_failure_primary_preserved(
        widget, monkeypatch, tmp_path, qapp):
    """Family 2. Inject at `set_run_writing`, part-way through the projection:
    the LATER steps still recover and the injected exception is still the one
    that propagates."""
    _make_ready(widget, tmp_path, qapp)
    before = _control_inventory(widget)
    _enter_run_with_integrator_lock(widget, monkeypatch)

    _finish_integrator_with_injection(
        widget, monkeypatch, widget.h5viewer, "set_run_writing")

    after = _control_inventory(widget)
    assert after == before, f"later restoration steps did not run: {after}"
    truth = _lifecycle_truth(widget)
    assert truth["run_active"] is False
    assert truth["mode_row_enabled"] is True
    assert truth["stop_enabled"] is False


# --------------------------------------------------------------------------- #
# Family 3 — UNREADY profile: Start stays disabled, the rest releases.
# --------------------------------------------------------------------------- #

def test_unready_profile_keeps_start_disabled_but_releases_the_lock(
        widget, monkeypatch, qapp):
    """Family 3. Lifecycle cleanup must never force Start on: a canonically
    unready profile stays disabled while every other run-owned lock releases."""
    _make_unready(widget, qapp)
    _enter_run_with_integrator_lock(widget, monkeypatch)
    assert widget.controls.startButton.isEnabled() is False

    _finish_integrator_with_injection(
        widget, monkeypatch, widget.displayframe, "finish_processing")

    try:
        assert widget.controls.startButton.isEnabled() is False, (
            "cleanup forced Start on for an unready profile")
        truth = _lifecycle_truth(widget)
        assert truth["run_active"] is False
        assert truth["run_writing"] is False
        assert truth["mode_row_enabled"] is True
        assert truth["stop_enabled"] is False
    finally:
        # `_on_mode_changed` PERSISTS the mode, so restore a real processing mode:
        # otherwise every widget constructed later in this process (and in other
        # modules) would start in Image Viewer.
        idx = widget.controls.modeCombo.findText("Int 2D")
        if idx >= 0:
            widget.controls.modeCombo.setCurrentIndex(idx)
            qapp.processEvents()


# --------------------------------------------------------------------------- #
# Family 4 — an UNOBSERVABLE other owner never unlocks anything.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("other", ("wrangler", "integrator", "stitch"))
@pytest.mark.parametrize("fault", ("raises", "missing"))
def test_unknown_other_owner_holds_every_shared_lock(
        widget, monkeypatch, other, fault):
    """Family 4. `"active"` AND `"unknown"` both hold the shared lifecycle;
    only a positively-idle observation may close it."""
    threads = {
        "wrangler": widget.wrangler.thread,
        "integrator": widget.integratorTree.integrator_thread,
        "stitch": widget.stitch_thread,
    }
    # The owner that will FINISH is never the faulted one.
    origin = "integrator" if other != "integrator" else "stitch"
    for label, thread in threads.items():
        monkeypatch.setattr(thread, "isRunning", lambda: False)
    monkeypatch.setattr(threads[origin], "isRunning", lambda: True)
    widget._enter_run_state()
    assert widget.h5viewer._run_writing is True

    monkeypatch.setattr(threads[origin], "isRunning", lambda: False)
    if fault == "raises":
        monkeypatch.setattr(threads[other], "isRunning", _boom)
    else:
        monkeypatch.setattr(threads[other], "isRunning", None, raising=False)
    if other == "wrangler":
        monkeypatch.setattr(
            widget.wrangler, "_run_phase", "running", raising=False)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)

    slot = (widget.integrator_thread_finished if origin == "integrator"
            else widget.stitch_thread_finished)
    slot()

    assert widget._run_active is True, (
        f"an unobservable {other} owner ({fault}) unlocked the shared latch")
    assert widget.h5viewer._run_writing is True, "H5 writing guard was dropped"
    assert widget.controls.modeCombo.isEnabled() is False

    # A later positively-idle delivery closes exactly once.
    monkeypatch.setattr(threads[other], "isRunning", lambda: False)
    if other == "wrangler":
        monkeypatch.setattr(
            widget.wrangler, "_run_phase", "idle", raising=False)
    slot()
    assert _lifecycle_truth(widget) == _TRUTHFUL_IDLE


# --------------------------------------------------------------------------- #
# Family 5 — source cleanup never replaces the rich-body primary.
# --------------------------------------------------------------------------- #

def test_source_cleanup_failure_stays_secondary_to_the_rich_primary(
        widget, monkeypatch, caplog):
    """Family 5. Both the rich wrangler body AND source release fail: the rich
    failure remains primary, source cleanup is a NAMED secondary, and lifecycle
    closure still runs."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    # Inject the rich failure at a wrangler-body seam AFTER the owner probes, so
    # every owner stays POSITIVELY idle and this case isolates exception PRIORITY
    # rather than entangling itself with overlap.  (Injecting at the integrator
    # probe instead would make that owner "unknown", which §34.6.B requires to
    # HOLD the shared lifecycle — a different case; see Boundary 16.)
    monkeypatch.setattr(
        widget, "_flush_pending_update",
        lambda: (_ for _ in ()).throw(RuntimeError("primary rich failure")))

    def _source_boom(_w):
        raise RuntimeError("secondary source cleanup failure")

    monkeypatch.setattr(
        staticWidget, "_clear_controls_v2_run_source_authority", _source_boom)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="primary rich failure"):
            widget.wrangler_finished()

    assert _lifecycle_truth(widget) == _TRUTHFUL_IDLE
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "source" in messages.lower(), (
        f"the failed source-release seam was not named: {messages}")


def test_no_primary_source_cleanup_failure_is_surfaced_visibly(
        widget, monkeypatch, caplog):
    """Family 6. With no primary, a source-release failure must reach the same
    visible/structured incomplete-cleanup seam — never a false success and never
    an unqualified replacement exception."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    statuses = []
    monkeypatch.setattr(
        widget.wrangler, "_set_status_text",
        lambda text: statuses.append(text), raising=False)

    def _source_boom(_w):
        raise RuntimeError("secondary source cleanup failure")

    monkeypatch.setattr(
        staticWidget, "_clear_controls_v2_run_source_authority", _source_boom)

    with caplog.at_level(logging.WARNING):
        widget.wrangler_finished()

    assert _lifecycle_truth(widget) == _TRUTHFUL_IDLE
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("source" in r.getMessage().lower() for r in records), (
        "a no-primary source-cleanup failure was swallowed")
    assert statuses, "no visible status surfaced the incomplete cleanup"


# --------------------------------------------------------------------------- #
# Family 7 — idempotence and preservation across a normal + duplicate finish.
# --------------------------------------------------------------------------- #

def test_normal_and_duplicate_finish_add_no_extra_work(
        widget, monkeypatch, tmp_path, qapp):
    """Family 7. A normal finish plus duplicate deliveries add no extra readiness
    refresh, journal write, source reconciliation, freeze, or generation change."""
    import copy

    from xrd_tools.session.run_configuration import RunIntent

    _make_ready(widget, tmp_path, qapp)
    _idle_owners(widget, monkeypatch)
    intent = widget._controls_v2_ensure_run_intent()
    widget._enter_run_state()
    widget._on_controls_v2_field_changed(("Int1D", "points"), 5150)
    journal_before = copy.deepcopy(widget._controls_v2_edit_journal_dict())
    generation_before = int(intent.generation)

    refreshes, reconciles, freezes, staged = [], [], [], []
    monkeypatch.setattr(
        widget, "_refresh_controls_v2_profile",
        lambda **kw: refreshes.append(kw))
    monkeypatch.setattr(
        widget, "_sync_controls_v2_source_index",
        lambda *a, **k: reconciles.append(True))
    monkeypatch.setattr(
        widget, "stage_controls_transaction",
        lambda *a, **k: staged.append(True))
    real_freeze = RunIntent.freeze
    monkeypatch.setattr(
        RunIntent, "freeze",
        lambda self, **kw: (freezes.append(True), real_freeze(self, **kw))[1])

    widget.integrator_thread_finished()
    after_first = len(refreshes)
    assert _lifecycle_truth(widget) == _TRUTHFUL_IDLE

    widget.integrator_thread_finished()
    widget.integrator_thread_finished()

    assert _lifecycle_truth(widget) == _TRUTHFUL_IDLE
    assert len(refreshes) == after_first, (
        "a duplicate finish delivery added another readiness refresh")
    assert reconciles == [] and staged == [] and freezes == []
    assert widget._controls_v2_edit_journal_dict() == journal_before
    assert int(intent.generation) == generation_before
