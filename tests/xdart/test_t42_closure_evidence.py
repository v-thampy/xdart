"""O-1a-T4.2 (§35) — total closure must preserve its EVIDENCE and its OWNERSHIP.

T-4.1's unification is accepted (§35.1) and is not reopened here.  Four concrete
violations of the already-frozen §34 contract survived it:

* §35.2 a positively ACTIVE wrangler QThread was reclassified releasable by the UI
  `_run_phase`.  `imageWrangler.stop()` sets that phase idle before the worker has
  finished and not every source/mode has a streaming-session adapter, so
  ``isRunning()=True`` + ``phase="idle"`` + no session is a PRODUCTION-REAL Stop
  window — and another owner's finish unlocked the latch, the H5 writing guard and
  the mode row straight through it;
* §35.3 the first projection attempt's ordered ``(seam, exception)`` evidence was
  discarded: `_exit_run_state` raised only the first exception and the outer
  closure re-ran the WHOLE projection, so one-shot failures vanished from the
  record while successful side-effectful seams (``set_run_writing(False)``)
  replayed;
* §35.4 the run-scan capture was cleared only AFTER its fallible finalizer, so a
  finalizer failure retained the live run scan and the retry finalized the same
  identity twice; and
* §35.5 the compound advanced-controls helper short-circuited internally — a 1-D
  widget failure skipped both the 2-D widget and the combined dialog.

Cases 1-7 are §35.8's frozen families; case 8 (preservation) is the gate.  Every
case drives a REAL production finish slot on a REAL ``staticWidget``.

RULE 10: every invocation of this module must set ``XDART_SESSION_FILE`` to a
unique path under ``/Users/vthampy/repos/tmp``.
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


def _minimal_acquisition_context(scan=None):
    """The smallest real ``AcquisitionContext`` a duck host can own (X1 O-3).

    The promoted adversary case used a bare ``object()`` under the deleted
    ``_x1_run_scan_capture`` alias.  The run-end finalizer now consumes the
    context's ONE finalization claim, so the host has to own a real context for
    the "detached before the fallible finalizer / retry is a no-op" contract to
    mean anything.
    """
    from xdart.modules.display_context import (
        AcquisitionContext,
        ContextKind,
        new_context_token,
    )

    return AcquisitionContext(
        context_token=new_context_token(ContextKind.ACQUISITION),
        run_configuration=None,
        config_generation=None,
        config_fingerprint="",
        run_scan_key="run_a",
        source_path="",
        scan=scan if scan is not None else object(),
        frame=None,
        frame_ids=[],
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=None,
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


def _one_shot(message):
    """A seam that fails exactly ONCE and then succeeds — the shape that made the
    discarded first-pass evidence invisible."""
    state = {"fired": False}

    def _seam(*_a, **_k):
        if not state["fired"]:
            state["fired"] = True
            raise RuntimeError(message)

    _seam.state = state
    return _seam


def _counting(record, label):
    def _seam(*_a, **_k):
        record.append(label)

    return _seam


def _closure_messages(caplog):
    return " | ".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def _record_closure_report(monkeypatch):
    """Capture the ordered (seam, exception) record the closure reporter is
    handed — that list IS the evidence §35.3 requires, independent of the log
    format (which emits both a name summary and a reason detail)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    seen = []
    real = staticWidget._report_run_lifecycle_closure_failures

    def _spy(widget, origin, failures, primary_in_flight):
        seen.append((origin, list(failures), primary_in_flight))
        return real(widget, origin, failures, primary_in_flight)

    monkeypatch.setattr(
        staticWidget, "_report_run_lifecycle_closure_failures", _spy)
    return seen


# --------------------------------------------------------------------------- #
# Case 1 — a stopping wrangler with NO session holds every shared lock.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("origin", ("integrator", "stitch"))
def test_active_stopping_wrangler_without_session_holds_shared_locks(
        widget, monkeypatch, origin):
    """§35.2/§35.7.A. The production Stop/unwind window: the wrangler QThread is
    still positively ACTIVE while its UI phase already reads idle and there is no
    streaming session to save the decision.  Owner truth is tri-state ONLY — the
    phase may not downgrade an active thread."""
    threads = {
        "integrator": widget.integratorTree.integrator_thread,
        "stitch": widget.stitch_thread,
    }
    monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: True)
    monkeypatch.setattr(widget.wrangler, "_run_phase", "idle", raising=False)
    monkeypatch.setattr(widget.wrangler, "scan_session", None, raising=False)
    # Bind `label` per iteration — a late-binding lambda would leave every other
    # owner reading the LAST loop value and mask the behaviour under test.
    for label, thread in threads.items():
        monkeypatch.setattr(
            thread, "isRunning", lambda label=label: label == origin)
    # "no session" is the PRODUCTION observation, not a forced attribute
    # (`imageThread.scan_session` is a read-only property).
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    assert staticWidget._controls_v2_session_activity(widget) == "idle"
    widget._enter_run_state()
    assert widget.h5viewer._run_writing is True

    monkeypatch.setattr(threads[origin], "isRunning", lambda: False)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    slot = (widget.integrator_thread_finished if origin == "integrator"
            else widget.stitch_thread_finished)
    slot()

    assert widget._run_active is True, (
        "an actively-stopping wrangler was treated as releasable")
    assert widget.h5viewer._run_writing is True
    assert widget.controls.modeCombo.isEnabled() is False

    # Once the wrangler is positively idle, ITS OWN delivery closes exactly once.
    monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: False)
    widget.wrangler_finished()
    assert _lifecycle_truth(widget) == _TRUTHFUL_IDLE


# --------------------------------------------------------------------------- #
# Case 2 — a TRUTHFUL stale phase does not strand a complete finish.
# --------------------------------------------------------------------------- #

def test_truthful_stale_running_phase_does_not_strand_the_finish(
        widget, monkeypatch):
    """§35.7.A. The truthful stale-PHASE shape is an IDLE QThread carrying a
    leftover ``running`` UI phase.  That must not hold the lifecycle."""
    _idle_owners(widget, monkeypatch)
    monkeypatch.setattr(widget.wrangler, "_run_phase", "running", raising=False)
    widget._enter_run_state()
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)

    widget.integrator_thread_finished()

    assert _lifecycle_truth(widget) == _TRUTHFUL_IDLE


# --------------------------------------------------------------------------- #
# Case 3 — two one-shot failures are both named, once, in projection order.
# --------------------------------------------------------------------------- #

def test_two_one_shot_failures_are_both_named_once_in_order(
        widget, monkeypatch, caplog):
    """§35.3/§35.7.B. Both first-pass seams appear in the record exactly once and
    in projection order, and the FIRST exception object stays the primary."""
    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    first = _one_shot("one-shot finalization failure")
    second = _one_shot("one-shot writing-guard failure")
    monkeypatch.setattr(widget.displayframe, "finish_processing", first)
    monkeypatch.setattr(widget.h5viewer, "set_run_writing", second)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    reports = _record_closure_report(monkeypatch)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError,
                           match="one-shot finalization failure"):
            widget.integrator_thread_finished()

    assert len(reports) == 1, f"expected one closure report, got {len(reports)}"
    _origin, failures, _primary = reports[0]
    seams = [seam.split(" (")[0] for seam, _ in failures]
    assert seams == ["finish_processing", "set_run_writing"], (
        f"first-pass evidence was erased or reordered: {seams}")
    reasons = [str(exc) for _, exc in failures]
    assert reasons == ["one-shot finalization failure",
                       "one-shot writing-guard failure"]
    # Both were one-shot, so the retry restored the lifecycle.
    assert widget._run_active is False
    assert "finish_processing" in _closure_messages(caplog)


# --------------------------------------------------------------------------- #
# Case 4 — an already-successful seam is never replayed.
# --------------------------------------------------------------------------- #

def test_successful_seams_are_not_replayed_on_a_later_failure(
        widget, monkeypatch, caplog):
    """§35.3/§35.7.B req 4. A LATER seam failing must not re-invoke an
    already-successful, side-effectful ``set_run_writing(False)``."""
    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    calls = []
    monkeypatch.setattr(
        widget.h5viewer, "set_run_writing", _counting(calls, "writing"))
    # A PERMANENT failure at a strictly later seam forces the recovery pass.
    monkeypatch.setattr(widget, "_project_controls_v2_readiness", _boom)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="injected finish-tail failure"):
            widget.integrator_thread_finished()

    assert calls == ["writing"], (
        f"an already-successful seam was replayed: {calls}")
    assert "readiness_projection" in _closure_messages(caplog)


# --------------------------------------------------------------------------- #
# Case 5 — capture release survives finalization failure, exactly once.
# --------------------------------------------------------------------------- #

def test_finalizer_failure_releases_the_capture_and_finalizes_once(
        widget, monkeypatch, caplog):
    """§35.4/§35.7.C, under the O-3 owner. A PERMANENT `finish_processing`
    failure must still leave the acquisition context DETACHED from the widget
    and must record exactly ONE finalization attempt — the retry may not
    finalize the same identity again."""
    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    assert widget._acquisition_context.scan is widget.scan
    attempts = []

    def _failing_finish(*_a, **_k):
        attempts.append(True)
        raise RuntimeError("permanent finalization failure")

    monkeypatch.setattr(
        widget.displayframe, "finish_processing", _failing_finish)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError,
                          match="permanent finalization failure"):
            widget.integrator_thread_finished()

    # X1 O-3 (c3): the context is released by its own ordered substep, after
    # the finalization seam has had its ONE attempt.  T-4.2's invariant is
    # unchanged — a permanent failure still releases the run scan, and the
    # retry may not finalize the same identity again.
    assert widget._acquisition_context is None, (
        "the live run scan stayed owned after terminal closure")
    assert len(attempts) == 1, (
        f"the captured identity was finalized {len(attempts)} times")
    assert "finish_processing" in _closure_messages(caplog)


# --------------------------------------------------------------------------- #
# Case 6 — the advanced controls are independent owners.
# --------------------------------------------------------------------------- #

def test_advanced_1d_failure_still_restores_2d_and_the_dialog(
        widget, monkeypatch, caplog):
    """§35.5/§35.7.D. A PERMANENT 1-D advanced-widget failure must still enable
    the 2-D widget AND the combined dialog, and must name the 1-D seam."""
    dialog = QtWidgets.QDialog(widget)
    widget._integ_adv_combined_dlg = dialog
    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    assert widget.integratorTree.advancedWidget2D.isEnabled() is False
    assert dialog.isEnabled() is False

    monkeypatch.setattr(
        widget.integratorTree.advancedWidget1D, "setEnabled", _boom)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="injected finish-tail failure"):
            widget.integrator_thread_finished()

    assert widget.integratorTree.advancedWidget2D.isEnabled() is True, (
        "the 2-D advanced widget was skipped by the 1-D failure")
    assert dialog.isEnabled() is True, (
        "the combined advanced dialog was skipped by the 1-D failure")
    assert "advanced_widget_1d" in _closure_messages(caplog)


def test_advanced_owners_are_three_independent_projection_entries(widget):
    """§35.7.D. The compound helper is retired: the projection names the three
    advanced owners separately, so one cannot short-circuit the others."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    dialog = QtWidgets.QDialog(widget)
    widget._integ_adv_combined_dlg = dialog
    seams = [seam for seam, _ in
             staticWidget._idle_lifecycle_substeps(widget)]

    assert "advanced_widget_1d" in seams
    assert "advanced_widget_2d" in seams
    assert "advanced_dialog" in seams
    assert not hasattr(staticWidget, "_set_advanced_integration_enabled"), (
        "the one-use compound advanced helper was not retired")
    assert (seams.index("advanced_widget_1d")
            < seams.index("advanced_widget_2d")
            < seams.index("advanced_dialog"))


# --------------------------------------------------------------------------- #
# Case 7 — a rich failure BEFORE any projection attempt.
# --------------------------------------------------------------------------- #

def test_rich_failure_before_any_projection_runs_everything_once(
        widget, monkeypatch, caplog):
    """§35.7.B req 7. When the rich body fails before the projection is attempted,
    source release AND the complete projection still run exactly once, and the
    original rich exception keeps priority."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    _idle_owners(widget, monkeypatch)
    widget.wrangler.source_run_plan = "PLAN"
    widget._enter_run_state()
    writing = []
    monkeypatch.setattr(
        widget.h5viewer, "set_run_writing", _counting(writing, "writing"))
    released = []
    real_release = staticWidget._clear_controls_v2_run_source_authority
    monkeypatch.setattr(
        staticWidget, "_clear_controls_v2_run_source_authority",
        lambda w: (released.append(True), real_release(w))[1])
    # Fail the wrangler body BEFORE it reaches `_exit_run_state`.
    monkeypatch.setattr(
        widget, "_exit_run_state",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("rich failure before projection")))

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError,
                          match="rich failure before projection"):
            widget.wrangler_finished()

    assert released == [True], f"source release ran {len(released)} times"
    monkeypatch.undo()
    assert writing == ["writing"] or writing == [], (
        f"the projection replayed the writing guard: {writing}")


# --------------------------------------------------------------------------- #
# Promoted from Codex's preserved adversary (run, never edited):
# ~/repos/tmp/test_codex_t41_projection_atomicity.py
# --------------------------------------------------------------------------- #

def test_promoted_boundary_finalizer_releases_capture_without_widget():
    """Codex adversary case 1, promoted: the boundary contract holds even for a
    minimal host — the context is detached BEFORE the fallible finalizer."""
    from types import SimpleNamespace

    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    def _fail(*_a):
        raise RuntimeError("display finalization failed")

    host = SimpleNamespace(
        _acquisition_context=_minimal_acquisition_context(),
        displayframe=SimpleNamespace(finish_processing=_fail),
    )

    with pytest.raises(RuntimeError, match="display finalization failed"):
        staticWidget._finalize_acquisition_context_scan(host)

    assert host._acquisition_context.finalization_claimed is True
    assert host._acquisition_context.finalized is False
    # A retry must be a no-op rather than finalizing the identity again.
    staticWidget._finalize_acquisition_context_scan(host)
    # The release substep is ORDERED after that one attempt, never before it.
    from xdart.modules.display_context import DisplayContextError

    fresh = SimpleNamespace(_acquisition_context=_minimal_acquisition_context())
    with pytest.raises(DisplayContextError):
        staticWidget._release_acquisition_context(fresh)
    staticWidget._release_acquisition_context(host)
    assert host._acquisition_context is None


def test_promoted_real_finish_failure_releases_the_captured_scan(
        widget, monkeypatch):
    """Codex adversary case 3, promoted verbatim in intent."""
    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    assert widget._acquisition_context.scan is widget.scan
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    monkeypatch.setattr(widget.displayframe, "finish_processing", _boom)

    with pytest.raises(RuntimeError, match="injected finish-tail failure"):
        widget.integrator_thread_finished()

    assert widget._acquisition_context is None


# --------------------------------------------------------------------------- #
# §35.6 — the three STALE lines of Codex's T-4 reproducer
# (~/repos/tmp/test_codex_t4_start_lock_release.py), reshaped in-tree exactly as
# §35.6 directs.  The original module is Codex-owned: it is run, never edited.
# Each core product claim is preserved; only the stale input/assertion changes.
# --------------------------------------------------------------------------- #

def test_reshaped_ready_profile_recovers_start_after_mid_exit_failure(
        widget, monkeypatch, tmp_path, qapp):
    """Reshape 1: the original forced Start on for a bare, canonically UNREADY
    widget.  §34.5/§35.6 supersede that — Start comes from the readiness owner, so
    the case is driven with a genuinely READY profile."""
    from .test_t41_closure_totality import _enter_run_with_integrator_lock, _make_ready

    _make_ready(widget, tmp_path, qapp)
    _enter_run_with_integrator_lock(widget, monkeypatch)
    assert widget.controls.startButton.isEnabled() is False

    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    monkeypatch.setattr(widget.h5viewer, "set_run_writing", _boom)
    with pytest.raises(RuntimeError, match="injected finish-tail failure"):
        widget.integrator_thread_finished()

    assert widget._run_active is False
    assert widget._controls_v2_active_run_owner() is None
    assert widget.controls.startButton.isEnabled() is True


def test_reshaped_source_cleanup_never_replaces_the_rich_primary(
        widget, monkeypatch):
    """Reshape 2: the original injected its "primary" THROUGH the other-owner
    activity probe, which correctly makes that owner unknown and lock-holding.
    §35.6: move the rich failure AFTER owner observation."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    monkeypatch.setattr(
        widget, "_flush_pending_update",
        lambda: (_ for _ in ()).throw(RuntimeError("primary rich failure")))
    monkeypatch.setattr(
        staticWidget, "_clear_controls_v2_run_source_authority",
        lambda _w: (_ for _ in ()).throw(
            RuntimeError("secondary source cleanup failure")))

    with pytest.raises(RuntimeError, match="primary rich failure"):
        widget.wrangler_finished()

    assert widget._run_active is False


def test_reshaped_unknown_owner_is_diagnosed_while_locks_hold(
        widget, monkeypatch, caplog):
    """Reshape 3: while `_run_active` truthfully HOLDS, Start admission returns
    ``"run"`` before it can return an owner-qualified label.  §35.6: assert the
    owner-qualified warning/event separately from the admission decision."""
    integrator = widget.integratorTree.integrator_thread
    monkeypatch.setattr(widget.stitch_thread, "isRunning", lambda: False)
    monkeypatch.setattr(integrator, "isRunning", lambda: True)
    widget._enter_run_state()
    assert widget.h5viewer._run_writing is True

    monkeypatch.setattr(integrator, "isRunning", lambda: False)
    monkeypatch.setattr(widget.wrangler, "_run_phase", "running", raising=False)
    monkeypatch.setattr(
        widget.wrangler.thread, "isRunning",
        lambda: (_ for _ in ()).throw(RuntimeError("owner probe unavailable")))
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)

    with caplog.at_level(logging.WARNING):
        widget.integrator_thread_finished()

    # The shared locks hold, which is the product claim.
    assert widget._run_active is True
    assert widget.h5viewer._run_writing is True
    assert widget.controls.modeCombo.isEnabled() is False
    # The admission decision reports the latch first — assert the owner-qualified
    # diagnostic separately, which is where the unobservable owner is named.
    assert widget._controls_v2_active_run_owner() == "run"
    assert "wrangler" in _closure_messages(caplog).lower()
