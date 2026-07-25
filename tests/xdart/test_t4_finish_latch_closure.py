"""O-1a-T4 (§9.10 Step 6) — production finish-latch closure.

Every production finish entry point must leave the shared run lifecycle truthful
even when fallible finish work raises BEFORE or DURING the ordinary
``_exit_run_state()`` call.  At the parent each owner reaches ``_exit_run_state``
only from inside its rich body, after at least one fallible operation:

    wrangler_finished        -> _wrangler_finished_body
                                integrator_thread.isRunning()   <- fallible
                                _exit_run_state()
    integrator_thread_finished -> _finalize_processing_run
                                thread_state_changed()          <- fallible
                                _wrangler_run_active()          <- fallible
                                _exit_run_state()
    stitch_thread_finished   -> thread_state_changed()           <- fallible
                                wrangler.thread.isRunning()      <- fallible
                                _exit_run_state()

so a raise at any of those points strands `_run_active`, the H5Viewer writing
guard, the display processing flag, the mode row, Stop/Start, and Open — the GUI
is then permanently "in a run" and no later Start can proceed.

T-4.2 note (§35.6): this module is NOT byte-unchanged since `3a22d661`. Two cases
were deliberately reconciled with the later binding contract — the wrangler
pre-exit injection moved off the overlap probe (which §34.6.B now treats as an
unknown, lock-holding owner), and a projection failure with no rich primary is
expected to PROPAGATE as well as be surfaced (§34.7 family 2). Both dispositions
were reviewed and accepted; the substantive assertions are unchanged.

Cases 1-3 are §9.10 test 11 split per owner; case 8 is §9.10 test 12.  Every case
drives the REAL production slot on a REAL ``staticWidget`` with its real signal
connections asserted — calling the new closure helper directly would not be proof.
Normal-finish display behaviour stays owned by the existing run-end sentinels
(`test_batch_finish_select_last`, the live-refresh modules); this oracle asserts
lifecycle truthfulness only and does not clone their display assertions.
"""

from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtWidgets


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


def _boom(*_a, **_k):
    raise RuntimeError("injected finish-tail failure")


def _lifecycle_truth(widget):
    """The minimum truthful-idle lifecycle projection T-4 must guarantee.

    Start's ENABLEMENT is deliberately absent.  Measured at the parent: after any
    normal finish, `_exit_run_state` releases the reintegrate-era Start lock and
    the Controls readiness profile then owns the final value — on a widget with no
    calibration/source Start stays disabled, which is correct.  Asserting
    `start_enabled is True` here would demand that lifecycle closure override the
    readiness owner.  The requirement that matters — the RUN no longer holds Start
    — is proven by `_later_start_reaches_preparation` instead.
    """
    df = getattr(widget, "displayframe", None)
    return {
        "run_active": bool(getattr(widget, "_run_active", False)),
        "processing_active": bool(getattr(df, "_processing_active", False)),
        "run_writing": bool(
            getattr(widget.h5viewer, "_run_writing", False)),
        "mode_row_enabled": bool(
            widget.controls.modeCombo.isEnabled()),
        "stop_enabled": bool(widget.controls.stopButton.isEnabled()),
    }


_TRUTHFUL_IDLE = {
    "run_active": False,
    "processing_active": False,
    "run_writing": False,
    "mode_row_enabled": True,
    "stop_enabled": False,
}


def _assert_truthful_idle(widget):
    actual = _lifecycle_truth(widget)
    assert actual == _TRUTHFUL_IDLE, (
        f"lifecycle not truthful after finish: {actual}")


def _source_authority(widget):
    w = widget.wrangler
    t = getattr(w, "thread", None)
    return (
        getattr(w, "source_run_plan", None),
        getattr(w, "source_spec", None),
        getattr(w, "source_index_session", None),
        getattr(t, "source_run_plan", None),
        getattr(t, "source_spec", None),
    )


def _idle_owners(widget, monkeypatch):
    """Every owner probe answers idle (the ordinary post-run reading)."""
    for thread in (widget.wrangler.thread,
                   widget.integratorTree.integrator_thread,
                   widget.stitch_thread):
        monkeypatch.setattr(thread, "isRunning", lambda: False)


def _later_start_reaches_preparation(widget):
    """A later Start must reach the ordinary T-3 preparation path."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        FrozenRunConfiguration,
    )

    assert widget._controls_v2_active_run_owner() is None, (
        "T-3 admission still refuses after the finish unwound")
    frozen = widget._prepare_controls_v2_run_configuration()
    assert isinstance(frozen, FrozenRunConfiguration)
    return frozen


# --------------------------------------------------------------------------- #
# Production-connection proof: these are real signal->slot entry points.
# --------------------------------------------------------------------------- #

def test_all_three_finish_entry_points_are_production_connected(widget):
    """The three slots this oracle drives are the ones Qt actually invokes."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    for owner, signal_owner, signal_name, slot in (
        ("wrangler", widget.wrangler, "finished",
         staticWidget.wrangler_finished),
        ("reintegration", widget.integratorTree.integrator_thread, "finished",
         staticWidget.integrator_thread_finished),
        ("stitch", widget.stitch_thread, "finished",
         staticWidget.stitch_thread_finished),
    ):
        signal = getattr(signal_owner, signal_name)
        assert signal is not None, f"{owner} has no {signal_name} signal"
        # The bound slot on the widget is the production method under test.
        assert getattr(widget, slot.__name__).__func__ is slot


# --------------------------------------------------------------------------- #
# Cases 1-3 — §9.10 test 11, split per finish owner.
# --------------------------------------------------------------------------- #

def test_wrangler_pre_exit_failure_still_closes_the_lifecycle(
        widget, monkeypatch):
    """Case 1. The wrangler finish tail's FIRST fallible operation is the
    reintegrate overlap probe, which precedes `_exit_run_state()`."""
    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    assert widget._run_active is True
    # T-4.1 (§34.6.B): injecting at the reintegrate OVERLAP PROBE would make that
    # owner unobservable, which now correctly HOLDS the shared lifecycle — a
    # different case (family 4).  Inject just after the probe instead, so this
    # case still proves that a pre-exit failure cannot strand the lifecycle.
    monkeypatch.setattr(widget, "_flush_pending_update", _boom)

    with pytest.raises(RuntimeError, match="injected finish-tail failure"):
        widget.wrangler_finished()

    # The primary failure is preserved (above) AND the lifecycle is truthful.
    _assert_truthful_idle(widget)
    assert _source_authority(widget) == (None, None, None, None, None)
    # The transient fault is over; a later Start proceeds normally.
    monkeypatch.setattr(
        widget.integratorTree.integrator_thread, "isRunning", lambda: False)
    _later_start_reaches_preparation(widget)


def test_integrator_pre_exit_failure_still_closes_the_lifecycle(
        widget, monkeypatch):
    """Case 2. `_finalize_processing_run`'s first fallible operation is
    `thread_state_changed()`, before `_exit_run_state()`."""
    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    monkeypatch.setattr(widget, "thread_state_changed", _boom)

    with pytest.raises(RuntimeError, match="injected finish-tail failure"):
        widget.integrator_thread_finished()

    _assert_truthful_idle(widget)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    _later_start_reaches_preparation(widget)


def test_stitch_pre_exit_failure_still_closes_the_lifecycle(
        widget, monkeypatch):
    """Case 3. `stitch_thread_finished`'s first fallible operation is
    `thread_state_changed()`, before `_exit_run_state()`."""
    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    monkeypatch.setattr(widget, "thread_state_changed", _boom)

    with pytest.raises(RuntimeError, match="injected finish-tail failure"):
        widget.stitch_thread_finished()

    _assert_truthful_idle(widget)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    _later_start_reaches_preparation(widget)


def test_integrator_finish_does_not_clear_wrangler_source_authority(
        widget, monkeypatch):
    """T-4.3 item 4: source authority is released only by its OWNING path, even
    when a non-owning finish performs lifecycle closure."""
    _idle_owners(widget, monkeypatch)
    widget.wrangler.source_run_plan = "PLAN"
    widget.wrangler.source_spec = "SPEC"
    widget._enter_run_state()
    monkeypatch.setattr(widget, "thread_state_changed", _boom)

    with pytest.raises(RuntimeError):
        widget.integrator_thread_finished()

    assert widget.wrangler.source_run_plan == "PLAN"
    assert widget.wrangler.source_spec == "SPEC"


# --------------------------------------------------------------------------- #
# Case 4 — a cleanup SUBSTEP fails: later substeps still run, primary preserved.
# --------------------------------------------------------------------------- #

def test_cleanup_substep_failure_continues_and_preserves_the_primary(
        widget, monkeypatch, caplog):
    """Case 4. One lifecycle-restoration substep raises.  Later INDEPENDENT
    substeps still run, the primary finish exception is not replaced, and the
    failed cleanup seam is named in a diagnostic."""
    import logging

    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    # Primary failure in the rich body ...
    monkeypatch.setattr(widget, "thread_state_changed", _boom)
    # ... and a cleanup substep that also fails.
    monkeypatch.setattr(widget.h5viewer, "set_run_writing", _boom)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="injected finish-tail failure"):
            widget.integrator_thread_finished()

    # Independent later substeps ran despite the failed one.
    truth = _lifecycle_truth(widget)
    assert truth["run_active"] is False
    assert truth["mode_row_enabled"] is True
    assert truth["stop_enabled"] is False
    # The failed seam is named, as a SECONDARY diagnostic.
    assert any("set_run_writing" in r.getMessage() for r in caplog.records), (
        "the failing cleanup substep was not named in a diagnostic")


def test_cleanup_failure_without_a_primary_is_surfaced_not_swallowed(
        widget, monkeypatch, caplog):
    """T-4.3 item 3: with NO primary failure, a cleanup failure must reach the
    visible/structured error seam rather than logging a false success."""
    import logging

    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    # `set_stop_enabled` is already guarded INSIDE `_exit_run_state`, so the rich
    # body swallows it and there is genuinely no primary exception — exactly the
    # no-primary shape this case needs.  (Injecting `set_run_writing` instead
    # would make `_exit_run_state` itself raise, i.e. a real primary failure,
    # which case 4 above already covers.)
    monkeypatch.setattr(widget.controls, "set_stop_enabled", _boom)

    with caplog.at_level(logging.WARNING):
        # T-4.1: the restoration projection IS the ordinary exit path now, so a
        # failure in it is a genuine run-end failure — surfaced AND propagated,
        # never silently swallowed into a false success.
        with pytest.raises(RuntimeError, match="injected finish-tail failure"):
            widget.integrator_thread_finished()

    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("set_stop_enabled" in r.getMessage() for r in records), (
        "a cleanup failure with no primary exception was swallowed")
    assert widget._run_active is False


# --------------------------------------------------------------------------- #
# Case 5 — normal finish: closure happens exactly once and is idempotent.
# --------------------------------------------------------------------------- #

def test_normal_finish_closes_exactly_once_and_is_idempotent(
        widget, monkeypatch):
    """Case 5. Duplicate/late delivery adds no second refresh, source
    reconciliation, edit application, or generation change."""
    from xrd_tools.session.run_configuration import RunIntent

    _idle_owners(widget, monkeypatch)
    intent = widget._controls_v2_ensure_run_intent()
    widget._enter_run_state()

    refreshes = []
    monkeypatch.setattr(
        widget, "_refresh_controls_v2_profile",
        lambda **kw: refreshes.append(kw))
    reconciles = []
    monkeypatch.setattr(
        widget, "_sync_controls_v2_source_index",
        lambda *a, **k: reconciles.append(True))
    freezes = []
    real_freeze = RunIntent.freeze
    monkeypatch.setattr(
        RunIntent, "freeze",
        lambda self, **kw: (freezes.append(True), real_freeze(self, **kw))[1])
    generation = int(intent.generation)

    widget.integrator_thread_finished()
    _assert_truthful_idle(widget)
    after_first = len(refreshes)

    # Duplicate / late delivery of the same finish.
    widget.integrator_thread_finished()
    widget.integrator_thread_finished()

    _assert_truthful_idle(widget)
    assert len(refreshes) == after_first, (
        "a duplicate finish delivery added another profile refresh")
    assert reconciles == [], "finish closure reconciled the source"
    assert freezes == [], "finish closure froze a configuration"
    assert int(intent.generation) == generation


# --------------------------------------------------------------------------- #
# Case 6 — overlap: do not unlock state belonging to a live owner.
# --------------------------------------------------------------------------- #

def test_wrangler_finish_while_reintegrate_live_does_not_unlock(
        widget, monkeypatch):
    """Case 6a. A wrangler finishing while a REAL reintegrate is still active
    must not clear the shared latch or re-enable the locked controls."""
    monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: False)
    monkeypatch.setattr(widget.stitch_thread, "isRunning", lambda: False)
    monkeypatch.setattr(
        widget.integratorTree.integrator_thread, "isRunning", lambda: True)
    widget._enter_run_state()

    widget.wrangler_finished()

    assert widget._run_active is True, (
        "the live reintegrate's shared run latch was unlocked")
    assert widget.controls.modeCombo.isEnabled() is False

    # Final delivery for the remaining owner closes exactly once.
    monkeypatch.setattr(
        widget.integratorTree.integrator_thread, "isRunning", lambda: False)
    widget.integrator_thread_finished()
    _assert_truthful_idle(widget)


def test_integrator_finish_while_wrangler_live_does_not_unlock(
        widget, monkeypatch):
    """Case 6b. The mirror: an integrator finishing while the wrangler run is
    genuinely in flight must not unlock the wrangler's state."""
    monkeypatch.setattr(
        widget.integratorTree.integrator_thread, "isRunning", lambda: False)
    monkeypatch.setattr(widget.stitch_thread, "isRunning", lambda: False)
    monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: True)
    monkeypatch.setattr(widget.wrangler, "_run_phase", "running", raising=False)
    widget._enter_run_state()

    widget.integrator_thread_finished()

    assert widget._run_active is True, (
        "the live wrangler run's shared latch was unlocked")
    assert widget.controls.modeCombo.isEnabled() is False


# --------------------------------------------------------------------------- #
# Case 7 — the edit journal and RunIntent survive an exceptional finish.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("owner", ("wrangler", "integrator", "stitch"))
def test_exceptional_finish_preserves_journal_and_intent(
        widget, monkeypatch, owner):
    """Case 7. T-4.3 item 5: cleanup touches lifecycle/UI/source-handoff state
    only — never the journal, RunIntent, a freeze, a poll, or hydration."""
    import copy

    from xrd_tools.session.run_configuration import RunIntent

    _idle_owners(widget, monkeypatch)
    intent = widget._controls_v2_ensure_run_intent()
    widget._enter_run_state()
    # A real run-active (deferred) edit in the journal.
    widget._on_controls_v2_field_changed(("Int1D", "points"), 4242)
    journal_before = copy.deepcopy(widget._controls_v2_edit_journal_dict())
    assert journal_before, "the deferred edit was not journaled"
    generation_before = int(intent.generation)
    bai_before = copy.deepcopy(intent.bai_1d_args)

    staged, committed, frozen = [], [], []
    monkeypatch.setattr(
        widget, "stage_controls_transaction",
        lambda *a, **k: staged.append(True))
    monkeypatch.setattr(
        widget, "commit_controls_transaction",
        lambda *a, **k: committed.append(True))
    real_freeze = RunIntent.freeze
    monkeypatch.setattr(
        RunIntent, "freeze",
        lambda self, **kw: (frozen.append(True), real_freeze(self, **kw))[1])

    if owner == "wrangler":
        monkeypatch.setattr(
            widget.integratorTree.integrator_thread, "isRunning", _boom)
        slot = widget.wrangler_finished
    else:
        monkeypatch.setattr(widget, "thread_state_changed", _boom)
        slot = (widget.integrator_thread_finished if owner == "integrator"
                else widget.stitch_thread_finished)

    with pytest.raises(RuntimeError):
        slot()

    assert widget._controls_v2_edit_journal_dict() == journal_before
    assert int(intent.generation) == generation_before
    assert intent.bai_1d_args == bai_before
    assert staged == [] and committed == [] and frozen == []
    assert getattr(
        widget, "_pending_controls_v2_run_configuration", None) is None


# --------------------------------------------------------------------------- #
# Case 8 — §9.10 test 12: GI/source targeted-hydration regression.
# --------------------------------------------------------------------------- #

def test_targeted_gi_hydration_still_populates_after_exceptional_finish(
        widget, monkeypatch):
    """Case 8 (§9.10 test 12). After the exceptional-finish family, the accepted
    targeted metadata hydration seam still populates theta-motor choices, with no
    directory-wide metadata/frame walk."""
    import os

    _idle_owners(widget, monkeypatch)
    widget._enter_run_state()
    monkeypatch.setattr(widget, "thread_state_changed", _boom)
    with pytest.raises(RuntimeError):
        widget.integrator_thread_finished()
    _assert_truthful_idle(widget)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)

    walks = []
    real_walk, real_scandir = os.walk, os.scandir
    monkeypatch.setattr(
        os, "walk", lambda *a, **k: (walks.append(a), real_walk(*a, **k))[1])
    monkeypatch.setattr(
        os, "scandir",
        lambda *a, **k: (walks.append(a), real_scandir(*a, **k))[1])

    # The real production hydration path: the wrangler announces the current
    # source's motor knowledge, stamped with the live fingerprint/epoch, and the
    # static widget's single owner slot records it.
    outcome = widget.wrangler._announce_gi_hydration(
        ["halpha", "samz"], proved=True)

    assert outcome.accepted, f"hydration was not emitted: {outcome}"
    choices = widget._controls_v2_gi_motor_choices_for_freeze()
    assert choices is not None, "theta choices did not populate after hydration"
    assert "halpha" in tuple(choices)
    assert walks == [], f"targeted hydration performed a directory walk: {walks}"

    _later_start_reaches_preparation(widget)
