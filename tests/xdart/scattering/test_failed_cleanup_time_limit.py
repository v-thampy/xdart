"""A failed Run whose cleanup keeps failing is given up after a time limit, so Run and the scans unlock (maintainer,
2026-10-26, after ab9bfdee: "will this apply for such freezes of the run button in general?"; his pick "Add the time
limit": "If the same cleanup failure repeats for about 10 s, the page unlocks Run and the scans, and warns which output
is still held ... A held output stays as the failure left it: Replace rebuilds it, and until xdart restarts a new Run
can't write that one output, though others are fine"). The real page, executor and Live Runs over real TIFF + .txt
files (test_failed_run_cleanup_page's failed Run); the failed Run's own output never finishes, as the beamline's held
writer did not before ab9bfdee, and later Runs' outputs finish."""
from __future__ import annotations

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _pump
from tests.xdart.scattering.test_failed_run_cleanup_page import _failed_run, _intent, older_output  # noqa: F401
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering import page as page_module
from xdart.gui.tabs.scattering.adapters.dynamic_output import DynamicOutputAdapter
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.state_machine import RunPhase

_STUCK = "writer flush incomplete: checkpoint requires EXECUTING, got integrity-hold"


@pytest.fixture
def stuck_cleanup(monkeypatch):
    """The first output that finishes from here on (the failed Run's) fails every time; the time limit is 2 s."""
    finish, stuck = DynamicOutputAdapter.finish_all, []

    def failing(self, *, stopped=False):
        if not stuck:
            stuck.append(self)
        if self is stuck[0]:
            raise RuntimeError(_STUCK)
        return finish(self, stopped=stopped)
    monkeypatch.setattr(DynamicOutputAdapter, "finish_all", failing)
    monkeypatch.setattr(page_module, "_FAILED_CLEANUP_GIVE_UP_S", 2.0, raising=False)
    return stuck


def _given_up(qapp, page, observed, older_output, notices):  # noqa: F811
    terminal = _failed_run(qapp, page, older_output, observed)
    assert terminal.kind is StandardEventKind.FAILED and terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING, (
        terminal.kind, terminal.cleanup_status)
    executor = page._run_executor
    assert _pump(qapp, lambda: page._start_permitted() == (True, ""), 15.), (
        page._start_permitted(), notices[-3:])
    return terminal, executor


def test_a_failed_run_whose_cleanup_keeps_failing_is_given_up_and_run_and_the_scans_unlock(
        qapp, tmp_path, older_output, stuck_cleanup, monkeypatch):  # noqa: F811
    with _page(qapp, _intent(tmp_path)) as (page, _view, observed):
        notices = []
        notice = page._notice
        monkeypatch.setattr(page, "_notice", lambda text, *args, **kwargs: (
            notices.append(text), notice(text, *args, **kwargs))[1])
        terminal, executor = _given_up(qapp, page, observed, older_output, notices)
        assert stuck_cleanup, "the failed Run's output never finished"
        assert page._lifecycle.reset_permitted and page._shell.browser.scans.isEnabled()
        assert page._context_controller.browse_lifecycle_admissible(), "the browser's scans stay locked"
        assert page._shell.run_controls.startButton.isEnabled()
        [unlocked] = [text for text in notices if text.startswith("Run unlocked")]
        assert unlocked == (
            "Run unlocked: the failed Run's cleanup did not finish in 2 s. old_int1d.nexus stays as the failure left "
            "it (Replace rebuilds it); until xdart restarts, a new Run skips it."), unlocked
        assert executor._exact_run(terminal.run_identity) is None  # it left the executor ...
        closed = executor.close(terminal.run_identity)  # ... and closing it again answers at once, kept as it was
        assert closed.cleanup_status is CleanupStatus.CLEANED and any(
            _STUCK in failure.message for failure in closed.cleanup_failures), closed
        # Quitting does not wait for it again (the app's close polls the page's).
        assert _pump(qapp, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED, 5.)


@pytest.mark.parametrize("mode", ["Overwrite", "Append"])
def test_a_directory_run_skips_only_an_output_still_held_and_processes_the_others(tmp_path, mode):
    """An output held as a Run given up on holds it (its lease is process-wide): a directory Run skips that scan,
    names it, and processes the others (before: the whole Run failed "target already leased: ...")."""
    from tests.xdart.scattering.test_append_already_processed import _run
    from tests.xdart.scattering.test_append_changed_images import _frame as _scan_frame, _processed
    from xrd_tools.io.output_transaction import get_output_transaction_coordinator

    raw, output = _processed(tmp_path, others=("scanb",))
    other = output.with_name("scanb_int1d.nexus")
    _scan_frame(raw, "scana", 3)
    _scan_frame(raw, "scanb", 2)
    before = {path.name: path.stat().st_mtime_ns for path in (output, other)}
    coordinator = get_output_transaction_coordinator()
    hold = coordinator.hold_target(output, label="given-up")
    try:
        events, _catalog = _run(tmp_path, raw, 9311, mode=mode)
    finally:
        coordinator.release_target(hold)
    [terminal] = [event for event in events if event.kind in _TERMINAL]
    assert terminal.kind is StandardEventKind.FINISHED, (terminal.kind, terminal.detail)
    assert terminal.warnings == (
        "Skipped scana_int1d.nexus: it is still held by another Run or operation of this xdart; if a Run's cleanup "
        "could not finish, restart xdart to write it again",), terminal.warnings
    assert {path.name: path.stat().st_mtime_ns != before[path.name] for path in (output, other)} == {
        "scana_int1d.nexus": False, "scanb_int1d.nexus": True}


def test_after_the_limit_a_new_run_skips_only_the_held_output(qapp, tmp_path, older_output, stuck_cleanup,
                                                               monkeypatch):  # noqa: F811
    with _page(qapp, _intent(tmp_path)) as (page, _view, observed):
        _given_up(qapp, page, observed, older_output, [])
        observed.clear()
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: page._lifecycle.phase is RunPhase.RUNNING, 30.), page._lifecycle.phase
        _pump(qapp, lambda: False, 1.)
        _frame(older_output, "old", 3, "txt", 3.0)  # the held output's scan grows ...
        _frame(older_output, "fresh", 4, "txt", 4.0)  # ... and another scan does
        assert _pump(qapp, lambda: any(event.kind is StandardEventKind.FRAME_READY and str(event.artifact).endswith(
            "fresh_int1d.nexus") for event in observed), 60.), "the other scan was never processed"
        _pump(qapp, lambda: False, 2.)
        page._shell.run_controls.stopButton.click()
        assert _pump(qapp, lambda: any(event.kind in _TERMINAL for event in observed), 60.)
        terminal = [event for event in observed if event.kind in _TERMINAL][-1]
        assert terminal.kind is StandardEventKind.STOPPED, (terminal.kind, terminal.detail)
        assert terminal.cleanup_status is CleanupStatus.CLEANED, terminal.cleanup_status
        held = [warning for warning in terminal.warnings if "old_int1d.nexus" in warning]
        assert held == ["Skipped old_int1d.nexus: it is still held by another Run or operation of this xdart; if a "
                        "Run's cleanup could not finish, restart xdart to write it again"], terminal.warnings
