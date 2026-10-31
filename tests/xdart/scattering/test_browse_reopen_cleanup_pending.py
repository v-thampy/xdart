"""BROWSE-REOPEN-CLEANUP-PENDING: a scan clicked while the shown scan's Browse cleanup is still settling opens with
that one click once the cleanup settles (maintainer 2026-10-27, "Yes fix the refusal in 1.2.7"). Before the fix the
click was refused with "Browse refused: previous Browse cleanup is pending" and dropped; a second click worked.
Real pages, real processed files, the real Browse loader and hydration owners."""

from __future__ import annotations

import logging
import threading

import pytest
from pyqtgraph.Qt import QtWidgets

import xdart.gui.tabs.scattering.page as page_module
from tests.xdart.scattering.test_e4_accumulation_selection_parity import _click_frame
from tests.xdart.scattering.test_e4_preview_transport import _write_processed
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from xdart.gui.tabs.scattering.browse_1d_hydration import Browse1DHydrationLane
from xdart.gui.tabs.scattering.browse_hydration import _BrowseHydrationOwner
from xdart.gui.tabs.scattering.browse_values import BrowseCleanupReceipt
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent

_PAGE_LOG = "xdart.gui.tabs.scattering.page"


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _shows(page, path, label=None):
    controller = page._context_controller
    frame = controller.navigation.current
    return True if (
        frame is not None and frame.artifact == str(path)
        and (label is None or frame.local_frame_label == label)
        and not controller.browse_pending and not page._scientific_repaint_pending
    ) else None


def _click_scan(page, path):
    page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(path), path=("artifact",)))


def _begin_spy(monkeypatch, page):
    """Record every Browse admission attempt as (value, refusal text or None, page notice when it began); the real
    method runs unchanged."""
    attempts = []
    controller = page._context_controller
    real = controller.begin_browse

    def spy(value, *args, **kwargs):
        notice = page._notice_text
        try:
            request = real(value, *args, **kwargs)
        except Exception as error:
            attempts.append((value, str(error), notice))
            raise
        attempts.append((value, None, notice))
        return request

    monkeypatch.setattr(controller, "begin_browse", spy)
    return attempts


def _stuck_release(monkeypatch):
    """Keep every Browse hydration release pending while the returned flag is set (a cleanup that never settles)."""
    stuck = threading.Event()
    stuck.set()
    real = _BrowseHydrationOwner.release

    def release(self, loader, browse, **kwargs):
        if stuck.is_set():
            return BrowseCleanupReceipt(browse.load_request, CleanupStatus.CLEANUP_PENDING)
        return real(self, loader, browse, **kwargs)

    monkeypatch.setattr(_BrowseHydrationOwner, "release", release)
    return stuck


def _close(page, qapp, *flags):
    for flag in flags:
        flag.clear()
    _wait(lambda: True if page.close_workspace().cleanup_status is CleanupStatus.CLEANED else None)
    page.deleteLater()
    qapp.processEvents()


def _int_page(tmp_path, monkeypatch, names):
    paths = tuple(_write_processed(tmp_path / name, labels=(1, 2))[0] for name in names)
    store = RunIntentStore(RunIntent(project_root=str(tmp_path), processing_mode="Int 1D"))
    page, _ = _page(tmp_path, monkeypatch, store=store)
    page.resize(1200, 800)
    page.show()
    return page, paths


def test_one_click_opens_a_scan_while_the_shown_scans_1d_read_is_in_flight(tmp_path, monkeypatch, qapp):
    page, (first, second) = _int_page(tmp_path, monkeypatch, ("first", "second"))
    controller = page._context_controller
    hold = threading.Event()
    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        # Hold the shown scan's next 1-D read in flight: its Browse context cannot be released until it ends.
        real_read = Browse1DHydrationLane._open_and_read

        def held_read(self, task, labels, fact_labels):
            hold.wait(10)
            return real_read(self, task, labels, fact_labels)

        monkeypatch.setattr(Browse1DHydrationLane, "_open_and_read", held_read)
        _click_frame(qapp, page._shell, 1, settle=False)
        assert _wait(lambda: True if controller.browse_preview_polling_needed else None)
        attempts = _begin_spy(monkeypatch, page)

        _click_scan(page, second)  # the only click on the second scan

        assert [(value, refusal) for value, refusal, _ in attempts] == [
            (str(second), "previous Browse cleanup is pending")]
        threading.Timer(0.3, hold.set).start()
        _wait(lambda: _shows(page, second, 1))
        assert [value for value, refusal, _ in attempts if refusal is None] == [str(second)]
        assert "Browse refused" not in page._notice_text
    finally:
        hold.set()
        _close(page, qapp)


def test_a_kept_click_gives_up_at_the_limit_with_one_warning_and_no_retry_storm(
    tmp_path, monkeypatch, qapp, caplog,
):
    page, (first, second) = _int_page(tmp_path, monkeypatch, ("first", "second"))
    monkeypatch.setattr(page_module, "_BROWSE_REOPEN_LIMIT_S", 1.0)
    stuck = None
    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        stuck = _stuck_release(monkeypatch)
        attempts = _begin_spy(monkeypatch, page)
        waiting = "Opening scan.nexus once the current scan finishes loading."
        with caplog.at_level(logging.INFO, logger=_PAGE_LOG):
            _click_scan(page, second)
            assert page._deferred_browse_reopen is not None
            assert page._notice_text == waiting
            _wait(lambda: True if page._deferred_browse_reopen is None and page._notice_text.startswith(
                "Browse refused") else None)

        assert page._notice_text == "Browse refused: previous Browse cleanup is pending"
        records = [record for record in caplog.records if record.name == _PAGE_LOG]
        warnings = [record for record in records if record.levelno == logging.WARNING]
        assert len(warnings) == 1 and "Browse refused: previous Browse cleanup is pending" in warnings[0].getMessage()
        assert not [record for record in records if record.exc_info or record.levelno >= logging.ERROR]
        # The click, then retries at doubling intervals (0.25 s, 0.5 s, ...) within the limit; never a storm.
        assert 1 <= len(attempts) <= 4, attempts
        assert all(value == str(second) and refusal for value, refusal, _ in attempts)
        # The waiting notice stays up while the click is kept (a cancelled open settling must not blank it).
        assert all(notice == waiting for _, _, notice in attempts[1:]), attempts
        assert _shows(page, first, 1)
    finally:
        _close(page, qapp, *(() if stuck is None else (stuck,)))


def test_a_newer_click_replaces_the_kept_one(tmp_path, monkeypatch, qapp):
    page, (first, second, third) = _int_page(tmp_path, monkeypatch, ("first", "second", "third"))
    controller = page._context_controller
    stuck = None
    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        stuck = _stuck_release(monkeypatch)
        attempts = _begin_spy(monkeypatch, page)
        _click_scan(page, second)
        assert page._deferred_browse_reopen.value == str(second)
        _wait(lambda: True if not controller.browse_pending else None)
        _click_scan(page, third)
        assert page._deferred_browse_reopen.value == str(third)

        stuck.clear()
        _wait(lambda: _shows(page, third, 1))
        assert [value for value, refusal, _ in attempts if refusal is None] == [str(third)]
        assert page._deferred_browse_reopen is None
    finally:
        _close(page, qapp, *(() if stuck is None else (stuck,)))


def test_a_kept_click_lapses_quietly_once_browse_is_not_admissible(tmp_path, monkeypatch, qapp):
    """A Run resumed (or started) while a click is kept: Browse waits for the Run, so the click lapses without a
    stale refusal. The lifecycle predicate the page reads is switched directly; a real paused Run is out of scope."""
    page, (first, second) = _int_page(tmp_path, monkeypatch, ("first", "second"))
    controller = page._context_controller
    stuck = None
    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        stuck = _stuck_release(monkeypatch)
        attempts = _begin_spy(monkeypatch, page)
        _click_scan(page, second)
        assert page._deferred_browse_reopen is not None
        monkeypatch.setattr(controller, "browse_lifecycle_admissible", lambda: False)
        stuck.clear()
        _wait(lambda: True if page._deferred_browse_reopen is None else None)
        assert [refusal for _, refusal, _ in attempts] == ["previous Browse cleanup is pending"]
        assert page._notice_text == ""
        assert _shows(page, first, 1)
    finally:
        _close(page, qapp, *(() if stuck is None else (stuck,)))


def test_refresh_cancels_the_kept_click_and_its_waiting_notice(tmp_path, monkeypatch, qapp):
    page, (first, second) = _int_page(tmp_path, monkeypatch, ("first", "second"))
    hold = threading.Event()
    real_read = Browse1DHydrationLane._open_and_read

    def held_read(self, task, labels, fact_labels):
        assert hold.wait(10)
        return real_read(self, task, labels, fact_labels)

    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        monkeypatch.setattr(Browse1DHydrationLane, "_open_and_read", held_read)
        _click_frame(qapp, page._shell, 1, settle=False)
        _wait(lambda: True if page._context_controller.browse_preview_polling_needed else None)
        _click_scan(page, second)
        waiting = page._notice_text
        assert page._deferred_browse_reopen is not None and waiting.startswith("Opening ")

        page._handle_shell_command(ShellCommand(ShellCommandKind.REFRESH_BROWSER))

        assert page._deferred_browse_reopen is None
        assert page._notice_text != waiting
        hold.set()
        controller = page._context_controller
        _wait(lambda: True if not controller.browse_pending and not controller.browse_preview_polling_needed else None)
        assert controller.navigation.current.artifact == str(first)
        assert page._deferred_browse_reopen is None
    finally:
        hold.set()
        _close(page, qapp)
