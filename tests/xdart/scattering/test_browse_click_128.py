"""BROWSE-CLICK-128: a re-click checks the shown result's file and reloads only if it changed; click timing.

Frozen before the product change (DESIGN-BROWSE-CLICK-128.md). His words: "Check and reload if changed", "Keep my
frame (Recommended)". A recheck travels as SELECT_SCAN with path ("artifact", "recheck") through the existing
BrowseLoader request owner, carrying the shown context's load revision.
"""

from __future__ import annotations

import logging
import os
import threading

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.xdart.scattering.test_browse_reopen_cleanup_pending import _click_scan, _close, _int_page, _shows
from tests.xdart.scattering.test_e4_accumulation_selection_parity import _click_frame
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from xdart.gui.tabs.scattering.adapters import browse_loader as loader_module
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _recheck(page, path) -> None:
    page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(path), path=("artifact", "recheck")))


def _context(page):
    return page._context_controller._runtime._browse


def _begin_requests(monkeypatch):
    requests = []
    real = loader_module.BrowseLoader.begin

    def spy(self, request):
        requests.append(request)
        return real(self, request)

    monkeypatch.setattr(loader_module.BrowseLoader, "begin", spy)
    return requests


def _rechecks(requests):
    return [request for request in requests if getattr(request, "expected_revision", None) is not None]


def _settled(page):
    return True if not page._context_controller.browse_pending and not page._scientific_repaint_pending else None


def _change_revision(path, kind: str) -> None:
    stat = os.stat(path)
    if kind == "mtime":
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    else:
        # Same bytes, same mtime: only the descriptor's change time moves.
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_clicking_the_selected_result_asks_for_a_recheck(tmp_path, monkeypatch, qapp):
    page, (first,) = _int_page(tmp_path, monkeypatch, ("first",))
    sent: list[ShellCommand] = []
    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        browser = page._shell.browser
        item = next(
            browser.scans.item(row) for row in range(browser.scans.count())
            if browser.scans.item(row).isSelected()
        )
        browser.commandRequested.connect(sent.append)
        QtTest.QTest.mouseClick(
            browser.scans.viewport(), QtCore.Qt.MouseButton.LeftButton,
            pos=browser.scans.visualItemRect(item).center(),
        )
        qapp.processEvents()
        rechecks = [
            command for command in sent
            if command.kind is ShellCommandKind.SELECT_SCAN and command.path == ("artifact", "recheck")
        ]
        assert [command.value for command in rechecks] == [str(first)]
    finally:
        _close(page, qapp)


def test_an_unchanged_recheck_keeps_the_context_frame_and_selection(tmp_path, monkeypatch, qapp):
    page, (first,) = _int_page(tmp_path, monkeypatch, ("first",))
    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        _click_frame(qapp, page._shell, 1)
        _wait(lambda: _shows(page, first, 2))
        context = _context(page)
        requests = _begin_requests(monkeypatch)
        _recheck(page, first)
        _wait(lambda: True if _rechecks(requests) and _settled(page) else None)
        assert [request.source_path for request in _rechecks(requests)] == [context.requested_path]
        assert _context(page) is context
        assert _shows(page, first, 2)
    finally:
        _close(page, qapp)


@pytest.mark.parametrize("kind", ("mtime", "ctime-only"))
def test_a_changed_file_reloads_on_recheck_and_keeps_the_frame(tmp_path, monkeypatch, qapp, kind):
    page, (first,) = _int_page(tmp_path, monkeypatch, ("first",))
    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        _click_frame(qapp, page._shell, 1)
        _wait(lambda: _shows(page, first, 2))
        context = _context(page)
        _change_revision(first, kind)
        _recheck(page, first)
        _wait(lambda: True if _context(page) is not context and _shows(page, first, 2) else None)
        assert _context(page).requested_path == context.requested_path
    finally:
        _close(page, qapp)


def test_a_newer_click_wins_over_a_pending_recheck(tmp_path, monkeypatch, qapp):
    page, (first, second) = _int_page(tmp_path, monkeypatch, ("first", "second"))
    entered, release = threading.Event(), threading.Event()
    real_capture = loader_module.capture_target_revision

    def held_capture(path):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(first)) and not release.is_set():
            entered.set()
            assert release.wait(20)
        return real_capture(path)

    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        _change_revision(first, "mtime")
        monkeypatch.setattr(loader_module, "capture_target_revision", held_capture)
        _recheck(page, first)
        assert entered.wait(20), "the recheck reached the loader worker"
        _click_scan(page, second)
        release.set()
        _wait(lambda: _shows(page, second, 1))
        qapp.processEvents()
        assert _context(page).requested_path.endswith(os.path.basename(str(second)))
    finally:
        release.set()
        _close(page, qapp)


def test_a_run_handoff_result_rechecks_through_its_load_revision(qapp, tmp_path, monkeypatch):
    from tests.xdart.scattering.test_predicted_ring_overlays import run_page

    with run_page(qapp, tmp_path / "handoff") as (page, _view):
        context = _context(page)
        assert context is not None and context.load_revision is not None
        requests = _begin_requests(monkeypatch)
        _recheck(page, context.requested_path)
        _wait(lambda: True if _rechecks(requests) and _settled(page) else None)
        assert _context(page) is context


def _perf_lines(caplog):
    return [record.getMessage() for record in caplog.records if "[PERF-BROWSE-CLICK]" in record.getMessage()]


def test_a_click_logs_first_trace_and_image_once_with_the_worker_stages(tmp_path, monkeypatch, qapp, caplog):
    monkeypatch.setenv("XDART_PERF", "1")
    caplog.set_level(logging.INFO)
    page, (first, second) = _int_page(tmp_path, monkeypatch, ("first", "second"))
    try:
        _click_scan(page, first)
        _click_scan(page, second)  # supersedes the first before it can display
        _wait(lambda: _shows(page, second, 1))
        _wait(lambda: True if any("image=" in line for line in _perf_lines(caplog)) else None)
        lines = _perf_lines(caplog)
        assert lines and all(str(second) in line for line in lines)
        assert len([line for line in lines if "first-trace=" in line]) == 1
        assert len([line for line in lines if "image=" in line]) == 1
        assert all("worker=" in line and "record-iteration=" in line for line in lines)
    finally:
        _close(page, qapp)
