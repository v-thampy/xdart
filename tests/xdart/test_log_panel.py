"""The log panel (``widgets/log_panel.py``; maintainer, 2026-10-23, for 1.2.4): a checkable Show log button below
Reintegrate/Advanced reveals the messages written to xdart.log in a read-only pane in the free space between those
buttons and the Run controls, hidden at start, on Int, Stitch and RSM alike (hard rule 14: one shared component, its
text box the shared ReadOnlyTextPane). "It can be dragged and made bigger, but the controls panels above should get
higher priority"; hidden, it does no GUI work.

The feed is a real logging handler fed by real logger calls; the panels are the real pages' panels."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtCore, QtGui, QtTest, QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.widgets import log_panel as log_panel_module
from xdart.gui.widgets.log_panel import LOG_FEED, LogFeed, LogPanel, install_log_feed
from xdart.gui.widgets.read_only_text import ReadOnlyTextPane
from xrd_tools.session.readiness import SectionId

from tests.xdart.scattering.e3_shell_support import make_shell_projection

_REFRESH_S = log_panel_module._REFRESH_MS / 1000


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def root_feed():
    """``LOG_FEED`` on the root logger, which takes INFO, as ``_gui_main.run`` leaves them."""
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.INFO)
    install_log_feed()
    try:
        yield LOG_FEED
    finally:
        root.removeHandler(LOG_FEED)
        root.setLevel(level)


class _Status:
    def show(self, text, timeout_ms=0):
        pass


def _fed_logger(name, feed):
    """A logger that only ``feed`` hears, at INFO."""
    log = logging.getLogger(name)
    log.handlers[:] = [feed]
    log.propagate = False
    log.setLevel(logging.INFO)
    return log


def _settle(qapp, seconds=0.05):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        qapp.processEvents()
        time.sleep(0.002)


def _shell(qapp, height=1080):
    from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
    shell = ScatteringWorkspaceShell()
    shell.resize(1488, height)
    shell.show()
    shell.apply_state(make_shell_projection(frame_count=5))
    previous = None
    for _ in range(50):  # the columns settle over a few event-loop passes after the first show
        _settle(qapp, 0.02)
        current = (shell.splitter.sizes(), shell.controls.size())
        if current == previous:
            break
        previous = current
    return shell


def _dispose(shell, qapp):
    shell.close()
    shell.deleteLater()
    qapp.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _traces(shell):
    return tuple((np.array(item.xData, copy=True), np.array(item.yData, copy=True))
                 for item in shell.scientific.curve.listDataItems())


def _cards(controls):
    return [card.geometry() for card in (controls.project_card, controls.experiment_card, controls.source_card,
                                         controls.processing_card)]


def _bar_in_view(panel, scroll):
    top = panel.bar.mapTo(scroll.viewport(), QtCore.QPoint(0, 0)).y()
    return 0 <= top and top + panel.bar.height() <= scroll.viewport().height()


def _drag(handle, dy):
    """The user drags the split's handle ``dy`` pixels (up when negative)."""
    centre = handle.rect().center()
    QtTest.QTest.mousePress(handle, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, centre)
    QtTest.QTest.mouseMove(handle, centre + QtCore.QPoint(0, dy))
    QtTest.QTest.mouseRelease(handle, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
                              centre + QtCore.QPoint(0, dy))


def test_the_feed_numbers_a_bounded_history_that_logging_threads_append_to():
    feed = LogFeed(capacity=5)
    log = _fed_logger("xdart.test.log_feed", feed)
    try:
        assert feed.formatter._fmt == log_panel_module.LOG_FORMAT and feed.level == logging.INFO
        log.debug("not at INFO")
        for index in range(3):
            log.info("line %d", index)
        lines, end, missed = feed.since(feed.first)
        assert [line.split(": ", 1)[1] for line in lines] == ["line 0", "line 1", "line 2"]
        assert (end, missed) == (3, 0) and " INFO xdart.test.log_feed: " in lines[0]
        for index in range(3, 7):
            log.info("line %d", index)
        lines, after, missed = feed.since(end)
        assert [line.split(": ", 1)[1] for line in lines] == [f"line {i}" for i in range(3, 7)]
        assert (after, missed) == (7, 0)
        lines, _, missed = feed.since(0)  # the history keeps the newest five
        assert missed == 2 and [line.split(": ", 1)[1] for line in lines] == [f"line {i}" for i in range(2, 7)]
        assert feed.since(after) == ([], 7, 0) and feed.held()[1] == 7

        threads = [threading.Thread(target=lambda t=t: [log.info("thread %d %d", t, i) for i in range(500)])
                   for t in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        held, end = feed.held()
        assert end == 7 + 2000 and len(held) == 5 and feed.first == end - 5
    finally:
        log.handlers[:] = []


@pytest.mark.parametrize("page_kind", ("int", "stitch", "rsm"))
def test_each_page_mounts_the_shared_log_under_its_controls(qapp, page_kind):
    """The same component on every page: its Show log row right after the Processing card (under
    Reintegrate/Advanced), in the controls' own layout; the log under the controls' scroll area in a split that takes
    the scroll area's place over the Run controls. Hidden at start, each part with a tooltip, its one text box the
    shared read-only pane."""
    from xdart.gui.pages.values import PageCleanup
    if page_kind == "int":
        from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
        page = ScatteringWorkspaceShell()
        controls, run_bar = page.controls, page.run_controls
    else:
        if page_kind == "rsm":
            from xdart.gui.tabs.rsm.page import RSMPage as Page
        else:
            from xdart.gui.tabs.stitch.page import StitchPage as Page
        page = Page(empty_host_services(_Status()))
        controls, run_bar = page.controls, page.run_bar
    try:
        panel = page.log_panel
        assert type(panel) is LogPanel and type(panel.pane) is ReadOnlyTextPane and panel.pane.isReadOnly()
        assert panel.findChildren(QtWidgets.QPlainTextEdit) == [panel.pane]
        root = controls._root_layout
        assert panel.bar.parent() is controls and panel.toggle.parent() is panel.bar
        assert panel.latest.parent() is panel.bar
        processing = root.indexOf(controls.processing_card)
        assert root.indexOf(panel.bar) == processing + 1 and root.count() == processing + 3
        assert root.itemAt(root.count() - 1).spacerItem() is not None
        split = panel.split
        assert type(split) is QtWidgets.QSplitter and split.orientation() is QtCore.Qt.Orientation.Vertical
        assert split.count() == 2 and split.widget(0).widget() is controls and split.widget(1) is panel
        assert not split.childrenCollapsible()
        column = run_bar.parentWidget().layout()
        assert column.count() == 2 and column.itemAt(0).widget() is split and column.itemAt(1).widget() is run_bar
        assert panel.isHidden() and panel.latest.isHidden() and not panel.toggle.isChecked()
        assert panel.toggle.isCheckable() and panel.toggle.text() == "Show log"
        for part in (panel.toggle, panel.latest, panel.pane, split.handle(1)):
            assert part.toolTip()
    finally:
        if page_kind == "int":
            _dispose(page, qapp)
        else:
            close = page.close_page if page_kind == "rsm" else page.shutdown
            assert close().status is PageCleanup.CLEAN
            page.deleteLater()
            qapp.processEvents()


def test_show_log_takes_only_the_free_space_and_hide_log_gives_it_back(qapp):
    """The maintainer's screenshot: free space between Reintegrate/Advanced and the Run controls. Shown, the log takes
    that space and no more; the controls keep their whole height (no scrolling), and the cards, the column widths and
    the fixed Run controls are unchanged. Hidden, the space is the controls' again."""
    shell = _shell(qapp, height=1600)
    try:
        controls, panel, run_bar, scroll = shell.controls, shell.log_panel, shell.run_controls, shell.control_scroll
        cards, sizes, run_geometry = _cards(controls), shell.splitter.sizes(), run_bar.geometry()
        column = sum(panel.split.sizes())  # the controls' height while the log is hidden
        needed = controls.sizeHint().height() + 2 * scroll.frameWidth()
        assert column - needed > panel.sizeHint().height()
        QtTest.QTest.mouseClick(panel.toggle, QtCore.Qt.MouseButton.LeftButton)
        _settle(qapp)
        assert panel.isVisible() and panel.toggle.isChecked() and panel.toggle.text() == "Hide log"
        assert panel.latest.isVisible() and panel._timer.isActive()
        total = column - panel.split.handleWidth()
        assert panel.split.sizes() == [needed, total - needed]
        assert scroll.verticalScrollBar().maximum() == 0 and _bar_in_view(panel, scroll)
        assert _cards(controls) == cards and shell.splitter.sizes() == sizes and run_bar.geometry() == run_geometry
        assert scroll.horizontalScrollBar().maximum() == 0
        shell.resize(1488, 1400)  # the controls keep priority: a smaller window takes from the log
        _settle(qapp)
        assert panel.split.sizes() == [needed, total - needed - 200]
        shell.resize(1488, 1600)
        _settle(qapp)
        assert panel.split.sizes() == [needed, total - needed]
        QtTest.QTest.mouseClick(panel.toggle, QtCore.Qt.MouseButton.LeftButton)
        _settle(qapp)
        assert panel.isHidden() and panel.latest.isHidden() and panel.toggle.text() == "Show log"
        assert not panel._timer.isActive() and panel.split.sizes() == [column, 0]
        assert _cards(controls) == cards and run_bar.geometry() == run_geometry
    finally:
        _dispose(shell, qapp)


def test_where_the_controls_fill_the_column_the_log_opens_small_and_drags_taller(qapp):
    """No free space (this window): the log opens a few lines tall and the controls keep the rest, scrolled so that
    Hide log stays in view. The handle drags the log taller; hidden and shown again, it keeps the dragged height. A
    projection rebuild or Stitch's/RSM's run lock leaves the button, its state and its wiring."""
    shell = _shell(qapp, height=1080)
    try:
        controls, panel, run_bar, scroll = shell.controls, shell.log_panel, shell.run_controls, shell.control_scroll
        run_geometry, column = run_bar.geometry(), sum(panel.split.sizes())
        assert column < controls.sizeHint().height() + 2 * scroll.frameWidth() + panel.sizeHint().height()
        panel.toggle.click()
        _settle(qapp)
        total, small = column - panel.split.handleWidth(), panel._lines_height(log_panel_module._PANEL_LINES)
        assert panel.split.sizes() == [total - small, small] and panel.height() == small
        assert _bar_in_view(panel, scroll) and run_bar.geometry() == run_geometry
        _drag(panel.split.handle(1), -120)
        _settle(qapp)
        assert panel.split.sizes() == [total - small - 120, small + 120]
        panel.toggle.click()
        _settle(qapp)
        assert panel.split.sizes() == [column, 0]
        panel.toggle.click()
        _settle(qapp)
        assert panel.split.sizes() == [total - small - 120, small + 120]
        controls.set_editors_enabled(False)  # Stitch's and RSM's run lock
        assert panel.toggle.isEnabled()
        controls.set_editors_enabled(True)
        projection = controls.projection  # a schema rebuild clears the cards' rows
        actions = dict(projection.section_actions)
        actions[SectionId.PROCESSING] = actions[SectionId.PROCESSING][:-1]
        assert controls.reconcile(replace(projection, section_actions=actions)) is False
        root = controls._root_layout
        assert root.indexOf(panel.bar) == root.indexOf(controls.processing_card) + 1
        assert panel.toggle.isChecked() and panel.isVisible()
        QtTest.QTest.mouseClick(panel.toggle, QtCore.Qt.MouseButton.LeftButton)
        assert panel.isHidden() and panel.toggle.text() == "Show log" and run_bar.geometry() == run_geometry
    finally:
        _dispose(shell, qapp)


def test_the_panel_follows_the_newest_line_and_holds_a_scrolled_back_view(qapp):
    feed = LogFeed(capacity=300)
    log = _fed_logger("xdart.test.log_follow", feed)
    panel = LogPanel(QtWidgets.QScrollArea(), feed)
    panel.split.resize(320, 400)
    panel.split.show()
    try:
        panel.toggle.setChecked(True)
        _settle(qapp)
        for index in range(50):
            log.info("first %d", index)
        panel.refresh()
        pane, bar = panel.pane, panel.pane.verticalScrollBar()
        assert pane.blockCount() == 50 and bar.value() == bar.maximum() > 0
        assert panel.latest.text() == "Latest" and not panel.latest.isEnabled()

        bar.setValue(0)  # scrolled back, with a selection: neither moves while lines arrive
        cursor = pane.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(5, QtGui.QTextCursor.MoveMode.KeepAnchor)
        pane.setTextCursor(cursor)
        for index in range(10):
            log.info("second %d", index)
        panel.refresh()
        assert pane.blockCount() == 50 and bar.value() == 0
        assert pane.textCursor().selectedText() == pane.document().firstBlock().text()[:5]
        assert panel.latest.text() == "Latest (10 new)" and panel.latest.isEnabled()
        panel.latest.click()
        assert pane.blockCount() == 60 and bar.value() == bar.maximum()
        assert pane.document().lastBlock().text().endswith("second 9") and not panel.latest.isEnabled()

        # A paused view can be enlarged until all text fits: the scroll value
        # stays zero, so Latest must resume following without a valueChanged.
        bar.setValue(0)
        panel.refresh()
        assert panel.latest.isEnabled()
        panel.split.resize(320, 4000)
        _settle(qapp)
        assert bar.value() == bar.maximum() == 0
        panel.latest.click()
        assert not panel.latest.isEnabled()
        panel.split.resize(320, 400)
        _settle(qapp)

        for index in range(250):  # more at once than an append should trim: the pane is rebuilt, still bounded
            log.info("third %d", index)
        panel.refresh()
        assert pane.blockCount() == 300 and pane.document().firstBlock().text().endswith("first 10")
        assert bar.value() == bar.maximum()

        panel.toggle.setChecked(False)  # hidden: no refreshes, the history moves on
        assert panel.isHidden() and not panel._timer.isActive()
        for index in range(400):
            log.info("fourth %d", index)
        assert pane.document().lastBlock().text().endswith("third 249")
        panel.toggle.setChecked(True)  # shown again: the history it holds, and how much it no longer holds
        assert panel._timer.isActive() and pane.blockCount() == 301
        assert pane.document().firstBlock().text() == "… 100 earlier lines are only in xdart.log"
        assert pane.document().lastBlock().text().endswith("fourth 399") and bar.value() == bar.maximum()
    finally:
        log.handlers[:] = []
        panel.bar.deleteLater()
        panel.split.deleteLater()
        qapp.processEvents()


def test_the_log_keeps_following_the_newest_line_when_its_pane_is_resized(qapp, root_feed):
    """The pane is first refreshed before its layout settles its height, then a smaller window and a dragged handle
    resize it: a view at the newest line stays there at once, before any refresh, and new lines keep showing (before
    this, it stopped at "Latest (N new)")."""
    log = logging.getLogger("xdart.test.log_resize")
    for index in range(300):
        log.info("before %d", index)
    shell = _shell(qapp, height=1080)
    try:
        panel = shell.log_panel
        bar = panel.pane.verticalScrollBar()
        panel.toggle.click()
        for step, tag in enumerate(("shown", "smaller", "dragged")):
            _settle(qapp, 2 * _REFRESH_S)
            panel._timer.stop()  # no refresh to put the view back: the resize itself must leave it at the newest line
            if step == 1:
                shell.resize(1488, 880)
            elif step == 2:
                _drag(panel.split.handle(1), -150)
            _settle(qapp, 0.1)
            assert bar.value() == bar.maximum() > 0
            panel._timer.start()
            for index in range(30):
                log.info("%s %d", tag, index)
            _settle(qapp, 2.5 * _REFRESH_S)
            assert panel.pane.document().lastBlock().text().endswith(f"xdart.test.log_resize: {tag} 29")
            assert bar.value() == bar.maximum() > 0
            assert panel.latest.text() == "Latest" and not panel.latest.isEnabled()
    finally:
        _dispose(shell, qapp)


def test_a_log_on_a_hidden_page_stops_refreshing_and_catches_up_when_shown(qapp, root_feed):
    shell = _shell(qapp)
    log = logging.getLogger("xdart.test.log_page")
    try:
        panel = shell.log_panel
        panel.toggle.setChecked(True)
        _settle(qapp)
        assert panel._timer.isActive()
        shell.hide()  # another page in front
        assert not panel._timer.isActive() and panel.toggle.text() == "Hide log"
        text = panel.pane.toPlainText()
        log.info("while the page was hidden")
        _settle(qapp, 2 * _REFRESH_S)
        assert panel.pane.toPlainText() == text
        shell.show()
        assert panel._timer.isActive() and panel.isVisible()
        assert panel.pane.document().lastBlock().text().endswith("xdart.test.log_page: while the page was hidden")
    finally:
        _dispose(shell, qapp)


def _burst(qapp, log, tag, during=None, threads=4, records=2500):
    """``threads`` threads log ``records`` records each through the real root logger while the GUI runs a 10 ms
    heartbeat, and ``during`` runs once they have started; returns the largest heartbeat gap in seconds and the
    seconds the burst took."""
    ticks = []
    heartbeat = QtCore.QTimer()
    heartbeat.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: ticks.append(time.perf_counter()))
    workers = [threading.Thread(target=lambda t=t: [log.info("%s burst thread %d record %d", tag, t, i)
                                                    for i in range(records)]) for t in range(threads)]
    heartbeat.start()
    start = time.perf_counter()
    for worker in workers:
        worker.start()
    if during is not None:
        during()
    while any(worker.is_alive() for worker in workers):
        qapp.processEvents()
        time.sleep(0.001)
    took = time.perf_counter() - start
    _settle(qapp, 2.5 * _REFRESH_S)  # two refreshes after the last record
    heartbeat.stop()
    for worker in workers:
        worker.join()
    return float(np.diff(ticks).max(initial=0.0)), took


def test_a_burst_of_logging_keeps_controls_responsive_and_plots_unchanged_shown_or_hidden(
        qapp, monkeypatch, tmp_path, root_feed):
    """Production wiring (the file handler and the feed on the root logger, as ``run`` installs them) and the real
    Int shell. 4 × 2500 records from worker threads with the log shown, then hidden: the GUI heartbeat stays within
    the shell's 0.25 s bar, a Run-control click mid-burst is delivered, the pane is edited at most once a refresh
    (shown) or never (hidden), the file gets every record, and the plotted traces and frame selection are unchanged."""
    from xdart import _gui_main
    from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
    monkeypatch.delenv("XDART_LOG_FILE", raising=False)
    monkeypatch.setenv("XDART_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", None)
    monkeypatch.setattr(_gui_main, "_FILE_LOG_HANDLER", None)
    path = _gui_main._install_file_log_handler()
    shell = _shell(qapp)
    log = logging.getLogger("xdart.test.log_burst")
    try:
        panel = shell.log_panel
        traces, operations = _traces(shell), shell.scientific._selector_operations
        frame = shell.scientific.frame_selector.currentIndex()
        assert traces
        commands = []
        shell.commandRequested.connect(commands.append)
        edits = []
        for name in ("appendPlainText", "set_text"):  # count the pane's edits, still made by the pane
            method = getattr(panel.pane, name)
            monkeypatch.setattr(panel.pane, name, lambda text, method=method: (edits.append(1), method(text)))

        panel.toggle.setChecked(True)
        _settle(qapp)
        edits.clear()
        gap, took = _burst(qapp, log, "shown", during=lambda: QtTest.QTest.mouseClick(
            shell.run_controls.batchButton, QtCore.Qt.MouseButton.LeftButton))
        assert gap < 0.25, f"GUI heartbeat stalled {gap:.3f} s with the log shown"
        assert [command.kind for command in commands] == [ShellCommandKind.SET_BATCH]
        assert 1 <= len(edits) <= (took + 2.5 * _REFRESH_S) / _REFRESH_S + 2
        last = panel.pane.document().lastBlock().text()
        assert panel.pane.blockCount() <= LOG_FEED.capacity + 1
        assert " shown burst thread " in last and last.endswith(" record 2499")

        panel.toggle.setChecked(False)
        _settle(qapp)
        text, edits[:] = panel.pane.toPlainText(), []
        gap, _took = _burst(qapp, log, "hidden")
        assert gap < 0.25, f"GUI heartbeat stalled {gap:.3f} s with the log hidden"
        assert edits == [] and panel.pane.toPlainText() == text and not panel._timer.isActive()

        _gui_main._FILE_LOG_HANDLER.flush()
        written = path.read_text(encoding="utf-8")
        assert written.count(" burst thread ") == 2 * 4 * 2500
        assert all(f"{tag} burst thread {t} record 2499" in written for tag in ("shown", "hidden") for t in range(4))
        after = _traces(shell)
        assert len(after) == len(traces)
        for (x0, y0), (x1, y1) in zip(traces, after):
            np.testing.assert_array_equal(x0, x1)
            np.testing.assert_array_equal(y0, y1)
        assert shell.scientific._selector_operations == operations
        assert shell.scientific.frame_selector.currentIndex() == frame
    finally:
        _dispose(shell, qapp)
        root = logging.getLogger()
        for handler in [h for h in root.handlers if isinstance(h, _gui_main.RotatingFileHandler)]:
            root.removeHandler(handler)
            handler.close()


def test_debug_logging_and_the_log_panel_are_independent(qapp, monkeypatch, tmp_path, root_feed):
    """Show log changes no logging level. Config ▸ Debug ▸ Debug Logging sets the feed's level with the file's, so
    the panel lists DEBUG records exactly while the file gets them, shown or hidden; ``run``'s install is once."""
    from xdart import _gui_main
    monkeypatch.delenv("XDART_LOG_FILE", raising=False)
    monkeypatch.setenv("XDART_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", None)
    monkeypatch.setattr(_gui_main, "_FILE_LOG_HANDLER", None)
    for flag in _gui_main.DEBUG_LOGGING_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    for name in ("xdart", "xrd_tools"):
        monkeypatch.setattr(logging.getLogger(name), "level", logging.getLogger(name).level)
    monkeypatch.setattr(LOG_FEED, "level", LOG_FEED.level)
    root = logging.getLogger()
    path = _gui_main._install_file_log_handler()
    panel = LogPanel(QtWidgets.QScrollArea())
    try:
        file_handler = _gui_main._FILE_LOG_HANDLER
        assert _gui_main.install_log_feed() is LOG_FEED and root.handlers.count(LOG_FEED) == 1
        assert file_handler.formatter._fmt == LOG_FEED.formatter._fmt == log_panel_module.LOG_FORMAT

        def levels():
            return (logging.getLogger("xdart").level, logging.getLogger("xrd_tools").level, file_handler.level,
                    LOG_FEED.level)

        before = levels()
        panel.toggle.setChecked(True)
        panel.toggle.setChecked(False)
        assert levels() == before and LOG_FEED.level == file_handler.level == logging.INFO
        probe, start = logging.getLogger("xrd_tools.io.metadata"), LOG_FEED.end
        probe.debug("hidden, before the switch")
        _gui_main.set_debug_logging(True)
        assert LOG_FEED.level == file_handler.level == logging.DEBUG
        probe.debug("hidden, while debugging")
        panel.toggle.setChecked(True)
        probe.debug("shown, while debugging")
        panel.toggle.setChecked(False)
        assert LOG_FEED.level == logging.DEBUG
        _gui_main.set_debug_logging(False)
        probe.debug("after the switch")
        assert LOG_FEED.level == file_handler.level == logging.INFO
        listed = [line.split(": ", 1)[1] for line in LOG_FEED.since(start)[0] if " xrd_tools.io.metadata: " in line]
        assert listed == ["hidden, while debugging", "shown, while debugging"]
        file_handler.flush()
        written = [line.split(": ", 1)[1] for line in path.read_text(encoding="utf-8").splitlines()
                   if " xrd_tools.io.metadata: " in line]
        assert written == listed
    finally:
        panel.bar.deleteLater()
        panel.split.deleteLater()
        qapp.processEvents()
        for handler in [h for h in root.handlers if isinstance(h, _gui_main.RotatingFileHandler)]:
            root.removeHandler(handler)
            handler.close()
