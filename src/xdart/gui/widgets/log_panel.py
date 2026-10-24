# -*- coding: utf-8 -*-
"""The log panel: the messages written to xdart.log, under the controls of Int, Stitch and RSM (maintainer, 2026-10-23,
for 1.2.4: a checkable "Show log" button below Reintegrate/Advanced reveals a read-only log in the free space between
those buttons and the Run controls, hidden at start; "it can be dragged and made bigger, but the controls panels above
should get higher priority").

``LOG_FEED`` is a logging handler that ``_gui_main.run`` adds to the root logger beside the log file's handler, at
that handler's level (Config ▸ Debug ▸ Debug Logging sets both), so it receives the records the file gets. A logging
thread only formats its record and appends the line to a bounded history; it never touches a widget.

``LogPanel`` shows that history in a ``ReadOnlyTextPane`` (hard rule 14). A GUI-thread timer, running only while the
panel is visible, appends the lines the panel has not shown in one edit while the view is at the newest line.
Scrolled back, the view, the text and any selection are left alone (an append that trims the oldest lines would move
them) until the user returns to the bottom or presses Latest. Hidden, it does no GUI work: hiding it is what keeps
text handling off the GUI thread.

``mount_log_panel`` mounts the Show log row below the controls' Processing card, and the log under the controls'
scroll area in a split whose handle drags it taller.
"""
from __future__ import annotations

import collections
import logging
import threading

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.utils.log_events import EventFormatter

from .read_only_text import ReadOnlyTextPane

#: The log file's line format; the panel shows the time of day without the date, except on a Run's or scan's
#: banner and a significant event (``xdart.utils.log_events``).
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
#: The lines the feed keeps, and so each panel.
LOG_HISTORY_LINES = 5000
_REFRESH_MS = 250
#: The log's height in text lines where the controls leave no free space, and the least it can be dragged to.
_PANEL_LINES, _PANEL_MIN_LINES = 5, 3
#: More new lines than this rebuild the pane: an append that trims as many old lines costs more (about 30 µs a line)
#: than setting the whole history (about 7 ms for 5000 lines).
_APPEND_LINES = 200


class LogFeed(logging.Handler):
    """The log's records as text lines, newest last, numbered so that each panel reads only what it has not shown."""

    def __init__(self, capacity: int = LOG_HISTORY_LINES):
        super().__init__(logging.INFO)
        self.setFormatter(EventFormatter(LOG_FORMAT, "%H:%M:%S"))
        self.capacity = int(capacity)
        self._lines: collections.deque[str] = collections.deque(maxlen=self.capacity)
        self._end = 0  # the lines ever fed: the next line's number
        self._guard = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            self.handleError(record)
            return
        with self._guard:
            self._lines.append(line)
            self._end += 1

    @property
    def end(self) -> int:
        """The number after the newest line's."""
        return self._end

    @property
    def first(self) -> int:
        """The oldest held line's number."""
        with self._guard:
            return self._end - len(self._lines)

    def since(self, number: int) -> tuple[list[str], int, int]:
        """The held lines from ``number`` on, the number to read from next, and how many lines from ``number`` on
        the history no longer holds."""
        with self._guard:
            end, first = self._end, self._end - len(self._lines)
            lines = list(self._lines)[max(number, first) - first:] if number < end else []
        return lines, end, max(0, first - number)

    def held(self) -> tuple[list[str], int]:
        """Every held line, oldest first, and the number to read from next."""
        with self._guard:
            return list(self._lines), self._end


#: The one feed, beside the log file's handler on the root logger once ``install_log_feed`` has run.
LOG_FEED = LogFeed()


def install_log_feed() -> LogFeed:
    """Add ``LOG_FEED`` to the root logger, once."""
    root = logging.getLogger()
    if LOG_FEED not in root.handlers:
        root.addHandler(LOG_FEED)
    return LOG_FEED


class LogPanel(QtWidgets.QFrame):
    """The recent log lines in a read-only pane (``pane``), hidden until its Show log button (``toggle``) is checked.
    ``bar`` holds that button and, while the log is shown, Latest (``latest``). ``split`` holds ``above`` (the
    controls' scroll area) over the log; its handle drags the log taller."""

    def __init__(self, above: QtWidgets.QScrollArea, feed: LogFeed = LOG_FEED):
        super().__init__()
        self.setObjectName("logPanel")
        # the controls set the width; the log wraps to it
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self._feed = feed
        self._next = feed.first  # a panel opens on the recent history
        self.bar = QtWidgets.QWidget()
        self.bar.setObjectName("logPanelBar")
        row = QtWidgets.QHBoxLayout(self.bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)  # the action rows' spacing
        self.toggle = QtWidgets.QPushButton("Show log")
        self.toggle.setObjectName("controlsActionButton")
        self.toggle.setCheckable(True)
        self.toggle.toggled.connect(self._toggled)
        self.latest = QtWidgets.QPushButton("Latest")
        self.latest.setObjectName("controlsActionButton")
        self.latest.setToolTip("Scroll the log to the newest message and follow new ones again.")
        self.latest.clicked.connect(self.show_latest)
        row.addWidget(self.toggle, 1)
        row.addWidget(self.latest, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.pane = ReadOnlyTextPane(self)
        self.pane.setObjectName("logPanelText")
        self.pane.setMaximumBlockCount(feed.capacity + 1)  # and the line saying the history moved on
        self.pane.setToolTip(f"The messages written to xdart.log, newest last (up to {feed.capacity} lines). Select "
                             "text to copy it (right-click ▸ Copy). Scrolled up, new messages wait until you return "
                             "to the bottom or press Latest.")
        layout.addWidget(self.pane)
        self._at_end = True  # the view is at the newest line: new lines follow
        scroll = self.pane.verticalScrollBar()
        scroll.valueChanged.connect(self._track_end)
        scroll.rangeChanged.connect(self._keep_end)
        self.split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.split.setObjectName("logPanelSplit")
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(above)
        self.split.addWidget(self)
        self.split.setStretchFactor(0, 0)  # a taller window gives the log the room; the controls keep theirs
        self.split.setStretchFactor(1, 1)
        self.split.handle(1).setToolTip("Drag to make the log taller or shorter; the controls above keep the rest.")
        self._dragged = False  # a height the user dragged the log to is kept
        self.split.splitterMoved.connect(self._drag)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._toggled(False)

    def _toggled(self, shown: bool) -> None:
        self.toggle.setText("Hide log" if shown else "Show log")
        self.toggle.setToolTip(
            "Hide the log, so the GUI does no work for it. xdart.log keeps every message (Help ▸ Open Log "
            "Location)." if shown else
            "Show the messages written to xdart.log under these controls, newest last. Detailed diagnostics appear "
            "only while Config ▸ Debug ▸ Debug Logging is on.")
        self.latest.setVisible(bool(shown))
        self.setVisible(bool(shown))  # explicitly hidden at start, so a page shown later keeps it hidden
        if shown:  # once laid out
            QtCore.QTimer.singleShot(0, self, self._open_height)

    def _drag(self, _position: int, _index: int) -> None:
        self._dragged = True

    def _open_height(self) -> None:
        """The free space under the controls, or a few lines where they leave none; the controls keep the rest, and
        Hide log stays in view. A height the user dragged to is kept."""
        above = self.split.widget(0)
        controls = above.widget()
        if not self.isVisible() or controls is None:
            return
        if not self._dragged:
            total, width = sum(self.split.sizes()), above.viewport().width()
            needed = controls.heightForWidth(width) if controls.hasHeightForWidth() else -1
            needed = (needed if needed > 0 else controls.sizeHint().height()) + 2 * above.frameWidth()
            log = max(self._lines_height(_PANEL_LINES), total - needed)
            self.split.setSizes([total - log, log])
        if controls.isAncestorOf(self.bar):
            above.ensureWidgetVisible(self.bar, 0, 0)

    def _lines_height(self, lines: int) -> int:
        pane = self.pane
        return (lines * pane.fontMetrics().lineSpacing() + 2 * pane.frameWidth()
                + 2 * round(pane.document().documentMargin()))

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, self._lines_height(_PANEL_LINES))

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, self._lines_height(_PANEL_MIN_LINES))

    def _track_end(self, value: int) -> None:
        self._at_end = value >= self.pane.verticalScrollBar().maximum()

    def _keep_end(self, _minimum: int, maximum: int) -> None:
        # new text, or a pane resized by the split or the window: a view at the newest line stays there
        if self._at_end:
            self.pane.verticalScrollBar().setValue(maximum)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()
        self._timer.start()

    def hideEvent(self, event) -> None:
        # hidden by its button, or with its page: no refreshes until shown again; the log file is unaffected
        super().hideEvent(event)
        self._timer.stop()

    def refresh(self) -> None:
        """Show the lines logged since the last refresh, in one edit, while the view is at the newest line; scrolled
        back, count them on Latest and leave the view, the text and any selection alone."""
        bar = self.pane.verticalScrollBar()
        if not self._at_end:
            self._show_waiting(self._feed.end - self._next, True)
            return
        lines, end, missed = self._feed.since(self._next)
        if missed or len(lines) > _APPEND_LINES:
            lines, end = self._feed.held()
            missed = max(0, end - len(lines) - self._next)  # lines the history moved past before this panel showed them
            self.pane.set_text("\n".join([f"… {missed} earlier lines are only in xdart.log"] * bool(missed) + lines))
        elif lines:
            self.pane.appendPlainText("\n".join(lines))
        self._next = end
        bar.setValue(bar.maximum())
        self._show_waiting(0, False)

    def _show_waiting(self, count: int, scrolled_back: bool) -> None:
        self.latest.setText(f"Latest ({count} new)" if count else "Latest")
        self.latest.setEnabled(scrolled_back)

    def show_latest(self) -> None:
        """Scroll to the newest line and follow new ones again."""
        self._at_end = True  # all text may fit already, so setValue need not emit valueChanged
        bar = self.pane.verticalScrollBar()
        bar.setValue(bar.maximum())
        self.refresh()


def mount_log_panel(controls, above: QtWidgets.QScrollArea) -> LogPanel:
    """Int's, Stitch's or RSM's log: the Show log row below the controls' Processing card (under its
    Reintegrate/Advanced row), and the log under ``above``, the controls' scroll area. The page puts the panel's
    ``split`` where it would have put ``above``, over its Run controls."""
    panel = LogPanel(above)
    controls.add_footer_widget(panel.bar)
    return panel


__all__ = ["LOG_FEED", "LOG_FORMAT", "LOG_HISTORY_LINES", "LogFeed", "LogPanel", "install_log_feed",
           "mount_log_panel"]
