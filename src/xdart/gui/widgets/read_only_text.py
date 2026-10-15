# -*- coding: utf-8 -*-
"""Read-only text, and a window of it: every page's, tool's and pop-up's one owner for text the user reads but
does not edit (hard rule 14; Codex's combined stage-2b review F2, 2026-10-01).

Its consumers: RSM's and Stitch's Preview windows, Stitch's saved-provenance window, and the shared calibration
window's fit and direct-beam summaries. ``ReadOnlyTextPane`` is a ``QPlainTextEdit`` set read-only, so the
themes' input rules style it as before and its consumers keep the plain-text calls they make (``setPlainText``,
``appendPlainText``, ``toPlainText``, its scroll bars). ``ReadOnlyTextWindow`` is a non-modal window holding one
pane, with its title and size.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets


class ReadOnlyTextPane(QtWidgets.QPlainTextEdit):
    """Text the user reads, selects and copies, but does not edit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    def set_text(self, text: str) -> None:
        """Show ``text`` in place of what was shown."""
        self.setPlainText(str(text))


class ReadOnlyTextWindow(QtWidgets.QDialog):
    """A non-modal window of read-only text (``pane``, a ``ReadOnlyTextPane``) with its title and size: a Preview,
    a saved record. ``message_lines`` (opt-in, LT-007): the content mounted before the pane and the pane share the
    window in an adjustable vertical split, the pane opening that many text lines tall."""

    def __init__(self, title: str, size: tuple[int, int], parent=None, *, message_lines: int = 0):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(self)
        self.pane = ReadOnlyTextPane(self)
        self._split, self._message_lines = None, int(message_lines)
        if self._message_lines:
            self._split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
            self._split.setChildrenCollapsible(False)
            self._split.addWidget(self.pane)
            layout.addWidget(self._split)
        else:
            layout.addWidget(self.pane)
        self.resize(*size)

    def set_text(self, text: str) -> None:
        """Show ``text`` in the window's pane."""
        self.pane.set_text(text)

    def insert_content(self, widget, index: int = 0) -> None:
        """Mount another shared owner before the text, without changing the plain window contract; in a split window,
        above the pane in the split."""
        if self._split is None:
            self.layout().insertWidget(index, widget)
            return
        self._split.insertWidget(index, widget)
        self._split.setStretchFactor(self._split.indexOf(widget), 1)
        self._split.setStretchFactor(self._split.indexOf(self.pane), 0)
        handle = self._split.handle(self._split.indexOf(self.pane))
        handle.setToolTip('Drag to share the height between the views above and the messages below.')

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._split is not None and self._message_lines:  # the pane opens message_lines tall, once
            lines, self._message_lines = self._message_lines, 0
            pane = self.pane
            height = (lines * pane.fontMetrics().lineSpacing() + 2 * pane.frameWidth()
                      + 2 * round(pane.document().documentMargin()))
            sizes, index = self._split.sizes(), self._split.indexOf(pane)
            others = max(len(sizes) - 1, 1)
            rest = max(sum(sizes) - height, others)
            self._split.setSizes([height if i == index else rest // others for i in range(len(sizes))])

    def insert_content_row(self, widgets, index: int = 0) -> None:
        """Mount shared controls beside each other without adding another editor/display."""
        row = QtWidgets.QHBoxLayout()
        for widget in widgets:
            row.addWidget(widget)
        self.layout().insertLayout(index, row)


__all__ = ["ReadOnlyTextPane", "ReadOnlyTextWindow"]
