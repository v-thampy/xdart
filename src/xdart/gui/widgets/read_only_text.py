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

from pyqtgraph.Qt import QtWidgets


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
    a saved record."""

    def __init__(self, title: str, size: tuple[int, int], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(self)
        self.pane = ReadOnlyTextPane(self)
        layout.addWidget(self.pane)
        self.resize(*size)

    def set_text(self, text: str) -> None:
        """Show ``text`` in the window's pane."""
        self.pane.set_text(text)


__all__ = ["ReadOnlyTextPane", "ReadOnlyTextWindow"]
