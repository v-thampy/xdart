"""Offline help routing and one modeless shared-text window per participating root.

Tags are passive; buttons resolve their top-level owner at click time so retained
pages can be mounted in Main without creating a second shortcut or window.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xrd_tools.help_topics import TOPICS, get_topic, search_topics
from .read_only_text import ReadOnlyTextWindow


def bind_topic(widget, topic_id: str, *, details=None) -> None:
    """Tag a shared control; optional details reads only already displayed strings."""
    widget.setProperty("helpTopic", str(topic_id))
    if details is not None:
        widget._help_details = details


def _context(widget, root):
    topic = None
    details = ""
    while widget is not None:
        if not details and hasattr(widget, "_help_details"):
            details = str(widget._help_details())
        if topic is None and widget.property("helpTopic"):
            topic = str(widget.property("helpTopic"))
        if widget is root:
            break
        widget = widget.parentWidget()
    return topic, details


def preserve_focused_draft(root):
    """Help focus transfer is presentation, not a FormRow editingFinished commit."""
    widget = QtWidgets.QApplication.focusWidget()
    while widget is not None and widget.window() is root:
        if hasattr(widget, "preserve_help_draft"):
            widget.preserve_help_draft()
            break
        widget = widget.parentWidget()


class HelpWindow(ReadOnlyTextWindow):
    """Compose Int's text, keyed browser and form row; no alternative editors."""

    def __init__(self, parent):
        super().__init__("Scientific help", (920, 700), parent)
        # Lazy composition avoids a module-level widgets → tabs/scattering edge.
        from .controls_panel import FormRow
        from xdart.gui.tabs.scattering.browser_view import BrowserResult, BrowserView, keyed_list_profile
        from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
        self._entry_type = BrowserResult
        self.topic_id = None
        self.search_row = FormRow(label="Search help", path=("help", "search"), value="")
        self.search_row.set_purpose_tooltip("Search topic IDs, titles and text; every entered word must match.")
        self.browser = BrowserView(self, profile=keyed_list_profile("Help topics"))
        self.browser.scans_label.setToolTip("Select an offline topic to read its instructions.")
        self.browser.scans.setToolTip("Matching topics in catalogue order; selecting one changes only help text.")
        self.browser.setMaximumHeight(160)
        self.pane.setToolTip("Read, select and copy these instructions and literal diagnostics; text cannot be edited.")
        self.insert_content(self.search_row)
        self.insert_content(self.browser, 1)
        self.search_row.editor.textChanged.connect(self._search)
        self.browser.commandRequested.connect(lambda command:
            self.show_topic(command.value) if command.kind is ShellCommandKind.SELECT_RESULT else None)
        self._f1 = QtGui.QShortcut(QtGui.QKeySequence("F1"), self)
        self._f1.activated.connect(self.show_index)
        self.show_index()

    def _search(self, query: str) -> None:
        topics = search_topics(query)
        self.browser.reconcile_results(tuple(self._entry_type(t.identifier, t.title, t.identifier)
                                             for t in topics), self.topic_id)

    def show_index(self, requested: str | None = None, details: str = "") -> None:
        self.topic_id = None
        self.search_row.editor.clear()
        self._search("")
        text = "Scientific help\n\nSearch or select a topic. F1 opens help for the focused control.\n\n"
        if requested:
            text += f"Unknown topic: {requested}\n\n"
        text += "\n".join(f"{t.title} — {t.identifier}" for t in TOPICS)
        self.set_text(text + ("\n\nCurrent context\n" + details if details else ""))
        self.search_row.editor.setFocus()

    def show_topic(self, identifier: str, details: str = "") -> None:
        topic = get_topic(identifier)
        if topic is None:
            self.show_index(identifier, details)
            return
        self.topic_id = identifier
        self._search(self.search_row.editor.text())
        self.set_text(topic.title + "\n" + topic.identifier + "\n\n" + topic.text +
                      ("\n\nCurrent context\n" + details if details else ""))


class ContextHelpOwner(QtCore.QObject):
    """Own help and its F1 binding only for one top-level widget's lifetime."""

    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.window = None
        self.shortcut = None
        self._held_row = None
        root.installEventFilter(self)
        self._bind_shortcut()

    def _bind_shortcut(self):
        if self.shortcut is None:
            self.shortcut = QtGui.QShortcut(QtGui.QKeySequence("F1"), self.root)
            self.shortcut.activated.connect(self.show_focused)

    def eventFilter(self, watched, event):
        if watched is self.root:
            if event.type() == QtCore.QEvent.Type.Hide:
                self.shutdown()
            elif event.type() == QtCore.QEvent.Type.Show:
                self._bind_shortcut()
            elif event.type() == QtCore.QEvent.Type.WindowActivate:
                # Run after Qt restores the root's former focus widget.
                QtCore.QTimer.singleShot(0, self, self._resume_draft)
        elif watched is self.window and event.type() == QtCore.QEvent.Type.Hide:
            self.root.activateWindow()
            QtCore.QTimer.singleShot(0, self, self._resume_draft)
        return False

    def hold_draft(self, row):
        """One row reference for this visit; FormRow remains the sole value owner."""
        if self._held_row is row:
            return
        self._resume_draft(restore=False)
        self._held_row = row
        row.destroyed.connect(self._forget_draft)

    def _forget_draft(self):
        if self.sender() is self._held_row:
            self._held_row = None

    def _resume_draft(self, *, restore=True):
        row = self._held_row
        if row is None:
            return
        self._held_row = None
        row.destroyed.disconnect(self._forget_draft)
        held = row._help_focus_transfer
        row._help_focus_transfer = False
        if restore and held and row.editor.isVisible() and row.editor.isEnabled():
            row.editor.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _open(self):
        preserve_focused_draft(self.root)
        self._bind_shortcut()
        if self.window is None:
            self.window = HelpWindow(self.root)
            self.window.installEventFilter(self)
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        return self.window

    def show_index(self):
        self._open().show_index()

    def preserve_draft(self):
        preserve_focused_draft(self.root)

    def show_topic(self, identifier: str, details: str = ""):
        self._open().show_topic(identifier, details)

    def show_context(self, widget):
        topic, details = _context(widget, self.root)
        if topic:
            self.show_topic(topic, details)
        else:
            self._open().show_index(details=details)

    def show_focused(self):
        widget = QtWidgets.QApplication.focusWidget()
        self.show_context(widget if widget is not None and widget.window() is self.root else self.root)

    def shutdown(self):
        """Release callbacks/windows on actual retirement; ignored busy Close keeps both."""
        self._resume_draft(restore=False)
        if self.shortcut is not None:
            self.shortcut.setEnabled(False)
            self.shortcut.deleteLater()
            self.shortcut = None
        if self.window is not None:
            self.window.removeEventFilter(self)
            self.window.close()
            self.window.deleteLater()
            self.window = None


def owner_for(widget) -> ContextHelpOwner:
    root = widget.window()
    owner = getattr(root, "_context_help_owner", None)
    if owner is None:
        owner = root._context_help_owner = ContextHelpOwner(root)
    return owner
