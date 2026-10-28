"""One update offer for startup and Help: the shared read-only text window (hard rule 14) with the installed and
available versions above its pane and the choice below it."""
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .read_only_text import ReadOnlyTextWindow


class UpdateDialog(ReadOnlyTextWindow):
    """``progress_window``: False where the update runs hidden and reopens xdart itself (Linux)."""

    def __init__(self, installed, target, parent=None, *, progress_window=True):
        super().__init__("Update xdart", (620, 440), parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.notes = self.pane
        self.notes.set_text("Loading release notes…")
        self.notes.setToolTip("Scroll to read the changes in versions newer than the one installed. Text can be copied.")
        self.versions = self._label(f"Installed: {installed}    •    Available: {target}",
                                    "The running version and the newer version available for this installation.")
        self.insert_content(self.versions, 0)
        self.explanation = self._label(
            "Update closes xdart. A progress window stays open; restart when it finishes." if progress_window else
            "Update closes xdart and reopens it when the update finishes, usually within a few minutes. Nothing is "
            "shown meanwhile.",
            "Nothing is installed until you choose Update. A processing Run postpones the offer.")
        self.explanation.setWordWrap(True)
        self.insert_content(self.explanation, self.layout().count())
        buttons = QtWidgets.QDialogButtonBox()
        self.link = buttons.addButton("Open release notes", QtWidgets.QDialogButtonBox.ButtonRole.HelpRole)
        self.link.setToolTip("Read the full published release notes in your browser.")
        self._url = "https://github.com/v-thampy/xdart/releases"
        self.link.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._url)))
        self.update_button = buttons.addButton("Update", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.update_button.setToolTip("Close xdart and start the update in its progress window." if progress_window
                                      else "Close xdart, update it, and reopen it when the update finishes.")
        self.cancel_button = buttons.addButton("Cancel", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        self.cancel_button.setToolTip("Keep this version. The update can be offered again the next time xdart starts.")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.insert_content(buttons, self.layout().count())

    @staticmethod
    def _label(text, tip):
        label = QtWidgets.QLabel(text)
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        label.setToolTip(tip)
        return label

    def set_notes(self, notes):
        self.notes.set_text(notes.text)
        self._url = notes.url
