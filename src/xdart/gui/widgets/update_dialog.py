"""One update offer for startup and Help: the shared read-only text window (hard rule 14) with the installed and
available versions above its pane and the choice below it."""
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .read_only_text import ReadOnlyTextWindow


class UpdateDialog(ReadOnlyTextWindow):
    """``manual``: this system has no update progress window (Linux), so the dialog's button shows the update command
    instead of updating."""

    def __init__(self, installed, target, parent=None, *, manual=False):
        super().__init__("Update xdart", (620, 440), parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.notes = self.pane
        self.notes.set_text("Loading release notes…")
        self.notes.setToolTip("Scroll to read the changes in versions newer than the one installed. Text can be copied.")
        self.versions = self._label(f"Installed: {installed}    •    Available: {target}",
                                    "The running version and the newer version available for this installation.")
        self.insert_content(self.versions, 0)
        if manual:
            self.explanation = self._label(
                "xdart cannot show an update window on this system. Show Update Command gives the command to run in "
                "a terminal after closing xdart.",
                "Nothing is installed by xdart here; the command updates this installation when you run it.")
        else:
            self.explanation = self._label(
                "Update closes xdart. A progress window stays open; restart when it finishes.",
                "Nothing is installed until you choose Update. A processing Run postpones the offer.")
        self.explanation.setWordWrap(True)
        self.insert_content(self.explanation, self.layout().count())
        buttons = QtWidgets.QDialogButtonBox()
        self.link = buttons.addButton("Open release notes", QtWidgets.QDialogButtonBox.ButtonRole.HelpRole)
        self.link.setToolTip("Read the full published release notes in your browser.")
        self._url = "https://github.com/v-thampy/xdart/releases"
        self.link.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._url)))
        self.update_button = buttons.addButton("Show Update Command" if manual else "Update",
                                               QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.update_button.setToolTip("Show the command that updates this installation." if manual
                                      else "Close xdart and start the update in its progress window.")
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
