"""One update offer for startup and Help; text is owned by the shared read-only pane."""
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .read_only_text import ReadOnlyTextPane


class UpdateDialog(QtWidgets.QDialog):
    def __init__(self, installed, target, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update xdart")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(620, 440)
        layout = QtWidgets.QVBoxLayout(self)
        self.versions = QtWidgets.QLabel(f"Installed: {installed}    •    Available: {target}")
        self.versions.setToolTip("The running version and the newer version available for this installation.")
        layout.addWidget(self.versions)
        self.notes = ReadOnlyTextPane(self)
        self.notes.set_text("Loading release notes…")
        self.notes.setToolTip("Scroll to read the changes in versions newer than the one installed. Text can be copied.")
        layout.addWidget(self.notes, 1)
        self.explanation = QtWidgets.QLabel("Update closes xdart. A progress window stays open; restart when it finishes.")
        self.explanation.setWordWrap(True)
        self.explanation.setToolTip("Nothing is installed until you choose Update. A processing Run postpones the offer.")
        layout.addWidget(self.explanation)
        self.link = QtWidgets.QPushButton("Open release notes")
        self.link.setToolTip("Read the full published release notes in your browser.")
        self._url = "https://github.com/v-thampy/xdart/releases"
        self.link.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._url)))
        buttons = QtWidgets.QDialogButtonBox()
        self.update_button = buttons.addButton("Update", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.update_button.setToolTip("Close xdart and start the update in its progress window.")
        self.cancel_button = buttons.addButton("Cancel", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        self.cancel_button.setToolTip("Keep this version. The update can be offered again the next time xdart starts.")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.link)
        row.addStretch()
        row.addWidget(buttons)
        layout.addLayout(row)

    def set_notes(self, notes):
        self.notes.set_text(notes.text)
        self._url = notes.url
