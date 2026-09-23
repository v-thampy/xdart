"""Modeless raw display composed from the shared scientific image pane."""
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.shell_widgets import ScientificImagePane


class StitchRawPopup(QtWidgets.QDialog):
    modeChanged = QtCore.Signal()
    closed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stitch contributing frame")
        self.resize(720, 620)
        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        self.caption = QtWidgets.QLabel("Select a contributing frame")
        self.caption.setWordWrap(True)
        self.caption.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(("Thumbnail", "Full Raw"))
        self.mode.currentIndexChanged.connect(self.modeChanged)
        toolbar.addWidget(self.caption, 1)
        toolbar.addWidget(self.mode)
        layout.addLayout(toolbar)
        self.image = ScientificImagePane(lock_aspect=True)
        layout.addWidget(self.image, 1)

    def render(self, frame_view, *, label, color_map="viridis", log=False):
        full = self.mode.currentText() == "Full Raw"
        raw = frame_view.raw if full else None
        image = raw if raw is not None else frame_view.thumbnail
        if image is None:
            self.image.clear_image_buffers()
            self.caption.setText(f"{label} · Image unavailable")
            return
        mode = "Full Raw" if raw is not None else (
            "Thumbnail · raw source unavailable" if full else "Thumbnail")
        self.caption.setText(f"{label} · {mode}")
        self.image.render(image, color_map="viridis" if color_map == "Default" else color_map,
                          log_scale=log)

    def closeEvent(self, event):
        self.image.clear_image_buffers()
        self.closed.emit()
        super().closeEvent(event)
