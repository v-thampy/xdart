"""Stitch's provenance beside the shared metadata table, whose source selector (one per member scan of a
grouped result) is the shared dialog's own (Codex's Stitch/Int reuse audit R3)."""
import json

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.analysis_mount import MetadataResultDialog, MetadataSource


class StitchMetadataDialog(MetadataResultDialog):
    def __init__(self, provenance_json, parent=None):
        super().__init__(parent)
        self.provenance_button = QtWidgets.QPushButton("Provenance…", self)
        self.layout().addWidget(self.provenance_button)
        self.provenance_dialog = QtWidgets.QDialog(self)
        self.provenance_dialog.setWindowTitle("Saved Stitch provenance")
        layout = QtWidgets.QVBoxLayout(self.provenance_dialog)
        self.provenance_text = QtWidgets.QPlainTextEdit(self.provenance_dialog)
        self.provenance_text.setReadOnly(True)
        self.provenance_text.setPlainText(json.dumps(json.loads(provenance_json), indent=2))
        layout.addWidget(self.provenance_text)
        self.provenance_dialog.resize(720, 520)
        self.provenance_button.clicked.connect(self.provenance_dialog.show)
        self.resize(760, 450)

    def adopt_sources(self, sources):
        """The owner's read metadata (``StitchMetadataTable``), one source per member scan."""
        super().adopt_sources(tuple(MetadataSource(
            source.source_label, source.table, source.selected_labels, source.row_labels,
            f"{len(source.selected_labels)} contributing frames · Experimental metadata from original source")
            for source in sources))

    def closeEvent(self, event):
        self.provenance_dialog.close()
        super().closeEvent(event)
