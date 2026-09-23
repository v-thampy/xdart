"""Stitch source selection and provenance around the shared metadata table."""
import json

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.analysis_mount import MetadataResultDialog
from xrd_tools.analysis.scan_operations import AnalysisDisposition


class StitchMetadataDialog(MetadataResultDialog):
    def __init__(self, provenance_json, parent=None):
        super().__init__(parent)
        self._sources = ()
        self.source_selector = QtWidgets.QComboBox(self)
        self.source_selector.currentIndexChanged.connect(self._select_source)
        self.layout().insertWidget(0, self.source_selector)
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
        self._sources = sources
        self.source_selector.clear()
        self.source_selector.addItems([source.source_label for source in sources])

    def _select_source(self, index):
        self.clear_result()
        if not 0 <= index < len(self._sources):
            return
        source = self._sources[index]
        if source.table.disposition is AnalysisDisposition.COMPLETED:
            self.adopt_result(source.table, selected_labels=source.selected_labels,
                              row_labels=source.row_labels)
            self.status.setText(f"{len(source.selected_labels)} contributing frames · Experimental metadata from original source")
        else:
            self.status.setText(source.table.code)

    def closeEvent(self, event):
        self.provenance_dialog.close()
        super().closeEvent(event)
