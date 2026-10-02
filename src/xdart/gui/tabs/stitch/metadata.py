"""Stitch's provenance beside the shared metadata table, whose source selector (one per member scan of a
grouped result) is the shared dialog's own (Codex's Stitch/Int reuse audit R3)."""
import json

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.analysis_mount import MetadataResultDialog, MetadataSource
from xdart.gui.widgets.read_only_text import ReadOnlyTextWindow


class StitchMetadataDialog(MetadataResultDialog):
    def __init__(self, provenance_json, parent=None):
        super().__init__(parent)
        self.provenance_button = QtWidgets.QPushButton("Provenance…", self)
        self.layout().addWidget(self.provenance_button)
        # the saved provenance: the shared read-only text window (Codex's combined stage-2b review F2)
        self.provenance_dialog = ReadOnlyTextWindow("Saved Stitch provenance", (720, 520), self)
        self.provenance_text = self.provenance_dialog.pane
        self.provenance_dialog.set_text(json.dumps(json.loads(provenance_json), indent=2))
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
