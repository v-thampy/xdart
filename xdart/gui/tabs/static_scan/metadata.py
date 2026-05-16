# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports

# Other imports
import pandas as pd

# Qt imports
from pyqtgraph import Qt, QtCore

# This module imports
from ...gui_utils import DFTableModel


class metadataWidget(Qt.QtWidgets.QWidget):
    """Widget for displaying metadata.

    attributes:
        layout: QVBoxLayout, widget layout
        tableview: QTableView, viewer for table model

    methods:
        update: Updates the data displayed
    """
    def __init__(self, sphere, arch, arch_ids, arches, parent=None,
                 data_1d=None):
        super().__init__(parent)
        self.layout = Qt.QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.tableview = Qt.QtWidgets.QTableView()
        self.tableview.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.tableview.setModel(DFTableModel())
        self.layout.addWidget(self.tableview)
        self.sphere = sphere
        self.arch = arch
        self.arch_ids = arch_ids
        self.arches = arches
        # O4: data_1d lets us pull scan_info for whichever frame is
        # currently selected, not the persistent placeholder ``arch``
        # (which stays at its constructor default unless the wrangler
        # happens to repurpose it).  Keeping the placeholder as a
        # fallback for legacy code paths.
        self.data_1d = data_1d

    def _resolve_selected_arch(self):
        """Return the arch whose metadata should be shown, or None.

        O4: drives off ``arch_ids`` (the H5Viewer's current selection)
        rather than ``self.arch.idx`` (the persistent placeholder).
        Tries the in-memory caches first; falls back to the placeholder
        only if neither holds the selected frame.
        """
        if not self.arch_ids:
            return None
        # arch_ids may be ints or strings depending on the entry path —
        # try both lookups.
        sel = self.arch_ids[0]
        try:
            sel_int = int(sel)
        except (TypeError, ValueError):
            sel_int = sel

        if self.data_1d is not None and sel_int in self.data_1d:
            return self.data_1d[sel_int]
        if sel_int in self.arches:
            return self.arches[sel_int]
        # Fallback to the placeholder if it's been populated for this
        # frame (live mode: wrangler stamps the latest arch on it).
        if (getattr(self.arch, "idx", None) is not None
                and int(self.arch.idx) == sel_int):
            return self.arch
        return None

    def update(self):
        """Updates the table with new data.

        O4: shows the selected frame's ``scan_info`` (from the data
        caches) when a selection exists, otherwise the whole-scan
        ``sphere.scan_data`` table.
        """
        selected = self._resolve_selected_arch()
        if selected is not None and getattr(selected, "scan_info", None):
            data = pd.DataFrame(
                selected.scan_info, index=[selected.idx],
            )
            self.tableview.setModel(DFTableModel(data.transpose()))
            return
        # No specific selection — show the whole-scan motor table.
        self.tableview.setModel(
            DFTableModel(self.sphere.scan_data.transpose())
        )
