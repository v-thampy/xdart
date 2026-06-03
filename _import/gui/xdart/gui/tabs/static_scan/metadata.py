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
    def __init__(self, scan, frame, frame_ids, frames, parent=None,
                 data_1d=None, publication_store=None):
        super().__init__(parent)
        self.layout = Qt.QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.tableview = Qt.QtWidgets.QTableView()
        self.tableview.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.tableview.setModel(DFTableModel())
        self.layout.addWidget(self.tableview)
        self.scan = scan
        self.frame = frame
        self.frame_ids = frame_ids
        self.frames = frames
        # O4: data_1d lets us pull scan_info for whichever frame is
        # currently selected, not the persistent placeholder ``frame``
        # (which stays at its constructor default unless the wrangler
        # happens to repurpose it).  Keeping the placeholder as a
        # fallback for legacy code paths.
        self.data_1d = data_1d
        self.publication_store = publication_store

    def showEvent(self, event):
        """Refresh when the panel becomes visible — ``update()`` short-
        circuits while hidden (perf), so a deferred render is needed when
        the user brings the panel back on screen."""
        super().showEvent(event)
        try:
            self.update()
        except Exception:
            pass

    def _resolve_selected_frame(self):
        """Return the frame whose metadata should be shown, or None.

        O4: drives off ``frame_ids`` (the H5Viewer's current selection)
        rather than ``self.frame.idx`` (the persistent placeholder).
        Tries the in-memory caches first; falls back to the placeholder
        only if neither holds the selected frame.
        """
        if not self.frame_ids:
            return None
        # frame_ids may be ints or strings depending on the entry path —
        # try both lookups.
        sel = self.frame_ids[0]
        try:
            sel_int = int(sel)
        except (TypeError, ValueError):
            sel_int = sel

        if self.data_1d is not None and sel_int in self.data_1d:
            return self.data_1d[sel_int]
        if sel_int in self.frames:
            return self.frames[sel_int]
        # Fallback to the placeholder if it's been populated for this
        # frame (live mode: wrangler stamps the latest frame on it).
        if (getattr(self.frame, "idx", None) is not None
                and int(self.frame.idx) == sel_int):
            return self.frame
        return None

    def _resolve_selected_publication(self):
        if not self.frame_ids or self.publication_store is None:
            return None
        sel = self.frame_ids[0]
        try:
            sel = int(sel)
        except (TypeError, ValueError):
            pass
        return self.publication_store.get(sel)

    def update(self):
        """Updates the table with new data.

        Shows the **whole-scan** metadata table (``scan.scan_data``,
        transposed so rows are metadata keys and columns are frames) —
        this is the historical behaviour the metadata panel is expected
        to have.  Only when there is no scan-wide table yet (e.g. the
        file/XYE viewer, or live mode before scan_data is populated) does
        it fall back to the selected frame's ``scan_info``.
        """
        # Skip the (potentially large) transpose + model rebuild when the
        # panel isn't on screen — during a fast live scan this slot fires
        # per refresh, and rebuilding a hidden table is wasted work.  The
        # panel re-renders from the current scan_data the next time it
        # becomes visible (Qt repaints on show).
        #
        # Gate on the TABLEVIEW, not ``self``: the host installs only this
        # widget's *layout* into its metaFrame (``metaFrame.setLayout(
        # metawidget.layout)``), so the metadataWidget QWidget itself is
        # never shown and ``self.isVisible()`` is always False — which
        # silently blanked the panel.  The tableview is the widget actually
        # on screen, so its visibility is the correct gate.
        if not self.tableview.isVisible():
            return
        sd = getattr(self.scan, "scan_data", None)
        if sd is not None and len(sd.index) and len(sd.columns):
            self.tableview.setModel(DFTableModel(sd.transpose()))
            return
        # No whole-scan table — fall back to the selected frame's info.
        publication = self._resolve_selected_publication()
        if publication is not None and publication.metadata_raw:
            data = pd.DataFrame(publication.metadata_raw, index=[publication.label])
            self.tableview.setModel(DFTableModel(data.transpose()))
            return
        selected = self._resolve_selected_frame()
        if selected is not None and getattr(selected, "scan_info", None):
            data = pd.DataFrame(selected.scan_info, index=[selected.idx])
            self.tableview.setModel(DFTableModel(data.transpose()))
            return
        self.tableview.setModel(DFTableModel())
