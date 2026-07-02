# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import threading

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
                 viewer_rows_1d=None, publication_store=None, data_lock=None):
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
        # O4: viewer_rows_1d lets us pull scan_info for whichever frame is
        # currently selected, not the persistent placeholder ``frame``
        # (which stays at its constructor default unless the wrangler
        # happens to repurpose it).  Keeping the placeholder as a
        # fallback for legacy code paths.
        self.viewer_rows_1d = viewer_rows_1d
        self.publication_store = publication_store
        self.data_lock = data_lock if data_lock is not None else threading.RLock()
        self.viewer_mode = None

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

        with self.data_lock:
            # Phase 3c: the publication store (consulted FIRST in update()) is
            # the metadata source for every mode now — live integration AND the
            # Image/XYE/NeXus viewers all mirror their selected-row metadata into
            # it.  This frame fallback reads only the in-memory ``frames`` browse
            # cache.
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

    def _selected_scan_data(self, scan_data):
        """Return scan_data rows for the current selection when possible."""

        if not self.frame_ids:
            return scan_data
        labels = []
        for item in self.frame_ids:
            try:
                labels.append(int(item))
            except (TypeError, ValueError):
                labels.append(item)
        present = [label for label in labels if label in scan_data.index]
        if not present:
            return scan_data
        return scan_data.loc[present]

    def update(self):
        """Updates the table with new data.

        Shows selected rows from the scan metadata table when possible,
        transposed so rows are metadata keys and columns are frames.  This
        keeps live refresh bounded to the visible selection instead of
        rebuilding the whole scan table every tick.  Only when there is no
        scan-wide table yet (e.g. the file/XYE viewer, or live mode before
        scan_data is populated) does it fall back to the selected frame's
        ``scan_info``.
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
        if (
            self.viewer_mode is None
            and sd is not None
            and len(sd.index)
            and len(sd.columns)
        ):
            self.tableview.setModel(DFTableModel(self._selected_scan_data(sd).transpose()))
            return
        # No whole-scan table — fall back to the selected frame's info.  The
        # publication store is the source for every mode now (live integration
        # AND the Image/XYE/NeXus viewers all upsert their selected-row metadata);
        # the legacy frame path stays only for store-less callers (some tests).
        publication = self._resolve_selected_publication()
        if publication is not None and publication.metadata_raw:
            visible_info = {
                key: value
                for key, value in publication.metadata_raw.items()
                if not str(key).startswith("_")        # hide internal keys (parity
            }                                          # with the frame path)
            data = pd.DataFrame(visible_info, index=[publication.label])
            self.tableview.setModel(DFTableModel(data.transpose()))
            return
        selected = self._resolve_selected_frame()
        if selected is not None and getattr(selected, "scan_info", None):
            visible_info = {
                key: value
                for key, value in selected.scan_info.items()
                if not str(key).startswith("_")
            }
            data = pd.DataFrame(visible_info, index=[selected.idx])
            self.tableview.setModel(DFTableModel(data.transpose()))
            return
        self.tableview.setModel(DFTableModel())
