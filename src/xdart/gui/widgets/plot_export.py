# -*- coding: utf-8 -*-
"""The plots' shared Export... dialog (PLOT-EXPORT-TARGET-TYPE, PLOT-EXPORT-EMPTY-HDF5).

pyqtgraph gives every scene an Export... action and one Export dialog, but it offers every format for every target,
and its CSV, HDF5 and Matplotlib exporters then raise inside a Qt slot (HDF5 after it has created the file). Every
RectViewBox routes its scene's action here once (:func:`attach`). The first click makes one
:class:`SharedPlotExportDialog` for the scene, parented to the scene's view. A format the chosen target cannot take
is unavailable and says why. An exporter checks the plot it was opened for again when its delayed file name
arrives, and reports a refusal in the dialog before any file is touched. Its Save dialog comes from
:mod:`file_dialogs`, parented to the Export dialog, so deleting the plot's view retires the Export dialog, its Save
dialogs and its selection box.

pyqtgraph 0.14.0's SVG export raises for every plot under Qt 6.11; that failure is reported here too (maintainer,
October 14: "Report SVG's error in the dialog (Recommended)").

Built on pyqtgraph 0.14.0 internals; changing pyqtgraph invalidates the shared-export rows. They are
GraphicsScene.contextMenu/showExportDialog, ExportDialog.ui (formatList, itemTree, exportBtn, copyBtn, gridLayout),
its updateFormatList/exportFormatChanged/exportClicked, FormatExportListWidgetItem.expClass, and
Exporter.fileSaveDialog/fileSaveFinished.
"""
from __future__ import annotations

import functools

from pyqtgraph import ErrorBarItem, PlotItem, exporters
from pyqtgraph.GraphicsScene.exportDialog import ExportDialog, FormatExportListWidgetItem
from pyqtgraph.Qt import QtCore, QtWidgets, isQObjectAlive

#: Exporters that need a plot as their target: upstream raises for a scene or a bare view box.
_PLOT_ONLY = frozenset({"CSVExporter", "HDF5Exporter", "MatplotlibExporter"})
STATUS_TOOLTIP = ("Why the last export did not run: the plot it was opened for changed or is gone, or this format "
                  "cannot be written for it. Nothing was written.")


def _has_csv_data(plot) -> bool:
    """Whether CSV has a curve or error bars with data to write, read as CSV reads them (original data)."""
    for child in plot.items:
        if isinstance(child, ErrorBarItem):
            if child.opts["x"] is not None and child.opts["y"] is not None:
                return True
        elif hasattr(child, "implements") and child.implements("plotData"):
            data = child.getOriginalDataset() if hasattr(child, "getOriginalDataset") else child.getData()
            if data[0] is not None:
                return True
    return False


def unavailable(exporter, item) -> str:
    """Why the upstream ``exporter`` class cannot export ``item``, or "" when it can."""
    name = exporter.__name__
    if name not in _PLOT_ONLY:
        return ""
    if not isinstance(item, PlotItem):
        return f"{exporter.Name} exports a plot: choose a Plot above, not the entire scene or a view box."
    if name == "CSVExporter" and not _has_csv_data(item):
        return "This plot has no curve or error-bar data to write."
    if name == "HDF5Exporter":
        if not item.curves:
            return "HDF5 writes curves, and this plot has none."
        if any(x is None or y is None for x, y in (curve.getData() for curve in item.curves)):
            return "HDF5 writes every curve, and a curve on this plot has no data."
    return ""


class _OwnedExport:
    """An exporter that belongs to one SharedPlotExportDialog: its Save dialog is the dialog's, and it checks the
    plot it was opened for again before writing (also when the delayed file name arrives)."""

    owner = None
    upstream = None  # the pyqtgraph exporter class this one extends

    def fileSaveDialog(self, filter=None, opts=None):
        from xdart.gui.widgets.file_dialogs import save_later
        if self.owner is not None and isQObjectAlive(self.owner):
            # the callback holds this exporter: the dialog replaces its current one when its target changes
            self.fileDialog = save_later(self.owner, "Export", file_filter=filter, opts=opts,
                                         on_selected=lambda path, exporter=self: exporter.fileSaveFinished(path))

    def export(self, *args, **kwargs):
        owner = self.owner
        if owner is None or not isQObjectAlive(owner):
            return None  # its dialog went with the plot's view: nothing to report and nothing written
        reason = (unavailable(self.upstream, self.item) if isQObjectAlive(self.item)
                  else "The plot this export was opened for is gone.")
        if reason:
            owner.report_refusal(reason)
            return None
        writing = (args and args[0] is not None) or any(kwargs.get(key) for key in ("fileName", "toBytes", "copy"))
        if self.upstream.__name__ != "SVGExporter" or not writing:
            return super().export(*args, **kwargs)
        try:
            return super().export(*args, **kwargs)
        except ValueError as error:  # pyqtgraph 0.14.0 cannot read Qt 6.11's SVG; it raises before writing
            owner.report_refusal(f"{self.Name} failed in pyqtgraph, so nothing was written ({error}).")
            return None


@functools.cache
def _owned(exporter):
    """The upstream ``exporter`` class as one belonging to a SharedPlotExportDialog (made once per class)."""
    return type(exporter.__name__, (_OwnedExport, exporter), {"upstream": exporter})


class SharedPlotExportDialog(ExportDialog):
    """pyqtgraph's Export dialog for one scene, owned by the scene's view: a format the target cannot take is
    unavailable with the reason, and a refused or failed export says so here, reopening the dialog if needed."""

    def __init__(self, scene, view):
        super().__init__(scene)
        self.setParent(view, QtCore.Qt.WindowType.Window)
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("plotExportStatus")
        self.status.setWordWrap(True)
        self.status.setToolTip(STATUS_TOOLTIP)
        self.status.hide()
        self.ui.gridLayout.addWidget(self.status, 7, 0, 1, 3)

    def show(self, item=None):
        if item is not None:
            self.status.hide()  # a fresh open from a plot's menu
        super().show(item)

    def report_refusal(self, reason: str) -> None:
        """Say why an export did not run, reopening this dialog if it was closed meanwhile."""
        self.status.setText(reason)
        self.status.show()
        if not self.isVisible():
            super().show()

    def updateFormatList(self):
        current = self.ui.formatList.currentItem()
        kept = None if current is None else current.expClass.upstream
        row = self.ui.itemTree.currentItem()
        target = None if row is None else row.gitem
        self.status.hide()
        self.ui.formatList.clear()
        chosen = None
        for exporter in exporters.listExporters():
            item = FormatExportListWidgetItem(_owned(exporter),
                                              QtCore.QCoreApplication.translate("Exporter", exporter.Name))
            reason = unavailable(exporter, target)
            if reason:
                item.setFlags(item.flags() & ~(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable))
                item.setToolTip(reason)
            self.ui.formatList.addItem(item)
            if not reason and (chosen is None or exporter is kept):
                chosen = item  # the format chosen before, while it stays available; else the first available
        self.ui.formatList.setCurrentItem(chosen)

    def exportFormatChanged(self, item, prev):
        super().exportFormatChanged(item, prev)
        exporter = self.currentExporter
        if exporter is not None:
            exporter.owner = self
        else:
            self.ui.copyBtn.setEnabled(False)
        self.ui.exportBtn.setEnabled(exporter is not None)

    def exportClicked(self):
        self.status.hide()  # a new attempt: an earlier refusal no longer applies
        if self.currentExporter is not None:
            super().exportClicked()


def _open(scene, *_checked) -> None:
    """The scene's one Export dialog: made by the first click, parented to the scene's view, reused after."""
    dialog = scene.exportDialog
    if not isinstance(dialog, SharedPlotExportDialog) or not isQObjectAlive(dialog):
        dialog = scene.exportDialog = SharedPlotExportDialog(scene, scene.getViewWidget())
    dialog.show(scene.contextMenuItem)


def attach(scene) -> None:
    """Route ``scene``'s Export... action to its shared dialog, once (RectViewBox's scene seam)."""
    if getattr(scene, "_shared_plot_export", False) or not getattr(scene, "contextMenu", None):
        return
    scene._shared_plot_export = True
    action = scene.contextMenu[0]
    action.triggered.disconnect(scene.showExportDialog)
    action.triggered.connect(functools.partial(_open, scene))


__all__ = ["STATUS_TOOLTIP", "SharedPlotExportDialog", "attach", "unavailable"]
