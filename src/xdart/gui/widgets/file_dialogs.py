# -*- coding: utf-8 -*-
"""File-dialog filters and the Browse rule every page shares.

A dialog opens in the directory of the last successful pick, then the field's own
folder, then the page's fallback (the Project folder); a pick is remembered for the
next one (:mod:`xdart.utils.browse`). Int and Stitch offer the same file types for
the same kinds of input.
"""
from __future__ import annotations

from xdart.utils.browse import browse_start_dir, remember_browse_path

PONI_FILTER = "PONI files (*.poni);;All files (*)"
MASK_FILTER = "Detector masks (*.edf *.npy)"
#: A dark or flat field (Int Advanced and Stitch Advanced).
CORRECTION_FILTER = "Detector images (*.npy *.edf *.tif *.tiff *.cbf);;All files (*)"
BACKGROUND_IMAGE_FILTER = "Detector images (*.tif *.tiff *.cbf *.edf *.img *.mar3450 *.raw)"
#: A Stitch calibration: a PONI or a goniometer / XU calibration JSON.
CALIBRATION_FILTER = "Calibrations (*.poni *.json);;All files (*)"
NEXUS_FILTER = "NeXus / HDF5 (*.nexus *.nxs *.h5 *.hdf5);;All files (*)"
ANY_FILTER = "All files (*)"


def browse(parent, title: str, *, folder: bool = False, many: bool = False,
           file_filter: str = ANY_FILTER, current: str = "", fallback: str = "") -> list[str]:
    """Ask for a folder, a file or files; return the picks ([] when cancelled)."""
    from pyqtgraph.Qt import QtWidgets

    start = browse_start_dir(current, fallback=fallback)
    picks: list[str]
    if folder:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(parent, title, start)
        picks = [chosen] if chosen else []
    elif many:
        picks, _filter = QtWidgets.QFileDialog.getOpenFileNames(parent, title, start, file_filter)
        picks = list(picks or ())
    else:
        chosen, _filter = QtWidgets.QFileDialog.getOpenFileName(parent, title, start, file_filter)
        picks = [chosen] if chosen else []
    if picks:
        remember_browse_path(picks[0])
    return picks


#: The file types a SPEC picker offers: SPEC-named files first (``is_spec_name``), then all.
SPEC_LABEL = "SPEC files"
SPEC_FILTERS = (f"{SPEC_LABEL} (*)", ANY_FILTER)


def is_spec_name(name: str) -> bool:
    """A SPEC data file's name: no extension, as SPEC writes them, or ``.spec``."""
    if name.startswith("."):
        return False
    _stem, dot, suffix = name.rpartition(".")
    return not dot or suffix.casefold() == "spec"


def spec_dialog(parent, title: str, start: str, *, many: bool = False, places=()):
    """The SPEC picker, unopened: folders and SPEC-named files, "All files" one choice away.

    Qt's own dialog, because a native one filters only by extension patterns and a SPEC
    file has no extension (the images and .pdi/.csv files beside it would all show).
    """
    from pyqtgraph.Qt import QtCore, QtWidgets

    class SpecNames(QtCore.QSortFilterProxyModel):
        spec_only = True

        def filterAcceptsRow(self, row, parent_index):
            model = self.sourceModel()
            index = model.index(row, 0, parent_index)
            return not self.spec_only or model.isDir(index) or is_spec_name(model.fileName(index))

    dialog = QtWidgets.QFileDialog(parent, title, start)
    dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setOption(QtWidgets.QFileDialog.Option.HideNameFilterDetails, True)
    dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles if many
                       else QtWidgets.QFileDialog.FileMode.ExistingFile)
    names = SpecNames(dialog)
    dialog.setProxyModel(names)
    dialog.setNameFilters(list(SPEC_FILTERS))

    def filter_chosen(text):  # the type list's text, which hides the "(*)"
        names.spec_only = text.startswith(SPEC_LABEL)
        names.invalidate()
    dialog.filterSelected.connect(filter_chosen)
    places = [QtCore.QDir.homePath(), start, *places]  # its sidebar: home, the start and project folders
    dialog.setSidebarUrls([QtCore.QUrl.fromLocalFile(place) for place in dict.fromkeys(places) if place])
    return dialog


def browse_spec(parent, title: str, *, many: bool = False, current: str = "",
                fallback: str = "") -> list[str]:
    """Ask for SPEC files (as :func:`browse`: the last folder picked first); [] when cancelled."""
    dialog = spec_dialog(parent, title, browse_start_dir(current, fallback=fallback), many=many,
                         places=(fallback,))
    picks = [path for path in dialog.selectedFiles() if path] if dialog.exec() else []
    if picks:
        remember_browse_path(picks[0])
    return picks


__all__ = [
    "ANY_FILTER", "BACKGROUND_IMAGE_FILTER", "CALIBRATION_FILTER", "CORRECTION_FILTER",
    "MASK_FILTER", "NEXUS_FILTER", "PONI_FILTER", "SPEC_FILTERS", "browse", "browse_spec",
    "is_spec_name", "spec_dialog",
]
