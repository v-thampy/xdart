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


__all__ = [
    "ANY_FILTER", "BACKGROUND_IMAGE_FILTER", "CALIBRATION_FILTER", "CORRECTION_FILTER",
    "MASK_FILTER", "NEXUS_FILTER", "PONI_FILTER", "browse",
]
