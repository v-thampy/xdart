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


def browse_spec(parent, title: str, *, many: bool = False, current: str = "",
                fallback: str = "") -> list[str]:
    """Ask for SPEC files in the platform's own dialog, as :func:`browse` (maintainer, 2026-09-30:
    "all others bring up the native mac os style dialog"). It lists every file, since a SPEC file has
    no extension to filter on; a pick that is not a SPEC file (by its first line) is refused, saying
    so, and the SPEC files among the picks are kept. [] when cancelled."""
    from pathlib import Path

    from pyqtgraph.Qt import QtWidgets

    from xrd_tools.io.spec import is_spec_file

    picks = browse(parent, title, many=many, current=current, fallback=fallback)
    refused = [path for path in picks if not is_spec_file(path)]
    if refused:
        names = ", ".join(Path(path).name for path in refused[:5]) + (" …" if len(refused) > 5 else "")
        QtWidgets.QMessageBox.warning(parent, title, f"Not a SPEC file, so not added: {names}")
    return [path for path in picks if path not in refused]


__all__ = [
    "ANY_FILTER", "BACKGROUND_IMAGE_FILTER", "CALIBRATION_FILTER", "CORRECTION_FILTER",
    "MASK_FILTER", "NEXUS_FILTER", "PONI_FILTER", "browse", "browse_spec",
]
