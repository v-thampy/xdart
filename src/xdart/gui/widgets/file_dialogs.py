# -*- coding: utf-8 -*-
"""File-dialog filters and the Browse rule every page shares.

A dialog opens in the directory of the last successful pick, then the field's own
folder, then the page's fallback (the Project folder); a pick is remembered for the
next one (:mod:`xdart.utils.browse`). Int and Stitch offer the same file types for
the same kinds of input. A save offers its default file name in that folder.
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
#: An original detector frame to draw a mask on: an image file, or a raw NeXus / Eiger master.
DETECTOR_SOURCE_FILTER = ("Detector images and raw NeXus (*.tif *.tiff *.cbf *.edf *.img *.mar3450 *.raw "
                          "*.nxs *.h5 *.hdf5);;All files (*)")
#: A page's saved settings (Stitch's File ▸ Save / Load Settings).
SETTINGS_FILTER = "Settings (*.json);;All files (*)"
ANY_FILTER = "All files (*)"


def export_name(result: str, tool: str, extension: str) -> str:
    """A portable result/tool name for the shared export save dialogs."""
    import re
    from pathlib import Path
    stem = re.sub(r'[^\w.-]+', '_', Path(result).stem).strip('._') or 'result'
    return f'{stem}_{tool}{extension}'


def ask(parent, title: str, *, start: str = "", folder: bool = False, many: bool = False,
        save: bool = False, file_filter: str = ANY_FILTER) -> list[str]:
    """The one place a file dialog opens (Codex's Stitch/Int reuse audit R5): ask for a folder, a file or
    files, or (``save``) a file to write, starting in ``start`` exactly as given; [] when cancelled.
    Nothing is remembered: :func:`browse` applies the shared Browse rule around it, and a caller that
    keeps that rule itself (Int's workspace seeds from ``browse_start_dir`` and remembers a pick only
    once it is accepted) calls this."""
    from pyqtgraph.Qt import QtWidgets

    if save:
        chosen, _filter = QtWidgets.QFileDialog.getSaveFileName(parent, title, start, file_filter)
        return [chosen] if chosen else []
    if folder:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(parent, title, start)
        return [chosen] if chosen else []
    if many:
        picks, _filter = QtWidgets.QFileDialog.getOpenFileNames(parent, title, start, file_filter)
        return list(picks or ())
    chosen, _filter = QtWidgets.QFileDialog.getOpenFileName(parent, title, start, file_filter)
    return [chosen] if chosen else []


def browse(parent, title: str, *, folder: bool = False, many: bool = False, save: bool = False,
           file_filter: str = ANY_FILTER, current: str = "", fallback: str = "",
           default_name: str = "") -> list[str]:
    """Ask for a folder, a file or files, or (``save``) a file to write; return the picks ([] when
    cancelled, and nothing is remembered then).

    A save offers ``default_name`` in the start folder. An absolute ``default_name`` is offered as it
    is: a managed location the caller then checks (Stitch's calibration save, inside its Project).
    """
    from os.path import isabs, join

    start = browse_start_dir(current, fallback=fallback)
    if save:
        offered = (default_name if isabs(default_name or "")
                   else join(start, default_name) if start and default_name else default_name or start)
        picks = ask(parent, title, start=offered, save=True, file_filter=file_filter)
    else:
        picks = ask(parent, title, start=start, folder=folder, many=many, file_filter=file_filter)
    if picks:
        remember_browse_path(picks[0])
    return picks


def browse_spec(parent, title: str, *, many: bool = False, current: str = "",
                fallback: str = "") -> list[str]:
    """Show folders, extensionless files and case-insensitive .spec files; validate picked content.

    SPEC alone uses Qt's widget dialog: native wildcard filters cannot express extensionless
    files without also listing other extensions. The shared Browse start/pick rule still applies.
    """
    from pathlib import Path

    from pyqtgraph.Qt import QtCore, QtWidgets

    from xrd_tools.io.spec import is_spec_file

    class SpecFiles(QtCore.QSortFilterProxyModel):
        def filterAcceptsRow(self, row, parent):
            model = self.sourceModel()
            info = model.fileInfo(model.index(row, 0, parent))
            return info.isDir() or Path(info.fileName()).suffix.lower() in ("", ".spec")

    dialog = QtWidgets.QFileDialog(parent, title, browse_start_dir(current, fallback=fallback))
    dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles if many else QtWidgets.QFileDialog.ExistingFile)
    dialog.setProxyModel(SpecFiles(dialog))
    try:
        picks = dialog.selectedFiles() if dialog.exec() == QtWidgets.QDialog.Accepted else []
    finally:
        dialog.deleteLater()
    if picks:
        remember_browse_path(picks[0])
    refused = [path for path in picks if not is_spec_file(path)]
    if refused:
        names = ", ".join(Path(path).name for path in refused[:5]) + (" …" if len(refused) > 5 else "")
        QtWidgets.QMessageBox.warning(parent, title, f"Not a SPEC file, so not added: {names}")
    return [path for path in picks if path not in refused]


__all__ = [
    "ANY_FILTER", "BACKGROUND_IMAGE_FILTER", "CALIBRATION_FILTER", "CORRECTION_FILTER",
    "DETECTOR_SOURCE_FILTER", "MASK_FILTER", "NEXUS_FILTER", "PONI_FILTER", "SETTINGS_FILTER",
    "ask", "browse", "browse_spec",
]
