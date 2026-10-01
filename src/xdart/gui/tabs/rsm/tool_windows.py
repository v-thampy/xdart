"""The RSM page's two pop-up tools (AUD-9 stage 2b G9): the "3D Viewer" and "Custom Slice".

The maintainer, 2026-10-01: "A button to bring up a 3d viewer for the entire 3d RSM volume"; "The ability to only
slices which are not just along h, k or l. So say HH and l"; "The two extra features can be implemented as extra
tools that pop up separate windows to keep the scientific view clean for now". Each is a non-modal window that
only mounts shared components (hard rule 14): the 3D Viewer Int's ``VolumeView``. The page opens them from its
view's page tools and keeps them on the result it shows.
"""
from __future__ import annotations

import numpy as np
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.volume_view import VolumeProjection, VolumeView


class RSMVolumeWindow(QtWidgets.QDialog):
    """The 3D Viewer: the whole RSM volume on the shared ``VolumeView``, titled after its result. The volume is
    the viewer values' own buffer (not copied); the view's RGBA is 4 bytes a voxel (32 MB at Full 200³)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rsmVolumeWindow")
        self.setModal(False)
        self.setWindowTitle("RSM 3D Viewer")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = VolumeView()
        layout.addWidget(self.view)
        self.resize(900, 720)

    def show_result(self, name: str, values) -> None:
        self.setWindowTitle(f"RSM 3D Viewer — {name}")
        axes = tuple(np.asarray(axis) for _axis_name, axis in values.axes)
        self.view.reconcile(VolumeProjection(values.intensity, axes, tuple(values.coordinate_frame.axis_labels),
                                             name))

    def release(self) -> None:
        self.view.release()
        self.setWindowTitle("RSM 3D Viewer")

    def hideEvent(self, event) -> None:  # closed by the user: the volume and its RGBA go until it is shown again
        self.release()
        super().hideEvent(event)


__all__ = ["RSMVolumeWindow"]
