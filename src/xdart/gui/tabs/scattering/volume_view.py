"""A 3-D view of one volume: the RSM's "3D Viewer" (AUD-9 stage 2b G9).

The maintainer, 2026-10-01: "A button to bring up a 3d viewer for the entire 3d RSM volume", as one of the
"extra tools that pop up separate windows to keep the scientific view clean for now". A capability no owner had,
so a new shared component beside ``ScientificView`` (hard rule 14), not a page's own widget.

pyqtgraph's ``GLVolumeItem`` in a ``GLViewWidget``, inside a labelled bounding box, under Int's own controls: the
colour-map list and Log toggle (``shell_widgets``' shared builders, as ``ScientificView``'s top bar mounts them)
and ``IntensityControls`` for the levels, whose lower bound is also the opacity floor. The volume is drawn through
the headless ``xrd_tools.rsm.volume_render.volume_rgba`` (colour from the map between the levels; transparent at
or below the floor and where empty).

Contract, unlike ``ScientificView``'s: the host gives the data (``reconcile(VolumeProjection)``); the colour map,
Log and levels are this view's own state, since a 3D Viewer's presentation is nothing a page persists, so the view
draws itself again when they change. A volume above the 3D Viewer's limit is refused with the reason and a
"Show the 2× view" button (``downsample_volume_2x``). Where OpenGL is not available (an offscreen display) no GL
widget is built: the view says so and still computes ``rgba``, so the pipeline is testable everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.widgets.intensity_controls import IntensityControls
from xrd_tools.rsm.volume_render import (
    MAX_RENDER_VOXELS, VolumeRenderRefused, check_render_size, downsample_volume_2x, log_transform, volume_levels,
    volume_rgba,
)

from .shell_widgets import color_map_combo, log_toggle, set_combo_value

_GL_AVAILABLE: bool | None = None


def opengl_available() -> bool:
    """Whether this display gives an OpenGL context (an offscreen display does not); probed once."""
    global _GL_AVAILABLE
    if _GL_AVAILABLE is None:
        try:
            _GL_AVAILABLE = bool(QtGui.QOpenGLContext().create())
        except Exception:  # no GL module or platform plugin: no 3-D drawing, the view says so
            _GL_AVAILABLE = False
    return _GL_AVAILABLE


@dataclass(frozen=True)
class VolumeProjection:
    """One volume for ``VolumeView``: ``values`` (shared, not copied) on its three bin-centre ``axes``, with the
    axes' labels and a title."""

    values: np.ndarray
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    labels: tuple[str, str, str]
    title: str = ""

    def __post_init__(self) -> None:
        if type(self.values) is not np.ndarray or self.values.ndim != 3:
            raise TypeError("a volume projection holds one 3-D volume")
        if (len(self.axes) != 3 or len(self.labels) != 3
                or tuple(np.shape(axis) for axis in self.axes) != tuple((n,) for n in self.values.shape)):
            raise TypeError("a volume projection's three axes hold one centre per voxel")


class VolumeView(QtWidgets.QFrame):
    """The 3D Viewer: a volume in a labelled box under Int's colour map, Log and intensity controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("volumeView")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        bar = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("")
        self.color_map = color_map_combo()
        self.log_scale = log_toggle()
        self.intensity = IntensityControls(self)
        bar.addWidget(self.title, 1)
        bar.addWidget(self.intensity)
        bar.addWidget(self.color_map)
        bar.addWidget(self.log_scale)
        layout.addLayout(bar)
        self.notice = QtWidgets.QLabel("")
        self.notice.setObjectName("volumeViewNotice")
        self.notice.setWordWrap(True)
        self.notice.setVisible(False)
        layout.addWidget(self.notice)
        self.half_view = QtWidgets.QPushButton("Show the 2× view")
        self.half_view.setObjectName("volumeViewHalf")
        self.half_view.setVisible(False)
        self.half_view.clicked.connect(self.show_half_view)
        layout.addWidget(self.half_view, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.gl = self._volume = self._box = None
        self._texts: list = []
        if opengl_available():
            import pyqtgraph.opengl as gl  # lazy: an offscreen display never loads the GL items
            self.gl = gl.GLViewWidget()
            self.gl.setObjectName("volumeGLView")
            layout.addWidget(self.gl, 1)
        else:
            unavailable = QtWidgets.QLabel("3-D drawing needs OpenGL, which this display does not provide.")
            unavailable.setObjectName("volumeViewNoOpenGL")
            unavailable.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(unavailable, 1)
        #: the RGBA volume drawn now (uint8, the drawn volume's shape + (4,)), or None
        self.rgba: np.ndarray | None = None
        #: the 3D Viewer's limit (``volume_render.MAX_RENDER_VOXELS``: 256³, 64 MiB of RGBA)
        self.max_voxels = MAX_RENDER_VOXELS
        self._projection: VolumeProjection | None = None
        self._half = None  # the 2× view of the projection, once asked for
        self.color_map.currentTextChanged.connect(lambda _name: self._draw())
        self.log_scale.toggled.connect(lambda _checked: self._draw(reset_levels=True))
        self.intensity.rangeChanged.connect(lambda _low, _high: self._draw())
        self.intensity.autoToggled.connect(lambda _checked: self._draw())

    @property
    def projection(self) -> VolumeProjection | None:
        return self._projection

    @property
    def drawn_shape(self) -> tuple[int, int, int] | None:
        """The shape drawn now: the volume's, or its 2× view's."""
        return None if self.rgba is None else tuple(self.rgba.shape[:3])

    def reconcile(self, projection: VolumeProjection) -> None:
        """Show ``projection``'s volume at full resolution (a 2× view asked for an earlier volume is dropped)."""
        if type(projection) is not VolumeProjection:
            raise TypeError("a volume view draws a VolumeProjection")
        self._projection, self._half = projection, None
        self.title.setText(projection.title)
        self._draw(reset_levels=True)

    def show_half_view(self) -> None:
        """Draw the 2× view (2×2×2 block means) of a volume above the 3D Viewer's limit."""
        if self._projection is None:
            return
        self._half = downsample_volume_2x(self._projection.axes, self._projection.values)
        self._draw(reset_levels=True)

    def set_color_map(self, name: str) -> None:
        set_combo_value(self.color_map, name, fallback="Default")

    def release(self) -> None:
        """Let go of the volume and its RGBA (the window closing)."""
        self._projection = self._half = self.rgba = None
        self.title.setText("")
        self._clear_items()
        self.intensity.sync(None, None, reset=True)

    def _drawn(self):
        if self._half is not None:
            return self._half.values, self._half.axes
        return self._projection.values, self._projection.axes

    def _draw(self, *, reset_levels: bool = False) -> None:
        if self._projection is None:
            return
        values, axes = self._drawn()
        try:
            check_render_size(values.shape, self.max_voxels)
        except VolumeRenderRefused as refused:
            self.rgba = None
            self._clear_items()
            self._show_notice(str(refused), offer_half=self._half is None)
            return
        log = self.log_scale.isChecked()
        shown = log_transform(values) if log else values  # the displayed space, transformed once for this draw
        finite = np.isfinite(shown)
        if not finite.any():
            self.rgba = None
            self._clear_items()
            self._show_notice("The volume holds no finite value to draw.", offer_half=False)
            return
        domain = (float(np.min(shown, where=finite, initial=np.inf)),
                  float(np.max(shown, where=finite, initial=-np.inf)))
        del finite
        auto = volume_levels(shown, log=log)
        blocker = QtCore.QSignalBlocker(self.intensity)
        self.intensity.sync(domain if domain[1] > domain[0] else (domain[0], domain[0] + 1.0), auto,
                            reset=reset_levels)
        del blocker
        levels = auto if self.intensity.autoscale.isChecked() else tuple(float(v) for v in self.intensity.values())
        try:
            # ``shown`` is already in the displayed space and the levels are in it, so volume_rgba draws it as it
            # is (log=False) rather than taking Log of the same values again (Codex's combined stage-2b review,
            # priority 5): the same bytes, as test_rsm_volume_render pins volume_rgba(v, log=True, levels) ==
            # volume_rgba(log_transform(v), levels)
            self.rgba = volume_rgba(shown, self.color_map.currentText(), False, levels=levels,
                                    max_voxels=self.max_voxels)
        except VolumeRenderRefused as refused:
            self.rgba = None
            self._clear_items()
            self._show_notice(str(refused), offer_half=False)
            return
        del shown  # the Log copy goes before the GL items are built
        self._show_notice("", offer_half=False)
        self._draw_items(axes)

    def _show_notice(self, text: str, *, offer_half: bool) -> None:
        self.notice.setText(text)
        self.notice.setVisible(bool(text))
        self.half_view.setVisible(offer_half)

    def _clear_items(self) -> None:
        if self.gl is None:
            return
        for item in (self._volume, self._box, *self._texts):
            if item is not None:
                self.gl.removeItem(item)
        self._volume = self._box = None
        self._texts = []

    def _draw_items(self, axes) -> None:
        if self.gl is None:
            return
        import pyqtgraph.opengl as gl
        self._clear_items()
        nx, ny, nz = self.rgba.shape[:3]
        self._volume = gl.GLVolumeItem(self.rgba, sliceDensity=1, smooth=True, glOptions="translucent")
        self._box = gl.GLBoxItem(size=QtGui.QVector3D(nx, ny, nz))
        labels = self._projection.labels
        ends = ((nx, 0, 0), (0, ny, 0), (0, 0, nz))
        self._texts = [gl.GLTextItem(pos=end, text=f"{label} {float(axis[0]):.4g}…{float(axis[-1]):.4g}")
                       for label, axis, end in zip(labels, axes, ends, strict=True)]
        for item in (self._volume, self._box, *self._texts):
            self.gl.addItem(item)
        self.gl.opts["center"] = QtGui.QVector3D(nx / 2.0, ny / 2.0, nz / 2.0)
        self.gl.setCameraPosition(distance=2.0 * max(nx, ny, nz))


__all__ = ["VolumeProjection", "VolumeView", "opengl_available"]
