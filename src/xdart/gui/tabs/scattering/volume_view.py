"""A 3-D view of one volume: the RSM's "3D Viewer" (AUD-9 stage 2b G9).

The maintainer, 2026-10-01: "A button to bring up a 3d viewer for the entire 3d RSM volume", as one of the
"extra tools that pop up separate windows to keep the scientific view clean for now". A capability no owner had,
so a new shared component beside ``ScientificView`` (hard rule 14), not a page's own widget.

pyqtgraph's ``GLVolumeItem`` in a ``GLViewWidget``, inside a labelled bounding box, under Int's own controls: the
colour-map list and Log toggle (``shell_widgets``' shared builders, as ``ScientificView``'s top bar mounts them)
and ``IntensityControls`` for the levels, whose lower bound is also the opacity floor. The volume is coloured by
the headless ``xrd_tools.rsm.volume_render`` (colour from the map between the levels; transparent at or below the
floor and where empty), ``volume_rgba``'s bytes, through an 8-bit index volume and a 256-entry colour table.

Off the GUI thread (AUD-9 stage 2b item 6; the maintainer: "yes go ahead with both"): each draw is computed on the
shared ``LatestRequestOwner`` (``widgets/latest_request.py``: a ``SingleWorkerOwner`` and its poll timer, which
Custom Slice mounts too), and only the GL upload stays on the GUI thread. The latest draw asked for wins: a draw is
painted only if it is still the one asked for, a superseded one is cancelled and its result dropped, at most one is
computed at a time, and a draw that would show the image shown computes nothing. While one is computed the view
keeps the image it shows and says so. A Log or levels change makes a new index; a colour-map change applies the new
table to the index kept with the image. ``release`` (the window closed) drops a draw still computing when it
returns, and ``shutdown`` closes the worker as the page's other owners close (PENDING while a draw runs).

Contract, unlike ``ScientificView``'s: the host gives the data (``reconcile(VolumeProjection)``); the colour map,
Log and levels are this view's own state, since a 3D Viewer's presentation is nothing a page persists, so the view
draws itself again when they change. A volume above the 3D Viewer's limit is refused with the reason and a
"Show the 2× view" button (``downsample_volume_2x``). Where OpenGL is not available (an offscreen display) no GL
widget is built: the view says so and still computes ``rgba``, so the pipeline is testable everywhere.

Three shared RangeRows select displayed bin centres along the projection's labeled axes. Blank bounds use
the full axis; a crop shares the existing volume buffer and changes neither the scientific result nor its grid.
Each new result resets these presentation ranges, and closing the view releases their selected buffer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.pages.operation_owner import OperationCancelled, OperationTerminalStatus
from xdart.gui.widgets.controls_panel import RangeRow, SectionCard
from xdart.gui.widgets.intensity_controls import IntensityControls
from xdart.gui.widgets.latest_request import LatestRequestOwner
from xrd_tools.rsm.volume_render import (
    MAX_RENDER_VOXELS, VolumeRenderRefused, check_render_size, color_table, colour_volume, downsample_volume_2x,
    log_transform, volume_index, volume_levels,
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


@dataclass(frozen=True, eq=False)
class _DrawRequest:
    """One draw asked for: the volume drawn (shared, not copied) and its axes, Log, the levels (``automatic``:
    Int's percentiles of this draw; else the typed ``levels``), ``reset`` (the intensity controls go back to
    automatic) and the colour table. With ``index`` and ``alpha`` (a colour-map change), only the table is applied
    to that index, coloured at ``levels``."""

    values: np.ndarray
    axes: tuple
    log: bool
    automatic: bool
    reset: bool
    levels: tuple[float, float] | None
    color_map: str
    table: np.ndarray
    index: np.ndarray | None = None
    alpha: np.ndarray | None = None
    percentiles: tuple[float, float] | None = None

    def matches(self, other) -> bool:
        """The same draw: the same volume, index and settings."""
        return (other is not None and other.values is self.values and other.index is self.index
                and (other.log, other.automatic, other.reset, other.levels, other.color_map, other.percentiles)
                == (self.log, self.automatic, self.reset, self.levels, self.color_map, self.percentiles))


@dataclass(eq=False)
class _Drawn:
    """A draw's result: the RGBA, the index and opacity it was coloured from at ``levels``, and (a full draw) the
    domain and automatic levels the intensity controls take."""

    rgba: np.ndarray
    index: np.ndarray
    alpha: np.ndarray
    levels: tuple[float, float]
    domain: tuple[float, float] | None = None
    auto: tuple[float, float] | None = None


def _draw_volume(request: _DrawRequest, cancel, _progress):
    """A draw on the view's worker: the RGBA of ``request`` (``volume_rgba``'s bytes), or the refusal the view
    shows. Cancelled between its steps when a later draw supersedes it."""
    def check():
        if cancel.is_set():
            raise OperationCancelled()

    if request.index is not None:  # a colour-map change: the new table on the index kept
        return _Drawn(colour_volume(request.table, request.index, request.alpha), request.index, request.alpha,
                      request.levels)
    shown = log_transform(request.values) if request.log else request.values
    check()
    finite = np.isfinite(shown)
    if not finite.any():
        return VolumeRenderRefused("VOLUME_EMPTY", "The volume holds no finite value to draw.")
    domain = (float(np.min(shown, where=finite, initial=np.inf)),
              float(np.max(shown, where=finite, initial=-np.inf)))
    del finite
    check()
    auto = volume_levels(shown, log=request.log, percentiles=request.percentiles)
    levels = auto if request.automatic or request.percentiles is not None else request.levels
    try:
        index, alpha = volume_index(shown, levels)
    except VolumeRenderRefused as refused:
        return refused
    del shown  # the Log copy goes before the RGBA is made
    check()
    return _Drawn(colour_volume(request.table, index, alpha), index, alpha, tuple(float(v) for v in levels), domain,
                  auto)


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
        self.intensity = IntensityControls(self, percentiles=(0.5, 99.5))
        bar.addWidget(self.title, 1)
        bar.addWidget(self.intensity)
        bar.addWidget(self.color_map)
        bar.addWidget(self.log_scale)
        layout.addLayout(bar)
        self.range_card = SectionCard("Display ranges", collapsible=True)
        self.range_card.setToolTip("Crop the displayed volume by coordinate; the scientific result is unchanged.")
        self.axis_ranges = []
        self._range_settings = {}
        for index, label in enumerate(("H", "K", "L")):
            row = RangeRow(
                label=label,
                low={"path": ("volume_range", str(index), "low"), "value": "", "placeholder": "Auto"},
                high={"path": ("volume_range", str(index), "high"), "value": "", "placeholder": "Auto"})
            for widget in (row, row.label, row._low, row._high):
                widget.setToolTip(
                    "Minimum and maximum displayed coordinates only. Blank bounds use the full axis; "
                    "the scientific result is unchanged. Available when a volume is shown.")
            row.valueChanged.connect(self._range_edited)
            self.range_card.add_row(row)
            self.axis_ranges.append(row)
        self.range_card.setEnabled(False)
        layout.addWidget(self.range_card)
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
        self._display = None
        self._crop_key = None
        self._range_refusal = False
        self._half = None  # the 2× view of the projection, once asked for
        # the drawn image's index and opacity (2 B a voxel), what they were made from and at, and its colour map: a
        # colour-map change applies another table to them
        self._index = self._alpha = self._index_values = self._index_log = None
        self._index_levels = self._index_auto = self._index_domain = self._drawn_map = None
        self._index_percentiles = None
        # the draws, on the shared latest-request owner (the job bound late, so a test can hold its result); a draw
        # that would show the image shown is one whose index is the kept one, under the drawn colour map
        self._requests = LatestRequestOwner(
            lambda request, cancel, progress: _draw_volume(request, cancel, progress), self._done,
            same=lambda request, other: request.matches(other),
            unchanged=lambda request, _shown: (request.index is not None and request.index is self._index
                                               and request.color_map == self._drawn_map),
            parent=self)
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

    @property
    def computing(self) -> bool:
        """Whether a draw asked for is still being computed (the image shown is the one before it)."""
        return self._requests.computing

    @property
    def _wanted(self) -> _DrawRequest | None:
        """The draw asked for and not yet drawn (the shared owner's), or None."""
        return self._requests.wanted

    @property
    def _running(self):
        """``(identity, request)`` of the draw on the worker (the shared owner's), or None."""
        return self._requests.running

    def reconcile(self, projection: VolumeProjection) -> None:
        """Show ``projection``'s volume at full resolution (a 2× view asked for an earlier volume is dropped)."""
        if type(projection) is not VolumeProjection:
            raise TypeError("a volume view draws a VolumeProjection")
        self._projection, self._half = projection, None
        self._range_settings.clear()
        self._crop_key = None
        self.range_card.setEnabled(True)
        for label, axis, row in zip(projection.labels, projection.axes, self.axis_ranges, strict=True):
            row.label.setText(label)
            tip = f"Displayed {label} coordinate range only. Blank bounds use the full axis; bin centres are selected."
            for widget in (row, row.label, row._low, row._high):
                widget.setToolTip(tip)
            for editor, bound in ((row._low, np.min(axis)), (row._high, np.max(axis))):
                editor.setText("")
                editor.setPlaceholderText(f"Auto · {float(bound):.6g}")
        self.title.setText(projection.title)
        self._apply_ranges(reset_levels=True)

    def _range_edited(self, path, value) -> None:
        text = str(value).strip()
        if self._range_settings.get(tuple(path), "") == text:
            return
        self._range_settings[tuple(path)] = text
        self._apply_ranges()

    def _apply_ranges(self, *, reset_levels=False) -> None:
        """Select existing bin centres with array views; never read or regrid the source."""
        if self._projection is None:
            return
        slices = []
        try:
            for index, (axis, label) in enumerate(zip(self._projection.axes, self._projection.labels, strict=True)):
                bounds = []
                for side, default in (("low", np.min(axis)), ("high", np.max(axis))):
                    text = self._range_settings.get(("volume_range", str(index), side), "")
                    try:
                        value = float(text) if text else float(default)
                    except ValueError:
                        raise ValueError(f"{label} {side} needs a finite number") from None
                    if not np.isfinite(value):
                        raise ValueError(f"{label} {side} needs a finite number")
                    bounds.append(value)
                if bounds[0] > bounds[1]:
                    raise ValueError(f"{label} minimum must not exceed maximum")
                selected = np.flatnonzero((axis >= bounds[0]) & (axis <= bounds[1]))
                if not selected.size:
                    raise ValueError(f"{label} range contains no bin centres")
                slices.append(slice(int(selected[0]), int(selected[-1]) + 1))
        except ValueError as error:
            self._range_refusal = True
            self._ask(None)  # an earlier draw cannot clear this refusal when it returns
            self._show_notice(str(error), offer_half=False)
            return
        self._range_refusal = False
        key = tuple((selection.start, selection.stop) for selection in slices)
        if key == self._crop_key:
            self._show_notice("", offer_half=False)
            self._draw()
            return
        self._crop_key = key
        full = all(selection.start == 0 and selection.stop == size
                   for selection, size in zip(slices, self._projection.values.shape, strict=True))
        self._display = (self._projection.values if full else self._projection.values[tuple(slices)],
                         tuple(axis[selection] for axis, selection in zip(self._projection.axes, slices, strict=True)))
        self._half = None
        self._draw(reset_levels=reset_levels)

    def show_half_view(self) -> None:
        """Draw the 2× view (2×2×2 block means) of a volume above the 3D Viewer's limit."""
        if self._projection is None:
            return
        values, axes = self._display
        self._half = downsample_volume_2x(axes, values)
        self._draw(reset_levels=True)

    def set_color_map(self, name: str) -> None:
        set_combo_value(self.color_map, name, fallback="Default")

    def release(self) -> None:
        """Let go of the volume, its RGBA and its index (the window closing); a draw still computing is dropped,
        and its buffers freed, when it returns."""
        self._requests.release()
        self._projection = self._half = self.rgba = None
        self._display = self._crop_key = None
        self._range_refusal = False
        self._range_settings.clear()
        self.range_card.setEnabled(False)
        self._forget_index()
        self.title.setText("")
        self._clear_items()
        self._show_notice("", offer_half=False)
        self.intensity.sync(None, None, reset=True)

    def shutdown(self):
        """Close the view's worker (its host closing), as the page's owners close: CLEAN, or PENDING while a draw
        still runs (the host asks again); nothing is drawn after it."""
        return self._requests.shutdown()

    def _drawn(self):
        if self._half is not None:
            return self._half.values, self._half.axes
        return self._display

    def _draw(self, *, reset_levels: bool = False) -> None:
        """Ask for a draw of the volume shown under the controls now (the size check here, the colouring on the
        worker): a new index for a new volume, Log or levels, the kept index under another colour map, nothing
        when the image shown is already this draw."""
        if self._projection is None or self._range_refusal:
            return
        values, axes = self._drawn()
        try:
            check_render_size(values.shape, self.max_voxels)
        except VolumeRenderRefused as refused:
            self._ask(None)
            self.rgba = None
            self._forget_index()
            self._clear_items()
            self._show_notice(str(refused), offer_half=self._half is None)
            return
        wanted = self._requests.wanted
        if wanted is not None and wanted.reset:
            reset_levels = True  # a reset not yet drawn (Log's) holds for the draw that supersedes it
        log = self.log_scale.isChecked()
        self.intensity.set_percentile_defaults((0.1, 99.9) if log else (0.5, 99.5))
        automatic = reset_levels or self.intensity.autoscale.isChecked()
        percentiles = self.intensity.percentile_values()
        if percentiles is not None and reset_levels:
            percentiles = (0.1, 99.9) if log else (0.5, 99.5)
        levels = None if automatic or percentiles is not None else tuple(float(v) for v in self.intensity.values())
        kept = (self._index is not None and self._index_values is values and self._index_log == log
                and self._index_percentiles == percentiles
                and (percentiles is not None or self._index_levels == (self._index_auto if automatic else levels)))
        if kept and reset_levels:
            # Log on and off can return to the image already drawn. Reset the controls even if the
            # latest-request owner skips that unchanged image or only recolours its cached index.
            blocker = QtCore.QSignalBlocker(self.intensity)
            self.intensity.sync(self._index_domain, self._index_auto, reset=True)
            del blocker
        name = self.color_map.currentText()
        try:
            table = color_table(name)
        except VolumeRenderRefused as refused:  # a colour map matplotlib lacks: refused as volume_rgba refused it
            self._apply(None, refused)
            self._ask(None)
            return
        index = (self._index, self._alpha) if kept else (None, None)
        self._ask(_DrawRequest(values, axes, log, automatic, reset_levels,
                               self._index_levels if kept else levels, name, table, *index, percentiles))

    def _ask(self, request: _DrawRequest | None) -> None:
        """Ask the shared owner for ``request`` (None: none; the latest wins), the notice saying a draw is on its
        way while the image shown stays."""
        if self._requests.ask(request):
            self._show_notice(_DRAWING, offer_half=False)
        elif self.notice.text() == _DRAWING:
            self._show_notice("", offer_half=False)

    def _done(self, request: _DrawRequest, terminal) -> bool:
        """The draw asked for returned (the shared owner's ``done``): drawn, or its refusal or failure named;
        whether it is drawn."""
        if terminal.status is OperationTerminalStatus.RETURNED:
            return self._apply(request, terminal.payload)
        return self._apply(request, VolumeRenderRefused(
            "VOLUME_DRAW_FAILED", f"the 3D view could not be drawn ({terminal.failure_type}: "
                                  f"{terminal.failure_message})"))

    def _apply(self, request: _DrawRequest, drawn) -> bool:
        """Show a draw (the GUI thread's part: the intensity controls, then the GL upload) or its refusal; whether
        it is drawn."""
        if isinstance(drawn, VolumeRenderRefused):
            self.rgba = None
            self._forget_index()
            self._clear_items()
            self._show_notice(str(drawn), offer_half=False)
            return False
        if drawn.domain is not None:  # a full draw: the controls take its domain and automatic levels
            domain = drawn.domain
            blocker = QtCore.QSignalBlocker(self.intensity)
            self._index_domain = domain if domain[1] > domain[0] else (domain[0], domain[0] + 1.0)
            self.intensity.sync(self._index_domain, drawn.levels, reset=request.reset)
            del blocker
            self._index_auto = drawn.auto
        self.rgba = drawn.rgba
        self._index, self._alpha, self._index_levels = drawn.index, drawn.alpha, drawn.levels
        self._index_values, self._index_log, self._drawn_map = request.values, request.log, request.color_map
        self._index_percentiles = request.percentiles
        self._show_notice("", offer_half=False)
        self._draw_items(request.axes)
        return True

    def _forget_index(self) -> None:
        self._requests.forget()  # no image shown: no draw is the image shown
        self._index = self._alpha = self._index_values = self._index_log = None
        self._index_levels = self._index_auto = self._index_domain = self._drawn_map = None
        self._index_percentiles = None

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


_DRAWING = "Drawing the 3D view…"

__all__ = ["VolumeProjection", "VolumeView", "opengl_available"]
