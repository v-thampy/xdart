"""Raw, cake, and retained 1-D rendering for the passive E3 shell."""

from __future__ import annotations

from xdart.gui.widgets.scientific_plot import (
    PlotCursorReadout,
    scaled_intensity as _scaled_intensity,
    scaled_intensity_label as _scaled_intensity_label,
    overlay_step,
    reference_colors,
    trace_colors,
    data_curve_style,
)

from dataclasses import dataclass, replace
import functools
import logging
import math
import os
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.modules.display_context import (
    Viewer2DRendererClearReceipt, Viewer2DRendererClearRequest,
)
from xdart.gui.gui_utils import RectViewBox
from xdart.gui.themes import apply_seaborn_plot_style
from xdart.gui.widgets.intensity_controls import IntensityControls
from xrd_tools.session.display_logic import (
    PanelKey,
    PanelRole,
    canonical_axis_key,
    resample_image_axis_to_uniform,
    waterfall_display_rows,
)
from xrd_tools.session.viewer_1d import (
    Viewer1DRendererClearRequest,
    _new_viewer_1d_renderer_clear_receipt,
)

from .display_values import DisplayFrameKey
from .scientific_axes import (
    slice_region_orientation,
    trace_normalization_scope,
)
from .scientific_plot_options import (
    PLOT_OPTION_BINDINGS,
    PlotOptionsDialog,
    waterfall_should_be_active,
)
from .scientific_curves import add_curve_item, add_text_item, draw_reference_lines
from xdart.gui.widgets.latest_request import LatestRequestOwner
from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup
from xrd_tools.analysis.phase_overlays import PhaseOverlayJob, PhaseOverlayRequest, phase_floor, positions_for_axis
from .presentation_background import DisplayBackgroundRendererReleaseReceipt
from .processed_browser import TerminalRebindAuthorization
from .shell_widgets import (
    _axis_presentation,
    aggregate_traces,
    color_map_choices,
    color_map_combo,
    CompactFrameSelector,
    ContentFitComboBox,
    frame_caption,
    log_toggle,
    same_grid,
    ScientificImagePane,
    scrollable_toolbar,
    set_combo,
    set_combo_value,
    spin,
)
from .shell_values import (
    AxisProjection,
    BrowseTraceSnapshot,
    FrameNavigationProjection,
    FrameSelectionIntent,
    KeyedImageProjection,
    KeyedFigureProjection,
    KeyedPanelsProjection,
    RectRegion,
    ScientificPlotOptions,
    ScientificProjection,
    ShellCommand,
    ShellCommandKind,
    TraceProjection,
)


PLOT_TOOLBAR_INTRA_GROUP_GAP = 3
PLOT_TOOLBAR_INTER_GROUP_GAP = 40
PLOT_TOOLBAR_MENU_INDICATOR = "▾"
SCIENTIFIC_TOOLBAR_EDGE_INSET = 4
SCIENTIFIC_FOOTER_STATUS_LEFT_INSET = 8
ONE_D_PLOT_BOTTOM_MARGIN = 6
MAX_WATERFALL_DISPLAY_ROWS = 256
MAX_VIEWER_LOADING_SNAPSHOT_PIXELS = 2_000_000
_CONVERTIBLE_RADIAL_AXIS_KEYS = frozenset({"q_A^-1", "2th_deg"})
#: the top bar's Phases toggle (C6)
PHASES_TOOLTIP = ("Switch every reference phase listed in the Reference Phases window off or on at once; the list, "
                  "each phase's include check and every minimum intensity are kept.")
PHASES_NONE_LISTED = "No reference phase is listed: add one in the Tools panel's Reference Phases window."
PHASES_OFF_NOTE = ("Reference phases are switched off with the top bar's Phases button; the list, include checks and "
                   "minimums are kept.")
_STANDARD_PLOT_AXIS_CHOICES = (
    ("Q (Å⁻¹)", "Q"),
    ("2θ (°)", "2theta"),
    ("χ (°)", "chi"),
)
_STANDARD_IMAGE_AXIS_CHOICES = (
    ("Q-χ", "Q-Chi"),
    ("2θ-χ", "2Th-Chi"),
)
_GI_PLOT_AXIS_CHOICES = {
    "q_total": (
        ("Q (Å⁻¹)", "Q"),
        ("2θ (°)", "2theta"),
    ),
    "q_ip": (("Qᵢₚ (Å⁻¹)", "q_ip"),),
    "q_oop": (("Qₒₒₚ (Å⁻¹)", "q_oop"),),
    "exit_angle": (("Exit angle (°)", "exit_angle"),),
    "chi_gi": (("χGI (°)", "chi_gi"),),
}
_GI_CAKE_PLOT_AXIS_CHOICES = {
    "qip_qoop": (
        ("Qᵢₚ (Å⁻¹)", "q_ip"),
        ("Qₒₒₚ (Å⁻¹)", "q_oop"),
    ),
    "q_chi": (
        ("Q (Å⁻¹)", "Q"),
        ("χ (°)", "chi"),
    ),
    "exit_angles": (("Exit angle (°)", "exit_angle"),),
}
#: A GI q–χ cake written by the fiber integrator stores its azimuth as χGI
#: (``chigi_deg``).  Asking that cake for the transmission ``chi`` axis matches
#: nothing, so the χ cut is offered under the axis the cake really has.
_GI_CHIGI_CAKE_PLOT_AXIS_CHOICES = (
    ("Q (Å⁻¹)", "Q"),
    ("χGI (°)", "chi_gi"),
)
#: The same two cuts read off the re-binned display map, labelled as such.
_GI_DERIVED_CAKE_PLOT_AXIS_CHOICES = (
    ("Q (Å⁻¹, derived)", "Q"),
    ("χGI (°, derived)", "chi_gi"),
)
_GI_IMAGE_AXIS_LABELS = {
    "qip_qoop": "Qᵢₚ-Qₒₒₚ",
    "q_chi": "Q-χ",
    "q_chi_derived": "Q-χ (derived)",
    "exit_angles": "Exit angles",
}
_GI_IMAGE_AXIS_CHOICES = {
    mode: ((_GI_IMAGE_AXIS_LABELS[mode], mode),)
    for mode in ("qip_qoop", "q_chi", "exit_angles")
}


@dataclass(slots=True)
class _TraceAggregateCache:
    """Prepared immutable-prefix rows shared by Average and Sum renders."""

    rows: tuple[tuple[tuple[object, ...], TraceProjection], ...]
    intensity_scale: str
    values: np.ndarray
    projections: dict[str, TraceProjection]


def _is_readonly_array(values: np.ndarray) -> bool:
    # FrameView's repository-wide immutability contract freezes the exposed
    # array object.  Producers that retain a mutable alias must pass a copy;
    # walking hidden ndarray bases here would reject ordinary NumPy axes such
    # as read-only views returned by np.linspace.
    return not values.flags.writeable


class _CompactMenuButton(QtWidgets.QPushButton):
    """A menu button with one compact text indicator and no native duplicate."""

    def __init__(
        self,
        label: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(
            f"{label} {PLOT_TOOLBAR_MENU_INDICATOR}",
            parent,
        )
        self.setAccessibleName(label)

    def _paint_option(self) -> QtWidgets.QStyleOptionButton:
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        option.features &= (
            ~QtWidgets.QStyleOptionButton.ButtonFeature.HasMenu
        )
        return option

    def paintEvent(self, _event) -> None:
        painter = QtWidgets.QStylePainter(self)
        painter.drawControl(
            QtWidgets.QStyle.ControlElement.CE_PushButton,
            self._paint_option(),
        )


def _plot_toolbar_group(
    object_name: str,
    widgets: tuple[QtWidgets.QWidget, ...],
) -> QtWidgets.QWidget:
    group = QtWidgets.QWidget()
    group.setObjectName(object_name)
    group.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Maximum,
        QtWidgets.QSizePolicy.Policy.Preferred,
    )
    layout = QtWidgets.QHBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(PLOT_TOOLBAR_INTRA_GROUP_GAP)
    for widget in widgets:
        layout.addWidget(widget)
    return group


def _waterfall_rows_on_reference_axis(traces) -> np.ndarray | None:
    """Return trace rows sampled on the first trace's display axis.

    Live integrators can publish numerically equivalent grids whose floating
    samples differ by a few ulps.  Requiring byte-identical axes made the
    production 16-trace Single/Overlay waterfall silently fall back to curves.
    Preserve the first displayed grid and interpolate only the mismatching
    rows; values beyond another row's coverage remain NaN rather than being
    extrapolated.
    """

    if not traces:
        return None
    reference = np.asarray(traces[0].axis.values, dtype=float)
    if (
        reference.ndim != 1
        or reference.size < 2
        or not np.all(np.isfinite(reference))
    ):
        return None
    reference_step = np.diff(reference)
    if not (np.all(reference_step > 0.0) or np.all(reference_step < 0.0)):
        return None
    rows = []
    for trace in traces:
        values = np.asarray(trace.axis.values, dtype=float)
        intensity = np.asarray(trace.intensity, dtype=float)
        if (
            values.ndim != 1
            or values.size < 2
            or values.shape != intensity.shape
            or not np.all(np.isfinite(values))
        ):
            return None
        if values.shape == reference.shape and np.array_equal(
            values, reference
        ):
            rows.append(intensity)
            continue
        step = np.diff(values)
        if np.all(step < 0.0):
            values = values[::-1]
            intensity = intensity[::-1]
        elif not np.all(step > 0.0):
            return None
        target = reference if reference_step[0] > 0.0 else reference[::-1]
        row = np.interp(
            target,
            values,
            intensity,
            left=np.nan,
            right=np.nan,
        )
        rows.append(row if reference_step[0] > 0.0 else row[::-1])
    return np.stack(rows)


def _pixmap_pixel_count(pixmap) -> int:
    # QPixmap dimensions already count backing pixels. DevicePixelRatio only
    # converts them to logical screen coordinates; multiplying again would
    # unnecessarily downsample a Retina preview a second time.
    return pixmap.width() * pixmap.height()


#: A signed panel's map (a fit's residual: ``KeyedImageProjection.presentation``): diverging about zero.
SIGNED_COLOR_MAP = "RdBu_r"
_REGION_PEN = (240, 200, 90)


class _KeyedImagePanel:
    """A keyed layout's image panel: Int's image pane, titled, over the frame navigation's own widgets, which
    step its slice along its normal (``SET_PANEL_INDEX``); a rectangle region on it (``set_region``)."""

    def __init__(self, view: "ScientificView", key: PanelKey) -> None:
        self.key = key
        self.detector_picker = view._detector_picker
        self._detector_shape = None
        self.container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.pane = ScientificImagePane(lock_aspect=False)
        self.pane.setObjectName(f"e7KeyedImage_{key.instance}")
        self.pane.canvas.histogram.sigLevelsChanged.connect(lambda bar: view._keyed_colorbar_changed(key, bar))
        layout.addWidget(self.pane, 1)
        navigation = QtWidgets.QHBoxLayout()
        navigation.setContentsMargins(0, 0, 0, 0)
        self.previous = QtWidgets.QToolButton()
        self.previous.setObjectName("e4FrameNavigationButton")
        self.previous.setText("<")
        self.selector = CompactFrameSelector()
        self.selector.setObjectName("e3FrameSelector")
        self.next = QtWidgets.QToolButton()
        self.next.setObjectName("e4FrameNavigationButton")
        self.next.setText(">")
        self.previous.setToolTip("Show the previous frame or slice, keeping the selected regions.")
        self.next.setToolTip("Show the next frame or slice, keeping the selected regions.")
        self.selector.setToolTip("Choose a frame or slice; use arrow keys or the mouse wheel to step.")
        navigation.addStretch(1)
        for widget in (self.previous, self.selector, self.next):
            navigation.addWidget(widget)
        navigation.addStretch(1)
        layout.addLayout(navigation)
        self._captions: tuple[str, ...] = ()
        #: the panel's presentation and the (label, unit) of its two axes, as last drawn; its rectangle, or None
        self.presentation = "shared"
        self.axes: tuple[tuple[str, str], tuple[str, str]] | None = None
        self.region: pg.RectROI | None = None
        self.regions: dict[str, pg.RectROI] = {}
        self.selector.currentIndexChanged.connect(
            lambda index: index >= 0 and view._emit_panel_index(key, index))
        self.previous.clicked.connect(lambda: self.step(-1))
        self.next.clicked.connect(lambda: self.step(1))
        self.clear()

    def step(self, offset: int) -> None:
        index = self.selector.currentIndex() + offset
        if 0 <= index < self.selector.count():
            self.selector.setCurrentIndex(index)

    def draw(self, panel: KeyedImageProjection, color_map: str, log_scale: bool) -> None:
        signed = panel.presentation == "signed"
        if not panel.image.size:
            self.clear()
            for side, axis in (("bottom", panel.x_axis), ("left", panel.y_axis)):
                label, unit = _axis_presentation(axis.label, axis.unit)
                self.pane.plot.setLabel(side, label, units=unit or None)
            rendered = False
        elif self.detector_picker:
            # Detector selection uses pixel edges 0..N and row zero at the top, without the raw viewer's mirror.
            rows, columns = panel.image.shape
            rect = QtCore.QRectF(0, 0, columns, rows)
            self.pane.canvas.setImage(panel.image.T, rect=rect, scale="Log" if log_scale else "Linear",
                                      cmap=color_map, linear_percentiles=(2, 98), expand_degenerate_levels=False)
            box = self.pane.canvas.imageViewBox
            box.setAspectLocked(True)
            box.invertY(True)
            if self._detector_shape != panel.image.shape:
                box.setRange(rect, padding=0)
                self._detector_shape = panel.image.shape
            rendered = True
        else:
            rendered = self.pane.render(panel.image, x_axis=panel.x_axis, y_axis=panel.y_axis,
                            color_map=SIGNED_COLOR_MAP if signed else color_map,
                            log_scale=False if signed else log_scale)
        if rendered:  # drawn again (not kept): its own levels and
            # range, for the view's intensity controls (``_refresh_keyed_intensity``); none for a slice with no finite
            # value, whose colour bar the pane leaves as it was
            histogram = self.pane.canvas.histogram
            finite = np.isfinite(panel.image)
            if signed:  # its own limits, symmetric about zero, so a sign is never lost to the levels
                bound = float(np.max(np.abs(panel.image), where=finite, initial=0.0))
                bound = bound if bound > 0.0 else 1.0
                histogram.lo_lim, histogram.hi_lim = -bound, bound
                histogram.setLevels((-bound, bound))
            self.levels = (((histogram.lo_lim, histogram.hi_lim), tuple(histogram.levels()))
                           if finite.any() else None)
        self.presentation = panel.presentation
        self.pane.canvas.histogram.setToolTip(
            "Adjust the common color scale of all shared slices; manual limits remain across slice changes."
            if panel.presentation == "shared" else "Adjust this panel's independent color scale."
        )
        axes = ((panel.x_axis.label, panel.x_axis.unit), (panel.y_axis.label, panel.y_axis.unit))
        if self.region is not None and axes != self.axes:  # its rectangle was in other coordinates: it goes
            self.set_region(None, None)
        if axes != self.axes:
            for name in tuple(self.regions):
                self.set_named_region(name, None, None)
        self.axes = axes
        self.pane.plot.setTitle(panel.title)
        self.pane.setToolTip(f"{panel.title}. Drag to zoom into this map; right-click for plot controls.")
        for widget in (self.previous, self.selector, self.next):  # a plane with no index to step has none
            widget.setVisible(panel.navigation)
        if not panel.navigation:
            return
        blocker = QtCore.QSignalBlocker(self.selector)
        if panel.index_captions != self._captions:
            self.selector.clear()
            for caption in panel.index_captions:
                QtWidgets.QComboBox.addItem(self.selector, caption)
            self._captions = panel.index_captions
        self.selector.setCurrentIndex(panel.index)
        del blocker
        self.selector.setEnabled(True)
        self.previous.setEnabled(panel.index > 0)
        self.next.setEnabled(panel.index < len(panel.index_captions) - 1)

    def set_region(self, region: RectRegion | None, moved) -> None:
        """Show ``region`` as a rectangle the user can move and resize (None: none), ``moved()`` called when they
        have; it must be in the coordinates the panel draws."""
        self.set_named_region("", region, moved, width=1.5, live=False, corner_handle=False)

    def region_shown(self) -> RectRegion | None:
        """The rectangle where it is now (moved or not), typed by the panel's axes."""
        if self.region is None:
            return None
        position, size = self.region.pos(), self.region.size()
        x = tuple(sorted((float(position.x()), float(position.x() + size.x()))))
        y = tuple(sorted((float(position.y()), float(position.y() + size.y()))))
        return RectRegion(x, y, *self.axes)

    def set_named_region(self, name, region, moved, *, color=_REGION_PEN, dashed=False, visible=True,
                         width=2, live=True, corner_handle=True):
        """Independent signal/background marks, all owned here; programmatic reconciliation is silent."""
        box = self.pane.canvas.imageViewBox
        item = self.regions.get(name)
        if region is None:
            if item is not None:
                box.removeItem(item)
                del self.regions[name]
            if not name:
                self.region = None
            return
        if self.axes is None or (region.x_axis, region.y_axis) != self.axes:
            raise TypeError(f"a region on {(region.x_axis, region.y_axis)} cannot be drawn on a panel of {self.axes}")
        (x0, x1), (y0, y1) = region.x, region.y
        pen = pg.mkPen(color, width=width, style=(QtCore.Qt.PenStyle.DashLine if dashed
                                             else QtCore.Qt.PenStyle.SolidLine))
        if item is None:
            item = pg.RectROI((x0, y0), (x1 - x0, y1 - y0), pen=pen, movable=True, resizable=True)
            if corner_handle:
                item.addScaleHandle([0, 0], [1, 1])
            item.setZValue(10)
            self.regions[name] = item
            signal = item.sigRegionChanged if live else item.sigRegionChangeFinished
            signal.connect(lambda _roi: moved())
            box.addItem(item)
        else:
            blocker = QtCore.QSignalBlocker(item)
            item.setPos((x0, y0), update=False, finish=False)
            item.setSize((x1 - x0, y1 - y0), finish=False)
            item.setPen(pen)
            del blocker
        item.setVisible(bool(visible))
        if not name:
            self.region = item

    def named_region_shown(self, name):
        item = self.regions.get(name)
        if item is None:
            return None
        position, size = item.pos(), item.size()
        return RectRegion(tuple(sorted((float(position.x()), float(position.x() + size.x())))),
                          tuple(sorted((float(position.y()), float(position.y() + size.y())))), *self.axes)

    def clear(self) -> None:
        self.pane.clear_image_buffers()
        self.pane.plot.setTitle("")
        #: the slice's own (range, automatic levels) as last drawn, in what the pane draws (log10 with Log)
        self.levels: tuple[tuple[float, float], tuple[float, float]] | None = None
        blocker = QtCore.QSignalBlocker(self.selector)
        self.selector.clear()
        del blocker
        self._captions = ()
        for widget in (self.previous, self.selector, self.next):
            widget.setEnabled(False)


class _KeyedTracePanel:
    """A keyed layout's 1-D panel: a plot styled as Int's 1-D plot (box zoom, the seaborn style, the cursor
    readout), one curve."""

    def __init__(self, key: PanelKey) -> None:
        self.key = key
        self.container = self.plot = pg.PlotWidget(viewBox=RectViewBox())
        self.plot.setObjectName(f"e7KeyedPlot_{key.instance}")
        apply_seaborn_plot_style(self.plot.getPlotItem(), grid=True)
        self.readout = PlotCursorReadout(self.plot)
        self.curve = self.plot.plot(connect="finite")

    def draw(self, panel, color) -> None:
        self.curve.setData(np.asarray(panel.axis.values, dtype=float), np.asarray(panel.intensity, dtype=float),
                           **data_curve_style(color))
        self.plot.setTitle(panel.title)
        self.plot.setToolTip(f"{panel.title}. Drag to zoom into this curve; right-click for plot controls.")
        label, unit = _axis_presentation(panel.axis.label, panel.axis.unit)
        self.plot.setLabel("bottom", label, units=unit or None)
        self.plot.setLabel("left", panel.y_label)

    def clear(self) -> None:
        self.curve.setData([], [])
        self.plot.setTitle("")


class _KeyedFigurePanel:
    """Typed figure rendering inside Int's keyed display owner, with native Export."""

    def __init__(self, key, clicked):
        self.key = key
        self._clicked = clicked
        self._image_shape = None
        self.container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        self.viewbox = RectViewBox()
        self.plot = pg.PlotWidget(viewBox=self.viewbox)
        apply_seaborn_plot_style(self.plot.getPlotItem(), grid=False)
        self.viewbox.setAspectLocked(True, ratio=1)
        self.caption = QtWidgets.QLabel()
        self.caption.setWordWrap(True)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.caption)
        self.curves, self.overlay_items = {}, []
        self.plot.scene().sigMouseClicked.connect(self._image_clicked)

    def _image_clicked(self, event):
        if self._image_shape is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if not self.viewbox.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.viewbox.mapSceneToView(event.scenePos())
        rows, columns = self._image_shape
        if 0 <= point.x() <= columns-1 and 0 <= point.y() <= rows-1:
            self._clicked(self.key, float(point.x()), float(point.y()))

    def draw(self, panel, color):
        self.clear()
        figure = panel.figure
        self.plot.setTitle(panel.title or figure.title)
        self.plot.setToolTip('Computed figure; axes and caption state its coordinate frame. '
                             'Drag to zoom; right-click for native Export.')
        self.caption.setText(figure.caption)
        self.caption.setToolTip(figure.caption)
        units = figure.units if figure.units == 'px' else None
        self.plot.setLabel('bottom', figure.x_label, units=units)
        self.plot.setLabel('left', figure.y_label, units=figure.units or None)
        self.viewbox.invertY(figure.units == 'px')
        if figure.image is not None:
            self._image_shape = figure.image.shape
            item = pg.ImageItem(figure.image, axisOrder='row-major')
            item.setRect(QtCore.QRectF(-.5, -.5, figure.image.shape[1], figure.image.shape[0]))
            item.setZValue(-10)
            item.setToolTip('Image in stored pixel coordinates. '+figure.caption)
            self.plot.addItem(item)
        for curve in figure.curves:
            item = add_curve_item(self.plot, curve, color)
            self.curves[curve.name] = item
            self.overlay_items.append(item)
        for marker in figure.markers:
            item = pg.ScatterPlotItem([marker.xy[0]], [marker.xy[1]],
                symbol='x' if marker.role == 'rejected' else '+', size=12, pen=pg.mkPen(color, width=2))
            self._overlay(item, marker.tooltip)
        for arrow in figure.arrows:
            dx, dy = np.subtract(arrow.end, arrow.start)
            item = pg.ArrowItem(pos=arrow.end, angle=180-math.degrees(math.atan2(dy, dx)),
                headLen=10, tailLen=max(math.hypot(dx, dy)-10, 0), tailWidth=2,
                pen=pg.mkPen(color), brush=pg.mkBrush(color))
            self._overlay(item, arrow.tooltip)
        for text in figure.texts:
            self.overlay_items.append(add_text_item(self.plot, text, color))
        self.viewbox.autoRange()

    def _overlay(self, item, tooltip):
        item.setToolTip(tooltip)
        self.plot.addItem(item)
        self.overlay_items.append(item)

    def clear(self):
        self.plot.clear()
        self._image_shape = None
        self.curves, self.overlay_items = {}, []
        self.caption.clear()


class ScientificView(QtWidgets.QFrame):
    commandRequested = QtCore.Signal(object)
    #: Deliberate left click on a typed Figure image: key, column, row.
    figureImageClicked = QtCore.Signal(object, float, float)
    #: The user moved or resized a keyed image panel's rectangle (``set_keyed_region``): its ``PanelKey`` and the
    #: ``RectRegion`` it now covers.
    keyedRegionChanged = QtCore.Signal(object, object)
    #: A named region's live move: panel key, region identity, and typed geometry.
    keyedNamedRegionChanged = QtCore.Signal(object, str, object)
    coordinateRangesChanged = QtCore.Signal(object)
    #: The plot-options dialog was accepted. Emitted after the view has read its own options
    #: and before it emits their commands, so a host reads the choices it added
    #: (``plot_options_dialog.add_choice``) before any of its own redraws resets them.
    plotOptionsAccepted = QtCore.Signal()
    #: A host page's raw pop-up (``show_raw_popup``) was closed; its image is already cleared.
    rawPopupClosed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, detector_picker: bool = False) -> None:
        super().__init__(parent)
        self._detector_picker = bool(detector_picker)
        self.setObjectName("e3ScientificView")
        self.setMinimumWidth(300)
        self._hidden_tools: frozenset[str] = frozenset()
        # the keyed layout (``reconcile_keyed``): built on its first draw, so Int's and Stitch's views never
        # allocate it
        self._keyed_area: QtWidgets.QSplitter | None = None
        self._coordinate_range_bar = None
        self.coordinate_ranges = []
        self._keyed_layout: tuple[tuple[PanelKey, ...], ...] = ()
        self._keyed_panels: dict[PanelKey, _KeyedImagePanel | _KeyedTracePanel] = {}
        self._keyed_shown: KeyedPanelsProjection | None = None
        self._keyed_hidden: tuple[QtWidgets.QWidget, ...] = ()
        self._figure_hidden: tuple[QtWidgets.QWidget, ...] = ()
        # Share Axis over it: the axes drawing each coordinate shared by two or more, the range links made, and
        # where Share Axis sits in Int's plot bar while the keyed row borrows it
        self._keyed_coordinates: dict[tuple[str, str], tuple[tuple[pg.ViewBox, int], ...]] = {}
        self._keyed_links: list[tuple[object, object]] = []
        self._keyed_share_on = False
        self._share_axis_home = -1
        self._heavy_available: frozenset[DisplayFrameKey] = frozenset()
        self._frame_keys: tuple[DisplayFrameKey, ...] = ()
        self._current_key: DisplayFrameKey | None = None
        self._selected_keys: tuple[DisplayFrameKey, ...] = ()
        self._plot_mode = "Single"
        self._single_mode = True
        self._label_indices: dict[int, list[int]] = {}
        self._selector_operations = 0
        self._trace_history_scope: tuple[object, ...] | None = None
        self._trace_selection_keys: tuple[DisplayFrameKey, ...] = ()
        self._trace_history_keys: tuple[DisplayFrameKey, ...] = ()
        self._trace_row_count = 0
        self._trace_history_by_identity: dict[int, TraceProjection] = {}
        self._trace_aggregate_fold: _TraceAggregateCache | None = None
        self._pinned_trace_scope: tuple[object, ...] | None = None
        self._pinned_trace_by_id: dict[
            tuple[object, ...], TraceProjection
        ] = {}
        self._rendered_trace_keys: tuple[tuple[object, ...], ...] = ()
        self._curve_mounted_keys: tuple[tuple[object, ...], ...] = ()
        self._curve_items_by_key: dict[tuple[object, ...], object] = {}
        self._curve_item_contracts: dict[
            tuple[object, ...], tuple[object, ...]
        ] = {}
        self._rendered_axis_labels: tuple[
            str, str | None, str
        ] | None = None
        self._rendered_plot_mode = ""
        self._rendered_plot_options: ScientificPlotOptions | None = None
        self._rendered_overlay_step: float | None = None
        self._rendered_browse_science_contract: tuple[object, ...] | None = None
        self._bottom_waterfall_active = False
        self._waterfall_y_values: tuple[float, ...] = ()
        self._waterfall_y_label = "Frame #"
        self._waterfall_last_draw = 0.0
        self._waterfall_source_keys: tuple[tuple[object, ...], ...] = ()
        self._waterfall_render_contract: tuple[object, ...] | None = None
        self._processing_mode = ""
        self._layout_mode = ""
        self._viewer_intensity_contract = None
        self._viewer_intensity_domain = None
        self._viewer_auto_levels = None
        self._viewer_2d_payload = None
        self._viewer_2d_known_empty = False
        self._viewer_2d_shape = None
        self._rendered_detector_source = "none"
        self._expected_background_key = self._rendered_background_key = None
        self.raw_popup_dialog = None
        self.raw_popup_image = None
        #: True while a host page (Stitch) owns the raw pop-up; Int's own (its Raw button, Int 1D) when False
        self._raw_popup_host = False
        self._rendered_image_axis: str | None = None
        self._rendered_cake_axis_key: str | None = None
        self._rendered_cake_x_axis: AxisProjection | None = None
        self._rendered_cake_y_axis: AxisProjection | None = None
        self._rendered_trace_axis_key: str | None = None
        self._slice_extent_lines: tuple[pg.InfiniteLine, ...] = ()
        self._slice_extent_scope: tuple[object, ...] | None = None
        self._rendered_slice_contract: tuple[object, ...] | None = None
        self._share_link_on = False
        self._share_axis_syncing = False
        self._cake_x_pinned_by_share = False
        self._align_seq = 0
        self._align_cake_seq = 0
        self._align_curve_pending = False
        self._align_cake_pending = False
        self._reflow_follows = 0
        self._share_cake_handler = self._on_cake_xrange_changed
        self._share_curve_handler = self._on_curve_xrange_changed
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.top_bar = self._make_top_bar()
        layout.addWidget(scrollable_toolbar(
            self.top_bar,
            minimum_width=520,
            horizontal_inset=SCIENTIFIC_TOOLBAR_EDGE_INSET,
        ))

        self.vertical_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical
        )
        self.image_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal
        )
        self.raw = ScientificImagePane(lock_aspect=True)
        self.raw.setObjectName("e3RawImage")
        self.cake = ScientificImagePane(lock_aspect=False)
        self.cake.setObjectName("e3CakeImage")
        self.image_splitter.addWidget(self.raw)
        self.image_splitter.addWidget(self.cake)
        self.image_splitter.setSizes([1, 1])
        self.vertical_splitter.addWidget(self.image_splitter)

        one_d = QtWidgets.QFrame()
        one_d_layout = QtWidgets.QVBoxLayout(one_d)
        one_d_layout.setContentsMargins(0, 0, 0, 0)
        # as the 2-D views and the retired Static Scan page: left-drag draws a box to zoom into
        # (pyqtgraph's default pans), middle-click shows all, right-click keeps pyqtgraph's menu
        self.curve = pg.PlotWidget(viewBox=RectViewBox())
        self.curve.setObjectName("e3OneDPlot")
        apply_seaborn_plot_style(self.curve.getPlotItem(), grid=True)
        self.curve.getPlotItem().layout.setContentsMargins(
            0,
            0,
            0,
            ONE_D_PLOT_BOTTOM_MARGIN,
        )
        self.legend = self.curve.addLegend()
        # A phase's legend sample includes or excludes that phase (REFERENCE-PHASES-20261011).
        self.legend.sigSampleClicked.connect(self._legend_sample_clicked)
        self._cursor_readout = PlotCursorReadout(self.curve)
        self.cursor_position = self._cursor_readout.label
        self._mouse_proxy = self._cursor_readout.proxy
        self.waterfall = ScientificImagePane(lock_aspect=False)
        self.waterfall.setObjectName("e4WaterfallPlot")
        self.bottom_stack = QtWidgets.QStackedWidget()
        self.bottom_stack.setObjectName("e4BottomPlotStack")
        self.bottom_stack.addWidget(self.curve)
        self.bottom_stack.addWidget(self.waterfall)
        self.bottom_stack.setCurrentWidget(self.curve)
        one_d_layout.addWidget(self.bottom_stack, 1)
        self.plot_bar = self._make_plot_bar()
        self.plot_toolbar = scrollable_toolbar(
            self.plot_bar,
            minimum_width=760,
            horizontal_inset=SCIENTIFIC_TOOLBAR_EDGE_INSET,
        )
        one_d_layout.addWidget(self.plot_toolbar)
        self.vertical_splitter.addWidget(one_d)
        self.vertical_splitter.setStretchFactor(0, 1)
        self.vertical_splitter.setStretchFactor(1, 1)
        self.vertical_splitter.setSizes([500, 500])
        layout.addWidget(self.vertical_splitter, 1)
        self._main_layout = layout

        self.footer = QtWidgets.QHBoxLayout()
        self.viewer_intensity = IntensityControls(self)
        self.viewer_intensity.hide()
        self.viewer_intensity.rangeChanged.connect(self._set_viewer_intensity)
        self.viewer_intensity.autoToggled.connect(self._toggle_viewer_autoscale)
        self.viewer_intensity_row = QtWidgets.QWidget(self)
        self.viewer_intensity_row_layout = QtWidgets.QHBoxLayout(self.viewer_intensity_row)
        self.viewer_intensity_row_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_intensity_row_layout.addStretch(1)
        self.plot_bar.addWidget(self.viewer_intensity)
        self.viewer_intensity_row.hide()
        layout.addWidget(self.viewer_intensity_row)
        self.status = QtWidgets.QLabel("")
        self.status.setContentsMargins(
            SCIENTIFIC_FOOTER_STATUS_LEFT_INSET,
            0,
            0,
            0,
        )
        self.progress = QtWidgets.QLabel("0/0")
        self.progress.setObjectName("e3GlobalProgress")
        self.footer.addWidget(self.status, 1)
        self.footer.addWidget(self.previous_frame)
        self.footer.addWidget(self.frame_selector)
        self.footer.addWidget(self.next_frame)
        self.footer.addWidget(self.progress)
        layout.addLayout(self.footer)
        self._viewer_loading_mode: str | None = None
        self._viewer_loading_target: QtWidgets.QWidget | None = None
        self._viewer_loading_watched: tuple[QtWidgets.QWidget, ...] = ()
        self._viewer_loading_overlay = QtWidgets.QFrame(self)
        self._viewer_loading_overlay.setObjectName("e6ViewerLoadingPreviousView")
        self._viewer_loading_overlay.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._viewer_loading_overlay.setStyleSheet(
            "background-color: rgba(22, 28, 36, 210);"
        )
        self._viewer_loading_overlay.setFocusPolicy(
            QtCore.Qt.FocusPolicy.NoFocus
        )
        self._viewer_loading_pixmap = QtWidgets.QLabel(
            self._viewer_loading_overlay
        )
        self._viewer_loading_pixmap.setObjectName("e6ViewerLoadingPixmap")
        self._viewer_loading_pixmap.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self._viewer_loading_pixmap.setScaledContents(True)
        for widget in (
            self._viewer_loading_overlay,
            self._viewer_loading_pixmap,
        ):
            widget.installEventFilter(self)
        self._viewer_loading_overlay.hide()
        self._install_share_geometry_hooks()
        if self._detector_picker:
            self._detector_picker_image = None
            self._detector_picker_navigation = ((), 0, "")
            self._main_layout.itemAt(0).widget().hide()
            scale = QtWidgets.QHBoxLayout()
            scale.addWidget(QtWidgets.QLabel("Intensity"))
            self.default_scale = QtWidgets.QPushButton("Default")
            self.detector_log = QtWidgets.QPushButton("Log")
            self.default_scale.setToolTip("Show detector image intensity on a linear scale.")
            self.detector_log.setToolTip("Show detector image intensity on a logarithmic scale; ROI values are unchanged.")
            group = self._detector_scale_group = QtWidgets.QButtonGroup(self)
            group.setExclusive(True)
            for button in (self.default_scale, self.detector_log):
                button.setCheckable(True)
                group.addButton(button)
                scale.addWidget(button)
            self.default_scale.setChecked(True)
            scale.addStretch(1)
            self._main_layout.addLayout(scale)
            self.detector_log.toggled.connect(lambda _on: self.reconcile_detector_image(
                self._detector_picker_image, *self._detector_picker_navigation))

    def reconcile_detector_image(self, image, captions=(), index=0, title="") -> PanelKey:
        """Detector-picker profile: Int's image/scale owners, row-major pixel coordinates, independent ROI marks."""
        if not self._detector_picker:
            raise RuntimeError("reconcile_detector_image needs the detector_picker profile")
        self._detector_picker_image = image
        self._detector_picker_navigation = (tuple(captions), index, title)
        key = PanelKey(PanelRole.SLICE_2D, "detector")
        if image is None:
            self.clear_keyed()
            return key
        rows, columns = np.asarray(image).shape
        panel = KeyedImageProjection(key, title, np.asarray(image),
            AxisProjection(np.arange(columns, dtype=float) + 0.5, "column", "px"),
            AxisProjection(np.arange(rows, dtype=float) + 0.5, "row", "px"), index, tuple(captions),
            navigation=bool(captions),
            presentation="own")
        self.reconcile_keyed(KeyedPanelsProjection(((key,),), (panel,), color_map="viridis",
                                                  log_scale=self.detector_log.isChecked()))
        self.viewer_intensity_row.hide()
        self.status.hide()
        return key

    def reconcile_operation_status(self, status: object) -> bool:
        """Replace only the scalar scientific-footer status text."""

        if type(status) is not str or not status:
            return False
        self.status.setText(status)
        return True

    @property
    def rendered_image_axis(self) -> str | None:
        """Identity of the cake presentation actually accepted by the view."""

        return self._rendered_image_axis

    @property
    def viewer_loading_snapshot_visible(self) -> bool:
        """Whether a non-scientific pending-view raster is currently shown."""

        return self._viewer_loading_overlay.isVisible()

    @property
    def viewer_loading_snapshot_pixels(self) -> int:
        """Pixel count retained by the pending-view raster, if any."""

        pixmap = self._viewer_loading_pixmap.pixmap()
        return (
            0
            if pixmap is None or pixmap.isNull()
            else _pixmap_pixel_count(pixmap)
        )

    @property
    def trace_history_keys(self) -> tuple[DisplayFrameKey, ...]:
        """Exact 1-D rows accepted by the last successful reconciliation."""

        return self._trace_history_keys

    @property
    def trace_history_projections(self) -> tuple[TraceProjection, ...]:
        """Exact retained trace rows in their rendered navigation order."""

        rows = tuple(
            self._trace_history_by_identity.get(id(frame))
            for frame in self._trace_history_keys
        )
        return (
            rows
            if all(type(row) is TraceProjection for row in rows)
            else ()
        )

    @property
    def navigation_frame_keys(self) -> tuple[DisplayFrameKey, ...]:
        """Exact footer identities owned by the accepted presentation."""

        return self._frame_keys

    @property
    def navigation_current_key(self) -> DisplayFrameKey | None:
        """Exact current identity owned by the accepted presentation."""

        return self._current_key

    @property
    def navigation_selected_keys(self) -> tuple[DisplayFrameKey, ...]:
        """Exact selected identities owned by the accepted presentation."""

        return self._selected_keys

    @property
    def heavy_available_keys(self) -> frozenset[DisplayFrameKey]:
        """Exact heavy-payload frame identities retained by the view."""

        return self._heavy_available

    @property
    def presentation_plot_mode(self) -> str:
        """Plot mode of the accepted trace presentation."""

        return self._rendered_plot_mode

    @property
    def trace_row_count(self) -> int:
        """Exact rendered live-plus-pinned row count from the last paint."""

        return self._trace_row_count

    @property
    def bottom_waterfall_active(self) -> bool:
        """Whether the last successful paint selected the Waterfall panel."""

        return self._bottom_waterfall_active

    @property
    def waterfall_source_frame_keys(self) -> tuple[DisplayFrameKey, ...]:
        """Exact source domain of the last successfully painted Waterfall."""

        if not self._bottom_waterfall_active:
            return ()
        frames_by_identity = {
            id(frame): frame for frame in self._trace_history_keys
        }
        resolved = []
        for key in self._waterfall_source_keys:
            if (
                type(key) is not tuple
                or len(key) != 2
                or key[0] != "live"
                or type(key[1]) is not int
            ):
                return ()
            frame = frames_by_identity.get(key[1])
            if frame is None or id(frame) != key[1]:
                return ()
            resolved.append(frame)
        return tuple(resolved)

    def expect_display_background(self, active_key) -> None:
        key = active_key if type(active_key) is tuple else None
        if key != self._expected_background_key:
            self._rendered_trace_keys = (); self._rendered_plot_mode = ""
            self._waterfall_source_keys = (); self._waterfall_render_contract = None
            self._rendered_browse_science_contract = None
            self._clear_trace_aggregate_fold()
        self._expected_background_key = key

    def release_display_background(self, active_key):
        if active_key != self._rendered_background_key: return DisplayBackgroundRendererReleaseReceipt(active_key, False)
        domain = active_key[2]
        try:
            if domain == "raw":
                self.raw.clear_image_buffers(); self._viewer_2d_payload = None
                self._viewer_2d_known_empty = False
                self._rendered_detector_source = "none"
                if self.raw_popup_image is not None: self.raw_popup_image.clear_image_buffers()
            elif domain == "integrated_2d":
                self.cake.clear_image_buffers(); self._rendered_cake_x_axis = self._rendered_cake_y_axis = self._rendered_cake_axis_key = self._rendered_image_axis = None
            elif domain == "keyed":  # a keyed layout drawn minus its background: every panel let go of
                for widgets in self._keyed_panels.values():
                    widgets.clear()
                self._keyed_shown = None
            else:
                self.curve.clear(); self.waterfall.clear(); self._trace_history_by_identity.clear(); self._pinned_trace_by_id.clear()
                self._rendered_trace_keys = self._trace_history_keys = self._waterfall_source_keys = ()
                self._curve_mounted_keys = ()
                self._curve_items_by_key.clear(); self._curve_item_contracts.clear()
                self._clear_trace_aggregate_fold()
                self._rendered_axis_labels = None
                self._trace_row_count = 0
                self._rendered_browse_science_contract = None
            self._rendered_background_key = self._expected_background_key = None
        except Exception: return DisplayBackgroundRendererReleaseReceipt(active_key, False)
        return DisplayBackgroundRendererReleaseReceipt(active_key, True)

    def _make_top_bar(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.norm = ContentFitComboBox()
        self.norm.setObjectName("e3NormChannel")
        self.background = QtWidgets.QPushButton("Set BG")
        #: switches every listed reference phase off or on at once (C6); it never takes input focus
        self.phases = QtWidgets.QPushButton("Phases")
        self.phases.setObjectName("e3PhasesToggle")
        self.phases.setCheckable(True)
        self.phases.setChecked(True)
        self.phases.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.phases.setEnabled(False)
        self.phases.setToolTip(PHASES_TOOLTIP + ' ' + PHASES_NONE_LISTED)
        self.title = QtWidgets.QLabel("Current")
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.color_map = color_map_combo()  # Int's list, shared with the 3D Viewer (shell_widgets)
        self._color_map_choices = color_map_choices()
        self.log_scale = log_toggle()
        self.raw_popup_button = QtWidgets.QPushButton("Raw")
        row.addWidget(self.norm)
        row.addWidget(self.background)
        row.addWidget(self.phases)
        row.addWidget(self.raw_popup_button)
        self._top_bar_row = row
        self._page_tool_buttons: dict[str, QtWidgets.QPushButton] = {}
        row.addWidget(self.title, 1)
        row.addWidget(self.color_map)
        row.addWidget(self.log_scale)
        self.norm.currentTextChanged.connect(
            lambda value: self._emit(
                ShellCommandKind.SET_NORM_CHANNEL, value
            )
        )
        self.background.clicked.connect(
            lambda: self._emit(ShellCommandKind.SET_BACKGROUND)
        )
        self.phases.clicked.connect(lambda on: self._emit(
            ShellCommandKind.SET_PLOT_OPTION, bool(on), ("other", "phase_references_shown")))
        self.color_map.currentTextChanged.connect(
            lambda value: self._emit(ShellCommandKind.SET_COLOR_MAP, value)
        )
        self.log_scale.toggled.connect(
            lambda value: self._emit(
                ShellCommandKind.SET_LOG_SCALE, bool(value)
            )
        )
        self.raw_popup_button.clicked.connect(self._open_raw_popup)
        return row

    def _ensure_raw_popup(self) -> None:
        if self.raw_popup_dialog is not None:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Raw Image")
        dialog.resize(600, 600)
        layout = QtWidgets.QVBoxLayout(dialog)
        controls = QtWidgets.QHBoxLayout()
        self.raw_popup_thumbnail = QtWidgets.QPushButton("Thumbnail")
        self.raw_popup_full = QtWidgets.QPushButton("Full Raw")
        self.raw_popup_full.setToolTip(
            "Read the full-resolution detector image with the saved reduction's value masking. "
            "The original file is unchanged; unavailable originals fall back to the saved thumbnail.")
        self.raw_popup_thumbnail.setToolTip("Show the saved detector thumbnail without reading the original image.")
        self.raw_popup_mode_group = QtWidgets.QButtonGroup(dialog)
        self.raw_popup_mode_group.setExclusive(True)
        for button in (self.raw_popup_thumbnail, self.raw_popup_full):
            button.setCheckable(True)
            self.raw_popup_mode_group.addButton(button)
            controls.addWidget(button)
        controls.addStretch(1)
        self.raw_popup_intensity = IntensityControls(dialog, percentiles=(2, 98))
        controls.addWidget(self.raw_popup_intensity)
        self.raw_popup_intensity.rangeChanged.connect(lambda _lo, _hi: self._refresh_raw_popup_intensity())
        self.raw_popup_intensity.autoToggled.connect(lambda _on: self._refresh_raw_popup_intensity())
        layout.addLayout(controls)
        self.raw_popup_image = ScientificImagePane(lock_aspect=True)
        layout.addWidget(self.raw_popup_image, 1)
        self.raw_popup_status = QtWidgets.QLabel("")
        layout.addWidget(self.raw_popup_status)
        self.raw_popup_thumbnail.clicked.connect(lambda: self._emit(
            ShellCommandKind.SET_DETECTOR_MODE, "thumbnail", ("popup",)))
        self.raw_popup_full.clicked.connect(lambda: self._emit(
            ShellCommandKind.SET_DETECTOR_MODE, "full", ("popup",)))
        dialog.finished.connect(self._release_raw_popup)
        self.raw_popup_dialog = dialog

    def _open_raw_popup(self) -> None:
        if not self.raw_popup_button.isEnabled():
            return
        self._ensure_raw_popup()
        self._raw_popup_host = False
        self.raw_popup_dialog.setWindowTitle("Raw Image")
        self.raw_popup_dialog.show(); self.raw_popup_dialog.raise_()
        self._emit(ShellCommandKind.SET_DETECTOR_MODE, "thumbnail", ("popup",))
        self.raw_popup_full.setChecked(True)
        self._emit(ShellCommandKind.SET_DETECTOR_MODE, "full", ("popup",))

    def _release_raw_popup(self, *_args) -> None:
        if self.raw_popup_image is not None:
            self.raw_popup_image.clear_image_buffers()
        if self._processing_mode == 'Int 1D':
            self._clear_phase_popup()
            self._clear_grid_overlay()
        if hasattr(self, "raw_popup_status"): self.raw_popup_status.setText("")
        self.raw_popup_intensity.sync(None, None, reset=True)
        if self._raw_popup_host:
            self.rawPopupClosed.emit()
        elif self._processing_mode == "Int 1D":
            self._emit(ShellCommandKind.SET_DETECTOR_MODE, "thumbnail", ("popup",))

    # -- the raw pop-up for a host page's image (Codex's Stitch/Int reuse audit R2: Stitch's contributing
    # frame opens here, not in a page-built dialog). Int's own pop-up path above is unchanged.
    def show_raw_popup(self, title: str, *, mode: str = "thumbnail") -> None:
        """Open the view's raw pop-up for a host page's image, which the host then fills with
        ``show_raw_image`` and ``show_raw_status``. Its Thumbnail / Full Raw buttons emit
        ``SET_DETECTOR_MODE`` (path ``("popup",)``) for the host to read its image again in that
        mode; closing it clears the image and emits ``rawPopupClosed``. The host's image stays through
        the view's own reconciles and layout changes (Int's pop-up follows Int 1D only)."""
        if mode not in {"thumbnail", "full"}:
            raise ValueError("the raw pop-up mode is 'thumbnail' or 'full'")
        self._ensure_raw_popup()
        self._raw_popup_host = True
        self.raw_popup_dialog.setWindowTitle(title)
        (self.raw_popup_full if mode == "full" else self.raw_popup_thumbnail).setChecked(True)
        self.raw_popup_full.setEnabled(True)
        self.raw_popup_dialog.show(); self.raw_popup_dialog.raise_()

    def raw_popup_shown(self) -> bool:
        """Whether a host page's raw pop-up is open."""
        return (self._raw_popup_host and self.raw_popup_dialog is not None
                and self.raw_popup_dialog.isVisible())

    def raw_popup_mode(self) -> str:
        """The raw pop-up's mode: 'full' when Full Raw is checked, else 'thumbnail'."""
        return ("full" if self.raw_popup_dialog is not None and self.raw_popup_full.isChecked()
                else "thumbnail")

    def show_raw_image(self, image, *, status: str, color_map: str, log_scale: bool,
                       detector_shape: tuple[int, int] | None = None) -> None:
        """A host page's image in its open raw pop-up, with the line below it."""
        if not self.raw_popup_shown():
            return
        self._clear_phase_popup()  # a host's own image carries no ring geometry
        self.raw_popup_image.render(image, detector_shape=detector_shape, color_map=color_map,
                                    log_scale=log_scale)
        self.raw_popup_intensity.set_percentile_defaults((0.1, 99.9) if log_scale else (2, 98))
        self._refresh_raw_popup_intensity()
        self.raw_popup_status.setText(status)

    def _refresh_raw_popup_intensity(self) -> None:
        """The shared intensity owner maps the popup's already-displayed detector pixels."""
        pane, controls = self.raw_popup_image, self.raw_popup_intensity
        if pane.image.image is None or not np.isfinite(pane.canvas.displayed_image).any():
            controls.sync(None, None)
            return
        percentile = controls.percentile_values()
        levels = pane.canvas.percentile_levels(percentile) if percentile is not None else (
            pane.canvas.percentile_levels() if controls.autoscale.isChecked()
            else controls.values())
        domain = pane.canvas._display_domain
        pane.canvas.histogram.lo_lim = min(domain[0], levels[0])
        pane.canvas.histogram.hi_lim = max(domain[1], levels[1])
        pane.canvas.histogram.setLevels(levels)
        controls.sync(domain, levels)

    def show_raw_status(self, status: str, *, clear_image: bool = False) -> None:
        """The line below a host page's open raw pop-up; ``clear_image`` empties the image too."""
        if not self.raw_popup_shown():
            return
        if clear_image:
            self.raw_popup_image.clear_image_buffers()
        self.raw_popup_status.setText(status)

    def _reconcile_raw_popup(self, state: ScientificProjection) -> None:
        if self._raw_popup_host:  # a host page's image: Int's frame does not replace it
            return
        heavy = state.heavy
        dialog = self.raw_popup_dialog
        if dialog is None or not dialog.isVisible():
            return
        if heavy is None or heavy.raw is None:
            self.raw_popup_image.clear_image_buffers()
            return
        if heavy.detector_source != "full" and not self.raw_popup_image.render_matches(
            heavy.raw, detector_shape=heavy.detector_shape,
            color_map=state.color_map, log_scale=state.log_scale,
        ):
            self.raw_popup_image.clear_image_buffers()
        self.raw_popup_image.render(
            heavy.raw, detector_shape=heavy.detector_shape,
            color_map=state.color_map, log_scale=state.log_scale,
            level_scan_token=(id(heavy.frame), id(heavy.raw)),
        )
        self.raw_popup_intensity.set_percentile_defaults((0.1, 99.9) if state.log_scale else (2, 98))
        self._refresh_raw_popup_intensity()
        self.raw_popup_thumbnail.setChecked(state.detector_mode == "thumbnail")
        self.raw_popup_full.setChecked(state.detector_mode == "full")
        self.raw_popup_full.setEnabled(state.detector_available)
        self.raw_popup_status.setText(
            "Loading Full Raw…" if state.detector_pending
            else state.detector_diagnostic if state.detector_mode == "full"
            else ""
        )

    def reconcile_action_availability(
        self, state: ScientificProjection,
    ) -> None:
        """Update action gating without touching either scientific canvas."""

        self.background.setEnabled(state.background_enabled)

    def _make_plot_bar(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(0)
        self.image_axis = ContentFitComboBox()
        for label, value in _STANDARD_IMAGE_AXIS_CHOICES:
            self.image_axis.addItem(label, value)
        self.plot_axis = ContentFitComboBox()
        for label, value in _STANDARD_PLOT_AXIS_CHOICES:
            self.plot_axis.addItem(label, value)
        self.slice = QtWidgets.QPushButton("χ (c/w)")
        self.slice.setCheckable(True)
        self.slice_center = spin(-180.0, 180.0, 0.0)
        self.slice_width = spin(0.0, 360.0, 10.0)
        self.pin = QtWidgets.QPushButton("Pin")
        self.plot_mode = ContentFitComboBox()
        self.plot_mode.addItems(list(self._PLOT_MODES))
        self.options = _CompactMenuButton("Options")
        self.options.setToolTip("Edit plot options and display limits. Display limit changes apply immediately.")
        self.plot_options_dialog = PlotOptionsDialog(self)
        from .reference_phases import ReferencePhasesWindow
        self.reference_phases = ReferencePhasesWindow(self)
        self.reference_phases.optionEdited.connect(
            lambda path, value: self._emit(ShellCommandKind.SET_PLOT_OPTION, value, path))
        self._grid_cache = self._grid_prediction = self._grid_frame = None
        self._phase_job = PhaseOverlayJob()
        self._phase_requests = LatestRequestOwner(self._phase_job, self._phase_done,
                                                  same=lambda a, b: a == b, parent=self)
        self.phase_worker = self._phase_requests.worker
        self.phase_families = {}
        self._phase_markers, self._phase_legends, self._phase_colors = {}, {}, {}
        self._phase_drawings = {}
        self._phase_line_context = None
        self._phase_wavelengths = ()
        self._phase_display_contract = None
        #: the last accepted detector-ring context (frame, snapshot, shape), kept across a retained display
        self._phase_ring_context = None
        #: every listed phase with its include flag, as the window resolved it (legend entries and colours)
        self._phase_listed = ()
        #: each listed phase's colour slot, kept from when it joined the list until it leaves it
        self._phase_color_slots = {}
        self.clear = QtWidgets.QPushButton("Clear")
        self.share_axis = QtWidgets.QPushButton("Share Axis")
        self.share_axis.setCheckable(True)
        self.share_axis.setMinimumWidth(90)
        self.previous_frame = QtWidgets.QToolButton()
        self.previous_frame.setObjectName("e4FrameNavigationButton")
        self.previous_frame.setText("<")
        self.frame_selector = CompactFrameSelector()
        self.frame_selector.setObjectName("e3FrameSelector")
        self.next_frame = QtWidgets.QToolButton()
        self.next_frame.setObjectName("e4FrameNavigationButton")
        self.next_frame.setText(">")
        self.axis_display_group = _plot_toolbar_group(
            "e4AxisDisplayControls",
            (
                self.plot_axis,
                self.slice,
                self.slice_center,
                self.slice_width,
                self.pin,
            ),
        )
        self.plot_action_group = _plot_toolbar_group(
            "e4PlotActionControls",
            (
                self.plot_mode,
                self.options,
                self.clear,
            ),
        )
        row.addWidget(self.axis_display_group)
        row.addSpacing(PLOT_TOOLBAR_INTER_GROUP_GAP)
        self._plot_group_gap = row.itemAt(row.count() - 1).spacerItem()
        row.addWidget(self.plot_action_group)
        row.addStretch(1)
        row.addWidget(self.share_axis)
        row.addSpacing(PLOT_TOOLBAR_INTRA_GROUP_GAP)
        row.addWidget(self.image_axis)
        self.image_axis.currentIndexChanged.connect(
            lambda index: self._emit(
                ShellCommandKind.SET_IMAGE_AXIS,
                self.image_axis.itemData(index),
            )
        )
        self.plot_axis.currentIndexChanged.connect(
            lambda index: self._emit(
                ShellCommandKind.SET_PLOT_AXIS,
                self.plot_axis.itemData(index),
            )
        )
        self.slice.toggled.connect(
            lambda value: self._emit(
                ShellCommandKind.SET_SLICE_ENABLED, bool(value)
            )
        )
        self.slice_center.valueChanged.connect(
            lambda value: self._emit(
                ShellCommandKind.SET_SLICE_CENTER, float(value)
            )
        )
        self.slice_width.valueChanged.connect(
            lambda value: self._emit(
                ShellCommandKind.SET_SLICE_WIDTH, float(value)
            )
        )
        self.pin.clicked.connect(
            lambda: self._emit(ShellCommandKind.PIN_SLICE)
        )
        self.plot_mode.currentTextChanged.connect(
            lambda value: self._emit(
                ShellCommandKind.SET_PLOT_MODE, value
            )
        )
        self.options.clicked.connect(self._show_plot_options)
        self.plot_options_dialog.accepted.connect(
            self._accept_plot_options
        )
        self.clear.clicked.connect(
            lambda: self._emit(ShellCommandKind.CLEAR_1D)
        )
        self.share_axis.toggled.connect(
            lambda value: self._emit(
                ShellCommandKind.SET_SHARE_AXIS, bool(value)
            )
        )
        self.frame_selector.currentIndexChanged.connect(
            self._frame_selected
        )
        self.previous_frame.clicked.connect(lambda: self._navigate(-1))
        self.next_frame.clicked.connect(lambda: self._navigate(1))
        return row

    def show_reference_phases(self) -> None:
        """The Tools panel's Reference Phases window (REFERENCE-PHASES-20261011)."""
        self.reference_phases.show()
        self.reference_phases.raise_()
        self.reference_phases.activateWindow()

    def _show_plot_options(self) -> None:
        self._emit(ShellCommandKind.SHOW_WATERFALL_OPTIONS)
        self.plot_options_dialog.show()
        self.plot_options_dialog.raise_()
        self.plot_options_dialog.activateWindow()

    def _accept_plot_options(self) -> None:
        prior = self.plot_options_dialog.projection
        edited = self.plot_options_dialog.edited_options()
        self.plotOptionsAccepted.emit()
        for path, field in PLOT_OPTION_BINDINGS:
            old_value = getattr(prior, field)
            new_value = getattr(edited, field)
            if new_value != old_value:
                self._emit(
                    ShellCommandKind.SET_PLOT_OPTION,
                    new_value,
                    path,
                )

    def _remove_phase_drawings(self, identifier):
        family = 'phase:' + identifier
        self._phase_drawings.pop(identifier, None)
        for pane in (self.raw, self.cake, self.raw_popup_image):
            if pane is not None:
                pane.render_overlay_family(family, ())
        marker = self._phase_markers.pop(identifier, None)
        if marker is not None:
            self.curve.removeItem(marker)
        self.phase_families.pop(identifier, None)

    def _remove_phase_family(self, identifier):
        self._remove_phase_drawings(identifier)
        legend = self._phase_legends.pop(identifier, None)
        if legend is not None:
            self.legend.removeItem(legend)

    def _clear_phase_popup(self):
        if self.raw_popup_image is not None:
            for family in tuple(self.raw_popup_image.overlay_families):
                if family.startswith('phase:'):
                    self.raw_popup_image.render_overlay_family(family, ())

    def _clear_phase_overlays(self):
        self._phase_requests.release()
        self._phase_job.clear_cache()
        for identifier in set(self.phase_families) | set(self._phase_legends) | set(self._phase_markers):
            self._remove_phase_family(identifier)
        self._phase_drawings.clear()
        self._phase_colors.clear()
        self._phase_line_context = None
        self._phase_wavelengths = ()
        self._phase_display_contract = None
        self._phase_ring_context = None
        self._phase_listed = ()
        self._phase_color_slots = {}

    def shutdown_phase_overlays(self):
        """Close the phase worker and the window's Add CIF… worker: CLEAN, or PENDING while either runs (the host
        asks again)."""
        self.reference_phases.close()  # the view's window does not outlive its overlays
        self._clear_phase_overlays()
        receipts = (self._phase_requests.shutdown(), self.reference_phases.phase_source.shutdown())
        return next((receipt for receipt in receipts if receipt.status is not PageCleanup.CLEAN), receipts[0])

    @staticmethod
    def _phase_axis(axis):
        if axis is None:
            return None
        key = ScientificView._axis_key(axis)
        values = np.asarray(axis.values)
        values = values[np.isfinite(values)]
        return (key, (float(values.min()), float(values.max()))) if values.size else None

    def _sync_phases_button(self, options, available):
        """The Phases toggle (C6): its state follows the option; it switches something only while a phase is
        listed, and says so."""
        listed = available and bool(options.phase_references)
        blocker = QtCore.QSignalBlocker(self.phases)
        self.phases.setChecked(options.phase_references_shown)
        del blocker
        self.phases.setEnabled(listed)
        self.phases.setToolTip(PHASES_TOOLTIP + (' Now on.' if options.phase_references_shown else ' Now off.')
                               + ('' if listed else ' ' + PHASES_NONE_LISTED))

    def _reconcile_phase_overlay(self, state, navigation):
        from xrd_tools import geometry_figures as gf
        options, window = state.plot_options, self.reference_phases
        visible = state.processing_mode != '2D Viewer'
        window.set_available(visible, '' if visible else 'Reference phases are unavailable in the 2D Viewer.')
        window.reconcile(options)
        self._sync_phases_button(options, visible)
        listed = window.phase_source.listed_entries() if visible else ()
        listed_ids = {entry.id for entry, _shown in listed}
        # Switched off (C6): nothing draws, while the list, each include flag, every minimum and the colour slots stay.
        switched_off = not options.phase_references_shown
        entries = () if switched_off else tuple(entry for entry, shown in listed if shown)
        shown_ids = {entry.id for entry in entries}
        for identifier in set(self.phase_families) | set(self._phase_legends) | set(self._phase_markers):
            if identifier not in listed_ids or switched_off:
                self._remove_phase_family(identifier)
            elif identifier not in shown_ids:
                self._remove_phase_drawings(identifier)  # a hidden phase keeps its greyed legend entry
        # Each phase keeps the colour slot it joined with, so hiding or removing a phase recolours no other.
        slots = {identifier: slot for identifier, slot in self._phase_color_slots.items() if identifier in listed_ids}
        for entry, _shown in listed:
            if entry.id not in slots:
                slots[entry.id] = min(set(range(len(listed) + len(slots))) - set(slots.values()))
        self._phase_color_slots = slots
        palette = reference_colors(max(slots.values(), default=-1) + 1)
        self._phase_colors = {identifier: palette[slot] for identifier, slot in slots.items()}
        self._phase_listed = () if switched_off else listed
        if not entries:
            self._phase_requests.release()
            window.set_rings_reason(PHASES_OFF_NOTE if switched_off and listed else '')
            self._draw_phase_legends()
            return
        wavelengths = self._phase_wavelengths
        wavelength = wavelengths[0]*1e10 if len(wavelengths) == 1 and wavelengths[0] is not None else None
        context = self._phase_line_context
        cake = self._phase_axis(self._rendered_cake_x_axis)
        if context is None or context[0] not in _CONVERTIBLE_RADIAL_AXIS_KEYS:
            context = cake
        key, bounds = context if context is not None else ('', (0., 1.))
        unit = {'q_A^-1': 'q', '2th_deg': '2theta'}.get(key, '')
        reason = ''
        if state.measurement_mode == 'Grazing':
            reason = 'Reference lines and maps require Standard q or 2θ coordinates; GI detector rings remain detector-frame references.'
        elif not unit:
            reason = 'Reference phases require a q or 2θ axis.'
        elif unit == '2theta' and wavelength is None:
            reason = '2θ reference lines require one displayed energy; mixed or missing energies cannot share reference lines.'
        heavy = state.heavy
        is_int = (state.processing_mode == 'Int 1D' and 'raw' not in self._hidden_tools
                  or state.processing_mode == 'Int 2D' and 'raw_pane' not in self._hidden_tools)
        raw_shown = is_int or state.processing_mode in ('Average', 'Reintegrate')
        snapshot, shape, retained = (), None, self._phase_ring_context
        if (heavy is None and state.retain_display and is_int and retained is not None
                and retained[0] is navigation.current):
            snapshot, shape = retained[1], retained[2]  # the accepted image+geometry pair (R10)
            ring_reason = ''
        else:
            current = is_int and heavy is not None and heavy.frame is navigation.current and heavy.raw is not None
            geometry = heavy.ring_geometry if current else None
            ring_reason = gf.detector_ring_unavailable(geometry, heavy.detector_shape if current else None)
            if current and not ring_reason:
                snapshot, shape = tuple(geometry.items()), heavy.detector_shape
                self._phase_ring_context = (navigation.current, snapshot, shape)
            else:
                self._phase_ring_context = None
        window.set_rings_reason(ring_reason if raw_shown else '')
        self._phase_display_contract = (state.measurement_mode, self._phase_line_context, cake,
            self._phase_axis(self._rendered_cake_y_axis), wavelength, shape, bool(snapshot), state.processing_mode)
        floors = () if options.phase_global_floor else tuple(
            pair for pair in options.phase_floors if pair[0] in shown_ids)
        request = PhaseOverlayRequest(entries, options.phase_min_intensity, unit, bounds, wavelength,
                                      snapshot, shape, line_reason=reason, floors=floors)
        shown = self._phase_requests.shown
        if shown is not None and request != shown:
            def context(value):
                return (value.line_unit, value.line_range, value.wavelength_angstrom, value.ring_snapshot,
                        value.ring_shape, value.step_px, value.line_reason)
            if context(shown) != context(request):
                for identifier in tuple(self.phase_families):
                    self._remove_phase_drawings(identifier)
            else:
                before = {entry.id: (entry, phase_floor(shown, entry)) for entry in shown.entries}
                for entry in entries:
                    if before.get(entry.id) != (entry, phase_floor(request, entry)):
                        self._remove_phase_drawings(entry.id)  # only the phase that changed waits for its lines
        self._phase_requests.ask(request)
        self._draw_phase_overlays()

    def _phase_done(self, request, terminal):
        if terminal.status is OperationTerminalStatus.RETURNED:
            self.phase_families = {family.id: family for family in terminal.payload}
        else:
            from xrd_tools.analysis.phase_overlays import PhaseOverlayFamily
            self.phase_families = {entry.id: PhaseOverlayFamily(entry.id, entry.name, entry.source_kind,
                reason=str(terminal.failure_message or 'Reference phase computation failed.')) for entry in request.entries}
        self._draw_phase_overlays()
        return True

    @staticmethod
    def _phase_tip(family):
        from xrd_tools.analysis.phase_overlays import RING_CAVEAT
        if family.source_kind == 'calibrant':
            tip = f'{family.name} (built-in calibrant): reference positions from pyFAI d-spacings. {RING_CAVEAT}'
        else:
            tip = f'{family.name}: {family.omitted_count} reflections omitted below the floor. Display reference only.'
        return tip + (' ' + family.reason if family.reason else '')

    def _draw_phase_legends(self):
        for entry, shown in self._phase_listed:
            identifier = entry.id
            color = self._phase_colors.get(identifier, (240, 200, 90))
            family = self.phase_families.get(identifier)
            tip = (self._phase_tip(family) if family is not None and shown else
                   f'{entry.name}: hidden. Click to show it again. Display reference only.' if not shown else
                   f'{entry.name}: computing reference positions. Display reference only.')
            legend = self._phase_legends.get(identifier)
            pen = pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.DashLine)  # the lines' dash (D4)
            if legend is None:
                legend = pg.PlotDataItem(pen=pen)
                self._phase_legends[identifier] = legend
            if not any(item.item is legend for item, _label in self.legend.items):
                self.legend.addItem(legend, entry.name)
            legend.setPen(pen)
            legend.setVisible(bool(shown))  # pyqtgraph greys a hidden sample
            legend.setToolTip(tip)
            for sample, label in self.legend.items:
                if sample.item is legend:  # the entry the legend shows, not only its proxy
                    sample.setToolTip(tip)
                    label.setToolTip(tip)
                    sample.update()

    def _legend_sample_clicked(self, item):
        identifier = next((key for key, legend in self._phase_legends.items() if legend is item), None)
        if identifier is None:
            return  # a trace's own entry: pyqtgraph shows or hides the trace itself
        shown = bool(item.isVisible())  # the sample toggled its proxy before this signal
        options = self.reference_phases.projection
        references = tuple((key, shown if key == identifier else flag) for key, flag in options.phase_references)
        if references != options.phase_references:
            self._emit(ShellCommandKind.SET_PLOT_OPTION, references, ('other', 'phase_references'))

    def _draw_phase_overlays(self):
        from xrd_tools.geometry_figures import Curve
        self._draw_phase_legends()
        if self._phase_display_contract is None:
            return
        measurement, line, cake, cake_y, wavelength, shape, raw_allowed, mode = self._phase_display_contract
        line_key = line[0] if line else ''
        for identifier, family in self.phase_families.items():
            name = 'phase:' + identifier
            color = self._phase_colors.get(identifier, (240, 200, 90))
            tip = self._phase_tip(family)
            positions = positions_for_axis(family.q_lines,
                {'q_A^-1': 'q', '2th_deg': '2theta'}.get(line_key, ''), wavelength)
            values = positions.values if measurement != 'Grazing' else ()
            # One joined 1D item per phase (maintainer, October 11), not one item per reflection.
            marker = self._phase_markers.get(identifier)
            if (marker is None or marker.reference_positions != tuple(values)
                    or marker not in self.curve.getPlotItem().items
                    or marker.lines.opts['pen'].color().getRgb()[:3] != color[:3]):
                if marker is not None:
                    self.curve.removeItem(marker)
                marker = draw_reference_lines(self.curve, values, color=color, family=name,
                                              tooltip=tip) if values else None
                if marker is None:
                    self._phase_markers.pop(identifier, None)
                else:
                    self._phase_markers[identifier] = marker
            if marker is not None:
                marker.set_tooltip(tip)
            # The family object, unchanged by another phase's edit, keeps its curves and so every pane's items.
            scalars = (measurement, cake, cake_y, wavelength, color, tip)
            drawing = self._phase_drawings.get(identifier)
            if drawing is None or drawing[0] is not family or drawing[1] != scalars:
                rings = tuple(replace(curve, tooltip=tip) for curve in family.rings)
                cuts = ()
                if measurement != 'Grazing' and cake is not None and cake_y is not None:
                    cake_values = positions_for_axis(family.q_lines,
                        {'q_A^-1': 'q', '2th_deg': '2theta'}.get(cake[0], ''), wavelength).values
                    cuts = tuple(Curve(family.name, np.asarray(((value, cake_y[1][0]), (value, cake_y[1][1]))),
                                       'predicted', tip) for value in cake_values)
                drawing = (family, scalars, rings, cuts)
                self._phase_drawings[identifier] = drawing
            self.cake.render_overlay_family(name, drawing[3], color=color, joined=True)
            raw_curves = drawing[2] if raw_allowed else ()
            self.raw.render_overlay_family(name, raw_curves if mode == 'Int 2D' else (),
                                           native_shape=shape, color=color, joined=True)
            if self.raw_popup_image is not None:
                own_popup = mode == 'Int 1D' and not self._raw_popup_host and self.raw_popup_dialog.isVisible()
                self.raw_popup_image.render_overlay_family(name, raw_curves if own_popup else (),
                                                           native_shape=shape, color=color, joined=True)

    def _clear_grid_overlay(self, *, retain_cache=False):
        self._grid_frame = None
        if not retain_cache:
            self._grid_cache = self._grid_prediction = None
        self.raw.render_overlay_family('grid', ())
        if self.raw_popup_image is not None:
            self.raw_popup_image.render_overlay_family('grid', ())

    def _reconcile_grid_overlay(self, state, navigation):
        from xrd_tools import geometry_figures as gf
        heavy, mode = state.heavy, state.processing_mode
        shown = (mode in ('Average', 'Reintegrate')
                 or mode == 'Int 1D' and 'raw' not in self._hidden_tools
                 or mode == 'Int 2D' and 'raw_pane' not in self._hidden_tools)
        is_int = shown and mode in ('Int 1D', 'Int 2D')
        if (heavy is None and state.retain_display and is_int
                and self._grid_frame is navigation.current):
            return
        matches = is_int and heavy is not None and heavy.frame is navigation.current and heavy.raw is not None
        snapshot, shape = ((heavy.ring_geometry, heavy.detector_shape) if matches else (None, None))
        reason = gf.detector_grid_unavailable(snapshot, shape)
        self.plot_options_dialog.set_grid_available(not reason, reason, visible=shown)
        choice = state.plot_options.coordinate_grid
        if reason or choice == 'Off':
            self._clear_grid_overlay(retain_cache=True)
            return
        dimension = '1d' if mode == 'Int 1D' else '2d'
        key = (tuple(snapshot.items()), shape, choice, dimension)
        if self._grid_cache != key:
            self._clear_grid_overlay()
            try:
                self._grid_prediction = gf.predict_detector_grid(snapshot, shape=shape, choice=choice,
                                                                  mode=dimension)
            except ValueError as error:
                self.plot_options_dialog.set_grid_available(False, str(error))
                return
            self._grid_cache = key
        self._grid_frame = navigation.current
        target = self.raw if mode == 'Int 2D' else self.raw_popup_image
        if target is not None and (mode == 'Int 2D' or
                                  self.raw_popup_dialog.isVisible() and not self._raw_popup_host):
            target.render_overlay_family('grid', self._grid_prediction.curves,
                                         texts=self._grid_prediction.texts, native_shape=shape)

    def reconcile(
        self,
        state: ScientificProjection,
        navigation: FrameNavigationProjection,
        *,
        completed: int,
        total: int,
        detail: str,
    ) -> None:
        self.reconcile_action_availability(state)
        if state.processing_mode == '2D Viewer':
            self._clear_grid_overlay()
            self.plot_options_dialog.set_grid_available(False, visible=False)
            self._clear_phase_overlays()
            self.reference_phases.set_available(False, 'Reference phases are unavailable in the 2D Viewer.')
        coverage = (
            state.heavy.cake_coverage
            if state.heavy is not None
            and state.processing_mode == "Int 2D"
            and state.image_axis in {"Q-Chi", "q_chi"}
            and not state.derived_2d
            else None
        )
        self.plot_options_dialog.set_coverage_available(coverage is not None)
        if coverage is None or not state.plot_options.show_coverage_holes:
            self.cake.render_coverage(None)
        self.background.setText(("Clear" if state.background_set else "Set") + " " + {"Int 1D": "1D", "1D Viewer": "1D", "Int 2D": "2D", "2D Viewer": "Raw"}.get(state.processing_mode, "BG") + " BG")
        self._rendered_background_key = self._expected_background_key if state.background_set else None
        if state.processing_mode == "2D Viewer":
            entering_viewer = self._processing_mode != "2D Viewer"
            self._processing_mode = "2D Viewer"
            if state.heavy is None:
                if self._viewer_2d_known_empty and self._frame_keys:
                    blocker = QtCore.QSignalBlocker(self.frame_selector)
                    self._rebuild_frames(navigation.frames, navigation.current)
                    del blocker
                elif not ScientificView.clear_viewer_2d(self, None, failure=True):
                    raise RuntimeError("2D Viewer render failed.") from None
                self.title.setText(state.title)
                self.status.setText(state.status)
                self.progress.setText("0/0")
                return
            try:
                frame = navigation.current
                heavy = state.heavy
                if frame is None or heavy.frame is not frame or heavy.raw is None:
                    raise ValueError("viewer projection is incomplete")
                detector_source = getattr(
                    heavy, "detector_source", "full",
                )
                source_transition = (
                    not entering_viewer
                    and not self._viewer_2d_known_empty
                    and self._rendered_detector_source != detector_source
                )
                if ((entering_viewer or source_transition)
                        and not ScientificView.clear_viewer_2d(
                            self, None, failure=True)):
                    raise ValueError("viewer reset is incomplete")
                blockers = [QtCore.QSignalBlocker(widget) for widget in (
                    self.frame_selector, self.color_map, self.log_scale)]
                retained_range = (
                    self.raw.canvas.imageViewBox.targetRect()
                    if self._viewer_2d_known_empty and self._frame_keys
                    and getattr(self, "_viewer_2d_shape", None) == heavy.raw.shape
                    else None
                )
                self.raw.render(heavy.raw, detector_shape=heavy.detector_shape,
                                color_map=state.color_map,
                                log_scale=state.log_scale,
                                level_scan_token=(id(frame), id(heavy.raw)),
                                view_range=retained_range)
                self._rendered_detector_source = detector_source
                self.raw.canvas.imageItem.pos_label.setText("")
                self.cake.canvas.imageItem.pos_label.setText("")
                self._heavy_available = state.heavy_available
                self._selected_keys = navigation.selected
                self._plot_mode, self._single_mode = "Single", True
                self._rebuild_frames(navigation.frames, frame)
                set_combo_value(self.color_map, state.color_map, fallback="Default")
                self.log_scale.setChecked(state.log_scale)
                self._viewer_2d_payload = heavy.raw
                self._viewer_2d_shape = heavy.raw.shape
                self._viewer_2d_known_empty = False
                self.title.setText(state.title)
                self.status.setText(state.status)
                self.progress.setText(f"{completed}/{total}")
                current = next((index for index, item in enumerate(navigation.frames)
                                if item is frame), None)
                self.previous_frame.setEnabled(current is not None and current > 0)
                self.next_frame.setEnabled(
                    current is not None and current < len(navigation.frames) - 1)
                self._apply_processing_layout("2D Viewer")
                self._refresh_viewer_intensity()
                del blockers
            except Exception:
                ScientificView.clear_viewer_2d(self, None, failure=True)
            else:
                return
            raise RuntimeError("2D Viewer render failed.") from None
        slice_contract = (
            state.plot_axis,
            state.slice_enabled,
            state.slice_center,
            state.slice_width,
            tuple(pin.projection_id for pin in state.slice_pins),
        )
        slice_contract_changed = (
            self._rendered_slice_contract is not None
            and slice_contract != self._rendered_slice_contract
        )
        widgets = (
            self.norm,
            self.color_map,
            self.log_scale,
            self.image_axis,
            self.plot_axis,
            self.slice,
            self.slice_center,
            self.slice_width,
            self.plot_mode,
            self.share_axis,
            self.frame_selector,
        )
        blockers = [QtCore.QSignalBlocker(widget) for widget in widgets]
        set_combo(self.norm, state.norm_channels, state.norm_channel)
        color_map = set_combo_value(
            self.color_map,
            state.color_map,
            fallback="Default",
        )
        self.log_scale.setChecked(state.log_scale)
        replace_presentation = bool(
            state.browse_trace_snapshot is not None
            or state.heavy is not None
            or not state.retain_display
        )
        if replace_presentation:
            self._reconcile_axis_choices(state)
        self._set_data_combo(self.image_axis, state.image_axis)
        self._set_data_combo(self.plot_axis, state.plot_axis)
        index = self.plot_mode.findText(state.plot_mode)
        if index >= 0:
            self.plot_mode.setCurrentIndex(index)
        self.slice.setChecked(state.slice_enabled)
        self.slice_center.setValue(state.slice_center)
        self.slice_width.setValue(state.slice_width)
        self.share_axis.setChecked(state.share_axis)
        self.plot_options_dialog.reconcile(state.plot_options)
        self._apply_processing_layout(state.processing_mode)
        self.raw_popup_button.setEnabled(state.detector_available)
        explanation = state.detector_diagnostic if not state.detector_available else ""
        self.raw_popup_button.setToolTip(explanation or "Show exact-current raw image")
        if replace_presentation:
            # A GI frame offers a choice only when it has more than one map.
            self.image_axis.setEnabled(
                state.measurement_mode != "GI" or len(state.gi_maps) > 1
            )
        if replace_presentation:
            self.title.setText(state.title)
        self._heavy_available = state.heavy_available
        self._plot_mode = state.plot_mode
        self._single_mode = state.plot_mode == "Single"
        self._selected_keys = navigation.selected
        footer_frames = (
            tuple(
                frame
                for frame in navigation.frames
                if frame.artifact == navigation.current.artifact
            )
            if navigation.current is not None
            else ()
        )
        self._rebuild_frames(footer_frames, navigation.current)
        prior_syncing = self._share_axis_syncing
        self._share_axis_syncing = True
        try:
            if state.heavy is not None:
                if state.heavy.raw is not None and state.processing_mode != "Int 1D":
                    raw_matches = self.raw.render_matches(
                        state.heavy.raw,
                        detector_shape=state.heavy.detector_shape,
                        color_map=color_map,
                        log_scale=state.log_scale,
                    )
                    if state.heavy.detector_source != "full" and (
                        # Thumbnail replacement overwrites the existing image;
                        # a source-kind transition still scrubs full buffers.
                        (not raw_matches and state.heavy.detector_source != "thumbnail")
                        or self._rendered_detector_source
                        != state.heavy.detector_source
                    ):
                        self.raw.clear_image_buffers()
                    self.raw.render(
                        state.heavy.raw,
                        detector_shape=state.heavy.detector_shape,
                        color_map=color_map,
                        log_scale=state.log_scale,
                        level_scan_token=(
                            id(state.heavy.frame),
                            id(state.heavy.raw),
                        ),
                    )
                    self._rendered_detector_source = (
                        state.heavy.detector_source
                    )
                else:
                    self.raw.clear_image_buffers()
                    self._rendered_detector_source = "none"
                if state.heavy.cake is not None:
                    self.cake.render(
                        state.heavy.cake,
                        x_axis=state.heavy.cake_x,
                        y_axis=state.heavy.cake_y,
                        color_map=color_map,
                        log_scale=state.log_scale,
                        autorange_finite_y=state.derived_2d,
                        level_scan_token=(
                            id(state.heavy.frame),
                            id(state.heavy.cake),
                        ),
                    )
                    self.cake.render_coverage(
                        coverage if state.plot_options.show_coverage_holes else None
                    )
                    self._rendered_cake_axis_key = self._axis_key(
                        state.heavy.cake_x
                    )
                    self._rendered_cake_x_axis = state.heavy.cake_x
                    self._rendered_cake_y_axis = state.heavy.cake_y
                    self._rendered_image_axis = state.image_axis
                else:
                    self.cake.clear_image_buffers()
                    self._rendered_cake_axis_key = None
                    self._rendered_cake_x_axis = None
                    self._rendered_cake_y_axis = None
                    self._rendered_image_axis = None
            elif (
                state.retain_display
                and state.browse_trace_snapshot is None
            ):
                # A qualified hydration is pending.  Keep the last accepted
                # title/images/traces together until its exact result arrives.
                pass
            else:
                # An accepted absence is a complete transition, not permission
                # to leave a stale raw/cake hybrid on screen.
                self.raw.clear_image_buffers()
                self._rendered_detector_source = "none"
                self.cake.clear_image_buffers()
                self._rendered_cake_axis_key = None
                self._rendered_cake_x_axis = None
                self._rendered_cake_y_axis = None
                self._rendered_image_axis = None
            if replace_presentation:
                self._render_traces(
                    state,
                    navigation,
                    live_update=state.live_update,
                )
            if state.processing_mode == "Int 1D":
                self._reconcile_raw_popup(state)
            self._reconcile_grid_overlay(state, navigation)
            self._reconcile_phase_overlay(state, navigation)
        finally:
            self._share_axis_syncing = prior_syncing
        if replace_presentation:
            self._reconcile_slice_extent(state)
        pin_available = (
            state.heavy is not None
            and state.plot_mode in {"Overlay", "Waterfall"}
            and state.slice_enabled
            and slice_region_orientation(
                state.plot_axis,
                self._rendered_cake_x_axis,
                self._rendered_cake_y_axis,
            ) is not None
        )
        self.pin.setEnabled(pin_available)
        self.pin.setToolTip(
            "Pin the current slice cut into the accumulating plot."
            if pin_available
            else (
                "Pin is available for an accepted 2-D slice in Overlay "
                "or Waterfall."
            )
        )
        self._apply_share_axis_state(state.share_axis)
        if (
            replace_presentation
            and slice_contract_changed
            and state.slice_enabled
            and self.curve.listDataItems()
        ):
            self._autorange_slice_projection(self._share_link_on)
        if replace_presentation:
            self._rendered_slice_contract = slice_contract
        self.status.setText(state.detector_diagnostic if state.detector_mode == "full" and not state.detector_pending and state.heavy is not None and state.heavy.detector_source != "full" else state.status or detail)
        self.progress.setText(f"{completed}/{total}")
        self._refresh_viewer_intensity()
        del blockers

    def _viewer_loading_snapshot_target(
        self, mode: str,
    ) -> QtWidgets.QWidget | None:
        if mode == "2D Viewer":
            return self.raw.canvas
        if mode in {"1D Viewer", "Int 1D"}:
            return self.bottom_stack.currentWidget()
        if mode == "Int 2D":
            return self.vertical_splitter
        return None

    def _layout_viewer_loading_snapshot(self) -> None:
        target = self._viewer_loading_target
        if target is None or not self._viewer_loading_overlay.isVisible():
            return
        origin = target.mapTo(self, QtCore.QPoint())
        self._viewer_loading_overlay.setGeometry(
            QtCore.QRect(origin, target.size())
        )
        self._viewer_loading_pixmap.setGeometry(
            self._viewer_loading_overlay.rect()
        )
        self._viewer_loading_overlay.raise_()

    def _begin_viewer_loading_snapshot(self, mode: str) -> None:
        """Keep one capped screen raster while the selected display reloads."""

        # A rapid replacement must continue to describe the original pending
        # view, never capture the loading layer or build a snapshot history.
        if self._viewer_loading_overlay.isVisible():
            return
        target = self._viewer_loading_snapshot_target(mode)
        if target is None or target.width() < 1 or target.height() < 1:
            return
        pixmap = target.grab()
        if pixmap.isNull():
            return
        pixels = _pixmap_pixel_count(pixmap)
        if pixels > MAX_VIEWER_LOADING_SNAPSHOT_PIXELS:
            scale = math.sqrt(MAX_VIEWER_LOADING_SNAPSHOT_PIXELS / pixels)
            pixmap = pixmap.scaled(
                max(1, int(pixmap.width() * scale)),
                max(1, int(pixmap.height() * scale)),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation,
            )
        self._viewer_loading_mode = mode
        self._viewer_loading_target = target
        self._watch_viewer_loading_geometry(target)
        self._viewer_loading_pixmap.setPixmap(pixmap)
        self._viewer_loading_overlay.show()
        self._layout_viewer_loading_snapshot()

    def _watch_viewer_loading_geometry(self, target: QtWidgets.QWidget) -> None:
        # The overlay covers the target in this view's coordinates, but the
        # target's row can still be re-laid while the load is pending (the
        # previous mode's rows collapsing, a splitter settling on a slow
        # host).  Follow every geometry change on the target and on each
        # ancestor below this view.  Filters stay installed: some of these
        # widgets already carry the share-axis hook, and removing the filter
        # would drop that too.
        chain = []
        widget: QtWidgets.QWidget | None = target
        while widget is not None and widget is not self:
            widget.installEventFilter(self)
            chain.append(widget)
            widget = widget.parentWidget()
        self._viewer_loading_watched = tuple(chain)

    def drop_viewer_loading_snapshot(self) -> None:
        """Release the transient raster without touching scientific payloads."""

        self._viewer_loading_mode = None
        self._viewer_loading_target = None
        self._viewer_loading_watched = ()
        self._viewer_loading_pixmap.clear()
        self._viewer_loading_overlay.hide()

    def clear_viewer_2d(self, request, *, failure=False, preserve_navigation=False):
        if not failure and type(request) is not Viewer2DRendererClearRequest:
            return None
        keep_chrome = bool(
            preserve_navigation and not failure and self._processing_mode == "2D Viewer"
            and self._current_key is not None
            and request.context_token == self._current_key.run_identity.fingerprint
            and request.label == self._current_key.local_frame_label
        )
        if keep_chrome:
            self._begin_viewer_loading_snapshot("2D Viewer")
        else:
            self.drop_viewer_loading_snapshot()
        cleared = ScientificView._clear_viewer_payloads(
            self, keep_chrome=keep_chrome, failure=failure,
        )
        if not cleared:
            self.drop_viewer_loading_snapshot()
        return cleared if failure else Viewer2DRendererClearReceipt(request, cleared)

    def _clear_viewer_payloads(self, *, keep_chrome=False, failure=False):
        """Release scientific buffers; navigation need not dismantle the layout."""
        self._clear_phase_overlays()
        cleared, canonical = True, self._viewer_2d_payload
        def scrub(target, name, value=None, *, read=False):
            nonlocal cleared
            try:
                member = getattr(target, name, None)
                if read:
                    return member
                if callable(member):
                    if value is None:
                        return member()
                    return member(value)
                setattr(target, name, value)
                actual = getattr(target, name, None)
                if (canonical is not None and (actual is canonical
                        or isinstance(actual, np.ndarray)
                        and np.shares_memory(actual, canonical))):
                    cleared = False
            except Exception:
                cleared = False
        def verify_released(target, name):
            nonlocal cleared
            value = scrub(target, name, read=True)
            if (
                canonical is not None
                and (
                    value is canonical
                    or isinstance(value, np.ndarray)
                    and np.shares_memory(value, canonical)
                )
            ):
                cleared = False
        for pane in (self.raw, self.cake):
            try:
                pane.clear_image_buffers(keep_chrome=keep_chrome)
            except Exception:
                cleared = False
            canvas = scrub(pane, "canvas", read=True)
            image = scrub(canvas, "imageItem", read=True)
            for target, name in (
                (canvas, "raw_image"), (canvas, "displayed_image"),
                (image, "image"), (image, "qimage"), (image, "levels"),
                (image, "_displayBuffer"), (image, "_processingBuffer"),
            ):
                verify_released(target, name)
        for root in (self.curve, self.waterfall):
            scrub(root, "clear")
        try:
            blocker = QtCore.QSignalBlocker(self.frame_selector)
        except Exception:
            blocker = None
            cleared = False
        if not keep_chrome:
            scrub(self.frame_selector, "clear")
            scrub(self.frame_selector, "setCurrentIndex", -1)
        del blocker
        bottom = scrub(self.vertical_splitter, "widget", 1)
        for widget in (
            self.image_splitter, self.raw, self.cake, bottom, self.norm, self.background,
            self.image_axis, self.share_axis, self.slice,
            self.slice_center, self.slice_width, self.pin,
        ):
            if not keep_chrome:
                scrub(widget, "hide")
        for name, value in (
            ("_viewer_intensity_contract", None),
            ("_viewer_intensity_domain", None), ("_viewer_auto_levels", None),
            ("_viewer_2d_payload", None), ("_frame_keys", ()), ("_current_key", None),
            ("_viewer_2d_shape", None),
            ("_rendered_detector_source", "none"),
            ("_selected_keys", ()),
            ("_trace_selection_keys", ()), ("_trace_history_keys", ()), ("_rendered_trace_keys", ()),
            ("_trace_aggregate_fold", None),
            ("_curve_mounted_keys", ()), ("_curve_items_by_key", {}),
            ("_curve_item_contracts", {}), ("_rendered_axis_labels", None),
            ("_trace_row_count", 0),
            ("_waterfall_y_values", ()), ("_waterfall_source_keys", ()), ("_label_indices", {}),
            ("_heavy_available", frozenset()), ("_trace_history_scope", None), ("_pinned_trace_scope", None),
            ("_rendered_plot_options", None), ("_rendered_overlay_step", None), ("_trace_history_by_identity", {}),
            ("_pinned_trace_by_id", {}), ("_rendered_plot_mode", ""), ("_bottom_waterfall_active", False),
            ("_rendered_browse_science_contract", None),
            ("_waterfall_render_contract", None), ("_rendered_image_axis", None), ("_rendered_cake_axis_key", None),
            ("_rendered_cake_x_axis", None), ("_rendered_cake_y_axis", None), ("_rendered_trace_axis_key", None),
        ):
            if keep_chrome and name in {
                "_frame_keys", "_current_key", "_selected_keys", "_label_indices",
                "_viewer_2d_shape",
            }:
                continue
            scrub(self, name, value)
        for widget, method, value in (
            (self.previous_frame, "setEnabled", False),
            (self.next_frame, "setEnabled", False),
            (self.title, "setText", (
                "2D Viewer · Render failed" if failure else "Current")),
            (self.status, "setText", (
                "2D Viewer · Render failed; retry available."
                if failure else "")),
            (self.progress, "setText", "0/0"),
        ):
            if not keep_chrome:
                scrub(widget, method, value)
        self._viewer_2d_known_empty = bool(cleared)
        self.viewer_intensity.sync(None, None)
        if not keep_chrome:
            self.viewer_intensity.hide()
        return cleared

    def clear_viewer_1d(self, request, *, failure=False, preserve_navigation=False):
        if not failure and type(request) is not Viewer1DRendererClearRequest:
            return None
        keep_chrome = bool(preserve_navigation and not failure
                           and self._processing_mode == "1D Viewer")
        if keep_chrome:
            self._begin_viewer_loading_snapshot("1D Viewer")
        else:
            self.drop_viewer_loading_snapshot()
        cleared = ScientificView._clear_viewer_payloads(
            self, keep_chrome=keep_chrome, failure=failure,
        )
        try:
            if not keep_chrome:
                self.title.setText("1D Viewer · Render failed" if failure else "Current")
                self.status.setText(
                    "1D Viewer · Render failed; retry available." if failure else "")
            cleared = bool(cleared and not self.curve.listDataItems()
                           and not self._trace_history_by_identity
                           and not self._pinned_trace_by_id
                           and not self._trace_history_keys
                           and not self._rendered_trace_keys)
        except Exception:
            cleared = False
        if not cleared:
            self.drop_viewer_loading_snapshot()
        return (cleared if failure else
                _new_viewer_1d_renderer_clear_receipt(request, cleared))

    def clear_workspace(self) -> bool:
        """Drop every mounted scientific payload before owner retirement."""
        self._clear_phase_overlays()
        self._clear_grid_overlay()
        self.drop_viewer_loading_snapshot()
        cleared = ScientificView.clear_viewer_1d(
            self, None, failure=True
        )
        try:
            if self.raw_popup_dialog is not None:
                self.raw_popup_dialog.close()
            self.title.setText("Current")
            self.status.setText("")
        except Exception:
            return False
        return bool(cleared)

    def _reconcile_slice_extent(self, state: ScientificProjection) -> None:
        orientation = (
            slice_region_orientation(
                state.plot_axis,
                self._rendered_cake_x_axis,
                self._rendered_cake_y_axis,
            )
            if state.slice_enabled
            else None
        )
        slice_axis = (
            self._rendered_cake_y_axis
            if orientation == "horizontal"
            else self._rendered_cake_x_axis
            if orientation == "vertical"
            else None
        )
        extent: tuple[float, float] | None = None
        if slice_axis is not None:
            finite = np.asarray(slice_axis.values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                lower = max(
                    float(np.min(finite)),
                    float(state.slice_center - state.slice_width),
                )
                upper = min(
                    float(np.max(finite)),
                    float(state.slice_center + state.slice_width),
                )
                if lower <= upper:
                    extent = (lower, upper)
        scope = (
            orientation,
            extent,
            None if slice_axis is None else id(slice_axis.values),
        )
        if scope == self._slice_extent_scope:
            return
        self._clear_slice_extent()
        self._slice_extent_scope = scope
        if orientation is None or extent is None:
            return
        angle = 0.0 if orientation == "horizontal" else 90.0
        pen = pg.mkPen((255, 255, 255, 225), width=1.5)
        lines = tuple(
            pg.InfiniteLine(
                pos=value,
                angle=angle,
                movable=False,
                pen=pen,
            )
            for value in extent
        )
        for line in lines:
            self.cake.canvas.imageViewBox.addItem(
                line,
                ignoreBounds=True,
            )
        self._slice_extent_lines = lines

    def _clear_slice_extent(self) -> None:
        for line in self._slice_extent_lines:
            try:
                self.cake.canvas.imageViewBox.removeItem(line)
            except (RuntimeError, ValueError):
                pass
        self._slice_extent_lines = ()

    def _autorange_slice_projection(self, share_axis: bool) -> None:
        """Immediately fit a changed slice and retain continuous tracking."""

        view = self.curve.getPlotItem().getViewBox()
        if share_axis:
            view.enableAutoRange(
                axis=pg.ViewBox.XAxis,
                enable=False,
            )
            view.enableAutoRange(
                axis=pg.ViewBox.YAxis,
                enable=True,
            )
            return
        self.curve.autoRange()
        self.curve.enableAutoRange()

    def _rebuild_frames(
        self,
        frames: tuple[DisplayFrameKey, ...],
        selected: DisplayFrameKey | None,
    ) -> None:
        prior = self._frame_keys
        prefix = len(frames) >= len(prior) and all(
            frame is frames[index] for index, frame in enumerate(prior)
        )
        start = len(prior)
        if not prefix:
            self.frame_selector.clear()
            self._label_indices.clear()
            self._selector_operations += 1
            start = 0
        for index, frame in enumerate(frames[start:], start):
            label = frame.local_frame_label
            indices = self._label_indices.setdefault(label, [])
            self.frame_selector.add_frame(
                frame_caption(frame, frozenset(), position=index),
                frame,
                f"{frame.source_scan}:{frame.local_frame_label}",
            )
            indices.append(index)
            self._selector_operations += 1
        if selected is None:
            self.frame_selector.setCurrentIndex(-1)
        else:
            for index in self._label_indices.get(
                selected.local_frame_label, ()
            ):
                if self.frame_selector.itemData(index) is selected:
                    self.frame_selector.setCurrentIndex(index)
                    break
        self._frame_keys = frames
        self._current_key = selected

    def rebind_navigation(
        self,
        authorization: TerminalRebindAuthorization,
        *,
        heavy_available: frozenset[DisplayFrameKey],
    ) -> bool:
        """Rebind an identity-distinct but scientifically identical context.

        Terminal Browse creates fresh frame identities for the artifact that
        the acquisition view has already painted.  Re-key the detached trace
        ownership and footer without touching any numeric array or plot item.
        The page admits this path only after canonical artifact, label,
        selection, and presentation-contract checks.
        """

        if type(authorization) is not TerminalRebindAuthorization:
            raise TypeError("navigation rebind requires an exact projection")
        source_navigation = authorization.source_navigation
        navigation = authorization.browse_navigation
        current = navigation.current
        frames = navigation.frames
        source_frames = source_navigation.frames
        frame_by_id = {id(frame): frame for frame in frames}
        rebound_by_id = {
            id(old): new for old, new in authorization.frame_pairs
        }
        identity_map = {
            id(old): id(new) for old, new in authorization.frame_pairs
        }
        if (
            current is None
            or type(heavy_available) is not frozenset
            or len(self._frame_keys) != len(source_frames)
            or any(
                rendered is not source
                for rendered, source in zip(
                    self._frame_keys, source_frames, strict=True,
                )
            )
            or len(navigation.selected) != len(self._trace_history_keys)
            or not any(frame is current for frame in heavy_available)
            or any(
                type(frame) is not DisplayFrameKey
                or frame_by_id.get(id(frame)) is not frame
                for frame in heavy_available
            )
            or len(source_navigation.selected) != len(self._trace_history_keys)
            or any(
                rendered is not source
                for rendered, source in zip(
                    self._trace_history_keys,
                    source_navigation.selected,
                    strict=True,
                )
            )
            or len(navigation.selected) != len(source_navigation.selected)
            or any(
                id(selected) != identity_map.get(id(source))
                for source, selected in zip(
                    source_navigation.selected,
                    navigation.selected,
                    strict=True,
                )
            )
        ):
            return False
        old_traces = tuple(
            self._trace_history_by_identity.get(id(frame))
            for frame in self._trace_history_keys
        )
        if any(type(trace) is not TraceProjection for trace in old_traces):
            return False
        rebound = tuple(
            replace(trace, frame=rebound_by_id[id(trace.frame)])
            for trace in old_traces
        )

        def rekey(row_key: tuple[object, ...]) -> tuple[object, ...]:
            if (
                type(row_key) is tuple
                and len(row_key) == 2
                and row_key[0] == "live"
                and row_key[1] in identity_map
            ):
                return ("live", identity_map[row_key[1]])
            return row_key

        blocker = QtCore.QSignalBlocker(self.frame_selector)
        self._trace_history_by_identity = {
            id(trace.frame): trace for trace in rebound
        }
        self._trace_history_keys = tuple(trace.frame for trace in rebound)
        self._trace_selection_keys = navigation.selected
        self._selected_keys = navigation.selected
        self._heavy_available = heavy_available
        self._rendered_trace_keys = tuple(
            rekey(key) for key in self._rendered_trace_keys
        )
        self._curve_mounted_keys = tuple(
            rekey(key) for key in self._curve_mounted_keys
        )
        self._curve_items_by_key = {
            rekey(key): item
            for key, item in self._curve_items_by_key.items()
        }
        self._curve_item_contracts = {
            rekey(key): value
            for key, value in self._curve_item_contracts.items()
        }
        self._waterfall_source_keys = tuple(
            rekey(key) for key in self._waterfall_source_keys
        )
        self._clear_trace_aggregate_fold()
        self._rebuild_frames(frames, current)
        current_index = next(
            (
                index
                for index, frame in enumerate(frames)
                if frame is current
            ),
            None,
        )
        self.previous_frame.setEnabled(
            current_index is not None and current_index > 0
        )
        self.next_frame.setEnabled(
            current_index is not None
            and current_index < len(frames) - 1
        )
        del blocker
        return True

    def reconcile_rebound_trace_axis(
        self,
        prior_traces: tuple[TraceProjection, ...],
        rebound: ScientificProjection,
        navigation: FrameNavigationProjection,
    ) -> bool:
        """Repair trace-axis controls without repainting accepted science."""

        if (
            type(prior_traces) is not tuple
            or not prior_traces
            or any(type(trace) is not TraceProjection for trace in prior_traces)
            or type(rebound) is not ScientificProjection
            or type(navigation) is not FrameNavigationProjection
            or not rebound.traces
        ):
            return False
        current_traces = self.trace_history_projections
        if (
            len(prior_traces) != len(current_traces)
            or len(current_traces) != len(navigation.selected)
        ):
            return False
        axis_keys = tuple(self._axis_key(trace.axis) for trace in current_traces)
        if (
            any(key is None for key in axis_keys)
            or len(set(axis_keys)) != 1
            or self._rendered_trace_axis_key != axis_keys[0]
            or any(
                new.frame is not frame
                or self._trace_history_keys[index] is not frame
                or old.axis is not new.axis
                or old.axis.values is not new.axis.values
                or old.intensity is not new.intensity
                for index, (old, new, frame) in enumerate(zip(
                    prior_traces,
                    current_traces,
                    navigation.selected,
                    strict=True,
                ))
            )
        ):
            return False
        retained_by_frame = {
            id(trace.frame): trace for trace in current_traces
        }
        if any(
            (retained := retained_by_frame.get(id(trace.frame))) is None
            or retained.frame is not trace.frame
            or retained.axis is not trace.axis
            or retained.intensity is not trace.intensity
            for trace in rebound.traces
        ):
            return False
        axis_key = axis_keys[0]
        plot_choices = self._plot_axis_choices(rebound, axis_key)
        plot_choice = next(
            (
                value
                for _label, value in plot_choices
                if canonical_axis_key(value) == axis_key
            ),
            None,
        )
        if plot_choice is None:
            return False
        active_plot = (
            self.waterfall.plot
            if (
                self._bottom_waterfall_active
                and self.bottom_stack.currentWidget() is self.waterfall
            )
            else self.curve
            if (
                not self._bottom_waterfall_active
                and self.bottom_stack.currentWidget() is self.curve
            )
            else None
        )
        if active_plot is None:
            return False
        blocker = QtCore.QSignalBlocker(self.plot_axis)
        self._replace_combo_choices(self.plot_axis, plot_choices)
        self._set_data_combo(self.plot_axis, plot_choice)
        axis = current_traces[0].axis
        label, unit = _axis_presentation(axis.label, axis.unit)
        active_plot.setLabel("bottom", label, units=unit)
        if active_plot is self.curve:
            self._rendered_axis_labels = None
        del blocker
        return self.plot_axis.currentData() == plot_choice

    def _clear_trace_aggregate_fold(self) -> None:
        self._trace_aggregate_fold = None

    @staticmethod
    def _aggregate_rows_are_an_exact_prefix(
        prior: tuple[tuple[tuple[object, ...], TraceProjection], ...],
        rows: tuple[tuple[tuple[object, ...], TraceProjection], ...],
    ) -> bool:
        return len(rows) >= len(prior) and all(
            old_key == new_key and old_trace is new_trace
            for (old_key, old_trace), (new_key, new_trace) in zip(
                prior,
                rows,
            )
        )

    @staticmethod
    def _aggregate_rows_are_cacheable(
        rows: tuple[tuple[tuple[object, ...], TraceProjection], ...],
        reference: np.ndarray,
    ) -> bool:
        return all(
            _is_readonly_array(trace.axis.values)
            and _is_readonly_array(trace.intensity)
            # a grid equal to rounding is the same grid, as aggregate_traces folds it
            and same_grid(trace.axis.values, reference)
            for _key, trace in rows
        )

    @staticmethod
    def _aggregate_projection(
        fold: _TraceAggregateCache,
        mode: str,
    ) -> TraceProjection:
        cached = fold.projections.get(mode)
        if cached is not None:
            return cached
        # Preserve aggregate_traces exactly: the prepared prefix has the same
        # C-contiguous shape, dtype, values, and row order as np.stack(rows),
        # and is reduced with the same NumPy operation.  A carried running sum
        # would change floating-point association for one-point traces.
        values = fold.values[:len(fold.rows)]
        if mode == "Sum":
            intensity = np.nansum(values, axis=0)
        elif mode == "Average":
            intensity = np.nanmean(values, axis=0)
        else:
            raise ValueError(f"unsupported aggregate mode: {mode}")
        intensity.setflags(write=False)
        first = fold.rows[0][1]
        projection = TraceProjection(
            first.frame,
            first.axis,
            intensity,
            mode,
        )
        fold.projections[mode] = projection
        return projection

    def _aggregate_trace_rows(
        self,
        rows: tuple[tuple[tuple[object, ...], TraceProjection], ...],
        *,
        intensity_scale: str,
        mode: str,
    ) -> tuple[TraceProjection, ...]:
        """Fold one exact immutable append prefix without restacking history."""

        if not rows:
            self._clear_trace_aggregate_fold()
            return ()
        reference = rows[0][1].axis.values
        prior = self._trace_aggregate_fold
        if (
            prior is not None
            and prior.intensity_scale == intensity_scale
            and self._aggregate_rows_are_an_exact_prefix(prior.rows, rows)
        ):
            suffix = rows[len(prior.rows):]
            if not suffix:
                return (self._aggregate_projection(prior, mode),)
            if self._aggregate_rows_are_cacheable(suffix, reference):
                scaled = tuple(
                    _scaled_intensity(trace.intensity, intensity_scale)
                    for _key, trace in suffix
                )
                result_dtype = np.result_type(
                    prior.values.dtype,
                    *(values.dtype for values in scaled),
                )
                if result_dtype == prior.values.dtype:
                    size = len(rows)
                    capacity = prior.values.shape[0]
                    values = prior.values
                    if size > capacity:
                        capacity = 1 << (size - 1).bit_length()
                        values = np.empty(
                            (capacity, reference.size),
                            dtype=prior.values.dtype,
                        )
                        values[:len(prior.rows)] = prior.values[
                            :len(prior.rows)
                        ]
                    values[len(prior.rows):size] = scaled
                    fold = _TraceAggregateCache(
                        rows,
                        intensity_scale,
                        values,
                        {},
                    )
                    projection = self._aggregate_projection(fold, mode)
                    self._trace_aggregate_fold = fold
                    return (projection,)

        scaled_traces = tuple(
            replace(
                trace,
                intensity=_scaled_intensity(
                    trace.intensity,
                    intensity_scale,
                ),
            )
            for _key, trace in rows
        )
        if not self._aggregate_rows_are_cacheable(rows, reference):
            self._clear_trace_aggregate_fold()
            return aggregate_traces(scaled_traces, mode)
        stacked = np.stack(tuple(
            trace.intensity for trace in scaled_traces
        ))
        fold = _TraceAggregateCache(
            rows,
            intensity_scale,
            stacked,
            {},
        )
        projection = self._aggregate_projection(fold, mode)
        self._trace_aggregate_fold = fold
        return (projection,)

    def _render_traces(
        self,
        state: ScientificProjection,
        navigation: FrameNavigationProjection,
        *,
        live_update: bool,
    ) -> None:
        browse_snapshot = state.browse_trace_snapshot
        if browse_snapshot is not None:
            if (
                type(browse_snapshot) is not BrowseTraceSnapshot
                or not browse_snapshot.science_contract
                or browse_snapshot.science_contract
                != state.browse_science_contract
                or browse_snapshot.plot_mode != state.plot_mode
                or state.slice_enabled
                or state.slice_pins
                or state.pinned_traces
                or tuple(trace.frame for trace in state.traces)
                != browse_snapshot.display_frames
                or any(
                    not any(owned is frame for owned in navigation.frames)
                    for frame in browse_snapshot.logical_frames
                )
            ):
                raise ValueError("Browse trace snapshot changed presentation scope")
            compatible_single = bool(
                browse_snapshot.plot_mode == "Single"
                and self._rendered_plot_mode == "Single"
                and self._rendered_browse_science_contract
                == browse_snapshot.science_contract
            )
            # Stacked receipts are whole replacements.  A compatible Single
            # receipt reseeds semantic history but retains its one mounted item
            # so the normal setData path can update it in place.
            self._trace_history_by_identity.clear()
            self._trace_aggregate_fold = None
            if not compatible_single:
                self._rendered_trace_keys = ()
                self._rendered_plot_mode = ""
            self._waterfall_source_keys = ()
            self._waterfall_render_contract = None
        else:
            self._rendered_browse_science_contract = None
        if state.replace_trace_history:
            self._trace_history_by_identity.clear()
            self._pinned_trace_by_id.clear()
        live_traces = self._merge_trace_history(state, navigation)
        pinned = self._merge_pinned_trace_history(state)
        rows = (
            *((("pin", *pin_id), trace) for pin_id, trace in pinned),
            *((("live", id(trace.frame)), trace) for trace in live_traces),
        )
        fold = self._trace_aggregate_fold
        if (
            fold is not None
            and not self._aggregate_rows_are_an_exact_prefix(fold.rows, rows)
        ):
            self._clear_trace_aggregate_fold()
        if state.plot_mode not in {"Average", "Sum"}:
            self._clear_trace_aggregate_fold()
        if browse_snapshot is None:
            self._trace_row_count = len(rows)
            presented_by_id = {
                id(trace.frame): trace.frame
                for _row_key, trace in rows
            }
            self._trace_history_keys = tuple(
                frame
                for frame in navigation.selected
                if presented_by_id.get(id(frame)) is frame
            )
        else:
            self._trace_row_count = len(browse_snapshot.logical_frames)
            self._trace_history_keys = browse_snapshot.logical_frames
            self._trace_selection_keys = browse_snapshot.logical_frames
        self._bottom_waterfall_active = (
            browse_snapshot.waterfall_active
            if browse_snapshot is not None
            else waterfall_should_be_active(
                state.plot_mode, len(rows),
                was_active=self._bottom_waterfall_active,
                viewer_1d=state.processing_mode == "1D Viewer")
        )
        waterfall_scope = rows
        stacked_selection = (
            browse_snapshot.stacked_options_applied
            if browse_snapshot is not None
            else state.plot_mode in {"Overlay", "Waterfall"}
            or (state.plot_mode == "Single" and len(rows) > 1)
        )
        if (browse_snapshot is None
                and (stacked_selection or self._bottom_waterfall_active)
                and state.processing_mode != "1D Viewer"):
            start_index = state.plot_options.waterfall_start - 1
            stop_index = state.plot_options.waterfall_stop or None
            rows = rows[
                start_index:stop_index:state.plot_options.waterfall_step
            ]
        if self._bottom_waterfall_active:
            source_keys = tuple(row_key for row_key, _trace in rows)
            render_contract = (
                self._trace_history_scope,
                state.plot_options,
                state.color_map,
                state.norm_channel,
            )
            if browse_snapshot is None and self._skip_live_waterfall(
                source_keys,
                render_contract,
                live_update=live_update,
            ):
                self.bottom_stack.setCurrentWidget(self.waterfall)
                self.legend.setVisible(False)
                if self._share_link_on:
                    self._schedule_curve_under_cake()
                return
            if browse_snapshot is None:
                rows = self._bounded_waterfall_rows(rows)
        self._phase_wavelengths = tuple(sorted({getattr(trace, "wavelength_m", None)
            for _, trace in rows}, key=lambda value: -1. if value is None else value))
        if state.plot_mode in {"Average", "Sum"}:
            traces = self._aggregate_trace_rows(
                rows,
                intensity_scale=state.plot_options.intensity_scale,
                mode=state.plot_mode,
            )
            keys = (("aggregate",),) if traces else ()
        else:
            rows = tuple(
                (
                    row_key,
                    replace(
                        trace,
                        intensity=_scaled_intensity(
                            trace.intensity,
                            state.plot_options.intensity_scale,
                        ),
                    ),
                )
                for row_key, trace in rows
            )
            keys = tuple(row_key for row_key, _trace in rows)
            traces = tuple(trace for _row_key, trace in rows)
        axis_keys = {
            self._axis_key(trace.axis)
            for trace in traces
        }
        self._phase_line_context = None
        if len(axis_keys) == 1 and traces:
            finite = np.concatenate([np.asarray(t.axis.values)[np.isfinite(t.axis.values)] for t in traces])
            if finite.size:
                self._phase_line_context = (next(iter(axis_keys)), (float(finite.min()), float(finite.max())))
        prior_trace_axis_key = self._rendered_trace_axis_key
        self._rendered_trace_axis_key = (
            next(iter(axis_keys))
            if len(axis_keys) == 1
            else None
        )
        norm = state.norm_channel.strip()
        intensity = (
            "I"
            if norm in {"", "None", "Norm Channel"}
            else f"I / {norm}"
        )
        intensity = _scaled_intensity_label(
            intensity,
            state.plot_options.intensity_scale,
        )
        position_by_key = (
            {}
            if not self._bottom_waterfall_active
            else {
                row_key: float(position)
                for (row_key, _trace), position in zip(
                    rows,
                    browse_snapshot.logical_positions,
                    strict=True,
                )
            }
            if browse_snapshot is not None
            else {
                row_key: float(index + 1)
                for index, (row_key, _trace) in enumerate(waterfall_scope)
            }
        )
        if self._bottom_waterfall_active and self._render_waterfall(
            traces,
            row_keys=keys,
            waterfall_scope=waterfall_scope,
            position_by_key=position_by_key,
            y_axis_choice=state.plot_options.waterfall_y_axis,
            color_map=state.color_map,
            logical_epochs=(
                None
                if browse_snapshot is None
                else browse_snapshot.logical_epochs
            ),
        ):
            self.bottom_stack.setCurrentWidget(self.waterfall)
            self.legend.setVisible(False)
            self._rendered_trace_keys = keys
            self._rendered_plot_mode = state.plot_mode
            self._rendered_plot_options = state.plot_options
            self._rendered_overlay_step = None
            self._waterfall_source_keys = source_keys
            self._waterfall_render_contract = render_contract
            self._rendered_browse_science_contract = (
                browse_snapshot.science_contract
                if browse_snapshot is not None
                else None
            )
            if self._share_link_on:
                self._schedule_curve_under_cake()
            return
        self._bottom_waterfall_active = False
        self._waterfall_source_keys = ()
        self._waterfall_render_contract = None
        overlay_step = (
            _overlay_step(
                traces,
                state.plot_options.overlay_offset,
            )
            if stacked_selection
            else 0.0
        )
        self._reconcile_curve_items(
            keys,
            traces,
            color_map=state.color_map,
            overlay_step=(overlay_step if stacked_selection else 0.0),
            clip_single_live=state.plot_mode == "Single",
            allow_single_rebind=(
                state.plot_mode == "Single"
                and state.plot_options == self._rendered_plot_options
                and self._rendered_trace_axis_key == prior_trace_axis_key
                and not state.share_axis
                and (
                    browse_snapshot is not None
                    or (
                        not state.pinned_traces
                        and not state.slice_pins
                    )
                )
                and self.bottom_stack.currentWidget() is self.curve
            ),
        )
        if traces:
            axis = traces[0].axis
            bottom_label, bottom_unit = _axis_presentation(
                axis.label, axis.unit
            )
        else:
            bottom_label, bottom_unit = "", ""
        self._update_curve_axis_labels(
            bottom_label,
            bottom_unit,
            f"{intensity} (a.u.)",
        )
        # Keep an accepted waterfall visible until the replacement curve is
        # fully populated; changing the stack first exposes an empty/stale
        # curve during the synchronous rebuild.
        self.bottom_stack.setCurrentWidget(self.curve)
        self.legend.setVisible(state.plot_options.show_legend)
        self._rendered_trace_keys = keys
        self._rendered_plot_mode = state.plot_mode
        self._rendered_plot_options = state.plot_options
        self._rendered_overlay_step = overlay_step
        self._rendered_browse_science_contract = (
            browse_snapshot.science_contract
            if browse_snapshot is not None
            else None
        )

    def _reconcile_curve_items(
        self,
        keys: tuple[tuple[object, ...], ...],
        traces: tuple[TraceProjection, ...],
        *,
        color_map: str = "Default",
        overlay_step: float,
        clip_single_live: bool,
        allow_single_rebind: bool,
    ) -> None:
        """Diff one curve presentation without recreating retained items."""

        if len(keys) != len(traces) or len(set(keys)) != len(keys):
            raise ValueError("curve rows do not have unique presentation keys")
        mounted = tuple(self.curve.listDataItems())
        expected_mounted = tuple(
            self._curve_items_by_key.get(key)
            for key in self._curve_mounted_keys
        )
        if (
            len(expected_mounted) != len(mounted)
            or any(
                expected is not actual
                for expected, actual in zip(
                    expected_mounted, mounted, strict=True
                )
            )
        ):
            self.curve.clear()
            self._curve_mounted_keys = ()
            self._curve_items_by_key.clear()
            self._curve_item_contracts.clear()
            self._rendered_axis_labels = None
            mounted = ()

        if (
            allow_single_rebind
            and len(keys) == 1
            and keys[0] not in self._curve_items_by_key
            and len(self._curve_mounted_keys) == 1
        ):
            prior_key = self._curve_mounted_keys[0]
            item = self._curve_items_by_key.pop(prior_key)
            prior_contract = self._curve_item_contracts.pop(prior_key, None)
            self._curve_items_by_key[keys[0]] = item
            if prior_contract is not None:
                self._curve_item_contracts[keys[0]] = prior_contract

        colors = trace_colors(color_map, len(traces))
        desired_items = []
        for index, (key, trace) in enumerate(
            zip(keys, traces, strict=True)
        ):
            title = trace.title or str(trace.frame.local_frame_label)
            offset = 0.0 if index == 0 else index * overlay_step
            data_contract = (
                id(trace.axis.values),
                id(trace.intensity),
                offset,
                title,
            )
            style_contract = colors[index]
            item = self._curve_items_by_key.get(key)
            prior_contract = self._curve_item_contracts.get(key)
            clip_eligible = (
                prior_contract[2]
                if prior_contract is not None
                and prior_contract[0][0] == data_contract[0]
                else _axis_supports_view_clipping(trace.axis.values)
            )
            contract = (data_contract, style_contract, clip_eligible)
            data_changed = (
                item is None
                or prior_contract is None
                or prior_contract[0] != data_contract
            )
            if data_changed:
                self._reflow_follows = 0
                plot_x, plot_y = trace.axis.values, trace.intensity
                if self._axis_key(trace.axis) == "chigi_deg":
                    # NaN intensity at finite chi still expands pyqtgraph's X
                    # bounds. Trim only empty ends of the plotted view; keep
                    # interior gaps and the scientific projection unchanged.
                    valid = np.flatnonzero(np.isfinite(plot_x) & np.isfinite(plot_y))
                    measured = slice(valid[0], valid[-1] + 1) if valid.size else slice(0, 0)
                    plot_x, plot_y = plot_x[measured], plot_y[measured]
            if item is None:
                item = self.curve.plot(
                    plot_x,
                    _offset_intensity(
                        plot_y,
                        index,
                        overlay_step,
                    ),
                    name=title,
                    **data_curve_style(style_contract),
                )
                self._curve_items_by_key[key] = item
            else:
                if data_changed:
                    item.setData(
                        plot_x,
                        _offset_intensity(
                            plot_y,
                            index,
                            overlay_step,
                        ),
                        name=title,
                        connect="finite",
                    )
                if (
                    prior_contract is None
                    or prior_contract[1] != style_contract
                ):
                    style = data_curve_style(style_contract)
                    item.setPen(style['pen'])
                    item.setSymbolBrush(style['symbolBrush'])
                    item.setSymbolPen(style['symbolPen'])
                if (
                    prior_contract is None
                    or prior_contract[0][-1] != title
                ):
                    label = self.legend.getLabel(item)
                    if label is not None:
                        label.setText(title)
            item.setVisible(True)
            self._curve_item_contracts[key] = contract
            desired_items.append(item)

        desired = tuple(desired_items)
        current = tuple(self.curve.listDataItems())
        if current != desired:
            for item in current:
                if bool(item.opts.get("clipToView", False)):
                    item.setClipToView(False)
                self.curve.removeItem(item)
            for item in desired:
                self.curve.addItem(item)
        for key, item in zip(keys, desired, strict=True):
            desired_clip = bool(
                clip_single_live
                and key[0] == "live"
                and self._curve_item_contracts[key][2]
            )
            if bool(item.opts.get("clipToView", False)) != desired_clip:
                item.setClipToView(desired_clip)
        self._curve_mounted_keys = keys

        desired_keys = set(keys)
        for key in tuple(self._curve_items_by_key):
            if (
                len(self._curve_items_by_key)
                <= _MAX_RETAINED_CURVE_ITEMS
                or key in desired_keys
            ):
                continue
            self._curve_items_by_key.pop(key, None)
            self._curve_item_contracts.pop(key, None)

    def _update_curve_axis_labels(
        self,
        bottom_label: str,
        bottom_unit: str | None,
        left_label: str,
    ) -> None:
        labels = (bottom_label, bottom_unit, left_label)
        prior = self._rendered_axis_labels
        if prior is None or prior[:2] != labels[:2]:
            self.curve.setLabel(
                "bottom",
                bottom_label,
                units=bottom_unit,
            )
        if prior is None or prior[2] != left_label:
            self.curve.setLabel("left", left_label)
        self._rendered_axis_labels = labels

    def _merge_trace_history(
        self,
        state: ScientificProjection,
        navigation: FrameNavigationProjection,
    ) -> tuple[TraceProjection, ...]:
        """Merge exact selected identities into detached 1-D history."""

        scope = (
            # Plot mode changes only the presentation of these numeric trace
            # projections.  Keep the detached history warm across
            # Single/Overlay/Average/Sum switches.
            state.processing_mode,
            state.plot_axis,
            state.slice_enabled,
            state.slice_center,
            state.slice_width,
            tuple(pin.projection_id for pin in state.slice_pins),
            # Aggregate revision remains visible provenance but does not
            # change an earlier trace's per-frame metadata divisor.  Only the
            # semantic normalization regime reseeds detached history.
            *trace_normalization_scope(
                state.norm_identity,
                state.norm_channel,
            ),
        )
        selected = navigation.selected
        viewer_single = (
            state.processing_mode == "1D Viewer"
            and state.plot_mode == "Single"
        )
        same_scope = scope == self._trace_history_scope
        if not same_scope:
            self._trace_history_by_identity.clear()
            self._trace_aggregate_fold = None
            self._rendered_trace_keys = ()
            self._rendered_plot_mode = ""
            self._rendered_plot_options = None
            self._rendered_overlay_step = None
            self._waterfall_source_keys = ()
            self._waterfall_render_contract = None
        elif viewer_single:
            # Viewer Single is the exact selected science. Its semantic
            # history is reseeded, while compatible rendered items stay
            # mounted for the setData reuse path below.
            self._trace_history_by_identity.clear()
        selected_by_id = {id(frame): frame for frame in selected}
        for trace in state.traces:
            frame = trace.frame
            if selected_by_id.get(id(frame)) is frame:
                retained = self._trace_history_by_identity.get(id(frame))
                if (
                    same_scope
                    and retained is not None
                    and retained.frame is frame
                    and (
                        retained is trace
                        or (
                            retained.axis.label == trace.axis.label
                            and retained.axis.unit == trace.axis.unit
                            and retained.axis.values.dtype
                            == trace.axis.values.dtype
                            and np.array_equal(
                                retained.axis.values,
                                trace.axis.values,
                                equal_nan=True,
                            )
                            and retained.intensity.dtype
                            == trace.intensity.dtype
                            and np.array_equal(
                                retained.intensity,
                                trace.intensity,
                                equal_nan=True,
                            )
                            and retained.title == trace.title
                            and retained.epoch == trace.epoch
                            and getattr(retained, "wavelength_m", None) == getattr(trace, "wavelength_m", None)
                        )
                    )
                ):
                    # Projection may derive a fresh normalized ndarray for the
                    # current frame on each repaint.  The exact frame and
                    # numeric scope identify immutable per-frame science.  An
                    # equal derived wrapper therefore keeps the already
                    # detached identity; a real replacement still invalidates
                    # the aggregate prefix below.
                    continue
                self._trace_history_by_identity[id(frame)] = trace
        self._trace_history_by_identity = {
            identity: trace
            for identity, trace in self._trace_history_by_identity.items()
            if selected_by_id.get(identity) is trace.frame
        }
        self._trace_history_scope = scope
        self._trace_selection_keys = selected
        traces = tuple(
            self._trace_history_by_identity[id(frame)]
            for frame in selected
            if id(frame) in self._trace_history_by_identity
        )
        self._trace_history_keys = tuple(trace.frame for trace in traces)
        return traces

    def _merge_pinned_trace_history(
        self,
        state: ScientificProjection,
    ) -> tuple[tuple[tuple[object, ...], TraceProjection], ...]:
        pin_ids = tuple(pin.projection_id for pin in state.slice_pins)
        scope = (
            state.processing_mode,
            state.plot_axis,
            pin_ids,
            state.norm_identity,
            state.norm_channel,
        )
        if (
            scope != self._pinned_trace_scope
            or (
                not state.retain_display
                and not state.pinned_traces
                and not state.slice_pins
            )
        ):
            self._pinned_trace_by_id.clear()
        desired = set(pin_ids)
        for pinned in state.pinned_traces:
            pin_id = pinned.pin.projection_id
            if pin_id in desired:
                self._pinned_trace_by_id[pin_id] = pinned.trace
        self._pinned_trace_by_id = {
            pin_id: trace
            for pin_id, trace in self._pinned_trace_by_id.items()
            if pin_id in desired
        }
        self._pinned_trace_scope = scope
        return tuple(
            (pin_id, self._pinned_trace_by_id[pin_id])
            for pin_id in pin_ids
            if pin_id in self._pinned_trace_by_id
        )

    def _skip_live_waterfall(
        self,
        source_keys: tuple[tuple[object, ...], ...],
        render_contract: tuple[object, ...],
        *,
        live_update: bool,
    ) -> bool:
        """Throttle a compatible prefix before scaling or stacking rows."""

        if (
            not live_update
            or self.bottom_stack.currentWidget() is not self.waterfall
            or self.waterfall.image.image is None
            or render_contract != self._waterfall_render_contract
            or len(source_keys) < len(self._waterfall_source_keys)
            or not all(
                key == source_keys[index]
                for index, key in enumerate(self._waterfall_source_keys)
            )
        ):
            return False
        return time.monotonic() - self._waterfall_last_draw < 0.5

    @staticmethod
    def _bounded_waterfall_rows(
        rows: tuple[tuple[tuple[object, ...], TraceProjection], ...],
    ) -> tuple[tuple[tuple[object, ...], TraceProjection], ...]:
        """Choose one display-only row set before any numeric transforms."""

        if len(rows) <= MAX_WATERFALL_DISPLAY_ROWS:
            return rows
        empty_rows = np.empty((len(rows), 0), dtype=float)
        _display, _keys, indices = waterfall_display_rows(
            empty_rows,
            tuple(row_key for row_key, _trace in rows),
            MAX_WATERFALL_DISPLAY_ROWS,
        )
        if indices is None:
            return rows
        return tuple(rows[int(index)] for index in indices)

    def _render_waterfall(
        self,
        traces,
        *,
        row_keys,
        waterfall_scope,
        position_by_key: dict[tuple[object, ...], float],
        y_axis_choice: str,
        color_map: str,
        logical_epochs: tuple[float, ...] | None = None,
    ) -> bool:
        if not traces:
            return False
        axis = traces[0].axis
        rows = _waterfall_rows_on_reference_axis(traces)
        if rows is None:
            return False
        now = time.monotonic()
        rows, x_values = resample_image_axis_to_uniform(
            rows,
            axis.values,
            axis=1,
        )
        y_values, y_label = self._waterfall_axis(
            waterfall_scope,
            traces,
            row_keys,
            position_by_key,
            y_axis_choice,
            logical_epochs,
        )
        if y_values.shape != (rows.shape[0],):
            return False
        self.waterfall.render(
            rows,
            x_axis=AxisProjection(
                np.asarray(x_values, dtype=float),
                axis.label,
                axis.unit,
            ),
            y_axis=AxisProjection(y_values, y_label),
            color_map=color_map,
            level_scan_token=(
                tuple(
                    (row_key, id(trace.intensity))
                    for row_key, trace in zip(row_keys, traces, strict=True)
                ),
                y_axis_choice,
            ),
        )
        self._waterfall_y_values = tuple(float(value) for value in y_values)
        self._waterfall_y_label = y_label
        self._waterfall_last_draw = now
        return True

    @staticmethod
    def _waterfall_axis(
        waterfall_scope,
        traces,
        row_keys,
        position_by_key: dict[tuple[object, ...], float],
        y_axis_choice: str,
        logical_epochs: tuple[float, ...] | None = None,
    ) -> tuple[np.ndarray, str]:
        positions = np.asarray(
            [position_by_key[row_key] for row_key in row_keys],
            dtype=float,
        )
        if y_axis_choice == "Frame #":
            return positions, y_axis_choice
        if logical_epochs is not None:
            if not logical_epochs:
                return positions, "Frame #"
            baseline = min(logical_epochs)
            values = np.asarray(
                [
                    logical_epochs[int(position_by_key[row_key]) - 1]
                    - baseline
                    for row_key in row_keys
                ],
                dtype=float,
            )
            if y_axis_choice == "Time (minutes)":
                values /= 60.0
            return values, y_axis_choice
        epochs = {
            row_key: trace.epoch
            for row_key, trace in waterfall_scope
        }
        if not epochs or any(value is None for value in epochs.values()):
            return positions, "Frame #"
        baseline = min(float(value) for value in epochs.values())
        values = np.asarray(
            [float(epochs[row_key]) - baseline for row_key in row_keys],
            dtype=float,
        )
        if y_axis_choice == "Time (minutes)":
            values /= 60.0
        return values, y_axis_choice

    def _mouse_moved(self, positions: object) -> None:
        self._cursor_readout.mouse_moved(positions)

    @staticmethod
    def _set_data_combo(combo: QtWidgets.QComboBox, value: str) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def _reconcile_axis_choices(self, state: ScientificProjection) -> None:
        native_key = (
            None
            if not state.traces
            else self._axis_key(state.traces[0].axis)
        )
        plot_choices = self._plot_axis_choices(state, native_key)
        self.plot_axis.setToolTip(
            "To plot χGI, first select Q-χ or Q-χ (derived) in the 2D display selector."
            if state.measurement_mode == "GI" and state.gi_mode_2d == "qip_qoop"
            else "Choose the horizontal axis for the 1D plot."
        )
        if state.measurement_mode == "GI":
            # The maps this frame actually has, primary first; one map keeps the
            # historical single (disabled) entry.
            image_choices = tuple(
                (_GI_IMAGE_AXIS_LABELS[mode], mode)
                for mode in state.gi_maps if mode in _GI_IMAGE_AXIS_LABELS
            ) or _GI_IMAGE_AXIS_CHOICES.get(
                state.gi_mode_2d,
                _GI_IMAGE_AXIS_CHOICES["q_chi"],
            )
        else:
            image_choices = _STANDARD_IMAGE_AXIS_CHOICES
        self._replace_combo_choices(self.plot_axis, plot_choices)
        self._replace_combo_choices(self.image_axis, image_choices)

    def _plot_axis_choices(
        self,
        state: ScientificProjection,
        native_key: str | None,
    ) -> tuple[tuple[str, str], ...]:
        if state.measurement_mode == "GI":
            native_choices = self._merged_axis_choices(*(
                _GI_PLOT_AXIS_CHOICES.get(mode, ()) for mode in state.gi_modes_1d
            )) if state.gi_modes_1d else _GI_PLOT_AXIS_CHOICES.get(
                state.gi_mode_1d,
                _GI_PLOT_AXIS_CHOICES["q_total"],
            )
            return (
                native_choices
                if state.processing_mode == "Int 1D"
                else self._merged_axis_choices(
                    native_choices,
                    _GI_DERIVED_CAKE_PLOT_AXIS_CHOICES
                    if state.derived_2d
                    else _GI_CHIGI_CAKE_PLOT_AXIS_CHOICES
                    if state.gi_mode_2d == "q_chi"
                    and state.heavy is not None
                    and state.heavy.cake_y is not None
                    and self._axis_key(state.heavy.cake_y) == "chigi_deg"
                    else _GI_CAKE_PLOT_AXIS_CHOICES.get(state.gi_mode_2d, ()),
                )
            )
        if state.processing_mode == "Int 1D" and native_key == "chi_deg":
            return (("χ (°)", "chi"),)
        return (
            _STANDARD_PLOT_AXIS_CHOICES[:2]
            if state.processing_mode == "Int 1D"
            else _STANDARD_PLOT_AXIS_CHOICES
        )

    @staticmethod
    def _merged_axis_choices(
        *groups: tuple[tuple[str, str], ...],
    ) -> tuple[tuple[str, str], ...]:
        merged: list[tuple[str, str]] = []
        seen: set[str] = set()
        for group in groups:
            for choice in group:
                if choice[1] not in seen:
                    seen.add(choice[1])
                    merged.append(choice)
        return tuple(merged)

    @staticmethod
    def _replace_combo_choices(
        combo: QtWidgets.QComboBox,
        choices: tuple[tuple[str, str], ...],
    ) -> None:
        current = tuple(
            (combo.itemText(index), combo.itemData(index))
            for index in range(combo.count())
        )
        if current == choices:
            return
        combo.clear()
        for label, value in choices:
            combo.addItem(label, value)

    def _frame_selected(self, index: int) -> None:
        if index < 0:
            return
        frame = self.frame_selector.itemData(index)
        if type(frame) is not DisplayFrameKey:
            return
        if os.environ.get("XDART_VIEWER_DEBUG") == "1":
            print("viewer_footer", {"mode": self._processing_mode,
                "plot_mode": self._plot_mode, "current": frame.local_frame_label,
                "selected": [key.local_frame_label for key in self._selected_keys]}, flush=True)
        if (self._processing_mode == "1D Viewer"
                and frame.source_scan == frame.artifact == "viewer-1d"
                and any(frame is item for item in self._frame_keys)):
            self.commandRequested.emit(ShellCommand(
                ShellCommandKind.SELECT_FRAME, frame=frame,
                frames=(frame,), intent=(
                    FrameSelectionIntent.VISIT
                    if self._plot_mode in {"Overlay", "Waterfall"}
                    else FrameSelectionIntent.EXACT)))
            return
        if (self._processing_mode == "2D Viewer"
                and frame.source_scan == frame.artifact == "viewer-2d"
                and any(frame is item for item in self._frame_keys)):
            self.commandRequested.emit(ShellCommand(
                ShellCommandKind.SELECT_FRAME, frame=frame, frames=(frame,)))
            return
        kind = (
            ShellCommandKind.SELECT_FRAME
            if any(frame is item for item in self._heavy_available)
            else ShellCommandKind.HYDRATE_FRAME
        )
        if self._single_mode:
            membership = (frame,)
        elif self._plot_mode in {"Overlay", "Waterfall"}:
            membership = self._selected_keys or (frame,)
        else:
            membership = (
                self._selected_keys
                if any(frame is item for item in self._selected_keys)
                else (*self._selected_keys, frame)
            )
        self.commandRequested.emit(
            ShellCommand(kind, frame=frame, frames=membership)
        )

    @staticmethod
    def _axis_key(axis) -> str | None:
        if axis is None:
            return None
        key = canonical_axis_key(f"{axis.label} ({axis.unit})")
        return key or None

    def _apply_share_axis_state(self, requested: bool) -> None:
        exact_match = (
            self._processing_mode == "Int 2D"
            and self._rendered_cake_axis_key is not None
            and self._rendered_trace_axis_key
            == self._rendered_cake_axis_key
        )
        derivable = (
            self._processing_mode == "Int 2D"
            and self._rendered_cake_axis_key
            in _CONVERTIBLE_RADIAL_AXIS_KEYS
            and self._rendered_trace_axis_key
            in _CONVERTIBLE_RADIAL_AXIS_KEYS
        )
        self.share_axis.setEnabled(exact_match or derivable)
        linked = bool(requested and exact_match)
        if requested and not exact_match:
            self.share_axis.setChecked(False)
        self.plot_axis.setEnabled(not linked)
        was_linked = self._share_link_on
        self._set_share_link(linked)
        if linked and was_linked:
            # A live run can reconcile faster than the event-loop geometry
            # coalescer.  Geometry is already settled here (image + traces
            # rendered), so converge synchronously; the range guard keeps an
            # already-aligned repaint a no-op.  A real trigger, so it opens
            # a fresh re-flow follow epoch like the scheduled ones do.
            self._reflow_follows = 0
            self._align_curve_under_cake()

    # -- the keyed layout: display_logic's repeated PanelKey roles (the H11/WS-X2 seam) -----------------------
    def reconcile_keyed(self, state: KeyedPanelsProjection) -> None:
        """Draw a keyed layout: ``state.layout``'s rows of ``PanelKey``s, a SLICE_2D key on Int's image pane
        over the frame navigation's own widgets (stepping its slice along its normal: ``SET_PANEL_INDEX``),
        a PROJ_1D key on a 1-D plot styled as Int's. The top bar's colour map, Log and Set BG drive every
        panel; below the panels, Int's intensity controls set one set of levels for every slice
        (``_refresh_keyed_intensity``) and Share Axis (``state.share_axis``) links the panels drawing one
        coordinate (``_link_keyed_axes``). Int's own layout is hidden while it is shown, and
        ``_apply_mode_layout`` brings it back for Int's modes. All or nothing: on a failure the prior panels are
        drawn again (or cleared) and the error raised."""
        if type(state) is not KeyedPanelsProjection:
            raise TypeError("a keyed layout is drawn from a KeyedPanelsProjection")
        self._clear_phase_overlays()
        self.reference_phases.set_available(False, "Reference phases are unavailable in keyed views.", visible=False)
        prior = self._keyed_shown
        try:
            self._paint_keyed(state)
        except BaseException:
            try:
                if prior is None:
                    self.clear_keyed()
                else:
                    self._paint_keyed(prior)
            except BaseException:  # the primary failure is the one reported
                pass
            raise
        self._keyed_shown = state

    def set_coordinate_ranges(self, axes, labels, bounds=None):
        """Mount Int's coordinate editors/handles; synchronization emits no edits."""
        from xdart.gui.widgets.controls_panel import CoordinateRangeControl
        if not axes:
            if self._coordinate_range_bar is not None:
                self._coordinate_range_bar.hide()
            return
        if self._coordinate_range_bar is None:
            self._coordinate_range_bar = QtWidgets.QWidget(self)
            layout = QtWidgets.QHBoxLayout(self._coordinate_range_bar)
            layout.setContentsMargins(4, 0, 4, 0)
            for label in labels:
                control = CoordinateRangeControl(label)
                control.rangeChanged.connect(lambda _low, _high: self.coordinateRangesChanged.emit(
                    tuple(row._bounds for row in self.coordinate_ranges)))
                layout.addWidget(control, 1)
                self.coordinate_ranges.append(control)
            self._main_layout.insertWidget(self._main_layout.indexOf(self.viewer_intensity_row),
                                           self._coordinate_range_bar)
        if len(axes) != len(self.coordinate_ranges):
            raise ValueError("coordinate ranges must match their mounted axes")
        for i, (row, axis, label) in enumerate(zip(self.coordinate_ranges, axes, labels, strict=True)):
            row.reconcile(label, axis, None if bounds is None else bounds[i])
        self._coordinate_range_bar.show()

    @property
    def coordinate_range_bounds(self):
        """Detached operator bounds; the scientific projection remains the page's authority."""
        return tuple(row._bounds for row in self.coordinate_ranges)

    @property
    def keyed_projection(self) -> KeyedPanelsProjection | None:
        """The keyed layout drawn now, or None."""
        return self._keyed_shown

    def keyed_panel(self, key: PanelKey):
        """A keyed panel's widgets (an image panel's ``pane``, ``previous``, ``selector`` and ``next``; a 1-D
        panel's ``plot`` and ``curve``), for its host's checks."""
        return self._keyed_panels[key]

    def set_keyed_region(self, key: PanelKey, region: RectRegion | None) -> None:
        """Show ``region`` as a rectangle the user can move and resize on the keyed image panel ``key`` (None: no
        rectangle there), drawn in the panel's coordinates: a region in other coordinates than the panel draws is
        refused (``TypeError``). It stays over the panel's redraws in the same coordinates; a redraw in other ones
        or a new layout drops it. Setting it emits nothing; the user's moves emit ``keyedRegionChanged``."""
        widgets = self._keyed_panels.get(key)
        if type(widgets) is not _KeyedImagePanel:
            raise KeyError(f"no keyed image panel {key!r}")
        widgets.set_region(region, lambda: self.keyedRegionChanged.emit(key, widgets.region_shown()))

    def keyed_region(self, key: PanelKey) -> RectRegion | None:
        """The keyed image panel's rectangle where it is now, or None."""
        widgets = self._keyed_panels.get(key)
        return widgets.region_shown() if type(widgets) is _KeyedImagePanel else None

    def set_keyed_named_region(self, key: PanelKey, name: str, region: RectRegion | None, *,
                               color=_REGION_PEN, dashed: bool = False, visible: bool = True) -> None:
        """Reconcile one independent mark by identity. Other marks stay; live gestures emit typed geometry.
        None removes it; hidden backgrounds keep their geometry. Axes changes drop every mark."""
        widgets = self._keyed_panels.get(key)
        if type(widgets) is not _KeyedImagePanel:
            raise KeyError(f"no keyed image panel {key!r}")
        widgets.set_named_region(str(name), region,
            lambda: self.keyedNamedRegionChanged.emit(key, str(name), widgets.named_region_shown(str(name))),
            color=color, dashed=dashed, visible=visible)

    def keyed_named_region(self, key: PanelKey, name: str) -> RectRegion | None:
        widgets = self._keyed_panels.get(key)
        return widgets.named_region_shown(str(name)) if type(widgets) is _KeyedImagePanel else None

    def keyed_region_item(self, key: PanelKey, name: str = ""):
        """Borrowed interaction handle for compatibility and gesture checks; lifetime stays with the view."""
        widgets = self._keyed_panels[key]
        return widgets.regions.get(str(name)) if name else widgets.region

    def clear_keyed(self, detail: str = "") -> None:
        """Empty every keyed panel, keeping the layout."""
        for widgets in self._keyed_panels.values():
            widgets.clear()
        if self._layout_mode == "keyed":
            self.title.setText("")
            self.status.setText(str(detail))
            self._refresh_keyed_intensity()
        self._keyed_shown = None

    def _paint_keyed(self, state: KeyedPanelsProjection) -> None:
        self._enter_keyed_layout()
        self._unlink_keyed_axes()  # linked again below, on this draw's axes
        if state.layout != self._keyed_layout:
            self._build_keyed(state.layout)
        color = trace_colors(state.color_map, 1)[0]
        for panel in state.panels:
            widgets = self._keyed_panels[panel.key]
            if type(panel) is KeyedImageProjection:
                widgets.draw(panel, state.color_map, state.log_scale)
            else:
                widgets.draw(panel, color)
        blockers = [QtCore.QSignalBlocker(widget) for widget in (self.color_map, self.log_scale, self.share_axis)]
        set_combo_value(self.color_map, state.color_map, fallback="Default")
        self.log_scale.setChecked(state.log_scale)
        self.share_axis.setChecked(state.share_axis)
        del blockers
        self.background.setEnabled(state.background_enabled)
        self.background.setText(("Clear" if state.background_set else "Set") + " BG")
        self.background.setToolTip(state.background_note)
        # the background drawn now, which ``release_display_background`` lets go of (as ``reconcile`` does)
        self._rendered_background_key = self._expected_background_key if state.background_set else None
        self.title.setText(state.title)
        self.status.setText(state.status)
        self._keyed_coordinates = self._keyed_axes(state)
        self.share_axis.setEnabled(bool(self._keyed_coordinates))
        self._set_share_link(state.share_axis)
        self._refresh_keyed_intensity()

        if all(type(panel) is KeyedFigureProjection for panel in state.panels):
            if not self._figure_hidden:
                self._figure_hidden = tuple(w for w in (self.norm, self.background, self.color_map,
                    self.log_scale, self.title, self.status, self.viewer_intensity_row) if not w.isHidden())
                for widget in self._figure_hidden:
                    widget.hide()
        else:
            self._restore_figure_chrome()

    def _keyed_axes(self, state: KeyedPanelsProjection) -> dict[tuple[str, str], tuple[tuple[pg.ViewBox, int], ...]]:
        """The keyed panels' axes (view box, 0 for x or 1 for y) by the coordinate each draws, its label and unit,
        in the layout's order: those drawn by two or more, which Share Axis links. Not ``_axis_key``: Int's
        canonical key takes the Q frame's Qx, Qy and Qz alike as Q."""
        coordinates: dict[tuple[str, str], list[tuple[pg.ViewBox, int]]] = {}
        for row in state.layout:
            for key in row:
                panel, widgets = state.panel(key), self._keyed_panels[key]
                if type(panel) is KeyedImageProjection:
                    box = widgets.pane.canvas.imageViewBox
                    drawn = ((box, 0, panel.x_axis), (box, 1, panel.y_axis))
                elif type(panel) is KeyedFigureProjection:
                    continue  # figure axes are declared locally, not linked as scientific slices
                else:
                    drawn = ((widgets.plot.getViewBox(), 0, panel.axis),)
                for box, axis, projection in drawn:
                    coordinates.setdefault((projection.label, projection.unit), []).append((box, axis))
        return {coordinate: tuple(axes) for coordinate, axes in coordinates.items() if len(axes) > 1}

    def _link_keyed_axes(self, on: bool) -> None:
        """Share Axis over a keyed layout (``_set_share_link``'s keyed side; Int's own links stay off): the axes
        drawing one coordinate (RSM's H on the HK and HL slices and the H projection) take the first one's range,
        then follow each other's pan and zoom along it. Off, they are independent, and the 1-D panels follow their
        data again, as Int's 1-D plot does when Share Axis goes off."""
        self._unlink_keyed_axes()
        if on:
            for axes in self._keyed_coordinates.values():
                first, axis = axes[0]
                self._sync_keyed_axis(axes, tuple(first.viewRange()[axis]), axes[0])
                for member in axes:
                    signal = member[0].sigXRangeChanged if member[1] == 0 else member[0].sigYRangeChanged
                    slot = functools.partial(self._on_keyed_range_changed, axes, member)
                    signal.connect(slot)
                    self._keyed_links.append((signal, slot))
        elif self._keyed_share_on:
            for widgets in self._keyed_panels.values():
                if type(widgets) is _KeyedTracePanel:
                    widgets.plot.getViewBox().enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        self._keyed_share_on = bool(on and self._keyed_coordinates)

    def _unlink_keyed_axes(self) -> None:
        for signal, slot in self._keyed_links:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._keyed_links = []

    def _on_keyed_range_changed(self, axes, source, _box, view_range) -> None:
        if not self._keyed_share_on or self._share_axis_syncing:
            return
        self._sync_keyed_axis(axes, tuple(view_range), source)

    def _sync_keyed_axis(self, axes, view_range, source) -> None:
        """Every axis of ``axes`` but ``source`` to ``view_range``, under Int's share guard."""
        prior = self._share_axis_syncing
        self._share_axis_syncing = True
        try:
            for box, axis in axes:
                if box is not source[0] or axis != source[1]:
                    (box.setXRange if axis == 0 else box.setYRange)(*view_range, padding=0)
        finally:
            self._share_axis_syncing = prior

    def _refresh_keyed_intensity(self, *, source=None) -> None:
        """Int's intensity controls over a keyed layout's slices: one set of levels for them all, in what each
        pane draws (log10 with Log, as Int's viewer's). Autoscale spans the slices' own automatic levels (each
        Int's rule over its finite values), and the controls' range their values; typed or dragged levels are
        kept across draws. A shared slice's colour bar edits this same manual range. A panel
        presented on its own ("own", "signed": ``KeyedImageProjection.presentation``) keeps its own levels."""
        controls = self.viewer_intensity
        controls.set_percentile_defaults((0.1, 99.9) if self.log_scale.isChecked() else (0.5, 99.5))
        drawn = [widgets.levels for widgets in self._keyed_panels.values()
                 if type(widgets) is _KeyedImagePanel and widgets.levels is not None
                 and widgets.presentation == "shared"]
        if not drawn:
            self._viewer_intensity_domain = self._viewer_auto_levels = None
            controls.sync(None, None)
            return
        self._viewer_intensity_domain = (min(found[0] for found, _levels in drawn),
                                         max(found[1] for found, _levels in drawn))
        percentile = controls.percentile_values()
        mapped = [widgets.pane.canvas.percentile_levels(percentile)
                  for widgets in self._keyed_panels.values()
                  if type(widgets) is _KeyedImagePanel and widgets.levels is not None
                  and widgets.presentation == "shared"] if percentile is not None else [levels for _found, levels in drawn]
        self._viewer_auto_levels = (min(levels[0] for levels in mapped), max(levels[1] for levels in mapped))
        controls.sync(self._viewer_intensity_domain, self._viewer_auto_levels)
        chosen = self._viewer_auto_levels if percentile is not None or controls.autoscale.isChecked() else controls.values()
        self._set_keyed_levels(chosen, source=source)
        controls.sync(self._viewer_intensity_domain, chosen)

    def _set_keyed_levels(self, levels, *, source=None) -> None:
        """Every drawn slice's colour bar to ``levels``, its limits widened to take them."""
        if levels is None or not np.isfinite(levels).all() or levels[1] <= levels[0]:
            return
        low, high = levels
        for widgets in self._keyed_panels.values():
            if (type(widgets) is _KeyedImagePanel and widgets.levels is not None
                    and widgets.presentation == "shared"):
                histogram = widgets.pane.canvas.histogram
                (own_low, own_high), _automatic = widgets.levels
                histogram.lo_lim, histogram.hi_lim = min(own_low, low), max(own_high, high)
                if histogram is not source:  # Keep the active drag's original anchor until release.
                    histogram.setLevels((low, high))

    def _keyed_colorbar_changed(self, key, bar) -> None:
        """An operator's shared-slice colorbar edit drives the existing common scale."""
        widgets = self._keyed_panels.get(key)
        if (self._layout_mode == "keyed" and type(widgets) is _KeyedImagePanel
                and widgets.presentation == "shared" and widgets.levels is not None
                and bar is widgets.pane.canvas.histogram):
            self.viewer_intensity.set_absolute_limits(*bar.levels(), emit=False)
            self._refresh_keyed_intensity(source=bar)

    def _build_keyed(self, layout: tuple[tuple[PanelKey, ...], ...]) -> None:
        for widgets in self._keyed_panels.values():
            widgets.container.setParent(None)
            widgets.container.deleteLater()
        self._keyed_panels, self._keyed_layout = {}, ()
        while self._keyed_area.count():
            old_row = self._keyed_area.widget(0)
            old_row.setParent(None)
            old_row.deleteLater()
        for row_index, row in enumerate(layout):
            container = QtWidgets.QWidget()
            across = QtWidgets.QHBoxLayout(container)
            across.setContentsMargins(0, 0, 0, 0)
            across.setSpacing(4)
            for key in row:
                if key.role is PanelRole.SLICE_2D:
                    widgets = _KeyedImagePanel(self, key)
                elif key.role is PanelRole.PROJ_1D:
                    widgets = _KeyedTracePanel(key)
                elif key.role is PanelRole.FIGURE:
                    widgets = _KeyedFigurePanel(key, self.figureImageClicked.emit)
                else:
                    raise ValueError(f"a keyed layout draws SLICE_2D and PROJ_1D panels, not {key.role.value}")
                across.addWidget(widgets.container, 1)
                self._keyed_panels[key] = widgets
            self._keyed_area.addWidget(container)
            self._keyed_area.setStretchFactor(row_index, 1)
        self._keyed_area.setSizes([500] * len(layout))
        self._keyed_layout = layout

    def _enter_keyed_layout(self) -> None:
        if self._layout_mode == "keyed":
            return
        if self._keyed_area is None:
            self._keyed_area = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
            self._keyed_area.setObjectName("e7KeyedPanels")
            self._keyed_area.setChildrenCollapsible(False)
            self._keyed_area.setHandleWidth(8)
            self._main_layout.insertWidget(self._main_layout.indexOf(self.vertical_splitter) + 1,
                                           self._keyed_area, 1)
        self.drop_viewer_loading_snapshot()
        self._set_share_link(False)
        if self.raw_popup_dialog is not None and not self._raw_popup_host:
            self.raw_popup_dialog.close()
        hidden = tuple(widget for widget in (
            self.vertical_splitter, self.previous_frame, self.frame_selector,
            self.next_frame, self.progress, self.raw_popup_button, self.phases,
        ) if not widget.isHidden())
        for widget in hidden:
            widget.setVisible(False)
        self._keyed_hidden = hidden
        # Share Axis and the intensity controls on the 2D Viewer's intensity row, below the panels; an Int mode
        # gives them back (``_leave_keyed_layout``, ``_apply_mode_layout``)
        self._share_axis_home = self.plot_bar.indexOf(self.share_axis)
        options_layout = self.options.parentWidget().layout()
        self._options_home = options_layout, options_layout.indexOf(self.options)
        self.viewer_intensity_row_layout.addWidget(self.share_axis)
        self.viewer_intensity_row_layout.addWidget(self.options)
        self.viewer_intensity_row_layout.addWidget(self.viewer_intensity)
        self.viewer_intensity.sync(None, None, reset=True)
        self._viewer_intensity_contract = None
        self._viewer_intensity_domain = self._viewer_auto_levels = None
        for widget in (self.share_axis, self.viewer_intensity, self.viewer_intensity_row, self._keyed_area):
            widget.setVisible(True)
        self.options.show()
        self._layout_mode = self._processing_mode = "keyed"

    def _leave_keyed_layout(self) -> None:
        self._restore_figure_chrome()
        self._link_keyed_axes(False)
        if self._keyed_area is not None:
            self._keyed_area.setVisible(False)
        if self._coordinate_range_bar is not None:
            self._coordinate_range_bar.hide()
        for widget in self._keyed_hidden:
            widget.setVisible(True)
        self._keyed_hidden = ()
        if self._share_axis_home >= 0:  # back where Int's plot bar keeps it; the controls go where the mode does
            self.plot_bar.insertWidget(self._share_axis_home, self.share_axis)
        options_layout, options_index = self._options_home
        options_layout.insertWidget(options_index, self.options)
        self.viewer_intensity_row.setVisible(False)

    def _restore_figure_chrome(self):
        for widget in self._figure_hidden:
            widget.show()
        self._figure_hidden = ()

    def _emit_panel_index(self, key: PanelKey, index: int) -> None:
        self.commandRequested.emit(ShellCommand(ShellCommandKind.SET_PANEL_INDEX, int(index), (key.instance,)))

    def set_page_tools(self, *tools: tuple[str, str, str]) -> None:
        """A host page's own tools, as (name, label, tooltip): buttons in the top bar where Int's Raw pop-up
        button sits, each emitting ``PAGE_TOOL`` with its name (RSM's "3D Viewer" and "Custom Slice", whose
        windows the page opens). Calling it again replaces them."""
        if any(type(tool) is not tuple or len(tool) != 3 or any(type(part) is not str or not part
                                                                for part in tool) for tool in tools):
            raise TypeError("a page tool is (name, label, tooltip), all non-empty text")
        if len({tool[0] for tool in tools}) != len(tools):
            raise ValueError("page tools are named uniquely")
        for button in self._page_tool_buttons.values():
            self._top_bar_row.removeWidget(button)
            button.deleteLater()
        self._page_tool_buttons = {}
        at = self._top_bar_row.indexOf(self.raw_popup_button) + 1
        for offset, (name, label, tooltip) in enumerate(tools):
            button = QtWidgets.QPushButton(label)
            button.setObjectName(f"e7PageTool_{name}")
            button.setToolTip(tooltip)
            button.clicked.connect(lambda _checked=False, name=name: self._emit(ShellCommandKind.PAGE_TOOL, name))
            self._top_bar_row.insertWidget(at + offset, button)
            self._page_tool_buttons[name] = button

    def page_tool(self, name: str) -> QtWidgets.QPushButton:
        """A host page's tool button, by name."""
        return self._page_tool_buttons[name]

    #: Top-bar tools a host page may leave out, by name (some name several widgets).
    _HOST_TOOLS = {"norm": "norm", "background": "background", "raw": "raw_popup_button",
                   "raw_pane": "raw", "slice": ("slice", "slice_center", "slice_width", "pin"),
                   "phases": "phases"}
    _PLOT_MODES = ("Single", "Overlay", "Average", "Sum", "Waterfall")

    def set_plot_modes(self, *names: str) -> None:
        """Offer only the plot modes the host can show (Int and Stitch offer all). Without Waterfall,
        the options dialog leaves out its Waterfall section."""
        unknown = set(names) - set(self._PLOT_MODES)
        if not names or unknown:
            raise ValueError(f"unknown plot modes: {sorted(unknown) or 'none given'}")
        current = self.plot_mode.currentText()
        blocker = QtCore.QSignalBlocker(self.plot_mode)
        self.plot_mode.clear()
        self.plot_mode.addItems([mode for mode in self._PLOT_MODES if mode in names])
        self.plot_mode.setCurrentIndex(max(0, self.plot_mode.findText(current)))
        del blocker
        self.plot_options_dialog.set_waterfall_visible("Waterfall" in names)

    def set_hidden_tools(self, *names: str) -> None:
        """Leave out top-bar tools the host page has no operation for (Stitch: no
        normalization channel, 1-D Background or raw popup), the raw detector pane
        ('raw_pane': Stitch's map has no one detector image; Int 2D's layout keeps its cake),
        and the χ (c/w) slice with its Pin ('slice': RSM's keyed layout has no cake to cut).
        Int shows them all."""
        unknown = set(names) - set(self._HOST_TOOLS)
        if unknown:
            raise ValueError(f"unknown scientific-view tools: {sorted(unknown)}")
        self._hidden_tools = frozenset(names)
        self._apply_hidden_tools()

    def _apply_hidden_tools(self) -> None:
        for name in self._hidden_tools:
            widgets = self._HOST_TOOLS[name]
            for widget in (widgets if type(widgets) is tuple else (widgets,)):
                getattr(self, widget).setVisible(False)

    def _apply_processing_layout(self, mode: str) -> None:
        self._apply_mode_layout(mode)
        self._apply_hidden_tools()

    def _apply_mode_layout(self, mode: str) -> None:
        if self._layout_mode == "keyed":
            self._leave_keyed_layout()
        normalized = str(mode or "")
        if (
            self._viewer_loading_mode is not None
            and normalized != self._viewer_loading_mode
        ):
            self.drop_viewer_loading_snapshot()
        previous_layout = self._layout_mode
        changed = previous_layout != normalized
        self._layout_mode = normalized
        viewer = normalized in {"1D Viewer", "2D Viewer"}
        if changed:
            if viewer:
                host = self.viewer_intensity_row_layout if normalized == "2D Viewer" else self.plot_bar
                host.addWidget(self.viewer_intensity)
            else:
                self.plot_options_dialog.mount_display_scale(self.viewer_intensity)
        self.viewer_intensity_row.setVisible(normalized == "2D Viewer")
        self.viewer_intensity.setVisible(True)
        self.axis_display_group.setVisible(not viewer)
        self.plot_axis.setVisible(not viewer)
        if changed:
            self._plot_group_gap.changeSize(
                0 if viewer else PLOT_TOOLBAR_INTER_GROUP_GAP, 0,
            )
            self.plot_bar.invalidate()
            self.viewer_intensity.sync(None, None, reset=True)
            self._viewer_intensity_contract = None
            self._viewer_intensity_domain = self._viewer_auto_levels = None
            self.curve.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.raw_popup_button.setVisible(normalized == "Int 1D")
        # Reference phases are unavailable in the 2D Viewer, so is their switch (keyed layouts hide it too).
        self.phases.setVisible(normalized != "2D Viewer" and "phases" not in self._hidden_tools)
        if (normalized != "Int 1D" and self.raw_popup_dialog is not None
                and not self._raw_popup_host):
            self.raw_popup_dialog.close()
        aggregate_enabled = normalized != "1D Viewer"
        combo = getattr(self, "plot_mode", None)
        for choice in ("Average", "Sum"):
            item = (None if combo is None else
                    combo.model().item(combo.findText(choice)))
            if item is not None: item.setEnabled(aggregate_enabled)
        if not aggregate_enabled:
            self._processing_mode = normalized
            for widget in (
                self.image_splitter, self.raw, self.cake, self.norm,
                self.image_axis,
                getattr(self, "plot_axis", self.image_axis),
                self.share_axis, self.slice, self.slice_center,
                self.slice_width, self.pin,
            ):
                widget.setVisible(False)
            self.background.setVisible(True); self.vertical_splitter.widget(1).setVisible(True)
            self._set_share_link(False)
            return
        if normalized == "2D Viewer":
            self._processing_mode = "2D Viewer"
            for widget in (
                self.cake, self.vertical_splitter.widget(1), self.image_axis,
                self.share_axis, self.slice,
                self.slice_center, self.slice_width, self.pin,
            ):
                widget.setVisible(False)
            self._set_share_link(False)
            self.raw.setVisible(True)
            if self.image_splitter.isHidden():
                self.image_splitter.refresh()
            self.image_splitter.setVisible(True)
            self.background.setVisible(True)
            return
        has_2d = normalized != "Int 1D"
        prior_has_2d = previous_layout not in {"Int 1D", "1D Viewer", "2D Viewer"}
        self._processing_mode = normalized
        self.norm.setVisible(True)
        self.background.setVisible(True)
        self.raw.setVisible(True)
        self.cake.setVisible(True)
        if changed or self.image_splitter.isHidden():
            # Hidden children leave QSplitter's cached maximum at (0, 0).
            # Recompute it before the vertical splitter restores this row.
            self.image_splitter.refresh()
        self.image_splitter.setVisible(has_2d)
        self.vertical_splitter.widget(1).setVisible(True)
        for widget in (
            self.image_axis,
            self.share_axis,
            self.slice,
            self.slice_center,
            self.slice_width,
            self.pin,
        ):
            widget.setVisible(has_2d)
        if not has_2d:
            self._set_share_link(False)
        elif not prior_has_2d:
            self.vertical_splitter.setSizes([500, 500])

    def _viewer_intensity_image(self):
        if self._processing_mode == "2D Viewer":
            return self.raw
        if self._processing_mode == "1D Viewer" and self._bottom_waterfall_active:
            return self.waterfall
        if self._processing_mode == "Int 2D":
            return self.cake
        return None

    def _refresh_viewer_intensity(self) -> None:
        """Synchronize scalar display limits, never retain another pixel buffer."""
        if self._layout_mode == "keyed":
            self._refresh_keyed_intensity()
            return
        controls = self.viewer_intensity
        if self._processing_mode in {"1D Viewer", "Int 1D"}:
            controls.set_percentile_defaults(None)
        else:
            defaults = (0.5, 99.5) if self._processing_mode == "Int 2D" else (2, 98)
            controls.set_percentile_defaults((0.1, 99.9) if self.log_scale.isChecked() else defaults)
        pane = self._viewer_intensity_image()
        if pane is not None:
            if pane.image.image is None or not np.isfinite(pane.canvas.displayed_image).any():
                controls.sync(None, None)
                return
            contract = (self._processing_mode, pane._render_contract)
            if contract != self._viewer_intensity_contract:
                self._viewer_intensity_contract = contract
                histogram = pane.canvas.histogram
                self._viewer_intensity_domain = (histogram.lo_lim, histogram.hi_lim)
                self._viewer_auto_levels = tuple(histogram.levels())
            percentile = controls.percentile_values()
            levels = pane.canvas.percentile_levels(percentile) if percentile is not None else (
                self._viewer_auto_levels if controls.autoscale.isChecked() else controls.values())
            controls.sync(self._viewer_intensity_domain, levels)
            self._set_viewer_intensity(*controls.values())
            return
        ranges = []
        for item in self.curve.listDataItems():
            _x, y = item.getData()
            if y is not None:
                finite = y[np.isfinite(y)]
                if finite.size:
                    ranges.append((float(finite.min()), float(finite.max())))
        if not ranges:
            controls.sync(None, None)
            return
        domain = (min(lo for lo, _hi in ranges), max(hi for _lo, hi in ranges))
        plot = self.curve.getViewBox()
        if controls.autoscale.isChecked():
            plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            plot.updateAutoRange()
        else:
            self._set_viewer_intensity(*controls.values())
        controls.sync(domain, plot.viewRange()[1])

    def _set_viewer_intensity(self, lo, hi) -> None:
        if self._layout_mode == "keyed":  # the keyed slices' one set of levels
            self._refresh_keyed_intensity()
            return
        if (not np.isfinite((lo, hi)).all() or hi <= lo):
            return
        pane = self._viewer_intensity_image()
        if pane is None:
            self.curve.getViewBox().setYRange(lo, hi, padding=0)
        elif pane.image.image is not None:
            percentile = self.viewer_intensity.percentile_values()
            if percentile is not None:
                lo, hi = pane.canvas.percentile_levels(percentile)
            histogram = pane.canvas.histogram
            domain = self._viewer_intensity_domain or (lo, hi)
            histogram.lo_lim = min(domain[0], lo)
            histogram.hi_lim = max(domain[1], hi)
            histogram.setLevels((lo, hi))
            self.viewer_intensity.sync(domain, (lo, hi))

    def _toggle_viewer_autoscale(self, enabled) -> None:
        if self._layout_mode == "keyed":  # Autoscale's span of the keyed slices, or the levels shown, kept
            self._refresh_keyed_intensity()
            return
        if not enabled:
            self._set_viewer_intensity(*self.viewer_intensity.values())
            return
        pane = self._viewer_intensity_image()
        if pane is not None and pane.image.image is not None and self._viewer_auto_levels:
            histogram = pane.canvas.histogram
            histogram.lo_lim, histogram.hi_lim = self._viewer_intensity_domain
            histogram.setLevels(self._viewer_auto_levels)
        self._refresh_viewer_intensity()

    def _set_share_link(self, on: bool) -> None:
        if self._layout_mode == "keyed":  # the keyed panels' coordinates (``_link_keyed_axes``)
            self._link_keyed_axes(on)
            return
        cake_view = self.cake.canvas.imageViewBox
        curve_view = self.curve.getPlotItem().getViewBox()
        waterfall_view = self.waterfall.canvas.imageViewBox
        if on and not self._share_link_on:
            self._share_link_on = True
            cake_view.sigXRangeChanged.connect(self._share_cake_handler)
            curve_view.sigXRangeChanged.connect(self._share_curve_handler)
            waterfall_view.sigXRangeChanged.connect(
                self._share_curve_handler
            )
            self._schedule_curve_under_cake()
        elif not on and self._share_link_on:
            self._share_link_on = False
            self._align_seq += 1
            self._align_cake_seq += 1
            self._align_curve_pending = False
            self._align_cake_pending = False
            try:
                cake_view.sigXRangeChanged.disconnect(
                    self._share_cake_handler
                )
            except (RuntimeError, TypeError):
                pass
            for bottom_view in (curve_view, waterfall_view):
                try:
                    bottom_view.sigXRangeChanged.disconnect(
                        self._share_curve_handler
                    )
                except (RuntimeError, TypeError):
                    pass
                bottom_view.enableAutoRange(
                    axis=pg.ViewBox.XAxis,
                    enable=True,
                )
            if self._cake_x_pinned_by_share:
                cake_view.enableAutoRange(
                    axis=pg.ViewBox.XAxis,
                    enable=True,
                )
                self._cake_x_pinned_by_share = False

    def _on_cake_xrange_changed(self, *_args) -> None:
        if not self._share_link_on or self._share_axis_syncing:
            return
        self._schedule_curve_under_cake()

    def _on_curve_xrange_changed(self, *_args) -> None:
        if not self._share_link_on or self._share_axis_syncing:
            return
        self._schedule_cake_under_curve()

    def _install_share_geometry_hooks(self) -> None:
        self.vertical_splitter.splitterMoved.connect(
            self._on_share_geometry_changed
        )
        self.image_splitter.splitterMoved.connect(
            self._on_share_geometry_changed
        )
        self.cake.canvas.image_win.installEventFilter(self)
        self.curve.installEventFilter(self)
        self.waterfall.canvas.image_win.installEventFilter(self)
        # An AxisItem sizes itself lazily at paint, so a tick-label re-flow
        # moves a view box without any widget Resize/Show, splitter move or
        # resizeEvent: follow the view boxes themselves.  The scheduler
        # coalesces and budgets these follows (see _schedule_curve_under_cake).
        for view_box in (
            self.cake.canvas.imageViewBox,
            self.curve.getPlotItem().getViewBox(),
            self.waterfall.canvas.imageViewBox,
        ):
            view_box.sigResized.connect(self._on_share_view_resized)

    def _on_share_geometry_changed(self, *_args) -> None:
        self._layout_viewer_loading_snapshot()
        if self._share_link_on:
            self._schedule_curve_under_cake()

    def _on_share_view_resized(self, *_args) -> None:
        if self._share_link_on:
            self._schedule_curve_under_cake(from_reflow=True)

    def eventFilter(self, watched, event) -> bool:
        if watched in {
            self._viewer_loading_overlay,
            self._viewer_loading_pixmap,
        }:
            if event.type() in {
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.MouseButtonRelease,
                QtCore.QEvent.Type.MouseButtonDblClick,
                QtCore.QEvent.Type.MouseMove,
                QtCore.QEvent.Type.Wheel,
                QtCore.QEvent.Type.KeyPress,
                QtCore.QEvent.Type.KeyRelease,
                QtCore.QEvent.Type.ContextMenu,
            }:
                return True
        if (
            watched in self._viewer_loading_watched
            and event.type() in {
                QtCore.QEvent.Type.Move,
                QtCore.QEvent.Type.Resize,
                QtCore.QEvent.Type.Show,
                QtCore.QEvent.Type.Hide,
            }
        ):
            self._layout_viewer_loading_snapshot()
        if (
            watched
            in {
                self.cake.canvas.image_win,
                self.curve,
                self.waterfall.canvas.image_win,
            }
            and event.type() in {
                QtCore.QEvent.Type.Resize,
                QtCore.QEvent.Type.Show,
            }
            and self._share_link_on
        ):
            self._schedule_curve_under_cake()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_viewer_loading_snapshot()
        if self._share_link_on:
            self._schedule_curve_under_cake()

    def _schedule_curve_under_cake(self, *, from_reflow: bool = False) -> None:
        # Coalesce onto the first pending pair instead of restarting a
        # trailing-edge debounce.  The event-loop callback makes progress
        # during a resize/show storm; the bounded follow-up observes the final
        # pyqtgraph layout after axes and color bars settle.
        if not from_reflow:
            # A real trigger (a widget event, a cake X-range change, the
            # link, new curve data) opens a fresh follow epoch even when it
            # coalesces onto a pending pair; only the axis-driven follows
            # below are budgeted.
            self._reflow_follows = 0
        if self._align_curve_pending:
            return
        if from_reflow:
            # A clipped-to-view curve autoranges Y over the visible samples,
            # so an align can move the tick-label width, which moves the
            # align: when no fixed point exists this would ping-pong forever.
            # Budget the axis-driven follows between real triggers.
            if self._reflow_follows >= _MAX_REFLOW_FOLLOWS:
                return
            self._reflow_follows += 1
        self._align_curve_pending = True
        sequence = self._align_seq + 1
        self._align_seq = sequence

        def settle() -> None:
            if sequence != self._align_seq:
                return
            self._align_curve_pending = False
            self._align_curve_under_cake()

        def align() -> None:
            if sequence != self._align_seq:
                return
            self._align_curve_under_cake()
            QtCore.QTimer.singleShot(50, settle)

        QtCore.QTimer.singleShot(0, align)

    def _schedule_cake_under_curve(self) -> None:
        # Only a user/external curve X-range change reaches here (the link's
        # own feedback is fenced by _share_axis_syncing): a real trigger for
        # the re-flow budget as much as a cake-side change is.
        self._reflow_follows = 0
        if self._align_cake_pending:
            return
        self._align_cake_pending = True
        sequence = self._align_cake_seq + 1
        self._align_cake_seq = sequence

        def settle() -> None:
            if sequence != self._align_cake_seq:
                return
            self._align_cake_pending = False
            self._align_cake_under_curve()

        def align() -> None:
            if sequence != self._align_cake_seq:
                return
            self._align_cake_under_curve()
            QtCore.QTimer.singleShot(50, settle)

        QtCore.QTimer.singleShot(0, align)

    @staticmethod
    def _global_xspan(widget, view_box) -> tuple[float, float]:
        # pyqtgraph maps the view range onto view_box.rect(); the bounding
        # rect is half a pen wider, the box sits at a fractional scene x
        # (axis widths follow the font metrics), and mapFromScene rounds to
        # whole pixels.  Any of those errors extrapolates into a visible
        # column offset on the linked plot, so take the rect the transform
        # targets and keep its sub-pixel position: map through the viewport
        # transform and add the viewport's integer origin.
        rect = view_box.mapRectToScene(view_box.rect())
        transform = widget.viewportTransform()
        origin = float(
            widget.viewport().mapToGlobal(QtCore.QPoint(0, 0)).x()
        )
        left = origin + transform.map(rect.topLeft()).x()
        right = origin + transform.map(rect.bottomRight()).x()
        return float(left), float(right)

    def _share_geometry(self):
        cake_widget = self.cake.canvas.image_win
        curve_widget, curve_view = self._active_bottom_plot()
        if not (
            self._share_link_on
            and cake_widget.isVisible()
            and curve_widget.isVisible()
            and self._rendered_cake_axis_key is not None
            and self._rendered_cake_axis_key
            == self._rendered_trace_axis_key
        ):
            return None
        cake_view = self.cake.canvas.imageViewBox
        cake_span = self._global_xspan(cake_widget, cake_view)
        curve_span = self._global_xspan(curve_widget, curve_view)
        if (
            cake_span[1] - cake_span[0] <= 1.0
            or curve_span[1] - curve_span[0] <= 1.0
        ):
            return None
        return cake_view, curve_view, cake_span, curve_span

    def _active_bottom_plot(self):
        if self.bottom_stack.currentWidget() is self.waterfall:
            return (
                self.waterfall.canvas.image_win,
                self.waterfall.canvas.imageViewBox,
            )
        return self.curve, self.curve.getPlotItem().getViewBox()

    def _align_curve_under_cake(self) -> None:
        geometry = self._share_geometry()
        if geometry is None:
            return
        cake_view, curve_view, (cx0, cx1), (px0, px1) = geometry
        cq0, cq1 = (
            float(value) for value in cake_view.viewRange()[0]
        )
        scale = (cq1 - cq0) / (cx1 - cx0)
        desired = (
            cq0 - (cx0 - px0) * scale,
            cq0 + (px1 - cx0) * scale,
        )
        if not self._range_needs_update(
            curve_view,
            desired,
            pixel_scale=abs(scale),
        ):
            return
        prior_syncing = self._share_axis_syncing
        self._share_axis_syncing = True
        try:
            # X only.  Y autorange is owned elsewhere (mode entry, the
            # explicit intensity autoscale, a changed slice) and a follow
            # must not discard a manual Y zoom: with clipToView the zoom
            # itself re-flows the tick labels, which is one of the very
            # triggers that land here.
            curve_view.enableAutoRange(
                axis=pg.ViewBox.XAxis,
                enable=False,
            )
            curve_view.setXRange(*desired, padding=0.0)
        finally:
            self._share_axis_syncing = prior_syncing

    def _align_cake_under_curve(self) -> None:
        geometry = self._share_geometry()
        if geometry is None:
            return
        cake_view, curve_view, (cx0, cx1), (px0, px1) = geometry
        pq0, pq1 = (
            float(value) for value in curve_view.viewRange()[0]
        )
        scale = (pq1 - pq0) / (px1 - px0)
        desired = (
            pq0 + (cx0 - px0) * scale,
            pq0 + (cx1 - px0) * scale,
        )
        if not self._range_needs_update(
            cake_view,
            desired,
            pixel_scale=abs(scale),
        ):
            return
        prior_syncing = self._share_axis_syncing
        self._share_axis_syncing = True
        try:
            cake_view.enableAutoRange(
                axis=pg.ViewBox.XAxis,
                enable=False,
            )
            cake_view.setXRange(*desired, padding=0.0)
            self._cake_x_pinned_by_share = True
        finally:
            self._share_axis_syncing = prior_syncing

    @staticmethod
    def _range_needs_update(
        view_box,
        desired: tuple[float, float],
        *,
        pixel_scale: float,
    ) -> bool:
        if (
            not all(np.isfinite(value) for value in desired)
            or desired[1] <= desired[0]
        ):
            return False
        current = tuple(
            float(value) for value in view_box.viewRange()[0]
        )
        tolerance = max(1.0e-12, pixel_scale * 0.5)
        return not np.allclose(
            current,
            desired,
            rtol=0.0,
            atol=tolerance,
        )

    def _navigate(self, offset: int) -> None:
        if not self._frame_keys:
            return
        current = self.frame_selector.currentIndex()
        if current < 0:
            return
        target = min(max(current + offset, 0), len(self._frame_keys) - 1)
        if target != current:
            self.frame_selector.setCurrentIndex(target)

    def _emit_range(self, axis: str, edge: str, value: float) -> None:
        self.commandRequested.emit(
            ShellCommand(
                ShellCommandKind.SET_RANGE,
                float(value),
                (axis, edge),
            )
        )

    def _emit(
        self,
        kind: ShellCommandKind,
        value=None,
        path: tuple[str, ...] = (),
    ) -> None:
        self.commandRequested.emit(ShellCommand(kind, value, path))


def _offset_intensity(
    values: np.ndarray,
    index: int,
    offset_step: float,
) -> np.ndarray:
    if index == 0 or offset_step == 0.0:
        return values
    return values + index * offset_step


def _axis_supports_view_clipping(values: np.ndarray) -> bool:
    """Return whether pyqtgraph may safely clip this ordered x domain."""

    axis = np.asarray(values)
    try:
        return bool(
            axis.ndim == 1
            and axis.size > 1
            and np.isfinite(axis).all()
            and np.all(axis[1:] > axis[:-1])
        )
    except (TypeError, ValueError):
        return False


def _overlay_step(
    traces,
    offset_percent: float,
) -> float:
    return overlay_step((trace.intensity for trace in traces), offset_percent)
_MAX_RETAINED_CURVE_ITEMS = 32
# Axis-re-flow-driven share-link follows allowed between real triggers: a
# legitimate chain is two (the follow's own Y autorange may re-flow once).
_MAX_REFLOW_FOLLOWS = 3


__all__ = ["ScientificView"]
