"""Raw, cake, and retained 1-D rendering for the passive E3 shell."""

from __future__ import annotations

from dataclasses import replace
import time

from matplotlib import colormaps as matplotlib_colormaps
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.modules.display_context import (
    Viewer2DRendererClearReceipt, Viewer2DRendererClearRequest,
)
from xdart.gui.themes import apply_seaborn_plot_style
from xrd_tools.session.display_logic import (
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
    PLOT_OPTION_COMMAND_PATHS,
    PlotOptionsDialog,
    waterfall_should_be_active,
)
from .presentation_background import DisplayBackgroundRendererReleaseReceipt
from .shell_widgets import (
    _axis_presentation,
    aggregate_traces,
    CompactFrameSelector,
    ContentFitComboBox,
    frame_caption,
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
_PLOT_OPTION_FIELDS = (
    "waterfall_y_axis",
    "waterfall_start",
    "waterfall_stop",
    "waterfall_step",
    "overlay_offset",
    "show_legend",
    "intensity_scale",
)
_CONVERTIBLE_RADIAL_AXIS_KEYS = frozenset({"q_A^-1", "2th_deg"})
_STANDARD_PLOT_AXIS_CHOICES = (
    ("Q (Å⁻¹)", "Q"),
    ("2θ (°)", "2theta"),
    ("χ (°)", "chi"),
)
_STANDARD_IMAGE_AXIS_CHOICES = (
    ("Q-χ", "Q-Chi"),
    ("2θ-χ", "2Th-Chi"),
    ("Qz-Qxy", "Qz-Qxy"),
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
_GI_IMAGE_AXIS_CHOICES = {
    "qip_qoop": (("Qᵢₚ-Qₒₒₚ", "qip_qoop"),),
    "q_chi": (("Q-χ", "q_chi"),),
    "exit_angles": (("Exit angles", "exit_angles"),),
}


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


class ScientificView(QtWidgets.QFrame):
    commandRequested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("e3ScientificView")
        self.setMinimumWidth(300)
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
        self._pinned_trace_scope: tuple[object, ...] | None = None
        self._pinned_trace_by_id: dict[
            tuple[object, ...], TraceProjection
        ] = {}
        self._rendered_trace_keys: tuple[tuple[object, ...], ...] = ()
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
        self._viewer_2d_payload = None
        self._viewer_2d_known_empty = False
        self._rendered_detector_source = "none"
        self._expected_background_key = self._rendered_background_key = None
        self.raw_popup_dialog = None
        self.raw_popup_image = None
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
        self.curve = pg.PlotWidget()
        self.curve.setObjectName("e3OneDPlot")
        apply_seaborn_plot_style(self.curve.getPlotItem(), grid=True)
        self.curve.getPlotItem().layout.setContentsMargins(
            0,
            0,
            0,
            ONE_D_PLOT_BOTTOM_MARGIN,
        )
        self.legend = self.curve.addLegend()
        self.cursor_position = pg.LabelItem(justify="right")
        self.cursor_position.setParentItem(self.curve.getPlotItem())
        self.cursor_position.anchor(
            itemPos=(1, 0),
            parentPos=(1, 0),
            offset=(-14, 8),
        )
        self._mouse_proxy = pg.SignalProxy(
            signal=self.curve.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._mouse_moved,
        )
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

        self.footer = QtWidgets.QHBoxLayout()
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
        self._install_share_geometry_hooks()

    @property
    def rendered_image_axis(self) -> str | None:
        """Identity of the cake presentation actually accepted by the view."""

        return self._rendered_image_axis

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

    def expect_display_background(self, active_key) -> None:
        key = active_key if type(active_key) is tuple else None
        if key != self._expected_background_key:
            self._rendered_trace_keys = (); self._rendered_plot_mode = ""
            self._waterfall_source_keys = (); self._waterfall_render_contract = None
            self._rendered_browse_science_contract = None
        self._expected_background_key = key

    def release_display_background(self, active_key):
        if active_key != self._rendered_background_key: return DisplayBackgroundRendererReleaseReceipt(active_key, False)
        domain = active_key[2]
        try:
            if domain == "raw":
                self._scrub_detector_pane(self.raw); self._viewer_2d_payload = None
                self._viewer_2d_known_empty = False
                self._rendered_detector_source = "none"
                if self.raw_popup_image is not None: self._scrub_detector_pane(self.raw_popup_image)
            elif domain == "integrated_2d":
                self.cake.clear(); self._rendered_cake_x_axis = self._rendered_cake_y_axis = self._rendered_cake_axis_key = self._rendered_image_axis = None
            else:
                self.curve.clear(); self.waterfall.clear(); self._trace_history_by_identity.clear(); self._pinned_trace_by_id.clear()
                self._rendered_trace_keys = self._trace_history_keys = self._waterfall_source_keys = ()
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
        self.title = QtWidgets.QLabel("Current")
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.color_map = ContentFitComboBox()
        self._color_map_choices = (
            "Default",
            *tuple(matplotlib_colormaps),
        )
        self.color_map.addItems(self._color_map_choices)
        self.log_scale = QtWidgets.QPushButton("Log")
        self.log_scale.setCheckable(True)
        self.detector_controls = QtWidgets.QWidget()
        detector_layout = QtWidgets.QHBoxLayout(self.detector_controls)
        detector_layout.setContentsMargins(0, 0, 0, 0)
        detector_layout.setSpacing(0)
        self.detector_thumbnail = QtWidgets.QPushButton("Thumbnail")
        self.detector_full = QtWidgets.QPushButton("Full Raw")
        self.detector_mode_group = QtWidgets.QButtonGroup(self)
        self.detector_mode_group.setExclusive(True)
        for button in (self.detector_thumbnail, self.detector_full):
            button.setCheckable(True)
            self.detector_mode_group.addButton(button)
            detector_layout.addWidget(button)
        self.raw_popup_button = QtWidgets.QPushButton("Raw")
        row.addWidget(self.norm)
        row.addWidget(self.background)
        row.addWidget(self.raw_popup_button)
        row.addWidget(self.detector_controls)
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
        self.color_map.currentTextChanged.connect(
            lambda value: self._emit(ShellCommandKind.SET_COLOR_MAP, value)
        )
        self.log_scale.toggled.connect(
            lambda value: self._emit(
                ShellCommandKind.SET_LOG_SCALE, bool(value)
            )
        )
        self.detector_thumbnail.clicked.connect(
            lambda: self._emit(ShellCommandKind.SET_DETECTOR_MODE, "thumbnail")
        )
        self.detector_full.clicked.connect(
            lambda: self._emit(ShellCommandKind.SET_DETECTOR_MODE, "full")
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
        self.raw_popup_mode_group = QtWidgets.QButtonGroup(dialog)
        self.raw_popup_mode_group.setExclusive(True)
        for button in (self.raw_popup_thumbnail, self.raw_popup_full):
            button.setCheckable(True)
            self.raw_popup_mode_group.addButton(button)
            controls.addWidget(button)
        controls.addStretch(1)
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
        self.raw_popup_dialog.show(); self.raw_popup_dialog.raise_()
        self._emit(ShellCommandKind.SET_DETECTOR_MODE, "thumbnail", ("popup",))
        self.raw_popup_full.setChecked(True)
        self._emit(ShellCommandKind.SET_DETECTOR_MODE, "full", ("popup",))

    def _release_raw_popup(self, *_args) -> None:
        if self.raw_popup_image is not None:
            self._scrub_detector_pane(self.raw_popup_image)
        if hasattr(self, "raw_popup_status"): self.raw_popup_status.setText("")
        if self._processing_mode == "Int 1D":
            self._emit(ShellCommandKind.SET_DETECTOR_MODE, "thumbnail", ("popup",))

    @staticmethod
    def _scrub_detector_pane(pane) -> None:
        pane.clear()
        for name in ("qimage", "levels", "_defferedLevels", "_displayBuffer", "_processingBuffer", "_imageNanLocations", "_imageHasNans"):
            setattr(pane.image, name, None)

    def _reconcile_raw_popup(self, state: ScientificProjection) -> None:
        heavy = state.heavy
        dialog = self.raw_popup_dialog
        if dialog is None or not dialog.isVisible():
            return
        if heavy is None or heavy.raw is None:
            self._scrub_detector_pane(self.raw_popup_image)
            return
        if heavy.detector_source != "full":
            self._scrub_detector_pane(self.raw_popup_image)
        self.raw_popup_image.render(
            heavy.raw, detector_shape=heavy.detector_shape,
            color_map=state.color_map, log_scale=state.log_scale,
            level_scan_token=(id(heavy.frame), id(heavy.raw)),
        )
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
        self.plot_mode.addItems(
            ["Single", "Overlay", "Average", "Sum", "Waterfall"]
        )
        self.options = _CompactMenuButton("Options")
        self.plot_options_dialog = PlotOptionsDialog(self)
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

    def _show_plot_options(self) -> None:
        self._emit(ShellCommandKind.SHOW_WATERFALL_OPTIONS)
        self.plot_options_dialog.show()
        self.plot_options_dialog.raise_()
        self.plot_options_dialog.activateWindow()

    def _accept_plot_options(self) -> None:
        prior = self.plot_options_dialog.projection
        edited = self.plot_options_dialog.edited_options()
        for field, path in zip(
            _PLOT_OPTION_FIELDS,
            PLOT_OPTION_COMMAND_PATHS,
            strict=True,
        ):
            old_value = getattr(prior, field)
            new_value = getattr(edited, field)
            if new_value != old_value:
                self._emit(
                    ShellCommandKind.SET_PLOT_OPTION,
                    new_value,
                    path,
                )

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
        self.background.setText(("Clear" if state.background_set else "Set") + " " + {"Int 1D": "1D", "1D Viewer": "1D", "Int 2D": "2D", "2D Viewer": "Raw"}.get(state.processing_mode, "BG") + " BG")
        self._rendered_background_key = self._expected_background_key if state.background_set else None
        if state.processing_mode == "2D Viewer":
            entering_viewer = self._processing_mode != "2D Viewer"
            self._processing_mode = "2D Viewer"
            if state.heavy is None:
                cleared = ScientificView.clear_viewer_2d(self, None, failure=True)
                if not cleared:
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
                self.raw.render(heavy.raw, detector_shape=heavy.detector_shape,
                                color_map=state.color_map,
                                log_scale=state.log_scale,
                                level_scan_token=(id(frame), id(heavy.raw)))
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
            self.detector_thumbnail,
            self.detector_full,
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
        self.detector_thumbnail.setChecked(state.detector_mode == "thumbnail")
        self.detector_full.setChecked(state.detector_mode == "full")
        self.detector_full.setEnabled(state.detector_available)
        self.raw_popup_button.setEnabled(state.detector_available)
        explanation = state.detector_diagnostic if not state.detector_available else ""
        self.detector_full.setToolTip(explanation)
        self.raw_popup_button.setToolTip(explanation or "Show exact-current raw image")
        if replace_presentation:
            self.image_axis.setEnabled(state.measurement_mode != "GI")
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
                        not raw_matches
                        or self._rendered_detector_source
                        != state.heavy.detector_source
                    ):
                        self._scrub_detector_pane(self.raw)
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
                    self._scrub_detector_pane(self.raw)
                    self._rendered_detector_source = "none"
                if state.heavy.cake is not None:
                    self.cake.render(
                        state.heavy.cake,
                        x_axis=state.heavy.cake_x,
                        y_axis=state.heavy.cake_y,
                        color_map=color_map,
                        log_scale=state.log_scale,
                        level_scan_token=(
                            id(state.heavy.frame),
                            id(state.heavy.cake),
                        ),
                    )
                    self._rendered_cake_axis_key = self._axis_key(
                        state.heavy.cake_x
                    )
                    self._rendered_cake_x_axis = state.heavy.cake_x
                    self._rendered_cake_y_axis = state.heavy.cake_y
                    self._rendered_image_axis = state.image_axis
                else:
                    self.cake.clear()
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
                self._scrub_detector_pane(self.raw)
                self._rendered_detector_source = "none"
                self.cake.clear()
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
        del blockers

    def clear_viewer_2d(self, request, *, failure=False):
        if not failure and type(request) is not Viewer2DRendererClearRequest:
            return None
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
        for pane in (self.raw, self.cake):
            scrub(pane, "clear")
            canvas = scrub(pane, "canvas", read=True)
            image = scrub(canvas, "imageItem", read=True)
            histogram = scrub(canvas, "histogram", read=True)
            view = scrub(canvas, "imageViewBox", read=True)
            hover = scrub(image, "pos_label", read=True)
            scrub(hover, "setText", "")
            for target, name, value in (
                (canvas, "raw_image", np.zeros(0)), (canvas, "displayed_image", np.zeros(0)),
                (canvas, "_level_cache", None), (canvas, "_level_scan_token", None),
                (histogram, "lo_lim", None), (histogram, "hi_lim", None),
                (image, "image", None), (image, "qimage", None), (image, "levels", None),
                (image, "_defferedLevels", None), (image, "_lastDownsample", (1, 1)),
                (image, "_displayBuffer", None), (image, "_processingBuffer", None),
                (image, "_imageNanLocations", None), (image, "_imageHasNans", None),
            ):
                scrub(target, name, value)
            scrub(histogram, "setLevels", (0.0, 1.0))
            scrub(view, "setRange", QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
            for method in ("resetTransform", "prepareGeometryChange",
                           "informViewBoundsChanged", "update"):
                scrub(image, method)
        for root in (self.curve, self.waterfall):
            scrub(root, "clear")
        try:
            blocker = QtCore.QSignalBlocker(self.frame_selector)
        except Exception:
            blocker = None
            cleared = False
        scrub(self.frame_selector, "clear")
        scrub(self.frame_selector, "setCurrentIndex", -1)
        del blocker
        bottom = scrub(self.vertical_splitter, "widget", 1)
        for widget in (
            self.image_splitter, self.raw, self.cake, bottom, self.norm, self.background,
            self.image_axis, self.share_axis, self.slice,
            self.slice_center, self.slice_width, self.pin,
        ):
            scrub(widget, "hide")
        for name, value in (
            ("_viewer_2d_payload", None), ("_frame_keys", ()), ("_current_key", None),
            ("_rendered_detector_source", "none"),
            ("_selected_keys", ()),
            ("_trace_selection_keys", ()), ("_trace_history_keys", ()), ("_rendered_trace_keys", ()),
            ("_trace_row_count", 0),
            ("_waterfall_y_values", ()), ("_waterfall_source_keys", ()), ("_label_indices", {}),
            ("_heavy_available", frozenset()), ("_trace_history_scope", None), ("_pinned_trace_scope", None),
            ("_rendered_plot_options", None), ("_rendered_overlay_step", None), ("_trace_history_by_identity", {}),
            ("_pinned_trace_by_id", {}), ("_rendered_plot_mode", ""), ("_bottom_waterfall_active", False),
            ("_rendered_browse_science_contract", None),
            ("_waterfall_render_contract", None), ("_rendered_image_axis", None), ("_rendered_cake_axis_key", None),
            ("_rendered_cake_x_axis", None), ("_rendered_cake_y_axis", None), ("_rendered_trace_axis_key", None),
        ):
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
            scrub(widget, method, value)
        self._viewer_2d_known_empty = bool(cleared)
        return cleared if failure else Viewer2DRendererClearReceipt(request, cleared)

    def clear_viewer_1d(self, request, *, failure=False):
        if not failure and type(request) is not Viewer1DRendererClearRequest:
            return None
        cleared = ScientificView.clear_viewer_2d(self, None, failure=True)
        try:
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
        return (cleared if failure else
                _new_viewer_1d_renderer_clear_receipt(request, cleared))

    def clear_workspace(self) -> bool:
        """Drop every mounted scientific payload before owner retirement."""
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
            viewer_scope = {"1D Viewer": "viewer-1d",
                            "2D Viewer": "viewer-2d"}.get(self._processing_mode)
            viewer = frame.source_scan == frame.artifact == viewer_scope
            label = frame.local_frame_label
            indices = self._label_indices.setdefault(label, [])
            self.frame_selector.add_frame(
                str(label) if viewer else
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
        navigation: FrameNavigationProjection,
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

        if type(navigation) is not FrameNavigationProjection:
            raise TypeError("navigation rebind requires an exact projection")
        current = navigation.current
        frames = (
            tuple(
                frame
                for frame in navigation.frames
                if frame.artifact == current.artifact
            )
            if current is not None
            else ()
        )
        frame_by_id = {id(frame): frame for frame in frames}
        if (
            current is None
            or type(heavy_available) is not frozenset
            or len(frames) != len(self._frame_keys)
            or len(navigation.selected) != len(self._trace_history_keys)
            or not any(frame is current for frame in heavy_available)
            or any(
                type(frame) is not DisplayFrameKey
                or frame_by_id.get(id(frame)) is not frame
                for frame in heavy_available
            )
            or any(
                old.local_frame_label != new.local_frame_label
                for old, new in zip(self._frame_keys, frames, strict=True)
            )
            or any(
                old.local_frame_label != new.local_frame_label
                for old, new in zip(
                    self._trace_history_keys,
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
        identity_map = {
            id(old): id(new)
            for old, new in zip(
                self._trace_history_keys,
                navigation.selected,
                strict=True,
            )
        }
        rebound = tuple(
            replace(trace, frame=frame)
            for trace, frame in zip(
                old_traces, navigation.selected, strict=True,
            )
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
        self._waterfall_source_keys = tuple(
            rekey(key) for key in self._waterfall_source_keys
        )
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
        del blocker
        return self.plot_axis.currentData() == plot_choice

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
            if not compatible_single:
                self._rendered_trace_keys = ()
                self._rendered_plot_mode = ""
            self._waterfall_source_keys = ()
            self._waterfall_render_contract = None
        else:
            self._rendered_browse_science_contract = None
        live_traces = self._merge_trace_history(state, navigation)
        pinned = self._merge_pinned_trace_history(state)
        rows = (
            *((("pin", *pin_id), trace) for pin_id, trace in pinned),
            *((("live", id(trace.frame)), trace) for trace in live_traces),
        )
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
            else state.plot_mode == "Waterfall"
            if state.processing_mode == "1D Viewer"
            else waterfall_should_be_active(
                state.plot_mode, len(rows),
                was_active=self._bottom_waterfall_active)
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
        if state.plot_mode in {"Average", "Sum"}:
            source_keys = keys
            traces = aggregate_traces(traces, state.plot_mode)
            keys = (("aggregate", state.plot_mode, *source_keys),)
        axis_keys = {
            self._axis_key(trace.axis)
            for trace in traces
        }
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
            {
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
        overlay_step = _overlay_step(
            traces,
            state.plot_options.overlay_offset,
        )
        incremental = (
            state.plot_mode in {"Overlay", "Waterfall"}
            and state.plot_mode == self._rendered_plot_mode
            and state.plot_options == self._rendered_plot_options
            and self._rendered_overlay_step == overlay_step
            and len(keys) >= len(self._rendered_trace_keys)
            and all(
                key == self._rendered_trace_keys[index]
                for index, key in enumerate(self._rendered_trace_keys)
            )
        )
        existing_items = tuple(self.curve.listDataItems())
        if incremental and len(existing_items) != len(
            self._rendered_trace_keys
        ):
            incremental = False
        reuse_single = (
            not incremental
            and state.plot_mode == "Single"
            and len(keys) == 1
            and keys[0][0] == "live"
            and len(existing_items) == 1
            and len(self._rendered_trace_keys) == 1
            and self._rendered_plot_mode == "Single"
            and state.plot_options == self._rendered_plot_options
            and self._rendered_trace_axis_key == prior_trace_axis_key
            and not state.share_axis
            and self.bottom_stack.currentWidget() is self.curve
        )
        start = len(self._rendered_trace_keys) if incremental else 0
        if reuse_single:
            trace = traces[0]
            title = trace.title or str(trace.frame.local_frame_label)
            item = existing_items[0]
            item.setData(
                trace.axis.values,
                trace.intensity,
                name=title,
                connect="finite",
            )
            label = self.legend.getLabel(item)
            if label is not None:
                label.setText(title)
            start = 1
        elif not incremental:
            self.curve.clear()
            legend = self.curve.getPlotItem().legend
            self.legend = (
                self.curve.addLegend()
                if legend is None
                else legend
            )
        for index, trace in enumerate(traces[start:], start):
            y_values = _offset_intensity(
                trace.intensity,
                index,
                (
                    overlay_step
                    if stacked_selection
                    else 0.0
                ),
            )
            color = _TRACE_COLORS[index % len(_TRACE_COLORS)]
            self.curve.plot(
                trace.axis.values,
                y_values,
                name=trace.title or str(trace.frame.local_frame_label),
                pen=pg.mkPen(
                    color=color,
                    width=1.4,
                    style=QtCore.Qt.PenStyle.SolidLine,
                ),
                symbol="o",
                symbolBrush=color,
                symbolPen=color,
                symbolSize=4,
                connect="finite",
            )
        if traces:
            axis = traces[0].axis
            label, unit = _axis_presentation(axis.label, axis.unit)
            self.curve.setLabel(
                "bottom",
                label,
                units=unit,
            )
        else:
            self.curve.setLabel("bottom", "", units="")
        self.curve.setLabel("left", f"{intensity} (a.u.)")
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

    def _merge_trace_history(
        self,
        state: ScientificProjection,
        navigation: FrameNavigationProjection,
    ) -> tuple[TraceProjection, ...]:
        """Merge exact selected identities into detached 1-D history."""

        scope = (
            state.plot_mode,
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
            self._rendered_trace_keys = ()
            self._rendered_plot_mode = ""
            self._rendered_plot_options = None
            self._rendered_overlay_step = None
            self._waterfall_source_keys = ()
            self._waterfall_render_contract = None
        elif viewer_single:
            # Viewer Single is exact-current science, so its semantic history
            # is reseeded for each navigation step.  Its compatible rendered
            # item remains mounted for the setData reuse path below.
            self._trace_history_by_identity.clear()
        selected_by_id = {id(frame): frame for frame in selected}
        for trace in state.traces:
            frame = trace.frame
            if selected_by_id.get(id(frame)) is frame:
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
        if self._processing_mode == "1D Viewer":
            x_values = axis.values
        else:
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
        position = positions
        if type(positions) is tuple:
            if not positions:
                return
            position = positions[0]
        if not isinstance(position, QtCore.QPointF):
            return
        plot = self.curve.getPlotItem()
        if not plot.sceneBoundingRect().contains(position):
            self.cursor_position.setText("")
            self.curve.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            return
        point = plot.vb.mapSceneToView(position)
        self.cursor_position.setText(
            f"x={point.x():.2f}, y={point.y():.2e}"
        )
        self.curve.setCursor(QtCore.Qt.CursorShape.CrossCursor)

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
        if state.measurement_mode == "GI":
            image_choices = _GI_IMAGE_AXIS_CHOICES.get(
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
            native_choices = _GI_PLOT_AXIS_CHOICES.get(
                state.gi_mode_1d,
                _GI_PLOT_AXIS_CHOICES["q_total"],
            )
            return (
                native_choices
                if state.processing_mode == "Int 1D"
                else self._merged_axis_choices(
                    native_choices,
                    _GI_CAKE_PLOT_AXIS_CHOICES.get(state.gi_mode_2d, ()),
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
        if (self._processing_mode == "1D Viewer"
                and frame.source_scan == frame.artifact == "viewer-1d"
                and any(frame is item for item in self._frame_keys)):
            membership = (
                (frame,)
                if self._single_mode
                else self._selected_keys or (frame,)
            )
            self.commandRequested.emit(ShellCommand(
                ShellCommandKind.SELECT_FRAME, frame=frame,
                frames=membership))
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
            # already-aligned repaint a no-op.
            self._align_curve_under_cake()

    def _apply_processing_layout(self, mode: str) -> None:
        normalized = str(mode or "")
        self.detector_controls.setVisible(normalized == "Int 2D")
        self.raw_popup_button.setVisible(normalized == "Int 1D")
        if normalized != "Int 1D" and self.raw_popup_dialog is not None:
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
            self.vertical_splitter.setSizes([0, 1])
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
            for widget in (self.raw, self.image_splitter, self.background): widget.setVisible(True)
            return
        has_2d = normalized != "Int 1D"
        prior_has_2d = self._processing_mode not in {"Int 1D", "1D Viewer"}
        self._processing_mode = normalized
        self.norm.setVisible(True)
        self.background.setVisible(True)
        self.image_splitter.setVisible(has_2d)
        self.raw.setVisible(True)
        self.cake.setVisible(True)
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

    def _set_share_link(self, on: bool) -> None:
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

    def _on_share_geometry_changed(self, *_args) -> None:
        if self._share_link_on:
            self._schedule_curve_under_cake()

    def eventFilter(self, watched, event) -> bool:
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
        if self._share_link_on:
            self._schedule_curve_under_cake()

    def _schedule_curve_under_cake(self) -> None:
        # Coalesce onto the first pending pair instead of restarting a
        # trailing-edge debounce.  The event-loop callback makes progress
        # during a resize/show storm; the bounded follow-up observes the final
        # pyqtgraph layout after axes and color bars settle.
        if self._align_curve_pending:
            return
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
        rect = view_box.sceneBoundingRect()
        left = widget.mapToGlobal(
            widget.mapFromScene(rect.topLeft())
        ).x()
        right = widget.mapToGlobal(
            widget.mapFromScene(rect.bottomRight())
        ).x()
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
            curve_view.enableAutoRange(
                axis=pg.ViewBox.XAxis,
                enable=False,
            )
            curve_view.enableAutoRange(
                axis=pg.ViewBox.YAxis,
                enable=True,
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


def _scaled_intensity(values: np.ndarray, scale: str) -> np.ndarray:
    if scale == "Linear":
        return values
    result = np.asarray(values, dtype=float).copy()
    finite = result[np.isfinite(result)]
    if not finite.size:
        return result
    if scale == "Log":
        minimum = float(np.min(finite))
        if minimum < 1.0:
            result -= minimum - 1.0
        return np.log10(result)
    if scale == "Sqrt":
        return np.sign(result) * np.sqrt(np.abs(result))
    raise ValueError(f"unsupported intensity scale: {scale}")


def _scaled_intensity_label(label: str, scale: str) -> str:
    if scale == "Log":
        return f"Log {label}"
    if scale == "Sqrt":
        return f"√{label}"
    return label


def _offset_intensity(
    values: np.ndarray,
    index: int,
    offset_step: float,
) -> np.ndarray:
    if index == 0 or offset_step == 0.0:
        return values
    return values + index * offset_step


def _overlay_step(
    traces,
    offset_percent: float,
) -> float:
    minimum = np.inf
    maximum = -np.inf
    for trace in traces:
        finite = trace.intensity[np.isfinite(trace.intensity)]
        if finite.size:
            minimum = min(minimum, float(np.min(finite)))
            maximum = max(maximum, float(np.max(finite)))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return 0.0
    return float(offset_percent) * (maximum - minimum) / 100.0


_TRACE_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)


__all__ = ["ScientificView"]
