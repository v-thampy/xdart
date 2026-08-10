"""Small passive Qt helpers shared by E3 shell views."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.widgets.image_widget import pgImageWidget
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.display_logic import pretty_unit, x_axis_for_unit
from xrd_tools.session.readiness import (
    ControlPanelRenderState,
    ProcessingPage,
)
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    is_single_image_spec,
)

from .contracts import (
    SourceCountScope,
    SourceObservation,
    SourceObservationStatus,
)
from .controls_readiness import (
    ControlsReadinessProjection,
    SectionHeaderProjection,
    project_directories_ready,
)
from .detector_projection import detector_calibration_ready
from .display_values import DisplayFrameKey
from .shell_values import AxisProjection, TraceProjection


def _axis_text(text: str) -> str:
    normalized = text.strip().lower()
    if normalized in {"chi", "χ"}:
        return "χ"
    if normalized in {"2th", "2theta", "2θ"}:
        return "2θ"
    return text


def _unit_text(text: str) -> str:
    normalized = text.strip().lower().replace(" ", "")
    if normalized in {"a^-1", "a-1", "å^-1", "å⁻¹", "q_a^-1"}:
        return "Å⁻¹"
    if normalized in {
        "deg",
        "degree",
        "degrees",
        "chi_deg",
        "2th_deg",
        "°",
    }:
        return "°"
    return text


def _axis_presentation(label: str, unit: str) -> tuple[str, str]:
    canonical_label, canonical_unit = x_axis_for_unit(unit)
    if canonical_label != "x" or canonical_unit:
        return canonical_label, _unit_text(canonical_unit)
    return _axis_text(label), _unit_text(pretty_unit(unit))


class _ScientificImageWidget(pgImageWidget):
    """Canonical image widget with processed-image level semantics."""

    def _cached_levels(
        self,
        scale,
        cmap,
        pct,
        data_range,
        *,
        expand_degenerate=False,
    ):
        if self.raw:
            return super()._cached_levels(
                scale,
                cmap,
                pct,
                data_range,
                expand_degenerate=expand_degenerate,
            )
        raw = self.raw_image
        try:
            # Cakes are processed intensities, not detector pixels.  A
            # non-image provenance shape keeps the shared percentile/cache
            # path while disabling its dtype-lost detector-ceiling heuristic.
            self.raw_image = np.empty(0, dtype=np.asarray(raw).dtype)
            return super()._cached_levels(
                scale,
                cmap,
                pct,
                data_range,
                expand_degenerate=expand_degenerate,
            )
        finally:
            self.raw_image = raw


class ScientificImagePane(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        lock_aspect: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Match the accepted canonical adapter: resolve the default map before
        # a live frame arrives on the GUI heartbeat.
        pg.colormap.getFromMatplotlib("viridis")
        self.canvas = _ScientificImageWidget(
            lockAspect=lock_aspect,
            raw=lock_aspect,
        )
        self.plot = self.canvas.image_plot
        self.image = self.canvas.imageItem
        self.color_scale = self.canvas.histogram
        layout.addWidget(self.canvas, 1)

    def clear(self) -> None:
        self.canvas.raw_image = np.zeros(0)
        self.canvas.displayed_image = np.zeros(0)
        self.canvas._level_cache = None
        self.canvas._level_scan_token = None
        self.image.clear()

    def render(
        self,
        data: np.ndarray,
        *,
        x_axis: AxisProjection | None = None,
        y_axis: AxisProjection | None = None,
        color_map: str = "viridis",
        log_scale: bool = False,
        level_scan_token: object | None = None,
    ) -> None:
        source = np.asarray(data)
        image = source.T
        if x_axis is not None and y_axis is not None:
            x0, x1 = axis_extent(x_axis.values)
            y0, y1 = axis_extent(y_axis.values)
            rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
            linear_percentiles = (0.5, 99.5)
            expand_degenerate = True
        else:
            height, width = source.shape
            # Detector data are stored row-major from the lower-left detector
            # origin.  The canonical raw view transposes to pyqtgraph's
            # column-major image order and mirrors detector Y exactly once.
            image = image[:, ::-1]
            rect = QtCore.QRectF(0.0, 0.0, float(width), float(height))
            linear_percentiles = (2.0, 98.0)
            expand_degenerate = False
        self.canvas.setImage(
            image,
            rect=rect,
            scale="Log" if log_scale else "Linear",
            cmap=color_map,
            linear_percentiles=linear_percentiles,
            expand_degenerate_levels=expand_degenerate,
            level_scan_token=level_scan_token,
        )
        self.canvas.imageViewBox.setRange(rect, padding=0.0)
        if x_axis is not None and y_axis is not None:
            x_label, x_unit = _axis_presentation(
                x_axis.label,
                x_axis.unit,
            )
            y_label, y_unit = _axis_presentation(
                y_axis.label,
                y_axis.unit,
            )
            self.plot.setLabel(
                "bottom",
                x_label,
                units=x_unit,
            )
            self.plot.setLabel(
                "left",
                y_label,
                units=y_unit,
            )
        else:
            # Do not pass ``units=`` here: AxisItem interprets it as an SI
            # quantity and silently relabels a 3020-pixel detector as
            # ``3.02 kpixel``.  Raw axes are literal detector indices.
            for side, label in (
                ("bottom", "x (Pixels)"),
                ("left", "y (Pixels)"),
            ):
                axis = self.plot.getAxis(side)
                axis.setLabel(label)
                axis.enableAutoSIPrefix(False)
                axis.setScale(1.0)
                axis.autoSIPrefixScale = 1.0
                axis.labelUnitPrefix = ""
                axis.updateAutoSIPrefix()


class CompactFrameSelector(QtWidgets.QComboBox):
    """A five-glyph frame field sized by the active font and style."""

    display_alignment = QtCore.Qt.AlignmentFlag.AlignCenter

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._exact_frames: list[DisplayFrameKey] = []
        self._apply_metric_width()

    def add_frame(
        self,
        caption: str,
        frame: DisplayFrameKey,
        tooltip: str,
    ) -> None:
        super().addItem(caption)
        self._exact_frames.append(frame)
        super().setItemData(
            self.count() - 1,
            tooltip,
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )

    def clear(self) -> None:
        super().clear()
        if hasattr(self, "_exact_frames"):
            self._exact_frames.clear()

    def itemData(
        self,
        index: int,
        role: int = QtCore.Qt.ItemDataRole.UserRole,
    ):
        if role == QtCore.Qt.ItemDataRole.UserRole:
            if 0 <= index < len(self._exact_frames):
                return self._exact_frames[index]
            return None
        return super().itemData(index, role)

    def currentData(
        self,
        role: int = QtCore.Qt.ItemDataRole.UserRole,
    ):
        return self.itemData(self.currentIndex(), role)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in {
            QtCore.QEvent.Type.FontChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._apply_metric_width()

    def paintEvent(self, _event) -> None:
        """Paint the closed value centrally while retaining native combo chrome."""

        option = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(option)
        painter = QtWidgets.QStylePainter(self)
        painter.drawComplexControl(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox,
            option,
        )
        edit_rect = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox,
            option,
            QtWidgets.QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        color_group = (
            option.palette.currentColorGroup()
            if option.state
            & QtWidgets.QStyle.StateFlag.State_Enabled
            else QtGui.QPalette.ColorGroup.Disabled
        )
        painter.setPen(
            option.palette.color(
                color_group,
                QtGui.QPalette.ColorRole.Text,
            )
        )
        painter.drawText(
            edit_rect,
            self.display_alignment,
            option.currentText,
        )

    def _apply_metric_width(self) -> None:
        option = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(option)
        content = QtCore.QSize(
            self.fontMetrics().horizontalAdvance("00000"),
            self.fontMetrics().height(),
        )
        width = self.style().sizeFromContents(
            QtWidgets.QStyle.ContentsType.CT_ComboBox,
            option,
            content,
            self,
        ).width()
        self.setFixedWidth(width)


class ContentFitComboBox(QtWidgets.QComboBox):
    """A non-eliding combo sized from its current font, style, and items."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setItemDelegate(QtWidgets.QStyledItemDelegate(self))
        self.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._apply_metric_width()

    def addItem(self, *args) -> None:
        super().addItem(*args)
        self._apply_metric_width()

    def addItems(self, texts) -> None:
        super().addItems(texts)
        self._apply_metric_width()

    def insertItem(self, *args) -> None:
        super().insertItem(*args)
        self._apply_metric_width()

    def insertItems(self, index: int, texts) -> None:
        super().insertItems(index, texts)
        self._apply_metric_width()

    def clear(self) -> None:
        super().clear()
        self._apply_metric_width()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in {
            QtCore.QEvent.Type.ApplicationFontChange,
            QtCore.QEvent.Type.FontChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._apply_metric_width()

    def showPopup(self) -> None:
        self._apply_metric_width()
        super().showPopup()

    def _apply_metric_width(self) -> None:
        view = self.view()
        view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        metrics = self.fontMetrics()
        widest = max(
            (
                metrics.horizontalAdvance(self.itemText(index))
                for index in range(self.count())
            ),
            default=metrics.horizontalAdvance(""),
        )
        option = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(option)
        content = QtCore.QSize(widest, metrics.height())
        closed_width = self.style().sizeFromContents(
            QtWidgets.QStyle.ContentsType.CT_ComboBox,
            option,
            content,
            self,
        ).width()
        self.setMinimumWidth(max(1, closed_width))

        style = self.style()
        popup_chrome = (
            style.pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent,
                None,
                view,
            )
            + 2
            * style.pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_DefaultFrameWidth,
                None,
                view,
            )
            + 2
            * style.pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_FocusFrameHMargin,
                None,
                view,
            )
        )
        view.setMinimumWidth(max(closed_width, widest + popup_chrome))


def project_header_projection(
    state: ControlPanelRenderState,
) -> SectionHeaderProjection:
    """Qualify PROJECT from the mounted intent's directory facts."""

    values = _bound_values(state, "project")
    ready, detail = project_directories_ready(
        values.get(("Project", "project_folder")),
        values.get(("Project", "h5_dir")),
    )
    return SectionHeaderProjection(
        "",
        ready,
        detail,
    )


def experiment_header_projection(
    state: ControlPanelRenderState,
) -> SectionHeaderProjection:
    """Qualify EXPERIMENT from the production parsed-PONI fact."""

    values = _bound_values(state, "experiment")
    text = "grazing" if values.get(("GI", "Grazing")) is True else "standard"
    raw_poni = values.get(("Signal", "poni_file"))
    poni_file = raw_poni if type(raw_poni) is str else ""
    ready = detector_calibration_ready(poni_file)
    return SectionHeaderProjection(
        text,
        ready,
        (
            "Detector calibration is readable and parseable."
            if ready
            else "Choose a readable, parseable PONI calibration."
        ),
    )


def source_header_projection(
    observation: SourceObservation,
) -> SectionHeaderProjection:
    """Project only readiness proven by the accepted source observation."""

    if type(observation) is not SourceObservation:
        raise TypeError("source header requires an exact SourceObservation")
    if observation.status is not SourceObservationStatus.AVAILABLE:
        return SectionHeaderProjection(
            observation.status.value,
            False,
            observation.reason,
        )

    source = observation.source
    is_directory_source = type(source) is DirectorySourceSpec
    is_single_image = (
        type(source) is SourceSpec
        and (
            source.kind is SourceKind.IMAGE_FILE
            or is_single_image_spec(source)
        )
    )
    is_image_series = (
        type(source) is SourceSpec
        and not is_single_image
        and source.kind in {
            SourceKind.TIFF_SERIES,
            SourceKind.NEXUS_STACK,
            SourceKind.EIGER_MASTER,
            SourceKind.PROCESSED_NEXUS,
        }
    )
    if is_directory_source:
        kind = "Image Directory"
    elif is_image_series:
        kind = "Image Series"
    else:
        kind = "Single Image"

    parts: list[str] = []
    direct_count = observation.direct_child_count
    count = observation.observed_file_count
    if count is not None:
        unit = (
            "frame" if count == 1 else "frames"
        ) if is_image_series else (
            "file" if count == 1 else "files"
        )
        count_text = f"{count} {unit}"
        if (
            observation.file_count_scope
            is SourceCountScope.SELECTED_PLUS_IMMEDIATE
        ):
            count_text += " (folder + 1 level)"
        parts.append(count_text)
    parts.append(kind)
    frozen_series_ready = bool(
        type(source) is SourceSpec
        and source.kind is SourceKind.TIFF_SERIES
        and observation.exists
        and direct_count is not None
        and direct_count > 0
        and observation.candidate_fingerprint
    )
    directory_ready = bool(
        is_directory_source
        and observation.exists
        and observation.is_directory
        and count is not None
        and count > 0
        and observation.candidate_fingerprint
    )
    explicit_file_ready = bool(
        type(source) is SourceSpec
        and source.kind is not SourceKind.TIFF_SERIES
        and observation.exists
        and not observation.is_directory
        and observation.candidate_fingerprint
    )
    ready = frozen_series_ready or directory_ready or explicit_file_ready
    if ready:
        if frozen_series_ready:
            detail = (
                "The selected source image is present."
                if is_single_image
                else "Every frozen image-series member is present."
            )
        elif directory_ready:
            detail = (
                "The source directory and at least one matching candidate "
                "were observed; content is qualified just in time."
            )
        else:
            detail = "The selected source file is present."
    elif count:
        detail = (
            "Matching names are present; content readiness has not "
            "been observed."
        )
    else:
        detail = "The source exists, but no matching frame was observed."
    if (
        observation.file_count_scope
        is SourceCountScope.SELECTED_PLUS_IMMEDIATE
    ):
        detail = (
            f"{detail} Only the selected folder and immediate subfolders are "
            "processed. Deeper subfolders are outside the supported Run scope."
        )
    elif observation.subdirectories_deferred:
        detail = (
            f"{detail} Run is limited to immediate subfolders."
        )
    return SectionHeaderProjection(" · ".join(parts), ready, detail)


def processing_header_projection(
    state: ControlPanelRenderState,
) -> SectionHeaderProjection:
    """Qualify typed processing fields without borrowing whole-run readiness."""

    if type(state) is not ControlPanelRenderState:
        raise TypeError(
            "processing header requires an exact ControlPanelRenderState"
        )
    page = state.profile.processing_page
    text = str(getattr(page, "value", page)).replace("_", " ")
    bound = state.bound_controls
    if bound is None or page not in {
        ProcessingPage.INT_1D,
        ProcessingPage.INT_2D,
    }:
        return SectionHeaderProjection(text)

    values = {field.path: field.value for field in bound.fields}
    required = [
        (("Int1D", "axis"), _nonempty_text),
        (("Int1D", "points"), _positive_int),
    ]
    range_roots = [("Int1D", "radial"), ("Int1D", "azim")]
    if page is ProcessingPage.INT_2D:
        required.extend((
            (("Int2D", "axis"), _nonempty_text),
            (("Int2D", "radial_points"), _positive_int),
            (("Int2D", "azim_points"), _positive_int),
        ))
        range_roots.extend((
            ("Int2D", "radial"),
            ("Int2D", "azim"),
        ))
    ready = all(
        path in values and predicate(values[path])
        for path, predicate in required
    ) and all(
        _range_is_configured(values, root, stem)
        for root, stem in range_roots
    ) and _threshold_is_configured(values)
    return SectionHeaderProjection(
        text,
        ready,
        "Typed processing inputs are configured." if ready else "",
    )


def apply_vnext_controls_readiness(
    panel: QtWidgets.QWidget,
    projection: ControlsReadinessProjection,
) -> None:
    """Apply already-qualified markers after the shared panel reconciles."""

    if type(projection) is not ControlsReadinessProjection:
        raise TypeError(
            "Controls readiness must be an exact immutable projection"
        )

    project = getattr(panel, "project_card", None)
    if project is not None:
        _apply_section_header(project, projection.project)
    experiment = getattr(panel, "experiment_card", None)
    if experiment is not None:
        _apply_section_header(experiment, projection.experiment)
    processing = getattr(panel, "processing_card", None)
    if processing is not None:
        _apply_section_header(processing, projection.processing)
    source = panel.findChild(
        QtWidgets.QWidget,
        "scatteringSourceStatusView",
    )
    reapply = getattr(source, "reapply_header", None)
    if callable(reapply):
        reapply()


def apply_section_header(
    card: QtWidgets.QWidget,
    projection: SectionHeaderProjection,
) -> None:
    """Public adapter used by the asynchronous source-status view."""

    _apply_section_header(card, projection)


def _apply_section_header(
    card: QtWidgets.QWidget,
    projection: SectionHeaderProjection,
) -> None:
    card.set_status_text(projection.text)
    card.status.setToolTip(projection.detail)
    card.set_valid_marker(projection.ready, projection.detail)


def _bound_values(
    state: ControlPanelRenderState,
    owner: str,
) -> dict[tuple[str, ...], object]:
    if type(state) is not ControlPanelRenderState:
        raise TypeError(
            f"{owner} header requires an exact ControlPanelRenderState"
        )
    bound = state.bound_controls
    return (
        {}
        if bound is None
        else {field.path: field.value for field in bound.fields}
    )


def _nonempty_text(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def _positive_int(value: object) -> bool:
    if type(value) is bool:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0 and number.is_integer()


def _finite(value: object) -> float | None:
    if type(value) is bool:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _range_is_configured(
    values: dict[tuple[str, ...], object],
    root: str,
    stem: str,
) -> bool:
    auto_path = (root, f"{stem}_auto")
    low_path = (root, f"{stem}_low")
    high_path = (root, f"{stem}_high")
    if auto_path in values and values[auto_path] is True:
        return True
    if low_path not in values or high_path not in values:
        return False
    low = _finite(values[low_path])
    high = _finite(values[high_path])
    return low is not None and high is not None and low <= high


def _threshold_is_configured(
    values: dict[tuple[str, ...], object],
) -> bool:
    if values.get(("Mask", "Threshold")) is not True:
        return True
    low = _finite(values.get(("Mask", "min")))
    high = _finite(values.get(("Mask", "max")))
    return low is not None and high is not None and low <= high


def scrollable_toolbar(
    row: QtWidgets.QLayout,
    *,
    minimum_width: int,
    height: int = 42,
    horizontal_inset: int = 0,
) -> QtWidgets.QScrollArea:
    # QScrollArea constrains the viewport to ``height``.  Default layout
    # margins make the content six pixels taller than that viewport, clipping
    # the lower margin and making otherwise equal-height controls appear
    # bottom-justified.  Zero margins leave the fixed-height widgets centered
    # by QBoxLayout inside the exact viewport.
    row.setContentsMargins(
        horizontal_inset,
        0,
        horizontal_inset,
        0,
    )
    content = QtWidgets.QWidget()
    content.setLayout(row)
    content.setMinimumWidth(minimum_width)
    area = QtWidgets.QScrollArea()
    area.setObjectName("e3ScrollableToolbar")
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setVerticalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    area.setHorizontalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    area.setFixedHeight(height)
    area.setWidget(content)
    return area


def spin(
    low: float, high: float, value: float
) -> QtWidgets.QDoubleSpinBox:
    widget = QtWidgets.QDoubleSpinBox()
    widget.setRange(low, high)
    widget.setValue(value)
    widget.setDecimals(2)
    widget.setMaximumWidth(72)
    return widget


def set_combo(
    combo: QtWidgets.QComboBox,
    values: tuple[str, ...],
    selected: str,
) -> None:
    combo.clear()
    combo.addItems(values)
    set_combo_value(combo, selected)


def set_combo_value(
    combo: QtWidgets.QComboBox,
    value: str,
    *,
    fallback: str | None = None,
) -> str:
    index = combo.findText(value) if type(value) is str else -1
    if index < 0 and fallback is not None:
        index = combo.findText(fallback)
    if index >= 0:
        combo.setCurrentIndex(index)
    return combo.currentText()


def axis_extent(values: np.ndarray) -> tuple[float, float]:
    first, last = float(values[0]), float(values[-1])
    if first == last:
        return first - 0.5, last + 0.5
    return min(first, last), max(first, last)


def repeated_labels(
    frames: tuple[DisplayFrameKey, ...],
) -> frozenset[int]:
    counts: dict[int, int] = {}
    for frame in frames:
        label = frame.local_frame_label
        counts[label] = counts.get(label, 0) + 1
    return frozenset(label for label, count in counts.items() if count > 1)


def frame_caption(
    frame: DisplayFrameKey,
    repeated: frozenset[int],
    *,
    position: int | None = None,
) -> str:
    del repeated
    if position is not None:
        return str(position + 1)
    return str(frame.local_frame_label)


def aggregate_traces(
    traces: tuple[TraceProjection, ...], mode: str
) -> tuple[TraceProjection, ...]:
    if not traces:
        return ()
    first = traces[0]
    if not all(
        trace.axis.values.shape == first.axis.values.shape
        and np.array_equal(trace.axis.values, first.axis.values)
        for trace in traces
    ):
        return (first,)
    values = np.stack([trace.intensity for trace in traces])
    intensity = (
        np.nanmean(values, axis=0)
        if mode == "Average"
        else np.nansum(values, axis=0)
    )
    return (
        TraceProjection(first.frame, first.axis, intensity, mode),
    )


__all__ = [
    "apply_section_header",
    "apply_vnext_controls_readiness",
    "axis_extent",
    "aggregate_traces",
    "ContentFitComboBox",
    "experiment_header_projection",
    "frame_caption",
    "processing_header_projection",
    "project_header_projection",
    "repeated_labels",
    "ScientificImagePane",
    "SectionHeaderProjection",
    "scrollable_toolbar",
    "set_combo",
    "set_combo_value",
    "source_header_projection",
    "spin",
]
