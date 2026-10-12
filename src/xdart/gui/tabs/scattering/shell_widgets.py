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
    ControlsProjection,
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
from .scientific_curves import add_curve_item, add_text_item, reference_entry
from xdart.gui.widgets.scientific_plot import HOVER_PX, HoverTooltip


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


class _OverlayHitIndex:
    """One joined overlay family's drawn points (the item's own array, not a copy) bucketed in a uniform grid over
    their view-coordinate extent, so a hover finds the curves within ``HOVER_PX`` without scanning every point (C7,
    CR7). Built once per drawing; a zoom changes only the query's reach. Memory: two int32 entries per point."""

    CELLS = 256

    def __init__(self, points, values, color):
        self.points, self.values, self.color = points, values, color
        finite = np.isfinite(points).all(axis=1)
        #: the separator rows: curve k ends before ``breaks[k]`` (one separator follows each joined curve)
        self.breaks = np.flatnonzero(~finite)
        index = np.flatnonzero(finite)
        self.empty = index.size == 0
        if self.empty:
            return
        xy = points[index]
        self.origin = xy.min(axis=0)
        self.cell = np.maximum((xy.max(axis=0) - self.origin) / self.CELLS, 1e-12)
        joined = np.diff(index) == 1  # consecutive drawn points of one curve
        half = np.abs(np.diff(xy, axis=0))[joined] / 2
        #: half the longest segment, per axis: a segment can pass the cursor with both ends this far away
        self.reach = half.max(axis=0) if half.size else np.zeros(2)
        cells = self._cells(xy)
        order = np.argsort(cells, kind='stable')
        self.cells, self.order = cells[order].astype(np.int32), index[order].astype(np.int32)

    def _cells(self, xy):
        ij = np.clip(np.floor((xy - self.origin) / self.cell), 0, self.CELLS).astype(np.int64)
        return ij[..., 1] * (self.CELLS + 1) + ij[..., 0]

    def hits(self, x, y, pixel):
        """The curves (indices into ``values``) passing within ``HOVER_PX`` screen pixels of view (x, y)."""
        if self.empty:
            return ()
        scale = np.abs(np.asarray(pixel, dtype=float))
        reach = HOVER_PX * scale + self.reach
        low, high = (np.array((x, y)) - reach - self.origin) / self.cell, (np.array((x, y)) + reach - self.origin) / self.cell
        if (high < 0).any() or (low > self.CELLS).any():
            return ()
        (i0, j0), (i1, j1) = np.clip(np.floor(low), 0, self.CELLS).astype(int), np.clip(np.floor(high), 0, self.CELLS).astype(int)
        rows = [self.order[np.searchsorted(self.cells, j * (self.CELLS + 1) + i0):
                           np.searchsorted(self.cells, j * (self.CELLS + 1) + i1, side='right')]
                for j in range(j0, j1 + 1)]
        near = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int32)
        if near.size == 0:
            return ()
        points, last = self.points, len(self.points) - 1
        starts = np.unique(np.concatenate((near, near - 1)))
        starts = starts[(starts >= 0) & (starts < last)]
        ends = starts + 1
        segments = starts[np.isfinite(points[starts]).all(axis=1) & np.isfinite(points[ends]).all(axis=1)]
        cursor = np.array((x, y)) / scale
        a, b = points[segments] / scale, points[segments + 1] / scale
        d = b - a
        t = np.clip(((cursor - a) * d).sum(axis=1) / np.maximum((d * d).sum(axis=1), 1e-300), 0., 1.)
        hit = np.hypot(*(a + t[:, None] * d - cursor).T) <= HOVER_PX
        single = near[~np.isin(near, np.concatenate((segments, segments + 1)))]
        alone = single[np.hypot(*((points[single] / scale) - cursor).T) <= HOVER_PX] if single.size else single
        curves = np.searchsorted(self.breaks, np.concatenate((segments[hit], alone)))
        return tuple(int(curve) for curve in np.unique(curves) if curve < len(self.values))


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
        self.coverage_overlay = pg.ImageItem(axisOrder="col-major")
        self.coverage_overlay.setZValue(1)
        self.coverage_overlay.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.coverage_overlay.setAcceptHoverEvents(False)
        self.plot.addItem(self.coverage_overlay)
        self.coverage_overlay.hide()
        self._render_contract: tuple[object, ...] | None = None
        self._geometry_contract: tuple[object, ...] | None = None
        self._axis_presentation_contract: tuple[object, ...] | None = None
        self.overlay_families = {}
        self._overlay_contracts = {}
        #: hoverable families (curves carrying values: reference phases' rings and lines, C7): their drawn points,
        #: the curves' values and colour; each index is built on the first hover after its drawing
        self._hover_data, self._hover_indexes = {}, {}
        self.hover = HoverTooltip(self.canvas.image_win)
        self._hover_proxy = pg.SignalProxy(signal=self.canvas.image_win.scene().sigMouseMoved, rateLimit=60,
                                           slot=self._hover_moved)
        layout.addWidget(self.canvas, 1)

    def clear(self) -> None:
        self.clear_overlay_families()
        self._render_contract = None
        self._geometry_contract = None
        self._axis_presentation_contract = None
        self.canvas.raw_image = np.zeros(0)
        self.canvas.displayed_image = np.zeros(0)
        self.canvas._level_cache = None
        self.canvas._level_scan_token = None
        self.image.clear()
        self.render_coverage(None)

    def clear_overlay_families(self):
        for family in tuple(self.overlay_families):
            self.render_overlay_family(family, ())

    def _map_overlay_points(self, points, native_shape):
        if native_shape is None:
            return np.asarray(points)
        c, r = self.image.image.shape[:2]
        rows, columns = native_shape
        native = np.asarray(points)
        x = native[:, 0]*(c-1)/(columns-1)+.5 if columns > 1 else np.full(len(native), .5)
        y = r-.5-native[:, 1]*(r-1)/(rows-1) if rows > 1 else np.full(len(native), .5)
        transform, pos = self.image.transform(), self.image.pos()
        return np.column_stack((
            transform.m11()*x+transform.m21()*y+transform.dx()+pos.x(),
            transform.m12()*x+transform.m22()*y+transform.dy()+pos.y()))

    def render_overlay_family(self, family, curves, *, texts=(), native_shape=None,
                              color=None, joined=False):
        """Independent passive family using one native-pixel display mapper."""
        contract = None
        if curves and (native_shape is None or self.image.image is not None):
            transform, pos = self.image.transform(), self.image.pos()
            contract = (tuple(id(curve) for curve in curves), tuple(id(text) for text in texts),
                        native_shape, None if self.image.image is None else self.image.image.shape,
                        transform.m11(), transform.m12(), transform.m21(), transform.m22(),
                        transform.dx(), transform.dy(), pos.x(), pos.y(), color, joined)
        if contract == self._overlay_contracts.get(family):
            return
        for item in self.overlay_families.pop(family, ()):
            self.plot.removeItem(item)
        self._overlay_contracts.pop(family, None)
        self._hover_data.pop(family, None)
        self._hover_indexes.pop(family, None)
        self.hover.hide()  # what it named may be gone or moved
        if contract is None:
            return
        self._overlay_contracts[family] = contract
        color = color or '#8dd3ec'
        mapped = [self._map_overlay_points(curve.xy, native_shape) for curve in curves]
        pairs = [(curves[0], np.vstack([np.vstack((points, (np.nan, np.nan))) for points in mapped]))]
        values = tuple(getattr(curve, 'value', None) for curve in curves)
        if joined and any(value is not None for value in values):
            self._hover_data[family] = (pairs[0][1], values, pg.mkColor(color).getRgb()[:3])
        if not joined:
            pairs = zip(curves, mapped)
        items = [add_curve_item(self.plot, curve, color, xy=points) for curve, points in pairs]
        items.extend(add_text_item(self.plot, text, color,
                     xy=self._map_overlay_points([text.xy], native_shape)[0]) for text in texts)
        for item in items:
            item.setZValue(2)
        self.overlay_families[family] = tuple(items)

    def _hover_moved(self, positions):
        """The overlay curves within ``HOVER_PX`` of the cursor name their reflections in one tooltip (C7)."""
        position = positions[0] if type(positions) is tuple and positions else positions
        box = self.canvas.imageViewBox
        if (not isinstance(position, QtCore.QPointF) or not self._hover_data
                or not box.sceneBoundingRect().contains(position)):
            self.hover.hide()
            return
        point, pixel = box.mapSceneToView(position), box.viewPixelSize()
        entries = []
        for family, (points, values, color) in self._hover_data.items():
            index = self._hover_indexes.get(family)
            if index is None:
                index = self._hover_indexes[family] = _OverlayHitIndex(points, values, color)
            entries.extend(reference_entry(values[curve], color) for curve in index.hits(point.x(), point.y(), pixel)
                           if values[curve] is not None)
        if entries:
            self.hover.show(position, "<br>".join(entries))
        else:
            self.hover.hide()

    def render_coverage(self, coverage: np.ndarray | None) -> None:
        """Shade zero-support bins separately from the scientific image and its color scale."""
        if coverage is None:
            self.coverage_overlay.clear()
            # ImageItem.clear drops pixels, but its rendered QImage/buffers can still own them.
            for name in ("qimage", "_displayBuffer", "_processingBuffer", "_imageNanLocations", "_imageHasNans"):
                setattr(self.coverage_overlay, name, None)
            self.coverage_overlay.hide()
            return
        if self.image.image is None or coverage.T.shape != self.image.image.shape[:2]:
            raise ValueError("coverage must match the rendered image")
        rgba = np.zeros((*coverage.T.shape, 4), dtype=np.uint8)
        rgba[..., :3] = 128
        rgba[..., 3] = (coverage.T == 0) * 180
        self.coverage_overlay.setImage(rgba, autoLevels=False)
        self.coverage_overlay.setTransform(self.image.transform())
        self.coverage_overlay.setPos(self.image.pos())
        self.coverage_overlay.show()

    def clear_image_buffers(self, *, keep_chrome: bool = True) -> None:
        """Release pyqtgraph image state while leaving failures observable."""
        errors: list[Exception] = []

        def clear_member(target, name, value=None, *, call=False) -> None:
            try:
                if call:
                    member = getattr(target, name)
                    if value is None:
                        member()
                    else:
                        member(value)
                else:
                    setattr(target, name, value)
            except Exception as error:
                errors.append(error)

        clear_member(self, "clear", call=True)
        # Ordinary source/absence transitions historically scrubbed only the
        # image buffers.  Retain their ViewBox, histogram and geometry state;
        # full viewer retirement below explicitly requests a chrome reset.
        if keep_chrome:
            for name in (
                "qimage", "levels", "_defferedLevels", "_displayBuffer",
                "_processingBuffer", "_imageNanLocations", "_imageHasNans",
            ):
                clear_member(self.image, name, None)
            if errors:
                raise errors[0]
            return
        canvas = self.canvas
        image = self.image
        histogram = self.color_scale
        try:
            hover = image.pos_label
        except Exception as error:
            errors.append(error)
        else:
            clear_member(hover, "setText", "", call=True)
        for target, name, value in (
            (canvas, "raw_image", np.zeros(0)),
            (canvas, "displayed_image", np.zeros(0)),
            (canvas, "_level_cache", None),
            (canvas, "_level_scan_token", None),
            (histogram, "lo_lim", None),
            (histogram, "hi_lim", None),
            (image, "image", None),
            (image, "qimage", None),
            (image, "levels", None),
            (image, "_defferedLevels", None),
            (image, "_lastDownsample", (1, 1)),
            (image, "_displayBuffer", None),
            (image, "_processingBuffer", None),
            (image, "_imageNanLocations", None),
            (image, "_imageHasNans", None),
        ):
            clear_member(target, name, value)
        if not keep_chrome:
            clear_member(histogram, "setLevels", (0.0, 1.0), call=True)
            clear_member(
                canvas.imageViewBox,
                "setRange",
                QtCore.QRectF(0.0, 0.0, 1.0, 1.0),
                call=True,
            )
        for method in (
            "resetTransform", "prepareGeometryChange", "informViewBoundsChanged",
            "update",
        ):
            if keep_chrome and method == "resetTransform":
                continue
            clear_member(image, method, call=True)
        if errors:
            raise errors[0]

    @staticmethod
    def _axis_render_contract(
        axis: AxisProjection | None,
    ) -> tuple[object, ...] | None:
        return (
            None
            if axis is None
            else (
                id(axis.values), axis.label, axis.unit, axis.values.shape, axis.edges,
            )
        )

    @staticmethod
    def _image_render_contract(
        data: np.ndarray,
        *,
        x_axis: AxisProjection | None,
        y_axis: AxisProjection | None,
        detector_shape: tuple[int, int] | None,
        color_map: str,
        log_scale: bool,
        autorange_finite_y: bool = False,
    ) -> tuple[object, ...]:
        source = np.asarray(data)
        return (
            id(source), source.shape, source.dtype.str,
            ScientificImagePane._axis_render_contract(x_axis),
            ScientificImagePane._axis_render_contract(y_axis),
            detector_shape, color_map, bool(log_scale), autorange_finite_y,
        )

    @staticmethod
    def _image_geometry_contract(
        data: np.ndarray,
        *,
        x_axis: AxisProjection | None,
        y_axis: AxisProjection | None,
        detector_shape: tuple[int, int] | None,
    ) -> tuple[object, ...]:
        source = np.asarray(data)
        if x_axis is not None and y_axis is not None:
            x0, x1 = axis_span(x_axis)
            y0, y1 = axis_span(y_axis)
            return (
                "axes", source.shape, x_axis.values.shape, y_axis.values.shape,
                x0, x1, y0, y1,
            )
        return ("detector", source.shape, detector_shape or source.shape)

    @staticmethod
    def _axis_presentation_key(
        x_axis: AxisProjection | None,
        y_axis: AxisProjection | None,
    ) -> tuple[object, ...]:
        if x_axis is None or y_axis is None:
            return ("raw",)
        return (
            "axes",
            *_axis_presentation(x_axis.label, x_axis.unit),
            *_axis_presentation(y_axis.label, y_axis.unit),
        )

    def render_matches(
        self,
        data: np.ndarray,
        *,
        x_axis: AxisProjection | None = None,
        y_axis: AxisProjection | None = None,
        detector_shape: tuple[int, int] | None = None,
        color_map: str = "viridis",
        log_scale: bool = False,
        autorange_finite_y: bool = False,
    ) -> bool:
        source = np.asarray(data)
        axes = tuple(
            np.asarray(axis.values)
            for axis in (x_axis, y_axis)
            if axis is not None
        )
        if source.flags.writeable or any(axis.flags.writeable for axis in axes):
            return False
        return self._render_contract == self._image_render_contract(
            source,
            x_axis=x_axis,
            y_axis=y_axis,
            detector_shape=detector_shape,
            color_map=color_map,
            log_scale=log_scale,
            autorange_finite_y=autorange_finite_y,
        )

    def render(self, *args, **kwargs) -> bool:
        try:
            return self._render_image(*args, **kwargs)
        except Exception:
            self.clear_overlay_families()
            raise

    def _render_image(
        self,
        data: np.ndarray,
        *,
        x_axis: AxisProjection | None = None,
        y_axis: AxisProjection | None = None,
        detector_shape: tuple[int, int] | None = None,
        color_map: str = "viridis",
        log_scale: bool = False,
        level_scan_token: object | None = None,
        view_range: QtCore.QRectF | None = None,
        autorange_finite_y: bool = False,
    ) -> bool:
        """Draw ``data``; False when the identical immutable image is already drawn (only ``view_range`` is
        applied), True when it is drawn again: its colour bar's levels and limits are then the image's own, unless
        it has no finite value (the canvas leaves the colour bar as it was)."""
        source = np.asarray(data)
        render_contract = ScientificImagePane._image_render_contract(
            source,
            x_axis=x_axis,
            y_axis=y_axis,
            detector_shape=detector_shape,
            color_map=color_map,
            log_scale=log_scale,
            autorange_finite_y=autorange_finite_y,
        )
        geometry_contract = ScientificImagePane._image_geometry_contract(
            source,
            x_axis=x_axis,
            y_axis=y_axis,
            detector_shape=detector_shape,
        ) + (autorange_finite_y,)
        axis_presentation = ScientificImagePane._axis_presentation_key(
            x_axis, y_axis,
        )
        # FrameView arrays and axes are immutable.  Plot-mode, selection, and
        # status-only reconciliations therefore must not resend identical
        # detector/cake pixels through pyqtgraph's setImage path.
        axes = tuple(
            np.asarray(axis.values)
            for axis in (x_axis, y_axis)
            if axis is not None
        )
        immutable = (
            not source.flags.writeable
            and not any(axis.flags.writeable for axis in axes)
        )
        if immutable and getattr(self, "_render_contract", None) == render_contract:
            if view_range is not None:
                self.canvas.imageViewBox.setRange(view_range, padding=0.0)
            return False
        # A failed repaint may already have changed pixels, range, or labels.
        # Retire the prior identity contract before the first mutation so a
        # later request for the old immutable source cannot incorrectly fast
        # return against a partially updated canvas.  Publish only after the
        # complete render succeeds below.
        self._render_contract = None
        prior_geometry = self._geometry_contract
        prior_axis_presentation = self._axis_presentation_contract
        self._geometry_contract = None
        self._axis_presentation_contract = None
        image = source.T
        if x_axis is not None and y_axis is not None:
            x0, x1 = axis_span(x_axis)
            y0, y1 = axis_span(y_axis)
            # Explicit bin edges retain the centres' direction without reordering pixels.
            if x_axis.edges is not None and x_axis.values[0] > x_axis.values[-1]:
                x0, x1 = x1, x0
            if y_axis.edges is not None and y_axis.values[0] > y_axis.values[-1]:
                y0, y1 = y1, y0
            rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
            linear_percentiles = (0.5, 99.5)
            expand_degenerate = True
        else:
            height, width = detector_shape or source.shape
            # Detector data are stored row-major from the lower-left detector
            # origin.  The canonical raw view transposes to pyqtgraph's
            # column-major image order and mirrors detector Y exactly once.
            image = image[:, ::-1]
            x0, x1 = (-0.5, 0.5) if width == 1 else (0.0, float(width - 1))
            y0, y1 = (-0.5, 0.5) if height == 1 else (0.0, float(height - 1))
            rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
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
        if (
            view_range is not None
            or prior_geometry != geometry_contract
            or prior_axis_presentation != axis_presentation
        ):
            initial_range = rect
            if autorange_finite_y and y_axis is not None:
                rows = np.flatnonzero(np.isfinite(source).any(axis=1))
                if rows.size:
                    low, high = axis_extent(y_axis.values[rows[[0, -1]]])
                    initial_range = QtCore.QRectF(x0, low, x1 - x0, high - low)
            self.canvas.imageViewBox.setRange(
                initial_range if view_range is None else view_range, padding=0.0,
            )
        if (
            x_axis is not None and y_axis is not None
            and prior_axis_presentation != axis_presentation
        ):
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
        elif prior_axis_presentation != axis_presentation:
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
        # A mutable render cannot safely seed the identity-only reuse cache:
        # a writable alias may mutate the pixels before this array is marked
        # read-only and presented again.  Only immutable inputs publish a
        # reusable contract.
        self._render_contract = render_contract if immutable else None
        self._geometry_contract = geometry_contract
        self._axis_presentation_contract = axis_presentation
        return True


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
    state: ControlsProjection,
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
    state: ControlsProjection,
    *,
    calibration_ready: bool | None = None,
) -> SectionHeaderProjection:
    """Qualify EXPERIMENT from the production parsed-PONI fact."""

    values = _bound_values(state, "experiment")
    text = "grazing" if values.get(("GI", "Grazing")) is True else "standard"
    raw_poni = values.get(("Signal", "poni_file"))
    poni_file = raw_poni if type(raw_poni) is str else ""
    ready = (detector_calibration_ready(poni_file)
             if calibration_ready is None else calibration_ready)
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
    state: ControlsProjection,
) -> SectionHeaderProjection:
    """Qualify typed processing fields without borrowing whole-run readiness."""

    if type(state) is not ControlsProjection:
        raise TypeError(
            "processing header requires an exact ControlsProjection"
        )
    page = state.processing_page
    text = str(getattr(page, "value", page)).replace("_", " ")
    if page not in {
        ProcessingPage.INT_1D,
        ProcessingPage.INT_2D,
    }:
        return SectionHeaderProjection(text)

    values = {field.path: field.value for field in state.fields}
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
    state: ControlsProjection,
    owner: str,
) -> dict[tuple[str, ...], object]:
    if type(state) is not ControlsProjection:
        raise TypeError(
            f"{owner} header requires an exact ControlsProjection"
        )
    return {field.path: field.value for field in state.fields}


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
    upper = values.get(("Mask", "max"))
    # A blank maximum is valid: automatic limits await source admission,
    # and legacy explicit bands may intentionally be open-ended above.
    high = _finite(upper)
    return low is not None and (upper is None or (high is not None and low <= high))


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


def axis_span(axis: AxisProjection) -> tuple[float, float]:
    """Where an image spans along ``axis``: its ``edges`` when given, else its first and last values."""
    return axis.edges if axis.edges is not None else axis_extent(axis.values)


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


#: Curves on different grids are folded on one grid of at most this many points (display only), and at most
#: _FOLD_WORK interpolated values in all; never fewer points than the longest curve's.
_FOLD_GRID_POINTS = 50_000
_FOLD_WORK = 20_000_000


def same_grid(values: np.ndarray, reference: np.ndarray) -> bool:
    """One grid: equal, or equal to rounding (a billionth of each value; live integrators can publish
    numerically equivalent grids a few ulps apart)."""
    return values.shape == reference.shape and (
        np.array_equal(values, reference) or np.allclose(values, reference, rtol=1e-9, atol=0.0))


def _on_one_grid(traces: tuple[TraceProjection, ...]) -> bool:
    """The traces share the first's grid (``same_grid``)."""
    first = traces[0].axis.values
    return all(same_grid(trace.axis.values, first) for trace in traces)


def _fold_grid(traces: tuple[TraceProjection, ...]) -> np.ndarray | None:
    """One grid for curves on different grids of one axis: from the lowest point of any to the highest, as
    fine as the finest (its mean spacing), within ``_FOLD_GRID_POINTS`` and ``_FOLD_WORK``; None when the
    axes differ (Q and 2θ) or one is not finite and strictly monotonic with two points or more."""
    first = traces[0].axis
    lows, highs, steps, sizes = [], [], [], []
    for trace in traces:
        if (trace.axis.label, trace.axis.unit) != (first.label, first.unit):
            return None
        values = np.asarray(trace.axis.values, dtype=float)
        if values.size < 2 or not np.all(np.isfinite(values)):
            return None
        step = np.diff(values)
        if not (np.all(step > 0.0) or np.all(step < 0.0)):
            return None
        lows.append(float(values.min()))
        highs.append(float(values.max()))
        steps.append((highs[-1] - lows[-1]) / (values.size - 1))
        sizes.append(values.size)
    low, high = min(lows), max(highs)
    points = int(math.ceil((high - low) / min(steps))) + 1
    points = max(max(sizes), min(points, _FOLD_GRID_POINTS, _FOLD_WORK // len(traces)))
    return np.linspace(low, high, points)


def foldable(traces: tuple[TraceProjection, ...]) -> bool:
    """Whether Average and Sum fold these traces (one grid, or grids of one axis), else they show the first."""
    return not traces or _on_one_grid(traces) or _fold_grid(traces) is not None


def aggregate_traces(
    traces: tuple[TraceProjection, ...], mode: str
) -> tuple[TraceProjection, ...]:
    """The traces' Average or Sum, for display. On one grid, point by point. On different grids of one axis
    (maintainer, 2026-09-30: "can this be changed and made more flexible, at least for sum? This is only for
    display" -- "bin them together on a common grid"), each is interpolated linearly onto ``_fold_grid``'s
    and every point takes the curves that reach it (none: no value); a curve that alone reaches a stretch
    is shown there as it is. On different axes the first is shown."""
    if not traces:
        return ()
    first = traces[0]
    if _on_one_grid(traces):
        values = np.stack([trace.intensity for trace in traces])
        intensity = (
            np.nanmean(values, axis=0)
            if mode == "Average"
            else np.nansum(values, axis=0)
        )
        return (
            TraceProjection(first.frame, first.axis, intensity, mode),
        )
    grid = _fold_grid(traces)
    if grid is None:
        return (first,)
    total = np.zeros(grid.shape)
    count = np.zeros(grid.shape, dtype=np.int64)
    for trace in traces:  # one curve at a time: the fold holds the grid's arrays only
        values = np.asarray(trace.axis.values, dtype=float)
        intensity = np.asarray(trace.intensity, dtype=float)
        if values[0] > values[-1]:
            values, intensity = values[::-1], intensity[::-1]
        row = np.interp(grid, values, intensity, left=np.nan, right=np.nan)
        reached = ~np.isnan(row)
        total[reached] += row[reached]
        count += reached
    intensity = np.full(grid.shape, np.nan)
    reached = count > 0
    intensity[reached] = total[reached] / count[reached] if mode == "Average" else total[reached]
    return (
        TraceProjection(first.frame, AxisProjection(grid, first.axis.label, first.axis.unit), intensity, mode),
    )


def color_map_choices() -> tuple[str, ...]:
    """Int's colour-map list: "Default" (viridis) first, then every matplotlib map."""
    from matplotlib import colormaps  # lazy: importing this module loads no plotting library

    return ("Default", *tuple(colormaps))


def color_map_combo() -> ContentFitComboBox:
    """Int's colour-map selector, the one its scientific view's top bar shows: built here once for every view
    that draws an intensity (``ScientificView``, the 3D Viewer's ``VolumeView``)."""
    combo = ContentFitComboBox()
    combo.addItems(color_map_choices())
    return combo


def log_toggle() -> QtWidgets.QPushButton:
    """Int's Log toggle, shared as ``color_map_combo`` is."""
    button = QtWidgets.QPushButton("Log")
    button.setCheckable(True)
    return button


__all__ = [
    "apply_section_header",
    "apply_vnext_controls_readiness",
    "axis_extent",
    "axis_span",
    "aggregate_traces",
    "color_map_choices",
    "color_map_combo",
    "foldable",
    "ContentFitComboBox",
    "experiment_header_projection",
    "frame_caption",
    "log_toggle",
    "processing_header_projection",
    "project_header_projection",
    "repeated_labels",
    "same_grid",
    "ScientificImagePane",
    "SectionHeaderProjection",
    "scrollable_toolbar",
    "set_combo",
    "set_combo_value",
    "source_header_projection",
    "spin",
]
