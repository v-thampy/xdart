"""The RSM page's two pop-up tools (AUD-9 stage 2b G9): the "3D Viewer" and "Custom Slice".

The maintainer, 2026-10-01: "A button to bring up a 3d viewer for the entire 3d RSM volume"; "The ability to only
slices which are not just along h, k or l. So say HH and l"; "The two extra features can be implemented as extra
tools that pop up separate windows to keep the scientific view clean for now". Each is a non-modal window that
only mounts shared components (hard rule 14): the 3D Viewer the shared ``VolumeView``; Custom Slice Int's
``ScientificView`` (its keyed layout: the plane over its two projections) beside a ``SectionCard`` of Int's rows. The
page opens them from its view's page tools and keeps them on the result it shows.

Both compute off the GUI thread (item 6; the maintainer: "yes go ahead with both"), each on its own shared
``LatestRequestOwner`` (``widgets/latest_request.py``: a ``SingleWorkerOwner``, the pages' worker owner, and its poll
timer): the latest request wins, and ``shutdown`` closes the worker with the page's other owners.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.pages.operation_owner import OperationCancelled, OperationTerminalStatus
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection, KeyedImageProjection, KeyedPanelsProjection, KeyedTraceProjection, ShellCommandKind,
)
from xdart.gui.tabs.scattering.volume_view import VolumeProjection, VolumeView
from xdart.gui.widgets.controls_panel import FormRow, RangeRow, SectionCard
from xdart.gui.widgets.latest_request import LatestRequestOwner
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from xrd_tools.rsm.custom_slice import CustomSliceRefused, custom_slice, plane_offsets, volume_centre
from xrd_tools.session.display_logic import PanelKey, PanelRole
from xrd_tools.session.readiness import integer_field_refusal
from xrd_tools.session.rsm_viewer_model import rsm_axis_bounds

_LOG = logging.getLogger(__name__)


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

    @property
    def computing(self) -> bool:
        """Whether a draw asked for is still being computed (``VolumeView.computing``)."""
        return self.view.computing

    def release(self) -> None:
        self.view.release()
        self.setWindowTitle("RSM 3D Viewer")

    def shutdown(self):
        """The view's worker closed with the page (``VolumeView.shutdown``)."""
        return self.view.shutdown()

    def hideEvent(self, event) -> None:
        if not event.spontaneous():  # closed (not minimised): the volume and its RGBA go until it is shown again
            self.release()
        super().hideEvent(event)


@dataclass(frozen=True, eq=False)
class _CutRequest:
    """One plane asked for: the result's viewer values (their volume shared, not copied) and every input
    ``custom_slice`` takes, with the plane's offsets for its axes."""

    values: object
    axes: tuple
    through: tuple
    first: tuple
    second: tuple
    ranges: tuple
    points: tuple
    symbols: tuple
    unit: str
    thickness: float
    offsets: tuple

    def matches(self, other) -> bool:
        """The same plane of the same volume."""
        return other is not None and other.values is self.values and self._inputs() == other._inputs()

    def _inputs(self):
        return (self.through, self.first, self.second, self.ranges, self.points, self.symbols, self.unit,
                self.thickness)


def _cut_plane(request: _CutRequest, cancel, _progress):
    """A plane on the window's worker: ``custom_slice``'s result, or its refusal as a value (shown as the window
    shows a refusal). Dropped before and after the cut when a later request supersedes it."""
    if cancel.is_set():
        raise OperationCancelled()
    try:
        cut = custom_slice(request.values.intensity, request.axes, request.through, request.first, request.second,
                           request.ranges[0], request.ranges[1], request.points, symbols=request.symbols,
                           unit=request.unit, thickness=request.thickness)
    except CustomSliceRefused as refused:
        return refused
    if cancel.is_set():
        raise OperationCancelled()
    return cut


_PLANE = PanelKey(PanelRole.SLICE_2D, "custom")
_ALONG_FIRST, _ALONG_SECOND = PanelKey(PanelRole.PROJ_1D, "first"), PanelKey(PanelRole.PROJ_1D, "second")
_MAX_POINTS = 2000  # each way: 2000 × 2000 is custom_slice's MAX_SLICE_POINTS


def _setting(name: str) -> tuple[str, str]:
    return ("custom_slice", name)


def _unit(frame) -> str:
    """The frame's coordinates' unit, as the plane's axes are labelled."""
    return "r.l.u." if frame is RSMCoordinateFrame.HKL else "Å⁻¹"


def _numbers(text: str, what: str) -> tuple[float, float, float]:
    parts = str(text).replace(",", " ").split()
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        values = ()
    if len(values) != 3 or not all(np.isfinite(values)):
        raise CustomSliceRefused("CUSTOM_SLICE_TEXT_INVALID", f"{what} needs three numbers")
    return values


class RSMCustomSliceWindow(QtWidgets.QDialog):
    """Custom Slice: the RSM volume on any plane, HH–L by default (the maintainer: "So say HH and l").

    The plane through ``Through`` along Direction 1 and Direction 2 is cut by ``custom_slice`` (interpolated from
    the gridded volume, as the window says) and drawn on Int's ``ScientificView``: the plane over its two mean
    projections, with Int's colour map and Log. Beside it a ``SectionCard`` of Int's rows, written through
    (``valueChanged``) to this window's settings; a plane ``custom_slice`` refuses is named in the card's
    readiness. The axes and the ranges are coordinates along the directions (``plane_offsets``): H and K on an
    H–K plane, HH and L on HH–L. In Q coordinates the directions and the point are Qx Qy Qz.

    Thickness (the maintainer, 2026-10-01: "Do you want the Thickness option? Yes"), 0 by default, averages the
    slab across ±Thickness/2 along the plane's normal (``custom_slice``'s ``thickness``); the note names its
    planes, and in HKL that the normal and the thickness are r.l.u. ones (H, K and L taken as orthonormal).

    The cut runs on the window's worker (item 6): the rows are read here, where a typed refusal is named at once;
    the latest plane asked for wins, one is computed at a time, and while it is the card says so over the plane
    shown. A plane still computing when the window closes is dropped when it returns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rsmCustomSliceWindow")
        self.setModal(False)
        self.setWindowTitle("RSM Custom Slice")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        self.view = ScientificView()
        self.view.setObjectName("rsmCustomSliceView")
        self.view.set_hidden_tools("norm", "background", "raw", "raw_pane", "slice")
        self.view.commandRequested.connect(self._view_command)
        splitter.addWidget(self.view)
        self.card = SectionCard("Custom Slice", collapsible=False)

        def bounds(name):
            return {"path": _setting(name), "value": "", "placeholder": "Auto"}

        self.rows = {
            "first": FormRow(label="Direction 1", path=_setting("first"), value="1 1 0"),
            "second": FormRow(label="Direction 2", path=_setting("second"), value="0 0 1"),
            "through": FormRow(label="Through", path=_setting("through"), value=""),
            "thickness": FormRow(label="Thickness", path=_setting("thickness"), value="0"),
            "first_range": RangeRow(label="Along 1", low=bounds("first_low"), high=bounds("first_high")),
            "second_range": RangeRow(label="Along 2", low=bounds("second_low"), high=bounds("second_high")),
            "first_points": FormRow(label="Points along 1", path=_setting("first_points"), value="", kind="int",
                                    minimum=2, maximum=_MAX_POINTS, allow_blank=True),
            "second_points": FormRow(label="Points along 2", path=_setting("second_points"), value="", kind="int",
                                     minimum=2, maximum=_MAX_POINTS, allow_blank=True),
        }
        self.rows["through"].editor.setPlaceholderText("Auto: the volume's centre")
        self.rows["thickness"].editor.setPlaceholderText("0: the plane alone")
        self.rows["thickness"].editor.setToolTip(
            "The slab averaged across ±Thickness/2 along the plane's normal (Direction 1 × Direction 2), in the "
            "volume's units; 0 is the plane alone")
        for name in ("first_points", "second_points"):
            self.rows[name].editor.setPlaceholderText("Auto")
        #: the rows' committed text, by path: what the plane is cut from
        self._settings: dict[tuple[str, str], str] = {}
        for row in self.rows.values():
            self.card.add_row(row)
            edits = row.current_edits() if isinstance(row, RangeRow) else ((row.path, row.current_value()),)
            self._settings.update((tuple(path), str(value)) for path, value in edits)
            row.valueChanged.connect(self._edited)
        self.note = QtWidgets.QLabel("")
        self.note.setObjectName("rsmCustomSliceNote")
        self.note.setWordWrap(True)
        self.card.add_row(self.note)
        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.addWidget(self.card)
        side_layout.addStretch(1)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        self._values = self._slice = None
        self._offsets = (0.0, 0.0)
        self._name = ""
        self._color_map, self._log_scale, self._share_axis = "Default", False, True
        # the planes, on the shared latest-request owner (the cut bound late, so a test can hold its result); the
        # plane drawn is its ``shown`` request, so asking for it again cuts nothing
        self._requests = LatestRequestOwner(
            lambda request, cancel, progress: _cut_plane(request, cancel, progress), self._done,
            same=lambda request, other: request.matches(other), parent=self)
        self.resize(1100, 720)

    @property
    def custom(self):
        """The plane drawn now (``custom_slice``'s result, its ``t`` and ``s`` offsets from the point), or None."""
        return self._slice

    @property
    def computing(self) -> bool:
        """Whether a plane asked for is still being computed (the plane shown is the one before it)."""
        return self._requests.computing

    @property
    def _wanted(self):
        """The plane asked for and not yet drawn (the shared owner's), or None."""
        return self._requests.wanted

    @property
    def _running(self):
        """``(identity, request)`` of the plane on the worker (the shared owner's), or None."""
        return self._requests.running

    def show_result(self, name: str, values) -> None:
        """Cut the plane from ``values`` (the page's shown result; its volume is shared, not copied)."""
        self._values, self._name = values, name
        self.setWindowTitle(f"RSM Custom Slice — {name}")
        frame = values.coordinate_frame
        symbols = "h k l" if frame is RSMCoordinateFrame.HKL else " ".join(frame.axis_symbols)
        for key, label in (("first", "Direction 1"), ("second", "Direction 2"), ("through", "Through")):
            self.rows[key].label.setText(f"{label} ({symbols})")
        self.rows["thickness"].label.setText(f"Thickness ({_unit(frame)})")
        centre = volume_centre(axis for _axis_name, axis in values.axes)
        self.rows["through"].editor.setPlaceholderText("Auto: " + " ".join(f"{item:.6g}" for item in centre))
        self.recompute()

    def _edited(self, path, value) -> None:
        text = "" if value is None else str(value)
        if self._settings.get(tuple(path)) == text:
            return  # a blur that commits the same text cuts nothing again
        self._settings[tuple(path)] = text
        self.recompute()

    def recompute(self) -> None:
        """Ask for the plane the rows say (read here: a typed refusal is named at once); the cut runs on the
        window's worker and is drawn when it returns, if it is still the plane asked for."""
        values = self._values
        if values is None:
            return
        frame = values.coordinate_frame
        axes = tuple(np.asarray(axis) for _axis_name, axis in values.axes)
        try:
            first = _numbers(self._settings[_setting("first")], "Direction 1")
            second = _numbers(self._settings[_setting("second")], "Direction 2")
            through_text = self._settings[_setting("through")].strip()
            through = volume_centre(axes) if not through_text else _numbers(through_text, "Through")
            offsets = plane_offsets(through, first, second)
            ranges = [self._range(name, offset) for name, offset in zip(("first", "second"), offsets, strict=True)]
            points = tuple(self._points(name, max(values.intensity.shape)) for name in ("first", "second"))
            request = _CutRequest(values, axes, tuple(through), tuple(first), tuple(second), tuple(ranges), points,
                                  tuple(frame.axis_symbols), _unit(frame), self._thickness(), offsets)
        except CustomSliceRefused as refused:
            self._ask(None)
            self._refuse(refused)
            return
        self._ask(request)  # the plane drawn asked for again cuts nothing (the shared owner's ``shown``)

    def _ask(self, request) -> None:
        """Ask the shared owner for ``request`` (None: none; the latest wins), the card saying a plane is being
        cut over the plane shown, which stays until the new one is drawn."""
        if self._requests.ask(request):
            self.card.set_status_text(_COMPUTING)
        elif self.card.status.text() == _COMPUTING:
            self.card.set_status_text("")

    def _done(self, request, terminal) -> bool:
        """The plane asked for returned (the shared owner's ``done``): drawn, or its refusal or failure named;
        whether it is drawn."""
        if terminal.status is OperationTerminalStatus.RETURNED:
            return self._draw(request, terminal.payload)
        _LOG.warning("Custom Slice failed: %s: %s", terminal.failure_type, terminal.failure_message)
        self._refuse(CustomSliceRefused(
            "CUSTOM_SLICE_FAILED", f"the plane could not be cut ({terminal.failure_type}: "
                                   f"{terminal.failure_message})"))
        return False

    def _draw(self, request, cut) -> bool:
        """Draw a plane the worker cut (or name its refusal, as before the worker); whether it is drawn."""
        if isinstance(cut, CustomSliceRefused):
            self._refuse(cut)
            return False
        try:
            projection = self._projection(cut, request.offsets)
        except CustomSliceRefused as refused:
            self._refuse(refused)
            return False
        frame = request.values.coordinate_frame
        self._slice, self._offsets = cut, request.offsets
        self.card.set_status_text("")
        self.card.set_valid_marker(True, "The plane is drawn")
        self.note.setText(f"The plane is {cut.note}." + (
            " In HKL the normal and the thickness are r.l.u. ones, H, K and L taken as orthonormal: for a "
            "non-orthogonal lattice they are not the reciprocal-space normal and distance."
            if cut.thickness and frame is RSMCoordinateFrame.HKL else ""))
        self.view.reconcile_keyed(projection)
        return True

    def _refuse(self, refused) -> None:
        """A plane refused: the card's readiness and the view say why, and nothing is drawn."""
        self._requests.forget()
        self._slice = None
        self.card.set_status_text(str(refused))
        self.card.set_valid_marker(False)
        self.note.setText("")
        self.view.clear_keyed(str(refused))

    def _range(self, name: str, offset: float):
        """A direction's typed range, as offsets from the point (``custom_slice``'s ``t``); None: Auto."""
        label = self.rows[f"{name}_range"].label.text()
        texts = [self._settings[_setting(f"{name}_{end}")].strip() for end in ("low", "high")]
        if not any(texts):
            return None
        if not all(texts):
            raise CustomSliceRefused("CUSTOM_SLICE_RANGE_INVALID", f"{label} needs both bounds, or neither (Auto)")
        try:
            return tuple(float(text) + offset for text in texts)
        except ValueError:
            raise CustomSliceRefused("CUSTOM_SLICE_RANGE_INVALID", f"{label} needs two numbers") from None

    def _thickness(self) -> float:
        """The typed Thickness, blank 0 (the plane alone); refused unless a finite number, 0 or more."""
        row = self.rows["thickness"]
        text = self._settings[row.path].strip()
        try:
            value = float(text) if text else 0.0
        except ValueError:
            value = float("nan")
        if not (np.isfinite(value) and value >= 0.0):
            raise CustomSliceRefused("CUSTOM_SLICE_TEXT_INVALID", f"{row.label.text()}: a number, 0 or more")
        return value

    def _points(self, name: str, longest: int) -> int:
        row = self.rows[f"{name}_points"]
        text = self._settings[row.path].strip()
        refusal = integer_field_refusal(text, minimum=2, maximum=_MAX_POINTS, allow_blank=True)
        if refusal:
            raise CustomSliceRefused("CUSTOM_SLICE_TEXT_INVALID", f"{row.label.text()}: {refusal}")
        return int(text) if text else longest

    def _projection(self, cut, offsets) -> KeyedPanelsProjection:
        """The plane over its two projections, on the coordinates along the directions (``t − a``, ``s − b``)."""
        along_first, along_second = cut.t - offsets[0], cut.s - offsets[1]
        try:
            edges = rsm_axis_bounds(along_first), rsm_axis_bounds(along_second)
        except ValueError:  # a range too narrow for its points: its coordinates do not increase
            raise CustomSliceRefused("CUSTOM_SLICE_RANGE_INVALID", "a range is too narrow for its points") from None
        image = KeyedImageProjection(
            _PLANE, f"{cut.t_label} – {cut.s_label}", cut.values.T,  # (y, x): the plane's s by t, not copied
            AxisProjection(along_first, cut.t_label, "", edges=edges[0]),
            AxisProjection(along_second, cut.s_label, "", edges=edges[1]), 0, (), navigation=False)
        traces = tuple(KeyedTraceProjection(key, f"{label} mean projection", AxisProjection(axis, label, ""),
                                            np.asarray(projection), "Mean I")
                       for key, label, axis, projection in (
                           (_ALONG_FIRST, cut.t_label, along_first, cut.projection_t),
                           (_ALONG_SECOND, cut.s_label, along_second, cut.projection_s)))
        through = " ".join(f"{item:.6g}" for item in cut.origin)
        return KeyedPanelsProjection(
            ((_PLANE,), (_ALONG_FIRST, _ALONG_SECOND)), (image, *traces), title=self._name,
            status=f"Through {through}; interpolated from the gridded volume", color_map=self._color_map,
            log_scale=self._log_scale, share_axis=self._share_axis)

    def _view_command(self, command) -> None:
        if command.kind is ShellCommandKind.SET_COLOR_MAP:
            self._color_map = str(command.value)
        elif command.kind is ShellCommandKind.SET_LOG_SCALE:
            self._log_scale = bool(command.value)
        elif command.kind is ShellCommandKind.SET_SHARE_AXIS:  # the plane's two directions with their projections
            self._share_axis = bool(command.value)
        else:
            return
        if self._slice is not None:  # the plane drawn again under the new presentation
            self.view.reconcile_keyed(self._projection(self._slice, self._offsets))

    def release(self) -> None:
        """Let go of the volume and the plane (the window closing); the rows keep what was typed. A plane still
        computing is dropped, and its buffers freed, when it returns."""
        self._requests.release()
        self._values = self._slice = None
        self.view.clear_keyed("")
        self.card.set_status_text("")
        self.note.setText("")
        self.setWindowTitle("RSM Custom Slice")

    def shutdown(self):
        """Close the window's worker with the page, as the page's owners close: CLEAN, or PENDING while a plane
        is still cut (the page asks again); nothing is cut after it."""
        return self._requests.shutdown()

    def hideEvent(self, event) -> None:
        if not event.spontaneous():  # closed (not minimised): the volume goes until it is shown again
            self.release()
        super().hideEvent(event)


_COMPUTING = "Computing the plane…"

__all__ = ["RSMCustomSliceWindow", "RSMVolumeWindow"]
