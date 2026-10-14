"""Immutable values and typed commands for the opt-in E3 visual shell."""

from __future__ import annotations

import math

from dataclasses import dataclass, field, replace
from enum import Enum
from os.path import normcase
from types import MappingProxyType
from typing import Mapping

import numpy as np

from xrd_tools.core.angular_support import AngularSupportProfile
from xrd_tools.session.display_logic import PanelKey, PanelRole
from xrd_tools.session.readiness import ControlsProjection

from .browser_catalog import BrowserCatalogEntry, natural_name_key
from .controls_readiness import ControlsReadinessProjection
from .display_values import DisplayFrameKey, StandardTerminalTiming
from .external_tools import (
    ExternalToolsProjection,
    unavailable_external_tools,
)


Scalar = str | int | float | bool | None


class ShellCommandKind(str, Enum):
    MENU = "menu"
    REFRESH_BROWSER = "refresh_browser"
    SET_DATE_SORT = "set_date_sort"
    SELECT_SCAN = "select_scan"
    SELECT_BROWSER_FRAMES = "select_browser_frames"
    SHOW_ALL = "show_all"
    SHOW_METADATA = "show_metadata"
    SET_AUTO_LAST = "set_auto_last"
    LAUNCH_TOOL = "launch_tool"
    LAUNCH_EXTERNAL_VIEWER = "launch_external_viewer"
    SET_NORM_CHANNEL = "set_norm_channel"
    SET_BACKGROUND = "set_background"
    SET_COLOR_MAP = "set_color_map"
    SET_LOG_SCALE = "set_log_scale"
    SET_DETECTOR_MODE = "set_detector_mode"
    SET_IMAGE_AXIS = "set_image_axis"
    SET_PLOT_AXIS = "set_plot_axis"
    SET_PLOT_MODE = "set_plot_mode"
    SET_SLICE_ENABLED = "set_slice_enabled"
    SET_SLICE_CENTER = "set_slice_center"
    SET_SLICE_WIDTH = "set_slice_width"
    PIN_SLICE = "pin_slice"
    SET_SHARE_AXIS = "set_share_axis"
    SET_RANGE = "set_range"
    SHOW_WATERFALL_OPTIONS = "show_waterfall_options"
    SET_PLOT_OPTION = "set_plot_option"
    CLEAR_1D = "clear_1d"
    SELECT_FRAME = "select_frame"
    HYDRATE_FRAME = "hydrate_frame"
    CONTROL_EDIT = "control_edit"
    CONTROL_DRAFT = "control_draft"
    CONTROL_BROWSE = "control_browse"
    CONTROL_ACTION = "control_action"
    ANALYSIS_ACTION = "analysis_action"
    SET_PROCESSING_MODE = "set_processing_mode"
    SET_BATCH = "set_batch"
    SET_CORES = "set_cores"
    SET_LIVE = "set_live"
    RUN_ACTION = "run_action"
    STOP = "stop"
    SET_OUTPUT_POLICY = "set_output_policy"
    #: a results-profile Data Browser (another page's results list): one result chosen, or Open… pressed
    SELECT_RESULT = "select_result"
    OPEN_RESULT = "open_result"
    #: an extended results list's selection changed (its value: the selected identities); a result's own
    #: entry chosen in the second pane (Stitch's contributing frame); its Raw image action pressed
    SELECT_RESULTS = "select_results"
    SELECT_RESULT_FRAME = "select_result_frame"
    SHOW_RAW_IMAGE = "show_raw_image"
    #: a keyed layout's image panel stepped to another index along its normal (an RSM slice): its value the
    #: index, its path the panel's instance (``PanelKey.instance``)
    SET_PANEL_INDEX = "set_panel_index"
    #: a host page's own tool button pressed (``ScientificView.set_page_tools``): its value the tool's name
    PAGE_TOOL = "page_tool"


class FrameSelectionIntent(str, Enum):
    EXACT = "exact"
    VISIT = "visit"
    TOGGLE_TRACE = "toggle_trace"
    REMOVE_TRACE_RANGE = "remove_trace_range"


@dataclass(frozen=True, slots=True)
class ShellCommand:
    kind: ShellCommandKind
    value: Scalar | tuple[tuple[str, bool], ...] | tuple[tuple[str, float], ...] = None
    path: tuple[str, ...] = ()
    frame: DisplayFrameKey | None = None
    frames: tuple[DisplayFrameKey, ...] = ()
    intent: FrameSelectionIntent = FrameSelectionIntent.EXACT
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ShellCommandKind:
            raise TypeError("shell command kind must be exact")
        phase_selection = (self.kind is ShellCommandKind.SET_PLOT_OPTION
            and self.path == ("other", "phase_references") and type(self.value) is tuple
            and len(self.value) <= 8 and all(type(item) is tuple and len(item) == 2
                and type(item[0]) is str and bool(item[0]) and type(item[1]) is bool for item in self.value)
            and len({item[0] for item in self.value}) == len(self.value))
        # REFERENCE-PHASES-20261011: per-phase minimums, (phase id, percent) pairs
        phase_floors = (self.kind is ShellCommandKind.SET_PLOT_OPTION
            and self.path == ("other", "phase_floors") and type(self.value) is tuple
            and len(self.value) <= 8 and all(type(item) is tuple and len(item) == 2
                and type(item[0]) is str and bool(item[0]) and type(item[1]) in (int, float)
                and math.isfinite(item[1]) for item in self.value)
            and len({item[0] for item in self.value}) == len(self.value))
        if not _scalar_is_valid(self.value) and not phase_selection and not phase_floors:
            raise TypeError("shell command values must be detached scalars")
        if (
            type(self.path) is not tuple
            or not all(type(part) is str for part in self.path)
        ):
            raise TypeError("shell command path must contain strings")
        if self.frame is not None and type(self.frame) is not DisplayFrameKey:
            raise TypeError("shell command frame must be an exact frame key")
        if (
            type(self.frames) is not tuple
            or not all(type(item) is DisplayFrameKey for item in self.frames)
        ):
            raise TypeError("shell command frames must be exact frame keys")
        if type(self.intent) is not FrameSelectionIntent:
            raise TypeError("frame selection intent must be exact")
        if (
            type(self.artifacts) is not tuple
            or not all(type(item) is str and item for item in self.artifacts)
            or len(set(self.artifacts)) != len(self.artifacts)
        ):
            raise TypeError("shell command artifacts must be unique paths")


class ShellPhase(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class BrowserScan:
    identifier: str
    label: str
    detail: str = ""
    is_directory: bool = False

    def __post_init__(self) -> None:
        if (
            not self.identifier
            or not self.label
            or type(self.is_directory) is not bool
        ):
            raise ValueError("browser scan identity and label are required")


@dataclass(frozen=True, slots=True)
class BrowserScanIndex:
    """Prevalidated static Browser rows and their membership index."""

    scans: tuple[BrowserScan, ...] = ()
    identifiers: frozenset[str] = frozenset()
    date_sorted: bool = False
    paths: Mapping[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.scans) is not tuple
            or not all(type(scan) is BrowserScan for scan in self.scans)
            or type(self.identifiers) is not frozenset
            or not all(
                type(identifier) is str and identifier
                for identifier in self.identifiers
            )
            or self.identifiers
            != frozenset(scan.identifier for scan in self.scans)
            or type(self.date_sorted) is not bool
        ):
            raise TypeError("browser scan index is invalid")
        # Cache filesystem comparison keys with the catalog, preserving the
        # original spelling for labels, tooltips and selection commands.
        object.__setattr__(self, "paths", MappingProxyType({
            normcase(scan.identifier): scan.identifier for scan in self.scans
        }))


def build_browser_scan_index(
    catalog: tuple[BrowserCatalogEntry, ...],
    date_sorted: bool,
) -> BrowserScanIndex:
    """Convert one immutable source catalog into reusable static rows."""

    if (
        type(catalog) is not tuple
        or not all(type(entry) is BrowserCatalogEntry for entry in catalog)
        or type(date_sorted) is not bool
    ):
        raise TypeError("browser source catalog is invalid")
    entries = (
        tuple(
            sorted(
                catalog,
                key=lambda entry: (
                    0 if entry.label == ".." else 1,
                    0 if entry.label == ".." else -entry.modified_ns,
                    natural_name_key(
                        entry.label.removesuffix("/")
                        if entry.is_directory
                        else entry.label
                    ),
                ),
            )
        )
        if date_sorted
        else catalog
    )
    scans = tuple(
        BrowserScan(
            entry.artifact,
            entry.label,
            entry.artifact,
            entry.is_directory,
        )
        for entry in entries
    )
    return BrowserScanIndex(
        scans,
        frozenset(scan.identifier for scan in scans),
        date_sorted,
    )


@dataclass(frozen=True, slots=True)
class FrameNavigationProjection:
    frames: tuple[DisplayFrameKey, ...] = ()
    current: DisplayFrameKey | None = None
    selected: tuple[DisplayFrameKey, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.frames) is not tuple
            or not all(type(frame) is DisplayFrameKey for frame in self.frames)
        ):
            raise TypeError("navigation frames must be exact frame keys")
        if len({id(frame) for frame in self.frames}) != len(self.frames):
            raise ValueError("navigation frame identities must be unique")
        frame_ids = {id(frame) for frame in self.frames}
        if self.current is not None and type(self.current) is not DisplayFrameKey:
            raise TypeError("navigation current must be an exact frame key")
        if self.current is not None and id(self.current) not in frame_ids:
            raise ValueError(
                "navigation current must belong to frames by identity"
            )
        if (
            type(self.selected) is not tuple
            or not all(
                type(frame) is DisplayFrameKey for frame in self.selected
            )
        ):
            raise TypeError("navigation selected must contain exact frame keys")
        selected_ids = [id(frame) for frame in self.selected]
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("navigation selected identities must be unique")
        if any(identity not in frame_ids for identity in selected_ids):
            raise ValueError(
                "navigation selected must belong to frames by identity"
            )
        if self.current is None:
            if self.selected:
                raise ValueError(
                    "navigation selected requires a current frame"
                )


@dataclass(frozen=True, slots=True)
class BrowserProjection:
    directory: str = ""
    scans: tuple[BrowserScan, ...] = ()
    selected_scan: str = ""
    date_sorted: bool = False
    auto_last: bool = True
    frames: tuple[DisplayFrameKey, ...] = ()
    selected_artifacts: tuple[str, ...] = ()
    multi_artifact_selection: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.frames) is not tuple
            or not all(
                type(frame) is DisplayFrameKey for frame in self.frames
            )
        ):
            raise TypeError(
                "browser frames must contain exact frame keys"
            )
        if len({id(frame) for frame in self.frames}) != len(self.frames):
            raise ValueError("browser frame identities must be unique")
        if (
            type(self.selected_artifacts) is not tuple
            or not all(
                type(artifact) is str and artifact
                for artifact in self.selected_artifacts
            )
            or len(set(self.selected_artifacts))
            != len(self.selected_artifacts)
            or type(self.multi_artifact_selection) is not bool
        ):
            raise TypeError("browser artifact selection is invalid")


@dataclass(frozen=True, slots=True)
class AxisProjection:
    values: np.ndarray
    label: str
    unit: str = ""
    #: The image's outer edges along this axis, when they are not its first and last values: an axis of bin
    #: centres gives its bins' outer edges (half a bin beyond the first and last centre), so each pixel sits
    #: at its bin's centre (an RSM slice: Codex's RSM stage-1 review F3). None: the first and last values.
    edges: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if type(self.values) is not np.ndarray or self.values.ndim != 1:
            raise TypeError("axis values must be a one-dimensional ndarray")
        if self.edges is not None:
            if (type(self.edges) is not tuple or len(self.edges) != 2
                    or any(type(value) is not float or not math.isfinite(value) for value in self.edges)
                    or not self.edges[0] < self.edges[1]):
                raise TypeError("axis edges must be two finite increasing floats")


#: A keyed image panel's presentations (``KeyedImageProjection.presentation``).
KEYED_IMAGE_PRESENTATIONS = ("shared", "own", "signed")


@dataclass(frozen=True, slots=True)
class RectRegion:
    """A rectangle on a keyed image panel (``ScientificView.set_keyed_region``), typed by the coordinates it is in.

    ``x`` and ``y`` are its (low, high) along the panel's x axis (the image's columns) and y axis (its rows), in the
    panel's own coordinates; ``x_axis`` and ``y_axis`` are those axes' (label, unit), which say what the numbers
    are: a detector's pixels (an ROI picked on a detector image) or a slice's coordinates (H and K, Qx and Qz: a
    region to fit). The view refuses a region whose axes are not the ones its panel draws. One type for both:
    ``index_bounds`` gives the rows and columns whose bins lie inside, on either (FIT-TOOLS-2026-10, Codex's design
    review F7: "Track D's detector ROI and F's slice ROI should use that same typed region API")."""

    x: tuple[float, float]
    y: tuple[float, float]
    x_axis: tuple[str, str]
    y_axis: tuple[str, str]

    def __post_init__(self) -> None:
        for bounds in (self.x, self.y):
            if (type(bounds) is not tuple or len(bounds) != 2
                    or any(type(value) not in {int, float} or not math.isfinite(value) for value in bounds)
                    or not bounds[0] < bounds[1]):
                raise TypeError("a region's x and y are (low, high): two finite increasing numbers")
        for axis in (self.x_axis, self.y_axis):
            if type(axis) is not tuple or len(axis) != 2 or any(type(part) is not str for part in axis):
                raise TypeError("a region's axes are (label, unit) text")
        object.__setattr__(self, "x", (float(self.x[0]), float(self.x[1])))
        object.__setattr__(self, "y", (float(self.y[0]), float(self.y[1])))

    def index_bounds(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple[slice, slice]:
        """The rows and columns, as ``(rows, columns)`` slices of a (y, x) image, whose bin centres (the panel's axis
        values, increasing or decreasing) lie inside the rectangle, edges included: views of the image, no copies.
        An empty slice where none does."""
        return tuple(_inside(np.asarray(values, dtype=float), bounds)
                     for values, bounds in ((y_values, self.y), (x_values, self.x)))


def _inside(values: np.ndarray, bounds: tuple[float, float]) -> slice:
    found = np.flatnonzero((values >= bounds[0]) & (values <= bounds[1]))
    return slice(0, 0) if found.size == 0 else slice(int(found[0]), int(found[-1]) + 1)


@dataclass(frozen=True, slots=True)
class KeyedImageProjection:
    """One image panel of a keyed layout (``display_logic``'s repeated ``PanelKey`` roles, the H11/WS-X2
    seam): an RSM slice. ``image`` is (y, x), as Int's cake; ``index`` of ``index_count`` is where the slice is
    taken along its plane's normal, each index captioned by ``index_captions``."""

    key: PanelKey
    title: str
    image: np.ndarray
    x_axis: AxisProjection
    y_axis: AxisProjection
    index: int
    index_captions: tuple[str, ...]
    #: False for a plane with no index to step (Custom Slice): no navigation is drawn, ``index`` is 0 and
    #: ``index_captions`` is empty
    navigation: bool = True
    #: How the panel is presented (FIT-TOOLS-2026-10, Codex's design review F7): "shared", the view's colour map and
    #: Log and the levels every shared slice takes from the intensity controls (RSM's slices; a fit's data and
    #: model); "own", the view's colour map and Log with the panel's own automatic levels; "signed", a signed
    #: difference (a fit's residual): linear whatever Log says, a diverging map and its own symmetric limits, so its
    #: sign and zero always show
    presentation: str = "shared"

    def __post_init__(self) -> None:
        if type(self.key) is not PanelKey or self.key.role is not PanelRole.SLICE_2D:
            raise TypeError("a keyed image panel is a SLICE_2D PanelKey")
        if self.presentation not in KEYED_IMAGE_PRESENTATIONS:
            raise TypeError(f"a keyed image's presentation is one of {KEYED_IMAGE_PRESENTATIONS}")
        if (type(self.image) is not np.ndarray or self.image.ndim != 2
                or self.image.shape != (self.y_axis.values.size, self.x_axis.values.size)):
            raise TypeError("a keyed image is (y, x) on its two axes")
        if type(self.navigation) is not bool:
            raise TypeError("a keyed image's navigation is a bool")
        if not self.navigation:
            if self.index != 0 or self.index_captions != ():
                raise TypeError("a keyed image without navigation has index 0 and no captions")
        elif (type(self.index_captions) is not tuple or not self.index_captions
                or any(type(caption) is not str for caption in self.index_captions)
                or type(self.index) is not int or not 0 <= self.index < len(self.index_captions)):
            raise TypeError("a keyed image's index is one of its captions")


@dataclass(frozen=True, slots=True)
class KeyedTraceProjection:
    """One 1-D panel of a keyed layout: an RSM mean projection along ``axis``."""

    key: PanelKey
    title: str
    axis: AxisProjection
    intensity: np.ndarray
    y_label: str

    def __post_init__(self) -> None:
        if type(self.key) is not PanelKey or self.key.role is not PanelRole.PROJ_1D:
            raise TypeError("a keyed trace panel is a PROJ_1D PanelKey")
        if type(self.intensity) is not np.ndarray or self.intensity.shape != self.axis.values.shape:
            raise TypeError("a keyed trace's intensity follows its axis")


@dataclass(frozen=True, slots=True)
class KeyedFigureProjection:
    """A source-free computed figure, rendered by the full ScientificView."""

    key: PanelKey
    title: str
    figure: object

    def __post_init__(self):
        from xrd_tools.geometry_figures import Figure
        if type(self.key) is not PanelKey or self.key.role is not PanelRole.FIGURE:
            raise TypeError('a keyed figure requires a FIGURE PanelKey')
        if not isinstance(self.figure, Figure):
            raise TypeError('a keyed figure requires a computed Figure')


@dataclass(frozen=True, slots=True)
class KeyedPanelsProjection:
    """A keyed layout for ``ScientificView.reconcile_keyed``: rows of ``PanelKey``s (``DisplayState.layout``),
    each drawn from its one panel, under the view's own top bar (colour map, Log, Set BG), with the view's own
    intensity controls and Share Axis (``share_axis``: the panels drawing one coordinate pan and zoom together)."""

    layout: tuple[tuple[PanelKey, ...], ...]
    panels: tuple[KeyedImageProjection | KeyedTraceProjection | KeyedFigureProjection, ...]
    title: str = ""
    status: str = ""
    color_map: str = "Default"
    log_scale: bool = False
    background_enabled: bool = False
    background_set: bool = False
    #: why the background is (or is not) drawn, for the Set BG button's tip (``apply_keyed_presentation_background``)
    background_note: str = ""
    share_axis: bool = False

    def __post_init__(self) -> None:
        if type(self.share_axis) is not bool:
            raise TypeError("a keyed layout's share_axis is a bool")
        keys = [key for row in self.layout for key in row]
        if (type(self.layout) is not tuple or not keys
                or any(type(row) is not tuple for row in self.layout)
                or len(set(keys)) != len(keys)
                or sorted(map(repr, keys)) != sorted(repr(panel.key) for panel in self.panels)
                or len(self.panels) != len(keys)):
            raise TypeError("a keyed layout draws each of its keys from exactly one panel")

    def panel(self, key: PanelKey) -> KeyedImageProjection | KeyedTraceProjection | KeyedFigureProjection:
        return next(panel for panel in self.panels if panel.key == key)


@dataclass(frozen=True, slots=True)
class TraceProjection:
    frame: DisplayFrameKey
    axis: AxisProjection
    intensity: np.ndarray
    title: str = ""
    epoch: float | None = None
    wavelength_m: float | None = None
    #: This native profile's angular coverage, or the reason it has none (COVERAGE-SUPPORT-20261013); None when its
    #: producer carries none. A display conversion of the axis keeps the profile, each bin's own result.
    angular_support: AngularSupportProfile | None = None

    def __post_init__(self) -> None:
        if (
            type(self.frame) is not DisplayFrameKey
            or type(self.intensity) is not np.ndarray
            or self.intensity.ndim != 1
            or self.intensity.shape != self.axis.values.shape
        ):
            raise TypeError("trace projection is invalid")
        if self.angular_support is not None and (
            type(self.angular_support) is not AngularSupportProfile
            or self.angular_support.available and self.angular_support.supported_deg.shape != self.intensity.shape
        ):
            raise TypeError("trace angular support must be its own profile's AngularSupportProfile or None")
        if self.epoch is not None and (
            type(self.epoch) is not float or not np.isfinite(self.epoch)
        ):
            raise TypeError("trace epoch must be a finite float or None")
        if self.wavelength_m is not None and (type(self.wavelength_m) is not float
                or not math.isfinite(self.wavelength_m) or self.wavelength_m <= 0):
            raise TypeError("trace wavelength must be a positive finite float or None")


@dataclass(frozen=True, slots=True, eq=False)
class BrowseTraceSnapshot:
    """Already-planned sparse Browse rows and their full logical extent."""

    logical_frames: tuple[DisplayFrameKey, ...]
    display_frames: tuple[DisplayFrameKey, ...]
    logical_positions: tuple[int, ...]
    logical_epochs: tuple[float, ...] | None = None
    plot_mode: str = "Single"
    waterfall_active: bool = False
    stacked_options_applied: bool = False
    science_contract: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        logical = self.logical_frames
        display = self.display_frames
        positions = self.logical_positions
        if (
            type(logical) is not tuple
            or not logical
            or any(type(frame) is not DisplayFrameKey for frame in logical)
            or len({id(frame) for frame in logical}) != len(logical)
            or type(display) is not tuple
            or not display
            or len(display) > 256
            or any(type(frame) is not DisplayFrameKey for frame in display)
            or type(positions) is not tuple
            or len(positions) != len(display)
            or any(type(position) is not int for position in positions)
            or self.logical_epochs is not None
            and (
                type(self.logical_epochs) is not tuple
                or len(self.logical_epochs) != len(logical)
                or any(
                    type(value) is not float or not np.isfinite(value)
                    for value in self.logical_epochs
                )
            )
            or self.plot_mode not in {"Single", "Overlay", "Waterfall"}
            or type(self.waterfall_active) is not bool
            or type(self.stacked_options_applied) is not bool
            or self.stacked_options_applied
            != (self.plot_mode in {"Overlay", "Waterfall"}
                or self.plot_mode == "Single" and len(logical) > 1)
            or type(self.science_contract) is not tuple
        ):
            raise TypeError("Browse trace snapshot is invalid")
        previous = 0
        for frame, position in zip(display, positions, strict=True):
            if (
                position <= previous
                or position > len(logical)
                or logical[position - 1] is not frame
            ):
                raise ValueError("Browse trace positions changed ownership")
            previous = position


@dataclass(frozen=True, slots=True)
class SlicePin:
    """One scan-qualified immutable slice recipe owned by the page."""

    frame: DisplayFrameKey
    plot_axis: str
    center: float
    width: float

    def __post_init__(self) -> None:
        if (
            type(self.frame) is not DisplayFrameKey
            or type(self.plot_axis) is not str
            or not self.plot_axis
            or type(self.center) is not float
            or not np.isfinite(self.center)
            or type(self.width) is not float
            or not np.isfinite(self.width)
            or self.width < 0.0
        ):
            raise TypeError("slice pin is invalid")

    @property
    def projection_id(self) -> tuple[object, ...]:
        return (
            id(self.frame),
            self.plot_axis,
            self.center,
            self.width,
        )


@dataclass(frozen=True, slots=True)
class PinnedTraceProjection:
    pin: SlicePin
    trace: TraceProjection

    def __post_init__(self) -> None:
        if (
            type(self.pin) is not SlicePin
            or type(self.trace) is not TraceProjection
            or self.trace.frame is not self.pin.frame
        ):
            raise TypeError("pinned trace projection is invalid")


@dataclass(frozen=True, slots=True)
class HeavyProjection:
    frame: DisplayFrameKey
    raw: np.ndarray | None = None
    cake: np.ndarray | None = None
    cake_x: AxisProjection | None = None
    cake_y: AxisProjection | None = None
    detector_shape: tuple[int, int] | None = None
    detector_source: str = "none"
    cake_coverage: np.ndarray | None = None
    ring_geometry: Mapping | None = None
    #: ``"saved"`` when ``ring_geometry`` was rebuilt from a Browse output's own record
    #: (BROWSE-REFERENCE-RINGS-20261013) rather than published by the Run being shown.
    ring_origin: str = ""

    def __post_init__(self) -> None:
        if type(self.frame) is not DisplayFrameKey:
            raise TypeError("heavy projection frame must be exact")
        if self.ring_origin not in {"", "saved"}:
            raise ValueError("ring origin must be exact")
        if self.raw is not None and (
            type(self.raw) is not np.ndarray or self.raw.ndim != 2
        ):
            raise TypeError("raw projection must be a two-dimensional ndarray")
        if self.cake is not None and (
            type(self.cake) is not np.ndarray or self.cake.ndim != 2
        ):
            raise TypeError("cake projection must be a two-dimensional ndarray")
        if self.cake is not None and (
            self.cake_x is None
            or self.cake_y is None
            or self.cake_x.values.shape != (self.cake.shape[1],)
            or self.cake_y.values.shape != (self.cake.shape[0],)
        ):
            raise TypeError("cake axes do not match the cake image")
        if self.detector_shape is not None and (
            type(self.detector_shape) is not tuple
            or len(self.detector_shape) != 2
            or any(type(value) is not int or value <= 0 for value in self.detector_shape)
        ):
            raise TypeError("detector shape must be an exact positive pair")
        if self.detector_source not in {"none", "thumbnail", "full"}:
            raise ValueError("detector source must be exact")
        if self.cake_coverage is not None and (
            type(self.cake_coverage) is not np.ndarray
            or self.cake is None
            or self.cake_coverage.shape != self.cake.shape
            or self.cake_coverage.dtype.kind not in "fiu"
            or not np.isfinite(self.cake_coverage).all()
            or (self.cake_coverage < 0).any()
        ):
            raise ValueError("cake coverage must be finite nonnegative counts matching the image")


@dataclass(frozen=True, slots=True)
class ScientificPlotOptions:
    waterfall_y_axis: str = "Frame #"
    waterfall_start: int = 1
    waterfall_stop: int = 0
    waterfall_step: int = 1
    overlay_offset: float = 5.0
    show_legend: bool = True
    intensity_scale: str = "Linear"
    show_coverage_holes: bool = False
    coordinate_grid: str = 'Off'
    phase_references: tuple[tuple[str, bool], ...] = ()
    phase_min_intensity: float = 5.
    #: "Same minimum for all phases" (REFERENCE-PHASES-20261011): on, ``phase_min_intensity`` applies to every CIF
    #: phase; off, each uses its own ``phase_floors`` value. Stored values survive the switch.
    phase_global_floor: bool = True
    #: per-phase minimum intensities ((listed phase id, percent), ...), in list order
    phase_floors: tuple[tuple[str, float], ...] = ()
    #: the top bar's Phases toggle (C6): off draws no reference phase, keeping the list, flags and minimums
    phase_references_shown: bool = True
    #: the opt-in angular coverage strip below the 1-D plot (COVERAGE-SUPPORT-20261013); display only
    show_angular_support: bool = False

    def __post_init__(self) -> None:
        if type(self.waterfall_y_axis) is not str or not self.waterfall_y_axis:
            raise TypeError("waterfall y-axis must be a nonempty string")
        if type(self.waterfall_start) is not int or self.waterfall_start < 1:
            raise ValueError("waterfall start must be one-based")
        if type(self.waterfall_stop) is not int or self.waterfall_stop < 0:
            raise ValueError("waterfall stop must be zero or one-based")
        if type(self.waterfall_step) is not int or self.waterfall_step < 1:
            raise ValueError("waterfall step must be positive")
        if (
            type(self.overlay_offset) is not float
            or not np.isfinite(self.overlay_offset)
        ):
            raise ValueError("overlay offset must be a finite float")
        if type(self.show_legend) is not bool:
            raise TypeError("legend visibility must be boolean")
        if self.intensity_scale not in {"Linear", "Sqrt", "Log"}:
            raise ValueError("unsupported 1-D intensity scale")
        if type(self.show_coverage_holes) is not bool:
            raise TypeError("coverage shading must be boolean")
        if self.coordinate_grid not in ('Off', 'q', 'χ', 'q + χ'):
            raise ValueError('unsupported coordinate grid')
        if (type(self.phase_references) is not tuple or len(self.phase_references) > 8
                or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str
                       or not row[0] or type(row[1]) is not bool for row in self.phase_references)
                or len({row[0] for row in self.phase_references}) != len(self.phase_references)):
            raise ValueError('select at most eight unique phase IDs with visibility flags')
        if (type(self.phase_min_intensity) not in (int, float)
                or not math.isfinite(self.phase_min_intensity) or not 0 <= self.phase_min_intensity <= 100):
            raise ValueError('phase floor must be between 0 and 100 percent')
        if type(self.phase_global_floor) is not bool:
            raise TypeError('the global phase floor switch must be boolean')
        listed = {row[0] for row in self.phase_references}
        if (type(self.phase_floors) is not tuple or len(self.phase_floors) > 8
                or any(type(row) is not tuple or len(row) != 2 or row[0] not in listed
                       or type(row[1]) not in (int, float) or not math.isfinite(row[1]) or not 0 <= row[1] <= 100
                       for row in self.phase_floors)
                or len({row[0] for row in self.phase_floors}) != len(self.phase_floors)):
            raise ValueError('per-phase floors name listed phases once each, between 0 and 100 percent')
        if type(self.phase_references_shown) is not bool:
            raise TypeError('the reference phases switch must be boolean')
        if type(self.show_angular_support) is not bool:
            raise TypeError('the angular coverage strip switch must be boolean')


def phase_reference_document(options: ScientificPlotOptions) -> dict:
    """The saved reference-phase section of Int's session and Stitch's settings (its ``global_floor`` and
    ``floors`` keys are additive since REFERENCE-PHASES-20261011, ``shown`` since HELP-RESIDUALS-20261011)."""
    return {"references": [list(row) for row in options.phase_references],
            "min_intensity": options.phase_min_intensity,
            "global_floor": options.phase_global_floor,
            "floors": [list(row) for row in options.phase_floors],
            "shown": options.phase_references_shown}


def restore_phase_references(options: ScientificPlotOptions, document) -> ScientificPlotOptions:
    """``options`` with a saved reference-phase section. An older document (no switch, no floors) restores the
    global minimum; a saved minimum naming a phase that is no longer listed is dropped; no ``shown`` means on."""
    references = tuple(tuple(row) for row in document.get("references", options.phase_references))
    listed = {row[0] for row in references if row}
    floors = tuple((row[0], row[1]) for row in (tuple(item) for item in document.get("floors", ()))
                   if len(row) == 2 and row[0] in listed)
    return replace(options, phase_references=references,
                   phase_min_intensity=document.get("min_intensity", options.phase_min_intensity),
                   phase_global_floor=document.get("global_floor", True), phase_floors=floors,
                   phase_references_shown=document.get("shown", True))


@dataclass(frozen=True, slots=True, eq=False)
class SavedSupportSource:
    """A Browse Int output's facts a frame's angular coverage may be rebuilt from (COVERAGE-SUPPORT-20261013, D5):
    the name its frames carry (``artifact``), the file read, its entry, the load's snapshot and its frames. Scalars
    only: no handle, frame or array crosses into the reconstruction."""

    artifact: str
    path: str
    entry: str
    snapshot: object
    labels: tuple[int, ...]

    def request(self, label: int):
        """The reconstruction request for one of the output's frames."""
        from xrd_tools.analysis.angular_support import SavedSupportRequest
        return SavedSupportRequest(self.path, self.entry, self.snapshot, self.labels, int(label))


@dataclass(frozen=True, slots=True)
class ScientificProjection:
    heavy_available: frozenset[DisplayFrameKey] = frozenset()
    traces: tuple[TraceProjection, ...] = ()
    heavy: HeavyProjection | None = None
    title: str = "Current"
    processing_mode: str = "Int 2D"
    measurement_mode: str = "Standard"
    gi_mode_1d: str = ""
    #: Every 1-D GI mode the shown results carry (a stitched GI result: q_total and its cut); the
    #: 1-D plot then offers all their axes. Empty: ``gi_mode_1d``'s alone.
    gi_modes_1d: tuple[str, ...] = ()
    #: The GI mode of the 2-D map being SHOWN (the pane's choice among
    #: ``gi_maps``), which is what the cake-derived 1-D axes follow.
    gi_mode_2d: str = ""
    #: The 2-D maps available for the current GI frame, primary first; the
    #: display-only derived q–χ map is ``"q_chi_derived"``.
    gi_maps: tuple[str, ...] = ()
    #: The shown map was re-binned from the q_ip–q_oop cake for display.
    derived_2d: bool = False
    norm_channels: tuple[str, ...] = ("Norm Channel",)
    norm_channel: str = "Norm Channel"
    color_maps: tuple[str, ...] = ("Default", "viridis", "magma")
    color_map: str = "Default"
    log_scale: bool = False
    image_axis: str = "Q-Chi"
    plot_axis: str = "Q"
    plot_mode: str = "Single"
    share_axis: bool = False
    slice_enabled: bool = False
    slice_center: float = 0.0
    slice_width: float = 10.0
    slice_pins: tuple[SlicePin, ...] = ()
    pinned_traces: tuple[PinnedTraceProjection, ...] = ()
    q_range: tuple[float, float] = (0.0, 10.0)
    chi_range: tuple[float, float] = (-180.0, 180.0)
    plot_options: ScientificPlotOptions = ScientificPlotOptions()
    background_set: bool = False
    background_enabled: bool = True
    status: str = ""
    retain_display: bool = False
    live_update: bool = False
    #: The accepted aggregate's identity and revision remain truthful
    #: projection provenance.  Trace caches use identity plus effective
    #: ``norm_channel``; revision alone cannot invalidate immutable per-frame
    #: divisor results.  Backward-safe defaults mean "no accepted value".
    norm_identity: tuple[object, ...] | None = None
    norm_revision: int = 0
    detector_mode: str = "thumbnail"
    detector_available: bool = False
    detector_pending: bool = False
    detector_diagnostic: str = ""
    browse_trace_snapshot: BrowseTraceSnapshot | None = None
    # Whole detached cut batches replace history, including empty/refused rows.
    # Acquisition's incremental trace publications keep their existing merge.
    replace_trace_history: bool = False
    #: Whether the authoritative display selection is the acquisition, rather than Browse or a viewer.
    #: Frame RunIdentity alone cannot distinguish Browse, which can retain the acquisition's identity.
    acquisition_selected: bool = False
    #: A Browse Int frame's saved rings (BROWSE-REFERENCE-RINGS-20261013): the specific reason its output's
    #: record refuses them, or, when it supplies them, whether they wait for the frame's image.
    saved_ring_reason: str = ""
    saved_ring_expected: bool = False
    #: A Browse Int output the strip may rebuild older frames' angular coverage from (COVERAGE-SUPPORT-20261013, D5)
    saved_support_source: SavedSupportSource | None = None

    def __post_init__(self) -> None:
        if type(self.acquisition_selected) is not bool:
            raise TypeError("acquisition selection must be boolean")
        if type(self.saved_ring_reason) is not str or type(self.saved_ring_expected) is not bool:
            raise TypeError("saved ring availability must be exact")
        if self.saved_support_source is not None and type(self.saved_support_source) is not SavedSupportSource:
            raise TypeError("the saved support source must be exact")
        if self.detector_mode not in {"thumbnail", "full"}:
            raise ValueError("detector mode must be exact")
        if type(self.detector_available) is not bool or type(self.detector_pending) is not bool:
            raise TypeError("detector availability must be boolean")
        if type(self.detector_diagnostic) is not str:
            raise TypeError("detector diagnostic must be text")
        if type(self.background_enabled) is not bool:
            raise TypeError("background availability must be boolean")
        if (
            self.browse_trace_snapshot is not None
            and type(self.browse_trace_snapshot) is not BrowseTraceSnapshot
        ):
            raise TypeError("Browse trace snapshot must be exact")

    @property
    def browse_science_contract(self) -> tuple[object, ...]:
        """All presentation facts that affect copied Browse trace meaning."""

        return (
            self.processing_mode,
            self.measurement_mode,
            self.gi_mode_1d,
            self.gi_mode_2d,
            self.derived_2d,
            self.norm_identity,
            self.norm_revision,
            self.norm_channel,
            self.plot_axis,
            self.plot_mode,
            self.share_axis,
            self.slice_enabled,
            self.slice_center,
            self.slice_width,
            tuple(pin.projection_id for pin in self.slice_pins),
            self.plot_options,
            self.color_map,
        )


@dataclass(frozen=True, slots=True)
class RunStripProjection:
    phase: ShellPhase = ShellPhase.IDLE
    modes: tuple[str, ...] = ("Int 2D",)
    disabled_modes: tuple[tuple[str, str], ...] = ()
    mode: str = "Int 2D"
    batch: bool = False
    cores: int = 1
    max_cores: int = 1
    live: bool = False
    output_policy: str = "Append"
    readiness: str = "Needs setup"
    readiness_tooltip: str = ""
    ready: bool = False
    run_enabled: bool = False
    stop_enabled: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactProgress:
    artifact: str
    completed: int
    total: int
    # Exact prefix represented by navigation events.
    published: int

    def __post_init__(self) -> None:
        if not (
            type(self.artifact) is str
            and self.artifact
            and type(self.completed) is int
            and type(self.total) is int
            and 0 <= self.completed <= self.total
            and type(self.published) is int
            and 0 <= self.published <= self.completed
        ):
            raise ValueError("artifact progress is invalid")


@dataclass(frozen=True, slots=True)
class DirectoryFileProgress:
    processed: int
    skipped: int
    pending: int
    discovered: int

    def __post_init__(self) -> None:
        values = (
            self.processed,
            self.skipped,
            self.pending,
            self.discovered,
        )
        if not (
            all(type(value) is int and value >= 0 for value in values)
            and self.processed + self.skipped + self.pending
            == self.discovered
        ):
            raise ValueError("directory file progress is invalid")

    def text(self, state: str) -> str:
        return (
            f"{state} · {self.processed} processed · "
            f"{self.skipped} skipped · {self.pending} pending · "
            f"{self.discovered} discovered"
        )


@dataclass(frozen=True, slots=True)
class ProgressProjection:
    completed: int = 0
    total: int = 0
    detail: str = ""
    artifacts: tuple[ArtifactProgress, ...] = ()
    directory_files: DirectoryFileProgress | None = None
    terminal: bool = False
    terminal_timing: StandardTerminalTiming | None = None

    def for_artifact(self, artifact: str) -> ArtifactProgress | None:
        return next(
            (
                value
                for value in self.artifacts
                if value.artifact == artifact
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class ShellProjection:
    revision: int
    browser: BrowserProjection
    scientific: ScientificProjection
    navigation: FrameNavigationProjection
    controls: ControlsProjection
    run: RunStripProjection
    progress: ProgressProjection = ProgressProjection()
    controls_readiness: ControlsReadinessProjection = (
        ControlsReadinessProjection()
    )
    external_tools: ExternalToolsProjection = unavailable_external_tools()

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("shell revision must be nonnegative")
        if type(self.navigation) is not FrameNavigationProjection:
            raise TypeError("shell navigation must be exact")
        if type(self.controls_readiness) is not ControlsReadinessProjection:
            raise TypeError("shell Controls readiness must be exact")
        if type(self.external_tools) is not ExternalToolsProjection:
            raise TypeError("shell external viewer projection must be exact")


def _scalar_is_valid(value: object) -> bool:
    return value is None or type(value) in {str, int, float, bool}


__all__ = [
    "ArtifactProgress",
    "AxisProjection",
    "BrowserProjection",
    "DirectoryFileProgress",
    "BrowserScan",
    "BrowserScanIndex",
    "build_browser_scan_index",
    "FrameSelectionIntent",
    "FrameNavigationProjection",
    "HeavyProjection",
    "KEYED_IMAGE_PRESENTATIONS",
    "KeyedImageProjection",
    "KeyedFigureProjection",
    "KeyedPanelsProjection",
    "KeyedTraceProjection",
    "ProgressProjection",
    "PinnedTraceProjection",
    "RectRegion",
    "RunStripProjection",
    "ScientificPlotOptions",
    "ScientificProjection",
    "SlicePin",
    "ShellCommand",
    "ShellCommandKind",
    "ShellPhase",
    "ShellProjection",
    "TraceProjection",
]
