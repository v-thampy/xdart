"""Immutable values and typed commands for the opt-in E3 visual shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from os.path import normcase
from types import MappingProxyType
from typing import Mapping

import numpy as np

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


class FrameSelectionIntent(str, Enum):
    EXACT = "exact"
    VISIT = "visit"
    TOGGLE_TRACE = "toggle_trace"
    REMOVE_TRACE_RANGE = "remove_trace_range"


@dataclass(frozen=True, slots=True)
class ShellCommand:
    kind: ShellCommandKind
    value: Scalar = None
    path: tuple[str, ...] = ()
    frame: DisplayFrameKey | None = None
    frames: tuple[DisplayFrameKey, ...] = ()
    intent: FrameSelectionIntent = FrameSelectionIntent.EXACT
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ShellCommandKind:
            raise TypeError("shell command kind must be exact")
        if not _scalar_is_valid(self.value):
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

    def __post_init__(self) -> None:
        if type(self.values) is not np.ndarray or self.values.ndim != 1:
            raise TypeError("axis values must be a one-dimensional ndarray")


@dataclass(frozen=True, slots=True)
class TraceProjection:
    frame: DisplayFrameKey
    axis: AxisProjection
    intensity: np.ndarray
    title: str = ""
    epoch: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.frame) is not DisplayFrameKey
            or type(self.intensity) is not np.ndarray
            or self.intensity.ndim != 1
            or self.intensity.shape != self.axis.values.shape
        ):
            raise TypeError("trace projection is invalid")
        if self.epoch is not None and (
            type(self.epoch) is not float or not np.isfinite(self.epoch)
        ):
            raise TypeError("trace epoch must be a finite float or None")


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

    def __post_init__(self) -> None:
        if type(self.frame) is not DisplayFrameKey:
            raise TypeError("heavy projection frame must be exact")
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


@dataclass(frozen=True, slots=True)
class ScientificPlotOptions:
    waterfall_y_axis: str = "Frame #"
    waterfall_start: int = 1
    waterfall_stop: int = 0
    waterfall_step: int = 1
    overlay_offset: float = 5.0
    show_legend: bool = True
    intensity_scale: str = "Linear"

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


@dataclass(frozen=True, slots=True)
class ScientificProjection:
    heavy_available: frozenset[DisplayFrameKey] = frozenset()
    traces: tuple[TraceProjection, ...] = ()
    heavy: HeavyProjection | None = None
    title: str = "Current"
    processing_mode: str = "Int 2D"
    measurement_mode: str = "Standard"
    gi_mode_1d: str = ""
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

    def __post_init__(self) -> None:
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
    "ProgressProjection",
    "PinnedTraceProjection",
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
