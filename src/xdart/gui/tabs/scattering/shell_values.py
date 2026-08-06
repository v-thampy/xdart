"""Immutable values and typed commands for the opt-in E3 visual shell."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from xrd_tools.session.readiness import ControlPanelRenderState

from .controls_readiness import ControlsReadinessProjection
from .display_values import DisplayFrameKey


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
    SET_NORM_CHANNEL = "set_norm_channel"
    SET_BACKGROUND = "set_background"
    SET_COLOR_MAP = "set_color_map"
    SET_LOG_SCALE = "set_log_scale"
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


@dataclass(frozen=True, slots=True)
class ShellCommand:
    kind: ShellCommandKind
    value: Scalar = None
    path: tuple[str, ...] = ()
    frame: DisplayFrameKey | None = None
    frames: tuple[DisplayFrameKey, ...] = ()

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

    def __post_init__(self) -> None:
        if not self.identifier or not self.label:
            raise ValueError("browser scan identity and label are required")


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
    gi_mode_2d: str = ""
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
    status: str = ""
    retain_display: bool = False
    live_update: bool = False
    #: The accepted aggregate's identity and revision remain truthful
    #: projection provenance.  Trace caches use identity plus effective
    #: ``norm_channel``; revision alone cannot invalidate immutable per-frame
    #: divisor results.  Backward-safe defaults mean "no accepted value".
    norm_identity: tuple[object, ...] | None = None
    norm_revision: int = 0


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
    ready: bool = False
    run_enabled: bool = False
    stop_enabled: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactProgress:
    artifact: str
    completed: int
    total: int
    # Exact prefix represented by navigation events. ``None`` keeps detached
    # callers that only know durable progress on the legacy suffix fallback.
    published: int | None = None

    def __post_init__(self) -> None:
        if not (
            type(self.artifact) is str
            and self.artifact
            and type(self.completed) is int
            and type(self.total) is int
            and 0 <= self.completed <= self.total
            and (
                self.published is None
                or type(self.published) is int
                and 0 <= self.published <= self.completed
            )
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
    controls: ControlPanelRenderState
    run: RunStripProjection
    progress: ProgressProjection = ProgressProjection()
    controls_readiness: ControlsReadinessProjection = (
        ControlsReadinessProjection()
    )

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("shell revision must be nonnegative")
        if type(self.navigation) is not FrameNavigationProjection:
            raise TypeError("shell navigation must be exact")
        if type(self.controls_readiness) is not ControlsReadinessProjection:
            raise TypeError("shell Controls readiness must be exact")


def _scalar_is_valid(value: object) -> bool:
    return value is None or type(value) in {str, int, float, bool}


__all__ = [
    "ArtifactProgress",
    "AxisProjection",
    "BrowserProjection",
    "DirectoryFileProgress",
    "BrowserScan",
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
