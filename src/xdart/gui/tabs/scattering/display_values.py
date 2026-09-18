from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import os

import numpy as np

from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.io.output_transaction import (
    StreamTerminal,
    stream_terminal_object_revision,
)

from .events import CleanupStatus, DetachedDiagnostic, RunIdentity, detached_diagnostic_is_valid

class StandardEventKind(str, Enum):
    DISCOVERY = "discovery"
    CONTEXT_READY = "context_ready"
    FRAME_READY = "frame_ready"
    DISPLAY_READY = "display_ready"
    FINISHED = "finished"
    STOPPED = "stopped"
    FAILED = "failed"

@dataclass(frozen=True, slots=True)
class DisplayFrameKey:
    """Immutable identity for one frame in one admitted output artifact."""

    run_identity: RunIdentity
    source_scan: str
    artifact: str
    local_frame_label: int
    work_ordinal: int

    def __post_init__(self) -> None:
        if (
            type(self.run_identity) is not RunIdentity
            or type(self.source_scan) is not str
            or not self.source_scan
            or type(self.artifact) is not str
            or not self.artifact
            or type(self.local_frame_label) is not int
            or type(self.work_ordinal) is not int
            or self.work_ordinal < 1
        ):
            raise TypeError("display frame identity is invalid")


@dataclass(frozen=True, slots=True)
class DisplayFrameCatalog:
    """Array-free ordered navigation facts for one exact run."""

    run_identity: RunIdentity
    entries: tuple[DisplayFrameKey, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.run_identity) is not RunIdentity
            or type(self.entries) is not tuple
            or not all(
                type(item) is DisplayFrameKey
                and item.run_identity is self.run_identity
                for item in self.entries
            )
            or len(set(self.entries)) != len(self.entries)
        ):
            raise TypeError("display frame catalog is invalid")

@dataclass(frozen=True, slots=True)
class DisplayNavigationDelta:
    """One catalog append and the exact keys retired by that operation."""

    appended: DisplayFrameKey
    retired: tuple[DisplayFrameKey, ...] = ()

    def __post_init__(self) -> None:
        identity = self.appended.run_identity
        if (
            type(self.appended) is not DisplayFrameKey
            or type(self.retired) is not tuple
            or not all(
                type(item) is DisplayFrameKey
                and item.run_identity is identity
                for item in self.retired
            )
        ):
            raise TypeError("display navigation delta is invalid")


@dataclass(frozen=True, slots=True)
class StandardQuartileTiming:
    """Four temporal timing buckets captured at run-completion quartiles."""

    frame_counts: tuple[int, int, int, int]
    details: tuple[
        tuple[str, tuple[float, float, float, float]], ...
    ]
    compute_counts: tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self) -> None:
        counts = self.frame_counts
        details = self.details
        valid_counts = (
            type(counts) is tuple
            and len(counts) == 4
            and all(type(value) is int and value > 0 for value in counts)
        )
        valid_compute_counts = (
            type(self.compute_counts) is tuple
            and len(self.compute_counts) == 4
            and all(
                type(value) is int and value >= 0
                for value in self.compute_counts
            )
        )
        valid_details = (
            type(details) is tuple
            and all(
                type(item) is tuple
                and len(item) == 2
                and type(item[0]) is str
                and bool(item[0])
                and type(item[1]) is tuple
                and len(item[1]) == 4
                and all(
                    type(value) is float
                    and np.isfinite(value)
                    and value >= 0.0
                    for value in item[1]
                )
                for item in details
            )
        )
        names = tuple(item[0] for item in details) if valid_details else ()
        if (
            not valid_counts
            or not valid_compute_counts
            or not valid_details
            or not details
            or len(set(names)) != len(names)
        ):
            raise TypeError("quartile timing is invalid")


@dataclass(frozen=True, slots=True)
class StandardTerminalTiming:
    """Executor-clock terminal duration plus available measured components."""

    elapsed_seconds: float
    work_seconds: float
    cleanup_seconds: float
    details: tuple[tuple[str, float], ...] = ()
    quartiles: StandardQuartileTiming | None = None

    def __post_init__(self) -> None:
        details = self.details
        durations = (
            self.elapsed_seconds,
            self.work_seconds,
            self.cleanup_seconds,
        )
        valid_details = (
            type(details) is tuple
            and all(
                type(item) is tuple
                and len(item) == 2
                and type(item[0]) is str
                and bool(item[0])
                and type(item[1]) is float
                and np.isfinite(item[1])
                and item[1] >= 0.0
                for item in details
            )
        )
        names = (
            tuple(item[0] for item in details)
            if valid_details
            else ()
        )
        if (
            any(
                type(value) is not float
                or not np.isfinite(value)
                or value < 0.0
                for value in durations
            )
            or not valid_details
            or len(set(names)) != len(names)
            or (
                self.quartiles is not None
                and type(self.quartiles) is not StandardQuartileTiming
            )
        ):
            raise TypeError("terminal timing is invalid")


@dataclass(frozen=True, slots=True)
class StandardRunEvent:
    run_identity: RunIdentity
    kind: StandardEventKind
    completed: int = 0
    total: int = 0
    artifact: str = ""
    detail: str = ""
    cleanup_status: CleanupStatus = CleanupStatus.CLEANED
    primary: DetachedDiagnostic | None = None
    cleanup_failures: tuple[DetachedDiagnostic, ...] = ()
    artifacts: tuple[str, ...] = ()
    frame_key: DisplayFrameKey | None = None
    selection_generation: int = 0
    navigation_delta: DisplayNavigationDelta | None = None
    artifact_completed: int = 0
    artifact_total: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_pending: int = 0
    files_discovered: int = 0
    terminal_timing: StandardTerminalTiming | None = None
    terminal_commit_identity: StreamTerminal | None = None

@dataclass(frozen=True, slots=True)
class StandardDisplayPayload:
    selection_generation: int
    frame_key: DisplayFrameKey
    title: str
    view: FrameView
    status: str = "running"
    measurement_mode: str = "Standard"
    gi_incidence_motor: str = ""
    gi_resolved_motor: str = ""
    gi_mode_1d: str = ""
    gi_mode_2d: str = ""
    wavelength_m: float | None = None
    averaged: bool = False
    #: The frame's other DIRECT 2-D maps by GI mode key (``view`` holds the
    #: primary).  They are what the 2-D pane may show instead; empty for a
    #: one-map result.
    extra_views_2d: Mapping[str, FrameView] = field(default_factory=dict)
    #: True only on a PRESENTED payload whose 2-D map was re-binned from the
    #: q_ip–q_oop cake for display (never a stored or direct ``q_chi`` result).
    derived_2d: bool = False


def companion_views_2d(record: FrameRecord | None) -> dict[str, FrameView]:
    """The complete non-active 2-D views of *record*, by mode key."""
    if record is None or len(record.results_2d) < 2:
        return {}
    return {
        mode: view for mode, view in record.results_2d.items()
        if mode != record.active_mode_2d and view.has_2d
    }


def standard_event_is_valid(value: object, identity: RunIdentity) -> bool:
    try:
        return (type(value) is StandardRunEvent and value.run_identity is identity
                and type(value.kind) is StandardEventKind
                and _nonnegative_int(value.completed) and _nonnegative_int(value.total)
                and type(value.artifact) is str and type(value.detail) is str
                and type(value.cleanup_status) is CleanupStatus
                and (value.primary is None or detached_diagnostic_is_valid(value.primary))
                and type(value.cleanup_failures) is tuple
                and all(detached_diagnostic_is_valid(item) for item in value.cleanup_failures)
                and type(value.artifacts) is tuple
                and all(type(item) is str and item for item in value.artifacts)
                and (
                    value.frame_key is None
                    or type(value.frame_key) is DisplayFrameKey
                    and value.frame_key.run_identity is identity
                )
                and (
                    value.navigation_delta is None
                    or type(value.navigation_delta) is DisplayNavigationDelta
                    and value.navigation_delta.appended is value.frame_key
                    and value.navigation_delta.appended.run_identity is identity
                )
                and _nonnegative_int(value.selection_generation)
                and _nonnegative_int(value.artifact_completed)
                and _nonnegative_int(value.artifact_total)
                and value.artifact_completed <= value.artifact_total
                and _nonnegative_int(value.files_processed)
                and _nonnegative_int(value.files_skipped)
                and _nonnegative_int(value.files_pending)
                and _nonnegative_int(value.files_discovered)
                and value.files_processed
                + value.files_skipped
                + value.files_pending
                == value.files_discovered
                and (
                    value.terminal_timing is None
                    or type(value.terminal_timing) is StandardTerminalTiming
                    and value.kind in {
                        StandardEventKind.FINISHED,
                        StandardEventKind.STOPPED,
                        StandardEventKind.FAILED,
                    }
                )
                and (
                    value.terminal_commit_identity is None
                    or _stream_terminal_is_valid(
                        value.terminal_commit_identity,
                    )
                    and os.path.normcase(os.path.abspath(value.artifact))
                    == value.terminal_commit_identity.target
                    and value.kind is StandardEventKind.FINISHED
                    and value.cleanup_status is CleanupStatus.CLEANED
                ))
    except Exception:
        return False


def standard_progress(run: object, session: object | None = None) -> tuple[int, int]:
    try:
        current = run.session if session is None else session
        completed = run.completed + current.frames_completed if current is not None else run.completed
        total = max(run.total, len(run.scan)) if run.scan is not None else max(run.total, completed)
        if _nonnegative_int(completed) and _nonnegative_int(total):
            return completed, total
    except Exception:
        pass
    return 0, 0


def display_payload_is_valid(
    value: object,
    identity: RunIdentity,
    frame: DisplayFrameKey,
    selection_generation: int,
) -> bool:
    try:
        view = value.view
        label = frame.local_frame_label
        return (type(frame) is DisplayFrameKey and frame.run_identity is identity
                and type(value) is StandardDisplayPayload
                and type(value.selection_generation) is int
                and value.selection_generation == selection_generation
                and value.frame_key is frame
                and type(value.title) is str and type(value.status) is str
                and type(value.averaged) is bool
                and value.measurement_mode in {"Standard", "GI"}
                and type(value.gi_incidence_motor) is str
                and type(value.gi_resolved_motor) is str
                and type(value.gi_mode_1d) is str
                and type(value.gi_mode_2d) is str
                and all(
                    type(mode) is str and type(extra) is FrameView
                    for mode, extra in value.extra_views_2d.items()
                )
                and type(value.derived_2d) is bool
                and (
                    value.wavelength_m is None
                    or type(value.wavelength_m) is float
                    and np.isfinite(value.wavelength_m)
                    and value.wavelength_m > 0.0
                )
                and (
                    value.measurement_mode == "Standard"
                    or bool(value.gi_resolved_motor)
                )
                and type(view) is FrameView and type(view.label) is int
                and view.label == label and _array_is_valid(view.raw, 2, nonempty=True)
                and _array_is_valid(view.thumbnail, 2, nonempty=True)
                and _axis_is_valid(view.axis_1d) and _array_is_valid(view.intensity_1d, 1)
                and (view.axis_1d is None or view.intensity_1d is None
                     or view.axis_1d.values is None or (view.intensity_1d.size > 0
                     and view.axis_1d.values.shape == view.intensity_1d.shape))
                and _axis_is_valid(view.axis_2d_x) and _axis_is_valid(view.axis_2d_y)
                and _array_is_valid(view.intensity_2d, 2)
                and (view.axis_2d_x is None or view.axis_2d_y is None
                     or view.intensity_2d is None or view.axis_2d_x.values is None
                     or view.axis_2d_y.values is None or (view.intensity_2d.size > 0
                     and view.axis_2d_x.values.shape == (view.intensity_2d.shape[1],)
                     and view.axis_2d_y.values.shape == (view.intensity_2d.shape[0],))))
    except Exception:
        return False

def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _stream_terminal_is_valid(value: object) -> bool:
    return (
        type(value) is StreamTerminal
        and type(value.target) is str
        and bool(value.target)
        and _nonnegative_int(value.size)
        and type(value.digest) is str
        and len(value.digest) == 64
        and all(character in "0123456789abcdef" for character in value.digest)
        and type(value.ordinal) is int
        and value.ordinal >= 1
        and stream_terminal_object_revision(value) is not None
    )

def _axis_is_valid(value: object) -> bool:
    return value is None or (type(value) is Axis and type(value.label) is str and type(value.unit) is str and _array_is_valid(value.values, 1))

def _array_is_valid(value: object, dimensions: int, *, nonempty: bool = False) -> bool:
    return value is None or (type(value) is np.ndarray and value.ndim == dimensions and (not nonempty or value.size > 0) and np.issubdtype(value.dtype, np.number))
