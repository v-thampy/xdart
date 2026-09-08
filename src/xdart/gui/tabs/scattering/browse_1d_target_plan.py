"""Pure target planning and linear result custody for Browse 1-D cache reads.

Planning performs no filesystem access, starts no worker, and mutates no
runtime value.  It applies stacked row options exactly once before cache
hydration, then bounds rows only when the scientific waterfall image is
active, so a later renderer consumes the carried targets and their original
one-based logical positions verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from xdart.modules.display_context import (
    BrowseContext,
    ContextKind,
    DisplaySelection,
)
from xrd_tools.io import FrameScalarCatalog

from .browse_1d_projection import (
    Browse1DBorrowBundle,
    Browse1DProjectionStatus,
)
from .display_values import DisplayFrameKey, StandardDisplayPayload
from .scientific_waterfall_policy import waterfall_should_be_active
from .shell_projection import ScientificPreferences
from .shell_values import FrameNavigationProjection, ScientificPlotOptions


# Display planning retains the accepted 256 sampled rows.  The detached-copy
# boundary packs every identity-deduplicated array into bounded shared storage,
# so this is both the row ceiling and the maximum admitted copied-root count;
# logical_frames/positions still carry the complete scan.
MAX_BROWSE_1D_DETACHED_ROOTS = 256
MAX_BROWSE_1D_DISPLAY_TARGETS = 256
_MAX_DIAGNOSTIC_CHARS = 512
_PLOT_MODES = frozenset({"Single", "Overlay", "Waterfall", "Average", "Sum"})


class Browse1DTargetPlanStatus(str, Enum):
    PLANNED = "planned"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True, eq=False)
class Browse1DTargetPlan:
    """One immutable logical selection and its already-filtered paint rows."""

    selection: DisplaySelection
    navigation: FrameNavigationProjection
    plot_mode: str
    waterfall_active: bool
    stacked_options_applied: bool
    logical_frames: tuple[DisplayFrameKey, ...]
    display_targets: tuple[DisplayFrameKey, ...]
    logical_positions: tuple[int, ...]
    logical_epochs: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        logical = self.logical_frames
        targets = self.display_targets
        positions = self.logical_positions
        if (
            type(self.selection) is not DisplaySelection
            or self.selection.kind is not ContextKind.BROWSE
            or type(self.navigation) is not FrameNavigationProjection
            or type(self.plot_mode) is not str
            or self.plot_mode not in _PLOT_MODES
            or type(self.waterfall_active) is not bool
            or type(self.stacked_options_applied) is not bool
            or self.stacked_options_applied
            != (self.plot_mode in {"Overlay", "Waterfall"}
                or self.plot_mode == "Single" and len(logical) > 1)
            or type(logical) is not tuple
            or not logical
            or any(type(frame) is not DisplayFrameKey for frame in logical)
            or len({id(frame) for frame in logical}) != len(logical)
            or type(targets) is not tuple
            or not targets
            or len(targets) > MAX_BROWSE_1D_DISPLAY_TARGETS
            or any(type(frame) is not DisplayFrameKey for frame in targets)
            or type(positions) is not tuple
            or len(positions) != len(targets)
            or any(type(position) is not int for position in positions)
            or self.logical_epochs is not None
            and (
                type(self.logical_epochs) is not tuple
                or len(self.logical_epochs) != len(logical)
                or any(
                    type(value) is not float or not math.isfinite(value)
                    for value in self.logical_epochs
                )
            )
        ):
            raise TypeError("Browse 1-D target plan is invalid")
        previous = 0
        for target, position in zip(targets, positions, strict=True):
            if (
                position <= previous
                or position > len(logical)
                or logical[position - 1] is not target
            ):
                raise ValueError("Browse 1-D logical positions are inconsistent")
            previous = position


@dataclass(frozen=True, slots=True, eq=False)
class Browse1DTargetPlanOutcome:
    status: Browse1DTargetPlanStatus
    plan: Browse1DTargetPlan | None = None
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.status) is not Browse1DTargetPlanStatus
            or self.plan is not None and type(self.plan) is not Browse1DTargetPlan
            or type(self.diagnostic) is not str
            or len(self.diagnostic) > _MAX_DIAGNOSTIC_CHARS
        ):
            raise TypeError("Browse 1-D target plan outcome is invalid")
        if (self.status is Browse1DTargetPlanStatus.PLANNED) != (
            self.plan is not None
        ):
            raise ValueError("Browse 1-D target plan outcome changed status")


@dataclass(frozen=True, slots=True, eq=False)
class Browse1DRuntimeProjection:
    """Linear handoff from dormant runtime projection to the future painter.

    ``borrow_bundle`` is retained on both COMPLETE and cleanup-debt REFUSED
    paths.  This object never releases it: a later boundary must transfer the
    exact bundle into display-lifetime custody (or bounded copied storage),
    because renderer history may retain direct array references after paint.
    """

    status: Browse1DProjectionStatus
    plan: Browse1DTargetPlan | None = None
    payloads: tuple[StandardDisplayPayload, ...] = ()
    borrow_bundle: Browse1DBorrowBundle | None = None
    submission_identity: object | None = None
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.status) is not Browse1DProjectionStatus
            or self.plan is not None and type(self.plan) is not Browse1DTargetPlan
            or type(self.payloads) is not tuple
            or any(type(item) is not StandardDisplayPayload for item in self.payloads)
            or self.borrow_bundle is not None
            and type(self.borrow_bundle) is not Browse1DBorrowBundle
            or self.borrow_bundle is not None
            and self.borrow_bundle.released
            or self.submission_identity is not None
            and type(self.submission_identity) is not object
            or type(self.diagnostic) is not str
            or len(self.diagnostic) > _MAX_DIAGNOSTIC_CHARS
        ):
            raise TypeError("Browse 1-D runtime projection is invalid")
        if self.status is Browse1DProjectionStatus.COMPLETE:
            if (
                self.plan is None
                or not self.payloads
                or self.borrow_bundle is None
                or self.borrow_bundle.released
                or self.submission_identity is not None
                or len(self.payloads) != len(self.plan.display_targets)
                or any(
                    payload.frame_key is not frame
                    or payload.selection_generation
                    != self.plan.selection.display_generation
                    for payload, frame in zip(
                        self.payloads,
                        self.plan.display_targets,
                        strict=True,
                    )
                )
            ):
                raise ValueError("complete Browse 1-D runtime custody is invalid")
        elif self.status is Browse1DProjectionStatus.INCOMPLETE:
            if (
                self.plan is None
                or self.payloads
                or self.borrow_bundle is not None
                or type(self.submission_identity) is not object
            ):
                raise ValueError("incomplete Browse 1-D runtime custody is invalid")
        elif self.payloads or self.submission_identity is not None:
            raise ValueError("refused Browse 1-D runtime result exposed payloads")

    def __copy__(self):
        raise TypeError("Browse 1-D runtime projections cannot be copied")

    def __deepcopy__(self, _memo):
        raise TypeError("Browse 1-D runtime projections cannot be copied")

    def __reduce__(self):
        raise TypeError("Browse 1-D runtime projections cannot be serialized")

    def __reduce_ex__(self, _protocol):
        raise TypeError("Browse 1-D runtime projections cannot be serialized")


class _Refused(RuntimeError):
    pass


def _diagnostic(error: BaseException | str) -> str:
    text = error if type(error) is str else f"{type(error).__name__}: {error}"
    return str(text)[:_MAX_DIAGNOSTIC_CHARS]


def _sample_indices(count: int, maximum: int) -> tuple[int, ...]:
    if count <= maximum:
        return tuple(range(count))
    if maximum == 1:
        return (count - 1,)
    span = count - 1
    divisor = maximum - 1
    # Integer half-up rounding is platform-independent and, because
    # span/divisor > 1 here, produces a strictly increasing index set.
    return tuple(
        (index * span * 2 + divisor) // (2 * divisor)
        for index in range(maximum)
    )


def _include_current_index(
    sampled: tuple[int, ...], current_index: int | None,
) -> tuple[int, ...]:
    """Keep exact-current science without changing the bounded row count."""

    if current_index is None or current_index in sampled:
        return sampled
    replaceable = range(1, len(sampled) - 1)
    if not replaceable:
        replaceable = range(len(sampled))
    victim = min(
        replaceable,
        key=lambda index: (abs(sampled[index] - current_index), index),
    )
    return tuple(sorted((*sampled[:victim], current_index, *sampled[victim + 1:])))


def _logical_epochs(
    context: BrowseContext,
    frames: tuple[DisplayFrameKey, ...],
) -> tuple[float, ...] | None:
    """Project complete scalar-catalog time provenance or no time axis."""

    catalog = context.scalar_catalog
    if type(catalog) is not FrameScalarCatalog:
        return None
    values: list[float] = []
    for frame in frames:
        row = catalog.row(frame.local_frame_label)
        value = None if row is None else row.metadata_numeric.get("epoch")
        if type(value) is not float or not math.isfinite(value):
            return None
        values.append(value)
    return tuple(values)


def _admit_scope(
    context: object,
    selection: object,
    navigation: object,
    preferences: object,
    owned_frame_by_id: object,
    *,
    current_selection: object,
    current_navigation: object,
    was_waterfall_active: object,
) -> tuple[str, ScientificPlotOptions]:
    if (
        type(context) is not BrowseContext
        or type(selection) is not DisplaySelection
        or selection is not current_selection
        or type(navigation) is not FrameNavigationProjection
        or navigation is not current_navigation
        or type(preferences) is not ScientificPreferences
        or type(preferences.plot_options) is not ScientificPlotOptions
        or type(owned_frame_by_id) is not dict
        or type(was_waterfall_active) is not bool
        or selection.kind is not ContextKind.BROWSE
        or type(selection.display_generation) is not int
        or selection.display_generation < 1
        or not context.loaded
        or context.invalidated
        or context.released
        or context.commit_gate.cancelled
        or not selection.names(context)
        or selection.owner != context.hydration_owner
        or type(context.frame_ids) is not tuple
        or context.frame_ids is not context.loaded_labels
        or not context.frame_ids
        or any(
            type(label) is not int or label < 0
            for label in context.frame_ids
        )
        or any(
            left >= right
            for left, right in zip(
                context.frame_ids,
                context.frame_ids[1:],
            )
        )
    ):
        raise _Refused("Browse 1-D planning scope is malformed or stale")
    plot_mode = preferences.plot_mode
    options = preferences.plot_options
    if (
        type(plot_mode) is not str
        or plot_mode not in _PLOT_MODES
        or type(options.waterfall_start) is not int
        or options.waterfall_start < 1
        or type(options.waterfall_stop) is not int
        or options.waterfall_stop < 0
        or type(options.waterfall_step) is not int
        or options.waterfall_step < 1
    ):
        raise _Refused("Browse 1-D plotting policy is malformed")
    frames = navigation.frames
    if (
        type(frames) is not tuple
        or len(frames) != len(context.frame_ids)
        or len(owned_frame_by_id) != len(frames)
        or navigation.current is None
        or type(navigation.selected) is not tuple
        or not navigation.selected
    ):
        raise _Refused("Browse 1-D navigation is incomplete")
    run_identity = frames[0].run_identity
    for ordinal, frame in enumerate(frames, 1):
        if (
            type(frame) is not DisplayFrameKey
            or owned_frame_by_id.get(id(frame)) is not frame
            or frame.run_identity is not run_identity
            or frame.source_scan != context.scan_key
            or frame.artifact != context.requested_path
            or type(frame.work_ordinal) is not int
            or frame.work_ordinal != ordinal
            or type(frame.local_frame_label) is not int
            or context.frame_ids[ordinal - 1] != frame.local_frame_label
        ):
            raise _Refused("Browse 1-D navigation changed ownership or order")
    if owned_frame_by_id.get(id(navigation.current)) is not navigation.current:
        raise _Refused("Browse 1-D current frame is foreign")
    previous = 0
    selected_ids: set[int] = set()
    for frame in navigation.selected:
        if (
            type(frame) is not DisplayFrameKey
            or id(frame) in selected_ids
            or owned_frame_by_id.get(id(frame)) is not frame
            or frame.work_ordinal <= previous
        ):
            raise _Refused("Browse 1-D selection is foreign or out of order")
        selected_ids.add(id(frame))
        previous = frame.work_ordinal
    return plot_mode, options


def plan_browse_1d_targets(
    context: object,
    selection: object,
    navigation: object,
    preferences: object,
    owned_frame_by_id: object,
    *,
    current_selection: object,
    current_navigation: object,
    was_waterfall_active: object,
) -> Browse1DTargetPlanOutcome:
    """Plan exact cache targets without I/O, mutation, sorting, or fallback."""

    try:
        plot_mode, options = _admit_scope(
            context,
            selection,
            navigation,
            preferences,
            owned_frame_by_id,
            current_selection=current_selection,
            current_navigation=current_navigation,
            was_waterfall_active=was_waterfall_active,
        )
        stacked = tuple(navigation.selected)
        waterfall_active = waterfall_should_be_active(
            plot_mode,
            len(stacked),
            was_active=was_waterfall_active,
        )
        # Single replaces on an ordinary click, but explicit modifier
        # membership is still authoritative, just as in acquisition display.
        if plot_mode in {"Single", "Overlay", "Waterfall"}:
            logical = stacked
        else:
            logical = (navigation.current,)
        candidates = logical
        positions = tuple(range(1, len(logical) + 1))
        stacked_options_applied = (
            plot_mode in {"Overlay", "Waterfall"}
            or plot_mode == "Single" and len(logical) > 1
        )
        if stacked_options_applied:
            start = options.waterfall_start - 1
            stop = options.waterfall_stop or None
            step = options.waterfall_step
            candidates = candidates[start:stop:step]
            positions = positions[start:stop:step]
        if not candidates:
            raise _Refused("Browse 1-D waterfall options selected no rows")
        sampled = (
            _sample_indices(
                len(candidates), MAX_BROWSE_1D_DISPLAY_TARGETS,
            )
            if waterfall_active
            else tuple(range(len(candidates)))
        )
        current_index = next(
            (
                index
                for index, frame in enumerate(candidates)
                if frame is navigation.current
            ),
            None,
        )
        sampled = _include_current_index(sampled, current_index)
        targets = tuple(candidates[index] for index in sampled)
        logical_positions = tuple(positions[index] for index in sampled)
        plan = Browse1DTargetPlan(
            selection,
            navigation,
            plot_mode,
            waterfall_active,
            stacked_options_applied,
            logical,
            targets,
            logical_positions,
            _logical_epochs(context, logical),
        )
        return Browse1DTargetPlanOutcome(
            Browse1DTargetPlanStatus.PLANNED,
            plan,
        )
    except BaseException as error:
        return Browse1DTargetPlanOutcome(
            Browse1DTargetPlanStatus.REFUSED,
            diagnostic=_diagnostic(error),
        )


__all__ = [
    "Browse1DRuntimeProjection",
    "Browse1DTargetPlan",
    "Browse1DTargetPlanOutcome",
    "Browse1DTargetPlanStatus",
    "MAX_BROWSE_1D_DETACHED_ROOTS",
    "MAX_BROWSE_1D_DISPLAY_TARGETS",
    "plan_browse_1d_targets",
]
