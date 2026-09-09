"""Store-free values and pure normalization for the context runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xdart.modules.display_context import (
    ContextKind, DisplaySelection, HydrationRequest,
)

from .browse_values import BrowseLoadRequest
from .display_values import (
    DisplayFrameKey, StandardDisplayPayload,
)
from .shell_values import FrameNavigationProjection


class BrowseMissReason(Enum):
    """Finite reasons the current Browse frame cannot hydrate this pass."""

    CLOSED = "closed"
    RETIRED = "retired"
    FOREIGN = "foreign"
    UNRESOLVABLE = "unresolvable"


@dataclass(frozen=True, slots=True)
class QualifiedPayload:
    """The current Browse frame resolved to one complete display payload."""

    payload: StandardDisplayPayload

    def __post_init__(self) -> None:
        if type(self.payload) is not StandardDisplayPayload:
            raise TypeError("QualifiedPayload requires one exact payload")


@dataclass(frozen=True, slots=True)
class HydrationEligibleMiss:
    """The current Browse frame misses; carries THE submit-ready request."""

    request: HydrationRequest

    def __post_init__(self) -> None:
        if type(self.request) is not HydrationRequest:
            raise TypeError("HydrationEligibleMiss requires one exact request")


@dataclass(frozen=True, slots=True)
class TerminalMiss:
    """The current Browse frame cannot hydrate in this pass at all."""

    reason: BrowseMissReason

    def __post_init__(self) -> None:
        if type(self.reason) is not BrowseMissReason:
            raise TypeError("TerminalMiss requires one finite reason")


BrowseProjectionResolution = (
    QualifiedPayload | HydrationEligibleMiss | TerminalMiss)


@dataclass(frozen=True, slots=True)
class _BrowseProjectionPass:
    """One exact Browse projection pass: what was resolved, for which view.
    Residency is DERIVED from the typed resolution while the exact
    selection/current/generation still qualify — no resident boolean."""

    selection: DisplaySelection
    current: DisplayFrameKey
    generation: int
    resolution: BrowseProjectionResolution

    def __post_init__(self) -> None:
        if (
            type(self.selection) is not DisplaySelection
            or self.selection.kind is not ContextKind.BROWSE
            or type(self.current) is not DisplayFrameKey
            or type(self.generation) is not int
            or type(self.resolution) not in (
                QualifiedPayload, HydrationEligibleMiss, TerminalMiss,
            )
        ):
            raise TypeError("Browse projection pass is invalid")


@dataclass(frozen=True, slots=True)
class _PendingBrowseReplacement:
    """Exact C request composed with B's immutable public values."""

    request: BrowseLoadRequest
    selection: DisplaySelection
    navigation: FrameNavigationProjection

    def __post_init__(self) -> None:
        if (
            type(self.request) is not BrowseLoadRequest
            or type(self.selection) is not DisplaySelection
            or self.selection.kind is not ContextKind.BROWSE
            or type(self.navigation) is not FrameNavigationProjection
        ):
            raise TypeError("pending Browse replacement is invalid")


def _normalized_navigation(
    frames: tuple[DisplayFrameKey, ...],
    current: DisplayFrameKey | None,
    selected: tuple[DisplayFrameKey, ...],
    *,
    prefer_last: bool,
) -> tuple[
    tuple[DisplayFrameKey, ...],
    DisplayFrameKey | None,
    tuple[DisplayFrameKey, ...],
]:
    canonical = {id(frame): frame for frame in frames}
    if current is None or canonical.get(id(current)) is not current:
        current = (
            None
            if not frames
            else frames[-1] if prefer_last else frames[0]
        )
    selected = tuple(
        frame
        for frame in selected
        if canonical.get(id(frame)) is frame
    )
    if current is None:
        selected = ()
    elif not selected:
        selected = (current,)
    return frames, current, selected
