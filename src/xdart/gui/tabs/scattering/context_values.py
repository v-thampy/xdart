"""Store-free values and pure normalization for the context runtime."""

from __future__ import annotations

from dataclasses import dataclass

from xdart.modules.display_context import ContextKind, DisplaySelection

from .browse_values import BrowseLoadRequest
from .display_values import DisplayFrameKey, DisplayNavigationDelta
from .shell_values import FrameNavigationProjection


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


def _navigation_after_append(
    navigation: FrameNavigationProjection,
    delta: DisplayNavigationDelta,
    plot_mode: str,
    *,
    follow_latest: bool = True,
) -> tuple[
    tuple[DisplayFrameKey, ...],
    DisplayFrameKey,
    tuple[DisplayFrameKey, ...],
]:
    """Apply one accepted live append while retaining exact multi-selection."""

    retired = {id(frame) for frame in delta.retired}
    frames = tuple(
        frame
        for frame in navigation.frames
        if id(frame) not in retired
    )
    if not any(frame is delta.appended for frame in frames):
        frames = (*frames, delta.appended)
    if not follow_latest:
        current = (
            navigation.current
            if any(frame is navigation.current for frame in frames)
            else delta.appended
        )
        selected = tuple(
            frame
            for frame in navigation.selected
            if any(candidate is frame for candidate in frames)
        )
        if not selected:
            selected = (current,)
        return frames, current, selected
    if plot_mode not in {"Overlay", "Waterfall", "Average", "Sum"}:
        return frames, delta.appended, (delta.appended,)
    frame_identities = {id(frame) for frame in frames}
    selected_identities = {
        id(frame)
        for frame in navigation.selected
        if id(frame) in frame_identities
    }
    selected_identities.add(id(delta.appended))
    selected = tuple(
        frame for frame in frames if id(frame) in selected_identities
    )
    return frames, delta.appended, selected
