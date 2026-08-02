"""Owner-bound Browse repaint, polling, terminal, and release lifecycle."""

from __future__ import annotations

from xdart.modules.display_context import BrowseContext

from .browse_hydration import _BrowseHydrationOwner
from .browse_values import (
    BrowseCleanupReceipt,
    BrowseLoadRequest,
)
from .display_runtime import DetectorHydrationOutcome
from .events import CleanupStatus


def _current_browse_hydration(runtime, owner):
    browse = runtime.browse_context
    selection = runtime.selection
    if (
        type(owner) is not _BrowseHydrationOwner
        or type(browse) is not BrowseContext
        or browse.invalidated
        or browse.released
        or selection is None
        or not selection.names(browse)
        or not owner.names(browse)
    ):
        return None
    return owner


def cold_browse_detector_outcome(
    browse: BrowseContext, label, owner
) -> DetectorHydrationOutcome | None:
    """Return the bound owner's exact terminal detector fact."""

    return (
        owner.detector_outcome(browse, label)
        if type(owner) is _BrowseHydrationOwner
        else None
    )


def browse_preview_repaint_ready(runtime, owner) -> bool:
    """Consume one cold-B completion wake for the exact current selection."""

    bound = _current_browse_hydration(runtime, owner)
    return bound is not None and bound.consume_repaint()


def browse_preview_polling_needed(runtime, owner) -> bool:
    """Whether a cold-B worker or unpainted completion still needs a tick."""

    bound = _current_browse_hydration(runtime, owner)
    return bound is not None and bound.polling_needed()


def qualified_event_frame(runtime, event):
    """Resolve one DISPLAY_READY event to the exact owned frame it repaints:
    an acquisition event by its exact ``frame_key``; a key-less Browse
    repaint hint by the exact CURRENT browse selection when the live browse
    artifact matches."""
    from xdart.modules.display_context import ContextKind

    selection = runtime.selection
    frame = getattr(event, "frame_key", None)
    if frame is not None:
        if (
            selection is not None
            and selection.kind is ContextKind.ACQUISITION
            and runtime.owns_frame(frame)
        ):
            return frame
        return None
    browse = runtime.browse_context
    current = runtime.navigation.current
    if (
        selection is None
        or selection.kind is not ContextKind.BROWSE
        or browse is None
        or browse.invalidated
        or browse.released
        or getattr(event, "artifact", None) != browse.requested_path
    ):
        return None
    return current


def release_browse(
    loader, browse: BrowseContext, owner
) -> BrowseCleanupReceipt:
    """Release only through the one exact controller-bound owner."""

    if type(owner) is _BrowseHydrationOwner:
        return owner.release(loader, browse)
    request = (
        browse.load_request
        if type(browse) is BrowseContext
        and type(browse.load_request) is BrowseLoadRequest
        else None
    )
    return BrowseCleanupReceipt(request, CleanupStatus.CLEANUP_PENDING)


__all__ = [
    "browse_preview_polling_needed",
    "browse_preview_repaint_ready",
    "cold_browse_detector_outcome",
    "qualified_event_frame",
    "release_browse",
]
