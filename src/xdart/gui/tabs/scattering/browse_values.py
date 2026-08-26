"""Immutable, store-free values for one processed-scan browse load."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .events import (
    CleanupStatus,
    DetachedDiagnostic,
    detached_diagnostic_is_valid,
)


class BrowseLoadStatus(str, Enum):
    READY = "ready"
    REFUSED = "refused"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class BrowseLoadRequest:
    token: str
    load_generation: int
    source_path: str

    def __post_init__(self) -> None:
        if (
            type(self.token) is not str
            or not self.token
            or type(self.load_generation) is not int
            or self.load_generation < 1
            or type(self.source_path) is not str
            or not self.source_path
        ):
            raise TypeError("browse load request is invalid")

@dataclass(frozen=True, slots=True)
class BrowseLoadOutcome:
    request: BrowseLoadRequest
    status: BrowseLoadStatus
    detail: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.request) is not BrowseLoadRequest
            or type(self.status) is not BrowseLoadStatus
            or type(self.detail) is not str
        ):
            raise TypeError("browse load outcome is invalid")


@dataclass(frozen=True, slots=True)
class BrowseCleanupReceipt:
    """Exact, store-free cleanup truth for one Browse request."""

    request: BrowseLoadRequest | None
    cleanup_status: CleanupStatus
    cleanup_failures: tuple[DetachedDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.request is not None
            and type(self.request) is not BrowseLoadRequest
        ):
            raise TypeError("browse cleanup request is invalid")
        if type(self.cleanup_status) is not CleanupStatus:
            raise TypeError("browse cleanup status is invalid")
        if (
            type(self.cleanup_failures) is not tuple
            or not all(
                detached_diagnostic_is_valid(item)
                for item in self.cleanup_failures
            )
        ):
            raise TypeError("browse cleanup diagnostics are invalid")


def canonical_browse_scan_key(source_path: str) -> str:
    """Resolve one name through the registered headless format adapter."""

    if type(source_path) is not str or not source_path:
        return ""
    # Importing registry installs the built-in immutable format descriptors.
    from xrd_tools.sources import registry as _registry  # noqa: F401
    from xrd_tools.core.scan import SourceKind
    from xrd_tools.sources.adapters import explicit_source_owner

    path = Path(source_path)
    owner = explicit_source_owner(path, SourceKind.PROCESSED_NEXUS)
    return "" if owner is None else str(owner.scan_name(path) or "")


def canonical_browse_source_identity(view, artifact_path: str) -> str:
    """Name one persisted frame identically before and after hydration.

    Processed frame labels are commonly one-based while detector-source frame
    indices are zero-based.  The latter is the persisted member identity when
    present; the processed label is only the legacy fallback.
    """

    if type(artifact_path) is not str or not artifact_path:
        raise TypeError("browse artifact identity must be a nonempty string")
    source_path = getattr(view, "source_path", None) or artifact_path
    source_frame_index = getattr(view, "source_frame_index", None)
    member = (
        getattr(view, "label")
        if source_frame_index is None
        else source_frame_index
    )
    return f"{source_path}#{member}"


__all__ = [
    "BrowseLoadOutcome",
    "BrowseLoadRequest",
    "BrowseLoadStatus",
    "BrowseCleanupReceipt",
    "canonical_browse_scan_key",
    "canonical_browse_source_identity",
]
