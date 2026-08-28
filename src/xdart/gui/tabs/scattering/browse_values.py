"""Immutable, store-free values for one processed-scan browse load."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
from pathlib import Path

from xdart.modules.display_context import (
    BrowseContext,
    ContextKind,
    DisplaySelection,
)
from xrd_tools.io.output_transaction import StreamTerminal
from xrd_tools.io.output_transaction import TargetSnapshot

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
class BrowseLoadTiming:
    """One request-owned aggregate of the terminal Browse worker tail."""

    canonical_path: str
    seal_mode: str
    initial_seal_s: float
    scan_open_s: float
    record_iteration_s: float
    record_count: int
    presentation_read_s: float
    final_seal_s: float
    context_build_s: float
    worker_total_s: float

    def __post_init__(self) -> None:
        durations = (
            self.initial_seal_s,
            self.scan_open_s,
            self.record_iteration_s,
            self.presentation_read_s,
            self.final_seal_s,
            self.context_build_s,
            self.worker_total_s,
        )
        if (
            type(self.canonical_path) is not str
            or not self.canonical_path
            or self.seal_mode not in {"terminal", "snapshot"}
            or type(self.record_count) is not int
            or self.record_count < 0
            or not all(
                type(value) is float
                and math.isfinite(value)
                and value >= 0.0
                for value in durations
            )
        ):
            raise TypeError("browse load timing is invalid")


@dataclass(frozen=True, slots=True)
class BrowseLoadRequest:
    token: str
    load_generation: int
    source_path: str
    terminal_commit_identity: StreamTerminal | None = None
    source_root: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.token) is not str
            or not self.token
            or type(self.load_generation) is not int
            or self.load_generation < 1
            or type(self.source_path) is not str
            or not self.source_path
            or (
                self.source_root is not None
                and (
                    type(self.source_root) is not str
                    or not self.source_root
                    or not os.path.isabs(self.source_root)
                    or os.path.normcase(os.path.normpath(self.source_root))
                    != self.source_root
                )
            )
            or (
                self.terminal_commit_identity is not None
                and type(self.terminal_commit_identity) is not StreamTerminal
            )
        ):
            raise TypeError("browse load request is invalid")


@dataclass(frozen=True, slots=True)
class LoadedBrowseCapture:
    """One exact stable loaded-Browse context and its persisted target facts."""

    context: BrowseContext
    request: BrowseLoadRequest
    selection: DisplaySelection
    target: str
    entry: str
    target_snapshot: TargetSnapshot
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        valid = (
            type(self.context) is BrowseContext
            and type(self.request) is BrowseLoadRequest
            and type(self.selection) is DisplaySelection
            and self.selection.kind is ContextKind.BROWSE
            and type(self.target) is str
            and bool(self.target)
            and self.request.source_path == self.target
            and type(self.entry) is str
            and bool(self.entry)
            and type(self.target_snapshot) is TargetSnapshot
            and self.target_snapshot.exists
            and type(self.labels) is tuple
            and bool(self.labels)
            and self.labels == tuple(sorted(set(self.labels)))
            and all(type(label) is int and label >= 0 for label in self.labels)
        )
        if not valid:
            raise ValueError("loaded Browse capture is invalid")

    def is_exactly(self, other: object) -> bool:
        """Compare frozen facts while requiring every live owner by identity."""

        return bool(
            type(other) is LoadedBrowseCapture
            and other.context is self.context
            and other.request is self.request
            and other.selection is self.selection
            and other.target == self.target
            and other.entry == self.entry
            and other.target_snapshot == self.target_snapshot
            and other.labels == self.labels
        )

@dataclass(frozen=True, slots=True)
class BrowseLoadOutcome:
    request: BrowseLoadRequest
    status: BrowseLoadStatus
    detail: str = ""
    timing: BrowseLoadTiming | None = None

    def __post_init__(self) -> None:
        if (
            type(self.request) is not BrowseLoadRequest
            or type(self.status) is not BrowseLoadStatus
            or type(self.detail) is not str
            or (
                self.timing is not None
                and type(self.timing) is not BrowseLoadTiming
            )
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


def canonical_browse_source_identity(
    view,
    artifact_path: str,
    *,
    source_base: str | None = None,
    source_root: str | None = None,
) -> str:
    """Name one persisted frame identically before and after hydration.

    Processed frame labels are commonly one-based while detector-source frame
    indices are zero-based.  The latter is the persisted member identity when
    present; the processed label is only the legacy fallback.
    """

    if type(artifact_path) is not str or not artifact_path:
        raise TypeError("browse artifact identity must be a nonempty string")
    for role, root in (("source base", source_base), ("source root", source_root)):
        if root is not None and (
            type(root) is not str
            or not root
            or not os.path.isabs(root)
            or os.path.normcase(os.path.normpath(root)) != root
        ):
            raise TypeError(f"browse {role} must be normalized absolute text or None")
    from xdart.modules.frame_publication import canonical_frame_source_identity

    # The selected Project root is authoritative for a moved tree.  Only when
    # none was selected may the record's authenticated source_base own the
    # relative locator.  Never guess from the artifact directory.
    return canonical_frame_source_identity(
        view,
        source_base=(source_root if source_root is not None else source_base),
        fallback_path=artifact_path,
    )


__all__ = [
    "BrowseLoadOutcome",
    "BrowseLoadRequest",
    "BrowseLoadStatus",
    "BrowseLoadTiming",
    "BrowseCleanupReceipt",
    "LoadedBrowseCapture",
    "canonical_browse_scan_key",
    "canonical_browse_source_identity",
]
