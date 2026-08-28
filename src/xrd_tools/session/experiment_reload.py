"""Typed, fail-closed reload of exact current Experiment provenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from xrd_tools.core.provenance import read_provenance_from_handle
from xrd_tools.session.experiment_state import ExperimentState


class ReloadStatus(str, Enum):
    EXACT = "exact"
    ABSENT = "absent"
    OPEN_FAILURE = "open_failure"


@dataclass(frozen=True, slots=True)
class PersistedExperimentFacts:
    status: ReloadStatus
    experiment: ExperimentState | None = None
    content_fingerprint: str | None = None
    reason: str = ""


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


class ExperimentRecordReader:
    """Read only the current exact ``ExperimentState`` payload."""

    def read(
        self,
        path: str | Path,
        *,
        entry: str = "entry",
    ) -> PersistedExperimentFacts:
        import h5py

        try:
            handle = h5py.File(Path(path), "r")
        except OSError as exc:
            return PersistedExperimentFacts(
                ReloadStatus.OPEN_FAILURE,
                reason=f"{type(exc).__name__}: {exc}",
            )
        try:
            with handle:
                from xrd_tools.io.processed_scan_id import require_current_processed
                require_current_processed(handle, entry)
                return self._read_handle(handle, entry)
        except Exception as exc:
            return PersistedExperimentFacts(
                ReloadStatus.ABSENT,
                reason=(
                    "exact experiment decode failed closed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    @staticmethod
    def _read_handle(handle: Any, entry: str) -> PersistedExperimentFacts:
        if entry not in handle:
            return PersistedExperimentFacts(
                ReloadStatus.ABSENT,
                reason=f"entry {entry!r} is absent",
            )
        persisted = read_provenance_from_handle(handle, entry=entry)
        if not persisted:
            return PersistedExperimentFacts(
                ReloadStatus.ABSENT,
                reason="reduction provenance is absent",
            )
        config = _mapping(persisted.get("config"))
        if config is None:
            return PersistedExperimentFacts(
                ReloadStatus.ABSENT,
                reason="reduction config is not a mapping",
            )
        exact = _mapping(config.get("experiment"))
        if exact is None:
            return PersistedExperimentFacts(
                ReloadStatus.ABSENT,
                reason="exact experiment provenance is absent",
            )
        state = ExperimentState.from_provenance(exact)
        return PersistedExperimentFacts(
            ReloadStatus.EXACT,
            experiment=state,
            content_fingerprint=state.content_fingerprint,
        )


__all__ = [
    "ExperimentRecordReader",
    "PersistedExperimentFacts",
    "ReloadStatus",
]
