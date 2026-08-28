"""Import-pure values for one exact hydration read and presentation."""
from __future__ import annotations
import os
from threading import Lock
from dataclasses import dataclass
from enum import StrEnum
__all__ = ["HydrationPurpose", "HydrationOutcome", "HydrationScope", "HydrationReadKey", "HydrationToken", "HydrationCompletion"]
class HydrationPurpose(StrEnum):
    ONE_D = "1d"
    PREVIEW = "2d"
    FULL = "full"
class HydrationOutcome(StrEnum):
    HYDRATED = "hydrated"
    ALREADY_RESIDENT = "already_resident"
    SUPERSEDED = "superseded"
    OWNER_MISMATCH = "owner_mismatch"
    FAILED = "failed"
    CANCELLED = "cancelled"
def _text(name, value, *, allow_empty=False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise TypeError(f"{name} must be an exact nonempty string")
    return value
@dataclass(frozen=True, slots=True)
class HydrationScope:
    context_token: str
    scan_key: str
    source: str
    epoch: int
    def __post_init__(self):
        for name in ("context_token", "scan_key", "source"):
            _text(name, getattr(self, name), allow_empty=True)
        if type(self.epoch) is not int or self.epoch < 0:
            raise TypeError("epoch must be an exact nonnegative integer")
    @property
    def qualified(self) -> bool:
        return bool(self.context_token and self.scan_key and self.source and self.epoch > 0)
@dataclass(frozen=True, slots=True)
class HydrationReadKey:
    scope: HydrationScope
    artifact_identity: str
    frame_identity: int | str
    purpose: HydrationPurpose
    source_root: str | None = None
    def __post_init__(self):
        if type(self.scope) is not HydrationScope or not self.scope.qualified:
            raise TypeError("read key requires a qualified HydrationScope")
        _text("artifact_identity", self.artifact_identity)
        if type(self.frame_identity) not in (int, str) or self.frame_identity == "":
            raise TypeError("frame_identity must be an exact nonempty value")
        if type(self.purpose) is not HydrationPurpose:
            raise TypeError("read key purpose must be a HydrationPurpose")
        if self.source_root is not None and (
            type(self.source_root) is not str
            or not self.source_root
            or not os.path.isabs(self.source_root)
            or os.path.normcase(os.path.normpath(self.source_root))
            != self.source_root
        ):
            raise TypeError(
                "read key source root must be normalized absolute text or None"
            )
@dataclass(frozen=True, slots=True)
class HydrationToken:
    read_key: HydrationReadKey
    presentation_generation: int
    def __post_init__(self):
        if type(self.read_key) is not HydrationReadKey:
            raise TypeError("read_key must be a HydrationReadKey")
        if type(self.presentation_generation) is not int or self.presentation_generation < 0:
            raise TypeError("presentation_generation must be nonnegative")
@dataclass(frozen=True, slots=True)
class HydrationCompletion:
    token: HydrationToken
    outcome: HydrationOutcome
    diagnostic: str | None = None
    def __post_init__(self):
        if type(self.token) is not HydrationToken:
            raise TypeError("token must be a HydrationToken")
        if type(self.outcome) is not HydrationOutcome:
            raise TypeError("outcome must be a HydrationOutcome")
        if self.diagnostic is not None and type(self.diagnostic) is not str:
            raise TypeError("diagnostic must be a detached string or None")


@dataclass(frozen=True, slots=True, eq=False)
class _CheckpointHydrationToken:
    """Private read capability for one verified live-writer checkpoint."""
    artifact_identity: str
    run_lineage: object
    checkpoint_identity: object
    generation: int
    def __post_init__(self):
        if (type(self.artifact_identity) is not str or not self.artifact_identity
                or self.run_lineage is None or self.checkpoint_identity is None
                or type(self.generation) is not int or self.generation < 1):
            raise TypeError("checkpoint hydration token is malformed")


class _CheckpointHydrationGate:
    """Linearize bounded display insertion against the next writer mutation."""
    __slots__ = ("_lock", "_artifact", "_lineage", "_generation", "_token")
    def __init__(self):
        self._lock = Lock(); self._artifact = None; self._lineage = None
        self._generation = 0; self._token = None
    def bind(self, artifact_identity, run_lineage) -> None:
        if type(artifact_identity) is not str or not artifact_identity or run_lineage is None:
            raise TypeError("checkpoint hydration lineage is malformed")
        with self._lock:
            if self._artifact is None:
                self._artifact, self._lineage = artifact_identity, run_lineage
            elif self._artifact != artifact_identity or self._lineage is not run_lineage:
                raise RuntimeError("checkpoint hydration lineage cannot change")
    @property
    def bound(self) -> bool:
        with self._lock: return self._artifact is not None
    def authorize(self, checkpoint_identity):
        with self._lock:
            if self._artifact is None or checkpoint_identity is None: return None
            self._generation += 1
            self._token = _CheckpointHydrationToken(
                self._artifact, self._lineage, checkpoint_identity, self._generation)
            return self._token
    def revoke(self) -> None:
        with self._lock:
            self._generation += 1; self._token = None
    def authority(self):
        with self._lock: return self._token, self if self._artifact is not None else None
    def enter(self, token) -> bool:
        self._lock.acquire()
        if token is not self._token or type(token) is not _CheckpointHydrationToken:
            self._lock.release(); return False
        return True
    def leave(self) -> None:
        try: self._lock.release()
        except RuntimeError: pass
