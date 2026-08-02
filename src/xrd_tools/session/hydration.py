"""Import-pure values for one exact hydration read and presentation."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
__all__ = ["HydrationPurpose", "HydrationOutcome", "HydrationScope", "HydrationReadKey", "HydrationToken", "HydrationCompletion", "normalize_hydration_purpose"]
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
_PURPOSE_ALIASES = {"1d": HydrationPurpose.ONE_D, "2d": HydrationPurpose.PREVIEW, "preview": HydrationPurpose.PREVIEW, "full": HydrationPurpose.FULL, "raw": HydrationPurpose.FULL}
def normalize_hydration_purpose(value) -> HydrationPurpose:
    """The sole legacy spelling adapter."""
    if type(value) is HydrationPurpose:
        return value
    if type(value) is not str:
        raise TypeError("hydration purpose must be an enum or string")
    try:
        return _PURPOSE_ALIASES[value]
    except KeyError:
        raise ValueError(f"unknown hydration purpose {value!r}") from None
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
    def __post_init__(self):
        if type(self.scope) is not HydrationScope or not self.scope.qualified:
            raise TypeError("read key requires a qualified HydrationScope")
        _text("artifact_identity", self.artifact_identity)
        if type(self.frame_identity) not in (int, str) or self.frame_identity == "":
            raise TypeError("frame_identity must be an exact nonempty value")
        if type(self.purpose) is not HydrationPurpose:
            raise TypeError("read key purpose must be a HydrationPurpose")
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
