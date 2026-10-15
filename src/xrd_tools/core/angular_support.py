"""Angular support of a native radial profile: the values every owner shares.

For each radial bin of an I(q) or I(2θ) profile, angular support is how many degrees of the profile's requested
azimuthal window the detector's unmasked pixels actually reach, averaged over the bin (the bin mean). It describes
geometry and the geometric mask only: value-based exclusions (thresholds, saturation, invalid flat or dark pixels)
never reduce it, and it is not a pixel count. A profile without support carries a named reason instead; unknown is
never zero.

This module is import-light (numpy and the standard library only). The kernel that computes support lives in
:mod:`xrd_tools.analysis.angular_support`; the saved-record codec in :mod:`xrd_tools.io.angular_support`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

__all__ = [
    "AngularSupportProfile",
    "REASON_TEXT",
    "SUPPORT_CONVENTIONS",
    "SUPPORT_METRIC",
    "SupportOrigin",
    "SupportReason",
]

#: The metric of every saved and computed profile: the mean over each native radial bin.
SUPPORT_METRIC = "bin_mean_v1"
#: Detector χ (Standard and Stitch) or the grazing-incidence χ_GI of Int's total-q profile.
SUPPORT_CONVENTIONS = ("detector_chi", "gi_chi")


class SupportOrigin(str, Enum):
    """Where a profile's numbers came from: the Run that made the result, its saved record, or a reconstruction."""

    RUN = "run"
    SAVED = "saved"
    RECONSTRUCTED = "reconstructed"


class SupportReason(str, Enum):
    """Why a profile has no angular support. The values are persisted; never rename one."""

    NOT_RADIAL_PROFILE = "not_radial_profile"
    UNSUPPORTED_UNIT = "unsupported_unit"
    DEGENERATE_WINDOW = "degenerate_window"
    WINDOW_OUTSIDE_DOMAIN = "window_outside_domain"
    WINDOW_TRUNCATED = "window_truncated"
    UNQUALIFIED_GEOMETRY = "unqualified_geometry"
    XU_GEOMETRY_UNQUALIFIED = "xu_geometry_unqualified"
    OVER_BUDGET = "over_budget"
    AVERAGE_RESULT = "average_result"
    AGGREGATE_TRACE = "aggregate_trace"
    DIAGNOSTIC_TRACE = "diagnostic_trace"
    NO_RECORD = "no_record"
    NO_SAVED_SUPPORT = "no_saved_support"
    STALE_SAVED_SUPPORT = "stale_saved_support"
    MALFORMED_SAVED_SUPPORT = "malformed_saved_support"
    LEGACY_INSUFFICIENT = "legacy_insufficient"
    LEGACY_REINTEGRATE = "legacy_reintegrate"
    LEGACY_STITCH = "legacy_stitch"
    PENDING = "pending"
    #: “Record angular coverage” was off when processed; saved Standard Int may qualify for display reconstruction.
    RECORDING_DISABLED = "recording_disabled"


#: One sentence per reason, shown wherever a profile has no support (tooltips, the strip, hover).
REASON_TEXT: Mapping[SupportReason, str] = MappingProxyType({
    SupportReason.NOT_RADIAL_PROFILE: (
        "Not a native radial profile: angular coverage describes I(q) and I(2θ) integrations only."),
    SupportReason.UNSUPPORTED_UNIT: "Angular coverage is available for q and 2θ axes only.",
    SupportReason.DEGENERATE_WINDOW: "The requested χ window has no width.",
    SupportReason.WINDOW_OUTSIDE_DOMAIN: "The requested χ_GI window lies outside −180° to 180°.",
    SupportReason.WINDOW_TRUNCATED: (
        "The integration did not admit part of the requested χ window that the detector reaches "
        "(the window crosses χ = ±180°)."),
    SupportReason.UNQUALIFIED_GEOMETRY: (
        "This detector geometry (parallax, curved, distorted or non-uniform pixels) is not qualified for "
        "angular coverage."),
    SupportReason.XU_GEOMETRY_UNQUALIFIED: (
        "The xrayutilities backend's pixel corners could not be qualified for angular coverage."),
    SupportReason.OVER_BUDGET: "Angular coverage was not computed: it would exceed this Run's time budget.",
    SupportReason.AVERAGE_RESULT: "An averaged result has no single detector footprint.",
    SupportReason.AGGREGATE_TRACE: "A sum, average or waterfall of several profiles has no single angular coverage.",
    SupportReason.DIAGNOSTIC_TRACE: "Pixel-count and normalization diagnostics have no angular coverage.",
    SupportReason.NO_RECORD: "This file has no processing record to derive angular coverage from.",
    SupportReason.NO_SAVED_SUPPORT: "This output saved no angular coverage for this frame.",
    SupportReason.STALE_SAVED_SUPPORT: "The saved angular coverage no longer matches this frame's saved profile.",
    SupportReason.MALFORMED_SAVED_SUPPORT: "The saved angular coverage is malformed.",
    SupportReason.LEGACY_INSUFFICIENT: (
        "This older output's record is not sufficient to reconstruct angular coverage."),
    SupportReason.LEGACY_REINTEGRATE: "Older Reintegrate result: reprocess it to record angular coverage.",
    SupportReason.LEGACY_STITCH: "Older Stitch result: reprocess it to record angular coverage.",
    SupportReason.PENDING: "Computing angular coverage…",
    SupportReason.RECORDING_DISABLED: (
        "Angular coverage was not recorded: “Record angular coverage” was off when this result was processed."),
})


def _frozen(value: Any) -> Any:
    """A detached, immutable copy of a JSON-like provenance value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"angular support provenance holds a non-JSON value: {type(value).__name__}")


def _interval(value: Any, name: str) -> tuple[float, float]:
    try:
        lo, hi = (float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a pair of numbers") from error
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{name} must be finite")
    return lo, hi


@dataclass(frozen=True, slots=True)
class AngularSupportProfile:
    """One native profile's angular support, or the named reason it has none.

    ``supported_deg`` holds degrees of the requested window reached in each radial bin (a read-only float64 copy),
    ``window_deg`` is that window's width (360 for no window), so ``fraction`` is the 0–1 share the strip shows.
    ``effective_window_deg`` lists the χ intervals the reducer actually admitted, in its own raw frame.
    ``bin_edges`` are the first and last native radial bin edges; the bins are uniform between them.
    """

    origin: SupportOrigin
    supported_deg: np.ndarray | None = None
    window_deg: float = math.nan
    requested_window_deg: tuple[float, float] | None = None
    effective_window_deg: tuple[tuple[float, float], ...] = ()
    convention: str = ""
    radial_unit: str = ""
    bin_edges: tuple[float, float] | None = None
    reason: SupportReason | None = None
    detail: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metric: str = SUPPORT_METRIC

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", SupportOrigin(self.origin))
        if self.reason is not None:
            object.__setattr__(self, "reason", SupportReason(self.reason))
        if (self.supported_deg is None) == (self.reason is None):
            raise ValueError("an angular support profile has either values or a reason, never both or neither")
        if self.metric != SUPPORT_METRIC:
            raise ValueError(f"unknown angular support metric {self.metric!r}")
        if type(self.detail) is not str:
            raise TypeError("angular support detail must be text")
        if self.requested_window_deg is not None:
            object.__setattr__(self, "requested_window_deg", _interval(self.requested_window_deg,
                                                                       "requested window"))
        object.__setattr__(self, "effective_window_deg", tuple(
            _interval(item, "effective window") for item in self.effective_window_deg))
        if self.bin_edges is not None:
            object.__setattr__(self, "bin_edges", _interval(self.bin_edges, "bin edges"))
        object.__setattr__(self, "provenance", _frozen(dict(self.provenance)))
        if self.supported_deg is None:
            return
        values = self.supported_deg
        if not (isinstance(values, np.ndarray) and values.dtype == np.float64 and values.base is None
                and values.flags.c_contiguous and not values.flags.writeable):
            values = np.array(values, dtype=np.float64, copy=True)  # a frozen float64 array of its own is adopted
        if values.ndim != 1 or not values.size:
            raise ValueError("angular support must be a nonempty 1-D array")
        window = float(self.window_deg)
        if not (math.isfinite(window) and 0.0 < window <= 360.0):
            raise ValueError("angular support needs a requested window width in (0, 360] degrees")
        if self.convention not in SUPPORT_CONVENTIONS:
            raise ValueError(f"unknown angular support convention {self.convention!r}")
        if not self.radial_unit or self.bin_edges is None:
            raise ValueError("angular support needs its native radial unit and bin edges")
        finite = values[np.isfinite(values)]
        if finite.size and (finite.min() < -1e-9 or finite.max() > window + 1e-6):
            raise ValueError("angular support lies outside its requested window")
        values.setflags(write=False)
        object.__setattr__(self, "supported_deg", values)
        object.__setattr__(self, "window_deg", window)

    @property
    def available(self) -> bool:
        return self.supported_deg is not None

    @property
    def fraction(self) -> np.ndarray | None:
        """The share (0–1) of the requested window reached in each bin, or None without support."""
        if self.supported_deg is None:
            return None
        share = self.supported_deg / self.window_deg
        share.setflags(write=False)
        return share

    @property
    def edges(self) -> np.ndarray | None:
        """All native bin edges (uniform between ``bin_edges``), or None without support."""
        if self.supported_deg is None or self.bin_edges is None:
            return None
        edges = np.linspace(self.bin_edges[0], self.bin_edges[1], self.supported_deg.size + 1)
        edges.setflags(write=False)
        return edges

    @property
    def text(self) -> str:
        """The reason's sentence, or ``""`` for an available profile."""
        if self.reason is None:
            return ""
        return REASON_TEXT[self.reason] + (f" ({self.detail})" if self.detail else "")

    def with_origin(self, origin: SupportOrigin) -> "AngularSupportProfile":
        """The same profile relabelled as saved after its record round-trip; never the other way."""
        return AngularSupportProfile(
            origin=origin, supported_deg=self.supported_deg, window_deg=self.window_deg,
            requested_window_deg=self.requested_window_deg, effective_window_deg=self.effective_window_deg,
            convention=self.convention, radial_unit=self.radial_unit, bin_edges=self.bin_edges,
            reason=self.reason, detail=self.detail, provenance=self.provenance, metric=self.metric)

    @classmethod
    def unavailable(cls, reason: SupportReason, *, origin: SupportOrigin = SupportOrigin.RUN, detail: str = "",
                    **facts: Any) -> "AngularSupportProfile":
        """A profile carrying only its reason (and any known facts, such as the radial unit)."""
        return cls(origin=origin, reason=SupportReason(reason), detail=detail, **facts)

    def metadata(self) -> dict[str, Any]:
        """The JSON-able facts of this profile (everything except the per-bin values)."""
        return {
            "status": "available" if self.available else "unavailable",
            "reason": None if self.reason is None else self.reason.value,
            "detail": self.detail,
            "convention": self.convention,
            "radial_unit": self.radial_unit,
            "requested_window_deg": None if self.requested_window_deg is None else list(self.requested_window_deg),
            "effective_window_deg": [list(item) for item in self.effective_window_deg],
            "window_deg": None if not math.isfinite(self.window_deg) else self.window_deg,
            "bin_edges": None if self.bin_edges is None else list(self.bin_edges),
            "metric": self.metric,
            "provenance": json.loads(json.dumps(_thaw(self.provenance))),
        }


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
