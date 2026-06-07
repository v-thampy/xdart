"""Typed headless analysis plans built on the public notebook APIs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Sequence

import numpy as np

from ssrl_xrd_tools.analysis.fitting import FitConfig, PhaseFitter, fit_peaks, fit_sequence
from ssrl_xrd_tools.analysis.strain import sin2psi_analysis
from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D, PONI
from ssrl_xrd_tools.core.scan import FrameSource
from ssrl_xrd_tools.integrate.multi import stitch_images
from ssrl_xrd_tools.rsm.pipeline import ScanInput, grid_scans_streaming, process_scan_from_nexus
from ssrl_xrd_tools.sources import ensure_frame_source


@dataclass(slots=True)
class AnalysisResult:
    """Small JSON-friendly envelope around an analysis payload."""

    kind: str
    payload: Any
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload_type": type(self.payload).__name__,
            "provenance": _json_safe(self.provenance),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True, slots=True)
class StitchPlan:
    """Plan for MultiGeometry stitching of a frame source."""

    base_poni: PONI | None = None
    rot1_key: str = "rot1"
    rot2_key: str | None = None
    monitor_key: str | None = None
    mode: str = "1d"
    npt_1d: int = 2000
    npt_rad_2d: int = 1500
    npt_azim_2d: int = 720
    unit: str = "q_A^-1"
    method: str = "BBox"
    radial_range: tuple[float, float] | None = None
    azimuth_range: tuple[float, float] | None = None
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    extra: dict[str, Any] = field(default_factory=dict)


def run_stitch(
    plan: StitchPlan,
    source: FrameSource,
    *,
    frame_indices: Sequence[int] | None = None,
) -> AnalysisResult:
    """Run MultiGeometry stitching over any headless frame source."""

    src = ensure_frame_source(source)
    labels = [int(i) for i in (frame_indices or src.frame_indices)]
    if not labels:
        raise ValueError("run_stitch requires at least one frame")
    base_poni = plan.base_poni or getattr(src, "poni", None)
    if base_poni is None:
        raise ValueError("StitchPlan.base_poni or source.poni is required")

    images = [np.asarray(src.load_frame(i), dtype=float) for i in labels]
    rot1 = _metadata_series(src, labels, plan.rot1_key)
    rot2 = (
        _metadata_series(src, labels, plan.rot2_key)
        if plan.rot2_key is not None else None
    )
    normalization = (
        _metadata_series(src, labels, plan.monitor_key)
        if plan.monitor_key is not None else None
    )
    payload = stitch_images(
        images,
        base_poni,
        rot1_angles=rot1,
        rot2_angles=rot2,
        mode=plan.mode,
        npt_1d=plan.npt_1d,
        npt_rad_2d=plan.npt_rad_2d,
        npt_azim_2d=plan.npt_azim_2d,
        unit=plan.unit,
        method=plan.method,
        radial_range=plan.radial_range,
        azimuth_range=plan.azimuth_range,
        mask=plan.mask,
        normalization=normalization,
        **plan.extra,
    )
    return AnalysisResult(
        kind="stitch",
        payload=payload,
        provenance={
            "plan": _plan_dict(plan),
            "source": getattr(src, "name", type(src).__name__),
            "frame_indices": labels,
        },
    )


@dataclass(frozen=True, slots=True)
class RSMPlan:
    """Plan for streaming reciprocal-space map gridding."""

    mapper: Any = field(repr=False, compare=False)
    diff_motors: tuple[str, ...] = ()
    bins: tuple[int, int, int] = (101, 101, 101)
    UB: np.ndarray | None = field(default=None, repr=False, compare=False)
    energy: float | None = None
    chunk_size: int = 8
    q_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] | None = None
    roi: tuple[int, int, int, int] | None = None
    static_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    scout_pad: float = 0.0


def run_rsm(plan: RSMPlan, source: FrameSource | Sequence[FrameSource]) -> AnalysisResult:
    """Run the streaming RSM pipeline for one source or a list of sources."""

    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        inputs = [
            ScanInput(scan=ensure_frame_source(scan), energy=plan.energy, UB=plan.UB, roi=plan.roi)
            for scan in source
        ]
        payload = grid_scans_streaming(
            plan.mapper,
            inputs,
            plan.diff_motors,
            plan.bins,
            chunk_size=plan.chunk_size,
            q_bounds=plan.q_bounds,
            static_mask=plan.static_mask,
            scout_pad=plan.scout_pad,
        )
        n_sources = len(inputs)
    else:
        src = ensure_frame_source(source)  # type: ignore[arg-type]
        payload = process_scan_from_nexus(
            src,
            plan.mapper,
            plan.diff_motors,
            plan.bins,
            UB=plan.UB,
            energy=plan.energy,
            chunk_size=plan.chunk_size,
            q_bounds=plan.q_bounds,
            roi=plan.roi,
            static_mask=plan.static_mask,
            scout_pad=plan.scout_pad,
        )
        n_sources = 1
    return AnalysisResult(
        kind="rsm",
        payload=payload,
        provenance={"plan": _plan_dict(plan), "n_sources": n_sources},
    )


@dataclass(frozen=True, slots=True)
class PeakFitPlan:
    positions: tuple[float, ...] | None = None
    model: str = "pseudovoigt"
    n_peaks: int | None = None
    background: str = "linear"
    sigma_init: float | Sequence[float] | None = None
    sigma_bounds: tuple[float, float] | None = None
    amplitude_init: float | Sequence[float] | None = None
    amplitude_bounds: tuple[float, float] | None = None
    center_bounds_delta: float | None = None
    fraction_init: float = 0.5
    fit_kwargs: dict[str, Any] = field(default_factory=dict)


def run_peak_fit(plan: PeakFitPlan, x: np.ndarray, y: np.ndarray) -> AnalysisResult:
    payload = fit_peaks(
        x,
        y,
        positions=None if plan.positions is None else list(plan.positions),
        model=plan.model,
        n_peaks=plan.n_peaks,
        background=plan.background,
        sigma_init=plan.sigma_init,
        sigma_bounds=plan.sigma_bounds,
        amplitude_init=plan.amplitude_init,
        amplitude_bounds=plan.amplitude_bounds,
        center_bounds_delta=plan.center_bounds_delta,
        fraction_init=plan.fraction_init,
        **plan.fit_kwargs,
    )
    return AnalysisResult(kind="peak_fit", payload=payload, provenance={"plan": _plan_dict(plan)})


@dataclass(frozen=True, slots=True)
class PhaseFitPlan:
    """Plan wrapper for phase-aware fitting."""

    config: FitConfig = field(default_factory=FitConfig)
    sequential: bool = False


def run_phase_fit(
    plan: PhaseFitPlan,
    patterns: Sequence[
        tuple[np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray | None]
        | IntegrationResult1D
    ],
    phases: list[Any],
    *,
    labels: Sequence[str] | None = None,
    progress_callback=None,
    fit_background_template: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
) -> AnalysisResult:
    normalized = [
        (p.radial, p.intensity, p.sigma)
        if isinstance(p, IntegrationResult1D) else p
        for p in patterns
    ]
    payload = fit_sequence(
        normalized,
        phases,
        plan.config,
        sequential=plan.sequential,
        labels=labels,
        progress_callback=progress_callback,
        fit_background_template=fit_background_template,
    )
    return AnalysisResult(kind="phase_fit", payload=payload, provenance={"plan": _plan_dict(plan)})


@dataclass(frozen=True, slots=True)
class Sin2PsiPlan:
    q_range: tuple[float, float]
    chi_centers: tuple[float, ...] | None = None
    chi_width: float = 5.0
    n_sectors: int | None = None
    chi_range: tuple[float, float] | None = None
    model: str = "pseudovoigt"
    background: str = "linear"
    sigma_init: float | None = None
    sigma_bounds: tuple[float, float] | None = None
    center_bounds_delta: float | None = None
    E: float | None = None
    nu: float | None = None


def run_sin2psi(plan: Sin2PsiPlan, result2d: IntegrationResult2D) -> AnalysisResult:
    payload = sin2psi_analysis(
        result2d,
        q_range=plan.q_range,
        chi_centers=None if plan.chi_centers is None else list(plan.chi_centers),
        chi_width=plan.chi_width,
        n_sectors=plan.n_sectors,
        chi_range=plan.chi_range,
        model=plan.model,
        background=plan.background,
        sigma_init=plan.sigma_init,
        sigma_bounds=plan.sigma_bounds,
        center_bounds_delta=plan.center_bounds_delta,
        E=plan.E,
        nu=plan.nu,
    )
    return AnalysisResult(kind="sin2psi", payload=payload, provenance={"plan": _plan_dict(plan)})


def make_phase_fitter(
    result: IntegrationResult1D | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray | None],
    **kwargs: Any,
) -> PhaseFitter:
    """Convenience factory that accepts either an IntegrationResult1D or arrays."""

    if isinstance(result, IntegrationResult1D):
        return PhaseFitter(result.radial, result.intensity, result.sigma, **kwargs)
    return PhaseFitter(*result, **kwargs)


def _metadata_series(source: FrameSource, labels: Sequence[int], key: str | None) -> np.ndarray:
    if key is None:
        raise ValueError("metadata key must not be None")
    motors = getattr(source, "motors", None)
    if motors and key in motors:
        values = np.asarray(motors[key], dtype=float)
        if values.shape[0] == len(getattr(source, "frame_indices")):
            row_of = {int(label): i for i, label in enumerate(source.frame_indices)}
            return np.asarray([values[row_of[int(label)]] for label in labels], dtype=float)
    frame_for = getattr(source, "frame_for", None)
    metadata_for = getattr(source, "metadata_for", None)
    out: list[float] = []
    for label in labels:
        metadata = None
        if callable(metadata_for):
            metadata = metadata_for(int(label))
        elif callable(frame_for):
            metadata = frame_for(int(label)).metadata
        if metadata is None:
            metadata = {}
        value = _metadata_get_case_insensitive(metadata, key)
        if value is None:
            raise KeyError(f"frame {label} has no metadata key {key!r}")
        out.append(float(value))
    return np.asarray(out, dtype=float)


def _metadata_get_case_insensitive(metadata: Any, key: str) -> Any:
    if key in metadata:
        return metadata[key]
    key_lower = str(key).lower()
    for candidate, value in dict(metadata).items():
        if str(candidate).lower() == key_lower:
            return value
    return None


def _plan_dict(plan: Any) -> dict[str, Any]:
    if is_dataclass(plan):
        return _json_safe(asdict(plan))
    return _json_safe(plan)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


__all__ = [
    "AnalysisResult",
    "PeakFitPlan",
    "PhaseFitPlan",
    "RSMPlan",
    "Sin2PsiPlan",
    "StitchPlan",
    "make_phase_fitter",
    "run_peak_fit",
    "run_phase_fit",
    "run_rsm",
    "run_sin2psi",
    "run_stitch",
]
