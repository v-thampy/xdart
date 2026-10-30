"""Blocking, Qt-free operations for one already-displayed detached trace."""

from __future__ import annotations

import builtins
import hashlib
import math
import os
import stat
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xrd_tools.analysis.axis_units import require_inverse_angstrom
from xrd_tools.analysis.phase_references import (
    MAX_PHASES as _MAX_PHASES, MAX_CIF_BYTES as _MAX_CIF_BYTES,
    MAX_TOTAL_CIF_BYTES as _MAX_TOTAL_CIF_BYTES,
    MAX_REFLECTIONS_PER_PHASE as _MAX_REFLECTIONS_PER_PHASE,
    MAX_REFLECTIONS_TOTAL as _MAX_REFLECTIONS_TOTAL,
)
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    InvalidCanonicalValue,
    _canonical_charge,
    _canonical_frame_charge,
    _digest,
    _owned,
    _owned_missing,
)
__all__ = [
    "DisplayedTraceReceipt", "DisplayedTraceInput", "DisplayedPeakFitPlan", "DisplayedPeakFitResult",
    "CifAssetReceipt", "DisplayedPhaseFitPlan", "DisplayedPhaseFitResult",
    "run_displayed_peak_fit", "run_displayed_phase_fit", "displayed_fit_axis",
]
_MAX_TRACE_POINTS = 1_000_000
_MAX_TRACE_BYTES, _MAX_FIT_BYTES = 16 * 1024 * 1024, 64 * 1024 * 1024
_MAX_FIT_PARAMETERS, _MAX_PARAMETER_NAME_BYTES = 512, 256
_MAX_PARAMETER_NAMES_BYTES = 64 * 1024
_MAX_DIAGNOSTIC_BYTES = 1024
_PEAK_POLICY_VERSION = "p37-peak-selection-v1"
#: PHASE-ANALYSIS-128: the notebook's canonical choices (the kernel's ``_PHASE_PROFILE_SPEC`` and model specs)
_PHASE_PROFILES = frozenset({"gaussian", "lorentzian", "pseudovoigt", "voigt", "lorentzian_squared", "pearson7",
                             "splitlorentzian", "moffat", "pearson4", "studentst", "skewedgaussian"})
_WIDTH_MODELS = frozenset({"caglioti", "scherrer", "fixed"})
_TEXTURES = frozenset({"none", "march_dollase", "free"})
_FIT_BACKGROUNDS = frozenset({"none", "linear", "polynomial", "chebyshev", "spline", "template", "template+linear",
                              "template+polynomial", "template+chebyshev"})
_AMORPHOUS_PROFILES = frozenset({"none", "gaussian", "pseudovoigt", "lorentzian", "voigt", "lorentzian_squared"})
_RANGE_UNITS = frozenset({"q_A^-1", "2th_deg", "2th_rad"})
#: each new option's default: only a value that differs joins the plan's identity, so earlier plans keep theirs
_PHASE_OPTION_DEFAULTS = (
    ("width_model", "caglioti"), ("width_limits", (None, None)), ("texture", "none"), ("march_axis", (0, 0, 1)),
    ("free_height_range", (0.0, 10.0)), ("q_shift_bound", 0.05), ("lock_cross_phase", False), ("snip_width", None),
    ("fit_background", "none"), ("fit_background_degree", 3), ("amorphous_peak", "none"), ("amorphous_center", 1.5),
    ("amorphous_sigma", 0.3), ("fit_range", None), ("fit_range_unit", "q_A^-1"), ("initial_values", ()),
    ("initial_topology", ()),
)
def _tuple_once(value: Any) -> tuple[Any, ...]:
    if type(value) is tuple:
        return value
    if type(value) is list:
        return tuple(value)
    raise InvalidCanonicalValue("plan sequences require an exact tuple or list")
def _optional_tuple_once(value: Any, *, scalar: bool = False) -> Any:
    if value is None or type(value) is tuple:
        return value
    if type(value) is list:
        return tuple(value)
    if scalar and type(value) in {bool, int, float}:
        return value
    raise InvalidCanonicalValue("plan values require an exact tuple, list, or scalar")
def _receipt_value(receipt: "DisplayedTraceReceipt") -> tuple[Any, ...]:
    return (
        receipt.schema_version, receipt.run_generation, receipt.run_fingerprint,
        receipt.source_scan, receipt.artifact, receipt.frame_label, receipt.work_ordinal,
    )
def _frozen_array(value: Any) -> np.ndarray:
    result = _owned(value)
    if result.ndim != 1 or result.dtype.kind not in "biuf":
        raise ValueError("displayed traces require a numeric rank-1 array")
    if not np.isfinite(result).all():
        result = _owned_missing(np.where(np.isfinite(result), result, np.nan))
    return result
@dataclass(frozen=True, slots=True)
class DisplayedTraceReceipt:
    schema_version: str
    run_generation: int
    run_fingerprint: str
    source_scan: str | None
    artifact: Path
    frame_label: str
    work_ordinal: int | None = None
    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or type(self.run_generation) is not int or type(self.run_fingerprint) is not str or (self.source_scan is not None and type(self.source_scan) is not str) or type(self.frame_label) is not str or (self.work_ordinal is not None and type(self.work_ordinal) is not int):
            raise TypeError("invalid displayed trace receipt")
        object.__setattr__(self, "artifact", Path(self.artifact))
@dataclass(frozen=True, slots=True)
class DisplayedTraceInput:
    axis: np.ndarray
    intensity: np.ndarray
    label: str
    axis_unit: str
    title: str
    epoch: int
    receipt: DisplayedTraceReceipt
    trace_fingerprint: str = ""
    storage_bytes: int = 0
    sigma: np.ndarray | None = None
    def __post_init__(self) -> None:
        if not isinstance(self.receipt, DisplayedTraceReceipt) or any(type(value) is not str for value in (self.label, self.axis_unit, self.title)) or type(self.epoch) is not int:
            raise TypeError("invalid displayed trace identity")
        axis, intensity = _frozen_array(self.axis), _frozen_array(self.intensity)
        if axis.shape != intensity.shape:
            raise ValueError("displayed axis and intensity must have equal shape")
        identity = (
            axis, intensity, self.label,
            self.axis_unit, self.title, self.epoch, _receipt_value(self.receipt),
        )
        if self.sigma is not None:
            sigma = _frozen_array(self.sigma)
            if sigma.shape != intensity.shape:
                raise ValueError("displayed sigma and intensity must have equal shape")
            object.__setattr__(self, "sigma", sigma)
            identity += ("sigma", sigma)
        fingerprint = _digest(identity, allow_missing=True)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "trace_fingerprint", fingerprint)
        object.__setattr__(self, "storage_bytes", _canonical_charge((identity, fingerprint), allow_missing=True))
@dataclass(frozen=True, slots=True)
class DisplayedPeakFitPlan:
    trace: DisplayedTraceInput
    fit_bounds: tuple[float, float] | None = None
    selection_mode: str = "auto"
    manual_centers: tuple[float, ...] = ()
    n_peaks: int = 1
    model: str = "pseudovoigt"
    background: str = "linear"
    sigma_init: float | tuple[float, ...] | None = None
    sigma_bounds: tuple[float, float] | None = None
    amplitude_init: float | tuple[float, ...] | None = None
    amplitude_bounds: tuple[float, float] | None = None
    center_bounds_delta: float | None = None
    fraction_init: float = 0.5
    max_nfev: int | None = None
    use_uncertainties: bool = False
    def __post_init__(self) -> None:
        object.__setattr__(self, "manual_centers", _tuple_once(self.manual_centers))
        for name in ("fit_bounds", "sigma_bounds", "amplitude_bounds"):
            object.__setattr__(
                self, name, _optional_tuple_once(getattr(self, name)),
            )
        for name in ("sigma_init", "amplitude_init"):
            object.__setattr__(
                self, name,
                _optional_tuple_once(getattr(self, name), scalar=True),
            )
@dataclass(frozen=True, slots=True)
class DisplayedPeakFitResult:
    disposition: AnalysisDisposition
    code: str
    diagnostics: tuple[str, ...] = ()
    trace_fingerprint: str = ""
    trace_receipt: DisplayedTraceReceipt | None = None
    plan_fingerprint: str = ""
    policy_fingerprint: str = ""
    fit_success: bool | None = None
    message: str = ""
    label: str = ""
    axis_unit: str = ""
    parameter_names: tuple[str, ...] = ()
    parameter_values: tuple[float, ...] = ()
    parameter_stderr: tuple[float | None, ...] = ()
    fit: np.ndarray | None = None
    background: np.ndarray | None = None
    residual: np.ndarray | None = None
    marker_positions: np.ndarray | None = None
    storage_bytes: int = 0
    result_fingerprint: str = ""
@dataclass(frozen=True, slots=True)
class CifAssetReceipt:
    lexical_path: Path
    resolved_path: Path
    phase_name: str
    byte_count: int
    sha256: str
    expected_sha256: str | None
    pre_state: tuple[int, int, int, int, int, int]
    post_state: tuple[int, int, int, int, int, int]
    receipt_fingerprint: str
def _nested_tuple(value: Any) -> Any:
    """A JSON-loaded sequence (lists) as the plan keeps it: tuples all the way down."""
    if type(value) in {tuple, list}:
        return tuple(_nested_tuple(item) for item in value)
    return value
@dataclass(frozen=True, slots=True)
class DisplayedPhaseFitPlan:
    trace: DisplayedTraceInput
    cif_paths: tuple[Path, ...]
    expected_hashes: tuple[str | None, ...]
    phase_names: tuple[str, ...]
    wavelength_angstrom: float
    prefit_background: str = "none"
    phase_profile: str = "pseudovoigt"
    lattice_pct: float = 0.05
    min_intensity: float = 0.0
    max_nfev: int | None = None
    use_uncertainties: bool = False
    phase_entries: tuple[Any, ...] = ()
    #: each phase's own minimum intensity, aligned with ``phase_names`` (REFERENCE-PHASES-20261011); empty: every
    #: phase uses ``min_intensity``. A lattice phase's value is ignored (it has no intensities).
    phase_min_intensities: tuple[float, ...] = ()
    #: PHASE-ANALYSIS-128: the HZO notebook's options. Each default reproduces the earlier fit, and only a value that
    #: differs from its default joins the plan's identity (``_phase_plan_value``).
    width_model: str = "caglioti"
    width_limits: tuple[float | None, float | None] = (None, None)
    texture: str = "none"
    march_axis: tuple[int, int, int] = (0, 0, 1)
    free_height_range: tuple[float, float] = (0.0, 10.0)
    q_shift_bound: float = 0.05
    lock_cross_phase: bool = False
    snip_width: int | None = None
    fit_background: str = "none"
    fit_background_degree: int = 3
    #: the substrate's pattern, captured through the fit's own input path (a q trace)
    background_template: DisplayedTraceInput | None = None
    amorphous_peak: str = "none"
    amorphous_center: float = 1.5
    amorphous_sigma: float = 0.3
    #: the solve's interval in ``fit_range_unit`` (the whole finite trace is still prepared and returned)
    fit_range: tuple[float, float] | None = None
    fit_range_unit: str = "q_A^-1"
    #: a Sequential batch's seed: the previous completed fit's (name, value) pairs and its phase/reflection topology
    initial_values: tuple[tuple[str, float], ...] = ()
    initial_topology: tuple[Any, ...] = ()
    def __post_init__(self) -> None:
        object.__setattr__(self, "cif_paths", tuple(Path(p) for p in _tuple_once(self.cif_paths)))
        object.__setattr__(self, "expected_hashes", _tuple_once(self.expected_hashes))
        object.__setattr__(self, "phase_names", _tuple_once(self.phase_names))
        object.__setattr__(self, "phase_entries", _tuple_once(self.phase_entries))
        object.__setattr__(self, "phase_min_intensities", _tuple_once(self.phase_min_intensities))
        for name in ("width_limits", "march_axis", "free_height_range"):
            object.__setattr__(self, name, _tuple_once(getattr(self, name)))
        object.__setattr__(self, "fit_range", _optional_tuple_once(self.fit_range))
        object.__setattr__(self, "initial_values", tuple(_tuple_once(pair) for pair in _tuple_once(self.initial_values)))
        object.__setattr__(self, "initial_topology", _nested_tuple(self.initial_topology))
@dataclass(frozen=True, slots=True)
class DisplayedPhaseFitResult:
    disposition: AnalysisDisposition
    code: str
    diagnostics: tuple[str, ...] = ()
    trace_fingerprint: str = ""
    trace_receipt: DisplayedTraceReceipt | None = None
    plan_fingerprint: str = ""
    policy_fingerprint: str = ""
    cif_receipts: tuple[CifAssetReceipt, ...] = ()
    wavelength_angstrom: float | None = None
    fit_success: bool | None = None
    message: str = ""
    label: str = ""
    axis_unit: str = ""
    parameter_names: tuple[str, ...] = ()
    parameter_values: tuple[float, ...] = ()
    parameter_stderr: tuple[float | None, ...] = ()
    phase_fractions: tuple[tuple[str, float], ...] = ()
    lattice_parameters: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    phase_components: tuple[np.ndarray, ...] = ()
    fit: np.ndarray | None = None
    background: np.ndarray | None = None
    residual: np.ndarray | None = None
    marker_positions: np.ndarray | None = None
    storage_bytes: int = 0
    result_fingerprint: str = ""
    phase_names: tuple[str, ...] = ()
    phase_fractions_reason: str = ""
    #: PHASE-ANALYSIS-128: the fitted background and the amorphous peak (None when the fit has none)
    fit_background: np.ndarray | None = None
    amorphous: np.ndarray | None = None
    #: each fitted phase's (id, kind, ordered reflection hkls): a Sequential seed applies only to the same listing.
    #: Derived from the pinned inputs, so it stays out of the result fingerprint.
    seed_topology: tuple[Any, ...] = ()
class _Refusal(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
def _valid_number(value: Any, *, minimum=None, maximum=None) -> bool:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return False
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)
def _valid_event(token: threading.Event | None) -> bool:
    return token is None or type(token) is threading.Event
def _progress(callback: Callable[[int, int], None] | None, done: int, total: int) -> None:
    if callback is not None:
        try:
            callback(done, total)
        except Exception:
            pass
def _fit_trace(trace: DisplayedTraceInput) -> None:
    if not isinstance(trace, DisplayedTraceInput):
        raise _Refusal("INVALID_DISPLAYED_TRACE", "trace has the wrong type")
    if trace.axis.size > _MAX_TRACE_POINTS:
        raise _Refusal("TRACE_POINT_LIMIT_EXCEEDED", "trace point limit exceeded")
    if trace.storage_bytes > _MAX_TRACE_BYTES:
        raise _Refusal("TRACE_BYTE_LIMIT_EXCEEDED", "trace byte limit exceeded")
def _fit_plan_value(plan: DisplayedPeakFitPlan) -> tuple[Any, ...]:
    value = (
        plan.trace.trace_fingerprint, plan.fit_bounds, plan.selection_mode,
        plan.manual_centers, plan.n_peaks, plan.model, plan.background,
        plan.sigma_init, plan.sigma_bounds, plan.amplitude_init,
        plan.amplitude_bounds, plan.center_bounds_delta, plan.fraction_init,
        plan.max_nfev,
    )
    return value + (("use_uncertainties", True),) if plan.use_uncertainties else value

def _fit_sigma(plan, mask: np.ndarray) -> np.ndarray | None:
    """Use measured uncertainties only when explicitly requested, on admitted fit samples."""
    if type(plan.use_uncertainties) is not bool:
        raise _Refusal("INVALID_FIT_UNCERTAINTIES", "use_uncertainties must be a bool")
    if not plan.use_uncertainties:
        return None
    if plan.trace.sigma is None:
        raise _Refusal("FIT_UNCERTAINTIES_REQUIRED", "this input has no measured uncertainties")
    sigma = plan.trace.sigma[mask]
    if not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise _Refusal("INVALID_FIT_UNCERTAINTIES", "fitted uncertainties must be finite and positive")
    return sigma
def _terminal(result_type, disposition, plan: Any, code: str, message: str):
    trace = getattr(plan, "trace", None)
    keywords = {}
    if result_type is DisplayedPhaseFitResult:
        wavelength = getattr(plan, "wavelength_angstrom", None)
        if _valid_number(wavelength) and wavelength > 0:
            keywords["wavelength_angstrom"] = float(wavelength)
    return result_type(
        disposition, code, (message,),
        trace_fingerprint=getattr(trace, "trace_fingerprint", ""),
        trace_receipt=getattr(trace, "receipt", None),
        label=getattr(trace, "label", ""), axis_unit=getattr(trace, "axis_unit", ""),
        **keywords,
    )
def _peak_refusal(plan: Any, code: str, message: str) -> DisplayedPeakFitResult:
    return _terminal(DisplayedPeakFitResult, AnalysisDisposition.REFUSED, plan, code, message)
def _phase_refusal(plan: Any, code: str, message: str) -> DisplayedPhaseFitResult:
    return _terminal(DisplayedPhaseFitResult, AnalysisDisposition.REFUSED, plan, code, message)
def _cancelled_peak(plan: DisplayedPeakFitPlan) -> DisplayedPeakFitResult:
    return _terminal(DisplayedPeakFitResult, AnalysisDisposition.CANCELLED, plan, "CANCELLED", "operation cancelled")
def _cancelled_phase(plan: DisplayedPhaseFitPlan) -> DisplayedPhaseFitResult:
    return _terminal(DisplayedPhaseFitResult, AnalysisDisposition.CANCELLED, plan, "CANCELLED", "operation cancelled")
def _load_peak_backend():
    from xrd_tools.analysis.plans import PeakFitPlan, run_peak_fit
    return PeakFitPlan, run_peak_fit
def _load_phase_backend():
    from pymatgen.core import Structure
    from xrd_tools.analysis.fitting.phase_fitting import PhaseFitter
    from xrd_tools.analysis.phase import PhaseModel
    return Structure, PhaseModel, PhaseFitter
def _auto_peak_policy(x: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
    from scipy.signal import find_peaks
    if x.size < 5:
        return ()
    span = float(np.max(y) - np.min(y))
    noise = 1.4826 * float(np.median(np.abs(np.diff(y) - np.median(np.diff(y))))) / math.sqrt(2.0)
    prominence = max(0.04 * span, 5.0 * noise)
    if not math.isfinite(prominence) or prominence <= 0:
        return ()
    window = max(3, (x.size // 150) | 1)
    smooth = np.convolve(y, np.ones(window) / window, mode="same")
    indices, props = find_peaks(
        smooth, prominence=prominence, distance=max(1, x.size // 80), width=2,
    )
    ranked = sorted(
        zip(indices, props.get("prominences", np.zeros(indices.size))),
        key=lambda item: float(item[1]), reverse=True,
    )[:12]
    return tuple(sorted(float(x[index]) for index, _ in ranked))
def _fitted_mask(trace: DisplayedTraceInput, bounds: tuple[float, float]) -> np.ndarray:
    return ((trace.axis >= bounds[0]) & (trace.axis <= bounds[1])
            & np.isfinite(trace.axis) & np.isfinite(trace.intensity))
def displayed_fit_axis(plan: DisplayedPeakFitPlan | DisplayedPhaseFitPlan) -> np.ndarray:
    """The axis values a completed displayed fit's arrays (fit, background, residual, components) are on: the
    trace's finite samples, within a peak fit's bounds (its whole finite extent when it has none)."""
    trace = plan.trace
    finite = trace.axis[np.isfinite(trace.axis)]
    extent = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 0.0)
    bounds = getattr(plan, "fit_bounds", None) or extent
    return trace.axis[_fitted_mask(trace, bounds)]
def _validate_peak(plan: DisplayedPeakFitPlan) -> tuple[np.ndarray, np.ndarray, tuple[float, ...] | None, int]:
    if not isinstance(plan, DisplayedPeakFitPlan):
        raise _Refusal("INVALID_PEAK_PLAN", "plan has the wrong type")
    _fit_trace(plan.trace)
    if type(plan.selection_mode) is not str or plan.selection_mode not in {"auto", "count"} or type(plan.n_peaks) is not int or not 1 <= plan.n_peaks <= 12:
        raise _Refusal("INVALID_PEAK_PLAN", "invalid peak selection")
    if type(plan.model) is not str or plan.model not in {"pseudovoigt", "gaussian", "lorentzian", "voigt"} or type(plan.background) is not str or plan.background not in {"linear", "constant", "none", "chebyshev3"}:
        raise _Refusal("INVALID_PEAK_PLAN", "invalid peak model")
    finite_axis = plan.trace.axis[np.isfinite(plan.trace.axis)]
    if finite_axis.size == 0:
        raise _Refusal("INVALID_PEAK_PLAN", "trace has no finite axis")
    extent = (float(np.min(finite_axis)), float(np.max(finite_axis)))
    bounds = extent if plan.fit_bounds is None else plan.fit_bounds
    if type(bounds) is not tuple or len(bounds) != 2 or not all(_valid_number(v) for v in bounds) or not extent[0] <= bounds[0] < bounds[1] <= extent[1]:
        raise _Refusal("INVALID_PEAK_PLAN", "invalid fit bounds")
    if type(plan.manual_centers) is not tuple or len(plan.manual_centers) > 12 or not all(_valid_number(v) and bounds[0] <= v <= bounds[1] for v in plan.manual_centers):
        raise _Refusal("INVALID_PEAK_PLAN", "invalid manual centers")
    if plan.manual_centers and len(plan.manual_centers) != plan.n_peaks:
        raise _Refusal("INVALID_PEAK_PLAN", "manual center count differs from n_peaks")
    def scalar_or_tuple(value, *, positive: bool) -> bool:
        if value is None:
            return True
        values = value if type(value) is tuple else (value,)
        if type(value) is tuple and len(values) != plan.n_peaks:
            return False
        return all(_valid_number(v) and (v > 0 if positive else v >= 0) for v in values)
    if not scalar_or_tuple(plan.sigma_init, positive=True) or not scalar_or_tuple(plan.amplitude_init, positive=False):
        raise _Refusal("INVALID_PEAK_PLAN", "invalid parameter initialization")
    for pair, allow_zero in ((plan.sigma_bounds, False), (plan.amplitude_bounds, True)):
        if pair is not None and (type(pair) is not tuple or len(pair) != 2 or not all(_valid_number(v) for v in pair) or not ((0 <= pair[0] if allow_zero else 0 < pair[0]) and pair[0] < pair[1])):
            raise _Refusal("INVALID_PEAK_PLAN", "invalid parameter bounds")
    span = bounds[1] - bounds[0]
    if plan.center_bounds_delta is not None and (not _valid_number(plan.center_bounds_delta) or not 0 < plan.center_bounds_delta <= span):
        raise _Refusal("INVALID_PEAK_PLAN", "invalid center bound")
    if not _valid_number(plan.fraction_init, minimum=0, maximum=1):
        raise _Refusal("INVALID_PEAK_PLAN", "invalid fraction")
    if plan.max_nfev is not None and (type(plan.max_nfev) is not int or not 1 <= plan.max_nfev <= 100_000):
        raise _Refusal("INVALID_PEAK_PLAN", "invalid max_nfev")
    mask = _fitted_mask(plan.trace, bounds)
    _fit_sigma(plan, mask)
    x, y = plan.trace.axis[mask], plan.trace.intensity[mask]
    positions = tuple(sorted(float(v) for v in plan.manual_centers)) or None
    count = len(positions) if positions else plan.n_peaks
    return x, y, positions, count
def _parameters(params: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float | None, ...]]:
    if len(params) > _MAX_FIT_PARAMETERS:
        raise _Refusal("FIT_PARAMETER_LIMIT_EXCEEDED", "fit parameter limit exceeded")
    names, values, errors, name_bytes = [], [], [], 0
    for name, parameter in params.items():
        if type(name) is not str:
            raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "invalid parameter name")
        size = len(name.encode("utf-8")); name_bytes += size
        if size > _MAX_PARAMETER_NAME_BYTES or name_bytes > _MAX_PARAMETER_NAMES_BYTES:
            raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "parameter names exceed limit")
        value, error = float(parameter.value), getattr(parameter, "stderr", None)
        if not math.isfinite(value):
            raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "nonfinite fit parameter")
        error = None if error is None or not math.isfinite(float(error)) else float(error)
        names.append(name); values.append(value); errors.append(error)
    return tuple(names), tuple(values), tuple(errors)
def _fit_message(value: Any) -> str:
    message = str(value)
    if len(message.encode("utf-8")) > _MAX_DIAGNOSTIC_BYTES:
        raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "fit message exceeds limit")
    return message
def _result_array(value: Any, size: int) -> np.ndarray:
    result = _owned(value, dtype=float)
    if result.ndim != 1 or result.size != size or not np.isfinite(result).all():
        raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "invalid fit result array")
    return result
def _result_base(value: Any) -> int:
    size = _canonical_charge((value, (), "0" * 64))
    if size > _MAX_FIT_BYTES:
        raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "fit result byte limit exceeded")
    return size
def _result_grow(size: int, extra: int) -> int:
    size += extra
    if size > _MAX_FIT_BYTES:
        raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "fit result byte limit exceeded")
    return size
def _result_add(size: int, value: Any) -> int:
    return _result_grow(size, _canonical_frame_charge(value))
def run_displayed_peak_fit(
    plan: DisplayedPeakFitPlan, *, cancel_token: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DisplayedPeakFitResult:
    if not _valid_event(cancel_token):
        return _peak_refusal(plan, "INVALID_CANCEL_TOKEN", "cancel_token must be an exact Event")
    try:
        x, y, positions, count = _validate_peak(plan)
        if cancel_token is not None and cancel_token.is_set():
            return _cancelled_peak(plan)
        if positions is None and plan.selection_mode == "auto":
            if type(plan.sigma_init) is tuple or type(plan.amplitude_init) is tuple:
                raise _Refusal(
                    "INVALID_PEAK_PLAN",
                    "automatic selection requires scalar or absent initializers",
                )
            try:
                positions = _auto_peak_policy(x, y)
            except ImportError:
                return _peak_refusal(plan, "PEAK_FIT_DEPENDENCY_UNAVAILABLE", "Peak fitting dependency unavailable")
            if not positions:
                raise _Refusal("PEAKS_NOT_FOUND", "automatic policy found no peaks")
            count = len(positions)
        if x.size < max(5, 3 * count):
            raise _Refusal("INSUFFICIENT_FIT_POINTS", "insufficient finite fit points")
        if cancel_token is not None and cancel_token.is_set():
            return _cancelled_peak(plan)
        _progress(progress_callback, 0, 1)
        if cancel_token is not None and cancel_token.is_set():
            return _cancelled_peak(plan)
        try:
            PeakFitPlan, runner = _load_peak_backend()
        except ImportError:
            return _peak_refusal(plan, "PEAK_FIT_DEPENDENCY_UNAVAILABLE", "Peak fitting dependency unavailable")
        fit_kwargs = {} if plan.max_nfev is None else {"max_nfev": plan.max_nfev}
        finite_axis = plan.trace.axis[np.isfinite(plan.trace.axis)]
        sigma = _fit_sigma(plan, _fitted_mask(plan.trace, plan.fit_bounds or
                           (float(np.min(finite_axis)), float(np.max(finite_axis)))))
        if sigma is not None:
            fit_kwargs["weights"] = 1.0 / sigma
        backend_plan = PeakFitPlan(
            positions=positions, n_peaks=count, model=plan.model,
            background=plan.background, sigma_init=plan.sigma_init,
            sigma_bounds=plan.sigma_bounds, amplitude_init=plan.amplitude_init,
            amplitude_bounds=plan.amplitude_bounds,
            center_bounds_delta=plan.center_bounds_delta,
            fraction_init=plan.fraction_init,
            fit_kwargs=fit_kwargs,
        )
        try:
            payload = runner(backend_plan, x, y).payload
        except ImportError:
            return _peak_refusal(plan, "PEAK_FIT_DEPENDENCY_UNAVAILABLE", "Peak fitting dependency unavailable")
        if cancel_token is not None and cancel_token.is_set():
            return _cancelled_peak(plan)
        fit_result = payload.fit_result
        names, values, errors = _parameters(payload.params)
        plan_fp, policy_fp = _digest(_fit_plan_value(plan)), _digest((_PEAK_POLICY_VERSION,))
        scalar = (
            "completed", "OK", (), plan.trace.trace_fingerprint,
            _receipt_value(plan.trace.receipt), plan_fp, policy_fp, bool(payload.success),
            _fit_message(fit_result.message), plan.trace.label, plan.trace.axis_unit,
            names, values, errors,
        )
        size = _result_base(scalar)
        fit = _result_array(payload.best_fit, x.size); size = _result_add(size, fit)
        components = fit_result.eval_components(x=x)
        selected = [np.asarray(value, dtype=float) for key, value in components.items() if "background" in str(key).lower() or str(key).lower().startswith("bg")]
        background = _result_array(sum(selected, np.zeros(x.size)), x.size)
        size = _result_add(size, background)
        residual = _result_array(y - fit if plan.use_uncertainties else fit_result.residual, x.size)
        size = _result_add(size, residual)
        markers = _owned(np.asarray(payload.peak_centers, dtype=float))
        if markers.ndim != 1 or markers.size > 12 or not np.isfinite(markers).all():
            raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "invalid peak markers")
        size = _result_add(size, markers)
        arrays = (fit, background, residual, markers)
        fingerprint = _digest((scalar, arrays))
        _progress(progress_callback, 1, 1)
        if cancel_token is not None and cancel_token.is_set():
            return _cancelled_peak(plan)
        return DisplayedPeakFitResult(
            AnalysisDisposition.COMPLETED, "OK",
            trace_fingerprint=plan.trace.trace_fingerprint,
            trace_receipt=plan.trace.receipt, plan_fingerprint=plan_fp,
            policy_fingerprint=policy_fp, fit_success=bool(payload.success),
            message=scalar[8], label=plan.trace.label,
            axis_unit=plan.trace.axis_unit, parameter_names=names,
            parameter_values=values, parameter_stderr=errors, fit=fit,
            background=background, residual=residual, marker_positions=markers,
            storage_bytes=size, result_fingerprint=fingerprint,
        )
    except _Refusal as exc:
        return _peak_refusal(plan, exc.code, str(exc))
def _phase_plan_value(
    plan: DisplayedPhaseFitPlan, wavelength_angstrom: float,
) -> tuple[Any, ...]:
    value = (
        plan.trace.trace_fingerprint, plan.cif_paths, plan.expected_hashes,
        plan.phase_names, wavelength_angstrom,
        plan.prefit_background, plan.phase_profile,
        plan.lattice_pct, plan.min_intensity, plan.max_nfev,
    )
    if plan.phase_entries:
        value += (("phase_entries", tuple((entry.id, entry.name, entry.source_kind, entry.lattice) for entry in plan.phase_entries)),)
    if plan.phase_min_intensities:
        value += (("phase_min_intensities", plan.phase_min_intensities),)
    value += tuple((name, getattr(plan, name)) for name, default in _PHASE_OPTION_DEFAULTS
                   if getattr(plan, name) != default)
    if plan.background_template is not None:
        template = plan.background_template
        value += (("background_template", template.trace_fingerprint, _receipt_value(template.receipt)),)
    return value + (("use_uncertainties", True),) if plan.use_uncertainties else value
def _validate_phase(plan: DisplayedPhaseFitPlan) -> float:
    if not isinstance(plan, DisplayedPhaseFitPlan):
        raise _Refusal("INVALID_PHASE_PLAN", "plan has the wrong type")
    _fit_trace(plan.trace)
    count = len(plan.phase_entries) if plan.phase_entries else len(plan.cif_paths)
    if plan.phase_entries:
        from .phase_references import PhaseEntry
        if plan.cif_paths or plan.expected_hashes or any(not isinstance(entry, PhaseEntry) for entry in plan.phase_entries):
            raise _Refusal("INVALID_PHASE_PLAN", "invalid phase entry mode")
        if tuple(entry.name for entry in plan.phase_entries) != plan.phase_names:
            raise _Refusal("INVALID_PHASE_PLAN", "phase entry names do not match")
    if not 1 <= count <= _MAX_PHASES or (not plan.phase_entries and len(plan.expected_hashes) != count) or len(plan.phase_names) != count:
        raise _Refusal("INVALID_PHASE_PLAN", "invalid CIF tuple lengths")
    if any(type(name) is not str or not name or len(name.encode("utf-8")) > 256 for name in plan.phase_names) or len(set(plan.phase_names)) != count:
        raise _Refusal("INVALID_PHASE_PLAN", "invalid phase names")
    if any(value is not None and (type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)) for value in plan.expected_hashes):
        raise _Refusal("INVALID_PHASE_PLAN", "invalid expected hash")
    if not _valid_number(plan.wavelength_angstrom) or plan.wavelength_angstrom <= 0:
        raise _Refusal("INVALID_PHASE_PLAN", "invalid wavelength")
    if type(plan.prefit_background) is not str or plan.prefit_background not in {"none", "snip", "chebyshev"} or type(plan.phase_profile) is not str or plan.phase_profile not in _PHASE_PROFILES:
        raise _Refusal("INVALID_PHASE_PLAN", "invalid phase option")
    if not _valid_number(plan.lattice_pct, minimum=0, maximum=0.25) or not _valid_number(plan.min_intensity, minimum=0, maximum=100):
        raise _Refusal("INVALID_PHASE_PLAN", "invalid phase numeric option")
    if plan.phase_min_intensities and (len(plan.phase_min_intensities) != count or not all(
            _valid_number(value, minimum=0, maximum=100) for value in plan.phase_min_intensities)):
        raise _Refusal("INVALID_PHASE_PLAN", "invalid per-phase minimum intensities")
    if plan.max_nfev is not None and (type(plan.max_nfev) is not int or not 1 <= plan.max_nfev <= 100_000):
        raise _Refusal("INVALID_PHASE_PLAN", "invalid max_nfev")
    finite = np.isfinite(plan.trace.axis) & np.isfinite(plan.trace.intensity)
    _fit_sigma(plan, finite)
    if np.count_nonzero(finite) < 10:
        raise _Refusal("INSUFFICIENT_FIT_POINTS", "phase fit needs ten finite points")
    _validate_phase_options(plan, finite)
    return float(plan.wavelength_angstrom)
def _invalid(field: str) -> _Refusal:
    return _Refusal("INVALID_PHASE_PLAN", f"invalid {field}")
def _choice(value: Any, choices: frozenset) -> bool:
    return type(value) is str and value in choices
def _validate_phase_options(plan: DisplayedPhaseFitPlan, finite: np.ndarray) -> None:
    """PHASE-ANALYSIS-128: each notebook option checked by name before any backend work."""
    if any(getattr(entry, "source_kind", "") == "calibrant" for entry in plan.phase_entries):
        raise _Refusal("INVALID_PHASE_PLAN", "calibrants are reference lines only and are not fitted")
    if not _choice(plan.width_model, _WIDTH_MODELS):
        raise _invalid("width_model")
    limits = plan.width_limits
    if len(limits) != 2 or any(value is not None and not (_valid_number(value) and value > 0) for value in limits) or (
            None not in limits and not limits[0] < limits[1]):
        raise _invalid("width_limits")
    # The kernel fills each omitted bound independently, using the full finite trace (not the solve mask).
    floor = 1e-5 if limits[0] is None else limits[0]
    cap = float(np.ptp(plan.trace.axis[finite])) / 4 if limits[1] is None else limits[1]
    if not floor < cap or (plan.width_model == "caglioti" and not max(floor ** 2, 1e-12) < cap ** 2):
        raise _invalid("width_limits")
    if not _choice(plan.texture, _TEXTURES):
        raise _invalid("texture")
    axis = plan.march_axis
    if len(axis) != 3 or any(type(value) is not int or abs(value) > 20 for value in axis) or (
            plan.texture == "march_dollase" and not any(axis)):
        raise _invalid("march_axis")
    heights = plan.free_height_range
    if len(heights) != 2 or not all(_valid_number(value) for value in heights) or not 0 <= heights[0] < heights[1]:
        raise _invalid("free_height_range")
    if not _valid_number(plan.q_shift_bound) or not plan.q_shift_bound > 0:
        raise _invalid("q_shift_bound")  # zero is refused: equal bounds cannot be fitted
    if type(plan.lock_cross_phase) is not bool:
        raise _invalid("lock_cross_phase")
    if plan.snip_width is not None and (type(plan.snip_width) is not int
                                        or not 1 <= plan.snip_width <= int(np.count_nonzero(finite))):
        raise _invalid("snip_width")
    if not _choice(plan.fit_background, _FIT_BACKGROUNDS):
        raise _invalid("fit_background")
    if type(plan.fit_background_degree) is not int or not 2 <= plan.fit_background_degree <= 15:
        raise _invalid("fit_background_degree")
    if plan.fit_background.startswith("template") != (plan.background_template is not None):
        raise _invalid("background_template")
    if plan.background_template is not None:
        _template_arrays(plan.background_template)
    if not _choice(plan.amorphous_peak, _AMORPHOUS_PROFILES):
        raise _invalid("amorphous_peak")
    if not _valid_number(plan.amorphous_center):
        raise _invalid("amorphous_center")
    if not _valid_number(plan.amorphous_sigma) or not plan.amorphous_sigma > 0:
        raise _invalid("amorphous_sigma")
    if plan.amorphous_peak != "none":
        axis_values = plan.trace.axis[finite]
        if not float(np.min(axis_values)) <= plan.amorphous_center <= float(
                np.max(axis_values)):
            raise _invalid("amorphous_center")
    if not _choice(plan.fit_range_unit, _RANGE_UNITS):
        raise _invalid("fit_range_unit")
    if plan.fit_range is not None:
        if len(plan.fit_range) != 2 or not all(_valid_number(value) for value in plan.fit_range) or not (
                plan.fit_range[0] < plan.fit_range[1]):
            raise _invalid("fit_range")
        if np.count_nonzero(_solve_mask(plan, finite)) < 10:
            raise _Refusal("INSUFFICIENT_FIT_POINTS", "the fit range holds fewer than ten finite points")
    seeds = plan.initial_values
    if seeds:
        if (len(seeds) > _MAX_FIT_PARAMETERS or any(len(pair) != 2 or type(pair[0]) is not str or not pair[0]
                                                   or len(pair[0].encode("utf-8")) > _MAX_PARAMETER_NAME_BYTES
                                                   or not _valid_number(pair[1]) for pair in seeds)
                or len({name for name, _value in seeds}) != len(seeds)
                or sum(len(name.encode("utf-8")) for name, _value in seeds) > _MAX_PARAMETER_NAMES_BYTES):
            raise _invalid("initial_values")
        if not plan.initial_topology:
            raise _invalid("initial_topology")
    topology = plan.initial_topology
    if type(topology) is not tuple:
        raise _invalid("initial_topology")
    if topology:
        if len(topology) > _MAX_PHASES:
            raise _invalid("initial_topology")
        reflections = 0
        for phase in topology:
            if (type(phase) is not tuple or len(phase) != 3 or type(phase[0]) is not str or not phase[0]
                    or len(phase[0].encode("utf-8")) > 256 or phase[1] not in ("cif", "lattice")
                    or type(phase[2]) is not tuple or len(phase[2]) > _MAX_REFLECTIONS_PER_PHASE
                    or any(type(hkl) is not tuple or len(hkl) != 3 or any(
                        type(v) is not int or abs(v) > 2**31 - 1 for v in hkl)
                           for hkl in phase[2])):
                raise _invalid("initial_topology")
            reflections += len(phase[2])
        if reflections > _MAX_REFLECTIONS_TOTAL:
            raise _invalid("initial_topology")
def _q_bounds(plan: DisplayedPhaseFitPlan) -> tuple[float, float] | None:
    """The fit range in Å⁻¹ (a 2θ range converted with the plan's wavelength), or None."""
    if plan.fit_range is None:
        return None
    if plan.fit_range_unit == "q_A^-1":
        low, high = plan.fit_range
    else:
        from xrd_tools.core.containers import convert_radial_axis
        low, high = (float(value) for value in convert_radial_axis(
            np.asarray(plan.fit_range, dtype=float), plan.fit_range_unit, "q_A^-1", plan.wavelength_angstrom))
    return (min(low, high), max(low, high))
def _solve_mask(plan: DisplayedPhaseFitPlan, finite: np.ndarray) -> np.ndarray:
    low, high = _q_bounds(plan)
    return finite & (plan.trace.axis >= low) & (plan.trace.axis <= high)
def _template_arrays(template: Any) -> tuple[np.ndarray, np.ndarray]:
    """A substrate admitted as the trace is (limits, Å⁻¹), keeping finite paired points: at least two distinct q."""
    if not isinstance(template, DisplayedTraceInput):
        raise _invalid("background_template")
    try:
        _fit_trace(template)
        require_inverse_angstrom(template.axis_unit, operation="a substrate template")
    except (_Refusal, ValueError) as exc:
        raise _invalid("background_template") from exc
    keep = np.isfinite(template.axis) & np.isfinite(template.intensity)
    x, y = np.asarray(template.axis)[keep], np.asarray(template.intensity)[keep]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if x.size < 2 or np.any(np.diff(x) <= 0):
        raise _invalid("background_template")
    return x, y
def _fit_background_spec(plan: DisplayedPhaseFitPlan) -> str | None:
    degree = plan.fit_background_degree
    return {"none": None, "linear": "polynomial1", "polynomial": f"polynomial{degree}",
            "chebyshev": f"chebyshev{degree}", "spline": f"spline{max(4, degree)}", "template": "template",
            "template+linear": "template+polynomial1", "template+polynomial": f"template+polynomial{degree}",
            "template+chebyshev": f"template+chebyshev{degree}"}[plan.fit_background]
def _seed_topology(fitter: Any, entries: list) -> tuple[Any, ...]:
    """Each fitted phase's (id, kind, ordered reflection hkls): a seed's parameters name reflections by position."""
    return tuple((entry.id, entry.source_kind, hkls) for entry, hkls in zip(entries, fitter.reflection_hkls(),
                                                                            strict=True))
def _apply_seed(params: Any, seeds: tuple[tuple[str, float], ...]) -> None:
    """Start independent varying parameters at the previous fit's values, kept inside their bounds."""
    for name, value in seeds:
        parameter = params.get(name)
        if parameter is None or not parameter.vary or parameter.expr is not None:
            continue
        low = parameter.min if math.isfinite(parameter.min) else -math.inf
        high = parameter.max if math.isfinite(parameter.max) else math.inf
        parameter.set(value=min(max(float(value), low), high))
def _file_state(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (value.st_mode, value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
def _capture_cif(path: Path, name: str, expected: str | None) -> tuple[bytes, CifAssetReceipt]:
    lexical = Path(path)
    try:
        resolved = lexical.resolve(strict=True)
        with builtins.open(resolved, "rb") as handle:
            pre = os.fstat(handle.fileno())
            if not stat.S_ISREG(pre.st_mode):
                raise _Refusal("CIF_UNAVAILABLE", "CIF is not a regular file")
            raw = handle.read(_MAX_CIF_BYTES + 1)
            post = os.fstat(handle.fileno())
    except OSError as exc:
        raise _Refusal("CIF_UNAVAILABLE", "CIF is unavailable") from exc
    pre_state, post_state = _file_state(pre), _file_state(post)
    if pre_state != post_state:
        raise _Refusal("CIF_REVISION_CHANGED", "CIF changed during capture")
    if len(raw) > _MAX_CIF_BYTES:
        raise _Refusal("CIF_BYTE_LIMIT_EXCEEDED", "CIF byte limit exceeded")
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise _Refusal("CIF_HASH_MISMATCH", "CIF hash does not match expectation")
    identity = (lexical, resolved, name, len(raw), digest, expected, pre_state, post_state)
    return raw, CifAssetReceipt(
        lexical, resolved, name, len(raw), digest, expected, pre_state,
        post_state, _digest(identity),
    )
def run_displayed_phase_fit(
    plan: DisplayedPhaseFitPlan, *, cancel_token: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DisplayedPhaseFitResult:
    if not _valid_event(cancel_token):
        return _phase_refusal(plan, "INVALID_CANCEL_TOKEN", "cancel_token must be an exact Event")
    try:
        if not isinstance(plan, DisplayedPhaseFitPlan):
            raise _Refusal("INVALID_PHASE_PLAN", "plan has the wrong type")
        _fit_trace(plan.trace)
        try:
            require_inverse_angstrom(plan.trace.axis_unit, operation="phase fitting")
        except ValueError:
            return _phase_refusal(plan, "PHASE_Q_UNIT_REQUIRED", "phase fitting requires inverse angstrom")
        wavelength_angstrom = _validate_phase(plan)
        if cancel_token is not None and cancel_token.is_set():
            return _cancelled_phase(plan)
        try:
            Structure, PhaseModel, PhaseFitter = _load_phase_backend()
        except ImportError:
            return _phase_refusal(plan, "PHASE_FIT_DEPENDENCY_UNAVAILABLE", "Phase fitting dependency unavailable")
        finite = np.isfinite(plan.trace.axis) & np.isfinite(plan.trace.intensity)
        x, y = plan.trace.axis[finite], plan.trace.intensity[finite]
        q_range = (float(np.min(x)), float(np.max(x)))
        from .phase_references import PhaseEntry, prepare_phase_models
        receipts, entries, markers, total_bytes = [], [], [], 0
        sources = tuple(plan.phase_entries) if plan.phase_entries else tuple(
            zip(plan.cif_paths, plan.phase_names, plan.expected_hashes))
        for index, source in enumerate(sources):
            if cancel_token is not None and cancel_token.is_set():
                return _cancelled_phase(plan)
            if isinstance(source, PhaseEntry) and source.source_kind == "lattice":
                entries.append(source)
                continue
            if isinstance(source, PhaseEntry):
                if source.cif_path is None:
                    raise _Refusal("CIF_UNAVAILABLE", "library CIF path is unavailable")
                path, name, expected = source.cif_path, source.name, source.id
            else:
                path, name, expected = source
            raw, receipt = _capture_cif(path, name, expected)
            total_bytes += len(raw)
            if total_bytes > _MAX_TOTAL_CIF_BYTES:
                raise _Refusal("CIF_BYTE_LIMIT_EXCEEDED", "cumulative CIF byte limit exceeded")
            entries.append(PhaseEntry(receipt.sha256, name, "cif", raw, cif_path=path))
            receipts.append(receipt)
            _progress(progress_callback, index + 1, len(sources) + 1)
            if cancel_token is not None and cancel_token.is_set():
                return _cancelled_phase(plan)
        try:
            phases, references = prepare_phase_models(
                *entries, wavelength_angstrom=wavelength_angstrom, q_range=q_range,
                min_intensity=plan.min_intensity, floors=plan.phase_min_intensities or None,
                model_class=PhaseModel, structure_class=Structure,
            )
        except ValueError as exc:
            code = "PHASE_REFLECTION_LIMIT_EXCEEDED" if "reflection limit" in str(exc) else "CIF_PARSE_FAILED"
            raise _Refusal(code, str(exc)) from exc
        markers.extend(peak.q for reference in references for peak in reference.reflections)
        profile_extra = {"pseudovoigt": 1, "voigt": 1, "pearson7": 1, "splitlorentzian": 1, "moffat": 1,
                         "skewedgaussian": 1, "pearson4": 2}.get(plan.phase_profile, 0)
        # an early guard only: free heights (every lattice peak; every CIF peak with free texture), not the
        # background or amorphous terms, which the exact count on the built parameters (``_parameters``) catches
        heights = sum(len(phase.peaks) for phase in phases
                      if getattr(phase, "source_kind", "") == "lattice" or plan.texture == "free")
        if 1 + len(phases) * (8 + 3 + profile_extra + 2) + heights > _MAX_FIT_PARAMETERS:
            raise _Refusal("FIT_PARAMETER_LIMIT_EXCEEDED", "predicted fit parameter limit exceeded")
        sigma = _fit_sigma(plan, finite)
        options = {"prefit_background": plan.prefit_background}
        if plan.snip_width is not None:
            options["prefit_background_kwargs"] = {"snip_width": plan.snip_width}
        spec = _fit_background_spec(plan)
        if spec is not None:
            options["fit_background"] = spec
        if plan.background_template is not None:
            options["fit_background_template"] = _template_arrays(plan.background_template)
        if plan.amorphous_peak != "none":
            options.update(amorphous_peak=plan.amorphous_peak, amorphous_init={
                "center": float(plan.amorphous_center), "sigma": float(plan.amorphous_sigma)})
        fitter = PhaseFitter(x, y, **options) if sigma is None else PhaseFitter(x, y, sigma, **options)
        floors = plan.phase_min_intensities or (plan.min_intensity,) * len(phases)
        for phase, floor in zip(phases, floors):  # the same per-phase value at both filter sites
            fitter.add_phase(phase, q_range=q_range, min_intensity=floor)
        params = fitter.build_parameters(
            lattice_pct=plan.lattice_pct, phase_profile=plan.phase_profile, texture=plan.texture,
            march_axis=plan.march_axis, pk_scale_range=plan.free_height_range, width_model=plan.width_model,
            width_min=plan.width_limits[0], width_max=plan.width_limits[1],
            lock_cross_phase=plan.lock_cross_phase, q_shift_bound=plan.q_shift_bound,
        )
        topology, diagnostics = _seed_topology(fitter, entries), ()
        if plan.initial_values:
            if plan.initial_topology == topology:
                _apply_seed(params, plan.initial_values)
            else:
                diagnostics = ("PHASE_SEED_DROPPED: the previous fit's reflections differ, so this fit started "
                               "from the defaults",)
        names, values, errors = _parameters(params)
        fit_kwargs = {} if plan.max_nfev is None else {"max_nfev": plan.max_nfev}
        bounds = _q_bounds(plan)
        if bounds is not None:
            fit_kwargs["q_range"] = bounds  # the solve only: the whole finite trace is prepared and returned
        backend = fitter.fit(
            params=params, phase_profile=plan.phase_profile,
            lattice_pct=plan.lattice_pct, texture=plan.texture, **fit_kwargs,
        )
        if cancel_token is not None and cancel_token.is_set():
            return _cancelled_phase(plan)
        names, values, errors = _parameters(backend.params)
        plan_fp = _digest(_phase_plan_value(plan, wavelength_angstrom))
        policy_fp = _digest(("p37-phase-fit-v1",))
        receipt_values = tuple(
            (receipt.lexical_path, receipt.resolved_path, receipt.phase_name,
             receipt.byte_count, receipt.sha256, receipt.expected_sha256,
             receipt.pre_state, receipt.post_state, receipt.receipt_fingerprint)
            for receipt in receipts
        )
        success = bool(backend.success)
        scalar_floor = (
            "completed", "OK", diagnostics, plan.trace.trace_fingerprint,
            _receipt_value(plan.trace.receipt), plan_fp, policy_fp,
            wavelength_angstrom, receipt_values, success, "",
            plan.trace.label, plan.trace.axis_unit, names, values, errors,
            (), (),
        )
        size = _result_base(scalar_floor)
        message = _fit_message(backend.lmfit_result.message)
        size = _result_grow(size, len(message.encode("utf-8")))
        fractions_reason = getattr(backend, "phase_fractions_reason", "")
        fractions_map = backend.phase_fractions()
        if not isinstance(fractions_map, Mapping) or (
            len(fractions_map) != (0 if fractions_reason else len(plan.phase_names))
            or any(type(key) is not str for key in fractions_map)
            or set(fractions_map) != (set() if fractions_reason else set(plan.phase_names))
        ):
            raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "invalid phase fractions")
        fraction_values = []
        for name in (() if fractions_reason else plan.phase_names):
            try:
                value = float(fractions_map[name])
            except (TypeError, ValueError, OverflowError) as exc:
                raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "invalid phase fractions") from exc
            if not math.isfinite(value):
                raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "invalid phase fractions")
            entry = (name, value); size = _result_add(size, entry)
            fraction_values.append(entry)
        fractions = tuple(fraction_values)
        allowed_lattice = ("a", "b", "c", "alpha", "beta", "gamma")
        lattice_values = []
        for index, name in enumerate(plan.phase_names):
            raw_lattice = backend.lattice_params(index)
            if not isinstance(raw_lattice, Mapping) or any(
                type(key) is not str or key not in allowed_lattice
                for key in raw_lattice
            ):
                raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "invalid lattice result")
            items = []
            for key in allowed_lattice:
                if key not in raw_lattice:
                    continue
                try:
                    value = float(raw_lattice[key])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "invalid lattice result") from exc
                if not math.isfinite(value):
                    raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "invalid lattice result")
                items.append((key, value))
            entry = (name, tuple(items)); size = _result_add(size, entry)
            lattice_values.append(entry)
        lattice = tuple(lattice_values)
        scalar = (
            "completed", "OK", diagnostics, plan.trace.trace_fingerprint,
            _receipt_value(plan.trace.receipt), plan_fp, policy_fp,
            wavelength_angstrom, receipt_values, success, message,
            plan.trace.label, plan.trace.axis_unit, names, values, errors,
            fractions, lattice,
        )
        if fractions_reason:
            scalar += (("phase_fractions_reason", fractions_reason),)
        size = _result_base(scalar)
        fit = _result_array(fitter.eval_model(backend.params), x.size)
        size = _result_add(size, fit)
        background = _result_array(fitter.background, x.size)
        size = _result_add(size, background)
        residual = _result_array(y - fit, x.size)
        size = _result_add(size, residual)
        marker_array = _owned(np.asarray(markers, dtype=float))
        size = _result_add(size, marker_array)
        component_values = []
        for index in range(len(phases)):
            component = _result_array(
                fitter.eval_phase(index, backend.params), x.size,
            )
            size = _result_add(size, component)
            component_values.append(component)
        components = tuple(component_values)
        arrays = (fit, background, residual, marker_array, *components)
        extra = {}
        for name, requested in (("fit_background", spec is not None), ("amorphous", plan.amorphous_peak != "none")):
            if requested:  # evaluated only when the plan asks for the component
                values_array = (fitter.eval_fit_background(backend.params) if name == "fit_background"
                                else fitter.eval_amorphous(backend.params))
                extra[name] = _result_array(values_array, x.size)
                size = _result_add(size, extra[name])
                arrays += ((name, extra[name]),)  # joins the fingerprint only when present
        size = _result_add(size, topology)  # retained seed metadata shares the existing result-byte budget
        fingerprint = _digest((scalar, arrays))
        _progress(progress_callback, len(sources) + 1, len(sources) + 1)
        if cancel_token is not None and cancel_token.is_set():
            return _cancelled_phase(plan)
        return DisplayedPhaseFitResult(
            AnalysisDisposition.COMPLETED, "OK",
            trace_fingerprint=plan.trace.trace_fingerprint,
            trace_receipt=plan.trace.receipt, plan_fingerprint=plan_fp,
            policy_fingerprint=policy_fp, cif_receipts=tuple(receipts),
            wavelength_angstrom=wavelength_angstrom,
            fit_success=success, message=message,
            label=plan.trace.label, axis_unit=plan.trace.axis_unit,
            parameter_names=names, parameter_values=values,
            parameter_stderr=errors, phase_fractions=fractions,
            phase_names=plan.phase_names, phase_fractions_reason=fractions_reason,
            lattice_parameters=lattice, phase_components=components,
            fit=fit, background=background, residual=residual,
            marker_positions=marker_array, storage_bytes=size,
            result_fingerprint=fingerprint, diagnostics=diagnostics,
            fit_background=extra.get("fit_background"), amorphous=extra.get("amorphous"),
            seed_topology=topology,
        )
    except _Refusal as exc:
        return _phase_refusal(plan, exc.code, str(exc))
