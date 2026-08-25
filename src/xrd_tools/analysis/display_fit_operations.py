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
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    InvalidCanonicalValue,
    _canonical_charge,
    _canonical_frame_charge,
    _digest,
    _owned,
)
__all__ = [
    "DisplayedTraceReceipt", "DisplayedTraceInput", "DisplayedPeakFitPlan", "DisplayedPeakFitResult",
    "CifAssetReceipt", "DisplayedPhaseFitPlan", "DisplayedPhaseFitResult",
    "run_displayed_peak_fit", "run_displayed_phase_fit",
]
_MAX_TRACE_POINTS = 1_000_000
_MAX_TRACE_BYTES, _MAX_FIT_BYTES = 16 * 1024 * 1024, 64 * 1024 * 1024
_MAX_PHASES = 8
_MAX_CIF_BYTES, _MAX_TOTAL_CIF_BYTES = 4 * 1024 * 1024, 16 * 1024 * 1024
_MAX_REFLECTIONS_PER_PHASE, _MAX_REFLECTIONS_TOTAL = 2048, 4096
_MAX_FIT_PARAMETERS, _MAX_PARAMETER_NAME_BYTES = 512, 256
_MAX_PARAMETER_NAMES_BYTES = 64 * 1024
_MAX_DIAGNOSTIC_BYTES = 1024
_PEAK_POLICY_VERSION = "p37-peak-selection-v1"
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
        fingerprint = _digest(identity)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "trace_fingerprint", fingerprint)
        object.__setattr__(self, "storage_bytes", _canonical_charge((identity, fingerprint)))
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
    def __post_init__(self) -> None:
        object.__setattr__(self, "cif_paths", tuple(Path(p) for p in _tuple_once(self.cif_paths)))
        object.__setattr__(self, "expected_hashes", _tuple_once(self.expected_hashes))
        object.__setattr__(self, "phase_names", _tuple_once(self.phase_names))
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
    return (
        plan.trace.trace_fingerprint, plan.fit_bounds, plan.selection_mode,
        plan.manual_centers, plan.n_peaks, plan.model, plan.background,
        plan.sigma_init, plan.sigma_bounds, plan.amplitude_init,
        plan.amplitude_bounds, plan.center_bounds_delta, plan.fraction_init,
        plan.max_nfev,
    )
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
    mask = (plan.trace.axis >= bounds[0]) & (plan.trace.axis <= bounds[1]) & np.isfinite(plan.trace.axis) & np.isfinite(plan.trace.intensity)
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
        backend_plan = PeakFitPlan(
            positions=positions, n_peaks=count, model=plan.model,
            background=plan.background, sigma_init=plan.sigma_init,
            sigma_bounds=plan.sigma_bounds, amplitude_init=plan.amplitude_init,
            amplitude_bounds=plan.amplitude_bounds,
            center_bounds_delta=plan.center_bounds_delta,
            fraction_init=plan.fraction_init,
            fit_kwargs={} if plan.max_nfev is None else {"max_nfev": plan.max_nfev},
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
        residual = _result_array(fit_result.residual, x.size)
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
    return (
        plan.trace.trace_fingerprint, plan.cif_paths, plan.expected_hashes,
        plan.phase_names, wavelength_angstrom,
        plan.prefit_background, plan.phase_profile,
        plan.lattice_pct, plan.min_intensity, plan.max_nfev,
    )
def _validate_phase(plan: DisplayedPhaseFitPlan) -> float:
    if not isinstance(plan, DisplayedPhaseFitPlan):
        raise _Refusal("INVALID_PHASE_PLAN", "plan has the wrong type")
    _fit_trace(plan.trace)
    count = len(plan.cif_paths)
    if not 1 <= count <= _MAX_PHASES or len(plan.expected_hashes) != count or len(plan.phase_names) != count:
        raise _Refusal("INVALID_PHASE_PLAN", "invalid CIF tuple lengths")
    if any(type(name) is not str or not name or len(name.encode("utf-8")) > 256 for name in plan.phase_names) or len(set(plan.phase_names)) != count:
        raise _Refusal("INVALID_PHASE_PLAN", "invalid phase names")
    if any(value is not None and (type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)) for value in plan.expected_hashes):
        raise _Refusal("INVALID_PHASE_PLAN", "invalid expected hash")
    if not _valid_number(plan.wavelength_angstrom) or plan.wavelength_angstrom <= 0:
        raise _Refusal("INVALID_PHASE_PLAN", "invalid wavelength")
    if type(plan.prefit_background) is not str or plan.prefit_background not in {"none", "snip", "chebyshev"} or type(plan.phase_profile) is not str or plan.phase_profile not in {"pseudovoigt", "gaussian", "lorentzian", "voigt"}:
        raise _Refusal("INVALID_PHASE_PLAN", "invalid phase option")
    if not _valid_number(plan.lattice_pct, minimum=0, maximum=0.25) or not _valid_number(plan.min_intensity, minimum=0, maximum=100):
        raise _Refusal("INVALID_PHASE_PLAN", "invalid phase numeric option")
    if plan.max_nfev is not None and (type(plan.max_nfev) is not int or not 1 <= plan.max_nfev <= 100_000):
        raise _Refusal("INVALID_PHASE_PLAN", "invalid max_nfev")
    finite = np.isfinite(plan.trace.axis) & np.isfinite(plan.trace.intensity)
    if np.count_nonzero(finite) < 10:
        raise _Refusal("INSUFFICIENT_FIT_POINTS", "phase fit needs ten finite points")
    return float(plan.wavelength_angstrom)
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
        receipts, phases, markers, total_bytes, total_reflections = [], [], [], 0, 0
        for index, (path, name, expected) in enumerate(zip(plan.cif_paths, plan.phase_names, plan.expected_hashes)):
            if cancel_token is not None and cancel_token.is_set():
                return _cancelled_phase(plan)
            raw, receipt = _capture_cif(path, name, expected)
            total_bytes += len(raw)
            if total_bytes > _MAX_TOTAL_CIF_BYTES:
                raise _Refusal("CIF_BYTE_LIMIT_EXCEEDED", "cumulative CIF byte limit exceeded")
            try:
                text = raw.decode("utf-8-sig", errors="strict")
                structure = Structure.from_str(text, fmt="cif")
                phase = PhaseModel(name)
                phase.structure = structure
                phase.calculate_peaks(wavelength=wavelength_angstrom)
            except (UnicodeError, ValueError) as exc:
                raise _Refusal("CIF_PARSE_FAILED", "CIF could not be parsed") from exc
            admitted = tuple(
                peak for peak in phase.peaks
                if q_range[0] <= float(peak.q) <= q_range[1]
                and float(peak.intensity) >= plan.min_intensity
            )
            if len(admitted) > _MAX_REFLECTIONS_PER_PHASE:
                raise _Refusal("PHASE_REFLECTION_LIMIT_EXCEEDED", "per-phase reflection limit exceeded")
            total_reflections += len(admitted)
            if total_reflections > _MAX_REFLECTIONS_TOTAL:
                raise _Refusal("PHASE_REFLECTION_LIMIT_EXCEEDED", "cumulative reflection limit exceeded")
            markers.extend(float(peak.q) for peak in admitted)
            receipts.append(receipt); phases.append(phase)
            _progress(progress_callback, index + 1, len(plan.cif_paths) + 1)
            if cancel_token is not None and cancel_token.is_set():
                return _cancelled_phase(plan)
        profile_extra = 1 if plan.phase_profile in {"pseudovoigt", "voigt"} else 0
        if 1 + len(phases) * (8 + 3 + profile_extra + 2) > _MAX_FIT_PARAMETERS:
            raise _Refusal("FIT_PARAMETER_LIMIT_EXCEEDED", "predicted fit parameter limit exceeded")
        fitter = PhaseFitter(x, y, prefit_background=plan.prefit_background)
        for phase in phases:
            fitter.add_phase(phase, q_range=q_range, min_intensity=plan.min_intensity)
        params = fitter.build_parameters(
            lattice_pct=plan.lattice_pct, phase_profile=plan.phase_profile,
            texture="none",
        )
        names, values, errors = _parameters(params)
        fit_kwargs = {} if plan.max_nfev is None else {"max_nfev": plan.max_nfev}
        backend = fitter.fit(
            params=params, phase_profile=plan.phase_profile,
            lattice_pct=plan.lattice_pct, texture="none", **fit_kwargs,
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
            "completed", "OK", (), plan.trace.trace_fingerprint,
            _receipt_value(plan.trace.receipt), plan_fp, policy_fp,
            wavelength_angstrom, receipt_values, success, "",
            plan.trace.label, plan.trace.axis_unit, names, values, errors,
            (), (),
        )
        size = _result_base(scalar_floor)
        message = _fit_message(backend.lmfit_result.message)
        size = _result_grow(size, len(message.encode("utf-8")))
        fractions_map = backend.phase_fractions()
        if not isinstance(fractions_map, Mapping) or (
            len(fractions_map) != len(plan.phase_names)
            or any(type(key) is not str for key in fractions_map)
            or set(fractions_map) != set(plan.phase_names)
        ):
            raise _Refusal("FIT_RESULT_LIMIT_EXCEEDED", "invalid phase fractions")
        fraction_values = []
        for name in plan.phase_names:
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
            "completed", "OK", (), plan.trace.trace_fingerprint,
            _receipt_value(plan.trace.receipt), plan_fp, policy_fp,
            wavelength_angstrom, receipt_values, success, message,
            plan.trace.label, plan.trace.axis_unit, names, values, errors,
            fractions, lattice,
        )
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
        fingerprint = _digest((scalar, arrays))
        _progress(progress_callback, len(plan.cif_paths) + 1, len(plan.cif_paths) + 1)
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
            lattice_parameters=lattice, phase_components=components,
            fit=fit, background=background, residual=residual,
            marker_positions=marker_array, storage_bytes=size,
            result_fingerprint=fingerprint,
        )
    except _Refusal as exc:
        return _phase_refusal(plan, exc.code, str(exc))
