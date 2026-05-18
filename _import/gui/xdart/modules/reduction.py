"""Adapters from legacy xdart Ewald objects to ssrl scan/frame APIs.

This module is the migration boundary for the thin-GUI refactor.  The rest of
xdart may still speak in terms of ``EwaldArch`` / ``EwaldSphere`` for now, but
new headless reduction work should cross into ``ssrl_xrd_tools`` as
``Frame`` / ``Scan`` / ``ReductionPlan`` objects.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Any

import numpy as np

from ssrl_xrd_tools.reduction import (
    Frame,
    Integration1DPlan,
    Integration2DPlan,
    ReductionPlan,
    Scan,
    run_reduction,
)

_GI_ONLY_ARGS = {
    "gi_mode_1d",
    "gi_mode_2d",
    "npt_oop",
    "npt_ip",
    "sample_orientation",
    "tilt_angle",
    "x_range",
    "y_range",
}


def frame_from_ewald_arch(
    arch: Any,
    *,
    include_image: bool = True,
) -> Frame:
    """Build an ``ssrl_xrd_tools.reduction.Frame`` from an ``EwaldArch``."""
    image = getattr(arch, "map_raw", None) if include_image else None
    source_path = _source_path(arch)
    mask = _arch_mask_as_bool(arch)
    metadata = dict(getattr(arch, "scan_info", {}) or {})
    bg_raw = getattr(arch, "bg_raw", 0)
    if np.ndim(bg_raw) == 0:
        metadata.setdefault("bg_raw", float(bg_raw))

    return Frame(
        index=int(getattr(arch, "idx", 0) or 0),
        image=image,
        metadata=metadata,
        source_path=source_path,
        source_frame_index=int(getattr(arch, "source_frame_idx", 0) or 0),
        background=getattr(arch, "bg_raw", None),
        mask=mask,
    )


def scan_from_ewald_sphere(
    sphere: Any,
    *,
    frame_indices: Iterable[int] | None = None,
    include_images: bool = True,
) -> Scan:
    """Build an ``ssrl_xrd_tools.reduction.Scan`` from an ``EwaldSphere``."""
    indices = list(frame_indices) if frame_indices is not None else list(sphere.arches.index)
    frames = [
        frame_from_ewald_arch(sphere.arches[int(idx)], include_image=include_images)
        for idx in indices
    ]
    first_arch = sphere.arches[int(indices[0])] if indices else None
    poni = getattr(first_arch, "poni", None) if first_arch is not None else None
    wavelength_A = None
    try:
        wavelength_m = float((getattr(sphere, "mg_args", {}) or {}).get("wavelength", 0))
        wavelength_A = wavelength_m * 1e10 if wavelength_m > 0 else None
    except (TypeError, ValueError):
        wavelength_A = None

    motors = {}
    scan_data = getattr(sphere, "scan_data", None)
    if scan_data is not None:
        try:
            motors = {
                str(col): np.asarray(scan_data[col].values, dtype=float)
                for col in scan_data.columns
            }
        except Exception:
            motors = {}

    return Scan(
        name=str(getattr(sphere, "name", "scan")),
        frames=frames,
        poni=poni,
        wavelength=wavelength_A,
        motors=motors,
        output_path=getattr(sphere, "data_file", None),
        extra={"source": "xdart.EwaldSphere"},
    )


def plan_from_ewald_sphere(
    sphere: Any,
    *,
    integrate_1d: bool = True,
    integrate_2d: bool | None = None,
    gi_incident_angle: float | None = None,
    chunk_size: int = 1,
) -> ReductionPlan:
    """Create a ``ReductionPlan`` using xdart's current sphere settings."""
    if integrate_2d is None:
        integrate_2d = not bool(getattr(sphere, "skip_2d", False))

    args_1d = dict(getattr(sphere, "bai_1d_args", {}) or {})
    args_2d = dict(getattr(sphere, "bai_2d_args", {}) or {})
    unit_1d = _pop_first(args_1d, ("unit",), None)
    unit_2d = _pop_first(args_2d, ("unit",), None)
    method_1d = str(_pop_first(args_1d, ("method",), "csr"))
    method_2d = str(_pop_first(args_2d, ("method",), "csr"))
    npt_1d = int(_pop_first(args_1d, ("npt", "numpoints", "npt_rad"), 1000))
    npt_rad_2d, npt_azim_2d = _npt_2d(args_2d)
    radial_range = _pop_first(args_1d, ("radial_range",), None)
    radial_range_2d = _pop_first(args_2d, ("radial_range",), None)
    azimuth_range_1d = _pop_first(args_1d, ("azimuth_range",), None)
    azimuth_range_2d = _pop_first(args_2d, ("azimuth_range",), None)
    monitor_key = _pop_first(args_1d, ("monitor",), None)
    monitor_key_2d = _pop_first(args_2d, ("monitor",), None)
    chi_offset_1d = _pop_first(args_1d, ("chi_offset",), 0.0)
    chi_offset_2d = _pop_first(args_2d, ("chi_offset",), 0.0)
    if chi_offset_1d:
        azimuth_range_1d = _offset_range(azimuth_range_1d, float(chi_offset_1d))
    error_model = _pop_first(args_1d, ("error_model",), None)
    error_model_2d = _pop_first(args_2d, ("error_model",), None)
    polarization_factor = _pop_first(args_1d, ("polarization_factor",), None)
    polarization_factor_2d = _pop_first(args_2d, ("polarization_factor",), None)
    _pop_first(args_1d, ("normalization_factor",), None)
    _pop_first(args_2d, ("normalization_factor",), None)

    gi = bool(getattr(sphere, "gi", False))
    if not gi:
        _strip_nonstandard_args(args_1d)
        _strip_nonstandard_args(args_2d)
    if gi and gi_incident_angle is None:
        gi_incident_angle = getattr(sphere, "_cached_fiber_integrator_angle", None)
    if gi and gi_incident_angle is None:
        raise ValueError(
            "Cannot build a GI ReductionPlan without gi_incident_angle."
        )

    mask_shape = None
    try:
        first_idx = sphere.arches.index[0]
        first_img = getattr(sphere.arches[int(first_idx)], "map_raw", None)
        mask_shape = getattr(first_img, "shape", None)
    except Exception:
        mask_shape = None

    gi_method = _pop_first(args_1d, ("gi_method_1d",), None)
    if gi_method is None:
        gi_method = _pop_first(args_2d, ("gi_method_2d",), "no")

    return ReductionPlan(
        integration_1d=(
            Integration1DPlan(
                npt=npt_1d,
                unit=str(unit_1d or "q_A^-1"),
                method=method_1d,
                radial_range=radial_range,
                azimuth_range=azimuth_range_1d,
                monitor_key=monitor_key,
                error_model=error_model,
                polarization_factor=polarization_factor,
                extra=args_1d,
            )
            if integrate_1d else None
        ),
        integration_2d=(
            Integration2DPlan(
                npt_rad=npt_rad_2d,
                npt_azim=npt_azim_2d,
                unit=str(unit_2d or "q_A^-1"),
                method=method_2d,
                radial_range=radial_range_2d,
                azimuth_range=azimuth_range_2d,
                azimuth_offset=float(chi_offset_2d or 0.0),
                monitor_key=monitor_key_2d,
                error_model=error_model_2d,
                polarization_factor=polarization_factor_2d,
                extra=args_2d,
            )
            if integrate_2d else None
        ),
        gi=gi,
        mask=_flat_mask_as_bool(getattr(sphere, "global_mask", None), mask_shape),
        chunk_size=chunk_size,
        gi_incident_angle=gi_incident_angle,
        gi_method=str(gi_method),
    )


def reduce_ewald_arch(
    arch: Any,
    plan: ReductionPlan,
    *,
    scan_name: str = "scan",
    global_mask: Any = None,
    integrator: Any = None,
) -> Any:
    """Reduce one ``EwaldArch`` through ``ssrl_xrd_tools.reduction``.

    The returned object is the same ``arch`` instance, populated with
    ``int_1d`` / ``int_2d`` so existing xdart display and writer code can
    continue to operate while the computation crosses the new Scan/Frame API.
    """
    if getattr(arch, "gi", False):
        raise ValueError("reduce_ewald_arch currently handles non-GI arches only.")
    frame = frame_from_ewald_arch(arch)
    plan = _plan_with_mask_for_arch(plan, global_mask, arch)
    scan = Scan(
        name=scan_name,
        frames=[frame],
        poni=getattr(arch, "poni", None),
        integrator=integrator if integrator is not None else getattr(arch, "integrator", None),
    )
    result = run_reduction(plan, scan)
    reduction = result.frames[int(arch.idx)]
    arch.int_1d = reduction.result_1d
    arch.int_2d = reduction.result_2d
    arch.map_norm = _frame_norm(frame, plan)
    return arch


def _source_path(arch: Any) -> Path | None:
    resolver = getattr(arch, "_resolved_source_path", None)
    path = resolver() if callable(resolver) else getattr(arch, "source_file", "")
    return Path(path) if path else None


def _normalization_factor(arch: Any) -> float | None:
    value = getattr(arch, "map_norm", None)
    if value is None:
        return None
    try:
        norm = float(value)
        if not np.isfinite(norm) or norm in (0.0, 1.0):
            return None
        return norm
    except (TypeError, ValueError):
        return None


def _frame_norm(frame: Frame, plan: ReductionPlan) -> float:
    if frame.normalization_factor is not None:
        return float(frame.normalization_factor)
    integration = plan.integration_1d or plan.integration_2d
    if integration and integration.monitor_key:
        key = integration.monitor_key
        value = frame.metadata.get(key)
        if value is None:
            value = frame.metadata.get(key.upper())
        if value is None:
            value = frame.metadata.get(key.lower())
        try:
            value = float(value)
            return value if np.isfinite(value) and value != 0 else 1.0
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _plan_with_mask_for_arch(
    plan: ReductionPlan,
    global_mask: Any,
    arch: Any,
) -> ReductionPlan:
    shape = getattr(getattr(arch, "map_raw", None), "shape", None)
    gmask = _flat_mask_as_bool(global_mask, shape)
    if plan.mask is None:
        return replace(plan, mask=gmask)
    if gmask is None:
        return plan
    return replace(plan, mask=np.asarray(plan.mask, dtype=bool) | gmask)


def _arch_mask_as_bool(arch: Any) -> np.ndarray | None:
    mask = getattr(arch, "mask", None)
    image = getattr(arch, "map_raw", None)
    shape = getattr(image, "shape", None)
    return _flat_mask_as_bool(mask, shape)


def _flat_mask_as_bool(mask: Any, shape: tuple[int, int] | None) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if shape is None:
        if arr.ndim == 2:
            return arr.astype(bool, copy=False)
        return None
    if arr.ndim == 2:
        if arr.shape != shape:
            raise ValueError(f"mask shape {arr.shape} does not match image shape {shape}")
        return arr.astype(bool, copy=False)
    if arr.ndim != 1:
        raise ValueError(f"flat mask must be 1D; got shape {arr.shape}")
    if arr.dtype == bool:
        if arr.size != int(np.prod(shape)):
            raise ValueError(
                f"flat boolean mask length {arr.size} does not match image shape {shape}"
            )
        return arr.reshape(shape)
    out = np.zeros(int(np.prod(shape)), dtype=bool)
    flat = np.asarray(arr, dtype=int).ravel()
    if np.any(flat < 0) or np.any(flat >= out.size):
        raise ValueError(f"flat mask indices out of bounds for image shape {shape}")
    out[flat] = True
    return out.reshape(shape)


def _npt_2d(args_2d: dict[str, Any]) -> tuple[int, int]:
    npt = args_2d.pop("npt", None)
    if isinstance(npt, (tuple, list)) and len(npt) == 2:
        return int(npt[0]), int(npt[1])
    npt_rad = args_2d.pop("npt_rad", None)
    npt_azim = args_2d.pop("npt_azim", None)
    if npt_rad is None:
        npt_rad = npt if npt is not None else 1000
    if npt_azim is None:
        npt_azim = 360
    return int(npt_rad), int(npt_azim)


def _pop_first(args: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in args:
            return args.pop(key)
    return default


def _strip_nonstandard_args(args: dict[str, Any]) -> None:
    for key in _GI_ONLY_ARGS:
        args.pop(key, None)


def _offset_range(
    value: tuple[float, float] | list[float] | None,
    offset: float,
) -> tuple[float, float] | None:
    if value is None:
        return None
    return float(value[0]) - offset, float(value[1]) - offset


__all__ = [
    "frame_from_ewald_arch",
    "scan_from_ewald_sphere",
    "plan_from_ewald_sphere",
    "reduce_ewald_arch",
]
