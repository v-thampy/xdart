"""Adapters from xdart live objects to ssrl scan/frame APIs.

This module is the migration boundary for the thin-GUI refactor.  The rest of
xdart may still import the transitional ``Ewald*`` aliases for now, but new
headless reduction work should cross into ``ssrl_xrd_tools`` as ``Frame`` /
``Scan`` / ``ReductionPlan`` objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Any

import numpy as np

from dataclasses import fields as _dc_fields

from ssrl_xrd_tools.reduction import (
    Frame,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    MaskSpec,
    ReductionPlan,
    Scan,
    run_reduction,
)

# S4: GI-only sphere kwargs that must NOT flow through to the standard
# pyFAI integrator path.  Derived from :class:`GIMode` (so adding a GIMode
# field automatically excludes that name) plus a small set of legacy
# xdart-only names that the GI widgets emit but :class:`GIMode` doesn't
# own.  Anything not in this set rides through ``Integration*Plan.extra``.
_GI_ONLY_ARGS: frozenset[str] = frozenset(
    {field.name for field in _dc_fields(GIMode)}
    | {
        "gi_mode_1d",
        "gi_mode_2d",
        "npt_oop",
        "npt_ip",
        "x_range",
        "y_range",
    }
)


def frame_from_live_frame(
    live_frame: Any,
    *,
    include_image: bool = True,
    include_background: bool = True,
) -> Frame:
    """Build an ``ssrl_xrd_tools.reduction.Frame`` from a ``LiveFrame``."""
    image = getattr(live_frame, "map_raw", None) if include_image else None
    source_path = _source_path(live_frame)
    mask = _live_frame_mask_as_bool(live_frame)
    metadata = dict(getattr(live_frame, "scan_info", {}) or {})
    bg_raw = getattr(live_frame, "bg_raw", None) if include_background else None
    if bg_raw is not None and np.ndim(bg_raw) == 0:
        metadata.setdefault("bg_raw", float(bg_raw))

    return Frame(
        index=int(getattr(live_frame, "idx", 0) or 0),
        image=image,
        metadata=metadata,
        source_path=source_path,
        source_frame_index=int(getattr(live_frame, "source_frame_idx", 0) or 0),
        background=bg_raw,
        mask=mask,
    )


def scan_from_live_scan(
    live_scan: Any,
    *,
    frame_indices: Iterable[int] | None = None,
    include_images: bool = True,
    include_backgrounds: bool | None = None,
) -> Scan:
    """Build an ``ssrl_xrd_tools.reduction.Scan`` from a ``LiveScan``."""
    if include_backgrounds is None:
        include_backgrounds = include_images
    indices = list(frame_indices) if frame_indices is not None else list(live_scan.arches.index)
    frames = [
        frame_from_live_frame(
            live_scan.arches[int(idx)],
            include_image=include_images,
            include_background=include_backgrounds,
        )
        for idx in indices
    ]
    first_frame = live_scan.arches[int(indices[0])] if indices else None
    poni = getattr(first_frame, "poni", None) if first_frame is not None else None
    wavelength_A = None
    try:
        wavelength_m = float((getattr(live_scan, "mg_args", {}) or {}).get("wavelength", 0))
        wavelength_A = wavelength_m * 1e10 if wavelength_m > 0 else None
    except (TypeError, ValueError):
        wavelength_A = None

    motors = {}
    scan_data = getattr(live_scan, "scan_data", None)
    if scan_data is not None:
        try:
            motors = {
                str(col): np.asarray(scan_data[col].values, dtype=float)
                for col in scan_data.columns
            }
        except Exception:
            motors = {}

    return Scan(
        name=str(getattr(live_scan, "name", "scan")),
        frames=frames,
        poni=poni,
        wavelength=wavelength_A,
        motors=motors,
        output_path=getattr(live_scan, "data_file", None),
        extra={"source": "xdart.LiveScan"},
    )


def plan_from_live_scan(
    live_scan: Any,
    *,
    integrate_1d: bool = True,
    integrate_2d: bool | None = None,
    gi_incident_angle: float | None = None,
) -> ReductionPlan:
    """Create a ``ReductionPlan`` using xdart's current live scan settings.

    Note: ``chunk_size`` and other execution-policy knobs live on
    :func:`run_reduction` (and on :func:`reduce_live_frame` by way of
    the single-frame call here), not on the plan itself.
    """
    if integrate_2d is None:
        integrate_2d = not bool(getattr(live_scan, "skip_2d", False))

    args_1d = dict(getattr(live_scan, "bai_1d_args", {}) or {})
    args_2d = dict(getattr(live_scan, "bai_2d_args", {}) or {})
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

    is_gi = bool(getattr(live_scan, "gi", False))
    gi_method = _pop_first(args_1d, ("gi_method_1d",), None)
    if gi_method is None:
        gi_method = _pop_first(args_2d, ("gi_method_2d",), "no")

    if not is_gi:
        _strip_nonstandard_args(args_1d)
        _strip_nonstandard_args(args_2d)
    if is_gi and gi_incident_angle is None:
        gi_incident_angle = getattr(live_scan, "_cached_fiber_integrator_angle", None)
    if is_gi and gi_incident_angle is None:
        raise ValueError(
            "Cannot build a GI ReductionPlan without gi_incident_angle."
        )

    mask_shape = None
    try:
        first_idx = live_scan.arches.index[0]
        first_img = getattr(live_scan.arches[int(first_idx)], "map_raw", None)
        mask_shape = getattr(first_img, "shape", None)
    except Exception:
        mask_shape = None

    gi_mode = (
        GIMode(
            incident_angle=float(gi_incident_angle),
            tilt_angle=float(getattr(live_scan, "tilt_angle", 0.0) or 0.0),
            sample_orientation=int(getattr(live_scan, "sample_orientation", 1) or 1),
            method=str(gi_method),
        )
        if is_gi else None
    )

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
        gi=gi_mode,
        mask=_mask_for_plan(getattr(live_scan, "global_mask", None), mask_shape),
    )


def reduce_live_frame(
    live_frame: Any,
    plan: ReductionPlan,
    *,
    scan_name: str = "scan",
    global_mask: Any = None,
    integrator: Any = None,
) -> Any:
    """Reduce one ``LiveFrame`` through ``ssrl_xrd_tools.reduction``.

    The returned object is the same ``live_frame`` instance, populated with
    ``int_1d`` / ``int_2d`` so existing xdart display and writer code can
    continue to operate while the computation crosses the new Scan/Frame API.
    """
    if getattr(live_frame, "gi", False):
        raise ValueError("reduce_live_frame currently handles non-GI frames only.")
    frame = frame_from_live_frame(live_frame)
    plan = _plan_with_mask_for_live_frame(plan, global_mask, live_frame)
    scan = Scan(
        name=scan_name,
        frames=[frame],
        poni=getattr(live_frame, "poni", None),
        integrator=integrator if integrator is not None else getattr(live_frame, "integrator", None),
    )
    result = run_reduction(plan, scan)
    reduction = result.frames[int(live_frame.idx)]
    live_frame.int_1d = reduction.result_1d
    live_frame.int_2d = reduction.result_2d
    live_frame.map_norm = _frame_norm(frame, plan)
    return live_frame


# ---------------------------------------------------------------------------
# S3 + C1 helpers — used by every wrangler so the GI-vs-standard dispatch
# and the per-sphere plan cache live in exactly one place.
# ---------------------------------------------------------------------------

def _plan_signature(
    live_scan: Any,
    integrate_1d: bool,
    integrate_2d: bool,
) -> tuple:
    """Hashable signature of the inputs that ``plan_from_live_scan`` reads.

    Used by :class:`StandardPlanCache` to skip plan rebuilds when nothing
    relevant on the sphere has changed.  Covers the bai_*_args dicts
    (sorted) and a digest of ``global_mask`` (shape + dtype + size +
    head/tail/sum for numeric masks).
    """
    def _items(args: Any) -> tuple:
        return tuple(
            sorted((str(key), repr(value)) for key, value in (args or {}).items())
        )

    mask = getattr(live_scan, "global_mask", None)
    if mask is None:
        mask_sig: Any = None
    else:
        arr = np.asarray(mask)
        flat = arr.ravel()
        if np.issubdtype(arr.dtype, np.number) and flat.size:
            mask_sum = float(np.sum(flat, dtype=np.float64))
            head = tuple(flat[:8].tolist())
            tail = tuple(flat[-8:].tolist())
        else:
            mask_sum = None
            head = ()
            tail = ()
        mask_sig = (arr.shape, str(arr.dtype), int(arr.size), mask_sum, head, tail)

    return (
        id(live_scan),
        bool(integrate_1d),
        bool(integrate_2d),
        _items(getattr(live_scan, "bai_1d_args", {})),
        _items(getattr(live_scan, "bai_2d_args", {})),
        mask_sig,
    )


class StandardPlanCache:
    """Per-owner cache for the standard (non-GI) :class:`ReductionPlan`.

    Wrappers (wranglers, integrator threads) keep one instance for the
    lifetime of a scan; the cached plan is rebuilt only when one of the
    sphere settings ``_plan_signature`` covers actually changes.

    Returns ``None`` for GI spheres so callers can drop straight into the
    legacy fiber-integrator path without an "is GI" check at every call
    site (the per-dispatch helper below already does that).
    """

    __slots__ = ("_plan", "_key")

    def __init__(self) -> None:
        self._plan: ReductionPlan | None = None
        self._key: tuple | None = None

    def get(
        self,
        live_scan: Any,
        *,
        integrate_1d: bool = True,
        integrate_2d: bool = True,
    ) -> ReductionPlan | None:
        if getattr(live_scan, "gi", False):
            return None
        key = _plan_signature(live_scan, integrate_1d, integrate_2d)
        if self._plan is None or self._key != key:
            self._plan = plan_from_live_scan(
                live_scan,
                integrate_1d=integrate_1d,
                integrate_2d=integrate_2d,
            )
            self._key = key
        return self._plan

    def invalidate(self) -> None:
        self._plan = None
        self._key = None


def dispatch_live_frame_reduction(
    live_frame: Any,
    live_scan: Any,
    *,
    standard_plan: ReductionPlan | None,
    integrator: Any,
    global_mask: Any,
    legacy_gi: Callable[[], None],
) -> None:
    """Run reduction for one live frame via the right path (standard or GI).

    Single dispatch point shared by all wrangler workers so the
    ``if self.gi: <legacy>; else: reduce_live_frame(...)`` fork lives
    in exactly one place.

    Parameters
    ----------
    live_frame, live_scan
        The live frame to reduce and its parent live scan.
    standard_plan
        ``ReductionPlan`` for the non-GI path; pass ``None`` to force
        the legacy callback (matches what
        :meth:`StandardPlanCache.get` returns for GI spheres).
    integrator
        Pre-built pyFAI integrator for the worker (typically borrowed
        from an :class:`IntegratorPool`).
    global_mask
        Scan-level mask passed through unchanged.
    legacy_gi
        Zero-arg callback that runs the GI fiber-integrator path.
        Invoked when ``standard_plan`` is ``None`` or
        ``live_frame.gi`` is ``True``.  Caller is responsible for borrowing
        the right fiber integrator inside this callback.
    """
    if standard_plan is None or getattr(live_frame, "gi", False):
        legacy_gi()
        return
    reduce_live_frame(
        live_frame,
        standard_plan,
        scan_name=str(getattr(live_scan, "name", "scan")),
        global_mask=global_mask,
        integrator=integrator,
    )


def _source_path(arch: Any) -> Path | None:
    resolver = getattr(arch, "_resolved_source_path", None)
    path = resolver() if callable(resolver) else getattr(arch, "source_file", "")
    return Path(path) if path else None


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


def _plan_with_mask_for_live_frame(
    plan: ReductionPlan,
    global_mask: Any,
    live_frame: Any,
) -> ReductionPlan:
    shape = getattr(getattr(live_frame, "map_raw", None), "shape", None)
    gmask = _flat_mask_as_bool(global_mask, shape)
    plan_mask = _flat_mask_as_bool(plan.mask, shape)
    if plan_mask is None:
        return replace(plan, mask=gmask)
    if gmask is None:
        return replace(plan, mask=plan_mask)
    return replace(plan, mask=plan_mask | gmask)


def _live_frame_mask_as_bool(live_frame: Any) -> np.ndarray | None:
    mask = getattr(live_frame, "mask", None)
    image = getattr(live_frame, "map_raw", None)
    shape = getattr(image, "shape", None)
    return _mask_for_plan(mask, shape)


def _mask_for_plan(mask: Any, shape: tuple[int, int] | None) -> np.ndarray | MaskSpec | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if shape is None and arr.ndim == 1:
        return MaskSpec(arr.copy())
    return _flat_mask_as_bool(mask, shape)


def _flat_mask_as_bool(mask: Any, shape: tuple[int, int] | None) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, MaskSpec):
        if shape is None:
            return None
        return mask.to_bool(shape)
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


# Transitional compatibility adapter names. Keep these aliases for one release
# while callers move from Ewald* vocabulary to Live* vocabulary.
frame_from_ewald_arch = frame_from_live_frame
scan_from_ewald_sphere = scan_from_live_scan
plan_from_ewald_sphere = plan_from_live_scan
reduce_ewald_arch = reduce_live_frame
dispatch_arch_reduction = dispatch_live_frame_reduction


__all__ = [
    "StandardPlanCache",
    "dispatch_live_frame_reduction",
    "dispatch_arch_reduction",
    "frame_from_live_frame",
    "frame_from_ewald_arch",
    "scan_from_live_scan",
    "scan_from_ewald_sphere",
    "plan_from_live_scan",
    "plan_from_ewald_sphere",
    "reduce_live_frame",
    "reduce_ewald_arch",
]
