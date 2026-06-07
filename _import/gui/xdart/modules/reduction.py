"""Adapters from xdart live objects to ssrl scan/frame APIs.

This module is the migration boundary for the thin-GUI refactor.  The rest of
xdart may still import the transitional ``Ewald*`` aliases for now, but new
headless reduction work should cross into ``ssrl_xrd_tools`` as ``Frame`` /
``Scan`` / ``ReductionPlan`` objects.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Any

import numpy as np

logger = logging.getLogger(__name__)

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

# S4: GI-only scan kwargs that must NOT flow through to the standard
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
    indices = list(frame_indices) if frame_indices is not None else list(live_scan.frames.index)
    scan_data = getattr(live_scan, "scan_data", None)
    frames = []
    for idx in indices:
        frame = frame_from_live_frame(
            live_scan.frames[int(idx)],
            include_image=include_images,
            include_background=include_backgrounds,
        )
        frame.metadata.update(
            {
                key: value
                for key, value in _scan_data_row(scan_data, int(idx)).items()
                if key not in frame.metadata
            }
        )
        frames.append(frame)
    first_frame = live_scan.frames[int(indices[0])] if indices else None
    poni = getattr(first_frame, "poni", None) if first_frame is not None else None
    wavelength_A = None
    try:
        wavelength_m = float((getattr(live_scan, "mg_args", {}) or {}).get("wavelength", 0))
        wavelength_A = wavelength_m * 1e10 if wavelength_m > 0 else None
    except (TypeError, ValueError):
        wavelength_A = None

    motors = {}
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
    gi_mode_1d = _pop_first(args_1d, ("gi_mode_1d",), "q_total")
    gi_mode_2d = _pop_first(args_2d, ("gi_mode_2d",), "qip_qoop")
    npt_oop = _pop_first(args_1d, ("npt_oop",), None)
    if npt_oop is None:
        npt_oop = _pop_first(args_2d, ("npt_oop",), None)
    gi_method = _pop_first(args_1d, ("gi_method_1d",), None)
    if gi_method is None:
        gi_method = _pop_first(args_2d, ("gi_method_2d",), "no")

    if not is_gi:
        _strip_nonstandard_args(args_1d)
        _strip_nonstandard_args(args_2d)
    if is_gi and gi_incident_angle is None:
        gi_incident_angle = getattr(live_scan, "_cached_fiber_integrator_angle", None)
    incidence_motor = getattr(live_scan, "incidence_motor", None)
    if is_gi and gi_incident_angle is None and incidence_motor is not None:
        try:
            gi_incident_angle = float(incidence_motor)
            incidence_motor = None
        except (TypeError, ValueError):
            pass
    if is_gi and gi_incident_angle is None and not incidence_motor:
        raise ValueError(
            "Cannot build a GI ReductionPlan without gi_incident_angle or incidence_motor."
        )

    mask_shape = None
    try:
        first_idx = live_scan.frames.index[0]
        first_img = getattr(live_scan.frames[int(first_idx)], "map_raw", None)
        mask_shape = getattr(first_img, "shape", None)
    except Exception:
        mask_shape = None

    gi_mode = (
        GIMode(
            incident_angle=(float(gi_incident_angle) if gi_incident_angle is not None else None),
            incidence_motor=str(incidence_motor) if incidence_motor else None,
            tilt_angle=float(getattr(live_scan, "tilt_angle", 0.0) or 0.0),
            sample_orientation=int(getattr(live_scan, "sample_orientation", 1) or 1),
            method=str(gi_method),
            mode_1d=str(gi_mode_1d),
            mode_2d=str(gi_mode_2d),
            npt_oop=(int(npt_oop) if npt_oop is not None else None),
        )
        if is_gi else None
    )
    unit_1d = _gi_1d_unit_default(unit_1d, str(gi_mode_1d), is_gi=is_gi)
    unit_2d = _gi_2d_unit_default(unit_2d, str(gi_mode_2d), is_gi=is_gi)

    return ReductionPlan(
        integration_1d=(
            Integration1DPlan(
                npt=npt_1d,
                unit=unit_1d,
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
                unit=unit_2d,
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


def reduce_live_frames(
    live_frames: Iterable[Any],
    plan: ReductionPlan,
    *,
    scan_name: str = "scan",
    global_mask: Any = None,
    integrator: Any = None,
    poni: Any = None,
    executor: Any = None,
    chunk_size: int | None = None,
    gi_freeze_mode: str | None = None,
) -> list[Any]:
    """Reduce a batch of ``LiveFrame`` objects through one headless run."""

    frames = list(live_frames)
    if not frames:
        return []
    plan = _plan_with_mask_for_live_frame(plan, global_mask, frames[0])
    headless_frames = [frame_from_live_frame(frame) for frame in frames]
    scan = Scan(
        name=scan_name,
        frames=headless_frames,
        poni=poni if poni is not None else getattr(frames[0], "poni", None),
        integrator=integrator if integrator is not None else getattr(frames[0], "integrator", None),
    )
    result = run_reduction(
        plan,
        scan,
        executor=executor,
        chunk_size=chunk_size or (len(frames) if executor is not None else 1),
        gi_freeze_mode=gi_freeze_mode,
    )
    by_index = {int(frame.idx): frame for frame in frames}
    for headless_frame in headless_frames:
        live_frame = by_index[int(headless_frame.index)]
        reduction = result.frames[int(headless_frame.index)]
        live_frame.int_1d = reduction.result_1d
        live_frame.int_2d = reduction.result_2d
        live_frame.map_norm = _frame_norm(headless_frame, plan)
    return frames


# ---------------------------------------------------------------------------
# S3 + C1 helpers — used by every wrangler so the GI-vs-standard dispatch
# and the per-scan plan cache live in exactly one place.
# ---------------------------------------------------------------------------

_UNSET = object()  # sentinel: "mask_sig not supplied" (distinct from None)


def _mask_signature(mask: Any) -> Any:
    """Content digest of a detector mask (shape + dtype + size + sum +
    head/tail for numeric masks).  This is the O(N) part — it touches the
    whole array via ``np.sum`` — so callers in per-frame hot loops should
    memoize it by mask identity rather than recompute it every frame
    (see :meth:`StandardPlanCache._mask_sig_for`)."""
    if mask is None:
        return None
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
    return (arr.shape, str(arr.dtype), int(arr.size), mask_sum, head, tail)


def _plan_signature(
    live_scan: Any,
    integrate_1d: bool,
    integrate_2d: bool,
    *,
    mask_sig: Any = _UNSET,
) -> tuple:
    """Hashable signature of the inputs that ``plan_from_live_scan`` reads.

    Used by :class:`StandardPlanCache` to skip plan rebuilds when nothing
    relevant on the scan has changed.  Covers the bai_*_args dicts
    (sorted) and a digest of ``global_mask``.

    ``mask_sig`` lets the caller pass an already-computed mask digest so
    the O(N) :func:`_mask_signature` isn't recomputed on every per-frame
    call; when omitted it's derived from ``live_scan.global_mask``.
    """
    def _items(args: Any) -> tuple:
        return tuple(
            sorted((str(key), repr(value)) for key, value in (args or {}).items())
        )

    if mask_sig is _UNSET:
        mask_sig = _mask_signature(getattr(live_scan, "global_mask", None))

    return (
        id(live_scan),
        bool(integrate_1d),
        bool(integrate_2d),
        bool(getattr(live_scan, "gi", False)),
        repr(getattr(live_scan, "incidence_motor", None)),
        repr(getattr(live_scan, "tilt_angle", None)),
        repr(getattr(live_scan, "sample_orientation", None)),
        _items(getattr(live_scan, "bai_1d_args", {})),
        _items(getattr(live_scan, "bai_2d_args", {})),
        mask_sig,
    )


class StandardPlanCache:
    """Per-owner cache for the standard (non-GI) :class:`ReductionPlan`.

    Wrappers (wranglers, integrator threads) keep one instance for the
    lifetime of a scan; the cached plan is rebuilt only when one of the
    scan settings ``_plan_signature`` covers actually changes.

    GI scans now get real headless plans too; callers may still pass
    ``None`` explicitly to the dispatch helper as an escape hatch for a
    known-legacy site, but the cache no longer forces that fork.
    """

    __slots__ = ("_plan", "_key", "_mask_obj", "_mask_sig")

    def __init__(self) -> None:
        self._plan: ReductionPlan | None = None
        self._key: tuple | None = None
        # Memoized mask digest, keyed by the mask *object* (see below).
        self._mask_obj: Any = _UNSET
        self._mask_sig: Any = None

    def _mask_sig_for(self, mask: Any) -> Any:
        """Return the mask digest, recomputing the O(N) part only when the
        mask object itself changes.

        ``global_mask`` is built once per scan (detector mask + user mask)
        and *replaced* — not mutated in place — when the user swaps the
        mask file, so object identity is a sound proxy for "contents
        unchanged".  Holding a reference in ``_mask_obj`` also pins the id
        so it can't be reused by a later array.  This keeps the per-frame
        ``get()`` off the full-array ``np.sum`` that dominated mask digest
        cost on large detectors.
        """
        if mask is self._mask_obj:
            return self._mask_sig
        self._mask_obj = mask
        self._mask_sig = _mask_signature(mask)
        return self._mask_sig

    def get(
        self,
        live_scan: Any,
        *,
        integrate_1d: bool = True,
        integrate_2d: bool = True,
    ) -> ReductionPlan | None:
        mask_sig = self._mask_sig_for(getattr(live_scan, "global_mask", None))
        key = _plan_signature(
            live_scan, integrate_1d, integrate_2d, mask_sig=mask_sig,
        )
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
        self._mask_obj = _UNSET
        self._mask_sig = None


def dispatch_live_frame_reduction(
    live_frame: Any,
    live_scan: Any,
    *,
    standard_plan: ReductionPlan | None,
    integrator: Any,
    global_mask: Any,
) -> None:
    """Run reduction for one live frame through the headless path.

    Single dispatch point shared by wrangler workers.  Standard and GI frames
    both reduce through ``ssrl_xrd_tools.reduction.run_reduction`` now; a
    missing plan is a programmer error, not a fallback to xdart's old
    integration engine.

    Parameters
    ----------
    live_frame, live_scan
        The live frame to reduce and its parent live scan.
    standard_plan
        ``ReductionPlan`` for the headless path; pass ``None`` to force the
        legacy callback.
    integrator
        Pre-built pyFAI integrator for the worker (typically borrowed
        from an :class:`IntegratorPool`).
    global_mask
        Scan-level mask passed through unchanged.
    """
    if standard_plan is None:
        raise ValueError("dispatch_live_frame_reduction requires a ReductionPlan.")
    reduce_live_frame(
        live_frame,
        standard_plan,
        scan_name=str(getattr(live_scan, "name", "scan")),
        global_mask=global_mask,
        integrator=integrator,
    )


def sync_live_scan_gi_settings(
    live_scan: Any,
    *,
    incidence_motor: Any = None,
    sample_orientation: Any = None,
    tilt_angle: Any = None,
) -> None:
    """Mirror wrangler-thread GI settings onto a live scan before planning."""

    if not bool(getattr(live_scan, "gi", False)):
        return
    if incidence_motor is not None:
        live_scan.incidence_motor = incidence_motor
        live_scan.th_mtr = incidence_motor
    if sample_orientation is not None:
        live_scan.sample_orientation = sample_orientation
    if tilt_angle is not None:
        live_scan.tilt_angle = tilt_angle


def _source_path(frame: Any) -> Path | None:
    resolver = getattr(frame, "_resolved_source_path", None)
    path = resolver() if callable(resolver) else getattr(frame, "source_file", "")
    return Path(path) if path else None


def _incidence_available(live_scan: Any, incidence_motor: Any) -> bool:
    if incidence_motor is None:
        return False
    try:
        float(incidence_motor)
        return True
    except (TypeError, ValueError):
        pass
    key = str(incidence_motor).lower()
    scan_data = getattr(live_scan, "scan_data", None)
    if scan_data is not None and hasattr(scan_data, "columns"):
        if any(str(col).lower() == key for col in scan_data.columns):
            return True
    frames = getattr(live_scan, "frames", None)
    for idx in list(getattr(frames, "index", []) or []):
        try:
            info = getattr(frames[int(idx)], "scan_info", {}) or {}
        except Exception:
            continue
        if any(str(candidate).lower() == key for candidate in info):
            return True
    return False


def _scan_data_row(scan_data: Any, idx: int) -> dict[str, Any]:
    if scan_data is None or not hasattr(scan_data, "loc"):
        return {}
    try:
        row = scan_data.loc[int(idx)]
    except (KeyError, TypeError, ValueError):
        return {}
    if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
        row = row.iloc[0]
    try:
        return {
            str(key): value
            for key, value in row.to_dict().items()
        }
    except AttributeError:
        return {}


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
    # A mask that doesn't fit this image (wrong detector/calibration, a
    # resized frame, a stale flat-index mask, …) is ignored with a warning
    # rather than crashing the whole scan — reducing unmasked is far better
    # than aborting the run.  Structural problems degrade the same way.
    if arr.ndim == 2:
        if arr.shape != shape:
            logger.warning(
                "Ignoring mask: shape %s does not match image shape %s.",
                arr.shape, shape,
            )
            return None
        return arr.astype(bool, copy=False)
    if arr.ndim != 1:
        logger.warning("Ignoring mask: expected 1D flat mask, got shape %s.", arr.shape)
        return None
    if arr.dtype == bool:
        if arr.size != int(np.prod(shape)):
            logger.warning(
                "Ignoring boolean mask: length %d does not match image shape %s.",
                arr.size, shape,
            )
            return None
        return arr.reshape(shape)
    out = np.zeros(int(np.prod(shape)), dtype=bool)
    flat = np.asarray(arr, dtype=int).ravel()
    if flat.size and (flat.min() < 0 or flat.max() >= out.size):
        logger.warning(
            "Ignoring mask: flat indices out of bounds for image shape %s "
            "(index range [%d, %d], image has %d pixels).",
            shape, int(flat.min()), int(flat.max()), out.size,
        )
        return None
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


def _gi_1d_unit_default(unit: Any, mode: str, *, is_gi: bool) -> str:
    if not is_gi:
        return str(unit or "q_A^-1")
    if mode == "q_ip":
        return "qip_A^-1"
    if mode == "q_oop":
        return "qoop_A^-1"
    return str(unit or "q_A^-1")


def _gi_2d_unit_default(unit: Any, mode: str, *, is_gi: bool) -> str:
    text = str(unit or "").strip()
    if not is_gi:
        return text or "q_A^-1"
    if mode == "qip_qoop":
        return text if text.startswith("qip_") else "qip_A^-1"
    return text or "q_A^-1"


__all__ = [
    "StandardPlanCache",
    "dispatch_live_frame_reduction",
    "frame_from_live_frame",
    "scan_from_live_scan",
    "plan_from_live_scan",
    "reduce_live_frame",
    "reduce_live_frames",
    "sync_live_scan_gi_settings",
]
