"""Headless goniometer refinement — the producer/inverse of
:meth:`Diffractometer.from_pyfai_goniometer`.

Fits a per-axis-linear pyFAI ``GeometryTransformation`` (``rotN = scale·motor +
offset``) to identified powder control points across calibration frames at known
motor positions, by minimising the per-control-point |q| residual with
``scipy.least_squares`` (NOT pyFAI ``refine3``, whose simplex diverges for the
stacked psic — design §3.5).  The result is a fitted, axis-separable
``GeometryTransformation`` that round-trips straight through
``from_pyfai_goniometer`` into the canonical :class:`Diffractometer`.

This is the **Refine button** backend (Vivek, Jun 2026): a beamline picks a base
``.poni`` on one frame, adds calibration frames spanning the angular range, and
fits the goniometer offsets + scales headlessly.  pyFAI / scipy are lazy imports
so importing :mod:`xrd_tools.core` never pulls them in.

What the |q| residual **can** and **cannot** constrain (enforced here, not left
to the caller — powder |q| has real blind spots):

* ``rot3`` (rotation about the beam) leaves |q| invariant, so it is **never
  fit** — it is pinned at the base value.  Fitting it as a free LM parameter let
  it diverge to ~1e9 rad and silently corrupted the azimuth/cake/GI-χ/RSM of the
  returned object.
* a per-axis ``scale`` is only identifiable if that motor **spans** a range; a
  starved axis (e.g. ``del`` fixed) has a rank-deficient Jacobian and collapses
  the scale to garbage at a deceptively low in-sample RMS.  Such a scale is
  **frozen** at ``deg2rad`` and flagged, not fit.
* the fit is checked for physical plausibility + conditioning post-hoc;
  :attr:`RefineResult.success` is ``False`` (with a ``message``) when the fit is
  unusable, so a bad fit cannot masquerade as a good one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import Diffractometer

logger = logging.getLogger(__name__)

_DEG2RAD = float(np.deg2rad(1.0))
#: a motor must span at least this many degrees for its scale to be identifiable.
_MIN_SPAN_DEG = 1.0
#: a fitted scale this far (relative) from deg2rad is non-physical (real fits are
#: ~0.96–1.0·deg2rad); a starved-axis collapse lands orders of magnitude out.
_SCALE_TOL_REL = 0.5
#: Jacobian condition number above which the fit is under-determined.
_MAX_CONDITION = 1e8


@dataclass(frozen=True)
class ControlFrame:
    """One calibration frame's identified control points + its motor state.

    ``rows``/``cols`` are detector pixel indices (axis-1 / axis-2) of points
    lying on powder rings; ``rings`` is the integer ring index of each point
    (into the calibrant d-spacing list).  ``motors`` maps the goniometer motor
    names (e.g. ``{"nu": 5.0, "del": 20.0}``, degrees) to this frame's values.
    """

    rows: np.ndarray
    cols: np.ndarray
    rings: np.ndarray
    motors: Mapping[str, float]

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=int)
        cols = np.asarray(self.cols, dtype=int)
        rings = np.asarray(self.rings, dtype=int)
        if not (len(rows) == len(cols) == len(rings)):
            raise ValueError(
                f"control point arrays have mismatched lengths: rows={len(rows)},"
                f" cols={len(cols)}, rings={len(rings)} (motors={dict(self.motors)})")
        for name, arr in (("rows", rows), ("cols", cols), ("rings", rings)):
            if arr.size and arr.min() < 0:
                raise ValueError(
                    f"control point {name} has a negative index "
                    f"{int(arr.min())} (motors={dict(self.motors)}); pixel/ring "
                    "indices must be non-negative")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "rings", rings)


@dataclass(frozen=True)
class RefineResult:
    """The fitted :class:`Diffractometer` plus fit diagnostics.

    ``success`` is ``True`` only when the optimiser converged AND the result is
    physically plausible AND well-conditioned — a low ``rms_q`` alone is *not*
    sufficient (a rank-deficient fit is deceptively low-RMS in-sample).
    """

    diffractometer: Diffractometer
    rms_q: float                      # RMS |q| residual (Å⁻¹)
    n_points: int
    params: Mapping[str, float]       # the fitted GeometryTransformation params
    success: bool
    condition_number: float           # Jacobian condition number (∞ if unavailable)
    frozen_scales: tuple[str, ...]    # axes whose scale was frozen (motor un-spanned)
    message: str                      # "" when good; else why the fit is unusable


def _resolve_calibrant_q(calibrant: Any, wavelength_m: float,
                         n_rings: int) -> np.ndarray:
    """Ring index → |q| (Å⁻¹) for the calibrant: q = 2π / d, d in Å."""
    if isinstance(calibrant, str):
        from pyFAI.calibrant import get_calibrant  # noqa: PLC0415
        cal = get_calibrant(calibrant)
    else:
        cal = calibrant
    cal.set_wavelength(float(wavelength_m))
    d = np.asarray(cal.get_dSpacing(), dtype=float)  # Å, descending
    if n_rings > len(d):
        raise ValueError(
            f"control points reference ring {n_rings - 1} but the calibrant "
            f"only has {len(d)} d-spacings")
    return 2.0 * np.pi / d


def _build_detector(name: str, detector_config: Mapping[str, Any] | None):
    from pyFAI.detectors import detector_factory  # noqa: PLC0415
    cfg = dict(detector_config or {})
    # A pyFAI too old for config= raises here — surface it (do NOT silently drop
    # the orientation, which would put the rings in the wrong place).
    return detector_factory(name, config=cfg) if cfg else detector_factory(name)


def _detector_config_from_poni(path: str | Path) -> dict[str, Any]:
    """Read ``Detector_config`` (e.g. ``{"orientation": 3}``) from a .poni, with
    any pyFAI enum coerced to a plain int so it is JSON-serialisable."""
    try:
        from pyFAI.io.ponifile import PoniFile  # noqa: PLC0415
        det = PoniFile(str(path)).detector
        cfg = dict(det.get_config()) if det is not None else {}
    except Exception:
        logger.debug("could not read Detector_config from %s", path, exc_info=True)
        return {}
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        out[k] = int(v) if hasattr(v, "__int__") and not isinstance(v, bool) else v
    return out


def refine_goniometer(
    base_calibration: PONI | str | Path,
    control_frames: Sequence[ControlFrame | Mapping[str, Any]],
    *,
    rot1_motor: str,
    rot2_motor: str,
    calibrant: Any = "LaB6",
    wavelength: float | None = None,
    detector: str | None = None,
    detector_config: Mapping[str, Any] | None = None,
    fit_scales: bool = True,
    base: Diffractometer | None = None,
    preset: str = "fitted",
    max_nfev: int = 400,
) -> RefineResult:
    """Fit an axis-separable goniometer to powder control points.

    Parameters
    ----------
    base_calibration : PONI or .poni path
        Seeds ``dist``/``poni1``/``poni2``/``rot3`` and supplies the detector +
        wavelength when not given explicitly.  When it is a ``.poni`` *path*, its
        ``Detector_config`` (e.g. the panel ``orientation``) is read and used
        unless ``detector_config=`` is given; a bare :class:`PONI` carries no
        detector config, so pass ``detector_config=`` explicitly in that case.
    control_frames : sequence of ControlFrame (or dict)
        One per calibration frame: identified ring points + the frame's motor
        positions (must include ``rot1_motor`` and ``rot2_motor``, degrees).
    rot1_motor, rot2_motor : str
        Which motor drives ``rot1`` / ``rot2`` (e.g. ``"nu"`` / ``"del"`` for a
        psic).  The fit is ``rotN = scale·motor + offset``; ``rot3`` is pinned at
        the base value (powder |q| cannot constrain a beam-axis rotation).
    fit_scales : bool
        Fit the per-axis scales (default).  A scale is fit ONLY if its motor
        spans ≥ 1°; an un-spanned axis's scale is frozen at ``deg2rad`` (it is
        unidentifiable) and reported in ``frozen_scales``.  ``False`` fixes both
        scales at ``deg2rad`` and fits only the motor-zero offsets + base.
    base : Diffractometer
        Donates the xrayutilities half (circle stacks, ``camera``, HXRD refs).

    Returns
    -------
    RefineResult
    """
    from scipy.optimize import least_squares  # noqa: PLC0415
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: PLC0415

    # --- base geometry -----------------------------------------------------
    base_poni: PONI
    if isinstance(base_calibration, PONI):
        base_poni = base_calibration
    else:
        from xrd_tools.integrate.calibration import load_poni  # noqa: PLC0415
        base_poni = load_poni(base_calibration)
        if detector_config is None:
            detector_config = _detector_config_from_poni(base_calibration)
    wl_m = float(wavelength if wavelength is not None else base_poni.wavelength)
    if not wl_m:
        raise ValueError("wavelength is required (give wavelength= or a base "
                         ".poni carrying one)")
    det_name = detector or base_poni.detector
    if not det_name:
        raise ValueError("detector is required (give detector= or a base .poni "
                         "naming one)")
    det = _build_detector(det_name, detector_config)

    # --- control points ----------------------------------------------------
    frames = [f if isinstance(f, ControlFrame) else ControlFrame(**f)
              for f in control_frames]
    if not frames:
        raise ValueError("control_frames is empty")
    max_ring = max(int(f.rings.max()) for f in frames if f.rings.size) + 1
    q_ring = _resolve_calibrant_q(calibrant, wl_m, max_ring)
    n_points = int(sum(len(f.rows) for f in frames))

    # --- identifiability: a scale needs its motor to span a range ----------
    span1 = float(np.ptp([f.motors[rot1_motor] for f in frames]))
    span2 = float(np.ptp([f.motors[rot2_motor] for f in frames]))
    fit_r1s = fit_scales and span1 >= _MIN_SPAN_DEG
    fit_r2s = fit_scales and span2 >= _MIN_SPAN_DEG
    frozen: list[str] = []
    if fit_scales and not fit_r1s:
        frozen.append(rot1_motor)
        logger.warning("refine_goniometer: %s spans only %.3g° (< %g°) — its "
                       "scale is unidentifiable and frozen at deg2rad; add "
                       "frames spanning %s.", rot1_motor, span1, _MIN_SPAN_DEG,
                       rot1_motor)
    if fit_scales and not fit_r2s:
        frozen.append(rot2_motor)
        logger.warning("refine_goniometer: %s spans only %.3g° (< %g°) — its "
                       "scale is unidentifiable and frozen at deg2rad; add "
                       "frames spanning %s.", rot2_motor, span2, _MIN_SPAN_DEG,
                       rot2_motor)

    # --- parameter packing (rot3 is NEVER fit; pinned at base) -------------
    # p = [dist, poni1, poni2, rot1_offset, rot2_offset, (rot1_scale?), (rot2_scale?)]
    seed = [base_poni.dist, base_poni.poni1, base_poni.poni2, 0.0, 0.0]
    if fit_r1s:
        seed.append(_DEG2RAD)
    if fit_r2s:
        seed.append(_DEG2RAD)
    seed = np.asarray(seed, dtype=float)
    r3_pinned = float(base_poni.rot3)

    def _unpack(p):
        dist, poni1, poni2, r1o, r2o = p[:5]
        i = 5
        r1s = p[i] if fit_r1s else _DEG2RAD
        i += int(fit_r1s)
        r2s = p[i] if fit_r2s else _DEG2RAD
        return dist, poni1, poni2, r1o, r2o, r3_pinned, r1s, r2s

    def _residual(p):
        dist, poni1, poni2, r1o, r2o, r3o, r1s, r2s = _unpack(p)
        out = []
        for f in frames:
            m1 = float(f.motors[rot1_motor])
            m2 = float(f.motors[rot2_motor])
            ai = AzimuthalIntegrator(
                dist=dist, poni1=poni1, poni2=poni2,
                rot1=r1s * m1 + r1o, rot2=r2s * m2 + r2o, rot3=r3o,
                detector=det, wavelength=wl_m,
            )
            q_pred = ai.qArray()[f.rows, f.cols] / 10.0  # pyFAI q is nm⁻¹ → Å⁻¹
            out.append(q_pred - q_ring[f.rings])
        return np.concatenate(out)

    res = least_squares(_residual, seed, method="lm", max_nfev=max_nfev,
                        x_scale="jac")
    rms = float(np.sqrt(np.mean(res.fun ** 2)))
    dist, poni1, poni2, r1o, r2o, r3o, r1s, r2s = _unpack(res.x)

    # --- conditioning (the Jacobian least_squares already returns) ---------
    try:
        sv = np.linalg.svd(np.asarray(res.jac), compute_uv=False)
        condition = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    except Exception:  # noqa: BLE001
        condition = float("inf")

    # --- physical plausibility + conditioning → the success signal ---------
    problems: list[str] = []
    if not res.success:
        problems.append(f"optimiser did not converge (status={res.status})")
    if not (0.0 < dist < 100.0):
        problems.append(f"non-physical dist={dist:.4g} m")
    if abs(r1o) > np.pi or abs(r2o) > np.pi:
        problems.append("non-physical rotation offset (> π rad)")
    for fit_s, sc, mot in ((fit_r1s, r1s, rot1_motor), (fit_r2s, r2s, rot2_motor)):
        if fit_s and abs(sc - _DEG2RAD) > _SCALE_TOL_REL * _DEG2RAD:
            problems.append(
                f"{mot} scale {sc:.4g} is implausible (expected ≈ deg2rad "
                f"{_DEG2RAD:.4g}) — likely under-constrained")
    if condition > _MAX_CONDITION:
        problems.append(f"under-determined fit (Jacobian condition {condition:.2g}"
                        f" > {_MAX_CONDITION:.0g}); add frames spanning each motor")
    ok = not problems
    message = "" if ok else "; ".join(problems)
    log = logger.info if ok else logger.warning
    log("refine_goniometer: RMS |q| = %.5g Å⁻¹ over %d points (%d frames), "
        "cond=%.3g, success=%s%s", rms, n_points, len(frames), condition, ok,
        "" if ok else f" — {message}")

    # --- canonical object via the standard GeometryTransformation ----------
    # Safe placeholder position names in the expressions (a real motor name like
    # ``del`` is a Python keyword and would break the numexpr evaluation in
    # from_pyfai_goniometer); map them back via source_motors.
    param_names = ["dist", "poni1", "poni2", "rot1_offset", "rot1_scale",
                   "rot2_offset", "rot2_scale", "rot3_offset"]
    param = [dist, poni1, poni2, r1o, r1s, r2o, r2s, r3o]
    gonio = {
        "content": "Goniometer calibration v2",
        "detector": det_name,
        "detector_config": dict(detector_config or {}),
        "wavelength": wl_m,
        "param_names": param_names,
        "param": param,
        "pos_names": ["rot1_pos", "rot2_pos"],
        "trans_function": {
            "content": "GeometryTransformation",
            "param_names": param_names,
            "pos_names": ["rot1_pos", "rot2_pos"],
            "dist_expr": "dist", "poni1_expr": "poni1", "poni2_expr": "poni2",
            "rot1_expr": "rot1_scale * rot1_pos + rot1_offset",
            "rot2_expr": "rot2_scale * rot2_pos + rot2_offset",
            "rot3_expr": "rot3_offset",
            "constants": {"pi": float(np.pi)},
        },
    }
    diff = Diffractometer.from_pyfai_goniometer(
        gonio, source_motors={"rot1_pos": rot1_motor, "rot2_pos": rot2_motor},
        base=base, preset=preset)
    return RefineResult(
        diffractometer=diff, rms_q=rms, n_points=n_points,
        params=dict(zip(param_names, param)), success=ok,
        condition_number=condition, frozen_scales=tuple(frozen), message=message,
    )


__all__ = ["ControlFrame", "RefineResult", "refine_goniometer"]
