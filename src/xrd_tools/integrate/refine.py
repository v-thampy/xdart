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
        object.__setattr__(self, "rows", np.asarray(self.rows, dtype=int))
        object.__setattr__(self, "cols", np.asarray(self.cols, dtype=int))
        object.__setattr__(self, "rings", np.asarray(self.rings, dtype=int))


@dataclass(frozen=True)
class RefineResult:
    """The fitted :class:`Diffractometer` plus fit diagnostics."""

    diffractometer: Diffractometer
    rms_q: float                      # RMS |q| residual (Å⁻¹)
    n_points: int
    params: Mapping[str, float]       # the fitted GeometryTransformation params
    success: bool


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
    try:
        return detector_factory(name, config=cfg) if cfg else detector_factory(name)
    except TypeError:  # very old pyFAI without config=
        det = detector_factory(name)
        if "orientation" in cfg and hasattr(det, "set_orientation"):
            det.set_orientation(cfg["orientation"])
        return det


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
        Seeds ``dist``/``poni1``/``poni2`` and supplies the detector +
        ``detector_config`` + wavelength when not given explicitly.
    control_frames : sequence of ControlFrame (or dict)
        One per calibration frame: identified ring points + the frame's motor
        positions (must include ``rot1_motor`` and ``rot2_motor``, degrees).
    rot1_motor, rot2_motor : str
        Which motor drives ``rot1`` / ``rot2`` (e.g. ``"nu"`` / ``"del"`` for a
        psic).  The fit is ``rotN = scale·motor + offset``.
    calibrant : str or pyFAI Calibrant
        Powder standard (default ``"LaB6"``) → ring |q| = 2π/d.
    fit_scales : bool
        Fit the per-axis scales (default).  When ``False`` the scales are fixed
        at ``deg2rad`` (the uncalibrated convention) and only the motor-zero
        offsets + base ``dist``/``poni`` are fit.
    base : Diffractometer
        Donates the xrayutilities half (circle stacks, ``camera``, HXRD refs) —
        the control-point |q| fit does not constrain it.  Pass
        ``Diffractometer.psic()`` so the result also feeds RSM.

    Returns
    -------
    RefineResult
        ``.diffractometer`` (the fitted canonical object), ``.rms_q`` (Å⁻¹),
        ``.n_points``, ``.params``, ``.success``.
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
    max_ring = max(int(f.rings.max()) for f in frames) + 1
    q_ring = _resolve_calibrant_q(calibrant, wl_m, max_ring)
    n_points = int(sum(len(f.rows) for f in frames))

    # --- parameter packing -------------------------------------------------
    # p = [dist, poni1, poni2, rot1_offset, rot2_offset, rot3_offset,
    #      (rot1_scale, rot2_scale if fit_scales)]
    seed = [base_poni.dist, base_poni.poni1, base_poni.poni2, 0.0, 0.0, 0.0]
    if fit_scales:
        seed += [_DEG2RAD, _DEG2RAD]
    seed = np.asarray(seed, dtype=float)

    def _unpack(p):
        dist, poni1, poni2, r1o, r2o, r3o = p[:6]
        r1s, r2s = (p[6], p[7]) if fit_scales else (_DEG2RAD, _DEG2RAD)
        return dist, poni1, poni2, r1o, r2o, r3o, r1s, r2s

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
    logger.info("refine_goniometer: RMS |q| = %.5g Å⁻¹ over %d points "
                "(%d frames), success=%s", rms, n_points, len(frames),
                res.success)

    # --- canonical object via the standard GeometryTransformation ----------
    # Use safe placeholder position names in the expressions (a real motor
    # name like ``del`` is a Python keyword and would break the numexpr
    # evaluation in from_pyfai_goniometer); map them back to the real motor
    # names via source_motors.  This is exactly why pyFAI itself names the
    # del/nu axes ``del_value``/``nu_value``.
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
        params=dict(zip(param_names, param)), success=bool(res.success),
    )


__all__ = ["ControlFrame", "RefineResult", "refine_goniometer"]
