"""Headless goniometer refinement — the producer/inverse of
:meth:`Diffractometer.from_pyfai_goniometer`.

Fits an affine or exact del-on-nu pyFAI ``GeometryTransformation`` to identified
powder control points across calibration frames at known motor positions,
by minimising the per-control-point |q| residual with
``scipy.least_squares`` (NOT pyFAI ``refine3``, whose simplex diverges for the
stacked psic — design §3.5).  The result is a fitted
``GeometryTransformation`` that round-trips straight through
``from_pyfai_goniometer`` into the canonical :class:`Diffractometer`.

This is the **Refine button** backend (Vivek, Jun 2026): a beamline picks a base
``.poni`` on one frame, adds calibration frames spanning the angular range, and
fits the goniometer offsets + scales headlessly.  pyFAI / scipy are lazy imports
so importing :mod:`xrd_tools.core` never pulls them in.

What the |q| residual **can** and **cannot** constrain (enforced here, not left
to the caller — powder |q| has real blind spots):

* A free beam-axis rotation leaves |q| invariant, so it is **never fit** —
  it is pinned at the base value. The stacked model additionally derives a
  motor-dependent Euler ``rot3`` from the physical pose. Fitting beam rotation as a free parameter let
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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import Diffractometer, DetectorCalibration, StackedDetectorArm
from xrd_tools.core.geometry.stacked_arm import orientation_signs

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
    ``row_positions``/``col_positions`` keep the sub-pixel coordinates (pyFAI
    convention: an integer is a pixel centre); they default to the values given
    for ``rows``/``cols`` before those are truncated to indices.
    """

    rows: np.ndarray
    cols: np.ndarray
    rings: np.ndarray
    motors: Mapping[str, float]
    row_positions: np.ndarray | None = None
    col_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        row_positions = np.asarray(
            self.rows if self.row_positions is None else self.row_positions, dtype=float)
        col_positions = np.asarray(
            self.cols if self.col_positions is None else self.col_positions, dtype=float)
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
        if (row_positions.shape != rows.shape or col_positions.shape != cols.shape
                or not np.isfinite(row_positions).all() or not np.isfinite(col_positions).all()):
            raise ValueError("sub-pixel control point positions must be finite and match the indices")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "rings", rings)
        object.__setattr__(self, "row_positions", row_positions)
        object.__setattr__(self, "col_positions", col_positions)


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
    #: one-sigma parameter errors from the Jacobian (arm extension fits only)
    param_errors: Mapping[str, float] = field(default_factory=dict)
    #: control points dropped by robust outlier rejection
    rejected_points: int = 0
    #: parameter correlation matrix {"names": [...], "matrix": [[...]]} (arm fits only)
    param_correlations: Mapping[str, Any] = field(default_factory=dict)


def _resolve_calibrant_q(calibrant: Any, wavelength_m: float,
                         n_rings: int) -> np.ndarray:
    """Ring index → |q| (Å⁻¹) for the calibrant: q = 2π / d, d in Å."""
    if isinstance(calibrant, str):
        from pyFAI.calibrant import get_calibrant  # noqa: PLC0415
        cal = get_calibrant(calibrant)
    else:
        cal = calibrant
    cal.wavelength = float(wavelength_m)
    d = np.asarray(cal.dspacing, dtype=float)  # Å, descending
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


def _base_geometry(base_calibration, wavelength, detector, detector_config):
    """Seed PONI, parallax, detector config, wavelength and detector for a fit."""
    base_poni: PONI
    base_parallax: bool | None = None
    if isinstance(base_calibration, PONI):
        base_poni = base_calibration
    else:
        from xrd_tools.integrate.calibration import (  # noqa: PLC0415
            load_detector_calibration,
        )
        strict_calibration = load_detector_calibration(base_calibration)
        base_poni = strict_calibration.poni
        if detector_config is None:
            detector_config = dict(strict_calibration.detector_config)
        elif (
            strict_calibration.parallax is not None
            and dict(detector_config) != dict(strict_calibration.detector_config)
        ):
            raise ValueError(
                "PONI v3 refinement cannot replace its accepted detector config"
            )
        base_parallax = strict_calibration.parallax
    wl_m = float(wavelength if wavelength is not None else base_poni.wavelength)
    if not wl_m:
        raise ValueError("wavelength is required (give wavelength= or a base "
                         ".poni carrying one)")
    det_name = detector or base_poni.detector
    if not det_name:
        raise ValueError("detector is required (give detector= or a base .poni "
                         "naming one)")
    det = _build_detector(det_name, detector_config)
    return base_poni, base_parallax, detector_config, wl_m, det_name, det


def refine_goniometer(
    base_calibration: PONI | str | Path,
    control_frames: Sequence[ControlFrame | Mapping[str, Any]],
    *,
    rot1_motor: str | None,
    rot2_motor: str | None,
    calibrant: Any = "LaB6",
    wavelength: float | None = None,
    detector: str | None = None,
    detector_config: Mapping[str, Any] | None = None,
    fit_scales: bool = True,
    base: Diffractometer | None = None,
    preset: str = "fitted",
    max_nfev: int = 400,
    distance_seed_m: float | None = None,
    distance_bounds_m: tuple[float, float] | None = None,
    cancelled=None,
    model: str = 'affine',
    beam_rotation: float | None = None,
) -> RefineResult:
    """Fit affine or exact del-on-nu geometry to powder control points.

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
        One of them ``None`` is a single detector arm (a fourc or twoc 2θ
        circle): that circle is held at zero and only its offset is fitted —
        ``rot1_motor=None`` for a vertical 2θ arm (del), ``rot2_motor=None``
        for a horizontal one (nu).
    fit_scales : bool
        Fit the per-axis scales (default).  A scale is fit ONLY if its motor
        spans ≥ 1°; an un-spanned axis's scale is frozen at ``deg2rad`` (it is
        unidentifiable) and reported in ``frozen_scales``.  ``False`` fixes both
        scales at ``deg2rad`` and fits only the motor-zero offsets + base.
        ``deg2rad`` carries the detector orientation's sign for that rotation
        (:func:`~xrd_tools.core.geometry.stacked_arm.orientation_signs`), so a
        motor turns the detector the same way in every pyFAI orientation.
    base : Diffractometer
        Donates the xrayutilities half (circle stacks, ``camera``, HXRD refs).
    distance_seed_m, distance_bounds_m
        Optional perpendicular detector distance seed and explicit fit bounds,
        in metres. A seed alone does not constrain the fitted distance. Bounds
        use the bounded trust-region solver; the historical unconstrained LM
        path is unchanged when bounds are absent.
    model : {'affine', 'stacked_del_on_nu', 'stacked_del_on_nu_extended'}
        Explicitly choose the physical nesting. The legacy default stays affine.
        In stacked mode rot1_motor is outer nu and rot2_motor is inner del;
        both offsets are motor zero offsets in radians. The induced rot3
        is computed from the pose, not fitted as an unconstrained powder angle.
        The extended model also fits the detector mount (tilt, roll) and the
        sample offset from the rotation centre; see :func:`refine_stacked_arm`.
    beam_rotation : float, optional
        The pinned rotation about the beam (``rot3``'s offset, radians); |q|
        cannot see it. ``None`` keeps the base PONI's ``rot3`` (the extended
        model: zero).

    Returns
    -------
    RefineResult
    """
    from scipy.optimize import least_squares  # noqa: PLC0415
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: PLC0415
    if model == 'stacked_del_on_nu_extended':
        if fit_scales:
            raise ValueError('motor scales are not fitted together with the arm extension')
        return refine_stacked_arm(
            base_calibration, control_frames, nu_motor=rot1_motor, del_motor=rot2_motor,
            calibrant=calibrant, wavelength=wavelength, detector=detector,
            detector_config=detector_config, base=base, preset=preset,
            distance_seed_m=distance_seed_m, distance_bounds_m=distance_bounds_m,
            max_nfev=max(max_nfev, 2000), cancelled=cancelled,
            beam_rotation=0. if beam_rotation is None else float(beam_rotation))
    if model not in {'affine', 'stacked_del_on_nu'}:
        raise ValueError('unknown goniometer model')
    if rot1_motor is None and rot2_motor is None:
        raise ValueError('a detector arm needs at least one moving circle')
    if (rot1_motor is None or rot2_motor is None) and model != 'stacked_del_on_nu':
        raise ValueError('a single detector arm is fitted as the stacked arm with one circle held at zero')

    # --- base geometry -----------------------------------------------------
    base_poni, base_parallax, detector_config, wl_m, det_name, det = _base_geometry(
        base_calibration, wavelength, detector, detector_config)

    # --- control points ----------------------------------------------------
    frames = [f if isinstance(f, ControlFrame) else ControlFrame(**f)
              for f in control_frames]
    if not frames:
        raise ValueError("control_frames is empty")
    max_ring = max(int(f.rings.max()) for f in frames if f.rings.size) + 1
    q_ring = _resolve_calibrant_q(calibrant, wl_m, max_ring)
    n_points = int(sum(len(f.rows) for f in frames))

    # --- identifiability: a scale needs its motor to span a range ----------
    single = rot1_motor is None or rot2_motor is None  # a single detector arm: one circle held

    def reader(name):  # a held circle reads zero
        return (lambda f: 0.0) if name is None else (lambda f: float(f.motors[name]))
    motor1, motor2 = reader(rot1_motor), reader(rot2_motor)
    span1 = float(np.ptp([motor1(f) for f in frames]))
    span2 = float(np.ptp([motor2(f) for f in frames]))
    fit_r1s = fit_scales and rot1_motor is not None and span1 >= _MIN_SPAN_DEG
    fit_r2s = fit_scales and rot2_motor is not None and span2 >= _MIN_SPAN_DEG
    frozen: list[str] = []
    if fit_scales and rot1_motor is not None and not fit_r1s:
        frozen.append(rot1_motor)
        logger.warning("refine_goniometer: %s spans only %.3g° (< %g°) — its "
                       "scale is unidentifiable and frozen at deg2rad; add "
                       "frames spanning %s.", rot1_motor, span1, _MIN_SPAN_DEG,
                       rot1_motor)
    if fit_scales and rot2_motor is not None and not fit_r2s:
        frozen.append(rot2_motor)
        logger.warning("refine_goniometer: %s spans only %.3g° (< %g°) — its "
                       "scale is unidentifiable and frozen at deg2rad; add "
                       "frames spanning %s.", rot2_motor, span2, _MIN_SPAN_DEG,
                       rot2_motor)

    # --- parameter packing (rot3 is NEVER fit; pinned at base) -------------
    # p = [dist, poni1, poni2, rot1_offset, rot2_offset, (rot1_scale?), (rot2_scale?)]
    distance = base_poni.dist if distance_seed_m is None else float(distance_seed_m)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance seed must be finite and positive metres")
    if distance_bounds_m is not None:
        lo, hi = (float(v) for v in distance_bounds_m)
        if not (np.isfinite([lo, hi]).all() and 0 < lo < hi
                and lo <= distance <= hi):
            raise ValueError("distance bounds require 0 < low < high and a seed inside them")
    seed = [distance, base_poni.poni1, base_poni.poni2, 0.0, 0.0]
    r1_sign, r2_sign = orientation_signs(det.orientation)
    r1_nominal, r2_nominal = r1_sign * _DEG2RAD, r2_sign * _DEG2RAD
    if fit_r1s:
        seed.append(r1_nominal)
    if fit_r2s:
        seed.append(r2_nominal)
    seed = np.asarray(seed, dtype=float)
    r3_pinned = float(base_poni.rot3 if beam_rotation is None else beam_rotation)

    def _unpack(p):
        dist, poni1, poni2, r1o, r2o = p[:5]
        i = 5
        r1s = p[i] if fit_r1s else r1_nominal
        i += int(fit_r1s)
        r2s = p[i] if fit_r2s else r2_nominal
        return dist, poni1, poni2, r1o, r2o, r3_pinned, r1s, r2s

    def _residual(p):
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("goniometer refinement cancelled")
        dist, poni1, poni2, r1o, r2o, r3o, r1s, r2s = _unpack(p)
        out = []
        for f in frames:
            m1 = motor1(f)
            m2 = motor2(f)
            rotations = (r1s * m1 + r1o, r2s * m2 + r2o, r3o)
            if model == 'stacked_del_on_nu':
                rotations = StackedDetectorArm(nu_motor=rot1_motor, del_motor=rot2_motor,
                    nu_offset=r1o, nu_scale=r1s, del_offset=r2o, del_scale=r2s,
                    beam_rotation=r3o).angles(m2, m1)
            ai = AzimuthalIntegrator(
                dist=dist, poni1=poni1, poni2=poni2,
                rot1=rotations[0], rot2=rotations[1], rot3=rotations[2],
                detector=det, wavelength=wl_m,
            )
            if base_parallax is True:
                ai.enable_parallax()
            q_pred = ai.qArray()[f.rows, f.cols] / 10.0  # pyFAI q is nm⁻¹ → Å⁻¹
            out.append(q_pred - q_ring[f.rings])
        return np.concatenate(out)

    bounds = (-np.inf, np.inf)
    if distance_bounds_m is not None:
        lower, upper = np.full(len(seed), -np.inf), np.full(len(seed), np.inf)
        lower[0], upper[0] = lo, hi
        bounds = (lower, upper)
    res = least_squares(_residual, seed,
                        method="lm" if distance_bounds_m is None else "trf",
                        bounds=bounds, max_nfev=max_nfev, x_scale="jac")
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
    for fit_s, sc, mot, nominal in ((fit_r1s, r1s, rot1_motor, r1_nominal),
                                    (fit_r2s, r2s, rot2_motor, r2_nominal)):
        if fit_s and abs(sc - nominal) > _SCALE_TOL_REL * _DEG2RAD:
            problems.append(
                f"{mot} scale {sc:.4g} is implausible (expected ≈ deg2rad "
                f"{nominal:.4g}) — likely under-constrained")
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
    if base_parallax is not None:
        gonio["parallax"] = base_parallax
    sources = {"rot1_pos": rot1_motor, "rot2_pos": rot2_motor}
    if model == 'stacked_del_on_nu':
        arm = StackedDetectorArm(nu_motor=rot1_motor, del_motor=rot2_motor,
            nu_offset=r1o, nu_scale=r1s, del_offset=r2o, del_scale=r2s,
            beam_rotation=r3o)
        calibration = DetectorCalibration(
            replace(base_poni, dist=dist, poni1=poni1, poni2=poni2,
                    rot1=0., rot2=0., rot3=0., wavelength=wl_m, detector=det_name),
            detector_config=dict(detector_config or {}), parallax=base_parallax,
        )
        gonio = arm.to_pyfai_goniometer(calibration)
        param_names, param = gonio['param_names'], gonio['param']
        sources = {name: motor for name, motor in (('nu_angle', rot1_motor), ('del_angle', rot2_motor))
                   if motor is not None}
    diff = Diffractometer.from_pyfai_goniometer(
        gonio, source_motors=sources,
        base=base, preset=preset)
    return RefineResult(
        diffractometer=diff, rms_q=rms, n_points=n_points,
        params=dict(zip(param_names, param)), success=ok,
        condition_number=condition, frozen_scales=tuple(frozen), message=message,
    )


#: Plausibility of the arm extension. The scan-14 prototype fitted a mount tilt
#: of about 1 degree, a roll of about -0.3 degree and sample offsets of about 2 mm.
_MAX_MOUNT_RAD = float(np.deg2rad(5.0))
_MAX_SAMPLE_OFFSET_M = 0.02
_MAX_MOUNT_ERROR_RAD = float(np.deg2rad(0.5))
_MAX_SAMPLE_OFFSET_ERROR_M = 0.002
#: The extension is identified only by frames from several nu positions.
_MIN_NU_POSITIONS = 3
#: Robust outlier cuts never go below this |q| residual (about 3e-4 px), so
#: near-perfect control points are all kept instead of a zero-width cut.
_MIN_OUTLIER_CUT_Q = 1e-6
_ARM_NAMES = ("dist", "poni1", "poni2", "nu_offset", "del_offset")
_EXTENSION_NAMES = ("mount_tilt", "mount_roll", "sample_offset1", "sample_offset2", "sample_offset3")


def refine_stacked_arm(
    base_calibration: PONI | str | Path,
    control_frames: Sequence[ControlFrame | Mapping[str, Any]],
    *,
    nu_motor: str | None,
    del_motor: str | None,
    calibrant: Any = "LaB6",
    wavelength: float | None = None,
    detector: str | None = None,
    detector_config: Mapping[str, Any] | None = None,
    extended: bool = True,
    reject_outliers: float | None = None,
    beam_rotation: float = 0.0,
    base: Diffractometer | None = None,
    preset: str = "fitted",
    distance_seed_m: float | None = None,
    distance_bounds_m: tuple[float, float] | None = None,
    max_nfev: int = 2000,
    cancelled=None,
) -> RefineResult:
    """Fit the del-on-nu arm at sub-pixel control-point positions.

    The five arm parameters are the distance, the PONI and the nu and del zero
    offsets. ``nu_motor=None`` or ``del_motor=None`` is a single detector arm
    with that circle held at zero; its offset (the arm's yaw, or the detector's
    pitch) is still fitted, and it has no extension.
    ``extended=True`` adds the detector mount on the arm (tilt, roll)
    and the sample offset from the rotation centre. Motor scales stay at one
    degree per degree (signed for the detector orientation, see
    :func:`~xrd_tools.core.geometry.stacked_arm.orientation_signs`) and the beam rotation is pinned at zero, because |q|
    cannot see either; a mount rotation about the del axis is excluded because
    it duplicates the del offset. The extension needs frames from at least
    three nu positions, and a fit whose extension is implausible or poorly
    determined is reported unsuccessful.

    ``reject_outliers`` drops points beyond that many robust (MAD) sigmas and
    refits, twice; use it for automatically extracted points. ``beam_rotation``
    is carried unchanged: |q| cannot see it, but it sets the chi reference.

    With a parallax-enabled calibration every trial pose is evaluated through
    the runtime integrator builder, because pyFAI's parallax correction reads
    the integrator's own pose rather than a trial ``param``.
    """
    from scipy.optimize import least_squares  # noqa: PLC0415
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: PLC0415

    base_poni, base_parallax, detector_config, wl_m, det_name, det = _base_geometry(
        base_calibration, wavelength, detector, detector_config)
    frames = [f if isinstance(f, ControlFrame) else ControlFrame(**f)
              for f in control_frames]
    frames = [f for f in frames if len(f.rows)]
    if not frames:
        raise ValueError("control_frames is empty")
    if nu_motor is None and del_motor is None:
        raise ValueError("a detector arm needs at least one moving circle")
    nu_deg = np.array([0.0 if nu_motor is None else float(f.motors[nu_motor]) for f in frames])
    del_deg = np.array([0.0 if del_motor is None else float(f.motors[del_motor]) for f in frames])
    if extended and (nu_motor is None or del_motor is None):
        raise ValueError("a single detector arm has no nu positions, which the arm extension "
                         "needs; fit it without the extension")
    if extended and len(np.unique(np.round(nu_deg, 6))) < _MIN_NU_POSITIONS:
        raise ValueError(
            f"the arm extension needs control frames from at least {_MIN_NU_POSITIONS} "
            "nu positions; add frames spanning nu or fit without the extension")
    max_ring = max(int(f.rings.max()) for f in frames) + 1
    q_ring = _resolve_calibrant_q(calibrant, wl_m, max_ring)
    targets = [q_ring[f.rings] for f in frames]
    geometry = AzimuthalIntegrator(detector=det, wavelength=wl_m)
    runtime = None
    if base_parallax is True:
        from xrd_tools.integrate.calibration import (  # noqa: PLC0415
            detector_calibration_to_integrator,
        )
        template = DetectorCalibration(
            replace(base_poni, rot1=0., rot2=0., rot3=0., wavelength=wl_m, detector=det_name),
            detector_config=dict(detector_config or {}), parallax=True)

        def runtime(param):
            return detector_calibration_to_integrator(
                template, dist=param[0], poni1=param[1], poni2=param[2],
                rot1=param[3], rot2=param[4], rot3=param[5])

    distance = base_poni.dist if distance_seed_m is None else float(distance_seed_m)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance seed must be finite and positive metres")
    bounds = (-np.inf, np.inf)
    if distance_bounds_m is not None:
        lo, hi = (float(v) for v in distance_bounds_m)
        if not (np.isfinite([lo, hi]).all() and 0 < lo < hi and lo <= distance <= hi):
            raise ValueError("distance bounds require 0 < low < high and a seed inside them")

    nu_sign, del_sign = orientation_signs(det.orientation)

    def arm_for(p):
        extension = dict(zip(_EXTENSION_NAMES, p[5:])) if len(p) > 5 else {}
        return StackedDetectorArm(nu_motor=nu_motor, del_motor=del_motor,
                                  nu_offset=p[3], del_offset=p[4],
                                  nu_scale=nu_sign * _DEG2RAD, del_scale=del_sign * _DEG2RAD,
                                  beam_rotation=beam_rotation, **extension)

    def predicted(p, keep):
        arm = arm_for(p)
        rot1, rot2, rot3 = arm.angles(del_deg, nu_deg)
        d_dist, d_poni1, d_poni2 = arm.displacements(del_deg, nu_deg)
        out = []
        for i, f in enumerate(frames):
            param = [p[0] + d_dist[i], p[1] + d_poni1[i], p[2] + d_poni2[i],
                     rot1[i], rot2[i], rot3[i]]
            if runtime is None:
                q = geometry.qFunction(f.row_positions[keep[i]], f.col_positions[keep[i]], param=param)
            else:
                q = runtime(param).qFunction(f.row_positions[keep[i]], f.col_positions[keep[i]])
            out.append(np.asarray(q, dtype=float) / 10.0 - targets[i][keep[i]])
        return np.concatenate(out)

    def solve(p0, keep):
        def residual(p):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("goniometer refinement cancelled")
            return predicted(p, keep)
        if distance_bounds_m is None:
            return least_squares(residual, p0, method="lm", x_scale="jac", max_nfev=max_nfev)
        lower, upper = np.full(len(p0), -np.inf), np.full(len(p0), np.inf)
        lower[0], upper[0] = lo, hi
        return least_squares(residual, p0, method="trf", bounds=(lower, upper),
                             x_scale="jac", max_nfev=max_nfev)

    keep = [np.ones(len(f.rows), dtype=bool) for f in frames]
    p0 = np.array([distance, base_poni.poni1, base_poni.poni2, 0.0, 0.0])
    res = solve(p0, keep)
    if extended:
        res = solve(np.r_[res.x, np.zeros(len(_EXTENSION_NAMES))], keep)
    rejected = 0
    if reject_outliers is not None:
        for _ in range(2):
            r = predicted(res.x, [np.ones(len(f.rows), dtype=bool) for f in frames])
            cut = max(_MIN_OUTLIER_CUT_Q,
                      float(reject_outliers) * 1.4826 * float(np.median(np.abs(r - np.median(r)))))
            split = np.cumsum([len(f.rows) for f in frames])[:-1]
            keep = [np.abs(part) <= cut for part in np.split(r, split)]
            rejected = int(sum((~k).sum() for k in keep))
            res = solve(res.x, keep)
    p = res.x
    rms = float(np.sqrt(np.mean(res.fun ** 2)))
    n_points = int(len(res.fun))
    jac = np.asarray(res.jac)
    try:
        sv = np.linalg.svd(jac, compute_uv=False)
        condition = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        dof = max(1, n_points - len(p))
        covariance = np.linalg.pinv(jac.T @ jac) * float(res.fun @ res.fun) / dof
        errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
        scale = np.where(errors > 0, errors, 1.0)
        correlation = covariance / np.outer(scale, scale)
    except Exception:  # noqa: BLE001
        condition, errors = float("inf"), np.full(len(p), np.inf)
        correlation = np.full((len(p), len(p)), np.nan)
    names = _ARM_NAMES + (_EXTENSION_NAMES if extended else ())
    param_errors = dict(zip(names, (float(e) for e in errors)))
    param_correlations = {"names": list(names),
                          "matrix": [[float(v) for v in row] for row in correlation]}

    problems: list[str] = []
    if not res.success:
        problems.append(f"optimiser did not converge (status={res.status})")
    if not (0.0 < p[0] < 100.0):
        problems.append(f"non-physical dist={p[0]:.4g} m")
    if abs(p[3]) > np.pi or abs(p[4]) > np.pi:
        problems.append("non-physical rotation offset (> π rad)")
    if condition > _MAX_CONDITION:
        problems.append(f"under-determined fit (Jacobian condition {condition:.2g} > {_MAX_CONDITION:.0g})")
    if extended:
        if max(abs(p[5]), abs(p[6])) > _MAX_MOUNT_RAD:
            problems.append("detector mount tilt/roll exceeds 5 degrees")
        if np.max(np.abs(p[7:10])) > _MAX_SAMPLE_OFFSET_M:
            problems.append("sample offset exceeds 20 mm")
        if max(errors[5], errors[6]) > _MAX_MOUNT_ERROR_RAD or np.max(errors[7:10]) > _MAX_SAMPLE_OFFSET_ERROR_M:
            problems.append("arm extension is poorly determined; add frames spanning nu and del")
    ok = not problems
    message = "" if ok else "; ".join(problems)
    (logger.info if ok else logger.warning)(
        "refine_stacked_arm: RMS |q| = %.5g Å⁻¹ over %d points (%d frames, %d rejected), "
        "extended=%s, cond=%.3g, success=%s%s", rms, n_points, len(frames), rejected,
        extended, condition, ok, "" if ok else f" — {message}")

    arm = arm_for(p)
    calibration = DetectorCalibration(
        replace(base_poni, dist=float(p[0]), poni1=float(p[1]), poni2=float(p[2]),
                rot1=0., rot2=0., rot3=0., wavelength=wl_m, detector=det_name),
        detector_config=dict(detector_config or {}), parallax=base_parallax,
    )
    gonio = arm.to_pyfai_goniometer(calibration)
    diff = Diffractometer.from_pyfai_goniometer(
        gonio, source_motors={name: motor for name, motor in (("nu_angle", nu_motor), ("del_angle", del_motor))
                              if motor is not None},
        base=base, preset=preset)
    return RefineResult(
        diffractometer=diff, rms_q=rms, n_points=n_points,
        params=dict(zip(gonio["param_names"], gonio["param"])), success=ok,
        condition_number=condition, frozen_scales=(), message=message,
        param_errors=param_errors, rejected_points=rejected,
        param_correlations=param_correlations,
    )


__all__ = ["ControlFrame", "RefineResult", "refine_goniometer", "refine_stacked_arm"]
