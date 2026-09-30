"""Transfer a pyFAI stacked detector arm to the xrayutilities SURFACE model (asset v3).

A five-parameter arm, or a single 2θ arm (one circle held at zero), follows by
formula: XU's centre pixel is ``cch = poni / pitch − 0.5`` (measured: 1e-9 Å⁻¹
against 3e-3 for ``poni / pitch``), the distance is the same and the motor offsets
are negated. An extended arm's detector mount (``detrot``, ``tilt``,
``tiltazimuth``), centre and sample displacement (``sampledis``) are fitted to
the arm's per-pixel q on frames spread over the scan, on a coarse virtual
detector whose pixels are exactly the compared ones (every 6th row, 23rd
column), from one tilt azimuth; the other three are tried only when that one
fails the gate. The motor offsets are held at the arm's: XU's del offset trades
exactly against its tilt and centre, and the fit would otherwise stop anywhere
along that valley. Held, the mount comes out as the arm's own roll and tilt.

Every result passes the same gates before it is returned: on every pixel of every
validation frame, through Stitch's own XU path (the admitted asset, its runtime
geometry and q provider), |q| within 1e-8 Å⁻¹ and the polarization factor within
1e-6 (relative, f 0.99) of pyFAI. The sample displacement is in the detector
distance's units (mm here): the fit recovers the arm's offsets exactly as
``(−s1, s3, s2)``.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path

import numpy as np

_SUBGRID = (6, 23)
_SEED_AZIMUTHS = (0.0, 90.0, 180.0, 270.0)
_MAX_FIT_FRAMES = 20
#: pyFAI's polarization factor and the XU path's agree to this, relative (measured 6e-8).
POLARIZATION_GATE = 1e-6
_POLARIZATION_FACTOR = 0.99


class XuTransferRefused(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _fit_subset(angles, count):
    """Evenly spread frames (by order), always including the first and last."""
    if len(angles) <= count:
        return list(angles)
    picks = np.unique(np.round(np.linspace(0, len(angles) - 1, count)).astype(int))
    return [angles[i] for i in picks]


def _stitch_path_gate(asset, validation, integrators, shape, cancelled, progress):
    """The worst |Δq| (Å⁻¹) and relative polarization difference against pyFAI on every
    pixel of every validation frame, computed as a Stitch run computes them."""
    import tempfile
    from xrd_tools.analysis.xu_stitch_calibration import (
        XuStitchCalibrationInput, _canonical_bytes, capture_xu_stitch_calibration,
    )
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.xu_stitch import (
        XuPowderQProvider, _polarization_axes, resolve_xu_stitch_effective_geometry,
    )
    rows, cols = np.indices(shape, dtype=float)
    worst_q = worst_p = 0.0
    with tempfile.TemporaryDirectory(prefix='xdart-xu-transfer-') as scratch:
        staged = Path(scratch).resolve() / 'calibration.json'
        staged.write_bytes(_canonical_bytes(asset))
        receipt = capture_xu_stitch_calibration(XuStitchCalibrationInput(staged), project_root=staged.parent)
        with xu_runtime_session() as runtime:
            geometry = resolve_xu_stitch_effective_geometry(receipt, runtime)
            provider = XuPowderQProvider(geometry, asset['acquisition'])
            polarization = (_POLARIZATION_FACTOR, geometry.projection.xu_mapping_wavelength_A,
                            *_polarization_axes(asset))
            for done, ((delta, nu), integrator) in enumerate(zip(validation, integrators)):
                if cancelled is not None and cancelled.is_set():
                    raise InterruptedError('XU transfer cancelled')
                reference = integrator.polarization(factor=_POLARIZATION_FACTOR)
                reference = np.asarray(reference[0] if isinstance(reference, tuple) else reference, dtype=float)
                with provider.frame(nu=nu, del_=delta) as lease:
                    worst_q = max(worst_q, float(np.max(np.abs(
                        lease.q_magnitude() - integrator.qFunction(rows, cols) / 10.0))))
                    worst_p = max(worst_p, float(np.max(np.abs(
                        lease.polarization(*polarization) - reference) / reference)))
                if progress is not None and (done + 1) % 16 == 0:
                    progress('Validating the transfer', done + 1, len(validation))
    return worst_q, worst_p


def transfer_arm_to_xu(geometry, *, validation_angles, fit_angles=None, source_motors=('del', 'nu'),
                       image_rotation=0, gate=None, cancelled=None, progress=None):
    """Transfer a pyFAI stacked arm to an XU SURFACE v3 asset; return (asset, diagnostics).

    ``geometry`` is a pyFAI goniometer record (mapping or path) of a stacked arm: the
    five-parameter arm, the extended arm, or a single 2θ arm, with degree motor
    scales, no beam rotation, no parallax and pyFAI's default detector orientation.
    ``source_motors`` is (del motor, nu motor), None for the circle a single arm
    holds at zero. ``validation_angles`` are (del, nu) circle angles (a held circle
    at 0), normally every frame the asset will serve; ``fit_angles`` (an extended
    arm's fit) default to at most 20 of them spread in order.
    """
    from scipy.optimize import least_squares
    from pyFAI.detectors import detector_factory
    from xrd_tools.analysis.xu_stitch_calibration import (
        EXTENDED_TRANSFER_GATE_A, extended_surface_calibration,
    )
    from xrd_tools.core.energy import calibration_recorded_energy_eV, wavelength_m_to_energy_eV
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
    from xrd_tools.integrate.xu_stitch import XuStitchScienceRefused

    gate = EXTENDED_TRANSFER_GATE_A if gate is None else float(gate)
    del_motor, nu_motor = source_motors
    record = (json.loads(json.dumps(geometry)) if isinstance(geometry, Mapping)
              else json.loads(Path(geometry).read_text()))
    try:
        diff = Diffractometer.from_pyfai_goniometer(record, source_motors={
            position: motor for position, motor in (('del_angle', del_motor), ('nu_angle', nu_motor))
            if motor is not None})
    except (KeyError, TypeError, ValueError) as error:
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', f'the geometry cannot be read as a detector arm: {error}') from error
    arm, calibration = diff.stacked_arm, diff.calibration
    if arm is None or calibration is None:
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', 'the transfer needs a stacked (del-on-nu) detector arm')
    if (arm.del_motor is None) != (del_motor is None) or (arm.nu_motor is None) != (nu_motor is None):
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', 'the arm and the source motors hold different circles')
    if int(dict(calibration.detector_config).get('orientation', 3)) != 3:  # as the XU fit refuses it
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', "the XU model takes pyFAI's default detector "
                                'orientation (3): fit the arm in orientation 3 to transfer it')
    if arm.nu_scale != math.pi / 180 or arm.del_scale != math.pi / 180:
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', 'XU takes angles in degrees: fitted motor scales cannot be transferred')
    if arm.beam_rotation != 0:
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', "XU's azimuth reference is fixed: a beam rotation cannot be transferred")
    if calibration.parallax is not None:
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', 'XU has no parallax model')
    detector = detector_factory(calibration.poni.detector, config=dict(calibration.detector_config))
    if detector.sensor is not None or not detector.IS_FLAT or not detector.uniform_pixel:
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', 'XU needs a flat uniform detector without a sensor model')
    # every frame once, in order; a held circle is at zero whatever the caller passed
    validation = list(dict.fromkeys((0.0 if del_motor is None else float(d), 0.0 if nu_motor is None else float(n))
                                    for d, n in validation_angles))
    fit = _fit_subset(validation, _MAX_FIT_FRAMES) if fit_angles is None else [
        (float(d), float(n)) for d, n in fit_angles]
    if len(validation) < 3 or (arm.extended and len(fit) < 3):
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', 'the transfer needs at least three frames')
    shape = tuple(int(v) for v in detector.shape)
    pitch1, pitch2 = float(detector.pixel1), float(detector.pixel2)
    poni = calibration.poni
    energy = float(wavelength_m_to_energy_eV(poni.wavelength))

    def integrators(angles):
        return create_multigeometry_integrators_from_geometry(diff, {
            motor: np.array([angle[column] for angle in angles])
            for column, motor in enumerate((arm.del_motor, arm.nu_motor)) if motor is not None})

    def check():
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError('XU transfer cancelled')

    rows_sub, cols_sub = np.arange(0, shape[0], _SUBGRID[0]), np.arange(0, shape[1], _SUBGRID[1])
    formula = [poni.poni1 / pitch1 - 0.5, poni.poni2 / pitch2 - 0.5, poni.dist * 1e3,
               -math.degrees(arm.del_offset), -math.degrees(arm.nu_offset)]
    attempts, params = [], None
    with xu_runtime_session() as runtime:
        xu = runtime.xu

        def area(params, coarse=False):
            c1, c2, dist, _d0, _n0, detrot, tilt, azimuth = params[:8]
            qconv = xu.QConversion(['x+', 'z-', 'y+', 'z-'], ['x+', 'z-'], [0, 1, 0])
            hxrd = xu.HXRD([0, 1, 0], [0, 0, 1], geometry='real', en=energy, qconv=qconv)
            # the coarse detector's pixel i is the full detector's pixel i·step, exactly
            step1, step2 = _SUBGRID if coarse else (1, 1)
            hxrd.Ang2Q.init_area('x-', 'z+', cch1=c1 / step1, cch2=c2 / step2,
                                 Nch1=len(rows_sub) if coarse else shape[0],
                                 Nch2=len(cols_sub) if coarse else shape[1],
                                 pwidth1=pitch1 * 1e3 * step1, pwidth2=pitch2 * 1e3 * step2, distance=dist,
                                 detrot=detrot, tilt=tilt, tiltazimuth=azimuth)
            return hxrd

        def q_xu(hxrd, params, angle):
            displaced = {} if not np.any(params[8:11]) else {'sampledis': list(params[8:11])}
            qx, qy, qz = hxrd.Ang2Q.area(0., 0., 0., 0., angle[1] - params[4], angle[0] - params[3], **displaced)
            return np.hypot(np.hypot(qx, qy), qz)

        def worst(params):
            """The candidate's |Δq| on every pixel of every validation frame."""
            hxrd, value = area(params), 0.0
            rows, cols = np.indices(shape, dtype=float)
            for angle, integrator in zip(validation, integrators(validation)):
                check()
                value = max(value, float(np.max(np.abs(q_xu(hxrd, params, angle)
                                                       - integrator.qFunction(rows, cols) / 10.0))))
            return value

        if not arm.extended:
            candidate = formula + [0.] * 6
            attempts.append({'method': 'formula', 'max_abs_dq_A^-1': worst(candidate)})
            if attempts[-1]['max_abs_dq_A^-1'] <= gate:
                params = candidate
        if params is None:
            coarse_rows, coarse_cols = np.meshgrid(rows_sub.astype(float), cols_sub.astype(float), indexing='ij')
            targets = [integrator.qFunction(coarse_rows, coarse_cols) / 10.0 for integrator in integrators(fit)]

            # The motor offsets stay the arm's: XU's del offset trades exactly against its tilt and
            # centre along a valley whose far members leave the v3 bounds (all seeds reach 1e-10).
            def full(free):
                return [*free[:3], *formula[3:5], *free[3:]]

            def residual(free):
                check()
                candidate = full(free)
                hxrd = area(candidate, coarse=True)
                return np.concatenate([(q_xu(hxrd, candidate, angle) - target).ravel()
                                       for angle, target in zip(fit, targets)])

            seed = formula[:3] + [math.degrees(arm.mount_roll), abs(math.degrees(arm.mount_tilt))]
            for done, azimuth in enumerate(_SEED_AZIMUTHS):
                result = least_squares(residual, seed + [azimuth, 0., 0., 0.], method='lm', x_scale='jac',
                                       max_nfev=4000)
                candidate = [float(v) for v in full(result.x)]
                if candidate[6] < 0:  # a negative tilt is the same tilt about the opposite azimuth
                    candidate[6], candidate[7] = -candidate[6], candidate[7] + 180.0
                candidate[7] %= 360.0
                attempts.append({'method': 'coarse_fit', 'tiltazimuth_seed_deg': azimuth, 'evaluations': result.nfev,
                                 'max_abs_dq_A^-1': worst(candidate)})
                if progress is not None:
                    progress('Transferring to xrayutilities', done + 1, len(_SEED_AZIMUTHS))
                if attempts[-1]['max_abs_dq_A^-1'] <= gate:
                    params = candidate
                    break
        xu_wavelength = float(xu.en2lam(energy))
    if params is None:
        best = min(attempt['max_abs_dq_A^-1'] for attempt in attempts)
        raise XuTransferRefused('XU_TRANSFER_GATE_FAILED',
                                f'the XU transfer reaches {best:.3g} Å⁻¹ on a validation pixel (gate {gate:.0e})')
    compact = {
        'marker': 'xdart_stacked_arm_v2', 'detector': type(detector).__name__,
        'detector_config': dict(calibration.detector_config), 'wavelength_m': float(poni.wavelength),
        'dist_m': float(poni.dist), 'poni1_m': float(poni.poni1), 'poni2_m': float(poni.poni2),
        'nu_offset': float(arm.nu_offset), 'nu_scale': float(arm.nu_scale),
        'del_offset': float(arm.del_offset), 'del_scale': float(arm.del_scale),
        'beam_rotation': float(arm.beam_rotation), 'mount_tilt': float(arm.mount_tilt),
        'mount_roll': float(arm.mount_roll), 'sample_offset1': float(arm.sample_offset1),
        'sample_offset2': float(arm.sample_offset2), 'sample_offset3': float(arm.sample_offset3),
        'source_motors': {'del_angle': del_motor, 'nu_angle': nu_motor}}

    def domain(angles):
        return {'del': [min(a[0] for a in angles), max(a[0] for a in angles)],
                'nu': [min(a[1] for a in angles), max(a[1] for a in angles)]}
    fitted = attempts[-1]['method'] == 'coarse_fit'
    worst_dq = attempts[-1]['max_abs_dq_A^-1']
    try:  # the v3 admission and Stitch's own XU path refuse in their own terms: said as the transfer's
        asset = extended_surface_calibration(
            detector=detector,
            xu_parameters={'cch1_px': params[0], 'cch2_px': params[1], 'distance_m': params[2] / 1e3,
                           'del_offset_deg': params[3], 'nu_offset_deg': params[4], 'detrot_deg': params[5],
                           'tilt_deg': params[6], 'tiltazimuth_deg': params[7],
                           'sampledis_mm': list(params[8:11])},
            arm=compact, energy_eV=energy, xu_wavelength_A=xu_wavelength,
            source_energy_eV=calibration_recorded_energy_eV(record),
            source_motors={'del': del_motor, 'nu': nu_motor}, rotation=int(image_rotation),
            control_domain=domain(fit if fitted else validation), domain=domain(validation),
            fit_frames=len(fit) if fitted else 0, validation_frames=len(validation),
            validation_pixels=len(validation) * shape[0] * shape[1], max_dq=worst_dq)
        stitch_dq, polarization = _stitch_path_gate(asset, validation, integrators(validation), shape,
                                                    cancelled, progress)
    except (ValueError, XuStitchScienceRefused) as error:
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED',
                                f'xrayutilities refuses the transferred calibration: {error}') from error
    if not stitch_dq <= gate:
        raise XuTransferRefused('XU_TRANSFER_GATE_FAILED', f"Stitch's XU path reaches {stitch_dq:.3g} Å⁻¹ on a "
                                f'validation pixel (gate {gate:.0e})')
    if not polarization <= POLARIZATION_GATE:
        raise XuTransferRefused('XU_TRANSFER_GATE_FAILED', f"Stitch's XU polarization factor differs from pyFAI's "
                                f'by {polarization:.3g} (relative) on a validation pixel (gate {POLARIZATION_GATE:.0e})')
    diagnostics = {'method': attempts[-1]['method'], 'max_abs_dq_A^-1': worst_dq, 'gate_A^-1': gate,
                   'stitch_path_max_abs_dq_A^-1': stitch_dq, 'polarization_max_relative': polarization,
                   'polarization_gate': POLARIZATION_GATE, 'polarization_factor': _POLARIZATION_FACTOR,
                   'fit_frames': len(fit) if fitted else 0, 'validation_frames': len(validation),
                   'attempts': attempts}
    return asset, diagnostics


__all__ = ['POLARIZATION_GATE', 'XuTransferRefused', 'transfer_arm_to_xu']
