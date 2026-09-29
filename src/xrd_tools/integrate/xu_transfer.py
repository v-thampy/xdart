"""Transfer a pyFAI extended stacked arm to the xrayutilities SURFACE model (asset v3).

xrayutilities' ``Ang2Q.area`` can carry a detector mount (``detrot``, ``tilt``,
``tiltazimuth``) and a sample displacement (``sampledis``), but those parameters do
not map one-to-one onto the arm's: the centre and the tilt trade off. They are
therefore fitted numerically to the arm's per-pixel q on frames spread over the
scan, seeded at four tilt azimuths, and the result is accepted only if every pixel
of every validation frame agrees within the gate (1e-8 Å⁻¹ by default).

XU's centre pixel is ``cch = poni / pitch − 0.5`` (measured: 1e-9 Å⁻¹ against 3e-3
for ``poni / pitch``). The sample displacement is in the detector distance's units
(mm here): the fit recovers the arm's offsets exactly as ``(−s1, s3, s2)``.
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


def transfer_extended_arm_to_xu(geometry, *, validation_angles, fit_angles=None,
                                source_motors=('del', 'nu'), image_rotation=0,
                                gate=None, cancelled=None, progress=None):
    """Fit the XU SURFACE extension to a pyFAI extended arm; return (asset, diagnostics).

    ``geometry`` is a pyFAI goniometer record (mapping or path) of an extended
    stacked arm with degree motor scales, no beam rotation and no parallax.
    ``validation_angles`` are (del, nu) source angles, typically every frame of the
    scan; ``fit_angles`` default to at most 20 of them spread in order.
    """
    from scipy.optimize import least_squares
    from pyFAI.detectors import detector_factory
    from xrd_tools.analysis.xu_stitch_calibration import (
        EXTENDED_TRANSFER_GATE_A, extended_surface_calibration,
    )
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry

    gate = EXTENDED_TRANSFER_GATE_A if gate is None else float(gate)
    del_motor, nu_motor = source_motors
    record = (json.loads(json.dumps(geometry)) if isinstance(geometry, Mapping)
              else json.loads(Path(geometry).read_text()))
    diff = Diffractometer.from_pyfai_goniometer(
        record, source_motors={'del_angle': del_motor, 'nu_angle': nu_motor})
    arm, calibration = diff.stacked_arm, diff.calibration
    if arm is None or calibration is None or not arm.extended:
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', 'the transfer needs an extended stacked (del-on-nu) arm')
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
    validation = [(float(d), float(n)) for d, n in validation_angles]
    fit = _fit_subset(validation, _MAX_FIT_FRAMES) if fit_angles is None else [
        (float(d), float(n)) for d, n in fit_angles]
    if len(fit) < 3 or len(validation) < 3:
        raise XuTransferRefused('XU_TRANSFER_UNSUPPORTED', 'the transfer needs at least three frames')
    shape = tuple(int(v) for v in detector.shape)
    pitch1, pitch2 = float(detector.pixel1), float(detector.pixel2)
    poni = calibration.poni
    energy = float(wavelength_m_to_energy_eV(poni.wavelength))

    def pyfai_q(angles):
        motors = {arm.del_motor: np.array([a[0] for a in angles]), arm.nu_motor: np.array([a[1] for a in angles])}
        rows, cols = np.indices(shape, dtype=float)
        for integrator in create_multigeometry_integrators_from_geometry(diff, motors):
            yield integrator.qFunction(rows, cols) / 10.0

    def check():
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError('XU transfer cancelled')

    rows_sub = slice(0, shape[0], _SUBGRID[0])
    cols_sub = slice(0, shape[1], _SUBGRID[1])
    targets = [q[rows_sub, cols_sub] for q in pyfai_q(fit)]
    with xu_runtime_session() as runtime:
        xu = runtime.xu

        def area(params):
            c1, c2, dist, _d0, _n0, detrot, tilt, azimuth = params[:8]
            qconv = xu.QConversion(['x+', 'z-', 'y+', 'z-'], ['x+', 'z-'], [0, 1, 0])
            hxrd = xu.HXRD([0, 1, 0], [0, 0, 1], geometry='real', en=energy, qconv=qconv)
            hxrd.Ang2Q.init_area('x-', 'z+', cch1=c1, cch2=c2, Nch1=shape[0], Nch2=shape[1],
                                 pwidth1=pitch1 * 1e3, pwidth2=pitch2 * 1e3, distance=dist,
                                 detrot=detrot, tilt=tilt, tiltazimuth=azimuth)
            return hxrd

        def q_xu(hxrd, params, angle):
            displaced = {} if not np.any(params[8:11]) else {'sampledis': list(params[8:11])}
            qx, qy, qz = hxrd.Ang2Q.area(0., 0., 0., 0., angle[1] - params[4], angle[0] - params[3], **displaced)
            return np.hypot(np.hypot(qx, qy), qz)

        def residual(params):
            check()
            hxrd = area(params)
            return np.concatenate([(q_xu(hxrd, params, angle)[rows_sub, cols_sub] - target).ravel()
                                   for angle, target in zip(fit, targets)])

        seed = [poni.poni1 / pitch1 - 0.5, poni.poni2 / pitch2 - 0.5, poni.dist * 1e3,
                -math.degrees(arm.del_offset), -math.degrees(arm.nu_offset), math.degrees(arm.mount_roll)]
        seeds, best = [], None
        for done, azimuth in enumerate(_SEED_AZIMUTHS):
            result = least_squares(residual, seed + [abs(math.degrees(arm.mount_tilt)), azimuth, 0., 0., 0.],
                                   method='lm', x_scale='jac', max_nfev=4000)
            seeds.append({'tiltazimuth_seed_deg': azimuth, 'max_abs_dq_A^-1': float(np.max(np.abs(result.fun)))})
            if best is None or result.cost < best.cost:
                best = result
            if progress is not None:
                progress('Transferring to xrayutilities', done + 1, len(_SEED_AZIMUTHS))
        params = [float(v) for v in best.x]
        if params[6] < 0:  # a negative tilt is the same tilt about the opposite azimuth
            params[6], params[7] = -params[6], params[7] + 180.0
        params[7] %= 360.0
        hxrd = area(params)  # the gate checks exactly what the asset stores
        worst = 0.0
        for done, (angle, target) in enumerate(zip(validation, pyfai_q(validation))):
            check()
            worst = max(worst, float(np.max(np.abs(q_xu(hxrd, params, angle) - target))))
            if progress is not None and (done + 1) % 16 == 0:
                progress('Validating the transfer', done + 1, len(validation))
        xu_wavelength = float(xu.en2lam(energy))
    if not worst <= gate:
        raise XuTransferRefused('XU_TRANSFER_GATE_FAILED',
                                f'the XU transfer reaches {worst:.3g} Å⁻¹ on a validation pixel (gate {gate:.0e})')
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
    asset = extended_surface_calibration(
        detector=detector,
        xu_parameters={'cch1_px': params[0], 'cch2_px': params[1], 'distance_m': params[2] / 1e3,
                       'del_offset_deg': params[3], 'nu_offset_deg': params[4], 'detrot_deg': params[5],
                       'tilt_deg': params[6], 'tiltazimuth_deg': params[7],
                       'sampledis_mm': list(params[8:11])},
        arm=compact, energy_eV=energy, xu_wavelength_A=xu_wavelength,
        source_motors={'del': del_motor, 'nu': nu_motor}, rotation=int(image_rotation),
        control_domain=domain(fit), domain=domain(validation), fit_frames=len(fit),
        validation_frames=len(validation), validation_pixels=len(validation) * shape[0] * shape[1],
        max_dq=worst)
    diagnostics = {'max_abs_dq_A^-1': worst, 'gate_A^-1': gate, 'fit_frames': len(fit),
                   'validation_frames': len(validation), 'seeds': seeds}
    return asset, diagnostics


__all__ = ['XuTransferRefused', 'transfer_extended_arm_to_xu']
