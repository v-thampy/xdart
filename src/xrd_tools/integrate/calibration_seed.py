"""A starting geometry and the expected calibrant rings for one guided calibration frame.

The seed is a goniometer calibration (this session's fit, a calibration file, or a
rough geometry) evaluated at the frame's own del/nu, with the frame's own energy.
A single detector arm (fourc/twoc) holds one circle at zero: ``nu_motor=None`` for a
vertical 2θ arm, ``del_motor=None`` for a horizontal one.
pyFAI-calib2 opens with it, so its ring extraction ("Fit screen") places and numbers
the rings; the same prediction lists which calibrant rings reach the detector.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np


def calibration_file_identity(path):
    """Path and bytes of a committed primary geometry and its optional provenance."""
    path = Path(path).resolve()
    provenance = path.with_suffix(path.suffix + '.provenance.json')
    return (str(path), sha256(path.read_bytes()).hexdigest(),
            sha256(provenance.read_bytes() if provenance.exists() else b'').hexdigest())


def calibration_seed_inputs(path, *, fallback_instrument):
    """Decode supported committed arm inputs, retaining the full file as the refinement seed.

    The beam and distance are evaluated at the arm's motor zeros by the same
    physical intersection used to construct rough geometries. No fit is recreated.
    """
    from xrd_tools.core.geometry import Diffractometer, ImageOrientation
    from xrd_tools.core.geometry.camera import stitch_camera
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.session.instrument import InstrumentGeometry
    path = Path(path).resolve()
    identity = calibration_file_identity(path)
    record = json.loads(path.read_text(encoding='utf-8'))
    sidecar = path.with_suffix(path.suffix + '.provenance.json')
    provenance = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
    geometry = (InstrumentGeometry.from_record(provenance['instrument_geometry'])
                if provenance.get('instrument_geometry') else fallback_instrument)
    del_motor, nu_motor = geometry.stitch_arm()
    mapping = provenance.get('source_motors') or {'del_angle': del_motor, 'nu_angle': nu_motor}
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors=mapping)
    arm, calibration = diff.stacked_arm, diff.calibration
    if arm is None:
        raise ValueError('this older independent-angle geometry has no supported detector arm')
    energy_eV = float(wavelength_m_to_energy_eV(calibration.poni.wavelength))
    zeros = arm.motor_zeros()
    motors, instrument_zeros = dict(geometry.motors), dict(geometry.zeros)
    for circle in geometry.circles:
        role = geometry.as_psic().get(circle, circle)
        if role in ('del', 'nu'):
            motor = getattr(arm, role + '_motor')
            if motor is not None:
                motors[circle], instrument_zeros[circle] = motor, zeros[role]
    geometry = replace(geometry, motors=tuple(motors.items()), zeros=tuple(instrument_zeros.items()))
    turned = calibration.turned_from or {}
    orientation = ImageOrientation(rotation=turned.get('image_rotation', provenance.get('image_rotation', 0)),
                                   flip_horizontal=turned.get('image_flip', provenance.get('image_flip', False)))
    readings = {getattr(arm, role + '_motor'): zero for role, zero in zeros.items()}
    (row, column), distance = _direct_beam_hit(record, del_motor=arm.del_motor,
                                              nu_motor=arm.nu_motor, readings=readings)
    if not all(math.isfinite(v) for v in (row, column, distance)) or distance <= 0:
        raise ValueError('the calibration has no finite, positive direct-beam distance')
    if identity != calibration_file_identity(path):
        raise ValueError('the calibration changed while loading; select it again')
    return dict(identity=identity, instrument=geometry,
                camera=stitch_camera(orientation), detector=turned.get('detector', calibration.poni.detector),
                orientation=int(calibration.detector_config.get('orientation', 3)),
                distance_m=distance, beam_px=(column, row), energy_eV=energy_eV,
                mount_deg=tuple(map(math.degrees, (arm.mount_tilt, arm.mount_roll))))


@dataclass(frozen=True)
class ExpectedRing:
    number: int  # pyFAI's ring number, 1-based as its peak picking shows it
    two_theta_deg: float
    d_spacing_A: float


@dataclass(frozen=True)
class FrameSeed:
    integrator: object  # the frame's predicted pyFAI geometry
    rings: tuple[ExpectedRing, ...]


#: A single arm's held circle, for a calibration that has a position for it: read as zero.
_HELD = {'del': 'del (held at zero)', 'nu': 'nu (held at zero)'}


def _source_motors(record, del_motor, nu_motor):
    """The calibration's angle names (del_angle / del_value ...) bound to the scan's motors;
    a circle the frame's arm holds (its motor None) reads zero, and a single-arm
    calibration declares only its moving circle."""
    trans = record.get('trans_function') or {}
    names = list(trans.get('pos_names') or record.get('pos_names') or ())
    motors = {circle: _HELD[circle] if motor is None else motor
              for circle, motor in (('del', del_motor), ('nu', nu_motor))}
    mapping = {name: motors['del'] for name in names if 'del' in name.casefold()}
    mapping.update({name: motors['nu'] for name in names if 'nu' in name.casefold() and name not in mapping})
    if (len(mapping) != len(names) or len(set(mapping.values())) != len(names)
            or all(motor in _HELD.values() for motor in mapping.values())):
        raise ValueError("the calibration's angles are not one del and one nu")
    return mapping


def expected_rings(integrator, calibrant, shape, mask=None) -> tuple[ExpectedRing, ...]:
    """The calibrant rings whose 2θ falls on an unmasked pixel of this geometry: the detector's
    mask and ``mask`` (True: left out, in the frame's image), when it has the frame's shape."""
    from pyFAI.calibrant import get_calibrant
    reference = get_calibrant(calibrant) if isinstance(calibrant, str) else calibrant
    reference.wavelength = float(integrator.wavelength)
    two_theta = np.asarray(integrator.center_array(tuple(shape), unit='2th_rad', scale=False), dtype=float)
    valid = np.isfinite(two_theta)
    for excluded in (getattr(integrator.detector, 'mask', None), mask):
        if excluded is not None and np.shape(excluded) == two_theta.shape:
            valid &= ~np.asarray(excluded, dtype=bool)
    if not valid.any():
        return ()
    low, high = float(two_theta[valid].min()), float(two_theta[valid].max())
    rings = []
    for number, (angle, d) in enumerate(zip(reference.get_2th(), reference.dspacing), 1):
        if angle is not None and low <= angle <= high:
            rings.append(ExpectedRing(number, math.degrees(angle), float(d)))
    return tuple(rings)


def predict_frame_seed(record, frame, *, calibrant='LaB6', del_motor='del', nu_motor='nu', mask=None,
                       conditioning=None, saturation=None) -> FrameSeed:
    """The frame's geometry predicted from a goniometer calibration, with its rings (none on
    ``mask``, the calibration's mask in the frame's image)."""
    from pyFAI import units
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.integrate.multi import create_multigeometry_integrator_series_from_geometry
    if not isinstance(record, dict):
        record = json.loads(Path(record).read_text(encoding='utf-8'))
    bound = _source_motors(record, del_motor, nu_motor)
    for circle, motor in (('del', del_motor), ('nu', nu_motor)):
        if motor is not None and motor not in bound.values() and float(frame.motors[motor]) != 0:
            raise ValueError(f'a single-arm calibration holds {circle} at zero; this frame has {circle} '
                             f'{float(frame.motors[motor]):g}')
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors=bound)
    motors = {motor: np.array([0.0 if motor in _HELD.values() else float(frame.motors[motor])])
              for motor in bound.values()}
    integrator = next(iter(create_multigeometry_integrator_series_from_geometry(diff, motors)))
    if tuple(integrator.detector.shape) != tuple(frame.shape):
        raise ValueError('the calibration detector does not match the staged frame')
    # the frame's own energy, as the scan recorded it; else an energy-refined calibration's recorded
    # one, so every frame of a session is calibrated at the same (recorded) energy
    from xrd_tools.core.energy import calibration_recorded_energy_eV
    energy_eV = frame.energy_eV or calibration_recorded_energy_eV(record)
    if energy_eV:
        integrator.wavelength = units.hc / (float(energy_eV) / 1000.) * 1e-10
    from xrd_tools.integrate.stitch_calibration import calibration_frame_mask
    mask = calibration_frame_mask(frame, mask, conditioning=conditioning, saturation=saturation)
    return FrameSeed(integrator, expected_rings(integrator, calibrant, frame.shape, mask))


def session_prediction(frames, *, del_motor='del', nu_motor='nu', cancelled=None, bounds=None,
                       distance_seed_m=None, distance_bounds_m=None, instrument_geometry=None, turned_from=None):
    """The arm fitted to this session's calibrated frames, to predict the others' rings and
    seed pyFAI with them: one calibrated frame is enough to place the rest (a saved
    calibration still needs three). None without a calibrated frame, or when they do not fit.
    ``bounds``, ``distance_seed_m`` and ``distance_bounds_m``: the estimates and freedom the saved fit
    takes (fit_calibration's); ``instrument_geometry`` its circles' senses; ``turned_from`` the model the frames'
    camera turns by a quarter (fit_calibration's).
    """
    from xrd_tools.integrate.stitch_calibration import fit_calibration
    done = tuple(frame for frame in frames if frame.complete)
    if not done:
        return None
    try:
        return fit_calibration(done, backend='multigeometry', del_motor=del_motor, nu_motor=nu_motor,
                               cancelled=cancelled, min_frames=1, bounds=bounds, distance_seed_m=distance_seed_m,
                               distance_bounds_m=distance_bounds_m, instrument_geometry=instrument_geometry,
                               turned_from=turned_from).asset['calibration']
    except (ValueError, RuntimeError, KeyError):
        return None


#: Listed first, in this order; then every other pyFAI detector by name.
_PREFERRED_DETECTORS = ('Pilatus100k', 'Pilatus300kw', 'Eiger1M', 'Eiger2_500k', 'Eiger4M',
                        'RayonixMx225', 'RayonixMx300')
_DETECTOR_LABELS = {'RayonixMx225': 'MarCCD 225 / Rayonix MX225', 'RayonixMx300': 'Rayonix MX300'}

#: pyFAI's detector orientations it supports (EXIF codes); 3 is its default.
DETECTOR_ORIENTATIONS = ((3, 'BottomRight (pyFAI default)'), (1, 'TopLeft'), (2, 'TopRight'),
                         (4, 'BottomLeft'))


@lru_cache(maxsize=1)
def detector_choices(*, raw_details=False) -> tuple[tuple[str, str], ...]:
    """pyFAI's detectors as (label, factory name): the common ones first, then the rest by label.
    Only detectors built from their name alone are listed (not the generic pixel-size one)."""
    from pyFAI.detectors import ALL_DETECTORS, detector_factory
    labels = {}
    for cls in set(ALL_DETECTORS.values()):
        name = cls.__name__
        try:
            detector = detector_factory(name)
        except Exception:
            continue
        if detector.shape is None or not detector.pixel1 or not detector.pixel2:
            continue
        label = _DETECTOR_LABELS.get(name, (getattr(cls, 'aliases', None) or [name])[0])
        labels[name] = (f'{label} [{detector.shape[0]} × {detector.shape[1]} rows × cols; RAW int32]'
                        if raw_details else label)  # int32 is the RAW reader's default, not a detector property
    first = [(labels[name], name) for name in _PREFERRED_DETECTORS if name in labels]
    rest = sorted(((label, name) for name, label in labels.items() if name not in _PREFERRED_DETECTORS),
                  key=lambda item: item[0].casefold())
    return tuple(first + rest)


def detector_shape(detector) -> tuple[int, int] | None:
    """The detector's (rows, columns), or None for a name pyFAI does not know."""
    from pyFAI.detectors import detector_factory
    try:
        rows, columns = detector_factory(detector).shape
        return int(rows), int(columns)
    except Exception:
        return None


def rough_arm_record(*, distance_m, detector, beam_px, zeros_deg=None, del_motor='del', nu_motor='nu',
                     wavelength_m=1e-10, orientation=3, mount_deg=(0., 0.), beam_readings=None,
                     senses=None, camera=None) -> dict:
    """A nominal stacked (del-on-nu) arm from a rough distance, for a new setup.

    ``zeros_deg`` gives each moving circle's motor-zero reading ('del', 'nu'; physical angle =
    motor readback − zero; a circle not given reads 0 at its zero). With both moving circles at
    their zero readings the detector faces the beam, and the direct beam hits ``beam_px`` (x, y
    in the oriented image) with the detector ``distance_m`` from the sample; the angles move the
    detector as a del-on-nu arm, one degree per degree. Good enough to predict which rings a frame
    shows; the fit refines it.
    ``nu_motor=None`` or ``del_motor=None`` is a single arm (a fourc/twoc 2θ): that
    circle is held at zero. ``orientation`` is pyFAI's detector orientation: the PONI is
    the beam pixel's centre as that oriented detector places it (1, 2 and 4 reflect the
    row, the column or both), and a non-default one travels in the detector
    config (the seed PONI pyFAI opens with, and so its saved PONIs). The motor scales
    take its signs (``orientation_signs``), so every orientation predicts the same rings, and
    each circle's sense (``senses``: {'nu': -1} for a nu turning opposite to psic's; ``arm_signs``).
    ``mount_deg`` holds the detector mount's tilt and roll fixed (the arm's extension; a single
    arm has none), zero for a detector facing the beam squarely at every angle. With a tilted
    mount the beam still hits ``beam_px`` at the zero pose, and ``distance_m`` is the distance
    along the direct beam there (pyFAI's direct distance), not the PONI's.
    ``beam_readings`` ({motor: reading}) is the pose the beam was seen at ``beam_px`` when it is
    not the zero pose: a direct-beam scan turns one circle and holds the other where it read (17-2's
    scan 3 holds del at 0, its zero 2.3°), so the PONI moves to put the beam on ``beam_px`` there
    (maintainer, 2026-09-29: a refinement started from that geometry "looks totally off").
    ``camera`` is the turn and mirror the window's images take (``ImageOrientation``; None: none): for a quarter
    turn the detector is the model turned with them (``turned_detector.window_detector``: pyFAI's generic detector of
    the transposed shape, the model and turn recorded beside it), so ``beam_px`` and the record are in that frame.
    """
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.core.geometry.diffractometer import ImageOrientation
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm, arm_signs
    from xrd_tools.integrate.turned_detector import window_detector
    if not math.isfinite(float(distance_m)) or float(distance_m) <= 0:
        raise ValueError('the rough distance must be a positive number')
    pixels, name, config, turned = window_detector(detector, camera or ImageOrientation(),
                                                   pyfai_orientation=int(orientation))
    x, y = (float(value) for value in beam_px)
    moving = {circle: motor for circle, motor in (('del', del_motor), ('nu', nu_motor)) if motor is not None}
    held = set(zeros_deg or {}) - set(moving)
    if held:
        raise ValueError(f'{sorted(held)[0]} is held at zero by this arm: it has no motor zero')
    zeros = {circle: float((zeros_deg or {}).get(circle, 0.)) for circle in moving}
    if not all(math.isfinite(value) for value in zeros.values()):
        raise ValueError('the motor zeros must be finite numbers of degrees')
    nu_sign, del_sign = arm_signs(pixels.orientation, senses)
    scales = dict(nu_scale=nu_sign * math.pi / 180, del_scale=del_sign * math.pi / 180)
    offset = {f'{circle}_offset': -scales[f'{circle}_scale'] * zero for circle, zero in zeros.items()}
    # the configured detector's own conversion; for orientation 3 it is (y + .5, x + .5) pitches exactly
    poni1, poni2, _ = pixels.calc_cartesian_positions(d1=np.array([y]), d2=np.array([x]))
    poni = PONI(dist=float(distance_m), poni1=float(poni1[0]), poni2=float(poni2[0]),
                wavelength=float(wavelength_m), detector=name)
    tilt, roll = (math.radians(float(value)) for value in mount_deg)
    arm = StackedDetectorArm(nu_motor=nu_motor, del_motor=del_motor, **scales, **offset,
                             mount_tilt=tilt, mount_roll=roll)
    record = arm.to_pyfai_goniometer(DetectorCalibration(poni=poni, detector_config=config, turned_from=turned))
    if tilt or roll:  # the tilted plane moves the beam off its pixel (Codex review F2): put it back
        pose = dict(del_motor=del_motor, nu_motor=nu_motor,
                    readings={moving[circle]: zero for circle, zero in zeros.items()})
        along = _direct_beam_hit(record, **pose)[1]
        poni.dist *= float(distance_m) / along  # the plane's distance scales the one along the beam
        record = arm.to_pyfai_goniometer(DetectorCalibration(poni=poni, detector_config=config, turned_from=turned))
        (hit1, hit2), _ = _direct_beam_hit(record, **pose)
        at1, at2, _ = pixels.calc_cartesian_positions(d1=np.array([hit1]), d2=np.array([hit2]))
        poni.poni1 += float(poni1[0] - at1[0])  # moving the PONI translates the pattern in the plane
        poni.poni2 += float(poni2[0] - at2[0])
        record = arm.to_pyfai_goniometer(DetectorCalibration(poni=poni, detector_config=config, turned_from=turned))
    if beam_readings:  # the beam was seen off the zero pose: put it on its pixel there
        pose = dict(del_motor=del_motor, nu_motor=nu_motor,
                    readings={**{moving[circle]: zero for circle, zero in zeros.items()},
                              **{str(motor): float(value) for motor, value in beam_readings.items()}})
        (hit1, hit2), _ = _direct_beam_hit(record, **pose)
        at1, at2, _ = pixels.calc_cartesian_positions(d1=np.array([hit1]), d2=np.array([hit2]))
        poni.poni1 += float(poni1[0] - at1[0])
        poni.poni2 += float(poni2[0] - at2[0])
        record = arm.to_pyfai_goniometer(DetectorCalibration(poni=poni, detector_config=config, turned_from=turned))
    return record


def _direct_beam_hit(record, *, del_motor, nu_motor, readings):
    """Where the direct beam meets the detector with the motors at these readings (the zero
    pose): its (row, column) pixel coordinate, and its distance from the sample along the beam (m)."""
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.integrate.multi import create_multigeometry_integrator_series_from_geometry
    bound = _source_motors(record, del_motor, nu_motor)
    motors = {motor: np.array([float(readings.get(motor, 0.))]) for motor in bound.values()}
    integrator = next(iter(create_multigeometry_integrator_series_from_geometry(
        Diffractometer.from_pyfai_goniometer(record, source_motors=bound), motors)))
    # lab positions are affine in the pixel coordinates (a flat detector): solve for x = y = 0
    z, y, x = integrator.calc_pos_zyx(d1=np.array([0., 1., 0.]), d2=np.array([0., 0., 1.]))
    z, y, x = (np.ravel(v) for v in (z, y, x))
    step = np.array([[y[1] - y[0], y[2] - y[0]], [x[1] - x[0], x[2] - x[0]]])
    rows, columns = np.linalg.solve(step, -np.array([y[0], x[0]]))
    return (float(rows), float(columns)), float(z[0] + (z[1] - z[0]) * rows + (z[2] - z[0]) * columns)


def write_seed_poni(seed: FrameSeed, path, *, calibrant='LaB6') -> Path:
    """The predicted geometry as a PONI pyFAI-calib2 opens with (``--poni``)."""
    from pyFAI.io.ponifile import PoniFile
    poni = PoniFile(seed.integrator)
    poni.extra['Calibrant'] = str(calibrant)
    path = Path(path)
    with path.open('wt', encoding='utf-8') as stream:
        poni.write(stream)
    return path


def ring_summary(rings) -> str:
    """Compact ring numbers for a table cell: 4–7, 9."""
    numbers = [ring.number for ring in rings]
    parts, start = [], None
    for index, number in enumerate(numbers):
        if start is None:
            start = number
        if index + 1 == len(numbers) or numbers[index + 1] != number + 1:
            parts.append(str(start) if start == number else f'{start}–{number}')
            start = None
    return ', '.join(parts)


__all__ = ['ExpectedRing', 'FrameSeed', 'expected_rings', 'predict_frame_seed', 'ring_summary',
           'rough_arm_record', 'write_seed_poni']
