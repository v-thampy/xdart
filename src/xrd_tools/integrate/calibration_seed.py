"""A starting geometry and the expected calibrant rings for one guided calibration frame.

The seed is a goniometer calibration (this session's fit, a calibration file, or a
rough geometry) evaluated at the frame's own del/nu, with the frame's own energy.
pyFAI-calib2 opens with it, so its ring extraction ("Fit screen") places and numbers
the rings; the same prediction lists which calibrant rings reach the detector.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExpectedRing:
    number: int  # pyFAI's ring number, 1-based as its peak picking shows it
    two_theta_deg: float
    d_spacing_A: float


@dataclass(frozen=True)
class FrameSeed:
    integrator: object  # the frame's predicted pyFAI geometry
    rings: tuple[ExpectedRing, ...]


def _source_motors(record, del_motor, nu_motor):
    """The calibration's angle names (del_angle / del_value ...) bound to the scan's motors."""
    trans = record.get('trans_function') or {}
    names = list(trans.get('pos_names') or record.get('pos_names') or ())
    mapping = {name: del_motor for name in names if 'del' in name.casefold()}
    mapping.update({name: nu_motor for name in names if 'nu' in name.casefold() and name not in mapping})
    if len(names) != 2 or len(mapping) != 2 or len(set(mapping.values())) != 2:
        raise ValueError("the calibration's angles are not one del and one nu")
    return mapping


def expected_rings(integrator, calibrant, shape) -> tuple[ExpectedRing, ...]:
    """The calibrant rings whose 2θ falls on an unmasked pixel of this geometry."""
    from pyFAI.calibrant import get_calibrant
    reference = get_calibrant(calibrant) if isinstance(calibrant, str) else calibrant
    reference.wavelength = float(integrator.wavelength)
    two_theta = np.asarray(integrator.center_array(tuple(shape), unit='2th_rad', scale=False), dtype=float)
    valid = np.isfinite(two_theta)
    mask = getattr(integrator.detector, 'mask', None)
    if mask is not None and np.shape(mask) == two_theta.shape:
        valid &= ~np.asarray(mask, dtype=bool)
    if not valid.any():
        return ()
    low, high = float(two_theta[valid].min()), float(two_theta[valid].max())
    rings = []
    for number, (angle, d) in enumerate(zip(reference.get_2th(), reference.dspacing), 1):
        if angle is not None and low <= angle <= high:
            rings.append(ExpectedRing(number, math.degrees(angle), float(d)))
    return tuple(rings)


def predict_frame_seed(record, frame, *, calibrant='LaB6', del_motor='del', nu_motor='nu') -> FrameSeed:
    """The frame's geometry predicted from a goniometer calibration, with its rings."""
    from pyFAI import units
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.integrate.multi import create_multigeometry_integrator_series_from_geometry
    if not isinstance(record, dict):
        record = json.loads(Path(record).read_text(encoding='utf-8'))
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors=_source_motors(record, del_motor, nu_motor))
    motors = {del_motor: np.array([float(frame.motors[del_motor])]),
              nu_motor: np.array([float(frame.motors[nu_motor])])}
    integrator = next(iter(create_multigeometry_integrator_series_from_geometry(diff, motors)))
    if tuple(integrator.detector.shape) != tuple(frame.shape):
        raise ValueError('the calibration detector does not match the staged frame')
    if frame.energy_eV:  # the frame's own energy, as the scan recorded it
        integrator.wavelength = units.hc / (float(frame.energy_eV) / 1000.) * 1e-10
    return FrameSeed(integrator, expected_rings(integrator, calibrant, frame.shape))


def rough_arm_record(*, distance_m, detector, beam_px, zero_deg=0., zero_motor='del',
                     del_motor='del', nu_motor='nu', wavelength_m=1e-10) -> dict:
    """A nominal stacked (del-on-nu) arm from a rough distance, for a new setup.

    At normal incidence (``zero_motor`` at ``zero_deg``, the other angle at 0) the direct
    beam hits ``beam_px`` (x, y in the oriented image) with the detector ``distance_m``
    from the sample; the angles move the detector as a del-on-nu arm, one degree per
    degree. Good enough to predict which rings a frame shows; the fit refines it.
    """
    from pyFAI.detectors import detector_factory
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    if not math.isfinite(float(distance_m)) or float(distance_m) <= 0:
        raise ValueError('the rough distance must be a positive number')
    pixels = detector_factory(detector)
    x, y = (float(value) for value in beam_px)
    if zero_motor not in (del_motor, nu_motor):
        raise ValueError(f'the normal-incidence zero names {zero_motor!r}, neither {del_motor!r} nor {nu_motor!r}')
    offset = {('del_offset' if zero_motor == del_motor else 'nu_offset'): -math.radians(float(zero_deg))}
    poni = PONI(dist=float(distance_m), poni1=(y + .5) * pixels.pixel1, poni2=(x + .5) * pixels.pixel2,
                wavelength=float(wavelength_m), detector=str(detector))
    arm = StackedDetectorArm(nu_motor=nu_motor, del_motor=del_motor, **offset)
    return arm.to_pyfai_goniometer(DetectorCalibration(poni=poni))


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
