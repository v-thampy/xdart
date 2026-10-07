"""Computed, source-free geometry figures in stored-pixel or declared lab coordinates.

The detector camera, arm calibration, ring metadata and beam intersection retain
their existing owners. These values contain no GUI, integrator, source or job.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from xrd_tools.core.geometry.camera import (
    DetectorCamera, mounting_camera, mounting_description, stitch_orientation,
)


def _points(value):
    points = np.array(value, dtype=float, copy=True)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('figure points have shape (n, 2)')
    if not np.isfinite(points).all():
        raise ValueError('figure points must be finite')
    points.setflags(write=False)
    return points


@dataclass(frozen=True)
class Curve:
    name: str
    xy: object
    role: str
    tooltip: str
    two_theta_deg: float | None = None

    def __post_init__(self):
        object.__setattr__(self, 'xy', _points(self.xy))


@dataclass(frozen=True)
class Marker:
    name: str
    xy: tuple[float, float]
    role: str
    tooltip: str

    def __post_init__(self):
        object.__setattr__(self, 'xy', tuple(_points((self.xy,))[0]))


@dataclass(frozen=True)
class Arrow:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    role: str
    tooltip: str

    def __post_init__(self):
        pair = _points((self.start, self.end))
        object.__setattr__(self, 'start', tuple(pair[0]))
        object.__setattr__(self, 'end', tuple(pair[1]))


@dataclass(frozen=True)
class Text:
    name: str
    xy: tuple[float, float]
    text: str
    tooltip: str

    def __post_init__(self):
        object.__setattr__(self, 'xy', tuple(_points((self.xy,))[0]))


@dataclass(frozen=True)
class Figure:
    title: str
    x_label: str
    y_label: str
    units: str
    caption: str
    curves: tuple[Curve, ...] = ()
    markers: tuple[Marker, ...] = ()
    arrows: tuple[Arrow, ...] = ()
    texts: tuple[Text, ...] = ()
    image: np.ndarray | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        for values, kind in ((self.curves, Curve), (self.markers, Marker),
                             (self.arrows, Arrow), (self.texts, Text)):
            if type(values) is not tuple or any(type(v) is not kind for v in values):
                raise TypeError('figure overlays are typed tuples')
        if self.image is not None:
            image = np.array(self.image, copy=True)
            if image.ndim != 2:
                raise ValueError('figure image has shape (row, column)')
            image.setflags(write=False)
            object.__setattr__(self, 'image', image)


@dataclass(frozen=True)
class GeometryInputs:
    shape: tuple[int, int] = (1043, 981)
    detector: str = 'Pilatus1M'
    distance_m: float = .2
    wavelength_m: float = 1e-10
    beam_offset_px: tuple[float, float] = (-190., 229.)
    mount_deg: tuple[float, float] = (0., 0.)
    readings: tuple[tuple[str, float], ...] = (('del', 0.), ('nu', 0.))
    first_pixel: int = 3

    def __post_init__(self):
        if len(self.shape) != 2 or any(type(v) is not int or v < 2 for v in self.shape):
            raise ValueError('detector shape is positive (rows, columns)')
        if (len(self.mount_deg) != 2 or len(self.beam_offset_px) != 2
                or self.first_pixel not in (1, 2, 3, 4)):
            raise ValueError('declare two mount angles, a beam offset and first-pixel corner')
        if set(dict(self.readings)) != {'del', 'nu'}:
            raise ValueError('declare both del and nu motor readbacks')
        scalars = (self.distance_m, self.wavelength_m, *self.beam_offset_px,
                   *self.mount_deg, *(v for _, v in self.readings))
        if not all(math.isfinite(v) for v in scalars) or min(self.distance_m, self.wavelength_m) <= 0:
            raise ValueError('geometry inputs must be finite, with positive distance and wavelength')


@dataclass(frozen=True)
class CameraCandidate:
    camera: DetectorCamera
    label: str
    tooltip: str

    @property
    def orientation(self):
        return stitch_orientation(self.camera)

    @property
    def lab_up(self):
        return self.camera.raw_vector('x+')

    @property
    def lab_z(self):
        return self.camera.raw_vector('z+')

    def raw_pixel(self, xy, shape):
        return self.orientation.unapply_pixel(xy, shape)


def camera_candidates():
    return tuple(CameraCandidate(camera, mounting_description(camera),
        f'Looking downstream; stored rows {camera.rows}, columns {camera.columns}. '
        'This is a mounting hypothesis, separate from the first-pixel corner.')
        for rotation in (0, 90, 180, 270) for flip in (False, True)
        for camera in (mounting_camera(rotation, flip),))


def first_pixel_figure(code, shape):
    from pyFAI.detectors.orientation import Orientation
    if code not in (1, 2, 3, 4):
        raise ValueError('first-pixel corner is 1–4')
    rows, columns = shape
    corner = {1: (0., 0.), 2: (columns-1., 0.),
              3: (columns-1., rows-1.), 4: (0., rows-1.)}[code]
    return FirstPixelFigure('First-pixel corner', 'Stored column', 'Stored row', 'px',
        'Synthetic pixel centres. The first-pixel corner is separate from detector mounting.',
        markers=(Marker('first pixel', corner, 'beam', 'pyFAI first-pixel corner.'),),
        corner_name=Orientation(code).name)


@dataclass(frozen=True)
class FirstPixelFigure(Figure):
    corner_name: str = ''


def _geometry(value, camera, senses=(1, 1)):
    from xrd_tools.integrate.calibration_seed import rough_arm_record, predict_frame_seed
    from xrd_tools.integrate.stitch_calibration import CalibrationFrame
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    orientation = stitch_orientation(camera)
    shape = value.shape[::-1] if orientation.swaps_axes else value.shape
    centre = (np.array(shape)[::-1]-1)/2
    record = rough_arm_record(detector=value.detector, distance_m=value.distance_m,
        wavelength_m=value.wavelength_m, beam_px=centre+value.beam_offset_px,
        camera=orientation, mount_deg=value.mount_deg, orientation=value.first_pixel,
        senses=dict(zip(('del', 'nu'), senses)))
    frame = CalibrationFrame('synthetic', 'synthetic', 'spec', {}, 0, '', 0, '', '', shape,
                             0, dict(value.readings), wavelength_m_to_energy_eV(value.wavelength_m))
    return record, predict_frame_seed(record, frame), shape


@dataclass(frozen=True)
class RingPrediction(Figure):
    rings: tuple = ()
    beam_xy: tuple[float, float] | None = None
    beam_off_detector: bool = False


def predict_ring_curves(value, camera, senses=(1, 1)):
    """Trace the trusted integrator's calibrated 2θ contours, then return raw pixels.

    The map is released with its integrator. No generated mockup or rasterized
    prediction supplies a synthetic observed image.
    """
    from contourpy import contour_generator
    from xrd_tools.integrate.calibration_seed import _direct_beam_hit
    record, seed, shape = _geometry(value, camera, senses)
    ai = seed.integrator
    contours = contour_generator(z=ai.center_array(shape, unit='2th_rad', scale=False))
    transform = stitch_orientation(camera)
    curves = []
    for ring in seed.rings:
        for index, line in enumerate(contours.lines(math.radians(ring.two_theta_deg))):
            raw = tuple(transform.unapply_pixel(xy, value.shape) for xy in line)
            curves.append(Curve(f'Ring {ring.number}, arc {index+1}', raw, 'predicted',
                f'Predicted LaB6 ring {ring.number}: 2θ {ring.two_theta_deg:.4g}°.', ring.two_theta_deg))
    (row, column), _ = _direct_beam_hit(record, del_motor='del', nu_motor='nu',
                                      readings=dict(value.readings))
    beam = transform.unapply_pixel((column, row), value.shape)
    outside = not (0 <= beam[0] <= value.shape[1]-1 and 0 <= beam[1] <= value.shape[0]-1)
    arrows = ()
    if outside:
        start = tuple(np.clip(beam, (0, 0), np.array(value.shape)[::-1]-1))
        vector = np.array(beam)-start
        end = tuple(np.array(start)+vector/max(np.linalg.norm(vector), 1)*30)
        arrows = (Arrow('off-detector beam', start, end, 'beam', 'Direct beam lies outside this detector.'),)
    return RingPrediction(mounting_description(camera), 'Stored column', 'Stored row', 'px',
        'Synthetic prediction with fixed distance, beam, energy, mount and motor zeros. '
        'Re-fitting those inputs may compensate a different mounting; ring agreement alone is conditional.',
        curves=tuple(curves), markers=(Marker('beam', beam, 'beam', 'Predicted direct beam at this pose.'),),
        arrows=arrows, rings=seed.rings, beam_xy=beam, beam_off_detector=outside)


def synthetic_orientation_example(value=None, camera=None):
    """Upstream calibrant image, independent of the new plotted polylines."""
    from pyFAI.calibrant import get_calibrant
    value = value or GeometryInputs()
    camera = camera or DetectorCamera('x-', 'z+')
    _, seed, shape = _geometry(value, camera)
    calibrant = get_calibrant('LaB6')
    calibrant.wavelength = value.wavelength_m
    pitch = seed.integrator.detector.pixel1
    image = calibrant.fake_calibration_image(seed.integrator, shape=shape, Imax=200.,
                                            resolution=np.degrees(pitch/value.distance_m))
    prediction = predict_ring_curves(value, camera)
    return replace(prediction, image=stitch_orientation(camera).unapply(image),
                   caption='Synthetic LaB6 image from pyFAI; overlays are predicted, not measured control points. '
                   'Off-centre rings can discriminate only away from mirror lines and diagonals, with trusted inputs.')
