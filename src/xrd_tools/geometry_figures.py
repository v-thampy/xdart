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


def residual_example():
    """Evaluate three declared trial geometries at the same synthetic ring points.

    These are mechanisms, not a classifier for residuals on a user's data. The
    float64 point call is the existing calibration residual owner (qFunction/10).
    """
    from scipy.optimize import brentq
    value = GeometryInputs(mount_deg=(2., -1.))
    camera = DetectorCamera('x-', 'z+')
    _, seed, _ = _geometry(value, camera)
    ai = seed.integrator
    transform = stitch_orientation(camera)
    prediction = predict_ring_curves(value, camera)
    points, targets = [], []
    for curve in prediction.curves:
        ring = next(r for r in prediction.rings if r.two_theta_deg == curve.two_theta_deg)
        target = 2*math.pi/ring.d_spacing_A
        for raw in curve.xy[::max(1, len(curve.xy)//8)]:
            column, row = transform.apply_pixel(raw, value.shape)
            residual = lambda t: float(ai.qFunction(np.array([row+t]), np.array([column]))[0]/10-target)
            if residual(-1)*residual(1) > 0:
                continue  # tangency: keep points for which the normal crosses a row
            row += brentq(residual, -1., 1., xtol=1e-12)
            points.append(transform.unapply_pixel((column, row), value.shape))
            targets.append(target)
    raw = np.asarray(points)
    targets = np.asarray(targets)
    examples = {}
    trials = (('good fit', value, camera),
              ('zero offset', replace(value, readings=(('del', 2.), ('nu', 0.))), camera),
              ('wrong mounting', value, DetectorCamera('x+', 'z+')))
    for name, trial, mounting in trials:
        _, trial_seed, _ = _geometry(trial, mounting)
        xy = np.asarray([stitch_orientation(mounting).apply_pixel(p, value.shape) for p in raw])
        dq = trial_seed.integrator.qFunction(xy[:, 1], xy[:, 0])/10-targets
        examples[name] = Figure(name, 'Ring q', 'q residual', 'Å⁻¹',
            f'Synthetic {name}: the same LaB6 control points, fixed beam, distance and energy. '
            'Crosses are illustrative rejected points; real residual trends do not uniquely identify a fault.',
            curves=(Curve('q residual', np.column_stack((targets, dq)), 'observed',
                          'Trial q minus calibrant q, using float64 calibration point evaluation.'),),
            markers=tuple(Marker('rejected point', (targets[i], dq[i]), 'rejected',
                                 'Illustrative exclusion, not an automatic rejection rule.')
                          for i in range(0, len(targets), 11)))
    return examples


@dataclass(frozen=True)
class InstrumentCircle:
    name: str
    axis: str
    angle_deg: float
    lab_axis: tuple[float, float, float]


@dataclass(frozen=True)
class InstrumentFigure:
    circles: tuple[InstrumentCircle, ...] = ()
    panels: tuple[Figure, ...] = ()
    reason: str = ''


def instrument_figure(instrument, readings):
    """Project the configured, separate sample and detector stacks in three views.

    Schematic locations are illustrative. Axis directions and nested rotations
    come from InstrumentGeometry and xrayutilities, never the drawing mockups.
    """
    from xrd_tools.session.instrument import InstrumentGeometry
    from xrayutilities.math import getVector, ArbRotation, rotarb
    if not isinstance(instrument, InstrumentGeometry):
        return InstrumentFigure(reason='Unsupported schematic geometry: use psic, fourc or twoc.')
    missing = [instrument.motor(c) for c in instrument.circles if instrument.motor(c) not in readings]
    if missing:
        return InstrumentFigure(reason='Missing motor readback: '+', '.join(missing))
    if not all(math.isfinite(float(readings[instrument.motor(c)])) for c in instrument.circles):
        return InstrumentFigure(reason='Nonfinite motor readback.')
    axes = dict(instrument.circle_axes())
    circles = []
    for role in ('sample', 'detector'):
        prefix = np.eye(3)
        for name in instrument.circles:
            if instrument.role(name) != role:
                continue
            axis = axes[name]
            vector = getVector(axis)
            angle = float(readings[instrument.motor(name)])-(instrument.zero(name) or 0.)
            circles.append(InstrumentCircle(name, axis, angle, tuple(prefix @ vector)))
            prefix = prefix @ ArbRotation(vector, angle).matrix
    panels = []
    for title, pair, labels in (('Side view', (1, 0), ('Lab y (beam)', 'Lab x (up)')),
                               ('Top view', (1, 2), ('Lab y (beam)', 'Lab z')),
                               ('Looking downstream', (2, 0), ('Lab z (left)', 'Lab x (up)'))):
        project = lambda vector: tuple(np.asarray(vector)[list(pair)])
        arrows = [Arrow('incident beam', project((0., -.8, 0.)), project((0., .8, 0.)),
                        'beam', 'Lab +y points downstream along the incident beam.')]
        curves, texts = [], []
        for i, circle in enumerate(circles):
            axis = np.asarray(circle.lab_axis)
            base = np.array((0., .12*i, 0.))
            arrows.append(Arrow(circle.name, project(base), project(base+.6*axis), 'predicted',
                f'{circle.name}: signed axis {circle.axis}, readback minus zero {circle.angle_deg:g}°. '
                'Positive sense follows the right-hand rule; sample and detector stacks are independent.'))
            normal = np.eye(3)[int(np.argmin(abs(axis)))]
            tangent = np.cross(axis, normal); tangent /= np.linalg.norm(tangent)
            arc = [project(base+.15*rotarb(tangent, axis, angle)) for angle in np.linspace(0, 80, 25)]
            curves.append(Curve(circle.name+' positive sense', arc, 'predicted',
                                f'Positive rotation about the signed lab axis of {circle.name}.'))
            texts.append(Text(circle.name, project(base+.7*axis),
                              f'{circle.name} {circle.angle_deg:g}°', arrows[-1].tooltip))
        panels.append(Figure(f'{instrument.geometry}: {title}', *labels, 'schematic',
            'Computed schematic, looking in the declared lab view; positions are not to scale. '
            'Configured conventions, not experimentally validated circle senses. '
            'Zero is subtracted once; reversed sense is carried by the signed axis.',
            curves=tuple(curves), arrows=tuple(arrows), texts=tuple(texts)))
    return InstrumentFigure(tuple(circles), tuple(panels))


HELP_FIGURE_TITLES = (('coordinates', 'Array, public image and lab coordinates'),
    ('mountings', 'Eight detector mountings'), ('first-pixel', 'First-pixel corners'),
    ('off-centre', 'Off-centre powder rings'), ('centred', 'Centred rings cannot decide'),
    ('arm-motion', 'Detector-arm motion'), ('mirror-tie', 'Flip or motor sense?'),
    ('tilt', 'Tilt and elliptical rings'), ('psic', 'PSIC geometry'),
    ('fourc', 'Four-circle and two-circle geometry'), ('residuals', 'Residual mechanisms'))


def help_figures(key, candidate_index=0):
    """Finite computed teaching catalogue; never load a file or dispatch a job."""
    candidate = camera_candidates()[candidate_index]
    camera = candidate.camera
    if key in ('psic', 'fourc'):
        from xrd_tools.session.instrument import InstrumentGeometry
        names = ('psic',) if key == 'psic' else ('fourc', 'twoc')
        return tuple(panel for name in names for panel in instrument_figure(
            InstrumentGeometry(name), {c: 0. for c in InstrumentGeometry(name).circles}).panels)
    if key == 'residuals':
        return tuple(residual_example().values())
    if key == 'first-pixel':
        return tuple(first_pixel_figure(code, (5, 7)) for code in (1, 2, 3, 4))
    if key in ('coordinates', 'mountings'):
        image = np.arange(35).reshape(5, 7)
        values = camera_candidates() if key == 'mountings' else (candidate,)
        figures = []
        for item in values:
            origin = np.array((3., 2.))
            arrows = tuple(Arrow(label, tuple(origin), tuple(origin+2*np.asarray(vector)), 'beam',
                f'{label} in stored pixels, looking from behind towards the sample; '
                'looking downstream reverses left and right.')
                for label, vector in (('Lab +x (up)', item.lab_up), ('Lab +z', item.lab_z)))
            figures.append(Figure(item.label, 'Stored column / public x', 'Stored row / public y', 'px',
                'Synthetic numbered array: array[row, column]; public (x, y)=(column, row). '
                'Lab x is up, y downstream, z left looking downstream. Mounting is separate from the corner.',
                arrows=arrows, image=image))
        if key == 'coordinates':
            figures += [replace(figures[0], title='Public image (x, y)'),
                        Figure('Canonical lab view', 'Canonical column', 'Canonical row', 'px',
                            'Synthetic array transformed once through the selected camera. '
                            'The canonical pyFAI frame has rows +x and columns −z.',
                            image=candidate.orientation.apply(image))]
        return tuple(figures)
    if key == 'arm-motion':
        return tuple(predict_ring_curves(replace(GeometryInputs(), readings=(('del', a), ('nu', 0.))), camera)
                     for a in (0., 15., 30.))
    value = GeometryInputs()
    if key == 'centred':
        value = replace(value, beam_offset_px=(0., 0.))
    elif key == 'mirror-tie':
        value = replace(value, beam_offset_px=(0., 229.), readings=(('del', 15.), ('nu', 0.)))
    elif key == 'tilt':
        value = replace(value, mount_deg=(2., -1.))
    elif key != 'off-centre':
        raise ValueError('Unknown computed help figure: '+str(key))
    observed = synthetic_orientation_example(value, DetectorCamera('x-', 'z+'))
    prediction = predict_ring_curves(value, camera)
    reference = tuple(replace(c, name='Reference '+c.name, role='observed',
                              tooltip='Computed synthetic truth; solid reference, not measured control points.')
                      for c in observed.curves)
    note = ('Centred untilted rings are not identifiable from this frame. '
            if key == 'centred' else 'Mirror-line flips and corresponding motor-sense reversals can remain tied. '
            if key == 'mirror-tie' else 'Ring discrimination is conditional on trusted geometry and adequate support. ')
    return (replace(prediction, image=observed.image, curves=reference+prediction.curves,
                    caption=note+'Synthetic LaB6 image: solid reference, dashed candidate. '
                    'First-pixel corner is fixed; no camera or motor setting is changed.'),)
