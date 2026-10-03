"""Exact del-on-nu detector pose and its standard pyFAI JSON encoding.

Angles in source metadata are degrees; fitted scales are radians per degree.
The physical pose is R3(beam_rotation) R1(nu_offset+nu_scale*nu)
R2(del_offset+del_scale*del), using pyFAI's left-handed R1/R2.

The optional arm extension adds the detector mount on the arm and the sample
offset from the rotation centre::

    M = R3(beam_rotation) R1(n) R2(d) R3(mount_roll) R1(mount_tilt)
    lab = M v - R3(beam_rotation) s,  s = (sample_offset1, 2, 3) in metres

A mount rotation about the del axis is absent on purpose: it only duplicates
``del_offset``. The sample term moves the effective PONI and distance of every
frame (u = M0^T s: poni1 + u1, poni2 + u2, dist - u3), so it is carried by
:meth:`StackedDetectorArm.translations`; :meth:`rotations` stays rotation-only.
An arm whose extension is zero encodes and evaluates exactly as before.

A single detector arm (the 2θ circle of a fourc or twoc) is this arm with one
circle held at zero. A vertical 2θ arm plays del with nu held (``nu_motor=None``;
``nu_offset`` stays as the arm's fixed yaw); a horizontal one plays nu with del
held (``del_motor=None``; ``del_offset`` stays as the detector's fixed pitch).
Its record declares the one moving position (``del_angle`` or ``nu_angle``),
carries no scale for the held circle, and has no extension.
"""
from dataclasses import dataclass, replace
import math

import numpy as np


STACKED_ARM_MARKER = 'xdart_stacked_arm_v1'
EXTENDED_ARM_MARKER = 'xdart_stacked_arm_v2'
SINGLE_ARM_MARKER = 'xdart_single_arm_v1'
STACKED_ARM_MARKERS = (STACKED_ARM_MARKER, EXTENDED_ARM_MARKER, SINGLE_ARM_MARKER)
#: A goniometer record's key for a detector turned by a quarter into the calibration's frame (AUD-9 stage 2b
#: item 8): the pyFAI detector the record names is then the generic one of the turned shape, and this names the
#: model and the turn its images take, so its module gaps can be rebuilt (``xrd_tools.integrate.turned_detector``).
#: Absent for every other detector.
TURNED_DETECTOR_KEY = 'xdart_turned_detector'
_PARAM_NAMES = ('dist', 'poni1', 'poni2', 'nu_offset', 'nu_scale',
                'del_offset', 'del_scale', 'beam_rotation')
_SINGLE_NAMES = ('dist', 'poni1', 'poni2', 'nu_offset', 'del_offset', 'del_scale', 'beam_rotation')
_SINGLE_NU_NAMES = ('dist', 'poni1', 'poni2', 'nu_offset', 'nu_scale', 'del_offset', 'beam_rotation')


def orientation_signs(orientation=3) -> tuple[int, int]:
    """The signs of rot1 (nu's) and rot2 (del's) that keep the arm physical in pyFAI's
    detector ``orientation``.

    pyFAI reflects the pixel coordinates before rotating and the transverse output after
    (F·R·F; Codex R18-F1), so the same rot1/rot2/rot3 tilt the detector another way in another
    orientation: rot1 turns the other way in 1 and 4, rot2 in 1 and 2, and the induced rot3
    follows their product. Motor scales carrying these signs give orientation 3's geometry in
    every orientation, which then changes only how the PONI and the arm are written.
    """
    orientation = int(orientation)
    if orientation not in (1, 2, 3, 4):
        raise ValueError(f"pyFAI's detector orientation is 1 to 4, not {orientation}")
    return (-1 if orientation in (1, 4) else 1), (-1 if orientation in (1, 2) else 1)


def arm_signs(orientation=3, senses=None) -> tuple[int, int]:
    """The signs of nu's and del's motor scales in pyFAI's native frame:
    negative :func:`orientation_signs`, each reversed for a circle that turns opposite to psic's
    (``senses``: {'nu': -1, 'del': -1}, as ``InstrumentGeometry.arm_senses`` gives them; a
    circle not named turns as psic's, nu about x+ and del about z-)."""
    senses = dict(senses or {})
    if set(senses) - {'nu', 'del'} or any(value not in (1, -1) for value in senses.values()):
        raise ValueError(f"a circle's sense is 1 or -1 for nu and del, not {senses}")
    nu, delta = orientation_signs(orientation)
    # pyFAI's native t1=x, t2=-z frame: standard physical nu x+ and del z- have negative Euler scales.
    return -nu * senses.get('nu', 1), -delta * senses.get('del', 1)


def arm_senses(arm, orientation=3) -> dict[str, int]:
    """The senses an arm's moving circles turn with (:func:`arm_signs`'s ``senses``), read from
    its motor scales' signs in this detector orientation."""
    signs = dict(zip(('nu', 'del'), arm_signs(orientation)))
    return {circle: (1 if getattr(arm, f'{circle}_scale') > 0 else -1) * signs[circle]
            for circle, motor in (('nu', arm.nu_motor), ('del', arm.del_motor)) if motor is not None}


#: A single arm's moving position and parameters, by the circle it holds at zero.
_SINGLE = {'nu': ('del_angle', _SINGLE_NAMES), 'del': ('nu_angle', _SINGLE_NU_NAMES)}
_EXTENDED_NAMES = ('mount_tilt', 'mount_roll',
                   'sample_offset1', 'sample_offset2', 'sample_offset3')
ARM_EXTENSION_FIELDS = _EXTENDED_NAMES


def stacked_arm_transformation(extended=False, *, held=None):
    """Closed-form Euler decomposition; numexpr-compatible, ordinary pyFAI JSON.

    ``held`` ('nu' or 'del') is a single arm's circle held at zero; its offset stays.
    """
    if held not in (None, 'nu', 'del'):
        raise ValueError(f'a single arm holds nu or del, not {held!r}')
    if extended:
        if held:
            raise ValueError('a single detector arm has no extension')
        return _extended_transformation()
    n = '(nu_offset)' if held == 'nu' else '(nu_offset+nu_scale*nu_angle)'
    d = '(del_offset)' if held == 'del' else '(del_offset+del_scale*del_angle)'
    x = f'(cos({n})*sin({d}))'
    y = f'sin({n})'
    z = f'(cos({n})*cos({d}))'
    return dict(content='GeometryTransformation',
        param_names=list(_SINGLE[held][1] if held else _PARAM_NAMES),
        pos_names=[_SINGLE[held][0]] if held else ['del_angle', 'nu_angle'],
        dist_expr='dist', poni1_expr='poni1',
        poni2_expr='poni2', rot1_expr=f'arctan2({y},{z})',
        rot2_expr=f'arctan2({x},sqrt({y}**2+{z}**2))',
        rot3_expr=f'beam_rotation+arctan2(sin({n})*sin({d}),cos({d}))',
        constants={'pi': math.pi, SINGLE_ARM_MARKER if held else STACKED_ARM_MARKER: 1})


def _extended_transformation():
    """Generic decomposition of M0 = R1(n) R2(d) R3(roll) R1(tilt), plus u = M0^T s."""
    n, d = '(nu_offset+nu_scale*nu_angle)', '(del_offset+del_scale*del_angle)'
    arm = [[f'cos({d})', '', f'(-sin({d}))'],
           [f'(sin({n})*sin({d}))', f'cos({n})', f'(sin({n})*cos({d}))'],
           [f'(cos({n})*sin({d}))', f'(-sin({n}))', f'(cos({n})*cos({d}))']]
    mount = [['cos(mount_roll)', '(-sin(mount_roll)*cos(mount_tilt))', '(-sin(mount_roll)*sin(mount_tilt))'],
             ['sin(mount_roll)', '(cos(mount_roll)*cos(mount_tilt))', '(cos(mount_roll)*sin(mount_tilt))'],
             ['', '(-sin(mount_tilt))', 'cos(mount_tilt)']]
    m = [['(' + '+'.join(f'{arm[i][k]}*{mount[k][j]}' for k in range(3)
                         if arm[i][k] and mount[k][j]) + ')' for j in range(3)] for i in range(3)]
    u = ['(' + '+'.join(f'{m[i][j]}*sample_offset{i + 1}' for i in range(3)) + ')' for j in range(3)]
    return dict(content='GeometryTransformation', param_names=list(_PARAM_NAMES + _EXTENDED_NAMES),
        pos_names=['del_angle', 'nu_angle'], dist_expr=f'dist-{u[2]}',
        poni1_expr=f'poni1+{u[0]}', poni2_expr=f'poni2+{u[1]}',
        rot1_expr=f'arctan2(-{m[2][1]},{m[2][2]})',
        rot2_expr=f'arctan2({m[2][0]},sqrt({m[2][1]}**2+{m[2][2]}**2))',
        rot3_expr=f'beam_rotation+arctan2({m[1][0]},{m[0][0]})',
        constants={'pi': math.pi, EXTENDED_ARM_MARKER: 1})


def is_single_arm(record) -> bool:
    """Whether a goniometer record is a single detector arm (one circle held at zero)."""
    trans = record.get('trans_function') if isinstance(record, dict) else None
    constants = trans.get('constants') if isinstance(trans, dict) else None
    return isinstance(constants, dict) and SINGLE_ARM_MARKER in constants


def is_arm_record(record) -> bool:
    """Whether a goniometer record is one of these arms (stacked, extended or single), not
    an independent-angle calibration."""
    trans = record.get('trans_function') if isinstance(record, dict) else None
    constants = trans.get('constants') if isinstance(trans, dict) else None
    return isinstance(constants, dict) and any(marker in constants for marker in STACKED_ARM_MARKERS)


def checked_turned_detector(value):
    """A ``TURNED_DETECTOR_KEY`` value as a plain dict: the model (its pyFAI name) and the quarter turn (90 or 270)
    and mirror its images take into the calibration's frame; None for None; ValueError for anything else."""
    if value is None:
        return None
    if (not hasattr(value, 'keys') or set(value) != {'detector', 'image_rotation', 'image_flip'}
            or type(value['detector']) is not str or not value['detector']
            or type(value['image_rotation']) is not int or value['image_rotation'] not in (90, 270)
            or type(value['image_flip']) is not bool):
        raise ValueError('the turned detector record is malformed')
    return {'detector': value['detector'], 'image_rotation': value['image_rotation'],
            'image_flip': value['image_flip']}


@dataclass(frozen=True)
class StackedDetectorArm:
    """An explicit nu-outer, del-inner pose; not independent Euler mappings.

    A single detector arm has no motor for one circle, which is held at zero:
    ``nu_motor=None`` (a vertical 2θ arm) or ``del_motor=None`` (a horizontal one).
    """
    nu_motor: str | None = 'nu'
    del_motor: str | None = 'del'
    nu_offset: float = 0.
    nu_scale: float = -math.pi / 180
    del_offset: float = 0.
    del_scale: float = -math.pi / 180
    beam_rotation: float = 0.
    mount_tilt: float = 0.
    mount_roll: float = 0.
    sample_offset1: float = 0.
    sample_offset2: float = 0.
    sample_offset3: float = 0.

    def __post_init__(self):
        motors = tuple(name for name in (self.nu_motor, self.del_motor) if name is not None)
        if (not motors or any(not isinstance(name, str) or not name for name in motors)
                or len(set(motors)) != len(motors)):
            raise ValueError('stacked arm requires distinct, nonempty motor names')
        for name in _PARAM_NAMES[3:] + _EXTENDED_NAMES:
            value = getattr(self, name)
            if isinstance(value, (bool, str)) or not math.isfinite(float(value)):
                raise ValueError('stacked arm parameters must be finite numbers')
            object.__setattr__(self, name, float(value))
        if self.single and self.extended:
            raise ValueError('a single detector arm has no mount or sample-offset extension: '
                             'it needs frames from both circles')

    @property
    def held(self):
        """The circle a single detector arm holds at zero ('nu' or 'del'), else None."""
        return 'nu' if self.nu_motor is None else 'del' if self.del_motor is None else None

    @property
    def single(self):
        """True for a single detector arm (one circle held at zero)."""
        return self.held is not None

    @property
    def extended(self):
        """True when the mount or sample-offset extension is in use."""
        return any(getattr(self, name) != 0 for name in _EXTENDED_NAMES)

    @property
    def has_sample_offset(self):
        return any(getattr(self, name) != 0 for name in _EXTENDED_NAMES[2:])

    def motor_zeros(self) -> dict:
        """Each moving circle's motor-zero reading in degrees ('del', 'nu'): its readback where the
        circle is at zero, physical angle = readback − zero. The internal offset is
        −scale × zero, the scale carrying the detector orientation's sign (a held circle has no
        motor: its offset is a fixed angle, not a zero)."""
        return {circle: -getattr(self, f'{circle}_offset') / getattr(self, f'{circle}_scale')
                for circle, motor in (('del', self.del_motor), ('nu', self.nu_motor)) if motor is not None}

    def _pose(self, delta, nu):
        """M0 = R1(n) R2(d) R3(roll) R1(tilt) for each frame, shape (..., 3, 3)."""
        delta, nu = np.broadcast_arrays(np.asarray(delta, dtype=float), np.asarray(nu, dtype=float))
        d = self.del_offset + self.del_scale * delta
        n = self.nu_offset + self.nu_scale * nu
        cn, sn, cd, sd = np.cos(n), np.sin(n), np.cos(d), np.sin(d)
        zero = np.zeros_like(d)
        arm = np.stack([np.stack([cd, zero, -sd], -1), np.stack([sn * sd, cn, sn * cd], -1),
                        np.stack([cn * sd, -sn, cn * cd], -1)], -2)
        ct, st = math.cos(self.mount_tilt), math.sin(self.mount_tilt)
        cr, sr = math.cos(self.mount_roll), math.sin(self.mount_roll)
        mount = np.array([[cr, -sr * ct, -sr * st], [sr, cr * ct, cr * st], [0., -st, ct]])
        return arm @ mount

    def angles(self, delta, nu):
        """Return pyFAI Euler angles in radians, including the induced roll."""
        if self.mount_tilt or self.mount_roll:
            m = self._pose(delta, nu)
            return (np.arctan2(-m[..., 2, 1], m[..., 2, 2]),
                    np.arctan2(m[..., 2, 0], np.hypot(m[..., 2, 1], m[..., 2, 2])),
                    self.beam_rotation + np.arctan2(m[..., 1, 0], m[..., 0, 0]))
        d = self.del_offset + self.del_scale * np.asarray(delta, dtype=float)
        n = self.nu_offset + self.nu_scale * np.asarray(nu, dtype=float)
        x = np.cos(n) * np.sin(d)
        y = np.sin(n)
        z = np.cos(n) * np.cos(d)
        return (np.arctan2(y, z), np.arctan2(x, np.hypot(y, z)),
                self.beam_rotation + np.arctan2(np.sin(n) * np.sin(d), np.cos(d)))

    def displacements(self, delta, nu):
        """Per-frame (d_dist, d_poni1, d_poni2) in metres from the sample offset."""
        s = np.array([self.sample_offset1, self.sample_offset2, self.sample_offset3])
        u = np.einsum('...ij,i->...j', self._pose(delta, nu), s)
        return -u[..., 2], u[..., 0], u[..., 1]

    def _arrays(self, motors):
        read = {name: np.atleast_1d(np.asarray(motors[name], dtype=float))
                for name in (self.del_motor, self.nu_motor) if name is not None}
        held = np.zeros(next(iter(read.values())).shape)  # a single arm's other circle
        arrays = [read.get(self.del_motor, held) if self.del_motor is not None else held,
                  read.get(self.nu_motor, held) if self.nu_motor is not None else held]
        if (any(a.ndim != 1 or not a.size or not np.isfinite(a).all() for a in arrays)
                or arrays[0].shape != arrays[1].shape):
            raise ValueError('stacked arm needs finite paired motor arrays')
        return arrays

    def rotations(self, motors):
        arrays = self._arrays(motors)
        result = dict(zip(('rot1', 'rot2', 'rot3'), self.angles(*arrays)))
        if any(not np.isfinite(a).all() for a in result.values()):
            raise ValueError('stacked arm rotations are not finite')
        return result

    def translations(self, motors, poni):
        """Per-frame (dist, poni1, poni2), or None when there is no sample offset."""
        if not self.has_sample_offset:
            return None
        values = self.displacements(*self._arrays(motors))
        dist, poni1, poni2 = (float(poni.dist) + values[0], float(poni.poni1) + values[1],
                              float(poni.poni2) + values[2])
        if not (np.isfinite(dist).all() and np.isfinite(poni1).all() and np.isfinite(poni2).all()
                and (dist > 0).all()):
            raise ValueError('stacked arm sample offset gives a non-physical detector distance')
        return tuple(zip(dist.tolist(), poni1.tolist(), poni2.tolist()))

    def without_sample_offset(self):
        """The instrument part only, for a sample centred on the rotation axis."""
        return replace(self, sample_offset1=0., sample_offset2=0., sample_offset3=0.)

    def to_pyfai_goniometer(self, calibration):
        """Encode the exact pose without a pyFAI plugin or custom evaluator."""
        poni = calibration.poni
        if any(getattr(poni, axis) != 0 for axis in ('rot1', 'rot2', 'rot3')):
            raise ValueError('stacked arm carries all rotations; base PONI rotations must be zero')
        held = self.held
        names = (_SINGLE[held][1] if held
                 else _PARAM_NAMES + (_EXTENDED_NAMES if self.extended else ()))
        values = [float(getattr(poni, name)) for name in _PARAM_NAMES[:3]]
        values += [getattr(self, name) for name in names[3:]]
        record = dict(content='Goniometer calibration v2', detector=poni.detector,
            detector_config=dict(calibration.detector_config), wavelength=poni.wavelength,
            param_names=list(names), param=values,
            pos_names=[_SINGLE[held][0]] if held else ['del_angle', 'nu_angle'],
            trans_function=stacked_arm_transformation(self.extended, held=held))
        if calibration.parallax is not None:
            record['parallax'] = calibration.parallax
        turned = getattr(calibration, 'turned_from', None)
        if turned is not None:  # a quarter-turned detector: its model and turn travel with the record
            record[TURNED_DETECTOR_KEY] = dict(turned)
        return record

    @classmethod
    def from_pyfai_goniometer(cls, data, motor_for):
        """Admit only these explicit models; never approximate arbitrary expressions."""
        trans = data['trans_function']
        constants = trans.get('constants', {})
        positions, held = ['del_angle', 'nu_angle'], None
        if STACKED_ARM_MARKER in constants:
            marker, names, expected = STACKED_ARM_MARKER, _PARAM_NAMES, stacked_arm_transformation()
        elif EXTENDED_ARM_MARKER in constants:
            marker, names = EXTENDED_ARM_MARKER, _PARAM_NAMES + _EXTENDED_NAMES
            expected = stacked_arm_transformation(extended=True)
        elif SINGLE_ARM_MARKER in constants:  # the declared position says which circle moves
            held = next((circle for circle, (position, _names) in _SINGLE.items()
                         if trans.get('pos_names') == [position]), None)
            if held is None:
                raise ValueError('unsupported or inconsistent stacked arm transformation')
            marker, positions, names = SINGLE_ARM_MARKER, [_SINGLE[held][0]], _SINGLE[held][1]
            expected = stacked_arm_transformation(held=held)
        else:
            return None
        if (trans != expected
                or type(trans['constants'][marker]) not in (int, float)
                or data.get('param_names') != list(names)
                or data.get('pos_names') != positions
                or len(data.get('param', ())) != len(names)
                or any(isinstance(v, (bool, str)) or not math.isfinite(float(v))
                       for v in data['param'])):
            raise ValueError('unsupported or inconsistent stacked arm transformation')
        params = dict(zip(names, data['param']))
        arm = cls(nu_motor=None if held == 'nu' else motor_for['nu_angle'],
                  del_motor=None if held == 'del' else motor_for['del_angle'],
                  **{name: params[name] for name in names[3:]})
        if marker == EXTENDED_ARM_MARKER and not arm.extended:
            raise ValueError('an extended stacked arm record must carry a nonzero extension')
        return arm
