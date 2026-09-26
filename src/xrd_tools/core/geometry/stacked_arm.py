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

A single detector arm (the 2θ circle of a fourc or twoc) is this arm with nu
held at zero: the 2θ circle plays del, and ``nu_offset`` stays as the arm's
fixed yaw. It has no nu motor (``nu_motor=None``), so its record declares one
position, ``del_angle``, and carries no nu scale and no extension.
"""
from dataclasses import dataclass, replace
import math

import numpy as np


STACKED_ARM_MARKER = 'xdart_stacked_arm_v1'
EXTENDED_ARM_MARKER = 'xdart_stacked_arm_v2'
SINGLE_ARM_MARKER = 'xdart_single_arm_v1'
STACKED_ARM_MARKERS = (STACKED_ARM_MARKER, EXTENDED_ARM_MARKER, SINGLE_ARM_MARKER)
_PARAM_NAMES = ('dist', 'poni1', 'poni2', 'nu_offset', 'nu_scale',
                'del_offset', 'del_scale', 'beam_rotation')
_SINGLE_NAMES = ('dist', 'poni1', 'poni2', 'nu_offset', 'del_offset', 'del_scale', 'beam_rotation')
_EXTENDED_NAMES = ('mount_tilt', 'mount_roll',
                   'sample_offset1', 'sample_offset2', 'sample_offset3')
ARM_EXTENSION_FIELDS = _EXTENDED_NAMES


def stacked_arm_transformation(extended=False, *, single=False):
    """Closed-form Euler decomposition; numexpr-compatible, ordinary pyFAI JSON."""
    if extended:
        if single:
            raise ValueError('a single detector arm has no extension')
        return _extended_transformation()
    n = '(nu_offset)' if single else '(nu_offset+nu_scale*nu_angle)'
    d = '(del_offset+del_scale*del_angle)'
    x = f'(cos({n})*sin({d}))'
    y = f'sin({n})'
    z = f'(cos({n})*cos({d}))'
    return dict(content='GeometryTransformation',
        param_names=list(_SINGLE_NAMES if single else _PARAM_NAMES),
        pos_names=['del_angle'] if single else ['del_angle', 'nu_angle'],
        dist_expr='dist', poni1_expr='poni1',
        poni2_expr='poni2', rot1_expr=f'arctan2({y},{z})',
        rot2_expr=f'arctan2({x},sqrt({y}**2+{z}**2))',
        rot3_expr=f'beam_rotation+arctan2(sin({n})*sin({d}),cos({d}))',
        constants={'pi': math.pi, SINGLE_ARM_MARKER if single else STACKED_ARM_MARKER: 1})


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
    """Whether a goniometer record is a single detector arm (nu held at zero)."""
    trans = record.get('trans_function') if isinstance(record, dict) else None
    constants = trans.get('constants') if isinstance(trans, dict) else None
    return isinstance(constants, dict) and SINGLE_ARM_MARKER in constants


@dataclass(frozen=True)
class StackedDetectorArm:
    """An explicit nu-outer, del-inner pose; not independent Euler mappings.

    ``nu_motor=None`` is a single detector arm: nu is held at zero.
    """
    nu_motor: str | None = 'nu'
    del_motor: str = 'del'
    nu_offset: float = 0.
    nu_scale: float = math.pi / 180
    del_offset: float = 0.
    del_scale: float = math.pi / 180
    beam_rotation: float = 0.
    mount_tilt: float = 0.
    mount_roll: float = 0.
    sample_offset1: float = 0.
    sample_offset2: float = 0.
    sample_offset3: float = 0.

    def __post_init__(self):
        motors = (self.del_motor,) if self.nu_motor is None else (self.nu_motor, self.del_motor)
        if (any(not isinstance(name, str) or not name for name in motors)
                or self.nu_motor == self.del_motor):
            raise ValueError('stacked arm requires distinct, nonempty motor names')
        for name in _PARAM_NAMES[3:] + _EXTENDED_NAMES:
            value = getattr(self, name)
            if isinstance(value, (bool, str)) or not math.isfinite(float(value)):
                raise ValueError('stacked arm parameters must be finite numbers')
            object.__setattr__(self, name, float(value))
        if self.single and self.extended:
            raise ValueError('a single detector arm has no mount or sample-offset extension: '
                             'it needs frames from several nu positions')

    @property
    def single(self):
        """True for a single detector arm (no nu motor; nu held at zero)."""
        return self.nu_motor is None

    @property
    def extended(self):
        """True when the mount or sample-offset extension is in use."""
        return any(getattr(self, name) != 0 for name in _EXTENDED_NAMES)

    @property
    def has_sample_offset(self):
        return any(getattr(self, name) != 0 for name in _EXTENDED_NAMES[2:])

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
        delta = np.atleast_1d(np.asarray(motors[self.del_motor], dtype=float))
        arrays = [delta, np.zeros_like(delta) if self.single
                  else np.atleast_1d(np.asarray(motors[self.nu_motor], dtype=float))]
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
        names = (_SINGLE_NAMES if self.single
                 else _PARAM_NAMES + (_EXTENDED_NAMES if self.extended else ()))
        values = [float(getattr(poni, name)) for name in _PARAM_NAMES[:3]]
        values += [getattr(self, name) for name in names[3:]]
        record = dict(content='Goniometer calibration v2', detector=poni.detector,
            detector_config=dict(calibration.detector_config), wavelength=poni.wavelength,
            param_names=list(names), param=values,
            pos_names=['del_angle'] if self.single else ['del_angle', 'nu_angle'],
            trans_function=stacked_arm_transformation(self.extended, single=self.single))
        if calibration.parallax is not None:
            record['parallax'] = calibration.parallax
        return record

    @classmethod
    def from_pyfai_goniometer(cls, data, motor_for):
        """Admit only these explicit models; never approximate arbitrary expressions."""
        trans = data['trans_function']
        constants = trans.get('constants', {})
        positions = ['del_angle', 'nu_angle']
        if STACKED_ARM_MARKER in constants:
            marker, names, expected = STACKED_ARM_MARKER, _PARAM_NAMES, stacked_arm_transformation()
        elif EXTENDED_ARM_MARKER in constants:
            marker, names = EXTENDED_ARM_MARKER, _PARAM_NAMES + _EXTENDED_NAMES
            expected = stacked_arm_transformation(extended=True)
        elif SINGLE_ARM_MARKER in constants:
            marker, names, positions = SINGLE_ARM_MARKER, _SINGLE_NAMES, ['del_angle']
            expected = stacked_arm_transformation(single=True)
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
        arm = cls(nu_motor=None if marker == SINGLE_ARM_MARKER else motor_for['nu_angle'],
                  del_motor=motor_for['del_angle'], **{name: params[name] for name in names[3:]})
        if marker == EXTENDED_ARM_MARKER and not arm.extended:
            raise ValueError('an extended stacked arm record must carry a nonzero extension')
        return arm
