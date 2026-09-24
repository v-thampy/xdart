"""Exact del-on-nu detector pose and its standard pyFAI JSON encoding.

Angles in source metadata are degrees; fitted scales are radians per degree.
The physical pose is R3(beam_rotation) R1(nu_scale*nu)
R2(del_offset+del_scale*del) R1(rot1_base), using pyFAI's left-handed R1/R2.
"""
from dataclasses import dataclass
import math

import numpy as np


STACKED_ARM_MARKER = 'xdart_stacked_arm_v1'
_PARAM_NAMES = ('dist', 'poni1', 'poni2', 'rot1_base', 'nu_scale',
                'del_offset', 'del_scale', 'beam_rotation')


def stacked_arm_transformation():
    """Closed-form Euler decomposition; numexpr-compatible, ordinary pyFAI JSON."""
    n, d, b = '(nu_scale*nu_angle)', '(del_offset+del_scale*del_angle)', 'rot1_base'
    x = f'(cos({n})*sin({d}))'
    y = f'(cos({n})*cos({d})*sin({b})+sin({n})*cos({b}))'
    z = f'(cos({n})*cos({d})*cos({b})-sin({n})*sin({b}))'
    return dict(content='GeometryTransformation', param_names=list(_PARAM_NAMES),
        pos_names=['del_angle', 'nu_angle'], dist_expr='dist', poni1_expr='poni1',
        poni2_expr='poni2', rot1_expr=f'arctan2({y},{z})',
        rot2_expr=f'arctan2({x},sqrt({y}**2+{z}**2))',
        rot3_expr=f'beam_rotation+arctan2(sin({n})*sin({d}),cos({d}))',
        constants={'pi': math.pi, STACKED_ARM_MARKER: 1})


@dataclass(frozen=True)
class StackedDetectorArm:
    """An explicit nu-outer, del-inner pose; not independent Euler mappings."""
    nu_motor: str = 'nu'
    del_motor: str = 'del'
    rot1_base: float = 0.
    nu_scale: float = math.pi / 180
    del_offset: float = 0.
    del_scale: float = math.pi / 180
    beam_rotation: float = 0.

    def __post_init__(self):
        if (any(not isinstance(name, str) or not name for name in (self.nu_motor, self.del_motor))
                or self.nu_motor == self.del_motor):
            raise ValueError('stacked arm requires distinct, nonempty motor names')
        for name in _PARAM_NAMES[3:]:
            value = getattr(self, name)
            if isinstance(value, (bool, str)) or not math.isfinite(float(value)):
                raise ValueError('stacked arm parameters must be finite numbers')
            object.__setattr__(self, name, float(value))

    def angles(self, delta, nu):
        """Return pyFAI Euler angles in radians, including the induced roll."""
        d = self.del_offset + self.del_scale * np.asarray(delta, dtype=float)
        n = self.nu_scale * np.asarray(nu, dtype=float)
        b = self.rot1_base
        x = np.cos(n) * np.sin(d)
        y = np.cos(n) * np.cos(d) * np.sin(b) + np.sin(n) * np.cos(b)
        z = np.cos(n) * np.cos(d) * np.cos(b) - np.sin(n) * np.sin(b)
        return (np.arctan2(y, z), np.arctan2(x, np.hypot(y, z)),
                self.beam_rotation + np.arctan2(np.sin(n) * np.sin(d), np.cos(d)))

    def rotations(self, motors):
        arrays = [np.atleast_1d(np.asarray(motors[name], dtype=float))
                  for name in (self.del_motor, self.nu_motor)]
        if (any(a.ndim != 1 or not a.size or not np.isfinite(a).all() for a in arrays)
                or arrays[0].shape != arrays[1].shape):
            raise ValueError('stacked arm needs finite paired motor arrays')
        result = dict(zip(('rot1', 'rot2', 'rot3'), self.angles(*arrays)))
        if any(not np.isfinite(a).all() for a in result.values()):
            raise ValueError('stacked arm rotations are not finite')
        return result

    def to_pyfai_goniometer(self, calibration):
        """Encode the exact pose without a pyFAI plugin or custom evaluator."""
        poni = calibration.poni
        if any(getattr(poni, axis) != 0 for axis in ('rot1', 'rot2', 'rot3')):
            raise ValueError('stacked arm carries all rotations; base PONI rotations must be zero')
        values = [float(getattr(poni, name)) for name in _PARAM_NAMES[:3]]
        values += [getattr(self, name) for name in _PARAM_NAMES[3:]]
        record = dict(content='Goniometer calibration v2', detector=poni.detector,
            detector_config=dict(calibration.detector_config), wavelength=poni.wavelength,
            param_names=list(_PARAM_NAMES), param=values, pos_names=['del_angle', 'nu_angle'],
            trans_function=stacked_arm_transformation())
        if calibration.parallax is not None:
            record['parallax'] = calibration.parallax
        return record

    @classmethod
    def from_pyfai_goniometer(cls, data, motor_for):
        """Admit only this explicit model; never approximate arbitrary expressions."""
        trans = data['trans_function']
        if STACKED_ARM_MARKER not in trans.get('constants', {}):
            return None
        if (trans != stacked_arm_transformation()
                or type(trans['constants'][STACKED_ARM_MARKER]) not in (int, float)
                or data.get('param_names') != list(_PARAM_NAMES)
                or data.get('pos_names') != ['del_angle', 'nu_angle']
                or len(data.get('param', ())) != len(_PARAM_NAMES)
                or any(isinstance(v, (bool, str)) or not math.isfinite(float(v))
                       for v in data['param'])):
            raise ValueError('unsupported or inconsistent stacked arm transformation')
        params = dict(zip(_PARAM_NAMES, data['param']))
        return cls(nu_motor=motor_for['nu_angle'], del_motor=motor_for['del_angle'],
                   **{name: params[name] for name in _PARAM_NAMES[3:]})
