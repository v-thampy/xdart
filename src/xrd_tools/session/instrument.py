"""The diffractometer an operation runs on: a standard circle geometry and its motor names.

One small value, shared by every page that needs to know which scan motor drives which
circle (Stitch and its calibration now, RSM next). The geometry names the circles; the
motors are whatever the beamline calls them, defaulting to the standard names. The
existing :class:`Diffractometer` presets carry the circle conventions.

fourc and twoc are psic's special cases (psic with mu = nu = 0 is the four-circle):
their 2θ circle plays psic's del and their θ plays eta, so Stitch's detector arm is
psic's with nu held at zero.

``halpha`` is a motor name (for the incidence circle, ``eta`` or ``th``), not a geometry.
xdart's ``sixc`` is an alias of psic; SPEC's own sixc is a different geometry and has no
preset here.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from xrd_tools.session.gi_motor import GI_MOTOR_PREFERENCE

#: Ordered circles per geometry: (circle, role). The standard motor name is the circle's.
GEOMETRIES = {
    'psic': (('mu', 'sample'), ('eta', 'sample'), ('chi', 'sample'), ('phi', 'sample'),
             ('nu', 'detector'), ('del', 'detector')),
    'fourc': (('th', 'sample'), ('chi', 'sample'), ('phi', 'sample'), ('tth', 'detector')),
    'twoc': (('th', 'sample'), ('tth', 'detector')),
}
GEOMETRY_LABELS = {'psic': 'psic (6-circle)', 'fourc': 'fourc (4-circle)', 'twoc': 'twoc (2-circle)'}
#: The incidence circle, whose motor is often named differently (halpha, th, ...).
INCIDENCE_CIRCLE = {'psic': 'eta', 'fourc': 'th', 'twoc': 'th'}
#: The psic circle each circle of a special case plays.
AS_PSIC = {'psic': {}, 'fourc': {'th': 'eta', 'chi': 'chi', 'phi': 'phi', 'tth': 'del'},
           'twoc': {'th': 'eta', 'tth': 'del'}}
#: Other names beamlines give a circle's motor (matched whole, ignoring case).
_ALIASES = {'del': ('delta',), 'tth': ('2theta', 'twotheta', 'two_theta', 'tt'),
            'th': ('theta', 'om', 'omega')}
#: Suffixes a calibration file's position names carry (del_angle, nu_value, ...).
_POSITION_SUFFIX = re.compile(r'[_\-. ]?(angle|value|pos|position|deg)$', re.IGNORECASE)


@dataclass(frozen=True)
class InstrumentGeometry:
    geometry: str = 'psic'
    motors: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.geometry not in GEOMETRIES:
            raise ValueError(f'unknown diffractometer geometry {self.geometry!r}')
        given = dict(self.motors)
        circles = [circle for circle, _role in GEOMETRIES[self.geometry]]
        if set(given) - set(circles):
            raise ValueError(f'{self.geometry} has no circle {sorted(set(given) - set(circles))}')
        motors = tuple((circle, str(given.get(circle) or circle).strip() or circle) for circle in circles)
        object.__setattr__(self, 'motors', motors)

    @property
    def circles(self) -> tuple[str, ...]:
        return tuple(circle for circle, _motor in self.motors)

    def motor(self, circle) -> str:
        return dict(self.motors)[circle]

    def role(self, circle) -> str:
        return dict(GEOMETRIES[self.geometry])[circle]

    def stitch_arm(self) -> tuple[str, str | None]:
        """The (del, nu) motors of Stitch's del-on-nu detector arm. fourc and twoc have one
        detector circle, their 2θ, which plays del; their nu is None, held at zero."""
        if self.geometry == 'psic':
            return self.motor('del'), self.motor('nu')
        return self.motor('tth'), None

    def arm_role(self, name) -> str | None:
        """The arm circle ('del' or 'nu', psic's names) a circle, motor or position name
        refers to; in fourc or twoc the 2θ circle is del."""
        circle = self.circle_for_position(name)
        role = AS_PSIC[self.geometry].get(circle, circle)
        return role if role in ('del', 'nu') else None

    def arm_motor(self, name) -> str | None:
        """The arm's motor (del's, or nu's) that a circle, motor or position name refers to."""
        return dict(zip(('del', 'nu'), self.stitch_arm())).get(self.arm_role(name))

    def arm_mismatch(self, positions, *, single_arm=False) -> str | None:
        """Why a calibration with these positions cannot drive this geometry's arm (None
        when it can). ``single_arm``: the calibration holds nu at zero (a 2θ arm)."""
        has_nu = any(InstrumentGeometry().circle_for_position(name) == 'nu' for name in positions)
        if self.stitch_arm()[1] is None and has_nu:
            return (f"This calibration is psic's del-on-nu arm; {self.geometry} has one detector circle "
                    "(2θ). Calibrate its 2θ arm, or choose psic under Calibrate ▸ 1 · Geometry.")
        if self.stitch_arm()[1] is not None and single_arm:
            return ("This calibration is a single 2θ arm (nu held at zero), but psic's nu moves. Choose "
                    "fourc or twoc under Calibrate ▸ 1 · Geometry, or calibrate psic's arm.")
        return None

    def circle_for_position(self, name) -> str | None:
        """The circle a calibration file's position (del_angle, nu_value, ...) belongs to;
        in fourc or twoc, psic's name is the circle that plays it (del_angle: tth)."""
        base = _POSITION_SUFFIX.sub('', str(name).strip()).casefold()
        for circle in self.circles:
            if base == circle or base in _ALIASES.get(circle, ()) or base == self.motor(circle).casefold():
                return circle
        for circle, psic in AS_PSIC[self.geometry].items():
            if base == psic or base in _ALIASES.get(psic, ()):
                return circle
        return None

    def diffractometer(self):
        """The core Diffractometer preset with these motor names. The fourc and twoc
        presets put 2θ on rot1 (unvalidated); Stitch never uses them (its arm is psic's)."""
        from xrd_tools.core.geometry import Diffractometer
        m = dict(self.motors)
        if self.geometry == 'psic':
            return Diffractometer.psic(del_=m['del'], nu=m['nu'], eta=m['eta'], chi=m['chi'], phi=m['phi'],
                                       mu=m['mu'])
        if self.geometry == 'fourc':
            return Diffractometer.fourc(tth=m['tth'], om=m['th'], chi=m['chi'], phi=m['phi'])
        return Diffractometer.two_circle(tth=m['tth'], th=m['th'])

    def to_record(self) -> dict:
        return {'geometry': self.geometry, 'motors': dict(self.motors)}

    @classmethod
    def from_record(cls, record) -> 'InstrumentGeometry':
        if not isinstance(record, dict) or not isinstance(record.get('motors', {}), dict):
            raise ValueError('an instrument geometry is {"geometry": ..., "motors": {circle: motor}}')
        return cls(str(record.get('geometry', 'psic')), tuple(record.get('motors', {}).items()))

    @classmethod
    def pick(cls, geometry, available, *, keep=None) -> 'InstrumentGeometry':
        """Each circle's motor from a scan's motors: an existing choice that the scan has,
        else the standard name or an alias, else (incidence) the GI preference, else the
        standard name."""
        names = [str(name) for name in available or ()]
        lower = {name.casefold(): name for name in names}
        kept = dict((keep.motors if keep is not None and keep.geometry == geometry else ()))
        chosen = []
        for circle, _role in GEOMETRIES[geometry]:
            if kept.get(circle, '').casefold() in lower:
                chosen.append((circle, lower[kept[circle].casefold()]))
                continue
            match = next((lower[alias] for alias in (circle, *_ALIASES.get(circle, ())) if alias in lower), None)
            if match is None and circle == INCIDENCE_CIRCLE[geometry]:  # the incidence rule Int uses
                match = next((lower[name] for name in GI_MOTOR_PREFERENCE if name in lower), None)
            chosen.append((circle, match or kept.get(circle) or circle))
        return cls(geometry, tuple(chosen))


__all__ = ['AS_PSIC', 'GEOMETRIES', 'GEOMETRY_LABELS', 'INCIDENCE_CIRCLE', 'InstrumentGeometry']
