"""The diffractometer an operation runs on: a standard circle geometry and its motor names.

One small value, shared by every page that needs to know which scan motor drives which
circle (Stitch and its calibration now, RSM next). The geometry names the circles; the
motors are whatever the beamline calls them, defaulting to the standard names. The
existing :class:`Diffractometer` presets carry the circle conventions.

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
#: Other names beamlines give a circle's motor (matched whole, ignoring case).
_ALIASES = {'del': ('delta',), 'tth': ('2theta', 'twotheta', 'tt'), 'th': ('theta', 'om', 'omega')}
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

    def stitch_arm(self) -> tuple[str, str]:
        """The (del, nu) motors of Stitch's del-on-nu detector arm."""
        if self.geometry != 'psic':
            raise ValueError("Stitch's detector arm needs nu and del (psic); a single 2θ arm "
                             f"({self.geometry}) is not supported yet")
        return self.motor('del'), self.motor('nu')

    def circle_for_position(self, name) -> str | None:
        """The circle a calibration file's position (del_angle, nu_value, ...) belongs to."""
        base = _POSITION_SUFFIX.sub('', str(name).strip()).casefold()
        for circle in self.circles:
            if base == circle or base in _ALIASES.get(circle, ()) or base == self.motor(circle).casefold():
                return circle
        return None

    def diffractometer(self):
        """The core Diffractometer preset with these motor names."""
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


__all__ = ['GEOMETRIES', 'GEOMETRY_LABELS', 'INCIDENCE_CIRCLE', 'InstrumentGeometry']
