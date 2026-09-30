"""The diffractometer an operation runs on: a standard circle geometry and its motor names.

One small value, shared by every page that needs to know which scan motor drives which
circle (Stitch and its calibration now, RSM next). The geometry names the circles; the
motors are whatever the beamline calls them, defaulting to the standard names. The
existing :class:`Diffractometer` presets carry the circle conventions.

fourc and twoc are psic's special cases. In the vertical scattering plane (psic with
mu = nu = 0 is the four-circle) their 2θ circle plays psic's del and their θ plays eta;
in the horizontal plane 2θ plays nu and θ plays mu. Stitch's detector arm is psic's
with the other detector circle held at zero. (For RSM, the sample side of the
horizontal mapping is not validated yet.)

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
#: The scattering plane of a fourc or twoc 2θ arm (psic moves both detector circles).
PLANES = ('vertical', 'horizontal')
PLANE_LABELS = {'vertical': 'Vertical (2θ plays del)', 'horizontal': 'Horizontal (2θ plays nu)'}
#: The psic circle each renamed circle of a fourc or twoc plays, by scattering plane
#: (chi and phi keep their names).
_AS_PSIC = {'vertical': {'th': 'eta', 'tth': 'del'}, 'horizontal': {'th': 'mu', 'tth': 'nu'}}
#: The scattering plane a 2θ arm moving one psic circle lies in.
_PLANE_OF = {'del': 'vertical', 'nu': 'horizontal'}
#: The lab frame every geometry is described in (psic's, You 1999).
LAB_FRAME = 'x up, y along the beam, z = x × y (left, looking downstream)'
#: psic's rotation axes (a positive angle turns right-handed about it); the same lists as
#: ``Diffractometer.psic`` and the RSM notebooks: sample (mu, eta, chi, phi), detector (nu, del).
PSIC_AXES = {'mu': 'x+', 'eta': 'z-', 'chi': 'y+', 'phi': 'z-', 'nu': 'x+', 'del': 'z-'}
#: Other names beamlines give a circle's motor (matched whole, ignoring case).
_ALIASES = {'del': ('delta',), 'tth': ('2theta', 'twotheta', 'two_theta', 'tt'),
            'th': ('theta', 'om', 'omega')}
#: Suffixes a calibration file's position names carry (del_angle, nu_value, ...).
_POSITION_SUFFIX = re.compile(r'[_\-. ]?(angle|value|pos|position|deg)$', re.IGNORECASE)


def calibration_arm(record, source_motors) -> tuple[frozenset, frozenset]:
    """The detector-arm circles ('del', 'nu') a pyFAI goniometer calibration moves, and those
    it holds at zero, read from its transformation as bound to the scan's motors — never
    from how its positions are spelled. ValueError when they cannot be established."""
    from xrd_tools.core.geometry import Diffractometer
    try:
        diff = Diffractometer.from_pyfai_goniometer(dict(record), source_motors=dict(source_motors))
    except (KeyError, TypeError, ValueError, NotImplementedError, SyntaxError) as error:
        raise ValueError(f'its transformation cannot be read ({error})') from error
    arm = diff.stacked_arm
    if arm is not None:
        moving = frozenset(circle for circle, motor in (('del', arm.del_motor), ('nu', arm.nu_motor)) if motor)
        return moving, frozenset({'del', 'nu'}) - moving
    if diff.rot3.is_active:
        raise ValueError('a position drives the beam-axis rotation, which no psic detector circle does')
    # independent Euler mappings, as psic's: rot1 follows nu, rot2 follows del
    return frozenset(circle for circle, mapping in (('del', diff.rot2), ('nu', diff.rot1))
                     if mapping.is_active), frozenset()


@dataclass(frozen=True)
class InstrumentGeometry:
    geometry: str = 'psic'
    motors: tuple[tuple[str, str], ...] = ()
    #: fourc and twoc: the 2θ arm's scattering plane (psic moves both detector circles)
    plane: str = 'vertical'

    def __post_init__(self):
        if self.geometry not in GEOMETRIES:
            raise ValueError(f'unknown diffractometer geometry {self.geometry!r}')
        if self.plane not in PLANES:
            raise ValueError(f'unknown scattering plane {self.plane!r}')
        if self.geometry == 'psic' and self.plane != 'vertical':
            raise ValueError('psic moves both detector circles; the scattering plane is for fourc and twoc')
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

    def as_psic(self) -> dict[str, str]:
        """The psic circle each renamed circle of this geometry plays (none for psic)."""
        return {} if self.geometry == 'psic' else dict(_AS_PSIC[self.plane])

    def circle_axes(self) -> tuple[tuple[str, str], ...]:
        """Each circle's rotation axis in the lab frame (``LAB_FRAME``): the axis of the psic
        circle it plays, as xrayutilities writes it."""
        played = self.as_psic()
        return tuple((circle, PSIC_AXES[played.get(circle, circle)]) for circle in self.circles)

    def stitch_arm(self) -> tuple[str | None, str | None]:
        """The (del, nu) motors of Stitch's del-on-nu detector arm. fourc and twoc have one
        detector circle, their 2θ: it plays del in the vertical plane and nu in the
        horizontal one, and the other circle is None, held at zero."""
        if self.geometry == 'psic':
            return self.motor('del'), self.motor('nu')
        return (self.motor('tth'), None) if self.plane == 'vertical' else (None, self.motor('tth'))

    def arm_role(self, name) -> str | None:
        """The arm circle ('del' or 'nu', psic's names) a circle, motor or position name
        refers to; in fourc or twoc the 2θ circle is del or nu by the scattering plane."""
        circle = self.circle_for_position(name)
        role = self.as_psic().get(circle, circle)
        return role if role in ('del', 'nu') else None

    def arm_motor(self, name) -> str | None:
        """The arm's motor (del's, or nu's) that a circle, motor or position name refers to."""
        return dict(zip(('del', 'nu'), self.stitch_arm())).get(self.arm_role(name))

    def arm_mismatch(self, moving, holds=()) -> str | None:
        """Why a calibration that moves these arm circles ('del', 'nu') and holds these at
        zero cannot drive this geometry's arm (None when it can). Take the circles from the
        calibration's transformation (:func:`calibration_arm`), never from position names."""
        moving, holds = frozenset(moving), frozenset(holds)
        mine = frozenset(circle for circle, motor in zip(('del', 'nu'), self.stitch_arm()) if motor is not None)
        if moving - mine:
            if moving == {'del', 'nu'}:
                return (f"This calibration is psic's del-on-nu arm; {self.geometry} has one detector circle "
                        f"(a {self.plane} 2θ). Calibrate its 2θ arm, or choose psic under Calibrate ▸ "
                        "1 · Geometry.")
            other = next(iter(moving - mine))
            return (f"This calibration's arm moves psic's {other} (a {_PLANE_OF[other]} 2θ); {self.geometry}'s "
                    f"2θ arm is {self.plane} here. Choose the {_PLANE_OF[other]} scattering plane under "
                    "Calibrate ▸ 1 · Geometry, or calibrate this arm.")
        if holds & mine:
            held = next(iter(holds & mine))
            plane = _PLANE_OF[next(iter(moving))] if len(moving) == 1 else 'matching'
            return (f"This calibration is a single 2θ arm ({held} held at zero), but {self.geometry} moves "
                    f"{held}. Choose fourc or twoc (the {plane} scattering plane) under Calibrate ▸ "
                    "1 · Geometry, or calibrate psic's arm.")
        return None

    def circle_for_position(self, name) -> str | None:
        """The circle a calibration file's position (del_angle, nu_value, ...) belongs to;
        in fourc or twoc, psic's name is the circle that plays it (vertical del_angle: tth)."""
        base = _POSITION_SUFFIX.sub('', str(name).strip()).casefold()
        for circle in self.circles:
            if base == circle or base in _ALIASES.get(circle, ()) or base == self.motor(circle).casefold():
                return circle
        for circle, psic in self.as_psic().items():
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
        record = {'geometry': self.geometry, 'motors': dict(self.motors)}
        if self.geometry != 'psic':  # older records have no plane: vertical
            record['plane'] = self.plane
        return record

    @classmethod
    def from_record(cls, record) -> 'InstrumentGeometry':
        if not isinstance(record, dict) or not isinstance(record.get('motors', {}), dict):
            raise ValueError('an instrument geometry is {"geometry": ..., "motors": {circle: motor}}')
        return cls(str(record.get('geometry', 'psic')), tuple(record.get('motors', {}).items()),
                   str(record.get('plane', 'vertical')))

    @classmethod
    def pick(cls, geometry, available, *, keep=None, plane=None) -> 'InstrumentGeometry':
        """Each circle's motor from a scan's motors: an existing choice that the scan has,
        else the standard name or an alias (in fourc or twoc also psic's names for the circle
        it plays: vertical tth ← del, horizontal tth ← nu), else (incidence) the GI
        preference, else the standard name. The plane is kept unless given."""
        names = [str(name) for name in available or ()]
        lower = {name.casefold(): name for name in names}
        same = keep is not None and keep.geometry == geometry
        kept = dict(keep.motors if same else ())
        plane = plane or (keep.plane if same else 'vertical')
        as_psic = {} if geometry == 'psic' else _AS_PSIC[plane]
        chosen = []
        for circle, _role in GEOMETRIES[geometry]:
            if kept.get(circle, '').casefold() in lower:
                chosen.append((circle, lower[kept[circle].casefold()]))
                continue
            psic = as_psic.get(circle)
            aliases = (circle, *_ALIASES.get(circle, ()),
                       *((psic, *_ALIASES.get(psic, ())) if psic and psic != circle else ()))
            match = next((lower[alias] for alias in aliases if alias in lower), None)
            if match is None and circle == INCIDENCE_CIRCLE[geometry]:  # the incidence rule Int uses
                match = next((lower[name] for name in GI_MOTOR_PREFERENCE if name in lower), None)
            chosen.append((circle, match or kept.get(circle) or circle))
        return cls(geometry, tuple(chosen), plane)


__all__ = ['GEOMETRIES', 'GEOMETRY_LABELS', 'INCIDENCE_CIRCLE', 'InstrumentGeometry', 'PLANES',
           'PLANE_LABELS', 'calibration_arm']
