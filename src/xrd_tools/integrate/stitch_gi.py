"""Grazing incidence for a native Stitch 1D: the plan's GI value and the GI coordinates.

A GI Stitch 1D writes I(q) and, beside it, one or both GI cuts (maintainer, 2026-09-29: the axis is
Q, 2θ, Qip, Qoop or Qip + Qoop): I(q_ip) and I(q_oop), q_ip and q_oop taken relative to the sample
surface. As Int's GI ranges, a Qip range and a Qoop range serve both cuts: each cut is binned over
its own coordinate's range and keeps the pixels whose other coordinate lies in the other range, so
Qip + Qoop gives exactly the two single-cut results.

Both backends take the sample surface normal n̂ from psic's four sample circles, per frame: eta from
the incidence motor (or a Manual angle), chi from the tilt motor (or a Manual χ; 90° is level at
17-2's chi-90 mounting), mu and phi from the Geometry's motors (maintainer, 2026-09-29: "pyfai
should get the tilt from the data as well just like the incidence angle"). With q the lab-frame
scattering vector:

- q_oop = q·n̂;
- q_ip = |q − (q·n̂) n̂|, signed along the pyFAI sample orientation's horizontal axis projected into
  the surface.

xrayutilities bins these directly. pyFAI's fiber model takes the incident angle and tilt that give the
same n̂ in its sample orientation's frame (q_sample = Rx(−t)·Ry(α)·q_lab, so n̂ = (−sin α cos t,
−sin t, cos α cos t) in (beam, horizontal, vertical)). Probe G1-tilt: per pixel ≤ 1.8e-9 Å⁻¹ for
orientations 3, 4, 6 and 7, off-nominal poses (chi 85–92.5°, mu 0.5°, phi −1.7°) included; the
level-sample results of the earlier rule (signed along n̂ × beam) are unchanged at orientations 4
and 7.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

#: The GI cuts, in their one order (the plan's, the kernels', the record's and the fingerprint's).
GI_CUTS = ("q_ip", "q_oop")
#: Each cut's axis unit (pyFAI's fiber unit names).
GI_UNITS = {"q_ip": "qip_A^-1", "q_oop": "qoop_A^-1"}
#: The Geometry's sample circles besides the incidence (eta) and the tilt (chi).
GI_SAMPLE_ROLES = ("mu", "phi")
#: psic's sample circles in the SURFACE lab frame (x up, y along the beam, z horizontal), outermost
#: first, as xrayutilities' ('x+', 'z-', 'y+', 'z-'); the sample-frame surface normal is +z.
_PSIC_SAMPLE_AXES = ((0, 1), (2, -1), (1, 1), (2, -1))
_BEAM = np.array((0.0, 1.0, 0.0))
#: Each pyFAI sample orientation's vertical axis (the level surface normal) and horizontal axis in
#: that frame, measured from pyFAI's own fiber coordinates at zero incidence and tilt (probe G1-tilt).
_ORIENTATION_AXES = {
    1: ((-1, 0, 0), (0, 0, 1)), 2: ((-1, 0, 0), (0, 0, -1)), 3: ((1, 0, 0), (0, 0, -1)),
    4: ((1, 0, 0), (0, 0, 1)), 5: ((0, 0, -1), (1, 0, 0)), 6: ((0, 0, 1), (1, 0, 0)),
    7: ((0, 0, 1), (-1, 0, 0)), 8: ((0, 0, -1), (-1, 0, 0)),
}
#: The largest incidence or tilt (degrees) a sample orientation is taken to describe.
_MAX_ORIENTATION_ANGLE = 45.0


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"GI {name} must be a finite number")
    return float(value)


def _window(value, name):
    if value is None:
        return None
    if (type(value) not in (tuple, list) or len(value) != 2):
        raise ValueError(f"GI {name} must be (min, max) or None")
    low, high = (_finite(v, name) for v in value)
    if not low < high:
        raise ValueError(f"GI {name} needs min < max")
    return (low, high)


def _rotation(axis: int, sign: int, degrees: float) -> np.ndarray:
    """The right-handed rotation by ``sign * degrees`` about lab axis ``axis``."""
    angle = math.radians(sign * degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    i, j = ((1, 2), (2, 0), (0, 1))[axis]
    matrix = np.eye(3)
    matrix[i, i] = matrix[j, j] = cos
    matrix[i, j], matrix[j, i] = -sin, sin
    return matrix


def psic_normal(mu: float, eta: float, chi: float, phi: float) -> np.ndarray:
    """The sample surface normal in the lab for psic's four sample circles (degrees), as
    xrayutilities' ``transformSample2Lab`` gives it for the SURFACE asset."""
    normal = np.array((0.0, 0.0, 1.0))
    for (axis, sign), angle in reversed(tuple(zip(_PSIC_SAMPLE_AXES, (mu, eta, chi, phi)))):
        normal = _rotation(axis, sign, angle) @ normal
    return normal / np.linalg.norm(normal)


@dataclass(frozen=True, slots=True)
class StitchGI:
    """The GI part of a native Stitch 1D plan.

    ``cuts`` are the GI output axes ('q_ip', 'q_oop' or both, kept in ``GI_CUTS`` order), binned
    with the plan's 1-D points. ``ip_range`` and ``oop_range`` are Int's two GI ranges: a cut is
    binned over its own coordinate's range (None: the data's span) and keeps the pixels whose other
    coordinate lies in the other range (None: all). The sample normal comes from psic's circles per
    frame: eta is ``incidence_motor``'s value or ``incident_angle_deg`` (Manual), chi is
    ``tilt_motor``'s value or ``tilt_chi_deg`` (Manual, 90° by default: level at a chi-90 mounting),
    and mu and phi are ``sample_motors``' (None reads zero). ``sample_orientation`` is pyFAI's (1–8):
    the frame pyFAI's incidence and tilt are taken in, and the q_ip sign of both backends.
    """

    cuts: tuple[str, ...]
    incidence_motor: str | None = None
    incident_angle_deg: float | None = None
    tilt_motor: str | None = None
    tilt_chi_deg: float | None = None
    sample_orientation: int = 4
    sample_motors: tuple[tuple[str, str | None], ...] = tuple((role, None) for role in GI_SAMPLE_ROLES)
    ip_range: tuple[float, float] | None = None
    oop_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        cuts = self.cuts
        if (type(cuts) not in (tuple, list) or not cuts or any(cut not in GI_CUTS for cut in cuts)
                or len(set(cuts)) != len(cuts)):
            raise ValueError(f"GI cuts must be one or both of {GI_CUTS}")
        object.__setattr__(self, "cuts", tuple(cut for cut in GI_CUTS if cut in cuts))
        for motor, value, name in ((self.incidence_motor, "incident_angle_deg", "incidence"),
                                   (self.tilt_motor, "tilt_chi_deg", "tilt")):
            manual = getattr(self, value)
            if motor is None and manual is None and name == "tilt":
                object.__setattr__(self, value, manual := 90.0)  # level at a chi-90 mounting
            if (manual is not None) == (motor is not None):
                raise ValueError(f"GI needs exactly one of a {name} motor or a Manual {name} angle")
            if manual is not None:
                object.__setattr__(self, value, _finite(manual, f"{name} angle"))
            elif type(motor) is not str or not motor.strip():
                raise ValueError(f"GI {name} motor must be a non-empty name")
        if type(self.sample_orientation) is not int or self.sample_orientation not in _ORIENTATION_AXES:
            raise ValueError("GI sample orientation must be pyFAI's 1 to 8")
        motors = tuple(tuple(pair) for pair in self.sample_motors)
        if (tuple(role for role, _motor in motors) != GI_SAMPLE_ROLES
                or any(motor is not None and (type(motor) is not str or not motor.strip())
                       for _role, motor in motors)):
            raise ValueError(f"GI sample motors must name {GI_SAMPLE_ROLES} in order (None reads zero)")
        object.__setattr__(self, "sample_motors", motors)
        object.__setattr__(self, "ip_range", _window(self.ip_range, "Qip range"))
        object.__setattr__(self, "oop_range", _window(self.oop_range, "Qoop range"))

    def bins(self, cut: str) -> tuple[float, float] | None:
        """The range ``cut`` is binned over (None: the data's span)."""
        return self.ip_range if cut == "q_ip" else self.oop_range

    def window(self, cut: str) -> tuple[float, float] | None:
        """The range of the other coordinate that ``cut`` keeps (None: every pixel)."""
        return self.oop_range if cut == "q_ip" else self.ip_range

    def metadata_names(self) -> tuple[str, ...]:
        """The metadata columns every frame needs, for both backends: the incidence and tilt motors
        and the Geometry's other sample circles."""
        names = (self.incidence_motor, self.tilt_motor, *(motor for _role, motor in self.sample_motors))
        return tuple(dict.fromkeys(name for name in names if name is not None))

    def identity(self) -> tuple[object, ...]:
        """Canonical value for a plan fingerprint."""
        return ("stitch-gi-v3", self.cuts, self.incidence_motor, self.incident_angle_deg, self.tilt_motor,
                self.tilt_chi_deg, self.sample_orientation, self.sample_motors, self.ip_range, self.oop_range)

    def to_provenance(self) -> dict[str, object]:
        return {"cuts": list(self.cuts), "units": {cut: GI_UNITS[cut] for cut in self.cuts},
                "incidence_motor": self.incidence_motor, "incident_angle_deg": self.incident_angle_deg,
                "tilt_motor": self.tilt_motor, "tilt_chi_deg": self.tilt_chi_deg,
                "sample_orientation": self.sample_orientation,
                "sample_motors": {role: motor for role, motor in self.sample_motors},
                "ip_range": None if self.ip_range is None else list(self.ip_range),
                "oop_range": None if self.oop_range is None else list(self.oop_range)}

    def circles(self, metadata) -> tuple[float, float, float, float]:
        """This frame's psic sample circles (mu, eta, chi, phi) in degrees."""
        motors = dict(self.sample_motors)
        values = tuple(
            manual if motor is None and manual is not None else 0.0 if motor is None else float(metadata[motor])
            for motor, manual in ((motors["mu"], None), (self.incidence_motor, self.incident_angle_deg),
                                  (self.tilt_motor, self.tilt_chi_deg), (motors["phi"], None)))
        return tuple(_finite(value, "sample angle") for value in values)

    def normal(self, metadata) -> np.ndarray:
        """This frame's sample surface normal in the lab."""
        return psic_normal(*self.circles(metadata))

    def pyfai_angles(self, normal) -> tuple[float, float]:
        """pyFAI's (incident angle, tilt) in degrees giving ``normal`` in this sample orientation."""
        vertical, horizontal = (np.asarray(axis, dtype=np.float64) for axis in _ORIENTATION_AXES[self.sample_orientation])
        normal = np.asarray(normal, dtype=np.float64)
        incidence = math.degrees(math.atan2(-float(normal @ _BEAM), float(normal @ vertical)))
        tilt = math.degrees(-math.asin(max(-1.0, min(1.0, float(normal @ horizontal)))))
        if abs(incidence) >= _MAX_ORIENTATION_ANGLE or abs(tilt) >= _MAX_ORIENTATION_ANGLE:
            raise ValueError(f"the sample normal is {incidence:.1f}° / {tilt:.1f}° from pyFAI sample orientation "
                             f"{self.sample_orientation}'s level surface: choose the orientation of this mounting")
        return incidence, tilt

    def in_plane_axis(self, normal) -> np.ndarray:
        """The q_ip sign axis: this orientation's horizontal axis projected into the surface."""
        normal = np.asarray(normal, dtype=np.float64)
        horizontal = np.asarray(_ORIENTATION_AXES[self.sample_orientation][1], dtype=np.float64)
        axis = horizontal - float(horizontal @ normal) * normal
        length = float(np.linalg.norm(axis))
        if not length > 1e-9:
            raise ValueError("the sample normal lies along the orientation's horizontal axis")
        return axis / length


def gi_coordinates(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, normal, axis) -> tuple[np.ndarray, np.ndarray]:
    """(q_ip, q_oop) of lab-frame q components for the unit surface normal ``normal`` and the in-plane
    sign axis ``axis`` (``StitchGI.in_plane_axis``): q_oop = q·n̂; q_ip the in-plane magnitude, signed
    along ``axis``."""
    normal = np.asarray(normal, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    q_oop = qx * normal[0] + qy * normal[1] + qz * normal[2]
    ix, iy, iz = qx - normal[0] * q_oop, qy - normal[1] * q_oop, qz - normal[2] * q_oop
    q_ip = np.sqrt(ix * ix + iy * iy + iz * iz)
    q_ip *= np.sign(ix * axis[0] + iy * axis[1] + iz * axis[2])
    return q_ip, q_oop


__all__ = ["GI_CUTS", "GI_SAMPLE_ROLES", "GI_UNITS", "StitchGI", "gi_coordinates", "psic_normal"]
