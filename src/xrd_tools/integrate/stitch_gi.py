"""Grazing incidence for a native Stitch 1D: the plan's GI value and the GI coordinates.

A GI Stitch 1D writes I(q) and, beside it, a GI cut (maintainer, 2026-09-30): I(q_ip) inside a q_oop
window, or I(q_oop) inside a q_ip window, q_ip and q_oop taken relative to the sample surface.

The two backends give the same coordinates (probe G1, ≤ 2e-9 Å⁻¹ per pixel, arms off nu = 0 too).
With n̂ the sample normal in the lab and q the lab-frame scattering vector:

- q_oop = q·n̂;
- q_ip = |q − (q·n̂) n̂|, signed along normalize(n̂ × beam).

pyFAI's fiber model reproduces this with incident angle α (the incidence motor, or a Manual angle),
tilt 0 and sample orientation 4 for psic mounted at chi 90 with eta the incidence (17-2), or 7 at
chi 0 with mu the incidence. xrayutilities takes n̂ from its own sample transform of the SURFACE
asset's surface normal, with every sample circle's angle: mu, chi and phi from their motors, and
eta from the incidence (maintainer, 2026-09-30).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

#: The GI cut: its output axis, and the other coordinate's window.
GI_CUTS = ("q_ip", "q_oop")
#: xrayutilities' sample circles other than the incidence (psic's eta), in the SURFACE order.
GI_SAMPLE_ROLES = ("mu", "chi", "phi")
_MAX_POINTS = 1_000_000


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


@dataclass(frozen=True, slots=True)
class StitchGI:
    """The GI part of a native Stitch 1D plan.

    ``cut`` is the GI output axis ('q_ip' or 'q_oop'), binned with ``npt`` points over ``cut_range``
    (None: the data's span), keeping pixels whose other coordinate lies in ``window`` (None: all).
    The incidence is ``incidence_motor``'s value per frame, or ``incident_angle_deg`` (Manual).
    ``tilt_deg`` and ``sample_orientation`` are pyFAI's fiber parameters. ``sample_motors`` names
    the motors of xrayutilities' other sample circles (mu, chi, phi); None reads zero.
    """

    cut: str
    incidence_motor: str | None = None
    incident_angle_deg: float | None = None
    tilt_deg: float = 0.0
    sample_orientation: int = 4
    sample_motors: tuple[tuple[str, str | None], ...] = tuple((role, None) for role in GI_SAMPLE_ROLES)
    npt: int = 1000
    cut_range: tuple[float, float] | None = None
    window: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.cut not in GI_CUTS:
            raise ValueError(f"GI cut must be one of {GI_CUTS}")
        manual = self.incident_angle_deg is not None
        if manual == (self.incidence_motor is not None):
            raise ValueError("GI needs exactly one of an incidence motor or a Manual incident angle")
        if manual:
            object.__setattr__(self, "incident_angle_deg", _finite(self.incident_angle_deg, "incident angle"))
        elif type(self.incidence_motor) is not str or not self.incidence_motor.strip():
            raise ValueError("GI incidence motor must be a non-empty name")
        object.__setattr__(self, "tilt_deg", _finite(self.tilt_deg, "tilt"))
        if type(self.sample_orientation) is not int or not 1 <= self.sample_orientation <= 8:
            raise ValueError("GI sample orientation must be pyFAI's 1 to 8")
        motors = tuple(tuple(pair) for pair in self.sample_motors)
        if (tuple(role for role, _motor in motors) != GI_SAMPLE_ROLES
                or any(motor is not None and (type(motor) is not str or not motor.strip())
                       for _role, motor in motors)):
            raise ValueError(f"GI sample motors must name {GI_SAMPLE_ROLES} in order (None reads zero)")
        object.__setattr__(self, "sample_motors", motors)
        if type(self.npt) is not int or not 1 <= self.npt <= _MAX_POINTS:
            raise ValueError("GI points must be a bounded positive exact integer")
        object.__setattr__(self, "cut_range", _window(self.cut_range, "cut range"))
        object.__setattr__(self, "window", _window(self.window, "window"))

    @property
    def unit(self) -> str:
        """The cut's axis unit (pyFAI's fiber unit names)."""
        return "qip_A^-1" if self.cut == "q_ip" else "qoop_A^-1"

    def metadata_names(self, backend: str) -> tuple[str, ...]:
        """The metadata columns every frame needs: the incidence motor and, for xrayutilities,
        the other sample circles' motors."""
        names = () if self.incidence_motor is None else (self.incidence_motor,)
        if backend == "xu_hist":
            names += tuple(motor for _role, motor in self.sample_motors if motor is not None)
        return names

    def identity(self) -> tuple[object, ...]:
        """Canonical value for a plan fingerprint."""
        return ("stitch-gi-v1", self.cut, self.incidence_motor, self.incident_angle_deg, self.tilt_deg,
                self.sample_orientation, self.sample_motors, self.npt, self.cut_range, self.window)

    def to_provenance(self) -> dict[str, object]:
        return {"cut": self.cut, "unit": self.unit, "incidence_motor": self.incidence_motor,
                "incident_angle_deg": self.incident_angle_deg, "tilt_deg": self.tilt_deg,
                "sample_orientation": self.sample_orientation,
                "sample_motors": {role: motor for role, motor in self.sample_motors},
                "npt": self.npt, "cut_range": None if self.cut_range is None else list(self.cut_range),
                "window": None if self.window is None else list(self.window)}

    def incidence(self, metadata) -> float:
        """This frame's incident angle in degrees."""
        if self.incident_angle_deg is not None:
            return self.incident_angle_deg
        return _finite(float(metadata[self.incidence_motor]), "incident angle")


def gi_coordinates(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, normal, beam) -> tuple[np.ndarray, np.ndarray]:
    """(q_ip, q_oop) of lab-frame q components for the unit surface normal ``normal`` and the incident
    beam direction ``beam`` (lab frame): q_oop = q·n̂; q_ip the in-plane magnitude, signed along
    normalize(n̂ × beam)."""
    normal = np.asarray(normal, dtype=np.float64)
    axis = np.cross(normal, np.asarray(beam, dtype=np.float64))
    length = float(np.linalg.norm(axis))
    if not length > 1e-9:
        raise ValueError("the sample normal lies along the beam: no grazing geometry")
    axis /= length
    q_oop = qx * normal[0] + qy * normal[1] + qz * normal[2]
    ix, iy, iz = qx - normal[0] * q_oop, qy - normal[1] * q_oop, qz - normal[2] * q_oop
    q_ip = np.sqrt(ix * ix + iy * iy + iz * iz)
    q_ip *= np.sign(ix * axis[0] + iy * axis[1] + iz * axis[2])
    return q_ip, q_oop


def xu_sample_normal(qconversion, mu: float, eta: float, chi: float, phi: float,
                     surface_normal=(0.0, 0.0, 1.0)) -> np.ndarray:
    """The sample surface normal in the lab, by xrayutilities' own sample transform."""
    normal = np.asarray(qconversion.transformSample2Lab(np.asarray(surface_normal, dtype=np.float64),
                                                        mu, eta, chi, phi), dtype=np.float64)
    return normal / np.linalg.norm(normal)


__all__ = ["GI_CUTS", "GI_SAMPLE_ROLES", "StitchGI", "gi_coordinates", "xu_sample_normal"]
