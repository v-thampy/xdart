"""Low-level geometry configuration primitives.

Two unrelated geometry containers live here:

* :class:`DiffractometerConfig` — used to build ``xrayutilities`` HXRD
  objects for reciprocal-space mapping.  Imports ``xrayutilities``
  lazily inside :meth:`make_hxrd`.

* :class:`DiffractometerGeometry` — maps raw scan-file motor columns
  to per-frame pyFAI rotations (``rot1``/``rot2``/``rot3``) and the
  GI incidence angle.  Used by the v2 NeXus writer/reader (xdart 0.37+)
  to support flexible diffractometer conventions (2-circle, psic,
  psic-halpha, custom).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np

if TYPE_CHECKING:  # Only needed for type checkers; avoided at runtime.
    import xrayutilities as xu


@dataclass(slots=True)
class DiffractometerConfig:
    """Geometry configuration for ``xu.QConversion`` and ``xu.HXRD``."""

    sample_rot: tuple[str, ...] = ("z-", "y+", "z-")
    detector_rot: tuple[str, ...] = ("z-",)
    r_i: tuple[float, float, float] = (0.0, 1.0, 0.0)

    q_conv_kwargs: dict[str, Any] = field(default_factory=dict)

    hxrd_n: tuple[float, float, float] = (0.0, 1.0, 0.0)
    hxrd_q: tuple[float, float, float] = (1.0, 0.0, 0.0)
    hxrd_geometry: str = "real"
    hxrd_kwargs: dict[str, Any] = field(default_factory=dict)

    init_area_detrot: str = "z-"
    init_area_tiltazimuth: str = "x+"
    ang2q_kwargs: dict[str, Any] = field(default_factory=dict)

    def make_hxrd(self, energy: float) -> "xu.HXRD":
        """Build an ``xu.HXRD`` instance at the given energy.

        ``xrayutilities`` is imported lazily inside this method so that
        importing :mod:`ssrl_xrd_tools.core` never triggers the xu
        import (it is still a declared dependency and must be
        installed before calling this method).
        """
        import xrayutilities as xu  # noqa: PLC0415 — intentional lazy import

        qconversion = xu.QConversion(
            self.sample_rot,
            self.detector_rot,
            self.r_i,
            **self.q_conv_kwargs,
        )
        return xu.HXRD(
            self.hxrd_n,
            self.hxrd_q,
            geometry=self.hxrd_geometry,
            en=energy,
            qconv=qconversion,
            **self.hxrd_kwargs,
        )


# ---------------------------------------------------------------------------
# Flexible diffractometer geometry (xdart v2 NeXus writer)
# ---------------------------------------------------------------------------

Convention = Literal["two_circle", "psic", "psic_halpha", "custom"]


@dataclass(frozen=True)
class AngleMapping:
    """Linear mapping from one raw motor column to a derived angle.

    ``derived = sign * motor_value + offset``, all in the same units
    (degrees).  The consumer (``derive_per_frame``) handles
    deg→rad conversion for ``rot1/rot2/rot3``; the incidence angle
    stays in degrees.
    """

    source_motor: str = ""
    sign: float = 1.0
    offset: float = 0.0

    @property
    def is_active(self) -> bool:
        return bool(self.source_motor)

    def apply(self, motor_values: "np.ndarray | float") -> np.ndarray:
        arr = np.atleast_1d(np.asarray(motor_values, dtype=float))
        if not self.is_active:
            return np.zeros_like(arr)
        return self.sign * arr + self.offset


@dataclass(frozen=True)
class DiffractometerGeometry:
    """Maps scan-file motor columns to pyFAI rotations + GI incidence angle.

    Conventions:

    * ``two_circle``: ``rot1 ← tth``, ``incidence ← th`` (rot2/rot3 inactive).
    * ``psic``: ``rot1 ← nu``, ``rot2 ← del``, ``incidence ← eta``.
    * ``psic_halpha``: same as psic but ``incidence ← halpha``.
    * ``custom``: user-supplied mappings.

    ``sample_motors`` / ``detector_motors`` list ALL motors to persist
    verbatim into the NeXus output, not just the ones consumed by the
    rotation/incidence mappings.
    """

    convention: Convention = "two_circle"
    rot1: AngleMapping = field(default_factory=AngleMapping)
    rot2: AngleMapping = field(default_factory=AngleMapping)
    rot3: AngleMapping = field(default_factory=AngleMapping)
    incident_angle: AngleMapping = field(default_factory=AngleMapping)
    sample_motors: tuple[str, ...] = ()
    detector_motors: tuple[str, ...] = ()

    @classmethod
    def two_circle(cls, tth: str = "tth", th: str = "th",
                   gonchi: str | None = None) -> "DiffractometerGeometry":
        sample = (th,) if gonchi is None else (th, gonchi)
        return cls(
            convention="two_circle",
            rot1=AngleMapping(source_motor=tth),
            incident_angle=AngleMapping(source_motor=th),
            sample_motors=sample,
            detector_motors=(tth,),
        )

    @classmethod
    def psic(cls, del_: str = "del", nu: str = "nu", eta: str = "eta",
             chi: str = "chi", phi: str = "phi",
             mu: str = "mu") -> "DiffractometerGeometry":
        return cls(
            convention="psic",
            rot1=AngleMapping(source_motor=nu),
            rot2=AngleMapping(source_motor=del_),
            incident_angle=AngleMapping(source_motor=eta),
            sample_motors=(eta, chi, phi, mu),
            detector_motors=(del_, nu),
        )

    @classmethod
    def psic_halpha(cls, del_: str = "del", nu: str = "nu",
                    halpha: str = "halpha", chi: str = "chi",
                    phi: str = "phi",
                    mu: str = "mu") -> "DiffractometerGeometry":
        return cls(
            convention="psic_halpha",
            rot1=AngleMapping(source_motor=nu),
            rot2=AngleMapping(source_motor=del_),
            incident_angle=AngleMapping(source_motor=halpha),
            sample_motors=(halpha, chi, phi, mu),
            detector_motors=(del_, nu),
        )

    def derive_per_frame(self, motors: Mapping[str, "np.ndarray"]
                         ) -> dict[str, np.ndarray]:
        """Return ``{rot1, rot2, rot3, incident_angle}`` per-frame arrays.

        ``rot1/rot2/rot3`` come back in **radians**; ``incident_angle``
        in **degrees**.  Inactive mappings produce zero arrays of length
        ``N`` inferred from the active source motors.
        """
        active_arrays: list[np.ndarray] = []
        for mapping in (self.rot1, self.rot2, self.rot3, self.incident_angle):
            if mapping.is_active:
                if mapping.source_motor not in motors:
                    raise KeyError(
                        f"Active source motor {mapping.source_motor!r} not in "
                        f"motors dict (keys: {list(motors)})"
                    )
                active_arrays.append(
                    np.atleast_1d(np.asarray(motors[mapping.source_motor],
                                             dtype=float))
                )
        if not active_arrays:
            N = 1
        else:
            lengths = {a.shape[0] for a in active_arrays}
            if len(lengths) > 1:
                raise ValueError(
                    f"Active motor arrays have inconsistent lengths: {lengths}"
                )
            (N,) = lengths

        out: dict[str, np.ndarray] = {}
        for key in ("rot1", "rot2", "rot3"):
            mapping: AngleMapping = getattr(self, key)
            if mapping.is_active:
                out[key] = np.deg2rad(mapping.apply(motors[mapping.source_motor]))
            else:
                out[key] = np.zeros(N, dtype=float)
        inc = self.incident_angle
        if inc.is_active:
            out["incident_angle"] = inc.apply(motors[inc.source_motor])
        else:
            out["incident_angle"] = np.zeros(N, dtype=float)
        return out

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "DiffractometerGeometry":
        d = json.loads(s)
        return cls(
            convention=d.get("convention", "two_circle"),
            rot1=AngleMapping(**d.get("rot1", {})),
            rot2=AngleMapping(**d.get("rot2", {})),
            rot3=AngleMapping(**d.get("rot3", {})),
            incident_angle=AngleMapping(**d.get("incident_angle", {})),
            sample_motors=tuple(d.get("sample_motors", ())),
            detector_motors=tuple(d.get("detector_motors", ())),
        )

    def all_referenced_motors(self) -> tuple[str, ...]:
        """All motor columns this geometry needs from the meta file."""
        seen: dict[str, None] = {}
        for mapping in (self.rot1, self.rot2, self.rot3, self.incident_angle):
            if mapping.is_active and mapping.source_motor not in seen:
                seen[mapping.source_motor] = None
        for m in self.sample_motors + self.detector_motors:
            if m and m not in seen:
                seen[m] = None
        return tuple(seen.keys())


__all__ = [
    "AngleMapping",
    "Convention",
    "DiffractometerConfig",
    "DiffractometerGeometry",
]
