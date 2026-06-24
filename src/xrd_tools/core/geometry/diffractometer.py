"""Low-level diffractometer geometry primitives.

Two unrelated geometry containers live here:

* :class:`DiffractometerConfig` — used to build ``xrayutilities`` HXRD
  objects for reciprocal-space mapping.  Imports ``xrayutilities`` lazily
  inside :meth:`make_hxrd`.

* :class:`DiffractometerGeometry` — maps raw scan-file motor columns to
  per-frame pyFAI rotations (``rot1``/``rot2``/``rot3``) and the GI
  incidence angle.  Used by the v2 NeXus writer/reader (xdart 0.37+) to
  support flexible diffractometer conventions (2-circle, psic,
  psic-halpha, custom).

These are the *per-frame, scalar* geometry primitives.  Per-pixel q-space
mapping for RSM lives in :mod:`xrd_tools.core.geometry.pixel_q`.
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

    def __post_init__(self) -> None:
        # Normalise tuple-typed fields after construction.  JSON load
        # (``ExperimentConfig.from_file``) delivers lists here; without
        # this normalisation a round-tripped config compares unequal
        # to the original.  Also catches callers who pass lists by
        # mistake.
        for name in ("sample_rot", "detector_rot", "r_i", "hxrd_n", "hxrd_q"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                setattr(self, name, tuple(value))

    def make_hxrd(self, energy: float) -> "xu.HXRD":
        """Build an ``xu.HXRD`` instance at the given energy.

        ``xrayutilities`` is imported lazily inside this method so that
        importing :mod:`xrd_tools.core` never triggers the xu
        import (it is still a declared dependency and must be installed
        before calling this method).
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
    (degrees).  The consumer (``derive_per_frame``) handles deg→rad
    conversion for ``rot1/rot2/rot3``; the incidence angle stays in
    degrees.
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


# ---------------------------------------------------------------------------
# Canonical Diffractometer (one description, two derived adapter views)
# ---------------------------------------------------------------------------
#
# ``DiffractometerConfig`` (xrayutilities / RSM) and ``DiffractometerGeometry``
# (pyFAI per-frame rotations / GI incidence) are two consumer-specific
# encodings of the *same* physical goniometer, authored independently — so they
# can silently drift.  ``Diffractometer`` is the single declarative description
# that holds the complete instrument and emits BOTH views on demand:
#
#   * ``to_pyfai_per_frame(motors)`` == today's ``DiffractometerGeometry``
#     (motor columns → ``rot1/rot2/rot3`` (rad) + GI ``incident_angle`` (deg)),
#     consumed by integration / GI / stitch ``MultiGeometry``.
#   * ``to_qconversion()`` / ``to_hxrd(energy)`` == today's
#     ``DiffractometerConfig.make_hxrd`` (xu ``QConversion`` / ``HXRD``),
#     consumed by RSM ``PixelQMap`` / gridding.
#
# A preset (``psic()``, ``two_circle()`` …) authors BOTH halves in one place —
# the single correspondence point, so the two views can no longer disagree.
# See ``docs/design/design_diffractometer_geometry_jun2026.md`` §3.2.
#
# This object is the *description*, never a solver: angle→Q stays in
# xrayutilities (lazy-imported inside the adapters; the dataclass stays
# dependency-light so the ``[rsm]`` extra remains optional).


@dataclass(frozen=True)
class Diffractometer:
    """Canonical declarative goniometer description (one object, two views).

    Holds the complete instrument description and derives, on demand, the
    pyFAI per-frame view (:meth:`to_pyfai_per_frame`) and the xrayutilities
    view (:meth:`to_qconversion` / :meth:`to_hxrd`).  ``preset`` records
    which factory built it (data, not a closed enum); ``"custom"`` for an
    ad-hoc instance and ``"fitted"`` once a calibration has been refined in.

    The two halves:

    * **pyFAI / integration** — ``rot1``/``rot2``/``rot3``/``incident_angle``
      :class:`AngleMapping`\\s (motor column → derived angle).  ``rot*`` come
      back in **radians** from :meth:`to_pyfai_per_frame` (``deg2rad`` folded
      in); ``incident_angle`` stays in **degrees**.  ``AngleMapping.sign``
      doubles as a fitted dimensionless scale, so a calibrated goniometer
      (``rot = scale·motor + offset``) maps on with no extra field.
    * **xrayutilities / RSM** — the ordered ``sample_circles`` /
      ``detector_circles`` axis-direction stacks, ``r_i`` (primary-beam
      direction), the ``camera`` orientation ``(init_area_detrot,
      init_area_tiltazimuth)``, and the ``hxrd_*`` reference vectors.

    ``circle_motors`` (xu: which motor drives each circle, aligned to
    ``sample_circles + detector_circles``) is carried for completeness /
    persistence; it is not yet consumed by the adapters (RSM passes its
    motor list explicitly), so a preset may leave it empty.
    """

    preset: str = "custom"

    # --- pyFAI / integration view (per-frame 3-rotation + GI incidence) -----
    rot1: AngleMapping = field(default_factory=AngleMapping)
    rot2: AngleMapping = field(default_factory=AngleMapping)
    rot3: AngleMapping = field(default_factory=AngleMapping)
    incident_angle: AngleMapping = field(default_factory=AngleMapping)

    # --- xrayutilities / RSM view (circle stack + camera + HXRD refs) -------
    sample_circles: tuple[str, ...] = ("z-", "y+", "z-")
    detector_circles: tuple[str, ...] = ("z-",)
    r_i: tuple[float, float, float] = (0.0, 1.0, 0.0)
    #: detector-on-arm orientation: (init_area_detrot, init_area_tiltazimuth).
    camera: tuple[str, str] = ("z-", "x+")
    hxrd_n: tuple[float, float, float] = (0.0, 1.0, 0.0)
    hxrd_q: tuple[float, float, float] = (1.0, 0.0, 0.0)
    hxrd_geometry: str = "real"
    #: xu: motor driving each circle, aligned to sample_circles + detector_circles.
    circle_motors: tuple[AngleMapping, ...] = ()

    # --- motors to persist verbatim (superset of the consumed ones) ---------
    sample_motors: tuple[str, ...] = ()
    detector_motors: tuple[str, ...] = ()

    # --- xu construction kwargs (three separate passthrough dicts) ----------
    qconv_kwargs: Mapping[str, Any] = field(default_factory=dict)
    hxrd_kwargs: Mapping[str, Any] = field(default_factory=dict)
    ang2q_kwargs: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Presets — author BOTH halves consistently
    # ------------------------------------------------------------------

    @classmethod
    def two_circle(cls, tth: str = "tth", th: str = "th",
                   gonchi: str | None = None) -> "Diffractometer":
        """Vertical theta–2theta two-circle (``rot1←tth``, incidence``←th``)."""
        sample = (th,) if gonchi is None else (th, gonchi)
        return cls(
            preset="two_circle",
            rot1=AngleMapping(source_motor=tth),
            incident_angle=AngleMapping(source_motor=th),
            sample_circles=("z-",),
            detector_circles=("z-",),
            camera=("z-", "x+"),
            sample_motors=sample,
            detector_motors=(tth,),
            circle_motors=(AngleMapping(source_motor=th),
                           AngleMapping(source_motor=tth)),
        )

    @classmethod
    def psic(cls, del_: str = "del", nu: str = "nu", eta: str = "eta",
             chi: str = "chi", phi: str = "phi",
             mu: str = "mu") -> "Diffractometer":
        """6-circle psic (``rot1←nu``, ``rot2←del``, incidence``←eta``).

        The xu axis stacks + camera orientation are the validated values for
        the SSRL Pilatus-300k-w psic arm (``xu_geometry_del_nu.json``).  The
        ``circle_motors`` ordering follows the conventional SPEC psic order
        ``(mu, eta, chi, phi)`` sample, ``(nu, del)`` detector; it is not yet
        adapter-consumed, so the exact motor↔axis order is cross-validated
        against real ``Ang2Q.area`` usage when the stitch/RSM gate lands.
        """
        return cls(
            preset="psic",
            rot1=AngleMapping(source_motor=nu),
            rot2=AngleMapping(source_motor=del_),
            incident_angle=AngleMapping(source_motor=eta),
            sample_circles=("x+", "z-", "y+", "z-"),
            detector_circles=("x+", "z-"),
            camera=("x-", "z+"),
            sample_motors=(eta, chi, phi, mu),
            detector_motors=(del_, nu),
            circle_motors=(
                AngleMapping(source_motor=mu), AngleMapping(source_motor=eta),
                AngleMapping(source_motor=chi), AngleMapping(source_motor=phi),
                AngleMapping(source_motor=nu), AngleMapping(source_motor=del_),
            ),
        )

    #: ``sixc`` and ``psic`` are the same 6 circles (a psic *is* a sixc); the
    #: alias keeps the design's preset vocabulary while reusing one authoring.
    @classmethod
    def sixc(cls, **kwargs) -> "Diffractometer":
        """Alias of :meth:`psic` (a psic is a 6-circle in a fixed order)."""
        return cls.psic(**kwargs)

    @classmethod
    def fourc(cls, tth: str = "tth", om: str = "om", chi: str = "chi",
              phi: str = "phi") -> "Diffractometer":
        """Eulerian four-circle (``rot1←tth``, incidence``←om``).

        The xu sample stack ``(om, chi, phi)`` + single detector arm uses the
        conventional Eulerian axis directions as a *starting* convention; the
        signs are refined by a real-data fit (no in-repo fixture validates
        them), so only the structural preset-consistency test guards this one.
        """
        return cls(
            preset="fourc",
            rot1=AngleMapping(source_motor=tth),
            incident_angle=AngleMapping(source_motor=om),
            sample_circles=("z-", "y+", "z-"),
            detector_circles=("z-",),
            camera=("z-", "x+"),
            sample_motors=(om, chi, phi),
            detector_motors=(tth,),
            circle_motors=(
                AngleMapping(source_motor=om), AngleMapping(source_motor=chi),
                AngleMapping(source_motor=phi), AngleMapping(source_motor=tth),
            ),
        )

    @classmethod
    def psic_halpha(cls, del_: str = "del", nu: str = "nu",
                    halpha: str = "halpha", chi: str = "chi",
                    phi: str = "phi", mu: str = "mu") -> "Diffractometer":
        """psic with the GI incidence taken from ``halpha`` instead of ``eta``."""
        return cls(
            preset="psic_halpha",
            rot1=AngleMapping(source_motor=nu),
            rot2=AngleMapping(source_motor=del_),
            incident_angle=AngleMapping(source_motor=halpha),
            sample_circles=("x+", "z-", "y+", "z-"),
            detector_circles=("x+", "z-"),
            camera=("x-", "z+"),
            sample_motors=(halpha, chi, phi, mu),
            detector_motors=(del_, nu),
            circle_motors=(
                AngleMapping(source_motor=mu), AngleMapping(source_motor=halpha),
                AngleMapping(source_motor=chi), AngleMapping(source_motor=phi),
                AngleMapping(source_motor=nu), AngleMapping(source_motor=del_),
            ),
        )

    # ------------------------------------------------------------------
    # Adapter view 1 — pyFAI per-frame (== DiffractometerGeometry)
    # ------------------------------------------------------------------

    def to_pyfai_per_frame(self, motors: Mapping[str, "np.ndarray"]
                           ) -> dict[str, np.ndarray]:
        """Return ``{rot1, rot2, rot3, incident_angle}`` per-frame arrays.

        ``rot1/rot2/rot3`` come back in **radians**; ``incident_angle`` in
        **degrees**.  Byte-equal to ``DiffractometerGeometry.derive_per_frame``
        — the GI incidence is intentionally *not* ``deg2rad``-converted.
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
            mapping = getattr(self, key)
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

    # ------------------------------------------------------------------
    # Adapter view 2 — xrayutilities (== DiffractometerConfig.make_hxrd)
    # ------------------------------------------------------------------

    def to_qconversion(self) -> "xu.QConversion":
        """Build the ``xu.QConversion`` for this instrument (energy-free).

        Byte-equal to the ``QConversion`` ``DiffractometerConfig.make_hxrd``
        builds: ``QConversion(sample_circles, detector_circles, r_i,
        **qconv_kwargs)`` — positional arg order is load-bearing.  Energy is
        *not* a ``QConversion`` parameter; it flows through :meth:`to_hxrd`.
        """
        import xrayutilities as xu  # noqa: PLC0415 — intentional lazy import

        return xu.QConversion(
            self.sample_circles,
            self.detector_circles,
            self.r_i,
            **self.qconv_kwargs,
        )

    def to_hxrd(self, energy: float) -> "xu.HXRD":
        """Build an ``xu.HXRD`` at the given energy (eV).

        Byte-equal to ``DiffractometerConfig.make_hxrd``:
        ``HXRD(hxrd_n, hxrd_q, geometry=hxrd_geometry, en=energy,
        qconv=to_qconversion(), **hxrd_kwargs)``.
        """
        import xrayutilities as xu  # noqa: PLC0415 — intentional lazy import

        return xu.HXRD(
            self.hxrd_n,
            self.hxrd_q,
            geometry=self.hxrd_geometry,
            en=energy,
            qconv=self.to_qconversion(),
            **self.hxrd_kwargs,
        )

    # ------------------------------------------------------------------
    # Motor bookkeeping + serialisation
    # ------------------------------------------------------------------

    def all_referenced_motors(self) -> tuple[str, ...]:
        """All motor columns this instrument needs from the meta file."""
        seen: dict[str, None] = {}
        for mapping in (self.rot1, self.rot2, self.rot3, self.incident_angle):
            if mapping.is_active and mapping.source_motor not in seen:
                seen[mapping.source_motor] = None
        for cm in self.circle_motors:
            if cm.is_active and cm.source_motor not in seen:
                seen[cm.source_motor] = None
        for m in self.sample_motors + self.detector_motors:
            if m and m not in seen:
                seen[m] = None
        return tuple(seen.keys())

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Diffractometer":
        d = json.loads(s)

        def _am(key: str) -> AngleMapping:
            return AngleMapping(**d.get(key, {}))

        return cls(
            preset=d.get("preset", "custom"),
            rot1=_am("rot1"),
            rot2=_am("rot2"),
            rot3=_am("rot3"),
            incident_angle=_am("incident_angle"),
            sample_circles=tuple(d.get("sample_circles", ())),
            detector_circles=tuple(d.get("detector_circles", ())),
            r_i=tuple(d.get("r_i", (0.0, 1.0, 0.0))),
            camera=tuple(d.get("camera", ("z-", "x+"))),
            hxrd_n=tuple(d.get("hxrd_n", (0.0, 1.0, 0.0))),
            hxrd_q=tuple(d.get("hxrd_q", (1.0, 0.0, 0.0))),
            hxrd_geometry=d.get("hxrd_geometry", "real"),
            circle_motors=tuple(
                AngleMapping(**cm) for cm in d.get("circle_motors", ())
            ),
            sample_motors=tuple(d.get("sample_motors", ())),
            detector_motors=tuple(d.get("detector_motors", ())),
            qconv_kwargs=dict(d.get("qconv_kwargs", {})),
            hxrd_kwargs=dict(d.get("hxrd_kwargs", {})),
            ang2q_kwargs=dict(d.get("ang2q_kwargs", {})),
        )


__all__ = [
    "AngleMapping",
    "Convention",
    "Diffractometer",
    "DiffractometerConfig",
    "DiffractometerGeometry",
]
