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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np

from xrd_tools.core.containers import PONI

if TYPE_CHECKING:  # Only needed for type checkers; avoided at runtime.
    import xrayutilities as xu


@dataclass(slots=True)
class DiffractometerConfig:
    """Geometry configuration for ``xu.QConversion`` and ``xu.HXRD``.

    .. deprecated::
        Superseded by :class:`Diffractometer` (the one canonical object, ADR-0007),
        which exposes the same ``make_hxrd`` + ``init_area_*`` surface as a drop-in.
        Kept as the value-preserving reference + as the source/target of
        :meth:`Diffractometer.from_diffractometer_config` /
        :meth:`Diffractometer.to_diffractometer_config`, and to parse legacy
        experiment-config JSON.  Do not author new geometry with it.
    """

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

    .. deprecated::
        Superseded by :class:`Diffractometer` (the one canonical object, ADR-0007),
        which carries the same per-frame ``rot*``/``incident_angle`` mappings and
        exposes ``derive_per_frame`` as a drop-in.  Kept as the value-preserving
        reference + as the source/target of
        :meth:`Diffractometer.from_diffractometer_geometry` /
        :meth:`Diffractometer.to_diffractometer_geometry`, and to parse a legacy
        ``mapping_json`` on reload.  Do not author new geometry with it.

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
# Detector calibration (static, per-scan) — PONI + Detector_config + mount
# ---------------------------------------------------------------------------
#
# ``PONI`` (core/containers) carries the pyFAI calibration geometry but DROPS
# ``Detector_config`` (a non-default orientation / custom mask / binning is
# silently lost — stitching GAP B) and has no notion of the beamline-specific
# image-array orientation needed to match the raw detector frame to the
# calibration (GAP E).  ``DetectorCalibration`` wraps a ``PONI`` and restores
# both.  It is *static detector calibration*, NOT goniometer state — the
# per-frame rotations live on ``Diffractometer`` (design §3.2).


def _json_canonical(value: Any) -> Any:
    """Coerce a value to its JSON-canonical form so ``to_json`` never crashes on
    a numpy scalar and a serialize→parse round-trip is idempotent (tuples →
    lists, numpy scalars → Python scalars, recursively).  Applied to the
    free-form serialization-bound fields (``detector_config`` + the xu kwargs)
    whose contents are written verbatim as JSON."""
    if isinstance(value, Mapping):
        return {str(k): _json_canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_canonical(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ImageOrientation:
    """Beamline-specific transform mapping a raw detector array to the
    orientation the calibration was computed in (stitching GAP E).

    The detector's pyFAI ``Detector_config`` orientation alone is *not*
    sufficient (the validated psic case needed a 180° array rotation on top
    of ``orientation 3``).  Applied to the trailing two axes so a single 2D
    frame ``(H, W)`` or a stack ``(N, H, W)`` both work.  A 90°/270° rotation
    or a transpose swaps the detector dimensions (:attr:`swaps_axes`).
    """

    rotation: int = 0          # CCW degrees, one of {0, 90, 180, 270}
    flip_vertical: bool = False
    flip_horizontal: bool = False
    transpose: bool = False

    def __post_init__(self) -> None:
        # coerce a numpy/parsed scalar to a plain int so membership + to_json work
        object.__setattr__(self, "rotation", int(self.rotation))
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError(
                f"rotation must be one of 0/90/180/270, got {self.rotation!r}"
            )

    @property
    def is_identity(self) -> bool:
        return (self.rotation == 0 and not self.flip_vertical
                and not self.flip_horizontal and not self.transpose)

    @property
    def swaps_axes(self) -> bool:
        """True if the transform swaps the detector's two dimensions."""
        return (self.rotation in (90, 270)) ^ self.transpose

    def apply(self, image: "np.ndarray") -> np.ndarray:
        """Apply the transform to the trailing two axes of ``image``."""
        out = np.asarray(image)
        if self.transpose:
            out = np.swapaxes(out, -1, -2)
        if self.rotation:
            out = np.rot90(out, k=self.rotation // 90, axes=(-2, -1))
        if self.flip_vertical:
            out = out[..., ::-1, :]
        if self.flip_horizontal:
            out = out[..., :, ::-1]
        return np.ascontiguousarray(out)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ImageOrientation":
        return cls(
            rotation=int(d.get("rotation", 0)),
            flip_vertical=bool(d.get("flip_vertical", False)),
            flip_horizontal=bool(d.get("flip_horizontal", False)),
            transpose=bool(d.get("transpose", False)),
        )


@dataclass(frozen=True)
class DetectorCalibration:
    """Static detector calibration: a ``PONI`` + ``Detector_config`` + mount.

    Restores the two things a bare ``PONI`` drops:

    * ``detector_config`` — the pyFAI ``Detector_config`` dict
      (``{"orientation": N}``, custom mask, binning) so a non-default mount
      survives a round-trip (GAP B).
    * ``image_orientation`` — the raw-array transform that aligns the
      detector frame to the calibration (GAP E).

    The pyFAI 90° panel mount and the xu camera orientation are the *same*
    physics in two conventions; the mount is held once here (the pyFAI
    ``detector_config``) and once on :class:`Diffractometer` (the xu
    ``camera`` tuple) — both populated by the preset/fit, not derived from
    each other (the correspondence is fit-determined, design §3.5).
    """

    poni: PONI
    detector_config: Mapping[str, Any] = field(default_factory=dict)
    image_orientation: ImageOrientation = field(default_factory=ImageOrientation)

    def __post_init__(self) -> None:
        # store detector_config JSON-canonical so to_json never crashes on a
        # numpy scalar and the round-trip is idempotent (tuple -> list, etc.)
        object.__setattr__(self, "detector_config",
                           _json_canonical(dict(self.detector_config)))

    def to_json(self) -> str:
        return json.dumps(self._as_jsonable(), separators=(",", ":"),
                          sort_keys=True)

    def _as_jsonable(self) -> dict[str, Any]:
        return {
            "poni": self.poni.to_dict(),
            "detector_config": dict(self.detector_config),
            "image_orientation": self.image_orientation.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DetectorCalibration":
        return cls(
            poni=PONI.from_dict(d.get("poni", {})),
            detector_config=dict(d.get("detector_config", {})),
            image_orientation=ImageOrientation.from_dict(
                d.get("image_orientation", {})),
        )

    @classmethod
    def from_json(cls, s: str) -> "DetectorCalibration":
        return cls.from_dict(json.loads(s))


_DEG2RAD = float(np.deg2rad(1.0))


def _eval_geometry_expr(expr: str, variables: Mapping[str, Any]) -> float:
    """Safely evaluate one pyFAI ``GeometryTransformation`` expression.

    The expressions (``"rot2_scale * pos + rot2_offset"``, ``"0.0"``, …) are
    pure arithmetic, so they are evaluated with ``numexpr`` (arithmetic-only,
    no arbitrary code — unlike ``eval`` on file-sourced strings) with an empty
    global scope.
    """
    import numexpr  # noqa: PLC0415 — lazy; only needed by the gonio bridge

    return float(numexpr.evaluate(str(expr), local_dict=dict(variables),
                                  global_dict={}))


def _resolve_source_motors(
    pos_names: Sequence[str],
    source_motors: "str | Sequence[str] | Mapping[str, str] | None",
) -> dict[str, str]:
    """Map each goniometer position axis name to a real scan motor column."""
    if source_motors is None:
        return {n: n for n in pos_names}
    if isinstance(source_motors, str):
        if len(pos_names) != 1:
            raise ValueError(
                f"a single source_motors string needs exactly one position "
                f"axis, but the goniometer has {pos_names}"
            )
        return {pos_names[0]: source_motors}
    if isinstance(source_motors, Mapping):
        return {n: str(source_motors.get(n, n)) for n in pos_names}
    seq = list(source_motors)
    if len(seq) != len(pos_names):
        raise ValueError(
            f"source_motors has {len(seq)} entries for {len(pos_names)} "
            f"position axes {pos_names}"
        )
    return {n: str(m) for n, m in zip(pos_names, seq)}


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
    # The defaults are the validated **psic** orientation (the house standard;
    # examples/.../RSM/RSM_process.ipynb production calls) — a bare
    # ``Diffractometer()`` is psic-oriented (motors unwired until a preset is
    # applied).  ``sample_circles`` == (mu, eta, chi, phi); ``detector_circles``
    # == (nu, del); ``camera`` == camera_or; HXRD idir/ndir == [0,1,0]/[0,0,1].
    sample_circles: tuple[str, ...] = ("x+", "z-", "y+", "z-")
    detector_circles: tuple[str, ...] = ("x+", "z-")
    r_i: tuple[float, float, float] = (0.0, 1.0, 0.0)
    #: detector-on-arm orientation: (init_area_detrot, init_area_tiltazimuth).
    camera: tuple[str, str] = ("z-", "x-")
    hxrd_n: tuple[float, float, float] = (0.0, 1.0, 0.0)
    hxrd_q: tuple[float, float, float] = (0.0, 0.0, 1.0)
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

    # --- static detector calibration (None until calibrated/fitted) ---------
    #: the base ``DetectorCalibration`` (dist/poni/rot + Detector_config +
    #: image mount).  Per-frame full geometry = this base ⊕
    #: :meth:`to_pyfai_per_frame`.  ``None`` for a bare preset.
    calibration: "DetectorCalibration | None" = None

    def __post_init__(self) -> None:
        # store the free-form xu passthrough dicts JSON-canonical so to_json
        # never crashes on a numpy scalar and a round-trip is idempotent
        # (tuple -> list); these are written verbatim into the JSON blob.
        for name in ("qconv_kwargs", "hxrd_kwargs", "ang2q_kwargs"):
            object.__setattr__(self, name,
                               _json_canonical(dict(getattr(self, name))))

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
            hxrd_n=(0.0, 1.0, 0.0), hxrd_q=(1.0, 0.0, 0.0),
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

        The house-standard default geometry.  The xu axis stacks, camera
        orientation, HXRD idir/ndir, and motor order are the production values
        from ``examples/.../RSM/RSM_process.ipynb`` (``sample_or``/``det_or``/
        ``beam_dr``/``camera_or`` + ``diff_motors=['mu','eta','chi','phi',
        'nu','del']``), validated on real RSM data.  ``circle_motors`` therefore
        follows that confirmed order: ``(mu, eta, chi, phi)`` sample,
        ``(nu, del)`` detector.
        """
        return cls(
            preset="psic",
            rot1=AngleMapping(source_motor=nu),
            rot2=AngleMapping(source_motor=del_),
            incident_angle=AngleMapping(source_motor=eta),
            sample_circles=("x+", "z-", "y+", "z-"),
            detector_circles=("x+", "z-"),
            camera=("z-", "x-"),
            hxrd_n=(0.0, 1.0, 0.0), hxrd_q=(0.0, 0.0, 1.0),
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
            hxrd_n=(0.0, 1.0, 0.0), hxrd_q=(1.0, 0.0, 0.0),
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
            camera=("z-", "x-"),
            hxrd_n=(0.0, 1.0, 0.0), hxrd_q=(0.0, 0.0, 1.0),
            sample_motors=(halpha, chi, phi, mu),
            detector_motors=(del_, nu),
            circle_motors=(
                AngleMapping(source_motor=mu), AngleMapping(source_motor=halpha),
                AngleMapping(source_motor=chi), AngleMapping(source_motor=phi),
                AngleMapping(source_motor=nu), AngleMapping(source_motor=del_),
            ),
        )

    # ------------------------------------------------------------------
    # Bridge — load a pyFAI GoniometerRefinement calibration (closes GAP D)
    # ------------------------------------------------------------------

    @classmethod
    def from_pyfai_goniometer(
        cls,
        gonio: "Mapping[str, Any] | str | Path",
        *,
        source_motors: "str | Sequence[str] | Mapping[str, str] | None" = None,
        base: "Diffractometer | None" = None,
        image_orientation: "ImageOrientation | None" = None,
        preset: str = "fitted",
    ) -> "Diffractometer":
        """Build a *fitted* :class:`Diffractometer` from a pyFAI goniometer JSON.

        Parses a standard pyFAI ``GoniometerRefinement`` serialization (the
        ``trans_function`` ``GeometryTransformation`` with ``rotN_expr``
        strings) into per-axis :class:`AngleMapping`\\s + a base
        :class:`DetectorCalibration` (incl. ``Detector_config``) — so a
        beamline can calibrate in pyFAI and stitch/RSM headlessly with no
        pyFAI ``Goniometer`` at runtime (design §3.2a, closes stitching GAP D).

        Decomposition (each ``rotN_expr`` is linear in the position axes):

        * a **constant** rotation (no position dependence, e.g. ``rot1_expr =
          "rot1"`` or ``rot3_expr = "0.0"``) → the base ``PONI.rotN`` (radians),
          an **inactive** :class:`AngleMapping`.
        * a **position-linear** rotation (``rotN = scale·pos + offset``) → an
          :class:`AngleMapping` carrying the *whole* rotation
          (``sign = scale/deg2rad``, ``offset = offset_rad/deg2rad`` — both in
          degrees, since :meth:`to_pyfai_per_frame` folds ``deg2rad`` back in),
          with the base ``PONI.rotN = 0``.

        So per-frame ``rotN = calibration.poni.rotN +
        to_pyfai_per_frame(motors)[rotN]`` reproduces pyFAI ``get_ai(pos).rotN``.

        Parameters
        ----------
        gonio : Mapping, path, or JSON string
            The goniometer record.
        source_motors : str / sequence / mapping, optional
            Maps each goniometer position axis (``trans_function.pos_names``,
            generically ``"pos"``) to a real scan motor column (e.g. ``"del"``).
            Defaults to the position-axis names verbatim.
        base : Diffractometer, optional
            Donates the xrayutilities half (circle stacks, ``camera``, HXRD
            refs, motor lists) — the gonio JSON carries only the pyFAI side.
            Pass e.g. ``Diffractometer.psic()`` so the result also feeds RSM.
        image_orientation : ImageOrientation, optional
            The raw-array mount transform (GAP E); not in the pyFAI JSON.
        preset : str
            Recorded tag (default ``"fitted"``).

        Raises
        ------
        NotImplementedError
            For custom goniometer subclasses (no ``trans_function`` /
            ``ExtendedTransformation`` / non-linear position dependence).
        """
        if isinstance(gonio, Mapping):
            data: dict[str, Any] = dict(gonio)
        else:
            p = Path(gonio)
            text = p.read_text() if p.exists() else str(gonio)
            data = json.loads(text)

        trans = data.get("trans_function")
        if trans is None:
            raise NotImplementedError(
                f"{data.get('content')!r} is not a standard pyFAI "
                "GeometryTransformation (no 'trans_function'); custom "
                "goniometer subclasses (StackedArmGoniometer / "
                "GeometrySurfaceModel / ...) are out of scope"
            )
        tcontent = trans.get("content", "GeometryTransformation")
        if tcontent not in ("GeometryTransformation", "GeometryTranslation"):
            raise NotImplementedError(
                f"transformation {tcontent!r} is unsupported "
                "(only GeometryTransformation)"
            )

        param_names = list(data.get("param_names", []))
        param = list(data.get("param", []))
        params = dict(zip(param_names, param))
        pos_names = list(trans.get("pos_names",
                                   data.get("pos_names", ["pos"])))
        # numexpr needs ``pi`` available; pyFAI ships it in constants but
        # default it so an expr referencing pi never KeyErrors.
        constants = {"pi": float(np.pi), **dict(trans.get("constants", {}))}
        local_base = {**constants, **params}
        motor_for = _resolve_source_motors(pos_names, source_motors)

        pos_zero = {n: 0.0 for n in pos_names}

        def _ev(expr: str, posvals: Mapping[str, float]) -> float:
            return _eval_geometry_expr(expr, {**local_base, **posvals})

        def _require_pos_independent(label: str, expr: str) -> float:
            """Base PONI fields must be constant in position (the per-frame
            geometry emits only rot*; a moving-detector base is out of scope)."""
            f0 = _ev(expr, pos_zero)
            for name in pos_names:
                if abs(_ev(expr, {**pos_zero, name: 1.0}) - f0) > 1e-12 * (1 + abs(f0)):
                    raise NotImplementedError(
                        f"{label}_expr depends on position axis {name!r} "
                        "(moving-detector goniometer); only a fixed base "
                        "dist/poni is supported"
                    )
            return f0

        # Base geometry (constant for a standard GeometryTransformation).
        dist = _require_pos_independent("dist", trans["dist_expr"])
        poni1 = _require_pos_independent("poni1", trans["poni1_expr"])
        poni2 = _require_pos_independent("poni2", trans["poni2_expr"])

        rot_mappings: dict[str, AngleMapping] = {}
        base_rots: dict[str, float] = {}
        for axis in ("rot1", "rot2", "rot3"):
            expr = trans[f"{axis}_expr"]
            f0 = _ev(expr, pos_zero)
            scales = []
            for name in pos_names:
                f1 = _ev(expr, {**pos_zero, name: 1.0})
                f2 = _ev(expr, {**pos_zero, name: 2.0})
                if abs((f2 - f1) - (f1 - f0)) > 1e-12 * (1 + abs(f1)):
                    raise NotImplementedError(
                        f"{axis}_expr is non-linear in {name!r}; only linear "
                        "goniometer transformations are supported"
                    )
                scales.append(f1 - f0)
            # Reject a coupled (cross-term, e.g. nu*del) dependence: each axis
            # alone can look linear yet a pure product slips through as a sum of
            # zero slopes (mis-read as constant).  A genuinely axis-separable
            # linear expr satisfies f(all=1) == f0 + Σ scales; a cross-term does
            # not.  Fail loud, consistent with the non-linear rejection above.
            if len(pos_names) > 1:
                f_all = _ev(expr, {n: 1.0 for n in pos_names})
                if abs(f_all - (f0 + sum(scales))) > 1e-12 * (1 + abs(f_all)):
                    raise NotImplementedError(
                        f"{axis}_expr couples multiple position axes "
                        "(cross-term); only axis-separable linear goniometer "
                        "transformations are supported"
                    )
            nz = [i for i, s in enumerate(scales) if abs(s) > 1e-15]
            if not nz:
                base_rots[axis] = f0           # constant → base PONI
                rot_mappings[axis] = AngleMapping()
            elif len(nz) == 1:
                i = nz[0]
                base_rots[axis] = 0.0           # whole rotation in the mapping
                rot_mappings[axis] = AngleMapping(
                    source_motor=motor_for[pos_names[i]],
                    sign=scales[i] / _DEG2RAD,
                    offset=f0 / _DEG2RAD,
                )
            else:
                raise NotImplementedError(
                    f"{axis}_expr depends on multiple position axes; "
                    "custom/surface goniometers are out of scope"
                )

        cal = DetectorCalibration(
            poni=PONI(
                dist=dist, poni1=poni1, poni2=poni2,
                rot1=base_rots["rot1"], rot2=base_rots["rot2"],
                rot3=base_rots["rot3"],
                wavelength=float(data.get("wavelength", 0.0)),
                detector=str(data.get("detector", "")),
            ),
            detector_config=dict(data.get("detector_config", {})),
            image_orientation=image_orientation or ImageOrientation(),
        )

        # The xu half is not in the pyFAI JSON — donate it from ``base``.
        xu_fields: dict[str, Any] = {}
        if base is not None:
            xu_fields = dict(
                sample_circles=base.sample_circles,
                detector_circles=base.detector_circles,
                r_i=base.r_i, camera=base.camera,
                hxrd_n=base.hxrd_n, hxrd_q=base.hxrd_q,
                hxrd_geometry=base.hxrd_geometry,
                circle_motors=base.circle_motors,
                sample_motors=base.sample_motors,
                detector_motors=base.detector_motors,
            )
        else:
            xu_fields = dict(
                detector_motors=tuple(motor_for[n] for n in pos_names),
            )

        return cls(
            preset=preset,
            rot1=rot_mappings["rot1"],
            rot2=rot_mappings["rot2"],
            rot3=rot_mappings["rot3"],
            incident_angle=AngleMapping(),  # pyFAI gonios carry no GI incidence
            calibration=cal,
            **xu_fields,
        )

    # ------------------------------------------------------------------
    # Legacy interop — lift / lower the two old encodings (compat shims)
    # ------------------------------------------------------------------
    #
    # The two old classes stay authoritative for one release; these bridges
    # let a consumer build the canonical object from whatever it holds today
    # (``Scan.geometry`` = a ``DiffractometerGeometry``; RSM config = a
    # ``DiffractometerConfig``) and extract the old view back out, all
    # value-preserving so step 4 can repoint callers without behaviour change.

    @classmethod
    def from_diffractometer_geometry(
        cls, geom: "DiffractometerGeometry", **overrides: Any
    ) -> "Diffractometer":
        """Lift a legacy ``DiffractometerGeometry`` (pyFAI per-frame view).

        Copies the pyFAI half verbatim; the xu half stays at defaults unless
        supplied via ``overrides`` (or later donated from a config/preset).
        """
        fields: dict[str, Any] = dict(
            preset=geom.convention,
            rot1=geom.rot1, rot2=geom.rot2, rot3=geom.rot3,
            incident_angle=geom.incident_angle,
            sample_motors=geom.sample_motors,
            detector_motors=geom.detector_motors,
        )
        fields.update(overrides)
        return cls(**fields)

    def to_diffractometer_geometry(self) -> "DiffractometerGeometry":
        """Lower to the legacy pyFAI per-frame view (byte-equal mappings)."""
        conv = (self.preset
                if self.preset in ("two_circle", "psic", "psic_halpha", "custom")
                else "custom")
        return DiffractometerGeometry(
            convention=conv,  # type: ignore[arg-type]
            rot1=self.rot1, rot2=self.rot2, rot3=self.rot3,
            incident_angle=self.incident_angle,
            sample_motors=self.sample_motors,
            detector_motors=self.detector_motors,
        )

    @classmethod
    def from_diffractometer_config(
        cls, config: "DiffractometerConfig", **overrides: Any
    ) -> "Diffractometer":
        """Lift a legacy ``DiffractometerConfig`` (xrayutilities view)."""
        fields: dict[str, Any] = dict(
            sample_circles=config.sample_rot,
            detector_circles=config.detector_rot,
            r_i=config.r_i,
            camera=(config.init_area_detrot, config.init_area_tiltazimuth),
            hxrd_n=config.hxrd_n, hxrd_q=config.hxrd_q,
            hxrd_geometry=config.hxrd_geometry,
            qconv_kwargs=dict(config.q_conv_kwargs),
            hxrd_kwargs=dict(config.hxrd_kwargs),
            ang2q_kwargs=dict(config.ang2q_kwargs),
        )
        fields.update(overrides)
        return cls(**fields)

    def to_diffractometer_config(self) -> "DiffractometerConfig":
        """Lower to the legacy xrayutilities view (byte-equal QConversion)."""
        return DiffractometerConfig(
            sample_rot=self.sample_circles,
            detector_rot=self.detector_circles,
            r_i=self.r_i,
            q_conv_kwargs=dict(self.qconv_kwargs),
            hxrd_n=self.hxrd_n, hxrd_q=self.hxrd_q,
            hxrd_geometry=self.hxrd_geometry,
            hxrd_kwargs=dict(self.hxrd_kwargs),
            init_area_detrot=self.camera[0],
            init_area_tiltazimuth=self.camera[1],
            ang2q_kwargs=dict(self.ang2q_kwargs),
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

    def derive_per_frame(self, motors: Mapping[str, "np.ndarray"]
                         ) -> dict[str, np.ndarray]:
        """Legacy-compatible alias of :meth:`to_pyfai_per_frame`.

        A ``Diffractometer`` is duck-typed exactly like the old
        ``DiffractometerGeometry`` (the writer's ``Scan.geometry`` consumer
        calls only ``.derive_per_frame`` + ``.all_referenced_motors``), so it
        is a drop-in replacement — the step-4 seam.
        """
        return self.to_pyfai_per_frame(motors)

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

    # -- PixelQMap drop-in: the same names DiffractometerConfig exposes, so a
    #    Diffractometer is a transparent ``PixelQMap.diff_config`` (the RSM
    #    pipeline reads make_hxrd + init_area_detrot/tiltazimuth + ang2q_kwargs).
    def make_hxrd(self, energy: float) -> "xu.HXRD":
        """Legacy-compatible alias of :meth:`to_hxrd`."""
        return self.to_hxrd(energy)

    @property
    def init_area_detrot(self) -> str:
        return self.camera[0]

    @property
    def init_area_tiltazimuth(self) -> str:
        return self.camera[1]

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
            sample_circles=tuple(d.get("sample_circles",
                                       ("x+", "z-", "y+", "z-"))),
            detector_circles=tuple(d.get("detector_circles", ("x+", "z-"))),
            r_i=tuple(d.get("r_i", (0.0, 1.0, 0.0))),
            camera=tuple(d.get("camera", ("z-", "x-"))),
            hxrd_n=tuple(d.get("hxrd_n", (0.0, 1.0, 0.0))),
            hxrd_q=tuple(d.get("hxrd_q", (0.0, 0.0, 1.0))),
            hxrd_geometry=d.get("hxrd_geometry", "real"),
            circle_motors=tuple(
                AngleMapping(**cm) for cm in d.get("circle_motors", ())
            ),
            sample_motors=tuple(d.get("sample_motors", ())),
            detector_motors=tuple(d.get("detector_motors", ())),
            qconv_kwargs=dict(d.get("qconv_kwargs", {})),
            hxrd_kwargs=dict(d.get("hxrd_kwargs", {})),
            ang2q_kwargs=dict(d.get("ang2q_kwargs", {})),
            calibration=(
                DetectorCalibration.from_dict(d["calibration"])
                if d.get("calibration") is not None else None
            ),
        )


def assemble_circle_angles(
    diffractometer: "Diffractometer",
    scan: Any,
    indices: "list[int] | None" = None,
) -> list[np.ndarray]:
    """Assemble the per-frame xu sample+detector angle arrays from a
    :class:`Diffractometer`'s ``circle_motors`` + a scan's per-frame motor table.

    Returns a list aligned to ``sample_circles + detector_circles``: for each
    circle, ``sign·motor + offset`` (its :class:`AngleMapping`) of the source
    motor pulled from ``scan.scan_data``, in the exact order
    ``xu.QConversion``/``Ang2Q.area`` expects.

    The circle ORDER + per-circle sign/offset — i.e. the q-convention — are
    **carried in ``circle_motors``** (authored in the validated preset, e.g.
    psic from ``RSM_process.ipynb``), never invented here; this only mechanises
    the lookup + :meth:`AngleMapping.apply`.  It is the drop-in replacement for
    the legacy explicit-``diff_motors`` angle list (the per-frame angle assembly
    the design calls "the one wiring task" shared by RSM and the xu_hist stitch).

    The ABSOLUTE convention (the circle order/signs themselves) is validated
    against real data (the RSM/stitch notebooks), not here — this pins only that
    the assembly faithfully reproduces the legacy explicit path.
    """
    circles = getattr(diffractometer, "circle_motors", ())
    if not circles:
        raise ValueError(
            "Diffractometer.circle_motors is empty — this preset did not wire "
            "the xu circle→motor map needed for the angle assembly; use a preset "
            "that wires circle_motors (e.g. psic) or pass diff_motors explicitly.")
    scan_data = getattr(scan, "scan_data", None)
    if scan_data is None:
        raise ValueError("scan has no scan_data per-frame motor table")
    cols = (list(scan_data.columns) if hasattr(scan_data, "columns")
            else list(scan_data))
    missing = [m.source_motor for m in circles if m.source_motor not in cols]
    if missing:
        raise KeyError(
            f"circle motors {missing!r} not in scan.scan_data (have {cols!r})")

    rows: "list[int] | None" = None
    if indices is not None:
        labels = [int(idx) for idx in getattr(scan, "frame_indices")]
        row_of = {label: r for r, label in enumerate(labels)}
        rows = [row_of[int(idx)] for idx in indices]

    out: list[np.ndarray] = []
    for m in circles:
        col = np.asarray(scan_data[m.source_motor], dtype=float)
        if rows is not None:
            col = col[rows]
        out.append(np.asarray(m.apply(col), dtype=float))
    return out


__all__ = [
    "AngleMapping",
    "Convention",
    "DetectorCalibration",
    "Diffractometer",
    "DiffractometerConfig",
    "DiffractometerGeometry",
    "ImageOrientation",
    "assemble_circle_angles",
]
