"""Grazing-incidence per-pixel corrections (the GI half of the shared stack).

Built on xrayutilities optical constants (xu provides δ/β/n/χ0/αc/absorption_length
but NO ready footprint/Fresnel/solid-angle helpers — those are here).  The hard
invariant (get it wrong and intensity silently corrupts):

  * **footprint, Fresnel/Vineyard, absorption are INTENSITY weights** → they
    multiply into the ``Σnorm`` denominator (same contract as solid-angle /
    polarization: ``I(bin) = Σ raw / Σ norm``).
  * **refraction is a POSITION correction** → it rewrites the per-pixel q (the
    out-of-plane component) and NEVER touches the weight.

αi (incidence) is a per-frame scalar (degrees, from
``Diffractometer.derive_per_frame(...)["incident_angle"]``); αf (exit) is a
per-pixel array (radians) from the GI integrator's exit-angle / qoop map.

All angles are small (grazing), all in **radians** inside the formulas, all
optical constants energy-dependent.  Reference (verified, xrayutilities 1.7.12):
Si @ 10 keV → δ=4.887e-6, β=7.591e-8, αc=0.17913°, absorption_length=129.98 µm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: floor on αf before any 1/sin or transmission (the notebook's grazing guard).
_AF_FLOOR_DEG = 0.01


def _floor_alpha(alpha_rad: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(alpha_rad, dtype=float), np.radians(_AF_FLOOR_DEG), None)


# ---------------------------------------------------------------------------
# Unambiguous physics primitives (each pinned by a committed gate)
# ---------------------------------------------------------------------------

def fresnel_transmission_sq(alpha_rad: np.ndarray | float, ac_rad: float,
                            beta: float) -> np.ndarray:
    """``|T(α)|²`` — single-interface amplitude transmission (Vineyard factor).

    ``T(α) = 2α / (α + sqrt(α² − αc² + 2iβ))``.  Peaks at ``α = αc`` (the Yoneda
    enhancement); the ``+2iβ`` regularizes the square-root kink so it is smooth
    through αc.
    """
    a = np.asarray(alpha_rad, dtype=float)
    kz_internal = np.sqrt(a * a - ac_rad ** 2 + 2j * beta)
    t = 2.0 * a / (a + kz_internal)
    return np.abs(t) ** 2


def footprint_weight(incident_angle_rad: np.ndarray | float) -> np.ndarray:
    """``1/sin(αi)`` — the over-illumination (illuminated-area) correction factor.

    A per-frame scalar (αi is constant across a frame), so it **cancels in a
    normalized I(q) at fixed αi** and only matters when αi varies frame-to-frame.
    Convention follows the GI-corrections notebook + design doc (``C = 1/sin αi``).
    """
    return 1.0 / np.sin(np.asarray(incident_angle_rad, dtype=float))


def refracted_angle(alpha_rad: np.ndarray | float, ac_rad: float,
                    beta: float) -> np.ndarray:
    """Internal (refracted) angle ``Re(sqrt(α² − αc² + 2iβ))``.

    Always ``< α``; tends to 0 (the evanescent regime) as ``α → αc`` from above
    and below.  Smooth through αc (the complex form, not the real-only clamp).
    """
    a = np.asarray(alpha_rad, dtype=float)
    return np.real(np.sqrt(a * a - ac_rad ** 2 + 2j * beta))


def absorption_path(incident_angle_rad: np.ndarray | float,
                    alpha_f_rad: np.ndarray | float) -> np.ndarray:
    """Geometric path-length sum ``1/sin αi + 1/sin αf`` (thickness-free form A).

    Both angles are floored at 0.01° so a sub-horizon/zero incidence or exit can
    never produce a ``±inf`` or sign-flipped path (1/sin diverges at 0)."""
    return (1.0 / np.sin(_floor_alpha(incident_angle_rad))
            + 1.0 / np.sin(_floor_alpha(alpha_f_rad)))


def film_absorption(incident_angle_rad: np.ndarray | float,
                    alpha_f_rad: np.ndarray | float, mu_per_A: float,
                    thickness_A: float) -> np.ndarray:
    """Finite-film averaged transmission ``A = (1 − e^{−μt·P}) / (μt·P)`` (≤ 1),
    ``P = 1/sin αi + 1/sin αf`` (design / Gasser-2025 form B).  → 1 (thin film) /
    → ``1/(μt·P)`` (thick); the grazing-exit path makes it small at small αf."""
    p = absorption_path(incident_angle_rad, alpha_f_rad)
    x = float(mu_per_A) * float(thickness_A) * p
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(x > 1e-12, (1.0 - np.exp(-x)) / x, 1.0)
    return a


# ---------------------------------------------------------------------------
# The GI correction stack (mode-configured; reads optical constants from xu)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GICorrectionStack:
    """Grazing-incidence per-pixel corrections, configured by material + energy.

    The intensity factors (footprint · Fresnel · absorption) compose into the
    ``Σnorm`` denominator via :meth:`gi_normalization`; refraction rewrites the
    q-map via :meth:`refract_q` (a position correction, never a weight).
    ``None``/off factors are no-ops.

    ⚠ Composition-sign note (flagged for review): the *primitives* above are
    unambiguous + gate-pinned, but the *direction* in which footprint and the
    path-length absorption enter the normalization is convention-dependent
    (the GI notebook and the design doc differ).  We follow the design contract
    (footprint → ``norm·=sin αi`` so ``corrected = result/sin αi``; absorption
    → ``norm·=A`` (form B) or ``norm·=1/P`` (form A) so grazing-exit intensity is
    raised) — verify against a worked GIXSGUI example when live data is available.
    """

    material: str | None = None
    energy_eV: float | None = None
    density_kg_m3: float | None = None        # set → Amorphous; None → predefined Crystal
    film_thickness_A: float | None = None     # None → path-length absorption (form A)
    footprint: bool = True
    refraction: bool = True                   # the POSITION correction
    fresnel: bool = True
    absorption: bool = True

    # -- optical constants from xrayutilities (lazy) ----------------------
    def _material_obj(self):
        import xrayutilities as xu  # noqa: PLC0415
        if not self.material:
            raise ValueError("GICorrectionStack.material is required for GI corrections")
        if self.density_kg_m3 is not None:
            return xu.materials.Amorphous(self.material, float(self.density_kg_m3))
        try:
            return getattr(xu.materials, self.material)
        except AttributeError as exc:
            raise ValueError(
                f"{self.material!r} is not a predefined xrayutilities material; "
                "pass density_kg_m3= to build an Amorphous material instead"
            ) from exc

    def optical_constants(self) -> dict[str, float]:
        """``{αc_rad, beta, k0_per_A, mu_per_A, delta}`` at this material+energy."""
        import xrayutilities as xu  # noqa: PLC0415
        if self.energy_eV is None:
            raise ValueError("GICorrectionStack.energy_eV is required")
        mat = self._material_obj()
        en = float(self.energy_eV)
        lam_A = float(xu.utilities.en2lam(en))
        absL_A = float(mat.absorption_length(en)) * 1.0e4  # µm → Å
        return {
            "ac_rad": float(mat.critical_angle(en, deg=False)),
            "beta": float(mat.ibeta(en)),
            "delta": float(mat.delta(en)),
            "k0_per_A": 2.0 * np.pi / lam_A,
            "mu_per_A": 1.0 / absL_A,
        }

    # -- INTENSITY factors → the Σnorm denominator ------------------------
    def gi_normalization(self, *, incident_angle_deg: float,
                         alpha_f_rad: np.ndarray) -> np.ndarray:
        """Per-pixel GI intensity normalization (footprint·Fresnel·absorption).

        Multiply this into the base (solid-angle·polarization) normalization;
        the accumulator forms ``I = Σ raw / Σ norm``.  Refraction is NOT here.
        """
        oc = self.optical_constants()
        # floor αi (like αf): a zero/sub-horizon incidence must not make 1/sin αi
        # diverge or sin αi flip sign (silent intensity corruption).
        ai = float(_floor_alpha(np.radians(float(incident_angle_deg))))
        af = _floor_alpha(alpha_f_rad)
        norm = np.ones(np.shape(af), dtype=float)
        if self.footprint:
            # corrected = result/sin αi  ⇒  sin αi in the denominator (per-frame).
            norm = norm * np.sin(ai)
        if self.fresnel:
            v = (fresnel_transmission_sq(ai, oc["ac_rad"], oc["beta"])
                 * fresnel_transmission_sq(af, oc["ac_rad"], oc["beta"]))
            norm = norm * v                              # measured enhanced near αc
        if self.absorption:
            if self.film_thickness_A is not None:
                norm = norm * film_absorption(ai, af, oc["mu_per_A"],
                                              self.film_thickness_A)
            else:
                norm = norm / absorption_path(ai, af)    # path-length boost (form A)
        return norm

    # -- POSITION correction → the q-map ----------------------------------
    def refract_q(self, *, incident_angle_deg: float, alpha_f_rad: np.ndarray,
                  q_total: np.ndarray, q_z: np.ndarray) -> np.ndarray:
        """Refraction-shifted ``|q|``: replace the out-of-plane component with the
        refracted ``qz`` (Snell), keeping the in-plane part.  Maps measured qz
        **down** (peaks shift to smaller qz); the shift vanishes far above αc.

        Contract: reflection-geometry GI — every pixel is taken to be in the
        **upper (qz ≥ 0) half-plane**.  The sign of the input ``q_z`` is not used
        (only ``q_total² − q_z²`` enters, recovering the in-plane magnitude), and
        the returned ``|q|`` is always built from a non-negative refracted ``qz``.
        A genuinely below-horizon pixel would be folded to ``+qz`` — do not feed
        transmission-geometry (qz < 0) pixels here."""
        oc = self.optical_constants()
        k0 = oc["k0_per_A"]
        ai_in = refracted_angle(np.radians(float(incident_angle_deg)),
                                oc["ac_rad"], oc["beta"])
        af_in = refracted_angle(_floor_alpha(alpha_f_rad), oc["ac_rad"], oc["beta"])
        qz_refr = k0 * (np.sin(af_in) + np.sin(ai_in))
        q_ip2 = np.asarray(q_total, dtype=float) ** 2 - np.asarray(q_z, dtype=float) ** 2
        return np.sqrt(np.clip(q_ip2 + qz_refr ** 2, 0.0, None))

    # -- provenance -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "material": self.material, "energy_eV": self.energy_eV,
            "density_kg_m3": self.density_kg_m3,
            "film_thickness_A": self.film_thickness_A,
            "footprint": self.footprint, "refraction": self.refraction,
            "fresnel": self.fresnel, "absorption": self.absorption,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GICorrectionStack":
        return cls(
            material=d.get("material"), energy_eV=d.get("energy_eV"),
            density_kg_m3=d.get("density_kg_m3"),
            film_thickness_A=d.get("film_thickness_A"),
            footprint=bool(d.get("footprint", True)),
            refraction=bool(d.get("refraction", True)),
            fresnel=bool(d.get("fresnel", True)),
            absorption=bool(d.get("absorption", True)),
        )


__all__ = [
    "GICorrectionStack",
    "absorption_path",
    "film_absorption",
    "footprint_weight",
    "fresnel_transmission_sq",
    "refracted_angle",
]
