"""The shared per-pixel intensity-correction stack (the pre-weight for every
stitch backend + RSM).

A binned intensity is ``Σ(raw) / Σ(normalization)`` over the pixels in a bin —
pyFAI's scheme — where ``normalization = solid_angle · polarization · …``.  This
is **not** the naive ``Σ(raw · weight) / N``: with more than one pixel per bin and
a per-pixel-varying correction the two disagree, and only the ``Σraw/Σnorm`` form
reproduces pyFAI (and is the statistically correct weighted mean).  So this module
produces the per-pixel **normalization** array (the denominator the accumulator
sums); :meth:`CorrectionStack.weight` (``1/normalization``) is offered only for
correcting a *display* image, never for the accumulator.

The factors are read from a pyFAI ``AzimuthalIntegrator`` (built per frame from
the ``DetectorCalibration`` ⊕ the ``Diffractometer`` rotations) so stitch and RSM
share identical, validated arrays (design_intensity_corrections_jun2026.md §3).
Grazing-incidence factors (footprint / refraction / Fresnel) are a separate
GI-mode stack (P2b).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator


@dataclass(frozen=True)
class CorrectionStack:
    """Mode-configured per-pixel correction stack.

    Parameters
    ----------
    solid_angle : bool
        Flat-panel pixels subtend a solid angle ``∝ cos³(2θ)/dist²``; on by
        default (matches pyFAI ``correctSolidAngle=True``).
    polarization_factor : float or None
        Horizontal-polarization factor (synchrotron ≈ 0.9–1.0); ``None`` = off
        (matches pyFAI's default — polarization must be switched on explicitly,
        even for powder stitch).
    air_absorption_mu : float or None
        Air-path linear attenuation ``μ_air`` (1/m); ``None`` = off.  Approximate
        (``T = exp(-μ·dist/cos2θ)``) and not pyFAI-validated — a secondary factor.
    """

    solid_angle: bool = True
    polarization_factor: float | None = None
    air_absorption_mu: float | None = None

    def normalization(self, ai: "AzimuthalIntegrator",
                      shape: tuple[int, int] | None = None) -> np.ndarray:
        """Per-pixel normalization array ``= solid_angle · polarization · …``.

        The accumulator forms ``I(bin) = Σ raw / Σ normalization``.  Returns an
        all-ones array when every correction is off (a no-op normalization).
        """
        shp = tuple(shape) if shape is not None else tuple(ai.detector.shape)
        norm = np.ones(shp, dtype=float)
        if self.solid_angle:
            norm = norm * np.asarray(ai.solidAngleArray(shape=shp), dtype=float)
        if self.polarization_factor is not None:
            norm = norm * np.asarray(
                ai.polarization(shape=shp, factor=float(self.polarization_factor)),
                dtype=float)
        if self.air_absorption_mu is not None:
            tth = np.asarray(ai.twoThetaArray(shape=shp), dtype=float)
            # air path sample→pixel ≈ dist / cos(2θ); transmitted fraction
            # T = exp(-μ·path).  Absorption removes counts, so dividing raw by T
            # (i.e. T in the normalization denominator) restores them.
            path = float(ai.dist) / np.clip(np.cos(tth), 1e-6, None)
            norm = norm * np.exp(-float(self.air_absorption_mu) * path)
        return norm

    def weight(self, ai: "AzimuthalIntegrator",
               shape: tuple[int, int] | None = None) -> np.ndarray:
        """``1/normalization`` — the per-pixel factor to correct a **display**
        image (``corrected = raw · weight``).  NOT for the accumulator; use
        :meth:`normalization` + ``Σraw/Σnorm`` there.  Non-finite where the
        normalization vanishes."""
        norm = self.normalization(ai, shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / norm
        w[~np.isfinite(w)] = np.nan
        return w

    @property
    def is_identity(self) -> bool:
        return (not self.solid_angle and self.polarization_factor is None
                and self.air_absorption_mu is None)

    def to_dict(self) -> dict[str, Any]:
        """Provenance dict (which corrections + params were applied)."""
        return {
            "solid_angle": bool(self.solid_angle),
            "polarization_factor": self.polarization_factor,
            "air_absorption_mu": self.air_absorption_mu,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CorrectionStack":
        return cls(
            solid_angle=bool(d.get("solid_angle", True)),
            polarization_factor=d.get("polarization_factor"),
            air_absorption_mu=d.get("air_absorption_mu"),
        )


__all__ = ["CorrectionStack"]
