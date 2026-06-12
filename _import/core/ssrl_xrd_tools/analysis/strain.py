"""
Strain analysis from GIXRD data via the sin²(ψ) method.

Workflow
--------
1. Obtain a GI-corrected 2D polar map ``(q_total, χ)`` from
   ``integrate_gi_polar`` (pyFAI FiberIntegrator).
2. Extract 1D I(q) slices in narrow χ sectors with :func:`extract_chi_sectors`.
3. Fit a peak in each sector to obtain d(ψ) using :func:`fit_peak_vs_psi`.
4. Perform the sin²(ψ) linear regression with :func:`sin2psi_regression`.

Definitions
-----------
- χ (chi) : polar angle = arctan(q_ip / q_oop), measured from the surface
  normal toward the in-plane direction.  pyFAI ``chigi_deg`` follows this
  convention (``eq_chi_gi`` returns ``arctan2(q_ip, q_oop)``).
  χ = 0° → out-of-plane (surface normal); χ = 90° → in-plane.
- ψ (psi) : angle of the scattering vector from the surface normal = |χ|.
  ψ = 0° → planes parallel to surface; ψ = 90° → planes perpendicular.
- Missing-wedge: near χ = ±90° (ψ ≈ 90°) the GI geometry has an inaccessible
  region.  Expect usable ψ coverage out to ~70–80°.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import linregress

from ssrl_xrd_tools.analysis.fitting.fit import fit_peaks
from ssrl_xrd_tools.core.containers import IntegrationResult2D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ChiSector:
    """A single I(q) slice extracted from a (q, χ) polar map."""
    chi_center: float       # centre of the χ sector (degrees)
    chi_width: float        # full width of the sector (degrees)
    psi: float              # ψ = |χ_center|
    q: np.ndarray           # q-axis for this sector
    intensity: np.ndarray   # mean intensity in the sector


@dataclass
class PeakFitResult:
    """Result of fitting a single peak in one χ sector."""
    psi: float              # ψ value (degrees)
    sin2psi: float          # sin²(ψ)
    q_center: float         # fitted peak centre in q (Å⁻¹)
    d_spacing: float        # d = 2π / q_center (Å)
    q_center_err: float     # uncertainty on q_center
    d_spacing_err: float    # propagated uncertainty on d
    fit_result: Any = field(repr=False)  # lmfit ModelResult


@dataclass
class Sin2PsiResult:
    """Result of the full sin²(ψ) linear regression."""
    d0: float               # unstrained d-spacing (intercept)
    slope: float            # Δd / Δ(sin²ψ) — proportional to stress
    d0_err: float           # stderr on intercept
    slope_err: float        # stderr on slope
    r_squared: float        # R² of the linear fit
    psi_deg: np.ndarray     # ψ values used (degrees)
    sin2psi: np.ndarray     # sin²(ψ) values
    d_values: np.ndarray    # fitted d-spacings
    d_errors: np.ndarray    # uncertainties on d-spacings
    peak_fits: list[PeakFitResult]  # individual sector fits
    # Elastic constants and derived stress (populated when E, nu are
    # supplied to sin2psi_regression; otherwise None).
    E: float | None = None          # Young's modulus (same units as stress)
    nu: float | None = None         # Poisson's ratio
    stress: float | None = None     # in-plane normal stress σ_φ
    stress_err: float | None = None # propagated 1σ uncertainty on stress


# ---------------------------------------------------------------------------
# Step 1: extract χ sectors from (q, χ) polar map
# ---------------------------------------------------------------------------

def extract_chi_sectors(
    result2d: IntegrationResult2D,
    chi_centers: np.ndarray | list[float] | None = None,
    chi_width: float = 5.0,
    n_sectors: int | None = None,
    chi_range: tuple[float, float] | None = None,
) -> list[ChiSector]:
    """
    Extract I(q) slices at specified χ sectors from a (q, χ) polar map.

    Parameters
    ----------
    result2d : IntegrationResult2D
        2D polar-coordinate result from ``integrate_gi_polar``.
        ``result2d.radial`` = q_total axis, ``result2d.azimuthal`` = χ axis.
        ``result2d.intensity.shape == (len(radial), len(azimuthal))``.
    chi_centers : array-like or None
        Explicit list of χ centre values (degrees).  If *None*, sectors are
        generated automatically from *n_sectors* and *chi_range*.
    chi_width : float
        Full angular width of each sector in degrees.
    n_sectors : int or None
        Number of equally-spaced sectors to generate (used when
        *chi_centers* is None).  Default: enough sectors of width
        *chi_width* to tile *chi_range* without overlap.
    chi_range : tuple of float or None
        ``(chi_min, chi_max)`` to use for auto-generated sectors.  Default:
        the full range of ``result2d.azimuthal``.

    Returns
    -------
    list of ChiSector
        One entry per usable sector (sectors with no valid data are skipped).
    """
    chi_axis = np.asarray(result2d.azimuthal, dtype=float)
    q_axis = np.asarray(result2d.radial, dtype=float)
    intensity = np.asarray(result2d.intensity, dtype=float)  # (nq, nchi)

    if intensity.shape != (len(q_axis), len(chi_axis)):
        raise ValueError(
            f"Intensity shape {intensity.shape} does not match "
            f"(len(radial), len(azimuthal)) = ({len(q_axis)}, {len(chi_axis)}). "
            f"Check that result2d axes match the intensity array."
        )

    if chi_range is None:
        chi_range = (float(chi_axis.min()), float(chi_axis.max()))

    if chi_centers is None:
        if n_sectors is None:
            n_sectors = max(1, int(np.floor(
                (chi_range[1] - chi_range[0]) / chi_width
            )))
        chi_centers = np.linspace(
            chi_range[0] + chi_width / 2,
            chi_range[1] - chi_width / 2,
            n_sectors,
        )
    chi_centers = np.asarray(chi_centers, dtype=float)

    sectors: list[ChiSector] = []
    for cc in chi_centers:
        lo = cc - chi_width / 2
        hi = cc + chi_width / 2
        mask = (chi_axis >= lo) & (chi_axis <= hi)
        if not np.any(mask):
            continue

        # Mean intensity over the chi sector: shape (nq,)
        sector_I = np.nanmean(intensity[:, mask], axis=1)

        # Skip sectors that are all NaN / zero (missing wedge)
        if np.all(~np.isfinite(sector_I)) or np.nanmax(sector_I) <= 0:
            continue

        psi = abs(cc)
        sectors.append(ChiSector(
            chi_center=float(cc),
            chi_width=chi_width,
            psi=float(psi),
            q=q_axis.copy(),
            intensity=sector_I,
        ))

    logger.info("Extracted %d usable χ sectors from polar map", len(sectors))
    return sectors


# ---------------------------------------------------------------------------
# Step 2: fit peak position in each sector → d(ψ)
# ---------------------------------------------------------------------------

def fit_peak_vs_psi(
    sectors: list[ChiSector],
    q_range: tuple[float, float],
    model: str = "pseudovoigt",
    background: str = "linear",
    sigma_init: float | None = None,
    sigma_bounds: tuple[float, float] | None = None,
    center_bounds_delta: float | None = None,
) -> list[PeakFitResult]:
    """
    Fit a peak in a specified q range for each χ sector.

    Parameters
    ----------
    sectors : list of ChiSector
        Output from :func:`extract_chi_sectors`.
    q_range : tuple of float
        ``(q_min, q_max)`` window isolating the peak of interest (Å⁻¹).
    model : str
        Peak profile: 'gaussian', 'lorentzian', 'voigt', 'pseudovoigt',
        or 'lorentzian_squared'.
    background : str
        Background model: 'linear', 'constant', 'chebyshev2'–'chebyshev4',
        or 'none'.
    sigma_init : float or None
        Initial peak width.  None → auto-estimate from q_range.
    sigma_bounds : (min, max) or None
        Bounds on peak width.  None → auto.
    center_bounds_delta : float or None
        Constrain peak centre to ± delta around the initial position.

    Returns
    -------
    list of PeakFitResult
        One entry per sector where the fit converged.  Sorted by ψ.
    """
    results: list[PeakFitResult] = []
    for sector in sectors:
        q = sector.q
        I = sector.intensity
        mask = (q >= q_range[0]) & (q <= q_range[1])
        q_cut = q[mask]
        I_cut = I[mask]

        if len(q_cut) < 5:
            logger.warning(
                "χ=%.1f° (ψ=%.1f°): too few points (%d) in q range, skipping",
                sector.chi_center, sector.psi, len(q_cut),
            )
            continue

        try:
            fit_result = fit_peaks(
                q_cut, I_cut,
                model=model,
                n_peaks=1,
                background=background,
                sigma_init=sigma_init,
                sigma_bounds=sigma_bounds,
                center_bounds_delta=center_bounds_delta,
            )
        except Exception as exc:
            logger.warning(
                "χ=%.1f° (ψ=%.1f°): fit failed: %s", sector.chi_center,
                sector.psi, exc,
            )
            continue

        if not fit_result.success:
            logger.warning(
                "χ=%.1f° (ψ=%.1f°): fit did not converge",
                sector.chi_center, sector.psi,
            )
            continue

        q_cen = fit_result.peak_centers[0]
        q_cen_err = fit_result.peak_centers_err[0]

        d = 2.0 * np.pi / q_cen
        # Error propagation: δd = (2π / q²) * δq
        d_err = (2.0 * np.pi / q_cen**2) * q_cen_err

        results.append(PeakFitResult(
            psi=sector.psi,
            sin2psi=np.sin(np.deg2rad(sector.psi))**2,
            q_center=q_cen,
            d_spacing=d,
            q_center_err=q_cen_err,
            d_spacing_err=d_err,
            fit_result=fit_result,
        ))

    results.sort(key=lambda r: r.psi)
    logger.info("Peak fits converged for %d / %d sectors", len(results),
                len(sectors))
    return results


# ---------------------------------------------------------------------------
# Step 3: sin²(ψ) linear regression → strain / stress
# ---------------------------------------------------------------------------

def sin2psi_regression(
    peak_fits: list[PeakFitResult],
    E: float | None = None,
    nu: float | None = None,
) -> Sin2PsiResult:
    """
    Perform a d vs. sin²(ψ) linear regression.

    Parameters
    ----------
    peak_fits : list of PeakFitResult
        Output from :func:`fit_peak_vs_psi`.
    E : float, optional
        Young's modulus (X-ray elastic constant) for the specific (hkl).
        Units set the units of the returned stress (e.g. GPa, MPa).
    nu : float, optional
        Poisson's ratio (X-ray elastic constant) for the specific (hkl).

    Returns
    -------
    Sin2PsiResult
        Contains the regression slope (proportional to stress), intercept
        (d₀), R² value, and all individual data points. If both ``E`` and
        ``nu`` are provided, the result also includes the computed
        in-plane normal stress ``σ_φ = (E / (1+ν)) · (slope / d₀)``.

    Notes
    -----
    The linear model is ``d(ψ) = d₀ + m · sin²(ψ)``, where:

    - ``d₀`` is the unstrained lattice spacing (at ψ = 0, planes ∥ surface).
    - ``m > 0`` → tensile in-plane stress.
    - ``m < 0`` → compressive in-plane stress.

    Biaxial sin²ψ stress relation (assuming σ_33 = 0 and the chosen φ
    direction)::

        ε_φψ = ((1+ν)/E) · σ_φ · sin²ψ  −  (ν/E) · (σ_11 + σ_22)

    which together with ``ε_φψ ≈ (d_ψ − d₀)/d₀`` gives::

        σ_φ = (E / (1+ν)) · (m / d₀)

    where ``m`` is the regression slope. The returned ``stress_err`` is
    propagated from ``slope_err`` and ``d0_err`` assuming they are
    independent.
    """
    if len(peak_fits) < 3:
        raise ValueError(
            f"Need at least 3 data points for sin²(ψ) regression, "
            f"got {len(peak_fits)}"
        )

    sin2psi = np.array([r.sin2psi for r in peak_fits])
    d_vals = np.array([r.d_spacing for r in peak_fits])
    d_errs = np.array([r.d_spacing_err for r in peak_fits])

    reg = linregress(sin2psi, d_vals)

    # Optional stress computation
    stress: float | None = None
    stress_err: float | None = None
    if E is not None and nu is not None:
        if reg.intercept == 0 or not np.isfinite(reg.intercept):
            raise ValueError(
                f"Cannot compute stress: intercept d0={reg.intercept} is "
                "zero or non-finite."
            )
        if nu <= -1.0:
            raise ValueError(f"Invalid Poisson's ratio nu={nu} (need ν > −1).")
        prefactor = E / (1.0 + nu)
        stress = prefactor * (reg.slope / reg.intercept)
        # σ = k · m / d0  →  (δσ/σ)² = (δm/m)² + (δd0/d0)²
        rel_slope = (reg.stderr / reg.slope) if reg.slope != 0 else 0.0
        rel_d0 = (reg.intercept_stderr / reg.intercept)
        stress_err = abs(stress) * float(np.hypot(rel_slope, rel_d0))
    elif (E is None) != (nu is None):
        raise ValueError(
            "Provide both E and nu (or neither) to compute stress."
        )

    return Sin2PsiResult(
        d0=reg.intercept,
        slope=reg.slope,
        d0_err=reg.intercept_stderr,
        slope_err=reg.stderr,
        r_squared=reg.rvalue**2,
        psi_deg=np.array([r.psi for r in peak_fits]),
        sin2psi=sin2psi,
        d_values=d_vals,
        d_errors=d_errs,
        peak_fits=peak_fits,
        E=E,
        nu=nu,
        stress=stress,
        stress_err=stress_err,
    )


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def sin2psi_analysis(
    result2d: IntegrationResult2D,
    q_range: tuple[float, float],
    chi_centers: np.ndarray | list[float] | None = None,
    chi_width: float = 5.0,
    n_sectors: int | None = None,
    chi_range: tuple[float, float] | None = None,
    model: str = "pseudovoigt",
    background: str = "linear",
    sigma_init: float | None = None,
    sigma_bounds: tuple[float, float] | None = None,
    center_bounds_delta: float | None = None,
    E: float | None = None,
    nu: float | None = None,
) -> Sin2PsiResult:
    """
    End-to-end sin²(ψ) strain analysis from a GI-corrected polar map.

    This is a convenience wrapper that calls :func:`extract_chi_sectors`,
    :func:`fit_peak_vs_psi`, and :func:`sin2psi_regression` in sequence.

    Parameters
    ----------
    result2d : IntegrationResult2D
        2D polar map ``(q_total, χ)`` from ``integrate_gi_polar``.
    q_range : tuple of float
        ``(q_min, q_max)`` isolating the peak of interest (Å⁻¹).
    chi_centers, chi_width, n_sectors, chi_range
        Passed to :func:`extract_chi_sectors`.
    model, background, sigma_init, sigma_bounds, center_bounds_delta
        Passed to :func:`fit_peak_vs_psi`.

    Returns
    -------
    Sin2PsiResult
    """
    sectors = extract_chi_sectors(
        result2d,
        chi_centers=chi_centers,
        chi_width=chi_width,
        n_sectors=n_sectors,
        chi_range=chi_range,
    )
    peak_fits = fit_peak_vs_psi(
        sectors, q_range=q_range, model=model,
        background=background, sigma_init=sigma_init,
        sigma_bounds=sigma_bounds,
        center_bounds_delta=center_bounds_delta,
    )
    return sin2psi_regression(peak_fits, E=E, nu=nu)


__all__ = [
    "ChiSector",
    "PeakFitResult",
    "Sin2PsiResult",
    "extract_chi_sectors",
    "fit_peak_vs_psi",
    "sin2psi_regression",
    "sin2psi_analysis",
]
