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

from xrd_tools.analysis.axis_units import require_inverse_angstrom
from xrd_tools.core.containers import IntegrationResult2D

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
    from xrd_tools.analysis.fitting import fit_peaks

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
    require_inverse_angstrom(result2d.unit, operation="sin2psi analysis")
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


# ---------------------------------------------------------------------------
# STRAIN-ANALYSIS-128: the qualified sin²(ψ) path
# ---------------------------------------------------------------------------
# The deferred ledger's STRAIN/A3 gates: ψ from the bins a sector uses, an explicit and labelled d0 (the intercept is
# never a stress-free d0), a weighted regression with its covariance, reflection-specific elastic constants, and a
# refraction check. Directional stress (uniaxial, shear) is refused: in one GI fiber map the +χ and −χ branches sit at
# the in-plane directions φ and −φ, and φ changes with ψ (maintainer, 2026-10-30: "Defer directional stress
# (Recommended)"). The functions above keep their behaviour for their headless callers.

_GEOMETRY_REFUSAL = (
    "needs each point's in-plane direction (azimuth): a GI fiber map gives only the polar angle psi, and its +chi "
    "and -chi branches are not fixed in-plane directions, so that geometry model is deferred")


@dataclass(frozen=True, eq=False)
class StrainSector:
    """One χ sector of a GI fiber map.

    ``psi`` is the mean |χ| of the bins the sector uses: a narrow-sector reporting angle (sin² of a mean is not the
    mean of sin²). ``branch`` is χ's sign. Per q row, ``intensity`` is the mean of the samples whose intensity (and σ,
    where the map has one) is finite, and ``sigma`` is √(Σσᵢ²)/N over those same samples (None without a σ map); it
    leaves out correlation between bins. ``excluded`` names why the sector is left out (None: it is used).
    """

    chi_center: float
    chi_width: float
    psi: float
    branch: int
    q: np.ndarray = field(repr=False)
    intensity: np.ndarray = field(repr=False)
    sigma: np.ndarray | None = field(repr=False)
    finite_fraction: float
    excluded: str | None = None


def strain_sectors(
    result2d: IntegrationResult2D,
    *,
    chi_centers=None,
    chi_width: float = 5.0,
    n_sectors: int | None = None,
    chi_range: tuple[float, float] | None = None,
    branches: tuple[int, ...] = (1,),
    min_finite_fraction: float = 0.5,
) -> tuple[StrainSector, ...]:
    """The χ sectors of a GI fiber map (radial q_total, azimuthal χ_GI, intensity (nq, nχ)).

    Explicit ``chi_centers`` keep their sign; otherwise ``n_sectors`` (default: as many as tile the range) of width
    ``chi_width`` tile the |χ| range ``chi_range`` on each of ``branches`` (+1, −1). No sector spans χ = 0. A sector
    whose finite fraction is under ``min_finite_fraction`` (the missing wedge, read from the map's own coverage, or
    detector gaps) stays in the list with its reason.
    """
    chi = np.asarray(result2d.azimuthal, dtype=float)
    q = np.asarray(result2d.radial, dtype=float)
    intensity = np.asarray(result2d.intensity, dtype=float)
    if intensity.shape != (q.size, chi.size):
        raise ValueError(
            f"Intensity shape {intensity.shape} does not match (len(radial), len(azimuthal)) = "
            f"({q.size}, {chi.size})")
    sigma = None if result2d.sigma is None else np.asarray(result2d.sigma, dtype=float)
    if sigma is not None and sigma.shape != intensity.shape:
        raise ValueError("the map's sigma does not match its intensity")
    width = float(chi_width)
    if not width > 0:
        raise ValueError("the sector width must be positive")
    if chi_centers is not None:
        centers = [float(center) for center in chi_centers]
        for center in centers:
            if center - width / 2 < 0 < center + width / 2:
                raise ValueError(
                    f"the sector at chi {center:g} deg spans chi = 0: keep the +chi and -chi branches apart")
    else:
        if chi_range is None:
            chi_range = (0.0, float(np.nanmax(np.abs(chi))))
        low, high = sorted(abs(float(value)) for value in chi_range)
        if n_sectors is None:
            n_sectors = max(1, int(np.floor((high - low) / width + 1e-9)))
        side = tuple(int(branch) for branch in branches)
        if not side or any(branch not in (1, -1) for branch in side):
            raise ValueError("branches are +1 and/or -1")
        magnitudes = np.linspace(low + width / 2, high - width / 2, int(n_sectors))
        centers = [branch * float(magnitude) for branch in side for magnitude in magnitudes]
    sectors = []
    for center in centers:
        branch = 1 if center > 0 else -1
        columns = (chi >= center - width / 2) & (chi <= center + width / 2) & (chi >= 0 if branch > 0 else chi <= 0)
        block = intensity[:, columns]
        finite = np.isfinite(block)
        sigmas = None
        if sigma is not None:
            sigmas = sigma[:, columns]
            finite &= np.isfinite(sigmas)
        count = finite.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, np.where(finite, block, 0.0).sum(axis=1) / count, np.nan)
            error = None if sigmas is None else np.where(
                count > 0, np.sqrt((np.where(finite, sigmas, 0.0) ** 2).sum(axis=1)) / count, np.nan)
        used = np.abs(chi[columns][finite.any(axis=0)])
        fraction = float(finite.sum() / finite.size) if finite.size else 0.0
        excluded = None
        if used.size == 0:
            psi, excluded = abs(center), "no data in this sector"
        else:
            psi = float(np.mean(used))
            if fraction < min_finite_fraction:
                excluded = (f"only {fraction:.0%} of the sector has data (the missing wedge or detector gaps)")
        sectors.append(StrainSector(center, width, psi, branch, q.copy(), mean, error, fraction, excluded))
    return tuple(sectors)


@dataclass(frozen=True, eq=False)
class StrainPoint:
    """One ψ point of a sin²ψ fit: ψ in degrees, the spacing ``d`` in Å and its standard error (None: unknown), the χ
    branch (+1, −1), the fitted ``q`` and its error. ``excluded`` names why the point is left out."""

    psi: float
    d: float
    d_err: float | None
    branch: int = 1
    q: float | None = None
    q_err: float | None = None
    excluded: str | None = None
    sector: StrainSector | None = field(default=None, repr=False)
    fit: Any = field(default=None, repr=False)

    @property
    def sin2psi(self) -> float:
        return float(np.sin(np.deg2rad(self.psi)) ** 2)


def _peak_screen(x: np.ndarray, y: np.ndarray, sigma: np.ndarray | None) -> tuple[float, float, float]:
    """The window's smoothed peak height above its median, the noise of that smoothed profile (from σ when the map has
    one, else from the spread of neighbour differences) and the smoothed argmax as the fit's start. A screening
    heuristic for weak sectors, not a significance test."""
    size = max(1, min(9, (len(y) // 3) | 1))
    smooth = np.convolve(np.pad(y, size // 2, mode="edge"), np.ones(size) / size, mode="valid")
    peak = int(np.argmax(smooth))
    height = float(smooth[peak] - np.median(y))
    if sigma is not None:
        noise = float(np.median(sigma)) / np.sqrt(size)
    else:
        noise = float(np.median(np.abs(np.diff(y)))) / (0.6745 * np.sqrt(2.0) * np.sqrt(size))
    return height, max(noise, np.finfo(float).tiny), float(x[peak])


def fit_strain_sectors(
    sectors,
    q_range: tuple[float, float],
    *,
    model: str = "pseudovoigt",
    background: str = "linear",
    sigma_init: float | None = None,
    sigma_bounds: tuple[float, float] | None = None,
    center_bounds_delta: float | None = None,
    min_height_sigma: float = 5.0,
) -> tuple[StrainPoint, ...]:
    """One :class:`StrainPoint` per sector, in order: the peak in ``q_range`` fitted with the shared fitter, weighted
    by the sector's σ where it has one. Each fit starts at the window's smoothed maximum (``center_bounds_delta``
    bounds the centre around it) and, if that fails, at its neighbour's fitted centre. A sector whose peak is under
    ``min_height_sigma`` of its noise, or whose fit fails or leaves the window, is kept with its reason."""
    from xrd_tools.analysis.fitting import fit_peaks

    low, high = sorted(float(value) for value in q_range)
    points: list[StrainPoint] = []
    previous: float | None = None
    for sector in sectors:
        def left_out(reason: str, sector: StrainSector = sector) -> StrainPoint:
            return StrainPoint(sector.psi, float("nan"), None, sector.branch, excluded=reason, sector=sector)

        if sector.excluded:
            points.append(left_out(sector.excluded))
            continue
        window = (sector.q >= low) & (sector.q <= high) & np.isfinite(sector.intensity)
        if sector.sigma is not None:
            window &= np.isfinite(sector.sigma) & (sector.sigma > 0)
        if window.sum() < 5:
            points.append(left_out("fewer than 5 points in the q window"))
            continue
        x, y = sector.q[window], sector.intensity[window]
        weights = None if sector.sigma is None else 1.0 / sector.sigma[window]
        height, noise, start = _peak_screen(x, y, None if weights is None else sector.sigma[window])
        if not height >= min_height_sigma * noise:
            points.append(left_out(
                f"weak: the peak is {height / noise:.1f} times its noise, under {min_height_sigma:g}"))
            continue
        seeds = [start] + ([previous] if previous is not None and previous != start else [])
        fit, reason = None, "the fit did not converge"
        for seed in seeds:
            try:
                candidate = fit_peaks(
                    x, y, positions=[seed], model=model, background=background, sigma_init=sigma_init,
                    sigma_bounds=sigma_bounds, center_bounds_delta=center_bounds_delta,
                    **({} if weights is None else {"weights": weights}))
            except Exception as exc:  # noqa: BLE001 - one sector's failure is that sector's reason
                reason = f"the fit failed: {exc}"
                continue
            if candidate.success and low < candidate.peak_centers[0] < high:
                fit = candidate
                break
            if candidate.success:
                reason = "the fitted peak sits outside the q window"
        if fit is None:
            points.append(left_out(reason))
            continue
        q_center = float(fit.peak_centers[0])
        q_err = fit.peak_centers_err[0]
        q_err = float(q_err) if q_err is not None and np.isfinite(q_err) and q_err > 0 else None
        d = 2.0 * np.pi / q_center
        d_err = None if q_err is None else 2.0 * np.pi / q_center ** 2 * q_err
        points.append(StrainPoint(sector.psi, d, d_err, sector.branch, q_center, q_err, None, sector, fit))
        previous = q_center
    return tuple(points)


@dataclass(frozen=True)
class D0Reference:
    """The stress-free spacing a strain is taken against, with where it came from: ``typed``, ``library`` (a phase's
    hkl line: a chosen reference, not a measured stress-free value), ``reference`` (a stress-free result) or
    ``psi_star`` (model-derived from the same fit). ``stderr`` None means its uncertainty is unknown."""

    kind: str
    value: float
    stderr: float | None = None
    label: str = ""

    @classmethod
    def typed(cls, value: float, stderr: float | None = None) -> "D0Reference":
        value = float(value)
        if not (np.isfinite(value) and value > 0):
            raise ValueError("a typed d0 must be a positive spacing in Å")
        return cls("typed", value, None if stderr is None else float(stderr), f"typed d0 {value:.5f} Å")


@dataclass(frozen=True)
class ElasticConstants:
    """A reflection's diffraction elastic constants s1 and ½s2 (1/Pa), with their source."""

    hkl: tuple[int, ...]
    s1: float
    half_s2: float
    source: str = "s1 and ½s2 as given"

    @classmethod
    def from_E_nu(cls, hkl, *, E: float, nu: float) -> "ElasticConstants":
        """The isotropic assumption: s1 = −ν/E, ½s2 = (1 + ν)/E, labelled as such for this hkl."""
        E, nu = float(E), float(nu)
        if not (np.isfinite(E) and E > 0 and -1.0 < nu < 0.5):
            raise ValueError("E must be positive and nu between -1 and 0.5")
        return cls(tuple(int(index) for index in hkl), -nu / E, (1.0 + nu) / E,
                   f"isotropic, from E {E:.4g} Pa and nu {nu:g}")


def d0_from_library(entry, hkl, *, wavelength_angstrom: float) -> D0Reference:
    """d0 from a phase-library entry's line ``hkl`` at the record's wavelength (a calibrant carries no hkl)."""
    from xrd_tools.analysis.phase_references import prepare_phase_models

    if getattr(entry, "source_kind", None) == "calibrant":
        raise ValueError("a calibrant line carries no hkl to choose: pick a CIF or lattice phase")
    wanted = tuple(int(index) for index in hkl)
    _, (reference,) = prepare_phase_models(entry, wavelength_angstrom=float(wavelength_angstrom), min_intensity=0)
    exact = [line for line in reference.reflections if wanted in tuple(tuple(h) for h in line.hkls)]
    family = sorted(abs(index) for index in wanted)
    similar = [line for line in reference.reflections
               if any(sorted(abs(index) for index in h) == family for h in line.hkls)]
    matches = exact or (similar if len(similar) == 1 else [])
    text = "".join(str(index) for index in wanted)
    if not matches:
        raise ValueError(f"{entry.name} lists no {text} line at this wavelength")
    return D0Reference(
        "library", float(matches[0].d_spacing), None,
        f"{entry.name} {text} from the phase library at {float(wavelength_angstrom):.5f} Å "
        "(a chosen reference, not a measured stress-free value)")


def d0_from_reference(points) -> D0Reference:
    """d0 as a stress-free result's weighted mean spacing (the plain mean when its points carry no errors)."""
    valid = [point for point in points if point.excluded is None and np.isfinite(point.d)]
    if not valid:
        raise ValueError("the stress-free reference has no usable points")
    d = np.array([point.d for point in valid])
    errors = [point.d_err for point in valid]
    if all(error is not None and np.isfinite(error) and error > 0 for error in errors):
        weights = np.array([1.0 / error ** 2 for error in errors])
        value, stderr = float(np.sum(weights * d) / np.sum(weights)), float(1.0 / np.sqrt(np.sum(weights)))
    else:
        value = float(d.mean())
        stderr = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else None
    return D0Reference("reference", value, stderr,
                       f"stress-free reference: mean d of {len(valid)} points")


@dataclass(frozen=True, eq=False)
class StrainFit:
    """d against sin²ψ: ``slope`` and ``intercept`` with their covariance ``cov`` in (slope, intercept) order,
    unscaled from the points' errors when ``weighted``, else residual-scaled. ``admitted`` holds the points the fit
    used. The intercept is never a stress-free d0. ``strain`` is (d − d0)/d0 for every given point, only against an
    independent d0. ``stress`` (Pa) is the equibiaxial in-plane stress m/(½s2·d0)."""

    slope: float
    intercept: float
    cov: tuple[tuple[float, float], tuple[float, float]]
    weighted: bool
    n_used: int
    admitted: tuple[StrainPoint, ...] = field(repr=False)
    notice: str
    r_squared: float
    reading: str
    d0: D0Reference | None = None
    strain: tuple[float, ...] | None = None
    stress: float | None = None
    stress_err: float | None = None
    sin2psi_star: float | None = None
    shear: float | None = None
    elastic: ElasticConstants | None = None

    @property
    def slope_err(self) -> float:
        return float(np.sqrt(self.cov[0][0]))

    @property
    def intercept_err(self) -> float:
        return float(np.sqrt(self.cov[1][1]))


def _error_given(error) -> bool:
    return error is not None and np.isfinite(error) and error > 0


def strain_regression(points, *, d0=None, elastic: ElasticConstants | None = None,
                      stress_state: str = "equibiaxial", shear: bool = False) -> StrainFit:
    """Fit d = intercept + slope·sin²ψ over the usable points.

    With three or more points carrying a finite, positive error the fit is weighted (1/σ²) on those points, naming
    any others; otherwise it is unweighted over every usable point, with a notice. It needs three points and two
    distinct ψ. ``d0`` is a :class:`D0Reference`, ``"psi_star"`` (model-derived from this fit, equibiaxial only) or
    None. Uniaxial stress and a shear term are refused until a geometry model gives each point its in-plane direction.
    """
    if shear:
        raise ValueError("a shear term " + _GEOMETRY_REFUSAL)
    if stress_state != "equibiaxial":
        raise ValueError(f"{stress_state} stress " + _GEOMETRY_REFUSAL)
    points = tuple(points)
    usable = [point for point in points
              if point.excluded is None and np.isfinite(point.psi) and np.isfinite(point.d)]
    with_errors = [point for point in usable if _error_given(point.d_err)]
    notes: list[str] = []
    if len(with_errors) >= 3:
        admitted, weighted = with_errors, True
        missing = [point for point in usable if not _error_given(point.d_err)]
        if missing:
            notes.append("left out of the weighted fit, no error: "
                         + ", ".join(f"psi {point.psi:.1f} deg" for point in missing))
    else:
        admitted, weighted = usable, False
        if len(usable) >= 3:
            notes.append("unweighted fit: fewer than three points carry an error")
    if len(admitted) < 3:
        raise ValueError(f"a sin²ψ fit needs at least three points; {len(admitted)} are usable")
    x = np.array([point.sin2psi for point in admitted])
    y = np.array([point.d for point in admitted])
    if np.unique(np.round(x, 12)).size < 2:
        raise ValueError("the points need at least two distinct psi values (rank 2)")
    design = np.column_stack([x, np.ones_like(x)])
    weights = (np.array([1.0 / point.d_err ** 2 for point in admitted]) if weighted else np.ones_like(x))
    inverse = np.linalg.inv(design.T @ (weights[:, None] * design))
    beta = inverse @ (design.T @ (weights * y))
    residual = y - design @ beta
    ss_res = float(np.sum(weights * residual ** 2))
    cov = inverse if weighted else inverse * (ss_res / (len(admitted) - 2))
    mean = float(np.sum(weights * y) / np.sum(weights))
    ss_tot = float(np.sum(weights * (y - mean) ** 2))
    slope, intercept = float(beta[0]), float(beta[1])
    star = None if elastic is None else -2.0 * elastic.s1 / elastic.half_s2
    reference, strain, stress, stress_err = None, None, None, None
    if isinstance(d0, str):
        if d0 != "psi_star":
            raise ValueError("d0 is a D0Reference, 'psi_star' or None")
        if elastic is None:
            raise ValueError("psi* needs the reflection's elastic constants")
        value = intercept + star * slope
        gradient = np.array([star, 1.0])
        reference = D0Reference(
            "psi_star", value, float(np.sqrt(gradient @ cov @ gradient)),
            f"psi* d0 from this fit, equibiaxial (sin²psi* = {star:.4f}); model-derived, not independent")
        stress = slope / (elastic.half_s2 * value)
        gradient = np.array([intercept, -slope]) / (elastic.half_s2 * value ** 2)
        stress_err = float(np.sqrt(gradient @ cov @ gradient))
    elif d0 is not None:
        if not isinstance(d0, D0Reference):
            raise TypeError("d0 is a D0Reference, 'psi_star' or None")
        reference = d0
        strain = tuple(float((point.d - d0.value) / d0.value) for point in points)
        if elastic is not None:
            stress = slope / (elastic.half_s2 * d0.value)
            variance = cov[0, 0] / (elastic.half_s2 * d0.value) ** 2
            if d0.stderr:
                variance += (stress * d0.stderr / d0.value) ** 2
            else:
                notes.append("the stress error leaves out d0's uncertainty, which is not given")
            stress_err = float(np.sqrt(variance))
    sign = slope if stress is None else stress
    return StrainFit(
        slope=slope, intercept=intercept, cov=tuple(tuple(float(v) for v in row) for row in cov),
        weighted=weighted, n_used=len(admitted), admitted=tuple(admitted), notice="; ".join(notes),
        r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        reading="tensile" if sign > 0 else "compressive" if sign < 0 else "none",
        d0=reference, strain=strain, stress=stress, stress_err=stress_err, sin2psi_star=star, elastic=elastic)


def combined_stress(fits) -> tuple[float, float]:
    """Several reflections' stresses as one inverse-variance mean and its error. It assumes independent reflection
    errors and one stress model; shared calibration, reference or model errors do not shrink with the count."""
    fits = tuple(fits)
    if not fits or any(fit.stress is None or not _error_given(fit.stress_err) for fit in fits):
        raise ValueError("every reflection needs a stress and its error to combine")
    weights = np.array([1.0 / fit.stress_err ** 2 for fit in fits])
    values = np.array([fit.stress for fit in fits])
    return float(np.sum(weights * values) / np.sum(weights)), float(1.0 / np.sqrt(np.sum(weights)))


@dataclass(frozen=True)
class RefractionEstimate:
    """How much refraction alone would move the fitted slope (Å), on the fit's own points and weights; ``apparent``
    when the stress should be read as apparent. ``slope`` is None when it cannot be estimated (``notice`` says why).
    No correction is applied: that is C3."""

    slope: float | None
    apparent: bool
    alpha_c_deg: float | None
    min_exit_deg: float | None
    notice: str


def refraction_estimate(points, fit: StrainFit, *, incident_angle_deg: float, energy_eV: float, material,
                        density_kg_m3: float | None = None, ratio: float = 0.10,
                        critical_factor: float = 3.0) -> RefractionEstimate:
    """Refraction's change of the sin²ψ slope through the shared GI correction owner, on the points ``fit`` admitted.

    Apparent when the incidence or any point's exit angle is under ``critical_factor``·α_c, or when the change is at
    least ``ratio`` of the measured slope (screening thresholds, not physical bounds). An unknown material or a ray
    outside the reflection geometry gives an apparent stress with no estimate.
    """
    from xrd_tools.corrections.grazing import GICorrectionStack

    if not material:
        return RefractionEstimate(None, True, None, None,
                                  "refraction unknown: no optical material chosen; stress is apparent")
    try:
        stack = GICorrectionStack(material=material, energy_eV=float(energy_eV), density_kg_m3=density_kg_m3)
        constants = stack.optical_constants()
    except Exception as exc:  # noqa: BLE001 - an unknown medium is a reason, never a guess
        return RefractionEstimate(None, True, None, None,
                                  f"refraction unknown: no optical material {material!r} ({exc}); stress is apparent")
    alpha_c = float(constants["ac_rad"])
    alpha_c_deg = float(np.rad2deg(alpha_c))
    incidence = np.deg2rad(float(incident_angle_deg))
    chosen = fit.admitted
    psi = np.deg2rad(np.array([point.psi for point in chosen]))
    d = np.array([point.d for point in chosen])
    q = 2.0 * np.pi / d
    q_z = q * np.cos(psi)
    sin_exit = q_z / float(constants["k0_per_A"]) - np.sin(incidence)
    if not np.all(np.isfinite(sin_exit)) or np.any(sin_exit < 0) or np.any(sin_exit >= 1):
        return RefractionEstimate(None, True, alpha_c_deg, None,
                                  "refraction unknown: a point's exit ray lies outside the reflection geometry "
                                  "(below the sample horizon); stress is apparent")
    exit_angle = np.arcsin(sin_exit)
    corrected = stack.refract_q(incident_angle_deg=float(incident_angle_deg), alpha_f_rad=exit_angle,
                                q_total=q, q_z=q_z)
    delta = 2.0 * np.pi / corrected - d
    x = np.sin(psi) ** 2
    weights = np.array([1.0 / point.d_err ** 2 for point in chosen]) if fit.weighted else np.ones_like(x)
    design = np.column_stack([x, np.ones_like(x)])
    slope = float(np.linalg.solve(design.T @ (weights[:, None] * design), design.T @ (weights * delta))[0])
    reasons = []
    if incidence < critical_factor * alpha_c:
        reasons.append(f"the incidence {np.rad2deg(incidence):.2f} deg is under {critical_factor:g} alpha_c "
                       f"({critical_factor * alpha_c_deg:.2f} deg)")
    lowest = float(np.min(exit_angle))
    if lowest < critical_factor * alpha_c:
        reasons.append(f"an exit angle ({np.rad2deg(lowest):.2f} deg) is under {critical_factor:g} alpha_c")
    if abs(slope) >= ratio * abs(fit.slope) and not (slope == 0.0 and fit.slope == 0.0):
        reasons.append(f"refraction alone moves the slope by {slope:.2e} A, at least {ratio:.0%} of it")
    notice = ("apparent stress: " + "; ".join(reasons)) if reasons else (
        f"refraction alone moves the slope by {slope:.2e} A, under {ratio:.0%} of it")
    return RefractionEstimate(slope, bool(reasons), alpha_c_deg, float(np.rad2deg(lowest)), notice)


@dataclass(frozen=True, eq=False)
class StrainAnalysisResult:
    """One frame's strain analysis: its sectors, the points fitted from them, the fit over both branches, each
    branch's own descriptive fit (no stress claim), and the refraction estimate (None without incidence and
    energy)."""

    sectors: tuple[StrainSector, ...] = field(repr=False)
    points: tuple[StrainPoint, ...] = field(repr=False)
    fit: StrainFit
    branch_fits: "dict[int, StrainFit]" = field(default_factory=dict)
    refraction: RefractionEstimate | None = None


__all__ = [
    "ChiSector",
    "D0Reference",
    "ElasticConstants",
    "PeakFitResult",
    "RefractionEstimate",
    "Sin2PsiResult",
    "StrainAnalysisResult",
    "StrainFit",
    "StrainPoint",
    "StrainSector",
    "combined_stress",
    "d0_from_library",
    "d0_from_reference",
    "extract_chi_sectors",
    "fit_peak_vs_psi",
    "fit_strain_sectors",
    "refraction_estimate",
    "sin2psi_regression",
    "sin2psi_analysis",
    "strain_regression",
    "strain_sectors",
]
