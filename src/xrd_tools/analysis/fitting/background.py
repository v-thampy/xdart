"""
Background estimation and subtraction for 1D XRD patterns.

Methods
-------
snip_1d
    Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).
    Good for automatic baseline estimation with no prior knowledge.
chebyshev_background
    Fit a Chebyshev polynomial to the baseline.  Useful as a refinement
    after SNIP pre-subtraction, or on its own for smooth baselines.
"""

from __future__ import annotations

import numpy as np

from ssrl_xrd_tools.core.containers import IntegrationResult1D


def snip_1d(y: np.ndarray, snip_width: int = 20) -> np.ndarray:
    """
    Estimate background using the SNIP (Statistics-sensitive
    Non-linear Iterative Peak-clipping) algorithm.

    Parameters
    ----------
    y : ndarray
        1D input signal.
    snip_width : int
        Number of iterations (width of clipping window).

    Returns
    -------
    ndarray
        Estimated background baseline of the same shape as ``y``.
    """
    bg = np.copy(y)
    n = len(y)
    for p in range(snip_width, 0, -1):
        avg = (bg[:-2*p] + bg[2*p:]) / 2.0
        bg[p:n-p] = np.minimum(bg[p:n-p], avg)
    return bg


def chebyshev_background(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    n_iter: int = 5,
    sigma_clip: float = 2.0,
) -> np.ndarray:
    """
    Estimate background by iteratively fitting a Chebyshev polynomial,
    clipping points that rise above the fit (i.e. peaks).

    Parameters
    ----------
    x : ndarray
        1D x-axis (e.g. q or 2θ).
    y : ndarray
        1D intensity.
    degree : int
        Degree of the Chebyshev polynomial.
    n_iter : int
        Number of sigma-clipping iterations.
    sigma_clip : float
        Points more than ``sigma_clip × std`` above the fit are excluded.

    Returns
    -------
    ndarray
        Background estimate evaluated at ``x``.
    """
    mask = np.isfinite(y) & np.isfinite(x)
    xm, ym = x[mask], y[mask]

    if len(xm) < degree + 1:
        # Not enough valid points for the requested polynomial degree
        return np.full_like(y, np.nan)

    # Iterative sigma-clipping: remove points above the fit (peaks)
    keep = np.ones(len(xm), dtype=bool)
    for _ in range(n_iter):
        if np.sum(keep) < degree + 1:
            break  # Too few points to fit; stop clipping
        coeffs = np.polynomial.chebyshev.chebfit(xm[keep], ym[keep], degree)
        bg_fit = np.polynomial.chebyshev.chebval(xm, coeffs)
        residual = ym - bg_fit
        std = np.std(residual[keep])
        if std < 1e-15:
            break  # Converged; all residuals essentially zero
        keep = residual < sigma_clip * std

    # Final fit with surviving points
    if np.sum(keep) < degree + 1:
        keep = np.ones(len(xm), dtype=bool)  # Fall back to unclipped fit
    coeffs = np.polynomial.chebyshev.chebfit(xm[keep], ym[keep], degree)

    # Evaluate at all original x points
    bg = np.full_like(y, np.nan)
    bg[mask] = np.polynomial.chebyshev.chebval(xm, coeffs)
    return bg


def fit_background(
    result: IntegrationResult1D,
    method: str = "snip",
    **kwargs,
) -> np.ndarray:
    """
    Extract the background from an integration result.

    Parameters
    ----------
    result : IntegrationResult1D
        Input 1D integration result.
    method : str
        Background algorithm: 'snip' or 'chebyshev'.
    **kwargs
        Passed to the underlying background algorithm.
        For 'snip': ``snip_width`` (int, default 20).
        For 'chebyshev': ``degree`` (int, default 3), ``n_iter`` (int, default 5),
        ``sigma_clip`` (float, default 2.0).

    Returns
    -------
    ndarray
        1D array containing the background intensity.
    """
    if method == "snip":
        snip_width = kwargs.get("snip_width", 20)
        return snip_1d(result.intensity, snip_width=snip_width)
    elif method in ("chebyshev", "cheb"):
        return chebyshev_background(result.radial, result.intensity, **kwargs)
    else:
        raise ValueError(f"Unknown background method: {method!r}. Use 'snip' or 'chebyshev'.")


def subtract_background(
    result: IntegrationResult1D,
    method: str = "snip",
    **kwargs,
) -> IntegrationResult1D:
    """
    Subtract background from an integration result and return a new
    IntegrationResult1D object.

    Parameters
    ----------
    result : IntegrationResult1D
        Input 1D integration result.
    method : str
        Background algorithm: 'snip' or 'chebyshev'.
    **kwargs
        Passed to the background algorithm.

    Returns
    -------
    IntegrationResult1D
        New object with the background-subtracted intensity.
    """
    bg = fit_background(result, method=method, **kwargs)
    new_intensity = result.intensity - bg
    return IntegrationResult1D(
        radial=result.radial.copy(),
        intensity=new_intensity,
        sigma=result.sigma.copy() if result.sigma is not None else None,
        unit=result.unit,
    )


__all__ = [
    "chebyshev_background",
    "fit_background",
    "snip_1d",
    "subtract_background",
]
