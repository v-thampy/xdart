"""
1D and 2D peak fitting using lmfit.

Two fitting modes are provided for 1D data:

**Structure-agnostic** (this module)
    Fit individual peaks without crystal-structure knowledge.  Peak positions
    are either specified manually or found via :func:`extract_peaks`.

**Structure-informed** (``phase_fitting.py``)
    Fit a multi-phase pattern using CIF-derived peak positions / intensities
    from ``PhaseModel`` objects.  See :class:`PhaseFitter`.

Both modes share the same lmfit model zoo and background tools.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from lmfit import Model as LmfitModel
from lmfit.models import (
    GaussianModel,
    LorentzianModel,
    PseudoVoigtModel,
    VoigtModel,
    LinearModel,
    ConstantModel,
    PolynomialModel,
)

from ssrl_xrd_tools.analysis.fitting.models import (
    lorentzian_squared,
    LorentzianSquaredModel,
    Gaussian2DModel,
    LorentzianSquared2DModel,
    PseudoVoigt2DModel,
    ChebyshevModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Peak-model registry
# ---------------------------------------------------------------------------

_PEAK_MODEL_MAP: dict[str, type] = {
    "gaussian": GaussianModel,
    "lorentzian": LorentzianModel,
    "lorentz": LorentzianModel,
    "voigt": VoigtModel,
    "pseudovoigt": PseudoVoigtModel,
    "pvoigt": PseudoVoigtModel,
    "lorentzian_squared": LorentzianSquaredModel,
    "lor2": LorentzianSquaredModel,
}


def get_peak_model(model: str, prefix: str = "") -> LmfitModel:
    """
    Return an lmfit peak Model by name.

    Parameters
    ----------
    model : str
        One of: 'gaussian', 'lorentzian', 'voigt', 'pseudovoigt', 'lorentzian_squared' (or 'lor2').
    prefix : str
        Prefix for parameter names (needed when combining multiple peaks).

    Returns
    -------
    lmfit.Model
    """
    key = model.lower().replace(" ", "").replace("-", "")
    cls = _PEAK_MODEL_MAP.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown peak model {model!r}. "
            f"Choose from: {', '.join(sorted(_PEAK_MODEL_MAP))}"
        )
    return cls(prefix=prefix)


def _get_background_model(
    background: str,
    prefix: str = "bg_",
    x_range: tuple[float, float] | None = None,
) -> LmfitModel | None:
    """Return an lmfit background model.

    Parameters
    ----------
    background : str
        'linear', 'constant', 'chebyshev2', 'chebyshev3', 'chebyshev4',
        'polynomial2' .. 'polynomial7', or 'none'.
    prefix : str
        lmfit parameter prefix.
    x_range : (float, float), optional
        The ``(x_min, x_max)`` of the fit window. Required for
        ``'chebyshev*'`` so coefficients can be normalized to [-1, 1];
        ignored for other background models.
    """
    bg = background.lower().strip()
    if bg == "none":
        return None
    if bg == "linear":
        return LinearModel(prefix=prefix)
    if bg == "constant":
        return ConstantModel(prefix=prefix)
    # Chebyshev / polynomial of degree N — extract trailing digits
    import re
    if bg.startswith("chebyshev") or bg.startswith("cheb"):
        m = re.search(r'\d+', bg)
        degree = int(m.group()) if m else 3
        if x_range is None:
            raise ValueError(
                "Chebyshev backgrounds require x_range=(x_min, x_max) "
                "so coefficients can be mapped to [-1, 1]."
            )
        return ChebyshevModel(degree=degree, x_range=x_range, prefix=prefix)
    if bg.startswith("poly"):
        m = re.search(r'\d+', bg)
        degree = int(m.group()) if m else 2
        return PolynomialModel(degree, prefix=prefix)
    raise ValueError(f"Unknown background model: {background!r}")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PeakFitResult1D:
    """Result of a multi-peak 1D fit.

    Attributes
    ----------
    fit_result : lmfit.model.ModelResult
        The raw lmfit result (access ``params``, ``best_fit``, ``residual``, etc.).
    peak_centers : list of float
        Fitted centre positions for each peak.
    peak_centers_err : list of float
        Uncertainties on peak centres.
    peak_sigmas : list of float
        Fitted sigma (width) for each peak.
    peak_amplitudes : list of float
        Fitted amplitudes for each peak.
    n_peaks : int
        Number of peaks in the model.
    model_name : str
        Name of the peak profile used.
    background_name : str
        Name of the background model used.
    """
    fit_result: Any = field(repr=False)
    peak_centers: list[float] = field(default_factory=list)
    peak_centers_err: list[float] = field(default_factory=list)
    peak_sigmas: list[float] = field(default_factory=list)
    peak_amplitudes: list[float] = field(default_factory=list)
    n_peaks: int = 0
    model_name: str = ""
    background_name: str = ""

    @property
    def success(self) -> bool:
        return self.fit_result.success

    @property
    def best_fit(self) -> np.ndarray:
        return self.fit_result.best_fit

    @property
    def params(self):
        return self.fit_result.params


# ---------------------------------------------------------------------------
# Main 1D fitting function
# ---------------------------------------------------------------------------

def fit_peaks( #TODO - this should include ability to fix certain parameters
    x: np.ndarray,
    y: np.ndarray,
    positions: list[float] | np.ndarray | None = None,
    model: str = "pseudovoigt",
    n_peaks: int | None = None,
    background: str = "linear",
    sigma_init: float | list[float] | None = None,
    sigma_bounds: tuple[float, float] | None = None,
    amplitude_init: float | list[float] | None = None,
    amplitude_bounds: tuple[float, float] | None = None,
    center_bounds_delta: float | None = None,
    fraction_init: float = 0.5,
    **fit_kwargs: Any,
) -> PeakFitResult1D:
    """
    Fit one or more peaks in a 1D pattern.

    This is the main structure-agnostic fitting function.  Peak positions
    can be specified explicitly (from manual inspection or a peak-finder)
    or auto-estimated.

    Parameters
    ----------
    x, y : ndarray
        1D data (e.g. q vs intensity).
    positions : list of float or None
        Initial peak centre positions.  If provided, ``n_peaks`` is inferred.
        If *None*, peaks are auto-estimated from the data.
    model : str
        Peak profile: 'gaussian', 'lorentzian', 'voigt', 'pseudovoigt',
        'lorentzian_squared'.
    n_peaks : int or None
        Number of peaks (required if *positions* is None).  Ignored if
        *positions* is given.
    background : str
        Background model: 'linear', 'constant', 'chebyshev2'..'chebyshev4',
        'polynomial2'..'polynomial7', or 'none'.
    sigma_init : float or list of float or None
        Initial sigma for each peak.  Scalar → same for all peaks.
        None → auto-estimate from data range.
    sigma_bounds : (min, max) or None
        Bounds on sigma.  None → (0, data_range / 2).
    amplitude_init : float or list of float or None
        Initial amplitude.  None → auto-estimate from max intensity.
    amplitude_bounds : (min, max) or None
        Bounds on amplitude.  None → (0, inf).
    center_bounds_delta : float or None
        Each peak centre is bounded to ± this delta around its initial
        position.  None → no bounds on centres beyond the data range.
    fraction_init : float
        Initial pseudo-Voigt mixing fraction (only for 'pseudovoigt').
    **fit_kwargs
        Passed to ``lmfit.Model.fit()`` (e.g. ``method='leastsq'``).

    Returns
    -------
    PeakFitResult1D
    """
    # Clean NaN / inf
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[valid]
    y = np.asarray(y, dtype=float)[valid]

    if len(x) < 5:
        raise ValueError(f"Too few valid data points ({len(x)}) to fit.")

    # Determine number of peaks
    if positions is not None:
        positions = list(np.asarray(positions, dtype=float).ravel())
        n_peaks = len(positions)
    elif n_peaks is None:
        n_peaks = 1
        positions = [x[np.argmax(y)]]
    else:
        # Auto-space peaks across the data range
        positions = list(np.linspace(x.min(), x.max(), n_peaks + 2)[1:-1])

    # Defaults
    data_range = x.max() - x.min()
    if sigma_init is None:
        sigma_init = data_range / (4 * max(n_peaks, 1))
    if isinstance(sigma_init, (int, float)):
        sigma_init = [float(sigma_init)] * n_peaks

    if sigma_bounds is None:
        sigma_bounds = (1e-6, data_range / 2)

    if amplitude_init is None:
        amplitude_init = float(np.nanmax(y) - np.nanmin(y))
    if isinstance(amplitude_init, (int, float)):
        amplitude_init = [float(amplitude_init)] * n_peaks

    if amplitude_bounds is None:
        amplitude_bounds = (0, None)

    # Build composite model
    bg_model = _get_background_model(
        background,
        prefix="bg_",
        x_range=(float(np.nanmin(x)), float(np.nanmax(x))),
    )
    composite = bg_model

    for i in range(n_peaks):
        prefix = f"p{i}_"
        peak_mod = get_peak_model(model, prefix=prefix)
        composite = peak_mod if composite is None else composite + peak_mod

    if composite is None:
        raise ValueError("No model to fit (n_peaks=0 and background='none').")

    # Build parameters
    pars = composite.make_params()

    # Background init
    if bg_model is not None:
        if "bg_slope" in pars:
            pars["bg_slope"].set(value=0)
            pars["bg_intercept"].set(value=float(np.nanmin(y)))
        if "bg_c" in pars:
            pars["bg_c"].set(value=float(np.nanmin(y)))
        # Polynomial: c0, c1, c2, ...
        for key in pars:
            if key.startswith("bg_c") and key[4:].isdigit():
                pars[key].set(value=0)
        if "bg_c0" in pars:
            pars["bg_c0"].set(value=float(np.nanmin(y)))

    # Peak parameters
    for i in range(n_peaks):
        prefix = f"p{i}_"

        # Centre
        cen = positions[i]
        if center_bounds_delta is not None:
            pars[f"{prefix}center"].set(
                value=cen,
                min=cen - center_bounds_delta,
                max=cen + center_bounds_delta,
            )
        else:
            pars[f"{prefix}center"].set(
                value=cen, min=x.min(), max=x.max(),
            )

        # Sigma
        pars[f"{prefix}sigma"].set(
            value=sigma_init[i],
            min=sigma_bounds[0],
            max=sigma_bounds[1],
        )

        # Amplitude
        amp_min = amplitude_bounds[0] if amplitude_bounds[0] is not None else None
        amp_max = amplitude_bounds[1] if amplitude_bounds[1] is not None else None
        pars[f"{prefix}amplitude"].set(
            value=amplitude_init[i], min=amp_min, max=amp_max,
        )

        # Gamma (Voigt)
        gamma_key = f"{prefix}gamma"
        if gamma_key in pars:
            pars[gamma_key].set(value=sigma_init[i], min=0)

        # Fraction (PseudoVoigt)
        frac_key = f"{prefix}fraction"
        if frac_key in pars:
            pars[frac_key].set(value=fraction_init, min=0, max=1)

    # Fit
    result = composite.fit(y, pars, x=x, **fit_kwargs)

    # Extract results
    centers, centers_err, sigmas, amplitudes = [], [], [], []
    for i in range(n_peaks):
        prefix = f"p{i}_"
        centers.append(result.params[f"{prefix}center"].value)
        centers_err.append(result.params[f"{prefix}center"].stderr or 0.0)
        sigmas.append(result.params[f"{prefix}sigma"].value)
        amplitudes.append(result.params[f"{prefix}amplitude"].value)

    return PeakFitResult1D(
        fit_result=result,
        peak_centers=centers,
        peak_centers_err=centers_err,
        peak_sigmas=sigmas,
        peak_amplitudes=amplitudes,
        n_peaks=n_peaks,
        model_name=model,
        background_name=background,
    )


# ---------------------------------------------------------------------------
# Legacy wrapper (backwards compatible)
# ---------------------------------------------------------------------------

def fit_line_cut(
    axis_vals: np.ndarray,
    intensity: np.ndarray,
    model: str = "gaussian",
    n_peaks: int = 1,
    background: str = "linear",
) -> Any:
    """
    Fit a 1D line cut with lmfit (peak model + optional background).

    This is a simplified wrapper around :func:`fit_peaks` for backwards
    compatibility.  For new code, prefer :func:`fit_peaks` which provides
    more control over initial positions and constraints.

    Parameters
    ----------
    axis_vals : np.ndarray
        1D x values.
    intensity : np.ndarray
        1D intensity.
    model : str, optional
        Peak model: 'gaussian', 'lorentzian', 'voigt', 'pseudovoigt',
        'lorentzian_squared'.
    n_peaks : int, optional
        Number of peaks.
    background : str, optional
        'linear', 'constant', or 'none'.

    Returns
    -------
    lmfit.model.ModelResult
        Fit result (the raw lmfit object for backwards compatibility).
    """
    result = fit_peaks(
        axis_vals, intensity,
        model=model, n_peaks=n_peaks, background=background,
    )
    return result.fit_result


# ---------------------------------------------------------------------------
# 2D fitting (unchanged)
# ---------------------------------------------------------------------------

def fit_2d_slice(
    slice_2d: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    model: str = "gaussian2d",
) -> tuple[Any, np.ndarray]:
    """
    Fit a 2D slice with lmfit (Gaussian2D, Lorentzian2D, or PseudoVoigt2D).

    Parameters
    ----------
    slice_2d : np.ndarray
        2D intensity.
    x_axis, y_axis : np.ndarray
        1D axes (lengths match slice_2d).
    model : str, optional
        'gaussian2d', 'lorentzian2d'/'lor2_2d', or 'pvoigt2d'.

    Returns
    -------
    result : lmfit.model.ModelResult
        Fit result.
    residuals : np.ndarray
        2D residuals.
    """
    from lmfit.models import Gaussian2dModel

    _CUSTOM_MAP = {
        "lorentzian2d": LorentzianSquared2DModel,
        "lor2_2d": LorentzianSquared2DModel,
        "pvoigt2d": PseudoVoigt2DModel,
    }
    _ALL_KEYS = {"gaussian2d"} | set(_CUSTOM_MAP)
    model_key = model.lower().replace(" ", "")
    if model_key not in _ALL_KEYS:
        raise ValueError(
            f"model={model!r} not supported. Choose from: "
            + ", ".join(f"{k!r}" for k in sorted(_ALL_KEYS))
        )

    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
    z_flat = np.asarray(slice_2d, dtype=float).ravel()
    X_flat, Y_flat = X.ravel(), Y.ravel()

    if model_key == "gaussian2d":
        mod = Gaussian2dModel()
        pars = mod.guess(slice_2d, x=X, y=Y)
        result = mod.fit(slice_2d, pars, x=X, y=Y, nan_policy="omit")
        best_fit = np.asarray(result.best_fit)
    else:
        mod = _CUSTOM_MAP[model_key]()
        pars = mod.guess(z_flat, x=X_flat, y=Y_flat)
        result = mod.fit(z_flat, pars, x=X_flat, y=Y_flat, nan_policy="omit")
        best_fit = result.best_fit.reshape(slice_2d.shape)

    residuals = np.asarray(slice_2d, dtype=float) - best_fit
    return result, residuals
