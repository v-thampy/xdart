"""1D/2D peak fitting using lmfit and ssrl_xrd_tools.analysis.fitting.models."""
from __future__ import annotations

from typing import Any

import numpy as np
from lmfit.models import (
    GaussianModel,
    LorentzianModel,
    VoigtModel,
    LinearModel,
    ConstantModel,
)

from ssrl_xrd_tools.analysis.fitting.models import (
    lorentzian_squared,
    pvoigt,
    Gaussian2DModel,
    LorentzianSquared2DModel,
    Pvoigt2DModel,
)


def _get_peak_model(model: str, prefix: str):
    """Return lmfit peak model for 1D fit. model: gaussian, lorentzian, voigt, pseudovoigt, lorentzian_squared."""
    from lmfit import Model as LmfitModel

    model_lower = model.lower().replace(" ", "")
    if model_lower == "gaussian":
        return GaussianModel(prefix=prefix)
    if model_lower in ("lorentzian", "lorentz"):
        return LorentzianModel(prefix=prefix)
    if model_lower == "voigt":
        return VoigtModel(prefix=prefix)
    if model_lower in ("pseudovoigt", "pvoigt"):
        return LmfitModel(pvoigt, prefix=prefix)
    if model_lower in ("lorentzian_squared", "lor2"):
        return LmfitModel(lorentzian_squared, prefix=prefix)
    return GaussianModel(prefix=prefix)


def fit_line_cut(
    axis_vals: np.ndarray,
    intensity: np.ndarray,
    model: str = "gaussian",
    n_peaks: int = 1,
    background: str = "linear",
) -> Any:
    """
    Fit a 1D line cut with lmfit (peak model + optional background).

    Parameters
    ----------
    axis_vals : np.ndarray
        1D x values.
    intensity : np.ndarray
        1D intensity.
    model : str, optional
        Peak model: 'gaussian', 'lorentzian', 'voigt', 'pseudovoigt', 'lorentzian_squared'.
    n_peaks : int, optional
        Number of peaks.
    background : str, optional
        'linear', 'constant', or 'none'.

    Returns
    -------
    lmfit.model.ModelResult
        Fit result.
    """
    nan_mask = np.isfinite(intensity) & np.isfinite(axis_vals)
    x = np.asarray(axis_vals)[nan_mask]
    y = np.asarray(intensity)[nan_mask]

    if background == "linear":
        mod = LinearModel(prefix="b_")
    elif background == "constant":
        mod = ConstantModel(prefix="b_")
    else:
        mod = None

    for i in range(n_peaks):
        peak = _get_peak_model(model, prefix=f"p{i}_")
        mod = peak if mod is None else mod + peak

    if mod is None:
        raise ValueError(
            "Both n_peaks=0 and background='none' were specified, leaving no model to fit."
        )

    pars = mod.make_params()
    if "b_slope" in pars:
        pars["b_slope"].set(0)
        pars["b_intercept"].set(np.nanmin(y))
    if "b_c" in pars:
        pars["b_c"].set(np.nanmin(y))

    sig0 = (np.max(x) - np.min(x)) / (4 * max(n_peaks, 1))
    for i in range(n_peaks):
        center_key = f"p{i}_center"
        if center_key in pars:
            pars[center_key].set(np.median(x))
        for skey in ("sigma", "sigma_x"):
            if f"p{i}_{skey}" in pars:
                pars[f"p{i}_{skey}"].set(sig0)
        if f"p{i}_amplitude" in pars:
            pars[f"p{i}_amplitude"].set(np.nanmax(y) - np.nanmin(y))
        if f"p{i}_gamma" in pars:
            pars[f"p{i}_gamma"].set(sig0)
        if f"p{i}_fraction" in pars:
            pars[f"p{i}_fraction"].set(0.5, min=0, max=1)

    return mod.fit(y, pars, x=x)


def fit_2d_slice(
    slice_2d: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    model: str = "gaussian2d",
) -> tuple[Any, np.ndarray]:
    """
    Fit a 2D slice with lmfit (Gaussian2d, Lorentzian2d, or PseudoVoigt2d).

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
        "pvoigt2d": Pvoigt2DModel,
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
