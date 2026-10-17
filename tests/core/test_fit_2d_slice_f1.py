"""FIT-TOOLS-2026-10 oracle row 3 (F1): the existing 2-D slice fitter and its models.

The truth below is written from the profile definitions (half height at the stated half width), independent of
``xrd_tools.analysis.fitting.models``. Noiseless recovery tolerances (frozen): centres within 0.05 bin, FWHM within
1 %, evaluated-surface RMS within 1e-4 of the dynamic range.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from xrd_tools.analysis.fitting.fit import fit_2d_slice
from xrd_tools.analysis.fitting.models import (
    Gaussian2DModel, LorentzianSquared2DModel, PseudoVoigt2DModel, gauss_2D, lor2_2D, pvoigt_2D,
)

LN2 = np.log(2.0)
# A non-square 11 x 13 grid with unequal spacings; data order is (x, y): slice[i, j] sits at (X[i], Y[j]).
X = -0.9 + 0.2 * np.arange(11)
Y = 2.0 + 0.15 * np.arange(13)
DX, DY = 0.2, 0.15
CX, CY = -0.9 + 0.2 * 4.3, 2.0 + 0.15 * 6.6
FX, FY = 0.5, 0.4  # full widths at half maximum
HEIGHT = 7.0


def _r2(gx, gy):
    """Squared distance in half-widths: 1 on the half-height contour."""
    return ((gx - CX) / (FX / 2)) ** 2 + ((gy - CY) / (FY / 2)) ** 2


def _truth(profile: str, fraction: float = 0.0) -> np.ndarray:
    gx, gy = np.meshgrid(X, Y, indexing="ij")
    r2 = _r2(gx, gy)
    gaussian = HEIGHT * np.exp(-LN2 * r2)                   # = HEIGHT/2 at r2 = 1
    lorentzian2 = HEIGHT / (1.0 + (np.sqrt(2.0) - 1.0) * r2) ** 2  # = HEIGHT/2 at r2 = 1
    return {"gaussian": gaussian, "lorentzian2": lorentzian2,
            "pvoigt": (1.0 - fraction) * gaussian + fraction * lorentzian2}[profile]


def _assert_recovered(result, residuals, truth, data):
    params = result.params
    assert abs(params["centerx"].value - CX) <= 0.05 * DX
    assert abs(params["centery"].value - CY) <= 0.05 * DY
    assert params["fwhmx"].value == pytest.approx(FX, rel=1e-2)
    assert params["fwhmy"].value == pytest.approx(FY, rel=1e-2)
    gx, gy = np.meshgrid(X, Y, indexing="ij")
    surface = result.eval(x=gx, y=gy)  # the model on the full grid, invalid samples included
    assert surface.shape == truth.shape
    rms = np.sqrt(np.mean((surface - truth) ** 2))
    assert rms <= 1e-4 * (truth.max() - truth.min())
    valid = np.isfinite(data)
    assert residuals.shape == data.shape
    np.testing.assert_array_equal(np.isfinite(residuals), valid)
    np.testing.assert_allclose(residuals[valid], (data - surface)[valid], rtol=0, atol=1e-12)


@pytest.mark.parametrize("nan_corner", [False, True], ids=["finite", "nan-corner"])
def test_default_gaussian_fits_an_11x13_slice_and_keeps_its_amplitude_meaning(nan_corner):
    data = _truth("gaussian")
    if nan_corner:
        data[0, 0] = np.nan
    result, residuals = fit_2d_slice(data, X, Y)
    _assert_recovered(result, residuals, _truth("gaussian"), data)
    sigma_x, sigma_y = FX / (2 * np.sqrt(2 * LN2)), FY / (2 * np.sqrt(2 * LN2))
    # the default (lmfit) Gaussian keeps amplitude = integrated volume, height reported apart
    assert result.params["amplitude"].value == pytest.approx(HEIGHT * 2 * np.pi * sigma_x * sigma_y, rel=1e-2)
    assert result.params["height"].value == pytest.approx(HEIGHT, rel=1e-2)
    if nan_corner:
        assert np.isnan(residuals[0, 0])


@pytest.mark.parametrize(("model", "profile", "fraction"), [
    ("lorentzian2d", "lorentzian2", 0.0),
    ("lor2_2d", "lorentzian2", 0.0),
    ("pvoigt2d", "pvoigt", 0.25),
    ("pvoigt2d", "pvoigt", 0.75),
])
@pytest.mark.parametrize("nan_corner", [False, True], ids=["finite", "nan-corner"])
def test_custom_models_recover_independent_half_height_widths(model, profile, fraction, nan_corner):
    truth = _truth(profile, fraction)
    data = truth.copy()
    if nan_corner:
        data[0, 0] = np.nan
    result, residuals = fit_2d_slice(data, X, Y, model=model)
    _assert_recovered(result, residuals, truth, data)
    assert result.params["amplitude"].value == pytest.approx(HEIGHT, rel=1e-2)  # custom models: amplitude = height
    if model == "pvoigt2d":
        assert result.params["fraction"].value == pytest.approx(fraction, abs=1e-3)


def _half_height_full_width(function, **params) -> float:
    """Twice the distance from the centre to the half-height crossing along x (independent root find)."""
    peak = function(0.0, 0.0, **params)

    def excess(x):
        return function(x, 0.0, **params) - peak / 2.0

    return 2.0 * brentq(excess, 0.0, 50.0 * params["sigmax"], xtol=1e-14, rtol=1e-14)


@pytest.mark.parametrize(("cls", "function", "extra"), [
    (Gaussian2DModel, gauss_2D, {}),
    (LorentzianSquared2DModel, lor2_2D, {}),
    (PseudoVoigt2DModel, pvoigt_2D, {"fraction": 0.0}),
    (PseudoVoigt2DModel, pvoigt_2D, {"fraction": 0.4}),
    (PseudoVoigt2DModel, pvoigt_2D, {"fraction": 1.0}),
])
def test_reported_fwhm_is_the_half_height_width(cls, function, extra):
    params = dict(amplitude=3.0, centerx=0.0, centery=0.0, sigmax=0.7, sigmay=1.3, **extra)
    model = cls()
    values = model.make_params(**params)
    for name, value in extra.items():
        values[name].set(value=value)
    values.update_constraints()
    # the FWHM expressions carry seven decimals (lmfit's own fwhm_expr convention)
    assert values["fwhmx"].value == pytest.approx(_half_height_full_width(function, **params), rel=1e-7)
    assert values["fwhmx"].value / params["sigmax"] == pytest.approx(
        {Gaussian2DModel: 2.0 * np.sqrt(2.0 * LN2),                       # 2.354820 sigma
         LorentzianSquared2DModel: 2.0 * np.sqrt(np.sqrt(2.0) - 1.0),     # the squared-Lorentzian factor kept
         PseudoVoigt2DModel: 2.0}[cls], rel=1e-7)                          # common HWHM = sigma


def test_preserved_signature_and_refusals():
    data = _truth("gaussian")
    result, residuals = fit_2d_slice(data, X, Y, "gaussian2d")  # the original positional call
    assert residuals.shape == (11, 13)
    with pytest.raises(ValueError, match="not supported"):
        fit_2d_slice(data, X, Y, model="voigt2d")
    with pytest.raises(ValueError, match="shape"):
        fit_2d_slice(data.T, X, Y)  # displayed (y, x) order is not the fitter's (x, y) data order
    with pytest.raises(ValueError, match="finite"):
        fit_2d_slice(np.full_like(data, np.nan), X, Y)
