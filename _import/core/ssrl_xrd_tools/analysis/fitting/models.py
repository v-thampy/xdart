"""1D/2D peak models and helpers for lmfit."""
from __future__ import annotations

from typing import Any

import numpy as np
from lmfit import Model
from lmfit.models import fwhm_expr, update_param_vals, gaussian, lorentzian
from scipy.special import erf


def index_of(arr, val):
    """Return index of array nearest to a value."""
    if val < np.min(arr):
        return 0
    return np.abs(arr - val).argmin()


def _fwhm_expr_2D(model, parameter="sigma"):
    """Return constraint expression for fwhm."""
    return "%.7f*%s%s" % (model.fwhm_factor, model.prefix, parameter)


def guess_from_peak(model, y, x, negative, ampscale=1.0, sigscale=1.0, amp_area=True):
    """Estimate starting parameters for 1D peak fits.

    The parameters are: Amplitude (can be area or peak, see amp_area),
    Center, Sigma.
    """
    if x is None:
        return model.make_params(amplitude=1.0, center=0.0, sigma=1.0)
    maxy, miny = max(y), min(y)
    maxx, minx = max(x), min(x)
    imaxy = index_of(y, maxy)

    amp = maxy - (y[0] + y[-1]) / 2.0
    cen = x[imaxy]
    sig = (maxx - minx) / 6.0

    halfmax_vals = np.where(y > (maxy + miny) / 2.0)[0]
    if negative:
        imaxy = index_of(y, miny)
        amp = -(maxy - miny) * 2.0
        halfmax_vals = np.where(y < (maxy + miny) / 2.0)[0]
    if len(halfmax_vals) > 2:
        sig = (x[halfmax_vals[-1]] - x[halfmax_vals[0]]) / 2.0
        cen = x[halfmax_vals].mean()
    amp = amp * ampscale
    if amp_area:
        amp *= sig * 2.0
    sig = sig * sigscale

    pars = model.make_params(amplitude=amp, center=cen, sigma=sig)
    key = "%ssigma" % model.prefix
    if key in pars:
        pars[key].set(min=0.0)
    return pars


def guess_from_peak_2D(model, data, x, y, **kwargs):
    """Estimate starting parameters for 2D peak fits. Delegates to _guess_2d_params."""
    return _guess_2d_params(model, data, x, y, **kwargs)


def update_param_hints(pars, **kwargs):
    """Update parameter hints with keyword arguments."""
    for pname, hints in kwargs.items():
        if pname in pars:
            for hint, val in hints.items():
                if val is not None:
                    setattr(pars[pname], hint, val)
    return pars


def _guess_2d_params(model, data, x, y, **kwargs):
    """Delegate 2D parameter guessing to lmfit.models.Gaussian2dModel."""
    from lmfit.models import Gaussian2dModel
    g_pars = Gaussian2dModel().guess(data, x, y)
    pars = model.make_params(
        amplitude=g_pars["amplitude"].value,
        centerx=g_pars["centerx"].value,
        centery=g_pars["centery"].value,
        sigmax=g_pars["sigmax"].value,
        sigmay=g_pars["sigmay"].value,
    )
    return update_param_vals(pars, model.prefix, **kwargs)


COMMON_DOC = """

Parameters
----------
independent_vars: list of strings to be set as variable names
missing: None, 'drop', or 'raise'
prefix: string to prepend to parameter names, needed to add two Models that
    have parameter names in common. None by default.
"""

# --- 1D functions ---

def lorentzian_squared(x, amplitude=1.0, center=0.0, sigma=1.0):
    r"""
    Lorentzian squared: amplitude*(1/(1 +((x - center)/sigma)**2))**2
    HWHM = sqrt(sqrt(2)-1)*sigma
    """
    return amplitude * (1 / (1 + ((x - center) / sigma) ** 2)) ** 2


def pvoigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5):
    """
    1D pseudo-Voigt: (1-fraction)*gaussian + fraction*lorentzian, same FWHM.
    sigma_g = sigma (Gaussian and Lorentzian share FWHM 2*sigma).
    """
    sigma_g = sigma
    return (1 - fraction) * gaussian(x, amplitude, center, sigma_g) + fraction * lorentzian(
        x, amplitude, center, sigma
    )


def assymetric_rectangle(
    x,
    amplitude1=1.0,
    center1=0.0,
    sigma1=1.0,
    amplitude2=1.0,
    center2=1.0,
    sigma2=1.0,
    form="linear",
):
    """
    Step-up and step-down function. form: 'linear', 'erf', 'atan'/'arctan', 'logistic'.
    """
    if abs(sigma1) < 1.0e-13:
        sigma1 = 1.0e-13
    if abs(sigma2) < 1.0e-13:
        sigma2 = 1.0e-13

    arg1 = (x - center1) / sigma1
    arg2 = (center2 - x) / sigma2
    if form == "erf":
        out = 0.5 * (amplitude1 * (erf(arg1) + 1) + amplitude2 * (erf(arg2) + 1))
    elif form.startswith("logi"):
        out = 0.5 * (
            amplitude1 * (1.0 - 1.0 / (1.0 + np.exp(arg1)))
            + amplitude2 * (1.0 - 1.0 / (1.0 + np.exp(arg2)))
        )
    elif form in ("atan", "arctan"):
        out = (amplitude1 * np.arctan(arg1) + amplitude2 * np.arctan(arg2)) / np.pi
    else:
        arg1 = np.clip(arg1, 0.0, 1.0)
        arg2 = np.clip(arg2, -1.0, 0.0)
        out = amplitude1 * arg1 + amplitude2 * arg2
    return out


# --- 2D functions ---

def plane(x, y, intercept=0.0, slope_x=0.0, slope_y=0.0):
    """2D plane: intercept + slope_x*x + slope_y*y."""
    return intercept + slope_x * x + slope_y * y


def lor2_2D(x, y, amplitude=1.0, centerx=0.0, centery=0.0, sigmax=1.0, sigmay=1.0):
    r"""2D Lorentzian squared. HWHM = sqrt(sqrt(2)-1)*sigma."""
    return amplitude * (
        1
        / (1 + ((x - centerx) / sigmax) ** 2 + ((y - centery) / sigmay) ** 2)
    ) ** 2


def gauss_2D(x, y, amplitude=1.0, centerx=0.0, centery=0.0, sigmax=1.0, sigmay=1.0):
    """2D Gaussian by amplitude."""
    return amplitude * np.exp(
        -1.0 * (x - centerx) ** 2 / (2 * sigmax**2)
        - 1.0 * (y - centery) ** 2 / (2 * sigmay**2)
    )


def pvoigt_2D(
    x, y, amplitude=1.0, centerx=0.0, centery=0.0, sigmax=1.0, sigmay=1.0, fraction=0.5
):
    """2D pseudo-Voigt: (1-fraction)*Gaussian2D + fraction*LorentzianSquared2D, same FWHM."""
    sigmax_g = sigmax / 1.17741
    sigmay_g = sigmay / 1.17741
    return (1 - fraction) * gauss_2D(
        x, y, amplitude, centerx, centery, sigmax_g, sigmay_g
    ) + fraction * lor2_2D(x, y, amplitude, centerx, centery, sigmax, sigmay)


# --- 1D model classes ---

class LorentzianSquaredModel(Model):
    r"""Lorentzian squared 1D. HWHM = sqrt(sqrt(2)-1)*sigma."""
    __doc__ = (lorentzian_squared.__doc__ or "") + COMMON_DOC
    fwhm_factor = 2.0 * np.sqrt(np.sqrt(2) - 1)

    def __init__(self, *args, **kwargs):
        super().__init__(lorentzian_squared, *args, **kwargs)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("fwhm", expr=fwhm_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative, ampscale=0.5, amp_area=False)
        return update_param_vals(pars, self.prefix, **kwargs)


class PseudoVoigtModel(Model):
    """1D pseudo-Voigt (Gaussian + Lorentzian mix, same FWHM)."""
    __doc__ = (pvoigt.__doc__ or "") + COMMON_DOC
    fwhm_factor = 2.0

    def __init__(self, *args, **kwargs):
        super().__init__(pvoigt, *args, **kwargs)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("fraction", min=0, max=1, value=0.5)
        self.set_param_hint("fwhm", expr=fwhm_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative, ampscale=0.5, amp_area=False)
        if f"{self.prefix}fraction" in pars:
            pars[f"{self.prefix}fraction"].set(value=0.5, min=0, max=1)
        return update_param_vals(pars, self.prefix, **kwargs)


class AssymetricRectangleModel(Model):
    r"""Step-up and step-down with center1, center2, sigma1, sigma2, form."""

    def __init__(self, independent_vars=("x",), prefix="", missing=None, name=None, **kwargs):
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": list(independent_vars)})
        super().__init__(assymetric_rectangle, **kwargs)
        self.set_param_hint("center1")
        self.set_param_hint("center2")
        self.set_param_hint(
            "midpoint",
            expr="(%scenter1+%scenter2)/2.0" % (self.prefix, self.prefix),
        )

    def guess(self, data, x=None, negative=False, **kwargs):
        if x is None:
            return self.make_params()
        if negative:
            data = np.asarray(data) * -1
        ymin1 = np.min(data[: len(data) // 4])
        ymin2 = np.min(data[3 * (len(data) // 4) :])
        ymax = np.max(data)
        xmin1, xmin2, xmax = np.min(x), np.min(x), np.max(x)
        if negative:
            data = data * -1
            ymin1, ymin2, ymax = -ymin1, -ymin2, -ymax
        pars = self.make_params(
            amplitude1=(ymax - ymin1),
            amplitude2=(ymax - ymin2),
            center1=xmin1 + (xmax - xmin1) / 5,
            center2=xmin1 + (xmax - xmin1) / 2,
        )
        pars[f"{self.prefix}sigma1"].set(value=abs(xmax - xmin1) / 7.0, min=0.0)
        pars[f"{self.prefix}sigma2"].set(value=abs(xmax - xmin2) / 7.0, min=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)


# --- 2D model classes ---

class Gaussian2DModel(Model):
    """2D Gaussian (amplitude, centerx, centery, sigmax, sigmay)."""
    __doc__ = (gauss_2D.__doc__ or "") + COMMON_DOC
    fwhm_factor = 2.0 * np.sqrt(np.log(2))

    def __init__(self, independent_vars=("x", "y"), prefix="", **kwargs):
        kwargs.update({"prefix": prefix, "independent_vars": list(independent_vars)})
        super().__init__(gauss_2D, **kwargs)
        self.set_param_hint("sigmax", min=0)
        self.set_param_hint("sigmay", min=0)
        self.set_param_hint("fwhmx", expr=_fwhm_expr_2D(self, parameter="sigmax"))
        self.set_param_hint("fwhmy", expr=_fwhm_expr_2D(self, parameter="sigmay"))

    def guess(self, data, x=None, y=None, **kwargs):
        return _guess_2d_params(self, data, x, y, **kwargs)


class LorentzianSquared2DModel(Model):
    r"""2D Lorentzian squared. HWHM = sqrt(sqrt(2)-1)*sigma."""
    __doc__ = (lor2_2D.__doc__ or "") + COMMON_DOC
    fwhm_factor = 2.0 * np.sqrt(np.sqrt(2) - 1)

    def __init__(self, independent_vars=("x", "y"), prefix="", **kwargs):
        kwargs.update({"prefix": prefix, "independent_vars": list(independent_vars)})
        super().__init__(lor2_2D, **kwargs)
        self.set_param_hint("sigmax", min=0)
        self.set_param_hint("sigmay", min=0)
        self.set_param_hint("fwhmx", expr=_fwhm_expr_2D(self, parameter="sigmax"))
        self.set_param_hint("fwhmy", expr=_fwhm_expr_2D(self, parameter="sigmay"))

    def guess(self, data, x=None, y=None, **kwargs):
        return _guess_2d_params(self, data, x, y, **kwargs)


class Pvoigt2DModel(Model):
    """2D pseudo-Voigt. Parameters: amplitude, centerx, centery, sigmax, sigmay, fraction."""
    __doc__ = (pvoigt_2D.__doc__ or "") + COMMON_DOC
    fwhm_factor = 2.0

    def __init__(self, independent_vars=("x", "y"), prefix="", **kwargs):
        kwargs.update({"prefix": prefix, "independent_vars": list(independent_vars)})
        super().__init__(pvoigt_2D, **kwargs)
        self.set_param_hint("sigmax", min=0)
        self.set_param_hint("sigmay", min=0)
        self.set_param_hint("fraction", min=0, max=1, value=0.5)
        self.set_param_hint("fwhmx", expr=_fwhm_expr_2D(self, parameter="sigmax"))
        self.set_param_hint("fwhmy", expr=_fwhm_expr_2D(self, parameter="sigmay"))

    def guess(self, data, x=None, y=None, **kwargs):
        pars = _guess_2d_params(self, data, x, y, **kwargs)
        if f"{self.prefix}fraction" in pars:
            pars[f"{self.prefix}fraction"].set(value=0.5, min=0, max=1)
        return pars


class PlaneModel(Model):
    """2D plane: intercept + slope_x*x + slope_y*y."""
    __doc__ = (plane.__doc__ or "") + COMMON_DOC

    def __init__(self, independent_vars=("x", "y"), prefix="", **kwargs):
        kwargs.update({"prefix": prefix, "independent_vars": list(independent_vars)})
        super().__init__(plane, **kwargs)

    def guess(self, data, x=None, y=None, **kwargs):
        oval, sxval, syval = 0.0, 0.0, 0.0
        if x is not None and y is not None:
            mask = np.isfinite(data)
            if mask.any():
                sxval, oval = np.polyfit(x[mask], data[mask], 1)
                syval, _ = np.polyfit(y[mask], data[mask], 1)
        pars = self.make_params(intercept=oval, slope_x=sxval, slope_y=syval)
        return update_param_vals(pars, self.prefix, **kwargs)
