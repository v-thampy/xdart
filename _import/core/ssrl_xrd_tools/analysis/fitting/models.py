"""
Custom 1D/2D peak models for lmfit.

Models provided here extend lmfit's built-in set.  For standard profiles
use lmfit directly::

    from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel

Custom 1D models
    LorentzianSquaredModel  — Lorentzian², heavier tails than Voigt
    AsymmetricRectangleModel — step-up / step-down (erf, atan, logistic)
    ChebyshevModel          — Chebyshev-polynomial background on a fixed x-range

Custom 2D models
    Gaussian2DModel, LorentzianSquared2DModel, PseudoVoigt2DModel, PlaneModel
"""
from __future__ import annotations

from typing import Any

import numpy as np
from lmfit import Model
from lmfit.models import fwhm_expr, update_param_vals, gaussian, lorentzian
from scipy.special import erf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def index_of(arr, val):
    """Return index of array nearest to a value."""
    if val < np.min(arr):
        return 0
    return np.abs(arr - val).argmin()


def _fwhm_expr_2D(model, parameter="sigma"):
    """Return constraint expression for FWHM of a 2D model."""
    return "%.7f*%s%s" % (model.fwhm_factor, model.prefix, parameter)


def guess_from_peak(model, y, x, negative, ampscale=1.0, sigscale=1.0, amp_area=True):
    """Estimate starting parameters (amplitude, center, sigma) for 1D peak fits."""
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


# ---------------------------------------------------------------------------
# 1D functions (only those NOT in lmfit)
# ---------------------------------------------------------------------------

def lorentzian_squared(x, amplitude=1.0, center=0.0, sigma=1.0):
    r"""
    Lorentzian squared: amplitude * (1/(1 + ((x - center)/sigma)²))²

    Heavier tails than a standard Lorentzian.
    HWHM = sqrt(sqrt(2) - 1) * sigma.
    """
    return amplitude * (1 / (1 + ((x - center) / sigma) ** 2)) ** 2


def asymmetric_rectangle(
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
    Step-up and step-down function.

    Parameters
    ----------
    form : {'linear', 'erf', 'atan', 'arctan', 'logistic'}
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
    elif form == "linear":
        arg1 = np.clip(arg1, 0.0, 1.0)
        arg2 = np.clip(arg2, -1.0, 0.0)
        out = amplitude1 * arg1 + amplitude2 * arg2
    else:
        raise ValueError(
            f"Unknown form {form!r}. Choose from: 'linear', 'erf', 'atan', 'logistic'"
        )
    return out


# Preserve the old misspelled name for backwards compatibility
assymetric_rectangle = asymmetric_rectangle


# ---------------------------------------------------------------------------
# 2D functions
# ---------------------------------------------------------------------------

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
    # Convert Lorentzian HWHM to Gaussian sigma for equal FWHM: sqrt(2*ln2) ≈ 1.17741
    _FWHM_FACTOR = np.sqrt(2 * np.log(2))
    sigmax_g = sigmax / _FWHM_FACTOR
    sigmay_g = sigmay / _FWHM_FACTOR
    return (1 - fraction) * gauss_2D(
        x, y, amplitude, centerx, centery, sigmax_g, sigmay_g
    ) + fraction * lor2_2D(x, y, amplitude, centerx, centery, sigmax, sigmay)


# ---------------------------------------------------------------------------
# 1D model classes (only those NOT in lmfit)
# ---------------------------------------------------------------------------

class LorentzianSquaredModel(Model):
    r"""Lorentzian squared 1D model. HWHM = sqrt(sqrt(2)-1)*sigma."""
    __doc__ = (lorentzian_squared.__doc__ or "") + COMMON_DOC
    fwhm_factor = 2.0 * np.sqrt(np.sqrt(2) - 1)

    def __init__(self, *args, **kwargs):
        super().__init__(lorentzian_squared, *args, **kwargs)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("fwhm", expr=fwhm_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative, ampscale=0.5, amp_area=False)
        return update_param_vals(pars, self.prefix, **kwargs)


class AsymmetricRectangleModel(Model):
    r"""Step-up and step-down with center1, center2, sigma1, sigma2, form."""

    def __init__(self, independent_vars=("x",), prefix="", missing=None, name=None, **kwargs):
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": list(independent_vars)})
        super().__init__(asymmetric_rectangle, **kwargs)
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


# Backwards-compatible alias for the old misspelled name
AssymetricRectangleModel = AsymmetricRectangleModel


# ---------------------------------------------------------------------------
# Chebyshev polynomial background
# ---------------------------------------------------------------------------

def _build_chebyshev_func(
    degree: int,
    x_min: float,
    x_max: float,
):
    """Generate a bare Python function ``f(x, c0, c1, ..., cN)`` that
    evaluates a Chebyshev polynomial after linearly mapping ``x`` from
    ``[x_min, x_max]`` onto ``[-1, 1]``.

    lmfit's :class:`~lmfit.model.Model` introspects the wrapped function's
    signature to discover its parameter names, so we cannot use ``**kwargs``
    — the coefficient parameters must appear as explicit positional
    arguments. Since ``degree`` is only known at runtime, we ``exec`` the
    function source in a fresh namespace. This is the standard trick for
    building lmfit models with a dynamic number of parameters.
    """
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError(
            f"Invalid x_range for Chebyshev background: ({x_min!r}, {x_max!r})"
        )
    if degree < 0:
        raise ValueError(f"Chebyshev degree must be >= 0, got {degree}")

    span = x_max - x_min
    arg_list = ", ".join(f"c{i}" for i in range(degree + 1))
    ns = {"np": np, "_x_min": float(x_min), "_span": float(span)}
    src = (
        f"def _cheb(x, {arg_list}):\n"
        f"    t = 2.0 * (np.asarray(x, dtype=float) - _x_min) / _span - 1.0\n"
        f"    return np.polynomial.chebyshev.chebval(t, [{arg_list}])\n"
    )
    exec(src, ns)
    return ns["_cheb"]


class ChebyshevModel(Model):
    """lmfit Model for a Chebyshev polynomial background.

    Unlike :class:`~lmfit.models.PolynomialModel`, Chebyshev polynomials
    are orthogonal on [-1, 1], so the fit is much better-conditioned when
    coefficients are allowed to float independently. The independent
    variable ``x`` is linearly mapped from ``x_range`` to [-1, 1] before
    evaluation. ``x_range`` is captured at construction time and held
    fixed; the coefficients ``c0``..``cN`` are the only free parameters.

    Parameters
    ----------
    degree : int
        Polynomial degree. Creates coefficients ``c0``..``c{degree}``.
    x_range : (float, float)
        The ``(x_min, x_max)`` window used for the [-1, 1] mapping.
        Required.
    prefix : str, optional
        lmfit prefix for parameter names (e.g. ``'bg_'``).

    Examples
    --------
    >>> bg = ChebyshevModel(degree=3, x_range=(1.0, 5.0), prefix='bg_')
    >>> params = bg.guess(intensity, x=q)
    """

    def __init__(
        self,
        degree: int,
        x_range: tuple[float, float],
        independent_vars=("x",),
        prefix: str = "",
        **kwargs,
    ):
        if x_range is None:
            raise ValueError(
                "ChebyshevModel requires x_range=(x_min, x_max) so "
                "coefficients can be mapped to [-1, 1]."
            )
        x_min, x_max = float(x_range[0]), float(x_range[1])
        func = _build_chebyshev_func(degree, x_min, x_max)

        # Store for introspection / guessing
        self.degree = int(degree)
        self.x_min = x_min
        self.x_max = x_max

        kwargs.update({
            "prefix": prefix,
            "independent_vars": list(independent_vars),
        })
        super().__init__(func, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Starting guess: ``c0 = mean(data)``, higher coefficients = 0."""
        pars = self.make_params()
        if data is not None and len(np.asarray(data)):
            c0_key = f"{self.prefix}c0"
            if c0_key in pars:
                pars[c0_key].set(value=float(np.nanmean(data)))
            for i in range(1, self.degree + 1):
                key = f"{self.prefix}c{i}"
                if key in pars:
                    pars[key].set(value=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)


def make_chebyshev_model(
    degree: int,
    prefix: str = "",
    x_range: tuple[float, float] | None = None,
) -> Model:
    """Thin wrapper that returns a :class:`ChebyshevModel` instance.

    Kept for callers that prefer a factory-function style; new code should
    instantiate :class:`ChebyshevModel` directly.
    """
    if x_range is None:
        raise ValueError(
            "make_chebyshev_model requires x_range=(x_min, x_max) so "
            "coefficients can be mapped to [-1, 1]."
        )
    return ChebyshevModel(degree=degree, x_range=x_range, prefix=prefix)


# ---------------------------------------------------------------------------
# 2D model classes
# ---------------------------------------------------------------------------

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


class PseudoVoigt2DModel(Model):
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


# Backwards-compatible alias
Pvoigt2DModel = PseudoVoigt2DModel


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
