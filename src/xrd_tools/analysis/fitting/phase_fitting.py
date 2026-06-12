"""
Multi-phase 1D XRD pattern fitting built on lmfit composites.

The fit model is assembled from independent :class:`lmfit.Model` pieces so
that the Levenberg-Marquardt minimiser sees a single composite with all
parameters (lattice constants, widths, scales, amorphous-peak parameters,
background coefficients, q-shift) refined together:

    composite = fit_background         (optional: polynomial or Chebyshev)
              + amorphous_peak          (optional: Gaussian / PseudoVoigt / ...)
              + phase_0                 (lattice-driven pseudo-Voigt peaks)
              + phase_1
              + ...                     (one :class:`PhasePeakModel` per phase)

Each :class:`PhasePeakModel` evaluates its symmetry-allowed reflections at
positions computed *analytically* from the current lattice parameters via
the reciprocal metric tensor — no pymatgen calls inside the inner loop.
The per-peak pseudo-Voigt evaluation is fully vectorised, so a phase with
dozens of reflections is still a single NumPy broadcast.

Background handling is split into two layers:

* **Pre-fit baseline** — ``snip_1d`` / ``chebyshev_background`` / a user
  array are subtracted from ``y`` *before* the fit runs; they contribute
  no free parameters. Select with ``prefit_background=``.
* **In-fit background model** — an lmfit polynomial or Chebyshev model
  added to the composite so its coefficients refine alongside everything
  else. Select with ``fit_background=``.

Example
-------
>>> from xrd_tools.analysis.phase import PhaseModel
>>> from xrd_tools.analysis.fitting.phase_fitting import PhaseFitter
>>>
>>> ortho = PhaseModel.from_cif("HfO2_ortho.cif", name="Ortho")
>>> mono  = PhaseModel.from_cif("HfO2_mono.cif",  name="Mono")
>>>
>>> fitter = PhaseFitter(
...     q, intensity, sigma=sigma,
...     prefit_background='snip',           # SNIP-subtract before fitting
...     fit_background='chebyshev3',        # refine a Cheb-3 polynomial too
...     amorphous_peak='gaussian',          # refine a SiO2 amorphous peak
...     amorphous_init=dict(center=1.5, sigma=0.3),
... )
>>> fitter.add_phase(ortho)
>>> fitter.add_phase(mono)
>>> result = fitter.fit()
>>> print(result.summary())
"""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from lmfit import Model, Parameters
from lmfit.models import (
    GaussianModel,
    LorentzianModel,
    MoffatModel,
    Pearson4Model,
    Pearson7Model,
    PseudoVoigtModel,
    SkewedGaussianModel,
    SplitLorentzianModel,
    StudentsTModel,
    VoigtModel,
    PolynomialModel,
    SplineModel,
)

from xrd_tools.analysis.fitting.background import snip_1d, chebyshev_background
from xrd_tools.analysis.fitting.models import (
    ChebyshevModel,
    LorentzianSquaredModel,
)

logger = logging.getLogger(__name__)

__all__ = ["PhaseFitter", "MultiPhaseResult", "PhasePeakModel"]


# ---------------------------------------------------------------------------
# Analytical d-spacing helpers (kept module-level for test backwards compat)
# ---------------------------------------------------------------------------

def _metric_tensor(a: float, b: float, c: float,
                   alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return the reciprocal metric tensor G* for a general triclinic lattice.

    Angles are in **degrees**.  The tensor is 3×3 and symmetric; G*_ij lets
    you compute  1/d²(hkl) = h_i G*_ij h_j  for Miller indices (h, k, l).
    """
    ar, br, gr = np.radians(alpha), np.radians(beta), np.radians(gamma)
    ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
    sa, sb, sg = np.sin(ar), np.sin(br), np.sin(gr)

    vol_sq_term = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    if vol_sq_term <= 0:
        raise ValueError(
            f"Invalid lattice angles (alpha={alpha}, beta={beta}, gamma={gamma}): "
            f"volume determinant is non-positive ({vol_sq_term:.6g})"
        )
    vol = a * b * c * np.sqrt(vol_sq_term)
    if vol < 1e-12:
        raise ValueError(
            f"Degenerate lattice: volume {vol:.6g} is too small "
            f"(a={a}, b={b}, c={c})"
        )

    # Reciprocal lattice parameters
    a_star = b * c * sa / vol
    b_star = a * c * sb / vol
    c_star = a * b * sg / vol

    cos_alpha_star = (cb * cg - ca) / (sb * sg)
    cos_beta_star  = (ca * cg - cb) / (sa * sg)
    cos_gamma_star = (ca * cb - cg) / (sa * sb)

    G = np.array([
        [a_star**2,
         a_star * b_star * cos_gamma_star,
         a_star * c_star * cos_beta_star],
        [a_star * b_star * cos_gamma_star,
         b_star**2,
         b_star * c_star * cos_alpha_star],
        [a_star * c_star * cos_beta_star,
         b_star * c_star * cos_alpha_star,
         c_star**2],
    ])
    return G


def _q_from_hkl(hkl: np.ndarray, G_star: np.ndarray) -> np.ndarray:
    """Compute q = 2π/d for an array of (N, 3) Miller indices."""
    inv_d2 = np.einsum("ij,jk,ik->i", hkl.astype(float), G_star, hkl.astype(float))
    inv_d2 = np.clip(inv_d2, 1e-30, None)
    return 2.0 * np.pi * np.sqrt(inv_d2)


def _march_dollase(
    hkl: np.ndarray,
    G_star: np.ndarray,
    march_axis: np.ndarray,
    march_r: float,
) -> np.ndarray:
    """March-Dollase preferred-orientation correction for each peak.

    Parameters
    ----------
    hkl : (N, 3) ndarray
        Miller indices for each peak.
    G_star : (3, 3) ndarray
        Reciprocal metric tensor (from :func:`_metric_tensor`).
    march_axis : (3,) ndarray
        Preferred orientation direction [h₀ k₀ l₀] (integer Miller
        indices of the texture axis, e.g. ``[0, 0, 1]`` for c-axis
        texture).
    march_r : float
        March-Dollase parameter.  ``r = 1`` → random powder (no
        correction); ``r < 1`` → plate-like texture (enhanced
        reflections parallel to the axis); ``r > 1`` → needle-like.

    Returns
    -------
    (N,) ndarray
        Multiplicative intensity correction per peak.  Always > 0.
    """
    h0 = np.asarray(march_axis, dtype=float)
    # cos²α between each peak (hkl) and the texture axis (h₀) in
    # reciprocal space:  cos α = (hkl·G*·h₀) / sqrt(hkl·G*·hkl · h₀·G*·h₀)
    num = np.einsum("ij,jk,k->i", hkl.astype(float), G_star, h0) ** 2
    denom_hkl = np.einsum("ij,jk,ik->i", hkl.astype(float), G_star, hkl.astype(float))
    denom_h0 = float(h0 @ G_star @ h0)
    cos2a = np.clip(num / (denom_hkl * denom_h0 + 1e-30), 0.0, 1.0)
    sin2a = 1.0 - cos2a
    r = float(march_r)
    r = max(r, 1e-10)
    # MD = (r²·cos²α + sin²α / r) ^ (-3/2)
    return (r ** 2 * cos2a + sin2a / r) ** (-1.5)


# ---------------------------------------------------------------------------
# Pseudo-Voigt evaluation (scalar helper kept for tests; vector version for fits)
# ---------------------------------------------------------------------------

_GAUSS_FWHM_FACTOR = np.sqrt(2.0 * np.log(2.0))


def _pseudo_voigt(x: np.ndarray, center: float, amplitude: float,
                  sigma: float, fraction: float) -> np.ndarray:
    """Scalar-centre pseudo-Voigt, area-normalised.

    Uses the lmfit convention::

        pV = (1 - η) G + η L

    where G (Gaussian) and L (Lorentzian) share the same FWHM = 2σ.
    Kept as a public-ish helper because the existing test suite and
    :func:`tests.test_phase_fitting._synthetic_pattern` import it by name.
    New code should prefer :func:`_vector_pseudo_voigt`.
    """
    sig_g = sigma / _GAUSS_FWHM_FACTOR if sigma > 0 else 1e-30
    gauss = (amplitude / (sig_g * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - center) / sig_g) ** 2
    )
    lorentz = (amplitude / np.pi) * (sigma / ((x - center) ** 2 + sigma**2))
    return (1.0 - fraction) * gauss + fraction * lorentz


def _vector_pseudo_voigt(
    x: np.ndarray,
    centers: np.ndarray,
    amplitudes: np.ndarray,
    sigmas: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Sum of area-normalised pseudo-Voigt peaks, vectorised over peaks.

    Parameters
    ----------
    x : (M,) ndarray
        Evaluation points (already q-shifted if applicable).
    centers : (N,) ndarray
        Peak centres.
    amplitudes : (N,) ndarray
        Per-peak areas.
    sigmas : (N,) ndarray
        Per-peak widths (Lorentzian HWHM = σ, Gaussian σ_g = σ/√(2 ln 2)).
    fraction : float
        Gaussian/Lorentzian mixing fraction in ``[0, 1]``.

    Returns
    -------
    (M,) ndarray
    """
    x = np.asarray(x, dtype=float)
    if centers.size == 0:
        return np.zeros_like(x)

    # Broadcast to (M, N)
    dx = x[:, None] - centers[None, :]
    sig = np.clip(sigmas, 1e-12, None)[None, :]
    amp = amplitudes[None, :]

    sig_g = sig / _GAUSS_FWHM_FACTOR
    gauss = (amp / (sig_g * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (dx / sig_g) ** 2)
    lorentz = (amp / np.pi) * (sig / (dx * dx + sig * sig))
    return ((1.0 - fraction) * gauss + fraction * lorentz).sum(axis=1)


def _caglioti_sigma(q: float | np.ndarray, U: float, V: float, W: float) -> float | np.ndarray:
    """Caglioti-like FWHM in Q-space::

        σ²(Q) = U·Q² + V·|Q| + W

    U has units Å⁻², V has units Å⁻¹, W is dimensionless.  Returns σ.
    """
    sigma2 = U * q**2 + V * np.abs(q) + W
    return np.sqrt(np.clip(sigma2, 1e-10, None))


# Scherrer constant (spherical crystallites, FWHM form).
_SCHERRER_K = 0.94


def _scherrer_sigma(
    q: float | np.ndarray, D: float, eps: float,
) -> float | np.ndarray:
    """Williamson-Hall broadening in Q-space.

    Returns the same quantity as :func:`_caglioti_sigma` — i.e. the
    per-peak ``sigma`` fed into the vectorised peak profile — computed
    from two physical parameters:

    * ``D`` : volume-weighted crystallite size (Å).  Size broadening is
      size-independent in q and equals ``2π·K/D`` in FWHM.
    * ``eps`` : microstrain (dimensionless).  Strain broadening scales
      linearly with q and equals ``2·eps·q`` in FWHM.

    The combined FWHM²(q) = (2π·K/D)² + (2·eps·q)² and the returned
    ``sigma`` = FWHM / 2, matching the ``W = sigma²`` convention used
    by the Caglioti branch.
    """
    D_safe = max(float(D), 1e-3)
    fwhm_sq = (2.0 * np.pi * _SCHERRER_K / D_safe) ** 2 + (2.0 * float(eps) * q) ** 2
    return 0.5 * np.sqrt(np.clip(fwhm_sq, 1e-20, None))


def _resolve_width_model(
    width_model: str | None, caglioti: bool | None,
) -> str:
    """Map (width_model, caglioti) kwargs to a canonical string.

    Precedence: if ``width_model`` is given, use it; else fall back to
    the legacy boolean ``caglioti`` (True → 'caglioti', False → 'fixed').
    """
    if width_model is not None:
        wm = str(width_model).lower().strip()
        if wm not in ("caglioti", "scherrer", "fixed"):
            raise ValueError(
                f"width_model must be 'caglioti', 'scherrer', or 'fixed'; "
                f"got {width_model!r}"
            )
        return wm
    return "caglioti" if bool(caglioti) else "fixed"


# ---------------------------------------------------------------------------
# Vectorised peak profile library
# ---------------------------------------------------------------------------
#
# Each entry summed over peaks (axis=1 of the (M, N) broadcast arrays).
# All profiles are area-normalised: the peak's "amplitude" is its total
# area rather than its peak height.  This matches the lmfit model
# convention and makes phase fractions directly interpretable.
#
# ``sigmas`` comes from Caglioti σ(Q) or the fixed per-phase sigma and is
# clipped to avoid divide-by-zero.  Extra shape parameters are passed as
# keyword arguments.

def _broadcast_peaks(x, centers, amplitudes, sigmas):
    """Build the shared (M, N) broadcast arrays used by every profile."""
    x = np.asarray(x, dtype=float)
    if centers.size == 0:
        return None, None, None, None
    dx = x[:, None] - centers[None, :]
    sig = np.clip(sigmas, 1e-12, None)[None, :]
    amp = amplitudes[None, :]
    return x, dx, sig, amp


def _vp_gaussian(x, centers, amplitudes, sigmas, **_):
    x, dx, sig, amp = _broadcast_peaks(x, centers, amplitudes, sigmas)
    if dx is None:
        return np.zeros_like(x)
    sig_g = sig / _GAUSS_FWHM_FACTOR
    g = (amp / (sig_g * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (dx / sig_g) ** 2)
    return g.sum(axis=1)


def _vp_lorentzian(x, centers, amplitudes, sigmas, **_):
    x, dx, sig, amp = _broadcast_peaks(x, centers, amplitudes, sigmas)
    if dx is None:
        return np.zeros_like(x)
    return ((amp / np.pi) * (sig / (dx * dx + sig * sig))).sum(axis=1)


def _vp_pseudovoigt(x, centers, amplitudes, sigmas, *, fraction=0.5, **_):
    return _vector_pseudo_voigt(x, centers, amplitudes, sigmas, fraction)


def _vp_voigt(x, centers, amplitudes, sigmas, *, gamma=None, **_):
    """Voigt = Re[wofz((dx + i γ) / (σ √2))] / (σ √(2π)).

    When ``gamma`` is None we default to ``gamma = sigma`` (lmfit default).
    """
    from scipy.special import wofz
    x, dx, sig, amp = _broadcast_peaks(x, centers, amplitudes, sigmas)
    if dx is None:
        return np.zeros_like(x)
    if gamma is None:
        gam = sig
    else:
        gam = np.full_like(sig, float(max(gamma, 1e-12)))
    z = (dx + 1j * gam) / (sig * np.sqrt(2.0))
    v = amp * np.real(wofz(z)) / (sig * np.sqrt(2.0 * np.pi))
    return v.sum(axis=1)


def _vp_lorentzian_squared(x, centers, amplitudes, sigmas, **_):
    """Lorentzian-squared profile (see fitting.models.LorentzianSquaredModel).

    Normalisation: ``amp · (2/π) · σ³ / ((x−c)² + σ²)²`` — area = amp.
    """
    x, dx, sig, amp = _broadcast_peaks(x, centers, amplitudes, sigmas)
    if dx is None:
        return np.zeros_like(x)
    denom = (dx * dx + sig * sig) ** 2
    return (amp * (2.0 / np.pi) * sig ** 3 / denom).sum(axis=1)


def _vp_pearson7(x, centers, amplitudes, sigmas, *, expo=1.5, **_):
    """Pearson VII, area-normalised (matches lmfit's Pearson7Model)."""
    from scipy.special import gamma as gamma_fn
    x, dx, sig, amp = _broadcast_peaks(x, centers, amplitudes, sigmas)
    if dx is None:
        return np.zeros_like(x)
    m = float(max(expo, 0.1))
    # lmfit Pearson7 uses the form:
    #   amp * (Γ(m) / (σ · Γ(m − 1/2) · √π)) · (1 + ((x−c)/σ)² · (2^(1/m) − 1))^(−m)
    norm = gamma_fn(m) / (sig * gamma_fn(m - 0.5) * np.sqrt(np.pi))
    core = (1.0 + (dx / sig) ** 2 * (2.0 ** (1.0 / m) - 1.0)) ** (-m)
    return (amp * norm * core).sum(axis=1)


def _vp_splitlorentzian(x, centers, amplitudes, sigmas, *, sigma_r=None, **_):
    """Split Lorentzian: different HWHM on the two sides of each centre."""
    x, dx, sig, amp = _broadcast_peaks(x, centers, amplitudes, sigmas)
    if dx is None:
        return np.zeros_like(x)
    sig_l = sig
    if sigma_r is None:
        sig_r_arr = sig_l
    else:
        sig_r_arr = np.full_like(sig_l, float(max(sigma_r, 1e-12)))
    left = dx < 0
    sig_eff = np.where(left, sig_l, sig_r_arr)
    # Area-normalised so total area = amp (independent of the asymmetry).
    norm = 2.0 / (np.pi * (sig_l + sig_r_arr))
    return (amp * norm * (sig_eff ** 2) / (dx * dx + sig_eff * sig_eff)).sum(axis=1)


def _vp_moffat(x, centers, amplitudes, sigmas, *, beta=1.0, **_):
    """Moffat profile: amp · (β−1)/(π σ²) · (1 + (dx/σ)²)^(−β)."""
    x, dx, sig, amp = _broadcast_peaks(x, centers, amplitudes, sigmas)
    if dx is None:
        return np.zeros_like(x)
    b = float(max(beta, 1.01))
    norm = (b - 1.0) / (np.pi * sig * sig)
    return (amp * norm * (1.0 + (dx / sig) ** 2) ** (-b)).sum(axis=1)


def _vp_pearson4(x, centers, amplitudes, sigmas, *, expo=1.5, skew=0.0, **_):
    """Pearson IV — complex, uses scipy loggamma for numerical stability.

    Follows lmfit's Pearson4Model:
        amp · |Γ(m + iν/2) / Γ(m)|² / (σ · B(m − ½, ½)) ·
              [1 + ((x−c)/σ)²]^(−m) · exp(−ν · arctan((x−c)/σ))
    """
    from scipy.special import loggamma, betaln
    x, dx, sig, amp = _broadcast_peaks(x, centers, amplitudes, sigmas)
    if dx is None:
        return np.zeros_like(x)
    m = float(max(expo, 0.51))
    nu = float(skew)
    arg = dx / sig
    log_norm = (2.0 * np.real(loggamma(m + 0.5j * nu))
                - 2.0 * np.real(loggamma(m))
                - np.log(sig) - betaln(m - 0.5, 0.5))
    log_core = -m * np.log(1.0 + arg * arg) - nu * np.arctan(arg)
    return (amp * np.exp(log_norm + log_core)).sum(axis=1)


def _vp_studentst(x, centers, amplitudes, sigmas, **_):
    """Student's t profile with ν = 2·σ/FWHM-ish; area = amp."""
    from scipy.special import gamma as gamma_fn
    x, dx, sig, amp = _broadcast_peaks(x, centers, amplitudes, sigmas)
    if dx is None:
        return np.zeros_like(x)
    # lmfit StudentsT has a single width parameter σ; treat ν≡σ per lmfit
    # (see lmfit source). Formula:
    #   amp · Γ((σ+1)/2) / (√(σπ) · Γ(σ/2)) · (1 + dx²/σ)^(−(σ+1)/2)
    # We clamp σ ≥ 1.0 to keep the fit stable.
    nu = np.clip(sig, 1.0, None)
    norm = gamma_fn((nu + 1.0) * 0.5) / (np.sqrt(nu * np.pi) * gamma_fn(nu * 0.5))
    core = (1.0 + (dx * dx) / nu) ** (-(nu + 1.0) * 0.5)
    return (amp * norm * core).sum(axis=1)


def _vp_skewedgaussian(x, centers, amplitudes, sigmas, *, gamma=0.0, **_):
    """Skewed Gaussian: 2 · φ(z) · Φ(γ z) · amp / σ (lmfit convention)."""
    from scipy.special import erf
    x, dx, sig, amp = _broadcast_peaks(x, centers, amplitudes, sigmas)
    if dx is None:
        return np.zeros_like(x)
    sig_g = sig / _GAUSS_FWHM_FACTOR
    z = dx / sig_g
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + erf(float(gamma) * z / np.sqrt(2.0)))
    return (amp * 2.0 * phi * Phi / sig_g).sum(axis=1)


# Profile registry:  name → (vector function, extra shape param names,
# default values, bounds).  "Extra" means beyond scale / σ / lattice /
# q_shift, which every profile has.
# Note: extras must NOT collide with base params (scale, a, b, c,
# alpha, beta, gamma, q_shift, U, V, W, sigma).  A few profiles
# natively use ``beta`` / ``gamma`` as shape names — those are exposed
# under disambiguated names below (``moffat_beta``, ``voigt_gamma``,
# ``skew_gamma``) and remapped back to their kwarg names when forwarded
# to the ``_vp_*`` vectorised evaluator via ``kw_map``.
_PHASE_PROFILE_SPEC: dict[str, dict[str, Any]] = {
    "gaussian":           dict(func=_vp_gaussian,           extras=(),                     defaults={}),
    "lorentzian":         dict(func=_vp_lorentzian,         extras=(),                     defaults={}),
    "pseudovoigt":        dict(func=_vp_pseudovoigt,        extras=("fraction",),          defaults={"fraction": 0.5}),
    "voigt":              dict(func=_vp_voigt,              extras=("voigt_gamma",),       defaults={"voigt_gamma": 0.02},
                               kw_map={"voigt_gamma": "gamma"}),
    "lorentzian_squared": dict(func=_vp_lorentzian_squared, extras=(),                     defaults={}),
    "pearson7":           dict(func=_vp_pearson7,           extras=("expo",),              defaults={"expo": 1.5}),
    "splitlorentzian":    dict(func=_vp_splitlorentzian,    extras=("sigma_r",),           defaults={"sigma_r": 0.02}),
    "moffat":             dict(func=_vp_moffat,             extras=("moffat_beta",),       defaults={"moffat_beta": 1.0},
                               kw_map={"moffat_beta": "beta"}),
    "pearson4":           dict(func=_vp_pearson4,           extras=("expo", "skew"),       defaults={"expo": 1.5, "skew": 0.0}),
    "studentst":          dict(func=_vp_studentst,          extras=(),                     defaults={}),
    "skewedgaussian":     dict(func=_vp_skewedgaussian,     extras=("skew_gamma",),        defaults={"skew_gamma": 0.0},
                               kw_map={"skew_gamma": "gamma"}),
}


def _canonical_profile(name: str) -> str:
    """Map a user-supplied profile string to a registry key."""
    key = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "pvoigt": "pseudovoigt",
        "pseudo": "pseudovoigt",
        "lorentz": "lorentzian",
        "lor": "lorentzian",
        "lor2": "lorentzian_squared",
        "lorentziansquared": "lorentzian_squared",
        "gauss": "gaussian",
        "pearsonvii": "pearson7",
        "pearsoniv": "pearson4",
        "splitlor": "splitlorentzian",
        "studentt": "studentst",
        "studentsts": "studentst",
        "skewedgauss": "skewedgaussian",
    }
    if key in _PHASE_PROFILE_SPEC:
        return key
    if key in aliases:
        return aliases[key]
    raise ValueError(
        f"Unknown phase_profile {name!r}. Known: "
        f"{', '.join(sorted(_PHASE_PROFILE_SPEC))}."
    )


# ---------------------------------------------------------------------------
# Per-phase lmfit Model
# ---------------------------------------------------------------------------

def _make_phase_eval(
    hkl: np.ndarray,
    template_amp: np.ndarray,
    caglioti: bool | None = None,
    fixed_q: np.ndarray | None = None,
    profile: str = "pseudovoigt",
    texture: str = "none",
    march_axis: tuple[int, int, int] = (0, 0, 1),
    width_model: str | None = None,
):
    """Build an lmfit-compatible function + explicit parameter name list.

    Returns
    -------
    func, param_names : (callable, list[str])
        ``func`` has a ``**kwargs`` signature; lmfit is told the parameter
        names explicitly so it still builds the correct
        :class:`~lmfit.Parameters`.

    Notes
    -----
    ``width_model`` controls the width parameterisation (preferred over
    the legacy ``caglioti`` bool kwarg, which still works):

    * ``"caglioti"`` — parameters include ``U, V, W`` and the Caglioti
      model σ²(Q) = U·Q² + V·|Q| + W sets per-peak widths.
    * ``"scherrer"`` — Williamson-Hall in q-space: parameters ``D``
      (crystallite size in Å) and ``eps`` (microstrain), with
      FWHM²(q) = (2π·K/D)² + (2·eps·q)².
    * ``"fixed"`` — a single scalar ``sigma`` is used for every peak.

    ``profile`` selects the peak shape (see :data:`_PHASE_PROFILE_SPEC`).
    Profile-specific shape parameters (e.g. ``fraction`` for pseudo-Voigt,
    ``gamma`` for Voigt, ``expo`` for Pearson VII …) are appended to the
    parameter list automatically.  If ``fixed_q`` is supplied the peak
    centres come from the cached array and lattice parameters become
    inert placeholders.

    ``texture`` selects the peak-intensity correction:

    * ``"none"`` — amplitudes = ``template_amp * scale`` (the default).
    * ``"march_dollase"`` — adds a ``march_r`` parameter; amplitudes
      are multiplied by the March-Dollase correction for each (hkl)
      relative to ``march_axis``.  ``march_r = 1`` → no correction.
    * ``"free"`` — adds one ``pk{j}`` multiplier per peak; amplitudes
      become ``template_amp[j] * scale * pk{j}``.  Most flexible,
      but adds O(N_peaks) free parameters.
    """
    use_fixed = fixed_q is not None
    if use_fixed:
        fixed_q = np.asarray(fixed_q, dtype=float)

    wm = _resolve_width_model(width_model, caglioti)

    n_peaks = int(hkl.shape[0]) if hkl.size else 0
    march_axis_arr = np.asarray(march_axis, dtype=float)
    if texture == "march_dollase" and np.allclose(march_axis_arr, 0):
        raise ValueError(
            "march_axis must be a non-zero Miller index direction, "
            f"got {march_axis!r}."
        )

    spec = _PHASE_PROFILE_SPEC[_canonical_profile(profile)]
    vector_fn = spec["func"]
    extras: tuple[str, ...] = tuple(spec["extras"])
    kw_map: dict[str, str] = dict(spec.get("kw_map", {}))

    def _peak_positions(a, b, c, alpha, beta, gamma):
        if use_fixed:
            return fixed_q
        G = _metric_tensor(a, b, c, alpha, beta, gamma)
        return _q_from_hkl(hkl, G)

    # Shared kernel used by all width parameterisations.
    def _kernel(x, *, scale, q_shift, a, b, c, alpha, beta, gamma,
                width_args, shape_kwargs, texture_args):
        x_arr = np.asarray(x, dtype=float) - q_shift
        if hkl.size == 0:
            return np.zeros_like(x_arr)
        q_pos = _peak_positions(a, b, c, alpha, beta, gamma)
        if wm == "caglioti":
            U, V, W = width_args
            sig = _caglioti_sigma(q_pos, U, V, W)
        elif wm == "scherrer":
            D, eps = width_args
            sig = _scherrer_sigma(q_pos, D, eps)
        else:  # "fixed"
            (sigma_scalar,) = width_args
            sig = np.full_like(q_pos, float(sigma_scalar))

        # --- amplitude with texture correction ---
        if texture == "march_dollase":
            (march_r,) = texture_args
            G_star = _metric_tensor(a, b, c, alpha, beta, gamma)
            md = _march_dollase(hkl, G_star, march_axis_arr, march_r)
            amps = template_amp * scale * md
        elif texture == "free":
            pk_mults = np.asarray(texture_args, dtype=float)
            amps = template_amp * scale * pk_mults
        else:
            amps = template_amp * scale
        return vector_fn(x_arr, q_pos, amps, sig, **shape_kwargs)

    # Build the entry point.  lmfit's Model.__init__ introspects the
    # function signature and refuses any parameter name that isn't in
    # ``func.__code__.co_varnames`` (or the **kws bucket is missing),
    # so we can't just use ``**kw`` — we need to synthesise a function
    # with an explicit signature matching the parameter set for this
    # profile/width combination.
    base_params = ["scale", "a", "b", "c", "alpha", "beta", "gamma", "q_shift"]
    if wm == "caglioti":
        width_param_names = ["U", "V", "W"]
    elif wm == "scherrer":
        width_param_names = ["D", "eps"]
    else:  # "fixed"
        width_param_names = ["sigma"]

    # Texture-dependent parameter names.
    if texture == "march_dollase" and n_peaks > 0:
        texture_param_names: list[str] = ["march_r"]
    elif texture == "free" and n_peaks > 0:
        texture_param_names = [f"pk{j}" for j in range(n_peaks)]
    else:
        texture_param_names = []

    param_names = (
        base_params + width_param_names + list(extras) + texture_param_names
    )

    # Build an explicit-signature wrapper via exec. All parameters are
    # declared as keyword arguments with default ``1.0`` (arbitrary —
    # real values come from the lmfit Parameters at fit time).
    sig_args = ", ".join(f"{p}=1.0" for p in param_names)
    call_args = ", ".join(f"{p}={p}" for p in param_names)
    src = (
        f"def _phase(x, {sig_args}):\n"
        f"    return _phase_impl(x, {call_args})\n"
    )
    ns: dict = {}

    def _phase_impl(x, **kw):
        width_args = tuple(kw[k] for k in width_param_names)
        shape_kwargs = {kw_map.get(k, k): kw[k] for k in extras}
        texture_args = tuple(kw[k] for k in texture_param_names)
        return _kernel(
            x,
            scale=kw["scale"],
            q_shift=kw["q_shift"],
            a=kw["a"], b=kw["b"], c=kw["c"],
            alpha=kw["alpha"], beta=kw["beta"], gamma=kw["gamma"],
            width_args=width_args,
            shape_kwargs=shape_kwargs,
            texture_args=texture_args,
        )

    ns["_phase_impl"] = _phase_impl
    exec(src, ns)
    _phase = ns["_phase"]

    return _phase, param_names


class PhasePeakModel(Model):
    """lmfit Model for one crystallographic phase.

    Evaluates all symmetry-allowed reflections as a sum of pseudo-Voigt
    peaks whose centres are derived from the current lattice parameters
    via the reciprocal metric tensor.  The peak *list* (Miller indices
    and template intensities) is frozen at construction time; only the
    lattice, widths, mixing fraction, overall scale, and q-shift refine.

    Parameters
    ----------
    phase : PhaseModel
        Must have ``.peaks`` populated.  A reference to the original
        object is kept for reporting (lattice type, name).
    hkl : (N, 3) ndarray
        Miller indices for the peaks included in the fit.
    template_amp : (N,) ndarray
        Normalised template intensities (max = 1).
    prefix : str
        lmfit prefix for parameter namespacing (e.g. ``"p0_"``).

    Notes
    -----
    Peak profile is pseudo-Voigt.  Gaussian-only / Lorentzian-only
    profiles are obtained by fixing ``fraction`` to 0 or 1 — see
    :meth:`PhaseFitter._apply_phase_profile_lock`.
    """

    def __init__(
        self,
        phase: Any,
        hkl: np.ndarray,
        template_amp: np.ndarray,
        prefix: str = "",
        caglioti: bool | None = None,
        fixed_q: np.ndarray | None = None,
        profile: str = "pseudovoigt",
        texture: str = "none",
        march_axis: tuple[int, int, int] = (0, 0, 1),
        width_model: str | None = None,
    ):
        wm = _resolve_width_model(width_model, caglioti)
        func, param_names = _make_phase_eval(
            hkl, template_amp,
            width_model=wm, fixed_q=fixed_q, profile=profile,
            texture=texture, march_axis=march_axis,
        )
        super().__init__(
            func, prefix=prefix,
            independent_vars=["x"], param_names=param_names,
        )
        self.phase = phase
        self.hkl = np.asarray(hkl, dtype=float)
        self.template_amp = np.asarray(template_amp, dtype=float)
        self.width_model = wm
        self.caglioti = (wm == "caglioti")  # kept for backward compat
        self.profile = _canonical_profile(profile)
        self.texture = texture
        self.march_axis = march_axis
        self.fixed_q = None if fixed_q is None else np.asarray(fixed_q, dtype=float)


# ---------------------------------------------------------------------------
# Peak profile registry (for amorphous peak selection)
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
    "pearson7": Pearson7Model,
    "pearson4": Pearson4Model,
    "splitlorentzian": SplitLorentzianModel,
    "moffat": MoffatModel,
    "studentst": StudentsTModel,
    "skewedgaussian": SkewedGaussianModel,
}


def _make_amorphous_model(profile: str, prefix: str = "am_") -> Model:
    key = profile.lower().replace(" ", "").replace("-", "")
    cls = _PEAK_MODEL_MAP.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown amorphous peak profile {profile!r}. "
            f"Choose from: {', '.join(sorted(_PEAK_MODEL_MAP))}."
        )
    return cls(prefix=prefix)


# ---------------------------------------------------------------------------
# In-fit background (polynomial / Chebyshev — contributes free params)
# ---------------------------------------------------------------------------

def _build_high_degree_polynomial_model(degree: int, prefix: str = "bg_") -> Model:
    """Build an lmfit :class:`Model` for a polynomial of arbitrary degree.

    lmfit's stock :class:`PolynomialModel` is hard-capped at degree 7.
    We synthesise an equivalent Model with parameters ``c0, c1, …, cN``
    (matching lmfit's convention) so ``params[prefix + 'c0']`` keeps
    working for the existing initialiser.
    """
    if degree < 0:
        raise ValueError("polynomial degree must be non-negative")
    names = [f"c{i}" for i in range(degree + 1)]

    # No dynamic code (no ``exec``): the evaluator takes ``**coeffs`` but
    # advertises an explicit ``(x, c0=0.0, …, cN=0.0)`` signature via
    # ``__signature__``.  lmfit introspects that signature to discover the
    # parameter names + defaults, so this behaves exactly like the old
    # exec-compiled function while remaining static-analysis / frozen-build
    # friendly and free of code-injection risk.
    def _poly(x, **coeffs):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        for i, n in enumerate(names):
            out = out + coeffs.get(n, 0.0) * x ** i
        return out

    import inspect
    sig_params = [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    sig_params += [
        inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=0.0)
        for n in names
    ]
    _poly.__signature__ = inspect.Signature(sig_params)
    return Model(_poly, prefix=prefix)


def _make_fit_background_template_model(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    prefix: str = "bg_",
) -> Model:
    """Build a one-parameter scaled-template background model.

    The model evaluates as ``A * np.interp(x, x_ref, y_ref)`` — i.e. the
    reference spectrum is interpolated to whatever x-grid the fitter is
    currently using, then scaled by a single positive amplitude ``A``.

    Parameters
    ----------
    x_ref, y_ref : ndarray
        Reference spectrum (e.g. an integrated substrate-only pattern).
    prefix : str
        Parameter prefix.  Default ``"bg_"`` so the amplitude becomes
        ``bg_A`` alongside the other fit parameters.
    """
    x_arr = np.asarray(x_ref, dtype=float)
    y_arr = np.asarray(y_ref, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"Template x and y must have the same shape, got "
            f"{x_arr.shape} vs {y_arr.shape}"
        )
    # Pre-sort once so np.interp is valid at eval time
    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    def _template(x, A=1.0):
        return float(A) * np.interp(
            np.asarray(x, dtype=float), x_sorted, y_sorted,
        )

    m = Model(_template, prefix=prefix)
    m.set_param_hint(f"{prefix}A", value=1.0, min=0.0)
    return m


def _make_fit_background_model(
    spec: str,
    x_range: tuple[float, float],
    prefix: str = "bg_",
    x_for_spline: np.ndarray | None = None,
) -> Model:
    """Build an lmfit background model from a short string spec.

    Accepted specs (case-insensitive):

    * ``"polynomial{N}"`` / ``"poly{N}"`` — lmfit ``PolynomialModel`` of degree N
      (degree 1 → linear, 2 → quadratic, up to 7 supported by lmfit).
    * ``"chebyshev{N}"`` / ``"cheb{N}"`` — :class:`~xrd_tools.analysis.fitting.models.ChebyshevModel`
      of degree N on the given ``x_range``.
    * ``"spline{N}"`` — lmfit :class:`SplineModel` with ``N`` interior
      knots evenly distributed across ``x_range``.  ``N`` must be ≥ 2.
      ``x_for_spline`` must be supplied so lmfit can bracket the knots.

    Note: ``"template"`` is handled separately by
    :func:`_make_fit_background_template_model` because it requires a
    reference array rather than just a spec string.
    """
    s = spec.lower().strip()
    m = re.search(r"\d+", s)
    if m is None:
        raise ValueError(
            f"Background spec {spec!r} must include a degree/knot count, "
            "e.g. 'polynomial3' or 'chebyshev4' or 'spline8'."
        )
    degree = int(m.group())

    if s.startswith("poly"):
        # lmfit's PolynomialModel caps at degree 7; build a custom one
        # for higher degrees so the widget can expose the full 2..15
        # range uniformly.
        if degree <= 7:
            return PolynomialModel(degree=degree, prefix=prefix)
        return _build_high_degree_polynomial_model(degree=degree, prefix=prefix)
    if s.startswith("cheb"):
        return ChebyshevModel(degree=degree, x_range=x_range, prefix=prefix)
    if s.startswith("spline"):
        if degree < 4:
            raise ValueError(
                "spline background needs at least 4 knots (cubic B-spline)."
            )
        lo, hi = float(x_range[0]), float(x_range[1])
        # Evenly spaced interior knots (endpoints excluded — SplineModel
        # handles boundary knots internally).
        knots = np.linspace(lo, hi, degree).tolist()
        return SplineModel(xknots=knots, prefix=prefix)

    raise ValueError(
        f"Unknown fit_background spec {spec!r}. "
        "Use 'polynomial{N}', 'chebyshev{N}', or 'spline{N}'."
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class MultiPhaseResult:
    """Thin wrapper around an lmfit ModelResult with convenience accessors.

    Compatible with the old hand-rolled ``PhaseFitter`` result: phase
    parameters live under ``p{i}_...``, the global calibration offset is
    ``q_shift``, the optional amorphous peak under ``am_...``, and the
    optional in-fit background under ``bg_...``.
    """

    def __init__(self, lmfit_result: Any, fitter: "PhaseFitter"):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.fitter = fitter

    # ---- quick accessors --------------------------------------------------

    @property
    def q_shift(self) -> float:
        if "q_shift" in self.params:
            return float(self.params["q_shift"].value)
        for i in range(len(self.fitter.phases)):
            key = f"p{i}_q_shift"
            if key in self.params:
                return float(self.params[key].value)
        return 0.0

    def phase_scale(self, idx: int) -> float:
        return float(self.params[f"p{idx}_scale"].value)

    def phase_fractions(self) -> dict[str, float]:
        """Phase fractions defined as scale_i / Σ scales."""
        scales = {
            ph.name: float(self.params[f"p{i}_scale"].value)
            for i, ph in enumerate(self.fitter.phases)
        }
        total = sum(scales.values()) or 1.0
        return {k: v / total for k, v in scales.items()}

    def lattice_params(self, idx: int) -> dict[str, float]:
        pre = f"p{idx}_"
        out: dict[str, float] = {}
        for k in ("a", "b", "c", "alpha", "beta", "gamma"):
            key = f"{pre}{k}"
            if key in self.params:
                out[k] = float(self.params[key].value)
        return out

    def width_params(self, idx: int) -> dict[str, float]:
        pre = f"p{idx}_"
        out: dict[str, float] = {}
        for k in ("U", "V", "W", "sigma", "fraction"):
            key = f"{pre}{k}"
            if key in self.params:
                out[k] = float(self.params[key].value)
        return out

    @property
    def redchi(self) -> float:
        return float(self.lmfit_result.redchi)

    @property
    def success(self) -> bool:
        return bool(self.lmfit_result.success)

    def summary(self) -> str:
        lines = [
            f"Fit success: {self.success}",
            f"Reduced χ²: {self.redchi:.6g}",
            f"Q-shift: {self.q_shift:.6f} Å⁻¹",
        ]
        if self.fitter._fit_background_spec:
            lines.append(f"Fit background: {self.fitter._fit_background_spec}")
        if self.fitter._amorphous_profile:
            lines.append(f"Amorphous peak: {self.fitter._amorphous_profile}")
        lines.append("")

        fracs = self.phase_fractions()
        for i, ph in enumerate(self.fitter.phases):
            lines.append(f"--- {ph.name} ---")
            lines.append(f"  scale      = {self.phase_scale(i):.4g}")
            lines.append(f"  fraction   = {fracs[ph.name]:.4f}")
            lp = self.lattice_params(i)
            if lp:
                lines.append(
                    f"  a={lp.get('a',0):.5f}  b={lp.get('b',0):.5f}  "
                    f"c={lp.get('c',0):.5f}"
                )
            wp = self.width_params(i)
            if wp:
                lines.append(
                    f"  U={wp.get('U',0):.4g}  V={wp.get('V',0):.4g}  "
                    f"W={wp.get('W',0):.4g}  η={wp.get('fraction',0.5):.3f}"
                )
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lattice ordering helpers (used by build_parameters)
# ---------------------------------------------------------------------------

def _group_equal_axes(lat0: dict) -> list[list[str]]:
    """Group a, b, c into clusters of symmetry-equal axes (rtol=1e-4)."""
    groups: list[list[str]] = []
    remaining = ["a", "b", "c"]
    while remaining:
        head = remaining.pop(0)
        group = [head]
        still = []
        for k in remaining:
            if np.isclose(lat0[head], lat0[k], rtol=1e-4):
                group.append(k)
            else:
                still.append(k)
        remaining = still
        groups.append(group)
    return groups


def _add_ordering_gap(
    params: Parameters,
    gap_name: str,
    larger_key: str,
    smaller_key: str,
    v_larger: float,
    v_smaller: float,
    lattice_pct: float,
) -> None:
    """Add a non-negative gap parameter and tie *smaller_key* to it.

    Sets  ``smaller_key = larger_key − gap_name``  with  ``gap ≥ 0``.
    """
    gap_init = max(v_larger - v_smaller, 0.0)
    gap_max = max(
        v_larger * (1 + lattice_pct) - v_smaller * (1 - lattice_pct),
        gap_init * 2.0,
        1e-6,
    )
    params.add(gap_name, value=gap_init, min=0.0, max=gap_max, vary=True)
    params[smaller_key].set(expr=f"{larger_key} - {gap_name}")


def _chain_groups_with_gaps(
    params: Parameters,
    groups: list[list[str]],
    pre: str,
    lat0: dict,
    lattice_pct: float,
    lock_order: bool,
) -> None:
    """Chain groups of axes and tie equals within each group."""
    prev_rep: str | None = None
    for g in groups:
        rep = g[0]
        if prev_rep is not None and lock_order:
            _add_ordering_gap(
                params,
                f"{pre}gap_{prev_rep}_{rep}",
                f"{pre}{prev_rep}",
                f"{pre}{rep}",
                lat0[prev_rep],
                lat0[rep],
                lattice_pct,
            )
        for other in g[1:]:
            params[f"{pre}{other}"].set(expr=f"{pre}{rep}")
        prev_rep = rep


# ---------------------------------------------------------------------------
# Main fitter
# ---------------------------------------------------------------------------

class PhaseFitter:
    """Multi-phase 1D XRD pattern fitter built on lmfit composites.

    See the module docstring for the model structure.

    Parameters
    ----------
    x, y : array-like
        1D data (q in Å⁻¹, intensity).  ``x`` may also be an
        :class:`~xrd_tools.core.containers.IntegrationResult1D`; the
        other fields are picked up automatically in that case.
    sigma : array-like or None
        Per-point uncertainties (used as weights = 1/sigma if provided).
    prefit_background : {'none', 'snip', 'chebyshev'} or ndarray, default 'none'
        Baseline that is **subtracted from y before the fit runs** and
        contributes no free parameters.

        * ``'none'`` — leave ``y`` alone.
        * ``'snip'`` — call :func:`snip_1d` with ``prefit_background_kwargs``.
        * ``'chebyshev'`` — call :func:`chebyshev_background` (iterative
          sigma-clipping polyfit) with ``prefit_background_kwargs``.
        * ndarray — user-supplied baseline of the same length as ``x``.
    prefit_background_kwargs : dict or None
        Extra kwargs forwarded to the prefit baseline routine.  For
        ``'snip'`` the default is ``{'snip_width': max(len(x)*0.05, 3)}``.
    fit_background : str or None, default None
        In-fit background model whose coefficients refine alongside the
        phase parameters.  Pass ``'polynomial{N}'`` or ``'chebyshev{N}'``;
        ``None`` disables this layer.
    amorphous_peak : str or None, default None
        Profile name for an optional amorphous peak added as an lmfit
        model (``'gaussian'``, ``'pseudovoigt'``, ``'lorentzian'``,
        ``'voigt'``, ``'lorentzian_squared'``).  ``None`` disables it.
    amorphous_init : dict or None
        Initial parameter values for the amorphous peak, e.g.
        ``{'center': 1.5, 'sigma': 0.3, 'amplitude': 100.0}``.  Each key
        must match an actual parameter name of the chosen profile.
    snip_width : int or None
        **Deprecated.** Equivalent to
        ``prefit_background='snip', prefit_background_kwargs={'snip_width': ...}``.
        Retained so older notebooks that pass ``snip_width=...`` with the
        default ``'snip'`` background keep working.

    Attributes
    ----------
    x, y : ndarray
        Raw input (before prefit subtraction).
    y_fit : ndarray
        Intensity after prefit subtraction — what the composite actually fits.
    background : ndarray
        The prefit baseline.  Same shape as ``y``.  Legacy attribute — old
        code reads ``fitter.background`` for plotting.
    phases : list
        Registered :class:`~xrd_tools.analysis.phase.PhaseModel` objects.
    composite : lmfit.Model
        The composite model built by :meth:`build_model`; ``None`` until a
        phase has been added and either :meth:`build_parameters` or
        :meth:`fit` has been called.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        x: np.ndarray | Any,
        y: np.ndarray | Any = None,
        sigma: np.ndarray | None = None,
        *,
        prefit_background: str | np.ndarray | None = "none",
        prefit_background_kwargs: dict | None = None,
        fit_background: str | None = None,
        fit_background_template: np.ndarray
                                | tuple[np.ndarray, np.ndarray]
                                | None = None,
        amorphous_peak: str | None = None,
        amorphous_init: dict | None = None,
        # Legacy kwargs (old API — kept for notebook compatibility)
        background: str | np.ndarray | None = None,
        snip_width: int | None = None,
    ):
        # Accept IntegrationResult1D transparently
        from xrd_tools.core.containers import IntegrationResult1D
        if isinstance(x, IntegrationResult1D):
            res = x
            x, y = res.radial, res.intensity
            if sigma is None:
                sigma = res.sigma

        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float) if sigma is not None else None

        # Legacy background= kwarg maps onto prefit_background=
        if background is not None:
            prefit_background = background

        # Legacy snip_width kwarg: push into prefit_background_kwargs
        if snip_width is not None:
            if isinstance(prefit_background, str) and prefit_background.lower() == "snip":
                prefit_background_kwargs = dict(prefit_background_kwargs or {})
                prefit_background_kwargs.setdefault("snip_width", int(snip_width))

        # ------- Prefit baseline -------
        self._prefit_kind: str
        self.background = self._compute_prefit_background(
            prefit_background, prefit_background_kwargs or {},
        )
        self.y_fit = self.y - self.background

        # ------- Phase storage -------
        self.phases: list[Any] = []          # PhaseModel instances
        self._hkl_arrays: list[np.ndarray] = []       # (N, 3) per phase
        self._template_amps: list[np.ndarray] = []    # (N,) normalised to max=1
        self._peak_q_arrays: list[np.ndarray] = []    # (N,) cached peak positions
        self._init_lattice: list[dict] = []           # initial lattice params

        # ------- Fit-background / amorphous settings (captured now, built lazily) -------
        self._fit_background_spec: str | None = fit_background
        self._amorphous_profile: str | None = amorphous_peak
        self._amorphous_init: dict = dict(amorphous_init or {})

        # Template-background reference spectrum (optional).  Stored as a
        # (x_ref, y_ref) tuple; if a bare ndarray is passed, assume it is
        # already on self.x.
        self._fit_background_template: tuple[np.ndarray, np.ndarray] | None = None
        if fit_background_template is not None:
            if isinstance(fit_background_template, tuple):
                x_ref, y_ref = fit_background_template
                self._fit_background_template = (
                    np.asarray(x_ref, dtype=float),
                    np.asarray(y_ref, dtype=float),
                )
            else:
                y_ref = np.asarray(fit_background_template, dtype=float)
                if y_ref.shape != self.x.shape:
                    raise ValueError(
                        f"fit_background_template ndarray has shape "
                        f"{y_ref.shape}, expected {self.x.shape} to match x. "
                        f"Pass a (x_ref, y_ref) tuple for different grids."
                    )
                self._fit_background_template = (self.x.copy(), y_ref)
            # Default spec to "template" if user only supplied the array
            if self._fit_background_spec is None:
                self._fit_background_spec = "template"

        # Populated by build_model() / build_parameters()
        self.composite: Model | None = None
        self._phase_models: list[PhasePeakModel] = []
        self._bg_model: Model | None = None
        self._amorphous_model: Model | None = None

        # Mask of points that participated in the most recent .fit() call.
        # Starts as all-True so pre-fit evaluation code can rely on it.
        self.fit_mask: np.ndarray = np.ones_like(self.x, dtype=bool)

    # ------------------------------------------------------------------
    # Prefit baseline
    # ------------------------------------------------------------------

    def _compute_prefit_background(
        self,
        spec: str | np.ndarray | None,
        kwargs: dict,
    ) -> np.ndarray:
        """Resolve the ``prefit_background`` argument into a concrete array."""
        # None or "none" → zero baseline
        if spec is None:
            self._prefit_kind = "none"
            return np.zeros_like(self.y)
        if isinstance(spec, str):
            kind = spec.lower().strip()
            if kind in ("", "none"):
                self._prefit_kind = "none"
                return np.zeros_like(self.y)
            if kind == "snip":
                width = kwargs.get("snip_width")
                if width is None:
                    width = max(int(len(self.y) * 0.05), 3)
                self._prefit_kind = f"SNIP (w={int(width)})"
                return snip_1d(self.y, snip_width=int(width))
            if kind == "chebyshev":
                degree = kwargs.get("degree", 3)
                n_iter = kwargs.get("n_iter", 5)
                sigma_clip = kwargs.get("sigma_clip", 2.0)
                self._prefit_kind = f"Chebyshev (deg={degree})"
                return chebyshev_background(
                    self.x, self.y,
                    degree=degree, n_iter=n_iter, sigma_clip=sigma_clip,
                )
            raise ValueError(
                f"Unknown prefit_background kind {spec!r}. "
                "Use 'none', 'snip', 'chebyshev', or a pre-computed array."
            )
        # Pre-computed array path
        bg_arr = np.asarray(spec, dtype=float)
        if bg_arr.shape != self.y.shape:
            raise ValueError(
                f"Pre-computed prefit_background has shape {bg_arr.shape}, "
                f"expected {self.y.shape} to match y."
            )
        self._prefit_kind = "user"
        return bg_arr

    @property
    def _background_kind(self) -> str:
        """Legacy accessor used by the old .plot() — returns prefit kind."""
        return self._prefit_kind

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def add_phase(
        self,
        phase: Any,
        *,
        q_range: tuple[float, float] | None = None,
        min_intensity: float = 0.5,
    ) -> None:
        """Register a :class:`~xrd_tools.analysis.phase.PhaseModel`.

        Parameters
        ----------
        phase : PhaseModel
            Must have ``.peaks`` populated (via ``from_cif`` /
            ``from_lattice`` / ``calculate_peaks``).
        q_range : (float, float) or None
            Restrict to peaks inside this window.  ``None`` → the data
            range with a 10% margin.
        min_intensity : float
            Drop peaks whose template intensity is below this fraction of
            the phase's maximum (pymatgen's 0–100 scale).
        """
        if not phase.peaks:
            raise ValueError(
                f"Phase '{phase.name}' has no peaks. "
                "Call phase.calculate_peaks() first."
            )

        if q_range is None:
            margin = 0.1 * (self.x.max() - self.x.min())
            q_range = (self.x.min() - margin, self.x.max() + margin)

        hkls, amps, qs = [], [], []
        for pk in phase.peaks:
            if pk.q < q_range[0] or pk.q > q_range[1]:
                continue
            if pk.intensity < min_intensity:
                continue
            hkls.append(pk.hkl)
            amps.append(pk.intensity)
            qs.append(pk.q)

        if not hkls:
            logger.warning("Phase '%s': no peaks in the fitting range.", phase.name)

        hkl_arr = np.array(hkls, dtype=float) if hkls else np.empty((0, 3))
        amp_arr = np.array(amps, dtype=float) if amps else np.empty(0)
        q_arr = np.array(qs, dtype=float) if qs else np.empty(0)

        if amp_arr.size and amp_arr.max() > 0:
            amp_arr = amp_arr / amp_arr.max()

        self.phases.append(phase)
        self._hkl_arrays.append(hkl_arr)
        self._template_amps.append(amp_arr)
        self._peak_q_arrays.append(q_arr)

        if phase.structure:
            lat = phase.structure.lattice
            self._init_lattice.append(dict(
                a=lat.a, b=lat.b, c=lat.c,
                alpha=lat.alpha, beta=lat.beta, gamma=lat.gamma,
            ))
        else:
            self._init_lattice.append({})

        # Invalidate a previously-built composite
        self.composite = None
        self._phase_models = []

    # ------------------------------------------------------------------
    # Composite / parameter construction
    # ------------------------------------------------------------------

    def build_model(
        self,
        caglioti: bool | None = None,
        phase_profile: str = "pseudovoigt",
        texture: str = "none",
        march_axis: tuple[int, int, int] = (0, 0, 1),
        width_model: str | None = None,
    ) -> Model:
        """Assemble (or re-assemble) the lmfit composite model.

        Returns the composite and caches it on ``self.composite``.
        Callers don't normally need this — :meth:`fit` calls it for you.
        ``phase_profile`` selects the peak shape used by every phase;
        see :data:`_PHASE_PROFILE_SPEC` for the full list.

        ``texture`` controls per-peak intensity correction:

        * ``"none"`` — template amplitudes * scale (default).
        * ``"march_dollase"`` — adds 1 ``march_r`` parameter per phase.
        * ``"free"`` — adds 1 multiplier per peak per phase.

        ``march_axis`` is the preferred-orientation direction (hkl) used
        by the March-Dollase mode.
        """
        # Zero phases are allowed as long as a background and/or amorphous
        # component is present — useful for background-only baseline fits.
        if not self.phases and not self._fit_background_spec and not self._amorphous_profile:
            raise ValueError(
                "No fit content: add at least one phase, or enable a fit "
                "background (fit_background=...) or amorphous component."
            )

        x_min = float(np.nanmin(self.x))
        x_max = float(np.nanmax(self.x))

        wm = _resolve_width_model(width_model, caglioti if caglioti is not None else True)
        self._width_model = wm
        self._caglioti = (wm == "caglioti")  # legacy alias
        self._phase_profile = _canonical_profile(phase_profile)
        self._texture = texture
        self._march_axis = march_axis

        # ---- Phase models (one per registered phase) ----
        # If a phase has no pymatgen Structure, fall back to the cached
        # peak positions so lattice params don't drive the fit.
        self._phase_models = []
        for i, (ph, hkl, amps, qs) in enumerate(zip(
            self.phases, self._hkl_arrays, self._template_amps, self._peak_q_arrays,
        )):
            fixed_q = None if ph.structure else qs
            self._phase_models.append(
                PhasePeakModel(
                    phase=ph, hkl=hkl, template_amp=amps, prefix=f"p{i}_",
                    width_model=wm, fixed_q=fixed_q,
                    profile=self._phase_profile,
                    texture=texture, march_axis=march_axis,
                )
            )

        # ---- Optional in-fit background ----
        self._bg_model = None
        if self._fit_background_spec:
            spec_l = str(self._fit_background_spec).lower().strip()
            # Accept combined specs like "template+cheb2", "template+poly1"
            # → scaled template + additive low-order polynomial/chebyshev.
            if spec_l == "template" or spec_l.startswith("template+"):
                if self._fit_background_template is None:
                    raise ValueError(
                        "fit_background='template' requires "
                        "fit_background_template=(x_ref, y_ref) to be set."
                    )
                x_ref, y_ref = self._fit_background_template
                tmpl_model = _make_fit_background_template_model(
                    x_ref, y_ref, prefix="bg_",
                )
                if "+" in spec_l:
                    extra_spec = spec_l.split("+", 1)[1]
                    if extra_spec:
                        extra_model = _make_fit_background_model(
                            extra_spec,
                            x_range=(x_min, x_max),
                            prefix="bgx_",
                            x_for_spline=self.x,
                        )
                        self._bg_model = tmpl_model + extra_model
                    else:
                        self._bg_model = tmpl_model
                else:
                    self._bg_model = tmpl_model
            else:
                self._bg_model = _make_fit_background_model(
                    self._fit_background_spec,
                    x_range=(x_min, x_max),
                    prefix="bg_",
                    x_for_spline=self.x,
                )

        # ---- Optional amorphous peak ----
        self._amorphous_model = None
        if self._amorphous_profile:
            self._amorphous_model = _make_amorphous_model(
                self._amorphous_profile, prefix="am_",
            )

        # ---- Compose ----
        pieces: list[Model] = []
        if self._bg_model is not None:
            pieces.append(self._bg_model)
        if self._amorphous_model is not None:
            pieces.append(self._amorphous_model)
        pieces.extend(self._phase_models)

        if not pieces:
            raise ValueError(
                "Nothing to fit: need at least one phase, an in-fit "
                "background, or an amorphous component."
            )

        composite: Model = pieces[0]
        for p in pieces[1:]:
            composite = composite + p
        self.composite = composite
        return composite

    def build_parameters(
        self,
        q_shift_bound: float = 0.05,
        lattice_pct: float = 0.05,
        caglioti: bool | None = None,
        phase_profile: str = "pseudovoigt",
        width_max: float | None = None,
        width_min: float | None = None,
        lock_lattice_order: bool = True,
        lock_cross_phase: bool = False,
        texture: str = "none",
        march_axis: tuple[int, int, int] = (0, 0, 1),
        pk_scale_range: tuple[float, float] = (0.0, 10.0),
        width_model: str | None = None,
    ) -> Parameters:
        """Build initial :class:`lmfit.Parameters` for the composite.

        Parameters
        ----------
        q_shift_bound : float
            Max |q-shift| allowed (Å⁻¹).  A single global value shared by
            all phases via lmfit ``expr`` ties.
        lattice_pct : float
            Fractional tolerance on lattice constants (e.g. 0.05 → ±5%).
        caglioti : bool
            If True, leave U/V/W free; otherwise refine a single scalar
            ``sigma`` per phase.
        phase_profile : str
            Phase peak profile.  Any key (or alias) in
            :data:`_PHASE_PROFILE_SPEC` — e.g. ``'pseudovoigt'``,
            ``'voigt'``, ``'pearson7'``, ``'splitlorentzian'`` …
        width_max : float, optional
            Cap on the per-peak sigma.  Applied directly in the
            non-Caglioti branch, and mapped onto W in the Caglioti branch
            (``W_max = width_max²``).  ``None`` uses the permissive
            ``data_range / 4`` default.
        width_min : float, optional
            Lower bound on the per-peak sigma.  Used directly in the
            non-Caglioti branch, and mapped onto W in the Caglioti branch
            (``W_min = width_min²``).  ``None`` uses a small positive
            floor (1e-5 / 1e-8).
        lock_lattice_order : bool
            If True (default), preserve the initial ordering of a, b, c
            within each phase throughout the fit.  Axes equal to within
            1e-4 are grouped and tied via expressions; distinct axes are
            chained with non-negative "gap" parameters, so e.g. an
            initial a > b > c stays a ≥ b ≥ c.  Only a, b, c are
            constrained — alpha/beta/gamma are never fit.
        lock_cross_phase : bool
            If True, extend the ordering constraint across phases for
            each axis letter (a, b, c) whose corresponding phase
            parameter is still independent after the intra-phase step.
            Default False.  See the gap-chain explanation below.
        texture : str
            Peak-intensity correction mode.  ``"none"`` (default) uses
            the template amplitudes directly.  ``"march_dollase"`` adds
            one ``march_r`` parameter per phase for a preferred-orientation
            correction along ``march_axis``.  ``"free"`` adds a per-peak
            amplitude multiplier ``pk{j}`` per phase (the most flexible,
            but adds many parameters).
        march_axis : tuple of int
            Preferred orientation direction (h₀, k₀, l₀) for the
            March-Dollase model.  Default ``(0, 0, 1)``.
        pk_scale_range : tuple of float
            (min, max) bounds for each per-peak multiplier when
            ``texture="free"``.  Default ``(0.0, 10.0)``.
        """
        profile_key = _canonical_profile(phase_profile)

        # Rebuild the composite if width_model, profile, or texture
        # changed, or the model isn't built yet.  Any of these changes
        # alter the parameter set and require new Model instances.
        wm = _resolve_width_model(
            width_model,
            caglioti if caglioti is not None else True,
        )
        need_rebuild = (
            self.composite is None
            or getattr(self, "_width_model", "caglioti") != wm
            or getattr(self, "_phase_profile", "pseudovoigt") != profile_key
            or getattr(self, "_texture", "none") != texture
            or getattr(self, "_march_axis", (0, 0, 1)) != tuple(march_axis)
        )
        if need_rebuild:
            self.build_model(
                width_model=wm, phase_profile=profile_key,
                texture=texture, march_axis=tuple(march_axis),
            )

        params = self.composite.make_params()

        # ---- Global q-shift ----
        self._init_q_shift(params, q_shift_bound)

        # ---- Per-phase init ----
        sigma_guess = 0.02
        scale_guess = max(float(np.nanmax(self.y_fit)) * sigma_guess, 1.0)
        sigma_cap, sigma_floor = self._sigma_bounds(width_max, width_min)

        for i, (pmodel, lat0) in enumerate(zip(self._phase_models, self._init_lattice)):
            pre = f"p{i}_"
            params[f"{pre}scale"].set(value=scale_guess, min=0.0, vary=True)
            self._init_width_params(params, pre, wm, sigma_guess, sigma_cap, sigma_floor)
            self._init_profile_params(params, pre, profile_key, sigma_cap)
            self._init_texture_params(params, pre, texture, pmodel, pk_scale_range)
            self._init_lattice_params(params, pre, lat0, lattice_pct, lock_lattice_order)

        # ---- Inter-phase lattice ordering (optional) ----
        if lock_cross_phase:
            self._init_cross_phase_order(params, lattice_pct)

        # ---- Background & amorphous ----
        self._init_background_params(params)
        self._init_amorphous_params(params)
        return params

    # ------------------------------------------------------------------
    # build_parameters helper methods
    # ------------------------------------------------------------------

    def _init_q_shift(self, params: Parameters, q_shift_bound: float) -> None:
        """Set up a single global q-shift tied to all phases."""
        if "q_shift" not in params:
            params.add("q_shift", value=0.0, min=-q_shift_bound,
                        max=q_shift_bound, vary=True)
        else:
            params["q_shift"].set(value=0.0, min=-q_shift_bound,
                                   max=q_shift_bound, vary=True)
        for i in range(len(self._phase_models)):
            params[f"p{i}_q_shift"].set(expr="q_shift")

    def _sigma_bounds(
        self, width_max: float | None, width_min: float | None,
    ) -> tuple[float, float]:
        """Return ``(sigma_cap, sigma_floor)`` from user-supplied bounds."""
        data_range = float(np.nanmax(self.x) - np.nanmin(self.x))
        sigma_cap = float(width_max) if width_max and width_max > 0 else data_range / 4.0
        sigma_floor = float(width_min) if width_min and width_min > 0 else 1e-5
        return sigma_cap, sigma_floor

    def _init_width_params(
        self, params: Parameters, pre: str, width_model: str | bool,
        sigma_guess: float, sigma_cap: float, sigma_floor: float,
    ) -> None:
        """Initialise Caglioti U/V/W, Scherrer D/eps, or scalar sigma bounds.

        ``width_model`` may be the canonical string ('caglioti', 'scherrer',
        'fixed') or the legacy bool (True → Caglioti, False → fixed).
        """
        if isinstance(width_model, bool):
            wm = "caglioti" if width_model else "fixed"
        else:
            wm = str(width_model).lower().strip()

        w_cap = sigma_cap ** 2
        w_floor = max(sigma_floor ** 2, 1e-12)

        if wm == "caglioti":
            params[f"{pre}U"].set(value=1e-6, min=0.0, max=1e-2, vary=True)
            params[f"{pre}V"].set(value=0.0, min=-1e-2, max=1e-2, vary=True)
            w_guess = min(max(sigma_guess ** 2, w_floor * 1.1), w_cap * 0.9)
            params[f"{pre}W"].set(value=w_guess, min=w_floor, max=w_cap, vary=True)
        elif wm == "scherrer":
            # D in Å; allowable range roughly [size smaller than 1 nm,
            # coherence length beyond which broadening is negligible].
            # Choose init so the size term matches sigma_guess in FWHM:
            # sigma_guess ≈ 2π·K/(2·D) → D = π·K/sigma_guess
            D_init = float(np.pi) * _SCHERRER_K / max(sigma_guess, 1e-3)
            D_min = float(np.pi) * _SCHERRER_K / max(sigma_cap, 1e-3)
            D_max = 1.0e5  # 10 µm — effectively zero size broadening
            D_init = min(max(D_init, D_min * 1.1), D_max * 0.9)
            params[f"{pre}D"].set(
                value=D_init, min=max(D_min, 1.0), max=D_max, vary=True,
            )
            # Microstrain initial guess 1e-3 (= 0.1%); allow 0..10%.
            params[f"{pre}eps"].set(
                value=1e-3, min=0.0, max=1e-1, vary=True,
            )
        else:  # "fixed"
            sigma_init = min(max(sigma_guess, sigma_floor * 1.1), sigma_cap * 0.9)
            params[f"{pre}sigma"].set(
                value=sigma_init, min=sigma_floor, max=sigma_cap, vary=True,
            )

    def _init_profile_params(
        self, params: Parameters, pre: str,
        profile_key: str, sigma_cap: float,
    ) -> None:
        """Set profile-specific shape parameter bounds (fraction, expo, …)."""
        profile_defaults = dict(_PHASE_PROFILE_SPEC[profile_key]["defaults"])
        _BOUNDS: dict[str, tuple[float, float]] = {
            "fraction":    (0.0,  1.0),
            "skew_gamma":  (-10.0, 10.0),
            "skew":        (-10.0, 10.0),
            "expo":        (0.55, 50.0),
            "moffat_beta": (1.01, 50.0),
        }
        _CAP_TO_SIGMA = {"voigt_gamma", "sigma_r"}
        for name, default in profile_defaults.items():
            pkey = f"{pre}{name}"
            if pkey not in params:
                continue
            if name in _BOUNDS:
                lo, hi = _BOUNDS[name]
                params[pkey].set(value=default, min=lo, max=hi, vary=True)
            elif name in _CAP_TO_SIGMA:
                params[pkey].set(value=default, min=1e-5, max=sigma_cap, vary=True)
            else:
                params[pkey].set(value=default, vary=True)

    def _init_texture_params(
        self, params: Parameters, pre: str,
        texture: str, pmodel: "PhasePeakModel",
        pk_scale_range: tuple[float, float] = (0.0, 10.0),
    ) -> None:
        """Initialise March-Dollase r or per-peak multipliers.

        Parameters
        ----------
        pk_scale_range : tuple of float
            (min, max) bounds for each per-peak multiplier when
            ``texture="free"``.  Default ``(0.0, 10.0)``.
        """
        if texture == "march_dollase":
            pkey = f"{pre}march_r"
            if pkey in params:
                params[pkey].set(value=1.0, min=0.1, max=5.0, vary=True)
        elif texture == "free":
            pk_lo, pk_hi = pk_scale_range
            n_peaks = int(pmodel.hkl.shape[0]) if pmodel.hkl.size else 0
            for j in range(n_peaks):
                pkey = f"{pre}pk{j}"
                if pkey in params:
                    params[pkey].set(value=1.0, min=pk_lo, max=pk_hi, vary=True)

    def _init_lattice_params(
        self, params: Parameters, pre: str,
        lat0: dict | None, lattice_pct: float, lock_order: bool,
    ) -> None:
        """Set lattice constants, symmetry ties, and ordering constraints."""
        if not lat0:
            for key, v0 in (("a", 1.0), ("b", 1.0), ("c", 1.0)):
                params[f"{pre}{key}"].set(value=v0, vary=False)
            for key in ("alpha", "beta", "gamma"):
                params[f"{pre}{key}"].set(value=90.0, vary=False)
            return

        for key in ("a", "b", "c"):
            v0 = lat0[key]
            params[f"{pre}{key}"].set(
                value=v0,
                min=v0 * (1 - lattice_pct),
                max=v0 * (1 + lattice_pct),
                vary=True,
            )
        for key in ("alpha", "beta", "gamma"):
            params[f"{pre}{key}"].set(value=lat0[key], vary=False)

        # Group symmetry-equal axes and chain ordering constraints.
        groups = _group_equal_axes(lat0)
        if lock_order:
            groups = sorted(groups, key=lambda g: lat0[g[0]], reverse=True)
        _chain_groups_with_gaps(params, groups, pre, lat0, lattice_pct, lock_order)

    def _init_cross_phase_order(
        self, params: Parameters, lattice_pct: float,
    ) -> None:
        """Chain the same axis letter across phases by initial value."""
        if not self._init_lattice:
            return
        for key in ("a", "b", "c"):
            indep: list[tuple[int, float]] = []
            for i, lat0 in enumerate(self._init_lattice):
                if not lat0:
                    continue
                p = params.get(f"p{i}_{key}")
                if p is not None and p.expr is None:
                    indep.append((i, float(lat0[key])))
            if len(indep) < 2:
                continue
            indep.sort(key=lambda t: t[1], reverse=True)
            for (prev_i, v_prev), (cur_i, v_cur) in zip(indep[:-1], indep[1:]):
                gap_name = f"cross_gap_{key}_p{prev_i}_p{cur_i}"
                _add_ordering_gap(params, gap_name,
                                  f"p{prev_i}_{key}", f"p{cur_i}_{key}",
                                  v_prev, v_cur, lattice_pct)

    def _init_background_params(self, params: Parameters) -> None:
        """Initialise polynomial/Chebyshev/spline/template background params."""
        if self._bg_model is None:
            return

        # Template background: one positive amplitude.  Start from the
        # ratio between the data-median and the template-median (both
        # evaluated on the fit x-grid), then clamp to a reasonable range.
        has_template = (
            "bg_A" in params and self._fit_background_template is not None
        )
        if has_template:
            x_ref, y_ref = self._fit_background_template
            order = np.argsort(np.asarray(x_ref, dtype=float))
            templ_on_fit = np.interp(
                self.x, np.asarray(x_ref, dtype=float)[order],
                np.asarray(y_ref, dtype=float)[order],
            )
            t_med = float(np.nanmedian(templ_on_fit))
            d_med = float(np.nanmedian(self.y_fit))
            if t_med > 0 and np.isfinite(d_med) and d_med > 0:
                a_guess = max(min(d_med / t_med, 10.0), 1e-3)
            else:
                a_guess = 1.0
            params["bg_A"].set(value=a_guess, min=0.0, vary=True)

        # Initialise any additive polynomial/Chebyshev/spline params.
        # For a plain bg model the prefix is ``bg_``; for the additive
        # correction on top of a template it's ``bgx_``.  Template+extra
        # gets a zero-centred init (it's a correction term), while a
        # standalone background gets the data-midpoint init.
        bg_prefixes = []
        if not has_template:
            bg_prefixes.append(("bg_", False))
        # Extra additive model (if any) always uses the bgx_ prefix.
        if any(k.startswith("bgx_") for k in params):
            bg_prefixes.append(("bgx_", True))

        if not bg_prefixes:
            return

        y_floor = float(np.nanmin(self.y_fit))
        mid = y_floor + 0.5 * float(np.nanmax(self.y_fit) - y_floor)

        for prefix, is_correction in bg_prefixes:
            c0 = f"{prefix}c0"
            if c0 in params:
                params[c0].set(value=(0.0 if is_correction else mid))
                for k in list(params.keys()):
                    if (k.startswith(prefix + "c") and k != c0
                            and k[len(prefix) + 1:].isdigit()):
                        params[k].set(value=0.0)
            else:
                init_v = 0.0 if is_correction else mid
                for k in list(params.keys()):
                    if (k.startswith(prefix + "s")
                            and k[len(prefix) + 1:].isdigit()):
                        params[k].set(value=init_v)

    def _init_amorphous_params(self, params: Parameters) -> None:
        """Initialise the optional amorphous peak (center, sigma, amplitude)."""
        if self._amorphous_model is None:
            return
        y_span = float(np.nanmax(self.y_fit) - np.nanmin(self.y_fit))
        defaults = dict(center=1.5, sigma=0.3, amplitude=max(y_span, 1.0))
        defaults.update(self._amorphous_init)
        for key, val in defaults.items():
            pkey = f"am_{key}"
            if pkey in params:
                params[pkey].set(value=float(val))
        for pos_key in ("am_amplitude", "am_sigma"):
            if pos_key in params:
                pv = params[pos_key].value or 0.0
                params[pos_key].set(min=0.0, value=max(pv, 1e-6))
        if "am_center" in params:
            params["am_center"].set(
                min=float(np.nanmin(self.x)),
                max=float(np.nanmax(self.x)),
            )

    # ------------------------------------------------------------------
    # Model evaluation (used by legacy .plot and by tests)
    # ------------------------------------------------------------------

    def eval_model(self, params: Parameters) -> np.ndarray:
        """Evaluate the fit composite at ``self.x`` **plus** the prefit
        baseline, so the result is directly comparable to ``self.y``.
        """
        if self.composite is None:
            self.build_model()
        y_fit_model = self.composite.eval(params=params, x=self.x)
        return np.asarray(y_fit_model) + self.background

    def eval_phase(self, idx: int, params: Parameters) -> np.ndarray:
        """Evaluate a single phase's contribution at ``self.x``."""
        if self.composite is None:
            self.build_model()
        return np.asarray(self._phase_models[idx].eval(params=params, x=self.x))

    def eval_amorphous(self, params: Parameters) -> np.ndarray | None:
        if self._amorphous_model is None:
            return None
        return np.asarray(self._amorphous_model.eval(params=params, x=self.x))

    def eval_fit_background(self, params: Parameters) -> np.ndarray | None:
        if self._bg_model is None:
            return None
        return np.asarray(self._bg_model.eval(params=params, x=self.x))

    # ------------------------------------------------------------------
    # Fit driver
    # ------------------------------------------------------------------

    def fit(
        self,
        params: Parameters | None = None,
        method: str = "leastsq",
        caglioti: bool | None = None,
        phase_profile: str = "pseudovoigt",
        q_shift_bound: float = 0.05,
        lattice_pct: float = 0.05,
        width_max: float | None = None,
        width_min: float | None = None,
        lock_lattice_order: bool = True,
        lock_cross_phase: bool = False,
        texture: str = "none",
        march_axis: tuple[int, int, int] = (0, 0, 1),
        pk_scale_range: tuple[float, float] = (0.0, 10.0),
        q_range: tuple[float, float] | None = None,
        width_model: str | None = None,
        **fit_kwargs: Any,
    ) -> MultiPhaseResult:
        """Run the multi-phase fit.

        Parameters
        ----------
        params : lmfit.Parameters or None
            Pre-built parameters.  ``None`` → :meth:`build_parameters` is
            called using the other kwargs.
        method : str
            lmfit minimisation method (``'leastsq'`` by default).
        caglioti, phase_profile, q_shift_bound, lattice_pct, width_max :
            Forwarded to :meth:`build_parameters` when ``params`` is None.
        texture, march_axis :
            Forwarded to :meth:`build_parameters` and :meth:`build_model`.
        q_range : tuple(qmin, qmax) or None
            Restrict the fit to the given q interval.  Evaluation helpers
            (:meth:`eval_model`, :meth:`eval_phase`, …) still return values
            over the full ``self.x`` grid; only the minimisation objective
            is restricted.  The mask used is stored on
            ``self.fit_mask`` for callers (e.g. the Panel viewer) that
            want to visualise which points participated.
        **fit_kwargs :
            Extra keyword args for :meth:`lmfit.Model.fit` (e.g.
            ``max_nfev``, ``scale_covar``, ``nan_policy``).
        """
        if self.composite is None:
            self.build_model()

        if params is None:
            params = self.build_parameters(
                q_shift_bound=q_shift_bound,
                lattice_pct=lattice_pct,
                caglioti=caglioti,
                phase_profile=phase_profile,
                width_max=width_max,
                width_min=width_min,
                lock_lattice_order=lock_lattice_order,
                lock_cross_phase=lock_cross_phase,
                texture=texture,
                march_axis=march_axis,
                pk_scale_range=pk_scale_range,
                width_model=width_model,
            )

        # Restrict the fit domain if requested.
        if q_range is not None:
            qmin, qmax = float(q_range[0]), float(q_range[1])
            if qmin > qmax:
                qmin, qmax = qmax, qmin
            mask = (self.x >= qmin) & (self.x <= qmax)
        else:
            mask = np.ones_like(self.x, dtype=bool)
        self.fit_mask = mask

        x_fit = self.x[mask]
        y_fit_slice = self.y_fit[mask]
        sigma_slice = self.sigma[mask] if self.sigma is not None else None

        weights = None
        if sigma_slice is not None:
            weights = np.where(sigma_slice > 0, 1.0 / sigma_slice, 1.0)

        lm_result = self.composite.fit(
            y_fit_slice, params=params, x=x_fit,
            method=method, weights=weights, **fit_kwargs,
        )
        return MultiPhaseResult(lm_result, self)

    # ------------------------------------------------------------------
    # Reporting / plotting (legacy matplotlib path kept for notebooks)
    # ------------------------------------------------------------------

    def report(self, result: MultiPhaseResult) -> str:
        txt = result.summary()
        print(txt)
        return txt

