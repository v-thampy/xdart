"""
Peak fitting, background subtraction, and peak finding for XRD data.

Two 1D fitting modes:

- **Structure-agnostic**: :func:`fit_peaks` / :func:`fit_line_cut` — fit
  individual peaks with explicit or auto-detected positions.
- **Structure-informed**: :class:`PhaseFitter` — multi-phase fitting using
  CIF-derived peak positions from :class:`PhaseModel`.

Both share the same lmfit model zoo (:mod:`models`) and background tools
(:mod:`background`).
"""
from .fit import fit_line_cut, fit_peaks, fit_2d_slice, PeakFitResult1D, get_peak_model
from .models import (
    LorentzianSquaredModel,
    AsymmetricRectangleModel,
    AssymetricRectangleModel,  # backwards-compatible alias
    Gaussian2DModel,
    LorentzianSquared2DModel,
    PseudoVoigt2DModel,
    Pvoigt2DModel,             # backwards-compatible alias
    PlaneModel,
    lorentzian_squared,
    gauss_2D,
    lor2_2D,
    pvoigt_2D,
)
from .background import snip_1d, chebyshev_background, fit_background, subtract_background
from .peaks import extract_peaks, peak_table
from .phase_fitting import PhaseFitter, MultiPhaseResult
from .batch import FitConfig, FitResultStore, fit_sequence
