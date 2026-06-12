"""Headless analysis API with lazy backend imports.

Importing :mod:`ssrl_xrd_tools.analysis` should be cheap and GUI-free.  The
fitting, strain, and plotting-adjacent implementations are loaded only when a
specific symbol is first requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "fit_line_cut": ("ssrl_xrd_tools.analysis.fitting", "fit_line_cut"),
    "fit_peaks": ("ssrl_xrd_tools.analysis.fitting", "fit_peaks"),
    "fit_2d_slice": ("ssrl_xrd_tools.analysis.fitting", "fit_2d_slice"),
    "PeakFitResult1D": ("ssrl_xrd_tools.analysis.fitting", "PeakFitResult1D"),
    "PhaseModel": ("ssrl_xrd_tools.analysis.phase", "PhaseModel"),
    "PeakData": ("ssrl_xrd_tools.analysis.phase", "PeakData"),
    "PhaseFitter": ("ssrl_xrd_tools.analysis.fitting.phase_fitting", "PhaseFitter"),
    "MultiPhaseResult": (
        "ssrl_xrd_tools.analysis.fitting.phase_fitting",
        "MultiPhaseResult",
    ),
    "ChiSector": ("ssrl_xrd_tools.analysis.strain", "ChiSector"),
    "PeakFitResult": ("ssrl_xrd_tools.analysis.strain", "PeakFitResult"),
    "Sin2PsiResult": ("ssrl_xrd_tools.analysis.strain", "Sin2PsiResult"),
    "extract_chi_sectors": (
        "ssrl_xrd_tools.analysis.strain",
        "extract_chi_sectors",
    ),
    "fit_peak_vs_psi": ("ssrl_xrd_tools.analysis.strain", "fit_peak_vs_psi"),
    "sin2psi_regression": (
        "ssrl_xrd_tools.analysis.strain",
        "sin2psi_regression",
    ),
    "sin2psi_analysis": ("ssrl_xrd_tools.analysis.strain", "sin2psi_analysis"),
    "AnalysisResult": ("ssrl_xrd_tools.analysis.plans", "AnalysisResult"),
    "PeakFitPlan": ("ssrl_xrd_tools.analysis.plans", "PeakFitPlan"),
    "PhaseFitPlan": ("ssrl_xrd_tools.analysis.plans", "PhaseFitPlan"),
    "RSMPlan": ("ssrl_xrd_tools.analysis.plans", "RSMPlan"),
    "Sin2PsiPlan": ("ssrl_xrd_tools.analysis.plans", "Sin2PsiPlan"),
    "StitchPlan": ("ssrl_xrd_tools.analysis.plans", "StitchPlan"),
    "make_phase_fitter": ("ssrl_xrd_tools.analysis.plans", "make_phase_fitter"),
    "run_peak_fit": ("ssrl_xrd_tools.analysis.plans", "run_peak_fit"),
    "run_phase_fit": ("ssrl_xrd_tools.analysis.plans", "run_phase_fit"),
    "run_rsm": ("ssrl_xrd_tools.analysis.plans", "run_rsm"),
    "run_sin2psi": ("ssrl_xrd_tools.analysis.plans", "run_sin2psi"),
    "run_stitch": ("ssrl_xrd_tools.analysis.plans", "run_stitch"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
