"""Headless analysis API with lazy backend imports.

Importing :mod:`xrd_tools.analysis` should be cheap and GUI-free.  The
fitting, strain, and plotting-adjacent implementations are loaded only when a
specific symbol is first requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "fit_line_cut": ("xrd_tools.analysis.fitting", "fit_line_cut"),
    "fit_peaks": ("xrd_tools.analysis.fitting", "fit_peaks"),
    "fit_2d_slice": ("xrd_tools.analysis.fitting", "fit_2d_slice"),
    "PeakFitResult1D": ("xrd_tools.analysis.fitting", "PeakFitResult1D"),
    "PhaseModel": ("xrd_tools.analysis.phase", "PhaseModel"),
    "PeakData": ("xrd_tools.analysis.phase", "PeakData"),
    "PhaseFitter": ("xrd_tools.analysis.fitting.phase_fitting", "PhaseFitter"),
    "MultiPhaseResult": (
        "xrd_tools.analysis.fitting.phase_fitting",
        "MultiPhaseResult",
    ),
    "ChiSector": ("xrd_tools.analysis.strain", "ChiSector"),
    "PeakFitResult": ("xrd_tools.analysis.strain", "PeakFitResult"),
    "Sin2PsiResult": ("xrd_tools.analysis.strain", "Sin2PsiResult"),
    "extract_chi_sectors": (
        "xrd_tools.analysis.strain",
        "extract_chi_sectors",
    ),
    "fit_peak_vs_psi": ("xrd_tools.analysis.strain", "fit_peak_vs_psi"),
    "sin2psi_regression": (
        "xrd_tools.analysis.strain",
        "sin2psi_regression",
    ),
    "sin2psi_analysis": ("xrd_tools.analysis.strain", "sin2psi_analysis"),
    "AnalysisResult": ("xrd_tools.analysis.plans", "AnalysisResult"),
    "PeakFitPlan": ("xrd_tools.analysis.plans", "PeakFitPlan"),
    "PhaseFitPlan": ("xrd_tools.analysis.plans", "PhaseFitPlan"),
    "RSMPlan": ("xrd_tools.analysis.plans", "RSMPlan"),
    "Sin2PsiPlan": ("xrd_tools.analysis.plans", "Sin2PsiPlan"),
    "StitchPlan": ("xrd_tools.analysis.plans", "StitchPlan"),
    "RoiSpec": ("xrd_tools.core.roi", "RoiSpec"),
    "RoiStatsPlan": ("xrd_tools.analysis.plans", "RoiStatsPlan"),
    "RoiStatsResult": ("xrd_tools.analysis.plans", "RoiStatsResult"),
    "run_roi_stats": ("xrd_tools.analysis.plans", "run_roi_stats"),
    "make_phase_fitter": ("xrd_tools.analysis.plans", "make_phase_fitter"),
    "run_peak_fit": ("xrd_tools.analysis.plans", "run_peak_fit"),
    "run_phase_fit": ("xrd_tools.analysis.plans", "run_phase_fit"),
    "run_rsm": ("xrd_tools.analysis.plans", "run_rsm"),
    "run_sin2psi": ("xrd_tools.analysis.plans", "run_sin2psi"),
    "run_stitch": ("xrd_tools.analysis.plans", "run_stitch"),
    # Analysis-agnostic live/batch runner contract.
    "AnalysisInput": ("xrd_tools.analysis.runner", "AnalysisInput"),
    "Overlay": ("xrd_tools.analysis.runner", "Overlay"),
    "AnalysisOutcome": ("xrd_tools.analysis.runner", "AnalysisOutcome"),
    "Analyzer": ("xrd_tools.analysis.runner", "Analyzer"),
    "PeakFitAnalyzer": ("xrd_tools.analysis.runner", "PeakFitAnalyzer"),
    "Sin2PsiAnalyzer": ("xrd_tools.analysis.runner", "Sin2PsiAnalyzer"),
    "PhaseFitAnalyzer": ("xrd_tools.analysis.runner", "PhaseFitAnalyzer"),
    "run_batch": ("xrd_tools.analysis.runner", "run_batch"),
    "batch_params_table": ("xrd_tools.analysis.runner", "batch_params_table"),
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
