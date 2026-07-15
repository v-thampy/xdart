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
    "canonical_q_unit": ("xrd_tools.analysis.axis_units", "canonical_q_unit"),
    "require_inverse_angstrom": (
        "xrd_tools.analysis.axis_units",
        "require_inverse_angstrom",
    ),
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
    "RoiSignal": ("xrd_tools.analysis.plans", "RoiSignal"),
    "RoiStatsPlan": ("xrd_tools.analysis.plans", "RoiStatsPlan"),
    "RoiStatsResult": ("xrd_tools.analysis.plans", "RoiStatsResult"),
    "run_roi_stats": ("xrd_tools.analysis.plans", "run_roi_stats"),
    "run_roi_signals": ("xrd_tools.analysis.plans", "run_roi_signals"),
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
    # Time-resolved series, preprocessing, fitting and thermal metrology.
    "FrameLocator": ("xrd_tools.analysis.time_resolved", "FrameLocator"),
    "TimeResolvedSeries": (
        "xrd_tools.analysis.time_resolved", "TimeResolvedSeries"),
    "LinearThermalExpansion": (
        "xrd_tools.analysis.time_resolved", "LinearThermalExpansion"),
    "TabulatedThermalExpansion": (
        "xrd_tools.analysis.time_resolved", "TabulatedThermalExpansion"),
    "discover_processed_scans": (
        "xrd_tools.analysis.time_resolved", "discover_processed_scans"),
    "load_time_resolved_series": (
        "xrd_tools.analysis.time_resolved", "load_time_resolved_series"),
    "normalize_monitor": (
        "xrd_tools.analysis.time_resolved", "normalize_monitor"),
    "normalize_reference_band": (
        "xrd_tools.analysis.time_resolved", "normalize_reference_band"),
    "flag_normalization_outliers": (
        "xrd_tools.analysis.time_resolved", "flag_normalization_outliers"),
    "bin_time_resolved": (
        "xrd_tools.analysis.time_resolved", "bin_time_resolved"),
    "select_time_zero": (
        "xrd_tools.analysis.time_resolved", "select_time_zero"),
    "fit_peak_series": (
        "xrd_tools.analysis.time_resolved", "fit_peak_series"),
    "flag_fit_quality": (
        "xrd_tools.analysis.time_resolved", "flag_fit_quality"),
    "lattice_from_q": (
        "xrd_tools.analysis.time_resolved", "lattice_from_q"),
    "add_lattice_results": (
        "xrd_tools.analysis.time_resolved", "add_lattice_results"),
    "temperature_rate": (
        "xrd_tools.analysis.time_resolved", "temperature_rate"),
    "add_temperature_results": (
        "xrd_tools.analysis.time_resolved", "add_temperature_results"),
    "export_time_resolved_results": (
        "xrd_tools.analysis.time_resolved", "export_time_resolved_results"),
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
