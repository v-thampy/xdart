"""GUI-side home for the analysis-launcher entry points (H16).

The headless ``xrd_tools.session.readiness.build_analysis_launchers`` describes
WHICH auxiliary analysis tools exist and WHEN each is enabled — by
``AnalysisTool`` identity plus result capabilities — but it must not name xdart
GUI classes.  Carrying ``xdart.gui.*`` dotted paths inside the headless reduction
core was a layering inversion (the Qt-free ``xrd_tools`` layer naming its GUI
consumer); H16 extracts those paths here, to the GUI side.

This module owns the ``AnalysisTool -> dialog entry point`` binding: the dotted
``module:Class`` path of the Qt dialog that services each tool.  Adding a new
analysis launcher is now a GUI-side addition (a binding entry here + the tool's
gating in the headless ``AnalysisTool`` enum + its opener), never an edit that
makes the headless core import-name a GUI class.

The live launch dispatch (``staticWidget._on_controls_v2_analysis_launch``) opens
these dialogs through their rich, hand-written openers; this binding is the
canonical registry of *where* each tool's dialog lives — consumed by tests today
and available to the future generic/introspective launcher (the "AnalysisContext"
binding named in the H16 plan).  Paths stay lazy strings (not eager class
imports) so binding this module does not drag in the heavy dialog dependencies.
"""
from __future__ import annotations

from xrd_tools.session.readiness import AnalysisTool

# AnalysisTool -> "module:Class" dotted path of the Qt dialog that services it.
# ROI_STATS reuses the ScanPlot dialog (its ROI reduction lives there).
ANALYSIS_LAUNCHER_ENTRY_POINTS: dict[AnalysisTool, str] = {
    AnalysisTool.PEAK_FIT: "xdart.gui.tabs.static_scan.peak_fit_dialog:PeakFitDialog",
    AnalysisTool.PHASE_FIT: "xdart.gui.tabs.static_scan.phase_fit_dialog:PhaseFitDialog",
    AnalysisTool.SCAN_PLOT: "xdart.gui.tabs.static_scan.scan_plot_dialog:ScanPlotDialog",
    AnalysisTool.ROI_STATS: "xdart.gui.tabs.static_scan.scan_plot_dialog:ScanPlotDialog",
    AnalysisTool.SIN2PSI: "xdart.gui.tabs.static_scan.strain_dialog:StrainDialog",
    AnalysisTool.TEXTURE: "xdart.gui.tabs.static_scan.texture_dialog:TextureDialog",
}


def analysis_launcher_entry_point(tool: AnalysisTool) -> str | None:
    """Return *tool*'s dialog as a ``"module:Class"`` dotted path, or ``None``
    when the tool has no GUI dialog bound (unknown / not-yet-wired)."""
    return ANALYSIS_LAUNCHER_ENTRY_POINTS.get(tool)


__all__ = ["ANALYSIS_LAUNCHER_ENTRY_POINTS", "analysis_launcher_entry_point"]
