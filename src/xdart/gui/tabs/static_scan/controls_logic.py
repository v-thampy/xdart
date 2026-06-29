# -*- coding: utf-8 -*-
"""Pure control-profile logic for the static-scan GUI.

This module is intentionally Qt-free.  It is the small decision layer behind
the forthcoming Controls Panel V2: source/result capabilities in, profile and
analysis launchers out.  Keeping this logic separate lets the GUI expose
experimental Stitch/RSM/GI controls as ready-to-wire scaffolding while keeping
run buttons gated until their real-data acceptance checks land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tool(str, Enum):
    INT_1D = "int_1d"
    INT_2D = "int_2d"
    IMAGE_VIEWER = "image_viewer"
    XYE_VIEWER = "xye_viewer"
    NEXUS_VIEWER = "nexus_viewer"
    STITCH = "stitch"
    RSM = "rsm"


class MeasMode(str, Enum):
    STANDARD = "standard"
    GI = "gi"


class ProcessingPage(str, Enum):
    INT_1D = "int_1d"
    INT_2D = "int_2d"
    VIEWER = "viewer"
    STITCH = "stitch"
    STITCH_GI = "stitch_gi"
    RSM = "rsm"


class AnalysisTool(str, Enum):
    PEAK_FIT = "peak_fit"
    PHASE_FIT = "phase_fit"
    SCAN_PLOT = "scan_plot"
    ROI_STATS = "roi_stats"
    SIN2PSI = "sin2psi"
    TEXTURE = "texture"


@dataclass(frozen=True, slots=True)
class SourceCaps:
    has_frames: bool = False
    has_raw: bool = False
    raw_reachable: bool = False
    has_metadata: bool = False
    has_motors: bool = False
    has_energy: bool = False
    has_geometry: bool = False
    has_psi_metadata: bool = False


@dataclass(frozen=True, slots=True)
class ResultCaps:
    has_1d: bool = False
    has_2d: bool = False
    has_raw: bool = False
    raw_reachable: bool = False
    has_scan_metadata: bool = False
    has_rsm: bool = False
    has_phase_result: bool = False
    has_psi_metadata: bool = False
    # None means "unknown / let the dialog show its own friendly dependency
    # hint".  A concrete frozenset means the launcher can gate on it.
    available_optional_deps: frozenset[str] | None = None

    def optional_dep_known(self) -> bool:
        return self.available_optional_deps is not None

    def has_optional_dep(self, name: str) -> bool:
        deps = self.available_optional_deps
        return deps is None or name in deps or "all" in deps


@dataclass(frozen=True, slots=True)
class GeomState:
    calibrated: bool = False
    energy_known: bool = False
    gi_enabled: bool = False
    sample_orientation_known: bool = False
    ub_known: bool = False
    material_known: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisLauncherSpec:
    tool: AnalysisTool
    label: str
    enabled: bool = True
    reason: str = ""
    live_capable: bool = False
    batch_capable: bool = False
    production_ready: bool = True


@dataclass(frozen=True, slots=True)
class ControlState:
    tool: Tool = Tool.INT_1D
    mode: MeasMode = MeasMode.STANDARD
    source_caps: SourceCaps = field(default_factory=SourceCaps)
    result_caps: ResultCaps = field(default_factory=ResultCaps)
    geom: GeomState = field(default_factory=GeomState)
    backend: str | None = None
    real_data_gates: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class ControlProfile:
    processing_page: ProcessingPage
    run_enabled: bool
    run_blockers: tuple[str, ...] = ()
    analysis_launchers: tuple[AnalysisLauncherSpec, ...] = ()
    show_experiment_card: bool = True
    show_processing_card: bool = True


def _disabled(tool: AnalysisTool, label: str, reason: str, *,
              live: bool = False, batch: bool = False,
              production_ready: bool = True) -> AnalysisLauncherSpec:
    return AnalysisLauncherSpec(
        tool=tool, label=label, enabled=False, reason=reason,
        live_capable=live, batch_capable=batch,
        production_ready=production_ready)


def _enabled(tool: AnalysisTool, label: str, *,
             reason: str = "", live: bool = False, batch: bool = False,
             production_ready: bool = True) -> AnalysisLauncherSpec:
    return AnalysisLauncherSpec(
        tool=tool, label=label, enabled=True, reason=reason,
        live_capable=live, batch_capable=batch,
        production_ready=production_ready)


def build_analysis_launchers(caps: ResultCaps) -> tuple[AnalysisLauncherSpec, ...]:
    """Return launcher state for auxiliary popup tools.

    Peak/Phase dialogs remain openable before data exists so they can attach to
    a live run.  Future tools with stricter data requirements stay disabled
    until the required result/metadata is present.
    """

    peak = (_enabled(AnalysisTool.PEAK_FIT, "Peak Fitting",
                     reason="Waiting for a 1D pattern." if not caps.has_1d else "",
                     live=True, batch=True)
            if caps.has_optional_dep("fitting")
            else _disabled(AnalysisTool.PEAK_FIT, "Peak Fitting",
                           "Install fitting dependencies.", live=True, batch=True))

    phase = (_enabled(AnalysisTool.PHASE_FIT, "Phase Fitting",
                      reason="Waiting for a 1D pattern." if not caps.has_1d else "",
                      batch=True)
             if caps.has_optional_dep("fitting")
             else _disabled(AnalysisTool.PHASE_FIT, "Phase Fitting",
                            "Install fitting dependencies.", batch=True))

    scan_plot = _enabled(
        AnalysisTool.SCAN_PLOT, "Plot Metadata",
        reason="" if caps.has_scan_metadata else "Choose a scan/source in the dialog.",
        batch=True)

    roi = (_enabled(AnalysisTool.ROI_STATS, "ROI Statistics", batch=True)
           if caps.raw_reachable
           else _disabled(AnalysisTool.ROI_STATS, "ROI Statistics",
                          "Raw frames are not reachable for ROI reduction.",
                          batch=True))

    strain_ready = caps.has_1d and caps.has_psi_metadata
    strain = (_enabled(AnalysisTool.SIN2PSI, "Strain / sin²ψ",
                       live=False, batch=True, production_ready=False)
              if strain_ready
              else _disabled(AnalysisTool.SIN2PSI, "Strain / sin²ψ",
                             "Needs 1D patterns with ψ metadata.",
                             batch=True, production_ready=False))

    texture_ready = caps.has_1d or caps.has_phase_result
    texture = (_enabled(AnalysisTool.TEXTURE, "Texture / Orientation",
                        batch=True, production_ready=False)
               if texture_ready
               else _disabled(AnalysisTool.TEXTURE, "Texture / Orientation",
                              "Needs fitted phases or suitable 1D patterns.",
                              batch=True, production_ready=False))
    return peak, phase, scan_plot, roi, strain, texture


def processing_page_for(tool: Tool, mode: MeasMode) -> ProcessingPage:
    if tool == Tool.INT_1D:
        return ProcessingPage.INT_1D
    if tool == Tool.INT_2D:
        return ProcessingPage.INT_2D
    if tool in (Tool.IMAGE_VIEWER, Tool.XYE_VIEWER, Tool.NEXUS_VIEWER):
        return ProcessingPage.VIEWER
    if tool == Tool.STITCH:
        return ProcessingPage.STITCH_GI if mode == MeasMode.GI else ProcessingPage.STITCH
    if tool == Tool.RSM:
        return ProcessingPage.RSM
    return ProcessingPage.INT_1D


def run_blockers_for(state: ControlState) -> tuple[str, ...]:
    blockers: list[str] = []
    caps = state.source_caps
    geom = state.geom
    if state.tool in (Tool.INT_1D, Tool.INT_2D, Tool.STITCH, Tool.RSM):
        if not caps.has_frames:
            blockers.append("Choose a frame source.")
        if not geom.calibrated:
            blockers.append("Load detector calibration.")
        if not geom.energy_known and not caps.has_energy:
            blockers.append("Set calibration wavelength or source energy.")
    if state.mode == MeasMode.GI:
        if not geom.sample_orientation_known:
            blockers.append("Set GI sample orientation.")
    if state.tool == Tool.STITCH:
        if state.mode == MeasMode.GI and "gi_stitch_real_data" not in state.real_data_gates:
            blockers.append("GI stitching awaits real-data gate.")
        if state.backend == "xu_hist" and "xu_hist_real_data" not in state.real_data_gates:
            blockers.append("xu_hist stitching awaits real-data gate.")
    if state.tool == Tool.RSM:
        if not caps.has_motors:
            blockers.append("RSM needs motor metadata.")
        if not geom.ub_known:
            blockers.append("Set/refine UB matrix before RSM.")
        if "rsm_real_data" not in state.real_data_gates:
            blockers.append("RSM GUI awaits real-data gate.")
    return tuple(dict.fromkeys(blockers))


def build_control_profile(state: ControlState) -> ControlProfile:
    page = processing_page_for(state.tool, state.mode)
    blockers = run_blockers_for(state)
    viewer = page == ProcessingPage.VIEWER
    return ControlProfile(
        processing_page=page,
        run_enabled=not blockers and not viewer,
        run_blockers=blockers,
        analysis_launchers=build_analysis_launchers(state.result_caps),
        show_experiment_card=True,
        show_processing_card=not viewer)


# Public spelling used by the design docs.  Keep the longer name for call sites
# that prefer explicitness during the transition.
build_profile = build_control_profile
