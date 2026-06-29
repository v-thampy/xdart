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
from types import MappingProxyType
from typing import Mapping


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


class SectionId(str, Enum):
    SOURCE = "source"
    EXPERIMENT = "experiment"
    PROCESSING = "processing"
    OUTPUT = "output"
    ANALYSIS = "analysis"


class ControlAction(str, Enum):
    """Intent emitted by the Controls V2 preview.

    The Qt widget renders these as buttons, but the static-scan tab owns the
    actual routing while V2 is hidden.  Keeping the action list pure avoids
    giving the preview direct access to wrangler/integrator internals.
    """

    CHOOSE_SOURCE = "choose_source"
    CALIBRATE = "calibrate"
    MAKE_MASK = "make_mask"
    REFINE_GEOMETRY = "refine_geometry"
    ADVANCED_PROCESSING = "advanced_processing"


class FieldId(str, Enum):
    SOURCE_PATH = "source.path"
    SOURCE_FRAMES = "source.frames"
    SOURCE_RAW = "source.raw"
    SOURCE_METADATA = "source.metadata"
    SOURCE_MOTORS = "source.motors"
    CALIBRATION_PONI = "calibration.poni"
    DETECTOR_MASK = "detector.mask"
    BEAM_ENERGY = "beam.energy"
    SAMPLE_GI = "sample.gi"
    SAMPLE_ORIENTATION = "sample.orientation"
    DIFFRACTOMETER_UB = "diffractometer.ub"
    PROCESSING_MODE = "processing.mode"
    PROCESSING_BACKEND = "processing.backend"
    OUTPUT_SAVE_PATH = "output.save_path"
    RESULT_1D = "result.1d"
    RESULT_2D = "result.2d"
    RESULT_RSM = "result.rsm"


class StatusKind(str, Enum):
    OK = "ok"
    MISSING = "missing"
    INFERRED = "inferred"
    SAVED = "saved"
    CONFLICT = "conflict"
    LOCKED = "locked"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    field_id: FieldId
    section: SectionId
    label: str
    owner: str
    session_key: str = ""
    headless_key: str = ""


FIELD_INVENTORY: tuple[FieldSpec, ...] = (
    FieldSpec(FieldId.SOURCE_PATH, SectionId.SOURCE, "Source", "source"),
    FieldSpec(FieldId.SOURCE_FRAMES, SectionId.SOURCE, "Frames", "source",
              headless_key="FrameSource.frame_indices"),
    FieldSpec(FieldId.SOURCE_RAW, SectionId.SOURCE, "Raw images", "source",
              headless_key="FrameSource.load_frame"),
    FieldSpec(FieldId.SOURCE_METADATA, SectionId.SOURCE, "Metadata", "source",
              headless_key="FrameSource.metadata_for"),
    FieldSpec(FieldId.SOURCE_MOTORS, SectionId.SOURCE, "Motors", "source",
              headless_key="FrameSource.motors"),
    FieldSpec(FieldId.CALIBRATION_PONI, SectionId.EXPERIMENT, "PONI", "experiment",
              session_key="integrator.poni_file", headless_key="geometry.poni"),
    FieldSpec(FieldId.DETECTOR_MASK, SectionId.EXPERIMENT, "Mask", "experiment",
              session_key="wrangler.mask_file", headless_key="mask"),
    FieldSpec(FieldId.BEAM_ENERGY, SectionId.EXPERIMENT, "Beam energy", "experiment",
              headless_key="diffractometer.wavelength"),
    FieldSpec(FieldId.SAMPLE_GI, SectionId.EXPERIMENT, "Grazing incidence", "experiment",
              session_key="integrator.gi", headless_key="gi.enabled"),
    FieldSpec(FieldId.SAMPLE_ORIENTATION, SectionId.EXPERIMENT, "Sample orientation",
              "experiment", session_key="integrator.gi.sample_orientation",
              headless_key="gi.sample_orientation"),
    FieldSpec(FieldId.DIFFRACTOMETER_UB, SectionId.EXPERIMENT, "UB matrix",
              "experiment", headless_key="diffractometer.ub"),
    FieldSpec(FieldId.PROCESSING_MODE, SectionId.PROCESSING, "Mode", "processing",
              session_key="processing_mode"),
    FieldSpec(FieldId.PROCESSING_BACKEND, SectionId.PROCESSING, "Backend",
              "processing", headless_key="StitchPlan.backend"),
    FieldSpec(FieldId.OUTPUT_SAVE_PATH, SectionId.OUTPUT, "Save path", "output",
              session_key="wrangler.h5_dir"),
    FieldSpec(FieldId.RESULT_1D, SectionId.ANALYSIS, "1D result", "analysis"),
    FieldSpec(FieldId.RESULT_2D, SectionId.ANALYSIS, "2D result", "analysis"),
    FieldSpec(FieldId.RESULT_RSM, SectionId.ANALYSIS, "RSM result", "analysis"),
)


FIELD_SPECS: Mapping[FieldId, FieldSpec] = MappingProxyType(
    {spec.field_id: spec for spec in FIELD_INVENTORY}
)


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
class ControlActionSpec:
    action: ControlAction
    label: str
    section: SectionId
    enabled: bool = True
    reason: str = ""
    production_ready: bool = True


@dataclass(frozen=True, slots=True)
class FieldStatus:
    field_id: FieldId
    label: str
    section: SectionId
    status: StatusKind
    value: str = ""
    reason: str = ""
    owner: str = ""
    session_key: str = ""
    headless_key: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {
            StatusKind.OK,
            StatusKind.INFERRED,
            StatusKind.SAVED,
            StatusKind.DEFERRED,
        }


@dataclass(frozen=True, slots=True)
class ControlState:
    tool: Tool = Tool.INT_1D
    mode: MeasMode = MeasMode.STANDARD
    source_caps: SourceCaps = field(default_factory=SourceCaps)
    result_caps: ResultCaps = field(default_factory=ResultCaps)
    geom: GeomState = field(default_factory=GeomState)
    backend: str | None = None
    source_label: str = ""
    save_path: str = ""
    frame_count: int = 0
    processing_mode: str = ""
    real_data_gates: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class ControlProfile:
    processing_page: ProcessingPage
    run_enabled: bool
    run_blockers: tuple[str, ...] = ()
    fields: Mapping[FieldId, FieldStatus] = field(default_factory=dict)
    section_actions: Mapping[SectionId, tuple[ControlActionSpec, ...]] = (
        field(default_factory=dict)
    )
    analysis_launchers: tuple[AnalysisLauncherSpec, ...] = ()
    show_experiment_card: bool = True
    show_processing_card: bool = True

    def fields_for(self, section: SectionId) -> tuple[FieldStatus, ...]:
        return tuple(
            status for spec in FIELD_INVENTORY
            if spec.section == section
            if (status := self.fields.get(spec.field_id)) is not None
        )

    def actions_for(self, section: SectionId) -> tuple[ControlActionSpec, ...]:
        return tuple(self.section_actions.get(section, ()))


def _field(
    field_id: FieldId,
    status: StatusKind,
    *,
    value: str = "",
    reason: str = "",
) -> FieldStatus:
    spec = FIELD_SPECS[field_id]
    return FieldStatus(
        field_id=field_id,
        label=spec.label,
        section=spec.section,
        status=status,
        value=value,
        reason=reason,
        owner=spec.owner,
        session_key=spec.session_key,
        headless_key=spec.headless_key,
    )


def build_field_statuses(state: ControlState) -> Mapping[FieldId, FieldStatus]:
    caps = state.source_caps
    geom = state.geom
    results = state.result_caps
    source_label = state.source_label or "No source selected"
    frame_value = str(state.frame_count) if state.frame_count else ""
    fields = {
        FieldId.SOURCE_PATH: _field(
            FieldId.SOURCE_PATH,
            StatusKind.OK if caps.has_frames or source_label != "No source selected"
            else StatusKind.MISSING,
            value=source_label,
            reason="" if source_label != "No source selected" else "Choose data."),
        FieldId.SOURCE_FRAMES: _field(
            FieldId.SOURCE_FRAMES,
            StatusKind.OK if caps.has_frames else StatusKind.MISSING,
            value=frame_value,
            reason="" if caps.has_frames else "No frame source is loaded."),
        FieldId.SOURCE_RAW: _field(
            FieldId.SOURCE_RAW,
            StatusKind.OK if caps.raw_reachable else (
                StatusKind.INFERRED if caps.has_raw else StatusKind.MISSING),
            value="reachable" if caps.raw_reachable else ("present" if caps.has_raw else ""),
            reason="" if caps.raw_reachable else "Raw frames are not reachable."),
        FieldId.SOURCE_METADATA: _field(
            FieldId.SOURCE_METADATA,
            StatusKind.OK if caps.has_metadata else StatusKind.MISSING,
            value="available" if caps.has_metadata else "",
            reason="" if caps.has_metadata else "Metadata not detected."),
        FieldId.SOURCE_MOTORS: _field(
            FieldId.SOURCE_MOTORS,
            StatusKind.OK if caps.has_motors else StatusKind.MISSING,
            value="available" if caps.has_motors else "",
            reason="" if caps.has_motors else "Motor metadata not detected."),
        FieldId.CALIBRATION_PONI: _field(
            FieldId.CALIBRATION_PONI,
            StatusKind.OK if geom.calibrated else StatusKind.MISSING,
            value="loaded" if geom.calibrated else "",
            reason="" if geom.calibrated else "Load detector calibration."),
        FieldId.DETECTOR_MASK: _field(
            FieldId.DETECTOR_MASK,
            StatusKind.OK if caps.has_raw else StatusKind.DEFERRED,
            value="configured" if caps.has_raw else "",
            reason="" if caps.has_raw else "Mask is optional until raw data is loaded."),
        FieldId.BEAM_ENERGY: _field(
            FieldId.BEAM_ENERGY,
            StatusKind.OK if geom.energy_known or caps.has_energy else StatusKind.MISSING,
            value="known" if geom.energy_known or caps.has_energy else "",
            reason="" if geom.energy_known or caps.has_energy
            else "Set calibration wavelength or source energy."),
        FieldId.SAMPLE_GI: _field(
            FieldId.SAMPLE_GI,
            StatusKind.OK if geom.gi_enabled else StatusKind.DEFERRED,
            value="on" if geom.gi_enabled else "standard",
            reason="" if geom.gi_enabled else "Standard geometry."),
        FieldId.SAMPLE_ORIENTATION: _field(
            FieldId.SAMPLE_ORIENTATION,
            StatusKind.OK if geom.sample_orientation_known else (
                StatusKind.MISSING if state.mode == MeasMode.GI else StatusKind.DEFERRED),
            value="known" if geom.sample_orientation_known else "",
            reason="" if geom.sample_orientation_known else "Needed for GI."),
        FieldId.DIFFRACTOMETER_UB: _field(
            FieldId.DIFFRACTOMETER_UB,
            StatusKind.OK if geom.ub_known else (
                StatusKind.MISSING if state.tool == Tool.RSM else StatusKind.DEFERRED),
            value="known" if geom.ub_known else "",
            reason="" if geom.ub_known else "Needed for RSM."),
        FieldId.PROCESSING_MODE: _field(
            FieldId.PROCESSING_MODE,
            StatusKind.OK,
            value=state.processing_mode or state.tool.value),
        FieldId.PROCESSING_BACKEND: _field(
            FieldId.PROCESSING_BACKEND,
            StatusKind.OK if state.backend else StatusKind.DEFERRED,
            value=state.backend or "default"),
        FieldId.OUTPUT_SAVE_PATH: _field(
            FieldId.OUTPUT_SAVE_PATH,
            StatusKind.OK if state.save_path else StatusKind.MISSING,
            value=state.save_path,
            reason="" if state.save_path else "Choose a save path."),
        FieldId.RESULT_1D: _field(
            FieldId.RESULT_1D,
            StatusKind.OK if results.has_1d else StatusKind.MISSING,
            value="available" if results.has_1d else "",
            reason="" if results.has_1d else "No 1D result yet."),
        FieldId.RESULT_2D: _field(
            FieldId.RESULT_2D,
            StatusKind.OK if results.has_2d else StatusKind.MISSING,
            value="available" if results.has_2d else "",
            reason="" if results.has_2d else "No 2D result yet."),
        FieldId.RESULT_RSM: _field(
            FieldId.RESULT_RSM,
            StatusKind.OK if results.has_rsm else StatusKind.DEFERRED,
            value="available" if results.has_rsm else "",
            reason="" if results.has_rsm else "RSM viewer is gated until real-data tests pass."),
    }
    return MappingProxyType(fields)


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


def build_section_actions(state: ControlState) -> Mapping[SectionId, tuple[ControlActionSpec, ...]]:
    """Return producer/inspector actions grouped by card section.

    These actions are intentionally small wrappers around existing production
    paths while the V2 panel is hidden.  Disabled future actions still render
    with a reason so the surface is ready for real-data gates without implying
    production readiness.
    """

    viewer = state.tool in (Tool.IMAGE_VIEWER, Tool.XYE_VIEWER, Tool.NEXUS_VIEWER)
    processing_enabled = state.tool in (Tool.INT_1D, Tool.INT_2D, Tool.STITCH, Tool.RSM)
    actions = {
        SectionId.SOURCE: (
            ControlActionSpec(
                ControlAction.CHOOSE_SOURCE,
                "Choose Source",
                SectionId.SOURCE,
                enabled=True,
                reason="Uses the current legacy source browser until the Source card is live.",
            ),
        ),
        SectionId.EXPERIMENT: (
            ControlActionSpec(
                ControlAction.CALIBRATE,
                "Calibrate",
                SectionId.EXPERIMENT,
                enabled=True,
                reason="Open the existing pyFAI calibration tool.",
            ),
            ControlActionSpec(
                ControlAction.MAKE_MASK,
                "Make Mask",
                SectionId.EXPERIMENT,
                enabled=True,
                reason="Open the existing mask editor.",
            ),
            ControlActionSpec(
                ControlAction.REFINE_GEOMETRY,
                "Refine",
                SectionId.EXPERIMENT,
                enabled=False,
                reason="Geometry refinement is scaffolded; real-data GUI gate pending.",
                production_ready=False,
            ),
        ),
        SectionId.PROCESSING: (
            ControlActionSpec(
                ControlAction.ADVANCED_PROCESSING,
                "Advanced",
                SectionId.PROCESSING,
                enabled=processing_enabled and not viewer,
                reason=(
                    "Open advanced integration settings."
                    if processing_enabled and not viewer
                    else "Advanced processing is not used in viewer modes."
                ),
            ),
        ),
    }
    return MappingProxyType(actions)


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


def tool_from_mode_text(mode_text: str | None) -> Tool:
    """Map the legacy mode-combo text to the typed Controls V2 tool.

    The current GUI still owns the mode strings.  Keeping this conversion pure
    lets tests pin the bridge while the V2 panel is feature-flagged.
    """
    text = str(mode_text or "").strip().lower()
    if "image viewer" in text:
        return Tool.IMAGE_VIEWER
    if "xye viewer" in text:
        return Tool.XYE_VIEWER
    if "nexus viewer" in text or "nexus" in text and "viewer" in text:
        return Tool.NEXUS_VIEWER
    if "stitch" in text:
        return Tool.STITCH
    if "rsm" in text:
        return Tool.RSM
    if "2d" in text and "1d" not in text:
        return Tool.INT_2D
    return Tool.INT_1D


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
    fields = build_field_statuses(state)
    return ControlProfile(
        processing_page=page,
        run_enabled=not blockers and not viewer,
        run_blockers=blockers,
        fields=fields,
        section_actions=build_section_actions(state),
        analysis_launchers=build_analysis_launchers(state.result_caps),
        show_experiment_card=True,
        show_processing_card=not viewer)


# Public spelling used by the design docs.  Keep the longer name for call sites
# that prefer explicitness during the transition.
build_profile = build_control_profile
