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
import math
from types import MappingProxyType
from typing import Mapping, Sequence


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
    PROJECT = "project"
    SOURCE = "source"
    EXPERIMENT = "experiment"
    PROCESSING = "processing"
    OUTPUT = "output"
    ANALYSIS = "analysis"


class ControlAction(str, Enum):
    """Intent emitted by the Controls V2 preview.

    The Qt widget renders these as buttons, but the static-scan tab owns the
    actual routing.  Keeping the action list pure avoids giving the preview
    direct access to wrangler/integrator internals.
    """

    CHOOSE_SOURCE = "choose_source"
    CHOOSE_PROJECT = "choose_project"
    CHOOSE_OUTPUT = "choose_output"
    CALIBRATE = "calibrate"
    MAKE_MASK = "make_mask"
    REFINE_GEOMETRY = "refine_geometry"
    REINTEGRATE_1D = "reintegrate_1d"
    REINTEGRATE_2D = "reintegrate_2d"
    ADVANCED_PROCESSING = "advanced_processing"


class FieldId(str, Enum):
    PROJECT_ROOT = "project.root"
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
    FieldSpec(FieldId.PROJECT_ROOT, SectionId.PROJECT, "Project folder", "project",
              session_key="wrangler.project_folder"),
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


class ResultCap(str, Enum):
    HAS_1D = "has_1d"
    HAS_2D = "has_2d"
    HAS_RAW = "has_raw"
    RAW_REACHABLE = "raw_reachable"
    SCAN_METADATA = "scan_metadata"
    PSI_METADATA = "psi_metadata"
    PHASE_RESULT = "phase_result"


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
    calibration_energy_eV: float | None = None
    source_energy_eV: float | None = None
    correction_energy_eV: float | None = None
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
    entry_point: str = ""
    required_caps: frozenset[ResultCap] = field(default_factory=frozenset)
    optional_deps: frozenset[str] = field(default_factory=frozenset)
    singleton_key: str = ""


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


class ControlFieldKind(str, Enum):
    LINE = "line"
    BOOL = "bool"
    COMBO = "combo"


@dataclass(frozen=True, slots=True)
class ControlFormField:
    """One editable transitional Controls V2 field.

    ``path`` is still the legacy wrangler parameter path for now.  Keeping the
    typed field description here, rather than in Qt, makes the next migration
    step straightforward: the same object can point to a native ControlState
    field while the renderer stays unchanged.
    """

    section: SectionId
    label: str
    path: tuple[str, ...]
    value: object = ""
    kind: ControlFieldKind = ControlFieldKind.LINE
    choices: tuple[str, ...] = ()
    browse: bool = False
    enabled: bool = True
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ControlFormEdit:
    """One value-change intent emitted by the transitional control form."""

    path: tuple[str, ...]
    value: object


@dataclass(frozen=True, slots=True)
class LegacyWidgetBinding:
    """One transitional Controls V2 field backed by an integrator widget.

    This is intentionally just metadata: the Qt-free logic decides what should
    render, while :mod:`static_scan_widget` maps the widget names into the
    existing legacy backend.  Keeping these bindings in one table prevents the
    render/read/write/membership lists from drifting during the migration.
    """

    section: SectionId
    label: str
    path: tuple[str, ...]
    kind: ControlFieldKind
    widget_name: str
    value_role: str
    choices_widget: str = ""
    tools: frozenset[Tool] = frozenset({Tool.INT_1D, Tool.INT_2D})
    visible_when: str = ""


@dataclass(frozen=True, slots=True)
class BoundControlState:
    """Typed snapshot of the currently editable right-panel controls."""

    fields: tuple[ControlFormField, ...] = ()

    def fields_for(self, section: SectionId) -> tuple[ControlFormField, ...]:
        return tuple(field for field in self.fields if field.section == section)

    def value_for(self, path: tuple[str, ...], default: object = "") -> object:
        path = tuple(path)
        for field in self.fields:
            if field.path == path:
                return field.value
        return default


INT_1D_OUTPUT_TOOLS = frozenset({Tool.INT_1D, Tool.INT_2D})
INT_2D_OUTPUT_TOOLS = frozenset({Tool.INT_2D})


_AA_INV = "Å⁻¹"
_DEG = "°"
_CHI = "χ"


def _unit_radial_label(unit: object) -> str:
    """Mirror the legacy integrator's radial range label for a pyFAI unit."""

    text = str(unit or "").lower()
    if text == "2th_deg" or "2θ" in text or "2th" in text or "2theta" in text:
        return f"2θ ({_DEG})"
    # The 1D chi/azimuthal-profile mode still edits a Q band, matching
    # integratorTree._update_standard_1d_label.
    return f"Q ({_AA_INV})"


def _axis_text_to_gi_1d_mode(axis: object) -> str:
    """Fallback for old tests/callers that pass only the displayed GI axis text."""

    text = str(axis or "")
    lower = text.lower()
    if "exit" in lower:
        return "exit_angle"
    if "χgi" in lower or "χ_gi" in lower or "chigi" in lower or "chi_gi" in lower:
        return "chi_gi"
    if "qₒₒₚ" in lower or "q_oop" in lower or "qoop" in lower:
        return "q_oop"
    if "qᵢₚ" in lower or "q_ip" in lower or "qip" in lower:
        return "q_ip"
    return "q_total"


def _axis_text_to_gi_2d_mode(axis: object) -> str:
    """Fallback for old tests/callers that pass only the displayed GI axis text."""

    text = str(axis or "")
    lower = text.lower()
    if "exit" in lower:
        return "exit_angles"
    if (
        "qᵢₚ" in lower
        or "qₒₒₚ" in lower
        or "q_ip" in lower
        or "q_oop" in lower
        or "qip" in lower
        or "qoop" in lower
        or ("ip" in lower and "oop" in lower)
    ):
        return "qip_qoop"
    return "q_chi"


def _range_axis_labels_1d(values: Mapping[tuple[str, ...], object]) -> tuple[str, str]:
    # In GI mode the AUTHORITATIVE source is gi_mode, NOT the live legacy label:
    # the integrator HIDES the radial label in polar modes (q_total) WITHOUT
    # resetting its stale text, so trusting the widget text mislabels the range
    # box (e.g. "Qip" while integrating polar Q).  gi_mode wins here; the live
    # label is trusted only in standard mode (no gi_mode), where the integrator
    # keeps it set + shown.
    gi_mode = values.get(("Int1D", "gi_mode"))
    if gi_mode is not None:
        mode = str(gi_mode)
        if mode in {"q_ip", "q_oop"}:
            return f"Qip ({_AA_INV})", f"Qoop ({_AA_INV})"
        if mode == "exit_angle":
            return f"Qip ({_AA_INV})", f"Exit ({_DEG})"
        if mode == "chi_gi":
            return f"Q ({_AA_INV})", f"{_CHI}GI ({_DEG})"
        return (
            _unit_radial_label(values.get(("Int1D", "unit"), "q_A^-1")),
            f"{_CHI} ({_DEG})",
        )

    live_radial = values.get(("Int1D", "radial_label"))
    live_azim = values.get(("Int1D", "azim_label"))
    if live_radial and live_azim:
        return str(live_radial), str(live_azim)

    axis = values.get(("Int1D", "axis"), "")
    mode = _axis_text_to_gi_1d_mode(axis)
    axis_text = str(axis or "").lower()
    if mode != "q_total" or "gi" in axis_text:
        return _range_axis_labels_1d({
            ("Int1D", "gi_mode"): mode,
            ("Int1D", "unit"): values.get(("Int1D", "unit"), "q_A^-1"),
        })
    return _unit_radial_label(axis), f"{_CHI} ({_DEG})"


def _range_axis_labels_2d(values: Mapping[tuple[str, ...], object]) -> tuple[str, str]:
    # GI mode: gi_mode is authoritative over the (possibly stale, hidden) live
    # label — see _range_axis_labels_1d.  q_chi hides the 2D radial label without
    # resetting it, so the widget text would mislabel polar Q as "Qip".
    gi_mode = values.get(("Int2D", "gi_mode"))
    if gi_mode is not None:
        mode = str(gi_mode)
        if mode == "qip_qoop":
            return f"Qip ({_AA_INV})", f"Qoop ({_AA_INV})"
        if mode == "exit_angles":
            return f"Qip ({_AA_INV})", f"Exit ({_DEG})"
        return (
            _unit_radial_label(values.get(("Int2D", "unit"), "q_A^-1")),
            f"{_CHI} ({_DEG})",
        )

    live_radial = values.get(("Int2D", "radial_label"))
    live_azim = values.get(("Int2D", "azim_label"))
    if live_radial and live_azim:
        return str(live_radial), str(live_azim)

    axis = values.get(("Int2D", "axis"), "")
    mode = _axis_text_to_gi_2d_mode(axis)
    axis_text = str(axis or "").lower()
    if mode != "q_chi" or "gi" in axis_text:
        return _range_axis_labels_2d({
            ("Int2D", "gi_mode"): mode,
            ("Int2D", "unit"): values.get(("Int2D", "unit"), "q_A^-1"),
        })
    return _unit_radial_label(axis), f"{_CHI} ({_DEG})"


def _integration_label_overrides(
    values: Mapping[tuple[str, ...], object],
) -> dict[tuple[str, ...], str]:
    radial_1d, azim_1d = _range_axis_labels_1d(values)
    radial_2d, azim_2d = _range_axis_labels_2d(values)
    return {
        ("Int1D", "radial_auto"): f"{radial_1d} Auto",
        ("Int1D", "radial_low"): f"{radial_1d} Low",
        ("Int1D", "radial_high"): f"{radial_1d} High",
        ("Int1D", "azim_auto"): f"{azim_1d} Auto",
        ("Int1D", "azim_low"): f"{azim_1d} Low",
        ("Int1D", "azim_high"): f"{azim_1d} High",
        ("Int2D", "radial_auto"): f"{radial_2d} Auto",
        ("Int2D", "radial_low"): f"{radial_2d} Low",
        ("Int2D", "radial_high"): f"{radial_2d} High",
        ("Int2D", "azim_auto"): f"{azim_2d} Auto",
        ("Int2D", "azim_low"): f"{azim_2d} Low",
        ("Int2D", "azim_high"): f"{azim_2d} High",
    }


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "checked"}
    return bool(value)


def _field_enabled_reason(
    values: Mapping[tuple[str, ...], object],
    path: tuple[str, ...],
    *,
    controls_enabled: bool,
) -> tuple[bool, str]:
    if not controls_enabled:
        return False, "Controls are locked during the active run."

    auto_dependencies = {
        ("Int1D", "radial_low"): ("Int1D", "radial_auto"),
        ("Int1D", "radial_high"): ("Int1D", "radial_auto"),
        ("Int1D", "azim_low"): ("Int1D", "azim_auto"),
        ("Int1D", "azim_high"): ("Int1D", "azim_auto"),
        ("Int2D", "radial_low"): ("Int2D", "radial_auto"),
        ("Int2D", "radial_high"): ("Int2D", "radial_auto"),
        ("Int2D", "azim_low"): ("Int2D", "azim_auto"),
        ("Int2D", "azim_high"): ("Int2D", "azim_auto"),
    }
    auto_path = auto_dependencies.get(path)
    if auto_path is not None and _truthy(values.get(auto_path, False)):
        return False, "Disable Auto to edit this range."

    if path in {("Mask", "min"), ("Mask", "max")} and not _truthy(
        values.get(("Mask", "Threshold"), False)
    ):
        return False, "Enable Threshold to edit this limit."

    return True, ""


INTEGRATOR_BACKED_CONTROL_SPECS: tuple[LegacyWidgetBinding, ...] = (
    LegacyWidgetBinding(
        SectionId.EXPERIMENT, "Grazing", ("GI", "Grazing"),
        ControlFieldKind.BOOL, "gi_enable", "checked"),
    LegacyWidgetBinding(
        SectionId.EXPERIMENT, "Theta Motor", ("GI", "th_motor"),
        ControlFieldKind.COMBO, "gi_motor", "current_text", "gi_motor",
        visible_when="grazing"),
    LegacyWidgetBinding(
        SectionId.EXPERIMENT, "Theta", ("GI", "th_val"),
        ControlFieldKind.LINE, "gi_motor_value", "text",
        visible_when="grazing_manual"),
    LegacyWidgetBinding(
        SectionId.EXPERIMENT, "Orientation", ("GI", "sample_orientation"),
        ControlFieldKind.LINE, "gi_sample_orientation", "value",
        visible_when="grazing"),
    LegacyWidgetBinding(
        SectionId.EXPERIMENT, "Tilt Angle", ("GI", "tilt_angle"),
        ControlFieldKind.LINE, "gi_tilt", "text",
        visible_when="grazing"),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "Threshold", ("Mask", "Threshold"),
        ControlFieldKind.BOOL, "threshold_enable", "checked"),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "Min", ("Mask", "min"),
        ControlFieldKind.LINE, "threshold_min", "text"),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "Max", ("Mask", "max"),
        ControlFieldKind.LINE, "threshold_max", "text"),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "Mask Saturated", ("MaskSat", "mask_sentinel"),
        ControlFieldKind.BOOL, "mask_saturated", "checked"),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "1D Axis", ("Int1D", "axis"),
        ControlFieldKind.COMBO, "axis1D", "current_text", "axis1D",
        tools=INT_1D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "1D Points", ("Int1D", "points"),
        ControlFieldKind.LINE, "npts_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "1D OOP Points", ("Int1D", "points_oop"),
        ControlFieldKind.LINE, "npts_oop_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS, visible_when="widget_visible"),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "1D Radial Auto", ("Int1D", "radial_auto"),
        ControlFieldKind.BOOL, "radial_autoRange_1D", "checked",
        tools=INT_1D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "1D Radial Low", ("Int1D", "radial_low"),
        ControlFieldKind.LINE, "radial_low_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "1D Radial High", ("Int1D", "radial_high"),
        ControlFieldKind.LINE, "radial_high_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "1D Azim Auto", ("Int1D", "azim_auto"),
        ControlFieldKind.BOOL, "azim_autoRange_1D", "checked",
        tools=INT_1D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "1D Azim Low", ("Int1D", "azim_low"),
        ControlFieldKind.LINE, "azim_low_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "1D Azim High", ("Int1D", "azim_high"),
        ControlFieldKind.LINE, "azim_high_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "2D Axis", ("Int2D", "axis"),
        ControlFieldKind.COMBO, "axis2D", "current_text", "axis2D",
        tools=INT_2D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "2D Radial Points", ("Int2D", "radial_points"),
        ControlFieldKind.LINE, "npts_radial_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "2D Azim Points", ("Int2D", "azim_points"),
        ControlFieldKind.LINE, "npts_azim_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "2D Radial Auto", ("Int2D", "radial_auto"),
        ControlFieldKind.BOOL, "radial_autoRange_2D", "checked",
        tools=INT_2D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "2D Radial Low", ("Int2D", "radial_low"),
        ControlFieldKind.LINE, "radial_low_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "2D Radial High", ("Int2D", "radial_high"),
        ControlFieldKind.LINE, "radial_high_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "2D Azim Auto", ("Int2D", "azim_auto"),
        ControlFieldKind.BOOL, "azim_autoRange_2D", "checked",
        tools=INT_2D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "2D Azim Low", ("Int2D", "azim_low"),
        ControlFieldKind.LINE, "azim_low_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "2D Azim High", ("Int2D", "azim_high"),
        ControlFieldKind.LINE, "azim_high_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
)

INTEGRATOR_BACKED_CONTROL_PATHS: tuple[tuple[str, ...], ...] = tuple(
    spec.path for spec in INTEGRATOR_BACKED_CONTROL_SPECS
)

INTEGRATION_CONTROL_SPECS: tuple[LegacyWidgetBinding, ...] = tuple(
    spec for spec in INTEGRATOR_BACKED_CONTROL_SPECS
    if spec.path[0] in {"Int1D", "Int2D"}
)

INTEGRATION_CONTROL_PATHS: tuple[tuple[str, ...], ...] = tuple(
    spec.path for spec in INTEGRATION_CONTROL_SPECS
)

BOUND_CONTROL_PATHS: tuple[tuple[str, ...], ...] = (
    ("Project", "project_folder"),
    ("Project", "h5_dir"),
    ("Calibration", "poni_file"),
    ("NeXus File", "nexus_file"),
    ("NeXus File", "entry"),
    ("Output", "h5_dir"),
    ("Signal", "poni_file"),
    ("Signal", "inp_type"),
    ("Signal", "File"),
    ("Signal", "img_dir"),
    ("Signal", "include_subdir"),
    ("Signal", "img_ext"),
    ("Signal", "series_average"),
    ("Signal", "meta_ext"),
    ("Signal", "meta_dir"),
    ("Signal", "Filter"),
    ("Signal", "mask_file"),
    *INTEGRATOR_BACKED_CONTROL_PATHS,
    ("BG", "bg_type"),
    ("BG", "File"),
    ("BG", "Scale"),
)


def coerce_control_edit_value(current: object, incoming: object) -> object:
    """Coerce a form edit to the type of the current backing value."""

    if isinstance(current, bool):
        if isinstance(incoming, str):
            return incoming.strip().lower() in {"1", "true", "yes", "on", "checked"}
        return bool(incoming)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(incoming))
    if isinstance(current, float):
        return float(incoming)
    return incoming


def build_bound_control_state(
    values: Mapping[tuple[str, ...], object] | None = None,
    choices: Mapping[tuple[str, ...], Sequence[object]] | None = None,
    *,
    tool: Tool | None = None,
    controls_enabled: bool = True,
) -> BoundControlState:
    """Build the transitional editable Controls V2 state from legacy values.

    The values still come from wrangler parameters, but the section/visibility
    rules live in this pure function.  That keeps Qt as a renderer and gives the
    native ControlState migration a stable target.
    """

    values = {tuple(path): value for path, value in (values or {}).items()}
    choices = {
        tuple(path): tuple(str(v) for v in vals)
        for path, vals in (choices or {}).items()
    }
    label_overrides = _integration_label_overrides(values)
    fields: list[ControlFormField] = []

    def add(
        section: SectionId,
        label: str,
        path: tuple[str, ...],
        *,
        kind: ControlFieldKind = ControlFieldKind.LINE,
        browse: bool = False,
    ) -> None:
        if path not in values:
            return
        enabled, reason = _field_enabled_reason(
            values,
            path,
            controls_enabled=controls_enabled,
        )
        fields.append(ControlFormField(
            section=section,
            label=label,
            path=path,
            value=values.get(path),
            kind=kind,
            choices=choices.get(path, ()),
            browse=browse,
            enabled=enabled,
            reason=reason,
        ))

    add(SectionId.PROJECT, "Folder", ("Project", "project_folder"), browse=True)
    if ("Project", "h5_dir") in values:
        add(SectionId.PROJECT, "Save Path", ("Project", "h5_dir"), browse=True)
    else:
        add(SectionId.PROJECT, "Save Path", ("Output", "h5_dir"), browse=True)

    nexus = ("NeXus File", "nexus_file") in values
    source_type = str(values.get(("Signal", "inp_type"), ""))
    if nexus:
        add(SectionId.SOURCE, "NeXus File", ("NeXus File", "nexus_file"), browse=True)
        add(SectionId.SOURCE, "Entry", ("NeXus File", "entry"))
        add(SectionId.EXPERIMENT, "Poni", ("Calibration", "poni_file"), browse=True)
        add(SectionId.EXPERIMENT, "Mask File", ("Signal", "mask_file"), browse=True)
    else:
        add(SectionId.SOURCE, "Source", ("Signal", "inp_type"),
            kind=ControlFieldKind.COMBO)
        if source_type == "Image Directory":
            add(SectionId.SOURCE, "Directory", ("Signal", "img_dir"), browse=True)
            add(SectionId.SOURCE, "File Type", ("Signal", "img_ext"),
                kind=ControlFieldKind.COMBO)
            add(SectionId.SOURCE, "Subdirs", ("Signal", "include_subdir"),
                kind=ControlFieldKind.BOOL)
            add(SectionId.SOURCE, "Filter", ("Signal", "Filter"))
        else:
            add(SectionId.SOURCE, "Image File", ("Signal", "File"), browse=True)
        add(SectionId.SOURCE, "Meta Type", ("Signal", "meta_ext"),
            kind=ControlFieldKind.COMBO)
        if str(values.get(("Signal", "meta_ext"), "")) == "SPEC":
            add(SectionId.SOURCE, "SPEC Dir", ("Signal", "meta_dir"), browse=True)
        add(SectionId.EXPERIMENT, "Poni", ("Signal", "poni_file"), browse=True)
        add(SectionId.EXPERIMENT, "Mask File", ("Signal", "mask_file"), browse=True)

    gi_on = bool(values.get(("GI", "Grazing"), False))
    manual_theta = str(values.get(("GI", "th_motor"), "")) == "Manual"
    for spec in INTEGRATOR_BACKED_CONTROL_SPECS:
        if tool is not None and tool not in spec.tools:
            continue
        # Progressive disclosure for the GI detail fields: motor/orientation/tilt
        # appear only in Grazing mode; the manual-theta Value also needs the
        # incidence motor to be 'Manual'.  The Grazing toggle itself is always
        # shown (it carries no visible_when).
        if spec.visible_when == "grazing" and not gi_on:
            continue
        if spec.visible_when == "grazing_manual" and not (gi_on and manual_theta):
            continue
        if spec.visible_when == "manual_theta" and not manual_theta:
            continue
        add(
            spec.section,
            label_overrides.get(spec.path, spec.label),
            spec.path,
            kind=spec.kind,
        )
    # Average Scan (frame averaging) is a processing choice, not a source
    # identity, so it renders in PROCESSING as a Conditioning pill next to Mask
    # Saturated rather than in SOURCE (design doc item 6).  Placing the add()
    # here (after the Mask spec block, both bools) lets flush_pills coalesce it
    # into Mask Saturated's PillRow.  Shown for any multi-frame image source
    # (Image Series OR Image Directory — averaging applies within a series in
    # both), hidden for Single Image and NeXus.  This matches the legacy
    # wrangler's set_inp_type, which reveals series_average for everything
    # except Single Image.
    if not nexus and source_type != "Single Image":
        add(SectionId.PROCESSING, "Average Scan", ("Signal", "series_average"),
            kind=ControlFieldKind.BOOL)
    add(SectionId.PROCESSING, "Background", ("BG", "bg_type"),
        kind=ControlFieldKind.COMBO)
    if str(values.get(("BG", "bg_type"), "None")) != "None":
        add(SectionId.PROCESSING, "BG File", ("BG", "File"), browse=True)
        add(SectionId.PROCESSING, "Scale", ("BG", "Scale"))

    return BoundControlState(fields=tuple(fields))


@dataclass(frozen=True, slots=True)
class ControlState:
    tool: Tool = Tool.INT_1D
    mode: MeasMode = MeasMode.STANDARD
    source_caps: SourceCaps = field(default_factory=SourceCaps)
    result_caps: ResultCaps = field(default_factory=ResultCaps)
    geom: GeomState = field(default_factory=GeomState)
    backend: str | None = None
    project_root: str = ""
    source_label: str = ""
    save_path: str = ""
    frame_count: int = 0
    processing_mode: str = ""
    real_data_gates: frozenset[str] = field(default_factory=frozenset)
    controls_locked: bool = False


@dataclass(frozen=True, slots=True)
class ControlProfile:
    processing_page: ProcessingPage
    run_enabled: bool
    run_blockers: tuple[str, ...] = ()
    valid_modes: frozenset[MeasMode] = field(
        default_factory=lambda: frozenset(MeasMode)
    )
    backend_required: str | None = None
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

    @property
    def can_run(self) -> bool:
        """Design-doc spelling for ``run_enabled``."""
        return self.run_enabled


@dataclass(frozen=True, slots=True)
class ControlPanelRenderState:
    """Single typed render input for the Controls Panel V2 widget.

    ``profile`` is the design-level status/gating description.  ``bound_controls``
    is the transitional editable form snapshot that still points at legacy
    wrangler parameters.  Keeping them in one immutable object gives Qt one
    state to render now, and lets the native control-state migration replace
    the bound legacy rows without changing the panel renderer again.
    """

    profile: ControlProfile
    bound_controls: BoundControlState | None = None


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


ENERGY_CONFLICT_RTOL = 1.0e-3


def _finite_positive_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


def _energy_conflict(
    energy_a_eV: object,
    energy_b_eV: object,
    *,
    what_a: str,
    what_b: str,
    rtol: float = ENERGY_CONFLICT_RTOL,
) -> str:
    """Return a user-facing conflict reason when two energy sources disagree."""

    a = _finite_positive_float(energy_a_eV)
    b = _finite_positive_float(energy_b_eV)
    if a is None or b is None:
        return ""
    if abs(a - b) <= rtol * max(abs(a), abs(b), 1.0):
        return ""
    return (
        f"{what_a} energy ({a:.1f} eV) disagrees with "
        f"{what_b} energy ({b:.1f} eV)."
    )


def _energy_value_text(geom: GeomState, caps: SourceCaps) -> str:
    pieces = []
    calibration = _finite_positive_float(geom.calibration_energy_eV)
    source = _finite_positive_float(geom.source_energy_eV)
    correction = _finite_positive_float(geom.correction_energy_eV)
    if calibration is not None:
        pieces.append(f"cal {calibration:.1f} eV")
    if source is not None:
        pieces.append(f"source {source:.1f} eV")
    if correction is not None:
        pieces.append(f"correction {correction:.1f} eV")
    if pieces:
        return "; ".join(pieces)
    return "known" if geom.energy_known or caps.has_energy else ""


def _energy_status_reason(geom: GeomState, caps: SourceCaps) -> tuple[StatusKind, str, str]:
    calibration = _finite_positive_float(geom.calibration_energy_eV)
    source = _finite_positive_float(geom.source_energy_eV)
    correction = _finite_positive_float(geom.correction_energy_eV)
    conflict = (
        _energy_conflict(
            calibration, source,
            what_a="Calibration", what_b="source")
        or _energy_conflict(
            calibration, correction,
            what_a="Calibration", what_b="correction")
    )
    value = _energy_value_text(geom, caps)
    if conflict:
        return StatusKind.CONFLICT, value, conflict
    if geom.energy_known or caps.has_energy or any(
        v is not None for v in (calibration, source, correction)
    ):
        return StatusKind.OK, value, ""
    return (
        StatusKind.MISSING,
        "",
        "Set calibration wavelength or source energy.",
    )


def _processing_backend_status_reason(
    state: ControlState,
) -> tuple[StatusKind, str]:
    """Return the status for the selected processing backend.

    F-1 keeps backend constraints in the same field-status stream as missing
    inputs.  For now this is intentionally narrow: GI stitching needs the
    histogram path because the MultiGeometry path has no GI correction support.
    The broader GI correction stack for Int modes is a later feature, not part
    of the Panel V2 carrier migration.
    """

    required = backend_required_for(state)
    if not state.backend:
        return StatusKind.DEFERRED, ""
    if required and state.backend != required:
        return (
            StatusKind.CONFLICT,
            f"{state.mode.value} {state.tool.value} requires backend {required}.",
        )
    return StatusKind.OK, ""


def build_field_statuses(state: ControlState) -> Mapping[FieldId, FieldStatus]:
    caps = state.source_caps
    geom = state.geom
    results = state.result_caps
    source_label = state.source_label or "No source selected"
    frame_value = str(state.frame_count) if state.frame_count else ""
    energy_status, energy_value, energy_reason = _energy_status_reason(geom, caps)
    backend_status, backend_reason = _processing_backend_status_reason(state)
    fields = {
        FieldId.PROJECT_ROOT: _field(
            FieldId.PROJECT_ROOT,
            StatusKind.OK if state.project_root else StatusKind.DEFERRED,
            value=state.project_root,
            reason="" if state.project_root else "Optional; improves portable paths."),
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
            energy_status,
            value=energy_value,
            reason=energy_reason),
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
            backend_status,
            value=state.backend or "default",
            reason=backend_reason),
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
              production_ready: bool = True,
              entry_point: str = "",
              required_caps: frozenset[ResultCap] | None = None,
              optional_deps: frozenset[str] | None = None,
              singleton_key: str = "") -> AnalysisLauncherSpec:
    return AnalysisLauncherSpec(
        tool=tool, label=label, enabled=False, reason=reason,
        live_capable=live, batch_capable=batch,
        production_ready=production_ready,
        entry_point=entry_point,
        required_caps=required_caps or frozenset(),
        optional_deps=optional_deps or frozenset(),
        singleton_key=singleton_key or tool.value)


def _enabled(tool: AnalysisTool, label: str, *,
             reason: str = "", live: bool = False, batch: bool = False,
             production_ready: bool = True,
             entry_point: str = "",
             required_caps: frozenset[ResultCap] | None = None,
             optional_deps: frozenset[str] | None = None,
             singleton_key: str = "") -> AnalysisLauncherSpec:
    return AnalysisLauncherSpec(
        tool=tool, label=label, enabled=True, reason=reason,
        live_capable=live, batch_capable=batch,
        production_ready=production_ready,
        entry_point=entry_point,
        required_caps=required_caps or frozenset(),
        optional_deps=optional_deps or frozenset(),
        singleton_key=singleton_key or tool.value)


def build_analysis_launchers(caps: ResultCaps) -> tuple[AnalysisLauncherSpec, ...]:
    """Return launcher state for auxiliary popup tools.

    Peak/Phase dialogs remain openable before data exists so they can attach to
    a live run.  Future tools with stricter data requirements stay disabled
    until the required result/metadata is present.
    """

    peak_caps = frozenset({ResultCap.HAS_1D})
    peak_entry = "xdart.gui.tabs.static_scan.peak_fit_dialog:PeakFitDialog"
    peak = (_enabled(AnalysisTool.PEAK_FIT, "Peak Fitting",
                     reason="Waiting for a 1D pattern." if not caps.has_1d else "",
                     live=True, batch=True,
                     entry_point=peak_entry,
                     required_caps=peak_caps,
                     optional_deps=frozenset({"fitting"}))
            if caps.has_optional_dep("fitting")
            else _disabled(AnalysisTool.PEAK_FIT, "Peak Fitting",
                           "Install fitting dependencies.", live=True, batch=True,
                           entry_point=peak_entry,
                           required_caps=peak_caps,
                           optional_deps=frozenset({"fitting"})))

    phase_caps = frozenset({ResultCap.HAS_1D})
    phase_entry = "xdart.gui.tabs.static_scan.phase_fit_dialog:PhaseFitDialog"
    phase = (_enabled(AnalysisTool.PHASE_FIT, "Phase Fitting",
                      reason="Waiting for a 1D pattern." if not caps.has_1d else "",
                      batch=True,
                      entry_point=phase_entry,
                      required_caps=phase_caps,
                      optional_deps=frozenset({"fitting"}))
             if caps.has_optional_dep("fitting")
             else _disabled(AnalysisTool.PHASE_FIT, "Phase Fitting",
                            "Install fitting dependencies.", batch=True,
                            entry_point=phase_entry,
                            required_caps=phase_caps,
                            optional_deps=frozenset({"fitting"})))

    scan_plot = _enabled(
        AnalysisTool.SCAN_PLOT, "Plot Metadata",
        reason="" if caps.has_scan_metadata else "Choose a scan/source in the dialog.",
        batch=True,
        entry_point="xdart.gui.tabs.static_scan.scan_plot_dialog:ScanPlotDialog",
        required_caps=frozenset({ResultCap.SCAN_METADATA}),
        singleton_key="scan_plot")

    roi = (_enabled(AnalysisTool.ROI_STATS, "ROI Statistics", batch=True)
           if caps.raw_reachable
           else _disabled(AnalysisTool.ROI_STATS, "ROI Statistics",
                          "Raw frames are not reachable for ROI reduction.",
                          batch=True))
    roi = AnalysisLauncherSpec(
        tool=roi.tool,
        label=roi.label,
        enabled=roi.enabled,
        reason=roi.reason,
        live_capable=roi.live_capable,
        batch_capable=roi.batch_capable,
        production_ready=roi.production_ready,
        entry_point="xdart.gui.tabs.static_scan.scan_plot_dialog:ScanPlotDialog",
        required_caps=frozenset({ResultCap.RAW_REACHABLE}),
        singleton_key="roi_stats",
    )

    strain_ready = caps.has_1d and caps.has_psi_metadata
    strain = (_enabled(AnalysisTool.SIN2PSI, "Strain / sin²ψ",
                       live=False, batch=True, production_ready=False,
                       entry_point="xdart.gui.tabs.static_scan.strain_dialog:StrainDialog",
                       required_caps=frozenset({
                           ResultCap.HAS_1D,
                           ResultCap.PSI_METADATA,
                       }),
                       singleton_key="sin2psi")
              if strain_ready
              else _disabled(AnalysisTool.SIN2PSI, "Strain / sin²ψ",
                             "Needs 1D patterns with ψ metadata.",
                             batch=True, production_ready=False,
                             entry_point="xdart.gui.tabs.static_scan.strain_dialog:StrainDialog",
                             required_caps=frozenset({
                                 ResultCap.HAS_1D,
                                 ResultCap.PSI_METADATA,
                             }),
                             singleton_key="sin2psi"))

    texture_ready = caps.has_1d or caps.has_phase_result
    texture = (_enabled(AnalysisTool.TEXTURE, "Texture / Orientation",
                        batch=True, production_ready=False,
                        entry_point="xdart.gui.tabs.static_scan.texture_dialog:TextureDialog",
                        required_caps=frozenset({ResultCap.PHASE_RESULT}),
                        singleton_key="texture")
               if texture_ready
               else _disabled(AnalysisTool.TEXTURE, "Texture / Orientation",
                              "Needs fitted phases or suitable 1D patterns.",
                              batch=True, production_ready=False,
                              entry_point="xdart.gui.tabs.static_scan.texture_dialog:TextureDialog",
                              required_caps=frozenset({ResultCap.PHASE_RESULT}),
                              singleton_key="texture"))
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
    unlocked = not state.controls_locked
    processing_actions = [
        ControlActionSpec(
            ControlAction.ADVANCED_PROCESSING,
            "Advanced",
            SectionId.PROCESSING,
            enabled=processing_enabled and not viewer and unlocked,
            reason=(
                "Open advanced integration settings."
                if processing_enabled and not viewer and unlocked
                else "Controls are locked during the active run."
                if state.controls_locked
                else "Advanced processing is not used in viewer modes."
            ),
        ),
    ]
    if processing_enabled and not viewer:
        processing_actions = [
            ControlActionSpec(
                ControlAction.REINTEGRATE_1D,
                "Reintegrate 1D",
                SectionId.PROCESSING,
                enabled=unlocked,
                reason=(
                    "Reintegrate the selected 1D output."
                    if unlocked
                    else "Controls are locked during the active run."
                ),
            ),
            ControlActionSpec(
                ControlAction.REINTEGRATE_2D,
                "Reintegrate 2D",
                SectionId.PROCESSING,
                enabled=unlocked,
                reason=(
                    "Reintegrate the selected 2D output."
                    if unlocked
                    else "Controls are locked during the active run."
                ),
            ),
            *processing_actions,
        ]
    experiment_actions = [
        ControlActionSpec(
            ControlAction.CALIBRATE,
            "Calibrate",
            SectionId.EXPERIMENT,
            enabled=unlocked,
            reason=(
                "Open the existing pyFAI calibration tool."
                if unlocked
                else "Controls are locked during the active run."
            ),
        ),
        ControlActionSpec(
            ControlAction.MAKE_MASK,
            "Make Mask",
            SectionId.EXPERIMENT,
            enabled=unlocked,
            reason=(
                "Open the existing mask editor."
                if unlocked
                else "Controls are locked during the active run."
            ),
        ),
    ]
    # Refine is geometry-refinement scaffolding; it isn't relevant to the plain
    # 1-D/2-D integration modes, so hide it there.  (Kept for Stitch/RSM/GI,
    # where geometry refinement will matter, once its real-data gate lands.)
    if state.tool not in (Tool.INT_1D, Tool.INT_2D):
        experiment_actions.append(
            ControlActionSpec(
                ControlAction.REFINE_GEOMETRY,
                "Refine",
                SectionId.EXPERIMENT,
                enabled=False,
                reason="Geometry refinement is scaffolded; real-data GUI gate pending.",
                production_ready=False,
            )
        )
    actions = {
        SectionId.SOURCE: (
            ControlActionSpec(
                ControlAction.CHOOSE_SOURCE,
                "Choose Source",
                SectionId.SOURCE,
                enabled=unlocked,
                reason=(
                    "Uses the current legacy source browser until the Source card is live."
                    if unlocked
                    else "Controls are locked during the active run."
                ),
            ),
        ),
        SectionId.PROJECT: (
            ControlActionSpec(
                ControlAction.CHOOSE_PROJECT,
                "Choose Project",
                SectionId.PROJECT,
                enabled=unlocked,
                reason=(
                    "Set the project folder used for portable source paths."
                    if unlocked
                    else "Controls are locked during the active run."
                ),
            ),
            ControlActionSpec(
                ControlAction.CHOOSE_OUTPUT,
                "Save Folder",
                SectionId.PROJECT,
                enabled=unlocked,
                reason=(
                    "Choose where processed NeXus files are written."
                    if unlocked
                    else "Controls are locked during the active run."
                ),
            ),
        ),
        SectionId.EXPERIMENT: tuple(experiment_actions),
        SectionId.PROCESSING: tuple(processing_actions),
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
    return run_blockers_from_fields(state, build_field_statuses(state))


_RUN_BLOCKER_TEXT: Mapping[FieldId, str] = MappingProxyType({
    FieldId.SOURCE_FRAMES: "Choose a frame source.",
    FieldId.CALIBRATION_PONI: "Load detector calibration.",
    FieldId.BEAM_ENERGY: "Set calibration wavelength or source energy.",
    FieldId.SAMPLE_ORIENTATION: "Set GI sample orientation.",
    FieldId.SOURCE_MOTORS: "RSM needs motor metadata.",
    FieldId.DIFFRACTOMETER_UB: "Set/refine UB matrix before RSM.",
})


def run_required_fields_for(state: ControlState) -> tuple[FieldId, ...]:
    required: list[FieldId] = []
    if state.tool in (Tool.INT_1D, Tool.INT_2D, Tool.STITCH, Tool.RSM):
        required.extend((
            FieldId.SOURCE_FRAMES,
            FieldId.CALIBRATION_PONI,
            FieldId.BEAM_ENERGY,
        ))
    if state.mode == MeasMode.GI:
        required.append(FieldId.SAMPLE_ORIENTATION)
    if state.tool == Tool.RSM:
        required.extend((FieldId.SOURCE_MOTORS, FieldId.DIFFRACTOMETER_UB))
    if backend_required_for(state):
        required.append(FieldId.PROCESSING_BACKEND)
    return tuple(dict.fromkeys(required))


def run_blockers_from_fields(
    state: ControlState,
    fields: Mapping[FieldId, FieldStatus],
) -> tuple[str, ...]:
    blockers: list[str] = []
    for field_id in run_required_fields_for(state):
        field = fields.get(field_id)
        if field is None:
            continue
        if field.status == StatusKind.MISSING:
            blockers.append(_RUN_BLOCKER_TEXT.get(field_id) or field.reason)
        elif field.status == StatusKind.CONFLICT:
            blockers.append(field.reason or _RUN_BLOCKER_TEXT.get(field_id, "Resolve conflict."))
    if state.tool == Tool.STITCH:
        if state.mode == MeasMode.GI and "gi_stitch_real_data" not in state.real_data_gates:
            blockers.append("GI stitching awaits real-data gate.")
        if state.backend == "xu_hist" and "xu_hist_real_data" not in state.real_data_gates:
            blockers.append("xu_hist stitching awaits real-data gate.")
    if state.tool == Tool.RSM:
        if "rsm_real_data" not in state.real_data_gates:
            blockers.append("RSM GUI awaits real-data gate.")
    return tuple(dict.fromkeys(blockers))


def valid_modes_for(tool: Tool) -> frozenset[MeasMode]:
    if tool == Tool.RSM:
        return frozenset()
    if tool in (Tool.IMAGE_VIEWER, Tool.XYE_VIEWER, Tool.NEXUS_VIEWER):
        return frozenset()
    return frozenset(MeasMode)


def backend_required_for(state: ControlState) -> str | None:
    if state.tool == Tool.STITCH and state.mode == MeasMode.GI:
        return "pyfai_hist"
    return None


def build_control_profile(state: ControlState) -> ControlProfile:
    page = processing_page_for(state.tool, state.mode)
    viewer = page == ProcessingPage.VIEWER
    fields = build_field_statuses(state)
    blockers = run_blockers_from_fields(state, fields)
    return ControlProfile(
        processing_page=page,
        run_enabled=not blockers and not viewer,
        run_blockers=blockers,
        valid_modes=valid_modes_for(state.tool),
        backend_required=backend_required_for(state),
        fields=fields,
        section_actions=build_section_actions(state),
        analysis_launchers=build_analysis_launchers(state.result_caps),
        show_experiment_card=True,
        show_processing_card=not viewer)


# Public spelling used by the design docs.  Keep the longer name for call sites
# that prefer explicitness during the transition.
build_profile = build_control_profile


def build_control_panel_state(
    state: ControlState,
    values: Mapping[tuple[str, ...], object] | None = None,
    choices: Mapping[tuple[str, ...], Sequence[object]] | None = None,
    *,
    bound: bool = True,
) -> ControlPanelRenderState:
    """Build the complete typed render state for Controls Panel V2."""

    return ControlPanelRenderState(
        profile=build_control_profile(state),
        bound_controls=(
            build_bound_control_state(
                values,
                choices,
                tool=state.tool,
                controls_enabled=not state.controls_locked,
            )
            if bound
            else None
        ),
    )
