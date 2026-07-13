# -*- coding: utf-8 -*-
"""Pure control-profile logic for the static-scan GUI.

This module is intentionally Qt-free.  It is the small decision layer behind
the forthcoming Controls Panel V2: source/result capabilities in, profile and
analysis launchers out.  Keeping this logic separate lets the GUI expose
experimental Stitch/RSM/GI controls as ready-to-wire scaffolding while keeping
run buttons gated until their real-data acceptance checks land.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


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


class RunTarget(str, Enum):
    SOURCE = "source"
    LOADED_SCAN = "loaded_scan"
    NONE = "none"


LOADED_SCAN_RUN_MESSAGE = (
    "Run needs a frame source - use Reintegrate for the loaded scan."
)


_UNSET = object()


_PROCESSING_COMPARED_FIELDS: tuple[tuple[str, str], ...] = (
    ("mode", "mode"),
    ("axis_1d", "1D axis"),
    ("axis_2d", "2D axis"),
    ("unit_1d", "1D unit"),
    ("unit_2d", "2D unit"),
    ("npt_1d", "1D points"),
    ("npt_oop_1d", "1D oop points"),
    ("npt_rad_2d", "2D radial points"),
    ("npt_azim_2d", "2D azimuth points"),
    ("radial_range_1d", "1D radial range"),
    ("azimuth_range_1d", "1D azimuth range"),
    ("radial_range_2d", "2D radial range"),
    ("azimuth_range_2d", "2D azimuth range"),
    # S-3: value-affecting, GRID-PRESERVING params.  These change the written
    # numbers while leaving the axis/npt/range grid identical, so they pass both
    # the modal and the axis backstop -> mixed provenance under a /entry/reduction
    # that claims the first run's config.  Compared BACKWARD-TOLERANTLY: a field
    # absent from a pre-upgrade stored config is _UNSET and skipped (no
    # false-positive modal on every existing .nxs).
    ("chi_offset_1d", "1D chi offset"),
    ("chi_offset_2d", "2D chi offset"),
    ("monitor_1d", "1D monitor"),
    ("monitor_2d", "2D monitor"),
    ("polarization_1d", "1D polarization"),
    ("polarization_2d", "2D polarization"),
    ("error_model_1d", "1D error model"),
    ("error_model_2d", "2D error model"),
    ("gi_incidence", "GI incidence angle"),
)


@dataclass(frozen=True, slots=True)
class ProcessingConfigSignature:
    """Data-affecting integration config used by Append run gates."""

    mode: MeasMode
    axis_1d: str
    axis_2d: str
    unit_1d: str
    unit_2d: str
    npt_1d: int | None
    npt_oop_1d: int | None
    npt_rad_2d: int | None
    npt_azim_2d: int | None
    radial_range_1d: object
    azimuth_range_1d: object
    radial_range_2d: object
    azimuth_range_2d: object
    # S-3 value-affecting/grid-preserving.  Default _UNSET == "not present in this
    # config"; the comparison skips any field that is _UNSET on either side, so a
    # pre-upgrade stored config (missing these keys) never triggers a modal.
    chi_offset_1d: object = _UNSET
    chi_offset_2d: object = _UNSET
    monitor_1d: object = _UNSET
    monitor_2d: object = _UNSET
    polarization_1d: object = _UNSET
    polarization_2d: object = _UNSET
    error_model_1d: object = _UNSET
    error_model_2d: object = _UNSET
    gi_incidence: object = _UNSET

    @property
    def display_mode(self) -> str:
        return "Grazing" if self.mode == MeasMode.GI else "Standard"

    def compared_items(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (field_name, getattr(self, attr))
            for attr, field_name in _PROCESSING_COMPARED_FIELDS
        )


@dataclass(frozen=True, slots=True)
class AppendConfigCheck:
    ok: bool
    reason: str = ""
    compared_fields: tuple[str, ...] = ()
    mismatched_fields: tuple[str, ...] = ()
    processed_label: str = ""
    current_label: str = ""


class AppendConfigMismatchError(RuntimeError):
    """An Append target's stored config differs from the current run config.

    Raised by run paths (e.g. the xdart image wrangler's ``initialize_scan``)
    when :func:`append_config_mismatch_check` fails after the Run-click modal
    can no longer intervene — a mid-run settings change, or a later
    auto-discovered scan of a directory run.  Subclasses ``RuntimeError`` so
    pre-existing broad handlers keep working, while run loops can catch THIS
    mismatch specifically and stop cleanly instead of crashing the worker
    thread.  Carries the :class:`AppendConfigCheck` (``reason``,
    ``mismatched_fields``, ``processed_label``/``current_label``) so the GUI
    can name what changed without re-deriving the comparison.  The append
    target itself is untouched — the guard preserved it.
    """

    def __init__(self, message: str, check: AppendConfigCheck):
        super().__init__(message)
        self.check = check


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "none", "null")
    return False


def _first_value(mapping: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in mapping and not _is_empty_value(mapping[key]):
            return mapping[key]
    return default


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int | None:
    if _is_empty_value(value):
        return default
    return _int_or_none(value)


def _bool_or_none(value: Any) -> bool | None:
    if _is_empty_value(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on", "gi", "grazing"):
        return True
    if text in ("0", "false", "f", "no", "n", "off", "standard"):
        return False
    return None


def _text_or_default(value: Any, default: str) -> str:
    if _is_empty_value(value):
        return default
    return str(value).strip()


def _range_or_none(value: Any) -> object:
    if _is_empty_value(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
    try:
        if len(value) != 2:  # type: ignore[arg-type]
            return repr(value)
        lo, hi = value  # type: ignore[misc]
    except TypeError:
        return repr(value)
    if lo is None or hi is None:
        return None
    try:
        return (round(float(lo), 12), round(float(hi), 12))
    except (TypeError, ValueError):
        return (str(lo), str(hi))


def _num_or_unset(mapping: Mapping[str, Any], keys: Sequence[str]) -> object:
    """S-3: a numeric config value, or ``_UNSET`` when NONE of the keys is
    present (so a pre-upgrade config that never stored the field is skipped)."""
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if value is None or value == "":
                return None
            try:
                return round(float(value), 12)
            except (TypeError, ValueError):
                return str(value)
    return _UNSET


def _str_or_unset(mapping: Mapping[str, Any], keys: Sequence[str]) -> object:
    """S-3: a string-valued config field, or ``_UNSET`` when absent."""
    for key in keys:
        if key in mapping:
            value = mapping[key]
            return None if value is None else str(value).strip().lower()
    return _UNSET


def _standard_axis_from_unit(unit: object, *, dim: str) -> str:
    text = str(unit or "").lower()
    if "chi" in text and dim == "1d":
        return "chi"
    if "2th" in text or "2θ" in text:
        return "2theta" if dim == "1d" else "2theta-chi"
    return "q" if dim == "1d" else "q-chi"


def _config_indicates_gi(
    bai_1d_args: Mapping[str, Any],
    bai_2d_args: Mapping[str, Any],
    gi_config: Mapping[str, Any],
) -> bool:
    return (
        bool(gi_config)
        or "gi_mode_1d" in bai_1d_args
        or "gi_mode_2d" in bai_2d_args
        or "npt_oop" in bai_1d_args
        or "npt_oop" in bai_2d_args
    )


def processing_config_from_args(
    bai_1d_args: Mapping[str, Any] | None,
    bai_2d_args: Mapping[str, Any] | None,
    *,
    gi_enabled: bool | None = None,
    gi_config: Mapping[str, Any] | None = None,
) -> ProcessingConfigSignature:
    """Return the data-shape/axis signature for an integration setup."""

    a1 = _mapping(bai_1d_args)
    a2 = _mapping(bai_2d_args)
    gic = _mapping(gi_config)
    is_gi = (
        _config_indicates_gi(a1, a2, gic)
        if gi_enabled is None
        else bool(gi_enabled)
    )
    mode = MeasMode.GI if is_gi else MeasMode.STANDARD
    unit_1d = _text_or_default(a1.get("unit"), "q_A^-1")
    unit_2d = _text_or_default(a2.get("unit"), "q_A^-1")
    if is_gi:
        axis_1d = _text_or_default(
            a1.get("gi_mode_1d")
            or gic.get("gi_mode_1d")
            or None,
            "q_total",
        )
        axis_2d = _text_or_default(
            a2.get("gi_mode_2d")
            or gic.get("gi_mode_2d")
            or None,
            "qip_qoop",
        )
    else:
        axis_1d = _standard_axis_from_unit(unit_1d, dim="1d")
        axis_2d = _standard_axis_from_unit(unit_2d, dim="2d")
    return ProcessingConfigSignature(
        mode=mode,
        axis_1d=axis_1d,
        axis_2d=axis_2d,
        unit_1d=unit_1d,
        unit_2d=unit_2d,
        npt_1d=_int_or_none(
            _first_value(a1, ("npt", "numpoints", "npt_rad"), 3000)
        ),
        npt_oop_1d=_int_or_none(a1.get("npt_oop")),
        npt_rad_2d=_int_or_none(_first_value(a2, ("npt_rad", "npt"), 500)),
        npt_azim_2d=_int_or_default(a2.get("npt_azim"), 500),
        radial_range_1d=_range_or_none(a1.get("radial_range")),
        azimuth_range_1d=_range_or_none(a1.get("azimuth_range")),
        radial_range_2d=_range_or_none(a2.get("radial_range")),
        azimuth_range_2d=_range_or_none(a2.get("azimuth_range")),
        # chi_offset is INERT for GI (S-4 zeroes the GI azimuth_offset; GI chi
        # goes to FiberIntegrator's own convention), so comparing it on a GI scan
        # trips a FALSE Append modal for a change that does not alter written GI
        # data.  Leave it unknown for GI; standard mode still compares it (it does
        # move the written chi axis).  [review follow-up]
        chi_offset_1d=(_UNSET if is_gi else _num_or_unset(a1, ("chi_offset",))),
        chi_offset_2d=(_UNSET if is_gi
                       else _num_or_unset(a2, ("azimuth_offset", "chi_offset"))),
        monitor_1d=_str_or_unset(a1, ("monitor",)),
        monitor_2d=_str_or_unset(a2, ("monitor",)),
        polarization_1d=_num_or_unset(a1, ("polarization_factor", "polarization")),
        polarization_2d=_num_or_unset(a2, ("polarization_factor", "polarization")),
        error_model_1d=_str_or_unset(a1, ("error_model",)),
        error_model_2d=_str_or_unset(a2, ("error_model",)),
        gi_incidence=_num_or_unset(
            gic, ("th_val", "incidence", "incident_angle", "incidence_angle")),
    )


def processing_config_from_mapping(
    value: Mapping[str, Any] | ProcessingConfigSignature | None,
) -> ProcessingConfigSignature | None:
    if value is None:
        return None
    if isinstance(value, ProcessingConfigSignature):
        return value
    config = _mapping(value)
    if "config" in config and "bai_1d_args" not in config:
        config = _mapping(config.get("config"))
    gi_marker = config.get("gi", _UNSET)
    gi_enabled = None if gi_marker is _UNSET else _bool_or_none(gi_marker)
    return processing_config_from_args(
        _mapping(config.get("bai_1d_args")),
        _mapping(config.get("bai_2d_args")),
        gi_enabled=gi_enabled,
        gi_config=_mapping(config.get("gi_config")),
    )


def processing_config_from_scan(
    scan: Any,
    *,
    prefer_stored: bool = False,
) -> ProcessingConfigSignature | None:
    """Build an Append/readiness signature from a scan-like object."""

    if scan is None:
        return None
    if prefer_stored:
        for attr in ("reduction_config", "_display_reduction_config"):
            stored = getattr(scan, attr, None)
            if isinstance(stored, Mapping) and stored:
                return processing_config_from_mapping(stored)
        return None
    return processing_config_from_args(
        _mapping(getattr(scan, "bai_1d_args", {}) or {}),
        _mapping(getattr(scan, "bai_2d_args", {}) or {}),
        gi_enabled=bool(getattr(scan, "gi", False)),
        gi_config=_mapping(getattr(scan, "gi_config", {}) or {}),
    )


def append_config_mismatch_check(
    write_mode: object,
    processed_config: Mapping[str, Any] | ProcessingConfigSignature | None,
    current_config: Mapping[str, Any] | ProcessingConfigSignature | None,
) -> AppendConfigCheck:
    """Return the Append config comparison result when configs differ."""

    compared_fields = tuple(label for _attr, label in _PROCESSING_COMPARED_FIELDS)
    if str(write_mode or "").strip().lower() != "append":
        return AppendConfigCheck(ok=True, compared_fields=compared_fields)
    processed = processing_config_from_mapping(processed_config)
    current = processing_config_from_mapping(current_config)
    if processed is None or current is None:
        return AppendConfigCheck(ok=True, compared_fields=compared_fields)

    # S-3 backward-tolerant: skip any field that is _UNSET on EITHER side (a
    # pre-upgrade stored config that never recorded it, or a run that doesn't set
    # it) -- comparing it would raise a false modal on every existing .nxs.
    mismatches = tuple(
        label for attr, label in _PROCESSING_COMPARED_FIELDS
        if getattr(processed, attr) is not _UNSET
        and getattr(current, attr) is not _UNSET
        and getattr(processed, attr) != getattr(current, attr)
    )
    if not mismatches:
        return AppendConfigCheck(
            ok=True,
            compared_fields=compared_fields,
            processed_label=processed.display_mode,
            current_label=current.display_mode,
        )

    logger.debug(
        "append config mismatch: processed=%s current=%s checked=%s "
        "differences=%s",
        processed.compared_items(),
        current.compared_items(),
        compared_fields,
        mismatches,
    )
    # Name WHAT changed: display_mode only carries the GI/Standard axis, so a
    # same-mode mismatch (e.g. a mid-run Int 1D -> Int 2D settings change) used
    # to read "processed: Standard · current: Standard" — true but useless.
    reason = (
        f"processed: {processed.display_mode} · current: {current.display_mode} "
        f"(differs: {', '.join(mismatches)}) "
        "— switch write mode to Replace, or revert settings"
    )
    return AppendConfigCheck(
        ok=False,
        reason=reason,
        compared_fields=compared_fields,
        mismatched_fields=mismatches,
        processed_label=processed.display_mode,
        current_label=current.display_mode,
    )


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
    parameter_group: str = ""


@dataclass(frozen=True, slots=True)
class ControlFormEdit:
    """One value-change intent emitted by the transitional control form."""

    path: tuple[str, ...]
    value: object


@dataclass(frozen=True, slots=True)
class LegacyWidgetBinding:
    """One transitional Controls V2 field backed by the Int carrier.

    This is intentionally just metadata: the Qt-free logic decides what should
    render, while :mod:`static_scan_widget` maps the widget or ParameterTree
    names into the existing legacy backend.  Keeping these bindings in one
    table prevents the render/read/write/membership lists from drifting during
    the migration.
    """

    section: SectionId
    label: str
    path: tuple[str, ...]
    kind: ControlFieldKind
    widget_name: str = ""
    value_role: str = "text"
    choices_widget: str = ""
    parameter_group: str = ""
    parameter_name: str = ""
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

    if (
        path in {("Int1D", "polarization_factor"),
                 ("Int2D", "polarization_factor")}
        and not _truthy(values.get((path[0], "apply_polarization"), False))
    ):
        return False, "Enable Polarization to edit this factor."

    return True, ""


def _int_advanced_specs(
    root: str,
    label_prefix: str,
    parameter_group: str,
    tools: frozenset[Tool],
) -> tuple[LegacyWidgetBinding, ...]:
    base = dict(
        section=SectionId.PROCESSING,
        parameter_group=parameter_group,
        tools=tools,
    )
    return (
        LegacyWidgetBinding(
            **base, label=f"{label_prefix} Solid Angle",
            path=(root, "correctSolidAngle"), kind=ControlFieldKind.BOOL,
            value_role="checked", parameter_name="correctSolidAngle"),
        LegacyWidgetBinding(
            **base, label=f"{label_prefix} Polarization",
            path=(root, "apply_polarization"), kind=ControlFieldKind.BOOL,
            value_role="checked", parameter_name="Apply polarization factor"),
        LegacyWidgetBinding(
            **base, label=f"{label_prefix} Pol. Factor",
            path=(root, "polarization_factor"), kind=ControlFieldKind.LINE,
            value_role="float", parameter_name="polarization_factor"),
        LegacyWidgetBinding(
            **base, label=f"{label_prefix} Method",
            path=(root, "method"), kind=ControlFieldKind.COMBO,
            value_role="current_text", parameter_name="method"),
        LegacyWidgetBinding(
            **base, label=f"{label_prefix} Dummy",
            path=(root, "dummy"), kind=ControlFieldKind.LINE,
            value_role="float", parameter_name="dummy"),
        LegacyWidgetBinding(
            **base, label=f"{label_prefix} Delta Dummy",
            path=(root, "delta_dummy"), kind=ControlFieldKind.LINE,
            value_role="float", parameter_name="delta_dummy"),
        LegacyWidgetBinding(
            **base, label=f"{label_prefix} Chi Offset",
            path=(root, "chi_offset"), kind=ControlFieldKind.LINE,
            value_role="float", parameter_name="chi_offset"),
        LegacyWidgetBinding(
            **base, label=f"{label_prefix} Safe",
            path=(root, "safe"), kind=ControlFieldKind.BOOL,
            value_role="checked", parameter_name="safe"),
    )


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
        SectionId.PROCESSING, "1D Unit", ("Int1D", "unit"),
        ControlFieldKind.COMBO, "unit_1D", "current_text", "unit_1D",
        parameter_group="1d", tools=INT_1D_OUTPUT_TOOLS),
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
    *_int_advanced_specs("Int1D", "1D", "1d", INT_1D_OUTPUT_TOOLS),
    LegacyWidgetBinding(
        SectionId.PROCESSING, "2D Unit", ("Int2D", "unit"),
        ControlFieldKind.COMBO, "unit_2D", "current_text", "unit_2D",
        parameter_group="2d", tools=INT_2D_OUTPUT_TOOLS),
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
    *_int_advanced_specs("Int2D", "2D", "2d", INT_2D_OUTPUT_TOOLS),
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


NATIVE_CONTROL_PATHS: frozenset[tuple[str, ...]] = frozenset({
    ("Source", "energy_preference"),
})


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


_NATIVE_GI_ONLY_ARGS: frozenset[str] = frozenset(
    {
        "incident_angle",
        "incidence_motor",
        "tilt_angle",
        "sample_orientation",
        "method",
        "mode_1d",
        "mode_2d",
        "npt_oop",
        "gi_mode_1d",
        "gi_mode_2d",
        "npt_ip",
        "x_range",
        "y_range",
    }
)


def _native_pop_first(
    args: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for key in keys:
        if key in args:
            return args.pop(key)
    return default


def _native_npt_2d(args_2d: dict[str, Any]) -> tuple[int, int]:
    npt = args_2d.pop("npt", None)
    if isinstance(npt, (tuple, list)) and len(npt) == 2:
        return int(npt[0]), int(npt[1])
    npt_rad = args_2d.pop("npt_rad", None)
    npt_azim = args_2d.pop("npt_azim", None)
    if npt_rad is None:
        npt_rad = npt if npt is not None else 1000
    if npt_azim is None:
        npt_azim = 360
    return int(npt_rad), int(npt_azim)


def _native_strip_nonstandard_args(args: dict[str, Any]) -> None:
    for key in _NATIVE_GI_ONLY_ARGS:
        args.pop(key, None)


def _native_gi_1d_unit_default(unit: Any, mode: str, *, is_gi: bool) -> str:
    if not is_gi:
        return str(unit or "q_A^-1")
    if mode == "q_ip":
        return "qip_A^-1"
    if mode == "q_oop":
        return "qoop_A^-1"
    return str(unit or "q_A^-1")


def _native_gi_2d_unit_default(unit: Any, mode: str, *, is_gi: bool) -> str:
    text = str(unit or "").strip()
    if not is_gi:
        return text or "q_A^-1"
    if mode == "qip_qoop":
        return text if text.startswith("qip_") else "qip_A^-1"
    return text or "q_A^-1"


def build_native_int_reduction_plan_from_args(
    bai_1d_args: Mapping[str, Any] | None,
    bai_2d_args: Mapping[str, Any] | None,
    *,
    gi_enabled: bool = False,
    gi_incident_angle: Any = None,
    incidence_motor: Any = None,
    tilt_angle: Any = 0.0,
    sample_orientation: Any = 4,
    integrate_1d: bool = True,
    integrate_2d: bool = True,
    threshold_min: Any = None,
    threshold_max: Any = None,
    mask_saturation: bool = False,
    detector_mask: Any = None,
    detector_shape: tuple[int, int] | None = None,
):
    """Build the native Controls V2 Int reduction plan.

    This mirrors ``xdart.modules.reduction.plan_from_live_scan`` but takes the
    already-synced Controls V2 argument dictionaries directly.  It stays in the
    pure controls layer as the pre-flip equivalence gate; callers still decide
    when this becomes the production plan source.
    """

    from xrd_tools.reduction import (  # lazy: preserve controls_logic import purity
        GIMode,
        Integration1DPlan,
        Integration2DPlan,
        ReductionPlan,
    )
    from xrd_tools.reduction.masks import _mask_for_plan

    args_1d = dict(bai_1d_args or {})
    args_2d = dict(bai_2d_args or {})

    unit_1d = _native_pop_first(args_1d, ("unit",), "q_A^-1")
    unit_2d = _native_pop_first(args_2d, ("unit",), "q_A^-1")
    method_1d = _native_pop_first(args_1d, ("method",), "csr")
    method_2d = _native_pop_first(args_2d, ("method",), "csr")

    npt_1d = int(_native_pop_first(args_1d, ("npt", "numpoints", "npt_rad"), 1000))
    npt_rad_1d = int(_native_pop_first(args_1d, ("chi_npt_rad",), 1000))
    npt_rad_2d, npt_azim_2d = _native_npt_2d(args_2d)

    radial_range_1d = _native_pop_first(args_1d, ("radial_range",), None)
    azimuth_range_1d = _native_pop_first(args_1d, ("azimuth_range",), None)
    radial_range_2d = _native_pop_first(args_2d, ("radial_range",), None)
    azimuth_range_2d = _native_pop_first(args_2d, ("azimuth_range",), None)
    azimuth_offset_2d = float(
        _native_pop_first(args_2d, ("azimuth_offset", "chi_offset"), 0.0) or 0.0
    )
    chi_offset_1d = float(_native_pop_first(args_1d, ("chi_offset",), 0.0) or 0.0)
    # S-4: carry chi_offset as the 1D plan's azimuth_offset and re-add it to the
    # OUTPUT chi axis (in reduction), mirroring the 2D EXACTLY -- instead of
    # shifting the INPUT range and leaving the written 1D chi axis in the raw
    # pyFAI frame 90deg out of frame with the 2D cake chi.  GI keeps offset 0
    # (its chi handling is separate; the 2D also zeroes azimuth_offset for GI).
    azimuth_offset_1d = chi_offset_1d if not gi_enabled else 0.0

    monitor_1d = _native_pop_first(args_1d, ("monitor",), None)
    monitor_2d = _native_pop_first(args_2d, ("monitor",), None)
    error_1d = _native_pop_first(args_1d, ("error_model",), None)
    error_2d = _native_pop_first(args_2d, ("error_model",), None)
    pol_1d = _native_pop_first(args_1d, ("polarization_factor",), None)
    pol_2d = _native_pop_first(args_2d, ("polarization_factor",), None)
    _native_pop_first(args_1d, ("normalization_factor",), None)
    _native_pop_first(args_2d, ("normalization_factor",), None)

    gi_mode_1d = str(_native_pop_first(args_1d, ("gi_mode_1d",), "q_total"))
    gi_mode_2d = str(_native_pop_first(args_2d, ("gi_mode_2d",), "qip_qoop"))
    npt_oop = _native_pop_first(args_1d, ("npt_oop",), None)
    if npt_oop is None:
        npt_oop = _native_pop_first(args_2d, ("npt_oop",), None)
    gi_method = _native_pop_first(args_1d, ("gi_method_1d",), None)
    if gi_method is None:
        gi_method = _native_pop_first(args_2d, ("gi_method_2d",), "no")
    gi_method = str(gi_method)

    if not gi_enabled:
        _native_strip_nonstandard_args(args_1d)
        _native_strip_nonstandard_args(args_2d)

    gi = None
    if gi_enabled:
        incident = gi_incident_angle
        motor = None if incidence_motor is None else str(incidence_motor)
        if incident is None and motor is not None:
            try:
                incident = float(motor)
                motor = None
            except (TypeError, ValueError):
                pass
        if incident is None and motor is None:
            raise ValueError(
                "GI reduction requires an incident angle or incidence motor"
            )
        gi = GIMode(
            incident_angle=incident,
            incidence_motor=motor,
            tilt_angle=float(tilt_angle or 0.0),
            sample_orientation=int(sample_orientation or 4),
            method=gi_method,
            mode_1d=gi_mode_1d,
            mode_2d=gi_mode_2d,
            npt_oop=None if npt_oop is None else int(npt_oop),
        )

    integration_1d = None
    if integrate_1d:
        integration_1d = Integration1DPlan(
            npt=npt_1d,
            npt_rad=npt_rad_1d,
            unit=_native_gi_1d_unit_default(
                unit_1d, gi_mode_1d, is_gi=gi_enabled
            ),
            method=str(method_1d),
            radial_range=radial_range_1d,
            azimuth_range=azimuth_range_1d,
            monitor_key=monitor_1d,
            error_model=error_1d,
            polarization_factor=pol_1d,
            azimuth_offset=azimuth_offset_1d,
            extra=args_1d,
        )

    integration_2d = None
    if integrate_2d:
        integration_2d = Integration2DPlan(
            npt_rad=npt_rad_2d,
            npt_azim=npt_azim_2d,
            unit=_native_gi_2d_unit_default(
                unit_2d, gi_mode_2d, is_gi=gi_enabled
            ),
            method=str(method_2d),
            radial_range=radial_range_2d,
            azimuth_range=azimuth_range_2d,
            azimuth_offset=azimuth_offset_2d,
            monitor_key=monitor_2d,
            error_model=error_2d,
            polarization_factor=pol_2d,
            extra=args_2d,
        )

    return ReductionPlan(
        integration_1d=integration_1d,
        integration_2d=integration_2d,
        gi=gi,
        mask=_mask_for_plan(detector_mask, detector_shape),
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        mask_saturation=bool(mask_saturation),
    )


def build_native_int_reduction_plan_from_scan(
    scan: Any,
    *,
    integrate_1d: bool = True,
    integrate_2d: bool | None = None,
    threshold_min: Any = None,
    threshold_max: Any = None,
    mask_saturation: bool = False,
):
    """Build the native Controls V2 Int plan from a live-scan-like object.

    This is the Qt-free run-path form of
    :func:`build_native_int_reduction_plan_from_args`: it reads only typed scan
    attributes and ``bai_*_args`` dictionaries, not legacy widgets.
    """

    if integrate_2d is None:
        integrate_2d = not bool(getattr(scan, "skip_2d", False))

    gi_config = dict(getattr(scan, "gi_config", {}) or {})

    def _gi_value(name: str, default: Any) -> Any:
        value = getattr(scan, name, None)
        if value is None:
            value = gi_config.get(name, None)
        return default if value is None else value

    detector_shape = getattr(scan, "detector_shape", None)
    if detector_shape is not None:
        try:
            detector_shape = (int(detector_shape[0]), int(detector_shape[1]))
        except (TypeError, ValueError, IndexError):
            detector_shape = None
    if detector_shape is None:
        try:
            first_idx = scan.frames.index[0]
            first_img = getattr(scan.frames[int(first_idx)], "map_raw", None)
            detector_shape = getattr(first_img, "shape", None)
        except Exception:
            detector_shape = None

    return build_native_int_reduction_plan_from_args(
        dict(getattr(scan, "bai_1d_args", {}) or {}),
        dict(getattr(scan, "bai_2d_args", {}) or {}),
        gi_enabled=bool(getattr(scan, "gi", False)),
        gi_incident_angle=getattr(scan, "_cached_fiber_integrator_angle", None),
        incidence_motor=getattr(scan, "incidence_motor", None),
        tilt_angle=_gi_value("tilt_angle", 0.0),
        sample_orientation=_gi_value("sample_orientation", 4),
        integrate_1d=integrate_1d,
        integrate_2d=bool(integrate_2d),
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        mask_saturation=mask_saturation,
        detector_mask=getattr(scan, "global_mask", None),
        detector_shape=detector_shape,
    )


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
        parameter_group: str = "",
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
            parameter_group=parameter_group,
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
        add(SectionId.SOURCE, "Energy", ("Source", "energy_preference"),
            kind=ControlFieldKind.COMBO)
        if str(values.get(("Signal", "meta_ext"), "")).strip().lower() == "spec":
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
            parameter_group=spec.parameter_group,
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
    project_root_required: bool = False
    project_root_valid: bool = True
    source_label: str = ""
    run_target: RunTarget | str = ""
    loaded_scan_available: bool = False
    save_path: str = ""
    write_mode: str = "Append"
    processed_config: ProcessingConfigSignature | Mapping[str, Any] | None = None
    current_config: ProcessingConfigSignature | Mapping[str, Any] | None = None
    frame_count: int = 0
    #: DIR-2: container directories report a FILE count, not frames —
    #: the summary chip renders 'N files' instead of 'N frames'.
    frame_count_is_files: bool = False
    processing_mode: str = ""
    real_data_gates: frozenset[str] = field(default_factory=frozenset)
    controls_locked: bool = False
    detector_summary: str = ""


@dataclass(frozen=True, slots=True)
class ControlProfile:
    processing_page: ProcessingPage
    run_enabled: bool
    run_blockers: tuple[str, ...] = ()
    append_confirm_reason: str = ""
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
    detector_summary: str = ""

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
        if required:
            return StatusKind.MISSING, f"Select backend {required}."
        return StatusKind.DEFERRED, ""
    if required and state.backend != required:
        return (
            StatusKind.CONFLICT,
            f"{state.mode.value} {state.tool.value} requires backend {required}.",
        )
    return StatusKind.OK, ""


def effective_run_target(state: ControlState) -> RunTarget:
    """Return what the Run button would operate on for this state.

    A configured frame source is the only target that enables a fresh Run.
    A loaded processed scan can still be re-integrated, but it is not a fresh
    source and must not satisfy the Run gate.
    """

    raw = getattr(state, "run_target", "")
    target = ""
    if isinstance(raw, RunTarget):
        return raw
    try:
        target = str(raw or "").strip().lower()
    except Exception:
        target = ""
    for candidate in RunTarget:
        if target == candidate.value:
            return candidate

    caps = state.source_caps
    if caps.has_frames or caps.raw_reachable:
        return RunTarget.SOURCE
    if getattr(state, "loaded_scan_available", False):
        return RunTarget.LOADED_SCAN
    return RunTarget.NONE


def run_target_readiness_note(
    state: ControlState, *, ready: bool = False
) -> str:
    """Return the optional readiness-row note for the current run target."""

    target = effective_run_target(state)
    if target == RunTarget.LOADED_SCAN:
        return LOADED_SCAN_RUN_MESSAGE
    return ""


def build_field_statuses(state: ControlState) -> Mapping[FieldId, FieldStatus]:
    caps = state.source_caps
    geom = state.geom
    results = state.result_caps
    project_root = str(state.project_root or "")
    project_ready = bool(project_root) and bool(state.project_root_valid)
    if project_ready:
        project_status = StatusKind.OK
        project_reason = ""
    elif state.project_root_required:
        project_status = StatusKind.MISSING
        project_reason = (
            "Choose a valid project folder."
            if project_root
            else "Choose a project folder."
        )
    elif project_root and not state.project_root_valid:
        project_status = StatusKind.CONFLICT
        project_reason = "Project folder does not exist."
    else:
        project_status = StatusKind.DEFERRED
        project_reason = "Optional; improves portable paths."
    run_target = effective_run_target(state)
    source_selected = run_target == RunTarget.SOURCE
    source_label = (
        state.source_label
        or ("Live source" if source_selected else "No source selected")
    )
    frame_value = str(state.frame_count) if state.frame_count else ""
    energy_status, energy_value, energy_reason = _energy_status_reason(geom, caps)
    backend_status, backend_reason = _processing_backend_status_reason(state)
    fields = {
        FieldId.PROJECT_ROOT: _field(
            FieldId.PROJECT_ROOT,
            project_status,
            value=project_root,
            reason=project_reason),
        FieldId.SOURCE_PATH: _field(
            FieldId.SOURCE_PATH,
            StatusKind.OK if source_selected else StatusKind.MISSING,
            value=source_label,
            reason=(
                ""
                if source_selected
                else LOADED_SCAN_RUN_MESSAGE
                if run_target == RunTarget.LOADED_SCAN
                else "Choose a data source."
            )),
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
              required_caps: frozenset[ResultCap] | None = None,
              optional_deps: frozenset[str] | None = None,
              singleton_key: str = "") -> AnalysisLauncherSpec:
    return AnalysisLauncherSpec(
        tool=tool, label=label, enabled=False, reason=reason,
        live_capable=live, batch_capable=batch,
        production_ready=production_ready,
        required_caps=required_caps or frozenset(),
        optional_deps=optional_deps or frozenset(),
        singleton_key=singleton_key or tool.value)


def _enabled(tool: AnalysisTool, label: str, *,
             reason: str = "", live: bool = False, batch: bool = False,
             production_ready: bool = True,
             required_caps: frozenset[ResultCap] | None = None,
             optional_deps: frozenset[str] | None = None,
             singleton_key: str = "") -> AnalysisLauncherSpec:
    return AnalysisLauncherSpec(
        tool=tool, label=label, enabled=True, reason=reason,
        live_capable=live, batch_capable=batch,
        production_ready=production_ready,
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
    peak = (_enabled(AnalysisTool.PEAK_FIT, "Peak Fitting",
                     reason="Waiting for a 1D pattern." if not caps.has_1d else "",
                     live=True, batch=True,
                     required_caps=peak_caps,
                     optional_deps=frozenset({"fitting"}))
            if caps.has_optional_dep("fitting")
            else _disabled(AnalysisTool.PEAK_FIT, "Peak Fitting",
                           "Install fitting dependencies.", live=True, batch=True,
                           required_caps=peak_caps,
                           optional_deps=frozenset({"fitting"})))

    phase_caps = frozenset({ResultCap.HAS_1D})
    phase = (_enabled(AnalysisTool.PHASE_FIT, "Phase Fitting",
                      reason="Waiting for a 1D pattern." if not caps.has_1d else "",
                      batch=True,
                      required_caps=phase_caps,
                      optional_deps=frozenset({"fitting"}))
             if caps.has_optional_dep("fitting")
             else _disabled(AnalysisTool.PHASE_FIT, "Phase Fitting",
                            "Install fitting dependencies.", batch=True,
                            required_caps=phase_caps,
                            optional_deps=frozenset({"fitting"})))

    scan_plot = _enabled(
        AnalysisTool.SCAN_PLOT, "Plot Metadata",
        reason="" if caps.has_scan_metadata else "Choose a scan/source in the dialog.",
        batch=True,
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
        required_caps=frozenset({ResultCap.RAW_REACHABLE}),
        singleton_key="roi_stats",
    )

    strain_ready = caps.has_1d and caps.has_psi_metadata
    strain = (_enabled(AnalysisTool.SIN2PSI, "Strain / sin²ψ",
                       live=False, batch=True, production_ready=False,
                       required_caps=frozenset({
                           ResultCap.HAS_1D,
                           ResultCap.PSI_METADATA,
                       }),
                       singleton_key="sin2psi")
              if strain_ready
              else _disabled(AnalysisTool.SIN2PSI, "Strain / sin²ψ",
                             "Needs 1D patterns with ψ metadata.",
                             batch=True, production_ready=False,
                             required_caps=frozenset({
                                 ResultCap.HAS_1D,
                                 ResultCap.PSI_METADATA,
                             }),
                             singleton_key="sin2psi"))

    texture_ready = caps.has_1d or caps.has_phase_result
    texture = (_enabled(AnalysisTool.TEXTURE, "Texture / Orientation",
                        batch=True, production_ready=False,
                        required_caps=frozenset({ResultCap.PHASE_RESULT}),
                        singleton_key="texture")
               if texture_ready
               else _disabled(AnalysisTool.TEXTURE, "Texture / Orientation",
                              "Needs fitted phases or suitable 1D patterns.",
                              batch=True, production_ready=False,
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
    FieldId.SOURCE_PATH: "Choose a data source.",
    FieldId.SOURCE_FRAMES: "Choose a frame source.",
    FieldId.CALIBRATION_PONI: "Load a PONI file.",
    FieldId.BEAM_ENERGY: "Load PONI wavelength or source energy.",
    FieldId.SAMPLE_ORIENTATION: "Set GI sample orientation.",
    FieldId.SOURCE_MOTORS: "RSM needs motor metadata.",
    FieldId.DIFFRACTOMETER_UB: "Set/refine UB matrix before RSM.",
})


def run_required_fields_for(state: ControlState) -> tuple[FieldId, ...]:
    required: list[FieldId] = []
    if state.tool in (Tool.INT_1D, Tool.INT_2D, Tool.STITCH, Tool.RSM):
        if state.project_root_required:
            required.append(FieldId.PROJECT_ROOT)
        required.extend((
            FieldId.CALIBRATION_PONI,
            FieldId.BEAM_ENERGY,
            FieldId.SOURCE_PATH,
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
    run_target = effective_run_target(state)
    for field_id in run_required_fields_for(state):
        field = fields.get(field_id)
        if field is None:
            continue
        if field.status == StatusKind.MISSING:
            if field_id == FieldId.SOURCE_PATH and run_target == RunTarget.LOADED_SCAN:
                blockers.append(LOADED_SCAN_RUN_MESSAGE)
            else:
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


def append_confirm_reason_for(state: ControlState) -> str:
    """Return the non-blocking Append overwrite confirmation reason, if any."""

    run_target = effective_run_target(state)
    if state.tool not in (Tool.INT_1D, Tool.INT_2D) or run_target != RunTarget.SOURCE:
        return ""
    append_check = append_config_mismatch_check(
        state.write_mode,
        state.processed_config,
        state.current_config,
    )
    if append_check.ok:
        return ""
    return str(append_check.reason or "")


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
    append_confirm_reason = append_confirm_reason_for(state)
    return ControlProfile(
        processing_page=page,
        run_enabled=not blockers and not viewer,
        run_blockers=blockers,
        append_confirm_reason=append_confirm_reason,
        valid_modes=valid_modes_for(state.tool),
        backend_required=backend_required_for(state),
        fields=fields,
        section_actions=build_section_actions(state),
        analysis_launchers=build_analysis_launchers(state.result_caps),
        show_experiment_card=True,
        show_processing_card=not viewer,
        detector_summary=state.detector_summary)


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
