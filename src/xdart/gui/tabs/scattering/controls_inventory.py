"""Native vNext Controls field inventory projected from ``RunIntent``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path

from xrd_tools.core.scan import SourceKind, SourceSpec, coerce_source_kind
from xrd_tools.session.control_labels import (
    range_axis_labels_1d,
    range_axis_labels_2d,
)
from xrd_tools.session.readiness import (
    BoundControlState,
    ControlFieldKind,
    ControlFormField,
    SectionId,
    Tool,
)
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    is_single_image_spec,
)

from .contracts import SourceSelection


PROJECT_ROOT = ("Project", "project_folder")
SAVE_PATH = ("Project", "h5_dir")
OUTPUT_MODE = ("Project", "output_mode")
PONI_FILE = ("Signal", "poni_file")
MASK_FILE = ("Signal", "mask_file")
GI_ENABLED = ("GI", "Grazing")
GI_MOTOR = ("GI", "th_motor")
GI_THETA = ("GI", "th_val")
GI_ORIENTATION = ("GI", "sample_orientation")
GI_TILT = ("GI", "tilt_angle")
THRESHOLD_ENABLED = ("Mask", "Threshold")
THRESHOLD_MIN = ("Mask", "min")
THRESHOLD_MAX = ("Mask", "max")
MASK_SATURATION = ("MaskSat", "mask_sentinel")
SOURCE_TYPE = ("Signal", "inp_type")
SOURCE_FILE = ("Signal", "File")
SOURCE_DIRECTORY = ("Signal", "img_dir")
SOURCE_RECURSIVE = ("Signal", "include_subdir")
SOURCE_SUFFIX = ("Signal", "img_ext")
SOURCE_FILTER = ("Signal", "Filter")
SOURCE_META = ("Signal", "meta_ext")
SOURCE_ENERGY = ("Source", "energy_preference")
AVERAGE_SCAN = ("Signal", "series_average")
BACKGROUND_TYPE = ("BG", "bg_type")
BACKGROUND_FILE = ("BG", "File")
BACKGROUND_DIRECTORY = ("BG", "Directory")
BACKGROUND_MATCH = ("BG", "Match")
BACKGROUND_METADATA_KEY = ("BG", "Metadata Key")
BACKGROUND_FILTER = ("BG", "Filter")
BACKGROUND_SCALE = ("BG", "Scale")
BACKGROUND_NORMALIZE = ("BG", "Normalize")
BACKGROUND_EDIT_PATHS = frozenset({BACKGROUND_TYPE, BACKGROUND_FILE,
    BACKGROUND_DIRECTORY, BACKGROUND_MATCH, BACKGROUND_METADATA_KEY,
    BACKGROUND_FILTER, BACKGROUND_SCALE, BACKGROUND_NORMALIZE})

INT_1D_AXIS = ("Int1D", "axis")
INT_1D_POINTS = ("Int1D", "points")
INT_1D_RADIAL_AUTO = ("Int1D", "radial_auto")
INT_1D_RADIAL_LOW = ("Int1D", "radial_low")
INT_1D_RADIAL_HIGH = ("Int1D", "radial_high")
INT_1D_AZIM_AUTO = ("Int1D", "azim_auto")
INT_1D_AZIM_LOW = ("Int1D", "azim_low")
INT_1D_AZIM_HIGH = ("Int1D", "azim_high")
INT_2D_AXIS = ("Int2D", "axis")
INT_2D_RADIAL_POINTS = ("Int2D", "radial_points")
INT_2D_AZIM_POINTS = ("Int2D", "azim_points")
INT_2D_RADIAL_AUTO = ("Int2D", "radial_auto")
INT_2D_RADIAL_LOW = ("Int2D", "radial_low")
INT_2D_RADIAL_HIGH = ("Int2D", "radial_high")
INT_2D_AZIM_AUTO = ("Int2D", "azim_auto")
INT_2D_AZIM_LOW = ("Int2D", "azim_low")
INT_2D_AZIM_HIGH = ("Int2D", "azim_high")

INT_PATHS = frozenset({
    INT_1D_AXIS,
    INT_1D_POINTS,
    INT_1D_RADIAL_AUTO,
    INT_1D_RADIAL_LOW,
    INT_1D_RADIAL_HIGH,
    INT_1D_AZIM_AUTO,
    INT_1D_AZIM_LOW,
    INT_1D_AZIM_HIGH,
    INT_2D_AXIS,
    INT_2D_RADIAL_POINTS,
    INT_2D_AZIM_POINTS,
    INT_2D_RADIAL_AUTO,
    INT_2D_RADIAL_LOW,
    INT_2D_RADIAL_HIGH,
    INT_2D_AZIM_AUTO,
    INT_2D_AZIM_LOW,
    INT_2D_AZIM_HIGH,
})


@dataclass(frozen=True, slots=True)
class ControlFieldSpec:
    """One native vNext field declaration.

    The declaration contains only facts consumed by the vNext projection.  It
    deliberately has no Qt widget names, value roles, or ParameterTree
    coordinates; the Qt-backed static-scan adapter owns those separately.
    """

    section: SectionId
    label: str
    path: tuple[str, ...]
    kind: ControlFieldKind = ControlFieldKind.LINE
    tools: frozenset[Tool] = frozenset({Tool.INT_1D, Tool.INT_2D})


_INT_1D_TOOLS = frozenset({Tool.INT_1D, Tool.INT_2D})
_INT_2D_TOOLS = frozenset({Tool.INT_2D})


CONTROL_FIELD_SPECS: tuple[ControlFieldSpec, ...] = (
    ControlFieldSpec(
        SectionId.EXPERIMENT,
        "Grazing",
        GI_ENABLED,
        ControlFieldKind.BOOL,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "Threshold",
        THRESHOLD_ENABLED,
        ControlFieldKind.BOOL,
    ),
    ControlFieldSpec(SectionId.PROCESSING, "Min", THRESHOLD_MIN),
    ControlFieldSpec(SectionId.PROCESSING, "Max", THRESHOLD_MAX),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "Mask Saturated",
        MASK_SATURATION,
        ControlFieldKind.BOOL,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "1D Axis",
        INT_1D_AXIS,
        ControlFieldKind.COMBO,
        _INT_1D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "1D Points",
        INT_1D_POINTS,
        tools=_INT_1D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "1D Radial Auto",
        INT_1D_RADIAL_AUTO,
        ControlFieldKind.BOOL,
        _INT_1D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "1D Radial Low",
        INT_1D_RADIAL_LOW,
        tools=_INT_1D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "1D Radial High",
        INT_1D_RADIAL_HIGH,
        tools=_INT_1D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "1D Azim Auto",
        INT_1D_AZIM_AUTO,
        ControlFieldKind.BOOL,
        _INT_1D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "1D Azim Low",
        INT_1D_AZIM_LOW,
        tools=_INT_1D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "1D Azim High",
        INT_1D_AZIM_HIGH,
        tools=_INT_1D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "2D Axis",
        INT_2D_AXIS,
        ControlFieldKind.COMBO,
        _INT_2D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "2D Radial Points",
        INT_2D_RADIAL_POINTS,
        tools=_INT_2D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "2D Azim Points",
        INT_2D_AZIM_POINTS,
        tools=_INT_2D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "2D Radial Auto",
        INT_2D_RADIAL_AUTO,
        ControlFieldKind.BOOL,
        _INT_2D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "2D Radial Low",
        INT_2D_RADIAL_LOW,
        tools=_INT_2D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "2D Radial High",
        INT_2D_RADIAL_HIGH,
        tools=_INT_2D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "2D Azim Auto",
        INT_2D_AZIM_AUTO,
        ControlFieldKind.BOOL,
        _INT_2D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "2D Azim Low",
        INT_2D_AZIM_LOW,
        tools=_INT_2D_TOOLS,
    ),
    ControlFieldSpec(
        SectionId.PROCESSING,
        "2D Azim High",
        INT_2D_AZIM_HIGH,
        tools=_INT_2D_TOOLS,
    ),
)

SOURCE_EDIT_PATHS = frozenset({
    SOURCE_DIRECTORY,
    SOURCE_RECURSIVE,
    SOURCE_SUFFIX,
    SOURCE_FILTER,
    SOURCE_META,
})

STANDARD_1D_AXES = {
    "Q (Å⁻¹)": "q_A^-1",
    "2θ (°)": "2th_deg",
    "χ (°)": "chi_deg",
}
STANDARD_2D_AXES = {
    "Q-χ": "q_A^-1",
    "2θ-χ": "2th_deg",
}
GI_1D_AXES = {
    "Q": "q_total",
    "Qip": "q_ip",
    "Qoop": "q_oop",
    "Exit": "exit_angle",
    "χGI": "chi_gi",
}
GI_2D_AXES = {
    "Qip-Qoop": "qip_qoop",
    "Q-χ": "q_chi",
    "Exit": "exit_angles",
}
SOURCE_FORMAT_SUFFIXES = {
    "tif": (".tif",),
    "tiff": (".tiff",),
    "h5": (".h5",),
    "hdf5": (".hdf5",),
    "nxs": (".nxs",),
    "cxi": (".cxi",),
    "cbf": (".cbf",),
    "edf": (".edf",),
    "raw": (".raw",),
}


def build_native_control_state(
    values: Mapping[tuple[str, ...], object] | None = None,
    choices: Mapping[tuple[str, ...], Sequence[object]] | None = None,
    *,
    tool: Tool | None = None,
    controls_enabled: bool = True,
) -> BoundControlState:
    """Build vNext's renderer state from native field declarations."""

    values = {tuple(path): value for path, value in (values or {}).items()}
    choices = {
        tuple(path): tuple(str(value) for value in options)
        for path, options in (choices or {}).items()
    }
    label_overrides = _integration_label_overrides(values)
    projected: list[ControlFormField] = []

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
        projected.append(ControlFormField(
            section=section,
            label=label,
            path=path,
            value=values[path],
            kind=kind,
            choices=choices.get(path, ()),
            browse=browse,
            enabled=enabled,
            reason=reason,
        ))

    add(SectionId.PROJECT, "Folder", PROJECT_ROOT, browse=True)
    add(SectionId.PROJECT, "Save Path", SAVE_PATH, browse=True)

    source_type = str(values.get(SOURCE_TYPE, ""))
    add(SectionId.SOURCE, "Source", SOURCE_TYPE, kind=ControlFieldKind.COMBO)
    if source_type == "Image Directory":
        add(SectionId.SOURCE, "Directory", SOURCE_DIRECTORY, browse=True)
        add(
            SectionId.SOURCE,
            "File Type",
            SOURCE_SUFFIX,
            kind=ControlFieldKind.COMBO,
        )
        add(
            SectionId.SOURCE,
            "Subdirs",
            SOURCE_RECURSIVE,
            kind=ControlFieldKind.BOOL,
        )
        add(SectionId.SOURCE, "Filter", SOURCE_FILTER)
    else:
        add(SectionId.SOURCE, "Image File", SOURCE_FILE, browse=True)
    add(
        SectionId.SOURCE,
        "Meta Type",
        SOURCE_META,
        kind=ControlFieldKind.COMBO,
    )
    add(
        SectionId.SOURCE,
        "Energy",
        SOURCE_ENERGY,
        kind=ControlFieldKind.COMBO,
    )
    add(SectionId.EXPERIMENT, "Poni", PONI_FILE, browse=True)
    add(SectionId.EXPERIMENT, "Mask File", MASK_FILE, browse=True)

    for spec in CONTROL_FIELD_SPECS:
        if tool is not None and tool not in spec.tools:
            continue
        add(
            spec.section,
            label_overrides.get(spec.path, spec.label),
            spec.path,
            kind=spec.kind,
        )

    if source_type != "Single Image" and AVERAGE_SCAN in values:
        add(
            SectionId.PROCESSING,
            "Average Scan",
            AVERAGE_SCAN,
            kind=ControlFieldKind.BOOL,
        )
    background_mode = str(values.get(BACKGROUND_TYPE, "None"))
    add(
        SectionId.PROCESSING,
        "Background",
        BACKGROUND_TYPE,
        kind=ControlFieldKind.COMBO,
    )
    if background_mode in {"Single BG File", "Series Average"}:
        add(
            SectionId.PROCESSING,
            (
                "Source File"
                if background_mode == "Single BG File"
                else "Series Member"
            ),
            BACKGROUND_FILE,
            browse=True,
        )
    elif background_mode == "BG Directory":
        add(
            SectionId.PROCESSING,
            "Directory",
            BACKGROUND_DIRECTORY,
            browse=True,
        )
        add(
            SectionId.PROCESSING,
            "Match",
            BACKGROUND_MATCH,
            kind=ControlFieldKind.COMBO,
        )
        if str(values.get(BACKGROUND_MATCH, "")) == "Metadata Key":
            add(
                SectionId.PROCESSING,
                "Metadata Key",
                BACKGROUND_METADATA_KEY,
            )
        add(
            SectionId.PROCESSING,
            "Filename Filter",
            BACKGROUND_FILTER,
        )
    if background_mode != "None":
        add(SectionId.PROCESSING, "Scale", BACKGROUND_SCALE)
        add(
            SectionId.PROCESSING,
            "Normalize",
            BACKGROUND_NORMALIZE,
        )

    return BoundControlState(fields=tuple(projected))


def _integration_label_overrides(
    values: Mapping[tuple[str, ...], object],
) -> dict[tuple[str, ...], str]:
    radial_1d, azim_1d = range_axis_labels_1d(values)
    radial_2d, azim_2d = range_axis_labels_2d(values)
    return {
        INT_1D_RADIAL_AUTO: f"{radial_1d} Auto",
        INT_1D_RADIAL_LOW: f"{radial_1d} Low",
        INT_1D_RADIAL_HIGH: f"{radial_1d} High",
        INT_1D_AZIM_AUTO: f"{azim_1d} Auto",
        INT_1D_AZIM_LOW: f"{azim_1d} Low",
        INT_1D_AZIM_HIGH: f"{azim_1d} High",
        INT_2D_RADIAL_AUTO: f"{radial_2d} Auto",
        INT_2D_RADIAL_LOW: f"{radial_2d} Low",
        INT_2D_RADIAL_HIGH: f"{radial_2d} High",
        INT_2D_AZIM_AUTO: f"{azim_2d} Auto",
        INT_2D_AZIM_LOW: f"{azim_2d} Low",
        INT_2D_AZIM_HIGH: f"{azim_2d} High",
    }


def _field_enabled_reason(
    values: Mapping[tuple[str, ...], object],
    path: tuple[str, ...],
    *,
    controls_enabled: bool,
) -> tuple[bool, str]:
    if not controls_enabled:
        return False, "Controls are locked during the active run."
    auto_dependencies = {
        INT_1D_RADIAL_LOW: INT_1D_RADIAL_AUTO,
        INT_1D_RADIAL_HIGH: INT_1D_RADIAL_AUTO,
        INT_1D_AZIM_LOW: INT_1D_AZIM_AUTO,
        INT_1D_AZIM_HIGH: INT_1D_AZIM_AUTO,
        INT_2D_RADIAL_LOW: INT_2D_RADIAL_AUTO,
        INT_2D_RADIAL_HIGH: INT_2D_RADIAL_AUTO,
        INT_2D_AZIM_LOW: INT_2D_AZIM_AUTO,
        INT_2D_AZIM_HIGH: INT_2D_AZIM_AUTO,
    }
    auto_path = auto_dependencies.get(path)
    if auto_path is not None and bool(values.get(auto_path, False)):
        return False, "Disable Auto to edit this range."
    if path in {THRESHOLD_MIN, THRESHOLD_MAX} and not bool(
        values.get(THRESHOLD_ENABLED, False)
    ):
        return False, "Enable Threshold to edit this limit."
    return True, ""


def bound_values(
    intent: RunIntent,
    *,
    motor_value: str,
    motor_choices: tuple[str, ...],
    source_mode: str | None = None,
) -> tuple[
    dict[tuple[str, ...], object],
    dict[tuple[str, ...], tuple[str, ...]],
]:
    values: dict[tuple[str, ...], object] = {
        PROJECT_ROOT: intent.project_root,
        SAVE_PATH: intent.save_path,
        PONI_FILE: intent.poni_file,
        MASK_FILE: intent.mask_file,
        GI_ENABLED: intent.gi.enabled,
        GI_MOTOR: motor_value,
        GI_THETA: intent.gi.th_val,
        GI_ORIENTATION: intent.gi.sample_orientation,
        GI_TILT: intent.gi.tilt_angle,
        THRESHOLD_ENABLED: intent.threshold.apply_threshold,
        THRESHOLD_MIN: intent.threshold.threshold_min,
        THRESHOLD_MAX: intent.threshold.threshold_max,
        MASK_SATURATION: intent.threshold.mask_saturation,
        BACKGROUND_TYPE: intent.background.mode,
        BACKGROUND_FILE: intent.background.locator or "",
        BACKGROUND_DIRECTORY: intent.background.locator or "",
        BACKGROUND_MATCH: intent.background.match_rule or "Scan Root + Frame Number",
        BACKGROUND_METADATA_KEY: intent.background.metadata_key or "",
        BACKGROUND_FILTER: intent.background.filename_filter,
        BACKGROUND_SCALE: intent.background.scale,
        BACKGROUND_NORMALIZE: intent.background.normalization_key or "None",
    }
    values.update(source_values(intent.source_spec, source_mode=source_mode))
    average = intent.run_options.get("series_average", False)
    if (
        type(intent.source_spec) is SourceSpec
        and not intent.live_mode
        and intent.output_mode == "Overwrite"
        and intent.processing_mode != "Int 1D (XYE)"
        and type(average) is bool
    ):
        values[AVERAGE_SCAN] = average
    values.update(integration_values(intent))
    choices: dict[tuple[str, ...], tuple[str, ...]] = {
        SOURCE_TYPE: (
            "Image Series",
            "Image Directory",
            "Single Image",
        ),
        SOURCE_SUFFIX: tuple(SOURCE_FORMAT_SUFFIXES),
        SOURCE_META: (
            "None",
            "Auto",
            "txt",
            "pdi",
            "metadata",
            "SPEC",
        ),
        SOURCE_ENERGY: ("poni", "metadata"),
        GI_MOTOR: motor_choices,
        BACKGROUND_TYPE: (
            "None",
            "Single BG File",
            "Series Average",
            "BG Directory",
        ),
        BACKGROUND_MATCH: ("Scan Root + Frame Number", "Metadata Key"),
    }
    if intent.gi.enabled:
        choices.update({
            INT_1D_AXIS: tuple(GI_1D_AXES),
            INT_2D_AXIS: tuple(GI_2D_AXES),
        })
    else:
        choices.update({
            INT_1D_AXIS: tuple(STANDARD_1D_AXES),
            INT_2D_AXIS: tuple(STANDARD_2D_AXES),
        })
    choices[("Int1D", "unit")] = (
        "q_A^-1",
        "2th_deg",
        "chi_deg",
    )
    choices[("Int2D", "unit")] = ("q_A^-1", "2th_deg")
    return values, choices


def source_values(
    source: SourceSelection | None,
    *,
    source_mode: str | None = None,
) -> dict[tuple[str, ...], object]:
    values: dict[tuple[str, ...], object] = {
        SOURCE_META: "Auto",
        SOURCE_ENERGY: "poni",
    }
    if type(source) is DirectorySourceSpec:
        suffix = next(
            (
                label
                for label, exact in SOURCE_FORMAT_SUFFIXES.items()
                if exact == source.suffixes
            ),
            ", ".join(source.suffixes),
        )
        values.update({
            SOURCE_TYPE: "Image Directory",
            SOURCE_DIRECTORY: str(source.root),
            SOURCE_RECURSIVE: source.recursive,
            SOURCE_SUFFIX: suffix,
            SOURCE_FILTER: source.name_filter or "",
            SOURCE_META: _metadata_format_label(source.metadata_format),
        })
        return values
    if type(source) is SourceSpec:
        try:
            kind = coerce_source_kind(source.kind)
        except (TypeError, ValueError):
            kind = SourceKind.UNKNOWN
        options = dict(source.options)
        selected = str(
            options.get("selected_file")
            or source.uri
            or ""
        )
        values.update({
            SOURCE_TYPE: (
                "Single Image"
                if (
                    kind is SourceKind.IMAGE_FILE
                    or is_single_image_spec(source)
                )
                else "Image Series"
            ),
            SOURCE_FILE: selected,
            SOURCE_META: _metadata_format_label(
                options.get("metadata_format", "auto")
            ),
        })
        return values
    if source_mode == "Image Directory":
        values.update({
            SOURCE_TYPE: "Image Directory",
            SOURCE_DIRECTORY: "",
            SOURCE_RECURSIVE: False,
            SOURCE_SUFFIX: next(iter(SOURCE_FORMAT_SUFFIXES)),
            SOURCE_FILTER: "",
        })
    else:
        values.update({
            SOURCE_TYPE: (
                "Single Image"
                if source_mode == "Single Image"
                else "Image Series"
            ),
            SOURCE_FILE: "",
        })
    return values


def source_mode(
    source: SourceSelection | None,
) -> str:
    """Visible source editor mode for one complete selection."""
    if type(source) is DirectorySourceSpec:
        return "Image Directory"
    if type(source) is SourceSpec:
        try:
            kind = coerce_source_kind(source.kind)
        except (TypeError, ValueError):
            kind = SourceKind.UNKNOWN
        return (
            "Single Image"
            if (
                kind is SourceKind.IMAGE_FILE
                or is_single_image_spec(source)
            )
            else "Image Series"
        )
    return "Image Series"


def integration_values(
    intent: RunIntent,
) -> dict[tuple[str, ...], object]:
    args_1d = dict(intent.bai_1d_args)
    args_2d = dict(intent.bai_2d_args)
    points = points_2d(args_2d)
    unit_1d = str(args_1d.get("unit") or "q_A^-1")
    unit_2d = str(args_2d.get("unit") or "q_A^-1")
    if intent.gi.enabled:
        axis_1d = label_for_value(
            GI_1D_AXES,
            intent.gi.mode_1d,
            "Q",
        )
        axis_2d = label_for_value(
            GI_2D_AXES,
            intent.gi.mode_2d,
            "Qip-Qoop",
        )
    else:
        axis_1d = label_for_value(
            STANDARD_1D_AXES,
            unit_1d,
            "Q (Å⁻¹)",
        )
        axis_2d = label_for_value(
            STANDARD_2D_AXES,
            unit_2d,
            "Q-χ",
        )
    values: dict[tuple[str, ...], object] = {
        ("Int1D", "unit"): unit_1d,
        INT_1D_AXIS: axis_1d,
        INT_1D_POINTS: first_int(
            args_1d,
            ("npt", "numpoints", "npt_rad"),
            1000,
        ),
        ("Int2D", "unit"): unit_2d,
        INT_2D_AXIS: axis_2d,
        INT_2D_RADIAL_POINTS: points[0],
        INT_2D_AZIM_POINTS: points[1],
    }
    if intent.gi.enabled:
        values[("Int1D", "gi_mode")] = intent.gi.mode_1d
        values[("Int2D", "gi_mode")] = intent.gi.mode_2d
    values.update(project_range(
        "Int1D",
        "radial",
        args_1d.get("radial_range"),
        default_radial_range(unit_1d),
    ))
    values.update(project_range(
        "Int1D",
        "azim",
        args_1d.get("azimuth_range"),
        (-180.0, 180.0),
    ))
    values.update(project_range(
        "Int2D",
        "radial",
        args_2d.get("radial_range"),
        default_radial_range(unit_2d),
    ))
    values.update(project_range(
        "Int2D",
        "azim",
        args_2d.get("azimuth_range"),
        (-180.0, 180.0),
    ))
    return values


def project_range(
    root: str,
    name: str,
    value: object,
    default: tuple[float, float],
) -> dict[tuple[str, ...], object]:
    parsed = range_value(value)
    low, high = default if parsed is None else parsed
    return {
        (root, f"{name}_auto"): parsed is None,
        (root, f"{name}_low"): low,
        (root, f"{name}_high"): high,
    }


def truthful_field(field: ControlFormField) -> ControlFormField:
    if not field.enabled:
        return field
    reasons = {
        SOURCE_ENERGY: "Energy-source selection is not available in vNext yet.",
    }
    reason = (
        "Use Choose source to replace this exact complete source."
        if (
            field.path == SOURCE_SUFFIX
            and field.value not in SOURCE_FORMAT_SUFFIXES
        )
        else reasons.get(field.path)
    )
    return (
        field
        if reason is None
        else replace(field, enabled=False, reason=reason)
    )


def _metadata_format_label(value: object) -> str:
    if value is None:
        return "None"
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "Auto"
    if normalized == "spec":
        return "SPEC"
    return normalized


def label_for_value(
    choices: dict[str, str],
    value: str,
    default: str,
) -> str:
    return next(
        (
            label
            for label, candidate in choices.items()
            if candidate == value
        ),
        default,
    )


def first_int(
    values: dict[object, object],
    keys: tuple[str, ...],
    default: object,
) -> int:
    for key in keys:
        if key in values:
            try:
                return int(values[key])
            except (TypeError, ValueError):
                break
    return int(default)


def range_value(value: object) -> tuple[float, float] | None:
    if type(value) not in {tuple, list} or len(value) != 2:
        return None
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        return None
    return low, high


def points_2d(values: dict[object, object]) -> tuple[int, int]:
    combined = values.get("npt")
    if type(combined) in {tuple, list} and len(combined) == 2:
        try:
            return int(combined[0]), int(combined[1])
        except (TypeError, ValueError):
            pass
    return (
        first_int(values, ("npt_rad", "npt"), 1000),
        first_int(values, ("npt_azim",), 360),
    )


def default_radial_range(unit: str) -> tuple[float, float]:
    return (
        (0.0, 90.0)
        if unit == "2th_deg"
        else (-180.0, 180.0)
        if unit == "chi_deg"
        else (0.0, 5.0)
    )


def field(
    section: SectionId,
    label: str,
    path: tuple[str, ...],
    value: object,
    *,
    kind: ControlFieldKind = ControlFieldKind.LINE,
    choices: tuple[str, ...] = (),
    browse: bool = False,
    reason: str = "",
) -> ControlFormField:
    return ControlFormField(
        section,
        label,
        path,
        value,
        kind,
        choices,
        browse,
        reason=reason,
    )


def source_name(source: SourceSelection) -> str:
    return (
        source.root.name
        if type(source) is DirectorySourceSpec
        else str(source.uri).rsplit("/", 1)[-1]
    )
