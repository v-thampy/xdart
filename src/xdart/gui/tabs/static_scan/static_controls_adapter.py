# -*- coding: utf-8 -*-
"""Static-page widget bindings for the shared Controls renderer.

This module is the only owner of the Qt-backed static-scan page's widget and
ParameterTree mapping.  The vNext ScatteringWorkspace uses its independent,
value-only control inventory and does not import this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from xrd_tools.session.control_labels import (
    range_axis_labels_1d,
    range_axis_labels_2d,
)
from xrd_tools.session.readiness import (
    BoundControlState,
    ControlFieldKind,
    ControlFormField,
    ControlPanelRenderState,
    ControlState,
    SectionId,
    Tool,
    build_control_profile,
)


@dataclass(frozen=True, slots=True)
class StaticWidgetBinding:
    """One static-page field backed by Qt or ParameterTree.

    ScatteringWorkspace has its independent value-only ``ControlFieldSpec``
    inventory and never imports this adapter.
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


INT_1D_OUTPUT_TOOLS = frozenset({Tool.INT_1D, Tool.INT_2D})
INT_2D_OUTPUT_TOOLS = frozenset({Tool.INT_2D})

def _integration_label_overrides(
    values: Mapping[tuple[str, ...], object],
) -> dict[tuple[str, ...], str]:
    radial_1d, azim_1d = range_axis_labels_1d(values)
    radial_2d, azim_2d = range_axis_labels_2d(values)
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
) -> tuple[StaticWidgetBinding, ...]:
    base = dict(
        section=SectionId.PROCESSING,
        parameter_group=parameter_group,
        tools=tools,
    )
    return (
        StaticWidgetBinding(
            **base, label=f"{label_prefix} Solid Angle",
            path=(root, "correctSolidAngle"), kind=ControlFieldKind.BOOL,
            value_role="checked", parameter_name="correctSolidAngle"),
        StaticWidgetBinding(
            **base, label=f"{label_prefix} Polarization",
            path=(root, "apply_polarization"), kind=ControlFieldKind.BOOL,
            value_role="checked", parameter_name="Apply polarization factor"),
        StaticWidgetBinding(
            **base, label=f"{label_prefix} Pol. Factor",
            path=(root, "polarization_factor"), kind=ControlFieldKind.LINE,
            value_role="float", parameter_name="polarization_factor"),
        StaticWidgetBinding(
            **base, label=f"{label_prefix} Method",
            path=(root, "method"), kind=ControlFieldKind.COMBO,
            value_role="current_text", parameter_name="method"),
        StaticWidgetBinding(
            **base, label=f"{label_prefix} Dummy",
            path=(root, "dummy"), kind=ControlFieldKind.LINE,
            value_role="float", parameter_name="dummy"),
        StaticWidgetBinding(
            **base, label=f"{label_prefix} Delta Dummy",
            path=(root, "delta_dummy"), kind=ControlFieldKind.LINE,
            value_role="float", parameter_name="delta_dummy"),
        StaticWidgetBinding(
            **base, label=f"{label_prefix} Chi Offset",
            path=(root, "chi_offset"), kind=ControlFieldKind.LINE,
            value_role="float", parameter_name="chi_offset"),
        StaticWidgetBinding(
            **base, label=f"{label_prefix} Safe",
            path=(root, "safe"), kind=ControlFieldKind.BOOL,
            value_role="checked", parameter_name="safe"),
    )


INTEGRATOR_BACKED_CONTROL_SPECS: tuple[StaticWidgetBinding, ...] = (
    StaticWidgetBinding(
        SectionId.EXPERIMENT, "Grazing", ("GI", "Grazing"),
        ControlFieldKind.BOOL, "gi_enable", "checked"),
    StaticWidgetBinding(
        SectionId.EXPERIMENT, "Theta Motor", ("GI", "th_motor"),
        ControlFieldKind.COMBO, "gi_motor", "current_text", "gi_motor",
        visible_when="grazing"),
    StaticWidgetBinding(
        SectionId.EXPERIMENT, "Theta", ("GI", "th_val"),
        ControlFieldKind.LINE, "gi_motor_value", "text",
        visible_when="grazing_manual"),
    StaticWidgetBinding(
        SectionId.EXPERIMENT, "Orientation", ("GI", "sample_orientation"),
        ControlFieldKind.LINE, "gi_sample_orientation", "value",
        visible_when="grazing"),
    StaticWidgetBinding(
        SectionId.EXPERIMENT, "Tilt Angle", ("GI", "tilt_angle"),
        ControlFieldKind.LINE, "gi_tilt", "text",
        visible_when="grazing"),
    StaticWidgetBinding(
        SectionId.PROCESSING, "Threshold", ("Mask", "Threshold"),
        ControlFieldKind.BOOL, "threshold_enable", "checked"),
    StaticWidgetBinding(
        SectionId.PROCESSING, "Min", ("Mask", "min"),
        ControlFieldKind.LINE, "threshold_min", "text"),
    StaticWidgetBinding(
        SectionId.PROCESSING, "Max", ("Mask", "max"),
        ControlFieldKind.LINE, "threshold_max", "text"),
    StaticWidgetBinding(
        SectionId.PROCESSING, "Mask Saturated", ("MaskSat", "mask_sentinel"),
        ControlFieldKind.BOOL, "mask_saturated", "checked"),
    StaticWidgetBinding(
        SectionId.PROCESSING, "1D Unit", ("Int1D", "unit"),
        ControlFieldKind.COMBO, "unit_1D", "current_text", "unit_1D",
        parameter_group="1d", tools=INT_1D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "1D Axis", ("Int1D", "axis"),
        ControlFieldKind.COMBO, "axis1D", "current_text", "axis1D",
        tools=INT_1D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "1D Points", ("Int1D", "points"),
        ControlFieldKind.LINE, "npts_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "1D OOP Points", ("Int1D", "points_oop"),
        ControlFieldKind.LINE, "npts_oop_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS, visible_when="widget_visible"),
    StaticWidgetBinding(
        SectionId.PROCESSING, "1D Radial Auto", ("Int1D", "radial_auto"),
        ControlFieldKind.BOOL, "radial_autoRange_1D", "checked",
        tools=INT_1D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "1D Radial Low", ("Int1D", "radial_low"),
        ControlFieldKind.LINE, "radial_low_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "1D Radial High", ("Int1D", "radial_high"),
        ControlFieldKind.LINE, "radial_high_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "1D Azim Auto", ("Int1D", "azim_auto"),
        ControlFieldKind.BOOL, "azim_autoRange_1D", "checked",
        tools=INT_1D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "1D Azim Low", ("Int1D", "azim_low"),
        ControlFieldKind.LINE, "azim_low_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "1D Azim High", ("Int1D", "azim_high"),
        ControlFieldKind.LINE, "azim_high_1D", "text",
        tools=INT_1D_OUTPUT_TOOLS),
    *_int_advanced_specs("Int1D", "1D", "1d", INT_1D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "2D Unit", ("Int2D", "unit"),
        ControlFieldKind.COMBO, "unit_2D", "current_text", "unit_2D",
        parameter_group="2d", tools=INT_2D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "2D Axis", ("Int2D", "axis"),
        ControlFieldKind.COMBO, "axis2D", "current_text", "axis2D",
        tools=INT_2D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "2D Radial Points", ("Int2D", "radial_points"),
        ControlFieldKind.LINE, "npts_radial_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "2D Azim Points", ("Int2D", "azim_points"),
        ControlFieldKind.LINE, "npts_azim_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "2D Radial Auto", ("Int2D", "radial_auto"),
        ControlFieldKind.BOOL, "radial_autoRange_2D", "checked",
        tools=INT_2D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "2D Radial Low", ("Int2D", "radial_low"),
        ControlFieldKind.LINE, "radial_low_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "2D Radial High", ("Int2D", "radial_high"),
        ControlFieldKind.LINE, "radial_high_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "2D Azim Auto", ("Int2D", "azim_auto"),
        ControlFieldKind.BOOL, "azim_autoRange_2D", "checked",
        tools=INT_2D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "2D Azim Low", ("Int2D", "azim_low"),
        ControlFieldKind.LINE, "azim_low_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    StaticWidgetBinding(
        SectionId.PROCESSING, "2D Azim High", ("Int2D", "azim_high"),
        ControlFieldKind.LINE, "azim_high_2D", "text",
        tools=INT_2D_OUTPUT_TOOLS),
    *_int_advanced_specs("Int2D", "2D", "2d", INT_2D_OUTPUT_TOOLS),
)

INTEGRATOR_BACKED_CONTROL_PATHS: tuple[tuple[str, ...], ...] = tuple(
    spec.path for spec in INTEGRATOR_BACKED_CONTROL_SPECS
)

INTEGRATION_CONTROL_SPECS: tuple[StaticWidgetBinding, ...] = tuple(
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
    ("BG", "Directory"),
    ("BG", "Match"),
    ("BG", "Metadata Key"),
    ("BG", "Filter"),
    ("BG", "Scale"),
    ("BG", "Normalize"),
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

def build_bound_control_state(
    values: Mapping[tuple[str, ...], object] | None = None,
    choices: Mapping[tuple[str, ...], Sequence[object]] | None = None,
    *,
    tool: Tool | None = None,
    controls_enabled: bool = True,
) -> BoundControlState:
    """Adapt static-scan widget values to the shared Controls renderer."""

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
    # both), hidden for Single Image and NeXus.  This matches the static-page
    # wrangler's set_inp_type, which reveals series_average for everything
    # except Single Image.
    if not nexus and source_type != "Single Image":
        add(SectionId.PROCESSING, "Average Scan", ("Signal", "series_average"),
            kind=ControlFieldKind.BOOL)
    bg_mode = str(values.get(("BG", "bg_type"), "None"))
    add(
        SectionId.PROCESSING,
        "Background",
        ("BG", "bg_type"),
        kind=ControlFieldKind.COMBO,
    )
    if bg_mode in {"Single BG File", "Series Average"}:
        label = "Source File" if bg_mode == "Single BG File" else "Series Member"
        add(SectionId.PROCESSING, label, ("BG", "File"), browse=True)
    elif bg_mode == "BG Directory":
        add(SectionId.PROCESSING, "Directory", ("BG", "Directory"), browse=True)
        add(SectionId.PROCESSING, "Match", ("BG", "Match"), kind=ControlFieldKind.COMBO)
        if str(values.get(("BG", "Match"), "")) == "Metadata Key":
            add(SectionId.PROCESSING, "Metadata Key", ("BG", "Metadata Key"))
        add(SectionId.PROCESSING, "Filename Filter", ("BG", "Filter"))
    if bg_mode != "None":
        add(SectionId.PROCESSING, "Scale", ("BG", "Scale"))
        add(SectionId.PROCESSING, "Normalize", ("BG", "Normalize"))

    return BoundControlState(fields=tuple(fields))

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
