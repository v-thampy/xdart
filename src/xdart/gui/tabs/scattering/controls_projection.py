"""Production composition boundary for the mounted vNext Controls panel."""

from __future__ import annotations

from dataclasses import replace
from xrd_tools.session.intent_store import RunIntentSnapshot
from xrd_tools.session.readiness import (
    BoundControlState,
    ControlAction,
    ControlActionSpec,
    ControlFieldKind,
    ControlPanelRenderState,
    ControlProfile,
    ProcessingPage,
    SectionId,
    Tool,
    build_bound_control_state,
)

from .contracts import SourceObservation
from .controls_editing import (
    AdvancedSettingsValues,
    EditNoChange,
    EditRefusal,
    EditResult,
    advanced_settings_values,
    reduce_advanced_settings,
    reduce_control_edit,
    reduce_source_selection,
)
from .controls_inventory import (
    AVERAGE_SCAN,
    BACKGROUND_TYPE,
    GI_ENABLED,
    GI_MOTOR,
    GI_ORIENTATION,
    GI_THETA,
    GI_TILT,
    INT_1D_AXIS,
    INT_1D_AZIM_AUTO,
    INT_1D_AZIM_HIGH,
    INT_1D_AZIM_LOW,
    INT_1D_POINTS,
    INT_1D_RADIAL_AUTO,
    INT_1D_RADIAL_HIGH,
    INT_1D_RADIAL_LOW,
    INT_2D_AXIS,
    INT_2D_AZIM_AUTO,
    INT_2D_AZIM_HIGH,
    INT_2D_AZIM_LOW,
    INT_2D_AZIM_POINTS,
    INT_2D_RADIAL_AUTO,
    INT_2D_RADIAL_HIGH,
    INT_2D_RADIAL_LOW,
    INT_2D_RADIAL_POINTS,
    MASK_FILE,
    MASK_SATURATION,
    OUTPUT_MODE,
    PONI_FILE,
    PROJECT_ROOT,
    SAVE_PATH,
    SOURCE_DIRECTORY,
    SOURCE_ENERGY,
    SOURCE_FILE,
    SOURCE_FILTER,
    SOURCE_META,
    SOURCE_RECURSIVE,
    SOURCE_SUFFIX,
    SOURCE_TYPE,
    THRESHOLD_ENABLED,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    bound_values,
    field,
    source_name,
    source_mode,
    truthful_field,
)
from .detector_projection import detector_summary, poni_saturation_ceiling
from .state_machine import RunPhase


def project_controls(
    snapshot: RunIntentSnapshot,
    observation: SourceObservation | None,
    phase: RunPhase,
    *,
    advanced_editor_available: bool = False,
    source_mode_override: str | None = None,
    detector_summary_override: str | None = None,
) -> ControlPanelRenderState:
    intent = snapshot.thaw()
    raw_motor = intent.gi.incidence_motor
    motor_value = raw_motor
    motor_reason = ""
    motor_choices = ("Manual",) if raw_motor == "Manual" else ("Manual", raw_motor)
    if (
        observation is not None
        and observation.source == intent.source_spec
        and observation.gi_motor_choices is not None
    ):
        source_choices = observation.gi_motor_choices
        motor_choices = tuple(dict.fromkeys(("Manual", *source_choices)))
        if raw_motor != "Manual" and raw_motor not in source_choices:
            motor_reason = (
                f"Selected motor '{raw_motor}' is unavailable in this "
                "preview; choose an available motor or Manual. Run "
                "admission will recheck every admitted frame."
            )
    values, choices = bound_values(
        intent,
        motor_value=motor_value,
        motor_choices=motor_choices,
        source_mode=source_mode_override,
    )
    unlocked = phase in {RunPhase.IDLE, RunPhase.FAILED}
    processing_mode = str(intent.processing_mode or "")
    skip_2d = (
        "Viewer" not in processing_mode
        and "1D" in processing_mode
        and "2D" not in processing_mode
    )
    tool = Tool.INT_1D if skip_2d else Tool.INT_2D
    projected = build_bound_control_state(
        values,
        choices,
        tool=tool,
        controls_enabled=unlocked,
    )
    output_choices = ("Overwrite", "Append")
    output_reason = (
        ""
        if unlocked
        else "Controls are locked during the active run."
    )
    fields = list(projected.fields)
    gi_paths = {GI_MOTOR, GI_THETA, GI_ORIENTATION, GI_TILT}
    fields = [
        candidate for candidate in fields
        if candidate.path not in gi_paths and candidate.path != SOURCE_ENERGY
    ]
    gi_fields = ()
    if intent.gi.enabled:
        gi_fields = (
            field(
                SectionId.EXPERIMENT,
                "Incidence motor",
                GI_MOTOR,
                motor_value,
                kind=ControlFieldKind.COMBO,
                choices=motor_choices,
                reason=motor_reason,
            ),
        )
        if motor_value == "Manual":
            gi_fields += (
                field(
                    SectionId.EXPERIMENT,
                    "Incidence angle",
                    GI_THETA,
                    intent.gi.th_val,
                ),
            )
        gi_fields += (
            field(
                SectionId.EXPERIMENT,
                "Sample orientation",
                GI_ORIENTATION,
                intent.gi.sample_orientation,
            ),
            field(
                SectionId.EXPERIMENT,
                "Tilt angle",
                GI_TILT,
                intent.gi.tilt_angle,
            ),
        )
    if not unlocked:
        gi_fields = tuple(
            replace(
                candidate,
                enabled=False,
                reason="Controls are locked during the active run.",
            )
            for candidate in gi_fields
        )
    gi_insertion = next(
        (
            index + 1
            for index, candidate in enumerate(fields)
            if candidate.path == GI_ENABLED
        ),
        len(fields),
    )
    fields[gi_insertion:gi_insertion] = gi_fields
    output = field(
        SectionId.PROJECT,
        "Output mode",
        OUTPUT_MODE,
        intent.output_mode,
        kind=ControlFieldKind.COMBO,
        choices=output_choices,
        reason=output_reason,
    )
    output = replace(output, enabled=unlocked)
    insertion = next(
        (
            index + 1
            for index, candidate in enumerate(fields)
            if candidate.path == SAVE_PATH
        ),
        2,
    )
    fields.insert(insertion, output)
    fields = _threshold_auto_fields(fields, intent, unlocked=unlocked)
    fields = [truthful_field(candidate) for candidate in fields]
    reintegration_unavailable = (
        "Loaded-result reintegration is not available in this workspace yet."
    )
    operation_unavailable = "No vNext operation service is mounted."
    actions = {
        SectionId.EXPERIMENT: (
            ControlActionSpec(
                ControlAction.CALIBRATE,
                "Calibrate",
                SectionId.EXPERIMENT,
                False,
                operation_unavailable,
                False,
            ),
            ControlActionSpec(
                ControlAction.MAKE_MASK,
                "Make Mask",
                SectionId.EXPERIMENT,
                False,
                operation_unavailable,
                False,
            ),
        ),
        SectionId.PROCESSING: (
            ControlActionSpec(
                ControlAction.REINTEGRATE_1D,
                "Reintegrate 1D",
                SectionId.PROCESSING,
                False,
                reintegration_unavailable,
                False,
            ),
            ControlActionSpec(
                ControlAction.REINTEGRATE_2D,
                "Reintegrate 2D",
                SectionId.PROCESSING,
                False,
                reintegration_unavailable,
                False,
            ),
            ControlActionSpec(
                ControlAction.ADVANCED_PROCESSING,
                "Advanced",
                SectionId.PROCESSING,
                advanced_editor_available and unlocked,
                (
                    "Open advanced integration settings."
                    if advanced_editor_available and unlocked
                    else "Controls are locked during the active run."
                    if not unlocked
                    else "Advanced settings require a native vNext editor."
                ),
                True,
            ),
        )
    }
    profile = ControlProfile(
        processing_page=(
            ProcessingPage.INT_1D
            if tool is Tool.INT_1D
            else ProcessingPage.INT_2D
        ),
        run_enabled=False,
        run_blockers=("Execution is introduced in E1b.",),
        section_actions=actions,
        detector_summary=(
            detector_summary(
                intent.poni_file,
                intent.mask_file,
            )
            if detector_summary_override is None
            else detector_summary_override
        ),
    )
    return ControlPanelRenderState(
        profile,
        BoundControlState(tuple(fields)),
    )


#: Detector-scope hover guard: rides every disabled reason for the manual
#: max bound AND its enabled-state tooltip, so it is visible whichever state
#: the row renders in.
_MAX_BOUND_SCOPE_CAVEAT = (
    " The default max is the detector family's typical raw-stream ceiling "
    "— a display default only; masking follows the acquired frame's own "
    "data type."
)


def _threshold_auto_fields(
    fields: list,
    intent,
    *,
    unlocked: bool,
) -> list:
    """LV-UI-11: one Auto control for saturated-pixel masking vs manual band.

    The vNext Threshold row's Auto toggle IS the Mask-Saturated fact, so the
    separate True=apply Threshold field is not rendered here (the legacy
    static_scan panel keeps it).  The manual min/max bounds are editable
    exactly when Auto is off, and display the [0, detector-ceiling] defaults
    when the intent carries none — the ceiling blank until a valid PONI names
    a known detector family."""
    auto_on = bool(intent.threshold.mask_saturation)
    out = []
    for candidate in fields:
        if candidate.path == THRESHOLD_ENABLED:
            continue
        if candidate.path in {THRESHOLD_MIN, THRESHOLD_MAX}:
            value = candidate.value
            if value is None:
                value = (
                    0.0
                    if candidate.path == THRESHOLD_MIN
                    else poni_saturation_ceiling(intent.poni_file)
                )
            # The disabled reason takes tooltip precedence, so the max
            # bound's detector-scope caveat must ride EVERY disabled reason
            # — Auto-on AND run-locked — or the hover guard silently
            # disappears in that state (review P2 + DESIGN_STOP secondary,
            # 2026-08-04).
            caveat = (
                _MAX_BOUND_SCOPE_CAVEAT
                if candidate.path == THRESHOLD_MAX
                else ""
            )
            reason = (
                "Controls are locked during the active run." + caveat
                if not unlocked
                else "Auto masks saturated pixels; turn it off to set "
                "a manual threshold band." + caveat
                if auto_on
                else ""
            )
            out.append(replace(
                candidate,
                value=value,
                enabled=unlocked and not auto_on,
                reason=reason,
            ))
            continue
        out.append(candidate)
    return out


__all__ = [
    "AVERAGE_SCAN",
    "AdvancedSettingsValues",
    "BACKGROUND_TYPE",
    "EditNoChange",
    "EditRefusal",
    "EditResult",
    "GI_ENABLED",
    "GI_MOTOR",
    "GI_ORIENTATION",
    "GI_THETA",
    "GI_TILT",
    "INT_1D_AXIS",
    "INT_1D_POINTS",
    "INT_2D_AXIS",
    "INT_2D_RADIAL_POINTS",
    "INT_2D_AZIM_POINTS",
    "MASK_FILE",
    "MASK_SATURATION",
    "OUTPUT_MODE",
    "PONI_FILE",
    "PROJECT_ROOT",
    "SAVE_PATH",
    "SOURCE_DIRECTORY",
    "SOURCE_ENERGY",
    "SOURCE_FILE",
    "SOURCE_FILTER",
    "SOURCE_META",
    "SOURCE_RECURSIVE",
    "SOURCE_SUFFIX",
    "SOURCE_TYPE",
    "THRESHOLD_ENABLED",
    "THRESHOLD_MAX",
    "THRESHOLD_MIN",
    "advanced_settings_values",
    "project_controls",
    "reduce_advanced_settings",
    "reduce_control_edit",
    "reduce_source_selection",
    "source_mode",
]
