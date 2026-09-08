"""Production composition boundary for the mounted vNext Controls panel."""

from __future__ import annotations

from dataclasses import replace

from xrd_tools.session.intent_store import RunIntentSnapshot
from xrd_tools.session.readiness import (
    ControlAction,
    ControlActionSpec,
    ControlFieldKind,
    ControlsProjection,
    ProcessingPage,
    SectionId,
    Tool,
    tool_from_mode_text,
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
    BACKGROUND_DIRECTORY,
    BACKGROUND_FILE,
    BACKGROUND_FILTER,
    BACKGROUND_MATCH,
    BACKGROUND_METADATA_KEY,
    BACKGROUND_NORMALIZE,
    BACKGROUND_SCALE,
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
    project_control_fields,
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
    calibrate_available: bool = False,
    calibrate_dependency_available: bool = False,
    operation_busy: bool = False,
    calibration_active: bool = False,
    mask_available: bool = False,
    mask_dependency_available: bool = False,
    mask_active: bool = False,
    reintegrate_available: bool = False,
    reintegrate_active: bool = False,
    reintegrate_dimension: str | None = None,
    reintegrate_stop_accepted: bool = False,
) -> ControlsProjection:
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
    unlocked = phase in {RunPhase.IDLE, RunPhase.FAILED} and not operation_busy
    processing_mode = str(intent.processing_mode or "")
    tool = tool_from_mode_text(processing_mode)
    viewer = tool in {Tool.IMAGE_VIEWER, Tool.XYE_VIEWER}
    projected = project_control_fields(
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
    fields = list(projected)
    gi_paths = {GI_MOTOR, GI_THETA, GI_ORIENTATION, GI_TILT}
    fields = [
        candidate for candidate in fields
        if candidate.path not in gi_paths and candidate.path != SOURCE_ENERGY
        and (
            candidate.path != AVERAGE_SCAN
            or (
                tool in {Tool.INT_1D, Tool.INT_2D}
                and processing_mode != "Int 1D (XYE)"
            )
        )
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
    output = replace(output, enabled=unlocked and not viewer)
    insertion = next(
        (
            index + 1
            for index, candidate in enumerate(fields)
            if candidate.path == SAVE_PATH
        ),
        2,
    )
    fields.insert(insertion, output)
    fields = _threshold_fields(fields, intent, unlocked=unlocked)
    fields = [truthful_field(candidate) for candidate in fields]
    if viewer:
        fields = [candidate if candidate.path == PROJECT_ROOT else
                  replace(candidate, enabled=False,
                          reason=("1D Viewer" if tool is Tool.XYE_VIEWER
                                  else "2D Viewer") + " has no acquisition authority.")
                  for candidate in fields]
    reintegrate_1d_active = reintegrate_active and reintegrate_dimension == "1d"; reintegrate_2d_active = reintegrate_active and reintegrate_dimension == "2d"
    reintegrate_1d_enabled = (not reintegrate_stop_accepted if reintegrate_1d_active else reintegrate_available and unlocked); reintegrate_2d_enabled = (not reintegrate_stop_accepted if reintegrate_2d_active else reintegrate_available and unlocked)
    def reintegrate_reason(label, active, enabled):
        return f"Stopping the active Reintegrate {label} operation." if active and reintegrate_stop_accepted else f"Stop the active Reintegrate {label} operation." if active else f"Creates a new immutable {label} version using current {label} integration settings and core request; the selected artifact remains unchanged. Calibration, mask, threshold, Background, geometry, and shared GI facts come from the loaded artifact, not current shared-science controls." if enabled else "Another experiment operation owns the common slot." if operation_busy else "Controls are locked during the active run." if not unlocked else f"Load one stable processed Browse artifact to Reintegrate {label}."
    reintegration_reason = reintegrate_reason("1-D", reintegrate_1d_active, reintegrate_1d_enabled); reintegration_2d_reason = reintegrate_reason("2-D", reintegrate_2d_active, reintegrate_2d_enabled)
    operation_unavailable = "No vNext operation service is mounted."
    calibrate_enabled = calibration_active or (
        calibrate_available and unlocked
    )
    calibrate_reason = (
        "Cancel the standalone PONI calibration."
        if calibration_active
        else "Launch standalone pyFAI calibration to create a PONI file."
        if calibrate_enabled
        else "Another experiment operation owns the common slot."
        if operation_busy
        else "Controls are locked during the active run."
        if phase not in {RunPhase.IDLE, RunPhase.FAILED}
        else "pyFAI-calib2 is unavailable on PATH."
        if not calibrate_dependency_available
        else "Calibration is unavailable in the current workspace state."
    )
    mask_enabled = mask_active or (mask_available and unlocked)
    mask_reason = (
        "Cancel the standalone mask operation."
        if mask_active
        else (
            "Choose an explicit TIFF, HDF5, or NeXus image source and create "
            "its beside-source EDF mask."
        )
        if mask_enabled
        else "Another experiment operation owns the common slot."
        if operation_busy
        else "Controls are locked during the active run."
        if phase not in {RunPhase.IDLE, RunPhase.FAILED}
        else "pyFAI-drawmask is unavailable on PATH."
        if not mask_dependency_available
        else "Mask creation is unavailable in the current workspace state."
    )
    actions = {
        SectionId.EXPERIMENT: (
            ControlActionSpec(
                ControlAction.CALIBRATE,
                "Cancel Calibration" if calibration_active else "Calibrate",
                SectionId.EXPERIMENT,
                calibrate_enabled,
                calibrate_reason,
                True,
            ),
            ControlActionSpec(
                ControlAction.MAKE_MASK,
                "Cancel Mask" if mask_active else "Make Mask",
                SectionId.EXPERIMENT,
                mask_enabled,
                mask_reason,
                True,
            ),
        ),
        SectionId.PROCESSING: (
            ControlActionSpec(
                ControlAction.REINTEGRATE_1D,
                "Stopping Reintegrate 1-D" if reintegrate_1d_active and reintegrate_stop_accepted else "Stop Reintegrate 1-D" if reintegrate_1d_active else "Reintegrate 1-D",
                SectionId.PROCESSING,
                reintegrate_1d_enabled,
                reintegration_reason,
                True,
            ),
            ControlActionSpec(
                ControlAction.REINTEGRATE_2D,
                "Stopping Reintegrate 2-D" if reintegrate_2d_active and reintegrate_stop_accepted else "Stop Reintegrate 2-D" if reintegrate_2d_active else "Reintegrate 2-D",
                SectionId.PROCESSING,
                reintegrate_2d_enabled,
                reintegration_2d_reason,
                True,
            ),
            ControlActionSpec(
                ControlAction.ADVANCED_PROCESSING,
                "Advanced",
                SectionId.PROCESSING,
                advanced_editor_available and unlocked and not viewer,
                (
                    "Open advanced integration settings."
                    if advanced_editor_available and unlocked and not viewer
                    else "Controls are locked during the active run."
                    if not unlocked
                    else operation_unavailable
                ),
                True,
            ),
        )
    }
    return ControlsProjection(
        processing_page=(
            ProcessingPage.VIEWER
            if tool in {Tool.IMAGE_VIEWER, Tool.XYE_VIEWER}
            else ProcessingPage.INT_1D
            if tool is Tool.INT_1D
            else ProcessingPage.INT_2D
        ),
        fields=tuple(fields),
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


#: Detector-scope hover guard: rides every disabled reason for the manual
#: max bound AND its enabled-state tooltip, so it is visible whichever state
#: the row renders in.
_MAX_BOUND_SCOPE_CAVEAT = (
    " The default max is the detector family's typical raw-stream ceiling "
    "— a display default only; masking follows the acquired frame's own "
    "data type."
)


def _threshold_fields(
    fields: list,
    intent,
    *,
    unlocked: bool,
) -> list:
    """Project independent manual-threshold and saturated-mask controls.

    The Threshold row's compact toggle owns only ``apply_threshold``.  The
    separate Mask Saturated pill owns only ``mask_saturation``.  Manual bounds
    are editable exactly while manual thresholding is enabled, and display the
    [0, detector-ceiling] defaults when the intent carries none — the ceiling
    stays blank until a valid PONI names a known detector family.
    """
    manual_on = bool(intent.threshold.apply_threshold)
    out = []
    for candidate in fields:
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
                else "Enable Manual Threshold to edit this bound." + caveat
                if not manual_on
                else ""
            )
            out.append(replace(
                candidate,
                value=value,
                enabled=unlocked and manual_on,
                reason=reason,
            ))
            continue
        out.append(candidate)
    return out


__all__ = [
    "AVERAGE_SCAN",
    "BACKGROUND_DIRECTORY",
    "BACKGROUND_FILE",
    "BACKGROUND_FILTER",
    "BACKGROUND_MATCH",
    "BACKGROUND_METADATA_KEY",
    "BACKGROUND_NORMALIZE",
    "BACKGROUND_SCALE",
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
