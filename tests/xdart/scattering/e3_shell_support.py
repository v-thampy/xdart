from __future__ import annotations

import numpy as np

from xrd_tools.session.readiness import (
    ControlAction,
    ControlActionSpec,
    ControlPanelRenderState,
    ControlProfile,
    ProcessingPage,
    SectionId,
    Tool,
)

from xdart.gui.tabs.scattering.controls_inventory import (
    build_native_control_state,
)
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.controls_readiness import (
    ControlsReadinessProjection,
    SectionHeaderProjection,
)
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection,
    BrowserProjection,
    BrowserScan,
    FrameNavigationProjection,
    HeavyProjection,
    ProgressProjection,
    RunStripProjection,
    ScientificProjection,
    ShellPhase,
    ShellProjection,
    TraceProjection,
)


def make_shell_projection(
    *,
    revision: int = 1,
    frame_count: int = 5,
    phase: ShellPhase = ShellPhase.IDLE,
    selected_index: int = 0,
    heavy_indices: tuple[int, ...] = (0, 4),
    plot_mode: str = "Overlay",
    source_scan: str | None = None,
) -> ShellProjection:
    identity = RunIdentity(7, "e3-shell")
    frames = tuple(
        DisplayFrameKey(
            identity,
            source_scan or (
                "scan-a"
                if index < max(1, frame_count // 2)
                else "scan-b"
            ),
            "result.nxs",
            index % 3 + 1,
            index + 1,
        )
        for index in range(frame_count)
    )
    selected = frames[selected_index] if frames else None
    x = _frozen(np.linspace(0.1, 4.0, 64))
    q_axis = AxisProjection(x, "Q", "Å⁻¹")
    traces = tuple(
        TraceProjection(
            frame,
            q_axis,
            _frozen(np.sin(x * (index + 1) / 7.0) + index * 0.01),
            f"{frame.source_scan}:{frame.local_frame_label}",
        )
        for index, frame in enumerate(frames)
    )
    heavy_available = frozenset(
        frames[index]
        for index in heavy_indices
        if 0 <= index < len(frames)
    )
    heavy_frame = selected if selected in heavy_available else next(
        (frame for frame in frames if frame in heavy_available),
        None,
    )
    heavy = _heavy(heavy_frame) if heavy_frame is not None else None
    controls = _control_state(phase)
    active = phase in {
        ShellPhase.PREPARING,
        ShellPhase.RUNNING,
        ShellPhase.PAUSING,
        ShellPhase.PAUSED,
        ShellPhase.STOPPING,
    }
    return ShellProjection(
        revision,
        BrowserProjection(
            "/data/processed",
            (
                BrowserScan("scan-a", "scan-a", "first scan"),
                BrowserScan("scan-b", "scan-b", "second scan"),
            ),
            "scan-a",
            False,
            True,
            frames,
        ),
        ScientificProjection(
            heavy_available=heavy_available,
            traces=traces,
            heavy=heavy,
            title=(
                "Current"
                if selected is None
                else f"{selected.source_scan}:{selected.local_frame_label}"
            ),
            norm_channels=("Norm Channel", "Monitor", "I0", "I1"),
            norm_channel="Monitor",
            color_maps=("Default", "viridis", "magma", "inferno"),
            color_map="viridis",
            plot_mode=plot_mode,
            status="Ready",
        ),
        FrameNavigationProjection(
            frames,
            selected,
            (
                ()
                if selected is None
                else ((selected,) if plot_mode == "Single" else frames)
            ),
        ),
        controls,
        RunStripProjection(
            phase=phase,
            modes=("Int 1D", "Int 2D", "Int 1D (XYE)"),
            mode="Int 2D",
            batch=False,
            cores=4,
            max_cores=8,
            live=True,
            output_policy="Append",
            readiness="Ready - Int 2D",
            ready=True,
            run_enabled=not active,
            stop_enabled=active,
        ),
        ProgressProjection(
            completed=min(frame_count, max(0, selected_index + 1)),
            total=frame_count,
            detail="Processing" if active else "Ready",
        ),
        controls_readiness=ControlsReadinessProjection(
            processing=SectionHeaderProjection(
                "int 2d",
                True,
                "Typed processing inputs are configured.",
            ),
        ),
    )


def _heavy(frame: DisplayFrameKey) -> HeavyProjection:
    raw = _frozen(
        np.arange(48 * 64, dtype=float).reshape(48, 64)
        + frame.work_ordinal
    )
    cake = _frozen(
        np.arange(24 * 32, dtype=float).reshape(24, 32)
        + frame.work_ordinal
    )
    q = AxisProjection(_frozen(np.linspace(0.1, 3.2, 32)), "Q", "Å⁻¹")
    chi = AxisProjection(_frozen(np.linspace(-90.0, 90.0, 24)), "χ", "°")
    return HeavyProjection(frame, raw, cake, q, chi)


def _control_state(phase: ShellPhase) -> ControlPanelRenderState:
    unlocked = phase in {ShellPhase.IDLE, ShellPhase.FAILED}
    values = {
        ("Project", "project_folder"): "/project",
        ("Project", "h5_dir"): "/project/processed",
        ("Signal", "inp_type"): "Image Directory",
        ("Signal", "img_dir"): "/data/raw",
        ("Signal", "img_ext"): "tif",
        ("Signal", "include_subdir"): True,
        ("Signal", "Filter"): "",
        ("Signal", "meta_ext"): "None",
        ("Signal", "series_average"): False,
        ("Source", "energy_preference"): "poni",
        ("Signal", "poni_file"): "/data/calibration/detector.poni",
        ("Signal", "mask_file"): "/data/masks/mask.edf",
        ("GI", "Grazing"): False,
        ("GI", "th_motor"): "Manual",
        ("GI", "th_val"): "0.1",
        ("Mask", "Threshold"): False,
        ("Mask", "min"): "0",
        ("Mask", "max"): "100000",
        ("MaskSat", "mask_sentinel"): True,
        ("Int1D", "unit"): "q_A^-1",
        ("Int1D", "axis"): "Q (Å⁻¹)",
        ("Int1D", "points"): "3000",
        ("Int1D", "radial_auto"): False,
        ("Int1D", "radial_low"): "0.1",
        ("Int1D", "radial_high"): "4.0",
        ("Int1D", "azim_auto"): False,
        ("Int1D", "azim_low"): "-180",
        ("Int1D", "azim_high"): "180",
        ("Int2D", "unit"): "q_A^-1",
        ("Int2D", "axis"): "Q-χ",
        ("Int2D", "radial_points"): "500",
        ("Int2D", "azim_points"): "500",
        ("Int2D", "radial_auto"): False,
        ("Int2D", "radial_low"): "0.1",
        ("Int2D", "radial_high"): "4.0",
        ("Int2D", "azim_auto"): False,
        ("Int2D", "azim_low"): "-180",
        ("Int2D", "azim_high"): "180",
        ("BG", "bg_type"): "None",
    }
    choices = {
        ("Signal", "inp_type"): (
            "Image Series",
            "Image Directory",
            "Single Image",
        ),
        ("Signal", "img_ext"): ("tif", "cbf", "edf", "h5"),
        ("Signal", "meta_ext"): ("None", "SPEC", "PDI"),
        ("Source", "energy_preference"): ("poni", "metadata"),
        ("GI", "th_motor"): ("Manual", "theta"),
        ("Int1D", "unit"): ("q_A^-1", "2th_deg", "chi_deg"),
        ("Int1D", "axis"): ("Q (Å⁻¹)", "2θ (°)", "χ (°)"),
        ("Int2D", "unit"): ("q_A^-1", "2th_deg"),
        ("Int2D", "axis"): ("Q-χ", "2θ-χ"),
        ("BG", "bg_type"): ("None", "File"),
    }
    fields = build_native_control_state(
        values,
        choices,
        tool=Tool.INT_2D,
        controls_enabled=unlocked,
    )
    actions = {
        SectionId.EXPERIMENT: (
            ControlActionSpec(ControlAction.CALIBRATE, "Calibrate", SectionId.EXPERIMENT, unlocked),
            ControlActionSpec(ControlAction.MAKE_MASK, "Make Mask", SectionId.EXPERIMENT, unlocked),
        ),
        SectionId.PROCESSING: (
            ControlActionSpec(ControlAction.REINTEGRATE_1D, "Reintegrate 1D", SectionId.PROCESSING, unlocked),
            ControlActionSpec(ControlAction.REINTEGRATE_2D, "Reintegrate 2D", SectionId.PROCESSING, unlocked),
            ControlActionSpec(ControlAction.ADVANCED_PROCESSING, "Advanced", SectionId.PROCESSING, unlocked),
        ),
    }
    profile = ControlProfile(
        ProcessingPage.INT_2D,
        unlocked,
        () if unlocked else ("Run active",),
        section_actions=actions,
        detector_summary="Pilatus - Standard",
    )
    return ControlPanelRenderState(profile, fields)


def _frozen(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value
