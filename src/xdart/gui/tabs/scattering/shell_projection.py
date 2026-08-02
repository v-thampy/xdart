"""Stateless conversion from exact context values to one shell projection."""

from __future__ import annotations

from dataclasses import dataclass
import os

from xdart.modules.display_context import (
    AcquisitionContext,
    BrowseContext,
    DisplaySelection,
)
from .browser_catalog import BrowserCatalogEntry, natural_name_key
from .display_values import (
    DisplayFrameKey,
    StandardDisplayPayload,
    display_payload_is_valid,
)
from .run_mode_projection import (
    RUN_MODE_CHOICES,
    UNOWNED_RUN_MODE_REASONS,
    build_run_strip_projection,
)
from .shell_values import (
    BrowserProjection,
    BrowserScan,
    FrameNavigationProjection,
    ScientificPlotOptions,
    ScientificProjection,
)
from .scientific_axes import (
    heavy_projection,
    image_axis_choice,
    plot_axis_choice,
    requested_image_axis as project_requested_image_axis,
    share_plot_axis_for_image,
    trace_projection,
)
from .state_machine import RunPhase


@dataclass(frozen=True, slots=True)
class ScientificPreferences:
    norm_channel: str = "Norm Channel"
    color_map: str = "Default"
    log_scale: bool = False
    image_axis: str = "Q-Chi"
    plot_axis: str = "Q"
    plot_mode: str = "Single"
    share_axis: bool = False
    slice_enabled: bool = False
    slice_center: float = 0.0
    slice_width: float = 10.0
    q_range: tuple[float, float] = (0.0, 10.0)
    chi_range: tuple[float, float] = (-180.0, 180.0)
    plot_options: ScientificPlotOptions = ScientificPlotOptions()
    background_set: bool = False


def build_browser_projection(
    *,
    contexts: tuple[AcquisitionContext | BrowseContext, ...],
    selection: DisplaySelection | None,
    navigation: FrameNavigationProjection,
    browser_directory: str,
    date_sorted: bool,
    auto_last: bool,
    catalog: tuple[BrowserCatalogEntry, ...] = (),
    transient_frame: DisplayFrameKey | None = None,
) -> BrowserProjection:
    if (
        transient_frame is not None
        and type(transient_frame) is not DisplayFrameKey
    ):
        raise TypeError("transient browser frame must be exact")
    transient_owner = (
        transient_frame
        if transient_frame is not None
        and any(frame is transient_frame for frame in navigation.frames)
        else None
    )
    scans: list[BrowserScan] = []
    seen_artifacts: set[str] = set()
    entries = (
        tuple(
            sorted(
                catalog,
                key=lambda entry: (
                    0 if entry.label == ".." else 1,
                    0 if entry.label == ".." else -entry.modified_ns,
                    natural_name_key(
                        entry.label.removesuffix("/")
                        if entry.is_directory
                        else entry.label
                    ),
                ),
            )
        )
        if date_sorted
        else catalog
    )
    for entry in entries:
        seen_artifacts.add(entry.artifact)
        scans.append(
            BrowserScan(
                entry.artifact,
                entry.label,
                entry.artifact,
            )
        )
    for frame in navigation.frames:
        if (
            frame.artifact in seen_artifacts
            or transient_owner is None
            or frame.run_identity is not transient_owner.run_identity
            or frame.artifact != transient_owner.artifact
            or (
                browser_directory
                and os.path.abspath(os.path.dirname(frame.artifact))
                != os.path.abspath(browser_directory)
            )
        ):
            continue
        seen_artifacts.add(frame.artifact)
        scans.append(
            BrowserScan(
                frame.artifact,
                os.path.basename(frame.artifact) or frame.source_scan,
                frame.artifact,
            )
        )
    # Contexts and navigation are display-retention owners, not evidence that
    # an artifact still exists.  Only the exact latest in-flight artifact may
    # lend a transient row while NexusSink is publishing its hidden temp file.
    # Earlier artifacts from the same directory run stay catalog-owned.
    selected_scan = ""
    if (
        navigation.current is not None
        and navigation.current.artifact in seen_artifacts
    ):
        selected_scan = navigation.current.artifact
    elif (
        selection is not None
        and any(
            scan.identifier == selection.context_token
            for scan in scans
        )
    ):
        selected_scan = selection.context_token
    frames = tuple(
        frame
        for frame in navigation.frames
        if frame.artifact == selected_scan
    )
    return BrowserProjection(
        directory=browser_directory,
        scans=tuple(scans),
        selected_scan=selected_scan,
        date_sorted=date_sorted,
        auto_last=auto_last,
        frames=frames,
    )


def build_scientific_projection(
    payloads: tuple[StandardDisplayPayload, ...],
    navigation: FrameNavigationProjection,
    resident_frames: frozenset[DisplayFrameKey],
    preferences: ScientificPreferences,
    notice: str,
    phase: RunPhase | None = None,
    *,
    processing_mode: str = "Int 2D",
) -> ScientificProjection:
    frame_by_id = {id(frame): frame for frame in navigation.frames}
    selected_ids = {id(frame) for frame in navigation.selected}
    current_id = (
        None if navigation.current is None else id(navigation.current)
    )
    eligible_ids = set(selected_ids)
    if current_id is not None:
        eligible_ids.add(current_id)
    accepted_items: list[StandardDisplayPayload] = []
    payload_by_id: dict[int, StandardDisplayPayload] = {}
    for payload in payloads:
        if type(payload) is not StandardDisplayPayload:
            continue
        frame = payload.frame_key
        frame_id = id(frame)
        if (
            type(frame) is not DisplayFrameKey
            or frame_by_id.get(frame_id) is not frame
            or frame_id not in eligible_ids
        ):
            continue
        try:
            valid = display_payload_is_valid(
                payload,
                frame.run_identity,
                frame,
                payload.selection_generation,
            )
        except Exception:
            valid = False
        if not valid:
            continue
        accepted_items.append(payload)
        payload_by_id.setdefault(frame_id, payload)
    accepted = tuple(accepted_items)
    current_payload = (
        None
        if current_id is None
        else payload_by_id.get(current_id)
    )
    resident_ids = {id(frame) for frame in resident_frames}
    current_resident = (
        navigation.current is not None
        and current_id in resident_ids
    )
    identity_payload = current_payload or next(iter(accepted), None)
    measurement_mode = (
        "Standard"
        if identity_payload is None
        else identity_payload.measurement_mode
    )
    gi_mode_1d = (
        ""
        if identity_payload is None
        else identity_payload.gi_mode_1d
    )
    gi_mode_2d = (
        ""
        if identity_payload is None
        else identity_payload.gi_mode_2d
    )
    requested_image_axis = (
        preferences.image_axis
        if identity_payload is None
        else project_requested_image_axis(
            identity_payload,
            preferences.image_axis,
        )
    )
    heavy = (
        None
        if current_payload is None or not current_resident
        else heavy_projection(
            current_payload,
            requested_axis=project_requested_image_axis(
                current_payload,
                preferences.image_axis,
            ),
        )
    )
    rendered_image_axis = (
        requested_image_axis
        if heavy is None
        else image_axis_choice(
            heavy.cake_x,
            current_payload.view.two_d_kind,
            measurement_mode=measurement_mode,
            gi_mode_2d=gi_mode_2d,
        )
    )
    requested_plot_axis = preferences.plot_axis
    if preferences.share_axis:
        requested_plot_axis = (
            share_plot_axis_for_image(rendered_image_axis)
            or requested_plot_axis
        )
    traces = tuple(
        trace
        for payload in accepted
        if id(payload.frame_key) in selected_ids
        if (trace := trace_projection(
            payload,
            requested_axis=requested_plot_axis,
            allow_cake=processing_mode != "Int 1D",
            slice_enabled=preferences.slice_enabled,
            slice_center=preferences.slice_center,
            slice_width=preferences.slice_width,
        )) is not None
    )
    rendered_plot_axis = plot_axis_choice(
        traces,
        requested_plot_axis,
    )
    title = "Current"
    status = notice
    if current_payload is not None:
        title = _source_member_label(current_payload)
        status = notice or title
    # Payload projection and residency are two bounded reads.  A live commit
    # can land between them, making the current frame resident in the second
    # read while its payload was absent from the first.  Treat that split-view
    # state as pending and retain the last coherent three-panel presentation.
    retain = navigation.current is not None and (
        current_payload is None or not current_resident
    )
    return ScientificProjection(
        heavy_available=resident_frames,
        traces=traces,
        heavy=heavy,
        title=title,
        processing_mode=processing_mode,
        measurement_mode=measurement_mode,
        gi_mode_1d=gi_mode_1d,
        gi_mode_2d=gi_mode_2d,
        norm_channel=preferences.norm_channel,
        color_map=preferences.color_map,
        log_scale=preferences.log_scale,
        image_axis=rendered_image_axis,
        plot_axis=rendered_plot_axis,
        plot_mode=preferences.plot_mode,
        share_axis=preferences.share_axis,
        slice_enabled=preferences.slice_enabled,
        slice_center=preferences.slice_center,
        slice_width=preferences.slice_width,
        q_range=preferences.q_range,
        chi_range=preferences.chi_range,
        plot_options=preferences.plot_options,
        background_set=preferences.background_set,
        status=status or ("Loading frame…" if retain else ""),
        retain_display=retain,
        live_update=phase in {
            RunPhase.STARTING,
            RunPhase.RUNNING,
            RunPhase.PAUSING,
            RunPhase.PAUSED,
            RunPhase.RESUMING,
            RunPhase.STOPPING,
        },
    )


def _source_member_label(payload: StandardDisplayPayload) -> str:
    """Name the exact raw member represented by the displayed payload."""

    source = payload.view.source_path
    name = "" if not source else os.path.basename(str(source))
    if not name:
        return payload.title
    suffix = os.path.splitext(name)[1].casefold()
    if suffix not in {".h5", ".hdf5", ".nxs"}:
        return name
    source_frame = payload.view.source_frame_index
    label = (
        payload.frame_key.local_frame_label
        if source_frame is None
        else source_frame + 1
    )
    return f"{name} · frame {label}"


__all__ = [
    "RUN_MODE_CHOICES",
    "ScientificPreferences",
    "UNOWNED_RUN_MODE_REASONS",
    "build_browser_projection",
    "build_run_strip_projection",
    "build_scientific_projection",
    "share_plot_axis_for_image",
]
