"""Context-qualified, no-I/O display projection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
import numpy as np

from xdart.modules.display_context import (
    AcquisitionContext, BrowseContext, ContextKind, DisplaySelection,
    HydrationRequest, Viewer1DContext, Viewer1DState,
    Viewer2DContext, Viewer2DState,
)
from xdart.modules.frame_publication import PublicationStore
from xrd_tools.core import Axis, FrameView
from xrd_tools.core.angular_support import AngularSupportProfile, SupportOrigin, SupportReason
from xrd_tools.io.viewer_2d import (
    Viewer2DArtifactCatalog, Viewer2DFrame, Viewer2DSourceKind,
)
from xrd_tools.core.energy import WavelengthUnit, canonical_wavelength_m
from xrd_tools.session.hydration import (
    HydrationPurpose, HydrationReadKey, HydrationScope, HydrationToken,
)
from xrd_tools.session.readiness import ControlsProjection, Tool, tool_from_mode_text
from xrd_tools.session.run_configuration import RunIntent

from .browser_catalog import BrowserCatalogEntry
from .context_values import (
    BrowseMissReason,
    BrowseProjectionResolution,
    HydrationEligibleMiss,
    QualifiedPayload,
    TerminalMiss,
)
from .controls_readiness import ControlsReadinessProjection
from .display_values import (
    DisplayFrameKey, StandardDisplayPayload, companion_views_2d,
)
from .display_runtime import browse_publication_needs_hydration
from .events import RunIdentity
from .shell_projection import (
    ScientificPreferences, build_browser_projection,
    build_run_strip_projection, build_scientific_projection,
)
from .shell_values import (
    AxisProjection, BrowserScanIndex, FrameNavigationProjection, HeavyProjection,
    KeyedImageProjection, KeyedPanelsProjection, ProgressProjection, ScientificProjection,
    ShellProjection, TraceProjection,
)
from .state_machine import RunPhase
from .scientific_waterfall_policy import waterfall_should_be_active


#: Why the scan list is grayed out while Browse cannot open another scan (maintainer, 2026-10-25: "make it explicit
#: by disabling and graying out the data browser ... until paused"; his pick "Scan list only"). The page passes
#: context_controller's own admission, so the lock and Browse agree (Codex N-acb7b192 B-R2).
_SCANS_LOCKED = ("Other scans open while the Run is paused or after it stops. The running scan's frames stay "
                 "available below.")
_SCANS_LOCKED_CLEANUP = "Other scans open once the failed Run has finished cleaning up."


def _scientific_presentation_mode(processing_mode: str) -> str:
    """Map an output mode onto its existing scientific center layout."""

    return "Int 1D" if processing_mode == "Int 1D (XYE)" else processing_mode


@dataclass(frozen=True, slots=True)
class ProjectionRequest:
    run_identity: RunIdentity
    selection: DisplaySelection
    frame: DisplayFrameKey
    require_complete: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.run_identity) is not RunIdentity
            or type(self.selection) is not DisplaySelection
            or type(self.frame) is not DisplayFrameKey
            or type(self.require_complete) is not bool
        ):
            raise TypeError("context projection request is invalid")


def apply_presentation_background(
    state: ScientificProjection, projection,
) -> ScientificProjection:
    """A projection with the presentation background's values in place of its traces' (1-D) or its
    image's (raw, 2-D), for the frames the background was promoted for; Int's pages and Stitch's."""
    if projection is None:
        return replace(state, background_set=False)
    try:
        domain, rows, _active_key = projection
        values = dict(rows)
    except (TypeError, ValueError):
        return replace(state, background_set=False)
    if domain == "integrated_1d":
        def substitute(trace):
            value = values.get(id(trace.frame))
            return (replace(trace, intensity=value)
                    if type(value) is np.ndarray
                    and value.shape == trace.intensity.shape else trace)
        return replace(
            state, traces=tuple(substitute(trace) for trace in state.traces),
            pinned_traces=tuple(replace(item, trace=substitute(item.trace))
                                for item in state.pinned_traces),
            background_set=True)
    heavy = state.heavy
    if heavy is None:
        return replace(state, background_set=True)
    value = values.get(id(heavy.frame))
    if type(value) is not np.ndarray:
        return replace(state, background_set=True)
    if domain == "raw" and heavy.raw is not None and value.shape == heavy.raw.shape:
        heavy = replace(heavy, raw=value)
    elif (domain == "integrated_2d" and heavy.cake is not None
          and value.shape == heavy.cake.shape):
        heavy = replace(heavy, cake=value)
    return replace(state, heavy=heavy, background_set=True)


def _same_keyed_grid(shown, background) -> bool:
    """Whether two keyed panels sit on one grid: the same key, axes (values and edges) and, for a slice,
    the same index of the same count."""
    if shown.key != background.key or type(shown) is not type(background):
        return False
    pairs = ((shown.x_axis, background.x_axis), (shown.y_axis, background.y_axis)) \
        if type(shown) is KeyedImageProjection else ((shown.axis, background.axis),)
    for left, right in pairs:
        if (left.values.shape != right.values.shape or not np.array_equal(left.values, right.values)
                or (left.edges is None) != (right.edges is None)
                or left.edges is not None and tuple(left.edges) != tuple(right.edges)):
            return False
    return (type(shown) is not KeyedImageProjection
            or (shown.index == background.index
                and len(shown.index_captions) == len(background.index_captions)))


def apply_keyed_presentation_background(
    shown: KeyedPanelsProjection, background: KeyedPanelsProjection | None, name: str = "",
) -> KeyedPanelsProjection:
    """A keyed layout drawn minus its presentation background, panel by panel (AUD-9 stage 2b G8d): each
    slice minus the background's slice at the same index, each 1-D panel minus the background's. Display
    only. ``background`` is the background result's own panels, built by the same projection authority at the
    indices ``shown`` draws, so a result shown against itself draws zeros. A background on another grid (other
    keys, axes, edges or index count) is not subtracted: the panels are drawn as they are, and the status and
    the button's tip say why. None: no background."""
    if background is None:
        return replace(shown, background_set=False)
    label = name or "the background"
    by_key = {panel.key: panel for panel in background.panels}
    if set(by_key) != {panel.key for panel in shown.panels} or not all(
            _same_keyed_grid(panel, by_key[panel.key]) for panel in shown.panels):
        reason = f"{label} is on another grid: not subtracted (Clear BG to drop it)"
        return replace(shown, background_set=True, background_note=reason,
                       status=f"{shown.status} · {reason}" if shown.status else reason)
    panels = []
    for panel in shown.panels:
        other = by_key[panel.key]
        if type(panel) is KeyedImageProjection:
            panels.append(replace(panel, image=np.subtract(panel.image, other.image)))
        else:
            panels.append(replace(panel, intensity=np.subtract(panel.intensity, other.intensity),
                                  title=f"{panel.title} − background"))
    note = f"minus {label}"
    return replace(shown, panels=tuple(panels), background_set=True, background_note=note,
                   status=f"{shown.status} · {note}" if shown.status else note)


def _terminal_timing_tooltip(progress: ProgressProjection) -> str:
    timing = progress.terminal_timing
    if timing is None:
        return ""
    lines = [
        f"Total: {timing.elapsed_seconds:.2f} s",
        f"Work: {timing.work_seconds:.2f} s",
        f"Cleanup: {timing.cleanup_seconds:.2f} s",
    ]
    labels = {
        "source_read": "Source read",
        "submit_wait": "Submit/backpressure",
        "writer_batch": "NeXus write/checkpoint",
        "writer_flush": "Checkpoint flush/fsync",
        "xye": "XYE",
        "finish_wait": "Finish/drain (includes terminal seal)",
        "display": "Display",
    }
    lines.extend(
        f"{labels.get(name, name)}: {elapsed:.2f} s"
        for name, elapsed in timing.details
    )
    if timing.details:
        lines.append("Parallel detail timers may overlap.")
    quartiles = timing.quartiles
    if quartiles is not None:
        lines.append(
            "Within-run quartiles (Q1 → Q4; Q4 includes terminal tail)"
        )
        lines.append("Frames: " + " | ".join(
            str(value) for value in quartiles.frame_counts
        ))
        quartile_labels = {
            "wall": "Wall",
            "reducer_compute": "Reducer compute",
            "source_read": "Source read",
            "submit_wait": "Submit/backpressure",
            "writer_batch": "NeXus write/checkpoint",
            "writer_flush": "Checkpoint flush/fsync",
            "xye": "XYE",
            "completion_display": "Completion/display",
            "finish_wait": "Finish/drain",
            "gui_refresh": "GUI refresh (pre-terminal)",
        }
        for name, values in quartiles.details:
            if name == "reducer_compute" and any(
                quartiles.compute_counts
            ):
                formatted = " | ".join(
                    f"{value:.2f} s / {count} "
                    f"({(1000.0 * value / count if count else 0.0):.2f} ms/frame)"
                    for value, count in zip(
                        values, quartiles.compute_counts, strict=True,
                    )
                )
            else:
                formatted = " | ".join(
                    f"{value:.2f} s" for value in values
                )
            lines.append(
                f"{quartile_labels.get(name, name)}: {formatted}"
            )
        lines.append("Quartile timers may overlap and do not sum to Wall.")
        lines.append(
            "Submit/backpressure is orchestration wait, not pure integration."
        )
    return "\n".join(lines)
#: A 1D Viewer source is a text or array export with no processing record (COVERAGE-SUPPORT-20261013, D7).
_VIEWER_1D_SUPPORT = AngularSupportProfile.unavailable(SupportReason.NO_RECORD, origin=SupportOrigin.SAVED)


@dataclass(frozen=True, slots=True)
class _Viewer1DTraceProjection(TraceProjection):
    sigma: np.ndarray | None = None

    def __post_init__(self) -> None:
        TraceProjection.__post_init__(self)
        if self.sigma is not None and (type(self.sigma) is not np.ndarray
                or self.sigma.shape != self.intensity.shape
                or self.sigma.flags.writeable):
            raise TypeError("viewer 1-D sigma projection is invalid")
def _viewer_1d_payload(context, request, selection, frame_keys, owner):
    holder = getattr(owner, "holder", None); borrow = getattr(holder, "borrow", None)
    batch = getattr(holder, "batch", None); manifest = getattr(batch, "manifest", None)
    sources = getattr(manifest, "sources", None); index = request.frame.local_frame_label
    mode = (None if borrow is None or type(index) is not int else borrow.modes.get(index))
    owns = type(frame_keys) is dict and frame_keys.get(id(request.frame)) is request.frame
    if (type(context) is not Viewer1DContext or context.state is not Viewer1DState.READY
            or getattr(owner, "context", None) is not context or holder is None
            or getattr(owner, "batch_identity", None) != getattr(manifest, "identity", None)
            or type(sources) is not tuple or type(index) is not int
            or not 0 <= index < len(sources) or mode is None or not owns
            or request.selection is not selection or selection.kind is not ContextKind.VIEWER_1D
            or selection.context_token != context.context_token
            or request.run_identity is not request.frame.run_identity
            or request.run_identity.generation != context.generation
            or request.run_identity.fingerprint != context.context_token
            or request.frame.source_scan != "viewer-1d"
            or request.frame.artifact != "viewer-1d"):
        return None
    source = sources[index]
    view = FrameView(label=index, axis_1d=Axis(source.x_label, source.x_unit,
        values=mode.coordinate), intensity_1d=mode.intensity,
        sigma_1d=mode.uncertainty, source_path=source.canonical_path,
        source_frame_index=index)
    if (view.axis_1d.values is not mode.coordinate
            or view.intensity_1d is not mode.intensity
            or view.sigma_1d is not mode.uncertainty):
        return None
    name = os.path.basename(source.canonical_path)
    return StandardDisplayPayload(selection.display_generation, request.frame,
        f"{name} · 1D Viewer", view, f"1D Viewer · {name}")
def _viewer_2d_payload(context, request, selection, frame_keys, catalog, frame):
    owns = (type(frame_keys) is dict
            and frame_keys.get(id(request.frame)) is request.frame)
    if (type(context) is not Viewer2DContext or context.state is not Viewer2DState.READY
            or type(catalog) is not Viewer2DArtifactCatalog or type(frame) is not Viewer2DFrame
            or request.selection is not selection or selection.kind is not ContextKind.VIEWER_2D
            or selection.context_token != context.context_token
            or request.run_identity is not request.frame.run_identity
            or request.run_identity.generation != context.generation
            or request.run_identity.fingerprint != context.context_token
            or not owns or frame.catalog_identity != catalog.catalog_identity
            or frame.label != request.frame.local_frame_label):
        return None
    kind = frame.provenance.source_kind
    text = {
        Viewer2DSourceKind.RAW_DETECTOR: "Raw detector",
        Viewer2DSourceKind.PROCESSED_RAW: "Processed raw",
        Viewer2DSourceKind.PROCESSED_THUMBNAIL: "Thumbnail preview",
        Viewer2DSourceKind.CSV_MATRIX: "CSV matrix",
        Viewer2DSourceKind.NUMPY_ARRAY: "NumPy array",
    }[kind]
    status = f"2D Viewer · {text}"
    if kind is Viewer2DSourceKind.PROCESSED_THUMBNAIL:
        status += " · Raw source unavailable; displaying stored thumbnail."
    name = os.path.basename(catalog.canonical_path)
    view = FrameView(label=frame.label, raw=frame.array,
                     source_path=catalog.canonical_path,
                     source_frame_index=frame.provenance.frame_index)
    if view.raw is not frame.array:
        return None
    position = catalog.frame_labels.index(frame.label) + 1
    title = f"{name} · frame {position} · {text}"
    return StandardDisplayPayload(selection.display_generation, request.frame,
                                  title, view, status)
def _viewer_1d_scientific(navigation, payloads, resident, preferences, notice,
                          was_waterfall_active=False):
    effective = (preferences.plot_mode if preferences.plot_mode in {
        "Single", "Overlay", "Waterfall"} else "Single")
    payload_by_id = {id(item.frame_key): item for item in payloads
                     if type(item) is StandardDisplayPayload}
    # Single changes presentation, not an explicit multi-file selection.
    frames = navigation.selected
    frames = tuple(frame for frame in frames if frame is not None)
    accepted = tuple(payload_by_id.get(id(frame)) for frame in frames)
    ready = (bool(frames) and all(item is not None for item in accepted)
             and all(frame in resident for frame in frames))
    status = notice or ("1D Viewer" if ready else "Loading 1D Viewer…")
    # A newly adopted result starts its own comparison. One source is a curve,
    # including a rounded text export or a nonuniform native axis; there is no
    # image grid to validate until a second source joins the Waterfall.
    if ready and len({item.view.axis_1d.unit for item in accepted}) != 1:
        ready, status = False, "1D Viewer refused: conflicting units"
    waterfall = waterfall_should_be_active(
        effective, len(frames), was_active=was_waterfall_active, viewer_1d=True)
    if ready and waterfall:
        axes = tuple(item.view.axis_1d.values for item in accepted)
        if any(len(axis) < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0)
               for axis in axes):
            ready, status = False, "1D Viewer refused: axes must be finite and strictly increasing"
        else:
            if os.environ.get("XDART_VIEWER_DEBUG") == "1":
                print("viewer_grid", {"rows": len(axes), "points": len(axes[0]),
                    "max_uniform_error": float(np.max(np.abs(axes[0] - np.linspace(
                        axes[0][0], axes[0][-1], len(axes[0])))))}, flush=True)
            status = "1D Viewer · first selected 1D source display grid"
    traces = []
    if ready:
        # Keep provider coordinates, intensity and uncertainty native. The
        # renderer owns the same display-only grid mapping as integration.
        for frame, payload in zip(frames, accepted):
            view = payload.view; values = view.axis_1d.values
            intensity, sigma = view.intensity_1d, view.sigma_1d
            axis = AxisProjection(values, view.axis_1d.label, view.axis_1d.unit)
            traces.append(_Viewer1DTraceProjection(
                frame, axis, intensity, os.path.basename(view.source_path or ""),
                sigma=sigma, wavelength_m=payload.wavelength_m, angular_support=_VIEWER_1D_SUPPORT))
    current = payload_by_id.get(id(navigation.current))
    if ready and current is not None:
        status += f" · {os.path.basename(current.view.source_path or '')}"
    return ScientificProjection(heavy_available=resident, traces=tuple(traces),
        title=("Current" if current is None else current.title),
        processing_mode="1D Viewer", color_map=preferences.color_map,
        log_scale=preferences.log_scale,
        plot_axis=(preferences.plot_axis if not traces else traces[0].axis.label),
        plot_mode=effective, share_axis=False, plot_options=preferences.plot_options,
        status=status, retain_display=False)


class ContextProjection:
    """Resolve only stores named by the exact current selection."""

    def resident_frame_keys(
        self,
        context: AcquisitionContext | BrowseContext | None,
        frames: tuple[DisplayFrameKey, ...],
        browse_hydration_owner=None,
    ) -> frozenset[DisplayFrameKey]:
        """Project exact heavy residency without retaining store state."""

        if context is None or type(frames) is not tuple:
            return frozenset()
        if type(context) is BrowseContext:
            store = context.publication_store
            return frozenset(
                frame
                for frame in frames
                if not browse_publication_needs_hydration(
                    store.get(frame.local_frame_label),
                    _browse_detector_outcome(
                        context,
                        frame.local_frame_label,
                        browse_hydration_owner,
                    ),
                )
            )
        if type(context) is not AcquisitionContext:
            return frozenset()
        display = context.publication_store
        complete = getattr(display, "complete_frame_keys", None)
        if not callable(complete):
            return frozenset()
        try:
            return complete(frames)
        except Exception:
            return frozenset()

    def build_shell(
        self,
        *,
        revision: int,
        controls: ControlsProjection,
        controls_readiness: ControlsReadinessProjection,
        phase: RunPhase,
        intent: RunIntent,
        contexts: tuple[AcquisitionContext | BrowseContext, ...],
        selection: DisplaySelection | None,
        navigation: FrameNavigationProjection,
        payloads: tuple[StandardDisplayPayload, ...],
        resident_frames: frozenset[DisplayFrameKey],
        progress: ProgressProjection,
        preferences: ScientificPreferences,
        browser_directory: str,
        browser_catalog: tuple[BrowserCatalogEntry, ...] = (),
        browser_catalog_index: BrowserScanIndex | None = None,
        browser_transient_frame: DisplayFrameKey | None = None,
        viewer_1d_current_path: str = "",
        viewer_1d_selected_paths: tuple[str, ...] = (),
        viewer_2d_current_path: str = "",
        viewer_waterfall_active: bool = False,
        date_sorted: bool,
        auto_last: bool,
        executor_available: bool,
        start_permitted: bool,
        start_blocker: str,
        notice: str,
        source_count: int | None = None,
        source_count_is_files: bool = False,
        source_count_includes_immediate: bool = False,
        norm_aggregate: object = None,
        presentation_background=None,
        browse_open: bool = True,
    ) -> ShellProjection:
        """Build the one complete shell value; retain no input or output. ``browse_open``: whether Browse can open
        another scan now (context_controller.browse_lifecycle_admissible); the scan list is grayed out otherwise."""

        tool = tool_from_mode_text(intent.processing_mode)
        viewer_1d_selected = (selection is not None
                              and selection.kind is ContextKind.VIEWER_1D)
        viewer_2d_selected = (selection is not None
                              and selection.kind is ContextKind.VIEWER_2D)
        viewer_selected = viewer_1d_selected or viewer_2d_selected
        viewer_requested = tool in {Tool.XYE_VIEWER, Tool.IMAGE_VIEWER}
        viewer_navigation = (navigation if selection is not None
                             and selection.kind in {
                                 ContextKind.VIEWER_1D, ContextKind.VIEWER_2D}
                             else FrameNavigationProjection())
        run = build_run_strip_projection(
            phase,
            intent,
            executor_available=executor_available,
            start_permitted=start_permitted,
            start_blocker=start_blocker,
            source_count=None if viewer_requested else source_count,
            source_count_is_files=source_count_is_files,
            source_count_includes_immediate=source_count_includes_immediate,
            calibration_ready=(controls_readiness.experiment.ready
                               if controls_readiness.experiment.detail else None),
            calibration_reason=controls_readiness.experiment.detail,
        )
        if viewer_requested:
            run = replace(run, mode=("1D Viewer" if tool is Tool.XYE_VIEWER else "2D Viewer"))
        progress_detail = progress.detail
        if progress.directory_files is not None and (
            not progress.terminal or phase is RunPhase.FAILED
        ):
            progress_detail = progress.directory_files.text(
                "Failed"
                if progress.terminal
                else {
                    RunPhase.PREPARING: "Starting",
                    RunPhase.STARTING: "Starting",
                    RunPhase.RUNNING: "Running",
                    RunPhase.PAUSING: "Pausing",
                    RunPhase.PAUSED: "Paused",
                    RunPhase.RESUMING: "Resuming",
                    RunPhase.STOPPING: "Stopping",
                    RunPhase.FINALIZING: "Finalizing",
                }.get(phase, "Running")
            )
        if (not viewer_selected and
            (
                phase not in {RunPhase.IDLE, RunPhase.FAILED, RunPhase.CLOSED}
                or progress.terminal
            )
            and not (
                phase in {RunPhase.IDLE, RunPhase.FAILED}
                and not start_permitted
            )
            and progress_detail
        ):
            run = replace(
                run,
                readiness=progress_detail,
                readiness_tooltip=_terminal_timing_tooltip(progress),
            )
        current = viewer_navigation.current
        payload = next((item for item in payloads if item.frame_key is current), None)
        viewer_ready = (current is not None
            and current.source_scan == "viewer-2d" and current.artifact == "viewer-2d"
            and payload is not None and current in resident_frames
            and payload.view.raw is not None
        )
        viewer_scientific = ScientificProjection(
            heavy_available=resident_frames if viewer_ready else frozenset(),
            heavy=(HeavyProjection(current, raw=payload.view.raw)
                   if viewer_ready else None),
            title=payload.title if viewer_ready else "Current",
            processing_mode="2D Viewer", color_map=preferences.color_map,
            log_scale=preferences.log_scale,
            status=payload.status if viewer_ready else notice,
            retain_display=False,
        )
        if viewer_1d_selected:
            viewer_scientific = _viewer_1d_scientific(
                viewer_navigation, payloads, resident_frames, preferences, notice,
                viewer_waterfall_active)
        scientific = (viewer_scientific if viewer_selected else
            build_scientific_projection(
                payloads, navigation, resident_frames, preferences, notice,
                phase,
                processing_mode=_scientific_presentation_mode(
                    intent.processing_mode
                ),
                norm_aggregate=norm_aggregate))
        scientific = apply_presentation_background(
            scientific, presentation_background)
        scientific = replace(scientific, acquisition_selected=(
            selection is not None and selection.kind is ContextKind.ACQUISITION))
        if not viewer_selected and selection is not None and selection.kind is ContextKind.BROWSE:
            browse = next((context for context in contexts
                           if type(context) is BrowseContext and selection.names(context)), None)
            if browse is not None and not browse.released:
                scientific = attach_saved_rings(scientific, browse.ring_basis, navigation.current)
                scientific = attach_saved_support_source(scientific, browse)
        return ShellProjection(
            revision,
            build_browser_projection(
                contexts=() if viewer_selected else contexts,
                selection=None if viewer_selected else selection,
                navigation=navigation,
                browser_directory=browser_directory,
                date_sorted=date_sorted,
                auto_last=auto_last,
                catalog=browser_catalog,
                catalog_index=browser_catalog_index,
                transient_frame=browser_transient_frame,
                selected_artifacts=(
                    viewer_1d_selected_paths if viewer_1d_selected else ()
                ),
                # The 2D Viewer's file intent leads, as 1-D intent does: while the
                # next file's catalog loads (or the prior file clears) nothing is
                # selected, and an empty current row sent the next arrow key to
                # the list's top.
                current_artifact=(
                    viewer_2d_current_path
                    if tool is Tool.IMAGE_VIEWER and viewer_2d_current_path
                    else next((os.path.abspath(context.original_path)
                               for context in contexts
                               if type(context) is Viewer2DContext
                               and selection.names(context)), "")
                    if viewer_2d_selected else viewer_1d_current_path
                ),
                multi_artifact_selection=viewer_1d_selected or tool is Tool.XYE_VIEWER,
                show_all_frames=viewer_selected,
                scans_locked="" if browse_open else (
                    _SCANS_LOCKED_CLEANUP if phase is RunPhase.FAILED else _SCANS_LOCKED),
            ),
            scientific,
            viewer_navigation if viewer_selected else navigation,
            controls,
            run,
            ProgressProjection() if viewer_selected else progress,
            controls_readiness,
        )

    def project(
        self,
        context: AcquisitionContext | BrowseContext | Viewer1DContext | Viewer2DContext,
        request: ProjectionRequest,
        current_selection: DisplaySelection,
        accepted_run_identity: RunIdentity | None,
        frame_keys: dict[int, DisplayFrameKey],
        browse_hydration_owner=None,
        *,
        viewer_1d_owner=None,
        viewer_catalog=None,
        viewer_frame=None,
    ) -> StandardDisplayPayload | None:
        owns_frame = (
            type(frame_keys) is dict
            and frame_keys.get(id(request.frame)) is request.frame
        )
        if type(context) is Viewer1DContext:
            return _viewer_1d_payload(
                context, request, current_selection, frame_keys, viewer_1d_owner)
        if type(context) is Viewer2DContext:
            return _viewer_2d_payload(
                context, request, current_selection, frame_keys,
                viewer_catalog, viewer_frame,
            )
        if (
            request.run_identity is not accepted_run_identity
            or request.selection is not current_selection
            or not current_selection.names(context)
            or current_selection.owner != context.hydration_owner
            or not owns_frame
        ):
            return None
        if type(context) is AcquisitionContext:
            display = context.publication_store
            if (
                context.record_store is not display
                or getattr(display, "identity", None)
                is not request.run_identity
            ):
                return None
            return display.project(
                request.frame,
                request.selection.display_generation,
                closed=False,
                owner=context.hydration_owner,
                commit_gate=context.commit_gate,
                require_complete=request.require_complete,
            )
        if type(context) is not BrowseContext or context.released:
            return None
        frame = request.frame
        if (
            frame.run_identity is not request.run_identity
            or frame.source_scan != context.scan_key
            or frame.artifact != context.requested_path
        ):
            return None
        label = frame.local_frame_label
        publication = context.publication_store.get(label)
        if publication is None or publication.scan_key != context.scan_key:
            return None
        if (
            request.require_complete
            and browse_publication_needs_hydration(
                publication,
                _browse_detector_outcome(
                    context, label, browse_hydration_owner
                ),
            )
        ):
            return None
        return _browse_payload(
            context,
            publication,
            frame,
            request.selection.display_generation,
        )
    def resolve_browse(
        self,
        context: BrowseContext,
        request: ProjectionRequest,
        current_selection: DisplaySelection | None,
        current_frame: DisplayFrameKey | None,
        owner,
    ) -> BrowseProjectionResolution:
        """Resolve the CURRENT Browse frame with at most ONE store read.

        The runtime-owned anchors refuse a request whose selection or frame
        is not those exact objects by identity, before the read; the Browse
        request/read key/token are constructed only here, the transport never.
        """
        from .browse_hydration import _BrowseHydrationOwner

        if (
            type(context) is not BrowseContext
            or type(request) is not ProjectionRequest
            or type(owner) is not _BrowseHydrationOwner
        ):
            return TerminalMiss(BrowseMissReason.UNRESOLVABLE)
        selection = request.selection
        frame = request.frame
        if selection is not current_selection or frame is not current_frame:
            return TerminalMiss(BrowseMissReason.FOREIGN)
        generation = selection.display_generation
        if context.released or context.invalidated or not context.loaded:
            return TerminalMiss(BrowseMissReason.RETIRED)
        if context.commit_gate.cancelled:
            return TerminalMiss(BrowseMissReason.CLOSED)
        if (
            selection.kind is not ContextKind.BROWSE
            or not selection.names(context)
            or selection.owner != context.hydration_owner
            or not owner.names(context)
            or frame.source_scan != context.scan_key
            or frame.artifact != context.requested_path
        ):
            return TerminalMiss(BrowseMissReason.FOREIGN)
        hydration_owner = context.hydration_owner
        if not hydration_owner.qualified:
            return TerminalMiss(BrowseMissReason.UNRESOLVABLE)
        label = frame.local_frame_label
        publication = context.publication_store.get(label)
        if (
            publication is not None
            and publication.scan_key != context.scan_key
        ):
            return TerminalMiss(BrowseMissReason.FOREIGN)
        if publication is not None and not browse_publication_needs_hydration(
            publication,
            _browse_detector_outcome(context, label, owner),
        ):
            return QualifiedPayload(
                _browse_payload(context, publication, frame, generation)
            )
        try:
            read_key = HydrationReadKey(
                HydrationScope(*hydration_owner.as_tuple()),
                context.requested_path,
                label,
                HydrationPurpose.PREVIEW,
                context.load_request.source_root,
            )
            request = HydrationRequest(
                label,
                HydrationPurpose.PREVIEW,
                generation,
                hydration_owner,
                (context.publication_store,),
                context.commit_gate,
                read_key=read_key,
                token=HydrationToken(read_key, generation),
            )
        except (TypeError, ValueError):
            return TerminalMiss(BrowseMissReason.UNRESOLVABLE)
        return HydrationEligibleMiss(request)


def attach_saved_rings(scientific: ScientificProjection, basis, current) -> ScientificProjection:
    """A Browse Int frame's own saved ring snapshot on its heavy projection (BROWSE-REFERENCE-RINGS-20261013).

    The snapshot and native shape come only from the output's basis. A thumbnail takes the saved detector's shape;
    a full raw image or recorded native shape that differs refuses. A frame whose image has not arrived keeps
    ``saved_ring_expected``, so the view says its rings wait for it; a refused frame carries its reason. An Average
    result or a Reintegrate version is not an Int Run's record and keeps the view's generic reason.
    """
    from xrd_tools.geometry_figures import RING_SHAPE_MISMATCH_REASON

    if (basis is None or basis.kind != "run" or current is None
            or scientific.processing_mode not in {"Int 1D", "Int 2D"}):
        return scientific
    label = current.local_frame_label
    snapshot = basis.frame_snapshot(label)
    if snapshot is None:
        return replace(scientific, saved_ring_reason=basis.frame_reason(label))
    heavy = scientific.heavy
    if heavy is None or heavy.frame is not current:
        return replace(scientific, saved_ring_expected=True)
    if heavy.raw is None:
        return scientific
    shape = basis.shape
    if (heavy.detector_source == "full" and tuple(heavy.raw.shape) != shape
            or heavy.detector_shape is not None and heavy.detector_shape != shape):
        return replace(scientific, heavy=replace(heavy, ring_geometry=None, ring_origin=""),
                       saved_ring_reason=RING_SHAPE_MISMATCH_REASON)
    return replace(scientific, saved_ring_expected=True, heavy=replace(
        heavy, ring_geometry=snapshot, detector_shape=shape, ring_origin="saved"))


def attach_saved_support_source(scientific: ScientificProjection, browse: BrowseContext) -> ScientificProjection:
    """A Browse Int output's own facts, from which the strip may rebuild a frame's angular coverage when the output
    saved none (COVERAGE-SUPPORT-20261013, D5): its frames' name, file, entry, load snapshot and labels."""
    from .shell_values import SavedSupportSource

    catalog = browse.scalar_catalog
    if (scientific.processing_mode not in {"Int 1D", "Int 2D"} or browse.target_snapshot is None
            or catalog is None or not catalog.labels):
        return scientific
    return replace(scientific, saved_support_source=SavedSupportSource(
        browse.requested_path, catalog.artifact_path, browse.target_entry or catalog.entry, browse.target_snapshot,
        tuple(catalog.labels)))


def _browse_payload(
    context: BrowseContext,
    publication,
    frame: DisplayFrameKey,
    generation: int,
) -> StandardDisplayPayload:
    """Build the one Browse payload shape from an already-read publication."""
    label = frame.local_frame_label
    view = publication.view
    records = context.record_store
    record = (
        None
        if records is None
        else records.get(label)
    )
    if record is not None:
        light_view = record.active_view()
        # Browse's headless record store may have thinned an old, persisted
        # row under its independent heavy bound.  Never let that missing copy
        # erase the complete cheap 1-D projection retained by the publication
        # store for Overlay/Waterfall.
        if light_view.has_1d:
            view = replace(
                view,
                axis_1d=light_view.axis_1d,
                intensity_1d=light_view.intensity_1d,
                sigma_1d=light_view.sigma_1d,
            )
    measurement, motor = _browse_measurement(context, publication)
    return StandardDisplayPayload(
        generation,
        frame,
        f"Browse · {context.scan_key} · frame {label}",
        view,
        "browse",
        measurement_mode=measurement,
        gi_incidence_motor=motor,
        gi_resolved_motor=motor,
        gi_mode_1d=(
            publication.record.active_mode_1d
            if measurement == "GI"
            else ""
        ),
        gi_mode_2d=(
            publication.record.active_mode_2d
            if measurement == "GI"
            else ""
        ),
        wavelength_m=_browse_wavelength(context),
        extra_views_2d=(
            companion_views_2d(publication.record) if measurement == "GI" else {}
        ),
    )


def _browse_detector_outcome(
    context: BrowseContext, label, browse_hydration_owner=None
):
    # Local import preserves the existing browse_preview -> ProjectionRequest
    # module direction while querying only B's already-created exact owner.
    from .browse_preview import cold_browse_detector_outcome

    return cold_browse_detector_outcome(
        context, label, browse_hydration_owner
    )


def _browse_measurement(
    context: BrowseContext, publication,
) -> tuple[str, str]:
    if publication.view.two_d_kind.value == "q_chi":
        return "Standard", ""
    try:
        facts = json.loads(context.calibration_identity or "{}")
        gi = facts.get("gi")
        if type(gi) is dict:
            motor = str(
                gi.get("resolved_motor")
                or gi.get("incidence_motor")
                or ""
            )
            if motor:
                return "GI", motor
        geometry = facts.get("geometry") or {}
        mapping = geometry.get("mapping_json") or {}
        incident = mapping.get("incident_angle") or {}
        motor = str(incident.get("source_motor") or "")
        if motor:
            return "GI", motor
    except (AttributeError, TypeError, ValueError):
        pass
    return "GI", "Manual"


def _browse_wavelength(context: BrowseContext) -> float | None:
    """Recover only the loader's explicit, detached metre provenance."""

    try:
        facts = json.loads(context.calibration_identity or "{}")
    except (TypeError, ValueError):
        return None
    if type(facts) is not dict:
        return None
    value = facts.get("wavelength_m")
    if type(value) not in {int, float}:
        return None
    return canonical_wavelength_m(value, WavelengthUnit.METRE)


__all__ = ["ContextProjection", "ProjectionRequest", "apply_presentation_background", "attach_saved_rings"]
