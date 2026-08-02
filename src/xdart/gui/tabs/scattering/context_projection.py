"""Context-qualified, no-I/O display projection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json

from xdart.modules.display_context import (
    AcquisitionContext,
    BrowseContext,
    ContextKind,
    DisplaySelection,
    HydrationRequest,
)
from xrd_tools.core.energy import (
    WavelengthUnit,
    canonical_wavelength_m,
)
from xrd_tools.session.hydration import (
    HydrationPurpose,
    HydrationReadKey,
    HydrationScope,
    HydrationToken,
)
from xrd_tools.session.readiness import ControlPanelRenderState
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
    DisplayFrameKey,
    StandardDisplayPayload,
)
from .display_runtime import (
    publication_needs_hydration,
)
from .events import RunIdentity
from .shell_projection import (
    ScientificPreferences,
    build_browser_projection,
    build_run_strip_projection,
    build_scientific_projection,
)
from .shell_values import (
    FrameNavigationProjection,
    ProgressProjection,
    ShellProjection,
)
from .state_machine import RunPhase


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
                if not publication_needs_hydration(
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
        resident: list[DisplayFrameKey] = []
        for frame in frames:
            try:
                owner = display.artifacts.get(frame.artifact)
                publication = (
                    None
                    if owner is None
                    else owner.publications.get(frame.local_frame_label)
                )
                detector_outcome = display.detector_outcome(frame)
                if not publication_needs_hydration(
                    publication,
                    detector_outcome,
                ):
                    resident.append(frame)
            except Exception:
                continue
        return frozenset(resident)

    def build_shell(
        self,
        *,
        revision: int,
        controls: ControlPanelRenderState,
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
        browser_transient_frame: DisplayFrameKey | None = None,
        date_sorted: bool,
        auto_last: bool,
        executor_available: bool,
        start_permitted: bool,
        start_blocker: str,
        notice: str,
        source_count: int | None = None,
        source_count_is_files: bool = False,
        source_count_includes_immediate: bool = False,
    ) -> ShellProjection:
        """Build the one complete shell value; retain no input or output."""

        run = build_run_strip_projection(
            phase,
            intent,
            executor_available=executor_available,
            start_permitted=start_permitted,
            start_blocker=start_blocker,
            source_count=source_count,
            source_count_is_files=source_count_is_files,
            source_count_includes_immediate=(
                source_count_includes_immediate
            ),
        )
        controls = replace(
            controls,
            profile=replace(
                controls.profile,
                run_enabled=run.run_enabled,
                run_blockers=(
                    () if run.ready else (run.readiness,)
                ),
            ),
        )
        return ShellProjection(
            revision,
            build_browser_projection(
                contexts=contexts,
                selection=selection,
                navigation=navigation,
                browser_directory=browser_directory,
                date_sorted=date_sorted,
                auto_last=auto_last,
                catalog=browser_catalog,
                transient_frame=browser_transient_frame,
            ),
            build_scientific_projection(
                payloads,
                navigation,
                resident_frames,
                preferences,
                notice,
                phase,
                processing_mode=intent.processing_mode,
            ),
            navigation,
            controls,
            run,
            progress,
            controls_readiness,
        )

    def project(
        self,
        context: AcquisitionContext | BrowseContext,
        request: ProjectionRequest,
        current_selection: DisplaySelection,
        accepted_run_identity: RunIdentity | None,
        frame_keys: tuple[DisplayFrameKey, ...] | dict[int, DisplayFrameKey],
        browse_hydration_owner=None,
    ) -> StandardDisplayPayload | None:
        if type(frame_keys) is dict:
            owns_frame = frame_keys.get(id(request.frame)) is request.frame
        elif type(frame_keys) is tuple:
            # Compatibility for direct port tests and third-party callers.
            # Production passes the runtime's exact O(1) identity index.
            owns_frame = any(frame is request.frame for frame in frame_keys)
        else:
            owns_frame = False
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
            and publication_needs_hydration(
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
        if publication is not None and not publication_needs_hydration(
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


__all__ = ["ContextProjection", "ProjectionRequest"]
