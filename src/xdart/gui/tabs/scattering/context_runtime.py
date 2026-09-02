"""Exact context, selection, and navigation state owned by E3."""

from __future__ import annotations

from dataclasses import dataclass

from xdart.modules.display_context import (
    AcquisitionContext, BrowseContext, ContextKind, DisplaySelection,
    HydrationOwner, Viewer1DContext, Viewer1DState, Viewer2DContext,
)
from xrd_tools.io.viewer_2d import Viewer2DArtifactCatalog, Viewer2DFrame
from xrd_tools.session.hydration import (
    HydrationPurpose, HydrationReadKey, HydrationToken)
from xrd_tools.session.scan_norm import (
    ScanNormAggregate, accepts_norm_aggregate)

from .browse_1d_projection import (
    Browse1DProjectionOutcome,
    Browse1DProjectionStatus,
)
from .browse_1d_target_plan import (
    Browse1DRuntimeProjection,
    Browse1DTargetPlan,
    Browse1DTargetPlanStatus,
    plan_browse_1d_targets,
)
from .browse_hydration import _BrowseHydrationOwner
from .browse_preview import qualified_event_frame
from .browse_values import BrowseLoadRequest
from .context_projection import ContextProjection, ProjectionRequest
from .context_values import (
    _BrowseProjectionPass,
    _navigation_after_append,
    _normalized_navigation,
    _PendingBrowseReplacement,
    BrowseMissReason,
    HydrationEligibleMiss,
    QualifiedPayload,
    TerminalMiss,
)
from .display_retirement import DisplayRetirementReceipt
from .display_values import (
    DisplayFrameKey,
    DisplayNavigationDelta,
    StandardDisplayPayload,
    StandardEventKind,
    StandardRunEvent,
    display_payload_is_valid,
)
from .events import CleanupStatus, RunIdentity
from .scientific_axes import (
    resolve_norm_presentation,
    trace_normalization_scope,
)
from .shell_values import FrameNavigationProjection, SlicePin


@dataclass(frozen=True, slots=True)
class _PendingTraceProjection:
    scope: tuple[object, ...]
    target: tuple[DisplayFrameKey, ...]


def _qualified_browse_resolution(resolution, identity, current, generation,
                                 browse):
    """THE one typed-resolution qualifier (§22.3): an exactly valid payload,
    a miss whose every carried field names the exact current Browse read, or
    a finite terminal — anything else disqualifies to None."""
    if type(resolution) is QualifiedPayload:
        return resolution if display_payload_is_valid(
            resolution.payload, identity, current, generation) else None
    if type(resolution) is TerminalMiss:
        return (resolution
                if type(resolution.reason) is BrowseMissReason else None)
    if type(resolution) is not HydrationEligibleMiss:
        return None
    request = resolution.request
    label = current.local_frame_label
    read_key = request.read_key
    return resolution if (
        request.label == label
        and request.purpose is HydrationPurpose.PREVIEW
        and request.generation == generation
        and request.owner == browse.hydration_owner
        and request.stores == (browse.publication_store,)
        and request.commit_gate is browse.commit_gate
        and type(read_key) is HydrationReadKey
        and read_key.scope == request.scope
        and read_key.artifact_identity == browse.requested_path
        and read_key.source_root == browse.load_request.source_root
        and read_key.frame_identity == label
        and read_key.purpose is HydrationPurpose.PREVIEW
        and type(request.token) is HydrationToken
        and request.token.read_key is read_key
        and request.token.presentation_generation == generation
    ) else None


class _ContextRuntime:
    """Single value owner behind the command-facing context controller."""

    def __init__(self) -> None:
        self._run_identity: RunIdentity | None = None
        self._acquisition: AcquisitionContext | None = None
        self._browse: BrowseContext | None = None
        self._viewer_1d: Viewer1DContext | None = None
        self._viewer_2d: Viewer2DContext | None = None
        self._viewer_2d_catalog: Viewer2DArtifactCatalog | None = None
        self._viewer_2d_frame: Viewer2DFrame | None = None
        self._browse_projection_identity: RunIdentity | None = None
        self._selection: DisplaySelection | None = None
        self._acquisition_navigation = FrameNavigationProjection()
        self._browse_navigation = FrameNavigationProjection()
        self._viewer_1d_navigation = FrameNavigationProjection()
        self._viewer_2d_navigation = FrameNavigationProjection()
        self._acquisition_frame_by_id: dict[int, DisplayFrameKey] = {}
        self._browse_frame_by_id: dict[int, DisplayFrameKey] = {}
        self._viewer_1d_frame_by_id: dict[int, DisplayFrameKey] = {}
        self._viewer_2d_frame_by_id: dict[int, DisplayFrameKey] = {}
        self._pending_replacement: _PendingBrowseReplacement | None = None
        self._display_generation = 0
        self._committed_trace_scope: tuple[object, ...] | None = None
        self._committed_trace_selection: tuple[DisplayFrameKey, ...] = ()
        self._pending_trace_projection: _PendingTraceProjection | None = None
        self._browse_pass: _BrowseProjectionPass | None = None
        self._norm_aggregate: ScanNormAggregate | None = None
        self._norm_capture_scope: tuple[object, object, object] | None = None

    @property
    def norm_aggregate(self) -> ScanNormAggregate | None:
        """The one §25.3 runtime-held snapshot, borrowed per refresh."""
        return self._norm_aggregate

    @property
    def run_identity(self) -> RunIdentity | None:
        return self._run_identity
    @property
    def acquisition_context(self) -> AcquisitionContext | None:
        return self._acquisition
    @property
    def browse_context(self) -> BrowseContext | None:
        return self._browse
    @property
    def selection(self) -> DisplaySelection | None:
        pending = self._pending_replacement
        return pending.selection if pending is not None else self._selection
    @property
    def retained_contexts(self) -> tuple:
        return tuple(
            context
            for context in (self._acquisition, self._browse,
                            self._viewer_1d, self._viewer_2d)
            if context is not None
        )

    @property
    def projectable_contexts(self) -> tuple:
        browse = self._browse
        return tuple(
            context
            for context in (
                self._acquisition,
                (
                    browse
                    if browse is not None
                    and not browse.invalidated
                    and not browse.released
                    else None
                ),
                self._viewer_1d,
                self._viewer_2d,
            )
            if context is not None
        )

    @property
    def navigation(self) -> FrameNavigationProjection:
        pending = self._pending_replacement
        if pending is not None:
            return pending.navigation
        selection = self._selection
        if selection is not None and selection.kind is ContextKind.VIEWER_1D:
            return self._viewer_1d_navigation
        if selection is not None and selection.kind is ContextKind.VIEWER_2D:
            return self._viewer_2d_navigation
        if selection is not None and selection.kind is ContextKind.BROWSE:
            return self._browse_navigation
        return self._acquisition_navigation

    @property
    def frame_keys(self) -> tuple[DisplayFrameKey, ...]:
        return self.navigation.frames if self.selection is not None else ()

    def resident_frame_keys(
        self, projection: ContextProjection, browse_hydration_owner=None
    ) -> frozenset[DisplayFrameKey]:
        if self._pending_replacement is not None:
            return frozenset()
        selection = self._selection
        if selection is None:
            return frozenset()
        if selection.kind is ContextKind.VIEWER_1D:
            context = self._viewer_1d
            return (frozenset(self._viewer_1d_navigation.frames)
                    if context is not None and context.state is Viewer1DState.READY
                    else frozenset())
        if selection.kind is ContextKind.VIEWER_2D:
            current = self._viewer_2d_navigation.current
            frame = self._viewer_2d_frame
            owns_current = (current is not None and frame is not None
                            and frame.label == current.local_frame_label)
            return frozenset((current,)) if owns_current else frozenset()
        context = (
            self._browse
            if selection.kind is ContextKind.BROWSE
            else self._acquisition
        )
        if (
            type(context) is AcquisitionContext
            and selection.owner != context.hydration_owner
        ):
            # The executor can rescope the one mutable acquisition context to
            # the next short directory artifact before the GUI drains its
            # CONTEXT_READY event.  Payload projection already rejects that
            # stale owner.  Residency must reject it too so the shell sees one
            # coherent pending state instead of "resident but no payload".
            return frozenset()
        if selection.kind is ContextKind.BROWSE:
            # Residency REUSES only a still-qualified QUALIFIED pass for
            # current — never a store reread — and clears a disqualified one.
            current = self._browse_navigation.current
            others = tuple(
                frame for frame in self.frame_keys if frame is not current
            )
            resident = projection.resident_frame_keys(
                context, others, browse_hydration_owner
            )
            snapshot = self._browse_pass
            if snapshot is not None and not self._browse_scope_qualified(
                snapshot.selection, snapshot.current,
                snapshot.generation, browse_hydration_owner,
            ):
                self._browse_pass = None
                snapshot = None
            if (current is not None and snapshot is not None
                    and type(snapshot.resolution) is QualifiedPayload):
                resident = resident | {current}
            return resident
        return projection.resident_frame_keys(
            context, self.frame_keys, browse_hydration_owner
        )

    def owns_frame(self, frame: object) -> bool:
        return (
            self._pending_replacement is None
            and type(frame) is DisplayFrameKey
            and self._selected_frame_by_id().get(id(frame)) is frame
        )

    def _full_raw_target(self):
        selection, context = self._selection, self._acquisition
        current = self._acquisition_navigation.current
        return ((context, current) if selection is not None and context is not None
                and current is not None and selection.kind is ContextKind.ACQUISITION
                and selection.names(context) and selection.owner == context.hydration_owner
                and self.owns_frame(current) else None)

    def full_raw_availability(self) -> tuple[bool, str]:
        return ((True, "") if self._full_raw_target() is not None else (False,
            "Full Raw is available only for an exact acquisition frame."))

    def full_raw_status(self) -> tuple[bool, bool, str | None]:
        target = self._full_raw_target()
        return ((False, False, self.full_raw_availability()[1]) if target is None else
                target[0].publication_store.full_raw_status(target[1]))

    def request_full_current(self):
        target = self._full_raw_target()
        if target is None: return None
        context, frame = target; selection = self._selection
        return context.publication_store.request_full(frame, selection.display_generation,
            owner=selection.owner, commit_gate=context.commit_gate)

    def clear_full_raw(self) -> bool:
        if self._acquisition is None: return False
        self._acquisition.publication_store.invalidate_full_demand(clear_raw=True)
        return True

    def prepare_viewer_1d_begin(self, context: Viewer1DContext):
        if type(context) is not Viewer1DContext: raise TypeError(
            "viewer 1-D context must be exact")
        self.clear_full_raw()
        navigation = FrameNavigationProjection()
        selection = DisplaySelection(ContextKind.VIEWER_1D, HydrationOwner(
            context.context_token, "viewer-1d", "viewer-1d", context.commit_gate.epoch),
            context.generation)
        return navigation, {}, selection
    def prepare_viewer_1d_navigation(self, context: Viewer1DContext, count: int):
        if (type(context) is not Viewer1DContext or type(count) is not int
                or count != len(context.paths) or count < 1):
            raise TypeError("viewer 1-D navigation candidate is invalid")
        identity = RunIdentity(context.generation, context.context_token)
        frames = tuple(DisplayFrameKey(
            identity, "viewer-1d", "viewer-1d", index, index + 1)
            for index in range(count))
        current_index = (
            0
            if context.current_path is None
            else context.paths.index(context.current_path)
        )
        navigation = FrameNavigationProjection(
            frames,
            frames[current_index],
            frames,
        )
        return navigation, {id(frame): frame for frame in frames}
    def replace_viewer_1d_context(self, context: Viewer1DContext) -> bool:
        current = self._viewer_1d
        if (type(context) is not Viewer1DContext or current is None
                or context.context_token != current.context_token
                or context.commit_gate is not current.commit_gate): return False
        self._viewer_1d = context; return True
    def select_viewer_1d(
        self,
        frame: DisplayFrameKey,
        selected: tuple[DisplayFrameKey, ...],
    ) -> bool:
        if (type(frame) is not DisplayFrameKey
                or self._viewer_1d_frame_by_id.get(id(frame)) is not frame): return False
        if (
            type(selected) is not tuple
            or len({id(item) for item in selected}) != len(selected)
            or any(
                type(item) is not DisplayFrameKey
                or self._viewer_1d_frame_by_id.get(id(item)) is not item
                for item in selected
            )
        ):
            return False
        navigation = self._viewer_1d_navigation
        self._set_viewer_1d_navigation(FrameNavigationProjection(
            navigation.frames, frame, selected))
        self._reset_trace_projection(); return True
    def clear_viewer_1d(self) -> None:
        if (self._selection is not None
                and self._selection.kind is ContextKind.VIEWER_1D):
            self._selection = None
        self._viewer_1d = None
        self._set_viewer_1d_navigation(FrameNavigationProjection())
        self._reset_trace_projection()
    def adopt_viewer_2d(self, context: Viewer2DContext,
                        catalog: Viewer2DArtifactCatalog) -> None:
        if (type(context) is not Viewer2DContext
                or type(catalog) is not Viewer2DArtifactCatalog):
            raise TypeError("viewer context/catalog must be exact")
        self.clear_full_raw()
        identity = RunIdentity(context.generation, context.context_token)
        frames = tuple(DisplayFrameKey(
            identity, "viewer-2d", "viewer-2d", label, index + 1)
            for index, label in enumerate(catalog.frame_labels))
        navigation = FrameNavigationProjection(frames, frames[0], (frames[0],))
        self._display_generation += 1
        selection = DisplaySelection(ContextKind.VIEWER_2D, HydrationOwner(
            context.context_token, "viewer-2d", "viewer-2d",
            context.commit_gate.epoch), self._display_generation)
        self._viewer_2d, self._viewer_2d_catalog, self._viewer_2d_frame = (
            context, catalog, None)
        self._set_viewer_2d_navigation(navigation)
        self._selection, self._pending_replacement = selection, None
        self._reset_trace_projection()

    def replace_viewer_2d_context(self, context: Viewer2DContext) -> bool:
        current = self._viewer_2d
        if (type(context) is not Viewer2DContext or current is None
                or context.context_token != current.context_token
                or context.generation != current.generation
                or context.original_path != current.original_path
                or context.commit_gate is not current.commit_gate):
            return False
        self._viewer_2d = context
        return True

    def select_viewer_2d(self, label: int, *, advance: bool = True) -> DisplaySelection:
        context = self._viewer_2d
        if context is None or type(label) is not int:
            raise RuntimeError("no selectable 2D Viewer context")
        frame = next((item for item in self._viewer_2d_navigation.frames
                      if item.local_frame_label == label), None)
        if frame is None:
            raise RuntimeError("2D Viewer frame is not in the catalog")
        current = self._viewer_2d_navigation.current
        if (not advance and current is frame and self._selection is not None
                and self._selection.kind is ContextKind.VIEWER_2D):
            return self._selection
        self._display_generation += int(advance)
        self._selection = DisplaySelection(
            ContextKind.VIEWER_2D,
            HydrationOwner(context.context_token, "viewer-2d", "viewer-2d",
                           context.commit_gate.epoch), self._display_generation)
        navigation = FrameNavigationProjection(
            self._viewer_2d_navigation.frames, frame, (frame,))
        self._set_viewer_2d_navigation(navigation)
        self._pending_replacement = None
        self._reset_trace_projection()
        return self._selection

    def clear_viewer_2d(self, *, release: bool = False) -> None:
        self._viewer_2d_frame = None
        if not release:
            return
        if (self._selection is not None
                and self._selection.kind is ContextKind.VIEWER_2D):
            self._selection = None
        self._viewer_2d = self._viewer_2d_catalog = None
        self._set_viewer_2d_navigation(FrameNavigationProjection())
        self._reset_trace_projection()

    def adopt_acquisition(
        self, run_identity: RunIdentity, context: AcquisitionContext
    ) -> DisplaySelection:
        if (
            type(run_identity) is not RunIdentity
            or type(context) is not AcquisitionContext
            or context.run_configuration is None
            or context.run_configuration.identity
            != (run_identity.generation, run_identity.fingerprint)
        ):
            raise RuntimeError("executor has no exact acquisition context")
        if (
            self._acquisition is not None
            and self._acquisition is not context
        ):
            raise RuntimeError("context runtime already owns acquisition")
        self.invalidate_browse_pass()
        self._run_identity = run_identity
        self._acquisition = context
        try:
            entries = context.publication_store.catalog_snapshot().entries
        except Exception:
            entries = ()
        if (
            type(entries) is not tuple
            or not all(
                type(frame) is DisplayFrameKey
                and frame.run_identity is run_identity
                for frame in entries
            )
        ):
            entries = ()
        navigation = self._acquisition_navigation
        self._set_acquisition_navigation(FrameNavigationProjection(
            *_normalized_navigation(
                entries, navigation.current, navigation.selected,
                prefer_last=True)))
        pending = self._pending_replacement
        if pending is not None:
            # A delayed acquisition wake-up must not discard the released
            # Browse presentation intentionally held while its replacement
            # loads.  The acquisition catalog above is still refreshed for a
            # later explicit return to acquisition.
            return pending.selection
        selection = self._selection
        browse = self._browse
        if selection is not None and selection.kind in {
                ContextKind.VIEWER_1D, ContextKind.VIEWER_2D}:
            viewer = (self._viewer_1d if selection.kind is ContextKind.VIEWER_1D
                      else self._viewer_2d)
            if viewer is not None and selection.names(viewer): return selection
        if (
            selection is not None
            and selection.kind is ContextKind.BROWSE
            and browse is not None
            and not browse.invalidated
            and not browse.released
            and selection.names(browse)
        ):
            # CONTEXT_READY is only an acquisition-side wake-up.  Paused and
            # idle Browse selection remains operator-owned until an explicit
            # acquisition selection or lifecycle transition changes it.
            return selection
        if (
            selection is not None
            and selection.kind is ContextKind.ACQUISITION
            and selection.owner == context.hydration_owner
        ):
            # CONTEXT_READY is a wake-up hint, not a new display ownership
            # generation.  Several short artifacts may queue identical hints
            # before the GUI drains them; adopting the already-selected exact
            # owner again must not invalidate requests made between hints.
            return selection
        return self._select(context)

    def accept_navigation(
        self,
        delta: DisplayNavigationDelta,
        *,
        plot_mode: str = "Single",
        follow_latest: bool = True,
    ) -> bool:
        context = self._acquisition
        identity = self._run_identity
        if (
            type(delta) is not DisplayNavigationDelta
            or type(follow_latest) is not bool
            or context is None
            or identity is None
            or delta.appended.run_identity is not identity
            or context.publication_store.resolve_frame(delta.appended)
            is not delta.appended
        ):
            return False
        navigation = FrameNavigationProjection(
            *_navigation_after_append(
                self._acquisition_navigation,
                delta,
                plot_mode,
                follow_latest=follow_latest,
            )
        )
        if (
            navigation.current is not self._acquisition_navigation.current
            and self._acquisition is not None
        ):
            self._acquisition.publication_store.invalidate_full_demand()
        self._acquisition_navigation = navigation
        for retired in delta.retired:
            self._acquisition_frame_by_id.pop(id(retired), None)
        self._acquisition_frame_by_id[id(delta.appended)] = delta.appended
        return True

    def select_latest_navigation(
        self, *, plot_mode: str = "Single"
    ) -> bool:
        if self._pending_replacement is not None:
            return False
        selection = self._selection
        if selection is None:
            return False
        if selection.kind in {ContextKind.VIEWER_1D, ContextKind.VIEWER_2D}:
            return False
        navigation = (
            self._browse_navigation if selection.kind is ContextKind.BROWSE
            else self._acquisition_navigation
        )
        if not navigation.frames:
            return False
        latest = navigation.frames[-1]
        replacement = FrameNavigationProjection(
            navigation.frames,
            latest,
            (
                navigation.selected
                if plot_mode in {"Overlay", "Waterfall"}
                else navigation.frames
                if plot_mode in {"Average", "Sum"}
                else (latest,)
            ),
        )
        if selection.kind is ContextKind.BROWSE:
            self._set_browse_navigation(replacement)
        else:
            self._set_acquisition_navigation(replacement)
        return True

    def select_navigation(
        self,
        current: DisplayFrameKey | None,
        selected: tuple[DisplayFrameKey, ...],
    ) -> bool:
        if self._pending_replacement is not None or type(selected) is not tuple:
            return False
        try:
            navigation = FrameNavigationProjection(
                self.frame_keys, current, selected
            )
        except (TypeError, ValueError):
            return False
        selection = self._selection
        if selection is not None and selection.kind is ContextKind.VIEWER_1D:
            return bool(
                current is not None
                and self.select_viewer_1d(current, selected)
            )
        if selection is not None and selection.kind is ContextKind.VIEWER_2D:
            return False
        elif selection is not None and selection.kind is ContextKind.BROWSE:
            self._set_browse_navigation(navigation)
        else:
            self._set_acquisition_navigation(navigation)
        return True

    def retirement_matches(self, receipt: DisplayRetirementReceipt) -> bool:
        return (
            type(receipt) is DisplayRetirementReceipt
            and receipt.cleanup_status is CleanupStatus.CLEANED
            and receipt.run_identity is self._run_identity
        )

    def apply_display_retirement(
        self, receipt: DisplayRetirementReceipt
    ) -> bool:
        if not self.retirement_matches(receipt):
            return False
        if self._run_identity is None:
            return self._acquisition is None
        self.clear_full_raw()
        self.clear_browse(select_acquisition=False)
        self._selection = None
        if self._acquisition is not None:
            self._acquisition.retire()
        self._acquisition = None
        self._set_acquisition_navigation(FrameNavigationProjection())
        self._run_identity = None
        self._pending_replacement = None
        self._reset_trace_projection()
        return True

    def release_acquisition(self, identity: RunIdentity | None) -> bool:
        if identity is None:
            return self._acquisition is None and self._run_identity is None
        if identity is not self._run_identity or self._acquisition is None:
            return False
        self.clear_full_raw()
        self._acquisition.retire()
        self._acquisition = None
        self._set_acquisition_navigation(FrameNavigationProjection())
        self._run_identity = None
        if (
            self._selection is not None
            and self._selection.kind is ContextKind.ACQUISITION
        ):
            self._selection = None
        self._pending_replacement = None
        self._reset_trace_projection()
        return True

    def select_acquisition(self) -> DisplaySelection:
        if self._acquisition is None:
            raise RuntimeError("no acquisition context")
        return self._select(self._acquisition)

    def select_browse(self) -> DisplaySelection:
        if self._browse is None or self._browse.invalidated:
            raise RuntimeError("no selectable browse context")
        return self._select(self._browse)

    def select_browser_target(self, identifier: str) -> bool:
        if type(identifier) is not str or not identifier:
            return False
        pending = self._pending_replacement
        if pending is not None:
            acquisition = self._acquisition
            held = {
                pending.selection.context_token,
                pending.selection.source_path,
                *(frame.artifact for frame in pending.navigation.frames),
            }
            return (
                identifier in held
                or acquisition is not None
                and identifier == acquisition.context_token
            )
        browse = self._browse
        if (
            browse is not None
            and not browse.invalidated
            and not browse.released
            and identifier in {
                browse.context_token,
                browse.requested_path,
            }
        ):
            self.select_browse()
            self._select_artifact_navigation(identifier, browse=True)
            return True
        acquisition = self._acquisition
        if acquisition is not None and (
            identifier == acquisition.context_token
            or any(
                frame.artifact == identifier
                for frame in self._acquisition_navigation.frames
            )
        ):
            self.select_acquisition()
            self._select_artifact_navigation(identifier, browse=False)
            return True
        return False

    def project_request(
        self,
        frame: object,
        *,
        require_complete: bool = True,
    ) -> ProjectionRequest:
        identity = self._selected_projection_identity()
        if (
            identity is None
            or self._selection is None
            or not self.owns_frame(frame)
            or type(require_complete) is not bool
        ):
            raise RuntimeError("no display selection")
        return ProjectionRequest(
            identity,
            self._selection,
            frame,
            require_complete,
        )

    def invalidate_browse_pass(self) -> None:
        self._browse_pass = None

    def _browse_scope_qualified(
        self, selection, current, generation, owner
    ) -> bool:
        """THE one exact-scope predicate (§22.3), requalified around every
        miss submit, qualified install and pass reuse: the exact live view, a
        loaded non-invalidated non-released Browse behind an open gate, still
        named by its selection and bound owner, with nothing pending."""
        browse = self._browse
        if browse is None or type(browse) is not BrowseContext:
            return False
        return (
            self._pending_replacement is None
            and type(owner) is _BrowseHydrationOwner
            and selection is not None
            and selection is self._selection
            and selection.kind is ContextKind.BROWSE
            and current is not None
            and current is self._browse_navigation.current
            and generation == selection.display_generation
            and browse.loaded
            and not browse.invalidated
            and not browse.released
            and not browse.commit_gate.cancelled
            and selection.names(browse)
            and owner.names(browse)
        )

    def _resolve_browse_pass(
        self, projection: ContextProjection, request: ProjectionRequest, owner
    ) -> _BrowseProjectionPass | None:
        """One-read resolve of the exact anchored current-Browse request: a
        miss is scope-requalified before and after its verbatim submit, a
        payload before install; a terminal truthfully records a dead/foreign
        scope, so it rechecks only exact view identity.  Any exception,
        malformed result, refused submit or failed qualification: NO pass."""
        self._browse_pass = None
        selection = request.selection
        current = request.frame
        generation = selection.display_generation
        browse = self._browse
        try:
            resolution = projection.resolve_browse(
                browse, request, self._selection,
                self._browse_navigation.current, owner,
            )
        except Exception:
            return None
        resolution = _qualified_browse_resolution(
            resolution, request.run_identity, current, generation, browse)
        if resolution is None:
            return None
        if type(resolution) is HydrationEligibleMiss:
            if not self._browse_scope_qualified(
                selection, current, generation, owner
            ):
                return None
            try:
                admitted = owner.submit(resolution.request)
            except Exception:
                return None
            if admitted is None:
                return None
        if type(resolution) is TerminalMiss:
            if (
                selection is not self._selection
                or current is not self._browse_navigation.current
                or generation != selection.display_generation
                or self._pending_replacement is not None
            ):
                return None
        elif not self._browse_scope_qualified(
            selection, current, generation, owner
        ):
            return None
        snapshot = _BrowseProjectionPass(
            selection, current, generation, resolution)
        self._browse_pass = snapshot
        return snapshot

    def resolve_projection(
        self,
        projection: ContextProjection,
        request: ProjectionRequest,
        browse_hydration_owner=None,
        viewer_1d_owner=None,
    ) -> StandardDisplayPayload | None:
        identity = self._selected_projection_identity()
        if (
            type(request) is not ProjectionRequest
            or request.run_identity is not identity
            or request.selection is not self._selection
        ):
            return None
        context = (
            self._acquisition
            if request.selection.kind is ContextKind.ACQUISITION
            else self._viewer_1d
            if request.selection.kind is ContextKind.VIEWER_1D
            else self._viewer_2d
            if request.selection.kind is ContextKind.VIEWER_2D
            else self._browse
        )
        if context is None:
            return None
        if (
            request.selection.kind is ContextKind.BROWSE
            and request.require_complete
            and request.frame is self._browse_navigation.current
        ):
            # THE authoritative current-Browse path: a disqualified held pass
            # clears and fails this call closed; a qualified one repaints with
            # no reread; anything else re-resolves once and is consumed HERE.
            snapshot = self._browse_pass
            if snapshot is not None and not self._browse_scope_qualified(
                snapshot.selection, snapshot.current,
                snapshot.generation, browse_hydration_owner,
            ):
                self._browse_pass = None
                payload = None
            else:
                if (snapshot is None
                        or type(snapshot.resolution) is not QualifiedPayload):
                    snapshot = self._resolve_browse_pass(
                        projection, request, browse_hydration_owner)
                resolution = None if snapshot is None else snapshot.resolution
                payload = (resolution.payload
                           if type(resolution) is QualifiedPayload else None)
        else:
            payload = projection.project(
                context,
                request,
                self._selection,
                identity,
                self._selected_frame_by_id(),
                browse_hydration_owner,
                viewer_1d_owner=viewer_1d_owner,
                viewer_catalog=self._viewer_2d_catalog,
                viewer_frame=self._viewer_2d_frame,
            )
        if (
            type(payload) is not StandardDisplayPayload
            or not display_payload_is_valid(
                payload, request.run_identity, request.frame,
                request.selection.display_generation)
        ):
            return None
        return payload

    def _capture_norm_aggregate(self) -> None:
        """THE one §25.3 capture: at most ONE candidate from the exact
        current selection, admitted only through the Q2 acceptance gate.

        A dead, cancelled or stale live-Browse scope and a rescoped
        acquisition owner are capture NO-OPS — a refusal cannot admit a
        candidate.  Within the owned context the held object survives the
        refusal untouched; a held identity FOREIGN to the owned
        context clears on the refused switch (§27.2, §28).  A context switch
        never exposes the prior identity's aggregate; revision 0 and older
        revisions are refused; an equal-revision replay keeps the exact
        held object.
        """
        if self._pending_replacement is not None:
            return
        selection = self._selection
        expected = None
        candidate = None
        if selection is not None and selection.kind is ContextKind.BROWSE:
            browse = self._browse
            if browse is not None:
                if (
                    browse.invalidated
                    or browse.released
                    or not browse.loaded
                    or browse.commit_gate.cancelled
                    or not selection.names(browse)
                ):
                    # §27.2: a refusal is a presentation no-op only INSIDE
                    # the currently owned context.  A held aggregate whose
                    # identity is foreign to the owned context must not
                    # survive a refused switch — Browse frame coverage
                    # compares scan/path, so a stale prior-token hold
                    # would divide the new context's frames.
                    held = self._norm_aggregate
                    if held is not None and held.identity != (
                        browse.context_token,
                        browse.scan_key,
                        browse.requested_path,
                    ):
                        self._norm_aggregate = None
                    return
                expected = (
                    browse.context_token,
                    browse.scan_key,
                    browse.requested_path,
                )
                candidate = browse.norm_aggregate
        elif selection is not None and selection.kind in {
                ContextKind.VIEWER_1D, ContextKind.VIEWER_2D}:
            self._norm_aggregate = None
            return
        elif selection is not None:
            context = self._acquisition
            identity = self._run_identity
            current = self._acquisition_navigation.current
            if (
                context is not None
                and identity is not None
                and current is not None
            ):
                if selection.owner != context.hydration_owner:
                    held = self._norm_aggregate
                    if held is not None and held.identity != (
                        identity.generation, identity.fingerprint,
                        str(current.artifact), current.source_scan,
                    ):
                        self._norm_aggregate = None
                    return
                expected = (
                    identity.generation,
                    identity.fingerprint,
                    str(current.artifact),
                    current.source_scan,
                )
                try:
                    candidate = (
                        context.publication_store.frame_norm_aggregate(
                            current
                        )
                    )
                except Exception:
                    candidate = None
        held = self._norm_aggregate
        if held is not None and (
            expected is None or held.identity != expected
        ):
            held = None
        if (
            expected is not None
            and type(candidate) is ScanNormAggregate
            and candidate.revision > 0
            and accepts_norm_aggregate(held, candidate, expected)
        ):
            held = candidate
        self._norm_aggregate = held

    def project_navigation(
        self,
        projection: ContextProjection,
        *,
        preferences: object | None = None,
        processing_mode: str = "Int 2D",
        live_update: bool = False,
        browse_hydration_owner=None,
        viewer_1d_owner=None,
    ) -> tuple[StandardDisplayPayload, ...]:
        self._consume_norm_capture_for_projection()
        payloads: list[StandardDisplayPayload] = []
        navigation = self.navigation
        selected = navigation.selected
        browse_selected = (
            self._selection is not None
            and self._selection.kind is ContextKind.BROWSE
        )
        viewer_1d = (self._selection is not None
                     and self._selection.kind is ContextKind.VIEWER_1D)
        accumulating = (not viewer_1d and
            preferences is not None
            and getattr(preferences, "plot_mode", None)
            in {"Overlay", "Waterfall"}
        )
        pending: _PendingTraceProjection | None = None
        if viewer_1d:
            frames, pending = list(selected), None
        elif accumulating:
            scope = self._trace_projection_scope(
                preferences,
                processing_mode,
            )
            reseed = scope != self._committed_trace_scope
            committed_ids = {
                id(frame) for frame in self._committed_trace_selection
            }
            planned = (
                selected
                if reseed
                else tuple(
                    frame for frame in selected
                    if id(frame) not in committed_ids
                )
            )
            pending = _PendingTraceProjection(
                scope,
                selected,
            )
            frames = list(planned)
            if reseed:
                # A semantic reseed must rebuild retained pin recipes even
                # after their source frame leaves selected/current.  This is
                # deliberately absent from the steady suffix cadence path.
                owned_by_id = self._selected_frame_by_id()
                planned_ids = {id(frame) for frame in frames}
                for pin in getattr(preferences, "slice_pins", ()):
                    if type(pin) is not SlicePin:
                        continue
                    frame = pin.frame
                    frame_id = id(frame)
                    if (
                        owned_by_id.get(frame_id) is frame
                        and frame_id not in planned_ids
                    ):
                        frames.append(frame)
                        planned_ids.add(frame_id)
        else:
            if preferences is not None:
                # Leaving an accumulating mode invalidates its projection
                # ledger.  Non-view consumers still require a complete
                # Single/Average/Sum projection; changing that API to a delta
                # would need a separate projection-owner redesign.
                self._reset_trace_projection()
            else:
                self._pending_trace_projection = None
            frames = list(selected)
        current = navigation.current
        if (
            current is not None
            and not any(frame is current for frame in frames)
        ):
            frames.append(current)
        for frame in frames:
            require_complete = frame is current
            try:
                request = self.project_request(
                    frame,
                    require_complete=require_complete,
                )
                payload = self.resolve_projection(
                    projection, request, browse_hydration_owner
                    , viewer_1d_owner
                )
                if payload is None and require_complete and not browse_selected:
                    if (self._selection is not None and
                            self._selection.kind is ContextKind.VIEWER_2D):
                        continue
                    # E4's incomplete-acquisition fallback only: a typed
                    # Browse miss/terminal never buys a second current read.
                    request = self.project_request(
                        frame,
                        require_complete=False,
                    )
                    payload = self.resolve_projection(
                        projection, request, browse_hydration_owner
                        , viewer_1d_owner
                    )
            except (RuntimeError, TypeError):
                payload = None
            if type(payload) is StandardDisplayPayload:
                payloads.append(payload)
        self._pending_trace_projection = pending
        return tuple(payloads)

    def project_browse_1d_cache(
        self,
        browse_hydration_owner,
        *,
        preferences: object,
        was_waterfall_active: bool,
    ) -> Browse1DRuntimeProjection:
        """Project the one planned sparse cache scope for display adoption.

        This method does not touch the legacy trace ledger or display state.
        A COMPLETE return retains its exact cache borrows for the caller;
        INCOMPLETE starts only the already-planned sparse row read.
        """

        self._consume_norm_capture_for_projection()
        browse = self._browse
        selection = self._selection
        navigation = self._browse_navigation
        owner = browse_hydration_owner
        planned = plan_browse_1d_targets(
            browse,
            selection,
            navigation,
            preferences,
            self._browse_frame_by_id,
            current_selection=self._selection,
            current_navigation=self._browse_navigation,
            was_waterfall_active=was_waterfall_active,
        )
        if planned.status is not Browse1DTargetPlanStatus.PLANNED:
            return Browse1DRuntimeProjection(
                Browse1DProjectionStatus.REFUSED,
                diagnostic=planned.diagnostic,
            )
        plan = planned.plan
        if (
            type(plan) is not Browse1DTargetPlan
            or not self._browse_1d_plan_is_current(
                browse, selection, navigation, owner, plan,
            )
        ):
            return Browse1DRuntimeProjection(
                Browse1DProjectionStatus.REFUSED,
                diagnostic="Browse 1-D runtime scope is stale",
            )
        try:
            projected = owner.project_1d(
                browse,
                selection,
                navigation,
                plan.display_targets,
                current_selection=self._selection,
            )
        except BaseException as error:
            return Browse1DRuntimeProjection(
                Browse1DProjectionStatus.REFUSED,
                plan=plan,
                diagnostic=f"{type(error).__name__}: {error}"[:512],
            )
        if type(projected) is not Browse1DProjectionOutcome:
            return Browse1DRuntimeProjection(
                Browse1DProjectionStatus.REFUSED,
                plan=plan,
                diagnostic="Browse 1-D projector returned a foreign result",
            )
        bundle = projected.borrow_bundle
        current = self._browse_1d_plan_is_current(
            browse, selection, navigation, owner, plan,
        )
        aligned = (
            projected.status is not Browse1DProjectionStatus.COMPLETE
            or len(projected.payloads) == len(plan.display_targets)
            and all(
                payload.frame_key is frame
                and payload.selection_generation
                == selection.display_generation
                for payload, frame in zip(
                    projected.payloads,
                    plan.display_targets,
                    strict=True,
                )
            )
        )
        if not current or not aligned:
            return Browse1DRuntimeProjection(
                Browse1DProjectionStatus.REFUSED,
                plan=plan,
                borrow_bundle=bundle,
                diagnostic="Browse 1-D projection completed outside its scope",
            )
        if projected.status is Browse1DProjectionStatus.COMPLETE:
            return Browse1DRuntimeProjection(
                Browse1DProjectionStatus.COMPLETE,
                plan,
                projected.payloads,
                bundle,
            )
        if projected.status is Browse1DProjectionStatus.REFUSED:
            return Browse1DRuntimeProjection(
                Browse1DProjectionStatus.REFUSED,
                plan=plan,
                borrow_bundle=bundle,
                diagnostic=projected.diagnostic,
            )
        if bundle is not None:
            return Browse1DRuntimeProjection(
                Browse1DProjectionStatus.REFUSED,
                plan=plan,
                borrow_bundle=bundle,
                diagnostic="Incomplete Browse projection retained borrow custody",
            )
        try:
            submission = owner.submit_1d(
                browse,
                selection,
                plan.display_targets,
            )
        except BaseException as error:
            return Browse1DRuntimeProjection(
                Browse1DProjectionStatus.REFUSED,
                plan=plan,
                diagnostic=f"{type(error).__name__}: {error}"[:512],
            )
        if (
            type(submission) is not object
            or not self._browse_1d_plan_is_current(
                browse, selection, navigation, owner, plan,
            )
        ):
            return Browse1DRuntimeProjection(
                Browse1DProjectionStatus.REFUSED,
                plan=plan,
                diagnostic="Browse 1-D hydration admission was refused or stale",
            )
        return Browse1DRuntimeProjection(
            Browse1DProjectionStatus.INCOMPLETE,
            plan=plan,
            submission_identity=submission,
            diagnostic=projected.diagnostic,
        )

    def _browse_1d_plan_is_current(
        self,
        browse,
        selection,
        navigation,
        owner,
        plan,
    ) -> bool:
        """Requalify exact runtime identities without reading any store."""

        return bool(
            self._pending_replacement is None
            and type(browse) is BrowseContext
            and browse is self._browse
            and type(selection) is DisplaySelection
            and selection is self._selection
            and selection.kind is ContextKind.BROWSE
            and navigation is self._browse_navigation
            and type(owner) is _BrowseHydrationOwner
            and owner.names(browse)
            and type(plan) is Browse1DTargetPlan
            and plan.selection is selection
            and plan.navigation is navigation
            and all(
                self._browse_frame_by_id.get(id(frame)) is frame
                for frame in plan.logical_frames
            )
            and all(
                self._browse_frame_by_id.get(id(frame)) is frame
                for frame in plan.display_targets
            )
            and browse.loaded
            and not browse.invalidated
            and not browse.released
            and not browse.commit_gate.cancelled
        )

    def browse_1d_plan_is_current(
        self,
        browse_hydration_owner,
        plan: object,
    ) -> bool:
        """Requalify one detached plan immediately before presentation."""

        return self._browse_1d_plan_is_current(
            self._browse,
            self._selection,
            self._browse_navigation,
            browse_hydration_owner,
            plan,
        )

    def capture_norm_aggregate_for_refresh(self) -> None:
        """Capture once for the frozen selection used by one shell refresh."""

        self._capture_norm_aggregate()
        self._norm_capture_scope = (
            self._selection,
            self._browse,
            self._acquisition,
        )

    def _consume_norm_capture_for_projection(self) -> None:
        scope = (self._selection, self._browse, self._acquisition)
        held = self._norm_capture_scope
        if (
            held is None
            or any(
                prior is not current
                for prior, current in zip(held, scope, strict=True)
            )
        ):
            self.capture_norm_aggregate_for_refresh()
        self._norm_capture_scope = None

    def project_background_contributors(
        self, projection: ContextProjection, browse_hydration_owner=None,
        viewer_1d_owner=None, *, pins=(),
    ) -> tuple[StandardDisplayPayload, ...]:
        """Resolve selected contributors followed by owned pin-only targets."""
        if (self._pending_replacement is not None or self._selection is None
                or type(pins) is not tuple):
            return ()
        selected = self.navigation.selected
        frames = list(selected)
        if not frames:
            return ()
        owned = self._selected_frame_by_id(); frame_ids = {id(frame) for frame in frames}
        for pin in pins:
            if (type(pin) is SlicePin and owned.get(id(pin.frame)) is pin.frame
                    and id(pin.frame) not in frame_ids):
                frames.append(pin.frame); frame_ids.add(id(pin.frame))
        payloads: list[StandardDisplayPayload] = []
        for index, frame in enumerate(frames):
            try:
                request = self.project_request(
                    frame, require_complete=index < len(selected))
                payload = self.resolve_projection(
                    projection, request, browse_hydration_owner,
                    viewer_1d_owner,
                )
            except (RuntimeError, TypeError):
                return ()
            if type(payload) is not StandardDisplayPayload:
                return ()
            payloads.append(payload)
        return tuple(payloads)

    def reseed_background_projection(self) -> None:
        self._reset_trace_projection()

    def commit_navigation_projection(
        self,
        presented_frames: tuple[DisplayFrameKey, ...],
    ) -> bool:
        """Acknowledge the exact selected identities accepted by the shell."""

        pending = self._pending_trace_projection
        if pending is None or type(presented_frames) is not tuple:
            return False
        target_by_id = {id(frame): frame for frame in pending.target}
        if any(
            type(frame) is not DisplayFrameKey
            or target_by_id.get(id(frame)) is not frame
            for frame in presented_frames
        ):
            return False
        presented_ids = {id(frame) for frame in presented_frames}
        committed = tuple(
            frame
            for frame in pending.target
            if id(frame) in presented_ids
        )
        self._committed_trace_scope = pending.scope
        self._committed_trace_selection = committed
        self._pending_trace_projection = None
        return True

    def commit_rebound_navigation_projection(
        self,
        presented_frames: tuple[DisplayFrameKey, ...],
        *,
        preferences: object,
        processing_mode: str,
    ) -> bool:
        """Acknowledge a semantically identical terminal-Browse rebind."""

        selected = self.navigation.selected
        plot_mode = getattr(preferences, "plot_mode", None)
        if (
            self._selection is None
            or self._selection.kind is not ContextKind.BROWSE
            or type(presented_frames) is not tuple
            or len(presented_frames) != len(selected)
            or any(
                presented is not expected
                for presented, expected in zip(
                    presented_frames, selected, strict=True,
                )
            )
            or plot_mode not in {"Single", "Overlay", "Waterfall"}
        ):
            return False
        if plot_mode == "Single":
            if (
                len(selected) != 1
                or self.navigation.current is not selected[0]
            ):
                return False
            self._reset_trace_projection()
            return True
        self._committed_trace_scope = self._trace_projection_scope(
            preferences, processing_mode,
        )
        self._committed_trace_selection = selected
        self._pending_trace_projection = None
        return True

    def qualify_display_event(
        self,
        projection: ContextProjection,
        event: StandardRunEvent,
        browse_hydration_owner=None,
    ) -> StandardDisplayPayload | None:
        selection = self._selection
        identity = self._selected_projection_identity()
        frame = qualified_event_frame(self, event)
        if (
            type(event) is not StandardRunEvent
            or event.kind is not StandardEventKind.DISPLAY_READY
            or event.run_identity is not identity
            or selection is None
            or type(frame) is not DisplayFrameKey
            or event.selection_generation != selection.display_generation
        ):
            return None
        request = ProjectionRequest(event.run_identity, selection, frame)
        return self.resolve_projection(
            projection, request, browse_hydration_owner
        )

    def adopt_browse(
        self, context: BrowseContext, request: BrowseLoadRequest
    ) -> DisplaySelection:
        pending = self._pending_replacement
        if (
            type(context) is not BrowseContext
            or type(request) is not BrowseLoadRequest
            or context.operation is not request
            or context.load_request is not request
            or not context.matches(request.token, request.load_generation)
            or pending is not None
            and pending.request is not request
        ):
            raise RuntimeError("Browse replacement identity is not exact")
        self.clear_full_raw()
        identity = self._run_identity or RunIdentity(
            request.load_generation, request.token
        )
        frames = tuple(
            DisplayFrameKey(
                identity, context.scan_key, context.requested_path,
                int(label), ordinal,
            )
            for ordinal, label in enumerate(context.frame_ids, 1)
        )
        current = frames[0] if frames else None
        selected = () if current is None else (current,)
        navigation = FrameNavigationProjection(frames, current, selected)
        generation = self._display_generation + 1
        selection = DisplaySelection.for_context(context, generation)
        self._display_generation = generation
        self._browse = context
        self._browse_projection_identity = identity
        self._set_browse_navigation(navigation)
        self._selection = selection
        self._pending_replacement = None
        return selection

    def begin_replacement(
        self, request: BrowseLoadRequest
    ) -> _PendingBrowseReplacement:
        browse = self._browse
        selection = self._selection
        if (
            type(request) is not BrowseLoadRequest
            or browse is None
            or not browse.released
            or selection is None
            or not selection.names(browse)
        ):
            raise RuntimeError("no released Browse presentation to hold")
        pending = _PendingBrowseReplacement(
            request, selection, self._browse_navigation)
        self._browse = None
        self._set_browse_navigation(FrameNavigationProjection())
        self._selection = None
        self._pending_replacement = pending
        return pending

    def retarget_replacement(
        self,
        expected: BrowseLoadRequest,
        replacement: BrowseLoadRequest,
    ) -> bool:
        pending = self._pending_replacement
        if pending is None or pending.request is not expected:
            return False
        self._pending_replacement = _PendingBrowseReplacement(
            replacement, pending.selection, pending.navigation)
        return True

    def finish_replacement(self, request: BrowseLoadRequest) -> bool:
        pending = self._pending_replacement
        if pending is None or pending.request is not request:
            return False
        self.invalidate_browse_pass()
        self._pending_replacement = None
        self._selection = None
        self._browse_projection_identity = None
        if self._acquisition is not None:
            self.select_acquisition()
        return True

    def invalidate_browse(self) -> None:
        self.invalidate_browse_pass()
        if self._browse is not None:
            self._browse.invalidate()

    def clear_browse(self, *, select_acquisition: bool) -> None:
        browse = self._browse
        if (
            select_acquisition
            and browse is not None
            and self._selection is not None
            and self._selection.names(browse)
            and self._acquisition is not None
        ):
            self.select_acquisition()
        self._browse = None
        self._browse_projection_identity = None
        self._set_browse_navigation(FrameNavigationProjection())
        self._reset_trace_projection()

    def close_selection(self) -> None:
        self.clear_full_raw()
        self.clear_browse(select_acquisition=False)
        self.clear_viewer_1d()
        self.clear_viewer_2d(release=True)
        self._selection = None
        self._pending_replacement = None

    def require_identity(self) -> RunIdentity:
        if self._run_identity is None:
            raise RuntimeError("no acquisition identity")
        return self._run_identity

    def _selected_projection_identity(self) -> RunIdentity | None:
        selection = self._selection
        if selection is None:
            return None
        if selection.kind is ContextKind.VIEWER_1D:
            current = self._viewer_1d_navigation.current
            return None if current is None else current.run_identity
        if selection.kind is ContextKind.VIEWER_2D:
            current = self._viewer_2d_navigation.current
            return None if current is None else current.run_identity
        return (self._browse_projection_identity
                if selection.kind is ContextKind.BROWSE else self._run_identity)

    def _select(
        self, context: AcquisitionContext | BrowseContext
    ) -> DisplaySelection:
        self.invalidate_browse_pass()
        if type(context) is BrowseContext: self.clear_full_raw()
        self._display_generation += 1
        selection = DisplaySelection.for_context(
            context, self._display_generation)
        navigation = (
            self._acquisition_navigation
            if type(context) is AcquisitionContext
            else self._browse_navigation
        )
        navigation = FrameNavigationProjection(
            *_normalized_navigation(
                navigation.frames, navigation.current, navigation.selected,
                prefer_last=False))
        if type(context) is AcquisitionContext:
            self._set_acquisition_navigation(navigation)
        else:
            self._set_browse_navigation(navigation)
        self._selection = selection
        self._pending_replacement = None
        return selection

    def _select_artifact_navigation(
        self, identifier: str, *, browse: bool
    ) -> None:
        navigation = (
            self._browse_navigation
            if browse
            else self._acquisition_navigation
        )
        matching = tuple(
            frame
            for frame in navigation.frames
            if frame.artifact == identifier
        )
        if not matching:
            return
        current = matching[0] if browse else matching[-1]
        replacement = FrameNavigationProjection(
            navigation.frames, current, (current,))
        if browse:
            self._set_browse_navigation(replacement)
        else:
            self._set_acquisition_navigation(replacement)

    def _selected_frame_by_id(self) -> dict[int, DisplayFrameKey]:
        selection = self._selection
        if selection is None:
            return {}
        return (
            self._viewer_1d_frame_by_id
            if selection.kind is ContextKind.VIEWER_1D
            else
            self._viewer_2d_frame_by_id
            if selection.kind is ContextKind.VIEWER_2D
            else self._browse_frame_by_id
            if selection.kind is ContextKind.BROWSE
            else self._acquisition_frame_by_id
        )

    def _set_acquisition_navigation(
        self,
        navigation: FrameNavigationProjection,
    ) -> None:
        prior_frames = self._acquisition_navigation.frames
        if navigation.current is not self._acquisition_navigation.current \
                and self._acquisition is not None:
            self._acquisition.publication_store.invalidate_full_demand()
        self._acquisition_navigation = navigation
        if navigation.frames is not prior_frames:
            self._acquisition_frame_by_id = {
                id(frame): frame for frame in navigation.frames
            }

    def _set_browse_navigation(
        self,
        navigation: FrameNavigationProjection,
    ) -> None:
        self.invalidate_browse_pass()
        prior_frames = self._browse_navigation.frames
        self._browse_navigation = navigation
        if navigation.frames is not prior_frames:
            self._browse_frame_by_id = {
                id(frame): frame for frame in navigation.frames
            }

    def _set_viewer_2d_navigation(
        self, navigation: FrameNavigationProjection) -> None:
        prior = self._viewer_2d_navigation.frames
        self._viewer_2d_navigation = navigation
        if navigation.frames is not prior:
            self._viewer_2d_frame_by_id = {id(frame): frame
                                           for frame in navigation.frames}

    def _set_viewer_1d_navigation(
        self, navigation: FrameNavigationProjection) -> None:
        prior = self._viewer_1d_navigation.frames
        self._viewer_1d_navigation = navigation
        if navigation.frames is not prior:
            self._viewer_1d_frame_by_id = {id(frame): frame
                                           for frame in navigation.frames}

    def _trace_projection_scope(
        self,
        preferences: object,
        processing_mode: str,
    ) -> tuple[object, ...]:
        selection = self._selection
        identity = self._selected_projection_identity()
        # The SAME resolution the projection consumes.  Aggregate revision is
        # provenance, not a numeric trace-cache key: an appended frame cannot
        # change an earlier frame's own metadata divisor.  A changed identity
        # or effective channel still reseeds the complete selected set.
        norm_identity, _norm_revision, effective_channel, _ = (
            resolve_norm_presentation(
                self._norm_aggregate,
                getattr(preferences, "norm_channel", None),
                self.navigation.selected,
            )
        )
        return (
            id(identity),
            None if selection is None else selection.kind,
            None if selection is None else selection.context_token,
            # Plot mode is a presentation choice; the per-frame scientific
            # projection is the same in Single, Overlay, Average, and Sum.
            processing_mode,
            getattr(preferences, "plot_axis", None),
            getattr(preferences, "share_axis", None),
            getattr(preferences, "image_axis", None),
            getattr(preferences, "slice_enabled", None),
            getattr(preferences, "slice_center", None),
            getattr(preferences, "slice_width", None),
            tuple(
                pin.projection_id
                for pin in getattr(preferences, "slice_pins", ())
            ),
            *trace_normalization_scope(
                norm_identity,
                effective_channel,
            ),
        )

    def _reset_trace_projection(self) -> None:
        self._committed_trace_scope = None
        self._committed_trace_selection = ()
        self._pending_trace_projection = None
