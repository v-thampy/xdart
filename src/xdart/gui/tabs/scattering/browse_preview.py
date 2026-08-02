"""Typed Browse preview issuing and transport-gated release (E4-R2).

Transport-selection helpers the controller routes through own no lifecycle
state: the exact B context supplies the owner, store and gate carried by every
typed request.  A live acquisition supplies its already-composed transport; a
cold Browse store carries a subordinate fallback transport so the same request
remains hydratable without inventing an acquisition or lifecycle identity."""

from __future__ import annotations

from dataclasses import replace
from queue import Empty, SimpleQueue
from threading import Lock

from xdart.modules.display_context import BrowseContext, HydrationRequest
from xdart.modules.frame_publication import FramePublication
from xrd_tools.core import FrameRecord
from xrd_tools.session.hydration import (
    HydrationCompletion,
    HydrationOutcome,
    HydrationPurpose,
    HydrationReadKey,
    HydrationScope,
    HydrationToken,
)

from .browse_values import BrowseCleanupReceipt
from .context_projection import ProjectionRequest
from .display_runtime import (
    DetectorHydrationOutcome,
    publication_needs_hydration,
)
from .events import CleanupStatus
from .hydration_transport import HydrationTransport, PreparedHydrationCommit


_BROWSE_PREVIEW_OWNER = "_xdart_browse_preview_owner"


class _BrowsePreviewTransportOwner:
    """Commit one cold Browse transport only into its exact carried store."""

    def __init__(self, browse: BrowseContext) -> None:
        self._store = browse.publication_store
        self._gate = browse.commit_gate
        self._context_token = browse.context_token
        self._scan_key = browse.scan_key
        self._artifact = browse.requested_path
        self._scope = browse.hydration_owner.as_tuple()
        self._terminal_lock = Lock()
        self._terminal_reads: set[HydrationReadKey] = set()
        self._repaints = SimpleQueue()
        self.transport = HydrationTransport(
            self.commit_preview,
            self.derive_target,
            completion_sink=self._complete,
        )

    def names(self, browse: BrowseContext) -> bool:
        return (
            type(browse) is BrowseContext
            and browse.publication_store is self._store
            and browse.commit_gate is self._gate
            and browse.context_token == self._context_token
            and browse.scan_key == self._scan_key
            and browse.requested_path == self._artifact
        )

    def _request_is_exact(self, request: object) -> bool:
        return (
            type(request) is HydrationRequest
            and type(request.read_key) is HydrationReadKey
            and len(request.stores) == 1
            and request.stores[0] is self._store
            and request.commit_gate is self._gate
            and request.owner.as_tuple() == self._scope
            and request.read_key.scope == HydrationScope(*self._scope)
            and request.read_key.artifact_identity == self._artifact
        )

    def derive_target(self, _request: HydrationRequest):
        # Browse has no accepted detector-mask projection.  Exact target
        # qualification is repeated by the atomic commit after the disk read.
        return None, None

    def _read_key_is_exact(self, read_key: object) -> bool:
        return (
            type(read_key) is HydrationReadKey
            and read_key.scope == HydrationScope(*self._scope)
            and read_key.artifact_identity == self._artifact
            and read_key.purpose is HydrationPurpose.PREVIEW
        )

    def _terminalize(self, read_key: object) -> None:
        if not self._read_key_is_exact(read_key):
            return
        with self._terminal_lock:
            self._terminal_reads.add(read_key)

    def _complete(self, completion: HydrationCompletion) -> None:
        if (
            type(completion) is HydrationCompletion
            and completion.outcome is not HydrationOutcome.SUPERSEDED
        ):
            read_key = completion.token.read_key
            publication = self._store.get(read_key.frame_identity)
            if (
                completion.outcome
                not in {
                    HydrationOutcome.HYDRATED,
                    HydrationOutcome.ALREADY_RESIDENT,
                }
                or publication_needs_hydration(publication, None)
            ):
                # Every refused/failed exact terminal is suppressed.  A
                # successful complete read remains re-hydratable after a later
                # store eviction; only a successful but detector-incomplete
                # read is terminal for this immutable Browse context.
                self._terminalize(read_key)
        self._repaints.put(None)

    def submit(self, request: HydrationRequest):
        read_key = getattr(request, "read_key", None)
        if self._read_key_is_exact(read_key):
            with self._terminal_lock:
                if read_key in self._terminal_reads:
                    return None
        return self.transport.submit(request)

    def detector_outcome(self, browse: BrowseContext, label):
        if (
            not self.names(browse)
            or browse.invalidated
            or browse.released
            or not browse.loaded
            or browse.hydration_owner.as_tuple() != self._scope
        ):
            return None
        try:
            read_key = HydrationReadKey(
                HydrationScope(*self._scope),
                self._artifact,
                label,
                HydrationPurpose.PREVIEW,
            )
        except (TypeError, ValueError):
            return None
        with self._terminal_lock:
            return (
                DetectorHydrationOutcome.DETECTOR_UNAVAILABLE
                if read_key in self._terminal_reads
                else None
            )

    def commit_preview(
        self, prepared: PreparedHydrationCommit
    ) -> HydrationOutcome:
        if type(prepared) is not PreparedHydrationCommit:
            raise TypeError(
                "cold Browse preview requires one prepared commit"
            )
        request = prepared.request
        preview = prepared.preview
        if (
            preview.read_key != request.read_key
            or not self._request_is_exact(request)
        ):
            return HydrationOutcome.OWNER_MISMATCH
        gate = request.commit_gate
        if not gate.enter(request.epoch):
            return (
                HydrationOutcome.CANCELLED
                if bool(getattr(gate, "cancelled", False))
                else HydrationOutcome.OWNER_MISMATCH
            )
        try:
            view = preview.view
            if preview.raw is not None:
                view = replace(view, raw=preview.raw)
            record = FrameRecord.from_view(view)
            detector_unavailable = (
                view.raw is None and view.thumbnail is None
            )
            source_identity = (
                f"{view.source_path or ''}#{view.source_frame_index}"
            )
            committed = self._store.upsert(
                FramePublication(
                    view,
                    record=record,
                    source_identity=source_identity,
                    scan_key=request.owner.scan_key,
                )
            )
            if (
                detector_unavailable
                or publication_needs_hydration(committed, None)
            ):
                # A persisted 1D-only/source-pointer frame remains a truthful
                # light Browse payload, but cold Browse owns no detector-mask
                # projection with which to materialize raw.  Record the exact
                # completed read as terminal before waking the GUI so repaint
                # can show 1D without starting the same I/O again.
                self._terminalize(request.read_key)
        finally:
            gate.leave()
        return HydrationOutcome.HYDRATED

    def consume_repaint(self) -> bool:
        consumed = False
        while True:
            try:
                self._repaints.get_nowait()
            except Empty:
                return consumed
            consumed = True

    def polling_needed(self) -> bool:
        worker = self.transport.worker
        return (
            (worker is not None and worker.is_alive())
            or not self._repaints.empty()
        )

    def retire(self) -> bool:
        return self.transport.retire(join_timeout=0.0)


def _browse_preview_owner(
    browse: BrowseContext | None,
    *,
    create: bool,
) -> _BrowsePreviewTransportOwner | None:
    if type(browse) is not BrowseContext:
        return None
    store = browse.publication_store
    owner = getattr(store, _BROWSE_PREVIEW_OWNER, None)
    if type(owner) is _BrowsePreviewTransportOwner:
        return owner if owner.names(browse) else None
    if not create:
        return None
    owner = _BrowsePreviewTransportOwner(browse)
    try:
        setattr(store, _BROWSE_PREVIEW_OWNER, owner)
    except Exception:
        return None
    return owner


def _transport_for_browse(runtime, browse, *, create: bool):
    owner = _browse_preview_owner(browse, create=False)
    if owner is not None:
        return owner.transport
    acquisition = runtime.acquisition_context
    display = None if acquisition is None else acquisition.publication_store
    transport = getattr(display, "transport", None)
    if transport is not None or not create:
        return transport
    owner = _browse_preview_owner(browse, create=True)
    return None if owner is None else owner.transport


def display_transport(runtime):
    """Return A's transport or the exact cold B store's fallback transport."""

    return _transport_for_browse(
        runtime,
        runtime.browse_context,
        create=True,
    )


def _current_browse_preview_owner(runtime):
    browse = runtime.browse_context
    selection = runtime.selection
    if (
        type(browse) is not BrowseContext
        or browse.invalidated
        or browse.released
        or selection is None
        or not selection.names(browse)
    ):
        return None
    return _browse_preview_owner(browse, create=False)


def cold_browse_detector_outcome(
    browse: BrowseContext, label
) -> DetectorHydrationOutcome | None:
    """Return B's exact terminal detector fact without creating an owner."""

    owner = _browse_preview_owner(browse, create=False)
    return None if owner is None else owner.detector_outcome(browse, label)


def browse_preview_repaint_ready(runtime) -> bool:
    """Consume one cold-B completion wake for the exact current selection."""

    owner = _current_browse_preview_owner(runtime)
    return owner is not None and owner.consume_repaint()


def browse_preview_polling_needed(runtime) -> bool:
    """Whether a cold-B worker or unpainted completion still needs a tick."""

    owner = _current_browse_preview_owner(runtime)
    return owner is not None and owner.polling_needed()


def request_browse_preview(runtime, request: ProjectionRequest) -> None:
    """Submit one typed B ``PREVIEW`` for the exact current browse frame
    whose detector tier is missing/demoted; refuse everything else."""
    from xdart.modules.display_context import ContextKind

    selection = runtime.selection
    browse = runtime.browse_context
    frame = getattr(request, "frame", None)
    if (
        type(request) is not ProjectionRequest
        or not request.require_complete
        or selection is None
        or request.selection is not selection
        or selection.kind is not ContextKind.BROWSE
        or browse is None
        or browse.invalidated
        or browse.released
        or not browse.loaded
        or not runtime.owns_frame(frame)
        or frame.source_scan != browse.scan_key
        or frame.artifact != browse.requested_path
    ):
        return
    publication = browse.publication_store.get(frame.local_frame_label)
    if not publication_needs_hydration(publication, None):
        return
    transport = display_transport(runtime)
    owner = browse.hydration_owner
    if transport is None or not owner.qualified:
        return
    generation = selection.display_generation
    try:
        read_key = HydrationReadKey(
            HydrationScope(*owner.as_tuple()),
            browse.requested_path,
            frame.local_frame_label,
            HydrationPurpose.PREVIEW,
        )
        typed = HydrationRequest(
            frame.local_frame_label,
            HydrationPurpose.PREVIEW,
            generation,
            owner,
            (browse.publication_store,),
            browse.commit_gate,
            read_key=read_key,
            token=HydrationToken(read_key, generation),
        )
    except (TypeError, ValueError):
        return
    fallback = _browse_preview_owner(browse, create=False)
    if fallback is not None and transport is fallback.transport:
        fallback.submit(typed)
    else:
        transport.submit(typed)


def qualified_event_frame(runtime, event):
    """Resolve one DISPLAY_READY event to the exact owned frame it repaints:
    an acquisition event by its exact ``frame_key``; a key-less Browse
    repaint hint by the exact CURRENT browse selection when the live browse
    artifact matches."""
    from xdart.modules.display_context import ContextKind

    selection = runtime.selection
    frame = getattr(event, "frame_key", None)
    if frame is not None:
        if (
            selection is not None
            and selection.kind is ContextKind.ACQUISITION
            and runtime.owns_frame(frame)
        ):
            return frame
        return None
    browse = runtime.browse_context
    current = runtime.navigation.current
    if (
        selection is None
        or selection.kind is not ContextKind.BROWSE
        or browse is None
        or browse.invalidated
        or browse.released
        or getattr(event, "artifact", None) != browse.requested_path
    ):
        return None
    return current


def release_browse(
    loader, runtime, browse: BrowseContext
) -> BrowseCleanupReceipt:
    """Release B only after the transport drops its request-carried
    stores/gate; otherwise defer through a pending cleanup receipt."""
    fallback = _browse_preview_owner(browse, create=False)
    transport = _transport_for_browse(runtime, browse, create=False)
    if transport is not None:
        transport.cancel_gate(browse.commit_gate)
        if transport.retains_gate(browse.commit_gate):
            return BrowseCleanupReceipt(
                browse.load_request, CleanupStatus.CLEANUP_PENDING
            )
    if fallback is not None and not fallback.retire():
        return BrowseCleanupReceipt(
            browse.load_request, CleanupStatus.CLEANUP_PENDING
        )
    return loader.release_context(browse)


__all__ = [
    "browse_preview_polling_needed",
    "browse_preview_repaint_ready",
    "cold_browse_detector_outcome",
    "display_transport",
    "qualified_event_frame",
    "release_browse",
    "request_browse_preview",
]
