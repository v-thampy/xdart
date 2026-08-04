"""One concrete hydration owner bound to one admitted Browse context."""

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

from .browse_values import BrowseCleanupReceipt, BrowseLoadRequest
from .display_runtime import (
    DetectorHydrationOutcome,
    publication_needs_hydration,
)
from .events import CleanupStatus
from .hydration_transport import HydrationTransport, PreparedHydrationCommit


class _BrowseHydrationOwner:
    """Own or borrow one transport for one exact admitted Browse object."""

    def __init__(
        self,
        browse: BrowseContext,
        *,
        borrowed_transport: HydrationTransport | None = None,
    ) -> None:
        if type(browse) is not BrowseContext:
            raise TypeError("Browse hydration requires one exact context")
        self._browse = browse
        self._store = browse.publication_store
        self._gate = browse.commit_gate
        self._context_token = browse.context_token
        self._scan_key = browse.scan_key
        self._artifact = browse.requested_path
        self._scope = browse.hydration_owner.as_tuple()
        self._terminal_lock = Lock()
        self._terminal_reads: set[HydrationReadKey] = set()
        self._repaints = SimpleQueue()
        #: Exact tokens this owner admitted onto a BORROWED transport and
        #: whose completion it has not yet observed.  The borrowed transport's
        #: sink belongs to the acquisition owner, so the Browse completion is
        #: read from its public snapshot instead — by exact token, never by a
        #: deque offset, which rotation would invalidate.
        self._borrowed_tokens: set[HydrationToken] = set()
        self._owns_transport = borrowed_transport is None
        self.transport = (
            HydrationTransport(
                self.commit_preview,
                self.derive_target,
                completion_sink=self._complete,
            )
            if borrowed_transport is None
            else borrowed_transport
        )

    @property
    def owns_transport(self) -> bool:
        return self._owns_transport

    def _owns(self, browse: BrowseContext) -> bool:
        return (
            type(browse) is BrowseContext
            and browse is self._browse
            and browse.publication_store is self._store
            and browse.commit_gate is self._gate
            and browse.context_token == self._context_token
            and browse.scan_key == self._scan_key
            and browse.requested_path == self._artifact
        )

    def names(self, browse: BrowseContext) -> bool:
        return (
            self._owns(browse)
            and browse.hydration_owner.as_tuple() == self._scope
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
        return None, None

    def _read_key_is_exact(self, read_key: object) -> bool:
        return (
            type(read_key) is HydrationReadKey
            and read_key.scope == HydrationScope(*self._scope)
            and read_key.artifact_identity == self._artifact
            and read_key.purpose is HydrationPurpose.PREVIEW
        )

    def _terminalize(self, read_key: object) -> None:
        if self._read_key_is_exact(read_key):
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
                self._terminalize(read_key)
        self._repaints.put(None)

    def submit(self, request: HydrationRequest):
        if not self._request_is_exact(request):
            return None
        read_key = request.read_key
        with self._terminal_lock:
            if read_key in self._terminal_reads:
                return None
        token = self.transport.submit(request)
        if (
            not self._owns_transport
            and type(token) is HydrationToken
            and token == request.token
        ):
            # Retain the fact only for the exact token the transport admitted:
            # a refusal (None) or any other token is not this Browse read.
            with self._terminal_lock:
                self._borrowed_tokens.add(token)
        return token

    def detector_outcome(self, browse: BrowseContext, label):
        if (
            not self.names(browse)
            or browse.invalidated
            or browse.released
            or not browse.loaded
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
            raise TypeError("Browse preview requires one prepared commit")
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
            detector_unavailable = view.raw is None and view.thumbnail is None
            source_identity = f"{view.source_path or ''}#{view.source_frame_index}"
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
                self._terminalize(request.read_key)
        finally:
            gate.leave()
        return HydrationOutcome.HYDRATED

    def _observe_borrowed(self) -> None:
        """Turn this owner's own borrowed completions into owned-path wakes.

        Reads only the transport's public completion snapshot and matches by
        exact admitted token, so a foreign acquisition read, an equal-valued
        foreign Browse read and a superseded presentation generation are all
        inert.  Each exact token is consumed once, giving one repaint per
        terminal completion rather than an unbounded sequence.
        """

        with self._terminal_lock:
            if not self._borrowed_tokens:
                return
        for completion in self.transport.completions():
            if type(completion) is not HydrationCompletion:
                continue
            with self._terminal_lock:
                if completion.token not in self._borrowed_tokens:
                    continue
                self._borrowed_tokens.discard(completion.token)
            # Released before _complete: _terminalize takes the same lock.
            self._complete(completion)

    def consume_repaint(self) -> bool:
        if not self._owns_transport:
            self._observe_borrowed()
        consumed = False
        while True:
            try:
                self._repaints.get_nowait()
            except Empty:
                return consumed
            consumed = True

    def polling_needed(self) -> bool:
        if not self._owns_transport:
            # Never the shared worker: unrelated acquisition reads must not
            # keep this page awake.  Only this owner's own outstanding read
            # or an unconsumed wake does.
            with self._terminal_lock:
                awaiting = bool(self._borrowed_tokens)
            return awaiting or not self._repaints.empty()
        worker = self.transport.worker
        return (
            (worker is not None and worker.is_alive())
            or not self._repaints.empty()
        )

    def retire(self) -> bool:
        return (
            self.transport.retire(join_timeout=0.0)
            if self._owns_transport
            else True
        )

    def release(self, loader, browse: BrowseContext) -> BrowseCleanupReceipt:
        request = (
            browse.load_request
            if type(browse) is BrowseContext
            and type(browse.load_request) is BrowseLoadRequest
            else None
        )
        if not self._owns(browse):
            return BrowseCleanupReceipt(
                request, CleanupStatus.CLEANUP_PENDING
            )
        self.transport.cancel_gate(self._gate)
        if self.transport.retains_gate(self._gate):
            return BrowseCleanupReceipt(
                request, CleanupStatus.CLEANUP_PENDING
            )
        if not self.retire():
            return BrowseCleanupReceipt(
                request, CleanupStatus.CLEANUP_PENDING
            )
        return loader.release_context(browse)


__all__ = []
