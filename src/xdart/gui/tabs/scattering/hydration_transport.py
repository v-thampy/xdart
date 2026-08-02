"""One bounded typed hydration transport for the vNext scattering display.

The single owner of background preview reads (E4-R): one active read, one
latest queued reference, worker-thread execution through the shared one-open
:func:`read_frame_preview`, and one terminal :class:`HydrationCompletion` for
every admitted token.  It replaces the deleted private flight protocol (whose
old module name may not appear here — the census oracle scans this package).
The queued reference carries only the typed ``HydrationRequest`` (owner, exact
stores, gate, read key, token), the frozen values-only projection derived ONCE
at submit from the exact carried target, presentation facts, and the resolved
catalog key — never arrays, open handles, widgets or callbacks.  The read
result is consumed only by the ONE owner-bound target port
(``RunDisplayState.commit_preview``), which validates under its own lock plus
the exact request-carried ``CommitGate`` and publishes last; a commit
exception retains that exact prepared commit for one idempotent retry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock, Thread, current_thread

from xdart.modules.display_context import HydrationRequest
from xrd_tools.io.frame_preview import (
    DetectorPreviewProjection,
    FramePreview,
    read_frame_preview,
)
from xrd_tools.session.hydration import (
    HydrationCompletion,
    HydrationOutcome,
    HydrationToken,
)

from .display_values import DisplayFrameKey


@dataclass(slots=True)
class _TransportEntry:
    """One admitted read: the request, its frozen projection, and the LATEST
    presentation token/closed facts (moved by same-read resubmission)."""

    request: HydrationRequest
    projection: DetectorPreviewProjection | None
    key: DisplayFrameKey | None
    closed: bool
    token: HydrationToken
    #: The token snapshot the commit was built with (worker-only writer).
    committed_token: HydrationToken | None = None


@dataclass(frozen=True, slots=True)
class PreparedHydrationCommit:
    """The one frozen value the target-port operation may consume."""

    request: HydrationRequest
    token: HydrationToken
    key: DisplayFrameKey | None
    closed: bool
    preview: FramePreview
    projection: DetectorPreviewProjection | None


class HydrationTransport:
    """Externally observable, internally synchronized single-lane transport."""

    def __init__(self, commit, derive, *, completion_sink=None) -> None:
        # commit/derive are owner-bound at construction, never queue-carried.
        self._commit = commit
        self._derive = derive
        self._completion_sink = completion_sink
        self._lock = RLock()
        self._active: _TransportEntry | None = None
        self._queued: _TransportEntry | None = None
        self._worker: Thread | None = None
        self._retired = False
        self._counters: dict[HydrationOutcome, int] = {
            outcome: 0 for outcome in HydrationOutcome
        }
        self._completions: deque[HydrationCompletion] = deque(maxlen=32)

    # -- observability ------------------------------------------------------ #

    @property
    def active_token(self) -> HydrationToken | None:
        with self._lock:
            return self._active.token if self._active is not None else None

    @property
    def queued_token(self) -> HydrationToken | None:
        with self._lock:
            return self._queued.token if self._queued is not None else None

    @property
    def worker(self) -> Thread | None:
        with self._lock:
            return self._worker

    def completions(self) -> tuple[HydrationCompletion, ...]:
        with self._lock:
            return tuple(self._completions)

    def counters(self) -> dict[HydrationOutcome, int]:
        with self._lock:
            return dict(self._counters)

    # -- admission ---------------------------------------------------------- #

    def submit(
        self, request: HydrationRequest, *, closed: bool = False
    ) -> HydrationToken | None:
        """Admit one typed artifact-bearing request, or refuse with ``None``.

        The inventoried canonical legacy shape (``read_key=None``) is refused:
        terminal-total accounting begins only for the typed E4 shape.
        """
        if (
            type(request) is not HydrationRequest
            or request.read_key is None
            or request.token is None
            or not request.enqueueable
        ):
            return None
        token = request.token
        with self._lock:
            if self._retired or request.commit_gate.cancelled:
                return None
            active = self._active
            if (
                active is not None
                and active.request.read_key == request.read_key
            ):
                # Same exact read: reuse it, move only the presentation token.
                displaced = active.token
                active.token = token
                active.closed = bool(closed)
                if displaced != token:
                    self._complete_locked(displaced, HydrationOutcome.SUPERSEDED)
                # The newest selection IS this active read: any older queued
                # entry is no longer latest and terminalizes SUPERSEDED now
                # (§20.4 — the active read itself is never cancelled).
                queued = self._queued
                if queued is not None:
                    self._queued = None
                    self._complete_locked(
                        queued.token, HydrationOutcome.SUPERSEDED
                    )
                return token
            projection, key = self._derive(request)
            entry = _TransportEntry(
                request, projection, key, bool(closed), token
            )
            queued = self._queued
            if queued is not None:
                self._complete_locked(queued.token, HydrationOutcome.SUPERSEDED)
            self._queued = entry
            self._ensure_worker_locked()
            return token

    # -- cancellation and retirement ---------------------------------------- #

    def cancel_gate(self, gate) -> None:
        """Drop the queued reference carrying *gate*; an active read is never
        forcibly interrupted — its cancelled gate refuses the commit instead."""
        with self._lock:
            queued = self._queued
            if queued is not None and queued.request.commit_gate is gate:
                self._queued = None
                self._complete_locked(queued.token, HydrationOutcome.CANCELLED)

    def retains_gate(self, gate) -> bool:
        """Whether any admitted request still carries *gate* (stores/gate)."""
        with self._lock:
            return any(
                entry is not None and entry.request.commit_gate is gate
                for entry in (self._active, self._queued)
            )

    def retire(self, *, join_timeout: float) -> bool:
        """Cancel the queued reference and join or report the exact worker."""
        with self._lock:
            self._retired = True
            queued, self._queued = self._queued, None
            if queued is not None:
                self._complete_locked(queued.token, HydrationOutcome.CANCELLED)
            worker = self._worker
        if (
            worker is not None
            and worker is not current_thread()
            and worker.ident is not None
        ):
            worker.join(timeout=max(0.0, float(join_timeout)))
        return worker is None or not worker.is_alive()

    # -- worker ------------------------------------------------------------- #

    def _ensure_worker_locked(self) -> None:
        worker = self._worker
        if worker is not None and worker.is_alive():
            return
        thread = Thread(
            target=self._run,
            name="scattering-preview-transport",
            daemon=True,
        )
        self._worker = thread
        try:
            thread.start()
        except Exception:
            if self._worker is thread:
                self._worker = None
                queued, self._queued = self._queued, None
                if queued is not None:
                    self._complete_locked(
                        queued.token,
                        HydrationOutcome.FAILED,
                        "transport worker failed to start",
                    )

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._retired or self._queued is None:
                    self._worker = None
                    return
                entry, self._queued = self._queued, None
                self._active = entry
            outcome, diagnostic = self._execute(entry)
            with self._lock:
                final = entry.token
                self._active = None
                committed = entry.committed_token
                if committed is not None and final != committed:
                    # The presentation moved after the read was committed: the
                    # data is resident, the newer token re-projects from it.
                    self._complete_locked(committed, outcome, diagnostic)
                    self._complete_locked(
                        final, HydrationOutcome.ALREADY_RESIDENT
                    )
                else:
                    self._complete_locked(final, outcome, diagnostic)

    def _execute(self, entry: _TransportEntry):
        try:
            preview = read_frame_preview(
                entry.request.read_key,
                detector_projection=entry.projection,
            )
        except Exception as error:
            return HydrationOutcome.FAILED, _diagnostic(error)
        with self._lock:
            token = entry.token
            closed = entry.closed
        prepared = PreparedHydrationCommit(
            entry.request, token, entry.key, closed, preview, entry.projection
        )
        entry.committed_token = token
        for attempt in (0, 1):
            try:
                outcome = self._commit(prepared)
            except Exception as error:
                # Retain the exact prepared commit for ONE idempotent retry.
                if attempt:
                    return HydrationOutcome.FAILED, _diagnostic(error)
                continue
            if type(outcome) is HydrationOutcome:
                return outcome, preview.detector_diagnostic
            return HydrationOutcome.FAILED, "target port returned no outcome"
        return HydrationOutcome.FAILED, "unreachable"

    # -- completion bookkeeping --------------------------------------------- #

    def _complete_locked(
        self,
        token: HydrationToken,
        outcome: HydrationOutcome,
        diagnostic: str | None = None,
    ) -> None:
        completion = HydrationCompletion(token, outcome, diagnostic or None)
        self._counters[outcome] += 1
        self._completions.append(completion)
        sink = self._completion_sink
        if sink is not None:
            try:
                sink(completion)
            except Exception:
                pass


def _diagnostic(error: BaseException) -> str:
    return str(error) or type(error).__name__


__all__ = [
    "HydrationTransport",
    "PreparedHydrationCommit",
]
