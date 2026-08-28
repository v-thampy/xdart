"""Qt-free ownership for processed Browser state and lifecycle."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
import logging
import math
import os
import threading
import time
from typing import Callable

from xdart.modules.display_context import BrowseContext
from xrd_tools.io.output_transaction import (
    StreamTerminal,
    stream_terminal_object_revision,
)
from xrd_tools.io.viewer_1d import SUPPORTED_VIEWER_1D_SUFFIXES
from xrd_tools.io.viewer_2d import SUPPORTED_VIEWER_SUFFIXES
from xrd_tools.session.intent_store import RunIntentSnapshot
from xrd_tools.session.readiness import Tool, tool_from_mode_text

from .browser_catalog import (
    BrowserCatalogEntry,
    DirectoryModifiedCache,
    enumerate_processed_artifacts,
    processed_directory,
)
from .browse_values import (
    BrowseLoadOutcome,
    BrowseLoadRequest,
    BrowseLoadStatus,
    BrowseLoadTiming,
    LoadedBrowseCapture,
)
from .display_values import DisplayFrameKey
from .events import RunIdentity, detached_exception_strings


_LOG = logging.getLogger(__name__)


class BrowserRefreshEffect(Enum):
    """Smallest page refresh required by one Browser transition."""

    NONE = "none"
    FULL = "full"
    CATALOG = "catalog"


@dataclass(frozen=True, slots=True)
class BrowserCatalogRequest:
    """One immutable catalog policy snapshot."""

    token: int
    directory: str
    accepted_suffixes: frozenset[str] | None
    inspect_directory_contents: bool

    def __post_init__(self) -> None:
        if (
            type(self.token) is not int
            or self.token <= 0
            or type(self.directory) is not str
            or (
                self.accepted_suffixes is not None
                and (
                    type(self.accepted_suffixes) is not frozenset
                    or not self.accepted_suffixes
                    or any(
                        type(suffix) is not str
                        or not suffix.startswith(".")
                        or suffix != suffix.casefold()
                        for suffix in self.accepted_suffixes
                    )
                )
            )
            or type(self.inspect_directory_contents) is not bool
        ):
            raise ValueError("Browser catalog request is invalid")


@dataclass(frozen=True, slots=True)
class BrowserCatalogWake:
    """Opaque exact completion token delivered across the Qt bridge."""

    token: int

    def __post_init__(self) -> None:
        if type(self.token) is not int or self.token <= 0:
            raise ValueError("Browser catalog wake is invalid")


@dataclass(frozen=True, slots=True)
class ReintegrateReloadDirective:
    """Exact invalidated Browse target that must be loaded again."""

    request: BrowseLoadRequest
    target: str
    terminal_commit_identity: StreamTerminal | None = None

    def __post_init__(self) -> None:
        if (
            type(self.request) is not BrowseLoadRequest
            or type(self.target) is not str
            or not self.target
            or self.request.source_path != self.target
            or (
                self.terminal_commit_identity is not None
                and (
                    stream_terminal_object_revision(
                        self.terminal_commit_identity
                    )
                    is None
                    or self.terminal_commit_identity.target != self.target
                )
            )
        ):
            raise ValueError("Reintegrate reload directive is invalid")


@dataclass(frozen=True, slots=True)
class AverageReloadDirective:
    """Committed Average target plus its run-frozen Project root."""

    target: str
    entry: str
    terminal_commit_identity: StreamTerminal
    source_root: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.target) is not str
            or not self.target
            or type(self.entry) is not str
            or not self.entry
            or stream_terminal_object_revision(self.terminal_commit_identity)
            is None
            or self.terminal_commit_identity.target != self.target
            or (
                self.source_root is not None
                and (
                    type(self.source_root) is not str
                    or not self.source_root
                    or not os.path.isabs(self.source_root)
                    or os.path.normcase(os.path.normpath(self.source_root))
                    != self.source_root
                )
            )
        ):
            raise ValueError("Average reload directive is invalid")


BrowserReloadDirective = ReintegrateReloadDirective | AverageReloadDirective


class TerminalPaintMode(Enum):
    REBIND = "rebind"
    REPAINT = "repaint"
    REPAINT_FALLBACK = "repaint-fallback"


@dataclass(frozen=True, slots=True)
class TerminalBrowseTimingStart:
    """Owner-issued pre-submit boundary for terminal Browse telemetry."""

    started_at: float

    def __post_init__(self) -> None:
        if type(self.started_at) is not float or not math.isfinite(
            self.started_at
        ):
            raise ValueError("terminal Browse timing start is invalid")


@dataclass(frozen=True, slots=True)
class TerminalBrowseHandoff:
    request: BrowseLoadRequest
    run_identity: RunIdentity
    source_artifact: str
    current_label: int | None
    selected_labels: tuple[int, ...]
    commit_identity: StreamTerminal | None = None

    def __post_init__(self) -> None:
        if (
            type(self.request) is not BrowseLoadRequest
            or type(self.run_identity) is not RunIdentity
            or type(self.source_artifact) is not str
            or not self.source_artifact
            or (
                self.current_label is not None
                and (
                    type(self.current_label) is not int
                    or self.current_label < 0
                )
            )
            or type(self.selected_labels) is not tuple
            or any(
                type(label) is not int or label < 0
                for label in self.selected_labels
            )
            or (
                self.commit_identity is not None
                and type(self.commit_identity) is not StreamTerminal
            )
        ):
            raise ValueError("terminal Browse handoff is invalid")


@dataclass(frozen=True, slots=True)
class TerminalBrowsePresentation:
    request: BrowseLoadRequest
    context: BrowseContext

    def __post_init__(self) -> None:
        if (
            type(self.request) is not BrowseLoadRequest
            or type(self.context) is not BrowseContext
        ):
            raise ValueError("terminal Browse presentation is invalid")


@dataclass(frozen=True, slots=True)
class TerminalBrowseSettlement:
    handoff: TerminalBrowseHandoff
    presentation: TerminalBrowsePresentation
    reuse_seal_authorized: bool

    def __post_init__(self) -> None:
        if (
            type(self.handoff) is not TerminalBrowseHandoff
            or type(self.presentation) is not TerminalBrowsePresentation
            or self.presentation.request is not self.handoff.request
            or type(self.reuse_seal_authorized) is not bool
        ):
            raise ValueError("terminal Browse settlement is invalid")


@dataclass(frozen=True, slots=True)
class TerminalRebindAuthorization:
    run_identity: RunIdentity
    source_artifact: str
    browse_artifact: str

    def __post_init__(self) -> None:
        if (
            type(self.run_identity) is not RunIdentity
            or type(self.source_artifact) is not str
            or not self.source_artifact
            or type(self.browse_artifact) is not str
            or not self.browse_artifact
        ):
            raise ValueError("terminal rebind authorization is invalid")


@dataclass(frozen=True, slots=True)
class TerminalBrowsePaintRequest:
    presentation: TerminalBrowsePresentation
    mode: TerminalPaintMode
    authorization: TerminalRebindAuthorization | None = None
    started_at: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.presentation) is not TerminalBrowsePresentation
            or type(self.mode) is not TerminalPaintMode
            or (
                self.mode is TerminalPaintMode.REBIND
                and type(self.authorization) is not TerminalRebindAuthorization
            )
            or (
                self.mode is not TerminalPaintMode.REBIND
                and self.authorization is not None
            )
            or (
                self.started_at is not None
                and (
                    type(self.started_at) is not float
                    or not math.isfinite(self.started_at)
                )
            )
        ):
            raise ValueError("terminal Browse paint request is invalid")


@dataclass(frozen=True, slots=True)
class TerminalBrowsePaintReceipt:
    request: TerminalBrowsePaintRequest
    applied: bool
    repaint_pending: bool

    def __post_init__(self) -> None:
        if (
            type(self.request) is not TerminalBrowsePaintRequest
            or type(self.applied) is not bool
            or type(self.repaint_pending) is not bool
        ):
            raise ValueError("terminal Browse paint receipt is invalid")


@dataclass(frozen=True, slots=True)
class TerminalPaintCompletion:
    accepted: bool
    schedule_repaint: bool = False
    retired: bool = False

    def __post_init__(self) -> None:
        if not all(
            type(value) is bool
            for value in (
                self.accepted,
                self.schedule_repaint,
                self.retired,
            )
        ):
            raise ValueError("terminal paint completion is invalid")


@dataclass(slots=True)
class _TerminalBrowsePerf:
    request: BrowseLoadRequest
    started_at: float
    poll_adopt_count: int = 0
    poll_adopt_s: float = 0.0
    settle_s: float = 0.0
    presentation_s: float = 0.0
    worker: BrowseLoadTiming | None = None
    fallback_pending: bool = False


@dataclass(frozen=True, slots=True)
class ProcessedBrowserProjection:
    """Detached Browser values consumed by shell projection."""

    directory: str
    catalog: tuple[BrowserCatalogEntry, ...]
    transient_frame: DisplayFrameKey | None
    date_sorted: bool
    auto_last: bool
    explicit_directory: bool

    def __post_init__(self) -> None:
        if (
            type(self.directory) is not str
            or type(self.catalog) is not tuple
            or not all(type(entry) is BrowserCatalogEntry for entry in self.catalog)
            or (
                self.transient_frame is not None
                and type(self.transient_frame) is not DisplayFrameKey
            )
            or type(self.date_sorted) is not bool
            or type(self.auto_last) is not bool
            or type(self.explicit_directory) is not bool
        ):
            raise ValueError("processed Browser projection is invalid")


@dataclass(frozen=True, slots=True)
class ProcessedBrowserTransition:
    """Detached page effect from one Browser ownership transition."""

    refresh: BrowserRefreshEffect
    notice: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.refresh) is not BrowserRefreshEffect
            or (self.notice is not None and type(self.notice) is not str)
        ):
            raise ValueError("processed Browser transition is invalid")


@dataclass(frozen=True, slots=True)
class _BrowserCatalogOperation:
    request: BrowserCatalogRequest
    cancelled: threading.Event
    future: Future[object]
    wake: BrowserCatalogWake


CatalogDelivery = Callable[[BrowserCatalogWake], None]
CatalogReader = Callable[..., tuple[BrowserCatalogEntry, ...]]


def browser_suffixes_for_mode(mode: str) -> frozenset[str] | None:
    """Return the immutable artifact suffix policy for one processing mode."""

    tool = tool_from_mode_text(mode)
    if tool is Tool.XYE_VIEWER:
        return SUPPORTED_VIEWER_1D_SUFFIXES
    if tool is Tool.IMAGE_VIEWER:
        return SUPPORTED_VIEWER_SUFFIXES
    return None


class ProcessedBrowserOwner:
    """Sole owner of processed catalog, reload, and terminal-paint state."""

    def __init__(
        self,
        *,
        save_path: str,
        processing_mode: str,
        deliver: CatalogDelivery,
        catalog_reader: CatalogReader | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if catalog_reader is None:
            catalog_reader = enumerate_processed_artifacts
        if (
            not callable(deliver)
            or not callable(catalog_reader)
            or not callable(clock)
        ):
            raise TypeError("Browser catalog ports must be callable")
        self._deliver = deliver
        self._catalog_reader = catalog_reader
        self._clock = clock
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)
        self._operation: _BrowserCatalogOperation | None = None
        self._queued: BrowserCatalogRequest | None = None
        self._token = 0
        self._directory = processed_directory(save_path)
        self._accepted_suffixes = browser_suffixes_for_mode(processing_mode)
        self._catalog: tuple[BrowserCatalogEntry, ...] = ()
        self._directory_time_cache = DirectoryModifiedCache()
        self._explicit_directory = False
        self._date_sorted = False
        self._auto_last = True
        self._follow_identity: RunIdentity | None = None
        self._seen_artifacts: set[str] = set()
        self._transient_frame: DisplayFrameKey | None = None
        self._transient_clear_token: int | None = None
        self._reload: BrowserReloadDirective | None = None
        self._terminal_handoff: TerminalBrowseHandoff | None = None
        self._terminal_presentation: TerminalBrowsePresentation | None = None
        self._terminal_rebind: TerminalRebindAuthorization | None = None
        self._terminal_paint: TerminalBrowsePaintRequest | None = None
        self._terminal_perf: _TerminalBrowsePerf | None = None
        self._terminal_timing_start: TerminalBrowseTimingStart | None = None
        self._closing = False
        self._closed = False

    @property
    def directory(self) -> str:
        return self._directory

    @property
    def catalog(self) -> tuple[BrowserCatalogEntry, ...]:
        return self._catalog

    @property
    def transient_frame(self) -> DisplayFrameKey | None:
        return self._transient_frame

    @property
    def date_sorted(self) -> bool:
        return self._date_sorted

    @property
    def auto_last(self) -> bool:
        return self._auto_last

    @property
    def explicit_directory(self) -> bool:
        return self._explicit_directory

    @property
    def active_request(self) -> BrowserCatalogRequest | None:
        operation = self._operation
        return None if operation is None else operation.request

    @property
    def active_wake(self) -> BrowserCatalogWake | None:
        operation = self._operation
        return None if operation is None else operation.wake

    @property
    def queued_request(self) -> BrowserCatalogRequest | None:
        return self._queued

    @property
    def pool_open(self) -> bool:
        return self._pool is not None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_reintegrate_reload(
        self,
    ) -> ReintegrateReloadDirective | None:
        directive = self._reload
        return (
            directive
            if type(directive) is ReintegrateReloadDirective
            else None
        )

    @property
    def pending_average_reload(self) -> AverageReloadDirective | None:
        directive = self._reload
        return directive if type(directive) is AverageReloadDirective else None

    @property
    def busy(self) -> bool:
        return self._reload is not None

    @property
    def polling_needed(self) -> bool:
        return not self._closing and not self._closed and self.busy

    @property
    def preserve_science(self) -> bool:
        return self.busy

    @property
    def terminal_handoff(self) -> TerminalBrowseHandoff | None:
        return self._terminal_handoff

    @property
    def terminal_presentation(self) -> TerminalBrowsePresentation | None:
        return self._terminal_presentation

    @property
    def terminal_request(self) -> BrowseLoadRequest | None:
        handoff = self._terminal_handoff
        if handoff is not None:
            return handoff.request
        presentation = self._terminal_presentation
        return None if presentation is None else presentation.request

    @property
    def terminal_perf_active(self) -> bool:
        return self._terminal_perf is not None

    def _now(self) -> float | None:
        try:
            value = float(self._clock())
        except BaseException:
            return None
        return value if math.isfinite(value) else None

    def begin_terminal_timing(
        self, *, enabled: bool,
    ) -> TerminalBrowseTimingStart | None:
        """Capture the GUI-total boundary before Browse worker submission."""

        if type(enabled) is not bool:
            raise TypeError("terminal Browse timing enablement must be bool")
        # A new GUI submission boundary supersedes an abandoned, never-bound
        # sample.  The exact returned object remains the sole authority.
        self._terminal_timing_start = None
        if not enabled or self._closing or self._closed:
            return None
        started_at = self._now()
        timing_start = (
            None
            if started_at is None
            else TerminalBrowseTimingStart(started_at)
        )
        self._terminal_timing_start = timing_start
        return timing_start

    def retire_terminal_timing(
        self, timing_start: TerminalBrowseTimingStart | None,
    ) -> bool:
        """Retire only the exact unbound timing authority, at most once."""

        if timing_start is None:
            return self._terminal_timing_start is None
        if self._terminal_timing_start is not timing_start:
            return False
        self._terminal_timing_start = None
        return True

    def begin_terminal_handoff(
        self,
        request: BrowseLoadRequest,
        run_identity: RunIdentity,
        source_artifact: str,
        current_label: int | None,
        selected_labels: tuple[int, ...],
        commit_identity: StreamTerminal | None,
        *,
        timing_start: TerminalBrowseTimingStart | None,
    ) -> TerminalBrowseHandoff | None:
        if (
            self._closing
            or self._closed
            or timing_start is not self._terminal_timing_start
        ):
            return None
        handoff = TerminalBrowseHandoff(
            request,
            run_identity,
            source_artifact,
            current_label,
            selected_labels,
            commit_identity,
        )
        # Bind the exact pre-submit sample to this accepted request and make
        # replay, foreign-owner transfer, and double consumption impossible.
        self._terminal_timing_start = None
        self.retire_terminal(force=True)
        self._terminal_handoff = handoff
        self._terminal_perf = (
            None
            if timing_start is None
            else _TerminalBrowsePerf(request, timing_start.started_at)
        )
        return handoff

    def update_terminal_selection(
        self,
        request: BrowseLoadRequest,
        *,
        run_identity: RunIdentity,
        source_artifact: str,
        current_label: int,
        selected_labels: tuple[int, ...],
    ) -> bool:
        handoff = self._terminal_handoff
        if (
            handoff is None
            or handoff.request is not request
            or handoff.run_identity is not run_identity
            or handoff.source_artifact != source_artifact
        ):
            return False
        self._terminal_handoff = replace(
            handoff,
            current_label=current_label,
            selected_labels=selected_labels,
        )
        return True

    def begin_poll_timing(self, request: BrowseLoadRequest) -> float | None:
        perf = self._terminal_perf
        if perf is None or perf.request is not request:
            return None
        started = self._now()
        if started is None:
            self._terminal_perf = None
        return started

    def finish_poll_timing(
        self, request: BrowseLoadRequest, started: float | None,
    ) -> bool:
        perf = self._terminal_perf
        if (
            perf is None
            or perf.request is not request
            or type(started) is not float
        ):
            return False
        ended = self._now()
        if ended is None:
            self._terminal_perf = None
            return False
        perf.poll_adopt_count += 1
        perf.poll_adopt_s += max(0.0, ended - started)
        return True

    def begin_settle_timing(self, request: BrowseLoadRequest) -> float | None:
        return self.begin_poll_timing(request)

    def finish_settle_timing(
        self,
        request: BrowseLoadRequest,
        started: float | None,
        outcome: BrowseLoadOutcome,
    ) -> bool:
        perf = self._terminal_perf
        if (
            perf is None
            or perf.request is not request
            or type(started) is not float
            or type(outcome) is not BrowseLoadOutcome
            or outcome.request is not request
        ):
            return False
        ended = self._now()
        if ended is None:
            self._terminal_perf = None
            return False
        perf.settle_s += max(0.0, ended - started)
        if (
            outcome.status is BrowseLoadStatus.READY
            and type(outcome.timing) is BrowseLoadTiming
        ):
            perf.worker = outcome.timing
            return True
        self._terminal_perf = None
        return False

    def settle_terminal(
        self,
        outcome: object,
        capture: LoadedBrowseCapture | None,
    ) -> TerminalBrowseSettlement | None:
        handoff = self._terminal_handoff
        if (
            type(outcome) is not BrowseLoadOutcome
            or handoff is None
            or outcome.request is not handoff.request
        ):
            return None
        if (
            outcome.status is not BrowseLoadStatus.READY
            or type(capture) is not LoadedBrowseCapture
            or capture.request is not handoff.request
        ):
            self.retire_terminal(handoff.request)
            return None
        presentation = TerminalBrowsePresentation(
            handoff.request, capture.context
        )
        self._terminal_handoff = None
        self._terminal_presentation = presentation
        self._terminal_rebind = None
        self._terminal_paint = None
        commit_identity = handoff.commit_identity
        seal_authorized = bool(
            handoff.request.terminal_commit_identity is commit_identity
            and type(commit_identity) is StreamTerminal
            and capture.target_snapshot.size == commit_identity.size
            and capture.target_snapshot.digest == commit_identity.digest
        )
        return TerminalBrowseSettlement(
            handoff, presentation, seal_authorized
        )

    def terminal_presentation_is_current(
        self,
        presentation: TerminalBrowsePresentation,
        capture: LoadedBrowseCapture | None,
        *,
        owns_request: bool,
    ) -> bool:
        return bool(
            type(presentation) is TerminalBrowsePresentation
            and self._terminal_presentation is presentation
            and type(owns_request) is bool
            and owns_request
            and type(capture) is LoadedBrowseCapture
            and capture.context is presentation.context
            and capture.request is presentation.request
        )

    def authorize_terminal_rebind(
        self,
        presentation: TerminalBrowsePresentation,
        authorization: TerminalRebindAuthorization,
    ) -> bool:
        if (
            self._terminal_presentation is not presentation
            or type(authorization) is not TerminalRebindAuthorization
        ):
            return False
        self._terminal_rebind = authorization
        return True

    def begin_terminal_paint(
        self,
        presentation: TerminalBrowsePresentation,
        capture: LoadedBrowseCapture | None,
        *,
        owns_request: bool,
        reuse_science: bool,
    ) -> TerminalBrowsePaintRequest | None:
        if not self.terminal_presentation_is_current(
            presentation, capture, owns_request=owns_request
        ):
            self.retire_terminal(presentation.request)
            return None
        if type(reuse_science) is not bool:
            return None
        perf = self._terminal_perf
        mode = (
            TerminalPaintMode.REBIND
            if reuse_science and self._terminal_rebind is not None
            else TerminalPaintMode.REPAINT_FALLBACK
            if perf is not None and perf.fallback_pending
            else TerminalPaintMode.REPAINT
        )
        started = None
        if perf is not None and perf.request is presentation.request:
            started = self._now()
            if started is None:
                self._terminal_perf = None
        request = TerminalBrowsePaintRequest(
            presentation,
            mode,
            self._terminal_rebind if mode is TerminalPaintMode.REBIND else None,
            started,
        )
        self._terminal_paint = request
        return request

    def complete_terminal_paint(
        self,
        receipt: TerminalBrowsePaintReceipt,
    ) -> TerminalPaintCompletion:
        request = None if type(receipt) is not TerminalBrowsePaintReceipt else receipt.request
        if request is None or self._terminal_paint is not request:
            return TerminalPaintCompletion(False)
        self._terminal_paint = None
        presentation = request.presentation
        if self._terminal_presentation is not presentation:
            return TerminalPaintCompletion(False)
        perf = self._terminal_perf
        ended = None
        if (
            perf is not None
            and perf.request is presentation.request
            and request.started_at is not None
        ):
            ended = self._now()
            if ended is None:
                self._terminal_perf = None
                perf = None
            else:
                perf.presentation_s += max(
                    0.0, ended - request.started_at
                )
        if not receipt.applied:
            return TerminalPaintCompletion(True, schedule_repaint=True)
        if (
            request.mode is TerminalPaintMode.REBIND
            and receipt.repaint_pending
        ):
            if perf is not None:
                perf.fallback_pending = True
            return TerminalPaintCompletion(True)
        if (
            request.mode is TerminalPaintMode.REPAINT_FALLBACK
            and receipt.repaint_pending
        ):
            return TerminalPaintCompletion(True)
        if perf is not None and ended is not None:
            self._log_terminal_perf(perf, request.mode, ended)
        self.retire_terminal(presentation.request)
        return TerminalPaintCompletion(True, retired=True)

    @staticmethod
    def _log_terminal_perf(
        perf: _TerminalBrowsePerf,
        mode: TerminalPaintMode,
        ended: float,
    ) -> None:
        worker = perf.worker
        if type(worker) is not BrowseLoadTiming:
            return
        try:
            _LOG.info(
                "[PERF-BROWSE] source=%s token=%s generation=%d "
                "seal=%s records=%d mode=%s | "
                "worker=%.3fs initial-seal=%.3fs scan-open=%.3fs "
                "record-iteration=%.3fs presentation-read=%.3fs "
                "final-seal=%.3fs context-build=%.3fs | "
                "gui-total=%.3fs poll/adopt=%.3fs(n=%d) "
                "settle=%.3fs presentation=%.3fs",
                worker.canonical_path,
                perf.request.token,
                perf.request.load_generation,
                worker.seal_mode,
                worker.record_count,
                mode.value,
                worker.worker_total_s,
                worker.initial_seal_s,
                worker.scan_open_s,
                worker.record_iteration_s,
                worker.presentation_read_s,
                worker.final_seal_s,
                worker.context_build_s,
                max(0.0, ended - perf.started_at),
                perf.poll_adopt_s,
                perf.poll_adopt_count,
                perf.settle_s,
                perf.presentation_s,
            )
        except BaseException:
            pass

    def retire_terminal(
        self,
        request: BrowseLoadRequest | None = None,
        *,
        force: bool = False,
    ) -> bool:
        owned = self.terminal_request
        if not force and (owned is None or request is not owned):
            return False
        self._terminal_handoff = None
        self._terminal_presentation = None
        self._terminal_rebind = None
        self._terminal_paint = None
        self._terminal_perf = None
        return owned is not None

    def projection(self) -> ProcessedBrowserProjection:
        return ProcessedBrowserProjection(
            self._directory,
            self._catalog,
            self._transient_frame,
            self._date_sorted,
            self._auto_last,
            self._explicit_directory,
        )

    def clear_directory_cache(self) -> None:
        if not self._closing and not self._closed:
            self._directory_time_cache.clear()

    def adopt_reload(
        self,
        directive: BrowserReloadDirective,
    ) -> BrowserReloadDirective:
        if (
            self._closing
            or self._closed
            or type(directive)
            not in {ReintegrateReloadDirective, AverageReloadDirective}
        ):
            raise RuntimeError("processed Browser cannot adopt reload")
        current = self._reload
        if current is not None and current is not directive:
            raise RuntimeError("processed Browser already owns another reload")
        self._reload = directive
        return directive

    def retire_reload(
        self,
        directive: BrowserReloadDirective,
    ) -> bool:
        if self._reload is not directive:
            return False
        self._reload = None
        return True

    def set_directory(
        self,
        selected: str,
        *,
        explicit: bool,
    ) -> ProcessedBrowserTransition:
        if self._closing or self._closed:
            return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)
        if type(selected) is not str or not selected or type(explicit) is not bool:
            return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)
        directory = os.path.abspath(os.path.expanduser(selected))
        self._explicit_directory = explicit
        self._directory = directory
        self._catalog = ()
        self.request_catalog()
        return ProcessedBrowserTransition(BrowserRefreshEffect.FULL, "")

    def set_date_sorted(self, requested: bool) -> bool:
        if self._closing or self._closed or type(requested) is not bool:
            return False
        changed = requested != self._date_sorted
        self._date_sorted = requested
        if changed:
            self.request_catalog()
        return changed

    def set_auto_last(self, requested: bool) -> bool:
        if self._closing or self._closed or type(requested) is not bool:
            return False
        changed = requested != self._auto_last
        self._auto_last = requested
        return changed

    def reconcile_intent(
        self,
        prior: RunIntentSnapshot,
        current: RunIntentSnapshot,
    ) -> ProcessedBrowserTransition:
        if self._closing or self._closed:
            return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)
        before = prior.thaw()
        after = current.thaw()
        accepted_suffixes = browser_suffixes_for_mode(after.processing_mode)
        policy_changed = accepted_suffixes != self._accepted_suffixes
        self._accepted_suffixes = accepted_suffixes
        directory_changed = (
            before.save_path != after.save_path
            and not self._explicit_directory
        )
        if directory_changed:
            self._directory = processed_directory(after.save_path)
        if directory_changed or policy_changed:
            self._catalog = ()
            self.request_catalog()
            return ProcessedBrowserTransition(BrowserRefreshEffect.FULL)
        return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)

    def begin_follow(self, identity: RunIdentity) -> None:
        if (
            self._closing
            or self._closed
            or type(identity) is not RunIdentity
        ):
            return
        self._explicit_directory = False
        self._follow_identity = identity
        self._seen_artifacts.clear()

    def follow_processed_artifact(
        self,
        frame: DisplayFrameKey,
    ) -> ProcessedBrowserTransition:
        if (
            self._closing
            or self._closed
            or type(frame) is not DisplayFrameKey
        ):
            return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)
        if frame.run_identity is not self._follow_identity:
            self._follow_identity = frame.run_identity
            self._seen_artifacts.clear()
        artifact = os.path.abspath(os.path.expanduser(frame.artifact))
        first_seen = artifact not in self._seen_artifacts
        self._seen_artifacts.add(artifact)
        if self._explicit_directory:
            return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)
        directory = os.path.dirname(artifact)
        changed = directory != self._directory
        if changed:
            self._directory = directory
            self._catalog = ()
        if changed or first_seen:
            self.request_catalog()
        return ProcessedBrowserTransition(
            BrowserRefreshEffect.FULL if changed else BrowserRefreshEffect.NONE
        )

    def set_transient_frame(self, frame: DisplayFrameKey | None) -> None:
        if self._closing or self._closed:
            return
        if frame is None or type(frame) is DisplayFrameKey:
            self._transient_frame = frame

    def mark_transient_catalog_barrier(
        self,
        request: BrowserCatalogRequest | None,
    ) -> None:
        if self._closing or self._closed:
            return
        self._transient_clear_token = (
            None if request is None else request.token
        )

    def request_catalog(self) -> BrowserCatalogRequest | None:
        if self._pool is None or self._closing or self._closed:
            return None
        self._token += 1
        request = BrowserCatalogRequest(
            self._token,
            self._directory,
            self._accepted_suffixes,
            self._date_sorted,
        )
        operation = self._operation
        if operation is not None:
            operation.cancelled.set()
            operation.future.cancel()
            self._queued = request
            return request
        self._launch_catalog(request)
        return request

    def _launch_catalog(self, request: BrowserCatalogRequest) -> None:
        pool = self._pool
        if (
            pool is None
            or self._closing
            or self._closed
            or self._operation is not None
        ):
            return
        cancelled = threading.Event()
        future = pool.submit(
            self._catalog_reader,
            request.directory,
            accepted_suffixes=request.accepted_suffixes,
            inspect_directory_contents=request.inspect_directory_contents,
            directory_time_cache=self._directory_time_cache,
            cancelled=cancelled,
        )
        wake = BrowserCatalogWake(request.token)
        operation = _BrowserCatalogOperation(
            request,
            cancelled,
            future,
            wake,
        )
        self._operation = operation
        future.add_done_callback(lambda _done: self._deliver(wake))

    def poll_catalog(self) -> ProcessedBrowserTransition:
        if self._closing or self._closed:
            return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)
        operation = self._operation
        if operation is not None and operation.future.done():
            return self.consume_catalog(operation.wake)
        if operation is None and self._queued is not None:
            request, self._queued = self._queued, None
            self._launch_catalog(request)
        elif operation is None:
            self.request_catalog()
        return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)

    def consume_catalog(self, wake: object) -> ProcessedBrowserTransition:
        operation = self._operation
        if (
            self._closing
            or self._closed
            or type(wake) is not BrowserCatalogWake
            or operation is None
            or wake is not operation.wake
        ):
            return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)
        self._operation = None
        queued, self._queued = self._queued, None
        request = operation.request
        current = bool(
            not operation.cancelled.is_set()
            and request.token == self._token
            and request.directory == self._directory
            and request.accepted_suffixes == self._accepted_suffixes
            and request.inspect_directory_contents == self._date_sorted
        )
        try:
            catalog: object = operation.future.result()
        except BaseException as error:
            if queued is not None:
                self._launch_catalog(queued)
            return (
                ProcessedBrowserTransition(BrowserRefreshEffect.NONE)
                if not current
                else ProcessedBrowserTransition(
                    BrowserRefreshEffect.FULL,
                    "Browser refresh failed: "
                    f"{detached_exception_strings(error)[2]}",
                )
            )
        if queued is not None:
            self._launch_catalog(queued)
        if not current:
            return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)
        if (
            type(catalog) is not tuple
            or not all(type(entry) is BrowserCatalogEntry for entry in catalog)
        ):
            return ProcessedBrowserTransition(
                BrowserRefreshEffect.FULL,
                "Browser refresh returned invalid data.",
            )
        clear_token = self._transient_clear_token
        transient_cleared = bool(
            clear_token is not None and request.token >= clear_token
        )
        if transient_cleared:
            self._transient_frame = None
            self._transient_clear_token = None
        if catalog == self._catalog and not transient_cleared:
            return ProcessedBrowserTransition(BrowserRefreshEffect.NONE)
        self._catalog = catalog
        return ProcessedBrowserTransition(BrowserRefreshEffect.CATALOG)

    def begin_close(self) -> bool:
        if self._closed:
            return True
        if not self._closing:
            self._closing = True
            self._terminal_timing_start = None
            self._reload = None
            self.retire_terminal(force=True)
            self._queued = None
            self._token += 1
            operation = self._operation
            if operation is not None:
                operation.cancelled.set()
                operation.future.cancel()
        return self.retry_close()

    def retry_close(self) -> bool:
        if self._closed:
            return True
        if not self._closing:
            return False
        operation = self._operation
        if operation is not None:
            if not operation.future.done():
                return False
            self._operation = None
            try:
                operation.future.result()
            except BaseException:
                pass
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        self._closed = True
        return True


__all__ = [
    "AverageReloadDirective",
    "BrowserReloadDirective",
    "BrowserCatalogRequest",
    "BrowserCatalogWake",
    "BrowserRefreshEffect",
    "ProcessedBrowserOwner",
    "ProcessedBrowserProjection",
    "ProcessedBrowserTransition",
    "ReintegrateReloadDirective",
    "TerminalBrowseHandoff",
    "TerminalBrowsePaintReceipt",
    "TerminalBrowsePaintRequest",
    "TerminalBrowsePresentation",
    "TerminalBrowseSettlement",
    "TerminalBrowseTimingStart",
    "TerminalPaintCompletion",
    "TerminalPaintMode",
    "TerminalRebindAuthorization",
    "browser_suffixes_for_mode",
]
