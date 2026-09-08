"""Single-flight, failure-total loader for one processed Browse context."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable

from xdart.modules.display_context import BrowseContext
from xdart.modules.frame_publication import PublicationStore
from xrd_tools.core.staging import browse_publication_max_items
from xrd_tools.io import (
    Browse1DCache,
    FrameScalarCatalog,
    FrameViewReader,
    ProcessedScan,
)
from xrd_tools.io.browse_1d_cache import Browse1DCachePhase
from xrd_tools.io.browse_presentation import read_browse_presentation
from xrd_tools.io.output_transaction import (
    TargetSnapshot,
    capture_target_snapshot,
    revalidate_stream_terminal,
    stream_terminal_object_revision,
)
from xrd_tools.session.frame_record_store import FrameRecordStore
from xrd_tools.session.scan_norm import (
    empty_norm_aggregate,
    fold_norm_metadata,
    next_norm_revision,
)

from ..browse_values import (
    BrowseCleanupReceipt,
    BrowseLoadOutcome,
    BrowseLoadRequest,
    BrowseLoadStatus,
    BrowseLoadTiming,
    canonical_browse_scan_key,
)
from ..events import CleanupStatus, DetachedDiagnostic, detach_exception


_CLEANUP_PENDING = "pending"
_CLEANUP_IN_PROGRESS = "in_progress"
_CLEANUP_CLEANED = "cleaned"


@dataclass(slots=True)
class _BrowseOperation:
    """The one exact active request, worker, result, and cleanup truth."""

    request: BrowseLoadRequest
    cancelled: Event
    perf_enabled: bool = False
    worker: Thread | None = None
    outcome: BrowseLoadOutcome | None = None
    context: BrowseContext | None = None
    terminal: bool = False
    # The exact worker-side reader is attached before __enter__ and remains
    # strongly owned here until its retryable close succeeds.  It is never
    # closed while BrowseLoader._lock is held.
    reader: object | None = None
    reader_entered: bool = False
    reader_cleanup_in_progress: bool = False
    # A newly allocated cache remains operation-owned until it is transferred
    # into the exact loaded context.  Failed/cancelled construction therefore
    # retains one retryable cleanup owner instead of leaking a cache.
    cache: Browse1DCache | None = None
    cache_cleanup_in_progress: bool = False
    cleanup_failures: list[DetachedDiagnostic] = field(default_factory=list)
    cleanup_state: str = _CLEANUP_PENDING


class BrowseLoader:
    """Own one active operation and at most one exact queued replacement."""

    def __init__(
        self,
        *,
        max_items: int | None = None,
        join_timeout: float = 5.0,
        open_scan=ProcessedScan,
        open_reader=FrameViewReader,
        open_cache=Browse1DCache,
        clock: Callable[[], float] = monotonic,
        perf_enabled: Callable[[], bool] | None = None,
    ) -> None:
        if max_items is None:
            max_items = browse_publication_max_items()
        if type(max_items) is not int or max_items < 1:
            raise ValueError("browse retention must be positive")
        self._max_items = max_items
        self._join_timeout = float(join_timeout)
        self._open_scan = open_scan
        if not callable(open_reader):
            raise TypeError("browse scalar reader factory must be callable")
        self._open_reader = open_reader
        if not callable(open_cache):
            raise TypeError("browse 1-D cache factory must be callable")
        self._open_cache = open_cache
        if not callable(clock):
            raise TypeError("browse worker clock must be callable")
        if perf_enabled is not None and not callable(perf_enabled):
            raise TypeError("browse performance gate must be callable")
        self._clock = clock
        self._perf_enabled = (
            perf_enabled
            if perf_enabled is not None
            else lambda: (
                bool(os.environ.get("XDART_PERF"))
                or os.environ.get(
                    "XDART_PERF_QUARTILES", ""
                ).strip() == "1"
            )
        )
        self._lock = Lock()
        self._active: _BrowseOperation | None = None
        self._queued: _BrowseOperation | None = None
        self._close: BrowseCleanupReceipt | None = None

    def _perf_requested(self) -> bool:
        try:
            return bool(self._perf_enabled())
        except BaseException:
            return False

    def _timing_now(self) -> float | None:
        try:
            value = float(self._clock())
        except BaseException:
            return None
        return value if math.isfinite(value) else None

    def _attach_reader(
        self, operation: _BrowseOperation, reader: object,
    ) -> None:
        """Publish worker reader custody before any reader-side open."""

        with self._lock:
            if (
                self._active is not operation
                or operation.reader is not None
                or operation.reader_cleanup_in_progress
                or operation.terminal
            ):
                raise RuntimeError("browse scalar reader admission raced")
            operation.reader = reader

    def _discard_unentered_reader(
        self, operation: _BrowseOperation, reader: object,
    ) -> None:
        """Drop a reader whose failure-total __enter__ did not return."""

        with self._lock:
            if (
                operation.reader is not reader
                or operation.reader_entered
                or operation.reader_cleanup_in_progress
            ):
                raise RuntimeError("browse unentered reader custody drifted")
            operation.reader = None

    def _close_operation_reader(
        self,
        operation: _BrowseOperation,
        *,
        attempts: int,
    ) -> bool:
        """Retryably close one exact reader, always outside the loader lock."""

        if type(attempts) is not int or attempts < 1:
            raise ValueError("browse reader close attempts must be positive")
        with self._lock:
            reader = operation.reader
            if reader is None:
                return True
            if (
                not operation.reader_entered
                or operation.reader_cleanup_in_progress
            ):
                return False
            operation.reader_cleanup_in_progress = True
        caught: BaseException | None = None
        for _attempt in range(attempts):
            try:
                reader.__exit__(None, None, None)
            except BaseException as error:
                caught = error
                continue
            with self._lock:
                if operation.reader is not reader:
                    operation.reader_cleanup_in_progress = False
                    raise RuntimeError("browse reader close custody drifted")
                operation.reader = None
                operation.reader_entered = False
                operation.reader_cleanup_in_progress = False
                operation.cleanup_failures.clear()
            return True
        assert caught is not None
        diagnostic = detach_exception(caught, "browse.reader_close")
        with self._lock:
            if operation.reader is reader:
                operation.reader_cleanup_in_progress = False
                operation.cleanup_failures[:] = [diagnostic]
        return False

    def _attach_cache(
        self, operation: _BrowseOperation, cache: Browse1DCache,
    ) -> None:
        """Publish exact cache custody before context construction."""

        if type(cache) is not Browse1DCache:
            raise TypeError("browse worker requires an exact Browse1DCache")
        with self._lock:
            if (
                self._active is not operation
                or operation.cache is not None
                or operation.cache_cleanup_in_progress
                or operation.terminal
            ):
                raise RuntimeError("browse 1-D cache admission raced")
            operation.cache = cache

    @staticmethod
    def _settle_cache(cache: Browse1DCache) -> None:
        """Close one exact cache without a loader/context lock."""

        if type(cache) is not Browse1DCache:
            raise TypeError("browse cleanup requires an exact Browse1DCache")
        cache.close()
        if cache.phase is not Browse1DCachePhase.CLOSED:
            raise RuntimeError("browse 1-D cache close did not settle")

    def _close_operation_cache(
        self,
        operation: _BrowseOperation,
        *,
        attempts: int,
    ) -> bool:
        """Retryably settle one operation-owned cache outside the loader lock."""

        if type(attempts) is not int or attempts < 1:
            raise ValueError("browse cache close attempts must be positive")
        with self._lock:
            cache = operation.cache
            if cache is None:
                return True
            if operation.cache_cleanup_in_progress:
                return False
            operation.cache_cleanup_in_progress = True
        caught: BaseException | None = None
        for _attempt in range(attempts):
            try:
                self._settle_cache(cache)
            except BaseException as error:
                caught = error
                continue
            with self._lock:
                if operation.cache is not cache:
                    operation.cache_cleanup_in_progress = False
                    raise RuntimeError("browse cache close custody drifted")
                operation.cache = None
                operation.cache_cleanup_in_progress = False
                operation.cleanup_failures.clear()
            return True
        assert caught is not None
        diagnostic = detach_exception(caught, "browse.cache_close")
        with self._lock:
            if operation.cache is cache:
                operation.cache_cleanup_in_progress = False
                operation.cleanup_failures[:] = [diagnostic]
        return False

    def _transfer_cache_to_context(
        self,
        operation: _BrowseOperation,
        context: BrowseContext,
        cache: Browse1DCache,
    ) -> None:
        """Transfer the exact cache from operation custody to one context."""

        if context.browse_1d_cache is not cache:
            raise RuntimeError("browse context cache identity changed")
        with self._lock:
            if (
                self._active is not operation
                or operation.cache is not cache
                or operation.cache_cleanup_in_progress
                or operation.terminal
            ):
                raise RuntimeError("browse cache transfer custody drifted")
            operation.cache = None

    def _catalog_from_reader(
        self,
        operation: _BrowseOperation,
        canonical_path: str,
        cancelled: Event,
    ) -> FrameScalarCatalog:
        if type(operation) is not _BrowseOperation:
            raise TypeError("browse scalar catalog requires its operation owner")
        reader = self._open_reader(
            canonical_path,
            resolve_source=False,
        )
        self._attach_reader(operation, reader)
        if cancelled.is_set():
            self._discard_unentered_reader(operation, reader)
            raise InterruptedError("Browse scalar catalog read cancelled")
        try:
            entered = reader.__enter__()
        except BaseException:
            # FrameViewReader.__enter__ is failure-total and closes its own
            # partially opened HDF graph before propagating.
            self._discard_unentered_reader(operation, reader)
            raise
        if entered is not reader:
            # A foreign context-manager replacement is not the object this
            # operation admitted and cannot become its close authority.
            with self._lock:
                if operation.reader is reader:
                    operation.reader_entered = True
            self._close_operation_reader(operation, attempts=2)
            raise RuntimeError("browse scalar reader changed identity")
        with self._lock:
            if operation.reader is not reader:
                raise RuntimeError("browse scalar reader custody drifted")
            operation.reader_entered = True
        try:
            catalog = reader.read_scalar_catalog(cancelled=cancelled.is_set)
            if type(catalog) is not FrameScalarCatalog:
                raise TypeError(
                    "browse scalar reader returned a foreign catalog"
                )
        except BaseException:
            if not self._close_operation_reader(operation, attempts=2):
                raise RuntimeError(
                    "browse scalar reader cleanup remains pending"
                )
            raise
        if not self._close_operation_reader(operation, attempts=2):
            raise RuntimeError("browse scalar reader cleanup remains pending")
        return catalog

    @property
    def _context(self) -> BrowseContext | None:
        return None if self._active is None else self._active.context

    @property
    def _worker(self) -> Thread | None:
        return None if self._active is None else self._active.worker

    def begin(self, request: BrowseLoadRequest) -> BrowseLoadRequest:
        if type(request) is not BrowseLoadRequest:
            raise TypeError("browse loader requires BrowseLoadRequest")
        launch = False
        with self._lock:
            if self._close is not None:
                raise RuntimeError("browse loader is closing")
            operation = self._active
            queued = self._queued
            if queued is not None:
                if (
                    queued.request is request
                    and not queued.cancelled.is_set()
                ):
                    return request
                if queued.cancelled.is_set():
                    raise RuntimeError("browse cleanup remains pending")
                raise RuntimeError(
                    "browse loader already owns one queued replacement"
                )
            if operation is None:
                launch = True
            elif operation.request is request:
                if operation.cancelled.is_set():
                    raise RuntimeError("browse cleanup remains pending")
                return request
            else:
                if operation.cancelled.is_set():
                    raise RuntimeError("browse cleanup remains pending")
                self._queued = _BrowseOperation(
                    request, Event(), self._perf_requested()
                )
                operation.cancelled.set()
        if launch:
            self._launch(request)
        else:
            self._progress()
        return request

    def poll(self, request: BrowseLoadRequest) -> BrowseLoadOutcome | None:
        self._progress()
        with self._lock:
            operation = self._active
            if (
                operation is None
                or operation.request is not request
                or operation.outcome is None
                or operation.outcome.request is not request
            ):
                return None
            return operation.outcome

    def owns_outcome(self, outcome: BrowseLoadOutcome) -> bool:
        """Whether *outcome* is the exact terminal value this loader owns."""

        with self._lock:
            operation = self._active
            return (
                type(outcome) is BrowseLoadOutcome
                and operation is not None
                and outcome is operation.outcome
                and outcome.request is operation.request
            )

    def owns_request(self, request: BrowseLoadRequest) -> bool:
        """Whether this loader still owns the exact request or its cleanup."""

        with self._lock:
            operation = self._active
            return (
                type(request) is BrowseLoadRequest
                and (
                    (
                        operation is not None
                        and operation.request is request
                    )
                    or (
                        self._queued is not None
                        and self._queued.request is request
                    )
                    or (
                        self._close is not None
                        and self._close.request is request
                    )
                )
            )

    def context_for_outcome(
        self, outcome: BrowseLoadOutcome
    ) -> BrowseContext | None:
        """Peek at an exact READY context without transferring ownership."""

        with self._lock:
            operation = self._active
            if (
                operation is None
                or outcome is not operation.outcome
                or outcome.request is not operation.request
                or operation.cancelled.is_set()
                or operation.cleanup_state != _CLEANUP_PENDING
                or operation.reader is not None
                or operation.cache is not None
            ):
                return None
            context = operation.context
            if context is not None and (
                type(context.scalar_catalog) is not FrameScalarCatalog
                or type(context.browse_1d_cache) is not Browse1DCache
                or context.prepared_reintegrate_offer is None
            ):
                return None
            return context

    def consume(self, outcome: BrowseLoadOutcome) -> BrowseContext | None:
        self._progress()
        next_operation = None
        with self._lock:
            operation = self._active
            if (
                operation is None
                or outcome is not operation.outcome
                or outcome.request is not operation.request
                or operation.cancelled.is_set()
                or operation.cleanup_state != _CLEANUP_PENDING
                or operation.reader is not None
                or operation.cache is not None
            ):
                return None
            context = (
                operation.context
                if outcome.status is BrowseLoadStatus.READY
                else None
            )
            if context is not None and (
                type(context.scalar_catalog) is not FrameScalarCatalog
                or type(context.browse_1d_cache) is not Browse1DCache
                or context.prepared_reintegrate_offer is None
            ):
                return None
            operation.context = None
            if self._close is None:
                next_operation = self._queued
            self._active = next_operation
            self._queued = None
        if next_operation is not None:
            self._launch(
                next_operation.request,
                propagate_failure=False,
                operation=next_operation,
            )
        return context

    def cancel(
        self, request: BrowseLoadRequest
    ) -> BrowseCleanupReceipt:
        if type(request) is not BrowseLoadRequest:
            raise TypeError("browse cleanup identity must be exact")
        owned = False
        with self._lock:
            operation = self._active
            queued = self._queued
            if queued is not None:
                if queued.request is not request:
                    return BrowseCleanupReceipt(
                        request, CleanupStatus.CLEANUP_PENDING
                    )
                queued.cancelled.set()
                if operation is not None:
                    operation.cancelled.set()
                owned = True
            elif operation is not None and operation.request is request:
                operation.cancelled.set()
                owned = True
            else:
                return BrowseCleanupReceipt(
                    request, CleanupStatus.CLEANUP_PENDING
                )
        self._progress()
        with self._lock:
            operation = self._active
            queued = self._queued
            if (
                queued is not None
                and queued.request is request
            ):
                failures = (
                    ()
                    if operation is None
                    else tuple(operation.cleanup_failures)
                )
                return BrowseCleanupReceipt(
                    request,
                    CleanupStatus.CLEANUP_PENDING,
                    failures,
                )
            if operation is not None and operation.request is request:
                if (
                    owned
                    and self._close is None
                    and operation.terminal
                    and operation.context is None
                    and operation.reader is None
                    and operation.cache is None
                    and operation.cleanup_state == _CLEANUP_CLEANED
                    and (
                        operation.worker is None
                        or not operation.worker.is_alive()
                    )
                ):
                    self._active = None
                    return BrowseCleanupReceipt(
                        request, CleanupStatus.CLEANED
                    )
                return BrowseCleanupReceipt(
                    request,
                    CleanupStatus.CLEANUP_PENDING,
                    tuple(operation.cleanup_failures),
                )
            if (
                owned
                and self._close is None
                and operation is None
                and queued is None
            ):
                return BrowseCleanupReceipt(
                    request, CleanupStatus.CLEANED
                )
            return BrowseCleanupReceipt(
                request, CleanupStatus.CLEANUP_PENDING
            )

    def release_context(
        self, context: BrowseContext
    ) -> BrowseCleanupReceipt:
        request = (
            context.load_request
            if type(context) is BrowseContext
            and type(context.load_request) is BrowseLoadRequest
            else None
        )
        if type(context) is not BrowseContext:
            return BrowseCleanupReceipt(
                request, CleanupStatus.CLEANUP_PENDING
            )
        try:
            if not context.released and not context.invalidated:
                # Withdraw hydration/publication authority before closing the
                # cache it would otherwise still be allowed to mutate.
                context.invalidate()
            cache = context.browse_1d_cache
            if cache is not None:
                if type(cache) is not Browse1DCache:
                    raise TypeError(
                        "Browse context owns a foreign 1-D cache"
                    )
                self._settle_cache(cache)
                context.detach_browse_1d_cache(cache)
            if not context.released:
                context.release()
        except BaseException as error:
            return BrowseCleanupReceipt(
                request,
                CleanupStatus.CLEANUP_PENDING,
                (detach_exception(error, "browse.release"),),
            )
        return BrowseCleanupReceipt(request, CleanupStatus.CLEANED)

    def close(
        self, expected: BrowseLoadRequest | None = None
    ) -> BrowseCleanupReceipt:
        if expected is not None and type(expected) is not BrowseLoadRequest:
            raise TypeError("browse close identity must be exact")
        with self._lock:
            if (
                self._close is not None
                and self._close.cleanup_status is CleanupStatus.CLEANED
            ):
                if (
                    expected is not None
                    and expected is not self._close.request
                ):
                    raise RuntimeError("browse close identity changed")
                return self._close
            operation = self._active
            owned = (
                self._queued.request
                if self._queued is not None
                else None if operation is None else operation.request
            )
            if self._close is None:
                self._close = BrowseCleanupReceipt(
                    expected if expected is not None else owned,
                    CleanupStatus.CLEANUP_PENDING,
                )
            elif (
                expected is not None
                and expected is not self._close.request
            ):
                raise RuntimeError("browse close identity changed")
            queued = self._queued
            if queued is not None:
                queued.cancelled.set()
                queued.outcome = BrowseLoadOutcome(
                    queued.request, BrowseLoadStatus.CANCELLED
                )
                queued.terminal = True
                queued.cleanup_state = _CLEANUP_CLEANED
            if operation is not None:
                operation.cancelled.set()
            elif queued is not None:
                self._active = queued
                self._queued = None
        self._progress(retire_cancelled=True)
        with self._lock:
            operation = self._active
            if operation is None:
                self._close = BrowseCleanupReceipt(
                    self._close.request, CleanupStatus.CLEANED
                )
                return self._close
            self._close = BrowseCleanupReceipt(
                self._close.request,
                CleanupStatus.CLEANUP_PENDING,
                tuple(operation.cleanup_failures),
            )
            return self._close

    def _launch(
        self,
        request: BrowseLoadRequest,
        *,
        propagate_failure: bool = True,
        operation: _BrowseOperation | None = None,
    ) -> None:
        fresh = operation is None
        if fresh:
            operation = _BrowseOperation(
                request, Event(), self._perf_requested()
            )
        assert operation is not None
        with self._lock:
            if fresh:
                if self._active is not None or self._close is not None:
                    raise RuntimeError("browse launch lost single ownership")
                self._active = operation
            elif (
                operation.request is not request
                or self._active is not operation
            ):
                raise RuntimeError("browse promotion lost exact ownership")
            if (
                operation.cancelled.is_set()
                or self._close is not None
            ):
                operation.outcome = BrowseLoadOutcome(
                    request, BrowseLoadStatus.CANCELLED
                )
                operation.terminal = True
                operation.cleanup_state = _CLEANUP_CLEANED
                return
            try:
                worker = Thread(
                    target=self._load,
                    args=(operation,),
                    name="scattering-browse",
                    daemon=True,
                )
                operation.worker = worker
                worker.start()
            except BaseException as error:
                if propagate_failure:
                    operation.cancelled.set()
                    self._active = None
                else:
                    operation.outcome = BrowseLoadOutcome(
                        request,
                        BrowseLoadStatus.FAILED,
                        detach_exception(
                            error, "browse.thread_start"
                        ).message,
                    )
                    operation.terminal = True
                if propagate_failure:
                    raise

    def _progress(self, *, retire_cancelled: bool = False) -> None:
        with self._lock:
            operation = self._active
            has_reader = bool(
                operation is not None
                and operation.terminal
                and operation.reader is not None
            )
        if has_reader:
            assert operation is not None
            self._close_operation_reader(operation, attempts=1)
            with self._lock:
                if (
                    self._active is operation
                    and operation.reader is not None
                ):
                    return
        with self._lock:
            operation = self._active
            has_cache = bool(
                operation is not None
                and operation.terminal
                and operation.cache is not None
            )
        if has_cache:
            assert operation is not None
            self._close_operation_cache(operation, attempts=1)
            with self._lock:
                if (
                    self._active is operation
                    and operation.cache is not None
                ):
                    return
        context = None
        with self._lock:
            operation = self._active
            if operation is None or not operation.terminal:
                return
            should_retire = (
                self._queued is not None
                or self._close is not None
                or retire_cancelled
            )
            should_cleanup = operation.cancelled.is_set() or should_retire
            if (
                operation.context is None
                and operation.cancelled.is_set()
                and operation.cleanup_state == _CLEANUP_PENDING
            ):
                operation.cleanup_state = _CLEANUP_CLEANED
                operation.cleanup_failures.clear()
            if operation.context is not None and should_cleanup:
                if operation.cleanup_state != _CLEANUP_PENDING:
                    return
                operation.cleanup_state = _CLEANUP_IN_PROGRESS
                context = operation.context
        if context is not None:
            try:
                receipt = self.release_context(context)
            except BaseException as error:
                receipt = BrowseCleanupReceipt(
                    operation.request,
                    CleanupStatus.CLEANUP_PENDING,
                    (detach_exception(error, "browse.release"),),
                )
            with self._lock:
                if self._active is not operation:
                    return
                if (
                    type(receipt) is not BrowseCleanupReceipt
                    or receipt.request is not operation.request
                    or receipt.cleanup_status is not CleanupStatus.CLEANED
                ):
                    operation.cleanup_state = _CLEANUP_PENDING
                    operation.cleanup_failures[:] = (
                        receipt.cleanup_failures
                        if type(receipt) is BrowseCleanupReceipt
                        and receipt.request is operation.request
                        else ()
                    )
                    return
                operation.context = None
                operation.cleanup_state = _CLEANUP_CLEANED
                operation.cleanup_failures.clear()
        if not should_retire:
            return
        next_operation = None
        with self._lock:
            if self._active is not operation:
                return
            if (
                operation.context is not None
                or operation.reader is not None
                or operation.cache is not None
                or operation.cache_cleanup_in_progress
                or operation.cleanup_state == _CLEANUP_IN_PROGRESS
                or (
                    operation.worker is not None
                    and operation.worker.is_alive()
                )
            ):
                return
            if self._close is None:
                next_operation = self._queued
            self._active = next_operation
            self._queued = None
        if next_operation is not None:
            if next_operation.cancelled.is_set():
                with self._lock:
                    if self._active is not next_operation:
                        return
                    next_operation.outcome = BrowseLoadOutcome(
                        next_operation.request,
                        BrowseLoadStatus.CANCELLED,
                    )
                    next_operation.terminal = True
                    next_operation.cleanup_state = _CLEANUP_CLEANED
                return
            self._launch(
                next_operation.request,
                propagate_failure=False,
                operation=next_operation,
            )

    def _load(self, operation: _BrowseOperation) -> None:
        request = operation.request
        cancelled = operation.cancelled
        context = None
        status = BrowseLoadStatus.READY
        detail = ""
        worker_started = (
            self._timing_now() if operation.perf_enabled else None
        )
        timing_fields: dict[str, object] | None = (
            {
                "valid": True,
                "prepared_capsule_s": 0.0,
                "prepared_bundle_bytes": 0,
                "prepared_1d_status": "UNMEASURED",
                "prepared_2d_status": "UNMEASURED",
            }
            if operation.perf_enabled and worker_started is not None
            else None
        )
        timing = None
        with self._lock:
            admitted = (
                self._active is operation
                and not cancelled.is_set()
            )
        try:
            if not admitted:
                status = BrowseLoadStatus.CANCELLED
            else:
                scan_key = canonical_browse_scan_key(
                    request.source_path
                )
            if status is BrowseLoadStatus.CANCELLED:
                pass
            elif not scan_key:
                status = BrowseLoadStatus.REFUSED
                detail = "No processed-scan format owns this path."
            elif not Path(request.source_path).is_file():
                status = BrowseLoadStatus.REFUSED
                detail = "Processed browse artifact is unavailable."
            elif not cancelled.is_set():
                context = self._read_context(
                    request,
                    scan_key,
                    cancelled,
                    _timing=timing_fields,
                    _operation=operation,
                )
                if context is None:
                    status = BrowseLoadStatus.CANCELLED
        except BaseException as error:
            if cancelled.is_set():
                status = BrowseLoadStatus.CANCELLED
            else:
                status = BrowseLoadStatus.FAILED
                detail = detach_exception(
                    error, "browse.load"
                ).message
        if cancelled.is_set():
            status = BrowseLoadStatus.CANCELLED
        if status is BrowseLoadStatus.READY and context is not None:
            ended = (
                None
                if timing_fields is None or worker_started is None
                else self._timing_now()
            )
            try:
                timing = (
                    None
                    if (
                        ended is None
                        or not bool(timing_fields.get("valid"))
                    )
                    else BrowseLoadTiming(
                        str(timing_fields["canonical_path"]),
                        str(timing_fields["seal_mode"]),
                        float(timing_fields["initial_seal_s"]),
                        float(timing_fields["scan_open_s"]),
                        float(timing_fields["record_iteration_s"]),
                        int(timing_fields["record_count"]),
                        float(timing_fields["presentation_read_s"]),
                        float(timing_fields["final_seal_s"]),
                        float(timing_fields["context_build_s"]),
                        max(0.0, ended - worker_started),
                        float(timing_fields["prepared_capsule_s"]),
                        int(timing_fields["prepared_bundle_bytes"]),
                        str(timing_fields["prepared_1d_status"]),
                        str(timing_fields["prepared_2d_status"]),
                    )
                )
            except BaseException:
                timing = None
        outcome = BrowseLoadOutcome(request, status, detail, timing)
        with self._lock:
            if self._active is not operation:
                orphan = context
            else:
                orphan = None
                operation.outcome = outcome
                operation.context = context
                operation.terminal = True
        if orphan is not None:
            self.release_context(orphan)
            return
        self._progress()

    def _read_context(
        self,
        request: BrowseLoadRequest,
        scan_key: str,
        cancelled: Event,
        *,
        _timing: dict[str, object] | None = None,
        _operation: _BrowseOperation,
    ) -> BrowseContext | None:
        def stage_start() -> float | None:
            if _timing is None or not bool(_timing.get("valid")):
                return None
            started = self._timing_now()
            if started is None:
                _timing["valid"] = False
            return started

        def stage_finish(name: str, started: float | None) -> None:
            if (
                _timing is None
                or started is None
                or not bool(_timing.get("valid"))
            ):
                return
            ended = self._timing_now()
            if ended is None:
                _timing["valid"] = False
                return
            _timing[name] = max(0.0, ended - started)

        path = Path(request.source_path).resolve()
        canonical_path = str(path)
        if _timing is not None:
            _timing["canonical_path"] = canonical_path
        if cancelled.is_set():
            return None
        terminal = request.terminal_commit_identity
        sealed_terminal = (
            terminal
            if stream_terminal_object_revision(terminal) is not None
            else None
        )
        if _timing is not None:
            _timing["seal_mode"] = (
                "terminal" if sealed_terminal is not None else "snapshot"
            )
        started = stage_start()
        before = (
            capture_target_snapshot(path)
            if sealed_terminal is None
            else revalidate_stream_terminal(path, sealed_terminal)
        )
        stage_finish("initial_seal_s", started)
        if type(before) is not TargetSnapshot or not before.exists:
            raise ValueError("processed browse target is unavailable")
        if cancelled.is_set():
            return None
        started = stage_start()
        scan = (
            self._open_scan(canonical_path)
            if request.source_root is None
            else self._open_scan(
                canonical_path,
                source_root=request.source_root,
            )
        )
        stage_finish("scan_open_s", started)
        records = FrameRecordStore(max_items=self._max_items)
        publications = PublicationStore(
            max_items=self._max_items,
            retain_1d_on_eviction=True,
        )
        # E6-NORM-N1: ONE revision-0 draft folded once per accepted record in
        # this sole scalar pass; the single advance happens only after the
        # complete noncancelled, nonempty pass and travels with the context.
        draft = empty_norm_aggregate(
            (request.token, scan_key, request.source_path)
        )
        started = stage_start()
        catalog = self._catalog_from_reader(
            _operation, canonical_path, cancelled,
        )
        if type(catalog) is not FrameScalarCatalog:
            raise TypeError("Browse requires an exact FrameScalarCatalog")
        if catalog.artifact_path != canonical_path:
            raise ValueError("Browse scalar catalog changed artifact identity")
        target_entry = str(getattr(scan, "entry", "entry"))
        if catalog.entry != target_entry:
            raise ValueError("Browse scalar catalog changed entry identity")
        labels = catalog.labels
        for row in catalog.rows:
            if cancelled.is_set():
                return None
            draft = fold_norm_metadata(draft, row.metadata_numeric)
        stage_finish("record_iteration_s", started)
        if _timing is not None:
            _timing["record_count"] = len(labels)
        # Cancellation may arrive as the iterator reports exhaustion after its
        # final yield.  Recheck before the one revision bump/context publish.
        if cancelled.is_set():
            return None
        if not labels:
            raise ValueError("processed browse artifact has no frames")
        started = stage_start()
        presentation, persisted_mask = read_browse_presentation(
            canonical_path,
        )
        stage_finish("presentation_read_s", started)
        if cancelled.is_set():
            return None
        # Ordinary Browse must not prepay Reintegration admission.  Install one
        # typed lazy-fallback offer; the operation worker reads and authenticates
        # the artifact only if the user actually requests Reintegration.
        started = stage_start()
        from xrd_tools.reduction.reintegrate_prepared import (
            PreparedReintegrateOffer,
            PreparedCapsuleMissCode,
            unprepared_reintegrate_offer,
        )
        prepared_offer = unprepared_reintegrate_offer()
        if type(prepared_offer) is not PreparedReintegrateOffer:
            raise TypeError("Browse lazy fallback returned a foreign offer")
        if (
            prepared_offer.disposition != "MISS"
            or prepared_offer.bundle is not None
            or prepared_offer.miss_code
            is not PreparedCapsuleMissCode.CAPSULE_NOT_SUPPLIED
        ):
            raise TypeError("Browse lazy fallback returned an invalid offer")
        one_d_status = two_d_status = "MISS"
        bundle_bytes = 0
        stage_finish("prepared_capsule_s", started)
        if _timing is not None:
            _timing["prepared_bundle_bytes"] = bundle_bytes
            _timing["prepared_1d_status"] = one_d_status
            _timing["prepared_2d_status"] = two_d_status
        if cancelled.is_set():
            return None
        started = stage_start()
        after = (
            capture_target_snapshot(path)
            if sealed_terminal is None
            else revalidate_stream_terminal(path, sealed_terminal)
        )
        stage_finish("final_seal_s", started)
        if type(after) is not TargetSnapshot or not after.exists or after != before:
            raise ValueError("processed browse target changed during load")
        if cancelled.is_set():
            return None
        started = stage_start()
        cache = self._open_cache()
        if type(cache) is not Browse1DCache:
            raise TypeError("browse worker returned a foreign 1-D cache")
        operation_owned = type(_operation) is _BrowseOperation
        attached = False
        try:
            if operation_owned:
                self._attach_cache(_operation, cache)
                attached = True
            if cancelled.is_set():
                raise InterruptedError("Browse context construction cancelled")
            context = BrowseContext(
                context_token=request.token,
                load_generation=request.load_generation,
                operation=request,
                requested_path=request.source_path,
                scan_key=scan_key,
                scan=scan,
                frame=None,
                frame_ids=labels,
                frames={},
                viewer_rows_1d={},
                viewer_rows_2d={},
                publication_store=publications,
                record_store=records,
                norm_aggregate=next_norm_revision(draft),
                scalar_catalog=catalog,
                browse_1d_cache=cache,
                target_entry=target_entry,
                loaded_labels=labels,
                target_snapshot=after,
                prepared_reintegrate_offer=prepared_offer,
            )
            context.adopt_load_request(request)
            context.stamp_provenance(
                calibration=json.dumps(
                    presentation,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                mask=str(
                    persisted_mask
                    or catalog.rows[0].mask_baked
                ),
                result=request.source_path,
            )
            context.mark_loaded()
            if cancelled.is_set():
                raise InterruptedError("Browse context construction cancelled")
            if operation_owned:
                self._transfer_cache_to_context(_operation, context, cache)
        except BaseException as error:
            if attached:
                if not self._close_operation_cache(_operation, attempts=2):
                    raise RuntimeError(
                        "browse 1-D cache cleanup remains pending"
                    ) from error
            else:
                self._settle_cache(cache)
            raise
        stage_finish("context_build_s", started)
        return context

__all__ = ["BrowseLoader"]
