"""Single-flight, failure-total loader for one processed Browse context."""

from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread, current_thread

from xdart.modules.display_context import BrowseContext
from xdart.modules.frame_publication import FramePublication, PublicationStore
from xrd_tools.core.energy import (
    WavelengthUnit,
    canonical_wavelength_m,
)
from xrd_tools.core.staging import browse_publication_max_items
from xrd_tools.core.provenance import read_provenance
from xrd_tools.io import ProcessedScan, iter_frame_records
from xrd_tools.io.output_transaction import TargetSnapshot, capture_target_snapshot
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
    canonical_browse_scan_key,
    canonical_browse_source_identity,
)
from ..events import CleanupStatus, DetachedDiagnostic, detach_exception


_CLEANUP_PENDING = "pending"
_CLEANUP_IN_PROGRESS = "in_progress"
_CLEANUP_CLEANED = "cleaned"


def _iter_browse_records(source: str):
    """Read every cheap 1-D row while deferring current-frame heavy pixels."""

    yield from iter_frame_records(source, include_heavy=False)


@dataclass(slots=True)
class _BrowseOperation:
    """The one exact active request, worker, result, and cleanup truth."""

    request: BrowseLoadRequest
    cancelled: Event
    worker: Thread | None = None
    outcome: BrowseLoadOutcome | None = None
    context: BrowseContext | None = None
    terminal: bool = False
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
        read_records=_iter_browse_records,
    ) -> None:
        if max_items is None:
            max_items = browse_publication_max_items()
        if type(max_items) is not int or max_items < 1:
            raise ValueError("browse retention must be positive")
        self._max_items = max_items
        self._join_timeout = float(join_timeout)
        self._open_scan = open_scan
        self._read_records = read_records
        self._lock = Lock()
        self._active: _BrowseOperation | None = None
        self._queued: _BrowseOperation | None = None
        self._close: BrowseCleanupReceipt | None = None

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
                self._queued = _BrowseOperation(request, Event())
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
            ):
                return None
            return operation.context

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
            ):
                return None
            context = (
                operation.context
                if outcome.status is BrowseLoadStatus.READY
                else None
            )
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
            if not context.released:
                context.release()
            store = context.record_store
            if store is not None and len(store):
                store.clear()
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
            self._queued = None
            if operation is not None:
                operation.cancelled.set()
                worker = operation.worker
            else:
                worker = None
        if (
            worker is not None
            and worker is not current_thread()
            and worker.ident is not None
            and worker.is_alive()
        ):
            worker.join(timeout=self._join_timeout)
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
            operation = _BrowseOperation(request, Event())
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
                    request, scan_key, cancelled
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
        outcome = BrowseLoadOutcome(request, status, detail)
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
    ) -> BrowseContext | None:
        path = Path(request.source_path).resolve()
        if str(path) != request.source_path:
            raise ValueError("processed browse target must be canonical")
        before = capture_target_snapshot(path)
        if type(before) is not TargetSnapshot or not before.exists:
            raise ValueError("processed browse target is unavailable")
        scan = self._open_scan(request.source_path)
        records = FrameRecordStore(max_items=self._max_items)
        publications = PublicationStore(
            max_items=self._max_items,
            retain_1d_on_eviction=True,
        )
        labels: list[int] = []
        first = None
        # E6-NORM-N1: ONE revision-0 draft folded once per accepted record in
        # this sole pass; the single advance happens only after the complete
        # noncancelled, nonempty pass and travels with the ready context.
        draft = empty_norm_aggregate(
            (request.token, scan_key, request.source_path)
        )
        for record in self._read_records(request.source_path):
            if cancelled.is_set():
                records.clear()
                publications.clear()
                return None
            if type(record.label) is not int:
                raise TypeError("processed frame labels must be integers")
            view = record.active_view()
            draft = fold_norm_metadata(draft, view.metadata_numeric)
            source = canonical_browse_source_identity(
                view, request.source_path,
            )
            records.upsert(
                record, source_identity=source, persisted=True
            )
            publications.upsert(
                FramePublication(
                    view,
                    record=record,
                    source_identity=source,
                    scan_key=scan_key,
                )
            )
            labels.append(record.label)
            if first is None:
                first = view
        # Cancellation may arrive as the iterator reports exhaustion after its
        # final yield.  Recheck before the one revision bump/context publish.
        if cancelled.is_set():
            records.clear()
            publications.clear()
            return None
        if not labels:
            raise ValueError("processed browse artifact has no frames")
        if tuple(labels) != tuple(sorted(set(labels))):
            records.clear(); publications.clear()
            raise ValueError("processed frame labels must be strictly increasing")
        persisted = read_provenance(request.source_path)
        provenance = (
            persisted.get("config", {})
            if isinstance(persisted, Mapping)
            else {}
        )
        if not isinstance(provenance, Mapping):
            provenance = {}
        presentation = {
            key: provenance.get(key)
            for key in ("poni_file", "geometry", "gi")
            if provenance.get(key) is not None
        }
        wavelength_m = _persisted_wavelength_m(scan)
        if wavelength_m is not None:
            presentation["wavelength_m"] = wavelength_m
        after = capture_target_snapshot(path)
        if type(after) is not TargetSnapshot or not after.exists or after != before:
            records.clear(); publications.clear()
            raise ValueError("processed browse target changed during load")
        context = BrowseContext(
            context_token=request.token, load_generation=request.load_generation,
            operation=request, requested_path=request.source_path,
            scan_key=scan_key, scan=scan, frame=None, frame_ids=labels, frames={},
            viewer_rows_1d={}, viewer_rows_2d={}, publication_store=publications,
            record_store=records, norm_aggregate=next_norm_revision(draft),
            target_entry=getattr(scan, "entry", "entry"),
            loaded_labels=tuple(labels), target_snapshot=after,
        )
        context.adopt_load_request(request)
        context.stamp_provenance(
            calibration=json.dumps(
                presentation,
                sort_keys=True,
                separators=(",", ":"),
            ),
            mask=str(
                provenance.get("mask_file")
                or getattr(first, "mask_baked", False)
            ),
            result=request.source_path,
        )
        context.mark_loaded()
        return context


def _persisted_wavelength_m(scan: object) -> float | None:
    """Read only explicit Angstrom evidence from the processed artifact."""

    try:
        metadata = scan.metadata
    except Exception:
        return None
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("wavelength_A")
    if type(value) is bool:
        return None
    return canonical_wavelength_m(value, WavelengthUnit.ANGSTROM)


__all__ = ["BrowseLoader"]
