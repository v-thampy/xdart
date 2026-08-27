"""Dedicated worker lane for sparse Browse 1-D row hydration.

This module deliberately owns no Qt object and writes no scientific display
store.  It fills only the exact :class:`Browse1DCache` attached to one loaded
Browse context, then emits an array-free wake for the existing Browse poller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from queue import Empty, SimpleQueue
from threading import Event, Lock, Thread
from typing import Callable

from xdart.modules.display_context import (
    BrowseContext,
    CommitGate,
    ContextKind,
    DisplaySelection,
)
from xrd_tools.core import Axis
from xrd_tools.io import (
    Browse1DCache,
    Browse1DCacheOperation,
    Browse1DLabelStoreCustodyError,
    Browse1DRowKey,
    Frame1DModeRows,
    Frame1DRows,
    FrameScalarCatalog,
    FrameViewReader,
    browse_1d_row_name,
)
from xrd_tools.io.output_transaction import (
    TargetSnapshot,
    capture_target_snapshot,
    revalidate_stream_terminal,
    stream_terminal_object_revision,
)

from .display_values import DisplayFrameKey


_MAX_DIAGNOSTIC_CHARS = 512
_UNKNOWN_INVENTORY = object()


class Browse1DHydrationStatus(str, Enum):
    READY = "ready"
    ALREADY_RESIDENT = "already-resident"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, eq=False)
class Browse1DHydrationTarget:
    """One exact runtime-owned frame and its persisted catalog coordinate."""

    frame: DisplayFrameKey
    ordinal: int
    label: int

    def __post_init__(self) -> None:
        if (
            type(self.frame) is not DisplayFrameKey
            or type(self.ordinal) is not int
            or self.ordinal < 1
            or type(self.label) is not int
            or self.label < 0
            or self.frame.work_ordinal != self.ordinal
            or self.frame.local_frame_label != self.label
        ):
            raise TypeError("Browse 1-D hydration target is invalid")


@dataclass(frozen=True, slots=True, eq=False)
class Browse1DHydrationRequest:
    """Immutable exact-object admission carried from GUI to one worker."""

    identity: object
    context: BrowseContext
    catalog: FrameScalarCatalog
    cache: Browse1DCache
    gate: CommitGate
    epoch: int
    selection: DisplaySelection
    artifact: str
    entry: str
    target_snapshot: TargetSnapshot
    targets: tuple[Browse1DHydrationTarget, ...]

    def __post_init__(self) -> None:
        if type(self.identity) is not object:
            raise TypeError("Browse 1-D request identity must be opaque")
        if (
            type(self.context) is not BrowseContext
            or type(self.catalog) is not FrameScalarCatalog
            or type(self.cache) is not Browse1DCache
            or type(self.gate) is not CommitGate
            or type(self.epoch) is not int
            or self.epoch < 1
            or type(self.selection) is not DisplaySelection
            or self.selection.kind is not ContextKind.BROWSE
            or type(self.artifact) is not str
            or not self.artifact
            or type(self.entry) is not str
            or not self.entry
            or type(self.target_snapshot) is not TargetSnapshot
            or not self.target_snapshot.exists
            or type(self.targets) is not tuple
            or not self.targets
        ):
            raise TypeError("Browse 1-D hydration request is invalid")
        previous = 0
        run_identity = self.targets[0].frame.run_identity
        for target in self.targets:
            if (
                type(target) is not Browse1DHydrationTarget
                or target.ordinal <= previous
                or target.frame.run_identity is not run_identity
            ):
                raise ValueError("Browse 1-D targets are not exact and ordered")
            previous = target.ordinal


@dataclass(frozen=True, slots=True, eq=False)
class Browse1DHydrationCompletion:
    """Array-free terminal worker fact consumed by the GUI poller."""

    request_identity: object
    status: Browse1DHydrationStatus
    labels: tuple[int, ...]
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.request_identity) is not object
            or type(self.status) is not Browse1DHydrationStatus
            or type(self.labels) is not tuple
            or any(type(label) is not int or label < 0 for label in self.labels)
            or type(self.diagnostic) is not str
            or len(self.diagnostic) > _MAX_DIAGNOSTIC_CHARS
        ):
            raise TypeError("Browse 1-D hydration completion is invalid")


@dataclass(frozen=True, slots=True, eq=False)
class Browse1DModeInventory:
    """Exact deterministic cache keys for one persisted 1-D mode."""

    mode: str
    axis: Browse1DRowKey
    intensity: Browse1DRowKey
    sigma: Browse1DRowKey | None = None

    def __post_init__(self) -> None:
        if (
            type(self.mode) is not str
            or not self.mode
            or type(self.axis) is not Browse1DRowKey
            or type(self.intensity) is not Browse1DRowKey
            or self.sigma is not None
            and type(self.sigma) is not Browse1DRowKey
        ):
            raise TypeError("Browse 1-D mode inventory is invalid")
        coordinate = (self.axis.frame, self.axis.label)
        if (
            (self.intensity.frame, self.intensity.label) != coordinate
            or self.sigma is not None
            and (self.sigma.frame, self.sigma.label) != coordinate
            or self.axis.name != browse_1d_row_name(self.mode, "axis")
            or self.intensity.name
            != browse_1d_row_name(self.mode, "intensity")
            or self.sigma is not None
            and self.sigma.name != browse_1d_row_name(self.mode, "sigma")
        ):
            raise ValueError("Browse 1-D mode inventory changed identity")

    @property
    def keys(self) -> tuple[Browse1DRowKey, ...]:
        return (
            (self.axis, self.intensity)
            if self.sigma is None
            else (self.axis, self.intensity, self.sigma)
        )


@dataclass(frozen=True, slots=True, eq=False)
class Browse1DLabelInventory:
    """Known per-label inventory; an empty modes tuple is exact completion."""

    ordinal: int
    label: int
    modes: tuple[Browse1DModeInventory, ...]

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal < 1
            or type(self.label) is not int
            or self.label < 0
            or type(self.modes) is not tuple
            or any(type(item) is not Browse1DModeInventory for item in self.modes)
        ):
            raise TypeError("Browse 1-D label inventory is invalid")
        names: list[str] = []
        for item in self.modes:
            if (
                item.mode in names
                or any(
                    key.frame != self.ordinal or key.label != self.label
                    for key in item.keys
                )
            ):
                raise ValueError("Browse 1-D label inventory is inconsistent")
            names.append(item.mode)

    def mode(self, name: str) -> Browse1DModeInventory | None:
        if type(name) is not str or not name:
            raise TypeError("Browse 1-D inventory mode must be exact")
        for item in self.modes:
            if item.mode == name:
                return item
        return None


class _HydrationTask:
    """Mutable worker custody; all references are retired before completion."""

    __slots__ = (
        "request", "signature", "cancelled", "thread", "running",
        "reader", "reader_entered", "operation", "borrows", "status",
        "pending_coordinate", "pending_keys", "diagnostic",
        "cleanup_pending",
    )

    def __init__(
        self,
        request: Browse1DHydrationRequest,
        signature: tuple[object, ...],
    ) -> None:
        self.request = request
        self.signature = signature
        self.cancelled = Event()
        self.thread: Thread | None = None
        self.running = False
        self.reader: object | None = None
        self.reader_entered = False
        self.operation: Browse1DCacheOperation | None = None
        self.borrows: list[object] = []
        self.status: Browse1DHydrationStatus | None = None
        self.pending_coordinate: tuple[int, int] | None = None
        self.pending_keys: tuple[object, ...] = ()
        self.diagnostic = ""
        self.cleanup_pending = False


class Browse1DHydrationLane:
    """One-active/one-latest worker lane bound to one Browse context."""

    def __init__(
        self,
        context: BrowseContext,
        *,
        open_reader: Callable[..., object] = FrameViewReader,
        capture_snapshot: Callable[[str], TargetSnapshot] = (
            capture_target_snapshot
        ),
    ) -> None:
        if type(context) is not BrowseContext:
            raise TypeError("Browse 1-D lane requires one exact context")
        if not callable(open_reader) or not callable(capture_snapshot):
            raise TypeError("Browse 1-D lane dependencies must be callable")
        self._context = context
        self._catalog = context.scalar_catalog
        self._cache = context.browse_1d_cache
        self._gate = context.commit_gate
        self._epoch = context.commit_epoch
        self._artifact = context.requested_path
        self._entry = context.target_entry
        self._snapshot = context.target_snapshot
        self._load_terminal = getattr(
            context.load_request, "terminal_commit_identity", None,
        )
        self._terminal = (
            self._load_terminal
            if stream_terminal_object_revision(self._load_terminal) is not None
            else None
        )
        self._open_reader = open_reader
        self._capture_snapshot = capture_snapshot
        self._lock = Lock()
        self._active: _HydrationTask | None = None
        self._queued: _HydrationTask | None = None
        self._wanted_identity: object | None = None
        self._known_keys: dict[tuple[int, int], tuple[object, ...]] = {}
        self._completions: SimpleQueue[Browse1DHydrationCompletion] = (
            SimpleQueue()
        )
        self._ready_pending = False
        self._terminal_failure: tuple[tuple[object, ...], str] | None = None
        self._retiring = False
        self._closed = False
        self._validate_bound_context()

    @property
    def context(self) -> BrowseContext:
        return self._context

    def expected_inventory(
        self, ordinal: int, label: int,
    ) -> Browse1DLabelInventory | None:
        """Return immutable expected keys, with ``None`` meaning unknown.

        This is deliberately not a residency assertion.  Projection acquires
        exact cache borrows for these keys; an evicted row is then an ordinary
        incomplete result rather than a stale array escape.
        """

        if (
            type(ordinal) is not int
            or ordinal < 1
            or type(label) is not int
            or label < 0
        ):
            raise TypeError("Browse 1-D inventory coordinate is invalid")
        with self._lock:
            value = self._known_keys.get((ordinal, label), _UNKNOWN_INVENTORY)
        if value is _UNKNOWN_INVENTORY:
            return None
        if type(value) is not tuple:
            raise RuntimeError("Browse 1-D expected inventory is malformed")
        scalar_row = self._catalog.row(label)
        if scalar_row is None:
            raise RuntimeError("Browse 1-D inventory left its scalar catalog")
        by_name: dict[str, Browse1DRowKey] = {}
        for key in value:
            if (
                type(key) is not Browse1DRowKey
                or key.frame != ordinal
                or key.label != label
                or key.name in by_name
            ):
                raise RuntimeError("Browse 1-D expected keys are inconsistent")
            by_name[key.name] = key
        modes: list[Browse1DModeInventory] = []
        for mode in scalar_row.modes_1d:
            axis_name = browse_1d_row_name(mode, "axis")
            intensity_name = browse_1d_row_name(mode, "intensity")
            sigma_name = browse_1d_row_name(mode, "sigma")
            axis = by_name.pop(axis_name, None)
            intensity = by_name.pop(intensity_name, None)
            sigma = by_name.pop(sigma_name, None)
            if axis is None or intensity is None:
                raise RuntimeError("Browse 1-D expected mode is incomplete")
            modes.append(Browse1DModeInventory(
                mode, axis, intensity, sigma,
            ))
        if by_name or (not scalar_row.modes_1d and value):
            raise RuntimeError("Browse 1-D expected inventory has extra keys")
        return Browse1DLabelInventory(ordinal, label, tuple(modes))

    def _validate_bound_context(self) -> None:
        context = self._context
        if (
            type(self._catalog) is not FrameScalarCatalog
            or type(self._cache) is not Browse1DCache
            or type(self._gate) is not CommitGate
            or type(self._snapshot) is not TargetSnapshot
            or not self._snapshot.exists
            or context.scalar_catalog is not self._catalog
            or context.browse_1d_cache is not self._cache
            or context.commit_gate is not self._gate
            or context.target_snapshot is not self._snapshot
            or getattr(
                context.load_request, "terminal_commit_identity", None,
            ) is not self._load_terminal
            or self._catalog.labels is not context.frame_ids
            or self._catalog.labels is not context.loaded_labels
            or self._catalog.artifact_path != self._artifact
            or self._catalog.entry != self._entry
            or not context.loaded
            or context.invalidated
            or context.released
        ):
            raise RuntimeError("Browse 1-D lane context is not exactly admitted")

    def _request_is_live(self, request: Browse1DHydrationRequest) -> bool:
        context = self._context
        return bool(
            request.context is context
            and request.catalog is self._catalog
            and request.cache is self._cache
            and request.gate is self._gate
            and request.epoch == self._epoch == context.commit_epoch
            and request.artifact == self._artifact == context.requested_path
            and request.entry == self._entry == context.target_entry
            and request.target_snapshot is self._snapshot
            and context.scalar_catalog is self._catalog
            and context.browse_1d_cache is self._cache
            and context.commit_gate is self._gate
            and context.target_snapshot is self._snapshot
            and context.loaded
            and not context.invalidated
            and not context.released
            and not self._gate.cancelled
            and request.selection.names(context)
            and request.selection.owner == context.hydration_owner
        )

    def _make_request(
        self,
        selection: DisplaySelection,
        frames: tuple[DisplayFrameKey, ...],
    ) -> tuple[Browse1DHydrationRequest, tuple[object, ...]]:
        self._validate_bound_context()
        context = self._context
        if (
            type(selection) is not DisplaySelection
            or selection.kind is not ContextKind.BROWSE
            or not selection.names(context)
            or selection.owner != context.hydration_owner
            or type(frames) is not tuple
            or not frames
        ):
            raise TypeError("Browse 1-D submission scope is invalid")
        by_id: set[int] = set()
        targets: list[Browse1DHydrationTarget] = []
        for frame in frames:
            if (
                type(frame) is not DisplayFrameKey
                or id(frame) in by_id
                or frame.source_scan != context.scan_key
                or frame.artifact != context.requested_path
                or frame.work_ordinal > len(self._catalog.labels)
                or self._catalog.labels[frame.work_ordinal - 1]
                != frame.local_frame_label
            ):
                raise ValueError("Browse 1-D frame does not belong to catalog")
            by_id.add(id(frame))
            targets.append(Browse1DHydrationTarget(
                frame, frame.work_ordinal, frame.local_frame_label,
            ))
        targets.sort(key=lambda item: item.ordinal)
        identity = object()
        request = Browse1DHydrationRequest(
            identity,
            context,
            self._catalog,
            self._cache,
            self._gate,
            self._epoch,
            selection,
            self._artifact,
            self._entry,
            self._snapshot,
            tuple(targets),
        )
        signature = (
            id(selection),
            selection.display_generation,
            tuple(id(target.frame) for target in request.targets),
        )
        return request, signature

    def submit(
        self,
        selection: DisplaySelection,
        frames: tuple[DisplayFrameKey, ...],
    ) -> object | None:
        """Admit one exact request, retaining only the newest queued intent."""

        try:
            request, signature = self._make_request(selection, frames)
        except (RuntimeError, TypeError, ValueError):
            return None
        start: _HydrationTask | None = None
        with self._lock:
            if self._closed or self._retiring:
                return None
            failure = self._terminal_failure
            if failure is not None:
                if failure[0] == signature:
                    return None
                self._terminal_failure = None
            for held in (self._active, self._queued):
                if (
                    held is not None
                    and not held.cancelled.is_set()
                    and held.signature == signature
                ):
                    self._wanted_identity = held.request.identity
                    return held.request.identity
            task = _HydrationTask(request, signature)
            self._wanted_identity = request.identity
            if self._active is None:
                self._active = task
                start = task
            else:
                self._active.cancelled.set()
                if self._queued is not None:
                    self._queued.cancelled.set()
                self._queued = task
        if start is not None:
            self._start(start)
        return request.identity

    def terminal_diagnostic(
        self,
        selection: DisplaySelection,
        frames: tuple[DisplayFrameKey, ...],
    ) -> str | None:
        """Return one exact failed-intent fact, clearing it on intent change."""

        try:
            _request, signature = self._make_request(selection, frames)
        except (RuntimeError, TypeError, ValueError):
            return None
        with self._lock:
            failure = self._terminal_failure
            if failure is None:
                return None
            if failure[0] != signature:
                self._terminal_failure = None
                return None
            return failure[1]

    def _fail_start(self, task: _HydrationTask, error: BaseException) -> None:
        """Retire one task whose worker could not acquire execution custody."""

        diagnostic = self._diagnostic(error)
        start: _HydrationTask | None = None
        with self._lock:
            if self._active is not task:
                return
            task.running = False
            task.thread = None
            task.diagnostic = diagnostic
            if task.cleanup_pending:
                # A retry worker failed to start.  Preserve the exact task and
                # its reader/operation/borrow custody for another poll.
                return
            task.status = Browse1DHydrationStatus.FAILED
            completion = Browse1DHydrationCompletion(
                task.request.identity,
                Browse1DHydrationStatus.FAILED,
                tuple(target.label for target in task.request.targets),
                diagnostic,
            )
            if not self._retiring:
                self._completions.put(completion)
                if self._wanted_identity is task.request.identity:
                    self._terminal_failure = (task.signature, diagnostic)
            self._active = None
            if self._queued is not None and not self._retiring:
                start = self._queued
                self._queued = None
                self._active = start
        if start is not None:
            self._start(start)

    def _start(self, task: _HydrationTask) -> None:
        try:
            worker = Thread(
                target=self._run,
                args=(task,),
                name="xdart-browse-1d-hydration",
                daemon=True,
            )
        except BaseException as error:
            self._fail_start(task, error)
            return
        with self._lock:
            if self._active is not task or task.running:
                return
            task.thread = worker
            task.running = True
        try:
            worker.start()
        except BaseException as error:
            # ``Thread.start`` can itself be fault-injected after the real
            # start.  In that case the worker remains the sole terminal owner.
            if getattr(worker, "ident", None) is not None:
                return
            self._fail_start(task, error)

    def _task_is_current(self, task: _HydrationTask) -> bool:
        with self._lock:
            return bool(
                self._active is task
                and self._wanted_identity is task.request.identity
                and not self._retiring
                and not task.cancelled.is_set()
            )

    def _checkpoint(self, task: _HydrationTask) -> bool:
        return bool(
            task.cancelled.is_set()
            or not self._task_is_current(task)
            or not self._request_is_live(task.request)
        )

    def _missing_targets(
        self, request: Browse1DHydrationRequest,
    ) -> tuple[Browse1DHydrationTarget, ...]:
        resident = request.cache.resident_keys
        resident_by_target: dict[tuple[int, int], tuple[object, ...]] = {}
        for target in request.targets:
            resident_by_target[(target.ordinal, target.label)] = tuple(
                key for key in resident
                if key.frame == target.ordinal and key.label == target.label
            )
        with self._lock:
            known = dict(self._known_keys)
        missing: list[Browse1DHydrationTarget] = []
        empty_updates: dict[tuple[int, int], tuple[object, ...]] = {}
        for target in request.targets:
            coordinate = (target.ordinal, target.label)
            expected = known.get(coordinate)
            scalar_row = request.catalog.row(target.label)
            if scalar_row is None:
                raise ValueError("Browse 1-D target label left its catalog")
            if expected is None and not scalar_row.modes_1d:
                empty_updates[coordinate] = ()
                continue
            if expected is not None and resident_by_target[coordinate] == expected:
                continue
            missing.append(target)
        if empty_updates:
            with self._lock:
                if self._active is not None and (
                    self._active.request is request
                ):
                    self._known_keys.update(empty_updates)
        return tuple(missing)

    @staticmethod
    def _diagnostic(error: BaseException) -> str:
        text = f"{type(error).__name__}: {error}"
        return text[:_MAX_DIAGNOSTIC_CHARS]

    def _capture_bound_snapshot(
        self, request: Browse1DHydrationRequest,
    ) -> TargetSnapshot:
        """Requalify in the exact domain used by the admitting loader."""

        terminal = self._terminal
        if terminal is not None:
            return revalidate_stream_terminal(request.artifact, terminal)
        return self._capture_snapshot(request.artifact)

    def _open_and_read(
        self,
        task: _HydrationTask,
        labels: tuple[int, ...],
    ) -> Frame1DRows:
        reader = self._open_reader(
            task.request.catalog.artifact_path,
            entry=task.request.entry,
            include_thumbnail=False,
            resolve_source=False,
        )
        enter = getattr(type(reader), "__enter__", None)
        exit_ = getattr(type(reader), "__exit__", None)
        if not callable(enter) or not callable(exit_):
            raise TypeError("Browse 1-D reader is not an exact context owner")
        task.reader = reader
        # The exact reader is cleanup custody before entering it.  A foreign
        # return or a dependency which acquires then raises must not orphan an
        # open HDF handle.  FrameViewReader self-cleans failed entry and its
        # CLOSED exit is idempotent, so this also preserves its failure-total
        # contract.
        task.reader_entered = True
        entered = enter(reader)
        if entered is not reader:
            raise RuntimeError("Browse 1-D reader changed identity")
        rows = reader.read_1d_rows(
            labels, cancelled=lambda: self._checkpoint(task),
        )
        self._close_reader(task)
        if type(rows) is not Frame1DRows:
            raise TypeError("Browse 1-D reader returned a foreign projection")
        return rows

    @staticmethod
    def _borrowed_by_name(
        task: _HydrationTask,
        target: Browse1DHydrationTarget,
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for borrowed in task.borrows:
            key = borrowed.key
            if key.frame != target.ordinal or key.label != target.label:
                continue
            if key.name in result:
                raise RuntimeError("Browse 1-D survivor inventory is duplicated")
            result[key.name] = borrowed.array
        return result

    @staticmethod
    def _borrow_is_held(task: _HydrationTask, key: object) -> bool:
        return any(
            not borrowed.released and borrowed.key == key
            for borrowed in task.borrows
        )

    def _protect_keys(
        self,
        task: _HydrationTask,
        target: Browse1DHydrationTarget,
        keys: tuple[object, ...],
    ) -> None:
        """Pin one exact expected inventory until the final joint proof."""

        for key in keys:
            if (
                key.frame != target.ordinal
                or key.label != target.label
            ):
                raise ValueError("Browse 1-D expected inventory changed target")
            if not self._borrow_is_held(task, key):
                task.borrows.append(task.request.cache.borrow(
                    target.ordinal, target.label, key.name,
                ))
        actual = tuple(
            key for key in task.request.cache.resident_keys
            if key.frame == target.ordinal and key.label == target.label
        )
        if actual != keys or any(
            not self._borrow_is_held(task, key) for key in keys
        ):
            raise RuntimeError("Browse 1-D inventory changed while protected")

    def _borrow_survivors(
        self, task: _HydrationTask, target: Browse1DHydrationTarget,
    ) -> None:
        keys = tuple(
            key for key in task.request.cache.resident_keys
            if key.frame == target.ordinal and key.label == target.label
        )
        self._protect_keys(task, target, keys)

    def _singleton_rows(
        self,
        task: _HydrationTask,
        rows: Frame1DRows,
        target: Browse1DHydrationTarget,
    ) -> Frame1DRows:
        borrowed = self._borrowed_by_name(task, target)
        expected_names: set[str] = set()
        modes: list[Frame1DModeRows] = []
        for descriptor in task.request.catalog.axes_1d:
            mode, axis_label, axis_unit, axis_log = descriptor
            source = rows.mode(mode)
            if source is None:
                raise ValueError("Browse 1-D projection omitted a catalog mode")
            result = source.row(target.label)
            if result is None:
                modes.append(Frame1DModeRows(
                    mode,
                    source.axis,
                    (),
                    (),
                    None,
                ))
                continue
            intensity, sigma = result
            axis_name = browse_1d_row_name(mode, "axis")
            intensity_name = browse_1d_row_name(mode, "intensity")
            sigma_name = browse_1d_row_name(mode, "sigma")
            expected_names.update((axis_name, intensity_name))
            axis_values = borrowed.get(axis_name, source.axis.values)
            intensity = borrowed.get(intensity_name, intensity)
            if sigma is not None:
                expected_names.add(sigma_name)
                sigma = borrowed.get(sigma_name, sigma)
            elif sigma_name in borrowed:
                raise ValueError("Browse 1-D survivor has an unexpected sigma")
            modes.append(Frame1DModeRows(
                mode,
                Axis(axis_label, axis_unit, axis_log, axis_values),
                (target.label,),
                (intensity,),
                None if sigma is None else (sigma,),
            ))
        extras = set(borrowed).difference(expected_names)
        if extras:
            raise ValueError("Browse 1-D survivor inventory has extra rows")
        return Frame1DRows(
            rows.artifact_path,
            rows.entry,
            (target.label,),
            tuple(modes),
            rows.primary_mode,
        )

    def _settle_operation(self, task: _HydrationTask) -> str | None:
        operation = task.operation
        if operation is None:
            return None
        try:
            direction = operation.run()
        except BaseException:
            direction = operation.recover()
        if direction not in {"accepted", "rolled-back"}:
            raise RuntimeError("Browse 1-D cache operation did not accept")
        task.operation = None
        coordinate = task.pending_coordinate
        if coordinate is not None and direction == "accepted":
            with self._lock:
                self._known_keys[coordinate] = task.pending_keys
        task.pending_coordinate = None
        task.pending_keys = ()
        return direction

    @staticmethod
    def _release_borrows(task: _HydrationTask) -> None:
        while task.borrows:
            borrowed = task.borrows[0]
            borrowed.release()
            task.borrows.pop(0)

    @staticmethod
    def _close_reader(task: _HydrationTask) -> None:
        reader = task.reader
        if reader is None:
            task.reader_entered = False
            return
        if task.reader_entered:
            reader.__exit__(None, None, None)
        task.reader_entered = False
        task.reader = None

    def _store_target(
        self,
        task: _HydrationTask,
        rows: Frame1DRows,
        target: Browse1DHydrationTarget,
    ) -> None:
        self._borrow_survivors(task, target)
        singleton = self._singleton_rows(task, rows, target)
        try:
            receipt = task.request.cache.begin_store_1d_label(
                task.request.catalog,
                singleton,
                target.ordinal,
                target.label,
            )
        except Browse1DLabelStoreCustodyError as error:
            task.operation = error.operation
            raise
        task.operation = receipt.operation
        task.pending_coordinate = (target.ordinal, target.label)
        task.pending_keys = tuple(receipt.keys)
        direction = self._settle_operation(task)
        if direction not in {None, "accepted"}:
            raise RuntimeError("Browse 1-D cache operation rolled back")
        keys = tuple(receipt.keys)
        with self._lock:
            self._known_keys[(target.ordinal, target.label)] = keys
        task.pending_coordinate = None
        task.pending_keys = ()
        self._protect_keys(task, target, keys)

    def _protect_complete_targets(
        self,
        task: _HydrationTask,
        missing: tuple[Browse1DHydrationTarget, ...],
    ) -> None:
        missing_ids = {id(target) for target in missing}
        with self._lock:
            known = dict(self._known_keys)
        for target in task.request.targets:
            if id(target) in missing_ids:
                continue
            expected = known.get((target.ordinal, target.label))
            if expected is None:
                raise RuntimeError("Browse 1-D complete inventory is unknown")
            self._protect_keys(task, target, expected)

    def _prove_requested_resident(self, task: _HydrationTask) -> None:
        resident = task.request.cache.resident_keys
        with self._lock:
            known = dict(self._known_keys)
        for target in task.request.targets:
            expected = known.get((target.ordinal, target.label))
            if expected is None:
                raise RuntimeError("Browse 1-D requested inventory is unknown")
            actual = tuple(
                key for key in resident
                if key.frame == target.ordinal and key.label == target.label
            )
            if actual != expected or any(
                not self._borrow_is_held(task, key) for key in expected
            ):
                raise RuntimeError(
                    "Browse 1-D requested inventories are not jointly resident"
                )

    def _execute(self, task: _HydrationTask) -> Browse1DHydrationStatus:
        request = task.request
        if self._checkpoint(task):
            raise InterruptedError("Browse 1-D request is stale")
        missing = self._missing_targets(request)
        self._protect_complete_targets(task, missing)
        if not missing:
            self._prove_requested_resident(task)
            return Browse1DHydrationStatus.ALREADY_RESIDENT
        before = self._capture_bound_snapshot(request)
        if type(before) is not TargetSnapshot or before != request.target_snapshot:
            raise ValueError("Browse 1-D artifact changed before read")
        if self._checkpoint(task):
            raise InterruptedError("Browse 1-D request was cancelled")
        labels = tuple(target.label for target in missing)
        rows = self._open_and_read(task, labels)
        if (
            rows.artifact_path != request.catalog.artifact_path
            or rows.entry != request.entry
            or rows.labels != labels
        ):
            raise ValueError("Browse 1-D row projection changed identity")
        after = self._capture_bound_snapshot(request)
        if type(after) is not TargetSnapshot or after != request.target_snapshot:
            raise ValueError("Browse 1-D artifact changed during read")
        if self._checkpoint(task):
            raise InterruptedError("Browse 1-D request was superseded")
        if not request.gate.enter(request.epoch):
            raise InterruptedError("Browse 1-D commit gate refused")
        try:
            if self._checkpoint(task):
                raise InterruptedError("Browse 1-D request lost ownership")
            for target in missing:
                if self._checkpoint(task):
                    raise InterruptedError("Browse 1-D request was superseded")
                self._store_target(task, rows, target)
        finally:
            request.gate.leave()
        self._prove_requested_resident(task)
        if self._checkpoint(task):
            raise InterruptedError("Browse 1-D completion is stale")
        return Browse1DHydrationStatus.READY

    def _cleanup_task(self, task: _HydrationTask) -> bool:
        try:
            self._settle_operation(task)
            self._release_borrows(task)
            self._close_reader(task)
        except BaseException as error:
            task.diagnostic = self._diagnostic(error)
            task.cleanup_pending = True
            return False
        task.cleanup_pending = False
        return True

    def _run(self, task: _HydrationTask) -> None:
        status = task.status
        diagnostic = task.diagnostic
        if status is None:
            try:
                status = self._execute(task)
            except InterruptedError as error:
                status = (
                    Browse1DHydrationStatus.SUPERSEDED
                    if task.cancelled.is_set() else
                    Browse1DHydrationStatus.CANCELLED
                )
                diagnostic = self._diagnostic(error)
            except BaseException as error:
                status = Browse1DHydrationStatus.FAILED
                diagnostic = self._diagnostic(error)
        task.status = status
        task.diagnostic = diagnostic
        cleaned = self._cleanup_task(task)
        start: _HydrationTask | None = None
        with self._lock:
            task.running = False
            if self._active is not task:
                return
            if not cleaned:
                return
            actual = status
            if self._retiring:
                actual = Browse1DHydrationStatus.CANCELLED
            elif task.cancelled.is_set() or (
                self._wanted_identity is not task.request.identity
            ):
                actual = Browse1DHydrationStatus.SUPERSEDED
            completion = Browse1DHydrationCompletion(
                task.request.identity,
                actual,
                tuple(target.label for target in task.request.targets),
                diagnostic,
            )
            if not self._retiring:
                self._completions.put(completion)
                if actual is Browse1DHydrationStatus.READY:
                    self._ready_pending = True
                elif (
                    actual is Browse1DHydrationStatus.FAILED
                    and self._wanted_identity is task.request.identity
                ):
                    self._terminal_failure = (
                        task.signature,
                        diagnostic or "Browse 1-D hydration failed",
                    )
            self._active = None
            if self._queued is not None and not self._retiring:
                start = self._queued
                self._queued = None
                self._active = start
        if start is not None:
            self._start(start)

    def _progress(self) -> None:
        start: _HydrationTask | None = None
        with self._lock:
            task = self._active
            if (
                task is not None
                and task.cleanup_pending
                and not task.running
            ):
                start = task
        if start is not None:
            self._start(start)

    def consume_repaint(self) -> bool:
        self._progress()
        repaint = False
        while True:
            try:
                completion = self._completions.get_nowait()
            except Empty:
                break
            with self._lock:
                current = (
                    not self._retiring
                    and completion.request_identity is self._wanted_identity
                )
                if completion.status is Browse1DHydrationStatus.READY:
                    self._ready_pending = False
            if current and completion.status is Browse1DHydrationStatus.READY:
                repaint = True
        return repaint

    def polling_needed(self) -> bool:
        self._progress()
        with self._lock:
            return bool(
                self._active is not None
                or self._queued is not None
                or self._ready_pending
                or not self._completions.empty()
            )

    def release(self, *, preserve_pending_repaint: bool = False) -> bool:
        """Cancel/settle exact lane custody before cache/context cleanup."""

        with self._lock:
            if self._closed:
                return True
            if preserve_pending_repaint and self._ready_pending:
                return False
            self._retiring = True
            self._wanted_identity = None
            if self._active is not None:
                self._active.cancelled.set()
            if self._queued is not None:
                self._queued.cancelled.set()
                self._queued = None
        self._progress()
        with self._lock:
            if self._active is not None:
                return False
            self._ready_pending = False
            self._terminal_failure = None
            while True:
                try:
                    self._completions.get_nowait()
                except Empty:
                    break
            self._known_keys.clear()
            self._closed = True
            return True


__all__ = [
    "Browse1DHydrationCompletion",
    "Browse1DHydrationLane",
    "Browse1DHydrationRequest",
    "Browse1DHydrationStatus",
    "Browse1DHydrationTarget",
    "Browse1DLabelInventory",
    "Browse1DModeInventory",
]
