"""One-active/one-latest saved-cake cuts, with no Qt or retained cake stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from threading import Event, Lock, Thread

import numpy as np

from xdart.modules.display_context import BrowseContext, ContextKind, DisplaySelection
from xrd_tools.io import FrameScalarCatalog, FrameViewReader
from xrd_tools.io.output_transaction import (
    TargetSnapshot, revalidate_stream_terminal,
    stream_terminal_object_revision,
)

from .display_values import DisplayFrameKey, StandardDisplayPayload
from .scientific_axes import DERIVED_Q_CHI, present_gi_map, trace_projection
from .shell_values import (
    FrameNavigationProjection, PinnedTraceProjection, SlicePin, TraceProjection,
)


@dataclass(frozen=True, slots=True)
class BrowseSliceResult:
    pending: bool = False
    diagnostic: str = ""
    traces: tuple[TraceProjection, ...] = ()
    pinned_traces: tuple[PinnedTraceProjection, ...] = ()


@dataclass(frozen=True, slots=True)
class _Request:
    key: tuple[object, ...]
    selection: DisplaySelection
    selected: tuple[DisplayFrameKey, ...]
    pins: tuple[SlicePin, ...]
    axis: str
    sliced: bool
    center: float
    width: float
    norm: str
    cached_pins: tuple[PinnedTraceProjection, ...]
    #: The 2-D map the pane shows; cuts are read off that map for every frame.
    image_map: str = ""


@dataclass(slots=True)
class _Task:
    request: _Request
    cancelled: Event
    thread: Thread | None = None
    reader: FrameViewReader | None = None
    executed: bool = False
    result: BrowseSliceResult = BrowseSliceResult()


class BrowseSliceLane:
    """Project only the latest exact selection; close readers on the worker.

    ``cancel`` supersedes pending work and drops curves but leaves the lane
    reusable. ``release`` also retires admission and reports a nonblocking
    close gate. A failed reader close retains that same reader for its ordinary
    retryable exit; it never permits another batch to open alongside it.
    """

    def __init__(self, browse: BrowseContext) -> None:
        if type(browse) is not BrowseContext:
            raise TypeError("Browse slice requires one exact context")
        from .context_projection import _browse_wavelength

        self._browse = browse
        self._catalog = browse.scalar_catalog
        self._gate = browse.commit_gate
        self._epoch = browse.commit_epoch
        self._snapshot = browse.target_snapshot
        self._load_request = browse.load_request
        self._terminal = getattr(self._load_request, "terminal_commit_identity", None)
        self._wavelength = _browse_wavelength(browse)
        self._lock = Lock()
        self._active: _Task | None = None
        self._queued: _Task | None = None
        self._wanted: tuple[object, ...] | None = None
        self._result: BrowseSliceResult | None = None
        self._pin_cache: dict[tuple[object, ...], PinnedTraceProjection] = {}
        self._repaint = False
        self._retiring = False
        if not self._context_live():
            raise RuntimeError("Browse slice context is not exactly admitted")
        self._revision = self._stat_revision()
        if self._revision[:4] != (
            self._snapshot.device, self._snapshot.inode,
            self._snapshot.size, self._snapshot.mtime_ns,
        ):
            raise RuntimeError("Browse slice saved artifact changed after loading")

    def _context_live(self) -> bool:
        browse, catalog = self._browse, self._catalog
        return bool(
            type(catalog) is FrameScalarCatalog
            and type(self._snapshot) is TargetSnapshot and self._snapshot.exists
            and browse.scalar_catalog is catalog
            and browse.target_snapshot is self._snapshot
            and browse.load_request is self._load_request
            and browse.commit_gate is self._gate
            and browse.commit_epoch == self._epoch
            and catalog.labels is browse.frame_ids
            and catalog.labels is browse.loaded_labels
            and catalog.artifact_path == browse.requested_path
            and catalog.entry == browse.target_entry
            and browse.loaded and not browse.invalidated and not browse.released
            and not self._gate.cancelled
        )

    def _selection_live(self, selection: DisplaySelection) -> bool:
        return bool(
            self._context_live()
            and type(selection) is DisplaySelection
            and selection.kind is ContextKind.BROWSE
            and selection.names(self._browse)
            and selection.owner == self._browse.hydration_owner
        )

    def _request(self, selection, navigation, preferences, norm) -> _Request:
        if (not self._selection_live(selection)
                or type(navigation) is not FrameNavigationProjection):
            raise ValueError("Browse slice selection is not current")
        frames = navigation.frames
        identity = frames[0].run_identity if frames else None
        previous = 0
        for frame in frames:
            if (frame.run_identity is not identity
                    or frame.source_scan != self._browse.scan_key
                    or frame.artifact != self._catalog.artifact_path
                    or not previous < frame.work_ordinal <= len(self._catalog.labels)
                    or self._catalog.labels[frame.work_ordinal - 1] != frame.local_frame_label):
                raise ValueError("Browse slice frame is outside its scalar catalog")
            previous = frame.work_ordinal
        owned = {id(frame) for frame in frames}
        pins = tuple(preferences.slice_pins)
        if any(type(pin) is not SlicePin or id(pin.frame) not in owned for pin in pins):
            raise ValueError("Browse slice pin is not an owned frame")
        pins = tuple(pin for pin in pins if pin.plot_axis == preferences.plot_axis)
        axis = str(preferences.plot_axis)
        sliced = bool(preferences.slice_enabled)
        center, width = float(preferences.slice_center), float(preferences.slice_width)
        if not np.isfinite(center) or not np.isfinite(width) or width < 0:
            raise ValueError("Browse slice settings are invalid")
        image_map = str(preferences.image_axis)
        key = (
            id(selection), selection.display_generation,
            tuple(id(frame) for frame in navigation.selected), axis,
            sliced, center, width, tuple(pin.projection_id for pin in pins), norm,
            image_map,
        )
        with self._lock:
            cached = tuple(
                replace(found, pin=pin) for pin in pins
                if (found := self._pin_cache.get((*pin.projection_id, norm))) is not None
            )
        return _Request(key, selection, navigation.selected, pins,
                        axis, sliced, center, width, norm, cached, image_map)

    def project(self, selection, navigation, *, preferences, norm_channel="") -> BrowseSliceResult:
        try:
            request = self._request(selection, navigation, preferences, norm_channel)
        except Exception as error:
            self.cancel()
            return BrowseSliceResult(diagnostic=self._diagnostic(error))
        with self._lock:
            if self._retiring:
                return BrowseSliceResult(diagnostic="Browse slice lane is released")
            if request.key != self._wanted:
                self._wanted = request.key
                self._result = None
                self._repaint = False
                self._pin_cache = {
                    (*item.pin.projection_id, request.norm): item
                    for item in request.cached_pins
                }
                if self._active is not None:
                    self._active.cancelled.set()
                self._queued = _Task(request, Event())
        self._progress()
        with self._lock:
            return self._result or BrowseSliceResult(pending=True)

    @staticmethod
    def _diagnostic(error: BaseException) -> str:
        return f"{type(error).__name__}: {error}"[:512]

    def _cancelled(self, task: _Task) -> bool:
        with self._lock:
            stale = self._retiring or task.request.key != self._wanted
        return stale or task.cancelled.is_set() or not self._selection_live(task.request.selection)

    def _validate_snapshot(self) -> None:
        terminal = self._terminal
        unchanged = (
            revalidate_stream_terminal(self._catalog.artifact_path, terminal) == self._snapshot
            if stream_terminal_object_revision(terminal) is not None
            else self._stat_revision() == self._revision
        )
        if not unchanged:
            raise RuntimeError("Browse slice saved artifact changed after loading")

    def _stat_revision(self) -> tuple[int, ...]:
        # The loader already hashed this admitted file. Display reads only
        # requalify its object/stat identity, including the observed ctime.
        observed = os.stat(self._catalog.artifact_path)
        return (observed.st_dev, observed.st_ino, observed.st_size,
                observed.st_mtime_ns, observed.st_ctime_ns)

    def _cut(self, payload, axis, sliced, center, width, norm):
        if sliced or axis == "chi":
            if not payload.view.has_2d:
                raise RuntimeError(f"Browse slice has no saved 2-D data for frame {payload.view.label}")
            # Cuts and full chi projections require a cake. Neither may fall
            # back to the stored radial 1-D row while that cake is unavailable.
            payload = replace(payload, view=replace(
                payload.view, axis_1d=None, intensity_1d=None, sigma_1d=None,
            ))
        trace = trace_projection(
            payload, requested_axis=axis, allow_cake=True, slice_enabled=sliced,
            slice_center=center, slice_width=width, norm_channel=norm,
        )
        if trace is None:
            return None
        # No retained array may keep a reader-owned cake or row alive through .base.
        x, y = np.array(trace.axis.values, copy=True), np.array(trace.intensity, copy=True)
        x.setflags(write=False)
        y.setflags(write=False)
        return replace(trace, axis=replace(trace.axis, values=x), intensity=y)

    def _execute(self, task: _Task) -> BrowseSliceResult:
        request = task.request
        if self._cancelled(task):
            return BrowseSliceResult()
        self._validate_snapshot()
        pins = {item.pin.projection_id: item for item in request.cached_pins}
        selected = {id(frame): frame for frame in request.selected}
        frames = dict(selected)
        for pin in request.pins:
            if pin.projection_id not in pins:
                frames[id(pin.frame)] = pin.frame
        traces = {}
        if frames:
            reader = FrameViewReader(self._catalog.artifact_path, entry=self._catalog.entry,
                                     include_thumbnail=False, resolve_source=False)
            task.reader = reader
            if reader.__enter__() is not reader:
                raise RuntimeError("Browse slice reader changed identity")
            for frame in sorted(frames.values(), key=lambda item: item.work_ordinal):
                if self._cancelled(task):
                    return BrowseSliceResult()
                row = self._catalog.row(frame.local_frame_label)
                # Cut the map the pane shows: another DIRECT map the file holds
                # for this frame is read as such; the display-only derived q–χ
                # is re-binned from the primary cake, exactly as the pane does.
                shown = request.image_map
                direct = shown != row.active_mode_2d and shown in (row.modes_2d or ())
                view = reader.read(row.label, mode_1d=row.active_mode_1d,
                                   mode_2d=shown if direct else row.active_mode_2d)
                if self._cancelled(task):
                    return BrowseSliceResult()
                payload = StandardDisplayPayload(
                    request.selection.display_generation, frame, "", view,
                    status="browse", wavelength_m=self._wavelength, averaged=row.averaged,
                )
                if shown == DERIVED_Q_CHI and row.active_mode_2d == "qip_qoop":
                    payload = present_gi_map(replace(
                        payload, measurement_mode="GI", gi_mode_2d="qip_qoop",
                    ), shown)
                if id(frame) in selected:
                    trace = self._cut(payload, request.axis, request.sliced,
                                      request.center, request.width, request.norm)
                    if trace is not None:
                        traces[id(frame)] = trace
                for pin in request.pins:
                    if pin.frame is frame and pin.projection_id not in pins:
                        trace = self._cut(payload, pin.plot_axis, True,
                                          pin.center, pin.width, request.norm)
                        if trace is not None:
                            pins[pin.projection_id] = PinnedTraceProjection(pin, trace)
                del payload, view
        pinned = tuple(pins[pin.projection_id] for pin in request.pins if pin.projection_id in pins)
        absorbed = {
            id(item.pin.frame) for item in pinned
            if request.sliced and item.pin.center == request.center and item.pin.width == request.width
        }
        return BrowseSliceResult(
            traces=tuple(traces[id(frame)] for frame in request.selected
                         if id(frame) in traces and id(frame) not in absorbed),
            pinned_traces=pinned,
        )

    def _run(self, task: _Task) -> None:
        if not task.executed:
            try:
                task.result = self._execute(task)
            except BaseException as error:
                task.result = BrowseSliceResult(diagnostic=self._diagnostic(error))
            task.executed = True
        if task.reader is not None:
            try:
                task.reader.__exit__(None, None, None)
            except BaseException as error:
                task.result = BrowseSliceResult(diagnostic=self._diagnostic(error))
                return
            task.reader = None
        if self._cancelled(task):
            task.result = BrowseSliceResult()
            return
        try:
            self._validate_snapshot()
        except BaseException as error:
            task.result = BrowseSliceResult(diagnostic=self._diagnostic(error))
        with self._lock:
            if (not self._retiring and not task.cancelled.is_set()
                    and task.request.key == self._wanted):
                self._result = task.result
                self._pin_cache = {(*item.pin.projection_id, task.request.norm): item
                                   for item in task.result.pinned_traces}
                self._repaint = True

    def _progress(self) -> None:
        with self._lock:
            task = self._active
            if task is not None:
                if task.thread is not None and task.thread.is_alive():
                    return
                if task.reader is None:
                    self._active = None
            if self._active is None and self._queued is not None and not self._retiring:
                self._active, self._queued = self._queued, None
            task = self._active
            if task is not None:
                task.thread = Thread(target=self._run, args=(task,), name="BrowseSlice", daemon=True)
                try:
                    task.thread.start()
                except RuntimeError as error:
                    self._result = BrowseSliceResult(diagnostic=self._diagnostic(error))
                    self._repaint = True
                    if task.reader is None:
                        self._active = None

    def consume_repaint(self) -> bool:
        self._progress()
        with self._lock:
            repaint, self._repaint = self._repaint, False
        return repaint

    def polling_needed(self) -> bool:
        self._progress()
        with self._lock:
            return self._active is not None or self._queued is not None or self._repaint

    def cancel(self) -> None:
        with self._lock:
            self._wanted = None
            self._queued = None
            self._result = None
            self._pin_cache.clear()
            self._repaint = False
            if self._active is not None:
                self._active.cancelled.set()
        self._progress()

    def release(self, *, preserve_pending_repaint=False) -> bool:
        with self._lock:
            if preserve_pending_repaint and self._repaint:
                return False
            self._retiring = True
        self.cancel()
        with self._lock:
            return self._active is None


__all__ = ["BrowseSliceLane", "BrowseSliceResult"]
