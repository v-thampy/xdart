"""Ordered provisional preview work on top of per-scan Live observations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import math

import numpy as np

from xrd_tools.analysis.live_preview_grid import enclosing_preview_bounds, preview_rsm_bins
from xrd_tools.core.scan import SourceKind


MAX_PREVIEW_POINT_BYTES = 128 * 1024 * 1024


@contextmanager
def preview_compute_slot(yield_priority, *, uses_xu=True):
    """Leave the XU waiter to native work and keep preview reads outside its lock."""
    if yield_priority():
        yield False
        return
    if not uses_xu:
        yield True
        return
    from xrd_tools.core.geometry.xu_runtime import XU_RUNTIME_LOCK

    if not XU_RUNTIME_LOCK.acquire(blocking=False):
        yield False
        return
    try:
        yield not yield_priority()
    finally:
        XU_RUNTIME_LOCK.release()


def largest_preview_point_planes(group):
    # The point-image policy is constant within one observed member.
    return max((len(member.points[0].policy.raw_indices(0)) for member in group.members
                if member.points and member.points[0].policy is not None), default=1)


@dataclass(frozen=True, slots=True)
class PreviewGroup:
    group: object
    rank: tuple[int, ...]
    members: tuple
    points: tuple
    ended: bool
    reason: str = ""
    member_ranks: dict = field(default_factory=dict)

    @property
    def identity(self):
        return self.group.identity

    def covered(self, count):
        """Keep the ordered point inputs that the displayed calculation used."""
        points = self.points[:count]
        members = tuple({id(member): member for member, _ in points}.values())
        return replace(self, points=points, members=members, ended=self.ended and count == len(self.points))

    @staticmethod
    def requirements(points):
        by_member = {}
        for member, point in points:
            by_member.setdefault(id(member), (member, []))[1].append(point)
        return by_member.values()

    def verifies(self, *, metadata=None, cancelled=None):
        from xrd_tools.sources.live_preview import preview_prefixes_hold

        return preview_prefixes_hold(
            self.requirements(self.points),
            metadata_requirements=() if metadata is None else metadata.requirements(metadata.points),
            cancelled=cancelled,
        )


@dataclass(frozen=True, slots=True)
class PreviewObservation:
    candidate: PreviewGroup | None
    ranks: dict
    observation_sweeps: int
    captures: int
    parser_opens: int


class PreviewGroupObserver:
    """Choose one candidate from the v3 listing, in the same metadata sweep."""

    def __init__(self, selection, read_options, select_labels):
        self.selection = selection
        self.read_options = read_options
        self.selector = None if select_labels is None else select_labels.__self__
        self.order = {}
        self.sweeps = self.captures = self.parser_opens = 0

    def __call__(self, rows, sweep):
        from xrd_tools.io.spec import spec_capture_activity
        from xrd_tools.sources.live_preview import observe_spec_member

        self.sweeps += 1
        files = {}
        for row in rows:
            for member in row.group.members:
                if member.kind is SourceKind.SPEC and member.path not in files:
                    files[member.path] = tuple(sweep.spec_keys(member.path))
        for position, (path, keys) in enumerate(files.items()):
            for ordinal, key in enumerate(keys):
                self.order.setdefault((path, key), (self.sweeps, -position, ordinal))
        for position, row in enumerate(rows):
            for ordinal, member in enumerate(row.group.members):
                if member.kind is not SourceKind.SPEC:
                    self.order.setdefault((member.path, member.key), (self.sweeps, -len(files) - position, ordinal))

        def newest(row):
            return max((self.order.get((item.path, item.key), (0, 0, 0)) for item in row.group.members),
                       default=(0, 0, 0))

        ranks = {row.group.identity: newest(row) for row in rows}
        candidate = waiting = None
        inspected = {}
        for row in sorted(rows, key=newest, reverse=True):
            raw_prefix = []
            members = []
            for item in row.group.members:
                if item.kind is not SourceKind.SPEC or item.key not in files.get(item.path, ()):
                    break
                key = item.path, item.key
                if key not in inspected:
                    inspected[key] = observe_spec_member(
                        item, self.selection, self.read_options, cancelled=sweep.cancelled,
                    )
                member = inspected[key]
                members.append(member)
                points = member.points
                if points and self.selector is not None:
                    stop = points[-1].label
                    if self.selector.stop_label is not None:
                        stop = min(stop, self.selector.stop_label)
                    try:
                        selected = (
                            replace(self.selector, stop_label=stop).select(tuple(point.label for point in points))
                            if stop >= self.selector.start_label else ()
                        )
                    except ValueError:
                        break
                    chosen = set(selected)
                    points = tuple(point for point in points if point.label in chosen)
                raw_prefix.extend((member, point) for point in points)
                if not member.end_of_stream:
                    break
            ended = len(members) == len(row.group.members) and all(member.end_of_stream for member in members)
            if raw_prefix:
                rank = max(self.order[member.path, member.key] for member, _ in raw_prefix)
                current = PreviewGroup(
                    row.group, rank, tuple(members), tuple(raw_prefix), ended,
                    member_ranks={(member.path, member.key): self.order[member.path, member.key]
                                  for member, _ in raw_prefix},
                )
                if candidate is None or current.rank > candidate.rank:
                    candidate = current
            elif members and waiting is None:
                waiting = PreviewGroup(row.group, ranks[row.group.identity], tuple(members), (), False,
                                       members[0].reason)
            if candidate is not None and candidate.points and candidate.rank == newest(row):
                break

        captures, parsers = spec_capture_activity()
        self.captures += captures
        self.parser_opens += parsers
        return PreviewObservation(candidate or waiting, ranks, self.sweeps, self.captures, self.parser_opens)


@dataclass(frozen=True, slots=True)
class PreviewSnapshot:
    group_identity: tuple
    rank: tuple
    points: tuple
    bounds: tuple
    axes: tuple
    intensity: np.ndarray
    input_prefix: PreviewGroup
    normalization: np.ndarray | None = None
    coverage: np.ndarray | None = None
    sigma: np.ndarray | None = None
    member_masks: dict = field(default_factory=dict)
    mask_pending: bool = False
    mask_update_queued: bool = False
    product: object = None
    display_bytes: int = 0
    metadata_prefix: PreviewGroup | None = None


@dataclass(frozen=True, slots=True)
class PreviewReplay:
    growth: bool
    mask: bool
    points: int
    span_before: tuple
    span_after: tuple


@dataclass(frozen=True, slots=True)
class PreviewProgress:
    included: int = 0
    held: int = 0
    excluded: int = 0
    waiting: bool = False
    end_of_stream: bool = False
    invalidated: int = 0
    decodes: int = 0
    forward_decodes: int = 0
    scout_decodes: int = 0
    captures: int = 0
    observation_sweeps: int = 0
    parser_opens: int = 0
    replay_target: int | None = None
    pending_watermark: int | None = None
    replays: tuple[PreviewReplay, ...] = ()
    replayed_points: int = 0
    resident_bytes: int = 0
    admitted_bytes: int = 0
    resource_restarts: int = 0

    @property
    def physical_replays(self):
        return len(self.replays)


@dataclass(slots=True)
class _HotMask:
    first: np.ndarray
    candidate: np.ndarray
    applied: np.ndarray | None = None
    count: int = 1
    applied_at: int = 0

    def add(self, point):
        self.count += 1
        self.candidate &= point == self.first

    @property
    def changed(self):
        return self.count >= 2 and (self.applied is None or not np.array_equal(self.applied, self.candidate))

    @property
    def due(self):
        return self.changed and (self.applied is None or self.count >= 2 * self.applied_at)

    def apply(self):
        if self.count >= 2:
            mask = self.candidate.copy()
            mask.setflags(write=False)
            self.applied = mask
            self.applied_at = self.count


@dataclass(slots=True)
class _RSMReplay:
    target: tuple
    cursor: int
    accumulator: object
    receipt: PreviewReplay


class RSMPreviewWork:
    """One active project's RSM accumulator, mask state and bounded staging."""

    def __init__(self, options):
        self.options = options
        self.progress = PreviewProgress()
        self.group = None
        self.kernel = self.bounds = self.grid = self.replay = None
        self.tokens = ()
        self.keys = ()
        self.decoded = None
        self.coordinates = {}
        self.masks = {}
        self.snapshot = None

    def _update(self, **changes):
        self.progress = replace(self.progress, **changes)

    def staging_bytes(self, group):
        options = self.options
        pixels = math.prod(options.detector_shape)
        grid = math.prod(preview_rsm_bins(options.bins)) * 4 * 8
        planes = min(pixels * 8 * largest_preview_point_planes(group), options.max_frame_bytes,
                     MAX_PREVIEW_POINT_BYTES)
        masks = 0 if options.conditioning.static_hot_threshold is None else pixels * 10 * len(group.members)
        # Four native grid channels, two resident grids, one candidate snapshot,
        # and one point's decoding, correction and coordinate working set.
        return 3 * grid + planes + masks + pixels * 128

    def reset(self, group):
        self.group = group
        self.kernel = self.bounds = self.grid = self.replay = self.snapshot = None
        self.tokens = self.keys = ()
        self.decoded = None
        self.coordinates.clear()
        self.masks.clear()
        self._update(included=0, waiting=False, end_of_stream=False, replay_target=None,
                     pending_watermark=None, resident_bytes=0, admitted_bytes=0)

    def retire_staging(self):
        if self.kernel is not None:
            self.reset(self.group)
            self._update(resource_restarts=self.progress.resource_restarts + 1)

    def verify_assets(self):
        if self.kernel is not None:
            self.kernel.verify_assets()

    def _decode(self, point, cancelled, *, replay):
        from xrd_tools.sources.live_preview import read_preview_point

        if self.decoded is None or self.decoded[0] != point.identity:
            options = self.options
            count = 1 if point.policy is None else len(point.policy.raw_indices(0))
            bound = min(MAX_PREVIEW_POINT_BYTES, options.max_frame_bytes, self.kernel.max_chunk_bytes)
            if math.prod(options.detector_shape) * 8 * count > bound:
                raise ValueError("The grouped detector point exceeds the preview memory bound")
            planes = read_preview_point(point, cancelled=cancelled)
            count = len(planes)
            self._update(decodes=self.progress.decodes + count,
                         forward_decodes=self.progress.forward_decodes + (0 if replay else count))
            self.decoded = point.identity, planes

    def _ready_frame(self, member, point, session):
        coordinates = self.coordinates.get(member.key)
        if coordinates is None:
            coordinates = self.kernel.coordinates(member, point, session)
            self.coordinates[member.key] = coordinates
        _, planes = self.decoded
        self.decoded = None
        return self.kernel.condition(point, planes), self.kernel.angles(point), coordinates

    def _mask(self, member, sample):
        cutoff = self.kernel.hot_threshold
        if cutoff is None:
            return None
        key = member.key
        signal = sample[0]
        if key not in self.masks:
            first = signal.copy()
            self.masks[key] = _HotMask(first, first > cutoff)
        else:
            self.masks[key].add(signal)
        return self.masks[key]

    def _add(self, grid, member, point, sample, angles, coordinates, session):
        mask = self.masks.get(member.key)
        self.kernel.add(grid, point, sample, angles, coordinates, None if mask is None else mask.applied,
                        session)

    def _step(self, group, cancelled, yield_priority, progress):
        allowed = 8
        while allowed and not cancelled() and not yield_priority():
            stage = self.replay
            offset = len(self.tokens)
            if stage is None and offset >= len(group.points):
                break
            member, point = group.points[offset] if stage is None else stage.target[stage.cursor]
            self._decode(point, cancelled, replay=stage is not None)
            if cancelled() or yield_priority():
                break
            allowed -= 1
            with preview_compute_slot(yield_priority) as admitted:
                if not admitted:
                    return self.snapshot
                with self.kernel.runtime() as session:
                    sample, angles, coordinates = self._ready_frame(member, point, session)
                    self._accumulate(group, stage, offset, member, point, sample, angles, coordinates,
                                     session, progress)
        if self.replay is not None or not self.keys or cancelled() or yield_priority():
            return self.snapshot
        if self.snapshot is not None and self.snapshot.points == self.keys:
            return self.snapshot
        with preview_compute_slot(yield_priority) as admitted:
            if not admitted:
                return self.snapshot
            with self.kernel.runtime() as session:
                volume = self.kernel.volume(self.grid, session)
        members = {key: mask.applied for key, mask in self.masks.items() if mask.applied is not None}
        self._update(included=len(self.keys), waiting=bool(group.reason),
                     end_of_stream=group.ended and len(self.keys) == len(group.points))
        storage = volume.intensity.nbytes + sum(axis.nbytes for _, axis in volume.axes)
        covered = group.covered(len(self.keys))
        rank = max(group.member_ranks.get((member.path, member.key), group.rank) for member in covered.members)
        self.snapshot = PreviewSnapshot(
            group.identity, rank, self.keys, tuple(self.bounds.values()), volume.axes, volume.intensity,
            member_masks=members, mask_pending=any(mask.count == 1 for mask in self.masks.values()),
            mask_update_queued=any(mask.changed for mask in self.masks.values()), product=volume,
            # Charge the detached volume, viewer conversion, and projection cache together.
            display_bytes=3 * storage + sum(mask.nbytes for mask in members.values()) + self.kernel.witness_bytes,
            input_prefix=covered,
        )
        return self.snapshot

    def _accumulate(self, group, stage, offset, member, point, sample, angles, coordinates, session, progress):
        if stage is not None:
            self._add(stage.accumulator, member, point, sample, angles, coordinates, session)
            stage.cursor += 1
            if stage.cursor == len(stage.target):
                self.grid = stage.accumulator
                self.replay = None
                self._update(replay_target=None, replays=(*self.progress.replays, stage.receipt),
                             replayed_points=self.progress.replayed_points + stage.cursor)
            return
        mask = self._mask(member, sample[1])
        footprint = self.kernel.footprint(angles, coordinates, session)
        bounds = enclosing_preview_bounds(footprint, frozenset(footprint), self.bounds)
        grown = self.bounds is not None and bounds != self.bounds
        changed_mask = mask is not None and (mask.due or grown and mask.changed)
        self.tokens = (*self.tokens, point.identity)
        self.keys = (*self.keys, (member.key, point.label))
        if self.grid is None:
            self.grid = self.kernel.new_grid(bounds, session)
        if grown or changed_mask:
            if changed_mask:
                mask.apply()
            previous = tuple(hi - lo for lo, hi in self.bounds.values()) if self.bounds is not None else ()
            current = tuple(hi - lo for lo, hi in bounds.values())
            self.replay = _RSMReplay(
                group.points[:offset + 1], 0, self.kernel.new_grid(bounds, session),
                PreviewReplay(grown, changed_mask, offset + 1, previous, current),
            )
            self._update(replay_target=offset + 1)
            progress("Rebuilding preview", 0, 1)
        else:
            self._add(self.grid, member, point, sample, angles, coordinates, session)
        self.bounds = bounds

    def run(self, group, cancelled, yield_priority, progress):
        from xrd_tools.analysis.live_preview_rsm import RSMPreviewKernel
        from xrd_tools.sources.live_preview import PreviewPointPending, preview_source_scope

        current = tuple(point.identity for _member, point in group.points)
        if self.group != group.identity or current[:len(self.tokens)] != self.tokens:
            self.reset(group.identity)
        if not group.points:
            self._update(waiting=True)
            return None
        try:
            with preview_source_scope():
                if self.kernel is None:
                    self.kernel = RSMPreviewKernel(
                        self.options, difference=any(point.policy is not None and point.policy.operation == "difference"
                                                     for _member, point in group.points),
                    )
                else:
                    self.verify_assets()
                snapshot = self._step(group, cancelled, yield_priority, progress)
            if snapshot is not None and not snapshot.input_prefix.verifies(cancelled=cancelled):
                self.invalidate(group.identity)
                return None
            grid_bytes = math.prod(self.kernel.bins) * 4 * 8
            live_bytes = grid_bytes * (2 if self.replay is not None else 1)
            live_bytes += sum(mask.first.nbytes + mask.candidate.nbytes
                              + (mask.applied.nbytes if mask.applied is not None else 0)
                              for mask in self.masks.values())
            live_bytes += 0 if snapshot is None else snapshot.intensity.nbytes
            self._update(resident_bytes=live_bytes, admitted_bytes=self.kernel.max_chunk_bytes + 3 * grid_bytes)
            return snapshot
        except (OSError, ValueError, PreviewPointPending):
            if self.tokens:
                self.invalidate(group.identity)
            self._update(waiting=True)
            return None

    def invalidate(self, group):
        self.reset(group)
        self._update(invalidated=self.progress.invalidated + 1)
