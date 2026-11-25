"""Ordered provisional preview work on top of per-scan Live observations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from xrd_tools.analysis.live_preview_grid import enclosing_preview_bounds
from xrd_tools.core.scan import SourceKind


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
        self.select_labels = select_labels
        self.order = {}
        self.sweeps = self.captures = self.parser_opens = 0

    def __call__(self, rows, sweep):
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

        def newest(row):
            return max((self.order.get((item.path, item.key), (0, 0, 0)) for item in row.group.members),
                       default=(0, 0, 0))

        ranks = {row.group.identity: newest(row) for row in rows}
        candidate = None
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
                if points and self.select_labels is not None:
                    chosen = set(self.select_labels(tuple(point.label for point in points)))
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
            elif members and candidate is None:
                candidate = PreviewGroup(row.group, ranks[row.group.identity], tuple(members), (), False,
                                         members[0].reason)
            if candidate is not None and candidate.points and candidate.rank == newest(row):
                break

        # v3 and preview share the capture owner. Count the SPEC files actually
        # inspected for point evidence, once within this observation.
        count = len({path for path, _key in inspected})
        self.captures += count
        self.parser_opens += count
        return PreviewObservation(candidate, ranks, self.sweeps, self.captures, self.parser_opens)


@dataclass(frozen=True, slots=True)
class PreviewSnapshot:
    group_identity: tuple
    rank: tuple
    points: tuple
    bounds: tuple
    axes: tuple
    intensity: np.ndarray
    normalization: np.ndarray | None = None
    coverage: np.ndarray | None = None
    sigma: np.ndarray | None = None
    member_masks: dict = field(default_factory=dict)
    mask_pending: bool = False
    mask_update_queued: bool = False
    product: object = None


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
        self.coordinates = {}
        self.masks = {}
        self.snapshot = None

    def _update(self, **changes):
        self.progress = replace(self.progress, **changes)

    def reset(self, group):
        self.group = group
        self.kernel = self.bounds = self.grid = self.replay = self.snapshot = None
        self.tokens = self.keys = ()
        self.coordinates.clear()
        self.masks.clear()
        self._update(included=0, waiting=False, end_of_stream=False, replay_target=None)

    @staticmethod
    def verifies(group, *, cancelled=None):
        from xrd_tools.sources.live_preview import preview_prefixes_hold

        return preview_prefixes_hold(
            ((member, tuple(point for source, point in group.points if source is member)) for member in group.members),
            cancelled=cancelled,
        )

    def verify_assets(self):
        if self.kernel is not None:
            self.kernel.verify_assets()

    def _read(self, member, point, session, cancelled, *, replay=False):
        from xrd_tools.sources.live_preview import read_preview_point

        coordinates = self.coordinates.get(member.key)
        if coordinates is None:
            coordinates = self.kernel.coordinates(member, session)
            self.coordinates[member.key] = coordinates
        planes = read_preview_point(point, cancelled=cancelled)
        count = len(planes)
        self._update(decodes=self.progress.decodes + count,
                     forward_decodes=self.progress.forward_decodes + (0 if replay else count))
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
        with self.kernel.runtime() as session:
            while allowed and not cancelled() and not yield_priority():
                if self.replay is not None:
                    stage = self.replay
                    if stage.cursor < len(stage.target):
                        member, point = stage.target[stage.cursor]
                        sample, angles, coordinates = self._read(member, point, session, cancelled, replay=True)
                        self._add(stage.accumulator, member, point, sample, angles, coordinates, session)
                        stage.cursor += 1
                        allowed -= 1
                    if stage.cursor == len(stage.target):
                        self.grid = stage.accumulator
                        self.replay = None
                        self._update(replay_target=None, replays=(*self.progress.replays, stage.receipt),
                                     replayed_points=self.progress.replayed_points + stage.cursor)
                    continue
                offset = len(self.tokens)
                if offset >= len(group.points):
                    break
                member, point = group.points[offset]
                sample, angles, coordinates = self._read(member, point, session, cancelled)
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
                    allowed -= 1
                self.bounds = bounds
            if self.replay is not None or not self.keys or cancelled() or yield_priority():
                return self.snapshot
            if self.snapshot is not None and self.snapshot.points == self.keys:
                return self.snapshot
            volume = self.kernel.volume(self.grid, session)
        members = {key: mask.applied for key, mask in self.masks.items() if mask.applied is not None}
        self._update(included=len(self.keys), waiting=bool(group.reason),
                     end_of_stream=group.ended and len(self.keys) == len(group.points))
        self.snapshot = PreviewSnapshot(
            group.identity, group.rank, self.keys, tuple(self.bounds.values()), volume.axes, volume.intensity,
            member_masks=members, mask_pending=any(mask.count == 1 for mask in self.masks.values()),
            mask_update_queued=any(mask.changed for mask in self.masks.values()), product=volume,
        )
        return self.snapshot

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
            if snapshot is not None and not self.verifies(group, cancelled=cancelled):
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
