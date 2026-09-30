"""Dropped-frame screening for streamed detector scans.

A frame whose detector counts collapse while the beam monitor looks normal (a
missed trigger, a closed shutter, an unfinished readout) still carries its full
solid-angle weight, so it dilutes every bin it touches. Each frame's statistic
is its monitor-normalised total over all valid detector pixels (not only the
output range, which legitimately covers less of the frames at its edges). It is
compared with the median of up to ``window`` neighbours on each side, in stream
order, that share its row (for Stitch, the outer detector circle, e.g. nu). A
frame below
``empty_ratio`` of that reference is ``empty``; below ``partial_ratio`` it is
``partial``. The policy excludes empty frames by default and partial frames
only on request. A frame with fewer than two same-row neighbours is kept
``unscreened``: a lone frame has no reference.

A decision needs at most ``window`` frames of look-ahead, so streaming callers
hold that many per-frame contributions and merge the released ones in their
original order. When nothing is excluded the merge is unchanged.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

EXCLUDE_CHOICES = ('none', 'empty', 'partial')


@dataclass(frozen=True)
class FrameScreenPolicy:
    """Which dropped frames a streamed reduction excludes."""

    exclude: str = 'empty'
    empty_ratio: float = 0.01
    partial_ratio: float = 0.2
    window: int = 2

    def __post_init__(self) -> None:
        if self.exclude not in EXCLUDE_CHOICES:
            raise ValueError(f'frame screen exclude must be one of {EXCLUDE_CHOICES}')
        for name in ('empty_ratio', 'partial_ratio'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('frame screen ratios must be finite numbers')
            object.__setattr__(self, name, float(value))
        if not 0 < self.empty_ratio < self.partial_ratio < 1:
            raise ValueError('frame screen ratios must satisfy 0 < empty < partial < 1')
        if type(self.window) is not int or not 1 <= self.window <= 8:
            raise ValueError('frame screen window must be an integer from 1 to 8')

    def identity(self) -> tuple[object, ...]:
        return ('frame-screen-v1', self.exclude, self.empty_ratio, self.partial_ratio, self.window)

    def to_provenance(self) -> dict[str, object]:
        return {'exclude': self.exclude, 'empty_ratio': self.empty_ratio,
                'partial_ratio': self.partial_ratio, 'window': self.window,
                'statistic': 'monitor-normalised total of valid detector pixels, '
                             'dark-subtracted when a dark is set'}


@dataclass(frozen=True)
class FrameVerdict:
    """The screening outcome for one frame."""

    position: int
    label: object
    total: float
    reference: float | None
    ratio: float | None
    status: str
    excluded: bool

    def to_dict(self) -> dict[str, object]:
        return {'position': self.position, 'label': self.label, 'total': self.total,
                'reference': self.reference, 'ratio': self.ratio, 'status': self.status,
                'excluded': self.excluded}


def _verdict(policy, position, label, total, neighbours):
    if len(neighbours) < 2:
        return FrameVerdict(position, label, total, None, None, 'unscreened', False)
    reference = float(np.median(neighbours))
    if not math.isfinite(total) or reference <= 0:
        return FrameVerdict(position, label, total, reference, None, 'unscreened', False)
    ratio = total / reference
    status = ('empty' if ratio < policy.empty_ratio
              else 'partial' if ratio < policy.partial_ratio else 'ok')
    excluded = (status == 'empty' and policy.exclude in ('empty', 'partial')) or (
        status == 'partial' and policy.exclude == 'partial')
    return FrameVerdict(position, label, total, reference, ratio, status, excluded)


def screen_totals(totals, labels, rows, policy: FrameScreenPolicy) -> tuple[FrameVerdict, ...]:
    """Screen a complete sequence of per-frame totals (same rule as the stream)."""
    totals = [float(t) for t in totals]
    if not (len(totals) == len(labels) == len(rows)):
        raise ValueError('frame screen totals, labels and rows differ in length')
    out = []
    for i, total in enumerate(totals):
        window = range(max(0, i - policy.window), min(len(totals), i + policy.window + 1))
        neighbours = [totals[j] for j in window if j != i and rows[j] == rows[i] and math.isfinite(totals[j])]
        out.append(_verdict(policy, i, labels[i], total, neighbours))
    return tuple(out)


class FrameScreen:
    """Screen a stream of frames with at most ``policy.window`` frames of look-ahead."""

    def __init__(self, policy: FrameScreenPolicy, labels, rows) -> None:
        if type(policy) is not FrameScreenPolicy:
            raise TypeError('frame screen requires an exact FrameScreenPolicy')
        self.policy = policy
        self._labels = tuple(labels)
        self._rows = tuple(rows)
        if len(self._labels) != len(self._rows):
            raise ValueError('frame screen labels and rows differ in length')
        self._totals: list[float] = []
        self._pending: deque = deque()
        self._verdicts: list[FrameVerdict] = []

    def _decide(self, position):
        i, w = position, self.policy.window
        window = range(max(0, i - w), min(len(self._totals), i + w + 1))
        neighbours = [self._totals[j] for j in window
                      if j != i and self._rows[j] == self._rows[i] and math.isfinite(self._totals[j])]
        verdict = _verdict(self.policy, i, self._labels[i], self._totals[i], neighbours)
        self._verdicts.append(verdict)
        return verdict

    def push(self, total, item):
        """Record the next frame; return the (item, verdict) pairs now decided, in order."""
        position = len(self._totals)
        if position >= len(self._labels):
            raise ValueError('frame screen received more frames than declared')
        self._totals.append(float(total))
        self._pending.append((position, item))
        released = []
        while self._pending and self._pending[0][0] <= position - self.policy.window:
            earlier, held = self._pending.popleft()
            released.append((held, self._decide(earlier)))
        return released

    def finish(self):
        """Decide the frames still held at the end of the stream."""
        released = []
        while self._pending:
            earlier, held = self._pending.popleft()
            released.append((held, self._decide(earlier)))
        return released

    @property
    def verdicts(self) -> tuple[FrameVerdict, ...]:
        return tuple(self._verdicts)

    def summary(self) -> dict[str, object]:
        verdicts = self.verdicts
        return {'policy': self.policy.to_provenance(),
                'screened': sum(v.status != 'unscreened' for v in verdicts),
                'excluded': [v.label for v in verdicts if v.excluded],
                'empty': [v.label for v in verdicts if v.status == 'empty'],
                'partial': [v.label for v in verdicts if v.status == 'partial'],
                'frames': [v.to_dict() for v in verdicts if v.status in ('empty', 'partial')]}


__all__ = ['EXCLUDE_CHOICES', 'FrameScreen', 'FrameScreenPolicy', 'FrameVerdict', 'screen_totals']
