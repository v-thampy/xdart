# -*- coding: utf-8 -*-
"""``DirectoryIndex`` — a headless, incremental, poll-based index of source
candidates under one directory root (R1).

It owns only source candidates and their cheap filesystem state.  It is NOT a
frame store, processed-record store, metadata cache, GUI model, or filesystem
watcher dependency — those stay owned by the modules that already own them
(:mod:`xrd_tools.session`, :mod:`xrd_tools.io.read`, the GUI display layer).
R2 will add one-open HDF5 cursors that CONSUME these candidates; this module
does not anticipate that shape.

Every :meth:`DirectoryIndex.poll` call is a plain, synchronous, name-only
:func:`~xrd_tools.sources.discover.enumerate_candidates` sweep — zero HDF5
opens, on both the first poll and every unchanged one after it.  An unchanged
poll returns the *same* :class:`Snapshot` object (not a new-but-equal one) so
a caller never re-sorts or reconstructs an unchanged candidate list.

:meth:`DirectoryIndex.record_probe` layers BOUNDED RETRY on top of an
explicit, caller-invoked probe (:mod:`xrd_tools.sources.adapters`,
:mod:`xrd_tools.sources.probe`) — the index itself never probes.  This is the
"newly readable HDF5 shell... remains provisional... not permanently retired
as imageless" policy: a fresh IN_PROGRESS/IMAGELESS verdict is held as
IN_PROGRESS for up to ``retry_deadline`` seconds (an injectable clock, no
wall-clock sleeps in tests) before the raw verdict is trusted, and the window
resets whenever the candidate's file stamp changes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrd_tools.sources.discover import Candidate, enumerate_candidates
from xrd_tools.sources.probe import ProbeResult, ProbeState

_UNSET: Any = object()

#: Probe states that leave a candidate's finality in doubt — the ONLY states
#: DirectoryIndex tracks bounded retry for (handoff: "retry state associated
#: only with provisional candidates").  READY/PROCESSED_OUTPUT are terminal
#: successes; INVALID is a terminal failure (an unreadable-forever file is
#: not what the young-container policy is protecting).
_PROVISIONAL_STATES = frozenset({ProbeState.IN_PROGRESS, ProbeState.IMAGELESS})

#: Default bounded-readiness window, matching the wrangler's
#: XDART_CONTAINER_READY_DEADLINE default (image_wrangler_thread.py) — the
#: same "young container" policy this generalizes, not a new default.
DEFAULT_RETRY_DEADLINE = 30.0


@dataclass(frozen=True, slots=True)
class RetryState:
    """Bounded-retry bookkeeping for one provisional candidate."""

    first_seen_at: float
    stamp: tuple[int, int] | None
    attempts: int
    last_result: ProbeResult

    def elapsed(self, *, now: float) -> float:
        return now - self.first_seen_at

    def exhausted(self, *, now: float, deadline: float) -> bool:
        return self.elapsed(now=now) >= deadline


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Immutable point-in-time view of a :class:`DirectoryIndex`'s
    candidates.

    ``generation`` increases monotonically — strictly, on every change
    (added/changed/removed candidate, or a config change via
    :meth:`DirectoryIndex.reconfigure`).  An unchanged poll never bumps it;
    :meth:`DirectoryIndex.poll` returns the prior ``Snapshot`` object
    unchanged in that case, not a new instance with the same generation.
    """

    generation: int
    candidates: tuple[Candidate, ...]
    root: Path
    recursive: bool
    name_filter: str | None

    def by_path(self) -> Mapping[Path, Candidate]:
        return {c.path: c for c in self.candidates}


@dataclass(frozen=True, slots=True)
class IndexDelta:
    """What changed on the most recent :meth:`DirectoryIndex.poll` call."""

    added: tuple[Candidate, ...] = ()
    changed: tuple[Candidate, ...] = ()
    removed: tuple[Path, ...] = ()

    @property
    def unchanged(self) -> bool:
        return not (self.added or self.changed or self.removed)


_EMPTY_DELTA = IndexDelta()


class DirectoryIndex:
    """Incremental, poll-based candidate index over one directory root.

    Construct once per (root, recursive, name_filter) configuration; call
    :meth:`poll` repeatedly (from a future worker thread, a GUI timer, or a
    script loop — this class does not spawn one itself).  Polling is
    authoritative: nothing here depends on filesystem-event notifications,
    though a future caller may use them as an extra *wakeup* hint to poll
    sooner (R1 does not add that; polling alone is sufficient and correct).
    """

    def __init__(self, root: str | Path, *, recursive: bool = False,
                 name_filter: str | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 retry_deadline: float = DEFAULT_RETRY_DEADLINE) -> None:
        self._root = Path(root)
        self._recursive = bool(recursive)
        self._name_filter = name_filter
        self._clock = clock
        self._retry_deadline = float(retry_deadline)
        self._snapshot = Snapshot(0, (), self._root, self._recursive, self._name_filter)
        self._last_delta = _EMPTY_DELTA
        self._retries: dict[Path, RetryState] = {}

    @property
    def snapshot(self) -> Snapshot:
        """The most recent :class:`Snapshot` (empty, generation 0, before
        the first :meth:`poll`)."""
        return self._snapshot

    @property
    def last_delta(self) -> IndexDelta:
        """The delta produced by the most recent :meth:`poll` call (empty
        before the first poll)."""
        return self._last_delta

    @property
    def root(self) -> Path:
        return self._root

    @property
    def recursive(self) -> bool:
        return self._recursive

    @property
    def name_filter(self) -> str | None:
        return self._name_filter

    def reconfigure(self, *, root: str | Path | None = None,
                    recursive: bool | None = None,
                    name_filter: str | None = _UNSET) -> bool:
        """Change ``root``/``recursive``/``name_filter``.

        Any parameter left unset keeps its current value — pass
        ``name_filter=None`` explicitly to CLEAR an existing filter (that is
        why it needs its own sentinel default rather than ``None``, which is
        a valid "no filter" value in its own right).  A real change
        invalidates the prior generation cleanly: the snapshot resets to an
        empty candidate tuple at the NEXT generation, to be repopulated by
        the next :meth:`poll`.  Returns True iff the configuration actually
        changed.
        """
        new_root = self._root if root is None else Path(root)
        new_recursive = self._recursive if recursive is None else bool(recursive)
        new_filter = self._name_filter if name_filter is _UNSET else name_filter

        if (new_root, new_recursive, new_filter) == (self._root, self._recursive, self._name_filter):
            return False

        self._root, self._recursive, self._name_filter = new_root, new_recursive, new_filter
        self._snapshot = Snapshot(
            self._snapshot.generation + 1, (), self._root, self._recursive, self._name_filter)
        self._last_delta = _EMPTY_DELTA
        self._retries.clear()
        return True

    def poll(self) -> Snapshot:
        """Enumerate candidates now (name-only, zero HDF5 opens) and return
        the resulting :class:`Snapshot`.

        When nothing changed since the prior poll (same candidate identities
        and version stamps, in the same order), returns the SAME ``Snapshot``
        object — no new tuple, no re-sort of anything beyond the plain
        filesystem walk :func:`enumerate_candidates` itself always performs.
        """
        fresh = tuple(enumerate_candidates(
            self._root, recursive=self._recursive, name_filter=self._name_filter))
        prior = self._snapshot

        if fresh == prior.candidates:
            self._last_delta = _EMPTY_DELTA
            return prior

        prior_by_path = prior.by_path()
        fresh_by_path = {c.path: c for c in fresh}

        added = tuple(c for c in fresh if c.path not in prior_by_path)
        changed = tuple(
            c for c in fresh
            if c.path in prior_by_path
            and prior_by_path[c.path].version_stamp != c.version_stamp
        )
        removed = tuple(p for p in prior_by_path if p not in fresh_by_path)

        # A changed or removed candidate's prior retry window is stale — the
        # file that was tracked no longer matches what is now on disk (or is
        # gone).  record_probe() below will open a FRESH window from the next
        # explicit probe, rather than silently reusing old timing.
        for c in changed:
            self._retries.pop(c.path, None)
        for p in removed:
            self._retries.pop(p, None)

        self._snapshot = Snapshot(
            prior.generation + 1, fresh, self._root, self._recursive, self._name_filter)
        self._last_delta = IndexDelta(added=added, changed=changed, removed=removed)
        return self._snapshot

    def record_probe(self, path: str | Path, result: ProbeResult, *,
                      retry_deadline: float | None = None) -> ProbeResult:
        """Record an explicit probe result for the candidate at *path* and
        return the EFFECTIVE result a caller should treat it as.

        DirectoryIndex never calls a probe itself — this is how a caller
        (having just called the candidate's adapter ``probe`` callable) tells
        the index what it found, so the index can apply the bounded-
        readiness policy on top:

        * a terminal result (READY / PROCESSED_OUTPUT / INVALID) clears any
          retry tracking for *path* and is returned unchanged;
        * a provisional result (IN_PROGRESS / IMAGELESS) starts or continues
          a bounded retry window keyed to the candidate's CURRENT version
          stamp — a stamp change (caught here or already cleared by
          :meth:`poll`) always starts a fresh window — and is surfaced as
          IN_PROGRESS while the window is open;
        * once the window exceeds ``retry_deadline`` (or the constructor
          default), the raw result is returned and retry tracking clears —
          the candidate is no longer treated as provisional.
        """
        path = Path(path)
        deadline = self._retry_deadline if retry_deadline is None else retry_deadline
        candidate = self._snapshot.by_path().get(path)
        stamp = candidate.version_stamp if candidate is not None else None
        now = self._clock()

        if result.state not in _PROVISIONAL_STATES:
            self._retries.pop(path, None)
            return result

        prior = self._retries.get(path)
        if prior is None or prior.stamp != stamp:
            self._retries[path] = RetryState(
                first_seen_at=now, stamp=stamp, attempts=1, last_result=result)
            return ProbeResult(ProbeState.IN_PROGRESS, reason=result.reason)

        entry = RetryState(
            first_seen_at=prior.first_seen_at, stamp=stamp,
            attempts=prior.attempts + 1, last_result=result)
        if entry.exhausted(now=now, deadline=deadline):
            self._retries.pop(path, None)
            return result
        self._retries[path] = entry
        return ProbeResult(ProbeState.IN_PROGRESS, reason=result.reason)

    def retry_state(self, path: str | Path) -> RetryState | None:
        """The active :class:`RetryState` for *path*, if it is currently
        tracked as provisional (``None`` otherwise)."""
        return self._retries.get(Path(path))

    def poll_forever(self, *, interval: float,
                     should_stop: Callable[[], bool] = lambda: False,
                     sleep: Callable[[float], None] | None = None,
                     ) -> Iterator[Snapshot]:
        """Persistent polling producer suitable for a future worker thread.

        A plain generator — this method does not spawn a thread itself; a
        caller runs it on whatever thread/loop it wants.  Yields a
        :class:`Snapshot` after every :meth:`poll` (including unchanged ones
        — check ``last_delta.unchanged`` if that matters to the caller),
        sleeping ``interval`` between polls via an injectable ``sleep``
        (defaults to :func:`time.sleep`; tests inject a no-op / fake-clock-
        advancing callable so nothing here ever blocks on real time) until
        ``should_stop()`` is true.
        """
        sleep_fn = time.sleep if sleep is None else sleep
        while not should_stop():
            yield self.poll()
            if should_stop():
                return
            sleep_fn(interval)


__all__ = [
    "Candidate",
    "DEFAULT_RETRY_DEADLINE",
    "DirectoryIndex",
    "IndexDelta",
    "RetryState",
    "Snapshot",
]
