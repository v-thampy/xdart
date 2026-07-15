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
candidate scan — zero HDF5 opens, on both the first poll and every unchanged
one after it.  It splits the CHEAP unordered collect+compare
(:func:`~xrd_tools.sources.discover.collect_candidates`) from the expensive
natural sort: an unchanged poll returns the *same* :class:`Snapshot` object
without a single ``os_sorted`` call, so a caller never re-sorts or
reconstructs an unchanged candidate list; the sort runs only when the
candidate map actually changed.

:meth:`DirectoryIndex.record_probe` layers BOUNDED RETRY on top of an
explicit, caller-invoked probe (:mod:`xrd_tools.sources.adapters`,
:mod:`xrd_tools.sources.probe`) — the index itself never probes.  This is the
"newly readable HDF5 shell... remains provisional... not permanently retired
as imageless" policy: a fresh IN_PROGRESS/IMAGELESS verdict is held as
IN_PROGRESS for up to ``retry_deadline`` seconds (an injectable clock, no
wall-clock sleeps in tests) before the raw verdict is trusted.  Once the
window resolves to a terminal verdict, that verdict is STICKY for the
candidate's current version stamp — repeated probes of the same stamp return
it without opening a new window — until a size/mtime change, removal, or
reconfiguration clears the resolution and permits a fresh bounded window.
This stamp-qualified resolution is a tiny value map (path -> stamp + verdict);
it is deliberately not a metadata cache.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrd_tools.sources.discover import Candidate, collect_candidates, sort_candidates
from xrd_tools.sources.probe import ProbeResult, ProbeState

_UNSET: Any = object()

_Stamp = tuple[int, int]

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


class StaleCandidateError(ValueError):
    """A probe completion no longer matches the current candidate identity.

    Raised by :meth:`DirectoryIndex.record_probe` /
    :meth:`DirectoryIndex.probe_candidate` when a result's path, version stamp,
    or owning ``adapter_id`` no longer matches the index's current snapshot —
    the candidate was removed, its bytes changed, or a registration flip
    changed its owning adapter since the probe began (R1-R10).  A
    :class:`ValueError` subclass so existing ``except ValueError`` handling
    still catches it, while a caller that wants to re-poll and re-probe the new
    owner can catch it by type.  The stale result is NEVER returned as an
    effective verdict: the failure is observable at the consumption boundary so
    a caller cannot act on an old owner's stale ``READY``."""


@dataclass(frozen=True, slots=True)
class RetryState:
    """Bounded-retry bookkeeping for one provisional candidate.

    ``imageless_result`` remembers the most recent IMAGELESS observation seen
    ANYWHERE in the window, not just the last probe: IMAGELESS means the file
    was READABLE (just carried no detector dataset), a strictly more-resolved
    state than IN_PROGRESS (not-yet-readable / unfinalized).  A window that was
    readable-but-imageless even once is a genuine imageless shell and resolves
    to IMAGELESS at exhaustion; only a window that was NEVER once readable (all
    IN_PROGRESS) escalates to INVALID.  Without this, a single transient
    IN_PROGRESS (a momentary lock/network stall — exactly what the window
    exists to absorb) landing on the deadline would wrongly condemn a good
    zero-frame file to permanent INVALID."""

    first_seen_at: float
    stamp: tuple[int, int] | None
    attempts: int
    last_result: ProbeResult
    imageless_result: ProbeResult | None = None

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
        # ACTIVE provisional retry windows only — retry_state() reflects exactly
        # this dict, so a resolved candidate never lingers here.
        self._retries: dict[Path, RetryState] = {}
        # Stamp-qualified TERMINAL resolutions (path -> (stamp, verdict)), kept
        # separate from _retries so a settled IMAGELESS/INVALID/READY verdict is
        # sticky for its stamp without reopening a window.  A tiny value map,
        # not a metadata cache; cleared per-path on change/removal/reconfigure.
        self._terminal: dict[Path, tuple[_Stamp, ProbeResult]] = {}

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
        self._terminal.clear()
        return True

    def poll(self) -> Snapshot:
        """Scan candidates now (name-only, zero HDF5 opens) and return the
        resulting :class:`Snapshot`.

        Collects candidates UNORDERED and cheaply (directory listing + one
        ``stat()`` per match, no natural sort), then compares the fresh
        candidate map against the prior snapshot's by full candidate identity
        — path, owning ``adapter_id``, and ``(size, mtime_ns)`` stamp.  When
        they are equal, returns the SAME prior ``Snapshot`` object with an
        empty delta and performs NO natural sort (R1-R5).  Only when the map
        actually changed does it sort once (from the already-collected list)
        and build a new generation.

        ``changed`` includes a candidate whose OWNING ADAPTER flipped for an
        unchanged path+stamp (a registration change), not just a byte change
        (R1-R3); provisional and terminal state for every changed/removed path
        is cleared, since the tracked file no longer matches what is on disk.
        """
        fresh_list = collect_candidates(
            self._root, recursive=self._recursive, name_filter=self._name_filter)
        fresh_by_path = {c.path: c for c in fresh_list}
        prior = self._snapshot
        prior_by_path = prior.by_path()

        # Order-independent unchanged check on the cheap map — no sort.  Candidate
        # equality is full (path/adapter_id/size/mtime_ns), so an owner-only flip
        # makes the maps unequal and is never mistaken for "unchanged".
        if fresh_by_path == prior_by_path:
            self._last_delta = _EMPTY_DELTA
            return prior

        fresh = tuple(sort_candidates(fresh_list))  # sort ONCE, only on real change

        # R1-R9: derive the delta queues from the SORTED snapshot, not the raw
        # filesystem-order list, so a consumer of last_delta.added/changed never
        # regains unsorted directory order.  ``removed`` follows prior snapshot
        # order (prior.candidates is already sorted; prior_by_path preserves it).
        added = tuple(c for c in fresh if c.path not in prior_by_path)
        changed = tuple(
            c for c in fresh
            if c.path in prior_by_path and prior_by_path[c.path] != c
        )
        removed = tuple(p for p in prior_by_path if p not in fresh_by_path)

        # A changed (bytes OR owner) or removed candidate's prior state no longer
        # describes what is on disk — clear BOTH the active retry window and any
        # sticky terminal resolution for that path so the next explicit probe
        # opens a fresh bounded window rather than reusing stale timing/verdict.
        for c in changed:
            self._retries.pop(c.path, None)
            self._terminal.pop(c.path, None)
        for p in removed:
            self._retries.pop(p, None)
            self._terminal.pop(p, None)

        self._snapshot = Snapshot(
            prior.generation + 1, fresh, self._root, self._recursive, self._name_filter)
        self._last_delta = IndexDelta(added=added, changed=changed, removed=removed)
        return self._snapshot

    def record_probe(self, target: "str | Path | Candidate", result: ProbeResult, *,
                      expected_stamp: _Stamp | None = None,
                      expected_adapter_id: str | None = None,
                      retry_deadline: float | None = None) -> ProbeResult:
        """Record an explicit probe result for a candidate and return the
        EFFECTIVE result a caller should treat as current.

        *target* is the candidate the *result* was computed for.  Pass the
        immutable :class:`~xrd_tools.sources.discover.Candidate` (preferred):
        its full identity — path, ``adapter_id``, and stamp — is checked
        against the current snapshot.  A bare ``str``/``Path`` is accepted ONLY
        with BOTH ``expected_stamp`` and ``expected_adapter_id`` supplied; an
        unguarded ``record_probe(path, result)`` raises :class:`ValueError`,
        because it cannot verify the completion belongs to the current owner
        (R1-R10).

        DirectoryIndex never calls a probe itself — this is how a caller
        (having just called the candidate's adapter ``probe`` callable) tells
        the index what it found, so the index can apply the bounded-readiness
        policy on top:

        * a completion whose full identity does NOT match a current candidate —
          the path absent from the snapshot, or a ``stamp``/``adapter_id`` that
          no longer matches the current candidate (a removal, a byte change, or
          a registration flip that changed the owner while the probe ran) — is
          a STALE completion.  It RAISES :class:`StaleCandidateError` and
          creates NO retry or terminal state (R1-R6, R1-R7, R1-R10).  The stale
          result is never returned as an effective verdict: the failure is
          observable at the consumption boundary, so a caller cannot act on an
          old owner's stale ``READY`` — it must re-poll and re-probe the new
          owner.  The ``adapter_id`` check catches a late result from an OLD
          owner after a flip left the bytes (and thus the stamp) unchanged — a
          case the stamp check alone cannot see;
        * a stamp already resolved to a TERMINAL verdict returns that sticky
          verdict without reopening a window (R1-R1);
        * a terminal result (READY / PROCESSED_OUTPUT / INVALID) resolves the
          stamp terminally, clears any active retry window, and is returned
          unchanged;
        * a provisional result (IN_PROGRESS / IMAGELESS) starts or continues a
          bounded retry window keyed to the candidate's current stamp and is
          surfaced as IN_PROGRESS while the window is open;
        * once the window exceeds ``retry_deadline`` (or the constructor
          default) it resolves TERMINALLY and sticks for the stamp: a window
          that was ever readable-but-imageless resolves to IMAGELESS (a
          stable, zero-detector-frame shell — the wrangler's "only a stable
          old file is retired as genuinely imageless" policy); a window that
          was NEVER once readable (all IN_PROGRESS — still unreadable /
          unfinalized the whole time) escalates to INVALID rather than
          retrying forever.  The escalation keys on whether the file EVER
          resolved during the window, not on the single last sample, so a lone
          transient IN_PROGRESS on the deadline cannot condemn a genuine
          imageless shell.
        """
        if isinstance(target, Candidate):
            path = target.path
            expected_stamp = target.version_stamp
            expected_adapter_id = target.adapter_id
        else:
            path = Path(target)
            # R1-R10: an unguarded bare-path recording cannot verify the
            # completion belongs to the current owner — forbid it outright.
            if expected_stamp is None or expected_adapter_id is None:
                raise ValueError(
                    "record_probe requires a Candidate, or BOTH expected_stamp and "
                    "expected_adapter_id for a bare path; an unguarded "
                    "record_probe(path, result) cannot verify the completion belongs "
                    "to the current candidate owner")
        deadline = self._retry_deadline if retry_deadline is None else retry_deadline
        candidate = self._snapshot.by_path().get(path)

        # R1-R6/R1-R7/R1-R10: a stale/unknown completion FAILS observably (no
        # state created, stale result never returned) — never silently ignored,
        # so probe_candidate cannot forward an old owner's verdict as current.
        if candidate is None:
            raise StaleCandidateError(
                f"{path} is not a current candidate (removed or never discovered)")
        stamp = candidate.version_stamp
        if expected_stamp != stamp:
            raise StaleCandidateError(
                f"{path} version stamp changed since the probe began "
                "(bytes changed); re-poll and re-probe")
        if expected_adapter_id != candidate.adapter_id:
            raise StaleCandidateError(
                f"{path} owning adapter changed from {expected_adapter_id!r} to "
                f"{candidate.adapter_id!r} since the probe began; re-poll and re-probe")

        # R1-R1: a stamp already resolved terminally stays terminal — no new
        # window.  poll() clears this on any change/removal for the path.
        settled = self._terminal.get(path)
        if settled is not None and settled[0] == stamp:
            return settled[1]

        now = self._clock()

        if result.state not in _PROVISIONAL_STATES:
            self._retries.pop(path, None)
            self._terminal[path] = (stamp, result)
            return result

        this_imageless = result if result.state is ProbeState.IMAGELESS else None
        prior = self._retries.get(path)
        if prior is None or prior.stamp != stamp:
            self._retries[path] = RetryState(
                first_seen_at=now, stamp=stamp, attempts=1, last_result=result,
                imageless_result=this_imageless)
            return ProbeResult(ProbeState.IN_PROGRESS, reason=result.reason)

        # Carry the most-resolved observation forward: an IMAGELESS seen
        # anywhere in the window survives a later transient IN_PROGRESS.
        imageless_seen = this_imageless or prior.imageless_result
        entry = RetryState(
            first_seen_at=prior.first_seen_at, stamp=stamp,
            attempts=prior.attempts + 1, last_result=result,
            imageless_result=imageless_seen)
        if entry.exhausted(now=now, deadline=deadline):
            self._retries.pop(path, None)
            if imageless_seen is not None:
                # readable-but-imageless at least once -> genuine imageless shell
                resolved = imageless_seen
            else:
                # never once readable the whole window -> stuck/corrupt
                resolved = ProbeResult(
                    ProbeState.INVALID,
                    reason=f"{result.reason} (exceeded {deadline:g}s retry window)",
                )
            self._terminal[path] = (stamp, resolved)
            return resolved
        self._retries[path] = entry
        return ProbeResult(ProbeState.IN_PROGRESS, reason=result.reason)

    def retry_state(self, path: str | Path) -> RetryState | None:
        """The active :class:`RetryState` for *path* — an OPEN provisional
        retry window only.  A terminally resolved candidate (settled
        IMAGELESS/INVALID/READY) is not a provisional window and returns
        ``None`` here even though its verdict is still sticky."""
        return self._retries.get(Path(path))

    def probe_candidate(self, candidate: Candidate, *,
                        retry_deadline: float | None = None) -> ProbeResult:
        """Explicitly probe ONE candidate through its owning adapter and
        record the result via :meth:`record_probe`.

        This is the ONLY method on this class that may open a file's content,
        and it is never called by :meth:`poll` or
        :func:`~xrd_tools.sources.discover.collect_candidates` — a caller
        decides which candidates are worth the I/O, one at a time (e.g. only
        the newest few, or only ones the GUI is about to display).

        Rejects a STALE *candidate* object — one whose path/adapter/stamp no
        longer matches the current snapshot (removed, byte-changed, or
        owner-flipped since it was enumerated) — with a
        :class:`StaleCandidateError`, BEFORE opening the file, so a late
        consumer of an old snapshot never probes a candidate the index no
        longer owns (R1-R6).

        The SAME failure is re-raised at the record step if ownership flips
        while the adapter's ``probe`` is running (a TOCTOU race): the stale
        result is NEVER returned as an effective verdict, so a processing
        caller can never act on an old owner's stale ``READY`` — it must
        catch :class:`StaleCandidateError`, re-poll, and probe the new owner
        (R1-R10).
        """
        current = self._snapshot.by_path().get(candidate.path)
        if (current is None
                or current.version_stamp != candidate.version_stamp
                or current.adapter_id != candidate.adapter_id):
            raise StaleCandidateError(
                f"stale candidate {candidate.path} no longer matches the current "
                "snapshot; re-poll before probing")
        from xrd_tools.sources.adapters import get_adapter
        adapter = get_adapter(candidate.adapter_id)
        if adapter is None:
            raise LookupError(f"no adapter registered for id {candidate.adapter_id!r}")
        raw = adapter.probe(candidate.path)
        # Pass the full Candidate so record_probe re-validates path+stamp+
        # adapter_id — a concurrent owner flip between this freshness check and
        # the record raises StaleCandidateError rather than returning A's stale
        # result (R1-R7 kept state clean; R1-R10 makes the failure observable).
        return self.record_probe(candidate, raw, retry_deadline=retry_deadline)

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
    "StaleCandidateError",
]
