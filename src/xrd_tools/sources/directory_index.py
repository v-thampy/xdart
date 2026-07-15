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
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrd_tools.sources.discover import Candidate, enumerate_candidates

_UNSET: Any = object()


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
                 name_filter: str | None = None) -> None:
        self._root = Path(root)
        self._recursive = bool(recursive)
        self._name_filter = name_filter
        self._snapshot = Snapshot(0, (), self._root, self._recursive, self._name_filter)
        self._last_delta = _EMPTY_DELTA

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

        self._snapshot = Snapshot(
            prior.generation + 1, fresh, self._root, self._recursive, self._name_filter)
        self._last_delta = IndexDelta(added=added, changed=changed, removed=removed)
        return self._snapshot


__all__ = ["Candidate", "DirectoryIndex", "IndexDelta", "Snapshot"]
