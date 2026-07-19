# -*- coding: utf-8 -*-
"""``RunCandidatePlan`` — the immutable Source-card → Run handoff seam (H19 §4).

The Source card displays an ordered set of source candidates (from a
:class:`~xrd_tools.sources.directory_index.Snapshot`) and, when the user starts
a run, the *exact* displayed set is what processing must consume.  This module
is the smallest value-only seam that carries that decision across the run
boundary without letting the worker re-glob the directory or silently switch
adapter ownership after the user saw readiness.

A :class:`RunCandidatePlan` is a frozen, ordered baseline of
:class:`~xrd_tools.sources.discover.Candidate` values — each already carrying
its full R1 identity (``path`` + ``version_stamp`` + ``adapter_id``).  It holds
no open handle, ``FrameSource``, cursor, or file object; it is pure value state
safe to hand across a thread boundary.

At consumption time the plan re-validates against the live filesystem/index and
preserves the accepted stale-candidate contract of
:class:`~xrd_tools.sources.directory_index.DirectoryIndex`:

* :meth:`RunCandidatePlan.validate` fails closed on a single baseline candidate
  whose current identity no longer matches — removal, byte (``version_stamp``)
  change, or an adapter-owner flip all raise
  :class:`~xrd_tools.sources.directory_index.StaleCandidateError` so the worker
  never processes stale bytes;
* :meth:`RunCandidatePlan.reconcile` computes the ordered run list against a
  fresh snapshot: the still-valid baseline in its FROZEN order (no re-glob
  reordering), the fail-closed ``stale`` set held back from processing, and the
  ``appended`` candidates that appeared after the plan was frozen (later
  live-discovered scans, in natural snapshot order).

This module is Qt-free and opens no file content; it only reads the immutable
value types produced by discovery.  It adds no registry, scanner, or readiness
model — it is a projection of one already-produced :class:`Snapshot`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.directory_index import Snapshot, StaleCandidateError


@dataclass(frozen=True, slots=True)
class RunReconcile:
    """The result of reconciling a :class:`RunCandidatePlan` against a fresh
    :class:`~xrd_tools.sources.directory_index.Snapshot`.

    ``run_order`` are the baseline candidates whose full identity still matches
    the current filesystem, in the plan's FROZEN order — the exact-run handoff.
    ``stale`` are baseline paths that were removed, byte-changed, or
    owner-flipped since the plan was frozen; they are held back (fail closed),
    never processed as stale bytes.  ``appended`` are candidates present in the
    fresh snapshot but absent from the baseline — later live-discovered scans,
    in natural snapshot order — so Directory Live still finds new candidates
    without the plan reordering or dropping the frozen baseline.
    """

    run_order: tuple[Candidate, ...]
    stale: tuple[Path, ...]
    appended: tuple[Candidate, ...]

    @property
    def all_ready(self) -> bool:
        """True iff every baseline candidate is still identity-valid (no
        stale paths)."""
        return not self.stale


@dataclass(frozen=True, slots=True)
class RunCandidatePlan:
    """Immutable, ordered baseline of source candidates frozen at Run start.

    Build one with :meth:`from_snapshot` from the ``Snapshot`` the Source card
    displayed as Ready.  The plan records the index ``generation`` and the
    ``(root, recursive, name_filter)`` configuration that produced the baseline
    alongside the ordered candidates, so a consumer can detect a whole-config
    change, not only per-candidate drift.
    """

    generation: int
    candidates: tuple[Candidate, ...]
    root: Path
    recursive: bool
    name_filter: str | None

    @classmethod
    def from_snapshot(cls, snapshot: Snapshot) -> "RunCandidatePlan":
        """Freeze the ordered candidate baseline from a directory
        :class:`~xrd_tools.sources.directory_index.Snapshot`.

        Copies only the immutable candidate values and the config identity; it
        holds no reference to the live :class:`DirectoryIndex`.
        """
        return cls(
            generation=snapshot.generation,
            candidates=tuple(snapshot.candidates),
            root=snapshot.root,
            recursive=snapshot.recursive,
            name_filter=snapshot.name_filter,
        )

    @property
    def paths(self) -> tuple[Path, ...]:
        """The baseline candidate paths, in frozen order."""
        return tuple(c.path for c in self.candidates)

    def by_path(self) -> Mapping[Path, Candidate]:
        """Baseline candidates keyed by path."""
        return {c.path: c for c in self.candidates}

    def __len__(self) -> int:
        return len(self.candidates)

    def __bool__(self) -> bool:
        return bool(self.candidates)

    def validate(self, current: Candidate | None, *,
                 path: str | Path | None = None) -> Candidate:
        """Fail-closed check of ONE baseline candidate against its current
        filesystem identity.

        *current* is the candidate the caller just observed for this path (e.g.
        ``fresh_snapshot.by_path().get(path)`` after a re-poll), or ``None`` if
        the path is no longer discovered.  *path* defaults to ``current.path``;
        pass it explicitly when *current* is ``None`` (a removal).

        Returns *current* unchanged when it matches the frozen baseline entry.
        Raises :class:`~xrd_tools.sources.directory_index.StaleCandidateError`
        when the candidate was removed, its ``version_stamp`` changed (bytes
        changed), or its ``adapter_id`` flipped (owner change) since the plan
        was frozen — the worker must re-poll/reselect, never process stale
        bytes.  Raises :class:`KeyError` when *path* is not part of this plan's
        baseline (a live-appended candidate — resolve it through
        :meth:`reconcile`, not this per-baseline check).
        """
        want_path = (
            Path(path) if path is not None
            else (current.path if current is not None else None))
        if want_path is None:
            raise ValueError(
                "validate needs a current Candidate or an explicit path")
        planned = self.by_path().get(want_path)
        if planned is None:
            raise KeyError(
                f"{want_path} is not in the run candidate plan; it is a "
                "live-appended candidate — resolve it via reconcile()")
        if current is None:
            raise StaleCandidateError(
                f"{want_path} was removed since Run started; re-poll and "
                "reselect (do not process stale bytes)")
        if current.version_stamp != planned.version_stamp:
            raise StaleCandidateError(
                f"{want_path} bytes changed since Run started "
                f"({planned.version_stamp} -> {current.version_stamp}); "
                "re-poll and reselect")
        if current.adapter_id != planned.adapter_id:
            raise StaleCandidateError(
                f"{want_path} owning adapter changed from {planned.adapter_id!r} "
                f"to {current.adapter_id!r} since Run started; re-poll and "
                "reselect")
        return current

    def reconcile(self, snapshot: Snapshot) -> RunReconcile:
        """Reconcile the frozen baseline against a fresh *snapshot*.

        Returns a :class:`RunReconcile` whose ``run_order`` is the baseline
        candidates whose full identity still matches the current filesystem, in
        the plan's FROZEN order (never the snapshot's — the run does not regain
        a re-globbed ordering); ``stale`` is the fail-closed set of baseline
        paths that were removed / byte-changed / owner-flipped; and ``appended``
        is the snapshot candidates absent from the baseline, in natural snapshot
        order (later live-discovered scans).
        """
        current_by_path = snapshot.by_path()
        run_order: list[Candidate] = []
        stale: list[Path] = []
        for planned in self.candidates:
            cur = current_by_path.get(planned.path)
            if (cur is not None
                    and cur.version_stamp == planned.version_stamp
                    and cur.adapter_id == planned.adapter_id):
                run_order.append(cur)
            else:
                stale.append(planned.path)
        baseline_paths = set(self.by_path())
        appended = tuple(
            c for c in snapshot.candidates if c.path not in baseline_paths)
        return RunReconcile(tuple(run_order), tuple(stale), appended)


__all__ = ["RunCandidatePlan", "RunReconcile"]
