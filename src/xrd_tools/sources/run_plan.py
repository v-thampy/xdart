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
safe to hand across a thread boundary.  Its path→candidate lookup is built once
at construction (plan-owned immutable state), so per-candidate validation is
O(1), not a full-map rebuild.

At consumption time the plan re-validates against the live filesystem/index and
distinguishes three transitions of a baseline candidate — matching the accepted
stale-candidate contract of
:class:`~xrd_tools.sources.directory_index.DirectoryIndex` while keeping a
legitimately growing NeXus source retryable:

* **removed** and **owner-flipped** are fail closed and typed: the old identity
  may never be opened.  :meth:`RunCandidatePlan.validate` raises
  :class:`~xrd_tools.sources.directory_index.StaleCandidateError`;
  :meth:`RunCandidatePlan.reconcile` reports them in ``removed`` /
  ``owner_flipped`` and never in ``run_order``.
* a **changed** stamp (same path, same owning adapter, grown bytes — the live
  ``.nxs`` shell case) is NOT retired.  It is *withheld*: reported in
  ``changed``, kept out of ``run_order`` (the old bytes are never opened), and
  recoverable only through an explicit re-poll + reprobe of the CURRENT
  candidate via :meth:`DirectoryIndex.probe_candidate`.  After a fresh READY
  probe the consumer calls :meth:`RunCandidatePlan.adopt` to swap the current
  identity into the same ordering slot exactly once; a still-``IN_PROGRESS``
  probe simply retries on a later poll and never blocks a later ready
  candidate.

:meth:`RunCandidatePlan.reconcile` also enforces the frozen source
configuration: a snapshot from a different ``root`` / ``recursive`` /
``name_filter`` is a superseded plan and fails closed
(:class:`SupersededPlanError`) with no ``run_order`` or ``appended`` output.

This module is Qt-free and opens no file content; it only reads the immutable
value types produced by discovery.  It adds no registry, scanner, or readiness
model — it is a projection of one already-produced :class:`Snapshot`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.directory_index import Snapshot, StaleCandidateError


class SupersededPlanError(ValueError):
    """A :class:`RunCandidatePlan` was reconciled against a snapshot from a
    different source configuration (``root`` / ``recursive`` / ``name_filter``).

    The plan is superseded as a whole — its candidates describe a directory
    scope the current snapshot no longer represents — so reconciliation fails
    closed with no ``run_order`` / ``appended`` output rather than classifying
    candidates from an unrelated scope.  A :class:`ValueError` subclass so
    existing ``except ValueError`` handling still catches it, while a caller
    that wants to rebuild the plan can catch it by type.
    """


@dataclass(frozen=True, slots=True)
class RunReconcile:
    """The result of reconciling a :class:`RunCandidatePlan` against a fresh
    :class:`~xrd_tools.sources.directory_index.Snapshot`.

    ``run_order`` — baseline candidates whose full identity still matches the
    current filesystem, in the plan's FROZEN order (the exact-run handoff;
    never a re-globbed snapshot order).  These are safe to open now.

    ``changed`` — baseline candidates present at the same path and owning
    adapter but with a different ``version_stamp`` (a growing/rewritten source).
    The CURRENT candidate is returned so the consumer can reprobe it; the old
    bytes are withheld from ``run_order`` and never opened.  Recover with an
    explicit reprobe + :meth:`RunCandidatePlan.adopt` (see module docstring).

    ``removed`` — baseline paths absent from the snapshot (fail closed, typed).

    ``owner_flipped`` — baseline paths whose owning ``adapter_id`` changed; the
    CURRENT candidate is returned for reporting (fail closed — never opened
    through the plan's old identity).

    ``appended`` — candidates present in the snapshot but absent from the
    baseline, in natural snapshot order (later live-discovered scans), so
    Directory Live still finds new candidates without the plan reordering or
    dropping the frozen baseline.
    """

    run_order: tuple[Candidate, ...]
    changed: tuple[Candidate, ...]
    removed: tuple[Path, ...]
    owner_flipped: tuple[Candidate, ...]
    appended: tuple[Candidate, ...]

    @property
    def baseline_current(self) -> bool:
        """True iff every baseline candidate's IDENTITY is exactly current — no
        changed stamp, removal, or owner flip.

        This is a name-only identity fact; it carries NO probe/readiness
        evidence (a candidate in ``run_order`` is byte-stamp-current, not proven
        content-ready).
        """
        return not (self.changed or self.removed or self.owner_flipped)


@dataclass(frozen=True)
class RunCandidatePlan:
    """Immutable, ordered baseline of source candidates frozen at Run start.

    Build one with :meth:`from_snapshot` from the ``Snapshot`` the Source card
    displayed as Ready.  The plan records the index ``generation`` and the
    ``(root, recursive, name_filter)`` configuration that produced the baseline
    alongside the ordered candidates, so :meth:`reconcile` can reject a
    superseded configuration, not only per-candidate drift.

    The path→candidate lookup is built once in ``__post_init__`` and held as
    plan-owned immutable state, so :meth:`validate` and :meth:`reconcile`
    membership are O(1) per candidate rather than rebuilding a full dict.
    """

    generation: int
    candidates: tuple[Candidate, ...]
    root: Path
    recursive: bool
    name_filter: str | None

    def __post_init__(self) -> None:
        index = {c.path: c for c in self.candidates}
        # object.__setattr__: populate the cached lookup on a frozen instance.
        # Not dataclass fields, so they do not affect eq/hash/repr.
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_index_view", MappingProxyType(index))

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
        """Baseline candidates keyed by path — a read-only view of the cached,
        plan-owned lookup (the SAME object across calls; never rebuilt)."""
        return self._index_view  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self.candidates)

    def __bool__(self) -> bool:
        return bool(self.candidates)

    def matches_config(self, snapshot: Snapshot) -> bool:
        """Whether *snapshot* shares this plan's frozen source configuration
        (``root`` / ``recursive`` / ``name_filter``)."""
        return (snapshot.root == self.root
                and snapshot.recursive == self.recursive
                and snapshot.name_filter == self.name_filter)

    def validate(self, current: Candidate | None, *,
                 path: str | Path | None = None) -> Candidate:
        """Fail-closed check of ONE baseline candidate against its current
        filesystem identity (O(1) lookup).

        *current* is the candidate the caller just observed for this path (e.g.
        ``fresh_snapshot.by_path().get(path)`` after a re-poll), or ``None`` if
        the path is no longer discovered.  *path* defaults to ``current.path``;
        pass it explicitly when *current* is ``None`` (a removal).

        Returns *current* unchanged when it matches the frozen baseline entry.
        Raises :class:`~xrd_tools.sources.directory_index.StaleCandidateError`
        when the candidate was removed, its ``version_stamp`` changed (a growth
        or rewrite — recover via :meth:`reconcile` + reprobe + :meth:`adopt`,
        never by opening the old bytes here), or its ``adapter_id`` flipped
        (owner change).  Raises :class:`KeyError` when *path* is not part of
        this plan's baseline (a live-appended candidate — resolve it through
        :meth:`reconcile`, not this per-baseline check).
        """
        want_path = (
            Path(path) if path is not None
            else (current.path if current is not None else None))
        if want_path is None:
            raise ValueError(
                "validate needs a current Candidate or an explicit path")
        planned = self._index.get(want_path)  # type: ignore[attr-defined]
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
                "reprobe and adopt (do not open the old bytes)")
        if current.adapter_id != planned.adapter_id:
            raise StaleCandidateError(
                f"{want_path} owning adapter changed from {planned.adapter_id!r} "
                f"to {current.adapter_id!r} since Run started; re-poll and "
                "reselect")
        return current

    def reconcile(self, snapshot: Snapshot) -> RunReconcile:
        """Reconcile the frozen baseline against a fresh *snapshot*.

        Enforces the frozen source configuration first: a snapshot from a
        different ``root`` / ``recursive`` / ``name_filter`` raises
        :class:`SupersededPlanError` with no output (RP2).

        Otherwise classifies each baseline candidate against the snapshot in one
        pass and returns a :class:`RunReconcile`: ``run_order`` (identity
        unchanged, in FROZEN order), ``changed`` (same path+adapter, grown
        stamp — the current candidate, withheld pending reprobe/adopt),
        ``removed`` and ``owner_flipped`` (fail closed), and ``appended``
        (snapshot candidates absent from the baseline, natural order).
        """
        if not self.matches_config(snapshot):
            raise SupersededPlanError(
                "snapshot configuration "
                f"(root={snapshot.root}, recursive={snapshot.recursive}, "
                f"name_filter={snapshot.name_filter!r}) does not match the "
                f"plan's frozen configuration (root={self.root}, "
                f"recursive={self.recursive}, name_filter={self.name_filter!r}); "
                "the plan is superseded — rebuild it from the current snapshot")
        current_by_path = snapshot.by_path()
        run_order: list[Candidate] = []
        changed: list[Candidate] = []
        removed: list[Path] = []
        owner_flipped: list[Candidate] = []
        for planned in self.candidates:
            cur = current_by_path.get(planned.path)
            if cur is None:
                removed.append(planned.path)
            elif cur.adapter_id != planned.adapter_id:
                owner_flipped.append(cur)
            elif cur.version_stamp != planned.version_stamp:
                changed.append(cur)
            else:
                run_order.append(cur)
        baseline = self._index  # type: ignore[attr-defined]
        appended = tuple(
            c for c in snapshot.candidates if c.path not in baseline)
        return RunReconcile(
            tuple(run_order), tuple(changed), tuple(removed),
            tuple(owner_flipped), appended)

    def adopt(self, current: Candidate) -> "RunCandidatePlan":
        """Return a NEW plan with *current* replacing the baseline candidate at
        the same path, retaining its ordering slot (the ``changed`` -> ready
        transition of RP1).

        *current* must be a genuinely CHANGED candidate the consumer has just
        re-observed and reprobed READY: same ``path`` and ``adapter_id`` as the
        baseline entry, different ``version_stamp``.  Once adopted, the path is
        identity-current, so a later :meth:`reconcile` places it in
        ``run_order`` (opened exactly once) rather than ``changed`` again.

        Raises :class:`KeyError` if *current.path* is not in the baseline,
        :class:`~xrd_tools.sources.directory_index.StaleCandidateError` if its
        ``adapter_id`` flipped (an owner change is not an adoptable growth), and
        :class:`ValueError` if its ``version_stamp`` is unchanged (nothing to
        adopt — do not resubmit an already-current candidate).
        """
        planned = self._index.get(current.path)  # type: ignore[attr-defined]
        if planned is None:
            raise KeyError(
                f"{current.path} is not in the run candidate plan baseline")
        if current.adapter_id != planned.adapter_id:
            raise StaleCandidateError(
                f"{current.path} owning adapter changed from "
                f"{planned.adapter_id!r} to {current.adapter_id!r}; an owner "
                "flip is not an adoptable growth — re-poll and reselect")
        if current.version_stamp == planned.version_stamp:
            raise ValueError(
                f"{current.path} identity is unchanged; nothing to adopt")
        new_candidates = tuple(
            current if c.path == current.path else c for c in self.candidates)
        return RunCandidatePlan(
            generation=self.generation, candidates=new_candidates,
            root=self.root, recursive=self.recursive,
            name_filter=self.name_filter)


__all__ = ["RunCandidatePlan", "RunReconcile", "SupersededPlanError"]
