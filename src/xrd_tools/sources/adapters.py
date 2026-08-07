# -*- coding: utf-8 -*-
"""Source-format adapter registry (R1).

One small immutable descriptor per source *format* (NeXus family, TIFF/image
series, SPEC, live, memory, ...) that declares everything the format needs to
participate in discovery, probing, and opening — filename rules, a canonical
scan-name function, a content probe/classifier, an ``open_source``-compatible
factory, and lazy hooks reserved for later work (metadata provider for R2,
finalization/recovery policy).  Adding a new format is **one
:func:`register_adapter` call**, never an edit to :func:`open_source`'s
dispatch or to :mod:`xrd_tools.sources.discover`'s per-kind branches — the
same seam a built-in format and an out-of-tree/synthetic one both go through
(pinned by ``tests/core/test_source_format_adapters.py``).

This module intentionally holds no open HDF5 objects, frame arrays, full
metadata tables, or GUI state — only the small callables/values above.  It
must stay import-light: no ``h5py``/``fabio``/``pyFAI``/Qt at module level, so
``import xrd_tools.sources`` (which lazily re-exports this module) never pulls
them in.  Concrete adapters do their heavy imports lazily, inside the
callables they register — the same pattern already used throughout
:mod:`xrd_tools.sources.registry`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrd_tools.core.scan import FrameSource, SourceKind, SourceSpec, coerce_source_kind

CandidatePredicate = Callable[[Path], bool]
"""Name-only test: does *path* look like this format?  Cheap file-stamp
inspection only (extension/stem rules, sidecar exclusion) — MUST NOT open the
file."""

ScanNameFn = Callable[[Path], str]
"""Canonical scan-name derivation for a candidate path of this format."""

ProbeFn = Callable[[Path], Any]
"""Explicit content probe/classifier — MAY open the file.  Only called when a
caller asks to probe a specific candidate, never during enumeration.  Returns
a :class:`xrd_tools.sources.probe.ProbeResult`; typed loosely here to avoid a
probe.py <-> adapters.py import-order dependency."""

SourceFactory = Callable[[SourceSpec], FrameSource]
"""``open_source``-compatible opener: SourceSpec -> FrameSource."""

MetadataProviderFactory = Callable[[Path], Any] | None
"""Reserved hook (R2): a lazy per-candidate metadata-provider factory.  Not
called anywhere in R1 — the field exists so R2 can add a provider without a
new adapter shape."""

FinalizationPolicy = Callable[[Path], Any] | None
"""Reserved hook: format-specific finalized/in-progress/recovery policy (see
:mod:`xrd_tools.sources.readiness`).  ``None`` means "no format-specific
policy" — the caller falls back to the generic stamp-only heuristics."""


@dataclass(frozen=True, slots=True)
class SourceFormatAdapter:
    """Immutable descriptor for one source format.

    ``kinds`` may list more than one :class:`SourceKind` when the exact kind
    cannot be told apart by filename alone (e.g. the NeXus family covers
    ``NEXUS_STACK``/``EIGER_MASTER``/``PROCESSED_NEXUS`` — telling them apart
    needs the content probe, not the candidate predicate).
    """

    id: str
    kinds: tuple[SourceKind, ...]
    is_candidate: CandidatePredicate
    scan_name: ScanNameFn
    probe: ProbeFn
    open: SourceFactory
    metadata_provider: MetadataProviderFactory = None
    finalization_policy: FinalizationPolicy = None
    is_output_format: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("SourceFormatAdapter.id must be a non-empty string")
        kinds = tuple(coerce_source_kind(k) for k in self.kinds)
        if not kinds:
            raise ValueError(f"adapter {self.id!r} must declare at least one SourceKind")
        object.__setattr__(self, "kinds", kinds)


@dataclass(frozen=True, slots=True)
class _Entry:
    """Registry bookkeeping for one adapter: the adapter, whether it is a
    built-in (lower-precedence tier), and a monotonic registration sequence."""

    adapter: SourceFormatAdapter
    builtin: bool
    seq: int


#: id -> entry.  Re-registering an id overwrites its entry, so there is NO
#: separate kind index that could go stale (R1-R2): every kind/candidate lookup
#: resolves live against ``adapter.kinds`` / ``is_candidate`` below.
_ADAPTERS: dict[str, _Entry] = {}
_next_seq = 0


def _rank(entry: _Entry) -> tuple[int, int]:
    """Precedence key — higher wins.  ONE rule shared by candidate discovery,
    :func:`adapter_for_kind`, probing, and opening (R1-R2):

    * an externally registered adapter (``builtin=False``) ALWAYS outranks a
      built-in, regardless of import order — so a plugin registered before the
      lazy built-in bootstrap is never displaced when the built-ins later
      register, and one registered after still wins;
    * within a tier, the most recently registered adapter wins
      (last-registered-wins), mirroring
      :func:`xrd_tools.sources.registry.register_source`'s override semantics.
    """
    return (0 if entry.builtin else 1, entry.seq)


def _best(entries: Iterable[_Entry]) -> _Entry | None:
    entries = list(entries)
    return max(entries, key=_rank) if entries else None


def register_adapter(adapter: SourceFormatAdapter, *, builtin: bool = False) -> None:
    """Register *adapter*, replacing any prior adapter with the same id.

    ``builtin`` marks the lower-precedence tier used by the in-tree formats
    (see :func:`_rank`); out-of-tree callers leave it False so their adapters
    outrank the built-ins deterministically.  Re-registering an id simply
    overwrites its entry; because kind/candidate ownership is resolved live
    (never cached in a side index), a replacement that declares fewer kinds
    leaves NO stale kind mapping behind.
    """
    global _next_seq
    _ADAPTERS[adapter.id] = _Entry(adapter=adapter, builtin=builtin, seq=_next_seq)
    _next_seq += 1


def get_adapter(adapter_id: str) -> SourceFormatAdapter | None:
    entry = _ADAPTERS.get(adapter_id)
    return entry.adapter if entry is not None else None


def adapter_for_kind(kind: SourceKind | str) -> SourceFormatAdapter | None:
    """The adapter currently owning *kind*, by the shared :func:`_rank`
    precedence (external outranks built-in; within a tier, last-registered
    wins).  ``None`` when no registered adapter declares *kind*."""
    k = coerce_source_kind(kind)
    best = _best(e for e in _ADAPTERS.values() if k in e.adapter.kinds)
    return best.adapter if best is not None else None


def candidate_owner(path: Path) -> SourceFormatAdapter | None:
    """The single adapter that owns *path* as a name-only candidate, by the
    same :func:`_rank` precedence used for :func:`adapter_for_kind` (R1-R2).
    ``None`` when no adapter claims *path*.

    ``candidate_owner`` and :func:`adapter_for_kind` apply the SAME precedence
    but answer DIFFERENT questions over DIFFERENT sets: "which adapter claims
    this filename" (``is_candidate``) versus "which adapter owns this kind"
    (``kind in adapter.kinds``).  For built-in adapters — which partition
    filenames and kinds together — they coincide.  They can legitimately
    differ for an out-of-tree adapter that declares an existing kind WITHOUT
    claiming a given file (e.g. a plugin owning ``IMAGE_FILE`` but only
    ``*.xyz``): that plugin owns the kind yet does not own an unrelated
    ``a.tif`` candidate.  This is not a contradiction — a discovered
    candidate records THIS owner's id and is always probed through it
    (:meth:`DirectoryIndex.probe_candidate` uses ``candidate.adapter_id``,
    never a kind re-resolution), so discovery and probe never disagree; a
    future opener (R2) must likewise open a discovered candidate through its
    recorded adapter rather than re-resolving by kind."""
    best = _best(e for e in _ADAPTERS.values() if e.adapter.is_candidate(path))
    return best.adapter if best is not None else None


def explicit_source_owner(
    path: Path,
    kind: SourceKind | str,
) -> SourceFormatAdapter | None:
    """Resolve one explicitly selected typed source through the registry.

    Explicit processed outputs deliberately do not reuse name-only raw
    discovery: ``.nexus`` is Browse-readable but is not a raw directory
    candidate.  The requested kind and the registered output capability are
    therefore the authority at this seam.
    """

    source_path = Path(path)
    source_kind = coerce_source_kind(kind)
    owner = adapter_for_kind(source_kind)
    if owner is None:
        return None
    if source_kind is SourceKind.PROCESSED_NEXUS:
        from xrd_tools.io.output_path import is_readable_output_path

        if not owner.is_output_format or not is_readable_output_path(
            source_path
        ):
            return None
    return owner


def all_adapters() -> tuple[SourceFormatAdapter, ...]:
    """All registered adapters, in first-registration order."""
    return tuple(e.adapter for e in _ADAPTERS.values())


__all__ = [
    "CandidatePredicate",
    "FinalizationPolicy",
    "MetadataProviderFactory",
    "ProbeFn",
    "ScanNameFn",
    "SourceFactory",
    "SourceFormatAdapter",
    "adapter_for_kind",
    "all_adapters",
    "candidate_owner",
    "explicit_source_owner",
    "get_adapter",
    "register_adapter",
]
