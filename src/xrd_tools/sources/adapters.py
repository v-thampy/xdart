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
from threading import RLock
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


#: id -> entry.  External registration replaces an existing id; lower-tier
#: builtin bootstrap never replaces an external entry with the same id.  There
#: is NO separate kind index that could go stale (R1-R2): every kind/candidate
#: lookup resolves live against ``adapter.kinds`` / ``is_candidate`` below.
_ADAPTERS: dict[str, _Entry] = {}
_next_seq = 0
#: Protects registry mutation and sequence allocation only.  Readers copy
#: entries while holding it and run all caller/plugin behavior after release.
_REGISTRY_LOCK = RLock()


def _rank(entry: _Entry) -> tuple[int, int]:
    """Precedence key — higher wins.  ONE rule shared by candidate discovery,
    :func:`adapter_for_kind`, probing, and opening (R1-R2):

    * an externally registered adapter (``builtin=False``) ALWAYS outranks a
      built-in, regardless of import order — so a plugin registered before the
      lazy built-in bootstrap is never displaced when the built-ins later
      register, and one registered after still wins;
    * within a tier, the most recently registered adapter wins
      (last-registered-wins), providing deterministic override semantics.
    """
    return (0 if entry.builtin else 1, entry.seq)


def _best(entries: Iterable[_Entry]) -> _Entry | None:
    entries = list(entries)
    return max(entries, key=_rank) if entries else None


def register_adapter(adapter: SourceFormatAdapter, *, builtin: bool = False) -> None:
    """Register *adapter* under its stable id.

    ``builtin`` marks the lower-precedence tier used by the in-tree formats
    (see :func:`_rank`); out-of-tree callers leave it False so their adapters
    outrank the built-ins deterministically.  A builtin bootstrap therefore
    leaves an existing external entry with the same id untouched.  Every other
    same-id registration replaces its entry; because kind/candidate ownership
    is resolved live (never cached in a side index), a replacement that declares
    fewer kinds leaves NO stale kind mapping behind.
    """
    global _next_seq
    with _REGISTRY_LOCK:
        existing = _ADAPTERS.get(adapter.id)
        if builtin and existing is not None and not existing.builtin:
            return
        _ADAPTERS[adapter.id] = _Entry(
            adapter=adapter,
            builtin=builtin,
            seq=_next_seq,
        )
        _next_seq += 1


def _ensure_builtin_adapters() -> None:
    """Lazily run the builtin bootstrap before reading the registry.

    Importing this module remains light.  The first public lookup imports the
    bootstrap owner; subsequent calls reuse Python's module cache.
    """
    import xrd_tools.sources.registry  # noqa: F401


def _snapshot_entries() -> tuple[_Entry, ...]:
    """Copy the immutable entries while holding the registry lock."""
    with _REGISTRY_LOCK:
        return tuple(_ADAPTERS.values())


def get_adapter(adapter_id: str) -> SourceFormatAdapter | None:
    _ensure_builtin_adapters()
    entries = _snapshot_entries()
    entry = next(
        (entry for entry in entries if entry.adapter.id == adapter_id),
        None,
    )
    return entry.adapter if entry is not None else None


def adapter_for_kind(kind: SourceKind | str) -> SourceFormatAdapter | None:
    """The adapter currently owning *kind*, by the shared :func:`_rank`
    precedence (external outranks built-in; within a tier, last-registered
    wins).  ``None`` when no registered adapter declares *kind*."""
    _ensure_builtin_adapters()
    entries = _snapshot_entries()
    k = coerce_source_kind(kind)
    best = _best(e for e in entries if k in e.adapter.kinds)
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
    :func:`xrd_tools.sources.registry.open_source` likewise prefers a compatible
    candidate owner before falling back to the kind owner."""
    _ensure_builtin_adapters()
    entries = _snapshot_entries()
    best = _best(e for e in entries if e.adapter.is_candidate(path))
    return best.adapter if best is not None else None


def explicit_source_owner(
    path: Path,
    kind: SourceKind | str,
) -> SourceFormatAdapter | None:
    """Resolve one explicitly selected typed source through the registry.

    Explicit processed outputs require strict current-schema admission rather
    than inheriting the broader raw-candidate suffix set.  The requested kind,
    registered output capability, and current processed identity must agree.
    """

    _ensure_builtin_adapters()
    entries = _snapshot_entries()
    source_path = Path(path)
    source_kind = coerce_source_kind(kind)
    best = _best(e for e in entries if source_kind in e.adapter.kinds)
    if best is None:
        return None
    owner = best.adapter
    if source_kind is SourceKind.PROCESSED_NEXUS:
        from xrd_tools.io.output_path import is_readable_output_path
        from xrd_tools.io.processed_scan_id import (
            is_current_processed_xdart_path,
        )

        if not owner.is_output_format or not is_readable_output_path(
            source_path
        ) or not is_current_processed_xdart_path(source_path):
            return None
    return owner


def all_adapters() -> tuple[SourceFormatAdapter, ...]:
    """All registered adapters, in first-registration order."""
    _ensure_builtin_adapters()
    entries = _snapshot_entries()
    return tuple(e.adapter for e in entries)


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
