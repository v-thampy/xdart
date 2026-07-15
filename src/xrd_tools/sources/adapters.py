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


_ADAPTERS: dict[str, SourceFormatAdapter] = {}
_KIND_INDEX: dict[SourceKind, str] = {}


def register_adapter(adapter: SourceFormatAdapter) -> None:
    """Register *adapter*, overriding any prior adapter with the same id.

    Also updates the kind->adapter index for each of ``adapter.kinds``: the
    most-recently-registered adapter for a given kind wins, mirroring
    :func:`xrd_tools.sources.registry.register_source`'s override semantics.
    """
    _ADAPTERS[adapter.id] = adapter
    for kind in adapter.kinds:
        _KIND_INDEX[kind] = adapter.id


def get_adapter(adapter_id: str) -> SourceFormatAdapter | None:
    return _ADAPTERS.get(adapter_id)


def adapter_for_kind(kind: SourceKind | str) -> SourceFormatAdapter | None:
    """The adapter currently owning *kind*, if any (last-registered wins)."""
    adapter_id = _KIND_INDEX.get(coerce_source_kind(kind))
    return _ADAPTERS.get(adapter_id) if adapter_id is not None else None


def all_adapters() -> tuple[SourceFormatAdapter, ...]:
    """All registered adapters, in registration order."""
    return tuple(_ADAPTERS.values())


def adapters_for_candidate_scan() -> Iterable[SourceFormatAdapter]:
    """Adapters worth consulting for name-only candidate enumeration —
    every registered adapter (built-in and out-of-tree alike)."""
    return all_adapters()


__all__ = [
    "CandidatePredicate",
    "FinalizationPolicy",
    "MetadataProviderFactory",
    "ProbeFn",
    "ScanNameFn",
    "SourceFactory",
    "SourceFormatAdapter",
    "adapter_for_kind",
    "adapters_for_candidate_scan",
    "all_adapters",
    "get_adapter",
    "register_adapter",
]
