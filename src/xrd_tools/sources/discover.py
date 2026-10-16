# -*- coding: utf-8 -*-
"""Directory scan discovery — walk a folder for a given source kind.

The "Directory" entry mode of the shared source panel: given a directory + a
scan kind, walk it (optionally recursively) and return one openable
:class:`SourceSpec` per scan found.  Generalizes
``TiffSeriesSource.from_directory`` across kinds.  Pure/Qt-free.

R1 adds :func:`enumerate_candidates`: a NAME-ONLY sweep across every
registered format adapter at once (vs. :func:`discover_scans`'s single
explicit ``kind`` per call) — the shape
:class:`~xrd_tools.sources.directory_index.DirectoryIndex` polls.  It performs
no HDF5 opens, detector-dataset resolution, or processed-file classification;
those live behind each adapter's explicit ``probe`` (see
:mod:`xrd_tools.sources.adapters`).  ``discover_scans`` itself is unchanged —
the GUI's ``scan_source_widget.py`` still calls it directly.
"""

from __future__ import annotations

import errno
import os
from dataclasses import dataclass
from pathlib import Path

# natsort is a hard dependency (pyproject).  Import it at module top and let an
# ImportError surface: a broken install must fail loud, not silently degrade to
# lexicographic order — which mis-orders a discovered stitch/RSM group
# (scan_1, scan_10, scan_2) with no warning, silently corrupting the merge.
from natsort import os_sorted

from xrd_tools.core.filters import compile_filter
from xrd_tools.core.scan import SourceKind, SourceSpec, coerce_source_kind

_NEXUS_EXTS = {".nxs", ".h5", ".hdf5", ".nexus", ".cxi"}


def _walk_files(directory: Path, recursive: bool) -> list[Path]:
    return list(os_sorted(_iter_files_unordered(directory, recursive)))


def _iter_files_unordered(directory: Path, recursive: bool) -> list[Path]:
    """Regular files under *directory* in raw filesystem-enumeration order —
    no natural sort.  The sort is the expensive part of a poll; splitting it
    out lets :class:`~xrd_tools.sources.directory_index.DirectoryIndex` detect
    an unchanged directory (order-independent map comparison) and skip the
    sort entirely (R1-R5)."""
    it = directory.rglob("*") if recursive else directory.iterdir()
    return [p for p in it if p.is_file()]


def discover_scans(directory, kind, *, recursive: bool = False,
                   **options) -> list[SourceSpec]:
    """Return one :class:`SourceSpec` per scan found in ``directory`` for ``kind``.

    * **SPEC** — every SPEC file (content-detected) × each of its scans →
      ``SourceSpec(spec_file, SPEC, options={"scan": "N.1", ...})``.
    * **NeXus / Eiger / processed NeXus** — every ``.nxs``/``.h5``/``.hdf5``/
      ``.nexus``/``.cxi`` master → one spec each.
    * **TIFF / RAW image series** — the directory itself as one image series
      (`TiffSeriesSource.from_directory`); per-``_scanN_`` splitting is a future
      refinement.

    ``options`` (e.g. ``image_dir`` / ``read_image_kwargs``) thread into every
    returned spec.  Raises ``ValueError`` for an unsupported kind."""
    directory = Path(directory)
    kind = coerce_source_kind(kind)
    if not directory.is_dir():
        return []
    files = _walk_files(directory, recursive)

    if kind is SourceKind.SPEC:
        from xrd_tools.io.spec import is_spec_file, list_spec_scans
        out: list[SourceSpec] = []
        for f in files:
            if not is_spec_file(f):
                continue
            for scan in list_spec_scans(f):
                out.append(SourceSpec(f, SourceKind.SPEC,
                                      options={"scan": scan, **options}))
        return out

    if kind in (SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER,
                SourceKind.PROCESSED_NEXUS):
        from xrd_tools.io.image import eiger_content
        from xrd_tools.sources.registry import (_nexus_is_candidate,
                                                guess_source_kind)
        out = []
        for f in files:
            if f.suffix.lower() not in _NEXUS_EXTS:
                continue
            if kind is SourceKind.EIGER_MASTER:
                # Masters link their data files by those exact names, so the
                # files are skipped unopened; any other is a master only by its
                # DECTRIS content, whatever it is called.
                if not _nexus_is_candidate(f) or not eiger_content(f)[0]:
                    continue
                out.append(SourceSpec(f, SourceKind.EIGER_MASTER,
                                      options=dict(options)))
                continue
            # Raw / processed NeXus: skip exact Eiger data sidecars, classify
            # each master to its real kind, and keep the requested lanes
            # disjoint.  Only a current .nexus record enters the processed lane;
            # historical processed markers never fall back to a raw stack.
            if not _nexus_is_candidate(f):
                continue
            try:
                actual = guess_source_kind(f)
            except Exception:
                # Name-only/partially-landed raw HDF remains discoverable, but
                # a processed request never gains authority from a failed probe.
                actual = SourceKind.NEXUS_STACK
            if actual is not kind:
                continue
            out.append(SourceSpec(f, actual, options=dict(options)))
        return out

    if kind in (SourceKind.TIFF_SERIES, SourceKind.IMAGE_FILE):
        from xrd_tools.io.image import SUPPORTED_EXTS
        has_images = any(f.suffix.lower() in SUPPORTED_EXTS for f in files)
        if not has_images:
            return []
        return [SourceSpec(directory, SourceKind.TIFF_SERIES,
                           options=dict(options))]

    raise ValueError(f"discover_scans: unsupported kind {kind.value!r}")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One name-only discovered source candidate and its cheap filesystem
    state.  Produced ONLY by :func:`enumerate_candidates` — never by opening
    the file.  ``(size, mtime_ns)`` is the cheap version stamp a
    :class:`~xrd_tools.sources.directory_index.DirectoryIndex` compares
    across polls to detect added/changed/removed candidates without reading
    file content."""

    path: Path
    adapter_id: str
    size: int
    mtime_ns: int

    @property
    def version_stamp(self) -> tuple[int, int]:
        return (self.size, self.mtime_ns)


def _ensure_builtin_adapters_registered() -> None:
    """Side-effect-only import: ``xrd_tools.sources.registry`` registers the
    built-in adapters at its own module-import time.  A caller who only ever
    touches ``discover``/``directory_index`` (never ``open_source`` or
    ``registry`` directly) would otherwise see an adapter registry with only
    whatever it registered itself — this guarantees the built-ins are present
    without making this module import registry's heavier dependency chain
    (h5py/fabio-touching source classes) at ITS OWN module-import time."""
    import xrd_tools.sources.registry  # noqa: F401


def collect_candidates(directory, *, recursive: bool = False,
                       name_filter: str | None = None) -> list[Candidate]:
    """Name-only candidate collection in raw filesystem order (UNORDERED).

    The cheap half of enumeration: directory listing, each candidate's owning
    adapter (by the shared precedence in
    :func:`~xrd_tools.sources.adapters.candidate_owner`), and one ``stat()``
    per matched file.  NEVER opens a file's content (no ``h5py.File``,
    detector-dataset resolution, metadata harvest, or processed-file
    classification); that is each adapter's explicit
    :meth:`~xrd_tools.sources.adapters.SourceFormatAdapter.probe`, called only
    when a caller asks to probe one specific candidate.

    Ownership follows ONE precedence rule shared with ``adapter_for_kind``
    (R1-R2): an externally registered adapter outranks a built-in regardless
    of import order, and within a tier the most recently registered adapter
    wins.  The discovered candidate records its owning adapter's id and is
    always probed through THAT adapter (never re-resolved by kind), so
    discovery and probe never disagree; see
    :func:`~xrd_tools.sources.adapters.candidate_owner` for how candidate
    ownership relates to (and can legitimately differ from) kind ownership.
    :func:`~xrd_tools.sources.directory_index.DirectoryIndex` calls this
    directly and sorts only when the candidate map actually changed;
    :func:`enumerate_candidates` sorts unconditionally for its public
    naturally-ordered contract."""
    directory = Path(directory)
    if not directory.is_dir():
        return []

    name_ok = compile_filter(name_filter)

    _ensure_builtin_adapters_registered()
    from xrd_tools.sources.adapters import candidate_owner

    out: list[Candidate] = []
    for f in _iter_files_unordered(directory, recursive):
        if not name_ok(f.name):
            continue
        owner = candidate_owner(f)
        if owner is None:
            continue
        try:
            stat = f.stat()
        except OSError:
            continue  # vanished mid-walk; not a candidate this poll
        out.append(Candidate(f, owner.id, stat.st_size, stat.st_mtime_ns))
    return out


#: The stat errors ``Path.is_file`` reads as "not a regular file".
_NOT_A_FILE_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})


def _scandir_files(directory: Path):
    """``(path, name, stat)`` per regular file in *directory*, as ``Path.iterdir`` + ``is_file`` sees them."""
    with os.scandir(directory) as entries:
        for entry in entries:
            try:
                regular = entry.is_file()
            except OSError as error:
                if error.errno not in _NOT_A_FILE_ERRNOS:
                    raise
                continue
            if regular:
                yield entry.path, entry.name, entry.stat


def collect_owned_candidates(directory, *, recursive: bool = False,
                             name_filter: str | None = None,
                             memo: dict | None = None) -> list[Candidate]:
    """:func:`collect_candidates` for one DirectoryIndex poll: the same candidates.

    One registry copy serves the whole listing, so no adapter predicate runs
    under the registry lock.  ``memo`` belongs to the caller's index.  While
    every registered adapter is a built-in whose predicate is a pure filename
    rule (``registry.PURE_NAME_PREDICATES``), it keeps each listed path's
    owner from poll to poll; any other registry, or any change of it, asks
    every predicate again.  A file whose path, owner and stamp are unchanged
    keeps its previous :class:`Candidate` object.
    """
    directory = Path(directory)
    if not directory.is_dir():
        if memo is not None:
            memo.clear()
        return []
    name_ok = compile_filter(name_filter)
    _ensure_builtin_adapters_registered()
    from xrd_tools.sources.adapters import candidate_owner_among, registry_entries, registry_identity
    from xrd_tools.sources.registry import name_pure_registry

    entries = registry_entries()
    registry = registry_identity(entries)
    pure = name_pure_registry(entries)
    memo = {} if memo is None else memo
    owners = memo.get("owners", {}) if pure and memo.get("registry") == registry else {}
    previous = memo.get("candidates", {})
    listed = (
        ((str(f), f.name, f.stat) for f in _iter_files_unordered(directory, True))
        if recursive else _scandir_files(directory)
    )
    kept_owners: dict = {}
    kept: dict[str, Candidate] = {}
    out: list[Candidate] = []
    for key, name, stat_of in listed:
        if not name_ok(name):
            continue
        owner = owners[key] if key in owners else candidate_owner_among(entries, Path(key))
        kept_owners[key] = owner
        if owner is None:
            continue
        try:
            state = stat_of()
        except OSError:
            continue  # vanished mid-walk; not a candidate this poll
        candidate = previous.get(key)
        if candidate is None or (candidate.adapter_id, candidate.size, candidate.mtime_ns) != (
                owner.id, state.st_size, state.st_mtime_ns):
            candidate = Candidate(Path(key), owner.id, state.st_size, state.st_mtime_ns)
        kept[key] = candidate
        out.append(candidate)
    memo.clear()
    memo.update(candidates=kept, **({"registry": registry, "owners": kept_owners} if pure else {}))
    return out


def sort_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Naturally order *candidates* by path (``natsort.os_sorted``), matching
    :func:`discover_scans`'s file ordering — the byte-identical order the
    pre-split :func:`enumerate_candidates` produced by sorting files first."""
    return list(os_sorted(candidates, key=lambda c: c.path))


def enumerate_candidates(directory, *, recursive: bool = False,
                         name_filter: str | None = None) -> list[Candidate]:
    """Name-only candidate enumeration across every registered format adapter
    at once, in deterministic natural order (R1).

    Equivalent to :func:`sort_candidates` of :func:`collect_candidates`; see
    those for the ownership-precedence and no-content-open guarantees.  The
    natural order is independent of filesystem enumeration order
    (:func:`natsort.os_sorted`, matching :func:`discover_scans`)."""
    return sort_candidates(
        collect_candidates(directory, recursive=recursive, name_filter=name_filter))


__all__ = [
    "Candidate",
    "collect_candidates",
    "discover_scans",
    "enumerate_candidates",
    "sort_candidates",
]
