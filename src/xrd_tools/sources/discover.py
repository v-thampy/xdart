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

from dataclasses import dataclass
from pathlib import Path

# natsort is a hard dependency (pyproject).  Import it at module top and let an
# ImportError surface: a broken install must fail loud, not silently degrade to
# lexicographic order — which mis-orders a discovered stitch/RSM group
# (scan_1, scan_10, scan_2) with no warning, silently corrupting the merge.
from natsort import os_sorted

from xrd_tools.core.filters import compile_filter
from xrd_tools.core.scan import SourceKind, SourceSpec, coerce_source_kind

_NEXUS_EXTS = {".nxs", ".h5", ".hdf5", ".cxi"}


def _walk_files(directory: Path, recursive: bool) -> list[Path]:
    it = directory.rglob("*") if recursive else directory.iterdir()
    files = [p for p in it if p.is_file()]
    return list(os_sorted(files))


def discover_scans(directory, kind, *, recursive: bool = False,
                   **options) -> list[SourceSpec]:
    """Return one :class:`SourceSpec` per scan found in ``directory`` for ``kind``.

    * **SPEC** — every SPEC file (content-detected) × each of its scans →
      ``SourceSpec(spec_file, SPEC, options={"scan": "N.1", ...})``.
    * **NeXus / Eiger / processed NeXus** — every ``.nxs``/``.h5``/``.hdf5``/
      ``.cxi`` master → one spec each.
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
        from xrd_tools.io.image import _is_eiger_master
        from xrd_tools.sources.registry import guess_source_kind
        out = []
        for f in files:
            if f.suffix.lower() not in _NEXUS_EXTS:
                continue
            if kind is SourceKind.EIGER_MASTER:
                if not _is_eiger_master(f):       # skip the sibling _data_ files
                    continue
                out.append(SourceSpec(f, SourceKind.EIGER_MASTER,
                                      options=dict(options)))
                continue
            # raw / processed NeXus: skip obvious Eiger data files, and classify
            # each master to its REAL kind so a processed .nxs opens as
            # PROCESSED_NEXUS (linked raw + scan_data), not a raw stack.
            if "_data_" in f.stem and not f.stem.endswith("_master"):
                continue
            try:
                actual = guess_source_kind(f)
            except Exception:
                actual = kind
            if actual not in (SourceKind.NEXUS_STACK, SourceKind.PROCESSED_NEXUS,
                              SourceKind.EIGER_MASTER):
                actual = kind
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


def enumerate_candidates(directory, *, recursive: bool = False,
                         name_filter: str | None = None) -> list[Candidate]:
    """Name-only candidate enumeration across every registered format adapter
    at once (R1).

    Pure filesystem inspection — directory listing, each adapter's filename
    rule, and one ``stat()`` per matched file.  NEVER opens a file's content
    (no ``h5py.File``, detector-dataset resolution, metadata harvest, or
    processed-file classification); that is each adapter's explicit
    :meth:`~xrd_tools.sources.adapters.SourceFormatAdapter.probe`, called only
    when a caller asks to probe one specific candidate.

    A file matched by more than one adapter's ``is_candidate`` keeps the
    FIRST match in registration order (the built-in adapters partition
    filenames by extension so this never happens for them; an out-of-tree
    adapter that wants priority over a built-in should register after it —
    :func:`~xrd_tools.sources.adapters.adapter_for_kind`'s "last registered
    wins" is a *kind* lookup and does not apply to candidate ownership).

    Deterministic natural order, independent of filesystem enumeration order
    (:func:`natsort.os_sorted`, matching :func:`discover_scans`)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = _walk_files(directory, recursive)

    name_ok = compile_filter(name_filter)

    _ensure_builtin_adapters_registered()
    from xrd_tools.sources.adapters import all_adapters
    adapters = all_adapters()

    out: list[Candidate] = []
    for f in files:
        if not name_ok(f.name):
            continue
        for adapter in adapters:
            if not adapter.is_candidate(f):
                continue
            try:
                stat = f.stat()
            except OSError:
                break  # vanished mid-walk; not a candidate this poll
            out.append(Candidate(f, adapter.id, stat.st_size, stat.st_mtime_ns))
            break
    return out


__all__ = ["Candidate", "discover_scans", "enumerate_candidates"]
