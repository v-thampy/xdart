# -*- coding: utf-8 -*-
"""Directory scan discovery — walk a folder for a given source kind.

The "Directory" entry mode of the shared source panel: given a directory + a
scan kind, walk it (optionally recursively) and return one openable
:class:`SourceSpec` per scan found.  Generalizes
``TiffSeriesSource.from_directory`` across kinds.  Pure/Qt-free.
"""

from __future__ import annotations

from pathlib import Path

from xrd_tools.core.scan import SourceKind, SourceSpec, coerce_source_kind

_NEXUS_EXTS = {".nxs", ".h5", ".hdf5", ".cxi"}


def _walk_files(directory: Path, recursive: bool) -> list[Path]:
    it = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in it if p.is_file())


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
        return [SourceSpec(f, kind, options=dict(options))
                for f in files if f.suffix.lower() in _NEXUS_EXTS]

    if kind in (SourceKind.TIFF_SERIES, SourceKind.IMAGE_FILE):
        from xrd_tools.io.image import SUPPORTED_EXTS
        has_images = any(f.suffix.lower() in SUPPORTED_EXTS for f in files)
        if not has_images:
            return []
        return [SourceSpec(directory, SourceKind.TIFF_SERIES,
                           options=dict(options))]

    raise ValueError(f"discover_scans: unsupported kind {kind.value!r}")


__all__ = ["discover_scans"]
