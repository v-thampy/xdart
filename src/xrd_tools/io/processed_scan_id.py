# xrd_tools/io/processed_scan_id.py
"""Canonical classifier: is this HDF5 file a *processed* xdart scan?

A processed xdart ``.nxs`` stores REDUCED results (the ``integrated_1d`` /
``integrated_2d`` NXdata stacks) and carries NO raw detector frames.  Fed to a
raw-detector resolver it is dangerous: the generic "largest 3-D dataset" fallback
happily returns ``/entry/integrated_2d/intensity`` — an integrated cake — as
though it were a detector stack (v1.1.2 finding F-NXS-2).

This module concentrates the "is-processed" test in ONE place so every seam that
must not re-ingest a processed output agrees, and by SCHEMA/CONTENT rather than
by filename suffix (a legacy processed ``.nxs`` and a future ``.nexus`` output
both classify correctly).  Ground truth: real processed files in the field carry
the ``integrated_1d`` / ``integrated_2d`` groups but do NOT always carry the
``ssrl_schema`` entry stamp (e.g. ones round-tripped through nexusformat), so the
content-group signal is primary and the schema stamp is a secondary sufficient
marker.

Import-light: depends only on :mod:`h5py` and the frozen schema-identity
constants — so :mod:`xrd_tools.io.image`, :mod:`~xrd_tools.io.nexus` and Qt-free
callers can all import it without a cycle or a heavy dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py

from xrd_tools.io.schema import ACCEPTED_SCHEMA_NAMES, SCHEMA_NAME_ATTR

logger = logging.getLogger(__name__)

__all__ = [
    "ProcessedXdartInputError",
    "is_processed_xdart_file",
    "is_processed_xdart_path",
]

# xdart's own reduced-result group names.  These are the output layout the GUI /
# headless writer produces; a raw acquisition (Eiger master, Bluesky NXWriter,
# plain detector NeXus) never contains them.
_PROCESSED_RESULT_GROUPS = ("integrated_1d", "integrated_2d")


class ProcessedXdartInputError(ValueError):
    """Raised when a processed xdart scan file is offered where a RAW detector
    container is required (e.g. the directory-watch image reader).

    A :class:`ValueError` subclass so existing broad ``except ValueError`` /
    ``except Exception`` handling still catches it, while a caller that wants to
    *skip a processed candidate and continue* can catch it by type.
    """


def _attr_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def is_processed_xdart_file(f: h5py.File, entry: str = "entry") -> bool:
    """True when the OPEN file *f* is a processed xdart scan.

    Positive when either signal is present:

    * an ``integrated_1d`` / ``integrated_2d`` group under the entry (content —
      the primary, always-present signal), or
    * the entry's ``ssrl_schema`` stamp names a processed schema
      (:data:`~xrd_tools.io.schema.ACCEPTED_SCHEMA_NAMES`, covering the
      pre-monorepo name too).

    Never raises: any traversal error answers ``False`` (let the raw path report
    a genuinely broken file its own way).
    """
    try:
        for group in _PROCESSED_RESULT_GROUPS:
            if f"{entry}/{group}" in f:
                return True
        grp = f.get(entry)
        if isinstance(grp, h5py.Group):
            stamp = _attr_str(grp.attrs.get(SCHEMA_NAME_ATTR))
            if stamp in ACCEPTED_SCHEMA_NAMES:
                return True
    except Exception:
        logger.debug("is_processed_xdart_file: traversal error", exc_info=True)
    return False


def is_processed_xdart_path(path, entry: str = "entry") -> bool:
    """True when *path* is a processed xdart scan file.

    Opens *path* read-only.  A path that cannot be opened as HDF5 answers
    ``False`` (it is not identifiable as processed, and the raw path will report
    the real problem).
    """
    try:
        with h5py.File(Path(path), "r") as f:
            return is_processed_xdart_file(f, entry)
    except OSError:
        return False
