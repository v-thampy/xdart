# xrd_tools/io/output_path.py
"""The single shared owner of generated-output *path selection* (P4/OUT-1).

New generated output is written as ``.nexus`` and only current ``.nexus``
artifacts participate in processed Browse/Append reuse.  Raw ``.nxs`` remains
supported by structure-aware source discovery.  Before
this owner existed the decision was duplicated as a hard-coded ``+ ".nxs"`` in
every producer — the headless series/watcher, ``LiveScan``, both wranglers, the
scratch placeholder and the run-end fallback — so the suffix could not be
changed anywhere without drifting somewhere else.

Scope (deliberately narrow).  This module **selects paths only**.  It does not
inspect file content, schema or lineage; it owns no writer, transaction or
lease; and it never creates, opens, moves or deletes a file.  Everything it
touches is a name and an existence test:

* **Trusted inputs** — the already validated scan name and destination
  directory, and the two existing write-mode spellings.
* **Fallible boundaries** — path existence, explicit-target spelling, and
  source/output identity.  Source/output collision checking stays with
  :mod:`xrd_tools.io.output_safety`, runs on the *resolved* target, and remains
  suffix-independent.
* **Out of scope** — a race between selection and the writer opening the file.
  That belongs to the future output-transaction owner (H23).

NeXus-family suffixes are raw candidates too: content probing, not a filename,
separates raw detector containers from current processed output.

Pure: depends only on :mod:`os` and :mod:`pathlib` — no h5py, no numpy, no Qt,
no ``xdart``, no writer and no schema import.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Suffix for every newly generated output file.
NEW_OUTPUT_SUFFIX = ".nexus"

#: Historical/raw NeXus spelling.  It remains a raw-source suffix but is no
#: longer a processed Browse or Append target.
LEGACY_OUTPUT_SUFFIX = ".nxs"

#: Current processed-output suffixes accepted by Browse/Append.
READABLE_OUTPUT_SUFFIXES = (NEW_OUTPUT_SUFFIX,)

#: The two existing write-mode spellings.
APPEND_MODE = "Append"
OVERWRITE_MODE = "Overwrite"

__all__ = [
    "NEW_OUTPUT_SUFFIX",
    "LEGACY_OUTPUT_SUFFIX",
    "READABLE_OUTPUT_SUFFIXES",
    "APPEND_MODE",
    "OVERWRITE_MODE",
    "default_output_path",
    "resolve_output_target",
    "is_readable_output_path",
]


def is_readable_output_path(path: "os.PathLike[str] | str") -> bool:
    """True when *path* names a processed-output file this reader accepts.

    Suffix recognition is case-insensitive (an operator's ``SCAN.NXS`` from a
    case-preserving share is the same file), and the path is never rewritten —
    the answer is a pure classification.
    """
    return Path(os.fspath(path)).suffix.lower() in READABLE_OUTPUT_SUFFIXES


def _is_append(mode: object) -> bool:
    """Whether *mode* is the explicit Append policy.

    Anything that is not recognisably ``"Append"`` falls through to a freshly
    generated ``.nexus`` target.
    """
    return str(mode).strip().lower() == APPEND_MODE.lower()


def _existing_sibling(directory: "os.PathLike[str] | str",
                      scan_name: str) -> "Path | None":
    """The real on-disk output sibling of *scan_name*, or ``None``.

    The suffix matches case-insensitively (§2 rule 5), the stem exactly.  The
    directory is enumerated rather than probed so the answer carries the file's
    REAL spelling: probing ``<stem>.nexus`` is True for a stored ``<stem>.NEXUS``
    on a case-insensitive filesystem, which silently respelled the path.
    Within the current suffix class the exact canonical lowercase spelling wins,
    and remaining ties resolve in sorted order.
    """
    root = Path(os.fspath(directory))
    try:
        names = os.listdir(os.fspath(root))
    except OSError:
        return None          # unreadable destination: generate, as before
    by_class: dict[str, list[str]] = {}
    for name in names:
        stem, dot, suffix = name.rpartition(".")
        cls = f".{suffix}".lower()
        # Real FILES only, filtered BEFORE ranking.  Filtering afterwards let a
        # directory named `scan.nexus` win its class on canonical spelling and
        # then fail the file test, abandoning the whole class — hiding a valid
        # `scan.NEXUS` file and dropping Append to a legacy sibling.
        if (dot and stem == scan_name and cls in READABLE_OUTPUT_SUFFIXES
                and (root / name).is_file()):
            by_class.setdefault(cls, []).append(name)
    for cls in READABLE_OUTPUT_SUFFIXES:
        matches = sorted(by_class.get(cls, ()))
        if matches:
            canonical = f"{scan_name}{cls}"
            return root / (canonical if canonical in matches else matches[0])
    return None


def default_output_path(directory: "os.PathLike[str] | str",
                        scan_name: str) -> Path:
    """``<directory>/<scan_name>.nexus`` — the generated target for a new run.

    *scan_name* is a bare scan stem (the canonical
    ``scan_name_from_source``/``Path.stem`` form used throughout the GUI and the
    headless layer), never a filename with a suffix.  Nothing is created here.
    """
    return Path(os.fspath(directory)) / f"{scan_name}{NEW_OUTPUT_SUFFIX}"


def resolve_output_target(
    directory: "os.PathLike[str] | str",
    scan_name: str,
    *,
    mode: object,
    explicit_target: "os.PathLike[str] | str | None" = None,
) -> Path:
    """The exact output file a run should write, by policy.

    Parameters
    ----------
    directory
        Destination directory for a generated target.
    scan_name
        Bare scan stem for a generated target.
    mode
        The run's write mode — the existing ``"Append"`` / ``"Overwrite"``
        spellings.
    explicit_target
        A complete requested path.  Its stem and directory are retained, while
        any non-current suffix is normalized to ``.nexus``.

    Policy
    ------
    1. An explicit target keeps its directory/stem and uses ``.nexus``.
    2. ``Overwrite`` always selects ``<scan_name>.nexus``.  It never discovers
       or replaces a sibling legacy ``.nxs`` implicitly.
    3. ``Append`` reuses an existing ``.nexus`` or selects a new one.  A sibling
       ``.nxs`` is raw/foreign to this policy and is never reused.
    4. Sibling discovery matches the suffix case-insensitively and returns the
       file's real on-disk spelling; the scan stem is matched exactly.

    Returns the selected path.  The file is neither created nor opened, and the
    caller still runs :func:`~xrd_tools.io.output_safety.check_output_not_source`
    on this resolved target before any writer opens it.
    """
    if explicit_target is not None and os.fspath(explicit_target) != "":
        requested = Path(os.fspath(explicit_target))
        return (
            requested
            if requested.suffix.casefold() == NEW_OUTPUT_SUFFIX
            else requested.with_suffix(NEW_OUTPUT_SUFFIX)
        )

    if _is_append(mode):
        existing = _existing_sibling(directory, scan_name)
        if existing is not None:
            return existing
    return default_output_path(directory, scan_name)
