# xrd_tools/io/output_safety.py
"""Headless source/output path-safety owner.

One pure, import-light guard that answers a single question before any
reduction writer opens or replaces a file: *would this output path overwrite,
or be re-ingested as, one of its own inputs?*

Two production-path failures motivated it (v1.1.2 NXS directory review):

* A container source keeps its full stem, so ``initialize_scan`` derives
  ``<save_dir>/<same-stem>.nxs`` and immediately Overwrite-saves it.  When Save
  Path equals the watched raw directory, source and output are the SAME file and
  the raw acquisition is destroyed before a single frame is reduced.
* An output written into the watched (recursive) traversal set can be
  re-discovered as an input on a later poll.

The guard is raised *before* the writer runs, so on rejection both the source
acquisition and any pre-existing destination keep their bytes.  It relies on
``os.path.samefile`` device+inode identity where the files exist and a resolved,
normcase-normalized comparison otherwise (a not-yet-created output, a
case-insensitive/`\\`-vs-`/` Windows spelling variant), so it never depends on
the filename suffix alone and holds for ``.nxs`` / ``.h5`` / ``.hdf5`` inputs and
for a future ``.nexus`` output extension alike.

Pure: depends only on :mod:`os` and :mod:`pathlib` — no h5py, no Qt, no numpy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


class OutputCollisionError(ValueError):
    """The reduction output would overwrite, or be re-ingested as, an input.

    Raised BEFORE any writer opens/replaces a file, so the source acquisition
    and any existing destination are preserved.  A :class:`ValueError` subclass
    so existing broad ``except ValueError`` / ``except Exception`` run-stop
    handlers still catch it, while call sites that care can catch it by type and
    surface an actionable operator message.
    """


def _norm(path: "os.PathLike[str] | str") -> str:
    """Resolved, normcase-normalized absolute path for equality comparison.

    ``os.path.realpath`` resolves symlinks and ``..``/``.`` segments; for a
    target that does not exist yet it still resolves the existing prefix and
    normalizes the remainder lexically.  ``os.path.normcase`` then folds case and
    ``/`` vs ``\\`` on Windows (identity on POSIX), so a not-yet-created output
    and a Windows spelling variant of the same file compare equal.
    """
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def paths_same_file(a: "os.PathLike[str] | str",
                    b: "os.PathLike[str] | str") -> bool:
    """True when *a* and *b* denote the same file.

    Prefers ``os.path.samefile`` (device+inode identity — catches hard links and
    symlinks) when both paths exist; falls back to resolved-normcase equality for
    a not-yet-created output or a platform/filesystem without stat identity.
    """
    a_s = os.fspath(a)
    b_s = os.fspath(b)
    try:
        if os.path.exists(a_s) and os.path.exists(b_s):
            return os.path.samefile(a_s, b_s)
    except OSError:
        # A transient stat failure must never mask a collision; fall through to
        # the lexical comparison rather than reporting "different".
        pass
    return _norm(a_s) == _norm(b_s)


def _same_dir(a: "os.PathLike[str] | str", b: "os.PathLike[str] | str") -> bool:
    return _norm(a) == _norm(b)


def path_within_dir(child: "os.PathLike[str] | str",
                    parent: "os.PathLike[str] | str") -> bool:
    """True when *child* is *parent* itself or a descendant of it.

    Component-aware (via resolved paths) so ``/data-bar`` is never treated as a
    descendant of ``/data``.
    """
    c = Path(_norm(child))
    p = Path(_norm(parent))
    return c == p or p in c.parents


def check_output_not_source(
    output_path: "os.PathLike[str] | str",
    *,
    input_files: Iterable["os.PathLike[str] | str"] = (),
    watched_dirs: Iterable["os.PathLike[str] | str"] = (),
    recursive: bool = False,
    container_directory_mode: bool = False,
) -> None:
    """Raise :class:`OutputCollisionError` if *output_path* is unsafe to write.

    Parameters
    ----------
    output_path
        The intended reduction output file (e.g. ``<save_dir>/<scan>.nxs``).
    input_files
        Known source file(s) for the run (a single-container input, or the
        container currently being consumed in directory mode).  Each is checked
        for an exact-file collision with *output_path*.
    watched_dirs
        Directory root(s) enumerated for inputs (Image Directory mode).  Used
        only for the directory-policy checks below.
    recursive
        Whether directory discovery descends into subdirectories.
    container_directory_mode
        True when the inputs are self-contained containers (``.nxs`` / ``.h5`` /
        ``.hdf5`` / Eiger master) discovered from a directory — the case whose
        per-container output keeps the source stem and can overwrite or be
        re-ingested.  The directory-policy checks apply only in this mode; the
        exact-file check always applies.

    Returns ``None`` when the output is safe.
    """
    out = os.fspath(output_path)
    out_dir = os.path.dirname(os.path.abspath(out))

    # 1) Exact-file collision with any known input file (always checked).
    for src in input_files:
        if not src:
            continue
        if paths_same_file(out, src):
            raise OutputCollisionError(
                f"Reduction output '{out}' is the same file as input "
                f"'{os.fspath(src)}'; the run would overwrite the raw source. "
                "Choose a Save Path outside the source, then re-run."
            )

    # 2) Directory-policy rejection for raw container directory mode: an output
    #    that lands inside the watched traversal set can be re-discovered as an
    #    input and, because a container output keeps the source stem, can
    #    silently overwrite it.
    if container_directory_mode:
        for root in watched_dirs:
            if not root:
                continue
            if _same_dir(out_dir, root):
                raise OutputCollisionError(
                    f"Save Path '{out_dir}' is the raw container source "
                    f"directory '{os.fspath(root)}'; the processed output would "
                    "overwrite or be re-ingested as a raw input. Choose a "
                    "separate Save Path, then re-run."
                )
            if recursive and path_within_dir(out, root):
                raise OutputCollisionError(
                    f"Save Path '{out_dir}' is inside the recursively-watched "
                    f"source directory '{os.fspath(root)}'; the processed output "
                    "would be re-discovered as a raw input. Choose a Save Path "
                    "outside the watched tree, then re-run."
                )
