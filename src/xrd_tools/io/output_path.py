# xrd_tools/io/output_path.py
"""The single shared owner of generated-output *path selection* (P4/OUT-1).

New generated output is written as ``.nexus`` and only current ``.nexus``
artifacts participate in processed Browse/Append reuse.  Raw ``.nxs`` remains
supported by structure-aware source discovery.  Before
this owner existed the decision was duplicated as a hard-coded ``+ ".nxs"`` in
every producer — headless series/watchers, GUI run owners, scratch placeholders,
and run-end fallbacks — so the suffix could not be
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

Pure: depends only on small standard-library path/hash helpers — no h5py, no
numpy, no Qt, no ``xdart``, no writer and no schema import.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

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
    "artifact_family_from_source",
    "is_artifact_family",
    "resolve_finite_output_target",
    "FINITE_OPERATION_SLOTS",
]


#: A public family must stay HUMAN-READABLE (ADR-0010), so the alphabet is wide
#: enough for real beamline stems -- spaces and accented letters are routine --
#: and narrow enough that nothing structural can hide in a filename.
#:
#: WIDENED 2026-09-04 on the maintainer's ruling.  The previous ASCII-only class
#: sent `Sample A 001` to an `artifact-<24 hex>` fallback, putting a content
#: hash in a public name in direct contradiction of the ADR carried by this same
#: work.  That fallback is now DELETED, not merely narrowed: an unusable stem
#: raises.  The class below is expressed as a FORBIDDEN set for that reason.
#: Characters a public family may never contain, and why: POSIX and Windows path
#: separators, the Windows drive/alternate-data-stream separator, the four
#: Windows wildcard/redirection characters, quote, and any control character.
#: Everything else -- spaces, `#`, parentheses, accented and non-Latin letters --
#: is ORDINARY punctuation in a filename and stays readable.
_FAMILY_FORBIDDEN = r'\x00-\x1f\x7f/\\:*?"<>|'

#: A family is at most 80 characters, contains nothing forbidden, and neither
#: starts with `.` `-` `_` or whitespace (hidden files, option-looking names, a
#: bare-slot look-alike, and leading blanks) nor ends with a space or dot, which
#: Windows silently strips.
_ARTIFACT_FAMILY = re.compile(
    rf"[^{_FAMILY_FORBIDDEN}.\-_\s][^{_FAMILY_FORBIDDEN}]{{0,78}}[^{_FAMILY_FORBIDDEN}\s.]\Z"
    rf"|[^{_FAMILY_FORBIDDEN}.\-_\s]\Z",
    re.UNICODE,
)

#: The CLOSED public operation vocabulary (ADR-0010, carried as 491b9413).
#: A generated public filename is ``<root-family><slot>.nexus`` and the
#: separator is an UNDERSCORE, not the dot the superseded immutable-successor
#: route used -- a dot reads as a suffix to every path tool, which is how
#: ``sample.average-<hex>.nexus`` came to look like a versioned sibling.
#:
#: Closed on purpose.  An open token would let a caller mint a public name this
#: policy never reviewed, and "do not parse generated suffixes to recover a root
#: family" then leaves no way to read that name back.
FINITE_OPERATION_SLOTS: dict[str, str] = {
    "int-1d": "_int1d",
    "int-2d": "_int2d",
    "average": "_average",
    "reintegrate-1d": "_reintegrate1d",
    "reintegrate-2d": "_reintegrate2d",
    "stitch-1d": "_stitch1d",
    "stitch-2d": "_stitch2d",
    "rsm": "_rsm",
}


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


def is_artifact_family(value: object) -> bool:
    """True when *value* may be a public result family.

    The one place to ASK the question, so callers stop re-deriving the rule.
    A name that fails this cannot produce a conforming public name at all --
    every finite operation resolves `<family><slot>.nexus` -- so a caller that
    is about to RECORD a family should check here first and refuse early,
    rather than let a run finish and strand its own artifact.
    """
    return type(value) is str and _ARTIFACT_FAMILY.fullmatch(value) is not None


def artifact_family_from_source(
    source: "os.PathLike[str] | str",
    persisted_family: str | None = None,
) -> str:
    """Return the bounded lineage family for an immutable successor.

    A valid persisted family is authoritative.  Without one, the complete
    source stem is retained verbatim when safe; generated-looking suffixes are
    deliberately *not* guessed or stripped.  An unsafe basename RAISES -- see
    the body -- because a public pathname must never carry a content hash.
    """

    if persisted_family is not None:
        if type(persisted_family) is not str or not _ARTIFACT_FAMILY.fullmatch(
            persisted_family
        ):
            raise ValueError("artifact family is not canonical")
        return persisted_family
    try:
        basename = os.path.basename(os.fspath(source))
    except TypeError as error:
        raise TypeError("artifact source must be path-like") from error
    if type(basename) is not str or not basename or "\x00" in basename:
        raise ValueError("artifact source basename is invalid")
    stem = Path(basename).stem
    if _ARTIFACT_FAMILY.fullmatch(stem):
        return stem
    # NO HASH FALLBACK.  It used to return `artifact-<24 hex>` here, which put a
    # content hash into a PUBLIC filename in direct contradiction of ADR-0010,
    # and did so silently -- the operator saw an unreadable name and no reason
    # for it.  With the alphabet widened to real beamline stems, what remains
    # unmatched is structurally unusable rather than merely unusual, so say so.
    raise ValueError(
        f"artifact source stem cannot be a public result family: {stem!r}. "
        "A family may not start with '.', '-' or '_', may not contain a path "
        "separator or ':', and is at most 80 characters. Rename the source, or "
        "pass an explicit result family."
    )


def resolve_finite_output_target(
    directory: "os.PathLike[str] | str",
    artifact_family: str,
    *,
    operation_token: str,
) -> Path:
    """Resolve the one stable public slot ``<family><slot>.nexus``.

    Repeating an operation for the same family returns the SAME path, occupied
    or not: replacing that operation's own slot is the policy, and selecting a
    different name on a second run is exactly the accumulating public sequence
    ADR-0010 forbids.  Whether an occupied slot may be replaced is a
    publication decision owned by the transaction, not a naming one.

    Deliberately impossible to misuse.  This owner takes no version identity and
    no explicit target, so no caller can reintroduce a public version name or
    bypass the vocabulary at this seam -- the previous signature accepted both,
    and the rule was merely unbroken rather than unbreakable.  The version, the
    science identity and the source set survive in persisted provenance, which
    is where exact identity belongs; the public name stays human-readable.
    """

    if type(artifact_family) is not str or not _ARTIFACT_FAMILY.fullmatch(
        artifact_family
    ):
        raise ValueError("artifact family is not canonical")
    slot = (
        FINITE_OPERATION_SLOTS.get(operation_token)
        if type(operation_token) is str
        else None
    )
    if slot is None:
        raise ValueError(
            "finite operation is not one of the stable public slots: "
            + ", ".join(sorted(FINITE_OPERATION_SLOTS))
        )
    try:
        root = Path(os.fspath(directory))
    except TypeError as error:
        raise TypeError("finite output directory must be path-like") from error
    return root / f"{artifact_family}{slot}{NEW_OUTPUT_SUFFIX}"


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
