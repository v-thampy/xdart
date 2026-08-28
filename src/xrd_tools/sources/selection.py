"""Typed, value-only source selections frozen at Run boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from xrd_tools.core.scan import SourceKind, SourceSpec, coerce_source_kind

_FRAME_SUFFIX = re.compile(r"^(.*?)[_-](\d+)$")
_SINGLE_IMAGE_SELECTION_MODE = "single_image"
_SINGLE_IMAGE_SUFFIXES = frozenset({
    ".cbf", ".edf", ".img", ".mar3450", ".raw", ".tif", ".tiff",
})


def normalize_metadata_format(value: object) -> str | None:
    """Normalize one typed image-metadata policy.

    ``None`` and the literal ``none`` are the explicit metadata-off policy;
    blank text selects automatic discovery.
    """

    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return "auto"
    if normalized == "none":
        return None
    return normalized


def normalize_image_source_metadata(source: SourceSpec) -> SourceSpec:
    """Materialize the automatic metadata policy on image source intent.

    Older callers could construct a TIFF ``SourceSpec`` without the option,
    while Controls projected that absence as Auto.  The run-intent boundary
    uses this helper so the displayed value and the executable source remain
    one typed fact before an operator edits anything.
    """

    if type(source) is not SourceSpec:
        raise TypeError("source must be SourceSpec")
    try:
        kind = coerce_source_kind(source.kind)
    except (TypeError, ValueError):
        return source
    options = dict(source.options)
    if (
        kind not in {
            SourceKind.IMAGE_FILE,
            SourceKind.TIFF_SERIES,
            SourceKind.NEXUS_STACK,
            SourceKind.EIGER_MASTER,
            SourceKind.PROCESSED_NEXUS,
        }
        or "metadata_format" in options
    ):
        return source
    options["metadata_format"] = "auto"
    return SourceSpec(
        source.uri,
        source.kind,
        metadata_uri=source.metadata_uri,
        entry=source.entry,
        options=options,
    )


@dataclass(frozen=True, slots=True)
class DirectorySourceSpec:
    """Immutable Image-Directory intent with no discovered file contents."""

    root: Path
    recursive: bool = False
    suffixes: tuple[str, ...] = ()
    name_filter: str | None = None
    generation: int = 0
    metadata_format: str | None = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())
        object.__setattr__(
            self,
            "suffixes",
            tuple(str(value).lower() for value in self.suffixes if value),
        )
        object.__setattr__(
            self,
            "name_filter",
            str(self.name_filter) if self.name_filter else None,
        )
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(
            self,
            "metadata_format",
            normalize_metadata_format(self.metadata_format),
        )


def single_image_spec(
    selected_file: str | Path,
    *,
    metadata_format: str | None = "auto",
) -> SourceSpec:
    """Freeze exactly one selected detector image for execution.

    Scattering execution already gives TIFF-series intent its strict source
    validation and per-frame metadata contract.  A single image therefore uses
    that same executable kind with one exact member, plus an explicit persisted
    selection-mode marker.  The marker, rather than the number of members,
    distinguishes this intent from an intentionally one-frame image series.
    """

    selected = Path(selected_file).expanduser()
    if selected.suffix.casefold() not in _SINGLE_IMAGE_SUFFIXES:
        raise ValueError(
            "Single Image requires one detector image file; "
            "select HDF5/NeXus containers through Image Series."
        )
    return SourceSpec(
        selected.parent,
        SourceKind.TIFF_SERIES,
        options={
            "selected_file": str(selected),
            "files": (str(selected),),
            "pattern": selected.name,
            "scan_name": selected.stem,
            "metadata_format": normalize_metadata_format(metadata_format),
            "selection_mode": _SINGLE_IMAGE_SELECTION_MODE,
        },
    )


def is_single_image_spec(source: object) -> bool:
    """Return whether *source* carries the exact single-image intent marker."""

    if type(source) is not SourceSpec:
        return False
    try:
        kind = coerce_source_kind(source.kind)
    except (TypeError, ValueError):
        return False
    if kind is not SourceKind.TIFF_SERIES:
        return False
    options = dict(source.options)
    files = options.get("files")
    selected = options.get("selected_file")
    return (
        options.get("selection_mode") == _SINGLE_IMAGE_SELECTION_MODE
        and type(files) is tuple
        and len(files) == 1
        and type(files[0]) is str
        and bool(files[0])
        and type(selected) is str
        and selected == files[0]
    )


def image_series_spec(
    selected_file: str | Path,
    *,
    metadata_format: str | None = "auto",
) -> SourceSpec:
    """Freeze the complete numbered series containing *selected_file*.

    A NeXus/HDF5 container already owns its complete frame series, so preserve
    it as one explicit container source.  Its content-derived exact kind
    (ordinary stack versus Eiger master) is qualified by the admission probe;
    treating it as a tuple of TIFF members loses both adapter ownership and
    the container's internal frame count.

    The selected member supplies its own suffix and parsed scan stem.  Hidden
    fields from another source mode therefore cannot affect membership, and
    selecting frame 4 does not silently redefine the series as frames 4 onward.
    Enumeration is name-only; no image payload is opened.
    """
    selected = Path(selected_file).expanduser()
    metadata_policy = normalize_metadata_format(metadata_format)
    if selected.suffix.lower() in {".h5", ".hdf5", ".nxs", ".cxi"}:
        return SourceSpec(
            selected,
            SourceKind.NEXUS_STACK,
            options={"metadata_format": metadata_policy},
        )
    match = _FRAME_SUFFIX.match(selected.stem)
    suffix = selected.suffix
    if match is None or not suffix:
        files = (selected,)
        scan_name = selected.stem
        pattern = selected.name
    else:
        scan_name = match.group(1)
        candidate_re = re.compile(
            rf"^{re.escape(scan_name)}[_-](\d+){re.escape(suffix)}$",
            re.IGNORECASE,
        )
        members: list[tuple[int, str, Path]] = []
        try:
            siblings = selected.parent.iterdir()
        except OSError:
            siblings = ()
        for sibling in siblings:
            if not sibling.is_file():
                continue
            sibling_match = candidate_re.match(sibling.name)
            if sibling_match is not None:
                members.append((
                    int(sibling_match.group(1)),
                    sibling.name.casefold(),
                    sibling,
                ))
        files = tuple(
            member[2] for member in sorted(members, key=lambda item: item[:2]))
        if not files and selected.is_file():
            files = (selected,)
        pattern = f"{scan_name}_*{suffix}"

    return SourceSpec(
        selected.parent,
        SourceKind.TIFF_SERIES,
        options={
            "selected_file": str(selected),
            "files": tuple(str(path) for path in files),
            "pattern": pattern,
            "scan_name": scan_name,
            "metadata_format": metadata_policy,
        },
    )


__all__ = [
    "DirectorySourceSpec",
    "image_series_spec",
    "is_single_image_spec",
    "normalize_image_source_metadata",
    "normalize_metadata_format",
    "single_image_spec",
]
