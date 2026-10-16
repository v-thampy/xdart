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
    Enumeration is name-only: only a sibling whose name matches the series is
    checked for being a file, and no image payload is opened.
    """
    selected = Path(selected_file).expanduser()
    metadata_policy = normalize_metadata_format(metadata_format)
    if selected.suffix.lower() in {
        ".h5", ".hdf5", ".nxs", ".nexus", ".cxi",
    }:
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
            sibling_match = candidate_re.match(sibling.name)
            if sibling_match is None or not sibling.is_file():
                continue
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


#: A Stitch or RSM page's one Data format (LT-SMALL-FIXES-20261014; maintainer, October 14: each page remembers its
#: own). It drives that page's Source rows and its calibration window's calibrant and direct-beam sources; Int keeps
#: its automatic file-type handling and has none. SPEC + Eiger keeps the calibration window's "SPEC + Eiger / NeXus"
#: meaning: SPEC metadata with HDF5 image containers (Eiger masters or NeXus image stacks).
DATA_FORMATS = ("spec_series", "spec_container", "nexus")
DATA_FORMAT_LABELS = {"spec_series": "SPEC + Image Series", "spec_container": "SPEC + Eiger", "nexus": "NeXus"}
DEFAULT_DATA_FORMAT = "spec_series"
#: How a page picks its Source: files one by one, or a folder whose files are found (its Subdirs and Filter).
INPUT_MODES = ("files", "directory")
INPUT_MODE_LABELS = {"files": "Files", "directory": "Folder"}
#: The two choices' purposes, as both pages show them.
DATA_FORMAT_PURPOSE = ("This page's data: SPEC metadata with an image series, SPEC with Eiger or other HDF5 image "
                       "files, or NeXus files holding both. It sets the Source rows and file choosers, and the "
                       "calibration window's calibrant and direct-beam sources. The other page keeps its own.")
INPUT_MODE_PURPOSE = ("Files: choose the SPEC or NeXus files themselves. Folder: choose one folder; every SPEC file "
                      "and NeXus or Eiger file found in it (and with Subdirs, its subfolders) is used, each scan or "
                      "file its own output, whatever the Data format.")
#: Each format's calibration-window source kinds: (calibrant, direct beam).
_ROLE_KINDS = {"spec_series": ("spec", "spec_series"), "spec_container": ("spec_container", "spec_container"),
               "nexus": ("nexus_stack", "nexus")}
#: A saved calibration-window kind's format family; the defaults ("spec", "spec_series") name SPEC, not its images.
_SAVED_ROLE_KIND = {"spec": "spec", "spec_series": "spec", "spec_container": "spec_container",
                    "nexus_stack": "nexus", "nexus": "nexus"}


def page_source_kind(data_format: str, input_mode: str) -> str:
    """The page's source-selection kind, in the unchanged tokens its saved identity keeps: a folder is Directory
    whatever the format (its discovery stays format-agnostic); files are NeXus or Spec."""
    if data_format not in DATA_FORMATS or input_mode not in INPUT_MODES:
        raise ValueError(f"unknown Data format or input: {data_format!r}, {input_mode!r}")
    if input_mode == "directory":
        return "Directory"
    return "NeXus" if data_format == "nexus" else "Spec"


def calibration_source_kinds(data_format: str) -> tuple[str, str]:
    """The calibration window's (calibrant, direct-beam) source kinds for a page's Data format."""
    return _ROLE_KINDS[data_format]


def legacy_data_format(page_kind: str, roles=()) -> tuple[str, str, str]:
    """An older page document's (Data format, input mode, conflict) from its Source kind and its calibration
    window's saved source kinds (D2, maintainer, October 14: a deterministic default).

    ``roles`` are (role, saved kind, whether that role names an input) triples, e.g. ("the calibrant scan",
    "nexus_stack", True); a role naming no input says nothing. A NeXus Source or role is NeXus; a SPEC + Eiger role is
    SPEC + Eiger; a Spec Source or a role at the SPEC default only says SPEC, and nothing at all is SPEC + Image
    Series. A conflict (NeXus beside SPEC) keeps the Source's own format and names the roles, for the user to choose:
    one format now serves them all."""
    mode = "directory" if page_kind == "Directory" else "files"
    families = []  # (who, family) the document names
    if page_kind == "NeXus":
        families.append(("the Source", "nexus"))
    elif page_kind == "Spec":
        families.append(("the Source", "spec"))
    for role, kind, named in roles:
        family = _SAVED_ROLE_KIND.get(kind) if named else None
        if family is not None:
            families.append((role, family))
    nexus = [who for who, family in families if family == "nexus"]
    spec = [who for who, family in families if family != "nexus"]
    container = any(family == "spec_container" for _who, family in families)
    if nexus and spec:
        own = "nexus" if page_kind == "NeXus" else "spec_container" if container else DEFAULT_DATA_FORMAT
        def read(who):
            return ", ".join(who) + (" reads" if len(who) == 1 else " read")
        return own, mode, (f"Saved settings disagree: {read(nexus)} NeXus, {read(spec)} SPEC. Choose this page's "
                           "Data format; it now serves all of them.")
    if nexus:
        return "nexus", mode, ""
    return ("spec_container" if container else DEFAULT_DATA_FORMAT), mode, ""


__all__ = [
    "DATA_FORMATS",
    "DATA_FORMAT_LABELS",
    "DATA_FORMAT_PURPOSE",
    "INPUT_MODE_PURPOSE",
    "DEFAULT_DATA_FORMAT",
    "DirectorySourceSpec",
    "INPUT_MODES",
    "INPUT_MODE_LABELS",
    "calibration_source_kinds",
    "image_series_spec",
    "is_single_image_spec",
    "legacy_data_format",
    "normalize_image_source_metadata",
    "normalize_metadata_format",
    "page_source_kind",
    "single_image_spec",
]
