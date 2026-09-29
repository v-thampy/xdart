"""Finite Stitch source selections, resolved by the existing format owners."""

from dataclasses import dataclass
from pathlib import Path

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.nexus import list_entries
from xrd_tools.io.spec import list_spec_scans
from xrd_tools.sources.descriptor import describe_container
from xrd_tools.sources.discover import discover_scans
from xrd_tools.sources.grouping import parse_stitch_scan_groups
from xrd_tools.sources.registry import open_source


@dataclass(frozen=True, slots=True)
class StitchSourceSelection:
    kind: str
    paths: tuple[str, ...]
    scans: str = ""
    stitch_all: bool = False
    image_dir: str = ""
    image_stem: str = ""

    def __post_init__(self):
        if self.kind not in {"Spec", "NeXus", "Directory"} or type(self.paths) is not tuple or not self.paths:
            raise ValueError("Choose Spec, NeXus or Directory and its source path")
        if len(self.paths) > 256 or any(type(path) is not str or not path.strip() or "\0" in path for path in self.paths):
            raise ValueError("Source selection must contain at most 256 valid paths")
        object.__setattr__(self, "paths", tuple(str(Path(path).expanduser().absolute()) for path in self.paths))
        if self.kind == "Directory" and len(self.paths) != 1:
            raise ValueError("Directory mode requires one folder")
        if self.kind == "Spec" and not self.stitch_all:
            parse_stitch_scan_groups(self.scans)
        if "\0" in self.image_dir or any(c in self.image_stem for c in ("/", "\\", "\0")):
            raise ValueError("Image directory or stem is invalid")

    @property
    def fingerprint_value(self):
        return self.kind, self.paths, self.scans, self.stitch_all, self.image_dir, self.image_stem


@dataclass(frozen=True, slots=True)
class StitchSourceResolution:
    groups: tuple[tuple[SourceSpec, ...], ...]
    skipped: tuple[tuple[str, str, str], ...]
    #: each group's members' scan names, as Int names them (``int_scan_name``)
    names: tuple[tuple[str, ...], ...] = ()


def int_scan_name(path) -> str:
    """The scan name Int gives a raw file (its source adapter's): an image series' stem without its
    frame number (``b_thampy_sample_scan14_0000.raw`` gives ``b_thampy_sample_scan14``), a NeXus
    container's full stem."""
    from xrd_tools.sources.adapters import candidate_owner
    path = Path(path)
    adapter = candidate_owner(path)
    return adapter.scan_name(path) if adapter is not None else path.stem


def spec_scan_name(path, key, image_dir, image_stem) -> str | None:
    """Int's scan name for one SPEC scan's images (its first image's), or None when none match: the
    automatic result name before Preview, from the files the source would read."""
    from xrd_tools.io.image import find_image_files
    stem = image_stem or f"{Path(path).stem}_scan{str(key).split('.')[0]}_"
    files = list(find_image_files(Path(image_dir or Path(path).parent), stem=stem))
    return int_scan_name(files[0]) if files else None


def resolve_stitch_sources(selection: StitchSourceSelection, read_image_kwargs) -> StitchSourceResolution:
    """Freeze discovery and skips once at Preview; never rediscover during Run."""
    if type(selection) is not StitchSourceSelection:
        raise TypeError("exact Stitch source selection required")
    groups, skipped, names = [], [], {}

    def spec_member(path, key):
        stem = selection.image_stem.format(spec=path.stem, scan=key.split(".")[0], occurrence=key)
        spec = SourceSpec(path, SourceKind.SPEC, options={
            "scan": key, "image_dir": selection.image_dir or str(path.parent),
            "image_stem": stem or None, "read_image_kwargs": dict(read_image_kwargs),
            "require_complete_images": True,
        })
        source = open_source(spec)
        try:
            if not source.capabilities.has_raw_references:
                folder = selection.image_dir or str(path.parent)
                searched = stem or f"{path.stem}_scan{key.split('.')[0]}_"
                skipped.append((str(path), key, f"No detector images named *{searched}* in {folder}"))
                return None
            names[id(spec)] = int_scan_name(source.raw_locator_for(0)[0])  # as Int names these images
            return spec
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()

    def spec_file(path, all_scans):
        keys = tuple(list_spec_scans(path))
        requested = tuple((key,) for key in keys) if all_scans else parse_stitch_scan_groups(selection.scans)
        for group in requested:
            members = []
            for token in group:
                key = token if token in keys else token + ".1"
                if key not in keys:
                    raise ValueError(f"{path.name}: scan {token} is not present")
                value = spec_member(path, key)
                if value is not None and value not in members:
                    members.append(value)
            if members:
                groups.append(tuple(members))

    def nexus_members(path):
        kind = describe_container(path).kind
        if kind not in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
            raise ValueError(f"{path.name} is not a raw NeXus detector source")
        entries = tuple(list_entries(path))
        if not entries:
            raise ValueError(f"{path.name} has no NeXus entries")
        # Each file remains one output/group; its exact entries contribute.
        members = tuple(SourceSpec(path, kind, entry=entry) for entry in entries)
        names.update(dict.fromkeys(map(id, members), int_scan_name(path)))
        return members

    if selection.kind == "Spec":
        for path in selection.paths:
            spec_file(Path(path), selection.stitch_all)
    elif selection.kind == "NeXus":
        groups.append(tuple(member for path in selection.paths for member in nexus_members(Path(path))))
    else:
        directory = Path(selection.paths[0])
        if not directory.is_dir():
            raise ValueError("Source directory is unavailable")
        specs = discover_scans(directory, SourceKind.SPEC)
        for path in dict.fromkeys(str(spec.uri) for spec in specs):
            spec_file(Path(path), True)
        seen_nexus = set()
        for kind in (SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER):
            for spec in discover_scans(directory, kind):
                path = Path(spec.uri)
                resolved = path.resolve()
                if resolved not in seen_nexus:
                    seen_nexus.add(resolved)
                    groups.append(nexus_members(path))
    if len(groups) > 4096:
        raise ValueError("Selection exceeds 4096 result groups")
    return StitchSourceResolution(tuple(groups), tuple(skipped),
                                  tuple(tuple(names[id(spec)] for spec in group) for group in groups))
