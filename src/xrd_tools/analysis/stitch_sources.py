"""Finite Stitch source selections, resolved by the existing format owners."""

from dataclasses import dataclass
import functools
from pathlib import Path

from xrd_tools.core.filters import compile_filter
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.nexus import list_entries
from xrd_tools.io.spec import list_spec_scans
from xrd_tools.sources.descriptor import describe_container
from xrd_tools.sources.discover import discover_scans
from xrd_tools.sources.grouping import parse_stitch_scan_groups
from xrd_tools.sources.registry import open_source
from xrd_tools.sources.point_images import PointImagePolicy, PointImagesRequired, inspect_point_images


@dataclass(frozen=True, slots=True)
class StitchSourceSelection:
    kind: str
    paths: tuple[str, ...]
    scans: str = ""
    stitch_all: bool = False
    image_dir: str = ""
    image_stem: str = ""
    #: Directory mode, as Int's Image Directory (maintainer, 2026-09-30): its subfolders too, and a
    #: file-name filter in the shared grammar (``xrd_tools.core.filters``: space = and, ``|`` = or,
    #: ``-term`` = not; case-insensitive substrings of the name without its extension)
    recursive: bool = False
    name_filter: str = ""
    point_image_choices: tuple[tuple[str, str, PointImagePolicy], ...] = ()
    #: Blank retains discovery of all entries; an explicit page override is exact.
    nexus_entry: str = ""

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
        if type(self.nexus_entry) is not str or "\0" in self.nexus_entry:
            raise ValueError("NeXus entry must be a group name")
        object.__setattr__(self, "nexus_entry", self.nexus_entry.strip())
        if "\0" in self.image_dir or any(c in self.image_stem for c in ("/", "\\", "\0")):
            raise ValueError("Image directory or stem is invalid")
        if type(self.recursive) is not bool or type(self.name_filter) is not str:
            raise ValueError("Directory subfolders and filter are invalid")
        try:
            compile_filter(self.name_filter)
        except ValueError as error:
            raise ValueError(f"Filter: {error}") from None
        if (type(self.point_image_choices) is not tuple or any(
                type(row) is not tuple or len(row) != 3 or type(row[0]) is not str
                or type(row[1]) is not str or type(row[2]) is not PointImagePolicy
                for row in self.point_image_choices)):
            raise ValueError("Point image choices must be exact confirmed source policies")

    @property
    def fingerprint_value(self):
        value = (self.kind, self.paths, self.scans, self.stitch_all, self.image_dir, self.image_stem)
        # the Directory options only when set: a selection without them keeps its identity
        return value + (("recursive",) if self.recursive else ()) + (
            ("filter", self.name_filter) if self.name_filter else ()) + (
            ("point_images", tuple((uri, scan, tuple(policy.to_dict().items()))
                                  for uri, scan, policy in self.point_image_choices))
            if self.point_image_choices else ()) + (
            ("nexus_entry", self.nexus_entry) if self.nexus_entry else ())


#: Files a Directory's SPEC search never opens: detector images, their notes, containers, results.
_NOT_SPEC = frozenset({".raw", ".tif", ".tiff", ".edf", ".cbf", ".img", ".mar3450", ".pdi", ".metadata",
                       ".npy", ".npz", ".png", ".jpg", ".jpeg", ".h5", ".hdf5", ".nxs", ".nexus", ".cxi",
                       ".poni", ".npt", ".json", ".csv"})
#: A NeXus container's suffixes, which Int's filter leaves out of the name it matches.
_CONTAINER_SUFFIXES = frozenset({".nxs", ".h5", ".hdf5", ".nexus", ".cxi"})


def _directory_spec_files(directory, recursive, name_ok):
    """The SPEC files in a Directory (its subfolders' too) whose names pass the filter, in discovery
    order. A known image or data file is never opened to find out: with subfolders, an images folder
    of thousands of frames would otherwise each be read."""
    from xrd_tools.io.spec import is_spec_file
    from xrd_tools.sources.discover import _walk_files
    return [path for path in _walk_files(directory, recursive)
            if path.suffix.lower() not in _NOT_SPEC and name_ok(path.name) and is_spec_file(path)]


@dataclass(frozen=True, slots=True)
class StitchSourceResolution:
    groups: tuple[tuple[SourceSpec, ...], ...]
    skipped: tuple[tuple[str, str, str], ...]
    #: each group's members' scan names, as Int names them (``int_scan_name``)
    names: tuple[tuple[str, ...], ...] = ()
    pending: tuple = ()


@functools.lru_cache(maxsize=8)
def _spec_frame_counts(path, _state, scans):
    """Cheap SPEC point counts, invalidated when the source file changes."""
    from silx.io.specfile import SfError
    from xrd_tools.io.spec import count_spec_scan_points
    try:
        return count_spec_scan_points(path, scans)
    except (OSError, ValueError, LookupError, SfError):
        return None


def _output_frames(members, counts):
    names = [scan[:-2] if scan.endswith(".1") else scan for scan in members]
    frames = sum(counts[scan] for scan in members)
    if len(members) == 1:
        return f"scan {names[0]}: {frames} frame{'s' * (frames != 1)}"
    return (f"scans {', '.join(names)} (one output): "
            f"{' + '.join(str(counts[scan]) for scan in members)} = {frames} frames")


def stitch_selection_frames(selection: StitchSourceSelection):
    """Selection's point/frame count and per-scan lines, without decoding images.

    SPEC counts remain metadata points until strict source admission confirms the images.
    An unknown selection returns None; Directory discovery stays on its existing worker.
    """
    if selection.kind == "Directory":
        return None
    if selection.kind == "NeXus":
        try:
            if selection.nexus_entry and any(selection.nexus_entry not in list_entries(path)
                                             for path in selection.paths):
                return None
            counts = tuple((Path(path).name, describe_container(
                path, entry=selection.nexus_entry or "entry").frame_count) for path in selection.paths)
        except (OSError, ValueError, LookupError):
            return None
        return sum(count for _name, count in counts), tuple(f"{name}: {count} frames" for name, count in counts)
    try:
        groups = None if selection.stitch_all else parse_stitch_scan_groups(selection.scans)
    except ValueError:
        return None
    wanted = None if groups is None else tuple(sorted({scan for group in groups for scan in group}))
    total, lines = 0, []
    for path in selection.paths:
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        counts = _spec_frame_counts(path, (stat.st_mtime_ns, stat.st_size), wanted)
        if counts is None or None in counts.values():
            return None
        lines += [f"{Path(path).name}:"] * (len(selection.paths) > 1)
        for group in ([(scan,) for scan in counts] if groups is None else groups):
            members = tuple(dict.fromkeys(group))
            total += sum(counts[scan] for scan in members)
            lines.append(("  " if len(selection.paths) > 1 else "") + _output_frames(members, counts))
    return total, tuple(lines)


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
    from xrd_tools.sources.spec import find_spec_image_files
    files = find_spec_image_files(Path(image_dir or Path(path).parent), Path(path).stem, key, image_stem)
    return int_scan_name(files[0]) if files else None


def resolve_stitch_sources(selection: StitchSourceSelection, read_image_kwargs) -> StitchSourceResolution:
    """Freeze discovery and skips once at Preview; never rediscover during Run."""
    if type(selection) is not StitchSourceSelection:
        raise TypeError("exact Stitch source selection required")
    groups, skipped, names, pending = [], [], {}, []
    choices = {(uri, scan): policy for uri, scan, policy in selection.point_image_choices}

    def confirmed(spec, *, directory=False):
        policy = choices.get((str(spec.uri), str(spec.options.get("scan") or spec.entry or "entry")))
        detection = (inspect_point_images(spec) if policy is not None or spec.kind is not SourceKind.SPEC else None)
        if (detection is not None and spec.kind is SourceKind.EIGER_MASTER
                and describe_container(Path(spec.uri), entry=spec.entry or "entry").motor_names == ()):
            # Maintainer, 2026-10-10: "A bare Eiger master will never be the only source in RSM and
            # stitching. Though it can be in int of course with a poni file." A burst master's images
            # are not one scan point's, so it is refused (skipped in Directory mode) before any grouping
            # question or confirmed grouping. A master without a burst keeps Stitch's preflight refusal.
            reason = (f"a detector-only Eiger master ({detection.image_count} images for {detection.point_count} "
                      "trigger(s)): Stitch and RSM read its scan from its metadata source (SPEC or NeXus); "
                      "select that source instead")
            if directory:
                skipped.append((str(spec.uri), str(spec.entry or "entry"), reason))
                return None
            raise ValueError(f"{Path(spec.uri).name} is {reason}")
        if detection is not None and (policy is None or policy.images_per_point != detection.images_per_point):
            pending.append(detection)
            return None
        if policy is not None:
            from dataclasses import replace
            spec = replace(spec, options={**spec.options, "point_images": policy.to_dict()})
        return spec

    def spec_member(path, key):
        stem = selection.image_stem.format(spec=path.stem, scan=key.split(".")[0], occurrence=key)
        spec = SourceSpec(path, SourceKind.SPEC, options={
            "scan": key, "image_dir": selection.image_dir or str(path.parent),
            "image_stem": stem or None, "read_image_kwargs": dict(read_image_kwargs),
            "require_complete_images": True,
            **({"nexus_entry": selection.nexus_entry} if selection.nexus_entry else {}),
        })
        spec = confirmed(spec)
        if spec is None:
            return None
        try:
            source = open_source(spec)
        except PointImagesRequired as error:
            pending.append(error.detection)
            return None
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

    def nexus_members(path, *, directory=False):
        if selection.nexus_entry and selection.nexus_entry not in list_entries(path):
            raise ValueError(f"{path.name}: NeXus entry {selection.nexus_entry!r} is not present")
        kind = describe_container(path, entry=selection.nexus_entry or "entry").kind
        if kind not in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
            raise ValueError(f"{path.name} is not a raw NeXus detector source")
        entries = (selection.nexus_entry,) if selection.nexus_entry else tuple(list_entries(path))
        if not entries:
            raise ValueError(f"{path.name} has no NeXus entries")
        # Each file remains one output/group; its exact entries contribute.
        members = tuple(value for entry in entries
                        if (value := confirmed(SourceSpec(path, kind, entry=entry), directory=directory))
                        is not None)
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
        name_ok = compile_filter(selection.name_filter)
        for path in _directory_spec_files(directory, selection.recursive, name_ok):
            spec_file(path, True)
        seen_nexus = set()
        for kind in (SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER):
            for spec in discover_scans(directory, kind, recursive=selection.recursive):
                path = Path(spec.uri)
                if not name_ok(path.stem if path.suffix.lower() in _CONTAINER_SUFFIXES else path.name):
                    continue
                resolved = path.resolve()
                if resolved not in seen_nexus:
                    seen_nexus.add(resolved)
                    groups.append(nexus_members(path, directory=True))
    if len(groups) > 4096:
        raise ValueError("Selection exceeds 4096 result groups")
    if pending:
        # A grouped output never silently loses a member awaiting confirmation.
        return StitchSourceResolution((), tuple(skipped), (), tuple(pending))
    return StitchSourceResolution(tuple(group for group in groups if group), tuple(skipped),
                                  tuple(tuple(names[id(spec)] for spec in group) for group in groups if group))
