"""Finite Stitch source selections, resolved by the existing format owners."""

from dataclasses import dataclass
import functools
from pathlib import Path

from xrd_tools.core.filters import compile_filter
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.nexus import list_entries
from xrd_tools.io.spec import UnusableSpecScan, list_spec_scans
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


@dataclass(frozen=True, slots=True)
class StitchSourceResolution:
    groups: tuple[tuple[SourceSpec, ...], ...]
    skipped: tuple[tuple[str, str, str], ...]
    #: each group's members' scan names, as Int names them (``int_scan_name``)
    names: tuple[tuple[str, ...], ...] = ()
    pending: tuple = ()


@dataclass(frozen=True, slots=True)
class PlannedStitchMember:
    path: Path
    key: str
    kind: SourceKind


@dataclass(frozen=True, slots=True)
class PlannedStitchGroup:
    identity: tuple
    members: tuple[PlannedStitchMember, ...]
    paths: tuple[Path, ...]
    every_scan: bool = False
    # Keep finite progress and absent-selector diagnostics from the requested tokens,
    # including a second search position for two spellings of the same occurrence.
    spec_searches: tuple[tuple[str, bool], ...] = ()
    spec_total: int = 0


def _spec_key(token):
    return token if "." in token else token + ".1"


def list_planned_stitch_groups(selection, *, check=lambda: None, allow_pending=False,
                               spec_keys=None, nexus_listing=None, directory_inventory=None):
    """Enumerate ordered output groups without opening their detector sources."""
    from xrd_tools.io.spec import is_spec_file
    from xrd_tools.sources.discover import _walk_files

    if type(selection) is not StitchSourceSelection:
        raise TypeError("exact Stitch source selection required")
    get_keys = list_spec_scans if spec_keys is None else spec_keys
    selection_groups = (parse_stitch_scan_groups(selection.scans)
                        if selection.kind == "Spec" and not selection.stitch_all else ())
    spec_total = 0

    def spec_groups(path, lane, every_scan):
        nonlocal spec_total
        check()
        keys = tuple(get_keys(path))
        requested = tuple((key,) for key in keys) if every_scan else selection_groups
        spec_total += sum(len(group) for group in requested)
        present = set(keys)
        for position, tokens in enumerate(requested):
            check()
            canonical = tuple(dict.fromkeys(_spec_key(token) for token in tokens))
            identity = (*lane, canonical[0] if every_scan else position)
            members = tuple(PlannedStitchMember(path, key, SourceKind.SPEC) for key in canonical)
            searches = tuple((token, _spec_key(token) in present) for token in tokens)
            yield PlannedStitchGroup(identity, members, (path,), every_scan, searches, spec_total)

    def nexus_members(path):
        check()
        if nexus_listing is None:
            entries = tuple(list_entries(path))
            kind = describe_container(path, entry=selection.nexus_entry or "entry").kind
        else:
            kind, entries = nexus_listing(path)
        entry = selection.nexus_entry
        if entry and entry not in entries:
            if not allow_pending:
                raise ValueError(f"{path.name}: NeXus entry {entry!r} is not present")
            entries = ()
        elif entry:
            entries = (entry,)
        if kind not in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
            if not allow_pending:
                raise ValueError(f"{path.name} is not a raw NeXus detector source")
            return ()
        if not entries and not allow_pending:
            raise ValueError(f"{path.name} has no NeXus entries")
        return tuple(PlannedStitchMember(path, key, kind) for key in entries)

    paths = tuple(Path(path) for path in selection.paths)
    if selection.kind == "Spec":
        for position, path in enumerate(paths):
            yield from spec_groups(path, ("spec", position, path), selection.stitch_all)
    elif selection.kind == "NeXus":
        members = tuple(member for path in paths for member in nexus_members(path))
        yield PlannedStitchGroup(("nexus", *paths), members, paths)
    else:
        directory = paths[0]
        if not directory.is_dir():
            raise ValueError("Source directory is unavailable")
        name_ok = compile_filter(selection.name_filter)
        if directory_inventory is None:
            files = _walk_files(directory, selection.recursive, check=check)
            spec_files = (path for path in files if path.suffix.lower() not in _NOT_SPEC
                          and name_ok(path.name) and is_spec_file(path))
            containers = (Path(spec.uri)
                          for kind in (SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER)
                          for spec in discover_scans(directory, kind, recursive=selection.recursive,
                                                     check=check, _files=files)
                          if name_ok(Path(spec.uri).stem if Path(spec.uri).suffix.lower()
                                     in _CONTAINER_SUFFIXES else Path(spec.uri).name))
        else:
            spec_files, containers = directory_inventory(directory, selection.recursive, name_ok, check)
        for path in spec_files:
            yield from spec_groups(path, ("directory-spec", path), True)
        seen = set()
        for path in containers:
            check()
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield PlannedStitchGroup(("directory-nexus", path), nexus_members(path), (path,))


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


class SourceSearchCancelled(InterruptedError):
    """A search of a Source's scans stopped between scans (NEXT-1210-RESPONSIVENESS C-16)."""


def resolve_stitch_sources(selection: StitchSourceSelection, read_image_kwargs, *, cancelled=None,
                           progress=None, planned_group=None) -> StitchSourceResolution:
    """Freeze discovery and skips once at Preview; never rediscover during Run.

    C-16: ``cancelled()`` is checked before each scan (``SourceSearchCancelled``); ``progress(done, total)`` follows the
    scans searched; each image folder is listed once for the whole search (``shared_image_listings``)."""
    from xrd_tools.io.image import shared_image_listings
    from xrd_tools.io.spec import spec_capture_scope
    with shared_image_listings(), spec_capture_scope():  # and each SPEC file read once (NEXT-1219-SPEC-OCCURRENCE)
        return _resolve_stitch_sources(selection, read_image_kwargs, cancelled, progress, planned_group)


def _resolve_stitch_sources(selection, read_image_kwargs, cancelled, progress, planned_group=None):
    if type(selection) is not StitchSourceSelection:
        raise TypeError("exact Stitch source selection required")
    groups, skipped, names, pending = [], [], {}, {}
    searched = [0, 0]  # scans searched, scans to search (a Directory's grows file by file)

    def step():
        if cancelled is not None and cancelled():
            raise SourceSearchCancelled("the scan search was stopped")

    def searched_one():
        searched[0] += 1
        if progress is not None:
            progress(searched[0], max(searched))
    choices = {(uri, scan): policy for uri, scan, policy in selection.point_image_choices}

    def choice_key(spec):
        return str(spec.uri), str(spec.options.get("scan") or spec.entry or "entry")

    def confirmed(spec, *, directory=False):
        policy = choices.get(choice_key(spec))
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
        if detection is None and policy is not None and spec.kind is SourceKind.SPEC:
            # C-13: a grouping stored while the scan recorded no longer fits once it reads one image per point (or an
            # unfinished count); the scan is read as it is now, its stored grouping set aside.
            policy = None
        if detection is not None and (policy is None or policy.images_per_point != detection.images_per_point):
            pending.setdefault(choice_key(spec), detection)
            return None
        if policy is not None:
            from dataclasses import replace
            spec = replace(spec, options={**spec.options, "point_images": policy.to_dict()})
        return spec

    def spec_member(path, key, every_scan):
        stem = selection.image_stem.format(spec=path.stem, scan=key.split(".")[0], occurrence=key)
        spec = SourceSpec(path, SourceKind.SPEC, options={
            "scan": key, "image_dir": selection.image_dir or str(path.parent),
            "image_stem": stem or None, "read_image_kwargs": dict(read_image_kwargs),
            "require_complete_images": True,
            **({"nexus_entry": selection.nexus_entry} if selection.nexus_entry else {}),
        })
        try:
            spec = confirmed(spec)
            if spec is None:
                return None
            source = open_source(spec)
        except PointImagesRequired as error:
            pending.setdefault(choice_key(error.detection.source_spec), error.detection)
            return None
        except UnusableSpecScan as error:
            # All Scans and Directory skip a scan they cannot grid and list why (C-10, the maintainer: "a lot of
            # scans in a spec file ... should be skipped"); an explicitly selected scan is still refused.
            if not every_scan:
                raise
            skipped.append((str(path), key, error.reason))
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

    def lower_group(group):
        members = []
        if group.spec_searches and planned_group is None:
            searched[1] = group.spec_total
            path = group.paths[0]
            for token, present in group.spec_searches:
                step()
                if not present:
                    raise ValueError(f"{path.name}: scan {token} is not present")
                value = spec_member(path, _spec_key(token), group.every_scan)
                searched_one()
                if value is not None and value not in members:
                    members.append(value)
            return tuple(members)
        for member in group.members:
            step()
            path, key = member.path, member.key
            if member.kind is SourceKind.SPEC:
                value = spec_member(path, key, False)
            else:
                value = confirmed(SourceSpec(path, member.kind, entry=key),
                                  directory=selection.kind == "Directory" and planned_group is None)
                if value is not None:
                    names[id(value)] = int_scan_name(path)
            if value is not None:
                members.append(value)
        return tuple(members)

    if planned_group is None:
        planned = list_planned_stitch_groups(selection, check=step)
    else:
        if type(planned_group) is not PlannedStitchGroup or not planned_group.members:
            raise ValueError("An exact planned group with members is required")
        planned = (planned_group,)
    for group in planned:
        group_sources = lower_group(group)
        if planned_group is not None and len(group_sources) != len(group.members):
            if not pending:
                raise ValueError("An exact planned group member is unavailable")
        if group_sources:
            groups.append(group_sources)
    step()
    if len(groups) > 4096:
        raise ValueError("Selection exceeds 4096 result groups")
    if pending:
        # A grouped output never silently loses a member awaiting confirmation.
        return StitchSourceResolution((), tuple(skipped), (), tuple(pending.values()))
    return StitchSourceResolution(tuple(group for group in groups if group), tuple(skipped),
                                  tuple(tuple(names[id(spec)] for spec in group) for group in groups if group))
