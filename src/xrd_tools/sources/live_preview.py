"""Detached source facts for the provisional native Live preview."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
import os
import re
import stat
import sys

import numpy as np
from silx.io.specfile import SfError

from xrd_tools.core.scan import SourceKind
from xrd_tools.io import descriptor_path, image, spec, stat_identity
from xrd_tools.io.metadata import _spec_descriptor_revision, _spec_name_matches_read
from xrd_tools.sources.point_images import PointImagePolicy
from xrd_tools.sources.registry import _cbf_header_incomplete_reason, _tiff_header_incomplete_reason
from xrd_tools.sources.spec import find_spec_image_files, spec_image_stem


_FLAT = frozenset((".raw", ".tif", ".tiff", ".cbf"))
_SLOT = re.compile(r"[_-]?([0-9]{1,18})\Z")
_DEFAULT_PLANE_LIMIT = 256 * 1024 * 1024
_HEADER_ALLOWANCE = 1 << 20


class PreviewPointPending(OSError):
    """A selected point could not be proved or read in this observation."""


@dataclass(frozen=True, slots=True)
class _RawPart:
    slot: int
    path: Path
    resolved: str
    revision: tuple
    descriptor_ctime: int | None


@dataclass(frozen=True, slots=True)
class PreviewPoint:
    """One logical point and the ordered detector inputs owned by its source."""

    label: int
    metadata: MappingProxyType
    columns: tuple[tuple[str, float], ...]
    raw: tuple[_RawPart, ...]
    policy: PointImagePolicy | None
    identity: tuple
    _shape: tuple[int, int]
    _dtype: str
    _skip: int
    _rotation: int
    _max_bytes: int


@dataclass(frozen=True, slots=True)
class PreviewMember:
    """An indexed occurrence and its available prefix, detached from all handles."""

    key: str
    path: Path
    ordinal: int = -1
    points: tuple[PreviewPoint, ...] = ()
    end_of_stream: bool = False
    reason: str = ""
    prefix: spec.SpecPreviewPrefix | None = None
    columns: tuple[str, ...] = ()
    motors: tuple[tuple[str, float], ...] = ()
    _image_dir: str = field(default="", repr=False)
    _image_stem: str | None = field(default=None, repr=False)
    _successor_required: bool = field(default=False, repr=False)


def _cancel(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Live preview source cancelled")


def _open_shared_read(path):
    if sys.platform == "win32":
        flags = (descriptor_path._FILE_SHARE_READ | descriptor_path._FILE_SHARE_WRITE
                 | descriptor_path._FILE_SHARE_DELETE)
        return descriptor_path._win32_open_existing(os.fspath(path), 0x80000000, flags, 0)
    return os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))


@contextmanager
def preview_source_scope():
    """One indexed capture per SPEC and a shared stem inventory for this sweep."""
    with spec.spec_capture_scope(opener=_open_shared_read), image.shared_image_listings(all_stems=True):
        yield


def _revision(status):
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns,
            stat_identity.identity_ctime_ns(status.st_ctime_ns))


def _raw_part(path, slot):
    resolved = os.path.normcase(os.path.realpath(path))
    status = path.stat()
    if not stat.S_ISREG(status.st_mode) or os.path.normcase(os.path.realpath(path)) != resolved:
        raise PreviewPointPending(f"{path.name} is not a stable detector file")
    descriptor_ctime = None
    if not stat_identity.IDENTITY_CARRIES_CTIME:
        descriptor = _open_shared_read(path)
        try:
            opened = os.fstat(descriptor)
            if (not _spec_name_matches_read(status, opened)
                    or not _spec_name_matches_read(path.stat(), opened)):
                raise PreviewPointPending(f"{path.name} changed while observing its detector slot")
            descriptor_ctime = opened.st_ctime_ns
        finally:
            os.close(descriptor)
    return _RawPart(slot, path, resolved, _revision(status), descriptor_ctime)


def _named_slots(path, key, directory, stem, cancelled):
    files = find_spec_image_files(directory, path.stem, key, stem)
    if any(file.suffix.casefold() not in _FLAT for file in files):
        raise PreviewPointPending("Container images are processed by per-scan Live")
    slots = {}
    for file in files:
        _cancel(cancelled)
        prefix = stem or spec_image_stem(file, path.stem, key)
        tail = file.stem.rpartition(prefix)[2]
        if (match := _SLOT.fullmatch(tail)) is None:
            continue
        index = int(match[1])
        slots[index] = None if index in slots else file
    return slots


def _observe_spec_member(member, selection, read_options, cancelled):
    from xrd_tools.analysis.live_scans import _planned_points
    from xrd_tools.analysis.stitch_sources import planned_member_source

    path, key = Path(member.path), member.key
    result = PreviewMember(key, path)
    if member.kind is not SourceKind.SPEC:
        return replace(result, reason="This source is processed by per-scan Live")
    source = planned_member_source(selection, read_options, member)
    candidate = spec.read_spec_preview_prefix(path, key)
    if candidate is None:
        keys = spec.list_spec_scans(path)
        return replace(result, ordinal=keys.index(key) if key in keys else -1,
                       reason=f"Waiting for complete SPEC rows in scan {key}")
    prefix, columns, motors = candidate
    options = source.options
    directory, stem = options["image_dir"], options["image_stem"]
    result = replace(result, ordinal=prefix.ordinal, prefix=prefix,
                     columns=tuple(name for name, _ in columns), motors=motors,
                     _image_dir=directory, _image_stem=stem)
    slots = _named_slots(path, key, directory, stem, cancelled)
    if slots.get(0) is None:
        return replace(result, reason=f"Waiting for the zero-index detector slot of scan {key}")
    number = key.partition(".")[0]
    if (any(other != key and other.partition(".")[0] == number for other in spec.list_spec_scans(path))
            and (not stem or key not in stem)):
        return replace(result, reason=f"Choose an image stem that identifies SPEC occurrence {key}")
    choice = next((policy for uri, scan, policy in selection.point_image_choices
                   if uri == str(source.uri) and scan == key), None)
    shape = tuple(read_options.get("detector_shape") or ())
    if len(shape) != 2 or not all(int(axis) > 0 for axis in shape):
        return replace(result, reason="Preview requires the configured detector dimensions")
    dtype = np.dtype(read_options.get("raw_dtype", "int32"))
    skip = int(read_options.get("raw_header_skip", 0))
    rotation = int(read_options.get("rotation", 0))
    max_bytes = int(read_options.get("max_frame_bytes", _DEFAULT_PLANE_LIMIT))
    width = 1 if choice is None else choice.images_per_point
    row_count = len(prefix.extent.marks) - 1
    command, successor = spec.spec_scan_completion_header(path, key)
    expected = _planned_points(command)
    by_points = expected is not None and row_count >= expected
    complete = prefix.complete and (by_points or successor)
    # Grouping a full set of rows needs the existing confirmed per-scan policy.
    # While the scan is open, an extra indexed file can belong to its next row.
    if (complete and choice is None and len(slots) > row_count and len(slots) % row_count == 0
            and all(slots.get(index) is not None for index in range(len(slots)))):
        return replace(result, reason=f"Confirm consecutive detector grouping for scan {key}")

    points = []
    reason = ""
    expected_raw = skip + shape[0] * shape[1] * dtype.itemsize
    for label in range(row_count):
        _cancel(cancelled)
        first = label * width
        parts = []
        try:
            for index in range(first, first + width):
                frame_path = slots.get(index)
                if frame_path is None:
                    raise PreviewPointPending(f"Waiting for detector slot {index} of scan {key}")
                part = _raw_part(frame_path, index)
                if frame_path.suffix.casefold() == ".raw" and part.revision[2] != expected_raw:
                    raise PreviewPointPending(f"Waiting for the exact RAW payload of {frame_path.name}")
                parts.append(part)
        except OSError as error:
            _cancel(cancelled)
            reason = str(error)
            break
        physical = tuple((name, float(values[label])) for name, values in columns)
        by_name = {}
        for name, value in physical:
            by_name.setdefault(name, value)
        raw = tuple(parts)
        identity = (prefix.identity, prefix.extent.marks[label + 1], raw, choice, shape, dtype.str, skip, rotation)
        points.append(PreviewPoint(label, MappingProxyType({**dict(motors), **by_name}), physical,
                                   raw, choice, identity, shape, dtype.str, skip, rotation, max_bytes))
    return replace(result, points=tuple(points), end_of_stream=complete and len(points) == row_count,
                   _successor_required=not by_points and successor,
                   reason=reason or ("" if complete else f"Watching SPEC scan {key}"))


def observe_spec_member(member, selection, read_options, *, cancelled=None):
    """Observe the gap-free per-point candidate prefix of one v3 planned member."""
    try:
        with preview_source_scope():
            return _observe_spec_member(member, selection, read_options, cancelled)
    except (OSError, ValueError, LookupError, SfError) as error:
        _cancel(cancelled)
        return PreviewMember(member.key, Path(member.path), reason=str(error))


def _read_flat_plane(part, point):
    suffix = part.path.suffix.casefold()
    with os.fdopen(_open_shared_read(part.path), "rb") as stream:
        descriptor = stream.fileno()
        before = os.fstat(descriptor)
        named = part.path.stat()
        if (not _spec_name_matches_read(named, before) or _revision(named) != part.revision
                or (part.descriptor_ctime is not None and before.st_ctime_ns != part.descriptor_ctime)
                or os.path.normcase(os.path.realpath(part.path)) != part.resolved):
            raise PreviewPointPending(f"{part.path.name} changed before decoding")
        overhead = point._skip if suffix == ".raw" else _HEADER_ALLOWANCE
        if before.st_size > point._max_bytes + overhead:
            raise PreviewPointPending(f"{part.path.name} exceeds the admitted detector frame bytes")
        saturation = None
        if suffix == ".raw":
            pixels = image._read_raw_binary(
                part.path, point._shape, point._dtype, point._skip, stream=stream)
        elif suffix == ".cbf":
            from fabio.cbfimage import CbfImage

            reason = _cbf_header_incomplete_reason(stream.read(min(before.st_size, _HEADER_ALLOWANCE)),
                                                   before.st_size)
            if reason is not None:
                raise PreviewPointPending(f"{part.path.name}: {reason}")
            stream.seek(0)
            # Fabio may close its stream. Give it a duplicate, never the lexical pathname.
            with os.fdopen(os.dup(descriptor), "rb") as codec_stream, CbfImage() as codec:
                codec.read(codec_stream)
                if codec.nframes != 1:
                    raise PreviewPointPending(f"{part.path.name} does not hold one exact detector plane")
                pixels = np.array(codec.data, copy=True)
                saturation = image.pilatus_count_cutoff(dict(codec.header))
        else:
            import tifffile

            try:
                reason = _tiff_header_incomplete_reason(stream, before.st_size)
            except (KeyError, OverflowError, ValueError):
                reason = None  # unrecognized TIFF structure still requires exact decode
            if isinstance(reason, str):
                raise PreviewPointPending(f"{part.path.name}: {reason}")
            stream.seek(0)
            with tifffile.TiffFile(stream, name=part.path.name, _multifile=False) as codec:
                first = codec.pages[0]
                try:
                    codec.pages[1]
                except IndexError:
                    pass
                else:
                    raise PreviewPointPending(f"{part.path.name} does not hold one exact detector plane")
                if first.shape != point._shape or first.dtype.itemsize * first.size > point._max_bytes:
                    raise PreviewPointPending(f"{part.path.name} does not match the admitted detector plane")
                pixels = first.asarray()
                saturation = image.pilatus_count_cutoff(first.description)
        after = os.fstat(descriptor)
        if (_spec_descriptor_revision(after) != _spec_descriptor_revision(before)
                or not _spec_name_matches_read(part.path.stat(), after)
                or os.path.normcase(os.path.realpath(part.path)) != part.resolved):
            raise PreviewPointPending(f"{part.path.name} changed during decoding")
    if pixels.ndim != 2 or pixels.shape != point._shape or not np.issubdtype(pixels.dtype, np.number):
        raise PreviewPointPending(f"{part.path.name} does not match the configured detector plane")
    if pixels.nbytes > point._max_bytes:
        raise PreviewPointPending(f"{part.path.name} exceeds the admitted detector frame bytes")
    rotated = image.apply_rotation(pixels, point._rotation)
    rotated.setflags(write=False)
    return rotated, saturation


def read_preview_point(point: PreviewPoint, *, cancelled=None, include_saturation=False):
    """Decode selected planes once. Optionally return their recorded cutoffs beside them."""
    try:
        planes, saturations = [], []
        indices = (0,) if point.policy is None else point.policy.raw_indices(0)
        for index in indices:
            _cancel(cancelled)
            pixels, saturation = _read_flat_plane(point.raw[index], point)
            planes.append(pixels)
            saturations.append(saturation)
        _cancel(cancelled)
        result = tuple(planes)
        return (result, tuple(saturations)) if include_saturation else result
    except PreviewPointPending:
        raise
    except Exception as error:
        _cancel(cancelled)
        raise PreviewPointPending(f"Waiting for detector point {point.label}: {error}") from error


def preview_prefix_holds(observed: PreviewMember, points=None, *, cancelled=None):
    """Check covered source bytes, exact logical/raw mapping and revisions for adoption."""
    return preview_prefixes_hold(((observed, points),), cancelled=cancelled)


def preview_prefixes_hold(requirements, *, cancelled=None):
    """Revalidate one slice, sharing the changed-file scan and the ready image inventory."""
    try:
        spec_checks = []
        with image.shared_image_listings(all_stems=True):
            for observed, points in requirements:
                _cancel(cancelled)
                points = observed.points if points is None else tuple(points)
                if not points:
                    continue
                if observed.prefix is None:
                    return False
                covered = max(point.label for point in points) + 1
                if covered > len(observed.points):
                    return False
                endpoint = observed._successor_required if observed.end_of_stream else None
                spec_checks.append((observed.prefix, covered, endpoint))
                slots = _named_slots(
                    observed.path, observed.key, observed._image_dir, observed._image_stem, cancelled)
                # A frame selection cannot hide a missing or remapped predecessor of this point.
                for point in observed.points[:covered]:
                    _cancel(cancelled)
                    for expected in point.raw:
                        file = slots.get(expected.slot)
                        if file != expected.path or _raw_part(file, expected.slot) != expected:
                            return False
        return spec.spec_preview_prefixes_hold(spec_checks, _open_shared_read, cancelled=cancelled)
    except (OSError, ValueError, LookupError):
        _cancel(cancelled)
        return False
