"""Bounded, headless catalog and one-frame reads for the 2D Viewer."""

from __future__ import annotations

import ast
import hashlib
import io
import math
import os
import re
import stat
import struct
import zipfile
import zlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from xrd_tools.core.staging import total_physical_ram_bytes as _physical_ram_bytes
from xrd_tools.io.stat_identity import identity_ctime_ns

SUPPORTED_VIEWER_SUFFIXES = frozenset({
    ".edf", ".tif", ".tiff", ".cbf", ".img", ".mar3450", ".raw",
    ".h5", ".hdf5", ".nxs", ".nexus", ".csv", ".npy", ".npz",
})
POLICY_VERSION = "viewer-2d-v2"
CATALOG_RESERVATION = (
    2 * 1024 * 1024 + 512 * 10_000 + 4096 + 128 * 1024 + 4 * 1024 * 1024
)
_MAX_FRAMES = 10_000
_MAX_DIMENSION = 65_536
_MAX_PIXELS = 67_108_864
_MAX_PATH = 4096
_MAX_MANIFEST = 128 * 1024
_MAX_CATALOG_MANIFEST = 4 * 1024 * 1024
_MAX_PROCESSED_FIELDS = 12 * _MAX_FRAMES
_MAX_PROCESSED_FIELD_BYTES = 64 * 1024**2
_MAX_LINE = 1024 * 1024
_MAX_CSV = 512 * 1024**2
_MAX_NPY = 64 * 1024**3
_MAX_NPZ = 2 * 1024**3
_MAX_FABIO = 256 * 1024**2
_HDF5_SUFFIXES = frozenset({".h5", ".hdf5", ".nxs", ".nexus"})
_RAW_DATASETS = (
    "/entry/instrument/detector/data", "/entry/instrument/detector/data_000001",
    "/entry/data/data", "/entry/measurement/data", "/entry/data/eiger_image",
)
# A processed record's source dataset hint that names one Eiger segment
# link, as _hdf_segments enumerates them: ``/<entry>/data/data_<number>``.
_SEGMENT_HINT = re.compile(r"^(?P<group>/(?P<entry>[^/]+)/data)/(?P<name>data_[0-9]+)$")


class Viewer2DSourceKind(str, Enum):
    RAW_DETECTOR = "raw_detector"
    PROCESSED_RAW = "processed_raw"
    PROCESSED_THUMBNAIL = "processed_thumbnail"
    CSV_MATRIX = "csv_matrix"
    NUMPY_ARRAY = "numpy_array"


class Viewer2DRefusalCode(str, Enum):
    UNSUPPORTED_FORMAT = "unsupported_format"
    PATH_INVALID = "path_invalid"
    FORMAT_INVALID = "format_invalid"
    LIMIT_EXCEEDED = "limit_exceeded"
    MEMORY_REFUSED = "memory_refused"
    FRAME_UNKNOWN = "frame_unknown"
    VIEWER_SOURCE_CHANGED = "viewer_source_changed"


class Viewer2DReadError(RuntimeError):
    def __init__(self, code: Viewer2DRefusalCode, message: str):
        if type(code) is not Viewer2DRefusalCode or type(message) is not str or not message:
            raise TypeError("viewer refusal is malformed")
        self.code = code
        super().__init__(message)


def _refuse(code, message):
    raise Viewer2DReadError(code, message)


def _reject_if(condition, code, message):
    if condition: _refuse(code, message)


def _sha256_text(value):
    return (type(value) is str and len(value) == 64
            and not set(value) - set("0123456789abcdef"))


def _optional(value, expected):
    return value is None or type(value) is expected


def _malformed(condition, name):
    if condition: raise TypeError(f"viewer {name} is malformed")


@dataclass(frozen=True, slots=True)
class Viewer2DFormatPolicy:
    policy_version: str = POLICY_VERSION
    raw_detector_shape: tuple[int, int] | None = None
    raw_dtype: str = "int32"
    raw_header_skip: int = 0
    source_root: str | None = None

    def __post_init__(self):
        shape = self.raw_detector_shape
        _malformed(self.policy_version != POLICY_VERSION or not _optional(shape, tuple)
            or shape is not None and len(shape) != 2 or type(self.raw_header_skip) is not int
            or self.raw_header_skip < 0 or not _optional(self.source_root, str), "format policy")
        if shape is not None: _shape_fhw(shape)
        _source_dtype(self.raw_dtype)

    @property
    def identity(self):
        return (self.policy_version, self.raw_detector_shape, self.raw_dtype,
                self.raw_header_skip, self.source_root)


@dataclass(frozen=True, slots=True)
class Viewer2DRevision:
    """File metadata for freshness checks, not a content fingerprint.

    Viewer sources are ordinary local/shared files, not hostile writers.
    Catalogs are immutable process-local values, validated at construction.
    """
    canonical_path: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    def __post_init__(self):
        values = (self.device, self.inode, self.size, self.mtime_ns, self.ctime_ns)
        _malformed(type(self.canonical_path) is not str or not self.canonical_path
            or len(os.fsencode(self.canonical_path)) > _MAX_PATH
            or any(type(value) is not int or value < 0 for value in values), "revision")


@dataclass(frozen=True, slots=True)
class Viewer2DDependency:
    locator: str
    dataset_path: str | None
    revision: Viewer2DRevision
    frame_start: int | None = None
    frame_stop: int | None = None
    logical_path: str | None = None

    def __post_init__(self):
        interval = (self.frame_start, self.frame_stop)
        if type(self.revision) is Viewer2DRevision:
            self.revision.__post_init__()
        _malformed(type(self.locator) is not str or not self.locator
            or not _optional(self.dataset_path, str) or type(self.revision) is not Viewer2DRevision
            or self.locator != self.revision.canonical_path
            or interval != (None, None) and (type(interval[0]) is not int
                or type(interval[1]) is not int or not 0 <= interval[0] < interval[1])
            or not _optional(self.logical_path, str)
            or any(value == "" for value in (self.dataset_path, self.logical_path)
                   if value is not None)
            or any(len(value.encode("utf-8", "strict")) > _MAX_PATH
                   for value in (self.locator, self.dataset_path, self.logical_path)
                   if value is not None), "dependency")


@dataclass(frozen=True, slots=True)
class Viewer2DFrameFact:
    label: int
    source_kind: Viewer2DSourceKind
    shape: tuple[int, int]
    dtype: str
    locator: str = ""
    dataset_path: str | None = None
    source_frame: int | None = None
    thumbnail_path: str | None = None
    source_catalog_identity: str = ""

    def __post_init__(self):
        _malformed(type(self.label) is not int or type(self.source_kind) is not Viewer2DSourceKind
            or self.label < 0
            or type(self.shape) is not tuple or len(self.shape) != 2
            or _source_dtype(self.dtype).str != self.dtype
            or not all(_optional(value, expected) for value, expected in ((self.dataset_path, str),
                (self.source_frame, int), (self.thumbnail_path, str)))
            or not all(type(value) is str for value in (self.locator, self.source_catalog_identity))
            or self.source_kind is Viewer2DSourceKind.PROCESSED_RAW and (
                not self.locator or self.source_frame is None or self.source_frame < 0
                or not _sha256_text(self.source_catalog_identity) or self.thumbnail_path is not None)
            or self.source_kind is Viewer2DSourceKind.PROCESSED_THUMBNAIL and (
                self.locator or self.dataset_path is not None or self.source_frame is not None
                or not self.thumbnail_path or self.source_catalog_identity
                or self.dtype not in {np.dtype("uint8").str, np.dtype("uint16").str}),
            "frame fact")
        _malformed(any(len(value.encode("utf-8", "strict")) > _MAX_PATH
                       for value in (self.locator, self.dataset_path, self.thumbnail_path)
                       if value is not None), "frame fact")
        _shape_fhw(self.shape)


@dataclass(frozen=True, slots=True)
class Viewer2DArtifactCatalog:
    canonical_path: str
    policy_version: str
    policy_identity: tuple
    source_kind: Viewer2DSourceKind
    frame_labels: tuple[int, ...]
    source_shape: tuple[int, ...]
    source_dtype: str
    primary_revision: Viewer2DRevision
    dependencies: tuple[Viewer2DDependency, ...]
    format_name: str
    catalog_identity: str
    member_name: str | None = None
    dataset_path: str | None = None
    frame_facts: tuple[Viewer2DFrameFact, ...] = ()

    def __post_init__(self):
        if type(self.primary_revision) is Viewer2DRevision:
            self.primary_revision.__post_init__()
        if type(self.dependencies) is tuple and type(self.frame_facts) is tuple:
            for value in (*self.dependencies, *self.frame_facts):
                if type(value) in (Viewer2DDependency, Viewer2DFrameFact):
                    value.__post_init__()
        manifest = (self.canonical_path, self.source_kind.value, self.frame_labels,
            self.source_shape, self.source_dtype, self.dependencies, self.format_name,
            self.member_name, self.dataset_path, self.frame_facts, self.policy_identity,
            self.primary_revision)
        _malformed(type(self.canonical_path) is not str or self.policy_version != POLICY_VERSION
            or type(self.source_kind) is not Viewer2DSourceKind or type(self.frame_labels) is not tuple
            or not self.frame_labels or any(type(label) is not int or label < 0 for label in self.frame_labels)
            or len(set(self.frame_labels)) != len(self.frame_labels)
            or type(self.policy_identity) is not tuple or type(self.source_shape) is not tuple
            or _source_dtype(self.source_dtype).str != self.source_dtype
            or type(self.primary_revision) is not Viewer2DRevision or type(self.dependencies) is not tuple
            or type(self.frame_facts) is not tuple
            or any(type(item) is not Viewer2DDependency for item in self.dependencies)
            or any(type(item) is not Viewer2DFrameFact for item in self.frame_facts)
            or self.canonical_path != self.primary_revision.canonical_path
            or len(os.fsencode(self.canonical_path)) > _MAX_PATH
            or type(self.format_name) is not str or not self.format_name
            or any(value == "" for value in (self.member_name, self.dataset_path)
                   if value is not None)
            or self.frame_facts and tuple(item.label for item in self.frame_facts) != self.frame_labels
            or not _sha256_text(self.catalog_identity)
            or len(repr(manifest).encode("utf-8", "strict")) > _MAX_CATALOG_MANIFEST
            or self.catalog_identity != _catalog_id(*manifest), "catalog")
        frame_count, _, _ = _shape_fhw(self.source_shape)
        _malformed(not self.frame_facts and frame_count != len(self.frame_labels), "catalog")
        _validate_catalog_cross_fields(self)


@dataclass(frozen=True, slots=True)
class Viewer2DFrameProvenance:
    requested_artifact: str
    catalog_identity: str
    format_name: str
    frame_label: int
    frame_index: int
    canonical_shape: tuple[int, int]
    canonical_dtype: str
    canonical_nbytes: int
    canonical_sha256: str
    source_sha256: str
    source_dtype: str
    source_nbytes: int
    source_kind: Viewer2DSourceKind
    primary_revision: Viewer2DRevision
    dependencies: tuple[Viewer2DDependency, ...]
    raw_locator: str = ""
    dataset_path: str | None = None
    raw_source_frame: int | None = None
    pixel_axis_policy: str = "implicit-pixel-v1"
    degraded_thumbnail: bool = False
    diagnostic: str = ""

    def __post_init__(self):
        if type(self.primary_revision) is Viewer2DRevision:
            self.primary_revision.__post_init__()
        if type(self.dependencies) is tuple:
            for value in self.dependencies:
                if type(value) is Viewer2DDependency:
                    value.__post_init__()
        _malformed(type(self.canonical_shape) is not tuple or len(self.canonical_shape) != 2
            or self.canonical_dtype != np.dtype(float).str or type(self.canonical_nbytes) is not int
            or self.canonical_nbytes != math.prod(self.canonical_shape) * np.dtype(float).itemsize
            or not all(_sha256_text(value) for value in
                (self.canonical_sha256, self.source_sha256))
            or _source_dtype(self.source_dtype).str != self.source_dtype
            or type(self.source_nbytes) is not int
            or self.source_nbytes != math.prod(self.canonical_shape) * np.dtype(self.source_dtype).itemsize
            or type(self.source_kind) is not Viewer2DSourceKind
            or type(self.primary_revision) is not Viewer2DRevision
            or type(self.dependencies) is not tuple
            or any(type(item) is not Viewer2DDependency for item in self.dependencies)
            or type(self.requested_artifact) is not str
            or self.requested_artifact != self.primary_revision.canonical_path
            or not _sha256_text(self.catalog_identity) or not self.format_name
            or type(self.frame_label) is not int or self.frame_label < 0
            or type(self.frame_index) is not int or self.frame_index < 0
            or self.pixel_axis_policy != "implicit-pixel-v1"
            or type(self.degraded_thumbnail) is not bool or type(self.diagnostic) is not str
            or self.raw_source_frame is not None and (
                type(self.raw_source_frame) is not int or self.raw_source_frame < 0)
            or self.source_kind is Viewer2DSourceKind.PROCESSED_THUMBNAIL and (
                not self.degraded_thumbnail or self.raw_locator or self.dataset_path is not None
                or self.raw_source_frame is not None
                or self.diagnostic != "Thumbnail preview; raw source unavailable."),
            "frame provenance")
        _shape_fhw(self.canonical_shape)
        simple = {
            Viewer2DSourceKind.CSV_MATRIX: {"csv"},
            Viewer2DSourceKind.NUMPY_ARRAY: {"npy", "npz"},
            Viewer2DSourceKind.PROCESSED_RAW: {"hdf5-processed"},
            Viewer2DSourceKind.PROCESSED_THUMBNAIL: {"hdf5-processed"},
        }
        _malformed(self.source_kind in simple and self.format_name not in simple[self.source_kind]
            or self.source_kind in (Viewer2DSourceKind.CSV_MATRIX,
                                    Viewer2DSourceKind.NUMPY_ARRAY) and (
                self.dependencies or self.raw_locator or self.dataset_path is not None
                or self.raw_source_frame is not None or self.degraded_thumbnail or self.diagnostic)
            or self.source_kind is Viewer2DSourceKind.PROCESSED_RAW and (
                not self.raw_locator or self.raw_source_frame is None or self.degraded_thumbnail
                or self.diagnostic or not self.dependencies
                or not any(item.locator == self.raw_locator for item in self.dependencies))
            or self.source_kind is Viewer2DSourceKind.RAW_DETECTOR and (
                self.format_name not in {"raw", "tiff", "edf", "cbf", "img", "mar3450",
                                         "hdf5", "hdf5-eiger"}
                or self.degraded_thumbnail or self.diagnostic
                or bool(self.raw_locator) != (self.raw_source_frame is not None)
                or self.dependencies and not any(
                    item.locator == self.raw_locator for item in self.dependencies))
            or not all(_optional(value, str) for value in (self.dataset_path,))
            or type(self.raw_locator) is not str,
            "frame provenance")


@dataclass(frozen=True, slots=True)
class Viewer2DFrame:
    catalog_identity: str
    label: int
    array: np.ndarray
    provenance: Viewer2DFrameProvenance

    def __post_init__(self):
        value = self.array
        if type(self.provenance) is Viewer2DFrameProvenance:
            self.provenance.__post_init__()
        _malformed(type(self.catalog_identity) is not str or type(self.label) is not int
            or type(value) is not np.ndarray or value.dtype != np.dtype(float) or value.ndim != 2
            or not value.flags.c_contiguous or not value.flags.owndata or value.flags.writeable
            or value.base is not None or type(self.provenance) is not Viewer2DFrameProvenance
            or self.provenance.catalog_identity != self.catalog_identity
            or self.provenance.frame_label != self.label
            or self.provenance.canonical_shape != value.shape
            or self.provenance.canonical_nbytes != value.nbytes
            or self.provenance.canonical_sha256 != hashlib.sha256(
                value.tobytes(order="C")).hexdigest(), "canonical frame")


@dataclass(frozen=True, slots=True)
class Viewer2DMemoryLedger:
    canonical_bytes: int
    catalog_reservation: int
    encoded_retained: int
    reader_peak: int
    renderer_slack_raw: int
    renderer_linear: int
    renderer_log_update: int
    renderer_paint: int
    renderer_peak: int
    concurrent_frame_bytes: int
    total: int
    admission: int
    budget: int

    def __post_init__(self):
        values = tuple(getattr(self, item) for item in self.__dataclass_fields__)
        c = self.canonical_bytes
        _malformed(any(type(value) is not int or value < 0 for value in values)
            or c <= 0 or c % 8 or self.catalog_reservation != CATALOG_RESERVATION
            or self.budget != _memory_budget()
            or self.reader_peak != max(3 * c, self.encoded_retained + 4 * c)
            or self.renderer_slack_raw != 5 * c // 2
            or self.renderer_linear != 7 * c // 2
            or self.renderer_log_update != 9 * c // 2
            or self.renderer_paint != 5 * c or self.renderer_peak != 5 * c
            or self.concurrent_frame_bytes != 6 * c
            or self.total != self.catalog_reservation + max(self.reader_peak, 6 * c)
            or self.admission != c + self.total,
            "memory ledger")


def _memory_budget(ram_bytes=None):
    detected = _physical_ram_bytes() if ram_bytes is None else ram_bytes
    return 884_736_000 if type(detected) is not int or detected <= 0 else min(
        1024**3, detected // 20
    )


def viewer_2d_memory_ledger(height, width, *, encoded_retained=0, ram_bytes=None):
    if type(height) is not int or type(width) is not int:
        raise TypeError("frame dimensions must be exact integers")
    _shape_fhw((height, width))
    if type(encoded_retained) is not int or encoded_retained < 0:
        raise TypeError("encoded retention must be a nonnegative integer")
    c = 8 * height * width
    q = max(3 * c, encoded_retained + 4 * c)
    concurrent = 6 * c
    total = CATALOG_RESERVATION + max(q, concurrent)
    budget = _memory_budget(ram_bytes)
    return Viewer2DMemoryLedger(
        c, CATALOG_RESERVATION, encoded_retained, q, 5 * c // 2,
        7 * c // 2, 9 * c // 2, 5 * c, 5 * c, concurrent, total,
        c + total, budget,
    )


def _admit(shape, *, encoded=0):
    _, h, w = _shape_fhw(shape)
    ledger = viewer_2d_memory_ledger(h, w, encoded_retained=encoded)
    if ledger.admission > ledger.budget:
        _refuse(Viewer2DRefusalCode.MEMORY_REFUSED, "selected frame exceeds memory budget")
    return ledger


def _shape_fhw(shape):
    if type(shape) is not tuple or len(shape) not in (2, 3):
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "only rank-two matrices and rank-three stacks are supported")
    if any(type(value) is not int or value <= 0 for value in shape):
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "array dimensions must be positive exact integers")
    f, h, w = (1, *shape) if len(shape) == 2 else shape
    if f > _MAX_FRAMES or h > _MAX_DIMENSION or w > _MAX_DIMENSION or h * w > _MAX_PIXELS:
        _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "array shape exceeds the 2D Viewer limits")
    return f, h, w


def _source_dtype(value):
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError):
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "invalid source dtype")
    if (dtype.fields is not None or dtype.subdtype is not None or dtype.hasobject
            or dtype.kind not in "biuf" or not 1 <= dtype.itemsize <= 8):
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "source dtype is not a closed real scalar dtype")
    return dtype


def _canonical_path(value):
    if not isinstance(value, (str, os.PathLike)):
        _refuse(Viewer2DRefusalCode.PATH_INVALID, "artifact path must be a string or Path")
    try:
        path = Path(value).expanduser().resolve(strict=True)
        encoded = os.fsencode(str(path))
        info = path.stat()
    except (OSError, ValueError, TypeError) as error:
        _refuse(Viewer2DRefusalCode.PATH_INVALID, str(error))
    if len(encoded) > _MAX_PATH or not stat.S_ISREG(info.st_mode):
        _refuse(Viewer2DRefusalCode.PATH_INVALID, "artifact must be a bounded regular file")
    return path


def _revision(path, info):
    return Viewer2DRevision(str(path), *_stat_identity(info))


def _stat_identity(info):
    """Normalize the platform's descriptor/pathname ctime difference once."""
    return (int(info.st_dev), int(info.st_ino), int(info.st_size),
            int(info.st_mtime_ns), identity_ctime_ns(info.st_ctime_ns))


def _revision_stat(revision):
    return (revision.device, revision.inode, revision.size,
            revision.mtime_ns, revision.ctime_ns)


def _path_stat(path):
    try:
        info = path.stat()
    except OSError:
        _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                "viewer source is no longer readable")
    return _stat_identity(info)


def _cert_stat_revision(revision, *, stream=None, message):
    """Refuse changed file metadata without reading the source payload."""

    descriptor = None
    if stream is not None:
        try:
            info = os.fstat(stream.fileno())
        except (OSError, ValueError):
            _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, message)
        descriptor = _stat_identity(info)
    try:
        pathname = _path_stat(Path(revision.canonical_path))
    except Viewer2DReadError:
        pathname = None
    if pathname != _revision_stat(revision) or (
            descriptor is not None
            and descriptor != _revision_stat(revision)):
        _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, message)


def _descriptor_revision(path, stream):
    revision = _revision(path, os.fstat(stream.fileno()))
    _cert_stat_revision(revision, stream=stream, message="viewer descriptor or path changed")
    return revision


def _stable_revision(path):
    return _revision(path, path.stat())


def _assert_primary(catalog):
    path = Path(catalog.canonical_path)
    try:
        current = _stable_revision(path)
    except OSError:
        _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, "viewer source is no longer readable")
    if current != catalog.primary_revision:
        _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, "viewer source changed")


def _catalog_id(*parts):
    payload = b"\0".join(str(part).encode("utf-8", "strict") for part in parts)
    return hashlib.sha256(payload).hexdigest()


def _dependency_chains(dependencies):
    chains, current = [], []
    for dependency in dependencies:
        if dependency.frame_start is None:
            if current:
                chains.append(tuple(current))
            current = [dependency]
        elif not current:
            return ()
        else:
            current.append(dependency)
    if current:
        chains.append(tuple(current))
    return tuple(chains)


def _chain_identities(catalog, chain, shape, dtype):
    """The source-catalog identities a raw fact of *shape* and *dtype* over
    the dependency *chain* (its base, then its frame intervals) may carry:
    the single-dataset "hdf5" catalog of an external link covering the chain
    (both source shapes when the chain is one frame long) and the
    "hdf5-eiger" catalog of a contiguous segment chain anchored at the base's
    dataset."""
    total = chain[-1].frame_stop
    intervals = chain[1:]
    identities = set()
    if (len(intervals) == 1 and intervals[0].frame_start == 0
            and intervals[0].frame_stop == total
            and intervals[0].logical_path == chain[0].dataset_path):
        shapes = ((total, *shape),)
        if total == 1:
            shapes += (shape,)
        for source_shape in shapes:
            manifest = (
                chain[0].locator, Viewer2DSourceKind.RAW_DETECTOR.value,
                tuple(range(total)), source_shape, dtype, intervals,
                "hdf5", None, chain[0].dataset_path, (),
                catalog.policy_identity, chain[0].revision)
            identities.add(_catalog_id(*manifest))
    cover = tuple((item.frame_start, item.frame_stop) for item in intervals)
    if (all(item.logical_path is not None for item in intervals)
            and intervals[0].logical_path == chain[0].dataset_path
            and cover == tuple((start, stop) for start, stop in zip(
                (0, *(item.frame_stop for item in intervals[:-1])),
                (item.frame_stop for item in intervals)))):
        manifest = (
            chain[0].locator, Viewer2DSourceKind.RAW_DETECTOR.value,
            tuple(range(total)), (total, *shape), dtype,
            intervals, "hdf5-eiger", None, chain[0].dataset_path, (),
            catalog.policy_identity, chain[0].revision)
        identities.add(_catalog_id(*manifest))
    return frozenset(identities)


def _validate_catalog_cross_fields(catalog):
    try:
        identity_policy = Viewer2DFormatPolicy(*catalog.policy_identity)
    except (TypeError, ValueError, Viewer2DReadError):
        identity_policy = None
    f, _, _ = _shape_fhw(catalog.source_shape)
    fmt = catalog.format_name
    kind = catalog.source_kind
    dependencies = catalog.dependencies
    intervals = tuple(item for item in dependencies if item.frame_start is not None)
    suffix = Path(catalog.canonical_path).suffix.lower()
    suffix_format = ({".tif": "tiff", ".tiff": "tiff"}.get(suffix, suffix[1:])
                     if suffix else "")
    _malformed(identity_policy is None or identity_policy.identity != catalog.policy_identity
        or len({repr(item) for item in dependencies}) != len(dependencies)
        or any((item.frame_start is None) != (item.frame_stop is None)
               for item in dependencies), "catalog")
    if catalog.frame_facts:
        first = catalog.frame_facts[0]
        chains = _dependency_chains(dependencies)
        raw_facts = tuple(fact for fact in catalog.frame_facts
                          if fact.source_kind is Viewer2DSourceKind.PROCESSED_RAW)
        selected = []
        for fact in raw_facts:
            matches = tuple(chain for chain in chains
                if chain[0].locator == fact.locator
                and chain[0].dataset_path == fact.dataset_path)
            chosen = tuple(item for item in matches[0][1:]
                if item.frame_start <= fact.source_frame < item.frame_stop) if len(matches) == 1 else ()
            selected.append((matches, chosen))
        _malformed(fmt != "hdf5-processed" or catalog.member_name is not None
            or catalog.dataset_path is not None or catalog.source_shape != first.shape
            or catalog.source_dtype != first.dtype or kind is not first.source_kind
            or any(fact.source_kind not in (Viewer2DSourceKind.PROCESSED_RAW,
                                             Viewer2DSourceKind.PROCESSED_THUMBNAIL)
                   for fact in catalog.frame_facts)
            or bool(raw_facts) != bool(chains)
            or any(len(matches) != 1 or (len(matches[0]) > 1 and len(chosen) != 1)
                   for matches, chosen in selected)
            or any(tuple(sorted(chain[1:], key=lambda item: item.frame_start)) != chain[1:]
                   or any(left.frame_stop > right.frame_start
                          for left, right in zip(chain[1:], chain[2:]))
                   or any(item.logical_path is None for item in chain[1:])
                   for chain in chains), "catalog")
        # A chain's admissible source-catalog identities depend on the fact
        # only through its shape and dtype: computed once per (chain, shape,
        # dtype), not once per raw fact -- each manifest strings the chain's
        # whole frame range, so per-fact hashing cost every read of a
        # 2,500-frame all-raw record ~0.3 s and a 3,621-frame one ~0.9 s.
        identities_by_chain = {}
        for fact, (matches, _) in zip(raw_facts, selected):
            chain = matches[0]
            _malformed(len(chain) == 1 and chain[0].dataset_path is not None
                and chain[0].dataset_path.rsplit("/", 1)[-1].startswith("data_")
                and chain[0].dataset_path.rsplit("/", 1)[-1][5:].isdigit(), "catalog")
            if len(chain) > 1:
                key = (id(chain), fact.shape, fact.dtype)
                identities = identities_by_chain.get(key)
                if identities is None:
                    identities = identities_by_chain[key] = _chain_identities(
                        catalog, chain, fact.shape, fact.dtype)
                _malformed(fact.source_catalog_identity not in identities, "catalog")
        return
    _malformed(catalog.frame_labels != tuple(range(f)), "catalog")
    if fmt == "csv":
        valid = (kind is Viewer2DSourceKind.CSV_MATRIX and len(catalog.source_shape) == 2
                 and catalog.member_name is catalog.dataset_path is None and not dependencies)
    elif fmt in {"npy", "npz"}:
        valid = (kind is Viewer2DSourceKind.NUMPY_ARRAY and catalog.dataset_path is None
                 and not dependencies and ((fmt == "npy" and catalog.member_name is None)
                 or (fmt == "npz" and type(catalog.member_name) is str
                     and catalog.member_name.endswith(".npy"))))
    elif fmt in {"raw", "tiff", "edf", "cbf", "img", "mar3450"}:
        valid = (kind is Viewer2DSourceKind.RAW_DETECTOR
                 and fmt == suffix_format and catalog.member_name is catalog.dataset_path is None
                 and not dependencies and (fmt != "raw" or len(catalog.source_shape) == 2
                 and catalog.frame_labels == (0,) and identity_policy.raw_detector_shape ==
                     catalog.source_shape and np.dtype(identity_policy.raw_dtype).str ==
                     catalog.source_dtype and catalog.primary_revision.size ==
                     identity_policy.raw_header_skip + math.prod(catalog.source_shape) *
                     np.dtype(catalog.source_dtype).itemsize))
    elif fmt == "hdf5":
        external = len(dependencies) == 1 and intervals == dependencies
        valid = (kind is Viewer2DSourceKind.RAW_DETECTOR
                 and catalog.member_name is None and type(catalog.dataset_path) is str
                 and suffix in _HDF5_SUFFIXES and (not dependencies or external)
                 and (not external or dependencies[0].frame_start == 0
                 and dependencies[0].frame_stop == f
                 and dependencies[0].logical_path == catalog.dataset_path))
    elif fmt == "hdf5-eiger":
        cover = tuple((item.frame_start, item.frame_stop) for item in intervals)
        valid = (kind is Viewer2DSourceKind.RAW_DETECTOR
                 and catalog.member_name is None and type(catalog.dataset_path) is str
                 and suffix in _HDF5_SUFFIXES
                 and len(intervals) == len(dependencies) and bool(intervals)
                 and all(item.logical_path is not None for item in intervals)
                 and intervals[0].logical_path == catalog.dataset_path
                 and cover == tuple((start, stop) for start, stop in
                     zip((0, *(item.frame_stop for item in intervals[:-1])),
                         (item.frame_stop for item in intervals)))
                 and intervals[-1].frame_stop == f)
    else:
        valid = False
    _malformed(not valid, "catalog")


def _selected_dependencies(catalog, label):
    index = catalog.frame_labels.index(label)
    fact = next((item for item in catalog.frame_facts if item.label == label), None)
    if fact is None:
        return tuple(item for item in catalog.dependencies
                     if item.frame_start is not None
                     and item.frame_start <= index < item.frame_stop)
    if fact.source_kind is Viewer2DSourceKind.PROCESSED_THUMBNAIL:
        return ()
    chains = tuple(chain for chain in _dependency_chains(catalog.dependencies)
        if chain[0].locator == fact.locator and chain[0].dataset_path == fact.dataset_path)
    _malformed(len(chains) != 1, "selected dependency chain")
    chain = chains[0]
    if len(chain) == 1:
        return chain
    selected = tuple(item for item in chain[1:]
        if item.frame_start <= fact.source_frame < item.frame_stop)
    _malformed(len(selected) != 1, "selected dependency interval")
    return chain[0], selected[0]


def viewer_2d_selected_ledger(catalog, label):
    if type(catalog) is not Viewer2DArtifactCatalog:
        raise TypeError("selected ledger requires an exact viewer catalog")
    if type(label) is not int or label not in catalog.frame_labels:
        raise TypeError("selected ledger label is not certified")
    fact = next((item for item in catalog.frame_facts if item.label == label), None)
    shape = fact.shape if fact is not None else catalog.source_shape[-2:]
    encoded = 0
    if catalog.format_name in {"edf", "cbf", "img", "mar3450"}:
        encoded = catalog.primary_revision.size
    elif fact is not None and fact.source_kind is Viewer2DSourceKind.PROCESSED_RAW \
            and Path(fact.locator).suffix.lower() in {".edf", ".cbf", ".img", ".mar3450"}:
        dependency = next((item for item in _selected_dependencies(catalog, label)
                           if item.locator == fact.locator), None)
        _malformed(dependency is None, "selected ledger")
        encoded = dependency.revision.size
    return viewer_2d_memory_ledger(*shape, encoded_retained=encoded)


def _validate_frame_against_catalog(catalog, label, frame):
    _malformed(type(catalog) is not Viewer2DArtifactCatalog
        or type(frame) is not Viewer2DFrame or type(label) is not int
        or label not in catalog.frame_labels, "selected frame")
    fact = next((item for item in catalog.frame_facts if item.label == label), None)
    p = frame.provenance
    dependencies = _selected_dependencies(catalog, label)
    kind = catalog.source_kind if fact is None else fact.source_kind
    shape = catalog.source_shape[-2:] if fact is None else fact.shape
    _malformed(frame.catalog_identity != catalog.catalog_identity or frame.label != label
        or p.requested_artifact != catalog.canonical_path
        or p.catalog_identity != catalog.catalog_identity or p.format_name != catalog.format_name
        or p.frame_label != label or p.frame_index != catalog.frame_labels.index(label)
        or p.primary_revision != catalog.primary_revision or p.dependencies != dependencies
        or p.source_kind is not kind or p.canonical_shape != shape, "selected frame")
    if fact is not None:
        _malformed(kind is Viewer2DSourceKind.PROCESSED_RAW and (
            p.raw_locator != fact.locator
            or p.dataset_path != (next((item.dataset_path for item in dependencies
                if item.frame_start is not None and
                item.frame_start <= fact.source_frame < item.frame_stop), None)
                or fact.dataset_path)
            or p.raw_source_frame != fact.source_frame or p.source_dtype != fact.dtype)
            or kind is Viewer2DSourceKind.PROCESSED_THUMBNAIL and (
                p.raw_locator or p.dataset_path is not None or p.raw_source_frame is not None
                or p.dependencies or not p.degraded_thumbnail
                or p.source_dtype != fact.dtype), "selected frame")
    elif catalog.format_name.startswith("hdf5"):
        index = catalog.frame_labels.index(label)
        dependency = next(iter(dependencies), None)
        locator = catalog.canonical_path if dependency is None else dependency.locator
        dataset = catalog.dataset_path if dependency is None else dependency.dataset_path
        source_frame = index if dependency is None else index - dependency.frame_start
        _malformed(p.raw_locator != locator or p.dataset_path != dataset
            or p.raw_source_frame != source_frame
            or p.source_dtype != catalog.source_dtype, "selected frame")


def _make_catalog(path, policy, kind, labels, shape, dtype, revision, dependencies,
                  format_name, *, member=None, dataset=None, facts=()):
    shape = tuple(shape)
    fixed = (str(path), kind.value, shape, np.dtype(dtype).str, format_name,
             member, dataset, policy.identity, revision)
    size = 24 + sum(len(repr(item).encode("utf-8")) for item in fixed) + 6
    retained = []
    for values, limit in ((labels, _MAX_FRAMES), (dependencies, None), (facts, None)):
        items, item_total = [], 0
        for item in values:
            if limit is not None and len(items) >= limit:
                _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "catalog exceeds 10,000 frames")
            encoded = len(repr(item).encode("utf-8"))
            count = len(items) + 1
            projected = item_total + encoded + (3 if count == 1 else 2 * count)
            if size - 2 + projected > _MAX_CATALOG_MANIFEST:
                _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                        "catalog manifest exceeds 4 MiB")
            items.append(item)
            item_total += encoded
        current = tuple(items)
        size += (2 if not items else item_total + (3 if len(items) == 1
                 else 2 * len(items))) - 2
        retained.append(current)
    labels, dependencies, facts = retained
    if not labels or len(set(labels)) != len(labels):
        _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "catalog labels are empty, duplicate, or over limit")
    manifest = (str(path), kind.value, labels, shape, np.dtype(dtype).str, dependencies,
                format_name, member, dataset, facts, policy.identity, revision)
    if len(repr(manifest).encode("utf-8")) > _MAX_CATALOG_MANIFEST:
        raise AssertionError("incremental viewer manifest accounting drifted")
    return Viewer2DArtifactCatalog(
        str(path), POLICY_VERSION, policy.identity, kind, labels, shape,
        np.dtype(dtype).str, revision, dependencies, format_name,
        _catalog_id(*manifest), member, dataset, facts,
    )


def _csv_scan(stream, target=None):
    stream.seek(0)
    digest = hashlib.sha256()
    rows = columns = encoded = 0
    while True:
        raw = stream.readline(_MAX_LINE + 2)
        if not raw:
            break
        encoded += len(raw)
        _reject_if(encoded > _MAX_CSV, Viewer2DRefusalCode.LIMIT_EXCEEDED,
                   "CSV exceeds 512 MiB")
        digest.update(raw)
        _reject_if(len(raw) > _MAX_LINE + 1 or (len(raw) > _MAX_LINE and not raw.endswith(b"\n")),
                   Viewer2DRefusalCode.LIMIT_EXCEEDED, "CSV logical line exceeds 1 MiB")
        _reject_if(rows == 0 and raw.startswith(b"\xef\xbb\xbf"),
                   Viewer2DRefusalCode.FORMAT_INVALID, "CSV BOM is not supported")
        try:
            text = raw.decode("utf-8", "strict").rstrip("\r\n")
        except UnicodeDecodeError:
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "CSV must be UTF-8")
        _reject_if(not text.strip() or text.lstrip().startswith("#") or '"' in text,
                   Viewer2DRefusalCode.FORMAT_INVALID,
                   "CSV must contain only unquoted numeric rows")
        tokens = text.split(",")
        _reject_if(any(not token.strip() for token in tokens),
                   Viewer2DRefusalCode.FORMAT_INVALID, "CSV fields must be nonempty")
        if columns == 0:
            columns = len(tokens)
            _reject_if(columns > _MAX_DIMENSION, Viewer2DRefusalCode.LIMIT_EXCEEDED,
                       "CSV has too many columns")
        _reject_if(len(tokens) != columns, Viewer2DRefusalCode.FORMAT_INVALID,
                   "CSV rows are ragged")
        _reject_if(rows >= _MAX_DIMENSION, Viewer2DRefusalCode.LIMIT_EXCEEDED,
                   "CSV has too many rows")
        if target is not None:
            for column, token in enumerate(tokens):
                try:
                    value = float(token)
                except ValueError:
                    _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "CSV contains a nonnumeric field")
                _reject_if(math.isinf(value), Viewer2DRefusalCode.FORMAT_INVALID,
                           "infinite values are not supported")
                target[rows, column] = value
        rows += 1
    _shape_fhw((rows, columns))
    return (rows, columns), digest.hexdigest()


def _catalog_csv(path, policy):
    if path.stat().st_size > _MAX_CSV:
        _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "CSV exceeds 512 MiB")
    with open(path, "rb", buffering=0) as stream:
        shape, digest = _csv_scan(stream)
        revision = _descriptor_revision(path, stream)
    _admit(shape)
    return _make_catalog(path, policy, Viewer2DSourceKind.CSV_MATRIX, (0,), shape,
                         np.dtype(float), revision, (), "csv")


def _npy_header(stream):
    _reject_if(stream.read(6) != b"\x93NUMPY", Viewer2DRefusalCode.FORMAT_INVALID,
               "invalid NPY magic")
    version = tuple(stream.read(2))
    _reject_if(version not in ((1, 0), (2, 0), (3, 0)),
               Viewer2DRefusalCode.FORMAT_INVALID, "unsupported NPY version")
    length_bytes = stream.read(2 if version == (1, 0) else 4)
    _reject_if(len(length_bytes) != (2 if version == (1, 0) else 4),
               Viewer2DRefusalCode.FORMAT_INVALID, "truncated NPY header")
    length = int.from_bytes(length_bytes, "little")
    _reject_if(length > 64 * 1024, Viewer2DRefusalCode.LIMIT_EXCEEDED,
               "NPY header exceeds 64 KiB")
    raw = stream.read(length)
    _reject_if(len(raw) != length, Viewer2DRefusalCode.FORMAT_INVALID,
               "truncated NPY header")
    try:
        header = ast.literal_eval(raw.decode("utf-8" if version == (3, 0) else "latin1"))
    except (UnicodeDecodeError, SyntaxError, ValueError):
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "malformed NPY header")
    _reject_if(type(header) is not dict or set(header) != {"descr", "fortran_order", "shape"},
               Viewer2DRefusalCode.FORMAT_INVALID, "NPY header fields are not exact")
    _reject_if(header["fortran_order"] is not False,
               Viewer2DRefusalCode.FORMAT_INVALID, "Fortran-order NPY is not supported")
    shape = header["shape"]
    f, h, w = _shape_fhw(shape)
    dtype = _source_dtype(header["descr"])
    return tuple(shape), dtype, stream.tell(), f * h * w * dtype.itemsize


def _catalog_npy(path, policy):
    size = path.stat().st_size
    if size > _MAX_NPY:
        _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "NPY exceeds 64 GiB")
    with open(path, "rb", buffering=0) as stream:
        before = _descriptor_revision(path, stream)
        stream.seek(0)
        shape, dtype, offset, payload = _npy_header(stream)
        if offset + payload != size:
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "NPY payload does not end at EOF")
        revision = _descriptor_revision(path, stream)
        if revision != before:
            _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                    "NPY changed during catalog metadata")
    f, _, _ = _shape_fhw(shape)
    _admit(shape)
    return _make_catalog(path, policy, Viewer2DSourceKind.NUMPY_ARRAY, range(f),
                         shape, dtype, revision, (), "npy")


@dataclass(frozen=True, slots=True)
class _ZipMember:
    name: str
    method: int
    crc32: int
    compressed: int
    uncompressed: int
    local_offset: int
    data_offset: int


def _zip_members(stream):
    size = os.fstat(stream.fileno()).st_size
    if size > _MAX_NPZ:
        _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "NPZ exceeds 2 GiB")
    tail_size = min(size, 65_557)
    stream.seek(size - tail_size)
    tail = stream.read(tail_size)
    index = tail.rfind(b"PK\x05\x06")
    if index < 0 or len(tail) - index < 22:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "NPZ end record is missing")
    eocd = struct.unpack_from("<4s4H2LH", tail, index)
    _, disk, cd_disk, disk_entries, entries, cd_size, cd_offset, comment = eocd
    if (disk or cd_disk or disk_entries != entries or entries in (0, 0xFFFF)
            or entries > 16 or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF):
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                "NPZ is split, ZIP64, empty, or over member limit")
    if comment > 4096 or index + 22 + comment != len(tail):
        _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                "NPZ archive comment/trailer is invalid")
    eocd_offset = size - tail_size + index
    if cd_size > _MAX_MANIFEST or cd_offset + cd_size != eocd_offset:
        _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                "NPZ central metadata is over limit or misplaced")
    stream.seek(cd_offset)
    central = stream.read(cd_size)
    members = []
    names = set()
    cursor = total = 0
    for _ in range(entries):
        if cursor + 46 > len(central):
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "truncated NPZ central directory")
        values = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        (sig, _made, needed, flags, method, _time, _date, crc, compressed,
         uncompressed, name_len, extra_len, member_comment, member_disk,
         _internal, _external, local_offset) = values
        span = 46 + name_len + extra_len + member_comment
        if sig != b"PK\x01\x02" or cursor + span > len(central):
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "invalid NPZ central member")
        name_raw = central[cursor + 46:cursor + 46 + name_len]
        try:
            name = name_raw.decode("utf-8", "strict")
        except UnicodeDecodeError:
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "NPZ names must be UTF-8")
        if (needed >= 45 or flags & ~0x800 or flags & 1 or method not in (0, 8)
                or member_disk or name_len > 256 or extra_len > 4096
                or not name.endswith(".npy") or "/" in name or "\\" in name
                or not name[:-4] or name in names
                or compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF):
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "unsafe or unsupported NPZ member")
        if total + uncompressed > 4 * 1024**3:
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                    "NPZ declared total exceeds 4 GiB")
        names.add(name)
        total += uncompressed
        stream.seek(local_offset)
        local = stream.read(30)
        if len(local) != 30:
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "truncated NPZ local header")
        lv = struct.unpack("<4s5H3L2H", local)
        lsig, _lv, lflags, lmethod, _lt, _ld, lcrc, lc, lu, ln, lx = lv
        local_name = stream.read(ln)
        local_extra = stream.read(lx)
        data_offset = local_offset + 30 + ln + lx
        if (lsig != b"PK\x03\x04" or (lflags, lmethod, lcrc, lc, lu) !=
                (flags, method, crc, compressed, uncompressed) or local_name != name_raw
                or lx > 4096 or len(local_extra) != lx
                or data_offset + compressed > cd_offset):
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                    "NPZ local/central metadata disagrees")
        members.append(_ZipMember(name, method, crc, compressed,
                                  uncompressed, local_offset, data_offset))
        cursor += span
    if cursor != cd_size:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "NPZ central metadata span disagrees")
    intervals = sorted((item.local_offset, item.data_offset + item.compressed) for item in members)
    if any(end > intervals[i + 1][0] for i, (_, end) in enumerate(intervals[:-1])):
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "NPZ members overlap")
    return tuple(members)


def _zip_scan(stream, member, *, capture_start=None, capture_size=0, prefix_cap=0):
    crc = count = 0
    captured = bytearray()
    try:
        with zipfile.ZipFile(stream) as archive:
            info = archive.getinfo(member.name)
            if ((info.CRC, info.compress_size, info.file_size, info.compress_type) !=
                    (member.crc32, member.compressed, member.uncompressed, member.method)):
                _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "NPZ selected metadata changed")
            with archive.open(info) as stream:
                while chunk := stream.read(1024 * 1024):
                    begin, count = count, count + len(chunk)
                    if count > member.uncompressed:
                        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "NPZ member exceeds declared size")
                    crc = zlib.crc32(chunk, crc)
                    if prefix_cap:
                        captured.extend(chunk[:prefix_cap - len(captured)])
                        if len(captured) >= prefix_cap:
                            return bytes(captured)
                    elif capture_start is not None:
                        left = max(begin, capture_start)
                        right = min(count, capture_start + capture_size)
                        if left < right: captured.extend(chunk[left - begin:right - begin])
    except Viewer2DReadError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile, RuntimeError, zlib.error) as error:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, str(error))
    if count != member.uncompressed or crc & 0xFFFFFFFF != member.crc32:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "NPZ member size or CRC disagrees")
    if capture_start is not None and len(captured) != capture_size:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "selected NPZ frame is truncated")
    return bytes(captured)


def _member_header(stream, member):
    prefix = _zip_scan(stream, member, prefix_cap=min(member.uncompressed, 64 * 1024 + 16))
    shape, dtype, offset, payload = _npy_header(io.BytesIO(prefix))
    if offset + payload != member.uncompressed:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "NPZ NPY payload does not end at member EOF")
    return shape, dtype, offset


def _member_limits(member):
    if (member.uncompressed > _MAX_NPZ or not member.compressed
            or member.uncompressed > 1000 * member.compressed):
        _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                "NPZ selected member exceeds size or ratio cap")


def _catalog_npz(path, policy):
    with open(path, "rb", buffering=0) as stream:
        before = _descriptor_revision(path, stream)
        members = _zip_members(stream)
        by_key = {member.name[:-4]: member for member in members}
        selected = header = None
        for preferred in ("image", "data"):
            if preferred in by_key:
                selected = by_key[preferred]
                _member_limits(selected)
                header = _member_header(stream, selected)
                break
        if selected is None:
            candidates = []
            for member in members:
                try:
                    _member_limits(member)
                    candidate = _member_header(stream, member)
                except Viewer2DReadError:
                    continue
                candidates.append((member, candidate))
                if len(candidates) > 1:
                    break
            if len(candidates) != 1:
                _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                        "NPZ numeric member is ambiguous or absent")
            selected, header = candidates[0]
        shape, dtype, _ = header
        _admit(shape)
        _zip_scan(stream, selected)
        revision = _descriptor_revision(path, stream)
        if revision != before:
            _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                    "NPZ changed during catalog metadata")
    f, _, _ = _shape_fhw(shape)
    return _make_catalog(path, policy, Viewer2DSourceKind.NUMPY_ARRAY, range(f),
                         shape, dtype, revision, (), "npz", member=selected.name)


def _catalog_detector(path, policy):
    suffix = path.suffix.lower()
    size = path.stat().st_size
    try:
        encoded = 0
        if suffix == ".raw":
            from xrd_tools.io.image import infer_raw_detector_shape
            dtype = _source_dtype(policy.raw_dtype)
            shape = policy.raw_detector_shape or infer_raw_detector_shape(
                path, raw_dtype=dtype.str, raw_header_skip=policy.raw_header_skip)
            if shape is None:
                _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                        "RAW detector shape is not uniquely known")
            expected = policy.raw_header_skip + math.prod(shape) * dtype.itemsize
            _reject_if(size != expected, Viewer2DRefusalCode.FORMAT_INVALID,
                       "RAW payload does not match its declared shape")
            frames, format_name = 1, "raw"
        elif suffix in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(path) as handle:
                frames = len(handle.pages)
                _reject_if(not 1 <= frames <= _MAX_FRAMES,
                           Viewer2DRefusalCode.LIMIT_EXCEEDED,
                           "detector frame count is out of range")
                shapes = {tuple(page.shape) for page in handle.pages}
                dtypes = {np.dtype(page.dtype).str for page in handle.pages}
            _reject_if(len(shapes) != 1 or len(dtypes) != 1,
                       Viewer2DRefusalCode.FORMAT_INVALID,
                       "TIFF frames must share shape and dtype")
            shape, dtype, format_name = shapes.pop(), _source_dtype(dtypes.pop()), "tiff"
        else:
            _reject_if(size > _MAX_FABIO, Viewer2DRefusalCode.LIMIT_EXCEEDED,
                       "Fabio detector image exceeds 256 MiB")
            import fabio
            image = fabio.openheader(str(path))
            try:
                shape, frames = tuple(image.shape), int(getattr(image, "nframes", 1))
                dtype = _source_dtype(getattr(image, "dtype",
                                      getattr(image, "_dtype", np.dtype(float))))
            finally:
                image.close()
            encoded, format_name = size, suffix[1:]
    except Viewer2DReadError:
        raise
    except Exception as error:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, str(error))
    _reject_if(len(shape) != 2, Viewer2DRefusalCode.FORMAT_INVALID,
               "detector frames must be two-dimensional")
    _shape_fhw(shape)
    _reject_if(not 1 <= frames <= _MAX_FRAMES, Viewer2DRefusalCode.LIMIT_EXCEEDED,
               "detector frame count is out of range")
    _admit(shape, encoded=encoded)
    return _make_catalog(path, policy, Viewer2DSourceKind.RAW_DETECTOR, range(frames),
                         (frames, *shape) if frames > 1 else shape, dtype,
                         _stable_revision(path), (), format_name)


def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "strict")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", "strict")
    return str(value)


class _HdfWalk:
    __slots__ = ("visits", "external", "path_bytes", "candidates", "attribute_bytes",
                 "processed_frames", "processed_path_bytes", "processed_fields",
                 "processed_field_bytes")

    def __init__(self):
        self.visits = self.external = self.path_bytes = self.candidates = 0
        self.attribute_bytes = 0
        self.processed_frames = self.processed_path_bytes = 0
        self.processed_fields = self.processed_field_bytes = 0

    def touch(self, path, *, external=False):
        encoded = len(path if type(path) is bytes else path.encode("utf-8", "strict"))
        if (self.visits >= 4096 or self.path_bytes + encoded > 4 * 1024**2
                or external and self.external >= 4096):
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "HDF5 bounded walk exceeded")
        self.visits += 1
        self.path_bytes += encoded
        self.external += int(external)

    def retain_candidate(self):
        if self.candidates >= 256:
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "too many HDF5 detector candidates")
        self.candidates += 1

    def retain_external(self):
        if self.external >= 4096:
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "HDF5 bounded walk exceeded")
        self.external += 1

    def retain_processed_frame(self, path):
        encoded = len(path if type(path) is bytes else path.encode("utf-8", "strict"))
        if (self.processed_frames >= _MAX_FRAMES
                or self.processed_path_bytes + encoded > 4 * 1024**2):
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                    "processed HDF5 frame catalog exceeded")
        self.processed_frames += 1
        self.processed_path_bytes += encoded

    def retain_processed_field(self, path):
        encoded = len(path if type(path) is bytes else path.encode("utf-8", "strict"))
        if (self.processed_fields >= _MAX_PROCESSED_FIELDS
                or self.processed_field_bytes + encoded > _MAX_PROCESSED_FIELD_BYTES):
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                    "processed HDF5 field catalog exceeded")
        self.processed_fields += 1
        self.processed_field_bytes += encoded

    def retain_attribute(self, size, *, processed=False):
        if size > 4096:
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                    "selected HDF5 attribute exceeds bounded storage")
        if processed:
            if self.processed_field_bytes + size > _MAX_PROCESSED_FIELD_BYTES:
                _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                        "processed HDF5 fields exceed bounded storage")
            self.processed_field_bytes += size
        else:
            if self.attribute_bytes + size > 128 * 1024:
                _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                        "selected HDF5 attributes exceed bounded storage")
            self.attribute_bytes += size

    def retain_processed_payload(self, size):
        if size > _MAX_PATH or self.processed_field_bytes + size > _MAX_PROCESSED_FIELD_BYTES:
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                    "processed HDF5 fields exceed bounded storage")
        self.processed_field_bytes += size


class _Viewer2DAdmission:
    """One catalog operation's exact revision and resolved-source cache."""

    __slots__ = ("revisions", "catalogs", "active")

    def __init__(self):
        self.revisions = {}
        self.catalogs = {}
        self.active = set()

    def revision(self, path):
        key = str(path)
        revision = self.revisions.get(key)
        if revision is None:
            revision = _stable_revision(path)
            self.revisions[key] = revision
        return revision


def _hnames(group, walk):
    import h5py
    prefix = h5py.h5i.get_name(group.id).rstrip(b"/")
    names = []
    for raw in group.id:
        walk.touch(prefix + b"/" + raw)
        names.append(raw.decode("utf-8", "strict"))
    return names


_HDF_MISSING = object()


def _hdf_dtype_is_fixed(dtype, h5py):
    if (dtype.hasobject or h5py.check_dtype(vlen=dtype) is not None
            or h5py.check_dtype(ref=dtype) is not None):
        return False
    if dtype.fields is not None:
        return all(_hdf_dtype_is_fixed(member[0], h5py)
                   for member in dtype.fields.values())
    if dtype.subdtype is not None:
        return _hdf_dtype_is_fixed(dtype.subdtype[0], h5py)
    return True


def _hattr(owner, name, walk, default=_HDF_MISSING, *, processed=False):
    import h5py
    encoded = name.encode("utf-8", "strict")
    owner_path = h5py.h5i.get_name(owner.id).decode("utf-8", "strict")
    field_path = owner_path + "/@" + name
    if processed:
        walk.retain_processed_field(field_path)
    else:
        walk.touch(field_path)
    if not h5py.h5a.exists(owner.id, encoded):
        if default is not _HDF_MISSING:
            return default
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "required HDF5 attribute is absent")
    attribute = h5py.h5a.open(owner.id, encoded)
    try:
        dtype, storage = attribute.dtype, attribute.get_storage_size()
        vlen = h5py.check_dtype(vlen=dtype)
        reference = h5py.check_dtype(ref=dtype)
        direct_string = (dtype.fields is None and dtype.subdtype is None
                         and vlen in (str, bytes) and reference is None)
        if attribute.shape != () or not direct_string and not _hdf_dtype_is_fixed(dtype, h5py):
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                    "selected HDF5 attribute is not a fixed scalar")
        if direct_string:
            source_type = memory_type = None
            try:
                source_type = attribute.get_type()
                cset = source_type.get_cset()
                if cset not in (h5py.h5t.CSET_ASCII, h5py.h5t.CSET_UTF8):
                    _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                            "selected HDF5 string encoding is unsupported")
                memory_type = h5py.h5t.C_S1.copy()
                memory_type.set_size(4097)
                memory_type.set_strpad(h5py.h5t.STR_NULLPAD)
                memory_type.set_cset(cset)
                destination = np.zeros((), dtype="S4097")
                attribute.read(destination, mtype=memory_type)
                raw = destination.tobytes()
                end = raw.find(b"\0")
                if end < 0:
                    _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                            "selected HDF5 attribute exceeds bounded storage")
                walk.retain_attribute(end, processed=processed)
                encoding = "utf-8" if cset == h5py.h5t.CSET_UTF8 else "ascii"
                return raw[:end].decode(encoding, "strict")
            finally:
                try:
                    if memory_type is not None:
                        memory_type.close()
                finally:
                    if source_type is not None:
                        source_type.close()
        walk.retain_attribute(storage, processed=processed)
        value = np.empty((), dtype=dtype)
        attribute.read(value)
        return value[()]
    finally:
        attribute.close()


def _hget(group, name, walk, *, processed=False):
    import h5py
    full = group.name.rstrip("/") + "/" + name
    if processed:
        walk.retain_processed_field(full)
    else:
        walk.touch(full)
    link = group.get(name, getlink=True)
    if isinstance(link, h5py.ExternalLink):
        walk.retain_external()
    try:
        return group.get(name)
    except Exception:
        return None


def _hpath(handle, path, walk):
    current = handle
    names = path.strip("/").split("/")
    for depth, name in enumerate(names):
        if depth >= 32 and depth < len(names) - 1:
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                    "HDF5 traversal depth exceeds 32")
        current = _hget(current, name, walk)
        if current is None:
            raise KeyError(path)
    return current


def _hdf_entry(handle, walk):
    import h5py
    for name in _hnames(handle, walk):
        obj = _hget(handle, name, walk)
        if isinstance(obj, h5py.Group) and (name == "entry"
                or _decode_scalar(_hattr(obj, "NX_class", walk, "")) == "NXentry"):
            return obj
    return None


def _hdf_dataset(handle, walk, preferred=None):
    import h5py
    if preferred:
        selected_path = "/" + preferred.strip("/")
        try:
            selected = _hpath(handle, selected_path, walk)
        except KeyError:
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                    "selected HDF5 detector dataset is absent")
        if not isinstance(selected, h5py.Dataset) or selected.ndim not in (2, 3):
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                    "selected HDF5 detector dataset is not rank two or three")
        _source_dtype(selected.dtype)
        _shape_fhw(tuple(int(value) for value in selected.shape))
        walk.retain_candidate()
        return selected_path, selected
    candidates, seen, stack = [], set(), [(handle, 0)]
    while stack:
        group, depth = stack.pop()
        if depth > 32:
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "HDF5 traversal depth exceeds 32")
        for name in _hnames(group, walk):
            full = group.name.rstrip("/") + "/" + name
            if depth >= 32 and h5py.h5o.get_info(
                    group.id, name.encode("utf-8", "strict")).type == h5py.h5o.TYPE_GROUP:
                _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                        "HDF5 traversal depth exceeds 32")
            obj = _hget(group, name, walk)
            if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                try:
                    _source_dtype(obj.dtype)
                    _shape_fhw(tuple(int(value) for value in obj.shape))
                except Viewer2DReadError:
                    continue
                walk.retain_candidate()
                candidates.append(full)
            elif isinstance(obj, h5py.Group):
                address = hash(obj.id)
                if address not in seen:
                    seen.add(address)
                    stack.append((obj, depth + 1))
    ordered = [path for path in _RAW_DATASETS if path in candidates]
    signalled = [path for path in candidates if _decode_scalar(
        _hattr(_hpath(handle, path, walk), "signal_type", walk, "")) == "detector"]
    chosen = ordered[:1] or signalled or candidates
    if len(chosen) != 1:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                "HDF5 detector dataset is absent or ambiguous")
    return chosen[0], _hpath(handle, chosen[0], walk)


def _dependency(locator, dataset, *, start=None, stop=None, logical=None,
                admission=None):
    path = _canonical_path(locator)
    revision = (_stable_revision(path) if admission is None
                else admission.revision(path))
    return Viewer2DDependency(str(path), dataset, revision,
                              start, stop, logical)


class _SegmentUnavailable(Viewer2DReadError):
    """One named Eiger segment link resolves to no rank-three dataset."""

    def __init__(self, logical):
        super().__init__(Viewer2DRefusalCode.FORMAT_INVALID,
                         f"Eiger segment is missing or not rank three: {logical}")
        self.logical = logical


def _hdf_segments(path, entry, walk, admission):
    import h5py
    group = _hget(entry, "data", walk)
    if not isinstance(group, h5py.Group):
        return None
    segments = []
    for name in _hnames(group, walk):
        logical = group.name.rstrip("/") + "/" + name
        walk.touch(logical)
        link = group.get(name, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            walk.retain_external()
        if not (isinstance(link, h5py.ExternalLink) and name.startswith("data_")
                and name[5:].isdigit()):
            continue
        try:
            dataset = group.get(name)
        except Exception:
            dataset = None
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
            raise _SegmentUnavailable(logical)
        walk.retain_candidate()
        shape = tuple(int(value) for value in dataset.shape)
        _, h, w = _shape_fhw(shape)
        segments.append((name, logical, Path(dataset.file.filename), dataset.name,
                         shape[0], (h, w), _source_dtype(dataset.dtype)))
    if not segments:
        return None
    segments.sort(key=lambda item: item[0])
    dependencies, total, frame_shape, dtype = [], 0, None, None
    for _, logical, actual, dataset_path, frames, current_shape, current_dtype in segments:
        if ((frame_shape is not None and frame_shape != current_shape)
                or (dtype is not None and dtype.str != current_dtype.str)):
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "Eiger segments disagree")
        if total + frames > _MAX_FRAMES:
            _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "Eiger frame count exceeds 10,000")
        dependencies.append(_dependency(actual, dataset_path, start=total,
            stop=total + frames, logical=logical, admission=admission))
        total, frame_shape, dtype = total + frames, current_shape, current_dtype
    shape = (total, *frame_shape)
    _admit(shape)
    return shape, dtype, tuple(dependencies), segments[0][1]


def _hinted_segments(path, handle, walk, admission, selected_path):
    """The Eiger segment chain the validated dataset hint *selected_path*
    anchors, or ``None`` when the hint is to be read as one dataset.

    A processed record stores the dataset the writer integrated from --
    for an Eiger master its first segment, ``/entry/data/data_000001`` --
    beside a frame index that counts across every segment of the master.
    Resolving the hint alone stopped the catalog at the first segment, so
    every frame past it was "unavailable" and fell back to its thumbnail
    (bo_2 frames >= 1000).  The hint is read as the chain's anchor when
    it names the first segment of an intact chain.  A hint that is not
    named like a segment, whose entry is absent, or that is not the first
    segment of the chain :func:`_hdf_segments` finds (none, when the name
    is an in-file dataset) keeps the single-dataset reading, as does a
    chain broken only at a segment AFTER the hinted one: the hinted
    segment's own frames stay readable.  A chain broken at the hinted
    segment or before it refuses -- the chain the record's frame index
    counts across is not there -- and so does every other chain refusal
    (segments that disagree, a chain over the frame cap).
    """
    import h5py
    match = _SEGMENT_HINT.match(selected_path)
    if match is None:
        return None
    try:
        entry = _hpath(handle, "/" + match.group("entry"), walk)
    except KeyError:
        return None
    if not isinstance(entry, h5py.Group):
        return None
    try:
        eiger = _hdf_segments(path, entry, walk, admission)
    except _SegmentUnavailable as error:
        group, _, name = error.logical.rpartition("/")
        if group != match.group("group") or name <= match.group("name"):
            raise
        return None
    if eiger is None or eiger[3] != selected_path:
        return None
    return eiger


def _processed_frame_groups(frames, walk):
    """Enumerate the owned xdart frame schema outside the foreign-HDF walk cap."""
    import h5py

    prefix = h5py.h5i.get_name(frames.id).rstrip(b"/")
    selected, labels = [], set()
    for raw in frames.id:
        try:
            name = raw.decode("utf-8", "strict")
        except UnicodeDecodeError:
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                    "processed frame name is not UTF-8")
        full = prefix + b"/" + raw
        if not name.startswith("frame_"):
            walk.touch(full)
            continue
        suffix = name[6:]
        if not suffix.isdigit():
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                    "processed frame label is malformed")
        label = int(suffix)
        if label in labels:
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                    "processed frame labels are not unique")
        walk.retain_processed_frame(full)
        labels.add(label)
        link = frames.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                    "processed frame must be an owned hard-link group")
        try:
            group = frames.get(name)
        except Exception:
            group = None
        if isinstance(group, h5py.Group):
            selected.append((label, name, group))
    return tuple(selected)


def _processed_frame_group(frames, label, walk):
    import h5py

    tried = set()
    for name in (f"frame_{label:04d}", f"frame_{label}", f"frame_{label:06d}"):
        if name in tried:
            continue
        tried.add(name)
        link = frames.get(name, getlink=True)
        if isinstance(link, h5py.HardLink):
            selected = frames.get(name)
            if isinstance(selected, h5py.Group):
                return selected
    for candidate, _name, group in _processed_frame_groups(frames, walk):
        if candidate == label:
            return group
    return None


def _processed_catalog(path, policy, handle, entry, walk, admission):
    import h5py
    from xrd_tools.io.read import resolve_source_master

    frames = _hget(entry, "frames", walk)
    if not isinstance(frames, h5py.Group):
        return None
    processed = any(_hget(entry, name, walk) is not None
                    for name in ("integrated_1d", "integrated_2d"))
    facts, dependencies, dep_keys = [], [], set()
    source_base = _decode_scalar(_hattr(entry, "source_base", walk, "")) or None
    if source_base is not None and len(os.fsencode(source_base)) > _MAX_PATH:
        _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                "processed source base is over the path limit")
    for label, _name, group in _processed_frame_groups(frames, walk):
        source = _hget(group, "source", walk, processed=True)
        thumbnail = _hget(group, "thumbnail", walk, processed=True)
        fact = None
        if isinstance(source, h5py.Group):
            path_value = _hget(source, "path", walk, processed=True)
            path_storage = (path_value.id.get_storage_size()
                            if isinstance(path_value, h5py.Dataset) else 0)
            if (not isinstance(path_value, h5py.Dataset) or path_value.size != 1
                    or path_storage > _MAX_PATH):
                _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                        "processed source locator is not a bounded scalar")
            walk.retain_processed_payload(path_storage)
            stored = path_value[()]
            if isinstance(stored, np.ndarray): stored = stored.item()
            raw = resolve_source_master(_decode_scalar(stored), scan_file=path,
                source_base=source_base, source_root=policy.source_root)
            if raw is not None:
                try:
                    hint = _decode_scalar(_hattr(
                        source, "dataset_path", walk, "", processed=True)) or None
                    if hint is not None and len(hint.encode("utf-8")) > _MAX_PATH:
                        _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                                "processed dataset hint is over the path limit")
                    source_catalog = _catalog_resolved(
                        _canonical_path(raw), policy, walk, hint, admission)
                    frame_obj = _hget(source, "frame_index", walk, processed=True)
                    if frame_obj is not None and (not isinstance(frame_obj, h5py.Dataset)
                            or frame_obj.size != 1 or frame_obj.dtype.kind not in "iu"):
                        _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                                "processed source frame is not an integer scalar")
                    frame = int(frame_obj[()]) if frame_obj is not None else label
                    if (source_catalog.format_name == "hdf5-processed"
                            or frame not in source_catalog.frame_labels):
                        _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                                "processed raw source frame is unavailable")
                    fact = Viewer2DFrameFact(label, Viewer2DSourceKind.PROCESSED_RAW,
                        source_catalog.source_shape[-2:], source_catalog.source_dtype,
                        source_catalog.canonical_path, source_catalog.dataset_path, frame,
                        source_catalog_identity=source_catalog.catalog_identity)
                    source_deps = (Viewer2DDependency(source_catalog.canonical_path,
                        source_catalog.dataset_path, source_catalog.primary_revision),
                        *source_catalog.dependencies)
                    for dependency in source_deps:
                        key = repr(dependency)
                        if key not in dep_keys:
                            dep_keys.add(key)
                            dependencies.append(dependency)
                except Viewer2DReadError as error:
                    if error.code in (Viewer2DRefusalCode.LIMIT_EXCEEDED,
                                      Viewer2DRefusalCode.MEMORY_REFUSED):
                        raise
        if fact is None and isinstance(thumbnail, h5py.Dataset) and thumbnail.ndim == 2:
            shape = tuple(int(value) for value in thumbnail.shape)
            _shape_fhw(shape)
            lut_dtype = _decode_scalar(_hattr(
                thumbnail, "dtype", walk, processed=True))
            if (lut_dtype not in ("uint8", "uint16")
                    or np.dtype(thumbnail.dtype).str != np.dtype(lut_dtype).str):
                _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "thumbnail LUT dtype is invalid")
            for attr in ("vmin", "vmax"):
                if not math.isfinite(float(_hattr(
                        thumbnail, attr, walk, processed=True))):
                    _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "thumbnail LUT is nonfinite")
            mask = _hget(group, "thumbnail_mask", walk, processed=True)
            if mask is not None and (not isinstance(mask, h5py.Dataset)
                    or mask.dtype.kind != "b" or tuple(mask.shape) != shape):
                _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "thumbnail mask is invalid")
            fact = Viewer2DFrameFact(label, Viewer2DSourceKind.PROCESSED_THUMBNAIL,
                shape, np.dtype(thumbnail.dtype).str, thumbnail_path=thumbnail.name)
        if fact is not None:
            if len(facts) >= _MAX_FRAMES:
                _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED, "processed catalog exceeds 10,000 frames")
            facts.append(fact)
        processed = processed or fact is not None
    facts.sort(key=lambda value: value.label)
    if not processed:
        return None
    if not facts or len({fact.label for fact in facts}) != len(facts):
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                "processed catalog has no displayable unique labels")
    revision = admission.revision(path)
    return _make_catalog(path, policy, facts[0].source_kind,
        (fact.label for fact in facts), facts[0].shape, facts[0].dtype, revision,
        tuple(dependencies), "hdf5-processed", facts=facts)


def _catalog_hdf5(path, policy, walk=None, preferred=None, admission=None):
    import h5py
    walk = _HdfWalk() if walk is None else walk
    admission = _Viewer2DAdmission() if admission is None else admission
    try:
        with h5py.File(path, "r") as handle:
            from xrd_tools.io.processed_scan_id import (
                has_processed_output_markers_file,
                require_current_processed_groups,
            )

            if preferred:
                # An exact source dataset hint avoids the bounded generic HDF
                # walk, but it cannot turn a processed artifact into raw data.
                if has_processed_output_markers_file(handle):
                    _refuse(
                        Viewer2DRefusalCode.FORMAT_INVALID,
                        "processed source cannot be selected as raw detector data",
                    )
                dataset_path, dataset = _hdf_dataset(handle, walk, preferred)
                eiger = _hinted_segments(path, handle, walk, admission, dataset_path)
                if eiger is not None:
                    shape, dtype, dependencies, dataset_path = eiger
                    return _make_catalog(
                        path, policy, Viewer2DSourceKind.RAW_DETECTOR,
                        range(shape[0]), shape, dtype, admission.revision(path),
                        dependencies, "hdf5-eiger", dataset=dataset_path)
            else:
                entry = _hdf_entry(handle, walk)
                entry_name = (
                    entry.name.lstrip("/") if entry is not None else "entry"
                )
                if has_processed_output_markers_file(handle, entry_name):
                    try:
                        current = require_current_processed_groups(
                            handle, entry_name, container=path,
                        )
                    except ValueError as error:
                        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, str(error))
                    processed = _processed_catalog(
                        path, policy, handle, current.entry, walk, admission,
                    )
                    if processed is None:
                        _refuse(
                            Viewer2DRefusalCode.FORMAT_INVALID,
                            "current processed record has no displayable raw frame or thumbnail",
                        )
                    return processed
                if entry is not None:
                    eiger = _hdf_segments(path, entry, walk, admission)
                    if eiger is not None:
                        shape, dtype, dependencies, dataset_path = eiger
                        return _make_catalog(
                            path, policy, Viewer2DSourceKind.RAW_DETECTOR,
                            range(shape[0]), shape, dtype, admission.revision(path),
                            dependencies, "hdf5-eiger", dataset=dataset_path)
                dataset_path, dataset = _hdf_dataset(handle, walk)
            shape = tuple(int(value) for value in dataset.shape)
            dtype, (f, _, _) = _source_dtype(dataset.dtype), _shape_fhw(shape)
            actual = Path(dataset.file.filename).resolve()
            dependencies = (() if actual == path else
                (_dependency(actual, dataset.name, start=0, stop=f,
                             logical=dataset_path, admission=admission),))
    except Viewer2DReadError:
        raise
    except Exception as error:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, str(error))
    _admit(shape)
    return _make_catalog(path, policy, Viewer2DSourceKind.RAW_DETECTOR, range(f),
        shape, dtype, admission.revision(path), dependencies,
        "hdf5", dataset=dataset_path)


def _catalog_resolved(path, policy, walk=None, preferred=None, admission=None):
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_VIEWER_SUFFIXES:
        _refuse(Viewer2DRefusalCode.UNSUPPORTED_FORMAT, "unsupported 2D Viewer suffix")
    admission = _Viewer2DAdmission() if admission is None else admission
    key = (str(path), preferred)
    cached = admission.catalogs.get(key)
    if cached is not None:
        return cached
    if key in admission.active:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID,
                "viewer source catalog contains a cycle")
    admission.active.add(key)
    walk = _HdfWalk() if suffix in _HDF5_SUFFIXES and walk is None else walk
    descriptor_bound = suffix in {".npy", ".npz"}
    if walk is not None:
        walk.touch(str(path))
    before = (None if descriptor_bound else admission.revision(path)
              if suffix in _HDF5_SUFFIXES else _stable_revision(path))
    try:
        if suffix == ".csv":
            catalog = _catalog_csv(path, policy)
        elif suffix == ".npy":
            catalog = _catalog_npy(path, policy)
        elif suffix == ".npz":
            catalog = _catalog_npz(path, policy)
        elif suffix in _HDF5_SUFFIXES:
            catalog = _catalog_hdf5(
                path, policy, walk, preferred, admission)
        else:
            catalog = _catalog_detector(path, policy)
        if suffix in _HDF5_SUFFIXES:
            if catalog.primary_revision != before:
                _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                        "viewer catalog source changed during metadata census")
            _cert_revision(before, walk)
        elif (not descriptor_bound and (_stable_revision(path) != before
                                        or catalog.primary_revision != before)):
            _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                    "viewer catalog source changed during metadata census")
        for dependency in catalog.dependencies:
            if walk is not None:
                _cert_revision(dependency.revision, walk)
            elif _stable_revision(Path(dependency.locator)) != dependency.revision:
                _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                        "viewer catalog dependency changed during metadata census")
        admission.catalogs[key] = catalog
        return catalog
    finally:
        admission.active.discard(key)


def catalog_viewer_2d(path, *, policy=None):
    policy = Viewer2DFormatPolicy() if policy is None else policy
    if type(policy) is not Viewer2DFormatPolicy:
        raise TypeError("policy must be a Viewer2DFormatPolicy")
    suffix = Path(path).suffix.lower() if isinstance(path, (str, os.PathLike)) else ""
    if suffix not in SUPPORTED_VIEWER_SUFFIXES:
        _refuse(Viewer2DRefusalCode.UNSUPPORTED_FORMAT, "unsupported 2D Viewer suffix")
    try:
        return _catalog_resolved(_canonical_path(path), policy)
    except Viewer2DReadError:
        raise
    except OSError as error:
        _refuse(Viewer2DRefusalCode.PATH_INVALID, str(error))


def _validate_values(array):
    value = np.asarray(array)
    dtype = _source_dtype(value.dtype)
    if value.ndim != 2:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "selected source frame is not two-dimensional")
    if value.dtype.kind in "iu" and value.size:
        if int(value.min()) < -(2**53) or int(value.max()) > 2**53:
            _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "integer source is not exactly representable")
    if value.dtype.kind == "f" and np.isinf(value).any():
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "infinite values are not supported")
    source = np.ascontiguousarray(value)
    canonical = (source if type(source) is np.ndarray
        and source.dtype == np.dtype(float) and source.flags.c_contiguous
        and source.flags.owndata and source.base is None
        else np.array(source, dtype=float, order="C", copy=True))
    canonical.setflags(write=False)
    return (canonical, hashlib.sha256(source.tobytes(order="C")).hexdigest(),
            dtype.str, source.nbytes)


def _finish(catalog, label, index, array, source_digest, *, kind=None, raw="",
            dataset=None, source_frame=None, degraded=False, diagnostic="",
            source_dtype_override=None, source_nbytes_override=None):
    fact = next((item for item in catalog.frame_facts if item.label == label), None)
    expected_shape = fact.shape if fact is not None else catalog.source_shape[-2:]
    expected_dtype = fact.dtype if fact is not None else catalog.source_dtype
    value = np.asarray(array)
    observed_dtype = (source_dtype_override if fact is not None
                      and source_dtype_override is not None else np.dtype(value.dtype).str)
    if tuple(value.shape) != tuple(expected_shape) or observed_dtype != expected_dtype:
        _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                "selected frame no longer matches its certified shape or dtype")
    canonical, selected_digest, source_dtype, source_nbytes = _validate_values(value)
    source_dtype = source_dtype if source_dtype_override is None else source_dtype_override
    source_nbytes = source_nbytes if source_nbytes_override is None else source_nbytes_override
    digest = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    provenance = Viewer2DFrameProvenance(
        catalog.canonical_path, catalog.catalog_identity, catalog.format_name,
        label, index, canonical.shape, canonical.dtype.str, canonical.nbytes,
        digest, source_digest or selected_digest, source_dtype, source_nbytes,
        kind or catalog.source_kind,
        catalog.primary_revision, _selected_dependencies(catalog, label),
        raw, dataset, source_frame,
        degraded_thumbnail=degraded, diagnostic=diagnostic,
    )
    frame = Viewer2DFrame(catalog.catalog_identity, label, canonical, provenance)
    _validate_frame_against_catalog(catalog, label, frame)
    return frame


def _read_numpy(catalog, index):
    path = Path(catalog.canonical_path)
    if catalog.format_name == "npy":
        with open(path, "rb", buffering=0) as stream:
            _cert_stat_revision(
                catalog.primary_revision,
                stream=stream,
                message="NPY descriptor does not match catalog",
            )
            shape, dtype, offset, payload = _npy_header(stream)
            if shape != catalog.source_shape or dtype.str != catalog.source_dtype:
                _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, "NPY header changed")
            _, h, w = _shape_fhw(shape)
            frame_bytes = h * w * dtype.itemsize
            stream.seek(offset + index * frame_bytes)
            raw = stream.read(frame_bytes)
            if len(raw) != frame_bytes:
                _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, "NPY frame truncated")
            _cert_stat_revision(
                catalog.primary_revision,
                stream=stream,
                message="NPY source changed",
            )
        return np.frombuffer(raw, dtype=dtype).reshape(h, w), hashlib.sha256(raw).hexdigest()
    with open(path, "rb", buffering=0) as stream:
        _cert_stat_revision(
            catalog.primary_revision,
            stream=stream,
            message="NPZ source changed",
        )
        members = _zip_members(stream)
        member = next((value for value in members if value.name == catalog.member_name), None)
        if member is None:
            _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, "NPZ selected member changed")
        shape, dtype, offset = _member_header(stream, member)
        if shape != catalog.source_shape or dtype.str != catalog.source_dtype:
            _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, "NPZ selected header changed")
        _, h, w = _shape_fhw(shape)
        frame_bytes = h * w * dtype.itemsize
        raw = _zip_scan(stream, member,
            capture_start=offset + index * frame_bytes, capture_size=frame_bytes)
        _cert_stat_revision(
            catalog.primary_revision,
            stream=stream,
            message="NPZ source changed",
        )
    return np.frombuffer(raw, dtype=dtype).reshape(h, w), hashlib.sha256(raw).hexdigest()


def _cert_revision(revision, walk):
    walk.touch(revision.canonical_path)
    _cert_stat_revision(revision, message="selected viewer dependency changed")


def _cert_dependencies(catalog, label, walk):
    import h5py
    fact = next((item for item in catalog.frame_facts if item.label == label), None)
    dependencies = _selected_dependencies(catalog, label)
    anchor = catalog.canonical_path if fact is None else fact.locator
    for dependency in dependencies:
        if dependency.logical_path is not None:
            with h5py.File(anchor, "r") as handle:
                selected = _hpath(handle, dependency.logical_path, walk)
                actual = Path(selected.file.filename).resolve()
                if (not isinstance(selected, h5py.Dataset)
                        or str(actual) != dependency.locator
                        or selected.name != dependency.dataset_path):
                    _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                            "selected external dependency changed")
        _cert_revision(dependency.revision, walk)
    return dependencies


def _read_hdf_dataset(locator, dataset_path, frame, walk, expected_shape, expected_dtype):
    import h5py
    with h5py.File(locator, "r") as handle:
        dataset = _hpath(handle, dataset_path, walk)
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim not in (2, 3):
            _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                    "selected HDF5 dataset changed")
        selected_shape = tuple(int(value) for value in dataset.shape[-2:])
        if (selected_shape != tuple(expected_shape)
                or np.dtype(dataset.dtype).str != expected_dtype
                or dataset.ndim == 3 and not 0 <= frame < dataset.shape[0]):
            _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                    "selected HDF5 dataset metadata changed")
        return dataset[frame] if dataset.ndim == 3 else dataset[()]


def _cert_processed_pointer(catalog, fact, policy, walk):
    import h5py
    from xrd_tools.io.read import resolve_source_master
    with h5py.File(catalog.canonical_path, "r") as handle:
        entry = _hdf_entry(handle, walk)
        frames = None if entry is None else _hget(entry, "frames", walk)
        selected = (_processed_frame_group(frames, fact.label, walk)
                    if isinstance(frames, h5py.Group) else None)
        source = _hget(selected, "source", walk) if isinstance(selected, h5py.Group) else None
        pointer = _hget(source, "path", walk) if isinstance(source, h5py.Group) else None
        frame = _hget(source, "frame_index", walk) if isinstance(source, h5py.Group) else None
        if not isinstance(pointer, h5py.Dataset):
            _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                    "processed source pointer changed")
        walk.touch(pointer.name)
        stored = pointer[()]
        if isinstance(stored, np.ndarray): stored = stored.item()
        source_base = _decode_scalar(_hattr(entry, "source_base", walk, "")) or None
        resolved = resolve_source_master(_decode_scalar(stored),
            scan_file=Path(catalog.canonical_path), source_base=source_base,
            source_root=policy.source_root)
        hint = _decode_scalar(_hattr(source, "dataset_path", walk, "")) or None
        if frame is not None:
            walk.touch(frame.name)
            source_frame = int(frame[()])
        else:
            source_frame = fact.label
    if (resolved is None or str(Path(resolved).resolve()) != fact.locator
            or source_frame != fact.source_frame
            or hint is not None and hint != fact.dataset_path):
        _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                "processed source pointer changed")


def _read_hdf5(catalog, label, policy, walk):
    import h5py
    fact = next((value for value in catalog.frame_facts if value.label == label), None)
    _cert_revision(catalog.primary_revision, walk)
    dependencies = _cert_dependencies(catalog, label, walk)
    if fact is not None and fact.source_kind is Viewer2DSourceKind.PROCESSED_RAW:
        _cert_processed_pointer(catalog, fact, policy, walk)
        interval = next((item for item in dependencies if item.frame_start is not None), None)
        base = next((item for item in dependencies if item.frame_start is None), None)
        if Path(fact.locator).suffix.lower() in _HDF5_SUFFIXES:
            dependency = interval or base
            frame = (fact.source_frame - interval.frame_start
                     if interval is not None else fact.source_frame)
            array = _read_hdf_dataset(
                dependency.locator, dependency.dataset_path, frame, walk,
                fact.shape, fact.dtype)
            value = (array, fact.source_kind, fact.locator, dependency.dataset_path,
                     fact.source_frame, False, "", "", None, None)
        else:
            source_catalog = _catalog_resolved(
                _canonical_path(fact.locator), policy, walk, fact.dataset_path)
            if source_catalog.catalog_identity != fact.source_catalog_identity:
                _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                        "processed raw dependency changed")
            source = _read_frame(source_catalog, fact.source_frame, policy, walk)
            provenance = source.provenance
            value = (source.array, fact.source_kind, fact.locator,
                     provenance.dataset_path, fact.source_frame, False, "",
                     provenance.source_sha256, provenance.source_dtype,
                     provenance.source_nbytes)
    elif fact is not None:
        with h5py.File(catalog.canonical_path, "r") as handle:
            dataset = _hpath(handle, fact.thumbnail_path, walk)
            lut_dtype = _decode_scalar(_hattr(dataset, "dtype", walk))
            if (not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2
                    or tuple(dataset.shape) != fact.shape
                    or np.dtype(dataset.dtype).str != fact.dtype
                    or np.dtype(lut_dtype).str != fact.dtype):
                _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                        "selected thumbnail metadata changed")
            encoded = np.ascontiguousarray(dataset[()])
            vmin, vmax = (float(_hattr(dataset, name, walk))
                          for name in ("vmin", "vmax"))
            scale = 65535.0 if encoded.dtype == np.dtype("uint16") else 255.0
            array = vmin + encoded.astype(float) / scale * (vmax - vmin)
            mask = _hget(dataset.parent, "thumbnail_mask", walk)
            if mask is not None:
                if (not isinstance(mask, h5py.Dataset) or mask.dtype.kind != "b"
                        or tuple(mask.shape) != fact.shape):
                    _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                            "selected thumbnail mask metadata changed")
                array[np.asarray(mask[()], dtype=bool)] = np.nan
        value = (array, fact.source_kind, "", None, None, True,
                 "Thumbnail preview; raw source unavailable.",
                 hashlib.sha256(encoded.tobytes(order="C")).hexdigest(),
                 encoded.dtype.str, encoded.nbytes)
    else:
        index = catalog.frame_labels.index(label)
        dependency = next(iter(dependencies), None)
        locator = catalog.canonical_path if dependency is None else dependency.locator
        dataset_path = catalog.dataset_path if dependency is None else dependency.dataset_path
        source_frame = index if dependency is None else index - dependency.frame_start
        value = (_read_hdf_dataset(locator, dataset_path, source_frame, walk,
                                   catalog.source_shape[-2:], catalog.source_dtype),
                 catalog.source_kind, locator, dataset_path, source_frame,
                 False, "", "", None, None)
    _cert_dependencies(catalog, label, walk)
    _cert_revision(catalog.primary_revision, walk)
    return value


def _read_detector_frame(catalog, index, policy):
    path = Path(catalog.canonical_path)
    if catalog.format_name == "raw":
        dtype = _source_dtype(policy.raw_dtype)
        with open(path, "rb", buffering=0) as stream:
            _cert_stat_revision(
                catalog.primary_revision,
                stream=stream,
                message="RAW descriptor does not match catalog",
            )
            stream.seek(policy.raw_header_skip)
            expected = math.prod(catalog.source_shape[-2:]) * dtype.itemsize
            raw = stream.read(expected)
            _cert_stat_revision(
                catalog.primary_revision,
                stream=stream,
                message="RAW source changed",
            )
        if len(raw) != expected:
            _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, "RAW source changed")
        return np.frombuffer(raw, dtype=dtype).reshape(catalog.source_shape[-2:]), \
            hashlib.sha256(raw).hexdigest()
    descriptor_bound_tiff = catalog.format_name == "tiff"
    # Fabio's public file-like API copies the whole input while identifying the
    # codec, and individual codecs impose extra stream semantics.  Keep its
    # previous full-revision path until a supported descriptor-bound API exists.
    if not descriptor_bound_tiff:
        _assert_primary(catalog)
    try:
        if descriptor_bound_tiff:
            with open(path, "rb", buffering=0) as stream:
                _cert_stat_revision(
                    catalog.primary_revision,
                    stream=stream,
                    message="TIFF source changed before selected frame read",
                )
                import tifffile
                with tifffile.TiffFile(stream) as handle:
                    page = handle.pages[index]
                    if (tuple(page.shape) != catalog.source_shape[-2:]
                            or np.dtype(page.dtype).str != catalog.source_dtype):
                        _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                                "selected TIFF metadata changed")
                    array = page.asarray()
                _cert_stat_revision(
                    catalog.primary_revision,
                    stream=stream,
                    message="TIFF source changed during selected frame read",
                )
        else:
            import fabio
            if path.stat().st_size > _MAX_FABIO:
                _refuse(Viewer2DRefusalCode.LIMIT_EXCEEDED,
                        "Fabio detector image exceeds 256 MiB")
            if catalog.format_name == "edf":
                header = fabio.edfimage.EdfImage(frames=[])
                header.filename = str(path)
                try:
                    header.readheader(str(path))
                    selected_header = header.get_frame(index)
                    selected_shape = selected_header._shape
                    selected_dtype = selected_header._dtype
                finally:
                    header.close()
            else:
                header = fabio.openheader(str(path))
                try:
                    selected_header = header if index == 0 else header.get_frame(index)
                    selected_shape = selected_header.shape
                    selected_dtype = getattr(selected_header, "dtype", getattr(
                        selected_header, "_dtype", None))
                finally:
                    header.close()
            if (selected_shape is None or selected_dtype is None
                    or tuple(selected_shape) != catalog.source_shape[-2:]
                    or np.dtype(selected_dtype).str != catalog.source_dtype):
                _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                        "selected Fabio metadata changed")
            image = fabio.open(str(path))
            try:
                selected = image if index == 0 else image.get_frame(index)
                array = np.array(selected.data, copy=True)
            finally:
                image.close()
    except Viewer2DReadError:
        raise
    except Exception as error:
        _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, str(error))
    if not descriptor_bound_tiff:
        _assert_primary(catalog)
    return array, ""


def read_viewer_2d_frame(catalog, label, *, policy=None):
    if type(catalog) is not Viewer2DArtifactCatalog:
        raise TypeError("catalog must be an exact Viewer2DArtifactCatalog")
    policy = Viewer2DFormatPolicy() if policy is None else policy
    if type(policy) is not Viewer2DFormatPolicy or policy.identity != catalog.policy_identity:
        _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "frame policy does not match catalog")
    if type(label) is not int or label not in catalog.frame_labels:
        _refuse(Viewer2DRefusalCode.FRAME_UNKNOWN, "frame label is not in the certified catalog")
    try:
        return _read_frame(catalog, label, policy, _HdfWalk())
    except Viewer2DReadError:
        raise
    except (OSError, KeyError, RuntimeError) as error:
        _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, str(error))


def _read_frame(catalog, label, policy, walk):
    index = catalog.frame_labels.index(label)
    fact = next((value for value in catalog.frame_facts if value.label == label), None)
    shape = fact.shape if fact is not None else catalog.source_shape[-2:]
    ledger = viewer_2d_selected_ledger(catalog, label)
    admitted = _admit(shape, encoded=ledger.encoded_retained)
    if admitted != ledger:
        raise AssertionError("selected viewer ledger derivation diverged")
    if catalog.format_name == "csv":
        with open(catalog.canonical_path, "rb", buffering=0) as stream:
            path = Path(catalog.canonical_path)
            opened = _revision(path, os.fstat(stream.fileno()))
            try:
                # A pathname view against the descriptor-recorded catalog
                # revision: comparable only through the win32 ctime seam.
                pathname = _stat_identity(path.stat())
            except OSError:
                _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                        "CSV source is no longer readable")
            if (opened != catalog.primary_revision
                    or pathname != _revision_stat(catalog.primary_revision)):
                _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                        "CSV descriptor does not match catalog")
            first_shape, first_digest = _csv_scan(stream)
            first_revision = _descriptor_revision(path, stream)
            if first_shape != shape or first_revision != catalog.primary_revision:
                _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                        "CSV first pass changed")
            array = np.empty(shape, dtype=float, order="C")
            try:
                checked_shape, digest = _csv_scan(stream, array)
                second_revision = _descriptor_revision(path, stream)
                if (checked_shape != first_shape or digest != first_digest
                        or second_revision != first_revision):
                    _refuse(Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED,
                            "CSV second pass changed")
            except Exception:
                array.fill(0.0)
                del array
                raise
        try:
            return _finish(catalog, label, index, array, digest)
        except Exception:
            array.setflags(write=True)
            array.fill(0.0)
            del array
            raise
    if catalog.format_name in {"npy", "npz"}:
        array, digest = _read_numpy(catalog, index)
        return _finish(catalog, label, index, array, digest)
    if catalog.format_name in {"raw", "tiff", "edf", "cbf", "img", "mar3450"}:
        array, digest = _read_detector_frame(catalog, index, policy)
        return _finish(catalog, label, index, array, digest)
    if catalog.format_name.startswith("hdf5"):
        value = _read_hdf5(catalog, label, policy, walk)
        if type(value) is Viewer2DFrame:
            return value
        (array, kind, raw, dataset, frame, degraded, diagnostic, source_digest,
         source_dtype, source_nbytes) = value
        return _finish(catalog, label, index, array, source_digest, kind=kind, raw=raw,
                       dataset=dataset, source_frame=frame, degraded=degraded,
                       diagnostic=diagnostic, source_dtype_override=source_dtype,
                       source_nbytes_override=source_nbytes)
    _refuse(Viewer2DRefusalCode.FORMAT_INVALID, "unsupported certified reader")


__all__ = [
    "SUPPORTED_VIEWER_SUFFIXES", "CATALOG_RESERVATION", "Viewer2DSourceKind",
    "Viewer2DRefusalCode", "Viewer2DReadError", "Viewer2DFormatPolicy",
    "Viewer2DRevision", "Viewer2DDependency", "Viewer2DFrameFact",
    "Viewer2DArtifactCatalog", "Viewer2DFrameProvenance", "Viewer2DFrame",
    "Viewer2DMemoryLedger", "viewer_2d_memory_ledger", "catalog_viewer_2d",
    "viewer_2d_selected_ledger", "read_viewer_2d_frame",
]
