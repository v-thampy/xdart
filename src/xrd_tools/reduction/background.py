"""Bounded, Qt-free display-background aggregation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable
from typing import Mapping
import hashlib, json, math, os, re, stat, struct
import h5py
import numpy as np

from xrd_tools.core.filters import compile_filter
from xrd_tools.core.metadata import resolve_monitor_norm
from xrd_tools.io.image import read_image
from xrd_tools.io.metadata import read_image_metadata_observed
_DOMAINS = frozenset({"raw", "integrated_1d", "integrated_2d"})
_BLOCK = 65_536
def _shape(value: object, *, ndim: int) -> tuple[int, ...]:
    if (type(value) is not tuple or len(value) != ndim
            or any(type(part) is not int or part <= 0 for part in value)):
        raise ValueError("display-background shape is invalid")
    return value
def _bytes_root(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(contiguous.shape)
    result.setflags(write=False)
    return result
def _is_bytes_backed(array: np.ndarray) -> bool:
    root: object = array
    seen: set[int] = set()
    while isinstance(root, np.ndarray) and id(root) not in seen:
        seen.add(id(root))
        root = root.base
    return isinstance(root, bytes) and not array.flags.writeable
@dataclass(frozen=True, slots=True)
class DisplayBackgroundPlan:
    domain: str
    contributor_ids: tuple[str, ...]
    value_shapes: tuple[tuple[int, ...], ...]
    axis_shapes: tuple[tuple[tuple[int, ...], ...], ...]
    axis_units: tuple[tuple[str, ...], ...]
    differing_grid_1d: bool = False
    def __post_init__(self) -> None:
        count = len(self.contributor_ids)
        ndim = 1 if self.domain == "integrated_1d" else 2
        axis_count = {"raw": 0, "integrated_1d": 1,
                      "integrated_2d": 2}.get(self.domain, -1)
        if (
            self.domain not in _DOMAINS
            or type(self.contributor_ids) is not tuple
            or not self.contributor_ids
            or any(type(item) is not str or not item for item in self.contributor_ids)
            or len(set(self.contributor_ids)) != count
            or type(self.value_shapes) is not tuple or len(self.value_shapes) != count
            or type(self.axis_shapes) is not tuple or len(self.axis_shapes) != count
            or type(self.axis_units) is not tuple or len(self.axis_units) != count
            or type(self.differing_grid_1d) is not bool
            or (self.differing_grid_1d and self.domain != "integrated_1d")
        ):
            raise ValueError("display-background plan is invalid")
        for value_shape, axis_shapes, axis_units in zip(
            self.value_shapes, self.axis_shapes, self.axis_units, strict=True
        ):
            _shape(value_shape, ndim=ndim)
            if (type(axis_shapes) is not tuple or len(axis_shapes) != axis_count
                or type(axis_units) is not tuple or len(axis_units) != axis_count
                or any(type(unit) is not str for unit in axis_units)
            ):
                raise ValueError("display-background axis identity is invalid")
            for axis_shape in axis_shapes:
                _shape(axis_shape, ndim=1)
        if self.domain != "integrated_1d" and any(
                shape != self.value_shapes[0] for shape in self.value_shapes):
            raise ValueError("display-background value shapes differ")
        if any(units != self.axis_units[0] for units in self.axis_units):
            raise ValueError("display-background axis units differ")
@dataclass(frozen=True, slots=True)
class DisplayBackgroundResult:
    domain: str
    contributor_ids: tuple[str, ...]
    values: np.ndarray
    finite_counts: np.ndarray
    axes: tuple[np.ndarray, ...]
    axis_units: tuple[str, ...]
    result_identity: tuple[object, ...]
    diagnostics: tuple[str, ...] = ()
    def __post_init__(self) -> None:
        if (
            self.domain not in _DOMAINS
            or type(self.contributor_ids) is not tuple
            or type(self.values) is not np.ndarray or self.values.dtype != np.dtype(np.float64)
            or type(self.finite_counts) is not np.ndarray or self.finite_counts.dtype != np.dtype(np.uint64)
            or self.values.shape != self.finite_counts.shape
            or type(self.axes) is not tuple or type(self.axis_units) is not tuple
            or len(self.axes) != len(self.axis_units)
            or any(type(axis) is not np.ndarray or axis.ndim != 1 for axis in self.axes)
            or not _is_bytes_backed(self.values)
            or not _is_bytes_backed(self.finite_counts)
            or any(not _is_bytes_backed(axis) for axis in self.axes)
            or type(self.result_identity) is not tuple
            or type(self.diagnostics) is not tuple
            or any(type(item) is not str for item in self.diagnostics)
        ):
            raise ValueError("display-background result is invalid")
def _strict_axis(axis: np.ndarray) -> None:
    if (type(axis) is not np.ndarray or axis.ndim != 1
            or axis.dtype != np.dtype(np.float64) or not axis.flags.c_contiguous):
        raise ValueError("display-background axis is not numeric contiguous 1-D")
    prior: float | None = None
    for item in axis:
        value = float(item)
        if not np.isfinite(value) or prior is not None and value <= prior:
            raise ValueError("display-background axis must be finite increasing")
        prior = value
def _same_axis(first: np.ndarray, second: np.ndarray) -> bool:
    if first.shape != second.shape:
        return False
    for start in range(0, first.size, _BLOCK):
        stop = min(first.size, start + _BLOCK)
        if not np.array_equal(first[start:stop], second[start:stop]):
            return False
    return True
def run_display_background(
    plan: DisplayBackgroundPlan,
    contributors: tuple[tuple[np.ndarray, ...], ...],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> DisplayBackgroundResult:
    """Return the promoted finite mean for one frozen ordered plan."""
    if type(plan) is not DisplayBackgroundPlan:
        raise TypeError("display-background plan must be exact")
    plan.__post_init__()
    if (
        type(contributors) is not tuple
        or len(contributors) != len(plan.contributor_ids)
        or cancelled is not None and not callable(cancelled)
    ):
        raise ValueError("display-background contributors are invalid")
    axis_count = {"raw": 0, "integrated_1d": 1, "integrated_2d": 2}[plan.domain]
    for index, item in enumerate(contributors):
        if type(item) is not tuple or len(item) != axis_count + 1:
            raise ValueError("display-background contributor tuple is invalid")
        value, *axes = item
        if (
            type(value) is not np.ndarray
            or value.dtype.kind not in "fiu"
            or not value.flags.c_contiguous
            or value.shape != plan.value_shapes[index]
            or tuple(axis.shape for axis in axes) != plan.axis_shapes[index]
        ):
            raise ValueError("display-background contributor identity differs")
        for axis in axes:
            _strict_axis(axis)
        if (plan.domain == "integrated_2d"
                and (axes[0].shape != (value.shape[0],)
                     or axes[1].shape != (value.shape[1],))):
            raise ValueError("display-background 2-D axes differ")
        if plan.domain == "integrated_1d" and axes[0].shape != value.shape:
            raise ValueError("display-background 1-D axis differs")
        if (plan.differing_grid_1d and (value.dtype != np.dtype(np.float64)
                                       or axes[0].dtype != np.dtype(np.float64))):
            raise ValueError("differing-grid 1-D requires C float64 inputs")
    reference_axes = contributors[0][1:]
    if plan.domain == "integrated_2d" and any(
            not _same_axis(reference, axis) for item in contributors[1:]
            for reference, axis in zip(reference_axes, item[1:], strict=True)):
        raise ValueError("display-background 2-D grids differ")
    if plan.domain == "integrated_1d":
        differs = any(not _same_axis(reference_axes[0], item[1])
                      for item in contributors[1:])
        if differs != plan.differing_grid_1d:
            raise ValueError("display-background interpolation policy differs")
    shape = plan.value_shapes[0]
    size = int(np.prod(shape))
    sums = np.zeros(size, dtype=np.float64)
    counts = np.zeros(size, dtype=np.uint64)
    work = np.empty(min(size, _BLOCK), dtype=np.float64)
    predicate = np.empty(work.shape, dtype=np.bool_)
    reference = reference_axes[0] if reference_axes else None
    for item in contributors:
        if cancelled is not None and cancelled():
            raise InterruptedError("display-background operation cancelled")
        source = item[0].reshape(-1)
        interpolated = None
        if reference is not None and not _same_axis(reference, item[1]):
            interpolated = np.interp(
                reference, item[1], item[0], left=np.nan, right=np.nan
            )
            source = interpolated
        contributor_finite = 0
        for start in range(0, size, _BLOCK):
            if cancelled is not None and cancelled():
                raise InterruptedError("display-background operation cancelled")
            stop = min(size, start + _BLOCK)
            width = stop - start
            np.copyto(work[:width], source[start:stop], casting="unsafe")
            np.isfinite(work[:width], out=predicate[:width])
            contributor_finite += int(predicate[:width].sum())
            np.add(counts[start:stop], predicate[:width], out=counts[start:stop])
            np.logical_not(predicate[:width], out=predicate[:width])
            np.copyto(work[:width], 0.0, where=predicate[:width])
            np.add(sums[start:stop], work[:width], out=sums[start:stop])
        if contributor_finite == 0:
            raise ValueError("display-background contributor has no finite values")
        del interpolated
    for start in range(0, size, _BLOCK):
        stop = min(size, start + _BLOCK)
        width = stop - start
        np.greater(counts[start:stop], 0, out=predicate[:width])
        np.divide(sums[start:stop], counts[start:stop], out=sums[start:stop],
                  where=predicate[:width])
        np.logical_not(predicate[:width], out=predicate[:width])
        np.copyto(sums[start:stop], np.nan, where=predicate[:width])
    values = _bytes_root(sums.reshape(shape), np.dtype(np.float64))
    finite_counts = _bytes_root(counts.reshape(shape), np.dtype(np.uint64))
    axes = tuple(_bytes_root(axis, np.dtype(np.float64)) for axis in reference_axes)
    identity = ("display-background", plan.domain, plan.contributor_ids,
                plan.value_shapes, plan.axis_shapes, plan.axis_units)
    return DisplayBackgroundResult(
        plan.domain, plan.contributor_ids, values, finite_counts, axes,
        plan.axis_units[0], identity,
        (f"contributors={len(contributors)}", f"finite={int(counts.sum())}"))

_FRAME_MODES = frozenset({"None", "Single BG File", "Series Average", "BG Directory"}); _NONCONTAINER_SUFFIXES = frozenset({".cbf", ".edf", ".img", ".mar3450", ".raw", ".tif", ".tiff"})
_CONTAINER_SUFFIXES = frozenset({".h5", ".hdf5", ".nxs"}); _MATCH_RULES = frozenset({"Scan Root + Frame Number", "Metadata Key"})
_STABLE_POLICY = "stat-sha256-decode-stat-v1"; _MEMBER = re.compile(r"^(.*?)[_-](\d+)$")
_MAX_FILE = 256 * 1024 ** 2; _MAX_PIXELS = 67_108_864
_MAX_MEMBERS = 10_000; _MAX_PATH_BYTES = 4 * 1024 ** 2
_MAX_RECEIPT = 16_384; _MAX_RECEIPTS = 64 * 1024 ** 2
_MAX_FRAME_FACT = 24_576
_MAX_DESCRIPTOR = 262_144
def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
def _path_text(value: object, *, required: bool = True, persisted: bool = False) -> str | None:
    if value is None and not required:
        return None
    canonical = (os.path.abspath(os.path.normpath(value)) if persisted else str(Path(os.path.expanduser(value)).resolve(strict=False))) if type(value) is str else ""
    if type(value) is not str or not value.strip() or value.startswith("//") or persisted and canonical != value or max(len(value.encode("utf-8")), len(canonical.encode("utf-8"))) > 4096:
        raise ValueError("background locator is invalid")
    return canonical
@dataclass(frozen=True, slots=True)
class FrameBackgroundPlan:
    version: int = 1
    mode: str = "None"
    locator: str | None = None
    dataset_path: str | None = None
    frame_index: int | None = None
    match_rule: str | None = None
    metadata_key: str | None = None
    filename_filter: str = ""
    scale: float = 1.0
    normalization_key: str | None = None
    metadata_format: str = "Auto"
    stable_read_policy: str = _STABLE_POLICY
    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1 or type(self.mode) is not str or self.mode not in _FRAME_MODES:
            raise ValueError("frame-background policy version/mode is invalid")
        if type(self.scale) is bool or not isinstance(self.scale, (int, float)) \
                or isinstance(self.scale, int) and abs(self.scale) > float.fromhex("0x1.fffffffffffffp+1023") \
                or not math.isfinite(float(self.scale)):
            raise ValueError("frame-background Scale must be finite")
        object.__setattr__(self, "scale", float(self.scale))
        for name in ("metadata_key", "normalization_key"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value.strip()
                                      or len(value.encode("utf-8")) > 256):
                raise ValueError(f"frame-background {name} is invalid")
        if type(self.filename_filter) is not str:
            raise ValueError("frame-background filter must be text")
        try: compile_filter(self.filename_filter)
        except ValueError as error: raise ValueError("frame-background filter is malformed") from error
        if type(self.metadata_format) is not str or not self.metadata_format.strip():
            raise ValueError("frame-background metadata format is invalid")
        if type(self.stable_read_policy) is not str or self.stable_read_policy != _STABLE_POLICY:
            raise ValueError("frame-background stable-read policy is unsupported")
        parts = (tuple(part for part in self.dataset_path.split("/") if part)
                 if type(self.dataset_path) is str else ())
        if self.dataset_path is not None and (
            type(self.dataset_path) is not str or not self.dataset_path.startswith("/")
            or len(self.dataset_path.encode("utf-8")) > 4096
            or not parts or len(parts) > 256 or self.dataset_path != "/" + "/".join(parts)
        ):
            raise ValueError("frame-background dataset selector is invalid")
        if self.frame_index is not None and (type(self.frame_index) is not int or self.frame_index < 0):
            raise ValueError("frame-background frame index is invalid")
        if self.mode != "None" and self.locator is not None: object.__setattr__(self, "locator", _path_text(self.locator))
    def _complete(self) -> None:
        if self.mode == "None": return
        locator = _path_text(self.locator, persisted=True)
        suffix = Path(locator).suffix.casefold()
        if self.mode == "BG Directory":
            if type(self.match_rule) is not str or self.match_rule not in _MATCH_RULES or (self.match_rule == "Metadata Key") != (self.metadata_key is not None):
                raise ValueError("BG Directory matching policy is incomplete")
            if self.dataset_path is not None or self.frame_index is not None:
                raise ValueError("BG Directory cannot carry a dataset selector")
        elif self.match_rule is not None or self.metadata_key is not None or self.filename_filter != "":
            raise ValueError("non-directory Background cannot carry a match rule")
        if self.mode == "Series Average" and (suffix not in _NONCONTAINER_SUFFIXES or _MEMBER.match(Path(locator).stem) is None):
            raise ValueError("Series Average requires a non-container member")
        if self.mode == "Series Average" and (self.dataset_path is not None or self.frame_index is not None):
            raise ValueError("Series Average cannot carry a dataset selector")
        if self.mode == "Single BG File":
            if suffix in _CONTAINER_SUFFIXES:
                if self.dataset_path is None or self.frame_index is None:
                    raise ValueError("direct container Background requires dataset and frame selectors")
            elif suffix not in _NONCONTAINER_SUFFIXES or self.dataset_path is not None or self.frame_index is not None:
                raise ValueError("Single Background source is unsupported")
    def to_mapping(self) -> dict[str, object]:
        self._complete()
        if self.mode == "None": return {"version": 1, "mode": "None"}
        return {name: getattr(self, name) for name in self.__slots__}
    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "FrameBackgroundPlan":
        if value is None: return cls()
        if not isinstance(value, Mapping): raise ValueError("background provenance must be a mapping")
        if set(value) == {"version", "mode"} and type(value["version"]) is int and type(value["mode"]) is str and dict(value) == {"version": 1, "mode": "None"}: return cls()
        if set(value) != set(cls.__slots__): raise ValueError("background provenance has an invalid keyset")
        raw = dict(value); plan = cls(**{**raw, "locator": None})
        if type(raw["scale"]) is not float or plan.scale != raw["scale"]: raise ValueError("background provenance is not canonical")
        for name in cls.__slots__: object.__setattr__(plan, name, raw[name])
        plan._complete()
        if plan.to_mapping() != raw: raise ValueError("background provenance is not canonical")
        return plan
def _bytes_array(value: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(value, dtype=np.float64)
    result = np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(source.shape)
    result.setflags(write=False)
    return result
@dataclass(frozen=True, slots=True)
class FrameBackgroundResult:
    disposition: str
    background: np.ndarray | None
    descriptor_bytes: bytes | None
    fingerprint: str | None
    diagnostics: tuple[str, ...] = ()
    def __post_init__(self) -> None:
        active = self.disposition == "RESOLVED" and self.background is not None
        pair = self.descriptor_bytes is not None or self.fingerprint is not None
        valid_array = active and type(self.background) is np.ndarray \
            and self.background.dtype == np.dtype(np.float64) and self.background.ndim == 2 \
            and self.background.flags.c_contiguous and _is_bytes_backed(self.background)
        if self.disposition not in {"RESOLVED", "RETRYABLE", "REFUSED", "CANCELLED"} \
                or type(self.diagnostics) is not tuple or len(self.diagnostics) > 1 or any(type(v) is not str or len(v) > 512 for v in self.diagnostics) \
                or active != pair or active != valid_array \
                or pair and (type(self.descriptor_bytes) is not bytes or not self.descriptor_bytes or len(self.descriptor_bytes) > _MAX_DESCRIPTOR or type(self.fingerprint) is not str
                             or _json(json.loads(self.descriptor_bytes.decode("utf-8", "strict"))) != self.descriptor_bytes or hashlib.sha256(self.descriptor_bytes).hexdigest() != self.fingerprint) \
                or not active and any(v is not None for v in (self.background, self.descriptor_bytes, self.fingerprint)):
            raise ValueError("frame-background result is invalid")
def _cancelled(value: Event | Callable[[], bool] | None) -> bool:
    if value is None: return False
    if isinstance(value, Event): return bool(value.is_set())
    if callable(value): return bool(value())
    raise TypeError("cancelled must be an Event or zero-argument predicate")
def _poll(value) -> None:
    if _cancelled(value): raise InterruptedError
def _array_digest(value, cancelled, dtype=None) -> str:
    contiguous = np.ascontiguousarray(value, dtype=dtype); _poll(cancelled); view = memoryview(contiguous).cast("B"); digest = hashlib.sha256()
    for start in range(0, len(view), 1024 ** 2): _poll(cancelled); digest.update(view[start:start + 1024 ** 2])
    _poll(cancelled); return digest.hexdigest()
def _digest(path: Path, cancelled) -> tuple[tuple[int, ...], str]:
    if _cancelled(cancelled): raise InterruptedError
    current = path.stat()
    if not stat.S_ISREG(current.st_mode) or current.st_size > _MAX_FILE:
        raise ValueError("background source is not a bounded regular file")
    identity = (current.st_dev, current.st_ino, current.st_mode, current.st_size,
                current.st_mtime_ns, current.st_ctime_ns)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if _cancelled(cancelled): raise InterruptedError
            chunk = stream.read(min(1024 ** 2, _MAX_FILE + 1 - stream.tell()))
            if not chunk: break
            if stream.tell() > _MAX_FILE: raise RuntimeError("Background source grew beyond its cap")
            digest.update(chunk)
    _poll(cancelled); return identity, digest.hexdigest()
def _changed(path, before, cancelled):
    try: return _digest(path, cancelled) != before
    except InterruptedError: raise
    except (OSError, ValueError): return True
def _guarded_call(call, path, before, cancelled):
    try: return call()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        if _changed(path, before, cancelled): raise RuntimeError("Background source changed during guarded call") from error
        if isinstance(error, RuntimeError): raise ValueError("Background third-party call is stably malformed") from error
        raise
def _one(values, key):
    sentinel = object(); matches = (value for name, value in values.items() if type(name) is str and name.casefold() == key.casefold())
    found = next(matches, sentinel)
    if found is sentinel or next(matches, sentinel) is not sentinel: raise ValueError("required Background metadata is invalid")
    return found
def _tag(value: object) -> tuple[str, object]:
    if isinstance(value, np.generic): value = value.item()
    if type(value) is bool: tagged = ("bool", value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and (not isinstance(value, int) or abs(value) <= float.fromhex("0x1.fffffffffffffp+1023")) and math.isfinite(numeric := float(value)):
        tagged = ("number", numeric.hex())
    elif type(value) is str and len(value) <= 4085 and len(value.encode("utf-8")) <= 4085: tagged = ("text", value)
    elif type(value) is bytes and len(value) <= 2042: tagged = ("bytes", value.hex())
    else: raise ValueError("background metadata scalar is unsupported")
    if len(_json(tagged)) > 4096: raise ValueError("background metadata scalar is too large")
    return tagged
def _untag(value: tuple[str, object]) -> object:
    kind, payload = value
    if kind == "number": return float.fromhex(str(payload))
    if kind == "bytes": return bytes.fromhex(str(payload))
    return payload
def _frame(value: tuple[object, ...], *, persisted=False) -> tuple[int, str, str | None, int, tuple[int, int], tuple]:
    if type(value) is not tuple or len(value) != 6: raise ValueError("frame_fact is invalid")
    label, locator, selector, index, shape, metadata = value
    if type(label) is not int or label < 0 or type(locator) is not str or not locator \
            or len(locator.encode()) > 4096 or persisted and locator.startswith("//") or (os.path.abspath(os.path.normpath(locator)) if persisted else _path_text(locator)) != locator \
            or selector is not None and (type(selector) is not str or len(selector.encode()) > 4096
                or not selector.startswith("/") or not (parts := tuple(part for part in selector.split("/") if part))
                or len(parts) > 256 or selector != "/" + "/".join(parts)) or len(metadata) > 2 \
            or type(index) is not int or index < 0 or type(shape) is not tuple or len(shape) != 2 \
            or any(type(v) is not int or v <= 0 for v in shape) or shape[0] * shape[1] > _MAX_PIXELS \
            or type(metadata) is not tuple or tuple(sorted(metadata)) != metadata:
        raise ValueError("frame_fact is invalid")
    seen = set()
    for item in metadata:
        if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str \
                or not item[0] or len(item[0].encode()) > 256 or item[0] in seen \
                or type(item[1]) is not tuple or _tag(_untag(item[1])) != item[1]:
            raise ValueError("frame_fact metadata is invalid")
        seen.add(item[0])
    if len(_json(value)) > _MAX_FRAME_FACT or len(_json(metadata)) > 12_288:
        raise ValueError("frame_fact exceeds its bounded encoding")
    return label, locator, selector, index, shape, metadata
def _metadata(path: Path, plan: FrameBackgroundPlan, keys: tuple[str, ...], cancelled):
    keys = tuple(sorted(set(keys)))
    if not keys: return (), None
    if _cancelled(cancelled): raise InterruptedError
    discovery = read_image_metadata_observed(path, meta_format=plan.metadata_format); _poll(cancelled)
    if discovery.source_path is None: raise ValueError("required Background metadata is absent")
    source = Path(discovery.source_path).resolve(strict=True); del discovery
    if len(str(source).encode()) > 4096: raise ValueError("Background metadata path exceeds cap")
    before = _digest(source, cancelled)
    if _cancelled(cancelled): raise InterruptedError
    accepted = _guarded_call(lambda: read_image_metadata_observed(path,
        meta_format=plan.metadata_format), source, before, cancelled); failure = None
    if _cancelled(cancelled): raise InterruptedError
    if accepted.source_path is None or Path(accepted.source_path).resolve(strict=True) != source:
        raise RuntimeError("Background metadata source changed")
    try: projected = tuple((key, _tag(_one(accepted.values, key))) for key in keys)
    except (TypeError, ValueError, OverflowError) as error: failure = str(error)[:512]; projected = ()
    del accepted
    if _changed(source, before, cancelled): raise RuntimeError("Background metadata changed while parsing")
    if failure is not None: raise ValueError(failure)
    if len(projected) != len(keys) or len(_json(projected)) > 12_288:
        raise ValueError("required Background metadata is invalid")
    return projected, {"locator": str(source), "state": list(before[0]), "sha256": before[1]}
def _hdf_proof(path: Path, selector: str, frame: int, shape: tuple[int, int], cancelled) -> dict:
    from xrd_tools.io.processed_scan_id import (
        has_processed_output_markers_file,
    )
    components = tuple(part for part in selector.split("/") if part)
    _poll(cancelled)
    with h5py.File(path, "r") as handle:
        _poll(cancelled); processed = has_processed_output_markers_file(handle); _poll(cancelled)
        if processed: raise ValueError("processed xdart source is refused")
        owner: h5py.Group = handle
        for index, name in enumerate(components):
            _poll(cancelled); link = owner.get(name, getlink=True); _poll(cancelled)
            if not isinstance(link, h5py.HardLink):
                raise ValueError("direct HDF dependency is not a hard link")
            value = owner[name]; _poll(cancelled)
            if index + 1 < len(components):
                if not isinstance(value, h5py.Group): raise ValueError("direct HDF ancestor is not a Group")
                owner = value
        dataset = handle[selector]; _poll(cancelled); parent, _, name = selector.rpartition("/")
        if not isinstance(dataset, h5py.Dataset): raise ValueError("direct HDF leaf is not a Dataset")
        text = lambda value: bytes(value).decode("utf-8", "strict") if isinstance(value, (bytes, np.bytes_)) else str(value)
        raw_kind = (parent == "/entry/data" and name.startswith("data_") and
                    len(name) == 11 and name[5:].isdigit() or
                    text(dataset.attrs.get("signal_type", "")).casefold() == "detector" or
                    any(text(handle["/" + "/".join(components[:i])].attrs.get("NX_class", "")).casefold() == "nxdetector"
                        for i in range(1, len(components))))
        _poll(cancelled)
        if dataset.is_virtual \
                or dataset.id.get_create_plist().get_external_count() != 0 \
                or dataset.dtype.kind not in "iuf" or not 1 <= dataset.dtype.itemsize <= 8 \
                or dataset.ndim not in {2, 3} or tuple(dataset.shape[-2:]) != shape \
                or dataset.ndim == 2 and frame != 0 or dataset.ndim == 3 and frame >= dataset.shape[0] \
                or not raw_kind:
            raise ValueError("direct HDF Dataset is not an admitted raw plane")
        if name.startswith("data_") and len(name) == 11 and name[5:].isdigit():
            domain = None
            for key in handle[parent or "/"]:
                _poll(cancelled)
                if key.startswith("data_") and len(key) == 11 and key[5:].isdigit():
                    if domain is not None: raise ValueError("direct HDF decoder domain is not singleton")
                    domain = f"{parent}/{key}"
            _poll(cancelled)
            if domain != selector: raise ValueError("direct HDF decoder domain is not singleton")
        _poll(cancelled); return {"dataset_path": selector, "frame_index": frame,
                "shape": list(dataset.shape), "dtype": dataset.dtype.str}
def _read(path_value: str, plan: FrameBackgroundPlan, shape: tuple[int, int], cancelled):
    path = Path(path_value).resolve(strict=True)
    if len(str(path).encode()) > 4096 or path.suffix.casefold() not in _NONCONTAINER_SUFFIXES | _CONTAINER_SUFFIXES:
        raise ValueError("Background source suffix/path is unsupported")
    before = _digest(path, cancelled)
    proof = None
    if path.suffix.casefold() in _CONTAINER_SUFFIXES:
        try: proof = _hdf_proof(path, plan.dataset_path or "", int(plan.frame_index or 0), shape, cancelled)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            if _changed(path, before, cancelled): raise RuntimeError("Background container changed during proof") from error
            if isinstance(error, RuntimeError): raise ValueError("Background container is stably malformed") from error
            if isinstance(error, OSError): raise error if error.errno is not None else ValueError("Background container is stably malformed")
            raise
    if _cancelled(cancelled): raise InterruptedError
    image = _guarded_call(lambda: read_image(path, frame=int(plan.frame_index or 0), rotation=0,
        detector_shape=shape if path.suffix.casefold() == ".raw" else None,
        dataset_path=plan.dataset_path, exact_frame=plan.dataset_path is not None), path, before, cancelled)
    if _changed(path, before, cancelled): raise RuntimeError("Background source changed while decoding")
    array = np.asarray(image)
    if array.ndim != 2 or array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError("Background detector shape/dtype differs")
    native = np.ascontiguousarray(array); _poll(cancelled)
    decoded = _array_digest(native, cancelled)
    converted = np.asarray(native, dtype=np.float64); _poll(cancelled); return converted, {
        "locator": str(path), "state": list(before[0]), "sha256": before[1],
        "hdf": proof, "decoded": {"shape": list(native.shape), "dtype": native.dtype.str,
                                   "sha256": decoded},
    }
def _manifest(prefix: bytes, receipts, cancelled=None) -> tuple[int, int, str]:
    digest = hashlib.sha256(prefix); count = total = 0
    for receipt in receipts:
        if _cancelled(cancelled): raise InterruptedError
        raw = _json(receipt)
        if len(raw) > _MAX_RECEIPT: raise ValueError("Background receipt exceeds cap")
        total += len(raw)
        if total > _MAX_RECEIPTS: raise ValueError("Background receipt stream exceeds cap")
        digest.update(struct.pack(">Q", len(raw))); digest.update(raw); count += 1
    return count, total, digest.hexdigest()
def _target_metadata(metadata, key: str | None) -> float:
    values = {name: _untag(tag) for name, tag in metadata}
    if key is None: return 1.0
    parsed = resolve_monitor_norm(values, key)
    if parsed is None: raise ValueError("target normalization is missing or invalid")
    return parsed
def _directory_paths(root: Path, target: Path, predicate, cancelled) -> tuple[Path, ...]:
    values = []; path_bytes = 0
    for item in root.iterdir():
        if _cancelled(cancelled): raise InterruptedError
        is_file = item.is_file(); _poll(cancelled)
        if not is_file or item.suffix.casefold() not in _NONCONTAINER_SUFFIXES or not predicate(item.name): continue
        candidate = item.resolve(strict=True); _poll(cancelled)
        if candidate.parent != root or candidate.name != item.name: raise ValueError("Background Directory member escapes its root")
        if candidate == target: continue
        encoded = len(str(candidate).encode())
        if len(values) == _MAX_MEMBERS or encoded > 4096 or path_bytes + encoded > _MAX_PATH_BYTES:
            raise ValueError("Background Directory membership exceeds its cap")
        values.append(candidate); path_bytes += encoded
    return tuple(sorted(values, key=lambda value: (value.name.casefold(), value.name, str(value))))
def _series_paths(value: str, cancelled) -> tuple[Path, ...]:
    anchor = Path(value).resolve(strict=True); parsed = re.match(r"^(.*?)([_-])(\d+)$", anchor.stem)
    if parsed is None: raise ValueError("Background Series anchor is not numbered")
    stem, delimiter, suffix = parsed.group(1), parsed.group(2), anchor.suffix
    pattern = re.compile(rf"^{re.escape(stem)}{re.escape(delimiter)}(\d+){re.escape(suffix)}$", re.IGNORECASE)
    values = []; path_bytes = 0
    for item in anchor.parent.iterdir():
        if _cancelled(cancelled): raise InterruptedError
        match = pattern.match(item.name)
        is_file = item.is_file(); _poll(cancelled)
        if not is_file or match is None: continue
        candidate = item.resolve(strict=True); _poll(cancelled)
        if candidate.parent != anchor.parent or candidate.name != item.name: raise ValueError("Background Series member escapes its root")
        encoded = len(str(candidate).encode())
        if len(values) == _MAX_MEMBERS or encoded > 4096 or path_bytes + encoded > _MAX_PATH_BYTES:
            raise ValueError("Background Series membership exceeds its cap")
        values.append((int(match.group(1)), candidate.name.casefold(), candidate.name,
                       str(candidate), candidate)); path_bytes += encoded
    members = tuple(value[-1] for value in sorted(values))
    if members.count(anchor) != 1: raise ValueError("Background Series anchor is absent or ambiguous")
    return members
def _directory_receipts(candidates, plan, target_tag, target_match, chosen,
                        selected_facts, shape, cancelled):
    matches = 0
    for ordinal, candidate in enumerate(candidates):
        if _cancelled(cancelled): raise InterruptedError
        keys = tuple(sorted(set(key for key in (plan.metadata_key, plan.normalization_key) if key)))
        projected, metadata_source = _metadata(candidate, plan, keys, cancelled) if keys else ((), None)
        if plan.match_rule == "Metadata Key": match = dict(projected).get(plan.metadata_key) == target_tag
        else:
            parsed = _MEMBER.match(candidate.stem)
            match = bool(target_match and parsed and int(parsed.group(2)) == int(target_match.group(2)) and
                         re.search(rf"(?:^|[_-]){re.escape(target_match.group(1))}(?:$|[_-])",
                                   parsed.group(1), re.IGNORECASE))
        if match: matches += 1
        if match != (candidate == chosen): raise RuntimeError("Background Directory match changed")
        receipt = {"ordinal": ordinal, "candidate": str(candidate), "match": match,
                   "metadata_items": projected, "metadata_source": metadata_source}
        if candidate == chosen:
            if (projected, metadata_source) != selected_facts[:2]:
                raise RuntimeError("Background Directory selected metadata changed")
            image, source = _read(str(candidate), plan, shape, cancelled)
            if source != selected_facts[2]: raise RuntimeError("Background Directory selected source changed")
            receipt["source"] = source; del image
        yield receipt
    if matches != 1: raise RuntimeError("Background Directory match changed")
def _result(plan, frame_fact, background, body, cancelled) -> FrameBackgroundResult:
    _poll(cancelled)
    if not np.isfinite(background).any(): raise ValueError("Background contains no finite pixels")
    _poll(cancelled); body["result_sha256"] = _array_digest(background, cancelled, np.float64)
    raw = _json(body)
    cap = _MAX_RECEIPT if plan.mode in {"Series Average", "BG Directory"} else _MAX_DESCRIPTOR
    if len(raw) > cap: raise ValueError("Background descriptor exceeds cap")
    frozen = _bytes_array(background); _poll(cancelled)
    return FrameBackgroundResult("RESOLVED", frozen, raw,
                                 hashlib.sha256(raw).hexdigest())
def resolve_frame_background(
    plan: FrameBackgroundPlan, frame_fact: tuple[object, ...], *,
    cancelled: Event | Callable[[], bool] | None = None,
) -> FrameBackgroundResult:
    """Resolve one bounded Background dependency for notebook, script, or GUI use."""
    if type(plan) is not FrameBackgroundPlan: raise TypeError("frame-background plan must be exact")
    if plan.mode == "None": return FrameBackgroundResult("RESOLVED", None, None, None)
    try:
        if _cancelled(cancelled): raise InterruptedError
        plan._complete()
        label, target, target_selector, target_index, shape, target_items = _frame(frame_fact)
        active_keys = tuple(sorted(set(key for key in (plan.metadata_key, plan.normalization_key) if key)))
        if tuple(name for name, _tagged in target_items) != active_keys: raise ValueError("frame_fact metadata keys differ from policy")
        target_norm = _target_metadata(target_items, plan.normalization_key)
        policy = plan.to_mapping()
        if plan.mode == "Single BG File":
            image, source = _read(plan.locator or "", plan, shape, cancelled)
            metadata, metadata_source = _metadata(Path(source["locator"]), plan,
                tuple(key for key in (plan.normalization_key,) if key), cancelled)
            denominator = _target_metadata(metadata, plan.normalization_key)
            image *= plan.scale * target_norm / denominator
            return _result(plan, frame_fact, image, {"version": 1, "mode": plan.mode,
                "policy": policy, "frame_fact": frame_fact, "source": source,
                "metadata_items": metadata, "metadata_source": metadata_source,
                "decoded": source["decoded"]}, cancelled)
        if plan.mode == "Series Average":
            members = _series_paths(plan.locator or "", cancelled)
            sums = np.zeros(shape, np.float64); counts = np.zeros(shape, np.uint64)
            manifest = hashlib.sha256(b"XDART-FRAME-BG-SERIES-MANIFEST-V1\0")
            receipt_bytes = 0; norm_sum = 0.0; itemsize = None
            for ordinal, member in enumerate(members):
                if _cancelled(cancelled): raise InterruptedError
                member = member.resolve(strict=True)
                image, source = _read(str(member), plan, shape, cancelled)
                observed_itemsize = np.dtype(source["decoded"]["dtype"]).itemsize
                if itemsize not in {None, observed_itemsize}: raise ValueError("Background Series native itemsize differs")
                itemsize = observed_itemsize
                metadata, metadata_source = _metadata(member, plan,
                    tuple(key for key in (plan.normalization_key,) if key), cancelled)
                norm_sum += _target_metadata(metadata, plan.normalization_key)
                finite = np.isfinite(image); np.add(sums, image, out=sums, where=finite); np.add(counts, finite, out=counts)
                raw = _json({"ordinal": ordinal, "source": source,
                             "metadata_items": metadata, "metadata_source": metadata_source})
                if len(raw) > _MAX_RECEIPT or receipt_bytes + len(raw) > _MAX_RECEIPTS:
                    raise ValueError("Background Series receipt stream exceeds cap")
                manifest.update(struct.pack(">Q", len(raw))); manifest.update(raw); receipt_bytes += len(raw)
                del image, finite, source, metadata, metadata_source, raw
            np.divide(sums, counts, out=sums, where=counts > 0); sums[counts == 0] = np.nan
            denominator = norm_sum / len(members); sums *= plan.scale * target_norm / denominator
            return _result(plan, frame_fact, sums, {"version": 1, "mode": plan.mode,
                "policy": policy, "frame_fact": frame_fact, "manifest": {
                    "version": 1, "count": len(members), "receipt_bytes": receipt_bytes,
                    "sha256": manifest.hexdigest()}, "selected": str(Path(plan.locator or "").resolve()),
                "normalization": {"target": _tag(target_norm), "denominator": _tag(denominator)}}, cancelled)
        root = Path(plan.locator or "").resolve(strict=True)
        if not root.is_dir(): raise ValueError("Background Directory is not a directory")
        predicate = compile_filter(plan.filename_filter)
        target_path = Path(target).resolve(strict=False)
        candidates = _directory_paths(root, target_path, predicate, cancelled)
        chosen = None
        target_tag = dict(target_items).get(plan.metadata_key) if plan.metadata_key else None
        target_match = _MEMBER.match(target_path.stem)
        for candidate in candidates:
            if _cancelled(cancelled): raise InterruptedError
            projected = (); metadata_source = None; match = False
            if plan.match_rule == "Metadata Key":
                projected, metadata_source = _metadata(candidate, plan,
                    tuple(sorted(set(key for key in (plan.metadata_key, plan.normalization_key) if key))), cancelled)
                match = dict(projected).get(plan.metadata_key) == target_tag
            else:
                parsed = _MEMBER.match(candidate.stem)
                if target_match and parsed and int(parsed.group(2)) == int(target_match.group(2)):
                    match = re.search(rf"(?:^|[_-]){re.escape(target_match.group(1))}(?:$|[_-])",
                                      parsed.group(1), re.IGNORECASE) is not None
            if match:
                if chosen is not None: raise ValueError("Background Directory match is not unique")
                chosen = (candidate, projected, metadata_source)
        if chosen is None: raise ValueError("Background Directory match is not unique")
        chosen, projected, metadata_source = chosen
        image, source = _read(str(chosen), plan, shape, cancelled)
        if plan.normalization_key and not projected:
            projected, metadata_source = _metadata(chosen, plan, (plan.normalization_key,), cancelled)
        image *= plan.scale * target_norm / _target_metadata(projected, plan.normalization_key)
        prefix = b"XDART-FRAME-BG-DIRECTORY-MANIFEST-V1\0"
        selected_facts = (projected, metadata_source, source)
        first = _manifest(prefix, _directory_receipts(
            candidates, plan, target_tag, target_match, chosen, selected_facts,
            shape, cancelled), cancelled); del candidates
        fresh = _directory_paths(root, target_path, predicate, cancelled)
        second = _manifest(prefix, _directory_receipts(
            fresh, plan, target_tag, target_match, chosen, selected_facts,
            shape, cancelled), cancelled); del fresh
        if first != second: raise RuntimeError("Background Directory membership changed")
        count, receipt_bytes, digest = first
        return _result(plan, frame_fact, image, {"version": 1, "mode": plan.mode,
            "policy": policy, "frame_fact": frame_fact, "manifest": {"version": 1,
                "count": count, "receipt_bytes": receipt_bytes, "sha256": digest},
            "selected": str(chosen), "metadata_items": projected,
            "metadata_source": metadata_source, "decoded": source["decoded"]}, cancelled)
    except InterruptedError:
        return FrameBackgroundResult("CANCELLED", None, None, None, ("cancelled",))
    except (FileNotFoundError, PermissionError, OSError, RuntimeError) as error:
        return FrameBackgroundResult("RETRYABLE", None, None, None, (str(error)[:512],))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        return FrameBackgroundResult("REFUSED", None, None, None, (str(error)[:512],))
__all__ = ["DisplayBackgroundPlan", "DisplayBackgroundResult",
           "run_display_background"]
__all__ += ["FrameBackgroundPlan", "FrameBackgroundResult", "resolve_frame_background"]
