"""One-open, one-frame processed previews with truthful detector fallback."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
from pathlib import Path, PurePosixPath
import numpy as np
from xrd_tools.core.frame_view import FrameView
from xrd_tools.core.invalid import UINT32_CEILING, saturation_pixels
from xrd_tools.io.frame_view import FrameViewReader
from xrd_tools.io.image import read_image
from xrd_tools.io.read import _decode, resolve_source_master
from xrd_tools.session.hydration import HydrationPurpose, HydrationReadKey
__all__ = ["DetectorPreviewProjection", "FramePreview", "read_frame_preview"]
def _check(condition, message, error=TypeError):
    if not condition:
        raise error(message)
def _source_projection_matches(
    locator, projected, *, source_base, source_root, artifact,
):
    if locator is None or projected is None:
        return locator is projected, False
    stored_text = locator.replace("\\", "/")
    projected_text = projected.replace("\\", "/")
    stored = PurePosixPath(stored_text)
    if ".." in stored.parts:
        return False, False
    try:
        resolved = resolve_source_master(
            locator, scan_file=artifact, source_base=source_base,
            source_root=source_root,
        )
        if resolved is not None:
            return str(resolved).replace("\\", "/") == projected_text, True
    except Exception:
        return False, False
    return stored_text == projected_text, False
@dataclass(frozen=True, slots=True)
class DetectorPreviewProjection:
    mask_available: bool
    mask_bytes: bytes = b""
    mask_dtype: str = "bool"
    mask_shape: tuple[int, ...] = ()
    apply_threshold: bool = False
    threshold_min: float | None = None
    threshold_max: float | None = None
    mask_saturation: bool = False
    saturation_ceiling: float | None = None
    def __post_init__(self):
        for name in ("mask_available", "apply_threshold", "mask_saturation"):
            _check(type(getattr(self, name)) is bool, f"{name} must be an exact bool")
        _check(type(self.mask_bytes) is bytes and type(self.mask_dtype) is str, "mask bytes/dtype are malformed")
        _check(type(self.mask_shape) is tuple and not any(type(value) is not int or value < 0 for value in self.mask_shape), "mask_shape must contain nonnegative integers")
        for name in ("threshold_min", "threshold_max", "saturation_ceiling"):
            value = getattr(self, name)
            if value is not None:
                _check(type(value) in (int, float), f"{name} must be an exact numeric scalar")
                value = float(value)
                _check(math.isfinite(value), f"{name} must be finite", ValueError)
                object.__setattr__(self, name, value)
        _check(self.threshold_min is None or self.threshold_max is None or self.threshold_min <= self.threshold_max, "threshold_min cannot exceed threshold_max", ValueError)
        _check(self.saturation_ceiling is None or self.saturation_ceiling > 0, "saturation_ceiling must be positive", ValueError)
        _check(not self.mask_saturation or self.saturation_ceiling is not None, "mask_saturation requires an accepted detector ceiling", ValueError)
    @classmethod
    def from_mask(cls, mask, **policy):
        array = np.ascontiguousarray(mask, dtype=bool)
        return cls(True, array.tobytes(), array.dtype.str, array.shape, **policy)
    @classmethod
    def without_static_mask(cls, **policy):
        return cls(True, **policy)
    @classmethod
    def unavailable(cls):
        return cls(False)
    def static_mask(self, shape) -> np.ndarray | None:
        _check(self.mask_available, "detector mask truth is unavailable", ValueError)
        if not self.mask_bytes:
            return None
        try:
            mask = np.frombuffer(self.mask_bytes, dtype=np.dtype(self.mask_dtype)).reshape(self.mask_shape)
        except (TypeError, ValueError) as error:
            raise ValueError("stored detector mask bytes do not match shape") from error
        _check(mask.shape == tuple(shape), f"detector mask shape {mask.shape} != raw {tuple(shape)}", ValueError)
        return np.asarray(mask, dtype=bool)
@dataclass(frozen=True, slots=True)
class FramePreview:
    read_key: HydrationReadKey
    view: FrameView
    thumbnail: np.ndarray | None
    raw: np.ndarray | None
    raw_locator: str | None
    raw_dataset_path: str | None
    source_frame_index: int | None
    source_base: str | None
    detector_fallback_used: bool = False
    detector_diagnostic: str | None = None
    def __post_init__(self):
        _check(type(self.read_key) is HydrationReadKey and type(self.view) is FrameView, "preview identity and view must be exact values")
        _check(type(self.view.label) is type(self.read_key.frame_identity) and self.view.label == self.read_key.frame_identity, "view label must match the exact read identity", ValueError)
        for name in ("thumbnail", "raw"):
            value = getattr(self, name)
            _check(value is None or (type(value) is np.ndarray and not value.flags.writeable), f"{name} must be a read-only ndarray or None")
        for name in ("raw_locator", "raw_dataset_path", "source_base"):
            value = getattr(self, name)
            _check(value is None or (type(value) is str and bool(value)), f"{name} must be a nonempty string or None")
        _check(self.source_frame_index is None or (type(self.source_frame_index) is int and self.source_frame_index >= 0), "source_frame_index must be nonnegative or None")
        _check(type(self.detector_fallback_used) is bool, "detector_fallback_used must be an exact bool")
        _check(self.detector_diagnostic is None or (type(self.detector_diagnostic) is str and bool(self.detector_diagnostic)), "detector_diagnostic must be nonempty or None")
        _check(self.thumbnail is self.view.thumbnail, "thumbnail must be the exact view thumbnail", ValueError)
        projection = _source_projection_matches(
            self.raw_locator, self.view.source_path,
            source_base=self.source_base,
            source_root=self.read_key.source_root,
            artifact=self.read_key.artifact_identity,
        )
        _check(
            projection[0]
            and self.source_frame_index == self.view.source_frame_index
            and (self.raw is None or projection[1]),
            "raw provenance must match the exact view", ValueError,
        )
        purpose, has_raw = self.read_key.purpose, self.raw is not None
        requested = purpose is HydrationPurpose.FULL or (purpose is HydrationPurpose.PREVIEW and self.thumbnail is None)
        fallback = False
        _check(self.detector_fallback_used == fallback, "detector fallback flag is inconsistent", ValueError)
        _check(not has_raw or purpose is HydrationPurpose.FULL, "purpose does not authorize detector raw", ValueError)
        _check(not has_raw or self.detector_diagnostic is None, "detector success cannot carry a diagnostic", ValueError)
        _check(not has_raw or (self.raw_locator is not None and self.source_frame_index is not None), "detector success requires exact provenance", ValueError)
        hdf_raw = has_raw and Path(self.raw_locator).suffix.lower() in {".h5", ".hdf5", ".nxs", ".nexus"}
        _check(not hdf_raw or self.raw_dataset_path is not None, "HDF detector success requires a dataset anchor", ValueError)
        _check(has_raw or bool(self.detector_diagnostic) == requested, "detector absence/diagnostic facts are inconsistent", ValueError)
def _source_provenance(reader, frame):
    entry = reader._entry
    group = None if entry is None else entry.get(f"frames/frame_{frame:04d}/source")
    if group is None:
        return None, None, None, reader._source_base, None
    locator = str(_decode(group["path"][()])) if "path" in group else None
    index, diagnostic = None, "raw frame identity unavailable"
    if "frame_index" in group:
        value = np.asarray(group["frame_index"][()])
        if value.size == 1 and value.dtype.kind in "iu" and int(value.ravel()[0]) >= 0:
            index, diagnostic = int(value.ravel()[0]), None
    dataset_path = None
    if "dataset_path" in group.attrs:
        value = _decode(group.attrs["dataset_path"])
        if type(value) is str and value:
            dataset_path = value
        else:
            diagnostic = diagnostic or "raw dataset identity is malformed"
    return locator, index, dataset_path, reader._source_base, diagnostic
def _masked_detector(raw, projection):
    source = np.asarray(raw)
    mask = np.zeros(source.shape, dtype=bool)
    if projection.mask_saturation:
        mask |= (source < 0) | (source >= UINT32_CEILING)
        mask |= saturation_pixels(source, ceiling=projection.saturation_ceiling)
    static = projection.static_mask(source.shape)
    if static is not None: mask |= static
    if projection.apply_threshold:
        if projection.threshold_min is not None: mask |= source < projection.threshold_min
        if projection.threshold_max is not None: mask |= source > projection.threshold_max
    result = np.asarray(source, dtype=float).copy()
    result[mask] = np.nan
    result.setflags(write=False)
    return result
def read_frame_preview(read_key: HydrationReadKey, *, detector_projection: DetectorPreviewProjection | None = None, entry: str = "entry") -> FramePreview:
    """Read one exact processed frame and at most one qualified raw frame."""
    if type(read_key) is not HydrationReadKey:
        raise TypeError("read_key must be an exact HydrationReadKey")
    if detector_projection is not None and type(detector_projection) is not DetectorPreviewProjection:
        raise TypeError("detector_projection must be exact or None")
    if type(read_key.frame_identity) is not int or read_key.frame_identity < 0:
        raise TypeError("preview frame identity must be an exact nonnegative integer")
    frame, artifact = read_key.frame_identity, Path(read_key.artifact_identity)
    with FrameViewReader(
        artifact,
        entry=entry,
        resolve_source=True,
        target_frame=frame,
        source_root=read_key.source_root,
    ) as reader:
        if not reader.has_frame(frame):
            raise KeyError(f"processed frame {frame} is absent")
        view = reader.read(frame)
        provenance = _source_provenance(reader, frame)
        dataset = None if reader._entry is None else reader._entry.get("instrument/detector/detector_shape")
        try: shape_values = np.asarray(dataset[()]) if dataset is not None else np.asarray(())
        except Exception: shape_values = np.asarray(())
        detector_shape = tuple(int(value) for value in shape_values) if shape_values.shape == (2,) and shape_values.dtype.kind in "iu" and np.all(shape_values > 0) else None
    locator, source_index, dataset_path, source_base, source_error = provenance
    extra = {**view.extra, **({"detector_shape": detector_shape} if detector_shape is not None else {})}
    view = replace(view, source_frame_index=source_index, extra=extra)
    raw = diagnostic = None
    if read_key.purpose is HydrationPurpose.PREVIEW and view.thumbnail is None:
        diagnostic = "stored thumbnail unavailable"
    elif read_key.purpose is HydrationPurpose.FULL:
        if detector_projection is None or not detector_projection.mask_available:
            diagnostic = "detector mask/value projection unavailable"
        elif not locator:
            diagnostic = "raw locator unavailable"
        elif source_index is None:
            diagnostic = source_error or "raw frame identity unavailable"
        else:
            resolved = resolve_source_master(
                locator,
                scan_file=artifact,
                source_base=source_base,
                source_root=read_key.source_root,
            )
            if resolved is None:
                diagnostic = "raw source unavailable"
            elif resolved.suffix.lower() in {".h5", ".hdf5", ".nxs", ".nexus"} and dataset_path is None:
                diagnostic = "raw dataset identity unavailable"
            else:
                try:
                    source = read_image(resolved, frame=source_index, dataset_path=dataset_path, preserve_dtype=True, exact_frame=True)
                    raw = _masked_detector(source, detector_projection)
                except Exception as error:
                    diagnostic = str(error) or type(error).__name__
    fallback = bool(raw is not None and read_key.purpose is HydrationPurpose.PREVIEW and view.thumbnail is None)
    return FramePreview(read_key, view, view.thumbnail, raw, locator, dataset_path, source_index, source_base, fallback, diagnostic)
