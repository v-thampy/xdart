"""Explicit, confirmed grouping of consecutive detector images per scan point.

Count ratios describe a possible grouping; they never authorize it. This source
policy does no normalization, geometry, or variance estimation. Value policies
are applied to native contributing images before their bounded combination.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xrd_tools.core.scan import ScanFrame, SourceCapabilities, SourceKind, SourceSpec
from xrd_tools.sources.base import BaseFrameSource


@dataclass(frozen=True, slots=True)
class PointImagePolicy:
    """A confirmed operation on each consecutive group of raw images."""

    images_per_point: int
    operation: str
    image_index: int = 1
    reverse: bool = False

    def __post_init__(self):
        if type(self.images_per_point) is not int or self.images_per_point < 1:
            raise ValueError("images per point must be a positive whole count")
        if self.operation not in {"average", "difference", "select"}:
            raise ValueError("point image operation must be average, difference or select")
        if (type(self.image_index) is not int or self.image_index < 1
                or self.image_index > self.images_per_point):
            raise ValueError("selected image index must be between 1 and images per point")
        if type(self.reverse) is not bool:
            raise TypeError("difference reversal must be a boolean")
        if self.operation == "difference" and self.images_per_point != 2:
            raise ValueError("difference requires exactly two images per point")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PointImagePolicy:
        if not isinstance(value, Mapping):
            raise TypeError("point image policy must be a mapping")
        return cls(**dict(value))

    def compatible_with(self, count: int) -> bool:
        """Whether the operation may be re-confirmed for another group size."""
        return (type(count) is int and count > 0 and (
            self.operation == "average"
            or self.operation == "select" and self.image_index <= count
            or self.operation == "difference" and count == 2))

    def raw_indices(self, point: int) -> range | tuple[int]:
        first = point * self.images_per_point
        if self.operation == "select":
            return (first + self.image_index - 1,)
        return range(first, first + self.images_per_point)


@dataclass(frozen=True, slots=True)
class PointImageDetection:
    """Pixel-free evidence of a possible uniform consecutive grouping."""

    scan_label: str
    point_count: int
    image_count: int
    images_per_point: int
    source_spec: SourceSpec


class PointImagesRequired(ValueError):
    """A strict source needs explicit confirmation of a detected grouping."""

    def __init__(self, detection: PointImageDetection):
        self.detection = detection
        super().__init__(f"{detection.image_count} images for {detection.point_count} scan points "
                         f"({detection.images_per_point} images per point); confirm consecutive "
                         "grouping and choose average, difference or one image")


def detect_point_images(spec: SourceSpec, scan_label: str, point_count: int,
                        image_count: int) -> PointImageDetection | None:
    """Return a ratio greater than one, without applying a source policy."""
    if point_count > 0 and image_count > point_count and image_count % point_count == 0:
        return PointImageDetection(str(scan_label), point_count, image_count,
                                   image_count // point_count, spec)
    return None


def _nexus_point_count(spec: SourceSpec, image_count: int) -> int | None:
    """Declared point counts only; never derive points from motor positions."""
    options = dict(spec.options)
    explicit = options.get("scan_point_count")
    if explicit is not None:
        if type(explicit) is not int or explicit <= 0:
            raise ValueError("scan point count must be a positive whole count")
        return explicit
    import h5py
    from xrd_tools.io.bluesky_nexus import resolve_nxentry

    with h5py.File(spec.uri, "r") as stream:
        entry = resolve_nxentry(stream, spec.entry or "entry")
        if entry is None:
            return None
        path = options.get("scan_point_count_path")
        if path is not None:
            if type(path) is not str or not path:
                raise ValueError("scan point count path must be a dataset path")
            dataset = entry.get(path)
            if dataset is None or dataset.shape != () or dataset.dtype.kind not in "iu":
                raise ValueError("scan point count dataset must be one positive whole count")
            value = int(dataset[()])
            if value <= 0:
                raise ValueError("scan point count dataset must be one positive whole count")
            return value
        specific = entry.get("instrument/detector/detectorSpecific")
        if specific is None:
            return None
        counts = []
        for name in ("nimages", "ntrigger"):
            dataset = specific.get(name)
            if dataset is None or dataset.shape != () or dataset.dtype.kind not in "iu":
                return None
            counts.append(int(dataset[()]))
        exposures, points = counts
        return points if exposures > 0 and points > 0 and exposures * points == image_count else None


def inspect_point_images(spec: SourceSpec) -> PointImageDetection | None:
    """Inspect recorded SPEC points / declared NeXus counts and image layouts.

    No detector pixels are decoded. Unknown point counts and non-integral
    ratios yield no suggestion; source opening still applies its strict checks.
    """
    if not isinstance(spec, SourceSpec):
        raise TypeError("point image inspection requires a SourceSpec")
    if spec.kind is SourceKind.SPEC:
        from xrd_tools.io.image import read_detector_image_layout
        from xrd_tools.sources.spec import SpecSource, find_spec_image_files

        options = dict(spec.options)
        metadata = SpecSource(spec.uri, scan=options.get("scan"))
        directory = options.get("image_dir")
        if not directory or metadata.scan_key is None:
            return None
        files = find_spec_image_files(directory, Path(spec.uri).stem, metadata.scan_key,
                                     options.get("image_stem"))
        decode = {key: value for key, value in dict(options.get("read_image_kwargs") or {}).items()
                  if key in {"detector_shape", "detector", "raw_dtype", "raw_header_skip"}}
        count = sum(read_detector_image_layout(path, **decode).frame_count for path in files)
        return detect_point_images(spec, metadata.scan_key, len(metadata.frame_indices), count)
    if spec.kind is SourceKind.TIFF_SERIES:
        from xrd_tools.io.metadata import _extract_scan_info, read_image_metadata_observed
        from xrd_tools.sources.spec import SpecSource, find_spec_image_files

        options = dict(spec.options)
        if options.get("metadata_format", "auto") not in {"spec", "auto"}:
            return None
        files = tuple(Path(value).expanduser() for value in options.get("files", ()))
        if not files:
            return None
        observed = read_image_metadata_observed(files[0], options.get("metadata_format", "auto"),
                                                meta_dir=options.get("meta_dir"))
        stem, scan, _image = _extract_scan_info(files[0])
        if observed.source_path is None or scan is None:
            return None
        metadata = SpecSource(observed.source_path, scan=scan)
        recorded = tuple(find_spec_image_files(files[0].parent, Path(observed.source_path).stem,
                                              metadata.scan_key, stem))
        if tuple(Path(value) for value in recorded) != files:
            raise ValueError("TIFF point grouping must cover the exact recorded scan image series")
        return detect_point_images(spec, metadata.scan_key, len(metadata.frame_indices), len(files))
    if spec.kind in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
        from xrd_tools.sources.descriptor import describe_container

        descriptor = describe_container(Path(spec.uri), entry=spec.entry or "entry")
        points = _nexus_point_count(spec, descriptor.frame_count)
        if points is not None:
            return detect_point_images(spec, descriptor.scan_name, points, descriptor.frame_count)
    return None


@dataclass(frozen=True, slots=True)
class PointImageResult:
    """One owning logical signal and optional native-count Poisson estimate."""

    signal: np.ndarray
    variance: np.ndarray | None
    exposure_multiplier: float


def combine_point_statistics(loader, raw_indices, policy: PointImagePolicy,
                             conditioning=None, saturation: int | None = None, *,
                             poisson: bool = False, native_transform=None) -> PointImageResult:
    """Mask native operands before combining, with independent counting variance.

    A tool may supply its existing native conditioning function. Common dark,
    background and normalization are applied later by that tool, with their
    correction coefficients squared for variance. Only used operands are read.
    """
    from xrd_tools.integrate.stitch_conditioning import condition_stitch_frame

    result = variance = None
    for position, raw_index in enumerate(raw_indices):
        native = np.asarray(loader(raw_index))
        image = (condition_stitch_frame(native, conditioning, saturation)
                 if native_transform is None else native_transform(native))
        image = np.array(image, dtype=np.float64, copy=True, order="C")
        if image.ndim != 2 or image.shape != native.shape:
            raise ValueError("native conditioning must preserve the detector frame shape")
        current_variance = None
        if poisson:
            current_variance = np.maximum(np.asarray(native, dtype=np.float64), 1.)
            current_variance[~np.isfinite(image)] = np.nan
        if result is None:
            result = image
            variance = current_variance
        else:
            if image.shape != result.shape:
                raise ValueError("images within a scan point must have identical detector shapes")
            if policy.operation == "difference":
                # The visible default is Image 2 - Image 1; reverse is Image 1 - Image 2.
                if policy.reverse:
                    np.subtract(result, image, out=result)
                else:
                    np.subtract(image, result, out=result)
            else:
                np.add(result, image, out=result)
            if variance is not None:
                np.add(variance, current_variance, out=variance)
    if result is None:
        raise ValueError("scan point has no contributing images")
    if policy.operation == "average":
        result /= policy.images_per_point
        if variance is not None:
            variance /= policy.images_per_point ** 2
    return PointImageResult(result, variance, 0. if policy.operation == "difference" else 1.)


def combine_point_images(loader, raw_indices, policy: PointImagePolicy,
                         conditioning=None, saturation: int | None = None, *,
                         native_transform=None) -> np.ndarray:
    """Combine bounded native operands without allocating an uncertainty array."""
    return combine_point_statistics(loader, raw_indices, policy, conditioning, saturation,
                                    native_transform=native_transform).signal


def point_image_metadata(policy: PointImagePolicy, locators) -> dict[str, Any]:
    """A JSON-safe source policy plus every exact contributing raw locator."""
    return {**policy.to_dict(), "raw_locators": [
        {"path": str(path), "frame": int(frame)} for path, frame in locators]}


class PointImageSource(BaseFrameSource):
    """Confirmed grouping wrapper for a raw NeXus source; base remains unchanged."""

    def __init__(self, source, spec: SourceSpec):
        self.source = source
        self.path = Path(spec.uri)
        self.entry = spec.entry or "entry"
        self.kind = spec.kind
        self.point_image_policy = PointImagePolicy.from_dict(spec.options["point_images"])
        count = len(source.frame_indices)
        points = _nexus_point_count(spec, count)
        if points is None or points <= 0:
            raise ValueError("confirmed NeXus grouping requires an explicit scan point count")
        if points * self.point_image_policy.images_per_point != count:
            raise ValueError("confirmed images per point does not match image and point counts")
        super().__init__(name=source.name, frame_indices=range(points), spec=spec,
                         capabilities=SourceCapabilities(supports_random_access=True,
                             supports_chunks=True, has_metadata=True, has_raw_references=True,
                             has_scan_manifest=True))

    def _raw_indices(self, index):
        if type(index) is not int or index not in range(len(self._frame_indices)):
            raise IndexError("scan point index is outside the source")
        return self.point_image_policy.raw_indices(index)

    def point_raw_locators(self, index):
        return tuple((self.path, raw) for raw in self._raw_indices(index))

    def raw_locator_for(self, index):
        return self.point_raw_locators(index)[0]

    def load_point_statistics(self, index, conditioning=None, saturation=None, *,
                              poisson=False, native_transform=None):
        native = getattr(self.source, 'load_native_frame', self.source.load_frame)
        return combine_point_statistics(native, self._raw_indices(index),
            self.point_image_policy, conditioning, saturation,
            poisson=poisson, native_transform=native_transform)

    def load_point_frame(self, index, conditioning=None, saturation=None, *, native_transform=None):
        return self.load_point_statistics(index, conditioning, saturation,
                                         native_transform=native_transform).signal

    def iter_point_statistics(self, chunk_size, *, poisson=False, native_transform=None,
                              conditioning=None, saturation=None, cancelled=None):
        """Combine serial native blocks through the existing sustained cursor.

        Only one native block and one unfinished point are held. Detector block
        boundaries may split a point; all used operands are consumed before
        publishing it. The caller charges logical outputs and scratch space.
        """
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError('point chunk size must be a positive whole count')
        stop = (lambda: False) if cancelled is None else cancelled
        k = self.point_image_policy.images_per_point
        method = getattr(self.source, '_iter_vnext_chunks', None)
        blocks = (method(chunk_size * k, stop) if callable(method)
                  else self.source.iter_chunks(chunk_size * k))
        def operands():
            for array, labels in blocks:
                for image, label in zip(array, labels, strict=True):
                    yield int(label), image
        native = operands()
        last, held = -1, None
        def take(wanted):
            nonlocal last, held
            while last < wanted:
                if stop():
                    raise InterruptedError('point-image consumption cancelled')
                try:
                    ordinal, image = next(native)
                except StopIteration as error:
                    raise ValueError('container ended within a confirmed scan point') from error
                if ordinal != last + 1:
                    raise ValueError('container point operands must be contiguous acquisition ordinals')
                last = ordinal
                held = image
            return held
        results, labels = [], []
        try:
            for point in self.frame_indices:
                results.append(combine_point_statistics(take, self._raw_indices(point),
                    self.point_image_policy, conditioning, saturation,
                    poisson=poisson, native_transform=native_transform))
                take((point + 1) * k - 1)
                labels.append(point)
                if len(labels) == chunk_size:
                    yield results, labels
                    results, labels = [], []
            if labels:
                yield results, labels
        finally:
            native.close()
            close = getattr(blocks, 'close', None)
            if callable(close):
                close()

    @property
    def allocation(self):
        return self.source.allocation

    def bind_allocation(self, allocation):
        self.source.bind_allocation(allocation)

    def bind_eiger_direct_chunk(self, allocation):
        self.source.bind_eiger_direct_chunk(allocation)

    def take_direct_chunk_fact(self):
        return self.source.take_direct_chunk_fact()

    def note_direct_chunk_bypass(self, reason, observed_frames=0):
        return self.source.note_direct_chunk_bypass(reason, observed_frames)

    def load_frame(self, index):
        return self.load_point_frame(index)

    def load_native_frame(self, index):
        return self.load_point_frame(index)

    def metadata_for(self, index):
        self._raw_indices(index)  # bounds, independently of the chosen exposure
        first = index * self.point_image_policy.images_per_point
        provider_method = getattr(self.source, '_metadata_provider', None)
        provider = provider_method() if callable(provider_method) else None
        if provider is None:
            result = dict(self.source.metadata_for(first))
        else:
            result = dict(provider.constants())
            for name, values in provider.scan_table().items():
                values = np.asarray(values)
                if len(values) == len(self._frame_indices):
                    position = index
                elif len(values) == len(self.source.frame_indices):
                    position = first
                else:
                    raise ValueError('NeXus counter metadata does not match image or point count')
                result[name] = values[position].item()
        for name, values in self.motors.items():
            result[name] = np.asarray(values)[index].item()
        result["point_images"] = point_image_metadata(self.point_image_policy, self.point_raw_locators(index))
        return result

    @property
    def descriptor(self):
        return self.source.container_descriptor()

    def container_descriptor(self):
        return self.source.container_descriptor()

    @property
    def motors(self):
        out = {}
        for name, values in self.source.motors.items():
            values = np.asarray(values)
            if len(values) == len(self._frame_indices):
                out[name] = values
            elif len(values) == len(self.source.frame_indices):
                out[name] = values[::self.point_image_policy.images_per_point]
            else:
                raise ValueError("NeXus motor metadata does not match image or point count")
        return out

    def frame_for(self, index):
        path, raw = self.raw_locator_for(index)
        return ScanFrame(index=index, metadata=self.metadata_for(index), source_path=path,
                         source_frame_index=raw, loader=lambda frame: self.load_frame(frame.index),
                         source_identity=self.name,
                         point_statistics_loader=self.load_point_statistics)

    def close(self):
        self.source.close()


__all__ = ["PointImagePolicy", "PointImageDetection", "PointImagesRequired",
           "inspect_point_images", "PointImageSource"]
