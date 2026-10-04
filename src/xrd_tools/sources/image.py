"""File and series frame sources built on existing image readers."""

from __future__ import annotations

import fnmatch
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from xrd_tools.core.scan import ScanFrame, SourceCapabilities, SourceKind, SourceSpec
from xrd_tools.io.image import count_frames, find_image_files, read_image
from xrd_tools.io.metadata import read_image_metadata_observed
from xrd_tools.sources.base import BaseFrameSource


class ImageFileSource(BaseFrameSource):
    """FrameSource for a single detector file readable by ``read_image``."""

    kind = SourceKind.IMAGE_FILE

    def __init__(
        self,
        path: str | Path,
        *,
        detector_shape: tuple[int, int] | None = None,
        detector: str | tuple[int, int] | None = None,
        raw_dtype: str = "int32",
        raw_header_skip: int = 0,
        metadata_format: str | None = None,
        meta_dir: str | Path | None = None,
        frame_indices: Sequence[int] | None = None,
    ) -> None:
        self.path = Path(path)
        self.detector_shape = detector_shape
        self.detector = detector
        self.raw_dtype = str(raw_dtype)
        self.raw_header_skip = int(raw_header_skip)
        self.metadata_format = metadata_format
        self.meta_dir = meta_dir
        if frame_indices is None:
            if self.path.suffix.lower() == ".raw":
                # Beamline RAW files are one frame per file.  Asking fabio for a
                # frame count cannot succeed without the operator-supplied
                # shape and produces a misleading warning before the real read.
                n = 1
            else:
                try:
                    n = int(count_frames(self.path))
                except Exception:
                    n = 1
            frame_indices = range(max(n, 1))
        super().__init__(
            name=self.path.stem,
            frame_indices=frame_indices,
            spec=SourceSpec(self.path, SourceKind.IMAGE_FILE),
            capabilities=SourceCapabilities(
                supports_random_access=True,
                supports_chunks=True,
                has_metadata=metadata_format is not None,
                # cheap, image-free metadata ⇒ a headless prepare() can sweep
                # the whole-scan extent (ADR-0006).
                has_scan_manifest=metadata_format is not None,
                has_raw_references=True,
            ),
        )

    def load_frame(self, index: int) -> np.ndarray:
        return np.asarray(
            read_image(
                self.path,
                frame=int(index),
                detector_shape=self.detector_shape,
                detector=self.detector,
                raw_dtype=self.raw_dtype,
                raw_header_skip=self.raw_header_skip,
                preserve_dtype=True,
            )
        )

    def metadata_for(
        self, index: int, *, max_input_bytes: int | None = None,
        spec_point_snapshot: bool = False,
    ) -> Mapping[str, Any]:
        if self.metadata_format is None:
            return {}
        return dict(read_image_metadata_observed(
            self.path, self.metadata_format, meta_dir=self.meta_dir,
            max_input_bytes=max_input_bytes,
            spec_point_snapshot=spec_point_snapshot,
        ).values)

    def frame_for(self, index: int) -> ScanFrame:
        return ScanFrame(
            index=int(index),
            metadata=dict(self.metadata_for(index)),
            source_path=self.path,
            source_frame_index=int(index),
            loader=lambda frame: self.load_frame(frame.source_frame_index or 0),
            source_identity=str(self.path),
        )


class TiffSeriesSource(BaseFrameSource):
    """FrameSource over an ordered TIFF-like file series."""

    kind = SourceKind.TIFF_SERIES

    def __init__(
        self,
        files: Sequence[str | Path],
        *,
        name: str | None = None,
        metadata_format: str | None = "txt",
        meta_dir: str | Path | None = None,
        detector_shape: tuple[int, int] | None = None,
        detector: str | tuple[int, int] | None = None,
        raw_dtype: str = "int32",
        raw_header_skip: int = 0,
        admitted_motor_values: Sequence[tuple[str, str, float]] = (),
        point_images=None,
    ) -> None:
        self.files = [Path(p) for p in files]
        self.metadata_format = metadata_format
        self.meta_dir = meta_dir
        self.detector_shape = detector_shape
        self.detector = detector
        self.raw_dtype = str(raw_dtype)
        self.raw_header_skip = int(raw_header_skip)
        self._admitted_motor_by_index: dict[int, tuple[str, float]] = {}
        self._observed_metadata_sources: dict[int, Path | None] = {}
        snapshots = tuple(admitted_motor_values)
        if snapshots:
            if len(snapshots) != len(self.files):
                raise ValueError(
                    "admitted motor values must cover every TIFF member"
                )
            motor_name: str | None = None
            for index, (path, snapshot) in enumerate(
                zip(self.files, snapshots),
                start=1,
            ):
                if type(snapshot) not in {tuple, list} or len(snapshot) != 3:
                    raise TypeError(
                        "admitted motor value must be a "
                        "path/motor/value triple"
                    )
                source_path, motor, raw_value = snapshot
                if (
                    type(source_path) is not str
                    or not os.path.isabs(source_path)
                    or type(motor) is not str
                    or not motor
                    or motor == "Manual"
                    or type(raw_value) is not float
                ):
                    raise TypeError("admitted motor value is invalid")
                value = raw_value
                if not math.isfinite(value):
                    raise ValueError("admitted motor value must be finite")
                actual = Path(os.path.abspath(path.expanduser()))
                admitted_path = Path(
                    os.path.abspath(Path(source_path).expanduser())
                )
                if admitted_path != actual:
                    raise ValueError(
                        "admitted motor value path does not match TIFF member"
                    )
                if motor_name is not None and motor != motor_name:
                    raise ValueError(
                        "admitted motor values must use one exact motor"
                    )
                motor_name = motor
                self._admitted_motor_by_index[index] = (motor, value)
        self._path_by_index = {
            int(index): path
            for index, path in zip(range(1, len(self.files) + 1), self.files)
        }
        self.point_image_policy = None
        self._point_source = None
        if point_images is not None:
            from xrd_tools.sources.spec import SpecSource, resolve_spec_image_scan
            from xrd_tools.sources.point_images import PointImagePolicy
            if not self.files or self.metadata_format not in {"spec", "auto"}:
                raise ValueError("TIFF point grouping requires recorded SPEC point metadata")
            resolved = resolve_spec_image_scan(self.files[0], meta_dir=self.meta_dir)
            if resolved is None:
                raise ValueError("TIFF point grouping requires recorded SPEC point metadata")
            spec_path, scan_number = resolved
            self.point_image_policy = PointImagePolicy.from_dict(point_images)
            self._point_source = SpecSource(spec_path, scan=scan_number,
                image_dir=self.files[0].parent, require_complete_images=True, point_images=point_images,
                read_image_kwargs={"detector_shape": detector_shape, "detector": detector,
                    "raw_dtype": raw_dtype, "raw_header_skip": raw_header_skip})
            if tuple(Path(path) for path, _frame in self._point_source._frame_map) != tuple(self.files):
                raise ValueError("TIFF point grouping must cover the exact recorded scan image series")
            self.metadata_format = "spec"
        super().__init__(
            name=name or (self.files[0].stem if self.files else "tiff_series"),
            frame_indices=range(1, (len(self._point_source.frame_indices) if self._point_source is not None else len(self.files)) + 1),
            spec=SourceSpec(str(self.files[0]) if self.files else "", SourceKind.TIFF_SERIES),
            capabilities=SourceCapabilities(
                supports_random_access=True,
                supports_chunks=True,
                has_metadata=metadata_format is not None,
                # per-file sidecar metadata is cheap + image-free ⇒ a headless
                # prepare() can sweep the whole-scan incidence extent (ADR-0006).
                has_scan_manifest=metadata_format is not None,
                has_raw_references=True,
            ),
        )

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        pattern: str = "*.tif*",
        metadata_format: str | None = "txt",
        meta_dir: str | Path | None = None,
        detector_shape: tuple[int, int] | None = None,
        detector: str | tuple[int, int] | None = None,
        raw_dtype: str = "int32",
        raw_header_skip: int = 0,
        admitted_motor_values: Sequence[tuple[str, str, float]] = (),
    ) -> "TiffSeriesSource":
        files = [
            path for path in find_image_files(directory)
            if fnmatch.fnmatch(path.name, pattern)
        ]
        return cls(
            files,
            metadata_format=metadata_format,
            meta_dir=meta_dir,
            detector_shape=detector_shape,
            detector=detector,
            raw_dtype=raw_dtype,
            raw_header_skip=raw_header_skip,
            admitted_motor_values=admitted_motor_values,
        )

    def _path_for(self, index: int) -> Path:
        try:
            return self._path_by_index[int(index)]
        except KeyError as exc:
            raise IndexError(f"frame {index} is not in TIFF series {self.name!r}") from exc

    def load_frame(self, index: int) -> np.ndarray:
        if self._point_source is not None:
            return self._point_source.load_frame(int(index) - 1)
        return self._read_path(self._path_for(index))

    def point_raw_locators(self, index):
        if self._point_source is None:
            return ((self._path_for(index), 0),)
        return self._point_source.point_raw_locators(int(index) - 1)

    def load_point_statistics(self, index, *args, **kwargs):
        if self._point_source is None:
            raise ValueError("TIFF source has no confirmed point grouping")
        return self._point_source.load_point_statistics(int(index) - 1, *args, **kwargs)

    def load_point_frame(self, index, *args, **kwargs):
        return self.load_point_statistics(index, *args, **kwargs).signal

    def iter_point_statistics(self, chunk_size, *, poisson=False, native_transform=None,
                              conditioning=None, saturation=None, cancelled=None):
        if self._point_source is None:
            raise ValueError("TIFF source has no confirmed point grouping")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("point chunk size must be a positive whole count")
        statistics, labels = [], []
        for label in self.frame_indices:
            if cancelled is not None and cancelled():
                raise InterruptedError("point-image consumption cancelled")
            statistics.append(self.load_point_statistics(label, conditioning, saturation,
                poisson=poisson, native_transform=native_transform))
            labels.append(label)
            if len(labels) == chunk_size:
                yield statistics, labels
                statistics, labels = [], []
        if labels:
            yield statistics, labels

    def _read_path(self, path: str | Path) -> np.ndarray:
        return np.asarray(read_image(
            path,
            detector_shape=self.detector_shape,
            detector=self.detector,
            raw_dtype=self.raw_dtype,
            raw_header_skip=self.raw_header_skip,
            preserve_dtype=True,
        ))

    def metadata_for(
        self, index: int, *, max_input_bytes: int | None = None,
        spec_point_snapshot: bool = False,
    ) -> Mapping[str, Any]:
        if self._point_source is not None:
            return self._point_source.metadata_for(int(index) - 1)
        path = self._path_for(index)
        if self.metadata_format is None:
            metadata = {}
            self._observed_metadata_sources[int(index)] = None
        else:
            observed = read_image_metadata_observed(
                path, self.metadata_format, meta_dir=self.meta_dir,
                max_input_bytes=max_input_bytes,
                spec_point_snapshot=spec_point_snapshot,
            )
            metadata = dict(observed.values)
            self._observed_metadata_sources[int(index)] = observed.source_path
        admitted = self._admitted_motor_by_index.get(int(index))
        if admitted is not None:
            motor, value = admitted
            folded = motor.casefold()
            for key in tuple(metadata):
                if type(key) is str and key.casefold() == folded:
                    del metadata[key]
            metadata[motor] = value
        return metadata

    def frame_for(self, index: int) -> ScanFrame:
        if self._point_source is not None:
            frame = self._point_source.frame_for(int(index) - 1)
            frame.index = int(index)
            frame.loader = lambda current: self.load_frame(current.index)
            frame.point_statistics_loader = self.load_point_statistics
            return frame
        path = self._path_for(index)
        return ScanFrame(
            index=int(index),
            metadata=dict(self.metadata_for(index)),
            source_path=path,
            source_frame_index=0,
            loader=lambda frame: self._read_path(frame.source_path),
            source_identity=str(path),
        )


__all__ = ["ImageFileSource", "TiffSeriesSource"]
