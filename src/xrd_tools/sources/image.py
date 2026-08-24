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
from xrd_tools.io.metadata import read_image_metadata, read_image_metadata_observed
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
    ) -> Mapping[str, Any]:
        if self.metadata_format is None:
            return {}
        return read_image_metadata(
            self.path, self.metadata_format, meta_dir=self.meta_dir,
            max_input_bytes=max_input_bytes,
        )

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
        super().__init__(
            name=name or (self.files[0].stem if self.files else "tiff_series"),
            frame_indices=range(1, len(self.files) + 1),
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
        return self._read_path(self._path_for(index))

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
    ) -> Mapping[str, Any]:
        path = self._path_for(index)
        if self.metadata_format is None:
            metadata = {}
            self._observed_metadata_sources[int(index)] = None
        else:
            observed = read_image_metadata_observed(
                path, self.metadata_format, meta_dir=self.meta_dir,
                max_input_bytes=max_input_bytes,
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
