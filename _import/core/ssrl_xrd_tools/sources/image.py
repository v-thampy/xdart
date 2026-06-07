"""File and series frame sources built on existing image readers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ssrl_xrd_tools.core.scan import ScanFrame, SourceCapabilities, SourceKind, SourceSpec
from ssrl_xrd_tools.io.image import count_frames, read_image
from ssrl_xrd_tools.io.metadata import read_image_metadata
from ssrl_xrd_tools.sources.base import BaseFrameSource


class ImageFileSource(BaseFrameSource):
    """FrameSource for a single detector file readable by ``read_image``."""

    kind = SourceKind.IMAGE_FILE

    def __init__(
        self,
        path: str | Path,
        *,
        detector_shape: tuple[int, int] | None = None,
        detector: str | tuple[int, int] | None = None,
        metadata_format: str | None = None,
        frame_indices: Sequence[int] | None = None,
    ) -> None:
        self.path = Path(path)
        self.detector_shape = detector_shape
        self.detector = detector
        self.metadata_format = metadata_format
        if frame_indices is None:
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
            )
        )

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        if self.metadata_format is None:
            return {}
        return read_image_metadata(self.path, self.metadata_format)

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
    ) -> None:
        self.files = [Path(p) for p in files]
        self.metadata_format = metadata_format
        super().__init__(
            name=name or (self.files[0].stem if self.files else "tiff_series"),
            frame_indices=range(1, len(self.files) + 1),
            spec=SourceSpec(str(self.files[0]) if self.files else "", SourceKind.TIFF_SERIES),
            capabilities=SourceCapabilities(
                supports_random_access=True,
                supports_chunks=True,
                has_metadata=metadata_format is not None,
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
    ) -> "TiffSeriesSource":
        return cls(sorted(Path(directory).glob(pattern)), metadata_format=metadata_format)

    def _path_for(self, index: int) -> Path:
        return self.files[self.frame_indices.index(int(index))]

    def load_frame(self, index: int) -> np.ndarray:
        return np.asarray(read_image(self._path_for(index)))

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        if self.metadata_format is None:
            return {}
        return read_image_metadata(self._path_for(index), self.metadata_format)

    def frame_for(self, index: int) -> ScanFrame:
        path = self._path_for(index)
        return ScanFrame(
            index=int(index),
            metadata=dict(self.metadata_for(index)),
            source_path=path,
            source_frame_index=0,
            loader=lambda frame: read_image(frame.source_path),
            source_identity=str(path),
        )


__all__ = ["ImageFileSource", "TiffSeriesSource"]
