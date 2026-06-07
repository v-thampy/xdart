"""NeXus and processed-scan frame sources."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ssrl_xrd_tools.core.frame_view import FrameView
from ssrl_xrd_tools.core.scan import ScanFrame, SourceCapabilities, SourceKind, SourceSpec
from ssrl_xrd_tools.io.frame_view import FrameViewReader
from ssrl_xrd_tools.io.nexus import open_nexus_image_stack
from ssrl_xrd_tools.sources.base import BaseFrameSource


class NexusStackSource(BaseFrameSource):
    """FrameSource over a raw image stack in a NeXus/HDF5/Eiger master file."""

    kind = SourceKind.NEXUS_STACK

    def __init__(self, path: str | Path, *, entry: str = "entry") -> None:
        self.path = Path(path)
        self.entry = entry
        with open_nexus_image_stack(self.path, entry) as stack:
            n = int(stack.shape[0])
        super().__init__(
            name=self.path.stem,
            frame_indices=range(n),
            spec=SourceSpec(self.path, SourceKind.NEXUS_STACK, entry=entry),
            capabilities=SourceCapabilities(
                supports_random_access=True,
                supports_chunks=True,
                has_raw_references=True,
            ),
        )

    def load_frame(self, index: int) -> np.ndarray:
        with open_nexus_image_stack(self.path, self.entry) as stack:
            return np.asarray(stack[int(index)])

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[np.ndarray, list[int]]]:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")
        labels = self.frame_indices
        with open_nexus_image_stack(self.path, self.entry) as stack:
            for start in range(0, len(labels), chunk_size):
                chunk_labels = labels[start:start + chunk_size]
                yield np.asarray(stack[start:start + len(chunk_labels)]), chunk_labels

    def frame_for(self, index: int) -> ScanFrame:
        return ScanFrame(
            index=int(index),
            source_path=self.path,
            source_frame_index=int(index),
            loader=lambda frame: self.load_frame(frame.source_frame_index or 0),
            source_identity=str(self.path),
        )


class ProcessedNexusSource(BaseFrameSource):
    """Source of reduced :class:`FrameView` records from processed NeXus."""

    kind = SourceKind.PROCESSED_NEXUS

    def __init__(self, path: str | Path, *, entry: str = "entry") -> None:
        self.path = Path(path)
        self.entry = entry
        with FrameViewReader(self.path, entry=entry, include_thumbnail=False) as reader:
            labels = reader.labels()
        super().__init__(
            name=self.path.stem,
            frame_indices=labels,
            spec=SourceSpec(self.path, SourceKind.PROCESSED_NEXUS, entry=entry),
            capabilities=SourceCapabilities(
                supports_random_access=True,
                supports_chunks=False,
                has_metadata=True,
                has_geometry=True,
                has_raw_references=True,
                has_thumbnails=True,
            ),
        )

    def read_view(self, index: int, *, include_thumbnail: bool = True) -> FrameView:
        with FrameViewReader(self.path, entry=self.entry, include_thumbnail=include_thumbnail) as reader:
            return reader.read(int(index))

    def iter_views(self, *, include_thumbnail: bool = True) -> Iterator[FrameView]:
        with FrameViewReader(self.path, entry=self.entry, include_thumbnail=include_thumbnail) as reader:
            for idx in self.frame_indices:
                yield reader.read(idx)

    def load_frame(self, index: int) -> np.ndarray:
        view = self.read_view(index, include_thumbnail=True)
        if view.raw is not None:
            return np.asarray(view.raw)
        if view.thumbnail is not None:
            return np.asarray(view.thumbnail)
        raise ValueError(f"processed frame {index} has no raw image or thumbnail")

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        return self.read_view(index, include_thumbnail=False).metadata_raw


__all__ = ["NexusStackSource", "ProcessedNexusSource"]
