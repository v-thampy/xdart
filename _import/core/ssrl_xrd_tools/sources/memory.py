"""In-memory and live frame sources."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Any

import numpy as np

from ssrl_xrd_tools.core.scan import Scan, ScanFrame, SourceCapabilities, SourceKind
from ssrl_xrd_tools.sources.base import BaseFrameSource


class MemoryFrameSource(BaseFrameSource):
    """FrameSource over an existing list of arrays or :class:`ScanFrame`s."""

    kind = SourceKind.MEMORY

    def __init__(self, frames, *, name: str = "memory") -> None:
        scan_frames: list[ScanFrame] = []
        for i, item in enumerate(frames):
            if isinstance(item, ScanFrame):
                scan_frames.append(item)
            else:
                scan_frames.append(ScanFrame(index=i, image=np.asarray(item)))
        self._frames = {int(f.index): f for f in scan_frames}
        super().__init__(
            name=name,
            frame_indices=sorted(self._frames),
            capabilities=SourceCapabilities(
                supports_random_access=True,
                supports_chunks=True,
                has_metadata=True,
                has_geometry=True,
            ),
        )

    def load_frame(self, index: int) -> np.ndarray:
        return np.asarray(self._frames[int(index)].load_image())

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        return self._frames[int(index)].metadata

    def frame_for(self, index: int) -> ScanFrame:
        return self._frames[int(index)]

    def to_scan(self, *, name: str | None = None, **kwargs: Any) -> Scan:
        return Scan(name or self.name, [self._frames[i] for i in self.frame_indices], **kwargs)


class LiveFrameSource(BaseFrameSource):
    """Appendable, thread-safe source for live acquisition."""

    kind = SourceKind.LIVE

    def __init__(self, *, name: str = "live") -> None:
        self._lock = RLock()
        self._frames: dict[int, ScanFrame] = {}
        super().__init__(
            name=name,
            frame_indices=[],
            capabilities=SourceCapabilities(
                is_streaming=True,
                supports_random_access=True,
                supports_chunks=True,
                has_metadata=True,
                has_geometry=True,
            ),
        )

    @property
    def frame_indices(self) -> list[int]:
        with self._lock:
            return sorted(self._frames)

    def append(self, frame: ScanFrame | np.ndarray, *, index: int | None = None,
               metadata: Mapping[str, Any] | None = None) -> ScanFrame:
        with self._lock:
            if isinstance(frame, ScanFrame):
                scan_frame = frame
            else:
                if index is None:
                    index = (max(self._frames) + 1) if self._frames else 0
                scan_frame = ScanFrame(
                    index=int(index),
                    image=np.asarray(frame),
                    metadata=dict(metadata or {}),
                )
            if int(scan_frame.index) in self._frames:
                raise ValueError(f"duplicate live frame index {scan_frame.index}")
            self._frames[int(scan_frame.index)] = scan_frame
            return scan_frame

    def load_frame(self, index: int) -> np.ndarray:
        with self._lock:
            frame = self._frames[int(index)]
        return np.asarray(frame.load_image())

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        with self._lock:
            return dict(self._frames[int(index)].metadata)

    def frame_for(self, index: int) -> ScanFrame:
        with self._lock:
            return self._frames[int(index)]


__all__ = ["LiveFrameSource", "MemoryFrameSource"]
