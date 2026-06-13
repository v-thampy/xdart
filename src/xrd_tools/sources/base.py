"""Shared source helpers for architecture-v2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from xrd_tools.core.scan import (
    FrameSource,
    Scan,
    ScanFrame,
    SourceCapabilities,
    SourceKind,
    SourceSpec,
)


class BaseFrameSource(ABC):
    """Convenience base for concrete :class:`FrameSource` implementations."""

    kind: SourceKind = SourceKind.UNKNOWN

    def __init__(
        self,
        *,
        name: str,
        frame_indices: Sequence[int],
        spec: SourceSpec | None = None,
        capabilities: SourceCapabilities | None = None,
    ) -> None:
        labels = [int(i) for i in frame_indices]
        if len(labels) != len(set(labels)):
            raise ValueError(f"{name} contains duplicate frame labels")
        self.name = str(name)
        self._frame_indices = labels
        self.spec = spec
        self._capabilities = capabilities or SourceCapabilities()

    @property
    def frame_indices(self) -> list[int]:
        return list(self._frame_indices)

    @property
    def capabilities(self) -> SourceCapabilities:
        return self._capabilities

    @abstractmethod
    def load_frame(self, index: int) -> np.ndarray:
        ...

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        return {}

    def scan_manifest(self) -> list[tuple[int, Mapping[str, Any]]] | None:
        """Cheap METADATA-ONLY pass: ``(frame_index, metadata)`` for every frame
        (ADR-0006).  MUST NOT load detector images.  Returns ``None`` when the
        whole-scan metadata cannot be cheaply enumerated — distinct from a real
        empty scan ``[]`` — so a caller treats ``None`` as "unverifiable extent,
        warn-and-proceed."  The default is gated on the ``has_scan_manifest``
        capability; a source flips it True only when ``metadata_for`` is cheap +
        image-free for every frame."""
        if not self._capabilities.has_scan_manifest:
            return None
        return [(idx, dict(self.metadata_for(idx))) for idx in self.frame_indices]

    def frame_for(self, index: int) -> ScanFrame:
        return ScanFrame(
            index=int(index),
            metadata=dict(self.metadata_for(index)),
            loader=lambda frame: self.load_frame(frame.index),
            source_identity=self.name,
        )

    def iter_frames(self) -> Iterator[ScanFrame]:
        for idx in self.frame_indices:
            yield self.frame_for(idx)

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[np.ndarray, list[int]]]:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")
        labels = self.frame_indices
        for start in range(0, len(labels), chunk_size):
            chunk_labels = labels[start:start + chunk_size]
            yield np.stack([self.load_frame(idx) for idx in chunk_labels]), chunk_labels

    def to_scan(self, *, name: str | None = None, **kwargs: Any) -> Scan:
        return Scan(name or self.name, list(self.iter_frames()), **kwargs)


def ensure_frame_source(source: FrameSource | Scan | BaseFrameSource) -> FrameSource:
    """Return *source* as a :class:`FrameSource` or raise a clear error."""

    if isinstance(source, FrameSource):
        return source
    raise TypeError(f"object does not implement FrameSource: {type(source)!r}")


__all__ = ["BaseFrameSource", "ensure_frame_source"]
