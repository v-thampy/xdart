"""NeXus and processed-scan frame sources."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from xrd_tools.core.frame_view import FrameView
from xrd_tools.core.scan import ScanFrame, SourceCapabilities, SourceKind, SourceSpec
from xrd_tools.io.frame_view import FrameViewReader
from xrd_tools.io.nexus import open_nexus_image_stack
from xrd_tools.sources.base import BaseFrameSource
from xrd_tools.sources.metadata_provider import MetadataProvider

_UNSET: Any = object()


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

    def open_cursor(self):
        """A context-managed :class:`~xrd_tools.sources.cursor.ContainerCursor`
        for sustained consumption: one open handle supplies descriptor,
        wavelength, metadata, frame count, single-frame reads, and block reads.
        Sustained consumers (and :meth:`iter_chunks`) use ONE cursor for their
        whole window; one-off :meth:`load_frame` uses a short-lived cursor."""
        from xrd_tools.sources.cursor import ContainerCursor

        return ContainerCursor(self.path, entry=self.entry)

    def load_frame(self, index: int) -> np.ndarray:
        # One-off: a short-lived cursor (one open handle), native dtype.
        with self.open_cursor() as cursor:
            return np.asarray(cursor.read_frame(int(index)))

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[np.ndarray, list[int]]]:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")
        labels = self.frame_indices
        # ONE cursor for the whole consumption window (no per-chunk reopen).
        with self.open_cursor() as cursor:
            for start in range(0, len(labels), chunk_size):
                chunk_labels = labels[start:start + chunk_size]
                block = cursor.read_block(start, start + len(chunk_labels))
                yield np.asarray(block.array), chunk_labels

    def frame_for(self, index: int) -> ScanFrame:
        return ScanFrame(
            index=int(index),
            metadata=dict(self.metadata_for(index)),
            source_path=self.path,
            source_frame_index=int(index),
            loader=lambda frame: self.load_frame(frame.source_frame_index or 0),
            source_identity=str(self.path),
        )

    # -- Bluesky / apstools NXWriter per-frame metadata --------------------
    # A Bluesky ``.nxs`` classifies as a NEXUS_STACK (raw image stack), so its
    # per-frame scan columns (motors + ion-chamber/photodiode counters + EPOCH)
    # would otherwise be invisible to Plot Metadata.  Surface them through the
    # R2 lazy metadata provider (motors as whole-array columns via ``motors``;
    # counters/EPOCH per-frame via ``metadata_for``): a plain image stack gets an
    # EmptyMetadataProvider (untouched), a Bluesky stack a BlueskyMetadataProvider
    # with identical column semantics.  Built once through a short-lived cursor
    # and MATERIALIZED before the cursor closes, so repeated metadata reads touch
    # only memory and reopen no master.
    def _metadata_provider(self) -> MetadataProvider | None:
        cache = self.__dict__.get("_provider_cache", _UNSET)
        if cache is not _UNSET:
            return cache
        provider: MetadataProvider | None = None
        try:
            with self.open_cursor() as cursor:
                p = cursor.metadata_provider()
                p.scan_table()  # force materialization while the handle is open
                provider = p
        except Exception:
            provider = None
        self._provider_cache = provider
        return provider

    @property
    def motors(self) -> dict[str, np.ndarray]:
        provider = self._metadata_provider()
        return dict(provider.motors()) if provider is not None else {}

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        provider = self._metadata_provider()
        if provider is None:
            return {}
        return dict(provider.metadata_for(int(index)))


class ProcessedNexusSource(BaseFrameSource):
    """Source of reduced :class:`FrameView` records from processed NeXus."""

    kind = SourceKind.PROCESSED_NEXUS

    def __init__(self, path: str | Path, *, entry: str = "entry",
                 source_root: str | Path | None = None) -> None:
        self.path = Path(path)
        self.entry = entry
        # N1: repoint a moved raw tree (overrides the stored @source_base) so
        # load_frame resolves the full-res master after the data relocates.
        self.source_root = source_root
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
        with FrameViewReader(self.path, entry=self.entry,
                             include_thumbnail=include_thumbnail,
                             source_root=self.source_root) as reader:
            return reader.read(int(index))

    def iter_views(self, *, include_thumbnail: bool = True) -> Iterator[FrameView]:
        with FrameViewReader(self.path, entry=self.entry,
                             include_thumbnail=include_thumbnail,
                             source_root=self.source_root) as reader:
            for idx in self.frame_indices:
                yield reader.read(idx)

    def load_frame(self, index: int) -> np.ndarray:
        """STRICT full-resolution raw load via the per-frame source pointer.

        Resolves the relative ``source/path`` against ``@source_base`` /
        ``source_root`` (absolute back-compat) and reads the full-res master.
        A headless analysis consumer (RSM / stitching / fitting) reading a
        processed ``.nxs`` as a FrameSource must NEVER silently get a downsampled,
        mask-baked THUMBNAIL in place of the raw — that would analyze preview
        data.  So ``allow_thumbnail=False``: if the master can't be resolved this
        raises ``KeyError`` (a clean error), rather than degrading.  The display
        path keeps the thumbnail fallback via
        :func:`xrd_tools.io.image_source.load_processed_raw_or_thumbnail`.
        """
        from xrd_tools.io.read import get_raw_frame
        return np.asarray(
            get_raw_frame(self.path, int(index), entry=self.entry,
                          allow_thumbnail=False, source_root=self.source_root),
            dtype=float,
        )

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        return self.read_view(index, include_thumbnail=False).metadata_raw

    def frame_for(self, index: int) -> ScanFrame:
        """Attach the ORIGINAL raw-master pointer (carried by the FrameView's
        ``source_path``/``source_frame_index``) so a stitch/RSM built from a
        processed ``.nxs`` persists resolvable contributing-frame records — the
        raw popup resolves the true master two hops out (stitch.nxs → this
        processed.nxs's per-frame source pointer → the master), not this
        already-reduced file.  (One reader open per frame; harvest is a one-time
        per-result step, not a hot loop.)"""
        view = self.read_view(int(index), include_thumbnail=False)
        return ScanFrame(
            index=int(index),
            metadata=dict(view.metadata_raw),
            source_path=view.source_path,
            source_frame_index=view.source_frame_index,
            loader=lambda fr: self.load_frame(int(index)),
            source_identity=str(self.path),
        )


__all__ = ["NexusStackSource", "ProcessedNexusSource"]
