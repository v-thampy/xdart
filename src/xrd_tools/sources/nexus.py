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

    def __init__(self, path: str | Path, *, entry: str = "entry",
                 cursor: Any | None = None) -> None:
        self.path = Path(path)
        self.entry = entry
        # ``cursor`` is an owner-thread-only handoff for a caller that already
        # opened the sustained source cursor (the benchmark's descriptor ->
        # metadata -> public-streaming route).  It is consumed and closed by
        # iter_chunks()/close(); no h5py object leaves the producer thread.
        self._consumption_cursor = cursor
        if cursor is None:
            with open_nexus_image_stack(self.path, entry) as stack:
                n = int(stack.shape[0])
        else:
            descriptor = cursor.descriptor
            if descriptor.path != self.path:
                raise ValueError(
                    f"cursor path {descriptor.path} does not match source {self.path}")
            n = int(descriptor.frame_count)
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

    #: H10-C2-B: the coordinator's exact allocation, bound before any read.
    #: This source NEVER resolves one; the legacy route leaves it ``None``.
    allocation = None

    def container_descriptor(self):
        """The pixel-free descriptor a coordinator derives requirements from."""
        cursor = self._consumption_cursor
        if cursor is not None:
            return cursor.descriptor
        with self.open_cursor() as cursor:
            return cursor.descriptor

    def bind_allocation(self, allocation) -> None:
        """Adopt the coordinator's EXACT allocation before reading.  Rebinding
        is IDENTITY-qualified: the same object is idempotent, an equal-but-
        distinct one rejects - the contract is one shared identity, not a value."""
        if self.allocation is not None and self.allocation is not allocation:
            raise ValueError(
                "NexusStackSource is already bound to a different allocation")
        self.allocation = allocation

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

    def close(self) -> None:
        """Release an optional owner-thread consumption cursor early."""
        cursor, self._consumption_cursor = self._consumption_cursor, None
        if cursor is not None:
            cursor.close()

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[np.ndarray, list[int]]]:
        """Yield source-owner blocks bounded by the descriptor's ``ReadPlan``.

        ``chunk_size`` remains a consumer/progress cap, never permission to
        decode a larger source block than the native-byte budget permits.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")

        cursor = self._consumption_cursor
        if cursor is not None:
            self._consumption_cursor = None
            try:
                yield from self._iter_cursor_chunks(cursor, chunk_size)
            finally:
                cursor.close()
            return

        # ONE cursor for the whole consumption window (no per-chunk reopen).
        with self.open_cursor() as cursor:
            yield from self._iter_cursor_chunks(cursor, chunk_size)

    def _iter_cursor_chunks(self, cursor, chunk_size: int) -> Iterator[
            tuple[np.ndarray, list[int]]]:
        """Yield a byte-planned sequence from an open owner-thread cursor."""
        from xrd_tools.core.staging import source_block_budget_bytes
        from xrd_tools.sources.read_plan import plan_reads

        labels = self.frame_indices
        if len(labels) != cursor.frame_count:
            # A source can only trim its contiguous tail for a bounded public
            # run; arbitrary label selection would need an explicit range map.
            if labels != list(range(len(labels))):
                raise ValueError(
                    "NexusStackSource cursor consumption requires contiguous "
                    "0-based frame labels")
        desc = cursor.descriptor
        # Coordinated route: only the bound grant.  The compatibility budget is
        # reached ONLY by the legacy uncoordinated ``run_reduction`` entry.
        budget = (self.allocation.owner_block_bytes
                  if self.allocation is not None else source_block_budget_bytes())
        read_plan = plan_reads(
            desc.frame_count,
            desc.frame_shape,
            desc.dtype,
            desc.chunks,
            budget,
            frame_interval=(0, len(labels)),
            requested_block_frames=chunk_size,
            two_d=desc.is_2d,
        )
        for block in cursor.iter_blocks(read_plan):
            yield np.asarray(block.array), labels[block.start:block.stop]

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
            cursor = self._consumption_cursor
            if cursor is not None:
                p = cursor.metadata_provider()
                p.scan_table()  # materialize before this cursor is consumed
                provider = p
            else:
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
