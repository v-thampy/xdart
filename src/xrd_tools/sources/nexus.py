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
            metadata=dict(self.metadata_for(index)),
            source_path=self.path,
            source_frame_index=int(index),
            loader=lambda frame: self.load_frame(frame.source_frame_index or 0),
            source_identity=str(self.path),
        )

    # -- Bluesky / apstools NXWriter per-frame metadata --------------------
    # A Bluesky ``.nxs`` classifies as a NEXUS_STACK (raw image stack), so its
    # per-frame scan columns (motors + ion-chamber/photodiode counters + EPOCH)
    # would otherwise be invisible to Plot Metadata.  Surface them here (motors
    # as whole-array columns via ``motors``; counters/EPOCH per-frame via
    # ``metadata_for``).  Guarded + cached so a plain image stack is untouched.
    def _bluesky_columns(self) -> dict | None:
        cache = getattr(self, "_bluesky_cache", "unset")
        if cache != "unset":
            return cache
        result: dict | None = None
        try:
            import h5py

            from xrd_tools.io.bluesky_nexus import (
                bluesky_angles,
                bluesky_constant_metadata,
                bluesky_per_frame_table,
                is_bluesky_nxwriter,
                resolve_nxentry,
            )
            with h5py.File(self.path, "r") as f:
                e = resolve_nxentry(f, self.entry)
                if e is not None and is_bluesky_nxwriter(e):
                    table = {k: np.asarray(v)
                             for k, v in bluesky_per_frame_table(e).items()}
                    result = {
                        "motors": {k: np.asarray(v)
                                   for k, v in bluesky_angles(e).items()},
                        "table": table,
                        # Held-fixed motors + eiger counting time broadcast as
                        # constant columns (see bluesky_constant_metadata).
                        "constants": bluesky_constant_metadata(
                            e, exclude=table.keys()),
                    }
        except Exception:
            result = None
        self._bluesky_cache = result
        return result

    @property
    def motors(self) -> dict[str, np.ndarray]:
        cols = self._bluesky_columns()
        return dict(cols["motors"]) if cols else {}

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        cols = self._bluesky_columns()
        if not cols:
            return {}
        table, motors = cols["table"], cols["motors"]
        try:
            pos = self._frame_indices.index(int(index))
        except ValueError:
            return {}
        out: dict[str, Any] = {}
        for name, arr in table.items():
            if name in motors:
                continue  # motors already surface as whole-array columns
            if 0 <= pos < len(arr):
                try:
                    out[name] = float(arr[pos])
                except (TypeError, ValueError):
                    pass
        # Constant per-scan columns (fixed motors + eiger count time) broadcast
        # to every frame; setdefault so a per-frame column always wins.
        for name, val in cols.get("constants", {}).items():
            out.setdefault(name, float(val))
        return out


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
