# -*- coding: utf-8 -*-
"""Composite FrameSource — combine several scans into one frame stream.

A scan **group** (stitch/RSM range syntax, e.g. ``1-3``) is processed as ONE
output: its member scans are concatenated into a single
:class:`CompositeFrameSource` and handed to ``run_stitch`` / ``run_roi_signals``
exactly like any other source.  This is also how stitch's cross-file "Multi"
(different files/kinds combined) is modelled — a composite over members from any
:func:`~xrd_tools.sources.open_source` kind.

Frames are re-indexed ``0..N-1`` across members; ``load_frame`` / ``metadata_for``
dispatch to the owning member; ``motors`` concatenate per key (NaN-padding a
member that lacks a key).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from xrd_tools.core.scan import SourceCapabilities, SourceKind
from xrd_tools.sources.base import BaseFrameSource


class CompositeFrameSource(BaseFrameSource):
    """Concatenate ``members`` (FrameSources or openable specs) into one source."""

    kind = SourceKind.UNKNOWN

    def __init__(self, members, *, name: str | None = None) -> None:
        self._members = [self._as_source(m) for m in members]
        if not self._members:
            raise ValueError("CompositeFrameSource needs at least one member")
        # global frame index -> (member position, member's own frame label)
        self._map: list[tuple[int, int]] = []
        for mi, member in enumerate(self._members):
            for label in member.frame_indices:
                self._map.append((mi, int(label)))
        self._motors_cache: dict[str, np.ndarray] | None = None
        caps = [m.capabilities for m in self._members]
        super().__init__(
            name=name or " + ".join(m.name for m in self._members),
            frame_indices=list(range(len(self._map))),
            capabilities=SourceCapabilities(
                supports_random_access=all(c.supports_random_access for c in caps),
                supports_chunks=all(c.supports_chunks for c in caps),
                has_metadata=any(c.has_metadata for c in caps),
                # raw is usable for the whole group only if EVERY member has it
                has_raw_references=all(c.has_raw_references for c in caps),
            ),
        )

    @staticmethod
    def _as_source(member):
        """A member may be an opened FrameSource OR an openable spec/URI — open
        specs here so ``concat_sources([spec_a, spec_b])`` (the grouping path)
        works as documented."""
        if hasattr(member, "frame_indices") and hasattr(member, "load_frame"):
            return member
        from xrd_tools.sources.registry import open_source
        return open_source(member)

    @property
    def members(self):
        return list(self._members)

    def load_frame(self, index: int) -> np.ndarray:
        mi, label = self._map[int(index)]
        return np.asarray(self._members[mi].load_frame(label))

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        mi, label = self._map[int(index)]
        return self._members[mi].metadata_for(label)

    @property
    def motors(self) -> dict[str, np.ndarray]:
        """Per-key concatenation of the members' whole-array ``motors``; a member
        missing a key contributes a NaN block so every column spans all frames."""
        if self._motors_cache is not None:
            return {k: v.copy() for k, v in self._motors_cache.items()}
        keys: list[str] = []
        member_motors = [
            (getattr(member, "motors", None) or {})
            for member in self._members
        ]
        for mm in member_motors:
            for k in mm:
                if k not in keys:
                    keys.append(k)
        out: dict[str, np.ndarray] = {}
        for k in keys:
            parts = []
            for member, mm in zip(self._members, member_motors):
                n = len(member.frame_indices)
                # Each block MUST be exactly n long so the concatenation stays
                # frame-aligned: a member whose motor array is longer than its
                # frame count (e.g. a partial SPEC scan with fewer images than
                # metadata points) is clipped; a shorter/absent one is NaN-padded.
                if k in mm:
                    arr = np.asarray(mm[k], dtype=float).ravel()
                    block = (arr[:n] if arr.shape[0] >= n
                             else np.concatenate([arr, np.full(n - arr.shape[0], np.nan)]))
                else:
                    block = np.full(n, np.nan)
                parts.append(block)
            out[k] = np.concatenate(parts) if parts else np.asarray([], dtype=float)
        self._motors_cache = {k: v.copy() for k, v in out.items()}
        return out


def concat_sources(members, *, name: str | None = None) -> CompositeFrameSource:
    """Convenience constructor — combine ``members`` into a CompositeFrameSource."""
    return CompositeFrameSource(members, name=name)


__all__ = ["CompositeFrameSource", "concat_sources"]
