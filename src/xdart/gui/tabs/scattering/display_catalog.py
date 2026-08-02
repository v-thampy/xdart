"""Run-owned array-free catalog and exact incremental navigation deltas."""

from __future__ import annotations

from collections import deque

from .display_values import (
    DisplayFrameCatalog,
    DisplayFrameKey,
    DisplayNavigationDelta,
)
from .events import RunIdentity


CATALOG_MAX_ITEMS = 10_000


class DisplayCatalogIndex:
    """Mutable O(1) catalog; immutable tuples exist only at read boundaries."""

    def __init__(
        self,
        identity: RunIdentity,
        *,
        max_items: int = CATALOG_MAX_ITEMS,
    ) -> None:
        if type(identity) is not RunIdentity:
            raise TypeError("display catalog requires an exact RunIdentity")
        if type(max_items) is not int or max_items < 1:
            raise ValueError("display catalog max_items must be positive")
        self.identity = identity
        self.max_items = max_items
        self._entries: deque[DisplayFrameKey] = deque()
        self._exact: dict[int, DisplayFrameKey] = {}
        self._by_value: dict[tuple[str, int], DisplayFrameKey] = {}
        self._work_ordinal = 0

    @property
    def entries(self) -> tuple[DisplayFrameKey, ...]:
        return tuple(self._entries)

    def resize(self, max_items: int) -> tuple[DisplayFrameKey, ...]:
        if type(max_items) is not int or max_items < 1:
            raise ValueError("display catalog max_items must be positive")
        self.max_items = max_items
        retired: list[DisplayFrameKey] = []
        while len(self._entries) > self.max_items:
            retired.append(self._discard_oldest())
        return tuple(retired)

    def append(
        self,
        source_scan: str,
        artifact: str,
        local_frame_label: int,
    ) -> DisplayNavigationDelta:
        self._work_ordinal += 1
        key = DisplayFrameKey(
            self.identity,
            source_scan,
            artifact,
            local_frame_label,
            self._work_ordinal,
        )
        self._entries.append(key)
        self._exact[id(key)] = key
        self._by_value[(artifact, local_frame_label)] = key
        retired = (
            (self._discard_oldest(),)
            if len(self._entries) > self.max_items
            else ()
        )
        return DisplayNavigationDelta(key, retired)

    def resolve(
        self, frame: DisplayFrameKey
    ) -> DisplayFrameKey | None:
        if type(frame) is not DisplayFrameKey:
            return None
        candidate = self._exact.get(id(frame))
        return candidate if candidate is frame else None

    def resolve_exact(
        self, artifact: str, local_frame_label: int
    ) -> DisplayFrameKey | None:
        """O(1) exact by-value lookup for the hydration commit boundary."""
        return self._by_value.get((artifact, local_frame_label))

    def snapshot(self) -> DisplayFrameCatalog:
        return DisplayFrameCatalog(self.identity, tuple(self._entries))

    def _discard_oldest(self) -> DisplayFrameKey:
        key = self._entries.popleft()
        self._exact.pop(id(key), None)
        exact = self._by_value.get((key.artifact, key.local_frame_label))
        if exact is key:
            self._by_value.pop((key.artifact, key.local_frame_label), None)
        return key


__all__ = ["CATALOG_MAX_ITEMS", "DisplayCatalogIndex"]
