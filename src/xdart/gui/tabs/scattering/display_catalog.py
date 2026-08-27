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

    def seed_at_work_ordinal(
        self,
        source_scan: str,
        artifact: str,
        local_frame_label: int,
        work_ordinal: int,
    ) -> DisplayNavigationDelta:
        """Install one persisted key at its exact absolute run ordinal."""

        return self.seed_many_at_work_ordinals((
            (source_scan, artifact, local_frame_label, work_ordinal),
        ))[0]

    def seed_many_at_work_ordinals(
        self,
        rows: tuple[tuple[str, str, int, int], ...],
    ) -> tuple[DisplayNavigationDelta, ...]:
        """Atomically install one preflighted persisted navigation prefix."""

        if type(rows) is not tuple:
            raise TypeError("persisted navigation rows must be an exact tuple")
        validated: list[tuple[str, str, int, int]] = []
        seen: set[tuple[str, int]] = set()
        for row in rows:
            if type(row) is not tuple or len(row) != 4:
                raise TypeError("persisted navigation row is invalid")
            source_scan, artifact, local_frame_label, work_ordinal = row
            self._validate_seed_fields(
                source_scan,
                artifact,
                local_frame_label,
                work_ordinal,
            )
            value_key = (artifact, local_frame_label)
            if value_key in seen:
                raise ValueError(
                    "persisted navigation prefix contains a duplicate key"
                )
            seen.add(value_key)
            validated.append(row)

        entries = deque(self._entries)
        exact = dict(self._exact)
        by_value = dict(self._by_value)
        work_high_water = self._work_ordinal
        deltas: list[DisplayNavigationDelta] = []
        changed = False
        for source_scan, artifact, local_frame_label, work_ordinal in validated:
            value_key = (artifact, local_frame_label)
            existing = by_value.get(value_key)
            if existing is not None:
                if existing.source_scan != source_scan:
                    raise ValueError("persisted key source scan conflicts")
                if existing.work_ordinal != work_ordinal:
                    raise ValueError("persisted key work ordinal conflicts")
                deltas.append(DisplayNavigationDelta(existing, ()))
                continue
            if work_ordinal <= work_high_water:
                raise ValueError(
                    "persisted work ordinal must advance high-water"
                )
            key = DisplayFrameKey(
                self.identity,
                source_scan,
                artifact,
                local_frame_label,
                work_ordinal,
            )
            changed = True
            work_high_water = work_ordinal
            entries.append(key)
            exact[id(key)] = key
            by_value[value_key] = key
            retired: tuple[DisplayFrameKey, ...] = ()
            if len(entries) > self.max_items:
                oldest = entries.popleft()
                exact.pop(id(oldest), None)
                oldest_value = (
                    oldest.artifact,
                    oldest.local_frame_label,
                )
                if by_value.get(oldest_value) is oldest:
                    by_value.pop(oldest_value, None)
                retired = (oldest,)
            deltas.append(DisplayNavigationDelta(key, retired))

        if changed:
            self._entries = entries
            self._exact = exact
            self._by_value = by_value
            self._work_ordinal = work_high_water
        return tuple(deltas)

    @staticmethod
    def _validate_seed_fields(
        source_scan: str,
        artifact: str,
        local_frame_label: int,
        work_ordinal: int,
    ) -> None:
        if type(source_scan) is not str or not source_scan:
            raise TypeError(
                "persisted source scan must be a nonempty exact string"
            )
        if type(artifact) is not str or not artifact:
            raise TypeError(
                "persisted artifact must be a nonempty exact string"
            )
        if type(local_frame_label) is not int:
            raise TypeError(
                "persisted local frame label must be an exact integer"
            )
        if type(work_ordinal) is not int:
            raise TypeError("persisted work ordinal must be an exact integer")
        if work_ordinal < 1:
            raise ValueError("persisted work ordinal must be positive")

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
