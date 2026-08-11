"""Run-level display residency accounting across exact artifact frame keys."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .display_values import DisplayFrameKey


@dataclass(frozen=True, slots=True)
class DisplayResidencyLimits:
    heavy: int
    thumbnails: int
    browse: int
    live: int


@dataclass(frozen=True, slots=True)
class DisplayResidencySnapshot:
    limits: DisplayResidencyLimits
    heavy: int
    thumbnails: int
    browse: int
    live: int


@dataclass(frozen=True, slots=True)
class _ResidentStores:
    records: Any
    publications: Any


class RunDisplayResidency:
    """One persisted-first FIFO owner for all display tiers in one run."""

    def __init__(self, limits: DisplayResidencyLimits) -> None:
        self.limits = limits
        self._stores: dict[DisplayFrameKey, _ResidentStores] = {}
        self._heavy: OrderedDict[DisplayFrameKey, None] = OrderedDict()
        self._thumbnails: OrderedDict[DisplayFrameKey, None] = OrderedDict()
        self._browse: OrderedDict[DisplayFrameKey, None] = OrderedDict()
        self._live: OrderedDict[DisplayFrameKey, None] = OrderedDict()

    def observe(
        self,
        key: DisplayFrameKey,
        *,
        records: Any,
        publications: Any,
        incoming_heavy: bool = False,
        incoming_thumbnail: bool = False,
    ) -> None:
        """Register one key's stores and tier membership (prepare half).

        ``incoming_heavy``/``incoming_thumbnail`` are detached values-only
        facts about a payload the caller is about to publish, so a
        publication-last commit keeps tier membership exact without the
        payload being publicly visible yet.  Cap enforcement is the separate
        :meth:`enforce` apply half, run only after the authoritative
        publication landed — a rejected candidate therefore never evicts
        prior public state (its touches are undone by :meth:`reconcile`).
        """
        stores = _ResidentStores(records, publications)
        self._stores[key] = stores
        self._touch(self._live, key)
        self._touch(self._browse, key)
        if (
            incoming_heavy
            or records.has_heavy_payload(key.local_frame_label)
            or publications.has_heavy_payload(key.local_frame_label)
        ):
            self._touch(self._heavy, key)
        if (
            incoming_thumbnail
            or publications.has_thumbnail(key.local_frame_label)
        ):
            self._touch(self._thumbnails, key)

    def enforce(self) -> None:
        """Apply the cap trims (demotion-only store operations)."""
        self._trim(self._heavy, self.limits.heavy, self._evict_heavy)
        self._trim(
            self._thumbnails,
            self.limits.thumbnails,
            self._evict_thumbnail,
        )
        self._trim(self._browse, self.limits.browse, self._evict_browse)
        self._trim(self._live, self.limits.live, self._evict_live)

    def capture(self, key: DisplayFrameKey) -> tuple:
        """Exact pre-attempt facts for ONE key (§22.3 prepare half).

        Detached values: the key's registration entry plus the COMPLETE
        four-tier key order, so :meth:`restore` can put back membership AND
        FIFO eviction order exactly — never re-derive them from stores the
        failed attempt already mutated (the §22.2 root cause).
        """
        return (
            key,
            key in self._stores,
            self._stores.get(key),
            tuple(self._heavy),
            tuple(self._thumbnails),
            tuple(self._browse),
            tuple(self._live),
        )

    def restore(self, captured: tuple) -> None:
        """Restore the exact captured state (§22.3 abort half).

        Only the capturing commit can have mutated this owner in between
        (both run under the one display lock), so rebuilding each tier from
        its captured key order is an exact undo — including the prior FIFO
        position of a re-observed public key and the complete removal of a
        never-published candidate.  Idempotent; pure dict operations.
        """
        key, registered, stores, heavy, thumbnails, browse, live = captured
        if registered:
            self._stores[key] = stores
        else:
            self._stores.pop(key, None)
        for tier, snapshot in (
            (self._heavy, heavy),
            (self._thumbnails, thumbnails),
            (self._browse, browse),
            (self._live, live),
        ):
            tier.clear()
            for entry in snapshot:
                tier[entry] = None

    def snapshot(self) -> DisplayResidencySnapshot:
        return DisplayResidencySnapshot(
            self.limits,
            len(self._heavy),
            len(self._thumbnails),
            len(self._browse),
            len(self._live),
        )

    def retire_navigation(
        self, keys: tuple[DisplayFrameKey, ...]
    ) -> None:
        """Drop catalog-qualified browse tiers while bounded live data drains."""
        for key in keys:
            if self._evict_browse(key):
                self._browse.pop(key, None)
                self._thumbnails.pop(key, None)

    @staticmethod
    def _touch(
        values: OrderedDict[DisplayFrameKey, None],
        key: DisplayFrameKey,
    ) -> None:
        values.pop(key, None)
        values[key] = None

    def _trim(self, values, limit: int, evict) -> None:
        while len(values) > limit:
            removed = False
            for key in tuple(values):
                if evict(key):
                    values.pop(key, None)
                    removed = True
                    break
            if not removed:
                return

    def _evict_heavy(self, key: DisplayFrameKey) -> bool:
        stores = self._stores.get(key)
        if stores is None:
            return False
        label = key.local_frame_label
        publication_done = (
            stores.publications.get(label) is None
            or not stores.publications.has_heavy_payload(label)
            or stores.publications.evict_heavy(label)
        )
        if not publication_done:
            return False
        return (
            not stores.records.has_heavy_payload(label)
            or stores.records.release_heavy(label)
        )

    def _evict_thumbnail(self, key: DisplayFrameKey) -> bool:
        stores = self._stores.get(key)
        if stores is None:
            return False
        label = key.local_frame_label
        return (
            stores.publications.get(label) is None
            or not stores.publications.has_thumbnail(label)
            or stores.publications.evict_thumbnail(label)
        )

    def _evict_browse(self, key: DisplayFrameKey) -> bool:
        stores = self._stores.get(key)
        if stores is None:
            return False
        label = key.local_frame_label
        publication = stores.publications.get(label)
        if (
            publication is not None
            and not stores.records.can_release_record(label)
        ):
            return False
        publication_done = (
            publication is None
            or stores.publications.discard(label)
        )
        if not publication_done:
            return False
        return True

    def _evict_live(self, key: DisplayFrameKey) -> bool:
        stores = self._stores.get(key)
        if stores is None:
            return False
        label = key.local_frame_label
        if not self._evict_browse(key):
            return False
        removed = (
            stores.records.get(label) is None
            or stores.records.release_record(label)
        )
        if removed:
            self._heavy.pop(key, None)
            self._thumbnails.pop(key, None)
            self._browse.pop(key, None)
            self._live.pop(key, None)
            self._stores.pop(key, None)
        return removed


__all__ = [
    "DisplayResidencyLimits",
    "DisplayResidencySnapshot",
    "RunDisplayResidency",
]
