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
        self._heavy_candidates: OrderedDict[
            DisplayFrameKey, None
        ] = OrderedDict()
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
        has_heavy = (
            incoming_heavy
            or records.has_heavy_payload(key.local_frame_label)
            or publications.has_heavy_residency(key.local_frame_label)
        )
        if has_heavy:
            self._touch(self._heavy, key)
            self._touch(self._heavy_candidates, key)
        elif key in self._heavy:
            self._rearm_heavy_key(key)
        if (
            incoming_thumbnail
            or publications.has_thumbnail(key.local_frame_label)
        ):
            self._touch(self._thumbnails, key)

    def enforce(
        self,
        *,
        protected=(),
        heavy_victim: DisplayFrameKey | None = None,
    ) -> None:
        """Apply the cap trims (demotion-only store operations)."""
        self._trim_heavy(protected, preferred=heavy_victim)
        self._trim(
            self._thumbnails,
            self.limits.thumbnails,
            self._evict_thumbnail,
            protected,
        )
        self._trim(self._browse, self.limits.browse, self._evict_browse, protected)
        self._trim(self._live, self.limits.live, self._evict_live, protected)

    def capture(self, key: DisplayFrameKey) -> tuple:
        """Exact pre-attempt facts for ONE key (§22.3 prepare half).

        Detached values: the key's registration entry plus the COMPLETE
        four-tier and heavy-probe key orders, so :meth:`restore` can put back
        membership AND FIFO eviction order exactly — never re-derive them from
        stores the failed attempt already mutated (the §22.2 root cause).
        """
        return (
            key,
            key in self._stores,
            self._stores.get(key),
            tuple(self._heavy),
            tuple(self._heavy_candidates),
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
        (
            key,
            registered,
            stores,
            heavy,
            heavy_candidates,
            thumbnails,
            browse,
            live,
        ) = captured
        if registered:
            self._stores[key] = stores
        else:
            self._stores.pop(key, None)
        for tier, snapshot in (
            (self._heavy, heavy),
            (self._heavy_candidates, heavy_candidates),
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

    def _trim(self, values, limit: int, evict, protected=()) -> None:
        while len(values) > limit:
            removed = False
            for key in tuple(values):
                if key in protected:
                    continue
                if evict(key):
                    values.pop(key, None)
                    removed = True
                    break
            if not removed:
                return

    def _rearm_heavy_owner(self, records: Any, publications: Any) -> None:
        armed = set(self._heavy_candidates)
        self._heavy_candidates.clear()
        for key in self._heavy:
            stores = self._stores.get(key)
            if (
                key in armed
                or (
                    stores is not None
                    and stores.records is records
                    and stores.publications is publications
                )
            ):
                self._heavy_candidates[key] = None

    def _rearm_heavy_labels(self, records, publications, labels) -> None:
        labels = set(labels)
        armed = set(self._heavy_candidates)
        self._heavy_candidates.clear()
        for key in self._heavy:
            stores = self._stores.get(key)
            if key in armed or (
                stores is not None
                and stores.records is records
                and stores.publications is publications
                and key.local_frame_label in labels
            ):
                self._heavy_candidates[key] = None

    def _rearm_heavy_key(self, key: DisplayFrameKey) -> None:
        armed = set(self._heavy_candidates)
        armed.add(key)
        self._heavy_candidates.clear()
        for candidate in self._heavy:
            if candidate in armed:
                self._heavy_candidates[candidate] = None

    def _trim_heavy(
        self,
        protected=(),
        *,
        preferred: DisplayFrameKey | None = None,
    ) -> None:
        if (
            len(self._heavy) > self.limits.heavy
            and preferred is not None
            and preferred in self._heavy
            and preferred not in protected
        ):
            publication_demoted = self._publication_heavy_missing(preferred)
            removed = self._evict_heavy(preferred)
            if removed:
                self._heavy_candidates.pop(preferred, None)
                self._heavy.pop(preferred, None)
            elif publication_demoted:
                # The publication owner has already selected this victim but
                # the record side is not durable yet.  Preserve and rearm the
                # exact victim for the durability retry; evicting a second
                # key here would split the two owners and underfill the cap.
                self._rearm_heavy_key(preferred)
                return
            else:
                self._heavy_candidates.pop(preferred, None)

        while len(self._heavy) > self.limits.heavy and self._heavy_candidates:
            key = next(
                (
                    candidate
                    for candidate in self._heavy_candidates
                    if candidate not in protected
                ),
                None,
            )
            if key is None:
                return
            publication_demoted = self._publication_heavy_missing(key)
            removed = self._evict_heavy(key)
            if removed:
                self._heavy_candidates.pop(key, None)
                self._heavy.pop(key, None)
            elif publication_demoted:
                # Local publication trimming ran before record durability.
                # Wait for the owner rearm instead of demoting another frame.
                return
            else:
                self._heavy_candidates.pop(key, None)

    def _publication_heavy_missing(self, key: DisplayFrameKey) -> bool:
        stores = self._stores.get(key)
        return bool(
            stores is not None
            and not stores.publications.has_heavy_residency(
                key.local_frame_label
            )
        )

    def _evict_heavy(self, key: DisplayFrameKey) -> bool:
        stores = self._stores.get(key)
        if stores is None:
            return False
        label = key.local_frame_label
        publication_done = (
            stores.publications.get(label) is None
            or not stores.publications.has_heavy_residency(label)
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
            self._heavy_candidates.pop(key, None)
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
