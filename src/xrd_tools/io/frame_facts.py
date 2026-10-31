"""Bounded resident per-frame facts for one Browse context (PERF-BROWSE-OPEN-128 stage B).

One store per loaded Browse context holds the immutable :class:`FrameFacts` (or :class:`FrameFactsFailure`) of
the frames Browse has shown. Its worker lane commits bounded batches; display code only reads resident values and
never opens the file. Least recently committed records leave first; an evicted frame is simply read again.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock

from xrd_tools.io.frame_view import FrameFacts, FrameFactsFailure


class FrameFactsStore:
    """At most 512 records AND 2 MiB of conservatively charged facts, least recently committed evicted first."""

    MAX_RECORDS = 512
    MAX_BYTES = 2 * 1024 * 1024
    #: Each record's fixed charge for its object, slots and geometry; its texts are charged on top.
    RECORD_BASE_BYTES = 512

    def __init__(self) -> None:
        self._lock = Lock()
        self._items: OrderedDict[int, FrameFacts | FrameFactsFailure] = OrderedDict()
        self._bytes = 0

    @classmethod
    def charge(cls, item: FrameFacts | FrameFactsFailure) -> int:
        """The deterministic retained-byte charge of one record."""

        if type(item) is FrameFacts:
            texts = (item.source_path, item.thumbnail_failure)
        elif type(item) is FrameFactsFailure:
            texts = (item.reason,)
        else:
            raise TypeError("frame facts records must be exact FrameFacts or FrameFactsFailure")
        return cls.RECORD_BASE_BYTES + sum(len(text.encode("utf-8")) for text in texts if text is not None)

    def get(self, label: int) -> FrameFacts | FrameFactsFailure | None:
        """Return one resident record, or ``None`` when the frame's facts are not resident."""

        with self._lock:
            return self._items.get(label)

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def resident_labels(self) -> tuple[int, ...]:
        """Resident labels, least recently committed first."""

        with self._lock:
            return tuple(self._items)

    def commit(
        self,
        items: tuple[FrameFacts | FrameFactsFailure, ...],
        *,
        retain: tuple[int, ...] = (),
    ) -> None:
        """Admit one batch atomically, keeping ``retain`` (resident labels of the same request) as recent.

        A batch that cannot fit with its retained labels is refused with the store unchanged.
        """

        if type(items) is not tuple or type(retain) is not tuple:
            raise TypeError("frame facts batches and retained labels must be exact tuples")
        charges: dict[int, int] = {}
        for item in items:
            charge = self.charge(item)
            if item.label in charges:
                raise ValueError("a frame facts batch names one frame twice")
            charges[item.label] = charge
        if len(set(retain)) != len(retain) or any(type(label) is not int or label < 0 for label in retain):
            raise ValueError("retained frame facts labels must be distinct exact nonnegative integers")
        with self._lock:
            kept = [label for label in retain if label not in charges]
            if any(label not in self._items for label in kept):
                raise ValueError("retained frame facts are not resident")
            needed = sum(charges.values()) + sum(self.charge(self._items[label]) for label in kept)
            if len(charges) + len(kept) > self.MAX_RECORDS or needed > self.MAX_BYTES:
                raise ValueError("a frame facts batch exceeds the store bounds")
            for label in kept:
                self._items.move_to_end(label)
            for item in items:
                previous = self._items.pop(item.label, None)
                if previous is not None:
                    self._bytes -= self.charge(previous)
                self._items[item.label] = item
                self._bytes += charges[item.label]
            while len(self._items) > self.MAX_RECORDS or self._bytes > self.MAX_BYTES:
                _label, evicted = self._items.popitem(last=False)
                self._bytes -= self.charge(evicted)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0


__all__ = ["FrameFactsStore"]
