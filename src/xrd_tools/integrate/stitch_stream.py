"""Detached native Stitch accumulation for a verified preview prefix."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class StitchPreviewResult:
    """The native payload and weight/coverage for each displayed product."""

    payload: Any
    diagnostics: Any
    profile: tuple | None = None
    gi_map: tuple | None = None
    gi_cuts: tuple = ()


class StitchPreviewStream:
    """A geometry scout or a detached accumulator on one immutable grid.

    Pass a fresh stream with no bounds to a native kernel to scout its actual
    geometry without opening any detector images. A bounded stream accepts
    successive ordered slices on the same grid. Only the caller can declare
    the selected stream ended and flush its frame screening tail.
    """

    def __init__(self, bounds=None):
        self.bounds = None if bounds is None else MappingProxyType(dict(bounds))
        self.footprint = MappingProxyType({})
        self.automatic = frozenset()
        self._sums = None
        self._closed = False

    def observe(self, footprint, automatic):
        if self.bounds is not None:
            raise ValueError("Stitch geometry scout cannot replace an allocated grid")
        self.footprint = MappingProxyType(dict(footprint))
        self.automatic = frozenset(automatic)

    def sums(self, candidate):
        if self.bounds is None or self._closed:
            raise ValueError("Stitch preview has no open grid")
        if self._sums is None:
            self._sums = candidate
        elif type(self._sums) is not type(candidate) or self._sums.identity != candidate.identity:
            raise ValueError("Stitch preview slice changes its allocated grid")
        return self._sums

    def snapshot(self):
        return None if self._sums is None else self._sums.snapshot()

    def finish(self, frame_screen):
        if self._closed:
            return
        if frame_screen is not None and self._sums is not None:
            for parts, verdict in frame_screen.finish():
                if not verdict.excluded:
                    self._sums.add_held(parts)
        self._closed = True
