# -*- coding: utf-8 -*-
"""``ContainerCursor`` — one owner-thread, one open handle per container (R2).

The cursor replaces the R1/legacy pattern of reopening a NeXus/Eiger master
independently for frame count, dataset lookup, wavelength, metadata, and every
frame read.  ``__enter__`` opens ONE backend handle and resolves the
entry/dataset once; the descriptor, wavelength, lazy metadata provider, frame
count, single-frame reads, and byte-bounded block reads all derive from that
same handle.  ``close()`` is idempotent and runs on success, Stop/cancel, probe
failure, read failure, generator abandonment, and normal exit; reads after
close fail clearly and never silently reopen.

Ownership rules (handoff §4.2):

* the cursor is owned and used by ONE thread — never pass its live h5py/dataset
  objects or a :class:`ReadBlock` view across threads;
* a 2-D dataset exposes exactly one frame at index 0 and rejects every other
  index;
* a stale R1 :class:`~xrd_tools.sources.discover.Candidate` (path/stamp changed
  since discovery) is rejected with ``StaleCandidateError`` BEFORE the handle is
  opened, so a late consumer never reads an old owner's bytes;
* block reads return NATIVE detector dtype (no ``float64`` expansion); frame
  views share memory with their owner block, which is charged once.

The cursor does not introduce automatic incomplete-file consumption or any new
SWMR/recovery policy: a provisional (unfinalized) container is described as
such, and reads of a container with no detector dataset fail clearly.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xrd_tools.core.energy import WavelengthUnit
from xrd_tools.sources.descriptor import (
    ContainerDescriptor,
    describe_container_from_open,
)
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.metadata_provider import (
    MetadataProvider,
    metadata_provider_for_open_entry,
)
from xrd_tools.sources.read_plan import ReadPlan

__all__ = [
    "ContainerCursor",
    "ReadBlock",
    "CursorClosedError",
    "ContainerNotReadyError",
    "open_container_cursor",
]


class CursorClosedError(RuntimeError):
    """Raised when a read is attempted on a closed :class:`ContainerCursor`."""


class ContainerNotReadyError(RuntimeError):
    """A valid container is still provisional; close it and retry later.

    This is intentionally a typed, retry-later outcome for callers that need
    an open cursor.  Use :func:`describe_container` when a provisional
    descriptor value is sufficient for a polling/readiness decision.
    """


@dataclass(frozen=True, slots=True)
class ReadBlock:
    """One native-dtype owner block covering frames ``[start, stop)``.

    ``array`` is the single retained allocation; :meth:`frame` returns a memory
    SHARING view into it, so many per-frame consumers cost the block once.
    :attr:`nbytes` is what the queue/consumer must charge exactly once.
    """

    start: int
    stop: int
    array: np.ndarray

    @property
    def n_frames(self) -> int:
        return self.stop - self.start

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes)

    def frame(self, index: int) -> np.ndarray:
        """A memory-sharing view of the frame at absolute ``index``."""
        offset = int(index) - self.start
        if not 0 <= offset < self.array.shape[0]:
            raise IndexError(
                f"frame {index} not in block [{self.start}, {self.stop})")
        return self.array[offset]


class ContainerCursor:
    """A context-managed, single-handle cursor over one NeXus/Eiger container."""

    def __init__(self, path: str | Path, *, entry: str = "entry",
                 candidate: Any = None) -> None:
        self._path = Path(path)
        self._entry = entry
        self._candidate = candidate
        self._h5: Any = None
        self._stack: Any = None
        self._entry_grp: Any = None
        self._descriptor: ContainerDescriptor | None = None
        self._provider: MetadataProvider | None = None
        self._opened = False
        self._closed = False

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self) -> "ContainerCursor":
        return self.open()

    def open(self) -> "ContainerCursor":
        if self._closed:
            raise CursorClosedError(
                f"cursor for {self._path} was already closed; open a new one")
        if self._opened:
            return self
        # Reject a stale candidate BEFORE touching the file (handoff §4.2 / R1).
        if self._candidate is not None:
            self._reject_if_stale()

        import h5py

        from xrd_tools.io.bluesky_nexus import resolve_nxentry
        from xrd_tools.io.nexus import NexusImageStack

        try:
            self._h5 = h5py.File(self._path, "r")
        except Exception:
            self._closed = True
            raise
        try:
            self._descriptor = describe_container_from_open(
                self._h5, path=self._path, entry=self._entry,
                candidate=self._candidate)
            # A finalized-link gap has no dataset to bind, so an open cursor
            # must return the typed retry-later result.  An NXWriter file may
            # legitimately expose a complete detector stack before it writes
            # ``end_time``; that existing live-append route stays readable.
            if (self._descriptor.state is ProbeState.IN_PROGRESS
                    and self._descriptor.dataset_path is None):
                raise ContainerNotReadyError(
                    f"{self._path} is not ready for cursor reads: "
                    f"{self._descriptor.reason}")
            if self._descriptor.state is ProbeState.INVALID:
                # NXS-DIM-1: a defective container (unsupported detector
                # rank) must fail the open loudly, not yield a stackless
                # cursor whose reads fail obscurely later.
                from xrd_tools.io.nexus import UnsupportedDetectorRankError
                err = (UnsupportedDetectorRankError
                       if "rank" in str(self._descriptor.reason or "")
                       else ValueError)
                raise err(
                    f"{self._path} is not a readable detector container: "
                    f"{self._descriptor.reason}")
            try:
                self._entry_grp = resolve_nxentry(self._h5, self._entry)
            except Exception:
                self._entry_grp = None
            # Build the read stack over the SAME open handle only when a detector
            # dataset was resolved; NexusImageStack takes ownership of the file.
            paths = self._resolved_paths(self._descriptor)
            if paths:
                self._stack = NexusImageStack(self._h5, list(paths))
            # Revalidate the FULL identity AFTER inspection: a path/stamp/owner/
            # removal race during the open must fail closed, never return a
            # usable cursor attributed to a stale owner.
            if self._candidate is not None:
                self._check_identity(when="after inspection")
        except Exception:
            self.close()
            raise
        self._opened = True
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Idempotent close — runs on success, Stop, error, and abandonment."""
        if self._closed:
            return
        self._closed = True
        self._opened = False
        stack, self._stack = self._stack, None
        h5, self._h5 = self._h5, None
        self._entry_grp = None
        # Seal a lazy provider that never materialized so a post-close metadata
        # read fails clearly instead of silently returning empty (§4.2.5).
        if self._provider is not None:
            try:
                self._provider.mark_source_closed()
            except Exception:
                pass
        self._provider = None
        if stack is not None:
            try:
                stack.close()  # NexusImageStack owns and closes the file handle
            except Exception:
                pass
            h5 = None  # already closed via the stack; do not double-close
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass

    @property
    def closed(self) -> bool:
        return self._closed

    # -- facts ----------------------------------------------------------------
    @property
    def descriptor(self) -> ContainerDescriptor:
        self._require_open()
        assert self._descriptor is not None
        return self._descriptor

    @property
    def wavelength(self) -> float | None:
        return self.descriptor.wavelength

    @property
    def frame_count(self) -> int:
        return int(self.descriptor.frame_count)

    @property
    def is_2d(self) -> bool:
        return bool(self.descriptor.is_2d)

    def metadata_provider(self) -> MetadataProvider:
        """The lazy metadata provider bound to this cursor's open handle.

        Built once and cached; repeated metadata reads reuse it and open no new
        master (handoff §4.3).
        """
        self._require_open()
        if self._provider is None:
            desc = self._descriptor
            assert desc is not None
            # R3-P1: ``ContainerDescriptor.wavelength`` comes from the io-layer
            # ``_read_wavelength``, which explicitly returns ANGSTROMS — declare
            # that unit here so the value becomes canonicalizable evidence.
            self._provider = metadata_provider_for_open_entry(
                self._entry_grp, frame_count=desc.frame_count,
                wavelength=desc.wavelength,
                wavelength_unit=WavelengthUnit.ANGSTROM,
                is_bluesky=desc.is_bluesky)
        return self._provider

    def metadata_for(self, frame_index: int) -> Any:
        return self.metadata_provider().metadata_for(frame_index)

    # -- reads (native dtype) -------------------------------------------------
    def read_frame(self, index: int) -> np.ndarray:
        """Read one frame as a native-dtype array from the open handle."""
        self._require_readable()
        idx = int(index)
        if self.is_2d and idx != 0:
            raise IndexError(
                f"2-D detector dataset exposes only frame 0; got {idx}")
        return np.asarray(self._stack[idx])

    def read_block(self, start: int, stop: int) -> ReadBlock:
        """Read frames ``[start, stop)`` as one native-dtype owner block.

        ``stop`` is clamped to the frame count and the returned block's ``stop``
        reflects the frames ACTUALLY read, so ``ReadBlock.n_frames`` can never
        overstate the array (an out-of-range ``stop`` would otherwise slice-clamp
        silently and leave ``n_frames`` lying about the block extent)."""
        self._require_readable()
        s = int(start)
        e = min(int(stop), self.frame_count)
        if e <= s:
            raise ValueError(
                f"empty block range [{s}, {stop}) for frame_count {self.frame_count}")
        array = np.asarray(self._stack[s:e])
        actual_stop = s + int(array.shape[0])
        return ReadBlock(start=s, stop=actual_stop, array=array)

    def iter_blocks(self, plan: ReadPlan) -> Iterator[ReadBlock]:
        """Yield each owner block for *plan*'s ranges, one at a time.

        Only one block is live per iteration step; the consumer must charge each
        :class:`ReadBlock` once and not retain more than the plan's inflight
        budget.
        """
        self._require_readable()
        for start, stop in plan.ranges:
            yield self.read_block(start, stop)

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _resolved_paths(descriptor: ContainerDescriptor) -> list[str]:
        if descriptor.segment_paths:
            return list(descriptor.segment_paths)
        if descriptor.dataset_path:
            return [descriptor.dataset_path]
        return []

    def _reject_if_stale(self) -> None:
        """Fail closed BEFORE opening the file when the candidate's R1 identity no
        longer matches (zero source opens on a stale candidate)."""
        self._check_identity(when="before opening")

    def _check_identity(self, *, when: str) -> None:
        """Validate the candidate's FULL R1 identity — path, ``(size, mtime)``
        stamp, AND owning adapter id — against the current filesystem/registry
        state.  Reuses the R1 registry seam (``candidate_owner``); an owner flip
        with unchanged bytes, a byte change, or a removal all fail closed.  Run
        both before the open and again after inspection so a path/stamp/owner/
        removal race cannot return a usable cursor result."""
        from xrd_tools.sources.directory_index import StaleCandidateError

        cand = self._candidate
        if cand is None:
            return
        if getattr(cand, "path", self._path) != self._path:
            raise StaleCandidateError(
                f"candidate path {getattr(cand, 'path', None)!r} does not match "
                f"cursor path {self._path!r}")
        try:
            st = self._path.stat()
        except OSError as exc:
            raise StaleCandidateError(
                f"{self._path} is no longer present ({when}): {exc}")
        stamp = getattr(cand, "version_stamp", None)
        if stamp is not None and (st.st_size, st.st_mtime_ns) != tuple(stamp):
            raise StaleCandidateError(
                f"{self._path} bytes changed since discovery ({when}); re-poll "
                "and re-probe before consuming")
        adapter_id = getattr(cand, "adapter_id", None)
        if adapter_id is not None:
            # Ensure the built-in adapters are registered so the owner lookup is
            # authoritative (a candidate produced by discovery already did this).
            import xrd_tools.sources.registry  # noqa: F401
            from xrd_tools.sources.adapters import candidate_owner

            owner = candidate_owner(self._path)
            current_id = owner.id if owner is not None else None
            if current_id != adapter_id:
                raise StaleCandidateError(
                    f"{self._path} owning adapter changed from {adapter_id!r} to "
                    f"{current_id!r} ({when}); re-poll and re-probe the new owner")

    def _require_open(self) -> None:
        if self._closed:
            raise CursorClosedError(f"cursor for {self._path} is closed")
        if not self._opened:
            raise RuntimeError(
                f"cursor for {self._path} is not open; use `with` or call open()")

    def _require_readable(self) -> None:
        self._require_open()
        if self._stack is None:
            desc = self._descriptor
            reason = desc.reason if desc is not None else "no detector dataset"
            raise ValueError(
                f"{self._path} has no readable detector dataset ({reason})")


def open_container_cursor(
    path: str | Path,
    *,
    entry: str = "entry",
    candidate: object | None = None,
) -> ContainerCursor:
    """Open one exact cursor through the reviewed source-adapter boundary."""

    return ContainerCursor(path, entry=entry, candidate=candidate).open()
