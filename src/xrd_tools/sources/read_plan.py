# -*- coding: utf-8 -*-
"""Pure, HDF5-object-free source read planning (R2).

A :class:`ReadPlan` decides *how many frames to read per block* from a NeXus /
HDF5 detector stack, given the dataset's native chunk layout, the native dtype,
and one explicit retained-byte budget.  It replaces the wrangler's historical
fixed ``_PREFETCH_READ_CHUNK = 16`` block size (a policy that both over-reads a
per-frame-compressed Eiger stack and float-expands a native ``uint16`` block)
with a layout-aware, byte-bounded choice.

This module is deliberately Qt-free AND h5py-free: :func:`plan_reads` takes only
plain integers/tuples/``numpy.dtype`` and returns a frozen value, so the same
arithmetic is exercised by unit tests, the benchmark, notebooks, and the GUI
prefetch worker without ever opening a file.  The cursor (see
:mod:`xrd_tools.sources.cursor`) reads the native chunk shape / dtype off the
resolved dataset once and hands them here; nothing in this module touches an
``h5py`` object.

Retained-memory accounting is charged to the *owner block* (one block-sized
native-dtype array that many per-frame views share), never to each frame view —
see :mod:`xrd_tools.core.staging`'s ``source_block_bytes`` for the byte model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["ReadPlan", "plan_reads"]


@dataclass(frozen=True, slots=True)
class ReadPlan:
    """An immutable, byte-bounded plan for reading a contiguous frame window.

    All byte quantities are NATIVE-dtype bytes (no ``float64`` expansion); the
    plan never decides a cast — conversion belongs at the integration boundary.
    """

    #: Total logical frames in the dataset (a 2-D detector dataset is 1 frame).
    frame_count: int
    #: Requested half-open window ``[requested_start, requested_stop)``.
    requested_start: int
    requested_stop: int
    #: Native bytes of ONE frame (``prod(frame_shape) * dtype.itemsize``).
    frame_bytes: int
    #: The native first-chunk frame count (``chunks[0]``) or ``None`` when the
    #: dataset is contiguous / non-chunked along the frame axis.
    native_chunk_frames: int | None
    #: Chosen frames per read block (>= 1 for a non-empty request).
    block_frames: int
    #: Half-open ``(start, stop)`` ranges covering the requested window in order.
    ranges: tuple[tuple[int, int], ...]
    #: Native bytes the whole requested window will move (``frame_bytes * n``).
    expected_logical_bytes: int
    #: Peak native bytes retained by ONE owner block (``frame_bytes *
    #: block_frames``).  The queue/consumer must charge this once per live block.
    max_owner_bytes: int
    #: Peak native bytes retained across an in-flight queue of ``queue_depth``
    #: blocks (``max_owner_bytes * queue_depth``); equals ``max_owner_bytes``
    #: when no queue depth was supplied.
    max_inflight_bytes: int
    #: True iff ``block_frames`` equals the native first-chunk dimension.
    chunk_aligned: bool
    #: Empty when chunk-aligned; otherwise why a fallback block size was chosen.
    fallback_reason: str
    #: True iff a single frame alone exceeds the byte budget (block is 1 frame).
    oversize_frame: bool
    #: The explicit retained-byte budget this plan honored.
    max_block_bytes: int

    @property
    def n_requested(self) -> int:
        return max(0, self.requested_stop - self.requested_start)

    @property
    def n_blocks(self) -> int:
        return len(self.ranges)


def _as_int_tuple(shape: Sequence[int] | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    return tuple(int(s) for s in shape)


def plan_reads(
    frame_count: int,
    frame_shape: Sequence[int] | None,
    dtype: Any,
    chunks: Sequence[int] | None,
    max_block_bytes: int,
    *,
    frame_interval: tuple[int, int] | None = None,
    two_d: bool = False,
    requested_block_frames: int | None = None,
    queue_depth: int | None = None,
) -> ReadPlan:
    """Plan block reads for a contiguous ``[start, stop)`` frame window.

    Parameters
    ----------
    frame_count
        Total logical frames in the dataset (a lone 2-D detector dataset is
        exactly one logical frame — pass ``frame_count=1, two_d=True``).
    frame_shape
        Native per-frame shape, e.g. ``(H, W)``.  Used only to size one frame.
    dtype
        Native ``numpy.dtype`` (or dtype-like) of the detector data; its
        ``itemsize`` sets the native frame byte cost.  No cast is implied.
    chunks
        The HDF5 dataset ``chunks`` tuple, or ``None`` for a contiguous dataset.
        ``chunks[0]`` is the native frame-axis chunk cadence.
    max_block_bytes
        Explicit maximum NATIVE bytes a single owner block may retain (> 0).
    frame_interval
        Optional half-open ``(start, stop)`` sub-window; defaults to the whole
        ``[0, frame_count)``.  Clamped into range.
    two_d
        When True the dataset is a single 2-D exposure: the plan always reads
        exactly one frame (rule 5), regardless of chunks/budget.
    requested_block_frames
        Optional consumer cap on the block size (>= 1).  Never raises the block
        above the layout/budget choice; only lowers it (and marks the plan
        unaligned when it undercuts the native cadence).
    queue_depth
        Optional in-flight block count, used only to report
        ``max_inflight_bytes``; it does not change the per-block size.

    Rules (handoff §4.4): never return zero frames for a non-empty request;
    never exceed the byte budget unless a single frame does (then read exactly
    one and flag it); prefer the native first-chunk cadence when its full block
    fits; non-chunked datasets use a bounded sequential fallback; a 2-D dataset
    always plans one frame; the arithmetic is deterministic and overflow-safe
    (Python big ints); the fixed-16 block is never used as policy.
    """
    frame_count = int(frame_count)
    if frame_count < 0:
        raise ValueError(f"frame_count must be >= 0; got {frame_count}")
    budget = int(max_block_bytes)
    if budget <= 0:
        raise ValueError(f"max_block_bytes must be > 0; got {budget}")

    fshape = _as_int_tuple(frame_shape) or ()
    np_dtype = np.dtype(dtype)
    itemsize = int(np_dtype.itemsize)
    # Overflow-safe: math.prod on Python ints stays exact and unbounded.
    n_pixels = 1
    for dim in fshape:
        n_pixels *= int(dim)
    frame_bytes = n_pixels * itemsize
    # Rule 6 (deterministic): a zero-area frame shape has no valid block-byte
    # arithmetic (the ``budget // frame_bytes`` fallback below would divide by
    # zero).  Reject it explicitly, matching the frame_count/budget guards —
    # a real detector dimension is never 0 (``frame_shape=None`` -> 1 pixel).
    if frame_bytes <= 0:
        raise ValueError(f"frame_shape must have positive pixel area; got {frame_shape!r}")

    # Resolve the requested half-open window, clamped into [0, frame_count].
    if frame_interval is None:
        start, stop = 0, frame_count
    else:
        start, stop = int(frame_interval[0]), int(frame_interval[1])
    start = max(0, min(start, frame_count))
    stop = max(start, min(stop, frame_count))
    n = stop - start

    native = None
    if chunks is not None:
        ctuple = _as_int_tuple(chunks) or ()
        if ctuple and not two_d:
            native = max(1, int(ctuple[0]))

    # Empty request: legitimately zero frames (this is NOT "zero for a non-empty
    # request").  Report a coherent, harmless plan.
    if n <= 0:
        return ReadPlan(
            frame_count=frame_count, requested_start=start, requested_stop=stop,
            frame_bytes=frame_bytes, native_chunk_frames=native, block_frames=0,
            ranges=(), expected_logical_bytes=0, max_owner_bytes=0,
            max_inflight_bytes=0, chunk_aligned=False,
            fallback_reason="empty request", oversize_frame=False,
            max_block_bytes=budget,
        )

    oversize = False
    fallback_reason = ""
    if two_d:
        # Rule 5: a 2-D dataset is exactly one logical frame.
        block_frames = 1
        chunk_aligned = False
        fallback_reason = "2-D detector dataset: exactly one frame"
        if frame_bytes > budget:
            oversize = True
    elif frame_bytes > budget:
        # Rule 2: a single frame exceeds the budget -> read exactly one frame
        # and disclose the oversize fact.
        block_frames = 1
        chunk_aligned = False
        oversize = True
        fallback_reason = (
            f"single frame {frame_bytes}B exceeds budget {budget}B; "
            "reading one frame per block")
    else:
        max_by_budget = budget // frame_bytes  # >= 1 since frame_bytes <= budget
        if native is not None:
            native_block_bytes = native * frame_bytes
            if native_block_bytes <= budget:
                # Rule 3: the full native chunk block fits -> native cadence.
                block_frames = native
                chunk_aligned = True
            else:
                # The native chunk block is larger than the budget allows: use a
                # bounded fallback and disclose why (rule 2 keeps it <= budget).
                block_frames = max_by_budget
                chunk_aligned = False
                fallback_reason = (
                    f"native chunk block {native} frames ({native_block_bytes}B) "
                    f"exceeds budget {budget}B; bounded to {block_frames} frames")
        else:
            # Rule 4: contiguous / non-chunked -> bounded sequential fallback.
            block_frames = max_by_budget
            chunk_aligned = False
            fallback_reason = (
                f"contiguous dataset (no frame-axis chunking); bounded "
                f"sequential fallback of {block_frames} frames")

    # Optional consumer cap: only ever lowers the block, never below one frame.
    if requested_block_frames is not None:
        cap = int(requested_block_frames)
        if cap < 1:
            raise ValueError(
                f"requested_block_frames must be >= 1; got {requested_block_frames}")
        if cap < block_frames:
            # Refresh the reason for EVERY path (aligned OR already-unaligned)
            # so the reported "why" always names the final capped block, never a
            # stale pre-cap frame count.
            if chunk_aligned:
                fallback_reason = (
                    f"native cadence {block_frames} capped to requested "
                    f"{cap} frames")
            else:
                fallback_reason = (
                    f"{fallback_reason}; capped to requested {cap} frames")
            chunk_aligned = False
            block_frames = cap

    ranges = tuple(
        (s, min(s + block_frames, stop)) for s in range(start, stop, block_frames))

    expected_logical_bytes = n * frame_bytes
    max_owner_bytes = block_frames * frame_bytes
    depth = 1 if queue_depth is None else max(1, int(queue_depth))
    max_inflight_bytes = max_owner_bytes * depth

    return ReadPlan(
        frame_count=frame_count,
        requested_start=start,
        requested_stop=stop,
        frame_bytes=frame_bytes,
        native_chunk_frames=native,
        block_frames=block_frames,
        ranges=ranges,
        expected_logical_bytes=expected_logical_bytes,
        max_owner_bytes=max_owner_bytes,
        max_inflight_bytes=max_inflight_bytes,
        chunk_aligned=chunk_aligned,
        fallback_reason=fallback_reason,
        oversize_frame=oversize,
        max_block_bytes=budget,
    )
