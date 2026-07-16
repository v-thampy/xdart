# -*- coding: utf-8 -*-
"""R2 Commit 1 — pure ReadPlan byte/chunk arithmetic (no GUI, no h5py).

These prove the block-sizing rules in handoff §4.4 independently of any HDF5
object: every case here is plain integers + a ``numpy.dtype``.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.sources.read_plan import ReadPlan, plan_reads


def _covers(plan: ReadPlan) -> list[int]:
    """Every frame index the plan's ranges visit, in order."""
    out: list[int] = []
    for start, stop in plan.ranges:
        out.extend(range(start, stop))
    return out


# --------------------------------------------------------------------------- #
# Native chunk cadence (rule 3): block == chunks[0] when its full block fits.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk0", [1, 2, 16, 32])
def test_native_chunk_cadence_dims(chunk0):
    # frame is tiny (16x16 uint16 = 512 B); a generous budget fits every cadence.
    plan = plan_reads(
        frame_count=100, frame_shape=(16, 16), dtype=np.uint16,
        chunks=(chunk0, 16, 16), max_block_bytes=32 * 512)
    assert plan.native_chunk_frames == chunk0
    assert plan.block_frames == chunk0
    assert plan.chunk_aligned is True
    assert plan.fallback_reason == ""
    assert _covers(plan) == list(range(100))
    # every block is <= the native cadence, never larger
    assert all((stop - start) <= chunk0 for start, stop in plan.ranges)


def test_native_chunk_block_exceeding_budget_bounded():
    # 32-frame chunk of a 512 B frame = 16384 B, but budget only fits 4 frames.
    plan = plan_reads(
        frame_count=40, frame_shape=(16, 16), dtype=np.uint16,
        chunks=(32, 16, 16), max_block_bytes=4 * 512)
    assert plan.native_chunk_frames == 32
    assert plan.block_frames == 4  # bounded to budget, not the 32-frame chunk
    assert plan.chunk_aligned is False
    assert "exceeds budget" in plan.fallback_reason
    assert plan.max_owner_bytes == 4 * 512
    assert plan.max_owner_bytes <= plan.max_block_bytes


# --------------------------------------------------------------------------- #
# Contiguous / non-chunked (rule 4): bounded sequential fallback.
# --------------------------------------------------------------------------- #
def test_contiguous_bounded_fallback():
    plan = plan_reads(
        frame_count=50, frame_shape=(10, 10), dtype=np.uint32,  # 400 B/frame
        chunks=None, max_block_bytes=10 * 400)
    assert plan.native_chunk_frames is None
    assert plan.block_frames == 10
    assert plan.chunk_aligned is False
    assert "contiguous" in plan.fallback_reason
    assert _covers(plan) == list(range(50))


def test_partial_tail_block_is_smaller():
    plan = plan_reads(
        frame_count=17, frame_shape=(4, 4), dtype=np.uint16,
        chunks=(5, 4, 4), max_block_bytes=100 * 32)
    # 5-frame blocks: (0,5)(5,10)(10,15)(15,17) — last is a 2-frame tail.
    assert plan.ranges[-1] == (15, 17)
    assert _covers(plan) == list(range(17))


def test_sub_window_starts_at_requested_offset():
    plan = plan_reads(
        frame_count=100, frame_shape=(8, 8), dtype=np.uint16,
        chunks=(4, 8, 8), max_block_bytes=100 * 128,
        frame_interval=(20, 33))
    assert plan.requested_start == 20
    assert plan.requested_stop == 33
    assert plan.ranges[0][0] == 20
    assert _covers(plan) == list(range(20, 33))
    assert plan.expected_logical_bytes == 13 * (8 * 8 * 2)


def test_interval_clamped_into_range():
    plan = plan_reads(
        frame_count=10, frame_shape=(2, 2), dtype=np.uint16,
        chunks=(2, 2, 2), max_block_bytes=10_000, frame_interval=(-5, 999))
    assert plan.requested_start == 0
    assert plan.requested_stop == 10
    assert _covers(plan) == list(range(10))


# --------------------------------------------------------------------------- #
# Oversize + 2-D (rules 2, 5).
# --------------------------------------------------------------------------- #
def test_one_frame_over_budget_reads_exactly_one():
    # 100x100 uint32 = 40000 B; budget only 10000 B -> a single frame is oversize.
    plan = plan_reads(
        frame_count=5, frame_shape=(100, 100), dtype=np.uint32,
        chunks=(2, 100, 100), max_block_bytes=10_000)
    assert plan.oversize_frame is True
    assert plan.block_frames == 1
    assert plan.chunk_aligned is False
    assert "exceeds budget" in plan.fallback_reason
    assert _covers(plan) == [0, 1, 2, 3, 4]
    # the ONLY case where a block may exceed the budget is this one frame
    assert plan.max_owner_bytes == 40000


def test_two_d_always_one_frame():
    plan = plan_reads(
        frame_count=1, frame_shape=(64, 64), dtype=np.uint16,
        chunks=(64, 64), max_block_bytes=10**9, two_d=True)
    assert plan.block_frames == 1
    assert plan.ranges == ((0, 1),)
    assert plan.chunk_aligned is False
    assert "2-D" in plan.fallback_reason
    assert plan.native_chunk_frames is None  # 2-D: no frame-axis chunk cadence


# --------------------------------------------------------------------------- #
# dtype-driven byte accounting.
# --------------------------------------------------------------------------- #
def test_uint16_vs_uint32_frame_bytes():
    p16 = plan_reads(10, (10, 10), np.uint16, (2, 10, 10), 10**6)
    p32 = plan_reads(10, (10, 10), np.uint32, (2, 10, 10), 10**6)
    assert p16.frame_bytes == 200
    assert p32.frame_bytes == 400
    assert p32.frame_bytes == 2 * p16.frame_bytes


# --------------------------------------------------------------------------- #
# Invariants (rules 1, 2, 6).
# --------------------------------------------------------------------------- #
def test_never_zero_frames_for_nonempty_request():
    # a budget smaller than one frame still yields a 1-frame block, never 0.
    plan = plan_reads(3, (50, 50), np.uint32, (1, 50, 50), max_block_bytes=1)
    assert plan.block_frames == 1
    assert plan.n_blocks == 3
    assert _covers(plan) == [0, 1, 2]


def test_never_exceeds_budget_except_oversize():
    plan = plan_reads(100, (10, 10), np.uint16, (7, 10, 10), max_block_bytes=3000)
    assert plan.oversize_frame is False
    assert plan.max_owner_bytes <= plan.max_block_bytes


def test_empty_request_yields_zero_frames():
    plan = plan_reads(10, (4, 4), np.uint16, (2, 4, 4), 10_000, frame_interval=(3, 3))
    assert plan.block_frames == 0
    assert plan.ranges == ()
    assert plan.expected_logical_bytes == 0
    assert plan.max_owner_bytes == 0


def test_zero_frame_dataset():
    plan = plan_reads(0, (4, 4), np.uint16, (2, 4, 4), 10_000)
    assert plan.ranges == ()
    assert plan.block_frames == 0


# --------------------------------------------------------------------------- #
# Owner-byte + queue accounting.
# --------------------------------------------------------------------------- #
def test_owner_bytes_and_inflight_scaling():
    plan = plan_reads(64, (16, 16), np.uint16, (8, 16, 16), 8 * 512, queue_depth=4)
    assert plan.block_frames == 8
    assert plan.max_owner_bytes == 8 * 512
    assert plan.max_inflight_bytes == 8 * 512 * 4


def test_no_queue_depth_inflight_equals_owner():
    plan = plan_reads(64, (16, 16), np.uint16, (8, 16, 16), 8 * 512)
    assert plan.max_inflight_bytes == plan.max_owner_bytes


def test_requested_block_cap_lowers_and_unaligns():
    plan = plan_reads(
        64, (16, 16), np.uint16, (16, 16, 16), 32 * 512, requested_block_frames=4)
    assert plan.block_frames == 4
    assert plan.chunk_aligned is False
    assert "capped" in plan.fallback_reason


def test_requested_block_cap_above_choice_is_noop():
    plan = plan_reads(
        64, (16, 16), np.uint16, (4, 16, 16), 32 * 512, requested_block_frames=100)
    assert plan.block_frames == 4  # native cadence unchanged
    assert plan.chunk_aligned is True


# --------------------------------------------------------------------------- #
# Determinism + validation.
# --------------------------------------------------------------------------- #
def test_deterministic_equal_plans():
    a = plan_reads(31, (128, 128), np.uint32, (2, 128, 128), 4 * 128 * 128 * 4)
    b = plan_reads(31, (128, 128), np.uint32, (2, 128, 128), 4 * 128 * 128 * 4)
    assert a == b  # frozen dataclass value equality


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        plan_reads(5, (4, 4), np.uint16, (2, 4, 4), max_block_bytes=0)


def test_negative_frame_count_rejected():
    with pytest.raises(ValueError):
        plan_reads(-1, (4, 4), np.uint16, None, 1000)


def test_requested_block_frames_must_be_positive():
    with pytest.raises(ValueError):
        plan_reads(5, (4, 4), np.uint16, (2, 4, 4), 10_000, requested_block_frames=0)
