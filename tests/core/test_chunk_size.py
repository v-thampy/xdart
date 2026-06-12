"""Tests for xrd_tools.io.chunk_size.adaptive_chunk_size."""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.io.chunk_size import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_FRAMES,
    adaptive_chunk_size,
)


class TestAdaptiveChunkSize:
    def test_pilatus_default_budget_returns_max_frames(self):
        """Pilatus 300k (195 × 487 × 8B) easily fits 64 frames in 512 MB."""
        chunk = adaptive_chunk_size((195, 487))
        assert chunk == DEFAULT_MAX_FRAMES

    def test_eiger_streaming_4arrays_returns_double_digits(self):
        """514 × 1030 × 4 arrays × float64 × 256 MB → about 15 frames."""
        chunk = adaptive_chunk_size(
            (514, 1030), n_arrays=4, memory_budget_mb=256,
        )
        # Hand calc: 514 * 1030 * 4 * 8 = 16,941,440 B/frame
        # 256 MB / that = 15.83 → floor → 15
        assert chunk == 15

    def test_budget_pressure_clamps_to_min_frames(self):
        """If even one frame exceeds the budget, return min_frames not 1."""
        # Pretend the detector + n_arrays + dtype is huge
        chunk = adaptive_chunk_size(
            (4096, 4096), n_arrays=4, memory_budget_mb=4,  # tiny budget
        )
        assert chunk == DEFAULT_MIN_FRAMES

    def test_tiny_detector_clamps_to_max_frames(self):
        """A 64 × 64 detector with huge budget shouldn't return 10000."""
        chunk = adaptive_chunk_size((64, 64), memory_budget_mb=8192)
        assert chunk == DEFAULT_MAX_FRAMES

    def test_dtype_changes_chunk_count(self):
        """Half-precision dtype halves the per-frame bytes → 2× chunk."""
        chunk64 = adaptive_chunk_size(
            (1024, 1024), dtype=np.float64, memory_budget_mb=128,
            n_arrays=4, min_frames=1, max_frames=10000,
        )
        chunk32 = adaptive_chunk_size(
            (1024, 1024), dtype=np.float32, memory_budget_mb=128,
            n_arrays=4, min_frames=1, max_frames=10000,
        )
        # Within rounding, chunk32 should be ≈ 2 × chunk64
        assert chunk32 // chunk64 == 2

    def test_n_arrays_changes_chunk_count(self):
        """4 arrays at once → 4x less chunk than 1 array (within rounding)."""
        chunk_1 = adaptive_chunk_size(
            (512, 512), n_arrays=1, memory_budget_mb=64,
            min_frames=1, max_frames=10000,
        )
        chunk_4 = adaptive_chunk_size(
            (512, 512), n_arrays=4, memory_budget_mb=64,
            min_frames=1, max_frames=10000,
        )
        assert chunk_1 // chunk_4 == 4

    def test_custom_min_max_respected(self):
        """min_frames=2, max_frames=8 caps result regardless of budget."""
        chunk = adaptive_chunk_size(
            (32, 32), memory_budget_mb=8192,
            min_frames=2, max_frames=8,
        )
        assert chunk == 8
        # And the floor case
        chunk = adaptive_chunk_size(
            (4096, 4096), n_arrays=8, memory_budget_mb=1,
            min_frames=2, max_frames=8,
        )
        assert chunk == 2

    @pytest.mark.parametrize("h,w", [(0, 100), (100, 0), (-1, 100)])
    def test_invalid_shape_raises(self, h, w):
        with pytest.raises(ValueError, match="detector_shape components"):
            adaptive_chunk_size((h, w))

    def test_invalid_n_arrays_raises(self):
        with pytest.raises(ValueError, match="n_arrays must be >= 1"):
            adaptive_chunk_size((100, 100), n_arrays=0)

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError, match="memory_budget_mb must be > 0"):
            adaptive_chunk_size((100, 100), memory_budget_mb=0)

    def test_invalid_min_max_raises(self):
        with pytest.raises(ValueError, match="min_frames"):
            adaptive_chunk_size((100, 100), min_frames=0, max_frames=10)
        with pytest.raises(ValueError, match="min_frames"):
            adaptive_chunk_size((100, 100), min_frames=10, max_frames=5)

    def test_re_export_from_io_package(self):
        """``from xrd_tools.io import adaptive_chunk_size`` works."""
        from xrd_tools.io import adaptive_chunk_size as ac
        assert ac is adaptive_chunk_size
