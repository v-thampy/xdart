"""Adaptive chunk-size helper for batched image processing.

xdart's SPEC batch loop, the RSM streaming gridder, and any future
"process N frames at a time" path want a chunk size that:

* doesn't exceed a memory budget (typically a few hundred MB per
  worker), and
* doesn't shrink to one frame on tiny detectors where the per-chunk
  overhead would dominate.

This module exposes :func:`adaptive_chunk_size`, a single function
that takes a detector shape + dtype + memory budget and returns a
sensible chunk size clamped to a configurable ``[min, max]`` range.

Callers like xdart's wrangler can replace hard-coded values (16 SPEC
frames, 8 RSM frames) with::

    from xrd_tools.io import adaptive_chunk_size

    chunk = adaptive_chunk_size(
        detector_shape=(514, 1030),
        n_arrays=4,            # 1 image + 3 q-arrays for RSM streaming
        memory_budget_mb=512,
    )
"""
from __future__ import annotations

import numpy as np


# Each chunk-resident array is roughly ``chunk × H × W × itemsize``
# bytes.  For RSM streaming we hold 1 image + 3 q-arrays at once, so
# ``n_arrays=4`` is the right default for that path; pure integration
# (xdart's SPEC batch) holds just the image stack, so ``n_arrays=1``.
DEFAULT_N_ARRAYS: int = 1
DEFAULT_MEMORY_BUDGET_MB: float = 512.0
DEFAULT_MIN_FRAMES: int = 4
DEFAULT_MAX_FRAMES: int = 64


def adaptive_chunk_size(
    detector_shape: tuple[int, int],
    *,
    dtype: np.dtype | type | str = np.float64,
    n_arrays: int = DEFAULT_N_ARRAYS,
    memory_budget_mb: float = DEFAULT_MEMORY_BUDGET_MB,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> int:
    """Return a chunk size that fits a per-worker memory budget.

    Parameters
    ----------
    detector_shape : (H, W)
        Per-frame image shape in pixels.  Both axes must be positive.
    dtype : numpy dtype, default float64
        The element size used in the memory calculation.  Pass
        ``np.float32`` for half-precision pipelines.
    n_arrays : int, default 1
        How many ``(chunk × H × W)`` arrays the consumer holds at the
        same time.  ``1`` for plain integration, ``4`` for RSM
        streaming (1 image + qx/qy/qz), etc.
    memory_budget_mb : float, default 512
        Upper bound on per-worker memory usage in megabytes.  The
        chunk is sized so ``chunk × H × W × itemsize × n_arrays`` does
        not exceed this.
    min_frames, max_frames : int
        Hard clamps applied after the budget calculation.  Defaults
        ``[4, 64]`` cover everything from tiny detectors (where small
        chunks would be silly because per-chunk overhead dominates) to
        very large ones (where huge chunks would defeat the streaming
        purpose).

    Returns
    -------
    int
        Recommended chunk size, in ``[min_frames, max_frames]``.

    Examples
    --------
    >>> # 514x1030 Eiger frame, 4 arrays (RSM streaming), 256 MB budget
    >>> adaptive_chunk_size((514, 1030), n_arrays=4,
    ...                     memory_budget_mb=256)  # doctest: +SKIP
    15
    >>> # Pilatus 300k (195x487), pure integration, default budget
    >>> adaptive_chunk_size((195, 487))  # doctest: +SKIP
    64
    """
    h, w = int(detector_shape[0]), int(detector_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(
            f"detector_shape components must be > 0; got ({h}, {w})"
        )
    if n_arrays < 1:
        raise ValueError(f"n_arrays must be >= 1; got {n_arrays}")
    if memory_budget_mb <= 0:
        raise ValueError(
            f"memory_budget_mb must be > 0; got {memory_budget_mb}"
        )
    if min_frames < 1 or max_frames < min_frames:
        raise ValueError(
            f"need 1 <= min_frames <= max_frames; "
            f"got min={min_frames}, max={max_frames}"
        )

    itemsize = int(np.dtype(dtype).itemsize)
    per_frame_bytes = h * w * itemsize * int(n_arrays)
    budget_bytes = float(memory_budget_mb) * 1024.0 * 1024.0
    raw = int(budget_bytes // per_frame_bytes)

    # Clamp to [min, max] inclusive.  Even if the budget allows just
    # one frame we still recommend min_frames — going under the floor
    # makes overhead dominate and isn't worth the saving.
    return max(min_frames, min(max_frames, max(1, raw)))


__all__ = ["adaptive_chunk_size"]
