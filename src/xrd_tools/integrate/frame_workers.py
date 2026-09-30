"""A stitched reduction's frames on a pool of threads, read and merged in frame order (PERF-STITCH-2/3).

Int's reduction delivers its frames the same way: one reader, per-frame workers, an in-order drain. Both
Stitch kernels use it: pyFAI MultiGeometry's integrations (``xrd_tools.integrate.multi``) and the
xrayutilities backend's NumPy histograms (``xrd_tools.integrate.xu_stitch``), whose xrayutilities work
stays on the reading thread, the one its runtime session belongs to.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor


def parallel_frames(count, read, compute, merge, workers, *, read_ahead=None):
    """``compute(index, *read(index))`` for the ``count`` frames on ``workers`` threads, while this thread
    ``read``s each frame and ``merge``s the results in frame order. At most ``read_ahead`` frames (default:
    two a worker) are read and not yet merged. A failure (a frame's, a read's, a cancellation) cancels the
    frames not started, waits for the running ones and is raised. Frames merge in order, so of two failing
    frames the earlier one's error is raised; a read failure (or a cancellation) during read-ahead may come
    before an earlier frame's own failure (Codex terminal review): either way nothing after it is merged."""
    limit = 2 * workers if read_ahead is None else read_ahead
    pending = deque()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="xdart-stitch-frame")
    try:
        for index in range(count):
            item = read(index)
            pending.append(pool.submit(compute, index, *item))
            del item
            while pending and (len(pending) >= limit or pending[0].done()):
                merge(pending.popleft().result())
        while pending:
            merge(pending.popleft().result())
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


__all__ = ["parallel_frames"]
