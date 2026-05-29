# -*- coding: utf-8 -*-
"""Per-worker pyFAI integrator pool for thread-safe parallel batch mode.

pyFAI's :class:`AzimuthalIntegrator` is **not** thread-safe across
different inputs on a shared instance.  Calling
``integrate1d_ng(method='csr')`` mutates intermediate scratch buffers
on ``self`` during the call; if two worker threads hit the same
instance with different images, the buffers get clobbered and the
returned intensities diverge from a serial baseline by up to ~67%
relative on random data.  Verified empirically on pyFAI 2025/2026.

The fix is one integrator per worker.  This module provides:

* :class:`IntegratorPool` — a thread-safe queue of N integrator
  deep-copies handed out via a ``borrow()`` context manager.
* :func:`ensure_integrator_pool` — convenience that caches the pool
  on an arbitrary scan-like object so its lifetime is one scan
  (not one batch dispatch), amortising the ~250 ms first-call CSR
  LUT cost across all subsequent integrations.

Lives in :mod:`xdart.utils` rather than the wrangler module because
the wrangler imports pyqtgraph transitively, which we don't want to
drag into a test fixture or a standalone profiler that just wants to
exercise the pool.
"""

from __future__ import annotations

import copy
import queue
from contextlib import contextmanager
from typing import Any, Iterator


class IntegratorPool:
    """Thread-safe pool of pyFAI integrator copies for parallel batch.

    Construct with the source integrator and a worker count; produces
    ``n`` deep-copies and parks them on a :class:`queue.Queue` for
    :meth:`borrow`-based access.

    Use as a context manager::

        with pool.borrow() as ai:
            frame.integrate_1d(integrator=ai, ...)

    :meth:`borrow` blocks if all integrators are out — never happens
    in practice because the caller's executor is bounded to the same
    ``n``.  The source integrator itself is intentionally **not** put
    in the pool; it's reserved for non-parallel consumers (GUI
    display, single-threaded re-integration) and the wrangler can
    safely detach pool members from per-frame frames without leaking
    a pool reference into long-lived scan state.
    """

    __slots__ = ("_integrators", "_q", "_size")

    def __init__(self, source_integrator: Any, n: int):
        if n < 1:
            raise ValueError("pool size must be >= 1")
        self._integrators = [
            copy.deepcopy(source_integrator) for _ in range(n)
        ]
        self._q: queue.Queue = queue.Queue()
        for ai in self._integrators:
            self._q.put(ai)
        self._size = n

    def __len__(self) -> int:
        return self._size

    @contextmanager
    def borrow(self) -> Iterator[Any]:
        """Yield an integrator; return it to the pool on context exit."""
        ai = self._q.get()
        try:
            yield ai
        finally:
            self._q.put(ai)


def ensure_integrator_pool(holder: Any, source_attr: str, n_workers: int,
                           pool_attr: str = "_integrator_pool"):
    """Build or reuse an :class:`IntegratorPool` cached on ``holder``.

    The pool lives as ``holder.<pool_attr>`` so it's shared across
    every batch dispatch in the same scan.  If a pool already exists
    with the correct worker count, it's reused; otherwise rebuilt
    from ``getattr(holder, source_attr)`` (the source integrator).

    Returns ``None`` if the source integrator hasn't been built yet
    (caller should fall back to serial dispatch).

    Typical use in a wrangler::

        pool = ensure_integrator_pool(scan, '_cached_integrator',
                                       n_workers=self.max_cores)
        if pool is None:
            return self._dispatch_batch_serial(scan, pending)
        ...
    """
    base = getattr(holder, source_attr, None)
    if base is None:
        return None
    pool = getattr(holder, pool_attr, None)
    if pool is not None and len(pool) == n_workers:
        return pool
    new_pool = IntegratorPool(base, n_workers)
    setattr(holder, pool_attr, new_pool)
    return new_pool
