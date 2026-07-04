"""RAM-aware sizing for the live heavy-staging caps (MEM-2).

One function decides the window; the three live heavy caps consume it —
``LiveFrameSeries._in_memory_cap`` (write-side staging), and the
``max_heavy_items`` of both ``FrameRecordStore`` and ``PublicationStore``.
Before MEM-2 all three hardcoded ``64``.

The window is ~25% of TOTAL physical RAM (total, not available — stable and
predictable across machines) divided by the AS-STORED per-frame heavy cost,
clamped to ``[16, 64]``.  A tiny detector never shrinks it below today's 64; a
small-RAM box shrinks it toward 16 so the staging set can't exhaust memory.

Pure + headless: only stdlib ``os`` (RAM detection), no numpy / Qt.
"""

from __future__ import annotations

import os

#: today's behavior + the clamp bounds for the computed window.
DEFAULT_WINDOW = 64
MIN_WINDOW = 16
MAX_WINDOW = 64
#: the manual override (``XDART_HEAVY_WINDOW``) may go wider than the computed clamp.
OVERRIDE_MIN = 8
OVERRIDE_MAX = 128
#: fraction of TOTAL physical RAM the heavy staging may occupy.
RAM_FRACTION = 0.25
ENV_OVERRIDE = "XDART_HEAVY_WINDOW"

_GIB = 1024 ** 3


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def total_physical_ram_bytes() -> int | None:
    """Total physical RAM in bytes, or ``None`` if it can't be detected."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def heavy_window(
    frame_bytes: int | None = None,
    *,
    total_ram_bytes: int | None = None,
    env=None,
) -> int:
    """Return the live heavy-staging window size.

    Resolution order:

    1. ``XDART_HEAVY_WINDOW`` (int) pins it manually, clamped to
       ``[8, 128]`` (wider than the computed clamp, for testing/simulation).
    2. else ``clamp(int(0.25 * total_RAM / frame_bytes), 16, 64)`` when
       ``frame_bytes`` is known (> 0) — the frame shape is known at run start.
    3. else RAM tiers (shape unknown): < 16 GiB → 16, < 32 GiB → 32, else 64.
    4. RAM detection failure → 64 (today's behavior).

    ``total_ram_bytes`` / ``env`` are injectable for testing.
    """
    source = env if env is not None else os.environ
    raw = source.get(ENV_OVERRIDE)
    if raw is not None and str(raw).strip():
        try:
            return _clamp(int(str(raw).strip()), OVERRIDE_MIN, OVERRIDE_MAX)
        except (TypeError, ValueError):
            pass  # malformed override → fall through to the computed window

    total = (
        total_ram_bytes
        if total_ram_bytes is not None
        else total_physical_ram_bytes()
    )
    if not total or total <= 0:
        return DEFAULT_WINDOW  # detection failed → today's behavior

    if frame_bytes and frame_bytes > 0:
        budget = RAM_FRACTION * total
        return _clamp(budget / frame_bytes, MIN_WINDOW, MAX_WINDOW)

    # Frame shape unknown at decision time → coarse RAM tiers.
    gib = total / _GIB
    if gib < 16:
        return MIN_WINDOW
    if gib < 32:
        return 32
    return DEFAULT_WINDOW


def heavy_window_log_line(
    window: int,
    frame_bytes: int | None,
    *,
    total_ram_bytes: int | None = None,
    overridden: bool = False,
) -> str:
    """One-line run-start summary, e.g.
    ``"heavy window: 32 frames (32 GB RAM, 64 MB/frame stored)"``."""
    total = total_ram_bytes if total_ram_bytes is not None else total_physical_ram_bytes()
    ram = f"{total / 1e9:.0f} GB RAM" if total else "RAM unknown"
    frame = (
        f"{frame_bytes / 1e6:.0f} MB/frame stored"
        if frame_bytes
        else "frame size unknown"
    )
    tag = " [XDART_HEAVY_WINDOW override]" if overridden else ""
    return f"heavy window: {window} frames ({ram}, {frame}){tag}"


# ── MEM-3: reduction worker-pool sizing ─────────────────────────────────────
#: measured throughput knee — past ~4 workers each one adds ~1 GB (an integrator
#: deepcopy + its in-flight scratch) for ~0 speed (651-frame Eiger: 4w=25.0s,
#: 8w=26.1s, 16w=26.8s; peak RSS 9/14/19 GB).
DEFAULT_REDUCTION_WORKERS = 4
MAX_REDUCTION_WORKERS = 16
REDUCTION_WORKERS_ENV = "XDART_REDUCTION_WORKERS"
#: below this TOTAL RAM the pool is floored to 2 (each worker ~1 GB of
#: duplicated integrator geometry — the budget pressure heavy_window responds to).
_SMALL_RAM_FLOOR_BYTES = 16 * _GIB


def reduction_worker_cap(
    requested: int | None = None,
    *,
    total_ram_bytes: int | None = None,
    env=None,
) -> int:
    """Number of reduction worker threads.

    Each worker deep-copies the pyFAI integrator (thread-safety workaround), so
    the pool is a memory consumer sharing the same RAM pressure as the heavy
    staging window.  Resolution:

    1. ``XDART_REDUCTION_WORKERS`` pins it, clamped ``[1, 16]``.
    2. an explicit ``requested`` count (the user's Cores) wins, clamped to
       ``[1, min(16, cpu)]`` — the cap replaces only the silent default, never a
       deliberate user choice.
    3. else the default knee ``min(4, cpu)``.
    4. a small-RAM box (< 16 GiB total) floors the result to 2.

    ALWAYS returns >= 1 — never ``None`` (the ``None`` path was the latent
    20-worker-default bug: ``n_workers==1 -> executor=None ->`` a
    ``min(32, cpu+4)`` pool).
    """
    source = env if env is not None else os.environ
    raw = source.get(REDUCTION_WORKERS_ENV)
    if raw is not None and str(raw).strip():
        try:
            return _clamp(int(str(raw).strip()), 1, MAX_REDUCTION_WORKERS)
        except (TypeError, ValueError):
            pass  # malformed override → fall through

    cpu = os.cpu_count() or DEFAULT_REDUCTION_WORKERS
    hard = min(MAX_REDUCTION_WORKERS, cpu)
    if requested is not None and int(requested) > 0:
        want = min(int(requested), hard)              # user's Cores wins
    else:
        want = min(DEFAULT_REDUCTION_WORKERS, cpu)     # the knee

    total = (
        total_ram_bytes
        if total_ram_bytes is not None
        else total_physical_ram_bytes()
    )
    if total and total < _SMALL_RAM_FLOOR_BYTES:
        want = min(want, 2)
    return max(1, want)


def reduction_worker_cap_log_line(workers, requested=None, *, overridden=False):
    """One-line run-start summary of the reduction pool size."""
    tag = " [XDART_REDUCTION_WORKERS override]" if overridden else ""
    req = f" (Cores requested {requested})" if requested else ""
    return f"reduction workers: {workers}{req}{tag}"


__all__ = [
    "heavy_window",
    "heavy_window_log_line",
    "total_physical_ram_bytes",
    "reduction_worker_cap",
    "reduction_worker_cap_log_line",
    "DEFAULT_WINDOW",
    "MIN_WINDOW",
    "MAX_WINDOW",
    "ENV_OVERRIDE",
    "DEFAULT_REDUCTION_WORKERS",
    "MAX_REDUCTION_WORKERS",
    "REDUCTION_WORKERS_ENV",
]
