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


__all__ = [
    "heavy_window",
    "heavy_window_log_line",
    "total_physical_ram_bytes",
    "DEFAULT_WINDOW",
    "MIN_WINDOW",
    "MAX_WINDOW",
    "ENV_OVERRIDE",
]
