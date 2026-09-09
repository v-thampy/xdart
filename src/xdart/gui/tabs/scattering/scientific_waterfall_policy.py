"""Toolkit-free scientific waterfall activation policy."""

from __future__ import annotations


def waterfall_should_be_active(
    plot_mode: str,
    trace_count: int,
    *,
    was_active: bool,
    viewer_1d: bool = False,
) -> bool:
    """Return the exact production bottom-panel waterfall state.

    Explicit Waterfall starts on the fourth trace (second for 1D Viewer).
    Overlay and legacy
    multi-selected Single start on the sixteenth, then retain the image view
    through eight traces and return to curves at seven. Aggregate modes never
    use the waterfall view.
    """

    count = max(0, int(trace_count))
    if plot_mode == "Waterfall":
        return count >= (2 if viewer_1d else 4)
    if plot_mode in {"Average", "Sum"}:
        return False
    if plot_mode in {"Overlay", "Single"}:
        return count >= (8 if was_active else 16)
    return False


__all__ = ["waterfall_should_be_active"]
