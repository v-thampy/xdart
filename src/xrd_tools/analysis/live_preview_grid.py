"""Grid allocation for provisional Live science."""

from __future__ import annotations

import math


def preview_rsm_bins(requested: tuple[int, int, int], limit: int = 1_000_000) -> tuple[int, int, int]:
    """Fit the requested proportions into a voxel cap while protecting each axis."""
    if math.prod(requested) <= limit:
        return requested

    def scaled(factor):
        return tuple(max(2, min(count, math.ceil(count * factor - .5))) for count in requested)

    high = math.cbrt(limit / math.prod(requested))
    bins = scaled(high)
    if math.prod(bins) <= limit:
        return bins
    # The lower bound records an already admissible integer grid at every step.
    low = 0.0
    bins = scaled(low)
    for _ in range(54):
        middle = (low + high) / 2
        trial = scaled(middle)
        if math.prod(trial) <= limit:
            low, bins = middle, trial
        else:
            high = middle
    return bins


def enclosing_preview_bounds(
    footprint: dict[str, tuple[float, float]],
    automatic: frozenset[str],
    prior: dict[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    """Allocate and extend automatic intervals with headroom and geometric growth."""
    bounds = {} if prior is None else dict(prior)
    for name, (lo, hi) in footprint.items():
        if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
            raise ValueError(f"Invalid preview footprint for {name}")
        old = bounds.get(name)
        if name not in automatic:
            if hi <= lo:
                raise ValueError(f"Fixed preview range for {name} must be increasing")
            if old is None:
                bounds[name] = lo, hi
            continue
        if old is not None and old[0] <= lo and hi <= old[1]:
            continue
        # Use an absolute floor for the first degenerate footprint. A later
        # measurement can extend it without reinterpreting the original grid.
        margin = max((hi - lo) * .0625, max(abs(lo), abs(hi), 1.0) * 1e-6)
        lower, upper = lo - margin, hi + margin
        if old is not None:
            lower, upper = min(old[0], lower), max(old[1], upper)
            growth = max(0.0, 2 * (old[1] - old[0]) - (upper - lower)) / 2
            if growth > 0:
                lower = math.nextafter(lower - growth, -math.inf)
                upper = math.nextafter(upper + growth, math.inf)
        bounds[name] = lower, upper
    return bounds
