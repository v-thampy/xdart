# -*- coding: utf-8 -*-
"""Scan-group range syntax (pure, Qt-free).

``parse_scan_groups("1-3, 5, 7-9") -> [[1,2,3], [5], [7,8,9]]`` — each
comma-separated token is one **group** (a single scan or an inclusive ``a-b``
range), and each group is *combined* into one output (a
:class:`~xrd_tools.sources.composite.CompositeFrameSource`) by stitch/RSM.  The
ROI plotter uses a single scan and ignores grouping.
"""

from __future__ import annotations


def parse_scan_groups(text: str) -> list[list[int]]:
    """Parse a scan-range spec into groups of scan numbers.

    ``"1-3, 5, 7-9"`` → ``[[1, 2, 3], [5], [7, 8, 9]]``.  Whitespace tolerant; an
    empty token is skipped; ``a-b`` expands inclusively (reversed ranges are
    normalised).  Raises ``ValueError`` on a non-integer token."""
    groups: list[list[int]] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
            if hi < lo:
                lo, hi = hi, lo
            groups.append(list(range(lo, hi + 1)))
        else:
            groups.append([int(token)])
    return groups


def flatten_scan_groups(groups: list[list[int]]) -> list[int]:
    """The de-duplicated, order-preserving flat scan list of all groups."""
    seen: set[int] = set()
    out: list[int] = []
    for group in groups:
        for n in group:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


__all__ = ["parse_scan_groups", "flatten_scan_groups"]
