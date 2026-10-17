# -*- coding: utf-8 -*-
"""Scan-group range syntax (pure, Qt-free).

``parse_scan_groups("1-3, 5, 7-9") -> [[1,2,3], [5], [7,8,9]]`` — each
comma-separated token is one **group** (a single scan or an inclusive ``a-b``
range), and each group is *combined* into one output (a
:class:`~xrd_tools.sources.composite.CompositeFrameSource`) by stitch/RSM.  The
ROI plotter uses a single scan and ignores grouping.
"""

from __future__ import annotations

import re


def parse_stitch_scan_groups(text: str) -> tuple[tuple[str, ...], ...]:
    """Bracketed scans share one result; every unbracketed scan is individual.

    ``1,2,[2-5,8]`` gives three outputs. Repeated membership across outputs is
    intentional. Exact SPEC occurrence keys (``2.3``) are accepted as singles.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 32768:
        raise ValueError("Enter scan numbers and bracketed groups")
    groups, group, token = [], None, ""
    count = 0

    def consume():
        nonlocal token, count
        value = token.strip()
        token = ""
        if not value:
            raise ValueError("Empty scan selector")
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", value)
        if match:
            first, last = int(match[1]), int(match[2] or match[1])
            if first < 1 or last < first or last - first >= 4096:
                raise ValueError("Scan ranges must increase and contain at most 4096 scans")
            values = tuple(str(i) for i in range(first, last + 1))
        elif re.fullmatch(r"[1-9]\d*\.[1-9]\d*", value):
            values = (value,)
        else:
            raise ValueError(f"Invalid scan selector: {value}")
        count += len(values)
        if count > 4096:
            raise ValueError("Scan selection exceeds 4096 memberships")
        if group is None:
            groups.extend((value,) for value in values)
        else:
            group.extend(values)

    after_group = False
    for character in text.strip():
        if character == "[":
            if group is not None or token.strip() or after_group:
                raise ValueError("Groups must be separate, non-nested bracket lists")
            group = []
        elif character == "]":
            if group is None:
                raise ValueError("Unmatched closing bracket")
            consume()
            groups.append(tuple(dict.fromkeys(group)))
            group = None
            after_group = True
        elif character == ",":
            if after_group:
                after_group = False
            else:
                consume()
        elif after_group and not character.isspace():
            raise ValueError("Separate groups with a comma")
        else:
            token += character
    if group is not None:
        raise ValueError("Unclosed scan group")
    if not after_group:
        consume()
    return tuple(groups)


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


__all__ = ["parse_scan_groups", "flatten_scan_groups", "parse_stitch_scan_groups"]
