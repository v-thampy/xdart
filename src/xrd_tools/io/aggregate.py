# -*- coding: utf-8 -*-
"""Whole-scan aggregation (Sum / Average) over a processed scan's PRIMARY
integrated stack — NaN-aware, off a single bounded read.

This is the headless half of greenfield Step 7b: it replaces the legacy
per-frame ``data_1d`` / ``data_2d`` display aggregation.  The primary GI mode's
*complete* stack lives on disk at the top-level ``integrated_1d`` /
``integrated_2d`` groups, so a whole-scan Sum/Average reads it in ONE call
(:func:`xrd_tools.io.read.get_1d` / :func:`get_2d` with ``frame=None``) and
reduces it — O(one read), not O(N per-frame hydrations).

A LIVE, in-progress scan has frames that are reduced but not yet flushed to disk;
the caller passes those as ``extra`` (the in-memory unflushed tail), and they are
concatenated onto the on-disk stack **deduped by frame label** (a freshly-flushed
frame can be both on disk and still resident in memory).  With persist-before-
evict, on-disk-prefix ⊕ in-memory-tail covers ALL frames without holding them in
RAM and without depending on the bounded display store — which is exactly why
this is the correct fix-forward for the bounded-store aggregation-truncation bug.

**Primary-mode-scoped only** (ADR-0003): a non-primary GI mode's per-frame
results are lazy/partial, so its on-disk stack is incomplete.  A whole-scan
aggregate in a non-primary mode must NOT be served from this partial stack — the
caller is responsible for detecting that and deferring (never silently
truncating).  This module only reads the top-level (primary) stack.

The 2D aggregate is returned in the file/cake convention ``(n_chi, n_q)`` (same
as :func:`get_2d`); a display caller that uses the ``(q, chi)`` IntegrationResult2D
convention must transpose.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from pathlib import Path
from typing import Sequence

import numpy as np

from .read import get_1d, get_2d

__all__ = ["Aggregated1D", "Aggregated2D", "aggregate_1d", "aggregate_2d"]

Aggregated1D = namedtuple("Aggregated1D", ["q", "intensity", "q_unit", "n_frames"])
Aggregated2D = namedtuple(
    "Aggregated2D", ["q", "chi", "intensity", "q_unit", "chi_unit", "n_frames"]
)

_METHODS = ("sum", "average")


def _reduce_stack(stack: np.ndarray, method: str):
    """NaN-aware reduce over axis 0 of an ``(n_frames, ...)`` stack.

    ``sum`` → ``nansum`` (an all-NaN bin → 0, matching the legacy display);
    ``average`` → ``nanmean`` (an all-NaN bin → NaN gap), with the
    "Mean of empty slice" RuntimeWarning suppressed.  Returns ``None`` for an
    empty stack (0 frames)."""
    stack = np.asarray(stack, dtype=float)
    if stack.ndim < 2 or stack.shape[0] == 0:
        return None
    if method == "sum":
        return np.nansum(stack, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(stack, axis=0)


def _as_stack(intensity: np.ndarray) -> np.ndarray:
    """Promote a single-frame read (frame axis dropped by get_*) to a 1-row stack."""
    intensity = np.asarray(intensity, dtype=float)
    return intensity[np.newaxis, ...] if intensity.ndim < 2 else intensity


def _combine(disk_intensity, disk_labels, extra):
    """Concatenate the on-disk stack with the in-memory tail, deduped by LABEL.

    ``extra`` is ``(labels, intensity_stack)`` for frames not yet on disk; a label
    present in both (freshly flushed yet still resident) is taken from the tail.
    Aggregation is in label space, never positional."""
    disk = _as_stack(disk_intensity)
    if extra is None:
        return disk
    extra_labels, extra_intensity = extra
    extra_stack = _as_stack(extra_intensity)
    if extra_stack.shape[0] == 0:
        return disk
    drop = {int(label) for label in extra_labels}
    keep = [i for i, label in enumerate(disk_labels) if int(label) not in drop]
    disk = disk[keep] if keep else disk[:0]
    if disk.shape[0] == 0:
        return extra_stack
    return np.concatenate([disk, extra_stack], axis=0)


def aggregate_1d(
    scan_file: str | Path,
    *,
    method: str = "average",
    frame=None,
    extra: tuple[Sequence, np.ndarray] | None = None,
    entry: str = "entry",
) -> Aggregated1D:
    """Whole-scan 1D aggregate over the primary ``integrated_1d`` stack.

    ``method`` is ``"sum"`` or ``"average"``; ``frame`` selects a label subset
    (``None`` = all on disk); ``extra=(labels, intensity (n_extra, n_q))`` adds the
    live in-memory tail (deduped vs the on-disk labels).  ``intensity`` is
    ``(n_q,)`` (or ``None`` if there are zero frames)."""
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    data = get_1d(scan_file, frame=frame, entry=entry)
    stack = _combine(data.intensity, list(data.frames), extra)
    agg = _reduce_stack(stack, method)
    return Aggregated1D(
        q=data.q, intensity=agg, q_unit=data.q_unit,
        n_frames=0 if agg is None else int(stack.shape[0]),
    )


def aggregate_2d(
    scan_file: str | Path,
    *,
    method: str = "average",
    frame=None,
    extra: tuple[Sequence, np.ndarray] | None = None,
    entry: str = "entry",
) -> Aggregated2D:
    """Whole-scan 2D (cake) aggregate over the primary ``integrated_2d`` stack.

    ``intensity`` is ``(n_chi, n_q)`` (the file/get_2d convention — a ``(q, chi)``
    display caller must transpose), or ``None`` for zero frames.  ``extra`` stacks
    are ``(labels, (n_extra, n_chi, n_q))``."""
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    data = get_2d(scan_file, frame=frame, entry=entry)
    stack = _combine(data.intensity, list(data.frames), extra)
    agg = _reduce_stack(stack, method)
    return Aggregated2D(
        q=data.q, chi=data.chi, intensity=agg,
        q_unit=data.q_unit, chi_unit=data.chi_unit,
        n_frames=0 if agg is None else int(stack.shape[0]),
    )
