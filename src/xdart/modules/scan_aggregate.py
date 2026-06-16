# -*- coding: utf-8 -*-
"""Whole-scan Sum/Average over a live scan = on-disk primary stack ⊕ in-memory
tail (greenfield Step 7b, the xdart half).

The bounded display store (``PublicationStore``, ``max_heavy_items=64``) cannot
hold every frame of a long scan, so a whole-scan Sum/Average built by collapsing
the store's resident traces silently drops the evicted frames (the Round-12
truncation bug).  This module fixes it at the source: the primary GI mode's
*complete* stack is on disk, and persist-before-evict guarantees the only frames
NOT on disk are the unflushed in-memory tail — so

    whole-scan aggregate = io.aggregate(disk, primary stack)  ⊕  in-memory tail

covers ALL frames in ``O(chunk)`` RAM, independent of the display store.  The
heavy read runs off the GUI thread (see ``AggregationWorker``); this module is
the Qt-free compute core it calls.

**Primary-mode-scoped** (ADR-0003): only the primary on-disk stack is read.  A
non-primary GI sub-mode's on-disk stack is partial, so a whole-scan aggregate in
that mode must be REFUSED, never served from the partial stack (that *is* the
Round-12 trap).  The caller gates on :func:`mode_aggregation_allowed` with the
currently-displayed mode; this module always reads the primary (top-level) stack
and folds the frames' *active* result as the tail.

**Normalization** is applied per-frame before reducing (review_2026-06-15 §2.B):
the caller passes ``norm={label: divisor}`` (the normChannel value per frame) and
:mod:`xrd_tools.io.aggregate` divides each row before folding.

The 2D tail is transposed on the way in: a live frame stores ``int_2d.intensity``
as ``(radial, azimuthal) = (n_q, n_chi)`` whereas the on-disk ``get_2d`` stack is
``(n_chi, n_q)`` — :func:`whole_scan_aggregate_2d` returns the disk/``get_2d``
convention ``(n_chi, n_q)`` (a ``(q, chi)`` display caller transposes).
"""

from __future__ import annotations

import contextlib
import os
from typing import Mapping

import numpy as np

from xrd_tools.core.frame_view import DEFAULT_MODE_KEY
from xrd_tools.io import aggregate as _agg

__all__ = [
    "whole_scan_aggregate_1d",
    "whole_scan_aggregate_2d",
    "mode_aggregation_allowed",
]


def _norm_mode(mode) -> str:
    """Normalize a mode key to its canonical on-disk spelling for comparison.

    ``None`` / empty / the legacy GUI sentinels all collapse to the primary
    default slot."""
    if mode is None:
        return DEFAULT_MODE_KEY
    text = str(mode).strip()
    return text if text else DEFAULT_MODE_KEY


def mode_aggregation_allowed(displayed_mode, primary_mode) -> bool:
    """True iff a whole-scan aggregate may be served from the primary on-disk
    stack for the currently displayed mode.

    A non-primary GI sub-mode's stacked on-disk results are partial/lazy, so an
    aggregate over them would silently truncate — the caller must defer (disable/
    annotate) instead.  Only the displayed-IS-primary case is servable here."""
    return _norm_mode(displayed_mode) == _norm_mode(primary_mode)


def _unflushed_tail(scan):
    """Snapshot the in-memory frames that are NOT yet persisted to disk.

    Returns a list of ``(label, LiveFrame)`` taken under the series cache lock
    (the snapshot is cheap — refs only — and released immediately so the heavy
    array reads below never hold the lock the wrangler thread needs)."""
    series = getattr(scan, "frames", None)
    in_memory = getattr(series, "_in_memory", None)
    if not in_memory:
        return []
    persisted = getattr(series, "_persisted", set())
    lock = getattr(series, "_cache_lock", None)
    cm = lock if lock is not None else contextlib.nullcontext()
    with cm:
        return [(int(idx), in_memory[idx]) for idx in in_memory
                if idx not in persisted]


def _stack_uniform(rows, labels):
    """Stack same-shaped rows into one array, dropping any whose shape differs
    from the first kept row (a grid mismatch would broadcast-error the fold; the
    frozen common grid makes this defensive, not expected)."""
    keep_rows, keep_labels, ref = [], [], None
    for label, row in zip(labels, rows):
        row = np.asarray(row, dtype=float)
        if ref is None:
            ref = row.shape
        if row.shape != ref:
            continue
        keep_rows.append(row)
        keep_labels.append(label)
    if not keep_rows:
        return None
    return keep_labels, np.stack(keep_rows, axis=0)


def _tail_1d(scan):
    labels, rows = [], []
    for label, fr in _unflushed_tail(scan):
        result = getattr(fr, "int_1d", None)
        intensity = getattr(result, "intensity", None)
        if intensity is None:
            continue
        labels.append(label)
        rows.append(intensity)
    return _stack_uniform(rows, labels)


def _tail_2d(scan):
    labels, rows = [], []
    for label, fr in _unflushed_tail(scan):
        result = getattr(fr, "int_2d", None)
        intensity = getattr(result, "intensity", None)
        if intensity is None:
            continue
        arr = np.asarray(intensity, dtype=float)
        if arr.ndim != 2:
            continue
        labels.append(label)
        rows.append(arr.T)               # (n_q, n_chi) live -> (n_chi, n_q) disk
    return _stack_uniform(rows, labels)


def _file_lock(scan):
    lock = getattr(scan, "file_lock", None)
    if lock is None:
        lock = getattr(getattr(scan, "frames", None), "file_lock", None)
    return lock if lock is not None else contextlib.nullcontext()


def whole_scan_aggregate_1d(scan, *, method="average", norm: Mapping | None = None,
                            chunk_size: int = _agg._DEFAULT_CHUNK):
    """Whole-scan 1D aggregate (primary on-disk stack ⊕ unflushed tail), or
    ``None`` when nothing is on disk yet (defer to the resident-store path).

    Reads under the scan's ``file_lock`` so a concurrent live write does not tear
    the read.  ``method`` is ``"sum"``/``"average"``; ``norm`` normalizes each
    frame before reducing (see module docstring).  Caller must have confirmed the
    displayed mode is primary (:func:`mode_aggregation_allowed`)."""
    data_file = getattr(scan, "data_file", None)
    if not data_file or not os.path.exists(data_file):
        return None
    tail = _tail_1d(scan)
    with _file_lock(scan):
        try:
            return _agg.aggregate_1d(data_file, method=method, extra=tail,
                                     norm=norm, chunk_size=chunk_size)
        except KeyError:
            return None                  # no integrated_1d on disk yet -> defer


def whole_scan_aggregate_2d(scan, *, method="average", norm: Mapping | None = None,
                            chunk_size: int = _agg._DEFAULT_CHUNK):
    """Whole-scan 2D (cake) aggregate in the ``(n_chi, n_q)`` disk convention, or
    ``None`` when nothing is on disk yet.  See :func:`whole_scan_aggregate_1d`."""
    data_file = getattr(scan, "data_file", None)
    if not data_file or not os.path.exists(data_file):
        return None
    tail = _tail_2d(scan)
    with _file_lock(scan):
        try:
            return _agg.aggregate_2d(data_file, method=method, extra=tail,
                                     norm=norm, chunk_size=chunk_size)
        except KeyError:
            return None                  # no integrated_2d on disk yet -> defer
