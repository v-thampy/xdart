# -*- coding: utf-8 -*-
"""Whole-scan aggregation (Sum / Average) over a processed scan's PRIMARY
integrated stack — NaN-aware, normalization-aware, streamed in bounded chunks.

This is the headless half of greenfield Step 7b: it replaces the legacy
per-frame ``data_1d`` / ``data_2d`` display aggregation.  The primary GI mode's
*complete* stack lives on disk at the top-level ``integrated_1d`` /
``integrated_2d`` groups, so a whole-scan Sum/Average reads it **in bounded
position-chunks** and folds a running ``nansum`` + ``nancount`` over them —
``O(chunk)`` peak RAM, not the ``O(whole stack)`` of ``get_2d(frame=None)``
(which would transiently allocate ~1.9 GB for a 651-frame 2D cake).  The
incremental ``nansum/nancount`` is exactly ``nanmean`` over the full stack
(``average``) or ``nansum`` (``sum``), computed without ever materializing it.

**Normalization before reducing** (review 2026-06-15 §2.B): the legacy display
divides each frame by its ``normChannel`` value *before* collapsing, so a naive
disk-stacked ``nansum``/``nanmean`` is silently wrong whenever normalization is
on.  Pass ``norm={label: divisor}`` (the per-frame scalar) and each row is
divided by its divisor before it is folded; a post-divide non-finite value
(zero/missing monitor) is treated as missing rather than poisoning the bin.

A LIVE, in-progress scan has frames that are reduced but not yet flushed to disk;
the caller passes those as ``extra`` (the in-memory unflushed tail), and they are
folded in **deduped by frame label** (a freshly-flushed frame can be both on disk
and still resident in memory — the on-disk copy of a tail label is skipped).
With persist-before-evict, on-disk-prefix ⊕ in-memory-tail covers ALL frames
without holding them in RAM and without depending on the bounded display store —
which is exactly why this is the correct fix-forward for the bounded-store
aggregation-truncation bug.

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

from collections import namedtuple
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from .read import _decode, _entry, _frame_index, _resolve_positions
from .schema import SCHEMA

__all__ = ["Aggregated1D", "Aggregated2D", "aggregate_1d", "aggregate_2d"]

Aggregated1D = namedtuple("Aggregated1D", ["q", "intensity", "q_unit", "n_frames"])
Aggregated2D = namedtuple(
    "Aggregated2D", ["q", "chi", "intensity", "q_unit", "chi_unit", "n_frames"]
)

_METHODS = ("sum", "average")
_DEFAULT_CHUNK = 64


def _divisors_for(labels: Sequence, norm: Mapping | None) -> np.ndarray | None:
    """Per-row divisor array for ``labels`` from a ``{label: scalar}`` map.

    A label missing from ``norm`` divides by 1.0 (un-normalized); ``None`` norm
    means no normalization at all (the caller returns ``None`` so the fold skips
    the divide entirely)."""
    if norm is None:
        return None
    return np.array([float(norm.get(int(lbl), 1.0)) for lbl in labels], dtype=float)


def _fold(acc_sum, acc_count, rows, divisors):
    """Fold one chunk's rows into the running ``nansum`` + ``nancount``.

    ``rows`` is ``(k, *bins)``; ``divisors`` is ``(k,)`` per-row scalars (or
    ``None``).  Each row is divided by its scalar first; a non-finite result
    (NaN gap, or zero/missing-monitor inf) counts as *missing* — it is not added
    and does not increment the per-bin count, so ``average`` divides by the true
    number of finite contributors."""
    rows = np.asarray(rows, dtype=float)
    if rows.ndim < 2:                       # a single-row read dropped the axis
        rows = rows[np.newaxis, ...]
    if divisors is not None:
        d = np.asarray(divisors, dtype=float).reshape(
            (rows.shape[0],) + (1,) * (rows.ndim - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            rows = rows / d
    mask = np.isfinite(rows)
    if acc_sum is None:
        acc_sum = np.zeros(rows.shape[1:], dtype=float)
        acc_count = np.zeros(rows.shape[1:], dtype=np.int64)
    acc_sum += np.where(mask, rows, 0.0).sum(axis=0)
    acc_count += mask.sum(axis=0)
    return acc_sum, acc_count


def _finalize(acc_sum, acc_count, method):
    if acc_sum is None:
        return None
    if method == "sum":
        return acc_sum                       # nansum: an all-missing bin -> 0
    with np.errstate(invalid="ignore", divide="ignore"):
        # nanmean: an all-missing bin -> NaN gap (never 0/0)
        return np.where(acc_count > 0, acc_sum / np.maximum(acc_count, 1), np.nan)


def _aggregate_stack(scan_file, group_name, axis_names, *, method, frame, extra,
                     norm, entry, chunk_size):
    """Shared 1D/2D engine: open once, fold the on-disk stack in label-chunks
    (deduped against the tail), then fold the in-memory tail.

    Returns ``(axes, units, intensity, n_frames)`` where ``axes``/``units`` are
    tuples aligned to ``axis_names``; ``intensity`` is the reduced array (or
    ``None`` for zero frames)."""
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")

    extra_labels = []
    if extra is not None:
        extra_labels = [int(lbl) for lbl in extra[0]]
    drop = set(extra_labels)

    acc_sum = acc_count = None
    n_frames = 0
    axes: list = [None] * len(axis_names)
    units: list = [None] * len(axis_names)

    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        if group_name not in e:
            raise KeyError(f"{scan_file} has no {group_name} group")
        g = e[group_name]
        for i, name in enumerate(axis_names):
            axes[i] = np.asarray(g[name][()])
            units[i] = (_decode(g[name].attrs.get("units"))
                        if "units" in g[name].attrs else None)
        frame_index = _frame_index(e, prefer=group_name)
        positions, frames, _ = _resolve_positions(frame_index, frame)
        # Drop the on-disk rows the tail supersedes (dedup by LABEL), so a freshly
        # flushed-yet-resident frame is counted once, from the tail.
        keep = np.array(
            [j for j, lbl in enumerate(frames) if int(lbl) not in drop], dtype=int)
        positions = positions[keep]
        frames = np.asarray(frames)[keep]
        dset = g["intensity"]
        for start in range(0, positions.size, chunk_size):
            sel = positions[start:start + chunk_size]
            labels = frames[start:start + chunk_size]
            order = np.argsort(sel, kind="stable")   # h5py needs increasing idx
            rows = np.asarray(dset[sel[order]])
            divisors = _divisors_for(labels[order], norm)
            acc_sum, acc_count = _fold(acc_sum, acc_count, rows, divisors)
            n_frames += int(sel.size)

    if extra is not None and len(extra_labels):
        extra_rows = np.asarray(extra[1], dtype=float)
        if extra_rows.ndim == len(axis_names):       # single tail row, axis dropped
            extra_rows = extra_rows[np.newaxis, ...]
        if extra_rows.shape[0]:
            divisors = _divisors_for(extra_labels, norm)
            acc_sum, acc_count = _fold(acc_sum, acc_count, extra_rows, divisors)
            n_frames += int(extra_rows.shape[0])

    intensity = _finalize(acc_sum, acc_count, method)
    return tuple(axes), tuple(units), intensity, (0 if intensity is None else n_frames)


def aggregate_1d(
    scan_file: str | Path,
    *,
    method: str = "average",
    frame=None,
    extra: tuple[Sequence, np.ndarray] | None = None,
    norm: Mapping | None = None,
    entry: str = "entry",
    chunk_size: int = _DEFAULT_CHUNK,
) -> Aggregated1D:
    """Whole-scan 1D aggregate over the primary ``integrated_1d`` stack.

    ``method`` is ``"sum"`` or ``"average"``; ``frame`` selects a label subset
    (``None`` = all on disk); ``extra=(labels, intensity (n_extra, n_q))`` adds
    the live in-memory tail (deduped vs the on-disk labels); ``norm={label:
    divisor}`` normalizes each frame before reducing.  ``intensity`` is
    ``(n_q,)`` (or ``None`` if there are zero frames).  The on-disk stack is read
    in ``chunk_size``-frame slabs (bounded peak RAM)."""
    (q_name,) = SCHEMA.groups["integrated_1d"].axes
    (q,), (q_unit,), intensity, n_frames = _aggregate_stack(
        scan_file, "integrated_1d", (q_name,), method=method, frame=frame,
        extra=extra, norm=norm, entry=entry, chunk_size=chunk_size)
    return Aggregated1D(q=q, intensity=intensity, q_unit=q_unit, n_frames=n_frames)


def aggregate_2d(
    scan_file: str | Path,
    *,
    method: str = "average",
    frame=None,
    extra: tuple[Sequence, np.ndarray] | None = None,
    norm: Mapping | None = None,
    entry: str = "entry",
    chunk_size: int = _DEFAULT_CHUNK,
) -> Aggregated2D:
    """Whole-scan 2D (cake) aggregate over the primary ``integrated_2d`` stack.

    ``intensity`` is ``(n_chi, n_q)`` (the file/get_2d convention — a ``(q, chi)``
    display caller must transpose), or ``None`` for zero frames.  ``extra`` stacks
    are ``(labels, (n_extra, n_chi, n_q))``.  See :func:`aggregate_1d` for the
    shared ``method``/``frame``/``norm``/``chunk_size`` semantics."""
    q_name, chi_name = SCHEMA.groups["integrated_2d"].axes
    (q, chi), (q_unit, chi_unit), intensity, n_frames = _aggregate_stack(
        scan_file, "integrated_2d", (q_name, chi_name), method=method, frame=frame,
        extra=extra, norm=norm, entry=entry, chunk_size=chunk_size)
    return Aggregated2D(
        q=q, chi=chi, intensity=intensity,
        q_unit=q_unit, chi_unit=chi_unit, n_frames=n_frames)
