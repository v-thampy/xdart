"""The trend x of a saved Stitch or RSM result: the "Contributing-row mean" of one scan column (headless).

FIT-TOOLS-2026-10 (F4). One value per result: the finite arithmetic mean of the chosen column over the result's
contributing rows, concatenated across its member scans, one vote per contributing row, with the vote count and
min/max. It is neither a mean of scan means nor a monitor-weighted mean.

The rows are the result's SAVED contributor selection; the columns come from the format-aware metadata owner
``run_metadata_table``: through :func:`~xrd_tools.analysis.stitch_metadata.read_stitch_metadata` for a Stitch
result and through :func:`read_rsm_metadata` (the same reader for the RSM provenance shape) for an RSM result.
There is no SPEC/NeXus parser here and no fallback to the current page's source.

A member lacking the column, lacking any finite value of it in its rows, or whose metadata cannot be read leaves
that result's trend x unavailable (never an average of the other members); a column absent from every member,
non-numeric or empty is a per-result failure. Neither raises: a batch keeps every other result's row, and the fit
does not depend on its optional trend x. Rows are keyed by the caller's result identity, never by the value, so
results with equal x stay separate rows.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading

import numpy as np

from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition, MetadataColumn, MetadataTablePlan, MetadataTableResult, _path_revision,
    run_metadata_table,
)
from xrd_tools.analysis.stitch_metadata import StitchMetadataTable, read_stitch_metadata
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactPayload, _stat_revision
from xrd_tools.io.read import resolve_source_master

__all__ = [
    "TREND_LABEL", "TrendMember", "TrendValue", "contributing_row_mean", "read_result_metadata",
    "read_rsm_metadata", "read_trend_values",
]

TREND_LABEL = "Contributing-row mean"
_MAX_METADATA_BYTES = 64 << 20  # the Stitch metadata reader's display limit, kept for RSM


@dataclass(frozen=True, slots=True)
class TrendMember:
    """One member scan's part: its contributing rows, the finite values among them and its state."""

    source_label: str
    rows: int
    values: int
    code: str  # OK | METADATA_UNAVAILABLE | ROWS_UNAVAILABLE | COLUMN_ABSENT | COLUMN_NOT_NUMERIC
    message: str = ""


@dataclass(frozen=True, slots=True)
class TrendValue:
    """One result's trend x. ``value`` is None unless ``status == "available"``."""

    column: str
    occurrence: int
    status: str  # available | unavailable | failed
    code: str
    message: str
    value: float | None
    count: int
    minimum: float | None
    maximum: float | None
    members: tuple[TrendMember, ...]
    label: str = TREND_LABEL

    @property
    def available(self) -> bool:
        return self.status == "available"


def _outcome(column, occurrence, status, code, message, members=(), votes=None) -> TrendValue:
    if votes is None:
        return TrendValue(column, occurrence, status, code, message, None, 0, None, None, tuple(members))
    return TrendValue(column, occurrence, status, code, message, float(np.mean(votes)), int(votes.size),
                      float(np.min(votes)), float(np.max(votes)), tuple(members))


def _find_column(table: MetadataTableResult, name: str, occurrence: int) -> MetadataColumn | None:
    seen = 0
    for column in table.columns:
        if column.name == name:
            if seen == occurrence:
                return column
            seen += 1
    return None


def contributing_row_mean(
    tables: Sequence[StitchMetadataTable],
    column: str,
    *,
    occurrence: int = 0,
) -> TrendValue:
    """The contributing-row mean of ``column`` (its ``occurrence``-th copy) over already-read member tables.

    ``tables`` is one :class:`StitchMetadataTable` per member scan, as :func:`read_stitch_metadata` and
    :func:`read_rsm_metadata` return them: ``selected_labels`` are the member's contributing rows.
    """
    if type(column) is not str or not column:
        raise ValueError("trend column must be a non-empty name")
    if type(occurrence) is not int or occurrence < 0:
        raise ValueError("trend column occurrence must be a non-negative integer")
    members: list[TrendMember] = []
    votes: list[np.ndarray] = []
    for table in tables:
        if type(table) is not StitchMetadataTable:
            raise TypeError("contributing-row means need StitchMetadataTable members")
        rows = len(table.selected_labels)
        result = table.table
        if result.disposition is not AnalysisDisposition.COMPLETED:
            members.append(TrendMember(table.source_label, rows, 0, "METADATA_UNAVAILABLE", result.code))
            continue
        found = _find_column(result, column, occurrence)
        if found is None:
            members.append(TrendMember(table.source_label, rows, 0, "COLUMN_ABSENT"))
            continue
        if found.kind != "numeric" or found.numeric is None:
            members.append(TrendMember(table.source_label, rows, 0, "COLUMN_NOT_NUMERIC"))
            continue
        position = {label: index for index, label in enumerate(result.labels)}
        if any(label not in position for label in table.selected_labels):
            members.append(TrendMember(table.source_label, rows, 0, "ROWS_UNAVAILABLE",
                                       "the metadata no longer holds the contributing rows"))
            continue
        values = np.asarray(found.numeric, dtype=np.float64)[[position[label] for label in table.selected_labels]]
        finite = values[np.isfinite(values)]
        votes.append(finite)
        members.append(TrendMember(table.source_label, rows, int(finite.size), "OK"))

    codes = [member.code for member in members]
    named = f"{column!r}" + (f" (occurrence {occurrence})" if occurrence else "")
    if not members:
        return _outcome(column, occurrence, "unavailable", "NO_MEMBERS", "the result names no member scan")
    unreadable = [m.source_label for m in members if m.code in {"METADATA_UNAVAILABLE", "ROWS_UNAVAILABLE"}]
    if unreadable:
        return _outcome(column, occurrence, "unavailable", "MEMBER_METADATA_UNAVAILABLE",
                        f"metadata unavailable for {', '.join(unreadable)}", members)
    if all(code == "COLUMN_ABSENT" for code in codes):
        return _outcome(column, occurrence, "failed", "COLUMN_MISSING", f"no member records {named}", members)
    if "COLUMN_NOT_NUMERIC" in codes:
        return _outcome(column, occurrence, "failed", "COLUMN_NOT_NUMERIC", f"{named} is not numeric", members)
    lacking = [m.source_label for m in members if m.code == "COLUMN_ABSENT"]
    if lacking:
        return _outcome(column, occurrence, "unavailable", "MEMBER_LACKS_COLUMN",
                        f"{', '.join(lacking)} does not record {named}", members)
    if sum(member.values for member in members) == 0:
        return _outcome(column, occurrence, "failed", "COLUMN_ALL_EMPTY",
                        f"{named} has no finite value in the contributing rows", members)
    empty = [m.source_label for m in members if m.values == 0]
    if empty:
        return _outcome(column, occurrence, "unavailable", "MEMBER_LACKS_VALUES",
                        f"{', '.join(empty)} has no finite {named} in its contributing rows", members)
    return _outcome(column, occurrence, "available", "OK", "", members, np.concatenate(votes))


def _member_source_spec(member: dict, resolve) -> SourceSpec:
    """The member's saved source, reading every metadata column (its selector projection is dropped)."""
    options = dict(member["source_options"])
    path = resolve(member["source_relative_path"])
    if "kind" in options:  # a raw NeXus stack: one exact entry, no reader options
        return SourceSpec(path, SourceKind(options["kind"]), entry=options["entry"])
    options.pop("metadata_column_projection", None)
    options["image_dir"] = str(resolve(options["image_dir"]))
    return SourceSpec(path, SourceKind.SPEC, options=options)


def read_rsm_metadata(
    payload: AnalysisArtifactPayload,
    *,
    source_root: str | Path,
    cancel_token: threading.Event | None = None,
) -> tuple[StitchMetadataTable, ...]:
    """Full experimental columns of a saved RSM result's members, with each member's contributing rows.

    The RSM counterpart of ``read_stitch_metadata``: the saved member sources resolve against the explicit Project
    folder ``source_root`` (an RSM result stores no source base, and paths are never guessed from the output's
    location); every saved source revision must still match before and after the read; a missing or changed
    source gives that member an explicit REFUSED table. No detector pixels are read.
    """
    if type(payload) is not AnalysisArtifactPayload or payload.kind is not AnalysisArtifactKind.RSM:
        raise TypeError("RSM metadata requires a strictly admitted RSM result")
    if source_root is None:
        raise ValueError("an RSM result's metadata needs its Project folder (source_root)")
    inspection = payload.inspection
    if _stat_revision(os.stat(inspection.path)) != inspection.storage_revision:
        raise ValueError("Saved result changed; reopen it before reading metadata")
    members = json.loads(payload.provenance_json).get("members") or ()
    results: list[StitchMetadataTable] = []
    total_bytes = 0
    for member in members:
        if cancel_token is not None and cancel_token.is_set():
            break
        relative = member["source_relative_path"]
        scan = member.get("source_scan") or ""
        title = relative + (f" · scan {scan}" if scan else "")
        labels = tuple(int(row["label"]) for row in member["contributions"])
        shown_scan = str(int(scan[:-2])) if scan.endswith(".1") and scan[:-2].isdecimal() else scan
        shown = tuple(f"{shown_scan}-{label}" if len(members) > 1 and shown_scan else str(label)
                      for label in labels)
        observed: dict[Path, tuple] = {}

        def resolve(stored):
            path = resolve_source_master(stored, scan_file=inspection.path, source_root=source_root)
            if path is None:
                raise ValueError(f"Source unavailable: {stored}")
            return path

        def check_sources(member=member, relative=relative, observed=observed):
            for item in ({"relative_path": relative, "revision": member["primary_revision"]},
                         *member["dependency_files"]):
                if item["revision"] is not None:
                    path = resolve(item["relative_path"])
                    state = _path_revision(path)
                    # as for Stitch: size/mtime admit a relocated tree; full identity must hold during the read
                    if (state is None or state[3:5] != tuple(item["revision"][3:5])
                            or state != observed.setdefault(path, state)):
                        raise ValueError(f"Source changed since processing: {item['relative_path']}")

        try:
            check_sources()
            table = run_metadata_table(MetadataTablePlan(_member_source_spec(member, resolve)),
                                       cancel_token=cancel_token)
            check_sources()
            if table.disposition is AnalysisDisposition.COMPLETED:
                if not set(labels).issubset(table.labels):
                    raise ValueError("Experimental metadata no longer contains the contributing frames")
                total_bytes += table.storage_bytes
                if total_bytes > _MAX_METADATA_BYTES:
                    raise ValueError("Experimental metadata exceeds the 64 MiB display limit")
        except (OSError, ValueError, TypeError, KeyError) as error:
            table = MetadataTableResult(AnalysisDisposition.REFUSED, str(error))
        results.append(StitchMetadataTable(title, labels, shown, table))
        if total_bytes > _MAX_METADATA_BYTES:
            break
    if _stat_revision(os.stat(inspection.path)) != inspection.storage_revision:
        raise ValueError("Saved result changed while reading metadata; reopen it")
    return tuple(results)


def read_result_metadata(
    payload: AnalysisArtifactPayload,
    *,
    source_root: str | Path | None = None,
    cancel_token: threading.Event | None = None,
) -> tuple[StitchMetadataTable, ...]:
    """Member metadata of a saved Stitch (stored source base, or ``source_root``) or RSM (``source_root``) result."""
    if type(payload) is not AnalysisArtifactPayload:
        raise TypeError("result metadata requires a strictly admitted result")
    if payload.kind is AnalysisArtifactKind.RSM:
        return read_rsm_metadata(payload, source_root=source_root, cancel_token=cancel_token)
    return read_stitch_metadata(payload, source_root=source_root, cancel_token=cancel_token)


def read_trend_values(
    results: Iterable[tuple[Hashable, AnalysisArtifactPayload]],
    column: str,
    *,
    occurrence: int = 0,
    source_root: str | Path | None = None,
    cancel_token: threading.Event | None = None,
) -> tuple[tuple[Hashable, TrendValue], ...]:
    """One ``(key, TrendValue)`` row per ``(key, payload)`` result, in order (equal values stay separate rows).

    Reads each result's member metadata in turn (on the caller's worker). A result whose metadata cannot be read
    gets an unavailable row; a cancellation stops before the next result and never yields a row computed from a
    partial read.
    """
    rows: list[tuple[Hashable, TrendValue]] = []
    for key, payload in results:
        if cancel_token is not None and cancel_token.is_set():
            break
        if type(payload) is AnalysisArtifactPayload and payload.kind is AnalysisArtifactKind.RSM \
                and source_root is None:
            rows.append((key, _outcome(column, occurrence, "unavailable", "SOURCE_ROOT_REQUIRED",
                                       "an RSM result's trend needs its Project folder")))
            continue
        try:
            tables = read_result_metadata(payload, source_root=source_root, cancel_token=cancel_token)
        except (OSError, ValueError, KeyError) as error:
            rows.append((key, _outcome(column, occurrence, "unavailable", "RESULT_METADATA_UNAVAILABLE",
                                       str(error))))
            continue
        if cancel_token is not None and cancel_token.is_set():
            break
        rows.append((key, contributing_row_mean(tables, column, occurrence=occurrence)))
    return tuple(rows)
