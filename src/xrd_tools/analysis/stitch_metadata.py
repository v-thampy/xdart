"""Experimental metadata for the exact contributors of a saved Stitch result.

The artifact already records source identities and relative locators. Reuse the
ordinary metadata-table reader and raw-source resolver; never treat provenance
or a changed source as a saved copy of the experimental table.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition, MetadataTablePlan, MetadataTableResult, _path_revision,
    run_metadata_table,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.analysis_artifact import AnalysisArtifactPayload, _stat_revision
from xrd_tools.io.read import resolve_source_master


@dataclass(frozen=True, slots=True)
class StitchMetadataTable:
    source_label: str
    selected_labels: tuple[int, ...]
    row_labels: tuple[str, ...]
    table: MetadataTableResult


def read_stitch_metadata(payload, *, source_root=None, cancel_token=None):
    """Read full experimental columns, retaining exact selected rows per source.

    Existing files have no persisted full experimental table. Missing or changed
    originals therefore produce an explicit unavailable result for that member.
    Source reads and their handles belong to ``run_metadata_table`` on the caller's
    worker. No detector pixels are read and no publication authority is created.
    """
    if type(payload) is not AnalysisArtifactPayload:
        raise TypeError("Stitch metadata requires a strictly admitted artifact")
    inspection = payload.inspection
    if _stat_revision(os.stat(inspection.path)) != inspection.storage_revision:
        raise ValueError("Saved result changed; reopen it before reading metadata")
    if inspection.frame_manifest_json is None:
        return (StitchMetadataTable("Source", (), (), MetadataTableResult(
            AnalysisDisposition.REFUSED, "Experimental metadata is unavailable for this older result")),)
    base = json.loads(inspection.frame_manifest_json)["source_base"]
    manifest = json.loads(payload.provenance_json)["source"]["input_manifest"]
    members = manifest.get("members", (manifest,))
    results = []
    total_bytes = 0
    for member in members:
        if cancel_token is not None and cancel_token.is_set():
            break
        source = member["source"]
        scan = source.get("scan") or source.get("entry") or Path(source["relative_path"]).stem
        title = source["relative_path"] + (f" · scan {scan}" if source.get("scan") else f" · {scan}")
        labels = tuple(row["label"] for row in member["contributions"])
        shown_scan = str(int(scan[:-2])) if scan.endswith(".1") and scan[:-2].isdecimal() else scan
        shown = tuple(f"{shown_scan}-{label}" if len(members) > 1 else str(label) for label in labels)
        observed = {}

        def resolve(relative):
            path = resolve_source_master(relative, scan_file=inspection.path,
                source_base=base, source_root=source_root)
            if path is None:
                raise ValueError(f"Source unavailable: {relative}")
            return path

        def check_sources():
            # Qualify metadata and frame association against saved revisions,
            # including linked NeXus dependencies, before and after the read.
            for item in ({"relative_path": source["relative_path"],
                          "revision": source["primary_revision"]}, *member["files"]):
                if item["revision"] is not None:
                    path = resolve(item["relative_path"])
                    state = _path_revision(path)
                    # Like Full Raw, size/mtime admit an explicitly relocated
                    # tree. Full identity must also remain stable during this read.
                    if (state is None or state[3:5] != tuple(item["revision"][3:5])
                            or state != observed.setdefault(path, state)):
                        raise ValueError(f"Source changed since stitching: {item['relative_path']}")

        try:
            check_sources()
            options = dict(source["options"])
            for key in ("image_dir", "selected_file", "meta_dir"):
                if options.get(key):
                    options[key] = str(resolve(options[key]))
            if "files" in options:
                options["files"] = tuple(str(resolve(item)) for item in options["files"])
            if "admitted_motor_values" in options:
                options["admitted_motor_values"] = tuple((str(resolve(path)), name, value)
                    for path, name, value in options["admitted_motor_values"])
            spec = SourceSpec(resolve(source["relative_path"]), SourceKind(source["kind"]),
                              entry=source.get("entry"), options=options)
            table = run_metadata_table(MetadataTablePlan(spec), cancel_token=cancel_token)
            check_sources()
            if table.disposition is AnalysisDisposition.COMPLETED:
                if not set(labels).issubset(table.labels):
                    raise ValueError("Experimental metadata no longer contains the contributing frames")
                total_bytes += table.storage_bytes
                if total_bytes > 64 << 20:
                    raise ValueError("Experimental metadata exceeds the 64 MiB display limit")
        except (OSError, ValueError, TypeError, KeyError) as error:
            table = MetadataTableResult(AnalysisDisposition.REFUSED, str(error))
        results.append(StitchMetadataTable(title, labels, shown, table))
        if total_bytes > 64 << 20:
            break
    if _stat_revision(os.stat(inspection.path)) != inspection.storage_revision:
        raise ValueError("Saved result changed while reading metadata; reopen it")
    return tuple(results)
