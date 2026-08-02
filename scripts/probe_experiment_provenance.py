#!/usr/bin/env python3
"""Reproduce the bounded historical NeXus wavelength-location census."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


COMMAND = (
    'PYTHONPATH=$PWD/src pixi run --locked python '
    'scripts/probe_experiment_provenance.py --data-root "$XDART_PROBE_ROOT" '
    '--manifest tests/core/fixtures/experiment_reload_probe_manifest.json '
    '--output tests/core/fixtures/experiment_reload_probe_results.json'
)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _classification(handle: Any) -> str:
    entry = handle.get("entry")
    if entry is None:
        return "absent"
    headless = "instrument/monochromator/wavelength" in entry
    legacy = "instrument/source/wavelength_A" in entry
    if headless and legacy:
        return "both"
    if headless:
        return "headless_only"
    if legacy:
        return "legacy_only"
    return "absent"


def build_manifest(data_root: Path, *, limit: int) -> dict[str, Any]:
    candidates: list[tuple[int, str]] = []
    for child in ("test_data", "tmp"):
        for path in (data_root / child).rglob("*.nxs"):
            relative = path.relative_to(data_root).as_posix()
            if "process" in relative.lower() and path.is_file():
                candidates.append((path.stat().st_mtime_ns, relative))
    candidates.sort()
    selected = candidates[:limit]
    if len(selected) != limit:
        raise ValueError(f"requested {limit} records, found {len(selected)}")
    return {
        "schema_version": 1,
        "selection": {
            "rule": "oldest mtime_ns then relative path; *.nxs path contains process",
            "roots": ["test_data", "tmp"],
            "count": limit,
        },
        "records": [
            {"record_id": f"record-{index:04d}", "relative_path": relative}
            for index, (_, relative) in enumerate(selected, 1)
        ],
    }


def run_probe(data_root: Path, manifest_path: Path) -> dict[str, Any]:
    import h5py

    manifest = json.loads(manifest_path.read_text())
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for expected in manifest["records"]:
        relative = expected["relative_path"]
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            failures.append(f"{expected['record_id']}: unsafe relative path")
            continue
        path = data_root / candidate
        if not path.is_file():
            failures.append(f"{expected['record_id']}: manifest input is missing")
            continue
        digest = _digest(path)
        readable, open_failure = True, False
        try:
            with h5py.File(path, "r") as handle:
                classification = _classification(handle)
        except OSError:
            readable, open_failure, classification = False, True, "open_failure"
        except Exception:
            classification = "decode_failure"
        rows.append({
            "record_id": expected["record_id"],
            "relative_path": relative,
            "readable": readable,
            "open_failure": open_failure,
            "classification": classification,
            "content_sha256": digest,
        })
    aggregate = Counter(row["classification"] for row in rows)
    return {
        "schema_version": 1,
        "command": COMMAND,
        "exit_status": 1 if failures else 0,
        "failures": failures,
        "aggregate": dict(sorted(aggregate.items())),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()
    if args.build_manifest:
        report = build_manifest(args.data_root, limit=args.limit)
    else:
        report = run_probe(args.data_root, args.manifest)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0 if args.build_manifest else int(report["exit_status"])


if __name__ == "__main__":
    raise SystemExit(main())
