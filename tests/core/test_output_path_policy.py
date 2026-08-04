# -*- coding: utf-8 -*-
"""P4/OUT-1 — frozen acceptance oracle for the shared ``.nexus`` output-path policy.

This file is the headless (PP-C0) half of the finite oracle frozen before
implementation.  It pins the sole Qt-free path owner
(:mod:`xrd_tools.io.output_path`), its public ``xrd_tools.io`` surface, and the
headless consumers that must generate ``.nexus`` while still reading ``.nxs``
forever.  The canonical GUI consumers are pinned in
``tests/xdart/test_output_path_policy_gui.py`` (PP-C1).

Row numbers below are the handoff §5 oracle rows.  Parent polarity is recorded
in the handback report; the contract is *not* that every row starts red — rows
8-14 are retained-green invariants that must survive the change.

The owner selects paths only.  It never inspects content, schema or lineage, and
it owns no writer, transaction or lease (handoff §2 rule 9).
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


# ---------------------------------------------------------------------------
# helpers — real files, no fakes on the seam under test (HARD RULE 2)
# ---------------------------------------------------------------------------

def _write_processed(path: Path, *, q_len: int = 9, n_frames: int = 2) -> Path:
    """A minimal real processed record: ``entry/integrated_1d`` with frames."""
    q = np.linspace(1.0, 5.0, q_len, dtype=np.float32)
    frames = np.arange(n_frames, dtype=np.int64)
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["ssrl_schema_version"] = 2
        g = entry.create_group("integrated_1d")
        g.create_dataset("frame_index", data=frames)
        q_ds = g.create_dataset("q", data=q)
        q_ds.attrs["units"] = "q_A^-1"
        g.create_dataset(
            "intensity",
            data=np.ones((n_frames, q_len), dtype=np.float32),
        )
        frame_group = entry.create_group("frames")
        for frame in frames:
            fg = frame_group.create_group(f"frame_{frame:04d}")
            thumb = fg.create_dataset(
                "thumbnail", data=np.arange(12, dtype=np.uint8).reshape(3, 4))
            thumb.attrs["vmin"] = 10.0
            thumb.attrs["vmax"] = 265.0
            thumb.attrs["dtype"] = "uint8"
    return path


def _write_raw_master(path: Path, *, n_frames: int = 3) -> Path:
    """A real raw detector container (not a processed record)."""
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "entry/instrument/detector/data",
            data=np.arange(n_frames * 4 * 5, dtype=float).reshape(n_frames, 4, 5),
        )
    return path


# ---------------------------------------------------------------------------
# Row 1 — the shared path-policy module and public API exist
# ---------------------------------------------------------------------------

def test_row1_owner_module_and_public_api_exist():
    from xrd_tools.io import output_path as owner

    assert owner.NEW_OUTPUT_SUFFIX == ".nexus"
    assert owner.LEGACY_OUTPUT_SUFFIX == ".nxs"
    assert owner.READABLE_OUTPUT_SUFFIXES == (".nexus", ".nxs")
    for name in ("default_output_path", "resolve_output_target",
                 "is_readable_output_path"):
        assert callable(getattr(owner, name)), name


def test_row1_surface_is_exported_through_xrd_tools_io():
    import xrd_tools.io as io

    assert io.NEW_OUTPUT_SUFFIX == ".nexus"
    assert io.LEGACY_OUTPUT_SUFFIX == ".nxs"
    assert io.READABLE_OUTPUT_SUFFIXES == (".nexus", ".nxs")
    assert callable(io.default_output_path)
    assert callable(io.resolve_output_target)
    assert callable(io.is_readable_output_path)


def test_row1_owner_is_the_only_path_policy_owner():
    """No second module may export the generated-output suffix decision."""
    owners = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^NEW_OUTPUT_SUFFIX\s*=", text, re.MULTILINE):
            owners.append(path.relative_to(SRC).as_posix())
    assert owners == ["xrd_tools/io/output_path.py"], owners


# ---------------------------------------------------------------------------
# Row 2 — default / Overwrite / absent-target Append select ``.nexus``
# ---------------------------------------------------------------------------

def test_row2_default_output_path_is_nexus(tmp_path):
    from xrd_tools.io import default_output_path

    assert default_output_path(tmp_path, "scan_042") == tmp_path / "scan_042.nexus"


def test_row2_overwrite_selects_nexus_on_an_empty_directory(tmp_path):
    from xrd_tools.io import resolve_output_target

    target = resolve_output_target(tmp_path, "scan_042", mode="Overwrite")
    assert target == tmp_path / "scan_042.nexus"


def test_row2_append_with_no_existing_sibling_creates_nexus(tmp_path):
    from xrd_tools.io import resolve_output_target

    target = resolve_output_target(tmp_path, "scan_042", mode="Append")
    assert target == tmp_path / "scan_042.nexus"
    assert not target.exists()          # the owner selects, it never creates


def test_row2_overwrite_never_discovers_a_sibling_legacy_file(tmp_path):
    """§2 rule 2: Overwrite must not implicitly find or replace a ``.nxs``."""
    from xrd_tools.io import resolve_output_target

    legacy = _write_processed(tmp_path / "scan_042.nxs")
    before = legacy.read_bytes()

    target = resolve_output_target(tmp_path, "scan_042", mode="Overwrite")

    assert target == tmp_path / "scan_042.nexus"
    assert legacy.read_bytes() == before


def test_row2_append_prefers_an_existing_nexus(tmp_path):
    from xrd_tools.io import resolve_output_target

    _write_processed(tmp_path / "scan_042.nexus")
    target = resolve_output_target(tmp_path, "scan_042", mode="Append")
    assert target == tmp_path / "scan_042.nexus"


# ---------------------------------------------------------------------------
# Row 3 — headless series and watcher output names use ``.nexus``
# ---------------------------------------------------------------------------

def test_row3_process_series_names_outputs_nexus(monkeypatch, tmp_path):
    """``process_series`` derives ``<stem>_processed.nexus`` per input scan.

    ``process_scan`` is substituted at the module boundary so this row pins the
    *name derivation* only — the reduction itself is covered by
    ``tests/core/test_batch.py``.
    """
    from xrd_tools.integrate import batch

    seen: list[Path] = []

    def _fake_process_scan(scan_path, ai, out_h5, **kwargs):
        seen.append(Path(out_h5))
        return Path(out_h5)

    monkeypatch.setattr(batch, "process_scan", _fake_process_scan)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in ("scan_1.h5", "scan_2.h5"):
        _write_raw_master(raw_dir / name)

    batch.process_series(
        sorted(raw_dir.iterdir()), object(), tmp_path / "out")

    assert [p.name for p in seen] == [
        "scan_1_processed.nexus", "scan_2_processed.nexus"]


def test_row3_directory_watcher_names_outputs_nexus(monkeypatch, tmp_path):
    from xrd_tools.integrate import batch

    seen: list[Path] = []

    def _fake_process_scan(path, ai, out_h5, **kwargs):
        seen.append(Path(out_h5))
        return Path(out_h5)

    monkeypatch.setattr(batch, "process_scan", _fake_process_scan)

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    source = _write_raw_master(watch / "live_007.h5")

    watcher = batch.DirectoryWatcher(watch, object(), out)
    watcher._process_new_file(source)

    assert [p.name for p in seen] == ["live_007_processed.nexus"]


# ---------------------------------------------------------------------------
# Row 4 — ``LiveScan(name)`` defaults to ``name.nexus``
# ---------------------------------------------------------------------------

def test_row4_livescan_default_data_file_is_nexus():
    from xdart.modules.ewald.scan import LiveScan

    scan = LiveScan("scan_042")
    assert scan.data_file == "scan_042.nexus"


def test_row4_livescan_preserves_an_explicit_data_file(tmp_path):
    """§2 rule 5: an explicit legacy target is preserved exactly."""
    from xdart.modules.ewald.scan import LiveScan

    explicit = os.fspath(tmp_path / "chosen.nxs")
    scan = LiveScan("scan_042", data_file=explicit)
    assert scan.data_file == explicit


# ---------------------------------------------------------------------------
# Row 5 — time-resolved discovery naturally orders a mixed set
# ---------------------------------------------------------------------------

def test_row5_time_resolved_discovery_orders_mixed_suffixes(tmp_path):
    from xrd_tools.analysis.time_resolved import discover_processed_scans

    _write_processed(tmp_path / "scan_2.nexus")
    _write_processed(tmp_path / "scan_10.nxs")
    _write_processed(tmp_path / "scan_1.nxs")

    found = discover_processed_scans(tmp_path)

    assert [p.name for p in found] == [
        "scan_1.nxs", "scan_2.nexus", "scan_10.nxs"]


def test_row5_explicit_pattern_still_narrows_discovery(tmp_path):
    """An explicitly passed pattern keeps its exact meaning."""
    from xrd_tools.analysis.time_resolved import discover_processed_scans

    _write_processed(tmp_path / "scan_2.nexus")
    _write_processed(tmp_path / "scan_10.nxs")

    found = discover_processed_scans(tmp_path, pattern="*.nxs")
    assert [p.name for p in found] == ["scan_10.nxs"]


# ---------------------------------------------------------------------------
# Row 6 (headless half) — image classification and counting open ``.nexus``
# ---------------------------------------------------------------------------

def test_row6_classify_image_source_recognizes_nexus(tmp_path):
    from xrd_tools.io import ImageSourceKind, classify_image_source

    processed = _write_processed(tmp_path / "scan.nexus")
    info = classify_image_source(processed)
    assert info.kind in (ImageSourceKind.PROCESSED_XDART,
                         ImageSourceKind.THUMBNAIL_ONLY)
    assert info.n_frames == 2


def test_row6_count_frames_reads_a_nexus_container(tmp_path):
    from xrd_tools.io import count_frames

    raw = _write_raw_master(tmp_path / "container.nexus", n_frames=3)
    assert count_frames(raw) == 3


def test_row6_read_image_rejects_a_processed_nexus_like_a_processed_nxs(tmp_path):
    """A processed record is not a raw detector stack, whatever its suffix."""
    from xrd_tools.io.image import read_image
    from xrd_tools.io.processed_scan_id import ProcessedXdartInputError

    for name in ("scan.nxs", "scan.nexus"):
        processed = _write_processed(tmp_path / name)
        with pytest.raises(ProcessedXdartInputError):
            read_image(processed)


# ---------------------------------------------------------------------------
# Row 8 — explicit ``.nxs`` remains readable and writable
# ---------------------------------------------------------------------------

def test_row8_explicit_legacy_target_is_preserved_by_the_resolver(tmp_path):
    from xrd_tools.io import resolve_output_target

    explicit = tmp_path / "operator_choice.nxs"
    for mode in ("Append", "Overwrite"):
        assert resolve_output_target(
            tmp_path, "scan_042", mode=mode,
            explicit_target=explicit) == explicit


def test_row8_explicit_new_target_is_preserved_by_the_resolver(tmp_path):
    from xrd_tools.io import resolve_output_target

    explicit = tmp_path / "operator_choice.nexus"
    assert resolve_output_target(
        tmp_path, "scan_042", mode="Append",
        explicit_target=explicit) == explicit


def test_row8_legacy_nxs_stays_readable(tmp_path):
    from xrd_tools.io import get_frames, is_readable_output_path

    legacy = _write_processed(tmp_path / "legacy.nxs")
    assert is_readable_output_path(legacy)
    assert list(get_frames(legacy)) == [0, 1]


def test_row8_suffix_recognition_is_case_insensitive(tmp_path):
    from xrd_tools.io import is_readable_output_path

    assert is_readable_output_path(tmp_path / "a.NXS")
    assert is_readable_output_path(tmp_path / "b.Nexus")
    assert not is_readable_output_path(tmp_path / "c.h5")
    assert not is_readable_output_path(tmp_path / "d.tif")


# ---------------------------------------------------------------------------
# Row 9 — a real existing ``.nxs`` Append is reused; refusal preserves bytes
# ---------------------------------------------------------------------------

def test_row9_append_reuses_a_sole_existing_legacy_file(tmp_path):
    from xrd_tools.io import resolve_output_target

    legacy = _write_processed(tmp_path / "scan_042.nxs")
    target = resolve_output_target(tmp_path, "scan_042", mode="Append")
    assert target == legacy


def test_row9_refused_append_leaves_the_legacy_bytes_intact(tmp_path):
    """The collision guard runs on the RESOLVED target and preserves bytes."""
    from xrd_tools.io import resolve_output_target
    from xrd_tools.io.output_safety import (
        OutputCollisionError, check_output_not_source,
    )

    legacy = _write_processed(tmp_path / "scan_042.nxs")
    before = legacy.read_bytes()

    target = resolve_output_target(tmp_path, "scan_042", mode="Append")
    with pytest.raises(OutputCollisionError):
        check_output_not_source(target, input_files=[legacy])

    assert legacy.read_bytes() == before


# ---------------------------------------------------------------------------
# Row 10 — both siblings exist: ``.nexus`` wins unless ``.nxs`` is explicit
# ---------------------------------------------------------------------------

def test_row10_nexus_wins_when_both_siblings_exist(tmp_path):
    from xrd_tools.io import resolve_output_target

    _write_processed(tmp_path / "scan_042.nxs")
    _write_processed(tmp_path / "scan_042.nexus")

    assert resolve_output_target(
        tmp_path, "scan_042", mode="Append") == tmp_path / "scan_042.nexus"


def test_row10_explicit_legacy_wins_over_an_existing_nexus(tmp_path):
    from xrd_tools.io import resolve_output_target

    legacy = _write_processed(tmp_path / "scan_042.nxs")
    _write_processed(tmp_path / "scan_042.nexus")

    assert resolve_output_target(
        tmp_path, "scan_042", mode="Append",
        explicit_target=legacy) == legacy


# ---------------------------------------------------------------------------
# Row 11 — ``open_source()`` explicitly routes ``.nexus`` as processed output
# ---------------------------------------------------------------------------

def test_row11_guess_source_kind_routes_nexus_as_processed(tmp_path):
    from xrd_tools.core.scan import SourceKind
    from xrd_tools.sources.registry import guess_source_kind

    processed = _write_processed(tmp_path / "scan.nexus")
    assert guess_source_kind(processed) is SourceKind.PROCESSED_NEXUS


def test_row11_open_source_opens_a_nexus_processed_record(tmp_path):
    from xrd_tools.sources.registry import open_source

    processed = _write_processed(tmp_path / "scan.nexus")
    source = open_source(processed)
    assert type(source).__name__ == "ProcessedNexusSource"
    assert list(source.frame_indices) == [0, 1]


# ---------------------------------------------------------------------------
# Row 12 — raw discovery excludes ``.nexus``
# ---------------------------------------------------------------------------

def test_row12_enumerate_candidates_excludes_nexus(tmp_path):
    from xrd_tools.sources.discover import enumerate_candidates

    _write_processed(tmp_path / "out.nexus")
    _write_raw_master(tmp_path / "raw.nxs")

    names = {Path(c.path).name for c in enumerate_candidates(tmp_path)}
    assert "raw.nxs" in names
    assert "out.nexus" not in names


def test_row12_discover_scans_excludes_nexus_from_raw_containers(tmp_path):
    """The raw NeXus-family directory scan must not surface a ``.nexus``."""
    from xrd_tools.sources.discover import discover_scans

    _write_raw_master(tmp_path / "raw.nxs")
    _write_processed(tmp_path / "out.nexus")

    names = {Path(spec.uri).name
             for spec in discover_scans(tmp_path, "nexus_stack")}
    assert "raw.nxs" in names
    assert "out.nexus" not in names


def test_row12_nexus_is_not_a_supported_raw_image_extension():
    from xrd_tools.io.image import SUPPORTED_EXTS

    assert ".nexus" not in SUPPORTED_EXTS


def test_row12_find_image_files_ignores_nexus(tmp_path):
    from xrd_tools.io.image import find_image_files

    (tmp_path / "frame_0001.tif").write_bytes(b"x")
    _write_processed(tmp_path / "out.nexus")

    found = {Path(p).name for p in find_image_files(tmp_path)}
    assert found == {"frame_0001.tif"}


# ---------------------------------------------------------------------------
# Row 13 — collision rejection is identical for ``.nexus``
# ---------------------------------------------------------------------------

def test_row13_same_file_collision_rejected_for_nexus(tmp_path):
    from xrd_tools.io.output_safety import (
        OutputCollisionError, check_output_not_source,
    )

    source = _write_processed(tmp_path / "scan.nexus")
    with pytest.raises(OutputCollisionError):
        check_output_not_source(source, input_files=[source])


def test_row13_symlink_collision_rejected_for_nexus(tmp_path):
    from xrd_tools.io.output_safety import (
        OutputCollisionError, check_output_not_source,
    )

    source = _write_processed(tmp_path / "scan.nexus")
    link = tmp_path / "alias.nexus"
    try:
        link.symlink_to(source)
    except (OSError, NotImplementedError):        # pragma: no cover
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(OutputCollisionError):
        check_output_not_source(link, input_files=[source])


def test_row13_watched_tree_collision_rejected_for_nexus(tmp_path):
    from xrd_tools.io.output_safety import (
        OutputCollisionError, check_output_not_source,
    )

    watched = tmp_path / "raw"
    watched.mkdir()
    out = watched / "scan.nexus"

    with pytest.raises(OutputCollisionError):
        check_output_not_source(
            out, watched_dirs=[watched], container_directory_mode=True)


def test_row13_a_safe_nexus_target_outside_the_tree_is_accepted(tmp_path):
    from xrd_tools.io.output_safety import check_output_not_source

    watched = tmp_path / "raw"
    watched.mkdir()
    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    check_output_not_source(
        out_dir / "scan.nexus",
        watched_dirs=[watched],
        container_directory_mode=True,
    )


# ---------------------------------------------------------------------------
# Row 14 — the owner is Qt-free, h5py-free, numpy-free, xdart-free
# ---------------------------------------------------------------------------

def test_row14_owner_module_is_pure():
    """``output_path.py``'s OWN imports pull nothing heavy.

    Loaded BY FILE PATH so the ``xrd_tools.io`` package ``__init__`` — which
    does import h5py/numpy — cannot mask the check (same trick as the
    cadence/display-logic purity guards).
    """
    module = SRC / "xrd_tools" / "io" / "output_path.py"
    code = (
        "import sys, importlib.util;"
        f"spec=importlib.util.spec_from_file_location('op_isolated', {os.fspath(module)!r});"
        "mod=importlib.util.module_from_spec(spec);"
        "sys.modules[spec.name]=mod; spec.loader.exec_module(mod);"
        "bad=[m for m in ('PySide6','PyQt5','pyqtgraph','h5py','numpy','pyFAI',"
        "'fabio','pandas','xdart') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"owner pulled heavy deps: {out.stdout!r}"


def test_row14_owner_imports_no_writer_or_schema_code():
    """Static check: the owner references no writer/schema/transaction module."""
    module = SRC / "xrd_tools" / "io" / "output_path.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("nexus", "schema", "write", "h5py", "numpy", "xdart",
                 "pyFAI", "fabio", "PySide6")
    offenders = [m for m in imported
                 if any(bad.lower() in m.lower() for bad in forbidden)]
    assert offenders == [], offenders
    assert set(imported) <= {"os", "pathlib", "__future__", "typing"}, imported


# ---------------------------------------------------------------------------
# Correction round 2 — case-insensitive, spelling-preserving reader policy
#
# §2 rule 5 promises case-insensitive suffix recognition and no silent path
# rewrite.  Classification honoured it; the implicit Append lookup and the
# default time-resolved globs did not, so on a case-sensitive filesystem an
# existing ``scan.NEXUS`` was ignored and silently respelled.
# ---------------------------------------------------------------------------

def _case_sensitive_fs(directory: Path) -> bool:
    """Whether *directory*'s filesystem distinguishes ``A`` from ``a``.

    macOS ships case-INsensitive APFS by default, where the two spellings of one
    stem are the same file and cannot both exist; Linux CI is case-sensitive.
    Only the within-class canonical-preference row needs both to coexist.
    """
    probe = directory / "_CaseProbe"
    probe.write_text("x", encoding="utf-8")
    try:
        return not (directory / "_caseprobe").exists()
    finally:
        probe.unlink()


def test_c2row1_append_preserves_a_real_uppercase_nexus_path(tmp_path):
    """C2 row 1 — a real ``scan.NEXUS`` is found and returned in ITS spelling."""
    from xrd_tools.io import resolve_output_target

    real = _write_processed(tmp_path / "scan_042.NEXUS")
    before = real.read_bytes()

    resolved = resolve_output_target(tmp_path, "scan_042", mode="Append")

    assert resolved.name == "scan_042.NEXUS"
    assert resolved == real
    assert real.read_bytes() == before          # selection never rewrites bytes


def test_c2row2_append_reuses_a_sole_uppercase_legacy_file(tmp_path):
    """C2 row 2a — a sole ``scan.NXS`` is reused, spelling preserved."""
    from xrd_tools.io import resolve_output_target

    real = _write_processed(tmp_path / "scan_042.NXS")

    resolved = resolve_output_target(tmp_path, "scan_042", mode="Append")

    assert resolved.name == "scan_042.NXS"
    assert resolved == real


def test_c2row2_mixed_case_both_siblings_prefer_the_new_suffix(tmp_path):
    """C2 row 2b — ``.nexus`` still wins across a case boundary (§2 rule 4)."""
    from xrd_tools.io import resolve_output_target

    _write_processed(tmp_path / "scan_042.NEXUS")
    _write_processed(tmp_path / "scan_042.nxs")

    resolved = resolve_output_target(tmp_path, "scan_042", mode="Append")

    assert resolved.name == "scan_042.NEXUS"


def test_c2row2_canonical_lowercase_wins_within_a_suffix_class(tmp_path):
    """Within one suffix class the exact canonical spelling is preferred.

    Requires a case-sensitive filesystem: elsewhere the two spellings ARE one
    file, so there is nothing to prefer.  Reported as a conditional row.
    """
    from xrd_tools.io import resolve_output_target

    if not _case_sensitive_fs(tmp_path):
        pytest.skip("case-insensitive filesystem: one file, nothing to rank")
    _write_processed(tmp_path / "scan_042.NEXUS")
    _write_processed(tmp_path / "scan_042.nexus")

    resolved = resolve_output_target(tmp_path, "scan_042", mode="Append")

    assert resolved.name == "scan_042.nexus"


def test_c2row2_the_scan_stem_itself_is_never_case_folded(tmp_path):
    """Only the SUFFIX is matched case-insensitively — never the stem."""
    from xrd_tools.io import resolve_output_target

    _write_processed(tmp_path / "SCAN_042.nexus")

    resolved = resolve_output_target(tmp_path, "scan_042", mode="Append")

    assert resolved.name == "scan_042.nexus"        # a different scan entirely


def test_c2row2_an_explicit_target_is_still_returned_unchanged(tmp_path):
    """Case-insensitive lookup must not touch the explicit-target path."""
    from xrd_tools.io import resolve_output_target

    _write_processed(tmp_path / "scan_042.NEXUS")
    explicit = tmp_path / "chosen.NXS"

    resolved = resolve_output_target(
        tmp_path, "scan_042", mode="Append", explicit_target=explicit)

    assert resolved == explicit


def test_c2row3_default_discovery_finds_mixed_case_processed_records(tmp_path):
    """C2 row 3a — default discovery is case-insensitive over both suffixes."""
    from xrd_tools.analysis.time_resolved import discover_processed_scans

    _write_processed(tmp_path / "scan_1.NEXUS")
    _write_processed(tmp_path / "scan_2.NXS")
    _write_processed(tmp_path / "scan_3.nexus")

    found = discover_processed_scans(tmp_path)

    assert [p.name for p in found] == [
        "scan_1.NEXUS", "scan_2.NXS", "scan_3.nexus"]


def test_c2row3_explicit_pattern_narrowing_stays_exact(tmp_path):
    """C2 row 3b — an explicit glob keeps its exact, case-sensitive meaning."""
    from xrd_tools.analysis.time_resolved import discover_processed_scans

    _write_processed(tmp_path / "scan_1.NEXUS")
    _write_processed(tmp_path / "scan_2.nxs")

    assert [p.name for p in discover_processed_scans(tmp_path, pattern="*.nxs")
            ] == ["scan_2.nxs"]
    assert discover_processed_scans(tmp_path, pattern="*.nexus") == []


# ---------------------------------------------------------------------------
# Post-design — a non-file entry is never an output sibling
# ---------------------------------------------------------------------------

def test_pd_row6_a_directory_never_shadows_a_valid_case_variant_file(tmp_path):
    """PD row 6 — real files are filtered BEFORE canonical/class ranking.

    Needs a case-sensitive filesystem: elsewhere the two spellings are one
    entry, so the shadowing is not expressible.
    """
    from xrd_tools.io import resolve_output_target

    if not _case_sensitive_fs(tmp_path):
        pytest.skip("case-insensitive filesystem: the two spellings are one entry")
    (tmp_path / "scan_042.nexus").mkdir()       # canonical spelling is a DIRECTORY
    real = _write_processed(tmp_path / "scan_042.NEXUS")

    resolved = resolve_output_target(tmp_path, "scan_042", mode="Append")

    assert resolved == real
    assert resolved.name == "scan_042.NEXUS"


def test_pd_row6_non_file_entries_are_never_returned_as_siblings(tmp_path):
    """Platform-independent half: a directory is not an output sibling."""
    from xrd_tools.io import resolve_output_target

    (tmp_path / "scan_042.nexus").mkdir()
    legacy = _write_processed(tmp_path / "scan_042.nxs")

    assert resolve_output_target(
        tmp_path, "scan_042", mode="Append") == legacy

    legacy.unlink()
    assert resolve_output_target(
        tmp_path, "scan_042", mode="Append") == tmp_path / "scan_042.nexus"


# ---------------------------------------------------------------------------
# Correction round 2 — collision preflight on the headless writer path
#
# ``source_architecture.md`` and handoff §2 rule 8 require the suffix-independent
# guard before EVERY writer open.  ``process_series`` and ``DirectoryWatcher``
# inherit it by calling through ``process_scan``; there is no second guard.
# ---------------------------------------------------------------------------

def test_c2row10_process_scan_rejects_a_same_inode_container_target(tmp_path):
    """C2 row 10 — a hardlink alias of the container source is refused."""
    from xrd_tools.integrate.batch import process_scan
    from xrd_tools.io.output_safety import OutputCollisionError

    source = _write_raw_master(tmp_path / "raw.h5")
    before = source.read_bytes()
    alias = tmp_path / "raw_alias.h5"
    os.link(source, alias)

    with pytest.raises(OutputCollisionError):
        process_scan(source, object(), alias, npt=2, npt_rad=2, npt_azim=2)

    assert source.read_bytes() == before


def test_c2row11_process_scan_rejects_a_target_aliasing_a_directory_member(
        tmp_path):
    """C2 row 11 — the guard covers every member of an image directory."""
    from xrd_tools.integrate.batch import process_scan
    from xrd_tools.io.output_safety import OutputCollisionError

    scan_dir = tmp_path / "images"
    scan_dir.mkdir()
    member = scan_dir / "frame_0002.tif"
    member.write_bytes(b"\x49\x49\x2a\x00not-a-real-tif")
    (scan_dir / "frame_0001.tif").write_bytes(b"\x49\x49\x2a\x00other")
    before = member.read_bytes()

    alias = tmp_path / "out.h5"
    os.link(member, alias)

    with pytest.raises(OutputCollisionError):
        process_scan(scan_dir, object(), alias, npt=2, npt_rad=2, npt_azim=2)

    assert member.read_bytes() == before


def test_c2row10_a_safe_headless_target_is_still_accepted(tmp_path):
    """Retained: the preflight must not reject an ordinary separate target."""
    from xrd_tools.integrate.batch import process_scan

    source = _write_raw_master(tmp_path / "raw.h5")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # ``ai=object()`` fails per frame and is logged, but the run must reach the
    # loop at all — the guard is what this row proves does NOT fire.
    result = process_scan(source, object(), out_dir / "raw_processed.nexus",
                          npt=2, npt_rad=2, npt_azim=2)

    assert Path(result).name == "raw_processed.nexus"


# ---------------------------------------------------------------------------
# Row 15 (headless half) — function-scoped owner census
# ---------------------------------------------------------------------------

#: (relative source path, qualified function name) for every authorized
#: HEADLESS generated-output constructor.  Each MUST consume the shared owner.
HEADLESS_PRODUCERS = (
    ("xrd_tools/integrate/batch.py", "process_series"),
    ("xrd_tools/integrate/batch.py", "DirectoryWatcher._process_new_file"),
    ("xdart/modules/ewald/scan.py", "LiveScan.__init__"),
    ("xrd_tools/analysis/time_resolved.py", "discover_processed_scans"),
)

#: The owner's real API names.  A bare ``output_path`` is deliberately absent:
#: as an AST name it matches ``process_scan``'s own ``output_path`` parameter
#: and locals such as ``_append_output_path``, which consume nothing.
OWNER_NAMES = ("resolve_output_target", "default_output_path",
               "NEW_OUTPUT_SUFFIX", "READABLE_OUTPUT_SUFFIXES",
               "is_readable_output_path")

def _function_node(path: Path, qualname: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node: ast.AST = tree
    for part in qualname.split("."):
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef))
                    and child.name == part):
                node = child
                break
        else:                                       # pragma: no cover
            raise AssertionError(f"{qualname} not found in {path}")
    return node


def _legacy_literals(path: Path, qualname: str) -> list[str]:
    """Every non-docstring string constant inside *qualname* naming ``.nxs``.

    AST-based rather than textual, so a comment or docstring that merely
    mentions the legacy suffix is not a false positive, while EVERY shape of
    hard-coded legacy target — ``name + '.nxs'``, ``f'{name}.nxs'``,
    ``f'{stem}_processed.nxs'``, ``'%s.nxs' % name`` — is caught.  An authorized
    generated-output constructor must build no legacy name of any shape; raw
    *source* suffix checks live outside these functions and may remain literal
    (§5 row 15).
    """
    node = _function_node(path, qualname)
    docstrings = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.ClassDef, ast.Module)):
            body = getattr(sub, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    found = []
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                and id(sub) not in docstrings
                and ".nxs" in sub.value.lower()):
            found.append(sub.value)
    return found


def _owner_references(path: Path, qualname: str) -> set[str]:
    """Owner API names actually LOADED or CALLED inside *qualname*.

    C2 row 12.  Counts real ``ast.Name``/``ast.Attribute`` loads — which is also
    every call target, since ``Call.func`` is one of those nodes — so a comment,
    a docstring, or a dead string that merely spells ``resolve_output_target``
    can never satisfy the census.  The superseded check was a substring scan over
    the function's source text and accepted exactly that (C2 mutation 9).
    """
    node = _function_node(path, qualname)
    seen: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            seen.add(sub.id)
        elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
            seen.add(sub.attr)
    return seen & set(OWNER_NAMES)


@pytest.mark.parametrize("rel_path,qualname", HEADLESS_PRODUCERS)
def test_row15_headless_producer_consumes_the_shared_owner(rel_path, qualname):
    assert _owner_references(SRC / rel_path, qualname), (
        f"{rel_path}::{qualname} does not consume the shared path owner")


def test_c2row12_census_counts_loads_not_comments_or_dead_strings(tmp_path):
    """C2 row 12 — prose naming the owner is not consumption of the owner."""
    decoy = tmp_path / "decoy_producer.py"
    decoy.write_text(
        "def build(directory, name):\n"
        '    """Resolves via resolve_output_target."""\n'
        '    # return resolve_output_target(directory, name, mode="Append")\n'
        '    _unused = "default_output_path"\n'
        '    return f"{directory}/{name}.nexus"\n',
        encoding="utf-8")
    assert _owner_references(decoy, "build") == set()
    assert any(n in decoy.read_text(encoding="utf-8") for n in OWNER_NAMES), (
        "the decoy must still satisfy the superseded substring census")

    honest = tmp_path / "honest_producer.py"
    honest.write_text(
        "from xrd_tools.io import resolve_output_target\n"
        "def build(directory, name):\n"
        "    return resolve_output_target(directory, name, mode='Append')\n",
        encoding="utf-8")
    assert _owner_references(honest, "build") == {"resolve_output_target"}


@pytest.mark.parametrize("rel_path,qualname", HEADLESS_PRODUCERS)
def test_row15_headless_producer_has_no_hard_coded_legacy_target(
        rel_path, qualname):
    found = _legacy_literals(SRC / rel_path, qualname)
    assert found == [], (
        f"{rel_path}::{qualname} still builds a hard-coded legacy target: {found}")
