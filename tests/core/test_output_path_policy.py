# -*- coding: utf-8 -*-
"""P4/OUT-1 — frozen acceptance oracle for the shared ``.nexus`` output-path policy.

This file is the headless (PP-C0) half of the finite oracle frozen before
implementation.  It pins the sole Qt-free path owner
(:mod:`xrd_tools.io.output_path`), its public ``xrd_tools.io`` surface, and the
headless consumers that must generate and ordinarily read only current
``.nexus`` output.  Historical ``.nxs`` remains raw/importer territory.  The
canonical GUI consumers are pinned in
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
        entry.attrs["ssrl_schema"] = "xrd_tools.processed_scan"
        entry.attrs["ssrl_schema_version"] = 2
        g = entry.create_group("integrated_1d")
        g.attrs["NX_class"] = "NXdata"
        g.attrs["signal"] = "intensity"
        g.attrs["axes"] = ("frame_index", "q")
        g.create_dataset(
            "frame_index",
            data=frames,
            maxshape=(None,),
            chunks=(max(1, min(n_frames, 64)),),
        )
        q_ds = g.create_dataset("q", data=q)
        q_ds.attrs["units"] = "q_A^-1"
        g.create_dataset(
            "intensity",
            data=np.ones((n_frames, q_len), dtype=np.float32),
            maxshape=(None, q_len),
            chunks=(max(1, min(n_frames, 8)), q_len),
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
    assert owner.READABLE_OUTPUT_SUFFIXES == (".nexus",)
    for name in ("default_output_path", "resolve_output_target",
                 "is_readable_output_path"):
        assert callable(getattr(owner, name)), name


def test_row1_surface_is_exported_through_xrd_tools_io():
    import xrd_tools.io as io

    assert io.NEW_OUTPUT_SUFFIX == ".nexus"
    assert io.LEGACY_OUTPUT_SUFFIX == ".nxs"
    assert io.READABLE_OUTPUT_SUFFIXES == (".nexus",)
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
# Row 5 — time-resolved discovery naturally orders a mixed set
# ---------------------------------------------------------------------------

def test_row5_time_resolved_discovery_orders_mixed_suffixes(tmp_path):
    from xrd_tools.analysis.time_resolved import discover_processed_scans

    _write_processed(tmp_path / "scan_2.nexus")
    _write_processed(tmp_path / "scan_10.nxs")
    _write_processed(tmp_path / "scan_1.nxs")

    found = discover_processed_scans(tmp_path)

    assert [p.name for p in found] == ["scan_2.nexus"]


def test_row5_explicit_pattern_does_not_bypass_current_admission(tmp_path):
    """An explicit legacy glob cannot bypass strict processed admission."""
    from xrd_tools.analysis.time_resolved import discover_processed_scans

    _write_processed(tmp_path / "scan_2.nexus")
    _write_processed(tmp_path / "scan_10.nxs")

    found = discover_processed_scans(tmp_path, pattern="*.nxs")
    assert found == []


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
# Row 8 — processed targets normalize to ``.nexus``
# ---------------------------------------------------------------------------

def test_row8_explicit_legacy_target_is_normalized_by_the_resolver(tmp_path):
    from xrd_tools.io import resolve_output_target

    explicit = tmp_path / "operator_choice.nxs"
    for mode in ("Append", "Overwrite"):
        assert resolve_output_target(
            tmp_path, "scan_042", mode=mode,
            explicit_target=explicit) == explicit.with_suffix(".nexus")


def test_row8_explicit_new_target_is_preserved_by_the_resolver(tmp_path):
    from xrd_tools.io import resolve_output_target

    explicit = tmp_path / "operator_choice.nexus"
    assert resolve_output_target(
        tmp_path, "scan_042", mode="Append",
        explicit_target=explicit) == explicit


def test_row8_legacy_nxs_is_not_a_processed_output_path(tmp_path):
    from xrd_tools.io import is_readable_output_path

    legacy = _write_processed(tmp_path / "legacy.nxs")
    assert not is_readable_output_path(legacy)


def test_row8_suffix_recognition_is_case_insensitive(tmp_path):
    from xrd_tools.io import is_readable_output_path

    assert not is_readable_output_path(tmp_path / "a.NXS")
    assert is_readable_output_path(tmp_path / "b.Nexus")
    assert not is_readable_output_path(tmp_path / "c.h5")
    assert not is_readable_output_path(tmp_path / "d.tif")


# ---------------------------------------------------------------------------
# Row 9 — Append ignores a historical ``.nxs`` sibling
# ---------------------------------------------------------------------------

def test_row9_append_ignores_a_sole_existing_legacy_file(tmp_path):
    from xrd_tools.io import resolve_output_target

    legacy = _write_processed(tmp_path / "scan_042.nxs")
    target = resolve_output_target(tmp_path, "scan_042", mode="Append")
    assert target == legacy.with_suffix(".nexus")


def test_row9_append_selects_a_distinct_target_and_preserves_legacy_bytes(tmp_path):
    from xrd_tools.io import resolve_output_target
    from xrd_tools.io.output_safety import check_output_not_source

    legacy = _write_processed(tmp_path / "scan_042.nxs")
    before = legacy.read_bytes()

    target = resolve_output_target(tmp_path, "scan_042", mode="Append")
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


def test_row10_explicit_legacy_normalizes_to_existing_nexus(tmp_path):
    from xrd_tools.io import resolve_output_target

    legacy = _write_processed(tmp_path / "scan_042.nxs")
    _write_processed(tmp_path / "scan_042.nexus")

    assert resolve_output_target(
        tmp_path, "scan_042", mode="Append",
        explicit_target=legacy) == legacy.with_suffix(".nexus")


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
# Row 12 — raw discovery includes ``.nexus`` for structure-based admission
# ---------------------------------------------------------------------------

def test_row12_enumerate_candidates_includes_nexus_for_probe(tmp_path):
    from xrd_tools.sources.discover import enumerate_candidates

    _write_processed(tmp_path / "out.nexus")
    _write_raw_master(tmp_path / "raw.nxs")

    names = {Path(c.path).name for c in enumerate_candidates(tmp_path)}
    assert "raw.nxs" in names
    assert "out.nexus" in names


def test_row12_discover_scans_partitions_raw_and_processed_nexus(tmp_path):
    from xrd_tools.sources.discover import discover_scans

    _write_raw_master(tmp_path / "raw.nxs")
    _write_processed(tmp_path / "out.nexus")

    raw_names = {Path(spec.uri).name
                 for spec in discover_scans(tmp_path, "nexus_stack")}
    processed_names = {
        Path(spec.uri).name
        for spec in discover_scans(tmp_path, "processed_nexus")
    }
    assert raw_names == {"raw.nxs"}
    assert processed_names == {"out.nexus"}


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
    assert set(imported) <= {
        "hashlib", "os", "pathlib", "re", "__future__", "typing",
    }, imported


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


def test_c2row2_append_ignores_a_sole_uppercase_legacy_file(tmp_path):
    from xrd_tools.io import resolve_output_target

    real = _write_processed(tmp_path / "scan_042.NXS")

    resolved = resolve_output_target(tmp_path, "scan_042", mode="Append")

    assert resolved.name == "scan_042.nexus"
    assert resolved != real


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


def test_c2row2_an_explicit_legacy_target_is_normalized(tmp_path):
    from xrd_tools.io import resolve_output_target

    _write_processed(tmp_path / "scan_042.NEXUS")
    explicit = tmp_path / "chosen.NXS"

    resolved = resolve_output_target(
        tmp_path, "scan_042", mode="Append", explicit_target=explicit)

    assert resolved == explicit.with_suffix(".nexus")


def test_c2row3_default_discovery_finds_only_nexus_records(tmp_path):
    from xrd_tools.analysis.time_resolved import discover_processed_scans

    _write_processed(tmp_path / "scan_1.NEXUS")
    _write_processed(tmp_path / "scan_2.NXS")
    _write_processed(tmp_path / "scan_3.nexus")

    found = discover_processed_scans(tmp_path)

    assert [p.name for p in found] == ["scan_1.NEXUS", "scan_3.nexus"]


def test_c2row3_explicit_pattern_never_bypasses_strict_admission(tmp_path):
    """C2 row 3b — a glob narrows candidates, not reader admission."""
    from xrd_tools.analysis.time_resolved import discover_processed_scans

    _write_processed(tmp_path / "scan_1.NEXUS")
    _write_processed(tmp_path / "scan_2.nxs")

    assert discover_processed_scans(tmp_path, pattern="*.nxs") == []
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
        tmp_path, "scan_042", mode="Append") == tmp_path / "scan_042.nexus"

    legacy.unlink()
    assert resolve_output_target(
        tmp_path, "scan_042", mode="Append") == tmp_path / "scan_042.nexus"


# ---------------------------------------------------------------------------
# Row 15 (headless half) — function-scoped owner census
# ---------------------------------------------------------------------------

#: (relative source path, qualified function name) for every authorized
#: HEADLESS generated-output constructor.  Each MUST consume the shared owner.
HEADLESS_PRODUCERS = (
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


# ---------------------------------------------------------------------------
# VNX-FIXED-OPERATION-SLOTS-20260903 — stable operation result slots
#
# Frozen BEFORE the implementation, against the authority in
# `docs/decisions/0010-proportional-integrity-and-scientific-correctness.md`
# (carried as public commit 491b9413) and the design note
# `finite_operation_output_slots_2026-09-03.md`.  The contract these rows pin:
# the public NeXus filename is `<root-family><slot>.nexus`, the separator is an
# UNDERSCORE, and no version, hash, timestamp, UUID or chained operation suffix
# may appear in a public name.
# ---------------------------------------------------------------------------

#: The complete public vocabulary, transcribed from the ADR table.  Written out
#: literally rather than imported from the owner so that a typo in the owner is
#: a test failure instead of a silently agreeing constant.
_ADR_SLOTS = {
    "int-1d": "_int1d",
    "int-2d": "_int2d",
    "average": "_average",
    "reintegrate-1d": "_reintegrate1d",
    "reintegrate-2d": "_reintegrate2d",
    "stitch-1d": "_stitch1d",
    "stitch-2d": "_stitch2d",
    "rsm": "_rsm",
}


@pytest.mark.parametrize("kind,slot", sorted(_ADR_SLOTS.items()))
def test_slots_resolve_the_exact_underscore_spelling(tmp_path, kind, slot):
    """Every operation resolves `<family><slot>.nexus` and nothing else."""
    from xrd_tools.io.output_path import resolve_finite_output_target

    resolved = resolve_finite_output_target(
        tmp_path, "sample", operation_token=kind,
    )
    assert resolved == tmp_path / f"sample{slot}.nexus"
    # The separator is an underscore, not the dot the superseded immutable
    # successor route used.  A dot would read as a suffix to every path tool.
    assert resolved.name == f"sample{slot}.nexus"
    assert f"sample.{kind}" not in resolved.name


def test_slot_resolution_refuses_an_unknown_operation(tmp_path):
    """A closed vocabulary: an unlisted operation cannot invent a public name."""
    from xrd_tools.io.output_path import resolve_finite_output_target

    for unknown in ("integrate", "average-2", "reintegrate", "", "RSM"):
        with pytest.raises(ValueError, match="finite operation"):
            resolve_finite_output_target(
                tmp_path, "sample", operation_token=unknown,
            )


def test_public_slot_names_carry_no_version_hash_or_counter(tmp_path):
    """No version, content hash, timestamp, UUID or counter in a public name."""
    from xrd_tools.io.output_path import resolve_finite_output_target

    for kind in _ADR_SLOTS:
        name = resolve_finite_output_target(
            tmp_path, "sample", operation_token=kind,
        ).name
        # Any run of 8+ hex characters would be a version/hash leak.  `sample`
        # and the slot words are deliberately not hex-shaped.
        assert re.search(r"[0-9a-f]{8,}", name) is None, name
        assert re.search(r"-\d+\b", name) is None, name
        assert name.count(".") == 1, name


def test_resolution_takes_no_version_identity(tmp_path):
    """The owner cannot be handed a version, so it cannot publish one.

    Structural, not behavioural: while the parameter exists a future caller can
    reintroduce a public version name by passing it.  Removing it from the
    signature is what makes the ADR's "no public version names" rule
    unbreakable at this seam.
    """
    import inspect

    from xrd_tools.io.output_path import resolve_finite_output_target

    parameters = inspect.signature(resolve_finite_output_target).parameters
    assert "version_identity" not in parameters
    assert "explicit_target" not in parameters


def test_repeating_an_operation_reuses_the_same_slot(tmp_path):
    """Repeat execution replaces its own slot; it never accumulates files."""
    from xrd_tools.io.output_path import resolve_finite_output_target

    first = resolve_finite_output_target(
        tmp_path, "sample", operation_token="average",
    )
    first.write_bytes(b"prior")
    second = resolve_finite_output_target(
        tmp_path, "sample", operation_token="average",
    )
    # An OCCUPIED slot is still the answer.  Selecting a different name on the
    # second run is exactly the accumulating public sequence the ADR forbids.
    assert second == first
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sample_average.nexus"]


def test_a_persisted_family_prevents_suffix_chaining(tmp_path):
    """Operating on a slot-named result consumes its family, never its stem.

    The superseded route derived the family from the source stem, so feeding a
    generated artifact back produced a doubled suffix.  Under stable slots that
    same mistake would yield `sample_average_reintegrate1d.nexus`, which the
    design forbids by name.
    """
    from xrd_tools.io.output_path import (
        artifact_family_from_source,
        resolve_finite_output_target,
    )

    published = resolve_finite_output_target(
        tmp_path, "sample", operation_token="average",
    )
    assert published.name == "sample_average.nexus"

    # The family travels in provenance, so the next operation is handed it.
    family = artifact_family_from_source(published, "sample")
    chained = resolve_finite_output_target(
        tmp_path, family, operation_token="reintegrate-1d",
    )
    assert chained == tmp_path / "sample_reintegrate1d.nexus"
    assert "_average_" not in chained.name

    # And without a persisted family the stem is retained verbatim -- the owner
    # must NOT strip `_average` to guess a root, because "do not parse
    # generated suffixes to recover a root family" is an explicit stop
    # condition.  This is why consuming the persisted family is mandatory.
    assert artifact_family_from_source(published) == "sample_average"


def test_the_gi_marker_is_part_of_the_family_and_is_never_repeated(tmp_path):
    """`sample_gi_int2d.nexus`, then `sample_gi_reintegrate1d.nexus`.

    The marker is decided once, where a family is first derived from a raw scan
    name.  Every later operation consumes the PERSISTED family verbatim, so an
    already processed input cannot acquire a second marker, and the operation
    slot stays last.  Standard names are unchanged.
    """
    from xrd_tools.io.output_path import (
        artifact_family_from_source,
        generated_artifact_family,
        is_artifact_family,
        resolve_finite_output_target,
    )

    assert generated_artifact_family("sample", grazing_incidence=False) == "sample"
    family = generated_artifact_family("sample", grazing_incidence=True)
    assert family == "sample_gi" and is_artifact_family(family)

    run = resolve_finite_output_target(tmp_path, family, operation_token="int-2d")
    assert run == tmp_path / "sample_gi_int2d.nexus"
    for token, name in (
        ("int-1d", "sample_gi_int1d.nexus"),
        ("reintegrate-1d", "sample_gi_reintegrate1d.nexus"),
        ("reintegrate-2d", "sample_gi_reintegrate2d.nexus"),
        ("average", "sample_gi_average.nexus"),
    ):
        consumed = artifact_family_from_source(run, family)
        assert resolve_finite_output_target(
            tmp_path, consumed, operation_token=token,
        ) == tmp_path / name

    # An older GI result persisted the unmarked family.  It stays readable under
    # its own name and its successors stay in ITS family: nothing reads the
    # scientific mode back out of a filename to rename them.
    legacy = artifact_family_from_source(tmp_path / "sample_int2d.nexus", "sample")
    assert resolve_finite_output_target(
        tmp_path, legacy, operation_token="reintegrate-1d",
    ) == tmp_path / "sample_reintegrate1d.nexus"


def test_readable_stems_stay_readable_and_unsafe_ones_are_refused(tmp_path):
    """Real beamline stems keep their name; only structurally unsafe ones hash.

    Fable F5 on `c167b71a`, RULED by the maintainer 2026-09-04: widen the family
    alphabet rather than document an exception. The previous ASCII-only class
    sent `Sample A 001` to the `artifact-<24 hex>` fallback, putting a content
    hash in a public name in direct contradiction of the ADR this same work
    carried, and stems with spaces are routine at a beamline.

    The fallback is kept for names that would be structurally dangerous rather
    than merely unusual, and each refusal has a reason: a leading dot (hidden
    files, and the private candidate namespace lives there), a leading dash
    (reads as an option to command-line tools), `:` (drive and alternate-data-
    stream separator on Windows), and a leading underscore (would read as a bare
    slot).
    """
    from xrd_tools.io.output_path import (
        artifact_family_from_source,
        resolve_finite_output_target,
    )

    for stem in ("Sample A 001", "basé", "日本語", "scan#12", "scan_12-a.b",
                 "sample_average"):
        family = artifact_family_from_source(tmp_path / f"{stem}.nexus")
        assert family == stem, f"{stem!r} should stay readable"
        name = resolve_finite_output_target(
            tmp_path, family, operation_token="average",
        ).name
        assert name == f"{stem}_average.nexus"
        # No hash reached the public name.
        assert re.search(r"[0-9a-f]{8,}", name) is None, name

    # A structurally unusable stem is REFUSED, not silently hashed. The old
    # `artifact-<24 hex>` fallback put a content hash in a public name against
    # the ADR, and gave the operator an unreadable filename with no reason for
    # it. With the alphabet widened, what is left is genuinely unusable.
    for stem in ("-leading-dash", ".hidden", "a:b", "_leading"):
        with pytest.raises(ValueError, match="public result family"):
            artifact_family_from_source(tmp_path / f"{stem}.nexus")

    # A stem that is already safe is retained verbatim -- no hashing, no
    # stripping, including one that merely LOOKS like a slot name.
    for stem in ("scan_12-a.b", "sample_average"):
        assert artifact_family_from_source(tmp_path / f"{stem}.nexus") == stem
