"""NEXT-1219-SPEC-OCCURRENCE: a finite RSM or Stitch job binds its own SPEC scan, not the whole SPEC file.

His words (2026-11-23): "Bind each scan's bytes (Recommended)": later scans being written no longer refuse a job;
a rewrite of its own bytes still does; "Regrid once (Recommended)" for HKL maps saved before the fix. Contract:
the task folder's design_spec_occurrence.v2.md (Codex). Real operations, SPEC files and images; an append lands at a
named boundary of the real run.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tests.core.test_rsm_operation_r2 import _form, _runnable, _write_member
from tests.core.test_stitch_operation import _native_form
from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleKind, ModuleSourceReceipt
from xrd_tools.analysis.rsm_operation import RSMUBOverride, run_rsm_operation_v2
from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    MetadataTablePlan,
    MetadataTableRequalificationPlan,
    run_metadata_table,
    run_metadata_table_requalification,
)
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.analysis.stitch_values import prepare_stitch_tool
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.spec import SpecSource

LATER_SCAN = ("\n#S 2 ascan eta 0 1 1 1\n#D Mon Jan 15 10:40:00 2024\n#P0 13000.007 10 20 30 40 50\n"
              "#G3 1 0 0 0 1 0 0 0 1\n#N 4\n#L eta  Seconds  Seconds  foil status\n0 1 10 101\n")


def _append(path, text=LATER_SCAN):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _rewrite(path, old, new):
    """A same-length edit (the file keeps its size)."""
    assert len(old) == len(new)
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    Path(path).write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def _committed(result):
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, (result.terminal.code, result.terminal)
    return result


def _rsm(root, *, members=1, ub=None):
    form = _form(root, tuple(_write_member(root, index) for index in range(members)))
    return form if ub is None else replace(form, ub=ub)


# --- S1: a later scan written after Preview --------------------------------------------------------------------
def test_s1_rsm_commits_when_a_later_scan_and_its_images_are_written_after_preview(tmp_path, monkeypatch):
    _runnable(monkeypatch)
    (tmp_path / "plain").mkdir()
    (tmp_path / "grown").mkdir()
    unchanged = _committed(run_rsm_operation_v2(prepare_rsm_tool_v2(_rsm(tmp_path / "plain")).request))
    form = _rsm(tmp_path / "grown")
    request = prepare_rsm_tool_v2(form).request
    member = form.members[0]
    _append(member.spec_path)
    tifffile.imwrite(Path(member.image_dir) / "RSM_synth_0_scan2_0000.tif", np.ones((4, 5), dtype=np.uint16))
    grown = _committed(run_rsm_operation_v2(request))
    assert grown.payload.result_fingerprint == unchanged.payload.result_fingerprint


def test_s1_control_a_later_image_alone_still_commits(tmp_path, monkeypatch):
    _runnable(monkeypatch)
    form = _rsm(tmp_path)
    request = prepare_rsm_tool_v2(form).request
    tifffile.imwrite(Path(form.members[0].image_dir) / "other_scan2_0000.tif", np.ones((4, 5), dtype=np.uint16))
    _committed(run_rsm_operation_v2(request))


# --- S2: a later scan written during gridding and right before each authority check ----------------------------
def test_s2_rsm_commits_when_later_scans_land_during_gridding_and_before_publication(tmp_path, monkeypatch):
    import xrd_tools.analysis.rsm_operation as rsm_operation
    _runnable(monkeypatch)
    form = _rsm(tmp_path)
    spec = form.members[0].spec_path
    request = prepare_rsm_tool_v2(form).request
    landed = []
    loader = SpecSource.load_frame

    def growing_frame(self, index):
        if "gridding" not in landed:
            _append(spec, LATER_SCAN.replace("#S 2", "#S 3"))
            landed.append("gridding")
        return loader(self, index)

    checks = rsm_operation._revalidate_rsm_v2_authorities

    def growing_check(*args, **kwargs):
        _append(spec, LATER_SCAN.replace("#S 2", f"#S {4 + len(landed)}"))
        landed.append("authority check")
        return checks(*args, **kwargs)

    monkeypatch.setattr(SpecSource, "load_frame", growing_frame)
    monkeypatch.setattr(rsm_operation, "_revalidate_rsm_v2_authorities", growing_check)
    _committed(run_rsm_operation_v2(request))
    assert "gridding" in landed and landed.count("authority check") >= 3, landed  # Run start, after science, prepublish


def test_s2_the_v1_rsm_route_commits_when_a_later_scan_lands_during_science(tmp_path, monkeypatch):
    import xrd_tools.analysis.rsm_operation as rsm_operation
    from tests.core.test_rsm_operation import _prepared_rsm, _synthetic_volume
    request, _frames = _prepared_rsm(tmp_path)
    spec = tmp_path / "source" / "RSM_synth"
    landed = []

    def growing_science(*_args, **_kwargs):
        _append(spec)
        landed.append("science")
        from types import SimpleNamespace
        return SimpleNamespace(payload=_synthetic_volume(request))

    monkeypatch.setattr(rsm_operation, "run_rsm", growing_science)
    result = rsm_operation.run_rsm_operation(request)
    assert landed == ["science"]
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal


# --- S5: a changed detector image still refuses ---------------------------------------------------------------------
def test_s5_a_changed_image_still_refuses_under_an_unchanged_scan(tmp_path, monkeypatch):
    _runnable(monkeypatch)
    form = _rsm(tmp_path)
    request = prepare_rsm_tool_v2(form).request
    image = Path(form.members[0].image_dir) / "member0_0000.tif"
    image.write_bytes(image.read_bytes() + b"drift")
    result = run_rsm_operation_v2(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED, result.terminal
    assert not Path(request.module.output.target).exists()


# --- S9: a SPEC sidecar of another source kind stays exact -----------------------------------------------------------
def test_s9_a_tiffs_spec_sidecar_keeps_the_exact_rule(tmp_path):
    spec = tmp_path / "sample"
    spec.write_text("#F sample\n#E 1\n#O0 theta\n\n#S 1 ascan theta 0 1 1 1\n#P0 5\n#N 2\n#L theta  I0\n"
                    "5 100\n6 200\n", encoding="utf-8", newline="\n")
    image = tmp_path / "sample_scan1_0000.tif"
    tifffile.imwrite(image, np.ones((4, 5), dtype=np.uint16))
    table = run_metadata_table(MetadataTablePlan(str(image), metadata_format="spec"))
    assert table.disposition is AnalysisDisposition.COMPLETED, table.code
    assert "I0" in {column.name for column in table.columns}, [column.name for column in table.columns]
    _append(spec, "\n#S 2 ascan theta 0 1 1 1\n#P0 5\n#N 2\n#L theta  I0\n5 100\n")
    requalified = run_metadata_table_requalification(
        MetadataTableRequalificationPlan(table.receipt, table.table_fingerprint))
    assert requalified.disposition is AnalysisDisposition.REFUSED, requalified.code


# --- S3: the job's own scan or its header changed: refused ----------------------------------------------------------
@pytest.mark.parametrize("change", ["selected row", "file header", "new selected row"])
def test_s3_rsm_refuses_when_its_own_scan_or_header_changes(tmp_path, monkeypatch, change):
    _runnable(monkeypatch)
    form = _rsm(tmp_path)
    spec = form.members[0].spec_path
    request = prepare_rsm_tool_v2(form).request
    if change == "selected row":
        _rewrite(spec, "1.0 2 20 110", "1.0 2 20 111")
    elif change == "file header":
        _rewrite(spec, "#O0 energy  mu", "#O0 energy  mx")
    else:
        _append(spec, "2.0 3 30 120\n")
    from xrd_tools.io.spec import UnusableSpecScan
    try:
        result = run_rsm_operation_v2(request)
    except UnusableSpecScan:  # pre-existing: a new selected row without its image escapes as the source's refusal
        assert change == "new selected row"
    else:
        assert result.terminal.disposition is ModuleDisposition.REFUSED, result.terminal
    assert not Path(request.module.output.target).exists()


def test_s3_a_later_scan_with_its_data_still_coming_does_not_refuse(tmp_path, monkeypatch):
    _runnable(monkeypatch)
    form = _rsm(tmp_path)
    request = prepare_rsm_tool_v2(form).request
    _append(form.members[0].spec_path, LATER_SCAN.split("#L")[0])  # #S 2 begun, no #L or row yet
    _committed(run_rsm_operation_v2(request))


# --- S4: HKL with a UB taken from a separate SPEC file -----------------------------------------------------------
def _hkl(root):
    ub_member = _write_member(root, 9, ub=(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0))
    return _rsm(root, ub=RSMUBOverride(spec_path=str(ub_member.spec_path), scan="1.1")), ub_member.spec_path


@pytest.mark.parametrize("growth", ["later scan", "blank line"])
def test_s4_an_hkl_map_commits_when_its_ub_spec_file_grows(tmp_path, monkeypatch, growth):
    _runnable(monkeypatch)
    form, ub_spec = _hkl(tmp_path)
    request = prepare_rsm_tool_v2(form).request
    _append(ub_spec, LATER_SCAN if growth == "later scan" else "\n")
    _committed(run_rsm_operation_v2(request))


def test_s4_an_hkl_map_refuses_when_its_ub_line_is_rewritten(tmp_path, monkeypatch):
    _runnable(monkeypatch)
    form, ub_spec = _hkl(tmp_path)
    request = prepare_rsm_tool_v2(form).request
    _rewrite(ub_spec, "#G3 2.0 0.0", "#G3 3.0 0.0")
    result = run_rsm_operation_v2(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED, result.terminal


def test_s4b_an_hkl_map_stays_current_for_append_after_its_ub_file_grows(tmp_path, monkeypatch):
    """His "Regrid once": a map made after the fix stays current across unrelated growth; a changed UB matrix
    does not."""
    _runnable(monkeypatch)
    form, ub_spec = _hkl(tmp_path)
    _committed(run_rsm_operation_v2(prepare_rsm_tool_v2(form).request))
    assert prepare_rsm_tool_v2(form).output_current is True
    _append(ub_spec)
    assert prepare_rsm_tool_v2(form).output_current is True
    _rewrite(ub_spec, "#G3 2.0 0.0", "#G3 3.0 0.0")
    assert prepare_rsm_tool_v2(form).output_current is False


# --- S6: Stitch with a SPEC source ---------------------------------------------------------------------------------
@pytest.mark.parametrize("scans", ["1", "[1,2]"])
def test_s6_stitch_commits_when_a_later_scan_is_written_after_preview(tmp_path, scans):
    (tmp_path / "plain").mkdir()
    (tmp_path / "grown").mkdir()
    unchanged = _committed(run_stitch_operation(prepare_stitch_tool(
        _native_form(tmp_path / "plain", "multigeometry", scans=scans)).request))
    form = _native_form(tmp_path / "grown", "multigeometry", scans=scans)
    request = prepare_stitch_tool(form).request
    _append(form.spec_path, "\n#S 4 ascan del 14 15 1 1\n#D today\n#P0 14 -10\n#N 2\n#L del  nu\n14 -10\n")
    grown = _committed(run_stitch_operation(request))
    np.testing.assert_array_equal(grown.payload.intensity, unchanged.payload.intensity)


def test_s6_stitch_refuses_when_its_own_scan_is_rewritten(tmp_path):
    form = _native_form(tmp_path, "multigeometry", scans="1")
    request = prepare_stitch_tool(form).request
    text = Path(form.spec_path).read_text(encoding="utf-8")
    first = text.index("#S 1")
    row = text.index("15 -10", first)
    Path(form.spec_path).write_text(text[:row] + "15 -11" + text[row + 6:], encoding="utf-8", newline="\n")
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED, result.terminal


# --- S7: the metadata table, its requalification and the module receipt factory --------------------------------
def _spec_table(root):
    member = _write_member(root, 0)
    plan = MetadataTablePlan(SourceSpec(member.spec_path, SourceKind.SPEC, options={
        "scan": "1.1", "image_dir": str(member.image_dir), "image_stem": member.image_stem}))
    table = run_metadata_table(plan)
    assert table.disposition is AnalysisDisposition.COMPLETED, table.code
    return member, table


def test_s7_requalification_and_the_receipt_factory_accept_a_later_scan(tmp_path):
    member, table = _spec_table(tmp_path)
    _append(member.spec_path)
    requalified = run_metadata_table_requalification(
        MetadataTableRequalificationPlan(table.receipt, table.table_fingerprint))
    assert requalified.disposition is AnalysisDisposition.COMPLETED, requalified.code
    receipt = ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.RSM)
    assert receipt.analysis == table.receipt and receipt.table_fingerprint == table.table_fingerprint


def test_s7_requalification_refuses_a_rewrite_of_the_scan(tmp_path):
    member, table = _spec_table(tmp_path)
    _rewrite(member.spec_path, "1.0 2 20 110", "1.0 2 20 111")
    requalified = run_metadata_table_requalification(
        MetadataTableRequalificationPlan(table.receipt, table.table_fingerprint))
    assert requalified.disposition is AnalysisDisposition.REFUSED


# --- S7 read bound: one capture and one parser per distinct SPEC file per preparation ----------------------------
def _counting(monkeypatch):
    """Spies (calling through) on every silx open and every Python-level open, by resolved path."""
    import builtins
    import os
    import silx.io.specfile as specfile
    import xrd_tools.io.spec as io_spec
    parsers, opens = {}, {}
    real_specfile, real_open, real_os_open = specfile.SpecFile, builtins.open, os.open

    def key(path):
        try:
            return str(Path(os.fsdecode(path)).resolve())
        except (TypeError, ValueError):
            return None

    def counted_specfile(path, *args, **kwargs):
        parsers[key(path)] = parsers.get(key(path), 0) + 1
        return real_specfile(path, *args, **kwargs)

    def counted_open(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)):
            opens[key(path)] = opens.get(key(path), 0) + 1
        return real_open(path, *args, **kwargs)

    def counted_os_open(path, *args, **kwargs):
        opens[key(path)] = opens.get(key(path), 0) + 1
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr(specfile, "SpecFile", counted_specfile)
    monkeypatch.setattr(io_spec, "SpecFile", counted_specfile)
    import io
    monkeypatch.setattr(builtins, "open", counted_open)
    monkeypatch.setattr(io, "open", counted_open)  # pathlib opens through io.open
    monkeypatch.setattr(os, "open", counted_os_open)
    return parsers, opens


def _two_members_one_file(root):
    member = _write_member(root, 0)
    spec = Path(member.spec_path)
    text = spec.read_text(encoding="utf-8")
    spec.write_text(text + "\n" + text[text.index("#S 1"):].replace("#S 1", "#S 2", 1),
                    encoding="utf-8", newline="\n")
    for label in range(2):
        tifffile.imwrite(Path(member.image_dir) / f"member0b_{label:04d}.tif", np.ones((4, 5), dtype=np.uint16))
    return _form(root, (member, replace(member, scan="2.1", image_stem="member0b_")))


@pytest.mark.parametrize("case", ["one member", "two members, one file", "member and UB file"])
def test_s7_a_preparation_reads_each_spec_file_once(tmp_path, monkeypatch, case):
    _runnable(monkeypatch)
    if case == "one member":
        form = _rsm(tmp_path)
    elif case == "two members, one file":
        form = _two_members_one_file(tmp_path)
    else:
        form, _ub = _hkl(tmp_path)
    files = {str(Path(member.spec_path).resolve()) for member in form.members}
    if form.ub is not None:
        files.add(str(Path(form.ub.spec_path).resolve()))
    parsers, opens = _counting(monkeypatch)
    prepare_rsm_tool_v2(form)
    assert sum(parsers.values()) <= len(files), parsers  # one parser per captured file, none of the live file
    for path in files:
        assert parsers.get(path, 0) <= 1 and opens.get(path, 0) <= 2, (
            Path(path).name, {Path(k).name: v for k, v in parsers.items() if k},
            {Path(k).name: v for k, v in opens.items() if k in files})


@pytest.mark.parametrize("ending", ["success", "refusal", "exception"])
def test_s7_captures_live_in_the_temporary_folder_and_none_survives(tmp_path, monkeypatch, ending):
    """The private copies are made in the system's temporary folder (never the Project or Save Path) and removed
    when the preparation ends, however it ends."""
    import tempfile
    import xrd_tools.io.spec as io_spec
    _runnable(monkeypatch)
    spool = tmp_path / "system-temp"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    made = []
    real = tempfile.NamedTemporaryFile

    def recording(*args, **kwargs):
        handle = real(*args, **kwargs)
        made.append(Path(handle.name))
        return handle

    monkeypatch.setattr(io_spec.tempfile, "NamedTemporaryFile", recording)
    form, ub_spec = _hkl(tmp_path / "project")
    if ending == "refusal":
        _rewrite(ub_spec, "#G3 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 2.0", "#X3 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 2.0")
        from xrd_tools.analysis.rsm_values import RSMToolPreflightRefused
        with pytest.raises(RSMToolPreflightRefused):
            prepare_rsm_tool_v2(form)
    elif ending == "exception":
        with pytest.raises(RuntimeError, match="stop here"), io_spec.spec_capture_scope():
            io_spec.list_spec_scans(form.members[0].spec_path)
            raise RuntimeError("stop here")
    else:
        prepare_rsm_tool_v2(form)
    assert made and all(path.parent == spool for path in made), made
    assert not any(path.exists() for path in made) and not list(spool.iterdir())


# --- S10: a repeated scan number -----------------------------------------------------------------------------------
def _repeated(root):
    """A member on the second #S 1 (key 1.2)."""
    member = _write_member(root, 0)
    spec = Path(member.spec_path)
    text = spec.read_text(encoding="utf-8")
    block = text[text.index("#S 1"):]
    spec.write_text(text + "\n" + block.replace("1.0 2 20 110", "1.0 2 20 120"), encoding="utf-8", newline="\n")
    return replace(_form(root, (replace(member, scan="1.2"),)), output_path=root / "output" / "rsm-1-2.nexus")


@pytest.mark.parametrize("change, outcome", [
    ("third occurrence", "COMMITTED"),
    ("earlier row, same width", "COMMITTED"),
    ("second occurrence row", "REFUSED"),
    ("earlier scan renumbered", "REFUSED"),
])
def test_s10_a_repeated_scan_number_binds_its_exact_occurrence(tmp_path, monkeypatch, change, outcome):
    _runnable(monkeypatch)
    form = _repeated(tmp_path)
    spec = form.members[0].spec_path
    request = prepare_rsm_tool_v2(form).request
    text = Path(spec).read_text(encoding="utf-8")
    if change == "third occurrence":
        _append(spec, "\n" + text[text.index("#S 1"):text.index("\n\n#S 1")] + "\n")
    elif change == "earlier row, same width":
        _rewrite(spec, "0.0 1 10 101\n1.0 2 20 110", "0.0 1 10 102\n1.0 2 20 110")
    elif change == "second occurrence row":
        _rewrite(spec, "1.0 2 20 120", "1.0 2 20 121")
    else:  # the first #S 1 becomes #S 7: the key 1.2 now names nothing or another block
        first = text.index("#S 1")
        Path(spec).write_text(text[:first] + "#S 7" + text[first + 4:], encoding="utf-8", newline="\n")
    result = run_rsm_operation_v2(request)
    assert result.terminal.disposition.value.upper() == outcome, (result.terminal.code, result.terminal)


# --- S11: the occurrence proof itself (new API) ---------------------------------------------------------------------
_TWO_SCANS = ("#F f\n#E 1\n#O0 eta\n\n#S 1 ascan eta 0 1 1 1\n#G3 1 0 0 0 1 0 0 0 1\n#L eta  I0\n0 10\n1 20\n\n"
              "#S 2 ascan eta 0 1 1 1\n#G3 1 0 0 0 1 0 0 0 1\n#L eta  I0\n0 30\n1 40\n")


def _proof(path, key):
    import os
    from xrd_tools.io.metadata import spec_occurrence_from_bytes
    status = os.stat(path)
    return spec_occurrence_from_bytes(Path(path).read_bytes(), path=str(path), resolved=str(Path(path).resolve()),
                                      device=status.st_dev, inode=status.st_ino, key=key)


@pytest.mark.parametrize("key, change, holds", [
    ("2.1", "later scan appended", True),
    ("2.1", "blank line appended", True),
    ("2.1", "next scan starting, unterminated", True),
    ("1.1", "later scan's row appended", True),
    ("2.1", "row appended to the scan", False),
    ("2.1", "trailing #G3 appended to the scan", False),
    ("2.1", "earlier #S renumbered to 2", False),
    ("2.1", "selected row rewritten", False),
    ("2.1", "file header rewritten", False),
    ("2.1", "earlier row of another scan, same width", True),
    ("2.1", "replaced by a copy with the same bytes", False),
])
def test_s11_the_occurrence_proof_holds_only_for_its_own_unchanged_scan(tmp_path, key, change, holds):
    from xrd_tools.io.metadata import spec_occurrence_holds
    path = tmp_path / "f"
    path.write_text(_TWO_SCANS, encoding="utf-8", newline="\n")
    proof = _proof(path, key)
    assert proof is not None and proof.rows == 2
    text = path.read_text(encoding="utf-8")
    if change == "later scan appended":
        _append(path, "\n#S 3 ascan eta 0 1 1 1\n#L eta  I0\n0 50\n")
    elif change == "blank line appended":
        _append(path, "\n")
    elif change == "next scan starting, unterminated":
        _append(path, "\n#S 3 asc")
    elif change == "later scan's row appended":
        _append(path, "2 50\n")  # row 3 of scan 2: scan 1's proof is untouched
    elif change == "row appended to the scan":
        _append(path, "2 50\n")
    elif change == "trailing #G3 appended to the scan":
        _append(path, "#G3 2 0 0 0 2 0 0 0 2\n")
    elif change == "earlier #S renumbered to 2":
        path.write_text(text.replace("#S 1 ascan", "#S 2 ascan", 1), encoding="utf-8", newline="\n")
    elif change == "selected row rewritten":
        _rewrite(path, "1 40\n", "1 41\n")
    elif change == "file header rewritten":
        _rewrite(path, "#O0 eta", "#O0 eba")
    elif change == "earlier row of another scan, same width":
        _rewrite(path, "0 10\n", "0 11\n")
    else:
        copy = tmp_path / "copy"
        copy.write_bytes(path.read_bytes())
        copy.replace(path)
    assert spec_occurrence_holds(proof) is holds


def test_s2_a_touch_of_the_spec_file_at_publication_leaves_the_job_standing(tmp_path, monkeypatch):
    """Only the scan's bytes bind the job: a new mtime with the same bytes (a sync, a backup) does not refuse it.
    (This replaces the touch that tests/core/test_rsm_operation_r2.py's prepublish row used to refuse.)"""
    import os
    import xrd_tools.analysis.rsm_operation as rsm_operation
    _runnable(monkeypatch)
    form = _rsm(tmp_path)
    spec = Path(form.members[0].spec_path)
    request = prepare_rsm_tool_v2(form).request
    real_write = rsm_operation.write_rsm
    touched = []

    def write_then_touch(*args, **kwargs):
        real_write(*args, **kwargs)
        status = spec.stat()
        os.utime(spec, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000))
        touched.append(True)

    monkeypatch.setattr(rsm_operation, "write_rsm", write_then_touch)
    _committed(run_rsm_operation_v2(request))
    assert touched == [True]
