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
        if not landed:
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
    assert landed[0] == "gridding" and landed.count("authority check") >= 2, landed


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
