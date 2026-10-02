"""FIT-TOOLS-2026-10 oracle row 6, trend parts (F4): the "Contributing-row mean" trend x of a saved result.

The finite arithmetic mean of one scan column over the result's contributing rows, concatenated across its member
scans (one vote per row), read through the saved contributor selection and the format-aware metadata owners.
A member lacking the column leaves the trend x unavailable; missing, non-numeric or all-empty columns fail per
result; rows are keyed by result identity, so repeated x values stay separate rows.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil

import numpy as np
import pytest

from xrd_tools.analysis.scan_operations import AnalysisDisposition, MetadataColumn, MetadataTableResult
from xrd_tools.analysis.stitch_metadata import StitchMetadataTable
from xrd_tools.analysis.trend_values import TrendValue, contributing_row_mean, read_trend_values


def _numeric(name, values) -> MetadataColumn:
    array = np.asarray(values, dtype="<f8")
    array.flags.writeable = False
    return MetadataColumn(name, "numeric", "nan", array, None, 0)


def _text(name, values) -> MetadataColumn:
    return MetadataColumn(name, "text", "none", None, tuple(values), 0)


def _member(source, labels, selected, *columns, refused=False) -> StitchMetadataTable:
    table = (MetadataTableResult(AnalysisDisposition.REFUSED, "Source changed since processing") if refused
             else MetadataTableResult(AnalysisDisposition.COMPLETED, "OK", labels=tuple(labels), columns=columns))
    return StitchMetadataTable(source, tuple(selected), tuple(str(item) for item in selected), table)


def test_contributing_row_mean_rules():
    a = _member("a", (0, 1, 2, 3), (0, 2, 3), _numeric("T", [1.0, 2.0, np.nan, 4.0]), _numeric("X", [0, 0, 0, 0]))
    b = _member("b", (5, 6), (6,), _numeric("T", [10.0, 20.0]))
    value = contributing_row_mean((a, b), "T")
    assert type(value) is TrendValue
    assert (value.status, value.code) == ("available", "OK")
    # one vote per finite contributing row: (1 + 4 + 20) / 3, not the mean of member means ((2.5 + 20) / 2)
    assert value.value == pytest.approx(25.0 / 3.0, rel=1e-15, abs=0)
    assert (value.count, value.minimum, value.maximum) == (3, 1.0, 20.0)
    assert [(m.rows, m.values, m.code) for m in value.members] == [(3, 2, "OK"), (1, 1, "OK")]
    assert value.label == "Contributing-row mean"

    lacking = contributing_row_mean((a, _member("b", (5, 6), (6,), _numeric("U", [1.0, 2.0]))), "T")
    assert (lacking.status, lacking.code, lacking.value) == ("unavailable", "MEMBER_LACKS_COLUMN", None)
    missing = contributing_row_mean((a, b), "NOPE")
    assert (missing.status, missing.code, missing.value) == ("failed", "COLUMN_MISSING", None)
    text = contributing_row_mean((a, _member("b", (5, 6), (6,), _text("T", ["warm", "hot"]))), "T")
    assert (text.status, text.code) == ("failed", "COLUMN_NOT_NUMERIC")
    empty_a = _member("a", (0, 1), (0, 1), _numeric("T", [np.nan, np.nan]))
    empty = contributing_row_mean((empty_a, _member("b", (5,), (5,), _numeric("T", [np.nan]))), "T")
    assert (empty.status, empty.code) == ("failed", "COLUMN_ALL_EMPTY")
    partial = contributing_row_mean((empty_a, b), "T")
    assert (partial.status, partial.code, partial.value) == ("unavailable", "MEMBER_LACKS_VALUES", None)
    refused = contributing_row_mean((a, _member("b", (), (6,), refused=True)), "T")
    assert (refused.status, refused.code, refused.value) == ("unavailable", "MEMBER_METADATA_UNAVAILABLE", None)
    twice = _member("a", (0, 1), (1,), _numeric("T", [1.0, 2.0]), _numeric("T", [5.0, 7.0]))
    assert contributing_row_mean((twice,), "T", occurrence=1).value == 7.0
    assert contributing_row_mean((twice,), "T").value == 2.0


def test_grouped_stitch_result_through_its_saved_contributors(tmp_path):
    from dataclasses import replace

    from tests.core.test_stitch_operation import _prepared, _source_receipt
    from xrd_tools.analysis.module_transaction import (
        ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceGroupReceipt,
    )
    from xrd_tools.analysis.stitch_operation import prepare_stitch_operation, run_stitch_operation
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
    from xrd_tools.io.output_path import resolve_finite_output_target

    plan = replace(_prepared(tmp_path).plan, native_records=True, conditioning=StitchConditioning("auto"))
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
    # each member's .txt metadata: I0 = 10, 11, 12 and del = 0, 5, 10 at labels 1, 2, 3
    source = ModuleSourceGroupReceipt.from_members((
        _source_receipt(tmp_path / "one", selected=(1, 2)),
        _source_receipt(tmp_path / "two", selected=(2,)),
    ), kind=ModuleKind.STITCH)
    output = ModuleOutputRequest(resolve_finite_output_target(tmp_path, "trend", operation_token="stitch-1d"),
                                 AnalysisArtifactKind.STITCH_1D, artifact_family="trend")
    result = run_stitch_operation(prepare_stitch_operation(source, output, plan, project_root=tmp_path))
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    rows = read_trend_values((("first", result.payload), ("again", result.payload)), "I0")
    assert [key for key, _value in rows] == ["first", "again"]  # one row per result, the same x kept twice
    for _key, value in rows:
        assert (value.status, value.count, value.minimum, value.maximum) == ("available", 3, 10.0, 11.0)
        assert value.value == pytest.approx(32.0 / 3.0, rel=1e-15, abs=0)  # (10 + 11) | (11)
        assert value.value != pytest.approx(10.75) and value.value != pytest.approx(11.0)  # member means; scan
    ((_key, delta),) = read_trend_values((("first", result.payload),), "del")
    assert delta.value == pytest.approx(10.0 / 3.0, rel=1e-15, abs=0)  # (0 + 5) | (5)


def _data_root() -> Path:
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured or not (Path(configured) / "RSM" / "STO_align").is_file():
        pytest.skip("XDART_TEST_DATA with RSM/STO_align is not configured")
    return Path(configured) / "RSM"


def _spec_column(path: Path, scan: int, column: str) -> list[float]:
    """An independent read of one SPEC scan column (``#L`` names are separated by two or more spaces)."""
    lines = path.read_text(errors="replace").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(f"#S {scan} "))
    header = next(index for index in range(start, len(lines)) if lines[index].startswith("#L "))
    names = re.split(r"\s{2,}", lines[header][3:].strip())
    rows = []
    for line in lines[header + 1:]:
        if line.startswith("#S "):
            break
        if line.strip() and not line.startswith("#"):
            rows.append(line.split())
    assert all(len(row) == len(names) for row in rows)
    return [float(row[names.index(column)]) for row in rows]


def test_grouped_spec_rsm_result_scan_42_and_43_through_its_saved_contributors(tmp_path):
    root = _data_root()
    from xdart.gui.tools.rsm_values import (
        RSMFrameSelector, RSMScanMemberForm, RSMToolFormV2, prepare_rsm_tool_v2, rsm_tool_preset,
    )
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.rsm_geometry_asset import (
        CANONICAL_RSM_GEOMETRY_LOCATOR, install_canonical_rsm_geometry_asset, rsm_geometry_asset_input,
    )
    from xrd_tools.analysis.rsm_operation import RSMFrameCorrections, run_rsm_operation_v2
    from xrd_tools.analysis.trend_values import read_rsm_metadata

    project = tmp_path / "project"
    (project / "images").mkdir(parents=True)
    shutil.copy2(root / "STO_align", project / "STO_align")
    for scan in (42, 43):  # the SPEC table numbers frames by their image files: copy the whole scan
        for path in sorted((root / "images").glob(f"b_thampy_STO_align_scan{scan}_*")):
            shutil.copy2(path, project / "images" / path.name)
    install_canonical_rsm_geometry_asset(project_root=project)
    selectors = {42: RSMFrameSelector(10, 150, 35), 43: RSMFrameSelector(5, 60, 11)}  # Step selections
    preset = rsm_tool_preset()
    members = tuple(RSMScanMemberForm(
        project / "STO_align", f"{scan}.1", project / "images", f"b_thampy_STO_align_scan{scan}_",
        selectors[scan], preset.detector_shape, preset.raw_dtype, preset.raw_header_skip, preset.motor_selectors,
    ) for scan in (42, 43))
    form = RSMToolFormV2(project, rsm_geometry_asset_input(CANONICAL_RSM_GEOMETRY_LOCATOR), members,
                         preset.conditioning, preset.normalization, (16, 16, 16), preset.chunk_size,
                         preset.max_frame_bytes, preset.max_chunk_bytes, project / "out" / "trend.nexus",
                         corrections=RSMFrameCorrections())
    result = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    payload = result.payload

    selected = {42: (10, 45, 80, 115, 150), 43: (5, 16, 27, 38, 49, 60)}
    tables = read_rsm_metadata(payload, source_root=project)
    assert [table.selected_labels for table in tables] == [selected[42], selected[43]]
    for scan, table in zip((42, 43), tables):
        assert table.table.labels == tuple(range(len(_spec_column(root / "STO_align", scan, "Epoch"))))

    epochs = {scan: _spec_column(root / "STO_align", scan, "Epoch") for scan in (42, 43)}
    votes = [epochs[scan][label] for scan in (42, 43) for label in selected[scan]]
    ((key, epoch),) = read_trend_values((("grouped", payload),), "Epoch", source_root=project)
    assert key == "grouped" and epoch.status == "available"
    assert epoch.value == pytest.approx(float(np.mean(votes)), rel=1e-12, abs=0)
    assert (epoch.count, epoch.minimum, epoch.maximum) == (11, min(votes), max(votes))
    member_means = [np.mean([epochs[scan][label] for label in selected[scan]]) for scan in (42, 43)]
    whole_scans = np.mean(epochs[42] + epochs[43])
    assert abs(epoch.value - np.mean(member_means)) > 1e-3 and abs(epoch.value - whole_scans) > 1e-3

    ((_key, ctemp),) = read_trend_values((("grouped", payload),), "CTEMP", source_root=project)
    expected = [_spec_column(root / "STO_align", scan, "CTEMP")[label] for scan in (42, 43) for label in selected[scan]]
    assert (ctemp.status, ctemp.value, ctemp.count) == ("available", float(np.mean(expected)), 11)
    ((_key, nope),) = read_trend_values((("grouped", payload),), "NOPE", source_root=project)
    assert (nope.status, nope.code) == ("failed", "COLUMN_MISSING")
    ((_key, rootless),) = read_trend_values((("grouped", payload),), "CTEMP")
    assert (rootless.status, rootless.code) == ("unavailable", "SOURCE_ROOT_REQUIRED")


def test_nexus_rsm_result_through_its_saved_contributors(tmp_path):
    from tests.core.test_rsm_nexus_members import _write_rsm_nexus
    from xdart.gui.tools.rsm_values import (
        RSMFrameSelector, RSMNexusMemberForm, RSMToolFormV2, prepare_rsm_tool_v2, rsm_tool_preset,
    )
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.rsm_geometry_asset import (
        CANONICAL_RSM_GEOMETRY_LOCATOR, install_canonical_rsm_geometry_asset, rsm_geometry_asset_input,
    )
    from xrd_tools.analysis.rsm_operation import (
        RSMNormalizationMode, RSMNormalizationPolicy, run_rsm_operation_v2,
    )
    from xrd_tools.core.scan import SourceKind

    install_canonical_rsm_geometry_asset(project_root=tmp_path)
    nexus = _write_rsm_nexus(tmp_path / "member.nxs", ub=True)  # eta = linspace(42.34, 42.44, 5)
    preset = rsm_tool_preset()
    member = RSMNexusMemberForm(nexus, SourceKind.NEXUS_STACK, "entry", RSMFrameSelector(1, 4, 3),
                                preset.detector_shape, preset.motor_selectors)
    identity = RSMNormalizationPolicy(RSMNormalizationMode.IDENTITY, None, None, (0.0, 0.0, 0.0, 0.0))
    form = RSMToolFormV2(tmp_path, rsm_geometry_asset_input(CANONICAL_RSM_GEOMETRY_LOCATOR), (member,),
                         preset.conditioning, identity, (12, 12, 12), preset.chunk_size, preset.max_frame_bytes,
                         None, tmp_path / "nexus-trend.nexus")
    result = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    ((_key, eta),) = read_trend_values((("nexus", result.payload),), "eta", source_root=tmp_path)
    votes = np.linspace(42.34, 42.44, 5)[[1, 4]]
    assert (eta.status, eta.count) == ("available", 2)
    assert eta.value == pytest.approx(float(np.mean(votes)), rel=1e-12, abs=0)
    # a changed source after the result was saved leaves its trend x unavailable, never a silent re-read
    state = nexus.stat()
    os.utime(nexus, ns=(state.st_atime_ns, state.st_mtime_ns + 1_000_000_000))
    ((_key, changed),) = read_trend_values((("nexus", result.payload),), "eta", source_root=tmp_path)
    assert (changed.status, changed.code, changed.value) == ("unavailable", "MEMBER_METADATA_UNAVAILABLE", None)
