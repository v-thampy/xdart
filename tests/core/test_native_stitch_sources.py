"""Actual grouped finite execution and source identity, without GUI doubles."""

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceGroupReceipt
from xrd_tools.analysis.stitch_operation import prepare_stitch_operation, run_stitch_operation
from xrd_tools.analysis.xu_stitch_calibration import XuStitchCalibrationInput, capture_xu_stitch_calibration
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
from xrd_tools.sources.spec import SpecSource
from tests.core.test_stitch_operation import _prepared, _xu_prepared
from tests.core.test_spec_source import _spec


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_grouped_operation_uses_all_original_selected_frames_once(tmp_path, backend):
    prepare = _prepared if backend == "multigeometry" else _xu_prepared
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    first, second = prepare(tmp_path / "a"), prepare(tmp_path / "b")
    reference = run_stitch_operation(first)
    assert reference.terminal.disposition is ModuleDisposition.COMMITTED
    group = ModuleSourceGroupReceipt.from_members((first.module.source, second.module.source), kind=ModuleKind.STITCH)
    plan = first.plan
    if backend == "xu_hist":
        plan = replace(plan, calibration=capture_xu_stitch_calibration(
            XuStitchCalibrationInput(Path(first.plan.calibration.project_root) / first.plan.calibration.resolved_relative_path), project_root=tmp_path,
        ))
    request = prepare_stitch_operation(group,
        ModuleOutputRequest(tmp_path / "group.nexus", AnalysisArtifactKind.STITCH_1D), plan, project_root=tmp_path)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal.diagnostic
    assert request.manifest.member_frames == tuple(
        (ordinal, label) for ordinal, member in enumerate(group.members) for label in member.selected_labels)
    np.testing.assert_allclose(result.payload.intensity, reference.payload.intensity, equal_nan=True)
    np.testing.assert_allclose(result.payload.coverage, 2 * reference.payload.coverage)
    np.testing.assert_allclose(result.payload.normalization, 2 * reference.payload.normalization)


def test_strict_spec_refuses_partial_images_and_native_read_retains_saturation(tmp_path):
    spec = _spec(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    values = np.array([[0, np.iinfo(np.uint16).max], [2, 3]], dtype=np.uint16)
    for index in (0, 2):
        values.tofile(images / f"myscan_scan5_{index:04}.raw")
    options = dict(scan="5.1", image_dir=images, require_complete_images=True,
        read_image_kwargs={"detector_shape": (2, 2), "raw_dtype": "uint16", "threshold": 100})
    # the refusal says what it counted, and where (a stale image stem was the live cause)
    with pytest.raises(ValueError, match=r"scan 5\.1: 3 metadata points but 2 image frames in 2 files matching "
                                         r"\*myscan_scan5_\* in .*images; check the image stem and folder"):
        SpecSource(spec, **options)
    values.tofile(images / "myscan_scan5_0001.raw")
    source = SpecSource(spec, **options)
    native = source.load_native_frame(1)
    assert native.dtype == np.uint16
    assert native[0, 1] == 65535
    assert source.metadata_for(1)["th"] == 1
    empty = SpecSource(spec, scan="6.1", image_dir=images, require_complete_images=True)
    assert not empty.capabilities.has_raw_references


def test_native_selection_brackets_all_and_directory_mixed_formats(tmp_path):
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection, resolve_stitch_sources
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    spec = _spec(tmp_path)
    for index in range(3):
        np.ones((2, 2), dtype=np.uint16).tofile(tmp_path / f"myscan_scan5_{index}.raw")
    decode = {"detector_shape": (2, 2), "raw_dtype": "uint16"}
    selected = resolve_stitch_sources(StitchSourceSelection("Spec", (str(spec),), "5,[5,6]"), decode)
    assert len(selected.groups) == 2
    assert [member.options["scan"] for group in selected.groups for member in group] == ["5.1", "5.1"]
    # no Images folder: the SPEC file's own folder is searched, for the scan's automatic stem
    assert selected.skipped == ((str(spec), "6.1",
                                 f"No detector images named *{spec.stem}_scan6_* in {spec.parent}"),)
    all_spec = resolve_stitch_sources(StitchSourceSelection("Spec", (str(spec),), stitch_all=True), decode)
    assert len(all_spec.groups) == 1
    first = _write_bluesky_nxwriter(tmp_path / "a.nxs")
    second = _write_bluesky_nxwriter(tmp_path / "b.nxs")
    nexus = resolve_stitch_sources(StitchSourceSelection("NeXus", (str(first), str(second))), decode)
    assert len(nexus.groups) == 1 and len(nexus.groups[0]) == 2
    directory = resolve_stitch_sources(StitchSourceSelection("Directory", (str(tmp_path),)), decode)
    assert len(directory.groups) == 3
    assert directory.skipped == all_spec.skipped


def test_a_directory_takes_its_subfolders_and_a_name_filter_as_int_does(tmp_path, monkeypatch):
    """Maintainer, 2026-09-30: "Directory mode should have subfolders option as well as filters like in
    Int 2d". The SPEC files a Directory takes show in the scans it skips for want of images."""
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection, resolve_stitch_sources
    from xrd_tools.io import spec as spec_io
    header = ("#F {}\n#E 1\n#D today\n#O0 del  nu\n\n#S 1 ascan del 14 15 1 1\n#D today\n#P0 14 -10\n#N 2\n"
              "#L del  nu\n14 -10\n15 -10\n")
    for relative in ("R1C1", "R1C2", "LaB6_17keV", "sub/R2C1", "sub/deeper/R2C2_LaB6"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header.format(path.name))
    (tmp_path / "images").mkdir()
    for name in ("other_scan1_0000.raw", "other_scan1_0000.raw.pdi"):
        (tmp_path / "images" / name).write_bytes(b"#S not a SPEC file\n")
    decode = {"detector_shape": (2, 2), "raw_dtype": "uint16"}

    def taken(**options):
        resolution = resolve_stitch_sources(StitchSourceSelection("Directory", (str(tmp_path),), **options), decode)
        return sorted(Path(path).relative_to(tmp_path).as_posix() for path, _scan, _reason in resolution.skipped)

    assert taken() == ["LaB6_17keV", "R1C1", "R1C2"]  # the folder alone, as before
    assert taken(recursive=True) == ["LaB6_17keV", "R1C1", "R1C2", "sub/R2C1", "sub/deeper/R2C2_LaB6"]
    assert taken(recursive=True, name_filter="R1C | R2C -LaB6") == ["R1C1", "R1C2", "sub/R2C1"]
    assert taken(name_filter="lab6") == ["LaB6_17keV"]  # case-insensitive
    with pytest.raises(ValueError, match="^Filter: "):
        StitchSourceSelection("Directory", (str(tmp_path),), name_filter="R1C NOT")
    opened = []  # an image is never opened to learn whether it is a SPEC file, even in subfolders
    original = spec_io.is_spec_file
    monkeypatch.setattr(spec_io, "is_spec_file", lambda path: opened.append(Path(path).name) or original(path))
    taken(recursive=True)
    assert "images" not in {Path(name).parent.name for name in opened}
    assert not any(name.startswith("other_scan1") for name in opened) and "R2C1" in opened
    # a selection without the options keeps its identity (its form's fingerprint)
    assert StitchSourceSelection("Directory", (str(tmp_path),)).fingerprint_value == (
        "Directory", (str(tmp_path),), "", False, "", "")


@pytest.mark.parametrize("mode,alias", (("NeXus", False), ("Directory", False), ("Directory", True)))
def test_eiger_master_selection_keeps_identity_and_admits_metadata_once(tmp_path, mode, alias):
    from xrd_tools.analysis.scan_operations import AnalysisDisposition, MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection, resolve_stitch_sources
    from xrd_tools.core.scan import SourceKind
    from tests.core.test_source_edge_contract import _dangling_canonical, _land_canonical

    master, sidecar = _dangling_canonical(tmp_path)
    _land_canonical(sidecar)
    if alias:
        (tmp_path / "scan_alias_master.h5").symlink_to(master)
    selected_path = master if mode == "NeXus" else tmp_path
    selection = resolve_stitch_sources(StitchSourceSelection(mode, (str(selected_path),)), {})

    assert len(selection.groups) == 1
    assert len(selection.groups[0]) == 1
    source = selection.groups[0][0]
    assert Path(source.uri).resolve() == master
    table = run_metadata_table(MetadataTablePlan(source))
    assert table.disposition is AnalysisDisposition.COMPLETED, (table.code, table.diagnostics)
    assert source.kind is SourceKind.EIGER_MASTER
    assert source.entry == "entry"
    assert table.labels == (0, 1, 2)
    assert table.receipt.resolved_entry == "entry"


def test_nexus_metadata_preserves_external_dependency_and_native_dtype(tmp_path):
    import h5py
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection, resolve_stitch_sources
    from xrd_tools.analysis.scan_operations import (
        AnalysisDisposition, MetadataTablePlan, MetadataTableRequalificationPlan,
        run_metadata_table, run_metadata_table_requalification,
    )
    path = _write_bluesky_nxwriter(tmp_path / "angles.nxs")
    sidecar = tmp_path / "detector.h5"
    with h5py.File(path, "r+") as stream, h5py.File(sidecar, "w") as detector:
        image = np.asarray(stream["entry/data/eiger_image"])
        detector.create_dataset("frames", data=image)
        del stream["entry/data/eiger_image"]
        del stream["entry/instrument/detectors/eiger/data"]
        stream["entry/data/eiger_image"] = h5py.ExternalLink(sidecar.name, "/frames")
        detector["frames"].attrs["signal_type"] = "detector"
    selection = resolve_stitch_sources(StitchSourceSelection("NeXus", (str(path),)), {})
    spec = selection.groups[0][0]
    table = run_metadata_table(MetadataTablePlan(spec))
    assert table.disposition is AnalysisDisposition.COMPLETED, (table.code, table.diagnostics)
    assert table.labels == tuple(range(5))
    with h5py.File(sidecar, "r+") as stream:
        stream["frames"][0, 0, 0] = 1234
    repeated = run_metadata_table_requalification(MetadataTableRequalificationPlan(table.receipt, table.table_fingerprint))
    assert repeated.disposition is AnalysisDisposition.REFUSED


_UNGRIDDABLE = """
#S 7 ascan th 0 2 2 1
#D today
#P0 0 5 10
#N 4
#L th  Epoch  i0  det
#C aborted before its first point

#S 8 ascan th 0 2 2 1
#D today
#P0 0 5 10
#N 4
#L th  Epoch  i0  det
#C aborted after its first image

#S 9 ascan th 0 2 2 1
#D today
#P0 0 5 10
#N 4
#L th  Epoch  i0  det
0 1 100 10
1 2 110 20
2 3 120 30

#S 10 ascan th 0 2 2 1
#D today
#P0 0 5 10
#N 4
#L th  Epoch  i0  det
0 1 100 10
1 2 110 20
2 3 120 30
"""


def _ungriddable_spec(tmp_path):
    """Scan 5 complete (three points, three images); 6 without images; 7 aborted without images; 8 aborted with its
    first image; 9 stopped part-way (three points, two images); 10 with a numbering gap (images 0, 1, 3)."""
    spec = _spec(tmp_path)
    spec.write_text(spec.read_text() + _UNGRIDDABLE)
    for scan, numbers in ((5, (0, 1, 2)), (8, (0,)), (9, (0, 1)), (10, (0, 1, 3))):
        for index in numbers:
            np.ones((2, 2), dtype=np.uint16).tofile(tmp_path / f"myscan_scan{scan}_{index:04}.raw")
    return spec


def _resolved_or_refusal(selection):
    from xrd_tools.analysis.stitch_sources import resolve_stitch_sources
    try:
        return resolve_stitch_sources(selection, {"detector_shape": (2, 2), "raw_dtype": "uint16"})
    except ValueError as error:
        return f"refused the whole Source: {error}"


@pytest.mark.parametrize("mode", ("Spec", "Directory"))
def test_all_scans_skip_every_scan_they_cannot_grid_and_say_why(tmp_path, mode):
    """NEXT-1210-RESPONSIVENESS C-10. The maintainer, 2026-11-12, after All refused his SPEC file with "SPEC #L
    labels do not match the physical data table": "in general there will be a lot of scans in a spec file which
    will not have detector data and should be skipped", and his pick "Also mismatched scans". All Scans (Stitch All,
    RSM All) and Directory grid the complete scans and skip, each with its reason, a scan with no detector images,
    no data points, an image count different from its points, or a numbering gap."""
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection
    spec = _ungriddable_spec(tmp_path)
    selection = (StitchSourceSelection("Spec", (str(spec),), stitch_all=True) if mode == "Spec"
                 else StitchSourceSelection("Directory", (str(tmp_path),)))
    resolved = _resolved_or_refusal(selection)
    assert not isinstance(resolved, str), resolved
    assert [[member.options["scan"] for member in group] for group in resolved.groups] == [["5.1"]]
    reasons = {key: reason for _path, key, reason in resolved.skipped}
    assert list(reasons) == ["6.1", "7.1", "8.1", "9.1", "10.1"]
    assert reasons["6.1"] == f"No detector images named *myscan_scan6_* in {tmp_path}"
    assert reasons["7.1"] == "no data points (an aborted scan)"
    assert reasons["8.1"] == "no data points (an aborted scan)"
    assert reasons["9.1"] == "3 scan points but 2 detector images"
    assert reasons["10.1"] == "detector image numbers have gaps or duplicates"


@pytest.mark.parametrize(("scans", "refusal"), (
    ("5,8", r"scan 8\.1: no data points \(an aborted scan\)"),
    ("5,9", r"scan 9\.1: 3 metadata points but 2 image frames in 2 files matching "),
    ("5,10", r"scan 10\.1: image point numbers contain gaps or duplicates"),
))
def test_an_explicitly_selected_scan_that_cannot_be_gridded_is_still_refused(tmp_path, scans, refusal):
    """C-10's other half: Select Scans naming such a scan refuses the Source, by that scan. An aborted scan's refusal
    now says it has no data points (1.2.9 said "SPEC #L labels do not match the physical data table")."""
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection
    resolved = _resolved_or_refusal(StitchSourceSelection("Spec", (str(_ungriddable_spec(tmp_path)),), scans))
    assert isinstance(resolved, str), resolved
    assert re.search(refusal, resolved), resolved
