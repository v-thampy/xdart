"""Actual grouped finite execution and source identity, without GUI doubles."""

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
    with pytest.raises(ValueError, match="image count"):
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
    assert selected.skipped == ((str(spec), "6.1", "No detector images"),)
    all_spec = resolve_stitch_sources(StitchSourceSelection("Spec", (str(spec),), stitch_all=True), decode)
    assert len(all_spec.groups) == 1
    first = _write_bluesky_nxwriter(tmp_path / "a.nxs")
    second = _write_bluesky_nxwriter(tmp_path / "b.nxs")
    nexus = resolve_stitch_sources(StitchSourceSelection("NeXus", (str(first), str(second))), decode)
    assert len(nexus.groups) == 1 and len(nexus.groups[0]) == 2
    directory = resolve_stitch_sources(StitchSourceSelection("Directory", (str(tmp_path),)), decode)
    assert len(directory.groups) == 3
    assert directory.skipped == all_spec.skipped


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
