"""Actual finite native operations, committed artifacts and shared raw readers."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from xrd_tools.analysis.module_transaction import (
    ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceGroupReceipt,
    ModuleArtifactRefused, module_artifact_request,
)
from xrd_tools.analysis.stitch_operation import (
    prepare_stitch_operation, run_stitch_operation, StitchOperationRefused,
)
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
from xrd_tools.io.frame_view import read_contributing_frame_view
from xrd_tools.io.output_path import resolve_finite_output_target
from tests.core.test_stitch_operation import _prepared, _xu_prepared


def _native(tmp_path, backend, unit, auto=True, *, grouped=True):
    original = _prepared(tmp_path) if backend == "multigeometry" else _xu_prepared(tmp_path)
    source = original.module.source
    if grouped:
        second = tmp_path / "second"
        second.mkdir()
        other = _prepared(second) if backend == "multigeometry" else _xu_prepared(second)
        source = ModuleSourceGroupReceipt.from_members((source, other.module.source), kind=ModuleKind.STITCH)
    mask = np.zeros((195, 487 if backend == "multigeometry" else 1475), dtype=np.uint8)
    mask[20:35] = 1
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)
    kwargs = dict(native_records=True, unit=unit, radial_range=None if auto else
                  ((.1, 6.) if unit == "q_A^-1" else (1., 60.)),
                  conditioning=StitchConditioning("auto"), mask_path=mask_path)
    if backend == "xu_hist":
        kwargs.update(q_min_A_inverse=None, q_max_A_inverse=None)
    plan = replace(original.plan, **kwargs)
    target = resolve_finite_output_target(tmp_path, "native", operation_token="stitch-1d")
    output = ModuleOutputRequest(target, AnalysisArtifactKind.STITCH_1D, artifact_family="native")
    return prepare_stitch_operation(source, output, plan, project_root=tmp_path)


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
@pytest.mark.parametrize("unit", ["q_A^-1", "2th_deg"])
@pytest.mark.parametrize("auto", [True, False])
def test_native_group_science_persists_dual_axes_and_exact_raw_members(tmp_path, backend, unit, auto):
    request = _native(tmp_path, backend, unit, auto)
    preview = request.provenance
    assert preview["plan"]["radial_range"] is None if auto else preview["plan"]["radial_range"] is not None
    assert preview["plan"]["conditioning"] == {"mode": "auto", "threshold_min": None, "threshold_max": None}
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    payload = read_analysis_artifact(request.module.output.target)
    assert payload.schema_version == (7 if backend == "xu_hist" else 6)
    assert payload.native_unit == unit
    companion = payload.radial_axis("2th_deg" if unit == "q_A^-1" else "q_A^-1")
    assert not np.allclose(np.diff(companion), np.diff(companion)[0], rtol=1e-5)
    assert len(payload.contributing_frames) == len(request.module.source.selected_labels)
    assert len({record["identity"]["source_id"] for record in payload.contributing_frames}) == 2
    for row in payload.contributing_frames:
        scan = row["key"].split("/")[0][5:]
        frame = read_contributing_frame_view(request.module.output.target,
            row["identity"]["frame_index"], scan=scan, full_raw=True, artifact_payload=payload)
        expected = tifffile.imread(tmp_path / row["source_path"])
        np.testing.assert_array_equal(frame.raw, expected)
        assert frame.thumbnail is not None
    assert (payload.execution_attestation_digest is not None) == (backend == "xu_hist")


def test_native_mask_change_after_preview_refuses_before_publication(tmp_path):
    request = _native(tmp_path, "multigeometry", "q_A^-1", grouped=False)
    np.save(request.plan.mask_path, np.ones((195, 487), dtype=np.uint8))
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "MASK_IDENTITY_MISMATCH"
    assert not Path(request.module.output.target).exists()


def test_native_intent_rejects_changed_unit_records_and_output_family(tmp_path):
    request = _native(tmp_path, "xu_hist", "q_A^-1", grouped=False)
    for mutate in (
        lambda value: value["plan"].update(unit="2th_deg"),
        lambda value: value["frame_members"].clear(),
        lambda value: value["output"].update(artifact_family="changed"),
    ):
        value = json.loads(request.provenance_json)
        mutate(value)
        with pytest.raises(ModuleArtifactRefused, match="NATIVE_STITCH_INTENT_MISMATCH"):
            module_artifact_request(request.module, value)


def test_native_real_scan14_five_member_preview_keeps_full_members_with_compact_binding(tmp_path):
    from tests.core.test_stitch_operation_real_data import _scan14_request, _stitching_root
    root = _stitching_root()
    prepared = [_scan14_request(root, tmp_path / f"member-{omitted}.nexus",
        tuple(label for label in range(287) if label != omitted)) for omitted in range(5)]
    source = ModuleSourceGroupReceipt.from_members(tuple(item.module.source for item in prepared),
        kind=ModuleKind.STITCH)
    plan = replace(prepared[0].plan, native_records=True, conditioning=StitchConditioning("auto"))
    output = ModuleOutputRequest(resolve_finite_output_target(tmp_path, "five-members", operation_token="stitch-1d"),
        AnalysisArtifactKind.STITCH_1D, artifact_family="five-members")
    request = prepare_stitch_operation(source, output, plan, project_root=root)
    assert source.selected_frame_count == 1430
    assert len(request.provenance_json.encode()) < 1 << 20
    assert request.provenance["frame_members"]["count"] == 1430
    from xrd_tools.analysis.stitch_native import native_frame_members
    members = native_frame_members(request)
    assert len(members) == 1430
    assert len({row["identity"]["source_id"] for row in members}) == 5
    assert len({row["identity"]["contribution_index"] for row in members}) == 1430
    for index in range(5):
        assert [row["identity"]["frame_index"] for row in members[index * 286:(index + 1) * 286]] == [
            label for label in range(287) if label != index]
    assert not Path(output.target).exists()


def test_native_nexus_group_keeps_exact_external_dataset_and_raw_dtype(tmp_path):
    import h5py
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    from tests.core.test_stitch_operation import _goniometer_record
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table, AnalysisDisposition
    from xrd_tools.analysis.module_transaction import ModuleSourceReceipt, MetadataColumnSelector
    from xrd_tools.analysis.stitch_operation import (
        StitchOperationPlan, StitchGeometryInput, StitchGeometryKind, capture_stitch_geometry,
    )
    from xrd_tools.core.scan import SourceSpec, SourceKind
    members = []
    originals = []
    for index in range(2):
        path = _write_bluesky_nxwriter(tmp_path / f"member{index}.nxs")
        with h5py.File(path, "r+") as stream:
            stream.move("entry", "sample")
            stream.attrs["default"] = "sample"
            cfg = stream["sample/instrument/bluesky/metadata/configuration/eiger/data"]
            cfg["eiger_cam_wavelength"][()] = 1.0
            cfg["eiger_cam_photon_energy"][()] = 12398.41984
            stream["sample/data"].create_dataset("del", data=np.linspace(5., 10., 5))
            stream["sample/data"].create_dataset("nu", data=np.zeros(5))
            for name in ("del", "nu"):
                positioner = stream["sample/instrument/positioners"].create_group(name)
                positioner.attrs["NX_class"] = "NXpositioner"
                positioner.create_dataset("value", data=stream[f"sample/data/{name}"][()])
            del stream["sample/data/eiger_image"]
            del stream["sample/instrument/detectors/eiger/data"]
            detector_data = stream["sample/data"].create_dataset("eiger_image",
                data=np.arange(5 * 195 * 487, dtype=np.uint32).reshape(5, 195, 487))
            detector_data.attrs["signal_type"] = "detector"
            stream["sample/instrument/detectors/eiger/data"] = detector_data
            originals.append(np.asarray(stream["sample/data/eiger_image"]))
            if index == 1:
                with h5py.File(tmp_path / "detector.h5", "w") as detector:
                    detector.create_dataset("frames", data=originals[-1])
                    detector["frames"].attrs["signal_type"] = "detector"
                del stream["sample/data/eiger_image"]
                del stream["sample/instrument/detectors/eiger/data"]
                stream["sample/data/eiger_image"] = h5py.ExternalLink("detector.h5", "/frames")
        table = run_metadata_table(MetadataTablePlan(SourceSpec(path, SourceKind.NEXUS_STACK, entry="sample")))
        assert table.disposition is AnalysisDisposition.COMPLETED, table
        members.append(ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.STITCH,
            selected_labels=(0, 4), resolved_selectors=(MetadataColumnSelector("del"), MetadataColumnSelector("nu"))))
    geometry = _goniometer_record()
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(geometry))
    plan = StitchOperationPlan(capture_stitch_geometry(StitchGeometryInput(path,
        StitchGeometryKind.PYFAI_GONIOMETER_JSON, source_motors=(("del_angle", "del"), ("nu_angle", "nu")))),
        native_records=True, unit="2th_deg", npt_1d=24, conditioning=StitchConditioning("auto"))
    output = ModuleOutputRequest(resolve_finite_output_target(tmp_path, "nexus-group", operation_token="stitch-1d"),
        AnalysisArtifactKind.STITCH_1D, artifact_family="nexus-group")
    source = ModuleSourceGroupReceipt.from_members(tuple(members), kind=ModuleKind.STITCH)
    request = prepare_stitch_operation(source, output, plan, project_root=tmp_path)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    from xrd_tools.analysis.stitch_metadata import read_stitch_metadata
    tables = read_stitch_metadata(result.payload)
    assert len(tables) == 2
    for table in tables:
        assert table.table.disposition is AnalysisDisposition.COMPLETED, table
        assert table.selected_labels == (0, 4)
        assert {column.name for column in table.table.columns} >= {"del", "nu"}
        np.testing.assert_array_equal(next(column.numeric for column in table.table.columns
            if column.name == "del"), np.linspace(5., 10., 5))
    for index, row in enumerate(result.payload.contributing_frames):
        assert row["source_snapshot"]["dataset_path"] == "/sample/data/eiger_image"
        assert np.dtype(row["source_snapshot"]["raw_dtype"]) == np.dtype("uint32")
        frame = read_contributing_frame_view(output.target, row["identity"]["frame_index"],
            scan=row["key"].split("/")[0][5:], full_raw=True, artifact_payload=result.payload)
        np.testing.assert_array_equal(frame.raw, originals[index // 2][(0, 4)[index % 2]])
    # A changed external detector segment must not inherit the unchanged master
    # stamp. The same saved result still supplies its original thumbnail/curve.
    import os
    dependency = tmp_path / "detector.h5"
    state = dependency.stat()
    with h5py.File(dependency, "r+") as stream:
        stream["frames"][0] = originals[1][0] + 50
    os.utime(dependency, ns=(state.st_atime_ns, state.st_mtime_ns + 1_000_000))
    row = result.payload.contributing_frames[2]
    changed = read_contributing_frame_view(output.target, row["identity"]["frame_index"],
        scan=row["key"].split("/")[0][5:], full_raw=True, artifact_payload=result.payload)
    assert changed.raw is None
    assert changed.thumbnail is not None
    assert changed.extra["raw_unavailable_reason"] == "raw source changed since processing"
    np.testing.assert_array_equal(read_analysis_artifact(output.target).intensity, result.payload.intensity)


def test_native_rejects_energy_conflict_in_later_group_member(tmp_path):
    original = _xu_prepared(tmp_path)
    other = _xu_prepared(tmp_path / "second", energy_eV=None)
    # Build a fresh frozen member with conflicting energy, rather than drift its receipt.
    text_path = tmp_path / "second/xu-source/scan_0001.txt"
    text_path.write_text(text_path.read_text().replace("I0 = 2", "I0 = 2, energy = 18000"))
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.module_transaction import ModuleSourceReceipt
    table = run_metadata_table(MetadataTablePlan(other.module.source.analysis.source_spec))
    member = ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.STITCH,
        selected_labels=other.module.source.selected_labels, resolved_selectors=other.module.source.resolved_selectors)
    group = ModuleSourceGroupReceipt.from_members((original.module.source, member), kind=ModuleKind.STITCH)
    plan = replace(original.plan, native_records=True)
    output = ModuleOutputRequest(resolve_finite_output_target(tmp_path, "conflict", operation_token="stitch-1d"),
        AnalysisArtifactKind.STITCH_1D, artifact_family="conflict")
    with pytest.raises(StitchOperationRefused, match="selected member energy"):
        prepare_stitch_operation(group, output, plan, project_root=tmp_path)


@pytest.mark.parametrize("rotation", [0, 180])
def test_authored_xu_operation_uses_fitted_geometry_mapping_and_orientation(tmp_path, rotation):
    from pyFAI.detectors import detector_factory
    from xrd_tools.analysis.xu_stitch_calibration import (
        authored_surface_calibration, capture_xu_stitch_calibration, XuStitchCalibrationInput,
    )
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.module_transaction import ModuleSourceReceipt, MetadataColumnSelector
    from xrd_tools.analysis.stitch_operation import XuStitchOperationPlan
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
    original = _xu_prepared(tmp_path)
    image_path = tmp_path / "xu-source/scan_0001.tif"
    raw = np.arange(195 * 1475, dtype=np.uint16).reshape(195, 1475)
    tifffile.imwrite(image_path, raw)
    text_path = image_path.with_suffix(".txt")
    text_path.write_text(text_path.read_text().replace("del =", "gamma =").replace("nu =", "delta ="))
    table = run_metadata_table(MetadataTablePlan(original.module.source.analysis.source_spec))
    source = ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.STITCH,
        selected_labels=(1,), resolved_selectors=tuple(MetadataColumnSelector(name) for name in ("gamma", "delta")))
    with xu_runtime_session() as runtime:
        asset = authored_surface_calibration(detector=detector_factory("Pilatus300kw"),
            cch1=220., cch2=235., distance_m=.395, del_offset_deg=3., nu_offset_deg=1.,
            energy_eV=17000.018, xu_wavelength_A=float(runtime.xu.en2lam(17000.018)),
            source_motors={"del": "gamma", "nu": "delta"}, rotation=rotation,
            control_domain={"del": [5., 40.], "nu": [-10., 10.]},
            domain={"del": [0., 50.], "nu": [-15., 30.]}, rms_q=.001, condition=100., n_frames=4, n_points=200)
    asset_path = tmp_path / "authored.json"
    asset_path.write_text(json.dumps(asset, sort_keys=True, separators=(",", ":")))
    receipt = capture_xu_stitch_calibration(XuStitchCalibrationInput(asset_path), project_root=tmp_path)
    with xu_runtime_session() as runtime:
        effective = resolve_xu_stitch_effective_geometry(receipt, runtime).projection
    assert effective.asset_version == 2
    assert effective.solid_angle_sha256 != original.plan.effective_geometry.solid_angle_sha256
    with pytest.raises(ValueError, match="canonical version 1"):
        XuStitchOperationPlan(receipt, effective, 1.0, 5.2, npt_1d=64)
    plan = XuStitchOperationPlan(receipt, effective, native_records=True, unit="2th_deg", npt_1d=64,
                                conditioning=StitchConditioning("auto"))
    output = ModuleOutputRequest(resolve_finite_output_target(tmp_path, "authored", operation_token="stitch-1d"),
        AnalysisArtifactKind.STITCH_1D, artifact_family="authored")
    request = prepare_stitch_operation(source, output, plan, project_root=tmp_path)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    assert result.payload.schema_version == 7
    row = result.payload.contributing_frames[0]
    assert row["source_snapshot"]["rotation"] == rotation
    frame = read_contributing_frame_view(output.target, 1, scan=row["key"].split("/")[0][5:],
        full_raw=True, artifact_payload=result.payload)
    np.testing.assert_array_equal(frame.raw, np.rot90(raw, rotation // 90))
