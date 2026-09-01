from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pytest
import tifffile
import xrd_tools.analysis.stitch_operation as stitch_operation

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleDisposition,
    ModuleKind,
    ModuleOutputRequest,
    ModuleSourceReceipt,
    ModuleOperationRequest,
    module_artifact_request,
    module_provenance_digest,
)
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    MetadataTablePlan,
    run_metadata_table,
)
from xrd_tools.analysis.stitch_operation import (
    StitchGeometryInput,
    StitchGeometryKind,
    StitchInputManifestReceipt,
    StitchManifestFile,
    StitchContribution,
    StitchOperationPlan,
    XuStitchOperationPlan,
    StitchOperationCleanupPending,
    StitchOperationRefused,
    StitchOperationVerificationError,
    capture_stitch_geometry,
    prepare_stitch_operation,
    run_stitch_operation,
)
from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationInput,
    canonical_surface_resource_bytes,
    capture_xu_stitch_calibration,
)
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
)
from xrd_tools.sources.selection import image_series_spec


def _goniometer_record() -> dict[str, object]:
    deg = float(np.deg2rad(1.0))
    names = [
        "dist",
        "poni1",
        "poni2",
        "rot1_offset",
        "rot1_scale",
        "rot2_offset",
        "rot2_scale",
    ]
    positions = ["del_angle", "nu_angle"]
    return {
        "content": "Goniometer calibration v2",
        "detector": "Pilatus100k",
        "detector_config": {"orientation": 3},
        "wavelength": 1.0e-10,
        "param": [0.2, 0.01677, 0.04188, 0.0, deg, 0.0, deg],
        "param_names": names,
        "pos_names": positions,
        "trans_function": {
            "content": "GeometryTransformation",
            "param_names": names,
            "pos_names": positions,
            "dist_expr": "dist",
            "poni1_expr": "poni1",
            "poni2_expr": "poni2",
            "rot1_expr": "rot1_scale * nu_angle + rot1_offset",
            "rot2_expr": "rot2_scale * del_angle + rot2_offset",
            "rot3_expr": "0.0",
            "constants": {"pi": float(np.pi)},
        },
    }


def _poni_record() -> str:
    return """\
poni_version: 2.1
Detector: Pilatus100k
Detector_config: {"orientation": 3}
Distance: 0.2
Poni1: 0.01677
Poni2: 0.04188
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1e-10
"""


def _v3_poni_record() -> str:
    return """\
poni_version: 3.0
Detector: Detector
Detector_config: {"pixel1":0.000172,"pixel2":0.000172,"max_shape":[195,487],"orientation":3,"sensor":{"material":"Si","thickness":0.00045}}
Distance: 0.2
Poni1: 0.01677
Poni2: 0.04188
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1e-10
Parallax: True
"""


def _ring(seed: int) -> np.ndarray:
    shape = (195, 487)
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]]
    radius = np.sqrt((y - shape[0] / 2.0) ** 2 + (x - shape[1] / 2.0) ** 2)
    return (
        500.0 * np.exp(-((radius - 60.0) / 10.0) ** 2)
        + rng.poisson(3, size=shape)
    ).astype(np.uint16)


def _source_receipt(tmp_path: Path, *, selected=(1, 3)) -> ModuleSourceReceipt:
    root = tmp_path / "source"
    root.mkdir()
    for index in range(3):
        image = root / f"scan_{index + 1:04d}.tif"
        tifffile.imwrite(image, _ring(index))
        image.with_suffix(".txt").write_text(
            "# Counters\n"
            f"I0 = {10 + index}\n"
            "# Motors\n"
            f"del = {5 * index}, nu = {index}\n"
            "User: stitch, time: Mon Jan 15 10:30:00 2024  # Temp\n",
            encoding="utf-8",
        )
    table = run_metadata_table(
        MetadataTablePlan(
            image_series_spec(root / "scan_0001.tif", metadata_format="txt")
        )
    )
    assert table.disposition is AnalysisDisposition.COMPLETED
    selectors = (
        MetadataColumnSelector("I0"),
        MetadataColumnSelector("del"),
        MetadataColumnSelector("nu"),
    )
    return ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.STITCH,
        selected_labels=selected,
        resolved_selectors=selectors,
    )


def _prepared(
    tmp_path: Path,
    *,
    mode="1d",
    max_frame_bytes=64 * 1024 * 1024,
):
    source = _source_receipt(tmp_path)
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(_goniometer_record()), encoding="utf-8")
    digest = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    geometry = capture_stitch_geometry(
        StitchGeometryInput(
            geometry_path,
            StitchGeometryKind.PYFAI_GONIOMETER_JSON,
            expected_sha256=digest,
            source_motors=(("del_angle", "del"), ("nu_angle", "nu")),
        )
    )
    plan = StitchOperationPlan(
        geometry,
        mode=mode,
        npt_1d=96,
        npt_rad_2d=48,
        npt_azim_2d=32,
        radial_range=(0.1, 6.0),
        azimuth_range=(-90.0, 90.0) if mode == "2d" else None,
        monitor_selector=MetadataColumnSelector("I0"),
        max_frame_bytes=max_frame_bytes,
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    output = ModuleOutputRequest(
        output_root / "stitched.nexus",
        (
            AnalysisArtifactKind.STITCH_1D
            if mode == "1d"
            else AnalysisArtifactKind.STITCH_2D
        ),
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    return prepare_stitch_operation(
        source,
        output,
        plan,
        project_root=tmp_path,
    )


def _xu_prepared(
    tmp_path: Path,
    *,
    del_value: float = 14.0,
    nu_value: float = -10.0,
    energy_eV: float | None = 17000.018,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "xu-source"
    root.mkdir()
    image = root / "scan_0001.tif"
    tifffile.imwrite(image, np.ones((195, 1475), dtype=np.uint16))
    counter_line = (
        "I0 = 2\n"
        if energy_eV is None
        else f"I0 = 2, energy = {energy_eV}\n"
    )
    image.with_suffix(".txt").write_text(
        "# Counters\n"
        f"{counter_line}"
        "# Motors\n"
        f"del = {del_value}, nu = {nu_value}\n"
        "User: stitch, time: Mon Jan 15 10:30:00 2024  # Temp\n",
        encoding="utf-8",
    )
    table = run_metadata_table(
        MetadataTablePlan(
            image_series_spec(image, metadata_format="txt")
        )
    )
    assert table.disposition is AnalysisDisposition.COMPLETED
    selectors = [
        MetadataColumnSelector("I0"),
        MetadataColumnSelector("del"),
        MetadataColumnSelector("nu"),
    ]
    if energy_eV is not None:
        selectors.append(MetadataColumnSelector("energy"))
    source = ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.STITCH,
        selected_labels=(1,),
        resolved_selectors=tuple(selectors),
    )
    asset_path = tmp_path / "calibration" / "xu" / "surface.json"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(canonical_surface_resource_bytes())
    calibration = capture_xu_stitch_calibration(
        XuStitchCalibrationInput("calibration/xu/surface.json"),
        project_root=tmp_path,
    )
    owner = xu_runtime_session()
    with owner as session:
        effective = resolve_xu_stitch_effective_geometry(
            calibration,
            session,
        ).projection
    assert owner.execution_record is not None
    plan = XuStitchOperationPlan(
        calibration,
        effective,
        1.0,
        5.2,
        npt_1d=64,
        monitor_selector=MetadataColumnSelector("I0"),
        max_frame_bytes=4 * 1024 * 1024,
    )
    output_root = tmp_path / "xu-output"
    output_root.mkdir()
    output = ModuleOutputRequest(
        output_root / "stitched.nexus",
        AnalysisArtifactKind.STITCH_1D,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    return prepare_stitch_operation(
        source,
        output,
        plan,
        project_root=tmp_path,
    )


def test_geometry_capture_is_hash_bound_and_rejects_duplicate_json(tmp_path):
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(_goniometer_record()), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = capture_stitch_geometry(
        StitchGeometryInput(
            path,
            StitchGeometryKind.PYFAI_GONIOMETER_JSON,
            expected_sha256=digest,
            source_motors=(("del_angle", "del"), ("nu_angle", "nu")),
        )
    )
    assert receipt.sha256 == digest
    assert receipt.content == path.read_bytes()

    with pytest.raises(StitchOperationRefused) as mismatch:
        capture_stitch_geometry(
            StitchGeometryInput(
                path,
                StitchGeometryKind.PYFAI_GONIOMETER_JSON,
                expected_sha256="0" * 64,
                source_motors=(("del_angle", "del"), ("nu_angle", "nu")),
            )
        )
    assert mismatch.value.code == "GEOMETRY_HASH_MISMATCH"

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"detector":"Pilatus100k","detector":"Pilatus100k"}', encoding="utf-8")
    with pytest.raises(StitchOperationRefused) as invalid:
        capture_stitch_geometry(
            StitchGeometryInput(
                duplicate,
                StitchGeometryKind.PYFAI_GONIOMETER_JSON,
                source_motors=(("del_angle", "del"),),
            )
        )
    assert invalid.value.code == "GEOMETRY_PARSE_FAILED"

    nonfinite = tmp_path / "nonfinite.json"
    record = _goniometer_record()
    record["wavelength"] = float("inf")
    nonfinite.write_text(
        json.dumps(record).replace("Infinity", "1e999"), encoding="utf-8"
    )
    with pytest.raises(StitchOperationRefused) as invalid_number:
        capture_stitch_geometry(
            StitchGeometryInput(
                nonfinite,
                StitchGeometryKind.PYFAI_GONIOMETER_JSON,
                source_motors=(("del_angle", "del"), ("nu_angle", "nu")),
            )
        )
    assert invalid_number.value.code == "GEOMETRY_PARSE_FAILED"


@pytest.mark.parametrize(
    "source_motors",
    (
        (("del_angle", "del"),),
        (
            ("del_angle", "del"),
            ("nu_angle", "nu"),
            ("other", "eta"),
        ),
    ),
)
def test_json_geometry_requires_exact_declared_position_mapping(
    tmp_path,
    source_motors,
):
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(_goniometer_record()), encoding="utf-8")

    with pytest.raises(StitchOperationRefused) as refused:
        capture_stitch_geometry(
            StitchGeometryInput(
                path,
                StitchGeometryKind.PYFAI_GONIOMETER_JSON,
                source_motors=source_motors,
            )
        )
    assert refused.value.code == "GEOMETRY_MOTOR_MAPPING_MISMATCH"


@pytest.mark.parametrize(
    "positions",
    (
        ["del_angle", "del_angle"],
        ["del_angle", ""],
        [],
    ),
)
def test_json_geometry_rejects_invalid_declared_positions(tmp_path, positions):
    record = _goniometer_record()
    record["pos_names"] = positions
    record["trans_function"]["pos_names"] = positions
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(StitchOperationRefused) as refused:
        capture_stitch_geometry(
            StitchGeometryInput(
                path,
                StitchGeometryKind.PYFAI_GONIOMETER_JSON,
                source_motors=(
                    ("del_angle", "del"),
                    ("nu_angle", "nu"),
                ),
            )
        )
    assert refused.value.code == "GEOMETRY_PARSE_FAILED"


def test_poni_capture_requires_one_strict_configured_record(tmp_path):
    record = _poni_record()
    path = tmp_path / "geometry.poni"
    path.write_text(record, encoding="utf-8")
    receipt = capture_stitch_geometry(
        StitchGeometryInput(
            path,
            StitchGeometryKind.PONI,
            source_motors=(("del", "del"), ("nu", "nu")),
            reference_motor_positions=(("del", 10.0), ("nu", -2.0)),
        )
    )
    assert receipt.sha256 == hashlib.sha256(record.encode()).hexdigest()
    diffractometer, _orientation = stitch_operation._runtime_geometry(
        receipt, receipt.content
    )
    np.testing.assert_allclose(
        np.deg2rad(diffractometer.rot1.apply(np.asarray([-2.0, 1.0]))),
        np.deg2rad([0.0, 3.0]),
    )
    np.testing.assert_allclose(
        np.deg2rad(diffractometer.rot2.apply(np.asarray([10.0, 12.0]))),
        np.deg2rad([0.0, 2.0]),
    )

    with pytest.raises(ValueError, match="source motor mapping"):
        StitchGeometryInput(path, StitchGeometryKind.PONI)

    appended = tmp_path / "appended.poni"
    appended.write_text(record + record, encoding="utf-8")
    with pytest.raises(StitchOperationRefused) as invalid:
        capture_stitch_geometry(
            StitchGeometryInput(
                appended,
                StitchGeometryKind.PONI,
                source_motors=(("del", "del"), ("nu", "nu")),
                reference_motor_positions=(("del", 0.0), ("nu", 0.0)),
            )
        )
    assert invalid.value.code == "GEOMETRY_PARSE_FAILED"


def test_prepare_binds_mode_source_selectors_and_provenance(tmp_path):
    request = _prepared(tmp_path)
    provenance = request.provenance
    assert request.module.plan_fingerprint == request.plan.fingerprint
    assert provenance["schema_version"] == "stitch-operation-v1"
    assert provenance["source"]["selected_labels"] == [1, 3]
    assert provenance["geometry"]["source_motors"] == [
        ["del_angle", "del"],
        ["nu_angle", "nu"],
    ]
    assert provenance["plan"]["monitor_selector"] == {
        "name": "I0",
        "occurrence": 0,
    }
    manifest = request.manifest
    assert manifest.source_relative_path == "source"
    assert manifest.files[0].relative_path == "source"
    assert [
        manifest.files[item.file_ordinal].relative_path
        for item in manifest.contributions
    ] == [
        "source/scan_0001.tif",
        "source/scan_0003.tif",
    ]
    options = json.loads(manifest.source_options_json)
    assert options["selected_file"] == "source/scan_0001.tif"
    assert options["files"] == [
        "source/scan_0001.tif",
        "source/scan_0002.tif",
        "source/scan_0003.tif",
    ]
    assert [item.label for item in manifest.contributions] == [1, 3]
    assert {
        item.relative_path for item in manifest.files
    } == {
        "source",
        "source/scan_0001.tif",
        "source/scan_0001.txt",
        "source/scan_0002.tif",
        "source/scan_0002.txt",
        "source/scan_0003.tif",
        "source/scan_0003.txt",
    }
    assert provenance["source"]["input_manifest"] == manifest.to_provenance()
    assert "project_root" not in manifest.to_provenance()

    def absolute_strings(value):
        if type(value) is str:
            return [value] if Path(value).is_absolute() else []
        if type(value) is dict:
            return [item for child in value.values() for item in absolute_strings(child)]
        if type(value) is list:
            return [item for child in value for item in absolute_strings(child)]
        return []

    assert absolute_strings(manifest.to_provenance()) == []
    assert provenance["geometry"]["relative_path"] == "geometry.json"


def test_xu_prepare_binds_exact_intent_and_commits_v2_artifact(tmp_path):
    request = _xu_prepared(tmp_path)
    assert type(request.plan) is XuStitchOperationPlan
    assert request.plan.backend == "xu_hist"
    assert request.module._xu_stitch_v2_bound is True
    assert request.provenance["schema_version"] == (
        "stitch-operation-v2-xu-intent"
    )
    assert request.provenance["output"] == {
        "target": request.module.output.target,
        "kind": request.module.output.kind.value,
        "overwrite": request.module.output.overwrite.value,
        "output_fingerprint": request.module.output.fingerprint,
    }
    assert request.provenance["observations"]["source_energy_range_eV"] == [
        17000.018,
        17000.018,
    ]

    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert result.terminal.request is request.module
    assert result.terminal.commit.request is request.module
    assert result.payload.schema_version == 2
    assert result.payload.execution_attestation_digest == (
        result.terminal.commit.execution_attestation_digest
    )
    attestation = json.loads(result.payload.execution_attestation_json)
    assert attestation["module_request_fingerprint"] == request.module.fingerprint
    assert attestation["selected_frame_count"] == 1
    assert attestation["release_check_frame_count"] == 1
    assert attestation["release_check_passed"] is True
    assert attestation["xu_runtime"]["restore_passed"] is True
    assert np.array_equal(
        result.payload.coverage,
        np.floor(result.payload.coverage),
    )


def test_xu_prepare_refuses_energy_and_domain_before_operation_request(tmp_path):
    with pytest.raises(StitchOperationRefused) as energy:
        _xu_prepared(tmp_path / "energy", energy_eV=18000.0)
    assert energy.value.code == "XU_SOURCE_ENERGY_CONFLICT"

    with pytest.raises(StitchOperationRefused) as domain:
        _xu_prepared(tmp_path / "domain", del_value=46.0)
    assert domain.value.code == "XU_CALIBRATION_DOMAIN_EXCEEDED"


def test_xu_asset_drift_after_prepare_refuses_before_science_or_output(tmp_path):
    request = _xu_prepared(tmp_path)
    Path(
        request.plan.calibration.project_root,
        request.plan.calibration.lexical_relative_path,
    ).touch()
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "XU_CALIBRATION_IDENTITY_MISMATCH"
    assert not Path(request.module.output.target).exists()


def test_xu_pre_cancel_retains_request_and_writes_nothing(tmp_path):
    request = _xu_prepared(tmp_path)
    cancel = threading.Event()
    cancel.set()
    result = run_stitch_operation(request, cancel_token=cancel)
    assert result.terminal.disposition is ModuleDisposition.CANCELLED
    assert result.terminal.request is request.module
    assert result.terminal.code == "CANCELLED"
    assert not Path(request.module.output.target).exists()


def test_xu_effective_geometry_drift_at_prepublish_refuses_commit(
    tmp_path,
    monkeypatch,
):
    request = _xu_prepared(tmp_path)
    asset = Path(
        request.plan.calibration.project_root,
        request.plan.calibration.lexical_relative_path,
    )
    real_resolve = stitch_operation.resolve_xu_stitch_effective_geometry
    calls = []

    def mutate_before_prepublish(receipt, session):
        calls.append("resolve")
        asset.touch()
        return real_resolve(receipt, session)

    monkeypatch.setattr(
        stitch_operation,
        "resolve_xu_stitch_effective_geometry",
        mutate_before_prepublish,
    )
    result = run_stitch_operation(request)
    assert calls == ["resolve"]
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "XU_CALIBRATION_IDENTITY_MISMATCH"
    assert not Path(request.module.output.target).exists()


def test_xu_transient_reload_retry_never_replays_science_or_writer(
    tmp_path,
    monkeypatch,
):
    request = _xu_prepared(tmp_path)
    real_read = stitch_operation.read_analysis_artifact
    real_write = stitch_operation.write_stitched
    real_science = stitch_operation.run_xu_hist_stitch_1d
    reads = []
    writes = []
    science = []

    def fail_read_once(*args, **kwargs):
        reads.append("read")
        if len(reads) == 1:
            raise OSError("transient strict reload fault")
        return real_read(*args, **kwargs)

    def write_once(*args, **kwargs):
        writes.append("write")
        assert kwargs.get("legacy_v1_unit_layout", False) is False
        return real_write(*args, **kwargs)

    def science_once(*args, **kwargs):
        science.append("science")
        return real_science(*args, **kwargs)

    monkeypatch.setattr(stitch_operation, "read_analysis_artifact", fail_read_once)
    monkeypatch.setattr(stitch_operation, "write_stitched", write_once)
    monkeypatch.setattr(
        stitch_operation,
        "run_xu_hist_stitch_1d",
        science_once,
    )
    with pytest.raises(StitchOperationVerificationError) as failed:
        run_stitch_operation(request)
    execution = failed.value.execution
    recovered = execution.retry_verification()
    assert recovered.terminal.disposition is ModuleDisposition.COMMITTED
    assert recovered.payload.schema_version == 2
    assert execution.retry_verification() is recovered
    assert reads == ["read", "read"]
    assert writes == ["write"]
    assert science == ["science"]

    wrong = ModuleOutputRequest(
        tmp_path / "wrong.nexus",
        AnalysisArtifactKind.STITCH_2D,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    with pytest.raises(StitchOperationRefused) as mismatch:
        prepare_stitch_operation(
            request.module.source,
            wrong,
            request.plan,
            project_root=tmp_path,
        )
    assert mismatch.value.code == "OUTPUT_KIND_MISMATCH"


def test_poni_v3_stitch_persists_and_cross_checks_effective_calibration(
    tmp_path,
    monkeypatch,
):
    source = _source_receipt(tmp_path)
    geometry_path = tmp_path / "geometry-v3.poni"
    record = _v3_poni_record()
    geometry_path.write_text(record, encoding="utf-8")
    geometry = capture_stitch_geometry(
        StitchGeometryInput(
            geometry_path,
            StitchGeometryKind.PONI,
            source_motors=(("del", "del"), ("nu", "nu")),
            reference_motor_positions=(("del", 0.0), ("nu", 0.0)),
        )
    )
    plan = StitchOperationPlan(
        geometry,
        npt_1d=96,
        radial_range=(0.1, 6.0),
        monitor_selector=MetadataColumnSelector("I0"),
        max_frame_bytes=64 * 1024 * 1024,
    )
    output_root = tmp_path / "output-v3"
    output_root.mkdir()
    output = ModuleOutputRequest(
        output_root / "stitched.nexus",
        AnalysisArtifactKind.STITCH_1D,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    request = prepare_stitch_operation(
        source,
        output,
        plan,
        project_root=tmp_path,
    )

    geometry_provenance = request.provenance["geometry"]
    effective = geometry_provenance["effective_calibration"]
    assert geometry_provenance["sha256"] == hashlib.sha256(
        record.encode()
    ).hexdigest()
    assert geometry_provenance["receipt_fingerprint"] == geometry.fingerprint
    assert effective["detector_config"] == {
        "pixel1": pytest.approx(0.000172),
        "pixel2": pytest.approx(0.000172),
        "max_shape": [195, 487],
        "orientation": 3,
        "sensor": {
            "material": "Si",
            "thickness": pytest.approx(0.00045),
        },
    }
    assert effective["parallax"] is True

    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert result.payload is not None
    persisted = json.loads(result.payload.provenance_json)
    assert persisted["geometry"]["effective_calibration"] == effective
    assert persisted["geometry"]["sha256"] == geometry.sha256
    assert persisted["geometry"]["receipt_fingerprint"] == geometry.fingerprint

    forged = request.provenance
    forged["geometry"]["effective_calibration"]["parallax"] = False
    module = ModuleOperationRequest(
        request.module.source,
        request.module.output,
        request.plan.fingerprint,
        module_provenance_digest(ModuleKind.STITCH, forged),
    )
    canonical = module_artifact_request(module, forged).provenance_json
    with pytest.raises(ValueError, match="exact request projection"):
        stitch_operation.StitchOperationRequest(
            module,
            request.plan,
            request.manifest,
            canonical,
            stitch_operation._REQUEST_FACTORY,
        )

    runtime_geometry = stitch_operation._runtime_geometry

    def lose_effective_parallax(receipt, raw):
        diffractometer, orientation = runtime_geometry(receipt, raw)
        calibration = replace(diffractometer.calibration, parallax=False)
        return replace(diffractometer, calibration=calibration), orientation

    monkeypatch.setattr(
        stitch_operation,
        "_runtime_geometry",
        lose_effective_parallax,
    )
    mismatch = run_stitch_operation(request)
    assert mismatch.terminal.disposition is ModuleDisposition.REFUSED
    assert mismatch.terminal.code == "GEOMETRY_EFFECTIVE_CALIBRATION_MISMATCH"


def test_prepare_refuses_unsupported_source_kind_before_source_replay(tmp_path):
    initial = _prepared(tmp_path)
    image = tmp_path / "single.tif"
    tifffile.imwrite(image, _ring(11))
    table = run_metadata_table(
        MetadataTablePlan(
            SourceSpec(
                image,
                SourceKind.IMAGE_FILE,
                options={"metadata_format": None},
            )
        )
    )
    assert table.disposition is AnalysisDisposition.COMPLETED
    source = ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.STITCH,
        selected_labels=(0,),
        resolved_selectors=(),
    )
    with pytest.raises(StitchOperationRefused) as refused:
        prepare_stitch_operation(
            source,
            initial.module.output,
            initial.plan,
            project_root=tmp_path,
        )
    assert refused.value.code == "SOURCE_KIND_UNSUPPORTED"


def test_request_factory_rejects_manifest_provenance_mismatch(tmp_path):
    request = _prepared(tmp_path)
    provenance = request.provenance
    provenance["source"]["input_manifest"]["source"]["relative_path"] = (
        "foreign/source"
    )
    module = ModuleOperationRequest(
        request.module.source,
        request.module.output,
        request.plan.fingerprint,
        module_provenance_digest(ModuleKind.STITCH, provenance),
    )
    canonical = module_artifact_request(module, provenance).provenance_json
    with pytest.raises(ValueError, match="exact request projection"):
        stitch_operation.StitchOperationRequest(
            module,
            request.plan,
            request.manifest,
            canonical,
            stitch_operation._REQUEST_FACTORY,
        )


def test_request_factory_rejects_foreign_manifest_contribution_mapping(tmp_path):
    request = _prepared(tmp_path)
    original = request.manifest
    first = original.contributions[0]
    assert first.file_ordinal != 0
    forged_contributions = (
        StitchContribution(
            first.label,
            0,
            first.source_frame_index,
            first.values,
        ),
        *original.contributions[1:],
    )
    forged = StitchInputManifestReceipt(
        original.project_root,
        original.source_relative_path,
        original.source_kind,
        original.source_entry,
        original.source_scan,
        original.source_spec_digest,
        original.source_options_json,
        original.primary_revision,
        original.files,
        forged_contributions,
        original.source_fingerprint,
        original.module_source_fingerprint,
        stitch_operation._REQUEST_FACTORY,
    )
    provenance = request.provenance
    provenance["source"]["input_manifest"] = forged.to_provenance()
    module = ModuleOperationRequest(
        request.module.source,
        request.module.output,
        request.plan.fingerprint,
        module_provenance_digest(ModuleKind.STITCH, provenance),
    )
    canonical = module_artifact_request(module, provenance).provenance_json

    with pytest.raises(ValueError, match="exact bound source projection"):
        stitch_operation.StitchOperationRequest(
            module,
            request.plan,
            forged,
            canonical,
            stitch_operation._REQUEST_FACTORY,
        )


def test_manifest_bounds_the_actual_persisted_projection(tmp_path):
    request = _prepared(tmp_path)
    original = request.manifest
    files = (
        StitchManifestFile("source", original.primary_revision),
        StitchManifestFile("source/raw.tif", original.primary_revision),
    )
    contributions = tuple(
        StitchContribution(
            label,
            1,
            0,
            (("del", 0, float(label)), ("nu", 0, float(label))),
        )
        for label in range(4096)
    )
    with pytest.raises(
        StitchOperationRefused,
        match="persisted input manifest exceeds 512 KiB",
    ):
        StitchInputManifestReceipt(
            original.project_root,
            original.source_relative_path,
            original.source_kind,
            original.source_entry,
            original.source_scan,
            original.source_spec_digest,
            original.source_options_json,
            original.primary_revision,
            files,
            contributions,
            original.source_fingerprint,
            original.module_source_fingerprint,
            stitch_operation._REQUEST_FACTORY,
        )


def test_poni_prepare_requires_its_del_nu_selectors_to_be_bound(tmp_path):
    initial = _prepared(tmp_path)
    table = run_metadata_table(
        MetadataTablePlan(initial.module.source.analysis.source_spec)
    )
    assert table.disposition is AnalysisDisposition.COMPLETED
    unbound = ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.STITCH,
        selected_labels=initial.module.source.selected_labels,
        resolved_selectors=(),
    )
    path = tmp_path / "single-position.poni"
    path.write_text(_poni_record(), encoding="utf-8")
    geometry = capture_stitch_geometry(
        StitchGeometryInput(
            path,
            StitchGeometryKind.PONI,
            source_motors=(("del", "del"), ("nu", "nu")),
            reference_motor_positions=(("del", 0.0), ("nu", 0.0)),
        )
    )
    plan = StitchOperationPlan(geometry, radial_range=(0.1, 6.0))
    output = ModuleOutputRequest(
        tmp_path / "poni-output.nexus",
        AnalysisArtifactKind.STITCH_1D,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    with pytest.raises(StitchOperationRefused) as refused:
        prepare_stitch_operation(
            unbound,
            output,
            plan,
            project_root=tmp_path,
        )
    assert refused.value.code == "UNBOUND_METADATA_SELECTOR"


def test_prepare_refuses_source_or_geometry_outside_project(tmp_path):
    request = _prepared(tmp_path)
    isolated_project = tmp_path / "isolated-project"
    isolated_project.mkdir()
    with pytest.raises(StitchOperationRefused) as refused:
        prepare_stitch_operation(
            request.module.source,
            request.module.output,
            request.plan,
            project_root=isolated_project,
        )
    assert refused.value.code == "INPUT_OUTSIDE_PROJECT"


@pytest.mark.parametrize(
    "contradiction",
    ("selected", "root", "single-marker"),
)
def test_prepare_refuses_incoherent_tiff_source_intent(tmp_path, contradiction):
    request = _prepared(tmp_path)
    admitted = request.module.source
    spec = admitted.analysis.source_spec
    options = dict(spec.options)
    if contradiction == "selected":
        unrelated = tmp_path / "unrelated.txt"
        unrelated.write_text("not a member", encoding="utf-8")
        options["selected_file"] = str(unrelated)
        uri = spec.uri
    elif contradiction == "root":
        unrelated = tmp_path / "unrelated.tif"
        tifffile.imwrite(unrelated, _ring(99))
        uri = unrelated
    else:
        options["selection_mode"] = "single_image"
        uri = spec.uri
    forged_spec = SourceSpec(uri, SourceKind.TIFF_SERIES, options=options)
    table = run_metadata_table(MetadataTablePlan(forged_spec))
    assert table.disposition is AnalysisDisposition.COMPLETED
    forged_source = ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.STITCH,
        selected_labels=admitted.selected_labels,
        resolved_selectors=admitted.resolved_selectors,
    )

    with pytest.raises(StitchOperationRefused) as refused:
        prepare_stitch_operation(
            forged_source,
            request.module.output,
            request.plan,
            project_root=tmp_path,
        )
    assert refused.value.code == "SOURCE_SPEC_UNSUPPORTED"


def test_pre_cancel_opens_no_output_and_returns_cancelled(tmp_path):
    request = _prepared(tmp_path)
    cancel = threading.Event()
    cancel.set()
    result = run_stitch_operation(request, cancel_token=cancel)
    assert result.terminal.disposition is ModuleDisposition.CANCELLED
    assert result.terminal.code == "CANCELLED"
    assert result.payload is None
    assert not Path(request.module.output.target).exists()


def test_frame_load_cancellation_stops_before_next_frame_and_writes_nothing(
    tmp_path,
    monkeypatch,
):
    request = _prepared(tmp_path)
    cancel = threading.Event()
    loaded = []
    real_load = stitch_operation._OrientedSource.load_frame

    def load_then_cancel(source, label):
        image = real_load(source, label)
        loaded.append(label)
        cancel.set()
        return image

    monkeypatch.setattr(
        stitch_operation._OrientedSource,
        "load_frame",
        load_then_cancel,
    )
    result = run_stitch_operation(request, cancel_token=cancel)
    assert result.terminal.disposition is ModuleDisposition.CANCELLED
    assert loaded == [request.module.source.selected_labels[0]]
    assert not Path(request.module.output.target).exists()


def test_frame_bound_covers_float64_processing_representation(tmp_path):
    raw_bytes = _ring(0).nbytes
    request = _prepared(tmp_path, max_frame_bytes=raw_bytes + 1)
    result = run_stitch_operation(request)

    assert result.terminal.disposition is ModuleDisposition.FAILED
    assert result.terminal.code == "SCIENCE_FAILED"
    assert "float64 oriented Stitch frame" in result.terminal.diagnostic
    assert not Path(request.module.output.target).exists()


@pytest.mark.parametrize("mode", ("1d", "2d"))
def test_operation_commits_and_strictly_reloads_detached_payload(tmp_path, mode):
    request = _prepared(tmp_path, mode=mode)
    progress = []
    result = run_stitch_operation(
        request,
        progress_callback=progress.append,
    )
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert result.terminal.commit is not None
    assert result.payload is not None
    assert result.payload.inspection.kind is request.module.output.kind
    assert result.payload.inspection.request_fingerprint == request.module.fingerprint
    assert (
        result.payload.inspection.result_fingerprint
        == result.terminal.commit.result_fingerprint
    )
    assert result.payload.intensity.flags.writeable is False
    assert result.payload.coverage is not None
    assert result.payload.normalization is not None
    assert result.payload.coverage.flags.writeable is False
    assert result.payload.normalization.flags.writeable is False
    assert result.payload.coverage.shape == result.payload.intensity.shape
    assert result.payload.normalization.shape == result.payload.intensity.shape
    assert np.any(result.payload.coverage > 0)
    assert np.any(result.payload.normalization > 0)
    assert result.payload.intensity.shape == (
        (96,) if mode == "1d" else (48, 32)
    )
    totals = {item.total for item in progress}
    assert totals == {len(request.module.source.selected_labels) + 4}
    assert [item.completed for item in progress] == sorted(
        item.completed for item in progress
    )
    assert progress[-1].stage == "reload"
    assert progress[-1].completed == progress[-1].total


def test_geometry_mutation_after_prepare_refuses_before_science_or_output(tmp_path):
    request = _prepared(tmp_path)
    geometry = Path(request.plan.geometry.lexical_path)
    replacement = dict(_goniometer_record())
    replacement["wavelength"] = 1.1e-10
    geometry.write_text(json.dumps(replacement), encoding="utf-8")
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "GEOMETRY_HASH_MISMATCH"
    assert not Path(request.module.output.target).exists()


def test_geometry_mutation_at_true_prepublish_refuses_commit(tmp_path, monkeypatch):
    request = _prepared(tmp_path)
    geometry = Path(request.plan.geometry.lexical_path)
    real_revalidate = stitch_operation._revalidate_geometry
    calls = 0

    def mutate_at_prepublish(receipt):
        nonlocal calls
        calls += 1
        if calls == 5:
            replacement = dict(_goniometer_record())
            replacement["wavelength"] = 1.1e-10
            geometry.write_text(json.dumps(replacement), encoding="utf-8")
        return real_revalidate(receipt)

    monkeypatch.setattr(
        stitch_operation,
        "_revalidate_geometry",
        mutate_at_prepublish,
    )
    result = run_stitch_operation(request)
    assert calls == 5
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "GEOMETRY_HASH_MISMATCH"
    assert not Path(request.module.output.target).exists()


def test_transient_strict_reload_retry_does_not_replay_science_or_writer(
    tmp_path,
    monkeypatch,
):
    request = _prepared(tmp_path)
    real_read = stitch_operation.read_analysis_artifact
    real_write = stitch_operation.write_stitched
    real_science = stitch_operation.run_stitch
    reads = []
    writes = []
    science = []

    def fail_read_once(*args, **kwargs):
        reads.append("read")
        if len(reads) == 1:
            raise OSError("transient strict reload fault")
        return real_read(*args, **kwargs)

    def write_once(*args, **kwargs):
        writes.append("write")
        assert kwargs["legacy_v1_unit_layout"] is True
        return real_write(*args, **kwargs)

    def science_once(*args, **kwargs):
        science.append("science")
        return real_science(*args, **kwargs)

    monkeypatch.setattr(stitch_operation, "read_analysis_artifact", fail_read_once)
    monkeypatch.setattr(stitch_operation, "write_stitched", write_once)
    monkeypatch.setattr(stitch_operation, "run_stitch", science_once)
    with pytest.raises(StitchOperationVerificationError) as failed:
        run_stitch_operation(request)
    execution = failed.value.execution
    recovered = execution.retry_verification()
    assert recovered.terminal.disposition is ModuleDisposition.COMMITTED
    assert recovered.payload is not None
    assert execution.retry_verification() is recovered
    assert reads == ["read", "read"]
    assert writes == ["write"]
    assert science == ["science"]


def test_cleanup_pending_retains_execution_and_retries_without_replay(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.io.output_transaction as transaction_api

    initial = _prepared(tmp_path)
    target = Path(initial.module.output.target)
    target.write_bytes(b"prior operator output")
    replacement = ModuleOutputRequest(
        target,
        AnalysisArtifactKind.STITCH_1D,
        AnalysisArtifactOverwrite.REPLACE,
    )
    request = prepare_stitch_operation(
        initial.module.source,
        replacement,
        initial.plan,
        project_root=tmp_path,
    )
    real_unlink = transaction_api._unlink
    real_write = stitch_operation.write_stitched
    real_science = stitch_operation.run_stitch
    failures = []
    writes = []
    science = []

    def fail_backup_once(path):
        if ".xdart-replacing-" in Path(path).name and not failures:
            failures.append("backup")
            raise OSError("backup cleanup fault")
        return real_unlink(path)

    def write_once(*args, **kwargs):
        writes.append("write")
        return real_write(*args, **kwargs)

    def science_once(*args, **kwargs):
        science.append("science")
        return real_science(*args, **kwargs)

    monkeypatch.setattr(transaction_api, "_unlink", fail_backup_once)
    monkeypatch.setattr(stitch_operation, "write_stitched", write_once)
    monkeypatch.setattr(stitch_operation, "run_stitch", science_once)
    with pytest.raises(StitchOperationCleanupPending) as pending:
        run_stitch_operation(request)
    recovered = pending.value.retry_cleanup()
    assert recovered.terminal.disposition is ModuleDisposition.COMMITTED
    assert recovered.payload is not None
    assert pending.value.execution.retry_cleanup() is recovered
    assert failures == ["backup"]
    assert writes == ["write"]
    assert science == ["science"]


def test_create_new_output_conflict_is_late_refusal_without_overwrite(tmp_path):
    request = _prepared(tmp_path)
    target = Path(request.module.output.target)
    original = b"operator-owned"
    target.write_bytes(original)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "OUTPUT_EXISTS"
    assert target.read_bytes() == original


def test_plan_rejects_unbounded_or_unsupported_science(tmp_path):
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(_goniometer_record()), encoding="utf-8")
    geometry = capture_stitch_geometry(
        StitchGeometryInput(
            path,
            StitchGeometryKind.PYFAI_GONIOMETER_JSON,
            source_motors=(("del_angle", "del"), ("nu_angle", "nu")),
        )
    )
    with pytest.raises(ValueError, match="radial range"):
        StitchOperationPlan(geometry)
    with pytest.raises(ValueError, match="multigeometry"):
        StitchOperationPlan(
            geometry,
            backend="pyfai_hist",
            radial_range=(0.1, 1.0),
        )
    with pytest.raises(ValueError, match="point limit"):
        StitchOperationPlan(
            geometry,
            mode="2d",
            npt_rad_2d=100_000,
            npt_azim_2d=100_000,
            radial_range=(0.1, 1.0),
            azimuth_range=(-1.0, 1.0),
        )
