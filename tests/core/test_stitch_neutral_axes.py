"""Neutral physical Stitch axes; logical coordinates and scientific bytes stay fixed."""

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactInvalid,
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
    AnalysisArtifactRequest,
    admit_analysis_artifact,
    analysis_execution_attestation_digest,
    inspect_analysis_artifact,
    project_analysis_artifact_result,
    read_analysis_artifact,
)
from xrd_tools.io.nexus import (
    read_stitched,
    validate_group_against_schema,
    write_stitched,
)
from xrd_tools.io.output_transaction import OutputTransactionCoordinator
from xrd_tools.io.schema import axis_display_metadata
from tests.core.test_analysis_artifact import _execution_attestation
from tests.core.test_headless_write_roundtrip import (
    _add_current_result, _current_entry, _r1d, _r2d,
)
from tests.core.test_stitch_operation import _prepared, _xu_prepared


def _assert_neutral(group, payload=None):
    two_d = group.name.endswith("stitched_2d")
    names = ("axis_1", "axis_2") if two_d else ("axis_1",)
    assert "q" not in group and "chi" not in group
    assert tuple(group.attrs["axes"].astype(str)) == names
    assert validate_group_against_schema(group, group.name.rsplit("/", 1)[1]) == []
    for physical, logical in zip(names, ("q", "chi")):
        axis = group[physical]
        units = axis.attrs["units"]
        if isinstance(units, bytes):
            units = units.decode()
        label = axis.attrs["long_name"]
        if isinstance(label, bytes):
            label = label.decode()
        assert label == axis_display_metadata(units)["long_name"]
        if payload is not None:
            np.testing.assert_array_equal(axis[()], payload.axis(logical))
    if payload is not None:
        np.testing.assert_array_equal(group["intensity"][()], payload.intensity)


@pytest.mark.parametrize("backend,mode,version", [
    ("multigeometry", "1d", 4), ("multigeometry", "2d", 4), ("xu_hist", "1d", 5),
])
def test_real_stitch_operation_writes_neutral_axes(tmp_path, backend, mode, version):
    request = _xu_prepared(tmp_path) if backend == "xu_hist" else _prepared(tmp_path, mode=mode)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert result.payload.schema_version == version
    reloaded = read_analysis_artifact(request.module.output.target,
                                     expected_kind=request.module.output.kind)
    assert reloaded.result_fingerprint == result.payload.result_fingerprint
    with h5py.File(request.module.output.target, "r") as handle:
        _assert_neutral(handle[f"entry/stitched_{mode}"], reloaded)


def _publish_fixture(directory, version, mode):
    kind = AnalysisArtifactKind.STITCH_1D if mode == "1d" else AnalysisArtifactKind.STITCH_2D
    q = np.linspace(0.1, 2.0, 7)
    chi = np.linspace(-30, 30, 3)
    axes = (("q", q),) if mode == "1d" else (("q", q), ("chi", chi))
    units = (("q", "q_A^-1"),) if mode == "1d" else (("q", "q_A^-1"), ("chi", "chi_deg"))
    shape = (7,) if mode == "1d" else (7, 3)
    projection = project_analysis_artifact_result(
        kind=kind, axes=axes, axis_units=units,
        intensity=np.arange(np.prod(shape), dtype=float).reshape(shape),
        sigma=None, coverage=np.ones(shape), normalization=np.ones(shape),
    )
    digest = hashlib.sha256(b"neutral-stitch-fixture").hexdigest()
    attestation = _execution_attestation(digest, projection.result_fingerprint) if version in (2, 5) else None
    attestation_digest = None if attestation is None else analysis_execution_attestation_digest(
        kind, attestation, request_fingerprint=digest,
    )
    request = AnalysisArtifactRequest(
        directory / f"artifact_{version}.nexus", kind, AnalysisArtifactOverwrite.CREATE_NEW,
        digest, digest, digest, digest, {"test": "neutral-axes"}, schema_version=version,
        execution_attestation=attestation, execution_attestation_digest=attestation_digest,
    )
    receipt = admit_analysis_artifact(request, coordinator=OutputTransactionCoordinator()).publish(
        lambda entry: write_stitched(entry, result_projection=projection,
                                     bounded_artifact=True, provenance=request.provenance_json),
    )
    return request, read_analysis_artifact(request.target, expected_receipt=receipt)


@pytest.mark.parametrize("old_version,new_version,mode", [(1, 4, "1d"), (1, 4, "2d"), (2, 5, "1d")])
def test_neutral_stitch_preserves_legacy_scientific_fingerprint(tmp_path, old_version, new_version, mode):
    old_request, old = _publish_fixture(tmp_path, old_version, mode)
    before = hashlib.sha256(Path(old_request.target).read_bytes()).hexdigest()
    new_request, new = _publish_fixture(tmp_path, new_version, mode)
    assert old.result_fingerprint == new.result_fingerprint
    assert old.execution_attestation_json == new.execution_attestation_json
    np.testing.assert_array_equal(old.intensity, new.intensity)
    np.testing.assert_array_equal(old.coverage, new.coverage)
    np.testing.assert_array_equal(old.normalization, new.normalization)
    for name in (("q",) if mode == "1d" else ("q", "chi")):
        np.testing.assert_array_equal(old.axis(name), new.axis(name))
    with h5py.File(new_request.target, "r") as handle:
        _assert_neutral(handle[f"entry/stitched_{mode}"], new)
    assert hashlib.sha256(Path(old_request.target).read_bytes()).hexdigest() == before


@pytest.mark.parametrize("version", [4, 5])
def test_neutral_artifact_refuses_old_physical_names_under_new_version(tmp_path, version):
    request, _ = _publish_fixture(tmp_path, version, "1d")
    with h5py.File(request.target, "r+") as handle:
        group = handle["entry/stitched_1d"]
        group.move("axis_1", "q")
        group.attrs["axes"] = ["q"]
    with pytest.raises(AnalysisArtifactInvalid, match="axes"):
        inspect_analysis_artifact(request.target)


def test_embedded_stitch_neutral_axes_keep_logical_coordinates(tmp_path):
    path = tmp_path / "embedded.nexus"
    one, two = _r1d(1), _r2d(2)
    with h5py.File(path, "w") as handle:
        entry = _current_entry(handle)
        _add_current_result(entry)
        write_stitched(entry, stitched_1d=one, stitched_2d=two)
        _assert_neutral(entry["stitched_1d"])
        _assert_neutral(entry["stitched_2d"])
    loaded = read_stitched(path)
    assert loaded.stitched_1d.dims == ("q",)
    assert loaded.stitched_2d.dims == ("q", "chi")
    np.testing.assert_array_equal(loaded.stitched_1d, np.asarray(one.intensity, np.float32))
    np.testing.assert_array_equal(loaded.stitched_2d, np.asarray(two.intensity, np.float32))
