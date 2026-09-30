"""Neutral physical Stitch axes; logical coordinates and scientific bytes stay fixed. A Stitch result is an
Int record (the old Stitch layout was removed, 2026-09-30: "Remove old format")."""

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
    AnalysisArtifactRequest,
    admit_analysis_artifact,
    analysis_execution_attestation_digest,
    project_analysis_artifact_result,
    read_analysis_artifact,
)
from xrd_tools.io.output_transaction import OutputTransactionCoordinator
from xrd_tools.io.schema import axis_display_metadata
from xrd_tools.io.stitch_record import write_stitch_projection
from tests.core.test_analysis_artifact import _execution_attestation
from tests.core.test_stitch_operation import _prepared, _xu_prepared


def _assert_record(entry, payload):
    """A Stitch 1D result is an Int record: its neutral axis and one row (``stitch_record``)."""
    group = entry["integrated_1d"]
    axis = group["axis_1"]
    units = axis.attrs["units"]
    units = units.decode() if isinstance(units, bytes) else units
    label = axis.attrs["long_name"]
    label = label.decode() if isinstance(label, bytes) else label
    assert "q" not in group and label == axis_display_metadata(units)["long_name"]
    np.testing.assert_array_equal(axis[()], payload.axis("q"))
    np.testing.assert_array_equal(group["intensity"][0], payload.intensity)


@pytest.mark.parametrize("backend,version", [("multigeometry", 4), ("xu_hist", 5)])
def test_real_stitch_operation_writes_neutral_axes(tmp_path, backend, version):
    request = _xu_prepared(tmp_path) if backend == "xu_hist" else _prepared(tmp_path)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert result.payload.schema_version == version
    reloaded = read_analysis_artifact(request.module.output.target,
                                     expected_kind=request.module.output.kind)
    assert reloaded.result_fingerprint == result.payload.result_fingerprint
    with h5py.File(request.module.output.target, "r") as handle:
        _assert_record(handle["entry"], reloaded)


def _publish_fixture(directory, version):
    kind = AnalysisArtifactKind.STITCH_1D
    axes, units, shape = (("q", np.linspace(0.1, 2.0, 7)),), (("q", "q_A^-1"),), (7,)
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
        lambda entry: write_stitch_projection(entry, request.target, projection))
    return request, read_analysis_artifact(request.target, expected_receipt=receipt)


@pytest.mark.parametrize("old_version,new_version", [(1, 4), (2, 5)])
def test_neutral_stitch_preserves_legacy_scientific_fingerprint(tmp_path, old_version, new_version):
    old_request, old = _publish_fixture(tmp_path, old_version)
    before = hashlib.sha256(Path(old_request.target).read_bytes()).hexdigest()
    new_request, new = _publish_fixture(tmp_path, new_version)
    assert old.result_fingerprint == new.result_fingerprint
    assert old.execution_attestation_json == new.execution_attestation_json
    np.testing.assert_array_equal(old.intensity, new.intensity)
    np.testing.assert_array_equal(old.coverage, new.coverage)
    np.testing.assert_array_equal(old.normalization, new.normalization)
    np.testing.assert_array_equal(old.axis("q"), new.axis("q"))
    with h5py.File(new_request.target, "r") as handle:
        _assert_record(handle["entry"], new)
    assert hashlib.sha256(Path(old_request.target).read_bytes()).hexdigest() == before
