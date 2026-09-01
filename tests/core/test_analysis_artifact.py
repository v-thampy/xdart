from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import pickle
import shutil

import h5py
import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.image import read_image
from xrd_tools.io.image_source import ImageSourceKind, classify_image_source
from xrd_tools.io.analysis_artifact import (
    ANALYSIS_KIND_ATTR,
    ANALYSIS_SCHEMA_ATTR,
    ANALYSIS_SCHEMA_NAME,
    ANALYSIS_SCHEMA_VERSION_V2,
    AnalysisArtifactCleanupPending,
    AnalysisArtifactError,
    AnalysisArtifactInvalid,
    AnalysisArtifactKind,
    AnalysisArtifactOutputSnapshot,
    AnalysisArtifactOverwrite,
    AnalysisArtifactPayload,
    AnalysisArtifactProjectionInvalid,
    AnalysisArtifactReceipt,
    AnalysisArtifactResultProjection,
    AnalysisArtifactRequest,
    admit_analysis_artifact,
    analysis_execution_attestation_digest,
    canonical_analysis_provenance,
    inspect_analysis_artifact,
    project_analysis_artifact_result,
    read_analysis_artifact,
)
from xrd_tools.io.nexus import (
    find_nexus_image_dataset,
    write_rsm,
    write_stitched,
)
from xrd_tools.io.output_transaction import (
    LeaseOwner,
    LeaseUnavailable,
    OutputTransactionCoordinator,
    TargetChanged,
    TransactionPhase,
    stream_terminal_object_revision,
)
from xrd_tools.io.processed_scan_id import (
    ProcessedXdartInputError,
    is_current_processed_xdart_path,
    require_raw_input,
)
from xrd_tools.rsm.volume import RSMVolume
from xrd_tools.sources.descriptor import describe_container
from xrd_tools.sources.probe import ProbeState


def test_analysis_artifact_types_are_public_lazy_io_exports():
    import xrd_tools.io as io_api

    assert io_api.AnalysisArtifactError is AnalysisArtifactError
    assert io_api.AnalysisArtifactOutputSnapshot is AnalysisArtifactOutputSnapshot
    assert io_api.AnalysisArtifactPayload is AnalysisArtifactPayload
    assert io_api.AnalysisArtifactResultProjection is AnalysisArtifactResultProjection
    assert io_api.project_analysis_artifact_result is project_analysis_artifact_result
    assert io_api.read_analysis_artifact is read_analysis_artifact
    assert "AnalysisArtifactError" in io_api.__all__
    assert "AnalysisArtifactOutputSnapshot" in io_api.__all__
    assert "AnalysisArtifactPayload" in io_api.__all__
    assert "AnalysisArtifactResultProjection" in io_api.__all__
    assert "project_analysis_artifact_result" in io_api.__all__
    assert "read_analysis_artifact" in io_api.__all__


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _request(
    target: Path,
    kind: AnalysisArtifactKind,
    *,
    overwrite: AnalysisArtifactOverwrite = AnalysisArtifactOverwrite.CREATE_NEW,
) -> AnalysisArtifactRequest:
    return AnalysisArtifactRequest(
        target,
        kind,
        overwrite,
        _digest("request"),
        _digest("source"),
        _digest("plan"),
        _digest("provenance"),
        {"kind": kind.value, "nested": {"frames": [1, 2]}},
    )


def _execution_attestation(
    request_fingerprint: str,
    result_fingerprint: str,
    *,
    frame_count: int = 7,
) -> dict[str, object]:
    return {
        "schema_version": "analysis-execution-attestation-v1",
        "module_request_fingerprint": request_fingerprint,
        "result_projection_policy": "analysis_artifact_stored_le_f4_v1",
        "result_fingerprint": result_fingerprint,
        "selected_frame_count": frame_count,
        "release_check_frame_count": frame_count,
        "release_check_passed": True,
        "q_root_policy": "shared_ultimate_ndarray_root_weakref_v1",
        "xu_runtime": {
            "lock_policy": "shared_xrd_tools_xu_rlock_v1",
            "xrayutilities_distribution_version": "1.7.12",
            "xrayutilities_module_version": "1.7.12",
            "numpy_version": "2.5.1",
            "config_epsilon": 1e-8,
            "config_digits": 8,
            "nthreads_before": 0,
            "nthreads_effective": 1,
            "nthreads_restored": 0,
            "restore_passed": True,
        },
    }


def _rsm_execution_attestation(
    request_fingerprint: str,
    result_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": "rsm-execution-attestation-v1",
        "module_request_fingerprint": request_fingerprint,
        "result_projection_policy": "analysis_artifact_stored_le_f4_v1",
        "result_fingerprint": result_fingerprint,
        "geometry_asset_receipt_fingerprint": _digest("rsm-asset"),
        "effective_geometry_fingerprint": _digest("rsm-effective"),
        "common_grid_fingerprint": _digest("rsm-grid"),
        "selected_scan_count": 2,
        "selected_frame_count": 7,
        "science_chunk_count": 3,
        "q_release_check_chunk_count": 3,
        "frame_release_check_frame_count": 7,
        "release_check_passed": True,
        "member_masks": [
            {
                "ordinal": 0,
                "member_preflight_fingerprint": _digest("rsm-member-0"),
                "mask_policy": "none",
                "full_shape": [4, 5],
                "full_raw_digest": None,
                "full_masked_pixel_count": 0,
                "cropped_shape": [3, 4],
                "cropped_raw_digest": None,
                "cropped_masked_pixel_count": 0,
                "mask_receipt_fingerprint": _digest("rsm-mask-0"),
            },
            {
                "ordinal": 1,
                "member_preflight_fingerprint": _digest("rsm-member-1"),
                "mask_policy": "exact-all-selected-frames-static-hot-v1",
                "full_shape": [4, 5],
                "full_raw_digest": _digest("rsm-full-mask-1"),
                "full_masked_pixel_count": 3,
                "cropped_shape": [3, 4],
                "cropped_raw_digest": _digest("rsm-cropped-mask-1"),
                "cropped_masked_pixel_count": 2,
                "mask_receipt_fingerprint": _digest("rsm-mask-1"),
            },
        ],
        "xu_runtime": {
            "lock_policy": "shared_xrd_tools_xu_rlock_v1",
            "xrayutilities_distribution_version": "1.7.12",
            "xrayutilities_module_version": "1.7.12",
            "numpy_version": "2.5.1",
            "config_epsilon": 1e-8,
            "config_digits": 8,
            "nthreads_before": 0,
            "nthreads_effective": 1,
            "nthreads_restored": 0,
            "restore_passed": True,
        },
    }


def _writer(request: AnalysisArtifactRequest):
    q = np.linspace(0.1, 2.0, 7)
    if request.kind is AnalysisArtifactKind.STITCH_1D:
        value = IntegrationResult1D(
            q,
            np.linspace(1.0, 2.0, 7),
            np.linspace(0.1, 0.2, 7),
            "q_A^-1",
        )
        return lambda entry: write_stitched(
            entry,
            stitched_1d=value,
            provenance=request.provenance_json,
            bounded_artifact=True,
        )
    if request.kind is AnalysisArtifactKind.STITCH_2D:
        chi = np.linspace(-10.0, 10.0, 5)
        value = IntegrationResult2D(
            radial=q,
            azimuthal=chi,
            intensity=np.arange(35, dtype=float).reshape(7, 5),
            unit="q_A^-1",
            azimuthal_unit="chi_deg",
        )
        return lambda entry: write_stitched(
            entry,
            stitched_2d=value,
            provenance=request.provenance_json,
            bounded_artifact=True,
        )
    volume = RSMVolume(
        h=np.linspace(-1, 1, 3),
        k=np.linspace(-2, 2, 4),
        l=np.linspace(0, 3, 5),
        intensity=np.arange(60, dtype=float).reshape(3, 4, 5),
    )
    return lambda entry: write_rsm(
        entry,
        volume,
        provenance=request.provenance_json,
        bounded_artifact=True,
    )


@pytest.mark.parametrize("kind", tuple(AnalysisArtifactKind))
def test_standalone_artifact_roundtrip_is_exact_and_raw_negative(tmp_path, kind):
    request = _request(tmp_path / f"{kind.value}.nexus", kind)
    output = admit_analysis_artifact(
        request, coordinator=OutputTransactionCoordinator()
    )
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(output)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(output)
    receipt = output.publish(_writer(request))
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(receipt)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(receipt)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(receipt)

    assert receipt.request is request
    assert receipt.inspection.kind is kind
    assert receipt.inspection.path == request.target
    assert receipt.inspection.group == kind.group
    assert receipt.inspection.provenance_json == request.provenance_json
    assert receipt.inspection.request_fingerprint == request.request_fingerprint
    assert receipt.inspection.source_fingerprint == request.source_fingerprint
    assert receipt.inspection.plan_fingerprint == request.plan_fingerprint
    assert receipt.inspection.provenance_digest == request.provenance_digest
    assert stream_terminal_object_revision(receipt.terminal) is not None
    assert receipt.terminal.target == request.target
    assert output.snapshot.phase is TransactionPhase.COMMITTED
    assert output.snapshot.remaining_lease_owners == ()
    assert output.snapshot.receipt is receipt
    assert inspect_analysis_artifact(request.target, expected_request=request).kind is kind
    payload = read_analysis_artifact(request.target, expected_receipt=receipt)
    assert payload.kind is kind
    assert payload.inspection == receipt.inspection
    assert payload.result_fingerprint == receipt.inspection.result_fingerprint
    assert payload.provenance_json == request.provenance_json
    assert tuple(name for name, _values in payload.axes) == {
        AnalysisArtifactKind.STITCH_1D: ("q",),
        AnalysisArtifactKind.STITCH_2D: ("q", "chi"),
        AnalysisArtifactKind.RSM: ("h", "k", "l"),
    }[kind]
    assert payload.intensity.shape == receipt.inspection.shape
    assert (payload.sigma is not None) is receipt.inspection.has_sigma
    assert not payload.intensity.flags.writeable
    assert all(not values.flags.writeable for _name, values in payload.axes)
    assert payload.intensity is not payload.intensity
    with pytest.raises(ValueError, match="read-only"):
        payload.intensity.flat[0] = 0
    with pytest.raises(ValueError):
        payload.intensity.flags.writeable = True
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(payload)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(payload)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(payload)
    np.testing.assert_array_equal(
        payload.axis(payload.axes[0][0]),
        payload.axes[0][1],
    )
    with pytest.raises(KeyError):
        payload.axis("missing")

    with h5py.File(request.target, "r") as handle:
        entry = handle["entry"]
        assert bytes(entry.attrs[ANALYSIS_SCHEMA_ATTR]).decode() == ANALYSIS_SCHEMA_NAME
        assert bytes(entry.attrs[ANALYSIS_KIND_ATTR]).decode() == kind.value
        assert bytes(entry.attrs["file_name"]).decode() == request.target
        assert not ({"integrated_1d", "integrated_2d"} & set(entry))
    assert not is_current_processed_xdart_path(request.target)
    with pytest.raises(ProcessedXdartInputError):
        require_raw_input(request.target)


def test_rsm_artifact_is_negative_at_every_raw_detector_front_door(tmp_path):
    request = _request(
        tmp_path / "rsm-front-door-negative.nexus",
        AnalysisArtifactKind.RSM,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    assert classify_image_source(request.target).kind is ImageSourceKind.UNKNOWN
    assert describe_container(request.target).state is ProbeState.INVALID
    with pytest.raises(ProcessedXdartInputError):
        find_nexus_image_dataset(request.target)
    with pytest.raises(ProcessedXdartInputError):
        read_image(request.target)


def test_stitch_2d_preserves_sigma_and_non_q_units(tmp_path):
    request = _request(
        tmp_path / "stitched-2d-sigma.nexus",
        AnalysisArtifactKind.STITCH_2D,
    )
    q = np.linspace(1.0, 5.0, 7)
    chi = np.linspace(-0.2, 0.2, 5)
    value = IntegrationResult2D(
        radial=q,
        azimuthal=chi,
        intensity=np.arange(35, dtype=float).reshape(7, 5),
        sigma=np.full((7, 5), 0.25),
        unit="2th_deg",
        azimuthal_unit="chi_rad",
    )
    receipt = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(
        lambda entry: write_stitched(
            entry,
            stitched_2d=value,
            provenance=request.provenance_json,
            bounded_artifact=True,
        )
    )
    assert receipt.inspection.shape == (7, 5)
    assert receipt.inspection.axis_units == (
        ("q", "2th_deg"),
        ("chi", "chi_rad"),
    )
    assert receipt.inspection.has_sigma is True
    payload = read_analysis_artifact(request.target, expected_receipt=receipt)
    np.testing.assert_allclose(payload.axis("q"), q.astype(np.float32))
    np.testing.assert_allclose(payload.axis("chi"), chi.astype(np.float32))
    np.testing.assert_allclose(payload.intensity, value.intensity.astype(np.float32))
    np.testing.assert_allclose(payload.sigma, 0.25)
    assert payload.inspection.axis_units == receipt.inspection.axis_units
    with h5py.File(request.target, "r") as handle:
        result = handle["entry/stitched_2d"]
        assert "sigma" in result
        np.testing.assert_allclose(result["sigma"][()], 0.25)
        assert bytes(result["q"].attrs["units"]).decode() == "2th_deg"
        assert bytes(result["chi"].attrs["units"]).decode() == "chi_rad"


def test_result_fingerprint_is_scientific_and_storage_independent(tmp_path):
    q = np.linspace(0.1, 1.0, 7)
    intensity = np.linspace(2.0, 3.0, 7)
    sigma = np.linspace(0.1, 0.2, 7)

    def publish(
        label,
        *,
        radial=q,
        values=intensity,
        errors=sigma,
        unit="q_A^-1",
        compression=None,
    ):
        request = AnalysisArtifactRequest(
            tmp_path / f"fingerprint-{label}.nexus",
            AnalysisArtifactKind.STITCH_1D,
            AnalysisArtifactOverwrite.CREATE_NEW,
            _digest(f"request-{label}"),
            _digest(f"source-{label}"),
            _digest(f"plan-{label}"),
            _digest(f"provenance-{label}"),
            {"storage-label": label},
        )
        result = IntegrationResult1D(
            np.asarray(radial),
            np.asarray(values),
            None if errors is None else np.asarray(errors),
            unit,
        )
        return admit_analysis_artifact(
            request,
            coordinator=OutputTransactionCoordinator(),
        ).publish(
            lambda entry: write_stitched(
                entry,
                stitched_1d=result,
                provenance=request.provenance_json,
                compression=compression,
                bounded_artifact=True,
            )
        ).inspection.result_fingerprint

    baseline = publish("baseline")
    assert baseline == (
        "357e3a6c2b07dbdfa4a5a4f6865a8dee"
        "5b09dfb7ee6521a0f0cad77eb42c4fc7"
    )
    assert publish("same-compressed", compression="gzip") == baseline

    changed_axis = q.copy()
    changed_axis[3] += 0.01
    changed_intensity = intensity.copy()
    changed_intensity[2] += 1.0
    changed_sigma = sigma.copy()
    changed_sigma[4] += 0.01
    variants = {
        publish("axis", radial=changed_axis),
        publish("unit", unit="2th_deg"),
        publish("intensity", values=changed_intensity),
        publish("sigma-absent", errors=None),
        publish("sigma-value", errors=changed_sigma),
    }
    assert baseline not in variants
    assert len(variants) == 5

    rsm_request = _request(
        tmp_path / "fingerprint-kind.nexus",
        AnalysisArtifactKind.RSM,
    )
    rsm = admit_analysis_artifact(
        rsm_request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(rsm_request))
    assert rsm.inspection.result_fingerprint != baseline


def test_public_result_projection_owns_exact_stored_bytes_and_fingerprint(tmp_path):
    q = np.linspace(0.1, 2.0, 7, dtype=np.float64)
    intensity = np.linspace(1.0, 2.0, 7, dtype=np.float64)
    intensity[3] = np.nan
    coverage = np.arange(7, dtype=np.float64)
    normalization = np.linspace(1.0, 7.0, 7, dtype=np.float64)
    projection = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.STITCH_1D,
        axes=(("q", q),),
        axis_units=(("q", "q_A^-1"),),
        intensity=intensity,
        sigma=None,
        coverage=coverage,
        normalization=normalization,
    )
    assert projection.policy == "analysis_artifact_stored_le_f4_v1"
    for values in (
        projection.axes[0][1],
        projection.intensity,
        projection.coverage,
        projection.normalization,
    ):
        assert values.dtype == np.dtype("<f4")
        assert values.flags.c_contiguous
        assert values.flags.writeable is False
        with pytest.raises(ValueError, match="WRITEABLE|writable"):
            values.flags.writeable = True
    assert projection.intensity.view("<u4")[3] == np.uint32(0x7FC00000)
    q[:] = 99.0
    intensity[:] = 99.0
    assert projection.axes[0][1][0] != np.float32(99.0)
    assert np.isnan(projection.intensity[3])
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(projection)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(projection)

    request = _request(tmp_path / "projected-fingerprint.nexus", AnalysisArtifactKind.STITCH_1D)
    receipt = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(
        lambda entry: write_stitched(
            entry,
            result_projection=projection,
            provenance=request.provenance_json,
            bounded_artifact=True,
        )
    )
    assert receipt.inspection.result_fingerprint == projection.result_fingerprint


@pytest.mark.parametrize(
    "q,intensity",
    [
        (np.array([1.0, 1.0 + 1e-9]), np.ones(2)),
        (np.array([1.0, 2.0]), np.array([1.0, np.finfo(np.float64).max])),
        (np.array([1.0, 2.0]), np.array([1.0, np.inf])),
    ],
)
def test_public_result_projection_refuses_unstorable_science(q, intensity):
    with pytest.raises(
        AnalysisArtifactProjectionInvalid,
        match="increasing|overflow|nonfinite",
    ):
        project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_1D,
            axes=(("q", q),),
            axis_units=(("q", "q_A^-1"),),
            intensity=intensity,
            sigma=None,
            coverage=None,
            normalization=None,
        )


@pytest.mark.parametrize(
    "coverage",
    [
        np.array([0.0, 1.5], dtype=np.float64),
        np.array([0.0, float(2**24 + 1)], dtype=np.float64),
        np.array([0.0, 0.5], dtype=np.float32),
        np.array([0.0, float(2**24 + 2)], dtype=np.float32),
        np.array([0, 2**24 + 1], dtype=np.int64),
        np.array([0.0, 0.5], dtype=np.float16),
    ],
)
def test_public_result_projection_refuses_inexact_stitch_counts(coverage):
    with pytest.raises(AnalysisArtifactProjectionInvalid, match="coverage"):
        project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_1D,
            axes=(("q", np.array([1.0, 2.0])),),
            axis_units=(("q", "q_A^-1"),),
            intensity=np.ones(2),
            sigma=None,
            coverage=coverage,
            normalization=np.ones(2),
        )


def test_public_result_projection_matches_axis_unit_and_nan_storage_bounds():
    with pytest.raises(AnalysisArtifactProjectionInvalid, match="bounded"):
        project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_1D,
            axes=(("q", np.linspace(0.0, 1.0, 1_000_001)),),
            axis_units=(("q", "q_A^-1"),),
            intensity=np.ones(1_000_001),
            sigma=None,
            coverage=None,
            normalization=None,
        )

    with pytest.raises(AnalysisArtifactProjectionInvalid, match="rows exceed"):
        project_analysis_artifact_result(
            kind=AnalysisArtifactKind.RSM,
            axes=(
                ("h", np.array([0.0])),
                ("k", np.arange(2049, dtype=np.float64)),
                ("l", np.arange(1024, dtype=np.float64)),
            ),
            axis_units=(("h", None), ("k", None), ("l", None)),
            intensity=np.ones((1, 2049, 1024), dtype=np.float32),
            sigma=None,
            coverage=None,
            normalization=None,
        )
    with pytest.raises(AnalysisArtifactProjectionInvalid, match="axes, units"):
        project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_1D,
            axes=(("q", np.array([1.0, 2.0])),),
            axis_units=(("q", "u" * 4097),),
            intensity=np.ones(2),
            sigma=None,
            coverage=None,
            normalization=None,
        )
    with pytest.raises(AnalysisArtifactProjectionInvalid, match="axes, units"):
        project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_1D,
            axes=(("q", np.array([1.0, 2.0])),),
            axis_units=(("q", "q_A^-1\x00"),),
            intensity=np.ones(2),
            sigma=None,
            coverage=None,
            normalization=None,
        )
    noncanonical_nan = np.array(
        [np.uint32(0x7FC01234), np.uint32(0x3F800000)], dtype="<u4"
    ).view("<f4")
    with pytest.raises(AnalysisArtifactProjectionInvalid, match="noncanonical"):
        project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_1D,
            axes=(("q", np.array([1.0, 2.0])),),
            axis_units=(("q", "q_A^-1"),),
            intensity=noncanonical_nan,
            sigma=None,
            coverage=None,
            normalization=None,
        )


def test_analysis_artifact_v2_round_trip_binds_separate_execution_attestation(
    tmp_path,
):
    projection = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.STITCH_1D,
        axes=(("q", np.linspace(0.1, 2.0, 7)),),
        axis_units=(("q", "q_A^-1"),),
        intensity=np.linspace(1.0, 2.0, 7),
        sigma=None,
        coverage=np.arange(1, 8, dtype=np.float64),
        normalization=np.linspace(1.0, 7.0, 7),
    )
    request_fingerprint = _digest("v2-request")
    attestation = _execution_attestation(
        request_fingerprint,
        projection.result_fingerprint,
    )
    attestation_digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.STITCH_1D,
        attestation,
        request_fingerprint=request_fingerprint,
    )
    request = AnalysisArtifactRequest(
        tmp_path / "artifact-v2.nexus",
        AnalysisArtifactKind.STITCH_1D,
        AnalysisArtifactOverwrite.CREATE_NEW,
        request_fingerprint,
        _digest("v2-source"),
        _digest("v2-plan"),
        _digest("v2-provenance"),
        {"schema_version": "stitch-operation-v2-xu-intent"},
        schema_version=ANALYSIS_SCHEMA_VERSION_V2,
        execution_attestation_digest=attestation_digest,
        execution_attestation=attestation,
    )
    receipt = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(
        lambda entry: write_stitched(
            entry,
            result_projection=projection,
            provenance=request.provenance_json,
            bounded_artifact=True,
        )
    )
    assert receipt.inspection.schema_version == ANALYSIS_SCHEMA_VERSION_V2
    assert receipt.inspection.execution_attestation_digest == attestation_digest
    assert (
        receipt.inspection.execution_attestation_json
        == request.execution_attestation_json
    )
    payload = read_analysis_artifact(request.target, expected_receipt=receipt)
    assert payload.schema_version == ANALYSIS_SCHEMA_VERSION_V2
    assert payload.execution_attestation_digest == attestation_digest
    assert payload.execution_attestation_json == request.execution_attestation_json
    assert payload.result_fingerprint == projection.result_fingerprint
    with h5py.File(request.target, "r") as handle:
        entry = handle["entry"]
        assert int(entry.attrs["ssrl_schema_version"]) == 2
        assert set(entry) == {
            "provenance_json",
            "execution_attestation_json",
            "stitched_1d",
        }
        assert (
            bytes(entry.attrs["execution_attestation_digest"]).decode()
            == attestation_digest
        )


def test_analysis_artifact_v2_rsm_round_trip_is_a_closed_exact_branch(
    tmp_path,
):
    projection = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM,
        axes=(
            ("h", np.linspace(-1.0, 1.0, 3)),
            ("k", np.linspace(-2.0, 2.0, 4)),
            ("l", np.linspace(0.0, 3.0, 5)),
        ),
        axis_units=(("h", None), ("k", None), ("l", None)),
        intensity=np.arange(60, dtype=np.float64).reshape(3, 4, 5),
        sigma=None,
        coverage=None,
        normalization=None,
    )
    request_fingerprint = _digest("rsm-v2-request")
    attestation = _rsm_execution_attestation(
        request_fingerprint,
        projection.result_fingerprint,
    )
    attestation_digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.RSM,
        attestation,
        request_fingerprint=request_fingerprint,
    )
    request = AnalysisArtifactRequest(
        tmp_path / "rsm-artifact-v2.nexus",
        AnalysisArtifactKind.RSM,
        AnalysisArtifactOverwrite.CREATE_NEW,
        request_fingerprint,
        _digest("rsm-v2-source-group"),
        _digest("rsm-v2-plan"),
        _digest("rsm-v2-provenance"),
        {"schema_version": "rsm-operation-v2-intent"},
        schema_version=ANALYSIS_SCHEMA_VERSION_V2,
        execution_attestation_digest=attestation_digest,
        execution_attestation=attestation,
    )
    receipt = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(
        lambda entry: write_rsm(
            entry,
            result_projection=projection,
            provenance=request.provenance_json,
            bounded_artifact=True,
        )
    )
    payload = read_analysis_artifact(request.target, expected_receipt=receipt)
    assert payload.kind is AnalysisArtifactKind.RSM
    assert payload.schema_version == ANALYSIS_SCHEMA_VERSION_V2
    assert payload.execution_attestation_digest == attestation_digest
    assert payload.execution_attestation_json == request.execution_attestation_json
    assert payload.result_fingerprint == projection.result_fingerprint
    assert payload.inspection.axis_units == (
        ("h", None),
        ("k", None),
        ("l", None),
    )
    assert payload.sigma is None
    assert payload.coverage is None
    assert payload.normalization is None
    with h5py.File(request.target, "r") as handle:
        entry = handle["entry"]
        result = entry["rsm"]
        assert set(entry) == {
            "provenance_json",
            "execution_attestation_json",
            "rsm",
        }
        assert set(result) == {"h", "k", "l", "intensity", "provenance_json"}
        assert all(len(result[name].attrs) == 0 for name in ("h", "k", "l"))


@pytest.mark.parametrize(
    "tamper",
    (
        "extra-key",
        "bool-count",
        "release-count",
        "q-count",
        "mask-order",
        "none-digest",
        "static-digest",
        "cropped-shape",
    ),
)
def test_rsm_execution_attestation_refuses_intrinsic_forgery(tamper):
    request_fingerprint = _digest(f"rsm-attestation-{tamper}")
    value = _rsm_execution_attestation(
        request_fingerprint,
        _digest("rsm-attested-result"),
    )
    if tamper == "extra-key":
        value["foreign"] = True
    elif tamper == "bool-count":
        value["selected_scan_count"] = True
    elif tamper == "release-count":
        value["frame_release_check_frame_count"] = 6
    elif tamper == "q-count":
        value["q_release_check_chunk_count"] = 2
    elif tamper == "mask-order":
        value["member_masks"].reverse()
    elif tamper == "none-digest":
        value["member_masks"][0]["full_raw_digest"] = _digest("forged-none")
    elif tamper == "static-digest":
        value["member_masks"][1]["cropped_raw_digest"] = None
    else:
        value["member_masks"][1]["cropped_shape"] = [5, 4]
    with pytest.raises(ValueError, match="attestation contract"):
        analysis_execution_attestation_digest(
            AnalysisArtifactKind.RSM,
            value,
            request_fingerprint=request_fingerprint,
        )


@pytest.mark.parametrize(
    "unit,sigma,diagnostics",
    [
        ("not_q_A^-1", None, True),
        ("q_A^-1", np.linspace(0.1, 0.2, 7), True),
        ("q_A^-1", None, False),
    ],
)
def test_analysis_artifact_v2_refuses_non_xu_result_schema(
    tmp_path,
    unit,
    sigma,
    diagnostics,
):
    projection = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.STITCH_1D,
        axes=(("q", np.linspace(0.1, 2.0, 7)),),
        axis_units=(("q", unit),),
        intensity=np.linspace(1.0, 2.0, 7),
        sigma=sigma,
        coverage=(np.arange(7, dtype=np.float64) if diagnostics else None),
        normalization=(
            np.linspace(1.0, 7.0, 7) if diagnostics else None
        ),
    )
    request_fingerprint = _digest(
        f"v2-invalid-result-{unit}-{sigma is not None}-{diagnostics}"
    )
    attestation = _execution_attestation(
        request_fingerprint,
        projection.result_fingerprint,
    )
    attestation_digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.STITCH_1D,
        attestation,
        request_fingerprint=request_fingerprint,
    )
    target = tmp_path / (
        f"v2-invalid-result-{sigma is not None}-{diagnostics}.nexus"
    )
    request = AnalysisArtifactRequest(
        target,
        AnalysisArtifactKind.STITCH_1D,
        AnalysisArtifactOverwrite.CREATE_NEW,
        request_fingerprint,
        _digest("v2-invalid-source"),
        _digest("v2-invalid-plan"),
        _digest("v2-invalid-provenance"),
        {"schema_version": "stitch-operation-v2-xu-intent"},
        schema_version=ANALYSIS_SCHEMA_VERSION_V2,
        execution_attestation_digest=attestation_digest,
        execution_attestation=attestation,
    )
    with pytest.raises(AnalysisArtifactInvalid, match="exact XU Stitch"):
        admit_analysis_artifact(
            request,
            coordinator=OutputTransactionCoordinator(),
        ).publish(
            lambda entry: write_stitched(
                entry,
                result_projection=projection,
                provenance=request.provenance_json,
                bounded_artifact=True,
            )
        )
    assert not target.exists()


def test_analysis_artifact_versions_refuse_cross_version_attestation_members(
    tmp_path,
):
    request = _request(tmp_path / "artifact-v1-exact.nexus", AnalysisArtifactKind.STITCH_1D)
    receipt = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    assert receipt.inspection.schema_version == 1
    with h5py.File(request.target, "r+") as handle:
        handle["entry"].create_dataset(
            "execution_attestation_json", data=np.bytes_(b"{}")
        )
    with pytest.raises(AnalysisArtifactInvalid, match="unknown entry graph"):
        inspect_analysis_artifact(request.target)

    with pytest.raises(ValueError, match="v1 cannot carry"):
        AnalysisArtifactRequest(
            tmp_path / "invalid-v1.nexus",
            AnalysisArtifactKind.STITCH_1D,
            AnalysisArtifactOverwrite.CREATE_NEW,
            _digest("invalid-v1-request"),
            _digest("invalid-v1-source"),
            _digest("invalid-v1-plan"),
            _digest("invalid-v1-provenance"),
            {},
            execution_attestation_digest=_digest("invalid-v1-attestation"),
            execution_attestation={},
        )


def test_analysis_artifact_v2_refuses_padded_attestation_digest_attribute(
    tmp_path,
):
    projection = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.STITCH_1D,
        axes=(("q", np.linspace(0.1, 2.0, 7)),),
        axis_units=(("q", "q_A^-1"),),
        intensity=np.linspace(1.0, 2.0, 7),
        sigma=None,
        coverage=np.arange(1, 8, dtype=np.float64),
        normalization=np.linspace(1.0, 7.0, 7),
    )
    request_fingerprint = _digest("v2-padded-request")
    attestation = _execution_attestation(
        request_fingerprint,
        projection.result_fingerprint,
    )
    attestation_digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.STITCH_1D,
        attestation,
        request_fingerprint=request_fingerprint,
    )
    request = AnalysisArtifactRequest(
        tmp_path / "artifact-v2-padded-digest.nexus",
        AnalysisArtifactKind.STITCH_1D,
        AnalysisArtifactOverwrite.CREATE_NEW,
        request_fingerprint,
        _digest("v2-padded-source"),
        _digest("v2-padded-plan"),
        _digest("v2-padded-provenance"),
        {"schema_version": "stitch-operation-v2-xu-intent"},
        schema_version=ANALYSIS_SCHEMA_VERSION_V2,
        execution_attestation_digest=attestation_digest,
        execution_attestation=attestation,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(
        lambda entry: write_stitched(
            entry,
            result_projection=projection,
            provenance=request.provenance_json,
            bounded_artifact=True,
        )
    )
    with h5py.File(request.target, "r+") as handle:
        entry = handle["entry"]
        del entry.attrs["execution_attestation_digest"]
        entry.attrs.create(
            "execution_attestation_digest",
            np.array(attestation_digest.encode("utf-8"), dtype="S65"),
        )
    with pytest.raises(AnalysisArtifactInvalid, match="exact digest text"):
        inspect_analysis_artifact(request.target)


def test_stitch_diagnostics_round_trip_and_bind_result_fingerprint(tmp_path):
    q = np.linspace(0.1, 2.0, 7)
    value = IntegrationResult1D(q, np.linspace(1.0, 2.0, 7), None, "q_A^-1")
    coverage = np.linspace(0.0, 6.0, 7)
    normalization = np.linspace(1.0, 7.0, 7)

    def publish(label, coverage_values, normalization_values):
        request = _request(
            tmp_path / f"diagnostics-{label}.nexus",
            AnalysisArtifactKind.STITCH_1D,
        )
        receipt = admit_analysis_artifact(
            request,
            coordinator=OutputTransactionCoordinator(),
        ).publish(
            lambda entry: write_stitched(
                entry,
                stitched_1d=value,
                provenance=request.provenance_json,
                coverage=coverage_values,
                normalization=normalization_values,
                bounded_artifact=True,
            )
        )
        return receipt, read_analysis_artifact(
            request.target,
            expected_receipt=receipt,
        )

    receipt, payload = publish("baseline", coverage, normalization)
    assert receipt.inspection.has_stitch_diagnostics is True
    np.testing.assert_allclose(payload.coverage, coverage.astype(np.float32))
    np.testing.assert_allclose(
        payload.normalization,
        normalization.astype(np.float32),
    )
    assert payload.coverage.flags.writeable is False
    assert payload.normalization.flags.writeable is False

    changed_coverage = coverage.copy()
    changed_coverage[2] += 1.0
    changed_normalization = normalization.copy()
    changed_normalization[3] += 1.0
    coverage_receipt, _ = publish(
        "coverage",
        changed_coverage,
        normalization,
    )
    normalization_receipt, _ = publish(
        "normalization",
        coverage,
        changed_normalization,
    )
    assert coverage_receipt.inspection.result_fingerprint != (
        receipt.inspection.result_fingerprint
    )
    assert normalization_receipt.inspection.result_fingerprint != (
        receipt.inspection.result_fingerprint
    )

    with h5py.File(tmp_path / "unpaired.h5", "w") as handle:
        entry = handle.create_group("entry")
        with pytest.raises(ValueError, match="must be paired"):
            write_stitched(
                entry,
                stitched_1d=value,
                coverage=coverage,
                bounded_artifact=True,
            )
def test_bounded_writer_refuses_unbounded_inputs_before_hdf_mutation(tmp_path):
    q = np.linspace(0.1, 1.0, 7)
    value = IntegrationResult1D(q, q, None, "q_A^-1")
    path = tmp_path / "bounded-writer.h5"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        with pytest.raises(TypeError, match="exact nonempty string"):
            write_stitched(
                entry,
                stitched_1d=value,
                provenance={"not": "canonical text"},
                bounded_artifact=True,
            )
        assert tuple(entry) == ()

        value.unit = "u" * 4097
        with pytest.raises(ValueError, match="bounded artifact size"):
            write_stitched(
                entry,
                stitched_1d=value,
                provenance="{}",
                bounded_artifact=True,
            )
        assert tuple(entry) == ()

        value.unit = "q_A^-1"
        with pytest.raises(ValueError, match="bounded artifact size"):
            write_stitched(
                entry,
                stitched_1d=value,
                provenance="x" * ((1 << 20) + 1),
                bounded_artifact=True,
            )
        assert tuple(entry) == ()


def test_v1_stitch_writer_preserves_accepted_raw_bytes(tmp_path):
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
    provenance = '{"schema_version":"compat-v1","x":1}'

    intensity_1d = np.array(
        [1.0, np.nan, 3.00000006, 4.0, 5.0, 6.0, 7.0],
        dtype=np.float64,
    )
    sigma_1d = np.array(
        [0.1, 0.2, np.nan, 0.4, 0.5, 0.6, 0.7],
        dtype=np.float64,
    )
    coverage_1d = np.arange(7, dtype=np.float64)
    normalization_1d = np.linspace(1, 7, 7, dtype=np.float64)
    value_1d = IntegrationResult1D(
        q,
        intensity_1d,
        sigma_1d,
        "q_A^-1",
    )
    projection_1d = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.STITCH_1D,
        axes=(("q", q),),
        axis_units=(("q", "q_A^-1"),),
        intensity=intensity_1d,
        sigma=sigma_1d,
        coverage=coverage_1d,
        normalization=normalization_1d,
    )

    chi = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    intensity_2d = np.arange(35, dtype=np.float64).reshape(7, 5)
    intensity_2d[2, 3] = np.nan
    sigma_2d = np.linspace(0.1, 3.5, 35).reshape(7, 5)
    coverage_2d = np.arange(35, dtype=np.float64).reshape(7, 5)
    normalization_2d = np.linspace(1.0, 35.0, 35).reshape(7, 5)
    value_2d = IntegrationResult2D(
        q,
        chi,
        intensity_2d,
        sigma_2d,
        "q_A^-1",
        "chi_deg",
    )
    projection_2d = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.STITCH_2D,
        axes=(("q", q), ("chi", chi)),
        axis_units=(("q", "q_A^-1"), ("chi", "chi_deg")),
        intensity=intensity_2d,
        sigma=sigma_2d,
        coverage=coverage_2d,
        normalization=normalization_2d,
    )

    cases = (
        (
            "1d",
            {"stitched_1d": value_1d},
            projection_1d,
            11_432,
            "5be323286c0ffd69b0226f254f0c93e49592de2117e02c57e67788952c8c735e",
        ),
        (
            "2d",
            {"stitched_2d": value_2d},
            projection_2d,
            11_880,
            "b671b55c5c81dc3df85ed2b676c9eead38ee0fc4fcbdbc516b35eee1ac0fdeae",
        ),
    )
    for label, legacy_value, projection, expected_size, expected_sha in cases:
        baseline = tmp_path / f"v1-{label}-baseline.h5"
        projected = tmp_path / f"v1-{label}-projected.h5"
        with h5py.File(baseline, "w") as handle:
            write_stitched(
                handle.create_group("entry"),
                **legacy_value,
                provenance=provenance,
                coverage=(coverage_1d if label == "1d" else coverage_2d),
                normalization=(
                    normalization_1d if label == "1d" else normalization_2d
                ),
                bounded_artifact=True,
            )
        with h5py.File(projected, "w") as handle:
            write_stitched(
                handle.create_group("entry"),
                result_projection=projection,
                legacy_v1_unit_layout=True,
                provenance=provenance,
                bounded_artifact=True,
            )
        baseline_bytes = baseline.read_bytes()
        assert projected.read_bytes() == baseline_bytes
        assert len(baseline_bytes) == expected_size
        assert hashlib.sha256(baseline_bytes).hexdigest() == expected_sha


def test_canonical_provenance_stops_oversized_structure_during_freeze():
    with pytest.raises(ValueError, match="bounded input size"):
        canonical_analysis_provenance({"many": [None] * 300_000})


def test_create_new_refuses_collision_and_replace_is_atomic(tmp_path):
    target = tmp_path / "replace.nexus"
    target.write_bytes(b"prior bytes")
    with pytest.raises(FileExistsError):
        admit_analysis_artifact(
            _request(target, AnalysisArtifactKind.STITCH_1D),
            coordinator=OutputTransactionCoordinator(),
        )
    request = _request(
        target,
        AnalysisArtifactKind.STITCH_1D,
        overwrite=AnalysisArtifactOverwrite.REPLACE,
    )
    receipt = admit_analysis_artifact(
        request, coordinator=OutputTransactionCoordinator()
    ).publish(_writer(request))
    assert receipt.terminal.digest != hashlib.sha256(b"prior bytes").hexdigest()
    assert inspect_analysis_artifact(target).kind is AnalysisArtifactKind.STITCH_1D


def test_create_new_uses_the_admitted_snapshot_not_a_racy_precheck(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "raced-create-new.nexus"
    coordinator = OutputTransactionCoordinator()
    real_admit = coordinator.admit

    def appear_before_admission(*args, **kwargs):
        target.write_bytes(b"foreign arrival")
        return real_admit(*args, **kwargs)

    monkeypatch.setattr(coordinator, "admit", appear_before_admission)
    with pytest.raises(FileExistsError):
        admit_analysis_artifact(
            _request(target, AnalysisArtifactKind.STITCH_1D),
            coordinator=coordinator,
        )
    assert target.read_bytes() == b"foreign arrival"


def test_untouched_abort_releases_normalized_target_lease(tmp_path):
    coordinator = OutputTransactionCoordinator()
    target = tmp_path / "leased.nexus"
    first = admit_analysis_artifact(
        _request(target, AnalysisArtifactKind.STITCH_1D),
        coordinator=coordinator,
    )
    alias = tmp_path / "nested" / ".." / target.name
    with pytest.raises(LeaseUnavailable):
        admit_analysis_artifact(
            _request(alias, AnalysisArtifactKind.STITCH_1D),
            coordinator=coordinator,
        )
    retired = first.abort()
    assert retired.phase is TransactionPhase.ABORTED
    assert retired.remaining_lease_owners == ()
    replacement = admit_analysis_artifact(
        _request(alias, AnalysisArtifactKind.STITCH_1D),
        coordinator=coordinator,
    )
    final = replacement.abort()
    assert final.phase is TransactionPhase.ABORTED
    assert final.remaining_lease_owners == ()


def test_abort_release_fault_is_typed_and_resumes_exact_owner_suffix(
    tmp_path,
    monkeypatch,
):
    coordinator = OutputTransactionCoordinator()
    request = _request(
        tmp_path / "abort-release-retry.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    output = admit_analysis_artifact(request, coordinator=coordinator)
    real_release = coordinator._release
    successful = []
    failed = []

    def fail_after_two(lease, role, owner):
        if len(successful) == 2 and not failed:
            failed.append(role)
            raise OSError("abort release transient")
        snapshot = real_release(lease, role, owner)
        successful.append(role)
        return snapshot

    monkeypatch.setattr(coordinator, "_release", fail_after_two)
    with pytest.raises(AnalysisArtifactCleanupPending) as pending:
        output.abort()
    assert isinstance(pending.value.__cause__, OSError)
    assert pending.value.snapshot.phase is TransactionPhase.ABORTED
    assert pending.value.snapshot.retryable is True
    assert pending.value.snapshot.remaining_lease_owners == tuple(LeaseOwner)[2:]
    recovered = output.retry_cleanup()
    assert recovered.phase is TransactionPhase.ABORTED
    assert recovered.remaining_lease_owners == ()
    assert successful == list(LeaseOwner)


def test_target_appearance_after_admission_is_an_integrity_hold(tmp_path):
    target = tmp_path / "post-admission-arrival.nexus"
    request = _request(target, AnalysisArtifactKind.STITCH_1D)
    output = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    )
    target.write_bytes(b"foreign after admission")
    writes = []
    with pytest.raises(AnalysisArtifactCleanupPending) as pending:
        output.publish(lambda entry: writes.append(entry))
    assert isinstance(pending.value.__cause__, TargetChanged)
    assert writes == []
    assert target.read_bytes() == b"foreign after admission"
    assert output.snapshot.phase is TransactionPhase.INTEGRITY_HOLD
    assert output.snapshot.remaining_lease_owners == tuple(LeaseOwner)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.retry_cleanup()
    assert target.read_bytes() == b"foreign after admission"
    target.unlink()
    recovered = output.retry_cleanup()
    assert recovered.phase is TransactionPhase.ABORTED
    assert recovered.remaining_lease_owners == ()


@pytest.mark.parametrize("prior", (None, b"exact prior"))
def test_writer_or_readback_failure_rolls_back_and_releases(tmp_path, prior):
    target = tmp_path / "rollback.nexus"
    if prior is not None:
        target.write_bytes(prior)
    request = _request(
        target,
        AnalysisArtifactKind.RSM,
        overwrite=(
            AnalysisArtifactOverwrite.REPLACE
            if prior is not None
            else AnalysisArtifactOverwrite.CREATE_NEW
        ),
    )
    output = admit_analysis_artifact(
        request, coordinator=OutputTransactionCoordinator()
    )

    def wrong_kind(entry):
        stitch_request = _request(
            target, AnalysisArtifactKind.STITCH_1D,
            overwrite=AnalysisArtifactOverwrite.REPLACE,
        )
        _writer(stitch_request)(entry)

    with pytest.raises(AnalysisArtifactInvalid):
        output.publish(wrong_kind)
    assert (target.read_bytes() if target.exists() else None) == prior
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()
    assert not tuple(tmp_path.glob(".*xdart-*"))


def test_provenance_is_canonical_bounded_and_strict():
    left = canonical_analysis_provenance({"b": [2, 3], "a": 1})
    right = canonical_analysis_provenance({"a": 1, "b": (2, 3)})
    assert left == right == '{"a":1,"b":[2,3]}'
    with pytest.raises(TypeError):
        canonical_analysis_provenance({1: "bad"})
    with pytest.raises(ValueError):
        canonical_analysis_provenance({"bad": float("nan")})
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError):
        canonical_analysis_provenance({"bad": cyclic})
    nested: object = 0
    for _ in range(63):
        nested = [nested]
    assert canonical_analysis_provenance({"nested": nested})
    nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        canonical_analysis_provenance({"nested": nested})


def test_strict_admission_rejects_indirect_or_ambiguous_result_graph(tmp_path):
    request = _request(tmp_path / "strict.nexus", AnalysisArtifactKind.STITCH_1D)
    output = admit_analysis_artifact(
        request, coordinator=OutputTransactionCoordinator()
    )

    def ambiguous(entry):
        _writer(request)(entry)
        entry["rsm"] = h5py.SoftLink("/entry/stitched_1d")

    with pytest.raises(AnalysisArtifactInvalid):
        output.publish(ambiguous)
    assert not Path(request.target).exists()


def test_marker_only_and_historical_result_groups_are_never_raw(tmp_path):
    cases = [("marker.nexus", True, None)] + [
        (f"historical-{group}.nexus", False, group)
        for group in ("stitched_1d", "stitched_2d", "rsm")
    ]
    for name, marker, group in cases:
        path = tmp_path / name
        with h5py.File(path, "w") as handle:
            entry = handle.create_group("entry")
            if marker:
                entry.attrs[ANALYSIS_SCHEMA_ATTR] = ANALYSIS_SCHEMA_NAME
            if group:
                entry.create_group(group)
        with pytest.raises(ProcessedXdartInputError):
            require_raw_input(path)


def test_committed_cleanup_retry_seals_receipt_without_writer_replay(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.io.output_transaction as transaction_api

    target = tmp_path / "cleanup-retry.nexus"
    target.write_bytes(b"prior")
    request = _request(
        target,
        AnalysisArtifactKind.STITCH_1D,
        overwrite=AnalysisArtifactOverwrite.REPLACE,
    )
    output = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    )
    backup = output._transaction.backup
    real_unlink = transaction_api._unlink
    failed = []
    writes = []

    def fail_backup_once(path):
        if Path(path) == backup and not failed:
            failed.append("backup")
            raise OSError("backup unlink fault")
        return real_unlink(path)

    def write_once(entry):
        writes.append("writer")
        _writer(request)(entry)

    monkeypatch.setattr(transaction_api, "_unlink", fail_backup_once)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(write_once)
    pending = output.snapshot
    assert pending.phase is TransactionPhase.CLEANUP_PENDING
    assert pending.receipt is None
    assert pending.remaining_lease_owners == tuple(LeaseOwner)

    recovered = output.retry_cleanup()
    assert recovered.phase is TransactionPhase.COMMITTED
    assert recovered.receipt is not None
    assert recovered.receipt.request is request
    assert recovered.receipt.inspection.result_fingerprint
    assert recovered.remaining_lease_owners == ()
    assert writes == ["writer"]


def test_transient_final_readback_retries_without_writer_replay(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.io.analysis_artifact as artifact_api

    request = _request(
        tmp_path / "readback-retry.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    output = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    )
    real_inspect = artifact_api.inspect_analysis_artifact
    failed = []
    writes = []

    def fail_final_once(path, **kwargs):
        if Path(path) == Path(request.target) and not failed:
            failed.append("final")
            raise OSError("final readback transient")
        return real_inspect(path, **kwargs)

    def write_once(entry):
        writes.append("writer")
        _writer(request)(entry)

    monkeypatch.setattr(artifact_api, "inspect_analysis_artifact", fail_final_once)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(write_once)
    assert output.snapshot.phase is TransactionPhase.COMMITTED
    assert output.snapshot.receipt is None
    assert output.snapshot.remaining_lease_owners == tuple(LeaseOwner)

    recovered = output.retry_cleanup()
    assert recovered.receipt is not None
    assert recovered.remaining_lease_owners == ()
    assert writes == ["writer"]


def test_invalid_candidate_and_cleanup_fault_preserve_primary_then_retry(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.io.output_transaction as transaction_api

    request = _request(
        tmp_path / "invalid-cleanup-retry.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    output = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    )
    real_unlink = transaction_api._unlink
    cleanup_failures = []
    writes = []

    def fail_candidate_cleanup_once(path):
        if ".xdart-candidate-" in Path(path).name and not cleanup_failures:
            cleanup_failures.append("candidate")
            raise OSError("candidate cleanup fault")
        return real_unlink(path)

    def invalid_writer(entry):
        writes.append("writer")
        _writer(request)(entry)
        entry.attrs["unexpected"] = "invalid"

    monkeypatch.setattr(transaction_api, "_unlink", fail_candidate_cleanup_once)
    with pytest.raises(AnalysisArtifactCleanupPending) as pending:
        output.publish(invalid_writer)
    assert isinstance(pending.value.__cause__, AnalysisArtifactInvalid)
    assert output.snapshot.phase is TransactionPhase.CLEANUP_PENDING
    assert output.snapshot.remaining_lease_owners == tuple(LeaseOwner)
    recovered = output.retry_cleanup()
    assert recovered.phase is TransactionPhase.ABORTED
    assert recovered.remaining_lease_owners == ()
    assert not Path(request.target).exists()
    assert writes == ["writer"]


def test_committed_readback_retry_refuses_changed_target(tmp_path, monkeypatch):
    import xrd_tools.io.analysis_artifact as artifact_api

    request = _request(
        tmp_path / "changed-before-readback-retry.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    output = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    )
    real_inspect = artifact_api.inspect_analysis_artifact
    failed = []

    def fail_final_once(path, **kwargs):
        if Path(path) == Path(request.target) and not failed:
            failed.append("final")
            raise OSError("final readback transient")
        return real_inspect(path, **kwargs)

    monkeypatch.setattr(artifact_api, "inspect_analysis_artifact", fail_final_once)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(_writer(request))
    assert output.snapshot.phase is TransactionPhase.COMMITTED
    assert output.snapshot.receipt is None
    Path(request.target).write_bytes(b"foreign committed occupant")
    with pytest.raises(AnalysisArtifactCleanupPending) as pending:
        output.retry_cleanup()
    assert pending.value.snapshot.receipt is None
    assert pending.value.snapshot.remaining_lease_owners == tuple(LeaseOwner)
    assert Path(request.target).read_bytes() == b"foreign committed occupant"


def test_detached_inspection_refuses_path_replacement_during_read(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.io.analysis_artifact as artifact_api

    request = _request(
        tmp_path / "detached-read-race.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    foreign = tmp_path / "foreign-replacement.nexus"
    foreign.write_bytes(b"foreign")
    real_axis = artifact_api._axis
    replaced = []

    def replace_during_axis(group, name):
        values = real_axis(group, name)
        if not replaced:
            replaced.append(name)
            os.replace(foreign, request.target)
        return values

    monkeypatch.setattr(artifact_api, "_axis", replace_during_axis)
    with pytest.raises(AnalysisArtifactInvalid, match="changed during"):
        inspect_analysis_artifact(request.target)
    assert Path(request.target).read_bytes() == b"foreign"


def test_payload_reader_refuses_path_replacement_during_materialization(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.io.analysis_artifact as artifact_api

    request = _request(
        tmp_path / "payload-read-race.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    receipt = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    foreign = tmp_path / "foreign-payload-replacement.nexus"
    foreign.write_bytes(b"foreign")
    real_materialize = artifact_api._materialize_bounded_values
    replaced = []

    def replace_during_materialization(dataset):
        values = real_materialize(dataset)
        if not replaced:
            replaced.append(dataset.name)
            os.replace(foreign, request.target)
        return values

    monkeypatch.setattr(
        artifact_api,
        "_materialize_bounded_values",
        replace_during_materialization,
    )
    with pytest.raises(AnalysisArtifactInvalid, match="changed during payload"):
        read_analysis_artifact(request.target, expected_receipt=receipt)
    assert replaced == ["/entry/stitched_1d/intensity"]
    assert Path(request.target).read_bytes() == b"foreign"


def test_detached_inspection_binds_the_opened_hdf_handle_during_symlink_swap(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.io.analysis_artifact as artifact_api

    request = _request(
        tmp_path / "symlink-source-a.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    source_a = Path(request.target)
    source_b = tmp_path / "symlink-source-b.nexus"
    shutil.copy2(source_a, source_b)
    alias = tmp_path / "switched-link.nexus"
    alias.symlink_to(source_a)
    for source in (source_a, source_b):
        with h5py.File(source, "r+") as handle:
            entry = handle["entry"]
            del entry.attrs["file_name"]
            entry.attrs.create(
                "file_name",
                np.bytes_(str(alias).encode("utf-8")),
            )
    with h5py.File(source_b, "r+") as handle:
        handle["entry/stitched_1d/intensity"][3] += 10.0

    baseline = inspect_analysis_artifact(alias)
    real_file = artifact_api.h5py.File

    def switch_only_while_opening(path, mode="r", *args, **kwargs):
        if Path(path) == alias and mode == "r":
            alias.unlink()
            alias.symlink_to(source_b)
            opened = real_file(path, mode, *args, **kwargs)
            alias.unlink()
            alias.symlink_to(source_a)
            return opened
        return real_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(artifact_api.h5py, "File", switch_only_while_opening)
    with pytest.raises(AnalysisArtifactInvalid, match="changed while opening"):
        inspect_analysis_artifact(alias)
    assert alias.stat().st_ino == baseline.storage_revision[1]


@pytest.mark.parametrize("kind", ("oversized", "directory"))
def test_detached_inspection_refuses_nonregular_or_oversized_storage(
    tmp_path,
    kind,
):
    path = tmp_path / f"{kind}.nexus"
    if kind == "directory":
        path.mkdir()
    else:
        with path.open("wb") as handle:
            handle.truncate((1 << 30) + 1)
    with pytest.raises(AnalysisArtifactInvalid, match="regular bounded file"):
        inspect_analysis_artifact(path)


def test_expected_request_requires_exact_path_and_consistent_kind(tmp_path):
    request = _request(
        tmp_path / "exact-request-path.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    alias = tmp_path / "hard-link-alias.nexus"
    os.link(request.target, alias)
    with pytest.raises(AnalysisArtifactInvalid, match="path does not match"):
        inspect_analysis_artifact(alias, expected_request=request)
    with pytest.raises(AnalysisArtifactInvalid, match="path does not match"):
        read_analysis_artifact(alias, expected_request=request)
    with pytest.raises(ValueError, match="conflicts"):
        inspect_analysis_artifact(
            request.target,
            expected_request=request,
            expected_kind=AnalysisArtifactKind.RSM,
        )
    other_request = _request(
        tmp_path / "other-receipt.nexus",
        AnalysisArtifactKind.RSM,
    )
    receipt = admit_analysis_artifact(
        other_request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(other_request))
    with pytest.raises(ValueError, match="conflicts"):
        read_analysis_artifact(
            request.target,
            expected_request=request,
            expected_receipt=receipt,
        )


def test_cached_receipt_retry_revalidates_before_final_lease_release(
    tmp_path,
    monkeypatch,
):
    coordinator = OutputTransactionCoordinator()
    request = _request(
        tmp_path / "cached-receipt-release.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    output = admit_analysis_artifact(request, coordinator=coordinator)
    real_release = coordinator._release
    failed = []

    def fail_release_once(*args, **kwargs):
        if not failed:
            failed.append("release")
            raise OSError("lease release transient")
        return real_release(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_release", fail_release_once)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(_writer(request))
    pending = output.snapshot
    assert pending.phase is TransactionPhase.COMMITTED
    assert pending.receipt is not None
    assert pending.remaining_lease_owners == tuple(LeaseOwner)
    target = Path(request.target)
    replacement = target.with_name("foreign-cached.nexus")
    replacement.write_bytes(b"foreign replacement")
    os.replace(replacement, target)
    with pytest.raises(AnalysisArtifactCleanupPending) as held:
        output.retry_cleanup()
    assert isinstance(held.value.__cause__, TargetChanged)
    assert held.value.snapshot.receipt is pending.receipt
    assert held.value.snapshot.remaining_lease_owners == pending.remaining_lease_owners
    assert target.read_bytes() == b"foreign replacement"


def test_release_retry_resumes_the_exact_remaining_owner_suffix(
    tmp_path,
    monkeypatch,
):
    coordinator = OutputTransactionCoordinator()
    request = _request(
        tmp_path / "release-suffix.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    output = admit_analysis_artifact(request, coordinator=coordinator)
    real_release = coordinator._release
    successful = []
    failed = []

    def fail_after_two(lease, role, owner):
        if len(successful) == 2 and not failed:
            failed.append(role)
            raise OSError("third owner release transient")
        snapshot = real_release(lease, role, owner)
        successful.append(role)
        return snapshot

    monkeypatch.setattr(coordinator, "_release", fail_after_two)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(_writer(request))
    pending = output.snapshot
    assert pending.receipt is not None
    assert pending.remaining_lease_owners == tuple(LeaseOwner)[2:]
    recovered = output.retry_cleanup()
    assert recovered.receipt is pending.receipt
    assert recovered.remaining_lease_owners == ()
    assert successful == list(LeaseOwner)


@pytest.mark.parametrize(
    "tamper",
    (
        "signal",
        "axes",
        "extra",
        "file-name",
        "axis-cap",
        "q-units",
        "result-attr",
        "q-attr",
        "intensity-attr",
    ),
)
def test_detached_admission_rejects_nonexact_schema(tmp_path, tamper):
    request = _request(
        tmp_path / f"strict-{tamper}.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    with h5py.File(request.target, "r+") as handle:
        entry = handle["entry"]
        result = entry["stitched_1d"]
        if tamper == "signal":
            result.attrs["signal"] = "q"
        elif tamper == "axes":
            result.attrs["axes"] = ["wrong"]
        elif tamper == "extra":
            result.create_dataset("foreign", data=np.ones(1, dtype=np.float32))
        elif tamper == "file-name":
            entry.attrs["file_name"] = str(tmp_path / "foreign.nexus")
        elif tamper == "q-units":
            del result["q"].attrs["units"]
            result["q"].attrs.create("units", np.bytes_(b""))
        elif tamper == "result-attr":
            result.attrs["foreign"] = 1
        elif tamper == "q-attr":
            result["q"].attrs["foreign"] = 1
        elif tamper == "intensity-attr":
            result["intensity"].attrs["foreign"] = 1
        else:
            del result["q"]
            axis = result.create_dataset("q", shape=(1_000_001,), dtype=np.float32)
            axis.attrs["units"] = "q_A^-1"
    with pytest.raises(AnalysisArtifactInvalid):
        inspect_analysis_artifact(request.target)


@pytest.mark.parametrize(
    "tamper",
    (
        "decreasing-axis",
        "float64-intensity",
        "infinite-intensity",
        "all-nan-intensity",
        "wrong-sigma-shape",
        "result-element-cap",
        "contiguous-row-cap",
    ),
)
def test_detached_admission_enforces_numeric_schema_bounds(tmp_path, tamper):
    kind = (
        AnalysisArtifactKind.RSM
        if tamper in {"result-element-cap", "contiguous-row-cap"}
        else AnalysisArtifactKind.STITCH_1D
    )
    request = _request(tmp_path / f"numeric-{tamper}.nexus", kind)
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    with h5py.File(request.target, "r+") as handle:
        result = handle[f"entry/{kind.group}"]
        if tamper == "decreasing-axis":
            values = result["q"][()]
            values[3] = values[2] - 0.01
            result["q"][...] = values
        elif tamper == "float64-intensity":
            values = result["intensity"][()].astype(np.float64)
            del result["intensity"]
            result.create_dataset("intensity", data=values)
        elif tamper == "infinite-intensity":
            result["intensity"][3] = np.inf
        elif tamper == "all-nan-intensity":
            result["intensity"][...] = np.nan
        elif tamper == "wrong-sigma-shape":
            del result["sigma"]
            result.create_dataset(
                "sigma",
                data=np.ones(6, dtype=np.float32),
            )
        else:
            shape = (
                (401, 401, 400)
                if tamper == "result-element-cap"
                else (2, 2049, 1024)
            )
            for name, length in zip(("h", "k", "l"), shape, strict=True):
                del result[name]
                result.create_dataset(
                    name,
                    data=np.arange(length, dtype=np.float32),
                )
            del result["intensity"]
            options = (
                {"chunks": (1, 1, shape[2]), "fillvalue": 0.0}
                if tamper == "result-element-cap"
                else {"fillvalue": 0.0}
            )
            result.create_dataset(
                "intensity",
                shape=shape,
                dtype=np.float32,
                **options,
            )
    with pytest.raises(AnalysisArtifactInvalid):
        inspect_analysis_artifact(request.target)


def test_detached_admission_rejects_vlen_schema_attribute_before_read(tmp_path):
    request = _request(
        tmp_path / "vlen-attribute.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    with h5py.File(request.target, "r+") as handle:
        entry = handle["entry"]
        del entry.attrs[ANALYSIS_KIND_ATTR]
        entry.attrs[ANALYSIS_KIND_ATTR] = request.kind.value
    with pytest.raises(AnalysisArtifactInvalid, match="bounded fixed text"):
        inspect_analysis_artifact(request.target)


@pytest.mark.parametrize(
    "tamper",
    ("kind-opaque", "axes-opaque", "provenance-opaque", "version-enum"),
)
def test_detached_admission_requires_exact_hdf5_type_classes(tmp_path, tamper):
    request = _request(
        tmp_path / f"hdf-type-{tamper}.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    with h5py.File(request.target, "r+") as handle:
        entry = handle["entry"]
        result = entry["stitched_1d"]
        if tamper == "kind-opaque":
            del entry.attrs[ANALYSIS_KIND_ATTR]
            entry.attrs.create(
                ANALYSIS_KIND_ATTR,
                np.bytes_(b"stitch-1d"),
                dtype=h5py.opaque_dtype(np.dtype("S9")),
            )
        elif tamper == "axes-opaque":
            del result.attrs["axes"]
            result.attrs.create(
                "axes",
                np.asarray([b"q"], dtype="S1"),
                dtype=h5py.opaque_dtype(np.dtype("S1")),
            )
        elif tamper == "provenance-opaque":
            raw = request.provenance_json.encode("utf-8")
            del entry["provenance_json"]
            entry.create_dataset(
                "provenance_json",
                data=np.bytes_(raw),
                dtype=h5py.opaque_dtype(np.dtype(f"S{len(raw)}")),
            )
        else:
            enum = h5py.enum_dtype({"ONE": 1}, basetype="u1")
            del entry.attrs["ssrl_schema_version"]
            entry.attrs.create(
                "ssrl_schema_version",
                np.uint8(1),
                dtype=enum,
            )
    with pytest.raises(AnalysisArtifactInvalid):
        inspect_analysis_artifact(request.target)


@pytest.mark.parametrize("location", ("entry", "result"))
def test_detached_admission_rejects_vlen_provenance_before_payload_read(
    tmp_path,
    location,
):
    request = _request(
        tmp_path / "vlen-provenance.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    receipt = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    with pytest.raises(TypeError):
        AnalysisArtifactReceipt(
            receipt.request,
            receipt.terminal,
            receipt.inspection,
        )
    with h5py.File(request.target, "r+") as handle:
        parent = (
            handle["entry"]
            if location == "entry"
            else handle["entry/stitched_1d"]
        )
        del parent["provenance_json"]
        parent.create_dataset(
            "provenance_json",
            data=request.provenance_json,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    with pytest.raises(AnalysisArtifactInvalid, match="fixed-width storage"):
        inspect_analysis_artifact(request.target)


def test_detached_admission_rejects_oversized_fixed_provenance(tmp_path):
    request = _request(
        tmp_path / "oversized-provenance.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    oversized = b"x" * ((1 << 20) + 1)
    with h5py.File(request.target, "r+") as handle:
        entry = handle["entry"]
        del entry["provenance_json"]
        entry.create_dataset("provenance_json", data=np.bytes_(oversized))
    with pytest.raises(AnalysisArtifactInvalid, match="provenance is oversized"):
        inspect_analysis_artifact(request.target)


def test_detached_admission_normalizes_deep_json_to_schema_refusal(tmp_path):
    request = _request(
        tmp_path / "deep-provenance.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    nested = '{"nested":' + ("[" * 65) + "0" + ("]" * 65) + "}"
    with h5py.File(request.target, "r+") as handle:
        entry = handle["entry"]
        del entry["provenance_json"]
        entry.create_dataset(
            "provenance_json",
            data=np.bytes_(nested.encode("utf-8")),
        )
    with pytest.raises(AnalysisArtifactInvalid, match="nesting"):
        inspect_analysis_artifact(request.target)


def test_detached_admission_rejects_oversized_compressed_chunk(tmp_path):
    request = _request(
        tmp_path / "oversized-chunk.nexus",
        AnalysisArtifactKind.RSM,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    axis_size = 1_000_000
    with h5py.File(request.target, "r+") as handle:
        result = handle["entry/rsm"]
        del result["l"]
        result.create_dataset(
            "l",
            data=np.linspace(0.0, 1.0, axis_size, dtype=np.float32),
        )
        del result["intensity"]
        result.create_dataset(
            "intensity",
            shape=(3, 4, axis_size),
            chunks=(3, 4, axis_size),
            dtype=np.float32,
            fillvalue=0.0,
            compression="gzip",
        )
    with pytest.raises(AnalysisArtifactInvalid, match="chunks exceed"):
        inspect_analysis_artifact(request.target)


def test_detached_admission_rejects_oversized_axis_chunk(tmp_path):
    request = _request(
        tmp_path / "oversized-axis-chunk.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    with h5py.File(request.target, "r+") as handle:
        result = handle["entry/stitched_1d"]
        del result["q"]
        q = result.create_dataset(
            "q",
            shape=(7,),
            maxshape=(None,),
            chunks=(3_000_000,),
            dtype=np.float32,
            fillvalue=0.0,
        )
        q.attrs["units"] = "q_A^-1"
    with pytest.raises(AnalysisArtifactInvalid, match="chunks exceed"):
        inspect_analysis_artifact(request.target)


@pytest.mark.parametrize(
    "indirection",
    ("soft", "alias", "vds", "external", "external-link"),
)
def test_detached_admission_rejects_indirect_and_aliased_payloads(
    tmp_path,
    indirection,
):
    request = _request(
        tmp_path / f"indirect-{indirection}.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    ).publish(_writer(request))
    with h5py.File(request.target, "r+") as handle:
        result = handle["entry/stitched_1d"]
        if indirection == "soft":
            del result["q"]
            result["q"] = h5py.SoftLink("/entry/stitched_1d/intensity")
        elif indirection == "alias":
            del result["provenance_json"]
            result["provenance_json"] = handle["entry/provenance_json"]
        elif indirection == "vds":
            source = tmp_path / "vds-source.h5"
            with h5py.File(source, "w") as source_handle:
                source_handle.create_dataset(
                    "values",
                    data=np.arange(7, dtype=np.float32),
                )
            del result["intensity"]
            layout = h5py.VirtualLayout(shape=(7,), dtype=np.float32)
            layout[:] = h5py.VirtualSource(str(source), "values", shape=(7,))
            result.create_virtual_dataset("intensity", layout)
        elif indirection == "external":
            external = tmp_path / "external.bin"
            external.write_bytes(np.arange(7, dtype=np.float32).tobytes())
            del result["intensity"]
            result.create_dataset(
                "intensity",
                shape=(7,),
                dtype=np.float32,
                external=[(str(external), 0, 7 * 4)],
            )
        else:
            external = tmp_path / "external-link.h5"
            with h5py.File(external, "w") as external_handle:
                external_handle.create_dataset(
                    "q",
                    data=np.linspace(0.1, 2.0, 7, dtype=np.float32),
                )
            del result["q"]
            result["q"] = h5py.ExternalLink(str(external), "/q")
    match = "hard-link alias" if indirection == "alias" else None
    with pytest.raises(AnalysisArtifactInvalid, match=match):
        inspect_analysis_artifact(request.target)


def test_unbound_frame_records_and_source_base_are_refused(tmp_path):
    source = tmp_path / "raw" / "frame.tif"
    source.parent.mkdir()
    source.write_bytes(b"raw")
    request = _request(
        tmp_path / "frames.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    value = IntegrationResult1D(
        np.linspace(0.1, 1.0, 7),
        np.linspace(1.0, 2.0, 7),
        None,
        "q_A^-1",
    )
    output = admit_analysis_artifact(
        request,
        coordinator=OutputTransactionCoordinator(),
    )
    with pytest.raises(ValueError, match="unbound frame records"):
        output.publish(
            lambda entry: write_stitched(
                entry,
                stitched_1d=value,
                provenance=request.provenance_json,
                bounded_artifact=True,
                frame_records=[{
                    "frame_index": 1,
                    "source_path": source,
                    "source_frame_index": 0,
                }],
                source_base=tmp_path,
            )
        )
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()
    assert not Path(request.target).exists()
