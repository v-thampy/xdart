from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import os
from pathlib import Path
import pickle
from threading import Event

import numpy as np
import pytest
import tifffile

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleArtifactRefused,
    ModuleArtifactOutputSnapshot,
    ModuleCommitReceipt,
    ModuleDisposition,
    ModuleKind,
    ModuleOperationRequest,
    ModuleOutputRequest,
    ModuleProgress,
    ModuleSourceGroupReceipt,
    ModuleSourceReceipt,
    ModuleTerminalResult,
    admit_module_artifact,
    module_artifact_request,
    module_plan_fingerprint,
    module_provenance_digest,
    _rsm_v2_module_request,
    xu_stitch_module_request,
)
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    AnalysisSourceLeaseRefused,
    MetadataTablePlan,
    analysis_canonical_fingerprint,
    requalified_analysis_source,
    run_metadata_table,
)
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactCleanupPending,
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
    AnalysisArtifactRequest,
    admit_analysis_artifact,
    analysis_execution_attestation_digest,
    project_analysis_artifact_result,
)
from xrd_tools.io.output_transaction import StreamTerminal, TargetChanged
from xrd_tools.io.output_transaction import OutputTransactionCoordinator
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.nexus import write_rsm, write_stitched
from xrd_tools.rsm.volume import RSMVolume
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from xrd_tools.sources.selection import image_series_spec


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _table(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    image = tmp_path / "scan_0001.tif"
    tifffile.imwrite(image, np.arange(12, dtype=np.uint16).reshape(3, 4))
    result = run_metadata_table(
        MetadataTablePlan(image_series_spec(image, metadata_format=None))
    )
    assert result.disposition is AnalysisDisposition.COMPLETED
    return image, result


def _multi_table(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    images = tuple(tmp_path / f"scan_{index:04d}.tif" for index in (1, 2, 3))
    for index, image in enumerate(images):
        tifffile.imwrite(
            image,
            np.arange(12, dtype=np.uint16).reshape(3, 4) + index,
        )
    result = run_metadata_table(
        MetadataTablePlan(image_series_spec(images[0], metadata_format=None))
    )
    assert result.disposition is AnalysisDisposition.COMPLETED
    assert len(result.labels) == 3
    return images, result


def _many_table(tmp_path: Path, count: int):
    tmp_path.mkdir(parents=True, exist_ok=True)
    images = tuple(tmp_path / f"scan_{index:04d}.tif" for index in range(count))
    for index, image in enumerate(images):
        tifffile.imwrite(image, np.full((2, 2), index, dtype=np.uint16))
    result = run_metadata_table(
        MetadataTablePlan(image_series_spec(images[0], metadata_format=None))
    )
    assert result.disposition is AnalysisDisposition.COMPLETED
    assert len(result.labels) == count
    return images, result


def _request(
    tmp_path: Path,
    kind=ModuleKind.STITCH,
    artifact_kind: AnalysisArtifactKind | None = None,
):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    _image, table = _table(source_root)
    source = ModuleSourceReceipt.from_metadata_table(table, kind=kind)
    output = ModuleOutputRequest(
        output_root / "result.nexus",
        artifact_kind
        if artifact_kind is not None
        else (
            AnalysisArtifactKind.STITCH_1D
            if kind is ModuleKind.STITCH
            else AnalysisArtifactKind.RSM
        ),
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    provenance = {
        "kind": kind.value,
        "frames": list(source.selected_labels),
    }
    return ModuleOperationRequest(
        source,
        output,
        module_plan_fingerprint(kind, ("plan", 1)),
        module_provenance_digest(kind, provenance),
    )


def _provenance(request: ModuleOperationRequest) -> dict[str, object]:
    return {
        "kind": request.kind.value,
        "frames": list(request.source.selected_labels),
    }


def _xu_request(tmp_path: Path):
    base = _request(tmp_path)
    asset_receipt = _digest("xu-asset-receipt")
    effective = _digest("xu-effective-geometry")
    provenance = {
        "schema_version": "stitch-operation-v2-xu-intent",
        "kind": "stitch",
        "backend": "xu_hist",
        "source": {
            "source_fingerprint": base.source.analysis.source_fingerprint,
            "module_source_fingerprint": base.source.fingerprint,
            "metadata_table_fingerprint": base.source.table_fingerprint,
            "selected_labels": list(base.source.selected_labels),
            "input_manifest": {"fingerprint": _digest("xu-manifest")},
        },
        "asset": {
            "lexical_relative_path": "calibration/xu/surface.json",
            "resolved_relative_path": "calibration/xu/surface.json",
            "byte_count": 4837,
            "raw_sha256": _digest("xu-raw"),
            "semantic_fingerprint": _digest("xu-semantic"),
            "receipt_fingerprint": asset_receipt,
        },
        "effective_geometry": {"fingerprint": effective},
        "detector": {"type": "Pilatus300kw"},
        "corrections": {"policy": "surface-v1"},
        "plan": {
            "backend": "xu_hist",
            "mode": "1d",
            "unit": "q_A^-1",
            "method": "numpy_histogram_center_v1",
            "radial_range": [1.0, 5.2],
            "npt_1d": 8,
            "monitor_selector": None,
            "use_detector_mask": True,
            "max_frame_bytes": 4 * 1024 * 1024,
            "asset_receipt_fingerprint": asset_receipt,
            "effective_geometry_fingerprint": effective,
            "plan_fingerprint": base.plan_fingerprint,
        },
        "observations": {"frame_count": len(base.source.selected_labels)},
        "runtime_requirements": {"lock_policy": "shared_xu_lock_v1"},
        "output": {
            "target": base.output.target,
            "kind": base.output.kind.value,
            "overwrite": base.output.overwrite.value,
            "output_fingerprint": base.output.fingerprint,
        },
        "holds": ["generic-xu-geometries"],
    }
    request = xu_stitch_module_request(
        base.source,
        base.output,
        base.plan_fingerprint,
        module_provenance_digest(ModuleKind.STITCH, provenance),
    )
    return request, provenance


def _attestation(request: ModuleOperationRequest, result_fingerprint: str):
    count = len(request.source.selected_labels)
    return {
        "schema_version": "analysis-execution-attestation-v1",
        "module_request_fingerprint": request.fingerprint,
        "result_projection_policy": "analysis_artifact_stored_le_f4_v1",
        "result_fingerprint": result_fingerprint,
        "selected_frame_count": count,
        "release_check_frame_count": count,
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


def _rsm_attestation(
    request: ModuleOperationRequest,
    provenance: dict[str, object],
    result_fingerprint: str,
    *,
    mask_receipts: tuple[object, ...] | None = None,
):
    members = provenance["members"]
    frame_count = request.source.selected_frame_count
    chunk_size = provenance["plan"]["chunk_size"]
    chunk_count = sum(
        (len(member["contributions"]) + chunk_size - 1) // chunk_size
        for member in members
    )
    return {
        "schema_version": "rsm-execution-attestation-v2",
        "module_request_fingerprint": request.fingerprint,
        "result_projection_policy": "analysis_artifact_stored_le_f4_v1",
        "result_fingerprint": result_fingerprint,
        "geometry_asset_receipt_fingerprint": provenance["asset"][
            "receipt_fingerprint"
        ],
        "effective_geometry_fingerprint": provenance["effective_geometry"][
            "fingerprint"
        ],
        "common_grid_fingerprint": provenance["common_grid"]["fingerprint"],
        "selected_scan_count": len(request.source.members),
        "selected_frame_count": frame_count,
        "science_chunk_count": chunk_count,
        "q_release_check_chunk_count": chunk_count,
        "frame_release_check_frame_count": frame_count,
        "release_check_passed": True,
        "member_masks": (
            [
                {
                    "ordinal": ordinal,
                    "member_preflight_fingerprint": member["fingerprint"],
                    "mask_policy": member["mask_policy_intent"][0],
                    "full_shape": member["detector_shape"],
                    "full_raw_digest": None,
                    "full_masked_pixel_count": 0,
                    "cropped_shape": member["cropped_shape"],
                    "cropped_raw_digest": None,
                    "cropped_masked_pixel_count": 0,
                    "mask_receipt_fingerprint": _digest(
                        f"rsm-mask-receipt-{ordinal}"
                    ),
                }
                for ordinal, member in enumerate(members)
            ]
            if mask_receipts is None
            else [
                receipt.to_attestation(ordinal)
                for ordinal, receipt in enumerate(mask_receipts)
            ]
        ),
        "coordinate_frame": provenance["coordinate_frame"]["name"],
        "axis_names": provenance["coordinate_frame"]["axis_names"],
        "axis_units": provenance["coordinate_frame"]["axis_units"],
        "matrix_policy": provenance["coordinate_frame"]["matrix_policy"],
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


def test_public_canonical_fingerprint_is_type_framed_and_domain_separated():
    assert analysis_canonical_fingerprint("x", {"b": 2, "a": 1}) == (
        analysis_canonical_fingerprint("x", {"a": 1, "b": 2})
    )
    assert analysis_canonical_fingerprint("x", [1, 2]) != (
        analysis_canonical_fingerprint("x", (1, 2))
    )
    assert analysis_canonical_fingerprint("x", Path("a")) != (
        analysis_canonical_fingerprint("x", "a")
    )
    assert analysis_canonical_fingerprint("x", 1) != (
        analysis_canonical_fingerprint("x", True)
    )
    assert analysis_canonical_fingerprint("x", (1,)) != (
        analysis_canonical_fingerprint("y", (1,))
    )
    with pytest.raises(ValueError):
        analysis_canonical_fingerprint("x", float("nan"))
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError):
        analysis_canonical_fingerprint("x", cyclic)


def test_module_values_freeze_content_but_keep_runtime_authority_exact(tmp_path):
    request = _request(tmp_path)
    assert request.source.kind is ModuleKind.STITCH
    assert request.output.kind is AnalysisArtifactKind.STITCH_1D
    assert request.output.target == os.path.abspath(tmp_path / "output/result.nexus")
    assert len(request.source.fingerprint) == len(request.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        request.plan_fingerprint = _digest("changed")
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(request)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(request)
    with pytest.raises(TypeError, match="not replaceable"):
        copy.replace(request)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(request)

    seconds_0 = MetadataColumnSelector("Seconds", 0)
    seconds_1 = MetadataColumnSelector("Seconds", 1)
    assert seconds_0 != seconds_1
    with pytest.raises(TypeError):
        MetadataColumnSelector("Seconds", True)
    with pytest.raises(TypeError):
        MetadataColumnSelector(" Seconds", 0)

    foreign_output = ModuleOutputRequest(
        request.output.target,
        request.output.kind,
        request.output.overwrite,
    )
    terminal = StreamTerminal(
        request.output.target,
        1,
        _digest("bytes"),
        1,
        1,
        2,
        3,
        4,
    )
    with pytest.raises(TypeError):
        ModuleCommitReceipt(request, foreign_output, terminal, _digest("result"))
    with pytest.raises(TypeError):
        ModuleCommitReceipt(request, request.output, terminal, _digest("result"))
    with pytest.raises(TypeError):
        ModuleTerminalResult(request, ModuleDisposition.COMMITTED, "OK")
    assert ModuleProgress(request, 1, "prepare", 0, 3).request is request


def test_source_and_output_kind_must_match(tmp_path):
    _image, table = _table(tmp_path)
    source = ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.RSM)
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(source)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(source)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(source)
    output = ModuleOutputRequest(
        tmp_path / "bad.nexus",
        AnalysisArtifactKind.STITCH_1D,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    with pytest.raises(ValueError, match="do not match"):
        ModuleOperationRequest(
            source,
            output,
            module_plan_fingerprint(ModuleKind.RSM, ("plan",)),
            module_provenance_digest(ModuleKind.RSM, {"kind": "rsm"}),
        )


def test_module_source_receipt_is_factory_only_and_revalidates_table(tmp_path):
    _image, table = _table(tmp_path)
    assert table.receipt is not None
    with pytest.raises(TypeError):
        ModuleSourceReceipt(
            table.receipt,
            ModuleKind.STITCH,
            table.table_fingerprint,
            table.labels,
        )
    forged = replace(table, table_fingerprint=_digest("forged table"))
    with pytest.raises(ValueError, match="no longer an exact source fact"):
        ModuleSourceReceipt.from_metadata_table(
            forged,
            kind=ModuleKind.STITCH,
        )
    with pytest.raises(ValueError, match="outside the exact table"):
        ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.RSM,
            resolved_selectors=(MetadataColumnSelector("Seconds", 0),),
        )


def test_rsm_source_group_is_factory_owned_ordered_and_bounded(tmp_path):
    _images, table = _multi_table(tmp_path)
    members = tuple(
        ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.RSM,
            selected_labels=(label,),
        )
        for label in table.labels
    )
    group = ModuleSourceGroupReceipt.from_members(members)
    assert group.kind is ModuleKind.RSM
    assert group.members == members
    assert group.selected_frame_count == len(members)
    assert group.fingerprint == analysis_canonical_fingerprint(
        "module-source-group-v1",
        tuple(member.fingerprint for member in members),
    )
    reversed_group = ModuleSourceGroupReceipt.from_members(tuple(reversed(members)))
    assert reversed_group.members == tuple(reversed(members))
    assert reversed_group.fingerprint != group.fingerprint

    with pytest.raises(TypeError, match="factory|invalid"):
        ModuleSourceGroupReceipt(ModuleKind.RSM, members)
    with pytest.raises(TypeError, match="exact tuple"):
        ModuleSourceGroupReceipt.from_members(list(members))
    with pytest.raises(TypeError, match="invalid"):
        ModuleSourceGroupReceipt.from_members(())
    with pytest.raises(TypeError, match="invalid"):
        ModuleSourceGroupReceipt.from_members((members[0], members[0]))
    stitch = ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.STITCH,
        selected_labels=(table.labels[0],),
    )
    with pytest.raises(TypeError, match="invalid"):
        ModuleSourceGroupReceipt.from_members((stitch,))

    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(group)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(group)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(group)
    with pytest.raises(TypeError):
        replace(group)
    if hasattr(copy, "replace"):
        with pytest.raises(TypeError, match="not replaceable"):
            copy.replace(group)


def test_rsm_source_group_enforces_member_and_aggregate_frame_bounds(tmp_path):
    _images, table = _many_table(tmp_path / "member-bound", 17)
    members = tuple(
        ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.RSM,
            selected_labels=(label,),
        )
        for label in table.labels
    )
    assert len(ModuleSourceGroupReceipt.from_members(members[:16]).members) == 16
    with pytest.raises(TypeError, match="invalid"):
        ModuleSourceGroupReceipt.from_members(members)
    with pytest.raises(TypeError, match="invalid"):
        ModuleSourceGroupReceipt.from_members((object(),))

    _images, large_table = _many_table(tmp_path / "frame-bound", 272)
    oversized = tuple(
        ModuleSourceReceipt.from_metadata_table(
            large_table,
            kind=ModuleKind.RSM,
            selected_labels=large_table.labels[: 257 + ordinal],
        )
        for ordinal in range(16)
    )
    assert sum(len(member.selected_labels) for member in oversized) > 4096
    with pytest.raises(ValueError, match="4096 selected frames"):
        ModuleSourceGroupReceipt.from_members(oversized)


def test_rsm_group_requalification_is_ordered_and_stops_on_first_failure(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.module_transaction as module_transaction
    from xrd_tools.analysis.scan_operations import MetadataTableRequalificationResult

    members = tuple(
        ModuleSourceReceipt.from_metadata_table(
            _table(tmp_path / f"member-{ordinal}")[1],
            kind=ModuleKind.RSM,
        )
        for ordinal in range(3)
    )
    group = ModuleSourceGroupReceipt.from_members(members)

    calls = []

    def completed(plan, *, cancel_token=None):
        ordinal = len(calls)
        calls.append((ordinal, plan.receipt, cancel_token))
        member = members[ordinal]
        return MetadataTableRequalificationResult(
            AnalysisDisposition.COMPLETED,
            "OK",
            member.analysis,
            member.table_fingerprint,
        )

    token = Event()
    monkeypatch.setattr(
        module_transaction,
        "run_metadata_table_requalification",
        completed,
    )
    result = module_transaction._source_requalification(
        group,
        cancel_token=token,
    )
    assert result.disposition is AnalysisDisposition.COMPLETED
    assert [item[1] for item in calls] == [member.analysis for member in members]
    assert all(item[2] is token for item in calls)

    for stop_at, disposition in (
        (0, AnalysisDisposition.REFUSED),
        (1, AnalysisDisposition.CANCELLED),
        (2, AnalysisDisposition.REFUSED),
    ):
        calls.clear()

        def stop(plan, *, cancel_token=None):
            ordinal = len(calls)
            calls.append((ordinal, plan.receipt, cancel_token))
            member = members[ordinal]
            state = disposition if ordinal == stop_at else AnalysisDisposition.COMPLETED
            return MetadataTableRequalificationResult(
                state,
                "STOP" if state is not AnalysisDisposition.COMPLETED else "OK",
                member.analysis if state is AnalysisDisposition.COMPLETED else None,
                member.table_fingerprint if state is AnalysisDisposition.COMPLETED else "",
            )

        monkeypatch.setattr(
            module_transaction,
            "run_metadata_table_requalification",
            stop,
        )
        result = module_transaction._source_requalification(group)
        assert result.disposition is disposition
        assert len(calls) == stop_at + 1

    calls.clear()

    def foreign_completed(plan, *, cancel_token=None):
        ordinal = len(calls)
        calls.append(ordinal)
        member = members[ordinal]
        return MetadataTableRequalificationResult(
            AnalysisDisposition.COMPLETED,
            "OK",
            members[(ordinal + 1) % len(members)].analysis,
            member.table_fingerprint,
        )

    monkeypatch.setattr(
        module_transaction,
        "run_metadata_table_requalification",
        foreign_completed,
    )
    with pytest.raises(ModuleArtifactRefused, match="SOURCE_IDENTITY_MISMATCH"):
        module_transaction._source_requalification(group)
    assert calls == [0]


def test_rsm_v2_module_request_is_private_tagged_and_fail_closed(tmp_path):
    _images, table = _multi_table(tmp_path / "source")
    members = tuple(
        ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.RSM,
            selected_labels=(label,),
        )
        for label in table.labels[:2]
    )
    group = ModuleSourceGroupReceipt.from_members(members)
    output = ModuleOutputRequest(
        tmp_path / "output" / "result.nexus",
        AnalysisArtifactKind.RSM,
    )
    plan = module_plan_fingerprint(ModuleKind.RSM, ("rsm-operation-plan-v2",))
    provenance = {"schema_version": "rsm-operation-v2-intent", "kind": "rsm"}
    provenance_digest = module_provenance_digest(ModuleKind.RSM, provenance)

    with pytest.raises(TypeError, match="one exact source receipt"):
        ModuleOperationRequest(group, output, plan, provenance_digest)
    with pytest.raises(TypeError, match="exact source group"):
        _rsm_v2_module_request(members[0], output, plan, provenance_digest)
    wrong_output = ModuleOutputRequest(
        tmp_path / "output" / "wrong.nexus",
        AnalysisArtifactKind.STITCH_1D,
    )
    with pytest.raises(ValueError, match="do not match"):
        _rsm_v2_module_request(group, wrong_output, plan, provenance_digest)
    with pytest.raises(
        TypeError,
        match="factory-owned intent",
    ):
        _rsm_v2_module_request(group, output, plan, provenance_digest)


def test_module_source_membership_is_nonempty_unique_subset_in_admitted_order(
    tmp_path,
):
    _images, table = _multi_table(tmp_path)
    first, second, third = table.labels
    complete = ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.STITCH,
    )
    subset = ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.STITCH,
        selected_labels=(first, third),
    )
    assert subset.analysis == complete.analysis == table.receipt
    assert subset.selected_labels == (first, third)
    assert subset.fingerprint != complete.fingerprint

    with pytest.raises(TypeError, match="exact tuple"):
        ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.STITCH,
            selected_labels=[first],
        )
    with pytest.raises(TypeError, match="invalid"):
        ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.STITCH,
            selected_labels=(),
        )
    with pytest.raises(TypeError, match="invalid"):
        ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.STITCH,
            selected_labels=(first, first),
        )
    with pytest.raises(ValueError, match="outside"):
        ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.STITCH,
            selected_labels=(max(table.labels) + 1,),
        )
    with pytest.raises(ValueError, match="retain admitted source order"):
        ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.STITCH,
            selected_labels=(third, second),
        )


def test_requalified_source_lease_fences_drift_and_closes_on_body_error(
    tmp_path, monkeypatch,
):
    image, table = _table(tmp_path)
    import xrd_tools.analysis.scan_operations as operations

    real_close = operations._close
    closes = []

    def close(source):
        closes.append(source)
        real_close(source)

    monkeypatch.setattr(operations, "_close", close)
    with pytest.raises(RuntimeError, match="body"):
        with requalified_analysis_source(table.receipt) as source:
            assert source.load_frame(table.labels[0]).shape == (3, 4)
            raise RuntimeError("body")
    assert len(closes) == 1

    with pytest.raises(AnalysisSourceLeaseRefused) as raised:
        with requalified_analysis_source(table.receipt) as source:
            assert source.load_frame(table.labels[0]).shape == (3, 4)
            prior = image.stat()
            payload = image.read_bytes()
            image.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))
            os.utime(image, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    assert raised.value.code == "SOURCE_REVISION_CHANGED"
    assert len(closes) == 2


def test_source_lease_honors_pre_cancel_without_opening(tmp_path):
    _image, table = _table(tmp_path)
    cancel = Event()
    cancel.set()
    with pytest.raises(AnalysisSourceLeaseRefused) as raised:
        with requalified_analysis_source(table.receipt, cancel_token=cancel):
            raise AssertionError("unreachable")
    assert raised.value.code == "CANCELLED"


def test_source_lease_close_failure_never_masks_body_or_fence(
    tmp_path,
    monkeypatch,
):
    image, table = _table(tmp_path)
    import xrd_tools.analysis.scan_operations as operations

    real_close = operations._close
    closes = []

    def close_then_fail(source):
        closes.append(source)
        real_close(source)
        raise RuntimeError("close fault")

    monkeypatch.setattr(operations, "_close", close_then_fail)
    with pytest.raises(ValueError, match="body primary") as body:
        with requalified_analysis_source(table.receipt):
            raise ValueError("body primary")
    assert "close fault" in " ".join(getattr(body.value, "__notes__", ()))

    with pytest.raises(AnalysisSourceLeaseRefused) as fence:
        with requalified_analysis_source(table.receipt):
            prior = image.stat()
            payload = image.read_bytes()
            image.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))
            os.utime(image, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    assert fence.value.code == "SOURCE_REVISION_CHANGED"
    assert "close fault" in " ".join(getattr(fence.value, "__notes__", ()))
    assert len(closes) == 2


def test_source_lease_hostile_close_text_preserves_body_and_admission_refusal(
    tmp_path,
    monkeypatch,
):
    image, table = _table(tmp_path)
    import xrd_tools.analysis.scan_operations as operations

    real_close = operations._close

    class HostileClose(RuntimeError):
        def __str__(self):
            raise RuntimeError("hostile close text")

    class HostilePrimary(ValueError):
        def add_note(self, _note):
            raise RuntimeError("hostile add_note")

    def close_then_fail(source):
        real_close(source)
        raise HostileClose()

    monkeypatch.setattr(operations, "_close", close_then_fail)
    with pytest.raises(ValueError, match="body primary") as body:
        with requalified_analysis_source(table.receipt):
            raise ValueError("body primary")
    assert "exception message unavailable" in " ".join(
        getattr(body.value, "__notes__", ())
    )

    hostile_primary = HostilePrimary("exact hostile primary")
    with pytest.raises(HostilePrimary) as hostile:
        with requalified_analysis_source(table.receipt):
            raise hostile_primary
    assert hostile.value is hostile_primary

    prior = image.stat()
    payload = image.read_bytes()
    image.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))
    os.utime(image, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    with pytest.raises(AnalysisSourceLeaseRefused) as refused:
        with requalified_analysis_source(table.receipt):
            raise AssertionError("unreachable")
    assert refused.value.code == "SOURCE_IDENTITY_MISMATCH"
    assert "exception message unavailable" in " ".join(
        getattr(refused.value, "__notes__", ())
    )


def test_source_lease_post_open_cancel_survives_close_failure(
    tmp_path,
    monkeypatch,
):
    _image, table = _table(tmp_path)
    import xrd_tools.analysis.scan_operations as operations

    cancel = Event()
    real_requalify = operations._requalify
    real_close = operations._close

    def requalify_then_cancel(receipt):
        snapshot = real_requalify(receipt)
        cancel.set()
        return snapshot

    def close_then_fail(source):
        real_close(source)
        raise RuntimeError("close fault")

    monkeypatch.setattr(operations, "_requalify", requalify_then_cancel)
    monkeypatch.setattr(operations, "_close", close_then_fail)
    with pytest.raises(AnalysisSourceLeaseRefused) as raised:
        with requalified_analysis_source(table.receipt, cancel_token=cancel):
            raise AssertionError("unreachable")
    assert raised.value.code == "CANCELLED"
    assert "close fault" in " ".join(getattr(raised.value, "__notes__", ()))


def test_module_artifact_admission_and_commit_keep_exact_request(tmp_path):
    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(output)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(output)
    assert not hasattr(output, "abort")
    value = IntegrationResult1D(
        radial=np.linspace(0.1, 1.0, 8),
        intensity=np.linspace(2.0, 3.0, 8),
        unit="q_A^-1",
    )
    result = output.publish(
        lambda entry: write_stitched(
            entry,
            stitched_1d=value,
            provenance=bound.provenance_json,
            bounded_artifact=True,
        ),
    )
    assert result.disposition is ModuleDisposition.COMMITTED
    assert result.request is request
    assert result.commit.request is request
    assert result.commit.output is request.output
    assert result.commit.terminal.target == request.output.target
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(result.commit)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(result.commit)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(result.commit)
    assert output.snapshot.published
    assert not output.snapshot.slot_held


def test_module_neutral_stitch_commit_binds_separate_attestation_and_same_request(tmp_path):
    request, provenance = _xu_request(tmp_path)
    projection = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.STITCH_1D,
        axes=(("q", np.linspace(0.1, 1.0, 8)),),
        axis_units=(("q", "q_A^-1"),),
        intensity=np.linspace(2.0, 3.0, 8),
        sigma=None,
        coverage=np.arange(1, 9, dtype=np.float64),
        normalization=np.linspace(1.0, 2.0, 8),
    )
    attestation = _attestation(request, projection.result_fingerprint)
    digest = analysis_execution_attestation_digest(
        request.output.kind,
        attestation,
        request_fingerprint=request.fingerprint,
    )
    bound = module_artifact_request(
        request,
        provenance,
        execution_attestation=attestation,
        execution_attestation_digest=digest,
    )
    assert bound.schema_version == 5
    output = admit_module_artifact(
        request,
        provenance,
        execution_attestation=attestation,
        execution_attestation_digest=digest,
        coordinator=OutputTransactionCoordinator(),
    )
    result = output.publish(
        lambda entry: write_stitched(
            entry,
            result_projection=projection,
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )
    )
    assert result.disposition is ModuleDisposition.COMMITTED
    assert result.request is request
    assert result.commit.request is request
    assert result.commit.execution_attestation_digest == digest
    expected = analysis_canonical_fingerprint(
        "module-commit-v2",
        (
            request.fingerprint,
            result.commit.terminal.target,
            result.commit.terminal.size,
            result.commit.terminal.digest,
            result.commit.terminal.ordinal,
            (
                result.commit.terminal.device,
                result.commit.terminal.inode,
                result.commit.terminal.size,
                result.commit.terminal.mtime_ns,
                result.commit.terminal.ctime_ns,
            ),
            projection.result_fingerprint,
            digest,
        ),
    )
    assert result.commit.fingerprint == expected

    clone = ModuleOperationRequest(
        request.source,
        request.output,
        request.plan_fingerprint,
        request.provenance_digest,
    )
    assert clone is not request
    assert clone.fingerprint != request.fingerprint
    replaced = replace(request)
    assert replaced is not request
    assert replaced._xu_stitch_v2_bound is False
    assert replaced.fingerprint == clone.fingerprint
    with pytest.raises(
        ModuleArtifactRefused,
        match="XU_INTENT_PROVENANCE_MISMATCH",
    ):
        module_artifact_request(
            clone,
            provenance,
            execution_attestation=attestation,
            execution_attestation_digest=digest,
        )
    with pytest.raises(
        ModuleArtifactRefused,
        match="XU_INTENT_PROVENANCE_MISMATCH",
    ):
        module_artifact_request(
            replaced,
            provenance,
            execution_attestation=attestation,
            execution_attestation_digest=digest,
        )
    with pytest.raises(TypeError, match="exact module request"):
        ModuleCommitReceipt.from_artifact(clone, output.snapshot.receipt)

    count_forged = dict(attestation)
    count_forged["selected_frame_count"] = 999
    count_forged["release_check_frame_count"] = 999
    count_digest = analysis_execution_attestation_digest(
        request.output.kind,
        count_forged,
        request_fingerprint=request.fingerprint,
    )
    with pytest.raises(
        ModuleArtifactRefused,
        match="EXECUTION_ATTESTATION_COUNT_MISMATCH",
    ):
        module_artifact_request(
            request,
            provenance,
            execution_attestation=count_forged,
            execution_attestation_digest=count_digest,
        )

    forged = dict(attestation)
    forged["release_check_frame_count"] = 2
    with pytest.raises((TypeError, ValueError, ModuleArtifactRefused)):
        module_artifact_request(
            request,
            provenance,
            execution_attestation=forged,
            execution_attestation_digest=digest,
        )


def test_rsm_v2_module_commit_binds_group_and_exact_attestation(
    tmp_path,
    monkeypatch,
):
    from tests.core.test_rsm_operation_r2 import _form, _write_member
    from xdart.gui.tools.rsm_values import prepare_rsm_tool_v2
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-1.0, 1.0),
            (-2.0, 2.0),
            (-3.0, 3.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(
            tmp_path,
            (_write_member(tmp_path, 0), _write_member(tmp_path, 1)),
        )
    ).request
    request = prepared.module
    provenance = prepared.provenance
    mask_receipts = tuple(
        rsm_operation._make_rsm_static_mask_receipt(
            member,
            prepared.plan.conditioning,
            None,
            None,
        )
        for member in prepared.preflight.members
    )
    bounds = prepared.preflight.common_grid.bounds
    bins = prepared.plan.bins
    projection = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM,
        axes=tuple(
            (
                name,
                np.linspace(axis_bounds[0], axis_bounds[1], count),
            )
            for name, axis_bounds, count in zip(
                ("h", "k", "l"),
                bounds,
                bins,
                strict=True,
            )
        ),
        axis_units=(("h", None), ("k", None), ("l", None)),
        intensity=np.arange(
            np.prod(bins),
            dtype=np.float64,
        ).reshape(bins),
        sigma=None,
        coverage=None,
        normalization=None,
    )
    attestation = _rsm_attestation(
        request,
        provenance,
        projection.result_fingerprint,
        mask_receipts=mask_receipts,
    )
    digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.RSM,
        attestation,
        request_fingerprint=request.fingerprint,
    )
    bound = module_artifact_request(
        request,
        provenance,
        execution_attestation=attestation,
        execution_attestation_digest=digest,
        rsm_mask_receipts=mask_receipts,
    )
    assert bound.schema_version == 2
    assert bound.source_fingerprint == request.source.fingerprint
    output = admit_module_artifact(
        request,
        provenance,
        execution_attestation=attestation,
        execution_attestation_digest=digest,
        rsm_mask_receipts=mask_receipts,
        coordinator=OutputTransactionCoordinator(),
    )
    result = output.publish(
        lambda entry: write_rsm(
            entry,
            result_projection=projection,
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )
    )
    assert result.disposition is ModuleDisposition.COMMITTED
    assert result.commit.request is request
    assert result.commit.execution_attestation_digest == digest
    assert output.snapshot.receipt.inspection.source_fingerprint == (
        request.source.fingerprint
    )

    forged = copy.deepcopy(attestation)
    forged["member_masks"][0]["mask_receipt_fingerprint"] = _digest(
        "arbitrary-mask-receipt"
    )
    forged_digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.RSM,
        forged,
        request_fingerprint=request.fingerprint,
    )
    with pytest.raises(
        ModuleArtifactRefused,
        match="RSM_EXECUTION_ATTESTATION_MISMATCH",
    ):
        module_artifact_request(
            request,
            provenance,
            execution_attestation=forged,
            execution_attestation_digest=forged_digest,
            rsm_mask_receipts=mask_receipts,
        )

    retagged = copy.deepcopy(attestation)
    foreign_frame = RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
    retagged.update(
        {
            "coordinate_frame": foreign_frame.value,
            "axis_names": list(foreign_frame.axis_names),
            "axis_units": list(foreign_frame.axis_units),
            "matrix_policy": foreign_frame.matrix_policy,
        }
    )
    retagged_digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.RSM,
        retagged,
        request_fingerprint=request.fingerprint,
    )
    with pytest.raises(
        ModuleArtifactRefused,
        match="RSM_EXECUTION_ATTESTATION_MISMATCH",
    ):
        module_artifact_request(
            request,
            provenance,
            execution_attestation=retagged,
            execution_attestation_digest=retagged_digest,
            rsm_mask_receipts=mask_receipts,
        )

    genuine = mask_receipts[0]
    clone = object.__new__(rsm_operation.RSMStaticMaskReceipt)
    for name in (
        "member_preflight_fingerprint",
        "mask_policy",
        "conditioning_fingerprint",
        "full_shape",
        "full_raw_digest",
        "full_masked_pixel_count",
        "cropped_shape",
        "cropped_raw_digest",
        "cropped_masked_pixel_count",
        "fingerprint",
    ):
        object.__setattr__(clone, name, getattr(genuine, name))
    clone_receipts = (clone, *mask_receipts[1:])
    clone_attestation = _rsm_attestation(
        prepared.module,
        prepared.provenance,
        _digest("exact-class-clone-result"),
        mask_receipts=clone_receipts,
    )
    clone_digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.RSM,
        clone_attestation,
        request_fingerprint=prepared.module.fingerprint,
    )
    with pytest.raises(
        ModuleArtifactRefused,
        match="RSM_EXECUTION_ATTESTATION_MISMATCH",
    ):
        module_artifact_request(
            prepared.module,
            prepared.provenance,
            execution_attestation=clone_attestation,
            execution_attestation_digest=clone_digest,
            rsm_mask_receipts=clone_receipts,
        )

    object.__setattr__(
        genuine,
        "member_preflight_fingerprint",
        _digest("mutated-issued-member"),
    )
    mutated_fingerprint = analysis_canonical_fingerprint(
        "rsm-static-mask-v1",
        (
            genuine.member_preflight_fingerprint,
            genuine.mask_policy,
            genuine.conditioning_fingerprint,
            genuine.full_shape,
            genuine.full_raw_digest,
            genuine.full_masked_pixel_count,
            genuine.cropped_shape,
            genuine.cropped_raw_digest,
            genuine.cropped_masked_pixel_count,
        ),
    )
    object.__setattr__(genuine, "fingerprint", mutated_fingerprint)
    mutated_attestation = _rsm_attestation(
        prepared.module,
        prepared.provenance,
        _digest("mutated-issued-result"),
        mask_receipts=mask_receipts,
    )
    mutated_digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.RSM,
        mutated_attestation,
        request_fingerprint=prepared.module.fingerprint,
    )
    with pytest.raises(
        ModuleArtifactRefused,
        match="RSM_EXECUTION_ATTESTATION_MISMATCH",
    ):
        module_artifact_request(
            prepared.module,
            prepared.provenance,
            execution_attestation=mutated_attestation,
            execution_attestation_digest=mutated_digest,
            rsm_mask_receipts=mask_receipts,
        )

    foreign_conditioning = rsm_operation.RSMImageConditioning(
        99.0,
        123.0,
        None,
    )
    foreign_receipts = tuple(
        rsm_operation._make_rsm_static_mask_receipt(
            member,
            foreign_conditioning,
            None,
            None,
        )
        for member in prepared.preflight.members
    )
    foreign_attestation = _rsm_attestation(
        request,
        provenance,
        projection.result_fingerprint,
        mask_receipts=foreign_receipts,
    )
    foreign_digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.RSM,
        foreign_attestation,
        request_fingerprint=request.fingerprint,
    )
    with pytest.raises(
        ModuleArtifactRefused,
        match="RSM_EXECUTION_ATTESTATION_MISMATCH",
    ):
        module_artifact_request(
            request,
            provenance,
            execution_attestation=foreign_attestation,
            execution_attestation_digest=foreign_digest,
            rsm_mask_receipts=foreign_receipts,
        )


def test_rsm_v2_mask_facts_require_the_factory_execution_receipts(
    tmp_path,
    monkeypatch,
):
    from tests.core.test_rsm_operation_r2 import _form, _write_member
    from xdart.gui.tools.rsm_values import prepare_rsm_tool_v2
    import xrd_tools.analysis.rsm_operation as rsm_operation
    from xrd_tools.analysis.rsm_operation import RSMImageConditioning

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-1.0, 1.0),
            (-2.0, 2.0),
            (-3.0, 3.0),
        ),
    )
    form = replace(
        _form(tmp_path, (_write_member(tmp_path, 0),)),
        conditioning=RSMImageConditioning(0.0, None, 100.0),
    )
    prepared = prepare_rsm_tool_v2(form).request
    member = prepared.preflight.members[0]
    full_mask = np.zeros(member.detector_shape, dtype=bool)
    cropped_mask = np.zeros(member.cropped_shape, dtype=bool)
    full_mask.setflags(write=False)
    cropped_mask.setflags(write=False)
    mask_receipts = (
        rsm_operation._make_rsm_static_mask_receipt(
            member,
            prepared.plan.conditioning,
            full_mask,
            cropped_mask,
        ),
    )
    attestation = _rsm_attestation(
        prepared.module,
        prepared.provenance,
        _digest("static-mask-result"),
        mask_receipts=mask_receipts,
    )
    digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.RSM,
        attestation,
        request_fingerprint=prepared.module.fingerprint,
    )
    module_artifact_request(
        prepared.module,
        prepared.provenance,
        execution_attestation=attestation,
        execution_attestation_digest=digest,
        rsm_mask_receipts=mask_receipts,
    )

    forged = copy.deepcopy(attestation)
    mask = forged["member_masks"][0]
    mask["full_raw_digest"] = _digest("self-consistent-forged-mask")
    conditioning_fingerprint = analysis_canonical_fingerprint(
        "rsm-conditioning-v2",
        prepared.plan.conditioning._canonical_value(),
    )
    mask["mask_receipt_fingerprint"] = analysis_canonical_fingerprint(
        "rsm-static-mask-v1",
        (
            mask["member_preflight_fingerprint"],
            mask["mask_policy"],
            conditioning_fingerprint,
            tuple(mask["full_shape"]),
            mask["full_raw_digest"],
            mask["full_masked_pixel_count"],
            tuple(mask["cropped_shape"]),
            mask["cropped_raw_digest"],
            mask["cropped_masked_pixel_count"],
        ),
    )
    forged_digest = analysis_execution_attestation_digest(
        AnalysisArtifactKind.RSM,
        forged,
        request_fingerprint=prepared.module.fingerprint,
    )
    with pytest.raises(
        ModuleArtifactRefused,
        match="RSM_EXECUTION_ATTESTATION_MISMATCH",
    ):
        module_artifact_request(
            prepared.module,
            prepared.provenance,
            execution_attestation=forged,
            execution_attestation_digest=forged_digest,
            rsm_mask_receipts=mask_receipts,
        )


def test_rsm_v2_factory_owner_refuses_retagged_nested_provenance(
    tmp_path,
    monkeypatch,
):
    from tests.core.test_rsm_operation_r2 import _form, _write_member
    from xdart.gui.tools.rsm_values import prepare_rsm_tool_v2
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-1.0, 1.0),
            (-2.0, 2.0),
            (-3.0, 3.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(
            tmp_path,
            (_write_member(tmp_path, 0), _write_member(tmp_path, 1)),
        )
    ).request
    base = prepared.provenance
    forged_values = []

    value = copy.deepcopy(base)
    value["effective_geometry"]["diffractometer"]["preset"] = "evil"
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["members"][0]["geometry_binding"]["evil"] = True
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["members"][0]["contributions"][0]["values"] = []
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["members"][0]["dependency_files"] = [{"evil": True}]
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["common_grid"]["bounds"] = "evil"
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["plan"]["max_frame_bytes"] = -1
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["conditioning"]["additive_offset"] = "evil"
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["normalization"]["mode"] = "evil"
    for member in value["members"]:
        member["normalization_policy"] = copy.deepcopy(value["normalization"])
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["holds"] = ["everything-now-authorized"]
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["runtime_requirements"] = ["evil"]
    value["effective_geometry"]["runtime_requirements"] = ["evil"]
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["members"][0]["source_options"]["evil"] = True
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["members"][0]["detector_shape"] = [2, 2]
    value["members"][0]["cropped_shape"] = [2, 2]
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["members"][0]["mask_policy_intent"] = ["none", "extra"]
    forged_values.append(value)
    value = copy.deepcopy(base)
    value["output"]["parent_relative_path"] = "../../evil"
    forged_values.append(value)
    value = copy.deepcopy(base)
    semantic = _digest("forged-semantic")
    value["asset"]["semantic_fingerprint"] = semantic
    value["effective_geometry"]["asset_semantic_fingerprint"] = semantic
    forged_values.append(value)

    for forged in forged_values:
        forged_request = _rsm_v2_module_request(
            prepared.module.source,
            prepared.module.output,
            prepared.plan.fingerprint,
            module_provenance_digest(ModuleKind.RSM, forged),
            plan_owner=prepared.plan,
            preflight_owner=prepared.preflight,
            output_authority_owner=prepared.output_authority,
        )
        attestation = _rsm_attestation(
            forged_request,
            forged,
            _digest("rsm-retagged-result"),
        )
        digest = analysis_execution_attestation_digest(
            AnalysisArtifactKind.RSM,
            attestation,
            request_fingerprint=forged_request.fingerprint,
        )
        with pytest.raises(
            ModuleArtifactRefused,
            match="RSM_INTENT_PROVENANCE_MISMATCH",
        ):
            module_artifact_request(
                forged_request,
                forged,
                execution_attestation=attestation,
                execution_attestation_digest=digest,
            )


@pytest.mark.parametrize("outcome", ("writer", "cancel", "refuse"))
def test_candidate_cleanup_warning_keeps_primary_outcome_and_names_orphan(tmp_path, monkeypatch, outcome):
    from xrd_tools.io import finite_artifact

    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(request, provenance)
    cancel = Event()
    cleanups = []

    def fail_unlink(_descriptor, path, _authority):
        cleanups.append(path)
        raise OSError("candidate unlink failed")

    def writer(entry):
        if outcome == "writer":
            raise ValueError("primary writer failure")
        write_stitched(
            entry, stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8), unit="q_A^-1",
            ), provenance=bound.provenance_json, bounded_artifact=True,
        )
        if outcome == "cancel":
            cancel.set()

    def prepublish():
        if outcome == "refuse":
            raise ModuleArtifactRefused("SOURCE_REVISION_CHANGED")

    monkeypatch.setattr(finite_artifact, "_unlink_candidate", fail_unlink)
    result = output.publish(writer, cancel_token=cancel, prepublish_check=prepublish)
    assert result.disposition is {
        "writer": ModuleDisposition.FAILED,
        "cancel": ModuleDisposition.CANCELLED,
        "refuse": ModuleDisposition.REFUSED,
    }[outcome]
    assert len(cleanups) == 1 and cleanups[0].exists()
    assert output.snapshot.artifact.hidden_orphan == str(cleanups[0])
    assert str(cleanups[0]) in result.diagnostic
    assert not output.snapshot.published and not output.snapshot.slot_held
    assert not Path(request.output.target).exists()


def test_module_commit_receipt_is_built_before_lower_lease_release(
    tmp_path,
    monkeypatch,
):
    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    observed = []
    real_from_artifact = ModuleCommitReceipt.from_artifact

    def inspect_before_release(cls, module_request, artifact):
        assert cls is ModuleCommitReceipt
        observed.append(output.snapshot.slot_held)
        return real_from_artifact(module_request, artifact)

    monkeypatch.setattr(
        ModuleCommitReceipt,
        "from_artifact",
        classmethod(inspect_before_release),
    )
    result = output.publish(
        lambda entry: write_stitched(
            entry,
            stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8),
                unit="q_A^-1",
            ),
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )
    )
    assert result.disposition is ModuleDisposition.COMMITTED
    assert observed == [True]
    assert not output.snapshot.slot_held


@pytest.mark.parametrize(
    ("module_kind", "artifact_kind"),
    (
        (ModuleKind.STITCH, AnalysisArtifactKind.STITCH_2D),
        (ModuleKind.RSM, AnalysisArtifactKind.RSM),
    ),
)
def test_shared_module_commit_covers_stitch_2d_and_rsm(
    tmp_path,
    module_kind,
    artifact_kind,
):
    request = _request(tmp_path, module_kind, artifact_kind)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    if artifact_kind is AnalysisArtifactKind.STITCH_2D:
        value = IntegrationResult2D(
            radial=np.linspace(0.1, 1.0, 8),
            azimuthal=np.linspace(-5.0, 5.0, 4),
            intensity=np.arange(32, dtype=float).reshape(8, 4),
            sigma=np.full((8, 4), 0.25),
            unit="q_A^-1",
            azimuthal_unit="chi_deg",
        )
        writer = lambda entry: write_stitched(
            entry,
            stitched_2d=value,
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )
    else:
        volume = RSMVolume(
            h=np.linspace(-1.0, 1.0, 3),
            k=np.linspace(-2.0, 2.0, 4),
            l=np.linspace(0.0, 3.0, 5),
            intensity=np.arange(60, dtype=float).reshape(3, 4, 5),
        )
        writer = lambda entry: write_rsm(
            entry,
            volume,
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )
    result = output.publish(writer)
    assert result.disposition is ModuleDisposition.COMMITTED
    assert result.commit is not None
    assert result.commit.request is request
    assert result.commit.output is request.output
    assert output.snapshot.receipt is not None
    assert output.snapshot.receipt.inspection.kind is artifact_kind
    assert not output.snapshot.slot_held


def test_invalid_writer_does_not_consume_module_publication(tmp_path):
    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    with pytest.raises(TypeError, match="writer"):
        output.publish(object())
    assert output.snapshot.slot_held and not output.snapshot.writer_started
    assert output.snapshot.writer_started is False
    result = output.publish(
        lambda entry: write_stitched(
            entry,
            stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8),
                unit="q_A^-1",
            ),
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )
    )
    assert result.disposition is ModuleDisposition.COMMITTED


def test_writer_failure_returns_exact_failed_terminal_and_releases(tmp_path):
    request = _request(tmp_path)
    output = admit_module_artifact(
        request,
        _provenance(request),
        coordinator=OutputTransactionCoordinator(),
    )

    def fail(_entry):
        raise RuntimeError("scientific writer failed")

    result = output.publish(fail)
    assert result.disposition is ModuleDisposition.FAILED
    assert result.code == "OUTPUT_FAILED"
    assert "scientific writer failed" in result.diagnostic
    assert result.request is request
    assert result.commit is None
    assert not output.snapshot.published and not output.snapshot.close_pending
    assert not output.snapshot.slot_held
    assert not Path(request.output.target).exists()
    with pytest.raises(RuntimeError, match="one-shot"):
        output.publish(fail)


def test_writer_cannot_impersonate_cleanup_pending(tmp_path):
    request = _request(tmp_path)
    output = admit_module_artifact(
        request,
        _provenance(request),
        coordinator=OutputTransactionCoordinator(),
    )

    def impersonate(_entry):
        raise AnalysisArtifactCleanupPending(output.snapshot)

    result = output.publish(impersonate)
    assert result.disposition is ModuleDisposition.FAILED
    assert result.code == "OUTPUT_FAILED"
    assert result.request is request
    assert not output.snapshot.published and not output.snapshot.close_pending
    assert not output.snapshot.slot_held


def test_module_prepublish_check_runs_after_writer_and_can_refuse_commit(tmp_path):
    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    order = []

    def write(entry):
        order.append("writer")
        write_stitched(
            entry,
            stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8),
                unit="q_A^-1",
            ),
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )

    def refuse():
        order.append("prepublish")
        raise ModuleArtifactRefused("GEOMETRY_IDENTITY_MISMATCH")

    terminal = output.publish(write, prepublish_check=refuse)
    assert terminal.disposition is ModuleDisposition.REFUSED
    assert terminal.code == "GEOMETRY_IDENTITY_MISMATCH"
    assert order == ["writer", "prepublish"]
    assert not Path(request.output.target).exists()
    assert not output.snapshot.published and not output.snapshot.close_pending
    assert not output.snapshot.slot_held


@pytest.mark.parametrize(
    "control_exception",
    ("_ModuleCommitCancelled", "_ModuleCommitRefused"),
)
def test_writer_cannot_impersonate_module_control_outcome(
    tmp_path,
    control_exception,
):
    import xrd_tools.analysis.module_transaction as module_api

    request = _request(tmp_path)
    output = admit_module_artifact(
        request,
        _provenance(request),
        coordinator=OutputTransactionCoordinator(),
    )
    exception_type = getattr(module_api, control_exception)

    def impersonate(_entry):
        raise exception_type("SOURCE_REVISION_CHANGED")

    result = output.publish(impersonate)
    assert result.disposition is ModuleDisposition.FAILED
    assert result.code == "OUTPUT_FAILED"
    assert control_exception in result.diagnostic
    assert result.request is request
    assert not output.snapshot.published and not output.snapshot.close_pending
    assert not output.snapshot.slot_held
    assert not Path(request.output.target).exists()


@pytest.mark.parametrize(
    "control_exception",
    ("_ModuleCommitCancelled", "_ModuleCommitRefused"),
)
def test_prepublish_check_cannot_impersonate_module_control_outcome(
    tmp_path,
    control_exception,
):
    import xrd_tools.analysis.module_transaction as module_api

    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    exception_type = getattr(module_api, control_exception)

    def write(entry):
        write_stitched(
            entry,
            stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8),
                unit="q_A^-1",
            ),
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )

    def impersonate():
        raise exception_type("FORGED_REFUSAL")

    result = output.publish(write, prepublish_check=impersonate)
    assert result.disposition is ModuleDisposition.FAILED
    assert result.code == "OUTPUT_FAILED"
    assert control_exception in result.diagnostic
    assert result.request is request
    assert not output.snapshot.published and not output.snapshot.close_pending
    assert not output.snapshot.slot_held
    assert not Path(request.output.target).exists()


@pytest.mark.parametrize("code", ("", None))
def test_module_artifact_refusal_requires_exact_nonempty_code(code):
    with pytest.raises(TypeError, match="nonempty string"):
        ModuleArtifactRefused(code)


@pytest.mark.parametrize("hostile_kind", ("malformed", "subclass"))
def test_prepublish_check_rejects_malformed_or_subclass_refusal(
    tmp_path,
    hostile_kind,
):
    import xrd_tools.analysis.module_transaction as module_api

    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )

    def write(entry):
        write_stitched(
            entry,
            stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8),
                unit="q_A^-1",
            ),
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )

    if hostile_kind == "malformed":
        refusal = ModuleArtifactRefused("VALID_AT_CONSTRUCTION")
        refusal.code = ""
    else:
        class HostileRefusal(ModuleArtifactRefused):
            def __getattribute__(self, name):
                if name == "code":
                    raise module_api._ModuleCommitCancelled("FORGED_CANCEL")
                return super().__getattribute__(name)

        refusal = HostileRefusal("PUBLIC_LOOKALIKE")

    def impersonate():
        raise refusal

    result = output.publish(write, prepublish_check=impersonate)
    assert result.disposition is ModuleDisposition.FAILED
    assert result.code == "OUTPUT_FAILED"
    assert result.request is request
    assert not output.snapshot.published and not output.snapshot.close_pending
    assert not output.snapshot.slot_held
    assert not Path(request.output.target).exists()


def test_module_commit_revalidates_the_factory_artifact(tmp_path):
    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    artifact = admit_analysis_artifact(
        bound,
        coordinator=OutputTransactionCoordinator(),
    ).publish(
        lambda entry: write_stitched(
            entry,
            stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8),
                unit="q_A^-1",
            ),
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )
    )
    target = Path(request.output.target)
    payload = target.read_bytes()
    target.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))
    with pytest.raises(TargetChanged):
        ModuleCommitReceipt.from_artifact(request, artifact)


def test_module_commit_refuses_byte_identical_replacement_inode(tmp_path):
    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    artifact = admit_analysis_artifact(
        bound,
        coordinator=OutputTransactionCoordinator(),
    ).publish(
        lambda entry: write_stitched(
            entry,
            stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8),
                unit="q_A^-1",
            ),
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )
    )
    target = Path(request.output.target)
    state = target.stat()
    replacement = target.with_name("replacement.nexus")
    replacement.write_bytes(target.read_bytes())
    os.utime(replacement, ns=(state.st_atime_ns, state.st_mtime_ns))
    os.replace(replacement, target)
    with pytest.raises(TargetChanged):
        ModuleCommitReceipt.from_artifact(request, artifact)


def test_module_commit_refuses_genuine_artifact_with_foreign_provenance(tmp_path):
    request = _request(tmp_path)
    foreign = AnalysisArtifactRequest(
        request.output.target,
        request.output.kind,
        request.output.overwrite,
        request.fingerprint,
        request.source.analysis.source_fingerprint,
        request.plan_fingerprint,
        request.provenance_digest,
        {"kind": request.kind.value, "frames": [999]},
    )
    artifact = admit_analysis_artifact(
        foreign,
        coordinator=OutputTransactionCoordinator(),
    ).publish(
        lambda entry: write_stitched(
            entry,
            stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8),
                unit="q_A^-1",
            ),
            provenance=foreign.provenance_json,
            bounded_artifact=True,
        )
    )
    with pytest.raises(TypeError, match="exact module request"):
        ModuleCommitReceipt.from_artifact(request, artifact)


@pytest.mark.parametrize(
    ("kind", "artifact_kind"),
    (
        (ModuleKind.STITCH, AnalysisArtifactKind.STITCH_1D),
        (ModuleKind.RSM, AnalysisArtifactKind.RSM),
    ),
)
def test_module_replacement_refuses_foreign_target_and_keeps_rsm_create_new(
    tmp_path,
    kind,
    artifact_kind,
):
    initial = _request(tmp_path, kind=kind, artifact_kind=artifact_kind)
    target = Path(initial.output.target)
    original = b"prior operator output"
    target.write_bytes(original)
    output_request = ModuleOutputRequest(
        target,
        artifact_kind,
        AnalysisArtifactOverwrite.REPLACE,
    )

    if kind is ModuleKind.RSM:
        with pytest.raises(ValueError, match="must create one new immutable artifact"):
            ModuleOperationRequest(
                initial.source, output_request,
                initial.plan_fingerprint, initial.provenance_digest,
            )
    else:
        request = ModuleOperationRequest(
            initial.source, output_request,
            initial.plan_fingerprint, initial.provenance_digest,
        )
        with pytest.raises(ModuleArtifactRefused) as refused:
            admit_module_artifact(request, _provenance(request))
        assert refused.value.code == "OUTPUT_NOT_PREVIOUS_ANALYSIS"

    assert target.read_bytes() == original


def test_repeated_module_postcommit_readback_retry_does_not_replay_writer(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.module_transaction as module_api

    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    real_inspect = module_api.inspect_analysis_artifact
    failures = []
    writes = []

    def fail_module_readback_twice(*args, **kwargs):
        if len(failures) < 2:
            failures.append("module readback")
            raise OSError("transient module readback")
        return real_inspect(*args, **kwargs)

    def write_once(entry):
        writes.append("writer")
        write_stitched(
            entry,
            stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8),
                unit="q_A^-1",
            ),
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )

    monkeypatch.setattr(module_api, "inspect_analysis_artifact", fail_module_readback_twice)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(write_once)
    assert output.snapshot.published
    assert output.snapshot.receipt is not None
    assert output.snapshot.module_pending is True
    assert output.snapshot.artifact.retryable is True
    assert not output.snapshot.slot_held
    assert output.snapshot.retryable is True
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.retry_cleanup()
    assert output.snapshot.published
    assert output.snapshot.receipt is not None
    assert output.snapshot.module_pending is True
    assert not output.snapshot.slot_held
    assert output.snapshot.retryable is True
    recovered = output.retry_cleanup()
    assert recovered.disposition is ModuleDisposition.COMMITTED
    assert recovered.request is request
    assert recovered.commit is not None
    assert output.retry_cleanup() is recovered
    assert output.snapshot.module_pending is False
    assert output.snapshot.retryable is False
    assert writes == ["writer"]


def test_pre_cancelled_module_never_runs_writer_or_retains_target_lease(tmp_path):
    coordinator = OutputTransactionCoordinator()
    request = _request(tmp_path)
    provenance = _provenance(request)
    cancel = Event()
    cancel.set()

    with pytest.raises(ModuleArtifactRefused) as refused:
        admit_module_artifact(
            request,
            provenance,
            cancel_token=cancel,
            coordinator=coordinator,
        )
    assert refused.value.code == "CANCELLED"

    output = admit_module_artifact(
        request,
        provenance,
        coordinator=coordinator,
    )
    writes = []

    def forbidden_writer(_entry):
        writes.append("writer")

    result = output.publish(forbidden_writer, cancel_token=cancel)
    assert result.disposition is ModuleDisposition.CANCELLED
    assert result.code == "CANCELLED"
    assert result.request is request
    assert result.commit is None
    assert writes == []
    assert not output.snapshot.published and not output.snapshot.close_pending
    assert not output.snapshot.slot_held
    assert not Path(request.output.target).exists()


def test_stale_source_refuses_before_output_and_drift_rolls_back_candidate(tmp_path):
    request = _request(tmp_path)
    provenance = _provenance(request)
    source_path = next(
        Path(resolved)
        for _lexical, resolved, _revision
        in request.source.analysis.dependency_revisions
        if Path(resolved).is_file()
    )
    admitted = source_path.stat()
    payload = source_path.read_bytes()
    source_path.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))
    os.utime(source_path, ns=(admitted.st_atime_ns, admitted.st_mtime_ns))
    with pytest.raises(ModuleArtifactRefused) as raised:
        admit_module_artifact(
            request,
            provenance,
            coordinator=OutputTransactionCoordinator(),
        )
    assert raised.value.code == "SOURCE_REVISION_CHANGED"
    assert not Path(request.output.target).exists()

    fresh_root = tmp_path / "fresh"
    fresh = _request(fresh_root)
    fresh_provenance = _provenance(fresh)
    bound = module_artifact_request(fresh, fresh_provenance)
    output = admit_module_artifact(
        fresh,
        fresh_provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    fresh_source = next(
        Path(resolved)
        for _lexical, resolved, _revision
        in fresh.source.analysis.dependency_revisions
        if Path(resolved).is_file()
    )
    value = IntegrationResult1D(
        radial=np.linspace(0.1, 1.0, 8),
        intensity=np.linspace(2.0, 3.0, 8),
        unit="q_A^-1",
    )

    def write_then_drift(entry):
        write_stitched(
            entry,
            stitched_1d=value,
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )
        state = fresh_source.stat()
        raw = fresh_source.read_bytes()
        fresh_source.write_bytes(raw[:-1] + bytes((raw[-1] ^ 1,)))
        os.utime(fresh_source, ns=(state.st_atime_ns, state.st_mtime_ns))

    terminal = output.publish(
        write_then_drift,
    )
    assert terminal.disposition is ModuleDisposition.REFUSED
    assert terminal.code == "SOURCE_REVISION_CHANGED"
    assert not Path(fresh.output.target).exists()
    assert not output.snapshot.published and not output.snapshot.close_pending
    assert not output.snapshot.slot_held


def test_foreign_completed_source_requalification_refuses_after_writer(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.module_transaction as module_api
    from xrd_tools.analysis.scan_operations import MetadataTableRequalificationResult

    request = _request(tmp_path / "primary")
    foreign = _request(tmp_path / "foreign")
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    writes = []

    monkeypatch.setattr(
        module_api,
        "run_metadata_table_requalification",
        lambda *_args, **_kwargs: MetadataTableRequalificationResult(
            AnalysisDisposition.COMPLETED,
            "OK",
            foreign.source.analysis,
            request.source.table_fingerprint,
        ),
    )

    def write_once(entry):
        writes.append("writer")
        write_stitched(
            entry,
            stitched_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(2.0, 3.0, 8),
                unit="q_A^-1",
            ),
            provenance=bound.provenance_json,
            bounded_artifact=True,
        )

    result = output.publish(write_once)
    assert result.disposition is ModuleDisposition.REFUSED
    assert result.code == "SOURCE_IDENTITY_MISMATCH"
    assert result.request is request
    assert not output.snapshot.published and not output.snapshot.close_pending
    assert not output.snapshot.slot_held
    assert not Path(request.output.target).exists()
    assert writes == ["writer"]


def test_mismatched_provenance_is_refused_before_output_admission(tmp_path):
    request = _request(tmp_path)
    with pytest.raises(ModuleArtifactRefused) as raised:
        admit_module_artifact(
            request,
            {"kind": "stitch", "frames": [999]},
            coordinator=OutputTransactionCoordinator(),
        )
    assert raised.value.code == "PROVENANCE_MISMATCH"
    assert not Path(request.output.target).exists()
