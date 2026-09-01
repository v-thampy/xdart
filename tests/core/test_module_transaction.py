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
    ModuleSourceReceipt,
    ModuleTerminalResult,
    admit_module_artifact,
    module_artifact_request,
    module_plan_fingerprint,
    module_provenance_digest,
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
)
from xrd_tools.io.output_transaction import LeaseOwner, StreamTerminal, TargetChanged
from xrd_tools.io.output_transaction import OutputTransactionCoordinator, TransactionPhase
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.nexus import write_rsm, write_stitched
from xrd_tools.rsm.volume import RSMVolume
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


def test_module_retry_unexpected_lower_phase_keeps_module_snapshot_type(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.io.analysis_artifact as artifact_api

    request = _request(tmp_path)
    output = admit_module_artifact(
        request,
        _provenance(request),
        coordinator=OutputTransactionCoordinator(),
    )

    monkeypatch.setattr(
        artifact_api.AnalysisArtifactOutput,
        "_retry_retained",
        lambda self: self.snapshot,
    )
    with pytest.raises(AnalysisArtifactCleanupPending) as pending:
        output.retry_cleanup()
    assert type(pending.value.snapshot) is ModuleArtifactOutputSnapshot
    assert pending.value.snapshot == output.snapshot
    assert pending.value.snapshot.artifact == output._output.snapshot


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
    assert output.snapshot.phase is TransactionPhase.COMMITTED
    assert output.snapshot.remaining_lease_owners == ()


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
        observed.append(output.snapshot.remaining_lease_owners)
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
    assert observed == [tuple(LeaseOwner)]
    assert output.snapshot.remaining_lease_owners == ()


def test_module_release_fault_after_receipt_resumes_exact_owner_suffix(
    tmp_path,
    monkeypatch,
):
    coordinator = OutputTransactionCoordinator()
    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=coordinator,
    )
    real_release = coordinator._release
    successful = []
    failed = []

    def fail_after_two(lease, role, owner):
        if len(successful) == 2 and not failed:
            failed.append(role)
            raise OSError("module release transient")
        snapshot = real_release(lease, role, owner)
        successful.append(role)
        return snapshot

    monkeypatch.setattr(coordinator, "_release", fail_after_two)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(
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
    retained = output._terminal
    assert retained is not None
    assert retained.disposition is ModuleDisposition.COMMITTED
    assert output.snapshot.module_pending is True
    assert output.snapshot.retryable is True
    assert output.snapshot.remaining_lease_owners == tuple(LeaseOwner)[2:]
    recovered = output.retry_cleanup()
    assert recovered is retained
    assert output.snapshot.module_pending is False
    assert output.snapshot.remaining_lease_owners == ()
    assert successful == list(LeaseOwner)


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
    assert output.snapshot.remaining_lease_owners == ()


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
    assert output.snapshot.phase is TransactionPhase.LEASED
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
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()
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
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()


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
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()


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
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()
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
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()
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
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()
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


def test_module_committed_cleanup_retry_returns_one_exact_terminal(tmp_path, monkeypatch):
    import xrd_tools.io.output_transaction as transaction_api

    initial = _request(tmp_path)
    target = Path(initial.output.target)
    target.write_bytes(b"prior")
    output_request = ModuleOutputRequest(
        target,
        initial.output.kind,
        AnalysisArtifactOverwrite.REPLACE,
    )
    request = ModuleOperationRequest(
        initial.source,
        output_request,
        initial.plan_fingerprint,
        initial.provenance_digest,
    )
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    backup = output._output._transaction.backup
    real_unlink = transaction_api._unlink
    failed = []
    writes = []

    def fail_backup_once(path):
        if Path(path) == backup and not failed:
            failed.append("backup")
            raise OSError("backup cleanup fault")
        return real_unlink(path)

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

    monkeypatch.setattr(transaction_api, "_unlink", fail_backup_once)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(write_once)
    assert output.snapshot.phase is TransactionPhase.CLEANUP_PENDING
    assert output.snapshot.receipt is None
    assert output.snapshot.remaining_lease_owners == tuple(LeaseOwner)
    recovered = output.retry_cleanup()
    assert recovered.disposition is ModuleDisposition.COMMITTED
    assert recovered.request is request
    assert recovered.commit is not None
    assert recovered.commit.request is request
    assert recovered.commit.output is request.output
    assert recovered.commit.result_fingerprint
    assert output.snapshot.remaining_lease_owners == ()
    assert output.retry_cleanup() is recovered
    assert writes == ["writer"]


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
    assert output.snapshot.phase is TransactionPhase.COMMITTED
    assert output.snapshot.receipt is not None
    assert output.snapshot.module_pending is True
    assert output.snapshot.artifact.retryable is True
    assert output.snapshot.remaining_lease_owners == tuple(LeaseOwner)
    assert output.snapshot.retryable is True
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.retry_cleanup()
    assert output.snapshot.phase is TransactionPhase.COMMITTED
    assert output.snapshot.receipt is not None
    assert output.snapshot.module_pending is True
    assert output.snapshot.remaining_lease_owners == tuple(LeaseOwner)
    assert output.snapshot.retryable is True
    recovered = output.retry_cleanup()
    assert recovered.disposition is ModuleDisposition.COMMITTED
    assert recovered.request is request
    assert recovered.commit is not None
    assert output.retry_cleanup() is recovered
    assert output.snapshot.module_pending is False
    assert output.snapshot.retryable is False
    assert writes == ["writer"]


def test_module_publication_and_cleanup_failure_settles_failed_without_replay(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.io.output_transaction as transaction_api

    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    candidate = output._output._transaction._candidate
    real_link = transaction_api._link
    real_unlink = transaction_api._unlink
    link_failed = []
    unlink_failed = []
    writes = []

    def fail_publication_once(source, destination):
        if Path(source) == candidate and not link_failed:
            link_failed.append("publication")
            raise OSError("publication link fault")
        return real_link(source, destination)

    def fail_candidate_cleanup_once(path):
        if Path(path) == candidate and not unlink_failed:
            unlink_failed.append("candidate")
            raise OSError("candidate cleanup fault")
        return real_unlink(path)

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

    monkeypatch.setattr(transaction_api, "_link", fail_publication_once)
    monkeypatch.setattr(transaction_api, "_unlink", fail_candidate_cleanup_once)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(write_once)
    recovered = output.retry_cleanup()
    assert recovered.disposition is ModuleDisposition.FAILED
    assert recovered.code == "OUTPUT_FAILED"
    assert recovered.request is request
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()
    assert not Path(request.output.target).exists()
    assert output.retry_cleanup() is recovered
    assert writes == ["writer"]


def test_module_cancel_cleanup_retry_retains_cancelled_terminal(tmp_path, monkeypatch):
    import xrd_tools.io.output_transaction as transaction_api

    request = _request(tmp_path)
    provenance = _provenance(request)
    bound = module_artifact_request(request, provenance)
    output = admit_module_artifact(
        request,
        provenance,
        coordinator=OutputTransactionCoordinator(),
    )
    candidate = output._output._transaction._candidate
    real_unlink = transaction_api._unlink
    failed = []
    writes = []
    cancel = Event()

    def fail_candidate_once(path):
        if Path(path) == candidate and not failed:
            failed.append("candidate")
            raise OSError("candidate cleanup fault")
        return real_unlink(path)

    def write_then_cancel(entry):
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
        cancel.set()

    monkeypatch.setattr(transaction_api, "_unlink", fail_candidate_once)
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(write_then_cancel, cancel_token=cancel)
    assert output.snapshot.phase is TransactionPhase.CLEANUP_PENDING
    assert output.snapshot.receipt is None
    assert output.snapshot.remaining_lease_owners == tuple(LeaseOwner)
    recovered = output.retry_cleanup()
    assert recovered.disposition is ModuleDisposition.CANCELLED
    assert recovered.code == "CANCELLED"
    assert recovered.request is request
    assert recovered.commit is None
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()
    assert not Path(request.output.target).exists()
    assert output.retry_cleanup() is recovered
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
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()
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
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()


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
    assert output.snapshot.phase is TransactionPhase.ABORTED
    assert output.snapshot.remaining_lease_owners == ()
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
