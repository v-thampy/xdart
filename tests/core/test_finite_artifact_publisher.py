from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat
import threading
from types import SimpleNamespace

import pytest

import xrd_tools.io.finite_artifact as finite_module
from xrd_tools.io.finite_artifact import (
    FINITE_LINEAGE_MAX_BYTES,
    FiniteArtifactCollision,
    FiniteArtifactCapacityError,
    FiniteArtifactDisposition,
    FiniteArtifactIntegrityError,
    FiniteArtifactPublicationHeld,
    FiniteArtifactPublisher,
    FiniteArtifactRequest,
    FiniteArtifactResult,
    FiniteCandidateBinding,
    FiniteCandidateValidation,
    FiniteCommittedInspection,
    FiniteDocumentAdapter,
    FiniteFileSnapshot,
    FiniteSeedBinding,
    FiniteSeededDocumentAdapter,
    FiniteOperationContext,
    FinitePredecessorReceipt,
    FiniteSourceAdmission,
    admit_finite_artifact_lineage,
    capture_finite_predecessor,
    capture_finite_source,
    finite_artifact_request,
    finite_lineage_hdf_path,
    finite_operation_context,
    require_finite_artifact_lineage,
    write_finite_artifact_lineage,
)
from xrd_tools.io.output_path import (
    artifact_family_from_source,
    resolve_finite_output_target,
)
from xrd_tools.io.output_transaction import (
    LeaseUnavailable,
    StreamTerminal,
    capture_target_snapshot,
)
from xrd_tools.io.processed_scan_id import require_current_output_path


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _request(
    root: Path,
    source: Path,
    *,
    explicit_target: Path | None = None,
    family: str | None = "scan",
    operation_kind: str = "reintegrate-1d",
    admission: FiniteSourceAdmission | None = None,
    operation_context: FiniteOperationContext | None = None,
    predecessor: FinitePredecessorReceipt | None = None,
    entry: str = "entry",
) -> FiniteArtifactRequest:
    source_admission = admission or capture_finite_source(source)
    selected_context = operation_context or finite_operation_context(
        source_admission,
        request_generation_identity=_digest("request-generation"),
        resource_allocation_identity=_digest("resource-allocation"),
        route_identity=_digest("route"),
        custody_identity=_digest("custody"),
    )
    return finite_artifact_request(
        source_admission=source_admission,
        operation_context=selected_context,
        predecessor=predecessor or capture_finite_predecessor(source_admission),
        destination_directory=root,
        explicit_target=explicit_target,
        artifact_family=family,
        operation_kind=operation_kind,
        source_graph_identity=_digest("source-graph"),
        entry=entry,
        scientific_identity=_digest("science"),
        output_schema="xdart-current-v4",
        algorithm_identity=_digest("algorithm"),
        preservation_identity=_digest("preservation"),
    )


def _payload(request: FiniteArtifactRequest, value: str = "result") -> bytes:
    return json.dumps(
        {
            "value": value,
            "version_identity": request.version_identity,
            "publication_identity": request.publication_identity,
            "operation_kind": request.operation_kind,
            "source_graph_identity": request.source_graph_identity,
            "scientific_identity": request.scientific_identity,
            "preservation_identity": request.preservation_identity,
            "lineage_json": request.lineage.canonical_json,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_payload(request: FiniteArtifactRequest, value: str = "result"):
    def write(binding: FiniteCandidateBinding) -> None:
        binding.seek(0)
        binding.truncate(0)
        binding.write(_payload(request, value))

    return write


def _validate_payload(request: FiniteArtifactRequest, value: str = "result"):
    def validate(binding: FiniteCandidateBinding) -> FiniteCandidateValidation:
        binding.seek(0)
        assert binding.read(len(_payload(request, value)) + 1) == _payload(
            request,
            value,
        )
        return FiniteCandidateValidation(request.lineage)

    return validate


def _inspect_payload(
    path: Path,
    request: FiniteArtifactRequest,
) -> FiniteCommittedInspection:
    raw = path.read_bytes()
    assert raw == _payload(request)
    lineage = admit_finite_artifact_lineage(json.loads(raw)["lineage_json"])
    snapshot = capture_target_snapshot(path)
    assert snapshot.exists
    state = os.stat(path)
    return FiniteCommittedInspection(
        StreamTerminal(
            str(path),
            int(snapshot.size),
            str(snapshot.digest),
            1,
            int(state.st_dev),
            int(state.st_ino),
            int(state.st_mtime_ns),
            int(state.st_ctime_ns),
        ),
        lineage,
    )


def _publish(
    request: FiniteArtifactRequest,
    *,
    publisher: FiniteArtifactPublisher | None = None,
    seed=None,
    writer=None,
    validate=None,
    prepublish=None,
    inspect_committed=None,
):
    selected = publisher or FiniteArtifactPublisher(request)
    adapter = FiniteDocumentAdapter(
        lambda binding: nullcontext(binding),
        writer or _write_payload(request),
        lambda binding: nullcontext(binding),
        validate or _validate_payload(request),
    )
    return selected.publish(
        adapter,
        inspect_committed=inspect_committed or _inspect_payload,
        seed=seed,
        prepublish=prepublish,
    )


def _publish_validated_seed(
    request: FiniteArtifactRequest,
    admission: FiniteSourceAdmission,
    accept,
    *,
    publisher: FiniteArtifactPublisher | None = None,
    inspect=None,
    prepublish=None,
):
    adapter = FiniteSeededDocumentAdapter(
        lambda binding: nullcontext(binding),
        lambda document, _seed, _candidate: _write_payload(request)(document),
        lambda binding: nullcontext(binding),
        _validate_payload(request),
    )
    return (publisher or FiniteArtifactPublisher(request)).publish(
        adapter,
        inspect_committed=inspect or _inspect_payload,
        seed=admission,
        prepublish=prepublish,
        accept_validated_commit=accept,
    )


def test_trusted_adapter_authority_boundary_is_explicit() -> None:
    adapter_contract = FiniteDocumentAdapter.__doc__ or ""
    publisher_contract = FiniteArtifactPublisher.publish.__doc__ or ""

    assert "Trusted" in adapter_contract
    assert "duplicate" in adapter_contract
    assert "a sandbox" in publisher_contract
    assert "hostile same-process Python" in publisher_contract


@pytest.mark.parametrize(
    ("shown", "persisted", "expected"),
    (
        ("scan.nexus", None, "scan"),
        ("scan.reintegrate-1d-deadbeef.nexus", None,
         "scan.reintegrate-1d-deadbeef"),
        ("ignored.nexus", "beamline-scan_7", "beamline-scan_7"),
        # Ordinary punctuation and non-Latin script stay READABLE. These used
        # to reach the `artifact-<24 hex>` fallback, which put a content hash in
        # a public name against ADR-0010; the fallback is gone.
        ("Sample A 001.nexus", None, "Sample A 001"),
        ("scan#12.nexus", None, "scan#12"),
        ("basé.nexus", None, "basé"),
    ),
)
def test_family_policy_never_guesses_generated_suffixes(
    tmp_path: Path,
    shown: str,
    persisted: str | None,
    expected: str,
) -> None:
    assert artifact_family_from_source(tmp_path / shown, persisted) == expected


def test_request_identities_are_deterministic_and_path_roles_are_separate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    first = _request(tmp_path, source)
    second = _request(tmp_path, source)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    third = _request(elsewhere, source)

    assert first == second
    assert first.version_identity == third.version_identity
    assert first.publication_identity != third.publication_identity
    assert first.source_artifact == str(source.resolve())
    assert first.output_artifact != first.source_artifact
    assert Path(first.output_artifact).name == "scan_reintegrate1d.nexus"
    # The version identity is still deterministic and still distinguishes
    # requests (asserted above); it simply never reaches the public name.
    assert first.version_identity[:32] not in first.output_artifact
    expected_version_payload = {
        "algorithm_identity": first.algorithm_identity,
        "domain": "xdart.finite-artifact-version.v1",
        "entry": first.entry,
        "operation_kind": first.operation_kind,
        "output_schema": first.output_schema,
        "preservation_identity": first.preservation_identity,
        "scientific_identity": first.scientific_identity,
        "source_graph_identity": first.source_graph_identity,
    }
    expected_version_json = json.dumps(
        expected_version_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert first.canonical_version_json == expected_version_json
    assert first.version_identity == hashlib.sha256(
        expected_version_json.encode("utf-8")
    ).hexdigest()
    expected_publication_payload = {
        "domain": "xdart.finite-artifact-publication.v1",
        "output_artifact": first.output_artifact,
        "version_identity": first.version_identity,
    }
    expected_publication_json = json.dumps(
        expected_publication_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert first.publication_identity == hashlib.sha256(
        expected_publication_json.encode("utf-8")
    ).hexdigest()
    expected_operation_payload = {
        "domain": "xdart.finite-artifact-operation.v1",
        "operation_context_identity": first.operation_context_identity,
        "output_artifact": first.output_artifact,
        "publication_identity": first.publication_identity,
        "source_artifact": first.source_artifact,
        "version_identity": first.version_identity,
    }
    expected_operation_json = json.dumps(
        expected_operation_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert first.operation_identity == hashlib.sha256(
        expected_operation_json.encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    "changed_field",
    (
        "request_generation_identity",
        "resource_allocation_identity",
        "route_identity",
        "custody_identity",
    ),
)
def test_operation_context_is_required_canonical_and_attempt_only(
    tmp_path: Path,
    changed_field: str,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    admission = capture_finite_source(source)
    values = {
        "request_generation_identity": _digest("request-generation"),
        "resource_allocation_identity": _digest("resource-allocation"),
        "route_identity": _digest("route"),
        "custody_identity": _digest("custody"),
    }
    first_context = finite_operation_context(admission, **values)
    first = _request(
        tmp_path,
        source,
        admission=admission,
        operation_context=first_context,
    )
    values[changed_field] = _digest(f"changed-{changed_field}")
    changed_context = finite_operation_context(admission, **values)
    changed = _request(
        tmp_path,
        source,
        admission=admission,
        operation_context=changed_context,
    )

    context_payload = json.loads(first_context.canonical_json)
    assert set(context_payload) == {
        "custody_identity",
        "domain",
        "request_generation_identity",
        "resource_allocation_identity",
        "route_identity",
        "source_snapshot",
    }
    assert set(context_payload["source_snapshot"]) == {
        "ctime_ns",
        "device",
        "digest",
        "inode",
        "mode",
        "mtime_ns",
        "size",
    }
    assert first_context.context_identity != changed_context.context_identity
    assert first.operation_identity != changed.operation_identity
    assert first.version_identity == changed.version_identity
    assert first.publication_identity == changed.publication_identity
    assert first.lineage == changed.lineage


def test_operation_context_payload_fixed_snapshot_bytes_and_hash() -> None:
    snapshot = FiniteFileSnapshot(
        path="/fixed/source.nexus",
        size=3,
        digest="e" * 64,
        device=4,
        inode=5,
        mode=6,
        mtime_ns=7,
        ctime_ns=8,
    )
    payload = finite_module._operation_context_payload_values(
        snapshot,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
    )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    expected = (
        '{"custody_identity":"' + "d" * 64
        + '","domain":"xdart.finite-artifact-operation-context.v1",'
        '"request_generation_identity":"' + "a" * 64
        + '","resource_allocation_identity":"' + "b" * 64
        + '","route_identity":"' + "c" * 64
        + '","source_snapshot":{"ctime_ns":8,"device":4,"digest":"'
        + "e" * 64
        + '","inode":5,"mode":6,"mtime_ns":7,"size":3}}'
    )
    assert canonical == expected
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (
        "975a9d6a3f5968c8e10a832fcab56e394e02db60a52239618aa2df59b771bef0"
    )


def test_lineage_is_canonical_acyclic_and_excludes_attempt_facts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source, entry="entrée")
    payload = json.loads(request.lineage.canonical_json)

    assert set(payload) == {
        "algorithm_identity",
        "artifact_family_v1",
        "entry",
        "operation_kind",
        "output_schema",
        "predecessor",
        "preservation_identity",
        "publication_identity",
        "publication_policy",
        "schema",
        "scientific_identity",
        "source_graph_identity",
        "version_identity",
    }
    assert set(payload["predecessor"]) == {
        "artifact_family_v1",
        "lineage_identity",
        "publication_identity",
        "source_artifact",
        "source_digest",
        "source_size",
        "terminal",
        "version_identity",
    }
    assert "entrée" in request.canonical_version_json
    assert "\\u00e9" not in request.canonical_version_json
    assert not {
        "operation_identity",
        "operation_context_identity",
        "route_identity",
        "custody_identity",
    } & set(payload)
    assert admit_finite_artifact_lineage(
        request.lineage.canonical_json.encode("utf-8")
    ) == request.lineage
    tampered = dict(payload)
    tampered["operation_identity"] = request.operation_identity
    with pytest.raises(ValueError, match="schema or canonical"):
        admit_finite_artifact_lineage(json.dumps(
            tampered,
            sort_keys=True,
            separators=(",", ":"),
        ))
    with pytest.raises(ValueError, match="encoded ceiling"):
        admit_finite_artifact_lineage(b"x" * (FINITE_LINEAGE_MAX_BYTES + 1))


def test_finite_predecessor_lineage_is_one_level_not_recursive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    first_request = _request(tmp_path, source)
    first_result = _publish(first_request)
    assert first_result.terminal is not None
    successor_source = Path(first_request.output_artifact)
    admission = capture_finite_source(successor_source)
    predecessor = capture_finite_predecessor(
        admission,
        terminal=first_result.terminal,
        artifact_family_v1=first_request.artifact_family,
        version_identity=first_request.version_identity,
        publication_identity=first_request.publication_identity,
        lineage_identity=first_request.lineage.lineage_identity,
    )
    second = _request(
        tmp_path,
        successor_source,
        family=first_request.artifact_family,
        operation_kind="reintegrate-2d",
        admission=admission,
        predecessor=predecessor,
    )
    parent = json.loads(second.lineage.canonical_json)["predecessor"]
    assert parent["version_identity"] == first_request.version_identity
    assert parent["publication_identity"] == first_request.publication_identity
    assert parent["lineage_identity"] == first_request.lineage.lineage_identity
    assert "predecessor" not in parent


@pytest.mark.parametrize(
    "field",
    ("version_identity", "publication_identity", "operation_identity"),
)
def test_request_recomputes_and_refuses_tampered_derived_identities(
    tmp_path: Path,
    field: str,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    with pytest.raises(ValueError, match="identities are not canonical"):
        replace(request, **{field: _digest(f"tampered-{field}")})


def test_explicit_target_no_longer_names_the_artifact(tmp_path: Path) -> None:
    """An explicit target constrains the DIRECTORY only; the slot names the file.

    This replaces the superseded contract in which an absent explicit target was
    honored verbatim and an occupied one fell back to a version-named sibling.
    Both halves are gone by policy: honouring a caller's filename would bypass
    the closed vocabulary, and the version-named fallback was the accumulating
    public sequence ADR-0010 forbids.
    """
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    explicit = tmp_path / "chosen.h5"

    first = _request(tmp_path, source, explicit_target=explicit)
    assert first.output_artifact == str((tmp_path / "scan_reintegrate1d.nexus").resolve())
    assert "chosen" not in first.output_artifact

    # An OCCUPIED slot is still the same slot -- a repeat replaces its own
    # result rather than accumulating a sibling.  Whether the replacement is
    # allowed is the transaction's decision, not the namer's.
    Path(first.output_artifact).write_bytes(b"foreign")
    second = _request(tmp_path, source, explicit_target=explicit)
    assert second.output_artifact == first.output_artifact
    assert Path(first.output_artifact).read_bytes() == b"foreign"


def test_explicit_target_in_another_directory_is_refused(tmp_path: Path) -> None:
    """A cross-directory request is refused, never quietly redirected.

    The explicit target no longer names the file, so the only honest options are
    to refuse or to silently write somewhere the caller did not ask for.  Refuse.
    """
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(ValueError, match="destination directory"):
        _request(tmp_path, source, explicit_target=elsewhere / "chosen.nexus")


@pytest.mark.parametrize(
    "mutation",
    (
        {"operation_kind": "Bad Kind"},
        {"source_graph_identity": "0" * 63},
        {"entry": ""},
        {"output_schema": ""},
        # A space is now a legal family character, so the invalid cases must be
        # structurally invalid: a path separator and a leading dot.
        {"artifact_family": "bad/family"},
        {"artifact_family": ".hidden"},
    ),
)
def test_request_refuses_noncanonical_or_unbounded_identity_inputs(
    tmp_path: Path,
    mutation: dict[str, str],
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    admission = capture_finite_source(source)
    values = {
        "source_admission": admission,
        "operation_context": finite_operation_context(
            admission,
            request_generation_identity=_digest("request-generation"),
            resource_allocation_identity=_digest("resource-allocation"),
            route_identity=_digest("route"),
            custody_identity=_digest("custody"),
        ),
        "predecessor": capture_finite_predecessor(admission),
        "destination_directory": tmp_path,
        "artifact_family": "scan",
        "operation_kind": "reintegrate-1d",
        "source_graph_identity": _digest("source-graph"),
        "entry": "entry",
        "scientific_identity": _digest("science"),
        "output_schema": "xdart-current-v4",
        "algorithm_identity": _digest("algorithm"),
        "preservation_identity": _digest("preservation"),
    }
    values.update(mutation)
    with pytest.raises((TypeError, ValueError)):
        finite_artifact_request(**values)


def test_private_candidate_is_factory_owned_mode_0600_and_not_public_nexus(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    observed: dict[str, object] = {}

    def writer(binding: FiniteCandidateBinding) -> None:
        observed["binding"] = binding
        assert not binding.closed
        assert not hasattr(binding, "path")
        assert not hasattr(binding, "name")
        assert not hasattr(binding, "fileno")
        binding.truncate(0)
        binding.write(_payload(request))

    def prepublish() -> None:
        candidates = tuple(tmp_path.glob(".xdart-finite-*.candidate"))
        assert len(candidates) == 1
        candidate = candidates[0]
        observed["path"] = candidate
        observed["mode"] = stat.S_IMODE(candidate.stat().st_mode)
        assert candidate.name.startswith(
            f".xdart-finite-{request.version_identity}-"
        )
        with pytest.raises(ValueError, match="must end in .nexus"):
            require_current_output_path(candidate)

    result = _publish(request, writer=writer, prepublish=prepublish)
    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert observed["mode"] == 0o600
    assert not Path(observed["path"]).exists()
    binding = observed["binding"]
    assert binding.closed
    with pytest.raises(ValueError, match="revoked"):
        binding.write(b"late")
    for action in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            action(binding)


def test_seed_receipt_proves_exact_copy_and_source_immutability(tmp_path: Path) -> None:
    source = tmp_path / "source.nexus"
    original = os.urandom(65_537)
    source.write_bytes(original)
    admitted = capture_finite_source(source)
    request = _request(tmp_path, source)

    def writer(binding: FiniteCandidateBinding) -> None:
        binding.seek(0)
        assert binding.read(len(original) + 1) == original
        binding.seek(0)
        binding.truncate(0)
        binding.write(_payload(request))

    result = _publish(request, seed=admitted, writer=writer)
    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert result.seed_receipt is not None
    assert result.seed_receipt.copy_strategy == "bounded-copy-v1"
    assert result.seed_receipt.source_digest == hashlib.sha256(original).hexdigest()
    assert result.seed_receipt.candidate_digest == result.seed_receipt.source_digest
    assert result.seed_receipt.byte_count == len(original)
    assert source.read_bytes() == original
    assert capture_finite_source(source) == admitted
    with pytest.raises(TypeError, match="capture-owned"):
        type(admitted)(admitted.snapshot)


def test_seeded_adapter_receives_only_publisher_owned_copy_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nexus"
    original = os.urandom(4097)
    source.write_bytes(original)
    admitted = capture_finite_source(source)
    request = _request(tmp_path, source)
    observed: dict[str, object] = {}
    accepted: list[FiniteCandidateValidation] = []
    inspected: list[Path] = []
    target_hashes: list[Path] = []
    snapshot_at = finite_module._snapshot_at

    def capture_snapshot(parent_descriptor, name, shown_path):
        shown = Path(shown_path)
        if shown == Path(request.output_artifact):
            target_hashes.append(shown)
        return snapshot_at(parent_descriptor, name, shown_path)

    def inspect(path, selected):
        inspected.append(path)
        return _inspect_payload(path, selected)

    monkeypatch.setattr(finite_module, "_snapshot_at", capture_snapshot)

    def writer(
        candidate: FiniteCandidateBinding,
        seed_binding: FiniteSeedBinding,
        admitted_candidate: FiniteCandidateBinding,
    ) -> None:
        assert type(seed_binding) is FiniteSeedBinding
        assert seed_binding.request is request
        assert seed_binding.receipt.source_snapshot is admitted.snapshot
        assert seed_binding.candidate_identity == candidate.candidate_identity
        assert admitted_candidate is candidate
        assert seed_binding.authorizes(admitted_candidate)
        candidate.seek(0)
        assert candidate.read(len(original) + 1) == original
        candidate.seek(0)
        candidate.truncate(0)
        candidate.write(_payload(request))
        observed["binding"] = seed_binding

    adapter = FiniteSeededDocumentAdapter(
        lambda binding: nullcontext(binding),
        writer,
        lambda binding: nullcontext(binding),
        _validate_payload(request),
    )
    result = FiniteArtifactPublisher(request).publish(
        adapter,
        inspect_committed=inspect,
        seed=admitted,
        accept_validated_commit=accepted.append,
    )

    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert observed["binding"].receipt is result.seed_receipt
    assert observed["binding"].active is False
    assert len(accepted) == 1
    assert accepted[0].lineage == request.lineage
    assert inspected == []
    assert target_hashes == [Path(request.output_artifact)]
    replay = FiniteArtifactPublisher(request).publish(
        adapter,
        inspect_committed=inspect,
        seed=admitted,
        accept_validated_commit=accepted.append,
    )
    # A repeat REBUILDS and replaces (ruling 2026-09-04); it no longer short
    # circuits to ALREADY_COMMITTED on finding the slot occupied.
    assert replay.disposition is FiniteArtifactDisposition.COMMITTED
    # NOT inspected.  The replay is a fresh, validated commit accepted through
    # the seeded shortcut, so it takes `_accept_validated_own_link`.  Previously
    # it found the slot occupied, returned ALREADY_COMMITTED and had to INSPECT
    # someone else's file to describe it -- the publisher no longer reads an
    # occupant at all, which is why this list is empty.
    assert inspected == []
    assert len(accepted) == 2
    with pytest.raises(TypeError, match="factory-owned"):
        replace(result.seed_receipt)
    with pytest.raises(TypeError, match="publisher-owned"):
        FiniteSeedBinding(request, result.seed_receipt, "0" * 64, object())
    for action in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            action(observed["binding"])


def test_seeded_adapter_refuses_unseeded_publication(tmp_path: Path) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    adapter = FiniteSeededDocumentAdapter(
        lambda binding: nullcontext(binding),
        lambda _document, _binding, _candidate: None,
        lambda binding: nullcontext(binding),
        _validate_payload(request),
    )

    with pytest.raises(TypeError, match="requires one source admission"):
        FiniteArtifactPublisher(request).publish(
            adapter,
            inspect_committed=_inspect_payload,
        )


def test_validated_commit_callback_refuses_an_unseeded_adapter(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    adapter = FiniteDocumentAdapter(
        lambda binding: nullcontext(binding),
        _write_payload(request),
        lambda binding: nullcontext(binding),
        _validate_payload(request),
    )

    with pytest.raises(TypeError, match="requires a seeded adapter"):
        FiniteArtifactPublisher(request).publish(
            adapter,
            inspect_committed=_inspect_payload,
            accept_validated_commit=lambda _validation: None,
        )


def test_validated_commit_hash_mismatch_is_held_after_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    admitted = capture_finite_source(source)
    request = _request(tmp_path, source)
    target = Path(request.output_artifact)
    accepted: list[FiniteCandidateValidation] = []
    snapshot_at = finite_module._snapshot_at
    target_hashes = 0

    def mutate_before_public_hash(parent_descriptor, name, shown_path):
        nonlocal target_hashes
        shown = Path(shown_path)
        if shown == target:
            target_hashes += 1
            state = shown.stat()
            payload = bytearray(shown.read_bytes())
            payload[0] ^= 1
            with shown.open("r+b") as stream:
                stream.write(payload)
            os.utime(
                shown,
                ns=(state.st_atime_ns, state.st_mtime_ns),
                follow_symlinks=False,
            )
        return snapshot_at(parent_descriptor, name, shown_path)

    monkeypatch.setattr(
        finite_module, "_snapshot_at", mutate_before_public_hash,
    )

    with pytest.raises(FiniteArtifactPublicationHeld) as captured:
        _publish_validated_seed(
            request, admitted, accepted.append,
        )

    assert isinstance(captured.value.cause, FiniteArtifactIntegrityError)
    assert target.exists()
    assert target_hashes == 1
    assert accepted == []
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize(
    "failure", ("post-hash-mutation", "callback-mutation", "callback"),
)
def test_validated_commit_late_failure_is_held_after_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    admitted = capture_finite_source(source)
    request = _request(tmp_path, source)
    target = Path(request.output_artifact)
    callbacks: list[FiniteCandidateValidation] = []
    snapshot_at = finite_module._snapshot_at

    def mutate_target() -> None:
        state = target.stat()
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 1
        with target.open("r+b") as stream:
            stream.write(payload)
        os.utime(
            target,
            ns=(state.st_atime_ns, state.st_mtime_ns),
            follow_symlinks=False,
        )

    def hash_then_mutate(parent_descriptor, name, shown_path):
        snapshot = snapshot_at(parent_descriptor, name, shown_path)
        if Path(shown_path) == target:
            mutate_target()
        return snapshot

    def accept(validation):
        callbacks.append(validation)
        if failure == "callback":
            raise LookupError("domain acceptance unavailable")
        if failure == "callback-mutation":
            mutate_target()

    if failure == "post-hash-mutation":
        monkeypatch.setattr(
            finite_module, "_snapshot_at", hash_then_mutate,
        )
    with pytest.raises(FiniteArtifactPublicationHeld) as captured:
        _publish_validated_seed(
            request,
            admitted,
            accept,
            inspect=lambda *_args: pytest.fail(
                "validated own link entered replay inspection"
            ),
        )

    assert isinstance(captured.value.cause, FiniteArtifactIntegrityError)
    assert target.exists()
    assert len(callbacks) == (0 if failure == "post-hash-mutation" else 1)
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


def test_validated_candidate_same_size_mutation_is_refused_before_visibility(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    admitted = capture_finite_source(source)
    request = _request(tmp_path, source)
    accepted: list[FiniteCandidateValidation] = []

    def mutate_candidate() -> None:
        candidate, = tuple(tmp_path.glob(".xdart-finite-*.candidate"))
        state = candidate.stat()
        payload = bytearray(candidate.read_bytes())
        payload[-1] ^= 1
        with candidate.open("r+b") as stream:
            stream.write(payload)
        os.utime(
            candidate,
            ns=(state.st_atime_ns, state.st_mtime_ns),
            follow_symlinks=False,
        )

    with pytest.raises(
        FiniteArtifactIntegrityError,
        match="candidate changed before publication",
    ):
        _publish_validated_seed(
            request,
            admitted,
            accepted.append,
            prepublish=mutate_candidate,
        )

    assert accepted == []
    assert not Path(request.output_artifact).exists()
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize("late_effect", ("link-error", "stop"))
def test_validated_commit_keeps_an_observed_own_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    late_effect: str,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    admitted = capture_finite_source(source)
    request = _request(tmp_path, source)
    token = threading.Event()
    accepted: list[FiniteCandidateValidation] = []
    # Patched onto `replace_into_place`, not `_link`: nothing calls `_link` any
    # more, so this injection had gone DEAD and the row proved nothing.
    publish = finite_module.replace_into_place

    def publish_with_late_effect(*args, **kwargs):
        publish(*args, **kwargs)
        if late_effect == "stop":
            token.set()
        else:
            raise OSError("uncertain after exact publication")

    monkeypatch.setattr(
        finite_module, "replace_into_place", publish_with_late_effect,
    )
    result = _publish_validated_seed(
        request,
        admitted,
        accepted.append,
        publisher=FiniteArtifactPublisher(request, cancel_token=token),
        inspect=lambda *_args: pytest.fail(
            "validated own link entered replay inspection"
        ),
    )

    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert Path(request.output_artifact).read_bytes() == _payload(request)
    assert len(accepted) == 1
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


def test_seed_publication_does_not_rehash_the_admitted_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(os.urandom(32_769))
    admitted = capture_finite_source(source)
    request = _request(tmp_path, source)
    real_snapshot = finite_module._snapshot_at
    captures: list[str] = []

    def spy(parent_descriptor, name, path):
        captures.append(str(Path(path)))
        return real_snapshot(parent_descriptor, name, path)

    monkeypatch.setattr(finite_module, "_snapshot_at", spy)
    result = _publish(request, seed=admitted)
    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert str(source) not in captures
    assert len(captures) == 3


@pytest.mark.parametrize("reuse_inode", (False, True))
def test_seed_substitution_is_detected_before_foreign_file_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reuse_inode: bool,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source seed")
    admission = capture_finite_source(source)
    request = _request(tmp_path, source, admission=admission)
    real_open = finite_module.os.open
    real_fstat = finite_module.os.fstat
    real_stat = finite_module.os.stat
    swapped: list[Path] = []
    identities: dict[str, tuple[int, int]] = {}
    reused_descriptor_observations: list[int] = []

    def reused_state(observed):
        if (
            reuse_inode
            and (observed.st_dev, observed.st_ino) == identities.get("foreign")
        ):
            # Model Linux recycling the unlinked reservation's inode. Keep
            # every other stat field from the real foreign file, and apply the
            # same identity to both descriptor admission and pathname cleanup.
            values = {
                name: getattr(observed, name)
                for name in dir(observed) if name.startswith("st_")
            }
            values["st_dev"], values["st_ino"] = identities["reserved"]
            return SimpleNamespace(**values)
        return observed

    def fstat(descriptor):
        observed = real_fstat(descriptor)
        result = reused_state(observed)
        if result is not observed:
            reused_descriptor_observations.append(descriptor)
        return result

    def named_stat(*args, **kwargs):
        return reused_state(real_stat(*args, **kwargs))

    def fault(path, flags, *args, **kwargs):
        dir_fd = kwargs.get("dir_fd")
        if (
            not swapped
            and dir_fd is not None
            and type(path) is str
            and path.endswith(".candidate")
            and flags & os.O_ACCMODE == os.O_WRONLY
        ):
            candidate = tmp_path / path
            reserved = real_stat(candidate)
            foreign = tmp_path / "foreign-replacement"
            foreign.write_bytes(b"foreign must survive")
            replacement = real_stat(foreign)
            identities["reserved"] = (reserved.st_dev, reserved.st_ino)
            identities["foreign"] = (replacement.st_dev, replacement.st_ino)
            assert identities["reserved"] != identities["foreign"]
            foreign.replace(candidate)
            swapped.append(candidate)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(finite_module.os, "open", fault)
    monkeypatch.setattr(finite_module.os, "fstat", fstat)
    monkeypatch.setattr(finite_module.os, "stat", named_stat)
    with pytest.raises(FiniteArtifactIntegrityError, match="descriptor identity"):
        _publish(request, seed=admission)
    assert bool(reused_descriptor_observations) is reuse_inode
    assert swapped[0].read_bytes() == b"foreign must survive"
    assert source.read_bytes() == b"source seed"
    assert not Path(request.output_artifact).exists()


def test_seed_partial_copy_failure_cleans_owned_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nexus"
    original = b"source seed" * 100_000
    source.write_bytes(original)
    admission = capture_finite_source(source)
    request = _request(tmp_path, source, admission=admission)
    real_read = finite_module.os.read
    source_reads = 0
    partial_copies: list[bytes] = []

    def fail_after_first_block(descriptor, size):
        nonlocal source_reads
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == (
            admission.snapshot.device, admission.snapshot.inode,
        ):
            source_reads += 1
            if source_reads == 2:
                candidate, = tmp_path.glob(".xdart-finite-*.candidate")
                partial_copies.append(candidate.read_bytes())
                raise OSError("source read failed after a partial seed copy")
        return real_read(descriptor, size)

    monkeypatch.setattr(finite_module.os, "read", fail_after_first_block)
    with pytest.raises(OSError, match="partial seed copy"):
        _publish(request, seed=admission)
    assert len(partial_copies) == 1
    assert 0 < len(partial_copies[0]) < len(original)
    assert original.startswith(partial_copies[0])
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))
    assert source.read_bytes() == original
    assert not Path(request.output_artifact).exists()


@pytest.mark.parametrize(
    "cancel_seam",
    ("before", "after-writer", "prepublish", "last-source-fence"),
)
def test_cancellation_before_publication_aborts_without_mutating_input(
    tmp_path: Path,
    cancel_seam: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nexus"
    original = b"immutable source"
    source.write_bytes(original)
    request = _request(tmp_path, source)
    token = threading.Event()
    calls: list[str] = []
    if cancel_seam == "before":
        token.set()
    source_checks = 0
    original_source_check = FiniteArtifactPublisher._require_request_source

    def source_check(publisher):
        nonlocal source_checks
        observed = original_source_check(publisher)
        source_checks += 1
        if cancel_seam == "last-source-fence" and source_checks == 2:
            token.set()
        return observed

    monkeypatch.setattr(
        FiniteArtifactPublisher, "_require_request_source", source_check,
    )

    def writer(binding: FiniteCandidateBinding) -> None:
        calls.append("writer")
        binding.truncate(0)
        binding.write(_payload(request))
        if cancel_seam == "after-writer":
            token.set()

    def prepublish() -> None:
        calls.append("prepublish")
        if cancel_seam == "prepublish":
            token.set()

    result = _publish(
        request,
        publisher=FiniteArtifactPublisher(request, cancel_token=token),
        writer=writer,
        prepublish=prepublish,
    )
    assert result.disposition is FiniteArtifactDisposition.ABORTED
    assert result.terminal is None
    assert not Path(request.output_artifact).exists()
    assert source.read_bytes() == original
    assert calls == ([] if cancel_seam == "before" else ["writer"] + (
        ["prepublish"]
        if cancel_seam in {"prepublish", "last-source-fence"}
        else []
    ))


@pytest.mark.parametrize(
    "seam",
    ("writer", "validator", "candidate-fsync", "prepublish", "source-drift"),
)
def test_prepublish_failures_preserve_primary_source_and_absent_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
) -> None:
    source = tmp_path / "source.nexus"
    original = b"source"
    source.write_bytes(original)
    admitted = capture_finite_source(source)
    request = _request(tmp_path, source)

    def writer(binding: FiniteCandidateBinding) -> None:
        if seam == "writer":
            raise LookupError("writer-primary")
        binding.truncate(0)
        binding.write(_payload(request))

    def validate(binding: FiniteCandidateBinding) -> FiniteCandidateValidation:
        if seam == "validator":
            raise LookupError("validator-primary")
        binding.seek(0)
        assert binding.read(len(_payload(request)) + 1) == _payload(request)
        return FiniteCandidateValidation(request.lineage)

    def prepublish() -> None:
        if seam == "prepublish":
            raise LookupError("prepublish-primary")
        if seam == "source-drift":
            source.write_bytes(b"changed")

    if seam == "candidate-fsync":
        monkeypatch.setattr(
            finite_module,
            "_fsync",
            lambda _descriptor: (_ for _ in ()).throw(
                OSError("file durability unavailable")
            ),
        )
    expected = {
        "source-drift": FiniteArtifactIntegrityError,
        "candidate-fsync": OSError,
    }.get(seam, LookupError)
    with pytest.raises(expected):
        _publish(
            request,
            seed=admitted,
            writer=writer,
            validate=validate,
            prepublish=prepublish,
        )
    assert not Path(request.output_artifact).exists()
    assert source.read_bytes() == (b"changed" if seam == "source-drift" else original)
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize(
    "seam",
    (
        "changed-before-publish",
        "changed-without-seed-before-link",
        # "changed-during-existing-inspection" REMOVED: it drove the occupant
        # inspection that ran before an ALREADY_COMMITTED reuse, and under
        # unconditional replacement the publisher never inspects an occupant.
        # The seam is gone, so the row would have asserted nothing.
    ),
)
def test_request_source_snapshot_is_revalidated_without_rehashing(
    tmp_path: Path,
    seam: str,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"admitted source")
    admitted = capture_finite_source(source)
    request = _request(tmp_path, source, admission=admitted)
    calls: list[str] = []
    seed = None

    if seam == "changed-before-publish":
        source.write_bytes(b"later source")
        seed = capture_finite_source(source)

    def writer(binding: FiniteCandidateBinding) -> None:
        calls.append("writer")
        binding.truncate(0)
        binding.write(_payload(request))

    def prepublish() -> None:
        calls.append("prepublish")
        if seam == "changed-without-seed-before-link":
            source.write_bytes(b"changed before link")

    with pytest.raises(
        FiniteArtifactIntegrityError,
        match="request source changed|seed admission does not match",
    ):
        _publish(
            request,
            seed=seed,
            writer=writer,
            prepublish=prepublish,
        )

    assert calls == (
        []
        if seam == "changed-before-publish"
        else ["writer", "prepublish"]
    )
    assert not Path(request.output_artifact).exists()
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


def test_an_identical_repeat_rebuilds_and_replaces_rather_than_reusing(
    tmp_path: Path,
) -> None:
    """A second identical publication does the work again and replaces.

    MAINTAINER RULING 2026-09-04, recorded here rather than assumed: an occupied
    slot is replaced UNCONDITIONALLY, and an IDENTICAL repeat replaces too --
    there is no no-op fast path.  The publisher no longer inspects the occupant
    at all, so it cannot be fooled by one; the CONCURRENT case is held off by
    the H23 slot hold instead of by a no-clobber link.

    This row previously asserted the opposite -- that the writer and validator
    were NOT replayed and the inode was unchanged.  That was the no-op fast
    path, and pinning its removal is the point of this row.
    """
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    first = _publish(request)
    before = Path(request.output_artifact).stat()
    calls: list[str] = []

    def counted_write(binding: FiniteCandidateBinding) -> None:
        calls.append("write")
        _write_payload(request)(binding)

    def counted_validate(binding):
        calls.append("validate")
        # MUST return the validation: it is what proves exact lineage.
        return _validate_payload(request)(binding)

    second = _publish(request, writer=counted_write, validate=counted_validate)
    after = Path(request.output_artifact).stat()

    assert first.disposition is FiniteArtifactDisposition.COMMITTED
    assert second.disposition is FiniteArtifactDisposition.COMMITTED
    # The work IS redone...
    assert calls == ["write", "validate"]
    # ...and a genuinely different file now occupies the slot.
    assert (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
    assert Path(request.output_artifact).read_bytes() == _payload(request)


@pytest.mark.parametrize("occupant", ("foreign", "symlink", "short-tag-collision"))
def test_an_existing_foreign_or_mismatched_occupant_is_replaced(
    tmp_path: Path,
    occupant: str,
) -> None:
    """Whatever sits at the slot is replaced.  This is the data-loss boundary.

    MAINTAINER RULING 2026-09-04, recorded here rather than assumed: an occupied
    slot is replaced UNCONDITIONALLY, and an IDENTICAL repeat replaces too --
    there is no no-op fast path.  The publisher no longer inspects the occupant
    at all, so it cannot be fooled by one; the CONCURRENT case is held off by
    the H23 slot hold instead of by a no-clobber link.

    Pinned per occupant, because each says something different:

    * `foreign` -- a hand-placed file at the slot name IS DESTROYED.  That is
      the accepted cost of the unconditional rule and it should be visible in a
      test rather than discovered in a folder.
    * `symlink` -- `os.replace` replaces the LINK, not what it points at, so the
      referent survives untouched.  A real safety property that the ruling does
      not change.
    * `short-tag-collision` -- a version-mismatched occupant.  This is exactly
      the "change npt and run again" workflow, which used to raise
      `FiniteArtifactCollision` and now simply succeeds.
    """
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    target = Path(request.output_artifact)
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"foreign")
    if occupant == "symlink":
        target.symlink_to(foreign)
    elif occupant == "short-tag-collision":
        target.write_bytes(_payload(request).replace(
            request.version_identity.encode(), _digest("other-version").encode()
        ))
    else:
        target.write_bytes(b"foreign")

    result = _publish(request)

    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert target.read_bytes() == _payload(request)
    assert not target.is_symlink()
    # `os.replace` onto a symlink consumes the LINK; the referent is untouched.
    assert foreign.read_bytes() == b"foreign"


@pytest.mark.parametrize("fault", ("before", "after"))
def test_a_failed_replacement_leaves_the_prior_slot_whole(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """The ADR guarantee: the prior slot is untouched until the rename lands.

    This replaced a row that injected faults into `finite_module._link`.  That
    function is no longer called, so every one of those injections had become
    DEAD -- the test still passed, but proved nothing.  A monkeypatch that never
    fires is worse than a red test, so the fault now targets the seam that
    actually exists.

    Its `exact-winner`, `foreign` and `symlink` cases are gone with the
    no-clobber semantics: under unconditional replacement there is no loser to
    observe an occupant.  Replacement of an occupant is pinned by
    `test_an_existing_foreign_or_mismatched_occupant_is_replaced`.
    """
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    target = Path(request.output_artifact)

    # A prior result really occupying the slot, so "untouched" is observable.
    _publish(request)
    prior_bytes = target.read_bytes()
    prior = target.stat()

    real_replace = finite_module.replace_into_place

    def faulty(*args, **kwargs):
        if fault == "after":
            real_replace(*args, **kwargs)
            raise OSError("uncertain after rename")
        raise OSError("known before rename")

    monkeypatch.setattr(finite_module, "replace_into_place", faulty)

    if fault == "before":
        # No effect reached the slot: the PRIOR file is still exactly there.
        with pytest.raises(OSError, match="known before"):
            _publish(request)
        after = target.stat()
        assert target.read_bytes() == prior_bytes
        assert (after.st_dev, after.st_ino) == (prior.st_dev, prior.st_ino)
        return

    # The rename DID land and then reported failure.  The observation shows our
    # exact inode, so the effect is proven ours and the publication COMMITS --
    # a spurious error after a verified rename must not be re-read as a
    # pre-publication abort.  (Same expectation the old no-clobber row had for
    # its "after" fault; only the injected seam moved.)
    result = _publish(request)
    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert target.read_bytes() == _payload(request)
    assert (target.stat().st_dev, target.stat().st_ino) != (
        prior.st_dev, prior.st_ino
    )


def test_own_link_commits_and_fsyncs_if_private_alias_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    real_parent_fsync = finite_module._fsync_parent
    fsync_calls: list[int] = []

    def observed_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_parent_fsync(descriptor)

    # No injection needed any more.  This row used to LINK and then unlink the
    # private alias by hand, to prove the publication survives its
    # disappearance.  `os.replace` consumes that alias as part of publishing,
    # so the condition the row was constructing is now simply what happens.
    monkeypatch.setattr(finite_module, "_fsync_parent", observed_fsync)
    result = _publish(request)
    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert len(fsync_calls) == 1
    assert result.hidden_orphan is None
    assert Path(request.output_artifact).read_bytes() == _payload(request)


@pytest.mark.parametrize("seam", ("existing", "prepublish"))
def test_output_parent_symlink_substitution_is_refused(
    tmp_path: Path,
    seam: str,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    destination = tmp_path / "destination"
    destination.mkdir()
    request = _request(destination, source)
    if seam == "existing":
        _publish(request)
    moved = tmp_path / "moved-destination"

    def substitute() -> None:
        destination.rename(moved)
        destination.symlink_to(moved, target_is_directory=True)

    if seam == "existing":
        substitute()
        with pytest.raises(FiniteArtifactIntegrityError, match="parent"):
            _publish(request)
    else:
        with pytest.raises(FiniteArtifactIntegrityError, match="parent"):
            _publish(request, prepublish=substitute)
        assert not (moved / Path(request.output_artifact).name).exists()
    destination.unlink()
    moved.rename(destination)


def test_delayed_writer_capability_cannot_mutate_published_final(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    release = threading.Event()
    attempted = threading.Event()
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []

    def writer(binding: FiniteCandidateBinding) -> None:
        binding.truncate(0)
        binding.write(_payload(request))

        def late_write() -> None:
            release.wait()
            try:
                binding.seek(0)
                binding.write(b"late corruption")
            except BaseException as error:
                errors.append(error)
            finally:
                attempted.set()

        thread = threading.Thread(target=late_write)
        thread.start()
        threads.append(thread)

    result = _publish(request, writer=writer)
    assert result.terminal is not None
    release.set()
    assert attempted.wait(2.0)
    threads[0].join(timeout=2.0)
    assert not threads[0].is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert Path(request.output_artifact).read_bytes() == _payload(request)
    assert capture_target_snapshot(request.output_artifact).digest == result.terminal.digest


def test_postpublication_parent_fsync_failure_commits_with_bounded_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)

    def fail_parent(_descriptor: int) -> None:
        assert Path(request.output_artifact).exists()
        raise OSError("directory durability unavailable")

    monkeypatch.setattr(finite_module, "_fsync_parent", fail_parent)
    result = _publish(request)
    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert result.terminal is not None
    assert result.diagnostics == (
        "FINITE_PARENT_DIRECTORY_FSYNC_UNCONFIRMED:OSError:directory durability unavailable",
    )
    assert Path(request.output_artifact).read_bytes() == _payload(request)


@pytest.mark.parametrize(
    "seam",
    ("semantic-inspection", "final-observation", "link-after-effect"),
)
def test_linked_publication_uncertainty_is_typed_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    target = Path(request.output_artifact)
    calls: list[str] = []
    if seam == "semantic-inspection":
        def inspect(
            _path: Path,
            _request: FiniteArtifactRequest,
        ) -> FiniteCommittedInspection:
            calls.append("inspect")
            raise ValueError("terminal semantic mismatch")

        action = lambda: FiniteArtifactPublisher(request).publish(
            FiniteDocumentAdapter(
                lambda binding: nullcontext(binding),
                _write_payload(request),
                lambda binding: nullcontext(binding),
                _validate_payload(request),
            ),
            inspect_committed=inspect,
        )
    else:
        observe = finite_module._try_observe_at

        def fail_final_observation(parent, name, shown):
            if Path(shown) == target and target.exists():
                raise FiniteArtifactIntegrityError(
                    "final observation unavailable"
                )
            return observe(parent, name, shown)

        monkeypatch.setattr(
            finite_module, "_try_observe_at", fail_final_observation,
        )
        if seam == "link-after-effect":
            # Seam moved with the syscall; on `_link` it no longer fired.
            publish = finite_module.replace_into_place

            def publish_then_raise(*args, **kwargs):
                publish(*args, **kwargs)
                raise OSError("uncertain after publication")

            monkeypatch.setattr(
                finite_module, "replace_into_place", publish_then_raise,
            )
        action = lambda: _publish(request)

    with pytest.raises(FiniteArtifactPublicationHeld) as captured:
        action()
    assert captured.value.request is request
    assert isinstance(
        captured.value.cause,
        FiniteArtifactIntegrityError
        if seam == "semantic-inspection" else FiniteArtifactCollision,
    )
    assert calls == (["inspect"] if seam == "semantic-inspection" else [])
    assert target.read_bytes() == _payload(request)
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


def test_publication_result_cannot_be_forged_by_a_session_caller(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)

    with pytest.raises(TypeError, match="result is invalid"):
        FiniteArtifactResult(
            FiniteArtifactDisposition.ABORTED,
            request,
            None,
            None,
            None,
            None,
            (),
        )


def test_candidate_capabilities_are_pathless_revoked_and_validator_read_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    retained: list[FiniteCandidateBinding] = []

    def writer(binding: FiniteCandidateBinding) -> None:
        retained.append(binding)
        binding.truncate(0)
        binding.write(_payload(request))

    def validate(binding: FiniteCandidateBinding) -> FiniteCandidateValidation:
        retained.append(binding)
        with pytest.raises(OSError, match="read-only"):
            binding.write(b"forbidden")
        binding.seek(0)
        assert binding.read(len(_payload(request)) + 1) == _payload(request)
        return FiniteCandidateValidation(request.lineage)

    result = _publish(request, writer=writer, validate=validate)
    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert len(retained) == 2
    assert all(binding.closed for binding in retained)
    for binding in retained:
        with pytest.raises(ValueError, match="revoked"):
            binding.read(1)
        with pytest.raises(ValueError, match="revoked"):
            binding.write(b"late")
    assert Path(request.output_artifact).read_bytes() == _payload(request)


def test_real_hdf5_session_is_closed_and_lineage_is_exact_before_publication(
    tmp_path: Path,
) -> None:
    import h5py

    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    retained = []

    def open_writer(binding: FiniteCandidateBinding):
        handle = h5py.File(binding, "w")
        retained.append(handle)
        return handle

    def write(handle) -> None:
        entry = handle.create_group(request.entry)
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("result", data=[1.0, 2.0, 3.0])
        write_finite_artifact_lineage(handle, request)

    def open_reader(binding: FiniteCandidateBinding):
        handle = h5py.File(binding, "r")
        retained.append(handle)
        return handle

    def validate(handle) -> FiniteCandidateValidation:
        assert handle[f"/{request.entry}/result"][...].tolist() == [1.0, 2.0, 3.0]
        return FiniteCandidateValidation(
            require_finite_artifact_lineage(handle, request)
        )

    def inspect(
        path: Path,
        expected: FiniteArtifactRequest,
    ) -> FiniteCommittedInspection:
        with h5py.File(path, "r") as handle:
            lineage = require_finite_artifact_lineage(handle, expected)
            node = handle[finite_lineage_hdf_path(expected.entry)]
            assert node.shape == (
                len(expected.lineage.canonical_json.encode("utf-8")),
            )
            assert node.maxshape == node.shape
            assert node.dtype.kind == "u"
            assert node.dtype.itemsize == 1
            assert node.chunks is None
            assert len(node.attrs) == 0
        snapshot = capture_target_snapshot(path)
        state = path.stat()
        return FiniteCommittedInspection(
            StreamTerminal(
                str(path),
                snapshot.size,
                snapshot.digest,
                1,
                state.st_dev,
                state.st_ino,
                state.st_mtime_ns,
                state.st_ctime_ns,
            ),
            lineage,
        )

    result = FiniteArtifactPublisher(request).publish(
        FiniteDocumentAdapter(open_writer, write, open_reader, validate),
        inspect_committed=inspect,
    )
    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert len(retained) == 2
    assert all(handle.id.valid == 0 for handle in retained)
    with pytest.raises(ValueError):
        retained[0].create_group("late")


@pytest.mark.parametrize("foreign_kind", ("regular", "symlink", "fifo", "directory"))
def test_visible_foreign_private_occupants_are_preserved_before_cleanup(
    tmp_path: Path,
    foreign_kind: str,
) -> None:
    candidate = tmp_path / ".xdart-finite-visible.candidate"
    candidate.write_bytes(b"owned")
    expected = finite_module._snapshot_path(candidate)
    candidate.unlink()
    foreign_target = tmp_path / "foreign-target"
    foreign_target.write_bytes(b"foreign target")
    if foreign_kind == "regular":
        candidate.write_bytes(b"foreign")
    elif foreign_kind == "symlink":
        candidate.symlink_to(foreign_target)
    elif foreign_kind == "fifo":
        os.mkfifo(candidate)
    else:
        candidate.mkdir()
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        with pytest.raises(FiniteArtifactIntegrityError, match="foreign object"):
            finite_module._unlink_candidate(
                parent_descriptor,
                candidate,
                expected,
            )
    finally:
        os.close(parent_descriptor)
    assert os.path.lexists(candidate)
    if foreign_kind == "regular":
        assert candidate.read_bytes() == b"foreign"
        candidate.unlink()
    elif foreign_kind == "symlink":
        assert candidate.is_symlink()
        candidate.unlink()
    elif foreign_kind == "fifo":
        assert stat.S_ISFIFO(candidate.lstat().st_mode)
        candidate.unlink()
    else:
        candidate.rmdir()


def test_publisher_is_one_shot_even_after_abort(tmp_path: Path) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    token = threading.Event()
    token.set()
    publisher = FiniteArtifactPublisher(request, cancel_token=token)
    assert _publish(request, publisher=publisher).disposition is FiniteArtifactDisposition.ABORTED
    with pytest.raises(RuntimeError, match="one-shot"):
        _publish(request, publisher=publisher)


def test_cancelled_exact_reuse_is_aborted_without_touching_prior_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    first = _publish(request)
    target = Path(request.output_artifact)
    before = target.read_bytes()
    token = threading.Event()
    token.set()

    second = _publish(
        request,
        publisher=FiniteArtifactPublisher(request, cancel_token=token),
    )

    assert first.disposition is FiniteArtifactDisposition.COMMITTED
    assert second.disposition is FiniteArtifactDisposition.ABORTED
    assert second.terminal is None
    assert target.read_bytes() == before
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize("shortfall", (0, 1))
def test_candidate_capacity_has_one_exact_boundary_and_no_hidden_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shortfall: int,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    required = source.stat().st_size + max(
        finite_module._CANDIDATE_SPACE_MARGIN_BYTES,
        (source.stat().st_size + 9) // 10,
    )
    available = required - shortfall
    monkeypatch.setattr(
        finite_module.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(
            f_frsize=1,
            f_bavail=available,
        ),
    )
    if shortfall == 0:
        result = _publish(request)
        assert result.disposition is FiniteArtifactDisposition.COMMITTED
    else:
        with pytest.raises(FiniteArtifactCapacityError) as captured:
            _publish(request)
        error = captured.value
        assert error.required_bytes == required
        assert error.available_bytes == available
        assert error.cleanup_directory == str(tmp_path)
        assert str(tmp_path) in str(error)
        assert not Path(request.output_artifact).exists()
        assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize(
    "seam",
    ("reservation", "writer-primary", "parent-commit", "parent-abort"),
)
def test_descriptor_close_faults_are_single_attempt_and_never_replace_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    real_close = finite_module._close_descriptor_once
    armed = seam != "writer-primary"
    injected = 0

    def close_once(descriptor: int):
        nonlocal armed, injected
        observed = os.fstat(descriptor)
        is_parent = stat.S_ISDIR(observed.st_mode)
        is_candidate = any(
            path.stat().st_ino == observed.st_ino
            and path.stat().st_dev == observed.st_dev
            for path in tmp_path.glob(".xdart-finite-*.candidate")
        )
        error = real_close(descriptor)
        should_fail = armed and (
            seam == "reservation" and is_candidate
            or seam.startswith("parent-") and is_parent
            or seam == "writer-primary" and is_candidate
        )
        if should_fail and injected == 0:
            injected += 1
            armed = False
            return OSError(f"synthetic {seam} close ambiguity")
        return error

    monkeypatch.setattr(finite_module, "_close_descriptor_once", close_once)
    if seam == "reservation":
        with pytest.raises(
            FiniteArtifactIntegrityError,
            match="reservation close",
        ):
            _publish(request)
    elif seam == "writer-primary":
        def writer(_binding):
            nonlocal armed
            armed = True
            raise LookupError("writer remains primary across close")

        with pytest.raises(LookupError, match="writer remains primary") as captured:
            _publish(request, writer=writer)
        assert any(
            "DESCRIPTOR_CLOSE_INCOMPLETE" in note
            for note in captured.value.__notes__
        )
    elif seam == "parent-commit":
        result = _publish(request)
        assert result.disposition is FiniteArtifactDisposition.COMMITTED
        assert any(
            "DESCRIPTOR_CLOSE_INCOMPLETE" in item
            for item in result.diagnostics
        )
    else:
        token = threading.Event()
        token.set()
        result = _publish(
            request,
            publisher=FiniteArtifactPublisher(request, cancel_token=token),
        )
        assert result.disposition is FiniteArtifactDisposition.ABORTED
        assert any(
            "DESCRIPTOR_CLOSE_INCOMPLETE" in item
            for item in result.diagnostics
        )
    assert injected == 1
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize(
    "outcome",
    ("aborted", "committed", "held-transient", "held-persistent"),
)
def test_candidate_cleanup_failure_reports_exact_hidden_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    real_unlink = finite_module._unlink_candidate
    paths: list[Path] = []

    def fail_cleanup(parent_descriptor: int, path: Path, *args, **kwargs) -> None:
        paths.append(path)
        if outcome == "held-transient" and len(paths) == 2:
            real_unlink(parent_descriptor, path, *args, **kwargs)
            return
        raise PermissionError("candidate cleanup denied")

    monkeypatch.setattr(finite_module, "_unlink_candidate", fail_cleanup)
    if outcome == "committed":
        # NOTHING TO STRAND.  The rename MOVED the candidate onto the slot, so
        # there is no private name left to unlink and the injected cleanup
        # failure is never even reached.  Under the old no-clobber link both
        # names existed after publication and a failed unlink left an orphan.
        result = _publish(request)
        assert result.disposition is FiniteArtifactDisposition.COMMITTED
        assert Path(request.output_artifact).exists()
        assert paths == []
        assert result.hidden_orphan is None
    elif outcome == "aborted":
        token = threading.Event()

        def writer(binding: FiniteCandidateBinding) -> None:
            binding.truncate(0)
            binding.write(_payload(request))
            token.set()

        result = _publish(
            request,
            publisher=FiniteArtifactPublisher(request, cancel_token=token),
            writer=writer,
        )
        assert result.disposition is FiniteArtifactDisposition.ABORTED
        assert not Path(request.output_artifact).exists()
        assert result.hidden_orphan == str(paths[0])
    else:
        def inspect(_path, _request):
            raise ValueError("post-link inspection failed")

        with pytest.raises(FiniteArtifactPublicationHeld) as captured:
            _publish(request, inspect_committed=inspect)
        held = captured.value
        assert Path(request.output_artifact).exists()
        # Same reason as the committed branch: the rename consumed the
        # candidate before the post-publication inspection ran, so a held
        # failure can no longer leave one behind either.
        assert held.hidden_orphan is None
        assert paths == []
        result = held
    if result.hidden_orphan is not None:
        assert Path(result.hidden_orphan).exists()
    monkeypatch.setattr(finite_module, "_unlink_candidate", real_unlink)
    if result.hidden_orphan is not None:
        Path(result.hidden_orphan).unlink()


def test_writer_failure_remains_primary_when_private_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    candidate_paths: list[Path] = []

    def writer(_binding: FiniteCandidateBinding) -> None:
        raise LookupError("writer remains primary")

    def fail_cleanup(
        _parent_descriptor: int,
        path: Path,
        *_args,
        **_kwargs,
    ) -> None:
        candidate_paths.append(path)
        raise PermissionError("cleanup is secondary")

    monkeypatch.setattr(finite_module, "_unlink_candidate", fail_cleanup)
    with pytest.raises(LookupError, match="writer remains primary") as captured:
        _publish(request, writer=writer)
    assert any("CLEANUP_INCOMPLETE" in note for note in captured.value.__notes__)
    assert candidate_paths[0].exists()
    assert not Path(request.output_artifact).exists()
    candidate_paths[0].unlink()


def test_foreign_private_candidate_collision_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    monkeypatch.setattr(finite_module.secrets, "token_hex", lambda _n: "a" * 32)
    candidate = tmp_path / (
        f".xdart-finite-{request.version_identity}-{'a' * 32}.candidate"
    )
    candidate.write_bytes(b"foreign candidate")
    with pytest.raises(FiniteArtifactCollision):
        _publish(request)
    assert candidate.read_bytes() == b"foreign candidate"
    assert not Path(request.output_artifact).exists()


def test_types_are_public_lazy_exports() -> None:
    import xrd_tools.io as io_api

    assert io_api.FiniteArtifactPublisher is FiniteArtifactPublisher
    assert io_api.FiniteArtifactRequest is FiniteArtifactRequest
    assert io_api.FiniteArtifactDisposition is FiniteArtifactDisposition
    assert io_api.FiniteDocumentAdapter is FiniteDocumentAdapter
    assert io_api.FiniteSeededDocumentAdapter is FiniteSeededDocumentAdapter
    assert io_api.FiniteSeedBinding is FiniteSeedBinding
    assert io_api.FiniteOperationContext is FiniteOperationContext
    assert io_api.FiniteCandidateValidation is FiniteCandidateValidation
    assert io_api.FiniteCommittedInspection is FiniteCommittedInspection
    assert io_api.capture_finite_source is capture_finite_source
    assert io_api.finite_artifact_request is finite_artifact_request
    assert io_api.finite_operation_context is finite_operation_context
    assert io_api.capture_finite_predecessor is capture_finite_predecessor
    assert io_api.write_finite_artifact_lineage is write_finite_artifact_lineage
    assert io_api.require_finite_artifact_lineage is require_finite_artifact_lineage
    assert io_api.artifact_family_from_source is artifact_family_from_source
    assert io_api.resolve_finite_output_target is resolve_finite_output_target
    assert io_api.FINITE_LINEAGE_NODE_NAME == "finite_artifact"
    assert io_api.FINITE_LINEAGE_SCHEMA == "xdart.finite-artifact-lineage.v1"
    assert io_api.FINITE_PUBLICATION_POLICY == "IMMUTABLE_SUCCESSOR_V1"
    assert io_api.FINITE_PARENT_DIRECTORY_FSYNC_WARNING == (
        "FINITE_PARENT_DIRECTORY_FSYNC_UNCONFIRMED"
    )
    assert "FiniteArtifactPublisher" in io_api.__all__


def test_a_second_publication_into_one_slot_is_refused_while_the_first_holds_it(
    tmp_path: Path,
) -> None:
    """The finite path holds an H23 lease on the PUBLIC SLOT. It did not before.

    Until 2026-09-04 `grep -rn "coordinator.admit\\|\\.acquire_lease(" src/`
    returned NOTHING in `finite_artifact.py`, `reintegrate_successor.py` or
    `scan_session.py`.  The no-clobber `os.link` at the end of publication WAS
    the entire concurrency guard for Reintegrate, Stitch and RSM: two concurrent
    operations on one slot were settled by whichever linked first.

    That matters because atomic replacement REMOVES that link.  Landing the
    lease first, on the still-no-clobber publisher, means the guard exists
    before the thing it replaces goes away -- so this row is what makes
    unconditional replacement safe rather than a silent last-writer-wins.

    Shaped after P1-B's b17, which pins the same typed refusal for ordinary Run:
    hold the first writer INSIDE its own adapter, and require the contender to
    fail with `LeaseUnavailable` naming the artifact.
    """
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)
    target = Path(request.output_artifact)

    inside = threading.Event()
    release = threading.Event()
    contender_error: list[BaseException] = []

    def held_write(binding: FiniteCandidateBinding) -> None:
        _write_payload(request)(binding)
        inside.set()
        assert release.wait(30.0)

    def first() -> None:
        _publish(request, writer=held_write)

    writer_thread = threading.Thread(target=first)
    writer_thread.start()
    try:
        assert inside.wait(30.0), "first publication never reached its writer"
        # A whole second publisher, mid-candidate on the same slot.
        try:
            _publish(_request(tmp_path, source))
        except BaseException as error:
            contender_error.append(error)
    finally:
        release.set()
        writer_thread.join(30.0)
    assert not writer_thread.is_alive()

    assert len(contender_error) == 1, contender_error
    refusal = contender_error[0]
    assert type(refusal) is LeaseUnavailable
    assert str(refusal) == (
        f"target already leased: {os.path.normcase(os.path.abspath(target))}"
    )
    # The first publication still completed, and it owns the slot.
    assert target.read_bytes() == _payload(request)

    # And the hold is GIVEN BACK: the slot is publishable again afterwards.
    again = FiniteArtifactPublisher(_request(tmp_path, source))
    assert again.publish(
        FiniteDocumentAdapter(
            lambda binding: nullcontext(binding),
            _write_payload(request),
            lambda binding: nullcontext(binding),
            _validate_payload(request),
        ),
        inspect_committed=_inspect_payload,
    ).disposition in {
        FiniteArtifactDisposition.COMMITTED,
        FiniteArtifactDisposition.ALREADY_COMMITTED,
    }


def _directory_is_case_insensitive(root: Path) -> bool:
    """Ask THIS filesystem rather than assuming the platform's usual answer."""
    probe = root / "CaseProbe"
    probe.mkdir()
    try:
        other = root / "caseprobe"
        return other.exists() and os.path.samefile(probe, other)
    finally:
        probe.rmdir()


def test_one_file_under_two_spellings_cannot_be_leased_twice(
    tmp_path: Path,
) -> None:
    """Codex F1 on `da228738`: the lease keyed on a STRING, not on a file.

    `_normalize_target` is `normcase(abspath(...))`, and on POSIX `normcase` is
    the identity function.  So two spellings of ONE directory produced two keys
    and two simultaneous leases.  Survivable while a no-clobber `os.link` still
    refused the second writer; NOT survivable once replacement removes it, which
    is why the fix ships alongside.

    The registry now keys on the parent's `(st_dev, st_ino)` plus the name, so
    aliasing collapses by IDENTITY.  No blanket lowercasing -- that would
    conflate genuinely distinct files on a case-sensitive volume, which is why
    the case leg below asks the filesystem first instead of assuming.
    """
    from xrd_tools.io.output_transaction import (
        LeaseOwner,
        LeaseUnavailable,
        OwnerToken,
        get_output_transaction_coordinator,
    )

    coordinator = get_output_transaction_coordinator()
    real = tmp_path / "processed"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)

    # A SYMLINKED directory is one directory.
    hold = coordinator.hold_target(real / "slot.nexus", label="first")
    try:
        with pytest.raises(LeaseUnavailable):
            coordinator.hold_target(
                tmp_path / "link" / "slot.nexus", label="second",
            )
    finally:
        coordinator.release_target(hold)

    # A CASE-ALIASED directory is one directory -- where this volume says so.
    if _directory_is_case_insensitive(tmp_path):
        assert os.path.samefile(real, tmp_path / "PROCESSED")
        hold = coordinator.hold_target(real / "slot.nexus", label="third")
        try:
            with pytest.raises(LeaseUnavailable):
                coordinator.hold_target(
                    tmp_path / "PROCESSED" / "slot.nexus", label="fourth",
                )
        finally:
            coordinator.release_target(hold)

    # CROSS-MECHANISM: an ordinary Run transaction lease and a finite hold are
    # the same exclusion. This is the property the whole ruling rests on -- a
    # Run and a Reintegrate must not both believe they own one file.
    transaction_owner = OwnerToken("run-transaction")
    target_owner = OwnerToken("run-target")
    transaction = coordinator.admit(
        real / "shared.nexus",
        transaction_owner=transaction_owner,
        target_owner=target_owner,
    )
    # The SAME token objects must come back at release; a fresh token with an
    # equal name is refused.
    run_owners = {r: OwnerToken(f"run-{r.value}") for r in LeaseOwner}
    lease = transaction.acquire_lease(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        owners=run_owners,
    )
    with pytest.raises(LeaseUnavailable):
        coordinator.hold_target(
            tmp_path / "link" / "shared.nexus", label="finite",
        )
    transaction.abandon(lease)
    for role in LeaseOwner:
        transaction.release_lease_owner(lease, role, run_owners[role])

    # And a genuinely DIFFERENT file is still free: the fix excludes aliases,
    # not neighbours.
    free = coordinator.hold_target(real / "other.nexus", label="fifth")
    coordinator.release_target(free)


def test_a_failed_slot_release_is_reported_on_the_error_that_caused_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex F2 on `da228738`: the warning was appended where nobody reads it.

    `diagnostics` reaches a caller only through a RESULT object.  A
    pre-publication failure RE-RAISES instead of returning one, so a release
    failure on that path was recorded into a list that was then discarded.  It
    is now attached to the exception as a note, exactly as a descriptor-close
    failure already is.
    """
    source = tmp_path / "source.nexus"
    source.write_bytes(b"source")
    request = _request(tmp_path, source)

    captured: list = []

    def failing_release(hold) -> None:
        # CAPTURE it: this test deliberately breaks release, and without giving
        # the hold back afterwards it leaks into the PROCESS-WIDE coordinator
        # and poisons every later test that touches this slot (Codex F7).
        captured.append(hold)
        raise OSError("slot release denied")

    def failing_writer(_binding) -> None:
        raise ValueError("the real publication failure")

    from xrd_tools.io.output_transaction import (
        OutputTransactionCoordinator, get_output_transaction_coordinator,
    )

    monkeypatch.setattr(
        OutputTransactionCoordinator,
        "release_target",
        lambda self, hold: failing_release(hold),
    )
    with pytest.raises(ValueError, match="the real publication failure") as caught:
        _publish(request, writer=failing_writer)

    # The primary error is still the one the caller came for...
    notes = getattr(caught.value, "__notes__", [])
    # ...and the release failure is visible ON it rather than lost.
    assert any(finite_module.FINITE_SLOT_LEASE_WARNING in note for note in notes), notes

    # GIVE THE HOLD BACK. `monkeypatch` restores the real method at teardown but
    # cannot undo the hold this test stranded; the registry is process-wide.
    assert len(captured) == 1
    monkeypatch.undo()
    get_output_transaction_coordinator().release_target(captured[0])


def test_a_hold_survives_the_file_appearing_underneath_it(tmp_path: Path) -> None:
    """Codex F2 (P2) on `e06d6123`: the lease key moved after acquisition.

    `_lease_key` is computed from MUTABLE filesystem state -- it folds the name
    only when the directory is observed to treat two spellings as one file, and
    that observation is only possible once the file EXISTS.  So holding
    `Scan.nexus` before creation and again after produced two DIFFERENT keys for
    the IDENTICAL string, and both were admitted.

    Worse, the first fix kept a reverse map from pathname to key; the second
    acquisition overwrote it, so the first hold became unreleasable and its
    registry entry was stranded for the life of the process.

    Acquisition now excludes by identity key AND by pathname, and release finds
    its entry by object identity rather than recomputing anything.
    """
    from xrd_tools.io.output_transaction import (
        LeaseUnavailable, get_output_transaction_coordinator,
    )

    coordinator = get_output_transaction_coordinator()
    target = tmp_path / "Scan.nexus"

    hold = coordinator.hold_target(target, label="first")
    try:
        target.write_bytes(b"the file appears mid-hold")
        with pytest.raises(LeaseUnavailable):
            coordinator.hold_target(target, label="second")
    finally:
        # THE ORIGINAL HOLD MUST STILL RELEASE. This is what the reverse map broke.
        coordinator.release_target(hold)

    # Positive control: acquirable again afterwards, so nothing is stranded.
    again = coordinator.hold_target(target, label="third")
    coordinator.release_target(again)


def test_a_hold_survives_its_parent_directory_appearing(tmp_path: Path) -> None:
    """The same transition through the OTHER key path.

    With no parent to stat, `_lease_key` falls back to the raw string; once the
    directory exists it returns a parent-identity key. Codex named this the same
    defect reached a second way.
    """
    from xrd_tools.io.output_transaction import (
        LeaseUnavailable, get_output_transaction_coordinator,
    )

    coordinator = get_output_transaction_coordinator()
    target = tmp_path / "not-yet" / "slot.nexus"

    hold = coordinator.hold_target(target, label="first")
    try:
        target.parent.mkdir()
        with pytest.raises(LeaseUnavailable):
            coordinator.hold_target(target, label="second")
    finally:
        coordinator.release_target(hold)
    again = coordinator.hold_target(target, label="third")
    coordinator.release_target(again)


def test_a_run_and_an_average_cannot_both_hold_a_slot_in_a_fresh_directory(
    tmp_path: Path,
) -> None:
    """Fable F2 (P1) on `58123dc7`: the two callers that matter acquire early.

    My disclosure claimed this gap was narrow because "the publisher opens its
    parent before acquiring". True of the FINITE publisher, false of ordinary
    Run and Average: both acquire BEFORE the destination directory exists --
    `_copy_stream_seed` creates it -- so the first got a raw-string key and the
    second, after the mkdir, a parent-identity key. Different keys, both
    granted, and the first hold then unreleasable.

    The same-pathname exclusion is what actually covers them, and this row
    drives the real sequence rather than the tidy one.
    """
    from xrd_tools.io.output_transaction import (
        LeaseOwner, LeaseUnavailable, OwnerToken,
        get_output_transaction_coordinator,
    )

    coordinator = get_output_transaction_coordinator()
    slot = tmp_path / "fresh" / "scan_average.nexus"   # 'fresh/' does NOT exist

    hold = coordinator.hold_target(slot, label="average-slot")
    try:
        slot.parent.mkdir()          # what the streamed seed does mid-operation
        transaction_owner = OwnerToken("run-transaction")
        target_owner = OwnerToken("run-target")
        transaction = coordinator.admit(
            slot,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
        )
        with pytest.raises(LeaseUnavailable):
            transaction.acquire_lease(
                admission=transaction.admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
                owners={r: OwnerToken(f"run-{r.value}") for r in LeaseOwner},
            )
    finally:
        coordinator.release_target(hold)

    # Nothing stranded: the slot is acquirable again afterwards.
    again = coordinator.hold_target(slot, label="after")
    coordinator.release_target(again)


def test_pre_rename_directory_collision_never_claims_publication(tmp_path):
    import h5py
    from xrd_tools.io.finite_artifact import _FinitePublicationSession

    target = tmp_path / "output.nexus"
    expected = capture_target_snapshot(target)
    owner = _FinitePublicationSession(target)
    document = owner.start(lambda path: h5py.File(path, "w"))
    document.create_dataset("value", data=[1, 2, 3])
    candidate = owner.candidate
    ordinal = owner.ordinal
    assert owner.close_document()
    target.mkdir()

    def observe(path, ordinal):
        raise AssertionError("no candidate was renamed; observation must not run")

    facts = None
    try:
        facts = owner.publish(expected, ordinal=ordinal, observe=observe)
    except OSError:
        pass
    finally:
        owner.abort()
    print("PRE_RENAME_COLLISION", "published", owner.published,
          "candidate_exists", candidate.exists(), "hidden_orphan", owner.hidden_orphan,
          "facts", facts, flush=True)
    assert target.is_dir()
    assert not owner.published
    assert not candidate.exists()


# ---------------------------------------------------------------------------
# The Windows stat shape (PR #1 2026-09-11): pathname ctime = creation time,
# descriptor ctime = change time.  ``_capture_regular`` compares four views of
# one file (lstat, fstat, fstat, lstat); only the seam lets them agree there.
# ---------------------------------------------------------------------------


def test_win32_pathname_ctime_shape_is_refused_while_ctime_is_identity(
    tmp_path: Path, monkeypatch, ctime_seam, win32_pathname_ctime,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"finite")
    monkeypatch.setattr(ctime_seam, "IDENTITY_CARRIES_CTIME", True)
    win32_pathname_ctime(source)
    with pytest.raises(FiniteArtifactIntegrityError, match="changed during observation"):
        capture_finite_source(source)


def test_win32_identity_admits_the_pathname_ctime_shape_and_records_the_descriptor_view(
    tmp_path: Path, monkeypatch, ctime_seam, win32_pathname_ctime,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"finite")
    with source.open("rb") as handle:
        handle_ctime_ns = os.fstat(handle.fileno()).st_ctime_ns
    monkeypatch.setattr(ctime_seam, "IDENTITY_CARRIES_CTIME", False)
    gap_ns = win32_pathname_ctime(source)
    # The two views really disagree by the runner's gap; only the compare is neutral.
    assert os.stat(source).st_ctime_ns == handle_ctime_ns - gap_ns
    admission = capture_finite_source(source)
    snapshot = admission.snapshot
    assert snapshot.ctime_ns == handle_ctime_ns
    assert (snapshot.size, snapshot.digest) == (6, hashlib.sha256(b"finite").hexdigest())
    # A descriptor re-observation of the same object still agrees with it,
    # ctime included: the recorded identity is the descriptor's view, exact.
    assert finite_module._observe_regular(source) == finite_module._snapshot_state(snapshot)
    assert finite_module._snapshot_state(snapshot)[5] == handle_ctime_ns
    # Only a compare against a pathname view goes through the seam.
    assert finite_module._comparable(finite_module._snapshot_state(snapshot))[5] == 0


def test_win32_identity_refuses_a_same_size_same_mtime_rewrite_after_admission(
    tmp_path: Path, monkeypatch, ctime_seam, win32_pathname_ctime,
) -> None:
    """``_require_source`` carries ``snapshot.digest`` forward on a stat
    compare alone, so it must see the descriptor's change time move even
    where the pathname compare is neutral (Codex PR #1 review, F1)."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"finite")
    monkeypatch.setattr(ctime_seam, "IDENTITY_CARRIES_CTIME", False)
    win32_pathname_ctime(source)
    admission = capture_finite_source(source)
    assert finite_module._require_source(admission) is admission.snapshot
    before = os.stat(source)
    source.write_bytes(b"FINITE")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    with source.open("rb") as handle:
        after = os.fstat(handle.fileno())
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    # Precondition of the row: only the descriptor's change time moved.
    assert after.st_ctime_ns != admission.snapshot.ctime_ns
    assert finite_module._observe_regular(source) != finite_module._snapshot_state(
        admission.snapshot
    )
    with pytest.raises(FiniteArtifactIntegrityError, match="changed after admission"):
        finite_module._require_source(admission)


@pytest.mark.parametrize("shape", ("posix", "win32"))
def test_capture_refuses_a_same_size_same_mtime_rewrite_inside_the_read_window(
    tmp_path: Path, monkeypatch, ctime_seam, win32_pathname_ctime, shape: str,
) -> None:
    """``_capture_regular`` brackets its read with two descriptor views held
    to each other exactly, ctime included: a rewrite that keeps size and mtime
    inside that window is refused rather than digested torn, also where the
    pathname compare is neutral (Codex PR #1 addendum, Fix 1 acceptance)."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"finite")
    monkeypatch.setattr(ctime_seam, "IDENTITY_CARRIES_CTIME", shape == "posix")
    if shape == "win32":
        win32_pathname_ctime(source)
    log: list[tuple[int, int]] = []

    class _RewritingOs:
        """The module's ``os`` with a ``read`` that rewrites the file in place
        (same size, same mtime) before the first block is read."""

        def __getattr__(self, name):
            return getattr(os, name)

        def read(self, descriptor, size):
            if not log:
                opened = os.fstat(descriptor)
                before = os.stat(source)
                source.write_bytes(b"FINITE")
                os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
                rewritten = os.fstat(descriptor)
                assert (rewritten.st_size, rewritten.st_mtime_ns) == (
                    opened.st_size, opened.st_mtime_ns,
                )
                log.append((opened.st_ctime_ns, rewritten.st_ctime_ns))
            return os.read(descriptor, size)

    monkeypatch.setattr(finite_module, "os", _RewritingOs())
    with pytest.raises(FiniteArtifactIntegrityError, match="changed during observation"):
        capture_finite_source(source)
    # Precondition of the row: the rewrite happened inside the window and
    # only the descriptor's change time moved.
    [(opened_ctime_ns, rewritten_ctime_ns)] = log
    assert rewritten_ctime_ns != opened_ctime_ns
    # The now-quiet file still captures under either shape.
    monkeypatch.setattr(finite_module, "os", os)
    assert capture_finite_source(source).snapshot.digest == hashlib.sha256(b"FINITE").hexdigest()


def test_win32_identity_still_refuses_a_pathname_mtime_disagreement(
    tmp_path: Path, monkeypatch, ctime_seam,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"finite")
    monkeypatch.setattr(ctime_seam, "IDENTITY_CARRIES_CTIME", False)
    real_lstat = os.lstat

    class _Shifted:
        def __init__(self, real):
            self._real = real
            self.st_mtime_ns = real.st_mtime_ns - 1

        def __getattr__(self, name):
            return getattr(self._real, name)

    def shifted_lstat(path, *args, **kwargs):
        result = real_lstat(path, *args, **kwargs)
        if isinstance(path, (str, os.PathLike)) and os.path.abspath(path) == str(source):
            return _Shifted(result)
        return result

    monkeypatch.setattr(os, "lstat", shifted_lstat)
    with pytest.raises(FiniteArtifactIntegrityError, match="changed during observation"):
        capture_finite_source(source)
