"""Reintegration through a validated candidate and stable operation slot.

This additive v4 API never opens its source artifact for writing.  It qualifies
the existing scientific recipe, builds one publisher-owned private HDF5 copy,
and publishes the operation-specific ``.nexus`` slot only after bounded
reduction and semantic validation complete. The GUI uses this path. The older
in-place module still supplies shared scientific/source helpers; it cannot be
removed wholesale until those dependencies have been extracted.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping

import h5py
import numpy as np

from xrd_tools.io.bounded_json import (
    BoundedJsonError,
    bounded_json_snapshot,
    bounded_utf8_size,
)
from xrd_tools.io.finite_artifact import (
    FiniteArtifactDisposition,
    FiniteArtifactIntegrityError,
    FiniteArtifactPublicationHeld,
    FiniteArtifactPublisher,
    FiniteArtifactRequest,
    FiniteArtifactResult,
    FiniteCandidateValidation,
    FiniteCandidateWriteDisposition,
    FiniteCommittedInspection,
    FiniteSeededDocumentAdapter,
    FiniteFileSnapshot,
    FiniteSourceAdmission,
    _replay_finite_artifact_request,
    _operation_context_payload_values,
    _operation_payload,
    _publication_payload,
    _version_payload,
    admit_finite_artifact_lineage,
    capture_finite_predecessor,
    capture_finite_source,
    finite_lineage_hdf_path,
    finite_artifact_request,
    finite_operation_context,
    require_finite_artifact_lineage,
)
from xrd_tools.io.output_transaction import (
    StreamTerminal,
    TargetSnapshot,
    revalidate_stream_terminal,
)
from xrd_tools.io.output_path import artifact_family_from_source
from xrd_tools.io.processed_scan_id import require_current_processed
from xrd_tools.io.record_writer import bind_prepared_manifest_receipt
from xrd_tools.io.schema import PROCESSED_SCHEMA_VERSION
from xrd_tools.reduction import reintegrate as _support
from xrd_tools.reduction.reintegrate_prepared import (
    LegacyRouteReason,
    PreparedCapsuleMiss,
    PreparedCapsuleMissCode,
    PreparedReintegrateBundle,
    PreparedReintegrateExecution,
    PreparedReintegrateOffer,
    PreparedRouteChanged,
    PreparedRouteRejected,
    admit_prepared_bundle,
    admit_prepared_execution,
    prepared_bundle_mapping,
    prepared_execution_mapping,
    preflight_prepared_execution,
    select_prepared_execution,
    source_topology_identity,
)


_PLAN_API_VERSION = 4
_RECIPE_VERSION = 4
_RECIPE_SCHEMA = "xrd_tools.reintegrate.plan"
_OUTPUT_SCHEMA = f"xdart-current-v4-processed-v{PROCESSED_SCHEMA_VERSION}"
_PUBLICATION_POLICY = "IMMUTABLE_SUCCESSOR_V1"
_ROUTES = {"bounded-legacy", "prepared"}
_STAGES = {"qualify", "read", "reduce", "write", "validate", "publish"}
_SHA = frozenset("0123456789abcdef")

_MAX_RECIPE_BYTES = 64 * 1024 * 1024
_MAX_RECIPE_KEY_BYTES = 8 * 1024 * 1024
_MAX_RECIPE_PATH_BYTES = 4096
_MAX_UNTRUSTED_STRING_BYTES = 8 * 1024 * 1024
_MAX_RECIPE_DEPTH = 24
_MAX_RECIPE_NODES = 262_144
_MAX_CONTAINER_CHILDREN = 16_384


class ReintegrateRecipeMigrationRequired(ValueError):
    """A recognized mutable/rollback recipe needs explicit v4 migration."""

    code = "IMMUTABLE_SUCCESSOR_V4_REQUIRED"

    def __init__(self, version: int) -> None:
        if type(version) is not int or version >= _RECIPE_VERSION:
            raise TypeError("migration refusal requires a recognized old version")
        self.version = version
        super().__init__(
            f"{self.code}: reintegration recipe v{version} must be migrated "
            "to immutable v4; rollback fields are not reinterpreted"
        )


def _sha(value: Any) -> str:
    raw = json.dumps(
        _support._plain(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA


def _exact_keys(value: object, expected: set[str], role: str) -> dict:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{role} has a noncanonical keyset")
    return value


def _bounded_recipe_snapshot(value: object) -> tuple[object, int]:
    """Detach and bound an untrusted JSON tree before schema work or file I/O."""
    return bounded_json_snapshot(
        value,
        role="recipe",
        max_encoded_bytes=_MAX_RECIPE_BYTES,
        max_key_bytes=_MAX_RECIPE_KEY_BYTES,
        max_string_bytes=_MAX_UNTRUSTED_STRING_BYTES,
        max_depth=_MAX_RECIPE_DEPTH,
        max_nodes=_MAX_RECIPE_NODES,
        max_children=_MAX_CONTAINER_CHILDREN,
    )


def _bounded_preparation_snapshot(
    value: object,
    *,
    role: str,
) -> tuple[object, int]:
    """Detach click science before either prepared or legacy schema work."""

    return bounded_json_snapshot(
        value,
        role=role,
        max_encoded_bytes=_MAX_RECIPE_BYTES,
        max_key_bytes=_MAX_RECIPE_KEY_BYTES,
        max_string_bytes=_MAX_UNTRUSTED_STRING_BYTES,
        max_depth=_MAX_RECIPE_DEPTH,
        max_nodes=_MAX_RECIPE_NODES,
        max_children=_MAX_CONTAINER_CHILDREN,
    )


def _bounded_recipe_text(value: object, role: str) -> str:
    bounded_utf8_size(
        value,
        role=f"recipe {role}",
        max_bytes=_MAX_RECIPE_PATH_BYTES,
    )
    return value


def _snapshot_mapping(value: FiniteFileSnapshot) -> dict[str, object]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _terminal_mapping(value: StreamTerminal | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _target_snapshot(value: FiniteFileSnapshot) -> TargetSnapshot:
    return TargetSnapshot(
        True, value.size, value.mtime_ns, value.device, value.inode, value.digest,
        ctime_ns=value.ctime_ns,
    )


def _named_snapshot_changed(value: FiniteFileSnapshot) -> bool:
    try:
        observed = os.stat(value.path, follow_symlinks=False)
    except OSError:
        return True
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_mode),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    ) != (
        value.device,
        value.inode,
        value.mode,
        value.size,
        value.mtime_ns,
        value.ctime_ns,
    )


def _exact_terminal_for_snapshot(
    snapshot: FiniteFileSnapshot,
    terminal: StreamTerminal | None,
) -> StreamTerminal | None:
    """Return only a terminal whose digest names all admitted source bytes.

    ``StreamTerminal`` also represents fast-regenerable durability evidence,
    whose digest intentionally covers a bounded proof rather than the whole
    file.  That receipt is valid for transaction settlement but cannot serve
    as immutable finite-artifact lineage.
    """

    if terminal is None:
        return None
    if (
        terminal.target == snapshot.path
        and terminal.size == snapshot.size
        and terminal.digest == snapshot.digest
        and (
            terminal.device,
            terminal.inode,
            terminal.mtime_ns,
            terminal.ctime_ns,
        )
        == (
            snapshot.device,
            snapshot.inode,
            snapshot.mtime_ns,
            snapshot.ctime_ns,
        )
    ):
        return terminal
    return None


def _exact_predecessor_terminal(
    admission: FiniteSourceAdmission,
    terminal: StreamTerminal | None,
) -> StreamTerminal | None:
    if type(admission) is not FiniteSourceAdmission:
        raise TypeError("finite predecessor terminal requires source admission")
    return _exact_terminal_for_snapshot(admission.snapshot, terminal)


def _predecessor(
    admission: FiniteSourceAdmission,
    terminal: StreamTerminal | None,
    entry: str,
    explicit_family: str | None = None,
):
    try:
        with h5py.File(admission.path, "r") as document:
            _require_source_document_snapshot(document, admission.snapshot)
            path = finite_lineage_hdf_path(entry)
            link = document.get(path, getlink=True)
            lineage = (
                None
                if link is None
                else require_finite_artifact_lineage(document, entry=entry)
            )
            _require_source_document_snapshot(document, admission.snapshot)
    except OSError as error:
        raise FiniteArtifactIntegrityError(
            "finite predecessor lineage is unavailable"
        ) from error
    if lineage is None:
        # No finite lineage node: this is a RUN or AVERAGE output (they write
        # through NexusSink, not the finite publisher) or a foreign/legacy file.
        # Consume the root family those writers stamp on the entry rather than
        # falling through to a stem-derived family, which is what produced
        # `<scan>_int2d_reintegrate1d.nexus`.
        return capture_finite_predecessor(
            admission,
            terminal=terminal,
            artifact_family_v1=_persisted_root_family(
                admission.path, entry, explicit_family,
            ),
        )
    payload = json.loads(lineage.canonical_json)
    return capture_finite_predecessor(
        admission,
        terminal=terminal,
        artifact_family_v1=payload["artifact_family_v1"],
        version_identity=payload["version_identity"],
        publication_identity=payload["publication_identity"],
        lineage_identity=lineage.lineage_identity,
    )


def _persisted_root_family(
    path, entry: str, explicit_family: str | None = None,
) -> str | None:
    """Read `entry/@artifact_family_v1`, falling back to *explicit_family*.

    A stable public name is `<family><slot>.nexus`.  The stamped family is
    authoritative when present, and NOTHING here ever parses a slot out of a
    stem -- stripping a generated suffix to recover a root is an explicit stop
    condition.

    Fable F4 on `4fe073e8`: this docstring used to describe a refusal for a
    slot-shaped stem without the attribute, which the body no longer performs.
    The reason it was removed is in the body comment below; the short version is
    that it refused `series_rsm.nexus`, a scan legitimately named `series_rsm`.
    """
    from xrd_tools.io.schema import ARTIFACT_FAMILY_ATTR

    family = None
    try:
        with h5py.File(path, "r") as document:
            group = document.get(entry)
            if isinstance(group, h5py.Group):
                stored = group.attrs.get(ARTIFACT_FAMILY_ATTR)
                if stored is not None:
                    family = (
                        stored.decode("utf-8", errors="replace")
                        if isinstance(stored, bytes) else str(stored)
                    )
    except OSError as error:
        raise FiniteArtifactIntegrityError(
            "finite predecessor root family is unavailable"
        ) from error
    if family is not None:
        return family
    # NO SUFFIX-TAIL REFUSAL.  An earlier revision refused a stem ending in a
    # known slot, on the theory that its root could not be recovered.  That was
    # wrong twice over: it refused `series_rsm.nexus`, a scan legitimately named
    # `series_rsm`, and it was unnecessary -- every artifact the slot-aware
    # writers produce carries the attribute, so a slot-shaped stem WITHOUT one
    # was never written under this policy and its stem IS its root family.
    # Retaining it verbatim is therefore correct rather than a chain, and it
    # keeps the "never interpret a generated suffix" rule intact by not looking
    # at the suffix at all.
    return explicit_family


def _derived_request_inputs(
    *,
    snapshot: FiniteFileSnapshot,
    terminal: StreamTerminal | None,
    qualified,
    artifact_family: str,
    route: str,
    source_graph_evidence: Mapping[str, object],
    prepared: PreparedReintegrateExecution | None,
    legacy_reason: LegacyRouteReason | None,
    miss_code: PreparedCapsuleMissCode | None,
) -> dict[str, str]:
    evidence = _support._plain(source_graph_evidence)
    if (
        type(evidence) is not dict
        or set(evidence) != {
            "facts_digest",
            "topology_identity",
            "source_execution_identity",
            "append_lineage_identity",
        }
        or not _is_sha(evidence["topology_identity"])
        or not _is_sha(evidence["source_execution_identity"])
        or evidence["append_lineage_identity"] is not None
        and not _is_sha(evidence["append_lineage_identity"])
    ):
        raise ValueError("source graph evidence is noncanonical")
    if route == "prepared":
        if (
            prepared is None
            or legacy_reason is not None
            or miss_code is not None
            or not _is_sha(evidence["facts_digest"])
        ):
            raise ValueError("prepared route seal is incomplete")
        execution_digest = prepared.execution_digest
    elif route == "bounded-legacy":
        if (
            prepared is not None
            or evidence["facts_digest"] is not None
            or type(legacy_reason) is not LegacyRouteReason
            or legacy_reason is LegacyRouteReason.DIRECT_FROM_ARTIFACT
            and miss_code is not None
            or legacy_reason is LegacyRouteReason.CAPSULE_MISS
            and type(miss_code) is not PreparedCapsuleMissCode
        ):
            raise ValueError("bounded legacy route seal is incomplete")
        execution_digest = None
    else:
        raise ValueError("immutable reintegration route is unsupported")
    request_generation = _sha({
        "artifact_family": artifact_family,
        "dimension": qualified.dimension,
        "entry": qualified.entry,
        "labels": list(qualified.labels),
        "selected_plan": _support._plain(qualified.selected_plan),
        "shared_science": _support._plain(qualified.requested_shared_science),
    })
    allocation = _sha(_support._allocation_recipe(qualified.resource_allocation))
    route_identity = _sha({
        "execution_digest": execution_digest,
        "legacy_reason": (
            None if legacy_reason is None else legacy_reason.value
        ),
        "miss_code": None if miss_code is None else miss_code.value,
        "route": route,
        "schema": "reintegrate-route-v2",
    })
    custody = _sha({
        "source": _snapshot_mapping(snapshot),
        "terminal": _terminal_mapping(terminal),
    })
    source_graph = _sha({
        "entry": qualified.entry,
        "output_schema": _OUTPUT_SCHEMA,
        "source_digest": snapshot.digest,
        "source_size": snapshot.size,
        "source_root": qualified.source_root,
        "gi_bootstrap_incidence": qualified.gi_bootstrap_incidence,
        "labels": list(qualified.labels),
        "source_topology_identity": evidence["topology_identity"],
        "source_execution_identity": evidence["source_execution_identity"],
        "append_lineage_identity": evidence["append_lineage_identity"],
        "schema": "reintegrate-source-graph-v1",
    })
    algorithm = _sha({
        "dimension": qualified.dimension,
        "engine": "bounded-jit-direct-hdf-v1",
        "schema": "reintegrate-algorithm-v1",
        "selected_plan": _support._plain(qualified.selected_plan),
    })
    preservation = _sha({
        "predecessor_digest": snapshot.digest,
        "schema": "reintegrate-preservation-v1",
        "selected_change_set": [
            f"/{qualified.entry}/integrated_{qualified.dimension}",
            f"/{qualified.entry}/reduction/config/"
            f"bai_{qualified.dimension}_args",
            f"/{qualified.entry}/reduction/config/"
            f"dimension_replacement_{qualified.dimension}",
            f"/{qualified.entry}/reduction/config/"
            f"dimension_replacement_{qualified.dimension}_result_seal",
            f"/{qualified.entry}/reduction/config/gi_config",
            f"/{qualified.entry}/reduction/config/source_execution",
            f"/{qualified.entry}/reduction/config/append_lineage",
            "/@file_name",
            f"/{qualified.entry}/reduction/config/finite_artifact",
        ],
    })
    return {
        "request_generation_identity": request_generation,
        "resource_allocation_identity": allocation,
        "route_identity": route_identity,
        "custody_identity": custody,
        "source_graph_identity": source_graph,
        "algorithm_identity": algorithm,
        "preservation_identity": preservation,
    }


def _request_from_values(
    *,
    admission: FiniteSourceAdmission,
    terminal: StreamTerminal | None,
    qualified,
    destination_directory: Path,
    explicit_target: Path | None,
    artifact_family: str | None,
    route: str,
    source_graph_evidence: Mapping[str, object],
    prepared: PreparedReintegrateExecution | None,
    legacy_reason: LegacyRouteReason | None,
    miss_code: PreparedCapsuleMissCode | None,
) -> FiniteArtifactRequest:
    predecessor = _predecessor(
        admission,
        _exact_predecessor_terminal(admission, terminal),
        qualified.entry,
        artifact_family,
    )
    persisted_family = predecessor.artifact_family_v1
    if (
        persisted_family is not None
        and artifact_family is not None
        and artifact_family != persisted_family
    ):
        raise ValueError("explicit artifact family conflicts with predecessor lineage")
    family = artifact_family_from_source(
        admission.path,
        persisted_family if persisted_family is not None else artifact_family,
    )
    identities = _derived_request_inputs(
        snapshot=admission.snapshot,
        terminal=terminal,
        qualified=qualified,
        artifact_family=family,
        route=route,
        source_graph_evidence=source_graph_evidence,
        prepared=prepared,
        legacy_reason=legacy_reason,
        miss_code=miss_code,
    )
    context = finite_operation_context(
        admission,
        request_generation_identity=identities["request_generation_identity"],
        resource_allocation_identity=(
            identities["resource_allocation_identity"]
        ),
        route_identity=identities["route_identity"],
        custody_identity=identities["custody_identity"],
    )
    return finite_artifact_request(
        source_admission=admission,
        operation_context=context,
        predecessor=predecessor,
        destination_directory=destination_directory,
        explicit_target=explicit_target,
        artifact_family=family,
        operation_kind=f"reintegrate-{qualified.dimension}",
        source_graph_identity=identities["source_graph_identity"],
        entry=qualified.entry,
        scientific_identity=qualified.science_identity,
        output_schema=_OUTPUT_SCHEMA,
        algorithm_identity=identities["algorithm_identity"],
        preservation_identity=identities["preservation_identity"],
    )


def _legacy_source_graph_evidence(qualified) -> Mapping[str, object]:
    source = Path(qualified.target)
    snapshot = qualified.expected_target_snapshot
    revision = _support._target_object_revision(source, snapshot)
    inspected = _support._inspect_artifact(
        source,
        qualified.entry,
        qualified.dimension,
        snapshot,
        revision,
        qualified.source_root,
        read_mask=False,
    )
    if (
        inspected.labels != qualified.labels
        or inspected.detector_shape != qualified.detector_shape
        or inspected.native_dtype != qualified.native_dtype
        or _support._plain(inspected.persisted_shared_science)
        != _support._plain(qualified.requested_shared_science)
    ):
        raise ValueError("RECIPE_ARTIFACT_FACTS_CHANGED")
    return MappingProxyType({
        "facts_digest": None,
        "topology_identity": source_topology_identity(inspected.topology),
        "source_execution_identity": inspected.topology.execution_digest,
        "append_lineage_identity": inspected.topology.lineage_digest,
    })


def _qualified_from_prepared(
    execution: PreparedReintegrateExecution,
    *,
    dimension: Literal["1d", "2d"],
    preparation: Mapping[str, object],
    source_root: str | os.PathLike[str] | None,
    expected_labels: tuple[int, ...] | None,
    cancel_token: threading.Event | None,
):
    _support._event(cancel_token)
    try:
        preparation, _preparation_bytes = _bounded_preparation_snapshot(
            preparation,
            role="prepared click science",
        )
        _support._keys(
            preparation,
            {
                "api_version", "selected_plan", "requested_shared_science",
                "resource_policy",
            },
            "preparation",
        )
    except BoundedJsonError as error:
        code = {
            "bytes": PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT,
            "cardinality": PreparedCapsuleMissCode.CAPSULE_CARDINALITY_LIMIT,
        }.get(
            error.reason,
            PreparedCapsuleMissCode.REQUESTED_SCIENCE_UNSUPPORTED,
        )
        raise PreparedCapsuleMiss(code) from error
    except (TypeError, ValueError) as error:
        raise PreparedCapsuleMiss(
            PreparedCapsuleMissCode.REQUESTED_SCIENCE_UNSUPPORTED
        ) from error
    if (
        type(preparation["api_version"]) is not int
        or preparation["api_version"] != 1
    ):
        raise PreparedCapsuleMiss(
            PreparedCapsuleMissCode.REQUESTED_SCIENCE_UNSUPPORTED
        )
    if expected_labels is not None and expected_labels != execution.labels:
        raise PreparedCapsuleMiss(PreparedCapsuleMissCode.LABELS_CHANGED)
    selected = preparation["selected_plan"]
    shared = preparation["requested_shared_science"]
    persisted = (
        type(shared) is dict
        and shared == {"version": 1, "kind": "persisted_target"}
    )
    try:
        if persisted:
            _support._validate_persisted_selected(selected, dimension)
            shared = _support._plain(
                execution.artifact.persisted_shared_science
            )
            selected = _support._resolve_persisted_selected(
                selected, shared, dimension, None,
            )
        else:
            _support._validate_science(selected, shared, dimension)
            if (
                _support._plain(execution.artifact.persisted_shared_science)
                != shared
            ):
                raise PreparedCapsuleMiss(
                    PreparedCapsuleMissCode.REQUESTED_SCIENCE_UNSUPPORTED
                )
        metadata_keys, include_geometry = _support._fact_projection(
            selected, shared,
        )
        if (
            metadata_keys
            or include_geometry
            or shared["background"]["mode"] != "None"
            or shared["geometry"] is not None
            or shared["gi"]["enabled"]
        ):
            raise PreparedCapsuleMiss(
                PreparedCapsuleMissCode.REQUEST_REQUIRES_UNPREPARED_FACTS
            )
        requirements = _support._requirements(
            execution.artifact.detector_shape,
            execution.artifact.native_dtype,
            selected,
            shared,
        )
        policy = _support._policy(
            requirements,
            preparation["resource_policy"],
            _support._PersistedMaskSpec(0, 0),
        )
    except PreparedCapsuleMiss:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise PreparedCapsuleMiss(
            PreparedCapsuleMissCode.REQUESTED_SCIENCE_UNSUPPORTED
        ) from error
    root = execution.artifact.source_base
    if source_root is not None:
        offered_root = os.path.normcase(os.path.abspath(os.fspath(source_root)))
        if offered_root != root:
            raise PreparedCapsuleMiss(PreparedCapsuleMissCode.SOURCE_ROOT_CHANGED)
    try:
        qualified = _support._make_plan(
            execution.target.snapshot.path,
            execution.artifact.entry,
            root,
            dimension,
            execution.labels,
            execution.artifact.detector_shape,
            execution.artifact.native_dtype,
            selected,
            shared,
            None,
            0,
            0,
            policy,
            snapshot=_target_snapshot(execution.target.snapshot),
        )
    except ValueError as error:
        raise PreparedCapsuleMiss(
            PreparedCapsuleMissCode.REQUESTED_SCIENCE_UNSUPPORTED
        ) from error
    _support._event(cancel_token)
    return qualified


def _value(cls, *values):
    instance = object.__new__(cls)
    for field, value in zip(fields(cls), values):
        object.__setattr__(instance, field.name, value)
    return instance


@dataclass(frozen=True, slots=True, init=False)
class ReintegrateSuccessorPlan:
    api_version: int
    source_artifact: str
    output_artifact: str
    entry: str
    source_root: str
    dimension: Literal["1d", "2d"]
    labels: tuple[int, ...]
    detector_shape: tuple[int, int]
    native_dtype: str
    selected_plan: Mapping[str, Any]
    requested_shared_science: Mapping[str, Any]
    gi_bootstrap_incidence: float | None
    retained_mask_bytes: int
    mask_decode_bytes: int
    session_policy: Any
    source_snapshot: FiniteFileSnapshot
    expected_terminal: StreamTerminal | None
    route: Literal["bounded-legacy", "prepared"]
    artifact_family: str
    source_graph_evidence: Mapping[str, Any]
    source_graph_identity: str
    science_identity: str
    output_schema: str
    algorithm_identity: str
    preservation_identity: str
    request_generation_identity: str
    resource_allocation_identity: str
    route_identity: str
    custody_identity: str
    operation_context_identity: str
    version_identity: str
    publication_identity: str
    operation_identity: str
    legacy_reason: LegacyRouteReason | None
    miss_code: PreparedCapsuleMissCode | None
    lineage_json: str
    lineage_identity: str
    _request: FiniteArtifactRequest | None
    _qualified: Any
    _prepared: PreparedReintegrateExecution | None

    def __new__(cls, *args, **kwargs):
        raise TypeError("ReintegrateSuccessorPlan is factory-constructed")

    @property
    def resource_allocation(self):
        return self.session_policy.allocation

    @property
    def target(self) -> str:
        """Read-only compatibility projection for the shared frame source."""

        return self.source_artifact

    @property
    def expected_target_snapshot(self) -> TargetSnapshot:
        """Read-only compatibility projection for direct-HDF admission."""

        return _target_snapshot(self.source_snapshot)

    @classmethod
    def from_artifact(
        cls,
        source_artifact: str | os.PathLike[str],
        *,
        entry: str,
        dimension: Literal["1d", "2d"],
        preparation: Mapping[str, object],
        source_root: str | os.PathLike[str] | None = None,
        expected_target_snapshot: TargetSnapshot | None = None,
        expected_terminal_identity: StreamTerminal | None = None,
        expected_labels: tuple[int, ...] | None = None,
        destination_directory: str | os.PathLike[str] | None = None,
        explicit_output: str | os.PathLike[str] | None = None,
        artifact_family: str | None = None,
        cancel_token: threading.Event | None = None,
    ) -> "ReintegrateSuccessorPlan":
        return _legacy_successor_plan(
            source_artifact,
            entry=entry,
            dimension=dimension,
            preparation=preparation,
            source_root=source_root,
            expected_target_snapshot=expected_target_snapshot,
            expected_terminal_identity=expected_terminal_identity,
            expected_labels=expected_labels,
            destination_directory=destination_directory,
            explicit_output=explicit_output,
            artifact_family=artifact_family,
            cancel_token=cancel_token,
            legacy_reason=LegacyRouteReason.DIRECT_FROM_ARTIFACT,
            miss_code=None,
        )

    @classmethod
    def from_prepared_or_artifact(
        cls,
        offered: PreparedReintegrateBundle | PreparedReintegrateOffer | None,
        source_artifact: str | os.PathLike[str],
        *,
        entry: str,
        dimension: Literal["1d", "2d"],
        preparation: Mapping[str, object],
        source_root: str | os.PathLike[str] | None = None,
        expected_target_snapshot: TargetSnapshot | None = None,
        expected_terminal_identity: StreamTerminal | None = None,
        expected_labels: tuple[int, ...] | None = None,
        destination_directory: str | os.PathLike[str] | None = None,
        explicit_output: str | os.PathLike[str] | None = None,
        artifact_family: str | None = None,
        cancel_token: threading.Event | None = None,
    ) -> "ReintegrateSuccessorPlan":
        """Select one prepared route or one sealed pre-effect fallback."""

        if dimension not in {"1d", "2d"}:
            raise ValueError("reintegration dimension is unsupported")

        def fallback(code):
            return _legacy_successor_plan(
                source_artifact,
                entry=entry,
                dimension=dimension,
                preparation=preparation,
                source_root=source_root,
                expected_target_snapshot=expected_target_snapshot,
                expected_terminal_identity=expected_terminal_identity,
                expected_labels=expected_labels,
                destination_directory=destination_directory,
                explicit_output=explicit_output,
                artifact_family=artifact_family,
                cancel_token=cancel_token,
                legacy_reason=LegacyRouteReason.CAPSULE_MISS,
                miss_code=code,
            )

        def select():
            if offered is None:
                raise PreparedCapsuleMiss(
                    PreparedCapsuleMissCode.CAPSULE_NOT_SUPPLIED
                )
            if type(offered) is PreparedReintegrateOffer:
                if offered.disposition == "MISS":
                    if (
                        offered.bundle is not None
                        or type(offered.miss_code)
                        is not PreparedCapsuleMissCode
                    ):
                        raise PreparedCapsuleMiss(
                            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
                        )
                    if (
                        offered.miss_code
                        is PreparedCapsuleMissCode.CAPSULE_NOT_SUPPLIED
                    ):
                        # The GUI's typed lazy offer deliberately defers all
                        # admission until the operator clicks Reintegration.
                        # This is the normal direct route, not a failed
                        # prepared-capsule attempt and therefore not a warning.
                        return cls.from_artifact(
                            source_artifact,
                            entry=entry,
                            dimension=dimension,
                            preparation=preparation,
                            source_root=source_root,
                            expected_target_snapshot=expected_target_snapshot,
                            expected_terminal_identity=expected_terminal_identity,
                            expected_labels=expected_labels,
                            destination_directory=destination_directory,
                            explicit_output=explicit_output,
                            artifact_family=artifact_family,
                            cancel_token=cancel_token,
                        )
                    raise PreparedCapsuleMiss(offered.miss_code)
                if offered.disposition != "READY" or offered.bundle is None:
                    raise PreparedCapsuleMiss(
                        PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
                    )
                bundle = offered.bundle
            elif type(offered) is PreparedReintegrateBundle:
                bundle = offered
            else:
                raise PreparedCapsuleMiss(
                    PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
                )
            try:
                bundle = admit_prepared_bundle(
                    prepared_bundle_mapping(bundle)
                )
            except PreparedRouteRejected as error:
                raise PreparedCapsuleMiss(error.code) from error
            offered_source = os.path.normcase(
                os.path.abspath(os.fspath(source_artifact))
            )
            if offered_source != bundle.target.snapshot.path:
                raise PreparedCapsuleMiss(
                    PreparedCapsuleMissCode.TARGET_CHANGED
                )
            if entry != bundle.artifact.entry:
                raise PreparedCapsuleMiss(
                    PreparedCapsuleMissCode.ENTRY_CHANGED
                )
            selected = bundle.one_d if dimension == "1d" else bundle.two_d
            if selected.disposition == "MISS":
                if type(selected.miss_code) is not PreparedCapsuleMissCode:
                    raise PreparedCapsuleMiss(
                        PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
                    )
                raise PreparedCapsuleMiss(selected.miss_code)
            if (
                expected_target_snapshot is not None
                and expected_target_snapshot
                != _target_snapshot(bundle.target.snapshot)
            ):
                raise PreparedCapsuleMiss(
                    PreparedCapsuleMissCode.TARGET_CHANGED
                )
            if (
                expected_terminal_identity is not None
                and expected_terminal_identity != bundle.target.terminal
            ):
                raise PreparedCapsuleMiss(
                    PreparedCapsuleMissCode.TERMINAL_CHANGED
                )
            return cls.from_prepared_capsule(
                bundle,
                dimension=dimension,
                preparation=preparation,
                source_root=source_root,
                expected_labels=expected_labels,
                destination_directory=destination_directory,
                explicit_output=explicit_output,
                artifact_family=artifact_family,
                cancel_token=cancel_token,
            )

        try:
            return select()
        except PreparedCapsuleMiss as miss:
            _support._event(cancel_token)
            return fallback(miss.code)

    @classmethod
    def from_prepared_capsule(
        cls,
        offered: PreparedReintegrateBundle | PreparedReintegrateOffer | None,
        *,
        dimension: Literal["1d", "2d"],
        preparation: Mapping[str, object],
        source_root: str | os.PathLike[str] | None = None,
        expected_labels: tuple[int, ...] | None = None,
        destination_directory: str | os.PathLike[str] | None = None,
        explicit_output: str | os.PathLike[str] | None = None,
        artifact_family: str | None = None,
        cancel_token: threading.Event | None = None,
    ) -> "ReintegrateSuccessorPlan":
        execution = select_prepared_execution(offered, dimension)
        qualified = _qualified_from_prepared(
            execution,
            dimension=dimension,
            preparation=preparation,
            source_root=source_root,
            expected_labels=expected_labels,
            cancel_token=cancel_token,
        )
        try:
            admission = capture_finite_source(execution.target.snapshot.path)
        except FiniteArtifactIntegrityError as error:
            raise PreparedCapsuleMiss(
                PreparedCapsuleMissCode.TARGET_CHANGED
            ) from error
        if admission.snapshot != execution.target.snapshot:
            raise PreparedCapsuleMiss(PreparedCapsuleMissCode.TARGET_CHANGED)
        try:
            preflight_prepared_execution(
                execution,
                source=admission,
                entry=qualified.entry,
                dimension=dimension,
                cancel_token=cancel_token,
            )
        except PreparedRouteRejected as error:
            raise PreparedCapsuleMiss(error.code) from error
        root = Path(
            destination_directory
            if destination_directory is not None
            else Path(qualified.target).parent
        ).resolve()
        target = None if explicit_output is None else Path(explicit_output)
        source_graph_evidence = {
            "facts_digest": execution.facts_digest,
            "topology_identity": execution.topology.topology_digest,
            "source_execution_identity": (
                execution._inspection.topology.execution_digest
            ),
            "append_lineage_identity": (
                execution._inspection.topology.lineage_digest
            ),
        }
        try:
            request = _request_from_values(
                admission=admission,
                terminal=execution.target.terminal,
                qualified=qualified,
                destination_directory=root,
                explicit_target=target,
                artifact_family=artifact_family,
                route="prepared",
                source_graph_evidence=source_graph_evidence,
                prepared=execution,
                legacy_reason=None,
                miss_code=None,
            )
        except (FiniteArtifactIntegrityError, OSError, ValueError) as error:
            if _named_snapshot_changed(admission.snapshot):
                raise PreparedCapsuleMiss(
                    PreparedCapsuleMissCode.TARGET_CHANGED
                ) from error
            raise
        if _named_snapshot_changed(admission.snapshot):
            raise PreparedCapsuleMiss(PreparedCapsuleMissCode.TARGET_CHANGED)
        _support._event(cancel_token)
        return _plan_from_request(
            qualified,
            admission.snapshot,
            execution.target.terminal,
            request,
            route="prepared",
            prepared=execution,
            source_graph_evidence=source_graph_evidence,
            legacy_reason=None,
            miss_code=None,
        )

    def as_recipe(self) -> dict[str, object]:
        qualification = _support._plan_mapping(self._qualified)
        qualification.pop("rollback_policy")
        qualification.pop("operation_identity")
        finite = {
            "source_snapshot": _snapshot_mapping(self.source_snapshot),
            "expected_terminal": _terminal_mapping(self.expected_terminal),
            "output_artifact": self.output_artifact,
            "artifact_family": self.artifact_family,
            "source_graph_evidence": _support._plain(
                self.source_graph_evidence
            ),
            "source_graph_identity": self.source_graph_identity,
            "science_identity": self.science_identity,
            "output_schema": self.output_schema,
            "algorithm_identity": self.algorithm_identity,
            "preservation_identity": self.preservation_identity,
            "request_generation_identity": self.request_generation_identity,
            "resource_allocation_identity": self.resource_allocation_identity,
            "route_identity": self.route_identity,
            "custody_identity": self.custody_identity,
            "operation_context_identity": self.operation_context_identity,
            "version_identity": self.version_identity,
            "publication_identity": self.publication_identity,
            "operation_identity": self.operation_identity,
            "lineage_json": self.lineage_json,
            "lineage_identity": self.lineage_identity,
        }
        return {
            "schema": _RECIPE_SCHEMA,
            "version": _RECIPE_VERSION,
            "plan": {
                "api_version": _PLAN_API_VERSION,
                "publication_policy": _PUBLICATION_POLICY,
                "route": self.route,
                "qualification": qualification,
                "finite": finite,
                "execution": {
                    "route": (
                        "prepared"
                        if self._prepared is not None else "legacy"
                    ),
                    "legacy_reason": (
                        None
                        if self.legacy_reason is None
                        else self.legacy_reason.value
                    ),
                    "miss_code": (
                        None if self.miss_code is None else self.miss_code.value
                    ),
                    "prepared": (
                        None
                        if self._prepared is None
                        else prepared_execution_mapping(self._prepared)
                    ),
                },
            },
        }

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, object]):
        try:
            recipe, _encoded_bytes = _bounded_recipe_snapshot(recipe)
        except BoundedJsonError as error:
            raise ValueError("RECIPE_BOUNDED_JSON_UNSUPPORTED") from error
        outer = _exact_keys(recipe, {"schema", "version", "plan"}, "recipe")
        if (
            outer["schema"] == _RECIPE_SCHEMA
            and type(outer["version"]) is int
            and 0 <= outer["version"] < _RECIPE_VERSION
        ):
            raise ReintegrateRecipeMigrationRequired(outer["version"])
        if (
            outer["schema"] != _RECIPE_SCHEMA
            or type(outer["version"]) is not int
            or outer["version"] != _RECIPE_VERSION
        ):
            raise ValueError("recipe schema/version is not immutable v4")
        plan = _exact_keys(
            outer["plan"],
            {
                "api_version", "publication_policy", "route", "qualification",
                "finite", "execution",
            },
            "recipe plan",
        )
        if (
            type(plan["api_version"]) is not int
            or plan["api_version"] != _PLAN_API_VERSION
            or plan["publication_policy"] != _PUBLICATION_POLICY
            or plan["route"] not in _ROUTES
        ):
            raise ValueError("recipe plan policy is unsupported")
        execution = _exact_keys(
            plan["execution"],
            {"route", "legacy_reason", "miss_code", "prepared"},
            "recipe execution",
        )
        prepared = None
        legacy_reason = None
        miss_code = None
        if plan["route"] == "bounded-legacy":
            if execution["route"] != "legacy" or execution["prepared"] is not None:
                raise ValueError("bounded recipe carries prepared execution")
            try:
                legacy_reason = LegacyRouteReason(execution["legacy_reason"])
                miss_code = (
                    None
                    if execution["miss_code"] is None
                    else PreparedCapsuleMissCode(execution["miss_code"])
                )
            except (TypeError, ValueError) as error:
                raise ValueError("bounded recipe route reason is invalid") from error
            if (
                legacy_reason is LegacyRouteReason.DIRECT_FROM_ARTIFACT
                and miss_code is not None
                or legacy_reason is LegacyRouteReason.CAPSULE_MISS
                and miss_code is None
            ):
                raise ValueError("bounded recipe miss reason is inconsistent")
        else:
            if (
                execution["route"] != "prepared"
                or execution["legacy_reason"] is not None
                or execution["miss_code"] is not None
                or execution["prepared"] is None
            ):
                raise ValueError("prepared recipe execution is incomplete")
            prepared = admit_prepared_execution(execution["prepared"])
        qualification = _exact_keys(
            plan["qualification"],
            {
                "api_version", "target", "entry", "source_root",
                "expected_target_snapshot", "dimension", "labels",
                "detector_shape", "native_dtype", "selected_plan",
                "requested_shared_science", "gi_bootstrap_incidence",
                "retained_mask_bytes", "mask_decode_bytes", "session_policy",
                "science_identity",
            },
            "recipe qualification",
        )
        if (
            type(qualification["api_version"]) is not int
            or qualification["api_version"] != 3
        ):
            raise ValueError("recipe qualification API is unsupported")
        finite = _exact_keys(
            plan["finite"],
            {
                "source_snapshot", "expected_terminal", "output_artifact",
                "artifact_family", "source_graph_evidence",
                "source_graph_identity", "science_identity",
                "output_schema", "algorithm_identity", "preservation_identity",
                "request_generation_identity", "resource_allocation_identity",
                "route_identity", "custody_identity", "operation_context_identity",
                "version_identity", "publication_identity", "operation_identity",
                "lineage_json", "lineage_identity",
            },
            "recipe finite projection",
        )
        snapshot_value = _exact_keys(
            finite["source_snapshot"],
            {field.name for field in fields(FiniteFileSnapshot)},
            "finite source snapshot",
        )
        _bounded_recipe_text(snapshot_value["path"], "source path")
        for name in ("target", "source_root", "entry", "native_dtype"):
            _bounded_recipe_text(qualification[name], f"qualification {name}")
        _bounded_recipe_text(finite["output_artifact"], "output path")
        _bounded_recipe_text(finite["artifact_family"], "artifact family")
        snapshot = FiniteFileSnapshot(**snapshot_value)
        terminal_value = finite["expected_terminal"]
        terminal = None
        if terminal_value is not None:
            terminal_value = _exact_keys(
                terminal_value,
                {field.name for field in fields(StreamTerminal)},
                "finite source terminal",
            )
            _bounded_recipe_text(terminal_value["target"], "terminal path")
            terminal = StreamTerminal(**terminal_value)
        target_snapshot = _exact_keys(
            qualification["expected_target_snapshot"],
            {field.name for field in fields(TargetSnapshot)},
            "qualification snapshot",
        )
        if TargetSnapshot(**target_snapshot) != _target_snapshot(snapshot):
            raise ValueError("recipe source snapshots disagree")
        session = _exact_keys(
            qualification["session_policy"], {"flush", "allocation"},
            "qualification session policy",
        )
        if session["flush"] != {"interval": 8, "cap": 64, "margin": 8}:
            raise ValueError("recipe flush policy is unsupported")
        selected = _support._plain(_support._freeze(qualification["selected_plan"]))
        shared = _support._plain(
            _support._freeze(qualification["requested_shared_science"])
        )
        dimension = qualification["dimension"]
        _support._validate_science(selected, shared, dimension)
        shape = tuple(qualification["detector_shape"])
        mask_spec = _support._mask_spec(
            qualification["retained_mask_bytes"],
            qualification["mask_decode_bytes"],
            shape,
            "recipe persisted-mask resources",
        )
        requirements = _support._requirements(
            shape, qualification["native_dtype"], selected, shared,
        )
        policy = _support._policy(
            requirements,
            {"version": 1, "kind": "explicit", "allocation": session["allocation"]},
            mask_spec,
        )
        qualified = _support._make_plan(
            qualification["target"], qualification["entry"],
            qualification["source_root"], dimension,
            tuple(qualification["labels"]), shape,
            qualification["native_dtype"], selected, shared,
            qualification["gi_bootstrap_incidence"],
            qualification["retained_mask_bytes"],
            qualification["mask_decode_bytes"], policy,
            snapshot=_target_snapshot(snapshot),
            expected_science=qualification["science_identity"],
        )
        if prepared is not None and (
            prepared.target.snapshot != snapshot
            or prepared.target.terminal != terminal
            or prepared.labels != qualified.labels
            or prepared.artifact.entry != qualified.entry
            or prepared.artifact.source_base != qualified.source_root
            or prepared.artifact.detector_shape != qualified.detector_shape
            or prepared.artifact.native_dtype != qualified.native_dtype
            or _support._plain(prepared.artifact.persisted_shared_science)
            != _support._plain(qualified.requested_shared_science)
            or prepared.selected_admission.dimension != qualified.dimension
            or finite["source_graph_evidence"] != {
                "facts_digest": prepared.facts_digest,
                "topology_identity": prepared.topology.topology_digest,
                "source_execution_identity": (
                    prepared._inspection.topology.execution_digest
                ),
                "append_lineage_identity": (
                    prepared._inspection.topology.lineage_digest
                ),
            }
        ):
            raise ValueError("prepared recipe qualification changed")
        _validate_finite_projection(
            finite,
            snapshot,
            terminal,
            qualified,
            plan["route"],
            prepared=prepared,
            legacy_reason=legacy_reason,
            miss_code=miss_code,
        )
        return _plan_from_projection(
            qualified,
            snapshot,
            terminal,
            finite,
            route=plan["route"],
            prepared=prepared,
            legacy_reason=legacy_reason,
            miss_code=miss_code,
        )


def _legacy_successor_plan(
    source_artifact,
    *,
    entry,
    dimension,
    preparation,
    source_root,
    expected_target_snapshot,
    expected_terminal_identity,
    expected_labels,
    destination_directory,
    explicit_output,
    artifact_family,
    cancel_token,
    legacy_reason,
    miss_code,
):
    _support._event(cancel_token)
    try:
        preparation, _preparation_bytes = _bounded_preparation_snapshot(
            preparation,
            role="bounded legacy click science",
        )
    except BoundedJsonError as error:
        raise ValueError("REINTEGRATE_PREPARATION_UNSUPPORTED") from error
    _support._event(cancel_token)
    if (
        expected_terminal_identity is not None
        and expected_target_snapshot is None
    ):
        expected_target_snapshot = revalidate_stream_terminal(
            Path(source_artifact).resolve(), expected_terminal_identity,
        )
    qualified = _support.ReintegratePlan.from_artifact(
        source_artifact,
        entry=entry,
        dimension=dimension,
        preparation=preparation,
        source_root=source_root,
        expected_target_snapshot=expected_target_snapshot,
        expected_terminal_identity=expected_terminal_identity,
        expected_labels=expected_labels,
        cancel_token=cancel_token,
    )
    admission = capture_finite_source(qualified.target)
    if _target_snapshot(admission.snapshot) != qualified.expected_target_snapshot:
        raise ValueError("TARGET_SNAPSHOT_CHANGED")
    root = Path(
        destination_directory
        if destination_directory is not None
        else Path(qualified.target).parent
    ).resolve()
    target = None if explicit_output is None else Path(explicit_output)
    source_graph_evidence = _legacy_source_graph_evidence(qualified)
    request = _request_from_values(
        admission=admission,
        terminal=expected_terminal_identity,
        qualified=qualified,
        destination_directory=root,
        explicit_target=target,
        artifact_family=artifact_family,
        route="bounded-legacy",
        source_graph_evidence=source_graph_evidence,
        prepared=None,
        legacy_reason=legacy_reason,
        miss_code=miss_code,
    )
    return _plan_from_request(
        qualified,
        admission.snapshot,
        expected_terminal_identity,
        request,
        route="bounded-legacy",
        prepared=None,
        source_graph_evidence=source_graph_evidence,
        legacy_reason=legacy_reason,
        miss_code=miss_code,
    )


def _validate_finite_projection(
    finite,
    snapshot,
    terminal,
    qualified,
    route,
    *,
    prepared,
    legacy_reason,
    miss_code,
):
    identity_names = (
        "source_graph_identity", "science_identity", "algorithm_identity",
        "preservation_identity", "request_generation_identity",
        "resource_allocation_identity", "route_identity", "custody_identity",
        "operation_context_identity", "version_identity",
        "publication_identity", "operation_identity", "lineage_identity",
    )
    if any(not _is_sha(finite[name]) for name in identity_names):
        raise ValueError("recipe finite identity is malformed")
    if finite["science_identity"] != qualified.science_identity:
        raise ValueError("recipe finite science identity changed")
    source_graph_evidence = finite["source_graph_evidence"]
    if type(source_graph_evidence) is not dict:
        raise ValueError("recipe source graph evidence is noncanonical")
    derived = _derived_request_inputs(
        snapshot=snapshot,
        terminal=terminal,
        qualified=qualified,
        artifact_family=finite["artifact_family"],
        route=route,
        source_graph_evidence=source_graph_evidence,
        prepared=prepared,
        legacy_reason=legacy_reason,
        miss_code=miss_code,
    )
    if any(finite[name] != value for name, value in derived.items()):
        raise ValueError("recipe finite constituent identity changed")
    output = finite["output_artifact"]
    if (
        type(output) is not str
        or not os.path.isabs(output)
        or os.path.normcase(os.path.abspath(os.path.normpath(output))) != output
        or Path(output).suffix != ".nexus"
        or output == snapshot.path
        or finite["output_schema"] != _OUTPUT_SCHEMA
    ):
        raise ValueError("recipe finite output is noncanonical")
    version_payload = _version_payload(
        algorithm_identity=finite["algorithm_identity"],
        entry=qualified.entry,
        operation_kind=f"reintegrate-{qualified.dimension}",
        output_schema=finite["output_schema"],
        preservation_identity=finite["preservation_identity"],
        scientific_identity=finite["science_identity"],
        source_graph_identity=finite["source_graph_identity"],
    )
    version = _sha(version_payload)
    publication = _sha(_publication_payload(output, version))
    context = _sha(_operation_context_payload_values(
        snapshot,
        finite["request_generation_identity"],
        finite["resource_allocation_identity"],
        finite["route_identity"],
        finite["custody_identity"],
    ))
    operation = _sha(_operation_payload(
        context, output, publication, snapshot.path, version,
    ))
    lineage = admit_finite_artifact_lineage(finite["lineage_json"])
    payload = json.loads(lineage.canonical_json)
    predecessor = payload["predecessor"]
    predecessor_terminal = _exact_terminal_for_snapshot(snapshot, terminal)
    expected_terminal = (
        None
        if predecessor_terminal is None
        else {
            name: getattr(predecessor_terminal, name)
            for name in (
                "target", "size", "digest", "device", "inode",
                "mtime_ns", "ctime_ns",
            )
        }
    )
    if (
        qualified.target != snapshot.path
        or version != finite["version_identity"]
        or publication != finite["publication_identity"]
        or context != finite["operation_context_identity"]
        or operation != finite["operation_identity"]
        or lineage.lineage_identity != finite["lineage_identity"]
        or payload.get("artifact_family_v1") != finite["artifact_family"]
        or payload.get("entry") != qualified.entry
        or payload.get("output_schema") != finite["output_schema"]
        or payload.get("version_identity") != version
        or payload.get("publication_identity") != publication
        or payload.get("operation_kind") != f"reintegrate-{qualified.dimension}"
        or payload.get("source_graph_identity") != finite["source_graph_identity"]
        or payload.get("scientific_identity") != finite["science_identity"]
        or payload.get("algorithm_identity") != finite["algorithm_identity"]
        or payload.get("preservation_identity") != finite["preservation_identity"]
        or predecessor.get("source_artifact") != snapshot.path
        or predecessor.get("source_digest") != snapshot.digest
        or predecessor.get("source_size") != snapshot.size
        or predecessor.get("terminal") != expected_terminal
    ):
        raise ValueError("recipe finite identity projection changed")


def _plan_values(
    qualified,
    snapshot,
    terminal,
    request,
    route,
    prepared,
    source_graph_evidence,
    legacy_reason,
    miss_code,
):
    context = request.operation_context
    return (
        _PLAN_API_VERSION, request.source_artifact, request.output_artifact,
        qualified.entry, qualified.source_root, qualified.dimension,
        qualified.labels, qualified.detector_shape, qualified.native_dtype,
        qualified.selected_plan, qualified.requested_shared_science,
        qualified.gi_bootstrap_incidence, qualified.retained_mask_bytes,
        qualified.mask_decode_bytes, qualified.session_policy, snapshot, terminal,
        route, request.artifact_family,
        _support._freeze(_support._plain(source_graph_evidence)),
        request.source_graph_identity,
        request.scientific_identity, request.output_schema,
        request.algorithm_identity, request.preservation_identity,
        context.request_generation_identity,
        context.resource_allocation_identity, context.route_identity,
        context.custody_identity, context.context_identity,
        request.version_identity, request.publication_identity,
        request.operation_identity, legacy_reason, miss_code,
        request.lineage.canonical_json,
        request.lineage.lineage_identity, request, qualified, prepared,
    )


def _plan_from_request(
    qualified,
    snapshot,
    terminal,
    request,
    *,
    route,
    prepared,
    source_graph_evidence,
    legacy_reason,
    miss_code,
):
    return _value(
        ReintegrateSuccessorPlan,
        *_plan_values(
            qualified,
            snapshot,
            terminal,
            request,
            route,
            prepared,
            source_graph_evidence,
            legacy_reason,
            miss_code,
        ),
    )


def _plan_from_projection(
    qualified,
    snapshot,
    terminal,
    finite,
    *,
    route,
    prepared,
    legacy_reason,
    miss_code,
):
    values = (
        _PLAN_API_VERSION, snapshot.path, finite["output_artifact"],
        qualified.entry, qualified.source_root, qualified.dimension,
        qualified.labels, qualified.detector_shape, qualified.native_dtype,
        qualified.selected_plan, qualified.requested_shared_science,
        qualified.gi_bootstrap_incidence, qualified.retained_mask_bytes,
        qualified.mask_decode_bytes, qualified.session_policy, snapshot, terminal,
        route, finite["artifact_family"],
        _support._freeze(finite["source_graph_evidence"]),
        finite["source_graph_identity"],
        finite["science_identity"], finite["output_schema"],
        finite["algorithm_identity"], finite["preservation_identity"],
        finite["request_generation_identity"],
        finite["resource_allocation_identity"], finite["route_identity"],
        finite["custody_identity"], finite["operation_context_identity"],
        finite["version_identity"], finite["publication_identity"],
        finite["operation_identity"], legacy_reason, miss_code,
        finite["lineage_json"],
        finite["lineage_identity"], None, qualified, prepared,
    )
    return _value(ReintegrateSuccessorPlan, *values)


@dataclass(frozen=True, slots=True, init=False)
class ReintegrateSuccessorProgress:
    operation_identity: str
    stage: str
    completed: int
    total: int
    revision: int

    def __new__(cls, *args, **kwargs):
        raise TypeError("ReintegrateSuccessorProgress is factory-constructed")


@dataclass(frozen=True, slots=True, init=False)
class ReintegrateSuccessorResult:
    disposition: str
    source_artifact: str
    output_artifact: str
    input_labels: tuple[int, ...]
    committed_labels: tuple[int, ...]
    publication_dropped_labels: tuple[int, ...]
    diagnostics: tuple[str, ...]
    science_identity: str
    version_identity: str
    publication_identity: str
    operation_identity: str
    audit_identity: str | None
    commit_identity: str | None
    terminal: StreamTerminal | None
    hidden_orphan: str | None

    def __new__(cls, *args, **kwargs):
        raise TypeError("ReintegrateSuccessorResult is factory-constructed")


def _progress(identity, stage, completed, total, revision):
    if (
        stage not in _STAGES
        or any(type(value) is not int or value < 0 for value in (
            completed, total, revision,
        ))
        or completed > total
    ):
        raise ValueError("invalid immutable reintegration progress")
    return _value(
        ReintegrateSuccessorProgress,
        identity, stage, completed, total, revision,
    )


def _request_for_plan(plan: ReintegrateSuccessorPlan) -> tuple[
    FiniteSourceAdmission, FiniteArtifactRequest,
]:
    admission = capture_finite_source(plan.source_artifact)
    if admission.snapshot != plan.source_snapshot:
        raise ValueError("TARGET_SNAPSHOT_CHANGED")
    if plan._request is not None:
        request = plan._request
        if request.operation_context.source_snapshot != admission.snapshot:
            raise ValueError("RECIPE_FINITE_REQUEST_CHANGED")
    else:
        context = finite_operation_context(
            admission,
            request_generation_identity=plan.request_generation_identity,
            resource_allocation_identity=plan.resource_allocation_identity,
            route_identity=plan.route_identity,
            custody_identity=plan.custody_identity,
        )
        if context.context_identity != plan.operation_context_identity:
            raise ValueError("RECIPE_FINITE_REQUEST_CHANGED")
        lineage = admit_finite_artifact_lineage(plan.lineage_json)
        lineage_payload = json.loads(lineage.canonical_json)
        predecessor_payload = lineage_payload["predecessor"]
        terminal_payload = predecessor_payload["terminal"]
        terminal = _exact_predecessor_terminal(
            admission, plan.expected_terminal,
        )
        if terminal_payload is None:
            if terminal is not None:
                raise ValueError("RECIPE_FINITE_REQUEST_CHANGED")
        else:
            if terminal is None or {
                key: getattr(terminal, key)
                for key in (
                    "target", "size", "digest", "device", "inode",
                    "mtime_ns", "ctime_ns",
                )
            } != terminal_payload:
                raise ValueError("RECIPE_FINITE_REQUEST_CHANGED")
        predecessor = _predecessor(admission, terminal, plan.entry)
        if {
            "artifact_family_v1": predecessor.artifact_family_v1,
            "lineage_identity": predecessor.lineage_identity,
            "publication_identity": predecessor.publication_identity,
            "source_artifact": predecessor.source_snapshot.path,
            "source_digest": predecessor.source_snapshot.digest,
            "source_size": predecessor.source_snapshot.size,
            "terminal": terminal_payload,
            "version_identity": predecessor.version_identity,
        } != predecessor_payload:
            raise ValueError("RECIPE_FINITE_REQUEST_CHANGED")
        request = _replay_finite_artifact_request(
            source_admission=admission,
            operation_context=context,
            predecessor=predecessor,
            output_artifact=plan.output_artifact,
            artifact_family=plan.artifact_family,
            operation_kind=f"reintegrate-{plan.dimension}",
            source_graph_identity=plan.source_graph_identity,
            entry=plan.entry,
            scientific_identity=plan.science_identity,
            output_schema=plan.output_schema,
            algorithm_identity=plan.algorithm_identity,
            preservation_identity=plan.preservation_identity,
            expected_lineage=lineage,
            expected_version_identity=plan.version_identity,
            expected_publication_identity=plan.publication_identity,
            expected_operation_identity=plan.operation_identity,
        )
    expected = (
        plan.source_graph_identity,
        plan.science_identity,
        plan.output_schema,
        plan.algorithm_identity,
        plan.preservation_identity,
        plan.operation_context_identity,
        plan.version_identity,
        plan.publication_identity,
        plan.operation_identity,
        plan.lineage_json,
        plan.lineage_identity,
    )
    observed = (
        request.source_graph_identity,
        request.scientific_identity,
        request.output_schema,
        request.algorithm_identity,
        request.preservation_identity,
        request.operation_context_identity,
        request.version_identity,
        request.publication_identity,
        request.operation_identity,
        request.lineage.canonical_json,
        request.lineage.lineage_identity,
    )
    if observed != expected:
        raise ValueError("RECIPE_FINITE_REQUEST_CHANGED")
    return admission, request


def _selected_labels(document: h5py.File, entry: str, dimension: str):
    from xrd_tools.io.record_writer import _read_replacement_frame_index

    group = document.get(f"/{entry}/integrated_{dimension}")
    if not isinstance(group, h5py.Group):
        raise FiniteArtifactIntegrityError("successor selected result is absent")
    link = group.get("frame_index", getlink=True)
    node = group.get("frame_index")
    if type(link) is not h5py.HardLink or not isinstance(node, h5py.Dataset):
        raise FiniteArtifactIntegrityError(
            "successor selected inventory is not local"
        )
    try:
        values = _read_replacement_frame_index(
            node, "successor selected inventory", require_nonempty=True,
        )
    except BaseException as error:
        raise FiniteArtifactIntegrityError(
            "successor selected inventory is malformed"
        ) from error
    labels = tuple(int(value) for value in values)
    if labels != tuple(sorted(set(labels))) or any(value < 0 for value in labels):
        raise FiniteArtifactIntegrityError(
            "successor selected inventory is noncanonical"
        )
    return labels


def _audit_from_document(document, plan):
    from xrd_tools.io.record_writer import _replacement_utf8_scalar

    node = document.get(
        f"/{plan.entry}/reduction/config/dimension_replacement_{plan.dimension}"
    )
    if not isinstance(node, h5py.Dataset):
        raise FiniteArtifactIntegrityError("successor deterministic audit is absent")
    try:
        text = _replacement_utf8_scalar(node, "successor deterministic audit")
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FiniteArtifactIntegrityError(
            "successor deterministic audit is malformed"
        ) from error
    if text != json.dumps(value, sort_keys=True, separators=(",", ":")):
        raise FiniteArtifactIntegrityError(
            "successor deterministic audit is noncanonical"
        )
    return value, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config_json(document, plan, name, role, *, required=True, max_bytes=None):
    from xrd_tools.io.record_writer import (
        _replacement_hard_group,
        _replacement_utf8_scalar,
    )

    config = _replacement_hard_group(
        document, f"{plan.entry}/reduction/config",
    )
    node = _replacement_hard_group(config, name, h5py.Dataset)
    if node is None and not required:
        return None, None
    if not isinstance(node, h5py.Dataset):
        raise FiniteArtifactIntegrityError(f"successor {role} is absent or nonlocal")
    try:
        text = _replacement_utf8_scalar(node, role, max_bytes=max_bytes)
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FiniteArtifactIntegrityError(
            f"successor {role} is malformed"
        ) from error
    if text != json.dumps(value, sort_keys=True, separators=(",", ":")):
        raise FiniteArtifactIntegrityError(
            f"successor {role} is noncanonical"
        )
    return value, text


def _expected_gi_values(plan, source_document=None) -> dict[str, object]:
    gi_name = f"gi_mode_{plan.dimension}"
    if plan._prepared is not None:
        receipt = plan._prepared.selected_admission.payload.manifest_receipt
        try:
            values = json.loads(receipt.gi_preserved_json)
        except json.JSONDecodeError as error:
            raise FiniteArtifactIntegrityError(
                "prepared GI preservation receipt is malformed"
            ) from error
    else:
        try:
            if not isinstance(source_document, h5py.File):
                raise OSError("legacy source document is not bracketed")
            values, _raw = _config_json(
                source_document,
                plan,
                "gi_config",
                "source GI config",
                required=False,
            )
        except (OSError, ValueError, TypeError) as error:
            raise FiniteArtifactIntegrityError(
                "source GI preservation proof is unavailable"
            ) from error
        values = {} if values is None else values
        if type(values) is not dict:
            raise FiniteArtifactIntegrityError(
                "source GI preservation proof is malformed"
            )
        values = dict(values)
        values.pop(gi_name, None)
    if type(values) is not dict:
        raise FiniteArtifactIntegrityError(
            "successor GI preservation proof is malformed"
        )
    expected = dict(values)
    selected = plan.selected_plan["gi_mode"]
    if selected is not None:
        expected[gi_name] = str(selected)
    return expected


def _validate_selected_source_projection(
    document, plan, inspected, *, expected_gi_values,
) -> None:
    from xrd_tools.io.record_writer import _replacement_hard_group

    bai, _raw = _config_json(
        document,
        plan,
        f"bai_{plan.dimension}_args",
        "selected BAI config",
    )
    if bai != _support._plain(plan.selected_plan["bai_args"]):
        raise FiniteArtifactIntegrityError(
            "successor selected BAI config changed"
        )
    gi, _raw = _config_json(
        document,
        plan,
        "gi_config",
        "GI config",
        required=False,
    )
    if (
        bool(expected_gi_values)
        and gi != expected_gi_values
        or not expected_gi_values
        and gi is not None
    ):
        raise FiniteArtifactIntegrityError("successor GI config changed")
    execution, execution_text = _config_json(
        document,
        plan,
        "source_execution",
        "source execution",
        max_bytes=64 << 20,
    )
    expected_execution = _support._plain(inspected.topology.execution)
    if (
        execution != expected_execution
        or execution_text != json.dumps(
            expected_execution, sort_keys=True, separators=(",", ":"),
        )
    ):
        raise FiniteArtifactIntegrityError(
            "successor source execution changed"
        )
    lineage, lineage_text = _config_json(
        document,
        plan,
        "append_lineage",
        "append lineage",
        required=False,
        max_bytes=64 << 20,
    )
    expected_lineage = (
        None
        if inspected.append_lineage is None
        else inspected.append_lineage.decode("utf-8", errors="strict")
    )
    if lineage_text != expected_lineage:
        raise FiniteArtifactIntegrityError(
            "successor append lineage changed"
        )
    entry = _replacement_hard_group(document, plan.entry)
    source_base = None if entry is None else entry.attrs.get("source_base")
    if isinstance(source_base, bytes):
        try:
            source_base = source_base.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise FiniteArtifactIntegrityError(
                "successor source root is malformed"
            ) from error
    if source_base != Path(plan.source_root).as_posix():
        raise FiniteArtifactIntegrityError("successor source root changed")


def _validate_candidate_document(
    document,
    plan,
    request,
    *,
    inspected,
    expected_gi_values,
    expected_audit=None,
):
    from xrd_tools.io.record_writer import (
        WriterStateError,
        require_finite_replacement_result_seal,
    )

    if (
        type(document) is not h5py.File
        or not document.id.valid
        or document.mode != "r"
        or document.attrs.get("file_name") != request.output_artifact
    ):
        raise FiniteArtifactIntegrityError(
            "successor candidate is not the exact current processed document"
        )
    try:
        from xrd_tools.io.processed_scan_id import require_current_writable_processed_groups
        require_current_writable_processed_groups(
            document, plan.entry, container=request.output_artifact,
        )
    except ValueError as error:
        raise FiniteArtifactIntegrityError(
            "successor candidate is not the exact current processed document"
        ) from error
    lineage = require_finite_artifact_lineage(document, request)
    labels = _selected_labels(document, plan.entry, plan.dimension)
    if not labels or not set(labels) <= set(plan.labels):
        raise FiniteArtifactIntegrityError(
            "successor candidate selected labels escaped the plan"
        )
    audit, identity = _audit_from_document(document, plan)
    if expected_audit is None:
        expected_audit = _support._dimension_audit(
            dimension=plan.dimension,
            operation_identity=request.version_identity,
            science_identity=plan.science_identity,
            acquisition_fingerprint=inspected.acquisition_fingerprint,
            requested_shared_science=plan.requested_shared_science,
            selected_plan=plan.selected_plan,
            append_lineage=inspected.append_lineage,
        )
    if (
        audit.get("operation_identity") != request.version_identity
        or audit.get("science_identity") != plan.science_identity
        or identity != _support._audit_identity(audit)
        or expected_audit is not None
        and audit != expected_audit
    ):
        raise FiniteArtifactIntegrityError(
            "successor candidate deterministic audit changed"
        )
    _validate_selected_source_projection(
        document,
        plan,
        inspected,
        expected_gi_values=expected_gi_values,
    )
    try:
        result_digest = require_finite_replacement_result_seal(
            document,
            entry=plan.entry,
            dimension=plan.dimension,
            audit_identity=identity,
        )
    except WriterStateError as error:
        raise FiniteArtifactIntegrityError(
            "successor candidate selected result seal changed"
        ) from error
    return FiniteCandidateValidation(lineage), labels, identity, result_digest


def _preservation_expectation(plan, source_document=None):
    from xrd_tools.io.record_writer import (
        WriterStateError,
        _replacement_exclusions_for,
        _replacement_hard_group,
        _replacement_manifest_digest_for,
    )

    try:
        if plan._prepared is not None:
            receipt = plan._prepared.selected_admission.payload.manifest_receipt
            return receipt.exclusions, receipt.manifest_digest
        else:
            if not isinstance(source_document, h5py.File):
                raise OSError("legacy source document is not bracketed")
            source_entry = _replacement_hard_group(source_document, plan.entry)
            source_exclusions = _replacement_exclusions_for(
                source_entry,
                plan.dimension,
                include_finite_lineage=True,
            )
            expected = _replacement_manifest_digest_for(
                source_document,
                plan.entry,
                source_exclusions,
                ignore_source_base=True,
                ignore_file_name=True,
                normalize_integrated_axes=True,
            )
            return source_exclusions, expected
    except (OSError, ValueError, TypeError, WriterStateError) as error:
        raise FiniteArtifactIntegrityError(
            "source preservation proof is unavailable"
        ) from error


def _require_source_document_snapshot(document, snapshot) -> None:
    try:
        descriptor = document.id.get_vfd_handle()
        observed = os.fstat(descriptor)
        named = os.stat(snapshot.path, follow_symlinks=False)
        shown = os.path.normcase(os.path.abspath(os.fspath(document.filename)))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise FiniteArtifactIntegrityError(
            "source expectation document identity is unavailable"
        ) from error
    expected = (
        snapshot.device,
        snapshot.inode,
        snapshot.mode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )
    names = (
        "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns",
    )
    if (
        type(descriptor) is not int
        or shown != os.path.normcase(os.path.abspath(snapshot.path))
        or tuple(int(getattr(observed, name)) for name in names) != expected
        or tuple(int(getattr(named, name)) for name in names) != expected
    ):
        raise FiniteArtifactIntegrityError(
            "source expectation document changed"
        )


def _source_expectations(plan):
    if plan._prepared is not None:
        return _expected_gi_values(plan), _preservation_expectation(plan)
    try:
        with h5py.File(plan.source_artifact, "r") as source:
            _require_source_document_snapshot(source, plan.source_snapshot)
            gi_values = _expected_gi_values(plan, source)
            preservation = _preservation_expectation(plan, source)
            _require_source_document_snapshot(source, plan.source_snapshot)
    except OSError as error:
        raise FiniteArtifactIntegrityError(
            "source expectations are unavailable"
        ) from error
    return gi_values, preservation


def _validate_committed_preservation(document, plan, expectation) -> None:
    from xrd_tools.io.record_writer import (
        WriterStateError,
        _replacement_exclusions_for,
        _replacement_hard_group,
        _replacement_manifest_digest_for,
    )

    expected_exclusions, expected = expectation
    try:
        candidate_entry = _replacement_hard_group(document, plan.entry)
        candidate_exclusions = _replacement_exclusions_for(
            candidate_entry, plan.dimension, include_finite_lineage=True,
        )
        if candidate_exclusions != expected_exclusions:
            raise WriterStateError("finite preservation exclusions changed")
        observed = _replacement_manifest_digest_for(
            document,
            plan.entry,
            candidate_exclusions,
            ignore_source_base=True,
            ignore_file_name=True,
            normalize_integrated_axes=True,
        )
    except (OSError, ValueError, TypeError, WriterStateError) as error:
        raise FiniteArtifactIntegrityError(
            "committed successor preservation proof is unavailable"
        ) from error
    if observed != expected:
        raise FiniteArtifactIntegrityError(
            "committed successor changed preserved source content"
        )


def _require_terminal_source_topology(inspected, cancel_token=None) -> None:
    try:
        _support._validate_terminal_topology(inspected.topology, cancel_token)
    except (OSError, ValueError, TypeError) as error:
        raise FiniteArtifactIntegrityError(
            "immutable reintegration source topology changed"
        ) from error


def _inspect_committed(
    path: Path,
    request: FiniteArtifactRequest,
    plan,
    *,
    inspected,
    expected_gi_values,
    expected_preservation,
    expected_result_digest=None,
    expected_labels=None,
    capture=None,
    preservation_prevalidated: bool = False,
):
    snapshot = _support.capture_target_snapshot(path)
    if not snapshot.exists:
        raise FiniteArtifactIntegrityError("committed successor is absent")
    expected_audit = _support._dimension_audit(
        dimension=plan.dimension,
        operation_identity=request.version_identity,
        science_identity=plan.science_identity,
        acquisition_fingerprint=inspected.acquisition_fingerprint,
        requested_shared_science=plan.requested_shared_science,
        selected_plan=plan.selected_plan,
        append_lineage=inspected.append_lineage,
    )
    with h5py.File(path, "r") as document:
        validation, labels, audit_identity, result_digest = (
            _validate_candidate_document(
            document,
            plan,
            request,
            inspected=inspected,
            expected_gi_values=expected_gi_values,
            expected_audit=expected_audit,
            )
        )
        if expected_labels is not None and labels != expected_labels:
            raise FiniteArtifactIntegrityError(
                "committed successor selected labels changed"
            )
        if (
            expected_result_digest is not None
            and result_digest != expected_result_digest
        ):
            raise FiniteArtifactIntegrityError(
                "committed successor selected result changed"
            )
        _validate_committed_preservation(
            document, plan, expected_preservation,
        )
    if not preservation_prevalidated:
        _require_terminal_source_topology(inspected)
    final = _support.capture_target_snapshot(path)
    state = os.stat(path)
    if final != snapshot:
        raise FiniteArtifactIntegrityError(
            "committed successor changed during inspection"
        )
    if capture is not None:
        capture(labels, audit_identity, result_digest)
    terminal = StreamTerminal(
        str(path), int(snapshot.size), str(snapshot.digest), 1,
        int(state.st_dev), int(state.st_ino), int(state.st_mtime_ns),
        int(state.st_ctime_ns),
    )
    return FiniteCommittedInspection(terminal, validation.lineage)


def _preflight_legacy_plan(plan, token):
    source_path = Path(plan.source_artifact)
    source_snapshot = _target_snapshot(plan.source_snapshot)
    revision = _support._target_object_revision(source_path, source_snapshot)
    inspected = _support._inspect_artifact(
        source_path,
        plan.entry,
        plan.dimension,
        source_snapshot,
        revision,
        plan.source_root,
        read_mask=False,
    )
    expected_mask = _support._PersistedMaskSpec(
        plan.retained_mask_bytes, plan.mask_decode_bytes,
    )
    evidence = {
        "facts_digest": None,
        "topology_identity": source_topology_identity(inspected.topology),
        "source_execution_identity": inspected.topology.execution_digest,
        "append_lineage_identity": inspected.topology.lineage_digest,
    }
    if (
        inspected.labels != plan.labels
        or inspected.detector_shape != plan.detector_shape
        or inspected.native_dtype != plan.native_dtype
        or inspected.mask_spec != expected_mask
        or _support._plain(inspected.persisted_shared_science)
        != _support._plain(plan.requested_shared_science)
        or evidence != _support._plain(plan.source_graph_evidence)
        or plan.gi_bootstrap_incidence is not None
        and inspected.gi_values.get(plan.labels[0])
        != plan.gi_bootstrap_incidence
    ):
        raise ValueError("RECIPE_ARTIFACT_FACTS_CHANGED")
    _support._event(token)
    return inspected


class _SuccessorRuntime:
    def __init__(self, plan, token, progress_cb):
        self.plan = plan
        self.token = token
        self.progress_cb = progress_cb
        self.revision = 0
        self.diagnostics: list[str] = (
            [f"PREPARED_CAPSULE_MISS:{plan.miss_code.value}"]
            if (
                plan.legacy_reason is LegacyRouteReason.CAPSULE_MISS
                and type(plan.miss_code) is PreparedCapsuleMissCode
            )
            else []
        )
        self.source = None
        self.sink = None
        self.session = None
        self.accounting = None
        self.seal = None
        self.audit = None
        self.audit_identity = None
        self.committed_labels: tuple[int, ...] = ()
        self.dropped_labels: tuple[int, ...] = ()
        self.candidate_preservation_validated = False
        self.preflight_inspection = None
        self.expected_gi_values = None
        self.expected_preservation = None
        self.candidate_result_digest = None
        self.candidate_labels = None
        self.candidate_audit_identity = None
        self.candidate_validation = None
        self.observed_committed_labels = None
        self.observed_audit_identity = None
        self.observed_result_digest = None

    def _report(self, stage, completed, total):
        self.revision += 1
        value = _progress(
            self.plan.operation_identity, stage, completed, total, self.revision,
        )
        if self.progress_cb is not None:
            try:
                self.progress_cb(value)
            except BaseException as error:
                self._note(error)

    def _note(self, value):
        if len(self.diagnostics) < 16:
            self.diagnostics.append(_support._diagnostic(value))

    def _pre_candidate_abort_result(self):
        return _value(
            ReintegrateSuccessorResult,
            FiniteArtifactDisposition.ABORTED.value,
            self.plan.source_artifact,
            self.plan.output_artifact,
            self.plan.labels,
            (),
            (),
            tuple(self.diagnostics),
            self.plan.science_identity,
            self.plan.version_identity,
            self.plan.publication_identity,
            self.plan.operation_identity,
            None,
            None,
            None,
            None,
        )

    def _stop(self, error):
        if self.session is None:
            return
        try:
            self.session._session._record_failure(error)
        except BaseException as cleanup:
            self._note(cleanup)
        try:
            self.session.stop()
        except BaseException as cleanup:
            self._note(cleanup)

    def _discard_private_candidate(self, reason: BaseException) -> None:
        nexus = self.session._require_external_publication_sink()
        try:
            self.session._abort_external_publication_preparation(nexus, reason)
        except BaseException as observed:
            if (
                observed is reason
                and self.session._dynamic_terminal_settled
                and not self.session._dynamic_external_publication_pending
            ):
                return
            raise
        raise RuntimeError("private candidate discard lost its terminal outcome")

    def _settle_external_publication(self, outcome, action) -> None:
        first: BaseException | None = None
        latest: BaseException | None = None
        for _attempt in range(3):
            try:
                action(self.seal, outcome)
            except BaseException as error:
                first = error if first is None else first
                latest = error
                if self.session._dynamic_terminal_settled:
                    self._note(error)
                    return
                continue
            if first is not None:
                self._note(first)
            return
        try:
            self.session._recover_external_publication_settlement(
                self.seal, outcome,
            )
        except BaseException as error:
            latest = error
            if self.session._dynamic_terminal_settled:
                self._note(first or error)
                return
        else:
            if first is not None:
                self._note(first)
            return
        if first is not None:
            raise first from latest
        raise RuntimeError("external publication settlement made no attempt")

    def _settle_publication(self, publication: FiniteArtifactResult) -> None:
        action = (
            self.session.commit_external_publication
            if publication.disposition in {
                FiniteArtifactDisposition.COMMITTED,
                FiniteArtifactDisposition.ALREADY_COMMITTED,
            }
            else self.session.abort_external_publication
        )
        self._settle_external_publication(publication, action)

    def _settle_publisher_failure(self, error: BaseException) -> None:
        action = (
            self.session.hold_external_publication
            if type(error) is FiniteArtifactPublicationHeld
            else self.session.abort_external_publication
        )
        try:
            self._settle_external_publication(error, action)
        except BaseException as cleanup:
            self._note(cleanup)

    def execute_candidate(
        self, document, request, seed_binding, candidate_binding,
    ):
        from xrd_tools.reduction.core import NexusSink
        from xrd_tools.session import (
            DynamicAccountingLimits,
            DynamicFrameIdentity,
            DynamicRunAccounting,
            StageLedger,
            required_result_modes,
        )
        from xrd_tools.session.scan_session import ScanSession

        plan = self.plan
        from xrd_tools.io.processed_scan_id import upgrade_private_integrated_axes
        upgrade_private_integrated_axes(document, plan.entry, container=request.output_artifact)
        try:
            _support._event(self.token)
        except _support.ReintegrateCancelled:
            return FiniteCandidateWriteDisposition.ABORTED
        self._report("qualify", 0, len(plan.labels))
        try:
            _support._event(self.token)
        except _support.ReintegrateCancelled:
            return FiniteCandidateWriteDisposition.ABORTED
        source_path = Path(plan.source_artifact)
        source_snapshot = _target_snapshot(plan.source_snapshot)
        revision = _support._target_object_revision(source_path, source_snapshot)
        prepared = plan._prepared
        inspected = (
            prepared._inspection
            if prepared is not None else self.preflight_inspection
        )
        if inspected is None:
            raise RuntimeError("bounded legacy route lost its preflight facts")
        if (
            _support.capture_target_snapshot(source_path) != source_snapshot
            or _support._target_object_revision(
                source_path, source_snapshot,
            ) != revision
        ):
            raise ValueError("TARGET_SNAPSHOT_CHANGED")
        expected_mask = _support._PersistedMaskSpec(
            plan.retained_mask_bytes, plan.mask_decode_bytes,
        )
        if (
            inspected.labels != plan.labels
            or inspected.detector_shape != plan.detector_shape
            or inspected.native_dtype != plan.native_dtype
            or inspected.mask_spec != expected_mask
            or _support._plain(inspected.persisted_shared_science)
            != _support._plain(plan.requested_shared_science)
            or plan.gi_bootstrap_incidence is not None
            and inspected.gi_values.get(plan.labels[0])
            != plan.gi_bootstrap_incidence
        ):
            raise ValueError("RECIPE_ARTIFACT_FACTS_CHANGED")
        mask = (
            _support._load_persisted_mask(
                source_path, plan.entry, inspected.detector_shape,
                source_snapshot, revision, inspected.mask_spec,
            )
            if prepared is None and inspected.mask_spec.retained_bytes else None
        )
        inspected = inspected._replace(mask=mask)
        try:
            _support._event(self.token)
        except _support.ReintegrateCancelled:
            return FiniteCandidateWriteDisposition.ABORTED
        self.audit = _support._dimension_audit(
            dimension=plan.dimension,
            operation_identity=request.version_identity,
            science_identity=plan.science_identity,
            acquisition_fingerprint=inspected.acquisition_fingerprint,
            requested_shared_science=plan.requested_shared_science,
            selected_plan=plan.selected_plan,
            append_lineage=inspected.append_lineage,
        )
        self.audit_identity = _support._audit_identity(self.audit)
        background = plan.requested_shared_science["background"]
        run = {} if background["mode"] == "None" else {
            "background": _support._plain(background),
        }
        lock = threading.RLock()
        self.source = _support._ReintegrateFrameSource(
            plan, self.token, inspected.raw_options, inspected.topology,
        )
        prepared_manifest_admission = None
        if prepared is not None:
            prepared_manifest_admission = bind_prepared_manifest_receipt(
                prepared.selected_admission.payload.manifest_receipt,
                facts_digest=prepared.facts_digest,
                request=request,
                seed_binding=seed_binding,
                candidate_binding=candidate_binding,
            )
        self.sink = NexusSink.for_finite_replacement(
            request.output_artifact,
            document,
            request,
            seed_binding,
            candidate_binding,
            dimension=plan.dimension,
            labels=plan.labels,
            audit_bytes=_support._canonical(self.audit),
            selected_plan=plan.selected_plan["bai_args"],
            selected_gi_mode=plan.selected_plan["gi_mode"],
            source_execution=_support._plain(inspected.topology.execution),
            append_lineage=inspected.append_lineage,
            cancel_token=self.token,
            entry=plan.entry,
            source_base=inspected.source_base,
            run_configuration_provenance=run,
            prepared_manifest_admission=prepared_manifest_admission,
            write_thumbnails=False,
            flush_every=None,
            file_lock=lock,
        )
        self.sink._configure_writer_batch_size(
            _support._replacement_writer_batch_size(plan.resource_allocation)
        )
        if prepared is None:
            self.source.bind_fact_reader(
                lambda label, **kwargs:
                self.sink._writer._detach_replacement_fact(label, **kwargs)
            )
        else:
            def prepared_fact(label, *, metadata_keys=(), include_geometry=False):
                if metadata_keys or include_geometry:
                    raise FiniteArtifactIntegrityError(
                        "prepared capsule phase-B integrity failure: "
                        "REQUEST_REQUIRES_UNPREPARED_FACTS"
                    )
                try:
                    return prepared._facts_by_label[int(label)]
                except (KeyError, TypeError, ValueError) as error:
                    raise FiniteArtifactIntegrityError(
                        "prepared capsule phase-B integrity failure: "
                        "ARTIFACT_FACTS_CHANGED"
                    ) from error

            self.source.bind_fact_reader(prepared_fact)
        core_plan = _support._core_plan(
            plan.selected_plan, plan.requested_shared_science, inspected.mask,
        )
        modes = required_result_modes(core_plan)
        targets = {
            mode: (f"nexus:{request.output_artifact}",) for mode in modes
        }
        ledger = StageLedger(required_modes=modes, targets_by_mode=targets)
        self.accounting = DynamicRunAccounting(
            ledger,
            run_generation=1,
            limits=DynamicAccountingLimits(1, 1, len(plan.labels)),
        )
        try:
            self.session = ScanSession(
                core_plan,
                self.source,
                self.sink,
                policy=plan.session_policy,
                cancel_token=self.token,
                clear_frame_images=True,
                accounting=ledger,
                dynamic_accounting=self.accounting,
                targets_by_mode=targets,
                _dynamic_batch_settlement_authority_cb=(
                    self.source.enqueue_settled_batch
                ),
            )
        except BaseException:
            self.sink._abort_external_publication()
            raise
        engine = self.session._session
        dead_writer = RuntimeError(
            "immutable reintegration writer exited before settlement"
        )
        self.source.bind_failure_probe(
            lambda: _support._replacement_engine_failure(engine, dead_writer)
        )
        total = len(plan.labels)
        submitted = 0
        settled = 0
        stopped = False
        cancelled_by_request = False

        def release_progress(attempts):
            nonlocal settled
            if not attempts:
                return
            settled += len(attempts)
            self._report("reduce", settled, total)
            self._report("write", settled, total)

        primary = None
        try:
            self.source.open_direct_hdf()
            self.session.start()
            self.sink._writer._replacement_read_context = (
                inspected.source_base,
                inspected.topology.lineage,
                inspected.topology.execution,
            )
            frames = tuple(self.session.scan.frames)
            for ordinal, frame in enumerate(frames):
                if self.token is not None and self.token.is_set():
                    self.session.stop()
                    stopped = True
                    cancelled_by_request = True
                    break
                release_progress(self.source.wait_for_capacity())
                self._report("read", ordinal, total)
                image, source_revision = self.source.prepare(frame)
                key = DynamicFrameIdentity(
                    plan.operation_identity, int(frame.index),
                )
                self.accounting.discover(
                    key, group=plan.operation_identity, ordinal=ordinal,
                    output_label=int(frame.index),
                )
                attempt = self.accounting.begin_attempt(
                    key, source_revision=source_revision,
                )
                self.accounting.record_enqueued(attempt)
                self.source.bind_attempt(frame.index, attempt)
                if not self.session.submit(
                    frame, image, attempt_token=attempt,
                ):
                    self.accounting.record_cancelled(
                        attempt, reason="immutable reintegration submission cancelled",
                    )
                    self.source.clear_label(frame.index)
                    self.session.stop()
                    stopped = True
                    cancelled_by_request = bool(
                        self.token is not None and self.token.is_set()
                    )
                    break
                submitted += 1
            _support._drain_reintegration_engine(
                engine, self.source, _support._TERMINAL_DRAIN_TIMEOUT_SECONDS,
            )
            if not stopped:
                release_progress(self.source.consume_settled())
                if (
                    submitted != total
                    or settled != total
                    or self.source.jit_labels
                    or self.source.pending_settlement_count
                ):
                    raise RuntimeError(
                        "immutable reintegration lost rolling settlement custody"
                    )
            if (
                not stopped
                and engine._current_failure() is None
                and (self.token is None or not self.token.is_set())
            ):
                self.session.flush(force=True)
                self.source.validate_terminal_topology()
            snapshot = self.accounting.snapshot()
            pairs = (
                snapshot.publication_dropped
                | snapshot.pending_publication_dropped
            )
            self.dropped_labels = tuple(
                label for label in plan.labels
                if any(
                    key.logical_frame_identity == label
                    for key, _mode in pairs
                )
                )
            if (
                not stopped
                and self.dropped_labels == plan.labels
                and primary is None
            ):
                self._discard_private_candidate(RuntimeError(
                    "finite reintegration produced no publishable rows"
                ))
                return FiniteCandidateWriteDisposition.ABORTED
        except _support.ReintegrateCancelled as error:
            primary = error
            stopped = True
            cancelled_by_request = True
            try:
                self.session.stop()
            except BaseException as cleanup:
                self._note(cleanup)
        except BaseException as error:
            primary = error
            self._stop(error)
        try:
            self.seal = self.session.prepare_external_publication(
                join_timeout=_support._TERMINAL_DRAIN_TIMEOUT_SECONDS,
            )
        except BaseException:
            if (
                primary is not None
                and type(primary) is not _support.ReintegrateCancelled
            ):
                raise primary
            if (
                (
                    cancelled_by_request
                    or self.token is not None
                    and self.token.is_set()
                )
                and self.session._dynamic_terminal_settled
            ):
                return FiniteCandidateWriteDisposition.ABORTED
            raise
        if primary is not None:
            raise primary
        return self.seal

    def validate_candidate(self, document, request):
        self._report("validate", 0, 1)
        validation, labels, audit_identity, result_digest = (
            _validate_candidate_document(
            document,
            self.plan,
            request,
            inspected=self.preflight_inspection,
            expected_gi_values=self.expected_gi_values,
            expected_audit=self.audit,
            )
        )
        _validate_committed_preservation(
            document, self.plan, self.expected_preservation,
        )
        expected_labels = tuple(
            label for label in self.plan.labels
            if label not in set(self.dropped_labels)
        )
        if labels != expected_labels:
            raise FiniteArtifactIntegrityError(
                "candidate selected labels disagree with publication accounting"
            )
        if self.audit_identity is not None and audit_identity != self.audit_identity:
            raise FiniteArtifactIntegrityError(
                "candidate validation changed the deterministic audit"
            )
        self.committed_labels = labels
        self.candidate_labels = labels
        self.candidate_audit_identity = audit_identity
        self.candidate_result_digest = result_digest
        self.candidate_validation = validation
        self._report("validate", 1, 1)
        self.candidate_preservation_validated = True
        return validation

    def _capture_committed_observation(
        self, labels, audit_identity, result_digest,
    ) -> None:
        observed = (tuple(labels), str(audit_identity), str(result_digest))
        prior = (
            self.observed_committed_labels,
            self.observed_audit_identity,
            self.observed_result_digest,
        )
        if prior != (None, None, None) and prior != observed:
            raise FiniteArtifactIntegrityError(
                "committed successor observation changed"
            )
        (
            self.observed_committed_labels,
            self.observed_audit_identity,
            self.observed_result_digest,
        ) = observed

    def _accept_validated_commit(self, validation) -> None:
        if (
            validation is not self.candidate_validation
            or self.candidate_labels is None
            or self.candidate_audit_identity is None
            or self.candidate_result_digest is None
            or not self.candidate_preservation_validated
        ):
            raise FiniteArtifactIntegrityError(
                "prepared successor lost its validated commit receipt"
            )
        self._capture_committed_observation(
            self.candidate_labels,
            self.candidate_audit_identity,
            self.candidate_result_digest,
        )

    def close(self):
        source = self.source
        writer = None if self.sink is None else self.sink._writer
        if source is not None:
            source.close_direct_hdf(validate=False)
            source.clear_jit()
            source._frames.clear()
            source._fact_reader = None
            source._topology = None
        if writer is not None:
            for name in (
                "_replacement_configuration", "_replacement_read_context",
                "_replacement_manifest", "_replacement_expected",
            ):
                setattr(writer, name, None)
            writer._row_cursors.clear()
            writer._replacement_labels = ()

    def run(self):
        if self.token is not None and self.token.is_set():
            return self._pre_candidate_abort_result()
        try:
            admission, request = _request_for_plan(self.plan)
        except (FiniteArtifactIntegrityError, OSError) as error:
            if self.token is not None and self.token.is_set():
                return self._pre_candidate_abort_result()
            if self.plan._prepared is not None:
                raise PreparedRouteChanged(
                    PreparedCapsuleMissCode.TARGET_CHANGED
                ) from error
            raise
        except ValueError as error:
            if self.token is not None and self.token.is_set():
                return self._pre_candidate_abort_result()
            if (
                self.plan._prepared is not None
                and str(error) == "TARGET_SNAPSHOT_CHANGED"
            ):
                raise PreparedRouteChanged(
                    PreparedCapsuleMissCode.TARGET_CHANGED
                ) from error
            raise
        if self.token is not None and self.token.is_set():
            return self._pre_candidate_abort_result()
        if self.plan._request is not None and request != self.plan._request:
            raise ValueError("FINITE_REQUEST_REPLAY_CHANGED")
        try:
            if self.plan._prepared is not None:
                self.preflight_inspection = self.plan._prepared._inspection
                if self.token is None or not self.token.is_set():
                    preflight_prepared_execution(
                        self.plan._prepared,
                        source=admission,
                        entry=self.plan.entry,
                        dimension=self.plan.dimension,
                        cancel_token=self.token,
                    )
            else:
                self.preflight_inspection = _preflight_legacy_plan(
                    self.plan,
                    None
                    if self.token is not None and self.token.is_set()
                    else self.token,
                )
        except _support.ReintegrateCancelled:
            if self.token is not None and self.token.is_set():
                return self._pre_candidate_abort_result()
            raise
        except PreparedRouteRejected as error:
            if self.plan._prepared is None:
                raise
            raise PreparedRouteChanged(error.code) from error
        if self.token is not None and self.token.is_set():
            return self._pre_candidate_abort_result()
        try:
            (
                self.expected_gi_values,
                self.expected_preservation,
            ) = _source_expectations(self.plan)
        except BaseException:
            if self.token is not None and self.token.is_set():
                return self._pre_candidate_abort_result()
            raise
        if self.token is not None and self.token.is_set():
            return self._pre_candidate_abort_result()
        adapter = FiniteSeededDocumentAdapter(
            lambda binding: h5py.File(binding, "r+"),
            lambda document, seed_binding, candidate_binding: self.execute_candidate(
                document, request, seed_binding, candidate_binding,
            ),
            lambda binding: h5py.File(binding, "r"),
            lambda document: self.validate_candidate(document, request),
        )
        publisher = FiniteArtifactPublisher(request, cancel_token=self.token)
        self._report("publish", 0, 1)
        try:
            publication = publisher.publish(
                adapter,
                inspect_committed=lambda path, selected: _inspect_committed(
                    path,
                    selected,
                    self.plan,
                    inspected=self.preflight_inspection,
                    expected_gi_values=self.expected_gi_values,
                    expected_preservation=self.expected_preservation,
                    expected_result_digest=self.candidate_result_digest,
                    expected_labels=self.candidate_labels,
                    capture=self._capture_committed_observation,
                    preservation_prevalidated=(
                        self.candidate_preservation_validated
                    ),
                ),
                seed=admission,
                prepublish=lambda: _require_terminal_source_topology(
                    self.preflight_inspection,
                ),
                accept_validated_commit=(
                    self._accept_validated_commit
                    if self.plan._prepared is not None
                    else None
                ),
            )
        except FiniteArtifactPublicationHeld as error:
            if self.session is not None and self.seal is not None:
                self._settle_publisher_failure(error)
            raise
        except BaseException as error:
            if (
                self.session is not None
                and self.seal is not None
                and self.session._dynamic_external_publication_pending
            ):
                self._settle_publisher_failure(error)
            raise
        if self.session is not None and self.seal is not None:
            self._settle_publication(publication)
        self._report("publish", 1, 1)
        if publication.disposition in {
            FiniteArtifactDisposition.COMMITTED,
            FiniteArtifactDisposition.ALREADY_COMMITTED,
        }:
            committed = self.observed_committed_labels
            audit_identity = self.observed_audit_identity
            if committed is None or audit_identity is None:
                raise RuntimeError(
                    "committed publication lost its validated observation"
                )
            dropped = tuple(
                label for label in self.plan.labels if label not in set(committed)
            )
        else:
            committed = ()
            dropped = self.dropped_labels
            audit_identity = None
        diagnostics = tuple(
            _support._diagnostic(item)
            for item in (*self.diagnostics, *publication.diagnostics)
        )[:16]
        return _value(
            ReintegrateSuccessorResult,
            publication.disposition.value,
            request.source_artifact,
            request.output_artifact,
            self.plan.labels,
            committed,
            dropped,
            diagnostics,
            self.plan.science_identity,
            self.plan.version_identity,
            self.plan.publication_identity,
            self.plan.operation_identity,
            audit_identity,
            publication.commit_identity,
            publication.terminal,
            publication.hidden_orphan,
        )


def run_reintegrate_successor(
    plan: ReintegrateSuccessorPlan,
    *,
    cancel_token: threading.Event | None = None,
    progress_cb: Callable[[ReintegrateSuccessorProgress], object] | None = None,
) -> ReintegrateSuccessorResult:
    if type(plan) is not ReintegrateSuccessorPlan:
        raise TypeError("immutable reintegration requires an exact v4 plan")
    _support._event(cancel_token, honor=False)
    runtime = _SuccessorRuntime(plan, cancel_token, progress_cb)
    try:
        return runtime.run()
    finally:
        if not (
            runtime.session is not None
            and runtime.session._dynamic_external_publication_pending
        ):
            runtime.close()


__all__ = [
    "ReintegrateRecipeMigrationRequired",
    "ReintegrateSuccessorPlan",
    "ReintegrateSuccessorProgress",
    "ReintegrateSuccessorResult",
    "run_reintegrate_successor",
]
