"""Bounded immutable preparation for the Reintegration successor.

The bundle is detached JSON-shaped evidence: it owns no HDF5 handle, ndarray,
writer authority, or candidate authority.  Selection authenticates one fixed
dimension and carries only that admission plus the sibling commitment.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Any, Literal, Mapping

import h5py

from xrd_tools.io.append import (
    _json as _canonical_append_json,
    _lineage_labels,
    _replacement_hard_group,
    decode_committed_append_prefix,
    decode_replacement_lineage,
)
from xrd_tools.io.bounded_json import (
    BoundedJsonError,
    bounded_json_snapshot,
    bounded_utf8_size,
)
from xrd_tools.io.finite_artifact import (
    FiniteFileSnapshot,
    FiniteSourceAdmission,
    capture_finite_source,
)
from xrd_tools.io.read import relative_source_path
from xrd_tools.io.output_transaction import (
    StreamTerminal,
    TargetSnapshot,
    revalidate_stream_terminal,
)
from xrd_tools.io.record_writer import (
    ReplacementManifestReceipt,
    ReplacementManifestTargetChanged,
    WriterStateError,
    _decode_replacement_fact,
    _read_replacement_frame_index,
    _replacement_exclusion_paths,
    _require_replacement_manifest_source,
    _validate_replacement_execution,
    admit_replacement_manifest_receipt,
    prepare_replacement_manifest_receipt,
    replacement_manifest_receipt_mapping,
)
from xrd_tools.reduction import reintegrate as _legacy


_CAPSULE_FACTORY = object()
_BUNDLE_SCHEMA = "xrd_tools.reintegrate.prepared_bundle"
_EXECUTION_SCHEMA = "xrd_tools.reintegrate.prepared_execution"
_VERSION = 1
_SHA_CHARS = frozenset("0123456789abcdef")

MAX_PREPARED_LABELS = 4096
MAX_PREPARED_TEXT_BYTES = 4096
MAX_PREPARED_TOPOLOGY_BYTES = 8 << 20
MAX_PREPARED_FACT_BYTES = 8192
MAX_PREPARED_FACTS_BYTES = 32 << 20
MAX_PREPARED_DIMENSION_BYTES = 1 << 20
MAX_PREPARED_BUNDLE_BYTES = 48 << 20
MAX_PREPARED_EXECUTION_BYTES = 48 << 20
MAX_PREPARED_SOURCE_EXECUTION_BYTES = 4 << 20
MAX_PREPARED_APPEND_LINEAGE_BYTES = 4 << 20
MAX_PREPARED_UNTRUSTED_STRING_BYTES = 8 << 20
MAX_PREPARED_KEY_BYTES = 8 << 20
MAX_PREPARED_DEPTH = 24
MAX_PREPARED_NODES = 262_144
MAX_PREPARED_CONTAINER_CHILDREN = 16_384


class LegacyRouteReason(str, Enum):
    DIRECT_FROM_ARTIFACT = "DIRECT_FROM_ARTIFACT"
    CAPSULE_MISS = "CAPSULE_MISS"


class PreparedCapsuleMissCode(str, Enum):
    CAPSULE_NOT_SUPPLIED = "CAPSULE_NOT_SUPPLIED"
    CAPSULE_SCHEMA_UNSUPPORTED = "CAPSULE_SCHEMA_UNSUPPORTED"
    CAPSULE_ALGORITHM_UNSUPPORTED = "CAPSULE_ALGORITHM_UNSUPPORTED"
    CAPSULE_BYTE_LIMIT = "CAPSULE_BYTE_LIMIT"
    CAPSULE_CARDINALITY_LIMIT = "CAPSULE_CARDINALITY_LIMIT"
    CAPSULE_DIGEST_MISMATCH = "CAPSULE_DIGEST_MISMATCH"
    TARGET_SCHEMA_UNSUPPORTED = "TARGET_SCHEMA_UNSUPPORTED"
    TARGET_CHANGED = "TARGET_CHANGED"
    TERMINAL_CHANGED = "TERMINAL_CHANGED"
    ENTRY_CHANGED = "ENTRY_CHANGED"
    DIMENSION_UNAVAILABLE = "DIMENSION_UNAVAILABLE"
    DIMENSION_LABELS_INCOMPATIBLE = "DIMENSION_LABELS_INCOMPATIBLE"
    LABELS_CHANGED = "LABELS_CHANGED"
    SOURCE_ROOT_CHANGED = "SOURCE_ROOT_CHANGED"
    SOURCE_TOPOLOGY_UNSUPPORTED = "SOURCE_TOPOLOGY_UNSUPPORTED"
    SOURCE_TOPOLOGY_CHANGED = "SOURCE_TOPOLOGY_CHANGED"
    SOURCE_REVISION_CHANGED = "SOURCE_REVISION_CHANGED"
    APPEND_LINEAGE_UNSUPPORTED = "APPEND_LINEAGE_UNSUPPORTED"
    ARTIFACT_FACTS_UNSUPPORTED = "ARTIFACT_FACTS_UNSUPPORTED"
    ARTIFACT_FACTS_CHANGED = "ARTIFACT_FACTS_CHANGED"
    MANIFEST_DOMAIN_UNSUPPORTED = "MANIFEST_DOMAIN_UNSUPPORTED"
    REQUESTED_SCIENCE_UNSUPPORTED = "REQUESTED_SCIENCE_UNSUPPORTED"
    REQUEST_REQUIRES_UNPREPARED_FACTS = "REQUEST_REQUIRES_UNPREPARED_FACTS"
    MANIFEST_RECEIPT_CHANGED = "MANIFEST_RECEIPT_CHANGED"


class PreparedCapsuleMiss(RuntimeError):
    def __init__(self, code: PreparedCapsuleMissCode):
        if type(code) is not PreparedCapsuleMissCode:
            raise TypeError("prepared capsule miss requires an exact code")
        self.code = code
        super().__init__(code.value)


class PreparedRouteRejected(RuntimeError):
    def __init__(self, code: PreparedCapsuleMissCode):
        if type(code) is not PreparedCapsuleMissCode:
            raise TypeError("prepared route rejection requires an exact code")
        self.code = code
        super().__init__(code.value)


class PreparedRouteChanged(RuntimeError):
    """A sealed prepared route drifted before transaction ownership."""

    def __init__(self, code: PreparedCapsuleMissCode):
        if type(code) is not PreparedCapsuleMissCode:
            raise TypeError("prepared route change requires an exact code")
        self.code = code
        super().__init__(code.value)


def _value(cls, *values):
    instance = object.__new__(cls)
    for field, value in zip(fields(cls), values):
        object.__setattr__(instance, field.name, value)
    return instance


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _legacy._plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_digest(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA_CHARS


def _freeze_json(value: object):
    return _legacy._freeze(json.loads(_canonical_bytes(value)))


def _exact_mapping(value: object, keys: set[str], role: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    return value


def _bounded_admission_snapshot(
    value: object,
    *,
    role: str,
    max_encoded_bytes: int,
) -> tuple[object, int]:
    try:
        return bounded_json_snapshot(
            value,
            role=role,
            max_encoded_bytes=max_encoded_bytes,
            max_key_bytes=MAX_PREPARED_KEY_BYTES,
            max_string_bytes=MAX_PREPARED_UNTRUSTED_STRING_BYTES,
            max_depth=MAX_PREPARED_DEPTH,
            max_nodes=MAX_PREPARED_NODES,
            max_children=MAX_PREPARED_CONTAINER_CHILDREN,
        )
    except BoundedJsonError as error:
        code = {
            "bytes": PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT,
            "cardinality": PreparedCapsuleMissCode.CAPSULE_CARDINALITY_LIMIT,
        }.get(error.reason, PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED)
        raise PreparedRouteRejected(code) from error


def _admitted_text(value: object, role: str) -> str:
    try:
        return _bounded_text(value, role)
    except PreparedCapsuleMiss as error:
        raise PreparedRouteRejected(error.code) from error


_PATH_VALUE_KEYS = {
    "dataset",
    "dataset_path",
    "external_parent",
    "lexical_filename",
    "lexical_path",
    "local_object",
    "local_path",
    "member_path",
    "path",
    "remote_path",
    "resolved_path",
    "resolved_target",
    "snapshot_dataset_path",
    "source_base",
    "source_path",
    "target",
}
_PATH_LIST_KEYS = {"dataset_paths", "external_paths", "revision_paths"}


def _admit_named_paths(value: object, role: str) -> None:
    """Apply the persisted 4 KiB ceiling to schema-known nested paths."""

    stack = [value]
    while stack:
        current = stack.pop()
        if type(current) is list:
            stack.extend(current)
            continue
        if type(current) is not dict:
            continue
        for key, child in current.items():
            if key in _PATH_VALUE_KEYS and child is not None:
                _admitted_text(child, f"{role} {key}")
            elif key in _PATH_LIST_KEYS:
                if type(child) is not list:
                    raise PreparedRouteRejected(
                        PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
                    )
                for path in child:
                    _admitted_text(path, f"{role} {key}")
            stack.append(child)


def _bounded_text(value: object, role: str) -> str:
    try:
        bounded_utf8_size(
            value,
            role=role,
            max_bytes=MAX_PREPARED_TEXT_BYTES,
        )
    except BoundedJsonError as error:
        code = (
            PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT
            if error.reason == "bytes"
            else PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
        raise PreparedCapsuleMiss(code) from error
    return value


def _snapshot_mapping(value: FiniteFileSnapshot) -> dict[str, object]:
    return {name: getattr(value, name) for name in value.__dataclass_fields__}


def _terminal_mapping(value: StreamTerminal | None):
    if value is None:
        return None
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _target_snapshot(value: FiniteFileSnapshot) -> TargetSnapshot:
    return TargetSnapshot(
        True,
        value.size,
        value.mtime_ns,
        value.device,
        value.inode,
        value.digest,
    )


@dataclass(frozen=True, slots=True, init=False)
class PreparedTargetReceipt:
    snapshot: FiniteFileSnapshot
    terminal: StreamTerminal | None
    target_digest: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("PreparedTargetReceipt is factory-constructed")


@dataclass(frozen=True, slots=True, init=False)
class PreparedArtifactFactsReceipt:
    entry: str
    detector_shape: tuple[int, int]
    native_dtype: str
    persisted_shared_science: Mapping[str, Any]
    acquisition_fingerprint: str
    source_base: str
    append_lineage_digest: str | None
    artifact_digest: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("PreparedArtifactFactsReceipt is factory-constructed")


@dataclass(frozen=True, slots=True, init=False)
class PreparedSourceTopologyReceipt:
    payload: Mapping[str, Any]
    topology_digest: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("PreparedSourceTopologyReceipt is factory-constructed")


@dataclass(frozen=True, slots=True, init=False)
class PreparedDimensionPayload:
    dimension: Literal["1d", "2d"]
    dimension_labels: tuple[int, ...]
    selected_plan: Mapping[str, Any]
    selected_facts_digest: str
    manifest_receipt: ReplacementManifestReceipt
    dimension_payload_digest: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("PreparedDimensionPayload is factory-constructed")


@dataclass(frozen=True, slots=True, init=False)
class PreparedDimensionAdmission:
    dimension: Literal["1d", "2d"]
    disposition: Literal["READY", "MISS"]
    miss_code: PreparedCapsuleMissCode | None
    payload: PreparedDimensionPayload | None
    admission_digest: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("PreparedDimensionAdmission is factory-constructed")


@dataclass(frozen=True, slots=True, init=False)
class AdmissionCommitment:
    dimension: Literal["1d", "2d"]
    admission_digest: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("AdmissionCommitment is factory-constructed")


@dataclass(frozen=True, slots=True, init=False)
class PreparedReintegrateBundle:
    schema: str
    version: int
    target: PreparedTargetReceipt
    artifact: PreparedArtifactFactsReceipt
    topology: PreparedSourceTopologyReceipt
    labels: tuple[int, ...]
    facts: tuple[Mapping[str, Any], ...]
    one_d: PreparedDimensionAdmission
    two_d: PreparedDimensionAdmission
    facts_digest: str
    bundle_digest: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("PreparedReintegrateBundle is factory-constructed")


@dataclass(frozen=True, slots=True, init=False)
class PreparedReintegrateOffer:
    disposition: Literal["READY", "MISS"]
    bundle: PreparedReintegrateBundle | None
    miss_code: PreparedCapsuleMissCode | None

    def __new__(cls, *args, **kwargs):
        raise TypeError("PreparedReintegrateOffer is factory-constructed")


@dataclass(frozen=True, slots=True, init=False)
class PreparedReintegrateExecution:
    schema: str
    version: int
    target: PreparedTargetReceipt
    artifact: PreparedArtifactFactsReceipt
    topology: PreparedSourceTopologyReceipt
    labels: tuple[int, ...]
    facts: tuple[Mapping[str, Any], ...]
    facts_digest: str
    selected_admission: PreparedDimensionAdmission
    sibling_commitment: AdmissionCommitment
    bundle_digest: str
    execution_digest: str
    _inspection: Any
    _facts_by_label: Mapping[int, Mapping[str, Any]]

    def __new__(cls, *args, **kwargs):
        raise TypeError("PreparedReintegrateExecution is factory-constructed")


def _target_receipt(
    snapshot: FiniteFileSnapshot,
    terminal: StreamTerminal | None,
) -> PreparedTargetReceipt:
    payload = {
        "schema": "xrd_tools.reintegrate.prepared_target",
        "version": 1,
        "snapshot": _snapshot_mapping(snapshot),
        "terminal": _terminal_mapping(terminal),
    }
    return _value(PreparedTargetReceipt, snapshot, terminal, _digest(payload))


def _target_receipt_mapping(value: PreparedTargetReceipt) -> dict[str, object]:
    return {
        "schema": "xrd_tools.reintegrate.prepared_target",
        "version": 1,
        "snapshot": _snapshot_mapping(value.snapshot),
        "terminal": _terminal_mapping(value.terminal),
        "target_digest": value.target_digest,
    }


def _artifact_mapping(value: PreparedArtifactFactsReceipt) -> dict[str, object]:
    return {
        "schema": "xrd_tools.reintegrate.prepared_artifact",
        "version": 1,
        "entry": value.entry,
        "detector_shape": list(value.detector_shape),
        "native_dtype": value.native_dtype,
        "persisted_shared_science": _legacy._plain(
            value.persisted_shared_science
        ),
        "acquisition_fingerprint": value.acquisition_fingerprint,
        "source_base": value.source_base,
        "append_lineage": value.append_lineage_digest,
        "raw_options": None,
        "retained_mask_bytes": 0,
        "mask_decode_bytes": 0,
        "gi_values": [],
        "artifact_digest": value.artifact_digest,
    }


def _artifact_receipt(
    inspection: Any,
    entry: str,
) -> PreparedArtifactFactsReceipt:
    values = (
        entry,
        tuple(inspection.detector_shape),
        inspection.native_dtype,
        _freeze_json(inspection.persisted_shared_science),
        inspection.acquisition_fingerprint,
        inspection.source_base,
        (
            None
            if inspection.append_lineage is None
            else hashlib.sha256(inspection.append_lineage).hexdigest()
        ),
    )
    provisional = _value(PreparedArtifactFactsReceipt, *values, "0" * 64)
    payload = _artifact_mapping(provisional)
    payload.pop("artifact_digest")
    return _value(PreparedArtifactFactsReceipt, *values, _digest(payload))


def _topology_mapping(topology: Any) -> dict[str, object]:
    def frame_route(label, route):
        hdf = None
        if route.hdf is not None:
            hdf = {
                "dataset_path": route.hdf.dataset_path,
                "start": route.hdf.start,
                "stop": route.hdf.stop,
                "member_path": route.hdf.member_path,
                "storage_slices": [
                    {
                        "lexical_path": item.lexical_path,
                        "resolved_path": item.resolved_path,
                        "offset": item.offset,
                        "size": item.size,
                        "frame_offset": item.frame_offset,
                    }
                    for item in route.hdf.storage_slices
                ],
            }
        return {
            "label": label,
            "source_path": route.source_path,
            "source_state": _legacy._plain(route.source_state),
            "snapshot_count": route.snapshot_count,
            "snapshot_dataset_path": route.snapshot_dataset_path,
            "self_contained": route.self_contained,
            "revision_paths": list(route.revision_paths),
            "hdf": hdf,
        }

    return {
        "schema": "xrd_tools.reintegrate.prepared_topology",
        "version": 1,
        "source_base": topology.source_base,
        "execution": _legacy._plain(topology.execution),
        "lineage": _legacy._plain(topology.lineage),
        "final_source": _legacy._plain(topology.final_source),
        "execution_digest": topology.execution_digest,
        "lineage_digest": topology.lineage_digest,
        "revision_signature": _legacy._plain(topology.revision_signature),
        "revisions": [
            {"path": key, "value": _legacy._plain(value)}
            for key, value in sorted(topology.revisions.items())
        ],
        "frame_routes": [
            frame_route(label, topology.frame_routes[label])
            for label in sorted(topology.frame_routes)
        ],
        "external_parent": topology.external_parent,
        "external_paths": list(topology.external_paths),
        "external_signature": [
            {
                name: getattr(item, name)
                for name in item._fields
            }
            for item in topology.external_signature
        ],
    }


def _topology_receipt(topology: Any) -> PreparedSourceTopologyReceipt:
    payload = _topology_mapping(topology)
    if (
        len(_canonical_bytes(payload["execution"]))
        > MAX_PREPARED_SOURCE_EXECUTION_BYTES
        or payload["lineage"] is not None
        and len(_canonical_bytes(payload["lineage"]))
        > MAX_PREPARED_APPEND_LINEAGE_BYTES
    ):
        raise PreparedCapsuleMiss(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    encoded = _canonical_bytes(payload)
    if len(encoded) > MAX_PREPARED_TOPOLOGY_BYTES:
        raise PreparedCapsuleMiss(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    return _value(
        PreparedSourceTopologyReceipt,
        _freeze_json(payload),
        hashlib.sha256(encoded).hexdigest(),
    )


def source_topology_identity(topology: Any) -> str:
    """Return the canonical complete raw-dependency topology identity."""

    return _digest(_topology_mapping(topology))


def _compact_fact(fact: Mapping[str, Any]) -> Mapping[str, Any]:
    value = {
        "label": fact["label"],
        "path": fact["path"],
        "frame_index": fact["frame_index"],
        "snapshot": _legacy._plain(fact["snapshot"]),
        "metadata": _legacy._plain(fact["metadata"]),
        "geometry": _legacy._plain(fact["geometry"]),
        "background_dependency": _legacy._plain(fact["background_dependency"]),
    }
    encoded = _canonical_bytes(value)
    if len(encoded) > MAX_PREPARED_FACT_BYTES:
        raise PreparedCapsuleMiss(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    return _freeze_json(value)


def _facts_digest_payload(
    target: PreparedTargetReceipt,
    artifact: PreparedArtifactFactsReceipt,
    topology: PreparedSourceTopologyReceipt,
    labels: tuple[int, ...],
    facts: tuple[Mapping[str, Any], ...],
) -> tuple[str, str]:
    per_frame = _digest([_legacy._plain(value) for value in facts])
    value = {
        "schema": "xrd_tools.reintegrate.prepared_facts",
        "version": 1,
        "target_digest": target.target_digest,
        "artifact_digest": artifact.artifact_digest,
        "topology_digest": topology.topology_digest,
        "per_frame_facts_digest": per_frame,
        "labels": list(labels),
        "fact_count": len(facts),
    }
    return per_frame, _digest(value)


def _payload_mapping(value: PreparedDimensionPayload) -> dict[str, object]:
    return {
        "schema": "xrd_tools.reintegrate.prepared_dimension_payload",
        "version": 1,
        "dimension": value.dimension,
        "dimension_labels": list(value.dimension_labels),
        "selected_plan": _legacy._plain(value.selected_plan),
        "selected_facts_digest": value.selected_facts_digest,
        "manifest_receipt": replacement_manifest_receipt_mapping(
            value.manifest_receipt
        ),
        "dimension_payload_digest": value.dimension_payload_digest,
    }


def _dimension_payload(
    dimension: Literal["1d", "2d"],
    labels: tuple[int, ...],
    selected_plan: Mapping[str, Any],
    receipt: ReplacementManifestReceipt,
) -> PreparedDimensionPayload:
    selected = _freeze_json(selected_plan)
    selected_digest = _digest({
        "dimension": dimension,
        "labels": list(labels),
        "selected_plan": _legacy._plain(selected),
    })
    values = (dimension, labels, selected, selected_digest, receipt)
    provisional = _value(PreparedDimensionPayload, *values, "0" * 64)
    mapping = _payload_mapping(provisional)
    mapping.pop("dimension_payload_digest")
    result = _value(PreparedDimensionPayload, *values, _digest(mapping))
    if len(_canonical_bytes(_payload_mapping(result))) > MAX_PREPARED_DIMENSION_BYTES:
        raise PreparedCapsuleMiss(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    return result


def _admission_preimage(
    dimension: str,
    disposition: str,
    miss_code: PreparedCapsuleMissCode | None,
    payload_digest: str | None,
) -> dict[str, object]:
    return {
        "schema": "xrd_tools.reintegrate.prepared_dimension_admission",
        "version": 1,
        "dimension": dimension,
        "disposition": disposition,
        "miss_code": None if miss_code is None else miss_code.value,
        "dimension_payload_digest": payload_digest,
    }


def _ready_admission(
    payload: PreparedDimensionPayload,
) -> PreparedDimensionAdmission:
    digest = _digest(_admission_preimage(
        payload.dimension, "READY", None, payload.dimension_payload_digest,
    ))
    admission = _value(
        PreparedDimensionAdmission,
        payload.dimension,
        "READY",
        None,
        payload,
        digest,
    )
    if (
        len(_canonical_bytes(_admission_mapping(admission)))
        > MAX_PREPARED_DIMENSION_BYTES
    ):
        raise PreparedCapsuleMiss(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    return admission


def _miss_admission(
    dimension: Literal["1d", "2d"],
    code: PreparedCapsuleMissCode,
) -> PreparedDimensionAdmission:
    digest = _digest(_admission_preimage(dimension, "MISS", code, None))
    admission = _value(
        PreparedDimensionAdmission, dimension, "MISS", code, None, digest,
    )
    if (
        len(_canonical_bytes(_admission_mapping(admission)))
        > MAX_PREPARED_DIMENSION_BYTES
    ):
        raise PreparedCapsuleMiss(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    return admission


def _admission_mapping(value: PreparedDimensionAdmission) -> dict[str, object]:
    return {
        **_admission_preimage(
            value.dimension,
            value.disposition,
            value.miss_code,
            None if value.payload is None else value.payload.dimension_payload_digest,
        ),
        "payload": None if value.payload is None else _payload_mapping(value.payload),
        "admission_digest": value.admission_digest,
    }


def _bundle_root(
    labels: tuple[int, ...],
    facts_digest: str,
    one_d_digest: str,
    two_d_digest: str,
) -> str:
    return _digest({
        "schema": _BUNDLE_SCHEMA,
        "version": _VERSION,
        "labels": list(labels),
        "facts_digest": facts_digest,
        "one_d_admission_digest": one_d_digest,
        "two_d_admission_digest": two_d_digest,
    })


def prepared_bundle_mapping(
    value: PreparedReintegrateBundle,
) -> dict[str, object]:
    if type(value) is not PreparedReintegrateBundle:
        raise TypeError("prepared bundle projection requires an exact bundle")
    return {
        "schema": value.schema,
        "version": value.version,
        "target": _target_receipt_mapping(value.target),
        "artifact": _artifact_mapping(value.artifact),
        "topology": {
            "payload": _legacy._plain(value.topology.payload),
            "topology_digest": value.topology.topology_digest,
        },
        "labels": list(value.labels),
        "facts": [_legacy._plain(fact) for fact in value.facts],
        "one_d": _admission_mapping(value.one_d),
        "two_d": _admission_mapping(value.two_d),
        "facts_digest": value.facts_digest,
        "bundle_digest": value.bundle_digest,
    }


def prepared_execution_mapping(
    value: PreparedReintegrateExecution,
) -> dict[str, object]:
    if type(value) is not PreparedReintegrateExecution:
        raise TypeError("prepared execution projection requires an exact execution")
    return {
        "schema": value.schema,
        "version": value.version,
        "target": _target_receipt_mapping(value.target),
        "artifact": _artifact_mapping(value.artifact),
        "topology": {
            "payload": _legacy._plain(value.topology.payload),
            "topology_digest": value.topology.topology_digest,
        },
        "labels": list(value.labels),
        "facts": [_legacy._plain(fact) for fact in value.facts],
        "facts_digest": value.facts_digest,
        "selected_admission": _admission_mapping(value.selected_admission),
        "sibling_commitment": {
            "dimension": value.sibling_commitment.dimension,
            "admission_digest": value.sibling_commitment.admission_digest,
        },
        "bundle_digest": value.bundle_digest,
        "execution_digest": value.execution_digest,
    }


def _admit_target(value: object) -> PreparedTargetReceipt:
    mapping = _exact_mapping(
        value,
        {"schema", "version", "snapshot", "terminal", "target_digest"},
        "prepared target",
    )
    if (
        mapping["schema"] != "xrd_tools.reintegrate.prepared_target"
        or type(mapping["version"]) is not int
        or mapping["version"] != 1
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_ALGORITHM_UNSUPPORTED
        )
    snapshot_value = _exact_mapping(
        mapping["snapshot"],
        set(FiniteFileSnapshot.__dataclass_fields__),
        "prepared target snapshot",
    )
    _admitted_text(snapshot_value.get("path"), "target snapshot path")
    try:
        snapshot = FiniteFileSnapshot(**snapshot_value)
    except TypeError as error:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        ) from error
    normalized_snapshot_path = os.path.normcase(os.path.normpath(
        os.path.abspath(snapshot.path)
    ))
    if (
        not os.path.isabs(snapshot.path)
        or normalized_snapshot_path != snapshot.path
        or not stat.S_ISREG(snapshot.mode)
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    terminal = None
    if mapping["terminal"] is not None:
        terminal_value = _exact_mapping(
            mapping["terminal"],
            {field.name for field in fields(StreamTerminal)},
            "prepared target terminal",
        )
        try:
            _admitted_text(
                terminal_value.get("target"), "target terminal path",
            )
            terminal = StreamTerminal(**terminal_value)
        except (TypeError, ValueError) as error:
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            ) from error
        if (
            terminal.target != snapshot.path
            or (
                terminal.size,
                terminal.device,
                terminal.inode,
                terminal.mtime_ns,
                terminal.ctime_ns,
            ) != (
                snapshot.size,
                snapshot.device,
                snapshot.inode,
                snapshot.mtime_ns,
                snapshot.ctime_ns,
            )
        ):
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
            )
    preimage = dict(mapping)
    digest = preimage.pop("target_digest")
    if not _is_digest(digest) or _digest(preimage) != digest:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    return _value(PreparedTargetReceipt, snapshot, terminal, digest)


def _admit_artifact(value: object) -> PreparedArtifactFactsReceipt:
    keys = {
        "schema", "version", "entry", "detector_shape", "native_dtype",
        "persisted_shared_science", "acquisition_fingerprint", "source_base",
        "append_lineage", "raw_options", "retained_mask_bytes",
        "mask_decode_bytes", "gi_values", "artifact_digest",
    }
    mapping = _exact_mapping(value, keys, "prepared artifact")
    if (
        mapping["schema"] != "xrd_tools.reintegrate.prepared_artifact"
        or type(mapping["version"]) is not int
        or mapping["version"] != 1
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_ALGORITHM_UNSUPPORTED
        )
    shape = mapping["detector_shape"]
    if (
        type(shape) is not list
        or len(shape) != 2
        or any(type(item) is not int or item <= 0 for item in shape)
        or mapping["append_lineage"] is not None
        and not _is_digest(mapping["append_lineage"])
        or mapping["raw_options"] is not None
        or type(mapping["retained_mask_bytes"]) is not int
        or mapping["retained_mask_bytes"] != 0
        or type(mapping["mask_decode_bytes"]) is not int
        or mapping["mask_decode_bytes"] != 0
        or mapping["gi_values"] != []
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    preimage = dict(mapping)
    digest = preimage.pop("artifact_digest")
    if not _is_digest(digest) or _digest(preimage) != digest:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    try:
        shared = _freeze_json(mapping["persisted_shared_science"])
        entry = _bounded_text(mapping["entry"], "entry")
        source_base = _bounded_text(mapping["source_base"], "source root")
        native = _bounded_text(mapping["native_dtype"], "native dtype")
    except PreparedCapsuleMiss as error:
        raise PreparedRouteRejected(error.code) from error
    if (
        not os.path.isabs(source_base)
        or os.path.normcase(os.path.normpath(source_base)) != source_base
        or not _is_digest(mapping["acquisition_fingerprint"])
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    return _value(
        PreparedArtifactFactsReceipt,
        entry,
        tuple(shape),
        native,
        shared,
        mapping["acquisition_fingerprint"],
        source_base,
        mapping["append_lineage"],
        digest,
    )


def _require_v1_restored_topology(topology: Any) -> None:
    """Join every detached v1 route to its exact admitted HDF authority."""

    def reject(condition: bool) -> None:
        if condition:
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            )

    def normalized(path: object) -> str:
        reject(type(path) is not str or not os.path.isabs(path))
        value = os.path.normcase(os.path.normpath(os.path.abspath(path)))
        reject(value != path)
        return value

    def resolved_join(parent: Path, path: object) -> str:
        reject(type(path) is not str or not path)
        return os.path.normcase(
            os.path.normpath(os.path.abspath(parent / path))
        )

    def posix_path(path: object) -> PurePosixPath:
        reject(type(path) is not str)
        value = PurePosixPath(path)
        reject(not value.is_absolute() or str(value) != path)
        return value

    execution = topology.execution
    reject(
        execution["adapter_id"] != "nexus_hdf5"
        or execution["frame_count"] < 1
        or execution["member_stamps"]
        or execution["dependency_files"]
        or execution["admitted_motor_values"]
        or execution["metadata_sources"]
        or len(execution["external_members"]) != 1
        or not topology.frame_routes
    )
    source_path = normalized(execution["path"])
    member = execution["external_members"][0]
    member_state = member["file"]
    member_path = normalized(member_state["path"])
    reject(
        member_path == source_path
        or member["first"] != 0
        or member["stop"] != execution["frame_count"]
        or member["epoch"] != 0
        or type(member["dataset"]) is not str
        or not posix_path(member["dataset"]).name
    )

    lineage = topology.lineage
    final = topology.final_source
    final_member = None
    if lineage is None:
        reject(final is not None or topology.lineage_digest is not None)
        labels = tuple(topology.frame_routes)
        total = execution["frame_count"]
        source_state = execution
        route_dataset = member["dataset"]
        route_start, route_stop = member["first"], member["stop"]
    else:
        expected_lineage_keys = {
            "version", "state", "entry", "source_base",
            "source_identity", "science_fingerprint", "modes", "epochs",
        }
        reject(
            set(lineage) != expected_lineage_keys
            or type(lineage["version"]) is not int
            or lineage["version"] != 1
            or lineage["state"] != "committed"
            or any(
                type(lineage[key]) is not str or not lineage[key]
                for key in ("entry", "source_identity", "science_fingerprint")
            )
            or lineage["source_base"] != topology.source_base
            or type(lineage["modes"]) is not tuple
            or not lineage["modes"]
            or any(type(mode) is not str or not mode for mode in lineage["modes"])
            or len(set(lineage["modes"])) != len(lineage["modes"])
            or type(lineage["epochs"]) is not tuple
            or len(lineage["epochs"]) != 1
        )
        try:
            labels = _lineage_labels(_legacy._plain(lineage))
        except (TypeError, ValueError, KeyError) as error:
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            ) from error
        reject(labels != tuple(topology.frame_routes))
        final = lineage["epochs"][0]["source"]
        reject(
            _canonical_bytes(topology.final_source)
            != _canonical_bytes(final)
        )
        reject(
            final["adapter_id"] != "nexus_hdf5"
            or normalized(final["path"]) != source_path
            or final["size"] < execution["size"]
            or final["mtime_ns"] < execution["mtime_ns"]
            or final["extent"] < execution["frame_count"]
            or final["image_members"]
            or len(final["external_members"]) != 1
        )
        total = final["extent"]
        final_member = final["external_members"][0]
        reject(
            normalized(final_member["path"]) != member_path
            or final_member["dataset_path"] != member["dataset"]
            or final_member["size"] < member_state["size"]
            or final_member["mtime_ns"] < member_state["mtime_ns"]
            or final_member["source_start"] != 0
            or final_member["source_stop"] != total
            or final_member["ordinal"] != 0
        )
        source_state = final
        route_dataset = final_member["dataset_path"]
        route_start = final_member["source_start"]
        route_stop = final_member["source_stop"]

    reject(
        not os.path.isabs(topology.source_base)
        or os.path.normcase(os.path.normpath(topology.source_base))
        != topology.source_base
        or topology.execution_digest != _legacy._digest(execution)
        or topology.lineage_digest
        != (None if lineage is None else _legacy._digest(lineage))
    )

    reject(len(topology.external_signature) != 1)
    link = topology.external_signature[0]
    local = posix_path(link.local_path)
    parent = str(local.parent)
    reject(
        link.kind != "external"
        or link.name != local.name
        or link.local_object is not None
        or link.remote_path != route_dataset
        or link.resolved_target is None
        or type(link.lexical_filename) is not str
        or not link.lexical_filename
        or resolved_join(Path(source_path).parent, link.lexical_filename)
        != member_path
        or topology.external_parent != parent
        or topology.external_paths != (link.local_path,)
        or final is not None
        and tuple(final["dataset_paths"]) not in {(), topology.external_paths}
    )

    expected_revisions = {
        source_path: (
            source_state, "source_file" if final is None else "final_source",
        ),
        member_path: (
            member_state if final is None else final_member,
            "external_member" if final is None else "final_external",
        ),
    }
    reject(set(topology.revisions) != set(expected_revisions))
    for raw_path, (state, role) in expected_revisions.items():
        revision = topology.revisions[raw_path]
        reject(type(revision) is not tuple or len(revision) != 4)
        resolved, observed_state, observed_role, observed = revision
        reject(
            type(resolved) is not str
            or not os.path.isabs(resolved)
            or os.path.normcase(os.path.normpath(resolved)) != resolved
            or _canonical_bytes(observed_state) != _canonical_bytes(state)
            or observed_role != role
            or type(observed) is not tuple
            or len(observed) != 6
            or observed[0] != resolved
            or any(type(value) is not int or value < 0 for value in observed[1:])
            or observed[1:3] != (state["size"], state["mtime_ns"])
            or final is None
            and observed[3:]
            != tuple(
                state[key] for key in ("ctime_ns", "device", "inode")
            )
        )
    try:
        signature = _legacy._revision_signature(topology.revisions)
    except (TypeError, ValueError, KeyError, IndexError) as error:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        ) from error
    reject(
        _canonical_bytes(topology.revision_signature)
        != _canonical_bytes(signature)
    )
    member_resolved = topology.revisions[member_path][0]
    reject(normalized(link.resolved_target) != normalized(member_resolved))

    revision_paths = _legacy._ordered_revision_paths(
        source_path,
        member_path,
        None if final_member is None else final_member["path"],
        member_resolved,
    )
    expected_hdf = _legacy._HdfFrameRoute(
        route_dataset,
        route_start,
        route_stop,
        member_resolved,
        (),
    )
    for label, route in topology.frame_routes.items():
        ordinal = label - execution["first_label"]
        reject(not 0 <= ordinal < total)
        expected = _legacy._FrameRoute(
            source_path,
            source_state,
            total,
            link.local_path,
            False,
            revision_paths,
            expected_hdf,
        )
        reject(_canonical_bytes(route) != _canonical_bytes(expected))


def _restore_topology(value: Mapping[str, Any]):
    plain = _legacy._plain(value)
    expected = {
        "schema", "version", "source_base", "execution", "lineage",
        "final_source", "execution_digest", "lineage_digest",
        "revision_signature", "revisions", "frame_routes",
        "external_parent", "external_paths", "external_signature",
    }
    mapping = _exact_mapping(plain, expected, "prepared topology payload")
    if (
        mapping["schema"] != "xrd_tools.reintegrate.prepared_topology"
        or type(mapping["version"]) is not int
        or mapping["version"] != 1
        or type(mapping["execution"]) is not dict
        or mapping["lineage"] is not None
        and type(mapping["lineage"]) is not dict
        or mapping["final_source"] is not None
        and type(mapping["final_source"]) is not dict
        or type(mapping["revisions"]) is not list
        or type(mapping["frame_routes"]) is not list
        or type(mapping["external_paths"]) is not list
        or type(mapping["external_signature"]) is not list
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    try:
        execution = _validate_replacement_execution(
            dict(mapping["execution"])
        )
    except (WriterStateError, TypeError, ValueError, KeyError) as error:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        ) from error
    lineage = mapping["lineage"]
    final_source = mapping["final_source"]
    _admit_named_paths(execution, "source execution")
    if lineage is not None:
        _admit_named_paths(lineage, "append lineage")
    if final_source is not None:
        _admit_named_paths(final_source, "final source")
    if (
        not _is_digest(mapping["execution_digest"])
        or _legacy._digest(execution) != mapping["execution_digest"]
        or (None if lineage is None else _legacy._digest(lineage))
        != mapping["lineage_digest"]
        or lineage is None
        and final_source is not None
        or lineage is not None
        and (
            type(lineage.get("epochs")) is not list
            or len(lineage["epochs"]) != 1
            or type(lineage["epochs"][0]) is not dict
            or lineage["epochs"][0].get("source") != final_source
        )
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    source_base = _admitted_text(mapping["source_base"], "source root")
    revisions = {}
    for item in mapping["revisions"]:
        row = _exact_mapping(item, {"path", "value"}, "prepared revision")
        revision_path = _admitted_text(row["path"], "revision path")
        if revision_path in revisions:
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            )
        revisions[revision_path] = _freeze_json(row["value"])
    routes = {}
    for item in mapping["frame_routes"]:
        row = _exact_mapping(
            item,
            {
                "label", "source_path", "source_state", "snapshot_count",
                "snapshot_dataset_path", "self_contained", "revision_paths",
                "hdf",
            },
            "prepared frame route",
        )
        label = row["label"]
        if (
            type(label) is not int
            or label < 0
            or label in routes
            or type(row["source_state"]) is not dict
            or type(row["snapshot_count"]) is not int
            or row["snapshot_count"] < 0
            or type(row["self_contained"]) is not bool
            or type(row["revision_paths"]) is not list
        ):
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            )
        source_path = _admitted_text(row["source_path"], "frame source path")
        snapshot_dataset_path = row["snapshot_dataset_path"]
        if snapshot_dataset_path is not None:
            snapshot_dataset_path = _admitted_text(
                snapshot_dataset_path, "frame snapshot dataset path",
            )
        revision_paths = tuple(
            _admitted_text(path, "frame revision path")
            for path in row["revision_paths"]
        )
        hdf = None
        if row["hdf"] is not None:
            hdf_value = _exact_mapping(
                row["hdf"],
                {"dataset_path", "start", "stop", "member_path", "storage_slices"},
                "prepared HDF route",
            )
            slices = []
            if (
                type(hdf_value["start"]) is not int
                or hdf_value["start"] < 0
                or type(hdf_value["stop"]) is not int
                or hdf_value["stop"] <= hdf_value["start"]
                or type(hdf_value["storage_slices"]) is not list
            ):
                raise PreparedRouteRejected(
                    PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
                )
            for selected in hdf_value["storage_slices"]:
                selected = _exact_mapping(
                    selected,
                    {
                        "lexical_path", "resolved_path", "offset", "size",
                        "frame_offset",
                    },
                    "prepared HDF storage slice",
                )
                selected = dict(selected)
                if any(
                    type(selected[field]) is not int
                    or selected[field] < 0
                    for field in ("offset", "size", "frame_offset")
                ):
                    raise PreparedRouteRejected(
                        PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
                    )
                selected["lexical_path"] = _admitted_text(
                    selected["lexical_path"], "HDF storage lexical path",
                )
                selected["resolved_path"] = _admitted_text(
                    selected["resolved_path"], "HDF storage resolved path",
                )
                slices.append(_legacy._HdfStorageSlice(**selected))
            try:
                hdf = _legacy._HdfFrameRoute(
                    _admitted_text(
                        hdf_value["dataset_path"], "HDF dataset path",
                    ),
                    hdf_value["start"],
                    hdf_value["stop"],
                    _admitted_text(
                        hdf_value["member_path"], "HDF member path",
                    ),
                    tuple(slices),
                )
            except (TypeError, ValueError) as error:
                raise PreparedRouteRejected(
                    PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
                ) from error
        try:
            routes[label] = _legacy._FrameRoute(
                source_path,
                _freeze_json(row["source_state"]),
                row["snapshot_count"],
                snapshot_dataset_path,
                row["self_contained"],
                revision_paths,
                hdf,
            )
        except (TypeError, ValueError) as error:
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            ) from error
    links = []
    for item in mapping["external_signature"]:
        admitted_link = dict(_exact_mapping(
            item,
            set(_legacy._HdfOwnerLink._fields),
            "prepared external link",
        ))
        for field in _legacy._HdfOwnerLink._fields:
            if admitted_link[field] is not None:
                admitted_link[field] = _admitted_text(
                    admitted_link[field], f"external link {field}",
                )
        links.append(_legacy._HdfOwnerLink(**admitted_link))
    external_parent = mapping["external_parent"]
    if external_parent is not None:
        external_parent = _admitted_text(
            external_parent, "external link parent",
        )
    external_paths = tuple(
        _admitted_text(path, "external HDF path")
        for path in mapping["external_paths"]
    )
    try:
        revision_lookup = _legacy._revision_lookup(revisions)
        topology = _legacy._SourceTopology(
            source_base,
            _freeze_json(execution),
            None if lineage is None else _freeze_json(lineage),
            None if final_source is None else _freeze_json(final_source),
            mapping["execution_digest"],
            mapping["lineage_digest"],
            _freeze_json(mapping["revision_signature"]),
            MappingProxyType(revisions),
            revision_lookup,
            MappingProxyType(routes),
            external_parent,
            external_paths,
            tuple(links),
        )
    except (TypeError, ValueError) as error:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        ) from error
    try:
        _require_v1_restored_topology(topology)
    except PreparedRouteRejected:
        raise
    except (AttributeError, TypeError, ValueError, KeyError, IndexError) as error:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        ) from error
    return topology


def _admit_topology(value: object) -> tuple[PreparedSourceTopologyReceipt, Any]:
    mapping = _exact_mapping(
        value, {"payload", "topology_digest"}, "prepared topology",
    )
    payload, _payload_bytes = _bounded_admission_snapshot(
        mapping["payload"],
        role="prepared topology",
        max_encoded_bytes=MAX_PREPARED_TOPOLOGY_BYTES,
    )
    if type(payload) is not dict or not {"execution", "lineage"} <= set(payload):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    execution, _execution_bytes = _bounded_admission_snapshot(
        payload["execution"],
        role="prepared source execution",
        max_encoded_bytes=MAX_PREPARED_SOURCE_EXECUTION_BYTES,
    )
    lineage = None
    if payload["lineage"] is not None:
        lineage, _lineage_bytes = _bounded_admission_snapshot(
            payload["lineage"],
            role="prepared append lineage",
            max_encoded_bytes=MAX_PREPARED_APPEND_LINEAGE_BYTES,
        )
    payload["execution"] = execution
    payload["lineage"] = lineage
    encoded = _canonical_bytes(payload)
    if (
        not _is_digest(mapping["topology_digest"])
        or hashlib.sha256(encoded).hexdigest() != mapping["topology_digest"]
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    frozen = _freeze_json(payload)
    topology = _restore_topology(frozen)
    return (
        _value(
            PreparedSourceTopologyReceipt,
            frozen,
            mapping["topology_digest"],
        ),
        topology,
    )


def _admit_facts(
    offered: object,
    labels: tuple[int, ...],
    topology: Any,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[int, Mapping[str, Any]]]:
    if type(offered) is not list or len(offered) != len(labels):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_CARDINALITY_LIMIT
        )
    facts = []
    indexed = {}
    total = 2 + max(0, len(offered) - 1)
    expected_keys = {
        "label", "path", "frame_index", "snapshot", "metadata", "geometry",
        "background_dependency",
    }
    for expected_label, item in zip(labels, offered):
        mapping = _exact_mapping(item, expected_keys, "prepared source fact")
        encoded = _canonical_bytes(mapping)
        total += len(encoded)
        if len(encoded) > MAX_PREPARED_FACT_BYTES:
            raise PreparedRouteRejected(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
        if (
            type(mapping["label"]) is not int
            or mapping["label"] != expected_label
            or type(mapping["frame_index"]) is not int
            or mapping["frame_index"] < 0
            or type(mapping["path"]) is not str
            or type(mapping["snapshot"]) is not dict
            or mapping["metadata"] != {}
            or mapping["geometry"] != {}
            or mapping["background_dependency"] is not None
        ):
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            )
        try:
            path = _bounded_text(mapping["path"], "source path")
        except PreparedCapsuleMiss as error:
            raise PreparedRouteRejected(error.code) from error
        if (
            not os.path.isabs(path)
            or os.path.normcase(os.path.normpath(path)) != path
        ):
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            )
        fact = MappingProxyType({
            "label": expected_label,
            "path": relative_source_path(path, topology.source_base),
            "frame_index": mapping["frame_index"],
            "snapshot": MappingProxyType(dict(mapping["snapshot"])),
            "source_base": topology.source_base,
            "source_execution": topology.execution,
            "append_lineage": topology.lineage,
            "metadata": MappingProxyType({}),
            "geometry": MappingProxyType({}),
            "background_dependency": None,
        })
        try:
            _legacy._require_fact_route(fact, topology, Path(path))
        except (WriterStateError, TypeError, ValueError, KeyError) as error:
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            ) from error
        compact = _freeze_json(mapping)
        facts.append(compact)
        indexed[expected_label] = fact
    if total > MAX_PREPARED_FACTS_BYTES:
        raise PreparedRouteRejected(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    return tuple(facts), MappingProxyType(indexed)


def _admit_payload(value: object) -> PreparedDimensionPayload:
    keys = {
        "schema", "version", "dimension", "dimension_labels",
        "selected_plan", "selected_facts_digest", "manifest_receipt",
        "dimension_payload_digest",
    }
    mapping = _exact_mapping(value, keys, "prepared dimension payload")
    if (
        mapping["schema"] != "xrd_tools.reintegrate.prepared_dimension_payload"
        or type(mapping["version"]) is not int
        or mapping["version"] != 1
        or mapping["dimension"] not in {"1d", "2d"}
        or type(mapping["dimension_labels"]) is not list
        or any(type(item) is not int or item < 0 for item in mapping["dimension_labels"])
        or not _is_digest(mapping["selected_facts_digest"])
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    preimage = dict(mapping)
    digest = preimage.pop("dimension_payload_digest")
    if not _is_digest(digest) or _digest(preimage) != digest:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    if len(_canonical_bytes(mapping)) > MAX_PREPARED_DIMENSION_BYTES:
        raise PreparedRouteRejected(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    selected = _freeze_json(mapping["selected_plan"])
    labels = tuple(mapping["dimension_labels"])
    if _digest({
        "dimension": mapping["dimension"],
        "labels": list(labels),
        "selected_plan": _legacy._plain(selected),
    }) != mapping["selected_facts_digest"]:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    try:
        receipt = admit_replacement_manifest_receipt(mapping["manifest_receipt"])
    except (TypeError, ValueError) as error:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        ) from error
    return _value(
        PreparedDimensionPayload,
        mapping["dimension"],
        labels,
        selected,
        mapping["selected_facts_digest"],
        receipt,
        digest,
    )


def _admit_admission(value: object) -> PreparedDimensionAdmission:
    keys = {
        "schema", "version", "dimension", "disposition", "miss_code",
        "dimension_payload_digest", "payload", "admission_digest",
    }
    mapping = _exact_mapping(value, keys, "prepared dimension admission")
    if len(_canonical_bytes(mapping)) > MAX_PREPARED_DIMENSION_BYTES:
        raise PreparedRouteRejected(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    if (
        mapping["schema"] != "xrd_tools.reintegrate.prepared_dimension_admission"
        or type(mapping["version"]) is not int
        or mapping["version"] != 1
        or mapping["dimension"] not in {"1d", "2d"}
        or mapping["disposition"] not in {"READY", "MISS"}
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    payload = None
    code = None
    if mapping["disposition"] == "READY":
        if mapping["miss_code"] is not None or mapping["payload"] is None:
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            )
        payload = _admit_payload(mapping["payload"])
        if (
            payload.dimension != mapping["dimension"]
            or payload.dimension_payload_digest
            != mapping["dimension_payload_digest"]
        ):
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
            )
    else:
        if mapping["payload"] is not None or mapping["dimension_payload_digest"] is not None:
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            )
        try:
            code = PreparedCapsuleMissCode(mapping["miss_code"])
        except (TypeError, ValueError) as error:
            raise PreparedRouteRejected(
                PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
            ) from error
    preimage = _admission_preimage(
        mapping["dimension"],
        mapping["disposition"],
        code,
        None if payload is None else payload.dimension_payload_digest,
    )
    if (
        not _is_digest(mapping["admission_digest"])
        or _digest(preimage) != mapping["admission_digest"]
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    return _value(
        PreparedDimensionAdmission,
        mapping["dimension"],
        mapping["disposition"],
        code,
        payload,
        mapping["admission_digest"],
    )


def _validate_shared_facts(
    target: PreparedTargetReceipt,
    artifact: PreparedArtifactFactsReceipt,
    topology_receipt: PreparedSourceTopologyReceipt,
    topology: Any,
    labels: tuple[int, ...],
    facts: tuple[Mapping[str, Any], ...],
    facts_digest: object,
) -> None:
    if (
        artifact.source_base != topology.source_base
        or tuple(topology.frame_routes) != labels
        or target.snapshot.path == topology.execution.get("path")
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    _per_frame, expected = _facts_digest_payload(
        target, artifact, topology_receipt, labels, facts,
    )
    if not _is_digest(facts_digest) or expected != facts_digest:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )


def _validate_dimension_receipt_join(
    admission: PreparedDimensionAdmission,
    *,
    target: PreparedTargetReceipt,
    artifact: PreparedArtifactFactsReceipt,
    facts_digest: str,
) -> None:
    payload = admission.payload
    if payload is None:
        return
    receipt = payload.manifest_receipt
    if (
        receipt.source_snapshot != target.snapshot
        or receipt.entry != artifact.entry
        or receipt.dimension != admission.dimension
        or receipt.facts_digest != facts_digest
        or receipt.exclusions != _replacement_exclusion_paths(
            artifact.entry,
            admission.dimension,
            include_finite_lineage=True,
        )
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )


def admit_prepared_bundle(value: Mapping[str, Any]) -> PreparedReintegrateBundle:
    keys = {
        "schema", "version", "target", "artifact", "topology", "labels",
        "facts", "one_d", "two_d", "facts_digest", "bundle_digest",
    }
    detached, _encoded_bytes = _bounded_admission_snapshot(
        value,
        role="prepared bundle",
        max_encoded_bytes=MAX_PREPARED_BUNDLE_BYTES,
    )
    mapping = _exact_mapping(detached, keys, "prepared bundle")
    if (
        mapping["schema"] != _BUNDLE_SCHEMA
        or type(mapping["version"]) is not int
        or mapping["version"] != _VERSION
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_ALGORITHM_UNSUPPORTED
        )
    labels_value = mapping["labels"]
    if (
        type(labels_value) is not list
        or not 1 <= len(labels_value) <= MAX_PREPARED_LABELS
        or any(type(label) is not int or label < 0 for label in labels_value)
        or labels_value != sorted(set(labels_value))
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_CARDINALITY_LIMIT
        )
    labels = tuple(labels_value)
    target = _admit_target(mapping["target"])
    artifact = _admit_artifact(mapping["artifact"])
    topology_receipt, topology = _admit_topology(mapping["topology"])
    facts, _indexed = _admit_facts(mapping["facts"], labels, topology)
    _validate_shared_facts(
        target,
        artifact,
        topology_receipt,
        topology,
        labels,
        facts,
        mapping["facts_digest"],
    )
    one_d = _admit_admission(mapping["one_d"])
    two_d = _admit_admission(mapping["two_d"])
    for admission in (one_d, two_d):
        _validate_dimension_receipt_join(
            admission,
            target=target,
            artifact=artifact,
            facts_digest=mapping["facts_digest"],
        )
    if (
        one_d.dimension != "1d"
        or two_d.dimension != "2d"
        or one_d.payload is not None
        and one_d.payload.dimension_labels != labels
        or two_d.payload is not None
        and two_d.payload.dimension_labels != labels
        or any(
            admission.payload is not None
            and admission.payload.manifest_receipt.facts_digest
            != mapping["facts_digest"]
            for admission in (one_d, two_d)
        )
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    root = _bundle_root(
        labels,
        mapping["facts_digest"],
        one_d.admission_digest,
        two_d.admission_digest,
    )
    if not _is_digest(mapping["bundle_digest"]) or root != mapping["bundle_digest"]:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    return _value(
        PreparedReintegrateBundle,
        _BUNDLE_SCHEMA,
        _VERSION,
        target,
        artifact,
        topology_receipt,
        labels,
        facts,
        one_d,
        two_d,
        mapping["facts_digest"],
        root,
    )


def _inspection_for_execution(
    artifact: PreparedArtifactFactsReceipt,
    topology: Any,
    labels: tuple[int, ...],
    selected_plan: Mapping[str, Any],
):
    append_lineage = (
        None
        if topology.lineage is None
        else _canonical_append_json(
            _legacy._plain(topology.lineage)
        ).encode("utf-8")
    )
    if (
        None if append_lineage is None else hashlib.sha256(
            append_lineage
        ).hexdigest()
    ) != artifact.append_lineage_digest:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    return _legacy._ArtifactInspection(
        labels,
        artifact.detector_shape,
        artifact.native_dtype,
        artifact.persisted_shared_science,
        selected_plan,
        artifact.acquisition_fingerprint,
        artifact.source_base,
        append_lineage,
        MappingProxyType({}),
        _legacy._PersistedMaskSpec(0, 0),
        None,
        None,
        topology,
    )


def admit_prepared_execution(
    value: Mapping[str, Any],
) -> PreparedReintegrateExecution:
    keys = {
        "schema", "version", "target", "artifact", "topology", "labels",
        "facts", "facts_digest", "selected_admission", "sibling_commitment",
        "bundle_digest", "execution_digest",
    }
    detached, _encoded_bytes = _bounded_admission_snapshot(
        value,
        role="prepared execution",
        max_encoded_bytes=MAX_PREPARED_EXECUTION_BYTES,
    )
    mapping = _exact_mapping(detached, keys, "prepared execution")
    if (
        mapping["schema"] != _EXECUTION_SCHEMA
        or type(mapping["version"]) is not int
        or mapping["version"] != _VERSION
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_ALGORITHM_UNSUPPORTED
        )
    labels_value = mapping["labels"]
    if (
        type(labels_value) is not list
        or not 1 <= len(labels_value) <= MAX_PREPARED_LABELS
        or labels_value != sorted(set(labels_value))
        or any(type(label) is not int or label < 0 for label in labels_value)
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_CARDINALITY_LIMIT
        )
    labels = tuple(labels_value)
    target = _admit_target(mapping["target"])
    artifact = _admit_artifact(mapping["artifact"])
    topology_receipt, topology = _admit_topology(mapping["topology"])
    facts, facts_by_label = _admit_facts(mapping["facts"], labels, topology)
    _validate_shared_facts(
        target,
        artifact,
        topology_receipt,
        topology,
        labels,
        facts,
        mapping["facts_digest"],
    )
    selected = _admit_admission(mapping["selected_admission"])
    if selected.disposition != "READY" or selected.payload is None:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    _validate_dimension_receipt_join(
        selected,
        target=target,
        artifact=artifact,
        facts_digest=mapping["facts_digest"],
    )
    sibling_value = _exact_mapping(
        mapping["sibling_commitment"],
        {"dimension", "admission_digest"},
        "prepared sibling commitment",
    )
    opposite = "2d" if selected.dimension == "1d" else "1d"
    if (
        sibling_value["dimension"] != opposite
        or not _is_digest(sibling_value["admission_digest"])
        or selected.payload.dimension_labels != labels
        or selected.payload.manifest_receipt.facts_digest
        != mapping["facts_digest"]
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    sibling = _value(
        AdmissionCommitment,
        opposite,
        sibling_value["admission_digest"],
    )
    one_digest, two_digest = (
        (selected.admission_digest, sibling.admission_digest)
        if selected.dimension == "1d"
        else (sibling.admission_digest, selected.admission_digest)
    )
    root = _bundle_root(
        labels,
        mapping["facts_digest"],
        one_digest,
        two_digest,
    )
    if not _is_digest(mapping["bundle_digest"]) or root != mapping["bundle_digest"]:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    preimage = dict(mapping)
    execution_digest = preimage.pop("execution_digest")
    if not _is_digest(execution_digest) or _digest(preimage) != execution_digest:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
        )
    inspection = _inspection_for_execution(
        artifact, topology, labels, selected.payload.selected_plan,
    )
    return _value(
        PreparedReintegrateExecution,
        _EXECUTION_SCHEMA,
        _VERSION,
        target,
        artifact,
        topology_receipt,
        labels,
        facts,
        mapping["facts_digest"],
        selected,
        sibling,
        root,
        execution_digest,
        inspection,
        facts_by_label,
    )


def select_prepared_execution(
    offered: PreparedReintegrateBundle | PreparedReintegrateOffer | None,
    dimension: Literal["1d", "2d"],
) -> PreparedReintegrateExecution:
    if offered is None:
        raise PreparedCapsuleMiss(PreparedCapsuleMissCode.CAPSULE_NOT_SUPPLIED)
    if type(offered) is PreparedReintegrateOffer:
        if offered.disposition == "MISS":
            raise PreparedCapsuleMiss(offered.miss_code)
        offered = offered.bundle
    if type(offered) is not PreparedReintegrateBundle or dimension not in {"1d", "2d"}:
        raise PreparedCapsuleMiss(
            PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
        )
    try:
        bundle = admit_prepared_bundle(prepared_bundle_mapping(offered))
    except PreparedRouteRejected as error:
        raise PreparedCapsuleMiss(error.code) from error
    selected = bundle.one_d if dimension == "1d" else bundle.two_d
    sibling = bundle.two_d if dimension == "1d" else bundle.one_d
    if selected.disposition == "MISS":
        raise PreparedCapsuleMiss(selected.miss_code)
    commitment = _value(
        AdmissionCommitment,
        sibling.dimension,
        sibling.admission_digest,
    )
    provisional = _value(
        PreparedReintegrateExecution,
        _EXECUTION_SCHEMA,
        _VERSION,
        bundle.target,
        bundle.artifact,
        bundle.topology,
        bundle.labels,
        bundle.facts,
        bundle.facts_digest,
        selected,
        commitment,
        bundle.bundle_digest,
        "0" * 64,
        None,
        MappingProxyType({}),
    )
    mapping = prepared_execution_mapping(provisional)
    mapping.pop("execution_digest")
    mapping["execution_digest"] = _digest(mapping)
    try:
        return admit_prepared_execution(mapping)
    except PreparedRouteRejected as error:
        raise PreparedCapsuleMiss(error.code) from error


def _known_inspection_miss(error: ValueError) -> PreparedCapsuleMissCode | None:
    message = str(error)
    if "selected frame inventory" in message:
        return PreparedCapsuleMissCode.DIMENSION_UNAVAILABLE
    if "current xdart" in message or "current processed" in message:
        return PreparedCapsuleMissCode.TARGET_SCHEMA_UNSUPPORTED
    if "TARGET_SNAPSHOT_CHANGED" in message:
        return PreparedCapsuleMissCode.TARGET_CHANGED
    if "SOURCE_REVISION" in message:
        return PreparedCapsuleMissCode.SOURCE_REVISION_CHANGED
    if "append" in message.lower() and "lineage" in message.lower():
        return PreparedCapsuleMissCode.APPEND_LINEAGE_UNSUPPORTED
    if (
        "SOURCE_LINEAGE" in message
        or "HDF5_DEPENDENCY" in message
        or "RAW_DECODER" in message
    ):
        return PreparedCapsuleMissCode.SOURCE_TOPOLOGY_UNSUPPORTED
    return None


def _is_self_external_topology(topology: Any) -> bool:
    execution = topology.execution
    external_members = execution.get("external_members", ())
    same_file_external = False
    if len(external_members) == 1:
        try:
            source_path = execution["path"]
            member_path = external_members[0]["file"]["path"]
            same_file_external = (
                os.path.normcase(os.path.normpath(source_path))
                == os.path.normcase(os.path.normpath(member_path))
            )
        except (KeyError, TypeError, ValueError):
            same_file_external = True
    return same_file_external or any(
        route.hdf is not None
        and os.path.normcase(os.path.normpath(route.source_path))
        == os.path.normcase(os.path.normpath(route.hdf.member_path))
        for route in topology.frame_routes.values()
    )


def _initial_artifact_miss(inspection: Any) -> PreparedCapsuleMissCode | None:
    shared = _legacy._plain(inspection.persisted_shared_science)
    if (
        inspection.mask_spec.retained_bytes
        or inspection.mask_spec.decode_bytes
        or shared.get("geometry") is not None
        or shared.get("background", {}).get("mode") != "None"
        or shared.get("gi", {}).get("enabled") is not False
        or inspection.gi_values
    ):
        return PreparedCapsuleMissCode.ARTIFACT_FACTS_UNSUPPORTED
    lineage = inspection.topology.lineage
    if inspection.append_lineage is not None or lineage is not None:
        epochs = None if type(lineage) is not dict else lineage.get("epochs")
        if (
            inspection.append_lineage is None
            or type(epochs) is not list
            or len(epochs) != 1
            or type(epochs[0]) is not dict
            or epochs[0].get("labels") != list(inspection.labels)
        ):
            return PreparedCapsuleMissCode.APPEND_LINEAGE_UNSUPPORTED
    if inspection.raw_options is not None:
        return PreparedCapsuleMissCode.SOURCE_TOPOLOGY_UNSUPPORTED
    topology = inspection.topology
    execution = topology.execution
    external_members = execution.get("external_members", ())
    if (
        execution.get("adapter_id") != "nexus_hdf5"
        or len(external_members) != 1
        or _is_self_external_topology(topology)
        or execution.get("dependency_files") != []
        or execution.get("metadata_sources") != []
        or execution.get("member_stamps") != []
        or len(topology.external_signature) != 1
        or any(
            route.hdf is None or route.hdf.storage_slices
            for route in topology.frame_routes.values()
        )
    ):
        return PreparedCapsuleMissCode.SOURCE_TOPOLOGY_UNSUPPORTED
    metadata_keys, include_geometry = _legacy._fact_projection(
        inspection.persisted_selected_plan,
        inspection.persisted_shared_science,
    )
    if metadata_keys or include_geometry:
        return PreparedCapsuleMissCode.SOURCE_TOPOLOGY_UNSUPPORTED
    return None


def _append_miss_before_topology(
    path: Path,
    entry: str,
    labels: tuple[int, ...],
    snapshot: TargetSnapshot,
    revision,
) -> PreparedCapsuleMissCode | None:
    """Give malformed/multi-epoch Append its normative miss precedence."""

    present = False
    try:
        with _legacy._open_target_hdf(path) as document:
            _legacy._target_hdf_fence(document, path, snapshot, revision)
            present = document.get(
                f"/{entry}/reduction/config/append_lineage",
                getlink=True,
            ) is not None
            if not present:
                _source_base, raw, lineage = decode_replacement_lineage(
                    document, entry=entry,
                )
                if raw is not None or lineage is not None:
                    raise ValueError("Append lineage presence is inconsistent")
                prefix = None
            else:
                prefix = decode_committed_append_prefix(document, entry=entry)
                lineage = json.loads(prefix.lineage_json)
            _legacy._target_hdf_fence(document, path, snapshot, revision)
    except (OSError, TypeError, ValueError, WriterStateError) as error:
        try:
            target_changed = _legacy.capture_target_snapshot(path) != snapshot
        except BaseException:
            target_changed = True
        if target_changed or "TARGET_SNAPSHOT_CHANGED" in str(error):
            raise ValueError("TARGET_SNAPSHOT_CHANGED") from error
        return (
            PreparedCapsuleMissCode.APPEND_LINEAGE_UNSUPPORTED
            if present else None
        )
    if not present:
        return None
    epochs = None if type(lineage) is not dict else lineage.get("epochs")
    if (
        type(epochs) is not list
        or len(epochs) != 1
        or type(epochs[0]) is not dict
        or epochs[0].get("labels") != list(labels)
        or prefix.intent.labels != labels
    ):
        return PreparedCapsuleMissCode.APPEND_LINEAGE_UNSUPPORTED
    return None


def _offer_miss(code: PreparedCapsuleMissCode) -> PreparedReintegrateOffer:
    return _value(PreparedReintegrateOffer, "MISS", None, code)


def _prepared_dimension_inventories(
    path: Path,
    entry: str,
    snapshot: TargetSnapshot,
    revision,
    *,
    cancel_token=None,
) -> Mapping[str, tuple[int, ...] | None]:
    """Read the two local result inventories before dimension inspection.

    A mismatched or absent dimension is a local admission fact.  Reading this
    tiny surface first prevents its unrelated source/config state from
    contaminating a matching sibling.
    """

    values: dict[str, tuple[int, ...] | None] = {}
    with _legacy._open_target_hdf(path) as document:
        _legacy._target_hdf_fence(document, path, snapshot, revision)
        root = _replacement_hard_group(document, entry)
        if root is None:
            raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        for dimension in ("1d", "2d"):
            _legacy._event(cancel_token)
            integrated = _replacement_hard_group(
                root, f"integrated_{dimension}", h5py.Group,
            )
            index = (
                None
                if integrated is None
                else _replacement_hard_group(
                    integrated, "frame_index", h5py.Dataset,
                )
            )
            if index is None:
                values[dimension] = None
                continue
            try:
                raw = _read_replacement_frame_index(
                    index,
                    "selected frame inventory",
                    require_nonempty=True,
                )
            except WriterStateError:
                values[dimension] = None
                continue
            labels = tuple(int(item) for item in raw)
            values[dimension] = (
                labels
                if labels
                and labels == tuple(sorted(set(labels)))
                and all(label >= 0 for label in labels)
                else None
            )
        _legacy._target_hdf_fence(document, path, snapshot, revision)
    return MappingProxyType(values)


def prepare_reintegrate_bundle(
    source: FiniteSourceAdmission,
    *,
    entry: str,
    labels: tuple[int, ...],
    expected_terminal: StreamTerminal | None = None,
    source_root: str | os.PathLike[str] | None = None,
    cancel_token=None,
) -> PreparedReintegrateOffer:
    """Prepare one fixed dual-dimension, handle-free Browse offer.

    Ordinary bounded ineligibility is returned as a typed offer.  Cancellation,
    unexpected implementation failures, and target-bracket drift are raised;
    they are never relabeled as a reason to fall back.
    """

    _legacy._event(cancel_token)
    if (
        type(source) is not FiniteSourceAdmission
        or type(entry) is not str
        or not entry
        or type(labels) is not tuple
        or not labels
        or labels != tuple(sorted(set(labels)))
        or any(type(label) is not int or label < 0 for label in labels)
    ):
        return _offer_miss(PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED)
    if len(labels) > MAX_PREPARED_LABELS:
        return _offer_miss(PreparedCapsuleMissCode.CAPSULE_CARDINALITY_LIMIT)
    try:
        _bounded_text(entry, "entry")
        requested_root = None
        if source_root is not None:
            requested_root = _bounded_text(os.fspath(source_root), "source root")
            requested_root = os.path.normcase(os.path.abspath(requested_root))
    except PreparedCapsuleMiss as error:
        return _offer_miss(error.code)
    path = Path(source.path)
    observed = capture_finite_source(path)
    if observed.snapshot != source.snapshot:
        raise ValueError("TARGET_SNAPSHOT_CHANGED")

    def miss(code: PreparedCapsuleMissCode) -> PreparedReintegrateOffer:
        """Only authorize fallback while the admitted source stays exact."""

        _legacy._event(cancel_token)
        try:
            current = capture_finite_source(path)
        except (OSError, TypeError, ValueError) as error:
            raise ValueError("TARGET_SNAPSHOT_CHANGED") from error
        if current.snapshot != source.snapshot:
            raise ValueError("TARGET_SNAPSHOT_CHANGED")
        _legacy._event(cancel_token)
        return _offer_miss(code)

    snapshot = _target_snapshot(source.snapshot)
    if expected_terminal is not None:
        if type(expected_terminal) is not StreamTerminal:
            return miss(PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED)
        try:
            terminal_snapshot = revalidate_stream_terminal(path, expected_terminal)
        except (OSError, ValueError) as error:
            raise ValueError("TERMINAL_CHANGED") from error
        if (
            not terminal_snapshot.exists
            or (
                terminal_snapshot.size,
                terminal_snapshot.mtime_ns,
                terminal_snapshot.device,
                terminal_snapshot.inode,
            ) != (
                snapshot.size,
                snapshot.mtime_ns,
                snapshot.device,
                snapshot.inode,
            )
        ):
            raise ValueError("TERMINAL_CHANGED")
    target = _target_receipt(source.snapshot, expected_terminal)
    revision = _legacy._target_object_revision(path, snapshot)
    append_code = _append_miss_before_topology(
        path, entry, labels, snapshot, revision,
    )
    if append_code is not None:
        return miss(append_code)
    inventories = _prepared_dimension_inventories(
        path,
        entry,
        snapshot,
        revision,
        cancel_token=cancel_token,
    )
    matching_dimensions = tuple(
        dimension
        for dimension in ("1d", "2d")
        if inventories[dimension] == labels
    )
    inspections: dict[str, Any] = {}
    dimensions_to_inspect = matching_dimensions
    shared_only = not dimensions_to_inspect
    if shared_only:
        dimensions_to_inspect = (
            next(
                (
                    dimension
                    for dimension in ("1d", "2d")
                    if inventories[dimension] is not None
                ),
                "1d",
            ),
        )
    for dimension in dimensions_to_inspect:
        _legacy._event(cancel_token)
        try:
            inspections[dimension] = _legacy._inspect_artifact(
                path,
                entry,
                dimension,
                snapshot,
                revision,
                None,
                read_mask=False,
                source_inventory_labels=(labels if shared_only else None),
                shared_only=shared_only,
            )
        except ValueError as error:
            if "TARGET_SNAPSHOT_CHANGED" in str(error):
                raise ValueError("TARGET_SNAPSHOT_CHANGED") from error
            code = _known_inspection_miss(error)
            if code is None:
                raise
            return miss(code)
    if not inspections:
        return miss(PreparedCapsuleMissCode.DIMENSION_UNAVAILABLE)
    matching = {
        dimension: inspection
        for dimension, inspection in inspections.items()
        if tuple(inspection.labels) == labels
    }
    base = (
        matching.get("1d")
        or matching.get("2d")
        or next(iter(inspections.values()))._replace(labels=labels)
    )
    if requested_root is not None and requested_root != base.source_base:
        whole_miss = PreparedCapsuleMissCode.SOURCE_ROOT_CHANGED
    elif _is_self_external_topology(base.topology):
        return miss(PreparedCapsuleMissCode.SOURCE_TOPOLOGY_UNSUPPORTED)
    else:
        whole_miss = _initial_artifact_miss(base)
    for inspection in inspections.values():
        neutral = (
            tuple(inspection.detector_shape),
            inspection.native_dtype,
            _legacy._plain(inspection.persisted_shared_science),
            inspection.acquisition_fingerprint,
            inspection.source_base,
            inspection.append_lineage,
            inspection.mask_spec,
            inspection.raw_options,
        )
        baseline = (
            tuple(base.detector_shape),
            base.native_dtype,
            _legacy._plain(base.persisted_shared_science),
            base.acquisition_fingerprint,
            base.source_base,
            base.append_lineage,
            base.mask_spec,
            base.raw_options,
        )
        if neutral != baseline:
            raise RuntimeError("prepared dual-dimension artifact facts disagree")
    for inspection in matching.values():
        if (
            _digest(_topology_mapping(inspection.topology))
            != _digest(_topology_mapping(base.topology))
        ):
            raise RuntimeError(
                "prepared matching-dimension source topology disagrees"
            )
    artifact = _artifact_receipt(base, entry)
    try:
        topology_receipt = _topology_receipt(base.topology)
    except PreparedCapsuleMiss as error:
        return miss(error.code)
    compact_facts = []
    context = (
        base.source_base,
        base.topology.lineage,
        base.topology.execution,
    )
    with _legacy._open_target_hdf(path) as document:
        _legacy._target_hdf_fence(document, path, snapshot, revision)
        for label in labels:
            _legacy._event(cancel_token)
            fact = _decode_replacement_fact(
                document,
                label,
                entry=entry,
                context=context,
                metadata_keys=(),
                include_geometry=False,
            )
            fact = dict(fact)
            fact["path"] = os.path.normcase(os.path.normpath(os.path.abspath(
                _legacy._resolve_source_locator(
                    fact["path"], base.source_base,
                )
            )))
            if (
                fact["metadata"]
                or fact["geometry"]
                or fact["background_dependency"] is not None
            ):
                whole_miss = (
                    whole_miss
                    or PreparedCapsuleMissCode.ARTIFACT_FACTS_UNSUPPORTED
                )
            try:
                _bounded_text(fact["path"], "source path")
                compact_facts.append(_compact_fact(fact))
            except PreparedCapsuleMiss as error:
                return miss(error.code)
        _legacy._target_hdf_fence(document, path, snapshot, revision)
    facts = tuple(compact_facts)
    if tuple(fact["label"] for fact in facts) != labels:
        raise RuntimeError("prepared source facts escaped their label inventory")
    if len(_canonical_bytes([_legacy._plain(fact) for fact in facts])) > MAX_PREPARED_FACTS_BYTES:
        return miss(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    _per_frame, facts_digest = _facts_digest_payload(
        target, artifact, topology_receipt, labels, facts,
    )
    admissions: dict[str, PreparedDimensionAdmission] = {}
    receipts: dict[str, ReplacementManifestReceipt] = {}
    ready_dimensions = [
        dimension
        for dimension in matching_dimensions
        if whole_miss is None
    ]
    if ready_dimensions:
        with h5py.File(path, "r") as document:
            for dimension in ready_dimensions:
                try:
                    receipts[dimension] = prepare_replacement_manifest_receipt(
                        document,
                        source.snapshot,
                        entry=entry,
                        dimension=dimension,
                        facts_digest=facts_digest,
                    )
                except ReplacementManifestTargetChanged as error:
                    raise ValueError("TARGET_SNAPSHOT_CHANGED") from error
                except WriterStateError:
                    try:
                        _require_replacement_manifest_source(
                            document, source.snapshot,
                        )
                    except ReplacementManifestTargetChanged as changed:
                        raise ValueError(
                            "TARGET_SNAPSHOT_CHANGED"
                        ) from changed
                    return miss(
                        PreparedCapsuleMissCode.MANIFEST_DOMAIN_UNSUPPORTED
                    )
    for dimension in ("1d", "2d"):
        inspection = inspections.get(dimension)
        if whole_miss is not None:
            admissions[dimension] = _miss_admission(dimension, whole_miss)
        elif inventories[dimension] is None:
            admissions[dimension] = _miss_admission(
                dimension,
                PreparedCapsuleMissCode.DIMENSION_UNAVAILABLE,
            )
        elif inventories[dimension] != labels:
            admissions[dimension] = _miss_admission(
                dimension,
                PreparedCapsuleMissCode.DIMENSION_LABELS_INCOMPATIBLE,
            )
        else:
            if inspection is None:
                raise RuntimeError(
                    "prepared matching dimension lacks inspection"
                )
            try:
                admissions[dimension] = _ready_admission(_dimension_payload(
                    dimension,
                    labels,
                    inspection.persisted_selected_plan,
                    receipts[dimension],
                ))
            except PreparedCapsuleMiss as error:
                admissions[dimension] = _miss_admission(
                    dimension, error.code,
                )
    root = _bundle_root(
        labels,
        facts_digest,
        admissions["1d"].admission_digest,
        admissions["2d"].admission_digest,
    )
    bundle = _value(
        PreparedReintegrateBundle,
        _BUNDLE_SCHEMA,
        _VERSION,
        target,
        artifact,
        topology_receipt,
        labels,
        facts,
        admissions["1d"],
        admissions["2d"],
        facts_digest,
        root,
    )
    if len(_canonical_bytes(prepared_bundle_mapping(bundle))) > MAX_PREPARED_BUNDLE_BYTES:
        return miss(PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT)
    try:
        admitted = admit_prepared_bundle(prepared_bundle_mapping(bundle))
    except PreparedRouteRejected as error:
        raise RuntimeError("prepared bundle factory emitted invalid evidence") from error
    after = capture_finite_source(path)
    if after.snapshot != source.snapshot:
        raise ValueError("TARGET_SNAPSHOT_CHANGED")
    _legacy._event(cancel_token)
    return _value(PreparedReintegrateOffer, "READY", admitted, None)


def preflight_prepared_execution(
    execution: PreparedReintegrateExecution,
    *,
    source: FiniteSourceAdmission,
    entry: str,
    dimension: Literal["1d", "2d"],
    cancel_token=None,
) -> None:
    """Final effect-free route check before publisher candidate ownership."""

    _legacy._event(cancel_token)
    if (
        type(execution) is not PreparedReintegrateExecution
        or type(source) is not FiniteSourceAdmission
        or execution.target.snapshot != source.snapshot
        or execution.artifact.entry != entry
        or execution.selected_admission.dimension != dimension
        or execution.selected_admission.payload is None
        or execution.selected_admission.payload.manifest_receipt.source_snapshot
        != source.snapshot
        or execution.selected_admission.payload.manifest_receipt.facts_digest
        != execution.facts_digest
    ):
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.ARTIFACT_FACTS_CHANGED
        )
    # Full finite-source recapture above the caller and the raw topology fence
    # here cover both the processed artifact and its external member graph.
    try:
        _legacy._validate_terminal_topology(
            execution._inspection.topology, cancel_token,
        )
    except OSError as error:
        raise PreparedRouteRejected(
            PreparedCapsuleMissCode.SOURCE_REVISION_CHANGED
        ) from error
    except ValueError as error:
        message = str(error)
        code = (
            PreparedCapsuleMissCode.SOURCE_REVISION_CHANGED
            if "REVISION" in message
            else PreparedCapsuleMissCode.SOURCE_TOPOLOGY_CHANGED
        )
        raise PreparedRouteRejected(code) from error
    _legacy._event(cancel_token)


__all__ = [
    "AdmissionCommitment",
    "LegacyRouteReason",
    "PreparedArtifactFactsReceipt",
    "PreparedCapsuleMiss",
    "PreparedCapsuleMissCode",
    "PreparedDimensionAdmission",
    "PreparedDimensionPayload",
    "PreparedReintegrateBundle",
    "PreparedReintegrateExecution",
    "PreparedReintegrateOffer",
    "PreparedRouteChanged",
    "PreparedRouteRejected",
    "PreparedSourceTopologyReceipt",
    "PreparedTargetReceipt",
    "admit_prepared_bundle",
    "admit_prepared_execution",
    "prepare_reintegrate_bundle",
    "prepared_bundle_mapping",
    "prepared_execution_mapping",
    "preflight_prepared_execution",
    "select_prepared_execution",
    "source_topology_identity",
]
