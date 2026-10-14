"""Strict standalone NeXus publication for Stitch and RSM results.

An analysis artifact is deliberately not a processed scan: it contains exactly
one scan-level result and never fabricates an integrated-frame group.  This
module supplies the schema and strict semantic readback to the shared finite
publication owner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

import h5py
import numpy as np

from xrd_tools.io.output_transaction import (
    OutputTransactionCoordinator,
    StreamTerminal,
    TargetSnapshot,
    TargetChanged,
    TransactionStateError,
    capture_target_snapshot,
    get_output_transaction_coordinator,
    revalidate_stream_terminal,
    stream_terminal_object_revision,
)
from xrd_tools.io.finite_artifact import _FinitePublicationSession
from xrd_tools.io.output_safety import check_output_not_source
from xrd_tools.io.stat_identity import identity_ctime_ns
from xrd_tools.io.stitch_record import (
    FRAMES_JSON as _STITCH_FRAMES_JSON,
    GI_MAP as _STITCH_GI_MAP,
    GI_MODES as _STITCH_GI_MODES,
    MAP as _STITCH_MAP,
    STITCH_GROUP as _STITCH_GROUP,
    is_stitch_record as _is_stitch_record,
    read_stitch_record as _read_stitch_record,
)


ANALYSIS_SCHEMA_ATTR = "ssrl_schema"
ANALYSIS_SCHEMA_NAME = "xrd_tools.analysis_artifact"
ANALYSIS_SCHEMA_VERSION_ATTR = "ssrl_schema_version"
ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION_V2 = 2
ANALYSIS_SCHEMA_VERSION_V3 = 3
ANALYSIS_SCHEMA_VERSION_STITCH_NEUTRAL = 4
ANALYSIS_SCHEMA_VERSION_XU_STITCH_NEUTRAL = 5
ANALYSIS_SCHEMA_VERSION_STITCH_NATIVE = 6
ANALYSIS_SCHEMA_VERSION_XU_STITCH_NATIVE = 7
_STITCH_NATIVE_VERSIONS = {6, 7}
#: The Stitch versions written only as a Stitch 1D (a native Stitch 2D is an Int 2D record: version 6,
#: MultiGeometry's, or 7, xrayutilities').
_STITCH_1D_ONLY_VERSIONS = {5}
# Keep ordinary and attested science contracts distinct; only physical names change.
_STITCH_NEUTRAL_CONTRACTS = {4: 1, 5: 2, 6: 1, 7: 2}
ANALYSIS_KIND_ATTR = "analysis_kind"
_ENTRY = "entry"
_PROVENANCE = "provenance_json"
_EXECUTION_ATTESTATION = "execution_attestation_json"
_EXECUTION_ATTESTATION_DIGEST_ATTR = "execution_attestation_digest"
_MAX_PROVENANCE_BYTES = 1 << 20
_MAX_PROVENANCE_NESTING = 64
_MAX_EXECUTION_ATTESTATION_BYTES = 16_384
_MAX_EXECUTION_ATTESTATION_NESTING = 4
_MAX_ATTRIBUTE_BYTES = 4096
_MAX_ATTRIBUTE_ITEMS = 16
_MAX_AXIS_POINTS = 1_000_000
_MAX_RESULT_ELEMENTS = 64_000_000
_MAX_READ_BLOCK_BYTES = 8 << 20
_MAX_ARTIFACT_FILE_BYTES = 1 << 30
_RESULT_FINGERPRINT_PREFIX = b"xrd_tools.analysis-artifact-result.v1\0"
_STORED_RESULT_PROJECTION_POLICY = "analysis_artifact_stored_le_f4_v1"
_STORED_QNAN_F4_BITS = np.uint32(0x7FC00000)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ANALYSIS_RECEIPT_FACTORY = object()
_ANALYSIS_PAYLOAD_FACTORY = object()
_ANALYSIS_CANDIDATE_INSPECTION = object()
_ANALYSIS_PROJECTION_FACTORY = object()
_ANALYSIS_V1_DECODE_PROJECTION = object()


def _artifact_contract_version(version: int) -> int:
    return _STITCH_NEUTRAL_CONTRACTS.get(version, version)


class AnalysisArtifactKind(str, Enum):
    STITCH_1D = "stitch-1d"
    STITCH_2D = "stitch-2d"
    RSM = "rsm"

    @property
    def group(self) -> str:
        return {
            AnalysisArtifactKind.STITCH_1D: "stitched_1d",
            AnalysisArtifactKind.STITCH_2D: "stitched_2d",
            AnalysisArtifactKind.RSM: "rsm",
        }[self]


class AnalysisArtifactOverwrite(str, Enum):
    CREATE_NEW = "create-new"
    REPLACE = "replace"


class AnalysisArtifactError(RuntimeError):
    pass


class AnalysisArtifactInvalid(AnalysisArtifactError):
    pass


class OlderStitchArtifact(AnalysisArtifactInvalid):
    """A file in the layout before Stitch results became Int records, whose envelope says Stitch: a
    regular bounded file holding only ``/entry``, a known schema version, the exact entry attributes, the
    schema marker and a Stitch kind (nothing inside it is read or checked). Not read, but that kind's
    result, so Replace may overwrite it ("stitch it again"; Codex terminal review R1: this is all the
    check promises)."""

    def __init__(self, kind: AnalysisArtifactKind) -> None:
        super().__init__("a Stitch result is an Int record; this older Stitch file is not read: stitch it again")
        self.kind = kind


class AnalysisArtifactProjectionInvalid(AnalysisArtifactError, ValueError):
    """A scientific result cannot be represented by the stored artifact schema."""

    code = "STITCH_RESULT_STORAGE_PROJECTION_INVALID"


class AnalysisArtifactCleanupPending(AnalysisArtifactError):
    def __init__(self, snapshot: object) -> None:
        self.snapshot = snapshot
        self.published = bool(getattr(snapshot, "published", False))
        super().__init__(
            "analysis artifact published; verification remains pending"
            if self.published else "analysis artifact document close remains pending"
        )


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{name} must be lowercase SHA-256 hex")
    return value


def _absolute_path(value: object) -> str:
    try:
        return os.path.normcase(
            os.path.abspath(os.path.expanduser(os.fspath(value)))
        )
    except TypeError as error:
        raise TypeError("analysis artifact target must be path-like") from error


def _normalize_target(value: object) -> str:
    target = _absolute_path(value)
    if Path(target).suffix.casefold() != ".nexus":
        raise ValueError("analysis artifact target must end in .nexus")
    return target


def _charge_json(budget: list[int], amount: int) -> None:
    budget[0] += int(amount)
    if budget[0] > _MAX_PROVENANCE_BYTES:
        raise ValueError("analysis provenance exceeds the bounded input size")


def _json_string_size(value: str) -> int:
    if len(value) > _MAX_PROVENANCE_BYTES:
        return _MAX_PROVENANCE_BYTES + 1
    size = 2
    for character in value:
        code = ord(character)
        if character in {'"', "\\"}:
            size += 2
        elif character in {"\b", "\t", "\n", "\f", "\r"}:
            size += 2
        elif code < 0x20:
            size += 6
        else:
            try:
                size += len(character.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError(
                    "analysis provenance contains invalid Unicode"
                ) from error
        if size > _MAX_PROVENANCE_BYTES:
            break
    return size


def _json_value(
    value: object,
    active: set[int],
    depth: int,
    budget: list[int],
) -> object:
    if type(value) in {dict, list, tuple} and depth > _MAX_PROVENANCE_NESTING:
        raise ValueError("analysis provenance exceeds the nesting bound")
    recursive = type(value) in {dict, list, tuple}
    marker = id(value)
    if recursive:
        if marker in active:
            raise ValueError("analysis provenance is cyclic")
        active.add(marker)
    try:
        if value is None:
            _charge_json(budget, 4)
            return value
        if type(value) is str:
            _charge_json(budget, _json_string_size(value))
            return value
        if type(value) is bool:
            _charge_json(budget, 4 if value else 5)
            return value
        if type(value) is int:
            if value.bit_length() > 3_500_000:
                raise ValueError("analysis provenance exceeds the bounded input size")
            _charge_json(budget, len(str(value)))
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("analysis provenance contains a nonfinite number")
            _charge_json(budget, len(repr(value)))
            return value
        if type(value) in {list, tuple}:
            _charge_json(budget, 2)
            frozen = []
            for index, item in enumerate(value):
                if index:
                    _charge_json(budget, 1)
                frozen.append(_json_value(item, active, depth + 1, budget))
            return frozen
        if type(value) is dict:
            _charge_json(budget, 2)
            frozen = {}
            for index, (key, item) in enumerate(value.items()):
                if type(key) is not str:
                    raise TypeError("analysis provenance keys must be exact strings")
                if index:
                    _charge_json(budget, 1)
                _charge_json(budget, _json_string_size(key) + 1)
                frozen[key] = _json_value(item, active, depth + 1, budget)
            return frozen
        raise TypeError(
            f"analysis provenance contains unsupported {type(value).__name__}"
        )
    finally:
        if recursive:
            active.remove(marker)


def canonical_analysis_provenance(value: Mapping[str, object]) -> str:
    if type(value) is not dict:
        raise TypeError("analysis provenance must be an exact dictionary")
    frozen = _json_value(value, set(), 1, [0])
    text = json.dumps(
        frozen,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if len(text.encode("utf-8")) > _MAX_PROVENANCE_BYTES:
        raise ValueError("analysis provenance exceeds the bounded input size")
    return text


def _execution_attestation_value(
    value: object,
    active: set[int],
    depth: int,
    budget: list[int],
) -> object:
    if type(value) in {dict, list, tuple} and depth > _MAX_EXECUTION_ATTESTATION_NESTING:
        raise ValueError("analysis execution attestation exceeds the nesting bound")
    recursive = type(value) in {dict, list, tuple}
    marker = id(value)
    if recursive:
        if marker in active:
            raise ValueError("analysis execution attestation is cyclic")
        active.add(marker)

    def charge(amount: int) -> None:
        budget[0] += int(amount)
        if budget[0] > _MAX_EXECUTION_ATTESTATION_BYTES:
            raise ValueError("analysis execution attestation is oversized")

    try:
        if value is None:
            charge(4)
            return None
        if type(value) is str:
            charge(_json_string_size(value))
            return value
        if type(value) is bool:
            charge(4 if value else 5)
            return value
        if type(value) is int:
            if value.bit_length() > 65_536:
                raise ValueError("analysis execution attestation is oversized")
            charge(len(str(value)))
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError(
                    "analysis execution attestation contains a nonfinite number"
                )
            charge(len(repr(value)))
            return value
        if type(value) in {list, tuple}:
            charge(2)
            frozen = []
            for index, item in enumerate(value):
                if index:
                    charge(1)
                frozen.append(
                    _execution_attestation_value(
                        item, active, depth + 1, budget
                    )
                )
            return frozen
        if type(value) is dict:
            charge(2)
            frozen = {}
            for index, (key, item) in enumerate(value.items()):
                if type(key) is not str:
                    raise TypeError(
                        "analysis execution attestation keys must be exact strings"
                    )
                if index:
                    charge(1)
                charge(_json_string_size(key) + 1)
                frozen[key] = _execution_attestation_value(
                    item, active, depth + 1, budget
                )
            return frozen
        raise TypeError(
            "analysis execution attestation contains unsupported "
            f"{type(value).__name__}"
        )
    finally:
        if recursive:
            active.remove(marker)


def canonical_analysis_execution_attestation(
    value: Mapping[str, object],
) -> str:
    if type(value) is not dict:
        raise TypeError("analysis execution attestation must be an exact dictionary")
    frozen = _execution_attestation_value(value, set(), 1, [0])
    text = json.dumps(
        frozen,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if len(text.encode("utf-8")) > _MAX_EXECUTION_ATTESTATION_BYTES:
        raise ValueError("analysis execution attestation is oversized")
    return text


def _validated_execution_attestation(
    value: Mapping[str, object],
    *,
    kind: AnalysisArtifactKind,
    request_fingerprint: str,
) -> tuple[str, dict[str, object]]:
    text = canonical_analysis_execution_attestation(value)
    parsed = json.loads(text)
    top_keys = {
        "schema_version",
        "module_request_fingerprint",
        "result_projection_policy",
        "result_fingerprint",
        "selected_frame_count",
        "release_check_frame_count",
        "release_check_passed",
        "q_root_policy",
        "xu_runtime",
    }
    runtime = parsed.get("xu_runtime")
    selected = parsed.get("selected_frame_count")
    released = parsed.get("release_check_frame_count")

    def valid_runtime_projection(value: object) -> bool:
        from xrd_tools.core.geometry.xu_runtime import XuRuntimeExecutionRecord

        if type(value) is not dict:
            return False
        try:
            record = XuRuntimeExecutionRecord(**value)
        except TypeError:
            return False
        # Preserve the original closed shape and admit the complete observed
        # environment extension. Partial, null, or unknown fields fail equality.
        return record.to_attestation() == value

    if kind is AnalysisArtifactKind.RSM:
        rsm_top_keys = {
            "schema_version",
            "module_request_fingerprint",
            "result_projection_policy",
            "result_fingerprint",
            "geometry_asset_receipt_fingerprint",
            "effective_geometry_fingerprint",
            "common_grid_fingerprint",
            "selected_scan_count",
            "selected_frame_count",
            "science_chunk_count",
            "q_release_check_chunk_count",
            "frame_release_check_frame_count",
            "release_check_passed",
            "member_masks",
            "xu_runtime",
        }
        attestation_version = parsed.get("schema_version")
        if attestation_version == "rsm-execution-attestation-v2":
            rsm_top_keys.update(
                {"coordinate_frame", "axis_names", "axis_units", "matrix_policy"}
            )
        mask_keys = {
            "ordinal",
            "member_preflight_fingerprint",
            "mask_policy",
            "full_shape",
            "full_raw_digest",
            "full_masked_pixel_count",
            "cropped_shape",
            "cropped_raw_digest",
            "cropped_masked_pixel_count",
            "mask_receipt_fingerprint",
        }
        selected_scans = parsed.get("selected_scan_count")
        science_chunks = parsed.get("science_chunk_count")
        q_released = parsed.get("q_release_check_chunk_count")
        frame_released = parsed.get("frame_release_check_frame_count")
        masks = parsed.get("member_masks")

        def valid_shape(value: object) -> bool:
            return (
                type(value) is list
                and len(value) == 2
                and all(type(item) is int and item >= 2 for item in value)
            )

        def valid_mask(item: object, ordinal: int) -> bool:
            if type(item) is not dict or set(item) != mask_keys:
                return False
            policy = item.get("mask_policy")
            full_shape = item.get("full_shape")
            cropped_shape = item.get("cropped_shape")
            full_count = item.get("full_masked_pixel_count")
            cropped_count = item.get("cropped_masked_pixel_count")
            if (
                item.get("ordinal") != ordinal
                or type(item.get("ordinal")) is not int
                or type(item.get("member_preflight_fingerprint")) is not str
                or _SHA256.fullmatch(
                    item["member_preflight_fingerprint"]
                )
                is None
                or type(policy) is not str
                or policy
                not in (
                    "none",
                    "exact-all-selected-frames-static-hot-v1",
                )
                or not valid_shape(full_shape)
                or not valid_shape(cropped_shape)
                or any(
                    cropped > full
                    for full, cropped in zip(
                        full_shape,
                        cropped_shape,
                        strict=True,
                    )
                )
                or type(full_count) is not int
                or type(cropped_count) is not int
                or not 0 <= full_count <= math.prod(full_shape)
                or not 0 <= cropped_count <= math.prod(cropped_shape)
                or cropped_count > full_count
                or type(item.get("mask_receipt_fingerprint")) is not str
                or _SHA256.fullmatch(item["mask_receipt_fingerprint"])
                is None
            ):
                return False
            if policy == "none":
                return (
                    item.get("full_raw_digest") is None
                    and item.get("cropped_raw_digest") is None
                    and full_count == 0
                    and cropped_count == 0
                )
            return (
                type(item.get("full_raw_digest")) is str
                and _SHA256.fullmatch(item["full_raw_digest"]) is not None
                and type(item.get("cropped_raw_digest")) is str
                and _SHA256.fullmatch(item["cropped_raw_digest"]) is not None
            )

        valid_frame = attestation_version == "rsm-execution-attestation-v1"
        if attestation_version == "rsm-execution-attestation-v2":
            from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame

            try:
                coordinate_frame = RSMCoordinateFrame(parsed.get("coordinate_frame"))
            except (TypeError, ValueError):
                coordinate_frame = None
            valid_frame = (
                coordinate_frame is not None
                and parsed.get("axis_names") == list(coordinate_frame.axis_names)
                and parsed.get("axis_units") == list(coordinate_frame.axis_units)
                and parsed.get("matrix_policy") == coordinate_frame.matrix_policy
            )
        if (
            set(parsed) != rsm_top_keys
            or attestation_version
            not in {
                "rsm-execution-attestation-v1",
                "rsm-execution-attestation-v2",
            }
            or not valid_frame
            or parsed.get("module_request_fingerprint") != request_fingerprint
            or parsed.get("result_projection_policy")
            != _STORED_RESULT_PROJECTION_POLICY
            or type(parsed.get("result_fingerprint")) is not str
            or _SHA256.fullmatch(parsed["result_fingerprint"]) is None
            or any(
                type(parsed.get(name)) is not str
                or _SHA256.fullmatch(parsed[name]) is None
                for name in (
                    "geometry_asset_receipt_fingerprint",
                    "effective_geometry_fingerprint",
                    "common_grid_fingerprint",
                )
            )
            or type(selected_scans) is not int
            or not 1 <= selected_scans <= 16
            or type(selected) is not int
            or not 1 <= selected <= 4096
            or selected_scans > selected
            or type(science_chunks) is not int
            or not selected_scans <= science_chunks <= selected
            or type(q_released) is not int
            or q_released != science_chunks
            or type(frame_released) is not int
            or frame_released != selected
            or parsed.get("release_check_passed") is not True
            or type(masks) is not list
            or len(masks) != selected_scans
            or any(
                not valid_mask(item, ordinal)
                for ordinal, item in enumerate(masks)
            )
            or len(
                {
                    item["member_preflight_fingerprint"]
                    for item in masks
                }
            )
            != selected_scans
            or len(
                {item["mask_receipt_fingerprint"] for item in masks}
            )
            != selected_scans
            or not valid_runtime_projection(runtime)
        ):
            raise ValueError("analysis execution attestation contract is invalid")
        return text, parsed
    if (
        kind not in {AnalysisArtifactKind.STITCH_1D, AnalysisArtifactKind.STITCH_2D}
        or set(parsed) != top_keys
        or parsed.get("schema_version") != "analysis-execution-attestation-v1"
        or parsed.get("module_request_fingerprint") != request_fingerprint
        or parsed.get("result_projection_policy")
        != _STORED_RESULT_PROJECTION_POLICY
        or type(parsed.get("result_fingerprint")) is not str
        or _SHA256.fullmatch(parsed["result_fingerprint"]) is None
        or type(selected) is not int
        or selected < 1
        or type(released) is not int
        or released != selected
        or parsed.get("release_check_passed") is not True
        or parsed.get("q_root_policy")
        != "shared_ultimate_ndarray_root_weakref_v1"
        or not valid_runtime_projection(runtime)
    ):
        raise ValueError("analysis execution attestation contract is invalid")
    return text, parsed


def analysis_execution_attestation_digest(
    kind: AnalysisArtifactKind,
    value: Mapping[str, object],
    *,
    request_fingerprint: str,
) -> str:
    """Return the exact closed execution-attestation identity domain."""

    if type(kind) is not AnalysisArtifactKind:
        raise TypeError("analysis attestation kind must be exact")
    _sha256(request_fingerprint, "request fingerprint")
    _text, parsed = _validated_execution_attestation(
        value,
        kind=kind,
        request_fingerprint=request_fingerprint,
    )
    # Importing the engine-light transaction enum lazily avoids the artifact /
    # transaction import cycle while preserving the frozen Enum framing.
    from xrd_tools.analysis.module_transaction import ModuleKind
    from xrd_tools.analysis.scan_operations import analysis_canonical_fingerprint

    if kind in {AnalysisArtifactKind.STITCH_1D, AnalysisArtifactKind.STITCH_2D}:
        # a Stitch 2D's attestation binds its own request (module_request_fingerprint)
        module_kind = ModuleKind.STITCH
        domain = "analysis-artifact-execution-attestation-v1"
    elif kind is AnalysisArtifactKind.RSM:
        module_kind = ModuleKind.RSM
        domain = (
            "analysis-artifact-rsm-execution-attestation-v2"
            if parsed.get("schema_version") == "rsm-execution-attestation-v2"
            else "analysis-artifact-rsm-execution-attestation-v1"
        )
    else:  # guarded by the exact validator above
        raise ValueError("analysis execution attestation kind is unsupported")
    return analysis_canonical_fingerprint(
        domain,
        (module_kind, parsed),
    )


def _rsm_frame_provenance_matches(
    provenance_json: str,
    attestation: Mapping[str, object],
) -> bool:
    """Return whether a frame-aware RSM attestation matches provenance."""

    if attestation.get("schema_version") != "rsm-execution-attestation-v2":
        return True
    from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame

    try:
        frame = RSMCoordinateFrame(attestation.get("coordinate_frame"))
        provenance = json.loads(provenance_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return type(provenance) is dict and provenance.get("coordinate_frame") == {
        "name": frame.value,
        "axis_names": list(frame.axis_names),
        "axis_units": list(frame.axis_units),
        "matrix_policy": frame.matrix_policy,
    }


@dataclass(eq=False, frozen=True, slots=True)
class AnalysisArtifactRequest:
    target: str | Path
    kind: AnalysisArtifactKind
    overwrite: AnalysisArtifactOverwrite
    request_fingerprint: str
    source_fingerprint: str
    plan_fingerprint: str
    provenance_digest: str
    provenance: InitVar[Mapping[str, object]]
    provenance_json: str = field(init=False, repr=False)
    schema_version: int = ANALYSIS_SCHEMA_VERSION
    execution_attestation_digest: str | None = None
    execution_attestation: InitVar[Mapping[str, object] | None] = None
    execution_attestation_json: str | None = field(init=False, default=None, repr=False)
    module_owner: InitVar[object | None] = None
    _module_owner: object | None = field(init=False, default=None, repr=False)

    def __post_init__(
        self,
        provenance: Mapping[str, object],
        execution_attestation: Mapping[str, object] | None,
        module_owner: object | None,
    ) -> None:
        if (
            type(self.kind) is not AnalysisArtifactKind
            or type(self.overwrite) is not AnalysisArtifactOverwrite
        ):
            raise TypeError("analysis artifact request enums must be exact")
        object.__setattr__(self, "_module_owner", module_owner)
        object.__setattr__(self, "target", _normalize_target(self.target))
        for value, name in (
            (self.request_fingerprint, "request fingerprint"),
            (self.source_fingerprint, "source fingerprint"),
            (self.plan_fingerprint, "plan fingerprint"),
            (self.provenance_digest, "provenance digest"),
        ):
            _sha256(value, name)
        object.__setattr__(
            self,
            "provenance_json",
            canonical_analysis_provenance(provenance),
        )
        if type(self.schema_version) is not int or self.schema_version not in {
            ANALYSIS_SCHEMA_VERSION,
            ANALYSIS_SCHEMA_VERSION_V2,
            ANALYSIS_SCHEMA_VERSION_V3,
            ANALYSIS_SCHEMA_VERSION_STITCH_NEUTRAL,
            ANALYSIS_SCHEMA_VERSION_XU_STITCH_NEUTRAL,
            ANALYSIS_SCHEMA_VERSION_STITCH_NATIVE,
            ANALYSIS_SCHEMA_VERSION_XU_STITCH_NATIVE,
        }:
            raise TypeError("analysis artifact schema version is unsupported")
        if self.schema_version in _STITCH_NEUTRAL_CONTRACTS and (
            self.kind not in {AnalysisArtifactKind.STITCH_1D, AnalysisArtifactKind.STITCH_2D}
            or (self.schema_version in _STITCH_1D_ONLY_VERSIONS
                and self.kind is not AnalysisArtifactKind.STITCH_1D)
        ):
            raise ValueError("neutral analysis artifact requires its Stitch kind")
        # the old Stitch 2D layout was removed (2026-09-30): a Stitch 2D is a native record
        if self.kind is AnalysisArtifactKind.STITCH_2D and self.schema_version not in _STITCH_NATIVE_VERSIONS:
            raise ValueError("a Stitch 2D artifact is a native record (version 6 or 7)")
        contract_version = _artifact_contract_version(self.schema_version)
        if contract_version == ANALYSIS_SCHEMA_VERSION:
            if (
                self.execution_attestation_digest is not None
                or execution_attestation is not None
            ):
                raise ValueError("analysis artifact v1 cannot carry an attestation")
            return
        if (
            type(self.execution_attestation_digest) is not str
            or execution_attestation is None
        ):
            raise ValueError("attested analysis artifact requires an execution attestation")
        _sha256(
            self.execution_attestation_digest,
            "execution attestation digest",
        )
        text, parsed = _validated_execution_attestation(
            execution_attestation,
            kind=self.kind,
            request_fingerprint=self.request_fingerprint,
        )
        observed = analysis_execution_attestation_digest(
            self.kind,
            parsed,
            request_fingerprint=self.request_fingerprint,
        )
        if observed != self.execution_attestation_digest:
            raise ValueError("analysis execution attestation digest changed")
        if (
            self.schema_version == ANALYSIS_SCHEMA_VERSION_V3
            and (
                self.kind is not AnalysisArtifactKind.RSM
                or parsed.get("schema_version") != "rsm-execution-attestation-v2"
                or parsed.get("coordinate_frame") != "q_sample_cartesian_xu"
            )
        ):
            raise ValueError("analysis artifact v3 requires the RSM frame contract")
        if (
            self.schema_version == ANALYSIS_SCHEMA_VERSION_V2
            and self.kind is AnalysisArtifactKind.RSM
            and not (
                parsed.get("schema_version") == "rsm-execution-attestation-v1"
                or (
                    parsed.get("schema_version")
                    == "rsm-execution-attestation-v2"
                    and parsed.get("coordinate_frame") == "hkl"
                )
            )
        ):
            raise ValueError("analysis artifact v2 requires an HKL contract")
        if (
            self.kind is AnalysisArtifactKind.RSM
            and not _rsm_frame_provenance_matches(self.provenance_json, parsed)
        ):
            raise ValueError(
                "analysis RSM frame provenance and attestation disagree"
            )
        object.__setattr__(self, "execution_attestation_json", text)


@dataclass(frozen=True, slots=True)
class AnalysisArtifactInspection:
    path: str
    storage_revision: tuple[int, int, int, int, int]
    kind: AnalysisArtifactKind
    group: str
    shape: tuple[int, ...]
    axes: tuple[tuple[str, int], ...]
    axis_units: tuple[tuple[str, str | None], ...]
    has_sigma: bool
    has_stitch_diagnostics: bool
    request_fingerprint: str
    source_fingerprint: str
    plan_fingerprint: str
    provenance_digest: str
    provenance_json: str = field(repr=False)
    provenance_sha256: str
    result_fingerprint: str
    schema_version: int = ANALYSIS_SCHEMA_VERSION
    execution_attestation_digest: str | None = None
    execution_attestation_json: str | None = field(default=None, repr=False)
    wavelength_A: float | None = None
    frame_manifest_json: str | None = field(default=None, repr=False)


def _profile_valid(profile, kind) -> bool:
    """A Stitch 2D's I(q) as a projection or payload carries it: None, or a 1-D projection beside a map."""
    return profile is None or (type(profile) is AnalysisArtifactResultProjection
                               and kind is AnalysisArtifactKind.STITCH_2D
                               and profile.kind is AnalysisArtifactKind.STITCH_1D)


def _gi_map_valid(gi_map, kind, cuts) -> bool:
    """A GI Stitch 2D's q_ip-q_oop map: None, or a 2-D projection beside a map with GI cuts."""
    return gi_map is None or (type(gi_map) is AnalysisArtifactResultProjection
                              and kind is AnalysisArtifactKind.STITCH_2D and bool(cuts)
                              and gi_map.kind is AnalysisArtifactKind.STITCH_2D)


def _gi_cuts_valid(cuts) -> bool:
    """GI cuts as a projection or payload carries them: (mode, projection) pairs, distinct modes in
    ``GI_MODES`` order."""
    return (type(cuts) is tuple
            and all(type(cut) is tuple and len(cut) == 2 and cut[0] in _STITCH_GI_MODES
                    and type(cut[1]) is AnalysisArtifactResultProjection for cut in cuts)
            and [mode for mode, _cut in cuts] == [mode for mode in _STITCH_GI_MODES
                                                   if mode in {mode for mode, _cut in cuts}])


@dataclass(eq=False, frozen=True, slots=True)
class AnalysisArtifactResultProjection:
    """The one exact little-endian float32 projection stored in an artifact."""

    kind: AnalysisArtifactKind
    axes: tuple[tuple[str, np.ndarray], ...]
    axis_units: tuple[tuple[str, str | None], ...]
    intensity: np.ndarray
    sigma: np.ndarray | None
    coverage: np.ndarray | None
    normalization: np.ndarray | None
    result_fingerprint: str
    policy: str = field(init=False, default=_STORED_RESULT_PROJECTION_POLICY)
    _claim: InitVar[object] = None
    wavelength_A: float | None = None
    frame_manifest_json: str | None = field(default=None, repr=False)
    #: A GI Stitch 1D's cuts beside I(q), each (its mode, its own 1-D projection) in
    #: ``GI_MODES`` order; () otherwise.
    gi: tuple[tuple[str, "AnalysisArtifactResultProjection"], ...] = ()
    #: A native Stitch 2D's I(q) beside the map, its own 1-D projection; None otherwise.
    profile: "AnalysisArtifactResultProjection | None" = None
    #: A GI Stitch 2D's q_ip-q_oop map, its own 2-D projection (the q_ip and q_oop axes in the q and chi
    #: slots); None otherwise.
    gi_map: "AnalysisArtifactResultProjection | None" = None

    def __post_init__(self, _claim: object) -> None:
        if (not _gi_cuts_valid(self.gi) or not _profile_valid(self.profile, self.kind)
                or not _gi_map_valid(self.gi_map, self.kind, self.gi)):
            raise TypeError("analysis artifact result projection is invalid")
        arrays = tuple(values for _name, values in self.axes) + (
            self.intensity,
        ) + (() if self.sigma is None else (self.sigma,)) + (
            ()
            if self.coverage is None
            else (self.coverage, self.normalization)
        )
        if (
            _claim is not _ANALYSIS_PROJECTION_FACTORY
            or type(self.kind) is not AnalysisArtifactKind
            or type(self.axes) is not tuple
            or type(self.axis_units) is not tuple
            or type(self.intensity) is not np.ndarray
            or (self.sigma is not None and type(self.sigma) is not np.ndarray)
            or (self.coverage is None) is not (self.normalization is None)
            or any(
                type(values) is not np.ndarray
                or values.dtype != np.dtype("<f4")
                or not values.flags.c_contiguous
                or values.flags.writeable
                for values in arrays
            )
        ):
            raise TypeError("analysis artifact result projection is invalid")
        _sha256(self.result_fingerprint, "result fingerprint")

    def __copy__(self):
        raise TypeError("analysis artifact result projection is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("analysis artifact result projection is not copyable")

    def __reduce__(self):
        raise TypeError("analysis artifact result projection is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("analysis artifact result projection is not serializable")


@dataclass(eq=False, frozen=True, slots=True)
class AnalysisArtifactPayload:
    """Detached scientific bytes with fresh read-only NumPy views."""

    inspection: AnalysisArtifactInspection
    _axis_values: InitVar[tuple[tuple[str, np.ndarray], ...]]
    _intensity_values: InitVar[np.ndarray]
    _sigma_values: InitVar[np.ndarray | None]
    _coverage_values: InitVar[np.ndarray | None]
    _normalization_values: InitVar[np.ndarray | None]
    _claim: InitVar[object] = None
    _axis_buffers: tuple[tuple[str, bytes], ...] = field(init=False, repr=False)
    _intensity_buffer: bytes = field(init=False, repr=False)
    _sigma_buffer: bytes | None = field(init=False, repr=False)
    _coverage_buffer: bytes | None = field(init=False, repr=False)
    _normalization_buffer: bytes | None = field(init=False, repr=False)
    #: A GI Stitch 1D's cuts beside I(q), each (its mode, its read-only projection) in
    #: ``GI_MODES`` order; () otherwise.
    gi: tuple[tuple[str, AnalysisArtifactResultProjection], ...] = ()
    #: A native Stitch 2D's I(q) beside the map, its read-only 1-D projection; None otherwise.
    profile: AnalysisArtifactResultProjection | None = None
    #: A GI Stitch 2D's q_ip-q_oop map, its read-only 2-D projection; None otherwise.
    gi_map: AnalysisArtifactResultProjection | None = None
    #: A Stitch result's I(q) angular support as saved (origin ``saved``), or the reason it has none
    #: (an older Stitch result: ``legacy_stitch``); None for an artifact without a native radial profile.
    angular_support: Any = None

    def __post_init__(
        self,
        _axis_values: tuple[tuple[str, np.ndarray], ...],
        _intensity_values: np.ndarray,
        _sigma_values: np.ndarray | None,
        _coverage_values: np.ndarray | None,
        _normalization_values: np.ndarray | None,
        _claim: object,
    ) -> None:
        if not _gi_cuts_valid(self.gi) or (
                self.gi and self.inspection.group not in {_STITCH_RECORD_RESULT, _STITCH_RECORD_MAP}):
            raise TypeError("analysis artifact payload GI cut is invalid")
        if not _gi_map_valid(self.gi_map, self.inspection.kind, self.gi):
            raise TypeError("analysis artifact payload GI map is invalid")
        if not _profile_valid(self.profile, self.inspection.kind) or (
                self.profile is not None and self.inspection.group != _STITCH_RECORD_MAP):
            raise TypeError("analysis artifact payload profile is invalid")
        if (
            _claim is not _ANALYSIS_PAYLOAD_FACTORY
            or type(self.inspection) is not AnalysisArtifactInspection
            or type(_axis_values) is not tuple
            or type(_intensity_values) is not np.ndarray
            or (
                _sigma_values is not None
                and type(_sigma_values) is not np.ndarray
            )
            or (
                _coverage_values is not None
                and type(_coverage_values) is not np.ndarray
            )
            or (
                _normalization_values is not None
                and type(_normalization_values) is not np.ndarray
            )
        ):
            raise TypeError("analysis artifact payload is invalid")
        expected_axes = self.inspection.axes
        if (
            len(_axis_values) != len(expected_axes)
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not np.ndarray
                or item[0] != expected_axes[index][0]
                or item[1].shape != (expected_axes[index][1],)
                or item[1].dtype != np.dtype(np.float32)
                or item[1].flags.writeable
                or not item[1].flags.c_contiguous
                for index, item in enumerate(_axis_values)
            )
            or _intensity_values.shape != self.inspection.shape
            or _intensity_values.dtype != np.dtype(np.float32)
            or _intensity_values.flags.writeable
            or not _intensity_values.flags.c_contiguous
            or (_sigma_values is None) is self.inspection.has_sigma
            or (_coverage_values is None)
            is not (_normalization_values is None)
            or (_coverage_values is not None)
            is not self.inspection.has_stitch_diagnostics
            or (
                _sigma_values is not None
                and (
                    _sigma_values.shape != self.inspection.shape
                    or _sigma_values.dtype != np.dtype(np.float32)
                    or _sigma_values.flags.writeable
                    or not _sigma_values.flags.c_contiguous
                )
            )
            or any(
                values is not None
                and (
                    values.shape != self.inspection.shape
                    or values.dtype != np.dtype(np.float32)
                    or values.flags.writeable
                    or not values.flags.c_contiguous
                    or not np.isfinite(values).all()
                    or np.any(values < 0)
                )
                for values in (_coverage_values, _normalization_values)
            )
        ):
            raise TypeError("analysis artifact payload arrays are invalid")
        object.__setattr__(
            self,
            "_axis_buffers",
            tuple(
                (name, values.tobytes(order="C"))
                for name, values in _axis_values
            ),
        )
        object.__setattr__(
            self,
            "_intensity_buffer",
            _intensity_values.tobytes(order="C"),
        )
        object.__setattr__(
            self,
            "_sigma_buffer",
            None if _sigma_values is None else _sigma_values.tobytes(order="C"),
        )
        object.__setattr__(
            self,
            "_coverage_buffer",
            None
            if _coverage_values is None
            else _coverage_values.tobytes(order="C"),
        )
        object.__setattr__(
            self,
            "_normalization_buffer",
            None
            if _normalization_values is None
            else _normalization_values.tobytes(order="C"),
        )

    @property
    def kind(self) -> AnalysisArtifactKind:
        return self.inspection.kind

    @property
    def provenance_json(self) -> str:
        return self.inspection.provenance_json

    @property
    def result_fingerprint(self) -> str:
        return self.inspection.result_fingerprint

    @property
    def schema_version(self) -> int:
        return self.inspection.schema_version

    @property
    def execution_attestation_digest(self) -> str | None:
        return self.inspection.execution_attestation_digest

    @property
    def execution_attestation_json(self) -> str | None:
        return self.inspection.execution_attestation_json

    @staticmethod
    def _view(buffer: bytes, shape: tuple[int, ...]) -> np.ndarray:
        return np.frombuffer(buffer, dtype=np.float32).reshape(shape)

    @property
    def axes(self) -> tuple[tuple[str, np.ndarray], ...]:
        return tuple(
            (
                name,
                self._view(buffer, (self.inspection.axes[index][1],)),
            )
            for index, (name, buffer) in enumerate(self._axis_buffers)
        )

    @property
    def wavelength_A(self) -> float | None:
        return self.inspection.wavelength_A

    @property
    def native_unit(self) -> str | None:
        return dict(self.inspection.axis_units).get("q")

    @property
    def resident_bytes(self) -> int:
        """What this loaded result holds in memory: its stored arrays (the result, its axes and
        diagnostics, a GI cut's, a Stitch 2D's I(q) and GI map) and its provenance and frame records."""
        def arrays(projection):
            values = [values for _name, values in projection.axes] + [
                value for value in (projection.intensity, projection.sigma, projection.coverage,
                                    projection.normalization) if value is not None]
            return sum(int(np.asarray(value).nbytes) for value in values)
        total = sum(len(buffer) for buffer in (self._intensity_buffer, self._sigma_buffer,
                                               self._coverage_buffer, self._normalization_buffer) if buffer)
        total += sum(len(buffer) for _name, buffer in self._axis_buffers)
        total += sum(arrays(projection) for projection in (
            self.profile, self.gi_map, *(projection for _mode, projection in self.gi)) if projection is not None)
        texts = (self.inspection.provenance_json, self.inspection.frame_manifest_json)
        return total + sum(len(text) for text in texts if text)

    @property
    def contributing_frames(self) -> tuple[dict, ...]:
        text = self.inspection.frame_manifest_json
        return () if text is None else tuple(json.loads(text)["records"])

    def radial_axis(self, unit: str) -> np.ndarray:
        if self.kind is not AnalysisArtifactKind.STITCH_1D:
            raise ValueError("radial coordinate switch requires Stitch 1D")
        if unit == self.native_unit:
            return self.axis("q")
        from xrd_tools.io.analysis_stitch_auxiliary import native_stitch_companion
        target, values = native_stitch_companion(self.axis("q"), self.native_unit, self.wavelength_A)
        if unit != target:
            raise ValueError("unsupported Stitch viewer coordinate")
        return values

    @property
    def intensity(self) -> np.ndarray:
        return self._view(self._intensity_buffer, self.inspection.shape)

    @property
    def sigma(self) -> np.ndarray | None:
        if self._sigma_buffer is None:
            return None
        return self._view(self._sigma_buffer, self.inspection.shape)

    @property
    def coverage(self) -> np.ndarray | None:
        if self._coverage_buffer is None:
            return None
        return self._view(self._coverage_buffer, self.inspection.shape)

    @property
    def normalization(self) -> np.ndarray | None:
        if self._normalization_buffer is None:
            return None
        return self._view(self._normalization_buffer, self.inspection.shape)

    def axis(self, name: str) -> np.ndarray:
        if type(name) is not str:
            raise TypeError("analysis artifact axis name must be an exact string")
        for index, (axis_name, buffer) in enumerate(self._axis_buffers):
            if axis_name == name:
                return self._view(buffer, (self.inspection.axes[index][1],))
        raise KeyError(name)

    def __copy__(self):
        raise TypeError("analysis artifact payload is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("analysis artifact payload is not copyable")

    def __reduce__(self):
        raise TypeError("analysis artifact payload is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("analysis artifact payload is not serializable")


@dataclass(eq=False, frozen=True, slots=True)
class AnalysisArtifactReceipt:
    request: AnalysisArtifactRequest
    terminal: StreamTerminal
    inspection: AnalysisArtifactInspection
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _ANALYSIS_RECEIPT_FACTORY
            or type(self.request) is not AnalysisArtifactRequest
            or type(self.terminal) is not StreamTerminal
            or self.terminal.target != self.request.target
            or type(self.inspection) is not AnalysisArtifactInspection
            or self.inspection.path != self.request.target
            or self.inspection.storage_revision
            != stream_terminal_object_revision(self.terminal)
            or self.inspection.kind is not self.request.kind
            or self.inspection.request_fingerprint
            != self.request.request_fingerprint
            or self.inspection.source_fingerprint
            != self.request.source_fingerprint
            or self.inspection.plan_fingerprint != self.request.plan_fingerprint
            or self.inspection.provenance_digest
            != self.request.provenance_digest
            or self.inspection.schema_version != self.request.schema_version
            or self.inspection.execution_attestation_digest
            != self.request.execution_attestation_digest
            or self.inspection.execution_attestation_json
            != self.request.execution_attestation_json
        ):
            raise TypeError("analysis artifact receipt is invalid")

    def __copy__(self):
        raise TypeError("analysis artifact receipt is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("analysis artifact receipt is not copyable")

    def __reduce__(self):
        raise TypeError("analysis artifact receipt is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("analysis artifact receipt is not serializable")


@dataclass(frozen=True, slots=True)
class AnalysisArtifactOutputSnapshot:
    target: str
    writer_started: bool
    writer_finished: bool
    published: bool
    close_pending: bool
    slot_held: bool
    retryable: bool
    receipt: AnalysisArtifactReceipt | None = None
    diagnostics: tuple[str, ...] = ()
    hidden_orphan: str | None = None


def _decode_fixed_text(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="strict")
    raise AnalysisArtifactInvalid("analysis artifact attribute is not fixed text")


def _hdf_type_class(identifier) -> int:
    data_type = identifier.get_type()
    try:
        return int(data_type.get_class())
    finally:
        data_type.close()


def _bounded_attr_text(owner, name: str) -> str:
    try:
        attribute = owner.attrs.get_id(name)
    except KeyError as error:
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is missing"
        ) from error
    if (
        attribute.shape != ()
        or _hdf_type_class(attribute) != h5py.h5t.STRING
        or attribute.dtype.kind != "S"
        or attribute.dtype.itemsize < 1
        or attribute.dtype.itemsize > _MAX_ATTRIBUTE_BYTES
        or attribute.get_storage_size() > _MAX_ATTRIBUTE_BYTES
    ):
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is not bounded fixed text"
        )
    return _decode_fixed_text(owner.attrs[name])


def _exact_sha256_attr_text(owner, name: str) -> str:
    """Read one v2-only digest with no fixed-width NUL padding."""

    try:
        attribute = owner.attrs.get_id(name)
    except KeyError as error:
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is missing"
        ) from error
    if (
        attribute.shape != ()
        or _hdf_type_class(attribute) != h5py.h5t.STRING
        or attribute.dtype.kind != "S"
        or attribute.dtype.itemsize != 64
        or attribute.get_storage_size() != 64
    ):
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is not exact digest text"
        )
    value = _decode_fixed_text(owner.attrs[name])
    if len(value.encode("utf-8")) != 64:
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is padded digest text"
        )
    return value


def _bounded_attr_texts(owner, name: str) -> tuple[str, ...]:
    try:
        attribute = owner.attrs.get_id(name)
    except KeyError as error:
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is missing"
        ) from error
    if (
        len(attribute.shape) != 1
        or attribute.shape[0] < 1
        or attribute.shape[0] > _MAX_ATTRIBUTE_ITEMS
        or _hdf_type_class(attribute) != h5py.h5t.STRING
        or attribute.dtype.kind != "S"
        or attribute.dtype.itemsize < 1
        or attribute.dtype.itemsize > _MAX_ATTRIBUTE_BYTES
        or attribute.get_storage_size()
        > _MAX_ATTRIBUTE_ITEMS * _MAX_ATTRIBUTE_BYTES
    ):
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is not bounded fixed text"
        )
    values = owner.attrs[name]
    return tuple(_decode_fixed_text(value) for value in values.tolist())


def _direct(group: h5py.Group, name: str, expected_type):
    try:
        link_type = group.id.links.get_info(name.encode("utf-8")).type
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise AnalysisArtifactInvalid(f"{group.name}/{name} is missing") from error
    if link_type != h5py.h5l.TYPE_HARD:
        raise AnalysisArtifactInvalid(f"{group.name}/{name} is not a local hard link")
    value = group.get(name)
    if not isinstance(value, expected_type):
        raise AnalysisArtifactInvalid(f"{group.name}/{name} has the wrong object type")
    if isinstance(value, h5py.Dataset):
        creation = value.id.get_create_plist()
        try:
            indirect = value.is_virtual or creation.get_external_count() != 0
        finally:
            creation.close()
        if indirect:
            raise AnalysisArtifactInvalid(f"{value.name} uses indirect storage")
    return value


def _object_address(value: h5py.Group | h5py.Dataset) -> int:
    return int(h5py.h5o.get_info(value.id).addr)


def _require_local_graph(group: h5py.Group, seen: set[int]) -> None:
    address = _object_address(group)
    if address in seen:
        raise AnalysisArtifactInvalid(f"hard-link alias detected at {group.name}")
    seen.add(address)
    for name in group:
        value = _direct(group, name, (h5py.Group, h5py.Dataset))
        address = _object_address(value)
        if address in seen:
            raise AnalysisArtifactInvalid(f"hard-link alias detected at {value.name}")
        if isinstance(value, h5py.Group):
            _require_local_graph(value, seen)
        else:
            seen.add(address)


def _read_scalar_text(group: h5py.Group, name: str) -> str:
    dataset = _direct(group, name, h5py.Dataset)
    if dataset.shape != ():
        raise AnalysisArtifactInvalid(f"{dataset.name} must be a scalar string")
    if len(dataset.attrs) != 0:
        raise AnalysisArtifactInvalid(f"{dataset.name} has unexpected attributes")
    if (
        _hdf_type_class(dataset.id) != h5py.h5t.STRING
        or dataset.dtype.kind != "S"
    ):
        raise AnalysisArtifactInvalid(
            "analysis provenance must use bounded fixed-width storage"
        )
    if (
        dataset.dtype.itemsize < 1
        or dataset.dtype.itemsize > _MAX_PROVENANCE_BYTES
        or dataset.id.get_storage_size() > _MAX_PROVENANCE_BYTES
    ):
        raise AnalysisArtifactInvalid("analysis provenance is oversized")
    raw = dataset[()]
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="strict")
    elif isinstance(raw, str):
        text = raw
    else:
        raise AnalysisArtifactInvalid(f"{dataset.name} must be UTF-8 text")
    if len(text.encode("utf-8")) > _MAX_PROVENANCE_BYTES:
        raise AnalysisArtifactInvalid("analysis provenance is oversized")
    _require_json_nesting_bound(text)
    try:
        parsed = json.loads(text)
    except (RecursionError, TypeError, ValueError) as error:
        raise AnalysisArtifactInvalid("analysis provenance is invalid JSON") from error
    if type(parsed) is not dict:
        raise AnalysisArtifactInvalid("analysis provenance must decode to an object")
    try:
        canonical = canonical_analysis_provenance(parsed)
    except (RecursionError, TypeError, ValueError) as error:
        raise AnalysisArtifactInvalid("analysis provenance is not canonicalizable") from error
    if canonical != text:
        raise AnalysisArtifactInvalid("analysis provenance is not canonical JSON")
    return text


def _require_execution_attestation_nesting_bound(text: str) -> None:
    depth = 0
    quoted = False
    escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_EXECUTION_ATTESTATION_NESTING:
                raise AnalysisArtifactInvalid(
                    "analysis execution attestation exceeds the nesting bound"
                )
        elif character in "]}":
            depth -= 1
            if depth < 0:
                break


def _read_scalar_execution_attestation(
    group: h5py.Group,
    name: str,
    *,
    kind: AnalysisArtifactKind,
    request_fingerprint: str,
    expected_digest: str,
) -> str:
    dataset = _direct(group, name, h5py.Dataset)
    if dataset.shape != () or len(dataset.attrs) != 0:
        raise AnalysisArtifactInvalid(
            "analysis execution attestation must be an attribute-free scalar"
        )
    if (
        _hdf_type_class(dataset.id) != h5py.h5t.STRING
        or dataset.dtype.kind != "S"
        or dataset.dtype.itemsize < 1
        or dataset.dtype.itemsize > _MAX_EXECUTION_ATTESTATION_BYTES
        or dataset.id.get_storage_size() > _MAX_EXECUTION_ATTESTATION_BYTES
    ):
        raise AnalysisArtifactInvalid(
            "analysis execution attestation must use bounded fixed-width storage"
        )
    raw = dataset[()]
    if not isinstance(raw, bytes):
        raise AnalysisArtifactInvalid(
            "analysis execution attestation must be UTF-8 text"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise AnalysisArtifactInvalid(
            "analysis execution attestation must be UTF-8 text"
        ) from error
    if len(text.encode("utf-8")) > _MAX_EXECUTION_ATTESTATION_BYTES:
        raise AnalysisArtifactInvalid("analysis execution attestation is oversized")
    if (
        dataset.dtype.itemsize != len(text.encode("utf-8"))
        or dataset.id.get_storage_size() != len(text.encode("utf-8"))
    ):
        raise AnalysisArtifactInvalid(
            "analysis execution attestation storage is not byte-exact"
        )
    _require_execution_attestation_nesting_bound(text)
    try:
        parsed = json.loads(text)
        canonical, frozen = _validated_execution_attestation(
            parsed,
            kind=kind,
            request_fingerprint=request_fingerprint,
        )
    except (RecursionError, TypeError, ValueError) as error:
        raise AnalysisArtifactInvalid(
            "analysis execution attestation is invalid"
        ) from error
    if canonical != text:
        raise AnalysisArtifactInvalid(
            "analysis execution attestation is not canonical JSON"
        )
    if analysis_execution_attestation_digest(
        kind,
        frozen,
        request_fingerprint=request_fingerprint,
    ) != expected_digest:
        raise AnalysisArtifactInvalid(
            "analysis execution attestation digest changed"
        )
    return text


def _require_json_nesting_bound(text: str) -> None:
    depth = 0
    quoted = False
    escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_PROVENANCE_NESTING:
                raise AnalysisArtifactInvalid(
                    "analysis provenance exceeds the nesting bound"
                )
        elif character in "]}":
            depth -= 1
            if depth < 0:
                break


def _axis(group: h5py.Group, name: str) -> np.ndarray:
    dataset = _direct(group, name, h5py.Dataset)
    if (
        _hdf_type_class(dataset.id) != h5py.h5t.FLOAT
        or dataset.dtype != np.dtype(np.float32)
        or dataset.ndim != 1
        or dataset.size < 1
        or dataset.size > _MAX_AXIS_POINTS
    ):
        raise AnalysisArtifactInvalid(f"{dataset.name} is not a nonempty float32 axis")
    _require_bounded_chunk(dataset)
    values = np.asarray(dataset[()], dtype=np.float32)
    if not np.isfinite(values).all() or (
        values.size > 1 and not np.all(np.diff(values.astype(np.float64)) > 0)
    ):
        raise AnalysisArtifactInvalid(f"{dataset.name} must be finite and increasing")
    return values


def _digest_text(digest, value: str) -> None:
    raw = value.encode("utf-8")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def _digest_array(digest, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype="<f4")
    _digest_text(digest, name)
    digest.update(array.ndim.to_bytes(8, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(memoryview(array).cast("B"))


def _require_bounded_chunk(dataset: h5py.Dataset) -> None:
    if dataset.chunks is not None and (
        math.prod(int(dimension) for dimension in dataset.chunks)
        * int(dataset.dtype.itemsize)
        > _MAX_READ_BLOCK_BYTES
    ):
        raise AnalysisArtifactInvalid(
            f"{dataset.name} storage chunks exceed the read bound"
        )


def _require_bounded_values(
    dataset: h5py.Dataset,
    *,
    digest=None,
    digest_name: str = "",
    require_finite: bool = False,
    require_nonnegative: bool = False,
) -> None:
    if (
        _hdf_type_class(dataset.id) != h5py.h5t.FLOAT
        or dataset.dtype != np.dtype(np.float32)
        or dataset.size < 1
        or dataset.size > _MAX_RESULT_ELEMENTS
        or dataset.ndim < 1
    ):
        raise AnalysisArtifactInvalid(f"{dataset.name} must be nonempty float32")
    _require_bounded_chunk(dataset)
    finite_seen = False
    row_bytes = max(1, int(np.prod(dataset.shape[1:], dtype=np.int64)) * 4)
    if row_bytes > _MAX_READ_BLOCK_BYTES:
        raise AnalysisArtifactInvalid(f"{dataset.name} rows exceed the read bound")
    rows = max(1, _MAX_READ_BLOCK_BYTES // row_bytes)
    if digest is not None:
        _digest_text(digest, digest_name)
        digest.update(dataset.ndim.to_bytes(8, "big"))
        for dimension in dataset.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
    pieces = (
        dataset[start : min(dataset.shape[0], start + rows)]
        for start in range(0, dataset.shape[0], rows)
    )
    for piece in pieces:
        values = np.asarray(piece)
        if np.isinf(values).any() or (
            require_finite and not np.isfinite(values).all()
        ):
            raise AnalysisArtifactInvalid(f"{dataset.name} contains infinity")
        if require_nonnegative and np.any(values < 0):
            raise AnalysisArtifactInvalid(
                f"{dataset.name} contains negative values"
            )
        finite_seen = finite_seen or bool(np.isfinite(values).any())
        if digest is not None:
            digest.update(
                memoryview(np.ascontiguousarray(values, dtype="<f4")).cast("B")
            )
    if not finite_seen:
        raise AnalysisArtifactInvalid(f"{dataset.name} has no finite result")


def _stat_revision(state: os.stat_result) -> tuple[int, int, int, int, int]:
    """Comparable identity of one stat view (descriptor or pathname)."""
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        identity_ctime_ns(state.st_ctime_ns),
    )


def _opened_hdf_revision(handle: h5py.File) -> tuple[int, int, int, int, int]:
    try:
        raw_handle = handle.id.get_vfd_handle()
        descriptor = raw_handle[0] if type(raw_handle) is tuple else raw_handle
        if isinstance(descriptor, bool):
            raise TypeError("boolean VFD handle")
        return _stat_revision(os.fstat(int(descriptor)))
    except (OSError, OverflowError, TypeError, ValueError) as error:
        raise AnalysisArtifactInvalid(
            "analysis artifact storage handle cannot be fenced"
        ) from error


def _freeze_float32(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.dtype("<f4"))
    frozen = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.dtype("<f4")
    )
    return frozen.reshape(contiguous.shape)


def _project_stored_float32(
    values: object,
    *,
    name: str,
    require_finite: bool,
    require_nonnegative: bool = False,
    max_elements: int = _MAX_RESULT_ELEMENTS,
) -> np.ndarray:
    if (
        type(values) is not np.ndarray
        or values.dtype.kind not in "biuf"
        or values.dtype.fields is not None
        or values.ndim < 1
        or values.size < 1
        or values.size > max_elements
    ):
        raise AnalysisArtifactProjectionInvalid(
            f"{name} is not an exact bounded real ndarray"
        )
    source = np.asarray(values)
    if source.dtype == np.dtype("<f4") and np.isnan(source).any():
        nan_bits = source.view(np.uint32)[np.isnan(source)]
        if np.any(nan_bits != _STORED_QNAN_F4_BITS):
            raise AnalysisArtifactProjectionInvalid(
                f"{name} contains a noncanonical stored NaN payload"
            )
    if np.isinf(source).any() or (require_finite and not np.isfinite(source).all()):
        raise AnalysisArtifactProjectionInvalid(f"{name} contains a nonfinite value")
    if require_nonnegative and np.any(source < 0):
        raise AnalysisArtifactProjectionInvalid(f"{name} contains a negative value")
    with np.errstate(over="ignore", invalid="ignore"):
        projected = np.array(source, dtype=np.dtype("<f4"), order="C", copy=True)
    if np.isinf(projected).any() or (
        require_finite and not np.isfinite(projected).all()
    ):
        raise AnalysisArtifactProjectionInvalid(
            f"{name} overflows the stored float32 representation"
        )
    if not require_finite:
        if not np.isfinite(projected).any():
            raise AnalysisArtifactProjectionInvalid(f"{name} has no finite result")
        projected.view(np.uint32)[np.isnan(projected)] = _STORED_QNAN_F4_BITS
    return _freeze_float32(projected)


def project_analysis_artifact_result(
    *,
    kind: AnalysisArtifactKind,
    axes: tuple[tuple[str, np.ndarray], ...],
    axis_units: tuple[tuple[str, str | None], ...],
    intensity: np.ndarray,
    sigma: np.ndarray | None,
    coverage: np.ndarray | None,
    normalization: np.ndarray | None,
    _stored_v1_claim: object = None,
    wavelength_A: float | None = None,
    frame_records=None,
    source_base=None,
    _frame_manifest_json: str | None = None,
    gi_cuts: tuple = (),
    profile: tuple | None = None,
    gi_map: tuple | None = None,
) -> AnalysisArtifactResultProjection:
    """Project one result exactly once to the bytes stored by the artifact.

    ``gi_cuts`` are a GI Stitch 1D's cuts, each ``(mode, unit, axis, intensity, coverage,
    normalization)`` with distinct modes in ``GI_MODES`` order: each is projected like a Stitch 1D
    result and its fingerprint folds into this one in that order (only when present, so every other
    result keeps its fingerprint, and a one-cut result the fingerprint it had).

    ``profile`` is a native Stitch 2D's I(q) beside the map, ``(unit, axis, intensity, sigma, coverage,
    normalization)``: projected like a Stitch 1D result, its fingerprint folds into this one after the
    map's (only when present). ``gi_map`` is a GI Stitch 2D's q_ip-q_oop map, ``(q_ip unit, q_oop unit,
    q_ip axis, q_oop axis, intensity (q_ip, q_oop), coverage, normalization)``: projected like a Stitch
    2D map, its fingerprint folds in last (only when present).

    The function deliberately retains the v1 result-fingerprint framing.  It
    is shared by runners, writers, strict inspection, and payload hydration so
    a pre-publication fingerprint cannot describe different bytes from those
    admitted after publication.
    """

    if type(kind) is not AnalysisArtifactKind:
        raise TypeError("analysis result kind must be exact AnalysisArtifactKind")
    expected_axis_names = {
        AnalysisArtifactKind.STITCH_1D: ("q",),
        AnalysisArtifactKind.STITCH_2D: ("q", "chi"),
    }.get(kind)
    if kind is AnalysisArtifactKind.RSM:
        axis_name_candidate = (
            tuple(item[0] for item in axes)
            if type(axes) is tuple
            and all(type(item) is tuple and len(item) == 2 for item in axes)
            else ()
        )
        allowed_rsm_axes = {
            ("h", "k", "l"): (None, None, None),
            ("qx", "qy", "qz"): ("q_A^-1", "q_A^-1", "q_A^-1"),
        }
        expected_axis_names = axis_name_candidate
        expected_rsm_units = allowed_rsm_axes.get(axis_name_candidate)
    else:
        expected_rsm_units = None

    def bounded_unit(value: object) -> bool:
        if type(value) is not str or not value or "\x00" in value:
            return False
        try:
            return len(value.encode("utf-8")) <= _MAX_ATTRIBUTE_BYTES
        except UnicodeEncodeError:
            return False

    if (
        type(axes) is not tuple
        or expected_axis_names is None
        or not expected_axis_names
        or len(axes) != len(expected_axis_names)
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or item[0] != expected_axis_names[index]
            or type(item[1]) is not np.ndarray
            for index, item in enumerate(axes)
        )
        or type(axis_units) is not tuple
        or len(axis_units) != len(expected_axis_names)
        or any(
            type(item) is not tuple
            or len(item) != 2
            or item[0] != expected_axis_names[index]
            or (
                kind in {
                    AnalysisArtifactKind.STITCH_1D,
                    AnalysisArtifactKind.STITCH_2D,
                }
                and not bounded_unit(item[1])
            )
            or (
                kind is AnalysisArtifactKind.RSM
                and (
                    expected_rsm_units is None
                    or item[1] != expected_rsm_units[index]
                )
            )
            for index, item in enumerate(axis_units)
        )
        or (coverage is None) is not (normalization is None)
        or (
            kind is AnalysisArtifactKind.RSM
            and (coverage is not None or sigma is not None)
        )
    ):
        raise AnalysisArtifactProjectionInvalid(
            "analysis result axes, units, or optional channels are invalid"
        )
    projected_axes: list[tuple[str, np.ndarray]] = []
    for name, values in axes:
        projected = _project_stored_float32(
            values,
            name=name,
            require_finite=True,
            max_elements=_MAX_AXIS_POINTS,
        )
        if projected.ndim != 1 or (
            projected.size > 1
            and not np.all(np.diff(projected.astype(np.float64)) > 0)
        ):
            raise AnalysisArtifactProjectionInvalid(
                f"{name} is not strictly increasing after float32 projection"
            )
        projected_axes.append((name, projected))
    expected_shape = tuple(values.size for _name, values in projected_axes)
    if (
        max(1, int(np.prod(expected_shape[1:], dtype=np.int64)) * 4)
        > _MAX_READ_BLOCK_BYTES
    ):
        raise AnalysisArtifactProjectionInvalid(
            "analysis result rows exceed the stored read bound"
        )
    projected_intensity = _project_stored_float32(
        intensity,
        name="intensity",
        require_finite=False,
    )
    if projected_intensity.shape != expected_shape:
        raise AnalysisArtifactProjectionInvalid(
            "analysis intensity shape does not match projected axes"
        )
    projected_sigma = None
    if sigma is not None:
        projected_sigma = _project_stored_float32(
            sigma,
            name="sigma",
            require_finite=False,
        )
        if projected_sigma.shape != expected_shape:
            raise AnalysisArtifactProjectionInvalid(
                "analysis sigma shape does not match projected axes"
            )
    projected_coverage = projected_normalization = None
    if coverage is not None:
        if (
            kind in {
                AnalysisArtifactKind.STITCH_1D,
                AnalysisArtifactKind.STITCH_2D,
            }
            and _stored_v1_claim is not _ANALYSIS_V1_DECODE_PROJECTION
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                invalid_counts = (
                    not np.isfinite(coverage).all()
                    or np.any(coverage < 0)
                    or np.any(coverage > 2**24)
                    or np.any(coverage != np.floor(coverage))
                )
            if invalid_counts:
                raise AnalysisArtifactProjectionInvalid(
                    "Stitch coverage must be exact bounded integer counts"
                )
            coverage_f4 = np.asarray(coverage, dtype=np.dtype("<f4"))
            if not np.array_equal(
                coverage_f4.astype(coverage.dtype), coverage
            ):
                raise AnalysisArtifactProjectionInvalid(
                    "Stitch coverage loses count identity in float32 storage"
                )
        projected_coverage = _project_stored_float32(
            coverage,
            name="coverage",
            require_finite=True,
            require_nonnegative=True,
        )
        projected_normalization = _project_stored_float32(
            normalization,
            name="normalization",
            require_finite=True,
            require_nonnegative=True,
        )
        if (
            projected_coverage.shape != expected_shape
            or projected_normalization.shape != expected_shape
        ):
            raise AnalysisArtifactProjectionInvalid(
                "analysis diagnostics shape does not match projected axes"
            )
    digest = hashlib.sha256(_RESULT_FINGERPRINT_PREFIX)
    _digest_text(digest, kind.value)
    for name, values in projected_axes:
        _digest_array(digest, name, values)
    for _name, units in axis_units:
        if units is not None:
            _digest_text(digest, units)
    _digest_array(digest, "intensity", projected_intensity)
    if projected_sigma is None:
        _digest_text(digest, "sigma:absent")
    else:
        _digest_array(digest, "sigma", projected_sigma)
    if projected_coverage is not None:
        _digest_array(digest, "coverage", projected_coverage)
        _digest_array(digest, "normalization", projected_normalization)
    frame_manifest_json = None
    if wavelength_A is not None or frame_records is not None or _frame_manifest_json is not None:
        from xrd_tools.io.analysis_stitch_auxiliary import (
            native_stitch_companion, analysis_contributing_frames_manifest,
            canonical_frame_manifest,
        )
        if kind not in {AnalysisArtifactKind.STITCH_1D, AnalysisArtifactKind.STITCH_2D}:
            raise AnalysisArtifactProjectionInvalid("native auxiliary records require a Stitch result")
        target_unit, companion = native_stitch_companion(
            projected_axes[0][1], axis_units[0][1], wavelength_A,
        )
        if frame_records is not None:
            frame_manifest_json = canonical_frame_manifest(
                analysis_contributing_frames_manifest(frame_records, source_base=source_base)
            )
        elif _frame_manifest_json is not None:
            frame_manifest_json = _frame_manifest_json
        else:
            raise AnalysisArtifactProjectionInvalid("native Stitch requires contributing frames")
        _digest_text(digest, "native-stitch-auxiliary-v1")
        _digest_text(digest, float(wavelength_A).hex())
        _digest_text(digest, frame_manifest_json)
        _digest_array(digest, target_unit, companion)
    if type(gi_cuts) is not tuple or (gi_cuts and kind not in {AnalysisArtifactKind.STITCH_1D,
                                                                AnalysisArtifactKind.STITCH_2D}) or any(
            type(cut) is not tuple or len(cut) != 6 for cut in gi_cuts) or [cut[0] for cut in gi_cuts] != [
            mode for mode in _STITCH_GI_MODES if mode in {cut[0] for cut in gi_cuts}]:
        raise AnalysisArtifactProjectionInvalid(
            f"GI cuts are Stitch 1D's, distinct modes of {_STITCH_GI_MODES} in that order, each with its "
            "unit, axis, intensity, coverage and normalization")
    gi = []
    for mode, unit, cut_axis, cut_intensity, cut_coverage, cut_normalization in gi_cuts:
        if cut_coverage is None or cut_normalization is None:
            raise AnalysisArtifactProjectionInvalid("a GI cut needs its coverage and normalization")
        cut = project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_1D, axes=(("q", cut_axis),), axis_units=(("q", unit),),
            intensity=cut_intensity, sigma=None, coverage=cut_coverage, normalization=cut_normalization,
            _stored_v1_claim=_stored_v1_claim,
        )
        _digest_text(digest, "stitch-gi-cut-v1")
        _digest_text(digest, mode)
        _digest_text(digest, cut.result_fingerprint)
        gi.append((mode, cut))
    if profile is not None:
        if (kind is not AnalysisArtifactKind.STITCH_2D or type(profile) is not tuple or len(profile) != 6
                or profile[4] is None or profile[5] is None):
            raise AnalysisArtifactProjectionInvalid(
                "a profile is a Stitch 2D's I(q): its unit, axis, intensity, sigma, coverage and normalization")
        unit, profile_axis, profile_intensity, profile_sigma, profile_coverage, profile_normalization = profile
        profile = project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_1D, axes=(("q", profile_axis),), axis_units=(("q", unit),),
            intensity=profile_intensity, sigma=profile_sigma, coverage=profile_coverage,
            normalization=profile_normalization, _stored_v1_claim=_stored_v1_claim,
        )
        _digest_text(digest, "stitch-profile-v1")
        _digest_text(digest, profile.result_fingerprint)
    if gi_map is not None:
        if (kind is not AnalysisArtifactKind.STITCH_2D or not gi or type(gi_map) is not tuple or len(gi_map) != 7
                or gi_map[5] is None or gi_map[6] is None):
            raise AnalysisArtifactProjectionInvalid(
                "a GI map is a GI Stitch 2D's: its units, axes, intensity, coverage and normalization")
        ip_unit, oop_unit, ip_axis, oop_axis, map_intensity, map_coverage, map_normalization = gi_map
        gi_map = project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_2D, axes=(("q", ip_axis), ("chi", oop_axis)),
            axis_units=(("q", ip_unit), ("chi", oop_unit)), intensity=map_intensity, sigma=None,
            coverage=map_coverage, normalization=map_normalization, _stored_v1_claim=_stored_v1_claim,
        )
        _digest_text(digest, "stitch-gi-map-v1")
        _digest_text(digest, gi_map.result_fingerprint)
    return AnalysisArtifactResultProjection(
        kind,
        tuple(projected_axes),
        axis_units,
        projected_intensity,
        projected_sigma,
        projected_coverage,
        projected_normalization,
        digest.hexdigest(),
        _ANALYSIS_PROJECTION_FACTORY,
        wavelength_A=wavelength_A,
        frame_manifest_json=frame_manifest_json,
        gi=tuple(gi),
        profile=profile,
        gi_map=gi_map,
    )


def _project_analysis_artifact_result_v1_compat(
    *,
    kind: AnalysisArtifactKind,
    axes: tuple[tuple[str, np.ndarray], ...],
    axis_units: tuple[tuple[str, str | None], ...],
    intensity: np.ndarray,
    sigma: np.ndarray | None,
    coverage: np.ndarray | None,
    normalization: np.ndarray | None,
    wavelength_A: float | None = None,
    frame_records=None,
    source_base=None,
    gi_cuts: tuple = (),
    profile: tuple | None = None,
    gi_map: tuple | None = None,
) -> AnalysisArtifactResultProjection:
    """Use the central projection for byte-compatible historical v1 Stitch."""

    return project_analysis_artifact_result(
        kind=kind,
        axes=axes,
        axis_units=axis_units,
        intensity=intensity,
        sigma=sigma,
        coverage=coverage,
        normalization=normalization,
        _stored_v1_claim=_ANALYSIS_V1_DECODE_PROJECTION,
        wavelength_A=wavelength_A,
        frame_records=frame_records,
        source_base=source_base,
        gi_cuts=gi_cuts,
        profile=profile,
        gi_map=gi_map,
    )


def _materialize_bounded_values(
    dataset: h5py.Dataset,
    *,
    require_finite: bool = False,
    require_nonnegative: bool = False,
) -> np.ndarray:
    if (
        _hdf_type_class(dataset.id) != h5py.h5t.FLOAT
        or dataset.dtype != np.dtype(np.float32)
        or dataset.size < 1
        or dataset.size > _MAX_RESULT_ELEMENTS
        or dataset.ndim < 1
    ):
        raise AnalysisArtifactInvalid(f"{dataset.name} must be nonempty float32")
    _require_bounded_chunk(dataset)
    row_bytes = max(1, int(np.prod(dataset.shape[1:], dtype=np.int64)) * 4)
    if row_bytes > _MAX_READ_BLOCK_BYTES:
        raise AnalysisArtifactInvalid(f"{dataset.name} rows exceed the read bound")
    rows = max(1, _MAX_READ_BLOCK_BYTES // row_bytes)
    values = np.empty(dataset.shape, dtype=np.float32)
    finite_seen = False
    for start in range(0, dataset.shape[0], rows):
        stop = min(dataset.shape[0], start + rows)
        piece = np.asarray(dataset[start:stop], dtype=np.float32)
        if np.isinf(piece).any() or (
            require_finite and not np.isfinite(piece).all()
        ):
            raise AnalysisArtifactInvalid(f"{dataset.name} contains infinity")
        if require_nonnegative and np.any(piece < 0):
            raise AnalysisArtifactInvalid(
                f"{dataset.name} contains negative values"
            )
        finite_seen = finite_seen or bool(np.isfinite(piece).any())
        values[start:stop] = piece
    if not finite_seen:
        raise AnalysisArtifactInvalid(f"{dataset.name} has no finite result")
    return _freeze_float32(values)


# Detached admission and later payload hydration deliberately retain separate
# race seams even though both feed the same public stored-result projection.
_materialize_inspection_values = _materialize_bounded_values


#: A stitched Int record's result group (``stitch_record``): I(q), or a Stitch 2D's map.
_STITCH_RECORD_RESULT = "integrated_1d"
_STITCH_RECORD_MAP = "integrated_2d"
#: Every version a Stitch 1D request may carry (v3 is RSM's).
_STITCH_RECORD_VERSIONS = frozenset({
    ANALYSIS_SCHEMA_VERSION,
    ANALYSIS_SCHEMA_VERSION_V2,
    ANALYSIS_SCHEMA_VERSION_STITCH_NEUTRAL,
    ANALYSIS_SCHEMA_VERSION_XU_STITCH_NEUTRAL,
    ANALYSIS_SCHEMA_VERSION_STITCH_NATIVE,
    ANALYSIS_SCHEMA_VERSION_XU_STITCH_NATIVE,
})


def _stitch_record_projection(handle: h5py.File, version: int):
    """A stitched record's patterns, read through the admitted handle, projected by the one projection."""
    native = version in _STITCH_NATIVE_VERSIONS
    content = _read_stitch_record(handle, entry=_ENTRY)
    carried = (content.wavelength_A is not None, content.frames_json is not None)
    if carried != (native, native):
        raise AnalysisArtifactInvalid(
            "a native stitched record carries its wavelength and contributing frames, a neutral one neither"
        )
    pattern, gi, stitched_map = content.pattern, content.gi, content.map
    if stitched_map is None:
        result = dict(kind=AnalysisArtifactKind.STITCH_1D, axes=(("q", pattern.axis),),
                      axis_units=(("q", pattern.unit),), intensity=pattern.intensity, sigma=pattern.sigma,
                      coverage=pattern.coverage, normalization=pattern.normalization)
    else:  # a Stitch 2D: the map, and I(q) beside it (a GI one's q_ip-q_oop map too)
        gi_map = content.gi_map
        result = dict(kind=AnalysisArtifactKind.STITCH_2D,
                      axes=(("q", stitched_map.radial), ("chi", stitched_map.azimuthal)),
                      axis_units=(("q", stitched_map.unit), ("chi", stitched_map.azimuthal_unit)),
                      intensity=stitched_map.intensity, sigma=stitched_map.sigma,
                      coverage=stitched_map.coverage, normalization=stitched_map.normalization,
                      profile=(pattern.unit, pattern.axis, pattern.intensity, pattern.sigma, pattern.coverage,
                               pattern.normalization),
                      gi_map=None if gi_map is None else (
                          gi_map.unit, gi_map.azimuthal_unit, gi_map.radial, gi_map.azimuthal, gi_map.intensity,
                          gi_map.coverage, gi_map.normalization))
    try:
        projection = project_analysis_artifact_result(
            **result,
            _stored_v1_claim=(
                _ANALYSIS_V1_DECODE_PROJECTION
                if _artifact_contract_version(version) == ANALYSIS_SCHEMA_VERSION
                else None
            ),
            wavelength_A=content.wavelength_A,
            _frame_manifest_json=content.frames_json,
            gi_cuts=tuple((mode, cut.unit, cut.axis, cut.intensity, cut.coverage, cut.normalization)
                          for mode, cut in gi),
        )
    except AnalysisArtifactProjectionInvalid as error:
        raise AnalysisArtifactInvalid(str(error)) from error
    return content, projection


def _inspect_stitch_record(
    entry: h5py.Group,
    shown: str,
    admitted_revision,
    *,
    expected_request: AnalysisArtifactRequest | None,
    expected_kind: AnalysisArtifactKind | None,
) -> AnalysisArtifactInspection:
    """Admit a stitched Int record: the request envelope on ``entry/stitch``, and the result
    fingerprint recomputed from the stored patterns."""
    stitch = _direct(entry, _STITCH_GROUP, h5py.Group)
    version_id = stitch.attrs.get_id(ANALYSIS_SCHEMA_VERSION_ATTR)
    if (
        version_id.shape != ()
        or _hdf_type_class(version_id) != h5py.h5t.INTEGER
        or version_id.dtype.itemsize > 8
    ):
        raise AnalysisArtifactInvalid("stitched record schema version is invalid")
    version = int(stitch.attrs[ANALYSIS_SCHEMA_VERSION_ATTR])
    if version not in _STITCH_RECORD_VERSIONS:
        raise AnalysisArtifactInvalid("stitched record schema version is unsupported")
    contract_version = _artifact_contract_version(version)
    attested = contract_version in {ANALYSIS_SCHEMA_VERSION_V2, ANALYSIS_SCHEMA_VERSION_V3}
    native = version in _STITCH_NATIVE_VERSIONS
    expected_attrs = {
        "NX_class", ANALYSIS_SCHEMA_ATTR, ANALYSIS_SCHEMA_VERSION_ATTR, ANALYSIS_KIND_ATTR,
        "request_fingerprint", "source_fingerprint", "plan_fingerprint", "provenance_digest", "file_name",
    } | ({"source_base"} if native else set()) | ({_EXECUTION_ATTESTATION_DIGEST_ATTR} if attested else set())
    if set(stitch.attrs) != expected_attrs:
        raise AnalysisArtifactInvalid("stitched record envelope attributes are not exact")
    if _bounded_attr_text(stitch, ANALYSIS_SCHEMA_ATTR) != ANALYSIS_SCHEMA_NAME:
        raise AnalysisArtifactInvalid("stitched record envelope marker is wrong")
    try:
        kind = AnalysisArtifactKind(_bounded_attr_text(stitch, ANALYSIS_KIND_ATTR))
    except ValueError as error:
        raise AnalysisArtifactInvalid("stitched record kind is invalid") from error
    expected = expected_request.kind if expected_request is not None else expected_kind
    if (kind not in {AnalysisArtifactKind.STITCH_1D, AnalysisArtifactKind.STITCH_2D}
            or (kind is AnalysisArtifactKind.STITCH_2D and version not in _STITCH_NATIVE_VERSIONS)
            or (expected is not None and kind is not expected)):
        raise AnalysisArtifactInvalid("stitched record kind does not match request")
    if expected_request is not None and version != expected_request.schema_version:
        raise AnalysisArtifactInvalid("stitched record schema version changed")
    fingerprints: dict[str, str] = {}
    for attr in ("request_fingerprint", "source_fingerprint", "plan_fingerprint", "provenance_digest"):
        fingerprints[attr] = _sha256(_bounded_attr_text(stitch, attr), attr)
        if expected_request is not None and fingerprints[attr] != getattr(expected_request, attr):
            raise AnalysisArtifactInvalid(f"stitched record {attr} changed")
    if _bounded_attr_text(stitch, "file_name") != (
        expected_request.target if expected_request is not None else shown
    ):
        raise AnalysisArtifactInvalid("stitched record records a private candidate path")
    if _bounded_attr_text(stitch, "NX_class") != "NXcollection":
        raise AnalysisArtifactInvalid("stitched record envelope is not an NXcollection")
    provenance = _read_scalar_text(stitch, _PROVENANCE)
    if expected_request is not None and provenance != expected_request.provenance_json:
        raise AnalysisArtifactInvalid("stitched record provenance changed")
    attestation_digest = attestation_json = None
    if attested:
        attestation_digest = _sha256(
            _exact_sha256_attr_text(stitch, _EXECUTION_ATTESTATION_DIGEST_ATTR),
            "execution attestation digest",
        )
        attestation_json = _read_scalar_execution_attestation(
            stitch, _EXECUTION_ATTESTATION, kind=kind,
            request_fingerprint=fingerprints["request_fingerprint"], expected_digest=attestation_digest,
        )
        if expected_request is not None and (
            attestation_digest != expected_request.execution_attestation_digest
            or attestation_json != expected_request.execution_attestation_json
        ):
            raise AnalysisArtifactInvalid("stitched record execution attestation changed")
    content, projection = _stitch_record_projection(entry.file, version)
    if projection.kind is not kind:
        raise AnalysisArtifactInvalid("stitched record kind does not match its result")
    # exactly the stitch graph this record's modes and version call for (Codex review F6)
    expected_children = {"coverage", "normalization", _PROVENANCE}
    expected_children |= {"frames", _STITCH_FRAMES_JSON} if native else set()
    expected_children |= {_EXECUTION_ATTESTATION} if attested else set()
    expected_children |= {mode for mode, _cut in content.gi}
    expected_children |= set() if content.map is None else {_STITCH_MAP}
    expected_children |= set() if content.gi_map is None else {_STITCH_GI_MAP}
    if set(stitch) != expected_children:
        raise AnalysisArtifactInvalid("stitched record contains an unknown stitch graph")
    for mode, _cut in content.gi + tuple((name, None) for name, value in (
            (_STITCH_MAP, content.map), (_STITCH_GI_MAP, content.gi_map)) if value is not None):
        cut = _direct(stitch, mode, h5py.Group)
        if (set(cut) != {"coverage", "normalization"} or set(cut.attrs) != {"NX_class"}
                or _bounded_attr_text(cut, "NX_class") != "NXcollection"):
            raise AnalysisArtifactInvalid("stitched record's GI cut diagnostics are not exact")
    if native:
        from xrd_tools.io.analysis_stitch_auxiliary import inspect_contributing_frames

        if inspect_contributing_frames(stitch, json.loads(provenance)) != content.frames_json:
            raise AnalysisArtifactInvalid("contributing frame manifest differs")
    if contract_version == ANALYSIS_SCHEMA_VERSION_V2 and (
        projection.axis_units[0][1] not in ({"q_A^-1", "2th_deg"} if native else {"q_A^-1"})
        or projection.sigma is not None
    ):
        raise AnalysisArtifactInvalid("analysis artifact v2 requires the exact XU Stitch result schema")
    if attestation_json is not None:
        attestation = json.loads(attestation_json)
        if attestation.get("result_fingerprint") != projection.result_fingerprint:
            raise AnalysisArtifactInvalid("analysis execution attestation result changed")
        if native and len(json.loads(content.frames_json)["records"]) != attestation["selected_frame_count"]:
            raise AnalysisArtifactInvalid("contributing frames differ from the science execution count")
    if contract_version == ANALYSIS_SCHEMA_VERSION_V2:
        for value in (projection, *(cut for _mode, cut in projection.gi),
                      *(value for value in (projection.profile, projection.gi_map) if value is not None)):
            empty = value.coverage == 0
            if (
                not np.array_equal(empty, value.normalization == 0)
                or not np.array_equal(empty, np.isnan(value.intensity))
                or not np.isfinite(value.intensity[~empty]).all()
            ):
                raise AnalysisArtifactInvalid("stitched record empty and occupied bins disagree")
    _require_local_graph(entry, set())
    return AnalysisArtifactInspection(
        path=shown,
        storage_revision=admitted_revision,
        kind=kind,
        group=_STITCH_RECORD_RESULT if content.map is None else _STITCH_RECORD_MAP,
        shape=projection.intensity.shape,
        axes=tuple((name, values.size) for name, values in projection.axes),
        axis_units=projection.axis_units,
        has_sigma=projection.sigma is not None,
        has_stitch_diagnostics=True,
        request_fingerprint=fingerprints["request_fingerprint"],
        source_fingerprint=fingerprints["source_fingerprint"],
        plan_fingerprint=fingerprints["plan_fingerprint"],
        provenance_digest=fingerprints["provenance_digest"],
        provenance_json=provenance,
        provenance_sha256=hashlib.sha256(provenance.encode("utf-8")).hexdigest(),
        result_fingerprint=projection.result_fingerprint,
        schema_version=version,
        wavelength_A=content.wavelength_A,
        frame_manifest_json=content.frames_json,
        execution_attestation_digest=attestation_digest,
        execution_attestation_json=attestation_json,
    )


def stitch_record_identity(path: str | Path) -> tuple[str, str] | None:
    """A saved Stitch record's plan fingerprint and provenance JSON, from its envelope alone (None
    when the file is not one). Its stored result is not verified: a processed file can always be
    stitched again from the raw data, so knowing whether it holds a stitch is not worth re-reading and
    re-hashing every stored pattern (maintainer, 2026-09-30: "we don't want to sacrifice perf ... to
    avoid rare edge cases where a processed file might get corrupted")."""
    try:
        with h5py.File(_absolute_path(path), "r") as handle:
            entry = handle.get(_ENTRY)
            if not isinstance(entry, h5py.Group) or not _is_stitch_record(entry):
                return None
            stitch = _direct(entry, _STITCH_GROUP, h5py.Group)
            if _bounded_attr_text(stitch, ANALYSIS_KIND_ATTR) not in {
                    AnalysisArtifactKind.STITCH_1D.value, AnalysisArtifactKind.STITCH_2D.value}:
                return None
            return (_sha256(_bounded_attr_text(stitch, "plan_fingerprint"), "plan_fingerprint"),
                    _read_scalar_text(stitch, _PROVENANCE))
    except (OSError, KeyError, ValueError, TypeError, AnalysisArtifactError):
        return None


def rsm_record_identity(path: str | Path) -> tuple[str, str] | None:
    """A saved RSM result's plan fingerprint and provenance JSON, as ``stitch_record_identity`` reads a
    Stitch record's: from the envelope alone, on the entry where an RSM keeps it (None when the file is
    not one). Its stored volume is not read or verified."""
    try:
        with h5py.File(_absolute_path(path), "r") as handle:
            entry = handle.get(_ENTRY)
            if (not isinstance(entry, h5py.Group)
                    or _bounded_attr_text(entry, ANALYSIS_SCHEMA_ATTR) != ANALYSIS_SCHEMA_NAME
                    or _bounded_attr_text(entry, ANALYSIS_KIND_ATTR) != AnalysisArtifactKind.RSM.value):
                return None
            return (_sha256(_bounded_attr_text(entry, "plan_fingerprint"), "plan_fingerprint"),
                    _read_scalar_text(entry, _PROVENANCE))
    except (OSError, KeyError, ValueError, TypeError, AnalysisArtifactError):
        return None


def inspect_analysis_artifact(
    path: str | Path,
    *,
    expected_request: AnalysisArtifactRequest | None = None,
    expected_kind: AnalysisArtifactKind | None = None,
    _candidate_claim: object = None,
) -> AnalysisArtifactInspection:
    shown = _absolute_path(path)
    if expected_request is None and Path(shown).suffix.casefold() != ".nexus":
        raise ValueError("analysis artifact path must end in .nexus")
    if expected_request is not None and type(expected_request) is not AnalysisArtifactRequest:
        raise TypeError("expected request must be exact AnalysisArtifactRequest")
    if expected_kind is not None and type(expected_kind) is not AnalysisArtifactKind:
        raise TypeError("expected kind must be exact AnalysisArtifactKind")
    if (
        expected_request is not None
        and shown != expected_request.target
        and _candidate_claim is not _ANALYSIS_CANDIDATE_INSPECTION
    ):
        raise AnalysisArtifactInvalid(
            "analysis artifact path does not match the expected request"
        )
    if (
        expected_request is not None
        and expected_kind is not None
        and expected_kind is not expected_request.kind
    ):
        raise ValueError("expected artifact kind conflicts with expected request")
    try:
        admitted_state = os.stat(shown)
        if (
            not stat.S_ISREG(admitted_state.st_mode)
            or admitted_state.st_size < 1
            or admitted_state.st_size > _MAX_ARTIFACT_FILE_BYTES
        ):
            raise AnalysisArtifactInvalid(
                "analysis artifact must be a regular bounded file"
            )
        admitted_revision = _stat_revision(admitted_state)
        with h5py.File(shown, "r") as handle:
            if _opened_hdf_revision(handle) != admitted_revision:
                raise AnalysisArtifactInvalid(
                    "analysis artifact changed while opening detached admission"
                )
            if _stat_revision(os.stat(shown)) != admitted_revision:
                raise AnalysisArtifactInvalid(
                    "analysis artifact changed while opening detached admission"
                )
            record_entry = handle.get(_ENTRY)
            if isinstance(record_entry, h5py.Group) and _is_stitch_record(record_entry):
                inspection = _inspect_stitch_record(
                    record_entry, shown, admitted_revision,
                    expected_request=expected_request, expected_kind=expected_kind,
                )
                if (
                    _opened_hdf_revision(handle) != admitted_revision
                    or _stat_revision(os.stat(shown)) != admitted_revision
                ):
                    raise AnalysisArtifactInvalid(
                        "analysis artifact changed during detached admission"
                    )
                return inspection
            if len(handle.attrs) != 0 or len(handle) != 1 or _ENTRY not in handle:
                raise AnalysisArtifactInvalid("analysis artifact must contain only /entry")
            entry = _direct(handle, _ENTRY, h5py.Group)
            try:
                version_id = entry.attrs.get_id(ANALYSIS_SCHEMA_VERSION_ATTR)
            except KeyError as error:
                raise AnalysisArtifactInvalid(
                    "analysis artifact schema version is missing"
                ) from error
            if (
                version_id.shape != ()
                or _hdf_type_class(version_id) != h5py.h5t.INTEGER
                or version_id.dtype.kind not in "iu"
                or version_id.dtype.itemsize > 8
            ):
                raise AnalysisArtifactInvalid(
                    "analysis artifact schema version is invalid"
                )
            version = int(entry.attrs[ANALYSIS_SCHEMA_VERSION_ATTR])
            if version not in {
                ANALYSIS_SCHEMA_VERSION,
                ANALYSIS_SCHEMA_VERSION_V2,
                ANALYSIS_SCHEMA_VERSION_V3,
                ANALYSIS_SCHEMA_VERSION_STITCH_NEUTRAL,
                ANALYSIS_SCHEMA_VERSION_XU_STITCH_NEUTRAL,
                ANALYSIS_SCHEMA_VERSION_STITCH_NATIVE,
                ANALYSIS_SCHEMA_VERSION_XU_STITCH_NATIVE,
            }:
                raise AnalysisArtifactInvalid(
                    "analysis artifact schema version is unsupported"
                )
            contract_version = _artifact_contract_version(version)
            expected_entry_attrs = {
                "NX_class",
                ANALYSIS_SCHEMA_ATTR,
                ANALYSIS_SCHEMA_VERSION_ATTR,
                ANALYSIS_KIND_ATTR,
                "request_fingerprint",
                "source_fingerprint",
                "plan_fingerprint",
                "provenance_digest",
                "file_name",
            }
            if contract_version in {ANALYSIS_SCHEMA_VERSION_V2, ANALYSIS_SCHEMA_VERSION_V3}:
                expected_entry_attrs.add(_EXECUTION_ATTESTATION_DIGEST_ATTR)
            if len(entry.attrs) != len(expected_entry_attrs) or any(
                name not in expected_entry_attrs for name in entry.attrs
            ):
                raise AnalysisArtifactInvalid(
                    "analysis artifact entry attributes are not exact"
                )
            if _bounded_attr_text(entry, "NX_class") != "NXentry":
                raise AnalysisArtifactInvalid("analysis artifact entry is not NXentry")
            if _bounded_attr_text(entry, ANALYSIS_SCHEMA_ATTR) != ANALYSIS_SCHEMA_NAME:
                raise AnalysisArtifactInvalid("analysis artifact schema marker is wrong")
            try:
                kind = AnalysisArtifactKind(_bounded_attr_text(entry, ANALYSIS_KIND_ATTR))
            except ValueError as error:
                raise AnalysisArtifactInvalid("analysis artifact kind is invalid") from error
            # A Stitch result is an Int record (stitch_record, admitted above): this layout is RSM's
            # (maintainer, 2026-09-29: no compatibility with earlier Stitch outputs; 2026-09-30:
            # "Remove old format")
            if kind is not AnalysisArtifactKind.RSM:
                raise OlderStitchArtifact(kind)
            if version in _STITCH_NEUTRAL_CONTRACTS:
                raise AnalysisArtifactInvalid("neutral analysis artifact requires its Stitch kind")
            expected = expected_request.kind if expected_request is not None else expected_kind
            if expected is not None and kind is not expected:
                raise AnalysisArtifactInvalid("analysis artifact kind does not match request")
            if (
                expected_request is not None
                and version != expected_request.schema_version
            ):
                raise AnalysisArtifactInvalid(
                    "analysis artifact schema version changed"
                )
            required_attrs = (
                ("request_fingerprint", "request_fingerprint"),
                ("source_fingerprint", "source_fingerprint"),
                ("plan_fingerprint", "plan_fingerprint"),
                ("provenance_digest", "provenance_digest"),
            )
            identity_fingerprints: dict[str, str] = {}
            for attr, request_attr in required_attrs:
                value = _sha256(_bounded_attr_text(entry, attr), attr)
                identity_fingerprints[attr] = value
                if expected_request is not None and value != getattr(expected_request, request_attr):
                    raise AnalysisArtifactInvalid(f"analysis artifact {attr} changed")
            execution_attestation_digest = None
            if contract_version in {ANALYSIS_SCHEMA_VERSION_V2, ANALYSIS_SCHEMA_VERSION_V3}:
                execution_attestation_digest = _sha256(
                    _exact_sha256_attr_text(
                        entry, _EXECUTION_ATTESTATION_DIGEST_ATTR
                    ),
                    "execution attestation digest",
                )
                if (
                    expected_request is not None
                    and execution_attestation_digest
                    != expected_request.execution_attestation_digest
                ):
                    raise AnalysisArtifactInvalid(
                        "analysis artifact execution attestation digest changed"
                    )
            final_name = _bounded_attr_text(entry, "file_name")
            expected_name = (
                expected_request.target
                if expected_request is not None
                else shown
            )
            if final_name != expected_name:
                raise AnalysisArtifactInvalid("analysis artifact records a private candidate path")
            expected_entry_children = 2 + int(
                contract_version in {ANALYSIS_SCHEMA_VERSION_V2, ANALYSIS_SCHEMA_VERSION_V3}
            )
            if len(entry) != expected_entry_children:
                raise AnalysisArtifactInvalid(
                    "analysis artifact contains an unknown entry graph"
                )
            present = tuple(
                item for item in ("stitched_1d", "stitched_2d", "rsm")
                if item in entry
            )
            if present != (kind.group,):
                raise AnalysisArtifactInvalid("analysis artifact result groups are ambiguous")
            # C0 excludes frame/source-base graphs until S1/R1 bind one exact
            # typed manifest into the operation request.  Provenance carries
            # source identity meanwhile; an unbound raw-popup graph must not
            # acquire artifact authority merely by being local-hard HDF5.
            allowed = {kind.group, _PROVENANCE}
            if contract_version in {ANALYSIS_SCHEMA_VERSION_V2, ANALYSIS_SCHEMA_VERSION_V3}:
                allowed.add(_EXECUTION_ATTESTATION)
            if any(name not in allowed for name in entry):
                raise AnalysisArtifactInvalid("analysis artifact contains an unknown entry graph")
            provenance = _read_scalar_text(entry, _PROVENANCE)
            if expected_request is not None and provenance != expected_request.provenance_json:
                raise AnalysisArtifactInvalid("analysis artifact provenance changed")
            execution_attestation_json = None
            if contract_version in {ANALYSIS_SCHEMA_VERSION_V2, ANALYSIS_SCHEMA_VERSION_V3}:
                execution_attestation_json = _read_scalar_execution_attestation(
                    entry,
                    _EXECUTION_ATTESTATION,
                    kind=kind,
                    request_fingerprint=identity_fingerprints[
                        "request_fingerprint"
                    ],
                    expected_digest=execution_attestation_digest,
                )
                if (
                    expected_request is not None
                    and execution_attestation_json
                    != expected_request.execution_attestation_json
                ):
                    raise AnalysisArtifactInvalid(
                        "analysis artifact execution attestation changed"
                    )
                parsed_attestation = json.loads(execution_attestation_json)
                if (
                    kind is AnalysisArtifactKind.RSM
                    and not _rsm_frame_provenance_matches(
                        provenance,
                        parsed_attestation,
                    )
                ):
                    raise AnalysisArtifactInvalid(
                        "analysis RSM frame provenance and attestation disagree"
                    )
                if version == ANALYSIS_SCHEMA_VERSION_V2 and kind is AnalysisArtifactKind.RSM:
                    valid_hkl_attestation = parsed_attestation.get(
                        "schema_version"
                    ) == "rsm-execution-attestation-v1" or (
                        parsed_attestation.get("schema_version")
                        == "rsm-execution-attestation-v2"
                        and parsed_attestation.get("coordinate_frame") == "hkl"
                    )
                    if not valid_hkl_attestation:
                        raise AnalysisArtifactInvalid(
                            "analysis artifact v2 requires an HKL attestation"
                        )
            result = _direct(entry, kind.group, h5py.Group)
            expected_result_attrs = {"NX_class", "signal", "axes"}
            if version == ANALYSIS_SCHEMA_VERSION_V3:
                expected_result_attrs.add("coordinate_frame")
            if len(result.attrs) != len(expected_result_attrs) or any(
                name not in expected_result_attrs for name in result.attrs
            ):
                raise AnalysisArtifactInvalid(
                    "analysis result attributes are not exact"
                )
            if _bounded_attr_text(result, "NX_class") != "NXdata":
                raise AnalysisArtifactInvalid("analysis result group is not NXdata")
            if _bounded_attr_text(result, "signal") != "intensity":
                raise AnalysisArtifactInvalid("analysis result signal is invalid")
            group_provenance = _read_scalar_text(result, _PROVENANCE)
            if group_provenance != provenance:
                raise AnalysisArtifactInvalid("result and artifact provenance differ")
            coordinate_frame = None
            if version == ANALYSIS_SCHEMA_VERSION_V3:
                from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame

                try:
                    coordinate_frame = RSMCoordinateFrame(
                        _bounded_attr_text(result, "coordinate_frame")
                    )
                except ValueError as error:
                    raise AnalysisArtifactInvalid(
                        "analysis RSM coordinate frame is invalid"
                    ) from error
                if coordinate_frame is not RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU:
                    raise AnalysisArtifactInvalid(
                        "analysis artifact v3 requires Cartesian-Q coordinates"
                    )
                axis_names = coordinate_frame.axis_names
            else:
                axis_names = ("h", "k", "l")
            if _bounded_attr_texts(result, "axes") != axis_names:
                raise AnalysisArtifactInvalid("analysis result axes are invalid")
            allowed_result = {"intensity", _PROVENANCE, *axis_names}
            if any(name not in allowed_result for name in result):
                raise AnalysisArtifactInvalid(
                    "analysis result contains an unknown graph"
                )
            has_sigma = has_stitch_diagnostics = False  # an RSM has neither
            if len(result) != 2 + len(axis_names):
                raise AnalysisArtifactInvalid(
                    "analysis result contains an unknown graph"
                )
            axes = tuple((name, _axis(result, name)) for name in axis_names)
            for name, _values in axes:
                axis = _direct(result, name, h5py.Dataset)
                expected_axis_attrs = set()
                if (
                    coordinate_frame is not None
                    and dict(
                        zip(
                            coordinate_frame.axis_names,
                            coordinate_frame.axis_units,
                            strict=True,
                        )
                    )[name]
                    is not None
                ):
                    expected_axis_attrs.add("units")
                if len(axis.attrs) != len(expected_axis_attrs) or any(
                    attr not in expected_axis_attrs for attr in axis.attrs
                ):
                    raise AnalysisArtifactInvalid(
                        f"analysis axis {name} attributes are not exact"
                    )
            axis_units: list[tuple[str, str | None]] = [
                (name, None) for name, _values in axes
            ]
            if coordinate_frame is not None:
                for index, (name, expected_units) in enumerate(
                    zip(
                        coordinate_frame.axis_names,
                        coordinate_frame.axis_units,
                        strict=True,
                    )
                ):
                    if expected_units is None:
                        continue
                    observed_units = _bounded_attr_text(
                        _direct(result, name, h5py.Dataset),
                        "units",
                    )
                    if observed_units != expected_units:
                        raise AnalysisArtifactInvalid(
                            f"analysis RSM axis {name} units are invalid"
                        )
                    axis_units[index] = (name, observed_units)
            if contract_version == ANALYSIS_SCHEMA_VERSION_V2:
                if axis_units != [("h", None), ("k", None), ("l", None)]:
                    raise AnalysisArtifactInvalid(
                        "analysis artifact v2 requires the exact RSM result "
                        "schema"
                    )
            elif version == ANALYSIS_SCHEMA_VERSION_V3:
                try:
                    provenance_value = json.loads(provenance)
                    attestation_value = json.loads(execution_attestation_json)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise AnalysisArtifactInvalid(
                        "analysis artifact v3 frame identity is malformed"
                    ) from error
                if (
                    coordinate_frame is None
                    or tuple(axis_units)
                    != tuple(
                        zip(
                            coordinate_frame.axis_names,
                            coordinate_frame.axis_units,
                            strict=True,
                        )
                    )
                    or has_sigma
                    or has_stitch_diagnostics
                    or provenance_value.get("coordinate_frame")
                    != {
                        "name": coordinate_frame.value,
                        "axis_names": list(coordinate_frame.axis_names),
                        "axis_units": list(coordinate_frame.axis_units),
                        "matrix_policy": coordinate_frame.matrix_policy,
                    }
                    or attestation_value.get("coordinate_frame")
                    != coordinate_frame.value
                    or attestation_value.get("axis_names")
                    != list(coordinate_frame.axis_names)
                    or attestation_value.get("axis_units")
                    != list(coordinate_frame.axis_units)
                    or attestation_value.get("matrix_policy")
                    != coordinate_frame.matrix_policy
                ):
                    raise AnalysisArtifactInvalid(
                        "analysis artifact v3 requires the exact frame-aware RSM schema"
                    )
            intensity = _direct(result, "intensity", h5py.Dataset)
            if len(intensity.attrs) != 0:
                raise AnalysisArtifactInvalid(
                    "analysis intensity attributes are not exact"
                )
            expected_shape = tuple(len(values) for _name, values in axes)
            if intensity.shape != expected_shape:
                raise AnalysisArtifactInvalid("analysis intensity shape does not match axes")
            intensity_values = _materialize_inspection_values(intensity)
            try:
                wavelength_A = frame_manifest_json = None
                projection = project_analysis_artifact_result(
                    wavelength_A=wavelength_A,
                    _frame_manifest_json=frame_manifest_json,
                    kind=kind,
                    axes=axes,
                    axis_units=tuple(axis_units),
                    intensity=intensity_values,
                    sigma=None,
                    coverage=None,
                    normalization=None,
                    _stored_v1_claim=(
                        _ANALYSIS_V1_DECODE_PROJECTION
                        if contract_version == ANALYSIS_SCHEMA_VERSION
                        else None
                    ),
                )
            except AnalysisArtifactProjectionInvalid as error:
                raise AnalysisArtifactInvalid(str(error)) from error
            if (
                execution_attestation_json is not None
                and json.loads(execution_attestation_json).get(
                    "result_fingerprint"
                )
                != projection.result_fingerprint
            ):
                raise AnalysisArtifactInvalid(
                    "analysis execution attestation result changed"
                )
            seen: set[int] = set()
            _require_local_graph(entry, seen)
            inspection = AnalysisArtifactInspection(
                path=shown,
                storage_revision=admitted_revision,
                kind=kind,
                group=kind.group,
                shape=expected_shape,
                axes=tuple((name, len(values)) for name, values in axes),
                axis_units=tuple(axis_units),
                has_sigma=has_sigma,
                has_stitch_diagnostics=has_stitch_diagnostics,
                request_fingerprint=identity_fingerprints["request_fingerprint"],
                source_fingerprint=identity_fingerprints["source_fingerprint"],
                plan_fingerprint=identity_fingerprints["plan_fingerprint"],
                provenance_digest=identity_fingerprints["provenance_digest"],
                provenance_json=provenance,
                provenance_sha256=hashlib.sha256(
                    provenance.encode("utf-8")
                ).hexdigest(),
                result_fingerprint=projection.result_fingerprint,
                schema_version=version,
                wavelength_A=wavelength_A,
                frame_manifest_json=frame_manifest_json,
                execution_attestation_digest=execution_attestation_digest,
                execution_attestation_json=execution_attestation_json,
            )
            if (
                _opened_hdf_revision(handle) != admitted_revision
                or _stat_revision(os.stat(shown)) != admitted_revision
            ):
                raise AnalysisArtifactInvalid(
                    "analysis artifact changed during detached admission"
                )
        final_state = os.stat(shown)
        if _stat_revision(final_state) != admitted_revision:
            raise AnalysisArtifactInvalid(
                "analysis artifact changed during detached admission"
            )
        return inspection
    except AnalysisArtifactInvalid:
        raise
    except (
        OSError,
        KeyError,
        TargetChanged,
        TypeError,
        ValueError,
        UnicodeError,
    ) as error:
        raise AnalysisArtifactInvalid(f"analysis artifact admission failed: {error}") from error


@dataclass(frozen=True, slots=True)
class AnalysisArtifactHeader:
    """An existing analysis record's kind and identity, admitted from its envelope alone: no result array,
    provenance or attestation is read (``inspect_analysis_artifact_header``)."""

    path: str
    kind: AnalysisArtifactKind
    schema_version: int
    request_fingerprint: str
    source_fingerprint: str
    plan_fingerprint: str
    provenance_digest: str


def _header_version(envelope, versions) -> int:
    """An envelope's schema version, as the strict admission checks it."""
    try:
        version_id = envelope.attrs.get_id(ANALYSIS_SCHEMA_VERSION_ATTR)
    except KeyError as error:
        raise AnalysisArtifactInvalid("analysis artifact schema version is missing") from error
    if (
        version_id.shape != ()
        or _hdf_type_class(version_id) != h5py.h5t.INTEGER
        or version_id.dtype.kind not in "iu"
        or version_id.dtype.itemsize > 8
    ):
        raise AnalysisArtifactInvalid("analysis artifact schema version is invalid")
    version = int(envelope.attrs[ANALYSIS_SCHEMA_VERSION_ATTR])
    if version not in versions:
        raise AnalysisArtifactInvalid("analysis artifact schema version is unsupported")
    return version


def inspect_analysis_artifact_header(
    path: str | Path,
    *,
    expected_kind: AnalysisArtifactKind,
) -> AnalysisArtifactHeader:
    """The bounded kind and identity admission of an existing analysis record: what Replace and a Preview's
    check of an existing target need, without reading the volume, curves, provenance or attestation (Codex's
    RSM stage-2 review, questions 1 and 2). It admits what the strict ``inspect_analysis_artifact`` admits of
    the envelope: a regular bounded file, opened revision-stable; an Int Stitch record's ``/entry/stitch``, or
    the analysis schema's ``/entry`` alone; the envelope's exact attribute names, NX class, schema marker,
    supported version and kind (``expected_kind``); its four identity fingerprints; and its recorded file name
    equal to its path. A Stitch result in the layout before Stitch records raises ``OlderStitchArtifact``, as
    the strict admission does. AnalysisArtifactInvalid: a foreign file, another kind, or a damaged envelope."""
    shown = _absolute_path(path)
    if Path(shown).suffix.casefold() != ".nexus":
        raise ValueError("analysis artifact path must end in .nexus")
    if type(expected_kind) is not AnalysisArtifactKind:
        raise TypeError("expected kind must be exact AnalysisArtifactKind")
    try:
        admitted_state = os.stat(shown)
        if (
            not stat.S_ISREG(admitted_state.st_mode)
            or admitted_state.st_size < 1
            or admitted_state.st_size > _MAX_ARTIFACT_FILE_BYTES
        ):
            raise AnalysisArtifactInvalid("analysis artifact must be a regular bounded file")
        admitted_revision = _stat_revision(admitted_state)
        with h5py.File(shown, "r") as handle:
            if (_opened_hdf_revision(handle) != admitted_revision
                    or _stat_revision(os.stat(shown)) != admitted_revision):
                raise AnalysisArtifactInvalid("analysis artifact changed while opening its header admission")
            entry = handle.get(_ENTRY)
            record = isinstance(entry, h5py.Group) and _is_stitch_record(entry)
            if record:
                envelope = _direct(entry, _STITCH_GROUP, h5py.Group)
                version = _header_version(envelope, _STITCH_RECORD_VERSIONS)
                nx_class = "NXcollection"
                native = {"source_base"} if version in _STITCH_NATIVE_VERSIONS else set()
            else:
                if len(handle.attrs) != 0 or len(handle) != 1 or _ENTRY not in handle:
                    raise AnalysisArtifactInvalid("analysis artifact must contain only /entry")
                envelope = _direct(handle, _ENTRY, h5py.Group)
                version = _header_version(envelope, {
                    ANALYSIS_SCHEMA_VERSION, ANALYSIS_SCHEMA_VERSION_V2, ANALYSIS_SCHEMA_VERSION_V3,
                    ANALYSIS_SCHEMA_VERSION_STITCH_NEUTRAL, ANALYSIS_SCHEMA_VERSION_XU_STITCH_NEUTRAL,
                    ANALYSIS_SCHEMA_VERSION_STITCH_NATIVE, ANALYSIS_SCHEMA_VERSION_XU_STITCH_NATIVE,
                })
                nx_class = "NXentry"
                native = set()
            attested = _artifact_contract_version(version) in {ANALYSIS_SCHEMA_VERSION_V2, ANALYSIS_SCHEMA_VERSION_V3}
            if set(envelope.attrs) != {
                "NX_class", ANALYSIS_SCHEMA_ATTR, ANALYSIS_SCHEMA_VERSION_ATTR, ANALYSIS_KIND_ATTR,
                "request_fingerprint", "source_fingerprint", "plan_fingerprint", "provenance_digest", "file_name",
            } | native | ({_EXECUTION_ATTESTATION_DIGEST_ATTR} if attested else set()):
                raise AnalysisArtifactInvalid("analysis artifact envelope attributes are not exact")
            if _bounded_attr_text(envelope, "NX_class") != nx_class:
                raise AnalysisArtifactInvalid(f"analysis artifact envelope is not an {nx_class}")
            if _bounded_attr_text(envelope, ANALYSIS_SCHEMA_ATTR) != ANALYSIS_SCHEMA_NAME:
                raise AnalysisArtifactInvalid("analysis artifact schema marker is wrong")
            try:
                kind = AnalysisArtifactKind(_bounded_attr_text(envelope, ANALYSIS_KIND_ATTR))
            except ValueError as error:
                raise AnalysisArtifactInvalid("analysis artifact kind is invalid") from error
            if record:
                if (kind not in {AnalysisArtifactKind.STITCH_1D, AnalysisArtifactKind.STITCH_2D}
                        or (kind is AnalysisArtifactKind.STITCH_2D and not native)):
                    raise AnalysisArtifactInvalid("stitched record kind is invalid")
            else:
                if kind is not AnalysisArtifactKind.RSM:  # the layout before Stitch records
                    raise OlderStitchArtifact(kind)
                if version in _STITCH_NEUTRAL_CONTRACTS:
                    raise AnalysisArtifactInvalid("neutral analysis artifact requires its Stitch kind")
            if kind is not expected_kind:
                raise AnalysisArtifactInvalid(
                    f"analysis artifact is a {kind.value} result, not {expected_kind.value}")
            fingerprints = {
                attr: _sha256(_bounded_attr_text(envelope, attr), attr)
                for attr in ("request_fingerprint", "source_fingerprint", "plan_fingerprint", "provenance_digest")
            }
            if _bounded_attr_text(envelope, "file_name") != shown:
                raise AnalysisArtifactInvalid("analysis artifact records a private candidate path")
            if (_opened_hdf_revision(handle) != admitted_revision
                    or _stat_revision(os.stat(shown)) != admitted_revision):
                raise AnalysisArtifactInvalid("analysis artifact changed during its header admission")
    except AnalysisArtifactInvalid:
        raise
    except (OSError, KeyError, TypeError, ValueError, UnicodeError) as error:
        raise AnalysisArtifactInvalid(f"analysis artifact header admission failed: {error}") from error
    return AnalysisArtifactHeader(shown, kind, version, **fingerprints)


_RESULT_NAMES = {
    AnalysisArtifactKind.STITCH_1D: "a Stitch 1D result",
    AnalysisArtifactKind.STITCH_2D: "a Stitch 2D result",
    AnalysisArtifactKind.RSM: "an RSM result",
}


def analysis_output_refusal(target: str | Path, kind: AnalysisArtifactKind) -> str | None:
    """Why the file at an analysis output's target cannot be replaced by a ``kind`` result, or None: no file,
    that kind's record, or that kind's result in the layout before Stitch records (Replace computes it again).
    Replace's own header admission (``inspect_analysis_artifact_header``), so a Preview refuses a foreign or
    other-kind target before any gridding or integration (Codex's RSM stage-2 review, question 1)."""
    if not os.path.lexists(target):
        return None
    try:
        inspect_analysis_artifact_header(target, expected_kind=kind)
    except OlderStitchArtifact as older:
        if older.kind is kind:
            return None
        reason = f"it is an older {older.kind.value} result"
    except AnalysisArtifactInvalid as error:
        reason = str(error)
    else:
        return None
    return (f"{Path(target).name} is not {_RESULT_NAMES[kind]} that Replace can overwrite ({reason}): "
            "choose another output, or move that file")


def read_analysis_artifact(
    path: str | Path,
    *,
    expected_request: AnalysisArtifactRequest | None = None,
    expected_kind: AnalysisArtifactKind | None = None,
    expected_receipt: AnalysisArtifactReceipt | None = None,
) -> AnalysisArtifactPayload:
    """Strictly admit and detach one standalone Stitch or RSM payload."""

    if expected_request is not None and type(expected_request) is not AnalysisArtifactRequest:
        raise TypeError("expected request must be exact AnalysisArtifactRequest")
    if expected_kind is not None and type(expected_kind) is not AnalysisArtifactKind:
        raise TypeError("expected kind must be exact AnalysisArtifactKind")
    if expected_receipt is not None:
        if type(expected_receipt) is not AnalysisArtifactReceipt:
            raise TypeError("expected receipt must be exact AnalysisArtifactReceipt")
        if (
            expected_request is not None
            and expected_request is not expected_receipt.request
        ):
            raise ValueError("expected request conflicts with expected receipt")
        if (
            expected_kind is not None
            and expected_kind is not expected_receipt.request.kind
        ):
            raise ValueError("expected artifact kind conflicts with expected receipt")
        expected_request = expected_receipt.request
        expected_kind = expected_receipt.request.kind

    shown = _absolute_path(path)
    try:
        if expected_receipt is not None:
            revalidate_stream_terminal(shown, expected_receipt.terminal)
        inspection = inspect_analysis_artifact(
            shown,
            expected_request=expected_request,
            expected_kind=expected_kind,
        )
        if (
            expected_receipt is not None
            and inspection != expected_receipt.inspection
        ):
            raise TargetChanged(
                "analysis artifact no longer matches its commit receipt"
            )
        admitted_revision = inspection.storage_revision
        if _stat_revision(os.stat(shown)) != admitted_revision:
            raise TargetChanged("analysis artifact changed before payload hydration")
        if inspection.group in {_STITCH_RECORD_RESULT, _STITCH_RECORD_MAP}:  # a stitched Int record
            with h5py.File(shown, "r") as handle:
                if (
                    _opened_hdf_revision(handle) != admitted_revision
                    or _stat_revision(os.stat(shown)) != admitted_revision
                ):
                    raise TargetChanged("analysis artifact changed while opening payload")
                _content, projection = _stitch_record_projection(handle, inspection.schema_version)
                if projection.result_fingerprint != inspection.result_fingerprint:
                    raise TargetChanged("analysis payload changed after detached admission")
                if (
                    _opened_hdf_revision(handle) != admitted_revision
                    or _stat_revision(os.stat(shown)) != admitted_revision
                ):
                    raise TargetChanged("analysis artifact changed during payload hydration")
            if expected_receipt is not None:
                revalidate_stream_terminal(shown, expected_receipt.terminal)
            return AnalysisArtifactPayload(
                inspection,
                projection.axes,
                projection.intensity,
                projection.sigma,
                projection.coverage,
                projection.normalization,
                _ANALYSIS_PAYLOAD_FACTORY,
                gi=projection.gi,
                profile=projection.profile,
                gi_map=projection.gi_map,
                angular_support=_content.angular_support,
            )

        with h5py.File(shown, "r") as handle:
            if (
                _opened_hdf_revision(handle) != admitted_revision
                or _stat_revision(os.stat(shown)) != admitted_revision
            ):
                raise TargetChanged(
                    "analysis artifact changed while opening payload"
                )
            entry = _direct(handle, _ENTRY, h5py.Group)
            result = _direct(entry, inspection.group, h5py.Group)  # an RSM (a Stitch is a record)
            axes = tuple((name, _axis(result, name)) for name, _length in inspection.axes)
            intensity = _materialize_bounded_values(
                _direct(result, "intensity", h5py.Dataset)
            )
            if intensity.shape != inspection.shape:
                raise AnalysisArtifactInvalid(
                    "analysis payload shape changed after admission"
                )
            if inspection.schema_version == ANALYSIS_SCHEMA_VERSION_V3:
                for name, expected_units in inspection.axis_units:
                    axis = _direct(result, name, h5py.Dataset)
                    if expected_units is None:
                        if "units" in axis.attrs:
                            raise AnalysisArtifactInvalid(
                                f"analysis RSM axis {name} units changed after admission"
                            )
                    elif _bounded_attr_text(axis, "units") != expected_units:
                        raise AnalysisArtifactInvalid(
                            f"analysis RSM axis {name} units changed after admission"
                        )
            try:
                projection = project_analysis_artifact_result(
                    wavelength_A=inspection.wavelength_A,
                    _frame_manifest_json=inspection.frame_manifest_json,
                    kind=inspection.kind,
                    axes=axes,
                    axis_units=inspection.axis_units,
                    intensity=intensity,
                    sigma=None,
                    coverage=None,
                    normalization=None,
                    _stored_v1_claim=(
                        _ANALYSIS_V1_DECODE_PROJECTION
                        if _artifact_contract_version(inspection.schema_version)
                        == ANALYSIS_SCHEMA_VERSION
                        else None
                    ),
                )
            except AnalysisArtifactProjectionInvalid as error:
                raise AnalysisArtifactInvalid(str(error)) from error
            if projection.result_fingerprint != inspection.result_fingerprint:
                raise TargetChanged(
                    "analysis payload changed after detached admission"
                )
            if (
                _opened_hdf_revision(handle) != admitted_revision
                or _stat_revision(os.stat(shown)) != admitted_revision
            ):
                raise TargetChanged(
                    "analysis artifact changed during payload hydration"
                )
        if _stat_revision(os.stat(shown)) != admitted_revision:
            raise TargetChanged(
                "analysis artifact changed during payload hydration"
            )
        if expected_receipt is not None:
            revalidate_stream_terminal(shown, expected_receipt.terminal)
        return AnalysisArtifactPayload(
            inspection,
            projection.axes,
            projection.intensity,
            projection.sigma,
            projection.coverage,
            projection.normalization,
            _ANALYSIS_PAYLOAD_FACTORY,
        )
    except AnalysisArtifactInvalid:
        raise
    except (
        OSError,
        KeyError,
        TargetChanged,
        TypeError,
        ValueError,
        UnicodeError,
    ) as error:
        raise AnalysisArtifactInvalid(
            f"analysis artifact payload admission failed: {error}"
        ) from error


class AnalysisArtifactOutput:
    """One analysis schema adapter over the shared finite publication owner."""

    def __init__(self, request: AnalysisArtifactRequest, *,
                 coordinator: OutputTransactionCoordinator,
                 protected_inputs: tuple[str | Path, ...] = ()) -> None:
        if type(request) is not AnalysisArtifactRequest:
            raise TypeError("analysis output requires exact request")
        if not Path(request.target).parent.is_dir():
            raise ValueError("analysis artifact parent directory must already exist")
        self.request = request
        self._protected_inputs = tuple(Path(value).expanduser().absolute() for value in protected_inputs)
        self._publication = _FinitePublicationSession(request.target, coordinator=coordinator)
        self._writer_started = False
        self._writer_finished = False
        self._validated_candidate: TargetSnapshot | None = None
        self._receipt: AnalysisArtifactReceipt | None = None
        self._completed = False
        self._close_pending = False
        self._on_published: Callable[[AnalysisArtifactReceipt], object] | None = None
        self._publication.reserve()
        try:
            self._ordinal = self._publication.ordinal
            self._check_input_aliases()
            self._expected_target = capture_target_snapshot(request.target)
            if request.overwrite is AnalysisArtifactOverwrite.CREATE_NEW and (
                self._expected_target.exists or os.path.lexists(request.target)
            ):
                raise FileExistsError(request.target)
            if self._expected_target.exists:
                # Its kind and identity from its header alone: Replace never reads the volume or curves it
                # replaces (Codex's RSM stage-2 review, rec. 3); the alias and snapshot checks stand
                try:
                    inspect_analysis_artifact_header(request.target, expected_kind=request.kind)
                except OlderStitchArtifact as older:  # Replace stitches it again, as its refusal says
                    if older.kind is not request.kind:
                        raise
                if capture_target_snapshot(request.target) != self._expected_target:
                    raise TargetChanged("previous analysis changed during admission")
        except BaseException:
            self._publication.abort()
            raise

    def _check_input_aliases(self) -> None:
        check_output_not_source(self.request.target, input_files=self._protected_inputs)

    def __copy__(self):
        raise TypeError("analysis artifact output owner is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("analysis artifact output owner is not copyable")

    @property
    def snapshot(self) -> AnalysisArtifactOutputSnapshot:
        owner = self._publication
        return AnalysisArtifactOutputSnapshot(
            self.request.target, self._writer_started, self._writer_finished,
            owner.published, self._close_pending, owner.slot_held,
            self._close_pending or owner.published and not self._completed,
            self._receipt, tuple(owner.diagnostics), owner.hidden_orphan,
        )

    def _write_candidate(
        self,
        write_result: Callable[[h5py.Group], object],
    ) -> None:
        self._writer_started = True
        # A Stitch result is an Int record (``stitch_record``): the writer fills a fresh open_nexus_writer
        # document first, and the request envelope then goes on its stitch group. An RSM is this layout.
        if self.request.kind is not AnalysisArtifactKind.RSM:
            from xrd_tools.io.nexus import open_nexus_writer

            def open_record(path):
                document = open_nexus_writer(path, entry=_ENTRY, compression="gzip", overwrite=True)
                # This owner admits the candidate itself once the writer is done (_validate_candidate); a
                # close-time re-admission would turn a failed writer's abort into a pending cleanup.
                document._disable_terminal_admission()
                return document

            handle = self._publication.open_document(open_record)
            entry = handle[_ENTRY]
            write_result(entry)
            envelope = _direct(entry, _STITCH_GROUP, h5py.Group)
        else:
            handle = self._publication.open_document(lambda path: h5py.File(path, "w"))
            entry = envelope = handle.create_group(_ENTRY)
            entry.attrs.create("NX_class", np.bytes_(b"NXentry"))
        envelope.attrs.create(
            ANALYSIS_SCHEMA_ATTR,
            np.bytes_(ANALYSIS_SCHEMA_NAME.encode("utf-8")),
        )
        envelope.attrs[ANALYSIS_SCHEMA_VERSION_ATTR] = self.request.schema_version
        for name, value in (
            (ANALYSIS_KIND_ATTR, self.request.kind.value),
            ("request_fingerprint", self.request.request_fingerprint),
            ("source_fingerprint", self.request.source_fingerprint),
            ("plan_fingerprint", self.request.plan_fingerprint),
            ("provenance_digest", self.request.provenance_digest),
            ("file_name", self.request.target),
        ):
            envelope.attrs.create(name, np.bytes_(value.encode("utf-8")))
        if _artifact_contract_version(self.request.schema_version) in {
            ANALYSIS_SCHEMA_VERSION_V2,
            ANALYSIS_SCHEMA_VERSION_V3,
        }:
            envelope.attrs.create(
                _EXECUTION_ATTESTATION_DIGEST_ATTR,
                np.bytes_(
                    self.request.execution_attestation_digest.encode(
                        "utf-8"
                    )
                ),
            )
        envelope.create_dataset(
            _PROVENANCE,
            data=np.bytes_(self.request.provenance_json.encode("utf-8")),
        )
        if _artifact_contract_version(self.request.schema_version) in {
            ANALYSIS_SCHEMA_VERSION_V2,
            ANALYSIS_SCHEMA_VERSION_V3,
        }:
            envelope.create_dataset(
                _EXECUTION_ATTESTATION,
                data=np.bytes_(
                    self.request.execution_attestation_json.encode("utf-8")
                ),
            )
        if self.request.kind is AnalysisArtifactKind.RSM:  # a record's writer ran first
            write_result(entry)
        handle.flush()

    def _validate_candidate(
        self,
        candidate: Path,
        captured: TargetSnapshot,
        prepublish: Callable[[], object] | None,
    ) -> None:
        before = capture_target_snapshot(candidate)
        if before != captured:
            raise TargetChanged(
                "analysis candidate changed before semantic readback"
            )
        inspect_analysis_artifact(
            candidate,
            expected_request=self.request,
            _candidate_claim=_ANALYSIS_CANDIDATE_INSPECTION,
        )
        after = capture_target_snapshot(candidate)
        if not after.exists or before != after:
            raise TargetChanged(
                "analysis candidate changed during semantic readback"
            )
        if prepublish is not None:
            prepublish()
        self._check_input_aliases()
        final = capture_target_snapshot(candidate)
        if final != after:
            raise TargetChanged(
                "analysis candidate changed after prepublication checks"
            )
        self._validated_candidate = final
        self._writer_finished = True

    def _terminal(self) -> tuple[StreamTerminal, AnalysisArtifactInspection]:
        before = capture_target_snapshot(self.request.target)
        inspection = inspect_analysis_artifact(
            self.request.target, expected_request=self.request
        )
        captured = capture_target_snapshot(self.request.target)
        if (
            before != captured
            or self._validated_candidate is None
            or captured != self._validated_candidate
            or not captured.exists
            or captured.size is None
            or captured.digest is None
            or captured.device is None
            or captured.inode is None
            or captured.mtime_ns is None
        ):
            raise TargetChanged("committed analysis artifact lacks exact storage facts")
        state = os.stat(self.request.target)
        if (
            int(state.st_dev),
            int(state.st_ino),
            int(state.st_size),
            int(state.st_mtime_ns),
        ) != (
            captured.device,
            captured.inode,
            captured.size,
            captured.mtime_ns,
        ):
            raise TargetChanged("analysis artifact changed while sealing commit")
        # The terminal's object revision is the capture's descriptor view
        # (its ctime is the change time on win32, never the pathname stat's
        # creation time), the origin ``revalidate_stream_terminal`` holds
        # its own descriptor views to.
        terminal = StreamTerminal(
            self.request.target,
            captured.size,
            captured.digest,
            self._ordinal,
            captured.device,
            captured.inode,
            captured.mtime_ns,
            captured.ctime_ns,
        )
        revalidate_stream_terminal(self.request.target, terminal)
        final_capture = capture_target_snapshot(self.request.target)
        final_state = os.stat(self.request.target)
        if (
            final_capture != captured
            or final_capture.ctime_ns != terminal.ctime_ns
            or (
                int(final_state.st_dev),
                int(final_state.st_ino),
                int(final_state.st_size),
                int(final_state.st_mtime_ns),
            ) != (
                terminal.device,
                terminal.inode,
                terminal.size,
                terminal.mtime_ns,
            )
        ):
            raise TargetChanged("analysis artifact changed after terminal validation")
        return terminal, inspection

    def _seal_receipt(self) -> AnalysisArtifactReceipt:
        if self._receipt is None:
            terminal, inspection = self._terminal()
            self._receipt = AnalysisArtifactReceipt(
                self.request,
                terminal,
                inspection,
                _ANALYSIS_RECEIPT_FACTORY,
            )
        else:
            revalidate_stream_terminal(
                self.request.target,
                self._receipt.terminal,
            )
            inspection = inspect_analysis_artifact(
                self.request.target,
                expected_request=self.request,
            )
            revalidate_stream_terminal(
                self.request.target,
                self._receipt.terminal,
            )
            if inspection != self._receipt.inspection:
                raise TargetChanged(
                    "cached analysis artifact receipt no longer matches the target"
                )
        return self._receipt

    def _observe_published(self, _path: Path, _ordinal: int) -> StreamTerminal:
        receipt = self._seal_receipt()
        if self._on_published is not None:
            self._on_published(receipt)
        self._completed = True
        self._on_published = None
        return receipt.terminal

    def publish(
        self,
        write_result: Callable[[h5py.Group], object],
        *,
        prepublish: Callable[[], object] | None = None,
        _on_published: Callable[[AnalysisArtifactReceipt], object] | None = None,
    ) -> AnalysisArtifactReceipt:
        if not callable(write_result):
            raise TypeError("analysis artifact writer must be callable")
        if prepublish is not None and not callable(prepublish):
            raise TypeError("analysis artifact prepublication check must be callable")
        if _on_published is not None and not callable(_on_published):
            raise TypeError("analysis published observer must be callable")
        if self._writer_started or not self._publication.slot_held:
            raise TransactionStateError("analysis artifact writer is one-shot")
        self._on_published = _on_published
        try:
            candidate = self._publication.candidate
            self._check_input_aliases()
            if capture_target_snapshot(self.request.target) != self._expected_target:
                raise TargetChanged("analysis target changed before writing")
            self._write_candidate(write_result)
            if not self._publication.close_document():
                raise OSError("analysis candidate document has not closed")
            self._validate_candidate(candidate, capture_target_snapshot(candidate), prepublish)
            facts = self._publication.publish(
                self._expected_target, ordinal=self._ordinal,
                observe=self._observe_published,
            )
        except BaseException as error:
            self._on_published = None
            self._close_pending = not self._publication.abort()
            for diagnostic in self._publication.diagnostics:
                error.add_note(diagnostic)
            if self._publication.hidden_orphan is not None:
                error.add_note(f"hidden candidate remains: {self._publication.hidden_orphan}")
            if self._close_pending:
                raise AnalysisArtifactCleanupPending(self.snapshot) from error
            raise
        if facts.error is not None:
            # The shared owner has already observed publication and released
            # the slot. Keep only immutable verification facts, never rollback.
            raise AnalysisArtifactCleanupPending(self.snapshot) from facts.error
        return self._receipt

    def abort(self) -> AnalysisArtifactOutputSnapshot:
        if self._publication.published:
            return self.snapshot
        self._on_published = None
        self._close_pending = not self._publication.abort()
        if self._close_pending:
            raise AnalysisArtifactCleanupPending(self.snapshot)
        return self.snapshot

    def retry_cleanup(self) -> AnalysisArtifactOutputSnapshot:
        if not self._publication.published:
            return self.abort()
        try:
            self._observe_published(Path(self.request.target), self._ordinal)
        except BaseException as error:
            raise AnalysisArtifactCleanupPending(self.snapshot) from error
        return self.snapshot


def admit_analysis_artifact(
    request: AnalysisArtifactRequest,
    *,
    coordinator: OutputTransactionCoordinator | None = None,
    protected_inputs: tuple[str | Path, ...] = (),
) -> AnalysisArtifactOutput:
    if type(request) is not AnalysisArtifactRequest:
        raise TypeError("analysis artifact admission requires exact request")
    selected = get_output_transaction_coordinator() if coordinator is None else coordinator
    if type(selected) is not OutputTransactionCoordinator:
        raise TypeError("analysis artifact coordinator must be exact")
    return AnalysisArtifactOutput(request, coordinator=selected, protected_inputs=protected_inputs)


__all__ = [
    "ANALYSIS_KIND_ATTR",
    "ANALYSIS_SCHEMA_ATTR",
    "ANALYSIS_SCHEMA_NAME",
    "ANALYSIS_SCHEMA_VERSION",
    "ANALYSIS_SCHEMA_VERSION_V2",
    "ANALYSIS_SCHEMA_VERSION_V3",
    "ANALYSIS_SCHEMA_VERSION_STITCH_NEUTRAL",
    "ANALYSIS_SCHEMA_VERSION_XU_STITCH_NEUTRAL",
    "ANALYSIS_SCHEMA_VERSION_STITCH_NATIVE",
    "ANALYSIS_SCHEMA_VERSION_XU_STITCH_NATIVE",
    "ANALYSIS_SCHEMA_VERSION_ATTR",
    "AnalysisArtifactCleanupPending",
    "AnalysisArtifactError",
    "AnalysisArtifactHeader",
    "AnalysisArtifactInspection",
    "AnalysisArtifactInvalid",
    "OlderStitchArtifact",
    "AnalysisArtifactKind",
    "AnalysisArtifactOutput",
    "AnalysisArtifactOutputSnapshot",
    "AnalysisArtifactOverwrite",
    "AnalysisArtifactPayload",
    "AnalysisArtifactProjectionInvalid",
    "AnalysisArtifactReceipt",
    "AnalysisArtifactResultProjection",
    "AnalysisArtifactRequest",
    "admit_analysis_artifact",
    "analysis_output_refusal",
    "analysis_execution_attestation_digest",
    "canonical_analysis_execution_attestation",
    "canonical_analysis_provenance",
    "inspect_analysis_artifact",
    "inspect_analysis_artifact_header",
    "stitch_record_identity",
    "rsm_record_identity",
    "project_analysis_artifact_result",
    "read_analysis_artifact",
]
