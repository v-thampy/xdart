from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace
from typing import Any

import numpy as np

from xrd_tools.io import (
    AppendDisposition,
    AppendIntent,
    AppendPreflightCleanupError,
    AppendPreflightState,
    AppendRefused,
    AppendSource,
    begin_same_run_lineage,
    extend_same_run_lineage,
    prepare_append_preflight,
    qualify_append,
    science_fingerprint,
)
from xrd_tools.io.append import seal_append_epoch
from xrd_tools.reduction import (
    CompositeSink,
    NexusSink,
    TransactionalXYESink,
)
from xrd_tools.session import (
    DynamicAccountingLimits,
    DynamicFrameIdentity,
    DynamicRunAccounting,
    DynamicRunState,
    FlushPolicy,
    Light1DBufferLayout,
    Light1DCustodySlot,
    Light1DFundingMode,
    Light1DLayout,
    Light1DModeLayout,
    SessionResourceAuthority,
    StageLedger,
    acquire_light_1d_retention,
    open_headless_scan_session,
    required_result_modes,
    resolve_session_policy,
)
from xrd_tools.session.policy import requirements_from
from xrd_tools.session.readiness import gi_companion_modes_2d
from xrd_tools.core import DEFAULT_MODE_KEY
from xrd_tools.core.scan import SourceKind
from xrd_tools.core.staging import (
    browse_publication_max_items,
    heavy_window,
    total_physical_ram_bytes,
)
from xrd_tools.session.run_configuration import heavy_residency_choice
from xrd_tools.session.display_logic import xye_prefix_for_unit
from xrd_tools.sources.execution_graph import (
    append_source_from_execution_graph,
    source_execution_projection,
    source_snapshots_projection,
    stable_lineage_projection,
)

from ..contracts import AdmittedOutput, PlannedOutput, SourceExecutionStamp
from ..output_preflight import (
    _background_resource_terms,
    _merge_background_binding,
)


_MISSING = object()
_SUPPORTED_LINEAGE_FRAME_CEILING = 1_000_000
_POST_G2_PIPELINE_V2_FIELDS = frozenset({
    "writer_settlement_batch_size", "nexus_record_batch_size",
    "reduction_inflight", "semantic_checkpoint_frame_cap",
    "staging_frame_cap",
})
_POST_G2_OUTPUT_DIAGNOSTICS_FIELDS = frozenset({
    "save_xye", "durable_fsync",
})
_UNSAFE_UNFUNDED_STAGING_ENV = (
    "XDART_UNSAFE_UNFUNDED_STAGING_DIAGNOSTIC"
)
_UNSAFE_UNFUNDED_STAGING_KEY = (
    "_post_g2_unfunded_staging_diagnostic_v1"
)
_UNSAFE_UNFUNDED_STAGING_FIELDS = frozenset({
    "mode", "checkpoint", "staging_frame_cap", "max_frames",
})
_UNSAFE_UNFUNDED_STAGING_PAIR = (10_000, 10_008)
_UNSAFE_UNFUNDED_STAGING_MAX_FRAMES = 3_621
_UNSAFE_UNFUNDED_STAGING_RESIDUAL_BYTES = 16 * 1024 ** 3
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PostG2PipelineV2Choice:
    writer_settlement_batch_size: int
    nexus_record_batch_size: int
    reduction_inflight: int
    semantic_checkpoint_frame_cap: int
    staging_frame_cap: int = 64

    @property
    def buffers_nexus_records_across_calls(self) -> bool:
        return self.nexus_record_batch_size > 1


@dataclass(frozen=True, slots=True)
class _PostG2OutputDiagnosticsChoice:
    save_xye: bool = True
    durable_fsync: bool = True
    explicit: bool = False


def _post_g2_pipeline_v2_choice(configuration, *, coordinated):
    value = configuration.run_options.get("_post_g2_pipeline_v2", _MISSING)
    if value is _MISSING:
        if (
            configuration.live_mode
            or configuration.batch_mode
            or configuration.output_mode != "Overwrite"
            or not coordinated
            or configuration.processing_mode == "Int 1D (XYE)"
        ):
            return None
        return _PostG2PipelineV2Choice(1, 8, 8, 16, 64)
    if not isinstance(value, Mapping):
        raise TypeError("post-G2 pipeline V2 choice must be one exact mapping")
    if set(value) != _POST_G2_PIPELINE_V2_FIELDS:
        raise ValueError("post-G2 pipeline V2 choice has an invalid exact keyset")
    row = tuple(value[field] for field in (
        "writer_settlement_batch_size", "nexus_record_batch_size",
        "reduction_inflight", "semantic_checkpoint_frame_cap",
    ))
    staging = value["staging_frame_cap"]
    row += (staging,)
    if any(type(item) is not int for item in row):
        raise TypeError("post-G2 pipeline V2 values must be exact integers")
    settlement, record, inflight, checkpoint, staging = row
    if not (1 <= settlement <= 16 and 1 <= record <= 16
            and 1 <= inflight <= 64 and 1 <= checkpoint <= 10_000
            and 9 <= staging <= 10_008):
        raise ValueError("post-G2 pipeline V2 values are out of bounds")
    if settlement > inflight or record > checkpoint or checkpoint % settlement:
        raise ValueError("post-G2 pipeline V2 values violate batching bounds")
    if checkpoint > staging - 8:
        raise ValueError(
            "post-G2 pipeline V2 semantic checkpoint exceeds the staging "
            "cap minus its 8-frame safety margin"
        )
    if configuration.live_mode:
        raise ValueError("post-G2 pipeline V2 requires a non-Live run")
    if configuration.batch_mode:
        raise ValueError("post-G2 pipeline V2 requires a non-batch run")
    if configuration.output_mode != "Overwrite":
        raise ValueError("post-G2 pipeline V2 requires Overwrite output")
    if not coordinated:
        raise ValueError("post-G2 pipeline V2 requires a coordinated run")
    if configuration.processing_mode == "Int 1D (XYE)":
        raise ValueError("post-G2 pipeline V2 requires Nexus output")
    return _PostG2PipelineV2Choice(*row)


def _unsafe_unfunded_staging_requested(
    pipeline_v2: _PostG2PipelineV2Choice | None,
    *,
    run_options: Mapping[str, object],
    env: Mapping[str, str],
    frame_count: int,
) -> bool:
    raw = env.get(_UNSAFE_UNFUNDED_STAGING_ENV)
    marker = run_options.get(_UNSAFE_UNFUNDED_STAGING_KEY, _MISSING)
    if marker is _MISSING:
        return False
    if raw is None or not str(raw).strip():
        raise ValueError(
            "unsafe unfunded staging marker requires its private process "
            "opt-in"
        )
    if str(raw).strip() != "1":
        raise ValueError(
            f"{_UNSAFE_UNFUNDED_STAGING_ENV} must be exactly 1 when enabled"
        )
    pair = None if pipeline_v2 is None else (
        pipeline_v2.semantic_checkpoint_frame_cap,
        pipeline_v2.staging_frame_cap,
    )
    if pair != _UNSAFE_UNFUNDED_STAGING_PAIR:
        raise ValueError(
            "unsafe unfunded staging requires the exact private "
            "checkpoint=10000/staging=10008 pair"
        )
    if not isinstance(marker, Mapping):
        raise ValueError(
            "unsafe unfunded staging requires its exact persisted marker"
        )
    if set(marker) != _UNSAFE_UNFUNDED_STAGING_FIELDS or marker != {
        "mode": "UNSAFE_UNFUNDED",
        "checkpoint": 10_000,
        "staging_frame_cap": 10_008,
        "max_frames": _UNSAFE_UNFUNDED_STAGING_MAX_FRAMES,
    }:
        raise ValueError("unsafe unfunded staging marker is invalid")
    if frame_count > _UNSAFE_UNFUNDED_STAGING_MAX_FRAMES:
        raise ValueError(
            "unsafe unfunded staging exceeds its 3621-frame source ceiling"
        )
    return True


def _unsafe_unfunded_staging_projection(
    allocation,
    *,
    frame_count: int,
    checkpoint: int,
) -> tuple[int, int, int]:
    requirements = allocation.requirements
    retained_rows = min(max(0, int(frame_count)), int(checkpoint))
    one_d = requirements.result_1d_bytes
    two_d = requirements.result_2d_bytes
    thumbnail = requirements.thumbnail_bytes
    writer_unfunded = max(
        0, retained_rows - allocation.staging_items,
    ) * (one_d + two_d + thumbnail)
    record_unfunded = (
        max(0, retained_rows - allocation.record_items) * one_d
        + max(0, retained_rows - allocation.record_heavy_items) * two_d
        + max(0, retained_rows - allocation.thumbnail_items) * thumbnail
    )
    publication_unfunded = (
        max(0, retained_rows - allocation.publication_items) * one_d
        + max(
            0, retained_rows - allocation.publication_heavy_items,
        ) * two_d
        + max(0, retained_rows - allocation.thumbnail_items) * thumbnail
    )
    unfunded_bytes = (
        writer_unfunded + record_unfunded + publication_unfunded
    )
    projected_bytes = (
        int(allocation.assigned_bytes) + unfunded_bytes
    )
    return retained_rows, unfunded_bytes, projected_bytes


def _post_g2_output_diagnostics_choice(configuration, *, pipeline_v2):
    value = configuration.run_options.get(
        "_post_g2_output_diagnostics_v1",
        _MISSING,
    )
    if value is _MISSING:
        return _PostG2OutputDiagnosticsChoice()
    if pipeline_v2 is None:
        raise ValueError("post-G2 output diagnostics require pipeline V2")
    if not isinstance(value, Mapping):
        raise TypeError(
            "post-G2 output diagnostics must be one exact mapping"
        )
    if set(value) != _POST_G2_OUTPUT_DIAGNOSTICS_FIELDS:
        raise ValueError(
            "post-G2 output diagnostics have an invalid exact keyset"
        )
    save_xye = value["save_xye"]
    durable_fsync = value["durable_fsync"]
    if type(save_xye) is not bool or type(durable_fsync) is not bool:
        raise TypeError(
            "post-G2 output diagnostics values must be exact booleans"
        )
    return _PostG2OutputDiagnosticsChoice(
        save_xye,
        durable_fsync,
        True,
    )


@dataclass(frozen=True, slots=True)
class HeavyResidencyFact:
    artifact: str; choice: str; resolution_source: str
    requested_heavy_bound: int; granted_staging_count: int
    granted_record_heavy_count: int; granted_publication_heavy_count: int; effective_display_count: int

    def log_line(self, prefix: str) -> str:
        return (
            f"{prefix} artifact={self.artifact} choice={self.choice} "
            f"source={self.resolution_source} requested={self.requested_heavy_bound} "
            f"staging={self.granted_staging_count} record-heavy={self.granted_record_heavy_count} "
            f"publication-heavy={self.granted_publication_heavy_count} "
            f"effective-display={self.effective_display_count}")


@dataclass(frozen=True, slots=True)
class _PreparedAdmission:
    owner: object; scan: object; plan: object; item: PlannedOutput
    provisional: AdmittedOutput; effective: AdmittedOutput; stop_signal: Event; context: tuple[object, ...]; resources: tuple[object, ...] | None; identity: object; consumed: bool = False; dormant_noop: bool = False
def _heavy_resolution_source(bound: int | None, env: dict[str, str]) -> str:
    if bound is not None: return "ui"
    raw = env.get("XDART_HEAVY_WINDOW")
    try:
        int(str(raw).strip())
    except (TypeError, ValueError): return "auto"
    return "environment" if str(raw).strip() else "auto"


def _direct_eiger_candidate(item, write_labels) -> bool:
    descriptor = getattr(item, "descriptor", None)
    stamp = getattr(item, "source_stamp", None)
    return (getattr(descriptor, "kind", None) is SourceKind.EIGER_MASTER
            and getattr(descriptor, "finalized", False)
            and len(write_labels) == getattr(stamp, "frame_count", -1))


def _light_policy_layout(
    configuration, plan, item, scan, write_labels, *, heavy_request=None,
    reduction_inflight=None, staging_frame_cap=None,
    semantic_checkpoint_frame_cap=None, unsafe_unfunded_staging=False,
    env=None, accepted_allocation=None,
):
    try:
        frame = next(value for value in scan.frames
                     if not write_labels or int(value.index) == int(write_labels[0]))
    except StopIteration:
        raise ValueError("dynamic light-1D write frame is absent from the Scan")
    descriptor = item.descriptor
    if descriptor is not None and descriptor.frame_shape is not None \
            and descriptor.dtype is not None:
        shape, native_dtype = descriptor.frame_shape, descriptor.dtype
    else:
        if frame.image is None:
            frame.load_image()
        shape, native_dtype = frame.image.shape, frame.image.dtype
    pixels = int(np.prod(shape)); G, R, Bw, Bmap = _background_resource_terms(configuration.background, pixels)
    requirements = requirements_from(SimpleNamespace(
        frame_shape=tuple(shape), dtype=np.dtype(native_dtype)), plan,
        background_bytes=G, resolver_background_bytes=R,
        worker_background_bytes=Bw, background_binding_bytes=Bmap)
    requested = max(1, int(configuration.max_cores))
    requests = {}
    if _direct_eiger_candidate(item, range(item.source_stamp.frame_count)):
        from xrd_tools.sources.eiger_direct_chunk import (
            direct_chunk_workspace_bytes,
        )
        requests["owner_block_bytes"] = direct_chunk_workspace_bytes(
            requirements.native_frame_bytes,
        )
    if heavy_request is not None:
        _, heavy_request = heavy_residency_choice({"heavy_window": heavy_request})
        requests.update({
            "staging_items": heavy_request, "record_heavy_items": heavy_request,
            "publication_heavy_items": heavy_request,
        })
    if staging_frame_cap is not None:
        if type(staging_frame_cap) is not int:
            raise TypeError("requested staging frame cap must be an exact int")
        if not 9 <= staging_frame_cap <= 10_008:
            raise TypeError(
                "requested staging frame cap must be in 9..10008"
            )
        if not unsafe_unfunded_staging:
            requests["staging_items"] = staging_frame_cap
    if reduction_inflight is not None:
        if type(reduction_inflight) is not int or reduction_inflight < 1:
            raise TypeError("requested reduction in-flight must be a positive int")
        requests["reduction_inflight"] = reduction_inflight
    frozen_env = dict(os.environ) if env is None else dict(env)
    policy = resolve_session_policy(
        requirements, requested_workers=requested, requests=requests or None,
        allocation=accepted_allocation, env=frozen_env,
    )
    if (
        reduction_inflight is None
        and not configuration.live_mode
        and policy.allocation.workers == 4
    ):
        policy = resolve_session_policy(
            requirements,
            requested_workers=requested,
            requests={**requests, "reduction_inflight": 16},
            allocation=accepted_allocation, env=frozen_env,
        )
    allocation = policy.allocation
    if (
        reduction_inflight is not None
        and allocation.reduction_inflight != reduction_inflight
    ):
        raise RuntimeError(
            "post-G2 pipeline in-flight request was not granted exactly"
        )
    if (
        staging_frame_cap is not None
        and not unsafe_unfunded_staging
        and allocation.staging_items != staging_frame_cap
    ):
        requirements = allocation.requirements
        unit_bytes = (
            requirements.native_frame_bytes
            + requirements.background_bytes
            + requirements.result_1d_bytes
            + requirements.result_2d_bytes
            + requirements.thumbnail_bytes
        )
        raise RuntimeError(
            "post-G2 staging request was not granted exactly: "
            f"requested={staging_frame_cap} granted={allocation.staging_items} "
            f"staging-unit={unit_bytes}B "
            f"requested-staging={staging_frame_cap * unit_bytes}B "
            f"modeled-session={allocation.assigned_bytes}B "
            f"envelope={allocation.envelope_bytes}B"
        )
    if (
        semantic_checkpoint_frame_cap is not None
        and not unsafe_unfunded_staging
    ):
        checkpoint = int(semantic_checkpoint_frame_cap)
        requirements = allocation.requirements
        retained_grants = {}
        if requirements.result_1d_bytes:
            retained_grants.update({
                "record-1d": allocation.record_items,
                "publication-1d": allocation.publication_items,
            })
        if requirements.result_2d_bytes:
            retained_grants.update({
                "record-2d": allocation.record_heavy_items,
                "publication-2d": allocation.publication_heavy_items,
            })
        if requirements.thumbnail_bytes:
            retained_grants["thumbnail"] = allocation.thumbnail_items
        short = {
            name: value
            for name, value in retained_grants.items()
            if value < checkpoint
        }
        if short:
            raise RuntimeError(
                "post-G2 semantic checkpoint retention was not funded: "
                f"checkpoint={checkpoint} grants={retained_grants} "
                f"insufficient={short}"
            )
    if unsafe_unfunded_staging:
        if staging_frame_cap is None:
            raise ValueError("unsafe unfunded staging requires an explicit cap")
        retained_rows, unfunded_bytes, projected_bytes = (
            _unsafe_unfunded_staging_projection(
                allocation,
                frame_count=item.source_stamp.frame_count,
                checkpoint=staging_frame_cap - 8,
            )
        )
        total = max(0, int(total_physical_ram_bytes() or 0))
        limit = min(
            total // 2,
            max(0, total - _UNSAFE_UNFUNDED_STAGING_RESIDUAL_BYTES),
        )
        if total <= 0 or projected_bytes > limit:
            raise RuntimeError(
                "unsafe unfunded staging exceeds its modeled host-memory "
                f"guard: retained={retained_rows} "
                f"unfunded={unfunded_bytes}B "
                f"projected={projected_bytes}B limit={limit}B total={total}B"
            )
    interval = 8 if plan.integration_2d is not None else 1000
    policy = replace(policy, flush=FlushPolicy(
        interval=interval,
        cap=int(
            staging_frame_cap
            if unsafe_unfunded_staging else allocation.staging_items
        ),
        margin=8,
    ))
    one_d = getattr(plan, "integration_1d", None)
    modes = tuple(mode.key for mode in required_result_modes(plan)
                  if mode.kind == "1d")
    if one_d is None or not modes:
        return policy, None, 0, 0, frame
    active = (configuration.gi.mode_1d if configuration.gi.enabled
              else DEFAULT_MODE_KEY)
    if active not in modes:
        raise ValueError("active 1-D mode is absent from the reduction plan")
    dtype = np.dtype(np.float64)

    def buffer(mode: str, role: str) -> Light1DBufferLayout:
        return Light1DBufferLayout(
            int(one_d.npt), dtype.itemsize, f"{mode}:{role}", dtype.str,
            shared=False,
        )

    layout = Light1DLayout(
        tuple(
            Light1DModeLayout(
                mode, buffer(mode, "coordinate"), buffer(mode, "intensity"),
                buffer(mode, "uncertainty") if one_d.error_model else None,
            )
            for mode in modes
        ),
        active,
    )
    total = max(0, int(total_physical_ram_bytes() or 0))
    rows = browse_publication_max_items(int(one_d.npt), total_ram_bytes=total)
    ceiling = min(1024 ** 3, int(0.05 * total)) if total else (
        layout.shared_bytes + rows * layout.per_row_unique_ndarray_bytes
    )
    return policy, layout, rows, ceiling, frame


def _supported_lineage_frame_count(stamp: SourceExecutionStamp) -> int:
    count = int(stamp.frame_count)
    if count > _SUPPORTED_LINEAGE_FRAME_CEILING:
        raise ValueError(
            "dynamic output lineage exceeds the supported "
            f"{_SUPPORTED_LINEAGE_FRAME_CEILING}-frame ceiling"
        )
    return count


def _target_key(path: Path | str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _stable_lineage(item: PlannedOutput) -> tuple[object, ...]:
    return stable_lineage_projection(
        item.graph, target=item.group.target,
    )


def _append_source(
    stamp: SourceExecutionStamp,
    item: PlannedOutput,
    *,
    generation: int,
) -> AppendSource:
    graph = item.graph
    if stamp is not item.source_stamp:
        raise ValueError("Append source stamp must be the admitted item stamp")
    return append_source_from_execution_graph(graph, generation=generation)


def _mode_tokens(configuration) -> tuple[str, ...]:
    one_key = (
        configuration.gi.mode_1d
        if configuration.gi.enabled else "default"
    )
    two_key = (
        configuration.gi.mode_2d
        if configuration.gi.enabled else "default"
    )
    modes = [f"1d:{one_key}"]
    if not configuration.skip_2d:
        modes.append(f"2d:{two_key}")
        if configuration.gi.enabled:
            # Every DIRECT 2-D result the run writes is part of the Append
            # identity, so a changed mode set takes the incompatibility flow.
            modes.extend(
                f"2d:{mode}"
                for mode in gi_companion_modes_2d(
                    configuration.bai_2d_args, primary=two_key,
                )
            )
    return tuple(modes)


def _science_projection(
    configuration,
    signed: object,
) -> dict[str, Any]:
    if type(signed) is not dict:
        raise TypeError("Append requires the exact signed science mapping")
    values = dict(signed)
    assets = values.pop("accepted_scientific_assets", None)
    if values != configuration.as_provenance():
        raise ValueError(
            "signed science mapping differs from the frozen run configuration"
        )
    if type(assets) is not dict or set(assets) != {
        "poni_values", "poni_detector_config_json", "poni_sha256", "mask_sha256",
    }:
        raise TypeError("accepted scientific asset identity is malformed")
    if assets["poni_values"] is not None and type(
        assets["poni_values"]
    ) is not dict:
        raise TypeError("accepted PONI values are malformed")
    config_text = assets["poni_detector_config_json"]
    if (assets["poni_values"] is None) != (config_text is None):
        raise TypeError("accepted PONI and detector config must be paired")
    if config_text is not None:
        try:
            config = json.loads(config_text)
            canonical = json.dumps(
                config, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TypeError("accepted detector config is malformed") from exc
        if (canonical != config_text or type(config) is not dict
                or type(config.get("orientation")) is not int
                or config["orientation"] not in range(1, 5)):
            raise TypeError("accepted detector config is malformed")
        try:
            from xrd_tools.integrate.calibration import (
                detector_calibration_from_projection,
            )
            calibration = detector_calibration_from_projection(
                assets["poni_values"], detector_config=config,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("accepted PONI projection is malformed") from exc
        if calibration.parallax is not None and (
            configuration.poni_values != assets["poni_values"]
        ):
            raise ValueError(
                "accepted PONI projection differs from frozen calibration"
            )
    for key in ("poni_sha256", "mask_sha256"):
        digest = assets[key]
        if digest is not None and (
            type(digest) is not str
            or len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)
        ):
            raise TypeError(f"accepted {key} is malformed")
    required = {
        "processing_mode", "bai_1d_args", "bai_2d_args", "gi",
        "threshold", "poni_values",
    }
    if not required.issubset(values):
        raise TypeError("signed science mapping is incomplete")
    if (
        type(values["processing_mode"]) is not str
        or type(values["bai_1d_args"]) is not dict
        or type(values["bai_2d_args"]) is not dict
        or type(values["gi"]) is not dict
        or type(values["threshold"]) is not dict
        or values["poni_values"] is not None
        and type(values["poni_values"]) is not dict
    ):
        raise TypeError("signed numerical science mapping is malformed")
    return {
        "processing_mode": values["processing_mode"],
        "bai_1d_args": values["bai_1d_args"],
        "bai_2d_args": values["bai_2d_args"],
        "gi": values["gi"],
        "threshold": values["threshold"],
        "poni_values": values["poni_values"],
        "accepted_scientific_assets": assets,
        **({"background": values["background"]} if "background" in values else {}),
    }


def _append_intent(
    configuration,
    item: PlannedOutput,
    *,
    generation: int,
    science_identity: str,
) -> AppendIntent:
    frame_count = _supported_lineage_frame_count(item.source_stamp)
    return AppendIntent(
        entry=item.source_spec.entry or "entry",
        source_base=configuration.project_root or "",
        source_identity=science_fingerprint(_stable_lineage(item)),
        science_fingerprint=science_identity,
        modes=_mode_tokens(configuration),
        source=_append_source(
            item.source_stamp, item, generation=generation,
        ),
        labels=tuple(range(
            item.source_stamp.first_label,
            item.source_stamp.first_label + frame_count,
        )),
    )


def preview_append_decision(
    configuration,
    item: PlannedOutput,
    signed_science: dict[str, Any],
):
    science_identity = science_fingerprint(
        _science_projection(configuration, signed_science)
    )
    return qualify_append(
        item.target,
        _append_intent(
            configuration,
            item,
            generation=1,
            science_identity=science_identity,
        ),
    )


def _writer_source_snapshots(
    item: PlannedOutput,
) -> dict[str, dict[str, Any]]:
    return source_snapshots_projection(
        item.graph, writer=True,
    )


class DynamicOutputAdapter:
    def __init__(self, configuration) -> None:
        self.configuration = configuration
        self._graphs: dict[str, dict[str, Any]] = {}
        self._pending_preflights: list[Any] = []
        self._pending_nexus: list[NexusSink] = []
        self._pending_xye: list[TransactionalXYESink] = []
        self._science_identity: str | None = None
        self._current: dict[str, Any] | None = None
        self._command_lock = RLock()
        self._activating = False
        self._stop_requested = False
        self.resource_facts: tuple[HeavyResidencyFact, ...] = ()

    @staticmethod
    def _discard_identity(values: list[Any], owner: Any) -> None:
        values[:] = [value for value in values if value is not owner]

    def _release_construction_custody(
        self, *, preflight=None, nexus=None, xye=None,
    ) -> None:
        if preflight is not None:
            self._discard_identity(self._pending_preflights, preflight)
        if nexus is not None:
            self._discard_identity(self._pending_nexus, nexus)
        if xye is not None:
            self._discard_identity(self._pending_xye, xye)

    def _settle_construction_custody(self) -> None:
        primary: BaseException | None = None
        for xye in tuple(self._pending_xye):
            try:
                xye.abort(None)
            except BaseException as error:
                primary = primary or error
            else:
                self._discard_identity(self._pending_xye, xye)
        for nexus in tuple(self._pending_nexus):
            try:
                nexus.abort(None)
            except BaseException as error:
                primary = primary or error
            else:
                self._discard_identity(self._pending_nexus, nexus)
                preflight = nexus.append_preflight
                if preflight is not None and preflight.snapshot.state in {
                    AppendPreflightState.NOOP,
                    AppendPreflightState.COMMITTED,
                    AppendPreflightState.ABORTED,
                }:
                    self._discard_identity(
                        self._pending_preflights, preflight,
                    )
        bound = {
            id(nexus.append_preflight)
            for nexus in self._pending_nexus
            if nexus.append_preflight is not None
        }
        for preflight in tuple(self._pending_preflights):
            if id(preflight) in bound:
                continue
            try:
                state = preflight.snapshot.state
                if state is AppendPreflightState.RESERVED:
                    state = preflight.abort().state
                elif state is AppendPreflightState.RETRYABLE:
                    state = preflight.retry_cleanup().state
                if state not in {
                    AppendPreflightState.NOOP,
                    AppendPreflightState.COMMITTED,
                    AppendPreflightState.ABORTED,
                }:
                    raise RuntimeError(
                        "Append construction cleanup remains unsettled: "
                        f"{state.value}"
                    )
            except BaseException as error:
                primary = primary or error
            else:
                self._discard_identity(
                    self._pending_preflights, preflight,
                )
        if primary is not None:
            raise primary

    @property
    def session(self):
        return None if self._current is None else self._current["session"]

    @property
    def sink(self):
        return None if self._current is None else self._current["sink"]

    @property
    def accounting(self):
        return None if self._current is None else self._current["accounting"]

    @property
    def write_labels(self) -> tuple[int, ...]:
        return (
            ()
            if self._current is None
            else tuple(self._current["write_labels"])
        )

    @property
    def persisted_prefix_labels(self) -> tuple[int, ...]:
        """Exact cross-run prefix captured by the locked Append preflight."""

        return (
            ()
            if self._current is None
            else tuple(self._current["persisted_prefix_labels"])
        )

    def finalized_display_owners(self) -> tuple[object, ...]:
        """Display owners whose output graphs are terminal and quiescent."""

        with self._command_lock:
            candidates = tuple(
                (
                    graph.get("display_owner"),
                    graph.get("session"),
                    graph.get("accounting"),
                    graph.get("transition"),
                )
                for graph in self._graphs.values()
                if graph.get("display_owner") is not None
            )
        terminal = {
            DynamicRunState.FINISHED,
            DynamicRunState.STOPPED,
            DynamicRunState.ABORTED,
        }
        owners: list[object] = []
        for owner, session, accounting, transition in candidates:
            if (
                session is not None
                and accounting is not None
                and transition is None
                and not session.is_running
                and accounting.snapshot().state in terminal
                and all(owner is not prior for prior in owners)
            ):
                owners.append(owner)
        return tuple(owners)

    def background_binding(self, label: int): return None if self._current is None else next((value for value in self._current["background_bindings"] if value[0] == int(label)), None)
    def background_admission_context(self):
        graph = self._current
        if graph is None or graph.get("policy") is None:
            raise RuntimeError("Background admission lost its active graph")
        return graph["item"], graph["policy"].allocation.requirements
    def admit_background_binding(self, binding):
        graph = self._current if self._current is not None else (_ for _ in ()).throw(RuntimeError("Background binding lost its active graph"))
        if graph.get("policy") is None:
            raise RuntimeError("Background binding lost its active graph")
        limit = graph["policy"].allocation.requirements.background_binding_bytes; merged = _merge_background_binding(graph["background_bindings"], binding, limit=limit); graph["background_bindings"] = merged; return next(value for value in merged if value[0] == binding[0])

    def predecessor_owner(self, item: PlannedOutput):
        graph = self._current
        if graph is None or graph.get("display_owner") is None:
            return None
        if graph.get("target") == _target_key(item.target):
            # Keep this target's owner through admission so activation can
            # reject a changed lineage before a fresh Overwrite sink is built.
            return None
        return graph["display_owner"]

    def predecessor_owner_for_target(self, target: Path):
        graph = self._current
        return (None if graph is None or graph.get("target") == _target_key(target)
                else graph.get("display_owner"))

    def drop_released_predecessor(self, owner: object) -> None:
        graph = self._current
        if graph is None or graph.get("display_owner") is not owner:
            raise RuntimeError("dynamic predecessor identity changed")
        if graph.get("dormant"):
            if graph["session"] is not None or graph["accounting"] is not None or any(getattr(owner, name) is not None for name in ("light_lease", "light_slot", "light_hooks", "light_retry_token", "light_unsubscribe")): raise RuntimeError("dormant predecessor still owns runtime roots")
            graph["background_bindings"] = (); self._current = None; return
        session = graph["session"]
        if (
            session.is_running
            or graph["accounting"].snapshot().state not in {
                DynamicRunState.FINISHED, DynamicRunState.STOPPED,
            }
            or graph["transition"] is not None or graph["projection_pending"]
            or any(getattr(owner, name) is not None for name in (
                "light_lease", "light_slot", "light_hooks",
                "light_retry_token", "light_unsubscribe",
            ))
        ):
            raise RuntimeError("dynamic predecessor graph is not terminal")
        self._graphs.pop(graph["target"], None)
        self._current = None

    def finish_predecessor(self, owner: object):
        graph = self._current
        if graph is None or graph.get("display_owner") is not owner:
            raise RuntimeError("dynamic predecessor identity changed")
        return SimpleNamespace(failed=False, error=None) if graph.get("dormant") else self._finish_transition(graph)

    def _revision(self, graph: dict[str, Any], item: PlannedOutput) -> int:
        exact = item.source_stamp.execution_identity_v1
        revisions = graph["revisions"]
        value = revisions.get(exact)
        if value is None:
            value = len(revisions) + 1
            revisions[exact] = value
        return int(value)

    def prepare_admission(self, scan, plan, item: PlannedOutput,
                          decision: AdmittedOutput, stop_signal: Event, *, qualify):
        """Freeze the final allocation and array-free Background bindings."""
        if type(item) is not PlannedOutput or type(decision) is not AdmittedOutput \
                or decision.item is not item or type(stop_signal) is not Event or not callable(qualify):
            raise TypeError("dynamic admission preparation inputs are not exact")
        if stop_signal.is_set(): raise RuntimeError("admission cancelled")
        env = dict(os.environ); frame_count = _supported_lineage_frame_count(item.source_stamp)
        pipeline_v2 = _post_g2_pipeline_v2_choice(self.configuration, coordinated=True)
        diagnostics = _post_g2_output_diagnostics_choice(self.configuration, pipeline_v2=pipeline_v2)
        unsafe = _unsafe_unfunded_staging_requested(
            pipeline_v2, run_options=self.configuration.run_options, env=env,
            frame_count=frame_count)
        if unsafe and not (getattr(item.descriptor, "kind", None) is SourceKind.EIGER_MASTER
                           and item.descriptor.finalized and item.descriptor.frame_shape is not None
                           and item.descriptor.dtype is not None):
            raise ValueError("unsafe unfunded staging requires one finalized descriptor-complete Eiger master")
        _choice, heavy = heavy_residency_choice(self.configuration.run_options)
        key = _target_key(item.target); graph = self._graphs.get(key)
        dormant_noop = (
            graph is None
            and self.configuration.output_mode == "Append"
            and not self.configuration.live_mode
            and self.configuration.background.mode == "None"
            and not decision.labels
        )
        if dormant_noop:
            effective = replace(decision, background_bindings=())
            prepared = _PreparedAdmission(
                self, scan, plan, item, decision, effective, stop_signal,
                (pipeline_v2, diagnostics, env, unsafe, heavy),
                None, None, dormant_noop=True,
            )
            object.__setattr__(prepared, "identity", prepared)
            return prepared
        dormant = self._current if graph is None and self._current is not None and self._current.get("dormant") and self._current.get("policy") is not None and self._current["target"] == key and self._current["lineage"] == _stable_lineage(item) else None; authority = graph if graph is not None else dormant; accepted = None if authority is None else authority["policy"].allocation
        policy, layout, rows, ceiling, first = _light_policy_layout(
            self.configuration, plan, item, scan, decision.labels, heavy_request=heavy,
            reduction_inflight=(pipeline_v2.reduction_inflight
                                if pipeline_v2 else None),
            staging_frame_cap=(pipeline_v2.staging_frame_cap if pipeline_v2 else None),
            semantic_checkpoint_frame_cap=(pipeline_v2.semantic_checkpoint_frame_cap if pipeline_v2 else None),
            unsafe_unfunded_staging=unsafe, env=env, accepted_allocation=accepted)
        prior = () if authority is None else authority["background_bindings"]
        if authority is not None and policy.allocation is not authority["policy"].allocation: raise RuntimeError("Live allocation identity changed")
        bindings = qualify(policy, prior)
        effective = replace(decision, background_bindings=bindings)
        prepared = _PreparedAdmission(self, scan, plan, item, decision, effective, stop_signal,
            (pipeline_v2, diagnostics, env, unsafe, heavy), (policy, layout, rows, ceiling, first), None)
        object.__setattr__(prepared, "identity", prepared); return prepared
    def activate(self, *args, cancelled=lambda: False, **kwargs):
        if len(args) != 1 or type(args[0]) is not _PreparedAdmission: raise TypeError("activation requires one prepared admission")
        with self._command_lock:
            if self._activating:
                raise RuntimeError("dynamic output activation is already active")
            if self._stop_requested or cancelled():
                raise RuntimeError("admission cancelled")
            self._activating = True
        value = error = None
        try:
            value = self._activate_owned(
                *args,
                cancelled=lambda: self._stop_requested or cancelled(),
                **kwargs,
            )
            if self._stop_requested or cancelled():
                raise RuntimeError("admission cancelled")
        except BaseException as primary:
            error = primary
            try:
                self._settle_construction_custody()
            except BaseException as cleanup:
                raise primary from cleanup
        finally:
            with self._command_lock:
                self._activating = False
                stop = self._stop_requested or cancelled()
                self._stop_requested = self._stop_requested or stop
        if stop:
            try:
                self.stop()
            except BaseException as cleanup:
                if error is not None:
                    raise error.with_traceback(error.__traceback__) from cleanup
                raise
        if error is not None:
            raise error.with_traceback(error.__traceback__)
        if stop:
            raise RuntimeError("admission cancelled")
        return value

    def _activate_owned(
        self,
        preparation: _PreparedAdmission,
        *,
        record_store,
        run_provenance,
        cancelled,
        publication_store=None,
        display_owner=None,
        display_state=None,
        source_owner=None,
        gui_thread_id=None,
        light_cancel=None,
        light_drain=None,
        light_verify=None,
        on_frame_completed=None,
        on_checkpoint_recoverable=None,
        resource_fact_sink=None,
    ):
        if type(preparation) is not _PreparedAdmission or preparation.owner is not self \
                or preparation.identity is not preparation or preparation.consumed or preparation.stop_signal.is_set():
            raise RuntimeError("dynamic admission preparation identity is invalid or consumed")
        object.__setattr__(preparation, "consumed", True)
        scan, plan, item, decision = (preparation.scan, preparation.plan,
                                      preparation.item, preparation.effective)
        if _target_key(item.target) != _target_key(item.group.target):
            raise ValueError("planned output target differs from its canonical group target")
        pipeline_v2, output_diagnostics, resource_env, \
            unsafe_unfunded_staging, heavy_request = preparation.context
        if preparation.dormant_noop:
            if preparation.resources is not None:
                raise RuntimeError("dormant Append no-op preparation is invalid")
            policy = layout = first_write_frame = None
            requested_rows = ceiling = 0
            if (
                self.configuration.output_mode != "Append"
                or self.configuration.live_mode
                or self.configuration.background.mode != "None"
                or preparation.provisional.labels
                or decision.labels
                or decision.background_bindings
            ):
                raise RuntimeError("dormant Append no-op preparation changed")
        else:
            if preparation.resources is None:
                raise RuntimeError("dynamic admission preparation lost resources")
            policy, layout, requested_rows, ceiling, first_write_frame = preparation.resources
        if type(run_provenance) is not dict:
            raise TypeError("dynamic output requires exact run provenance")
        mount_values = (
            publication_store, display_owner, display_state, gui_thread_id,
            light_cancel, light_drain, light_verify, on_frame_completed,
            on_checkpoint_recoverable,
        )
        pipeline_coordinated = all(value is not None for value in mount_values)
        frame_count = _supported_lineage_frame_count(item.source_stamp)
        science_identity = science_fingerprint(_science_projection(
            self.configuration,
            run_provenance.get("scientific_signature"),
        ))
        if cancelled():
            raise RuntimeError("admission cancelled")
        science_identity_was_unset = self._science_identity is None
        if self._science_identity is None:
            self._science_identity = science_identity
        elif self._science_identity != science_identity:
            raise ValueError(
                "dynamic output science identity changed within one run"
            )
        target = Path(item.group.target)
        key = _target_key(target)
        lineage = _stable_lineage(item)
        labels = tuple(range(
            item.source_stamp.first_label,
            item.source_stamp.first_label + frame_count,
        ))
        if tuple(int(frame.index) for frame in scan.frames) != labels:
            raise ValueError("dynamic output requires the complete exact Scan")
        graph = self._graphs.get(key)
        if preparation.dormant_noop and graph is not None:
            raise RuntimeError("dormant Append no-op acquired an active graph")
        if graph is not None:
            if cancelled():
                raise RuntimeError("admission cancelled")
            if graph["lineage"] != lineage:
                raise ValueError(
                    "same target cannot change its exact output lineage"
                )
            if graph["policy"].allocation is not policy.allocation:
                raise RuntimeError("prepared allocation identity changed")
            revision = self._revision(graph, item)
            intent = _append_intent(
                self.configuration,
                item,
                generation=revision,
                science_identity=science_identity,
            )
            if graph["intent"] != intent:
                nexus = graph["nexus"]
                if nexus is not None:
                    extension = graph["session"].extend_live(intent)
                    write_labels = tuple(extension.write_labels)
                    if self.configuration.output_mode == "Append":
                        graph["persisted_prefix_labels"] = tuple(
                            extension.skip_labels
                        )
                else:
                    extension = extend_same_run_lineage(
                        graph["xye_lineage"], graph["intent"], intent,
                    )
                    if extension.disposition is AppendDisposition.REFUSE:
                        raise AppendRefused(extension)
                    write_labels = tuple(extension.write_labels)
                    graph["xye_pending"] = extension
                graph["intent"] = intent
            else:
                write_labels = ()
            graph["scan"] = scan
            graph["item"] = item
            graph["write_labels"] = write_labels
            graph["background_bindings"] = decision.background_bindings
            self._arm(graph, write_labels, revision)
            self._current = graph
            if cancelled():
                raise RuntimeError("admission cancelled")
            return graph["session"], False

        revision = 1
        intent = _append_intent(
            self.configuration,
            item,
            generation=revision,
            science_identity=science_identity,
        )
        modes = required_result_modes(plan)
        if tuple(f"{mode.kind}:{mode.key}" for mode in modes) != intent.modes:
            raise ValueError("Append intent modes differ from the reduction plan")
        xye_only = self.configuration.processing_mode == "Int 1D (XYE)"
        if xye_only and self.configuration.output_mode == "Append":
            raise ValueError("XYE-only Append has no persisted lineage owner")
        coordinated = any(value is not None for value in mount_values)
        if coordinated and any(value is None for value in mount_values):
            raise TypeError("dynamic GUI light mount requires one exact graph")
        lease = hooks = slot = None
        session = None
        loaded_scout_frame = None

        preflight = None
        persisted_prefix_labels: tuple[int, ...] = ()
        if not xye_only and self.configuration.output_mode == "Append":
            if cancelled():
                raise RuntimeError("admission cancelled")
            try:
                preflight = prepare_append_preflight(
                    target, intent, file_lock=self._command_lock,
                )
            except AppendPreflightCleanupError as error:
                self._pending_preflights.append(error.owner)
                raise
            self._pending_preflights.append(preflight)
            preflight_snapshot = preflight.snapshot
            persisted_prefix_labels = tuple(preflight_snapshot.skip_labels)
            if (
                preparation.dormant_noop
                and preflight_snapshot.disposition is not AppendDisposition.SKIP
            ):
                raise RuntimeError(
                    "dormant Append no-op changed under the locked preflight"
                )
            if self.configuration.background.mode != "None" and persisted_prefix_labels:
                from xrd_tools.io.record_writer import _validate_persisted_background_bindings
                _validate_persisted_background_bindings(target, decision.background_bindings, persisted_prefix_labels)
            if preflight_snapshot.disposition is AppendDisposition.SKIP:
                preflight.complete_noop()
                self._release_construction_custody(
                    preflight=preflight,
                )
                self._current = {
                    "session": None, "sink": None, "accounting": None,
                    "dormant": True, "target": key, "lineage": lineage, "item": item, "display_owner": display_owner, "stop_requested": False,
                    "write_labels": (),
                    "persisted_prefix_labels": persisted_prefix_labels,
                    "policy": policy,
                    "background_bindings": decision.background_bindings,
                }
                return None, False

        xye = None
        nexus = None
        try:
            if cancelled():
                raise RuntimeError("admission cancelled")
            write_labels = labels if preflight is None else tuple(
                preflight.snapshot.write_labels
            )
            if coordinated:
                choice, _prepared_heavy_request = heavy_residency_choice(
                    self.configuration.run_options
                )
                allocation = policy.allocation
                if unsafe_unfunded_staging:
                    retained, unfunded_bytes, projected_bytes = (
                        _unsafe_unfunded_staging_projection(
                            allocation,
                            frame_count=frame_count,
                            checkpoint=(
                                pipeline_v2.semantic_checkpoint_frame_cap
                            ),
                        )
                    )
                    run_provenance[
                        "unsafe_unfunded_staging_diagnostic"
                    ] = {
                        "mode": "UNSAFE_UNFUNDED",
                        "requested_staging_frame_cap": (
                            pipeline_v2.staging_frame_cap
                        ),
                        "funded_staging_frame_cap": (
                            allocation.staging_items
                        ),
                        "semantic_checkpoint_frame_cap": (
                            pipeline_v2.semantic_checkpoint_frame_cap
                        ),
                        "modeled_retained_rows": retained,
                        "modeled_unfunded_bytes": unfunded_bytes,
                        "modeled_session_bytes": projected_bytes,
                        "host_physical_bytes": total_physical_ram_bytes(),
                    }
                requested_heavy = (
                    heavy_request if heavy_request is not None else heavy_window(
                        8 * allocation.requirements.pixels, env=resource_env,
                    )
                )
                effective_display = display_state.bind_heavy_allocation(
                    allocation
                )
                fact = HeavyResidencyFact(
                    str(target), choice,
                    _heavy_resolution_source(heavy_request, resource_env),
                    requested_heavy, allocation.staging_items,
                    allocation.record_heavy_items,
                    allocation.publication_heavy_items, effective_display,
                )
                self.resource_facts += (fact,)
                if resource_fact_sink is not None: resource_fact_sink(fact)
                logger.info("%s", fact.log_line("[RUN-RESOURCES]"))
                bind_source = getattr(source_owner, "bind_allocation", None)
                if callable(bind_source):
                    bind_source(allocation)
                if _direct_eiger_candidate(item, write_labels):
                    bind_direct = getattr(source_owner, "bind_eiger_direct_chunk", None)
                    if callable(bind_direct):
                        bind_direct(allocation)
                if first_write_frame.image is None:
                    first_write_frame.load_image()
                if layout is not None:
                    lease = acquire_light_1d_retention(
                        SessionResourceAuthority.from_allocation(allocation),
                        owner=f"scattering-light-1d:{key}",
                        generation=int(self.configuration.generation),
                        layout=layout,
                        requested_rows=requested_rows,
                        compatibility_byte_ceiling=ceiling,
                        gui_thread_id=int(gui_thread_id),
                        funding_mode=Light1DFundingMode.REPLACE_PUBLICATION_A1,
                        current_lineage_rows=(None if self.configuration.live_mode
                                              else frame_count),
                    )
                    display_state.stage_light_1d(display_owner, lease)
            # GI scouts must use the original scan extent on Append as well.
            # The submission ledger independently admits only write_labels.
            session_scan = scan if not coordinated or self.configuration.gi.enabled else replace(
                scan, frames=[frame for frame in scan.frames if int(frame.index) in write_labels])
            if self.configuration.gi.enabled and session_scan.frames[0].image is None:
                loaded_scout_frame = session_scan.frames[0]
                loaded_scout_frame.load_image()
            integration_1d = getattr(plan, "integration_1d", None)
            inflight_max = (
                max(1, self.configuration.max_cores)
                if policy is None else policy.allocation.reduction_inflight
            )
            if integration_1d is not None and output_diagnostics.save_xye:
                xye = TransactionalXYESink(
                    target.parent / str(scan.name),
                    prefix=xye_prefix_for_unit(integration_1d.unit),
                )
                self._pending_xye.append(xye)

            if not xye_only:
                sink_values = dict(
                    overwrite=self.configuration.output_mode == "Overwrite",
                    flush_every=None,
                    atomic=False,
                    incremental_finalization=True,
                    source_base=self.configuration.project_root or None,
                    # Persist the ROOT family so a later Reintegrate/Average
                    # consumes it instead of deriving `<scan>_int2d` from this
                    # artifact's own stem and publishing a chained name.
                    artifact_family=item.artifact_family or None,
                    run_configuration_provenance=run_provenance,
                    source_execution_provenance=source_execution_projection(
                        item.graph
                    ),
                    source_snapshots_provenance=(
                        _writer_source_snapshots(item)
                    ),
                    durable_fsync=output_diagnostics.durable_fsync,
                )
                if self.configuration.output_mode == "Append":
                    nexus = NexusSink(
                        target, append_preflight=preflight,
                        file_lock=self._command_lock, **sink_values,
                    )
                else:
                    nexus = NexusSink(
                        target, same_run_intent=intent,
                        file_lock=self._command_lock, **sink_values,
                    )
                if pipeline_v2 is not None:
                    nexus._configure_writer_batch_size(
                        pipeline_v2.writer_settlement_batch_size,
                    )
                    nexus._configure_nexus_record_batch_size(
                        pipeline_v2.nexus_record_batch_size,
                    )
                else:
                    nexus._configure_writer_batch_size(
                        1 if self.configuration.live_mode
                        else min(8, inflight_max)
                    )
                self._pending_nexus.append(nexus)
            if nexus is None:
                sink = xye
            elif xye is None:
                sink = nexus
            else:
                sink = CompositeSink(
                    (xye, nexus) if (
                        pipeline_v2 is not None
                        and pipeline_v2.buffers_nexus_records_across_calls
                    ) else (nexus, xye)
                )
            if sink is None:
                raise ValueError("dynamic output graph has no applicable sink")

            targets_by_mode = {}
            for mode in modes:
                targets = []
                if nexus is not None:
                    targets.append(f"nexus:{nexus.path}")
                if xye is not None and mode.kind == "1d":
                    targets.append(xye.canonical_target)
                targets_by_mode[mode] = tuple(targets)
            ledger = StageLedger(
                required_modes=modes,
                targets_by_mode=targets_by_mode,
            )
            accounting = DynamicRunAccounting(
                ledger,
                run_generation=int(self.configuration.generation),
                limits=DynamicAccountingLimits(
                    1, 32, _SUPPORTED_LINEAGE_FRAME_CEILING,
                ),
            )
            xye_lineage = (
                begin_same_run_lineage(intent) if xye_only else None
            )
            if (
                xye_lineage is not None
                and xye_lineage.disposition is not AppendDisposition.WRITE
            ):
                raise AppendRefused(xye_lineage)
            session = open_headless_scan_session(
                session_scan,
                plan,
                sink=sink,
                executor=(self.configuration.max_cores if policy is None
                          else policy.allocation.workers),
                inflight_max=inflight_max,
                gi_freeze_mode=(
                    "scout_union" if self.configuration.gi.enabled else None
                ),
                record_store=record_store,
                record_store_persisted_on_write=False,
                nexus_target=None if nexus is None else f"nexus:{nexus.path}",
                xye_target=None if xye is None else xye.canonical_target,
                accounting=accounting,
                xye_receipt_boundary=xye,
                policy=policy,
                background_plan=self.configuration.background,
                dynamic_nexus_checkpoint=bool(
                    nexus is not None
                    and policy is not None
                ),
                dynamic_nexus_checkpoint_threshold=(
                    pipeline_v2.semantic_checkpoint_frame_cap
                    if pipeline_v2 is not None else None
                ),
            )
            if loaded_scout_frame is not None:
                loaded_scout_frame.image = None
                loaded_scout_frame = None
            if coordinated:
                session.set_generation(int(self.configuration.generation))
            if lease is not None:
                publication_store.bind_allocation(policy.allocation)
                publication_store.bind_light_1d(lease)
                hooks = publication_store.light_1d_cleanup_hooks(
                    lease, cancel=light_cancel, drain=light_drain,
                    verify=light_verify,
                )
                display_state.stage_light_1d(display_owner, lease, hooks=hooks)
                display_state.bind_light_1d(display_owner, lease)
                slot = Light1DCustodySlot(
                    grant_id=lease.grant_id, owner=lease.owner,
                    generation=lease.generation, cleanup_hooks=hooks,
                )
                display_state.stage_light_1d(
                    display_owner, lease, hooks=hooks, slot=slot)
                display_state.bind_light_custody(display_owner, hooks, slot)
                accounting.bind_light_1d(
                    lease, cleanup_hooks=hooks, custody_slot=slot,
                )
            if nexus is not None and nexus.append_preflight is not None:
                write_labels = tuple(
                    nexus.append_preflight.snapshot.write_labels
                )
            else:
                write_labels = labels
            graph = {
                "target": key,
                "lineage": lineage,
                "scan": scan,
                "item": item,
                "intent": intent,
                "session": session,
                "sink": sink,
                "nexus": nexus,
                "xye": xye,
                "accounting": accounting,
                "record_store": record_store,
                "display_owner": display_owner,
                "revisions": {
                    item.source_stamp.execution_identity_v1: revision
                },
                "keys": {},
                "source_revisions": {},
                "armed_labels": frozenset(),
                "write_labels": write_labels,
                "persisted_prefix_labels": persisted_prefix_labels,
                "settled_labels": set(),
                "xye_lineage": xye_lineage,
                "xye_pending": None,
                "transition": None,
                "stop_requested": False,
                "projection_pending": False,
                "policy": policy,
                "background_bindings": decision.background_bindings,
            }
            self._graphs[key] = graph
            self._current = graph
            self._release_construction_custody(
                preflight=preflight, nexus=nexus, xye=xye,
            )
            if coordinated:
                display_state.bind_light_subscription(display_owner, session.on_frame_completed, on_frame_completed)
                session.on_checkpoint_recoverable(on_checkpoint_recoverable)
            self._arm(graph, write_labels, revision)
            if cancelled():
                raise RuntimeError("admission cancelled")
            if pipeline_v2 is not None:
                logger.info(
                    "[RUN-PIPELINE-V2] requested-settlement=%d "
                    "effective-settlement=%d requested-record=%d "
                    "effective-record=%d requested-inflight=%d "
                    "effective-inflight=%d requested-checkpoint=%d "
                    "effective-checkpoint=%d requested-workers=%d "
                    "effective-workers=%d requested-staging=%d "
                    "effective-staging=%d staging-mode=%s",
                    pipeline_v2.writer_settlement_batch_size,
                    nexus.writer_batch_size,
                    pipeline_v2.nexus_record_batch_size,
                    nexus.nexus_record_batch_size,
                    pipeline_v2.reduction_inflight,
                    policy.allocation.reduction_inflight,
                    pipeline_v2.semantic_checkpoint_frame_cap,
                    session._dynamic_nexus_checkpoint_threshold,
                    self.configuration.max_cores, policy.allocation.workers,
                    pipeline_v2.staging_frame_cap,
                    policy.allocation.staging_items,
                    (
                        "UNSAFE_UNFUNDED"
                        if unsafe_unfunded_staging else "FUNDED"
                    ),
                )
                if unsafe_unfunded_staging:
                    diagnostic = run_provenance[
                        "unsafe_unfunded_staging_diagnostic"
                    ]
                    logger.warning(
                        "[RUN-STAGING] mode=UNSAFE_UNFUNDED "
                        "requested-staging=%d funded-staging=%d "
                        "effective-flush-cap=%d requested-checkpoint=%d "
                        "effective-checkpoint=%d retained=%d "
                        "unfunded=%dB modeled-session=%dB host=%dB",
                        pipeline_v2.staging_frame_cap,
                        policy.allocation.staging_items,
                        policy.flush.cap,
                        pipeline_v2.semantic_checkpoint_frame_cap,
                        session._dynamic_nexus_checkpoint_threshold,
                        diagnostic["modeled_retained_rows"],
                        diagnostic["modeled_unfunded_bytes"],
                        diagnostic["modeled_session_bytes"],
                        diagnostic["host_physical_bytes"],
                    )
            if output_diagnostics.explicit:
                logger.info(
                    "[RUN-OUTPUT-DIAGNOSTICS] save-xye=%s "
                    "durable-fsync=%s durability=%s",
                    "on" if output_diagnostics.save_xye else "off",
                    "on" if output_diagnostics.durable_fsync else "off",
                    (
                        "DURABLE"
                        if output_diagnostics.durable_fsync
                        else "UNSAFE_SIMULATED"
                    ),
                )
            return session, True
        except BaseException as primary:
            if loaded_scout_frame is not None:
                loaded_scout_frame.image = None
            if science_identity_was_unset and not self._graphs:
                self._science_identity = None
            cleanup: BaseException | None = None
            try:
                terminal = DynamicRunState.ABORTED
                if session is not None:
                    try:
                        if graph is not None and self._graphs.get(key) is graph:
                            self._finish_transition(graph, stopped=True)
                        else:
                            if session.is_running:
                                session.stop()
                            session.finish(raise_on_failure=False)
                    except BaseException as error:
                        cleanup = error
                    try:
                        state = accounting.snapshot().state
                    except BaseException as error:
                        cleanup = cleanup or error
                    else:
                        if state in {
                            DynamicRunState.FINISHED,
                            DynamicRunState.STOPPED,
                            DynamicRunState.ABORTED,
                        }:
                            terminal = state
                if coordinated:
                    display_state.release_light_1d(
                        display_owner, reason="construction",
                        terminal=terminal,
                    )
            except BaseException as error:
                cleanup = cleanup or error
            if cleanup is not None:
                raise primary from cleanup
            raise

    def _arm(
        self,
        graph: dict[str, Any],
        labels: tuple[int, ...],
        source_revision: int,
    ) -> None:
        accounting = graph["accounting"]
        frames = {
            int(frame.index): frame for frame in graph["scan"].frames
        }
        armed_labels = []
        for label in labels:
            if label not in frames:
                raise ValueError(f"dynamic output label {label} has no frame")
            key = graph["keys"].get(label)
            if key is None:
                key = DynamicFrameIdentity(
                    graph["lineage"],
                    (graph["target"], int(label)),
                )
                accounting.discover(
                    key,
                    group=graph["lineage"],
                    ordinal=len(graph["keys"]),
                    output_label=int(label),
                )
                graph["keys"][label] = key
            label = int(label)
            graph["source_revisions"][label] = int(source_revision)
            armed_labels.append(label)
        graph["armed_labels"] = frozenset(armed_labels)

    def submit(self, frame, image: Any = _MISSING) -> bool:
        with self._command_lock:
            graph = self._current
            if graph is None:
                raise RuntimeError("dynamic output graph is not active")
            if self._stop_requested or graph["stop_requested"]:
                return False
            label = int(frame.index)
            if label not in graph["armed_labels"]:
                return True
            token = graph["accounting"].begin_attempt(
                graph["keys"][label],
                source_revision=graph["source_revisions"][label],
            )
            graph["accounting"].record_enqueued(token)
            session = graph["session"]
        if image is _MISSING:
            return bool(session.submit(frame, attempt_token=token))
        return bool(session.submit(frame, image, attempt_token=token))

    def _epoch_transition(self, graph: dict[str, Any]):
        with self._command_lock:
            if graph["transition"] not in {None, "epoch"}:
                raise RuntimeError("dynamic output has a pending finish")
            if graph["stop_requested"] and graph["transition"] is None:
                raise RuntimeError("admission cancelled")
            graph["transition"] = "epoch"
        value = graph["session"].commit_epoch()
        pending = graph.get("xye_pending") or graph.get("xye_lineage")
        if graph["nexus"] is None and pending is not None:
            graph["xye_lineage"] = seal_append_epoch(pending)
            graph["xye_pending"] = None
        with self._command_lock:
            graph["transition"] = None
            graph["projection_pending"] = True
            stop = graph["stop_requested"]
        if stop and graph["session"].is_running:
            graph["session"].stop()
        return value, stop

    def commit_epoch(self):
        graph = self._current
        if graph is None:
            raise RuntimeError("dynamic output graph is not active")
        return self._epoch_transition(graph)[0]

    def _finish_transition(self, graph: dict[str, Any], *, stopped=False,
                           stop_already_applied=False):
        with self._command_lock:
            if graph["transition"] not in {None, "finish"}:
                raise RuntimeError("dynamic output has a pending epoch")
            retrying = graph["transition"] == "finish"
            graph["transition"] = "finish"
            graph["stop_requested"] = graph["stop_requested"] or stopped
            stop = graph["stop_requested"]
        if (
            stop and not retrying and not stop_already_applied
            and graph["session"].is_running
        ):
            graph["session"].stop()
        result = graph["session"].finish(raise_on_failure=False)
        with self._command_lock:
            graph["transition"] = None
            graph["projection_pending"] = True
        return result

    def finish_current(self, *, stopped: bool = False):
        graph = self._current
        return None if graph is None else self._finish_transition(graph, stopped=stopped)

    def finish_all(self, *, stopped: bool = False) -> tuple[object, ...]:
        results = []
        primary = None
        try:
            self._settle_construction_custody()
        except BaseException as error:
            primary = error
        for graph in tuple(self._graphs.values()):
            try:
                with self._command_lock:
                    graph["stop_requested"] = (
                        graph["stop_requested"] or stopped
                    )
                    transition = graph["transition"]
                if transition == "epoch":
                    try:
                        _, stop_applied = self._epoch_transition(graph)
                    except BaseException as retry:
                        try:
                            result = graph["session"].finish(
                                raise_on_failure=False,
                            )
                        except BaseException as fallback:
                            raise fallback from retry
                        with self._command_lock:
                            graph["transition"] = None
                            graph["projection_pending"] = True
                        results.append(result)
                        continue
                results.append(self._finish_transition(
                    graph, stop_already_applied=(
                        transition == "epoch" and stop_applied
                    ),
                ))
            except BaseException as error:
                if primary is None:
                    primary = error
        if primary is not None:
            raise primary
        return tuple(results)

    def stop(self) -> None:
        with self._command_lock:
            self._stop_requested = True
            for graph in tuple(self._graphs.values()):
                graph["stop_requested"] = True

    def abort(self, _result=None) -> None:
        self.stop()
        self.finish_all(stopped=True)

    def project_new_durable(self, apply) -> None:
        for graph in tuple(self._graphs.values()):
            current = tuple(self._durable_labels_for(graph))
            with self._command_lock:
                prior = graph["settled_labels"]
                delta = tuple(sorted(set(current) - prior))
                pending = graph["projection_pending"]
            if not pending and not delta:
                continue
            apply(str(graph["item"].target), current, delta)
            with self._command_lock:
                graph["projection_pending"] = False
                prior.update(delta)

    @staticmethod
    def _durable_labels_for(graph: dict[str, Any]) -> tuple[int, ...]:
        accounting = graph["accounting"]
        snapshot = accounting.snapshot()
        required = accounting.ledger.targets_by_mode
        return tuple(sorted(
            int(key.logical_frame_identity[1])
            for key in snapshot.discovered
            if all(
                (key, mode, target) in snapshot.durable
                for mode, targets in required.items()
                for target in targets
            )
        ))

__all__ = ["DynamicOutputAdapter"]
