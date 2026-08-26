"""One finite worker slot for later externally owned operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import json
import math
import os, stat; from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable

from ..events import CleanupStatus, detached_exception_strings
from ..experiment_authoring import (
    CalibrationRequest, prepare_calibration_request, run_calibration,
    MaskRequest, resolve_mask_executable, run_mask,
)
from ..operation_values import (
    OperationCleanupReceipt, OperationContextStamp, OperationIdentity,
    OperationProgress, OperationTerminal, OperationTerminalStatus,
    OperationUpdate,
)
from ..presentation_background import PresentationBackgroundOwner
from xrd_tools.reduction.background import DisplayBackgroundPlan
from xrd_tools.io.output_transaction import TargetSnapshot
from xrd_tools.reduction import ReintegratePlan, ReintegrateProgress, ReintegrateResult, run_reintegrate
from xrd_tools.reduction.reintegrate import ReintegrateCancelled

def _average_scan_recipe(*args, **kwargs): from xrd_tools.reduction.average import AverageScanRecipe as owner; return owner(*args, **kwargs)
def _run_average_scan(*args, **kwargs): from xrd_tools.reduction.average import run_average_scan as owner; return owner(*args, **kwargs)
AverageScanRecipe = _average_scan_recipe; run_average_scan = _run_average_scan
def detector_calibration_to_integrator(*args, **kwargs): from xrd_tools.integrate.calibration import detector_calibration_to_integrator as owner; return owner(*args, **kwargs)
class _BackgroundPreterminalAbort(RuntimeError): pass
@dataclass(frozen=True, slots=True)
class _ReintegrateRequest:
    target: str
    entry: str
    expected_target_snapshot: TargetSnapshot
    expected_labels: tuple[int, ...]
    dimension: str
    preparation_json: str
@dataclass(frozen=True, slots=True)
class _AverageRequest:
    source_syntax: tuple[object, ...]; target: str; target_was_path: bool
    entry: str; source_base: str; source_base_was_path: bool
    output_mode: str; live_mode: bool; save_xye: bool; batch_mode: bool
    reduction_syntax: str; poni_file: str; mask_file: str
    background_syntax: str | None; numeric_metadata_keys: tuple[str, ...] | None
    invariant_metadata_keys: tuple[str, ...]; envelope_bytes: int | None
    resource_requests: tuple[tuple[str, int], ...]; resource_env: tuple[tuple[str, str], ...]
@dataclass(frozen=True, slots=True)
class _ScanPlotRequest:
    plan: object
    table: object
    roi_result: object | None
def _json_snapshot(value: object) -> str: return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
def _bounded_json_snapshot(value: object) -> str:
    count = [0]
    def detached(item, depth=0):
        count[0] += 1
        if count[0] > 4096 or depth > 8: raise ValueError
        if item is None or type(item) in {bool, int, str}: return item
        from xrd_tools.reduction.core import GI1DMode, GI2DMode
        if type(item) in {GI1DMode, GI2DMode}: return item.value
        if type(item) is float and math.isfinite(item): return item
        if is_dataclass(item) and not isinstance(item, type):
            return {field.name: detached(getattr(item, field.name), depth + 1)
                    for field in fields(item)}
        if isinstance(item, Mapping):
            if len(item) > 4096: raise ValueError
            if any(type(key) is not str or not key for key in item): raise TypeError
            return {key: detached(child, depth + 1)
                    for key, child in item.items()}
        if type(item) in {list, tuple}:
            if len(item) > 4096: raise ValueError
            return tuple(detached(child, depth + 1) for child in item)
        raise TypeError
    text = _json_snapshot(detached(value))
    if len(text.encode("utf-8")) > 65_536: raise ValueError
    return text
def _average_source_syntax(source: object) -> tuple[object, ...] | None:
    from xrd_tools.core.scan import SourceSpec
    if type(source) is not SourceSpec: return None
    options = source.options
    allowed = {
        "selected_file", "files", "pattern", "scan_name", "metadata_format",
        "meta_dir", "selection_mode", "detector", "detector_shape",
        "raw_dtype", "raw_header_skip", "admitted_motor_values",
    }
    if len(options) > 64 or any(type(key) is not str or key not in allowed for key in options): return None
    selection_mode = options.get("selection_mode")
    if selection_mode not in {None, "single_image"}: return None
    if selection_mode == "single_image":
        files, selected = options.get("files"), options.get("selected_file")
        if (type(files) is not tuple or len(files) != 1
                or type(files[0]) is not str or not files[0]
                or type(selected) is not str or selected != files[0]): return None
    names = (
        "selected_file", "pattern", "scan_name", "metadata_format",
        "meta_dir", "selection_mode", "detector", "detector_shape",
        "raw_dtype", "raw_header_skip",
    )
    count = [0]
    def detached(value, depth=0):
        count[0] += 1
        if count[0] > 4096 or depth > 8: raise ValueError
        if value is None or type(value) in {bool, int, str}: return value
        if type(value) is float and math.isfinite(value): return value
        if isinstance(value, Path): return str(value)
        if type(value) in {list, tuple}:
            return tuple(detached(item, depth + 1) for item in value)
        raise TypeError
    try:
        values = tuple(detached(options.get(name)) for name in names)
        if len(_json_snapshot(values).encode("utf-8")) > 65_536: return None
    except (TypeError, ValueError, OverflowError):
        return None
    path = lambda value: (str(value), isinstance(value, Path)) if value is not None else (None, False)
    uri, uri_path = path(source.uri); metadata_uri, metadata_path = path(source.metadata_uri)
    result = (uri, uri_path, source.kind.value, metadata_uri, metadata_path, source.entry, *values)
    try:
        if len(_json_snapshot(result).encode("utf-8")) > 65_536: return None
    except (TypeError, ValueError, OverflowError):
        return None
    return result
def _average_source_from_syntax(value: tuple[object, ...]):
    from xrd_tools.core.scan import SourceKind, SourceSpec
    if type(value) is not tuple or len(value) != 16: raise ValueError("AVERAGE_SOURCE_UNREPRESENTABLE")
    uri, uri_path, kind, metadata_uri, metadata_path, entry, *parts = value
    names = (
        "selected_file", "pattern", "scan_name", "metadata_format",
        "meta_dir", "selection_mode", "detector", "detector_shape",
        "raw_dtype", "raw_header_skip",
    )
    options = {name: item for name, item in zip(names, parts, strict=True) if item is not None}
    return SourceSpec(Path(uri) if uri_path else uri, SourceKind(kind),
        Path(metadata_uri) if metadata_path else metadata_uri, entry, options)
def _average_reduction_from_syntax(text: str):
    from xrd_tools.reduction import GIMode, Integration1DPlan, Integration2DPlan, ReductionPlan
    value = json.loads(text)
    if type(value) is not dict: raise ValueError("AVERAGE_REDUCTION_UNREPRESENTABLE")
    def integration(owner, row):
        if row is None: return None
        if type(row) is not dict: raise ValueError("AVERAGE_REDUCTION_UNREPRESENTABLE")
        for name in ("radial_range", "azimuth_range"):
            if row.get(name) is not None: row[name] = tuple(row[name])
        return owner(**row)
    gi = value.get("gi")
    return ReductionPlan(
        integration_1d=integration(Integration1DPlan, value.get("integration_1d")),
        integration_2d=integration(Integration2DPlan, value.get("integration_2d")),
        gi=None if gi is None else GIMode(**gi), mask=None,
        threshold_min=value.get("threshold_min"),
        threshold_max=value.get("threshold_max"),
        mask_saturation=value.get("mask_saturation", False),
        extra=value.get("extra", {}),
    )
def _average_background_from_syntax(text: str | None):
    if text is None: return None
    from xrd_tools.reduction.background import FrameBackgroundPlan
    value = json.loads(text)
    if type(value) is not dict: raise ValueError("AVERAGE_BACKGROUND_UNREPRESENTABLE")
    return FrameBackgroundPlan(**value)
def _average_calibration(request: _AverageRequest):
    from .. import output_preflight
    from xrd_tools.session.experiment_state import CalibrationState, FactStatus, MaskState, PoniValues
    assets = output_preflight._load_scientific_assets(request)
    if request.poni_file and assets.poni_values is None: raise ValueError("AVERAGE_CALIBRATION_UNAVAILABLE")
    if request.mask_file and assets.mask_bytes is None: raise ValueError("AVERAGE_MASK_UNAVAILABLE")
    config = None
    if assets.poni_values is not None:
        text = assets.poni_detector_config_json
        try:
            if type(text) is not str or len(text.encode("utf-8")) > 65_536: raise ValueError
            config = json.loads(text)
            if (type(config) is not dict or _json_snapshot(config) != text
                    or type(config.get("orientation")) is not int
                    or config["orientation"] not in range(1, 5)): raise ValueError
        except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
            raise ValueError("AVERAGE_DETECTOR_CONFIG_UNREPRESENTABLE") from error
    values = None
    if assets.poni_values is not None:
        try:
            values = PoniValues(*assets.poni_values[:7])
        except (TypeError, ValueError, OverflowError) as error: raise ValueError("AVERAGE_CALIBRATION_UNREPRESENTABLE") from error
    mask_state = MaskState.absent()
    if assets.mask_bytes is not None:
        try:
            import numpy as np
            dtype = np.dtype(assets.mask_dtype)
            shape = assets.mask_shape
            if (type(shape) is not tuple or not shape
                    or any(type(item) is not int or item <= 0 for item in shape)
                    or math.prod(shape) * dtype.itemsize != len(assets.mask_bytes)): raise ValueError
            np.frombuffer(assets.mask_bytes, dtype=dtype).reshape(shape)
            mask_state = MaskState(request.mask_file, assets.mask_sha256 or "", dtype.str, shape, FactStatus.PRESENT)
        except (TypeError, ValueError, OverflowError) as error: raise ValueError("AVERAGE_MASK_UNREPRESENTABLE") from error
    state = (CalibrationState(mask=mask_state) if values is None else
        CalibrationState(values, assets.poni_values[7], config, "",
            assets.poni_sha256 or "", request.poni_file, mask_state, FactStatus.PRESENT))
    if values is None: return state
    try:
        from xrd_tools.core import PONI
        from xrd_tools.core.geometry import DetectorCalibration
        accepted = assets.detector_calibration
        reconstructed = DetectorCalibration(PONI(values.dist, values.poni1, values.poni2,
            values.rot1, values.rot2, values.rot3, values.wavelength_m, state.detector_id),
            dict(state.detector_config))
        def truth(calibration):
            detector = detector_calibration_to_integrator(calibration).detector
            reported = detector.get_config(); shape = tuple(detector.shape); maximum = tuple(detector.max_shape)
            normalized = DetectorCalibration(calibration.poni, reported).to_json()
            return (type(detector), shape, maximum, float(detector.pixel1),
                float(detector.pixel2), int(detector.orientation), normalized)
        accepted_truth = truth(accepted); reconstructed_truth = truth(reconstructed)
        expected_max_shape = tuple(
            config["max_shape"]
            if "max_shape" in config
            else accepted_truth[2]
        )
        valid = (accepted_truth == reconstructed_truth
            and accepted_truth[0].__module__.startswith("pyFAI.detectors")
            and accepted_truth[0].__name__.casefold() == state.detector_id.casefold()
            and len(accepted_truth[1]) == len(expected_max_shape) == 2
            and all(type(item) is int and item > 0
                    for item in (*accepted_truth[1], *expected_max_shape))
            and all(current <= maximum for current, maximum in
                    zip(accepted_truth[1], expected_max_shape, strict=True))
            and accepted_truth[2] == expected_max_shape
            and accepted_truth[3] > 0 and accepted_truth[4] > 0
            and all(math.isfinite(value) for value in accepted_truth[3:5])
            and accepted_truth[5] == config["orientation"])
    except BaseException as error: raise ValueError("AVERAGE_DETECTOR_CONFIG_UNREPRESENTABLE") from error
    if not valid: raise ValueError("AVERAGE_DETECTOR_CONFIG_UNREPRESENTABLE")
    return state
class OperationSlot:
    """Own at most one joinable worker and its detached latest state."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._closed = False
        self._next_serial = 1
        self._identity: OperationIdentity | None = None
        self._stamp: OperationContextStamp | None = None
        self._frozen: object | None = None
        self._worker: Thread | None = None
        self._worker_started = False
        self._worker_identity: int | None = None
        self._cancel_event: Event | None = None
        self._progress: OperationProgress | None = None
        self._progress_delivered_revision = 0
        self._terminal: OperationTerminal | None = None
        self._stale = False
        self._close_cancel_accepted = False
        self._cancel_sealed = False
        self._clean_receipt: OperationCleanupReceipt | None = None
        self._finalize_hook: Callable[[str], None] | None = None; self._finalized = False
        self._terminal_committed = False
        self._abort_fact: tuple[str, str, str, str] | None = None

    @property
    def owned(self) -> bool:
        with self._lock:
            return self._identity is not None

    @property
    def current_identity(self) -> OperationIdentity | None:
        with self._lock:
            return self._identity

    def begin_calibrate(
        self, request: object, stamp: OperationContextStamp
    ) -> OperationIdentity | None:
        if type(request) is not CalibrationRequest:
            return None
        try:
            fresh = prepare_calibration_request(request.final_path)
        except (OSError, ValueError):
            return None
        if fresh != request:
            return None
        return self._begin(request, stamp, self._run_calibrate)

    def _run_calibrate(self, request, identity, cancelled, publish):
        return run_calibration(
            request, identity, cancelled, publish, self._seal_publication
        )

    def begin_mask(
        self, request: object, stamp: OperationContextStamp
    ) -> OperationIdentity | None:
        if type(request) is not MaskRequest:
            return None
        try:
            request.__post_init__(); source = Path(request.source_path)
            state = source.lstat(); valid = (
                source.resolve(strict=True) == source
                and stat.S_ISREG(state.st_mode) and os.access(source, os.R_OK)
                and not os.path.lexists(request.final_path)
                and Path(request.executable).stem.casefold() == "pyfai-drawmask" and resolve_mask_executable(request.executable) == request.executable
            )
        except (OSError, ValueError):
            return None
        return self._begin(request, stamp, self._run_mask) if valid else None

    def _run_mask(self, request, identity, cancelled, publish):
        return run_mask(request, identity, cancelled, publish, self._seal_publication)

    def begin_background(self, plan: object, stamp: OperationContextStamp,
                         owner: object, reservation: object) -> OperationIdentity | None:
        if (type(plan) is not DisplayBackgroundPlan
                or type(owner) is not PresentationBackgroundOwner
                or type(reservation) is not int or reservation < 1):
            return None
        def body(_plan, identity, cancelled, publish):
            publish("aggregate", 0, len(plan.contributor_ids))
            try:
                receipt = owner.run_and_stage(reservation, cancelled.is_set)
            except InterruptedError:
                return OperationTerminal(identity, OperationTerminalStatus.CANCELLED)
            publish("aggregate", len(plan.contributor_ids), len(plan.contributor_ids))
            try:
                return OperationTerminal(identity, OperationTerminalStatus.RETURNED,
                                         payload=receipt)
            except BaseException as error:
                raise _BackgroundPreterminalAbort(str(error)) from error
        return self._begin(plan, stamp, body,
            finalize=lambda outcome: owner.finalize(reservation, outcome))

    def begin_reintegrate(self, *, target: str, entry: str,
                          expected_target_snapshot: TargetSnapshot,
                          expected_labels: tuple[int, ...], dimension: str,
                          preparation_values: Mapping[str, object],
                          stamp: OperationContextStamp) -> OperationIdentity | None:
        valid = (type(target) is str and bool(target) and type(entry) is str and bool(entry)
                 and type(expected_target_snapshot) is TargetSnapshot and expected_target_snapshot.exists
                 and type(expected_labels) is tuple and bool(expected_labels)
                 and expected_labels == tuple(sorted(set(expected_labels)))
                 and all(type(value) is int and value >= 0 for value in expected_labels)
                 and type(dimension) is str and dimension in {"1d", "2d"}
                 and isinstance(preparation_values, Mapping))
        if not valid: return None
        try:
            preparation_json = json.dumps(preparation_values, sort_keys=True,
                separators=(",", ":"), allow_nan=False)
            detached = json.loads(preparation_json)
            if type(detached) is not dict or detached != preparation_values: return None
        except (TypeError, ValueError, OverflowError): return None
        request = _ReintegrateRequest(target, entry, expected_target_snapshot,
            expected_labels, dimension, preparation_json)
        return self._begin(request, stamp, self._run_reintegrate_request)

    def _run_reintegrate_request(self, request, identity, cancelled, publish):
        publish("prepare", 0, 1)
        try:
            plan = ReintegratePlan.from_artifact(request.target, entry=request.entry,
                dimension=request.dimension, preparation=json.loads(request.preparation_json),
                expected_target_snapshot=request.expected_target_snapshot,
                expected_labels=request.expected_labels, cancel_token=cancelled)
        except ReintegrateCancelled:
            return OperationTerminal(identity, OperationTerminalStatus.CANCELLED)
        publish("prepare", 1, 1); headless_revision = 0
        def progress(value):
            nonlocal headless_revision
            try:
                if (type(value) is not ReintegrateProgress
                        or value.operation_identity != plan.operation_identity
                        or value.revision <= headless_revision): return
                headless_revision = value.revision
                publish(value.stage, value.completed, value.total)
            except BaseException: return
        result = run_reintegrate(plan, cancel_token=cancelled, progress_cb=progress)
        if type(result) is not ReintegrateResult:
            raise TypeError("reintegration runner returned an invalid result")
        return OperationTerminal(identity, OperationTerminalStatus.RETURNED,
                                 payload=result)

    def begin_average(self, source: object, target: object, reduction: object, *,
                      entry: str = "entry", source_base: object = None,
                      output_mode: str = "Overwrite", live_mode: bool = False,
                      save_xye: bool = False, batch_mode: bool = False,
                      poni_file: str = "", mask_file: str = "",
                      background: object | None = None,
                      numeric_metadata_keys: tuple[str, ...] | None = None,
                      invariant_metadata_keys: tuple[str, ...] = (),
                      envelope_bytes: int | None = None,
                      resource_requests: Mapping[str, int] | None = None,
                      resource_env: Mapping[str, str] | None = None,
                      stamp: OperationContextStamp) -> OperationIdentity | None:
        from xrd_tools.reduction import ReductionPlan; from xrd_tools.reduction.background import FrameBackgroundPlan
        syntax = _average_source_syntax(source)
        pathlike = lambda value: isinstance(value, (str, Path)) and bool(str(value))
        if (syntax is None or not pathlike(target)
                or not (source_base is None or pathlike(source_base))
                or type(reduction) is not ReductionPlan or reduction.mask is not None
                or type(entry) is not str or not entry
                or any(type(value) is not bool for value in (live_mode, save_xye, batch_mode))
                or type(output_mode) is not str
                or type(poni_file) is not str or type(mask_file) is not str
                or background is not None and type(background) is not FrameBackgroundPlan
                or numeric_metadata_keys is not None and type(numeric_metadata_keys) is not tuple
                or type(invariant_metadata_keys) is not tuple
                or envelope_bytes is not None and (type(envelope_bytes) is not int or envelope_bytes <= 0)):
            return None
        try:
            if (numeric_metadata_keys is not None
                    and len(numeric_metadata_keys) > 64
                    or len(invariant_metadata_keys) > 64
                    or any(type(value) is not str or not value for value in (numeric_metadata_keys or ()))
                    or any(type(value) is not str or not value for value in invariant_metadata_keys)
                    or resource_requests is not None and (not isinstance(resource_requests, Mapping) or len(resource_requests) > 64)
                    or resource_env is not None and (not isinstance(resource_env, Mapping) or len(resource_env) > 64)):
                return None
            requests = {} if resource_requests is None else dict(resource_requests)
            environment = {} if resource_env is None else dict(resource_env)
            if ("owner_block_bytes" in requests
                    or any(type(key) is not str or type(value) is not int or type(value) is bool for key, value in requests.items())
                    or any(type(key) is not str or type(value) is not str for key, value in environment.items())):
                return None
            reduction_syntax = _bounded_json_snapshot(reduction)
            background_syntax = None if background is None else _bounded_json_snapshot(background)
            _bounded_json_snapshot((
                entry, str(target), None if source_base is None else str(source_base),
                output_mode, poni_file, mask_file, numeric_metadata_keys,
                invariant_metadata_keys, tuple(sorted(requests.items())),
                tuple(sorted(environment.items())),
            ))
        except (TypeError, ValueError, OverflowError):
            return None
        request = _AverageRequest(
            syntax, str(target), isinstance(target, Path), entry,
            "" if source_base is None else str(source_base),
            isinstance(source_base, Path), output_mode, live_mode, save_xye,
            batch_mode, reduction_syntax, poni_file, mask_file,
            background_syntax, numeric_metadata_keys, invariant_metadata_keys,
            envelope_bytes, tuple(sorted(requests.items())),
            tuple(sorted(environment.items())),
        )
        return self._begin(request, stamp, self._run_average_request)

    def _run_average_request(self, request, identity, cancelled, publish):
        publish("prepare", 0, 1)
        calibration = _average_calibration(request)
        source = _average_source_from_syntax(request.source_syntax)
        reduction = _average_reduction_from_syntax(request.reduction_syntax)
        background = _average_background_from_syntax(request.background_syntax)
        recipe = AverageScanRecipe(
            source, Path(request.target) if request.target_was_path else request.target,
            reduction, entry=request.entry,
            source_base=(Path(request.source_base) if request.source_base_was_path
                         else request.source_base),
            output_mode=request.output_mode, live_mode=request.live_mode,
            save_xye=request.save_xye, batch_mode=request.batch_mode,
            calibration=calibration, background=background,
            numeric_metadata_keys=request.numeric_metadata_keys,
            invariant_metadata_keys=request.invariant_metadata_keys,
            envelope_bytes=request.envelope_bytes,
            resource_requests=dict(request.resource_requests),
            resource_env=dict(request.resource_env),
        )
        publish("prepare", 1, 1)
        from xrd_tools.reduction.average import AverageScanProgress, AverageScanResult, _committed_average_mismatch
        headless_revision = -1; headless_identity = None
        def progress(value):
            nonlocal headless_revision, headless_identity
            try:
                if (type(value) is not AverageScanProgress or headless_identity is not None
                        and value.operation_identity != headless_identity or value.revision <= headless_revision): return
                headless_identity = value.operation_identity
                headless_revision = value.revision
                publish(value.stage, value.completed, value.total)
            except BaseException: return
        result = run_average_scan(
            recipe, cancel_token=cancelled, progress_cb=progress,
            publication_gate=lambda: self._seal_publication(identity),
        )
        if type(result) is not AverageScanResult or headless_identity is not None and result.operation_identity != headless_identity:
            raise TypeError("Average runner returned an invalid result")
        if result.disposition == "COMMITTED":
            mismatch = _committed_average_mismatch(result, request.target, request.entry)
            if mismatch is not None:
                return OperationTerminal(
                    identity, OperationTerminalStatus.FAILED,
                    f"AVERAGE_COMMIT_VERIFICATION_FAILED: {mismatch}",
                    payload=result,
                )
        if result.disposition in {"COMMITTED", "REFUSED"}:
            return OperationTerminal(identity, OperationTerminalStatus.RETURNED,
                                     payload=result)
        if result.disposition == "CANCELLED":
            return OperationTerminal(identity, OperationTerminalStatus.CANCELLED,
                                     payload=result)
        if result.disposition == "ABORTED":
            return OperationTerminal(
                identity, OperationTerminalStatus.FAILED,
                f"{result.diagnostic_code}: {result.diagnostic}", payload=result,
            )
        raise RuntimeError("Average runner retained settlement unexpectedly")

    @staticmethod
    def _analysis_terminal(identity, result, expected):
        if type(result) is not expected:
            raise TypeError("analysis runner returned an invalid result")
        disposition = getattr(result.disposition, "value", None)
        if disposition in {"completed", "refused"}:
            status = OperationTerminalStatus.RETURNED
        elif disposition == "cancelled":
            status = OperationTerminalStatus.CANCELLED
        else:
            raise TypeError("analysis runner retained settlement unexpectedly")
        return OperationTerminal(identity, status, payload=result)

    def begin_metadata(self, plan: object, stamp: OperationContextStamp):
        from xrd_tools.analysis.scan_operations import MetadataTablePlan
        return (self._begin(plan, stamp, self._run_metadata)
                if type(plan) is MetadataTablePlan else None)

    def _run_metadata(self, plan, identity, cancelled, publish):
        from xrd_tools.analysis.scan_operations import (
            MetadataTableResult, run_metadata_table,
        )
        result = run_metadata_table(
            plan, cancel_token=cancelled,
            progress_callback=lambda done, total: publish("metadata", done, total))
        return self._analysis_terminal(identity, result, MetadataTableResult)

    def begin_scan_plot(self, plan: object, table: object, roi_result: object,
                        stamp: OperationContextStamp):
        from xrd_tools.analysis.scan_operations import (
            RoiScanResult, ScanPlotPlan, MetadataTableResult,
        )
        if (type(plan) is not ScanPlotPlan or type(table) is not MetadataTableResult
                or roi_result is not None and type(roi_result) is not RoiScanResult):
            return None
        return self._begin(_ScanPlotRequest(plan, table, roi_result), stamp,
                           self._run_scan_plot)

    def _run_scan_plot(self, request, identity, cancelled, publish):
        from xrd_tools.analysis.scan_operations import ScanPlotResult, run_scan_plot
        result = run_scan_plot(
            request.plan, request.table, roi_result=request.roi_result,
            cancel_token=cancelled,
            progress_callback=lambda done, total: publish("scan_plot", done, total))
        return self._analysis_terminal(identity, result, ScanPlotResult)

    def begin_roi_preview(self, plan: object, stamp: OperationContextStamp):
        from xrd_tools.analysis.scan_operations import RoiPreviewPlan
        return (self._begin(plan, stamp, self._run_roi_preview)
                if type(plan) is RoiPreviewPlan else None)

    def _run_roi_preview(self, plan, identity, cancelled, publish):
        from xrd_tools.analysis.scan_operations import RoiPreviewResult, run_roi_preview
        result = run_roi_preview(
            plan, cancel_token=cancelled,
            progress_callback=lambda done, total: publish("roi_preview", done, total))
        return self._analysis_terminal(identity, result, RoiPreviewResult)

    def begin_roi_scan(self, plan: object, stamp: OperationContextStamp):
        from xrd_tools.analysis.scan_operations import RoiScanPlan
        return (self._begin(plan, stamp, self._run_roi_scan)
                if type(plan) is RoiScanPlan else None)

    def _run_roi_scan(self, plan, identity, cancelled, publish):
        from xrd_tools.analysis.scan_operations import RoiScanResult, run_roi_scan
        result = run_roi_scan(
            plan, cancel_token=cancelled,
            progress_callback=lambda done, total: publish("roi_scan", done, total))
        return self._analysis_terminal(identity, result, RoiScanResult)

    def begin_peak_fit(self, plan: object, stamp: OperationContextStamp):
        from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitPlan
        return (self._begin(plan, stamp, self._run_peak_fit)
                if type(plan) is DisplayedPeakFitPlan else None)

    def _run_peak_fit(self, plan, identity, cancelled, publish):
        from xrd_tools.analysis.display_fit_operations import (
            DisplayedPeakFitResult, run_displayed_peak_fit,
        )
        result = run_displayed_peak_fit(
            plan, cancel_token=cancelled,
            progress_callback=lambda done, total: publish("peak_fit", done, total))
        return self._analysis_terminal(identity, result, DisplayedPeakFitResult)

    def begin_phase_fit(self, plan: object, stamp: OperationContextStamp):
        from xrd_tools.analysis.display_fit_operations import DisplayedPhaseFitPlan
        return (self._begin(plan, stamp, self._run_phase_fit)
                if type(plan) is DisplayedPhaseFitPlan else None)

    def _run_phase_fit(self, plan, identity, cancelled, publish):
        from xrd_tools.analysis.display_fit_operations import (
            DisplayedPhaseFitResult, run_displayed_phase_fit,
        )
        result = run_displayed_phase_fit(
            plan, cancel_token=cancelled,
            progress_callback=lambda done, total: publish("phase_fit", done, total))
        return self._analysis_terminal(identity, result, DisplayedPhaseFitResult)

    def _seal_publication(self, identity: OperationIdentity) -> bool:
        with self._lock:
            event = self._cancel_event
            if (
                self._identity is not identity or self._terminal is not None
                or self._cancel_sealed or event is None or event.is_set()
            ):
                return False
            self._cancel_sealed = True
            return True

    def _begin(self, frozen: object, stamp: OperationContextStamp,
               body: Callable[..., object], finalize: Callable[[str], None] | None = None
               ) -> OperationIdentity | None:
        if (
            not self._is_frozen_dataclass(frozen)
            or type(stamp) is not OperationContextStamp
            or not callable(body) or finalize is not None and not callable(finalize)
        ):
            return None
        try:
            stamp.__post_init__()
        except BaseException:
            return None

        with self._lock:
            if self._closed or self._identity is not None:
                return None
            identity = OperationIdentity(self._next_serial)
            self._next_serial += 1
            cancel_event = Event()
            worker = Thread(target=self._run,
                args=(identity, frozen, cancel_event, body),
                name=f"scattering-operation-{identity.serial}",
                daemon=False)
            self._identity = identity
            self._stamp = stamp
            self._frozen = frozen
            self._worker = worker
            self._worker_started = False
            self._worker_identity = id(worker)
            self._cancel_event = cancel_event
            self._progress = None
            self._progress_delivered_revision = 0
            self._terminal = None
            self._stale = False
            self._close_cancel_accepted = False
            self._cancel_sealed = False
            self._clean_receipt = None
            self._finalize_hook = finalize; self._finalized = False
            self._terminal_committed = False; self._abort_fact = None
            try:
                worker.start()
            except BaseException as error:
                self._terminal = self._failed(identity, error)
            else:
                self._worker_started = True
            hook = self._take_finalize_locked("START_FAILED") if not self._worker_started else None
        self._invoke_finalize(hook, "START_FAILED")
        return identity

    def observe_stamp(self, stamp: object) -> None:
        valid = type(stamp) is OperationContextStamp
        if valid:
            try:
                stamp.__post_init__()
            except BaseException:
                valid = False
        with self._lock:
            if self._identity is None or self._stale:
                return
            try:
                mismatch = not valid or stamp != self._stamp
            except BaseException:
                mismatch = True
            if mismatch:
                self._stale = True

    def poll(self, identity: object) -> OperationUpdate | None:
        with self._lock:
            if self._identity is not identity:
                return None
            worker = self._worker
            started = self._worker_started
        if worker is None:
            return None
        joined, alive = self._join_state(worker, started)
        if not joined:
            return None

        with self._lock:
            if self._identity is not identity or self._worker is not worker:
                return None
            progress = self._undelivered_progress_locked()
            if alive:
                if progress is None:
                    return None
                return OperationUpdate(identity, progress=progress, stale=self._stale)
            terminal = self._terminal
            if terminal is None:
                if self._abort_fact is None:
                    return None
                hook = self._take_finalize_locked("ABORTED_WITHOUT_TERMINAL")
                self._retire_locked()
                update = None
            else:
                update = OperationUpdate(
                    identity,
                    progress=progress,
                    terminal=terminal,
                    stale=self._stale,
                )
                outcome = ("STALE" if self._stale else
                           "TRANSFERRED" if terminal.status is OperationTerminalStatus.RETURNED
                           else terminal.status.value.upper())
                hook = self._take_finalize_locked(outcome)
                self._retire_locked()
        self._invoke_finalize(hook, "ABORTED_WITHOUT_TERMINAL" if update is None else outcome)
        return update

    def cancel(self, identity: object) -> bool:
        with self._lock:
            event = self._cancel_event
            if (
                self._identity is not identity
                or self._terminal is not None
                or self._cancel_sealed
                or event is None
                or event.is_set()
            ):
                return False
            event.set()
            return True

    def close(self) -> OperationCleanupReceipt:
        with self._lock:
            if self._clean_receipt is not None:
                return self._clean_receipt
            self._closed = True
            identity = self._identity
            if identity is None:
                receipt = OperationCleanupReceipt(None, CleanupStatus.CLEANED)
                self._clean_receipt = receipt
                return receipt
            event = self._cancel_event
            if (
                self._terminal is None
                and not self._cancel_sealed
                and event is not None
                and not event.is_set()
            ):
                event.set()
                self._close_cancel_accepted = True
            worker = self._worker
            started = self._worker_started
            worker_identity = self._worker_identity
        if worker is None or worker_identity is None:
            raise RuntimeError("operation slot lost its worker")
        joined, alive = self._join_state(worker, started)

        with self._lock:
            if self._identity is not identity or self._worker is not worker:
                receipt = OperationCleanupReceipt(None, CleanupStatus.CLEANED)
                self._clean_receipt = receipt
                return receipt
            if not joined or alive:
                return OperationCleanupReceipt(identity,
                    CleanupStatus.CLEANUP_PENDING,
                    self._close_cancel_accepted, worker_identity,
                    stale=self._stale)
            terminal = self._terminal
            if terminal is None:
                if self._abort_fact is not None:
                    hook = self._take_finalize_locked("ABORTED_WITHOUT_TERMINAL")
                    self._retire_locked()
                    receipt = OperationCleanupReceipt(None, CleanupStatus.CLEANED)
                    self._clean_receipt = receipt
                else:
                    return OperationCleanupReceipt(identity,
                        CleanupStatus.CLEANUP_PENDING,
                        self._close_cancel_accepted, worker_identity,
                        stale=self._stale)
            else:
                receipt = OperationCleanupReceipt(identity, CleanupStatus.CLEANED,
                    self._close_cancel_accepted, worker_identity, terminal,
                    self._stale)
                outcome = ("STALE" if self._stale else
                           "TRANSFERRED" if terminal.status is OperationTerminalStatus.RETURNED
                           else terminal.status.value.upper())
                hook = self._take_finalize_locked(outcome)
                self._retire_locked()
                self._clean_receipt = receipt
        self._invoke_finalize(hook, "ABORTED_WITHOUT_TERMINAL" if terminal is None else outcome)
        return receipt

    def _run(self, identity: OperationIdentity, frozen: object,
             cancel_event: Event, body: Callable[..., object]) -> None:
        def publish(stage: object, completed: object, total: object) -> None:
            self._publish(identity, stage, completed, total)

        terminal = None
        try:
            candidate = body(frozen, identity, cancel_event, publish)
        except BaseException as error:
            if type(error) is _BackgroundPreterminalAbort:
                self._record_abort_without_terminal(identity, error); return
            try:
                terminal = self._failed(identity, error)
            except BaseException as terminal_error:
                self._record_abort_without_terminal(identity, terminal_error)
                return
        else:
            try:
                if (
                    type(candidate) is not OperationTerminal
                    or candidate.identity is not identity
                ):
                    raise ValueError("operation body returned an invalid terminal")
                candidate.__post_init__()
                terminal = candidate
            except BaseException as error:
                self._record_abort_without_terminal(identity, error)
                return
        try:
            with self._lock:
                if self._identity is identity and self._terminal is None:
                    self._terminal = terminal
                    self._terminal_committed = True
        except BaseException as error:
            self._record_abort_without_terminal(identity, error)

    def _record_abort_without_terminal(self, identity: OperationIdentity,
                                       error: BaseException) -> None:
        module, name, message = detached_exception_strings(error)
        try:
            with self._lock:
                if self._identity is identity and not self._terminal_committed:
                    self._abort_fact = (
                        "ABORTED_WITHOUT_TERMINAL", module, name, message)
                    self._terminal = None
        except BaseException:
            pass

    def _publish(self, identity: OperationIdentity, stage: object,
                 completed: object, total: object) -> None:
        try:
            with self._lock:
                if self._identity is not identity or self._terminal is not None:
                    return
                prior = self._progress
                revision = 1 if prior is None else prior.revision + 1
                candidate = OperationProgress(
                    identity, stage, completed, total, revision
                )
                if (
                    prior is not None
                    and candidate.stage == prior.stage
                    and candidate.completed < prior.completed
                ):
                    return
                self._progress = candidate
        except BaseException:
            return

    def _undelivered_progress_locked(self) -> OperationProgress | None:
        progress = self._progress
        if (
            progress is None
            or progress.revision <= self._progress_delivered_revision
        ):
            return None
        self._progress_delivered_revision = progress.revision
        return progress

    def _retire_locked(self) -> None:
        self._identity = None
        self._stamp = None
        self._frozen = None
        self._worker = None
        self._worker_started = False
        self._worker_identity = None
        self._cancel_event = None
        self._progress = None
        self._progress_delivered_revision = 0
        self._terminal = None
        self._stale = False
        self._close_cancel_accepted = False
        self._cancel_sealed = False
        self._finalize_hook = None; self._finalized = False
        self._terminal_committed = False; self._abort_fact = None

    def _take_finalize_locked(self, outcome: str):
        if self._finalized or self._finalize_hook is None:
            return None
        self._finalized = True; return self._finalize_hook

    @staticmethod
    def _invoke_finalize(hook, outcome: str) -> None:
        if hook is None:
            return
        try:
            hook(outcome)
        except BaseException:
            return

    @staticmethod
    def _join_state(worker: Thread, started: bool) -> tuple[bool, bool]:
        if not started:
            return True, False
        try:
            worker.join(timeout=0.0)
            return True, worker.is_alive()
        except RuntimeError:
            return False, True

    @staticmethod
    def _failed(identity: OperationIdentity,
                error: BaseException) -> OperationTerminal:
        module, name, message = detached_exception_strings(error)
        return OperationTerminal(identity, OperationTerminalStatus.FAILED,
                                 f"{module}.{name}: {message}")

    @staticmethod
    def _is_frozen_dataclass(value: object) -> bool:
        try:
            parameters = vars(type(value)).get("__dataclass_params__")
            return (
                is_dataclass(value)
                and not isinstance(value, type)
                and parameters is not None
                and parameters.frozen is True
            )
        except BaseException:
            return False
__all__ = ["OperationSlot"]
