from __future__ import annotations
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields
import hashlib, json, math, os
from pathlib import Path
import re, threading, time
import tempfile
from types import SimpleNamespace
from typing import Any
import numpy as np
from xrd_tools.core.invalid import detector_value_mask
from xrd_tools.core.provenance import read_provenance
from xrd_tools.core.scan import Scan, ScanFrame, SourceKind, SourceSpec
from xrd_tools.core.strictness import GIAllDummyError, MissingNormalizationError, StrictPolicy
from xrd_tools.io.append import AppendIntent, science_fingerprint
from xrd_tools.io.image import load_mask, read_detector_image_layout
from xrd_tools.io.nexus_record import _average_count_chunks, _average_count_digest
from xrd_tools.io.output_transaction import StreamTerminal, TargetSnapshot, capture_target_snapshot
from xrd_tools.io.read import _read_average_lineage, get_average_finite_counts, resolve_source_master
from xrd_tools.io.record_writer import WriterIncomplete
from xrd_tools.reduction.background import FrameBackgroundPlan, resolve_frame_background
from xrd_tools.reduction.core import GIMode, Integration1DPlan, Integration2DPlan, NexusSink, ReductionPlan, run_reduction
from xrd_tools.session.experiment_state import CalibrationState, FactStatus, MaskState, PoniValues
from xrd_tools.session import policy as _resource_policy
from xrd_tools.session.policy import SessionResourceAllocation, requirements_from, resolve_session_policy
from xrd_tools.session.run_configuration import FrozenSourceSpec
from xrd_tools.session.scan_session import required_result_modes
from xrd_tools.sources import execution_graph as _source_graph_owner
from xrd_tools.sources.execution_graph import PreparedSourceExecutionGraph, SourceRevisionChanged, append_source_from_execution_graph, open_source_execution_graph, qualify_source_execution_graph, requalify_source_execution_graph, source_execution_projection, source_graph_digest, source_snapshots_projection, stable_lineage_projection
from xrd_tools.sources.selection import image_series_spec, is_single_image_spec, single_image_spec
_COUNT_POLICY = 'average_scan_v1'
_COUNT_DTYPE = np.dtype('<u4')
_COUNT_DIGEST = re.compile('[0-9a-f]{64}\\Z')
_MAX_CONTRIBUTORS = 1000000
_READER_BINDING = 'average_closed_v1'
_JSON_MAX_BYTES, _JSON_MAX_DEPTH, _JSON_MAX_NODES = 1 << 16, 8, 4096
_METADATA_MAX_KEYS = 64
_STATIC_MASK_COPY_CHUNK_BYTES = 1 << 16
_STATIC_MASK_COMPATIBILITY_FLOOR_BYTES = 64 << 20
_STATIC_MASK_HEADER_ALLOWANCE_BYTES = 1 << 20
_STAGES = frozenset({'qualify', 'read', 'average', 'reduce', 'write', 'settle'})
def _reject(condition: bool, message: str, error=ValueError) -> None:
    if condition:
        raise error(message)
def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
def _freeze_json(value: Any, count: list[int], active: set[int], depth: int=0):
    count[0] += 1
    _reject(count[0] > _JSON_MAX_NODES, 'Average JSON extras exceed the occurrence limit')
    if value is None or type(value) in {bool, int, str}:
        return ('scalar', value)
    if type(value) is float:
        _reject(not math.isfinite(value), 'Average JSON extras contain a nonfinite float')
        return ('scalar', value)
    _reject(not isinstance(value, Mapping) and type(value) not in {list, tuple}, 'Average JSON extras contain a non-JSON value', TypeError)
    _reject(depth >= _JSON_MAX_DEPTH, 'Average JSON extras exceed the nesting limit')
    identity = id(value)
    _reject(identity in active, 'Average JSON extras contain a cycle')
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            rows = []
            for key, item in value.items():
                _reject(type(key) is not str or not key, 'Average JSON object keys must be nonempty strings', TypeError)
                rows.append((key, _freeze_json(item, count, active, depth + 1)))
            return ('object', tuple(sorted(rows)))
        return ('array', tuple((_freeze_json(item, count, active, depth + 1) for item in value)))
    finally:
        active.remove(identity)
def _thaw_json(value: tuple) -> Any:
    tag, body = value
    if tag == 'scalar':
        return body
    if tag == 'array':
        return tuple((_thaw_json(item) for item in body))
    if tag == 'object':
        return {key: _thaw_json(item) for key, item in body}
    raise ValueError('Average frozen JSON tag is invalid')
def _semantic_pairs(value: tuple[tuple[str, tuple], ...]) -> dict[str, Any]:
    return {key: _thaw_json(item) for key, item in value}
def _freeze_pairs(value: Mapping[str, Any], count: list[int]) -> tuple:
    _reject(not isinstance(value, Mapping), 'Average extras must be a mapping', TypeError)
    rows = []
    for key, item in value.items():
        _reject(type(key) is not str or not key, 'Average extra keys must be nonempty strings', TypeError)
        rows.append((key, _freeze_json(item, count, set())))
    return tuple(sorted(rows))
def _copy_calibration(value: CalibrationState) -> CalibrationState:
    _reject(type(value) is not CalibrationState, 'Average calibration must be an exact CalibrationState', TypeError); poni = value.values
    copied_values = None if poni is None else PoniValues(*(getattr(poni, item.name) for item in fields(PoniValues)))
    mask = value.mask; copied_mask = MaskState(mask.source_uri, mask.sha256, mask.dtype, tuple(mask.shape), mask.status)
    return CalibrationState(copied_values, value.detector_id, dict(value.detector_config), value.value_fingerprint, value.source_sha256, value.source_uri, copied_mask, value.status)
def _copy_background(value: FrameBackgroundPlan | None) -> FrameBackgroundPlan | None:
    if value is None: return None
    _reject(type(value) is not FrameBackgroundPlan, 'Average background must be an exact FrameBackgroundPlan', TypeError)
    if value.mode == 'None': return None
    return FrameBackgroundPlan(**{item.name: getattr(value, item.name) for item in fields(FrameBackgroundPlan)})
def _absolute(value: str | Path, *, optional: bool=False) -> str | None:
    if optional and (value is None or str(value) == ''): return None
    _reject(not isinstance(value, (str, Path)) or not str(value), 'Average path is empty or invalid'); return os.path.abspath(os.path.expanduser(str(value)))
def _metadata_keys(value, name: str, *, optional: bool) -> tuple[str, ...] | None:
    if optional and value is None: return None
    _reject(type(value) is not tuple, f'{name} must be an exact tuple', TypeError)
    _reject(any((type(item) is not str or not item for item in value)), f'{name} must contain nonempty exact strings')
    _reject(len(value) != len(set(value)), f'{name} contains a duplicate metadata key')
    _reject(len(value) > _METADATA_MAX_KEYS, f'{name} exceeds the metadata-key cap')
    return tuple(value)
def _compact_source(value: SourceSpec) -> FrozenSourceSpec:
    _reject(type(value) is not SourceSpec, 'Average source must be an exact SourceSpec', TypeError)
    kind = SourceKind(getattr(value.kind, 'value', value.kind))
    if kind is not SourceKind.TIFF_SERIES:
        return FrozenSourceSpec.from_source(value)
    options = dict(value.options)
    allowed = {'selected_file', 'files', 'pattern', 'scan_name', 'metadata_format', 'selection_mode', 'meta_dir', 'detector_shape', 'detector', 'raw_dtype', 'raw_header_skip', 'admitted_motor_values'}
    _reject(set(options) - allowed, 'Average TIFF source has unsupported options')
    selected = options.get('selected_file')
    _reject(type(selected) is not str or not selected, 'Average TIFF source has no selected member')
    expanded = single_image_spec(selected, metadata_format=options.get('metadata_format')) if is_single_image_spec(value) else image_series_spec(selected, metadata_format=options.get('metadata_format'))
    members = tuple((str(item) for item in expanded.options.get('files', ())))
    _reject(not members or sum((item == selected for item in members)) != 1, 'Average TIFF selection is empty or ambiguous')
    frozen_options = {key: item for key, item in dict(expanded.options).items() if key not in {'files', 'admitted_motor_values'}}
    for key in ('meta_dir', 'detector_shape', 'detector', 'raw_dtype', 'raw_header_skip'):
        if key in options:
            frozen_options[key] = options[key]
    frozen_options['average_compact_series_v1'] = True
    return FrozenSourceSpec.from_source(SourceSpec(expanded.uri, expanded.kind, metadata_uri=value.metadata_uri, entry=value.entry, options=frozen_options))
def _integration_1d(value: Integration1DPlan | None, count: list[int]):
    if value is None:
        return None
    _reject(type(value) is not Integration1DPlan, 'Average 1-D plan must be exact', TypeError)
    return (value.npt, value.unit, value.method, value.radial_range, value.azimuth_range, value.monitor_key, value.error_model, value.polarization_factor, value.npt_rad, value.azimuth_offset, _freeze_pairs(value.extra, count))
def _integration_2d(value: Integration2DPlan | None, count: list[int]):
    if value is None:
        return None
    _reject(type(value) is not Integration2DPlan, 'Average 2-D plan must be exact', TypeError)
    return (value.npt_rad, value.npt_azim, value.unit, value.method, value.radial_range, value.azimuth_range, value.azimuth_offset, value.monitor_key, value.error_model, value.polarization_factor, _freeze_pairs(value.extra, count))
def _gi(value: GIMode | None):
    if value is None:
        return None
    _reject(type(value) is not GIMode, 'Average GI plan must be exact', TypeError)
    return (value.incident_angle, value.incidence_motor, value.tilt_angle, value.sample_orientation, value.method, value.mode_1d.value, value.mode_2d.value, value.npt_oop)
@dataclass(frozen=True, slots=True, init=False)
class AverageScanRecipe:
    api_version: str; source: FrozenSourceSpec
    target: str; entry: str; source_base: str | None
    output_mode: str; live_mode: bool
    save_xye: bool; batch_mode: bool
    integration_1d: tuple | None; integration_2d: tuple | None
    integrator_gi: tuple | None
    threshold_min: float | None; threshold_max: float | None
    mask_saturation: bool; reduction_extra: tuple[tuple[str, tuple], ...]
    calibration: CalibrationState; background: FrameBackgroundPlan | None
    numeric_metadata_keys: tuple[str, ...] | None; invariant_metadata_keys: tuple[str, ...]
    envelope_bytes: int | None; resource_requests: tuple[tuple[str, int], ...]
    resource_env: tuple[tuple[str, str], ...]
    def __init__(self, source, target, reduction=None, *, entry='entry', source_base=None, output_mode='Overwrite', live_mode=False, save_xye=False, batch_mode=False, calibration=CalibrationState(), background=None, numeric_metadata_keys=None, invariant_metadata_keys=(), envelope_bytes=None, resource_requests=None, resource_env=None, **canonical) -> None:
        if type(source) is FrozenSourceSpec and reduction is None:
            required = {'api_version', 'integration_1d', 'integration_2d', 'integrator_gi', 'threshold_min', 'threshold_max', 'mask_saturation', 'reduction_extra'}
            _reject(set(canonical) != required or canonical['api_version'] != _COUNT_POLICY, 'Average canonical recipe reconstruction is invalid', TypeError)
            values = {'api_version': _COUNT_POLICY, 'source': source, 'target': target, 'entry': entry, 'source_base': source_base, 'output_mode': output_mode, 'live_mode': live_mode, 'save_xye': save_xye, 'batch_mode': batch_mode, **{key: canonical[key] for key in required - {'api_version'}}, 'calibration': _copy_calibration(calibration), 'background': _copy_background(background), 'numeric_metadata_keys': numeric_metadata_keys, 'invariant_metadata_keys': invariant_metadata_keys, 'envelope_bytes': envelope_bytes, 'resource_requests': tuple(resource_requests), 'resource_env': tuple(resource_env)}
            for name, item in values.items():
                object.__setattr__(self, name, item)
            return
        _reject(canonical or type(reduction) is not ReductionPlan or reduction.mask is not None, 'Average recipe requires one exact mask-free ReductionPlan', TypeError)
        _reject(type(entry) is not str or not entry or '/' in entry or (entry in {'.', '..'}), 'Average entry must be one nonempty component')
        mode = str(output_mode).strip().casefold()
        _reject(mode not in {'append', 'overwrite'}, 'Average output mode is invalid')
        _reject(any((type(value) is not bool for value in (live_mode, save_xye, batch_mode, reduction.mask_saturation))), 'Average Boolean fields must be exact', TypeError)
        numeric = _metadata_keys(numeric_metadata_keys, 'numeric_metadata_keys', optional=True)
        invariant = _metadata_keys(invariant_metadata_keys, 'invariant_metadata_keys', optional=False)
        _reject(numeric is not None and set(numeric) & set(invariant), 'Average metadata key roles overlap')
        _reject(envelope_bytes is not None and (type(envelope_bytes) is not int or envelope_bytes <= 0), 'Average envelope_bytes must be a positive exact integer')
        requests = resource_requests or {}
        env = resource_env or {}
        _reject(not isinstance(requests, Mapping) or 'owner_block_bytes' in requests, 'Average owns the owner_block_bytes request')
        _reject(any((type(key) is not str or type(item) is not int or type(item) is bool for key, item in requests.items())), 'Average resource requests are invalid', TypeError)
        _reject(not isinstance(env, Mapping) or len(env) > 32 or any((type(key) is not str or type(item) is not str for key, item in env.items())) or (sum((len(key.encode()) + len(item.encode()) for key, item in env.items())) > 8192), 'Average resource environment is invalid')
        count = [0]
        one = _integration_1d(reduction.integration_1d, count)
        two = _integration_2d(reduction.integration_2d, count)
        extra = _freeze_pairs(reduction.extra, count)
        values = {'api_version': _COUNT_POLICY, 'source': _compact_source(source), 'target': _absolute(target), 'entry': entry, 'source_base': _absolute(source_base, optional=True), 'output_mode': mode.title(), 'live_mode': live_mode, 'save_xye': save_xye, 'batch_mode': batch_mode, 'integration_1d': one, 'integration_2d': two, 'integrator_gi': _gi(reduction.gi), 'threshold_min': reduction.threshold_min, 'threshold_max': reduction.threshold_max, 'mask_saturation': reduction.mask_saturation, 'reduction_extra': extra, 'calibration': _copy_calibration(calibration), 'background': _copy_background(background), 'numeric_metadata_keys': numeric, 'invariant_metadata_keys': invariant, 'envelope_bytes': envelope_bytes, 'resource_requests': tuple(sorted(requests.items())), 'resource_env': tuple(sorted(env.items()))}
        _reject(len(_canonical(_recipe_payload_values(values))) > _JSON_MAX_BYTES, 'Average recipe JSON projection exceeds 64 KiB')
        for name, item in values.items():
            object.__setattr__(self, name, item)
def _thaw_reduction(recipe: AverageScanRecipe) -> ReductionPlan:
    one = recipe.integration_1d
    two = recipe.integration_2d
    gi = recipe.integrator_gi
    return ReductionPlan(integration_1d=None if one is None else Integration1DPlan(npt=one[0], unit=one[1], method=one[2], radial_range=one[3], azimuth_range=one[4], monitor_key=one[5], error_model=one[6], polarization_factor=one[7], npt_rad=one[8], azimuth_offset=one[9], extra=_semantic_pairs(one[10])), integration_2d=None if two is None else Integration2DPlan(npt_rad=two[0], npt_azim=two[1], unit=two[2], method=two[3], radial_range=two[4], azimuth_range=two[5], azimuth_offset=two[6], monitor_key=two[7], error_model=two[8], polarization_factor=two[9], extra=_semantic_pairs(two[10])), gi=None if gi is None else GIMode(incident_angle=gi[0], incidence_motor=gi[1], tilt_angle=gi[2], sample_orientation=gi[3], method=gi[4], mode_1d=gi[5], mode_2d=gi[6], npt_oop=gi[7]), mask=None, threshold_min=recipe.threshold_min, threshold_max=recipe.threshold_max, mask_saturation=recipe.mask_saturation, extra=_semantic_pairs(recipe.reduction_extra))
def _calibration_payload(value: CalibrationState) -> dict[str, Any]:
    poni = value.values
    return {'values': None if poni is None else {item.name: getattr(poni, item.name) for item in fields(PoniValues)}, 'detector_id': value.detector_id, 'detector_config': _plain_json(value.detector_config), 'value_fingerprint': value.value_fingerprint, 'source_sha256': value.source_sha256, 'source_uri': value.source_uri, 'mask': {item.name: getattr(value.mask, item.name).value if item.name == 'status' else getattr(value.mask, item.name) for item in fields(MaskState)}, 'status': value.status.value}
def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_plain_json(item) for item in value]
    return value
def _recipe_payload_values(values: Mapping[str, Any]) -> dict[str, Any]:
    semantic = lambda value: None if value is None else [*value[:-1], _semantic_pairs(value[-1])]
    background = values['background']
    return {'api_version': values['api_version'], 'source': values['source'].as_dict(), 'target': values['target'], 'entry': values['entry'], 'source_base': values['source_base'], 'output_mode': values['output_mode'], 'live_mode': values['live_mode'], 'save_xye': values['save_xye'], 'batch_mode': values['batch_mode'], 'integration_1d': semantic(values['integration_1d']), 'integration_2d': semantic(values['integration_2d']), 'integrator_gi': values['integrator_gi'], 'threshold_min': values['threshold_min'], 'threshold_max': values['threshold_max'], 'mask_saturation': values['mask_saturation'], 'reduction_extra': _semantic_pairs(values['reduction_extra']), 'calibration': _calibration_payload(values['calibration']), 'background': None if background is None else background.to_mapping(), 'numeric_metadata_keys': values['numeric_metadata_keys'], 'invariant_metadata_keys': values['invariant_metadata_keys'], 'envelope_bytes': values['envelope_bytes'], 'resource_requests': values['resource_requests'], 'resource_env': values['resource_env']}
def _recipe_payload(recipe: AverageScanRecipe) -> dict[str, Any]:
    return _recipe_payload_values({item.name: getattr(recipe, item.name) for item in fields(recipe)})
@dataclass(frozen=True, slots=True)
class AverageScanPlan:
    recipe: AverageScanRecipe; source_graph_digest: str; contributor_extent: int
    detector_shape: tuple[int, int]; native_dtype: str; expected_target_snapshot: TargetSnapshot
    numeric_metadata_keys: tuple[str, ...]; invariant_metadata_keys: tuple[str, ...]; allocation: SessionResourceAllocation
    science_identity: str; operation_identity: str
    logical_labels: tuple[int, ...] = (1,)
    def __post_init__(self) -> None:
        _reject(type(self.recipe) is not AverageScanRecipe or type(self.contributor_extent) is not int or (not 1 <= self.contributor_extent <= _MAX_CONTRIBUTORS) or (type(self.detector_shape) is not tuple) or (len(self.detector_shape) != 2) or any((type(item) is not int or item <= 0 for item in self.detector_shape)) or (type(self.expected_target_snapshot) is not TargetSnapshot) or (type(self.allocation) is not SessionResourceAllocation) or (self.logical_labels != (1,)), 'Average scan plan is invalid')
@dataclass(frozen=True, slots=True)
class AverageScanProgress:
    operation_identity: str; stage: str; completed: int; total: int; revision: int
    def __post_init__(self) -> None:
        _reject(type(self.operation_identity) is not str or self.stage not in _STAGES or any((type(item) is not int for item in (self.completed, self.total, self.revision))) or (not 0 <= self.completed <= self.total) or (self.revision < 0), 'Average progress value is invalid')
@dataclass(frozen=True, slots=True)
class AverageScanResult:
    disposition: str; target: str; entry: str; operation_identity: str; science_identity: str
    contributor_extent: int; logical_labels: tuple[int, ...]; committed_labels: tuple[int, ...]
    metadata_denominators: tuple[tuple[str, int], ...]; finite_counts: AverageFiniteCountsEvidence | None
    diagnostic_code: str; diagnostic: str; h23_phase: str | None; commit_identity: StreamTerminal | None
    def __post_init__(self):
        terminal = self.disposition in {'COMMITTED', 'REFUSED', 'CANCELLED', 'ABORTED', 'SETTLEMENT_PENDING'}
        denominators = self.metadata_denominators; logical = self.logical_labels; committed_labels = self.committed_labels
        base = all(type(value) is str for value in (self.disposition, self.target, self.entry, self.operation_identity, self.science_identity, self.diagnostic_code, self.diagnostic)) and type(self.contributor_extent) is int and 0 <= self.contributor_extent <= _MAX_CONTRIBUTORS and type(logical) is tuple and logical == (1,) and all(type(value) is int for value in logical) and type(denominators) is tuple and all(type(value) is tuple and len(value) == 2 and type(value[0]) is str and bool(value[0]) and type(value[1]) is int and value[1] >= 0 for value in denominators) and len({value[0] for value in denominators}) == len(denominators)
        committed = type(committed_labels) is tuple and committed_labels == (1,) and all(type(value) is int for value in committed_labels) and type(self.finite_counts) is AverageFiniteCountsEvidence and self.finite_counts.contributor_extent == self.contributor_extent and self.h23_phase == 'committed' and type(self.commit_identity) is StreamTerminal and not self.diagnostic_code and not self.diagnostic
        source_pending = self.h23_phase is None and self.diagnostic_code == self.diagnostic == 'AVERAGE_SOURCE_CLEANUP_PENDING'
        h23_pending = self.h23_phase in {'ready-to-retry', 'rollback-pending', 'cleanup-pending', 'integrity-hold'} and self.diagnostic_code == self.diagnostic == 'AVERAGE_H23_SETTLEMENT_PENDING'
        empty = type(committed_labels) is tuple and committed_labels == () and self.finite_counts is None and self.commit_identity is None
        valid = committed if self.disposition == 'COMMITTED' else empty and (source_pending or h23_pending) if self.disposition == 'SETTLEMENT_PENDING' else empty and self.h23_phase is None
        _reject(not terminal or not base or not valid, 'Average result contract is invalid')
@dataclass(frozen=True, slots=True)
class AverageContributor:
    ordinal: int; logical_index: int; source_path: str; source_frame_index: int
    dataset_path: str | None; source_start: int; source_stop: int; size: int; mtime_ns: int
    def __post_init__(self) -> None:
        integers = (self.ordinal, self.logical_index, self.source_frame_index, self.source_start, self.source_stop, self.size, self.mtime_ns)
        _reject(any((type(item) is not int or item < 0 for item in integers)) or self.ordinal != self.logical_index or self.source_stop <= self.source_start or (type(self.source_path) is not str) or (not self.source_path) or (self.dataset_path is not None and (type(self.dataset_path) is not str or not self.dataset_path)), 'Average contributor value is invalid')
@dataclass(frozen=True, slots=True)
class AverageFiniteCountsEvidence:
    policy: str; contributor_extent: int; shape: tuple[int, int]; dtype: str; sha256: str
    minimum: int; maximum: int; zero_count: int; chunks: tuple[int, int]
    compression: str; compression_opts: int; shuffle: bool; fletcher32: bool
    def __post_init__(self) -> None:
        height_width = self.shape
        extent = self.contributor_extent
        _reject(self.policy != _COUNT_POLICY or type(extent) is not int or (not 1 <= extent <= _MAX_CONTRIBUTORS) or (type(height_width) is not tuple) or (len(height_width) != 2) or any((type(value) is not int or value <= 0 for value in height_width)) or (self.dtype != '<u4') or (type(self.sha256) is not str) or (_COUNT_DIGEST.fullmatch(self.sha256) is None) or any((type(value) is not int for value in (self.minimum, self.maximum, self.zero_count))) or (not 0 <= self.minimum <= self.maximum <= extent) or (not 0 <= self.zero_count < height_width[0] * height_width[1]) or (type(self.chunks) is not tuple) or (len(self.chunks) != 2) or any((type(value) is not int or value <= 0 for value in self.chunks)) or (self.chunks[1] != height_width[1]) or (self.chunks[0] > height_width[0]) or (self.compression != 'gzip') or (type(self.compression_opts) is not int) or (self.compression_opts != 1) or (self.shuffle is not True) or (self.fletcher32 is not False), 'average finite-count extent/count/zero census or schema is invalid')
@dataclass(frozen=True, slots=True)
class AverageFiniteCounts:
    evidence: AverageFiniteCountsEvidence; values: np.ndarray
    def __post_init__(self) -> None:
        _reject(type(self.evidence) is not AverageFiniteCountsEvidence, 'average finite-count evidence is invalid')
        incoming = np.asarray(self.values)
        _reject(incoming.ndim != 2 or incoming.shape != self.evidence.shape, 'average finite-count shape is invalid')
        owned = np.frombuffer(np.ascontiguousarray(incoming, dtype=_COUNT_DTYPE).tobytes(order='C'), dtype=_COUNT_DTYPE).reshape(incoming.shape)
        owned.setflags(write=False)
        _reject(int(owned.min()) != self.evidence.minimum or int(owned.max()) != self.evidence.maximum or int(np.count_nonzero(owned == 0)) != self.evidence.zero_count or (int(owned.max()) > self.evidence.contributor_extent), 'average finite-count census is invalid')
        object.__setattr__(self, 'values', owned)
    @classmethod
    def _from_owned_array(cls, evidence: AverageFiniteCountsEvidence, owner: np.ndarray) -> 'AverageFiniteCounts':
        _reject(type(evidence) is not AverageFiniteCountsEvidence or type(owner) is not np.ndarray or owner.dtype != _COUNT_DTYPE or (owner.shape != evidence.shape) or (not owner.flags.c_contiguous) or (owner.base is not None), 'average finite-count reader owner is invalid')
        owner.setflags(write=False)
        exposed = owner.view()
        exposed.setflags(write=False)
        result = object.__new__(cls)
        object.__setattr__(result, 'evidence', evidence)
        object.__setattr__(result, 'values', exposed)
        return result
def _metadata_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    matches = [value for name, value in row.items() if type(name) is str and name.casefold() == key.casefold()]
    _reject(len(matches) > 1, 'AVERAGE_METADATA_KEY_AMBIGUOUS')
    return None if not matches else matches[0]
def _configured_key_union(*groups: tuple[str, ...]) -> tuple[str, ...]:
    domains: dict[str, str] = {}
    for group in groups:
        for key in group:
            prior = domains.setdefault(key.casefold(), key)
            _reject(prior != key, 'AVERAGE_METADATA_KEY_CASEFOLD_DOMAIN_COLLISION')
    return tuple(sorted(set(domains.values())))
def _project_metadata(row: Mapping[str, Any], numeric: tuple[str, ...], invariant: tuple[str, ...]) -> dict[str, Any]:
    return {key: _metadata_value(row, key) for key in (*numeric, *invariant)}
def _configured_roles(plan: ReductionPlan) -> tuple[str, ...]:
    monitors = tuple((item for item in (None if plan.integration_1d is None else plan.integration_1d.monitor_key, None if plan.integration_2d is None else plan.integration_2d.monitor_key) if item))
    motors = tuple((item for item in (None if plan.gi is None or plan.gi.incident_angle is not None or plan.gi.incidence_motor in {None, 'Manual'} else plan.gi.incidence_motor,) if item))
    return _configured_key_union(monitors, motors)
def _selected_metadata_keys(recipe: AverageScanRecipe, row: Mapping[str, Any], reduction: ReductionPlan) -> tuple[tuple[str, ...], tuple[str, ...]]:
    roles = _configured_roles(reduction)
    invariant = tuple(recipe.invariant_metadata_keys)
    numeric = recipe.numeric_metadata_keys
    if numeric is None:
        optional = tuple(sorted((key for key, value in row.items() if type(key) is str and type(value) in {int, float} and (type(value) is not bool) and math.isfinite(float(value)) and (key.casefold() not in {role.casefold() for role in (*roles, *invariant)}))))
        numeric = optional[:max(0, _METADATA_MAX_KEYS - len(roles))]
    numeric = _configured_key_union(tuple(numeric), roles)
    _configured_key_union(roles, tuple((key for key in (*numeric, *invariant) if key.casefold() in {role.casefold() for role in roles})))
    _reject(set(numeric) & set(invariant), 'AVERAGE_METADATA_KEY_ROLE_OVERLAP')
    _reject(len(numeric) > _METADATA_MAX_KEYS or len(invariant) > _METADATA_MAX_KEYS, 'AVERAGE_METADATA_KEY_DOMAIN_TOO_LARGE')
    return (tuple(sorted(numeric)), tuple(sorted(invariant)))
def _background_resource_terms(plan: FrameBackgroundPlan | None, pixels: int) -> tuple[int, int, int, int]:
    if plan is None or plan.mode == 'None':
        return (0, 0, 0, 0)
    _reject(type(plan) is not FrameBackgroundPlan, 'Average Background resource policy is invalid', TypeError)
    return (8 * pixels, (25 if plan.mode == 'Series Average' else 8) * pixels, 8 * pixels, 64 * 1024 ** 2)
def _owner_block_bytes(shape: tuple[int, int]) -> int:
    height, width = shape
    pixels = height * width
    frame = 8 * pixels
    counts = 4 * pixels
    mask = pixels
    rows, _ = _average_count_chunks(shape)
    evidence_scratch = 4 * (4 * rows * width)
    peak = max(2 * frame + counts + 4 * mask + _JSON_MAX_BYTES, frame + counts + mask + _JSON_MAX_BYTES + evidence_scratch)
    return (peak + frame - 1) // frame * frame
def _selected_motor(recipe: AverageScanRecipe) -> str | None:
    gi = recipe.integrator_gi
    return None if gi is None or gi[1] in (None, 'Manual') else gi[1]
def _average_allocation(recipe: AverageScanRecipe, shape: tuple[int, int], allocation: SessionResourceAllocation | None=None) -> SessionResourceAllocation:
    descriptor = SimpleNamespace(frame_shape=shape, dtype=np.dtype('<f8'))
    terms = _background_resource_terms(recipe.background, shape[0] * shape[1])
    requirements = requirements_from(descriptor, _derived_reduction(recipe), background_bytes=terms[0], resolver_background_bytes=terms[1], worker_background_bytes=terms[2], background_binding_bytes=terms[3])
    owner = _owner_block_bytes(shape); requests = dict(recipe.resource_requests); requests['owner_block_bytes'] = owner
    current = allocation if allocation is not None else resolve_session_policy(requirements, envelope_bytes=recipe.envelope_bytes, requests=requests, env=dict(recipe.resource_env)).allocation
    exact_categories = None if current is None else _resource_policy._categories(requirements, dict(current.counts))
    if current is None or current.requirements.fingerprint != requirements.fingerprint or current.owner_block_bytes != owner or dict(current.categories) != exact_categories:
        granted = None if current is None else current.owner_block_bytes
        code = 'AVERAGE_ALLOCATION_CHANGED' if allocation is not None else 'AVERAGE_OWNER_BLOCK_GRANT_INCOMPLETE'
        raise ValueError(f'{code}(required={owner}, granted={granted})')
    return current
def _science_payload(plan: AverageScanPlan) -> dict[str, Any]:
    recipe = plan.recipe
    payload = _recipe_payload(recipe)
    modes = required_result_modes(_derived_reduction(recipe))
    return {'api_version': _COUNT_POLICY, 'integration_1d': payload['integration_1d'], 'integration_2d': payload['integration_2d'], 'integrator_gi': payload['integrator_gi'], 'threshold_min': recipe.threshold_min, 'threshold_max': recipe.threshold_max, 'mask_saturation': recipe.mask_saturation, 'reduction_extra': payload['reduction_extra'], 'calibration': payload['calibration'], 'background': payload['background'], 'pixel_accumulator_policy': 'float64_ordered_inplace_v1', 'finite_count_policy': 'uint32_per_pixel_extent_1000000_v1', 'zero_count_policy': 'nan_and_all_zero_refusal_v1', 'overflow_policy': 'post_add_nonfinite_refusal_v1', 'numeric_metadata_keys': plan.numeric_metadata_keys, 'invariant_metadata_keys': plan.invariant_metadata_keys, 'numeric_metadata_policy': 'binary64_neumaier_per_key_denominator_v1', 'invariant_metadata_policy': 'canonical_json_scalar_exact_all_v1', 'metadata_row_policy': 'graph_bound_complete_scalar_row_256_names_65536_bytes_v1', 'metadata_source_policy': 'container_metadata_inputs_and_tiff_preparse_bounded_v2', 'configured_metadata_lookup_policy': 'exact_first_unique_casefold_single_output_domain_v2', 'default_optional_lookup_policy': 'exact_source_spelling_omit_configured_alias_v2', 'conditioning_order': ['threshold', 'frame_background', 'static_frame_mask', 'detector_value_mask', 'finite_accumulate'], 'result_modes': [f'{mode.kind}:{mode.key}' for mode in modes], 'reduction_strict_policy': {'policy': 'average_reduction_strict_v1', 'missing_normalization': True, 'gi_all_dummy': True, 'thumbnail_fallback': True}}
def _allocation_payload(value: SessionResourceAllocation) -> dict[str, Any]:
    return {'requirements': {item.name: getattr(value.requirements, item.name) for item in fields(value.requirements)}, 'envelope_bytes': value.envelope_bytes, 'counts': dict(value.counts), 'categories': dict(value.categories), 'minimum_bytes': value.minimum_bytes, 'floor_bytes': value.floor_bytes, 'assigned_bytes': value.assigned_bytes, 'origin': value.origin, 'oversize_excess_bytes': value.oversize_excess_bytes}
def _operation_payload(plan: AverageScanPlan) -> dict[str, Any]:
    recipe = plan.recipe
    return {'api_version': _COUNT_POLICY, 'source_graph_digest': plan.source_graph_digest, 'science_identity': plan.science_identity, 'target': recipe.target, 'entry': recipe.entry, 'source_base': recipe.source_base or '', 'output_mode': recipe.output_mode, 'live_mode': recipe.live_mode, 'save_xye': recipe.save_xye, 'batch_mode': recipe.batch_mode, 'contributor_extent': plan.contributor_extent, 'detector_shape': plan.detector_shape, 'native_dtype': plan.native_dtype, 'numeric_metadata_keys': plan.numeric_metadata_keys, 'invariant_metadata_keys': plan.invariant_metadata_keys, 'logical_labels': plan.logical_labels, 'expected_target_snapshot': {item.name: getattr(plan.expected_target_snapshot, item.name) for item in fields(plan.expected_target_snapshot)}, 'resource_inputs': {'envelope_bytes': recipe.envelope_bytes, 'resource_requests': recipe.resource_requests, 'resource_env': recipe.resource_env}, 'allocation': _allocation_payload(plan.allocation)}
def _derived_reduction(recipe: AverageScanRecipe) -> ReductionPlan:
    value = _thaw_reduction(recipe)
    value.mask = None
    value.threshold_min = value.threshold_max = None
    value.mask_saturation = False
    return value
def _cancelled(token: threading.Event | None) -> bool:
    _reject(token is not None and type(token) is not threading.Event, 'Average cancellation token must be an exact Event or None', TypeError)
    return bool(token is not None and token.is_set())
def _source_from_recipe(recipe: AverageScanRecipe) -> SourceSpec:
    source = recipe.source.thaw()
    _reject(type(source) is not SourceSpec, 'Average frozen source did not thaw to SourceSpec', TypeError)
    return source
def _source_for_requalification(source: SourceSpec,
                                graph: PreparedSourceExecutionGraph) -> SourceSpec:
    if (source.kind is SourceKind.NEXUS_STACK
            and graph.execution_source.kind is SourceKind.EIGER_MASTER):
        return SourceSpec(source.uri, SourceKind.EIGER_MASTER,
            metadata_uri=source.metadata_uri, entry=source.entry,
            options=dict(source.options))
    return source
def prepare_average_scan(recipe: AverageScanRecipe, *, cancel_token: threading.Event | None=None) -> AverageScanPlan:
    _reject(type(recipe) is not AverageScanRecipe, 'prepare_average_scan requires an exact recipe', TypeError)
    _cancelled(cancel_token) and (_ for _ in ()).throw(InterruptedError('Average preparation cancelled'))
    _reject(recipe.background is not None,
            'AVERAGE_BACKGROUND_AGGREGATE_PROVENANCE_UNSUPPORTED')
    _reject(recipe.live_mode, 'AVERAGE_LIVE_TERMINALITY_UNSUPPORTED')
    _reject(recipe.output_mode != 'Overwrite', 'AVERAGE_APPEND_UNSUPPORTED')
    _reject(recipe.save_xye, 'AVERAGE_NEXUS_REQUIRED')
    source = _source_from_recipe(recipe)
    selected_motor = _selected_motor(recipe)
    graph = qualify_source_execution_graph(source, selected_motor=selected_motor, reader_binding=_READER_BINDING, cancelled=None if cancel_token is None else cancel_token.is_set)
    extent = graph.stamp.frame_count
    _reject(type(extent) is not int or not 1 <= extent <= _MAX_CONTRIBUTORS, 'AVERAGE_CONTRIBUTOR_EXTENT_INVALID')
    _reject(graph.detector_shape is None or graph.native_dtype is None, 'AVERAGE_DETECTOR_LAYOUT_UNAVAILABLE')
    dtype = np.dtype(graph.native_dtype)
    _reject(dtype.kind not in 'iuf' or dtype.itemsize > 8 or dtype.fields is not None, 'AVERAGE_NATIVE_DTYPE_UNSUPPORTED')
    window = open_source_execution_graph(graph, cancelled=None if cancel_token is None else cancel_token.is_set)
    try: window.__enter__(); first_row = window.complete_metadata_for(0)
    finally: window.__exit__(None, None, None)
    try:
        requalify_source_execution_graph(_source_for_requalification(source, graph), graph, selected_motor=selected_motor, reader_binding=_READER_BINDING, cancelled=None if cancel_token is None else cancel_token.is_set)
    except SourceRevisionChanged as error:
        raise ValueError('AVERAGE_SOURCE_DRIFT') from error
    reduction = _thaw_reduction(recipe)
    numeric, invariant = _selected_metadata_keys(recipe, first_row, reduction)
    shape = tuple((int(item) for item in graph.detector_shape))
    allocation = _average_allocation(recipe, shape)
    graph_digest = source_graph_digest(graph)
    snapshot = capture_target_snapshot(recipe.target)
    provisional = AverageScanPlan(recipe, graph_digest, extent, shape, dtype.str, snapshot, numeric, invariant, allocation, '', '')
    science = hashlib.sha256(b'xdart.average-science.v1\x00' + _canonical(_science_payload(provisional))).hexdigest()
    prospective = AverageScanPlan(recipe, graph_digest, extent, shape, dtype.str, snapshot, numeric, invariant, allocation, science, '')
    operation = hashlib.sha256(b'xdart.average-operation.v1\x00' + _canonical(_operation_payload(prospective))).hexdigest()
    return AverageScanPlan(recipe, graph_digest, extent, shape, dtype.str, snapshot, numeric, invariant, allocation, science, operation)
class _AverageCancelled(RuntimeError): pass
def _poll(token: threading.Event | None) -> None:
    if _cancelled(token):
        raise _AverageCancelled('Average operation cancelled')
def _diagnostic(value: BaseException | str) -> str:
    try:
        return str(value).encode('utf-8')[:1024].decode('utf-8', 'ignore')
    except BaseException:
        return type(value).__name__
def _result(plan: AverageScanPlan, disposition: str, *, code: str='', diagnostic: str='', denominators=(), evidence=None, h23_phase=None, commit=None) -> AverageScanResult:
    committed = (1,) if disposition == 'COMMITTED' else ()
    return AverageScanResult(disposition, plan.recipe.target, plan.recipe.entry, plan.operation_identity, plan.science_identity, plan.contributor_extent, plan.logical_labels, committed, tuple(denominators), evidence if disposition == 'COMMITTED' else None, code, diagnostic, 'committed' if disposition == 'COMMITTED' else h23_phase, commit if disposition == 'COMMITTED' else None)
def _committed_average_mismatch(result: AverageScanResult, target: str | Path, entry: str) -> str | None:
    expected = _absolute(target); commit = result.commit_identity
    if result.target != expected or result.entry != entry: return 'target' if result.target != expected else 'entry'
    if type(commit) is not StreamTerminal or commit.target != expected or type(commit.ordinal) is not int or commit.ordinal <= 0: return 'commit'
    try: persisted = read_provenance(expected, entry=entry)['config'][_COUNT_POLICY]; snapshot = capture_target_snapshot(expected)
    except Exception: return 'commit'
    if type(persisted) is not dict or persisted.get('operation_identity') != result.operation_identity or persisted.get('science_identity') != result.science_identity: return 'operation' if type(persisted) is not dict or persisted.get('operation_identity') != result.operation_identity else 'science'
    return None if snapshot.exists and commit.size == snapshot.size and commit.digest == snapshot.digest else 'commit'
def _recipe_refusal(recipe: AverageScanRecipe, code: str) -> AverageScanResult:
    return AverageScanResult('REFUSED', recipe.target, recipe.entry, '', '', 0, (1,), (), (), None, code, code, None, None)
def _error_result(plan: AverageScanPlan, error: BaseException, *, h23: bool=False, denominators=()) -> AverageScanResult:
    if isinstance(error, _AverageCancelled):
        return _result(plan, 'CANCELLED', code='AVERAGE_CANCELLED', diagnostic=_diagnostic(error), denominators=denominators)
    if isinstance(error, SourceRevisionChanged):
        return _result(plan, 'ABORTED' if h23 else 'REFUSED', code='AVERAGE_SOURCE_DRIFT', diagnostic=_diagnostic(error), denominators=denominators)
    if isinstance(error, MissingNormalizationError):
        code = 'AVERAGE_REDUCTION_MISSING_NORMALIZATION'
    elif isinstance(error, GIAllDummyError):
        code = 'AVERAGE_REDUCTION_GI_ALL_DUMMY'
    else:
        message = _diagnostic(error)
        code = message.split('(', 1)[0] if message.startswith('AVERAGE_') else 'AVERAGE_EXECUTION_FAILED' if not h23 else 'AVERAGE_H23_FAILED'
    return _result(plan, 'ABORTED' if h23 else 'REFUSED', code=code, diagnostic=_diagnostic(error), denominators=denominators)
def _external_contributor_route(
    graph: PreparedSourceExecutionGraph,
) -> tuple[object, ...]:
    """Validate the ordered HDF member intervals once for a serial pass."""

    stamp = graph.stamp
    if stamp.members or not stamp.external_members:
        return ()
    prior_stop = 0
    prior_epoch = -1
    for member in stamp.external_members:
        if (member.first < prior_stop or member.stop > stamp.frame_count
                or member.epoch <= prior_epoch):
            raise SourceRevisionChanged(
                'Average external-member interval route changed')
        prior_stop = member.stop
        prior_epoch = member.epoch
    return stamp.external_members


def _validate_contributor(
    graph: PreparedSourceExecutionGraph, index: int,
    token: threading.Event | None, external_member=None,
) -> None:
    _poll(token)
    stamp = graph.stamp
    states = []
    if stamp.members:
        member = stamp.members[index]
        states.append(member)
        if stamp.metadata_sources:
            metadata = stamp.metadata_sources[index]
            if metadata.source_path != member.path:
                raise SourceRevisionChanged(
                    'Average TIFF metadata route changed')
            if metadata.metadata_file is not None:
                states.append(metadata.metadata_file)
    else:
        states.append(stamp.file)
        states.extend(stamp.dependency_files)
        if external_member is not None:
            if not (external_member.first <= index < external_member.stop):
                raise SourceRevisionChanged(
                    'Average external-member interval route changed')
            states.append(external_member.file)
    if any((not state.matches_disk() for state in states)):
        raise SourceRevisionChanged('Average contributor source changed')
    _poll(token)
def _background_fact(graph: PreparedSourceExecutionGraph, index: int, row: Mapping[str, Any], plan: FrameBackgroundPlan) -> tuple[object, ...]:
    if graph.stamp.members:
        path, selector, source_index = (graph.stamp.members[index].path, None, 0)
    else:
        path, selector, source_index = (graph.stamp.file.path, graph.dataset_paths[0] if graph.dataset_paths else None, index)
        for member in graph.stamp.external_members:
            if member.first <= index < member.stop:
                path, selector, source_index = (member.file.path, member.dataset, index - member.first)
                break
    items = []
    for key in sorted(set((item for item in (plan.metadata_key, plan.normalization_key) if item))):
        value = _metadata_value(row, key)
        _reject(value is None, 'AVERAGE_BACKGROUND_METADATA_MISSING')
        value = value.item() if isinstance(value, np.generic) else value
        if type(value) is bool:
            tagged = ('bool', value)
        elif type(value) in {int, float} and math.isfinite(float(value)):
            tagged = ('number', float(value).hex())
        elif type(value) is str and len(value.encode()) <= 4085:
            tagged = ('text', value)
        else:
            raise ValueError('AVERAGE_BACKGROUND_METADATA_INVALID')
        items.append((key, tagged))
    return (index + 1, str(Path(path).resolve(strict=False)), selector, source_index, graph.detector_shape, tuple(items))
def _load_static_mask(
    recipe: AverageScanRecipe, shape: tuple[int, int],
    token: threading.Event | None = None,
) -> np.ndarray | None:
    state = recipe.calibration.mask
    if state.status is not FactStatus.PRESENT:
        return None
    _poll(token)
    path = Path(state.source_uri)
    snapshot_limit = _static_mask_snapshot_limit(shape)
    with tempfile.TemporaryDirectory(prefix="xdart-average-mask-") as root:
        snapshot = Path(root) / ("mask" + path.suffix)
        digest = hashlib.sha256()
        copied = 0
        with path.open("rb") as source, snapshot.open("xb") as target:
            while True:
                _poll(token)
                payload = source.read(_STATIC_MASK_COPY_CHUNK_BYTES)
                if not payload:
                    break
                copied += len(payload)
                _reject(copied > snapshot_limit,
                        'AVERAGE_STATIC_MASK_FILE_TOO_LARGE')
                digest.update(payload)
                target.write(payload)
        _reject(digest.hexdigest() != state.sha256,
                'AVERAGE_STATIC_MASK_DIGEST_CHANGED')
        _poll(token)
        _qualify_static_mask_snapshot(snapshot, shape)
        _poll(token)
        value = _decode_static_mask_snapshot(snapshot)
        _poll(token)
    _reject(value.dtype.str != state.dtype or tuple(value.shape) != tuple(state.shape), 'AVERAGE_STATIC_MASK_SCHEMA_CHANGED')
    _reject(value.dtype != np.dtype(bool) or tuple(value.shape) != shape, 'AVERAGE_STATIC_MASK_SHAPE_INVALID')
    result = np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=bool).reshape(shape)
    result.setflags(write=False)
    return result
def _static_mask_snapshot_limit(shape: tuple[int, int]) -> int:
    pixels = int(shape[0]) * int(shape[1])
    _reject(pixels <= 0, 'AVERAGE_STATIC_MASK_SHAPE_INVALID')
    return max(
        _STATIC_MASK_COMPATIBILITY_FLOOR_BYTES,
        8 * pixels + _STATIC_MASK_HEADER_ALLOWANCE_BYTES,
    )
def _qualify_static_mask_snapshot(path: Path, shape: tuple[int, int]) -> None:
    if path.suffix.casefold() == ".npy":
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        try:
            observed_shape = tuple(value.shape)
            dtype = np.dtype(value.dtype)
        finally:
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()
        frame_count = 1
    else:
        layout = read_detector_image_layout(path)
        observed_shape = tuple(layout.shape)
        dtype = np.dtype(layout.dtype)
        frame_count = layout.frame_count
    _reject(
        observed_shape != tuple(shape)
        or frame_count != 1
        or dtype.kind not in "biuf"
        or dtype.itemsize > 8,
        'AVERAGE_STATIC_MASK_SCHEMA_CHANGED',
    )
def _decode_static_mask_snapshot(path: Path) -> np.ndarray:
    """Decode only the immutable file snapshot whose digest was accepted."""
    return load_mask(path)
def _row_value(row: Mapping[str, Any], key: str, configured: frozenset[str]) -> Any:
    return _metadata_value(row, key) if key in configured else row.get(key)
def _canonical_scalar(value: Any) -> bytes:
    value = value.item() if isinstance(value, np.generic) else value
    if value is None or type(value) in {bool, int, str}:
        return _canonical(value)
    if type(value) is float and math.isfinite(value):
        return _canonical(value)
    raise ValueError('AVERAGE_INVARIANT_METADATA_MISSING')
def _accumulate_metadata(row: Mapping[str, Any], plan: AverageScanPlan, configured: frozenset[str], sums: dict[str, list[float | int]], invariants: dict[str, tuple[bytes, Any]], first: bool) -> None:
    for key in plan.numeric_metadata_keys:
        value = _row_value(row, key, configured)
        if isinstance(value, np.generic):
            value = value.item()
        if type(value) not in {int, float} or type(value) is bool or (not math.isfinite(float(value))):
            continue
        total, compensation, denominator = sums[key]
        numeric = float(value)
        trial = total + numeric
        if abs(total) >= abs(numeric):
            compensation += total - trial + numeric
        else:
            compensation += numeric - trial + total
        _reject(not math.isfinite(trial) or not math.isfinite(compensation), 'AVERAGE_METADATA_SUM_OVERFLOW')
        sums[key] = [trial, compensation, int(denominator) + 1]
    for key in plan.invariant_metadata_keys:
        value = _row_value(row, key, configured)
        encoded = _canonical_scalar(value)
        if first:
            invariants[key] = (encoded, value.item() if isinstance(value, np.generic) else value)
        else:
            _reject(key not in invariants or invariants[key][0] != encoded, 'AVERAGE_INVARIANT_METADATA_CHANGED')
def _average_contributors(plan: AverageScanPlan, graph: PreparedSourceExecutionGraph, window, token: threading.Event | None, progress: Callable[[str, int, int], None]) -> dict[str, Any]:
    shape, extent = (plan.detector_shape, plan.contributor_extent)
    _reject(plan.recipe.background is not None,
            'AVERAGE_BACKGROUND_AGGREGATE_PROVENANCE_UNSUPPORTED')
    external_route = _external_contributor_route(graph)
    static = _load_static_mask(plan.recipe, shape, token)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=_COUNT_DTYPE)
    numeric = {key: [0.0, 0.0, 0] for key in plan.numeric_metadata_keys}
    invariants: dict[str, tuple[bytes, Any]] = {}
    configured = frozenset(_configured_roles(_thaw_reduction(plan.recipe)))
    needs_metadata = bool(
        plan.numeric_metadata_keys or plan.invariant_metadata_keys
    )
    detector_mask = None
    conditioned = working = valid = None
    external_cursor = 0
    for index in range(extent):
        while (external_cursor < len(external_route)
               and index >= external_route[external_cursor].stop):
            external_cursor += 1
        external_member = (
            external_route[external_cursor]
            if external_cursor < len(external_route)
            and external_route[external_cursor].first <= index
            else None
        )
        _validate_contributor(graph, index, token, external_member)
        native = np.asarray(window.read_native(index))
        row = window.complete_metadata_for(index) if needs_metadata else None
        _validate_contributor(graph, index, token, external_member)
        _reject(native.ndim != 2 or tuple(native.shape) != shape or native.dtype.str != plan.native_dtype or (native.dtype.kind not in 'iuf') or (native.dtype.itemsize > 8), 'AVERAGE_CONTRIBUTOR_LAYOUT_CHANGED')
        progress('read', index + 1, extent)
        if index == 0:
            mask = detector_value_mask(None, native, enabled=plan.recipe.mask_saturation)
            if mask is not None:
                detector_mask = np.frombuffer(np.ascontiguousarray(mask, dtype=bool).tobytes(), dtype=bool).reshape(shape)
                detector_mask.setflags(write=False)
            del mask
            conditioned = np.empty(shape, dtype=np.float64)
            working = np.empty(shape, dtype=bool)
            valid = np.empty(shape, dtype=bool)
        np.copyto(conditioned, native, casting='unsafe')
        working.fill(False)
        if plan.recipe.threshold_min is not None:
            np.less(native, plan.recipe.threshold_min, out=valid)
            np.logical_or(working, valid, out=working)
        if plan.recipe.threshold_max is not None:
            np.greater(native, plan.recipe.threshold_max, out=valid)
            np.logical_or(working, valid, out=working)
        if static is not None:
            np.logical_or(working, static, out=working)
        if detector_mask is not None:
            np.logical_or(working, detector_mask, out=working)
        np.isfinite(conditioned, out=valid)
        np.logical_not(valid, out=valid)
        np.logical_or(working, valid, out=working)
        np.logical_not(working, out=valid)
        with np.errstate(over='ignore', invalid='ignore'):
            np.add(sums, conditioned, out=sums, where=valid)
        working.fill(True)
        np.isfinite(sums, out=working, where=valid)
        _reject(not bool(working.all()), 'AVERAGE_PIXEL_SUM_OVERFLOW')
        np.add(counts, np.uint32(1), out=counts, where=valid)
        if needs_metadata:
            _accumulate_metadata(
                row, plan, configured, numeric, invariants, index == 0,
            )
        progress('average', index + 1, extent)
        del native, row
    del conditioned, working, valid
    zero = counts == 0
    _reject(bool(zero.all()), 'AVERAGE_ALL_PIXELS_INVALID')
    np.divide(sums, counts, out=sums, where=~zero)
    sums[zero] = np.nan
    output_metadata = {}
    denominators = []
    for key in plan.numeric_metadata_keys:
        total, compensation, denominator = numeric[key]
        value = math.nan if not denominator else (float(total) + float(compensation)) / int(denominator)
        _reject(denominator and (not math.isfinite(value)), 'AVERAGE_METADATA_SUM_OVERFLOW')
        output_metadata[key] = value
        denominators.append((key, int(denominator)))
    output_metadata.update(((key, invariants[key][1]) for key in plan.invariant_metadata_keys))
    metered_metadata = {key: None if type(value) is float and not math.isfinite(value) else value for key, value in output_metadata.items()}
    _reject(len(_canonical({'metadata': metered_metadata, 'denominators': denominators})) > _JSON_MAX_BYTES, 'AVERAGE_METADATA_STATE_TOO_LARGE')
    chunks = _average_count_chunks(shape)
    evidence = AverageFiniteCountsEvidence(_COUNT_POLICY, extent, shape, '<u4', _average_count_digest(counts, extent, rows=chunks[0]), int(counts.min()), int(counts.max()), int(np.count_nonzero(zero)), chunks, 'gzip', 1, True, False)
    finite = AverageFiniteCounts._from_owned_array(evidence, counts)
    return {'average': sums, 'zero': zero, 'metadata': output_metadata, 'denominators': tuple(denominators), 'finite': finite}
def _scan_calibration(recipe: AverageScanRecipe):
    values = recipe.calibration.values
    if values is None:
        return (None, object(), None)
    from xrd_tools.core import PONI
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    poni = PONI(values.dist, values.poni1, values.poni2, values.rot1, values.rot2, values.rot3, values.wavelength_m, recipe.calibration.detector_id)
    calibration = DetectorCalibration(poni, dict(recipe.calibration.detector_config))
    return (poni, detector_calibration_to_integrator(calibration), calibration)
def _reduce_average(plan: AverageScanPlan, science: dict[str, Any], token: threading.Event | None, progress: Callable[[str, int, int], None]):
    poni, integrator, calibration = _scan_calibration(plan.recipe)
    detector_values = None
    if calibration is not None:
        detector_values = dict(calibration.poni.to_dict())
        detector_values['detector_name'] = detector_values.pop('detector', '')
        config = calibration.detector_config
        detector_values['x_pixel_size'] = config.get('pixel2')
        detector_values['y_pixel_size'] = config.get('pixel1')
    frame = ScanFrame(1, image=science['average'], metadata=dict(science['metadata']), source_path=None, source_frame_index=0, background=None, mask=science['zero'])
    provenance = {**_science_payload(plan), 'science_identity': plan.science_identity, 'operation_identity': plan.operation_identity, 'source_graph_digest': plan.source_graph_digest, 'contributor_extent': plan.contributor_extent, 'metadata_denominators': science['denominators']}
    scan = Scan('average', [frame], poni=poni, integrator=integrator, output_path=plan.recipe.target, extra={'average_finite_counts': science['finite'], 'average_scan_provenance': provenance, 'detector_shape': plan.detector_shape, **({'detector_calibration': detector_values} if detector_values is not None else {})})
    progress('reduce', 0, 1)
    reduced = run_reduction(_derived_reduction(plan.recipe), scan, cancel_token=token, execution='chunked', retain_products=True, strict=StrictPolicy(True, True, True))
    progress('reduce', 1, 1)
    if reduced.failed and type(reduced.error) is str and reduced.error.startswith('monitor ') and reduced.error.endswith('Pass StrictPolicy.graceful() to allow it.'): raise MissingNormalizationError(reduced.error)
    _reject(reduced.failed or 1 not in reduced.frames, reduced.error or 'AVERAGE_REDUCTION_FAILED')
    return (scan, frame, reduced.frames[1], provenance)
def _append_intent(plan: AverageScanPlan, graph: PreparedSourceExecutionGraph) -> AppendIntent:
    modes = required_result_modes(_derived_reduction(plan.recipe))
    return AppendIntent(plan.recipe.entry, plan.recipe.source_base or '', science_fingerprint(stable_lineage_projection(graph, target=plan.recipe.target)), plan.science_identity, tuple((f'{item.kind}:{item.key}' for item in modes)), append_source_from_execution_graph(graph, generation=1), (1,))
def _average_sink(plan: AverageScanPlan, graph: PreparedSourceExecutionGraph):
    run_configuration = _recipe_payload(plan.recipe)
    if plan.recipe.background is None:
        run_configuration.pop('background')
    return NexusSink(plan.recipe.target, entry=plan.recipe.entry, overwrite=True, source_base=plan.recipe.source_base, run_configuration_provenance=run_configuration, source_execution_provenance=source_execution_projection(graph), source_snapshots_provenance=source_snapshots_projection(graph, writer=True), same_run_intent=_append_intent(plan, graph), rollback_until_commit=True)
def _resolved_lineage_path(stored: str, artifact: str | Path, source_base: str) -> str:
    value = resolve_source_master(stored, scan_file=artifact, source_base=source_base, allow_basename_fallbacks=True)
    _reject(value is None, 'Average lineage source is unavailable')
    return str(value.resolve(strict=True))
def iter_average_contributors(path, *, entry='entry', label=1) -> Iterator[AverageContributor]:
    _reject(type(entry) is not str or not entry or '/' in entry or (entry in {'.', '..'}), 'Average lineage entry is invalid')
    _reject(type(label) is not int or type(label) is bool or label != 1, 'Average lineage supports exact logical label 1')
    artifact = str(Path(path).resolve(strict=True))
    provenance = read_provenance(artifact, entry=entry)
    config = provenance.get('config') if type(provenance) is dict else None
    average = config.get(_COUNT_POLICY) if type(config) is dict else None
    counts = get_average_finite_counts(artifact, entry=entry)
    _reject(type(average) is not dict or average.get('api_version') != _COUNT_POLICY or average.get('contributor_extent') != counts.evidence.contributor_extent, 'Average lineage provenance or extent is invalid')
    source_base, source = _read_average_lineage(artifact, entry=entry)
    extent = counts.evidence.contributor_extent
    _reject(source.extent != extent, 'Average lineage source extent does not match counts')
    if source.image_members:
        for index, member in enumerate(source.image_members):
            yield AverageContributor(index, index, _resolved_lineage_path(member.path, artifact, source_base), 0, None, member.source_start, member.source_stop, member.size, member.mtime_ns)
        return
    if source.external_members:
        for member in source.external_members:
            resolved = _resolved_lineage_path(member.path, artifact, source_base)
            for index in range(member.source_start, member.source_stop):
                yield AverageContributor(index, index, resolved, index - member.source_start, member.dataset_path, member.source_start, member.source_stop, member.size, member.mtime_ns)
        return
    _reject(len(source.dataset_paths) != 1 or extent < 1, 'Average lineage singular source is malformed')
    resolved = _resolved_lineage_path(source.path, artifact, source_base)
    for index in range(extent):
        yield AverageContributor(index, index, resolved, index, source.dataset_paths[0], index, index + 1, source.size, source.mtime_ns)
class AverageScanRunner:
    def __init__(self, plan: AverageScanPlan) -> None:
        _reject(type(plan) is not AverageScanPlan, 'AverageScanRunner requires an exact plan', TypeError)
        self.plan = plan
        self._thread = threading.get_ident()
        self._state = 'NEW'
        self._ran = False
        self._result = None
        self._source_window = None
        self._source_error = None; self._science = self._reduction = None
        self._graph = None
        self._sink = None
        self._h23_target_snapshot = None
        self._cancel_token = None
        self._progress_cb = None
        self._publication_gate = None
        self._gate_called = False; self._revision = 0
    def _affine(self) -> None:
        _reject(threading.get_ident() != self._thread, 'AverageScanRunner is creating-thread affine', RuntimeError)
    def __enter__(self) -> 'AverageScanRunner':
        self._affine()
        _reject(self._state != 'NEW', 'AverageScanRunner can be entered only once', RuntimeError)
        self._state = 'ACTIVE'
        return self
    def _progress(self, stage: str, completed: int, total: int) -> None:
        self._revision += 1
        callback = self._progress_cb
        if callback is None:
            return
        try:
            callback(AverageScanProgress(self.plan.operation_identity, stage, completed, total, self._revision))
        except BaseException:
            pass
    def _terminal(self, value: AverageScanResult) -> AverageScanResult:
        self._result = value
        self._state = 'TERMINAL'
        self._science = self._reduction = None
        self._source_error = None
        return value
    def _pending(self, *, source: bool=False) -> AverageScanResult:
        self._state = 'SETTLEMENT_PENDING'
        raw = None if source or self._sink is None else self._sink._transaction.snapshot().phase.value
        phase = None if source else 'ready-to-retry' if raw in {None, 'executing'} else 'cleanup-pending' if raw in {'epoch-committed', 'committed'} else raw
        code = 'AVERAGE_SOURCE_CLEANUP_PENDING' if source else 'AVERAGE_H23_SETTLEMENT_PENDING'
        self._result = _result(self.plan, 'SETTLEMENT_PENDING', code=code, diagnostic=code, denominators=() if self._science is None else self._science['denominators'], h23_phase=phase)
        return self._result
    def _fresh_graph(self) -> PreparedSourceExecutionGraph:
        graph = _source_graph_owner.qualify_source_execution_graph(_source_from_recipe(self.plan.recipe), selected_motor=_selected_motor(self.plan.recipe), reader_binding=_READER_BINDING, cancelled=None if self._cancel_token is None else self._cancel_token.is_set)
        if source_graph_digest(graph) != self.plan.source_graph_digest or graph.stamp.frame_count != self.plan.contributor_extent or graph.detector_shape != self.plan.detector_shape or (graph.native_dtype != self.plan.native_dtype):
            raise SourceRevisionChanged('Average execution graph changed')
        _average_allocation(self.plan.recipe, self.plan.detector_shape, self.plan.allocation)
        return graph
    def run(self, *, cancel_token: threading.Event | None=None, progress_cb: Callable[[AverageScanProgress], object] | None=None, publication_gate: Callable[[], bool] | None=None) -> AverageScanResult:
        self._affine()
        _reject(self._state != 'ACTIVE' or self._ran, 'AverageScanRunner.run is one-shot and requires entry', RuntimeError)
        _reject(progress_cb is not None and (not callable(progress_cb)), 'Average progress callback must be callable', TypeError)
        _reject(publication_gate is not None and (not callable(publication_gate)), 'Average publication gate must be callable', TypeError)
        _cancelled(cancel_token)
        self._ran = True
        self._cancel_token = cancel_token
        self._progress_cb = progress_cb
        self._publication_gate = publication_gate
        error = None
        try:
            _poll(cancel_token)
            self._progress('qualify', 0, 1)
            self._graph = self._fresh_graph()
            self._progress('qualify', 1, 1)
            self._source_window = open_source_execution_graph(self._graph, cancelled=None if cancel_token is None else cancel_token.is_set)
            self._source_window.__enter__()
            self._science = _average_contributors(self.plan, self._graph, self._source_window, cancel_token, self._progress)
            self._reduction = _reduce_average(self.plan, self._science, cancel_token, self._progress)
        except BaseException as caught:
            error = (_AverageCancelled('Average operation cancelled') if isinstance(caught, InterruptedError) and _cancelled(cancel_token) else caught)
        if self._source_window is not None:
            try:
                self._source_window.close()
            except BaseException:
                self._source_error = error
                return self._pending(source=True)
            self._source_window = None
        if error is not None:
            return self._terminal(_error_result(self.plan, error))
        return self._after_source_clean()
    def _source_sweep(self) -> None:
        source = _source_for_requalification(
            _source_from_recipe(self.plan.recipe), self._graph)
        try: requalify_source_execution_graph(source, self._graph, selected_motor=_selected_motor(self.plan.recipe), reader_binding=_READER_BINDING, cancelled=None if self._cancel_token is None else self._cancel_token.is_set)
        except InterruptedError as error: raise (_AverageCancelled('Average operation cancelled') if _cancelled(self._cancel_token) else error)
        except SourceRevisionChanged as error: raise SourceRevisionChanged('AVERAGE_SOURCE_DRIFT') from error
    def _target_sweep(self, expected: TargetSnapshot) -> None:
        _reject(capture_target_snapshot(self.plan.recipe.target) != expected, 'AVERAGE_TARGET_DRIFT')
    def _final_sweep(self, expected: TargetSnapshot) -> None:
        self._source_sweep(); self._target_sweep(expected)
    def _abort(self, error: BaseException, *, disposition: str='ABORTED') -> AverageScanResult:
        if self._sink is not None:
            try:
                self._sink._scan = self._sink._plan = None
                self._sink._abort_composed()
            except BaseException:
                phase = self._sink._transaction.snapshot().phase.value
                if phase in {'ready-to-retry', 'rollback-pending', 'cleanup-pending', 'integrity-hold', 'executing'}:
                    return self._pending()
                raise
        value = _error_result(self.plan, error, h23=True, denominators=self._science['denominators'])
        if disposition == 'CANCELLED':
            value = _result(self.plan, 'CANCELLED', code='AVERAGE_CANCELLED' if isinstance(error, _AverageCancelled) else str(error), diagnostic=_diagnostic(error), denominators=self._science['denominators'])
        return self._terminal(value)
    def _gate(self) -> AverageScanResult | None:
        if _cancelled(self._cancel_token): return self._abort(_AverageCancelled('Average operation cancelled'), disposition='CANCELLED')
        if self._gate_called:
            return None
        self._gate_called = True
        if self._publication_gate is None:
            return None
        try:
            accepted = self._publication_gate()
        except BaseException as error:
            return self._abort(ValueError(f'AVERAGE_PUBLICATION_GATE_FAILED: {_diagnostic(error)}'))
        if accepted is not True:
            return self._abort(ValueError('AVERAGE_PUBLICATION_NOT_SEALED'), disposition='CANCELLED')
        return None
    def _commit(self) -> AverageScanResult:
        sink = self._sink
        try:
            snapshot = sink._transaction.commit_stream(sink._attempt, lease=sink._lease)
            sink._release_terminal_lease()
            terminal = sink._typed_terminal(snapshot)
            sink._scan = sink._plan = None
        except BaseException as error:
            phase = sink._transaction.snapshot().phase.value
            if phase in {'ready-to-retry', 'rollback-pending', 'cleanup-pending', 'integrity-hold', 'executing'}:
                return self._pending()
            return self._terminal(_error_result(self.plan, error, h23=True, denominators=self._science['denominators']))
        return self._terminal(_result(self.plan, 'COMMITTED', denominators=self._science['denominators'], evidence=self._science['finite'].evidence, commit=terminal.commit_identity))
    def _writer_pending(self) -> AverageScanResult:
        try: self._h23_target_snapshot = capture_target_snapshot(self.plan.recipe.target)
        except BaseException as error: return self._abort(error)
        return self._pending()
    def _after_source_clean(self) -> AverageScanResult:
        try:
            _poll(self._cancel_token)
            scan, frame, reduced, _provenance = self._reduction
            self._final_sweep(self.plan.expected_target_snapshot)
            self._sink = _average_sink(self.plan, self._graph)
            self._sink.begin(scan, _derived_reduction(self.plan.recipe))
            self._progress('write', 0, 1)
            self._sink.write(frame, reduced)
            self._sink._apply_pending_extension()
            self._sink._drain_pending_record_writes(force=True)
            writer = self._sink._writer
            writer.finish(self._sink._writer_finalization(writer)); self._h23_target_snapshot = capture_target_snapshot(self.plan.recipe.target)
            self._progress('write', 1, 1)
        except WriterIncomplete: return self._writer_pending()
        except BaseException as error:
            if self._sink is not None:
                return self._abort(error)
            return self._terminal(_error_result(self.plan, error))
        try:
            self._final_sweep(self._h23_target_snapshot)
        except BaseException as error:
            return self._abort(error)
        gated = self._gate()
        if gated is not None:
            return gated
        return self._commit()
    def finish_current(self) -> AverageScanResult:
        self._affine()
        if self._state in {'TERMINAL', 'CLOSED'} and self._result is not None:
            return self._result
        _reject(self._state != 'SETTLEMENT_PENDING', 'AverageScanRunner has no pending settlement', RuntimeError)
        if self._source_window is not None:
            time.sleep(0.05)
            try:
                self._source_window.close()
            except BaseException:
                return self._pending(source=True)
            self._source_window = None
            if self._source_error is not None:
                error, self._source_error = (self._source_error, None)
                return self._terminal(_error_result(self.plan, error))
            self._state = 'ACTIVE'
            return self._after_source_clean()
        sink = self._sink
        phase = sink._transaction.snapshot().phase.value
        if phase in {'cleanup-pending', 'epoch-committed', 'committed'}:
            try:
                terminal = sink.finish(SimpleNamespace(cancelled=False))
            except BaseException:
                return self._pending()
            return self._terminal(_result(self.plan, 'COMMITTED', denominators=self._science['denominators'], evidence=self._science['finite'].evidence, commit=terminal.commit_identity))
        try:
            self._final_sweep(self._h23_target_snapshot)
            sink._writer.finish()
        except WriterIncomplete: return self._writer_pending()
        except BaseException as error:
            return self._abort(error)
        try:
            self._h23_target_snapshot = capture_target_snapshot(self.plan.recipe.target)
            self._final_sweep(self._h23_target_snapshot)
        except BaseException as error: return self._abort(error)
        gated = self._gate()
        if gated is not None:
            return gated
        return self._commit()
    def close(self) -> AverageScanResult | None:
        self._affine()
        if self._state == 'CLOSED':
            return self._result
        while self._state == 'SETTLEMENT_PENDING':
            self.finish_current()
        self._state = 'CLOSED'
        return self._result
    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False
def run_average_scan(recipe: AverageScanRecipe, *, cancel_token: threading.Event | None=None, progress_cb: Callable[[AverageScanProgress], object] | None=None, publication_gate: Callable[[], bool] | None=None) -> AverageScanResult:
    _reject(type(recipe) is not AverageScanRecipe, 'run_average_scan requires an exact recipe', TypeError)
    for condition, code in ((recipe.background is not None, 'AVERAGE_BACKGROUND_AGGREGATE_PROVENANCE_UNSUPPORTED'), (recipe.live_mode, 'AVERAGE_LIVE_TERMINALITY_UNSUPPORTED'), (recipe.output_mode != 'Overwrite', 'AVERAGE_APPEND_UNSUPPORTED'), (recipe.save_xye, 'AVERAGE_NEXUS_REQUIRED')):
        if condition:
            return _recipe_refusal(recipe, code)
    try:
        plan = prepare_average_scan(recipe, cancel_token=cancel_token)
    except InterruptedError:
        return AverageScanResult('CANCELLED', recipe.target, recipe.entry, '', '', 0, (1,), (), (), None, 'AVERAGE_CANCELLED', 'Average preparation cancelled', None, None)
    except BaseException as error:
        code = _diagnostic(error).split('(', 1)[0]
        return _recipe_refusal(recipe, code if code.startswith('AVERAGE_') else 'AVERAGE_PREPARATION_FAILED')
    with AverageScanRunner(plan) as runner:
        result = runner.run(cancel_token=cancel_token, progress_cb=progress_cb, publication_gate=publication_gate)
        while result.disposition == 'SETTLEMENT_PENDING':
            result = runner.finish_current()
        return result
