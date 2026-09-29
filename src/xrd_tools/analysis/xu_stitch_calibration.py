"""Engine-light custody for the canonical xrayutilities Stitch calibration.

This module deliberately imports neither NumPy, pyFAI, nor xrayutilities.  It
admits the small, canonical JSON authority before any scientific runtime is
loaded and retains exact lexical and physical file identity for later fences.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
from importlib import resources
import json
import math
import os
from pathlib import Path
import stat
from collections.abc import Mapping
from types import MappingProxyType

from xrd_tools.analysis.scan_operations import analysis_canonical_fingerprint
from xrd_tools.io import descriptor_path
from xrd_tools.io.descriptor_path import (
    chain_components,
    chain_drift,
    directory_identity,
)
from xrd_tools.io.stat_identity import identity_ctime_ns


_MAX_ASSET_BYTES = 65_536
_MAX_DEPTH = 6
_MAX_NODES = 256
_MAX_STRING_LENGTH = 256
_RESOURCE_PARTS = ("assets", "xu", "psic_powder_1d_surface_v1.json")
_RESOURCE_BYTE_COUNT = 4_837
_RESOURCE_SHA256 = (
    "57857833c56eeed0db27ec9e3f64aa635d5cd1d1e74eaed0b957eb054b3356d3"
)
_RESOURCE_SEMANTIC_FINGERPRINT = (
    "90f3bec535ed9a21be1b9d93491e774364df5849155b9b9c1707eace848e40f8"
)
CANONICAL_XU_STITCH_CALIBRATION_LOCATOR = (
    "calibration/xu/psic_powder_1d_surface_v1.json"
)
_LEGACY_ORACLE_SHA256 = (
    "9bb13babeb60475128dd9d14c833cdd8cd85af7510424c1b860b87231d8c0c25"
)
_TOP_LEVEL_KEYS = {
    "schema",
    "version",
    "preset",
    "xrayutilities",
    "detector",
    "acquisition",
    "corrections",
    "validation",
}
_PROJECTION_FACTORY = object()
_RECEIPT_FACTORY = object()


class XuStitchCalibrationRefused(ValueError):
    def __init__(self, code: str, message: str | None = None):
        if type(code) is not str or not code:
            raise TypeError("XU calibration refusal code must be nonempty")
        self.code = code
        super().__init__(message or code)


def _refuse(code: str, message: str) -> None:
    raise XuStitchCalibrationRefused(code, message)


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _bounded_projection(value: object, *, depth: int, budget: list[int]) -> None:
    if depth > _MAX_DEPTH:
        _refuse("XU_CALIBRATION_PARSE_FAILED", "calibration exceeds depth 6")
    if type(value) is str:
        if len(value) > _MAX_STRING_LENGTH:
            _refuse(
                "XU_CALIBRATION_PARSE_FAILED",
                "calibration string exceeds 256 Unicode scalars",
            )
        return
    if value is None or type(value) is bool or type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _refuse(
                "XU_CALIBRATION_PARSE_FAILED",
                "calibration contains a nonfinite number",
            )
        return
    if type(value) is dict:
        budget[0] += len(value)
        if budget[0] > _MAX_NODES:
            _refuse("XU_CALIBRATION_PARSE_FAILED", "calibration is oversized")
        for key, item in value.items():
            if type(key) is not str or len(key) > _MAX_STRING_LENGTH:
                _refuse(
                    "XU_CALIBRATION_PARSE_FAILED",
                    "calibration object key is invalid",
                )
            _bounded_projection(item, depth=depth + 1, budget=budget)
        return
    if type(value) is list:
        budget[0] += len(value)
        if budget[0] > _MAX_NODES:
            _refuse("XU_CALIBRATION_PARSE_FAILED", "calibration is oversized")
        for item in value:
            _bounded_projection(item, depth=depth + 1, budget=budget)
        return
    _refuse(
        "XU_CALIBRATION_PARSE_FAILED",
        f"calibration contains unsupported {type(value).__name__}",
    )


def _freeze_projection(value: object) -> object:
    """Return a recursively immutable projection with no retained mutable root."""

    if type(value) is dict:
        frozen = {
            key: _freeze_projection(item)
            for key, item in value.items()
        }
        return MappingProxyType(frozen)
    if type(value) is list:
        return tuple(_freeze_projection(item) for item in value)
    return value


@dataclass(eq=False, frozen=True, slots=True)
class XuStitchCalibrationProjection:
    canonical_json: str
    raw_sha256: str
    semantic_fingerprint: str
    _value: Mapping[str, object] = field(repr=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _PROJECTION_FACTORY
            or type(self.canonical_json) is not str
            or self.raw_sha256 != hashlib.sha256(self.canonical_json.encode()).hexdigest()
            or self.semantic_fingerprint != analysis_canonical_fingerprint(
                "xu-stitch-calibration-asset-v1", json.loads(self.canonical_json))
            or not isinstance(self._value, MappingProxyType)
        ):
            raise TypeError("XU calibration projection is not factory-owned")

    @property
    def value(self) -> Mapping[str, object]:
        return self._value

    @property
    def content(self) -> bytes:
        return self.canonical_json.encode("utf-8")

    def __copy__(self):
        raise TypeError("XU calibration projection is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("XU calibration projection is not copyable")

    def __reduce__(self):
        raise TypeError("XU calibration projection is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("XU calibration projection is not serializable")


def parse_xu_stitch_calibration_bytes(
    raw: bytes,
) -> XuStitchCalibrationProjection:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_ASSET_BYTES:
        _refuse(
            "XU_CALIBRATION_PARSE_FAILED",
            "calibration must be nonempty bounded exact bytes",
        )
    if hashlib.sha256(raw).hexdigest() == _LEGACY_ORACLE_SHA256:
        _refuse(
            "XU_CALIBRATION_MIGRATION_REQUIRED",
            "the exact historical XU geometry requires reviewed migration",
        )
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw or raw.endswith(b"\n"):
        _refuse(
            "XU_CALIBRATION_PARSE_FAILED",
            "calibration contains a BOM, NUL, or trailing newline",
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_PARSE_FAILED",
            "calibration is not strict UTF-8",
        ) from error

    def object_pairs(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            _refuse(
                "XU_CALIBRATION_PARSE_FAILED",
                "calibration contains duplicate object keys",
            )
        return result

    def invalid_constant(_value):
        _refuse(
            "XU_CALIBRATION_PARSE_FAILED",
            "calibration contains a nonfinite JSON constant",
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except XuStitchCalibrationRefused:
        raise
    except (RecursionError, TypeError, ValueError) as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_PARSE_FAILED",
            "calibration is not one exact JSON object",
        ) from error
    if type(value) is not dict or set(value) != _TOP_LEVEL_KEYS:
        _refuse(
            "XU_CALIBRATION_SCHEMA_UNSUPPORTED",
            "calibration top-level schema is unsupported",
        )
    _bounded_projection(value, depth=1, budget=[0])
    try:
        canonical = _canonical_bytes(value)
    except (TypeError, ValueError, UnicodeError) as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_PARSE_FAILED",
            "calibration cannot be encoded canonically",
        ) from error
    if canonical != raw:
        _refuse(
            "XU_CALIBRATION_NOT_CANONICAL",
            "calibration bytes are not exact canonical JSON",
        )
    semantic = analysis_canonical_fingerprint(
        "xu-stitch-calibration-asset-v1", value
    )
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if value.get("version") == 2 and type(value.get("version")) is int:
        _validate_authored_surface(value)
    elif value.get("version") == 3 and type(value.get("version")) is int:
        _validate_extended_surface(value)
    elif (
        len(raw) != _RESOURCE_BYTE_COUNT
        or raw_sha256 != _RESOURCE_SHA256
        or semantic != _RESOURCE_SEMANTIC_FINGERPRINT
        or value.get("schema") != "xdart.xu_stitch_calibration"
        or type(value.get("version")) is not int
        or value.get("version") != 1
        or value.get("preset") != "psic_powder_1d"
    ):
        _refuse(
            "XU_CALIBRATION_SCHEMA_UNSUPPORTED",
            "calibration is not the authenticated SURFACE v1 projection",
        )
    return XuStitchCalibrationProjection(
        text,
        raw_sha256,
        semantic,
        _freeze_projection(value),
        _PROJECTION_FACTORY,
    )


def _validate_authored_surface(value):
    """Version 2 permits fitted values under the same fixed SURFACE model.

    V1 still requires its exact authenticated bytes. Authored calibrations
    have a distinct version, strict key sets and actual fit diagnostics.
    """
    reference = json.loads(canonical_surface_resource_bytes())
    def require(ok, message):
        if not ok:
            _refuse("XU_CALIBRATION_SCHEMA_UNSUPPORTED", message)
    def number(v):
        return type(v) in {int, float} and math.isfinite(v)
    require(value['schema'] == reference['schema'] and value['preset'] == reference['preset'],
            'unsupported authored calibration model')
    require(value['xrayutilities'] == reference['xrayutilities'], 'SURFACE circle/camera/runtime conventions must match')
    for section in ('detector', 'acquisition', 'corrections'):
        require(type(value[section]) is dict and set(value[section]) == set(reference[section]),
                f'authored {section} keys differ')
    detector, acquisition, correction, validation = (value[k] for k in ('detector', 'acquisition', 'corrections', 'validation'))
    for key in ('mask_policy', 'sensor_policy', 'sensor_material', 'sensor_thickness_m', 'parallax', 'pyfai_validated_version'):
        require(detector[key] == reference['detector'][key], f'unsupported detector {key}')
    shape = detector['shape']
    require(type(shape) is list and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape)
            and math.prod(shape) <= 1 << 25, 'invalid detector shape')
    for key in ('distance_m', 'pixel1_m', 'pixel2_m'):
        require(number(detector[key]) and detector[key] > 0, f'invalid detector {key}')
    require(0 < detector['distance_m'] < 100, 'implausible detector distance')
    for key, size in zip(('cch1_px', 'cch2_px'), shape):
        require(number(detector[key]) and -size <= detector[key] <= 2 * size, 'implausible detector centre')
    require(type(detector['detector_factory_name']) is str and bool(detector['detector_factory_name']), 'detector name required')
    cfg = detector['detector_config']
    require(type(cfg) is dict and set(cfg) <= {'orientation', 'pixel1', 'pixel2', 'max_shape'}, 'unsupported authored detector configuration')
    require(cfg.get('orientation') in (1, 2, 3, 4) and cfg.get('pixel1') == detector['pixel1_m']
            and cfg.get('pixel2') == detector['pixel2_m'], 'detector configuration disagrees with pitch/orientation')
    require('max_shape' not in cfg or cfg['max_shape'] == shape, 'detector max_shape differs')
    require(type(detector['mask_count']) is int and 0 <= detector['mask_count'] < math.prod(shape), 'invalid intrinsic mask count')
    for digest in (detector['mask_sha256'], correction['solid_angle_sha256']):
        require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'invalid correction digest')
    mutable = {'detector_offsets_deg', 'energy_eV', 'image_rotation_quarter_turns', 'source_motors',
               'xdart_wavelength_A', 'xu_mapping_wavelength_A'}
    require(all(acquisition[k] == v for k, v in reference['acquisition'].items() if k not in mutable), 'unsupported acquisition policy')
    require(type(acquisition['image_rotation_quarter_turns']) is int and acquisition['image_rotation_quarter_turns'] in range(4), 'invalid image rotation')
    for key in ('source_motors', 'detector_offsets_deg'):
        require(type(acquisition[key]) is dict and set(acquisition[key]) == {'del', 'nu'}, 'two detector motors required')
    motors = acquisition['source_motors']
    # a single 2θ arm (fourc, twoc) names no motor (null) for the circle it holds at zero:
    # nu for a vertical arm, del for a horizontal one
    held = [circle for circle, motor in motors.items() if motor is None]
    require(len(held) <= 1 and all(type(motor) is str and motor for motor in motors.values() if motor is not None)
            and len({motor for motor in motors.values() if motor is not None}) == 2 - len(held),
            'distinct source motors required')
    require(all(number(v) and abs(v) <= 5 for v in acquisition['detector_offsets_deg'].values()), 'motor offsets exceed fit bounds')
    require(number(acquisition['energy_eV']) and acquisition['energy_eV'] > 0, 'positive energy required')
    expected = 1.239841984e4 / acquisition['energy_eV']
    require(number(acquisition['xdart_wavelength_A']) and math.isclose(acquisition['xdart_wavelength_A'], expected, rel_tol=1e-12), 'energy and wavelength disagree')
    require(number(acquisition['xu_mapping_wavelength_A']) and math.isclose(acquisition['xu_mapping_wavelength_A'], expected, rel_tol=1e-6), 'XU energy and wavelength disagree')
    require(all(correction[k] == v for k, v in reference['corrections'].items()
                if k not in {'solid_angle_policy', 'solid_angle_sha256'}), 'unsupported correction policy')
    require(correction['solid_angle_policy'] == 'calibrated_pyfai_detector_header_v1', 'authored solid-angle policy required')
    keys = {'authoring_policy', 'control_domain_source_deg', 'validated_scan_domain_source_deg',
            'geometry_control_frame_count', 'geometry_control_point_count', 'geometry_rms_q_A^-1', 'condition_number'}
    require(type(validation) is dict and set(validation) == keys, 'authored diagnostics required')
    require(validation['authoring_policy'] == 'surface_control_points_bounded_fit_v1', 'unsupported authoring policy')
    require(type(validation['geometry_control_frame_count']) is int and 3 <= validation['geometry_control_frame_count'] <= 32
            and type(validation['geometry_control_point_count']) is int and validation['geometry_control_point_count'] > 5, 'insufficient control points')
    require(number(validation['geometry_rms_q_A^-1']) and 0 <= validation['geometry_rms_q_A^-1'] <= .03
            and number(validation['condition_number']) and 1 <= validation['condition_number'] <= 1e8, 'unusable calibration fit')
    for name in ('control_domain_source_deg', 'validated_scan_domain_source_deg'):
        domain = validation[name]
        require(type(domain) is dict and set(domain) == {'del', 'nu'}, 'invalid calibration domain')
        for limits in domain.values():
            require(type(limits) is list and len(limits) == 2 and all(number(v) for v in limits)
                    and -360 <= limits[0] <= limits[1] <= 360, 'invalid angular limits')
    require(all(validation['validated_scan_domain_source_deg'][motor][0] <= limits[0] <= limits[1]
                <= validation['validated_scan_domain_source_deg'][motor][1]
                for motor, limits in validation['control_domain_source_deg'].items()), 'run domain must enclose control coverage')
    require(all(validation[name][circle] == [0, 0] for circle in held for name in (
        'control_domain_source_deg', 'validated_scan_domain_source_deg')),
        f'a single 2θ arm holds {" and ".join(held) or "a circle"} at zero')


#: Version 3: the SURFACE model plus xrayutilities' detector tilt/rotation and sample
#: displacement, transferred from a pyFAI extended arm, with that arm's per-frame solid angle.
EXTENDED_SOLID_ANGLE_POLICY = "per_frame_pyfai_extended_arm_v1"
EXTENDED_AUTHORING_POLICY = "pyfai_extended_arm_transfer_v1"
EXTENDED_TRANSFER_GATE_A = 1e-8
_EXTENDED_DETECTOR_KEYS = frozenset({"detrot_deg", "tilt_deg", "tiltazimuth_deg"})
_EXTENDED_ARM_KEYS = frozenset({
    "marker", "detector", "detector_config", "wavelength_m", "dist_m", "poni1_m", "poni2_m",
    "nu_offset", "nu_scale", "del_offset", "del_scale", "beam_rotation", "mount_tilt",
    "mount_roll", "sample_offset1", "sample_offset2", "sample_offset3", "source_motors",
})
_EXTENDED_VALIDATION_KEYS = frozenset({
    "authoring_policy", "control_domain_source_deg", "validated_scan_domain_source_deg",
    "transfer_frame_count", "validation_frame_count", "validation_pixel_count",
    "transfer_max_abs_dq_A^-1", "transfer_gate_A^-1",
})


def extended_arm_digest(arm) -> str:
    """The attested identity of a v3 asset's pyFAI arm (canonical JSON SHA-256)."""
    return hashlib.sha256(_canonical_bytes(arm)).hexdigest()


def _validate_extended_surface(value):
    """Version 3: v2's fixed SURFACE conventions plus the transferred extension.

    The asset carries the pyFAI extended arm it was transferred from (a compact
    parameter record) and the transfer's exactness gate over its validation frames.
    """
    reference = json.loads(canonical_surface_resource_bytes())
    def require(ok, message):
        if not ok:
            _refuse("XU_CALIBRATION_SCHEMA_UNSUPPORTED", message)
    def number(v):
        return type(v) in {int, float} and math.isfinite(v)
    require(value['schema'] == reference['schema'] and value['preset'] == reference['preset'],
            'unsupported extended calibration model')
    require(value['xrayutilities'] == reference['xrayutilities'], 'SURFACE circle/camera/runtime conventions must match')
    detector, acquisition, correction, validation = (value[k] for k in ('detector', 'acquisition', 'corrections', 'validation'))
    require(type(detector) is dict and set(detector) == set(reference['detector']) | _EXTENDED_DETECTOR_KEYS,
            'extended detector keys differ')
    require(type(acquisition) is dict and set(acquisition) == set(reference['acquisition']) | {'sampledis_mm'},
            'extended acquisition keys differ')
    require(type(correction) is dict and set(correction) == set(reference['corrections']) | {'solid_angle_arm'},
            'extended correction keys differ')
    for key in ('mask_policy', 'sensor_policy', 'sensor_material', 'sensor_thickness_m', 'parallax', 'pyfai_validated_version'):
        require(detector[key] == reference['detector'][key], f'unsupported detector {key}')
    shape = detector['shape']
    require(type(shape) is list and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape)
            and math.prod(shape) <= 1 << 25, 'invalid detector shape')
    for key in ('distance_m', 'pixel1_m', 'pixel2_m'):
        require(number(detector[key]) and detector[key] > 0, f'invalid detector {key}')
    require(0 < detector['distance_m'] < 100, 'implausible detector distance')
    # A tilted detector moves XU's zero-angle centre pixel well off the panel.
    for key, size in zip(('cch1_px', 'cch2_px'), shape):
        require(number(detector[key]) and abs(detector[key]) <= 10 * size, 'implausible detector centre')
    require(number(detector['detrot_deg']) and abs(detector['detrot_deg']) <= 10, 'implausible detector rotation')
    require(number(detector['tilt_deg']) and 0 <= detector['tilt_deg'] <= 15, 'implausible detector tilt')
    require(number(detector['tiltazimuth_deg']) and abs(detector['tiltazimuth_deg']) <= 720, 'invalid tilt azimuth')
    require(type(detector['detector_factory_name']) is str and bool(detector['detector_factory_name']), 'detector name required')
    cfg = detector['detector_config']
    require(type(cfg) is dict and set(cfg) <= {'orientation', 'pixel1', 'pixel2', 'max_shape'}, 'unsupported detector configuration')
    require(cfg.get('orientation') in (1, 2, 3, 4) and cfg.get('pixel1') == detector['pixel1_m']
            and cfg.get('pixel2') == detector['pixel2_m'], 'detector configuration disagrees with pitch/orientation')
    require('max_shape' not in cfg or cfg['max_shape'] == shape, 'detector max_shape differs')
    require(type(detector['mask_count']) is int and 0 <= detector['mask_count'] < math.prod(shape), 'invalid intrinsic mask count')
    for digest in (detector['mask_sha256'], correction['solid_angle_sha256']):
        require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'invalid correction digest')
    mutable = {'detector_offsets_deg', 'energy_eV', 'image_rotation_quarter_turns', 'source_motors',
               'xdart_wavelength_A', 'xu_mapping_wavelength_A'}
    require(all(acquisition[k] == v for k, v in reference['acquisition'].items() if k not in mutable), 'unsupported acquisition policy')
    require(type(acquisition['image_rotation_quarter_turns']) is int and acquisition['image_rotation_quarter_turns'] in range(4), 'invalid image rotation')
    for key in ('source_motors', 'detector_offsets_deg'):
        require(type(acquisition[key]) is dict and set(acquisition[key]) == {'del', 'nu'}, 'two detector motors required')
    motors = acquisition['source_motors']
    held = [circle for circle, motor in motors.items() if motor is None]  # a single 2θ arm, as in v2
    require(len(held) <= 1 and all(type(motor) is str and motor for motor in motors.values() if motor is not None)
            and len({motor for motor in motors.values() if motor is not None}) == 2 - len(held),
            'distinct source motors required')
    require(all(number(v) and abs(v) <= 10 for v in acquisition['detector_offsets_deg'].values()), 'motor offsets exceed fit bounds')
    sampledis = acquisition['sampledis_mm']
    require(type(sampledis) is list and len(sampledis) == 3 and all(number(v) and abs(v) <= 20 for v in sampledis),
            'sample displacement must be three values within 20 mm')
    require(number(acquisition['energy_eV']) and acquisition['energy_eV'] > 0, 'positive energy required')
    expected = 1.239841984e4 / acquisition['energy_eV']
    require(number(acquisition['xdart_wavelength_A']) and math.isclose(acquisition['xdart_wavelength_A'], expected, rel_tol=1e-12), 'energy and wavelength disagree')
    require(number(acquisition['xu_mapping_wavelength_A']) and math.isclose(acquisition['xu_mapping_wavelength_A'], expected, rel_tol=1e-6), 'XU energy and wavelength disagree')
    require(all(correction[k] == v for k, v in reference['corrections'].items()
                if k not in {'solid_angle_policy', 'solid_angle_sha256'}), 'unsupported correction policy')
    require(correction['solid_angle_policy'] == EXTENDED_SOLID_ANGLE_POLICY, 'per-frame extended solid-angle policy required')
    arm = correction['solid_angle_arm']
    require(type(arm) is dict and set(arm) == _EXTENDED_ARM_KEYS and arm['marker'] == 'xdart_stacked_arm_v2',
            'the solid-angle arm must be a compact extended stacked arm')
    require(all(number(arm[k]) for k in _EXTENDED_ARM_KEYS - {'marker', 'detector', 'detector_config', 'source_motors'}),
            'the solid-angle arm has a non-numeric parameter')
    require(arm['nu_scale'] == arm['del_scale'] == math.pi / 180 and arm['beam_rotation'] == 0,
            'the transfer supports degree motor scales and no beam rotation')
    require(arm['detector'] == detector['detector_factory_name'] and arm['detector_config'] == cfg,
            'the solid-angle arm and the XU detector differ')
    require(math.isclose(arm['wavelength_m'] * 1e10, acquisition['xdart_wavelength_A'], rel_tol=1e-9),
            'the solid-angle arm and the XU energy differ')
    require(type(arm['source_motors']) is dict and arm['source_motors'] == {
        'del_angle': acquisition['source_motors']['del'], 'nu_angle': acquisition['source_motors']['nu']},
        'the solid-angle arm and the XU source motors differ')
    require(correction['solid_angle_sha256'] == extended_arm_digest(arm), 'solid-angle arm digest differs')
    require(type(validation) is dict and set(validation) == _EXTENDED_VALIDATION_KEYS, 'extended diagnostics required')
    require(validation['authoring_policy'] == EXTENDED_AUTHORING_POLICY, 'unsupported authoring policy')
    # a five-parameter or single arm transfers by formula, fitted on no frame
    for key, low in (('transfer_frame_count', 0), ('validation_frame_count', 3), ('validation_pixel_count', 1)):
        require(type(validation[key]) is int and validation[key] >= low, f'insufficient {key}')
    gate = validation['transfer_gate_A^-1']
    require(number(gate) and 0 < gate <= EXTENDED_TRANSFER_GATE_A, 'transfer gate exceeds 1e-8 A^-1')
    require(number(validation['transfer_max_abs_dq_A^-1']) and 0 <= validation['transfer_max_abs_dq_A^-1'] <= gate,
            'the transfer did not pass its exactness gate')
    for name in ('control_domain_source_deg', 'validated_scan_domain_source_deg'):
        domain = validation[name]
        require(type(domain) is dict and set(domain) == {'del', 'nu'}, 'invalid calibration domain')
        for limits in domain.values():
            require(type(limits) is list and len(limits) == 2 and all(number(v) for v in limits)
                    and -360 <= limits[0] <= limits[1] <= 360, 'invalid angular limits')
    require(all(validation['validated_scan_domain_source_deg'][motor][0] <= limits[0] <= limits[1]
                <= validation['validated_scan_domain_source_deg'][motor][1]
                for motor, limits in validation['control_domain_source_deg'].items()), 'run domain must enclose control coverage')
    require(all(validation[name][circle] == [0, 0] for circle in held for name in (
        'control_domain_source_deg', 'validated_scan_domain_source_deg')),
        f'a single 2θ arm holds {" and ".join(held) or "a circle"} at zero')


def extended_surface_calibration(*, detector, xu_parameters, arm, energy_eV, xu_wavelength_A,
                                 source_motors, rotation, control_domain, domain, fit_frames,
                                 validation_frames, validation_pixels, max_dq):
    """Build and strictly admit a transferred SURFACE v3 asset (imports are lazy).

    ``xu_parameters`` holds cch1/cch2 (px), distance_m, del/nu offsets (deg),
    detrot/tilt/tiltazimuth (deg) and sampledis_mm; ``arm`` is the compact pyFAI arm.
    """
    import numpy as np
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    value = json.loads(canonical_surface_resource_bytes())
    value['version'] = 3
    shape = tuple(int(v) for v in detector.shape)
    config = dict(detector.get_config())
    config['orientation'] = int(config.get('orientation', 3))
    config = json.loads(json.dumps(config))
    mask = detector.calc_mask()
    mask = np.zeros(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    p = xu_parameters
    value['detector'].update(cch1_px=p['cch1_px'], cch2_px=p['cch2_px'], distance_m=p['distance_m'],
        detrot_deg=p['detrot_deg'], tilt_deg=p['tilt_deg'], tiltazimuth_deg=p['tiltazimuth_deg'],
        detector_factory_name=detector.__class__.__name__, detector_config=config,
        shape=list(shape), pixel1_m=float(detector.pixel1), pixel2_m=float(detector.pixel2),
        mask_count=int(mask.sum()),
        mask_sha256=hashlib.sha256(np.ascontiguousarray(mask, dtype='|u1').tobytes()).hexdigest())
    value['acquisition'].update(energy_eV=energy_eV,
        xdart_wavelength_A=energy_eV_to_wavelength_m(energy_eV) * 1e10,
        xu_mapping_wavelength_A=xu_wavelength_A,
        image_rotation_quarter_turns=rotation // 90, source_motors=source_motors,
        detector_offsets_deg={'del': p['del_offset_deg'], 'nu': p['nu_offset_deg']},
        sampledis_mm=[float(v) for v in p['sampledis_mm']])
    arm = json.loads(json.dumps(arm))
    arm['detector_config'] = config
    value['corrections'].update(solid_angle_policy=EXTENDED_SOLID_ANGLE_POLICY,
                                solid_angle_sha256=extended_arm_digest(arm), solid_angle_arm=arm)
    value['validation'] = {
        'authoring_policy': EXTENDED_AUTHORING_POLICY, 'control_domain_source_deg': control_domain,
        'validated_scan_domain_source_deg': domain, 'transfer_frame_count': fit_frames,
        'validation_frame_count': validation_frames, 'validation_pixel_count': validation_pixels,
        'transfer_max_abs_dq_A^-1': max_dq, 'transfer_gate_A^-1': EXTENDED_TRANSFER_GATE_A}
    parse_xu_stitch_calibration_bytes(_canonical_bytes(value))
    return value


def authored_surface_calibration(*, detector, cch1, cch2, distance_m,
        del_offset_deg, nu_offset_deg, energy_eV, xu_wavelength_A, source_motors,
        rotation, control_domain, domain, rms_q, condition, n_frames, n_points):
    """Build and strictly admit a fitted SURFACE v2 asset (imports are lazy)."""
    import numpy as np
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    from xrd_tools.core.geometry import DetectorHeader
    from xrd_tools.corrections.stack import CorrectionStack
    from xrd_tools.rsm.corrections import detector_header_to_ai
    value = json.loads(canonical_surface_resource_bytes())
    value['version'] = 2
    shape = tuple(int(v) for v in detector.shape)
    config = dict(detector.get_config())
    config['orientation'] = int(config.get('orientation', 3))
    config = json.loads(json.dumps(config))
    mask = detector.calc_mask()
    mask = np.zeros(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    digest = lambda array, dtype: hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()
    value['detector'].update(cch1_px=cch1, cch2_px=cch2, distance_m=distance_m,
        detector_factory_name=detector.__class__.__name__, detector_config=config,
        shape=list(shape), pixel1_m=float(detector.pixel1), pixel2_m=float(detector.pixel2),
        mask_count=int(mask.sum()), mask_sha256=digest(mask, '|u1'))
    value['acquisition'].update(energy_eV=energy_eV,
        xdart_wavelength_A=energy_eV_to_wavelength_m(energy_eV) * 1e10,
        xu_mapping_wavelength_A=xu_wavelength_A,
        image_rotation_quarter_turns=rotation // 90, source_motors=source_motors,
        detector_offsets_deg={'del': del_offset_deg, 'nu': nu_offset_deg})
    header = DetectorHeader(cch1=cch1, cch2=cch2, pwidth1=float(detector.pixel1) * 1e3,
        pwidth2=float(detector.pixel2) * 1e3, distance=distance_m * 1e3, Nch1=shape[0], Nch2=shape[1])
    solid = CorrectionStack().normalization(detector_header_to_ai(header), shape)
    value['corrections'].update(solid_angle_policy='calibrated_pyfai_detector_header_v1',
                                solid_angle_sha256=digest(solid, '<f8'))
    value['validation'] = dict(authoring_policy='surface_control_points_bounded_fit_v1',
        control_domain_source_deg=control_domain, validated_scan_domain_source_deg=domain,
        geometry_control_frame_count=n_frames, geometry_control_point_count=n_points,
        condition_number=condition, **{'geometry_rms_q_A^-1': rms_q})
    parse_xu_stitch_calibration_bytes(_canonical_bytes(value))
    return value


def canonical_surface_resource_bytes() -> bytes:
    try:
        package_node = resources.files("xrd_tools")
        package_shown = os.fspath(package_node)
        if (
            type(package_shown) is not str
            or not package_shown
            or "\x00" in package_shown
        ):
            raise OSError("canonical package root is not a filesystem path")
        package_shown.encode("utf-8", errors="strict")
        package_parts = Path(package_shown).parts
        if (
            not os.path.isabs(package_shown)
            or not _exact_root(package_parts)
            or any(
                part in {"", os.curdir, os.pardir}
                for part in package_parts[1:]
            )
            or os.path.normpath(package_shown) != package_shown
        ):
            raise OSError("canonical package root is not an exact absolute path")
        package = package_shown
        relative = os.path.join(*_RESOURCE_PARTS)
        descriptor, opened_chain = _open_no_follow_chain(package, relative)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise OSError("canonical resource is not a regular file")
            raw = os.read(descriptor, _MAX_ASSET_BYTES + 1)
            trailing = os.read(descriptor, 1)
            closed_state = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current_chain = _lexical_chain_states(package, relative)
        drift = chain_drift(
            opened_chain, current_chain, chain_components(package, relative)
        )
        if (
            trailing
            or drift is not None
            or opened_chain[-1] != _state(opened)
            or _state(opened) != _state(closed_state)
        ):
            raise OSError(
                "canonical resource changed during read"
                + ("" if drift is None else f": {drift}")
            )
        parse_xu_stitch_calibration_bytes(raw)
        return raw
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        UnicodeEncodeError,
        ValueError,
        XuStitchCalibrationRefused,
    ) as error:
        raise XuStitchCalibrationRefused(
            "XU_CANONICAL_ASSET_UNAVAILABLE",
            "canonical SURFACE calibration resource is unavailable",
        ) from error


@dataclass(frozen=True, slots=True)
class XuStitchCalibrationInput:
    locator: str | Path

    def __post_init__(self) -> None:
        try:
            shown = os.fspath(self.locator)
        except TypeError as error:
            raise TypeError("XU calibration locator must be path-like") from error
        if type(shown) is not str or not shown or "\x00" in shown:
            raise TypeError("XU calibration locator must be a nonempty exact path")
        try:
            shown.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise TypeError(
                "XU calibration locator must be valid UTF-8 text"
            ) from error
        object.__setattr__(self, "locator", shown)


def _state(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    # The ctime slot rides the tree-wide win32 seam: the descriptor and
    # pathname views of one file this module compares disagree on it there.
    return (
        int(value.st_mode),
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        identity_ctime_ns(value.st_ctime_ns),
    )


# POSIX opens every component of the chain relative to the previous
# descriptor with O_NOFOLLOW|O_DIRECTORY, so nothing can be swapped for a
# link between two steps.  Windows can neither open a directory nor pass
# dir_fd (os.supports_dir_fd is empty there and the keyword raises
# NotImplementedError, which no OSError clause catches), so the chain is
# inspected component by component with lstat instead -- symbolic links
# and every other name-surrogate reparse point (junctions) refused -- the
# leaf is opened by name and must carry the inspected leaf's identity, and
# the same lexical chain is inspected again after the read.
_DESCRIPTOR_WALK = (
    os.open in getattr(os, "supports_dir_fd", frozenset())
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
)
# IsReparseTagNameSurrogate: the reparse point stands for another path.
_REPARSE_NAME_SURROGATE = 0x20000000
_LEAF_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_NOINHERIT", 0)
)


def _is_link(value: os.stat_result) -> bool:
    """A symbolic link, or on Windows any junction-like reparse point."""
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_reparse_tag", 0) & _REPARSE_NAME_SURROGATE
    )


def _exact_root(parts: tuple[str, ...]) -> bool:
    """*parts* starts at a filesystem root: ``/`` on POSIX; ``D:\\`` or
    ``\\\\server\\share\\`` on Windows, never a rootless drive or a
    drive-less root."""
    if not parts:
        return False
    if os.name == "nt":
        drive, root = os.path.splitdrive(parts[0])
        return bool(drive) and root == os.sep
    return parts[0] == os.sep


def _lexical_target(
    request: XuStitchCalibrationInput,
    project_root: str | Path,
) -> tuple[str, str, str]:
    try:
        project_shown = os.fspath(project_root)
    except TypeError as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_PROJECT_INVALID",
            "Project must be a nonempty exact path",
        ) from error
    if (
        type(project_shown) is not str
        or not project_shown
        or "\x00" in project_shown
    ):
        _refuse(
            "XU_CALIBRATION_PROJECT_INVALID",
            "Project must be a nonempty exact path",
        )
    try:
        project_shown.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_PROJECT_INVALID",
            "Project path must be valid UTF-8 text",
        ) from error
    try:
        project = os.path.normpath(os.path.abspath(project_shown))
        shown = os.fspath(request.locator)
        shown.encode("utf-8", errors="strict")
    except (OSError, ValueError, UnicodeEncodeError) as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_PROJECT_INVALID",
            "Project and calibration paths cannot be normalized",
        ) from error
    project_parts = Path(project).parts
    current = project_parts[0] if project_parts else os.sep
    try:
        for part in project_parts[1:]:
            current = os.path.join(current, part)
            state = os.lstat(current)
            if _is_link(state):
                _refuse(
                    "XU_CALIBRATION_SYMLINK_REFUSED",
                    "Project ancestry traverses a symbolic link",
                )
    except FileNotFoundError:
        _refuse(
            "XU_CALIBRATION_PROJECT_INVALID",
            "Project must be an existing real directory",
        )
    except OSError as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_PROJECT_INVALID",
            "Project ancestry cannot be inspected",
        ) from error
    if not os.path.isdir(project) or os.path.islink(project):
        _refuse("XU_CALIBRATION_PROJECT_INVALID", "Project must be a real directory")
    target = os.path.normpath(
        shown if os.path.isabs(shown) else os.path.join(project, shown)
    )
    try:
        if os.path.commonpath((project, target)) != project:
            raise ValueError
    except ValueError:
        _refuse(
            "XU_CALIBRATION_OUTSIDE_PROJECT",
            "calibration locator is outside Project",
        )
    relative = os.path.relpath(target, project)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        _refuse(
            "XU_CALIBRATION_OUTSIDE_PROJECT",
            "calibration locator is outside Project",
        )
    current = project
    relative_parts = Path(relative).parts
    for index, part in enumerate(relative_parts):
        current = os.path.join(current, part)
        try:
            state = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as error:
            raise XuStitchCalibrationRefused(
                "XU_CALIBRATION_UNAVAILABLE",
                "calibration path cannot be inspected",
            ) from error
        if _is_link(state):
            _refuse(
                "XU_CALIBRATION_SYMLINK_REFUSED",
                "calibration path traverses a symbolic link",
            )
        # POSIX reports a non-directory ancestor as ENOTDIR at the next
        # step; Windows reports it as not-found, so settle it here.
        if index < len(relative_parts) - 1 and not stat.S_ISDIR(state.st_mode):
            _refuse(
                "XU_CALIBRATION_UNAVAILABLE",
                "calibration path cannot be inspected",
            )
    resolved_project = os.path.realpath(project)
    resolved_target = os.path.realpath(target)
    try:
        # Different drives have no common path on Windows.
        if os.path.commonpath((resolved_project, resolved_target)) != resolved_project:
            raise ValueError
    except ValueError:
        _refuse(
            "XU_CALIBRATION_OUTSIDE_PROJECT",
            "resolved calibration locator is outside Project",
        )
    return project, target, relative


def _open_no_follow_chain(
    project: str,
    relative: str,
) -> tuple[int, tuple[tuple[int, int, int, int, int, int], ...]]:
    relative_parts = Path(relative).parts
    project_parts = Path(project).parts
    if (
        not relative_parts
        or not _exact_root(project_parts)
        or any(
            part in {"", os.curdir, os.pardir}
            for part in relative_parts
        )
    ):
        _refuse(
            "XU_CALIBRATION_OUTSIDE_PROJECT",
            "calibration locator has an invalid lexical component",
        )
    if not _DESCRIPTOR_WALK:
        return _open_inspected_chain(project_parts, relative_parts)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptors: list[int] = []
    states: list[tuple[int, int, int, int, int, int]] = []
    try:
        current = os.open(project_parts[0], directory_flags)
        descriptors.append(current)
        states.append(_state(os.fstat(current)))
        for part in project_parts[1:] + relative_parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
            states.append(_state(os.fstat(current)))
        descriptor = os.open(relative_parts[-1], file_flags, dir_fd=current)
        states.append(_state(os.fstat(descriptor)))
    except FileNotFoundError as error:
        for opened in reversed(descriptors):
            try:
                os.close(opened)
            except OSError:
                pass
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_UNAVAILABLE",
            "calibration path is unavailable",
        ) from error
    except OSError as error:
        for opened in reversed(descriptors):
            try:
                os.close(opened)
            except OSError:
                pass
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_SYMLINK_REFUSED",
            "calibration path could not be opened without link traversal",
        ) from error
    for opened in reversed(descriptors):
        os.close(opened)
    return descriptor, tuple(states)


def _open_inspected_chain(
    project_parts: tuple[str, ...],
    relative_parts: tuple[str, ...],
) -> tuple[int, tuple[tuple[int, int, int, int, int, int], ...]]:
    """The no-dir_fd walk: lstat every component, open the leaf by name."""
    states: list[tuple[int, int, int, int, int, int]] = []
    current = project_parts[0]
    try:
        states.append(_state(os.lstat(current)))
        for part in project_parts[1:] + relative_parts[:-1]:
            current = os.path.join(current, part)
            value = os.lstat(current)
            if _is_link(value):
                raise OSError("calibration ancestry traverses a link")
            if not stat.S_ISDIR(value.st_mode):
                raise FileNotFoundError(current)
            states.append(_state(value))
        leaf = os.path.join(current, relative_parts[-1])
        inspected = os.lstat(leaf)
        if _is_link(inspected):
            raise OSError("calibration path is a link")
        if not stat.S_ISREG(inspected.st_mode):
            # Settled before the open: a directory cannot be opened here
            # and a pipe would block it.  POSIX reports the same refusal
            # from the caller's fstat.
            _refuse(
                "XU_CALIBRATION_NOT_REGULAR",
                "calibration must be a regular non-symlink file",
            )
        descriptor = os.open(leaf, _LEAF_FLAGS)
    except FileNotFoundError as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_UNAVAILABLE",
            "calibration path is unavailable",
        ) from error
    except OSError as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_SYMLINK_REFUSED",
            "calibration path could not be opened without link traversal",
        ) from error
    try:
        if _state(os.fstat(descriptor)) != _state(inspected):
            raise OSError("calibration leaf changed between inspection and open")
    except OSError as error:
        os.close(descriptor)
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_SYMLINK_REFUSED",
            "calibration path could not be opened without link traversal",
        ) from error
    states.append(_state(inspected))
    return descriptor, tuple(states)


def _lexical_chain_states(
    project: str,
    relative: str,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    states: list[tuple[int, int, int, int, int, int]] = []
    project_parts = Path(project).parts
    current = project_parts[0]
    for part in project_parts[1:] + Path(relative).parts:
        current = os.path.join(current, part)
        value = os.lstat(current)
        if _is_link(value):
            _refuse(
                "XU_CALIBRATION_SYMLINK_REFUSED",
                "calibration path changed to a symbolic link",
            )
        states.append(_state(value))
    states.insert(0, _state(os.lstat(project_parts[0])))
    return tuple(states)


@dataclass(eq=False, frozen=True, slots=True)
class XuStitchCalibrationReceipt:
    request: XuStitchCalibrationInput
    project_root: str
    lexical_relative_path: str
    resolved_relative_path: str
    file_state: tuple[int, int, int, int, int, int]
    projection: XuStitchCalibrationProjection
    fingerprint: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RECEIPT_FACTORY
            or type(self.request) is not XuStitchCalibrationInput
            or type(self.projection) is not XuStitchCalibrationProjection
            or type(self.file_state) is not tuple
            or len(self.file_state) != 6
            or type(self.fingerprint) is not str
            or len(self.fingerprint) != 64
        ):
            raise TypeError("XU calibration receipt is not factory-owned")

    @property
    def byte_count(self) -> int:
        return len(self.projection.content)

    @property
    def raw_sha256(self) -> str:
        return self.projection.raw_sha256

    @property
    def semantic_fingerprint(self) -> str:
        return self.projection.semantic_fingerprint

    @property
    def content(self) -> bytes:
        return self.projection.content

    def __copy__(self):
        raise TypeError("XU calibration receipt is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("XU calibration receipt is not copyable")

    def __reduce__(self):
        raise TypeError("XU calibration receipt is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("XU calibration receipt is not serializable")


def capture_xu_stitch_calibration(
    request: XuStitchCalibrationInput,
    *,
    project_root: str | Path,
) -> XuStitchCalibrationReceipt:
    if type(request) is not XuStitchCalibrationInput:
        raise TypeError("XU calibration capture requires exact input")
    project, target, lexical_relative = _lexical_target(request, project_root)
    try:
        descriptor, opened_chain = _open_no_follow_chain(
            project, lexical_relative
        )
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                _refuse(
                    "XU_CALIBRATION_NOT_REGULAR",
                    "calibration must be a regular non-symlink file",
                )
            raw = os.read(descriptor, _MAX_ASSET_BYTES + 1)
            trailing = os.read(descriptor, 1)
            closed_state = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        _project_again, target_again, relative_again = _lexical_target(
            request, project
        )
        current_chain = _lexical_chain_states(project, lexical_relative)
    except OSError as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_UNAVAILABLE", "calibration file cannot be captured"
        ) from error
    drift = chain_drift(
        opened_chain, current_chain, chain_components(project, lexical_relative)
    )
    if (
        not stat.S_ISREG(opened.st_mode)
        or trailing
        or target_again != target
        or relative_again != lexical_relative
        or drift is not None
        or opened_chain[-1] != _state(opened)
        or _state(opened) != _state(closed_state)
    ):
        _refuse(
            "XU_CALIBRATION_IDENTITY_MISMATCH",
            "calibration changed during capture"
            + ("" if drift is None else f": {drift}"),
        )
    projection = parse_xu_stitch_calibration_bytes(raw)
    resolved_project = os.path.realpath(project)
    resolved_target = os.path.realpath(target)
    try:
        if os.path.commonpath((resolved_project, resolved_target)) != resolved_project:
            raise ValueError
    except ValueError:
        _refuse(
            "XU_CALIBRATION_OUTSIDE_PROJECT",
            "calibration resolved outside Project after capture",
        )
    resolved_relative = os.path.relpath(resolved_target, resolved_project)
    if resolved_relative == os.pardir or resolved_relative.startswith(
        os.pardir + os.sep
    ):
        _refuse(
            "XU_CALIBRATION_OUTSIDE_PROJECT",
            "calibration resolved outside Project after capture",
        )
    # Receipt paths are spelled with ``/`` on every platform so provenance
    # and the receipt fingerprint do not depend on the host separator.
    lexical_portable = Path(lexical_relative).as_posix()
    resolved_portable = Path(resolved_relative).as_posix()
    receipt_projection = (
        lexical_portable,
        resolved_portable,
        _state(closed_state),
        projection.raw_sha256,
        projection.semantic_fingerprint,
    )
    fingerprint = analysis_canonical_fingerprint(
        "xu-stitch-calibration-receipt-v1", receipt_projection
    )
    return XuStitchCalibrationReceipt(
        request,
        project,
        lexical_portable,
        resolved_portable,
        _state(closed_state),
        projection,
        fingerprint,
        _RECEIPT_FACTORY,
    )


def revalidate_xu_stitch_calibration(
    receipt: XuStitchCalibrationReceipt,
) -> bytes:
    if type(receipt) is not XuStitchCalibrationReceipt:
        raise TypeError("XU calibration revalidation requires exact receipt")
    current = capture_xu_stitch_calibration(
        receipt.request,
        project_root=receipt.project_root,
    )
    if (
        current.lexical_relative_path != receipt.lexical_relative_path
        or current.resolved_relative_path != receipt.resolved_relative_path
        or current.file_state != receipt.file_state
        or current.fingerprint != receipt.fingerprint
        or current.content != receipt.content
    ):
        _refuse(
            "XU_CALIBRATION_IDENTITY_MISMATCH",
            "calibration no longer matches its receipt",
        )
    return current.content


def _write_asset(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("canonical asset write made no progress")
        offset += written
    os.fsync(descriptor)


def _install_by_descriptor(project: str, relative: str, raw: bytes) -> None:
    """Create the asset through an O_NOFOLLOW|O_DIRECTORY descriptor chain."""
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    created = False
    parent_descriptor: int | None = None
    filename = Path(relative).parts[-1]
    project_parts = Path(project).parts
    try:
        current = os.open(project_parts[0], directory_flags)
        descriptors.append(current)
        for part in project_parts[1:] + Path(relative).parts[:-1]:
            try:
                opened = os.open(part, directory_flags, dir_fd=current)
            except FileNotFoundError:
                os.mkdir(part, mode=0o755, dir_fd=current)
                opened = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(opened)
            current = opened
        parent_descriptor = current
        descriptor = os.open(filename, file_flags, 0o644, dir_fd=current)
        created = True
        try:
            _write_asset(descriptor, raw)
        finally:
            os.close(descriptor)
        os.fsync(current)
    except FileExistsError as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_INSTALL_CONFLICT",
            "canonical calibration destination appeared during install",
        ) from error
    except OSError as error:
        if created and parent_descriptor is not None:
            try:
                os.unlink(filename, dir_fd=parent_descriptor)
            except OSError:
                pass
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_INSTALL_FAILED",
            "canonical calibration could not be installed safely",
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _install_by_name(project: str, relative: str, raw: bytes) -> None:
    """The no-dir_fd installer: every ancestor is inspected with lstat
    (links refused) before the leaf is created exclusively by name; only
    the file itself is flushed, a directory has no fsync on Windows.

    A by-name open cannot pin where the leaf lands, so before any byte is
    written the created object's final path must name the requested entry
    of the inspected parent directory (``created_leaf_misplacement``).  A
    leaf that landed elsewhere, or that could not be filled, is disposed
    of through the handle that created it while it is still open
    (``dispose_created_leaf``) and the install refused; nothing here ever
    unlinks a name, which by then may belong to somebody else's file.
    Where the platform cannot delete by descriptor the refusal names the
    leaf it leaves behind."""
    project_parts = Path(project).parts
    current = project_parts[0]
    try:
        value = os.lstat(current)
        for part in project_parts[1:] + Path(relative).parts[:-1]:
            current = os.path.join(current, part)
            try:
                value = os.lstat(current)
            except FileNotFoundError:
                os.mkdir(current, mode=0o755)
                value = os.lstat(current)
            if _is_link(value) or not stat.S_ISDIR(value.st_mode):
                raise OSError("canonical asset ancestry is not a real directory")
        parent_identity = directory_identity(value)
        name = Path(relative).parts[-1]
        target = os.path.join(current, name)
        descriptor = descriptor_path.create_exclusive_leaf(target)
        try:
            misplaced = descriptor_path.created_leaf_misplacement(
                descriptor, parent_identity, name
            )
            if misplaced is None:
                try:
                    _write_asset(descriptor, raw)
                except OSError as error:
                    if not descriptor_path.dispose_created_leaf(descriptor):
                        error.add_note(f"partial canonical asset left at {target}")
                    raise
            elif not descriptor_path.dispose_created_leaf(descriptor):
                misplaced = f"{misplaced}; empty leaf left there"
        finally:
            os.close(descriptor)
        if misplaced is not None:
            raise OSError(
                "canonical asset was created outside its inspected directory: "
                + misplaced
            )
    except FileExistsError as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_INSTALL_CONFLICT",
            "canonical calibration destination appeared during install",
        ) from error
    except OSError as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_INSTALL_FAILED",
            "canonical calibration could not be installed safely",
        ) from error


def install_canonical_xu_stitch_calibration(
    *,
    project_root: str | Path,
) -> XuStitchCalibrationReceipt:
    """Create or re-admit the exact bundled SURFACE asset inside Project.

    The installer is create-only: an existing byte-identical canonical asset is
    re-admitted, while any conflicting path is refused without replacement.
    """

    request = XuStitchCalibrationInput(
        CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    )
    project, target, relative = _lexical_target(request, project_root)
    raw = canonical_surface_resource_bytes()
    if os.path.lexists(target):
        try:
            current = capture_xu_stitch_calibration(
                request,
                project_root=project,
            )
        except XuStitchCalibrationRefused as error:
            raise XuStitchCalibrationRefused(
                "XU_CALIBRATION_INSTALL_CONFLICT",
                "canonical calibration destination already conflicts",
            ) from error
        if current.content != raw:
            _refuse(
                "XU_CALIBRATION_INSTALL_CONFLICT",
                "canonical calibration destination has different bytes",
            )
        return current

    if _DESCRIPTOR_WALK:
        _install_by_descriptor(project, relative, raw)
    else:
        _install_by_name(project, relative, raw)
    try:
        receipt = capture_xu_stitch_calibration(
            request,
            project_root=project,
        )
    except XuStitchCalibrationRefused as error:
        raise XuStitchCalibrationRefused(
            "XU_CALIBRATION_INSTALL_FAILED",
            "installed calibration could not be re-admitted",
        ) from error
    if receipt.content != raw:
        _refuse(
            "XU_CALIBRATION_INSTALL_FAILED",
            "installed calibration bytes changed",
        )
    return receipt


__all__ = [
    "CANONICAL_XU_STITCH_CALIBRATION_LOCATOR",
    "XuStitchCalibrationInput",
    "XuStitchCalibrationProjection",
    "XuStitchCalibrationReceipt",
    "XuStitchCalibrationRefused",
    "canonical_surface_resource_bytes",
    "capture_xu_stitch_calibration",
    "install_canonical_xu_stitch_calibration",
    "parse_xu_stitch_calibration_bytes",
    "revalidate_xu_stitch_calibration",
]
