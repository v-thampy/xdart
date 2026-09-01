"""Engine-light custody for the canonical RSM geometry authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import hashlib
from importlib import resources
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType

from xrd_tools.analysis.canonical_fingerprint import analysis_canonical_fingerprint


_MAX_ASSET_BYTES = 65_536
_MAX_DEPTH = 6
_MAX_NODES = 256
_MAX_STRING_LENGTH = 256
_MAX_PATH_UTF8_BYTES = 4_096
_RESOURCE_PARTS = (
    "assets",
    "rsm",
    "ssrl17_2_psic_pilatus300k_sto_v1.json",
)
_RESOURCE_BYTE_COUNT = 745
_RESOURCE_SHA256 = (
    "0f22b00363ff93fec7b97c7b5c31f8b8e389aac3aae334cbadbc213b629d9f84"
)
_RESOURCE_SEMANTIC_FINGERPRINT = (
    "a5841542c58c5ff33899ef13a6f3883c6f6e356eefb11c41d1f24936eed9f0fa"
)
CANONICAL_RSM_GEOMETRY_LOCATOR = (
    "calibration/rsm/ssrl17_2_psic_pilatus300k_sto_v1.json"
)
_TOP_LEVEL_KEYS = {
    "schema",
    "version",
    "preset",
    "diffractometer",
    "detector",
    "validation",
}
_EXPECTED_VALUE = {
    "schema": "xdart.rsm_geometry",
    "version": 1,
    "preset": "ssrl17_2_psic_pilatus300k_sto",
    "diffractometer": {
        "preset": "psic",
        "sample_circles": ["x+", "z-", "y+", "z-"],
        "detector_circles": ["x+", "z-"],
        "r_i": [0.0, 1.0, 0.0],
        "camera": ["z-", "x-"],
        "hxrd_n": [0.0, 1.0, 0.0],
        "hxrd_q": [0.0, 0.0, 1.0],
        "hxrd_geometry": "real",
        "motor_roles": ["mu", "eta", "chi", "phi", "nu", "del"],
    },
    "detector": {
        "header": {
            "cch1": 97.0,
            "cch2": 243.0,
            "pwidth1": 0.172,
            "pwidth2": 0.172,
            "distance": 1014.7173,
            "Nch1": 195,
            "Nch2": 487,
        },
        "image_orientation": {
            "rotation": 0,
            "flip_vertical": False,
            "flip_horizontal": False,
            "transpose": False,
        },
        "roi": [0, -1, 0, -1],
    },
    "validation": {
        "notebook": "RSM_process.ipynb",
        "notebook_sha256": (
            "980d6caae252b86cfd68e7cc605621f16a539f341b8abdb2ab946d43d84808ac"
        ),
        "reference_scan": "43.1",
    },
}
_PROJECTION_FACTORY = object()
_INPUT_FACTORY = object()
_RECEIPT_FACTORY = object()
_EFFECTIVE_FACTORY = object()
_MEMBER_BINDING_FACTORY = object()
_PSIC_ROLES = ("mu", "eta", "chi", "phi", "nu", "del")
_MAX_RSM_MEMBERS = 16


class RSMGeometryAssetRefused(ValueError):
    def __init__(self, code: str, message: str | None = None):
        if type(code) is not str or not code:
            raise TypeError("RSM geometry refusal code must be nonempty")
        self.code = code
        super().__init__(message or code)


def _refuse(code: str, message: str) -> None:
    raise RSMGeometryAssetRefused(code, message)


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
        _refuse("RSM_GEOMETRY_PARSE_FAILED", "geometry exceeds depth 6")
    if type(value) is str:
        if len(value) > _MAX_STRING_LENGTH:
            _refuse("RSM_GEOMETRY_PARSE_FAILED", "geometry string is oversized")
        return
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _refuse("RSM_GEOMETRY_PARSE_FAILED", "geometry number is nonfinite")
        return
    if type(value) is dict:
        budget[0] += len(value)
        if budget[0] > _MAX_NODES:
            _refuse("RSM_GEOMETRY_PARSE_FAILED", "geometry is oversized")
        for key, item in value.items():
            if type(key) is not str or len(key) > _MAX_STRING_LENGTH:
                _refuse("RSM_GEOMETRY_PARSE_FAILED", "geometry key is invalid")
            _bounded_projection(item, depth=depth + 1, budget=budget)
        return
    if type(value) is list:
        budget[0] += len(value)
        if budget[0] > _MAX_NODES:
            _refuse("RSM_GEOMETRY_PARSE_FAILED", "geometry is oversized")
        for item in value:
            _bounded_projection(item, depth=depth + 1, budget=budget)
        return
    _refuse("RSM_GEOMETRY_PARSE_FAILED", "geometry value type is unsupported")


def _freeze(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(eq=False, frozen=True, slots=True)
class RSMGeometryAssetProjection:
    canonical_json: str
    byte_count: int
    raw_sha256: str
    semantic_fingerprint: str
    frozen_value: Mapping[str, object] = field(repr=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _PROJECTION_FACTORY
            or type(self.canonical_json) is not str
            or self.byte_count != _RESOURCE_BYTE_COUNT
            or self.raw_sha256 != _RESOURCE_SHA256
            or self.semantic_fingerprint != _RESOURCE_SEMANTIC_FINGERPRINT
            or not isinstance(self.frozen_value, MappingProxyType)
        ):
            raise TypeError("RSM geometry projection is not factory-owned")

    @property
    def value(self) -> Mapping[str, object]:
        return self.frozen_value

    @property
    def content(self) -> bytes:
        return self.canonical_json.encode("utf-8")

    def __copy__(self):
        raise TypeError("RSM geometry projection is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("RSM geometry projection is not copyable")

    def __reduce__(self):
        raise TypeError("RSM geometry projection is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("RSM geometry projection is not serializable")


def parse_rsm_geometry_asset_bytes(raw: bytes) -> RSMGeometryAssetProjection:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_ASSET_BYTES:
        _refuse(
            "RSM_GEOMETRY_PARSE_FAILED",
            "geometry must be nonempty bounded exact bytes",
        )
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw or raw.endswith(b"\n"):
        _refuse(
            "RSM_GEOMETRY_PARSE_FAILED",
            "geometry contains a BOM, NUL, or trailing newline",
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_PARSE_FAILED", "geometry is not strict UTF-8"
        ) from error

    def object_pairs(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            _refuse("RSM_GEOMETRY_PARSE_FAILED", "geometry has duplicate keys")
        return result

    def invalid_constant(_value):
        _refuse("RSM_GEOMETRY_PARSE_FAILED", "geometry has nonfinite JSON")

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except RSMGeometryAssetRefused:
        raise
    except (RecursionError, TypeError, ValueError) as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_PARSE_FAILED", "geometry is not one exact JSON object"
        ) from error
    if type(value) is not dict or set(value) != _TOP_LEVEL_KEYS:
        _refuse("RSM_GEOMETRY_SCHEMA_UNSUPPORTED", "geometry schema is unsupported")
    _bounded_projection(value, depth=1, budget=[0])
    try:
        canonical = _canonical_bytes(value)
    except (TypeError, ValueError, UnicodeError) as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_PARSE_FAILED", "geometry cannot be canonicalized"
        ) from error
    if canonical != raw:
        _refuse("RSM_GEOMETRY_NOT_CANONICAL", "geometry is not canonical JSON")
    if value != _EXPECTED_VALUE:
        _refuse(
            "RSM_GEOMETRY_SCHEMA_UNSUPPORTED",
            "geometry is not the authenticated SSRL 17-2 RSM v1 projection",
        )
    semantic = analysis_canonical_fingerprint("rsm-geometry-asset-v1", value)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if (
        len(raw) != _RESOURCE_BYTE_COUNT
        or raw_sha256 != _RESOURCE_SHA256
        or semantic != _RESOURCE_SEMANTIC_FINGERPRINT
    ):
        _refuse(
            "RSM_GEOMETRY_SCHEMA_UNSUPPORTED",
            "geometry identity is not the authenticated RSM v1 projection",
        )
    return RSMGeometryAssetProjection(
        text,
        len(raw),
        raw_sha256,
        semantic,
        _freeze(value),
        _PROJECTION_FACTORY,
    )


def _state(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_mode),
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _validate_exact_relative(locator: str) -> tuple[str, ...]:
    if (
        type(locator) is not str
        or not locator
        or "\x00" in locator
        or os.path.isabs(locator)
        or os.path.normpath(locator) != locator
    ):
        raise TypeError("RSM geometry locator must be a normalized relative path")
    try:
        encoded = locator.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise TypeError("RSM geometry locator must be valid UTF-8 text") from error
    if len(encoded) > _MAX_PATH_UTF8_BYTES:
        raise TypeError("RSM geometry locator exceeds 4096 UTF-8 bytes")
    parts = Path(locator).parts
    if not parts or any(part in {"", os.curdir, os.pardir} for part in parts):
        raise TypeError("RSM geometry locator has an invalid component")
    return parts


@dataclass(eq=False, frozen=True, slots=True)
class RSMGeometryAssetInput:
    locator: str | Path
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if _claim is not _INPUT_FACTORY:
            raise TypeError("RSM geometry input is not factory-owned")
        try:
            shown = os.fspath(self.locator)
        except TypeError as error:
            raise TypeError("RSM geometry locator must be path-like") from error
        _validate_exact_relative(shown)
        object.__setattr__(self, "locator", shown)

    def __copy__(self):
        raise TypeError("RSM geometry input is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("RSM geometry input is not copyable")

    def __reduce__(self):
        raise TypeError("RSM geometry input is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("RSM geometry input is not serializable")

    def __replace__(self, /, **_changes):
        raise TypeError("RSM geometry input is not replaceable")


def rsm_geometry_asset_input(locator: str | Path) -> RSMGeometryAssetInput:
    """Create one immutable, factory-owned RSM geometry locator request."""

    return RSMGeometryAssetInput(locator, _INPUT_FACTORY)


def _project_path(project_root: str | Path) -> str:
    try:
        shown = os.fspath(project_root)
    except TypeError as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_PROJECT_INVALID", "Project must be path-like"
        ) from error
    if type(shown) is not str or not shown or "\x00" in shown:
        _refuse("RSM_GEOMETRY_PROJECT_INVALID", "Project path is invalid")
    try:
        encoded = shown.encode("utf-8", errors="strict")
        if len(encoded) > _MAX_PATH_UTF8_BYTES:
            raise ValueError("Project path exceeds 4096 UTF-8 bytes")
        project = os.path.normpath(os.path.abspath(shown))
        if len(project.encode("utf-8", errors="strict")) > _MAX_PATH_UTF8_BYTES:
            raise ValueError("normalized Project path exceeds 4096 UTF-8 bytes")
    except (OSError, ValueError, UnicodeEncodeError) as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_PROJECT_INVALID", "Project path cannot be normalized"
        ) from error
    current = os.sep
    try:
        for part in Path(project).parts[1:]:
            current = os.path.join(current, part)
            state = os.lstat(current)
            if stat.S_ISLNK(state.st_mode):
                _refuse(
                    "RSM_GEOMETRY_SYMLINK_REFUSED",
                    "Project ancestry traverses a symbolic link",
                )
            if not stat.S_ISDIR(state.st_mode):
                _refuse("RSM_GEOMETRY_PROJECT_INVALID", "Project is not a directory")
    except FileNotFoundError:
        _refuse("RSM_GEOMETRY_PROJECT_INVALID", "Project does not exist")
    except OSError as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_PROJECT_INVALID", "Project ancestry is unavailable"
        ) from error
    return project


def _open_no_follow_chain(
    project: str,
    relative: str,
    *,
    final_flags: int | None = None,
    final_mode: int = 0o644,
) -> tuple[int, tuple[tuple[int, int, int, int, int, int], ...]]:
    relative_parts = _validate_exact_relative(relative)
    project_parts = Path(project).parts
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if final_flags is None:
        final_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptors: list[int] = []
    states: list[tuple[int, int, int, int, int, int]] = []
    try:
        current = os.open(os.sep, directory_flags)
        descriptors.append(current)
        states.append(_state(os.fstat(current)))
        for part in project_parts[1:] + relative_parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
            state = os.fstat(current)
            if not stat.S_ISDIR(state.st_mode):
                raise OSError("ancestry component is not a directory")
            states.append(_state(state))
        descriptor = os.open(relative_parts[-1], final_flags, final_mode, dir_fd=current)
        states.append(_state(os.fstat(descriptor)))
    except FileNotFoundError as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_UNAVAILABLE", "geometry path is unavailable"
        ) from error
    except OSError as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_SYMLINK_REFUSED",
            "geometry path cannot be opened without link traversal",
        ) from error
    finally:
        for opened in reversed(descriptors):
            try:
                os.close(opened)
            except OSError:
                pass
    return descriptor, tuple(states)


def _lexical_chain_states(
    project: str, relative: str
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    states = [_state(os.lstat(os.sep))]
    current = os.sep
    for part in Path(project).parts[1:] + _validate_exact_relative(relative):
        current = os.path.join(current, part)
        value = os.lstat(current)
        if stat.S_ISLNK(value.st_mode):
            _refuse("RSM_GEOMETRY_SYMLINK_REFUSED", "geometry ancestry changed")
        states.append(_state(value))
    return tuple(states)


def _capture_exact(
    project: str, relative: str
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    try:
        before_chain = _lexical_chain_states(project, relative)
        descriptor, opened_chain = _open_no_follow_chain(project, relative)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                _refuse("RSM_GEOMETRY_NOT_REGULAR", "geometry is not regular")
            raw = os.read(descriptor, _MAX_ASSET_BYTES + 1)
            trailing = os.read(descriptor, 1)
            closed = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current_chain = _lexical_chain_states(project, relative)
    except RSMGeometryAssetRefused:
        raise
    except OSError as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_UNAVAILABLE", "geometry cannot be captured"
        ) from error
    if (
        trailing
        or len(raw) != int(opened.st_size)
        or before_chain != opened_chain
        or opened_chain != current_chain
        or opened_chain[-1] != _state(opened)
        or _state(opened) != _state(closed)
    ):
        _refuse("RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH", "geometry changed during capture")
    return raw, _state(closed)


def canonical_rsm_geometry_resource_bytes() -> bytes:
    try:
        package_node = resources.files("xrd_tools")
        package = os.fspath(package_node)
        if (
            type(package) is not str
            or not package
            or "\x00" in package
            or not os.path.isabs(package)
            or os.path.normpath(package) != package
        ):
            raise OSError("package root is not an exact absolute path")
        package.encode("utf-8", errors="strict")
        package_parts = Path(package).parts
        if not package_parts or package_parts[0] != os.sep or any(
            part in {"", os.curdir, os.pardir} for part in package_parts[1:]
        ):
            raise OSError("package root is not canonical")
        relative = os.path.join(*_RESOURCE_PARTS)
        raw, _revision = _capture_exact(package, relative)
        parse_rsm_geometry_asset_bytes(raw)
        return raw
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        UnicodeEncodeError,
        ValueError,
        RSMGeometryAssetRefused,
    ) as error:
        raise RSMGeometryAssetRefused(
            "RSM_CANONICAL_ASSET_UNAVAILABLE",
            "canonical RSM geometry resource is unavailable",
        ) from error


@dataclass(eq=False, frozen=True, slots=True)
class RSMGeometryAssetReceipt:
    request: RSMGeometryAssetInput = field(repr=False)
    project_root: str = field(repr=False)
    lexical_relative_path: str
    resolved_relative_path: str
    byte_count: int
    raw_sha256: str
    semantic_fingerprint: str
    file_revision: tuple[int, int, int, int, int, int]
    receipt_fingerprint: str
    projection: RSMGeometryAssetProjection
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RECEIPT_FACTORY
            or type(self.request) is not RSMGeometryAssetInput
            or type(self.projection) is not RSMGeometryAssetProjection
            or type(self.file_revision) is not tuple
            or len(self.file_revision) != 6
            or self.byte_count != self.projection.byte_count
            or self.raw_sha256 != self.projection.raw_sha256
            or self.semantic_fingerprint != self.projection.semantic_fingerprint
            or type(self.receipt_fingerprint) is not str
            or len(self.receipt_fingerprint) != 64
        ):
            raise TypeError("RSM geometry receipt is not factory-owned")

    @property
    def content(self) -> bytes:
        return self.projection.content

    @property
    def fingerprint(self) -> str:
        return self.receipt_fingerprint

    @property
    def file_state(self) -> tuple[int, int, int, int, int, int]:
        return self.file_revision

    def __copy__(self):
        raise TypeError("RSM geometry receipt is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("RSM geometry receipt is not copyable")

    def __reduce__(self):
        raise TypeError("RSM geometry receipt is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("RSM geometry receipt is not serializable")


def capture_rsm_geometry_asset(
    request: RSMGeometryAssetInput, *, project_root: str | Path
) -> RSMGeometryAssetReceipt:
    if type(request) is not RSMGeometryAssetInput:
        raise TypeError("RSM geometry capture requires exact input")
    project = _project_path(project_root)
    relative = request.locator
    raw, revision = _capture_exact(project, relative)
    projection = parse_rsm_geometry_asset_bytes(raw)
    target = os.path.join(project, relative)
    resolved_project = os.path.realpath(project)
    resolved_target = os.path.realpath(target)
    try:
        if os.path.commonpath((resolved_project, resolved_target)) != resolved_project:
            raise ValueError
    except ValueError:
        _refuse("RSM_GEOMETRY_OUTSIDE_PROJECT", "geometry resolves outside Project")
    resolved_relative = os.path.relpath(resolved_target, resolved_project)
    if resolved_relative != relative:
        _refuse("RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH", "geometry spelling changed")
    identity = (
        relative,
        resolved_relative,
        revision,
        projection.byte_count,
        projection.raw_sha256,
        projection.semantic_fingerprint,
    )
    fingerprint = analysis_canonical_fingerprint(
        "rsm-geometry-receipt-v1", identity
    )
    return RSMGeometryAssetReceipt(
        request,
        project,
        relative,
        resolved_relative,
        projection.byte_count,
        projection.raw_sha256,
        projection.semantic_fingerprint,
        revision,
        fingerprint,
        projection,
        _RECEIPT_FACTORY,
    )


def revalidate_rsm_geometry_asset(receipt: RSMGeometryAssetReceipt) -> bytes:
    if type(receipt) is not RSMGeometryAssetReceipt:
        raise TypeError("RSM geometry revalidation requires exact receipt")
    current = capture_rsm_geometry_asset(
        receipt.request, project_root=receipt.project_root
    )
    if (
        current.lexical_relative_path != receipt.lexical_relative_path
        or current.resolved_relative_path != receipt.resolved_relative_path
        or current.file_revision != receipt.file_revision
        or current.receipt_fingerprint != receipt.receipt_fingerprint
        or current.content != receipt.content
    ):
        _refuse(
            "RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH",
            "geometry no longer matches its receipt",
        )
    return current.content


def _uncopyable_geometry_value(kind: str):
    def copy_value(self):
        raise TypeError(f"{kind} is not copyable")

    def deepcopy_value(self, _memo):
        raise TypeError(f"{kind} is not copyable")

    def reduce_value(self):
        raise TypeError(f"{kind} is not serializable")

    def reduce_ex_value(self, _protocol):
        raise TypeError(f"{kind} is not serializable")

    def replace_value(self, /, **_changes):
        raise TypeError(f"{kind} is not replaceable")

    return (
        copy_value,
        deepcopy_value,
        reduce_value,
        reduce_ex_value,
        replace_value,
    )


@dataclass(eq=False, frozen=True, slots=True)
class RSMEffectiveGeometry:
    """One immutable lowering of the authenticated RSM geometry asset."""

    asset_receipt_fingerprint: str
    asset_semantic_fingerprint: str
    diffractometer_projection: object = field(repr=False)
    detector_header: object
    image_orientation: object
    roi: tuple[int, int, int, int]
    runtime_requirements: object
    fingerprint: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _EFFECTIVE_FACTORY
            or type(self.asset_receipt_fingerprint) is not str
            or len(self.asset_receipt_fingerprint) != 64
            or self.asset_semantic_fingerprint
            != _RESOURCE_SEMANTIC_FINGERPRINT
            or type(self.roi) is not tuple
            or self.roi != (0, -1, 0, -1)
            or type(self.fingerprint) is not str
            or len(self.fingerprint) != 64
        ):
            raise TypeError("RSM effective geometry is not factory-owned")

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _uncopyable_geometry_value("RSM effective geometry")


@dataclass(eq=False, frozen=True, slots=True)
class RSMMemberGeometryBinding:
    """One member ordinal's exact ordered motor-selector binding."""

    member_ordinal: int
    effective_geometry_fingerprint: str
    motor_selectors: tuple[tuple[str, object], ...]
    fingerprint: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _MEMBER_BINDING_FACTORY
            or type(self.member_ordinal) is not int
            or not 0 <= self.member_ordinal < _MAX_RSM_MEMBERS
            or type(self.effective_geometry_fingerprint) is not str
            or len(self.effective_geometry_fingerprint) != 64
            or type(self.motor_selectors) is not tuple
            or len(self.motor_selectors) != len(_PSIC_ROLES)
            or tuple(item[0] for item in self.motor_selectors) != _PSIC_ROLES
            or type(self.fingerprint) is not str
            or len(self.fingerprint) != 64
        ):
            raise TypeError("RSM member geometry binding is not factory-owned")

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _uncopyable_geometry_value("RSM member geometry binding")


def _effective_diffractometer_value(diffractometer: object) -> tuple[object, ...]:
    def mapping(value: object) -> tuple[object, ...]:
        return (value.source_motor, value.sign, value.offset)

    return (
        diffractometer.preset,
        tuple(
            mapping(getattr(diffractometer, name))
            for name in ("rot1", "rot2", "rot3", "incident_angle")
        ),
        tuple(diffractometer.sample_circles),
        tuple(diffractometer.detector_circles),
        tuple(diffractometer.r_i),
        tuple(diffractometer.camera),
        tuple(diffractometer.hxrd_n),
        tuple(diffractometer.hxrd_q),
        diffractometer.hxrd_geometry,
        tuple(mapping(item) for item in diffractometer.circle_motors),
        tuple(diffractometer.sample_motors),
        tuple(diffractometer.detector_motors),
        tuple(sorted(dict(diffractometer.qconv_kwargs).items())),
        tuple(sorted(dict(diffractometer.hxrd_kwargs).items())),
        tuple(sorted(dict(diffractometer.ang2q_kwargs).items())),
        diffractometer.calibration,
    )


def lower_rsm_effective_geometry(
    receipt: RSMGeometryAssetReceipt,
) -> RSMEffectiveGeometry:
    """Revalidate and lower one exact asset to the pinned psic geometry."""

    if type(receipt) is not RSMGeometryAssetReceipt:
        raise TypeError("RSM effective geometry requires an exact asset receipt")
    revalidate_rsm_geometry_asset(receipt)

    # These imports are intentionally delayed: importing the custody/parser
    # module itself remains stdlib-only and does not load NumPy or an engine.
    from xrd_tools.core.geometry.diffractometer import (
        Diffractometer,
        ImageOrientation,
    )
    from xrd_tools.core.geometry.pixel_q import DetectorHeader
    from xrd_tools.core.geometry.xu_runtime import (
        XuRuntimeRequirements,
        xu_runtime_requirements_projection,
    )

    value = receipt.projection.value
    diffractometer_value = value["diffractometer"]
    detector_value = value["detector"]
    roles = tuple(diffractometer_value["motor_roles"])
    if roles != _PSIC_ROLES:
        _refuse(
            "RSM_GEOMETRY_SCHEMA_UNSUPPORTED",
            "RSM motor role order is not the authenticated psic order",
        )
    diffractometer = Diffractometer.psic(
        mu=roles[0],
        eta=roles[1],
        chi=roles[2],
        phi=roles[3],
        nu=roles[4],
        del_=roles[5],
    )
    expected_diffractometer = (
        diffractometer_value["preset"],
        (
            (roles[4], 1.0, 0.0),
            (roles[5], 1.0, 0.0),
            ("", 1.0, 0.0),
            (roles[1], 1.0, 0.0),
        ),
        tuple(diffractometer_value["sample_circles"]),
        tuple(diffractometer_value["detector_circles"]),
        tuple(diffractometer_value["r_i"]),
        tuple(diffractometer_value["camera"]),
        tuple(diffractometer_value["hxrd_n"]),
        tuple(diffractometer_value["hxrd_q"]),
        diffractometer_value["hxrd_geometry"],
        tuple((role, 1.0, 0.0) for role in roles),
        (roles[1], roles[2], roles[3], roles[0]),
        (roles[5], roles[4]),
        (),
        (),
        (),
        None,
    )
    if _effective_diffractometer_value(diffractometer) != expected_diffractometer:
        _refuse(
            "RSM_GEOMETRY_LOWERING_MISMATCH",
            "Diffractometer.psic() differs from the authenticated asset",
        )
    if (
        diffractometer.calibration is not None
        or any(
            dict(getattr(diffractometer, name))
            for name in ("qconv_kwargs", "hxrd_kwargs", "ang2q_kwargs")
        )
    ):
        _refuse(
            "RSM_GEOMETRY_LOWERING_MISMATCH",
            "psic lowering contains undocumented calibration or kwargs",
        )
    # Diffractometer is frozen, but its three mapping fields are ordinary
    # dictionaries. Freeze those nested leaves before exposing the projection.
    for name in ("qconv_kwargs", "hxrd_kwargs", "ang2q_kwargs"):
        object.__setattr__(
            diffractometer,
            name,
            MappingProxyType(dict(getattr(diffractometer, name))),
        )

    header_value = detector_value["header"]
    detector_header = DetectorHeader(
        cch1=header_value["cch1"],
        cch2=header_value["cch2"],
        pwidth1=header_value["pwidth1"],
        pwidth2=header_value["pwidth2"],
        distance=header_value["distance"],
        Nch1=header_value["Nch1"],
        Nch2=header_value["Nch2"],
    )
    orientation_value = detector_value["image_orientation"]
    image_orientation = ImageOrientation(
        rotation=orientation_value["rotation"],
        flip_vertical=orientation_value["flip_vertical"],
        flip_horizontal=orientation_value["flip_horizontal"],
        transpose=orientation_value["transpose"],
    )
    roi = tuple(detector_value["roi"])
    requirements = XuRuntimeRequirements()
    identity = (
        receipt.receipt_fingerprint,
        receipt.semantic_fingerprint,
        expected_diffractometer,
        (
            detector_header.cch1,
            detector_header.cch2,
            detector_header.pwidth1,
            detector_header.pwidth2,
            detector_header.distance,
            detector_header.Nch1,
            detector_header.Nch2,
        ),
        (
            image_orientation.rotation,
            image_orientation.flip_vertical,
            image_orientation.flip_horizontal,
            image_orientation.transpose,
        ),
        roi,
        xu_runtime_requirements_projection(requirements),
    )
    fingerprint = analysis_canonical_fingerprint(
        "rsm-effective-geometry-v1", identity
    )
    return RSMEffectiveGeometry(
        receipt.receipt_fingerprint,
        receipt.semantic_fingerprint,
        diffractometer,
        detector_header,
        image_orientation,
        roi,
        requirements,
        fingerprint,
        _EFFECTIVE_FACTORY,
    )


def bind_rsm_member_geometry(
    effective_geometry: RSMEffectiveGeometry,
    *,
    member_ordinal: int,
    motor_selectors: tuple[tuple[str, object], ...],
) -> RSMMemberGeometryBinding:
    """Bind one member's exact motor occurrences to effective geometry."""

    if type(effective_geometry) is not RSMEffectiveGeometry:
        raise TypeError("RSM member binding requires exact effective geometry")
    if type(member_ordinal) is not int or not 0 <= member_ordinal < _MAX_RSM_MEMBERS:
        raise TypeError("RSM member ordinal must be an exact integer from 0 to 15")
    if type(motor_selectors) is not tuple or len(motor_selectors) != len(_PSIC_ROLES):
        raise TypeError("RSM motor selectors must be one exact six-tuple")
    from xrd_tools.analysis.module_transaction import MetadataColumnSelector

    admitted: list[tuple[str, MetadataColumnSelector]] = []
    for expected_role, item in zip(_PSIC_ROLES, motor_selectors, strict=True):
        if (
            type(item) is not tuple
            or len(item) != 2
            or item[0] != expected_role
            or type(item[1]) is not MetadataColumnSelector
        ):
            raise TypeError("RSM motor selector binding is invalid")
        admitted.append((expected_role, item[1]))
    keys = tuple((selector.name, selector.occurrence) for _role, selector in admitted)
    if len(set(keys)) != len(keys):
        raise ValueError("RSM motor selector occurrences must be unique")
    frozen = tuple(admitted)
    fingerprint = analysis_canonical_fingerprint(
        "rsm-member-geometry-binding-v1",
        (
            member_ordinal,
            effective_geometry.fingerprint,
            tuple(
                (role, selector.name, selector.occurrence)
                for role, selector in frozen
            ),
        ),
    )
    return RSMMemberGeometryBinding(
        member_ordinal,
        effective_geometry.fingerprint,
        frozen,
        fingerprint,
        _MEMBER_BINDING_FACTORY,
    )


def rsm_effective_pixel_q_map(effective_geometry: RSMEffectiveGeometry):
    """Build the exact PixelQMap projection for one effective geometry."""

    if type(effective_geometry) is not RSMEffectiveGeometry:
        raise TypeError("RSM PixelQMap requires exact effective geometry")
    from xrd_tools.core.geometry.pixel_q import PixelQMap

    return PixelQMap(
        effective_geometry.diffractometer_projection,
        effective_geometry.detector_header,
    )


def install_canonical_rsm_geometry_asset(
    *, project_root: str | Path
) -> RSMGeometryAssetReceipt:
    request = rsm_geometry_asset_input(CANONICAL_RSM_GEOMETRY_LOCATOR)
    project = _project_path(project_root)
    relative = request.locator
    target = os.path.join(project, relative)
    raw = canonical_rsm_geometry_resource_bytes()
    if os.path.lexists(target):
        try:
            current = capture_rsm_geometry_asset(request, project_root=project)
        except RSMGeometryAssetRefused as error:
            raise RSMGeometryAssetRefused(
                "RSM_GEOMETRY_INSTALL_CONFLICT", "geometry destination conflicts"
            ) from error
        if current.content != raw:
            _refuse("RSM_GEOMETRY_INSTALL_CONFLICT", "geometry bytes conflict")
        return current

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    filename = Path(relative).parts[-1]
    try:
        current = os.open(os.sep, directory_flags)
        descriptors.append(current)
        for part in Path(project).parts[1:] + Path(relative).parts[:-1]:
            try:
                opened = os.open(part, directory_flags, dir_fd=current)
            except FileNotFoundError:
                os.mkdir(part, mode=0o755, dir_fd=current)
                opened = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(opened)
            current = opened
        descriptor = os.open(filename, file_flags, 0o644, dir_fd=current)
        try:
            view = memoryview(raw)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("geometry asset write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(current)
    except FileExistsError as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_INSTALL_CONFLICT", "geometry destination appeared"
        ) from error
    except OSError as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_INSTALL_FAILED", "geometry could not be installed"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
    try:
        receipt = capture_rsm_geometry_asset(request, project_root=project)
    except RSMGeometryAssetRefused as error:
        raise RSMGeometryAssetRefused(
            "RSM_GEOMETRY_INSTALL_FAILED", "installed geometry was not admitted"
        ) from error
    if receipt.content != raw:
        _refuse("RSM_GEOMETRY_INSTALL_FAILED", "installed geometry changed")
    return receipt


__all__ = [
    "CANONICAL_RSM_GEOMETRY_LOCATOR",
    "RSMGeometryAssetInput",
    "RSMGeometryAssetProjection",
    "RSMGeometryAssetReceipt",
    "RSMGeometryAssetRefused",
    "RSMEffectiveGeometry",
    "RSMMemberGeometryBinding",
    "bind_rsm_member_geometry",
    "canonical_rsm_geometry_resource_bytes",
    "capture_rsm_geometry_asset",
    "install_canonical_rsm_geometry_asset",
    "lower_rsm_effective_geometry",
    "parse_rsm_geometry_asset_bytes",
    "revalidate_rsm_geometry_asset",
    "rsm_effective_pixel_q_map",
    "rsm_geometry_asset_input",
]
