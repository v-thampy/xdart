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

from xrd_tools.analysis.scan_operations import analysis_canonical_fingerprint


_MAX_ASSET_BYTES = 65_536
_MAX_DEPTH = 6
_MAX_NODES = 256
_MAX_STRING_LENGTH = 256
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
_RECEIPT_FACTORY = object()


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
        locator.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise TypeError("RSM geometry locator must be valid UTF-8 text") from error
    parts = Path(locator).parts
    if not parts or any(part in {"", os.curdir, os.pardir} for part in parts):
        raise TypeError("RSM geometry locator has an invalid component")
    return parts


@dataclass(frozen=True, slots=True)
class RSMGeometryAssetInput:
    locator: str | Path

    def __post_init__(self) -> None:
        try:
            shown = os.fspath(self.locator)
        except TypeError as error:
            raise TypeError("RSM geometry locator must be path-like") from error
        _validate_exact_relative(shown)
        object.__setattr__(self, "locator", shown)


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
        shown.encode("utf-8", errors="strict")
        project = os.path.normpath(os.path.abspath(shown))
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


def install_canonical_rsm_geometry_asset(
    *, project_root: str | Path
) -> RSMGeometryAssetReceipt:
    request = RSMGeometryAssetInput(CANONICAL_RSM_GEOMETRY_LOCATOR)
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
    created = False
    parent_descriptor: int | None = None
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
        parent_descriptor = current
        descriptor = os.open(filename, file_flags, 0o644, dir_fd=current)
        created = True
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
        if created and parent_descriptor is not None:
            try:
                os.unlink(filename, dir_fd=parent_descriptor)
            except OSError:
                pass
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
    "canonical_rsm_geometry_resource_bytes",
    "capture_rsm_geometry_asset",
    "install_canonical_rsm_geometry_asset",
    "parse_rsm_geometry_asset_bytes",
    "revalidate_rsm_geometry_asset",
]
