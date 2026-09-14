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
            or self.raw_sha256 != _RESOURCE_SHA256
            or self.semantic_fingerprint != _RESOURCE_SEMANTIC_FINGERPRINT
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
    if (
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
