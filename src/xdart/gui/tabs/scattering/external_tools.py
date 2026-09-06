"""Qt-free discovery and detached launch boundary for optional viewers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys


_NEXPY_ENV = "XDART_NEXPY_EXECUTABLE"
_SILX_ENV = "XDART_SILX_EXECUTABLE"
_QT_CHILD_KEYS = ("PYQTGRAPH_QT_LIB", "QT_API", "MPLBACKEND")
_WINDOWS_DETACHED_PROCESS = 0x00000008
_WINDOWS_NEW_PROCESS_GROUP = 0x00000200


class ExternalToolId(str, Enum):
    """The complete, deliberately non-extensible external-viewer inventory."""

    NEXPY_SELECTED = "nexpy_selected"
    SILX_H5VIEWER = "silx_h5viewer"


class ExternalToolAvailabilityStatus(str, Enum):
    UNAVAILABLE = "unavailable"
    REFUSED = "refused"
    AVAILABLE = "available"


class ExternalToolLaunchStatus(str, Enum):
    REFUSED = "refused"
    FAILED = "failed"
    ACCEPTED = "accepted"


@dataclass(frozen=True, slots=True)
class ExternalToolConfig:
    """Optional authoritative executable paths for the two fixed tools."""

    nexpy_executable: str | None = None
    silx_executable: str | None = None

    def __post_init__(self) -> None:
        if any(
            value is not None and type(value) is not str
            for value in (
                self.nexpy_executable,
                self.silx_executable,
            )
        ):
            raise TypeError("external viewer configuration must be text")

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str],
    ) -> "ExternalToolConfig":
        if not isinstance(environment, Mapping):
            raise TypeError("external viewer environment must be a mapping")
        return cls(
            environment.get(_NEXPY_ENV),
            environment.get(_SILX_ENV),
        )


@dataclass(frozen=True, slots=True)
class ExternalNexusQualification:
    """Fresh page-owned NeXus custody, or its exact refusal reason."""

    target: str | None
    reason: str

    def __post_init__(self) -> None:
        if (
            self.target is not None
            and (
                type(self.target) is not str
                or not os.path.isabs(self.target)
                or Path(self.target).suffix.casefold() != ".nexus"
            )
            or type(self.reason) is not str
            or not self.reason
        ):
            raise ValueError("external NeXus qualification is invalid")

    @classmethod
    def ready(cls, target: str) -> "ExternalNexusQualification":
        return cls(target, "Selected current processed NeXus is stable.")

    @classmethod
    def refused(cls, reason: str) -> "ExternalNexusQualification":
        return cls(None, reason)


@dataclass(frozen=True, slots=True)
class ExternalToolAvailability:
    tool: ExternalToolId
    status: ExternalToolAvailabilityStatus
    executable: str | None
    reason: str

    def __post_init__(self) -> None:
        if (
            type(self.tool) is not ExternalToolId
            or type(self.status) is not ExternalToolAvailabilityStatus
            or (
                self.executable is not None
                and (type(self.executable) is not str or not self.executable)
            )
            or type(self.reason) is not str
            or not self.reason
            or (
                self.status is ExternalToolAvailabilityStatus.UNAVAILABLE
                and self.executable is not None
            )
            or (
                self.status is not ExternalToolAvailabilityStatus.UNAVAILABLE
                and self.executable is None
            )
        ):
            raise ValueError("external viewer availability is invalid")

    @property
    def available(self) -> bool:
        return self.status is not ExternalToolAvailabilityStatus.UNAVAILABLE

    @property
    def enabled(self) -> bool:
        return self.status is ExternalToolAvailabilityStatus.AVAILABLE


@dataclass(frozen=True, slots=True)
class ExternalToolsProjection:
    tools: tuple[ExternalToolAvailability, ...]

    def __post_init__(self) -> None:
        if (
            type(self.tools) is not tuple
            or not all(
                type(item) is ExternalToolAvailability for item in self.tools
            )
            or tuple(item.tool for item in self.tools) != tuple(ExternalToolId)
        ):
            raise ValueError("external viewer projection must be exact")

    def for_tool(self, tool: ExternalToolId) -> ExternalToolAvailability:
        if type(tool) is not ExternalToolId:
            raise TypeError("external viewer identity must be exact")
        return next(item for item in self.tools if item.tool is tool)


@dataclass(frozen=True, slots=True)
class ExternalToolLaunchReceipt:
    tool: ExternalToolId
    status: ExternalToolLaunchStatus
    argv: tuple[str, ...]
    diagnostic: str

    def __post_init__(self) -> None:
        if (
            type(self.tool) is not ExternalToolId
            or type(self.status) is not ExternalToolLaunchStatus
            or type(self.argv) is not tuple
            or not all(type(value) is str and value for value in self.argv)
            or type(self.diagnostic) is not str
            or not self.diagnostic
            or (
                self.status is ExternalToolLaunchStatus.REFUSED
                and self.argv
            )
            or (
                self.status is not ExternalToolLaunchStatus.REFUSED
                and not self.argv
            )
        ):
            raise ValueError("external viewer launch receipt is invalid")


@dataclass(frozen=True, slots=True)
class _ExecutableReceipt:
    path: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def unavailable_external_tools() -> ExternalToolsProjection:
    return ExternalToolsProjection(tuple(
        ExternalToolAvailability(
            tool,
            ExternalToolAvailabilityStatus.UNAVAILABLE,
            None,
            "External viewer availability has not been checked.",
        )
        for tool in ExternalToolId
    ))


def external_tool_id(value: object) -> ExternalToolId | None:
    if type(value) is not str:
        return None
    try:
        return ExternalToolId(value)
    except ValueError:
        return None


def _capture_executable(
    value: object, *, require_canonical: bool,
) -> _ExecutableReceipt | None:
    if type(value) is not str or not value or not os.path.isabs(value):
        return None
    try:
        resolved = Path(value).resolve(strict=True)
        state = resolved.stat()
    except (OSError, TypeError, ValueError):
        return None
    canonical = os.path.normcase(os.path.normpath(str(resolved)))
    configured = os.path.normcase(os.path.normpath(value))
    if (
        require_canonical and canonical != configured
        or not stat.S_ISREG(state.st_mode)
        or not os.access(resolved, os.X_OK)
    ):
        return None
    return _ExecutableReceipt(
        str(resolved),
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_ctime_ns),
    )


def _resolve_executable(
    *,
    name: str,
    configured: str | None,
    configuration_name: str,
    interpreter: str,
    which: Callable[[str], str | None],
) -> tuple[_ExecutableReceipt | None, str]:
    if configured is not None:
        receipt = _capture_executable(configured, require_canonical=True)
        return (
            receipt,
            (
                f"Configured executable from {configuration_name} is ready."
                if receipt is not None
                else (
                    f"Configured executable from {configuration_name} "
                    "is unavailable."
                )
            ),
        )
    executable = _capture_executable(
        os.path.abspath(interpreter), require_canonical=False,
    )
    if executable is not None:
        sibling = _capture_executable(
            str(Path(executable.path).with_name(name)),
            require_canonical=False,
        )
        if sibling is not None:
            return sibling, f"Discovered {name} beside the active interpreter."
    found = which(name)
    if type(found) is str and found:
        candidate = _capture_executable(
            os.path.abspath(found), require_canonical=False,
        )
        if candidate is not None:
            return candidate, f"Discovered {name} on PATH."
    return None, (
        f"{name} executable was not found; configure {configuration_name}."
    )


def _selected_nexus_is_qualified(
    value: object, *, require_file: bool,
) -> bool:
    if type(require_file) is not bool:
        raise TypeError("external NeXus file requirement must be exact")
    if (
        type(value) is not str
        or not value
        or not os.path.isabs(value)
        or Path(value).suffix.casefold() != ".nexus"
    ):
        return False
    if not require_file:
        return True
    try:
        return stat.S_ISREG(Path(value).stat().st_mode)
    except OSError:
        return False


def _bounded_error(error: OSError) -> str:
    detail = str(error).replace("\n", " ").strip()
    if len(detail) > 300:
        detail = detail[:297] + "..."
    return f"{type(error).__name__}: {detail or 'external viewer spawn failed'}"


class ExternalToolRegistry:
    """Resolve once, revalidate at spawn, and retain no child ownership."""

    __slots__ = (
        "_configuration_reasons",
        "_environment",
        "_popen",
        "_receipts",
        "_windows",
    )

    def __init__(
        self,
        config: ExternalToolConfig | None = None,
        *,
        environment: Mapping[str, str] | None = None,
        interpreter: str | None = None,
        which: Callable[[str], str | None] = shutil.which,
        popen: Callable[..., object] = subprocess.Popen,
        windows: bool = os.name == "nt",
    ) -> None:
        if config is not None and type(config) is not ExternalToolConfig:
            raise TypeError("external viewer configuration must be exact")
        if (
            not callable(which)
            or not callable(popen)
            or type(windows) is not bool
        ):
            raise TypeError("external viewer adapters are invalid")
        base_environment = os.environ if environment is None else environment
        if not isinstance(base_environment, Mapping):
            raise TypeError("external viewer environment must be a mapping")
        if not all(
            type(key) is str and type(value) is str
            for key, value in base_environment.items()
        ):
            raise TypeError("external viewer environment must contain text")
        self._environment = dict(base_environment)
        config = config or ExternalToolConfig.from_environment(self._environment)
        interpreter = sys.executable if interpreter is None else interpreter
        if type(interpreter) is not str or not interpreter:
            raise TypeError("external viewer interpreter must be text")
        nexpy, nexpy_reason = _resolve_executable(
            name="nexpy",
            configured=config.nexpy_executable,
            configuration_name=_NEXPY_ENV,
            interpreter=interpreter,
            which=which,
        )
        silx, silx_reason = _resolve_executable(
            name="silx",
            configured=config.silx_executable,
            configuration_name=_SILX_ENV,
            interpreter=interpreter,
            which=which,
        )
        self._receipts = {
            ExternalToolId.NEXPY_SELECTED: nexpy,
            ExternalToolId.SILX_H5VIEWER: silx,
        }
        self._configuration_reasons = {
            ExternalToolId.NEXPY_SELECTED: nexpy_reason,
            ExternalToolId.SILX_H5VIEWER: silx_reason,
        }
        self._popen = popen
        self._windows = windows

    def project(
        self, *, nexus: ExternalNexusQualification,
    ) -> ExternalToolsProjection:
        if type(nexus) is not ExternalNexusQualification:
            raise TypeError("external NeXus qualification must be exact")
        return ExternalToolsProjection(tuple(
            self.availability(tool, nexus=nexus)
            for tool in ExternalToolId
        ))

    def availability(
        self,
        tool: ExternalToolId,
        *,
        nexus: ExternalNexusQualification | None = None,
    ) -> ExternalToolAvailability:
        if type(tool) is not ExternalToolId:
            raise TypeError("external viewer identity must be exact")
        if nexus is not None and type(nexus) is not ExternalNexusQualification:
            raise TypeError("external NeXus qualification must be exact")
        receipt = self._receipts[tool]
        if receipt is None:
            return ExternalToolAvailability(
                tool,
                ExternalToolAvailabilityStatus.UNAVAILABLE,
                None,
                self._configuration_reasons[tool],
            )
        if (
            tool is ExternalToolId.NEXPY_SELECTED
            and (
                nexus is None
                or nexus.target is None
                or not _selected_nexus_is_qualified(
                    nexus.target, require_file=False,
                )
            )
        ):
            return ExternalToolAvailability(
                tool,
                ExternalToolAvailabilityStatus.REFUSED,
                receipt.path,
                (
                    "Select one stable current processed .nexus file in Browse."
                    if nexus is None
                    else nexus.reason
                ),
            )
        return ExternalToolAvailability(
            tool,
            ExternalToolAvailabilityStatus.AVAILABLE,
            receipt.path,
            (
                "Open the selected current processed NeXus in NeXpy."
                if tool is ExternalToolId.NEXPY_SELECTED
                else "Open silx's general HDF5 viewer and choose a file."
            ),
        )

    def launch(
        self,
        tool: ExternalToolId,
        *,
        nexus: ExternalNexusQualification | None = None,
    ) -> ExternalToolLaunchReceipt:
        if type(tool) is not ExternalToolId:
            raise TypeError("external viewer identity must be exact")
        if nexus is not None and type(nexus) is not ExternalNexusQualification:
            raise TypeError("external NeXus qualification must be exact")
        receipt = self._receipts[tool]
        if receipt is None:
            return ExternalToolLaunchReceipt(
                tool,
                ExternalToolLaunchStatus.REFUSED,
                (),
                self._configuration_reasons[tool],
            )
        current = _capture_executable(receipt.path, require_canonical=True)
        if current != receipt:
            self._receipts[tool] = None
            diagnostic = "External viewer executable changed after discovery."
            self._configuration_reasons[tool] = diagnostic
            return ExternalToolLaunchReceipt(
                tool, ExternalToolLaunchStatus.REFUSED, (), diagnostic,
            )
        if (
            tool is ExternalToolId.NEXPY_SELECTED
            and (
                nexus is None
                or nexus.target is None
                or not _selected_nexus_is_qualified(
                    nexus.target, require_file=True,
                )
            )
        ):
            return ExternalToolLaunchReceipt(
                tool,
                ExternalToolLaunchStatus.REFUSED,
                (),
                (
                    "Select one stable current processed .nexus file in Browse."
                    if nexus is None
                    else nexus.reason
                    if nexus.target is None
                    else "The selected processed .nexus file is no longer available."
                ),
            )
        argv = (
            (receipt.path, nexus.target)
            if tool is ExternalToolId.NEXPY_SELECTED
            else (receipt.path, "view")
        )
        child_environment = dict(self._environment)
        # Both viewers can use the PySide6 supplied by xdart[gui]. Let their
        # own Qt selection run, without inheriting xdart's plotting overrides.
        for key in _QT_CHILD_KEYS:
            child_environment.pop(key, None)
        options: dict[str, object] = {
            "shell": False,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
            "env": child_environment,
        }
        if self._windows:
            options["creationflags"] = (
                getattr(
                    subprocess,
                    "DETACHED_PROCESS",
                    _WINDOWS_DETACHED_PROCESS,
                )
                | getattr(
                    subprocess,
                    "CREATE_NEW_PROCESS_GROUP",
                    _WINDOWS_NEW_PROCESS_GROUP,
                )
            )
        else:
            options["start_new_session"] = True
        try:
            self._popen(argv, **options)
        except OSError as error:
            return ExternalToolLaunchReceipt(
                tool,
                ExternalToolLaunchStatus.FAILED,
                argv,
                _bounded_error(error),
            )
        return ExternalToolLaunchReceipt(
            tool,
            ExternalToolLaunchStatus.ACCEPTED,
            argv,
            (
                "NeXpy launch request accepted for the selected NeXus."
                if tool is ExternalToolId.NEXPY_SELECTED
                else "silx HDF5 viewer launch request accepted."
            ),
        )


__all__ = [
    "ExternalNexusQualification",
    "ExternalToolAvailability",
    "ExternalToolAvailabilityStatus",
    "ExternalToolConfig",
    "ExternalToolId",
    "ExternalToolLaunchReceipt",
    "ExternalToolLaunchStatus",
    "ExternalToolRegistry",
    "ExternalToolsProjection",
    "external_tool_id",
    "unavailable_external_tools",
]
