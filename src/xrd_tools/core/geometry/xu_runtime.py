"""One process-wide owner for xrayutilities global runtime state."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import platform
import sys
import threading


XU_RUNTIME_LOCK = threading.RLock()
XU_RUNTIME_LOCK_POLICY = "shared_xrd_tools_xu_rlock_v1"
XU_RUNTIME_EFFECTIVE_NTHREADS = 1


class XuRuntimeUnsupported(RuntimeError):
    def __init__(self, code: str, message: str | None = None):
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True, slots=True)
class XuRuntimeRequirements:
    distribution_version: str = "1.7.12"
    module_version: str = "1.7.12"
    numpy_version: str = "2.5.1"
    config_epsilon: float = 1e-8
    config_digits: int = 8
    python_implementation: str = "CPython"
    python_version: str = "3.13.14"
    platform_system: str = "Darwin"
    platform_machine: str = "arm64"

    def __post_init__(self) -> None:
        strings = (
            self.distribution_version,
            self.module_version,
            self.numpy_version,
            self.python_implementation,
            self.python_version,
            self.platform_system,
            self.platform_machine,
        )
        if (
            any(type(value) is not str for value in strings)
            or self.distribution_version != "1.7.12"
            or self.module_version != "1.7.12"
            or self.numpy_version != "2.5.1"
            or type(self.config_epsilon) is not float
            or self.config_epsilon != 1e-8
            or type(self.config_digits) is not int
            or self.config_digits != 8
            or self.python_implementation != "CPython"
            or self.python_version != "3.13.14"
            or self.platform_system != "Darwin"
            or self.platform_machine != "arm64"
        ):
            raise TypeError("XU runtime requirements are invalid")


@dataclass(frozen=True, slots=True)
class XuRuntimeAvailability:
    available: bool
    code: str
    reason: str

    def __post_init__(self) -> None:
        if (
            type(self.available) is not bool
            or type(self.code) is not str
            or not self.code
            or type(self.reason) is not str
        ):
            raise TypeError("XU runtime availability is invalid")


@dataclass(frozen=True, slots=True)
class XuRuntimeExecutionRecord:
    lock_policy: str
    xrayutilities_distribution_version: str
    xrayutilities_module_version: str
    numpy_version: str
    config_epsilon: float
    config_digits: int
    nthreads_before: int
    nthreads_effective: int
    nthreads_restored: int
    restore_passed: bool

    def __post_init__(self) -> None:
        if (
            self.lock_policy != XU_RUNTIME_LOCK_POLICY
            or self.xrayutilities_distribution_version != "1.7.12"
            or self.xrayutilities_module_version != "1.7.12"
            or self.numpy_version != "2.5.1"
            or type(self.config_epsilon) is not float
            or self.config_epsilon != 1e-8
            or type(self.config_digits) is not int
            or self.config_digits != 8
            or type(self.nthreads_before) is not int
            or self.nthreads_before < 0
            or type(self.nthreads_effective) is not int
            or self.nthreads_effective != XU_RUNTIME_EFFECTIVE_NTHREADS
            or type(self.nthreads_restored) is not int
            or self.nthreads_restored != self.nthreads_before
            or self.restore_passed is not True
        ):
            raise TypeError("XU runtime execution record is invalid")

    def to_attestation(self) -> dict[str, object]:
        return {
            "lock_policy": self.lock_policy,
            "xrayutilities_distribution_version": (
                self.xrayutilities_distribution_version
            ),
            "xrayutilities_module_version": self.xrayutilities_module_version,
            "numpy_version": self.numpy_version,
            "config_epsilon": self.config_epsilon,
            "config_digits": self.config_digits,
            "nthreads_before": self.nthreads_before,
            "nthreads_effective": self.nthreads_effective,
            "nthreads_restored": self.nthreads_restored,
            "restore_passed": self.restore_passed,
        }


def xu_runtime_availability(
    requirements: XuRuntimeRequirements | None = None,
) -> XuRuntimeAvailability:
    selected = XuRuntimeRequirements() if requirements is None else requirements
    if type(selected) is not XuRuntimeRequirements:
        raise TypeError("XU runtime requirements must be exact")
    observed_platform = (
        platform.python_implementation(),
        platform.python_version(),
        platform.system(),
        platform.machine(),
    )
    required_platform = (
        selected.python_implementation,
        selected.python_version,
        selected.platform_system,
        selected.platform_machine,
    )
    if observed_platform != required_platform:
        return XuRuntimeAvailability(
            False,
            "XU_PLATFORM_UNVALIDATED",
            "xu_hist requires CPython 3.13.14 on Darwin arm64",
        )
    try:
        distribution = importlib.metadata.version("xrayutilities")
        numpy_version = importlib.metadata.version("numpy")
    except BaseException:
        return XuRuntimeAvailability(
            False,
            "XU_RUNTIME_UNSUPPORTED",
            "xrayutilities 1.7.12 and NumPy 2.5.1 are required",
        )
    if (
        distribution != selected.distribution_version
        or numpy_version != selected.numpy_version
    ):
        return XuRuntimeAvailability(
            False,
            "XU_RUNTIME_UNSUPPORTED",
            "xu_hist requires xrayutilities 1.7.12 and NumPy 2.5.1",
        )
    return XuRuntimeAvailability(True, "OK", "")


class XuRuntimeSession:
    """Reentrant serialized owner whose record exists only after restoration."""

    def __init__(self, requirements: XuRuntimeRequirements | None = None):
        selected = XuRuntimeRequirements() if requirements is None else requirements
        if type(selected) is not XuRuntimeRequirements:
            raise TypeError("XU runtime requirements must be exact")
        self.requirements = selected
        self.xu = None
        self.numpy = None
        self.execution_record: XuRuntimeExecutionRecord | None = None
        self._config = None
        self._before: int | None = None
        self._mutation_attempted = False
        self._entered = False
        self._owner_thread_id: int | None = None

    def __copy__(self):
        raise TypeError("XU runtime session is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("XU runtime session is not copyable")

    @property
    def active(self) -> bool:
        """Whether this exact session is active on the calling thread."""

        return (
            self._entered
            and self.xu is not None
            and self.numpy is not None
            and self._owner_thread_id == threading.get_ident()
        )

    def require_active(self) -> "XuRuntimeSession":
        """Return this session only to its active owning thread."""

        if not self.active:
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_SESSION_INACTIVE",
                "XU runtime session is not active on the calling thread",
            )
        return self

    def active_modules(self) -> tuple[object, object]:
        """Return ``(xrayutilities, numpy)`` only to the owning thread."""

        self.require_active()
        return self.xu, self.numpy

    def _restore_config(self) -> int:
        if self._config is None or self._before is None:
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_CONFIG_RESTORE_FAILED",
                "XU runtime owner has no captured global state",
            )
        try:
            self._config.NTHREADS = self._before
            restored = self._config.NTHREADS
        except BaseException as error:
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_CONFIG_RESTORE_FAILED",
                "xrayutilities NTHREADS restoration failed",
            ) from error
        if type(restored) is not int or restored != self._before:
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_CONFIG_RESTORE_FAILED",
                "xrayutilities NTHREADS restoration did not hold",
            )
        return restored

    def __enter__(self) -> "XuRuntimeSession":
        if self._entered:
            raise RuntimeError("XU runtime session is one-shot")
        self._entered = True
        XU_RUNTIME_LOCK.acquire()
        self._owner_thread_id = threading.get_ident()
        try:
            availability = xu_runtime_availability(self.requirements)
            if not availability.available:
                raise XuRuntimeUnsupported(availability.code, availability.reason)
            import numpy as np
            import xrayutilities as xu
            from xrayutilities import config

            if (
                xu.__version__ != self.requirements.module_version
                or np.__version__ != self.requirements.numpy_version
                or type(config.EPSILON) is not float
                or config.EPSILON != self.requirements.config_epsilon
                or type(config.DIGITS) is not int
                or config.DIGITS != self.requirements.config_digits
                or type(config.NTHREADS) is not int
                or config.NTHREADS < 0
            ):
                raise XuRuntimeUnsupported(
                    "XU_RUNTIME_UNSUPPORTED",
                    "xrayutilities runtime globals do not match the pinned contract",
                )
            self._config = config
            self._before = config.NTHREADS
            self._mutation_attempted = True
            config.NTHREADS = XU_RUNTIME_EFFECTIVE_NTHREADS
            if (
                type(config.NTHREADS) is not int
                or config.NTHREADS != XU_RUNTIME_EFFECTIVE_NTHREADS
            ):
                raise XuRuntimeUnsupported(
                    "XU_RUNTIME_CONFIG_RESTORE_FAILED",
                    "xrayutilities NTHREADS could not be set to one",
                )
            self.xu = xu
            self.numpy = np
            return self
        except BaseException as primary:
            restore_error = None
            try:
                if self._mutation_attempted:
                    try:
                        self._restore_config()
                    except XuRuntimeUnsupported as error:
                        restore_error = error
            finally:
                self.xu = None
                self.numpy = None
                self._owner_thread_id = None
                XU_RUNTIME_LOCK.release()
            if restore_error is not None:
                raise restore_error from primary
            if isinstance(primary, XuRuntimeUnsupported):
                raise
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_UNSUPPORTED",
                "xrayutilities runtime import or configuration failed",
            ) from primary

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._owner_thread_id != threading.get_ident():
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_SESSION_INACTIVE",
                "XU runtime session can only exit on its owning thread",
            )
        restore_error: BaseException | None = None
        try:
            try:
                restored = self._restore_config()
            except BaseException as error:
                restore_error = error
            else:
                self.execution_record = XuRuntimeExecutionRecord(
                    XU_RUNTIME_LOCK_POLICY,
                    self.requirements.distribution_version,
                    self.requirements.module_version,
                    self.requirements.numpy_version,
                    self.requirements.config_epsilon,
                    self.requirements.config_digits,
                    self._before,
                    XU_RUNTIME_EFFECTIVE_NTHREADS,
                    restored,
                    True,
                )
        finally:
            self.xu = None
            self.numpy = None
            self._owner_thread_id = None
            XU_RUNTIME_LOCK.release()
        if restore_error is not None:
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_CONFIG_RESTORE_FAILED",
                "xrayutilities NTHREADS restoration failed",
            ) from restore_error
        return False


def xu_runtime_session(
    requirements: XuRuntimeRequirements | None = None,
) -> XuRuntimeSession:
    return XuRuntimeSession(requirements)


def xu_runtime_requirements_projection(
    requirements: XuRuntimeRequirements,
) -> tuple[object, ...]:
    """Canonical stable runtime facts used by effective geometry."""

    if type(requirements) is not XuRuntimeRequirements:
        raise TypeError("XU runtime requirements must be exact")
    return (
        requirements.distribution_version,
        requirements.module_version,
        requirements.numpy_version,
        requirements.config_epsilon,
        requirements.config_digits,
        requirements.python_implementation,
        requirements.python_version,
        requirements.platform_system,
        requirements.platform_machine,
        XU_RUNTIME_LOCK_POLICY,
        XU_RUNTIME_EFFECTIVE_NTHREADS,
    )


def require_active_xu_runtime_session(
    session: XuRuntimeSession,
) -> XuRuntimeSession:
    """Admit only one exact active session on its owning thread."""

    if type(session) is not XuRuntimeSession:
        raise TypeError("XU runtime session must be exact")
    return session.require_active()


__all__ = [
    "XU_RUNTIME_LOCK",
    "XU_RUNTIME_LOCK_POLICY",
    "XU_RUNTIME_EFFECTIVE_NTHREADS",
    "XuRuntimeAvailability",
    "XuRuntimeExecutionRecord",
    "XuRuntimeRequirements",
    "XuRuntimeSession",
    "XuRuntimeUnsupported",
    "require_active_xu_runtime_session",
    "xu_runtime_availability",
    "xu_runtime_requirements_projection",
    "xu_runtime_session",
]
