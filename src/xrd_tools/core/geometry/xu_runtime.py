"""One process-wide owner for xrayutilities global runtime state."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import platform
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
    python_min_version: tuple[int, int] = (3, 13)
    platform_systems: tuple[str, ...] = ("Darwin", "Linux", "Windows")

    def __post_init__(self) -> None:
        strings = (
            self.distribution_version,
            self.module_version,
            self.numpy_version,
            self.python_implementation,
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
            or type(self.python_min_version) is not tuple
            or self.python_min_version != (3, 13)
            or any(type(value) is not int for value in self.python_min_version)
            or type(self.platform_systems) is not tuple
            or self.platform_systems != ("Darwin", "Linux", "Windows")
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
    # Absent only when reading the original attestation shape. New execution
    # captures these observed facts; they are not calibration validation pins.
    python_implementation: str | None = None
    python_version: str | None = None
    platform_system: str | None = None
    platform_machine: str | None = None

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
        environment = (
            self.python_implementation,
            self.python_version,
            self.platform_system,
            self.platform_machine,
        )
        if any(value is not None for value in environment) and (
            any(type(value) is not str or not value for value in environment)
            or not _supported_platform(*environment[:3])
        ):
            raise TypeError("XU runtime execution environment is invalid")

    def to_attestation(self) -> dict[str, object]:
        value = {
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
        if self.python_implementation is not None:
            value.update(
                python_implementation=self.python_implementation,
                python_version=self.python_version,
                platform_system=self.platform_system,
                platform_machine=self.platform_machine,
            )
        return value


def _supported_platform(implementation: str, version: str, system: str) -> bool:
    """Portable interpreter policy, independent of numerical kernel admission."""

    parts = version.split(".")
    return (
        implementation == "CPython"
        and len(parts) == 3
        and all(part.isascii() and part.isdecimal() for part in parts)
        and tuple(map(int, parts)) >= (3, 13, 0)
        and system in ("Darwin", "Linux", "Windows")
    )


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
    )
    if not _supported_platform(*observed_platform):
        return XuRuntimeAvailability(
            False,
            "XU_PLATFORM_UNVALIDATED",
            "XU requires CPython >=3.13 on macOS, Linux, or Windows",
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
        self._config_before: tuple[float, int, int] | None = None
        self._mutation_attempted = False
        self._mutation_detected = False
        self._entered = False
        self._owner_thread_id: int | None = None
        self._observed_runtime: tuple[str, str, str] | None = None
        self._observed_environment: tuple[str, str, str, str] | None = None

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
        if not self._effective_config_matches():
            self._mutation_detected = True
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_CONFIG_MUTATED",
                "xrayutilities runtime globals changed during the active session",
            )
        return self

    def active_modules(self) -> tuple[object, object]:
        """Return ``(xrayutilities, numpy)`` only to the owning thread."""

        self.require_active()
        return self.xu, self.numpy

    def _effective_config_matches(self) -> bool:
        if self._config is None:
            return False
        try:
            observed = (
                self._config.EPSILON,
                self._config.DIGITS,
                self._config.NTHREADS,
            )
        except BaseException:
            return False
        expected = (
            self.requirements.config_epsilon,
            self.requirements.config_digits,
            XU_RUNTIME_EFFECTIVE_NTHREADS,
        )
        return all(
            type(value) is type(required) and value == required
            for value, required in zip(observed, expected, strict=True)
        )

    def _restore_config(self) -> int:
        if self._config is None or self._config_before is None:
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_CONFIG_RESTORE_FAILED",
                "XU runtime owner has no captured global state",
            )
        failures: list[BaseException] = []
        for name, value in zip(
            ("EPSILON", "DIGITS", "NTHREADS"),
            self._config_before,
            strict=True,
        ):
            try:
                setattr(self._config, name, value)
            except BaseException as error:
                failures.append(error)
        try:
            restored = (
                self._config.EPSILON,
                self._config.DIGITS,
                self._config.NTHREADS,
            )
        except BaseException as error:
            failures.append(error)
            restored = ()
        if failures or len(restored) != 3 or any(
            type(value) is not type(required) or value != required
            for value, required in zip(
                restored,
                self._config_before,
                strict=True,
            )
        ):
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_CONFIG_RESTORE_FAILED",
                "xrayutilities runtime-global restoration failed",
            )
        return restored[2]

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

            distribution = importlib.metadata.version("xrayutilities")
            numpy_distribution = importlib.metadata.version("numpy")

            if (
                distribution != self.requirements.distribution_version
                or numpy_distribution != self.requirements.numpy_version
                or xu.__version__ != self.requirements.module_version
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
            self._config_before = (
                config.EPSILON,
                config.DIGITS,
                config.NTHREADS,
            )
            self._mutation_attempted = True
            config.NTHREADS = XU_RUNTIME_EFFECTIVE_NTHREADS
            if not self._effective_config_matches():
                raise XuRuntimeUnsupported(
                    "XU_RUNTIME_CONFIG_RESTORE_FAILED",
                    "xrayutilities effective runtime globals could not be set",
                )
            self.xu = xu
            self.numpy = np
            self._observed_runtime = (distribution, xu.__version__, np.__version__)
            self._observed_environment = (
                platform.python_implementation(),
                platform.python_version(),
                platform.system(),
                platform.machine(),
            )
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
        mutation_error: XuRuntimeUnsupported | None = None
        if not self._effective_config_matches():
            self._mutation_detected = True
        if self._mutation_detected:
            mutation_error = XuRuntimeUnsupported(
                "XU_RUNTIME_CONFIG_MUTATED",
                "xrayutilities runtime globals changed during the active session",
            )
        restore_error: BaseException | None = None
        try:
            try:
                restored = self._restore_config()
            except BaseException as error:
                restore_error = error
            else:
                assert self._before is not None
                if mutation_error is None:
                    assert self._observed_runtime is not None
                    assert self._observed_environment is not None
                    self.execution_record = XuRuntimeExecutionRecord(
                        XU_RUNTIME_LOCK_POLICY,
                        *self._observed_runtime,
                        self.requirements.config_epsilon,
                        self.requirements.config_digits,
                        self._before,
                        XU_RUNTIME_EFFECTIVE_NTHREADS,
                        restored,
                        True,
                        *self._observed_environment,
                    )
        finally:
            self.xu = None
            self.numpy = None
            self._owner_thread_id = None
            XU_RUNTIME_LOCK.release()
        if restore_error is not None:
            raise XuRuntimeUnsupported(
                "XU_RUNTIME_CONFIG_RESTORE_FAILED",
                "xrayutilities runtime-global restoration failed",
            ) from restore_error
        if mutation_error is not None:
            raise mutation_error from exc_value
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
        *requirements.python_min_version,
        *requirements.platform_systems,
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
