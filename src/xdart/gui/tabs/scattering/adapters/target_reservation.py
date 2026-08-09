from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock

from xrd_tools.sources.directory_session import DirectoryIndexSession

from ..contracts import (
    AdmissionFailure,
    AdmissionReceipt,
    AdmissionReleased,
    AdmissionToken,
    StartCapture,
)
from ..events import CleanupStatus, DetachedDiagnostic, detach_exception
from ..display_retirement import (
    DisplayRetirementOwner,
    DisplayRetirementReceipt,
    NO_DISPLAY_RETIREMENT,
)

@dataclass(slots=True)
class RunResources:
    admission: AdmissionReceipt | None
    directory_session: DirectoryIndexSession | None

    def cleanup(self) -> tuple[tuple[str, Exception], ...]:
        self.admission = None
        failures: list[tuple[str, Exception]] = []
        owner = self.directory_session
        if owner is not None:
            try:
                owner.close()
            except Exception as error:
                failures.append(("directory_session.close", error))
            else:
                self.directory_session = None
        return tuple(failures)

    @property
    def cleaned(self) -> bool:
        return self.directory_session is None


@dataclass(slots=True)
class AdmissionOperation:
    token: AdmissionToken
    capture: StartCapture
    cancelled: Event = field(default_factory=Event)
    directory_session: DirectoryIndexSession | None = None
    result: AdmissionReceipt | AdmissionFailure | None = None
    cleanup_failures: list[DetachedDiagnostic] = field(default_factory=list)
    worker_done: bool = False
    cleanup_in_flight: bool = False
    cleanup_requested: bool = False
    retirement_owner: DisplayRetirementOwner | None = None
    retirement_receipt: DisplayRetirementReceipt = (
        NO_DISPLAY_RETIREMENT
    )
    retirement_release_pending: bool = False
    lock: Lock = field(default_factory=Lock)

    def request_cancel(self) -> None:
        self.cancelled.set()
        with self.lock:
            if type(self.result) is AdmissionReceipt:
                self.result = None
            if self.cleanup_in_flight:
                self.cleanup_requested = True

    def retain_retirement_release(self) -> None:
        """Keep exact cleanup proof until the cancelling caller consumes it."""
        with self.lock:
            if self.retirement_owner is not None:
                self.retirement_release_pending = True

    def consume_retirement_release(self) -> None:
        with self.lock:
            self.retirement_release_pending = False

    def retirement_release_is_pending(self) -> bool:
        with self.lock:
            return self.retirement_release_pending

    def register_directory_session(
        self, session: DirectoryIndexSession | None
    ) -> None:
        if session is None:
            return
        with self.lock:
            if self.directory_session is not None:
                raise RuntimeError("admission already owns a Directory session")
            self.directory_session = session
            if self.cancelled.is_set() and self.cleanup_in_flight:
                self.cleanup_requested = True

    def finish_worker(
        self, result: AdmissionReceipt | AdmissionFailure | None
    ) -> None:
        with self.lock:
            self.worker_done = True
            if self.cleanup_in_flight:
                self.cleanup_requested = True
            if not self.cancelled.is_set():
                self.result = result

    def begin_cleanup(self) -> bool:
        with self.lock:
            if self._terminal_locked():
                return False
            if self.cleanup_in_flight:
                self.cleanup_requested = True
                return False
            self.cleanup_in_flight = True
            self.cleanup_requested = False
            return True

    def cleanup_once(self) -> bool:
        with self.lock:
            session = self.directory_session
        failures: list[tuple[str, Exception]] = []
        if session is not None:
            try:
                session.close()
            except Exception as error:
                failures.append(("directory_session.close", error))
            else:
                with self.lock:
                    if self.directory_session is session:
                        self.directory_session = None
        with self.lock:
            self.cleanup_failures.extend(
                detach_exception(error, context)
                for context, error in failures
            )
        return not failures

    def finish_cleanup(self) -> tuple[bool, bool]:
        with self.lock:
            self.cleanup_in_flight = False
            cleaned = self._terminal_locked()
            requested = self.cleanup_requested
            self.cleanup_requested = False
            return cleaned, requested

    def cleanup_receipt(self) -> AdmissionReleased:
        with self.lock:
            cleaned = self._terminal_locked()
            failures = tuple(self.cleanup_failures)
        return AdmissionReleased(
            self.token,
            (
                CleanupStatus.CLEANED
                if cleaned
                else CleanupStatus.CLEANUP_PENDING
            ),
            failures,
        )

    def _terminal_locked(self) -> bool:
        return (
            self.worker_done
            and not self.cleanup_in_flight
            and self.directory_session is None
        )

    def transfer(self, receipt: AdmissionReceipt) -> RunResources | None:
        with self.lock:
            if (
                self.cancelled.is_set()
                or not self.worker_done
                or self.result is not receipt
                or self.cleanup_in_flight
            ):
                return None
            resources = RunResources(
                receipt, self.directory_session
            )
            self.result = None
            self.directory_session = None
            return resources
