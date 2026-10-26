"""Exact single-owner retirement prerequisite for a historical display run."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, Thread
from typing import Callable

from .events import (
    CleanupStatus,
    ExecutorClosed,
    RunIdentity,
    detach_exception,
)


@dataclass(frozen=True, slots=True)
class DisplayRetirementReceipt:
    """Exact prior display disposition: absent, retired, or explicitly quarantined with resources still pending."""

    run_identity: RunIdentity | None
    cleanup_status: CleanupStatus
    quarantined: bool = False

    def __post_init__(self) -> None:
        if (
            self.run_identity is not None
            and type(self.run_identity) is not RunIdentity
        ):
            raise TypeError("display retirement identity is invalid")
        if type(self.cleanup_status) is not CleanupStatus:
            raise TypeError("display retirement status is invalid")
        if type(self.quarantined) is not bool or (
            self.quarantined
            and (self.run_identity is None or self.cleanup_status is not CleanupStatus.CLEANUP_PENDING)
        ):
            raise ValueError("display quarantine is invalid")

        if (
            self.run_identity is None
            and self.cleanup_status is not CleanupStatus.CLEANED
        ):
            raise ValueError("an absent display must be clean")

    @property
    def released_for_admission(self) -> bool:
        return self.cleanup_status is CleanupStatus.CLEANED or self.quarantined


NO_DISPLAY_RETIREMENT = DisplayRetirementReceipt(
    None, CleanupStatus.CLEANED
)


def apply_display_retirement(
    value: object,
    current: RunIdentity | None,
) -> tuple[bool, RunIdentity | None]:
    """Consume only the exact retired or explicitly quarantined disposition for the current display."""

    try:
        accepted = (
            type(value) is DisplayRetirementReceipt
            and value.released_for_admission
            and value.run_identity is current
        )
    except Exception:
        accepted = False
    return accepted, None if accepted else current


class DisplayRetirementOwner:
    """Serialize every command for one exact historical run."""

    def __init__(
        self,
        run_identity: RunIdentity,
        close: Callable[[], ExecutorClosed],
    ) -> None:
        if type(run_identity) is not RunIdentity or not callable(close):
            raise TypeError("display retirement owner inputs are invalid")
        self.run_identity = run_identity
        self._close = close
        self._lock = Lock()
        self._in_flight = False
        self._receipt: ExecutorClosed | None = None

    @property
    def receipt(self) -> ExecutorClosed:
        with self._lock:
            receipt = self._receipt
        return (
            receipt
            if receipt is not None
            else ExecutorClosed(
                self.run_identity, CleanupStatus.CLEANUP_PENDING
            )
        )

    @property
    def proof(self) -> DisplayRetirementReceipt:
        receipt = self.receipt
        return DisplayRetirementReceipt(
            self.run_identity, receipt.cleanup_status, receipt.quarantined
        )

    def quarantine(self, receipt: ExecutorClosed) -> bool:
        """Acknowledge explicit abandonment without replacing an in-flight retirement or claiming cleanup."""
        with self._lock:
            if (
                self._in_flight
                or not _receipt_is_exact(receipt, self.run_identity)
                or not receipt.quarantined
            ):
                return False
            self._receipt = receipt
            return True

    def attempt(self, *, background: bool = False) -> ExecutorClosed:
        with self._lock:
            receipt = self._receipt
            if (
                receipt is not None
                and (receipt.cleanup_status is CleanupStatus.CLEANED or receipt.quarantined)
            ):
                return receipt
            if self._in_flight:
                return ExecutorClosed(
                    self.run_identity, CleanupStatus.CLEANUP_PENDING
                )
            self._in_flight = True
        if background:
            # Claim single-flight ownership before dispatch. GUI polls receive
            # only this owner's exact receipt; close and its waits run here.
            try:
                Thread(target=self._finish_attempt, name="scattering-display-retire", daemon=True).start()
            except Exception as error:
                with self._lock:
                    self._in_flight = False
                    self._receipt = ExecutorClosed(
                        self.run_identity, CleanupStatus.CLEANUP_PENDING,
                        detach_exception(error, "display.retire.start"))
            return self.receipt
        return self._finish_attempt()

    def _finish_attempt(self) -> ExecutorClosed:
        try:
            receipt = self._close()
        except Exception as error:
            receipt = ExecutorClosed(
                self.run_identity,
                CleanupStatus.CLEANUP_PENDING,
                detach_exception(error, "display.retire"),
            )
        with self._lock:
            self._in_flight = False
            if _receipt_is_exact(receipt, self.run_identity):
                self._receipt = receipt
                return receipt
        return ExecutorClosed(
            self.run_identity, CleanupStatus.CLEANUP_PENDING
        )


def _receipt_is_exact(
    value: object, expected: RunIdentity
) -> bool:
    try:
        return (
            type(value) is ExecutorClosed
            and value.run_identity is expected
            and type(value.cleanup_status) is CleanupStatus
            and value.cleanup_status is not CleanupStatus.NOT_STARTED
            and type(value.quarantined) is bool
            and (not value.quarantined or value.cleanup_status is CleanupStatus.CLEANUP_PENDING)
        )
    except Exception:
        return False


__all__ = [
    "DisplayRetirementOwner",
    "DisplayRetirementReceipt",
    "NO_DISPLAY_RETIREMENT",
    "apply_display_retirement",
]
