"""Exact single-owner retirement prerequisite for a historical display run."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable

from .events import (
    CleanupStatus,
    ExecutorClosed,
    RunIdentity,
    detach_exception,
)


@dataclass(frozen=True, slots=True)
class DisplayRetirementReceipt:
    """Values-only proof that one prior display was absent or retired."""

    run_identity: RunIdentity | None
    cleanup_status: CleanupStatus

    def __post_init__(self) -> None:
        if (
            self.run_identity is not None
            and type(self.run_identity) is not RunIdentity
        ):
            raise TypeError("display retirement identity is invalid")
        if type(self.cleanup_status) is not CleanupStatus:
            raise TypeError("display retirement status is invalid")
        if (
            self.run_identity is None
            and self.cleanup_status is not CleanupStatus.CLEANED
        ):
            raise ValueError("an absent display must be clean")


NO_DISPLAY_RETIREMENT = DisplayRetirementReceipt(
    None, CleanupStatus.CLEANED
)


def apply_display_retirement(
    value: object,
    current: RunIdentity | None,
) -> tuple[bool, RunIdentity | None]:
    """Consume only exact clean proof for the page's current display."""

    try:
        accepted = (
            type(value) is DisplayRetirementReceipt
            and value.cleanup_status is CleanupStatus.CLEANED
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
            self.run_identity, receipt.cleanup_status
        )

    def attempt(self) -> ExecutorClosed:
        with self._lock:
            receipt = self._receipt
            if (
                receipt is not None
                and receipt.cleanup_status is CleanupStatus.CLEANED
            ):
                return receipt
            if self._in_flight:
                return ExecutorClosed(
                    self.run_identity, CleanupStatus.CLEANUP_PENDING
                )
            self._in_flight = True
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
        )
    except Exception:
        return False


__all__ = [
    "DisplayRetirementOwner",
    "DisplayRetirementReceipt",
    "NO_DISPLAY_RETIREMENT",
    "apply_display_retirement",
]
