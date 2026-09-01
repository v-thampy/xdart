"""Closed, Qt-free values shared by the page registry and application host."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType, TypeAlias


PageKey = NewType("PageKey", str)
SCATTERING_PAGE_KEY = PageKey("scattering-workspace")
STITCH_TOOL_KEY = PageKey("stitch")
RSM_TOOL_KEY = PageKey("rsm")


class PageCapability(str, Enum):
    OPEN_FOLDER = "open-folder"
    SETTINGS_PERSISTENCE = "settings-persistence"
    RUN_CONTROL = "run-control"
    WRITE_MODE_TOGGLE = "write-mode-toggle"
    SLICE_PIN = "slice-pin"
    RUN_ACTIVITY = "run-activity"
    APP_MENU_HOSTS = "app-menu-hosts"
    LAYOUT_DIAGNOSTICS = "layout-diagnostics"


class PageLifecycle(str, Enum):
    SWITCHABLE = "switchable"
    EXIT_ONLY = "exit-only"


class PageCleanup(str, Enum):
    CLEAN = "clean"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class ActionAccepted:
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ActionRefused:
    reason: str


@dataclass(frozen=True, slots=True)
class ActionCompleted:
    detail: str = ""


ActionOutcome: TypeAlias = ActionAccepted | ActionRefused | ActionCompleted


@dataclass(frozen=True, slots=True)
class CloseReceipt:
    status: PageCleanup
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, PageCleanup):
            raise TypeError("close receipt status must be PageCleanup")


CAPABILITY_UNAVAILABLE = "capability-unavailable"
APPEND_UNAVAILABLE = "append-unavailable"
EXIT_ONLY_PAGE = "exit-only-page"
PAGE_ACTIVE = "page-active"
CLEANUP_PENDING = "cleanup-pending"
UNKNOWN_PAGE = "unknown-page"
