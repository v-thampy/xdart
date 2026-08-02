"""Frozen, construction-free page and tool registry descriptors."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, TYPE_CHECKING, TypeAlias

from .values import PageCapability, PageKey, PageLifecycle

if TYPE_CHECKING:
    from pyqtgraph import QtWidgets
    from .handle import PageHandle
    from .services import HostServices


_KEY = re.compile(r"^[a-z][a-z0-9-]*$")
PageFactory: TypeAlias = Callable[
    ["HostServices", "QtWidgets.QWidget | None"], "PageHandle"
]


def _validate_common(key: PageKey, label: str, order: int, build: PageFactory) -> None:
    if not isinstance(key, str) or _KEY.fullmatch(key) is None:
        raise ValueError("descriptor key must match [a-z][a-z0-9-]*")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("descriptor label must be nonempty")
    if not isinstance(order, int) or isinstance(order, bool):
        raise TypeError("descriptor order must be an integer")
    if not callable(build):
        raise TypeError("descriptor build must be callable")


@dataclass(frozen=True, slots=True)
class PageDescriptor:
    key: PageKey
    label: str
    order: int
    build: PageFactory
    lifecycle: PageLifecycle
    capabilities: frozenset[PageCapability]
    description: str = ""
    icon_key: str = ""
    category: str = "workspace"

    def __post_init__(self) -> None:
        _validate_common(self.key, self.label, self.order, self.build)
        if not isinstance(self.lifecycle, PageLifecycle):
            raise TypeError("page lifecycle must be PageLifecycle")
        if type(self.capabilities) is not frozenset:
            raise TypeError("page capabilities must be an exact frozenset")
        if any(not isinstance(item, PageCapability) for item in self.capabilities):
            raise TypeError("page capabilities contain an unknown value")


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    key: PageKey
    label: str
    order: int
    build: PageFactory
    tool_kind: str
    description: str = ""
    icon_key: str = ""
    category: str = "tool"

    def __post_init__(self) -> None:
        _validate_common(self.key, self.label, self.order, self.build)
        if not isinstance(self.tool_kind, str) or not self.tool_kind.strip():
            raise ValueError("tool kind must be nonempty")


Descriptor: TypeAlias = PageDescriptor | ToolDescriptor
