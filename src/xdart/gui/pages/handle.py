"""Mounted-page handle and its closed set of optional capability ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TYPE_CHECKING

from .values import ActionOutcome, CloseReceipt, PageCapability, PageKey

if TYPE_CHECKING:
    from pyqtgraph import QtWidgets
    from .descriptors import PageDescriptor, ToolDescriptor


class OpenFolderPort(Protocol):
    def request(self) -> ActionOutcome: ...


class SettingsPersistencePort(Protocol):
    def load(self) -> ActionOutcome: ...
    def save(self) -> ActionOutcome: ...


class RunControlPort(Protocol):
    def run_pause(self) -> ActionOutcome: ...
    def stop(self) -> ActionOutcome: ...


class WriteModePort(Protocol):
    def toggle(self) -> ActionOutcome: ...


class SlicePinPort(Protocol):
    def pin(self) -> ActionOutcome: ...


class RunActivityPort(Protocol):
    def active(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class AppMenuHosts:
    config_menu: "QtWidgets.QMenu"
    help_menu: "QtWidgets.QMenu"


class AppMenusPort(Protocol):
    def mount_points(self) -> AppMenuHosts: ...


class LayoutDiagnosticsPort(Protocol):
    def describe_layout(self) -> str: ...


@dataclass(frozen=True, slots=True)
class PageHandle:
    key: PageKey
    widget: "QtWidgets.QWidget"
    close: Callable[[], CloseReceipt]
    open_folder: OpenFolderPort | None = None
    settings_io: SettingsPersistencePort | None = None
    run_control: RunControlPort | None = None
    write_mode: WriteModePort | None = None
    slice_pin: SlicePinPort | None = None
    activity: RunActivityPort | None = None
    app_menus: AppMenusPort | None = None
    diagnostics: LayoutDiagnosticsPort | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("page handle key must be nonempty")
        if self.widget is None:
            raise ValueError("page handle widget is mandatory")
        if not callable(self.close):
            raise TypeError("page handle close must be callable")


class CapabilityContractError(ValueError):
    pass


_PORTS = {
    PageCapability.OPEN_FOLDER: "open_folder",
    PageCapability.SETTINGS_PERSISTENCE: "settings_io",
    PageCapability.RUN_CONTROL: "run_control",
    PageCapability.WRITE_MODE_TOGGLE: "write_mode",
    PageCapability.SLICE_PIN: "slice_pin",
    PageCapability.RUN_ACTIVITY: "activity",
    PageCapability.APP_MENU_HOSTS: "app_menus",
    PageCapability.LAYOUT_DIAGNOSTICS: "diagnostics",
}


def validate_page_handle(descriptor: "PageDescriptor", handle: PageHandle) -> None:
    if handle.key != descriptor.key:
        raise CapabilityContractError(
            f"handle key {handle.key!r} != descriptor key {descriptor.key!r}")
    for capability, field_name in _PORTS.items():
        present = object.__getattribute__(handle, field_name) is not None
        declared = capability in descriptor.capabilities
        if present != declared:
            raise CapabilityContractError(
                f"{descriptor.key}: {field_name} presence does not match "
                f"{capability.name}")


def validate_tool_handle(descriptor: "ToolDescriptor", handle: PageHandle) -> None:
    """Validate the deliberately small standalone-tool handle surface.

    Tools own their dialog controls and may expose activity to the application
    updater/exit guard.  Page-selection commands and application-menu mount
    points remain page-only so opening an analysis tool cannot silently acquire
    workspace authority.
    """

    if handle.key != descriptor.key:
        raise CapabilityContractError(
            f"handle key {handle.key!r} != descriptor key {descriptor.key!r}"
        )
    forbidden = (
        "open_folder",
        "settings_io",
        "run_control",
        "write_mode",
        "slice_pin",
        "app_menus",
        "diagnostics",
    )
    for field_name in forbidden:
        if object.__getattribute__(handle, field_name) is not None:
            raise CapabilityContractError(
                f"{descriptor.key}: standalone tool cannot expose {field_name}"
            )
