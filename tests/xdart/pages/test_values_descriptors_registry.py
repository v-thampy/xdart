from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from xdart.gui.pages.descriptors import PageDescriptor, ToolDescriptor
from xdart.gui.pages.handle import (
    CapabilityContractError,
    PageHandle,
    validate_page_handle,
    validate_tool_handle,
)
from xdart.gui.pages.registry import (
    DuplicatePageKeyError,
    PageRegistry,
    RegistryFrozenError,
)
from xdart.gui.pages.values import (
    ActionAccepted,
    ActionCompleted,
    ActionRefused,
    CloseReceipt,
    PageCapability,
    PageCleanup,
    PageKey,
    PageLifecycle,
)


def _build(_services, _parent):
    raise AssertionError("descriptor factories are lazy")


def _page(key: str, *, category: str = "workspace", order: int = 0):
    return PageDescriptor(
        key=PageKey(key),
        label=key,
        order=order,
        build=_build,
        lifecycle=PageLifecycle.SWITCHABLE,
        capabilities=frozenset(),
        category=category,
    )


def test_closed_values_and_frozen_outcomes_are_exact():
    assert tuple(PageCapability) == (
        PageCapability.OPEN_FOLDER,
        PageCapability.SETTINGS_PERSISTENCE,
        PageCapability.RUN_CONTROL,
        PageCapability.WRITE_MODE_TOGGLE,
        PageCapability.SLICE_PIN,
        PageCapability.RUN_ACTIVITY,
        PageCapability.APP_MENU_HOSTS,
        PageCapability.LAYOUT_DIAGNOSTICS,
    )
    assert tuple(PageLifecycle) == (
        PageLifecycle.SWITCHABLE,
        PageLifecycle.EXIT_ONLY,
    )
    assert tuple(PageCleanup) == (PageCleanup.CLEAN, PageCleanup.PENDING)
    assert ActionAccepted("queued").detail == "queued"
    assert ActionRefused("unavailable").reason == "unavailable"
    assert ActionCompleted("done").detail == "done"
    receipt = CloseReceipt(PageCleanup.CLEAN, "verified")
    with pytest.raises(FrozenInstanceError):
        receipt.detail = "changed"


def test_descriptor_schemas_are_frozen_science_free_and_validated():
    assert {field.name for field in fields(PageDescriptor)} == {
        "key", "label", "order", "build", "lifecycle", "capabilities",
        "description", "icon_key", "category",
    }
    assert {field.name for field in fields(ToolDescriptor)} == {
        "key", "label", "order", "build", "tool_kind", "description",
        "icon_key", "category",
    }
    descriptor = _page("synthetic-a")
    with pytest.raises(FrozenInstanceError):
        descriptor.label = "changed"
    for invalid in ("", "Synthetic", "two words", "1-starts-with-number"):
        with pytest.raises(ValueError):
            _page(invalid)
    with pytest.raises(TypeError):
        PageDescriptor(
            key=PageKey("bad-caps"), label="bad", order=0, build=_build,
            lifecycle=PageLifecycle.SWITCHABLE,
            capabilities={PageCapability.RUN_CONTROL},
        )


def test_registry_refuses_page_and_tool_collisions_before_construction():
    builds = []

    def build(_services, _parent):
        builds.append("built")

    first = PageDescriptor(
        key=PageKey("synthetic-a"), label="A", order=0, build=build,
        lifecycle=PageLifecycle.SWITCHABLE, capabilities=frozenset(),
    )
    collision = ToolDescriptor(
        key=PageKey("synthetic-a"), label="Tool", order=0, build=build,
        tool_kind="analysis",
    )
    registry = PageRegistry()
    registry.register(first)
    with pytest.raises(DuplicatePageKeyError):
        registry.register(first)
    with pytest.raises(DuplicatePageKeyError):
        registry.register(collision)
    assert builds == []


def test_registry_freezes_and_iterates_in_deterministic_total_order():
    registry = PageRegistry()
    for descriptor in (
        _page("z-last", category="workspace", order=5),
        _page("b-key", category="analysis", order=2),
        _page("a-key", category="analysis", order=2),
        _page("first", category="analysis", order=1),
    ):
        registry.register(descriptor)
    registry.freeze()
    expected = ["first", "a-key", "b-key", "z-last"]
    assert [str(item.key) for item in registry] == expected
    assert [str(item.key) for item in registry] == expected
    assert registry.get(PageKey("a-key")).label == "a-key"
    with pytest.raises(RegistryFrozenError):
        registry.register(_page("too-late"))


def test_unknown_selection_falls_closed_without_construction():
    builds = []

    def default_build(_services, _parent):
        builds.append("default")

    default = PageDescriptor(
        key=PageKey("default-page"), label="Default", order=0,
        build=default_build, lifecycle=PageLifecycle.EXIT_ONLY,
        capabilities=frozenset(),
    )
    registry = PageRegistry((default, _page("other-page")))
    registry.freeze()
    selected = registry.select(PageKey("deleted-page"), default.key)
    assert selected is default
    assert builds == []


def test_handle_presence_must_match_every_declared_capability():
    descriptor = PageDescriptor(
        key=PageKey("declared-run"), label="Declared Run", order=0,
        build=_build, lifecycle=PageLifecycle.SWITCHABLE,
        capabilities=frozenset({PageCapability.RUN_CONTROL}),
    )
    handle = PageHandle(
        key=descriptor.key, widget=object(),
        close=lambda: CloseReceipt(PageCleanup.CLEAN, "verified"),
    )
    with pytest.raises(CapabilityContractError, match="run_control"):
        validate_page_handle(descriptor, handle)


def test_tool_handle_allows_only_activity_beside_widget_and_close():
    descriptor = ToolDescriptor(
        key=PageKey("analysis-tool"),
        label="Analysis Tool",
        order=0,
        build=_build,
        tool_kind="analysis",
    )
    valid = PageHandle(
        key=descriptor.key,
        widget=object(),
        close=lambda: CloseReceipt(PageCleanup.CLEAN, "verified"),
        activity=object(),
    )
    validate_tool_handle(descriptor, valid)

    invalid = PageHandle(
        key=descriptor.key,
        widget=object(),
        close=lambda: CloseReceipt(PageCleanup.CLEAN, "verified"),
        run_control=object(),
    )
    with pytest.raises(CapabilityContractError, match="run_control"):
        validate_tool_handle(descriptor, invalid)
