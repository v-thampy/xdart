from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtGui, QtWidgets

from xdart.gui.pages.descriptors import PageDescriptor, ToolDescriptor
from xdart.gui.pages.handle import (
    AppMenuHosts,
    PageHandle,
)
from xdart.gui.pages.services import (
    DiagnosticIdentity,
    ExecutionProfile,
    HostServices,
)
from xdart.gui.pages.values import (
    ActionAccepted,
    ActionCompleted,
    ActionRefused,
    CLEANUP_PENDING,
    EXIT_ONLY_PAGE,
    PAGE_ACTIVE,
    CloseReceipt,
    PageCapability,
    PageCleanup,
    PageKey,
    PageLifecycle,
)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Status:
    def __init__(self):
        self.messages = []

    def show(self, text: str, timeout_ms: int = 0) -> None:
        self.messages.append((text, timeout_ms))


class _RunIntents:
    def __init__(self, calls):
        self.calls = calls

    def store_for(self, key):
        self.calls.append(("intent", key))
        return ("intent", key)


class _Execution:
    def __init__(self, calls):
        self.calls = calls

    def executor_for(self, key):
        self.calls.append(("executor", key))
        return ("executor", key)


class _Sources:
    def __init__(self, calls):
        self.calls = calls

    def source_port_for(self, key):
        self.calls.append(("source", key))
        return ("source", key)


class _Experiments:
    def __init__(self, calls):
        self.calls = calls

    def experiment_for(self, key):
        self.calls.append(("experiment", key))
        return ("experiment", key)


def _services(calls):
    return HostServices(
        status=_Status(),
        run_intents=_RunIntents(calls),
        execution=_Execution(calls),
        sources=_Sources(calls),
        execution_profile=ExecutionProfile.TEST,
        diagnostics=DiagnosticIdentity("tests.synthetic"),
        _experiments=_Experiments(calls),
    )


@dataclass
class _OpenFolder:
    calls: list

    def request(self):
        self.calls.append("open")
        return ActionCompleted("opened")


@dataclass
class _Settings:
    calls: list

    def load(self):
        self.calls.append("load")
        return ActionCompleted("loaded")

    def save(self):
        self.calls.append("save")
        return ActionCompleted("saved")


@dataclass
class _Run:
    calls: list

    def run_pause(self):
        self.calls.append("run")
        return ActionAccepted("run")

    def stop(self):
        self.calls.append("stop")
        return ActionAccepted("stop")


@dataclass
class _Toggle:
    calls: list

    def toggle(self):
        self.calls.append("toggle")
        return ActionCompleted("toggled")


@dataclass
class _Pin:
    calls: list

    def pin(self):
        self.calls.append("pin")
        return ActionCompleted("pinned")


@dataclass
class _Activity:
    state: list

    def active(self):
        return self.state[0]


@dataclass
class _Menus:
    config: QtWidgets.QMenu
    help: QtWidgets.QMenu

    def mount_points(self):
        return AppMenuHosts(self.config, self.help)


@dataclass
class _Diagnostics:
    label: str

    def describe_layout(self):
        return self.label


def _descriptor(
    key,
    built,
    *,
    capabilities=frozenset(),
    lifecycle=PageLifecycle.SWITCHABLE,
    calls=None,
    activity=None,
    receipts=None,
    menus=False,
):
    calls = [] if calls is None else calls
    activity = [False] if activity is None else activity
    receipts = [CloseReceipt(PageCleanup.CLEAN, "verified")] if receipts is None else receipts

    def build(services, parent):
        built.append(key)
        assert services.run_intents.store_for(PageKey(key)) == ("intent", PageKey(key))
        assert services.execution.executor_for(PageKey(key)) == ("executor", PageKey(key))
        assert services.sources.source_port_for(PageKey(key)) == ("source", PageKey(key))
        assert services.experiment_for(PageKey(key)) == ("experiment", PageKey(key))
        assert services.execution.executor_for(PageKey("wrong-key")) is None
        assert services.experiment_for(PageKey("wrong-key")) is None
        widget = QtWidgets.QWidget(parent)
        menu_port = None
        if menus:
            menu_port = _Menus(QtWidgets.QMenu(widget), QtWidgets.QMenu(widget))

        def close():
            receipt = receipts.pop(0) if len(receipts) > 1 else receipts[0]
            calls.append(("close", receipt.status))
            return receipt

        return PageHandle(
            key=PageKey(key),
            widget=widget,
            close=close,
            open_folder=_OpenFolder(calls) if PageCapability.OPEN_FOLDER in capabilities else None,
            settings_io=_Settings(calls) if PageCapability.SETTINGS_PERSISTENCE in capabilities else None,
            run_control=_Run(calls) if PageCapability.RUN_CONTROL in capabilities else None,
            write_mode=_Toggle(calls) if PageCapability.WRITE_MODE_TOGGLE in capabilities else None,
            slice_pin=_Pin(calls) if PageCapability.SLICE_PIN in capabilities else None,
            activity=_Activity(activity) if PageCapability.RUN_ACTIVITY in capabilities else None,
            app_menus=menu_port,
            diagnostics=_Diagnostics(key) if PageCapability.LAYOUT_DIAGNOSTICS in capabilities else None,
        )

    return PageDescriptor(
        key=PageKey(key), label=key, order=0, build=build,
        lifecycle=lifecycle, capabilities=capabilities,
    )


def _tool_descriptor(
    key,
    built,
    *,
    order=0,
    calls=None,
    receipts=None,
    activity=None,
):
    calls = [] if calls is None else calls
    receipts = [CloseReceipt(PageCleanup.CLEAN, "verified")] if receipts is None else receipts

    def build(services, parent):
        built.append(key)
        assert services.run_intents.store_for(PageKey(key)) == (
            "intent", PageKey(key)
        )
        dialog = QtWidgets.QDialog(parent)
        dialog.setObjectName(key)

        def close():
            receipt = receipts.pop(0) if len(receipts) > 1 else receipts[0]
            calls.append(("tool-close", receipt.status))
            return receipt

        return PageHandle(
            key=PageKey(key),
            widget=dialog,
            close=close,
            activity=None if activity is None else _Activity(activity),
        )

    return ToolDescriptor(
        key=PageKey(key),
        label=key,
        order=order,
        build=build,
        tool_kind="analysis",
    )


def _action_texts(menu):
    return [action.text() for action in menu.actions() if not action.isSeparator()]


def test_two_synthetic_pages_mount_lazily_and_route_selected_only(qapp, tmp_path, monkeypatch):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    built, page_a_calls, provider_calls = [], [], []
    caps = frozenset({
        PageCapability.OPEN_FOLDER,
        PageCapability.SETTINGS_PERSISTENCE,
        PageCapability.RUN_CONTROL,
        PageCapability.RUN_ACTIVITY,
        PageCapability.LAYOUT_DIAGNOSTICS,
    })
    page_a = _descriptor("synthetic-a", built, capabilities=caps, calls=page_a_calls)
    page_b = _descriptor("synthetic-b", built)
    window = _gui_main.Main(
        page_descriptors=(page_a, page_b),
        host_services=_services(provider_calls),
        selected_page_key=page_a.key,
    )
    try:
        assert built == ["synthetic-a"]
        assert window.selected_page_key == page_a.key
        assert window.centralWidget() is window.main_widget
        assert provider_calls == [
            ("intent", page_a.key),
            ("executor", page_a.key),
            ("source", page_a.key),
            ("experiment", page_a.key),
        ]
        assert not window.actionToggleWriteMode.isEnabled()
        assert "unavailable" in window.actionToggleWriteMode.toolTip().lower()
        for action in (
            window.ui.actionOpen,
            window.actionLoadSettings,
            window.actionSaveSettings,
            window.actionRunPause,
            window.actionStopRun,
        ):
            action.trigger()
        assert page_a_calls == ["open", "load", "save", "run", "stop"]
        assert built == ["synthetic-a"]
    finally:
        window.close()
        qapp.processEvents()


def test_exit_only_refuses_switch_without_close_but_exit_repolls(qapp, tmp_path, monkeypatch):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    built, calls, provider_calls = [], [], []
    receipts = [
        CloseReceipt(PageCleanup.PENDING, "worker"),
        CloseReceipt(PageCleanup.PENDING, "worker"),
        CloseReceipt(PageCleanup.CLEAN, "verified"),
    ]
    old = _descriptor(
        "exit-page", built, lifecycle=PageLifecycle.EXIT_ONLY,
        calls=calls, receipts=receipts,
    )
    new = _descriptor("new-page", built)
    window = _gui_main.Main(
        page_descriptors=(old, new), host_services=_services(provider_calls),
        selected_page_key=old.key,
    )
    result = window.select_page(new.key)
    assert result == ActionRefused(EXIT_ONLY_PAGE)
    assert calls == []
    assert built == ["exit-page"]

    for accepted in (False, False, True):
        event = QtGui.QCloseEvent()
        window.closeEvent(event)
        assert event.isAccepted() is accepted
    assert [item[0] for item in calls] == ["close", "close", "close"]


def test_switchable_page_blocks_activity_and_pending_then_mounts(qapp, tmp_path, monkeypatch):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    built, calls, provider_calls = [], [], []
    active = [True]
    receipts = [
        CloseReceipt(PageCleanup.PENDING, "worker"),
        CloseReceipt(PageCleanup.CLEAN, "verified"),
    ]
    caps = frozenset({PageCapability.RUN_ACTIVITY})
    old = _descriptor(
        "switchable", built, capabilities=caps, calls=calls,
        activity=active, receipts=receipts,
    )
    new = _descriptor("destination", built)
    window = _gui_main.Main(
        page_descriptors=(old, new), host_services=_services(provider_calls),
        selected_page_key=old.key,
    )
    assert window.select_page(new.key) == ActionRefused(PAGE_ACTIVE)
    assert calls == []
    active[0] = False
    assert window.select_page(new.key) == ActionRefused(CLEANUP_PENDING)
    assert built == ["switchable"]
    assert window.select_page(new.key) == ActionCompleted("destination")
    assert built == ["switchable", "destination"]
    window.close()


def test_application_menu_actions_move_by_identity_and_never_rebuild_page(qapp, tmp_path, monkeypatch):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    built, provider_calls = [], []
    plain = _descriptor("plain-page", built)
    window = _gui_main.Main(
        page_descriptors=(plain,), host_services=_services(provider_calls),
        selected_page_key=plain.key,
    )
    try:
        widget = window.main_widget
        config_actions = tuple(window.application_config_actions)
        help_actions = tuple(window.application_help_actions)
        assert all(action in window.host_config_menu.actions() for action in config_actions)
        assert all(action in window.host_help_menu.actions() for action in help_actions)
        window._set_theme("light")
        window._set_application_font_size("large")
        assert window.main_widget is widget
        assert window.selected_page_key == plain.key
        assert built == ["plain-page"]
    finally:
        window.close()

    built.clear()
    mounted = _descriptor(
        "mounted-page", built,
        capabilities=frozenset({PageCapability.APP_MENU_HOSTS}), menus=True,
    )
    fallback = _descriptor("fallback-page", built)
    window = _gui_main.Main(
        page_descriptors=(mounted, fallback), host_services=_services([]),
        selected_page_key=mounted.key,
    )
    try:
        points = window.page_handle.app_menus.mount_points()
        config_actions = tuple(window.application_config_actions)
        help_actions = tuple(window.application_help_actions)
        assert all(action in points.config_menu.actions() for action in config_actions)
        assert all(action in points.help_menu.actions() for action in help_actions)
        assert all(action not in window.host_config_menu.actions() for action in config_actions)
        assert all(action not in window.host_help_menu.actions() for action in help_actions)
        assert window.select_page(fallback.key) == ActionCompleted("fallback-page")
        assert tuple(window.application_config_actions) == config_actions
        assert tuple(window.application_help_actions) == help_actions
        assert all(action in window.host_config_menu.actions() for action in config_actions)
        assert all(action in window.host_help_menu.actions() for action in help_actions)
    finally:
        window.close()


def test_unknown_or_tool_selection_falls_back_without_constructing_it(qapp, tmp_path, monkeypatch):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    built, provider_calls = [], []
    default = _descriptor("default-page", built)

    def build_tool(_services, _parent):
        built.append("tool-built")
        raise AssertionError("a tool is not a selectable page")

    tool = ToolDescriptor(
        key=PageKey("registered-tool"), label="Tool", order=0,
        build=build_tool, tool_kind="analysis",
    )
    window = _gui_main.Main(
        page_descriptors=(tool, default),
        host_services=_services(provider_calls),
        selected_page_key=tool.key,
    )
    try:
        assert window.selected_page_key == default.key
        assert built == ["default-page"]
    finally:
        window.close()


def test_analysis_tools_build_lazily_and_reuse_one_hidden_dialog(
    qapp,
    tmp_path,
    monkeypatch,
):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    built, provider_calls, tool_calls = [], [], []
    page = _descriptor("default-page", built)
    later = _tool_descriptor(
        "later-tool", built, order=2, calls=tool_calls
    )
    first = _tool_descriptor(
        "first-tool", built, order=1, calls=tool_calls
    )
    window = _gui_main.Main(
        page_descriptors=(later, page, first),
        host_services=_services(provider_calls),
    )
    try:
        assert built == ["default-page"]
        assert _action_texts(window.ui.menuAnalysis) == [
            "first-tool",
            "later-tool",
        ]
        assert window.open_tool(first.key) == ActionCompleted("first-tool")
        handle = window._tool_handles[first.key]
        dialog = handle.widget
        assert built == ["default-page", "first-tool"]
        assert dialog.isVisible()

        dialog.close()
        qapp.processEvents()
        assert not dialog.isVisible()
        assert window.open_tool(first.key) == ActionCompleted("first-tool")
        assert window._tool_handles[first.key] is handle
        assert built == ["default-page", "first-tool"]
    finally:
        window.close()
    assert [call[0] for call in tool_calls] == ["tool-close"]


def test_exit_aggregates_page_and_built_tool_cleanup_every_pass(
    qapp,
    tmp_path,
    monkeypatch,
):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    built, page_calls, tool_calls = [], [], []
    page = _descriptor(
        "default-page",
        built,
        calls=page_calls,
        receipts=[
            CloseReceipt(PageCleanup.PENDING, "page"),
            CloseReceipt(PageCleanup.CLEAN, "page"),
        ],
    )
    tool = _tool_descriptor(
        "analysis-tool",
        built,
        calls=tool_calls,
        receipts=[
            CloseReceipt(PageCleanup.PENDING, "tool"),
            CloseReceipt(PageCleanup.CLEAN, "tool"),
        ],
    )
    window = _gui_main.Main(
        page_descriptors=(page, tool),
        host_services=_services([]),
    )
    window.open_tool(tool.key)

    first = QtGui.QCloseEvent()
    window.closeEvent(first)
    assert not first.isAccepted()
    second = QtGui.QCloseEvent()
    window.closeEvent(second)
    assert second.isAccepted()
    assert [call[0] for call in page_calls] == ["close", "close"]
    assert [call[0] for call in tool_calls] == ["tool-close", "tool-close"]


def test_tool_activity_blocks_updates_but_not_page_switching(
    qapp,
    tmp_path,
    monkeypatch,
):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    built = []
    first_page = _descriptor("first-page", built)
    next_page = _descriptor("next-page", built)
    active = [True]
    tool = _tool_descriptor("analysis-tool", built, activity=active)
    window = _gui_main.Main(
        page_descriptors=(first_page, next_page, tool),
        host_services=_services([]),
        selected_page_key=first_page.key,
    )
    try:
        window.open_tool(tool.key)
        assert window._run_active() is True
        assert window.select_page(next_page.key) == ActionCompleted("next-page")
        active[0] = False
        assert window._run_active() is False
    finally:
        window.close()


def test_unbuilt_tool_is_never_constructed_or_closed(qapp, tmp_path, monkeypatch):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    built, calls = [], []
    page = _descriptor("default-page", built)
    tool = _tool_descriptor("analysis-tool", built, calls=calls)
    window = _gui_main.Main(
        page_descriptors=(page, tool),
        host_services=_services([]),
    )
    window.close()
    assert built == ["default-page"]
    assert calls == []


def test_host_owned_application_actions_stay_enabled_during_page_activity(qapp, tmp_path, monkeypatch):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    built = []
    descriptor = _descriptor(
        "active-page", built,
        capabilities=frozenset({PageCapability.RUN_ACTIVITY}),
        activity=[True],
    )
    window = _gui_main.Main(
        page_descriptors=(descriptor,), host_services=_services([]),
        selected_page_key=descriptor.key,
    )
    try:
        actions = (
            *window.application_config_actions,
            *window.application_help_actions,
        )
        assert all(action.isEnabled() for action in actions)
        assert all(action in window.host_config_menu.actions()
                   for action in window.application_config_actions)
        before = window.main_widget
        window._set_theme("light")
        assert window.main_widget is before
    finally:
        window.close()


def test_absent_activity_allows_check_only_but_blocks_update_on_exit(qapp, tmp_path, monkeypatch):
    from xdart import _gui_main

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    descriptor = _descriptor("no-activity", [])
    window = _gui_main.Main(
        page_descriptors=(descriptor,), host_services=_services([]),
        selected_page_key=descriptor.key,
    )
    try:
        assert window._run_active() is False
        assert window._run_active(require_known=True) is True
    finally:
        window.close()
