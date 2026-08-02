from __future__ import annotations

from types import SimpleNamespace

from xdart.gui.pages.catalog import DEFAULT_PAGE_KEY, LEGACY_STATIC_PAGE
from xdart.gui.pages.legacy_static import build_legacy_static
from xdart.gui.pages.services import (
    DiagnosticIdentity,
    ExecutionProfile,
    HostServices,
)
from xdart.gui.pages.values import PageCapability, PageCleanup, PageLifecycle


class _NoopStatus:
    def show(self, _text, timeout_ms=0):
        return None


class _NoneIntents:
    def store_for(self, _key):
        return None


class _NoneExecution:
    def executor_for(self, _key):
        return None


class _NoneSources:
    def source_port_for(self, _key):
        return None


def _services(profile=ExecutionProfile.TEST):
    return HostServices(
        status=_NoopStatus(), run_intents=_NoneIntents(),
        execution=_NoneExecution(), sources=_NoneSources(),
        execution_profile=profile,
        diagnostics=DiagnosticIdentity("tests.legacy"),
    )


class _Worker:
    def __init__(self):
        self.running = False

    def isRunning(self):
        return self.running


class _Timer:
    def __init__(self):
        self.active = False

    def isActive(self):
        return self.active


class _FakeLegacy:
    def __init__(self, _parent=None):
        self.calls = []
        self.displayframe = SimpleNamespace(_processing_active=False)
        self.worker = _Worker()
        self.timer = _Timer()
        self.wrangler = SimpleNamespace(thread=self.worker, timers=(self.timer,))
        self.integratorTree = SimpleNamespace(integrator_thread=_Worker())
        self.stitch_thread = _Worker()
        self.ui = SimpleNamespace(
            leftFrame=object(), middleFrame=object(), rightFrame=object())

    def enable_async_hydration(self):
        self.calls.append("hydrate")

    def close(self):
        self.calls.append("close")

    def shortcut_run_pause(self):
        self.calls.append("run")

    def shortcut_stop(self):
        self.calls.append("stop")

    def shortcut_toggle_write_mode(self):
        self.calls.append("toggle")

    def shortcut_pin_slice_cut(self):
        self.calls.append("pin")

    def shortcut_load_settings(self):
        self.calls.append("load")

    def shortcut_save_settings(self):
        self.calls.append("save")


def test_legacy_descriptor_is_lazy_exit_only_and_keeps_app_menus_host_owned():
    assert LEGACY_STATIC_PAGE.key == DEFAULT_PAGE_KEY
    assert LEGACY_STATIC_PAGE.lifecycle is PageLifecycle.EXIT_ONLY
    assert PageCapability.OPEN_FOLDER not in LEGACY_STATIC_PAGE.capabilities
    assert PageCapability.APP_MENU_HOSTS not in LEGACY_STATIC_PAGE.capabilities
    assert LEGACY_STATIC_PAGE.capabilities == frozenset({
        PageCapability.SETTINGS_PERSISTENCE,
        PageCapability.RUN_CONTROL,
        PageCapability.WRITE_MODE_TOGGLE,
        PageCapability.SLICE_PIN,
        PageCapability.RUN_ACTIVITY,
        PageCapability.LAYOUT_DIAGNOSTICS,
    })


def test_legacy_adapter_routes_declared_ports_and_live_profile_hydration():
    handle = build_legacy_static(
        _services(ExecutionProfile.LIVE), None, _widget_factory=_FakeLegacy)
    widget = handle.widget
    assert widget.calls == ["hydrate"]
    handle.settings_io.load()
    handle.settings_io.save()
    handle.run_control.run_pause()
    handle.run_control.stop()
    handle.write_mode.toggle()
    handle.slice_pin.pin()
    assert widget.calls == [
        "hydrate", "load", "save", "run", "stop", "toggle", "pin"]
    assert handle.app_menus is None
    assert handle.open_folder is None


def test_legacy_close_receipt_is_verified_and_latched_not_hard_coded_clean():
    handle = build_legacy_static(_services(), None, _widget_factory=_FakeLegacy)
    widget = handle.widget
    widget.displayframe._processing_active = True
    widget.worker.running = True
    widget.timer.active = True
    first = handle.close()
    assert first.status is PageCleanup.PENDING
    assert widget.calls == ["close"]
    widget.displayframe._processing_active = False
    widget.worker.running = False
    assert handle.close().status is PageCleanup.PENDING
    assert widget.calls == ["close"]
    widget.timer.active = False
    clean = handle.close()
    assert clean.status is PageCleanup.CLEAN
    assert handle.close() is clean
    assert widget.calls == ["close"]
