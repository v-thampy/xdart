"""Lazy adapter from the legacy static page to the generic page handle."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from .handle import PageHandle
from .services import ExecutionProfile, HostServices
from .values import (
    ActionAccepted,
    ActionCompleted,
    CloseReceipt,
    PageCleanup,
    STATIC_SCAN_PAGE_KEY,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SettingsIO:
    load_call: Callable[[], None]
    save_call: Callable[[], None]

    def load(self):
        self.load_call()
        return ActionCompleted("loaded")

    def save(self):
        self.save_call()
        return ActionCompleted("saved")


@dataclass(frozen=True, slots=True)
class _RunControl:
    run_call: Callable[[], None]
    stop_call: Callable[[], None]

    def run_pause(self):
        self.run_call()
        return ActionAccepted("run-pause")

    def stop(self):
        self.stop_call()
        return ActionAccepted("stop")


@dataclass(frozen=True, slots=True)
class _WriteMode:
    call: Callable[[], None]

    def toggle(self):
        self.call()
        return ActionCompleted("write-mode-toggled")


@dataclass(frozen=True, slots=True)
class _SlicePin:
    call: Callable[[], None]

    def pin(self):
        self.call()
        return ActionCompleted("slice-pinned")


@dataclass(frozen=True, slots=True)
class _Activity:
    query: Callable[[], bool]

    def active(self):
        try:
            return bool(self.query())
        except Exception:
            return True


@dataclass(frozen=True, slots=True)
class _Diagnostics:
    describe: Callable[[], str]

    def describe_layout(self):
        return self.describe()


def _is_running(owner: object) -> bool:
    try:
        return bool(owner.isRunning())
    except (AttributeError, RuntimeError):
        return False


def _is_active(owner: object) -> bool:
    try:
        return bool(owner.isActive())
    except (AttributeError, RuntimeError):
        return False


def _watched_owners(widget: object) -> tuple[tuple[object, ...], tuple[object, ...]]:
    threads, timers = [], []
    wrangler = getattr(widget, "wrangler", None)
    for thread in (
        getattr(wrangler, "thread", None),
        getattr(getattr(widget, "integratorTree", None), "integrator_thread", None),
        getattr(widget, "stitch_thread", None),
    ):
        if thread is not None:
            threads.append(thread)
    timers.extend(getattr(wrangler, "timers", ()) or ())
    try:
        from pyqtgraph.Qt import QtCore
        timers.extend(wrangler.findChildren(QtCore.QTimer))
    except (AttributeError, RuntimeError):
        pass
    return tuple(threads), tuple(dict.fromkeys(timers))


@dataclass(slots=True)
class _LegacyCloser:
    widget: object
    close_call: Callable[[], None]
    activity: _Activity
    close_started: bool = False
    watched_threads: tuple[object, ...] = field(default_factory=tuple)
    watched_timers: tuple[object, ...] = field(default_factory=tuple)
    clean_receipt: CloseReceipt | None = None

    def __call__(self) -> CloseReceipt:
        if self.clean_receipt is not None:
            return self.clean_receipt
        if not self.close_started:
            self.watched_threads, self.watched_timers = _watched_owners(self.widget)
            try:
                self.close_call()
            except Exception as exc:
                return CloseReceipt(PageCleanup.PENDING, f"close failed: {exc}")
            self.close_started = True
        if self.activity.active():
            return CloseReceipt(PageCleanup.PENDING, "legacy run is active")
        if any(_is_running(owner) for owner in self.watched_threads):
            return CloseReceipt(PageCleanup.PENDING, "legacy worker is running")
        if any(_is_active(owner) for owner in self.watched_timers):
            return CloseReceipt(PageCleanup.PENDING, "legacy timer is active")
        self.clean_receipt = CloseReceipt(PageCleanup.CLEAN, "verified quiescent")
        return self.clean_receipt


def build_legacy_static(
    services: HostServices,
    parent,
    *,
    _widget_factory=None,
) -> PageHandle:
    if _widget_factory is None:
        from xdart.gui.tabs.static_scan import staticWidget
        _widget_factory = staticWidget
    widget = _widget_factory(parent)
    if services.execution_profile is ExecutionProfile.LIVE:
        try:
            widget.enable_async_hydration()
        except Exception:
            logger.exception("Could not enable legacy async hydration")

    activity = _Activity(
        lambda: bool(widget.displayframe._processing_active)
    )

    def describe_layout() -> str:
        return (
            f"left browser={widget.ui.leftFrame.geometry()} "
            f"middle display={widget.ui.middleFrame.geometry()} "
            f"right controls={widget.ui.rightFrame.geometry()}"
        )

    return PageHandle(
        key=STATIC_SCAN_PAGE_KEY,
        widget=widget,
        close=_LegacyCloser(widget, widget.close, activity),
        settings_io=_SettingsIO(
            widget.shortcut_load_settings, widget.shortcut_save_settings),
        run_control=_RunControl(
            widget.shortcut_run_pause, widget.shortcut_stop),
        write_mode=_WriteMode(widget.shortcut_toggle_write_mode),
        slice_pin=_SlicePin(widget.shortcut_pin_slice_cut),
        activity=activity,
        diagnostics=_Diagnostics(describe_layout),
    )
