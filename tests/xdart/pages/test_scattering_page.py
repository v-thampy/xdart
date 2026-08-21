"""Opt-in coexistence mount of the vNext Scattering Workspace.

Production-wired (HARD RULE 2): the real catalog, the real ``Main`` host, the
real ``build_scattering_workspace`` factory and the real page — no fake stands
on the mount seam. The legacy page stays registered and remains the default;
selecting the vNext page is an explicit opt-in (persisted key or constructor
argument).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets

from xdart.gui.pages.catalog import (
    BUILTIN_PAGES,
    DEFAULT_PAGE_KEY,
    LEGACY_STATIC_PAGE,
    SCATTERING_WORKSPACE_PAGE,
)
from xdart.gui.pages.registry import PageRegistry
from xdart.gui.pages.values import PageCapability, PageCleanup, PageLifecycle


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def isolated_settings(tmp_path, monkeypatch):
    settings = tmp_path / "settings.ini"
    monkeypatch.setenv("XDART_SETTINGS_FILE", str(settings))
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    return settings


# ---------------------------------------------------------------------------
# Catalog shape — coexistence, defaults, declared capabilities
# ---------------------------------------------------------------------------

def test_catalog_registers_both_pages_and_keeps_legacy_default():
    assert tuple(page.key for page in BUILTIN_PAGES) == (
        "static-scan", "scattering-workspace")
    assert DEFAULT_PAGE_KEY == "static-scan"
    assert BUILTIN_PAGES[0] is LEGACY_STATIC_PAGE


def test_scattering_descriptor_declares_the_frozen_adoption_ports():
    page = SCATTERING_WORKSPACE_PAGE
    assert page.lifecycle is PageLifecycle.EXIT_ONLY
    assert page.order == 1
    assert page.capabilities == frozenset({
        PageCapability.OPEN_FOLDER,
        PageCapability.RUN_CONTROL,
        PageCapability.WRITE_MODE_TOGGLE,
        PageCapability.RUN_ACTIVITY,
        PageCapability.APP_MENU_HOSTS,
    })
    # Slice pin and settings I/O stay absent: the page has no such surface,
    # so the host disables those actions rather than claiming them.
    assert PageCapability.SLICE_PIN not in page.capabilities
    assert PageCapability.SETTINGS_PERSISTENCE not in page.capabilities


def test_default_selection_without_optin_is_the_legacy_page():
    registry = PageRegistry(BUILTIN_PAGES).freeze()
    assert registry.select(None, DEFAULT_PAGE_KEY) is LEGACY_STATIC_PAGE
    assert registry.select("unknown-page", DEFAULT_PAGE_KEY) is LEGACY_STATIC_PAGE
    assert registry.select(
        "scattering-workspace", DEFAULT_PAGE_KEY) is SCATTERING_WORKSPACE_PAGE


# ---------------------------------------------------------------------------
# Lazy import — registering the descriptor must not import the vNext package
# ---------------------------------------------------------------------------

def test_catalog_import_does_not_import_the_scattering_package():
    code = (
        "import sys\n"
        "import xdart.gui.pages.catalog\n"
        "loaded = [m for m in sys.modules if 'gui.tabs.scattering' in m]\n"
        "assert not loaded, loaded\n"
        "print('lazy-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert "lazy-ok" in result.stdout


# ---------------------------------------------------------------------------
# Real host path — opt-in mount, menus, activity, close-to-clean
# ---------------------------------------------------------------------------

def _mounted_host(selected_key=None):
    from xdart import _gui_main

    return _gui_main.Main(selected_page_key=selected_key)


def test_host_hides_only_the_redundant_vnext_application_statusbar(
        qapp, isolated_settings):
    workspace = _mounted_host(SCATTERING_WORKSPACE_PAGE.key)
    try:
        assert workspace.statusBar().isHidden()

        # The visible, authoritative status remains the scientific footer
        # immediately below the 1-D plot.  The hidden host presenter may still
        # receive the same notice for its existing logging/contract bridge.
        workspace.main_widget._notice("bridge probe notice")
        workspace.main_widget._refresh_shell()
        assert (
            workspace.main_widget._shell.scientific.status.text()
            == "bridge probe notice"
        )
        assert workspace.statusBar().currentMessage() == "bridge probe notice"
        assert workspace.page_handle.close().status is PageCleanup.CLEAN
    finally:
        workspace.close()
        workspace.deleteLater()
        qapp.processEvents()

    legacy = _mounted_host(LEGACY_STATIC_PAGE.key)
    try:
        assert not legacy.statusBar().isHidden()
    finally:
        legacy.close()
        legacy.deleteLater()
        qapp.processEvents()


def test_host_mounts_the_real_workspace_on_explicit_optin(
        qapp, isolated_settings):
    window = _mounted_host("scattering-workspace")
    try:
        assert window.selected_page_key == "scattering-workspace"
        assert window.page_descriptor is SCATTERING_WORKSPACE_PAGE
        assert window.main_widget.objectName() == "scatteringWorkspace"
        # The host attached the application appearance actions into the
        # page-owned Config menu (APP_MENU_HOSTS truthfully declared).
        points = window.page_handle.app_menus.mount_points()
        titles = {
            action.menu().title()
            for action in points.config_menu.actions()
            if action.menu() is not None
        }
        assert {"Theme", "Font Size", "Accent Color", "Spacing"} <= titles
        # Idle page: activity is truthfully quiet.
        assert window.page_handle.activity.active() is False
        handle = window.page_handle
        # Frozen adoption ports are present and functional; Append is held
        # through a PRESENT write-mode port whose refusal carries the
        # scattering owner's human-facing constant — asserted through the
        # exact host shortcut so the status bar shows the same sentence.
        from xdart.gui.pages.values import ActionAccepted, ActionRefused
        from xdart.gui.tabs.scattering.output_values import (
            APPEND_UNAVAILABLE as SCATTERING_APPEND_UNAVAILABLE,
        )
        assert handle.open_folder is not None
        refusal = window._shortcut_toggle_write_mode()
        assert refusal == ActionRefused(SCATTERING_APPEND_UNAVAILABLE)
        assert refusal.reason == (
            "Append is deferred to H23 shared output transaction support.")
        assert window.statusBar().currentMessage() == refusal.reason
        # Run/Stop dispatch through the page's one command owner; an idle
        # sourceless page refuses via notice and stays quiet — no thread.
        assert type(handle.run_control.run_pause()) is ActionAccepted
        assert type(handle.run_control.stop()) is ActionAccepted
        assert handle.activity.active() is False
        # Unsupported ports stay absent (actions disabled by the host).
        assert handle.slice_pin is None
        assert handle.settings_io is None
        # The page notice channel is bridged to the host status presenter.
        window.main_widget.noticeChanged.emit("bridge probe notice")
        assert window.statusBar().currentMessage() == "bridge probe notice"
        # Close converges to a clean receipt for an idle workspace.
        receipt = handle.close()
        assert receipt.status is PageCleanup.CLEAN, receipt.detail
        assert handle.close().status is PageCleanup.CLEAN  # re-entrant
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_host_default_startup_still_mounts_the_legacy_page(
        qapp, isolated_settings):
    window = _mounted_host(None)
    try:
        assert window.selected_page_key == "static-scan"
        assert window.page_descriptor is LEGACY_STATIC_PAGE
        assert window.main_widget.objectName() != "scatteringWorkspace"
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_persisted_settings_key_opts_into_the_workspace(
        qapp, isolated_settings):
    from xdart.gui.themes.typography import application_settings

    settings = application_settings()
    settings.setValue("page.selected", "scattering-workspace")
    settings.sync()
    window = _mounted_host(None)
    try:
        assert window.selected_page_key == "scattering-workspace"
        assert window.main_widget.objectName() == "scatteringWorkspace"
        assert window.page_handle.close().status is PageCleanup.CLEAN
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# Port mappings — finite rows over the real value contracts
# ---------------------------------------------------------------------------

def test_activity_maps_every_run_phase_truthfully():
    from types import SimpleNamespace

    from xdart.gui.pages.scattering_workspace import _WorkspaceActivity
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    quiet = {RunPhase.IDLE, RunPhase.FAILED, RunPhase.CLOSED}
    operation = SimpleNamespace(owned=False)
    for phase in RunPhase:
        port = _WorkspaceActivity(SimpleNamespace(phase=phase), operation)
        assert port.active() is (phase not in quiet), phase


def test_activity_includes_operation_owner_and_fails_closed():
    from types import SimpleNamespace

    from xdart.gui.pages.scattering_workspace import _WorkspaceActivity
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    lifecycle = SimpleNamespace(phase=RunPhase.IDLE)
    assert _WorkspaceActivity(
        lifecycle, SimpleNamespace(owned=False)
    ).active() is False
    assert _WorkspaceActivity(
        lifecycle, SimpleNamespace(owned=True)
    ).active() is True
    assert _WorkspaceActivity(
        SimpleNamespace(), SimpleNamespace(owned=False)
    ).active() is True
    assert _WorkspaceActivity(
        lifecycle, SimpleNamespace(owned="unknown")
    ).active() is True


def test_close_receipt_maps_cleanup_status_authoritatively():
    from types import SimpleNamespace

    from xdart.gui.pages.scattering_workspace import _WorkspaceCloser
    from xdart.gui.tabs.scattering.events import (
        CleanupStatus,
        DetachedDiagnostic,
        LifecycleResult,
        LifecycleStatus,
        RunPhase,
    )
    from xdart.gui.tabs.scattering.start_outcomes import StartClosed

    lifecycle = LifecycleResult(LifecycleStatus.APPLIED, RunPhase.CLOSED)
    retained = (DetachedDiagnostic("m", "TransientError", "boom", "cleanup"),)

    def closer_for(receipts):
        queue = list(receipts)
        return _WorkspaceCloser(SimpleNamespace(
            close_workspace=lambda: queue.pop(0) if len(queue) > 1
            else queue[0]))

    # CLEANED with no diagnostics -> CLEAN.
    clean = closer_for([StartClosed(lifecycle)])()
    assert clean.status is PageCleanup.CLEAN
    # CLEANED with RETAINED diagnostics is still authoritatively CLEAN; the
    # diagnostics ride in the detail (the anti-livelock contract).
    survived = closer_for([StartClosed(
        lifecycle, cleanup_status=CleanupStatus.CLEANED,
        cleanup_failures=retained)])()
    assert survived.status is PageCleanup.CLEAN
    assert "retained cleanup diagnostics" in survived.detail
    # CLEANUP_PENDING first, CLEANED on the host's re-poll -> PENDING then
    # CLEAN, with the transient diagnostics retained across both receipts.
    closer = closer_for([
        StartClosed(lifecycle, cleanup_status=CleanupStatus.CLEANUP_PENDING,
                    cleanup_failures=retained),
        StartClosed(lifecycle, cleanup_status=CleanupStatus.CLEANED,
                    cleanup_failures=retained),
    ])
    assert closer().status is PageCleanup.PENDING
    assert closer().status is PageCleanup.CLEAN


def test_providers_supply_the_exact_service_objects(qapp, isolated_settings):
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.pages.services import (
        DiagnosticIdentity,
        ExecutionProfile,
        HostServices,
    )
    from xdart.gui.tabs.scattering.adapters.run_executor import (
        StandardRunExecutor,
    )
    from xdart.gui.tabs.scattering.adapters.source import (
        FilesystemSourceAdapter,
    )

    store = RunIntentStore(RunIntent(output_mode="Overwrite"))
    executor = StandardRunExecutor()
    sources = FilesystemSourceAdapter()

    class _Provider:
        def __init__(self, value):
            self.value = value

        def store_for(self, _key):
            return self.value

        def executor_for(self, _key):
            return self.value

        def source_port_for(self, _key):
            return self.value

    services = HostServices(
        status=SimpleNamespaceStatus(),
        run_intents=_Provider(store),
        execution=_Provider(executor),
        sources=_Provider(sources),
        execution_profile=ExecutionProfile.TEST,
        diagnostics=DiagnosticIdentity("tests.scattering.page"),
    ).for_page(SCATTERING_WORKSPACE_PAGE.key)

    handle = SCATTERING_WORKSPACE_PAGE.build(services, None)
    try:
        widget = handle.widget
        assert widget._intents is store
        assert widget._run_executor is executor
        assert widget._sources is sources
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        QtWidgets.QApplication.processEvents()


class SimpleNamespaceStatus:
    def __init__(self):
        self.messages = []

    def show(self, text, timeout_ms=0):
        self.messages.append(text)


class _ExactProvider:
    def __init__(self, value):
        self.value = value

    def store_for(self, _key):
        return self.value

    def executor_for(self, _key):
        return self.value

    def source_port_for(self, _key):
        return self.value


def _build_mounted_workspace(store, status):
    from xdart.gui.pages.scattering_workspace import (
        SCATTERING_PAGE_KEY,
        build_scattering_workspace,
    )
    from xdart.gui.pages.services import (
        DiagnosticIdentity,
        ExecutionProfile,
        HostServices,
    )

    services = HostServices(
        status=status,
        run_intents=_ExactProvider(store),
        execution=_ExactProvider(None),
        sources=_ExactProvider(None),
        execution_profile=ExecutionProfile.TEST,
        diagnostics=DiagnosticIdentity("tests.scattering.mounted-chooser"),
    ).for_page(SCATTERING_PAGE_KEY)
    return build_scattering_workspace(services, None)


def test_mounted_control_path_chooser_uses_page_start_and_commits(
        qapp, isolated_settings, tmp_path, monkeypatch):
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import PONI_FILE
    from xdart.utils.browse import remember_browse_path

    current_directory = tmp_path / "current-control"
    dialog_start = tmp_path / "last-browse"
    selected_directory = tmp_path / "selected-control"
    for directory in (current_directory, dialog_start, selected_directory):
        directory.mkdir()
    current = current_directory / "current.poni"
    selected = selected_directory / "selected.poni"
    current.touch()
    selected.touch()
    remember_browse_path(dialog_start)

    starts = []

    def choose_file(_parent, _title, start_directory, _file_filter):
        starts.append(start_directory)
        return str(selected), "PONI files (*.poni)"

    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", choose_file)
    store = RunIntentStore(RunIntent(
        project_root=str(current_directory),
        poni_file=str(current),
        output_mode="Overwrite",
    ))
    status = SimpleNamespaceStatus()
    handle = _build_mounted_workspace(store, status)
    try:
        handle.widget._choose_control_path(PONI_FILE)

        assert starts == [str(dialog_start)]
        assert store.snapshot().thaw().poni_file == str(selected)
        assert not any("Browse failed" in message for message in status.messages)
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        qapp.processEvents()


def test_mounted_source_chooser_uses_page_start_and_preserves_policy(
        qapp, isolated_settings, tmp_path, monkeypatch):
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xrd_tools.sources.selection import DirectorySourceSpec
    from xdart.utils.browse import remember_browse_path

    prior_root = tmp_path / "prior-source"
    dialog_start = tmp_path / "last-browse"
    selected_root = tmp_path / "selected-source"
    for directory in (prior_root, dialog_start, selected_root):
        directory.mkdir()
    remember_browse_path(dialog_start)
    prior = DirectorySourceSpec(
        prior_root,
        recursive=True,
        suffixes=(".h5", ".nxs"),
        name_filter="scan*",
        generation=7,
        metadata_format=None,
    )
    starts = []

    def choose_directory(_parent, _title, start_directory):
        starts.append(start_directory)
        return str(selected_root)

    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", choose_directory)
    store = RunIntentStore(RunIntent(
        source_spec=prior,
        project_root=str(tmp_path),
        output_mode="Overwrite",
    ))
    status = SimpleNamespaceStatus()
    handle = _build_mounted_workspace(store, status)
    try:
        handle.widget._choose_source_selection("Image Directory")

        selected = store.snapshot().thaw().source_spec
        assert starts == [str(dialog_start)]
        assert type(selected) is DirectorySourceSpec
        assert selected.root == selected_root
        assert selected.recursive is prior.recursive
        assert selected.suffixes == prior.suffixes
        assert selected.name_filter == prior.name_filter
        assert selected.metadata_format == prior.metadata_format
        assert selected.generation == prior.generation + 1
        assert not any(
            "Choose source failed" in message for message in status.messages
        )
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        qapp.processEvents()


def test_mounted_chooser_cancel_does_not_mutate_or_emit_error(
        qapp, isolated_settings, tmp_path, monkeypatch):
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xrd_tools.sources.selection import DirectorySourceSpec
    from xdart.gui.tabs.scattering.controls_projection import PONI_FILE

    source_root = tmp_path / "source"
    project_root = tmp_path / "project"
    source_root.mkdir()
    project_root.mkdir()
    poni = project_root / "current.poni"
    poni.touch()
    prior = DirectorySourceSpec(
        source_root,
        recursive=True,
        suffixes=(".nxs",),
        name_filter="sample*",
        generation=4,
        metadata_format="auto",
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args: ("", ""),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *_args: "",
    )
    store = RunIntentStore(RunIntent(
        source_spec=prior,
        project_root=str(project_root),
        poni_file=str(poni),
        output_mode="Overwrite",
    ))
    before = store.snapshot()
    revision = store.revision
    status = SimpleNamespaceStatus()
    handle = _build_mounted_workspace(store, status)
    try:
        handle.widget._choose_control_path(PONI_FILE)
        handle.widget._choose_source_selection("Image Directory")

        assert store.snapshot() == before
        assert store.revision == revision
        assert not any(
            "Browse failed" in message or "Choose source failed" in message
            for message in status.messages
        )
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        qapp.processEvents()


def test_default_fallback_seeds_an_admittable_overwrite_intent(
        qapp, isolated_settings):
    from xdart.gui.pages.services import empty_host_services

    services = empty_host_services(SimpleNamespaceStatus()).for_page(
        SCATTERING_WORKSPACE_PAGE.key)
    handle = SCATTERING_WORKSPACE_PAGE.build(services, None)
    try:
        intent = handle.widget._intents.snapshot().thaw()
        assert intent.output_mode == "Overwrite"
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_fresh_process_default_startup_selects_legacy_without_scattering(
        tmp_path):
    code = (
        "import os, sys\n"
        "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        "from pyqtgraph.Qt import QtWidgets\n"
        "app = QtWidgets.QApplication([])\n"
        "from xdart import _gui_main\n"
        "window = _gui_main.Main()\n"
        "loaded = [m for m in sys.modules if 'gui.tabs.scattering' in m]\n"
        "print('SELECTED', window.selected_page_key, len(loaded), flush=True)\n"
        "os._exit(0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "XDART_SETTINGS_FILE": str(tmp_path / "settings.ini"),
            "XDART_SESSION_FILE": str(tmp_path / "session.json"),
        },
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "SELECTED static-scan 0" in result.stdout
