"""Default mount of the current Scattering Workspace.

Production-wired (HARD RULE 2): the real catalog, the real ``Main`` host, the
real ``build_scattering_workspace`` factory and the real page — no fake stands
on the mount seam. The Scattering Workspace is the sole built-in product page.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from threading import current_thread

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtCore, QtWidgets

from xdart.gui.pages.catalog import (
    BUILTIN_DESCRIPTORS,
    BUILTIN_PAGES,
    DEFAULT_PAGE_KEY,
    RSM_TOOL,
    SCATTERING_WORKSPACE_PAGE,
    STITCH_TOOL,
)
from xdart.gui.pages.registry import PageRegistry
from xdart.gui.pages.values import (
    PageCapability,
    PageCleanup,
    PageLifecycle,
    RSM_TOOL_KEY,
    STITCH_TOOL_KEY,
)


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
# Catalog shape — sole current page, default, declared capabilities
# ---------------------------------------------------------------------------

def test_catalog_registers_current_workspace_first_and_as_default():
    assert tuple(page.key for page in BUILTIN_PAGES) == (
        "scattering-workspace",
    )
    assert DEFAULT_PAGE_KEY == "scattering-workspace"
    assert BUILTIN_PAGES[0] is SCATTERING_WORKSPACE_PAGE
    assert BUILTIN_DESCRIPTORS == (*BUILTIN_PAGES, STITCH_TOOL, RSM_TOOL)
    assert STITCH_TOOL.key == STITCH_TOOL_KEY == "stitch"
    assert (
        STITCH_TOOL.label,
        STITCH_TOOL.category,
        STITCH_TOOL.order,
        STITCH_TOOL.tool_kind,
    ) == ("Stitching", "analysis", 100, "processing")
    assert RSM_TOOL.key == RSM_TOOL_KEY == "rsm"
    assert (
        RSM_TOOL.label,
        RSM_TOOL.category,
        RSM_TOOL.order,
        RSM_TOOL.tool_kind,
    ) == ("Reciprocal Space Map", "analysis", 110, "processing")


def test_scattering_descriptor_declares_the_frozen_adoption_ports():
    page = SCATTERING_WORKSPACE_PAGE
    assert page.lifecycle is PageLifecycle.EXIT_ONLY
    assert page.order == 0
    assert page.capabilities == frozenset({
        PageCapability.OPEN_FOLDER,
        PageCapability.SETTINGS_PERSISTENCE,
        PageCapability.RUN_CONTROL,
        PageCapability.WRITE_MODE_TOGGLE,
        PageCapability.SLICE_PIN,
        PageCapability.RUN_ACTIVITY,
        PageCapability.APP_MENU_HOSTS,
    })


def test_default_selection_is_the_current_workspace():
    registry = PageRegistry(BUILTIN_DESCRIPTORS).freeze()
    assert registry.select(None, DEFAULT_PAGE_KEY) is SCATTERING_WORKSPACE_PAGE
    assert registry.select(
        "unknown-page", DEFAULT_PAGE_KEY) is SCATTERING_WORKSPACE_PAGE
    assert registry.select(
        "scattering-workspace", DEFAULT_PAGE_KEY) is SCATTERING_WORKSPACE_PAGE
    assert registry.select(
        STITCH_TOOL_KEY, DEFAULT_PAGE_KEY) is SCATTERING_WORKSPACE_PAGE
    assert registry.select(
        RSM_TOOL_KEY, DEFAULT_PAGE_KEY) is SCATTERING_WORKSPACE_PAGE


# ---------------------------------------------------------------------------
# Lazy import — registering the descriptor must not import the vNext package
# ---------------------------------------------------------------------------

def test_catalog_import_does_not_import_the_scattering_package():
    code = (
        "import sys\n"
        "import xdart.gui.pages.catalog\n"
        "loaded = [m for m in sys.modules if "
        "'gui.tabs.scattering' in m "
        "or m.startswith('xdart.gui.tools.stitch') "
        "or m.startswith('xdart.gui.tools.rsm') "
        "or m == 'xrd_tools.analysis.stitch_operation' "
        "or m == 'xrd_tools.analysis.rsm_operation']\n"
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


def test_builtin_stitch_action_constructs_one_idle_tool_only_when_opened(
        qapp, isolated_settings):
    from xdart.gui.pages.values import ActionCompleted

    window = _mounted_host(None)
    try:
        assert STITCH_TOOL.key not in window._tool_handles
        assert window.ui.menuAnalysis is None
        page = window.page_handle.widget
        assert page.findChild(QtWidgets.QToolButton, "analysisMenuButton") is None
        combo = page._shell.run_controls.modeCombo
        assert combo.findText("Stitch") < 0
        assert combo.findText("Stitch 1D") >= 0
        assert combo.findText("Stitch 2D") >= 0
        assert combo.findText("RSM") >= 0
        requested = []
        page.toolRequested.connect(requested.append)
        combo.setCurrentText("Stitch 1D")
        assert page._shell.run_controls.startButton.isEnabled()
        page._shell.run_controls.startButton.click()
        qapp.processEvents()
        handle = window._tool_handles[STITCH_TOOL.key]
        assert handle.widget.objectName() == "stitchToolDialog"
        assert handle.widget.parity_hold.text() == (
            "Mode: 1-D only · 2-D held pending the scan-14 orientation parity oracle"
        )
        combo.setCurrentText("Stitch 2D")
        assert not page._shell.run_controls.startButton.isEnabled()
        page._shell.run_controls.startButton.click()
        qapp.processEvents()
        assert requested == ["stitch"]
        assert handle.widget.source_widget._external_execution is True
        assert handle.widget.source_widget._probe_executor is None
        assert handle.activity.active() is False
        assert window.open_tool(STITCH_TOOL.key) == ActionCompleted("stitch")
        assert window._tool_handles[STITCH_TOOL.key] is handle
        assert RSM_TOOL.key not in window._tool_handles
        assert handle.close().status is PageCleanup.CLEAN
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_builtin_rsm_action_constructs_one_idle_tool_only_when_opened(
        qapp, isolated_settings):
    from xdart.gui.pages.values import ActionCompleted

    window = _mounted_host(None)
    try:
        page_widget = window.main_widget
        selected_page_key = window.selected_page_key
        assert RSM_TOOL.key not in window._tool_handles
        assert window.ui.menuAnalysis is None
        page = window.page_handle.widget
        page._shell.run_controls.modeCombo.setCurrentText("RSM")
        assert page._shell.run_controls.startButton.isEnabled()
        page._shell.run_controls.startButton.click()
        qapp.processEvents()
        handle = window._tool_handles[RSM_TOOL.key]
        dialog = handle.widget
        assert handle.key == RSM_TOOL_KEY
        assert dialog.objectName() == "rsmToolDialog"
        assert dialog.isWindow()
        assert dialog.parent() is window
        assert handle.activity.active() is False
        dialog.close()
        qapp.processEvents()
        assert not dialog.isVisible()
        assert window.open_tool(RSM_TOOL.key) == ActionCompleted("rsm")
        assert window._tool_handles[RSM_TOOL.key] is handle
        assert window._tool_handles[RSM_TOOL.key].widget is dialog
        assert window.main_widget is page_widget
        assert window.selected_page_key == selected_page_key
        assert STITCH_TOOL.key not in window._tool_handles
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


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
        # Frozen adoption ports are present and functional.  H23 now owns
        # Append, so the host shortcut must reach the page's write-mode owner.
        from xdart.gui.pages.values import ActionAccepted, ActionCompleted
        assert handle.open_folder is not None
        changed = window._shortcut_toggle_write_mode()
        assert changed == ActionCompleted("write-mode-append")
        assert handle.widget._intents.snapshot().thaw().output_mode == "Append"
        # Run/Stop dispatch through the page's one command owner; an idle
        # sourceless page refuses via notice and stays quiet — no thread.
        assert type(handle.run_control.run_pause()) is ActionAccepted
        assert type(handle.run_control.stop()) is ActionAccepted
        assert handle.activity.active() is False
        assert handle.slice_pin is not None
        assert handle.settings_io is not None
        assert window.actionPinSliceCut.isEnabled()
        assert window.actionLoadSettings.isEnabled()
        assert window.actionSaveSettings.isEnabled()
        handle.widget._profile_path_chooser = lambda *_args: None
        assert window._shortcut_load_settings().detail == (
            "profile-load-requested"
        )
        assert window._shortcut_save_settings().detail == (
            "profile-save-requested"
        )
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


def test_host_default_startup_mounts_the_current_workspace(
        qapp, isolated_settings):
    window = _mounted_host(None)
    try:
        assert window.selected_page_key == "scattering-workspace"
        assert window.page_descriptor is SCATTERING_WORKSPACE_PAGE
        assert window.main_widget.objectName() == "scatteringWorkspace"
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
        lifecycle, SimpleNamespace(owned=False), lambda: True,
    ).active() is True
    assert _WorkspaceActivity(
        lifecycle, SimpleNamespace(owned=False), lambda: False,
    ).active() is False
    assert _WorkspaceActivity(
        lifecycle, SimpleNamespace(owned=False), lambda: "unknown",
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
        assert widget._source_selection._sources is sources
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


def _build_mounted_workspace(store, status, parent=None):
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
    return build_scattering_workspace(services, parent)


def _present_poni_confirmation(page, store, root, candidates=()):
    from xdart.gui.tabs.scattering.contracts import SourceFileState
    from xdart.gui.tabs.scattering.experiment_authoring import (
        CalibrationRequest,
        CalibrationResult,
    )
    from xdart.gui.tabs.scattering.authored_assets import AuthoredAssetPhase
    from xdart.gui.tabs.scattering.operation_values import (
        OperationIdentity,
        OperationTerminal,
        OperationTerminalStatus,
        OperationUpdate,
    )

    source = root / "authoring-source.tiff"
    executable = root / "pyFAI-calib2-test"
    if not source.exists():
        source.write_bytes(b"source-fixture")
    if not executable.exists():
        executable.write_bytes(b"executable-fixture")
        executable.chmod(0o700)
    directory_state = root.stat()
    request = CalibrationRequest(
        str(source),
        str(executable),
        SourceFileState.capture(executable),
        None,
        SourceFileState.capture(source),
        str(root),
        (directory_state.st_dev, directory_state.st_ino),
    )
    stamp = page._operation_context_stamp(store.revision)
    identity = OperationIdentity(900 + store.revision)
    page._apply_authored_asset_transition(
        page._authored_assets.adopt_operation(
            "poni", request, stamp, identity,
        )
    )
    result = CalibrationResult(
        request,
        tuple(candidates),
        0,
        (request.executable, request.source_path),
        request.monitored_directory,
    )
    assert page._consume_authored_asset_update(OperationUpdate(
        identity,
        terminal=OperationTerminal(
            identity, OperationTerminalStatus.RETURNED, payload=result,
        ),
    ))
    assert page._authored_assets.phase is AuthoredAssetPhase.TERMINAL_READY
    page._advance_authored_asset_confirmation()
    page._advance_authored_asset_confirmation()
    assert page._authored_assets.phase is AuthoredAssetPhase.CONFIRM_PRESENTED


def test_mounted_profile_ports_roundtrip_one_next_run_intent(
        qapp, isolated_settings, tmp_path):
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent

    profile = tmp_path / "beamtime-profile.json"
    initial = RunIntent(
        project_root=str(tmp_path / "project"),
        save_path=str(tmp_path / "project" / "xdart_processed_data"),
        output_mode="Overwrite",
        processing_mode="Int 1D (XYE)",
        batch_mode=True,
        max_cores=4,
        run_options={
            "_post_g2_pipeline_v2": {"pipeline": "stale"},
            "_post_g2_output_diagnostics_v1": {"save_xye": False},
            "unrelated": "preserved",
        },
    )
    store = RunIntentStore(initial)
    status = SimpleNamespaceStatus()
    handle = _build_mounted_workspace(store, status)
    handle.widget._profile_path_chooser = (
        lambda _action, _start: str(profile)
    )
    try:
        assert handle.settings_io is not None
        assert handle.settings_io.save().detail == "profile-save-requested"
        assert profile.is_file()
        assert '"schema": "xdart.run-intent-profile"' in (
            profile.read_text(encoding="utf-8")
        )
        assert store.revision == 0

        prior = store.snapshot()
        changed = prior.thaw()
        changed.processing_mode = "Int 2D"
        changed.batch_mode = False
        changed.max_cores = 1
        store.commit(changed, expected_revision=prior.revision)
        assert handle.settings_io.load().detail == "profile-load-requested"

        loaded = store.snapshot().thaw()
        assert store.revision == 2
        assert loaded.processing_mode == "Int 1D (XYE)"
        assert loaded.batch_mode is True
        assert loaded.max_cores == 4
        assert loaded.run_options == {"unrelated": "preserved"}
        assert any("Profile saved:" in message for message in status.messages)
        assert any("Profile loaded:" in message for message in status.messages)
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("quit_action", [False, True])
def test_application_close_restores_last_config_and_focused_edit(
        qapp, isolated_settings, tmp_path, monkeypatch, quit_action):
    from pyqtgraph.Qt import QtTest
    from xdart._gui_main import Main, _apply_cli_session_args
    from xdart.gui.tabs.scattering.controls_projection import PROJECT_ROOT
    from xdart.gui.widgets.controls_panel import FormRow
    from xdart.utils.session import load_session, save_session
    from xrd_tools.session.run_intent_profile import dump_run_intent_profile

    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    if quit_action:
        _apply_cli_session_args(["xdart", "-n", str(tmp_path / "named.json")])
    save_session({"unrelated_session_key": "preserve-other-session-fields"})
    window = Main()
    try:
        page = window.page_handle.widget
        store = page._intents
        changed = store.snapshot().thaw()
        changed.processing_mode = "Int 2D"
        changed.max_cores = 2
        changed.bai_1d_args = {"npt": 711, "unit": "q_A^-1"}
        changed.bai_2d_args = {"npt_rad": 401, "npt_azim": 91}
        changed.threshold.threshold_min = 12
        changed.threshold.threshold_max = 10000
        store.commit(changed, expected_revision=store.revision)
        page._refresh_shell()
        project = tmp_path / "my project"
        project.mkdir()
        row = next(row for row in page._shell.controls.findChildren(FormRow)
                   if row.path == PROJECT_ROOT)
        window.activateWindow()
        qapp.processEvents()
        row.editor.setFocus()
        qapp.processEvents()
        assert row.editor.hasFocus()
        row.editor.selectAll()
        QtTest.QTest.keyClicks(row.editor, str(project))
        assert store.snapshot().thaw().project_root != str(project)
        exits = []
        # Replace only process termination; the real Quit/close/save path runs.
        monkeypatch.setattr(window, "_terminate_process", lambda: exits.append(True))
        if quit_action:
            window.ui.actionExit.trigger()
        else:
            window.close()
        qapp.processEvents()
        assert not window.isVisible()
        if quit_action:
            assert exits == [True]
        assert store.snapshot().thaw().project_root == str(project)
        expected = dump_run_intent_profile(store.snapshot().thaw())
        assert load_session()["scattering_run_profile"] == expected
        assert load_session()["unrelated_session_key"] == "preserve-other-session-fields"
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()

    restored = Main()
    try:
        actual = restored.page_handle.widget._intents.snapshot().thaw()
        assert dump_run_intent_profile(actual) == expected
        assert actual.project_root == str(project)
        assert actual.max_cores == 2
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()


def test_fresh_application_does_not_restore_or_overwrite_config(
        qapp, isolated_settings, tmp_path, monkeypatch):
    from xdart._gui_main import Main
    from xdart.utils.session import save_session
    from xrd_tools.session.run_configuration import RunIntent
    from xrd_tools.session.run_intent_profile import dump_run_intent_profile

    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    save_session({"scattering_run_profile": dump_run_intent_profile(
        RunIntent(project_root=str(tmp_path / "saved-project"), max_cores=2))})
    session = Path(os.environ["XDART_SESSION_FILE"])
    before = session.read_bytes()
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    window = Main()
    try:
        assert window.page_handle.widget._intents.snapshot().thaw().project_root == ""
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
    assert session.read_bytes() == before


@pytest.mark.parametrize("total_gib, expected", ((8, 2), (15, 2), (16, 4), (64, 4)))
def test_default_cores_follow_the_small_ram_worker_cap(
        qapp, isolated_settings, monkeypatch, total_gib, expected):
    """The Cores a user never chose must not bypass the small-RAM default.

    The run treats Cores as a deliberate request, which the cap never trims;
    so the value this factory invents has to be shaped by the cap itself.
    """
    from xrd_tools.core import staging

    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    monkeypatch.delenv(staging.REDUCTION_WORKERS_ENV, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        staging, "total_physical_ram_bytes", lambda: total_gib * 1024 ** 3)
    handle = _build_mounted_workspace(None, SimpleNamespaceStatus())
    try:
        intent = handle.widget._intents.snapshot().thaw()
        assert intent.max_cores == expected
        # What the run will resolve from that Cores value, on the same host.
        assert staging.reduction_worker_cap(intent.max_cores) == expected
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        qapp.processEvents()


def test_default_cores_still_leave_one_core_for_the_interface(
        qapp, isolated_settings, monkeypatch):
    from xrd_tools.core import staging

    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    monkeypatch.delenv(staging.REDUCTION_WORKERS_ENV, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    monkeypatch.setattr(staging, "total_physical_ram_bytes", lambda: 64 * 1024 ** 3)
    handle = _build_mounted_workspace(None, SimpleNamespaceStatus())
    try:
        assert handle.widget._intents.snapshot().thaw().max_cores == 3
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        qapp.processEvents()


def test_injected_intent_store_does_not_load_or_overwrite_app_session(
        qapp, isolated_settings, tmp_path, monkeypatch):
    from xdart.utils.session import save_session
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xrd_tools.session.run_intent_profile import dump_run_intent_profile

    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    save_session({"scattering_run_profile": dump_run_intent_profile(
        RunIntent(project_root=str(tmp_path / "saved-project")))})
    session = Path(os.environ["XDART_SESSION_FILE"])
    before = session.read_bytes()
    store = RunIntentStore(RunIntent(project_root=str(tmp_path / "injected")))
    handle = _build_mounted_workspace(store, SimpleNamespaceStatus())
    try:
        assert handle.widget._intents is store
        assert store.snapshot().thaw().project_root == str(tmp_path / "injected")
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        qapp.processEvents()
    assert session.read_bytes() == before


def test_mounted_profile_load_is_fail_closed_on_malformed_json(
        qapp, isolated_settings, tmp_path):
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent

    profile = tmp_path / "malformed.json"
    profile.write_text("{not-json", encoding="utf-8")
    store = RunIntentStore(RunIntent(output_mode="Overwrite", max_cores=4))
    before = store.snapshot()
    status = SimpleNamespaceStatus()
    handle = _build_mounted_workspace(store, status)
    handle.widget._profile_path_chooser = (
        lambda _action, _start: str(profile)
    )
    try:
        assert handle.settings_io is not None
        handle.settings_io.load()
        assert store.snapshot() == before
        assert store.revision == before.revision
        assert any("Profile load failed:" in message for message in status.messages)
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        qapp.processEvents()


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


def test_mounted_authoring_source_chooser_is_distinct_and_asset_specific(
        qapp, isolated_settings, tmp_path, monkeypatch):
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent

    start = tmp_path / "source-root"
    start.mkdir()
    tiff = start / "source.tiff"
    tiff.write_bytes(b"fixture")
    calls = []

    def choose(_parent, title, directory, file_filter):
        calls.append((title, directory, file_filter))
        return str(tiff), file_filter

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", choose)
    handle = _build_mounted_workspace(
        RunIntentStore(RunIntent(project_root=str(start))),
        SimpleNamespaceStatus(),
    )
    try:
        page = handle.widget
        assert page._authoring_source_chooser is not page._control_path_chooser
        assert page._authoring_source_chooser("poni", str(start)) == str(tiff)
        assert page._authoring_source_chooser("mask", str(start)) == str(tiff)
        assert calls[0][0] == "Choose calibration source image"
        assert "*.h5" in calls[0][2] and "*.nexus" in calls[0][2]
        assert calls[1][0] == (
            "Choose TIFF or HDF5/NeXus source for mask"
        )
        assert calls[1][2] == (
            "Mask sources (*.tif *.tiff *.h5 *.hdf5 *.nxs *.nexus)"
        )
        assert [item[1] for item in calls] == [str(start), str(start)]
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        qapp.processEvents()


def test_mounted_popup_choose_another_keeps_modal_focus_and_validates_off_gui(
        qapp, isolated_settings, tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering.adapters import external_operation
    from xdart.gui.tabs.scattering.experiment_authoring import (
        qualify_calibration_candidate,
    )
    from xdart.gui.tabs.scattering.operation_values import OperationUpdate
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from tests.xdart.scattering.test_p3_calibrate_operation import _PONI

    root = tmp_path / "authoring-root"
    root.mkdir()
    alternate = root / "existing.poni"
    alternate.write_text(_PONI, encoding="utf-8")
    host = QtWidgets.QWidget()
    store = RunIntentStore(RunIntent(project_root=str(root)))
    status = SimpleNamespaceStatus()
    handle = _build_mounted_workspace(store, status, host)
    page = handle.widget
    chooser_calls, worker_threads = [], []
    real_validate = external_operation.validate_authored_asset

    def validate(request):
        worker_threads.append(current_thread().name)
        return real_validate(request)

    def choose(parent, title, start, file_filter):
        dialog = page._authored_asset_dialog
        assert dialog is not None
        assert dialog.isVisible()
        assert (dialog.windowModality()
                is QtCore.Qt.WindowModality.WindowModal)
        assert dialog.parent() is page
        chooser_calls.append((parent, title, start, file_filter))
        return str(alternate), file_filter

    monkeypatch.setattr(external_operation, "validate_authored_asset", validate)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", choose)
    try:
        host.show()
        page.show()
        qapp.processEvents()
        _present_poni_confirmation(page, store, root)
        qapp.processEvents()
        dialog = page._authored_asset_dialog
        assert dialog is not None
        assert dialog.paths.count() == 1
        assert dialog.selected_path is None
        dialog.choose_button.setFocus()
        dialog.choose_button.click()
        assert chooser_calls == [(
            host, "Choose PONI calibration", str(root),
            "PONI files (*.poni);;All files (*)",
        )]
        assert store.snapshot().thaw().poni_file == ""
        identity = page._authored_assets.operation_identity
        worker = page._workspace_operations._slot._worker
        assert identity is not None and worker is not None
        worker.join(3)
        assert not worker.is_alive()
        page._workspace_operations._slot.observe_stamp(
            page._operation_context_stamp()
        )
        update = page._workspace_operations._slot.poll(identity)
        assert type(update) is OperationUpdate
        assert page._consume_authored_asset_update(update)
        assert worker_threads and worker_threads[0].startswith(
            "scattering-operation-"
        )
        assert store.snapshot().thaw().poni_file == str(alternate)
        qapp.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete,
        )
        qapp.processEvents()

        revision = store.revision
        admitted = qualify_calibration_candidate(str(alternate))
        _present_poni_confirmation(page, store, root, (admitted,))
        qapp.processEvents()
        dialog = page._authored_asset_dialog
        assert dialog is not None
        assert dialog.selected_path == str(alternate)
        dialog.cancel_button.click()
        qapp.processEvents()
        assert store.revision == revision
        assert store.snapshot().thaw().poni_file == str(alternate)
        assert alternate.read_text(encoding="utf-8") == _PONI
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        page.deleteLater()
        host.deleteLater()
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


@pytest.mark.parametrize("total_gib, expected_cores", ((8, 2), (64, 3)))
def test_default_fallback_seeds_an_admittable_overwrite_intent(
        qapp, isolated_settings, monkeypatch, total_gib, expected_cores):
    from xdart.gui.pages.services import empty_host_services
    from xrd_tools.core import staging

    monkeypatch.delenv(staging.REDUCTION_WORKERS_ENV, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    monkeypatch.setattr(
        staging, "total_physical_ram_bytes", lambda: total_gib * 1024 ** 3)

    services = empty_host_services(SimpleNamespaceStatus()).for_page(
        SCATTERING_WORKSPACE_PAGE.key)
    handle = SCATTERING_WORKSPACE_PAGE.build(services, None)
    try:
        intent = handle.widget._intents.snapshot().thaw()
        assert intent.output_mode == "Overwrite"
        assert intent.max_cores == expected_cores
    finally:
        assert handle.close().status is PageCleanup.CLEAN
        handle.widget.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_fresh_process_default_startup_selects_current_workspace(
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
    assert "SELECTED scattering-workspace " in result.stdout
