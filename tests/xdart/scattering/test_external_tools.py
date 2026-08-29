from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.browse_values import (
    BrowseLoadRequest,
    LoadedBrowseCapture,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.external_tools import (
    ExternalNexusQualification,
    ExternalToolConfig,
    ExternalToolId,
    ExternalToolLaunchStatus,
    ExternalToolRegistry,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.tools_view import ToolsView
from xdart.modules.display_context import (
    BrowseContext,
    ContextKind,
    DisplaySelection,
    HydrationOwner,
)
from xrd_tools.io.output_transaction import TargetSnapshot
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _executable(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path.resolve())


def _status(registry: ExternalToolRegistry, tool: ExternalToolId, target=None):
    nexus = (
        ExternalNexusQualification.ready(target)
        if target is not None
        else ExternalNexusQualification.refused(
            "Select one stable current processed .nexus file in Browse."
        )
    )
    return registry.project(nexus=nexus).for_tool(tool)


def _loaded_capture(path: Path) -> LoadedBrowseCapture:
    target = str(path.resolve())
    state = path.stat()
    request = BrowseLoadRequest(
        "external-viewer-capture", 1, target,
        source_root=str(path.parent.resolve()),
    )
    context = object.__new__(BrowseContext)
    selection = DisplaySelection(
        ContextKind.BROWSE,
        HydrationOwner("external-viewer-capture", "scan", target, 1),
        2,
    )
    return LoadedBrowseCapture(
        context,
        request,
        selection,
        target,
        "entry",
        TargetSnapshot(
            True,
            int(state.st_size),
            int(state.st_mtime_ns),
            int(state.st_dev),
            int(state.st_ino),
            "f" * 64,
        ),
        (1,),
    )


def test_registry_is_exactly_two_tools_and_imports_no_gui_dependency() -> None:
    assert tuple(ExternalToolId) == (
        ExternalToolId.NEXPY_SELECTED,
        ExternalToolId.DASHPVA_H5VIEWER,
    )
    assert ToolsView._TOOLS == (
        ("∧ Peak Fitting", "peak_fitting"),
        ("≈ Phase Fitting", "phase_fitting"),
        ("▤ Plot Metadata", "plot_metadata"),
        ("▣ ROI Statistics", "roi_statistics"),
    )
    assert ToolsView._EXTERNAL_VIEWERS == (
        ("◇ Open Selected in NeXpy", "nexpy_selected"),
        ("▦ DashPVA HDF5 Viewer…", "dashpva_h5viewer"),
    )
    module_path = Path(__import__(
        "xdart.gui.tabs.scattering.external_tools", fromlist=["__file__"]
    ).__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not {"PyQt5", "PyQt6", "PySide6", "pyqtgraph", "nexpy", "dashpva"} & imported
    script = """
import sys
import xdart.gui.tabs.scattering.external_tools
forbidden = ('PyQt5', 'PyQt6', 'PySide6', 'pyqtgraph', 'h5py', 'nexpy', 'dashpva')
print(sorted(name for name in forbidden if name in sys.modules))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=module_path.parents[5],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_discovery_is_explicit_then_interpreter_sibling_then_path(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "pixi env" / "bin"
    interpreter = _executable(bin_dir / "python")
    sibling_nexpy = _executable(bin_dir / "nexpy")
    path_dashpva = _executable(tmp_path / "external tools" / "DashPVA")
    calls: list[str] = []

    def which(name: str) -> str | None:
        calls.append(name)
        return path_dashpva if name == "DashPVA" else None

    registry = ExternalToolRegistry(
        environment={}, interpreter=interpreter, which=which,
    )
    assert _status(
        registry, ExternalToolId.NEXPY_SELECTED
    ).executable == sibling_nexpy
    assert _status(
        registry, ExternalToolId.DASHPVA_H5VIEWER
    ).executable == path_dashpva
    assert calls == ["DashPVA"]

    configured_nexpy = _executable(tmp_path / "configured" / "nexpy")
    calls.clear()
    configured = ExternalToolRegistry(
        ExternalToolConfig(nexpy_executable=configured_nexpy),
        environment={}, interpreter=interpreter, which=which,
    )
    assert _status(
        configured, ExternalToolId.NEXPY_SELECTED
    ).executable == configured_nexpy
    assert calls == ["DashPVA"]


def test_invalid_explicit_configuration_blocks_fallback(tmp_path: Path) -> None:
    fallback = _executable(tmp_path / "fallback" / "nexpy")
    registry = ExternalToolRegistry(
        ExternalToolConfig(nexpy_executable="nexpy --unsafe"),
        environment={}, which=lambda _name: fallback,
    )
    status = _status(registry, ExternalToolId.NEXPY_SELECTED)
    assert status.executable is None
    assert status.available is status.enabled is False
    assert "configured" in status.reason.lower()

    from_environment = ExternalToolRegistry(
        environment={"XDART_NEXPY_EXECUTABLE": "/missing/nexpy"},
        which=lambda _name: fallback,
    )
    status = _status(from_environment, ExternalToolId.NEXPY_SELECTED)
    assert status.executable is None
    assert "XDART_NEXPY_EXECUTABLE" in status.reason


def test_projection_requires_current_processed_nexus_only_for_nexpy(
    tmp_path: Path,
) -> None:
    nexpy = _executable(tmp_path / "bin" / "nexpy")
    dashpva = _executable(tmp_path / "bin" / "DashPVA")
    registry = ExternalToolRegistry(
        ExternalToolConfig(nexpy, dashpva), environment={},
    )

    missing = registry.project(nexus=ExternalNexusQualification.refused(
        "Select one stable current processed .nexus file in Browse."
    ))
    nexpy_missing = missing.for_tool(ExternalToolId.NEXPY_SELECTED)
    dash = missing.for_tool(ExternalToolId.DASHPVA_H5VIEWER)
    assert nexpy_missing.available and not nexpy_missing.enabled
    assert "current processed .nexus" in nexpy_missing.reason
    assert dash.available and dash.enabled

    old_suffix = tmp_path / "old-output.nxs"
    old_suffix.write_bytes(b"old")
    with pytest.raises(ValueError, match="qualification"):
        ExternalNexusQualification.ready(str(old_suffix.resolve()))

    selected = tmp_path / "selected output.nexus"
    selected.write_bytes(b"current")
    ready = _status(
        registry, ExternalToolId.NEXPY_SELECTED, str(selected.resolve())
    )
    assert ready.available and ready.enabled and ready.reason


def test_launch_uses_exact_argv_detachment_and_isolated_child_environments(
    tmp_path: Path,
) -> None:
    nexpy = _executable(tmp_path / "bin with spaces" / "nexpy")
    dashpva = _executable(tmp_path / "bin with spaces" / "DashPVA")
    selected = tmp_path / "selected ; literal.nexus"
    selected.write_bytes(b"current")
    target = str(selected.resolve())
    parent_environment = {
        "PYQTGRAPH_QT_LIB": "PySide6",
        "QT_API": "PySide6",
        "MPLBACKEND": "QtAgg",
        "PRESERVE_ME": "yes",
    }
    parent_before = dict(parent_environment)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    processes: list[object] = []

    def popen(argv, **options):
        process = SimpleNamespace(pid=4100 + len(calls))
        processes.append(process)
        calls.append((tuple(argv), options))
        return process

    registry = ExternalToolRegistry(
        ExternalToolConfig(nexpy, dashpva),
        environment=parent_environment,
        popen=popen,
        windows=False,
    )
    nexpy_receipt = registry.launch(
        ExternalToolId.NEXPY_SELECTED,
        nexus=ExternalNexusQualification.ready(target),
    )
    dashpva_receipt = registry.launch(
        ExternalToolId.DASHPVA_H5VIEWER,
    )

    assert nexpy_receipt.status is ExternalToolLaunchStatus.ACCEPTED
    assert dashpva_receipt.status is ExternalToolLaunchStatus.ACCEPTED
    assert nexpy_receipt.argv == (nexpy, target)
    assert dashpva_receipt.argv == (dashpva, "h5viewer")
    assert [argv for argv, _options in calls] == [
        (nexpy, target),
        (dashpva, "h5viewer"),
    ]
    for _argv, options in calls:
        assert options["shell"] is False
        assert options["stdin"] is subprocess.DEVNULL
        assert options["stdout"] is subprocess.DEVNULL
        assert options["stderr"] is subprocess.DEVNULL
        assert options["close_fds"] is True
        assert options["start_new_session"] is True
        assert "creationflags" not in options
    nexpy_env = calls[0][1]["env"]
    assert nexpy_env == {"PRESERVE_ME": "yes"}
    dashpva_env = calls[1][1]["env"]
    assert dashpva_env == {
        "PYQTGRAPH_QT_LIB": "PyQt5",
        "QT_API": "pyqt5",
        "MPLBACKEND": "QtAgg",
        "PRESERVE_ME": "yes",
    }
    assert parent_environment == parent_before
    assert all(
        value is not process
        for slot in registry.__slots__
        for value in (getattr(registry, slot),)
        for process in processes
    )


def test_windows_launch_uses_both_detachment_flags(tmp_path: Path) -> None:
    dashpva = _executable(tmp_path / "DashPVA")
    calls = []

    def popen(argv, **options):
        calls.append((tuple(argv), options))
        return SimpleNamespace(pid=91)

    registry = ExternalToolRegistry(
        ExternalToolConfig(dashpva_executable=dashpva),
        environment={}, popen=popen, windows=True,
    )
    receipt = registry.launch(ExternalToolId.DASHPVA_H5VIEWER)
    assert receipt.status is ExternalToolLaunchStatus.ACCEPTED
    flags = calls[0][1]["creationflags"]
    assert flags & 0x00000008
    assert flags & 0x00000200
    assert "start_new_session" not in calls[0][1]


def test_changed_executable_and_spawn_failure_are_typed(tmp_path: Path) -> None:
    nexpy_path = Path(_executable(tmp_path / "nexpy"))
    selected = tmp_path / "selected.nexus"
    selected.write_bytes(b"current")
    calls = []
    registry = ExternalToolRegistry(
        ExternalToolConfig(nexpy_executable=str(nexpy_path)),
        environment={}, popen=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    nexpy_path.chmod(0o644)
    refused = registry.launch(
        ExternalToolId.NEXPY_SELECTED,
        nexus=ExternalNexusQualification.ready(str(selected.resolve())),
    )
    assert refused.status is ExternalToolLaunchStatus.REFUSED
    assert "changed" in refused.diagnostic.lower()
    assert calls == []

    failing_path = _executable(tmp_path / "DashPVA")

    def fail(_argv, **_options):
        raise OSError("synthetic spawn refusal")

    failing = ExternalToolRegistry(
        ExternalToolConfig(dashpva_executable=failing_path),
        environment={}, popen=fail,
    ).launch(ExternalToolId.DASHPVA_H5VIEWER)
    assert failing.status is ExternalToolLaunchStatus.FAILED
    assert "synthetic spawn refusal" in failing.diagnostic


def test_tools_view_reconciles_truthful_external_availability(
    qapp, tmp_path: Path,
) -> None:
    tools = ToolsView()
    try:
        unavailable = ExternalToolRegistry(
            ExternalToolConfig(
                nexpy_executable="/missing/nexpy",
                dashpva_executable="/missing/DashPVA",
            ),
            environment={},
        )
        tools.reconcile_external(unavailable.project(
            nexus=ExternalNexusQualification.refused(
                "Select one stable current processed .nexus file in Browse."
            )
        ))
        nexpy_button = tools.findChild(
            QtWidgets.QPushButton, "e3Tool_nexpy_selected"
        )
        dashpva_button = tools.findChild(
            QtWidgets.QPushButton, "e3Tool_dashpva_h5viewer"
        )
        assert nexpy_button is not None and not nexpy_button.isEnabled()
        assert dashpva_button is not None and not dashpva_button.isEnabled()
        assert "configured" in nexpy_button.toolTip().lower()
        assert "configured" in dashpva_button.toolTip().lower()
        assert all(
            tools.findChild(QtWidgets.QPushButton, f"e3Tool_{name}").isEnabled()
            for name in (
                "peak_fitting", "phase_fitting", "plot_metadata",
                "roi_statistics",
            )
        )

        nexpy = _executable(tmp_path / "nexpy")
        dashpva = _executable(tmp_path / "DashPVA")
        selected = tmp_path / "selected.nexus"
        selected.write_bytes(b"current")
        ready = ExternalToolRegistry(
            ExternalToolConfig(nexpy, dashpva), environment={},
        )
        tools.reconcile_external(ready.project(
            nexus=ExternalNexusQualification.ready(str(selected.resolve()))
        ))
        assert nexpy_button.isEnabled() and dashpva_button.isEnabled()
    finally:
        tools.close()
        tools.deleteLater()
        qapp.processEvents()


def test_page_qualifies_only_a_fresh_loaded_browse_capture(
    monkeypatch, qapp, tmp_path: Path,
) -> None:
    nexpy = _executable(tmp_path / "nexpy")
    launches = []
    selected = tmp_path / "selected.nexus"
    selected.write_bytes(b"current")
    capture = _loaded_capture(selected)
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        external_tool_registry=ExternalToolRegistry(
            ExternalToolConfig(nexpy_executable=nexpy),
            environment={},
            popen=lambda *args, **kwargs: launches.append((args, kwargs)),
        ),
    )
    try:
        monkeypatch.setattr(
            page, "_capture_current_loaded_browse", lambda: capture,
        )
        qualification = page._qualify_external_nexus(validate_disk=False)
        assert qualification.target == capture.target
        qualification = page._qualify_external_nexus(validate_disk=True)
        assert qualification.target == capture.target
        page._refresh_shell(
            preserve_display=True,
            preserve_scientific=True,
        )
        nexpy_button = page._shell.tools.findChild(
            QtWidgets.QPushButton, "e3Tool_nexpy_selected"
        )
        assert nexpy_button is not None and nexpy_button.isEnabled()

        selected.write_bytes(b"changed after browse admission")
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.LAUNCH_EXTERNAL_VIEWER,
            "nexpy_selected",
        ))
        assert launches == []
        assert "changed after it was loaded" in page._notice_text
        assert not nexpy_button.isEnabled()
        page._refresh_shell(
            preserve_display=True,
            preserve_scientific=True,
        )
        assert not nexpy_button.isEnabled()
        assert "changed after it was loaded" in nexpy_button.toolTip()

        real_operations = page._workspace_operations
        page._workspace_operations = SimpleNamespace(busy=True)
        try:
            qualification = page._qualify_external_nexus(
                validate_disk=False,
            )
            assert qualification.target is None
            assert "workspace operation is active" in qualification.reason
            assert page._external_nexus_refusal is not None
        finally:
            page._workspace_operations = real_operations
        qualification = page._qualify_external_nexus(validate_disk=False)
        assert qualification.target is None
        assert "changed after it was loaded" in qualification.reason

        monkeypatch.setattr(
            page, "_capture_current_loaded_browse", lambda: None,
        )
        qualification = page._qualify_external_nexus(validate_disk=False)
        assert qualification.target is None
        assert "Select one stable" in qualification.reason
        assert page._external_nexus_refusal is None

        replacement = tmp_path / "replacement.nexus"
        replacement.write_bytes(b"replacement")
        replacement_capture = _loaded_capture(replacement)
        monkeypatch.setattr(
            page,
            "_capture_current_loaded_browse",
            lambda: replacement_capture,
        )
        page._refresh_shell(
            preserve_display=True,
            preserve_scientific=True,
        )
        assert nexpy_button.isEnabled()

        legacy = tmp_path / "legacy.nxs"
        legacy.write_bytes(b"legacy")
        legacy_capture = _loaded_capture(legacy)
        monkeypatch.setattr(
            page, "_capture_current_loaded_browse", lambda: legacy_capture,
        )
        qualification = page._qualify_external_nexus(validate_disk=False)
        assert qualification.target is None
        assert "not a processed .nexus" in qualification.reason

        monkeypatch.setattr(
            page, "_capture_current_loaded_browse", lambda: object(),
        )
        qualification = page._qualify_external_nexus(validate_disk=False)
        assert qualification.target is None
        assert "Select one stable" in qualification.reason

        real_lifecycle = page._lifecycle
        page._lifecycle = SimpleNamespace(phase=RunPhase.RUNNING)
        try:
            qualification = page._qualify_external_nexus(validate_disk=False)
            assert qualification.target is None
            assert "acquisition writer is active" in qualification.reason
        finally:
            page._lifecycle = real_lifecycle

        real_operations = page._workspace_operations
        page._workspace_operations = SimpleNamespace(busy=True)
        try:
            qualification = page._qualify_external_nexus(validate_disk=False)
            assert qualification.target is None
            assert "workspace operation is active" in qualification.reason
        finally:
            page._workspace_operations = real_operations
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_page_recaptures_selection_at_click_and_never_owns_child(
    monkeypatch, qapp, tmp_path: Path,
) -> None:
    nexpy = _executable(tmp_path / "nexpy")
    dashpva = _executable(tmp_path / "DashPVA")
    projected = tmp_path / "projected-a.nexus"
    projected.write_bytes(b"projected")
    projected_capture = _loaded_capture(projected)
    selected = tmp_path / "clicked-b.nexus"
    selected.write_bytes(b"clicked")
    selected_path = str(selected.resolve())
    capture = _loaded_capture(selected)
    calls = []

    class HostileDetachedChild:
        pid = 777

        def poll(self):
            raise AssertionError("detached child must not be polled")

        def wait(self):
            raise AssertionError("detached child must not be waited")

        def terminate(self):
            raise AssertionError("detached child must not be terminated")

        def kill(self):
            raise AssertionError("detached child must not be killed")

    def popen(argv, **options):
        calls.append((tuple(argv), options))
        return HostileDetachedChild()

    registry = ExternalToolRegistry(
        ExternalToolConfig(nexpy, dashpva),
        environment={}, popen=popen,
    )
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        external_tool_registry=registry,
    )
    refreshes = []
    tool_reconciles = []
    monkeypatch.setattr(
        page, "_capture_current_loaded_browse", lambda: capture,
    )
    monkeypatch.setattr(
        page, "_refresh_shell", lambda **options: refreshes.append(options),
    )
    monkeypatch.setattr(
        page._shell.tools,
        "reconcile_external_tool",
        lambda projection: tool_reconciles.append(projection),
    )
    try:
        page._shell.tools.reconcile_external(registry.project(
            nexus=ExternalNexusQualification.ready(
                projected_capture.target
            )
        ))
        tool_reconciles.clear()
        assert not page._analysis_slot.owned
        assert not page._workspace_operations.owned
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.LAUNCH_EXTERNAL_VIEWER, "nexpy_selected",
        ))
        assert [argv for argv, _options in calls] == [(nexpy, selected_path)]
        assert not page._analysis_slot.owned
        assert not page._workspace_operations.owned
        assert page._notice_text == (
            "NeXpy launch request accepted for the selected NeXus."
        )
        assert refreshes == []
        assert len(tool_reconciles) == 1

        monkeypatch.setattr(
            page,
            "_capture_current_loaded_browse",
            lambda: (_ for _ in ()).throw(
                AssertionError("DashPVA must not inspect Browse selection")
            ),
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.LAUNCH_EXTERNAL_VIEWER,
            "dashpva_h5viewer",
        ))
        assert [argv for argv, _options in calls] == [
            (nexpy, selected_path),
            (dashpva, "h5viewer"),
        ]
        assert len(tool_reconciles) == 2

        monkeypatch.setattr(
            page, "_capture_current_loaded_browse", lambda: None,
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.LAUNCH_EXTERNAL_VIEWER, "nexpy_selected",
        ))
        assert len(calls) == 2
        assert "current processed .nexus" in page._notice_text
        assert refreshes == []
        assert len(tool_reconciles) == 3

        real_lifecycle = page._lifecycle
        page._lifecycle = SimpleNamespace(phase=RunPhase.RUNNING)
        try:
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.LAUNCH_EXTERNAL_VIEWER,
                "nexpy_selected",
            ))
            assert len(calls) == 2
            assert "acquisition writer is active" in page._notice_text
        finally:
            page._lifecycle = real_lifecycle

        real_operations = page._workspace_operations
        page._workspace_operations = SimpleNamespace(
            busy=True,
            observe_stamp=lambda *_args, **_kwargs: None,
        )
        try:
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.LAUNCH_EXTERNAL_VIEWER,
                "nexpy_selected",
            ))
            assert len(calls) == 2
            assert "workspace operation is active" in page._notice_text
        finally:
            page._workspace_operations = real_operations
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()
