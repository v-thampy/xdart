from __future__ import annotations

from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.core.scan import SourceKind, SourceSpec
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.controls_projection import (
    PONI_FILE,
    PROJECT_ROOT,
    SOURCE_DIRECTORY,
    SOURCE_FILE,
    SOURCE_META,
    SOURCE_SUFFIX,
    SOURCE_TYPE,
    source_mode,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.contracts import SourceObservationRequest
from xdart.gui.tabs.scattering.events import RequestId
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
)
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    image_series_spec,
    is_single_image_spec,
    normalize_image_source_metadata,
    single_image_spec,
)


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _source(root: str, *, generation: int = 0) -> DirectorySourceSpec:
    return DirectorySourceSpec(
        Path(root),
        suffixes=(".h5",),
        generation=generation,
    )


def test_single_image_intent_is_explicit_exact_and_round_trips(
    tmp_path: Path,
) -> None:
    members = tuple(
        tmp_path / f"scan_{index:04d}.tif" for index in range(1, 4)
    )
    for member in members:
        member.write_bytes(b"frame")

    single = single_image_spec(members[1], metadata_format="auto")
    other = tmp_path / "other_0001.tif"
    other.write_bytes(b"frame")
    one_member_series = image_series_spec(other)

    assert single.kind is SourceKind.TIFF_SERIES
    assert single.options["files"] == (str(members[1]),)
    assert single.options["selected_file"] == str(members[1])
    assert is_single_image_spec(single)
    assert source_mode(single) == "Single Image"
    # Member count is not the mode discriminator: an unmarked one-frame
    # series remains an Image Series.
    assert one_member_series.options["files"] == (str(other),)
    assert not is_single_image_spec(one_member_series)
    assert source_mode(one_member_series) == "Image Series"

    restored = RunIntent(source_spec=single).freeze().thaw_source_spec()
    assert restored == single
    assert is_single_image_spec(restored)
    captured = FilesystemSourceAdapter().capture(single, RequestId(1))
    assert captured.source == single
    assert is_single_image_spec(captured.source)


def test_page_keeps_and_restores_explicit_single_image_mode(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "scan_0002.tif"
    selected.write_bytes(b"frame")
    single = single_image_spec(selected)
    store = RunIntentStore(RunIntent(
        poni_file=str(tmp_path / "cal.poni"),
        save_path=str(tmp_path / "processed.nxs"),
        output_mode="Overwrite",
    ))
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        page.select_source(single)
        assert page._source_mode == "Single Image"
        state = page._project_controls(store.snapshot())
        fields = {
            field.path: field
            for field in state.bound_controls.fields
        }
        assert fields[SOURCE_TYPE].value == "Single Image"

        page._switch_source_mode("Image Series")
        assert store.snapshot().thaw().source_spec is None
        page._switch_source_mode("Single Image")
        assert store.snapshot().thaw().source_spec == single
        assert page._source_mode == "Single Image"

        page._cancel_observation()
        page._source_observation = FilesystemSourceAdapter().observe(
            SourceObservationRequest(2, store.revision, single)
        )
        page._run_executor = object()
        monkeypatch.setattr(
            page,
            "_start_permitted",
            lambda: (True, ""),
        )
        page._refresh_shell()
        assert page._shell.run_controls.readinessLabel.toolTip() == (
            "Ready · Int 2D · 1 file"
        )
    finally:
        page._run_executor = None
        page.close_workspace()
        page.close()


def test_host_choosers_commit_paths_and_complete_source_values(
    qapp: QtWidgets.QApplication,
) -> None:
    path_calls: list[tuple[tuple[str, ...], str]] = []
    source_calls: list[tuple[object, str | None]] = []

    def choose_path(
        path: tuple[str, ...],
        current: str,
        _start_directory: str,
    ) -> str | None:
        path_calls.append((path, current))
        return {
            PROJECT_ROOT: "/replacement/project",
            PONI_FILE: "/replacement/calibration.poni",
        }[path]

    replacement = _source("/replacement/source", generation=7)

    def choose_source(current, desired_mode, _start_directory):
        source_calls.append((current, desired_mode))
        return replacement

    store = RunIntentStore(RunIntent(
        source_spec=_source("/raw"),
        project_root="/raw",
        poni_file="/calibration/original.poni",
        save_path="/processed",
        output_mode="Overwrite",
    ))
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        control_path_chooser=choose_path,
        source_selection_chooser=choose_source,
    )
    try:
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_BROWSE,
            path=PROJECT_ROOT,
        ))
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_BROWSE,
            path=PONI_FILE,
        ))
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "Image Directory",
            path=SOURCE_TYPE,
        ))
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_BROWSE,
            path=SOURCE_DIRECTORY,
        ))

        intent = store.snapshot().thaw()
        assert path_calls == [
            (PROJECT_ROOT, "/raw"),
            (PONI_FILE, "/calibration/original.poni"),
        ]
        assert source_calls == [(_source("/raw"), "Image Directory")]
        assert intent.project_root == "/replacement/project"
        assert intent.poni_file == "/replacement/calibration.poni"
        assert intent.source_spec == replacement
        assert store.revision == 3
    finally:
        page.close_workspace()
        page.close()


def test_vnext_choosers_share_and_persist_last_successful_directory(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BD-1 spans control, source, and processed-browser chooser seams."""
    session_file = tmp_path / "session.json"
    monkeypatch.setenv("XDART_SESSION_FILE", str(session_file))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)

    initial_project = tmp_path / "initial-project"
    initial_source = tmp_path / "initial-source"
    initial_processed = tmp_path / "initial-processed"
    picked_control = tmp_path / "picked-control"
    picked_source = tmp_path / "picked-source"
    for directory in (
        initial_project,
        initial_source,
        initial_processed,
        picked_control,
        picked_source,
    ):
        directory.mkdir()
    initial_poni = initial_project / "initial.poni"
    initial_poni.touch()
    selected_poni = picked_control / "selected.poni"
    selected_poni.touch()

    path_starts: list[tuple[tuple[str, ...], str]] = []
    source_starts: list[str] = []
    browser_starts: list[str] = []
    path_results = iter((str(selected_poni), None))

    def choose_path(
        path: tuple[str, ...],
        _current: str,
        start_directory: str,
    ) -> str | None:
        path_starts.append((path, start_directory))
        return next(path_results)

    def choose_source(_current, _desired_mode, start_directory):
        source_starts.append(start_directory)
        return _source(str(picked_source), generation=1)

    def choose_browser(_current: str, start_directory: str) -> None:
        browser_starts.append(start_directory)
        return None

    store = RunIntentStore(RunIntent(
        source_spec=_source(str(initial_source)),
        project_root=str(initial_project),
        poni_file=str(initial_poni),
        save_path=str(initial_processed),
        output_mode="Overwrite",
    ))
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        browser_directory_chooser=choose_browser,
        control_path_chooser=choose_path,
        source_selection_chooser=choose_source,
    )
    try:
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_BROWSE,
            path=PONI_FILE,
        ))
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_BROWSE,
            path=PROJECT_ROOT,
        ))
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_BROWSE,
            path=SOURCE_DIRECTORY,
        ))
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU,
            "File:Open Folder",
        ))

        assert path_starts == [
            (PONI_FILE, str(initial_project)),
            (PROJECT_ROOT, str(picked_control)),
        ]
        assert source_starts == [str(picked_control)]
        assert browser_starts == [str(picked_source)]
    finally:
        page.close_workspace()
        page.close()

    # A reconstructed host reads the same persisted owner.  The cancelled
    # processed-browser dialog above must not have displaced the source pick.
    restart_starts: list[str] = []

    def choose_after_restart(_path, _current, start_directory):
        restart_starts.append(start_directory)
        return None

    restarted = ScatteringWorkspace(
        intents=RunIntentStore(store.snapshot().thaw()),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        control_path_chooser=choose_after_restart,
    )
    try:
        restarted._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_BROWSE,
            path=PONI_FILE,
        ))
        assert restart_starts == [str(picked_source)]

        default_dialog_starts: list[str] = []

        def cancel_default_dialog(_parent, _title, start):
            default_dialog_starts.append(start)
            return ""

        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getExistingDirectory",
            cancel_default_dialog,
        )
        restarted._choose_browser_directory()
        assert default_dialog_starts == [str(picked_source)]
    finally:
        restarted.close_workspace()
        restarted.close()


def test_source_form_is_the_only_picker_and_browse_replaces_complete_source(
    qapp: QtWidgets.QApplication,
) -> None:
    current = SourceSpec(
        Path("/raw"),
        SourceKind.TIFF_SERIES,
        options={
            "selected_file": "/raw/frame_0001.tif",
            "files": ("/raw/frame_0001.tif",),
        },
    )
    replacement = SourceSpec(
        Path("/replacement"),
        SourceKind.TIFF_SERIES,
        options={
            "selected_file": "/replacement/frame_0001.tif",
            "files": ("/replacement/frame_0001.tif",),
        },
    )
    source_calls: list[tuple[object, str | None]] = []

    def choose_source(selected, desired_mode, _start_directory):
        source_calls.append((selected, desired_mode))
        return replacement

    store = RunIntentStore(RunIntent(
        source_spec=current,
        output_mode="Overwrite",
    ))
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        source_selection_chooser=choose_source,
    )
    try:
        controls = page._shell.controls
        state = page._project_controls(store.snapshot())
        fields = {
            field.path: field
            for field in state.bound_controls.fields
        }

        assert controls.source_widget_visible() is False
        assert page._source_status.isHidden()
        assert not page._source_status._choose.isVisibleTo(page)
        assert fields[SOURCE_TYPE].enabled is True
        assert fields[SOURCE_FILE].enabled is True

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_BROWSE,
            path=SOURCE_FILE,
        ))

        assert source_calls == [(
            normalize_image_source_metadata(current),
            "Image Series",
        )]
        assert (
            store.snapshot().thaw().source_spec
            == normalize_image_source_metadata(replacement)
        )
        assert store.revision == 1
    finally:
        page.close_workspace()
        page.close()


def test_source_mode_switch_is_value_only_and_restores_each_mode(
    qapp: QtWidgets.QApplication,
) -> None:
    current = _source("/raw")
    source_calls: list[tuple[object, str | None]] = []

    def choose_source(selected, desired_mode, _start_directory):
        source_calls.append((selected, desired_mode))
        return None

    store = RunIntentStore(RunIntent(
        source_spec=current,
        output_mode="Overwrite",
    ))
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        source_selection_chooser=choose_source,
    )
    try:
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "Image Series",
            path=SOURCE_TYPE,
        ))

        assert source_calls == []
        assert store.snapshot().thaw().source_spec is None
        assert store.revision == 1
        state = page._project_controls(store.snapshot())
        fields = {
            field.path: field
            for field in state.bound_controls.fields
        }
        assert fields[SOURCE_TYPE].value == "Image Series"
        assert fields[SOURCE_FILE].value == ""

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_BROWSE,
            path=SOURCE_FILE,
        ))
        assert source_calls == [(None, "Image Series")]
        assert store.revision == 1

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "Image Directory",
            path=SOURCE_TYPE,
        ))
        assert source_calls == [(None, "Image Series")]
        assert store.snapshot().thaw().source_spec == current
        assert store.revision == 2
    finally:
        page.close_workspace()
        page.close()


def test_blank_directory_mode_defaults_meta_auto(
    qapp: QtWidgets.QApplication,
) -> None:
    current = SourceSpec(
        Path("/raw"),
        SourceKind.TIFF_SERIES,
        options={
            "selected_file": "/raw/frame_0001.tif",
            "files": ("/raw/frame_0001.tif",),
        },
    )
    store = RunIntentStore(RunIntent(
        source_spec=current,
        output_mode="Overwrite",
    ))
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "Image Directory",
            path=SOURCE_TYPE,
        ))
        fields = {
            field.path: field
            for field in page._project_controls(
                store.snapshot()
            ).bound_controls.fields
        }

        assert fields[SOURCE_TYPE].value == "Image Directory"
        assert fields[SOURCE_DIRECTORY].value == ""
        assert fields[SOURCE_SUFFIX].enabled is True
        assert fields[SOURCE_META].enabled is True
        assert fields[SOURCE_META].value == "Auto"

    finally:
        page.close_workspace()
        page.close()


def test_selected_directory_file_and_meta_edits_persist_by_mode(
    qapp: QtWidgets.QApplication,
) -> None:
    store = RunIntentStore(RunIntent(
        source_spec=DirectorySourceSpec(
            Path("/raw"),
            suffixes=(".h5",),
        ),
        output_mode="Overwrite",
    ))
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        fields = {
            field.path: field
            for field in page._project_controls(
                store.snapshot()
            ).bound_controls.fields
        }
        assert fields[SOURCE_META].enabled is True
        assert fields[SOURCE_META].value == "Auto"

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "tif",
            path=SOURCE_SUFFIX,
        ))
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "txt",
            path=SOURCE_META,
        ))

        edited = store.snapshot().thaw().source_spec
        assert type(edited) is DirectorySourceSpec
        assert edited.suffixes == (".tif",)
        assert edited.metadata_format == "txt"
        assert edited.generation == 2

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "Image Series",
            path=SOURCE_TYPE,
        ))
        assert store.snapshot().thaw().source_spec is None
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "Image Directory",
            path=SOURCE_TYPE,
        ))
        assert store.snapshot().thaw().source_spec == edited

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "None",
            path=SOURCE_META,
        ))
        disabled = store.snapshot().thaw().source_spec
        assert type(disabled) is DirectorySourceSpec
        assert disabled.metadata_format is None
    finally:
        page.close_workspace()
        page.close()


def test_intent_boundary_materializes_visible_auto_metadata_policy(
    qapp: QtWidgets.QApplication,
) -> None:
    source = SourceSpec(
        Path("/raw"),
        SourceKind.TIFF_SERIES,
        options={
            "selected_file": "/raw/frame_0001.tif",
            "files": ("/raw/frame_0001.tif",),
        },
    )
    store = RunIntentStore(RunIntent(
        source_spec=source,
        output_mode="Overwrite",
    ))
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        fields = {
            field.path: field
            for field in page._project_controls(
                store.snapshot()
            ).bound_controls.fields
        }
        assert fields[SOURCE_META].value == "Auto"
        assert "metadata_format" not in source.options
        typed = store.snapshot().thaw().source_spec
        assert type(typed) is SourceSpec
        assert typed.options["metadata_format"] == "auto"
        assert store.revision == 0

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "Auto",
            path=SOURCE_META,
        ))

        edited = store.snapshot().thaw().source_spec
        assert type(edited) is SourceSpec
        assert edited.options["metadata_format"] == "auto"
        assert store.revision == 0
    finally:
        page.close_workspace()
        page.close()


def test_unowned_control_and_analysis_actions_fail_closed(
    qapp: QtWidgets.QApplication,
) -> None:
    store = RunIntentStore(RunIntent(output_mode="Overwrite"))
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    notices: list[str] = []
    page.noticeChanged.connect(notices.append)
    try:
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.CONTROL_ACTION,
            "reintegrate_1d",
        ))
        assert notices[-1] == (
            "Reintegrate 1D is unavailable: no vNext operation service is "
            "mounted."
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.ANALYSIS_ACTION,
            "peak_fit",
        ))
        assert notices[-1] == (
            "Analysis action is unavailable: no vNext analysis service is "
            "mounted."
        )
        assert store.revision == 0
    finally:
        page.close_workspace()
        page.close()
