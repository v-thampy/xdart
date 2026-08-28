"""Focused standalone-Make-Mask operation oracle."""
from __future__ import annotations
import os, stat, subprocess
from dataclasses import fields, replace
from pathlib import Path
from threading import Event, current_thread
from types import SimpleNamespace
import numpy as np
import h5py
import pytest
import tifffile
from PIL import Image
from fabio.edfimage import EdfImage
from silx.io.url import DataUrl
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets
from xdart.gui.tabs.scattering import experiment_authoring as authoring
from xdart.gui.tabs.scattering import page as page_module
from xdart.gui.tabs.scattering.adapters import external_operation
from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import SourceFileState
from xdart.gui.tabs.scattering.controls_inventory import MASK_FILE
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity
from xdart.gui.tabs.scattering.experiment_authoring import (
    AssetValidationRequest, AuthoredAssetCandidate, MaskProof, MaskRequest,
    MaskResult, prepare_mask_request, run_mask,
)
from xdart.gui.tabs.scattering.operation_values import (
    OperationContextStamp, OperationIdentity, OperationTerminal,
    OperationTerminalStatus, OperationUpdate,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection, ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.io.image import load_mask
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.readiness import ControlAction, SectionId
from xrd_tools.session.run_configuration import RunIntent

@pytest.fixture
def qapp(): return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
def _binary(tmp_path: Path, monkeypatch) -> Path:
    binary = tmp_path / "bin" / "pyFAI-drawmask"; binary.parent.mkdir(exist_ok=True)
    binary.write_text("fixture"); binary.chmod(0o700)
    interpreter = binary.parent / "python"; interpreter.write_text("fixture"); interpreter.chmod(0o700)
    monkeypatch.setattr(authoring.sys, "executable", str(interpreter)); monkeypatch.setenv("PATH", str(binary.parent))
    return binary.resolve()
def _tiff(path: Path, data, **options) -> Path:
    tifffile.imwrite(path, np.asarray(data), **options); return path
def _request(tmp_path: Path, monkeypatch, data=None) -> MaskRequest:
    _binary(tmp_path, monkeypatch); source = tmp_path / "chosen.tiff"
    _tiff(source, np.arange(8, dtype=np.uint16).reshape(2, 4) if data is None else data)
    return prepare_mask_request(str(source))
def _mask_path(private: Path) -> Path:
    return private.with_name(os.path.splitext(private.name)[0] + "-mask.edf")
def _hdf_url(path: Path, data_path: str, frame: int | None = None) -> str:
    return DataUrl(
        file_path=str(path), data_path=data_path,
        data_slice=None if frame is None else (frame,), scheme="silx",
    ).path()
def _install_process(monkeypatch, mask=None, *, code=0, hook=None, missing=False, stderr=b""):
    calls = []
    class Process:
        pid = 7373
        def __init__(self, argv, **options):
            private, output = Path(argv[1]), _mask_path(Path(argv[1])); calls.append((tuple(argv), options, stat.S_IMODE(private.stat().st_mode)))
            options["stderr"].write(stderr)
            if not missing:
                if mask is None: output.write_bytes(b"not-edf")
                else:
                    payload = np.asarray(mask); EdfImage(data=payload.astype(np.uint8) if payload.dtype == bool else payload).write(str(output))
            if hook is not None: hook(private, output)
            if output.exists(): output.chmod(0o666)
        def wait(self, *, timeout): return code
        def terminate(self): calls.append("terminate")
        def kill(self): calls.append("kill")
    monkeypatch.setattr(authoring, "_popen", Process); return calls


def test_resolver_prefers_real_python_sibling_then_validated_path_fallback(
    tmp_path, monkeypatch,
) -> None:
    runtime = tmp_path / "runtime" / "bin"
    fallback = tmp_path / "fallback"
    runtime.mkdir(parents=True)
    fallback.mkdir()
    interpreter = runtime / "python"
    sibling = runtime / "pyFAI-drawmask"
    path_tool = fallback / "pyFAI-drawmask"
    for path in (interpreter, sibling, path_tool):
        path.write_text("fixture")
        path.chmod(0o700)
    monkeypatch.setattr(authoring.sys, "executable", str(interpreter))
    monkeypatch.setenv("PATH", str(fallback))
    assert authoring.resolve_mask_executable() == str(sibling)
    sibling.unlink()
    assert authoring.resolve_mask_executable() == str(path_tool)
    monkeypatch.setenv("PATH", "")
    assert authoring.resolve_mask_executable(str(path_tool)) == str(path_tool)
    path_tool.chmod(0o600)
    assert authoring.resolve_mask_executable(str(path_tool)) is None
def _direct(request: MaskRequest, *, seal=lambda _identity: True, cancelled=None):
    progress = []; cancelled = Event() if cancelled is None else cancelled
    terminal = run_mask(request, OperationIdentity(1), cancelled, lambda *value: progress.append(value), seal)
    assert type(terminal.payload) is MaskResult
    assert authoring.mask_terminal_result_valid(terminal, request)
    return terminal, progress
def _close(page, qapp):
    page.close_workspace(); page.deleteLater(); qapp.processEvents()


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(
            forged, field.name,
            changes.get(field.name, getattr(value, field.name)),
        )
    return forged

def test_parent_red_make_mask_command_uses_explicit_tiff_chooser(
    tmp_path, monkeypatch
) -> None:
    binary = tmp_path / "bin" / "pyFAI-drawmask"; binary.parent.mkdir()
    binary.write_text("fixture"); binary.chmod(0o700)
    interpreter = binary.parent / "python"; interpreter.write_text("fixture"); interpreter.chmod(0o700)
    monkeypatch.setattr(authoring.sys, "executable", str(interpreter)); monkeypatch.setenv("PATH", str(binary.parent))
    source = tmp_path / "chosen.tiff"; source.write_bytes(b"explicit TIFF"); chosen = []
    def chooser(asset, _start): chosen.append(asset); return str(source)
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = RunIntentStore(RunIntent(project_root=str(tmp_path)))
    page = ScatteringWorkspace(intents=store, lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(), authoring_source_chooser=chooser)
    command = ShellCommand(ShellCommandKind.CONTROL_ACTION, "make_mask")
    try:
        monkeypatch.setattr(page._operation_slot, "begin_mask", lambda *_args: None); page._handle_shell_command(command)
        assert chosen == ["mask"]
    finally: _close(page, qapp)


def test_live_mask_two_step_qualifies_eiger_metadata_then_stages_exact_frame(
    tmp_path, monkeypatch, qapp,
) -> None:
    from xdart.gui.pages import scattering_workspace as workspace

    _binary(tmp_path, monkeypatch)
    target = tmp_path / "scan_data_000001.h5"
    frames = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
    with h5py.File(target, "w") as handle:
        handle.create_dataset("/entry/data/data", data=frames)
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle.require_group("/entry/data")["data_000001"] = \
            h5py.ExternalLink(target.name, "/entry/data/data")
    chooser_calls = []
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName",
        lambda *args: (chooser_calls.append(args) or (str(master), "")),
    )
    real_qualify = authoring._mask_hdf_frame
    qualification_reads = []

    def qualify(*args, **kwargs):
        qualification_reads.append(kwargs.get("read"))
        return real_qualify(*args, **kwargs)

    monkeypatch.setattr(authoring, "_mask_hdf_frame", qualify)
    base_dialog = workspace._mask_hdf_dialog_type()

    class Dialog(base_dialog):
        def exec(self):
            assert self.findChild(
                QtWidgets.QLabel, "maskHdfSource").text() == str(master)
            assert self.dataset_path.text() == "/entry/data/data_000001"
            assert not self.no_frame.isChecked()
            assert self.frame_index.value() == 0
            self.frame_index.setValue(1)
            self.accept()
            return self.result()

    monkeypatch.setattr(workspace, "_mask_hdf_dialog_type", lambda: Dialog)
    parent = QtWidgets.QWidget()
    try:
        selected = workspace._authoring_source_chooser(parent)(
            "mask", str(tmp_path))
    finally:
        parent.deleteLater()
        qapp.processEvents()
    expected = _hdf_url(master, "/entry/data/data_000001", 1)
    assert selected == expected
    assert qualification_reads == [False]
    assert chooser_calls[0][1:] == (
        "Choose TIFF or HDF5/NeXus source for mask", str(tmp_path),
        "Mask sources (*.tif *.tiff *.h5 *.hdf5 *.nxs *.nexus)",
    )

    request = prepare_mask_request(selected)
    assert request.source_path == str(master)
    assert request.hdf_target_path == str(target)
    assert request.final_path == str(tmp_path / "scan_master-mask.edf")
    staged = []
    calls = _install_process(
        monkeypatch, np.ones((2, 4), dtype=np.uint8),
        hook=lambda private, _output: staged.append(tifffile.imread(private)),
    )
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.RETURNED
    assert np.array_equal(staged, [frames[1]])
    assert Path(calls[0][0][1]).name == "scan_master.tiff"


def test_hdf_selector_refuses_outside_link_before_target_open_or_pixel_read(
    tmp_path, monkeypatch, qapp,
) -> None:
    from xdart.gui.pages import scattering_workspace as workspace

    outside = tmp_path.parent / f"{tmp_path.name}-outside-selector.h5"
    with h5py.File(outside, "w") as handle:
        handle.create_dataset(
            "/entry/data/data", data=np.ones((2, 4), dtype="u2"))
    master = tmp_path / "outside_master.h5"
    with h5py.File(master, "w") as handle:
        handle.require_group("/entry/data")["data_000001"] = \
            h5py.ExternalLink(
                f"../{outside.name}", "/entry/data/data")
    opened = []
    real_file = h5py.File

    def tracked_file(path, *args, **kwargs):
        opened.append(str(Path(path)))
        return real_file(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", tracked_file)
    parent = QtWidgets.QWidget()
    dialog = workspace._mask_hdf_dialog_type()(parent, str(master))
    try:
        assert dialog.findChild(
            QtWidgets.QLabel, "maskHdfSource").text() == str(master)
        dialog.accept()
        assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
        assert dialog.selected_source is None
        assert "same-directory" in dialog.error_label.text()
        assert opened == [str(master)]
        dialog.reject()
    finally:
        dialog.deleteLater()
        parent.deleteLater()
        qapp.processEvents()


def test_hdf_selector_explicit_2d_control_omits_frame_slice(
    tmp_path, qapp,
) -> None:
    from xdart.gui.pages import scattering_workspace as workspace

    source = tmp_path / "direct.nxs"
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "/entry/image", data=np.ones((2, 4), dtype="u2"))
    parent = QtWidgets.QWidget()
    dialog = workspace._mask_hdf_dialog_type()(parent, str(source))
    try:
        dialog.dataset_path.setText("/entry/image")
        dialog.no_frame.setChecked(True)
        assert not dialog.frame_index.isEnabled()
        dialog.accept()
        assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
        assert dialog.error_label.text() == ""
        selected = DataUrl(dialog.selected_source)
        assert selected.file_path() == str(source)
        assert selected.data_path() == "/entry/image"
        assert selected.data_slice() is None
    finally:
        dialog.deleteLater()
        parent.deleteLater()
        qapp.processEvents()


def test_live_mask_chooser_returns_tiff_after_one_file_dialog(
    tmp_path, monkeypatch, qapp,
) -> None:
    from xdart.gui.pages import scattering_workspace as workspace

    source = tmp_path / "detector.tiff"
    calls = []
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName",
        lambda *args: (calls.append(args) or (str(source), "")),
    )
    monkeypatch.setattr(
        workspace, "_mask_hdf_dialog_type",
        lambda: pytest.fail("TIFF opened the HDF frame selector"),
    )
    parent = QtWidgets.QWidget()
    try:
        assert workspace._authoring_source_chooser(parent)(
            "mask", str(tmp_path)) \
            == str(source)
        assert len(calls) == 1
    finally:
        parent.deleteLater()
        qapp.processEvents()

def test_page_focus_selected_assets_cancel_and_projection_are_exact(tmp_path, monkeypatch, qapp) -> None:
    _binary(tmp_path, monkeypatch); source = _tiff(tmp_path / "selected.tiff", np.ones((2, 4), dtype=np.uint16))
    poni, mask = tmp_path / "current.poni", tmp_path / "current-mask.edf"; chosen = []
    def chooser(asset, start): chosen.append((asset, start)); return str(source)
    store = RunIntentStore(RunIntent(project_root=str(tmp_path), poni_file=str(poni), mask_file=str(mask)))
    page = ScatteringWorkspace(intents=store, lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(), authoring_source_chooser=chooser)
    displayed = DisplayFrameKey(RunIdentity(8, "displayed"), "scan", str(tmp_path / "displayed.nxs"), 7, 1)
    page._context_controller._runtime._acquisition_navigation = FrameNavigationProjection((displayed,), displayed, (displayed,))
    real_prepare, prepared, begun = page_module.prepare_mask_request, [], []; identity = OperationIdentity(9)
    def capture(selected, *, current_poni="", current_mask=""):
        prepared.append((selected, current_poni, current_mask)); return real_prepare(selected, current_poni=current_poni, current_mask=current_mask)
    monkeypatch.setattr(page_module, "prepare_mask_request", capture)
    monkeypatch.setattr(page._operation_slot, "begin_mask", lambda request, stamp: begun.append((request, stamp)) or identity)
    command = ShellCommand(ShellCommandKind.CONTROL_ACTION, "make_mask")
    try:
        monkeypatch.setattr(page, "_commit_focused_control_edit_for_run", lambda: False); page._handle_shell_command(command)
        assert chosen == [] and begun == []
        monkeypatch.setattr(page, "_commit_focused_control_edit_for_run", lambda: True); page._handle_shell_command(command)
        assert len(chosen) == 1 and chosen[0][0] == "mask" and Path(chosen[0][1]).is_dir()
        assert prepared == [(str(source), str(poni), str(mask))]
        assert len(begun) == 1 and begun[0][0].source_path == str(source) and begun[0][0].source_path != displayed.artifact
        assert type(begun[0][1]) is OperationContextStamp and page._mask_identity is identity
        direct = project_controls(store.snapshot(), None, RunPhase.IDLE, mask_available=True, mask_dependency_available=True)
        action = next(item for item in direct.profile.actions_for(SectionId.EXPERIMENT) if item.action is ControlAction.MAKE_MASK)
        assert action.enabled
        assert action.reason == (
            "Choose an explicit TIFF, HDF5, or NeXus image source and create "
            "its beside-source EDF mask."
        )
        active = project_controls(store.snapshot(), None, RunPhase.IDLE, operation_busy=True, mask_active=True)
        actions = active.profile.actions_for(SectionId.EXPERIMENT); assert actions[1].label == "Cancel Mask" and actions[1].enabled and not actions[0].enabled
        cancelled = []; monkeypatch.setattr(OperationSlot, "current_identity", property(lambda _slot: identity)); monkeypatch.setattr(page._operation_slot, "cancel", lambda item: cancelled.append(item) or True)
        page._refresh_shell(); page.show(); qapp.processEvents()
        assert any(button.text() == "Cancel Mask" and button.isVisible() for button in page.findChildren(QtWidgets.QPushButton))
        page._handle_shell_command(command); assert cancelled == [identity] and len(chosen) == 1
    finally: _close(page, qapp)

def test_request_canonicalizes_alias_and_begin_revalidates_fixed_facts(tmp_path, monkeypatch) -> None:
    binary = _binary(tmp_path, monkeypatch); source = _tiff(tmp_path / "source.tif", np.ones((2, 2), dtype=np.uint8))
    alias = tmp_path / "alias.tiff"; alias.symlink_to(source); request = prepare_mask_request(str(alias))
    assert request == MaskRequest(
        str(source.resolve()), str(tmp_path / "source-mask.edf"), str(binary),
        SourceFileState.capture(binary), SourceFileState.capture(source),
        str(tmp_path), authoring._directory_identity(tmp_path),
    )
    alias.unlink(); alias.symlink_to(tmp_path / "missing.tif"); assert request.source_path == str(source.resolve())
    with pytest.raises(ValueError, match="unavailable"): prepare_mask_request(str(alias))
    wrong = tmp_path / "source.png"; wrong.write_bytes(b"x")
    with pytest.raises(ValueError, match="suffix"): prepare_mask_request(str(wrong))
    slot, calls = OperationSlot(), []
    monkeypatch.setattr(slot, "_begin", lambda *args: calls.append(args) or OperationIdentity(9)); monkeypatch.setenv("PATH", "")
    assert slot.begin_mask(request, OperationContextStamp(0)) == OperationIdentity(9) and len(calls) == 1
    other = tmp_path / "other"; other.write_text("x"); other.chmod(0o700)
    forged = MaskRequest(
        request.source_path, request.final_path, str(other),
        SourceFileState.capture(other), request.source_state,
        request.source_directory, request.directory_identity,
    )
    assert slot.begin_mask(forged, OperationContextStamp(0)) is None
    Path(request.final_path).write_text("foreign"); assert slot.begin_mask(request, OperationContextStamp(0)) is None and len(calls) == 1
    assert Path(request.final_path).read_text() == "foreign" and slot.owned is False
    Path(request.final_path).unlink(); binary.unlink(); assert slot.begin_mask(request, OperationContextStamp(0)) is None


def test_hdf_requires_exact_numeric_frame_and_freezes_hard_dataset(
    tmp_path, monkeypatch,
) -> None:
    binary = _binary(tmp_path, monkeypatch)
    source = tmp_path / "direct.nxs"
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "/entry/data", data=np.arange(8, dtype="u2").reshape(2, 4),
        )
        handle.create_dataset(
            "/entry/stack", data=np.ones((3, 2, 4), dtype="u2"),
        )
    with pytest.raises(ValueError, match="exact dataset/frame"):
        prepare_mask_request(str(source))
    with pytest.raises(ValueError, match="bounded numeric frame"):
        prepare_mask_request(_hdf_url(source, "/entry/stack"))
    selected = _hdf_url(source, "/entry/data")
    request = prepare_mask_request(selected)
    assert request.source_path == str(source)
    assert request.exact_hdf_url == selected
    assert request.hdf_target_path == ""
    assert request.hdf_target_state is None
    assert request.executable == str(binary)


def test_eiger_external_frame_stages_private_tiff_and_requalifies_adoption(
    tmp_path, monkeypatch,
) -> None:
    _binary(tmp_path, monkeypatch)
    target = tmp_path / "scan_data_000001.h5"
    frames = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
    with h5py.File(target, "w") as handle:
        handle.create_dataset("/entry/data/data", data=frames)
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        group = handle.require_group("/entry/data")
        group["data_000001"] = h5py.ExternalLink(
            target.name, "/entry/data/data",
        )
    selected = _hdf_url(master, "/entry/data/data_000001", 1)
    request = prepare_mask_request(selected)
    assert request.hdf_target_path == str(target)
    assert request.hdf_target_state == SourceFileState.capture(target)
    staged = []

    def inspect(private, _output):
        staged.append((
            private.name, private.suffix, stat.S_IMODE(private.stat().st_mode),
            tifffile.imread(private),
        ))

    calls = _install_process(
        monkeypatch, np.ones((2, 4), dtype=np.uint8), hook=inspect,
    )
    terminal, progress = _direct(request)
    result = terminal.payload
    assert terminal.status is OperationTerminalStatus.RETURNED
    assert [item[0] for item in progress] == [
        "copy", "launch", "qualify", "publish",
    ]
    assert staged[0][:3] == ("scan_master.tiff", ".tiff", 0o600)
    assert np.array_equal(staged[0][3], frames[1])
    assert calls[0][0] == (
        request.executable, str(Path(result.cwd) / "scan_master.tiff"),
    )
    assert selected not in calls[0][0]
    assert result.proof.source_sha256 == result.proof.staged_sha256
    assert result.proof.shape == (2, 4)
    candidate = AuthoredAssetCandidate(
        "mask", request.final_path, result.proof, result.final_state,
        request.source_path,
    )
    validation = AssetValidationRequest(
        "mask", request.final_path, result.proof.shape, candidate, request,
    )
    assert authoring.validate_authored_asset(validation).candidate is candidate

    with h5py.File(target, "r+") as handle:
        handle["/entry/data/data"][1] = np.zeros((2, 4), dtype=np.uint16)
    rebound = replace(
        request, hdf_target_state=SourceFileState.capture(target),
    )
    drifted = AssetValidationRequest(
        "mask", request.final_path, result.proof.shape, candidate, rebound,
    )
    with pytest.raises(ValueError, match="generated mask changed"):
        authoring.validate_authored_asset(drifted)


def test_external_hdf_outside_chained_symlink_and_target_drift_refuse(
    tmp_path, monkeypatch,
) -> None:
    _binary(tmp_path, monkeypatch)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.h5"
    with h5py.File(outside, "w") as handle:
        handle.create_dataset("/entry/data/data", data=np.ones((2, 4), dtype="u2"))
    outside_master = tmp_path / "outside_master.h5"
    with h5py.File(outside_master, "w") as handle:
        handle["frame"] = h5py.ExternalLink(
            f"../{outside.name}", "/entry/data/data",
        )
    with pytest.raises(ValueError, match="same-directory"):
        prepare_mask_request(_hdf_url(outside_master, "/frame"))

    real_target = tmp_path / "real.h5"
    with h5py.File(real_target, "w") as handle:
        handle.create_dataset("/data", data=np.ones((2, 4), dtype="u2"))
    alias = tmp_path / "alias.h5"
    alias.symlink_to(real_target)
    symlink_master = tmp_path / "symlink_master.h5"
    with h5py.File(symlink_master, "w") as handle:
        handle["frame"] = h5py.ExternalLink(alias.name, "/data")
    with pytest.raises(ValueError, match="regular direct-child"):
        prepare_mask_request(_hdf_url(symlink_master, "/frame"))

    chained_target = tmp_path / "chained.h5"
    with h5py.File(chained_target, "w") as handle:
        handle["frame"] = h5py.ExternalLink(real_target.name, "/data")
    chained_master = tmp_path / "chained_master.h5"
    with h5py.File(chained_master, "w") as handle:
        handle["frame"] = h5py.ExternalLink(chained_target.name, "/frame")
    with pytest.raises(ValueError, match="locally owned"):
        prepare_mask_request(_hdf_url(chained_master, "/frame"))

    target = tmp_path / "drift_target.h5"
    with h5py.File(target, "w") as handle:
        handle.create_dataset("/data", data=np.ones((2, 4), dtype="u2"))
    master = tmp_path / "drift_master.h5"
    with h5py.File(master, "w") as handle:
        handle["frame"] = h5py.ExternalLink(target.name, "/data")
    request = prepare_mask_request(_hdf_url(master, "/frame"))
    with h5py.File(target, "r+") as handle:
        handle["/data"][...] = 2
    calls = _install_process(
        monkeypatch, np.ones((2, 4), dtype=np.uint8),
    )
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED
    assert "source context changed before launch" in terminal.diagnostic
    assert calls == []

def test_encoded_pixel_itemsize_and_eiger_metadata_caps_precede_decode(tmp_path, monkeypatch) -> None:
    source = _tiff(tmp_path / "source.tiff", np.arange(6, dtype=np.uint16).reshape(2, 3)); private = tmp_path / "private.tiff"
    monkeypatch.setattr(authoring, "_TIFF_LIMIT", source.stat().st_size)
    _, _, staged, _ = authoring._copy_tiff(source, private); assert stat.S_IMODE(private.stat().st_mode) == 0o600
    over = tmp_path / "over.tiff"; monkeypatch.setattr(authoring, "_TIFF_LIMIT", source.stat().st_size - 1)
    with pytest.raises(ValueError, match="512 MiB"): authoring._copy_tiff(source, over)
    assert not over.exists(); monkeypatch.setattr(authoring, "_TIFF_LIMIT", 512 << 20)
    real_read, decodes = authoring.TifImage.read, []
    def counted(self, *args, **kwargs): decodes.append(1); return real_read(self, *args, **kwargs)
    monkeypatch.setattr(authoring.TifImage, "read", counted); monkeypatch.setattr(authoring, "_PIXEL_LIMIT", 6); monkeypatch.setattr(authoring, "_DECODED_LIMIT", 12)
    assert authoring._qualify_tiff(private, staged)[0] == (2, 3) and len(decodes) == 1
    monkeypatch.setattr(authoring, "_PIXEL_LIMIT", 5)
    with pytest.raises(ValueError, match="envelope"): authoring._qualify_tiff(private, staged)
    monkeypatch.setattr(authoring, "_PIXEL_LIMIT", 1 << 25); monkeypatch.setattr(authoring, "_DECODED_LIMIT", 11)
    with pytest.raises(ValueError, match="envelope"): authoring._qualify_tiff(private, staged)
    monkeypatch.setattr(authoring, "_pillow_header", lambda _stream: ((2, 3), np.dtype(np.longdouble), np.dtype(np.longdouble).itemsize))
    with pytest.raises(ValueError, match="envelope"): authoring._qualify_tiff(private, staged)
    assert len(decodes) == 1
    monkeypatch.setattr(authoring, "_pillow_header", lambda _stream: ((4096, 4096), np.dtype("u4"), 4))
    monkeypatch.setattr(authoring, "_DECODED_LIMIT", 256 << 20)
    monkeypatch.setattr(authoring.TifImage, "read", lambda *_args: (_ for _ in ()).throw(RuntimeError("decode reached")))
    with pytest.raises(RuntimeError, match="decode reached"): authoring._qualify_tiff(private, staged)

@pytest.mark.parametrize("options", ({}, {"bigtiff": True}, {"tile": (16, 16)}, {"compression": "deflate"}))
def test_classic_bigtiff_tiled_compressed_union(tmp_path, options) -> None:
    path = _tiff(tmp_path / "union.tiff", np.ones((16, 16), dtype=np.uint16), **options)
    assert authoring._qualify_tiff(path, SourceFileState.capture(path)) == ((16, 16), np.dtype("u2").str)
@pytest.mark.parametrize("dtype", (np.uint32, np.int16, np.int8, np.float32, np.uint64, np.float64))
def test_pillow_container_mode_preserves_tagged_numeric_dtype(tmp_path, dtype) -> None:
    path = _tiff(tmp_path / "numeric.tiff", np.ones((2, 4), dtype=dtype))
    assert authoring._qualify_tiff(path, SourceFileState.capture(path))[1] == np.dtype(dtype).str
def test_fabio_header_fallback_is_explicitly_rewound(tmp_path, monkeypatch) -> None:
    path = _tiff(tmp_path / "fallback.tiff", np.ones((2, 4), dtype=np.uint16))
    monkeypatch.setattr(authoring, "_pillow_header", lambda stream: (stream.read(7), None)[1])
    monkeypatch.setattr(authoring, "_fabio_header", lambda stream: (pytest.fail("fallback was not rewound") if stream.tell() else ((2, 4), np.dtype("u2"), 2)))
    assert authoring._qualify_tiff(path, SourceFileState.capture(path))[0] == (2, 4)
def test_one_bit_and_semantic_multiframe_color_refuse_without_decode(tmp_path, monkeypatch) -> None:
    bit = tmp_path / "bit.tif"; Image.new("1", (4, 3), 1).save(bit)
    assert authoring._qualify_tiff(bit, SourceFileState.capture(bit))[0] == (3, 4)
    multi = tmp_path / "multi.tif"
    with tifffile.TiffWriter(multi) as writer: writer.write(np.ones((2, 2), dtype=np.uint8)); writer.write(np.ones((2, 2), dtype=np.uint8))
    color = _tiff(tmp_path / "color.tif", np.ones((2, 2, 3), dtype=np.uint8), photometric="rgb")
    palette = tmp_path / "palette.tif"; Image.new("P", (2, 2)).save(palette)
    monkeypatch.setattr(authoring.TifImage, "read", lambda *_args: pytest.fail("semantic refusal decoded pixels"))
    for path in (multi, color, palette):
        with pytest.raises(ValueError): authoring._qualify_tiff(path, SourceFileState.capture(path))
@pytest.mark.parametrize("header", (((3, 3), np.dtype("u2"), 2), ((2, 4), np.dtype("u4"), 4)))
def test_post_decode_lying_header_is_refused(tmp_path, monkeypatch, header) -> None:
    path = _tiff(tmp_path / "lying.tif", np.ones((2, 4), dtype=np.uint16)); monkeypatch.setattr(authoring, "_pillow_header", lambda _stream: header)
    with pytest.raises(ValueError, match="contradicts"): authoring._qualify_tiff(path, SourceFileState.capture(path))
@pytest.mark.parametrize("array, header, limit", (
    (np.ones((1, 2, 4), dtype=np.uint16), ((2, 4), np.dtype("u2"), 2), 256 << 20),
    (np.ones((2, 4), dtype=np.uint64), ((2, 4), np.dtype("u1"), 1), 8),
))
def test_post_decode_rank_and_nbytes_are_rechecked(tmp_path, monkeypatch, array, header, limit) -> None:
    path = _tiff(tmp_path / "decoded.tif", np.ones((2, 4), dtype=np.uint16)); decoded = SimpleNamespace(data=array, close=lambda: None)
    monkeypatch.setattr(authoring, "_pillow_header", lambda _stream: header); monkeypatch.setattr(authoring.TifImage, "read", lambda *_args: decoded)
    monkeypatch.setattr(authoring, "_DECODED_LIMIT", limit)
    with pytest.raises(ValueError, match="contradicts"): authoring._qualify_tiff(path, SourceFileState.capture(path))

@pytest.mark.parametrize("mask", (
    np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool),
    np.array([[0, -1, 2, 0], [0, 3, -4, 0]], dtype=np.int16),
    np.array([[0, 1, 2, 0], [0, 3, 4, 0]], dtype=np.uint8),
    np.array([[0, -1, 2.5, np.nan], [0, 3, -4, 0]], dtype=np.float32),
))
def test_bool_int_uint_float_public_truth_private_launch_and_publication(tmp_path, monkeypatch, mask) -> None:
    request = _request(tmp_path, monkeypatch, np.ones(mask.shape, dtype=np.uint32)); calls = _install_process(monkeypatch, mask)
    if mask.dtype == bool: monkeypatch.setattr(authoring, "read_image", lambda *_args, **_kwargs: mask.copy())
    terminal, progress = _direct(request); result = terminal.payload
    assert terminal.status is OperationTerminalStatus.RETURNED and result.published and result.proof is not None
    assert [item[0] for item in progress] == ["copy", "launch", "qualify", "publish"]
    assert len(calls) == 1
    argv, options, mode = calls[0]; assert argv == (request.executable, str(Path(options["cwd"]) / Path(request.source_path).name))
    assert options["cwd"] == result.cwd and options["shell"] is False and mode == 0o600
    assert options["stdin"] is options["stdout"] is subprocess.DEVNULL
    assert options["stderr"] is not subprocess.DEVNULL and options["stderr"].closed
    assert ("creationflags" in options) is authoring._WINDOWS and ("start_new_session" in options) is not authoring._WINDOWS
    expected = mask != 0
    if mask.dtype.kind == "f": expected |= np.isnan(mask)
    assert np.array_equal(load_mask(request.final_path), expected)
    assert result.proof.coercion_policy == "zero-false-real-nonzero-true-nan-true-v1"
    assert result.proof.source_sha256 == result.proof.staged_sha256 and stat.S_IMODE(Path(request.final_path).stat().st_mode) == 0o600
    assert result.final_state == SourceFileState.capture(Path(request.final_path))
    assert not Path(result.cwd).exists()


def test_nonzero_mask_surfaces_bounded_private_stderr_without_leak(
    tmp_path, monkeypatch,
) -> None:
    request = _request(tmp_path, monkeypatch)
    observed = []

    def inspect_stderr(private, _output):
        stage = private.parent
        stream = calls[0][1]["stderr"]
        observed.append((
            stat.S_IMODE(os.fstat(stream.fileno()).st_mode),
            tuple(stage.glob(".xdart-authoring-stderr-*")),
        ))

    payload = (b"x" * (authoring._CHILD_STDERR_BYTES_LIMIT + 32)
               + b"\nImportError: silx Qt binding failed\n")
    calls = _install_process(
        monkeypatch, code=19, hook=inspect_stderr, stderr=payload,
    )
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED
    assert "pyFAI-drawmask exited with status 19" in terminal.diagnostic
    assert "[stderr truncated]" in terminal.diagnostic
    assert "ImportError: silx Qt binding failed" in terminal.diagnostic
    assert len(terminal.diagnostic.encode("utf-8")) \
        <= authoring._DIAGNOSTIC_BYTES_LIMIT
    assert observed == [(0o600, ())]
    assert calls[0][1]["stderr"].closed
    assert not Path(terminal.payload.cwd).exists()

@pytest.mark.parametrize("bad", (
    np.ones(8, dtype=np.uint8), np.ones((3, 4), dtype=np.uint8),
    np.ones((2, 4), dtype=np.complex64), np.ones((2, 4), dtype=object), np.ones((2, 4), dtype="U1"),
))
def test_wrong_rank_shape_complex_and_string_masks_refuse(tmp_path, monkeypatch, bad) -> None:
    request = _request(tmp_path, monkeypatch); _install_process(monkeypatch)
    monkeypatch.setattr(authoring, "read_image", lambda *_args, **_kwargs: bad)
    assert _direct(request)[0].status is OperationTerminalStatus.FAILED and not Path(request.final_path).exists()
def test_mask_decoded_encoded_and_inverted_truth_refuse(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path, monkeypatch); _install_process(monkeypatch); array = np.ones((2, 4), dtype=np.uint8)
    monkeypatch.setattr(authoring, "read_image", lambda *_args, **_kwargs: array); monkeypatch.setattr(authoring, "_DECODED_LIMIT", array.nbytes - 1)
    assert _direct(request)[0].status is OperationTerminalStatus.FAILED
    monkeypatch.setattr(authoring, "_DECODED_LIMIT", 256 << 20); monkeypatch.setattr(authoring, "_MASK_LIMIT", 0)
    assert _direct(request)[0].status is OperationTerminalStatus.FAILED
    monkeypatch.setattr(authoring, "_MASK_LIMIT", 64 << 20); monkeypatch.setattr(authoring, "load_mask", lambda value: np.zeros(value.shape, dtype=bool))
    assert _direct(request)[0].status is OperationTerminalStatus.FAILED

@pytest.mark.parametrize("failure", ("source", "source_sha", "stage", "stage_sha", "seal", "foreign", "nonzero", "missing", "empty"))
def test_drift_foreign_nonzero_and_missing_output_are_zero_effect(tmp_path, monkeypatch, failure) -> None:
    request = _request(tmp_path, monkeypatch); mask = np.ones((2, 4), dtype=np.uint8); outputs = []
    def same_mtime_drift(path):
        before = path.stat(); payload = bytearray(path.read_bytes()); payload[-1] ^= 1; path.write_bytes(payload)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    def hook(private, _output):
        if failure == "source": Path(request.source_path).write_bytes(b"drift")
        if failure == "source_sha": same_mtime_drift(Path(request.source_path))
        if failure == "stage": private.write_bytes(b"drift")
        if failure == "stage_sha": same_mtime_drift(private)
        if failure == "seal": outputs.append(_output)
        if failure == "foreign": Path(request.final_path).write_bytes(b"foreign")
        if failure == "empty": _output.write_bytes(b"")
    _install_process(monkeypatch, mask, code=7 if failure == "nonzero" else 0, hook=hook, missing=failure == "missing")
    seal = (lambda _identity: (outputs[0].write_bytes(b"drift"), True)[1]) if failure == "seal" else (lambda _identity: True)
    terminal, _ = _direct(request, seal=seal); assert terminal.status is OperationTerminalStatus.FAILED and not terminal.payload.published
    if failure == "foreign": assert Path(request.final_path).read_bytes() == b"foreign"
    else: assert not Path(request.final_path).exists()


@pytest.mark.parametrize("phase", ("prelaunch", "postprocess"))
def test_mask_parent_rename_and_symlink_swap_is_zero_effect(
    tmp_path, monkeypatch, phase,
) -> None:
    source_dir = tmp_path / "source-dir"
    source_dir.mkdir()
    request = _request(source_dir, monkeypatch)
    moved = tmp_path / "moved-dir"

    def swap(*_args):
        source_dir.rename(moved)
        source_dir.symlink_to(moved, target_is_directory=True)

    if phase == "prelaunch":
        swap()
        calls = _install_process(
            monkeypatch, np.ones((2, 4), dtype=np.uint8),
            hook=lambda *_args: pytest.fail("child launched"),
        )
    else:
        calls = _install_process(
            monkeypatch, np.ones((2, 4), dtype=np.uint8), hook=swap,
        )
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED
    assert not terminal.payload.published
    assert not Path(request.final_path).exists()
    assert len(calls) == (0 if phase == "prelaunch" else 1)
def test_link_time_foreign_final_wins_without_open_or_cleanup(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path, monkeypatch); _install_process(monkeypatch, np.ones((2, 4), dtype=np.uint8)); real_link = authoring._link
    def race(source, final, **options):
        if options.get("dst_dir_fd") is not None:
            Path(request.final_path).write_bytes(b"foreign")
            raise FileExistsError(final)
        return real_link(source, final, **options)
    monkeypatch.setattr(authoring, "_link", race); terminal, _ = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED and not terminal.payload.published
    assert Path(request.final_path).read_bytes() == b"foreign"


def test_executable_replacement_after_qualification_refuses_before_popen(
    tmp_path, monkeypatch,
) -> None:
    request = _request(tmp_path, monkeypatch)
    calls = _install_process(
        monkeypatch, np.ones((2, 4), dtype=np.uint8),
    )
    real_qualify = authoring._qualify_tiff
    replaced = []

    def replace_after_qualification(path, state):
        result = real_qualify(path, state)
        if Path(path) != Path(request.source_path) and not replaced:
            executable = Path(request.executable)
            executable.unlink()
            executable.write_text("replacement")
            executable.chmod(0o700)
            replaced.append(True)
        return result

    monkeypatch.setattr(authoring, "_qualify_tiff", replace_after_qualification)
    terminal, _progress = _direct(request)
    assert replaced == [True]
    assert terminal.status is OperationTerminalStatus.FAILED
    assert "immediately before launch" in terminal.diagnostic
    assert calls == []
    assert not Path(request.final_path).exists()


def test_link_time_parent_swap_cannot_redirect_publication_and_cleans_admitted_parent(
    tmp_path, monkeypatch,
) -> None:
    source_dir = tmp_path / "source-dir"
    source_dir.mkdir()
    request = _request(source_dir, monkeypatch)
    _install_process(monkeypatch, np.ones((2, 4), dtype=np.uint8))
    real_link = authoring._link
    admitted = tmp_path / "admitted-parent"
    replacement = tmp_path / "replacement-parent"
    swapped = []

    def swap_at_publication(source, final, **options):
        if options.get("dst_dir_fd") is not None and not swapped:
            source_dir.rename(admitted)
            replacement.mkdir()
            source_dir.symlink_to(replacement, target_is_directory=True)
            swapped.append(True)
        return real_link(source, final, **options)

    monkeypatch.setattr(authoring, "_link", swap_at_publication)
    terminal, _progress = _direct(request)
    assert swapped == [True]
    assert terminal.status is OperationTerminalStatus.FAILED
    assert not terminal.payload.published
    assert terminal.payload.recovery_path == ""
    assert not (replacement / Path(request.final_path).name).exists()
    assert not (admitted / Path(request.final_path).name).exists()
    assert not tuple(admitted.glob(".xdart-mask-*"))
def test_stage_mode_failure_keeps_exact_cleanup_custody(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path, monkeypatch); real_chmod = authoring.os.chmod
    def fail_stage(path, mode):
        if Path(path).name.startswith(".xdart-mask-"): raise OSError("mode refusal")
        return real_chmod(path, mode)
    monkeypatch.setattr(authoring.os, "chmod", fail_stage); terminal, _ = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED and not Path(terminal.payload.cwd).exists()
def test_stage_first_lstat_failure_reports_recovery_custody(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path, monkeypatch); real_lstat, failed = Path.lstat, []
    def fail_once(path):
        if path.name.startswith(".xdart-mask-") and not failed: failed.append(path); raise OSError("lstat refusal")
        return real_lstat(path)
    monkeypatch.setattr(Path, "lstat", fail_once); terminal, _ = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED and terminal.payload.recovery_path == terminal.payload.cwd
    assert failed and Path(terminal.payload.recovery_path).is_dir()
def test_cancel_seal_and_platform_child_signals_are_exact(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path, monkeypatch); _install_process(monkeypatch, np.ones((2, 4), dtype=np.uint8))
    terminal, _ = _direct(request, seal=lambda _identity: False)
    assert terminal.status is OperationTerminalStatus.CANCELLED and not Path(request.final_path).exists()
    late = Event(); terminal, _ = _direct(request, seal=lambda _identity: (late.set(), True)[1], cancelled=late)
    assert terminal.status is OperationTerminalStatus.RETURNED and terminal.payload.published
    calls = []; process = SimpleNamespace(pid=44, terminate=lambda: calls.append("terminate"), kill=lambda: calls.append("kill"))
    monkeypatch.setattr(authoring, "_WINDOWS", False); monkeypatch.setattr(authoring, "_killpg", lambda pid, sig: calls.append((pid, sig)))
    assert authoring._signal_child(process, kill=False) == "" and calls[-1][0] == 44
    monkeypatch.setattr(authoring, "_WINDOWS", True); assert authoring._signal_child(process, kill=True) == "" and calls[-1] == "kill"

def _published(tmp_path: Path, serial=1):
    tmp_path.mkdir(exist_ok=True)
    source = _tiff(
        tmp_path / "source.tif", np.ones((1, 1), dtype=np.uint16),
    )
    final = tmp_path / "source-mask.edf"
    EdfImage(data=np.ones((1, 1), dtype=np.uint8)).write(str(final))
    final.chmod(0o600)
    executable = tmp_path / "pyFAI-drawmask"
    executable.write_bytes(b"x")
    request = MaskRequest(
        str(source), str(final), str(executable),
        SourceFileState.capture(executable), SourceFileState.capture(source),
        str(tmp_path), authoring._directory_identity(tmp_path),
    )
    state = SourceFileState.capture(final)
    source_sha = authoring._asset_digest(source, authoring._TIFF_LIMIT)[0]
    mask_sha = authoring._asset_digest(final, authoring._MASK_LIMIT)[0]
    proof = MaskProof(
        state, (1, 1), np.dtype("u2").str, np.dtype("u1").str,
        source_sha, source_sha, mask_sha,
        "zero-false-real-nonzero-true-nan-true-v1",
    )
    return OperationIdentity(serial), MaskResult(
        request, str(final), proof, state, 0,
        (str(executable), str(source)), str(tmp_path), True,
    )


def test_mask_proof_nested_facts_and_terminal_payload_matrix_are_strict(
    tmp_path,
) -> None:
    identity, result = _published(tmp_path)
    proof = result.proof
    assert proof is not None
    invalid = (
        _forge(proof, state=_forge(proof.state, path="relative.edf")),
        _forge(proof, shape=(0, 1)),
        _forge(proof, source_dtype="<c16"),
        _forge(proof, mask_dtype="|O"),
        _forge(proof, source_sha256="0" * 63),
        _forge(proof, staged_sha256="1" * 64),
        _forge(proof, mask_sha256="g" * 64),
        _forge(proof, coercion_policy="inverse-v1"),
    )
    for forged in invalid:
        with pytest.raises(ValueError, match="mask proof"):
            forged.__post_init__()
        forged_result = _forge(result, proof=forged)
        returned = OperationTerminal(
            identity, OperationTerminalStatus.RETURNED,
            payload=forged_result,
        )
        assert not authoring.mask_terminal_result_valid(
            returned, result.request,
        )

    good = OperationTerminal(
        identity, OperationTerminalStatus.RETURNED, payload=result,
    )
    assert authoring.mask_terminal_result_valid(good, result.request)
    for status in OperationTerminalStatus:
        diagnostic = "boom" if status is OperationTerminalStatus.FAILED else ""
        empty = OperationTerminal(identity, status, diagnostic)
        assert not authoring.mask_terminal_result_valid(empty, result.request)
        payload = (_forge(result, diagnostic=diagnostic)
                   if diagnostic else result)
        terminal = OperationTerminal(
            identity, status, diagnostic, payload,
        )
        assert authoring.mask_terminal_result_valid(
            terminal, result.request,
        ) is (status is OperationTerminalStatus.RETURNED)

    for changes in (
        {"exit_code": 7},
        {"argv": ()},
        {"cwd": str(tmp_path / "foreign")},
        {"diagnostic": "unpaired"},
        {"published": False},
        {"recovery_path": str(tmp_path), "recovery_class": ""},
    ):
        forged = _forge(result, **changes)
        assert not authoring.mask_terminal_result_valid(
            OperationTerminal(
                identity, OperationTerminalStatus.RETURNED, payload=forged,
            ),
            result.request,
        )
    for state_changes in (
        {"device": proof.state.device + 1},
        {"inode": proof.state.inode + 1},
        {"size": proof.state.size + 1},
        {"mtime_ns": proof.state.mtime_ns + 1},
        {"ctime_ns": result.final_state.ctime_ns + 1},
        {"path": str(tmp_path / "foreign-private.edf")},
    ):
        changed_state = _forge(proof.state, **state_changes)
        forged_proof = _forge(proof, state=changed_state)
        forged = _forge(result, proof=forged_proof)
        assert not authoring.mask_terminal_result_valid(
            OperationTerminal(
                identity, OperationTerminalStatus.RETURNED, payload=forged,
            ),
            result.request,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("shape", (1, 2)),
        ("source_dtype", np.dtype("u1").str),
        ("mask_dtype", np.dtype("u2").str),
        ("source_sha256", "0" * 64),
        ("mask_sha256", "0" * 64),
    ),
)
def test_forged_current_generated_mask_proof_never_adopts(
    tmp_path, qapp, field, value,
) -> None:
    store = RunIntentStore(RunIntent())
    page = ScatteringWorkspace(
        intents=store, lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        identity, result = _published(tmp_path, 50)
        proof = result.proof
        changes = {field: value}
        if field == "source_sha256":
            changes["staged_sha256"] = value
        forged_proof = _forge(proof, **changes)
        forged = _forge(result, proof=forged_proof)
        _queue_published_mask(page, store, identity, forged)
        owner = page._authored_asset_owner
        assert owner is not None
        page._show_queued_authored_asset_confirmation()
        owner.dialog.accept_button.click()
        update = _finish_mask_validation(page)
        assert update.terminal.status is OperationTerminalStatus.FAILED
        assert store.snapshot().thaw().mask_file == ""
        assert Path(result.request.final_path).exists()
    finally:
        _close(page, qapp)


def _queue_published_mask(page, store, identity, result, *, stale=False):
    stamp = page._operation_context_stamp(store.revision)
    page._mask_identity = identity
    page._mask_revision = store.revision
    page._mask_stamp = stamp
    page._mask_request = result.request
    update = OperationUpdate(
        identity,
        terminal=OperationTerminal(
            identity, OperationTerminalStatus.RETURNED, payload=result,
        ),
        stale=stale,
    )
    assert page._consume_mask_update(update)


def _finish_mask_validation(page):
    identity = page._asset_validation_identity
    assert type(identity) is OperationIdentity
    worker = page._operation_slot._worker
    assert worker is not None
    worker.join(3)
    assert not worker.is_alive()
    page._operation_slot.observe_stamp(page._operation_context_stamp())
    update = page._operation_slot.poll(identity)
    assert type(update) is OperationUpdate
    assert page._consume_asset_validation_update(update)
    return update


def test_timer_poll_dispatches_exact_mask_update(tmp_path, monkeypatch, qapp) -> None:
    page = ScatteringWorkspace(intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter())
    identity, result = _published(tmp_path); update = OperationUpdate(identity, terminal=OperationTerminal(identity, OperationTerminalStatus.RETURNED, payload=result))
    observed, polled, consumed = [], [], []; page._mask_identity = identity
    monkeypatch.setattr(OperationSlot, "current_identity", property(lambda _slot: identity))
    monkeypatch.setattr(page._operation_slot, "observe_stamp", lambda stamp: observed.append(stamp))
    monkeypatch.setattr(page._operation_slot, "poll", lambda item: polled.append(item) or update)
    monkeypatch.setattr(page, "_consume_calibration_update", lambda _update: False)
    monkeypatch.setattr(page, "_consume_mask_update", lambda item: consumed.append(item) or True)
    try:
        page._drain_executor()
        assert len(observed) == 1 and polled == [identity] and consumed == [update]
    finally: _close(page, qapp)


def test_mask_confirmation_accept_cancel_and_choose_alternate_are_transactional(
    tmp_path, monkeypatch, qapp,
) -> None:
    alternate = tmp_path / "alternate.edf"
    EdfImage(data=np.zeros((1, 1), dtype=np.uint8)).write(str(alternate))
    chooser_calls = []

    def chooser(control, current, start):
        chooser_calls.append((control, current, start))
        return str(alternate)

    store = RunIntentStore(RunIntent())
    page = ScatteringWorkspace(
        intents=store, lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(), control_path_chooser=chooser,
    )
    threads = []
    real_validate = external_operation.validate_authored_asset

    def counted(request):
        threads.append(current_thread().name)
        return real_validate(request)

    monkeypatch.setattr(external_operation, "validate_authored_asset", counted)
    try:
        identity, result = _published(tmp_path / "accept", 1)
        _queue_published_mask(page, store, identity, result)
        owner = page._authored_asset_owner
        assert owner is not None and owner.asset == "mask" and owner.queued
        assert owner.dialog.full_path.text() == result.request.final_path
        assert store.snapshot().thaw().mask_file == ""
        page._show_queued_authored_asset_confirmation()
        owner.dialog.accept_button.click()
        _finish_mask_validation(page)
        assert threads[-1].startswith("scattering-operation-")
        assert store.snapshot().thaw().mask_file == result.request.final_path

        identity2, result2 = _published(tmp_path / "cancel", 2)
        _queue_published_mask(page, store, identity2, result2)
        owner = page._authored_asset_owner
        assert owner is not None
        page._show_queued_authored_asset_confirmation()
        owner.dialog.cancel_button.click()
        qapp.processEvents()
        assert store.snapshot().thaw().mask_file == result.request.final_path
        assert Path(result2.request.final_path).exists()

        identity3, result3 = _published(tmp_path / "alternate", 3)
        _queue_published_mask(page, store, identity3, result3)
        owner = page._authored_asset_owner
        assert owner is not None
        page._show_queued_authored_asset_confirmation()
        owner.dialog.choose_button.click()
        _finish_mask_validation(page)
        assert chooser_calls == [(
            MASK_FILE, result.request.final_path,
            str(Path(result3.request.source_path).parent),
        )]
        assert store.snapshot().thaw().mask_file == str(alternate)
        assert Path(result3.request.final_path).exists()
        assert threads[-1].startswith("scattering-operation-")
    finally:
        _close(page, qapp)


@pytest.mark.parametrize("moment", ("popup", "accept"))
@pytest.mark.parametrize("drift", ("source", "parent"))
def test_mask_source_custody_survives_through_popup_and_accept(
    tmp_path, monkeypatch, qapp, moment, drift,
) -> None:
    source_dir = tmp_path / "source-dir"
    identity, result = _published(source_dir, 30)
    store = RunIntentStore(RunIntent())
    page = ScatteringWorkspace(
        intents=store, lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        _queue_published_mask(page, store, identity, result)
        owner = page._authored_asset_owner
        assert owner is not None
        if moment == "accept":
            page._show_queued_authored_asset_confirmation()

        source = Path(result.request.source_path)
        if drift == "source":
            source.unlink()
            _tiff(source, np.zeros((1, 1), dtype=np.uint16))
        else:
            moved = tmp_path / "moved-source-dir"
            source_dir.rename(moved)
            source_dir.symlink_to(moved, target_is_directory=True)

        if moment == "popup":
            page._show_queued_authored_asset_confirmation()
        else:
            owner.dialog.accept_button.click()
        assert page._authored_asset_owner is None
        assert page._asset_validation_identity is None
        assert store.snapshot().thaw().mask_file == ""
    finally:
        _close(page, qapp)


def test_real_published_mask_proof_survives_cleanup_and_is_adoptable(
    tmp_path, monkeypatch, qapp,
) -> None:
    request = _request(tmp_path, monkeypatch)
    _install_process(monkeypatch, np.ones((2, 4), dtype=np.uint8))
    terminal, _progress = _direct(request)
    result = terminal.payload
    assert result.final_state == SourceFileState.capture(Path(request.final_path))
    store = RunIntentStore(RunIntent())
    page = ScatteringWorkspace(
        intents=store, lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        _queue_published_mask(page, store, terminal.identity, result)
        owner = page._authored_asset_owner
        assert owner is not None
        page._show_queued_authored_asset_confirmation()
        owner.dialog.accept_button.click()
        update = _finish_mask_validation(page)
        assert update.terminal.status is OperationTerminalStatus.RETURNED
        assert store.snapshot().thaw().mask_file == request.final_path
    finally:
        _close(page, qapp)


def test_busy_single_mask_popup_suppresses_escape_and_close_until_exact_cleanup(
    tmp_path, monkeypatch, qapp,
) -> None:
    entered, release = Event(), Event()
    real_validate = external_operation.validate_authored_asset

    def held(request):
        entered.set()
        assert release.wait(3)
        return real_validate(request)

    monkeypatch.setattr(external_operation, "validate_authored_asset", held)
    store = RunIntentStore(RunIntent())
    page = ScatteringWorkspace(
        intents=store, lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    destroyed = []
    try:
        page.show()
        qapp.processEvents()
        identity, result = _published(tmp_path, 81)
        _queue_published_mask(page, store, identity, result)
        owner = page._authored_asset_owner
        assert owner is not None and len(owner.candidates) == 1
        dialog = owner.dialog
        dialog.destroyed.connect(lambda *_args: destroyed.append(True))
        page._show_queued_authored_asset_confirmation()
        qapp.processEvents()
        assert dialog.isVisible()
        assert (dialog.windowModality()
                is QtCore.Qt.WindowModality.WindowModal)
        assert dialog.parent() is page
        assert dialog.accept_button.hasFocus()
        dialog.accept_button.click()
        assert entered.wait(2) and dialog._busy
        assert not any((dialog.accept_button.isEnabled(),
                        dialog.choose_button.isEnabled(),
                        dialog.cancel_button.isEnabled()))

        QtTest.QTest.keyClick(dialog, QtCore.Qt.Key.Key_Escape)
        qapp.processEvents()
        assert page._authored_asset_owner is owner and dialog.isVisible()
        dialog.close()
        qapp.processEvents()
        assert page._authored_asset_owner is owner and dialog.isVisible()
        assert store.snapshot().thaw().mask_file == ""
        assert Path(result.request.final_path).exists()

        snapshot = store.snapshot()
        changed = snapshot.thaw()
        changed.project_root = str(tmp_path / "context-drift")
        store.commit(changed, expected_revision=snapshot.revision)
        release.set()
        update = _finish_mask_validation(page)
        assert update.stale
        assert page._authored_asset_owner is None
        assert page._operation_slot.owned is False
        assert page._experiment_operation_busy() is False
        assert page._consume_asset_validation_update(update) is False
        assert store.snapshot().thaw().mask_file == ""
        assert Path(result.request.final_path).exists()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete,
        )
        qapp.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete,
        )
        assert destroyed == [True]
    finally:
        release.set()
        _close(page, qapp)


def test_busy_delete_later_autonomously_closes_after_worker_join(
    tmp_path, monkeypatch, qapp,
) -> None:
    entered, release = Event(), Event()
    real_validate = external_operation.validate_authored_asset

    def held(request):
        result = real_validate(request)
        entered.set()
        assert release.wait(3)
        return result

    monkeypatch.setattr(external_operation, "validate_authored_asset", held)
    store = RunIntentStore(RunIntent())
    page = ScatteringWorkspace(
        intents=store, lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    slot = page._operation_slot
    page_destroyed = []
    page.destroyed.connect(lambda *_args: page_destroyed.append(True))
    try:
        identity, result = _published(tmp_path, 91)
        _queue_published_mask(page, store, identity, result)
        owner = page._authored_asset_owner
        assert owner is not None
        dialog = owner.dialog
        page._show_queued_authored_asset_confirmation()
        dialog.accept_button.click()
        assert entered.wait(2)
        validation_identity = owner.validation_identity
        worker = page._operation_slot._worker
        assert validation_identity is not None and worker is not None
        page.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete,
        )
        qapp.processEvents()
        assert page_destroyed == []
        assert page._closing and not page._closed
        assert page._terminal_close is None
        assert page._deferred_delete_pending
        assert page._deferred_delete_retry_timer.isActive()
        assert slot._clean_receipt is None
        assert slot._close_cancel_accepted
        assert slot._cancel_event is not None
        assert slot._cancel_event.is_set()
        assert page._authored_asset_owner is None
        assert slot.owned
        assert slot.current_identity is validation_identity
        assert worker.is_alive()
        assert store.snapshot().thaw().mask_file == ""
        assert Path(result.request.final_path).exists()

        release.set()
        for _attempt in range(300):
            worker.join(0.01)
            QtTest.QTest.qWait(10)
            qapp.processEvents()
            QtCore.QCoreApplication.sendPostedEvents(
                None, QtCore.QEvent.Type.DeferredDelete,
            )
            qapp.processEvents()
            if page_destroyed:
                break
        assert page_destroyed == [True], (
            page._closed,
            page._deferred_delete_pending,
            page._deferred_delete_reposted,
            page._deferred_delete_retry_timer.isActive(),
            slot.owned,
            slot._clean_receipt,
        )
        assert not worker.is_alive()
        operation_receipt = slot._clean_receipt
        assert operation_receipt is not None
        assert operation_receipt.cleanup_status is CleanupStatus.CLEANED
        assert operation_receipt.identity is validation_identity
        assert operation_receipt.terminal is not None
        assert (operation_receipt.terminal.status
                is OperationTerminalStatus.CANCELLED)
        assert slot.owned is False
        assert slot._worker is None
        assert store.snapshot().thaw().mask_file == ""
        assert Path(result.request.final_path).exists()
    finally:
        release.set()
        if not page_destroyed:
            page.close_workspace()
            page.deleteLater()
            qapp.processEvents()


def test_failed_published_mask_terminal_preserves_output_and_recovery_without_adoption(
    tmp_path, monkeypatch, qapp,
) -> None:
    request = _request(tmp_path, monkeypatch)
    _install_process(monkeypatch, np.ones((2, 4), dtype=np.uint8))
    monkeypatch.setattr(
        authoring, "_cleanup", lambda stage, *_args: str(stage),
    )
    terminal, _progress = _direct(request)
    result = terminal.payload
    assert terminal.status is OperationTerminalStatus.FAILED
    assert result.published and result.recovery_path
    assert result.recovery_class == "qualified-private-candidate"
    assert Path(request.final_path).exists()
    assert Path(result.recovery_path).exists()
    assert authoring.mask_terminal_result_valid(terminal, request)

    prior = str(tmp_path / "prior-mask.edf")
    store = RunIntentStore(RunIntent(mask_file=prior))
    page = ScatteringWorkspace(
        intents=store, lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        stamp = page._operation_context_stamp(store.revision)
        page._mask_identity = terminal.identity
        page._mask_revision = store.revision
        page._mask_stamp = stamp
        page._mask_request = request
        assert page._consume_mask_update(OperationUpdate(
            terminal.identity, terminal=terminal,
        ))
        assert page._authored_asset_owner is None
        assert store.revision == 0
        assert store.snapshot().thaw().mask_file == prior
        assert Path(request.final_path).exists()
        assert Path(result.recovery_path).exists()
        assert "failed" in page._notice_text.lower()
    finally:
        _close(page, qapp)


def test_mask_stale_and_context_drift_preserve_field_and_published_file(
    tmp_path, monkeypatch, qapp,
) -> None:
    prior = str(tmp_path / "prior.edf")
    store = RunIntentStore(RunIntent(mask_file=prior))
    page = ScatteringWorkspace(
        intents=store, lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        identity, result = _published(tmp_path / "stale", 4)
        _queue_published_mask(page, store, identity, result, stale=True)
        assert page._authored_asset_owner is None
        assert store.snapshot().thaw().mask_file == prior
        assert Path(result.request.final_path).exists()

        identity2, result2 = _published(tmp_path / "drift", 5)
        _queue_published_mask(page, store, identity2, result2)
        owner = page._authored_asset_owner
        assert owner is not None
        snapshot = store.snapshot()
        changed = snapshot.thaw()
        changed.project_root = str(tmp_path / "other")
        store.commit(changed, expected_revision=snapshot.revision)
        page._show_queued_authored_asset_confirmation()
        assert page._authored_asset_owner is None
        assert store.snapshot().thaw().mask_file == prior
        assert Path(result2.request.final_path).exists()
    finally:
        _close(page, qapp)
