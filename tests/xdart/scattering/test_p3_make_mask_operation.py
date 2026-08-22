"""Focused standalone-Make-Mask operation oracle."""
from __future__ import annotations
import os, stat, subprocess
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
import tifffile
from PIL import Image
from fabio.edfimage import EdfImage
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pyqtgraph.Qt import QtWidgets
from xdart.gui.tabs.scattering import experiment_authoring as authoring
from xdart.gui.tabs.scattering import page as page_module
from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import SourceFileState
from xdart.gui.tabs.scattering.controls_inventory import MASK_FILE
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.experiment_authoring import (
    MaskProof, MaskRequest, MaskResult, prepare_mask_request, run_mask,
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
    binary.write_text("fixture"); binary.chmod(0o700); monkeypatch.setenv("PATH", str(binary.parent))
    return binary.resolve()
def _tiff(path: Path, data, **options) -> Path:
    tifffile.imwrite(path, np.asarray(data), **options); return path
def _request(tmp_path: Path, monkeypatch, data=None) -> MaskRequest:
    _binary(tmp_path, monkeypatch); source = tmp_path / "chosen.tiff"
    _tiff(source, np.arange(8, dtype=np.uint16).reshape(2, 4) if data is None else data)
    return prepare_mask_request(str(source))
def _mask_path(private: Path) -> Path:
    return private.with_name(os.path.splitext(private.name)[0] + "-mask.edf")
def _install_process(monkeypatch, mask=None, *, code=0, hook=None, missing=False):
    calls = []
    class Process:
        pid = 7373
        def __init__(self, argv, **options):
            private, output = Path(argv[1]), _mask_path(Path(argv[1])); calls.append((tuple(argv), options, stat.S_IMODE(private.stat().st_mode)))
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
def _direct(request: MaskRequest, *, seal=lambda _identity: True, cancelled=None):
    progress = []; cancelled = Event() if cancelled is None else cancelled
    terminal = run_mask(request, OperationIdentity(1), cancelled, lambda *value: progress.append(value), seal)
    assert type(terminal.payload) is MaskResult; return terminal, progress
def _close(page, qapp):
    page.close_workspace(); page.deleteLater(); qapp.processEvents()

def test_parent_red_make_mask_command_uses_explicit_tiff_chooser(
    tmp_path, monkeypatch
) -> None:
    binary = tmp_path / "bin" / "pyFAI-drawmask"; binary.parent.mkdir()
    binary.write_text("fixture"); binary.chmod(0o700); monkeypatch.setenv("PATH", str(binary.parent))
    source = tmp_path / "chosen.tiff"; source.write_bytes(b"explicit TIFF"); chosen = []
    def chooser(path, _current, _start): chosen.append(path); return str(source)
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = RunIntentStore(RunIntent(project_root=str(tmp_path)))
    page = ScatteringWorkspace(intents=store, lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(), control_path_chooser=chooser)
    command = ShellCommand(ShellCommandKind.CONTROL_ACTION, "make_mask")
    try:
        monkeypatch.setattr(page._operation_slot, "begin_mask", lambda *_args: None); page._handle_shell_command(command)
        assert chosen == [MASK_FILE]
    finally: _close(page, qapp)

def test_page_focus_selected_assets_cancel_and_projection_are_exact(tmp_path, monkeypatch, qapp) -> None:
    _binary(tmp_path, monkeypatch); source = _tiff(tmp_path / "selected.tiff", np.ones((2, 4), dtype=np.uint16))
    poni, mask = tmp_path / "current.poni", tmp_path / "current-mask.edf"; chosen = []
    def chooser(path, current, start): chosen.append((path, current, start)); return str(source)
    store = RunIntentStore(RunIntent(project_root=str(tmp_path), poni_file=str(poni), mask_file=str(mask)))
    page = ScatteringWorkspace(intents=store, lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(), control_path_chooser=chooser)
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
        assert len(chosen) == 1 and chosen[0][:2] == (MASK_FILE, "") and Path(chosen[0][2]).is_dir()
        assert prepared == [(str(source), str(poni), str(mask))]
        assert len(begun) == 1 and begun[0][0].source_path == str(source) and begun[0][0].source_path != displayed.artifact
        assert type(begun[0][1]) is OperationContextStamp and page._mask_identity is identity
        direct = project_controls(store.snapshot(), None, RunPhase.IDLE, mask_available=True, mask_dependency_available=True)
        action = next(item for item in direct.profile.actions_for(SectionId.EXPERIMENT) if item.action is ControlAction.MAKE_MASK)
        assert action.enabled and "explicit tiff" in action.reason.lower()
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
    assert request == MaskRequest(str(source.resolve()), str(tmp_path / "source-mask.edf"), str(binary))
    alias.unlink(); alias.symlink_to(tmp_path / "missing.tif"); assert request.source_path == str(source.resolve())
    with pytest.raises(ValueError, match="unavailable"): prepare_mask_request(str(alias))
    wrong = tmp_path / "source.png"; wrong.write_bytes(b"x")
    with pytest.raises(ValueError, match="suffix"): prepare_mask_request(str(wrong))
    slot, calls = OperationSlot(), []
    monkeypatch.setattr(slot, "_begin", lambda *args: calls.append(args) or OperationIdentity(9)); monkeypatch.setenv("PATH", "")
    assert slot.begin_mask(request, OperationContextStamp(0)) == OperationIdentity(9) and len(calls) == 1
    forged = MaskRequest(request.source_path, request.final_path, str(tmp_path / "other")); (tmp_path / "other").write_text("x"); (tmp_path / "other").chmod(0o700)
    assert slot.begin_mask(forged, OperationContextStamp(0)) is None
    Path(request.final_path).write_text("foreign"); assert slot.begin_mask(request, OperationContextStamp(0)) is None and len(calls) == 1
    assert Path(request.final_path).read_text() == "foreign" and slot.owned is False
    Path(request.final_path).unlink(); binary.unlink(); assert slot.begin_mask(request, OperationContextStamp(0)) is None

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
    assert options["stdin"] is options["stdout"] is options["stderr"] is subprocess.DEVNULL
    assert ("creationflags" in options) is authoring._WINDOWS and ("start_new_session" in options) is not authoring._WINDOWS
    expected = mask != 0
    if mask.dtype.kind == "f": expected |= np.isnan(mask)
    assert np.array_equal(load_mask(request.final_path), expected)
    assert result.proof.coercion_policy == "zero-false-real-nonzero-true-nan-true-v1"
    assert result.proof.source_sha256 == result.proof.staged_sha256 and stat.S_IMODE(Path(request.final_path).stat().st_mode) == 0o600
    assert not Path(result.cwd).exists()

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
def test_link_time_foreign_final_wins_without_open_or_cleanup(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path, monkeypatch); _install_process(monkeypatch, np.ones((2, 4), dtype=np.uint8)); real_link = authoring._link
    def race(source, final):
        if Path(final) == Path(request.final_path): Path(final).write_bytes(b"foreign"); raise FileExistsError(final)
        return real_link(source, final)
    monkeypatch.setattr(authoring, "_link", race); terminal, _ = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED and not terminal.payload.published
    assert Path(request.final_path).read_bytes() == b"foreign"
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
    tmp_path.mkdir(exist_ok=True); source, final, executable = (tmp_path / name for name in ("source.tif", "source-mask.edf", "pyFAI-drawmask"))
    source.write_bytes(b"tiff"); final.write_bytes(b"edf"); executable.write_bytes(b"x")
    request = MaskRequest(str(source), str(final), str(executable)); state = SourceFileState.capture(final)
    proof = MaskProof(state, (1, 1), "<u2", "|u1", "a", "a", "b", "zero-false-real-nonzero-true-nan-true-v1")
    return OperationIdentity(serial), MaskResult(request, str(final), proof, state, 0, (str(executable), str(source)), str(tmp_path), True)
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
def test_page_exactly_one_mask_cas_and_stale_foreign_duplicate_are_inert(tmp_path, monkeypatch, qapp) -> None:
    store = RunIntentStore(RunIntent()); page = ScatteringWorkspace(intents=store, lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter())
    identity, result = _published(tmp_path); page._mask_identity, page._mask_revision = identity, store.revision
    real_reduce, calls = page_module.reduce_control_edit, []
    monkeypatch.setattr(page_module, "reduce_control_edit", lambda *args: calls.append(args) or real_reduce(*args))
    update = OperationUpdate(identity, terminal=OperationTerminal(identity, OperationTerminalStatus.RETURNED, payload=result))
    try:
        assert page._consume_calibration_update(update) is False and page._consume_mask_update(update) is True
        assert len(calls) == 1 and calls[0][1:] == (MASK_FILE, result.request.final_path)
        assert store.snapshot().thaw().mask_file == result.request.final_path and page._consume_mask_update(update) is False
        foreign = OperationIdentity(2); assert page._consume_mask_update(OperationUpdate(foreign, terminal=OperationTerminal(foreign, OperationTerminalStatus.RETURNED))) is False
        identity2, result2 = _published(tmp_path / "revision", 3); page._mask_identity, page._mask_revision = identity2, store.revision
        snapshot = store.snapshot(); changed = snapshot.thaw(); changed.poni_file = str(tmp_path / "new.poni"); store.commit(changed, expected_revision=snapshot.revision)
        revision = OperationUpdate(identity2, terminal=OperationTerminal(identity2, OperationTerminalStatus.RETURNED, payload=result2))
        assert page._consume_mask_update(revision) is True and len(calls) == 1 and store.snapshot().thaw().mask_file == result.request.final_path
        identity3, result3 = _published(tmp_path / "stale", 4); page._mask_identity, page._mask_revision = identity3, store.revision
        stale = OperationUpdate(identity3, terminal=OperationTerminal(identity3, OperationTerminalStatus.RETURNED, payload=result3), stale=True)
        assert page._consume_mask_update(stale) is True and len(calls) == 1 and Path(result3.request.final_path).exists()
        identity4, result4 = _published(tmp_path / "race", 5); page._mask_identity, page._mask_revision = identity4, store.revision; raced = str(tmp_path / "raced.edf")
        def racing(snapshot, path, value):
            candidate = real_reduce(snapshot, path, value); incumbent = store.snapshot().thaw(); incumbent.mask_file = raced
            store.commit(incumbent, expected_revision=snapshot.revision); calls.append((snapshot, path, value)); return candidate
        monkeypatch.setattr(page_module, "reduce_control_edit", racing)
        race = OperationUpdate(identity4, terminal=OperationTerminal(identity4, OperationTerminalStatus.RETURNED, payload=result4))
        assert page._consume_mask_update(race) is True and store.snapshot().thaw().mask_file == raced and len(calls) == 2
    finally: _close(page, qapp)
