"""Focused standalone-Calibrate operation oracle."""
from __future__ import annotations
import ast
import hashlib
import os
from pathlib import Path
import signal
import stat
import subprocess
from threading import Event
import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pyqtgraph.Qt import QtWidgets
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering import experiment_authoring as authoring
from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.experiment_authoring import (
    CalibrationRequest,
    CalibrationResult,
    prepare_calibration_request,
    run_calibration,
)
from xdart.gui.tabs.scattering.operation_values import (
    OperationContextStamp,
    OperationIdentity,
    OperationTerminalStatus,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.readiness import ControlAction, SectionId
from xrd_tools.session.run_configuration import RunIntent
def test_p3_1b_idle_calibrate_is_mounted_and_enabled(
    tmp_path, monkeypatch
) -> None:
    binary = tmp_path / "bin" / "pyFAI-calib2"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o700)
    monkeypatch.setenv(
        "PATH", f"{binary.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    )
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        state = page._project_controls(page._intents.snapshot())
        calibrate = next(
            action
            for action in state.profile.actions_for(SectionId.EXPERIMENT)
            if action.action is ControlAction.CALIBRATE
        )
        assert calibrate.enabled
        assert "standalone" in calibrate.reason.lower()
        assert "poni" in calibrate.reason.lower()
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()
_PONI = """poni_version: 2.1
Detector: Pilatus300kw
Detector_config: {"orientation":3}
Distance: 0.1234
Poni1: 0.05
Poni2: 0.06
Rot1: 0.01
Rot2: 0.02
Rot3: 0.03
Wavelength: 1e-10
"""
def _request(tmp_path: Path, name: str = "made.poni") -> CalibrationRequest:
    executable = tmp_path / "bin" / "pyFAI-calib2"
    executable.parent.mkdir(exist_ok=True)
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o700)
    return CalibrationRequest(str(tmp_path / name), str(executable.resolve()))
def _install_process(monkeypatch, payload=_PONI.encode(), code=0, hook=None):
    calls = []
    class Process:
        pid = 4242
        def __init__(self, argv, **options):
            private = Path(argv[2]); calls.append((tuple(argv), options,
                stat.S_IMODE(Path(options["cwd"]).stat().st_mode), private.exists()))
            if hook is not None: hook(private)
            elif payload is not None: private.write_bytes(payload)
        def wait(self, *, timeout): return code
        def terminate(self): calls.append("terminate")
        def kill(self): calls.append("kill")
    monkeypatch.setattr(authoring, "_popen", Process)
    return calls
def _direct(request, seal=lambda _identity: True):
    progress = []
    terminal = run_calibration(
        request, OperationIdentity(1), Event(),
        lambda *value: progress.append(value), seal,
    )
    assert type(terminal.payload) is CalibrationResult
    return terminal, progress
def _assert_launch(call, request, *, windows=False):
    argv, options, stage_mode, existed = call
    assert argv[0] == request.executable == str(Path(argv[0]).resolve())
    assert argv[1:] == ("--poni", str(Path(options["cwd"]) / Path(request.final_path).name))
    assert Path(options["cwd"]).is_absolute() and options["shell"] is False
    assert options["stdin"] is options["stdout"] is options["stderr"] is subprocess.DEVNULL
    assert options["close_fds"] is True and stage_mode == 0o700 and existed is False
    if windows:
        assert options["creationflags"] == 0x200
        assert "start_new_session" not in options
    else:
        assert options["start_new_session"] is True
        assert "creationflags" not in options
def test_request_preflight_is_unseeded_create_new_and_path_resolved(tmp_path, monkeypatch) -> None:
    binary = tmp_path / "bin" / "pyFAI-calib2"; binary.parent.mkdir(); binary.write_text("x"); binary.chmod(0o700); monkeypatch.setenv("PATH", str(binary.parent))
    made = prepare_calibration_request(str(tmp_path / "calibration")); current = tmp_path / "current.poni"
    assert (made.final_path, made.executable) == (str(tmp_path / "calibration.poni"), str(binary.resolve()))
    with pytest.raises(ValueError, match="differ"): prepare_calibration_request(str(current), current_poni=str(current))
    alias = tmp_path / "alias"; alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="differ"): prepare_calibration_request(str(tmp_path / "future.poni"), current_poni=str(alias / "future.poni"))
    regular, link, broken = (tmp_path / name for name in ("regular.poni", "link.poni", "broken.poni"))
    regular.write_text("occupied"); link.symlink_to(regular); broken.symlink_to(tmp_path / "missing.poni")
    for target in (regular, link, broken):
        with pytest.raises(ValueError, match="exists"): prepare_calibration_request(str(target))
    with pytest.raises(ValueError, match="parent"): prepare_calibration_request(str(tmp_path / "missing" / "x.poni"))
    with pytest.raises(ValueError, match="suffix"): prepare_calibration_request(str(tmp_path / "x.txt"))
    slot, calls = OperationSlot(), []; monkeypatch.setattr(slot, "_begin", lambda *args: calls.append(args) or OperationIdentity(99))
    forged = CalibrationRequest(str(tmp_path / "forged.poni"), str(tmp_path / "not-pyfai")); results = [slot.begin_calibrate(forged, OperationContextStamp(0))]
    occupied = prepare_calibration_request(str(tmp_path / "occupied-late.poni")); Path(occupied.final_path).write_text("foreign")
    results.append(slot.begin_calibrate(occupied, OperationContextStamp(0)))
    dependency = prepare_calibration_request(str(tmp_path / "dependency-late.poni")); monkeypatch.setenv("PATH", ""); results.append(slot.begin_calibrate(dependency, OperationContextStamp(0)))
    with pytest.raises(ValueError, match="PATH"): prepare_calibration_request(str(tmp_path / "absent.poni"))
    assert results == [None, None, None] and calls == []
    assert slot._worker is None and not slot.owned and slot._next_serial == 1
@pytest.mark.parametrize("windows", (False, True))
def test_exact_private_launch_strict_qualification_and_link_publication(
    tmp_path, monkeypatch, windows
) -> None:
    request = _request(tmp_path); monkeypatch.setenv("PATH", str(Path(request.executable).parent))
    monkeypatch.setattr(authoring, "_WINDOWS", windows)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200,
                        raising=False)
    calls = _install_process(monkeypatch)
    links, real_link = [], os.link
    def counted(source, target):
        links.append((Path(source), Path(target), stat.S_IMODE(Path(source).stat().st_mode)))
        return real_link(source, target)
    monkeypatch.setattr(authoring, "_link", counted)
    slot = OperationSlot()
    identity = slot.begin_calibrate(request, OperationContextStamp(0))
    worker = slot._worker; worker.join(3)
    update = slot.poll(identity)
    assert update is not None and update.terminal.status is OperationTerminalStatus.RETURNED
    assert slot.poll(identity) is None and len(calls) == 1
    _assert_launch(calls[0], request, windows=windows)
    result = update.terminal.payload
    assert result.published and result.exit_code == 0 and result.recovery_path == ""
    assert result.proof.detector_config_json == '{"orientation":3}'
    assert result.proof.sha256 == hashlib.sha256(_PONI.encode()).hexdigest()
    assert int(stat.S_IMODE(Path(request.final_path).stat().st_mode)) == 0o600
    final_links = [item for item in links if item[1] == Path(request.final_path)]
    assert len(final_links) == 1 and final_links[0][2] == 0o600
    assert [(source.name, target.name) for source, target, _mode in links[:2]] == [
        (".link-probe-source", ".link-probe-linked"),
        (".link-probe-source", ".link-probe-occupied"),
    ]
    assert len(links) == 3
    assert not Path(result.cwd).exists() and Path(request.final_path).read_text() == _PONI
@pytest.mark.parametrize("payload,code", [
    (_PONI.encode(), 7), (None, 0), (b"", 0),
    (b"partial", 0),
    (b"x" * ((1 << 20) + 1), 0),
], ids=("nonzero", "missing", "empty", "partial", "oversize"))
def test_invalid_child_output_never_publishes(
    tmp_path, monkeypatch, payload, code
) -> None:
    request = _request(tmp_path)
    calls = _install_process(monkeypatch, payload=payload, code=code)
    if payload is not None and len(payload) > 1 << 20:
        valid = tmp_path / "valid.poni"; valid.write_text(_PONI)
        parsed = authoring.load_detector_calibration(valid)
        monkeypatch.setattr(authoring, "load_detector_calibration",
                            lambda *_a, **_k: parsed)
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED
    assert not Path(request.final_path).exists() and len(calls) == 1
    _assert_launch(calls[0], request)
    assert not Path(terminal.payload.cwd).exists()
def test_output_change_during_strict_load_refuses_publication(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path); calls = _install_process(monkeypatch)
    real = authoring.load_detector_calibration
    def changing(path):
        value = real(path); Path(path).write_text(_PONI.replace("Rot1: 0.01", "Rot1: 0.02")); return value
    monkeypatch.setattr(authoring, "load_detector_calibration", changing)
    terminal, _ = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED and len(calls) == 1
    assert not Path(request.final_path).exists() and not Path(terminal.payload.cwd).exists()
@pytest.mark.parametrize("race,expected_status,expected", [
    ("foreign", OperationTerminalStatus.FAILED, b"foreign"),
    ("linked_raise", OperationTerminalStatus.RETURNED, _PONI.encode()),
    ("post_link_foreign", OperationTerminalStatus.FAILED, b"foreign"),
])
def test_publication_races_reconcile_once_without_clobber(
    tmp_path, monkeypatch, race, expected_status, expected
) -> None:
    request = _request(tmp_path); _install_process(monkeypatch)
    real_link, attempts = os.link, []
    def racing(source, target):
        if Path(target) != Path(request.final_path): return real_link(source, target)
        attempts.append(1)
        if race == "foreign": Path(target).write_bytes(b"foreign")
        else:
            real_link(source, target)
            if race == "post_link_foreign": Path(target).unlink(); Path(target).write_bytes(b"foreign")
        if race in {"foreign", "linked_raise"}: raise OSError("injected link result")
    monkeypatch.setattr(authoring, "_link", racing)
    terminal, _ = _direct(request)
    assert terminal.status is expected_status and attempts == [1]
    assert Path(request.final_path).read_bytes() == expected
    assert terminal.payload.published is (race == "linked_raise")
def test_preseal_refusal_cancels_without_final_link(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path); calls = _install_process(monkeypatch)
    final_links, real_link = [], os.link
    def counted(source, target):
        if Path(target) == Path(request.final_path): final_links.append(1)
        return real_link(source, target)
    monkeypatch.setattr(authoring, "_link", counted)
    terminal, _ = _direct(request, seal=lambda _identity: False)
    assert terminal.status is OperationTerminalStatus.CANCELLED
    assert final_links == [] and len(calls) == 1 and not Path(request.final_path).exists()
@pytest.mark.parametrize("swap", ("regular", "symlink"))
def test_postqualification_tightening_is_fd_bound_and_never_chmods_a_swap(
    tmp_path, monkeypatch, swap
) -> None:
    request = _request(tmp_path); _install_process(monkeypatch)
    outside = tmp_path / "outside.poni"
    outside.write_text(_PONI); outside.chmod(0o640)
    real_open, real_chmod, swapped = os.open, os.chmod, Event()
    def replace_private(path):
        candidate = Path(path)
        if swapped.is_set() or candidate.name != "made.poni" or not candidate.parent.name.startswith(".xdart-calibrate-"):
            return
        candidate.unlink()
        (candidate.symlink_to(outside) if swap == "symlink"
         else candidate.write_bytes(outside.read_bytes()))
        swapped.set()
    def opening(path, flags, *args, **kwargs):
        replace_private(path); return real_open(path, flags, *args, **kwargs)
    def chmod(path, mode, *args, **kwargs):
        replace_private(path); return real_chmod(path, mode, *args, **kwargs)
    monkeypatch.setattr(authoring.os, "open", opening)
    monkeypatch.setattr(authoring.os, "chmod", chmod)
    terminal, _ = _direct(request)
    assert swapped.is_set() and terminal.status is OperationTerminalStatus.FAILED
    assert not Path(request.final_path).exists()
    assert stat.S_IMODE(outside.stat().st_mode) == 0o640
def test_cleanup_failure_reports_reproved_recovery_class(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path); _install_process(monkeypatch)
    monkeypatch.setattr(authoring, "_cleanup", lambda *_args: "untrusted-spelling")
    terminal, _ = _direct(request); result = terminal.payload
    private, stage = Path(result.private_path), Path(result.cwd)
    try:
        assert terminal.status is OperationTerminalStatus.FAILED
        assert result.recovery_class == "qualified-private-candidate"
        assert result.recovery_path == str(private.resolve(strict=True))
        assert result.proof is not None and authoring._matches(result.proof, private)
    finally:
        Path(request.final_path).unlink(missing_ok=True)
        private.unlink(missing_ok=True); stage.rmdir()
def test_signal_and_wait_exceptions_do_not_relinquish_direct_child(
    tmp_path, monkeypatch
) -> None:
    request = _request(tmp_path); monkeypatch.setenv("PATH", str(Path(request.executable).parent)); entered, wait_error, killed, release = (Event() for _ in range(4))
    signals = []
    class Process:
        pid = 8181
        def __init__(self, argv, **_options):
            self.waits = 0; Path(argv[2]).write_text(_PONI); entered.set()
        def wait(self, *, timeout):
            self.waits += 1
            if self.waits == 1: wait_error.set(); raise OSError("injected wait error")
            if release.wait(timeout): return -9
            raise subprocess.TimeoutExpired("fake", timeout)
        def terminate(self): raise AssertionError("POSIX uses the process group")
        def kill(self): raise AssertionError("POSIX uses the process group")
    def killpg(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGTERM: raise OSError("injected signal error")
        killed.set()
    monkeypatch.setattr(authoring, "_popen", Process)
    monkeypatch.setattr(authoring, "_WINDOWS", False)
    monkeypatch.setattr(authoring, "_killpg", killpg)
    monkeypatch.setattr(authoring, "_CANCEL_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(authoring, "_monotonic", lambda: 0.0)
    slot = OperationSlot(); identity = slot.begin_calibrate(request, OperationContextStamp(0))
    assert entered.wait(2) and wait_error.wait(2)
    pending = slot.close()
    assert pending.cleanup_status.value == "cleanup_pending" and killed.wait(2)
    release.set(); slot._worker.join(2); receipt = slot.close()
    assert signals == [(8181, signal.SIGTERM), (8181, signal.SIGKILL)]
    assert receipt.cleanup_status.value == "cleaned"
    assert receipt.terminal.status is OperationTerminalStatus.FAILED
    assert receipt.terminal.payload.exit_code == -9 and "OSError" in receipt.terminal.diagnostic
@pytest.mark.parametrize("windows", (False, True))
def test_cancel_signals_only_the_platform_owned_process(tmp_path, monkeypatch, windows) -> None:
    signals, direct = [], []
    class Held:
        pid = 8181
        def __init__(self): self.waits = 0
        def wait(self, *, timeout):
            self.waits += 1
            if self.waits < 3: raise subprocess.TimeoutExpired("fake", timeout)
            return -9
        def terminate(self): direct.append("terminate")
        def kill(self): direct.append("kill")
    monkeypatch.setattr(authoring, "_WINDOWS", windows)
    monkeypatch.setattr(authoring, "_killpg", lambda pid, sig: signals.append((pid, sig)))
    values = iter((0.0, 1.0)); monkeypatch.setattr(authoring, "_monotonic", lambda: next(values))
    event = Event(); event.set()
    assert authoring._wait_child(Held(), event) == (-9, "")
    assert (direct, signals) == ((["terminate", "kill"], []) if windows else
                                 ([], [(8181, signal.SIGTERM), (8181, signal.SIGKILL)]))
def test_authoring_has_no_peer_worker_queue_or_persistent_owner() -> None:
    tree = ast.parse(Path(authoring.__file__).read_text())
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not ({"Thread", "Queue", "Timer", "ThreadPoolExecutor", "Process"} & calls)
    assert authoring.__file__.endswith("experiment_authoring.py")
