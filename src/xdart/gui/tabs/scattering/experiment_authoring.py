"""Bounded standalone experiment-asset authoring operations."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, os, shutil, signal, stat, subprocess, tempfile, time
from pathlib import Path
from typing import Callable
from xrd_tools.integrate.calibration import load_detector_calibration
from .contracts import SourceFileState
from .events import detached_exception_strings
from .operation_values import OperationIdentity, OperationTerminal, OperationTerminalStatus
_LIMIT = 1 << 20
_POLL_SECONDS, _CANCEL_GRACE_SECONDS = 0.05, 0.25
_WINDOWS = os.name == "nt"
_popen, _link = subprocess.Popen, os.link
_killpg, _monotonic = getattr(os, "killpg", None), time.monotonic
@dataclass(frozen=True, slots=True)
class CalibrationRequest:
    final_path: str; executable: str
    def __post_init__(self) -> None:
        if (not all(type(v) is str and os.path.isabs(v) for v in (self.final_path, self.executable))
                or Path(self.final_path).suffix.casefold() != ".poni"):
            raise ValueError("calibration request is invalid")
@dataclass(frozen=True, slots=True)
class CalibrationFileProof:
    state: SourceFileState; sha256: str; detector_config_json: str
    geometry: tuple[float, ...]
@dataclass(frozen=True, slots=True)
class CalibrationResult:
    request: CalibrationRequest; private_path: str
    proof: CalibrationFileProof | None; final_state: SourceFileState | None
    exit_code: int | None; argv: tuple[str, ...]; cwd: str; published: bool
    recovery_path: str = ""; recovery_class: str = ""; diagnostic: str = ""
class _Cancelled(RuntimeError): pass
def resolve_calibration_executable() -> str | None:
    """Resolve the sole supported authoring executable through current PATH."""
    found = shutil.which("pyFAI-calib2")
    if not found: return None
    try: path, state = Path(found).resolve(strict=True), Path(found).stat()
    except OSError: return None
    return str(path) if stat.S_ISREG(state.st_mode) and os.access(path, os.X_OK) else None
def prepare_calibration_request(selected: str, *, current_poni: str = "",
                                current_mask: str = "") -> CalibrationRequest:
    """Validate one unseeded save selection before worker launch."""
    if type(selected) is not str or not selected.strip(): raise ValueError("Choose a PONI output path.")
    raw = Path(selected.strip()).expanduser()
    if not raw.suffix: raw = raw.with_suffix(".poni")
    if raw.suffix.casefold() != ".poni": raise ValueError("Calibration output must use the .poni suffix.")
    try: parent = Path(os.path.abspath(raw.parent)).resolve(strict=True)
    except OSError as error:
        raise ValueError("Calibration output parent is unavailable.") from error
    if not parent.is_dir(): raise ValueError("Calibration output parent is not a directory.")
    final = parent / raw.name
    if os.path.lexists(final): raise ValueError("Calibration output already exists.")
    key = os.path.normcase(os.path.normpath(str(final)))
    current = {os.path.normcase(os.path.normpath(str(Path(value).expanduser().resolve(strict=False)))) for value in (current_poni, current_mask) if value}
    if key in current: raise ValueError("Calibration output must differ from current assets.")
    executable = resolve_calibration_executable()
    if executable is None: raise ValueError("pyFAI-calib2 is unavailable on PATH.")
    return CalibrationRequest(str(final), executable)
def _regular_state(path: Path) -> SourceFileState:
    raw = path.lstat()
    if stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode): raise ValueError("authored PONI is not a regular file")
    state = SourceFileState.capture(path)
    if state.size < 1 or state.size > _LIMIT: raise ValueError("authored PONI must be nonempty and no larger than 1 MiB")
    return state
def _digest(path: Path) -> str:
    value, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            count += len(chunk)
            if count > _LIMIT: raise ValueError("authored PONI exceeds 1 MiB")
            value.update(chunk)
    return value.hexdigest()
def _qualify(path: Path) -> CalibrationFileProof:
    before, digest = _regular_state(path), _digest(path)
    calibration = load_detector_calibration(path)
    after, repeated = _regular_state(path), _digest(path)
    if before != after or digest != repeated:
        raise ValueError("authored PONI changed during qualification")
    config = json.dumps(dict(calibration.detector_config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    poni = calibration.poni
    geometry = tuple(float(getattr(poni, name)) for name in (
        "dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"))
    return CalibrationFileProof(after, digest, config, geometry)
def _matches(proof: CalibrationFileProof, path: Path) -> SourceFileState | None:
    try:
        state = _regular_state(path)
        same = ((state.device, state.inode, state.size) == (proof.state.device,
                proof.state.inode, proof.state.size) and _digest(path) == proof.sha256)
    except (OSError, ValueError): return None
    return state if same else None
def _tighten_qualified(path: Path, proof: CalibrationFileProof) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int or not hasattr(os, "fchmod"): raise OSError("descriptor-bound mode tightening is unavailable")
    fd = os.open(path, os.O_RDONLY | nofollow)
    try:
        before = os.fstat(fd); observed = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        expected = (proof.state.device, proof.state.inode, proof.state.size, proof.state.mtime_ns, proof.state.ctime_ns)
        if not stat.S_ISREG(before.st_mode) or observed != expected: raise ValueError("qualified calibration identity changed before mode tightening")
        os.fchmod(fd, 0o600); after = os.fstat(fd)
        if (not stat.S_ISREG(after.st_mode) or stat.S_IMODE(after.st_mode) != 0o600 or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != observed[:4]): raise OSError("qualified calibration changed during mode tightening")
    finally: os.close(fd)
    if _matches(proof, path) is None: raise ValueError("qualified calibration changed after mode tightening")
def _probe_links(stage: Path) -> None:
    source, linked, occupied = (stage / name for name in (".link-probe-source", ".link-probe-linked", ".link-probe-occupied"))
    source.write_bytes(b"probe"); occupied.write_bytes(b"preserve")
    _link(source, linked)
    if not os.path.samefile(source, linked):
        raise OSError("hard-link probe did not preserve identity")
    try: _link(source, occupied)
    except FileExistsError: pass
    else: raise OSError("hard-link probe replaced an existing name")
    if occupied.read_bytes() != b"preserve":
        raise OSError("hard-link probe changed an existing name")
    linked.unlink(); source.unlink(); occupied.unlink()
def _signal_child(process: object, *, kill: bool) -> str:
    try:
        if _WINDOWS or _killpg is None: (process.kill if kill else process.terminate)()
        else: _killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
    except Exception as error:
        module, name, message = detached_exception_strings(error); return f"{module}.{name}: {message}"
    return ""
def _wait_child(process: object, cancelled: object) -> tuple[int, str]:
    terminated = killed = False; deadline = 0.0; diagnostic = ""
    while True:
        if cancelled.is_set() and not terminated:
            terminated = True; deadline = _monotonic() + _CANCEL_GRACE_SECONDS
            observed = _signal_child(process, kill=False); diagnostic = diagnostic or observed
        try: return int(process.wait(timeout=_POLL_SECONDS)), diagnostic
        except subprocess.TimeoutExpired:
            if terminated and not killed and _monotonic() >= deadline:
                killed = True; observed = _signal_child(process, kill=True); diagnostic = diagnostic or observed
        except Exception as error:
            if not diagnostic:
                module, name, message = detached_exception_strings(error); diagnostic = f"{module}.{name}: {message}"
            time.sleep(_POLL_SECONDS)
def _cleanup(stage: Path | None, identity: tuple[int, int] | None, names: tuple[Path, ...]) -> str:
    if stage is None or identity is None: return ""
    try:
        current = stage.lstat()
        if ((current.st_dev, current.st_ino) != identity
                or not stat.S_ISDIR(current.st_mode)): return str(stage)
        for path in names:
            if path.parent != stage or not os.path.lexists(path): continue
            if not stat.S_ISDIR(path.lstat().st_mode): path.unlink()
        stage.rmdir(); return ""
    except OSError: return str(stage)
def run_calibration(
    request: CalibrationRequest, identity: OperationIdentity, cancelled: object,
    publish: Callable[[str, int, int], None],
    seal: Callable[[OperationIdentity], bool],
) -> OperationTerminal:
    """Run, strictly qualify, and create-new publish one standalone PONI."""
    stage = private = None; stage_identity = None
    proof = final_state = None; code = None; argv: tuple[str, ...] = ()
    published = False; status = OperationTerminalStatus.RETURNED; diagnostic = ""
    try:
        request.__post_init__(); final = Path(request.final_path)
        if os.path.lexists(final):
            raise ValueError("calibration request is no longer publishable")
        stage = Path(tempfile.mkdtemp(prefix=".xdart-calibrate-", dir=final.parent))
        os.chmod(stage, 0o700); stage_stat = stage.lstat()
        stage_identity = (stage_stat.st_dev, stage_stat.st_ino)
        if (not stat.S_ISDIR(stage_stat.st_mode)
                or stat.S_IMODE(stage_stat.st_mode) != 0o700):
            raise OSError("private calibration stage is not mode 0700")
        private = stage / final.name; _probe_links(stage)
        if cancelled.is_set(): raise _Cancelled("calibration cancelled")
        publish("launch", 1, 3); argv = (request.executable, "--poni", str(private))
        options = dict(cwd=str(stage), shell=False, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       close_fds=True)
        options["creationflags" if _WINDOWS else "start_new_session"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if _WINDOWS else True)
        process = _popen(argv, **options); code, process_diagnostic = _wait_child(process, cancelled)
        if process_diagnostic: raise RuntimeError(f"child process control failed: {process_diagnostic}")
        if cancelled.is_set(): raise _Cancelled("calibration cancelled")
        if code != 0: raise RuntimeError(f"pyFAI-calib2 exited with status {code}")
        publish("qualify", 2, 3); proof = _qualify(private)
        if _matches(proof, private) is None or os.path.lexists(final):
            raise ValueError("qualified calibration changed before publication")
        _tighten_qualified(private, proof)
        if not seal(identity):
            raise _Cancelled("calibration cancelled before publication")
        publish("publish", 3, 3); link_error = None
        try: _link(private, final)
        except OSError as error: link_error = error
        final_state = _matches(proof, final)
        if final_state is None or _matches(proof, private) is None:
            raise RuntimeError("calibration publication could not be reconciled") from link_error
        published = True
    except _Cancelled: status = OperationTerminalStatus.CANCELLED
    except BaseException as error:
        status = OperationTerminalStatus.FAILED
        module, name, message = detached_exception_strings(error)
        diagnostic = f"{module}.{name}: {message}"
    names = (() if stage is None else tuple(stage / name for name in (
        ".link-probe-source", ".link-probe-linked", ".link-probe-occupied")))
    recovery = _cleanup(stage, stage_identity,
                        names + (() if private is None else (private,))); recovery_class = ""
    if recovery:
        status = OperationTerminalStatus.FAILED
        try:
            current = stage.lstat()
            qualified = ((current.st_dev, current.st_ino) == stage_identity
                         and stat.S_ISDIR(current.st_mode) and proof is not None
                         and private is not None and private.parent == stage
                         and _matches(proof, private) is not None)
            recovery = str((private if qualified else stage).resolve(strict=qualified))
        except (OSError, RuntimeError):
            qualified = False; recovery = os.path.abspath(stage)
        recovery_class = "qualified-private-candidate" if qualified else "unqualified-private-stage"
        diagnostic += ("; " if diagnostic else "") + (
            f"private calibration custody relinquished ({recovery_class}): {recovery}")
    result = CalibrationResult(
        request, "" if private is None else str(private), proof, final_state,
        code, argv, "" if stage is None else str(stage), published, recovery,
        recovery_class, diagnostic)
    return OperationTerminal(identity, status, diagnostic, result)
__all__ = [
    "CalibrationFileProof", "CalibrationRequest", "CalibrationResult",
    "prepare_calibration_request", "resolve_calibration_executable",
    "run_calibration",
]
