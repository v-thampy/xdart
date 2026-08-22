"""Bounded standalone experiment-asset authoring operations."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, os, shutil, signal, stat, subprocess, tempfile, time
from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image, ImageMode, UnidentifiedImageError
from fabio.TiffIO import TiffIO; from fabio.tifimage import TifImage
from xrd_tools.io.image import load_mask, read_image
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
def _tighten_qualified(path: Path, proof, matcher: Callable = _matches) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int or not hasattr(os, "fchmod"): raise OSError("descriptor-bound mode tightening is unavailable")
    fd = os.open(path, os.O_RDONLY | nofollow)
    try:
        before = os.fstat(fd); observed = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        expected = (proof.state.device, proof.state.inode, proof.state.size, proof.state.mtime_ns, proof.state.ctime_ns)
        if not stat.S_ISREG(before.st_mode) or observed != expected: raise ValueError("qualified asset identity changed before mode tightening")
        os.fchmod(fd, 0o600); after = os.fstat(fd)
        if (not stat.S_ISREG(after.st_mode) or stat.S_IMODE(after.st_mode) != 0o600 or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != observed[:4]): raise OSError("qualified asset changed during mode tightening")
    finally: os.close(fd)
    if stat.S_IMODE(path.lstat().st_mode) != 0o600 or matcher(proof, path) is None: raise ValueError("qualified asset changed after mode tightening")
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
    if stage is None or identity is None: return "" if stage is None else str(stage)
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
_TIFF_LIMIT, _MASK_LIMIT, _PIXEL_LIMIT, _DECODED_LIMIT = 512 << 20, 64 << 20, 1 << 25, 256 << 20
@dataclass(frozen=True, slots=True)
class MaskRequest:
    source_path: str; final_path: str; executable: str
    def __post_init__(self) -> None:
        expected = os.path.splitext(self.source_path)[0] + "-mask.edf"
        if (not all(type(value) is str and os.path.isabs(value) for value in (self.source_path, self.final_path, self.executable))
                or Path(self.source_path).suffix.casefold() not in {".tif", ".tiff"}
                or self.final_path != expected):
            raise ValueError("mask request is invalid")
@dataclass(frozen=True, slots=True)
class MaskProof:
    state: SourceFileState; shape: tuple[int, int]; source_dtype: str; mask_dtype: str
    source_sha256: str; staged_sha256: str; mask_sha256: str; coercion_policy: str
@dataclass(frozen=True, slots=True)
class MaskResult:
    request: MaskRequest; private_path: str; proof: MaskProof | None; final_state: SourceFileState | None
    exit_code: int | None; argv: tuple[str, ...]; cwd: str; published: bool; recovery_path: str = ""
    recovery_class: str = ""; diagnostic: str = ""
def resolve_mask_executable(fixed: str | None = None) -> str | None:
    found = shutil.which("pyFAI-drawmask") if fixed is None else fixed
    if not found or type(found) is not str or (fixed is not None and not os.path.isabs(found)): return None
    try: path, state = Path(found).resolve(strict=True), Path(found).stat()
    except OSError: return None
    if fixed is not None and os.path.normcase(os.path.normpath(str(path))) != os.path.normcase(os.path.normpath(fixed)): return None
    return str(path) if stat.S_ISREG(state.st_mode) and os.access(path, os.X_OK) else None
def prepare_mask_request(selected: str, *, current_poni: str = "", current_mask: str = "") -> MaskRequest:
    if type(selected) is not str or not selected.strip(): raise ValueError("Choose a TIFF input file.")
    try: source = Path(selected.strip()).expanduser().resolve(strict=True); state = source.lstat()
    except OSError as error: raise ValueError("TIFF input is unavailable.") from error
    if not stat.S_ISREG(state.st_mode) or not os.access(source, os.R_OK): raise ValueError("TIFF input must be a readable regular file.")
    if source.suffix.casefold() not in {".tif", ".tiff"}: raise ValueError("Mask input must use the .tif or .tiff suffix.")
    final = Path(os.path.splitext(str(source))[0] + "-mask.edf")
    if os.path.lexists(final): raise ValueError("Mask output already exists.")
    key = os.path.normcase(os.path.normpath(str(final)))
    current = {os.path.normcase(os.path.normpath(str(Path(value).expanduser().resolve(strict=False)))) for value in (current_poni, current_mask) if value}
    if key in current: raise ValueError("Mask output must differ from current assets.")
    executable = resolve_mask_executable()
    if executable is None: raise ValueError("pyFAI-drawmask is unavailable on PATH.")
    return MaskRequest(str(source), str(final), executable)
def _asset_digest(path: Path, limit: int) -> tuple[str, int]:
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            count += len(chunk)
            if count > limit: raise ValueError("experiment asset exceeds its encoded-byte cap")
            digest.update(chunk)
    return digest.hexdigest(), count
def _copy_tiff(source: Path, private: Path):
    raw, before = source.lstat(), SourceFileState.capture(source)
    if stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode) or before.size > _TIFF_LIMIT: raise ValueError("TIFF is not regular or exceeds 512 MiB")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    source_fd, target_fd, failed = os.open(source, os.O_RDONLY | nofollow), None, False
    digest, count = hashlib.sha256(), 0
    try:
        target_fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        opened = os.fstat(source_fd)
        if (not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino, opened.st_size)
                != (before.device, before.inode, before.size)):
            raise ValueError("TIFF identity changed before copy")
        os.fchmod(target_fd, 0o600)
        with os.fdopen(source_fd, "rb", closefd=False) as incoming, os.fdopen(target_fd, "wb", closefd=False) as outgoing:
            if not incoming.seekable() or not outgoing.seekable(): raise OSError("TIFF descriptors must be seekable")
            while chunk := incoming.read(1 << 20):
                count += len(chunk)
                if count > _TIFF_LIMIT: raise ValueError("TIFF exceeds 512 MiB")
                digest.update(chunk); outgoing.write(chunk)
    except BaseException:
        failed = True; raise
    finally:
        os.close(source_fd)
        if target_fd is not None: os.close(target_fd)
        if failed and target_fd is not None:
            try: private.unlink()
            except OSError: pass
    repeated, repeated_count = _asset_digest(source, _TIFF_LIMIT); after, staged = SourceFileState.capture(source), SourceFileState.capture(private)
    staged_digest, staged_count = _asset_digest(private, _TIFF_LIMIT)
    if (before != after or digest.hexdigest() != repeated or count != repeated_count
            or count != staged.size or count != staged_count
            or digest.hexdigest() != staged_digest or stat.S_IMODE(private.lstat().st_mode) != 0o600):
        raise ValueError("TIFF changed or staged copy did not match")
    return before, digest.hexdigest(), staged, staged_digest
def _pillow_header(stream) -> tuple[tuple[int, int], np.dtype, int] | None:
    try: image = Image.open(stream, formats=("TIFF",))
    except (UnidentifiedImageError, OSError): return None
    with image:
        if image.n_frames != 1 or image.is_animated: raise ValueError("TIFF must contain exactly one frame")
        if len(image.getbands()) != 1 or image.mode == "P": raise ValueError("TIFF must be one-band numeric data")
        try: mode_dtype = np.dtype(ImageMode.getmode(image.mode).typestr)
        except (KeyError, TypeError, ValueError): return None
        if mode_dtype.kind not in "biuf": raise ValueError("TIFF mode is not numeric")
        samples = image.tag_v2.get(277, 1); bits = image.tag_v2.get(258, 1 if image.mode == "1" else mode_dtype.itemsize * 8)
        sample_format = image.tag_v2.get(339, 1)
        if isinstance(samples, tuple): samples = samples[0] if len(samples) == 1 else samples
        if isinstance(bits, tuple): bits = bits[0] if len(bits) == 1 else bits
        if isinstance(sample_format, tuple): sample_format = sample_format[0] if len(sample_format) == 1 else sample_format
        if samples != 1 or type(bits) is not int or type(sample_format) is not int or sample_format not in {1, 2, 3}: raise ValueError("TIFF sample metadata is not one-band numeric")
        itemsize = 1 if bits == 1 else (bits + 7) // 8; kind = "u" if sample_format == 1 else "i" if sample_format == 2 else "f"
        tagged = np.dtype("?") if bits == 1 else np.dtype(f"{kind}{itemsize}")
        return (int(image.height), int(image.width)), tagged, max(tagged.itemsize, mode_dtype.itemsize)
def _fabio_header(stream) -> tuple[tuple[int, int], np.dtype, int]:
    header = TiffIO(stream, cache_length=1)
    if header.getNumberOfImages() != 1: raise ValueError("TIFF must contain exactly one frame")
    info = header.getInfo(0); bits = info.get("nBits"); sample = info.get("sampleFormat", 1)
    if (type(bits) is not int or bits < 1 or bits > 64 or sample not in {1, 2, 3, 4}
            or info.get("colormap") is not None or info.get("photometricInterpretation") not in {0, 1}):
        raise ValueError("TIFF metadata is not one-band numeric")
    itemsize = 1 if bits == 1 else (bits + 7) // 8
    kind = "i" if sample == 2 else "f" if sample == 3 else "u"
    expected = np.dtype("?") if bits == 1 else np.dtype(f"{kind}{itemsize}")
    return (int(info["nRows"]), int(info["nColumns"])), expected, expected.itemsize
def _qualify_tiff(path: Path, staged: SourceFileState) -> tuple[tuple[int, int], str]:
    nofollow = getattr(os, "O_NOFOLLOW", 0); fd = os.open(path, os.O_RDONLY | nofollow)
    try:
        observed = os.fstat(fd)
        if (observed.st_dev, observed.st_ino, observed.st_size) != (staged.device, staged.inode, staged.size): raise ValueError("staged TIFF identity changed")
        with os.fdopen(os.dup(fd), "rb") as stream: header = _pillow_header(stream)
        if header is None:
            with os.fdopen(os.dup(fd), "rb") as stream: stream.seek(0); header = _fabio_header(stream)
        shape, expected, allocation = header; pixels = shape[0] * shape[1]
        if (len(shape) != 2 or min(shape) < 1 or pixels > _PIXEL_LIMIT
                or expected.kind not in "biuf" or expected.itemsize > 8
                or pixels * allocation > _DECODED_LIMIT):
            raise ValueError("TIFF exceeds the decoded array envelope")
        with os.fdopen(os.dup(fd), "rb") as stream: stream.seek(0); decoded = TifImage().read(stream)
        try:
            array = np.asarray(decoded.data)
            if (array.ndim != 2 or tuple(array.shape) != shape or array.dtype.kind not in "biuf" or not array.dtype.isnative
                    or array.dtype.itemsize > 8 or array.nbytes > _DECODED_LIMIT
                    or (array.dtype.kind, array.dtype.itemsize) != (expected.kind, expected.itemsize)):
                raise ValueError("decoded TIFF contradicts its qualified header")
            dtype = array.dtype.str
        finally: decoded.close()
        return shape, dtype
    finally: os.close(fd)
def _unchanged(path: Path, state: SourceFileState, digest: str, limit: int) -> bool:
    try:
        raw, current = path.lstat(), SourceFileState.capture(path); repeated, count = _asset_digest(path, limit)
        after, latest = SourceFileState.capture(path), path.lstat()
        return (not stat.S_ISLNK(raw.st_mode) and not stat.S_ISLNK(latest.st_mode) and stat.S_ISREG(raw.st_mode)
                and stat.S_ISREG(latest.st_mode) and current == after == state and count == state.size and repeated == digest)
    except (OSError, ValueError): return False
def _qualify_mask(path: Path, shape: tuple[int, int], source_dtype: str, source_sha: str, staged_sha: str) -> MaskProof:
    raw = path.lstat(); before = SourceFileState.capture(path)
    if stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode): raise ValueError("authored mask is not a regular file")
    if before.size < 1 or before.size > _MASK_LIMIT: raise ValueError("authored mask exceeds 64 MiB")
    digest, _ = _asset_digest(path, _MASK_LIMIT); array = read_image(path, preserve_dtype=True, exact_frame=True)
    if (array.ndim != 2 or tuple(array.shape) != shape or array.dtype.kind not in "biuf" or not array.dtype.isnative
            or array.dtype.itemsize > 8 or array.nbytes > _DECODED_LIMIT):
        raise ValueError("authored mask array is outside the qualified envelope")
    coerced = load_mask(array); expected = array != 0
    nan_pixels = int(np.isnan(array).sum()) if array.dtype.kind == "f" else 0
    if nan_pixels: expected |= np.isnan(array)
    if coerced.dtype != np.bool_ or not np.array_equal(coerced, expected): raise ValueError("authored mask coercion is not public truth")
    if not _unchanged(path, before, digest, _MASK_LIMIT): raise ValueError("authored mask changed during qualification")
    return MaskProof(before, shape, source_dtype, array.dtype.str, source_sha, staged_sha, digest, "zero-false-real-nonzero-true-nan-true-v1")
def _mask_matches(proof: MaskProof, path: Path) -> SourceFileState | None:
    try:
        raw = path.lstat()
        if stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode) or stat.S_IMODE(raw.st_mode) != 0o600: return None
        state = SourceFileState.capture(path); latest = path.lstat()
        same = (stat.S_ISREG(latest.st_mode) and stat.S_IMODE(latest.st_mode) == 0o600 and (latest.st_dev, latest.st_ino, latest.st_size) == (state.device, state.inode, state.size) and (state.device, state.inode, state.size) == (proof.state.device, proof.state.inode, proof.state.size))
        if same and os.path.normcase(os.path.normpath(str(path))) == os.path.normcase(os.path.normpath(proof.state.path)):
            digest = _asset_digest(path, _MASK_LIMIT)[0]; after = SourceFileState.capture(path); latest = path.lstat()
            same = digest == proof.mask_sha256 and after == state and stat.S_ISREG(latest.st_mode) and stat.S_IMODE(latest.st_mode) == 0o600 and (latest.st_dev, latest.st_ino, latest.st_size) == (after.device, after.inode, after.size)
    except (OSError, ValueError): return None
    return state if same else None
def run_mask(request: MaskRequest, identity: OperationIdentity, cancelled: object, publish: Callable[[str, int, int], None], seal: Callable[[OperationIdentity], bool]) -> OperationTerminal:
    stage = private_source = private_mask = None; stage_identity = None
    proof = final_state = None; code = None; argv: tuple[str, ...] = (); published = False; status, diagnostic = OperationTerminalStatus.RETURNED, ""
    try:
        request.__post_init__(); source, final = Path(request.source_path), Path(request.final_path)
        if cancelled.is_set(): raise _Cancelled("mask cancelled")
        if os.path.lexists(final): raise ValueError("mask request is no longer publishable")
        stage = Path(tempfile.mkdtemp(prefix=".xdart-mask-", dir=final.parent)); created = stage.lstat(); stage_identity = (created.st_dev, created.st_ino)
        os.chmod(stage, 0o700); stage_stat = stage.lstat()
        if (stage_stat.st_dev, stage_stat.st_ino) != stage_identity or not stat.S_ISDIR(stage_stat.st_mode) or stat.S_IMODE(stage_stat.st_mode) != 0o700: raise OSError("private mask stage is not mode 0700")
        _probe_links(stage); private_source = stage / source.name; private_mask = stage / (os.path.splitext(source.name)[0] + "-mask.edf")
        publish("copy", 1, 4); source_state, source_sha, staged_state, staged_sha = _copy_tiff(source, private_source)
        shape, source_dtype = _qualify_tiff(private_source, staged_state)
        if not _unchanged(source, source_state, source_sha, _TIFF_LIMIT) or not _unchanged(private_source, staged_state, staged_sha, _TIFF_LIMIT): raise ValueError("TIFF changed before child launch")
        if cancelled.is_set(): raise _Cancelled("mask cancelled")
        publish("launch", 2, 4); argv = (request.executable, str(private_source))
        options = dict(cwd=str(stage), shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        options["creationflags" if _WINDOWS else "start_new_session"] = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if _WINDOWS else True)
        process = _popen(argv, **options); code, process_diagnostic = _wait_child(process, cancelled)
        if process_diagnostic: raise RuntimeError(f"child process control failed: {process_diagnostic}")
        if cancelled.is_set(): raise _Cancelled("mask cancelled")
        if code != 0: raise RuntimeError(f"pyFAI-drawmask exited with status {code}")
        if not _unchanged(source, source_state, source_sha, _TIFF_LIMIT) or not _unchanged(private_source, staged_state, staged_sha, _TIFF_LIMIT): raise ValueError("TIFF changed during child execution")
        publish("qualify", 3, 4); proof = _qualify_mask(private_mask, shape, source_dtype, source_sha, staged_sha)
        if not _unchanged(private_mask, proof.state, proof.mask_sha256, _MASK_LIMIT) or os.path.lexists(final): raise ValueError("qualified mask changed before publication")
        if not seal(identity): raise _Cancelled("mask cancelled before publication")
        _tighten_qualified(private_mask, proof, _mask_matches)
        publish("publish", 4, 4); link_error = None
        try: _link(private_mask, final)
        except FileExistsError as error: raise ValueError("mask output already exists") from error
        except OSError as error: link_error = error
        final_state = _mask_matches(proof, final)
        if final_state is None or _mask_matches(proof, private_mask) is None: raise RuntimeError("mask publication could not be reconciled") from link_error
        published = True
    except _Cancelled: status = OperationTerminalStatus.CANCELLED
    except BaseException as error:
        status = OperationTerminalStatus.FAILED; module, name, message = detached_exception_strings(error); diagnostic = f"{module}.{name}: {message}"
    names = (() if stage is None else tuple(stage / name for name in (".link-probe-source", ".link-probe-linked", ".link-probe-occupied")))
    owned = tuple(path for path in (private_source, private_mask) if path is not None)
    recovery = _cleanup(stage, stage_identity, names + owned); recovery_class = ""
    if recovery:
        status = OperationTerminalStatus.FAILED; recovery_class = "qualified-private-candidate" if proof is not None and private_mask is not None and _mask_matches(proof, private_mask) is not None else "unqualified-private-stage"
        recovery = str(private_mask if recovery_class.startswith("qualified") else stage)
        diagnostic += ("; " if diagnostic else "") + f"private mask custody relinquished ({recovery_class}): {recovery}"
    result = MaskResult(request, "" if private_mask is None else str(private_mask), proof, final_state, code, argv,
        "" if stage is None else str(stage), published, recovery, recovery_class, diagnostic)
    return OperationTerminal(identity, status, diagnostic, result)
__all__ = [
    "CalibrationFileProof", "CalibrationRequest", "CalibrationResult",
    "MaskProof", "MaskRequest", "MaskResult",
    "prepare_calibration_request", "resolve_calibration_executable",
    "prepare_mask_request", "resolve_mask_executable", "run_calibration", "run_mask",
]
