"""Bounded standalone experiment-asset authoring operations."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, os, shutil, signal, stat, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image, ImageMode, UnidentifiedImageError
from fabio.TiffIO import TiffIO; from fabio.tifimage import TifImage
from xrd_tools.io.image import load_mask, read_image
from xrd_tools.io.stat_identity import identity_ctime_ns
from xrd_tools.integrate.calibration import load_detector_calibration
from .contracts import SourceFileState
from .events import detached_exception_strings
from .operation_values import OperationIdentity, OperationTerminal, OperationTerminalStatus
_LIMIT = 1 << 20
_PONI_CHILD_LIMIT = 256
_PONI_AGGREGATE_BYTES_LIMIT = 16 << 20
_DIRECT_CHILD_LIMIT = 4096
_DIRECT_NAME_BYTES_LIMIT = 1 << 20
_NAME_BYTES_LIMIT = 4096
_PATH_BYTES_LIMIT = 16 << 10
_URL_BYTES_LIMIT = 32 << 10
_CONFIG_BYTES_LIMIT = 1 << 20
_DIAGNOSTIC_BYTES_LIMIT = 64 << 10
_CHILD_STDERR_BYTES_LIMIT = 16 << 10
_CALIBRATION_SUFFIXES = frozenset({
    ".edf", ".tif", ".tiff", ".cbf", ".img", ".mar3450", ".raw",
    ".h5", ".hdf5", ".nxs", ".nexus",
})
_HDF5_SUFFIXES = frozenset({".h5", ".hdf5", ".nxs", ".nexus"})
_POLL_SECONDS, _CANCEL_GRACE_SECONDS = 0.05, 0.25
_WINDOWS = os.name == "nt"
_popen, _link = subprocess.Popen, os.link
_killpg, _monotonic = getattr(os, "killpg", None), time.monotonic
_DIR_FD_PUBLICATION = all(
    function in getattr(os, "supports_dir_fd", ())
    for function in (os.open, os.link, os.unlink, os.stat, os.rmdir)
)


def _bounded_text(value: object, limit: int, *, allow_empty: bool = False) -> bool:
    if type(value) is not str or (not value and not allow_empty):
        return False
    try:
        return len(value.encode("utf-8")) <= limit
    except UnicodeEncodeError:
        return False


def _absolute_path(value: object, *, allow_empty: bool = False) -> bool:
    return (_bounded_text(value, _PATH_BYTES_LIMIT, allow_empty=allow_empty)
            and (allow_empty and value == "" or os.path.isabs(value)))


def _sha256(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _state_valid(
    value: object, *, path: str | None = None, size_limit: int | None = None,
    nonempty: bool = False,
) -> bool:
    if type(value) is not SourceFileState or not _absolute_path(value.path):
        return False
    try:
        value.__post_init__()
    except (TypeError, ValueError):
        return False
    return ((path is None or value.path == path)
            and (not nonempty or value.size > 0)
            and (size_limit is None or value.size <= size_limit))


def _dtype_valid(value: object) -> bool:
    if not _bounded_text(value, 64):
        return False
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError):
        return False
    return (dtype.str == value and dtype.fields is None and dtype.isnative
            and dtype.kind in "biuf" and 0 < dtype.itemsize <= 8)


def _shape_valid(value: object) -> bool:
    if (type(value) is not tuple or len(value) != 2
            or any(type(item) is not int or item < 1 for item in value)):
        return False
    pixels = value[0] * value[1]
    return pixels <= _PIXEL_LIMIT


def _hdf_url_bound(value: str, source_path: str) -> bool:
    try:
        from silx.io.url import DataUrl
        url = DataUrl(value)
        file_path = url.file_path()
        rebuilt = DataUrl(
            scheme="silx", file_path=source_path,
            data_path=url.data_path(), data_slice=url.data_slice(),
        ).path()
        return (url.is_valid() and url.scheme() == "silx"
                and type(file_path) is str and os.path.isabs(file_path)
                and _path_key(file_path) == _path_key(source_path)
                and url.data_path() is not None and rebuilt == value)
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class CalibrationRequest:
    source_path: str
    executable: str
    executable_state: SourceFileState
    exact_hdf_url: str | None
    source_state: SourceFileState
    monitored_directory: str
    directory_identity: tuple[int, int]
    def __post_init__(self) -> None:
        if (not all(_absolute_path(value)
                    for value in (self.source_path, self.executable,
                                  self.monitored_directory))
                or self.exact_hdf_url is not None
                and (not _bounded_text(self.exact_hdf_url, _URL_BYTES_LIMIT)
                     or not self.exact_hdf_url.startswith("silx:")
                     or not _hdf_url_bound(
                         self.exact_hdf_url, self.source_path))
                or not _state_valid(self.source_state, path=self.source_path)
                or not _state_valid(self.executable_state,
                                    path=self.executable, nonempty=True)
                or Path(self.source_path).parent != Path(self.monitored_directory)
                or type(self.directory_identity) is not tuple
                or len(self.directory_identity) != 2
                or any(type(value) is not int or value < 0
                       for value in self.directory_identity)):
            raise ValueError("calibration request is invalid")
@dataclass(frozen=True, slots=True)
class CalibrationFileProof:
    state: SourceFileState; sha256: str; detector_config_json: str
    geometry: tuple[float, ...]
    parallax: bool | None = None
    def __post_init__(self) -> None:
        if not _bounded_text(
                self.detector_config_json, _CONFIG_BYTES_LIMIT):
            raise ValueError("calibration proof is invalid")
        try:
            detector_config = json.loads(self.detector_config_json)
            canonical_config = json.dumps(
                detector_config, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("calibration proof is invalid") from error
        if (not _state_valid(self.state, size_limit=_LIMIT, nonempty=True)
                or not _sha256(self.sha256)
                or type(detector_config) is not dict
                or canonical_config != self.detector_config_json
                or type(self.geometry) is not tuple
                or len(self.geometry) != 7
                or any(type(value) is not float or not np.isfinite(value)
                       for value in self.geometry)
                or self.parallax is not None
                and type(self.parallax) is not bool):
            raise ValueError("calibration proof is invalid")
@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    path: str
    proof: CalibrationFileProof
    def __post_init__(self) -> None:
        if (not _absolute_path(self.path)
                or Path(self.path).suffix.casefold() != ".poni"
                or not _bounded_text(Path(self.path).name, _NAME_BYTES_LIMIT)
                or type(self.proof) is not CalibrationFileProof
                or self.proof.state.path != self.path):
            raise ValueError("calibration candidate is invalid")
        self.proof.__post_init__()
@dataclass(frozen=True, slots=True)
class CalibrationResult:
    request: CalibrationRequest
    candidates: tuple[CalibrationCandidate, ...]
    exit_code: int | None
    argv: tuple[str, ...]
    cwd: str
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if (type(self.request) is not CalibrationRequest
                or type(self.candidates) is not tuple
                or len(self.candidates) > _PONI_CHILD_LIMIT
                or self.exit_code is not None
                and type(self.exit_code) is not int
                or type(self.argv) is not tuple
                or len(self.argv) > 2
                or any(not _bounded_text(value, _URL_BYTES_LIMIT)
                       for value in self.argv)
                or not _absolute_path(self.cwd)
                or self.cwd != self.request.monitored_directory
                or not _bounded_text(self.diagnostic,
                                     _DIAGNOSTIC_BYTES_LIMIT,
                                     allow_empty=True)):
            raise ValueError("calibration result is invalid")
        self.request.__post_init__()
        paths: list[str] = []
        file_bytes = config_bytes = name_bytes = 0
        for candidate in self.candidates:
            if type(candidate) is not CalibrationCandidate:
                raise ValueError("calibration result candidates are invalid")
            path, proof = candidate.path, candidate.proof
            if (not _absolute_path(path)
                    or Path(path).suffix.casefold() != ".poni"
                    or not _bounded_text(Path(path).name, _NAME_BYTES_LIMIT)
                    or type(proof) is not CalibrationFileProof
                    or not _state_valid(proof.state, path=path,
                                        size_limit=_LIMIT, nonempty=True)
                    or not _sha256(proof.sha256)
                    or not _bounded_text(proof.detector_config_json,
                                         _CONFIG_BYTES_LIMIT)
                    or type(proof.geometry) is not tuple
                    or len(proof.geometry) != 7
                    or any(type(value) is not float
                           or not np.isfinite(value)
                           for value in proof.geometry)
                    or Path(path).parent
                    != Path(self.request.monitored_directory)):
                raise ValueError("calibration result candidates are invalid")
            paths.append(path)
            file_bytes += proof.state.size
            config_bytes += len(proof.detector_config_json.encode("utf-8"))
            name_bytes += len(Path(path).name.encode("utf-8"))
        keys = tuple(_path_key(path) for path in paths)
        if (len(keys) != len(set(keys))
                or file_bytes > _PONI_AGGREGATE_BYTES_LIMIT
                or config_bytes > _PONI_AGGREGATE_BYTES_LIMIT
                or name_bytes > _DIRECT_NAME_BYTES_LIMIT):
            raise ValueError("calibration result candidates are invalid")
        for candidate in self.candidates:
            candidate.__post_init__()
        if self.candidates != tuple(sorted(
                self.candidates,
                key=lambda candidate: (
                    -candidate.proof.state.mtime_ns,
                    _path_key(candidate.path)),
        )):
            raise ValueError("calibration result candidates are invalid")


@dataclass(frozen=True, slots=True)
class ExistingMaskProof:
    state: SourceFileState
    shape: tuple[int, int]
    dtype: str
    sha256: str
    coercion_policy: str

    def __post_init__(self) -> None:
        if (not _state_valid(self.state, size_limit=_MASK_LIMIT, nonempty=True)
                or not _shape_valid(self.shape)
                or not _dtype_valid(self.dtype)
                or self.shape[0] * self.shape[1] * np.dtype(self.dtype).itemsize
                > _DECODED_LIMIT
                or not _sha256(self.sha256)
                or self.coercion_policy != _MASK_COERCION_POLICY):
            raise ValueError("existing mask proof is invalid")


@dataclass(frozen=True, slots=True)
class AuthoredAssetCandidate:
    asset: str
    path: str
    proof: CalibrationFileProof | MaskProof | ExistingMaskProof
    state: SourceFileState
    source_path: str | None = None

    def __post_init__(self) -> None:
        generated = type(self.proof) is MaskProof
        if (self.asset not in {"poni", "mask"}
                or not _absolute_path(self.path)
                or not _state_valid(self.state, path=self.path,
                                    size_limit=_LIMIT if self.asset == "poni"
                                    else _MASK_LIMIT, nonempty=True)
                or (self.asset == "poni")
                != (type(self.proof) is CalibrationFileProof)
                or self.asset == "mask"
                and type(self.proof) not in {MaskProof, ExistingMaskProof}
                or generated != (self.source_path is not None)
                or self.source_path is not None
                and not _absolute_path(self.source_path)):
            raise ValueError("authored asset candidate is invalid")
        self.proof.__post_init__()
        if (type(self.proof) in {CalibrationFileProof, ExistingMaskProof}
                and self.state != self.proof.state):
            raise ValueError("authored asset candidate proof is inexact")
        if generated:
            proof_state = self.proof.state
            if ((self.state.device, self.state.inode, self.state.size,
                 self.state.mtime_ns)
                    != (proof_state.device, proof_state.inode,
                        proof_state.size, proof_state.mtime_ns)
                    or proof_state.ctime_ns > self.state.ctime_ns
                    or Path(proof_state.path).name != Path(self.path).name):
                raise ValueError("authored mask candidate proof is inexact")


@dataclass(frozen=True, slots=True)
class AssetValidationRequest:
    asset: str
    path: str
    expected_shape: tuple[int, int] | None = None
    candidate: AuthoredAssetCandidate | None = None
    source_request: CalibrationRequest | MaskRequest | None = None

    def __post_init__(self) -> None:
        if (self.asset not in {"poni", "mask"}
                or not _absolute_path(self.path)
                or (self.expected_shape is None) != (self.asset == "poni")
                or self.expected_shape is not None
                and (type(self.expected_shape) is not tuple
                     or len(self.expected_shape) != 2
                     or any(type(value) is not int or value < 1
                            for value in self.expected_shape))
                or self.candidate is not None
                and (type(self.candidate) is not AuthoredAssetCandidate
                     or self.candidate.asset != self.asset
                     or self.candidate.path != self.path
                     or self.expected_shape is not None
                     and self.candidate.proof.shape != self.expected_shape)
                or self.source_request is not None
                and (type(self.source_request)
                     not in {CalibrationRequest, MaskRequest}
                     or (self.asset == "poni")
                     != (type(self.source_request) is CalibrationRequest))):
            raise ValueError("asset validation request is invalid")
        if self.candidate is not None:
            self.candidate.__post_init__()


@dataclass(frozen=True, slots=True)
class AssetValidationResult:
    request: AssetValidationRequest
    candidate: AuthoredAssetCandidate

    def __post_init__(self) -> None:
        if (type(self.request) is not AssetValidationRequest
                or type(self.candidate) is not AuthoredAssetCandidate
                or self.candidate.asset != self.request.asset
                or self.candidate.path != self.request.path
                or self.request.expected_shape is not None
                and self.candidate.proof.shape != self.request.expected_shape
                or self.request.candidate is not None
                and self.candidate is not self.request.candidate):
            raise ValueError("asset validation result is invalid")
        self.request.__post_init__()
        self.candidate.__post_init__()
class _Cancelled(RuntimeError): pass
def _path_key(value: str) -> str:
    return os.path.normcase(os.path.normpath(value))


def _validated_executable(value: object, *, exact: bool) -> str | None:
    if (type(value) is not str or not value
            or exact and not os.path.isabs(value)):
        return None
    try:
        path = Path(value).resolve(strict=True)
        state = path.stat()
    except OSError:
        return None
    if (exact and _path_key(str(path)) != _path_key(value)
            or not stat.S_ISREG(state.st_mode) or not os.access(path, os.X_OK)):
        return None
    return str(path)


def _resolve_authoring_executable(
    name: str, fixed: str | None = None,
) -> str | None:
    """Prefer the tool installed beside the running real Python executable."""

    if fixed is not None:
        return _validated_executable(fixed, exact=True)
    try:
        interpreter = Path(sys.executable).resolve(strict=True)
    except (OSError, TypeError, ValueError):
        interpreter = None
    if interpreter is not None:
        sibling = _validated_executable(
            str(interpreter.with_name(name)), exact=False,
        )
        if sibling is not None:
            return sibling
    found = shutil.which(name)
    return _validated_executable(found, exact=False)


def resolve_calibration_executable(fixed: str | None = None) -> str | None:
    # The installed xdart entry point applies the pinned Rayonix backport in
    # the calibration child before delegating to the normal pyFAI GUI.
    return _resolve_authoring_executable("xdart-calib2", fixed)
def _directory_identity(path: Path) -> tuple[int, int]:
    state = path.lstat()
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise ValueError("calibration source directory is unavailable")
    return int(state.st_dev), int(state.st_ino)


def _source_context_current(
    source: Path, source_state: SourceFileState,
    directory: Path, directory_identity: tuple[int, int],
) -> bool:
    try:
        raw = source.lstat()
        return (source.parent == directory
                and not stat.S_ISLNK(raw.st_mode)
                and stat.S_ISREG(raw.st_mode)
                and SourceFileState.capture(source) == source_state
                and _directory_identity(directory) == directory_identity)
    except (OSError, TypeError, ValueError):
        return False


def _poni_inventory(
    directory: Path,
) -> tuple[tuple[int, int], tuple[tuple[str, SourceFileState], ...]]:
    """Return one stable, bounded direct-child PONI inventory."""

    before = directory.stat()
    identity = _directory_identity(directory)
    rows: list[tuple[str, SourceFileState]] = []
    child_count = poni_count = encoded_bytes = poni_bytes = 0
    try:
        iterator = os.scandir(directory)
    except OSError as error:
        raise ValueError("calibration source directory is unavailable") from error
    with iterator:
        for entry in iterator:
            child_count += 1
            if child_count > _DIRECT_CHILD_LIMIT:
                raise ValueError("calibration source directory exceeds the direct-child limit")
            name = entry.name
            try:
                size = len(name.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError("calibration source directory contains an invalid name") from error
            if size < 1 or size > _NAME_BYTES_LIMIT:
                raise ValueError("calibration source directory contains an oversized name")
            encoded_bytes += size
            if encoded_bytes > _DIRECT_NAME_BYTES_LIMIT:
                raise ValueError("calibration source directory exceeds the encoded-name limit")
            if Path(name).suffix.casefold() != ".poni":
                continue
            poni_count += 1
            if poni_count > _PONI_CHILD_LIMIT:
                raise ValueError("calibration source directory exceeds the PONI limit")
            candidate = directory / name
            try:
                raw = candidate.lstat()
                if (stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode)
                        or raw.st_size < 1 or raw.st_size > _LIMIT):
                    continue
                poni_bytes += raw.st_size
                if poni_bytes > _PONI_AGGREGATE_BYTES_LIMIT:
                    raise ValueError(
                        "calibration source directory exceeds the PONI byte limit"
                    )
                captured = SourceFileState.capture(candidate)
                if ((captured.device, captured.inode, captured.size,
                     captured.mtime_ns, captured.ctime_ns)
                        != (raw.st_dev, raw.st_ino, raw.st_size,
                            raw.st_mtime_ns, raw.st_ctime_ns)):
                    continue
            except OSError:
                continue
            rows.append((name, captured))
    after = directory.stat()
    if ((before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_mtime_ns,
                after.st_ctime_ns)
            or _directory_identity(directory) != identity):
        raise ValueError("calibration source directory changed during inventory")
    return identity, tuple(sorted(rows, key=lambda row: _path_key(row[0])))


def _calibration_source(
    selected: str,
) -> tuple[Path, str | None]:
    if type(selected) is not str or not selected.strip():
        raise ValueError("Choose a calibration source image.")
    value = selected.strip()
    input_argument = None
    if value.startswith("silx:"):
        try:
            from silx.io.url import DataUrl
            url = DataUrl(value)
            file_path = url.file_path()
        except Exception as error:
            raise ValueError("Calibration HDF URL is invalid.") from error
        if (not url.is_valid() or url.scheme() != "silx"
                or type(file_path) is not str or not file_path
                or url.data_path() is None):
            raise ValueError("Calibration HDF URL is not an exact dataset/frame selection.")
        source = Path(file_path).expanduser()
        input_argument = url
    else:
        source = Path(value).expanduser()
    try:
        source = source.resolve(strict=True)
        raw = source.lstat()
    except OSError as error:
        raise ValueError("Calibration source is unavailable.") from error
    if (stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode)
            or not os.access(source, os.R_OK)):
        raise ValueError("Calibration source must be a readable regular file.")
    if source.suffix.casefold() not in _CALIBRATION_SUFFIXES:
        raise ValueError("Calibration source type is unsupported.")
    if input_argument is not None:
        try:
            input_argument = type(input_argument)(
                scheme="silx", file_path=str(source),
                data_path=input_argument.data_path(),
                data_slice=input_argument.data_slice(),
            ).path()
        except Exception as error:
            raise ValueError("Calibration HDF URL is invalid.") from error
        if not _bounded_text(input_argument, _URL_BYTES_LIMIT):
            raise ValueError("Calibration HDF URL exceeds its encoded-byte cap.")
    return source, input_argument


def _hard_hdf_dataset(handle, path: str):
    import h5py
    parts = tuple(part for part in path.split("/") if part)
    if (not path.startswith("/") or not parts or len(parts) > 256
            or path != "/" + "/".join(parts)
            or len(path.encode("utf-8")) > 4096):
        raise ValueError("HDF dataset path is invalid.")
    owner: h5py.Group | h5py.File = handle
    for index, name in enumerate(parts):
        if not isinstance(owner.get(name, getlink=True), h5py.HardLink):
            raise ValueError("HDF dataset is not locally owned.")
        value = owner.get(name)
        if index + 1 == len(parts):
            if not isinstance(value, h5py.Dataset):
                raise ValueError("HDF selection is not a dataset.")
            return value
        if not isinstance(value, h5py.Group):
            raise ValueError("HDF dataset ancestry is invalid.")
        owner = value
    raise ValueError("HDF dataset is unavailable.")


def _qualify_hdf_argument(path: Path, argument: str) -> None:
    import h5py
    from silx.io.url import DataUrl
    url = DataUrl(argument)
    if (not url.is_valid() or url.scheme() != "silx"
            or _path_key(str(Path(url.file_path()).expanduser().resolve(strict=True)))
            != _path_key(str(path))):
        raise ValueError("Calibration HDF URL changed during qualification.")
    before = SourceFileState.capture(path)
    with h5py.File(path, "r") as handle:
        dataset = _hard_hdf_dataset(handle, url.data_path())
        properties = dataset.id.get_create_plist()
        try:
            external_count = int(properties.get_external_count())
        finally:
            properties.close()
        shape = tuple(int(value) for value in dataset.shape)
        dtype = np.dtype(dataset.dtype)
        selection = url.data_slice()
        if dataset.ndim == 2:
            exact = selection in (None, ())
            frame_shape = shape
        elif dataset.ndim == 3:
            exact = (type(selection) is tuple and len(selection) == 1
                     and type(selection[0]) is int
                     and 0 <= selection[0] < shape[0])
            frame_shape = shape[1:]
        else:
            exact = False; frame_shape = ()
        if (dataset.is_virtual or external_count != 0
                or not exact or len(frame_shape) != 2 or min(frame_shape) < 1
                or dtype.fields is not None or dtype.kind not in "biuf"
                or dtype.itemsize > 8
                or frame_shape[0] * frame_shape[1] > _PIXEL_LIMIT
                or frame_shape[0] * frame_shape[1] * dtype.itemsize
                > _DECODED_LIMIT):
            raise ValueError("Calibration HDF URL is not one bounded numeric frame.")
    if SourceFileState.capture(path) != before:
        raise ValueError("Calibration HDF source changed during qualification.")


def _mask_hdf_frame(
    path: Path, argument: str, *, read: bool,
    expected_source_state: SourceFileState | None = None,
    expected_target_path: str | None = None,
    expected_target_state: SourceFileState | None = None,
) -> tuple[
    SourceFileState, str, SourceFileState | None,
    tuple[int, int], str, np.ndarray | None, str,
]:
    """Qualify one local HDF frame and optionally return its bounded pixels."""

    import h5py
    from silx.io.url import DataUrl

    url = DataUrl(argument)
    if (not url.is_valid() or url.scheme() != "silx"
            or not _hdf_url_bound(argument, str(path))):
        raise ValueError("Mask HDF URL changed during qualification.")
    source_raw = path.lstat()
    before = SourceFileState.capture(path)
    if (stat.S_ISLNK(source_raw.st_mode)
            or not stat.S_ISREG(source_raw.st_mode)
            or (source_raw.st_dev, source_raw.st_ino, source_raw.st_size,
                source_raw.st_mtime_ns, source_raw.st_ctime_ns)
            != (before.device, before.inode, before.size,
                before.mtime_ns, before.ctime_ns)
            or expected_source_state is not None
            and before != expected_source_state):
        raise ValueError("Mask HDF source changed before qualification.")

    target_path = ""
    target_before = None
    target_handle = None
    array = None
    digest = ""
    try:
        with h5py.File(path, "r") as master:
            data_path = url.data_path()
            parts = tuple(part for part in data_path.split("/") if part)
            if (not data_path.startswith("/") or not parts
                    or len(parts) > 256
                    or data_path != "/" + "/".join(parts)
                    or len(data_path.encode("utf-8")) > 4096):
                raise ValueError("HDF dataset path is invalid.")
            owner: h5py.Group | h5py.File = master
            for name in parts[:-1]:
                if not isinstance(owner.get(name, getlink=True), h5py.HardLink):
                    raise ValueError("HDF dataset ancestry is not locally owned.")
                value = owner.get(name)
                if not isinstance(value, h5py.Group):
                    raise ValueError("HDF dataset ancestry is invalid.")
                owner = value
            name = parts[-1]
            link = owner.get(name, getlink=True)
            if isinstance(link, h5py.HardLink):
                dataset = owner.get(name)
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError("HDF selection is not a dataset.")
            elif isinstance(link, h5py.ExternalLink):
                filename = link.filename
                if (type(filename) is not str or not filename
                        or filename in {".", ".."}
                        or "/" in filename or "\\" in filename
                        or ":" in filename
                        or not _bounded_text(filename, _NAME_BYTES_LIMIT)):
                    raise ValueError(
                        "Mask HDF ExternalLink must name one relative "
                        "same-directory file.")
                target = path.parent / filename
                try:
                    raw = target.lstat()
                    resolved = target.resolve(strict=True)
                except OSError as error:
                    raise ValueError(
                        "Mask HDF ExternalLink target is unavailable.") from error
                if (stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode)
                        or not os.access(target, os.R_OK)
                        or resolved != target
                        or target == path
                        or target.suffix.casefold() not in _HDF5_SUFFIXES):
                    raise ValueError(
                        "Mask HDF ExternalLink target is not one readable "
                        "regular direct-child HDF file.")
                target_path = str(target)
                target_before = SourceFileState.capture(target)
                latest = target.lstat()
                if (stat.S_ISLNK(latest.st_mode)
                        or not stat.S_ISREG(latest.st_mode)
                        or (latest.st_dev, latest.st_ino, latest.st_size,
                            latest.st_mtime_ns, latest.st_ctime_ns)
                        != (target_before.device, target_before.inode,
                            target_before.size, target_before.mtime_ns,
                            target_before.ctime_ns)):
                    raise ValueError(
                        "Mask HDF ExternalLink target changed before open.")
                target_handle = h5py.File(target, "r")
                dataset = _hard_hdf_dataset(target_handle, link.path)
            else:
                raise ValueError(
                    "Mask HDF selection must be hard-owned or one bounded "
                    "same-directory ExternalLink.")

            if expected_target_path is not None:
                if target_path != expected_target_path:
                    raise ValueError("Mask HDF dataset ownership changed.")
                if (target_before is None) != (expected_target_state is None):
                    raise ValueError("Mask HDF target custody changed.")
                if (target_before is not None
                        and target_before != expected_target_state):
                    raise ValueError("Mask HDF target changed.")

            properties = dataset.id.get_create_plist()
            try:
                external_count = int(properties.get_external_count())
            finally:
                properties.close()
            shape = tuple(int(value) for value in dataset.shape)
            dtype = np.dtype(dataset.dtype)
            selection = url.data_slice()
            if dataset.ndim == 2:
                exact = selection in (None, ())
                frame_shape = shape
                index = ()
            elif dataset.ndim == 3:
                exact = (type(selection) is tuple and len(selection) == 1
                         and type(selection[0]) is int
                         and 0 <= selection[0] < shape[0])
                frame_shape = shape[1:]
                index = selection
            else:
                exact = False
                frame_shape = ()
                index = ()
            pixels = (frame_shape[0] * frame_shape[1]
                      if len(frame_shape) == 2 else 0)
            if (dataset.is_virtual or external_count != 0 or not exact
                    or len(frame_shape) != 2 or min(frame_shape) < 1
                    or dtype.fields is not None or dtype.kind not in "biuf"
                    or dtype.itemsize < 1 or dtype.itemsize > 8
                    or pixels > _PIXEL_LIMIT
                    or pixels * dtype.itemsize > _DECODED_LIMIT):
                raise ValueError(
                    "Mask HDF URL is not one bounded numeric frame.")
            native_dtype = (dtype if dtype.isnative
                            else dtype.newbyteorder("="))
            if read:
                observed = np.asarray(dataset[()] if not index else dataset[index])
                if (observed.ndim != 2
                        or tuple(observed.shape) != frame_shape
                        or observed.dtype != dtype
                        or observed.nbytes > _DECODED_LIMIT):
                    raise ValueError(
                        "Mask HDF pixels contradict the qualified dataset.")
                if not observed.dtype.isnative:
                    observed = observed.astype(native_dtype, copy=False)
                array = np.ascontiguousarray(observed)
                digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
            source_dtype = native_dtype.str
    finally:
        if target_handle is not None:
            target_handle.close()

    after = SourceFileState.capture(path)
    source_latest = path.lstat()
    if (before != after or stat.S_ISLNK(source_latest.st_mode)
            or not stat.S_ISREG(source_latest.st_mode)
            or (source_latest.st_dev, source_latest.st_ino,
                source_latest.st_size, source_latest.st_mtime_ns,
                source_latest.st_ctime_ns)
            != (after.device, after.inode, after.size,
                after.mtime_ns, after.ctime_ns)):
        raise ValueError("Mask HDF source changed during qualification.")
    if target_before is not None:
        target_after = SourceFileState.capture(Path(target_path))
        target_latest = Path(target_path).lstat()
        if (target_before != target_after
                or stat.S_ISLNK(target_latest.st_mode)
                or not stat.S_ISREG(target_latest.st_mode)
                or (target_latest.st_dev, target_latest.st_ino,
                    target_latest.st_size, target_latest.st_mtime_ns,
                    target_latest.st_ctime_ns)
                != (target_after.device, target_after.inode,
                    target_after.size, target_after.mtime_ns,
                    target_after.ctime_ns)):
            raise ValueError("Mask HDF target changed during qualification.")
    return (
        before, target_path, target_before, frame_shape, source_dtype,
        array, digest,
    )


def prepare_calibration_request(selected: str) -> CalibrationRequest:
    """Freeze the lightweight source identity selected on the GUI thread."""

    source, explicit = _calibration_source(selected)
    state = SourceFileState.capture(source)
    suffix = source.suffix.casefold()
    if explicit is not None:
        if suffix not in _HDF5_SUFFIXES:
            raise ValueError("Calibration dataset URL is not HDF5.")
    directory = source.parent
    identity = _directory_identity(directory)
    executable = resolve_calibration_executable()
    if executable is None:
        raise ValueError("xdart-calib2 is unavailable; reinstall xdart[gui] in this environment.")
    return CalibrationRequest(
        str(source), executable, SourceFileState.capture(Path(executable)), explicit,
        state, str(directory), identity,
    )
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
    return CalibrationFileProof(
        after, digest, config, geometry, calibration.parallax,
    )
def _matches(proof: CalibrationFileProof, path: Path) -> SourceFileState | None:
    try:
        state = _regular_state(path)
        same = ((state.device, state.inode, state.size) == (proof.state.device,
                proof.state.inode, proof.state.size) and _digest(path) == proof.sha256)
    except (OSError, ValueError): return None
    return state if same else None


def qualify_calibration_candidate(selected: str) -> CalibrationCandidate:
    """Strictly qualify one existing PONI without changing it."""

    if type(selected) is not str or not selected.strip():
        raise ValueError("Choose an existing PONI file.")
    try:
        selected_path = Path(selected.strip()).expanduser()
        raw = selected_path.lstat()
        if stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode):
            raise ValueError("PONI file must be a regular non-symlink file.")
        path = selected_path.resolve(strict=True)
    except OSError as error:
        raise ValueError("PONI file is unavailable.") from error
    if path.suffix.casefold() != ".poni":
        raise ValueError("PONI file must use the .poni suffix.")
    candidate = CalibrationCandidate(str(path), _qualify(path))
    if not calibration_candidate_current(candidate):
        raise ValueError("PONI file changed during qualification.")
    return candidate


def calibration_candidate_current(candidate: CalibrationCandidate) -> bool:
    """Revalidate the exact admitted PONI object and bytes."""

    if type(candidate) is not CalibrationCandidate:
        return False
    try:
        path = Path(candidate.path)
        state = _regular_state(path)
        return (state == candidate.proof.state
                and _digest(path) == candidate.proof.sha256
                and SourceFileState.capture(path) == state)
    except (OSError, ValueError):
        return False
def _private_mode(mode: int, expected: int) -> bool:
    # Windows inherits the directory ACL; its stat/chmod do not model POSIX
    # owner/group bits. Object identity and byte checks apply on both platforms.
    return _WINDOWS or stat.S_IMODE(mode) == expected


def _tighten_qualified(path: Path, proof, matcher: Callable = _matches) -> None:
    if _WINDOWS:
        if matcher(proof, path) is None:
            raise ValueError("qualified asset changed before publication")
        return
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int or not hasattr(os, "fchmod"): raise OSError("descriptor-bound mode tightening is unavailable")
    fd = os.open(path, os.O_RDONLY | nofollow)
    try:
        before = os.fstat(fd); observed = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, identity_ctime_ns(before.st_ctime_ns))
        expected = (proof.state.device, proof.state.inode, proof.state.size, proof.state.mtime_ns, identity_ctime_ns(proof.state.ctime_ns))
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
def _wait_child(
    process: object, cancelled: object, child_stderr=None,
) -> tuple[int, str, bool]:
    terminated = killed = stderr_truncated = False
    deadline = 0.0; diagnostic = ""
    while True:
        if cancelled.is_set() and not terminated:
            terminated = True; deadline = _monotonic() + _CANCEL_GRACE_SECONDS
            observed = _signal_child(process, kill=False); diagnostic = diagnostic or observed
        try:
            code = int(process.wait(timeout=_POLL_SECONDS))
            if child_stderr is not None:
                stderr_truncated |= _compact_child_stderr(child_stderr)
            return code, diagnostic, stderr_truncated
        except subprocess.TimeoutExpired:
            if child_stderr is not None:
                stderr_truncated |= _compact_child_stderr(child_stderr)
            if terminated and not killed and _monotonic() >= deadline:
                killed = True; observed = _signal_child(process, kill=True); diagnostic = diagnostic or observed
        except Exception as error:
            if child_stderr is not None:
                stderr_truncated |= _compact_child_stderr(child_stderr)
            if not diagnostic:
                module, name, message = detached_exception_strings(error); diagnostic = f"{module}.{name}: {message}"
            time.sleep(_POLL_SECONDS)


def _private_child_stderr(directory: Path):
    stream = tempfile.TemporaryFile(
        mode="w+b", prefix=".xdart-authoring-stderr-", dir=directory,
    )
    try:
        descriptor = stream.fileno()
        if not _WINDOWS:
            os.fchmod(descriptor, 0o600)
        observed = os.fstat(descriptor)
        if (not stat.S_ISREG(observed.st_mode)
                or not _private_mode(observed.st_mode, 0o600)):
            raise OSError("private child stderr is not mode 0600")
    except BaseException:
        stream.close()
        raise
    return stream


def _compact_child_stderr(stream) -> bool:
    stream.flush()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    if size <= _CHILD_STDERR_BYTES_LIMIT:
        return False
    start = max(0, size - _CHILD_STDERR_BYTES_LIMIT)
    stream.seek(start)
    payload = stream.read(_CHILD_STDERR_BYTES_LIMIT)
    stream.seek(0)
    stream.write(payload)
    stream.truncate()
    stream.flush()
    return True


def _bounded_child_stderr(stream, *, truncated: bool = False) -> str:
    truncated = _compact_child_stderr(stream) or truncated
    stream.seek(0)
    payload = stream.read(_CHILD_STDERR_BYTES_LIMIT)
    text = payload.decode("utf-8", errors="replace").replace("\x00", "�").strip()
    return ("[stderr truncated] " if truncated else "") + text


def _nonzero_child_error(tool: str, code: int, observed: str) -> RuntimeError:
    suffix = f": {observed}" if observed else ""
    return RuntimeError(f"{tool} exited with status {code}{suffix}")


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
    """Run pyFAI and discover bounded new or changed direct-child PONIs."""

    candidates: tuple[CalibrationCandidate, ...] = ()
    code = None; argv: tuple[str, ...] = ()
    status = OperationTerminalStatus.RETURNED; diagnostic = ""
    try:
        request.__post_init__()
        if (resolve_calibration_executable(request.executable)
                != request.executable
                or SourceFileState.capture(Path(request.executable))
                != request.executable_state):
            raise ValueError("calibration executable changed before launch")
        source = Path(request.source_path)
        directory = Path(request.monitored_directory)
        if not _source_context_current(
                source, request.source_state, directory,
                request.directory_identity):
            raise ValueError("calibration source context changed before preflight")
        suffix = source.suffix.casefold()
        input_argument = None
        if suffix in {".tif", ".tiff"}:
            if not 0 < request.source_state.size <= _TIFF_LIMIT:
                raise ValueError("calibration TIFF exceeds the encoded-byte cap")
            _qualify_tiff(source, request.source_state)
            input_argument = str(source)
        elif request.exact_hdf_url is not None:
            _qualify_hdf_argument(source, request.exact_hdf_url)
            input_argument = request.exact_hdf_url
        elif suffix not in _HDF5_SUFFIXES:
            input_argument = str(source)
        if not _source_context_current(
                source, request.source_state, directory,
                request.directory_identity):
            raise ValueError("calibration source changed during preflight")
        inventory_identity, prior_rows = _poni_inventory(directory)
        if (inventory_identity != request.directory_identity
                or not _source_context_current(
                    source, request.source_state, directory,
                    request.directory_identity)
                or SourceFileState.capture(Path(request.executable))
                != request.executable_state):
            raise ValueError("calibration source directory changed before launch")
        if cancelled.is_set(): raise _Cancelled("calibration cancelled")
        publish("launch", 1, 2)
        argv = ((request.executable,) if input_argument is None else
                (request.executable, input_argument))
        with _private_child_stderr(directory) as child_stderr:
            options = dict(
                cwd=str(directory), shell=False, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=child_stderr,
                close_fds=True,
            )
            options["creationflags" if _WINDOWS else "start_new_session"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if _WINDOWS else True)
            process = _popen(argv, **options)
            code, process_diagnostic, stderr_truncated = _wait_child(
                process, cancelled, child_stderr,
            )
            child_diagnostic = _bounded_child_stderr(
                child_stderr, truncated=stderr_truncated,
            )
            if process_diagnostic:
                raise RuntimeError(
                    f"child process control failed: {process_diagnostic}"
                    + (f": {child_diagnostic}" if child_diagnostic else ""))
            if cancelled.is_set():
                raise _Cancelled("calibration cancelled")
            if code != 0:
                raise _nonzero_child_error(
                    "pyFAI-calib2", code, child_diagnostic)
        if not _source_context_current(
                source, request.source_state, directory,
                request.directory_identity):
            raise ValueError("calibration source context changed during child execution")
        publish("discover", 2, 2)
        final_identity, final_rows = _poni_inventory(directory)
        if final_identity != request.directory_identity:
            raise ValueError("calibration source directory changed during discovery")
        prior = dict(prior_rows)
        qualified: list[CalibrationCandidate] = []
        for name, state in final_rows:
            if prior.get(name) == state:
                continue
            path = directory / name
            try:
                proof = _qualify(path)
                candidate = CalibrationCandidate(str(path), proof)
                if proof.state != state or not calibration_candidate_current(candidate):
                    continue
            except (OSError, ValueError, TypeError, OverflowError):
                continue
            qualified.append(candidate)
        candidates = tuple(sorted(
            qualified,
            key=lambda candidate: (
                -candidate.proof.state.mtime_ns, _path_key(candidate.path)),
        ))
        if not _source_context_current(
                source, request.source_state, directory,
                request.directory_identity):
            raise ValueError("calibration source context changed during discovery")
        if not seal(identity):
            raise _Cancelled("calibration cancelled before result transfer")
    except _Cancelled: status = OperationTerminalStatus.CANCELLED
    except BaseException as error:
        status = OperationTerminalStatus.FAILED
        module, name, message = detached_exception_strings(error)
        diagnostic = f"{module}.{name}: {message}"
    result = CalibrationResult(request, candidates, code, argv,
                               request.monitored_directory, diagnostic)
    return OperationTerminal(identity, status, diagnostic, result)
_TIFF_LIMIT, _MASK_LIMIT, _PIXEL_LIMIT, _DECODED_LIMIT = 512 << 20, 64 << 20, 1 << 25, 256 << 20
_MASK_COERCION_POLICY = "zero-false-real-nonzero-true-nan-true-v1"
@dataclass(frozen=True, slots=True)
class MaskRequest:
    source_path: str; final_path: str; executable: str
    executable_state: SourceFileState; source_state: SourceFileState
    source_directory: str; directory_identity: tuple[int, int]
    exact_hdf_url: str | None = None
    hdf_target_path: str = ""
    hdf_target_state: SourceFileState | None = None
    def __post_init__(self) -> None:
        expected = os.path.splitext(self.source_path)[0] + "-mask.edf"
        suffix = Path(self.source_path).suffix.casefold()
        source_kind_valid = (
            suffix in {".tif", ".tiff"} and self.exact_hdf_url is None
            and self.hdf_target_path == "" and self.hdf_target_state is None
            or suffix in _HDF5_SUFFIXES
            and type(self.exact_hdf_url) is str
            and _bounded_text(self.exact_hdf_url, _URL_BYTES_LIMIT)
            and self.exact_hdf_url.startswith("silx:")
            and _hdf_url_bound(self.exact_hdf_url, self.source_path)
            and bool(self.hdf_target_path)
            == (self.hdf_target_state is not None)
            and _absolute_path(self.hdf_target_path, allow_empty=True)
            and (self.hdf_target_state is None
                 or _state_valid(
                     self.hdf_target_state, path=self.hdf_target_path,
                     nonempty=True,
                 )
                 and Path(self.hdf_target_path).parent
                 == Path(self.source_path).parent)
        )
        if (not all(_absolute_path(value) for value in (
                    self.source_path, self.final_path, self.executable))
                or not source_kind_valid
                or self.final_path != expected
                or not _state_valid(self.executable_state,
                                    path=self.executable, nonempty=True)
                or not _state_valid(self.source_state,
                                    path=self.source_path, nonempty=True)
                or self.source_directory != str(Path(self.source_path).parent)
                or type(self.directory_identity) is not tuple
                or len(self.directory_identity) != 2
                or any(type(value) is not int or value < 0
                       for value in self.directory_identity)):
            raise ValueError("mask request is invalid")
@dataclass(frozen=True, slots=True)
class MaskProof:
    state: SourceFileState; shape: tuple[int, int]; source_dtype: str; mask_dtype: str
    source_sha256: str; staged_sha256: str; mask_sha256: str; coercion_policy: str

    def __post_init__(self) -> None:
        if (not _state_valid(self.state, size_limit=_MASK_LIMIT,
                            nonempty=True)
                or not _shape_valid(self.shape)
                or not _dtype_valid(self.source_dtype)
                or not _dtype_valid(self.mask_dtype)
                or self.shape[0] * self.shape[1]
                * max(np.dtype(self.source_dtype).itemsize,
                      np.dtype(self.mask_dtype).itemsize) > _DECODED_LIMIT
                or not _sha256(self.source_sha256)
                or self.source_sha256 != self.staged_sha256
                or not _sha256(self.mask_sha256)
                or self.coercion_policy != _MASK_COERCION_POLICY):
            raise ValueError("mask proof is invalid")
@dataclass(frozen=True, slots=True)
class MaskResult:
    request: MaskRequest; private_path: str; proof: MaskProof | None; final_state: SourceFileState | None
    exit_code: int | None; argv: tuple[str, ...]; cwd: str; published: bool; recovery_path: str = ""
    recovery_class: str = ""; diagnostic: str = ""

    def __post_init__(self) -> None:
        if type(self.request) is not MaskRequest:
            raise ValueError("mask result is invalid")
        self.request.__post_init__()
        proof = self.proof
        final = self.final_state
        if (not _absolute_path(self.private_path, allow_empty=True)
                or not _absolute_path(self.cwd, allow_empty=True)
                or type(proof) not in {MaskProof, type(None)}
                or type(final) not in {SourceFileState, type(None)}
                or self.exit_code is not None and type(self.exit_code) is not int
                or type(self.argv) is not tuple
                or any(not _bounded_text(value, _PATH_BYTES_LIMIT)
                       for value in self.argv)
                or type(self.published) is not bool
                or not _absolute_path(self.recovery_path, allow_empty=True)
                or self.recovery_class not in {
                    "", "qualified-private-candidate", "unqualified-private-stage",
                }
                or not _bounded_text(self.diagnostic,
                                     _DIAGNOSTIC_BYTES_LIMIT,
                                     allow_empty=True)
                or bool(self.recovery_path) != bool(self.recovery_class)
                or not self.argv and self.exit_code is not None
                or self.argv and (
                    self.argv != (self.request.executable,
                                  str(Path(self.cwd)
                                      / (Path(self.request.source_path).name
                                         if self.request.exact_hdf_url is None
                                         else Path(self.request.source_path).stem
                                         + ".tiff"))))
                or self.private_path and (
                    Path(self.private_path).parent != Path(self.cwd)
                    or Path(self.private_path).name
                    != Path(self.request.final_path).name)):
            raise ValueError("mask result is invalid")
        if proof is not None:
            proof.__post_init__()
            if not self.private_path or proof.state.path != self.private_path:
                raise ValueError("mask result proof is inexact")
        if self.published:
            if (proof is None or not _state_valid(
                    final, path=self.request.final_path,
                    size_limit=_MASK_LIMIT, nonempty=True)
                    or (final.device, final.inode, final.size, final.mtime_ns)
                    != (proof.state.device, proof.state.inode,
                        proof.state.size, proof.state.mtime_ns)
                    or proof.state.ctime_ns > final.ctime_ns):
                raise ValueError("mask result publication is inexact")
        elif final is not None:
            raise ValueError("unpublished mask result has a final state")
        if (self.recovery_class == "qualified-private-candidate"
                and (proof is None or self.recovery_path != self.private_path)
                or self.recovery_class == "unqualified-private-stage"
                and self.recovery_path != self.cwd):
            raise ValueError("mask result recovery custody is invalid")


def mask_terminal_result_valid(
    terminal: object, request: MaskRequest,
) -> bool:
    """Validate the exact Mask terminal/payload matrix at a trust boundary."""

    try:
        if (type(terminal) is not OperationTerminal
                or terminal.payload is None
                or type(terminal.payload) is not MaskResult
                or terminal.payload.request is not request):
            return False
        terminal.__post_init__()
        result = terminal.payload
        result.__post_init__()
        if result.diagnostic != terminal.diagnostic:
            return False
        if terminal.status is OperationTerminalStatus.RETURNED:
            return (result.published and result.exit_code == 0
                    and result.proof is not None and result.final_state is not None
                    and not result.diagnostic and not result.recovery_path)
        if terminal.status is OperationTerminalStatus.CANCELLED:
            return (not result.published and result.final_state is None
                    and not result.diagnostic and not result.recovery_path)
        if terminal.status is OperationTerminalStatus.FAILED:
            if not result.diagnostic:
                return False
            if result.published:
                return (result.exit_code == 0 and result.proof is not None
                        and result.final_state is not None
                        and bool(result.recovery_path))
            return result.final_state is None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return False
def resolve_mask_executable(fixed: str | None = None) -> str | None:
    return _resolve_authoring_executable("pyFAI-drawmask", fixed)


def _mask_source(selected: str) -> tuple[Path, str | None]:
    if type(selected) is not str or not selected.strip():
        raise ValueError("Choose a TIFF or exact HDF5/NeXus frame.")
    value = selected.strip()
    input_argument = None
    if value.startswith("silx:"):
        try:
            from silx.io.url import DataUrl
            url = DataUrl(value)
            file_path = url.file_path()
        except Exception as error:
            raise ValueError("Mask HDF URL is invalid.") from error
        if (not url.is_valid() or url.scheme() != "silx"
                or type(file_path) is not str or not file_path
                or url.data_path() is None):
            raise ValueError(
                "Mask HDF input must select one exact dataset/frame.")
        source = Path(file_path).expanduser()
        input_argument = url
    else:
        source = Path(value).expanduser()
    try:
        source = source.resolve(strict=True)
        raw = source.lstat()
    except OSError as error:
        raise ValueError("Mask input is unavailable.") from error
    if (stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode)
            or not os.access(source, os.R_OK)):
        raise ValueError("Mask input must be a readable regular file.")
    suffix = source.suffix.casefold()
    if suffix in {".tif", ".tiff"}:
        if input_argument is not None:
            raise ValueError("TIFF mask input must be selected as a file.")
    elif suffix in _HDF5_SUFFIXES:
        if input_argument is None:
            raise ValueError(
                "Mask HDF5/NeXus input must select one exact dataset/frame.")
        try:
            input_argument = type(input_argument)(
                scheme="silx", file_path=str(source),
                data_path=input_argument.data_path(),
                data_slice=input_argument.data_slice(),
            ).path()
        except Exception as error:
            raise ValueError("Mask HDF URL is invalid.") from error
        if not _bounded_text(input_argument, _URL_BYTES_LIMIT):
            raise ValueError("Mask HDF URL exceeds its encoded-byte cap.")
    else:
        raise ValueError(
            "Mask input suffix must identify TIFF, HDF5, or NeXus.")
    return source, input_argument


def prepare_mask_request(selected: str, *, current_poni: str = "", current_mask: str = "") -> MaskRequest:
    source, exact_hdf_url = _mask_source(selected)
    target_path = ""
    target_state = None
    if exact_hdf_url is None:
        source_state = SourceFileState.capture(source)
    else:
        (source_state, target_path, target_state, _shape, _dtype,
         _array, _digest) = _mask_hdf_frame(
            source, exact_hdf_url, read=False,
        )
    final = Path(os.path.splitext(str(source))[0] + "-mask.edf")
    key = os.path.normcase(os.path.normpath(str(final)))
    if current_poni and key == os.path.normcase(os.path.normpath(str(Path(current_poni).expanduser().resolve(strict=False)))):
        raise ValueError("Mask output must differ from the current calibration.")
    executable = resolve_mask_executable()
    if executable is None: raise ValueError("pyFAI-drawmask is unavailable on PATH.")
    return MaskRequest(
        str(source), str(final), executable,
        SourceFileState.capture(Path(executable)), source_state,
        str(source.parent), _directory_identity(source.parent), exact_hdf_url,
        target_path, target_state,
    )


def authoring_source_context_current(
    request: CalibrationRequest | MaskRequest,
) -> bool:
    """Revalidate the exact authoring source object and its lexical parent."""

    try:
        request.__post_init__()
        if type(request) is CalibrationRequest:
            directory = Path(request.monitored_directory)
        elif type(request) is MaskRequest:
            directory = Path(request.source_directory)
        else:
            return False
        current = _source_context_current(
            Path(request.source_path), request.source_state,
            directory, request.directory_identity,
        )
        if (not current or type(request) is not MaskRequest
                or request.exact_hdf_url is None):
            return current
        observed = _mask_hdf_frame(
            Path(request.source_path), request.exact_hdf_url, read=False,
            expected_source_state=request.source_state,
            expected_target_path=request.hdf_target_path,
            expected_target_state=request.hdf_target_state,
        )
        return (observed[0] == request.source_state
                and observed[1] == request.hdf_target_path
                and observed[2] == request.hdf_target_state)
    except (AttributeError, OSError, TypeError, ValueError):
        return False
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
        if not _WINDOWS:
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
            or digest.hexdigest() != staged_digest or not _private_mode(private.lstat().st_mode, 0o600)):
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


def _stage_hdf_tiff(
    request: MaskRequest, private: Path,
) -> tuple[
    SourceFileState, str, SourceFileState, str, str,
    tuple[int, int], str,
]:
    source = Path(request.source_path)
    observed = _mask_hdf_frame(
        source, request.exact_hdf_url, read=True,
        expected_source_state=request.source_state,
        expected_target_path=request.hdf_target_path,
        expected_target_state=request.hdf_target_state,
    )
    source_state, target_path, target_state, shape, dtype, array, digest = observed
    if (source_state != request.source_state
            or target_path != request.hdf_target_path
            or target_state != request.hdf_target_state
            or array is None or not digest):
        raise ValueError("Mask HDF frame custody changed before staging.")
    image = TifImage(data=array)
    try:
        image.write(str(private))
    except BaseException:
        try:
            private.unlink()
        except OSError:
            pass
        raise
    finally:
        image.close()
    descriptor = os.open(
        private, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not _WINDOWS:
            os.fchmod(descriptor, 0o600)
        raw = os.fstat(descriptor)
        if (not stat.S_ISREG(raw.st_mode) or raw.st_size < 1
                or raw.st_size > _TIFF_LIMIT
                or not _private_mode(raw.st_mode, 0o600)):
            raise ValueError("staged HDF frame TIFF is outside its envelope")
    finally:
        os.close(descriptor)
    staged = SourceFileState.capture(private)
    staged_shape, staged_dtype = _qualify_tiff(private, staged)
    staged_array = np.asarray(
        read_image(private, preserve_dtype=True, exact_frame=True),
    )
    staged_array = np.ascontiguousarray(staged_array)
    staged_frame_digest = hashlib.sha256(
        memoryview(staged_array).cast("B"),
    ).hexdigest()
    if (staged_shape != shape or staged_dtype != dtype
            or tuple(staged_array.shape) != shape
            or staged_array.dtype.str != dtype
            or staged_frame_digest != digest):
        raise ValueError("staged TIFF does not preserve the exact HDF frame")
    staged_file_digest, staged_count = _asset_digest(private, _TIFF_LIMIT)
    if (staged_count != staged.size
            or not _private_mode(private.lstat().st_mode, 0o600)):
        raise ValueError("staged HDF frame TIFF changed during qualification")
    return (
        source_state, digest, staged, digest, staged_file_digest,
        shape, dtype,
    )


def _mask_source_frame_current(
    request: MaskRequest, source_state: SourceFileState,
    source_sha: str, shape: tuple[int, int], source_dtype: str,
) -> bool:
    source = Path(request.source_path)
    if request.exact_hdf_url is None:
        return (authoring_source_context_current(request)
                and source_state == request.source_state
                and _unchanged(source, source_state, source_sha, _TIFF_LIMIT))
    try:
        observed = _mask_hdf_frame(
            source, request.exact_hdf_url, read=True,
            expected_source_state=request.source_state,
            expected_target_path=request.hdf_target_path,
            expected_target_state=request.hdf_target_state,
        )
        return (observed[0] == source_state == request.source_state
                and observed[1] == request.hdf_target_path
                and observed[2] == request.hdf_target_state
                and observed[3] == shape and observed[4] == source_dtype
                and observed[6] == source_sha)
    except (OSError, TypeError, ValueError):
        return False


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
    return MaskProof(before, shape, source_dtype, array.dtype.str, source_sha, staged_sha, digest, _MASK_COERCION_POLICY)
def _mask_matches(proof: MaskProof, path: Path) -> SourceFileState | None:
    try:
        proof.__post_init__()
        raw = path.lstat()
        if stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode) or not _private_mode(raw.st_mode, 0o600): return None
        state = SourceFileState.capture(path); latest = path.lstat()
        same = (stat.S_ISREG(latest.st_mode) and _private_mode(latest.st_mode, 0o600) and (latest.st_dev, latest.st_ino, latest.st_size) == (state.device, state.inode, state.size) and (state.device, state.inode, state.size) == (proof.state.device, proof.state.inode, proof.state.size))
        if same:
            digest = _asset_digest(path, _MASK_LIMIT)[0]; after = SourceFileState.capture(path); latest = path.lstat()
            same = digest == proof.mask_sha256 and after == state and stat.S_ISREG(latest.st_mode) and _private_mode(latest.st_mode, 0o600) and (latest.st_dev, latest.st_ino, latest.st_size) == (after.device, after.inode, after.size)
    except (OSError, ValueError): return None
    return state if same else None


def _open_admitted_directory(
    path: Path, identity: tuple[int, int],
) -> int | None:
    if _WINDOWS:
        if _directory_identity(path) != identity:
            raise ValueError("mask publication directory changed")
        return None
    if not _DIR_FD_PUBLICATION:
        raise OSError("descriptor-bound mask publication is unavailable")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        observed = os.fstat(descriptor)
        if (not stat.S_ISDIR(observed.st_mode)
                or (observed.st_dev, observed.st_ino) != identity):
            raise ValueError("mask publication directory changed")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _publication_state(
    directory_fd: int | None, name: str, proof: MaskProof, reported_path: str,
) -> SourceFileState:
    if _WINDOWS:
        # Capture through the pathname, like all other SourceFileState values;
        # Windows fstat and stat expose different ctime meanings.
        state = _mask_matches(proof, Path(reported_path))
        if state is None or state.mtime_ns != proof.state.mtime_ns:
            raise ValueError("published mask identity is inexact")
        return state
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        observed = os.fstat(descriptor)
        if (not stat.S_ISREG(observed.st_mode)
                or stat.S_IMODE(observed.st_mode) != 0o600
                or (observed.st_dev, observed.st_ino, observed.st_size,
                    observed.st_mtime_ns)
                != (proof.state.device, proof.state.inode, proof.state.size,
                    proof.state.mtime_ns)):
            raise ValueError("published mask identity is inexact")
        return SourceFileState(
            reported_path, int(observed.st_size), int(observed.st_mtime_ns),
            int(observed.st_ctime_ns), int(observed.st_dev),
            int(observed.st_ino),
        )
    finally:
        os.close(descriptor)


def _unlink_exact_publication(
    directory_fd: int | None, path: Path, proof: MaskProof,
) -> bool:
    try:
        state = _publication_state(directory_fd, path.name, proof, str(path))
        if (state.device, state.inode) != (proof.state.device, proof.state.inode):
            return False
        if _WINDOWS:
            path.unlink()
        else:
            os.unlink(path.name, dir_fd=directory_fd)
        return True
    except (FileNotFoundError, OSError, ValueError):
        return False


def _cleanup_at_admitted_parent(
    stage: Path | None, identity: tuple[int, int] | None,
    names: tuple[Path, ...], directory_fd: int | None,
) -> str:
    if stage is None or identity is None or directory_fd is None:
        return "" if stage is None else str(stage)
    try:
        observed = os.stat(
            stage.name, dir_fd=directory_fd, follow_symlinks=False,
        )
        if (not stat.S_ISDIR(observed.st_mode)
                or (observed.st_dev, observed.st_ino) != identity):
            return str(stage)
        for path in names:
            if path.parent != stage:
                continue
            relative = f"{stage.name}/{path.name}"
            try:
                child = os.stat(
                    relative, dir_fd=directory_fd, follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(child.st_mode):
                os.unlink(relative, dir_fd=directory_fd)
        os.rmdir(stage.name, dir_fd=directory_fd)
        return ""
    except OSError:
        return str(stage)


def _qualify_existing_mask(
    path: Path, expected_shape: tuple[int, int],
) -> ExistingMaskProof:
    if path.suffix.lower() not in {".edf", ".npy"}:
        raise ValueError("Mask files must use .edf or .npy")
    raw = path.lstat(); before = SourceFileState.capture(path)
    if stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode):
        raise ValueError("existing mask is not a regular file")
    if before.size < 1 or before.size > _MASK_LIMIT:
        raise ValueError("existing mask exceeds 64 MiB")
    if (expected_shape[0] * expected_shape[1] > _PIXEL_LIMIT
            or expected_shape[0] * expected_shape[1] * 8 > _DECODED_LIMIT):
        raise ValueError("expected mask shape exceeds the decoded array envelope")
    digest, _ = _asset_digest(path, _MASK_LIMIT)
    array = read_image(path, preserve_dtype=True, exact_frame=True)
    if (array.ndim != 2 or tuple(array.shape) != expected_shape
            or array.dtype.kind not in "biuf" or not array.dtype.isnative
            or array.dtype.itemsize > 8 or array.nbytes > _DECODED_LIMIT):
        raise ValueError("existing mask array is outside the qualified envelope")
    coerced = load_mask(array); expected = array != 0
    if array.dtype.kind == "f": expected |= np.isnan(array)
    if coerced.dtype != np.bool_ or not np.array_equal(coerced, expected):
        raise ValueError("existing mask coercion is not public truth")
    if not _unchanged(path, before, digest, _MASK_LIMIT):
        raise ValueError("existing mask changed during qualification")
    return ExistingMaskProof(
        before, expected_shape, array.dtype.str, digest,
        _MASK_COERCION_POLICY,
    )


def _existing_mask_current(proof: ExistingMaskProof, path: Path) -> bool:
    try:
        if type(proof) is not ExistingMaskProof:
            return False
        proof.__post_init__()
        return _unchanged(path, proof.state, proof.sha256, _MASK_LIMIT)
    except (TypeError, ValueError):
        return False


def _source_matches_mask_proof(
    proof: MaskProof, request: MaskRequest,
) -> bool:
    try:
        proof.__post_init__()
        request.__post_init__()
        path = Path(request.source_path)
        if request.exact_hdf_url is not None:
            return _mask_source_frame_current(
                request, request.source_state, proof.source_sha256,
                proof.shape, proof.source_dtype,
            ) and proof.source_sha256 == proof.staged_sha256
        raw = path.lstat()
        if (stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode)
                or raw.st_size < 1 or raw.st_size > _TIFF_LIMIT):
            return False
        before = SourceFileState.capture(path)
        digest, count = _asset_digest(path, _TIFF_LIMIT)
        shape, dtype = _qualify_tiff(path, before)
        repeated, repeated_count = _asset_digest(path, _TIFF_LIMIT)
        return (SourceFileState.capture(path) == before
                and count == repeated_count == before.size
                and digest == repeated == proof.source_sha256
                and proof.source_sha256 == proof.staged_sha256
                and shape == proof.shape and dtype == proof.source_dtype)
    except (OSError, TypeError, ValueError):
        return False


def validate_authored_asset(
    request: AssetValidationRequest,
) -> AssetValidationResult:
    """Qualify or revalidate one adoption candidate without mutating it."""

    request.__post_init__()
    if (request.source_request is not None
            and not authoring_source_context_current(request.source_request)):
        raise ValueError("authoring source context changed before validation")
    path = Path(request.path)
    candidate = request.candidate
    if candidate is None:
        if request.asset == "poni":
            poni = qualify_calibration_candidate(request.path)
            candidate = AuthoredAssetCandidate(
                "poni", poni.path, poni.proof, poni.proof.state,
            )
        else:
            proof = _qualify_existing_mask(path, request.expected_shape)
            candidate = AuthoredAssetCandidate(
                "mask", str(path), proof, proof.state,
            )
    elif request.asset == "poni":
        admitted = CalibrationCandidate(candidate.path, candidate.proof)
        qualified = qualify_calibration_candidate(candidate.path)
        if (candidate.state != candidate.proof.state
                or qualified.path != candidate.path
                or qualified.proof != candidate.proof
                or not calibration_candidate_current(admitted)):
            raise ValueError("PONI candidate changed before adoption")
    elif type(candidate.proof) is MaskProof:
        source_path = Path(candidate.source_path)
        matched = _mask_matches(candidate.proof, path)
        existing = (_qualify_existing_mask(path, candidate.proof.shape)
                    if matched is not None else None)
        if (matched != candidate.state
                or SourceFileState.capture(path) != candidate.state
                or existing is None
                or (existing.shape, existing.dtype, existing.sha256,
                    existing.coercion_policy)
                != (candidate.proof.shape, candidate.proof.mask_dtype,
                    candidate.proof.mask_sha256,
                    candidate.proof.coercion_policy)
                or type(request.source_request) is not MaskRequest
                or request.source_request.source_path != str(source_path)
                or not _source_matches_mask_proof(
                    candidate.proof, request.source_request)):
            raise ValueError("generated mask changed before adoption")
    elif (type(candidate.proof) is not ExistingMaskProof
            or candidate.state != candidate.proof.state
            or not _existing_mask_current(candidate.proof, path)):
        raise ValueError("existing mask changed before adoption")
    if (request.source_request is not None
            and not authoring_source_context_current(request.source_request)):
        raise ValueError("authoring source context changed during validation")
    result = AssetValidationResult(request, candidate)
    result.__post_init__()
    return result
def run_mask(request: MaskRequest, identity: OperationIdentity, cancelled: object, publish: Callable[[str, int, int], None], seal: Callable[[OperationIdentity], bool]) -> OperationTerminal:
    stage = private_source = private_mask = publication_alias = None; stage_identity = None
    proof = final_state = None; code = None; argv: tuple[str, ...] = (); published = False; status, diagnostic = OperationTerminalStatus.RETURNED, ""
    publication_fd = None; publication_linked = False
    try:
        request.__post_init__(); source, final = Path(request.source_path), Path(request.final_path)
        if (resolve_mask_executable(request.executable) != request.executable
                or SourceFileState.capture(Path(request.executable))
                != request.executable_state
                or not authoring_source_context_current(request)):
            raise ValueError("mask source context changed before launch")
        publication_fd = _open_admitted_directory(
            Path(request.source_directory), request.directory_identity,
        )
        if cancelled.is_set(): raise _Cancelled("mask cancelled")
        stage = Path(tempfile.mkdtemp(prefix=".xdart-mask-", dir=final.parent)); created = stage.lstat(); stage_identity = (created.st_dev, created.st_ino)
        os.chmod(stage, 0o700); stage_stat = stage.lstat()
        if (stage_stat.st_dev, stage_stat.st_ino) != stage_identity or not stat.S_ISDIR(stage_stat.st_mode) or not _private_mode(stage_stat.st_mode, 0o700): raise OSError("private mask stage is not mode 0700")
        _probe_links(stage)
        private_source = stage / (
            source.name if request.exact_hdf_url is None
            else source.stem + ".tiff"
        )
        private_mask = stage / (source.stem + "-mask.edf")
        publish("copy", 1, 4)
        if request.exact_hdf_url is None:
            source_state, source_sha, staged_state, staged_sha = _copy_tiff(
                source, private_source,
            )
            staged_file_sha = staged_sha
            shape, source_dtype = _qualify_tiff(private_source, staged_state)
        else:
            (source_state, source_sha, staged_state, staged_sha,
             staged_file_sha, shape, source_dtype) = _stage_hdf_tiff(
                request, private_source,
            )
        if (source_state != request.source_state
                or not authoring_source_context_current(request)):
            raise ValueError("mask source context changed during copy")
        if (not _mask_source_frame_current(
                    request, source_state, source_sha, shape, source_dtype)
                or not _unchanged(
                    private_source, staged_state, staged_file_sha,
                    _TIFF_LIMIT)):
            raise ValueError("mask source changed before child launch")
        if cancelled.is_set(): raise _Cancelled("mask cancelled")
        publish("launch", 2, 4); argv = (request.executable, str(private_source))
        if (resolve_mask_executable(request.executable) != request.executable
                or SourceFileState.capture(Path(request.executable))
                != request.executable_state):
            raise ValueError("mask executable changed immediately before launch")
        with _private_child_stderr(stage) as child_stderr:
            options = dict(
                cwd=str(stage), shell=False, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=child_stderr,
                close_fds=True,
            )
            options["creationflags" if _WINDOWS else "start_new_session"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if _WINDOWS else True)
            process = _popen(argv, **options)
            code, process_diagnostic, stderr_truncated = _wait_child(
                process, cancelled, child_stderr,
            )
            child_diagnostic = _bounded_child_stderr(
                child_stderr, truncated=stderr_truncated,
            )
            if process_diagnostic:
                raise RuntimeError(
                    f"child process control failed: {process_diagnostic}"
                    + (f": {child_diagnostic}" if child_diagnostic else ""))
            if cancelled.is_set():
                raise _Cancelled("mask cancelled")
            if code != 0:
                raise _nonzero_child_error(
                    "pyFAI-drawmask", code, child_diagnostic)
        if (not _mask_source_frame_current(
                    request, source_state, source_sha, shape, source_dtype)
                or not _unchanged(
                    private_source, staged_state, staged_file_sha,
                    _TIFF_LIMIT)):
            raise ValueError("mask source changed during child execution")
        try:
            private_mask.lstat()
        except FileNotFoundError:
            raise _Cancelled("mask editor closed without saving")
        publish("qualify", 3, 4); proof = _qualify_mask(private_mask, shape, source_dtype, source_sha, staged_sha)
        if (not authoring_source_context_current(request)
                or not _unchanged(private_mask, proof.state, proof.mask_sha256, _MASK_LIMIT)): raise ValueError("qualified mask changed before publication")
        if not seal(identity): raise _Cancelled("mask cancelled before publication")
        _tighten_qualified(private_mask, proof, _mask_matches)
        publish("publish", 4, 4); link_error = None
        # Keep the old mask until the edited replacement is qualified. Retain
        # the private proof file while atomically publishing a second link.
        publication_alias = stage / ".publish-mask.edf"
        try:
            if _WINDOWS:
                _link(private_mask, publication_alias)
                os.replace(publication_alias, final)
            else:
                relative_alias = str(Path(stage.name) / publication_alias.name)
                _link(
                    str(Path(stage.name) / private_mask.name), relative_alias,
                    src_dir_fd=publication_fd, dst_dir_fd=publication_fd,
                    follow_symlinks=False,
                )
                os.replace(relative_alias, final.name,
                           src_dir_fd=publication_fd, dst_dir_fd=publication_fd)
            publication_linked = True
        except OSError as error: link_error = error
        final_state = (_publication_state(
            publication_fd, final.name, proof, request.final_path,
        ) if publication_linked else None)
        if (not authoring_source_context_current(request)
                or final_state is None
                or _mask_matches(proof, private_mask) is None):
            raise RuntimeError("mask publication could not be reconciled") from link_error
        published = True
    except _Cancelled: status = OperationTerminalStatus.CANCELLED
    except BaseException as error:
        status = OperationTerminalStatus.FAILED; module, name, message = detached_exception_strings(error); diagnostic = f"{module}.{name}: {message}"
    if publication_linked and not published and proof is not None:
        if _unlink_exact_publication(
                publication_fd, Path(request.final_path), proof):
            publication_linked = False
        final_state = None
    names = (() if stage is None else tuple(stage / name for name in (".link-probe-source", ".link-probe-linked", ".link-probe-occupied")))
    owned = tuple(path for path in (private_source, private_mask, publication_alias) if path is not None)
    recovery = _cleanup(stage, stage_identity, names + owned)
    if recovery and not authoring_source_context_current(request):
        recovery = _cleanup_at_admitted_parent(
            stage, stage_identity, names + owned, publication_fd,
        )
    recovery_class = ""
    if recovery:
        status = OperationTerminalStatus.FAILED; recovery_class = "qualified-private-candidate" if proof is not None and private_mask is not None and _mask_matches(proof, private_mask) is not None else "unqualified-private-stage"
        recovery = str(private_mask if recovery_class.startswith("qualified") else stage)
        diagnostic += ("; " if diagnostic else "") + f"private mask custody relinquished ({recovery_class}): {recovery}"
    if published and proof is not None:
        try:
            reconciled = _publication_state(
                publication_fd, Path(request.final_path).name, proof,
                request.final_path,
            )
        except (OSError, ValueError):
            reconciled = None
        if (reconciled is None
                or not authoring_source_context_current(request)):
            status = OperationTerminalStatus.FAILED
            diagnostic += ("; " if diagnostic else "") + (
                "published mask changed during terminal cleanup"
            )
            if _unlink_exact_publication(
                    publication_fd, Path(request.final_path), proof):
                publication_linked = False
                published = False
                final_state = None
        else:
            final_state = reconciled
    if publication_fd is not None:
        try:
            os.close(publication_fd)
        except OSError:
            pass
    result = MaskResult(request, "" if private_mask is None else str(private_mask), proof, final_state, code, argv,
        "" if stage is None else str(stage), published, recovery, recovery_class, diagnostic)
    return OperationTerminal(identity, status, diagnostic, result)
__all__ = [
    "AssetValidationRequest", "AssetValidationResult",
    "AuthoredAssetCandidate", "CalibrationCandidate",
    "CalibrationFileProof", "CalibrationRequest", "CalibrationResult",
    "ExistingMaskProof", "MaskProof", "MaskRequest", "MaskResult",
    "authoring_source_context_current",
    "calibration_candidate_current", "qualify_calibration_candidate",
    "mask_terminal_result_valid",
    "prepare_calibration_request", "resolve_calibration_executable",
    "prepare_mask_request", "resolve_mask_executable", "run_calibration",
    "run_mask", "validate_authored_asset",
]
