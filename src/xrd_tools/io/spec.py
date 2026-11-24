"""SPEC file parsing: scan metadata, energy, UB, angles."""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import logging
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from silx.io.specfile import SpecFile

logger = logging.getLogger(__name__)


# ---- one immutable view of each SPEC file within a scope (NEXT-1219-SPEC-OCCURRENCE) ------------------------------
#: Within :func:`spec_capture_scope`, the resolved SPEC path -> its _SpecCapture.
_SPEC_CAPTURES = contextvars.ContextVar("xdart_spec_captures", default=None)
_CAPTURE_CHUNK = 1 << 20


class SpecCaptureFailed(OSError):
    """A SPEC file could not be captured as one immutable view (missing, not a regular file, replaced or truncated
    while it was read), or its private copy could not be closed or removed."""


@dataclass(frozen=True, slots=True)
class SpecFileView:
    """The SPEC file a scope read: its spelling, resolved path, filesystem object, the size it was captured at, and
    the SHA-256 of those bytes. In memory only."""

    lexical: str
    resolved: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str
    #: the captured descriptor's revision, as ``scan_operations`` records a file: (mode, device, inode, size, mtime,
    #: identity ctime)
    revision: tuple[int, int, int, int, int, int]


class _SpecCapture:
    """One SPEC file read once through one descriptor up to its size then, copied to a private file in the system's
    temporary folder, and parsed there by one silx object that this capture owns and closes."""

    def __init__(self, lexical: str, resolved: str):
        self.scratch: Path | None = None
        self._parser = None
        descriptor = None
        try:
            descriptor = os.open(lexical, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise SpecCaptureFailed(f"SPEC file {lexical} is not a regular file")
            named = os.stat(resolved)
            if (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino):
                raise SpecCaptureFailed(f"SPEC file {lexical} was replaced while it was opened")
            digest = hashlib.sha256()
            handle = tempfile.NamedTemporaryFile(prefix="xdart-spec-", delete=False)
            self.scratch = Path(handle.name)
            with handle:
                remaining = before.st_size
                while remaining > 0:
                    chunk = os.read(descriptor, min(_CAPTURE_CHUNK, remaining))
                    if not chunk:
                        raise SpecCaptureFailed(f"SPEC file {lexical} became shorter while it was read")
                    handle.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            from xrd_tools.io.stat_identity import identity_ctime_ns
            self.view = SpecFileView(
                lexical, resolved, before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                digest.hexdigest(), (before.st_mode, before.st_dev, before.st_ino, before.st_size,
                                     before.st_mtime_ns, identity_ctime_ns(before.st_ctime_ns)))
        except BaseException:
            self.close()
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @property
    def parser(self):
        if self._parser is None:
            self._parser = SpecFile(str(self.scratch))
        return self._parser

    def close(self) -> BaseException | None:
        failure = None
        for action in (None if self._parser is None else self._parser.close,
                       None if self.scratch is None else self._unlink):
            if action is not None:
                try:
                    action()
                except BaseException as error:  # reported once, after every step ran
                    failure = failure or error
        self._parser = None
        return failure

    def _unlink(self):
        scratch, self.scratch = self.scratch, None
        scratch.unlink(missing_ok=True)


def _capture_key(spec_file: Path | str) -> tuple[str, str]:
    lexical = os.path.abspath(os.fspath(spec_file))
    return lexical, os.path.normcase(os.path.realpath(lexical))


@contextlib.contextmanager
def spec_capture_scope():
    """Within this block, each SPEC file this module's readers open is captured once, on first use, as one
    immutable view (one descriptor, the size it had then), and every reader here parses that view: a preparation
    reads each file once however many scans, members or UB lookups it has. Nested blocks share the outermost
    one's captures. On leaving the outermost block every private copy is closed and removed."""
    captures = _SPEC_CAPTURES.get()
    if captures is not None:
        yield
        return
    captures = {}
    token = _SPEC_CAPTURES.set(captures)
    primary = None
    try:
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        _SPEC_CAPTURES.reset(token)
        failure = None
        for capture in captures.values():
            failure = capture.close() or failure
        if failure is not None and primary is None:
            raise SpecCaptureFailed(f"a SPEC capture could not be closed or removed: {failure}") from failure
        if failure is not None:
            logger.warning("SPEC capture cleanup failed after %r: %s", primary, failure)


def spec_file_view(spec_file: Path | str) -> SpecFileView | None:
    """The view the current scope captured of ``spec_file`` (capturing it now), or None outside a scope."""
    capture = _scoped_capture(spec_file)
    return None if capture is None else capture.view


def _scoped_capture(spec_file: Path | str) -> _SpecCapture | None:
    captures = _SPEC_CAPTURES.get()
    if captures is None:
        return None
    lexical, key = _capture_key(spec_file)
    capture = captures.get(key)
    if capture is None:
        capture = captures[key] = _SpecCapture(lexical, key)
    return capture


def _spec_parser(spec_file: Path | str):
    """The silx parser of ``spec_file``: the scope's capture, else a new parser of the file as it is now."""
    capture = _scoped_capture(spec_file)
    return SpecFile(str(spec_file)) if capture is None else capture.parser


def _spec_text_path(spec_file: Path | str) -> Path:
    """The bytes the strict row validator reads: the scope's capture, else the file itself."""
    capture = _scoped_capture(spec_file)
    return Path(spec_file) if capture is None else capture.scratch


def _get_spec_scan(spec_file: Path | str, scan_num: str):
    return _spec_parser(spec_file)[scan_num]


#: SPEC control lines the first non-empty line of a SPEC file starts with.
_SPEC_MARKERS = ("#F", "#S", "#E", "#C", "#O")


def is_spec_file(path: Path | str) -> bool:
    """True if *path* looks like a SPEC data file.

    SSRL SPEC files are **extensionless**, so detection is content-based: the
    first non-empty line of a SPEC file is a control line (``#F`` / ``#S`` / …).
    Returns False for a directory, a missing file, or a binary file."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with p.open("r", errors="ignore") as fh:
            for _ in range(50):
                line = fh.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped:
                    return stripped.startswith(_SPEC_MARKERS)
    except OSError:
        return False
    return False


def list_spec_scans(spec_file: Path | str) -> list[str]:
    """The scan keys (``"1.1"``, ``"2.1"``, …) in a SPEC file, in file order."""
    return list(_spec_parser(spec_file).keys())


def read_spec_scan_motor_names(
    spec_file: Path | str, scan_num: str | None = None
) -> tuple[str, ...]:
    """One scan's ``#O`` motor names from its header alone (its data table is not
    read); ``scan_num`` None: the file's first scan. KeyError when there is no such scan."""
    spec = _spec_parser(spec_file)
    keys = spec.keys()
    if scan_num is None and not len(keys):
        raise KeyError("the SPEC file has no scans")
    key = keys[0] if scan_num is None else scan_num
    return tuple(str(name) for name in spec[key].motor_names)


def read_spec_scan_counter_names(
    spec_file: Path | str, scan_num: str | None = None
) -> tuple[str, ...]:
    """One scan's ``#L`` columns that are not its ``#O`` motors (its counters, the beam monitors among
    them), from its header alone; ``scan_num`` None: the file's first scan. KeyError when there is no
    such scan."""
    spec = _spec_parser(spec_file)
    keys = spec.keys()
    if scan_num is None and not len(keys):
        raise KeyError("the SPEC file has no scans")
    scan = spec[keys[0] if scan_num is None else scan_num]
    motors = {str(name) for name in scan.motor_names}
    return tuple(str(label) for label in scan.labels if str(label) not in motors)


def count_spec_scan_points(
    spec_file: Path | str, scans: tuple[str, ...] | None = None
) -> dict[str, int | None]:
    """Each scan's point count (its data lines), keyed as asked: ``scans`` as a Stitch selection names
    them (``N`` is SPEC's ``N.1`` unless the file has an exact ``N.M`` key), None when the file lacks it;
    ``scans`` None: every scan by its ``N.M`` key. About half a millisecond a scan (silx's parser)."""
    spec = _spec_parser(spec_file)
    keys = spec.keys()
    wanted = {key: key for key in keys} if scans is None else {
        scan: scan if scan in keys else f"{scan}.1" for scan in scans}
    counts: dict[str, int | None] = {}
    for scan, key in wanted.items():
        if key not in keys:
            counts[scan] = None
            continue
        data = np.asarray(spec[key].data)
        counts[scan] = int(data.shape[1]) if data.ndim == 2 else 0
    return counts


def read_spec_scan_table(
    spec_file: Path | str, scan_num: str
) -> tuple[dict[str, np.ndarray], dict[str, float], int]:
    """Read ONE scan into ``(columns, motors, npts)``.

    ``columns`` maps each per-point ``#L`` label (the scanned motor + counters)
    to a length-``npts`` array; ``motors`` maps each ``#O``/``#P`` motor to its
    constant scan-start position.  The complete per-frame metadata for a SPEC
    scan, with no column pre-selection (cf. :func:`get_from_spec_file`)."""
    physical_columns, motor_positions, npts = read_spec_scan_columns(
        spec_file, scan_num
    )
    columns: dict[str, np.ndarray] = {}
    for name, values in physical_columns:
        # This legacy mapping cannot express duplicate physical #L names.
        # Preserve its historical first-occurrence behavior; occurrence-aware
        # consumers use read_spec_scan_columns instead.
        columns.setdefault(name, values)
    return columns, dict(motor_positions), npts


class UnusableSpecScan(ValueError):
    """One SPEC scan cannot be read as a grid of points (NEXT-1210-RESPONSIVENESS C-10): no data points (an aborted
    scan), a malformed data table, or detector images that do not match its points. All Scans and Directory skip
    such a scan and list ``reason``; an explicitly selected one is refused with the message."""

    def __init__(self, message: str, reason: str | None = None):
        super().__init__(message)
        self.reason = reason or message


_SPEC_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _spec_row_problem(tokens, labels):
    """The first token of one physical data row that is not a complete number of its ``#L`` column, as (column
    index, label, token), or None: a missing or an extra token is one too. The selected-row rule every SPEC reader
    shares (the strict scan validator below and the public metadata reader's bounded point scan)."""
    for index in range(max(len(tokens), len(labels))):
        token = tokens[index] if index < len(tokens) else None
        label = labels[index] if index < len(labels) else "<extra>"
        if index >= len(labels) or token is None or not _SPEC_NUMBER.fullmatch(token):
            return index, label, token
    return None


def _validate_spec_numeric_tokens(spec_file, scan_data, labels):
    """Guard silx's permissive conversion using the selected scan's physical rows.

    silx remains the format/value reader. It can extract digits from malformed
    tokens (``np.float64(1.0)`` becomes ``0.641``), so validate complete decimal
    tokens before requesting its array. Header comments and MCA records, including
    their backslash continuations, are not table rows.
    """
    scan_index = -1
    in_table = False
    mca_continuation = False
    with _spec_text_path(spec_file).open(encoding="utf-8", errors="replace") as stream:
        for physical_row, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if raw_line.startswith("#S") and raw_line[2:3].isspace():
                scan_index += 1
                if scan_index > scan_data.index:
                    break
                in_table = False
            if scan_index != scan_data.index:
                continue
            if line.startswith("#L"):
                in_table = True
            if not in_table or not line or line.startswith("#"):
                continue
            if mca_continuation or line.startswith("@"):
                mca_continuation = line.endswith("\\")
                continue
            problem = _spec_row_problem(line.split(), labels)
            if problem is not None:
                index, label, token = problem
                raise UnusableSpecScan(
                    f"SPEC {spec_file}, scan {scan_data.number}.{scan_data.order}, "
                    f"physical row {physical_row}, column {index + 1} ({label!r}): "
                    f"expected a complete decimal/scientific numeric token, got {token!r}"
                )


def read_spec_scan_columns(
    spec_file: Path | str, scan_num: str
) -> tuple[
    tuple[tuple[str, np.ndarray], ...],
    tuple[tuple[str, float], ...],
    int,
]:
    """Read one scan without collapsing physical ``#L`` occurrences.

    The first result is ordered exactly like the physical SPEC table and may
    therefore contain repeated names (for example two distinct ``Seconds``
    columns).  The second result retains the ``#O``/``#P`` motor order.  This
    occurrence-aware form is the source of truth for exact analysis preflight;
    :func:`read_spec_scan_table` remains the compatibility mapping view.
    """

    scan_data = _get_spec_scan(spec_file, scan_num)
    labels = tuple(str(label) for label in scan_data.labels)
    _validate_spec_numeric_tokens(spec_file, scan_data, labels)
    data = np.asarray(scan_data.data)
    if data.ndim != 2 or data.shape[0] != len(labels):
        if data.size == 0:  # #L written, no point recorded: the scan was aborted (1.2.9 said "labels do not match")
            reason = "no data points (an aborted scan)"
            raise UnusableSpecScan(f"SPEC scan {scan_data.number}.{scan_data.order}: {reason}", reason)
        raise UnusableSpecScan("SPEC #L labels do not match the physical data table")
    npts = int(data.shape[1])
    columns = tuple(
        (name, np.array(data[index], dtype=float, copy=True))
        for index, name in enumerate(labels)
    )
    motors: list[tuple[str, float]] = []
    for motor in scan_data.motor_names:
        try:
            motors.append(
                (str(motor), float(scan_data.motor_position_by_name(motor)))
            )
        except Exception:
            logger.debug("read_spec_scan_columns: motor %r unreadable", motor)
    return columns, tuple(motors), npts


def get_scan_path_info(scan: str) -> tuple[str, str]:
    """
    Parse scan name into sample name and SPEC-compatible scan number.
    """
    idx = scan.rfind("_")
    if idx < 0:
        return scan, "1.1"
    sample_name = scan[:idx]
    scan_num = scan[idx + 1 :]
    return sample_name, f"{scan_num}.1"


def get_energy_and_UB(spec_file: Path | str, scan_num: str) -> tuple[float, np.ndarray]:
    scan_data = _get_spec_scan(spec_file, scan_num)
    energy = scan_data.motor_position_by_name("energy")
    header_dict = scan_data.scan_header_dict
    UB = np.array(header_dict["G3"].split(), dtype=float).reshape(3, 3)
    return energy, UB


def get_UB(spec_file: Path | str, scan_num: str) -> np.ndarray:
    """Return a scan's ``#G3`` UB matrix (3 x 3) without requiring an energy motor. A scan with no ``#G3``
    line raises ``KeyError("G3")``; a scan the file does not hold raises silx's own ``KeyError``."""

    header_dict = _get_spec_scan(spec_file, scan_num).scan_header_dict
    return np.array(header_dict["G3"].split(), dtype=float).reshape(3, 3)


def get_orientation_header(spec_file: Path | str, scan_num: str) -> np.ndarray:
    """Return a scan's ``#G1`` values: the lattice, the reciprocal lattice, and the two orientation reflections
    (their HKL, angles and wavelengths) SPEC computed its ``#G3`` UB from. A scan with no ``#G1`` line raises
    ``KeyError("G1")``; a scan the file does not hold raises silx's own ``KeyError``."""

    header_dict = _get_spec_scan(spec_file, scan_num).scan_header_dict
    return np.array(header_dict["G1"].split(), dtype=float)


def get_energy(spec_file: Path | str, scan_num: str) -> float:
    """Return scan energy without requiring or parsing a ``#G3`` UB header."""

    scan_data = _get_spec_scan(spec_file, scan_num)
    return float(scan_data.motor_position_by_name("energy"))


def get_spec_scan_type(spec_file: Path | str, scan_num: str) -> str | list[str]:
    scan_data = _get_spec_scan(spec_file, scan_num)
    scan_hdr = scan_data.scan_header_dict["S"].split()
    x_col = scan_hdr[1]

    if x_col == "hklscan":
        hkl_ranges = np.array(
            [
                float(scan_hdr[3]) - float(scan_hdr[2]),
                float(scan_hdr[5]) - float(scan_hdr[4]),
                float(scan_hdr[7]) - float(scan_hdr[6]),
            ]
        )
        x_col = [x for x, r in zip(("H", "K", "L"), hkl_ranges != 0) if r]

    return x_col


def get_spec_scanned_axes(
    spec_file: Path | str,
    scan_num: str,
) -> tuple[str, ...]:
    """Return the per-point axes declared by a SPEC scan command.

    This is deliberately narrower than the scan's ``#L`` columns: those also
    contain counters, timers, and detector signals and therefore cannot be
    used to infer the default X axis by numeric spread.  Axis order follows the
    command, which matters for multi-axis scans.  Unknown/non-positional scan
    commands return an empty tuple so callers can explicitly fall back to
    frame index.
    """
    scan_data = _get_spec_scan(spec_file, scan_num)
    tokens = scan_data.scan_header_dict.get("S", "").split()
    if len(tokens) < 2:
        return ()

    command = tokens[1].lower()
    candidates: list[str] = []

    # ascan/dscan and their multi-motor forms (a2scan, d3scan, ...):
    # each motor contributes ``name start stop`` before the shared tail.
    match = re.fullmatch(r"[ad](\d*)scan", command)
    if match:
        count = int(match.group(1) or "1")
        candidates = [
            tokens[2 + 3 * i]
            for i in range(count)
            if 2 + 3 * i < len(tokens)
        ]
    elif command in {"mesh", "dmesh"}:
        # mesh m1 start stop intervals1 m2 start stop intervals2 time
        candidates = [tokens[i] for i in (2, 6) if i < len(tokens)]
    elif command == "hklscan" and len(tokens) >= 8:
        ranges = (
            float(tokens[3]) - float(tokens[2]),
            float(tokens[5]) - float(tokens[4]),
            float(tokens[7]) - float(tokens[6]),
        )
        candidates = [
            axis for axis, span in zip(("H", "K", "L"), ranges) if span != 0
        ]

    # Preserve the spelling used by #L and reject command arguments that are
    # not actual per-point columns in this scan.
    labels = {str(label).casefold(): str(label) for label in scan_data.labels}
    out: list[str] = []
    for candidate in candidates:
        label = labels.get(str(candidate).casefold())
        if label is not None and label not in out:
            out.append(label)
    return tuple(out)


def get_from_spec_file(
    fname: Path | str,
    scan_number: str,
    cols: list[str],
) -> dict[str, np.ndarray]:
    columns, motors, npts = read_spec_scan_table(fname, scan_number)
    spec_data: dict[str, np.ndarray] = {}

    for col in cols:
        if col in columns:
            spec_data[col] = columns[col]
        elif col in motors:
            spec_data[col] = np.full(npts, motors[col])
        else:
            logger.warning(
                "Column/motor %s does not exist in scan %s",
                col,
                scan_number,
            )

    return spec_data


def get_angles(
    spec_file: Path | str,
    scan_num: str,
    diff_motors: list[str] | tuple[str, ...],
) -> list[list[float]]:
    columns, motors, npts = read_spec_scan_table(spec_file, scan_num)
    angles: list[list[float]] = []

    for motor in diff_motors:
        if motor in columns:
            vals = columns[motor]
        else:
            vals = np.full(npts, motors[motor])
        angles.append(list(vals))

    return angles
