"""SPEC file parsing: scan metadata, energy, UB, angles."""
from __future__ import annotations

import codecs
import contextlib
import contextvars
import hashlib
import logging
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path

import numpy as np
from silx.io.specfile import SpecFile

logger = logging.getLogger(__name__)


# ---- one immutable view of each SPEC file within a scope (NEXT-1219-SPEC-OCCURRENCE) ------------------------------
#: Within :func:`spec_capture_scope`, the resolved SPEC path -> its _SpecCapture.
_SPEC_CAPTURES = contextvars.ContextVar("xdart_spec_captures", default=None)


class SpecCaptureFailed(OSError):
    """A SPEC file could not be captured or qualified as one immutable view."""


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
    ctime_ns: int
    sha256: str
    #: the captured descriptor's revision, as ``scan_operations`` records a file: (mode, device, inode, size, mtime,
    #: identity ctime)
    revision: tuple[int, int, int, int, int, int]


class _SpecCaptureStream:
    """Feed the structural index while writing the descriptor's fixed starting extent to the spool."""

    def __init__(self, descriptor, size, spool, lexical):
        self.descriptor = descriptor
        self.remaining = size
        self.spool = spool
        self.lexical = lexical
        self.digest = hashlib.sha256()

    def read(self, size):
        if self.remaining == 0:
            return b""
        data = os.read(self.descriptor, min(size, self.remaining))
        if not data:
            raise SpecCaptureFailed(f"SPEC file {self.lexical} became shorter while it was read")
        self.spool.write(data)
        self.digest.update(data)
        self.remaining -= len(data)
        return data


class _SpecCapture:
    """One immutable spool, its structural index, and the silx parser that reads it."""

    def __init__(self, lexical: str, resolved: str):
        from xrd_tools.io.metadata import iter_spec_occurrence_frames

        self.scratch: Path | None = None
        self._parser = None
        self._keys_checked = False
        self._strict_checked: set[str] = set()
        self._row_counts_checked: set[str] = set()
        self._commands: dict[str, tuple[str, bool]] | None = None
        descriptor = None
        captured = False
        try:
            descriptor = os.open(lexical, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise SpecCaptureFailed(f"SPEC file {lexical} is not a regular file")
            named = os.stat(resolved)
            if (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino):
                raise SpecCaptureFailed(f"SPEC file {lexical} was replaced while it was opened")
            handle = tempfile.NamedTemporaryFile(prefix="xdart-spec-", delete=False)
            self.scratch = Path(handle.name)
            with handle:
                stream = _SpecCaptureStream(descriptor, before.st_size, handle, lexical)
                self._frames = {frame.key: frame for frame in iter_spec_occurrence_frames(stream)}
                if stream.remaining:
                    raise SpecCaptureFailed(f"SPEC {lexical}: the capture did not reach its starting end")
            after = os.fstat(descriptor)
            named = os.stat(lexical)
            if ((named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
                    or os.path.normcase(os.path.realpath(lexical)) != resolved):
                raise SpecCaptureFailed(f"SPEC file {lexical} was replaced while it was read")
            if (after.st_size < before.st_size or (
                after.st_size == before.st_size
                and (after.st_mtime_ns, after.st_ctime_ns) != (before.st_mtime_ns, before.st_ctime_ns)
            )):
                raise SpecCaptureFailed(f"SPEC file {lexical} was rewritten while it was read")
            from xrd_tools.io.stat_identity import identity_ctime_ns
            self.view = SpecFileView(
                lexical, resolved, before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns, stream.digest.hexdigest(),
                (before.st_mode, before.st_dev, before.st_ino, before.st_size,
                 before.st_mtime_ns, identity_ctime_ns(before.st_ctime_ns)),
            )
            captured = True
        except BaseException:
            self.close()
            raise
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    if captured:
                        self.close()
                        raise
                    logger.exception("SPEC capture could not close the source descriptor for %s", lexical)

    @property
    def parser(self):
        if self._parser is None:
            self._parser = SpecFile(str(self.scratch))
        return self._parser

    def frame(self, key):
        if not self._keys_checked:
            indexed = [frame.key for frame in self._frames.values() if frame.start_complete]
            keys = list(self.parser.keys())
            # silx may ignore an unfinished final #S or list it with a garbage scan number.
            unfinished = len(indexed) < len(self._frames)
            if not (keys == indexed or (
                unfinished and len(keys) == len(indexed) + 1 and keys[:-1] == indexed
            )):
                raise SpecCaptureFailed(
                    f"SPEC {self.view.lexical}: the parser and the file's bytes list different scans"
                )
            self._keys_checked = True
        return self._frames.get(key)

    def check_row_count(self, frame):
        if frame.key in self._row_counts_checked:
            return
        table = np.asarray(self.parser[frame.key].data)
        rows = int(table.shape[1]) if table.ndim == 2 else 0
        if rows != frame.rows:
            raise SpecCaptureFailed(
                f"SPEC {self.view.lexical}: the parser read {rows} rows of scan {frame.key}, "
                f"its bytes hold {frame.rows}"
            )
        self._row_counts_checked.add(frame.key)

    def scan_commands(self):
        """Complete numbered headers in acquisition order from the captured bytes."""
        if self._commands is None:
            commands = {}
            with self.scratch.open("rb") as stream:
                for frame in self._frames.values():
                    if not frame.start_complete:
                        continue
                    stream.seek(frame.offset)
                    line = stream.readline(1 << 16)
                    parts = line.split(maxsplit=2)
                    if line.endswith(b"\n") and len(parts) == 3:
                        commands[frame.key] = parts[2].strip().decode("utf-8", errors="replace")
            last = next(reversed(commands), None)
            self._commands = {key: (command, key != last) for key, command in commands.items()}
        return self._commands

    def check_numeric_tokens(self, spec_file, scan_data, labels):
        key = f"{scan_data.number}.{scan_data.order}"
        if key in self._strict_checked:
            return
        frame = self.frame(key)
        if frame is None or not frame.start_complete:
            raise SpecCaptureFailed(f"SPEC {spec_file}: no complete scan header for {key}")
        with self.scratch.open("rb") as stream:
            stream.seek(frame.offset)
            lines = _indexed_spec_lines(stream, frame.block_end - frame.offset, frame.line_number)
            _validate_spec_rows(spec_file, scan_data, labels, lines)
        self._strict_checked.add(key)

    def close(self) -> None:
        if self._parser is not None:
            if not self._release("close the parser", self._parser.close):
                return
            self._parser = None
        scratch = self.scratch
        if scratch is not None and self._release("unlink the spool", lambda: scratch.unlink(missing_ok=True)):
            self.scratch = None

    def _release(self, step: str, action) -> bool:
        try:
            action()
        except BaseException:
            try:
                action()
            except BaseException:
                logger.exception("SPEC capture cleanup could not %s for private spool %s", step, self.scratch)
                return False
        return True


def _capture_key(spec_file: Path | str) -> tuple[str, str]:
    lexical = os.path.abspath(os.fspath(spec_file))
    return lexical, os.path.normcase(os.path.realpath(lexical))


@contextlib.contextmanager
def spec_capture_scope():
    """Within this block, each SPEC file this module's readers open is captured once, on first use, as one
    immutable view (one descriptor, the size it had then), and every reader here parses that view: a preparation
    reads each file once however many scans, members or UB lookups it has. Nested blocks share the outermost
    one's captures. The outermost owner retires its copies before control returns to the caller. It retries a
    failed cleanup step once and reports a remaining failure with the spool path, preserving the operation outcome."""
    captures = _SPEC_CAPTURES.get()
    if captures is not None:
        yield
        return
    captures = {}
    token = _SPEC_CAPTURES.set(captures)
    try:
        yield
    finally:
        _SPEC_CAPTURES.reset(token)
        for capture in captures.values():
            capture.close()


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


def spec_occurrence(spec_file: Path | str, key: str):
    """The occurrence proof (``xrd_tools.io.metadata.SpecOccurrence``) of scan ``key`` in the scope's captured view
    of ``spec_file``: the bytes the parser read. silx and the byte index must agree on the file's scan keys and on
    the scan's data rows, else ``SpecCaptureFailed``. None outside a scope or when the view has no such scan."""
    from xrd_tools.io.metadata import SpecOccurrence, _spec_key

    capture = _scoped_capture(spec_file)
    if capture is None:
        return None
    number, occurrence = _spec_key(key)
    frame = capture.frame(f"{number}.{occurrence}")
    if frame is None:
        return None
    lexical = os.path.abspath(os.fspath(spec_file))
    if not frame.start_complete or not frame.boundary_holds:
        raise SpecCaptureFailed(f"SPEC {lexical}: scan {frame.key} has an unqualified end")
    capture.check_row_count(frame)
    view = capture.view
    return SpecOccurrence(
        lexical, view.resolved, view.device, view.inode, frame.key, number, occurrence, frame.offset,
        frame.header, frame.header_sha256, frame.block_end, frame.block_sha256, frame.rows,
        (view.device, view.inode, view.size, view.mtime_ns, view.ctime_ns),
    )


def spec_scan_completion_header(spec_file: Path | str, key: str) -> tuple[str, bool]:
    """Return a selected command and whether a complete later numbered command follows."""
    capture = _scoped_capture(spec_file)
    if capture is None:
        raise RuntimeError("SPEC completion requires an active capture scope")
    return capture.scan_commands().get(key, ("", False))


def _spec_parser(spec_file: Path | str):
    """The silx parser of ``spec_file``: the scope's capture, else a new parser of the file as it is now."""
    capture = _scoped_capture(spec_file)
    return SpecFile(str(spec_file)) if capture is None else capture.parser


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


class _SpecSpanStream:
    """A reader limited to one indexed scan, starting at its first byte."""

    def __init__(self, stream, remaining):
        self.stream = stream
        self.remaining = remaining

    def read(self, size):
        if not self.remaining:
            return b""
        data = self.stream.read(min(size, self.remaining))
        if not data:
            raise SpecCaptureFailed("the private SPEC capture became shorter during validation")
        self.remaining -= len(data)
        return data


def _decode_spec_line_tail(parts, decoder):
    for part in parts:
        decoded = decoder.decode(part)
        if decoded:
            yield decoded
    decoded = decoder.decode(b"", final=True)
    if decoded:
        yield decoded


def _indexed_spec_lines(stream, remaining, start_line):
    from xrd_tools.io.metadata import iter_spec_record_parts

    for row, (head, tail) in enumerate(iter_spec_record_parts(_SpecSpanStream(stream, remaining)), start_line):
        if not tail:
            yield row, head.decode("utf-8", errors="replace"), ()
        else:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            yield row, decoder.decode(head), _decode_spec_line_tail(tail, decoder)


def _select_spec_lines(stream, index):
    scan_index = -1
    for physical_row, raw_line in enumerate(stream, 1):
        if raw_line.startswith("#S") and raw_line[2:3].isspace():
            scan_index += 1
            if scan_index > index:
                return
        if scan_index == index:
            yield physical_row, raw_line, ()


def _validate_spec_numeric_tokens(spec_file, scan_data, labels):
    """Validate the physical rows of a scan opened directly, outside the captured reader."""
    with Path(spec_file).open(encoding="utf-8", errors="replace") as stream:
        _validate_spec_rows(spec_file, scan_data, labels, _select_spec_lines(stream, scan_data.index))


_SPEC_DIGIT_RUN = re.compile(r"[0-9]+")
_SPEC_TOKEN_PART = re.compile(r"\s+|\S+")


def _long_spec_tokens(parts):
    """For a split row, match decimals without retaining an unbounded token.

    Replacing each nonempty ASCII digit run with one digit preserves _SPEC_NUMBER
    membership. Its longest compressed valid form is +0.0e+0. Keep a short
    original prefix for the diagnostic when a long token is invalid.
    """
    prefix = signature = ""
    for part in parts:
        for match in _SPEC_TOKEN_PART.finditer(part):
            text = match[0]
            if text[0].isspace():
                if prefix:
                    yield _display_spec_token(prefix, signature)
                    prefix = signature = ""
            else:
                prefix = (prefix + text)[:65]
                if len(signature) <= 7:
                    signature = _SPEC_DIGIT_RUN.sub("0", signature + text)[:8]
    if prefix:
        yield _display_spec_token(prefix, signature)


def _display_spec_token(prefix, signature):
    if len(prefix) <= 64:
        return prefix
    return signature if _SPEC_NUMBER.fullmatch(signature) else prefix[:64] + "…"


def _validate_spec_rows(spec_file, scan_data, labels, lines):
    """Check complete numeric tokens before silx's permissive conversion.

    Header comments and MCA records, including their backslash continuations, are not table rows.
    """
    in_table = False
    mca_continuation = False
    for physical_row, head, tail in lines:
        parts = chain((head,), tail)
        line = next((text.lstrip() for text in parts if text and not text.isspace()), "")
        if line == "#":
            line += next(parts, "")  # the #L marker can cross a physical read boundary
        if line.startswith("#L"):
            in_table = True
        if not in_table or not line or line.startswith("#"):
            for _ in parts:
                pass
            continue
        if mca_continuation or line.startswith("@"):
            last = ""
            for text in chain((line,), parts):
                stripped = text.rstrip()
                if stripped:
                    last = stripped[-1]
            mca_continuation = last == "\\"
            continue
        tokens = (tuple(islice(_long_spec_tokens(chain((line,), parts)), len(labels) + 1))
                  if tail else line.split())
        problem = _spec_row_problem(tokens, labels)
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

    capture = _scoped_capture(spec_file)
    parser = SpecFile(str(spec_file)) if capture is None else capture.parser
    scan_data = parser[scan_num]
    labels = tuple(str(label) for label in scan_data.labels)
    if capture is None:
        _validate_spec_numeric_tokens(spec_file, scan_data, labels)
    else:
        capture.check_numeric_tokens(spec_file, scan_data, labels)
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
