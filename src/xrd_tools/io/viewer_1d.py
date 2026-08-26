"""Pure bounded same-open inspection and decoding for standalone 1-D data."""
from __future__ import annotations

import ast
import ctypes
import hashlib
import os
import stat
import struct
import sys
import zipfile
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

VIEWER_1D_FD_RESERVE = 32
_POLICY_ID = "viewer-1d-v1"
_FORMATS = {".xye": 1, ".csv": 2, ".npy": 3, ".npz": 4}
SUPPORTED_VIEWER_1D_SUFFIXES = frozenset(_FORMATS)
_DTYPE_CODES = {
    "|b1": 0x01, "|i1": 0x10, "<i2": 0x11, ">i2": 0x12,
    "<i4": 0x13, ">i4": 0x14, "<i8": 0x15, ">i8": 0x16,
    "|u1": 0x20, "<u2": 0x21, ">u2": 0x22, "<u4": 0x23,
    ">u4": 0x24, "<u8": 0x25, ">u8": 0x26, "<f2": 0x30,
    ">f2": 0x31, "<f4": 0x32, ">f4": 0x33, "<f8": 0x34, ">f8": 0x35,
}


@dataclass(frozen=True, slots=True)
class Viewer1DFormatPolicy:
    identity: str = _POLICY_ID
    def __post_init__(self):
        if self.identity != _POLICY_ID: raise ValueError("unsupported viewer 1-D policy")


@dataclass(frozen=True, slots=True)
class Viewer1DMemoryLedger:
    counts: tuple[int, ...]
    sigmas: tuple[bool, ...]
    encoded: tuple[int, ...]
    B: int
    P: int
    C: int
    R: int
    N: int
    T: int
    A: int


@dataclass(frozen=True, slots=True)
class Viewer1DSourceManifest:
    canonical_path: str
    format: str
    points: int
    sigma_present: bool
    x_label: str
    x_unit: str
    y_label: str
    source_sha256: bytes
    role_sha256: bytes
    scalar_record: bytes


@dataclass(frozen=True, slots=True)
class Viewer1DBatchManifest:
    identity: str
    policy_identity: str
    sources: tuple[Viewer1DSourceManifest, ...]
    ledger: Viewer1DMemoryLedger


_INSPECTION_FACTORY = object()


class Viewer1DSourceInspection:
    """Factory-only same-open descriptor and scalar-manifest capability."""
    __slots__ = ("facts", "manifest", "ledger", "_streams", "_closed", "_claim")
    def __init_subclass__(cls, **kwargs): raise TypeError("viewer 1-D inspection is final")
    def __init__(self, facts, manifest, ledger, *, _claim=None):
        if _claim is not _INSPECTION_FACTORY: raise TypeError("foreign viewer 1-D inspection")
        self.facts, self.manifest, self.ledger = facts, manifest, ledger
        self._streams = [fact.stream for fact in facts]
        self._closed, self._claim = False, _INSPECTION_FACTORY


@dataclass(slots=True)
class _Fact:
    stream: object
    path: str
    state: tuple
    manifest: Viewer1DSourceManifest
    schema: int
    roles: object = None


def _is_control(error):
    return isinstance(error, MemoryError) or (isinstance(error, BaseException)
                                               and not isinstance(error, Exception))


def _descriptor_limit_and_count():
    if os.name == "nt":
        crt = ctypes.CDLL("ucrtbase.dll", use_errno=True)
        function = crt._getmaxstdio; function.argtypes = []; function.restype = ctypes.c_int
        limit = function()
        if not 1 <= limit <= 8192: raise RuntimeError("descriptor headroom")
        count = 0
        for descriptor in range(limit):
            try: os.fstat(descriptor); count += 1
            except OSError as error:
                if getattr(error, "errno", None) != 9: raise
        return limit, count
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = hard if soft == resource.RLIM_INFINITY else soft
    directory = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
    if type(limit) is not int or limit <= 0 or not os.path.isdir(directory):
        raise RuntimeError("descriptor headroom")
    return limit, len(os.listdir(directory))


def _state(fd, path):
    opened, named = os.fstat(fd), os.stat(path)
    return (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
            opened.st_ctime_ns, getattr(opened, "st_gen", None), named.st_dev,
            named.st_ino, named.st_size, named.st_mtime_ns, named.st_ctime_ns,
            getattr(named, "st_gen", None))


def _digest(stream):
    stream.seek(0); digest = hashlib.sha256(); total = 0
    while True:
        block = stream.read(1024 * 1024)
        if not block: break
        digest.update(block); total += len(block)
    return digest.digest(), total


def _npy_header(stream, *, label=False):
    stream.seek(0); magic = stream.read(6)
    if magic != b"\x93NUMPY": raise ValueError("NPY magic")
    version = tuple(stream.read(2))
    if version not in {(1, 0), (2, 0), (3, 0)}: raise ValueError("NPY version")
    width = 2 if version == (1, 0) else 4
    raw_length = stream.read(width)
    if len(raw_length) != width: raise ValueError("NPY header")
    length = int.from_bytes(raw_length, "little")
    if length > 65536: raise ValueError("NPY header")
    raw = stream.read(length)
    if len(raw) != length: raise ValueError("NPY header")
    try: header = ast.literal_eval(raw.decode("latin1" if version != (3, 0) else "utf8").strip())
    except BaseException as error:
        if _is_control(error): raise
        raise ValueError("NPY header") from error
    if type(header) is not dict or set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError("NPY header schema")
    dtype, shape = np.dtype(header["descr"]), header["shape"]
    if (header["fortran_order"] is not False or dtype.hasobject or dtype.fields is not None
            or dtype.subdtype is not None or (dtype.kind != "U" if label else
                dtype.kind not in "biuf" or dtype.itemsize > 8)
            or type(shape) is not tuple): raise ValueError("NPY dtype/shape")
    size = dtype.itemsize
    for value in shape:
        if type(value) is not int or value < 0: raise ValueError("NPY shape")
        size *= value
    return dtype, shape, magic + bytes(version) + raw_length + raw, size


def _shape(shape):
    if len(shape) == 1 and 1 <= shape[0] <= 1_000_000: return shape[0], 1, True
    if len(shape) == 2 and 1 <= shape[0] <= 1_000_000 and shape[1] in (2, 3):
        return shape[0], shape[1], False
    raise ValueError("NPY shape")


def _dtype_code(dtype):
    value = dtype.str
    if value[0] == "=": value = ("<" if sys.byteorder == "little" else ">") + value[1:]
    try: return _DTYPE_CODES[value]
    except KeyError: raise ValueError("NPY dtype") from None


def _scan_text(stream, csv):
    stream.seek(0); count = columns = 0; labels = ("x", "", "intensity"); header = False
    first = True
    while True:
        raw = stream.readline(4098)
        if not raw: break
        if len(raw) > 4096 or b"\x00" in raw: raise ValueError("text line bound")
        if first and raw.startswith(b"\xef\xbb\xbf"): raw = raw[3:]
        first = False
        try: line = raw.decode("utf8").strip()
        except UnicodeDecodeError as error: raise ValueError("text encoding") from error
        if not line or line.lstrip().startswith("#"): continue
        if '"' in line or "'" in line or "\\" in line: raise ValueError("text quoting")
        cells = [value.strip() for value in line.split(",")] if csv else line.split()
        if len(cells) not in (2, 3) or any(not value for value in cells): raise ValueError("text columns")
        try: values = tuple(float(value) for value in cells)
        except ValueError:
            if not csv or count or header: raise ValueError("CSV header")
            if any(len(value.encode("utf8")) > 128 for value in cells): raise ValueError("CSV label")
            labels, header = (cells[0], "", cells[1]), True
            continue
        columns = columns or len(values)
        if len(values) != columns or not np.isfinite(values[0]) or any(np.isinf(v) for v in values[1:]):
            raise ValueError("text numeric values")
        if len(values) == 3 and np.isfinite(values[2]) and values[2] < 0:
            raise ValueError("negative sigma")
        count += 1
        if count > 1_000_000: raise ValueError("point count")
    if count < 1: raise ValueError("empty source")
    return count, columns, labels


def _record(index, format_code, schema, columns, dtypes, sigma, synthesized,
            labels, ordinals, selected, members, path_bytes, points, encoded,
            compressed=0, uncompressed=0):
    value = bytearray(64); value[0:8] = bytes((1, format_code, schema, columns,
        *(dtypes + (0, 0, 0))[:3], int(sigma) | (int(synthesized) << 1)))
    struct.pack_into(">H", value, 8, index); value[10:16] = bytes(ordinals)
    value[16:19] = bytes(len(label.encode()) for label in labels)
    value[19:22] = bytes((selected, members, bool(schema) * sum(code != 0 for code in dtypes)))
    struct.pack_into(">HQQQQ", value, 22, path_bytes, points, encoded, compressed, uncompressed)
    return bytes(value)


def _inspect_npz(stream, index, path, digest, encoded):
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist(); names = [info.filename for info in infos]
        if (not 1 <= len(infos) <= 16 or len(set(names)) != len(names)
                or len(archive.comment) > 4096
                or sum(46 + len(info.filename.encode("utf8")) + len(info.extra)
                       + len(info.comment) for info in infos) > 128 * 1024):
            raise ValueError("NPZ member bound")
        for info in infos:
            parts = info.filename.replace("\\", "/").split("/")
            if (not info.filename.endswith(".npy") or info.is_dir() or info.flag_bits & 1
                    or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or info.file_size > 1024**3 or len(info.extra) > 4096
                    or len(info.filename.encode("utf8")) > 256 or info.filename.startswith("/")
                    or "\\" in info.filename or ".." in parts or any(not part for part in parts)
                    or len(parts[0]) >= 2 and parts[0][1] == ":"):
                raise ValueError("NPZ member metadata")
        if sum(info.file_size for info in infos) > 1024**3: raise ValueError("NPZ size")
        stems = {name[:-4]: name for name in names}; labels = {"x_label", "x_unit", "y_label"}
        numeric = set(stems) - labels
        if {"x", "y"} <= numeric and numeric <= {"x", "y", "sigma"}: schema, roles = 1, ("x", "y") + (("sigma",) if "sigma" in numeric else ())
        elif numeric == {"data"}: schema, roles = 2, ("data",)
        elif len(numeric) == 1 and not numeric & {"x", "y", "sigma", "data"}: schema, roles = 3, tuple(numeric)
        else: raise ValueError("ambiguous NPZ schema")
        if set(stems) != set(roles) | (set(stems) & labels): raise ValueError("NPZ extra member")
        raw_records, position = [], archive.start_dir
        for info in infos:
            stream.seek(position); central = stream.read(46)
            if len(central) != 46 or central[:4] != b"PK\x01\x02": raise ValueError("NPZ central directory")
            lengths = struct.unpack_from("<HHH", central, 28)
            central += stream.read(sum(lengths)); position += len(central)
            stream.seek(info.header_offset); local = stream.read(30)
            if len(local) != 30 or local[:4] != b"PK\x03\x04": raise ValueError("NPZ local header")
            local_lengths = struct.unpack_from("<HH", local, 26)
            if local_lengths[0] > 256 or local_lengths[1] > 4096:
                raise ValueError("NPZ local metadata")
            local += stream.read(sum(local_lengths))
            raw_records.append((central, local))
        parsed, label_values = {}, {"x_label": "x", "x_unit": "",
            "y_label": os.path.splitext(os.path.basename(path))[0]}
        meta = hashlib.sha256(b"V1DMETA1")
        for role in (*roles, *(name for name in ("x_label", "x_unit", "y_label") if name in stems)):
            info = infos[names.index(stems[role])]
            with archive.open(info) as member:
                dtype, shape, prefix, payload = _npy_header(member, label=role in labels)
                if member.tell() + payload != info.file_size: raise ValueError("NPZ declared bytes")
                if role in labels:
                    if dtype.kind != "U" or shape != () or payload > 512: raise ValueError("NPZ label")
                    raw = member.read(payload)
                    if len(raw) != payload or member.read(1): raise ValueError("NPZ member EOF")
                    order = dtype.byteorder
                    if order in {"=", "|"}: order = "<" if sys.byteorder == "little" else ">"
                    text = raw.decode("utf-32-le" if order == "<" else "utf-32-be").rstrip("\0")
                    if type(text) is not str or not 1 <= len(text.encode("utf8")) <= 128 or "\0" in text:
                        raise ValueError("NPZ label")
                    label_values[role] = text
                else:
                    parsed[role] = (dtype, shape, info)
                    while member.read(1024 * 1024): pass
                ordinal = names.index(info.filename)
                tag = ((roles.index(role) if role in roles else
                        3 + ("x_label", "x_unit", "y_label").index(role)))
                encoded_name = info.filename.encode("utf8")
                central, local = raw_records[ordinal]
                meta.update(bytes((tag, ordinal)) + len(encoded_name).to_bytes(2, "big")
                    + encoded_name + len(central).to_bytes(4, "big") + central
                    + len(local).to_bytes(4, "big") + local
                    + len(prefix).to_bytes(4, "big") + prefix)
        if schema == 1:
            shapes = [parsed[role][1] for role in roles]
            if any(len(shape) != 1 for shape in shapes) or len({shape[0] for shape in shapes}) != 1:
                raise ValueError("NPZ named shape")
            points, columns, synthesized = shapes[0][0], len(roles), False
            if not 1 <= points <= 1_000_000: raise ValueError("point count")
            dtypes = tuple(_dtype_code(parsed[role][0]) for role in roles)
        else:
            points, columns, synthesized = _shape(parsed[roles[0]][1])
            dtypes = (_dtype_code(parsed[roles[0]][0]),)
        ordinals = tuple(names.index(stems.get(role, "")) if role in stems else 0xff
                         for role in (roles + (None,) * 3)[:3] + ("x_label", "x_unit", "y_label"))
        selected = len(roles) + sum(name in stems for name in labels)
        selected_infos = [infos[names.index(stems[role])] for role in
            (*roles, *(name for name in ("x_label", "x_unit", "y_label") if name in stems))]
        compressed = sum(info.compress_size for info in selected_infos)
        uncompressed = sum(info.file_size for info in selected_infos)
        label_tuple = (label_values["x_label"], label_values["x_unit"], label_values["y_label"])
        if any(len(label.encode("utf8")) > 128 for label in label_tuple): raise ValueError("NPZ label")
        scalar = _record(index, 4, schema, columns, dtypes, columns == 3, synthesized,
                         label_tuple, ordinals, selected, len(infos), len(str(path).encode()),
                         points, encoded, compressed, uncompressed)
        manifest = Viewer1DSourceManifest(str(path), "npz", points, columns == 3,
            *label_tuple, digest, meta.digest(), scalar)
        return manifest, schema, MappingProxyType({role: stems[role] for role in roles})


def _inspect(stream, index, path, state, suffix, *, retain_roles=False):
    digest, encoded = _digest(stream)
    if encoded > 256 * 1024**2: raise ValueError("encoded size")
    code = _FORMATS.get(suffix)
    if code is None: raise ValueError("viewer suffix")
    if suffix == ".npz":
        manifest, schema, roles = _inspect_npz(stream, index, path, digest, encoded)
        return _Fact(stream, str(path), state, manifest, schema,
                     roles if retain_roles else None)
    if suffix in {".xye", ".csv"}:
        points, columns, labels = _scan_text(stream, suffix == ".csv")
        dtypes, synthesized = (), False
    else:
        stream.seek(0); dtype, shape, _, payload = _npy_header(stream)
        points, columns, synthesized = _shape(shape)
        if stream.tell() + payload != encoded: raise ValueError("NPY EOF")
        dtypes, labels = (_dtype_code(dtype),), ("x", "", os.path.splitext(os.path.basename(path))[0])
    if any(len(label.encode("utf8")) > 128 for label in labels):
        raise ValueError("viewer label bound")
    scalar = _record(index, code, 0, columns, dtypes, columns == 3, synthesized,
                     labels, (0xff,) * 6, 0, 0, len(str(path).encode()), points, encoded)
    manifest = Viewer1DSourceManifest(str(path), suffix[1:], points, columns == 3,
        *labels, digest, bytes(32), scalar)
    return _Fact(stream, str(path), state, manifest, 0, None)


def _columns(array):
    value = np.asarray(array)
    if value.ndim == 1: return np.arange(len(value), dtype=float), np.array(value, dtype=float, copy=True), None
    return (np.array(value[:, 0], dtype=float, copy=True),
            np.array(value[:, 1], dtype=float, copy=True),
            None if value.shape[1] == 2 else np.array(value[:, 2], dtype=float, copy=True))


def _read_text(stream, csv, points):
    stream.seek(0); columns = None; result = None; row = 0
    first = True
    while True:
        raw = stream.readline(4098)
        if not raw: break
        if first and raw.startswith(b"\xef\xbb\xbf"): raw = raw[3:]
        first = False; line = raw.decode("utf8").strip()
        if not line or line.lstrip().startswith("#"): continue
        cells = [value.strip() for value in line.split(",")] if csv else line.split()
        try: values = tuple(float(value) for value in cells)
        except ValueError: continue
        if result is None:
            columns = len(values)
            result = tuple(np.empty(points, dtype=float) for _ in range(columns))
        for target, value in zip(result, values): target[row] = value
        row += 1
    if result is None or row != points: raise ValueError("text point count changed")
    return result[0], result[1], None if columns == 2 else result[2]


def _read_fact(fact):
    stream, manifest = fact.stream, fact.manifest
    recertified = _inspect(stream, int.from_bytes(manifest.scalar_record[8:10], "big"),
                           fact.path, fact.state, "." + manifest.format, retain_roles=True)
    if recertified.manifest != manifest or _state(stream.fileno(), fact.path) != fact.state:
        raise ValueError("viewer source changed")
    if manifest.format in {"xye", "csv"}: values = _read_text(stream, manifest.format == "csv", manifest.points)
    elif manifest.format == "npy":
        stream.seek(0); dtype, shape, _, _ = _npy_header(stream)
        mapped = np.memmap(stream, dtype=dtype, mode="r", offset=stream.tell(), shape=shape)
        try: values = _columns(mapped)
        finally: mapped._mmap.close()
    else:
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            arrays = {role: np.load(archive.open(name), allow_pickle=False)
                      for role, name in recertified.roles.items()}
        if recertified.schema == 1:
            values = (np.array(arrays["x"], dtype=float, copy=True),
                      np.array(arrays["y"], dtype=float, copy=True),
                      None if "sigma" not in arrays else np.array(arrays["sigma"], dtype=float, copy=True))
        else: values = _columns(next(iter(arrays.values())))
    x, y, sigma = values
    if (len(x) != manifest.points or not np.all(np.isfinite(x))
            or np.any(np.isinf(y)) or sigma is not None and
            (np.any(np.isinf(sigma)) or np.any(sigma[np.isfinite(sigma)] < 0))):
        raise ValueError("viewer numeric values changed")
    if _inspect(stream, int.from_bytes(manifest.scalar_record[8:10], "big"), fact.path,
                fact.state, "." + manifest.format).manifest != manifest:
        raise ValueError("viewer source changed")
    if _state(stream.fileno(), fact.path) != fact.state: raise ValueError("viewer source changed")
    return values


def inspect_viewer_1d_sources(selected_paths, policy, capacity_bytes, reserved_bytes):
    if (type(selected_paths) is not tuple or not 1 <= len(selected_paths) <= 256
            or type(policy) is not Viewer1DFormatPolicy
            or type(capacity_bytes) is not int or type(reserved_bytes) is not int
            or not 0 < reserved_bytes <= capacity_bytes):
        raise TypeError("viewer 1-D inspection request is malformed")
    formats = tuple(os.path.splitext(path)[1].lower() for path in selected_paths)
    if any(value not in _FORMATS for value in formats): raise ValueError("viewer suffix")
    paths = tuple(os.path.realpath(os.path.expanduser(path)) for path in selected_paths)
    if any(len(os.fsencode(path)) > 4096 for path in paths): raise ValueError("path bound")
    limit, count = _descriptor_limit_and_count()
    if limit - count < len(paths) + VIEWER_1D_FD_RESERVE + 1:
        raise RuntimeError("descriptor headroom")
    streams, descriptors = [], []
    try:
        try:
            for _ in paths: descriptors.append(os.open(os.devnull, os.O_RDONLY))
            for descriptor, path in zip(descriptors, paths):
                source = os.open(path, os.O_RDONLY)
                try:
                    state = os.fstat(source)
                    if not stat.S_ISREG(state.st_mode) or state.st_size > 256 * 1024**2:
                        raise ValueError("regular source/encoded size")
                    os.lseek(source, 0, os.SEEK_CUR); os.dup2(source, descriptor)
                finally: os.close(source)
            for index, descriptor in enumerate(descriptors):
                streams.append(os.fdopen(descriptor, "rb", buffering=0)); descriptors[index] = -1
        finally:
            for descriptor in descriptors:
                if descriptor >= 0: os.close(descriptor)
        facts = tuple(_inspect(stream, index, path, _state(stream.fileno(), path), suffix)
            for index, (stream, path, suffix) in enumerate(zip(streams, paths, formats)))
        if any(_state(fact.stream.fileno(), fact.path) != fact.state for fact in facts):
            raise ValueError("viewer source changed")
        counts = tuple(fact.manifest.points for fact in facts)
        sigmas = tuple(fact.manifest.sigma_present for fact in facts)
        encoded = tuple(os.fstat(fact.stream.fileno()).st_size for fact in facts)
        if sum(encoded) > 1024**3: raise ValueError("encoded total")
        C = 8 * sum(n * (2 + int(sigma)) for n, sigma in zip(counts, sigmas))
        P, B, R = len(paths) * max(counts), capacity_bytes, reserved_bytes
        N, T = 9 * 8 * P, R + max(3 * C, 9 * 8 * P)
        ledger = Viewer1DMemoryLedger(counts, sigmas, encoded, B, P, C, R, N, T, C + T)
        if ledger.A > B: raise ValueError(f"viewer 1-D budget requires {ledger.A}/{B}")
        identity = hashlib.sha256(b"V1DBATCH1" + b"".join(fact.manifest.scalar_record
            + fact.manifest.source_sha256 + fact.manifest.role_sha256
            + fact.manifest.canonical_path.encode() for fact in facts)).hexdigest()
        manifest = Viewer1DBatchManifest(identity, policy.identity,
            tuple(fact.manifest for fact in facts), ledger)
        return Viewer1DSourceInspection(facts, manifest, ledger, _claim=_INSPECTION_FACTORY)
    except BaseException:
        while streams:
            stream = streams[-1]; stream.close(); streams.pop()
        raise


def decode_viewer_1d_sources(inspection):
    if not viewer_1d_inspection_is_open(inspection):
        raise TypeError("viewer 1-D inspection is stale")
    return tuple(_read_fact(fact) for fact in inspection.facts)


def viewer_1d_inspection_ledger(inspection):
    if not viewer_1d_inspection_is_open(inspection):
        raise TypeError("viewer 1-D inspection is stale")
    return inspection.ledger


def close_viewer_1d_sources(inspection):
    if type(inspection) is not Viewer1DSourceInspection or inspection._claim is not _INSPECTION_FACTORY:
        raise TypeError("foreign viewer 1-D inspection")
    while inspection._streams:
        stream = inspection._streams[-1]; stream.close(); inspection._streams.pop()
    inspection._closed = True


def viewer_1d_inspection_is_open(inspection):
    return (type(inspection) is Viewer1DSourceInspection
            and inspection._claim is _INSPECTION_FACTORY and not inspection._closed)


__all__ = [name for name in tuple(globals()) if name.startswith("Viewer1D") or name in {
    "VIEWER_1D_FD_RESERVE", "SUPPORTED_VIEWER_1D_SUFFIXES",
    "inspect_viewer_1d_sources", "decode_viewer_1d_sources",
    "close_viewer_1d_sources", "viewer_1d_inspection_is_open",
    "viewer_1d_inspection_ledger"}]
