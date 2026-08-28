from __future__ import annotations

import importlib
import hashlib
import io
import mmap
import os
import struct
import sys
import zipfile
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

_EXPECTED_R = 11_546_624

try:
    _PHYSICAL_RAM = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
except (ValueError, OSError, AttributeError):
    _PHYSICAL_RAM = 0
_EXPECTED_B = 884_736_000 if _PHYSICAL_RAM <= 0 else min(1024**3, _PHYSICAL_RAM // 20)


@pytest.fixture(autouse=True)
def _load_api():
    global api
    api = importlib.import_module("xrd_tools.io.viewer_2d")


def _npy_bytes(value: np.ndarray, *, version=(1, 0)) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=version, allow_pickle=False)
    return stream.getvalue()


def _npz(path: Path, members, *, compression=zipfile.ZIP_DEFLATED) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def _write_hdf_stack(path, value, dataset="entry/data/data"):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset, data=value)


def _processed_source(handle, label, locator, frame, dataset=None):
    entry = handle.require_group("entry")
    entry.require_group("integrated_1d")
    source = entry.require_group(f"frames/frame_{label:04d}/source")
    source.create_dataset("path", data=np.bytes_(locator))
    source.create_dataset("frame_index", data=frame)
    if dataset is not None:
        source.attrs["dataset_path"] = dataset


def _processed_thumbnail(handle, label, value, *, vmin, vmax, mask=None):
    entry = handle.require_group("entry")
    entry.require_group("integrated_1d")
    frame = entry.require_group(f"frames/frame_{label:04d}")
    thumbnail = frame.create_dataset("thumbnail", data=value)
    thumbnail.attrs.update(dtype=str(value.dtype), vmin=vmin, vmax=vmax)
    if mask is not None:
        frame.create_dataset("thumbnail_mask", data=mask)
    return thumbnail


def _external_master(path, links):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry/data")
        for name, segment in links:
            group[name] = h5py.ExternalLink(segment.name, "/entry/data/data")


def _eiger_master(tmp_path, values, master_name="master.nxs"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    arrays, segments = [], []
    for name, value in values:
        segment = tmp_path / name
        _write_hdf_stack(segment, value)
        arrays.append(value)
        segments.append(segment)
    master = tmp_path / master_name
    _external_master(master, ((segment.stem, segment) for segment in segments))
    return master, segments, arrays


def _after_call(original, action, entered, predicate=lambda *a, **k: True,
                transform=lambda value: value):
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        if not entered and predicate(*args, **kwargs):
            entered.append("metadata")
            action()
        return transform(result)
    return wrapped


def _raises(expected, operation, *args, **kwargs):
    with pytest.raises(expected) as caught:
        operation(*args, **kwargs)
    return caught.value


def _assert_refusal(code, operation, *args, **kwargs):
    caught = _raises(api.Viewer2DReadError, operation, *args, **kwargs)
    assert caught.code is code


def _assert_limit(operation, *args, **kwargs):
    _assert_refusal(api.Viewer2DRefusalCode.LIMIT_EXCEEDED, operation, *args, **kwargs)


def _assert_changed(operation, *args, **kwargs):
    _assert_refusal(api.Viewer2DRefusalCode.VIEWER_SOURCE_CHANGED, operation, *args, **kwargs)


def _assert_canonical(frame, expected):
    array = frame.array
    assert np.array_equal(array, np.asarray(expected, dtype=float), equal_nan=True)
    assert array.dtype == np.dtype(float)
    assert frame.provenance.canonical_dtype == np.dtype(float).str
    assert array.ndim == 2 and array.flags.c_contiguous and array.flags.owndata
    assert not array.flags.writeable and array.base is None
    assert frame.provenance.canonical_sha256
    assert frame.provenance.canonical_nbytes == array.nbytes


def _watch_hdf_attributes(monkeypatch, h5py):
    touches, opens, reads, charged, inspected = [], [], [], [], {}
    original_open, original_touch = h5py.h5a.open, api._HdfWalk.touch

    class AttributeProbe:
        def __init__(self, attribute, owner):
            self.attribute, self.owner = attribute, owner

        def __getattr__(self, name):
            kind = {"shape": "space", "get_space": "space", "dtype": "dtype",
                    "get_type": "dtype", "get_storage_size": "storage"}.get(name)
            if kind is not None:
                inspected.setdefault(self.owner, set()).add(kind)
            return getattr(self.attribute, name)

        def read(self, *args, **kwargs):
            reads.append((self.owner, frozenset(inspected.get(self.owner, ()))))
            return self.attribute.read(*args, **kwargs)

    def opened(location, name, *args, **kwargs):
        attribute = original_open(location, name, *args, **kwargs)
        if name not in (b"NX_class", "NX_class"):
            return attribute
        owner = h5py.h5i.get_name(location).decode()
        charged.append(touches[-1:] == [owner + "/@NX_class"])
        opens.append(owner)
        return AttributeProbe(attribute, owner)

    def touch(walk, name, **kwargs):
        touches.append(name)
        return original_touch(walk, name, **kwargs)

    monkeypatch.setattr(h5py.h5a, "open", opened)
    monkeypatch.setattr(api._HdfWalk, "touch", touch)
    return touches, opens, reads, charged, inspected


def _watch_named_hdf_attribute(monkeypatch, h5py, target):
    reads, high = [], []
    original_open, original_getitem = h5py.h5a.open, h5py.AttributeManager.__getitem__
    class Probe:
        def __init__(self, attribute): self.attribute = attribute
        def __getattr__(self, name): return getattr(self.attribute, name)
        def read(self, destination, *args, **kwargs):
            memory = kwargs.get("mtype")
            result = self.attribute.read(destination, *args, **kwargs)
            raw = destination.tobytes()
            reads.append((destination.dtype.str, destination.dtype.hasobject, destination.nbytes,
                len(raw), None if memory is None else
                (memory.get_size(), memory.get_strpad(), memory.get_cset()), raw.find(b"\0")))
            return result
    def opened(location, name, *args, **kwargs):
        attribute = original_open(location, name, *args, **kwargs)
        return Probe(attribute) if name in (target, target.encode()) else attribute
    def getitem(manager, name):
        if name in (target, target.encode()): high.append(name)
        return original_getitem(manager, name)
    monkeypatch.setattr(h5py.h5a, "open", opened)
    monkeypatch.setattr(h5py.AttributeManager, "__getitem__", getitem)
    return reads, high
def _slots_have_no_array(value):
    seen = set()

    def visit(member, trail):
        if id(member) in seen or isinstance(member, (str, bytes, int, float, bool, type(None), Enum)):
            return
        seen.add(id(member))
        assert not isinstance(member, (np.ndarray, io.IOBase, mmap.mmap, zipfile.ZipFile)), trail
        assert not (hasattr(member, "read") and hasattr(member, "close")), trail
        if is_dataclass(member):
            for item in fields(member):
                visit(getattr(member, item.name), f"{trail}.{item.name}")
        elif isinstance(member, (tuple, list, set, frozenset)):
            for index, value in enumerate(member):
                visit(value, f"{trail}[{index}]")
        elif isinstance(member, dict):
            for key, value in member.items():
                visit(value, f"{trail}[{key!r}]")

    visit(value, type(value).__name__)


def test_csv_catalog_is_array_free_and_second_pass_builds_one_canonical_frame(tmp_path, monkeypatch):
    path = tmp_path / "matrix.csv"
    path.write_bytes(b"1,2,nan\n3.5,-4,6\n")
    monkeypatch.setattr(np, "loadtxt", lambda *a, **k: pytest.fail("whole-file load"))
    original_scan = api._csv_scan
    original_empty = np.empty
    events = []

    def scan(stream, target=None):
        events.append(("scan", id(stream), target is None))
        return original_scan(stream, target)

    def empty(*args, **kwargs):
        events.append(("empty", None, None))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(api, "_csv_scan", scan)
    monkeypatch.setattr(np, "empty", empty)

    catalog = api.catalog_viewer_2d(path)
    assert catalog.source_kind is api.Viewer2DSourceKind.CSV_MATRIX
    assert catalog.frame_labels == (0,)
    assert catalog.source_shape == (2, 3)
    assert catalog.policy_version == "viewer-2d-v1"
    assert catalog.canonical_path == str(path.resolve())
    _slots_have_no_array(catalog)
    assert events == [("scan", events[0][1], True)]

    events.clear()
    frame = api.read_viewer_2d_frame(catalog, 0)
    assert [event[0] for event in events] == ["scan", "empty", "scan"]
    assert events[0][1] == events[2][1] and events[0][2] and not events[2][2]
    _assert_canonical(frame, [[1, 2, np.nan], [3.5, -4, 6]])
    assert frame.provenance.source_kind is api.Viewer2DSourceKind.CSV_MATRIX
    assert frame.provenance.source_sha256 == catalog.primary_revision.sha256


@pytest.mark.parametrize(
    "payload", [b"\xef\xbb\xbf1,2\n", b"# 1,2\n3,4\n", b"1,2\n3\n",
                b"1,,2\n", b'"1",2\n', b"\n"],
)
def test_csv_closed_grammar_refuses_schema_ragged_empty_quoted_and_infinite(tmp_path, payload):
    path = tmp_path / "bad.csv"
    path.write_bytes(payload)
    caught = _raises(api.Viewer2DReadError, api.catalog_viewer_2d, path)
    assert caught.code in {
        api.Viewer2DRefusalCode.FORMAT_INVALID,
        api.Viewer2DRefusalCode.LIMIT_EXCEEDED,
    }


@pytest.mark.parametrize("payload", [b"a,2\n3,4\n", b"1,inf\n", b"1,-Infinity\n"])
def test_csv_structurally_valid_numeric_invalid_catalogs_then_frame_refuses(tmp_path, payload):
    path = tmp_path / "numeric-invalid.csv"
    path.write_bytes(payload)
    catalog = api.catalog_viewer_2d(path)
    _slots_have_no_array(catalog)
    _assert_refusal(api.Viewer2DRefusalCode.FORMAT_INVALID, api.read_viewer_2d_frame, catalog, 0)


def test_csv_line_cap_is_checked_by_bounded_readline(tmp_path):
    path = tmp_path / "long.csv"
    path.write_bytes(b"1" * (1024 * 1024 + 1) + b"\n")
    _assert_limit(api.catalog_viewer_2d, path)


@pytest.mark.parametrize(
    ("shape", "index"),
    [((4, 6), 0), ((3, 4, 6), 2), ((5, 7, 3), 4), ((2, 3, 4), 1)],
)
def test_npy_rank_only_grammar_reads_exact_selected_scalar_frame(tmp_path, monkeypatch, shape, index):
    source = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    path = tmp_path / "stack.npy"
    np.save(path, source, allow_pickle=False)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("whole-stack load"))
    monkeypatch.setattr(np, "memmap", lambda *a, **k: pytest.fail("mmap"))
    monkeypatch.setattr(np, "fromfile", lambda *a, **k: pytest.fail("whole-file numeric read"))
    original_frombuffer = np.frombuffer
    numeric_reads = []

    def frombuffer(payload, *args, **kwargs):
        numeric_reads.append(len(payload))
        return original_frombuffer(payload, *args, **kwargs)

    monkeypatch.setattr(np, "frombuffer", frombuffer)

    catalog = api.catalog_viewer_2d(path)
    expected_count = 1 if len(shape) == 2 else shape[0]
    assert catalog.frame_labels == tuple(range(expected_count))
    assert catalog.source_shape == shape
    _slots_have_no_array(catalog)
    expected = source if len(shape) == 2 else source[index]
    _assert_canonical(api.read_viewer_2d_frame(catalog, index), expected)
    assert numeric_reads and max(numeric_reads) == expected.nbytes


@pytest.mark.parametrize(
    "value", [np.arange(3), np.zeros((2, 3, 4, 1)), np.zeros((2, 3), order="F"),
              np.array([[1 + 2j]]), np.array([["x"]]),
              np.array([[object()]], dtype=object), np.array([[np.inf]]),
              np.array([[2**53 + 1]], dtype=np.uint64)],
)
def test_npy_refuses_unsupported_rank_order_dtype_infinity_and_inexact_integer(tmp_path, value):
    path = tmp_path / "bad.npy"
    np.save(path, value, allow_pickle=True)
    with pytest.raises(api.Viewer2DReadError):
        catalog = api.catalog_viewer_2d(path)
        api.read_viewer_2d_frame(catalog, 0)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_npy_versions_and_exact_payload_eof(tmp_path, version):
    value = np.arange(12, dtype="<f4").reshape(3, 4)
    path = tmp_path / "version.npy"
    path.write_bytes(_npy_bytes(value, version=version))
    _assert_canonical(api.read_viewer_2d_frame(api.catalog_viewer_2d(path), 0), value)
    path.write_bytes(path.read_bytes() + b"x")
    _raises(api.Viewer2DReadError, api.catalog_viewer_2d, path)


def test_npz_priority_rank_rule_and_total_member_crc(tmp_path, monkeypatch):
    path = tmp_path / "stack.npz"
    preferred = np.arange(5 * 512 * 512, dtype=np.int16).reshape(5, 512, 512)
    secondary = np.arange(2 * 4 * 6, dtype=np.int32).reshape(2, 4, 6)
    ignored = np.full((2, 2), 99, dtype=np.int8)
    _npz(path, [("other.npy", _npy_bytes(ignored)), ("data.npy", _npy_bytes(secondary)),
                ("image.npy", _npy_bytes(preferred))],
         compression=zipfile.ZIP_STORED)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("NpzFile/full decode"))

    catalog = api.catalog_viewer_2d(path)
    assert catalog.member_name == "image.npy"
    assert catalog.frame_labels == tuple(range(5))
    _assert_canonical(api.read_viewer_2d_frame(catalog, 0), preferred[0])

    payload = bytearray(path.read_bytes())
    # Corrupt compressed payload bytes while keeping central metadata intact.
    local = 0
    while True:
        local = payload.index(b"PK\x03\x04", local)
        name_len, extra_len = struct.unpack_from("<HH", payload, local + 26)
        name = bytes(payload[local + 30:local + 30 + name_len])
        data_start = local + 30 + name_len + extra_len
        if name == b"image.npy":
            break
        compressed = struct.unpack_from("<L", payload, local + 18)[0]
        local = data_start + compressed
    payload[data_start + len(_npy_bytes(preferred)) - 1] ^= 0x20
    path.write_bytes(payload)
    monkeypatch.setattr(
        api, "_descriptor_revision", lambda *args, **kwargs: catalog.primary_revision,
    )
    _raises(api.Viewer2DReadError, api.read_viewer_2d_frame, catalog, 0)


def test_npz_preferred_invalid_never_falls_back_and_unnamed_must_be_unambiguous(tmp_path):
    valid = _npy_bytes(np.arange(6).reshape(2, 3))
    rank4 = _npy_bytes(np.zeros((2, 3, 4, 1)))
    preferred = tmp_path / "preferred.npz"
    _npz(preferred, [("image.npy", rank4), ("ok.npy", valid)])
    _raises(api.Viewer2DReadError, api.catalog_viewer_2d, preferred)

    ambiguous = tmp_path / "ambiguous.npz"
    _npz(ambiguous, [("a.npy", valid), ("b.npy", valid)])
    _raises(api.Viewer2DReadError, api.catalog_viewer_2d, ambiguous)

    single = tmp_path / "single.npz"
    shaped = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    _npz(single, [("only.npy", _npy_bytes(shaped))], compression=zipfile.ZIP_STORED)
    catalog = api.catalog_viewer_2d(single)
    assert catalog.member_name == "only.npy" and catalog.frame_labels == (0, 1)
    _assert_canonical(api.read_viewer_2d_frame(catalog, 1), shaped[1])


def test_npz_central_metadata_counts_member_comments_at_exact_cap(tmp_path):
    payload = _npy_bytes(np.arange(4).reshape(2, 2))

    def write(path, comments):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            for index, comment in enumerate(comments):
                info = zipfile.ZipInfo(f"m{index}.npy")
                info.comment = b"x" * comment
                archive.writestr(info, payload)

    # Three headers total 156 fixed/name bytes; comments fill exactly 128 KiB.
    accepted = tmp_path / "accepted.npz"
    write(accepted, (43638, 43638, 43640))
    # Metadata admits; the three otherwise-valid arrays are ambiguous.
    accepted_error = _raises(api.Viewer2DReadError, api.catalog_viewer_2d, accepted)
    assert accepted_error.code is not api.Viewer2DRefusalCode.LIMIT_EXCEEDED
    refused = tmp_path / "refused.npz"
    write(refused, (43638, 43638, 43641))
    _assert_limit(api.catalog_viewer_2d, refused)


def test_raw_tiff_and_hdf5_catalogs_read_only_one_selected_frame(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    h5py = pytest.importorskip("h5py")
    admissions = []
    original_admit = api._admit

    def admit(shape, *, encoded=0):
        admissions.append(encoded)
        return original_admit(shape, encoded=encoded)

    monkeypatch.setattr(api, "_admit", admit)

    tiff_data = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    tiff_path = tmp_path / "image.tiff"
    tifffile.imwrite(tiff_path, tiff_data, photometric="minisblack")
    tiff_catalog = api.catalog_viewer_2d(tiff_path)
    assert tiff_catalog.frame_labels == (0, 1, 2)
    _assert_canonical(api.read_viewer_2d_frame(tiff_catalog, 2), tiff_data[2])
    assert admissions[:2] == [0, 0]

    raw_data = np.arange(20, dtype=np.uint16).reshape(4, 5)
    raw_path = tmp_path / "detector.raw"
    raw_path.write_bytes(raw_data.tobytes())
    policy = api.Viewer2DFormatPolicy(raw_detector_shape=(4, 5), raw_dtype="uint16")
    raw_catalog = api.catalog_viewer_2d(raw_path, policy=policy)
    _assert_canonical(api.read_viewer_2d_frame(raw_catalog, 0, policy=policy), raw_data)
    assert admissions[2:4] == [0, 0]

    h5_data = np.arange(2 * 4 * 5, dtype=np.int32).reshape(2, 4, 5)
    h5_path = tmp_path / "master.nxs"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("entry/instrument/detector/data", data=h5_data)
    h5_catalog = api.catalog_viewer_2d(h5_path)
    assert h5_catalog.dataset_path == "/entry/instrument/detector/data"
    assert h5_catalog.frame_labels == (0, 1)
    _assert_canonical(api.read_viewer_2d_frame(h5_catalog, 1), h5_data[1])
    assert all(value == 0 for value in admissions[4:])

    fabio = pytest.importorskip("fabio")
    edf_path = tmp_path / "image.edf"
    fabio.edfimage.EdfImage(data=raw_data).write(edf_path)
    edf_catalog = api.catalog_viewer_2d(edf_path)
    _assert_canonical(api.read_viewer_2d_frame(edf_catalog, 0), raw_data)
    assert admissions[-2:] == [edf_path.stat().st_size] * 2


@pytest.mark.parametrize("family", ["raw", "tiff"])
def test_native_detector_decoders_have_no_shared_fallback(tmp_path, monkeypatch, family):
    from xrd_tools.io import image

    value = np.arange(12, dtype=np.uint16).reshape(3, 4)
    path = tmp_path / f"native.{family}"
    policy = api.Viewer2DFormatPolicy(raw_detector_shape=(3, 4), raw_dtype="uint16")
    _write_selected_source(path, family, value)
    catalog = api.catalog_viewer_2d(path, policy=policy)
    monkeypatch.setattr(image, "read_image", lambda *a, **k: pytest.fail("shared fallback"))
    _assert_canonical(api.read_viewer_2d_frame(catalog, 0, policy=policy), value)


def test_csv_replacement_between_f1_and_f2_refuses_before_allocation(tmp_path, monkeypatch):
    path = tmp_path / "matrix.csv"
    path.write_bytes(b"1,2\n3,4\n")
    catalog = api.catalog_viewer_2d(path)
    original_scan = api._csv_scan
    allocations = []
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(b"9,8\n7,6\n")

    def scan(stream, target=None):
        result = original_scan(stream, target)
        if target is None:
            os.replace(replacement, path)
        return result

    monkeypatch.setattr(api, "_csv_scan", scan)
    monkeypatch.setattr(np, "empty", lambda *a, **k: allocations.append(1))
    _assert_changed(api.read_viewer_2d_frame, catalog, 0)
    assert allocations == []


def test_npz_catalog_and_frame_each_use_one_closed_stable_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "single.npz"
    expected = np.arange(12, dtype=np.int16).reshape(3, 4)
    _npz(path, [("image.npy", _npy_bytes(expected))])
    real_open = open
    opened = []

    def tracked(value, *args, **kwargs):
        stream = real_open(value, *args, **kwargs)
        if Path(value) == path:
            opened.append(stream)
        return stream

    monkeypatch.setattr(api, "open", tracked, raising=False)
    catalog = api.catalog_viewer_2d(path)
    assert len(opened) == 1 and opened[0].closed
    opened.clear()
    _assert_canonical(api.read_viewer_2d_frame(catalog, 0), expected)
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.parametrize(
    "axis", ["object", "external", "path", "candidate", "name-plus-one",
             "depth-32", "depth-33"],
    ids=("existing-hdf-object-cap", "existing-hdf-external-cap",
         "existing-hdf-path-cap", "existing-hdf-candidate-cap",
         "r32-f05-name-retention-cap-plus-one",
         "r30-f05-depth-max", "r30-f05-depth-cap-plus-one"),
)
def test_hdf_census_and_depth_caps_refuse_before_cap_plus_one_retention(tmp_path, monkeypatch, axis):
    walk = api._HdfWalk()
    if axis in {"object", "external", "path", "candidate"}:
        setattr(walk, {"object": "visits", "external": "external",
                       "path": "path_bytes", "candidate": "candidates"}[axis],
                4 * 1024**2 if axis == "path" else 4096 if axis != "candidate" else 256)
        before = walk.visits, walk.external, walk.path_bytes, walk.candidates
        _raises(api.Viewer2DReadError,
                walk.retain_candidate if axis == "candidate" else walk.touch,
                **({} if axis == "candidate" else {"path": "x", "external": axis == "external"}))
        assert (walk.visits, walk.external, walk.path_bytes, walk.candidates) == before
        return
    h5py = pytest.importorskip("h5py")
    if axis == "name-plus-one":
        path = tmp_path / "name-retention.nxs"
        _write_hdf_stack(path, np.arange(6).reshape(2, 3), dataset="data")
        decoded, original_decode = [], h5py.Group._d

        def decode(group, name):
            decoded.append(name)
            return original_decode(group, name)

        with h5py.File(path) as handle:
            monkeypatch.setattr(h5py.Group, "_d", decode)
            walk.path_bytes = 4 * 1024**2
            before = walk.visits, walk.external, walk.path_bytes, walk.candidates
            _assert_limit(api._hdf_dataset, handle, walk)
            assert decoded == []
            assert (walk.visits, walk.external, walk.path_bytes, walk.candidates) == before
        return
    path = tmp_path / f"{axis}.nxs"
    with h5py.File(path, "w") as handle:
        group = handle
        for index in range(int(axis.split("-")[1])):
            group = group.create_group(f"g{index}")
        selected = group.create_dataset("data", data=np.ones((2, 2))).name
    with h5py.File(path) as handle:
        if axis == "depth-32":
            assert api._hdf_dataset(handle, walk, selected)[0] == selected
        else:
            original_get = h5py.Group.get

            def get(group, name, *args, **kwargs):
                if group.name.rstrip("/") + "/" + name == selected.rsplit("/", 1)[0]:
                    pytest.fail("depth-cap-plus-one group dereferenced before refusal")
                return original_get(group, name, *args, **kwargs)

            monkeypatch.setattr(h5py.Group, "get", get)
            _assert_limit(api._hdf_dataset, handle, walk, selected)


@pytest.mark.parametrize(
    "selected", ["dataset", "pointer", "thumbnail", "thumbnail_mask", "eiger_link",
                 "path-max", "path-plus-one"],
    ids=lambda selected: f"r30-f05-selected-{selected}",
)
def test_selected_hdf_lookup_charges_before_materialization(tmp_path, monkeypatch, selected):
    h5py = pytest.importorskip("h5py")
    if selected.startswith("path-"):
        path = tmp_path / f"selected-{selected}.nxs"
        _write_hdf_stack(path, np.arange(6).reshape(1, 2, 3))
        with h5py.File(path) as handle:
            walk = api._HdfWalk()
            full = "/entry/data/data"
            if selected == "path-max":
                walk.path_bytes = 4 * 1024**2 - len(full.encode())
                owner = handle["entry/data"]
                assert api._hget(owner, "data", walk).name == full
            else:
                walk.path_bytes = 4 * 1024**2 - len(full.encode()) + 1
                before = walk.visits, walk.external, walk.path_bytes, walk.candidates
                owner = handle["entry/data"]
                _assert_limit(api._hget, owner, "data", walk)
                assert (walk.visits, walk.external, walk.path_bytes, walk.candidates) == before
        return
    path = tmp_path / f"selected-{selected}.nxs"
    if selected in {"dataset", "pointer"}:
        _write_hdf_stack(path, np.arange(6).reshape(1, 2, 3))
        catalog = api.catalog_viewer_2d(path)
        selected_path = catalog.dataset_path
        if selected == "pointer":
            raw, path = path, tmp_path / "pointer.nxs"
            with h5py.File(path, "w") as handle:
                _processed_source(handle, 0, raw.name, 0, "/entry/data/data")
            catalog = api.catalog_viewer_2d(path)
            selected_path = "/entry/frames/frame_0000/source/path"
    elif selected == "eiger_link":
        segment = tmp_path / "data_000001.h5"
        _write_hdf_stack(segment, np.arange(6).reshape(1, 2, 3))
        _external_master(path, [("data_000001", segment)])
        catalog = api.catalog_viewer_2d(path)
        selected_path = catalog.dependencies[0].logical_path
    elif selected in {"thumbnail", "thumbnail_mask"}:
        with h5py.File(path, "w") as handle:
            _processed_thumbnail(
                handle, 0, np.arange(6, dtype=np.uint8).reshape(2, 3),
                vmin=0.0, vmax=5.0, mask=np.zeros((2, 3), dtype=bool))
        catalog = api.catalog_viewer_2d(path)
        selected_path = ("/entry/frames/frame_0000/thumbnail" if selected == "thumbnail"
                         else "/entry/frames/frame_0000/thumbnail_mask")

    monkeypatch.setattr(api, "_catalog_hdf5", lambda *a, **k: catalog)
    materialized = []
    _block_hdf_materialization(monkeypatch, h5py, selected_path, materialized)
    original_get = h5py.Group.get
    dereferenced = []

    def get(group, name, *args, **kwargs):
        if (group.name.rstrip("/") + "/" + name == selected_path
                and not kwargs.get("getlink")):
            dereferenced.append(selected_path)
        return original_get(group, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "get", get)
    original = api._HdfWalk.touch

    def touch(walk, name, **kwargs):
        if name == selected_path:
            setattr(walk, "external" if selected == "eiger_link" else "visits", 4096)
        return original(walk, name, **kwargs)

    monkeypatch.setattr(api._HdfWalk, "touch", touch)
    _assert_limit(api.read_viewer_2d_frame, catalog, 0)
    assert materialized == []
    assert dereferenced == []


@pytest.mark.parametrize(
    "count,entry_size,refused", [(1, 4096, False), (32, 4096, False),
                                 (1, 4097, True), (33, 7, True)],
    ids=("r30-f05-attribute-4k", "r30-f05-attribute-128k",
         "r30-f05-attribute-4k-plus-one", "r30-f05-attribute-128k-plus-one"),
)
def test_r30_hdf_attribute_storage_is_charged_before_value_read(tmp_path, monkeypatch,
                                                               count, entry_size, refused):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "attributes.nxs"
    with h5py.File(path, "w") as handle:
        for index in range(count):
            group = handle.create_group(f"a{index:02d}")
            size = (4090 if count == 33 and index == 31 else
                    entry_size if index == count - 1 else 4096)
            value = b"NXentry" if index == count - 1 else b"x" * size
            group.attrs.create("NX_class", np.array(value, dtype=f"S{size}"))
            if index == count - 1:
                group.create_dataset("data/data", data=np.arange(6).reshape(1, 2, 3))
    _, opens, reads, charged, inspected = _watch_hdf_attributes(monkeypatch, h5py)
    error = None
    if not refused:
        assert api.catalog_viewer_2d(path).frame_labels == (0,)
    else:
        error = _raises(api.Viewer2DReadError, api.catalog_viewer_2d, path)
    expected = count if not refused else 0 if entry_size > 4096 else 32
    owners = [f"/a{index:02d}" for index in range(count)]
    assert opens == owners and [owner for owner, _ in reads] == owners[:expected]
    if error is not None:
        assert error.code is api.Viewer2DRefusalCode.LIMIT_EXCEEDED
        assert charged == [True] * count
        assert all({"space", "dtype", "storage"} <= checks for _, checks in reads)
        assert {"space", "dtype", "storage"} <= inspected.get(owners[-1], set())


@pytest.mark.parametrize("kind", ["non-scalar", "reference", "object-member"],
                         ids=lambda kind: f"r32-f05-attribute-{kind}")
def test_r32_hdf_invalid_attribute_is_inspected_without_value_read(tmp_path, monkeypatch, kind):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / f"attribute-{kind}.nxs"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("a_candidate")
        group.create_dataset("data", data=np.arange(6).reshape(1, 2, 3))
        if kind == "non-scalar":
            group.attrs.create("NX_class", np.array([b"NXentry"]))
        elif kind == "reference":
            group.attrs.create("NX_class", group.ref, dtype=h5py.ref_dtype)
        else:
            group.attrs.create("NX_class", np.array((b"NXentry",),
                dtype=np.dtype([("value", h5py.string_dtype())])))
    _, opens, reads, charged, inspected = _watch_hdf_attributes(monkeypatch, h5py)
    _assert_refusal(api.Viewer2DRefusalCode.FORMAT_INVALID, api.catalog_viewer_2d, path)
    assert opens == ["/a_candidate"] and charged == [True] and reads == []
    assert {"space", "dtype", "storage"} <= inspected["/a_candidate"]


def test_r30_hdf_selected_anchor_reread_charges_links_without_candidate_retain(tmp_path):
    h5py = pytest.importorskip("h5py")
    path, target = tmp_path / "links.nxs", tmp_path / "target.nxs"
    _write_hdf_stack(target, np.arange(6).reshape(1, 2, 3))
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_dataset("hard", data=np.arange(6).reshape(1, 2, 3))
        entry["soft"] = h5py.SoftLink("/entry/hard")
        entry["external"] = h5py.ExternalLink(target.name, "/entry/data/data")
    with h5py.File(path) as handle:
        walk = api._HdfWalk()
        walk.retain_candidate()
        for link_kind in ("hard", "soft", "external"):
            before = (walk.visits, walk.path_bytes, walk.external, walk.candidates)
            api._hpath(handle, f"/entry/{link_kind}", walk)
            after = (walk.visits, walk.path_bytes, walk.external, walk.candidates)
            expected = (2, len(b"/entry") + len(f"/entry/{link_kind}".encode()),
                        int(link_kind == "external"), 0)
            assert tuple(b - a for a, b in zip(before, after)) == expected
        before = (walk.visits, walk.path_bytes, walk.external, walk.candidates)
        api._hpath(handle, "/entry/external", walk)
        after = (walk.visits, walk.path_bytes, walk.external, walk.candidates)
    assert tuple(b - a for a, b in zip(before, after)) == expected
    assert after[3] == 1


def test_tiff_frame_limit_is_checked_before_page_iteration(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "many.tiff"
    path.write_bytes(b"placeholder")

    class Pages:
        def __len__(self): return 10_001
        def __iter__(self): pytest.fail("pages retained before count cap")

    class Handle:
        pages = Pages()
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(tifffile, "TiffFile", lambda value: Handle())
    _assert_limit(api.catalog_viewer_2d, path)


def test_processed_gapped_labels_preserve_raw_or_explicit_thumbnail_provenance(tmp_path):
    h5py = pytest.importorskip("h5py")
    raw_path = tmp_path / "raw.nxs"
    raw = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    _write_hdf_stack(raw_path, raw)

    processed = tmp_path / "processed.nxs"
    with h5py.File(processed, "w") as handle:
        _processed_source(handle, 2, raw_path.name, 1, "/entry/data/data")
        quantized = np.array([[0, 64, 128], [255, 32, 16]], dtype=np.uint8)
        mask = np.array([[False, True, False], [False, False, False]], dtype=bool)
        _processed_thumbnail(handle, 7, quantized, vmin=10.0, vmax=20.0, mask=mask)

    catalog = api.catalog_viewer_2d(processed)
    assert catalog.frame_labels == (2, 7)
    assert catalog.source_dtype == catalog.frame_facts[0].dtype == np.dtype("uint16").str
    assert catalog.frame_facts[1].dtype == np.dtype("uint8").str
    raw_result = api.read_viewer_2d_frame(catalog, 2)
    _assert_canonical(raw_result, raw[1])
    assert raw_result.provenance.source_kind is api.Viewer2DSourceKind.PROCESSED_RAW
    assert raw_result.provenance.raw_locator == str(raw_path.resolve())
    assert raw_result.provenance.dataset_path == "/entry/data/data"
    assert raw_result.provenance.raw_source_frame == 1
    assert raw_result.provenance.dependencies

    thumb_result = api.read_viewer_2d_frame(catalog, 7)
    expected = 10.0 + quantized.astype(float) / 255.0 * 10.0
    expected[0, 1] = np.nan
    _assert_canonical(thumb_result, expected)
    assert thumb_result.provenance.source_kind is api.Viewer2DSourceKind.PROCESSED_THUMBNAIL
    assert thumb_result.provenance.source_dtype == np.dtype("uint8").str
    assert thumb_result.provenance.source_nbytes == quantized.nbytes
    assert thumb_result.provenance.source_sha256 == hashlib.sha256(
        quantized.tobytes(order="C")
    ).hexdigest()
    assert thumb_result.provenance.degraded_thumbnail
    assert thumb_result.provenance.diagnostic == "Thumbnail preview; raw source unavailable."


@pytest.mark.parametrize("family", ["raw", "tiff", "edf", "cbf"])
def test_processed_raw_keeps_non_hdf_detector_parity(tmp_path, family):
    h5py = pytest.importorskip("h5py")
    value = np.arange(12, dtype=np.int32).reshape(3, 4)
    source = tmp_path / f"source.{family}"
    _write_selected_source(source, "fabio" if family == "edf" else family, value)
    processed = tmp_path / f"processed-{family}.nxs"
    with h5py.File(processed, "w") as handle:
        _processed_source(handle, 5, source.name, 0)
    policy = api.Viewer2DFormatPolicy(raw_detector_shape=(3, 4), raw_dtype="int32")
    catalog = api.catalog_viewer_2d(processed, policy=policy)
    result = api.read_viewer_2d_frame(catalog, 5, policy=policy)
    _assert_canonical(result, value)
    assert result.provenance.source_kind is api.Viewer2DSourceKind.PROCESSED_RAW
    assert result.provenance.raw_locator == str(source.resolve())
    assert result.provenance.dataset_path is None
    assert result.provenance.raw_source_frame == 0


@pytest.mark.parametrize("locality", ["provenance", "unselected-drift"])
def test_eiger_external_segments_use_interval_local_identity_and_detect_drift(tmp_path, locality):
    h5py = pytest.importorskip("h5py")
    base = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    master, segments, arrays = _eiger_master(tmp_path, [
        (f"data_{number:06d}.h5", base + number * 100) for number in (1, 2)])

    catalog = api.catalog_viewer_2d(master)
    assert catalog.frame_labels == (0, 1, 2, 3)
    result = api.read_viewer_2d_frame(catalog, 2)
    _assert_canonical(result, arrays[1][0])
    assert result.provenance.raw_locator == str(segments[1].resolve())
    assert result.provenance.dataset_path == "/entry/data/data"
    assert result.provenance.raw_source_frame == 0
    if locality == "provenance":
        assert result.provenance.dependencies == (catalog.dependencies[1],)
    else:
        with h5py.File(segments[0], "r+") as handle:
            handle["entry/data/data"][0, 0, 0] += 1
        inert = api.read_viewer_2d_frame(catalog, 2)
        _assert_canonical(inert, arrays[1][0])
        assert inert.provenance.dependencies == (catalog.dependencies[1],)

    with h5py.File(segments[1], "r+") as handle:
        handle["entry/data/data"][0, 0, 0] += 1
    _assert_changed(api.read_viewer_2d_frame, catalog, 2)


def test_one_hdf_census_spans_processed_recursion_and_pre_read_post(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    raw = tmp_path / "raw.nxs"
    _write_hdf_stack(raw, np.arange(6).reshape(1, 2, 3))
    processed = tmp_path / "processed.nxs"
    with h5py.File(processed, "w") as handle:
        _processed_source(handle, 0, raw.name, 0, "/entry/data/data")

    catalog = api.catalog_viewer_2d(processed)
    seen = set()
    original = api._HdfWalk.touch

    def touch(self, *args, **kwargs):
        seen.add(id(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(api._HdfWalk, "touch", touch)
    _assert_canonical(api.read_viewer_2d_frame(catalog, 0), np.arange(6).reshape(2, 3))
    assert len(seen) == 1


def test_processed_651_catalog_uses_owned_frame_bound_and_hashes_each_file_once(
        tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    raw = np.arange(651 * 2 * 3, dtype=np.uint16).reshape(651, 2, 3)
    master, segments, _ = _eiger_master(
        tmp_path / "raw", [("data_000001.h5", raw)])
    processed = tmp_path / "processed.nexus"
    with h5py.File(processed, "w") as handle:
        for label in range(651):
            _processed_source(
                handle, label, str(master.resolve()), label,
                "/entry/data/data_000001")

    calls = []
    original = api._stable_revision

    def stable(path):
        calls.append(str(Path(path).resolve()))
        return original(path)

    monkeypatch.setattr(api, "_stable_revision", stable)
    catalog = api.catalog_viewer_2d(processed)

    assert catalog.frame_labels == tuple(range(651))
    assert catalog.frame_facts[0].source_frame == 0
    assert catalog.frame_facts[-1].source_frame == 650
    assert sorted(calls) == sorted({
        str(processed.resolve()), str(master.resolve()), str(segments[0].resolve()),
    })


def test_processed_651_pure_thumbnail_catalog_uses_owned_field_budget(tmp_path):
    h5py = pytest.importorskip("h5py")
    processed = tmp_path / "processed.nexus"
    thumbnail = np.arange(6, dtype=np.uint8).reshape(2, 3)
    with h5py.File(processed, "w") as handle:
        for label in range(651):
            _processed_thumbnail(
                handle, label, thumbnail, vmin=0.0, vmax=5.0)

    catalog = api.catalog_viewer_2d(processed)

    assert catalog.frame_labels == tuple(range(651))
    assert {fact.source_kind for fact in catalog.frame_facts} == {
        api.Viewer2DSourceKind.PROCESSED_THUMBNAIL,
    }


def test_processed_651_missing_source_falls_back_to_all_thumbnails(tmp_path):
    h5py = pytest.importorskip("h5py")
    processed = tmp_path / "processed.nexus"
    missing = tmp_path / "missing-master.h5"
    thumbnail = np.arange(6, dtype=np.uint16).reshape(2, 3)
    with h5py.File(processed, "w") as handle:
        for label in range(651):
            _processed_source(
                handle, label, str(missing.resolve()), label,
                "/entry/data/data")
            _processed_thumbnail(
                handle, label, thumbnail, vmin=0.0, vmax=5.0)

    catalog = api.catalog_viewer_2d(processed)

    assert catalog.frame_labels == tuple(range(651))
    assert {fact.source_kind for fact in catalog.frame_facts} == {
        api.Viewer2DSourceKind.PROCESSED_THUMBNAIL,
    }


def test_processed_fixed_name_misses_do_not_consume_frame_catalog_cap(
        tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    raw = tmp_path / "raw.h5"
    value = np.arange(6, dtype=np.uint16).reshape(1, 2, 3)
    _write_hdf_stack(raw, value)
    processed = tmp_path / "processed.nexus"
    with h5py.File(processed, "w") as handle:
        entry = handle.require_group("entry")
        entry.require_group("integrated_1d")
        for label in (1, 2, 3):
            source = entry.require_group(
                f"frames/frame_{label:05d}/source")
            source.create_dataset("path", data=np.bytes_(str(raw.resolve())))
            source.create_dataset("frame_index", data=0)
            source.attrs["dataset_path"] = "/entry/data/data"
    monkeypatch.setattr(api, "_MAX_FRAMES", 3)

    catalog = api.catalog_viewer_2d(processed)
    frame = api.read_viewer_2d_frame(catalog, 3)

    assert catalog.frame_labels == (1, 2, 3)
    _assert_canonical(frame, value[0])


def test_explicit_hdf_dataset_hint_bypasses_generic_foreign_census(
        tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "selected.nxs"
    value = np.arange(6).reshape(1, 2, 3)
    with h5py.File(path, "w") as handle:
        for index in range(2100):
            handle.create_group(f"foreign_{index:04d}")
        handle.create_dataset("selected", data=value)

    catalog = api._catalog_resolved(
        path.resolve(), api.Viewer2DFormatPolicy(), preferred="/selected")

    assert catalog.dataset_path == "/selected"
    assert catalog.frame_labels == (0,)
    _assert_canonical(api.read_viewer_2d_frame(catalog, 0), value[0])


def test_processed_external_preferred_path_bypasses_foreign_master_census(tmp_path):
    h5py = pytest.importorskip("h5py")
    value = np.arange(2 * 2 * 3, dtype=np.uint16).reshape(2, 2, 3)
    segment = tmp_path / "segment.h5"
    _write_hdf_stack(segment, value)
    master = tmp_path / "master.h5"
    with h5py.File(master, "w") as handle:
        for index in range(2100):
            handle.create_group(f"foreign_{index:04d}")
        handle["selected"] = h5py.ExternalLink(
            segment.name, "/entry/data/data")
    processed = tmp_path / "processed.nexus"
    with h5py.File(processed, "w") as handle:
        _processed_source(
            handle, 7, str(master.resolve()), 1, "/selected")

    catalog = api.catalog_viewer_2d(processed)
    frame = api.read_viewer_2d_frame(catalog, 7)

    assert catalog.frame_labels == (7,)
    assert catalog.frame_facts[0].dataset_path == "/selected"
    assert {dependency.locator for dependency in catalog.dependencies} == {
        str(master.resolve()), str(segment.resolve()),
    }
    _assert_canonical(frame, value[1])


@pytest.mark.parametrize("drift", ["before", "during"])
def test_eiger_frame_read_uses_stat_fences_and_rehashes_only_after_drift(
        tmp_path, monkeypatch, drift):
    h5py = pytest.importorskip("h5py")
    value = np.arange(2 * 2 * 3, dtype=np.uint16).reshape(2, 2, 3)
    master, segments, _ = _eiger_master(
        tmp_path, [("data_000001.h5", value)])
    catalog = api.catalog_viewer_2d(master)
    calls = []
    original = api._stable_revision

    def stable(path):
        calls.append(str(Path(path).resolve()))
        return original(path)

    monkeypatch.setattr(api, "_stable_revision", stable)
    _assert_canonical(api.read_viewer_2d_frame(catalog, 1), value[1])
    assert calls == []

    def mutate():
        with h5py.File(segments[0], "r+") as handle:
            handle["entry/data/data"][1, 0, 0] += 1

    if drift == "before":
        mutate()
    else:
        original_read = api._read_hdf_dataset

        def read_then_mutate(*args, **kwargs):
            result = original_read(*args, **kwargs)
            mutate()
            return result

        monkeypatch.setattr(api, "_read_hdf_dataset", read_then_mutate)
    _assert_changed(api.read_viewer_2d_frame, catalog, 1)
    assert calls == [str(segments[0].resolve())]


def _independent_ledger(canonical, *, encoded=17, reservation=_EXPECTED_R,
                        budget=_EXPECTED_B):
    reader = max(3 * canonical, encoded + 4 * canonical)
    concurrent = 6 * canonical
    total = reservation + max(reader, concurrent)
    return api.Viewer2DMemoryLedger(
        canonical, reservation, encoded, reader, 5 * canonical // 2,
        7 * canonical // 2, 9 * canonical // 2, 5 * canonical,
        5 * canonical, concurrent, total, canonical + total, budget,
    )


def _rehashed_catalog(catalog, **change):
    values = {item.name: getattr(catalog, item.name) for item in fields(catalog)}
    values.update(change)
    manifest = (
        values["canonical_path"], values["source_kind"].value,
        values["frame_labels"], values["source_shape"], values["source_dtype"],
        values["dependencies"], values["format_name"], values["member_name"],
        values["dataset_path"], values["frame_facts"], values["policy_identity"],
        values["primary_revision"],
    )
    values["catalog_identity"] = hashlib.sha256(b"\0".join(
        str(part).encode("utf-8", "strict") for part in manifest
    )).hexdigest()
    return api.Viewer2DArtifactCatalog(**values)


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in fields(value):
        member = changes.get(field.name, getattr(value, field.name))
        object.__setattr__(forged, field.name, member)
    return forged


def _block_hdf_materialization(monkeypatch, h5py, target, materialized):
    original = h5py.Dataset.__getitem__

    def getitem(dataset, key):
        if dataset.name == target:
            materialized.append(key)
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem)


def _write_selected_source(path, family, value):
    if family == "hdf":
        _write_hdf_stack(path, value[np.newaxis])
    elif family == "raw":
        path.write_bytes(value.tobytes())
    elif family == "tiff":
        pytest.importorskip("tifffile").imwrite(path, value)
    elif family == "fabio":
        pytest.importorskip("fabio").edfimage.EdfImage(data=value).write(path)
    elif family == "cbf":
        pytest.importorskip("fabio").cbfimage.CbfImage(data=value).write(path)
    else:
        h5py = pytest.importorskip("h5py")
        with h5py.File(path, "w") as handle:
            _processed_thumbnail(
                handle, 0, value, vmin=0.0, vmax=float(value.max()))


def _block_selected_materialization(monkeypatch, family, target, materialized):
    if family in {"hdf", "thumbnail"}:
        h5py = pytest.importorskip("h5py")
        _block_hdf_materialization(monkeypatch, h5py, target, materialized)
    elif family == "tiff":
        tifffile = pytest.importorskip("tifffile")
        original = tifffile.TiffPage.asarray

        def materialize(page, *args, **kwargs):
            materialized.append(page.shape)
            return original(page, *args, **kwargs)

        monkeypatch.setattr(tifffile.TiffPage, "asarray", materialize)
    else:
        fabio = pytest.importorskip("fabio")
        original = fabio.open

        def materialize(*args, **kwargs):
            materialized.append("fabio.open")
            return original(*args, **kwargs)

        monkeypatch.setattr(fabio, "open", materialize)


@pytest.mark.parametrize(
    "axis",
    ["generic-foreign", "generic-external", "cross-chain-overlap", "missing-selected",
     "foreign-chain", "eiger-foreign"],
    ids=lambda axis: f"r30-f01-{axis}",
)
def test_r30_hdf_dependency_grammar_is_locator_local(tmp_path, axis):
    h5py = pytest.importorskip("h5py")
    shape = (2, 2, 3)
    if axis == "generic-foreign":
        primary, foreign = tmp_path / "primary.nxs", tmp_path / "foreign.nxs"
        _write_hdf_stack(primary, np.arange(12).reshape(shape))
        _write_hdf_stack(foreign, np.arange(12).reshape(shape) + 100)
        catalog = api.catalog_viewer_2d(primary)
        dependency = api.Viewer2DDependency(
            str(foreign.resolve()), catalog.dataset_path,
            api._stable_revision(foreign),
        )
        _raises(TypeError, _rehashed_catalog, catalog, dependencies=(dependency,))
        return

    if axis == "generic-external":
        target, primary = tmp_path / "payload.h5", tmp_path / "external.nxs"
        value = np.arange(12).reshape(shape)
        _write_hdf_stack(target, value)
        _external_master(primary, [("signal", target)])
        catalog = api.catalog_viewer_2d(primary)
        dependency, = catalog.dependencies
        assert dependency == api.Viewer2DDependency(
            str(target.resolve()), "/entry/data/data", api._stable_revision(target),
            0, 2, "/entry/data/signal")
        result = api.read_viewer_2d_frame(catalog, 1)
        _assert_canonical(result, value[1])
        assert (result.provenance.raw_locator, result.provenance.dataset_path,
                result.provenance.raw_source_frame, result.provenance.dependencies) == (
                    str(target.resolve()), "/entry/data/data", 1, (dependency,))
        return

    if axis == "eiger-foreign":
        master, _, _ = _eiger_master(
            tmp_path, [("data_000001", np.arange(12).reshape(shape))])
        catalog = api.catalog_viewer_2d(master)
        dependency = catalog.dependencies[0]
        forged = _forge(
            dependency, logical_path="/entry/data/not_the_selected_link",
        )
        _raises(TypeError, _rehashed_catalog, catalog, dependencies=(forged,))
        return

    bases = [
        _eiger_master(
            tmp_path / stem, [("data_000001", np.arange(12).reshape(shape) + offset)])[0]
        for stem, offset in (("a", 0), ("b", 100))
    ]
    processed = tmp_path / "processed.nxs"
    with h5py.File(processed, "w") as handle:
        _processed_source(handle, 1, str(bases[0].resolve()), 1)
        if axis == "cross-chain-overlap":
            _processed_source(handle, 2, str(bases[1].resolve()), 1)
    if axis == "cross-chain-overlap":
        # Both source catalogs use numeric interval [0, 2); this is valid
        # because each interval belongs to a distinct base-master chain.
        catalog = api.catalog_viewer_2d(processed)
        chains = []
        for base in bases:
            source_catalog = api.catalog_viewer_2d(base)
            chains.append((api.Viewer2DDependency(
                source_catalog.canonical_path, source_catalog.dataset_path,
                source_catalog.primary_revision), *source_catalog.dependencies))
        assert catalog.dependencies == chains[0] + chains[1]
        for chain, (label, offset) in enumerate(((1, 0), (2, 100))):
            result = api.read_viewer_2d_frame(catalog, label)
            _assert_canonical(result, np.arange(12).reshape(shape)[1] + offset)
            assert result.provenance.raw_locator == str(bases[chain].resolve())
            assert result.provenance.dataset_path == "/entry/data/data"
            assert result.provenance.raw_source_frame == 1
            assert result.provenance.dependencies == chains[chain]
        return
    catalog = api.catalog_viewer_2d(processed)
    base, selected = catalog.dependencies
    if axis == "missing-selected":
        dependencies = (base,)
    else:
        foreign_catalog = api.catalog_viewer_2d(bases[1])
        foreign = foreign_catalog.dependencies[0]
        dependencies = (base, foreign)
        assert foreign.locator != selected.locator and foreign.frame_start == 0
    _raises(TypeError, _rehashed_catalog, catalog, dependencies=dependencies)


@pytest.mark.parametrize(
    "axis", ["tiff-as-edf", "raw-rank-three", "raw-transpose", "raw-dtype",
             "raw-header", "fabio-oversize"],
    ids=lambda axis: f"r30-f02-{axis}",
)
def test_r30_decoder_matrix_refuses_forgery_before_decode(tmp_path, monkeypatch, axis):
    value = np.arange(12, dtype=np.uint16).reshape(3, 4)
    if axis == "tiff-as-edf":
        path = tmp_path / "image.tiff"
        _write_selected_source(path, "tiff", value)
        catalog = api.catalog_viewer_2d(path)
        _raises(TypeError, _rehashed_catalog, catalog, format_name="edf")
        return
    if axis.startswith("raw-"):
        path = tmp_path / "image.raw"
        _write_selected_source(path, "raw", value)
        policy = api.Viewer2DFormatPolicy(raw_detector_shape=value.shape, raw_dtype="uint16")
        catalog = api.catalog_viewer_2d(path, policy=policy)
        if axis == "raw-rank-three":
            _raises(TypeError, _rehashed_catalog, catalog,
                    source_shape=(2, *value.shape), frame_labels=(0, 1))
        else:
            values = {"raw_detector_shape": value.shape, "raw_dtype": "uint16"}
            values.update({
                "raw-transpose": {"raw_detector_shape": value.shape[::-1]},
                "raw-dtype": {"raw_dtype": "uint32"},
                "raw-header": {"raw_header_skip": 1},
            }[axis])
            forged_policy = api.Viewer2DFormatPolicy(**values)
            try:
                forged = _rehashed_catalog(catalog, policy_identity=forged_policy.identity)
            except TypeError:
                return
            monkeypatch.setattr(np, "frombuffer", pytest.fail)
            _assert_refusal(api.Viewer2DRefusalCode.FORMAT_INVALID,
                api.read_viewer_2d_frame, forged, 0, policy=forged_policy)
        return
    fabio = pytest.importorskip("fabio")
    path, replacement = tmp_path / "small.edf", tmp_path / "large.edf"
    fabio.edfimage.EdfImage(data=value).write(path)
    catalog = api.catalog_viewer_2d(path)
    replacement.write_bytes(path.read_bytes())
    with replacement.open("r+b") as stream:
        stream.truncate(256 * 1024**2 + 1)
    os.replace(replacement, path)
    forged = _rehashed_catalog(catalog, primary_revision=api._stable_revision(path))
    monkeypatch.setattr(fabio, "open", pytest.fail)
    _assert_limit(api.read_viewer_2d_frame, forged, 0)


@pytest.mark.parametrize(
    "axis", ["hdf-shape", "hdf-dtype", "tiff-shape", "tiff-dtype",
             "fabio-shape", "fabio-dtype", "thumbnail-shape",
             "thumbnail-dataset-dtype", "thumbnail-lut-dtype"],
    ids=lambda axis: f"r30-f03-{axis}",
)
def test_r30_selected_metadata_mismatch_refuses_before_materialization(tmp_path, monkeypatch, axis):
    materialized = []
    family, drift = axis.split("-", 1)
    suffix = {"hdf": "nxs", "tiff": "tiff", "fabio": "edf",
              "thumbnail": "nxs"}[family]
    path = tmp_path / f"selected.{suffix}"
    source_dtype = np.uint8 if family == "thumbnail" else np.int16
    initial = np.arange(6, dtype=source_dtype).reshape(2, 3)
    if drift == "shape":
        replacement = np.arange(9, dtype=source_dtype).reshape(3, 3)
    elif family == "thumbnail" and drift == "lut-dtype":
        replacement = initial.copy()
    else:
        replacement = np.arange(6, dtype=(np.uint16 if family == "thumbnail"
                                else np.int32)).reshape(2, 3)
    _write_selected_source(path, family, initial)
    catalog = api.catalog_viewer_2d(path)
    if family == "thumbnail":
        fact = replace(catalog.frame_facts[0], dtype=np.dtype(source_dtype).str)
        catalog = _rehashed_catalog(catalog, source_dtype=fact.dtype, frame_facts=(fact,))
    _write_selected_source(path, family, replacement)
    if family == "thumbnail" and drift.endswith("dtype"):
        h5py = pytest.importorskip("h5py")
        with h5py.File(path, "r+") as handle:
            thumbnail = handle["/entry/frames/frame_0000/thumbnail"]
            thumbnail.attrs["dtype"] = ("uint16" if drift == "lut-dtype" else "uint8")
    target = (catalog.dataset_path if family == "hdf" else
              catalog.frame_facts[0].thumbnail_path if family == "thumbnail" else None)
    _block_selected_materialization(monkeypatch, family, target, materialized)
    forged = _rehashed_catalog(catalog, primary_revision=api._stable_revision(path))
    _assert_changed(api.read_viewer_2d_frame, forged, forged.frame_labels[0])
    assert materialized == []


@pytest.mark.parametrize(
    "axis", ["npy", "npz", "detector", "hdf-primary", "hdf-dependency", "hdf-base"],
    ids=("r30-f04-npy-inplace", "r30-f04-npz-inplace",
         "r30-f04-detector-primary-race", "r30-f04-hdf-primary-race",
         "r30-f04-hdf-retained-dependency-race", "r32-f04-hdf-base-race"),
)
def test_r30_catalog_revision_brackets_valid_metadata(tmp_path, monkeypatch, axis):
    entered = []
    if axis in {"npy", "npz"}:
        path, replacement = tmp_path / f"a.{axis}", tmp_path / f"b.{axis}"
        a, b = np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)
        if axis == "npy":
            np.save(path, a)
            np.save(replacement, b)
            monkeypatch.setattr(api, "_npy_header", _after_call(
                api._npy_header,
                lambda: path.write_bytes(replacement.read_bytes()), entered))
        else:
            _npz(path, [("image.npy", _npy_bytes(a))], compression=zipfile.ZIP_STORED)
            _npz(replacement, [("image.npy", _npy_bytes(b))], compression=zipfile.ZIP_STORED)
            monkeypatch.setattr(api, "_zip_scan", _after_call(
                api._zip_scan, lambda: path.write_bytes(replacement.read_bytes()),
                entered, lambda *a, **kwargs: not kwargs.get("prefix_cap")))
    elif axis in {"detector", "hdf-primary"}:
        suffix = "tiff" if axis == "detector" else "nxs"
        path, replacement = tmp_path / f"a.{suffix}", tmp_path / f"b.{suffix}"
        if axis == "detector":
            tifffile = pytest.importorskip("tifffile")
            tifffile.imwrite(path, np.arange(6).reshape(2, 3))
            tifffile.imwrite(replacement, np.arange(9).reshape(3, 3))
            original = tifffile.TiffPages.__iter__
            monkeypatch.setattr(tifffile.TiffPages, "__iter__", _after_call(
                lambda pages: list(original(pages)),
                lambda: os.replace(replacement, path), entered, transform=iter))
        else:
            h5py = pytest.importorskip("h5py")
            _write_hdf_stack(path, np.arange(6).reshape(1, 2, 3))
            _write_hdf_stack(replacement, np.arange(9).reshape(1, 3, 3))
            monkeypatch.setattr(api, "_hdf_dataset", _after_call(
                api._hdf_dataset, lambda: os.replace(replacement, path), entered))
    elif axis == "hdf-dependency":
        h5py = pytest.importorskip("h5py")
        segment, replacement = tmp_path / "segment.h5", tmp_path / "replacement.h5"
        path = tmp_path / "master.nxs"
        _write_hdf_stack(segment, np.arange(6).reshape(1, 2, 3))
        _write_hdf_stack(replacement, np.arange(6, 12).reshape(1, 2, 3))
        _external_master(path, [("data_000001", segment)])
        monkeypatch.setattr(api, "_dependency", _after_call(
            api._dependency, lambda: os.replace(replacement, segment), entered))
    else:
        h5py = pytest.importorskip("h5py")
        root = tmp_path / "base-race"
        master, _, _ = _eiger_master(
            root, [("data_000001", np.arange(6).reshape(1, 2, 3))])
        replacement = root / "replacement.nxs"
        _external_master(replacement, [("data_000001", root / "data_000001")])
        path = tmp_path / "processed-base-race.nxs"
        with h5py.File(path, "w") as handle:
            _processed_source(handle, 0, str(master.resolve()), 0)
        monkeypatch.setattr(api, "_catalog_resolved", _after_call(
            api._catalog_resolved, lambda: os.replace(replacement, master), entered,
            predicate=lambda candidate, *a, **k: Path(candidate) == master.resolve()))
    _assert_changed(api.catalog_viewer_2d, path)
    assert entered == ["metadata"]


@pytest.mark.parametrize(
    "axis", ["nested-revision", "negative-source-frame", "raw-format",
     "raw-dataset-type", "canonical-path-cap", "manifest-cap",
     "renderer-clear-sha", "renderer-clear-nested-request"],
    ids=("r30-f07-nested-dependency-revision",
         "r30-f07-negative-processed-source-frame",
         "r30-f07-raw-provenance-format", "r30-f07-raw-provenance-dataset-type",
         "r30-f07-canonical-path-cap", "r30-f07-manifest-cap",
         "r30-f07-renderer-clear-sha", "r30-f07-renderer-clear-nested-request"),
)
def test_r30_exported_values_recursively_close_isolated_forgery(tmp_path, axis):
    if axis == "nested-revision":
        path = tmp_path / "matrix.npy"
        np.save(path, np.arange(6).reshape(2, 3))
        revision = api.catalog_viewer_2d(path).primary_revision
        _raises(TypeError, api.Viewer2DDependency,
                revision.canonical_path, None, _forge(revision, sha256="x"))
    elif axis == "negative-source-frame":
        _raises(TypeError, api.Viewer2DFrameFact,
                0, api.Viewer2DSourceKind.PROCESSED_RAW, (2, 3), "<i8",
                "raw.nxs", source_frame=-1, source_catalog_identity="0" * 64)
    elif axis.startswith("raw-"):
        path = tmp_path / "image.raw"
        value = np.arange(6, dtype=np.int16).reshape(2, 3)
        path.write_bytes(value.tobytes())
        policy = api.Viewer2DFormatPolicy(raw_detector_shape=(2, 3), raw_dtype="int16")
        provenance = api.read_viewer_2d_frame(
            api.catalog_viewer_2d(path, policy=policy), 0, policy=policy).provenance
        change = ({"format_name": "csv"} if axis == "raw-format"
                  else {"dataset_path": object()})
        _raises(TypeError, replace, provenance, **change)
    elif axis in {"canonical-path-cap", "manifest-cap"}:
        path = tmp_path / ("matrix.npy" if axis == "canonical-path-cap" else "matrix.npz")
        if axis == "canonical-path-cap":
            np.save(path, np.arange(6).reshape(2, 3))
        else:
            _npz(path, [("image.npy", _npy_bytes(np.arange(6).reshape(2, 3)))])
        catalog = api.catalog_viewer_2d(path)
        change = ({"canonical_path": "/" + "x" * 4097,
                   "primary_revision": _forge(
                       catalog.primary_revision, canonical_path="/" + "x" * 4097)}
                  if axis == "canonical-path-cap" else
                  {"member_name": "x" * api._MAX_CATALOG_MANIFEST + ".npy"})
        _raises(TypeError, _rehashed_catalog, catalog, **change)
    else:
        dc = importlib.import_module("xdart.modules.display_context")
        request = dc.Viewer2DRendererClearRequest("viewer", 1, "0" * 64, None)
        operation = (dc.Viewer2DRendererClearRequest if axis == "renderer-clear-sha"
                     else dc.Viewer2DRendererClearReceipt)
        args = (("viewer", 1, "not-sha", None) if axis == "renderer-clear-sha"
                else (_forge(request, catalog_identity="not-sha"), True))
        _raises(TypeError, operation, *args)


def test_exported_values_reject_local_malformed_values(tmp_path):
    path = tmp_path / "matrix.npy"
    np.save(path, np.arange(6).reshape(2, 3))
    catalog = api.catalog_viewer_2d(path)
    frame = api.read_viewer_2d_frame(catalog, 0)
    ledger = api.viewer_2d_memory_ledger(2, 3, ram_bytes=10**12)

    _raises(TypeError, api.Viewer2DReadError, "format_invalid", "not an exact refusal code")
    _raises(TypeError, replace, catalog, frame_labels=(0, 0))
    _raises(TypeError, replace, catalog, canonical_path=str(tmp_path / "forged.npy"))
    _raises(TypeError, replace, frame.provenance, frame_index=-1)
    _raises(TypeError, replace, ledger, total=ledger.total + 1,
            admission=ledger.admission + 1)
    yielded = []

    def labels():
        for label in range(10_002):
            yielded.append(label)
            yield label

    caught = _raises(api.Viewer2DReadError, api._make_catalog,
            Path(catalog.canonical_path), api.Viewer2DFormatPolicy(),
            api.Viewer2DSourceKind.NUMPY_ARRAY, labels(), (10_000, 2, 3),
            np.dtype("int64"), catalog.primary_revision, (), "npy")
    assert caught.code is api.Viewer2DRefusalCode.LIMIT_EXCEEDED
    assert yielded == list(range(10_001))


@pytest.mark.parametrize(
    ("canonical", "reservation", "budget"),
    [(0, _EXPECTED_R, _EXPECTED_B), (10, _EXPECTED_R, _EXPECTED_B),
     (48, _EXPECTED_R + 1, _EXPECTED_B), (48, _EXPECTED_R, _EXPECTED_B - 1)],
    ids=("missing-ledger-zero-c", "missing-ledger-unaligned-c",
         "missing-ledger-wrong-r", "missing-ledger-wrong-b"),
)
def test_memory_ledger_rejects_each_isolated_invariant(canonical, reservation, budget):
    _raises(TypeError, _independent_ledger, canonical,
            encoded=0 if canonical == 0 else 17,
            reservation=reservation, budget=budget)


@pytest.mark.parametrize(
    "axis", ["format", "source-kind", "member", "dataset", "fact-summary"],
    ids=("missing-rehashed-catalog-format", "missing-rehashed-catalog-source-kind",
         "missing-rehashed-catalog-member", "missing-rehashed-catalog-dataset",
         "missing-rehashed-catalog-fact-summary"),
)
def test_rehashed_catalog_rejects_each_isolated_cross_field(tmp_path, axis):
    if axis == "fact-summary":
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "processed.nxs"
        with h5py.File(path, "w") as handle:
            _processed_thumbnail(
                handle, 0, np.arange(6, dtype=np.uint8).reshape(2, 3),
                vmin=0.0, vmax=5.0)
        catalog = api.catalog_viewer_2d(path)
        change = {"source_kind": api.Viewer2DSourceKind.PROCESSED_RAW}
    else:
        path = tmp_path / "matrix.npy"
        np.save(path, np.arange(6).reshape(2, 3))
        catalog = api.catalog_viewer_2d(path)
        change = {
            "format": {"format_name": "csv"},
            "source-kind": {"source_kind": api.Viewer2DSourceKind.CSV_MATRIX},
            "member": {"member_name": "image.npy"},
            "dataset": {"dataset_path": "/entry/data/data"},
        }[axis]
    _raises(TypeError, _rehashed_catalog, catalog, **change)


@pytest.mark.parametrize("axis", ["baseline", "unselected-drift", "selected-drift"],
                         ids=lambda axis: f"r45-v1-{axis}")
def test_processed_eiger_selection_is_interval_local(tmp_path, monkeypatch, axis):
    h5py = pytest.importorskip("h5py")
    raw = np.arange(2 * 3, dtype=np.uint16).reshape(1, 2, 3)
    master, segments, arrays = _eiger_master(tmp_path / axis, [
        (f"data_{number:06d}.h5", raw + number * 100) for number in (1, 2)])
    processed = tmp_path / f"{axis}.nxs"
    with h5py.File(processed, "w") as handle:
        _processed_source(handle, 7, str(master.resolve()), 1)
    catalog = api.catalog_viewer_2d(processed)
    base, earlier, selected = catalog.dependencies
    touched, opened, read = [], [], []
    original, original_hpath, original_read = api._cert_revision, api._hpath, api._read_hdf_dataset
    def certify(revision, walk):
        touched.append(revision.canonical_path)
        return original(revision, walk)
    def hpath(handle, path, walk):
        result = original_hpath(handle, path, walk)
        opened.append((str(Path(handle.filename).resolve()), path))
        return result
    def read_hdf(locator, dataset, frame, *args):
        read.append((locator, dataset, frame))
        return original_read(locator, dataset, frame, *args)
    monkeypatch.setattr(api, "_cert_revision", certify)
    monkeypatch.setattr(api, "_hpath", hpath)
    monkeypatch.setattr(api, "_read_hdf_dataset", read_hdf)
    if axis != "baseline":
        drifted = segments[0 if axis == "unselected-drift" else 1]
        with h5py.File(drifted, "r+") as handle:
            handle["entry/data/data"][0, 0, 0] += 1
    if axis == "selected-drift":
        _assert_changed(api.read_viewer_2d_frame, catalog, 7)
    else:
        frame = api.read_viewer_2d_frame(catalog, 7)
        _assert_canonical(frame, arrays[1][0])
        assert frame.provenance.dependencies == (base, selected)
    assert base.revision.canonical_path in touched
    assert earlier.revision.canonical_path not in touched
    assert selected.revision.canonical_path in touched
    assert all(earlier.logical_path != path for _, path in opened) and all(
        locator != earlier.locator for locator, _, _ in read)


@pytest.mark.parametrize("kind,size", [("utf8", 4096), ("bytes", 4096),
    ("utf8", 4097), ("bytes", 4097)], ids=("r45-v2-utf8-4096",
    "r45-v2-bytes-4096", "r45-v2-utf8-4097", "r45-v2-bytes-4097"))
def test_selected_direct_vlen_strings_use_one_fixed_bounded_read(tmp_path, monkeypatch, kind, size):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / f"{kind}-{size}.nxs"
    payload = ("é" * (size // 2) + "x" * (size % 2)
               if kind == "utf8" else b"x" * size)
    dtype = h5py.string_dtype("utf-8" if kind == "utf8" else "ascii")
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry")
        group.attrs.create("selected", payload, dtype=dtype)
    reads, high = _watch_named_hdf_attribute(monkeypatch, h5py, "selected")
    with h5py.File(path) as handle:
        walk = api._HdfWalk()
        if size == 4097:
            _assert_limit(api._hattr, handle["entry"], "selected", walk)
        else:
            expected = payload if kind == "utf8" else payload.decode("ascii")
            assert api._hattr(handle["entry"], "selected", walk) == expected
            assert walk.attribute_bytes == 4096
    cset = h5py.h5t.CSET_UTF8 if kind == "utf8" else h5py.h5t.CSET_ASCII
    assert reads == [("|S4097", False, 4097, 4097, (4097,
        h5py.h5t.STR_NULLPAD, cset), 4096 if size == 4096 else -1)]
    assert high == []


@pytest.mark.parametrize("axis", ["direct-vlen", "direct-ref", "compound-vlen",
    "subarray-vlen", "compound-ref", "subarray-ref"],
    ids=lambda axis: f"r45-v2-{axis}")
def test_selected_nested_or_nonstring_object_attributes_refuse_without_read(tmp_path, monkeypatch, axis):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / f"{axis}.nxs"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry")
        reads, high = _watch_named_hdf_attribute(monkeypatch, h5py, "selected")
        if axis == "direct-vlen":
            dtype = h5py.vlen_dtype(np.dtype("int32"))
            value = np.empty((), dtype=object)
            value[()] = np.array([1, 2], dtype=np.int32)
        elif axis == "direct-ref":
            dtype, value = h5py.ref_dtype, group.ref
        elif axis == "compound-vlen":
            dtype = np.dtype([("value", h5py.vlen_dtype(np.dtype("int32")))])
            value = np.array((np.array([1, 2], dtype=np.int32),), dtype=dtype)
        elif axis == "compound-ref":
            dtype = np.dtype([("value", h5py.ref_dtype)])
            value = np.array((group.ref,), dtype=dtype)
        else:
            base = (h5py.vlen_dtype(np.dtype("int32"))
                    if axis == "subarray-vlen" else h5py.ref_dtype)
            dtype = np.dtype((base, (2,)))
            value = np.empty((), dtype=object)
            typeid = h5py.h5t.array_create(h5py.h5t.vlen_create(h5py.h5t.NATIVE_INT32)
                if axis == "subarray-vlen" else h5py.h5t.STD_REF_OBJ, (2,))
        if axis.startswith("subarray"):
            # Faithful low-level dtype probe: h5py cannot persist this scalar shape.
            class Attribute:
                shape, dtype = (), typeid.dtype
                def get_space(self): return object()
                def get_type(self): return typeid.copy()
                def get_storage_size(self): return 16
                def read(self, *args, **kwargs): pytest.fail("forbidden value read")
                def close(self): typeid.close()
            monkeypatch.setattr(h5py.h5a, "exists", lambda *args: True)
            monkeypatch.setattr(h5py.h5a, "open", lambda *args: Attribute())
            owner = group
        else:
            group.attrs.create("selected", value, dtype=dtype)
            owner = group
        _assert_refusal(api.Viewer2DRefusalCode.FORMAT_INVALID,
                        api._hattr, owner, "selected", api._HdfWalk())
        assert reads == high == []
@pytest.mark.parametrize("drift", ["shape", "dtype"],
                         ids=lambda drift: f"r45-v3-fabio-{drift}")
def test_later_fabio_frame_metadata_refuses_before_global_data_decode(tmp_path, monkeypatch, drift):
    fabio = pytest.importorskip("fabio")
    path = tmp_path / f"later-{drift}.edf"
    first = np.arange(6, dtype=np.int16).reshape(2, 3)
    later = (np.arange(8, dtype=np.int16).reshape(2, 4) if drift == "shape"
             else np.arange(6, dtype=np.int32).reshape(2, 3))
    image = fabio.edfimage.EdfImage(data=first)
    image.append_frame(data=later)
    initial = fabio.edfimage.EdfImage(data=first)
    initial.append_frame(data=first.copy())
    initial.write(path)
    catalog = api.catalog_viewer_2d(path)
    image.write(path)
    catalog = _rehashed_catalog(catalog, primary_revision=api._stable_revision(path))
    decoded = []
    original = fabio.edfimage.EdfFrame._unpack
    def decode(frame):
        decoded.append(frame._index)
        return original(frame)
    monkeypatch.setattr(fabio.edfimage.EdfFrame, "_unpack", decode)
    _assert_changed(api.read_viewer_2d_frame, catalog, 1)
    assert decoded == []


def test_thumbnail_fact_dtype_is_closed_to_uint8_or_uint16(tmp_path):
    _raises(TypeError, api.Viewer2DFrameFact, 0,
            api.Viewer2DSourceKind.PROCESSED_THUMBNAIL, (2, 3),
            np.dtype("int16").str, thumbnail_path="/entry/frames/frame_0000/thumbnail")


@pytest.mark.parametrize("field", ["raw_locator", "dataset_path", "raw_source_frame",
    "source_dtype"], ids=lambda field: f"r45-v4-direct-{field.replace('_', '-')}")
def test_direct_hdf_provenance_rebinds_each_selected_field(tmp_path, field):
    path = tmp_path / "direct.nxs"
    _write_hdf_stack(path, np.arange(12, dtype=np.int64).reshape(2, 2, 3))
    catalog = api.catalog_viewer_2d(path)
    frame = api.read_viewer_2d_frame(catalog, 1)
    p = frame.provenance
    changes = {
        "raw_locator": str((tmp_path / "foreign.nxs").resolve()),
        "dataset_path": "/entry/foreign/data",
        "raw_source_frame": 0,
        "source_dtype": np.dtype("uint64").str,
    }
    forged_p = _forge(p, **{field: changes[field]})
    forged = _forge(frame, provenance=forged_p)
    forged.__post_init__()
    _raises(TypeError, api._validate_frame_against_catalog, catalog, 1, forged)


def test_primary_same_size_aba_and_dependency_change_refuse_without_publication(tmp_path):
    path = tmp_path / "stable.csv"
    path.write_bytes(b"1,2\n3,4\n")
    catalog = api.catalog_viewer_2d(path)
    stat = path.stat()
    path.write_bytes(b"9,8\n7,6\n")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    _assert_changed(api.read_viewer_2d_frame, catalog, 0)


def test_memory_ledger_uses_raw_only_r7_arithmetic_and_fails_closed(monkeypatch):
    ledger = api.viewer_2d_memory_ledger(2, 3, encoded_retained=17, ram_bytes=10**12)
    assert ledger.canonical_bytes == 48
    assert ledger.reader_peak == max(3 * 48, 17 + 4 * 48)
    assert ledger.renderer_slack_raw == 5 * 48 // 2
    assert ledger.renderer_linear == 7 * 48 // 2
    assert ledger.renderer_log_update == 9 * 48 // 2
    assert ledger.renderer_paint == 5 * 48
    assert ledger.renderer_peak == 5 * 48
    assert ledger.concurrent_frame_bytes == 6 * 48
    assert ledger.total == ledger.catalog_reservation + max(
        ledger.reader_peak, ledger.concurrent_frame_bytes
    )
    assert ledger.admission == ledger.canonical_bytes + ledger.total
    assert ledger.budget == 1024**3

    monkeypatch.setattr(api, "_physical_ram_bytes", lambda: None)
    assert api.viewer_2d_memory_ledger(1, 1).budget == 884_736_000


def test_suffix_set_is_closed_and_viewer_only(tmp_path):
    assert api.SUPPORTED_VIEWER_SUFFIXES == frozenset(
        {".edf", ".tif", ".tiff", ".cbf", ".img", ".mar3450", ".raw",
         ".h5", ".hdf5", ".nxs", ".nexus", ".csv", ".npy", ".npz"}
    )
    _raises(api.Viewer2DReadError, api.catalog_viewer_2d, tmp_path / "wrong.cxi")

    from xrd_tools.io import image
    assert ".csv" not in image.SUPPORTED_EXTS
    assert ".npy" not in image.SUPPORTED_EXTS
    assert ".npz" not in image.SUPPORTED_EXTS
    assert "xrd_tools.io.viewer_2d" in sys.modules
