"""Headless P3-3B frame-background science and bounded provenance."""
from __future__ import annotations
import ast, hashlib, json, os, struct
from pathlib import Path
from threading import Event
import h5py
import numpy as np
import pytest
from xrd_tools.io.metadata import ImageMetadataRead
from xrd_tools.reduction import (
    FrameBackgroundPlan,
    FrameBackgroundResult,
    resolve_frame_background,
)
def _fact(path: Path, *, label=0, shape=(2, 2), metadata=(), dataset=None):
    return (label, str(path.resolve()), dataset, 0, shape, tuple(metadata))
def _files(tmp_path: Path, names: tuple[str, ...]) -> tuple[Path, ...]:
    paths = tuple(tmp_path / name for name in names)
    for index, path in enumerate(paths):
        path.write_bytes(f"source-{index}".encode())
    return paths
def _decoder(monkeypatch, values: dict[str, np.ndarray]):
    from xrd_tools.reduction import background as module
    calls = []
    def read(path, **kwargs):
        calls.append((Path(path).resolve(), kwargs))
        return values[Path(path).name].copy()
    monkeypatch.setattr(module, "read_image", read)
    return calls
def _payload(result: FrameBackgroundResult) -> dict:
    assert result.descriptor_bytes is not None
    return json.loads(result.descriptor_bytes)
def _root(value):
    while isinstance(value, np.ndarray): value = value.base
    return value
def test_none_is_zero_io_and_identity_neutral(monkeypatch, tmp_path: Path) -> None:
    from xrd_tools.reduction import background as module
    fact = _fact(tmp_path / "absent.tif"); touched = []; event = Event(); event.set()
    active = FrameBackgroundPlan(mode="Single BG File", locator=str(tmp_path / "absent.tif"))
    monkeypatch.setattr(module, "read_image", lambda *a, **k: touched.append("read"))
    monkeypatch.setattr(module, "read_image_metadata_observed",
                        lambda *a, **k: touched.append("metadata"))
    monkeypatch.setattr(module, "Path", lambda *a, **k: touched.append("path"))
    monkeypatch.setattr(module.hashlib, "sha256", lambda *a, **k: touched.append("hash"))
    result = resolve_frame_background(FrameBackgroundPlan(), fact, cancelled=event)
    assert result == FrameBackgroundResult("RESOLVED", None, None, None, ())
    assert touched == []
    cancelled = resolve_frame_background(active, _fact(tmp_path / "target.tif"),
                                         cancelled=event)
    assert cancelled.disposition == "CANCELLED"
    assert cancelled.background is cancelled.descriptor_bytes is cancelled.fingerprint is None
    assert touched == []
    for disposition in ("RETRYABLE", "REFUSED", "CANCELLED"):
        outcome = FrameBackgroundResult(disposition, None, None, None)
        assert outcome.background is outcome.descriptor_bytes is outcome.fingerprint is None
        array = np.frombuffer(b"\0" * 32, dtype=np.float64).reshape(2, 2)
        for payload in ((array, None, None), (None, b"{}", None), (None, None, "0" * 64)):
            with pytest.raises(ValueError): FrameBackgroundResult(disposition, *payload)
    for diagnostics in (("a", "b"), ("x" * 513,)):
        with pytest.raises(ValueError): FrameBackgroundResult("REFUSED", None, None, None, diagnostics)
def test_headless_single_stable_float64_scale_and_descriptor(
    monkeypatch, tmp_path: Path,
) -> None:
    from xrd_tools.reduction import background as module
    real_decoder = module.read_image
    source, target = _files(tmp_path, ("background.tif", "target.tif"))
    calls = _decoder(monkeypatch, {source.name: np.arange(4, dtype=np.uint16).reshape(2, 2)})
    result = resolve_frame_background(
        FrameBackgroundPlan(mode="Single BG File", locator=str(source), scale=-2.0),
        _fact(target),
    )
    assert result.disposition == "RESOLVED" and len(calls) == 1
    np.testing.assert_array_equal(result.background, [[0.0, -2.0], [-4.0, -6.0]])
    assert result.background.dtype == np.float64 and result.background.flags.c_contiguous
    assert not result.background.flags.writeable and isinstance(_root(result.background), bytes)
    body = _payload(result)
    assert body["version"] == 1 and body["mode"] == "Single BG File"
    assert body["source"]["sha256"] and body["decoded"]["sha256"]
    assert body["result_sha256"] and result.fingerprint == __import__("hashlib").sha256(result.descriptor_bytes).hexdigest()
    assert body["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert body["decoded"]["sha256"] == hashlib.sha256(np.arange(4, dtype=np.uint16).tobytes()).hexdigest()
    assert body["result_sha256"] == hashlib.sha256(np.asarray(result.background).tobytes()).hexdigest()
    master = tmp_path / "raw.h5"
    with h5py.File(master, "w") as handle:
        handle.create_dataset("entry/data/data_000001", data=np.arange(4, dtype=np.uint16).reshape(1, 2, 2))
    monkeypatch.setattr(module, "read_image", real_decoder)
    direct = resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(master),
        dataset_path="/entry/data/data_000001", frame_index=0,
    ), _fact(target))
    assert direct.disposition == "RESOLVED"
    np.testing.assert_array_equal(direct.background, np.arange(4).reshape(2, 2))
    direct_body = _payload(direct)
    assert direct_body["source"]["hdf"]["dataset_path"] == "/entry/data/data_000001" and direct_body["result_sha256"] == hashlib.sha256(np.asarray(direct.background).tobytes()).hexdigest()
    monkeypatch.setattr(module, "read_image", lambda path, **kwargs: (os.utime(path, ns=(Path(path).stat().st_atime_ns, Path(path).stat().st_mtime_ns + 1)) or np.ones((2, 2))))
    assert resolve_frame_background(FrameBackgroundPlan(mode="Single BG File", locator=str(source)), _fact(target)).disposition == "RETRYABLE"
    source.write_bytes(b"stable"); real_digest = module._digest; stable_state = real_digest(source.resolve(), None)[0]
    def hidden_rewrite(path, **kwargs): prior = Path(path).stat(); Path(path).write_bytes(b"staple"); os.utime(path, ns=(prior.st_atime_ns, prior.st_mtime_ns)); return np.ones((2, 2))
    monkeypatch.setattr(module, "read_image", hidden_rewrite); monkeypatch.setattr(module, "_digest", lambda path, cancelled: (stable_state, real_digest(path, cancelled)[1]))
    assert resolve_frame_background(FrameBackgroundPlan(mode="Single BG File", locator=str(source)), _fact(target)).disposition == "RETRYABLE"; monkeypatch.setattr(module, "_digest", real_digest)
    def torn_error(path, **kwargs): Path(path).write_bytes(b"changed"); raise ValueError("decoder torn")
    monkeypatch.setattr(module, "read_image", torn_error)
    assert resolve_frame_background(FrameBackgroundPlan(mode="Single BG File", locator=str(source)), _fact(target)).disposition == "RETRYABLE"
    monkeypatch.setattr(module, "read_image", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("stable invalid")))
    assert resolve_frame_background(FrameBackgroundPlan(mode="Single BG File", locator=str(source)), _fact(target)).disposition == "REFUSED"
    for sibling in ("data_000002", "data_１２３４５６"):
        with h5py.File(master, "a") as handle: handle.create_dataset(f"entry/data/{sibling}", data=np.ones((1, 2, 2)))
        assert resolve_frame_background(FrameBackgroundPlan(mode="Single BG File", locator=str(master), dataset_path="/entry/data/data_000001", frame_index=0), _fact(target)).disposition == "REFUSED"
        with h5py.File(master, "a") as handle: del handle[f"entry/data/{sibling}"]
    real_proof = module._hdf_proof; monkeypatch.setattr(module, "_hdf_proof", lambda path, *args: (os.utime(path, ns=(Path(path).stat().st_atime_ns, Path(path).stat().st_mtime_ns + 1)), (_ for _ in ()).throw(ValueError("torn proof")))[1])
    assert resolve_frame_background(FrameBackgroundPlan(mode="Single BG File", locator=str(master), dataset_path="/entry/data/data_000001", frame_index=0), _fact(target)).disposition == "RETRYABLE"; monkeypatch.setattr(module, "_hdf_proof", real_proof)
    tree = ast.parse((Path(__file__).parents[2] / "src/xrd_tools/reduction/background.py").read_text())
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names} | {node.module or "" for node in ast.walk(tree)
                                           if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith(("PyQt", "PySide", "qtpy", "xdart")) for name in imports)
    assert sum(isinstance(node, ast.FunctionDef) and node.name == "resolve_frame_background"
               for node in tree.body) == 1
def test_series_any_anchor_complete_order_streaming_finite_mean(
    monkeypatch, tmp_path: Path,
) -> None:
    from xrd_tools.reduction import background as module
    first, anchor, last, wrong_delimiter, target = _files(
        tmp_path, ("scan_1.tif", "scan_004.tif", "scan_9.tif", "scan-8.tif", "target_7.tif"))
    calls = _decoder(monkeypatch, {
        first.name: np.array([[1.0, np.nan], [3.0, 5.0]]),
        anchor.name: np.array([[3.0, 7.0], [np.nan, 7.0]]),
        last.name: np.array([[5.0, 9.0], [9.0, np.nan]]),
        wrong_delimiter.name: np.full((2, 2), 99.0),
    })
    result = resolve_frame_background(FrameBackgroundPlan(
        mode="Series Average", locator=str(anchor),
    ), _fact(target))
    assert result.disposition == "RESOLVED"
    assert [path.name for path, _ in calls] == [first.name, anchor.name, last.name]
    np.testing.assert_allclose(result.background, [[3.0, 8.0], [6.0, 6.0]])
    body = _payload(result)
    assert body["manifest"]["count"] == 3
    assert body["manifest"]["receipt_bytes"] > 0
    assert len(body["manifest"]["sha256"]) == 64 and "members" not in body
    assert len(result.descriptor_bytes) <= 16_384
    digest = hashlib.sha256(b"XDART-FRAME-BG-SERIES-MANIFEST-V1\0"); total = 0
    for ordinal, path in enumerate((first, anchor, last)):
        native = np.ascontiguousarray({first: np.array([[1., np.nan], [3., 5.]]),
            anchor: np.array([[3., 7.], [np.nan, 7.]]), last: np.array([[5., 9.], [9., np.nan]])}[path])
        state, source_sha = module._digest(path.resolve(), None)
        receipt = {"ordinal": ordinal, "source": {"locator": str(path.resolve()),
            "state": list(state), "sha256": source_sha, "hdf": None,
            "decoded": {"shape": [2, 2], "dtype": native.dtype.str,
                        "sha256": hashlib.sha256(native.tobytes()).hexdigest()}},
            "metadata_items": (), "metadata_source": None}
        raw = module._json(receipt); digest.update(struct.pack(">Q", len(raw))); digest.update(raw); total += len(raw)
    assert body["manifest"] == {"version": 1, "count": 3,
        "receipt_bytes": total, "sha256": digest.hexdigest()}
    source_text = Path(module.__file__).read_text(); series_text = source_text.split('if plan.mode == "Series Average":', 1)[1].split("root = Path", 1)[0]
    assert ".append(" not in series_text and "manifest.update(" in series_text
    _decoder(monkeypatch, {first.name: np.ones((2, 2), np.uint16),
        anchor.name: np.ones((2, 2), np.uint32), last.name: np.ones((2, 2), np.uint16)})
    assert resolve_frame_background(FrameBackgroundPlan(
        mode="Series Average", locator=str(anchor)), _fact(target)).disposition == "REFUSED"
    unnumbered = tmp_path / "background.tif"; unnumbered.write_bytes(b"raw")
    assert resolve_frame_background(FrameBackgroundPlan(mode="Series Average", locator=str(unnumbered)), _fact(target)).disposition == "REFUSED"
    outside = tmp_path / "outside"; outside.mkdir(); escaped = outside / "scan_8.tif"; escaped.write_bytes(b"raw")
    _decoder(monkeypatch, {first.name: np.ones((2, 2)), anchor.name: np.ones((2, 2)), last.name: np.ones((2, 2)), escaped.name: np.ones((2, 2))})
    (tmp_path / "scan_8.tif").symlink_to(escaped); assert resolve_frame_background(FrameBackgroundPlan(mode="Series Average", locator=str(anchor)), _fact(target)).disposition == "REFUSED"
    (tmp_path / "scan_8.tif").unlink()
    event = Event(); real_read = module._read; read_count = []
    def cancel_after_one(*args, **kwargs):
        value = real_read(*args, **kwargs); read_count.append(1); event.set(); return value
    monkeypatch.setattr(module, "_read", cancel_after_one)
    cancelled = resolve_frame_background(FrameBackgroundPlan(
        mode="Series Average", locator=str(anchor)), _fact(target), cancelled=event)
    assert cancelled.disposition == "CANCELLED" and len(read_count) == 1
def test_directory_scan_frame_unique_boundary_and_self_exclusion(
    monkeypatch, tmp_path: Path,
) -> None:
    from xrd_tools.reduction import background as module
    root = tmp_path / "bg"; root.mkdir()
    candidate, wrong, target = _files(root, ("ref_scan-2.tif", "ref_scan-12.tif", "scan-2.tif"))
    calls = _decoder(monkeypatch, {candidate.name: np.ones((2, 2)), wrong.name: np.full((2, 2), 9.0)})
    plan = FrameBackgroundPlan(mode="BG Directory", locator=str(root),
                               match_rule="Scan Root + Frame Number")
    result = resolve_frame_background(plan, _fact(target))
    assert result.disposition == "RESOLVED" and [p for p, _ in calls] == [candidate.resolve()] * 3
    np.testing.assert_array_equal(result.background, np.ones((2, 2)))
    duplicate = root / "SCAN_ref-2.tif"; duplicate.write_bytes(b"duplicate")
    calls.clear(); calls_map = {**{candidate.name: np.ones((2, 2)), wrong.name: np.ones((2, 2))},
                               duplicate.name: np.ones((2, 2))}
    _decoder(monkeypatch, calls_map)
    refused = resolve_frame_background(plan, _fact(target))
    assert refused.disposition == "REFUSED" and refused.background is None
    duplicate.unlink(); escaped_root = tmp_path / "escaped"; escaped_root.mkdir(); outside = tmp_path / "foreign"; outside.mkdir()
    escaped = outside / "ref_scan-2.tif"; escaped.write_bytes(b"raw"); (escaped_root / escaped.name).symlink_to(escaped)
    _decoder(monkeypatch, {escaped.name: np.ones((2, 2))}); assert resolve_frame_background(FrameBackgroundPlan(mode="BG Directory", locator=str(escaped_root), match_rule="Scan Root + Frame Number"), _fact(target)).disposition == "REFUSED"
    real_paths = module._directory_paths; passes = []
    def mutate_second(*args, **kwargs):
        if passes: (root / "late_scan-2.tif").write_bytes(b"late")
        passes.append(1); return real_paths(*args, **kwargs)
    monkeypatch.setattr(module, "_directory_paths", mutate_second); _decoder(monkeypatch, {candidate.name: np.ones((2, 2)), wrong.name: np.ones((2, 2)), "late_scan-2.tif": np.ones((2, 2))})
    assert resolve_frame_background(plan, _fact(target)).disposition == "RETRYABLE" and len(passes) == 2
def test_directory_metadata_tagged_scalar_unique_match(
    monkeypatch, tmp_path: Path,
) -> None:
    from xrd_tools.reduction import background as module
    root = tmp_path / "bg"; root.mkdir()
    selected, other = _files(root, ("a_1.tif", "b_1.tif"))
    target = tmp_path / "target_1.tif"; target.write_bytes(b"target")
    sidecars = {path: path.with_suffix(".txt") for path in (selected, other)}
    for sidecar in sidecars.values(): sidecar.write_text("stable")
    calls: dict[Path, int] = {}
    def metadata(path, **kwargs):
        path = Path(path).resolve(); calls[path] = calls.get(path, 0) + 1
        accepted = 12.5 if path == selected.resolve() else 99.0
        values = {"energy": -1.0, "unused": "discard"} if calls[path] == 1 else {
            "energy": accepted, "unused": "must-not-persist"}
        return ImageMetadataRead(values, sidecars[path])
    monkeypatch.setattr(module, "read_image_metadata_observed", metadata)
    _decoder(monkeypatch, {selected.name: np.full((2, 2), 2.0), other.name: np.ones((2, 2))})
    result = resolve_frame_background(FrameBackgroundPlan(
        mode="BG Directory", locator=str(root), match_rule="Metadata Key",
        metadata_key="energy", metadata_format="txt",
    ), _fact(target, metadata=(("energy", ("number", float(12.5).hex())),)))
    assert result.disposition == "RESOLVED" and all(value == 6 for value in calls.values())
    body = _payload(result)
    encoded = json.dumps(body, sort_keys=True)
    assert "unused" not in encoded and "discard" not in encoded
    monkeypatch.setattr(module, "read_image_metadata_observed", lambda path, **kwargs: ImageMetadataRead({"energy": 1.0 if Path(path).resolve() == selected.resolve() else 99.0}, sidecars[Path(path).resolve()]))
    collision = resolve_frame_background(FrameBackgroundPlan(mode="BG Directory", locator=str(root), match_rule="Metadata Key", metadata_key="energy"), _fact(target, metadata=(("energy", ("bool", True)),)))
    assert collision.disposition == "REFUSED"
    monkeypatch.setattr(module, "read_image_metadata_observed", lambda path, **kwargs: ImageMetadataRead({"energy": 12.5, "ENERGY": 12.5}, sidecars[Path(path).resolve()]))
    assert resolve_frame_background(FrameBackgroundPlan(mode="BG Directory", locator=str(root), match_rule="Metadata Key", metadata_key="energy"), _fact(target, metadata=(("energy", ("number", float(12.5).hex())),))).disposition == "REFUSED"
    calls.clear()
    foreign = root / "foreign.txt"; foreign.write_text("stable")
    def switched(path, **kwargs):
        path = Path(path).resolve(); calls[path] = calls.get(path, 0) + 1
        source = sidecars[path] if calls[path] == 1 else foreign
        return ImageMetadataRead({"energy": 12.5}, source)
    monkeypatch.setattr(module, "read_image_metadata_observed", switched)
    changed_source = resolve_frame_background(FrameBackgroundPlan(
        mode="BG Directory", locator=str(root), match_rule="Metadata Key",
        metadata_key="energy", metadata_format="txt",
    ), _fact(target, metadata=(("energy", ("number", float(12.5).hex())),)))
    assert changed_source.disposition == "RETRYABLE"
    calls.clear()
    def torn(path, **kwargs):
        path = Path(path).resolve(); calls[path] = calls.get(path, 0) + 1
        if calls[path] == 2: sidecars[path].write_text("torn!!")
        return ImageMetadataRead({"energy": "x" * 4086}, sidecars[path])
    monkeypatch.setattr(module, "read_image_metadata_observed", torn)
    changed_bytes = resolve_frame_background(FrameBackgroundPlan(
        mode="BG Directory", locator=str(root), match_rule="Metadata Key",
        metadata_key="energy", metadata_format="txt",
    ), _fact(target, metadata=(("energy", ("number", float(12.5).hex())),)))
    assert changed_bytes.disposition == "RETRYABLE"
    for sidecar in sidecars.values(): sidecar.write_text("stable")
    def stable(path, **kwargs):
        path = Path(path).resolve()
        return ImageMetadataRead({"energy": 12.5 if path == selected.resolve() else 99.0}, sidecars[path])
    monkeypatch.setattr(module, "read_image_metadata_observed", stable)
    real_read = module._read; reads = []
    def mutate_after_selection(*args, **kwargs):
        value = real_read(*args, **kwargs); reads.append(1)
        if len(reads) == 1: sidecars[selected.resolve()].write_text("phase-boundary")
        return value
    monkeypatch.setattr(module, "_read", mutate_after_selection)
    assert resolve_frame_background(FrameBackgroundPlan(mode="BG Directory", locator=str(root),
        match_rule="Metadata Key", metadata_key="energy", metadata_format="txt"),
        _fact(target, metadata=(("energy", ("number", float(12.5).hex())),))).disposition == "RETRYABLE"
    event = Event()
    monkeypatch.setattr(module, "read_image_metadata_observed", lambda path, **kwargs:
        (event.set() or ImageMetadataRead({"energy": 12.5}, sidecars[Path(path).resolve()])))
    assert resolve_frame_background(FrameBackgroundPlan(mode="BG Directory", locator=str(root),
        match_rule="Metadata Key", metadata_key="energy"), _fact(target,
        metadata=(("energy", ("number", float(12.5).hex())),)), cancelled=event).disposition == "CANCELLED"
def test_filter_blank_match_all_and_malformed_refusal(monkeypatch, tmp_path: Path) -> None:
    from xrd_tools.core.filters import compile_filter
    root = tmp_path / "bg"; root.mkdir()
    candidate, target = _files(root, ("ref_scan_3.tif", "scan_3.tif"))
    _decoder(monkeypatch, {candidate.name: np.ones((2, 2))})
    blank = resolve_frame_background(FrameBackgroundPlan(
        mode="BG Directory", locator=str(root), match_rule="Scan Root + Frame Number",
        filename_filter="   ",
    ), _fact(target))
    assert blank.disposition == "RESOLVED"
    predicate = compile_filter("REF -bad | alternate")
    assert predicate("ref_scan_3.tif") and not predicate("ref_bad_scan_3.tif")
    assert compile_filter("not")("cannot_open") and not compile_filter("NOT bad")("bad.tif")
    for malformed in ("|", "NOT", "-", "a || b"):
        with pytest.raises(ValueError, match="filter"): FrameBackgroundPlan(mode="BG Directory", locator=str(root), match_rule="Scan Root + Frame Number", filename_filter=malformed)
def test_positive_finite_normalization_and_series_denominator(
    monkeypatch, tmp_path: Path,
) -> None:
    from xrd_tools.reduction import background as module
    one, two, target = _files(tmp_path, ("scan_1.tif", "scan_2.tif", "target_1.tif"))
    _decoder(monkeypatch, {one.name: np.full((2, 2), 2.0), two.name: np.full((2, 2), 4.0)})
    sidecars = {one.resolve(): one.with_suffix(".txt"), two.resolve(): two.with_suffix(".txt")}
    for value in sidecars.values(): value.write_text("stable")
    counts = {}
    def metadata(path, **kwargs):
        path = Path(path).resolve(); counts[path] = counts.get(path, 0) + 1
        accepted = 2.0 if path == one.resolve() else 4.0
        return ImageMetadataRead({"MONITOR": 999.0 if counts[path] % 2 else accepted}, sidecars[path])
    monkeypatch.setattr(module, "read_image_metadata_observed", metadata)
    plan = FrameBackgroundPlan(mode="Series Average", locator=str(two),
                               normalization_key="monitor", metadata_format="txt")
    result = resolve_frame_background(plan, _fact(
        target, metadata=(("monitor", ("number", float(6.0).hex())),)))
    assert result.disposition == "RESOLVED"
    np.testing.assert_allclose(result.background, np.full((2, 2), 6.0))
    bad = resolve_frame_background(plan, _fact(
        target, metadata=(("monitor", ("number", float(0.0).hex())),)))
    assert bad.disposition == "REFUSED" and bad.background is None
    for invalid in (None, 0.0, -1.0, float("inf"), float("nan")):
        def invalid_metadata(path, **kwargs):
            values = {} if invalid is None else {"monitor": invalid}
            return ImageMetadataRead(values, sidecars[Path(path).resolve()])
        monkeypatch.setattr(module, "read_image_metadata_observed", invalid_metadata)
        outcome = resolve_frame_background(plan, _fact(
            target, metadata=(("monitor", ("number", float(6.0).hex())),)))
        assert outcome.disposition == "REFUSED" and outcome.background is None
def test_shape_nonfinite_and_stable_read_refusals(monkeypatch, tmp_path: Path) -> None:
    source, target = _files(tmp_path, ("background.tif", "target.tif"))
    _decoder(monkeypatch, {source.name: np.ones((3, 2))})
    wrong = resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(source)), _fact(target))
    assert wrong.disposition == "REFUSED"
    _decoder(monkeypatch, {source.name: np.full((2, 2), np.nan)})
    nonfinite = resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(source)), _fact(target))
    assert nonfinite.disposition == "REFUSED"
    from xrd_tools.reduction import background as module
    assert (module._MAX_FILE, module._MAX_PIXELS, module._MAX_MEMBERS,
            module._MAX_PATH_BYTES, module._MAX_RECEIPT, module._MAX_RECEIPTS,
            module._MAX_FRAME_FACT, module._MAX_DESCRIPTOR) == (
            256 * 1024 ** 2, 67_108_864, 10_000, 4 * 1024 ** 2,
            16_384, 64 * 1024 ** 2, 24_576, 262_144)
    assert module._tag("x" * 4085)[0] == "text"
    with pytest.raises(ValueError): module._tag("x" * 4086)
    assert module._tag(b"x" * 2042)[0] == "bytes"
    for scalar in (b"x" * 2043, 10 ** 1000):
        with pytest.raises(ValueError): module._tag(scalar)
    valid_fact = (0, "/" + "t" * 4095, "/" + "s" * 4095, 0, (2, 2), ())
    assert module._frame(valid_fact) == valid_fact
    for fact in ((0, "relative.tif", None, 0, (2, 2), ()),
                 (0, str(target), "entry/data", 0, (2, 2), ()),
                 (0, str(target), "/entry//data", 0, (2, 2), ()),
                 (0, "/" + "t" * 4096, None, 0, (2, 2), ()),
                 (0, str(target), "/" + "s" * 4096, 0, (2, 2), ()),
                 (0, str(target), None, 0, (2, 2), (("a", ("number", "0x1p+0")),
                  ("b", ("number", "0x1p+0")), ("c", ("number", "0x1p+0"))))):
        with pytest.raises(ValueError): module._frame(fact)
    irrelevant = _fact(target, metadata=(("unused", ("number", float(1).hex())),))
    assert resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(source)), irrelevant).disposition == "REFUSED"
    def torn(path, **kwargs):
        Path(path).write_bytes(b"changed")
        return np.ones((2, 2))
    monkeypatch.setattr(module, "read_image", torn)
    changed = resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(source)), _fact(target))
    assert changed.disposition == "RETRYABLE"
    assert all(value is None for value in (changed.background,
        changed.descriptor_bytes, changed.fingerprint))
    source.write_bytes(b"source-0")
    real_digest = module._digest
    stable_state = real_digest(source.resolve(), None)[0]
    monkeypatch.setattr(module, "_digest",
                        lambda path, cancelled: (stable_state, real_digest(path, cancelled)[1]))
    def digest_only(path, **kwargs):
        old = Path(path).stat(); Path(path).write_bytes(b"changed!")
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
        return np.ones((2, 2))
    monkeypatch.setattr(module, "read_image", digest_only)
    digest_drift = resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(source)), _fact(target))
    assert digest_drift.disposition == "RETRYABLE"
    monkeypatch.setattr(module, "_digest", real_digest)
    master = tmp_path / "multi.h5"
    with h5py.File(master, "w") as handle:
        handle.create_dataset("entry/data/data_000001", data=np.ones((1, 2, 2)))
        handle.create_dataset("entry/data/data_000002", data=np.ones((1, 2, 2)))
    called = []
    real = module.read_image
    monkeypatch.setattr(module, "read_image", lambda *a, **k: called.append(a) or real(*a, **k))
    refused = resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(master),
        dataset_path="/entry/data/data_000001", frame_index=0,
    ), _fact(target))
    assert refused.disposition == "REFUSED" and called == []
    unicode_master = tmp_path / "unicode.h5"
    with h5py.File(unicode_master, "w") as handle:
        handle.create_dataset("entry/data/data_000001", data=np.ones((1, 2, 2)))
        handle.create_dataset("entry/data/data_１２３４５６", data=np.ones((1, 2, 2)))
    called.clear()
    unicode_refused = resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(unicode_master),
        dataset_path="/entry/data/data_000001", frame_index=0,
    ), _fact(target))
    assert unicode_refused.disposition == "REFUSED" and called == []
    with pytest.raises(ValueError):
        FrameBackgroundPlan(mode="Single BG File", locator="/" + "x" * 4096)
    sparse = tmp_path / "large.tif"
    with sparse.open("wb") as stream: stream.truncate(256 * 1024 ** 2 + 1)
    assert resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(sparse)), _fact(target)).disposition == "REFUSED"
    with pytest.raises(ValueError):
        FrameBackgroundPlan(mode="Single BG File", locator=str(master),
                            dataset_path="/" + "x" * 4096, frame_index=0)
    assert FrameBackgroundPlan(mode="Single BG File", locator=str(master),
        dataset_path="/" + "x" * 4095, frame_index=0).dataset_path
    for selector in ("/", "/entry//data", "/entry/data/"):
        with pytest.raises(ValueError): FrameBackgroundPlan(mode="Single BG File", locator=str(master), dataset_path=selector, frame_index=0)
    with pytest.raises(ValueError):
        FrameBackgroundPlan(mode="Single BG File", locator=str(master),
                            dataset_path="/" + "/".join("x" for _ in range(257)), frame_index=0)
    assert resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(source)),
        _fact(target, shape=(8192, 8193))).disposition == "REFUSED"
    series_root = tmp_path / "bounded_series"; series_root.mkdir()
    anchor, member = _files(series_root, ("scan_1.tif", "scan_2.tif"))
    monkeypatch.setattr(module, "_MAX_MEMBERS", 1)
    assert resolve_frame_background(FrameBackgroundPlan(mode="Series Average",
        locator=str(anchor)), _fact(target)).disposition == "REFUSED"
    monkeypatch.setattr(module, "_MAX_MEMBERS", 10_000)
    monkeypatch.setattr(module, "_MAX_PATH_BYTES", len(str(anchor.resolve()).encode()))
    assert resolve_frame_background(FrameBackgroundPlan(mode="Series Average",
        locator=str(anchor)), _fact(target)).disposition == "REFUSED"
    event = Event(); real_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda self: (event.set() or real_is_file(self)))
    assert resolve_frame_background(FrameBackgroundPlan(mode="Series Average",
        locator=str(anchor)), _fact(target), cancelled=event).disposition == "CANCELLED"
    monkeypatch.setattr(Path, "is_file", real_is_file)
    with pytest.raises(ValueError): module._manifest(b"domain", ({"x": "y" * 16_385},))
    real_json = module._json
    monkeypatch.setattr(module, "_json", lambda value: b"x" * 16_384)
    assert module._manifest(b"domain", ({} for _ in range(4096)))[:2] == (4096, 64 * 1024 ** 2)
    with pytest.raises(ValueError): module._manifest(b"domain", ({} for _ in range(4097)))
    monkeypatch.setattr(module, "_json", real_json)
    assert module._MAX_FRAME_FACT == 24_576
    with pytest.raises(ValueError): module._result(FrameBackgroundPlan(mode="Single BG File", locator=str(source)), _fact(target), np.ones((2, 2)), {"x": "y" * 262_145}, None)
    dependency = tmp_path / "dependency.h5"
    with h5py.File(dependency, "w") as handle:
        handle.create_dataset("data", data=np.ones((1, 2, 2)))
    invalid = []
    soft = tmp_path / "soft.h5"
    with h5py.File(soft, "w") as handle:
        handle.create_group("entry/data"); handle["entry/data/selected"] = h5py.SoftLink("/entry/data/raw")
        handle.create_dataset("entry/data/raw", data=np.ones((1, 2, 2)))
    invalid.append((soft, "/entry/data/selected"))
    external = tmp_path / "external.h5"
    with h5py.File(external, "w") as handle:
        handle.create_group("entry/data"); handle["entry/data/selected"] = h5py.ExternalLink(str(dependency), "/data")
    invalid.append((external, "/entry/data/selected"))
    vds = tmp_path / "vds.h5"
    layout = h5py.VirtualLayout(shape=(1, 2, 2), dtype="f8")
    layout[:] = h5py.VirtualSource(str(dependency), "/data", shape=(1, 2, 2))
    with h5py.File(vds, "w", libver="latest") as handle:
        handle.create_virtual_dataset("entry/data/selected", layout)
    invalid.append((vds, "/entry/data/selected"))
    stored = tmp_path / "external_storage.h5"
    with h5py.File(stored, "w") as handle:
        handle.create_dataset("entry/data/selected", shape=(1, 2, 2), dtype="f8",
                              external=[("payload.raw", 0, h5py.h5f.UNLIMITED)])
    invalid.append((stored, "/entry/data/selected"))
    unknown = tmp_path / "unknown.h5"
    with h5py.File(unknown, "w") as handle:
        handle.create_dataset("entry/data/selected", data=np.ones((1, 2, 2), dtype=np.complex128))
    invalid.append((unknown, "/entry/data/selected"))
    unknown_kind = tmp_path / "unknown_kind.h5"
    with h5py.File(unknown_kind, "w") as handle:
        handle.create_dataset("entry/data/selected", data=np.ones((1, 2, 2), dtype=np.float32))
        handle.create_dataset("entry/reduction/data", data=np.ones((1, 2, 2), dtype=np.float32))
        handle.create_group("entry/data/group")
    invalid.append((unknown_kind, "/entry/data/selected"))
    invalid.extend(((unknown_kind, "/entry/reduction/data"), (unknown_kind, "/entry/data/group")))
    malformed = tmp_path / "malformed.h5"; malformed.write_bytes(b"not an hdf5 file")
    invalid.append((malformed, "/entry/data/data_000001"))
    for path, selector in invalid:
        outcome = resolve_frame_background(FrameBackgroundPlan(
            mode="Single BG File", locator=str(path), dataset_path=selector,
            frame_index=0), _fact(target))
        assert outcome.disposition == "REFUSED"
        assert all(value is None for value in (outcome.background,
            outcome.descriptor_bytes, outcome.fingerprint))
    from xrd_tools.io import processed_scan_id
    real_processed = processed_scan_id.is_processed_xdart_file
    event = Event()
    monkeypatch.setattr(processed_scan_id, "is_processed_xdart_file",
                        lambda handle: (event.set() or False))
    cancelled = resolve_frame_background(FrameBackgroundPlan(mode="Single BG File",
        locator=str(master), dataset_path="/entry/data/data_000001", frame_index=0),
        _fact(target), cancelled=event)
    assert cancelled.disposition == "CANCELLED"
    monkeypatch.setattr(processed_scan_id, "is_processed_xdart_file", lambda handle: True)
    processed = resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(master),
        dataset_path="/entry/data/data_000001", frame_index=0), _fact(target))
    assert processed.disposition == "REFUSED"
    monkeypatch.setattr(processed_scan_id, "is_processed_xdart_file", real_processed)
