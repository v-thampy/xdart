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


def test_prepared_single_rebind_matches_cold_tiff_and_hdf(
    monkeypatch, tmp_path: Path,
) -> None:
    from xrd_tools.reduction import background as module

    source, first_target, second_target = _files(
        tmp_path, ("background.tif", "first.tif", "second.tif")
    )
    real_decoder = module.read_image
    calls = _decoder(
        monkeypatch,
        {source.name: np.arange(4, dtype=np.uint16).reshape(2, 2)},
    )
    plan = FrameBackgroundPlan(mode="Single BG File", locator=str(source))
    first_fact = _fact(first_target, label=1)
    second_fact = _fact(second_target, label=2)
    first = resolve_frame_background(plan, first_fact)
    expected = resolve_frame_background(plan, second_fact)
    prepared = module._prepare_single_frame_background(
        plan, first_fact, first,
    )
    assert prepared is not None and len(calls) == 2
    rebound = module._rebind_single_frame_background(
        prepared, plan, second_fact,
    )
    assert rebound is not None and rebound.disposition == "RESOLVED"
    assert rebound.descriptor_bytes == expected.descriptor_bytes
    assert rebound.fingerprint == expected.fingerprint
    assert rebound.background is first.background

    master = tmp_path / "background.h5"
    with h5py.File(master, "w") as handle:
        handle.create_dataset(
            "entry/data/data_000001",
            data=np.arange(4, dtype=np.uint16).reshape(1, 2, 2),
        )
    monkeypatch.setattr(module, "read_image", real_decoder)
    direct_plan = FrameBackgroundPlan(
        mode="Single BG File",
        locator=str(master),
        dataset_path="/entry/data/data_000001",
        frame_index=0,
    )
    direct_first = resolve_frame_background(direct_plan, first_fact)
    direct_expected = resolve_frame_background(direct_plan, second_fact)
    direct_prepared = module._prepare_single_frame_background(
        direct_plan, first_fact, direct_first,
    )
    assert direct_prepared is not None
    direct_rebound = module._rebind_single_frame_background(
        direct_prepared, direct_plan, second_fact,
    )
    assert direct_rebound is not None
    assert direct_rebound.descriptor_bytes == direct_expected.descriptor_bytes
    assert direct_rebound.fingerprint == direct_expected.fingerprint
    assert direct_rebound.background is direct_first.background


def test_prepared_single_reuses_one_decode_and_detects_content_drift(
    monkeypatch, tmp_path: Path,
) -> None:
    from xrd_tools.reduction import background as module

    source, target = _files(tmp_path, ("background.tif", "target.tif"))
    calls = _decoder(
        monkeypatch,
        {source.name: np.arange(4, dtype=np.uint16).reshape(2, 2)},
    )
    plan = FrameBackgroundPlan(mode="Single BG File", locator=str(source))
    first_fact = _fact(target, label=1)
    first = resolve_frame_background(plan, first_fact)
    prepared = module._prepare_single_frame_background(
        plan, first_fact, first,
    )
    assert prepared is not None

    extra_fact = _fact(
        target,
        label=51,
        metadata=(("unexpected", ("number", float(1.0).hex())),),
    )
    assert resolve_frame_background(plan, extra_fact).disposition == "REFUSED"
    assert module._rebind_single_frame_background(
        prepared, plan, extra_fact,
    ) is None

    descriptors = set()
    for label in range(2, 51):
        fact = _fact(target, label=label)
        rebound = module._rebind_single_frame_background(
            prepared, plan, fact,
        )
        assert rebound is not None and rebound.disposition == "RESOLVED"
        assert rebound.background is first.background
        assert _payload(rebound)["frame_fact"][0] == label
        descriptors.add(rebound.descriptor_bytes)
    assert len(calls) == 1 and len(descriptors) == 49

    cancelled = Event()
    cancelled.set()
    stopped = module._rebind_single_frame_background(
        prepared, plan, _fact(target, label=51), cancelled=cancelled,
    )
    assert stopped is not None and stopped.disposition == "CANCELLED"

    real_digest = module._digest
    original_state = real_digest(source.resolve(), None)[0]
    prior = source.stat()
    source.write_bytes(b"changed-0")
    os.utime(source, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    monkeypatch.setattr(
        module,
        "_digest",
        lambda path, signal: (
            original_state,
            real_digest(path, signal)[1],
        ),
    )
    assert module._rebind_single_frame_background(
        prepared, plan, _fact(target, label=52),
    ) is None
    assert len(calls) == 1


def test_prepared_single_rejects_same_content_path_replacement(
    monkeypatch, tmp_path: Path,
) -> None:
    from xrd_tools.reduction import background as module

    source, target = _files(tmp_path, ("background.tif", "target.tif"))
    _decoder(
        monkeypatch,
        {source.name: np.arange(4, dtype=np.uint16).reshape(2, 2)},
    )
    plan = FrameBackgroundPlan(mode="Single BG File", locator=str(source))
    first_fact = _fact(target, label=1)
    first = resolve_frame_background(plan, first_fact)
    prepared = module._prepare_single_frame_background(
        plan, first_fact, first,
    )
    assert prepared is not None

    real_digest = module._digest
    replaced = False

    def replace_after_digest(path, cancelled):
        nonlocal replaced
        receipt = real_digest(path, cancelled)
        if not replaced:
            replacement = path.with_name("replacement.tif")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            replaced = True
        return receipt

    monkeypatch.setattr(module, "_digest", replace_after_digest)
    assert module._rebind_single_frame_background(
        prepared, plan, _fact(target, label=2),
    ) is None


def test_prepared_single_normalization_and_sidecar_are_exact(
    monkeypatch, tmp_path: Path,
) -> None:
    from xrd_tools.reduction import background as module

    source, target = _files(tmp_path, ("background.tif", "target.tif"))
    sidecar = source.with_suffix(".txt")
    sidecar.write_text("monitor=2")
    calls = _decoder(monkeypatch, {source.name: np.ones((2, 2))})
    monkeypatch.setattr(
        module,
        "read_image_metadata_observed",
        lambda *_args, **_kwargs: ImageMetadataRead(
            {"monitor": 2.0}, sidecar,
        ),
    )
    plan = FrameBackgroundPlan(
        mode="Single BG File",
        locator=str(source),
        normalization_key="monitor",
        metadata_format="txt",
    )
    same = (("monitor", ("number", float(4.0).hex())),)
    different = (("monitor", ("number", float(8.0).hex())),)
    first_fact = _fact(target, label=1, metadata=same)
    first = resolve_frame_background(plan, first_fact)
    prepared = module._prepare_single_frame_background(
        plan, first_fact, first,
    )
    assert prepared is not None
    rebound = module._rebind_single_frame_background(
        prepared, plan, _fact(target, label=2, metadata=same),
    )
    assert rebound is not None and rebound.background is first.background
    extra = same + (("unexpected", ("number", float(1.0).hex())),)
    extra_fact = _fact(target, label=3, metadata=extra)
    assert resolve_frame_background(plan, extra_fact).disposition == "REFUSED"
    assert module._rebind_single_frame_background(
        prepared, plan, extra_fact,
    ) is None
    case_fact = _fact(
        target,
        label=4,
        metadata=(("Monitor", ("number", float(4.0).hex())),),
    )
    assert resolve_frame_background(plan, case_fact).disposition == "REFUSED"
    assert module._rebind_single_frame_background(
        prepared, plan, case_fact,
    ) is None
    assert module._rebind_single_frame_background(
        prepared, plan, _fact(target, label=5, metadata=different),
    ) is None
    sidecar.write_text("monitor=2 changed")
    assert module._rebind_single_frame_background(
        prepared, plan, _fact(target, label=6, metadata=same),
    ) is None
    assert len(calls) == 1
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
    real_processed = processed_scan_id.has_processed_output_markers_file
    event = Event()
    monkeypatch.setattr(processed_scan_id, "has_processed_output_markers_file",
                        lambda handle: (event.set() or False))
    cancelled = resolve_frame_background(FrameBackgroundPlan(mode="Single BG File",
        locator=str(master), dataset_path="/entry/data/data_000001", frame_index=0),
        _fact(target), cancelled=event)
    assert cancelled.disposition == "CANCELLED"
    monkeypatch.setattr(processed_scan_id, "has_processed_output_markers_file", lambda handle: True)
    processed = resolve_frame_background(FrameBackgroundPlan(
        mode="Single BG File", locator=str(master),
        dataset_path="/entry/data/data_000001", frame_index=0), _fact(target))
    assert processed.disposition == "REFUSED"
    monkeypatch.setattr(processed_scan_id, "has_processed_output_markers_file", real_processed)


def test_dark_corrected_flag_keeps_older_mappings_and_reports_no_scale(
    monkeypatch, tmp_path: Path,
) -> None:
    source, target = _files(tmp_path, ("background.tif", "target.tif"))
    _decoder(monkeypatch, {source.name: np.full((2, 2), 30, dtype=np.uint16)})
    plain = FrameBackgroundPlan(mode="Single BG File", locator=str(source), scale=2.0)
    marked = FrameBackgroundPlan(mode="Single BG File", locator=str(source), scale=2.0,
                                 dark_corrected=True)
    assert "dark_corrected" not in plain.to_mapping()  # existing records keep their mapping
    assert marked.to_mapping() == {**plain.to_mapping(), "dark_corrected": True}
    assert FrameBackgroundPlan.from_mapping(plain.to_mapping()) == plain
    assert FrameBackgroundPlan.from_mapping(marked.to_mapping()) == marked
    for bad in ({**plain.to_mapping(), "dark_corrected": False},
                {**plain.to_mapping(), "dark_corrected": 1}):
        with pytest.raises(ValueError):
            FrameBackgroundPlan.from_mapping(bad)
    assert resolve_frame_background(plain, _fact(target)).scale == 2.0
    result = resolve_frame_background(marked, _fact(target))
    # the reducer then subtracts the Background as it is: no factor to remove a dark with
    assert result.disposition == "RESOLVED" and result.scale is None
    np.testing.assert_array_equal(result.background, np.full((2, 2), 60.0))


def test_directory_last_saved_before_frame(monkeypatch, tmp_path: Path) -> None:
    """Darks collected during a scan: each frame takes the Background file saved last,
    not after the frame's own file."""
    root = tmp_path / "darks"; root.mkdir()
    frames = tmp_path / "frames"; frames.mkdir()
    early, late, tied = _files(root, ("dark_a.tif", "dark_b.tif", "dark_c.tif"))
    first, second, before_any = _files(frames, ("scan_0001.tif", "scan_0002.tif", "scan_0000.tif"))
    second_s = 1_000_000_000
    for path, seconds in ((early, 10), (first, 20), (late, 30), (second, 40), (before_any, 5),
                          (tied, 100)):
        os.utime(path, ns=(seconds * second_s, seconds * second_s))
    _decoder(monkeypatch, {early.name: np.full((2, 2), 1.0), late.name: np.full((2, 2), 2.0),
                           tied.name: np.full((2, 2), 3.0)})
    plan = FrameBackgroundPlan(mode="BG Directory", locator=str(root),
                               match_rule="Last Saved Before Frame")
    for frame, value in ((first, 1.0), (second, 2.0)):
        result = resolve_frame_background(plan, _fact(frame))
        assert result.disposition == "RESOLVED", result.diagnostics
        np.testing.assert_array_equal(result.background, np.full((2, 2), value))
    body = _payload(resolve_frame_background(plan, _fact(second)))
    assert body["selected"] == str(late.resolve()) and body["frame_saved_ns"] == 40 * second_s
    refused = resolve_frame_background(plan, _fact(before_any))
    assert refused.disposition == "REFUSED" and "no Background file was saved before" in refused.diagnostics[0]
    os.utime(tied, ns=(30 * second_s, 30 * second_s))  # saved together with dark_b: the later name
    np.testing.assert_array_equal(resolve_frame_background(plan, _fact(second)).background,
                                  np.full((2, 2), 3.0))
    moved = resolve_frame_background(plan, _fact(second))
    later = root / "dark_z.tif"; later.write_bytes(b"later")  # saved after the frame
    os.utime(later, ns=(90 * second_s, 90 * second_s))
    assert resolve_frame_background(plan, _fact(second)).fingerprint == moved.fingerprint
    os.utime(tied, ns=(31 * second_s, 31 * second_s))  # a changed file time changes the record
    assert resolve_frame_background(plan, _fact(second)).fingerprint != moved.fingerprint
    container = resolve_frame_background(plan, _fact(first, dataset="/entry/data/data"))
    assert container.disposition == "REFUSED" and "one image file per frame" in container.diagnostics[0]
    shared = FrameBackgroundPlan(mode="BG Directory", locator=str(frames),
                                 match_rule="Last Saved Before Frame")
    for name in ("dark_d.tif",):
        (frames / name).write_bytes(b"dark"); os.utime(frames / name, ns=(15 * second_s, 15 * second_s))
    _decoder(monkeypatch, {"dark_d.tif": np.full((2, 2), 4.0)})
    unfiltered = resolve_frame_background(shared, _fact(second))
    assert unfiltered.disposition == "REFUSED" and "Filename Filter" in unfiltered.diagnostics[0]
    filtered = resolve_frame_background(
        FrameBackgroundPlan(mode="BG Directory", locator=str(frames), filename_filter="dark",
                            match_rule="Last Saved Before Frame"), _fact(second))
    np.testing.assert_array_equal(filtered.background, np.full((2, 2), 4.0))


def test_the_dark_comes_off_the_background_before_scale_and_normalization(
    monkeypatch, tmp_path: Path,
) -> None:
    """With a dark (and "BG dark-corrected" unticked) the frame loses
    factor · (Background − dark), factor = Scale × monitor ratio."""
    from xrd_tools.reduction import Frame, ReductionPlan
    from xrd_tools.reduction import background as module
    from xrd_tools.reduction.core import _dark_corrected_background

    source, target = _files(tmp_path, ("background.tif", "target.tif"))
    sidecar = source.with_suffix(".txt"); sidecar.write_text("monitor=2")
    _decoder(monkeypatch, {source.name: np.full((2, 2), 30.0)})
    monkeypatch.setattr(module, "read_image_metadata_observed",
                        lambda *_args, **_kwargs: ImageMetadataRead({"monitor": 2.0}, sidecar))
    plan = FrameBackgroundPlan(mode="Single BG File", locator=str(source), scale=1.5,
                               normalization_key="monitor", metadata_format="txt")
    result = resolve_frame_background(
        plan, _fact(target, metadata=(("monitor", ("number", float(4.0).hex())),)))
    assert result.disposition == "RESOLVED" and result.scale == 3.0  # 1.5 × 4 / 2
    np.testing.assert_array_equal(result.background, np.full((2, 2), 90.0))
    frame = Frame(0, image=np.zeros((2, 2)), metadata={}, background=result.background,
                  background_scale=result.scale)
    corrected = _dark_corrected_background(frame, ReductionPlan(dark=np.full((2, 2), 10.0)))
    np.testing.assert_array_equal(corrected, np.full((2, 2), 60.0))  # 3 × (30 − 10)


def test_a_series_capsule_rebinds_by_stat_byte_for_byte_and_resolves_again_on_any_change(
    monkeypatch, tmp_path: Path,
) -> None:
    """BG-SERIES-REREAD (maintainer, 2026-09-30, on the per-frame proof: "Stat check (my recommendation): yes"):
    a run resolves its Series Average once; a later frame rebinds it after a stat proof (the member set listed
    again; every member's and metadata file's state) and scales the mean by its own normalization. Byte for
    byte the full resolution's result, for the same and for other normalizations; any member or metadata
    change, or a member added, resolves in full."""
    from xrd_tools.reduction import background as module
    one, two, three, target = _files(tmp_path, ("bg_1.tif", "bg_2.tif", "bg_3.tif", "target_1.tif"))
    calls = _decoder(monkeypatch, {one.name: np.array([[1., np.nan], [3., 5.]]),  # uneven counts
                                   two.name: np.array([[3., 7.], [np.nan, 7.]]),
                                   three.name: np.array([[5., 9.], [9., np.nan]])})
    sidecars = {path.resolve(): path.with_suffix(".txt") for path in (one, two, three)}
    for value in sidecars.values():
        value.write_text("stable")
    monitors = {one.resolve(): 2.0, two.resolve(): 4.0, three.resolve(): 6.0}
    monkeypatch.setattr(module, "read_image_metadata_observed", lambda path, **kwargs: ImageMetadataRead(
        {"monitor": monitors[Path(path).resolve()]}, sidecars[Path(path).resolve()]))
    plan = FrameBackgroundPlan(mode="Series Average", locator=str(two), normalization_key="monitor",
                               metadata_format="txt")

    def fact(label, monitor):
        return _fact(target, label=label, metadata=(("monitor", ("number", float(monitor).hex())),))
    with module.series_capsule_requested({}) as held:
        first = resolve_frame_background(plan, fact(0, 6.0))
    capsule = held["series"]
    assert first.disposition == "RESOLVED" and len(calls) == 3  # each member read once
    previous = None
    for label, monitor in ((1, 6.0), (2, 3.0), (3, 3.0), (4, 6.0)):
        rebound = module._rebind_series_frame_background(capsule, plan, fact(label, monitor))
        reads = len(calls)
        full = resolve_frame_background(plan, fact(label, monitor))
        assert len(calls) == reads + 3 and rebound.disposition == "RESOLVED"
        assert (rebound.descriptor_bytes, rebound.fingerprint) == (full.descriptor_bytes, full.fingerprint)
        np.testing.assert_array_equal(rebound.background, full.background)
        np.testing.assert_array_equal(np.asarray(rebound.variance_factor), np.asarray(full.variance_factor))
        assert rebound.scale == full.scale
        if label == 3:  # the last frame's normalization: its result again
            assert rebound.background is previous.background
        previous = rebound
    assert len(calls) == 3 + 3 * 4  # the rebinds read nothing; each full resolution reads every member
    assert module._rebind_series_frame_background(capsule, plan, fact(5, 6.0)).disposition == "RESOLVED"
    os.utime(sidecars[three.resolve()], ns=(1, 1))  # a metadata file touched
    assert module._rebind_series_frame_background(capsule, plan, fact(6, 6.0)) is None
    with module.series_capsule_requested({}) as held:
        resolve_frame_background(plan, fact(7, 6.0))
    capsule = held["series"]
    three.write_bytes(b"source-changed")  # a member rewritten
    assert module._rebind_series_frame_background(capsule, plan, fact(8, 6.0)) is None
    with module.series_capsule_requested({}) as held:
        resolve_frame_background(plan, fact(9, 6.0))
    capsule = held["series"]
    (tmp_path / "bg_4.tif").write_bytes(b"source-new")  # a member added
    assert module._rebind_series_frame_background(capsule, plan, fact(10, 6.0)) is None
    assert resolve_frame_background(plan, fact(11, 6.0)).disposition != "RESOLVED"  # bg_4 has no decoder value


def _monitor_sidecar(path: Path, monitor: float) -> None:
    path.write_text(f"# Counters\nmonitor = {monitor}\n# Motors\ndel = 0.0\n"
                    "User: review, time: Mon Jan 15 10:30:00 2024  # Temp\n")


def test_a_series_whose_metadata_is_reached_through_a_link_keeps_no_capsule(tmp_path: Path) -> None:
    """Codex BG-SERIES F1, its reproduction with the real TIFF and metadata readers: a member's sidecar is a
    symbolic link. The receipt names the resolved file only, so a link retargeted to another valid sidecar
    left every recorded state unchanged and the rebind reused a background 3.5 times too large. No capsule is
    kept for such a Series (each frame resolves in full, as before BG-SERIES-REREAD); without the link one is."""
    import tifffile
    from xrd_tools.reduction import background as module
    paths = []
    for index in range(3):
        path = tmp_path / f"bg_{index}.tif"
        tifffile.imwrite(path, np.arange(15, dtype=np.float32).reshape(3, 5) + 2 * index + 1)
        _monitor_sidecar(path.with_suffix(".txt"), (2, 5, 11)[index])
        paths.append(path)
    plan = FrameBackgroundPlan(mode="Series Average", locator=str(paths[0]), normalization_key="monitor",
                               metadata_format="txt")
    target = _fact(tmp_path / "target_0.tif", shape=(3, 5), metadata=(("monitor", ("number", (3.0).hex())),))
    with module.series_capsule_requested({}) as held:
        assert resolve_frame_background(plan, target).disposition == "RESOLVED"
    assert "series" in held  # the control: plain sidecars keep a capsule
    old_meta, new_meta = tmp_path / "old_monitor.txt", tmp_path / "new_monitor.txt"
    _monitor_sidecar(old_meta, 5)
    _monitor_sidecar(new_meta, 50)
    alias = paths[1].with_suffix(".txt")
    alias.unlink()
    alias.symlink_to(old_meta)
    with module.series_capsule_requested({}) as held:
        initial = resolve_frame_background(plan, target)
    assert initial.disposition == "RESOLVED" and initial.background[0, 0] == 1.5  # mean 3 × 3 / 6
    assert "series" not in held
    alias.unlink()
    alias.symlink_to(new_meta)
    retargeted = resolve_frame_background(plan, target)
    assert retargeted.background[0, 0] == pytest.approx(3 * 3 / 21, abs=0)  # monitors 2, 50, 11
    assert retargeted.fingerprint != initial.fingerprint


@pytest.mark.parametrize("suffix", [".edf", ".img", ".tif"])
def test_a_series_whose_pixels_live_in_another_file_keeps_no_capsule(tmp_path: Path, suffix: str) -> None:
    """Codex BG-SERIES F2 and its re-review F1, with the real readers: fabio reads an EDF's pixels from the file
    its header names in ``EDF_BinaryFileName``, which no stat of the member proves, so rewriting only that file
    left the rebind on the old pixels. Fabio detects EDF by content, so the same holds for an EDF named
    ``.img`` or ``.tif``, which the first fix (an ``.edf`` suffix test) missed. No capsule is kept for such a
    member; an EDF holding its own pixels, under the same name, keeps one."""
    import fabio
    from xrd_tools.reduction import background as module
    shape = (3, 4)
    own = tmp_path / f"own_1{suffix}"
    fabio.edfimage.EdfImage(data=np.full(shape, 5, dtype="<u2")).write(str(own))
    fact = _fact(tmp_path / "target_1.tif", shape=shape)
    with module.series_capsule_requested({}) as held:
        assert resolve_frame_background(FrameBackgroundPlan(mode="Series Average", locator=str(own)),
                                        fact).disposition == "RESOLVED"
    assert "series" in held  # the control
    member, payload = tmp_path / f"bg_1{suffix}", tmp_path / "pixels.bin"
    payload.write_bytes(np.full(shape, 3, dtype="<u2").tobytes())
    fabio.edfimage.EdfImage(data=np.zeros(shape, dtype="<u2"), header={
        "EDF_BinaryFileName": payload.name, "EDF_BinaryFileSize": "24", "EDF_BinaryFilePosition": "0",
    }).write(str(member))
    plan = FrameBackgroundPlan(mode="Series Average", locator=str(member))
    with module.series_capsule_requested({}) as held:
        initial = resolve_frame_background(plan, fact)
    np.testing.assert_array_equal(initial.background, np.full(shape, 3.0))
    assert "series" not in held
    payload.write_bytes(np.full(shape, 9, dtype="<u2").tobytes())
    np.testing.assert_array_equal(resolve_frame_background(plan, fact).background, np.full(shape, 9.0))


def test_a_raw_binary_series_keeps_its_capsule(tmp_path: Path) -> None:
    """The control for the content check: a ``.raw`` member fabio cannot open is decoded as raw binary, whose
    bytes are its pixels, so its Series keeps the capsule (the 17-2 Pilatus case)."""
    from xrd_tools.reduction import background as module
    shape = (3, 4)
    for index in (1, 2):
        (tmp_path / f"bg_{index}.raw").write_bytes(np.full(shape, index, dtype="<i4").tobytes())
    plan = FrameBackgroundPlan(mode="Series Average", locator=str(tmp_path / "bg_1.raw"))
    fact = _fact(tmp_path / "target_1.tif", shape=shape)
    with module.series_capsule_requested({}) as held:
        result = resolve_frame_background(plan, fact)
    assert result.disposition == "RESOLVED" and "series" in held
    np.testing.assert_array_equal(result.background, np.full(shape, 1.5))
