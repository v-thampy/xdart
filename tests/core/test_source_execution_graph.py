"""Focused closed source-execution graph and Dataset-ID ownership oracle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
import hashlib, subprocess, sys
import json
from pathlib import Path
import time
import weakref

import h5py
import numpy as np
import pytest
import tifffile

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io import bluesky_nexus
from xrd_tools.io import nexus as nexus_io
from xrd_tools.sources import cursor as cursor_module
from xrd_tools.sources import execution_graph as graph
from xrd_tools.sources import metadata_provider
from xrd_tools.sources.selection import image_series_spec, single_image_spec


def _tiffs(tmp_path: Path, names=("scan_0001.tif", "scan_0002.tif")) -> SourceSpec:
    for index, name in enumerate(names):
        tifffile.imwrite(tmp_path / name, np.full((2, 3), index + 1, dtype=np.uint16))
    return image_series_spec(tmp_path / names[0], metadata_format=None)


def _container(path: Path, *, entry="entry", extent=3, detector="image", bluesky=True):
    with h5py.File(path, "w") as handle:
        if bluesky: handle.attrs["creator"] = "NXWriter"
        root = handle.create_group(entry)
        root.attrs["NX_class"] = "NXentry"
        root.create_dataset("end_time", data=np.bytes_("done"))
        data = root.create_group("data"); data.attrs["NX_class"] = "NXdata"
        pixels = data.create_dataset(detector, data=np.arange(extent * 6, dtype=np.uint16).reshape(extent, 2, 3))
        pixels.attrs["signal_type"] = "detector"
        pos = root.create_group("instrument/positioners/theta")
        pos.create_dataset("value", data=np.arange(extent, dtype=float))
        data.create_dataset("theta", data=np.arange(extent, dtype=float))
        data.create_dataset("I0", data=np.arange(1, extent + 1, dtype=float))
        meta = root.require_group("instrument/bluesky/metadata")
        meta.create_dataset("motors", data=np.bytes_("!!python/tuple\n- theta\n"))
    return path


def _eiger_master(root: Path, extents=(2, 3)) -> tuple[Path, tuple[Path, ...]]:
    members = []
    for ordinal, extent in enumerate(extents, 1):
        path = root / f"data_{ordinal:06d}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "entry/data/data", data=np.full((extent, 2, 3), ordinal, dtype="u2"),
                chunks=(1, 2, 3),
            )
        members.append(path)
    master = root / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data"); data.attrs["NX_class"] = "NXdata"
        for ordinal, path in enumerate(members, 1):
            data[f"data_{ordinal:06d}"] = h5py.ExternalLink(path.name, "/entry/data/data")
    return master, tuple(members)


def _bind_stack(handle: h5py.File, entry: str = "entry"):
    slot = nexus_io._NexusDatasetOwnerSlot()
    binding = nexus_io._bind_nexus_stack_from_entry(
        handle[entry], declared_entry_name=entry,
        prefer_apstools_flat=False, owner_slot=slot,
    )
    assert slot.owner is binding
    return slot, binding


def test_cold_headless_average_bypasses_mutable_candidate_registry(tmp_path) -> None:
    source = tmp_path / "cold_0001.tif"
    tifffile.imwrite(source, np.ones((2, 3), dtype=np.uint16))
    candidate_src = Path(__file__).resolve().parents[2] / "src"
    code = """
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from xrd_tools.reduction import AverageScanRecipe, ReductionPlan, prepare_average_scan
from xrd_tools.sources import adapters
from xrd_tools.sources.selection import image_series_spec
assert adapters._ADAPTERS == {} and "xrd_tools.sources.registry" not in sys.modules
average = sys.modules["xrd_tools.reduction.average"]
bindings = []
qualify = average.qualify_source_execution_graph
def observed(*args, **kwargs):
    value = qualify(*args, **kwargs); bindings.append(value.reader_binding); return value
average.qualify_source_execution_graph = observed
plan = prepare_average_scan(AverageScanRecipe(image_series_spec(Path(sys.argv[2]), metadata_format=None), Path(sys.argv[3]), ReductionPlan()))
def has_array(value):
    if type(value).__module__ == "numpy" and type(value).__name__ == "ndarray": return True
    if is_dataclass(value): return any(has_array(getattr(value, item.name)) for item in fields(value))
    if isinstance(value, dict): return any(has_array(key) or has_array(item) for key, item in value.items())
    return isinstance(value, (tuple, list)) and any(has_array(item) for item in value)
assert plan.contributor_extent == 1 and bindings == ["average_closed_v1"] and not has_array(plan)
assert adapters._ADAPTERS == {} and "xrd_tools.sources.registry" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(candidate_src), str(source), str(tmp_path / "out.nxs")],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"


def test_compact_tiff_recipe_is_constant_size_and_reenumerates_exact_members(tmp_path) -> None:
    from xrd_tools.reduction import AverageScanRecipe, ReductionPlan
    small = tmp_path / "small"; large = tmp_path / "large"
    small.mkdir(); large.mkdir()
    small_source = _tiffs(small)
    large_source = _tiffs(large, tuple(f"scan_{index:04d}.tif" for index in range(1, 65)))
    recipes = tuple(AverageScanRecipe(source, root / "out.nxs", ReductionPlan())
                    for source, root in ((small_source, small), (large_source, large)))
    expected_fields = (
        "api_version", "source", "target", "entry", "source_base", "output_mode",
        "live_mode", "save_xye", "batch_mode", "integration_1d", "integration_2d",
        "integrator_gi", "threshold_min", "threshold_max", "mask_saturation",
        "reduction_extra", "calibration", "background", "numeric_metadata_keys",
        "invariant_metadata_keys", "envelope_bytes", "resource_requests", "resource_env",
    )
    def retained_cardinality(value):
        if hasattr(type(value), "__dataclass_fields__"):
            return 1 + sum(retained_cardinality(getattr(value, item.name)) for item in fields(value))
        if isinstance(value, Mapping):
            return 1 + sum(retained_cardinality(key) + retained_cardinality(item)
                           for key, item in value.items())
        if isinstance(value, (tuple, list)):
            return 1 + sum(retained_cardinality(item) for item in value)
        return 1
    assert tuple(item.name for item in fields(recipes[0])) == expected_fields
    assert retained_cardinality(recipes[0]) == retained_cardinality(recipes[1])
    for recipe in recipes:
        thawed = recipe.source.thaw()
        assert "files" not in thawed.options
        assert "admitted_motor_values" not in thawed.options
        assert thawed.options["average_compact_series_v1"] is True
        assert not any(isinstance(getattr(recipe, field.name), np.ndarray)
                       for field in fields(recipe))
    qualified = graph.qualify_source_execution_graph(
        recipes[1].source.thaw(), reader_binding="average_closed_v1",
    )
    assert qualified.stamp.frame_count == 64
    assert tuple(Path(value.path).name for value in qualified.stamp.members) == tuple(
        f"scan_{index:04d}.tif" for index in range(1, 65)
    )
    assert "average_compact_series_v1" not in qualified.execution_source.options
    compact_payload = {
        "uri": str(recipes[1].source.thaw().uri),
        "options": dict(recipes[1].source.thaw().options),
    }
    assert len(json.dumps(graph.source_graph_payload(qualified), default=str)) > len(
        json.dumps(compact_payload, default=str)
    )
    Path(recipes[0].source.thaw().options["selected_file"]).unlink()
    with pytest.raises((OSError, ValueError), match="empty|missing|unavailable"):
        graph.qualify_source_execution_graph(
            recipes[0].source.thaw(), reader_binding="average_closed_v1",
        )


def test_compact_tiff_preserves_unnumbered_and_empty_enumeration_singletons(tmp_path, monkeypatch) -> None:
    from xrd_tools.reduction import AverageScanRecipe, ReductionPlan
    anchor = tmp_path / "still.tif"; tifffile.imwrite(anchor, np.ones((2, 2), dtype=np.uint8))
    ordinary_recipe = AverageScanRecipe(
        image_series_spec(anchor, metadata_format=None), tmp_path / "ordinary.nxs", ReductionPlan(),
    )
    explicit_recipe = AverageScanRecipe(
        single_image_spec(anchor, metadata_format=None), tmp_path / "single.nxs", ReductionPlan(),
    )
    ordinary = graph.qualify_source_execution_graph(
        ordinary_recipe.source.thaw(), reader_binding="average_closed_v1",
    )
    explicit = graph.qualify_source_execution_graph(
        explicit_recipe.source.thaw(), reader_binding="average_closed_v1",
    )
    assert ordinary.stamp.frame_count == explicit.stamp.frame_count == 1
    assert ordinary.execution_source.options.get("selection_mode") != "single_image"
    assert explicit.execution_source.options["selection_mode"] == "single_image"
    numbered = tmp_path / "empty_0001.tif"; tifffile.imwrite(numbered, np.ones((2, 2), dtype="u1"))
    compact = AverageScanRecipe(
        image_series_spec(numbered, metadata_format=None), tmp_path / "fallback.nxs", ReductionPlan(),
    ).source.thaw()
    monkeypatch.setattr(Path, "iterdir", lambda _self: (_ for _ in ()).throw(OSError("closed")))
    assert graph.qualify_source_execution_graph(compact, reader_binding="average_closed_v1").stamp.frame_count == 1


def test_unmarked_ordinary_tiff_adds_zero_layout_reads_and_uses_null_layout(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(graph, "read_detector_image_layout", lambda *_a, **_k: calls.append(1))
    source = _tiffs(tmp_path)
    original = tuple(source.options["files"])
    tifffile.imwrite(tmp_path / "scan_0003.tif", np.full((2, 3), 3, dtype="u2"))
    value = graph.qualify_source_execution_graph(source)
    assert calls == [] and value.detector_shape is None and value.native_dtype is None
    assert tuple(item.path for item in value.stamp.members) == original
    assert value.reader_binding is None and value.scanned_motor_names is None


def test_source_read_policy_changes_graph_identity_for_auto_vs_txt_metadata(tmp_path, monkeypatch) -> None:
    root = tmp_path / "gráph"; root.mkdir()
    source = _tiffs(root)
    options = dict(source.options)
    auto = SourceSpec(source.uri, source.kind, options={**options, "metadata_format": "auto"})
    txt = SourceSpec(source.uri, source.kind, options={**options, "metadata_format": "txt"})
    auto_graph = graph.qualify_source_execution_graph(auto)
    txt_graph = graph.qualify_source_execution_graph(txt)
    assert graph.source_execution_projection(auto_graph) == graph.source_execution_projection(txt_graph)
    assert graph.source_snapshots_projection(auto_graph, writer=True) == graph.source_snapshots_projection(txt_graph, writer=True)
    payload = graph.source_graph_payload(auto_graph)
    assert set(payload) == {
        "schema_version", "reader_binding", "group_key", "motor_names",
        "scanned_motor_names", "dataset_paths", "detector_layout",
        "source_read_policy", "source_execution", "execution_identity_v1",
        "source_snapshots",
    }
    assert payload["schema_version"] == 1 and payload["reader_binding"] is None
    assert payload["detector_layout"] is None
    assert payload["source_execution"] == graph.source_execution_projection(auto_graph)
    assert payload["execution_identity_v1"] == graph.source_execution_identity_v1_projection(auto_graph)
    assert payload["source_snapshots"] == graph.source_snapshots_projection(auto_graph, writer=True)
    assert not ({"files", "admitted_motor_values"} & set(payload["source_read_policy"]["options"]))
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode()
    expected = hashlib.sha256(b"xdart.source-execution-graph.v1\0" + encoded).hexdigest()
    monkeypatch.setattr(graph.json, "dumps", lambda *_a, **_k: pytest.fail("aggregate JSON retained"))
    assert graph.source_graph_digest(auto_graph) == expected
    assert payload["source_read_policy"] != graph.source_graph_payload(txt_graph)["source_read_policy"]
    assert graph.source_graph_digest(auto_graph) != graph.source_graph_digest(txt_graph)
    monkeypatch.setattr(graph, "qualify_source_execution_graph", lambda *_a, **_k: pytest.fail("second graph built"))
    monkeypatch.setattr(graph, "freeze_source_execution_graph", lambda *_a, **_k: pytest.fail("second graph frozen"))
    assert graph.requalify_source_execution_graph(auto, auto_graph) is auto_graph


def test_graph_bound_average_open_ignores_legacy_same_kind_factory(tmp_path, monkeypatch) -> None:
    source = _tiffs(tmp_path, ("scan_0001.tif",))
    monkeypatch.setattr("xrd_tools.sources.adapters.candidate_owner",
                        lambda *_a, **_k: pytest.fail("closed reader consulted candidate registry"))
    Path(source.options["files"][0]).with_suffix(".txt").write_text(
        "# Counters\nI0 = 3.0\n# Motors\n\n"
        "User: p36, time: Mon Jan 15 10:30:00 2024  # Temp\n"
    )
    source = SourceSpec(source.uri, source.kind, options={
        **dict(source.options), "metadata_format": "txt",
    })
    with pytest.raises(ValueError, match="selected motor"):
        graph.qualify_source_execution_graph(source, selected_motor="theta", reader_binding="average_closed_v1")
    value = graph.qualify_source_execution_graph(
        source, selected_motor="I0", reader_binding="average_closed_v1",
    )
    monkeypatch.setattr("xrd_tools.sources.registry.open_source", lambda *_a, **_k: pytest.fail("legacy registry reached"))
    monkeypatch.setattr("xrd_tools.sources.nexus.NexusStackSource.metadata_for",
                        lambda *_a, **_k: pytest.fail("legacy metadata reached"))
    with graph.open_source_execution_graph(value) as window:
        np.testing.assert_array_equal(window.read_native(0), np.ones((2, 3), dtype=np.uint16))
        row = window.complete_metadata_for(0)
        assert row["I0"] == 3.0 and set(row) == {"I0", "epoch"}
    from xrd_tools.io import metadata as metadata_io
    real_observed = metadata_io.read_image_metadata_observed
    def changed(*args, **kwargs):
        observed = real_observed(*args, **kwargs)
        return metadata_io.ImageMetadataRead({**dict(observed.values), "I0": 4.0}, observed.source_path)
    monkeypatch.setattr(metadata_io, "read_image_metadata_observed", changed)
    monkeypatch.setattr("xrd_tools.sources.image.TiffSeriesSource.__init__",
                        lambda *_a, **_k: pytest.fail("TIFF reader opened before validation"))
    with pytest.raises(graph.SourceRevisionChanged, match="motor"):
        graph.open_source_execution_graph(value).__enter__()


def test_closed_eiger_reader_uses_descriptor_kind_not_runtime_stack_kind(tmp_path, monkeypatch) -> None:
    path, members = _eiger_master(tmp_path)
    monkeypatch.setattr("xrd_tools.sources.adapters.candidate_owner",
                        lambda *_a, **_k: pytest.fail("closed reader consulted candidate registry"))
    value = graph.qualify_source_execution_graph(
        SourceSpec(path, SourceKind.EIGER_MASTER, entry="entry"),
        reader_binding="average_closed_v1",
    )
    assert value.execution_source.kind is value.descriptor.kind is SourceKind.EIGER_MASTER
    assert tuple(Path(item.file.path) for item in value.stamp.external_members) == members
    assert tuple((item.first, item.stop) for item in value.stamp.external_members) == ((0, 2), (2, 5))
    monkeypatch.setattr(nexus_io.NexusImageStack, "kind", SourceKind.NEXUS_STACK, raising=False)
    with graph.open_source_execution_graph(value) as window:
        assert window.extent == 5
        assert int(window.read_native(3)[0, 0]) == 2


def test_bluesky_complete_metadata_rows_are_streaming_and_include_scanned_motor(tmp_path, monkeypatch) -> None:
    from xrd_tools.sources import metadata_provider
    from xrd_tools.sources.cursor import ContainerCursor
    path = _container(tmp_path / "scan.h5")
    value = graph.qualify_source_execution_graph(SourceSpec(path, SourceKind.NEXUS_STACK), reader_binding="average_closed_v1")
    calls = []
    real = metadata_provider.BlueskyMetadataProvider.complete_metadata_for
    monkeypatch.setattr(metadata_provider.BlueskyMetadataProvider, "complete_metadata_for",
                        lambda owner, index: calls.append(index) or real(owner, index))
    monkeypatch.setattr(ContainerCursor, "metadata_for",
                        lambda *_a, **_k: pytest.fail("cursor metadata shortcut reached"))
    monkeypatch.setattr(metadata_provider.BlueskyMetadataProvider, "scan_table",
                        lambda *_a: pytest.fail("whole scan table reached"))
    monkeypatch.setattr(metadata_provider.BlueskyMetadataProvider, "motors",
                        lambda *_a: pytest.fail("whole motor table reached"))
    with graph.open_source_execution_graph(value) as window:
        assert window.complete_metadata_for(2) == {"I0": 3.0, "theta": 2.0}
    assert calls == [2]


def test_million_frame_bluesky_metadata_window_retains_no_table_or_motor_arrays(tmp_path, monkeypatch) -> None:
    from xrd_tools.sources import metadata_provider
    extent = 1_000_000
    path = tmp_path / "large.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["creator"] = "NXWriter"
        entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("end_time", data=np.bytes_("done"))
        detector = entry.create_group("instrument/detector")
        detector.create_dataset("data", shape=(extent, 1, 1), dtype="<u2",
                                chunks=(1, 1, 1), fillvalue=7)
        positioner = entry.create_group("instrument/positioners/theta")
        positioner.create_dataset("value", shape=(extent,), dtype="<f8",
                                  chunks=(131072,), fillvalue=4.5)
        data = entry.create_group("data"); data.attrs["NX_class"] = "NXdata"
        data.create_dataset("theta", shape=(extent,), dtype="<f8",
                            chunks=(131072,), fillvalue=4.5)
        data.create_dataset("I0", shape=(extent,), dtype="<f8",
                            chunks=(131072,), fillvalue=2.0)
    monkeypatch.setattr(metadata_provider.BlueskyMetadataProvider, "scan_table",
                        lambda *_a: pytest.fail("million-row table allocated"))
    monkeypatch.setattr(metadata_provider.BlueskyMetadataProvider, "motors",
                        lambda *_a: pytest.fail("million-row motor allocated"))
    value = graph.qualify_source_execution_graph(SourceSpec(path, SourceKind.NEXUS_STACK), reader_binding="average_closed_v1")
    with graph.open_source_execution_graph(value) as window:
        assert window.extent == extent
        assert window.complete_metadata_for(extent - 1) == {"I0": 2.0, "theta": 4.5}
        owners = (value, window, window._cursor, window._cursor._provider)
        def retains_heavy(value, seen=None):
            seen = set() if seen is None else seen
            if id(value) in seen: return False
            seen.add(id(value))
            if isinstance(value, np.ndarray): return True
            if isinstance(value, Mapping):
                return len(value) >= extent or any(retains_heavy(item, seen) for item in value.values())
            if isinstance(value, (tuple, list)):
                return len(value) >= extent or any(retains_heavy(item, seen) for item in value)
            if is_dataclass(value):
                return any(retains_heavy(getattr(value, item.name), seen) for item in fields(value))
            names = set(vars(value)) if hasattr(value, "__dict__") else set()
            for cls in type(value).__mro__:
                slots = getattr(cls, "__slots__", ())
                names.update((slots,) if isinstance(slots, str) else slots)
            return any(retains_heavy(getattr(value, name), seen) for name in names
                       if name not in {"__dict__", "__weakref__"} and hasattr(value, name))
        assert all(not retains_heavy(owner) for owner in owners)


def test_bluesky_data_catalog_caps_precede_flat_descriptor_enumeration(tmp_path, monkeypatch) -> None:
    from xrd_tools.sources import descriptor as descriptor_module
    missing = _container(tmp_path / "missing.h5", entry="other")
    real_iter = h5py.Group.__iter__; real_keys = h5py.Group.keys
    with monkeypatch.context() as patch:
        patch.setattr(h5py.Group, "__iter__", lambda group: pytest.fail(
            "missing exact hint enumerated root") if group.name == "/" else real_iter(group))
        patch.setattr(h5py.Group, "keys", lambda group: pytest.fail(
            "missing exact hint enumerated root keys") if group.name == "/" else real_keys(group))
        with pytest.raises(ValueError, match="requested entry"):
            graph.qualify_source_execution_graph(
                SourceSpec(missing, SourceKind.NEXUS_STACK, entry="chosen"),
                reader_binding="average_closed_v1",
            )
    custom = _container(tmp_path / "custom.h5", entry="chosen")
    with h5py.File(custom, "r+") as handle:
        del handle["chosen"].attrs["NX_class"]
        other = handle.create_group("other"); other.attrs["NX_class"] = "NXentry"
        other.create_dataset("instrument/detector/data", data=np.zeros((2, 9, 9), dtype="u2"))
    selected = graph.qualify_source_execution_graph(
        SourceSpec(custom, SourceKind.NEXUS_STACK, entry="chosen"),
        reader_binding="average_closed_v1",
    )
    assert selected.descriptor.resolved_entry == "chosen" and selected.detector_shape == (2, 3)
    imageless = _container(tmp_path / "imageless.h5", bluesky=False)
    with h5py.File(imageless, "r+") as handle:
        scan2 = handle.create_group("scan2")
        scan2.create_dataset("end_time", data=np.bytes_("done"))
    monkeypatch.setattr(
        "xrd_tools.io.image._find_hdf5_image_dataset",
        lambda *_a, **_k: pytest.fail("imageless exact group borrowed a detector"),
    )
    with pytest.raises(ValueError, match="image|detector|IMAGELESS"):
        graph.qualify_source_execution_graph(
            SourceSpec(imageless, SourceKind.NEXUS_STACK, entry="scan2"),
            reader_binding="average_closed_v1",
        )
    cap = _container(tmp_path / "cap.h5")
    with h5py.File(cap, "r+") as handle:
        for index in range(4097): handle["entry/data"].create_dataset(f"extra_{index:04d}", data=index)
    monkeypatch.setattr(descriptor_module, "_apstools_flat_stack_paths",
                        lambda *_a, **_k: pytest.fail("descriptor enumerated before catalog cap"))
    monkeypatch.setattr(metadata_provider, "BlueskyMetadataProvider",
                        lambda *_a, **_k: pytest.fail("provider constructed before catalog cap"))
    def rejects_catalog(path):
        with pytest.raises(ValueError, match="AVERAGE_METADATA_CATALOG_TOO_LARGE"):
            graph.qualify_source_execution_graph(
                SourceSpec(path, SourceKind.NEXUS_STACK), reader_binding="average_closed_v1",
            )
    rejects_catalog(cap)
    long_name = _container(tmp_path / "long-name.h5")
    with h5py.File(long_name, "r+") as handle:
        handle["entry/data"]["é" * 129] = handle["entry/data/image"]
    rejects_catalog(long_name)
    aggregate = _container(tmp_path / "aggregate.h5")
    with h5py.File(aggregate, "r+") as handle:
        data = handle["entry/data"]
        for index in range(4093):
            data[f"{index:04d}" + "x" * 252] = data["image"]
        baseline = handle.create_group("entry/instrument/bluesky/streams/baseline")
        for index in range(4):
            baseline[f"b{index}" + "x" * 254] = data["image"]
    rejects_catalog(aggregate)


def test_bluesky_positioner_constant_reads_one_scalar_not_whole_dataset(tmp_path, monkeypatch) -> None:
    path = tmp_path / "positioner.h5"
    _container(path, extent=2)
    with h5py.File(path, "r+") as handle:
        del handle["entry/instrument/positioners/theta/value"]
        handle["entry/instrument/positioners/theta"].create_dataset(
            "value", shape=(1_000_000,), dtype="<f8", chunks=(131072,), fillvalue=4.5,
        )
    reads = []
    real_getitem = h5py.Dataset.__getitem__
    real_array = getattr(h5py.Dataset, "__array__", None)
    def getitem(dataset, key):
        if dataset.name.endswith("/positioners/theta/value"): reads.append(key)
        return real_getitem(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem)
    if real_array is not None:
        monkeypatch.setattr(h5py.Dataset, "__array__", lambda dataset, *_a, **_k:
            pytest.fail(f"whole dataset read: {dataset.name}"))
    monkeypatch.setattr(bluesky_nexus, "_first_numeric",
                        lambda *_a, **_k: pytest.fail("ordinary fallback reached from Average"))
    value = graph.qualify_source_execution_graph(
        SourceSpec(path, SourceKind.NEXUS_STACK), reader_binding="average_closed_v1",
    )
    with graph.open_source_execution_graph(value) as window:
        assert window.complete_metadata_for(1)["theta"] == 1.0
    assert reads == [0]


def test_bluesky_flat_and_grouped_baselines_share_pre_read_admission(
    tmp_path, monkeypatch,
) -> None:
    path = _container(tmp_path / "baseline-valid.h5")
    with h5py.File(path, "r+") as handle:
        base = handle["entry"].require_group("instrument/bluesky/streams/baseline")
        base.create_dataset("flat", data=np.array([2.0], dtype="<f8"))
        for name, field, value in (
            ("start", "value_start", 3.0),
            ("end", "value_end", 4.0),
            ("value", "value", 5.0),
        ):
            base.create_group(name).create_dataset(field, data=np.array([value], dtype="<f8"))
    with h5py.File(path, "r") as handle:
        assert bluesky_nexus.validate_average_container_metadata_inputs(
            handle["entry"], policy="average_bounded_v1",
        )[0] == ("theta",)
        assert bluesky_nexus.bluesky_baseline_values(handle["entry"]) == {
            "end": 4.0, "flat": 2.0, "start": 3.0, "value": 5.0,
        }

    routes = (
        "instrument/bluesky/streams/baseline/flat",
        "instrument/bluesky/streams/baseline/start/value_start",
        "instrument/bluesky/streams/baseline/end/value_end",
        "instrument/bluesky/streams/baseline/value/value",
    )
    for ordinal, route in enumerate(routes):
        bad = _container(tmp_path / f"baseline-bad-{ordinal}.h5")
        with h5py.File(bad, "r+") as handle:
            parent, name = route.rsplit("/", 1)
            group = handle["entry"].require_group(parent)
            if name in group:
                del group[name]
            group.create_dataset(name, data=np.array([b"not numeric"], dtype="S11"))
        reads = []
        real_getitem = h5py.Dataset.__getitem__
        with monkeypatch.context() as patch:
            def guarded(dataset, key):
                if "/baseline/" in dataset.name:
                    reads.append((dataset.name, key))
                return real_getitem(dataset, key)
            patch.setattr(h5py.Dataset, "__getitem__", guarded)
            with h5py.File(bad, "r") as handle:
                with pytest.raises(ValueError, match="AVERAGE_METADATA_INPUT_UNBOUNDED"):
                    bluesky_nexus.validate_average_container_metadata_inputs(
                        handle["entry"], policy="average_bounded_v1",
                    )
        assert reads == []


def test_average_metadata_layout_cap_covers_data_config_mono_and_vds(
    tmp_path, monkeypatch,
) -> None:
    exact = _container(tmp_path / "layout-exact.h5", extent=2)
    with h5py.File(exact, "r+") as handle:
        entry = handle["entry"]
        for route in (
            "instrument/positioners/theta/value", "data/theta",
        ):
            parent, name = route.rsplit("/", 1)
            del entry[route]
            entry[parent].create_dataset(
                name, shape=(131_072,), dtype="<f8", chunks=(131_072,),
                fillvalue=2.0,
            )
        base = entry.require_group("instrument/bluesky/streams/baseline")
        base.create_dataset("direct", data=np.array([3.0], dtype="<f8"))
        grouped = base.create_group("grouped")
        grouped.create_dataset("value_start", data=np.array([4.0], dtype="<f8"))
        base.create_group("grouped_end").create_dataset(
            "value_end", data=np.array([5.0], dtype="<f8"),
        )
        base.create_group("grouped_value").create_dataset(
            "value", data=np.array([6.0], dtype="<f8"),
        )
        config = entry.require_group(
            "instrument/bluesky/metadata/configuration/eiger/data"
        )
        config.create_dataset("eiger_cam_wavelength", data=np.float64(1.2))
        mono = entry.require_group("instrument/monochromator")
        mono.create_dataset("energy", data=np.array([12.0], dtype="<f8"))
    payload_reads = []
    real_getitem = h5py.Dataset.__getitem__
    def indexed(dataset, key):
        if dataset.name.endswith(("/positioners/theta/value", "/data/theta")):
            payload_reads.append((dataset.name, key))
        return real_getitem(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", indexed)
    open_cache = []
    live_metadata = []; metadata_peak = [0]
    real_file = h5py.File
    real_get = h5py.Group.get
    real_item = h5py.Group.__getitem__
    def opened(*args, **kwargs):
        open_cache.append(kwargs.get("rdcc_nbytes"))
        return real_file(*args, **kwargs)
    monkeypatch.setattr(graph, "_HDF5_FILE_OPEN", opened)
    monkeypatch.setattr(cursor_module, "_HDF5_FILE_OPEN", opened)
    def record_dataset(value):
        if isinstance(value, h5py.Dataset) and value.name.endswith((
            "/positioners/theta/value", "/data/theta", "/data/I0",
            "/baseline/direct", "/value_start", "/value_end", "/value",
            "/eiger_cam_wavelength", "/monochromator/energy", "/metadata/motors",
        )):
            live_metadata[:] = [(identifier, ref) for identifier, ref in live_metadata
                                if ref() is not None]
            identifier = int(value.id.id)
            if identifier not in {item for item, _ref in live_metadata}:
                live_metadata.append((identifier, weakref.ref(value)))
            metadata_peak[0] = max(metadata_peak[0], len(live_metadata))
        return value
    def tracked_get(group, name, *args, **kwargs):
        return record_dataset(real_get(group, name, *args, **kwargs))
    def tracked_item(group, name):
        return record_dataset(real_item(group, name))
    monkeypatch.setattr(h5py.Group, "get", tracked_get)
    monkeypatch.setattr(h5py.Group, "__getitem__", tracked_item)
    value = graph.qualify_source_execution_graph(
        SourceSpec(exact, SourceKind.NEXUS_STACK),
        reader_binding="average_closed_v1",
    )
    with graph.open_source_execution_graph(value) as window:
        row = window.complete_metadata_for(1)
        assert row["theta"] == 2.0
        provider = window._cursor._provider
        assert not any(isinstance(item, h5py.Dataset) for item in vars(provider).values())
    from xrd_tools.sources.cursor import ContainerCursor
    with ContainerCursor(exact) as ordinary:
        assert ordinary.descriptor.frame_count == 2
    assert open_cache == [1 << 20, 1 << 20, 1 << 20, None]
    assert metadata_peak[0] <= 3
    assert payload_reads == [
        ("/entry/data/theta", 1),
        ("/entry/instrument/positioners/theta/value", 0),
    ]

    routes = (
        "instrument/positioners/theta/value",
        "instrument/bluesky/streams/baseline/direct",
        "instrument/bluesky/streams/baseline/grouped/value_start",
        "instrument/bluesky/streams/baseline/grouped_end/value_end",
        "instrument/bluesky/streams/baseline/grouped_value/value",
        "data/theta",
        "instrument/bluesky/metadata/configuration/eiger/data/eiger_cam_wavelength",
        "instrument/monochromator/energy",
        "instrument/bluesky/metadata/motors",
    )
    for ordinal, route in enumerate(routes):
        for virtual in (False, True):
            path = _container(tmp_path / f"layout-{ordinal}-{virtual}.h5")
            with real_file(path, "r+") as handle:
                entry = handle["entry"]
                parent, name = route.rsplit("/", 1)
                group = entry.require_group(parent)
                if name in group:
                    del group[name]
                if route.endswith("/motors") and "instrument/positioners" in entry:
                    del entry["instrument/positioners"]
                dtype = "S1" if route.endswith("/motors") else "<f8"
                fill = b"-" if route.endswith("/motors") else 1.0
                if virtual:
                    backing = entry.create_dataset(
                        f"virtual_backing_{ordinal}", data=np.array([fill], dtype=dtype),
                    )
                    layout = h5py.VirtualLayout(shape=(1,), dtype=dtype)
                    layout[:] = h5py.VirtualSource(
                        str(path), backing.name, shape=(1,),
                    )
                    group.create_virtual_dataset(name, layout)
                else:
                    extent = ((1 << 20) + 1 if route.endswith("/motors") else 131_073)
                    group.create_dataset(
                        name, shape=(extent,), dtype=dtype,
                        chunks=(extent,), fillvalue=fill,
                    )
            touched = []
            with monkeypatch.context() as patch:
                def forbidden(dataset, key):
                    if dataset.name.endswith(f"/{name}"):
                        touched.append(key)
                        raise AssertionError("metadata payload read before layout refusal")
                    return real_getitem(dataset, key)
                patch.setattr(h5py.Dataset, "__getitem__", forbidden)
                patch.setattr("xrd_tools.sources.descriptor._describe_container_from_open_with_binding",
                    lambda *_a, **_k: pytest.fail("descriptor reached after invalid metadata"))
                patch.setattr(metadata_provider, "BlueskyMetadataProvider",
                    lambda *_a, **_k: pytest.fail("provider reached after invalid metadata"))
                diagnostic = ("AVERAGE_METADATA_VIRTUAL_UNSUPPORTED" if virtual
                              else "AVERAGE_METADATA_CHUNK_TOO_LARGE")
                with pytest.raises(ValueError, match=diagnostic):
                    graph.qualify_source_execution_graph(
                        SourceSpec(path, SourceKind.NEXUS_STACK),
                        reader_binding="average_closed_v1",
                    )
            assert touched == []


def test_average_metadata_indirection_and_external_storage_refuse_before_follow(
    tmp_path, monkeypatch,
) -> None:
    external = tmp_path / "missing-external.h5"
    real_getitem = h5py.Dataset.__getitem__
    routes = (
        "instrument/positioners/theta/value",
        "instrument/bluesky/streams/baseline/direct",
        "instrument/bluesky/streams/baseline/grouped/value_start",
        "instrument/bluesky/streams/baseline/grouped_end/value_end",
        "instrument/bluesky/streams/baseline/grouped_value/value",
        "data/I0",
        "instrument/bluesky/metadata/configuration/eiger/data/eiger_cam_wavelength",
        "instrument/monochromator/energy",
        "instrument/bluesky/metadata/motors",
    )
    targets = {}
    for route in routes:
        parts = route.split("/")
        for stop in range(1, len(parts) + 1):
            targets.setdefault("/".join(parts[:stop]), route)
    cases = ((route, target_route, link) for target_route, route in targets.items()
             for link in ("soft", "external"))
    for ordinal, (route, target_route, link) in enumerate(cases):
        path = _container(tmp_path / f"indirect-{ordinal}.h5")
        with h5py.File(path, "r+") as handle:
            entry = handle["entry"]
            parent, name = route.rsplit("/", 1)
            group = entry.require_group(parent)
            if name not in group:
                value = (np.bytes_("!!python/tuple\n- theta\n")
                         if route.endswith("/motors") else np.array([1.0], dtype="<f8"))
                group.create_dataset(name, data=value)
            if route.endswith("/motors"):
                del entry["instrument/positioners"]
            if "/" in target_route:
                target_parent, target_name = target_route.rsplit("/", 1)
                link_group = entry[target_parent]
            else:
                target_name = target_route; link_group = entry
            if link == "soft":
                retained = f"{target_route}_retained_{ordinal}"
                entry.move(target_route, retained)
                link_group[target_name] = h5py.SoftLink(f"/entry/{retained}")
            else:
                del link_group[target_name]
                link_group[target_name] = h5py.ExternalLink(str(external), "/values")
        reads = []
        with monkeypatch.context() as patch:
            real_group_get = h5py.Group.get
            real_group_item = h5py.Group.__getitem__
            link_path = f"/entry/{target_route}"
            def requested(group, name):
                raw = str(name)
                candidate = (raw if raw.startswith("/") else
                             f"{group.name.rstrip('/')}/{raw}")
                return candidate == link_path or candidate.startswith(f"{link_path}/")
            def guarded_get(group, name, *args, **kwargs):
                if requested(group, name) and kwargs.get("getlink") is not True:
                    raise AssertionError("metadata link target followed")
                return real_group_get(group, name, *args, **kwargs)
            def guarded_item(group, name):
                if requested(group, name):
                    raise AssertionError("metadata link target indexed")
                return real_group_item(group, name)
            def forbidden(dataset, key):
                reads.append((dataset.name, key))
                raise AssertionError("indirect metadata payload followed")
            patch.setattr(h5py.Group, "get", guarded_get)
            patch.setattr(h5py.Group, "__getitem__", guarded_item)
            patch.setattr(h5py.Dataset, "__getitem__", forbidden)
            patch.setattr("xrd_tools.sources.descriptor._describe_container_from_open_with_binding",
                lambda *_a, **_k: pytest.fail("descriptor followed invalid metadata"))
            patch.setattr(metadata_provider, "BlueskyMetadataProvider",
                lambda *_a, **_k: pytest.fail("provider followed invalid metadata"))
            with pytest.raises(ValueError, match="AVERAGE_METADATA_INDIRECTION_UNSUPPORTED"):
                graph.qualify_source_execution_graph(
                    SourceSpec(path, SourceKind.NEXUS_STACK),
                    reader_binding="average_closed_v1",
                )
        assert reads == []

    raw = tmp_path / "external.raw"; raw.write_bytes(b"\0" * 8)
    path = _container(tmp_path / "external-storage.h5")
    with h5py.File(path, "r+") as handle:
        del handle["entry/data/I0"]
        handle["entry/data"].create_dataset(
            "I0", shape=(1,), dtype="<f8", external=[(str(raw), 0, 8)],
        )
    raw.unlink()
    with pytest.raises(ValueError, match="AVERAGE_METADATA_INDIRECTION_UNSUPPORTED"):
        graph.qualify_source_execution_graph(
            SourceSpec(path, SourceKind.NEXUS_STACK),
            reader_binding="average_closed_v1",
        )

    eiger_root = tmp_path / "eiger"; eiger_root.mkdir()
    master, members = _eiger_master(eiger_root)
    accepted = graph.qualify_source_execution_graph(
        SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry"),
        reader_binding="average_closed_v1",
    )
    assert tuple(Path(item.file.path) for item in accepted.stamp.external_members) == members
    assert tuple((item.dataset, item.first, item.stop, item.epoch)
                 for item in accepted.stamp.external_members) == (
        ("/entry/data/data", 0, 2, 0), ("/entry/data/data", 2, 5, 1),
    )
    altered = replace(accepted, stamp=replace(
        accepted.stamp, external_members=(
            replace(accepted.stamp.external_members[0], stop=1),
            *accepted.stamp.external_members[1:],
        ),
    ))
    with monkeypatch.context() as patch:
        patch.setattr(cursor_module.ContainerCursor, "open",
                      lambda *_a, **_k: pytest.fail("container opened before range proof"))
        with pytest.raises(graph.SourceRevisionChanged, match="external"):
            graph.open_source_execution_graph(altered).__enter__()
    with h5py.File(master, "r+") as handle:
        del handle["entry/data/data_000002"]
        handle["entry/data/data_000002"] = h5py.ExternalLink(
            members[0].name, "/entry/data/data",
        )
    with pytest.raises(graph.SourceRevisionChanged):
        graph.requalify_source_execution_graph(
            SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry"), accepted,
            reader_binding="average_closed_v1",
        )


def test_average_prevalidated_motor_names_eliminate_per_row_yaml(
    tmp_path, monkeypatch,
) -> None:
    authoritative = _container(tmp_path / "authoritative.h5")
    real_parse = bluesky_nexus._parse_motors_yaml
    real_average_parse = bluesky_nexus._average_manifest_motor_names
    parse_calls = []
    monkeypatch.setattr(
        bluesky_nexus, "_parse_motors_yaml",
        lambda entry: parse_calls.append(entry.name) or real_parse(entry),
    )
    monkeypatch.setattr(
        bluesky_nexus, "_average_manifest_motor_names",
        lambda manifest: (
            parse_calls.append(
                "/" + manifest.name.lstrip("/").split("/", 1)[0]
            ) or real_average_parse(manifest)
        ),
    )
    value = graph.qualify_source_execution_graph(
        SourceSpec(authoritative, SourceKind.NEXUS_STACK),
        reader_binding="average_closed_v1",
    )
    assert parse_calls == []
    with graph.open_source_execution_graph(value) as window:
        assert window.complete_metadata_for(0)["theta"] == 0.0
        assert window.complete_metadata_for(1)["theta"] == 1.0
    assert parse_calls == []
    assert graph.requalify_source_execution_graph(
        SourceSpec(authoritative, SourceKind.NEXUS_STACK), value,
        reader_binding="average_closed_v1",
    ) is value
    assert parse_calls == []
    with h5py.File(authoritative, "r") as handle:
        assert bluesky_nexus.bluesky_motor_names(handle["entry"]) == ["theta"]
    assert parse_calls == ["/entry"]

    fallback = _container(tmp_path / "fallback.h5")
    with h5py.File(fallback, "r+") as handle:
        del handle["entry/instrument/positioners"]
        metadata = handle["entry/instrument/bluesky/metadata"]
        del metadata["motors"]
        metadata.create_dataset(
            "motors", data=np.bytes_("!!python/tuple\n- theta\n- theta\n"),
        )
    parse_calls.clear()
    source = SourceSpec(fallback, SourceKind.NEXUS_STACK)
    prepared = graph.qualify_source_execution_graph(
        source, reader_binding="average_closed_v1",
    )
    assert parse_calls == ["/entry"]
    assert prepared.scanned_motor_names == ("theta", "theta")
    assert prepared.motor_names[:2] == ("theta", "theta")
    parse_calls.clear()
    with graph.open_source_execution_graph(prepared) as window:
        window.complete_metadata_for(0)
        window.complete_metadata_for(1)
    assert parse_calls == ["/entry", "/entry"]
    parse_calls.clear()
    assert graph.requalify_source_execution_graph(
        source, prepared, reader_binding="average_closed_v1",
    ) is prepared
    assert parse_calls == ["/entry"]

    ambiguous = _container(tmp_path / "ambiguous-counter.h5")
    with h5py.File(ambiguous, "r+") as handle:
        handle["entry/data"].create_dataset("i0", data=np.ones(3, dtype="<f8"))
    with pytest.raises(ValueError, match="AVERAGE_METADATA_COUNTER_AMBIGUOUS"):
        graph.qualify_source_execution_graph(
            SourceSpec(ambiguous, SourceKind.NEXUS_STACK),
            reader_binding="average_closed_v1",
        )


def test_average_exact_entry_and_motor_pair_bind_descriptor_provider_and_graph(
    tmp_path, monkeypatch,
) -> None:
    path = _container(tmp_path / "custom.h5", entry="chosen")
    with h5py.File(path, "r+") as handle:
        base = handle["chosen"].require_group("instrument/bluesky/streams/baseline")
        base.create_dataset("fixed", data=np.array([4.0], dtype="<f8"))
        base.create_dataset("fixed_user_setpoint", data=np.array([4.0], dtype="<f8"))
    source = SourceSpec(path, SourceKind.NEXUS_STACK, entry="chosen")
    with pytest.raises(ValueError, match="selected motor"):
        graph.qualify_source_execution_graph(
            source, selected_motor="missing", reader_binding="average_closed_v1",
        )
    value = graph.qualify_source_execution_graph(
        source, selected_motor="theta", reader_binding="average_closed_v1",
    )
    assert value.descriptor.resolved_entry == "chosen"
    assert value.scanned_motor_names == ("theta",)
    assert value.motor_names == ("theta", "fixed")
    payload = graph.source_graph_payload(value)
    assert tuple(payload["scanned_motor_names"]) == value.scanned_motor_names
    assert tuple(payload["motor_names"]) == value.motor_names
    with graph.open_source_execution_graph(value) as window:
        cursor = window._cursor
        provider = cursor.metadata_provider()
        assert cursor._expected_scanned_motor_names is value.scanned_motor_names
        assert cursor._expected_all_motor_names is value.motor_names
        assert provider._scanned_motor_names is value.scanned_motor_names
        assert provider._all_motor_names is value.motor_names
        assert window.complete_metadata_for(1)["fixed"] == 4.0

    real_validate = bluesky_nexus.validate_average_container_metadata_inputs
    monkeypatch.setattr(
        bluesky_nexus, "validate_average_container_metadata_inputs",
        lambda *_a, **_k: (("fixed",), value.motor_names),
    )
    monkeypatch.setattr(
        "xrd_tools.sources.descriptor._describe_container_from_open_with_binding",
        lambda *_a, **_k: pytest.fail("descriptor reached before pair comparison"),
    )
    with pytest.raises(graph.SourceRevisionChanged, match="motor catalog changed"):
        graph.open_source_execution_graph(value).__enter__()
    with pytest.raises(graph.SourceRevisionChanged, match="motor catalog changed"):
        graph.requalify_source_execution_graph(
            source, value, selected_motor="theta", reader_binding="average_closed_v1",
        )
    monkeypatch.setattr(
        bluesky_nexus, "validate_average_container_metadata_inputs", real_validate,
    )
    for policy, scanned, all_names in (
        (None, ("theta",), ("theta",)),
        ("average_bounded_v1", None, ("theta",)),
        ("average_bounded_v1", ["theta"], ("theta",)),
    ):
        with pytest.raises(TypeError):
            cursor_module.ContainerCursor(
                path, metadata_input_policy=policy,
                expected_scanned_motor_names=scanned,
                expected_all_motor_names=all_names,
            )
    for scanned, all_names in ((None, ("theta",)), (["theta"], ("theta",))):
        with pytest.raises(TypeError):
            metadata_provider.metadata_provider_for_open_entry(
                None, frame_count=1, scanned_motor_names=scanned,
                all_motor_names=all_names,
            )


def test_average_exact_entry_group_survives_root_relink_without_rediscovery(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.sources import descriptor as descriptor_module
    path = _container(tmp_path / "relink.h5", entry="chosen")
    real_file = h5py.File
    monkeypatch.setattr(
        graph, "_HDF5_FILE_OPEN",
        lambda path, _mode="r", **kwargs: real_file(path, "r+", **kwargs),
    )
    real_describe = descriptor_module._describe_container_from_open_with_binding
    observed = []; armed = [False]
    real_group_item = h5py.Group.__getitem__; real_group_get = h5py.Group.get
    def guarded_item(group, name):
        if armed[0] and group.name == "/" and name == "chosen":
            pytest.fail("captured entry was reacquired by root name")
        return real_group_item(group, name)
    def guarded_get(group, name, *args, **kwargs):
        if armed[0] and group.name == "/" and name == "chosen":
            pytest.fail("captured entry was re-resolved by root name")
        return real_group_get(group, name, *args, **kwargs)
    monkeypatch.setattr(h5py.Group, "__getitem__", guarded_item)
    monkeypatch.setattr(h5py.Group, "get", guarded_get)
    def relink(*args, **kwargs):
        group = kwargs["resolved_entry_group"]
        observed.append(group)
        root = group.file
        root["captured"] = group
        del root["chosen"]
        replacement = root.create_group("chosen")
        replacement.attrs["NX_class"] = "NXentry"
        replacement.create_dataset(
            "instrument/detector/data", data=np.zeros((2, 9, 9), dtype="u2"),
        )
        armed[0] = True
        try:
            return real_describe(*args, **kwargs)
        except graph.SourceRevisionChanged:
            raise
    monkeypatch.setattr(descriptor_module, "_describe_container_from_open_with_binding", relink)
    for name in (
        "resolve_nxentry", "_resolved_entry_name", "_entry_and_root",
        "_find_hdf5_image_dataset",
    ):
        if hasattr(descriptor_module, name):
            monkeypatch.setattr(
                descriptor_module, name,
                lambda *_a, _name=name, **_k: pytest.fail(f"root rediscovery reached: {_name}"),
            )
    source = SourceSpec(path, SourceKind.NEXUS_STACK, entry="chosen")
    try:
        value = graph.qualify_source_execution_graph(
            source, reader_binding="average_closed_v1",
        )
    except graph.SourceRevisionChanged:
        value = None
    assert len(observed) == 1
    if value is not None:
        assert value.detector_shape == (2, 3)
        assert value.descriptor.resolved_entry in ("chosen", "captured")
        assert value.detector_shape != (9, 9)


def test_average_fallback_motor_names_are_direct_components_before_lookup(
    tmp_path, monkeypatch,
) -> None:
    bad_names = ("", "/entry1/data/evil", "nested/evil", ".", "..", "bad\0name")
    real_get = h5py.Group.get; real_item = h5py.Group.__getitem__
    for ordinal, bad_name in enumerate(bad_names):
        path = _container(tmp_path / f"motor-name-{ordinal}.h5")
        with h5py.File(path, "r+") as handle:
            del handle["entry/instrument/positioners"]
            metadata = handle["entry/instrument/bluesky/metadata"]
            del metadata["motors"]
            encoded = "''" if bad_name == "" else bad_name
            metadata.create_dataset(
                "motors", data=np.bytes_(f"!!python/tuple\n- {encoded}\n"),
            )
        lookups = []
        with monkeypatch.context() as patch:
            def guarded(group, name, *args, **kwargs):
                if name == bad_name:
                    lookups.append(("get", name))
                    raise AssertionError("invalid motor spelling reached HDF5 lookup")
                return real_get(group, name, *args, **kwargs)
            def guarded_item(group, name):
                if name == bad_name:
                    lookups.append(("getitem", name))
                    raise AssertionError("invalid motor spelling reached HDF5 indexing")
                return real_item(group, name)
            patch.setattr(h5py.Group, "get", guarded)
            patch.setattr(h5py.Group, "__getitem__", guarded_item)
            with pytest.raises(ValueError, match="AVERAGE_METADATA_MOTOR_NAME_INVALID"):
                graph.qualify_source_execution_graph(
                    SourceSpec(path, SourceKind.NEXUS_STACK),
                    reader_binding="average_closed_v1",
                )
        assert lookups == []


def test_average_motor_tuple_occurrence_caps_apply_to_scanned_and_all(
    tmp_path, monkeypatch,
) -> None:
    def fallback_file(name: str, occurrences: int, *, baseline=False) -> Path:
        path = _container(tmp_path / name)
        with h5py.File(path, "r+") as handle:
            entry = handle["entry"]
            del entry["instrument/positioners"]
            manifest = "!!python/tuple\n" + "- theta\n" * occurrences
            metadata = entry["instrument/bluesky/metadata"]
            del metadata["motors"]
            metadata.create_dataset("motors", data=np.bytes_(manifest))
            if baseline:
                base = entry.require_group("instrument/bluesky/streams/baseline")
                base.create_dataset("fixed", data=np.array([1.0], dtype="<f8"))
                base.create_dataset("fixed_user_setpoint", data=np.array([1.0], dtype="<f8"))
        return path

    exact = fallback_file("exact-256.h5", 256)
    admitted = graph.qualify_source_execution_graph(
        SourceSpec(exact, SourceKind.NEXUS_STACK),
        reader_binding="average_closed_v1",
    )
    assert admitted.scanned_motor_names == ("theta",) * 256
    assert admitted.motor_names == ("theta",) * 256
    assert tuple(graph.source_graph_payload(admitted)["scanned_motor_names"]) == (
        "theta",
    ) * 256

    for path in (
        fallback_file("scanned-257.h5", 257),
        fallback_file("all-257.h5", 256, baseline=True),
    ):
        with monkeypatch.context() as patch:
            patch.setattr("xrd_tools.sources.descriptor._describe_container_from_open_with_binding",
                          lambda *_a, **_k: pytest.fail("descriptor preceded motor cap"))
            patch.setattr(metadata_provider, "BlueskyMetadataProvider",
                          lambda *_a, **_k: pytest.fail("provider preceded motor cap"))
            with pytest.raises(ValueError, match="AVERAGE_METADATA_CATALOG_TOO_LARGE"):
                graph.qualify_source_execution_graph(
                    SourceSpec(path, SourceKind.NEXUS_STACK),
                    reader_binding="average_closed_v1",
                )


def test_average_binding_closes_partial_eiger_ids_on_every_throwable(
    tmp_path, monkeypatch,
) -> None:
    root = tmp_path / "partial"; root.mkdir()
    master, _members = _eiger_master(root, (2, 2, 2))
    with h5py.File(master, "r+") as handle:
        del handle["entry/data/data_000003"]
        handle["entry/data/data_000003"] = h5py.ExternalLink(
            "missing.h5", "/entry/data/data",
        )
    real_close = nexus_io._ResolvedNexusStack._close_dataset_id
    closed_ids = {}
    def recorded_close(dataset):
        identifier = dataset.id
        key = id(identifier); closed_ids[key] = closed_ids.get(key, 0) + 1
        return real_close(dataset)
    monkeypatch.setattr(
        nexus_io._ResolvedNexusStack, "_close_dataset_id",
        staticmethod(recorded_close),
    )
    with h5py.File(master, "r") as handle:
        slot = nexus_io._NexusDatasetOwnerSlot()
        with pytest.raises((KeyError, OSError, ValueError)):
            nexus_io._bind_nexus_stack_from_entry(
                handle["entry"], declared_entry_name="entry",
                prefer_apstools_flat=False, owner_slot=slot,
            )
        assert slot.owner is not None
        assert slot.owner.state == "CLOSED"
        assert len(closed_ids) == 2 and set(closed_ids.values()) == {1}

    for mode in ("rank", "layout"):
        closed_ids.clear()
        root_bad = tmp_path / mode; root_bad.mkdir()
        master_bad, members_bad = _eiger_master(root_bad, (2, 2))
        with h5py.File(members_bad[1], "w") as handle:
            shape = (2, 3) if mode == "rank" else (2, 3, 3)
            handle.create_dataset("entry/data/data", data=np.ones(shape, dtype="u2"))
        with h5py.File(master_bad, "r") as handle:
            slot = nexus_io._NexusDatasetOwnerSlot()
            expected = (nexus_io.UnsupportedDetectorRankError
                        if mode == "rank" else ValueError)
            with pytest.raises(expected, match="rank|shape|Inconsistent"):
                nexus_io._bind_nexus_stack_from_entry(
                    handle["entry"], declared_entry_name="entry",
                    prefer_apstools_flat=False, owner_slot=slot,
                )
            assert slot.owner is not None and slot.owner.state == "CLOSED"
            captured = tuple(dataset.id for dataset in slot.owner._datasets)
            assert len(captured) == 2 and all(not identifier.valid for identifier in captured)
            assert {id(identifier) for identifier in captured} == set(closed_ids)
            assert set(closed_ids.values()) == {1}

    root2 = tmp_path / "base-exception"; root2.mkdir()
    master2, _ = _eiger_master(root2, (2, 2))
    closed_ids.clear()
    with h5py.File(master2, "r") as handle:
        entry = handle["entry"]
        slot = nexus_io._NexusDatasetOwnerSlot()
        real_item = h5py.Group.__getitem__
        with monkeypatch.context() as patch:
            def interrupted(group, name):
                if group.name == "/entry/data" and name == "data_000002":
                    raise KeyboardInterrupt("injected bind interruption")
                return real_item(group, name)
            patch.setattr(h5py.Group, "__getitem__", interrupted)
            with pytest.raises(KeyboardInterrupt, match="bind interruption"):
                nexus_io._bind_nexus_stack_from_entry(
                    entry, declared_entry_name="entry",
                    prefer_apstools_flat=False, owner_slot=slot,
                )
        assert slot.owner is not None and slot.owner.state == "CLOSED"
        cancelled_ids = tuple(dataset.id for dataset in slot.owner._datasets)
        assert len(cancelled_ids) == 1
        assert all(not identifier.valid for identifier in cancelled_ids)
        assert {id(identifier) for identifier in cancelled_ids} == set(closed_ids)
        assert set(closed_ids.values()) == {1}

    root3 = tmp_path / "close-retry"; root3.mkdir()
    master3, _ = _eiger_master(root3, (2, 2))
    with h5py.File(master3, "r+") as handle:
        del handle["entry/data/data_000002"]
        handle["entry/data/data_000002"] = h5py.ExternalLink(
            "missing.h5", "/entry/data/data",
        )
    failed_id = []; physical = []
    with h5py.File(master3, "r") as handle:
        slot = nexus_io._NexusDatasetOwnerSlot()
        with monkeypatch.context() as patch:
            def fail_first_close(dataset):
                identifier = dataset.id
                physical.append(id(identifier))
                if identifier.valid and not failed_id:
                    failed_id.append(int(identifier.id))
                    raise OSError("injected Dataset-ID close failure")
                return real_close(dataset)
            patch.setattr(
                nexus_io._ResolvedNexusStack, "_close_dataset_id",
                staticmethod(fail_first_close),
            )
            with pytest.raises(OSError, match="close failure"):
                nexus_io._bind_nexus_stack_from_entry(
                    handle["entry"], declared_entry_name="entry",
                    prefer_apstools_flat=False, owner_slot=slot,
                )
            assert slot.owner is not None
            assert slot.owner.state == "BINDING_OWNS_DATASET_IDS"
            retained = tuple(dataset.id for dataset in slot.owner._datasets)
            valid = tuple(identifier for identifier in retained if identifier.valid)
            assert len(retained) == len(physical) == 1
            assert len(set(physical)) == 1 and set(physical) == {
                id(identifier) for identifier in retained
            }
            assert len(valid) == 1 and int(valid[0].id) == failed_id[0]
            assert all(physical.count(id(identifier)) == 1 for identifier in retained)
            slot.owner.close()
            assert physical[-1] == id(valid[0])
            assert physical.count(id(valid[0])) == 2
            assert all(physical.count(id(identifier)) == 1
                       for identifier in retained if identifier is not valid[0])
            assert slot.owner.state == "CLOSED"


def test_average_binding_atomic_transfer_has_one_close_authority(
    tmp_path, monkeypatch,
) -> None:
    path = _container(tmp_path / "transfer.h5")
    real_close = nexus_io._ResolvedNexusStack._close_dataset_id
    close_counts = {}; identity_anchors = {}
    def counted(dataset):
        identifier = dataset.id
        key = id(identifier); identity_anchors[key] = identifier; close_counts[key] = close_counts.get(key, 0) + 1
        return real_close(dataset)
    monkeypatch.setattr(
        nexus_io._ResolvedNexusStack, "_close_dataset_id",
        staticmethod(counted),
    )
    handle = h5py.File(path, "r")
    slot, binding = _bind_stack(handle)
    original_bound_constructor = nexus_io.NexusImageStack.__dict__["_from_bound_datasets"]
    owned = tuple(id(dataset.id) for dataset in binding._datasets)
    stack = binding.into_nexus_image_stack(handle, owner_slot=slot)
    assert binding.state == "STACK_OWNS_DATASET_IDS"
    assert slot.owner is stack
    binding.close()
    assert all(close_counts.get(identifier, 0) == 0 for identifier in owned)
    stack.close()
    assert all(close_counts.get(identifier, 0) == 1 for identifier in owned)
    binding.close(); stack.close()
    assert all(close_counts.get(identifier, 0) == 1 for identifier in owned)

    handle = h5py.File(path, "r")
    slot, binding = _bind_stack(handle)
    failed_owned = tuple(id(dataset.id) for dataset in binding._datasets)
    monkeypatch.setattr(
        nexus_io.NexusImageStack, "_from_bound_datasets",
        classmethod(lambda *_a, **_k: (_ for _ in ()).throw(
            KeyboardInterrupt("unpublished stack"))),
    )
    with pytest.raises(KeyboardInterrupt, match="unpublished stack"):
        binding.into_nexus_image_stack(handle, owner_slot=slot)
    assert slot.owner is binding
    assert binding.state == "BINDING_OWNS_DATASET_IDS"
    binding.close(); handle.close()
    assert binding.state == "CLOSED"
    assert all(close_counts.get(identifier, 0) == 1 for identifier in failed_owned)

    monkeypatch.setattr(nexus_io.NexusImageStack, "_from_bound_datasets",
                        original_bound_constructor)
    handle = h5py.File(path, "r"); slot, binding = _bind_stack(handle)
    anchor_ids = tuple(id(dataset.id) for dataset in binding._datasets)
    slot_setattr = nexus_io._NexusDatasetOwnerSlot.__setattr__
    def fail_anchor(owner, name, value):
        if owner is slot and name == "owner" and value is not binding:
            raise KeyboardInterrupt("stack anchor")
        return slot_setattr(owner, name, value)
    monkeypatch.setattr(nexus_io._NexusDatasetOwnerSlot, "__setattr__", fail_anchor)
    with pytest.raises(KeyboardInterrupt, match="stack anchor"):
        binding.into_nexus_image_stack(handle, owner_slot=slot)
    assert slot.owner is binding and binding.state == "BINDING_OWNS_DATASET_IDS"
    assert all(close_counts.get(identifier, 0) == 0 for identifier in anchor_ids)
    monkeypatch.setattr(nexus_io._NexusDatasetOwnerSlot, "__setattr__", slot_setattr)
    binding.close(); handle.close()
    assert all(close_counts.get(identifier, 0) == 1 for identifier in anchor_ids)

    from xrd_tools.sources.cursor import ContainerCursor
    source = SourceSpec(path, SourceKind.NEXUS_STACK)
    prepared = graph.qualify_source_execution_graph(
        source, reader_binding="average_closed_v1",
    )
    into = nexus_io._ResolvedNexusStack.into_nexus_image_stack; returned = []
    def throw_during_return(owner, *args, **kwargs):
        stack = into(owner, *args, **kwargs)
        returned.append((stack, tuple(dataset.id for dataset in stack._dsets)))
        raise KeyboardInterrupt("stack return")
    monkeypatch.setattr(nexus_io._ResolvedNexusStack, "into_nexus_image_stack",
                        throw_during_return)
    return_window = graph.open_source_execution_graph(prepared)
    with pytest.raises(KeyboardInterrupt, match="stack return"):
        return_window.__enter__()
    returned_stack, returned_ids = returned[0]
    returned_cursor = return_window._cursor; assert returned_cursor is not None
    assert (returned_cursor._opening_stack_owner.owner is returned_stack
            or all(not identifier.valid for identifier in returned_ids))
    monkeypatch.setattr(nexus_io._ResolvedNexusStack, "into_nexus_image_stack", into)
    return_window.close()
    assert all(not identifier.valid and close_counts.get(id(identifier), 0) == 1
               for identifier in returned_ids)

    window = graph.open_source_execution_graph(prepared)
    original_setattr = ContainerCursor.__setattr__; interrupted = []
    def assigned_then_interrupted(owner, name, value):
        original_setattr(owner, name, value)
        if name == "_stack" and value is not None and not interrupted:
            interrupted.append((value, tuple(dataset.id for dataset in value._dsets)))
            raise KeyboardInterrupt("cursor stack promotion")
    monkeypatch.setattr(ContainerCursor, "__setattr__", assigned_then_interrupted)
    with pytest.raises(KeyboardInterrupt, match="stack promotion"):
        window.__enter__()
    cursor = window._cursor; assert cursor is not None
    promoted, identifiers = interrupted[0]
    assert (cursor._stack is promoted or cursor._opening_stack_owner.owner is promoted
            or all(not identifier.valid for identifier in identifiers))
    monkeypatch.setattr(ContainerCursor, "__setattr__", original_setattr)
    window.close()
    assert all(not identifier.valid and close_counts.get(id(identifier), 0) == 1
               for identifier in identifiers)


def test_average_promoted_stack_close_pending_retains_owner_and_revalidates(
    tmp_path, monkeypatch,
) -> None:
    path = _container(tmp_path / "pending.h5")
    source = SourceSpec(path, SourceKind.NEXUS_STACK)
    value = graph.qualify_source_execution_graph(
        source, reader_binding="average_closed_v1",
    )
    window = graph.open_source_execution_graph(value); window.__enter__()
    cursor = window._cursor; owner = cursor._stack; master = cursor._h5
    real_close = type(owner).close
    attempts = []
    def fail_close(stack):
        attempts.append(stack)
        raise OSError("busy")
    monkeypatch.setattr(type(owner), "close", fail_close)
    for expected in range(1, 4):
        with pytest.raises(OSError, match="AVERAGE_SOURCE_CLEANUP_FAILED|busy"):
            window.close()
        assert len(attempts) == expected
        assert cursor._stack is owner and cursor._h5 is master
        assert not cursor.closed and not window._closed
    monkeypatch.setattr(type(owner), "close", real_close)
    window.close()
    assert cursor.closed and window._closed
    assert graph.requalify_source_execution_graph(
        source, value, reader_binding="average_closed_v1",
    ) is value

    timestamps = []
    window = graph.open_source_execution_graph(value)
    with monkeypatch.context() as patch:
        real_close = nexus_io.NexusImageStack.close
        remaining = [2]
        def twice(stack):
            timestamps.append(graph.time.monotonic())
            if remaining[0]:
                remaining[0] -= 1
                raise KeyboardInterrupt("retryable BaseException")
            return real_close(stack)
        patch.setattr(nexus_io.NexusImageStack, "close", twice)
        with window:
            pass
    assert len(timestamps) == 3
    assert all(later - earlier >= 0.045 for earlier, later in zip(timestamps, timestamps[1:]))

    from xrd_tools.core.containers import IntegrationResult1D
    from xrd_tools.reduction import AverageScanRecipe, Integration1DPlan, ReductionPlan
    from xrd_tools.reduction import average as average_module
    from xrd_tools.reduction import core as reduction_core

    def inputs(case):
        root = tmp_path / f"runner-{case}"; root.mkdir()
        dependency = None
        if case == "dependency":
            dependency = root / "pixels.h5"
            with h5py.File(dependency, "w") as handle:
                handle.create_dataset(
                    "pixels", data=np.arange(18, dtype="u2").reshape(3, 2, 3),
                )
            master = _container(root / "scan.nxs")
            with h5py.File(master, "r+", libver="latest") as handle:
                del handle["entry/data/image"]
                layout = h5py.VirtualLayout(shape=(3, 2, 3), dtype="u2")
                layout[:] = h5py.VirtualSource(
                    str(dependency), "/pixels", shape=(3, 2, 3),
                )
                handle["entry/data"].create_virtual_dataset("image", layout)
            source = SourceSpec(master, SourceKind.NEXUS_STACK, entry="entry")
        else:
            source = _tiffs(root)
            if case == "sidecar":
                for member in source.options["files"]:
                    Path(member).with_suffix(".txt").write_text(
                        "# Counters\nI0 = 1.0\n# Motors\n\n"
                        "User: p36, time: Mon Jan 15 10:30:00 2024  # Temp\n"
                    )
                source = SourceSpec(source.uri, source.kind, options={
                    **dict(source.options), "metadata_format": "txt",
                })
        target = root / "average.nxs"
        with h5py.File(target, "w") as handle:
            handle.create_dataset("prior", data=np.arange(4, dtype="<i4"))
        changed = (
            Path(source.options["files"][1]) if case == "member" else
            Path(source.options["files"][1]).with_suffix(".txt")
            if case == "sidecar" else dependency if case == "dependency"
            else target if case == "target" else None
        )
        return source, target, changed

    from xrd_tools.sources import execution_graph as execution_module
    for case in ("unchanged", "member", "sidecar", "dependency", "target", "commit-target"):
        source, target, changed = inputs(case)
        original_target = target.read_bytes()
        plan = average_module.prepare_average_scan(AverageScanRecipe(
            source, target,
            ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
            numeric_metadata_keys=("I0",) if case == "sidecar" else None,
        ))
        counts = {key: 0 for key in (
            "open", "pixel", "metadata", "background", "integrate", "write",
        )}
        events, sinks, close_attempts = [], [], []
        fail_count = 2 if case == "unchanged" else 1
        real_window_close = execution_module._AverageSourceReadWindow.close
        real_open = average_module.open_source_execution_graph
        real_pixel = execution_module._AverageSourceReadWindow.read_native
        real_metadata = execution_module._AverageSourceReadWindow.complete_metadata_for
        real_background = average_module.resolve_frame_background
        real_requalify = average_module.requalify_source_execution_graph
        real_target = average_module.capture_target_snapshot
        real_sink = average_module.NexusSink; real_write = real_sink.write

        def held_close(window):
            close_attempts.append((window, window._cursor, time.monotonic()))
            if len(close_attempts) <= fail_count:
                raise OSError("injected execution-window close hold")
            return real_window_close(window)
        def opened(*args, **kwargs):
            counts["open"] += 1
            return real_open(*args, **kwargs)
        def pixel(window, index):
            counts["pixel"] += 1
            return real_pixel(window, index)
        def metadata(window, index):
            counts["metadata"] += 1
            return real_metadata(window, index)
        def background(*args, **kwargs):
            counts["background"] += 1
            return real_background(*args, **kwargs)
        def integrate(_image, _ai, *, npt, **_kwargs):
            counts["integrate"] += 1
            return IntegrationResult1D(
                np.arange(npt, dtype=float), np.ones(npt), None, "q_A^-1",
            )
        def requalify(*args, **kwargs):
            events.append("source-sweep")
            if case == "commit-target" and events.count("source-sweep") == 2:
                target.write_bytes(target.read_bytes() + b"x")
            return real_requalify(*args, **kwargs)
        def target_snapshot(*args, **kwargs):
            events.append("target-sweep")
            return real_target(*args, **kwargs)
        def sink(*args, **kwargs):
            events.append("sink")
            value = real_sink(*args, **kwargs); sinks.append(value)
            return value
        def write(owner, *args, **kwargs):
            counts["write"] += 1
            return real_write(owner, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(execution_module._AverageSourceReadWindow, "close", held_close)
            patch.setattr(average_module, "open_source_execution_graph", opened)
            patch.setattr(execution_module._AverageSourceReadWindow, "read_native", pixel)
            patch.setattr(execution_module._AverageSourceReadWindow, "complete_metadata_for", metadata)
            patch.setattr(average_module, "resolve_frame_background", background)
            patch.setattr(reduction_core, "integrate_1d", integrate)
            patch.setattr(average_module, "requalify_source_execution_graph", requalify)
            patch.setattr(average_module, "capture_target_snapshot", target_snapshot)
            patch.setattr(average_module, "NexusSink", sink)
            patch.setattr(real_sink, "write", write)
            runner = average_module.AverageScanRunner(plan)
            assert runner.__enter__() is runner
            pending = runner.run()
            retained = runner._source_window
            retained_cursor = retained._cursor
            identifiers = (() if retained_cursor is None else
                tuple(dataset.id for dataset in retained_cursor._stack._dsets))
            assert pending.disposition == "SETTLEMENT_PENDING"
            assert pending.h23_phase is None
            assert pending.diagnostic_code == "AVERAGE_SOURCE_CLEANUP_PENDING"
            assert pending.committed_labels == ()
            assert pending.finite_counts is None and pending.commit_identity is None
            assert pending.operation_identity == plan.operation_identity
            assert runner._source_window is retained
            assert [(row[0], row[1]) for row in close_attempts] == [(retained, retained_cursor)]
            assert counts["open"] == 1 and sinks == []
            assert all(identifier.valid for identifier in identifiers)
            frozen_science = tuple(counts[key] for key in (
                "open", "pixel", "metadata", "background", "integrate",
            ))
            events.clear()
            if changed is not None:
                changed.write_bytes(changed.read_bytes() + b"x")
            preserved_target = target.read_bytes()
            for ordinal in range(1, fail_count):
                again = runner.finish_current()
                assert again.disposition == "SETTLEMENT_PENDING"
                assert again.h23_phase is None
                assert again.diagnostic_code == "AVERAGE_SOURCE_CLEANUP_PENDING"
                assert again.committed_labels == ()
                assert again.finite_counts is None and again.commit_identity is None
                assert again.operation_identity == pending.operation_identity
                assert runner._source_window is retained
                assert close_attempts[-1][:2] == (retained, retained_cursor)
                assert len(close_attempts) == ordinal + 1 and sinks == []
                assert tuple(counts[key] for key in (
                    "open", "pixel", "metadata", "background", "integrate",
                )) == frozen_science
                assert all(identifier.valid for identifier in identifiers)
            terminal = runner.close() if case == "unchanged" else runner.finish_current()
            assert tuple(counts[key] for key in (
                "open", "pixel", "metadata", "background", "integrate",
            )) == frozen_science
            assert len(close_attempts) == fail_count + 1
            assert all(attempt[:2] == (retained, retained_cursor) for attempt in close_attempts)
            if case == "unchanged":
                assert all(b[2] - a[2] >= 0.045 for a, b in zip(close_attempts, close_attempts[1:]))
            assert terminal.operation_identity == pending.operation_identity
            if case == "unchanged":
                assert events == [
                    "source-sweep", "target-sweep", "sink", "target-sweep",
                    "source-sweep", "target-sweep",
                ]
                assert terminal.disposition == "COMMITTED"
                assert len(sinks) == counts["write"] == 1
            elif case == "target":
                assert events == ["source-sweep", "target-sweep"]
                assert terminal.disposition != "COMMITTED"
                assert "TARGET" in terminal.diagnostic_code
                assert sinks == [] and counts["write"] == 0
                assert target.read_bytes() == preserved_target != original_target
            elif case == "commit-target":
                assert events == ["source-sweep", "target-sweep", "sink", "target-sweep", "source-sweep", "target-sweep"]
                assert terminal.disposition == "ABORTED" and "TARGET" in terminal.diagnostic_code
                assert target.read_bytes() == original_target
            else:
                assert events == ["source-sweep"]
                assert (terminal.disposition, terminal.diagnostic_code) == (
                    "REFUSED", "AVERAGE_SOURCE_DRIFT",
                )
                assert sinks == [] and counts["write"] == 0
                assert target.read_bytes() == original_target
            assert all(not identifier.valid for identifier in identifiers)
            terminal_events, terminal_counts = tuple(events), dict(counts)
            assert runner.finish_current() is terminal
            assert runner.close() is terminal
            assert tuple(events) == terminal_events and counts == terminal_counts


def test_average_value_descriptor_and_dependency_failures_close_binding(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.sources import descriptor as descriptor_module
    path = _container(tmp_path / "descriptor.h5")
    closes = []
    real_close = nexus_io._ResolvedNexusStack.close
    def observed_close(owner):
        before = tuple(dataset.id for dataset in owner._datasets)
        assert before and all(identifier.valid for identifier in before)
        result = real_close(owner)
        closes.append((owner, before))
        assert all(not identifier.valid for identifier in before)
        return result
    monkeypatch.setattr(nexus_io._ResolvedNexusStack, "close", observed_close)
    real_descriptor = descriptor_module.ContainerDescriptor
    monkeypatch.setattr(
        descriptor_module, "ContainerDescriptor",
        lambda *_a, **_k: (_ for _ in ()).throw(
            KeyboardInterrupt("descriptor value")),
    )
    with pytest.raises(KeyboardInterrupt, match="descriptor value"):
        graph.qualify_source_execution_graph(
            SourceSpec(path, SourceKind.NEXUS_STACK),
            reader_binding="average_closed_v1",
        )
    assert len(closes) == 1 and closes[0][0].state == "CLOSED"
    monkeypatch.setattr(descriptor_module, "ContainerDescriptor", real_descriptor)

    value = graph.qualify_source_execution_graph(
        SourceSpec(path, SourceKind.NEXUS_STACK),
        reader_binding="average_closed_v1",
    )
    closes.clear()
    with monkeypatch.context() as patch:
        patch.setattr(
            graph, "_capture_bound_container_dependencies",
            lambda *_a, **_k: (_ for _ in ()).throw(
                KeyboardInterrupt("dependency facts")),
        )
        with pytest.raises(KeyboardInterrupt, match="dependency facts"):
            graph.requalify_source_execution_graph(
                SourceSpec(path, SourceKind.NEXUS_STACK), value,
                reader_binding="average_closed_v1",
            )
    assert len(closes) == 1 and closes[0][0].state == "CLOSED"

    closes.clear()
    with h5py.File(path, "r") as handle:
        entry = handle["entry"]
        pair = bluesky_nexus.validate_average_container_metadata_inputs(
            entry, policy="average_bounded_v1",
        )
        described = descriptor_module.describe_container_from_open(
            handle, path=path, entry="entry", resolved_entry_group=entry,
            prevalidated_motor_names=pair,
        )
        assert described.frame_count == 3 and handle.id.valid
        assert len(closes) == 1 and closes[0][0].state == "CLOSED"
        assert all(not identifier.valid for identifier in closes[0][1])
    monkeypatch.setattr(nexus_io._ResolvedNexusStack, "close", real_close)

    real_bind = descriptor_module._describe_container_from_open_with_binding
    bindings = []; close_calls = []; failures = [2]
    def observed_bind(*args, **kwargs):
        result = real_bind(*args, **kwargs); bindings.append(result[1]); return result
    def twice_then_close(owner):
        close_calls.append((graph.time.monotonic(), owner))
        if failures[0]:
            failures[0] -= 1
            raise OSError("injected lexical binding close hold")
        return real_close(owner)
    with monkeypatch.context() as patch:
        patch.setattr(descriptor_module, "_describe_container_from_open_with_binding", observed_bind)
        patch.setattr(nexus_io._ResolvedNexusStack, "close", twice_then_close)
        refreshed = graph.qualify_source_execution_graph(
            SourceSpec(path, SourceKind.NEXUS_STACK),
            reader_binding="average_closed_v1",
        )
    assert refreshed.stamp.frame_count == 3 and len(bindings) == 2
    assert [owner for _, owner in close_calls[:3]] == [bindings[0]] * 3
    assert close_calls[3][1] is bindings[1] and bindings[0] is not bindings[1]
    assert all(later[0] - earlier[0] >= 0.045
               for earlier, later in zip(close_calls[:2], close_calls[1:3]))
    assert all(owner.state == "CLOSED" for owner in bindings)

    dependency = tmp_path / "dependency.h5"
    with h5py.File(dependency, "w") as handle:
        handle.create_dataset("pixels", data=np.ones((3, 2, 3), dtype="u2"))
    dependency_master = tmp_path / "dependency-master.h5"
    with h5py.File(dependency_master, "w", libver="latest") as handle:
        entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("end_time", data=np.bytes_("done"))
        data = entry.create_group("data"); data.attrs["NX_class"] = "NXdata"
        layout = h5py.VirtualLayout(shape=(3, 2, 3), dtype="u2")
        layout[:] = h5py.VirtualSource(str(dependency), "/pixels", shape=(3, 2, 3))
        image = data.create_virtual_dataset("image", layout)
        image.attrs["signal_type"] = "detector"
    dependency_source = SourceSpec(dependency_master, SourceKind.NEXUS_STACK)
    admitted = graph.qualify_source_execution_graph(
        dependency_source, reader_binding="average_closed_v1",
    )
    bindings.clear(); close_calls.clear(); held = []
    def mutate_then_close(owner):
        close_calls.append(owner)
        if not held:
            held.append(owner); dependency.write_bytes(dependency.read_bytes() + b"x")
            raise OSError("injected dependency qualification close hold")
        return real_close(owner)
    with monkeypatch.context() as patch:
        patch.setattr(descriptor_module, "_describe_container_from_open_with_binding", observed_bind)
        patch.setattr(nexus_io._ResolvedNexusStack, "close", mutate_then_close)
        with pytest.raises(graph.SourceRevisionChanged, match="container dependency changed"):
            graph.requalify_source_execution_graph(
                dependency_source, admitted, reader_binding="average_closed_v1",
            )
    assert len(bindings) == 2 and close_calls[:2] == [bindings[0], bindings[0]]
    assert close_calls[2:] == [bindings[1]] and all(owner.state == "CLOSED" for owner in bindings)

    bindings.clear(); close_calls.clear(); helper_calls = []; failures[:] = [1]
    def interrupted_dependencies(*_args, **_kwargs):
        helper_calls.append(1); raise KeyboardInterrupt("dependency facts after clean")
    with monkeypatch.context() as patch:
        patch.setattr(descriptor_module, "_describe_container_from_open_with_binding", observed_bind)
        patch.setattr(graph, "_capture_bound_container_dependencies", interrupted_dependencies)
        patch.setattr(nexus_io._ResolvedNexusStack, "close", twice_then_close)
        with pytest.raises(KeyboardInterrupt, match="dependency facts after clean"):
            graph.qualify_source_execution_graph(
                SourceSpec(path, SourceKind.NEXUS_STACK),
                reader_binding="average_closed_v1",
            )
    assert helper_calls == [1, 1] and len(bindings) == 2
    assert [owner for _, owner in close_calls] == [bindings[0], bindings[0], bindings[1]]
    assert all(owner.state == "CLOSED" for owner in bindings)


def test_average_bound_dependencies_use_same_group_without_master_reopen(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.sources import descriptor as descriptor_module
    root = tmp_path / "same"; root.mkdir()
    path, members = _eiger_master(root)
    opens = []
    real_file = h5py.File
    monkeypatch.setattr(
        graph, "_HDF5_FILE_OPEN",
        lambda *args, **kwargs: (
            opens.append((Path(args[0]), kwargs.get("rdcc_nbytes"))),
            real_file(*args, **kwargs),
        )[1],
    )
    monkeypatch.setattr(cursor_module, "_HDF5_FILE_OPEN", graph._HDF5_FILE_OPEN)
    monkeypatch.setattr(
        nexus_io.NexusImageStack, "__init__",
        lambda *_a, **_k: pytest.fail("path-reacquiring stack constructor reached"),
    )
    for name in (
        "resolve_stack_paths", "_find_eiger_external_link_paths",
        "find_nexus_image_dataset_in_open_file",
    ):
        if hasattr(descriptor_module, name):
            monkeypatch.setattr(
                descriptor_module, name,
                lambda *_a, _name=name, **_k: pytest.fail(
                    f"ordinary name-taking helper reached: {_name}"),
            )
    value = graph.qualify_source_execution_graph(
        SourceSpec(path, SourceKind.EIGER_MASTER, entry="entry"),
        reader_binding="average_closed_v1",
    )
    assert opens == [(path, 1 << 20)]; opens.clear()
    assert tuple(Path(item.file.path) for item in value.stamp.external_members) == members
    assert tuple((item.dataset, item.first, item.stop) for item in
                 value.stamp.external_members) == (
        ("/entry/data/data", 0, 2), ("/entry/data/data", 2, 5),
    )
    with graph.open_source_execution_graph(value) as window:
        cursor = window._cursor
        assert cursor._stack is not None
        assert cursor._entry_grp.file.id == cursor._h5.id
        assert cursor._opening_stack_owner.owner is None
        assert window.complete_metadata_for(0) == {}
        assert np.all(window.read_native(0) == 1)
    assert opens == [(path, 1 << 20), (path, 1 << 20)]


def test_average_nested_same_master_vds_uses_one_sequential_owned_dataset_id(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "vds.h5"
    with h5py.File(path, "w", libver="latest") as handle:
        entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("end_time", data=np.bytes_("done"))
        raw1 = entry.create_dataset("raw1", data=np.ones((1, 2, 3), dtype="u2"))
        raw2 = entry.create_dataset("raw2", data=np.full((1, 2, 3), 2, dtype="u2"))
        nested_layout = h5py.VirtualLayout(shape=(2, 2, 3), dtype="u2")
        nested_layout[0:1] = h5py.VirtualSource(str(path), raw1.name, shape=(1, 2, 3))
        nested_layout[1:2] = h5py.VirtualSource(str(path), raw2.name, shape=(1, 2, 3))
        nested = entry.create_virtual_dataset("nested", nested_layout)
        data = entry.create_group("data"); data.attrs["NX_class"] = "NXdata"
        layout = h5py.VirtualLayout(shape=(4, 2, 3), dtype="u2")
        layout[0:2] = h5py.VirtualSource(str(path), nested.name, shape=(2, 2, 3))
        layout[2:4] = h5py.VirtualSource(str(path), nested.name, shape=(2, 2, 3))
        data.create_virtual_dataset("image", layout)
    source = SourceSpec(path, SourceKind.NEXUS_STACK, entry="entry")
    value = graph.qualify_source_execution_graph(
        source, reader_binding="average_closed_v1",
    )
    with h5py.File(path, "r") as handle:
        slot, binding = _bind_stack(handle)
        opened = []; closed = []; active = []; selectors = []
        selected_ids = {int(dataset.id.id) for dataset in binding._datasets}
        kind = type(binding)
        real_open = kind.open_dependency_dataset
        real_close = kind.close_dependency_dataset
        def tracked_open(owner, selector):
            assert active == []
            assert owner._dependency_dataset is owner._dependency_file is None
            dataset = real_open(owner, selector)
            identifier = int(dataset.id.id)
            assert identifier not in selected_ids
            assert owner._dependency_dataset is dataset
            selectors.append(selector[1])
            active.append(identifier); opened.append(identifier)
            return dataset
        def tracked_close(owner):
            assert len(active) == 1
            identifier = active.pop()
            assert int(owner._dependency_dataset.id.id) == identifier
            result = real_close(owner)
            assert owner._dependency_dataset is owner._dependency_file is None
            closed.append(identifier)
            return result
        monkeypatch.setattr(kind, "open_dependency_dataset", tracked_open)
        monkeypatch.setattr(kind, "close_dependency_dataset", tracked_close)
        externals = []; dependencies = []
        graph._capture_bound_container_dependencies(
            binding, value.descriptor, graph.SourceFileState.capture(path),
            cancelled=lambda: False,
            emit_external=externals.append, emit_dependency=dependencies.append,
        )
        assert active == [] and opened == closed and len(opened) == 3
        assert selectors == ["/entry/nested", "/entry/raw1", "/entry/raw2"]
        assert externals == [] and dependencies == []
        monkeypatch.setattr(kind, "open_dependency_dataset", real_open)
        monkeypatch.setattr(kind, "close_dependency_dataset", real_close)
        binding.close()
        assert binding.state == "CLOSED" and slot.owner is binding

    with h5py.File(path, "r") as handle:
        slot, binding = _bind_stack(handle)
        selected_ids = {int(dataset.id.id) for dataset in binding._datasets}
        transient_opens, physical, failed = [], [], []
        real_id_close = kind._close_dataset_id
        def observed_open(owner, selector):
            transient_opens.append(selector[1])
            return real_open(owner, selector)
        def fail_first_transient_close(dataset):
            identifier = dataset.id
            if identifier.valid and int(identifier.id) not in selected_ids:
                physical.append(id(identifier))
                if not failed:
                    failed.append(identifier)
                    raise OSError("injected transient Dataset-ID close failure")
            return real_id_close(dataset)
        with monkeypatch.context() as patch:
            patch.setattr(kind, "open_dependency_dataset", observed_open)
            patch.setattr(
                kind, "_close_dataset_id",
                staticmethod(fail_first_transient_close),
            )
            with pytest.raises(OSError, match="transient Dataset-ID close failure"):
                graph._capture_bound_container_dependencies(
                    binding, value.descriptor, graph.SourceFileState.capture(path),
                    cancelled=lambda: False,
                    emit_external=lambda _value: None,
                    emit_dependency=lambda _value: None,
                )
            assert transient_opens == ["/entry/nested"]
            assert len(failed) == 1 and failed[0].valid
            assert physical == [id(failed[0])]
            assert binding._dependency_dataset.id is failed[0]
            assert slot.owner is binding and binding.state == "BINDING_OWNS_DATASET_IDS"
            binding.close()
            assert physical == [id(failed[0]), id(failed[0])]
            assert not failed[0].valid and binding.state == "CLOSED"
            assert transient_opens == ["/entry/nested"]

    external1 = tmp_path / "external1.h5"
    external2 = tmp_path / "external2.h5"
    for external, value in ((external1, 3), (external2, 4)):
        with h5py.File(external, "w") as handle:
            handle.create_dataset("pixels", data=np.full((1, 2, 3), value, dtype="u2"))
            handle.create_group("group")
    external_master = tmp_path / "external-master.h5"
    with h5py.File(external_master, "w", libver="latest") as handle:
        entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("end_time", data=np.bytes_("done"))
        data = entry.create_group("data"); data.attrs["NX_class"] = "NXdata"
        layout = h5py.VirtualLayout(shape=(2, 2, 3), dtype="u2")
        layout[0:1] = h5py.VirtualSource(str(external1), "/pixels", shape=(1, 2, 3))
        layout[1:2] = h5py.VirtualSource(str(external2), "/pixels", shape=(1, 2, 3))
        data.create_virtual_dataset("image", layout)
    external_value = graph.qualify_source_execution_graph(
        SourceSpec(external_master, SourceKind.NEXUS_STACK), reader_binding="average_closed_v1",
    )
    with h5py.File(external_master, "r") as handle:
        slot, binding = _bind_stack(handle); opened = []; file_attempts = []
        real_file_close = kind._close_dependency_file
        def external_open(owner, selector):
            opened.append(selector[0]); return real_open(owner, selector)
        def fail_first_file_close(owner):
            file_attempts.append(owner)
            if len(file_attempts) == 1: raise OSError("injected dependency File close failure")
            return real_file_close(owner)
        with monkeypatch.context() as patch:
            patch.setattr(kind, "open_dependency_dataset", external_open)
            patch.setattr(kind, "_close_dependency_file", staticmethod(fail_first_file_close))
            with pytest.raises(OSError, match="dependency File close failure"):
                graph._capture_bound_container_dependencies(
                    binding, external_value.descriptor, graph.SourceFileState.capture(external_master),
                    cancelled=lambda: False,
                    emit_external=lambda _value: None,
                    emit_dependency=lambda _value: None,
                )
            retained_file = file_attempts[0]
            assert opened == [str(external1)]
            assert binding._dependency_dataset is None
            assert binding._dependency_file is retained_file and retained_file.id.valid
            binding.close()
            assert file_attempts == [retained_file, retained_file]
            assert binding._dependency_dataset is binding._dependency_file is None
            assert not retained_file.id.valid and binding.state == "CLOSED"

    with h5py.File(external_master, "r") as handle:
        slot, binding = _bind_stack(handle)
        with pytest.raises(TypeError, match="not a Dataset") as caught:
            binding.open_dependency_dataset((str(external1), "/group"))
        traceback = caught.value.__traceback__
        while traceback.tb_next is not None: traceback = traceback.tb_next
        owner = traceback.tb_frame.f_locals["owner"]
        assert not owner.id.valid
        assert binding._dependency_dataset is binding._dependency_file is None
        binding.close(); assert binding.state == "CLOSED"


def test_average_container_scalar_inputs_refuse_unbounded_dtype_before_payload(
    tmp_path, monkeypatch,
) -> None:
    routes = (
        "instrument/positioners/theta/value",
        "instrument/bluesky/streams/baseline/direct",
        "data/theta",
        "instrument/bluesky/metadata/configuration/eiger/data/eiger_cam_wavelength",
        "instrument/monochromator/energy",
    )
    invalid = (
        (np.array([b"x"], dtype="S1"), None),
        (np.array(["x"], dtype=h5py.string_dtype(encoding="utf-8")), None),
        (np.array([(1, 2)], dtype=[("a", "<i4"), ("b", "<i4")]), None),
        (np.array([1 + 2j], dtype="<c16"), None),
        (np.array([1.0, 2.0], dtype="<f8"), "scalar_only"),
    )
    real_getitem = h5py.Dataset.__getitem__
    for route_index, route in enumerate(routes):
        for value_index, (data_value, restricted) in enumerate(invalid):
            if restricted and route in (
                "instrument/positioners/theta/value",
                "instrument/bluesky/streams/baseline/direct", "data/theta",
            ):
                continue
            path = _container(tmp_path / f"scalar-{route_index}-{value_index}.h5")
            with h5py.File(path, "r+") as handle:
                parent, name = route.rsplit("/", 1)
                group = handle["entry"].require_group(parent)
                if name in group:
                    del group[name]
                group.create_dataset(name, data=data_value)
            reads = []
            with monkeypatch.context() as patch:
                def guarded(dataset, key):
                    if dataset.name.endswith(f"/{name}"):
                        reads.append(key)
                        raise AssertionError("unbounded scalar payload read")
                    return real_getitem(dataset, key)
                patch.setattr(h5py.Dataset, "__getitem__", guarded)
                patch.setattr("xrd_tools.sources.descriptor._describe_container_from_open_with_binding",
                    lambda *_a, **_k: pytest.fail("descriptor reached invalid scalar"))
                patch.setattr(metadata_provider, "BlueskyMetadataProvider",
                    lambda *_a, **_k: pytest.fail("provider reached invalid scalar"))
                with pytest.raises(ValueError, match="AVERAGE_METADATA_INPUT_UNBOUNDED"):
                    graph.qualify_source_execution_graph(
                        SourceSpec(path, SourceKind.NEXUS_STACK),
                        reader_binding="average_closed_v1",
                    )
            assert reads == []

    wide = _container(tmp_path / "scalar-wide-projection.h5")
    with h5py.File(wide, "r+") as handle:
        config = handle["entry"].require_group(
            "instrument/bluesky/metadata/configuration/eiger/data"
        )
        config.create_dataset("eiger_cam_wavelength", data=np.float64(1.0))
    class _WideFloatDtype:
        kind = "f"
        itemsize = 16
        fields = None
        subdtype = None
        metadata = None

    wide_dtype = _WideFloatDtype()
    reads = []; dtype_descriptor = h5py.Dataset.dtype
    with monkeypatch.context() as patch:
        patch.setattr(h5py.Dataset, "dtype", property(lambda dataset:
            wide_dtype if dataset.name.endswith("/eiger_cam_wavelength")
            else dtype_descriptor.__get__(dataset, type(dataset))))
        def unread(dataset, key):
            if dataset.name.endswith("/eiger_cam_wavelength"):
                reads.append(key); raise AssertionError("wide scalar payload read")
            return real_getitem(dataset, key)
        patch.setattr(h5py.Dataset, "__getitem__", unread)
        with pytest.raises(ValueError, match="AVERAGE_METADATA_INPUT_UNBOUNDED"):
            graph.qualify_source_execution_graph(
                SourceSpec(wide, SourceKind.NEXUS_STACK),
                reader_binding="average_closed_v1",
            )
    assert wide_dtype.kind == "f" and wide_dtype.itemsize > 8
    assert reads == []

    valid = _container(tmp_path / "scalar-valid.h5")
    with h5py.File(valid, "r+") as handle:
        entry = handle["entry"]
        config = entry.require_group(
            "instrument/bluesky/metadata/configuration/eiger/data"
        )
        config.create_dataset("eiger_cam_wavelength", data=np.float16(1.0))
        mono = entry.require_group("instrument/monochromator")
        mono.create_dataset("energy", data=np.array([12.0], dtype="<f8"))
    with h5py.File(valid, "r") as handle:
        scanned, all_names = bluesky_nexus.validate_average_container_metadata_inputs(
            handle["entry"], policy="average_bounded_v1",
        )
    assert scanned == ("theta",) and all_names == ("theta",)

    for ordinal, route in enumerate((
        "instrument/bluesky/streams/baseline/direct", "data/theta",
    )):
        path = _container(tmp_path / f"million-scalar-{ordinal}.h5")
        with h5py.File(path, "r+") as handle:
            entry = handle["entry"]; del entry["data/image"]
            entry["data"].create_dataset("image", shape=(1_000_000, 2, 3),
                dtype="u2", chunks=(1, 2, 3), fillvalue=0).attrs["signal_type"] = "detector"
            del entry["data/I0"]
            parent, name = route.rsplit("/", 1); group = entry.require_group(parent)
            if name in group: del group[name]
            group.create_dataset(name, shape=(1_000_000,), dtype="<f8",
                                 chunks=(131_072,), fillvalue=7.0)
            if route.startswith("instrument/"): group.create_dataset("direct_user_setpoint", data=np.array([7.0], dtype="<f8"))
        reads = []
        with monkeypatch.context() as patch:
            def indexed(dataset, key):
                if dataset.name.endswith(f"/{name}"): reads.append(key)
                return real_getitem(dataset, key)
            patch.setattr(h5py.Dataset, "__getitem__", indexed)
            prepared = graph.qualify_source_execution_graph(
                SourceSpec(path, SourceKind.NEXUS_STACK),
                reader_binding="average_closed_v1",
            )
            with graph.open_source_execution_graph(prepared) as window:
                key = "theta" if route.startswith("data/") else "direct"
                assert window.complete_metadata_for(999_999)[key] == 7.0
        assert reads == [999_999 if route.startswith("data/") else 0]


def test_first_numeric_preserves_ordinary_mixed_convertibility(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "ordinary-numeric.h5"
    real_array, real_item = h5py.Dataset.__array__, h5py.Dataset.__getitem__; arrays = []
    def array(dataset, *args, **kwargs):
        arrays.append(dataset.name); return real_array(dataset, *args, **kwargs)
    def item(dataset, key):
        if dataset.name in {"/good", "/mixed"}: pytest.fail("ordinary branch indexed Dataset")
        return real_item(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__array__", array)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", item)
    with h5py.File(path, "w") as handle:
        good = handle.create_dataset("good", data=np.array([3.5, 4.5], dtype="<f8"))
        mixed = handle.create_dataset("mixed", data=np.array([b"3.5", b"bad"], dtype="S3"))
        assert bluesky_nexus._first_numeric(good) == 3.5
        assert bluesky_nexus._first_numeric(mixed) is None
    assert arrays == ["/good", "/mixed"]

    average = _container(tmp_path / "average-invalid.h5")
    with h5py.File(average, "r+") as handle:
        del handle["entry/data/I0"]
        handle["entry/data"].create_dataset("I0", data=np.array([b"3.5"], dtype="S3"))
    real_asarray = np.asarray
    calls = []
    def guarded(value, *args, **kwargs):
        if isinstance(value, h5py.Dataset) and value.name.endswith("/data/I0"):
            calls.append(value.name)
            raise AssertionError("Average reached ordinary whole conversion")
        return real_asarray(value, *args, **kwargs)
    monkeypatch.setattr(bluesky_nexus.np, "asarray", guarded)
    with h5py.File(average, "r") as handle:
        with pytest.raises(ValueError, match="AVERAGE_METADATA_INPUT_UNBOUNDED"):
            bluesky_nexus.validate_average_container_metadata_inputs(
                handle["entry"], policy="average_bounded_v1",
            )
    assert calls == []


def test_tiff_metadata_cap_precedes_auto_txt_pdi_generic_and_spec_parsers(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.io import metadata as metadata_io
    from silx.io import specfile as silx_spec

    image = tmp_path / "experiment_scan1_0.tif"
    tifffile.imwrite(image, np.ones((2, 2), dtype=np.uint8))
    exact_bytes = b"I0 = 1\n" + b" " * ((1 << 16) - len(b"I0 = 1\n"))
    parser_calls = []

    ordinary = image.with_suffix(".txt")
    ordinary.write_bytes(
        b"# Counters\nI0 = 1\n# Motors\n\nUser: p36, time: Mon Jan 15 10:30:00 2024  # Temp\n"
        + b" " * (1 << 16)
    )
    unrestricted = metadata_io.read_image_metadata_observed(image, meta_format="txt")
    assert unrestricted.source_path == ordinary and unrestricted.values["I0"] == 1.0
    ordinary.unlink()

    class Scan:
        data = np.array([[1.0]])
        labels = ("I0",)
        motor_names = ("theta",)
        def data_column_by_name(self, _name): return np.array([1.0])
        def motor_position_by_name(self, _name): return 2.0
    class Spec:
        def __init__(self, path): parser_calls.append(("spec", Path(path)))
        def __getitem__(self, _key): return Scan()
        def close(self): pass

    monkeypatch.setattr(silx_spec, "SpecFile", Spec)
    parsers = {
        "txt": "read_txt_metadata",
        "pdi": "read_pdi_metadata",
        "meta": "_parse_structured_sidecar_if_plausible",
    }
    for policy, parser_name in parsers.items():
        sidecar = image.with_suffix(f".{policy}")
        sidecar.write_bytes(exact_bytes)
        real_parser = getattr(metadata_io, parser_name)
        with monkeypatch.context() as patch:
            patch.setattr(
                metadata_io, parser_name,
                lambda path, *_a, _policy=policy, **_k:
                    parser_calls.append((_policy, Path(path))) or {"I0": 1.0},
            )
            parser_calls.clear()
            observed = metadata_io.read_image_metadata_observed(
                image, meta_format=policy, max_input_bytes=1 << 16,
            )
            assert observed.values["I0"] == 1.0
            assert observed.source_path == sidecar
            assert parser_calls == [(policy, sidecar)]
            sidecar.write_bytes(exact_bytes + b"x")
            parser_calls.clear()
            with pytest.raises(metadata_io.MetadataInputTooLarge):
                metadata_io.read_image_metadata_observed(
                    image, meta_format=policy, max_input_bytes=1 << 16,
                )
            assert parser_calls == []
        monkeypatch.setattr(metadata_io, parser_name, real_parser)
        sidecar.unlink()

    spec_source = tmp_path / "experiment"
    spec_source.write_bytes(exact_bytes)
    parser_calls.clear()
    observed = metadata_io.read_image_metadata_observed(
        image, meta_format="spec", max_input_bytes=1 << 16,
    )
    assert observed.values == {"I0": 1.0, "theta": 2.0}
    assert observed.source_path == spec_source
    assert len(parser_calls) == 1 and parser_calls[0][0] == "spec"
    spec_source.write_bytes(exact_bytes + b"x")
    parser_calls.clear()
    with pytest.raises(metadata_io.MetadataInputTooLarge):
        metadata_io.read_image_metadata_observed(
            image, meta_format="spec", max_input_bytes=1 << 16,
        )
    assert parser_calls == []
    spec_source.unlink()

    auto = image.with_suffix(".txt")
    auto.write_bytes(exact_bytes)
    metadata_io._AUTO_SIDECAR_CACHE.clear()
    with monkeypatch.context() as patch:
        patch.setattr(
            metadata_io, "read_txt_metadata",
            lambda path, *_a, **_k:
                parser_calls.append(("auto", Path(path))) or {"I0": 1.0},
        )
        patch.setattr(
            metadata_io, "read_pdi_metadata",
            lambda path, *_a, **_k:
                parser_calls.append(("pdi", Path(path))) or {"I0": 2.0},
        )
        for expected_cache_size in (1, 1):
            parser_calls.clear()
            observed = metadata_io.read_image_metadata_observed(
                image, meta_format="auto", max_input_bytes=1 << 16,
            )
            assert observed.source_path == auto and observed.values == {"I0": 1.0}
            assert parser_calls == [("auto", auto)]
            assert len(metadata_io._AUTO_SIDECAR_CACHE) == expected_cache_size
        preferred = image.with_suffix(".pdi")
        preferred.write_bytes(exact_bytes)
        auto.write_bytes(exact_bytes + b"x")
        metadata_io._AUTO_SIDECAR_CACHE.clear()
        parser_calls.clear()
        with pytest.raises(metadata_io.MetadataInputTooLarge):
            metadata_io.read_image_metadata_observed(
                image, meta_format="auto", max_input_bytes=1 << 16,
            )
        assert parser_calls == []

    growing = image.with_suffix(".meta")
    growing.write_bytes(b"G" * ((1 << 16) + 1))
    real_open = Path.open; real_fstat = metadata_io.os.fstat
    read_sizes, opens, fstats = [], [], []
    class GuardedReader:
        def __init__(self, owner): self._owner = owner
        def __enter__(self): return self
        def __exit__(self, *exc): return self._owner.__exit__(*exc)
        def __getattr__(self, name): return getattr(self._owner, name)
        def read(self, size=-1):
            read_sizes.append(size)
            assert 0 <= size <= (1 << 16) + 1
            return self._owner.read(size)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", lambda *_a, **_k: pytest.fail("bounded snapshot called Path.stat"))
        patch.setattr(Path, "open", lambda path, *a, **k:
            opens.append(path) or GuardedReader(real_open(path, *a, **k)))
        patch.setattr(metadata_io.os, "fstat", lambda fd:
            fstats.append(fd) or real_fstat(fd))
        with pytest.raises(metadata_io.MetadataInputTooLarge):
            metadata_io._bounded_metadata_snapshot(growing, 1 << 16)
    assert opens == [growing] and len(fstats) == 2
    assert fstats[0] == fstats[1] and read_sizes == [(1 << 16) + 1]


def test_spec_snapshot_closes_parser_before_unlink_on_every_path(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.io import metadata as metadata_io
    from silx.io import specfile as silx_spec

    image = tmp_path / "experiment_scan1_0.tif"
    tifffile.imwrite(image, np.ones((2, 2), dtype="u1"))
    source = tmp_path / "experiment"; source.write_text("#F experiment\n")
    scratch = []; events = []

    class Scan:
        data = np.array([[3.0]])
        labels = ("I0",)
        motor_names = ("theta",)
        def data_column_by_name(self, _name): return np.array([3.0])
        def motor_position_by_name(self, _name): return 4.0
    class Parser:
        bounded = True
        fail_body = False
        fail_close = False
        def __init__(self, path):
            self.path = Path(path); scratch.append(self.path); events.append("open")
            assert ((self.path != source and self.path.read_bytes() == source.read_bytes())
                    if self.bounded else self.path == source)
        def __getitem__(self, _key):
            events.append("body")
            if self.fail_body:
                if isinstance(self.fail_body, BaseException): raise self.fail_body
                raise ValueError("parser body")
            return Scan()
        def close(self):
            events.append("close")
            assert self.path.exists()
            if self.fail_close:
                raise OSError("parser close")

    monkeypatch.setattr(silx_spec, "SpecFile", Parser)
    Parser.bounded = False
    ordinary = metadata_io.read_image_metadata_observed(image, meta_format="spec")
    assert ordinary.values == {"I0": 3.0, "theta": 4.0}
    assert ordinary.source_path == source and events == ["open", "body"]
    assert scratch.pop() == source
    Parser.bounded = True; events.clear()
    observed = metadata_io.read_image_metadata_observed(
        image, meta_format="spec", max_input_bytes=1 << 16,
    )
    assert observed.values == {"I0": 3.0, "theta": 4.0}
    assert observed.source_path == source
    assert events == ["open", "body", "close"]
    assert len(scratch) == 1 and not scratch.pop().exists()

    events.clear(); Parser.fail_body = KeyboardInterrupt("parser BaseException")
    with pytest.raises(KeyboardInterrupt, match="parser BaseException"):
        metadata_io.read_image_metadata_observed(
            image, meta_format="spec", max_input_bytes=1 << 16,
        )
    assert events == ["open", "body", "close"]
    assert len(scratch) == 1 and not scratch.pop().exists()

    events.clear(); Parser.fail_close = True; primary = Parser.fail_body
    with pytest.raises(KeyboardInterrupt, match="parser BaseException") as caught:
        metadata_io.read_image_metadata_observed(
            image, meta_format="spec", max_input_bytes=1 << 16,
        )
    assert caught.value is primary
    assert isinstance(caught.value.__cause__, metadata_io.MetadataScratchCleanupFailed)
    assert isinstance(caught.value.__cause__.__cause__, OSError)
    assert events == ["open", "body", "close"]
    assert len(scratch) == 1 and not scratch.pop().exists()

    Parser.fail_body = Parser.fail_close = False
    real_temporary = metadata_io.tempfile.NamedTemporaryFile
    scratch_close_events = []; scratch_write_paths = []
    class InterruptedScratch:
        def __init__(self):
            self.owner = real_temporary(prefix="xdart-spec-test-", delete=False)
            self.name = self.owner.name
            scratch_write_paths.append(Path(self.name))
        @property
        def closed(self): return self.owner.closed
        def write(self, value):
            self.owner.write(value); raise KeyboardInterrupt("scratch write")
        def close(self):
            scratch_close_events.append(1); return self.owner.close()
    with monkeypatch.context() as patch:
        patch.setattr(metadata_io.tempfile, "NamedTemporaryFile", lambda **_kwargs: InterruptedScratch())
        with pytest.raises(KeyboardInterrupt, match="scratch write"):
            metadata_io.read_image_metadata_observed(
                image, meta_format="spec", max_input_bytes=1 << 16,
            )
    assert scratch_close_events == [1]
    assert len(scratch_write_paths) == 1 and not scratch_write_paths[0].exists()

    events.clear(); primary = ValueError("parser body"); Parser.fail_body = primary; Parser.fail_close = True
    with pytest.raises(ValueError, match="parser body") as caught:
        metadata_io.read_image_metadata_observed(
            image, meta_format="spec", max_input_bytes=1 << 16,
        )
    assert events == ["open", "body", "close"]
    assert caught.value is primary
    assert isinstance(caught.value.__cause__, metadata_io.MetadataScratchCleanupFailed)
    assert isinstance(caught.value.__cause__.__cause__, OSError)
    assert len(scratch) == 1 and not scratch.pop().exists()

    events.clear(); Parser.fail_body = False; Parser.fail_close = False
    real_unlink = Path.unlink
    with monkeypatch.context() as patch:
        def fail_unlink(path, *args, **kwargs):
            events.append("unlink")
            if path != source: raise OSError("snapshot unlink")
            return real_unlink(path, *args, **kwargs)
        patch.setattr(Path, "unlink", fail_unlink)
        with pytest.raises(metadata_io.MetadataScratchCleanupFailed, match="unlink"):
            metadata_io.read_image_metadata_observed(
                image, meta_format="spec", max_input_bytes=1 << 16,
            )
    assert events == ["open", "body", "close", "unlink"]
    assert len(scratch) == 1 and scratch.pop().exists()

    events.clear(); Parser.fail_body = True
    observed = metadata_io.read_image_metadata_observed(
        image, meta_format="spec", max_input_bytes=1 << 16,
    )
    assert observed.values == {} and observed.source_path == source
    assert events == ["open", "body", "close"]
    assert len(scratch) == 1 and not scratch.pop().exists()

    events.clear(); Parser.fail_body = False; Parser.fail_close = True
    with pytest.raises(metadata_io.MetadataScratchCleanupFailed, match="close"):
        metadata_io.read_image_metadata_observed(
            image, meta_format="spec", max_input_bytes=1 << 16,
        )
    assert events == ["open", "body", "close"]
    assert len(scratch) == 1 and not scratch.pop().exists()
