"""Finite P3-4A oracle for existing-dimension headless replacement."""
from __future__ import annotations
import copy, hashlib, json, shutil, threading, weakref; from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path; from types import SimpleNamespace; import h5py
import numpy as np
import pytest
from tests.core.reintegrate_support import (
    _plans,
    _preparation,
    _r1,
    _r2,
    _seed_existing,
    _stub_integrators,
)
def _module():
    from xrd_tools.reduction import reintegrate
    return reintegrate


def _successor_from_plan(plan, *, destination_directory=None):
    """Build the immutable v4 execution plan from admitted v3 support facts."""
    from xrd_tools.reduction import ReintegrateSuccessorPlan

    support = _module()
    mapping = support._plan_mapping(plan)
    preparation = {
        "api_version": 1,
        "selected_plan": support._plain(plan.selected_plan),
        "requested_shared_science": support._plain(
            plan.requested_shared_science,
        ),
        "resource_policy": {
            "version": 1,
            "kind": "explicit",
            "allocation": mapping["session_policy"]["allocation"],
        },
    }
    return ReintegrateSuccessorPlan.from_artifact(
        plan.target,
        entry=plan.entry,
        dimension=plan.dimension,
        preparation=preparation,
        source_root=plan.source_root,
        expected_target_snapshot=plan.expected_target_snapshot,
        expected_labels=plan.labels,
        destination_directory=destination_directory,
    )


def _run_successor(plan, **kwargs):
    from xrd_tools.reduction import run_reintegrate_successor

    return run_reintegrate_successor(_successor_from_plan(plan), **kwargs)
def _preserved_signature(path):
    allowed = ("entry/integrated_1d", "entry/reduction/config/bai_1d_args",
               "entry/reduction/config/gi_config",
               "entry/reduction/config/dimension_replacement_1d")
    def dtype(value):
        value = np.dtype(value); string, vlen, enum, ref = h5py.check_string_dtype(value), h5py.check_vlen_dtype(value), h5py.check_enum_dtype(value), h5py.check_ref_dtype(value); tag = lambda item: None if item is None else ("dtype", np.dtype(item).str, np.dtype(item).descr) if isinstance(item, np.dtype) else ("type", item.__module__, item.__qualname__); return value.str, value.descr, None if string is None else (string.encoding, string.length), tag(vlen), tuple(sorted((enum or {}).items())), tag(ref)
    def payload(value, declared=None):
        array = np.asarray(value); raw = json.dumps([("bytes", item.hex()) if isinstance(item, bytes) else ("text", str(item)) for item in array.ravel()], separators=(",", ":")).encode() if array.dtype.kind in "OU" else array.tobytes(); return dtype(array.dtype if declared is None else declared), array.shape, hashlib.sha256(raw).hexdigest()
    result = {}
    with h5py.File(path, "r") as handle:
        def walk(group, active=()):
            name = group.name.lstrip("/") or "/"; attrs = tuple((key, payload(group.attrs[key], group.attrs.get_id(key).dtype)) for key in sorted(group.attrs)); result[name] = result.get(name, ()) + ("group", attrs)
            for key in sorted(group):
                name = f"{group.name.rstrip('/')}/{key}".lstrip("/"); link = group.get(key, getlink=True); result[name] = ("link", type(link).__name__, getattr(link, "filename", None), getattr(link, "path", None))
                if any(name == root or name.startswith(root + "/") for root in allowed) or type(link) is not h5py.HardLink: continue
                obj = group.get(key); attrs = tuple((attr, payload(obj.attrs[attr], obj.attrs.get_id(attr).dtype)) for attr in sorted(obj.attrs))
                if isinstance(obj, h5py.Group): (None if any(obj.id == owner for owner in (*active, group.id)) else walk(obj, (*active, group.id)))
                else: result[name] += ("dataset", dtype(obj.dtype), obj.shape, obj.maxshape, obj.chunks, obj.compression, obj.compression_opts, attrs, payload(obj[()], obj.dtype))
        walk(handle)
    return {name: value for name, value in result.items() if name != "entry/reduction/config/dimension_replacement_1d"}


def test_selected_project_root_relocates_reintegrate_and_restamps_context(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.io.read import ProcessedScan
    from xrd_tools.reduction import run_reintegrate_successor

    module = _module()
    seeded = _seed_existing(
        tmp_path, name="old-project", labels=(2, 3), append=True,
    )
    old_root = seeded.target.parent.resolve()
    new_root = (tmp_path / "new-project").resolve()
    shutil.copytree(old_root, new_root)
    moved_target = new_root / seeded.target.name
    old_root.rename(tmp_path / "retired-old-project")

    _stub_integrators(monkeypatch)
    plan = module.ReintegratePlan.from_artifact(
        moved_target,
        entry="entry",
        dimension="1d",
        preparation=seeded.preparation,
        source_root=str(new_root),
        expected_target_snapshot=capture_target_snapshot(moved_target),
        expected_labels=seeded.labels,
    )
    assert plan.source_root == str(new_root)
    source_before = moved_target.read_bytes()
    result = run_reintegrate_successor(_successor_from_plan(plan))
    assert result.disposition == "COMMITTED"
    assert result.committed_labels == seeded.labels
    assert moved_target.read_bytes() == source_before

    with h5py.File(result.output_artifact, "r") as handle:
        entry = handle["entry"]
        assert entry.attrs["source_base"] == new_root.as_posix()
        execution = json.loads(
            entry["reduction/config/source_execution"].asstr()[()]
        )
        assert execution["path"] == str(new_root / seeded.source.name)
        lineage = json.loads(
            entry["reduction/config/append_lineage"].asstr()[()]
        )
        assert lineage["source_base"] == str(new_root)
        assert all(
            epoch["source"]["path"] == str(new_root / seeded.source.name)
            for epoch in lineage["epochs"]
        )

    scan = ProcessedScan(result.output_artifact, source_root=new_root)
    np.testing.assert_array_equal(scan.load_frame(2), seeded.raw[2])


def test_target_snapshot_mismatch_abandons_before_stream_and_releases_lease(tmp_path, monkeypatch):
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import NexusSink, ReductionPlan, Scan
    module = _module()
    target = tmp_path / "existing.nexus"
    target.write_bytes(b"one")
    wrong = module.TargetSnapshot(True, 3, 0, 0, 0, "0" * 64)
    effects: list[str] = []
    monkeypatch.setattr(module, "_inspect_artifact",
                        lambda *a, **k: effects.append("inspect"))
    monkeypatch.setattr(module, "_prepare_gi_scouts",
                        lambda *a, **k: effects.append("scout"))
    with pytest.raises(ValueError, match="TARGET_SNAPSHOT_CHANGED"):
        module.ReintegratePlan.from_artifact(
            target, entry="entry", dimension="1d",
            preparation=_preparation(), expected_target_snapshot=wrong,
        )
    assert effects == []
    seeded = _seed_existing(tmp_path, name="post-lease")
    original = capture_target_snapshot(seeded.target)
    selected = seeded.preparation["selected_plan"]
    sink = NexusSink.for_existing_replacement(
        seeded.target, expected_target_snapshot=original, dimension="1d",
        labels=seeded.labels, audit_bytes=b"{}", selected_plan=selected["bai_args"],
        selected_gi_mode=None, source_base=seeded.target.parent,
        file_lock=threading.RLock(), flush_every=None,
    )
    with h5py.File(seeded.target, "r+") as handle:
        handle["entry"].attrs["snapshot_bump"] = 1
    changed = seeded.target.read_bytes()
    with pytest.raises(ValueError, match="TARGET_SNAPSHOT_CHANGED"):
        sink.begin(Scan("mismatch", []), ReductionPlan(integration_2d=None))
    assert sink._transaction_owners is None
    retry = NexusSink.for_existing_replacement(
        seeded.target, expected_target_snapshot=capture_target_snapshot(seeded.target),
        dimension="1d", labels=seeded.labels, audit_bytes=b"{}",
        selected_plan=selected["bai_args"], selected_gi_mode=None,
        source_base=seeded.target.parent, file_lock=threading.RLock(), flush_every=None,
    )
    retry.begin(Scan("retry", []), ReductionPlan(integration_2d=None))
    assert retry.abort(None).disposition.value == "aborted"
    assert seeded.target.read_bytes() == changed


def test_processed_target_transient_replacement_is_refused_before_inspection(
    tmp_path, monkeypatch,
):
    import os

    module = _module()
    seeded = _seed_existing(tmp_path, name="target-inspection")
    foreign = _seed_existing(tmp_path, name="target-inspection-foreign")
    parked = tmp_path / "target-inspection-parked.nexus"
    original = seeded.target.read_bytes()
    real_file = module._open_target_hdf
    swaps = []

    def transient_file(path):
        if Path(path) != seeded.target or swaps:
            return real_file(path)
        os.replace(seeded.target, parked)
        os.replace(foreign.target, seeded.target)
        try:
            handle = real_file(path)
        finally:
            os.replace(seeded.target, foreign.target)
            os.replace(parked, seeded.target)
        swaps.append(1)
        return handle

    monkeypatch.setattr(module, "_open_target_hdf", transient_file)
    with pytest.raises(ValueError, match="TARGET_SNAPSHOT_CHANGED"):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert swaps == [1]
    assert seeded.target.read_bytes() == original


def test_processed_target_transient_replacement_is_refused_before_gi_scout(
    tmp_path, monkeypatch,
):
    import os

    from xrd_tools.reduction import GIMode

    module = _module()
    gi = GIMode(
        incidence_motor="theta", mode_1d="q_total", mode_2d="qip_qoop",
    )
    seeded = _seed_existing(tmp_path, name="target-scout", gi=gi)
    foreign = _seed_existing(
        tmp_path, name="target-scout-foreign", gi=gi,
    )
    parked = tmp_path / "target-scout-parked.nexus"
    original = seeded.target.read_bytes()
    real_file = module._open_target_hdf
    opens = []

    def transient_file(path):
        if Path(path) != seeded.target:
            return real_file(path)
        opens.append(1)
        if len(opens) != 2:
            return real_file(path)
        os.replace(seeded.target, parked)
        os.replace(foreign.target, seeded.target)
        try:
            handle = real_file(path)
        finally:
            os.replace(seeded.target, foreign.target)
            os.replace(parked, seeded.target)
        return handle

    monkeypatch.setattr(module, "_open_target_hdf", transient_file)
    with pytest.raises(ValueError, match="TARGET_SNAPSHOT_CHANGED"):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert len(opens) == 2
    assert seeded.target.read_bytes() == original


@pytest.mark.parametrize(
    "family", ("inventory", "detector", "geometry", "scan"),
)
def test_persisted_row_schema_refuses_before_dataset_read(
    tmp_path, monkeypatch, family,
):
    from xrd_tools.reduction import GIMode

    module = _module()
    gi = None if family not in {"geometry", "scan"} else GIMode(
        incidence_motor="theta", mode_1d="q_total", mode_2d="qip_qoop",
    )
    seeded = _seed_existing(
        tmp_path, labels=(2,), gi=gi, name=f"bounded-{family}",
    )
    guarded = ()
    with h5py.File(seeded.target, "r+") as handle:
        if family == "inventory":
            group = handle["entry/integrated_1d"]
            del group["frame_index"]
            group.create_dataset(
                "frame_index", data=np.asarray([2, 3], dtype=np.int64),
            )
            guarded = ("/entry/integrated_1d/frame_index",)
        elif family == "detector":
            group = handle["entry/instrument/detector"]
            del group["detector_shape"]
            group.create_dataset(
                "detector_shape", data=np.asarray([5, 7, 9], dtype=np.int64),
            )
            guarded = ("/entry/instrument/detector/detector_shape",)
        elif family == "geometry":
            group = handle["entry"].create_group("per_frame_geometry")
            group.create_dataset(
                "frame_index", data=np.asarray([2, 3], dtype=np.int64),
            )
            for name in ("rot1", "rot2", "rot3", "incident_angle"):
                group.create_dataset(
                    name, data=np.asarray([0.1, 0.2], dtype=np.float32),
                )
            guarded = tuple(
                f"/entry/per_frame_geometry/{name}"
                for name in ("frame_index", "rot1", "rot2", "rot3",
                             "incident_angle")
            )
        else:
            group = handle["entry/scan_data"]
            del group["frame_index"]
            del group["theta"]
            group.create_dataset(
                "frame_index", data=np.asarray([2, 3], dtype=np.int64),
            )
            group.create_dataset(
                "theta", data=np.asarray([0.2, 0.3], dtype=np.float32),
            )
            guarded = ("/entry/scan_data/frame_index", "/entry/scan_data/theta")

    if family == "geometry":
        shared = copy.deepcopy(seeded.preparation["requested_shared_science"])
        shared["geometry"] = {
            "convention": "bounded-test", "mapping_json": "{}",
            "motor_sources": {},
        }
        monkeypatch.setattr(
            module, "_validated_shared_science",
            lambda *_a, **_k: copy.deepcopy(shared),
        )
        monkeypatch.setattr(module, "_validate_science", lambda *_a, **_k: None)
        monkeypatch.setattr(
            module, "_canonical_acquisition_selected",
            lambda _run, _shared, _dimension:
                copy.deepcopy(seeded.preparation["selected_plan"]),
        )
    if family != "detector":
        import xrd_tools.io.record_writer as writer_module
        monkeypatch.setattr(writer_module, "_MAX_REPLACEMENT_FRAME_ROWS", 1)

    reads = []
    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name in guarded:
            reads.append((dataset.name, key))
            raise AssertionError(f"{family} dataset was materialized")
        return real_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    messages = {
        "inventory": "selected frame inventory is not exact",
        "detector": "processed detector descriptor is malformed",
        "geometry": "replacement geometry differs",
        "scan": "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    }
    with pytest.raises(ValueError, match=messages[family]):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert reads == []


def test_persisted_mask_external_storage_refuses_via_owned_dcpl_before_effect(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import NexusSink

    module = _module()
    seeded = _seed_existing(tmp_path, name="external-persisted-mask")
    external = seeded.target.parent / "persisted-mask.bin"
    external.write_bytes(np.asarray([0], dtype=np.int64).tobytes())
    with h5py.File(seeded.target, "r+") as handle:
        detector = handle["entry/instrument/detector"]
        del detector["mask"]
        mask = detector.create_dataset(
            "mask", shape=(1,), dtype=np.int64,
            external=[(str(external), 0, np.dtype(np.int64).itemsize)],
        )
        mask.attrs["description"] = "flat pixel indices, shape (N,)"
        properties = mask.id.get_create_plist()
        try:
            assert properties.get_layout() == h5py.h5d.CONTIGUOUS
            assert properties.get_external_count() == 1
        finally:
            properties.close()
    before = seeded.target.read_bytes()
    high_level_external = []
    pixels = []
    allocations = []
    sinks = []
    real_getitem = h5py.Dataset.__getitem__
    real_external = h5py.Dataset.external.fget

    def forbidden_external(dataset):
        if dataset.name == "/entry/instrument/detector/mask":
            high_level_external.append((dataset.file.filename, dataset.name))
            raise AssertionError("mask external storage was enumerated")
        return real_external(dataset)

    def guarded_getitem(dataset, key):
        if ((Path(dataset.file.filename) == seeded.target
             and dataset.name == "/entry/instrument/detector/mask")
                or (Path(dataset.file.filename) == seeded.source
                    and dataset.name == "/entry/instrument/detector/data")):
            pixels.append((dataset.file.filename, dataset.name, key))
            raise AssertionError("mask/source pixels were read")
        return real_getitem(dataset, key)

    monkeypatch.setattr(
        h5py.Dataset, "external", property(forbidden_external),
    )
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    monkeypatch.setattr(
        module, "_policy",
        lambda *args, **kwargs: allocations.append(1),
    )
    monkeypatch.setattr(
        NexusSink, "for_existing_replacement",
        classmethod(lambda cls, *args, **kwargs: sinks.append(1)),
    )
    with pytest.raises(ValueError, match="processed detector mask is malformed"):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert high_level_external == []
    assert pixels == []
    assert allocations == [] and sinks == []
    assert seeded.target.read_bytes() == before


@pytest.mark.parametrize(
    "family", ("config-geometry", "per-frame-geometry", "mask-attrs"),
)
def test_replacement_inventory_cardinality_refuses_before_iteration(
    tmp_path, monkeypatch, family,
) -> None:
    from xrd_tools.reduction import GIMode

    module = _module()
    seeded = _seed_existing(
        tmp_path, labels=(2,),
        gi=(GIMode(
            incidence_motor="theta", mode_1d="q_total",
            mode_2d="qip_qoop",
        ) if family == "per-frame-geometry" else None),
        name=f"excess-{family}",
    )
    guarded_group = None
    guarded_attrs = None
    with h5py.File(seeded.target, "r+") as handle:
        if family == "config-geometry":
            group = handle["entry/reduction/config"].create_group("geometry")
            for name, value in (
                ("convention", "test"), ("mapping_json", "{}"),
                ("motor_sources", "{}"), ("foreign", "foreign"),
            ):
                group.create_dataset(name, data=value)
            guarded_group = group.name
        elif family == "per-frame-geometry":
            group = handle["entry"].create_group("per_frame_geometry")
            group.create_dataset("frame_index", data=np.asarray([2], dtype=np.int64))
            for name in ("rot1", "rot2", "rot3", "incident_angle"):
                group.create_dataset(name, data=np.asarray([0.1], dtype=np.float32))
            group.create_dataset("foreign", data=np.asarray([0.1], dtype=np.float32))
            guarded_group = group.name
        else:
            mask = handle["entry/instrument/detector/mask"]
            mask.attrs["foreign"] = "foreign"
            guarded_attrs = mask.name

    if family == "per-frame-geometry":
        shared = copy.deepcopy(seeded.preparation["requested_shared_science"])
        shared["geometry"] = {
            "convention": "bounded-test", "mapping_json": "{}",
            "motor_sources": {},
        }
        monkeypatch.setattr(
            module, "_validated_shared_science",
            lambda *_args, **_kwargs: copy.deepcopy(shared),
        )
        monkeypatch.setattr(module, "_validate_science", lambda *_a, **_k: None)
        monkeypatch.setattr(
            module, "_canonical_acquisition_selected",
            lambda _run, _shared, _dimension:
                copy.deepcopy(seeded.preparation["selected_plan"]),
        )

    real_group_iter = h5py.Group.__iter__
    real_attrs_iter = h5py.AttributeManager.__iter__
    iterations = []

    def guarded_group_iter(group):
        if group.name == guarded_group:
            iterations.append(group.name)
            raise AssertionError("excess group inventory was iterated")
        return real_group_iter(group)

    def guarded_attrs_iter(attrs):
        name = h5py.h5i.get_name(attrs._id).decode()
        if name == guarded_attrs:
            iterations.append(name)
            raise AssertionError("excess attribute inventory was iterated")
        return real_attrs_iter(attrs)

    monkeypatch.setattr(h5py.Group, "__iter__", guarded_group_iter)
    monkeypatch.setattr(
        h5py.AttributeManager, "__iter__", guarded_attrs_iter,
    )
    message = (
        "selected BAI/GI/geometry is malformed"
        if family == "config-geometry" else
        "replacement geometry differs"
        if family == "per-frame-geometry" else
        "processed detector mask is malformed"
    )
    with pytest.raises(ValueError, match=message):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert iterations == []


@pytest.mark.parametrize("family", ("untouched-primary", "named-mode"))
def test_replacement_writer_bounds_every_untouched_cursor_before_effects(
    tmp_path, monkeypatch, family,
):
    from xrd_tools.io.record_writer import (
        NexusRecordWriter, WriterIncomplete,
    )
    from xrd_tools.reduction import NexusSink, Scan
    import xrd_tools.io.record_writer as writer_module

    module = _module()
    seeded = _seed_existing(
        tmp_path, labels=(2,), name=f"bounded-writer-{family}",
    )
    with h5py.File(seeded.target, "r+") as handle:
        top = handle["entry/integrated_2d"]
        if family == "untouched-primary":
            del top["frame_index"]
            node = top.create_dataset(
                "frame_index", data=np.asarray([2, 3], dtype=np.int64),
            )
        else:
            node = top.create_group("oversized_mode").create_dataset(
                "frame_index", data=np.asarray([2, 3], dtype=np.int64),
            )
        guarded = node.name

    monkeypatch.setattr(writer_module, "_MAX_REPLACEMENT_FRAME_ROWS", 1)
    plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d",
        preparation=seeded.preparation,
    )
    before = seeded.target.read_bytes()
    sink = NexusSink.for_existing_replacement(
        seeded.target,
        expected_target_snapshot=plan.expected_target_snapshot,
        dimension="1d", labels=plan.labels, audit_bytes=b"{}",
        selected_plan=plan.selected_plan["bai_args"],
        selected_gi_mode=plan.selected_plan["gi_mode"],
        source_base=seeded.target.parent, file_lock=threading.RLock(),
        flush_every=None,
    )
    reads, mutations = [], []
    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name == guarded:
            reads.append(key)
            raise AssertionError("oversized untouched cursor was materialized")
        return real_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    monkeypatch.setattr(
        NexusRecordWriter, "_authorize_transaction_mutation",
        lambda *_args, **_kwargs: mutations.append("mutation"),
    )
    with pytest.raises(WriterIncomplete, match="exact bounded int64 vector"):
        sink.begin(
            Scan("bounded-cursor", []),
            module._core_plan(
                plan.selected_plan, plan.requested_shared_science,
            ),
        )
    assert reads == [] and mutations == []
    assert sink._transaction_owners is None
    assert seeded.target.read_bytes() == before


def test_replacement_json_schema_refuses_before_read_or_decode(
    tmp_path, monkeypatch,
):
    import xrd_tools.io.record_writer as writer_module

    module = _module()
    seeded = _seed_existing(tmp_path, name="malformed-config-scalar")
    with h5py.File(seeded.target, "r+") as handle:
        config = handle["entry/reduction/config"]
        raw = config["run_configuration"][()]
        del config["run_configuration"]
        node = config.create_dataset(
            "run_configuration", data=np.asarray([raw], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        guarded = node.name

    reads, decodes = [], []
    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name == guarded:
            reads.append(key)
            raise AssertionError("malformed JSON scalar was materialized")
        return real_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    real_json = writer_module.json
    monkeypatch.setattr(writer_module, "json", SimpleNamespace(
        loads=lambda *_args, **_kwargs: decodes.append("decode"),
        dumps=real_json.dumps, JSONDecodeError=real_json.JSONDecodeError,
    ))
    with pytest.raises(ValueError, match="run configuration"):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert reads == [] and decodes == []


def test_replacement_numeric_vlen_metadata_refuses_before_cell_read(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.record_writer import (
        WriterStateError, _decode_replacement_fact,
    )

    seeded = _seed_existing(
        tmp_path, labels=(2,), name="numeric-vlen-metadata",
    )
    with h5py.File(seeded.target, "r+") as handle:
        scan = handle["entry/scan_data"]
        node = scan.create_dataset(
            "foreign_vlen", shape=(1,), dtype=h5py.vlen_dtype(np.int64),
        )
        node[0] = np.arange(3, dtype=np.int64)
        guarded = node.name

    reads = []
    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name == guarded:
            reads.append(key)
            raise AssertionError("numeric VLEN metadata cell was materialized")
        return real_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    with h5py.File(seeded.target, "r") as handle:
        with pytest.raises(WriterStateError, match="unsupported vlen schema"):
            _decode_replacement_fact(
                handle, 2, metadata_keys=("foreign_vlen",),
            )
    assert reads == []


def test_replacement_config_ceiling_refuses_before_getitem_or_json_decode(
    tmp_path, monkeypatch,
):
    import xrd_tools.io.append as append_module
    import xrd_tools.io.record_writer as writer_module

    module = _module()
    seeded = _seed_existing(tmp_path, name="oversized-config-scalar")
    guarded = "/entry/reduction/config/run_configuration"
    reads, decodes = [], []
    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name == guarded:
            reads.append(key)
            raise AssertionError("oversized config scalar used __getitem__")
        return real_getitem(dataset, key)

    monkeypatch.setattr(append_module, "_MAX_REPLACEMENT_CONFIG_UTF8_BYTES", 64)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    real_json = writer_module.json
    monkeypatch.setattr(writer_module, "json", SimpleNamespace(
        loads=lambda *_args, **_kwargs: decodes.append("decode"),
        dumps=real_json.dumps, JSONDecodeError=real_json.JSONDecodeError,
    ))
    with pytest.raises(ValueError, match="UTF-8 byte ceiling"):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert reads == [] and decodes == []


def test_replacement_geometry_text_ceiling_refuses_before_getitem(
    tmp_path, monkeypatch,
):
    import xrd_tools.io.append as append_module

    module = _module()
    seeded = _seed_existing(tmp_path, name="oversized-geometry-text")
    with h5py.File(seeded.target, "r+") as handle:
        geometry = handle["entry/reduction/config"].create_group("geometry")
        node = geometry.create_dataset("convention", data="x" * 5000)
        geometry.create_dataset("mapping_json", data="{}")
        geometry.create_dataset("motor_sources", data="{}")
        guarded = node.name

    reads = []
    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name == guarded:
            reads.append(key)
            raise AssertionError("oversized geometry text used __getitem__")
        return real_getitem(dataset, key)

    monkeypatch.setattr(append_module, "_MAX_REPLACEMENT_CONFIG_UTF8_BYTES", 4096)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    with pytest.raises(ValueError, match="UTF-8 byte ceiling"):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert reads == []


def test_replacement_source_execution_ceiling_refuses_before_decode(
    tmp_path, monkeypatch,
):
    import xrd_tools.io.record_writer as writer_module

    seeded = _seed_existing(tmp_path, labels=(2,), name="oversized-execution")
    guarded = "/entry/reduction/config/source_execution"
    reads, oversized_decodes = [], []
    real_getitem, real_json = h5py.Dataset.__getitem__, writer_module.json

    def guarded_getitem(dataset, key):
        if dataset.name == guarded:
            reads.append(key)
            raise AssertionError("oversized source_execution used __getitem__")
        return real_getitem(dataset, key)

    def guarded_loads(raw, *args, **kwargs):
        if isinstance(raw, str) and len(raw.encode("utf-8")) > 64:
            oversized_decodes.append(len(raw))
            raise AssertionError("oversized source_execution reached JSON")
        return real_json.loads(raw, *args, **kwargs)

    monkeypatch.setattr(writer_module, "_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES", 64)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    monkeypatch.setattr(writer_module, "json", SimpleNamespace(
        loads=guarded_loads, dumps=real_json.dumps,
        JSONDecodeError=real_json.JSONDecodeError,
    ))
    with h5py.File(seeded.target, "r") as handle:
        with pytest.raises(writer_module.WriterStateError, match="source context"):
            writer_module._decode_replacement_fact(handle, 2, entry="entry")
    assert reads == [] and oversized_decodes == []


def test_replacement_append_lineage_ceiling_refuses_before_decode(
    tmp_path, monkeypatch,
):
    import xrd_tools.io.append as append_module

    module = _module()
    seeded = _seed_existing(
        tmp_path, labels=(2,), append=True, name="oversized-lineage",
    )
    guarded = "/entry/reduction/config/append_lineage"
    reads, decodes = [], []
    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name == guarded:
            reads.append(key)
            raise AssertionError("oversized Append lineage used __getitem__")
        return real_getitem(dataset, key)

    monkeypatch.setattr(append_module, "_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES", 64)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    real_json = append_module.json
    monkeypatch.setattr(append_module, "json", SimpleNamespace(
        loads=lambda *_args, **_kwargs: decodes.append("decode"),
        dumps=real_json.dumps, JSONDecodeError=real_json.JSONDecodeError,
    ))
    with pytest.raises(ValueError, match="REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert reads == [] and decodes == []


def test_provenance_writer_enforces_replacement_read_ceiling(
    tmp_path, monkeypatch,
):
    import xrd_tools.io.append as append_module
    from xrd_tools.core.provenance import write_provenance

    monkeypatch.setattr(append_module, "_MAX_REPLACEMENT_CONFIG_UTF8_BYTES", 32)
    target = tmp_path / "bounded-provenance.nexus"
    with h5py.File(target, "w") as handle:
        with pytest.raises(ValueError, match="persisted UTF-8 byte ceiling"):
            write_provenance(
                handle, config={"run_configuration": {"value": "x" * 64}},
            )
        config = handle["entry/reduction/config"]
        assert "run_configuration" not in config
    geometry_target = tmp_path / "bounded-geometry-provenance.nexus"
    with h5py.File(geometry_target, "w") as handle:
        with pytest.raises(ValueError, match="persisted UTF-8 byte ceiling"):
            write_provenance(handle, config={"geometry": {
                "convention": "test", "mapping_json": "x" * 64,
                "motor_sources": {},
            }})
        geometry = handle["entry/reduction/config/geometry"]
        assert "mapping_json" not in geometry


def test_processed_mask_byte_ceiling_refuses_before_read(
    tmp_path, monkeypatch,
):
    module = _module()
    seeded = _seed_existing(tmp_path, name="oversized-persisted-mask")
    with h5py.File(seeded.target, "r+") as handle:
        detector = handle["entry/instrument/detector"]
        del detector["mask"]
        node = detector.create_dataset(
            "mask", data=np.arange(5, dtype=np.int64),
        )
        node.attrs["description"] = "flat pixel indices, shape (N,)"

    reads = []
    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name == "/entry/instrument/detector/mask":
            reads.append(key)
            raise AssertionError("oversized persisted mask was materialized")
        return real_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    # The 5-index vector is 40 bytes while the detector's expanded bool mask is
    # 35 bytes, so this isolates the persisted-vector ceiling without a large
    # test allocation.
    monkeypatch.setattr(module, "_MAX_PERSISTED_MASK_BYTES", 35)
    with pytest.raises(ValueError, match="processed detector mask is malformed"):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert reads == []


@pytest.mark.parametrize(
    "indices", (np.asarray([0, 0], dtype=np.int64),
                np.asarray([1, 0], dtype=np.int64)),
    ids=("duplicate", "inverted"),
)
def test_processed_mask_indices_must_be_strictly_canonical(tmp_path, indices):
    module = _module()
    seeded = _seed_existing(
        tmp_path, name=f"noncanonical-persisted-mask-{indices.tolist()}",
    )
    with h5py.File(seeded.target, "r+") as handle:
        detector = handle["entry/instrument/detector"]
        del detector["mask"]
        node = detector.create_dataset("mask", data=indices)
        node.attrs["description"] = "flat pixel indices, shape (N,)"

    with pytest.raises(ValueError, match="processed detector mask is malformed"):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )


def test_replay_cannot_hide_persisted_mask_before_allocation(tmp_path, monkeypatch):
    from xrd_tools.reduction import NexusSink
    from xrd_tools.reduction import run_reintegrate_successor

    module = _module()
    seeded = _seed_existing(tmp_path, name="false-mask-replay")
    admitted = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d",
        preparation=seeded.preparation,
    )
    requirements = module._requirements(
        admitted.detector_shape, admitted.native_dtype,
        admitted.selected_plan, admitted.requested_shared_science,
    )
    false_policy = module._policy(
        requirements, seeded.preparation["resource_policy"],
        module._PersistedMaskSpec(0, 0),
    )
    false_plan = module._make_plan(
        admitted.target, admitted.entry, admitted.source_root,
        admitted.dimension, admitted.labels,
        admitted.detector_shape, admitted.native_dtype,
        module._plain(admitted.selected_plan),
        module._plain(admitted.requested_shared_science),
        admitted.gi_bootstrap_incidence, 0, 0, false_policy,
        snapshot=admitted.expected_target_snapshot,
    )
    before = seeded.target.read_bytes()
    mask_reads = []
    sinks = []
    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name == "/entry/instrument/detector/mask":
            mask_reads.append(key)
            raise AssertionError("false replay materialized the mask")
        return real_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    monkeypatch.setattr(
        NexusSink, "for_finite_replacement",
        classmethod(lambda cls, *args, **kwargs: sinks.append(1)),
    )
    with pytest.raises(
        ValueError, match="REINTEGRATE_MASK_OWNER_BLOCK_GRANT_CHANGED",
    ):
        run_reintegrate_successor(_successor_from_plan(false_plan))
    assert mask_reads == []
    assert sinks == []
    assert seeded.target.read_bytes() == before


def test_mask_peak_has_exact_owner_grant_before_materialization(
    tmp_path, monkeypatch,
):
    from xrd_tools.session.policy import (
        FlushPolicy, minimum_bytes, resolve_session_policy,
    )

    module = _module()
    seeded = _seed_existing(tmp_path, name="mask-owner-floor")
    with h5py.File(seeded.target, "r+") as handle:
        detector = handle["entry/instrument/detector"]
        del detector["mask"]
        node = detector.create_dataset(
            "mask", data=np.arange(35, dtype=np.int64))
        node.attrs["description"] = "flat pixel indices, shape (N,)"
    plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d",
        preparation=seeded.preparation,
    )
    spec = module._PersistedMaskSpec(35, 35 * 8)
    expected_owner = module._mask_owner_block_bytes(
        plan.resource_allocation.requirements, spec)
    assert (plan.retained_mask_bytes, plan.mask_decode_bytes) == spec
    assert plan.resource_allocation.owner_block_bytes == expected_owner == 350

    requirements = plan.resource_allocation.requirements
    zero = module._policy(
        requirements, seeded.preparation["resource_policy"],
        module._PersistedMaskSpec(0, 0),
    )
    baseline = resolve_session_policy(
        requirements,
        envelope_bytes=seeded.preparation["resource_policy"]["envelope_bytes"],
        requests=seeded.preparation["resource_policy"]["requests"],
        flush=FlushPolicy(),
    )
    assert zero == baseline

    constrained = copy.deepcopy(seeded.preparation)
    constrained["resource_policy"]["envelope_bytes"] = minimum_bytes(
        requirements)
    reads = []
    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name == "/entry/instrument/detector/mask":
            reads.append(key)
            raise AssertionError("unfunded persisted mask was materialized")
        return real_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    with pytest.raises(
        ValueError, match="REINTEGRATE_MASK_OWNER_BLOCK_GRANT_INCOMPLETE",
    ):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=constrained,
        )
    assert reads == []


def test_exact_gapped_inventory_replaces_only_selected_dimension(tmp_path, monkeypatch):
    seeded = _seed_existing(tmp_path, name="gapped")
    before = _preserved_signature(seeded.target)
    seen = _stub_integrators(monkeypatch)
    plan = _module().ReintegratePlan.from_artifact(seeded.target, entry="entry", dimension="1d", preparation=seeded.preparation)
    result = _run_successor(plan)
    assert result.disposition == "COMMITTED" and result.committed_labels == seeded.labels
    assert type(result.commit_identity) is str
    assert seen == [(2, 3), (5, 3), (9, 3)]
    assert _preserved_signature(seeded.target) == before
    with h5py.File(result.output_artifact, "r") as handle:
        group = handle["entry/integrated_1d"]
        np.testing.assert_array_equal(group["frame_index"][()], seeded.labels)
        np.testing.assert_allclose(group["axis_1"][()], _r1(102, q0=.2).radial)
        assert not np.array_equal(group["axis_1"][()], _r1(2).radial)
        np.testing.assert_allclose(group["intensity"][()],
                                   np.stack([_r1(v + 100).intensity for v in seeded.labels]))
        np.testing.assert_allclose(group["sigma"][()],
                                   np.stack([_r1(v + 100).sigma for v in seeded.labels]))
        assert "dimension_replacement_1d" in handle["entry/reduction/config"]
    replay = _module().ReintegratePlan.from_artifact(result.output_artifact, entry="entry", dimension="1d", preparation=seeded.preparation)
    assert replay.labels == seeded.labels
    from xrd_tools.reduction.provenance_config import _integration_2d_args; two_prep = copy.deepcopy(seeded.preparation); two_args = _integration_2d_args(_plans()[1], None); two_args.pop("gi_mode_2d", None); two_prep["selected_plan"] = {"version": 1, "dimension": "2d", "bai_args": two_args, "gi_mode": None}; two_result = _run_successor(_module().ReintegratePlan.from_artifact(result.output_artifact, entry="entry", dimension="2d", preparation=two_prep)); assert two_result.committed_labels == seeded.labels


def test_reintegrate_detaches_selected_monitor_metadata(tmp_path, monkeypatch):
    import xrd_tools.reduction.core as core

    from xrd_tools.io.record_writer import WriterStateError, _decode_replacement_fact

    seeded = _seed_existing(
        tmp_path, name="monitor", monitor="monitor",
        monitor_metadata="MonItor",
    )
    observed = []

    def integrate(image, _integrator, *, normalization_factor=None, **_kwargs):
        label = int(np.asarray(image).flat[0] // 35)
        observed.append((label, normalization_factor))
        return _r1(label + 100)

    monkeypatch.setattr(core, "integrate_1d", integrate)
    module = _module()
    with h5py.File(seeded.target, "r") as handle:
        fact = _decode_replacement_fact(
            handle, seeded.labels[0], metadata_keys=("monitor",),
        )
    assert fact["metadata"] == {"MonItor": pytest.approx(float(seeded.labels[0] + 1))}
    with h5py.File(seeded.target, "r+") as handle:
        scan_data = handle["entry/scan_data"]
        scan_data.create_dataset("MONITOR", data=scan_data["MonItor"][()])
    with h5py.File(seeded.target, "r") as handle, pytest.raises(
        WriterStateError, match="ambiguous",
    ):
        _decode_replacement_fact(
            handle, seeded.labels[0], metadata_keys=("monitor",),
        )
    with h5py.File(seeded.target, "r+") as handle:
        del handle["entry/scan_data/MONITOR"]
    plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d",
        preparation=seeded.preparation,
    )
    assert plan.selected_plan["bai_args"]["monitor"] == "monitor"
    result = _run_successor(plan)
    assert result.disposition == "COMMITTED"
    assert observed == [(label, float(label + 1)) for label in seeded.labels]


def test_production_gui_provenance_is_canonicalized_from_authenticated_facts(tmp_path, monkeypatch):
    from xrd_tools.core.provenance import read_provenance_from_handle
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction.provenance_config import (
        _integration_1d_args, _integration_2d_args,
    )
    module = _module(); seeded = _seed_existing(
        tmp_path, name="production-gui", persisted_poni=False,
        legacy_bai=True, detector_descriptor=False, source_options=False)
    source_before = seeded.target.read_bytes()
    request = copy.deepcopy(seeded.preparation); request["requested_shared_science"] = {"version": 1, "kind": "persisted_target"}; request["selected_plan"] = {"version": 1, "dimension": "1d", "gi_mode": "q_total", "bai_args": {"numpoints": 4, "unit": "q_A^-1", "method": "numpy", "radial_range": None, "azimuth_range": None, "chi_offset": 90.0, "npt_oop": 1000}}
    plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d", preparation=request,
        expected_target_snapshot=capture_target_snapshot(seeded.target),
        expected_labels=seeded.labels)
    assert plan.detector_shape == seeded.raw.shape[1:]
    expected_one = {
        "version": 1, "dimension": "1d", "gi_mode": None,
        "bai_args": _integration_1d_args(_plans()[0], None),
    }
    assert module._plain(plan.selected_plan) == expected_one
    assert plan.requested_shared_science["poni_values"] == plan.requested_shared_science["accepted_scientific_assets"]["poni_values"]
    one_result = _run_successor(plan)
    assert one_result.committed_labels == seeded.labels
    two_args = _integration_2d_args(_plans()[1], None)
    two_args.pop("gi_mode_2d", None)
    two_request = copy.deepcopy(request)
    two_request["selected_plan"] = {
        "version": 1, "dimension": "2d", "bai_args": two_args,
        "gi_mode": "q_chi",
    }
    two_plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="2d",
        preparation=two_request,
        expected_target_snapshot=capture_target_snapshot(seeded.target),
        expected_labels=seeded.labels)
    two_result = _run_successor(two_plan)
    assert two_result.committed_labels == seeded.labels
    assert seeded.target.read_bytes() == source_before
    with h5py.File(one_result.output_artifact, "r") as handle: run = read_provenance_from_handle(handle)["config"]["run_configuration"]
    assert run["poni_values"] is None and "options" not in run["source"]
    conflicting = copy.deepcopy(run); conflicting["poni_values"] = {"conflict": True}; conflicting["scientific_signature"]["poni_values"] = {"conflict": True}
    with pytest.raises(ValueError, match="accepted scientific assets"): module._validated_shared_science(conflicting)
    explicit_null = _seed_existing(
        tmp_path, name="production-null-options", persisted_poni=False,
        legacy_bai=True, detector_descriptor=False, source_options=None)
    with pytest.raises(ValueError, match="REPLACEMENT_RAW_DECODER_UNRECORDED"):
        module.ReintegratePlan.from_artifact(
            explicit_null.target, entry="entry", dimension="1d",
            preparation=request,
            expected_target_snapshot=capture_target_snapshot(
                explicit_null.target),
            expected_labels=explicit_null.labels)
    import tifffile
    from xdart.gui.tabs.scattering.contracts import (
        SourceExecutionStamp, SourceFileState,
    )

    def mixed_fixture(name, *, raw_first, decoder):
        mixed = _seed_existing(
            tmp_path, name=name, labels=(0, 1), source_options=False,
        )
        paths = (
            mixed.target.parent / ("first.raw" if raw_first else "first.tif"),
            mixed.target.parent / ("later.tif" if raw_first else "later.raw"),
        )
        for path, pixels in zip(paths, mixed.raw, strict=True):
            if path.suffix == ".raw":
                path.write_bytes(pixels.tobytes())
            else:
                tifffile.imwrite(path, pixels)
        states = tuple(SourceFileState.capture(path) for path in paths)
        execution = SourceExecutionStamp(
            states[0], "tiff_series", 2, 0, members=states,
        )
        with h5py.File(mixed.target, "r+") as handle:
            config = handle["entry/reduction/config"]
            config["source_execution"][()] = json.dumps(
                execution.as_dict(), sort_keys=True, separators=(",", ":")
            )
            if decoder:
                run = json.loads(config["run_configuration"].asstr()[()])
                options = {
                    "selected_file": str(paths[0]), "files": list(map(str, paths)),
                    "pattern": "*", "scan_name": name, "metadata_format": None,
                    "raw_dtype": mixed.raw.dtype.str, "raw_header_skip": 0,
                    "detector_shape": list(mixed.raw.shape[1:]),
                }
                run["source"]["options"] = options
                run["scientific_signature"]["source"]["options"] = copy.deepcopy(options)
                config["run_configuration"][()] = json.dumps(
                    run, sort_keys=True, separators=(",", ":")
                )
            for label, path, state in zip(mixed.labels, paths, states, strict=True):
                source_group = handle[f"entry/frames/frame_{label:04d}/source"]
                source_group["path"][()] = str(path)
                source_group["frame_index"][()] = 0
                source_group.attrs["adapter_id"] = "tiff_series"
                source_group.attrs["file_size"] = state.size
                source_group.attrs["file_mtime_ns"] = state.mtime_ns
                source_group.attrs["frame_count"] = 1
                source_group.attrs["self_contained"] = True
                if "dataset_path" in source_group.attrs:
                    del source_group.attrs["dataset_path"]
        return mixed

    mixed = mixed_fixture("production-mixed-missing", raw_first=False, decoder=False)
    with pytest.raises(ValueError, match="REPLACEMENT_RAW_DECODER_UNRECORDED"):
        module.ReintegratePlan.from_artifact(
            mixed.target, entry="entry", dimension="1d",
            preparation=mixed.preparation,
            expected_target_snapshot=capture_target_snapshot(mixed.target),
            expected_labels=mixed.labels,
        )
    _stub_integrators(monkeypatch)
    for raw_first in (False, True):
        mixed = mixed_fixture(
            f"production-mixed-valid-{raw_first}",
            raw_first=raw_first, decoder=True,
        )
        plan = module.ReintegratePlan.from_artifact(
            mixed.target, entry="entry", dimension="1d",
            preparation=mixed.preparation,
            expected_target_snapshot=capture_target_snapshot(mixed.target),
            expected_labels=mixed.labels,
        )
        assert _run_successor(plan).committed_labels == mixed.labels
def test_hdf_transient_path_replacement_is_refused_before_pixel_decode(
    tmp_path, monkeypatch,
):
    import os
    import shutil

    from xdart.gui.tabs.scattering.contracts import (
        SourceExecutionStamp, SourceFileState,
    )
    module = _module()
    seeded = _seed_existing(tmp_path, name="descriptor-fence")
    source = seeded.source.resolve()
    original = source.read_bytes()
    state = SourceFileState.capture(source)
    execution = SourceExecutionStamp(state, "nexus_hdf5", 10, 0)
    fact = {
        "label": 2, "path": str(source), "frame_index": 2,
        "source_base": "", "snapshot": {
            "adapter_id": "nexus_hdf5", "size": state.size,
            "mtime_ns": state.mtime_ns, "frame_count": 10,
            "dataset_path": "/entry/instrument/detector/data",
            "self_contained": True,
        },
        "source_execution": execution.as_dict(), "append_lineage": None,
        "metadata": {}, "geometry": {}, "background_dependency": None,
    }
    foreign = tmp_path / "foreign-source.h5"
    parked = tmp_path / "parked-source.h5"
    shutil.copy2(source, foreign)
    topology = module._admit_source_topology(
        fact, full_inventory=True)
    real_open = h5py.File
    decodes = []

    def transient_open(path, *args, **kwargs):
        if Path(path) != source:
            return real_open(path, *args, **kwargs)
        os.replace(source, parked)
        os.replace(foreign, source)
        try:
            return real_open(path, *args, **kwargs)
        finally:
            os.replace(source, foreign)
            os.replace(parked, source)

    monkeypatch.setattr(h5py, "File", transient_open)
    real_getitem = h5py.Dataset.__getitem__
    monkeypatch.setattr(
        h5py.Dataset, "__getitem__",
        lambda dataset, key: (
            decodes.append(1)
            or pytest.fail("foreign HDF5 object reached pixel decode")
            if dataset.name == "/entry/instrument/detector/data"
            else real_getitem(dataset, key)
        ),
    )
    with pytest.raises(
        ValueError, match="REPLACEMENT_SOURCE_REVISION_CHANGED",
    ):
        module._source_fact(fact, read=True, topology=topology)
    assert decodes == []
    assert source.read_bytes() == original


def test_hdf_descriptor_mutation_after_decode_is_refused_before_restat(
    tmp_path, monkeypatch,
):
    import os

    from xdart.gui.tabs.scattering.contracts import (
        SourceExecutionStamp, SourceFileState,
    )
    module = _module()
    seeded = _seed_existing(tmp_path, name="descriptor-post-fence")
    source = seeded.source.resolve()
    original = source.read_bytes()
    state = SourceFileState.capture(source)
    execution = SourceExecutionStamp(state, "nexus_hdf5", 10, 0)
    fact = {
        "label": 2, "path": str(source), "frame_index": 2,
        "source_base": "", "snapshot": {
            "adapter_id": "nexus_hdf5", "size": state.size,
            "mtime_ns": state.mtime_ns, "frame_count": 10,
            "dataset_path": "/entry/instrument/detector/data",
            "self_contained": True,
        },
        "source_execution": execution.as_dict(), "append_lineage": None,
        "metadata": {}, "geometry": {}, "background_dependency": None,
    }
    topology = module._admit_source_topology(
        fact, full_inventory=True)
    real_getitem = h5py.Dataset.__getitem__
    real_qualified = module._qualified_fact
    qualified = []

    def mutate_after_read(dataset, key):
        value = real_getitem(dataset, key)
        if dataset.name != "/entry/instrument/detector/data":
            return value
        observed = source.stat()
        os.utime(source, ns=(
            observed.st_atime_ns, observed.st_mtime_ns + 1_000_000,
        ))
        return value

    monkeypatch.setattr(
        h5py.Dataset, "__getitem__", mutate_after_read,
    )
    monkeypatch.setattr(
        module, "_qualified_fact",
        lambda value, topology=None:
            qualified.append(value) or real_qualified(value, topology),
    )
    with pytest.raises(
        ValueError, match="REPLACEMENT_SOURCE_REVISION_CHANGED",
    ):
        module._source_fact(fact, read=True, topology=topology)
    assert qualified == [fact]
    assert source.read_bytes() == original


def _external_storage_case(tmp_path, *, count, name):
    from xdart.gui.tabs.scattering.contracts import (
        ExternalSourceState, SourceExecutionStamp, SourceFileState,
    )

    root = tmp_path / name
    root.mkdir()
    raw_paths = []
    for index in range(count):
        path = root / f"slot_{index:04d}.bin"
        path.write_bytes(bytes((index % 251,)))
        raw_paths.append(path)
    member = root / "member.h5"
    with h5py.File(member, "w") as handle:
        handle.create_group("entry/data").create_dataset(
            "data", shape=(count, 1, 1), dtype=np.uint8,
            external=[(str(path), 0, 1) for path in raw_paths],
        )
    master = root / "master.h5"
    with h5py.File(master, "w") as handle:
        handle.create_group("entry/data")["data_000001"] = (
            h5py.ExternalLink(member.name, "/entry/data/data"))
    master_state = SourceFileState.capture(master)
    member_state = SourceFileState.capture(member)
    execution = SourceExecutionStamp(
        master_state, "nexus_hdf5", count, 0,
        external_members=(ExternalSourceState(
            member_state, "/entry/data/data", 0, count, 0),),
        dependency_files=tuple(
            SourceFileState.capture(path) for path in raw_paths),
    ).as_dict()

    def fact(label):
        return {
            "label": label, "path": master_state.path,
            "frame_index": label, "source_base": "", "snapshot": {
                "adapter_id": "nexus_hdf5", "size": master_state.size,
                "mtime_ns": master_state.mtime_ns, "frame_count": count,
                "dataset_path": "/entry/data/data_000001",
                "self_contained": False,
            },
            "source_execution": execution, "append_lineage": None,
            "metadata": {}, "geometry": {},
            "background_dependency": None,
        }

    return SimpleNamespace(
        master=master, member=member, raw_paths=tuple(raw_paths),
        execution=execution, fact=fact,
    )


def test_external_raw_slot_transient_symlink_retarget_is_refused(
    tmp_path, monkeypatch,
):
    import os

    from xdart.gui.tabs.scattering.contracts import (
        ExternalSourceState, SourceExecutionStamp, SourceFileState,
    )

    module = _module()
    root = tmp_path / "external-slot-binding"
    root.mkdir()
    shape = (3, 4)
    accepted = np.full((1, *shape), 7, dtype=np.uint16)
    foreign_pixels = np.full((1, *shape), 19, dtype=np.uint16)
    accepted_raw = root / "accepted.bin"
    foreign_raw = root / "foreign.bin"
    slot = root / "detector-slot.bin"
    accepted_raw.write_bytes(b"\0" * accepted.nbytes)
    slot.symlink_to(accepted_raw.name)
    member = root / "member.h5"
    with h5py.File(member, "w") as handle:
        dataset = handle.create_group("entry/data").create_dataset(
            "data", shape=accepted.shape, dtype=accepted.dtype,
            external=[(str(slot), 0, h5py.h5f.UNLIMITED)],
        )
        dataset[...] = accepted
    foreign_raw.write_bytes(foreign_pixels.tobytes())
    master = root / "master.h5"
    with h5py.File(master, "w") as handle:
        data = handle.create_group("entry/data")
        data["data_000001"] = h5py.ExternalLink(
            member.name, "/entry/data/data",
        )

    master_state = SourceFileState.capture(master)
    member_state = SourceFileState.capture(member)
    slot_state = SourceFileState.capture(slot)
    execution = SourceExecutionStamp(
        master_state, "nexus_hdf5", 1, 0,
        external_members=(ExternalSourceState(
            member_state, "/entry/data/data", 0, 1, 0,
        ),),
        dependency_files=(slot_state,),
    )
    fact = {
        "label": 0, "path": master_state.path, "frame_index": 0,
        "source_base": "", "snapshot": {
            "adapter_id": "nexus_hdf5", "size": master_state.size,
            "mtime_ns": master_state.mtime_ns, "frame_count": 1,
            "dataset_path": "/entry/data/data_000001",
            "self_contained": False,
        },
        "source_execution": execution.as_dict(), "append_lineage": None,
        "metadata": {}, "geometry": {}, "background_dependency": None,
    }
    real_external_paths = module._hdf_external_paths
    observations = []

    def counted_external_paths(dataset, **kwargs):
        observations.append(1)
        return real_external_paths(dataset, **kwargs)

    monkeypatch.setattr(module, "_hdf_external_paths", counted_external_paths)
    topology = module._admit_source_topology(fact, full_inventory=True)
    real_open = os.open

    def transient_open(path, *args, **kwargs):
        if Path(path) != slot:
            return real_open(path, *args, **kwargs)
        slot.unlink()
        slot.symlink_to(foreign_raw.name)
        try:
            return real_open(path, *args, **kwargs)
        finally:
            slot.unlink()
            slot.symlink_to(accepted_raw.name)

    monkeypatch.setattr(os, "open", transient_open)
    with pytest.raises(
        ValueError, match="REPLACEMENT_SOURCE_REVISION_CHANGED",
    ):
        module._source_fact(fact, read=True, topology=topology)
    assert observations == [1]
    assert slot.resolve(strict=True) == accepted_raw.resolve(strict=True)
    assert accepted_raw.read_bytes() == accepted.tobytes()
    assert foreign_raw.read_bytes() == foreign_pixels.tobytes()


def test_651_external_slots_prebind_only_early_and_late_frame_closures(
    tmp_path, monkeypatch,
):
    import os

    module = _module()
    case = _external_storage_case(
        tmp_path, count=651, name="external-slot-routes")
    sweeps = []
    real_execution_revisions = module._execution_revisions

    def counted_sweep(*args, **kwargs):
        sweeps.append(kwargs.get("full"))
        return real_execution_revisions(*args, **kwargs)

    monkeypatch.setattr(module, "_execution_revisions", counted_sweep)
    topology = module._admit_source_topology(
        case.fact(0), full_inventory=True, selected_labels=(0, 650))
    assert sweeps == [True]
    assert tuple(topology.frame_routes) == (0, 650)
    early = topology.frame_routes[0].hdf
    late = topology.frame_routes[650].hdf
    assert early is not None and late is not None
    assert len(early.storage_slices) == len(late.storage_slices) == 1
    assert early.storage_slices[0].lexical_path == str(case.raw_paths[0])
    assert late.storage_slices[0].lexical_path == str(case.raw_paths[650])
    assert sum(
        len(route.hdf.storage_slices)
        for route in topology.frame_routes.values()
    ) == 2

    revision_paths = []
    raw_opens = []
    real_revision = module._revision
    real_open = os.open

    def counted_revision(path):
        revision_paths.append(str(Path(path)))
        return real_revision(path)

    def counted_open(path, *args, **kwargs):
        if Path(path).name.startswith("slot_"):
            raw_opens.append(str(Path(path)))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(module, "_revision", counted_revision)
    monkeypatch.setattr(os, "open", counted_open)
    for label in (0, 650):
        current = case.fact(label)
        current["source_execution"] = topology.execution
        shape, dtype, _path, image = module._source_fact(
            current, read=True, topology=topology)
        assert shape == (1, 1) and dtype == "|u1"
        assert int(image[0, 0]) == label % 251
    expected_raw = {str(case.raw_paths[0]), str(case.raw_paths[650])}
    assert set(raw_opens) == expected_raw and len(raw_opens) == 2
    observed_raw = {
        value for value in revision_paths
        if Path(value).name.startswith("slot_")
    }
    assert observed_raw == expected_raw
    assert all(
        str(path) not in revision_paths for path in case.raw_paths[1:650])
    assert sweeps == [True]
    module._validate_terminal_topology(topology)
    assert sweeps == [True, True]


def test_external_storage_census_refuses_count_path_overflow_and_overlap(
    tmp_path, monkeypatch,
):
    module = _module()
    owner = tmp_path / "census-owner.h5"
    owner.write_bytes(b"")
    raw = tmp_path / "slot.bin"
    raw.write_bytes(b"\0\0")

    class Properties:
        def __init__(self, count, rows):
            self.count = count
            self.rows = rows
            self.reads = []

        def get_external_count(self):
            return self.count

        def get_external(self, index):
            self.reads.append(index)
            return self.rows[index]

        def close(self):
            pass

    def dataset(properties, shape):
        return SimpleNamespace(
            file=SimpleNamespace(filename=str(owner)), shape=shape,
            dtype=np.dtype("u1"),
            id=SimpleNamespace(get_create_plist=lambda: properties),
        )

    count = Properties(module._MAX_HDF_EXTERNAL_SLOTS + 1, ())
    with pytest.raises(
        ValueError,
        match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    ):
        module._hdf_external_paths(dataset(count, (1,)), total_bytes=1)
    assert count.reads == []

    with monkeypatch.context() as limits:
        limits.setattr(module, "_MAX_HDF_EXTERNAL_PATH_BYTES", 2)
        path = Properties(1, ((b"long", 0, 1),))
        with pytest.raises(
            ValueError,
            match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        ):
            module._hdf_external_paths(dataset(path, (1,)), total_bytes=1)
        assert path.reads == [0]

    overflow = Properties(1, ((
        str(raw).encode(), module._MAX_HDF_EXTERNAL_ADDRESS, 2),))
    with pytest.raises(
        ValueError,
        match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    ):
        module._hdf_external_paths(dataset(overflow, (2,)), total_bytes=2)

    overlap = Properties(2, (
        (str(raw).encode(), 0, 1),
        (str(raw).encode(), 0, 1),
    ))
    with pytest.raises(
        ValueError,
        match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    ):
        module._hdf_external_paths(dataset(overlap, (2,)), total_bytes=2)


def test_external_raw_slot_requires_admitted_physical_extent_before_effect(
    tmp_path, monkeypatch,
):
    from xdart.gui.tabs.scattering.contracts import SourceFileState
    from xrd_tools.reduction import NexusSink

    module = _module()
    exact = _external_storage_case(
        tmp_path, count=1, name="external-slot-exact-size")
    topology = module._admit_source_topology(
        exact.fact(0), full_inventory=True, selected_labels=(0,))
    assert topology.frame_routes[0].hdf.storage_slices[0].size == 1
    assert exact.raw_paths[0].stat().st_size == 1

    short = _external_storage_case(
        tmp_path, count=1, name="external-slot-short-file")
    short.raw_paths[0].write_bytes(b"")
    fact = copy.deepcopy(short.fact(0))
    fact["source_execution"]["dependency_files"] = [
        SourceFileState.capture(short.raw_paths[0]).as_dict()
    ]
    sinks = []
    monkeypatch.setattr(
        NexusSink, "for_existing_replacement",
        classmethod(lambda cls, *args, **kwargs: sinks.append(1)),
    )
    with pytest.raises(
        ValueError,
        match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    ):
        module._admit_source_topology(
            fact, full_inventory=True, selected_labels=(0,))
    assert sinks == []


def test_source_topology_and_static_final_lineage_matrix(tmp_path, monkeypatch):
    import fabio, tifffile
    from xdart.gui.tabs.scattering.contracts import (
        ExternalSourceState, SourceExecutionStamp, SourceFileState,
    )
    module = _module(); root = tmp_path / "raw"; root.mkdir()
    shape = (5, 7); pixels = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    def fact(path, stamp, *, label=0, index=0, dataset=None, contained=True):
        state = SourceFileState.capture(path)
        return {"label": label, "path": str(path), "frame_index": index,
                "source_base": "", "snapshot": {
                    "adapter_id": stamp.adapter_id, "size": state.size,
                    "mtime_ns": state.mtime_ns,
                    "frame_count": 1 if stamp.adapter_id != "nexus_hdf5" else stamp.frame_count,
                    "dataset_path": dataset, "self_contained": contained},
                "source_execution": stamp.as_dict(), "append_lineage": None,
                "metadata": {}, "geometry": {}, "background_dependency": None}
    def assert_pixels(value, expected=pixels, raw_options=None):
        shape_seen, dtype, _path, image = module._source_fact(
            value, raw_options=raw_options, read=True)
        assert shape_seen == shape and dtype == expected.dtype.str
        np.testing.assert_array_equal(image, expected)
    tifffile.imwrite(root / "a.tif", pixels)
    assert module._resolve_source_locator("/abs/a.tif", "/ignored") == Path("/abs/a.tif")
    assert module._resolve_source_locator("raw/a.tif", str(tmp_path)) == tmp_path / "raw/a.tif"
    for locator, base in (("a.tif", ""), ("../a.tif", str(root)), ("a.tif", "relative")):
        with pytest.raises(ValueError, match="REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED"):
            module._resolve_source_locator(locator, base)
    series = []
    for index in range(3):
        path = root / f"series_{index}.tif"
        fabio.tifimage.TifImage(data=pixels + index).write(str(path))
        series.append(SourceFileState.capture(path))
    stamp = SourceExecutionStamp(series[0], "tiff_series", 3, 0, members=tuple(series))
    selected = fact(Path(series[1].path), stamp, label=1)
    selected["snapshot"].update(size=series[1].size, mtime_ns=series[1].mtime_ns)
    revisions, real_revision = [], module._revision
    def counted(path): revisions.append(str(Path(path).resolve())); return real_revision(path)
    monkeypatch.setattr(module, "_revision", counted); assert_pixels(selected, pixels + 1)
    assert set(revisions) == {str(Path(value.path).resolve()) for value in series}
    monkeypatch.setattr(module, "_revision", real_revision)
    injected = root / "injected.tif"
    tifffile.imwrite(injected, pixels + 99)
    selected_path = Path(series[1].path)
    real_open = Path.open
    with monkeypatch.context() as swap:
        swap.setattr(
            Path, "open",
            lambda self, *args, **kwargs:
                real_open(injected, *args, **kwargs)
                if self == selected_path and args and args[0] == "rb"
                else real_open(self, *args, **kwargs),
        )
        with pytest.raises(
            ValueError, match="REPLACEMENT_SOURCE_REVISION_CHANGED",
        ):
            module._source_fact(selected, read=True)
    multi = root / "multi.tif"; tifffile.imwrite(multi, pixels); tifffile.imwrite(multi, pixels + 1, append=True); multi_state = SourceFileState.capture(multi); multi_fact = fact(multi, SourceExecutionStamp(multi_state, "image_file", 1, 0))
    with pytest.raises(ValueError, match="REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"): module._source_fact(multi_fact, read=True)
    for suffix, image_type in (("edf", fabio.edfimage.EdfImage),
                               ("cbf", fabio.cbfimage.CbfImage)):
        path = root / f"single.{suffix}"; image_type(data=pixels).write(str(path))
        state = SourceFileState.capture(path)
        assert_pixels(fact(path, SourceExecutionStamp(state, "image_file", 1, 0)))
    dependency = root / "image-calibration.bin"; dependency.write_bytes(b"accepted"); dependency_state = SourceFileState.capture(dependency); dependent = fact(path, SourceExecutionStamp(state, "image_file", 1, 0, dependency_files=(dependency_state,))); dependency.write_bytes(b"changed dependency")
    with pytest.raises(ValueError, match="REPLACEMENT_SOURCE_REVISION_CHANGED"): module._source_fact(dependent, read=True)
    raw = root / "recorded.raw"; raw.write_bytes(pixels.astype("<i2").tobytes())
    raw_state = SourceFileState.capture(raw)
    raw_fact = fact(raw, SourceExecutionStamp(raw_state, "tiff_series", 1, 0,
                                              members=(raw_state,)))
    assert_pixels(raw_fact, pixels.astype("<i2"), {
        "raw_dtype": "<i2", "raw_header_skip": 0, "detector_shape": shape})
    with pytest.raises(ValueError, match="REPLACEMENT_RAW_DECODER_UNRECORDED"):
        module._source_fact(raw_fact, raw_options={"raw_dtype": "<i2"}, read=True)
    seeded = _seed_existing(tmp_path, name="topology-hdf")
    master_state = SourceFileState.capture(seeded.source)
    hdf_stamp = SourceExecutionStamp(master_state, "nexus_hdf5", 10, 0)
    assert_pixels(fact(seeded.source, hdf_stamp, label=2, index=2,
                       dataset="/entry/instrument/detector/data"), seeded.raw[2])
    from xrd_tools.io.record_writer import _decode_replacement_fact
    growing = _seed_existing(tmp_path, labels=(7, 8, 9), append=True, name="topology-growing")
    with h5py.File(growing.target, "r") as handle: final_fact = module._plain(_decode_replacement_fact(handle, 9))
    assert final_fact["append_lineage"]["epochs"][-1]["source"]["extent"] == 10
    final_fact["source_execution"]["frame_count"] = 3; assert_pixels(final_fact, growing.raw[9])
    for segments in (1, 2):
        members, dependencies, cursor = [], [], 0
        for index in range(segments):
            stop = cursor + index + 2; data_path = root / f"eiger_{segments}_{index}.h5"
            with h5py.File(data_path, "w") as handle:
                group = handle.create_group("entry/data"); data = np.full((stop - cursor, *shape), index, np.uint16)
                group.create_dataset("data", data=data, chunks=(1, *shape))
            members.append(ExternalSourceState(SourceFileState.capture(data_path), "/entry/data/data", cursor, stop, index))
            cursor = stop
        master = root / f"eiger_{segments}_master.h5"
        with h5py.File(master, "w") as handle:
            entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"; data = entry.create_group("data"); data.attrs["NX_class"] = "NXdata"
            for index, member in enumerate(members, 1):
                data[f"data_{index:06d}"] = h5py.ExternalLink(Path(member.file.path).name, member.dataset)
        master_state = SourceFileState.capture(master)
        eiger_stamp = SourceExecutionStamp(master_state, "nexus_hdf5", cursor, 0, external_members=tuple(members), dependency_files=tuple(dependencies))
        selected_index = members[-1].first if segments == 2 else 0
        eiger_fact = fact(master, eiger_stamp, label=selected_index, index=selected_index, dataset="/entry/data/data_000001", contained=False)
        if segments == 2:
            append_source = lambda chosen: {"path": master_state.path, "adapter_id": "nexus_hdf5", "size": master_state.size, "mtime_ns": master_state.mtime_ns, "extent": chosen[-1].stop, "digest": None, "dataset_paths": [f"/entry/data/data_{ordinal + 1:06d}" for ordinal in range(len(chosen))], "image_members": [], "external_members": [{"path": member.file.path, "dataset_path": member.dataset, "size": member.file.size, "mtime_ns": member.file.mtime_ns, "source_start": member.first, "source_stop": member.stop, "ordinal": member.epoch} for member in chosen], "generation": 0}
            historical = fact(master, eiger_stamp, label=0, index=0, dataset="/entry/data/data_000001", contained=False); historical["snapshot"]["frame_count"] = members[0].stop; historical["source_execution"]["frame_count"] = members[0].stop; historical["source_execution"]["external_members"] = historical["source_execution"]["external_members"][:1]; historical["source_execution"]["dependency_files"] = []
            historical["append_lineage"] = {"epochs": [{"source": append_source(members[:1]), "labels": [0, 1]}, {"source": append_source(members), "labels": [2, 3, 4]}]}
            assert_pixels(historical, np.zeros(shape, np.uint16))
            appended = copy.deepcopy(historical)
            appended.update(label=3, frame_index=3)
            appended["snapshot"]["frame_count"] = cursor
            topology = module._admit_source_topology(
                historical, full_inventory=True, selected_labels=(0, 3))
            appended["source_execution"] = topology.execution
            appended["append_lineage"] = topology.lineage
            historical["source_execution"] = topology.execution
            historical["append_lineage"] = topology.lineage
            assert module._source_fact(
                historical, read=True, topology=topology)[3][0, 0] == 0
            assert module._source_fact(
                appended, read=True, topology=topology)[3][0, 0] == 1
            corrupt = copy.deepcopy(module._plain(topology.lineage))
            corrupt["epochs"][1]["labels"] = [3, 4]
            refused = copy.deepcopy(appended)
            refused["source_execution"] = module._plain(topology.execution)
            refused["append_lineage"] = corrupt
            with pytest.raises(
                ValueError, match="REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
            ):
                module._admit_source_topology(
                    refused, full_inventory=True, selected_labels=(0, 3))
            legacy = copy.deepcopy(module._plain(historical)); legacy["append_lineage"]["epochs"][-1]["source"]["dataset_paths"] = []
            with pytest.raises(ValueError, match="REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"):
                module._source_fact(legacy, read=True)
            mismatch = copy.deepcopy(module._plain(historical)); mismatch["append_lineage"]["epochs"][-1]["source"]["dataset_paths"] = ["/entry/data/not_the_authenticated_links"]
            with pytest.raises(ValueError, match="REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"): module._source_fact(mismatch, read=True)
            owner_iterations = []
            real_group_iter = h5py.Group.__iter__
            def guarded_group_iter(group):
                if group.name == "/entry/data" and Path(group.file.filename) == master:
                    owner_iterations.append(1)
                    raise AssertionError("oversized owner was iterated")
                return real_group_iter(group)
            with monkeypatch.context() as limits:
                limits.setattr(module, "_MAX_HDF_OWNER_CHILDREN", 1)
                limits.setattr(h5py.Group, "__iter__", guarded_group_iter)
                with pytest.raises(
                    ValueError,
                    match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
                ):
                    module._admit_source_topology(
                        eiger_fact, full_inventory=True)
            assert owner_iterations == []
        got = module._source_fact(eiger_fact, read=True)
        assert got[:2] == (shape, "<u2") and int(got[3][0, 0]) == segments - 1
        if segments == 2:
            bad = copy.deepcopy(eiger_fact)
            bad["source_execution"]["external_members"][1]["first"] = 1
            with pytest.raises(ValueError, match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"):
                module._source_fact(bad, read=True)
    linked = _seed_existing(tmp_path, name="external-entry"); external = tmp_path / "external-entry.nxs"
    with h5py.File(linked.target, "r+") as local, h5py.File(external, "w") as remote:
        local.copy("entry", remote, name="entry"); del local["entry"]; local["entry"] = h5py.ExternalLink(str(external), "/entry")
    unchanged = linked.target.read_bytes()
    with pytest.raises(ValueError, match="replacement target is not a current xdart .nexus record"): module.ReintegratePlan.from_artifact(linked.target, entry="entry", dimension="1d", preparation=linked.preparation)
    assert linked.target.read_bytes() == unchanged
    for index, path in enumerate(("entry/frames", "entry/frames/frame_0002", "entry/frames/frame_0002/source", "entry/frames/frame_0002/source/path", "entry/frames/frame_0002/source/frame_index", "entry/reduction", "entry/reduction/config", "entry/reduction/config/source_execution")):
        broken = _seed_existing(tmp_path, name=f"foreign-{index}"); parent, leaf = path.rsplit("/", 1)
        with h5py.File(broken.target, "r+") as handle:
            del handle[path]; handle[parent][leaf] = h5py.SoftLink("/missing") if index % 2 else h5py.ExternalLink("missing.nxs", "/missing")
        snapshot = broken.target.read_bytes()
        with pytest.raises(ValueError, match="REPLACEMENT_"): module.ReintegratePlan.from_artifact(broken.target, entry="entry", dimension="1d", preparation=broken.preparation)
        assert broken.target.read_bytes() == snapshot
    preserved = _seed_existing(tmp_path, name="foreign-preserved")
    with h5py.File(preserved.target, "r+") as handle:
        scan = handle["entry/scan_data"]; del scan["theta"]; scan["theta"] = h5py.ExternalLink("missing.nxs", "/theta")
        notes = handle["entry"].create_group("operator_notes"); notes["foreign"] = h5py.SoftLink("/missing"); notes["self"] = notes
        handle["entry/integrated_2d"]["foreign"] = h5py.ExternalLink("missing.nxs", "/foreign")
    signature = _preserved_signature(preserved.target)
    with pytest.raises(
        ValueError, match="replacement target is not a current xdart .nexus record",
    ):
        module.ReintegratePlan.from_artifact(
            preserved.target, entry="entry", dimension="1d",
            preparation=preserved.preparation,
        )
    assert _preserved_signature(preserved.target) == signature
    with pytest.raises(ValueError, match="REPLACEMENT_SOURCE_FORMAT_UNSUPPORTED"):
        module._source_route(Path("processed.txt"))


def test_external_link_owner_ancestor_refuses_before_effect(
    tmp_path, monkeypatch,
):
    from xdart.gui.tabs.scattering.contracts import (
        ExternalSourceState, SourceExecutionStamp, SourceFileState,
    )
    from xrd_tools.reduction import NexusSink

    module = _module()
    seeded = _seed_existing(tmp_path, name="foreign-source-owner")
    root = seeded.target.parent
    member = root / "member.h5"
    with h5py.File(member, "w") as handle:
        handle.create_group("entry/data").create_dataset(
            "data", data=np.zeros((10, 5, 7), dtype=np.uint16))
    remote_owner = root / "remote-owner.h5"
    with h5py.File(remote_owner, "w") as handle:
        owner = handle.create_group("entry/data")
        owner["data_000001"] = h5py.ExternalLink(
            member.name, "/entry/data/data")
    master = root / "master.h5"
    with h5py.File(master, "w") as handle:
        handle["entry"] = h5py.ExternalLink(remote_owner.name, "/entry")
    master_state = SourceFileState.capture(master)
    member_state = SourceFileState.capture(member)
    execution = SourceExecutionStamp(
        master_state, "nexus_hdf5", 10, 0,
        external_members=(ExternalSourceState(
            member_state, "/entry/data/data", 0, 10, 0),),
    )
    encoded = json.dumps(
        execution.as_dict(), sort_keys=True, separators=(",", ":"))
    with h5py.File(seeded.target, "r+") as handle:
        handle["entry/reduction/config/source_execution"][()] = encoded
        for label in seeded.labels:
            source = handle[f"entry/frames/frame_{label:04d}/source"]
            source["path"][()] = str(master)
            source["frame_index"][()] = label
            for key, value in {
                "adapter_id": "nexus_hdf5",
                "file_size": master_state.size,
                "file_mtime_ns": master_state.mtime_ns,
                "frame_count": 10,
                "dataset_path": "/entry/data/data_000001",
                "self_contained": False,
            }.items():
                source.attrs.modify(key, value)

    before = seeded.target.read_bytes()
    dereferences = []
    publications = []
    real_get = h5py.Group.get

    def guarded_get(group, name, default=None, getclass=False, getlink=False):
        if (Path(group.file.filename) == master and group.name == "/"
                and name == "entry" and not getlink):
            dereferences.append(name)
            raise AssertionError("foreign owner ancestor was dereferenced")
        return real_get(group, name, default, getclass, getlink)

    monkeypatch.setattr(h5py.Group, "get", guarded_get)
    monkeypatch.setattr(
        NexusSink, "for_existing_replacement",
        classmethod(lambda cls, *args, **kwargs: publications.append(1)),
    )
    with pytest.raises(
        ValueError,
        match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    ):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d",
            preparation=seeded.preparation,
        )
    assert dereferences == []
    assert publications == []
    assert seeded.target.read_bytes() == before


@pytest.mark.parametrize("kind", ("external", "soft"))
def test_hdf_owner_link_value_ceiling_refuses_before_value_materialization(
    tmp_path, monkeypatch, kind,
):
    from xrd_tools.reduction import NexusSink

    module = _module()
    source = tmp_path / f"oversized-{kind}.h5"
    with h5py.File(source, "w") as handle:
        owner = handle.create_group("owner")
        owner["child"] = (
            h5py.ExternalLink("member-with-long-name.h5", "/remote/path")
            if kind == "external" else h5py.SoftLink("/long/soft/target"))
    value_reads = []
    sinks = []
    real_get = h5py.Group.get

    def guarded_get(group, name, default=None, getclass=False, getlink=False):
        if group.name == "/owner" and name == "child":
            value_reads.append((getlink, getclass))
            raise AssertionError("oversized link value was materialized")
        return real_get(group, name, default, getclass, getlink)

    monkeypatch.setattr(h5py.Group, "get", guarded_get)
    monkeypatch.setattr(
        NexusSink, "for_existing_replacement",
        classmethod(lambda cls, *args, **kwargs: sinks.append(1)),
    )
    monkeypatch.setattr(module, "_MAX_HDF_LINK_VALUE_BYTES", 1)
    with h5py.File(source, "r") as handle:
        with pytest.raises(
            ValueError,
            match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        ):
            module._bounded_hdf_owner_links(
                handle, "/owner", source, {})
    assert value_reads == [] and sinks == []


def test_terminal_owner_census_refuses_same_names_with_changed_link_signature(
    tmp_path, monkeypatch,
):
    module = _module()
    case = _external_storage_case(
        tmp_path, count=2, name="terminal-owner-signature")
    topology = module._admit_source_topology(
        case.fact(0), full_inventory=True, selected_labels=(0, 1))
    assert topology.external_signature
    frozen_inventory = module._execution_revisions(
        topology.execution, topology.final_source, validated=True, full=True)
    changed = tuple(
        link._replace(
            lexical_filename="changed-member.h5",
            resolved_target=str(tmp_path / "changed-member.h5"),
            remote_path="/changed/remote/path",
        ) if link.kind == "external" else link
        for link in topology.external_signature
    )
    assert tuple(link.name for link in changed) == tuple(
        link.name for link in topology.external_signature)
    assert changed != topology.external_signature
    monkeypatch.setattr(
        module, "_execution_revisions",
        lambda *args, **kwargs: frozen_inventory,
    )
    monkeypatch.setattr(
        module, "_hdf_owner_census",
        lambda *args, **kwargs: (changed, ()),
    )
    with pytest.raises(
        ValueError,
        match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    ):
        module._validate_terminal_topology(topology)


def test_external_member_is_opened_and_fenced_without_master_selector(
    tmp_path, monkeypatch,
):
    module = _module()
    case = _external_storage_case(
        tmp_path, count=2, name="direct-external-member")
    master_selector_reads = []
    member_events = []
    real_getitem = h5py.Group.__getitem__
    real_fence = module._hdf_handle_revision
    real_local = module._local_hdf_dataset

    def guarded_getitem(group, key):
        if (Path(group.file.filename).resolve() == case.master.resolve()
                and key == "/entry/data/data_000001"):
            master_selector_reads.append(key)
            raise AssertionError("master selector was dereferenced")
        return real_getitem(group, key)

    def fenced(handle, path, revisions):
        result = real_fence(handle, path, revisions)
        if Path(handle.filename).resolve() == case.member.resolve():
            member_events.append("fence")
        return result

    def local(handle, path):
        if Path(handle.filename).resolve() == case.member.resolve():
            assert member_events and member_events[-1] == "fence"
            member_events.append("local")
        return real_local(handle, path)

    monkeypatch.setattr(h5py.Group, "__getitem__", guarded_getitem)
    monkeypatch.setattr(module, "_hdf_handle_revision", fenced)
    monkeypatch.setattr(module, "_local_hdf_dataset", local)
    topology = module._admit_source_topology(
        case.fact(0), full_inventory=True, selected_labels=(0, 1))
    current = case.fact(1)
    current["source_execution"] = topology.execution
    assert module._source_fact(
        current, read=True, topology=topology)[3][0, 0] == 1
    assert master_selector_reads == []
    assert member_events.count("local") == 2
    assert member_events == [
        "fence", "local", "fence", "fence", "local", "fence",
    ]


@pytest.mark.parametrize(
    "hostile", ("soft_ancestor", "soft_leaf", "external_leaf"),
)
def test_external_member_refuses_nonlocal_remote_route_before_dereference(
    tmp_path, monkeypatch, hostile,
):
    from xdart.gui.tabs.scattering.contracts import (
        ExternalSourceState, SourceExecutionStamp, SourceFileState,
    )
    from xrd_tools.reduction import NexusSink

    module = _module()
    root = tmp_path / hostile
    root.mkdir()
    third = root / "third.h5"
    with h5py.File(third, "w") as handle:
        handle.create_dataset("payload", data=np.zeros((1, 1, 1), np.uint8))
    member = root / "member.h5"
    with h5py.File(member, "w") as handle:
        if hostile == "soft_ancestor":
            handle.create_group("real/data").create_dataset(
                "data", data=np.zeros((1, 1, 1), np.uint8))
            handle["entry"] = h5py.SoftLink("/real")
        else:
            data = handle.create_group("entry/data")
            if hostile == "soft_leaf":
                handle.create_dataset(
                    "real", data=np.zeros((1, 1, 1), np.uint8))
                data["data"] = h5py.SoftLink("/real")
            else:
                data["data"] = h5py.ExternalLink(third.name, "/payload")
    master = root / "master.h5"
    with h5py.File(master, "w") as handle:
        handle.create_group("entry/data")["data_000001"] = (
            h5py.ExternalLink(member.name, "/entry/data/data"))
    master_state = SourceFileState.capture(master)
    member_state = SourceFileState.capture(member)
    execution = SourceExecutionStamp(
        master_state, "nexus_hdf5", 1, 0,
        external_members=(ExternalSourceState(
            member_state, "/entry/data/data", 0, 1, 0),),
    ).as_dict()
    fact = {
        "label": 0, "path": master_state.path, "frame_index": 0,
        "source_base": "", "snapshot": {
            "adapter_id": "nexus_hdf5", "size": master_state.size,
            "mtime_ns": master_state.mtime_ns, "frame_count": 1,
            "dataset_path": "/entry/data/data_000001",
            "self_contained": False,
        },
        "source_execution": execution, "append_lineage": None,
        "metadata": {}, "geometry": {}, "background_dependency": None,
    }
    forbidden = (("/", "entry") if hostile == "soft_ancestor"
                 else ("/entry/data", "data"))
    dereferences = []
    sinks = []
    real_get = h5py.Group.get

    def guarded_get(group, name, default=None, getclass=False, getlink=False):
        if (Path(group.file.filename).resolve() == member.resolve()
                and (group.name, name) == forbidden and not getlink):
            dereferences.append(name)
            raise AssertionError("foreign member route was dereferenced")
        return real_get(group, name, default, getclass, getlink)

    monkeypatch.setattr(h5py.Group, "get", guarded_get)
    monkeypatch.setattr(
        NexusSink, "for_existing_replacement",
        classmethod(lambda cls, *args, **kwargs: sinks.append(1)),
    )
    with pytest.raises(
        ValueError,
        match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    ):
        module._admit_source_topology(fact, full_inventory=True)
    assert dereferences == [] and sinks == []


def test_selected_inventory_rows_is_one_linear_merge():
    module = _module()

    class Rows(tuple):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

        def index(self, *_args, **_kwargs):
            raise AssertionError("quadratic row lookup was used")

    rows = Rows(range(0, 20_000, 2))
    assert module._selected_inventory_rows(
        rows, (0, 2_000, 10_000, 19_998), "bad rows",
    ) == (0, 1_000, 5_000, 9_999)
    assert rows.iterations == 1
    for malformed in ((0, 2, 2, 4), (0, 4, 2, 6)):
        with pytest.raises(ValueError, match="bad rows"):
            module._selected_inventory_rows(
                malformed, (0, 2), "bad rows")


def test_651_frame_routes_use_two_whole_inventory_sweeps(
    tmp_path, monkeypatch,
):
    from xdart.gui.tabs.scattering.contracts import (
        SourceExecutionStamp, SourceFileState,
    )

    module = _module()
    root = tmp_path / "route-census"
    root.mkdir()
    states = []
    for label in range(651):
        path = root / f"frame_{label:04d}.tif"
        path.write_bytes(b"")
        states.append(SourceFileState.capture(path))
    execution = SourceExecutionStamp(
        states[0], "tiff_series", 651, 0, members=tuple(states),
    ).as_dict()

    def fact(label):
        state = states[label]
        return {
            "label": label, "path": state.path, "frame_index": 0,
            "source_base": "", "snapshot": {
                "adapter_id": "tiff_series", "size": state.size,
                "mtime_ns": state.mtime_ns, "frame_count": 1,
                "dataset_path": None, "self_contained": True,
            },
            "source_execution": execution, "append_lineage": None,
            "metadata": {}, "geometry": {},
            "background_dependency": None,
        }

    sweeps = []
    real_execution_revisions = module._execution_revisions

    def counted(*args, **kwargs):
        sweeps.append(kwargs.get("full"))
        return real_execution_revisions(*args, **kwargs)

    monkeypatch.setattr(module, "_execution_revisions", counted)
    topology = module._admit_source_topology(
        fact(0), full_inventory=True, selected_labels=tuple(range(651)))
    assert sweeps == [True]
    for label in range(651):
        current = fact(label)
        current["source_execution"] = topology.execution
        before = module._qualified_fact(current, topology)[2]
        after = module._qualified_fact(current, topology)[2]
        assert tuple(before) == tuple(after)
        assert len(before) == 1
    assert sweeps == [True]
    module._validate_terminal_topology(topology)
    assert sweeps == [True, True]


def test_later_member_layout_mismatch_refuses_before_pixel_decode(
    tmp_path, monkeypatch,
):
    import tifffile
    import xrd_tools.io.image as image_module
    from xdart.gui.tabs.scattering.contracts import (
        SourceExecutionStamp, SourceFileState,
    )

    expected = tmp_path / "first.tif"
    mismatched = tmp_path / "later.tif"
    tifffile.imwrite(expected, np.zeros((5, 7), dtype=np.uint16))
    tifffile.imwrite(mismatched, np.zeros((3, 4), dtype=np.uint16))
    states = tuple(SourceFileState.capture(path) for path in (
        expected, mismatched,
    ))
    execution = SourceExecutionStamp(
        states[0], "tiff_series", 2, 0, members=states,
    )
    later = states[1]
    fact = {
        "label": 1, "path": later.path, "frame_index": 0,
        "source_base": "", "snapshot": {
            "adapter_id": "tiff_series", "size": later.size,
            "mtime_ns": later.mtime_ns, "frame_count": 1,
            "dataset_path": None, "self_contained": True,
        },
        "source_execution": execution.as_dict(), "append_lineage": None,
        "metadata": {}, "geometry": {}, "background_dependency": None,
    }
    decodes = []
    monkeypatch.setattr(
        image_module, "read_image",
        lambda *_a, **_k: decodes.append(1) or pytest.fail(
            "mismatched member reached pixel decoder"
        ),
    )

    with pytest.raises(
        ValueError, match="REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    ):
        _module()._source_fact(
            fact, read=True, expected_shape=(5, 7), expected_dtype="<u2",
        )
    assert decodes == []
def test_shared_science_dimension_audit_and_untouched_manifest(tmp_path, monkeypatch):
    from xrd_tools.core.provenance import read_provenance_from_handle
    from xrd_tools.io.append import AppendDisposition, qualify_append
    from xrd_tools.io.nexus_record import read_background_dependency
    module = _module(); seeded = _seed_existing(tmp_path, labels=(0, 1, 2), append=True, name="append")
    before = _preserved_signature(seeded.target)
    with h5py.File(seeded.target, "r") as handle:
        run = read_provenance_from_handle(handle)["config"]["run_configuration"]
        append_before = handle["entry/reduction/config/append_lineage"][()]
        append_attrs = dict(handle["entry/reduction/config/append_lineage"].attrs)
    requested = seeded.preparation["requested_shared_science"]
    assert module._validated_shared_science(run, requested) == requested
    audit = module._dimension_audit(
        dimension="1d", operation_identity="c" * 64,
        science_identity="d" * 64, acquisition_fingerprint=run["fingerprint"],
        requested_shared_science=requested,
        selected_plan=seeded.preparation["selected_plan"], append_lineage=append_before,
    )
    assert set(audit) == {
        "schema_version", "operation", "dimension", "operation_identity",
        "science_identity", "acquisition_fingerprint",
        "shared_science_fingerprint", "selected_plan", "selected_gi_mode",
        "append_lineage_action", "append_lineage_sha256",
    }
    assert module._audit_identity(audit) == hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for branch in ("gi", "threshold", "accepted_scientific_assets", "geometry", "background"):
        changed = copy.deepcopy(seeded.preparation["requested_shared_science"])
        changed[branch] = {"different": True}
        with pytest.raises(ValueError): module._validated_shared_science(run, changed)
    broken = copy.deepcopy(run); broken["scientific_signature"]["threshold"] = {"different": True}
    with pytest.raises(ValueError, match="duplicated scientific signature"):
        module._validated_shared_science(broken)
    _stub_integrators(monkeypatch)
    plan = module.ReintegratePlan.from_artifact(seeded.target, entry="entry", dimension="1d", preparation=seeded.preparation)
    result = _run_successor(plan)
    assert _preserved_signature(seeded.target) == before
    with h5py.File(seeded.target, "r") as handle:
        config = handle["entry/reduction/config"]
        assert config["append_lineage"][()] == append_before
        assert dict(config["append_lineage"].attrs) == append_attrs
    with h5py.File(result.output_artifact, "r") as handle:
        config = handle["entry/reduction/config"]
        stored = json.loads(config["dimension_replacement_1d"].asstr()[()])
    assert set(stored) == set(audit) and module._audit_identity(stored) == result.audit_identity
    decision = qualify_append(seeded.target, seeded.append_intent)
    assert (decision.disposition, decision.reason) == (
        AppendDisposition.SKIP, "exact source already committed",
    )

    active = _seed_existing(tmp_path, name="active-background", background=True)
    with h5py.File(active.target, "r") as handle: pairs = tuple(read_background_dependency(handle[f"entry/frames/frame_{label:04d}"]) for label in active.labels)
    active_plan = module.ReintegratePlan.from_artifact(active.target, entry="entry", dimension="1d", preparation=active.preparation)
    assert active_plan.requested_shared_science["background"]["mode"] == "Single BG File" and _run_successor(active_plan).disposition == "COMMITTED"
    with h5py.File(active.target, "r") as handle: assert pairs == tuple(read_background_dependency(handle[f"entry/frames/frame_{label:04d}"]) for label in active.labels)
def test_stop_before_and_after_writes_roll_back_without_prefix(tmp_path, monkeypatch):
    from pathlib import Path
    from xrd_tools.reduction import NexusSink, run_reintegrate_successor
    module = _module()
    for name in ("pre", "qualify"):
        first = _seed_existing(tmp_path, name=f"stop-{name}"); before = first.target.read_bytes(); plan = module.ReintegratePlan.from_artifact(first.target, entry="entry", dimension="1d", preparation=first.preparation); successor = _successor_from_plan(plan); token = threading.Event(); token.set() if name == "pre" else None
        def cancel_qualify(value): token.set() if name == "qualify" and value.stage == "qualify" else None
        result = run_reintegrate_successor(successor, cancel_token=token, progress_cb=cancel_qualify)
        assert result.disposition == "ABORTED" and result.commit_identity is None and first.target.read_bytes() == before
        assert not Path(successor.output_artifact).exists()
    second = _seed_existing(tmp_path, name="stop-after"); before = second.target.read_bytes(); plan = module.ReintegratePlan.from_artifact(second.target, entry="entry", dimension="1d", preparation=second.preparation); successor = _successor_from_plan(plan); token = threading.Event(); real_write = NexusSink.write; writes = []
    def stop_after_one(self, *args, **kwargs): value = real_write(self, *args, **kwargs); writes.append(args[0].index); token.set(); return value
    with monkeypatch.context() as patch: patch.setattr(NexusSink, "write", stop_after_one); _stub_integrators(patch); result = run_reintegrate_successor(successor, cancel_token=token)
    assert result.disposition == "ABORTED" and result.commit_identity is None and writes == [2] and second.target.read_bytes() == before
    assert not Path(successor.output_artifact).exists()

def test_post_flush_terminal_topology_failure_rolls_back_without_commit(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import run_reintegrate_successor
    from xrd_tools.session.scan_session import ScanSession

    module = _module()
    seeded = _seed_existing(tmp_path, name="post-flush-topology-rollback")
    before = seeded.target.read_bytes()
    plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d",
        preparation=seeded.preparation)
    successor = _successor_from_plan(plan)
    _stub_integrators(monkeypatch)
    order = []
    real_flush = ScanSession.flush

    def flush(session, *args, **kwargs):
        result = real_flush(session, *args, **kwargs)
        order.append("flush")
        return result

    def fail_terminal(*_args, **_kwargs):
        assert order == ["flush"]
        order.append("terminal")
        raise ValueError("injected terminal topology failure")

    monkeypatch.setattr(ScanSession, "flush", flush)
    monkeypatch.setattr(module, "_validate_terminal_topology", fail_terminal)
    with pytest.raises(ValueError, match="injected terminal topology failure"):
        run_reintegrate_successor(successor)
    assert order == ["flush", "terminal"]
    assert seeded.target.read_bytes() == before
    assert not Path(successor.output_artifact).exists()



def test_v1_science_audit_is_readmitted_by_v3_plan(tmp_path, monkeypatch):
    module = _module()
    seeded = _seed_existing(tmp_path, name="v1-science-v3-plan")
    plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d",
        preparation=seeded.preparation)
    expected_science = module._digest({
        "api_version": 1, "dimension": plan.dimension,
        "selected_plan": plan.selected_plan,
        "requested_shared_science": plan.requested_shared_science,
    })
    assert plan.api_version == 3
    assert plan.science_identity == expected_science
    _stub_integrators(monkeypatch)
    result = _run_successor(plan)
    assert result.disposition == "COMMITTED"
    with h5py.File(result.output_artifact, "r") as handle:
        audit = json.loads(handle[
            "entry/reduction/config/dimension_replacement_1d"
        ].asstr()[()])
    assert audit["schema_version"] == 1
    assert audit["science_identity"] == expected_science
    replay = module.ReintegratePlan.from_artifact(
        result.output_artifact, entry="entry", dimension="1d",
        preparation=seeded.preparation)
    assert replay.api_version == 3
    assert replay.science_identity == expected_science


def test_plan_preparation_recipe_roundtrip_share_one_runner_and_writer(tmp_path, monkeypatch):
    from xrd_tools.reduction import GIMode
    module = _module()
    seeded = _seed_existing(tmp_path, name="recipe")
    plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d", preparation=seeded.preparation)
    recipe = plan.as_recipe(); values = recipe["plan"]; session = values["session_policy"]; allocation = session["allocation"]
    assert set(recipe) == {"schema", "version", "plan"} and recipe["schema"] == "xrd_tools.reintegrate.plan"
    assert recipe["version"] == values["api_version"] == plan.api_version == 3
    plan_keys = {"api_version", "target", "entry", "source_root", "expected_target_snapshot", "dimension", "labels", "detector_shape", "native_dtype", "selected_plan", "requested_shared_science", "gi_bootstrap_incidence", "retained_mask_bytes", "mask_decode_bytes", "session_policy", "rollback_policy", "science_identity", "operation_identity"}
    requirement_keys = {"height", "width", "native_itemsize", "background_bytes", "modes_1d", "modes_2d", "npt_1d", "npt_rad", "npt_azim", "sigma_1d", "sigma_2d", "resolver_background_bytes", "worker_background_bytes", "background_binding_bytes"}
    count_keys = {"workers", "reduction_inflight", "queue_depth", "owner_block_bytes", "staging_items", "record_heavy_items", "publication_heavy_items", "thumbnail_items", "record_items", "publication_items"}; category_keys = {"source_native", "staging", "records", "publication", "worker"}
    nodes = ((values, plan_keys), (values["expected_target_snapshot"], {"exists", "size", "mtime_ns", "device", "inode", "digest"}), (session, {"flush", "allocation"}), (session["flush"], {"interval", "cap", "margin"}), (allocation, {"requirements", "envelope_bytes", "counts", "categories", "minimum_bytes", "floor_bytes", "assigned_bytes", "origin", "oversize_excess_bytes"}), (allocation["requirements"], requirement_keys), (allocation["counts"], count_keys), (allocation["categories"], category_keys), (values["selected_plan"], {"version", "dimension", "bai_args", "gi_mode"}), (values["requested_shared_science"], {"version", "gi", "threshold", "poni_values", "accepted_scientific_assets", "geometry", "background"}))
    assert all(set(value) == keys for value, keys in nodes) and session["flush"] == {"interval": 8, "cap": 64, "margin": 8}
    from xrd_tools.io.schema import GI_MODE_KEYS_1D, GI_MODE_KEYS_2D
    assert set(GI_MODE_KEYS_1D) == {"q_total", "q_ip", "q_oop", "exit_angle", "chi_gi"} and set(GI_MODE_KEYS_2D) == {"qip_qoop", "q_chi", "exit_angles"}
    explicit = copy.deepcopy(seeded.preparation); explicit["resource_policy"] = {"version": 1, "kind": "explicit", "allocation": copy.deepcopy(allocation)}
    explicit_plan = module.ReintegratePlan.from_artifact(seeded.target, entry="entry", dimension="1d", preparation=explicit)
    assert explicit_plan.resource_allocation == plan.resource_allocation and explicit_plan.operation_identity == plan.operation_identity and set(seeded.preparation["resource_policy"]) == {"version", "kind", "envelope_bytes", "requests"}
    named_seed = _seed_existing(tmp_path, gi=GIMode(incidence_motor="theta"), name="recipe-bootstrap"); named_plan = module.ReintegratePlan.from_artifact(named_seed.target, entry="entry", dimension="1d", preparation=named_seed.preparation)
    facts = (named_plan.target, named_plan.entry, named_plan.source_root, named_plan.dimension, named_plan.labels, named_plan.detector_shape, named_plan.native_dtype, module._plain(named_plan.selected_plan), module._plain(named_plan.requested_shared_science)); first = module._make_plan(*facts, named_plan.gi_bootstrap_incidence, named_plan.retained_mask_bytes, named_plan.mask_decode_bytes, named_plan.session_policy, snapshot=named_plan.expected_target_snapshot); second = module._make_plan(*facts, .3, named_plan.retained_mask_bytes, named_plan.mask_decode_bytes, named_plan.session_policy, snapshot=named_plan.expected_target_snapshot)
    assert first.science_identity == second.science_identity == named_plan.science_identity and first.operation_identity == named_plan.operation_identity != second.operation_identity
    with monkeypatch.context() as patch:
        patch.setattr(module, "capture_target_snapshot", lambda *_: pytest.fail("recipe touched artifact"))
        patch.setattr(module, "_inspect_artifact", lambda *_: pytest.fail("recipe inspected artifact"))
        patch.setattr(module, "_prepare_gi_scouts", lambda *_: pytest.fail("recipe scouted"))
        legacy = copy.deepcopy(recipe)
        legacy["version"] = legacy["plan"]["api_version"] = 1
        legacy["plan"].pop("retained_mask_bytes")
        legacy["plan"].pop("mask_decode_bytes")
        with pytest.raises(ValueError, match="recipe schema/version"):
            module.ReintegratePlan.from_recipe(legacy)
        replay = module.ReintegratePlan.from_recipe(copy.deepcopy(recipe))
    assert replay == plan and replay.resource_allocation is replay.session_policy.allocation
    recipe["plan"]["labels"][0] = 999
    assert replay.labels == seeded.labels
    with pytest.raises(FrozenInstanceError): replay.target = "changed"
    with pytest.raises(TypeError): replay.selected_plan["bai_args"]["npt"] = 999
    for cls in (module.ReintegratePlan,):
        with pytest.raises(TypeError, match="factory-constructed"): cls()
    cases = (("science_identity", None, "SCIENCE_IDENTITY|version"), ("operation_identity", None, "OPERATION_IDENTITY|version"), ("native_dtype", "<f1", "plan facts"), ("dimension", [], "preparation version"), ("labels", tuple(seeded.labels), "noncanonical JSON"), ("entry", "other", "OPERATION_IDENTITY"))
    for key, value, message in cases:
        bad = plan.as_recipe(); bad["plan"][key] = value
        with pytest.raises(ValueError, match=message): module.ReintegratePlan.from_recipe(bad)
    bad = plan.as_recipe(); bad["plan"]["extra"] = 1
    with pytest.raises(ValueError, match="keyset"): module.ReintegratePlan.from_recipe(bad)
    for section in ("requirements", "counts", "categories"):
        for key in allocation[section]:
            bad = plan.as_recipe(); bad["plan"]["session_policy"]["allocation"][section][key] += 1
            with pytest.raises((TypeError, ValueError)): module.ReintegratePlan.from_recipe(bad)
    for key in ("envelope_bytes", "minimum_bytes", "floor_bytes", "assigned_bytes", "oversize_excess_bytes"):
        bad = plan.as_recipe(); bad["plan"]["session_policy"]["allocation"][key] += 1
        with pytest.raises((TypeError, ValueError)): module.ReintegratePlan.from_recipe(bad)
    for path in (("expected_target_snapshot", "size"), ("session_policy", "allocation", "requirements", "height"), ("session_policy", "allocation", "counts", "workers"), ("session_policy", "allocation", "assigned_bytes")):
        bad = plan.as_recipe(); node = bad["plan"]
        for key in path[:-1]: node = node[key]
        node[path[-1]] = True
        with pytest.raises((TypeError, ValueError)): module.ReintegratePlan.from_recipe(bad)
    for value in (b"x", np.array([1]), lambda: None, float("nan"), {1: "x"}):
        bad = plan.as_recipe(); bad["plan"]["selected_plan"]["bai_args"]["method"] = value
        with pytest.raises(ValueError): module.ReintegratePlan.from_recipe(bad)
    for malformed in ({"version": 1, "kind": "ambient"}, {"version": 1, "kind": "explicit", "allocation": {}}):
        bad = copy.deepcopy(seeded.preparation); bad["resource_policy"] = malformed
        with pytest.raises((TypeError, ValueError)): module.ReintegratePlan.from_artifact(seeded.target, entry="entry", dimension="1d", preparation=bad)
def test_descriptor_allocation_background_roots_and_pre_effect_refusal(tmp_path, monkeypatch):
    from xrd_tools.reduction import MemorySink, run_reintegrate_successor; from xrd_tools.session.scan_session import ScanSession
    module = _module()
    pixels = 5 * 7
    expected = {"None": (0, 0, 0, 0), "Single BG File": (8 * pixels, 8 * pixels, 8 * pixels, 64 << 20), "Series Average": (8 * pixels, 25 * pixels, 8 * pixels, 64 << 20), "BG Directory": (8 * pixels, 8 * pixels, 8 * pixels, 64 << 20)}
    assert {mode: module._background_resource_terms(mode, (5, 7))
            for mode in expected} == expected
    seeded = _seed_existing(tmp_path, name="descriptor")
    plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d", preparation=seeded.preparation)
    core_plan = module._core_plan(plan.selected_plan, plan.requested_shared_science, seeded.mask)
    order = []; source = module._ReintegrateFrameSource(plan); describe, bind = source.container_descriptor, source.bind_allocation
    source.container_descriptor = lambda: order.append("descriptor") or describe()
    source.bind_allocation = lambda allocation: order.append(("bind", allocation)) or bind(allocation)
    class Sink(MemorySink):
        def begin(self, scan, reduction_plan): order.append("sink"); return super().begin(scan, reduction_plan)
    session = ScanSession(core_plan, source, Sink(), policy=plan.session_policy, executor=1)
    try: assert order[:2] == ["descriptor", ("bind", plan.resource_allocation)] and order.count("descriptor") == 1 and order.index("sink") > 1
    finally: session.finish(raise_on_failure=False)
    assert source.bound_allocation is plan.resource_allocation and source.jit_roots == ()
    with pytest.raises(ValueError, match="RESOURCE_ALLOCATION_IDENTITY"):
        source.bind_allocation(SimpleNamespace())
    def refuse(descriptor, message):
        effects = []; candidate = module._ReintegrateFrameSource(plan); candidate.container_descriptor = lambda: effects.append("descriptor") or descriptor; candidate.bind_allocation = lambda *_: effects.append("bind")
        class Forbidden(MemorySink):
            def begin(self, *_): effects.append("sink")
        with pytest.raises(ValueError, match=message): ScanSession(core_plan, candidate, Forbidden(), policy=plan.session_policy, executor=1)
        assert effects == ["descriptor"]
    base = source.container_descriptor()
    poisoned_requirements = replace(plan.resource_allocation.requirements, background_bytes=1, resolver_background_bytes=2, worker_background_bytes=3, background_binding_bytes=4)
    poisoned_allocation = replace(plan.resource_allocation, requirements=poisoned_requirements)
    poisoned_plan = SimpleNamespace(requested_shared_science=plan.requested_shared_science, detector_shape=plan.detector_shape, native_dtype=plan.native_dtype, resource_allocation=poisoned_allocation)
    independent = module._ReintegrateFrameSource(poisoned_plan).container_descriptor(); assert (independent.background_bytes, independent.resolver_background_bytes, independent.worker_background_bytes, independent.background_binding_bytes) == (0, 0, 0, 0)
    refuse(SimpleNamespace(frame_shape=base.frame_shape, dtype=base.dtype, background_bytes=0), "all four Background")
    refuse(SimpleNamespace(frame_shape=base.frame_shape, dtype=base.dtype, background_bytes=1, resolver_background_bytes=0, worker_background_bytes=0, background_binding_bytes=0), "identity|requirements|allocation")
    refuse(SimpleNamespace(frame_shape=[5, 7], dtype=base.dtype), "shape/dtype")
    legacy = module._ReintegrateFrameSource(plan); legacy.container_descriptor = lambda: SimpleNamespace(frame_shape=base.frame_shape, dtype=base.dtype)
    session = ScanSession(core_plan, legacy, MemorySink(), policy=plan.session_policy, executor=1)
    try: assert legacy.bound_allocation is plan.resource_allocation
    finally: session.finish(raise_on_failure=False)
    successor_plan = _successor_from_plan(plan)
    runtime_policies = []; init = ScanSession.__init__
    def runtime_init(self, *args, policy=None, **kwargs): runtime_policies.append(policy); return init(self, *args, policy=policy, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module, "resolve_session_policy", lambda *_a, **_k: pytest.fail("runtime re-resolved its plan allocation")); patch.setattr(ScanSession, "__init__", runtime_init); _stub_integrators(patch); assert run_reintegrate_successor(successor_plan).disposition == "COMMITTED"
    assert runtime_policies == [successor_plan.session_policy] and runtime_policies[0].allocation is successor_plan.resource_allocation
def test_gi_bootstrap_freezes_before_sink_with_bounded_roots_and_stable_snapshot(tmp_path, monkeypatch):
    from xrd_tools.reduction import GIMode
    import xrd_tools.reduction.core as core
    module = _module()
    for name, quiet_gi, disabled_motor, expected_angle in (("disabled", None, None, None), ("disabled-named", None, "theta", None), ("manual", GIMode(incidence_motor="Manual", incident_angle=.3), None, .3)):
        quiet = _seed_existing(tmp_path, gi=quiet_gi, disabled_motor=disabled_motor, name=f"gi-{name}")
        if disabled_motor:
            with h5py.File(quiet.target, "r+") as handle: del handle["entry/scan_data/theta"]; handle["entry/scan_data"]["theta"] = h5py.SoftLink("/missing")
        with monkeypatch.context() as patch:
            patch.setattr(core, "_apply_gi_freeze_policy", lambda *_a, **_k: pytest.fail("quiet GI scouted")); patch.setattr(module, "_load_fact", lambda *_a, **_k: pytest.fail("quiet GI read pixels"))
            quiet_plan = module.ReintegratePlan.from_artifact(quiet.target, entry="entry", dimension="1d", preparation=quiet.preparation)
        assert quiet_plan.gi_bootstrap_incidence is None and getattr(module._core_plan(quiet_plan.selected_plan, quiet_plan.requested_shared_science).gi, "incident_angle", None) == expected_angle
    gi = GIMode(incidence_motor="theta", mode_1d="q_total", mode_2d="qip_qoop")
    seeded = _seed_existing(tmp_path, gi=gi, name="gi")
    scouts, roots, scout_frames = [], [], []
    def freeze(plan, scan, *, freeze_policy, fi, initial_incident_angle, **_kwargs):
        scouts.append(tuple(frame.index for frame in scan.frames)); scout_frames.extend(scan.frames); assert freeze_policy == "scout_union" and initial_incident_angle == float(np.float32(.2)) and int(fi.detector.orientation) == 3; np.testing.assert_array_equal(plan.mask, seeded.mask)
        for frame in scan.frames: image = frame.loader(frame); roots.append(weakref.ref(image))
        return replace(plan, integration_1d=replace(plan.integration_1d, radial_range=(.1, 1.)))
    monkeypatch.setattr(core, "_apply_gi_freeze_policy", freeze)
    snapshot = seeded.target.read_bytes(); plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d", preparation=seeded.preparation)
    assert scouts == [(2, 9)] and plan.gi_bootstrap_incidence == float(np.float32(.2)) and seeded.target.read_bytes() == snapshot and all(ref() is None for ref in roots) and all(frame.image is frame.background is frame.geometry is frame.source_identity is frame.source_path is frame.loader is frame.mask is None and frame.source_frame_index is frame.normalization_factor is frame.background_dependency_bytes is frame.background_dependency_fingerprint is None and not frame.metadata for frame in scout_frames)
    assert module._core_plan(plan.selected_plan, plan.requested_shared_science).gi.incident_angle is None
    monkeypatch.setattr(core, "_apply_gi_freeze_policy", lambda *_a, **_k: pytest.fail("execution repeated GI freeze"))
    executions, dropped = [], {5}
    def integrate(image, fi, _plan, mode, *, mask, incident_angle, **_kwargs):
        label = int(np.asarray(image).flat[0] // 35); executions.append((label, incident_angle, int(fi.detector.orientation), np.array(mask, copy=True))); return _r1(np.nan if label in dropped else label + 200)
    monkeypatch.setattr(core, "_run_gi_1d", integrate)
    result = _run_successor(plan); assert result.committed_labels == (2, 9) and result.publication_dropped_labels == (5,)
    assert seeded.target.read_bytes() == snapshot
    assert [(v[0], v[1], v[2]) for v in executions] == [(2, float(np.float32(.2)), 3), (5, float(np.float32(.5)), 3), (9, float(np.float32(.9)), 3)]
    assert all(np.array_equal(v[3], seeded.mask) for v in executions)
    repeat = copy.deepcopy(seeded.preparation); repeat["selected_plan"] = module._plain(plan.selected_plan); dropped.clear(); executions.clear()
    replay = module.ReintegratePlan.from_artifact(result.output_artifact, entry="entry", dimension="1d", preparation=repeat)
    assert replay.labels == (2, 9) and replay.gi_bootstrap_incidence == float(np.float32(.2))
    replay_root = tmp_path / "gi-replay"
    replay_root.mkdir()
    from xrd_tools.reduction import run_reintegrate_successor
    assert run_reintegrate_successor(
        _successor_from_plan(replay, destination_directory=replay_root),
    ).committed_labels == (2, 9)
    for name, labels, active in (("single", (2,), False), ("constant", (2, 5, 9), True)):
        case = _seed_existing(tmp_path, labels=labels, gi=gi, background=active, name=f"gi-{name}")
        if name == "constant":
            with h5py.File(case.target, "r+") as handle: handle["entry/scan_data/theta"][...] = .2
        observed, case_frames, tokens, token, load = [], [], [], threading.Event(), module._load_fact
        def case_load(*args, **kwargs): tokens.append(args[4]); return load(*args, **kwargs)
        def case_freeze(plan, scan, **_kwargs):
            observed.append(tuple(frame.index for frame in scan.frames)); case_frames.extend(scan.frames); [frame.loader(frame) for frame in scan.frames]; assert all((frame.background is not None) == active for frame in scan.frames); return replace(plan, integration_1d=replace(plan.integration_1d, radial_range=(.1, 1.)))
        with monkeypatch.context() as patch:
            patch.setattr(module, "_load_fact", case_load); patch.setattr(core, "_apply_gi_freeze_policy", case_freeze); prepared = module.ReintegratePlan.from_artifact(case.target, entry="entry", dimension="1d", preparation=case.preparation, cancel_token=token)
        assert observed == [(2,)] and prepared.gi_bootstrap_incidence == float(np.float32(.2)) and tokens and all(value is token for value in tokens) and all(frame.image is frame.background is frame.geometry is frame.source_identity is frame.source_path is frame.loader is frame.mask is None and frame.source_frame_index is frame.normalization_factor is frame.background_dependency_bytes is frame.background_dependency_fingerprint is None and not frame.metadata for frame in case_frames)
def test_gi_bootstrap_cancel_uses_one_token_and_clears_before_any_output_effect(tmp_path, monkeypatch):
    import xrd_tools.reduction.core as core; from xrd_tools.reduction import GIMode; module = _module()
    plain = _seed_existing(tmp_path, name="boundary-cancel"); seeded = _seed_existing(tmp_path, gi=GIMode(incidence_motor="theta"), name="gi-cancel"); before = seeded.target.read_bytes()
    pre = threading.Event(); pre.set()
    with monkeypatch.context() as patch, pytest.raises(module.ReintegrateCancelled):
        patch.setattr(module, "capture_target_snapshot", lambda *_: pytest.fail("pre-cancel touched target")); patch.setattr(module, "_inspect_artifact", lambda *_: pytest.fail("pre-cancel inspected target")); module.ReintegratePlan.from_artifact(seeded.target, entry="entry", dimension="1d", preparation=seeded.preparation, cancel_token=pre)
    capture = module.capture_target_snapshot
    for position in (1, 2):
        token, calls = threading.Event(), []
        def cancel_capture(*args, **kwargs): value = capture(*args, **kwargs); calls.append(1); token.set() if len(calls) == position else None; return value
        with monkeypatch.context() as patch, pytest.raises(module.ReintegrateCancelled):
            patch.setattr(module, "capture_target_snapshot", cancel_capture); patch.setattr(module, "_inspect_artifact" if position == 1 else "_make_plan", lambda *_a, **_k: pytest.fail("cancellation crossed its target-hash boundary")); module.ReintegratePlan.from_artifact(plain.target, entry="entry", dimension="1d", preparation=plain.preparation, cancel_token=token)
        assert len(calls) == position
    token = threading.Event(); effects, source = [], module._source_fact
    def cancel_source(*args, **kwargs): value = source(*args, **kwargs); (effects.append("raw"), token.set()) if kwargs.get("read") else None; return value
    def freeze(_plan, scan, **_kwargs): effects.append("freeze"); scan.frames[0].loader(scan.frames[0]); pytest.fail("post-raw cancellation crossed into scout reduction")
    with monkeypatch.context() as patch, pytest.raises(module.ReintegrateCancelled):
        patch.setattr(module, "_source_fact", cancel_source); patch.setattr(core, "_apply_gi_freeze_policy", freeze); module.ReintegratePlan.from_artifact(seeded.target, entry="entry", dimension="1d", preparation=seeded.preparation, cancel_token=token)
    assert effects == ["freeze", "raw"] and seeded.target.read_bytes() == before


def test_nonhdf_snapshot_copy_cancellation_closes_source_and_tempdir(
    tmp_path, monkeypatch,
):
    module = _module()
    source = tmp_path / "large-source.tif"
    source.write_bytes(b"x" * ((2 << 20) + 1))
    observed = module._revision(source)
    raw = module.os.path.normcase(module.os.path.normpath(
        module.os.path.abspath(source)
    ))
    revisions = {
        raw: (observed[0], {}, "source_member", observed),
    }
    token = threading.Event()
    opened = []
    temporary_roots = []
    real_open = Path.open
    real_temporary = module.tempfile.TemporaryDirectory

    class CancellingSource:
        def __init__(self, stream):
            self.stream = stream
            self.closed = False
        def fileno(self):
            return self.stream.fileno()
        def read(self, size=-1):
            payload = self.stream.read(size)
            token.set()
            return payload
        def close(self):
            self.closed = True
            self.stream.close()

    def controlled_open(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if Path(path) == source and mode == "rb":
            proxy = CancellingSource(stream)
            opened.append(proxy)
            return proxy
        return stream

    def tracked_temporary(*args, **kwargs):
        owner = real_temporary(*args, **kwargs)
        temporary_roots.append(Path(owner.name))
        return owner

    monkeypatch.setattr(Path, "open", controlled_open)
    monkeypatch.setattr(
        module.tempfile, "TemporaryDirectory", tracked_temporary,
    )
    from xrd_tools.io import image as image_module
    monkeypatch.setattr(
        image_module, "read_image",
        lambda *_args, **_kwargs: pytest.fail(
            "cancelled snapshot entered the decoder"
        ),
    )

    with pytest.raises(module.ReintegrateCancelled):
        module._decode_nonhdf_source(
            source, "fabio", {
                "dataset_path": None,
                "frame_count": 1,
                "self_contained": True,
            }, 0, None, True, token,
            None, None, revisions,
        )

    assert len(opened) == 1 and opened[0].closed
    assert temporary_roots and all(
        not root.exists() for root in temporary_roots
    )


def test_jit_stub_fields_scrub_on_every_terminal_and_refusal(tmp_path, monkeypatch):
    from xrd_tools.reduction import NexusSink
    module = _module()
    good, bad = _seed_existing(tmp_path, name="jit-good"), _seed_existing(tmp_path, name="jit-refusal")
    good_plan = module.ReintegratePlan.from_artifact(good.target, entry="entry", dimension="1d", preparation=good.preparation)
    bad_plan = module.ReintegratePlan.from_artifact(bad.target, entry="entry", dimension="1d", preparation=bad.preparation); bad_bytes = bad.target.read_bytes()
    sources, frames, sinks = [], [], []; source_init, to_scan = module._ReintegrateFrameSource.__init__, module._ReintegrateFrameSource.to_scan; factory = NexusSink.for_finite_replacement.__func__
    def initialize(self, *args, **kwargs): source_init(self, *args, **kwargs); sources.append(self)
    def scan(self, *args, **kwargs): value = to_scan(self, *args, **kwargs); frames.extend(value.frames); return value
    def sink(cls, *args, **kwargs): value = factory(cls, *args, **kwargs); sinks.append(value); return value
    monkeypatch.setattr(module._ReintegrateFrameSource, "__init__", initialize); monkeypatch.setattr(module._ReintegrateFrameSource, "to_scan", scan); monkeypatch.setattr(NexusSink, "for_finite_replacement", classmethod(sink)); _stub_integrators(monkeypatch)
    assert _run_successor(good_plan).disposition == "COMMITTED"
    assert sources and frames and sinks
    mismatch = _seed_existing(tmp_path, name="jit-mismatch"); mismatch_plan = module.ReintegratePlan.from_artifact(mismatch.target, entry="entry", dimension="1d", preparation=mismatch.preparation); mismatch_bytes = mismatch.target.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(module, "_geometry_fact", lambda *_: SimpleNamespace(rot1=0., rot2=0., rot3=0., incident_angle=0.))
        with pytest.raises(ValueError, match="replacement local JIT stub differs"): _run_successor(mismatch_plan)
    assert mismatch.target.read_bytes() == mismatch_bytes
    monkeypatch.setattr(module, "_source_fact", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")))
    with pytest.raises(ValueError, match="REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"): _run_successor(bad_plan)
    assert bad.target.read_bytes() == bad_bytes and all(source.jit_roots == () and source._fact_reader is None and source._frames == {} for source in sources)
    assert all(frame.image is frame.background is frame.geometry is frame.source_identity is frame.source_path is frame.loader is frame.mask is None and frame.source_frame_index is frame.normalization_factor is frame.background_dependency_bytes is frame.background_dependency_fingerprint is None and not frame.metadata for frame in frames)
    assert all(sink._writer._replacement_read_context is sink._writer._replacement_manifest is sink._writer._replacement_expected is None and not sink._writer._row_cursors and sink._writer._replacement_labels == () for sink in sinks)
def test_exact_event_flows_unchanged_through_scan_session_and_stop(tmp_path, monkeypatch):
    from xrd_tools.reduction import Frame, Integration1DPlan, ReductionPlan, Scan
    from xrd_tools.session import scan_session as session_module
    module = _module(); token = threading.Event(); captured = []; seeded = _seed_existing(tmp_path, name="event", background=True)
    plan = module.ReintegratePlan.from_artifact(seeded.target, entry="entry", dimension="1d", preparation=seeded.preparation)
    real_init, real_load, real_background = session_module.ScanSession.__init__, module._load_fact, module._background_fact
    def session_init(self, *args, cancel_token=None, **kwargs): captured.append(("session", cancel_token)); return real_init(self, *args, cancel_token=cancel_token, **kwargs)
    def load(*args, **kwargs): captured.append(("raw", args[4])); return real_load(*args, **kwargs)
    def background(*args, **kwargs): captured.append(("background", args[4])); return real_background(*args, **kwargs)
    monkeypatch.setattr(session_module.ScanSession, "__init__", session_init); monkeypatch.setattr(module, "_load_fact", load); monkeypatch.setattr(module, "_background_fact", background); _stub_integrators(monkeypatch)
    assert _run_successor(plan, cancel_token=token).disposition == "COMMITTED"
    assert {role for role, _ in captured} == {"session", "raw", "background"} and all(value is token for _, value in captured)
    legacy_seen = []
    class Legacy:
        cancelled = False
        def cancel(self): self.cancelled = True
    class FakeReductionSession:
        def __init__(self, *args, cancel_token=None, **kwargs):
            legacy_seen.append(cancel_token)
            self.cancel_token = cancel_token
            self.is_running = True
            self.is_paused = False
            self.scan = args[1]
        def drain(self, *a, **k): return True
        def finish(self, **kwargs): return SimpleNamespace(cancelled=token.is_set())
        def _rollback_construction(self, primary): pass
    monkeypatch.setattr(session_module, "ReductionSession", FakeReductionSession)
    scan = Scan("event", [Frame(0, image=np.ones((2, 2)))], integrator=object())
    core_plan = ReductionPlan(integration_1d=Integration1DPlan(npt=2))
    for stop_token in (threading.Event(), Legacy()):
        legacy_seen.clear(); session = session_module.ScanSession(core_plan, scan, cancel_token=stop_token); session.stop()
        assert legacy_seen == [stop_token] and (stop_token.is_set() if type(stop_token) is threading.Event else stop_token.cancelled)
    with pytest.raises(TypeError, match="cancellation token"):
        session_module.ScanSession(core_plan, scan, cancel_token=object())


def test_a_family_stamped_run_output_reintegrates_end_to_end(tmp_path, monkeypatch):
    """THE PRIMARY WORKFLOW, end to end: integrate a scan, reintegrate the result.

    Fable F3 (P1) on `58123dc7`. `dbb84c33` made ordinary Run and Average stamp
    `@artifact_family_v1` through `NexusSink` -- a writer with NO finite lineage
    node -- so FAMILY-ONLY became the shape of every real Run output, and
    `FinitePredecessorReceipt`'s all-or-none rule refused every one of them as a
    Reintegrate predecessor.

    It survived my own testing and TWO independent reviews because every
    predecessor fixture wrote either an unstamped artifact or a full finite
    lineage. The production shape had no fixture at all. This is that fixture:
    the seed writes through `NexusSink` exactly as a Run does, WITH the stamp.

    THIS ROW WAS AN XFAIL TWICE, FOR TWO WRONG REASONS, before it passed.

    First I recorded the blocker as Codex F5 (`scientific_signature`) reached
    from the Run side. Wrong: a real Run persists a signature -- different
    error, different site, mapped together by reading rather than running.
    Then I recorded it as a fixture limitation. Also wrong.

    The real cause was that `0015f756` decoupled the family from the lineage
    triple in `FinitePredecessorReceipt` and NOT in
    `admit_finite_artifact_lineage`, so a stamped Run output passed PLANNING and
    failed inside the run. Half a fix, which moved the failure later instead of
    removing it, and each of my explanations was a story that fit the symptom I
    happened to be looking at.
    """
    from xrd_tools.io.schema import ARTIFACT_FAMILY_ATTR
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name="run-shape", artifact_family="existing",
    )
    target = Path(seeded.target)
    with h5py.File(target, "r") as document:
        assert document["entry"].attrs[ARTIFACT_FAMILY_ATTR] == "existing"
    source_bytes = target.read_bytes()

    plan = ReintegrateSuccessorPlan.from_artifact(
        target,
        entry="entry",
        dimension="1d",
        preparation=copy.deepcopy(seeded.preparation),
        expected_terminal_identity=seeded.terminal.commit_identity,
        expected_labels=seeded.labels,
        destination_directory=target.parent,
        explicit_output=None,
    )
    result = run_reintegrate_successor(plan)

    assert result.disposition == "COMMITTED"
    # The STAMPED family is consumed, so the slot is the root family's rather
    # than a chain off the predecessor's own stem.
    assert Path(result.output_artifact).name == "existing_reintegrate1d.nexus"
    # And the predecessor it read is untouched.
    assert target.read_bytes() == source_bytes
