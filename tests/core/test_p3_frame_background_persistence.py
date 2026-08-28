"""P3-3B policy and detached dependency persistence."""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.core.provenance import read_provenance_from_handle
from xrd_tools.core.scan import Scan, ScanFrame
from xrd_tools.io.nexus_record import (
    read_background_dependency,
    write_background_dependency,
)
from xrd_tools.reduction import FrameBackgroundPlan, FrameReduction, NexusSink, ReductionPlan, ReductionResult, resolve_frame_background
from xrd_tools.session.run_configuration import RunIntent


def _r1(value):
    return IntegrationResult1D(radial=np.array([0.1, 0.2]),
        intensity=np.full(2, value, float), sigma=np.full(2, 0.1), unit="q_A^-1")


def _descriptor(value=1, *, mode="Single BG File", hdf=False, metadata=False) -> bytes:
    from xrd_tools.reduction import background as module
    locator = {"Single BG File": "/data/background.h5" if hdf else "/data/background.tif",
               "Series Average": "/data/background_1.tif",
               "BG Directory": "/data/backgrounds"}[mode]
    plan = FrameBackgroundPlan(mode=mode, locator=locator,
        **(({"match_rule": "Metadata Key", "metadata_key": "energy"} if metadata else {"match_rule": "Scan Root + Frame Number"}) if mode == "BG Directory" else {}),
        **({"dataset_path": "/entry/data/data_000001", "frame_index": 0} if hdf else {}))
    projected = (("energy", module._tag(1.0)),) if metadata else ()
    fact = (value, "/raw/sample_1.tif", None, 0, (2, 2), projected)
    native = np.full((2, 2), value, np.uint16)
    decoded_native = native.astype(np.float64) if hdf else native
    decoded = {"shape": [2, 2], "dtype": decoded_native.dtype.str,
               "sha256": hashlib.sha256(decoded_native.tobytes()).hexdigest()}
    body = {"version": 1, "mode": mode, "policy": plan.to_mapping(),
            "frame_fact": fact}
    if mode == "Single BG File":
        hdf_proof = ({"dataset_path": plan.dataset_path, "frame_index": 0,
            "shape": [1, 2, 2], "dtype": native.dtype.str} if hdf else None)
        body.update(source={"locator": locator,
            "state": [1, 2, stat.S_IFREG, native.nbytes, 3, 4],
            "sha256": hashlib.sha256(b"source").hexdigest(), "hdf": hdf_proof,
            "decoded": decoded}, metadata_items=(), metadata_source=None,
            decoded=decoded)
    else:
        body.update(manifest={"version": 1, "count": 1, "receipt_bytes": 7,
            "sha256": hashlib.sha256(b"receipt").hexdigest()},
            selected=(locator if mode == "Series Average" else
                      "/data/backgrounds/sample_1.tif"))
        if mode == "Series Average":
            body["normalization"] = {"target": module._tag(1.0),
                                      "denominator": module._tag(1.0)}
        else:
            body.update(metadata_items=projected, metadata_source=({"locator": "/data/backgrounds/sample_1.json", "state": [1, 2, stat.S_IFREG, 2, 3, 4], "sha256": hashlib.sha256(b"{}").hexdigest()} if metadata else None), decoded=decoded)
    result = module._result(plan, fact, np.full((2, 2), float(value)), body, None)
    assert result.descriptor_bytes is not None
    return result.descriptor_bytes


def test_active_policy_freezes_identity_and_none_preserves_legacy_shape(tmp_path) -> None:
    legacy = RunIntent().freeze()
    explicit_none = RunIntent(background=FrameBackgroundPlan()).freeze()
    assert legacy.fingerprint == explicit_none.fingerprint
    assert legacy.as_provenance() == explicit_none.as_provenance()
    assert "background" not in legacy.as_provenance()
    active = RunIntent(background=FrameBackgroundPlan(
        mode="Single BG File", locator="/data/background.tif", scale=-1.0,
    )).freeze()
    assert active.fingerprint != legacy.fingerprint
    assert active.as_provenance()["background"] == active.background.to_mapping()
    assert FrameBackgroundPlan.from_mapping(active.as_provenance()["background"]) == active.background
    with pytest.raises(ValueError):
        FrameBackgroundPlan.from_mapping({**active.background.to_mapping(), "locator": "relative.tif"})
    with pytest.raises(ValueError):
        FrameBackgroundPlan.from_mapping({"version": 2, "mode": "None"})
    from xrd_tools.session.readiness import (
        append_config_mismatch_check, processing_config_from_mapping)
    none_signature = processing_config_from_mapping({})
    assert none_signature == processing_config_from_mapping(
        {"background": {"version": 1, "mode": "None"}})
    active_signature = processing_config_from_mapping(
        {"background": active.background.to_mapping()})
    changed = FrameBackgroundPlan(mode="Single BG File",
        locator="/data/background.tif", scale=-2.0)
    check = append_config_mismatch_check("Append", active_signature,
        processing_config_from_mapping({"background": changed.to_mapping()}))
    assert not check.ok and "Background" in check.mismatched_fields
    with pytest.raises(ValueError): FrameBackgroundPlan.from_mapping({"version": True, "mode": "None"})
    bogus = active.background.to_mapping(); bogus["mode"] = "Bogus"
    with pytest.raises(ValueError): FrameBackgroundPlan.from_mapping(bogus)
    source = tmp_path / "source.tif"; source.write_bytes(b"raw")
    alias = tmp_path / "alias.tif"; alias.symlink_to(source)
    persisted = FrameBackgroundPlan(mode="Single BG File", locator=str(alias)).to_mapping()
    source.unlink(); alias.unlink(); alias.symlink_to(tmp_path / "retargeted.tif")
    assert FrameBackgroundPlan.from_mapping(persisted).to_mapping() == persisted
    class Text(str): pass
    for name, value in (("mode", Text("Single BG File")), ("stable_read_policy", Text(active.background.stable_read_policy))):
        malformed = dict(active.background.to_mapping()); malformed[name] = value
        with pytest.raises(ValueError): FrameBackgroundPlan.from_mapping(malformed)
    with pytest.raises(ValueError): FrameBackgroundPlan.from_mapping({"version": 1, "mode": Text("None")})


def test_frame_descriptor_roundtrip_and_missing_child_none(tmp_path) -> None:
    master = tmp_path / "background.h5"; target_source = tmp_path / "target.tif"; target_source.write_bytes(b"raw")
    with h5py.File(master, "w") as handle: handle.create_dataset("entry/data/data_000001", data=np.arange(4, dtype=np.uint16).reshape(1, 2, 2))
    direct = resolve_frame_background(FrameBackgroundPlan(mode="Single BG File", locator=str(master), dataset_path="/entry/data/data_000001", frame_index=0), (3, str(target_source.resolve()), None, 0, (2, 2), ()))
    assert direct.disposition == "RESOLVED" and direct.descriptor_bytes is not None
    assert json.loads(direct.descriptor_bytes)["source"]["hdf"]["dtype"] == "<u2" and json.loads(direct.descriptor_bytes)["decoded"]["dtype"] == "<f8"
    master.unlink(); target_source.unlink()
    raws = tuple(_descriptor(index, mode=mode) for index, mode in enumerate(
        ("Single BG File", "Series Average", "BG Directory"))) + (direct.descriptor_bytes, _descriptor(4, mode="BG Directory", metadata=True))
    path = tmp_path / "record.h5"
    with h5py.File(path, "w") as handle:
        frames = handle.create_group("entry/frames")
        for index, raw in enumerate(raws):
            frame = frames.create_group(f"frame_{index:04d}")
            assert read_background_dependency(frame) is None
            fingerprint = hashlib.sha256(raw).hexdigest()
            write_background_dependency(frame, raw, fingerprint)
            assert read_background_dependency(frame) == (raw, fingerprint)
            parsed = json.loads(raw); plan = FrameBackgroundPlan.from_mapping(parsed["policy"])
            fact = parsed["frame_fact"]
            from xrd_tools.reduction.background import _frame
            rebuilt = (fact[0], fact[1], fact[2], fact[3], tuple(fact[4]),
                       tuple((row[0], tuple(row[1])) for row in fact[5]))
            assert _frame(rebuilt, persisted=True) == rebuilt and plan.mode == parsed["mode"]
        frame = frames["frame_0000"]
        fingerprint = hashlib.sha256(raws[0]).hexdigest()
        assert set(frame["background_dependency"]) == {"descriptor_json", "fingerprint"}
        with pytest.raises(ValueError):
            write_background_dependency(frame, None, fingerprint)
        with pytest.raises(ValueError): write_background_dependency(frame, None, None)
        bad_values = []
        for raw in raws:
            parsed = json.loads(raw)
            variants = [dict(parsed), dict(parsed), dict(parsed)]
            variants[0]["policy"] = {"version": 1, "mode": "None"}
            variants[1]["frame_fact"] = [0, "relative.tif", None, 0, [2, 2], []]
            variants[2]["result_sha256"] = "A" * 64
            if "manifest" in parsed:
                bad = dict(parsed); bad["manifest"] = {**bad["manifest"], "sha256": "A" * 64}
                variants.append(bad)
                bad = json.loads(raw); bad["manifest"]["receipt_bytes"] = bad["manifest"]["count"] * 16_384 + 1; variants.append(bad)
            if parsed["mode"] == "Series Average":
                from xrd_tools.reduction.background import _tag
                bad = json.loads(raw); bad["normalization"]["denominator"] = _tag(2.0); variants.append(bad)
            if parsed.get("source", {}).get("hdf") is not None:
                for change in ({"shape": [2]}, {"shape": [1, 3, 2]}, {"frame_index": False}, {"dtype": "|S1"}):
                    bad = json.loads(raw); bad["source"]["hdf"].update(change); variants.append(bad)
            bad = json.loads(raw); bad["version"] = True; variants.append(bad)
            bad = json.loads(raw); bad["frame_fact"][4][0] = True; variants.append(bad)
            if "source" in parsed:
                bad = json.loads(raw); bad["source"]["state"][2] = stat.S_IFDIR; variants.append(bad)
            bad = dict(parsed); bad["members"] = ["hidden"]; variants.append(bad)
            bad_values.extend(variants)
        for index, bad in enumerate(bad_values, 10):
            encoded = json.dumps(bad, sort_keys=True, separators=(",", ":")).encode()
            with pytest.raises(ValueError):
                write_background_dependency(frames.create_group(f"frame_{index:04d}"),
                    encoded, hashlib.sha256(encoded).hexdigest())
        noncanonical = json.dumps(json.loads(raws[0]), sort_keys=True, indent=1).encode()
        for encoded in (b"\xff", noncanonical, b'{ "version": 1 }', b"x" * 262_145):
            with pytest.raises((ValueError, json.JSONDecodeError)):
                write_background_dependency(frames.create_group(f"bad_{len(encoded)}"),
                    encoded, hashlib.sha256(encoded).hexdigest())
        def populated(name):
            value = frames.create_group(name); raw = raws[0]
            write_background_dependency(value, raw, hashlib.sha256(raw).hexdigest()); return value
        broken = populated("bad_nx"); broken["background_dependency"].attrs["NX_class"] = "NXdata"
        with pytest.raises(ValueError): read_background_dependency(broken)
        broken = populated("bad_link"); frames.move("bad_link/background_dependency", "stored_dependency")
        broken["background_dependency"] = h5py.SoftLink("/entry/frames/stored_dependency")
        with pytest.raises(ValueError): read_background_dependency(broken)
        for name, mutate in (("bad_shape", "shape"), ("bad_dtype", "dtype"),
                             ("bad_oversize", "oversize"), ("bad_storage", "storage"),
                             ("bad_scalar_link", "link"), ("bad_utf8", "utf8")):
            broken = populated(name); dep = broken["background_dependency"]
            raw = raws[0]; del dep["descriptor_json"]
            if mutate == "shape": dep.create_dataset("descriptor_json", data=[raw], dtype=f"S{len(raw)}")
            elif mutate == "dtype": dep.create_dataset("descriptor_json", data=raw, dtype=f"S{len(raw)}")
            elif mutate == "oversize": dep.create_dataset("descriptor_json", data=raw,
                dtype=h5py.string_dtype("utf-8", length=262_145))
            elif mutate == "storage": dep.create_dataset("descriptor_json", shape=(),
                dtype=h5py.string_dtype("utf-8", length=len(raw)), fillvalue=raw)
            elif mutate == "utf8": dep.create_dataset("descriptor_json", data=b"\xff",
                dtype=h5py.string_dtype("utf-8", length=1))
            else:
                dep.create_dataset("stored", data=raw,
                    dtype=h5py.string_dtype("utf-8", length=len(raw)))
                dep["descriptor_json"] = h5py.SoftLink("stored")
            with pytest.raises(ValueError): read_background_dependency(broken)
        broken = populated("bad_extra"); broken["background_dependency"].create_dataset("extra", data=1)
        with pytest.raises(ValueError): read_background_dependency(broken)
    with h5py.File(path, "a") as handle:
        dep = handle["entry/frames/frame_0000/background_dependency"]
        dep["fingerprint"][...] = "0" * 64
        with pytest.raises(ValueError, match="fingerprint"):
            read_background_dependency(handle["entry/frames/frame_0000"])
        del dep["fingerprint"]
        with pytest.raises(ValueError, match="collection"):
            read_background_dependency(handle["entry/frames/frame_0000"])
    parsed = json.loads(raws[0]); plan = FrameBackgroundPlan.from_mapping(parsed["policy"])
    pair = (raws[0], hashlib.sha256(raws[0]).hexdigest())
    reduction = FrameReduction(0, result_1d=_r1(1))
    missing = ScanFrame(0, source_path="/raw/sample_1.tif", source_frame_index=0)
    guarded = NexusSink(tmp_path / "guard.nexus", run_configuration_provenance={"background": plan.to_mapping()})
    with pytest.raises(ValueError, match="active Background policy"):
        guarded._write_frame_record(missing, reduction, result_1d=_r1(1), result_2d=None, mode_1d="standard", mode_2d="standard")
    present = ScanFrame(0, source_path="/raw/sample_1.tif", source_frame_index=0,
        background_dependency_bytes=pair[0], background_dependency_fingerprint=pair[1])
    inactive = NexusSink(tmp_path / "inactive.nexus", run_configuration_provenance={})
    with pytest.raises(ValueError, match="inactive Background"):
        inactive._write_frame_record(present, reduction, result_1d=_r1(1), result_2d=None, mode_1d="standard", mode_2d="standard")
    changed = FrameBackgroundPlan(mode="Single BG File", locator="/data/background.tif", scale=2.0)
    mismatched = NexusSink(tmp_path / "mismatch.nexus", run_configuration_provenance={"background": changed.to_mapping()})
    with pytest.raises(ValueError, match="exact frame dependency"):
        mismatched._write_frame_record(present, reduction, result_1d=_r1(1), result_2d=None, mode_1d="standard", mode_2d="standard")
    from xrd_tools.io.record_writer import RecordWrite
    with pytest.raises(ValueError, match="label differs"):
        RecordWrite(label=9, result_1d=_r1(1), background_dependency_bytes=pair[0], background_dependency_fingerprint=pair[1])
    from xrd_tools.reduction.core import GIMode
    output = tmp_path / "writer-route.nexus"; present.image = np.ones((2, 2), np.uint16)
    scan = Scan("background", [present]); reduction_plan = ReductionPlan(gi=GIMode(mode_1d="q_total"))
    sink = NexusSink(output, overwrite=True, atomic=False, flush_every=None,
        write_thumbnails=False, run_configuration_provenance={"background": plan.to_mapping()})
    sink.begin(scan, reduction_plan); sink.write(present, reduction); sink.write(present, reduction)
    sink.replace(present, FrameReduction(0, result_1d=_r1(2)))
    sink.write(present, FrameReduction(0, result_1d=_r1(3), mode_1d="q_ip", write_frame_record=False))
    sink.finish(ReductionResult("background", {}, 1))
    with h5py.File(output, "r") as handle:
        provenance = read_provenance_from_handle(handle)
        assert FrameBackgroundPlan.from_mapping(provenance["config"]["run_configuration"]["background"]) == plan
        assert read_background_dependency(handle["entry/frames/frame_0000"]) == pair
        np.testing.assert_allclose(handle["entry/integrated_1d/intensity"][0], 2.0)
        np.testing.assert_allclose(handle["entry/integrated_1d/q_ip/intensity"][0], 3.0)


def test_append_replacement_and_live_unseen_descriptor_rules(tmp_path) -> None:
    from xdart.gui.tabs.scattering.output_preflight import _merge_background_binding

    first_raw, second_raw = _descriptor(3), _descriptor(4)
    first = (3, (3, "/raw/sample_1.tif", None, 0, (2, 2), ()), first_raw,
             hashlib.sha256(first_raw).hexdigest())
    second = (4, (4, "/raw/sample_1.tif", None, 0, (2, 2), ()), second_raw,
              hashlib.sha256(second_raw).hexdigest())
    bindings = _merge_background_binding((), first, limit=64 * 1024 * 1024)
    assert _merge_background_binding(bindings, first, limit=64 * 1024 * 1024) is bindings
    assert _merge_background_binding(bindings, second, limit=64 * 1024 * 1024) == (first, second)
    changed = (3, first[1], second_raw, second[3])
    with pytest.raises(ValueError, match="changed"):
        _merge_background_binding(bindings, changed, limit=64 * 1024 * 1024)
    huge = (4, second[1], b"x" * 262_145, hashlib.sha256(b"x" * 262_145).hexdigest())
    with pytest.raises(ValueError):
        _merge_background_binding(bindings, huge, limit=64 * 1024 * 1024)
    encoded = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()
    first_charge = 1024 + len(encoded(first[1])) + len(first[2]) + 64
    assert _merge_background_binding((), first, limit=first_charge) == (first,)
    with pytest.raises(ValueError, match="allocation"):
        _merge_background_binding((), first, limit=first_charge - 1)
    assert tuple(value[0] for value in _merge_background_binding((second,), first,
        limit=64 * 1024 * 1024)) == (3, 4)
    changed_fact = (3, (3, "/raw/changed.tif", None, 0, (2, 2), ()), first_raw, first[3])
    with pytest.raises(ValueError, match="changed"):
        _merge_background_binding(bindings, changed_fact, limit=64 * 1024 * 1024)
    prefix = tmp_path / "prefix.nexus"
    with h5py.File(prefix, "w") as handle:
        frames = handle.create_group("entry/frames")
        for binding in (first, second):
            frame = frames.create_group(f"frame_{binding[0]:04d}")
            write_background_dependency(frame, binding[2], binding[3])
    from xrd_tools.io.record_writer import _read_persisted_background_bindings, _validate_persisted_background_bindings
    loaded = _read_persisted_background_bindings(prefix, (3, 4), 64 * 1024 ** 2)
    assert loaded == (first, second); _validate_persisted_background_bindings(prefix, loaded, (3, 4))
    with pytest.raises(ValueError, match="allocation"):
        _read_persisted_background_bindings(prefix, (3,), first_charge - 1)
    with h5py.File(prefix, "a") as handle:
        handle["entry/frames/frame_0004/background_dependency/fingerprint"][...] = "0" * 64
    with pytest.raises(ValueError): _validate_persisted_background_bindings(prefix, loaded, (3, 4))
