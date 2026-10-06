from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactKind, AnalysisArtifactRequest, AnalysisArtifactOverwrite,
    AnalysisArtifactInvalid, admit_analysis_artifact, read_analysis_artifact,
    project_analysis_artifact_result, analysis_execution_attestation_digest,
    _project_analysis_artifact_result_v1_compat,
    canonical_analysis_provenance,
)
from xrd_tools.io.analysis_stitch_auxiliary import (
    analysis_contributing_frame_binding, analysis_contributing_frame_members, canonical_frame_manifest,
)
from xrd_tools.io.frame_view import read_contributing_frame_view
from xrd_tools.io.stitch_record import write_stitch_projection
from xrd_tools.io.output_transaction import OutputTransactionCoordinator
from tests.core.test_analysis_artifact import _execution_attestation


def _publish(tmp_path, *, unit="q_A^-1", xu=False, science_frame_count=3, fractional=False, source_stamp=True,
             legacy_members=False, point_images=None, target=None):
    project = tmp_path / "project"
    project.mkdir()
    raw = np.arange(24, dtype=np.int32).reshape(4, 6)
    raw_file = project / "frame.raw"
    raw.tofile(raw_file)
    state = raw_file.stat()
    snapshot = {"size": state.st_size, "mtime_ns": state.st_mtime_ns} if source_stamp else {}
    records = [dict(frame_index=index, scan_label=str(scan), source_path=str(raw_file),
        source_frame_index=0, thumbnail=raw.astype(np.float32) + index,
        mask_baked=False, thumbnail_mask_baked=False,
        source_snapshot=dict(raw_shape=[4, 6], raw_dtype="int32", raw_header_bytes=0, rotation=0, **snapshot),
        identity=dict(source_id="sample.spec", scan_id=str(scan), frame_index=index,
                      contribution_index=global_index))
        for global_index, (scan, index) in enumerate(((2, 0), (2, 1), (3, 0)))]
    if point_images is not None:
        from xrd_tools.sources.point_images import PointImagePolicy
        policy = PointImagePolicy.from_dict(point_images)
        second_file = project / "second.raw"
        (raw + 10).tofile(second_file)
        second_state = second_file.stat()
        for record in records:
            first = {key: record.pop(key) for key in (
                "source_path", "source_frame_index", "source_snapshot")}
            second = dict(source_path=str(second_file), source_frame_index=0,
                source_snapshot=dict(first["source_snapshot"], size=second_state.st_size,
                                     mtime_ns=second_state.st_mtime_ns))
            record["source_constituents"] = [second] if policy.operation == "select" else [first, second]
            record["point_images"] = policy.to_dict()
    members = analysis_contributing_frame_members(records, source_base=project)
    provenance = {"frame_members": members if legacy_members else analysis_contributing_frame_binding(members)}
    radial = np.linspace(0.1, 5.0, 40) if unit == "q_A^-1" else np.linspace(1, 100, 40)
    intensity = np.arange(40, dtype=float) + 1
    projector = _project_analysis_artifact_result_v1_compat if fractional else project_analysis_artifact_result
    projection = projector(kind=AnalysisArtifactKind.STITCH_1D,
        axes=(("q", radial),), axis_units=(("q", unit),), intensity=intensity,
        sigma=None, coverage=np.full(40, 0.25 if fractional else 1.), normalization=np.ones(40),
        wavelength_A=1.0, frame_records=records, source_base=project)
    digest = hashlib.sha256(b"native-stitch-test").hexdigest()
    extra = {}
    if xu:
        attestation = _execution_attestation(digest, projection.result_fingerprint, frame_count=science_frame_count)
        extra = dict(execution_attestation=attestation,
            execution_attestation_digest=analysis_execution_attestation_digest(
                AnalysisArtifactKind.STITCH_1D, attestation, request_fingerprint=digest))
    request = AnalysisArtifactRequest(tmp_path / "stitched.nexus" if target is None else target, AnalysisArtifactKind.STITCH_1D,
        AnalysisArtifactOverwrite.CREATE_NEW, digest, digest, digest, digest, provenance,
        schema_version=7 if xu else 6, **extra)
    output = admit_analysis_artifact(request, coordinator=OutputTransactionCoordinator())
    receipt = output.publish(lambda entry: write_stitch_projection(entry, request.target, projection,
        frame_records=records, source_base=project))
    return request.target, raw, project, receipt


def test_contributors_must_match_attested_science_frame_count(tmp_path):
    with pytest.raises(AnalysisArtifactInvalid, match="science execution count"):
        _publish(tmp_path, xu=True, science_frame_count=2)
    assert not (tmp_path / "stitched.nexus").exists()


@pytest.mark.parametrize("unit", ["q_A^-1", "2th_deg"])
@pytest.mark.parametrize("xu", [False, True])
@pytest.mark.parametrize("legacy_members", [False, True])
def test_native_axes_contributors_and_raw_relocation_round_trip(tmp_path, unit, xu, legacy_members):
    path, raw, project, receipt = _publish(tmp_path, unit=unit, xu=xu, legacy_members=legacy_members)
    payload = read_analysis_artifact(path, expected_receipt=receipt)
    assert payload.native_unit == unit
    assert payload.wavelength_A == 1.0
    companion = payload.radial_axis("2th_deg" if unit == "q_A^-1" else "q_A^-1")
    assert not np.allclose(np.diff(companion), np.diff(companion)[0])
    np.testing.assert_array_equal(payload.intensity, np.arange(40) + 1)
    assert [row["key"] for row in payload.contributing_frames] == [
        "scan_2/frame_0000", "scan_2/frame_0001", "scan_3/frame_0000"]
    frame = read_contributing_frame_view(path, 0, scan="3", full_raw=True, artifact_payload=payload)
    np.testing.assert_array_equal(frame.raw, raw)
    assert frame.metadata_raw["scan_id"] == "3"
    with h5py.File(path, "r") as handle:
        source = handle["entry/stitch/frames/scan_3/frame_0000/source"]
        assert source["path"].asstr()[()] == "frame.raw"
        assert "raw" not in handle["entry/stitch/frames/scan_3/frame_0000"]
    moved = tmp_path / "moved-project"
    shutil.move(project, moved)
    # Curve admission never follows raw pointers. Full Raw has explicit fallback.
    np.testing.assert_array_equal(read_analysis_artifact(path).intensity, payload.intensity)
    fallback = read_contributing_frame_view(path, 0, scan="3", full_raw=True)
    assert fallback.raw is None
    assert fallback.thumbnail.shape == raw.shape
    relocated = read_contributing_frame_view(path, 0, scan="3", source_root=moved, full_raw=True)
    np.testing.assert_array_equal(relocated.raw, raw)


@pytest.mark.parametrize("xu", [False, True])
@pytest.mark.parametrize("key,value", [
    ("version", 2), ("version", True), ("version", 1.0),
    ("count", 2), ("count", True), ("count", 3.0),
    ("sha256", "0" * 64), ("sha256", None), ("unexpected", 1),
])
def test_compact_member_binding_rejects_corruption(tmp_path, xu, key, value):
    path, _, _, _ = _publish(tmp_path, xu=xu)
    with h5py.File(path, "r+") as handle:
        envelope = handle["entry/stitch"]
        provenance = json.loads(bytes(envelope["provenance_json"][()]))
        provenance["frame_members"][key] = value
        del envelope["provenance_json"]
        envelope.create_dataset("provenance_json", data=np.bytes_(canonical_analysis_provenance(provenance).encode()))
    with pytest.raises(AnalysisArtifactInvalid, match="frozen selected members"):
        read_analysis_artifact(path)


@pytest.mark.parametrize("xu", [False, True])
@pytest.mark.parametrize("legacy_members", [False, True])
@pytest.mark.parametrize("mutation", ["reorder", "source-identity"])
def test_member_binding_rejects_changed_full_records(tmp_path, xu, legacy_members, mutation):
    path, _, _, _ = _publish(tmp_path, xu=xu, legacy_members=legacy_members)
    with h5py.File(path, "r+") as handle:
        entry = handle["entry/stitch"]
        manifest = json.loads(bytes(entry["contributing_frames_json"][()]))
        if mutation == "reorder":
            manifest["records"].reverse()
        else:
            manifest["records"][0]["identity"]["source_id"] = "different.spec"
        del entry["contributing_frames_json"]
        entry.create_dataset("contributing_frames_json", data=np.bytes_(canonical_frame_manifest(manifest).encode()))
    with pytest.raises(AnalysisArtifactInvalid, match="frozen selected members"):
        read_analysis_artifact(path)


@pytest.mark.parametrize("mutation", ["thumbnail", "membership", "raw-copy", "external"])
def test_strict_native_graph_rejects_corruption(tmp_path, mutation):
    path, _, _, _ = _publish(tmp_path)
    payload = read_analysis_artifact(path)
    with h5py.File(path, "r+") as handle:
        frame = handle["entry/stitch/frames/scan_2/frame_0000"]
        if mutation == "thumbnail":
            frame["thumbnail"][0, 0] = 100
        elif mutation == "membership":
            del handle["entry/stitch/frames/scan_2/frame_0001"]
        elif mutation == "raw-copy":
            frame.create_dataset("raw", data=np.ones((4, 6)))
        else:
            del frame["thumbnail"]
            frame["thumbnail"] = h5py.ExternalLink("not-opened.h5", "/pixels")
    with pytest.raises(AnalysisArtifactInvalid):
        read_analysis_artifact(path)
    with pytest.raises(AnalysisArtifactInvalid, match="changed since detached"):
        read_contributing_frame_view(path, 0, scan="2", artifact_payload=payload)


def test_exact_external_hdf_dataset_selector_and_original_frame(tmp_path):
    path, _, project, _ = _publish(tmp_path)
    raw_file = project / "linked.h5"
    external = project / "data.h5"
    values = np.arange(3 * 4 * 6).reshape(3, 4, 6).astype("uint16")
    with h5py.File(external, "w") as handle:
        handle.create_dataset("real", data=values)
    with h5py.File(raw_file, "w") as handle:
        handle.create_dataset("distractor", data=np.zeros_like(values))
        handle["selected"] = h5py.ExternalLink("data.h5", "/real")
    # Exercise the same shared raw owner with its exact persisted selector.
    from xrd_tools.io.nexus_record import write_contributing_frames
    from xrd_tools.io.read import get_raw_frame
    from tests.core.v2_fixture_factory import make_v2_entry
    ordinary = tmp_path / "ordinary.nexus"
    with h5py.File(ordinary, "w") as handle:
        entry = make_v2_entry(handle, frame_indices=(9,), with_2d=False)
        write_contributing_frames(entry, [dict(frame_index=9, scan_label="5", source_path=raw_file,
            source_frame_index=2, thumbnail=np.ones((2, 2)),
            source_snapshot=dict(dataset_path="/selected"))], source_base=project)
    np.testing.assert_array_equal(get_raw_frame(ordinary, 9, scan="5", allow_thumbnail=False), values[2])


def test_native_mg_preserves_fractional_pixel_split_coverage(tmp_path):
    path, _, _, receipt = _publish(tmp_path, fractional=True)
    payload = read_analysis_artifact(path, expected_receipt=receipt)
    np.testing.assert_array_equal(payload.coverage, np.full(40, .25))


def test_changed_raw_source_returns_saved_thumbnail_without_changing_curve(tmp_path):
    import os
    path, raw, project, receipt = _publish(tmp_path)
    payload = read_analysis_artifact(path, expected_receipt=receipt)
    before = read_contributing_frame_view(path, 0, scan="2", full_raw=True, artifact_payload=payload)
    original = project / "frame.raw"
    state = original.stat()
    (raw + 1000).tofile(original)
    os.utime(original, ns=(state.st_atime_ns, state.st_mtime_ns + 1_000_000))
    assert original.stat().st_size == state.st_size
    changed = read_contributing_frame_view(path, 0, scan="2", full_raw=True, artifact_payload=payload)
    assert changed.raw is None
    np.testing.assert_array_equal(changed.thumbnail, before.thumbnail)
    assert changed.extra["raw_unavailable_reason"] == "raw source changed since processing"
    np.testing.assert_array_equal(read_analysis_artifact(path).intensity, payload.intensity)


def test_unstamped_source_preserves_curve_and_thumbnail_but_cannot_claim_original_raw(tmp_path):
    path, raw, _project, _receipt = _publish(tmp_path, source_stamp=False)
    view = read_contributing_frame_view(path, 0, scan="2", full_raw=True)
    assert view.raw is None
    assert view.thumbnail.shape == raw.shape
    assert view.extra["raw_unavailable_reason"] == "saved raw source identity unavailable"
    assert read_analysis_artifact(path).intensity.size == 40


@pytest.mark.parametrize("operation,reverse,delta", [
    ("average", False, 5), ("difference", False, 10),
    ("difference", True, -10), ("select", False, 10),
])
def test_grouped_native_artifact_real_writer_and_full_raw(tmp_path, operation, reverse, delta):
    policy = dict(images_per_point=2, operation=operation, image_index=2, reverse=reverse)
    path, raw, project, receipt = _publish(tmp_path, point_images=policy)
    payload = read_analysis_artifact(path, expected_receipt=receipt)
    row = payload.contributing_frames[0]
    assert row["point_images"] == policy
    assert "source_path" not in row and "source_frame_index" not in row
    expected_paths = ["second.raw"] if operation == "select" else ["frame.raw", "second.raw"]
    assert [member["source_path"] for member in row["source_constituents"]] == expected_paths
    view = read_contributing_frame_view(path, 0, scan="2", full_raw=True, artifact_payload=payload)
    np.testing.assert_array_equal(view.raw, np.full(raw.shape, delta) if operation == "difference" else raw + delta)
    assert not view.raw.flags.writeable
    assert view.metadata_raw["point_images"] == policy
    assert view.source_path is None and view.source_frame_index is None
    with h5py.File(path, "r") as handle:
        source = handle["entry/stitch/frames/scan_2/frame_0000/source"]
        assert set(source) == {"constituents"}
        assert json.loads(source.attrs["point_images"]) == policy
        members = source["constituents"]
        assert list(members) == [f"{index:04d}" for index in range(len(expected_paths))]
        assert [members[name]["path"].asstr()[()] for name in members] == expected_paths


@pytest.mark.parametrize("mutation", [
    "missing", "extra", "external", "reordered", "policy", "snapshot", "manifest-policy", "first-pointer",
])
def test_grouped_bound_artifact_rejects_partial_or_changed_operands(tmp_path, mutation):
    policy = dict(images_per_point=2, operation="difference", image_index=1, reverse=False)
    path, _, _, _ = _publish(tmp_path, point_images=policy)
    with h5py.File(path, "r+") as handle:
        stitch = handle["entry/stitch"]
        source = stitch["frames/scan_2/frame_0000/source"]
        members = source["constituents"]
        if mutation == "missing":
            del members["0001"]
        elif mutation == "extra":
            members["0002"] = members["0000"]
        elif mutation == "external":
            del members["0001"]
            members["0001"] = h5py.ExternalLink("not-opened.h5", "/operand")
        elif mutation == "reordered":
            members.move("0000", "temporary")
            members.move("0001", "0000")
            members.move("temporary", "0001")
        elif mutation == "policy":
            source.attrs.modify("point_images", np.bytes_(canonical_frame_manifest(dict(policy, reverse=True)).encode()))
        elif mutation == "snapshot":
            members["0001"].attrs.modify("file_size", 7)
        elif mutation == "first-pointer":
            source.create_dataset("path", data=np.bytes_(b"frame.raw"))
        else:
            manifest = json.loads(bytes(stitch["contributing_frames_json"][()]))
            manifest["records"][0]["point_images"]["reverse"] = True
            del stitch["contributing_frames_json"]
            stitch.create_dataset("contributing_frames_json", data=np.bytes_(canonical_frame_manifest(manifest).encode()))
    with pytest.raises(AnalysisArtifactInvalid):
        read_analysis_artifact(path)


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_grouped_original_operand_unavailable_never_returns_partial_raw(tmp_path, mutation):
    import os
    policy = dict(images_per_point=2, operation="average", image_index=1, reverse=False)
    path, raw, project, receipt = _publish(tmp_path, point_images=policy)
    payload = read_analysis_artifact(path, expected_receipt=receipt)
    original = project / "second.raw"
    if mutation == "missing":
        original.unlink()
    else:
        state = original.stat()
        (raw + 1000).tofile(original)
        os.utime(original, ns=(state.st_atime_ns, state.st_mtime_ns + 1_000_000))
    view = read_contributing_frame_view(path, 0, scan="2", full_raw=True, artifact_payload=payload)
    assert view.raw is None
    assert view.thumbnail.shape == raw.shape
    np.testing.assert_array_equal(read_analysis_artifact(path).intensity, payload.intensity)
