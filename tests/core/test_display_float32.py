"""DISPLAY-FLOAT32-128: display arrays in float32 (design note v3 and the maintainer's choices in the task's
DECISIONS.txt: "Nothing that we do requires more precision than what float32 offers"; range "Refuse with a reason").

Rows named ``preservation`` hold on dev 7ed95ef7 too; every other row is new behaviour and fails there.
"""
from __future__ import annotations

import hashlib
import json
import tracemalloc
from dataclasses import replace

import h5py
import numpy as np
import pytest

from xrd_tools.core import FrameView
from xrd_tools.io import frame_preview
from xrd_tools.io import viewer_2d as api
from tests.core import test_viewer_2d as a0
from tests.core.test_frame_preview import _read_key, _write_processed

F32 = np.dtype(np.float32)
FULL = __import__("xrd_tools.session.hydration", fromlist=["HydrationPurpose"]).HydrationPurpose.FULL


def _read(path, label=0, policy=None):
    catalog = api.catalog_viewer_2d(path, policy=policy)
    return api.read_viewer_2d_frame(catalog, label, policy=policy)


def _assert_float32_canonical(frame, expected):
    array, provenance = frame.array, frame.provenance
    assert array.dtype == F32 and array.ndim == 2
    assert array.flags.c_contiguous and array.flags.owndata and not array.flags.writeable and array.base is None
    assert provenance.canonical_dtype == F32.str
    assert provenance.canonical_nbytes == array.nbytes == 4 * array.size
    assert provenance.canonical_sha256 == hashlib.sha256(array.tobytes()).hexdigest()
    np.testing.assert_array_equal(array, np.asarray(expected).astype(F32))


def _refusal(operation, *args, **kwargs):
    with pytest.raises(api.Viewer2DReadError) as caught:
        operation(*args, **kwargs)
    return caught.value


# F1: the 2D Viewer's canonical frame is float32 on every route; source facts keep their source meaning.

def test_tiff_and_edf_frames_are_float32_with_native_source_facts(tmp_path):
    native = np.array([[0, 1, 65535], [12345, 2, 3]], dtype=np.uint16)
    for family, name in (("tiff", "frame.tif"), ("fabio", "frame.edf")):
        path = tmp_path / name
        a0._write_selected_source(path, family, native)
        frame = _read(path)
        _assert_float32_canonical(frame, native)
        assert frame.provenance.source_dtype == native.dtype.str
        assert frame.provenance.source_nbytes == native.nbytes


def test_numpy_frames_are_float32_rounded_from_uint32_and_float64(tmp_path):
    counts = np.array([[16777216, 16777217], [16777219, 4294967295]], dtype=np.uint32)
    path = tmp_path / "counts.npy"
    np.save(path, counts)
    frame = _read(path)
    _assert_float32_canonical(frame, counts)
    assert frame.array[0, 1] == np.float32(16777216) and frame.array[1, 1] == np.float32(2**32)
    assert frame.provenance.source_dtype == counts.dtype.str and frame.provenance.source_nbytes == counts.nbytes
    assert frame.provenance.source_sha256 == hashlib.sha256(counts.tobytes()).hexdigest()

    values = np.array([[0.1, 1e30], [-2.5, 3.4e38]])
    path = tmp_path / "values.npy"
    np.save(path, values)
    frame = _read(path)
    _assert_float32_canonical(frame, values)
    assert np.isfinite(frame.array).all()
    assert frame.provenance.source_dtype == values.dtype.str and frame.provenance.source_nbytes == values.nbytes

    path = tmp_path / "stack.npz"
    stack = np.arange(24, dtype=np.int32).reshape(2, 3, 4) - 5
    a0._npz(path, [("image.npy", a0._npy_bytes(stack))])
    _assert_float32_canonical(_read(path, 1), stack[1])


def test_csv_frame_is_float32_and_keeps_its_float64_source_facts(tmp_path):
    path = tmp_path / "matrix.csv"
    path.write_text("0.1,2,3\n4,5.5,-6\n")
    catalog = api.catalog_viewer_2d(path)
    assert catalog.source_dtype == np.dtype(float).str
    frame = api.read_viewer_2d_frame(catalog, 0)
    _assert_float32_canonical(frame, [[0.1, 2, 3], [4, 5.5, -6]])
    assert frame.provenance.source_dtype == np.dtype(float).str
    assert frame.provenance.source_nbytes == 8 * 6


def test_raw_hdf_processed_raw_and_thumbnail_frames_are_float32(tmp_path):
    native = np.array([[1, 2, 70000], [16777217, 5, 6]], dtype=np.uint32)
    raw = tmp_path / "source.nxs"
    a0._write_hdf_stack(raw, native[np.newaxis])
    _assert_float32_canonical(_read(raw), native)

    thumb = np.array([[0, 7, 255], [128, 64, 1]], dtype=np.uint8)
    processed = tmp_path / "processed.nexus"
    with h5py.File(processed, "w") as handle:
        a0._processed_source(handle, 2, str(raw.resolve()), 0, "/entry/data/data")
        a0._processed_thumbnail(handle, 7, thumb, vmin=-1.5, vmax=3.0)
        a0._finalize_processed(handle)
    policy = api.Viewer2DFormatPolicy(source_root=str(tmp_path))
    dependent = _read(processed, 2, policy)
    _assert_float32_canonical(dependent, native)
    assert dependent.provenance.source_kind is api.Viewer2DSourceKind.PROCESSED_RAW
    assert dependent.provenance.source_dtype == native.dtype.str

    catalog = api.catalog_viewer_2d(processed, policy=policy)
    thumbnail = api.read_viewer_2d_frame(catalog, 7, policy=policy)
    expected = -1.5 + thumb.astype(float) / 255.0 * 4.5
    array = thumbnail.array
    assert array.dtype == F32 and array.flags.owndata and not array.flags.writeable
    assert thumbnail.provenance.canonical_dtype == F32.str
    assert thumbnail.provenance.canonical_sha256 == hashlib.sha256(array.tobytes()).hexdigest()
    np.testing.assert_allclose(array, expected, rtol=1e-6, atol=1e-6)
    assert thumbnail.provenance.degraded_thumbnail
    assert thumbnail.provenance.source_dtype == thumb.dtype.str
    assert thumbnail.provenance.source_sha256 == hashlib.sha256(thumb.tobytes()).hexdigest()


def test_thumbnail_with_extreme_endpoints_dequantises_to_its_finite_values(tmp_path):
    """Endpoints near opposite ends of float32's range: the span overflows float32, the decoded values do not. The
    frame equals the float64 dequantisation rounded once, and the read makes no float64 frame (REVIEW-2 R3)."""
    raw = tmp_path / "source.nxs"
    a0._write_hdf_stack(raw, np.zeros((1, 2, 3), dtype=np.uint16))
    thumb = (np.arange(2048 * 2048, dtype=np.uint32) % 65536).astype(np.uint16).reshape(2048, 2048)
    vmin, vmax = -3.0e38, 3.0e38
    processed = tmp_path / "processed.nexus"
    with h5py.File(processed, "w") as handle:
        a0._processed_source(handle, 2, str(raw.resolve()), 0, "/entry/data/data")
        a0._processed_thumbnail(handle, 7, thumb, vmin=vmin, vmax=vmax)
        a0._finalize_processed(handle)
    policy = api.Viewer2DFormatPolicy(source_root=str(tmp_path))
    catalog = api.catalog_viewer_2d(processed, policy=policy)
    tracemalloc.start()
    try:
        frame = api.read_viewer_2d_frame(catalog, 7, policy=policy)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    reference = (vmin + thumb.astype(float) / 65535.0 * (vmax - vmin)).astype(F32)
    assert np.isfinite(reference).all() and reference.max() > 2.9e38 and reference.min() < -2.9e38
    _assert_float32_canonical(frame, reference)
    assert peak < 12 * thumb.size, f"{peak / thumb.size:.1f} bytes per pixel"


def test_provenance_pins_float32_canonical_dtype_and_bytes(tmp_path):
    path = tmp_path / "frame.npy"
    np.save(path, np.arange(6, dtype=np.uint16).reshape(2, 3))
    frame = _read(path)
    with pytest.raises(TypeError):
        replace(frame.provenance, canonical_dtype=np.dtype(float).str)
    with pytest.raises(TypeError):
        replace(frame.provenance, canonical_nbytes=8 * 6)
    with pytest.raises(TypeError):
        api.Viewer2DFrame(frame.catalog_identity, frame.label,
                          np.array(frame.array, dtype=float), frame.provenance)


def test_values_beyond_float32_range_are_refused_with_a_reason(tmp_path):
    path = tmp_path / "huge.npy"
    np.save(path, np.array([[1.0, 1e39], [2.0, 3.0]]))
    refused = _refusal(_read, path)
    assert refused.code is api.Viewer2DRefusalCode.FORMAT_INVALID
    assert "float32 display range" in str(refused)


def test_integer_limit_refusal_no_longer_claims_exactness(tmp_path):
    path = tmp_path / "wide.npy"
    np.save(path, np.array([[0, 2**53 + 1]], dtype=np.int64))
    refused = _refusal(_read, path)
    assert refused.code is api.Viewer2DRefusalCode.FORMAT_INVALID
    assert str(refused) == "integer source exceeds the 2D Viewer's ±2^53 limit"


# F6: hashing reads the frame's buffer; no whole-frame bytes copy.

def test_viewer_read_hashes_without_whole_frame_copies(tmp_path):
    value = (np.arange(1024 * 1024, dtype=np.uint32) % 65521).astype(np.uint16).reshape(1024, 1024)
    path = tmp_path / "frame.npy"
    np.save(path, value)
    catalog = api.catalog_viewer_2d(path)
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        frame = api.read_viewer_2d_frame(catalog, 0)
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    pixels = value.size
    # The read buffer and source (2 B/px each), the float32 canonical (4 B/px) and no tobytes copies; dev's float64
    # canonical plus its whole-frame tobytes copies peak near 18-20 B/px.
    assert peak < 12 * pixels, peak / pixels
    assert frame.provenance.canonical_sha256 == hashlib.sha256(frame.array.tobytes()).hexdigest()
    assert frame.provenance.source_sha256 == hashlib.sha256(value.tobytes()).hexdigest()


# FrameView keeps a float32 display array it is given; integer and float64 raw still become float64.

def test_frame_view_keeps_float32_raw_by_identity_and_converts_the_rest():
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    value.setflags(write=False)
    view = FrameView(label=0, raw=value)
    assert view.raw is value
    assert replace(view, raw=value).raw is value
    for other in (np.arange(6, dtype=np.uint16), np.arange(6, dtype=np.int64), np.arange(6.0)):
        converted = FrameView(label=0, raw=other.reshape(2, 3)).raw
        assert converted.dtype == np.dtype(float) and not converted.flags.writeable
        np.testing.assert_array_equal(converted, other.reshape(2, 3))
    assert FrameView(label=0, intensity_2d=value).intensity_2d.dtype == np.dtype(float)


# F2: the Browse/Run detector preview is one float32 copy, classified on native values first.

def test_full_preview_is_float32_and_classified_on_native_counts(tmp_path):
    raw = np.array([[16777216, 16777217, 16777218, 16777219],
                    [0, 1, 2, 4294967295]], dtype=np.uint32)
    processed, _ = _write_processed(tmp_path, thumbnail=False, raw=raw)
    projection = frame_preview.DetectorPreviewProjection.without_static_mask(
        apply_threshold=True, threshold_min=16777217, mask_saturation=False)
    result = frame_preview.read_frame_preview(_read_key(processed, FULL), detector_projection=projection)
    masked = raw < 16777217  # native: 16777217 is kept; on rounded float32 data it would read 16777216 and be masked
    assert result.raw.dtype == F32 and not result.raw.flags.writeable
    np.testing.assert_array_equal(np.isnan(result.raw), masked)
    np.testing.assert_array_equal(result.raw[~masked], raw[~masked].astype(F32))


def test_masked_detector_makes_one_float32_copy():
    source = (np.arange(1024 * 1024, dtype=np.uint32) % 65521).astype(np.uint16).reshape(1024, 1024)
    projection = frame_preview.DetectorPreviewProjection.without_static_mask(
        mask_saturation=True, saturation_ceiling=65535)
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        result = frame_preview._masked_detector(source, projection)
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    assert result.dtype == F32 and not result.flags.writeable and result.flags.owndata
    # One float32 frame (4 B/px) plus the boolean masks; dev made two float64 frames (16 B/px).
    assert peak < 8 * source.size, peak / source.size
    expected = source.astype(F32)
    expected[source >= 65535] = np.nan
    np.testing.assert_array_equal(result, expected)


def test_point_image_full_preview_is_float32(tmp_path):
    from tests.core.test_point_images_int import _fixture, _plan, _run
    from xrd_tools.session.hydration import HydrationPurpose, HydrationReadKey, HydrationScope
    source, native, calibration = _fixture(tmp_path, "difference")
    target = tmp_path / "preview.nexus"
    _run(source, calibration, _plan(mask_saturation=False), target)
    key = HydrationReadKey(HydrationScope("point-int", "scan-5", str(tmp_path), 1), str(target), 0,
                           HydrationPurpose.FULL, None)
    projection = frame_preview.DetectorPreviewProjection.without_static_mask(
        mask_saturation=True, saturation_ceiling=65535)
    preview = frame_preview.read_frame_preview(key, detector_projection=projection)
    expected = native[1].astype(float) - native[0].astype(float)
    expected[2, 3] = np.nan
    assert preview.raw.dtype == F32 and not preview.raw.flags.writeable
    np.testing.assert_array_equal(preview.raw, expected.astype(F32))


def test_preview_beyond_float32_range_is_unavailable_with_the_reason(tmp_path):
    raw = np.array([[1.0, 1e39], [2.0, 3.0]])
    processed, _ = _write_processed(tmp_path, thumbnail=False, raw=raw)
    projection = frame_preview.DetectorPreviewProjection.without_static_mask(mask_saturation=False)
    result = frame_preview.read_frame_preview(_read_key(processed, FULL), detector_projection=projection)
    assert result.raw is None
    assert "float32 display range" in result.detector_diagnostic


# F0 (preservation): Stitch's scientific raw stays native and float64, a float32 source included.

def _stitched_nexus_group(tmp_path, frames, saturation):
    """Two Bluesky NeXus scans whose detector frames are ``frames``, stitched with Auto conditioning, in the pattern of
    test_native_stitch_operation's NeXus group row; returns the committed payload and its output path."""
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    from tests.core.test_stitch_operation import _goniometer_record
    from xrd_tools.analysis.module_transaction import (
        MetadataColumnSelector, ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceGroupReceipt,
        ModuleSourceReceipt)
    from xrd_tools.analysis.scan_operations import AnalysisDisposition, MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, StitchOperationPlan, capture_stitch_geometry,
        prepare_stitch_operation, run_stitch_operation)
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
    from xrd_tools.io.output_path import resolve_finite_output_target
    members = []
    for index in range(2):
        path = _write_bluesky_nxwriter(tmp_path / f"member{index}.nxs")
        with h5py.File(path, "r+") as stream:
            stream.move("entry", "sample")
            stream.attrs["default"] = "sample"
            configuration = stream["sample/instrument/bluesky/metadata/configuration/eiger/data"]
            configuration["eiger_cam_wavelength"][()] = 1.0
            configuration["eiger_cam_photon_energy"][()] = 12398.41984
            stream.require_group("sample/instrument/detector").create_dataset("saturation_value", data=saturation)
            stream["sample/data"].create_dataset("del", data=np.linspace(5., 10., 5))
            stream["sample/data"].create_dataset("nu", data=np.zeros(5))
            for name in ("del", "nu"):
                positioner = stream["sample/instrument/positioners"].create_group(name)
                positioner.attrs["NX_class"] = "NXpositioner"
                positioner.create_dataset("value", data=stream[f"sample/data/{name}"][()])
            del stream["sample/data/eiger_image"]
            del stream["sample/instrument/detectors/eiger/data"]
            detector = stream["sample/data"].create_dataset("eiger_image", data=frames)
            detector.attrs["signal_type"] = "detector"
            stream["sample/instrument/detectors/eiger/data"] = detector
        table = run_metadata_table(MetadataTablePlan(SourceSpec(path, SourceKind.NEXUS_STACK, entry="sample")))
        assert table.disposition is AnalysisDisposition.COMPLETED, table
        members.append(ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.STITCH, selected_labels=(0, 4),
            resolved_selectors=(MetadataColumnSelector("del"), MetadataColumnSelector("nu"))))
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps(_goniometer_record()))
    plan = StitchOperationPlan(capture_stitch_geometry(StitchGeometryInput(geometry,
        StitchGeometryKind.PYFAI_GONIOMETER_JSON, source_motors=(("del_angle", "del"), ("nu_angle", "nu")))),
        native_records=True, unit="2th_deg", npt_1d=24, conditioning=StitchConditioning("auto"))
    output = ModuleOutputRequest(resolve_finite_output_target(tmp_path, "nexus-group", operation_token="stitch-1d"),
        AnalysisArtifactKind.STITCH_1D, artifact_family="nexus-group")
    request = prepare_stitch_operation(ModuleSourceGroupReceipt.from_members(tuple(members), kind=ModuleKind.STITCH),
                                       output, plan, project_root=tmp_path)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    return result.payload, output.target


@pytest.mark.parametrize("dtype", (np.uint32, np.float32))
def test_stitch_raw_stays_native_float64_around_2_to_the_24(tmp_path, dtype):
    """Preservation: the full Raw read, which the contributor's conditioning and reconstruction consume, is the native
    values in float64, and the display preview's saturation is decided on them. A uint32 16777217 stays exact and sits
    at the ceiling (float32 would make it 16777216, below the ceiling); a float32 source is read as float64 too."""
    from xrd_tools.analysis.stitch_operation import stitch_contributor_raw_preview
    from xrd_tools.io.frame_view import read_contributing_frame_view
    frames = (np.arange(5 * 195 * 487, dtype=np.uint64) % 7 + 16777214).reshape(5, 195, 487).astype(dtype)
    ceiling = 16777217
    payload, target = _stitched_nexus_group(tmp_path, frames, ceiling)
    for index, row in enumerate(payload.contributing_frames):
        scan, label = row["key"].split("/")[0][5:], row["identity"]["frame_index"]
        assert np.dtype(row["source_snapshot"]["raw_dtype"]) == np.dtype(dtype)
        frame = read_contributing_frame_view(target, label, scan=scan, full_raw=True, artifact_payload=payload)
        expected = frames[(0, 4)[index % 2]].astype(float)
        assert frame.raw.dtype == np.dtype(float)
        np.testing.assert_array_equal(frame.raw, expected)
        preview = stitch_contributor_raw_preview(payload, frame.raw, label, scan=scan)
        np.testing.assert_array_equal(np.isfinite(preview), expected < ceiling)
        if dtype is np.uint32:
            assert (expected == ceiling).any() and (expected == ceiling - 1).any()
