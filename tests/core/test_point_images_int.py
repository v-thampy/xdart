"""Frozen direct Int oracle for confirmed multiple native images per point."""
from dataclasses import replace
import json

import h5py
import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry.diffractometer import DetectorCalibration
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.calibration import detector_calibration_to_integrator
from xrd_tools.io import open_scan
from xrd_tools.io.read import get_raw_frame
from xrd_tools.reduction import Frame, Integration1DPlan, Integration2DPlan, NexusSink, ReductionPlan, Scan, run_reduction
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources import open_source
from xrd_tools.sources.point_images import PointImagePolicy


def _fixture(tmp_path, operation="average", *, reverse=False):
    from fabio.tifimage import TifImage
    spec = tmp_path / "points"
    spec.write_text("#F points\n#E 1\n#O0 th\n\n#S 5 ascan th 0 2 2 1\n#P0 0\n#N 2\n#L th  i0\n0 2\n1 4\n2 8\n")
    native = []
    for index, value in enumerate((40, 20, 70, 30, 110, 50)):
        image = np.full((16, 16), value, np.uint16)
        image[0, 0] = value + 3
        image[2, 3] = 65535 if index == 0 else value
        image[4, 5] = 1 if index == 2 else value
        path = tmp_path / f"points_scan5_{index:04d}.tif"
        TifImage(data=image).write(str(path))
        native.append(image)
    policy = PointImagePolicy(2, operation, image_index=2, reverse=reverse)
    source = SourceSpec(spec, SourceKind.SPEC, options={"scan": "5.1", "image_dir": str(tmp_path), "require_complete_images": True, "point_images": policy.to_dict()})
    poni = PONI(.2, .001, .001, 0., 0., 0., 1e-10, "Pilatus100k")
    calibration = DetectorCalibration(poni, {"max_shape": [16, 16], "orientation": 3})
    return source, native, calibration


def _plan(**values):
    one = Integration1DPlan(npt=8, method="histogram", error_model="poisson", monitor_key="i0")
    two = Integration2DPlan(npt_rad=8, npt_azim=6, method="histogram", error_model="poisson", monitor_key="i0")
    return ReductionPlan(one, two, **values)


def _run(source_spec, calibration, plan, target=None, *, accepted=None):
    from xrd_tools.reduction.provenance_config import _integration_1d_args, _integration_2d_args
    from xrd_tools.session.run_configuration import ThresholdIntent
    from xrd_tools.sources.execution_graph import qualify_source_execution_graph, source_execution_projection, source_snapshots_projection
    graph = qualify_source_execution_graph(source_spec)
    source = open_source(source_spec)
    try:
        scan = source.to_scan(poni=calibration.poni, integrator=detector_calibration_to_integrator(calibration))
        intent = RunIntent(source_spec=source_spec, processing_mode="Int 1D + 2D", poni_values=calibration.poni.to_dict(),
            bai_1d_args=_integration_1d_args(plan.integration_1d, None), bai_2d_args=_integration_2d_args(plan.integration_2d, None),
            threshold=ThresholdIntent(apply_threshold=plan.threshold_min is not None or plan.threshold_max is not None,
                threshold_min=plan.threshold_min, threshold_max=plan.threshold_max, mask_saturation=plan.mask_saturation))
        provenance = intent.freeze().as_provenance()
        if target is not None:
            from pyFAI.io.ponifile import PoniFile
            from xdart.gui.tabs.scattering.contracts import SourceCapture, StartCapture
            from xdart.gui.tabs.scattering.events import RequestId
            from xdart.gui.tabs.scattering.output_preflight import OutputCandidate
            from xrd_tools.session.intent_store import RunIntentSnapshot
            from xrd_tools.session.scientific_assets import load_scientific_assets
            poni_path = target.with_suffix(".poni")
            with poni_path.open("w") as stream:
                PoniFile(scan.integrator).write(stream)
            intent.poni_file = str(poni_path)
            intent.save_path = str(target)
            capture = StartCapture(RequestId(1), 1, RunIntentSnapshot(0, intent),
                SourceCapture(RequestId(1), 0, source_spec))
            candidate = OutputCandidate.from_start_capture(capture, load_scientific_assets(intent))
            if accepted is not None:
                accepted.append(candidate)
            provenance = candidate._configuration.as_provenance()
            provenance["scientific_signature"] = candidate.processing_mapping()
        lineage = None
        if target is not None:
            from xrd_tools.io.append import AppendIntent, science_fingerprint
            from xdart.gui.tabs.scattering.adapters.dynamic_output import _science_projection
            from xrd_tools.sources.execution_graph import append_source_from_execution_graph, stable_lineage_projection
            lineage = AppendIntent("entry", str(target.parent), science_fingerprint(stable_lineage_projection(graph, target=target)),
                science_fingerprint(_science_projection(candidate._configuration, candidate.processing_mapping())), ("1d:default", "2d:default"),
                append_source_from_execution_graph(graph, generation=1), tuple(scan.frame_indices))
        sink = None if target is None else NexusSink(target, create_new=True, source_base=target.parent, run_configuration_provenance=provenance, source_execution_provenance=source_execution_projection(graph), source_snapshots_provenance=source_snapshots_projection(graph, writer=True), same_run_intent=lineage)
        return run_reduction(plan, scan, sink, executor=1)
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()


@pytest.mark.parametrize("operation,reverse", [("average", False), ("difference", False), ("difference", True), ("select", False)])
@pytest.mark.parametrize("dark_value", [None, 3.])
def test_real_three_point_int_signal_mask_sigma_and_normalization(tmp_path, operation, reverse, dark_value):
    source, native, calibration = _fixture(tmp_path, operation, reverse=reverse)
    dark = None if dark_value is None else np.full((16, 16), dark_value)
    plan = _plan(dark=dark, threshold_min=5., threshold_max=65534.)
    actual = _run(source, calibration, plan)
    assert actual.n_processed == 3 and tuple(actual.frames) == (0, 1, 2)
    for point in range(3):
        indices = (2 * point + 1,) if operation == "select" else (2 * point, 2 * point + 1)
        operands = [native[index].astype(float) for index in indices]
        invalid = np.logical_or.reduce([(operand < 5) | (operand >= 65535) for operand in operands])
        if operation == "average":
            signal = (operands[0] + operands[1]) / 2
            variance = (np.maximum(operands[0], 1) + np.maximum(operands[1], 1)) / 4
        elif operation == "difference":
            signal = operands[0] - operands[1] if reverse else operands[1] - operands[0]
            variance = np.maximum(operands[0], 1) + np.maximum(operands[1], 1)
        else:
            signal, variance = operands[0], np.maximum(operands[0], 1)
        signal[invalid] = np.nan
        variance[invalid] = np.nan
        if dark is not None and operation != "difference":
            signal -= dark
            variance += dark
        reference_plan = replace(plan, dark=None, mask_saturation=False, threshold_min=None, threshold_max=None, integration_1d=replace(plan.integration_1d, error_model=None, extra={"variance": variance}), integration_2d=replace(plan.integration_2d, error_model=None, extra={"variance": variance}))
        metadata = {"i0": (2., 4., 8.)[point]}
        if operation == "difference":
            metadata["point_images"] = {"operation": "difference"}
        expected = run_reduction(reference_plan, Scan("reference", [Frame(point, image=signal, metadata=metadata)], poni=calibration.poni, integrator=detector_calibration_to_integrator(calibration)), executor=1).frames[point]
        reduced = actual.frames[point]
        for first, second in ((reduced.result_1d, expected.result_1d), (reduced.result_2d, expected.result_2d)):
            np.testing.assert_array_equal(first.intensity, second.intensity)
            np.testing.assert_array_equal(first.sigma, second.sigma)
        assert reduced.metadata["th"] == point and reduced.metadata["i0"] == (2., 4., 8.)[point]


def test_combined_raw_records_guard_old_pointer_and_all_secondary_dependencies(tmp_path):
    source, native, calibration = _fixture(tmp_path, "difference")
    target = tmp_path / "combined.nexus"
    _run(source, calibration, _plan(mask_saturation=False), target)
    with h5py.File(target, "r") as document:
        record = document["entry/frames/frame_0000/source"]
        assert "path" not in record
        assert tuple(record["constituents"]) == ("0000", "0001")
        assert json.loads(record.attrs["point_images"])["operation"] == "difference"
        run = json.loads(document["entry/reduction/config/run_configuration"][()])
        assert run["point_images"]["operation"] == "difference"
    np.testing.assert_array_equal(get_raw_frame(target, 1, allow_thumbnail=False), native[3].astype(float) - native[2].astype(float))
    reader = open_scan(target)
    np.testing.assert_array_equal(reader.load_frame(1), native[3].astype(float) - native[2].astype(float))
    before = target.read_bytes()
    secondary = tmp_path / "points_scan5_0003.tif"
    secondary.unlink()
    with pytest.raises((ValueError, KeyError, FileNotFoundError), match="source|constituent|raw"):
        get_raw_frame(target, 1, allow_thumbnail=False)
    assert target.read_bytes() == before


@pytest.mark.parametrize("damage", ["mutation", "deletion"])
def test_full_raw_preview_is_combined_read_only_and_refuses_secondary_damage(tmp_path, damage):
    from xrd_tools.io.frame_preview import DetectorPreviewProjection, read_frame_preview
    from xrd_tools.io.read import observe_resolved_raw_source
    from xrd_tools.session.hydration import HydrationReadKey, HydrationPurpose, HydrationScope
    source, native, calibration = _fixture(tmp_path, "difference")
    target = tmp_path / "preview.nexus"
    _run(source, calibration, _plan(mask_saturation=False), target)
    assert observe_resolved_raw_source(target, 0).source is not None
    key = HydrationReadKey(HydrationScope("point-int", "scan-5", str(tmp_path), 1), str(target), 0, HydrationPurpose.FULL, None)
    projection = DetectorPreviewProjection.without_static_mask(mask_saturation=True, saturation_ceiling=65535)
    preview = read_frame_preview(key, detector_projection=projection)
    expected = native[1].astype(float) - native[0].astype(float)
    expected[2, 3] = np.nan
    np.testing.assert_array_equal(preview.raw, expected)
    assert not preview.raw.flags.writeable
    before = target.read_bytes()
    secondary = tmp_path / "points_scan5_0001.tif"
    if damage == "deletion":
        secondary.unlink()
    else:
        secondary.write_bytes(secondary.read_bytes() + b"changed")
    refused = read_frame_preview(key, detector_projection=projection)
    assert refused.raw is None and refused.detector_diagnostic
    assert target.read_bytes() == before


def test_point_policy_frozen_run_guard_is_additive_and_changes_science_identity(tmp_path):
    source, _, _ = _fixture(tmp_path)
    plain = RunIntent(source_spec=replace(source, options={key: value for key, value in source.options.items() if key != "point_images"})).freeze()
    grouped = RunIntent(source_spec=source).freeze()
    assert "point_images" not in plain.as_provenance()
    assert grouped.as_provenance()["point_images"] == dict(source.options["point_images"])
    assert grouped.fingerprint != plain.fingerprint


def test_append_refuses_changed_point_operation_without_mutating_target(tmp_path):
    from copy import deepcopy
    from xrd_tools.io.append import AppendDisposition, decode_committed_append_prefix, qualify_append, science_fingerprint
    source, _, calibration = _fixture(tmp_path, "average")
    target = tmp_path / "append.nexus"
    from xdart.gui.tabs.scattering.adapters.dynamic_output import _science_projection
    from xrd_tools.session.run_configuration import FrozenSourceSpec
    accepted = []
    _run(source, calibration, _plan(mask_saturation=False), target, accepted=accepted)
    before = target.read_bytes()
    with h5py.File(target, "r") as stream:
        prefix = decode_committed_append_prefix(stream, entry="entry")
        signature = json.loads(stream["entry/reduction/config/run_configuration"][()])["scientific_signature"]
    changed_source = replace(source, options={**source.options, "point_images": {**source.options["point_images"], "operation": "difference"}})
    changed_configuration = replace(accepted[0]._configuration, source=FrozenSourceSpec.from_source(changed_source))
    changed = {**changed_configuration.as_provenance(), "accepted_scientific_assets": deepcopy(signature["accepted_scientific_assets"])}
    actual_projection = _science_projection(changed_configuration, changed)
    assert actual_projection["point_images"]["operation"] == "difference"
    decision = qualify_append(target, replace(prefix.intent, science_fingerprint=science_fingerprint(actual_projection)))
    assert decision.disposition is AppendDisposition.REFUSE
    assert target.read_bytes() == before


def test_int_image_series_spec_metadata_has_three_grouped_points(tmp_path):
    from xrd_tools.sources.selection import image_series_spec
    source, _, calibration = _fixture(tmp_path)
    selected = image_series_spec(tmp_path / "points_scan5_0000.tif", metadata_format="spec")
    from xrd_tools.sources.point_images import inspect_point_images, PointImagesRequired
    from xrd_tools.sources.execution_graph import qualify_source_execution_graph
    detection = inspect_point_images(selected)
    assert (detection.point_count, detection.image_count, detection.images_per_point) == (3, 6, 2)
    single = replace(selected, options={**selected.options, "files": selected.options["files"][:1]})
    assert inspect_point_images(single) is None
    with pytest.raises(PointImagesRequired):
        qualify_source_execution_graph(selected)
    selected = replace(selected, options={**selected.options, "point_images": dict(source.options["point_images"])})
    opened = open_source(selected)
    try:
        assert opened.frame_indices == [1, 2, 3]
        assert [opened.metadata_for(label)["i0"] for label in opened.frame_indices] == [2., 4., 8.]
        chunks = list(opened.iter_point_statistics(2, poisson=True, native_transform=lambda image: image))
        assert [labels for _statistics, labels in chunks] == [[1, 2], [3]]
        assert all(item.variance is not None for statistics, _labels in chunks for item in statistics)
    finally:
        close = getattr(opened, "close", None)
        if close is not None:
            close()
    result = _run(selected, calibration, _plan(mask_saturation=False))
    assert result.n_processed == 3 and tuple(result.frames) == (1, 2, 3)


def test_shared_dark_covariance_uses_net_coefficient_squared():
    from xrd_tools.reduction.core import _poisson_variance
    signal = np.full((2, 2), 20.)
    dark = np.full((2, 2), 3.)
    plan = _plan(dark=dark)
    frame = Frame(0, background=np.full((2, 2), 8.), background_variance_factor=.25, background_scale=.5,
                  point_variance=np.full((2, 2), 7.), point_exposure_multiplier=1.)
    np.testing.assert_array_equal(_poisson_variance(signal, frame, plan), np.full((2, 2), 7 + .5 ** 2 * 3 + .25 * 8))
    frame.point_exposure_multiplier = 0.
    np.testing.assert_array_equal(_poisson_variance(signal, frame, plan), np.full((2, 2), 7 + .5 ** 2 * 3 + .25 * 8))


def test_real_average_flat_masks_native_pixels_and_scales_sigma_by_coefficient(tmp_path):
    source, native, calibration = _fixture(tmp_path, "average")
    flat = np.full((16, 16), 2.)
    flat[7, 7], flat[8, 8] = 0., np.nan
    dark = np.full((16, 16), 3.)
    plan = _plan(flat=flat, dark=dark, threshold_min=5., threshold_max=65534.)
    actual = _run(source, calibration, plan)
    for point in range(3):
        a, b = native[2 * point].astype(float), native[2 * point + 1].astype(float)
        invalid = (a < 5) | (b < 5) | (a > 65534) | (b > 65534) | ~np.isfinite(flat) | (flat <= 0)
        signal, variance = (a + b) / 2, (a + b) / 4 + dark
        signal[invalid] = variance[invalid] = np.nan
        reference_plan = replace(plan, flat=None, threshold_min=None, threshold_max=None, mask_saturation=False,
            integration_1d=replace(plan.integration_1d, error_model=None, extra={"variance": variance}),
            integration_2d=replace(plan.integration_2d, error_model=None, extra={"variance": variance}))
        expected = run_reduction(reference_plan, Scan("reference", [Frame(point, image=signal, metadata={"i0": (2., 4., 8.)[point]})],
            poni=calibration.poni, integrator=detector_calibration_to_integrator(calibration)), executor=1).frames[point]
        for first, second in ((actual.frames[point].result_1d, expected.result_1d), (actual.frames[point].result_2d, expected.result_2d)):
            np.testing.assert_array_equal(first.intensity, second.intensity / 2)
            np.testing.assert_array_equal(first.sigma, second.sigma / 2)


def test_real_difference_thumbnail_cancels_one_shared_dark(tmp_path):
    from xrd_tools.io.nexus_record import make_thumbnail_array, quantize_thumbnail
    source, native, calibration = _fixture(tmp_path, "difference")
    target = tmp_path / "difference-dark.nexus"
    _run(source, calibration, _plan(dark=np.full((16, 16), 3.), threshold_min=5., threshold_max=65534.), target)
    with h5py.File(target, "r") as document:
        for point in range(3):
            a, b = native[2 * point].astype(float), native[2 * point + 1].astype(float)
            signal = b - a
            signal[(a < 5) | (b < 5) | (a > 65534) | (b > 65534)] = np.nan
            expected, lut = quantize_thumbnail(make_thumbnail_array(signal))
            thumbnail = document[f"entry/frames/frame_{point:04d}/thumbnail"]
            np.testing.assert_array_equal(thumbnail[()], expected)
            assert (thumbnail.attrs["vmin"], thumbnail.attrs["vmax"], thumbnail.attrs["dtype"]) == lut


@pytest.mark.parametrize("operation", ["average", "difference", "select"])
def test_reintegrate_grouped_real_int_is_bit_exact_and_refuses_secondary_mutation(tmp_path, operation):
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor
    source, _, calibration = _fixture(tmp_path, operation)
    target = tmp_path / "original.nexus"
    _run(source, calibration, _plan(mask_saturation=False), target)
    before = target.read_bytes()
    preparation = {"api_version": 1,
        "selected_plan": {"version": 1, "dimension": "1d", "bai_args": {"npt": 8, "method": "histogram", "error_model": "poisson", "monitor": "i0"}, "gi_mode": "q_total"},
        "requested_shared_science": {"version": 1, "kind": "persisted_target"},
        "resource_policy": {"version": 1, "kind": "resolve", "envelope_bytes": 8 << 30, "requests": {"workers": 1, "reduction_inflight": 1}}}
    plan = ReintegrateSuccessorPlan.from_artifact(target, entry="entry", dimension="1d", preparation=preparation,
        expected_target_snapshot=capture_target_snapshot(target), expected_labels=(0, 1, 2), explicit_output=tmp_path / "reintegrated.nexus")
    result = run_reintegrate_successor(plan)
    assert result.disposition == "COMMITTED"
    first, second = open_scan(target), open_scan(result.output_artifact)
    for label in first.frame_indices:
        a, b = first.get_1d(label), second.get_1d(label)
        np.testing.assert_array_equal(a.intensity, b.intensity)
        np.testing.assert_array_equal(a.sigma, b.sigma)
    assert target.read_bytes() == before
    secondary = tmp_path / "points_scan5_0003.tif"
    secondary.write_bytes(secondary.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="REVISION|LINEAGE|changed|source"):
        ReintegrateSuccessorPlan.from_artifact(target, entry="entry", dimension="1d", preparation=preparation,
            expected_target_snapshot=capture_target_snapshot(target), expected_labels=(0, 1, 2), explicit_output=tmp_path / "refused.nexus")
    assert not (tmp_path / "refused.nexus").exists() and target.read_bytes() == before


@pytest.mark.parametrize("operation", ["average", "difference", "select"])
@pytest.mark.parametrize("dark_value", [None, 3.])
def test_average_scan_counts_combined_points_and_preserves_native_variance(tmp_path, operation, dark_value):
    import hashlib
    from xrd_tools.io.read import get_average_finite_counts
    from xrd_tools.reduction.average import AverageScanRecipe, AverageScanRunner, iter_average_contributors
    from xrd_tools.session.experiment_state import CalibrationState, DetectorImageState, FactStatus, PoniValues
    from xrd_tools.sources.selection import image_series_spec
    source, native, detector = _fixture(tmp_path, operation)
    series = image_series_spec(tmp_path / "points_scan5_0000.tif", metadata_format="spec")
    series = replace(series, options={**series.options, "point_images": dict(source.options["point_images"])})
    p = detector.poni
    calibration = CalibrationState(PoniValues(p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3, p.wavelength), p.detector,
                                   detector.detector_config, status=FactStatus.PRESENT)
    if dark_value is not None:
        dark_path = tmp_path / "dark.npy"
        np.save(dark_path, np.full((16, 16), dark_value))
        calibration = replace(calibration, dark=DetectorImageState(str(dark_path), hashlib.sha256(dark_path.read_bytes()).hexdigest(), (16, 16), FactStatus.PRESENT))
    runner = AverageScanRunner(AverageScanRecipe(series, tmp_path / "points.nexus", _plan(threshold_min=5., threshold_max=65534.), calibration=calibration, source_base=tmp_path))
    result = runner.start()
    assert runner.close() is result
    assert result.disposition == "COMMITTED", result
    assert result.contributor_extent == 3
    contributors = tuple(iter_average_contributors(result.target))
    assert len(contributors) == 3
    assert [len(item.source_constituents) for item in contributors] == [1 if operation == "select" else 2] * 3
    assert [item.logical_index for item in contributors] == [0, 1, 2]
    assert [tuple(member.source_path for member in item.source_constituents) for item in contributors] == [
        tuple(str(tmp_path / f"points_scan5_{raw:04d}.tif") for raw in ((2 * point + 1,) if operation == "select" else (2 * point, 2 * point + 1))) for point in range(3)]
    points, variances = [], []
    for point in range(3):
        a, b = native[2 * point].astype(float), native[2 * point + 1].astype(float)
        operands = [b] if operation == "select" else [a, b]
        invalid = np.logical_or.reduce([(operand < 5) | (operand > 65534) for operand in operands])
        signal = (a + b) / 2 if operation == "average" else b - a if operation == "difference" else b.copy()
        variance = (a + b) / 4 if operation == "average" else a + b if operation == "difference" else b.copy()
        signal[invalid] = variance[invalid] = np.nan
        points.append(signal); variances.append(variance)
    counts = np.sum(np.isfinite(points), axis=0)
    average = np.nansum(points, axis=0) / counts
    variance = np.nansum(variances, axis=0) / counts ** 2
    if dark_value is not None and operation != "difference":
        variance += dark_value
    reference_plan = _plan(mask_saturation=False, dark=(np.full((16, 16), dark_value) if dark_value is not None and operation != "difference" else None))
    reference_plan = replace(reference_plan, integration_1d=replace(reference_plan.integration_1d, error_model=None, extra={"variance": variance}), integration_2d=replace(reference_plan.integration_2d, error_model=None, extra={"variance": variance}))
    metadata = {"i0": 14. / 3., **({"average_point_images": {"operation": "difference"}} if operation == "difference" else {})}
    expected = run_reduction(reference_plan, Scan("reference", [Frame(1, image=average, metadata=metadata)], poni=p, integrator=detector_calibration_to_integrator(detector)), executor=1).frames[1]
    actual = open_scan(result.target)
    np.testing.assert_array_equal(actual.get_1d(1).intensity, expected.result_1d.intensity)
    np.testing.assert_array_equal(actual.get_1d(1).sigma, expected.result_1d.sigma)
    # The public processed 2-D reader exposes (azimuth, radial), while the
    # reduction container owns (radial, azimuth).
    np.testing.assert_array_equal(actual.get_2d(1).intensity, expected.result_2d.intensity.T.astype(np.float32))
    from xrd_tools.io.frame_view import FrameViewReader
    with FrameViewReader(result.target) as reader:
        np.testing.assert_array_equal(reader.read(1).sigma_2d, expected.result_2d.sigma.T.astype(np.float32))
    np.testing.assert_array_equal(get_average_finite_counts(result.target).values, counts)


def test_average_grouped_direct_eiger_select_keeps_native_cursor_census(tmp_path):
    import hdf5plugin
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.io.read import get_average_finite_counts
    from xrd_tools.reduction.average import AverageScanRecipe, AverageScanRunner, iter_average_contributors
    from xrd_tools.session.experiment_state import CalibrationState, FactStatus, PoniValues
    images = np.stack([np.full((17, 17), value, np.uint16) for value in (10, 200, 30, 400)])
    shard, master = tmp_path / "raw_data_000001.h5", tmp_path / "raw_master.h5"
    with h5py.File(shard, "w") as stream:
        stream.create_dataset("data", data=images, chunks=(1, 17, 17), **hdf5plugin.Bitshuffle(cname="lz4"))
    with h5py.File(master, "w") as stream:
        entry = stream.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data"); data.attrs["NX_class"] = "NXdata"
        data["data_000001"] = h5py.ExternalLink(shard.name, "/data")
    policy = PointImagePolicy(2, "select", image_index=1)
    source = SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry", options={"scan_point_count": 2, "point_images": policy.to_dict()})
    p = PONI(.2, .001, .001, 0., 0., 0., 1e-10, "Pilatus100k")
    detector = DetectorCalibration(p, {"max_shape": [17, 17], "orientation": 3})
    calibration = CalibrationState(PoniValues(p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3, p.wavelength), p.detector, detector.detector_config, status=FactStatus.PRESENT)
    direct_plan = _plan(mask_saturation=False)
    direct_plan = replace(direct_plan, integration_1d=replace(direct_plan.integration_1d, monitor_key=None), integration_2d=replace(direct_plan.integration_2d, monitor_key=None))
    runner = AverageScanRunner(AverageScanRecipe(source, tmp_path / "raw.nexus", direct_plan, calibration=calibration, source_base=tmp_path))
    result = runner.start(); assert runner.close() is result
    assert result.disposition == "COMMITTED", result
    assert runner.plan.direct_eiger_eligible and result.contributor_extent == 2
    direct = read_provenance(result.target)["config"]["average_scan_v1"]["direct_eiger_execution"]
    assert direct["direct_frames"] + direct["fallback_frames"] == 4
    np.testing.assert_array_equal(get_average_finite_counts(result.target).values, np.full((17, 17), 2))
    contributors = tuple(iter_average_contributors(result.target))
    assert [item.source_constituents[0].source_frame_index for item in contributors] == [0, 2]
    variance = np.full((17, 17), 10.)
    reference = direct_plan
    reference = replace(reference, integration_1d=replace(reference.integration_1d, error_model=None, extra={"variance": variance}), integration_2d=None)
    expected = run_reduction(reference, Scan("reference", [Frame(1, image=np.full((17, 17), 20.))], poni=p, integrator=detector_calibration_to_integrator(detector)), executor=1).frames[1].result_1d
    actual = open_scan(result.target).get_1d(1)
    np.testing.assert_array_equal(actual.intensity, expected.intensity)
    np.testing.assert_array_equal(actual.sigma, expected.sigma)
