"""Confirmed point grouping through RSM's real bounded Preview and Run owners."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from xrd_tools.analysis.module_transaction import MetadataColumnSelector, ModuleDisposition
from xrd_tools.analysis.rsm_geometry_asset import (
    CANONICAL_RSM_GEOMETRY_LOCATOR, install_canonical_rsm_geometry_asset,
    rsm_geometry_asset_input,
)
from xrd_tools.analysis.rsm_operation import (
    RSMFrameCorrections, RSMImageConditioning, RSMNormalizationMode,
    RSMNormalizationPolicy, run_rsm_operation_v2,
)
from xrd_tools.analysis.rsm_values import (
    RSMFrameSelector, RSMNexusMemberForm, RSMScanMemberForm, RSMToolFormV2,
    RSMToolPreflightRefused, prepare_rsm_tool_v2, rsm_member_groups,
)
from xrd_tools.analysis.stitch_sources import StitchSourceSelection
from xrd_tools.core.scan import SourceKind
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from xrd_tools.sources.point_images import PointImagePolicy, PointImagesRequired


SHAPE = (195, 487)
MOTORS = tuple((role, MetadataColumnSelector(role))
               for role in ("mu", "eta", "chi", "phi", "nu", "del"))


def _member(root, name, images, point_images=None):
    folder = root / name
    folder.mkdir()
    spec = folder / "point_scan"
    spec.write_text("""#F point_scan
#E 1
#D Mon Jan 15 10:30:00 2024
#O0 energy  mu  chi  phi  nu  del

#S 1 ascan eta 0 1 1 1
#D Mon Jan 15 10:30:00 2024
#P0 13000.007 10 20 30 40 50
#G3 1 0 0 0 1 0 0 0 1
#N 3
#L eta  Seconds  foil status
0 2 0
1 4 0
""", encoding="utf-8")
    for index, image in enumerate(images):
        tifffile.imwrite(folder / f"point_scan_scan1_{index:04d}.tif", image)
    return RSMScanMemberForm(spec, "1.1", folder, "point_scan_scan1_",
                             RSMFrameSelector(0, 1, 1), SHAPE, "uint16", 0,
                             MOTORS, point_images=point_images)


def _form(root, member, name, *, offset=0., corrections=None):
    install_canonical_rsm_geometry_asset(project_root=root)
    normalization = RSMNormalizationPolicy(
        RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE, MetadataColumnSelector("foil status"),
        MetadataColumnSelector("Seconds"),
        (1., 2., 3., 4.))
    return RSMToolFormV2(
        root, rsm_geometry_asset_input(CANONICAL_RSM_GEOMETRY_LOCATOR), (member,),
        RSMImageConditioning(offset, None, None), normalization, (5, 6, 7),
        1, 16 * 1024 * 1024, 32 * 1024 * 1024, root / f"{name}.nexus",
        coordinate_frame=RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU,
        corrections=corrections)


@pytest.mark.parametrize("nexus", [False, True])
def test_member_policy_is_immutable_and_part_of_its_identity(tmp_path, nexus):
    policy = PointImagePolicy(2, "average").to_dict()
    if nexus:
        plain = RSMNexusMemberForm(tmp_path / "master.h5", SourceKind.EIGER_MASTER,
                                  "entry", RSMFrameSelector(0, 1, 1), SHAPE, MOTORS)
    else:
        plain = _member(tmp_path, "ordinary", [np.ones(SHAPE, np.uint16)] * 2)
    grouped = replace(plain, point_images=policy)
    assert grouped.fingerprint != plain.fingerprint
    assert plain.point_images is None
    assert dict(grouped.point_images) == policy
    policy["operation"] = "select"
    assert grouped.point_images["operation"] == "average"
    with pytest.raises(TypeError):
        grouped.point_images["operation"] = "difference"
    with pytest.raises(ValueError):
        replace(plain, point_images={"images_per_point": 2, "operation": "select", "image_index": 3})


def test_source_selection_requires_confirmation_and_keeps_the_confirmed_member_policy(tmp_path):
    member = _member(tmp_path, "detected", [np.ones(SHAPE, np.uint16)] * 4)
    selection = StitchSourceSelection("Spec", (str(member.spec_path),), "1")
    kwargs = dict(frame_selector=member.frame_selector, detector_shape=SHAPE,
                  raw_dtype="uint16", raw_header_skip=0, motor_selectors=MOTORS)
    with pytest.raises(PointImagesRequired) as pending:
        rsm_member_groups(selection, **kwargs)
    assert pending.value.detection.images_per_point == 2
    policy = PointImagePolicy(2, "average")
    groups, skipped = rsm_member_groups(replace(selection, point_image_choices=(
        (str(member.spec_path), "1.1", policy),)), **kwargs)
    assert skipped == ()
    assert dict(groups[0][0].point_images) == policy.to_dict()


@pytest.mark.parametrize("operation,reverse,mode", [
    ("average", False, "manual"), ("average", False, "auto"),
    ("difference", False, "manual"), ("difference", True, "manual"),
    ("select", False, "manual"),
])
def test_grouped_rsm_matches_independently_prepared_point_signals(
        tmp_path, operation, reverse, mode):
    """Real SPEC/TIFF -> native mask -> grouping -> one point divisor -> real XU grid."""
    policy = PointImagePolicy(2, operation, image_index=2 if operation == "select" else 1,
                              reverse=reverse)
    raw = []
    expected = []
    for point in range(2):
        first = ((np.arange(np.prod(SHAPE)).reshape(SHAPE) % 59) + 40 + point).astype(np.uint16)
        second = first - 20
        first[20, 20] = 65535 if mode == "auto" else 180
        second[20, 20] = 20
        raw.extend((first, second))
        frames = [first.astype(np.float64), second.astype(np.float64)]
        for values in frames:
            values[values >= 65535 if mode == "auto" else values > 100] = np.nan
        if operation == "average":
            signal = (frames[0] + frames[1]) / 2
        elif operation == "difference":
            signal = frames[0] - frames[1] if reverse else frames[1] - frames[0]
        else:
            signal = frames[1]
        expected.append(signal)
    grouped = _member(tmp_path, "grouped", raw, policy.to_dict())
    reference = _member(tmp_path, "reference", expected)
    dark = tmp_path / "dark.tif"
    tifffile.imwrite(dark, np.full(SHAPE, 7., dtype=np.float64))
    threshold = StitchConditioning("auto") if mode == "auto" else StitchConditioning("manual", 0., 100.)
    corrections = RSMFrameCorrections(threshold=threshold, dark_path=str(dark),
                                      solid_angle=True, polarization_factor=.99)
    offset = 137. if operation == "difference" else 5.
    grouped_form = _form(tmp_path, grouped, "grouped-result", offset=offset, corrections=corrections)
    reference_corrections = replace(corrections, threshold=None,
                                    dark_path=None if operation == "difference" else str(dark))
    reference_form = _form(tmp_path, reference, "reference-result",
                           offset=0. if operation == "difference" else offset,
                           corrections=reference_corrections)
    grouped_request = prepare_rsm_tool_v2(grouped_form).request
    reference_request = prepare_rsm_tool_v2(reference_form).request
    member = grouped_request.preflight.members[0]
    assert json.loads(member.source_options_json)["point_images"] == policy.to_dict()
    assert member.contributions[0].normalization_divisor == 2.
    assert member.contributions[1].normalization_divisor == 4.
    for point, contribution in enumerate(member.contributions):
        raw_members = contribution.point_raw_locators
        assert len(raw_members) == (1 if operation == "select" else 2)
        expected_indices = (2 * point + 1,) if operation == "select" else (2 * point, 2 * point + 1)
        assert tuple(Path(member.dependency_files[file].relative_path).name
                     for file, _ in raw_members) == tuple(f"point_scan_scan1_{i:04d}.tif"
                                                        for i in expected_indices)
        assert all(frame == 0 for _, frame in raw_members)
    grouped_result = run_rsm_operation_v2(grouped_request)
    reference_result = run_rsm_operation_v2(reference_request)
    assert grouped_result.terminal.disposition is ModuleDisposition.COMMITTED
    assert reference_result.terminal.disposition is ModuleDisposition.COMMITTED
    np.testing.assert_array_equal(grouped_result.payload.intensity, reference_result.payload.intensity)
    attestation = json.loads(grouped_result.payload.execution_attestation_json)
    assert attestation["selected_frame_count"] == 2
    provenance = member.to_provenance()
    assert len(provenance["contributions"][0]["point_raw_locators"]) == len(member.contributions[0].point_raw_locators)
    if operation == "difference" and not reverse:
        assert np.nanmin(grouped_result.payload.intensity) < 0


@pytest.mark.parametrize("threshold", [None, StitchConditioning("auto")])
def test_select_offset_high_and_static_hot_use_only_selected_operands(tmp_path, threshold):
    """An unused varying/saturated operand cannot unmask a selected static bright pixel."""
    selected = []
    operands = []
    for point in range(2):
        used = np.full(SHAPE, 40 + point, np.uint16)
        used[20, 20] = 200  # unchanged selected hot pixel
        used[25, 25] = 499  # exceeds 500 only after the per-exposure +5 offset
        unused = np.full(SHAPE, 65535 if point == 0 else 3, np.uint16)
        selected.append(used)
        operands.extend((unused, used))
    policy = PointImagePolicy(2, "select", image_index=2)
    grouped = _member(tmp_path, "grouped", operands, policy.to_dict())
    reference = _member(tmp_path, "reference", selected)
    corrections = RSMFrameCorrections(threshold=threshold)
    form = replace(_form(tmp_path, grouped, "grouped-result", corrections=corrections),
                   conditioning=RSMImageConditioning(5., 500., 100.))
    reference_form = replace(form, members=(reference,), output_path=tmp_path / "reference-result.nexus")
    grouped_result = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
    reference_result = run_rsm_operation_v2(prepare_rsm_tool_v2(reference_form).request)
    assert grouped_result.terminal.disposition is ModuleDisposition.COMMITTED
    assert reference_result.terminal.disposition is ModuleDisposition.COMMITTED
    np.testing.assert_array_equal(grouped_result.payload.intensity, reference_result.payload.intensity)
    grouped_masks = json.loads(grouped_result.payload.execution_attestation_json)["member_masks"]
    reference_masks = json.loads(reference_result.payload.execution_attestation_json)["member_masks"]
    for key in ("full_raw_digest", "full_masked_pixel_count", "cropped_raw_digest", "cropped_masked_pixel_count"):
        assert grouped_masks[0][key] == reference_masks[0][key]


def test_average_high_threshold_runs_before_combination(tmp_path):
    raw, expected = [], []
    for point in range(2):
        first = np.full(SHAPE, 80 + point, np.uint16)
        second = np.full(SHAPE, 20 + point, np.uint16)
        first[25, 25], second[25, 25] = 499, 1
        raw.extend((first, second))
        signal = (first.astype(float) + second.astype(float)) / 2 + 5
        signal[25, 25] = np.nan
        expected.append(signal)
    policy = PointImagePolicy(2, "average")
    grouped = _member(tmp_path, "grouped", raw, policy.to_dict())
    reference = _member(tmp_path, "reference", expected)
    form = replace(_form(tmp_path, grouped, "grouped-result"),
                   conditioning=RSMImageConditioning(5., 500., None))
    reference_form = replace(form, members=(reference,), output_path=tmp_path / "reference-result.nexus",
                             conditioning=RSMImageConditioning(0., None, None))
    grouped_result = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
    reference_result = run_rsm_operation_v2(prepare_rsm_tool_v2(reference_form).request)
    assert grouped_result.terminal.disposition is ModuleDisposition.COMMITTED
    assert reference_result.terminal.disposition is ModuleDisposition.COMMITTED
    np.testing.assert_array_equal(grouped_result.payload.intensity, reference_result.payload.intensity)


@pytest.mark.parametrize("dark_invalid", ["nan", "finite"])
def test_difference_cancels_dark_only_invalid_pixels_but_keeps_independent_masks(tmp_path, dark_invalid):
    raw, expected = [], []
    for point in range(2):
        first = np.full(SHAPE, 40 + point, np.uint16)
        second = np.full(SHAPE, 20 + point, np.uint16)
        second[50, 50] = 400  # Must survive a dark-only invalid pixel.
        second[60, 60] = 450  # The explicit mask remains effective.
        second[70, 70] = 500  # The unusable flat pixel remains effective.
        raw.extend((first, second))
        expected.append(second.astype(float) - first)
    grouped = _member(tmp_path, "grouped", raw, PointImagePolicy(2, "difference").to_dict())
    reference = _member(tmp_path, "reference", expected)
    dark = np.full(SHAPE, 7., dtype=np.float64) if dark_invalid == "nan" else np.full(SHAPE, 7, np.int32)
    dark[50, 50] = np.nan if dark_invalid == "nan" else -1
    dark_path = tmp_path / "dark.tif"
    tifffile.imwrite(dark_path, dark)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[60, 60] = True
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)
    flat = np.ones(SHAPE)
    flat[70, 70] = 0.
    flat_path = tmp_path / "flat.npy"
    np.save(flat_path, flat)
    corrections = RSMFrameCorrections(dark_path=str(dark_path), mask_path=str(mask_path),
                                      flat_path=str(flat_path))
    form = _form(tmp_path, grouped, "grouped-result", corrections=corrections)
    reference_form = replace(_form(tmp_path, reference, "reference-result"),
                             corrections=replace(corrections, dark_path=None))
    grouped_result = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
    reference_result = run_rsm_operation_v2(prepare_rsm_tool_v2(reference_form).request)
    assert grouped_result.terminal.disposition is ModuleDisposition.COMMITTED
    assert reference_result.terminal.disposition is ModuleDisposition.COMMITTED
    np.testing.assert_array_equal(grouped_result.payload.intensity, reference_result.payload.intensity)


@pytest.mark.parametrize("image_count", [1, 3])
def test_rsm_headless_requires_one_to_one_or_confirmed_complete_groups(tmp_path, image_count):
    member = _member(tmp_path, "unmatched", [np.ones(SHAPE, np.uint16)] * image_count)
    form = _form(tmp_path, member, "refused")
    with pytest.raises(RSMToolPreflightRefused):
        prepare_rsm_tool_v2(form)
    assert not Path(form.output_path).exists()
