"""Replay concrete operator intent through the existing preparation and transaction owners."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from xrd_tools.analysis.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.analysis.rsm_operation import run_rsm_operation_v2
from xrd_tools.io.analysis_artifact import read_analysis_artifact


def _rsm_form(root):
    from tests.core.test_rsm_operation_r2 import _form, _write_member
    from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
    member = _write_member(root, 1)
    for index, path in enumerate(sorted(Path(member.image_dir).glob("*.tif"))):
        tifffile.imwrite(path, np.full(member.detector_shape, 10 + index, dtype=np.uint16))
    return replace(_form(root, (member,)), coordinate_frame=RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU)


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_capture_round_trip_keeps_stitch_form_and_prepared_fingerprints(tmp_path, backend):
    from tests.core.test_stitch_operation import _native_form
    from xrd_tools.notebooks.operation_replay import capture_stitch_form, _restore_forms
    form = _native_form(tmp_path, backend, scans="1")
    configuration = json.loads(json.dumps(capture_stitch_form(form)))
    restored, = _restore_forms("stitch", configuration, data_root=tmp_path, output_root=tmp_path)
    assert restored.fingerprint == form.fingerprint
    before, after = prepare_stitch_tool(form), prepare_stitch_tool(restored)
    assert tuple(r.module.fingerprint for r in before.requests) == tuple(r.module.fingerprint for r in after.requests)
    assert restored.backend == backend and restored.conditioning == form.conditioning


def test_capture_round_trip_keeps_rsm_v2_geometry_and_scientific_settings(tmp_path):
    from xrd_tools.notebooks.operation_replay import capture_rsm_forms, _restore_forms
    form = _rsm_form(tmp_path)
    configuration = json.loads(json.dumps(capture_rsm_forms((form,))))
    restored, = _restore_forms("rsm", configuration, data_root=tmp_path, output_root=tmp_path)
    assert restored.fingerprint == form.fingerprint
    assert restored.geometry_asset.locator == form.geometry_asset.locator
    assert restored.conditioning == form.conditioning and restored.normalization == form.normalization
    assert prepare_rsm_tool_v2(restored).request.module.fingerprint == prepare_rsm_tool_v2(form).request.module.fingerprint


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_native_stitch_replay_matches_science_and_keeps_original_bytes(tmp_path, backend):
    from tests.core.test_stitch_operation import _native_form
    from xrd_tools.notebooks.operation_replay import capture_stitch_form, run_operation_replay
    form = _native_form(tmp_path, backend, scans="1")
    original = run_stitch_operation(prepare_stitch_tool(form).requests[0])
    assert original.terminal.disposition is ModuleDisposition.COMMITTED
    original_path = Path(original.request.module.output.target)
    original_bytes = original_path.read_bytes()
    configuration = capture_stitch_form(form)
    output_root = tmp_path / "replayed"
    assert not output_root.exists()
    replayed, = run_operation_replay("stitch", configuration, data_root=tmp_path, output_root=output_root)
    assert replayed.terminal.disposition is ModuleDisposition.COMMITTED
    assert Path(replayed.request.module.output.target).is_relative_to(output_root)
    np.testing.assert_allclose(replayed.payload.intensity, original.payload.intensity, rtol=0, atol=0)
    assert original_path.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="WORKFLOW_OUTPUT_EXISTS"):
        run_operation_replay("stitch", configuration, data_root=tmp_path, output_root=output_root)
    assert original_path.read_bytes() == original_bytes


def test_real_rsm_v2_replay_uses_the_original_owner_and_fresh_namespace(tmp_path):
    from xrd_tools.notebooks.operation_replay import capture_rsm_forms, run_operation_replay
    form = _rsm_form(tmp_path)
    original = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
    assert original.terminal.disposition is ModuleDisposition.COMMITTED
    target = Path(original.request.module.output.target)
    original_bytes = target.read_bytes()
    configuration = capture_rsm_forms((form,))
    replayed, = run_operation_replay("rsm", configuration, data_root=tmp_path, output_root=tmp_path / "replayed")
    assert replayed.terminal.disposition is ModuleDisposition.COMMITTED
    np.testing.assert_allclose(replayed.payload.intensity, original.payload.intensity, rtol=0, atol=0)
    loaded = read_analysis_artifact(replayed.request.module.output.target)
    assert loaded.inspection.axes == original.payload.inspection.axes
    assert target.read_bytes() == original_bytes


def test_external_inputs_stay_explicit_and_external_outputs_are_refused(tmp_path):
    from tests.core.test_stitch_operation import _native_form
    from xrd_tools.notebooks.operation_replay import capture_stitch_form, run_operation_replay, _restore_forms
    form = _native_form(tmp_path, "multigeometry", scans="1")
    outside = tmp_path.parent / "external-mask.edf"
    configuration = capture_stitch_form(replace(form, mask_path=str(outside)))
    restored, = _restore_forms("stitch", configuration, data_root=tmp_path, output_root=tmp_path)
    assert restored.mask_path == str(outside)
    with pytest.raises(ValueError, match="WORKFLOW_OUTPUT_OUTSIDE_PROJECT"):
        run_operation_replay("stitch", configuration, data_root=tmp_path, output_root=tmp_path.parent / "outside")
    assert not (tmp_path.parent / "outside").exists()
    configuration["forms"][0]["fields"]["output_path"] = {"$path": str(form.output_path), "root": "external"}
    with pytest.raises(ValueError, match="WORKFLOW_OUTPUT_OUTSIDE_NAMESPACE"):
        run_operation_replay("stitch", configuration, data_root=tmp_path, output_root=tmp_path / "escape")
    assert not (tmp_path / "escape").exists()


def test_replay_import_and_concrete_type_transport_are_headless(tmp_path):
    command = """
import sys
from pathlib import Path
from xrd_tools.notebooks.operation_replay import capture_stitch_form, capture_rsm_forms, run_operation_replay
from xrd_tools.analysis.stitch_values import StitchToolForm
from xrd_tools.analysis.rsm_values import RSMToolFormV2
from tests.core.test_stitch_operation import _native_form
form = _native_form(Path(sys.argv[1]), 'multigeometry', scans='1')
assert capture_stitch_form(form)['forms']
assert not any(n == 'xdart' or n.startswith(('xdart.', 'PySide', 'PyQt', 'pyqtgraph')) for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", command, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _grouped_stitch_form(root, backend, operation):
    from tests.core.test_stitch_operation import _native_form
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    from xrd_tools.sources.point_images import PointImagePolicy
    form = _native_form(root, backend, scans="1")
    policy = PointImagePolicy(2, operation, image_index=2)
    for index in range(4):
        np.full(form.detector_shape, 40 if index % 2 == 0 else 20, np.int32).tofile(
            Path(form.image_dir) / f"sample_scan1_{index:04d}.raw")
    selection = replace(form.source_selection, point_image_choices=((str(form.spec_path), "1.1", policy),))
    return replace(form, source_selection=selection, conditioning=StitchConditioning("manual", 0., 100.))


def _grouped_rsm_form(root, operation):
    from tests.core.test_rsm_point_images import SHAPE, _member, _form
    from xrd_tools.sources.point_images import PointImagePolicy
    images = [np.full(SHAPE, 40 if index % 2 == 0 else 20, np.uint16) for index in range(4)]
    member = _member(root, "grouped", images, PointImagePolicy(2, operation, image_index=2).to_dict())
    return _form(root, member, "grouped-result")


@pytest.mark.parametrize("mode", ("stitch", "rsm"))
def test_default_point_fields_preserve_legacy_replay_payload_and_restore_defaults(tmp_path, mode):
    from tests.core.test_stitch_operation import _native_form
    from xrd_tools.notebooks.operation_replay import capture_stitch_form, capture_rsm_forms, _restore_forms
    form = _native_form(tmp_path, "multigeometry", scans="1") if mode == "stitch" else _rsm_form(tmp_path)
    configuration = capture_stitch_form(form) if mode == "stitch" else capture_rsm_forms((form,))
    fields = configuration["forms"][0]["fields"]
    nested = fields["source_selection"]["fields"] if mode == "stitch" else fields["members"]["$tuple"][0]["fields"]
    name = "point_image_choices" if mode == "stitch" else "point_images"
    assert name not in nested
    restored, = _restore_forms(mode, configuration, data_root=tmp_path, output_root=tmp_path)
    assert restored.fingerprint == form.fingerprint
    assert (restored.source_selection.point_image_choices if mode == "stitch" else restored.members[0].point_images) == (() if mode == "stitch" else None)


@pytest.mark.parametrize("mode", ("stitch", "rsm"))
@pytest.mark.parametrize("operation", ("average", "difference", "select"))
def test_grouped_form_round_trip_preserves_policy_identity_and_relocates_choices(tmp_path, mode, operation):
    from xrd_tools.notebooks.operation_replay import capture_stitch_form, capture_rsm_forms, _restore_forms
    form = _grouped_stitch_form(tmp_path, "multigeometry", operation) if mode == "stitch" else _grouped_rsm_form(tmp_path, operation)
    configuration = json.loads(json.dumps(capture_stitch_form(form) if mode == "stitch" else capture_rsm_forms((form,))))
    restored, = _restore_forms(mode, configuration, data_root=tmp_path, output_root=tmp_path)
    assert restored.fingerprint == form.fingerprint
    if mode == "stitch":
        assert restored.source_selection.point_image_choices == form.source_selection.point_image_choices
        assert prepare_stitch_tool(restored).requests[0].module.fingerprint == prepare_stitch_tool(form).requests[0].module.fingerprint
    else:
        assert dict(restored.members[0].point_images) == dict(form.members[0].point_images)
        assert prepare_rsm_tool_v2(restored).request.module.fingerprint == prepare_rsm_tool_v2(form).request.module.fingerprint
    relocated = tmp_path / "relocated"
    moved, = _restore_forms(mode, configuration, data_root=relocated, output_root=relocated / "outputs")
    if mode == "stitch":
        assert moved.source_selection.point_image_choices[0][0] == str(relocated / "source/sample")
        assert moved.source_selection.point_image_choices[0][2].operation == operation
    else:
        assert moved.members[0].spec_path == str(relocated / "grouped/point_scan")
        assert moved.members[0].point_images["operation"] == operation


@pytest.mark.parametrize("mode,backend", (("stitch", "multigeometry"), ("stitch", "xu_hist"), ("rsm", None)))
def test_real_grouped_difference_replay_preserves_signed_science_and_ordered_provenance(tmp_path, mode, backend):
    from xrd_tools.notebooks.operation_replay import capture_stitch_form, capture_rsm_forms, run_operation_replay
    form = _grouped_stitch_form(tmp_path, backend, "difference") if mode == "stitch" else _grouped_rsm_form(tmp_path, "difference")
    prepared = prepare_stitch_tool(form).requests[0] if mode == "stitch" else prepare_rsm_tool_v2(form).request
    original = (run_stitch_operation if mode == "stitch" else run_rsm_operation_v2)(prepared)
    assert original.terminal.disposition is ModuleDisposition.COMMITTED, original.terminal
    target = Path(original.request.module.output.target)
    original_bytes = target.read_bytes()
    configuration = capture_stitch_form(form) if mode == "stitch" else capture_rsm_forms((form,))
    output_root = tmp_path / "replayed-grouped"
    replayed, = run_operation_replay(mode, configuration, data_root=tmp_path, output_root=output_root)
    assert replayed.terminal.disposition is ModuleDisposition.COMMITTED, replayed.terminal
    assert Path(replayed.request.module.output.target).is_relative_to(output_root)
    np.testing.assert_array_equal(replayed.payload.intensity, original.payload.intensity)
    assert np.nanmin(replayed.payload.intensity) < 0
    if mode == "stitch":
        for before, after in zip(original.payload.contributing_frames, replayed.payload.contributing_frames):
            assert before["point_images"] == after["point_images"]
            assert before["source_constituents"] == after["source_constituents"]
            assert len(after["source_constituents"]) == 2
    else:
        before, after = original.request.preflight.members[0], replayed.request.preflight.members[0]
        assert before.to_provenance() == after.to_provenance()
        assert json.loads(after.source_options_json)["point_images"]["operation"] == "difference"
    assert target.read_bytes() == original_bytes
