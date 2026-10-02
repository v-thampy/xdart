"""Replay concrete operator intent through the existing preparation and transaction owners."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xdart.gui.tools.rsm_values import prepare_rsm_tool_v2
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
    from tests.xdart.tools.test_stitch_native_values import _native_form
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
    from tests.xdart.tools.test_stitch_native_values import _native_form
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
    from tests.xdart.tools.test_stitch_native_values import _native_form
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
from xrd_tools.notebooks.operation_replay import capture_stitch_form, capture_rsm_forms, run_operation_replay
from xrd_tools.analysis.stitch_values import StitchToolForm
from xrd_tools.analysis.rsm_values import RSMToolFormV2
assert not any(n == 'xdart' or n.startswith(('xdart.', 'PySide', 'PyQt', 'pyqtgraph')) for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
