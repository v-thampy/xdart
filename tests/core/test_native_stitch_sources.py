"""Actual grouped finite execution and source identity, without GUI doubles."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceGroupReceipt
from xrd_tools.analysis.stitch_operation import prepare_stitch_operation, run_stitch_operation
from xrd_tools.analysis.xu_stitch_calibration import XuStitchCalibrationInput, capture_xu_stitch_calibration
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
from xrd_tools.sources.spec import SpecSource
from tests.core.test_stitch_operation import _prepared, _xu_prepared
from tests.core.test_spec_source import _spec


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_grouped_operation_uses_all_original_selected_frames_once(tmp_path, backend):
    prepare = _prepared if backend == "multigeometry" else _xu_prepared
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    first, second = prepare(tmp_path / "a"), prepare(tmp_path / "b")
    reference = run_stitch_operation(first)
    assert reference.terminal.disposition is ModuleDisposition.COMMITTED
    group = ModuleSourceGroupReceipt.from_members((first.module.source, second.module.source), kind=ModuleKind.STITCH)
    plan = first.plan
    if backend == "xu_hist":
        plan = replace(plan, calibration=capture_xu_stitch_calibration(
            XuStitchCalibrationInput(Path(first.plan.calibration.project_root) / first.plan.calibration.resolved_relative_path), project_root=tmp_path,
        ))
    request = prepare_stitch_operation(group,
        ModuleOutputRequest(tmp_path / "group.nexus", AnalysisArtifactKind.STITCH_1D), plan, project_root=tmp_path)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal.diagnostic
    assert request.manifest.member_frames == tuple(
        (ordinal, label) for ordinal, member in enumerate(group.members) for label in member.selected_labels)
    np.testing.assert_allclose(result.payload.intensity, reference.payload.intensity, equal_nan=True)
    np.testing.assert_allclose(result.payload.coverage, 2 * reference.payload.coverage)
    np.testing.assert_allclose(result.payload.normalization, 2 * reference.payload.normalization)


def test_strict_spec_refuses_partial_images_and_native_read_retains_saturation(tmp_path):
    spec = _spec(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    values = np.array([[0, np.iinfo(np.uint16).max], [2, 3]], dtype=np.uint16)
    for index in (0, 2):
        values.tofile(images / f"myscan_scan5_{index:04}.raw")
    options = dict(scan="5.1", image_dir=images, require_complete_images=True,
        read_image_kwargs={"detector_shape": (2, 2), "raw_dtype": "uint16", "threshold": 100})
    with pytest.raises(ValueError, match="image count"):
        SpecSource(spec, **options)
    values.tofile(images / "myscan_scan5_0001.raw")
    source = SpecSource(spec, **options)
    native = source.load_native_frame(1)
    assert native.dtype == np.uint16
    assert native[0, 1] == 65535
    assert source.metadata_for(1)["th"] == 1
    empty = SpecSource(spec, scan="6.1", image_dir=images, require_complete_images=True)
    assert not empty.capabilities.has_raw_references
