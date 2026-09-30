"""Scan 14: Stitch screens dropped frames (detector counts collapse, monitor normal).

Frames 52 and 164 are empty (below 1 % of their nu-row neighbours) and are excluded
by default; 99, 106, 120 and 272 are partial exposures, flagged and excluded only on
request. Excluding a frame must equal never selecting it, bit for bit.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector, ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceReceipt,
)
from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
from xrd_tools.analysis.stitch_operation import (
    StitchGeometryInput, StitchGeometryKind, StitchOperationPlan, capture_stitch_geometry,
    prepare_stitch_operation, run_stitch_operation,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.frame_screen import FrameScreenPolicy
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactOverwrite


def _root() -> Path:
    configured = os.environ.get('XDART_TEST_DATA')
    if not configured:
        pytest.skip('XDART_TEST_DATA is not configured')
    root = Path(configured) / 'stitching'
    if not root.is_dir():
        pytest.skip('authenticated Stitching test data is unavailable')
    return root


def _run(root, target, labels, policy):
    data = root / 'data'
    spec = SourceSpec(data / 'calibration/LaB6_17keV', SourceKind.SPEC, options={
        'scan': '14.1', 'image_dir': str(data / 'images'), 'image_stem': 'b_thampy_LaB6_17keV_scan14_',
        'read_image_kwargs': {'detector_shape': (195, 1475), 'raw_dtype': 'int32',
                              'threshold': 8.0e5, 'rotation': 0}})
    table = run_metadata_table(MetadataTablePlan(spec))
    source = ModuleSourceReceipt.from_metadata_table(
        table, kind=ModuleKind.STITCH, selected_labels=labels,
        resolved_selectors=(MetadataColumnSelector('del'), MetadataColumnSelector('nu')))
    geometry = capture_stitch_geometry(StitchGeometryInput(
        data / 'calibration/MG_gonio_object_5images.json', StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        source_motors=(('del_angle', 'del'), ('nu_angle', 'nu')), image_rotation=0))
    plan = StitchOperationPlan(geometry, mode='1d', npt_1d=1500, radial_range=(1.0, 6.2),
                               use_detector_mask=True, max_frame_bytes=64 * 1024 * 1024,
                               frame_screen=policy)
    output = ModuleOutputRequest(target, AnalysisArtifactKind.STITCH_1D, AnalysisArtifactOverwrite.CREATE_NEW)
    result = run_stitch_operation(prepare_stitch_operation(source, output, plan, project_root=root))
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    return result


@pytest.mark.slow
def test_scan14_stitch_excludes_empty_frames_and_flags_partial_ones(tmp_path):
    root = _root()
    labels = tuple(range(287))
    default = _run(root, tmp_path / 'default.nexus', labels, FrameScreenPolicy())
    assert default.frame_screen['excluded'] == [52, 164]
    assert default.frame_screen['empty'] == [52, 164]
    assert default.frame_screen['partial'] == [99, 106, 120, 272]
    plan = json.loads(default.payload.provenance_json)['plan']
    assert plan['frame_screen']['exclude'] == 'empty'
    never_selected = _run(root, tmp_path / 'never.nexus', tuple(i for i in labels if i not in (52, 164)),
                          FrameScreenPolicy(exclude='none'))
    np.testing.assert_array_equal(default.payload.intensity, never_selected.payload.intensity)
    np.testing.assert_array_equal(default.payload.normalization, never_selected.payload.normalization)
    strict = _run(root, tmp_path / 'strict.nexus', labels, FrameScreenPolicy(exclude='partial'))
    assert strict.frame_screen['excluded'] == [52, 99, 106, 120, 164, 272]
    assert default.request.plan.fingerprint != strict.request.plan.fingerprint
