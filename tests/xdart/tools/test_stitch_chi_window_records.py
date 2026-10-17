"""I(q)'s χ window from the Stitch page's form to the record (maintainer, 2026-09-30: "a 1-D χ range" -- "yes,
mirror Int"), on both backends: a window reaches the plan, its identity and the record's provenance; I(q)
beside a Stitch 2D's map takes it as the Stitch 1D does, and the map keeps its own χ range. Auto (None, every
χ) keeps every identity it had (``tests/core/test_stitch_chi_window.py`` holds the kernels' oracles)."""
from dataclasses import replace
import json

import numpy as np
import pytest

from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
from tests.xdart.pages.test_stitch_page import _form
from tests.xdart.tools.test_stitch_native_values import _native_form


def _stitched(tmp_path, form, name):
    (tmp_path / name).mkdir()
    kind = AnalysisArtifactKind.STITCH_2D if form.mode == "2d" else AnalysisArtifactKind.STITCH_1D
    request = prepare_stitch_tool(replace(form, output_path=tmp_path / name / f"{name}_stitch{form.mode}.nexus")).request
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    return request, read_analysis_artifact(request.module.output.target, expected_kind=kind)


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_a_window_reaches_the_record_and_i_of_q_beside_the_map(tmp_path, backend):
    form = _native_form(tmp_path, backend, scans="1")
    map_form = replace(form, mode="2d", npt_rad_2d=23, npt_azim_2d=19)
    _request, every_map = _stitched(tmp_path, map_form, "every2d")
    chi = dict(every_map.axes)["chi"]  # Auto: the frames' own χ extent
    window = tuple(float(value) for value in np.percentile(chi, (30, 70)))
    every_request, every = _stitched(tmp_path, form, "every1d")
    windowed_form = replace(form, chi_range_1d=window)
    assert windowed_form.fingerprint != form.fingerprint
    request, windowed = _stitched(tmp_path, windowed_form, "window1d")
    assert request.plan.azimuth_range_1d == window and request.plan.fingerprint != every_request.plan.fingerprint
    assert json.loads(windowed.provenance_json)["plan"]["azimuth_range_1d"] == list(window)
    assert "azimuth_range_1d" not in every.provenance_json  # every χ: the record as before
    assert 0 < windowed.coverage.sum() < 0.7 * every.coverage.sum()
    _request, both = _stitched(tmp_path, replace(map_form, chi_range_1d=window), "window2d")
    for name in ("intensity", "coverage", "normalization"):
        np.testing.assert_array_equal(getattr(both.profile, name), getattr(windowed, name))
        np.testing.assert_array_equal(getattr(both, name), getattr(every_map, name))  # the map: its own


def test_a_window_is_a_native_stitchs_finite_increasing_pair(tmp_path):
    native = _native_form(tmp_path, "multigeometry", scans="1")
    assert replace(native, chi_range_1d=(-60, 45)).chi_range_1d == (-60.0, 45.0)
    for window in ((45.0, -60.0), (float("nan"), 1.0), (1.0,)):
        with pytest.raises(ValueError, match="χ range"):
            replace(native, chi_range_1d=window)
    (tmp_path / "older").mkdir()
    with pytest.raises(ValueError, match="native Stitch"):
        replace(_form(tmp_path / "older", "multigeometry"), chi_range_1d=(-60.0, 45.0))
