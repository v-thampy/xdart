"""A Stitch 2D's map on its own axis, as Int 2D's (maintainer, 2026-09-30: "like int 1d/2d the Processing panel
in Stitch should also have the capability of selecting a range for both axes in the 2d"; "there is no choice to
select the 2d integration axes"): the map's radial unit (Q-χ or 2θ-χ) and range are its own, and I(q) beside it
keeps the 1-D axis and range. Each is exactly what a stitch with that axis alone makes."""
from dataclasses import replace

import numpy as np
import pytest

from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.io.analysis_artifact import read_analysis_artifact
from tests.xdart.tools.test_stitch_native_values import _native_form


def _stitched(form, family):
    request = prepare_stitch_tool(replace(form, output_family=family)).request
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    return read_analysis_artifact(request.module.output.target), request


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
def test_the_map_and_i_of_q_each_keep_their_own_axis_and_range(tmp_path, backend):
    form = replace(_native_form(tmp_path, backend, scans="1"), mode="2d", npt_1d=41, npt_rad_2d=23,
                   npt_azim_2d=19, q_range=(1.0, 3.0), map_unit="2th_deg", map_radial_range=(10.0, 30.0))
    own, request = _stitched(form, "own")
    (_q, map_unit), (_chi, _chi_unit) = own.inspection.axis_units
    ((_p, profile_unit),) = own.profile.axis_units
    assert (map_unit, profile_unit) == ("2th_deg", "q_A^-1")
    radial = own.axis("q")  # the map's radial axis, in its own unit
    assert 10.0 < radial[0] < radial[-1] < 30.0
    assert 1.0 < own.profile.axes[0][1][0] < own.profile.axes[0][1][-1] < 3.0
    plan = request.provenance["plan"]
    assert (plan.get("map_unit"), plan.get("map_radial_range")) == ("2th_deg", [10.0, 30.0]) or \
        (plan["map"]["unit"], plan["map"]["radial_range"]) == ("2th_deg", [10.0, 30.0])
    # the map is the one a Stitch 2D on that axis alone makes, bit for bit
    alone, _request = _stitched(replace(form, unit="2th_deg", q_range=(10.0, 30.0)), "alone")
    np.testing.assert_array_equal(own.intensity, alone.intensity)
    np.testing.assert_array_equal(own.axis("q"), alone.axis("q"))
    np.testing.assert_array_equal(own.axis("chi"), alone.axis("chi"))
    # and I(q) the one a Stitch 1D makes over its own range
    one, _request = _stitched(replace(form, mode="1d"), "one")
    np.testing.assert_array_equal(own.profile.intensity, one.intensity)
    np.testing.assert_array_equal(own.profile.axes[0][1], one.axis("q"))


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
def test_an_auto_map_range_is_the_frames_extent_in_the_maps_unit(tmp_path, backend):
    form = replace(_native_form(tmp_path, backend, scans="1"), mode="2d", npt_1d=41, npt_rad_2d=23,
                   npt_azim_2d=19, q_range=(1.0, 3.0))
    auto, _request = _stitched(form, "auto")
    whole, _request = _stitched(replace(form, q_range=None), "whole")  # I(q) Auto too: the same extent
    np.testing.assert_array_equal(auto.intensity, whole.intensity)
    assert not np.array_equal(auto.profile.axes[0][1], whole.profile.axes[0][1])  # I(q) keeps its range


def test_a_grazing_map_is_binned_in_q(tmp_path):
    from xrd_tools.integrate.stitch_gi import StitchGI
    from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form
    form = replace(_gi_form(tmp_path, "multigeometry", StitchGI(("q_ip", "q_oop"), incidence_motor="eta",
                                                                sample_motors=_SAMPLE)), mode="2d")
    with pytest.raises(ValueError, match="grazing-incidence map is binned in Q"):
        replace(form, map_unit="2th_deg")
    with pytest.raises(ValueError, match="Q-χ or 2θ-χ"):
        replace(form, gi=None, map_unit="chi_deg")
