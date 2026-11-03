"""NEXT-128-WINDOW-POLISH item 14, Track options (the maintainer's live check, 2026-11-03: "to keep it simple, the track
options should also show the other peak pars like fraction for PV and height (even though it is derived), as well as
the peak independent bg pars that can be informative at times"): a displayed peak fit's outcome carries every fitted
parameter of each peak, its derived FWHM and height among them, and the background's own parameters, peak-independent;
the trend groups the background's as families of their own, with no peak."""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
from xrd_tools.analysis.runner import DisplayedPeakFitAnalyzer, run_batch


def _two_peaks(identity="scan"):
    q = np.linspace(1.0, 4.0, 301)
    y = (20.0 + 2.0 * q + 500.0 * np.exp(-0.5 * ((q - 2.5) / 0.03) ** 2)
         + 300.0 * np.exp(-0.5 * ((q - 3.1) / 0.04) ** 2))
    return FitInputSnapshot(FitInputKey(identity, f"/data/{identity}.nexus", f"fp-{identity}", 1),
                            (FitAxis("q", "q_A^-1", q),), y, wavelength_angstrom=1.0)


@pytest.mark.parametrize("model, background, shape, bg", [
    ("pseudovoigt", "linear", {"fraction"}, {"bg_slope", "bg_intercept"}),
    ("voigt", "constant", {"gamma"}, {"bg_c"}),
    ("pearson7", "chebyshev3", {"expon"}, {"bg_c0", "bg_c1", "bg_c2", "bg_c3"}),
    ("split_pseudovoigt", "none", {"fraction", "sigma_r"}, set()),
])
def test_every_peak_parameter_and_the_backgrounds_own_reach_the_trend(model, background, shape, bg):
    outcome, = run_batch(DisplayedPeakFitAnalyzer({"model": model, "background": background}), [_two_peaks()])
    assert outcome.ok, outcome.message
    values = dict(zip(outcome.result.parameter_names, outcome.result.parameter_values))
    peak = {"center", "fwhm", "amplitude", "sigma", *shape} | ({"height"} if f"p0_height" in values else set())
    # a Voigt's γ is named apart from a phase fit's lattice γ family
    trend = {"gamma": "voigt_gamma"}
    for index in (0, 1):
        for name in peak:
            key = f"{trend.get(name, name)}_{index}"
            assert outcome.params[key] == pytest.approx(values[f"p{index}_{name}"]), name
    assert {"height_0", "height_1"} <= set(outcome.params) or model == "split_pseudovoigt"
    for name in bg:
        assert outcome.params[name] == pytest.approx(values[name])
    assert set(outcome.params) - {"center_err_0", "center_err_1"} == (
        {f"{trend.get(name, name)}_{index}" for name in peak for index in (0, 1)} | bg)
    # the trend's order: the centre, its error, the widths, height and area first; the background last
    families = list(dict.fromkeys(key.rpartition("_")[0] if key.rpartition("_")[2].isdigit() else key
                                  for key in outcome.params))
    assert families[:3] == ["center", "center_err", "fwhm"] or families[:2] == ["center", "fwhm"]
    assert all(not family.startswith("bg") for family in families[:len(families) - len(bg)])


def test_a_template_backgrounds_scale_and_its_added_part_reach_the_trend():
    q = np.linspace(1.0, 4.0, 301)
    template = FitInputSnapshot(FitInputKey("bare", "/data/bare.nexus", "fp-bare", 1), (FitAxis("q", "q_A^-1", q),),
                                10.0 + 0.5 * q, wavelength_angstrom=1.0).trace_input()
    outcome, = run_batch(DisplayedPeakFitAnalyzer({"model": "gaussian", "background": "template+linear",
                                                   "background_template": template}), [_two_peaks()])
    assert outcome.ok, outcome.message
    values = dict(zip(outcome.result.parameter_names, outcome.result.parameter_values))
    assert {name for name in outcome.params if name.startswith("bg")} == {
        name for name in values if name.startswith(("bg_", "bgx_"))}
    assert "bg_A" in outcome.params


def test_the_trend_groups_background_parameters_as_families_with_no_peak():
    from xdart.gui.analysis.peak_fit_util import family_label, group_families
    grouped = group_families(["center_0", "center_1", "fraction_0", "fraction_1", "bg_slope", "bg_c0", "bgx_c1",
                              "bg_A"])
    assert grouped["center"] == [("center_0", 0), ("center_1", 1)]
    assert grouped["bg_slope"] == [("bg_slope", None)] and grouped["bg_c0"] == [("bg_c0", None)]
    assert grouped["bgx_c1"] == [("bgx_c1", None)] and grouped["bg_A"] == [("bg_A", None)]
    assert family_label("fraction") == "Lorentzian fraction" and family_label("height") == "Height"
    assert family_label("bg_slope") == "Background slope" and family_label("bg_c0") == "Background c0"
    assert family_label("bgx_c1") == "Background c1" and family_label("bg_A") == "Template scale"
    assert family_label("bg_c") == "Background constant" and family_label("bg_intercept") == "Background intercept"
    assert family_label("voigt_gamma") == "Voigt γ" and family_label("gamma") == "Lattice γ"
