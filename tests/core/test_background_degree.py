"""NEXT-128 background degrees: the headless rows H1-H6 (Claude, 2026-11-03).

The maintainer, 2026-11-03, during his 1.2.8 live check: "also, there should be more options for background -
Polynomial (this can replace Linear, constant). when polynomial is chosen, another box to choose the polynomial degree
should show in the same row. same with the chebyshev and Template + Linear (replace with template with polynomial) and
template + Chebyshev - add degree here as well"; for Phase Analysis his "Same in both (Recommended)".

Frozen before the product change (red on 57fca340). The plans fit each degree family within its range, keep the earlier
spellings (old plans, settings files and notebooks run as they did, with their parameter names), refuse by name what the
fit cannot take, and one headless owner maps every spelling to its window's choice and degree. Real fits through
``run_displayed_peak_fit`` and ``run_displayed_phase_fit``; nothing replaced. New APIs are reached inside the test
bodies, so a missing one fails its own row.
"""
import numpy as np
import pytest

from tests.core.test_phase_analysis_options import _cscl, _flat, _ok, _plan, _run, _trace, library  # noqa: F401


def _peak_trace(x, y, unit="q_A^-1", label="frame-7"):
    from tests.core.test_p37_display_fit_operations import _trace as p37_trace
    return p37_trace(axis=x, intensity=y, axis_unit=unit, label=label)


def _peak(background, **changes):
    """One Gaussian on a sloped background (50 + 10 x), fitted from its centre, through the displayed peak plan; for a
    template background, on twice the template plus a small sloped correction (5 + 0.5 x)."""
    from tests.core.test_p37_display_fit_operations import _peak_plan
    from xrd_tools.analysis.display_fit_operations import run_displayed_peak_fit
    x = np.linspace(1.0, 4.0, 601)
    peak = 200.0 * np.exp(-0.5 * ((x - 2.5) / 0.05) ** 2)
    y = 50.0 + 10.0 * x + peak
    if background.startswith("template"):
        template = 50.0 + 10.0 * np.sin(3.0 * x)
        y = 2.0 * template + 5.0 + 0.5 * x + peak
        changes.setdefault("background_template", _peak_trace(x, template, label="template"))
    plan = _peak_plan(_peak_trace(x, y), selection_mode="count", manual_centers=(2.5,), n_peaks=1, model="gaussian",
                      background=background, **changes)
    return run_displayed_peak_fit(plan)


def _background_names(result, prefix):
    return {name for name in result.parameter_names if name.startswith(prefix)}


def _coefficients(prefix, degree):
    return {f"{prefix}c{index}" for index in range(degree + 1)}


# ---- H1 the peak plan fits each degree family within its range ------------------------------------------------

@pytest.mark.parametrize("background, prefix, degree", (
    ("polynomial0", "bg_", 0), ("polynomial1", "bg_", 1), ("polynomial3", "bg_", 3), ("polynomial7", "bg_", 7),
    ("chebyshev0", "bg_", 0), ("chebyshev6", "bg_", 6), ("chebyshev15", "bg_", 15),
    ("template+polynomial0", "bgx_", 0), ("template+polynomial2", "bgx_", 2),
    ("template+chebyshev5", "bgx_", 5),
))
def test_h1_the_peak_plan_fits_each_degree_family_within_its_range(background, prefix, degree):
    result = _peak(background)
    assert result.code == "OK", (background, result.code, result.diagnostics)
    assert _background_names(result, prefix) == _coefficients(prefix, degree), sorted(result.parameter_names)
    if prefix == "bgx_":
        assert "bg_A" in result.parameter_names
    assert abs(dict(zip(result.parameter_names, result.parameter_values))["p0_center"] - 2.5) < 2e-3


# ---- H2 the earlier spellings still fit, with their own parameter names (old plans, settings and notebooks) -------

@pytest.mark.parametrize("background, names", (
    ("linear", {"bg_slope", "bg_intercept"}), ("constant", {"bg_c"}), ("chebyshev3", _coefficients("bg_", 3)),
    ("template+linear", {"bg_A", "bgx_slope", "bgx_intercept"}),
    ("template+chebyshev3", {"bg_A"} | _coefficients("bgx_", 3)),
))
def test_h2_the_earlier_spellings_still_fit_with_their_own_names(background, names):
    result = _peak(background)
    assert result.code == "OK", (background, result.code, result.diagnostics)
    assert _background_names(result, "bg") == names, sorted(result.parameter_names)


def test_h2_linear_and_polynomial_one_are_the_same_fit():
    """Polynomial 1 starts where Linear did (c0 = intercept = min(y), c1 = slope = 0, neither bounded), and ends
    there."""
    linear, polynomial = _peak("linear"), _peak("polynomial1")
    assert linear.code == polynomial.code == "OK"
    one = dict(zip(linear.parameter_names, linear.parameter_values))
    two = dict(zip(polynomial.parameter_names, polynomial.parameter_values))
    assert two["bg_c0"] == pytest.approx(one["bg_intercept"], rel=1e-9)
    assert two["bg_c1"] == pytest.approx(one["bg_slope"], rel=1e-9)
    for name in ("p0_center", "p0_sigma", "p0_amplitude"):
        assert two[name] == pytest.approx(one[name], rel=1e-9), name
    np.testing.assert_allclose(polynomial.fit, linear.fit, rtol=1e-9, atol=1e-9)


# ---- H3 the peak plan refuses by name what the fit cannot take ------------------------------------------------

@pytest.mark.parametrize("background", (
    "polynomial8",  # lmfit's PolynomialModel stops at 7: refused before the fit, never a TypeError mid-fit
    "chebyshev16", "template+polynomial8", "template+chebyshev16",
    "polynomial", "chebyshev", "polynomial01", "polynomial-1", "polynomial 2", "Polynomial2",
    "template+constant", "template+template", "spline4", "cubic", "",
))
def test_h3_the_peak_plan_refuses_a_background_the_fit_cannot_take(background):
    result = _peak(background)  # a template+ spelling carries its template, so only the background is wrong
    assert result.code == "INVALID_PEAK_PLAN", (background, result.code, result.diagnostics)
    assert "background" in " ".join(result.diagnostics), result.diagnostics
    assert result.fit is None


# ---- H4 one owner maps every peak spelling to its window's choice and degree ------------------------------------

def test_h4_one_owner_maps_every_peak_spelling_to_its_choice_and_degree():
    from xrd_tools.analysis.display_fit_operations import (
        PEAK_BACKGROUND_DEGREES, peak_background, peak_background_choice,
    )
    assert PEAK_BACKGROUND_DEGREES == {"polynomial": (0, 7), "chebyshev": (0, 15)}
    expected = {
        "none": ("none", None), "template": ("template", None),
        "polynomial0": ("polynomial", 0), "polynomial7": ("polynomial", 7), "chebyshev15": ("chebyshev", 15),
        "template+polynomial2": ("template+polynomial", 2), "template+chebyshev6": ("template+chebyshev", 6),
        # the earlier choices, as his request replaces them
        "linear": ("polynomial", 1), "constant": ("polynomial", 0), "chebyshev3": ("chebyshev", 3),
        "template+linear": ("template+polynomial", 1), "template+chebyshev3": ("template+chebyshev", 3),
    }
    for text, choice in expected.items():
        assert peak_background_choice(text) == choice, text
    class Alias(str):
        """A str subclass equal to a choice: not the plan's exact type."""
    for text in ("polynomial8", "chebyshev16", "polynomial", "polynomial01", "cubic", "template+constant", None, 3,
                 Alias("none"), Alias("polynomial1")):
        with pytest.raises(ValueError):
            peak_background_choice(text)
    for text in ("polynomial", "polynomial8", "template+polynomial9"):  # no degree (a refused box), or beyond 7
        with pytest.raises(ValueError, match="needs a polynomial degree from 0 to 7"):
            peak_background_choice(text)
    assert peak_background("polynomial", 1) == "polynomial1"
    assert peak_background("template+chebyshev", 6) == "template+chebyshev6"
    assert peak_background("none", None) == "none" and peak_background("template", None) == "template"
    for text in ("polynomial3", "chebyshev0", "template+polynomial7", "template", "none"):
        assert peak_background(*peak_background_choice(text)) == text, text


# ---- H5 the phase plan takes Polynomial 1 (the earlier Linear) and keeps the earlier values ---------------------

@pytest.mark.parametrize("choice, degree, prefix, highest", (
    ("polynomial", 1, "bg_", 1), ("polynomial", 0, "bg_", 0), ("chebyshev", 0, "bg_", 0), ("chebyshev", 1, "bg_", 1),
    ("template+polynomial", 1, "bgx_", 1),
))
def test_h5_the_phase_plan_fits_the_new_low_degrees(library, tmp_path, choice, degree, prefix, highest):
    entry = _cscl(library, tmp_path)
    q, y = _flat()
    extra = {"background_template": _trace(q, 1. + .2 * np.sin(q), label="substrate")} if "+" in choice else {}
    result = _ok(_run(_plan(_trace(q, y), (entry,), fit_background=choice, fit_background_degree=degree, **extra)))
    assert {name for name in result.parameter_names if name.startswith(prefix + "c")} == _coefficients(
        prefix, highest), sorted(result.parameter_names)


def test_h5_the_phase_plan_keeps_the_earlier_values_and_refuses_beyond_each_range(library, tmp_path):
    entry = _cscl(library, tmp_path)
    trace = _trace(*_flat())
    for options in ({"fit_background": "linear", "fit_background_degree": 3},
                    {"fit_background": "spline", "fit_background_degree": 2},  # fitted with four knots, as before
                    {"fit_background": "spline", "fit_background_degree": 3},
                    {"fit_background": "polynomial", "fit_background_degree": 15},
                    {"fit_background": "none", "fit_background_degree": 3}):
        _ok(_run(_plan(trace, (entry,), max_nfev=1, **options)))
    for options in ({"fit_background": "polynomial", "fit_background_degree": 16},
                    {"fit_background": "polynomial", "fit_background_degree": -1},
                    {"fit_background": "chebyshev", "fit_background_degree": -1},
                    {"fit_background": "spline", "fit_background_degree": 1},
                    {"fit_background": "polynomial", "fit_background_degree": 2.0}):
        result = _run(_plan(trace, (entry,), max_nfev=1, **options))
        assert result.code == "INVALID_PHASE_PLAN", (options, result.code)
        assert "fit_background_degree" in " ".join(result.diagnostics), (options, result.diagnostics)


# ---- H6 one owner maps a phase plan's background to its window's choice and degree --------------------------------

def test_h6_one_owner_maps_a_phase_background_to_its_choice_and_degree():
    from xrd_tools.analysis.display_fit_operations import PHASE_BACKGROUND_DEGREES, phase_background_choice
    assert PHASE_BACKGROUND_DEGREES == {"polynomial": (0, 15), "chebyshev": (0, 15), "spline": (4, 15)}
    expected = {
        ("linear", 3): ("polynomial", 1), ("template+linear", 7): ("template+polynomial", 1),
        ("spline", 2): ("spline", 4), ("spline", 3): ("spline", 4),  # the four knots the fit used
        ("spline", 8): ("spline", 8), ("polynomial", 5): ("polynomial", 5), ("chebyshev", 4): ("chebyshev", 4),
        ("template+chebyshev", 3): ("template+chebyshev", 3), ("none", 3): ("none", 3),
        ("template", 9): ("template", 9),
    }
    for (background, degree), choice in expected.items():
        assert phase_background_choice(background, degree) == choice, (background, degree)
