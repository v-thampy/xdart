"""PHASE-ANALYSIS-128 frozen headless oracle K1-K9 (Claude, 2026-10-29).

The displayed phase fit runs on real CIF and lattice entries kept in a real phase library, through
``run_displayed_phase_fit``, ``DisplayedPhaseFitAnalyzer``, the notebook recipes and the real lmfit kernel. No
listing, kernel or solver is replaced. New plan fields and APIs are reached inside the test bodies, so a missing one
fails its own row and never the collection of the keep rows (K1).
"""
import json
from pathlib import Path

import numpy as np
import pytest

WL = .729
RECEIPT_PATH = Path("/phase-analysis-128/output.nxs")
#: a literal CIF, so the library id (the SHA256 of its bytes) is the same on every platform and pymatgen version
CSCL_CIF = """data_CsCl
_symmetry_space_group_name_H-M   'P m -3 m'
_cell_length_a   4.10000000
_cell_length_b   4.10000000
_cell_length_c   4.10000000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   221
loop_
 _symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Cs  Cs0  0.00000000  0.00000000  0.00000000  1
  Cl  Cl1  0.50000000  0.50000000  0.50000000  1
"""
#: the parent's (83a41407) default-plan parameter names and plan fingerprints on the exactly representable K1 input
PARENT_NAMES = ('p0_scale', 'p0_a', 'p0_b', 'p0_c', 'p0_alpha', 'p0_beta', 'p0_gamma', 'p0_q_shift', 'p0_U', 'p0_V',
                'p0_W', 'p0_fraction', 'q_shift')
PARENT_FINGERPRINT = '313b53ba2bed02a19de07f57db16c729d717a48c87814a3161b8e0f4a90b939b'
PARENT_MIXED_FINGERPRINT = 'b031ac24304b7774e97bc63f4ee033862a08c377a9a2d81958b989f9403ab832'


@pytest.fixture
def library(tmp_path, monkeypatch):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "library"))
    return PhaseLibrary(tmp_path / "library")


def _cscl(library, root, name="CsCl", a=4.1):
    text = CSCL_CIF.replace("4.10000000", f"{a:.8f}").replace("data_CsCl", f"data_{name}")
    path = Path(root) / f"{name}.cif"
    path.write_text(text, encoding="utf-8")
    return library.add_cif(path, name=name)


def _receipt(label):
    from xrd_tools.analysis.display_fit_operations import DisplayedTraceReceipt
    return DisplayedTraceReceipt("displayed-trace-v1", 1, "run", "1", RECEIPT_PATH, label, 0)


def _trace(q, y, *, unit="1/angstrom", label="trace", sigma=None):
    from xrd_tools.analysis.display_fit_operations import DisplayedTraceInput
    return DisplayedTraceInput(axis=np.asarray(q, dtype=float), intensity=np.asarray(y, dtype=float), label=label,
                               axis_unit=unit, title=label, epoch=1, receipt=_receipt(label), sigma=sigma)


def _flat(n=251):
    """Exactly representable values: the trace fingerprint is the same on every platform."""
    q = np.linspace(1., 6., n)
    return q, np.ones_like(q) + 2.0 * (q > 3.)


def _plan(trace, entries, **options):
    from xrd_tools.analysis.display_fit_operations import DisplayedPhaseFitPlan
    options.setdefault("max_nfev", 5)
    return DisplayedPhaseFitPlan(trace=trace, cif_paths=(), expected_hashes=(),
                                 phase_names=tuple(entry.name for entry in entries), wavelength_angstrom=WL,
                                 phase_entries=tuple(entries), **options)


def _run(plan):
    from xrd_tools.analysis.display_fit_operations import run_displayed_phase_fit
    return run_displayed_phase_fit(plan)


def _ok(result):
    assert result.code == "OK", (result.code, result.diagnostics)
    return result


def _completed(outcome):
    """A run that completed (its solve need not converge: ``AnalysisOutcome.ok`` is lmfit's success)."""
    return outcome.result is not None and outcome.result.code == "OK"


def _values(result):
    return dict(zip(result.parameter_names, result.parameter_values))


def _synthetic(entry, q, *, scale=40., q_shift=0., sigma=.012, background=2.):
    """The kernel's own Gaussian, fixed-width model of ``entry`` on ``q`` at known truth."""
    from xrd_tools.analysis.phase_references import phase_model
    from xrd_tools.analysis.fitting.phase_fitting import PhaseFitter
    extent = (float(q.min()), float(q.max()))
    model = phase_model(entry, wavelength_angstrom=WL, q_range=extent)
    fitter = PhaseFitter(q, np.zeros_like(q))
    fitter.add_phase(model, q_range=extent, min_intensity=0.)
    params = fitter.build_parameters(phase_profile="gaussian", width_model="fixed")
    params["p0_scale"].set(value=scale)
    params["q_shift"].set(value=q_shift)
    params["p0_sigma"].set(value=sigma)
    return fitter.eval_model(params) + background


def _direct(trace, entries, *, profile="pseudovoigt", lattice_pct=.05, min_intensity=0., max_nfev=5, **fit):
    """The parent's runner sequence (display_fit_operations.py :634-663 at 83a41407), called on the kernel."""
    from pymatgen.core import Structure
    from xrd_tools.analysis.phase import PhaseModel
    from xrd_tools.analysis.phase_references import prepare_phase_models
    from xrd_tools.analysis.fitting.phase_fitting import PhaseFitter
    x, y = np.asarray(trace.axis), np.asarray(trace.intensity)
    extent = (float(x.min()), float(x.max()))
    phases, _references = prepare_phase_models(*entries, wavelength_angstrom=WL, q_range=extent,
                                                min_intensity=min_intensity, floors=None, model_class=PhaseModel,
                                                structure_class=Structure)
    fitter = PhaseFitter(x, y, prefit_background="none")
    for phase in phases:
        fitter.add_phase(phase, q_range=extent, min_intensity=min_intensity)
    params = fitter.build_parameters(lattice_pct=lattice_pct, phase_profile=profile, texture="none")
    return fitter.fit(params=params, phase_profile=profile, lattice_pct=lattice_pct, texture="none",
                      max_nfev=max_nfev, **fit)


# ---- K1 keep: today's defaults ----------------------------------------------------------------------------------

def test_k1_a_default_plan_keeps_the_parents_parameters_and_fingerprints(library, tmp_path):
    entry = _cscl(library, tmp_path)
    lattice = library.add_lattice("cell", a=4.2, space_group="Pm-3m")
    trace = _trace(*_flat())
    result = _ok(_run(_plan(trace, (entry,))))
    assert result.parameter_names == PARENT_NAMES
    assert result.plan_fingerprint == PARENT_FINGERPRINT
    mixed = _ok(_run(_plan(trace, (entry, lattice))))
    assert mixed.plan_fingerprint == PARENT_MIXED_FINGERPRINT
    assert getattr(result, "fit_background", None) is None and getattr(result, "amorphous", None) is None


def test_k1_b_a_default_fit_equals_the_parents_kernel_sequence(library, tmp_path):
    entry = _cscl(library, tmp_path)
    q = np.linspace(1., 6., 401)
    trace = _trace(q, _synthetic(entry, q, q_shift=.004))
    result = _ok(_run(_plan(trace, (entry,), max_nfev=60)))
    backend = _direct(trace, (entry,), max_nfev=60)
    assert result.parameter_names == tuple(backend.params)
    np.testing.assert_array_equal(np.asarray(result.parameter_values),
                                  np.asarray([backend.params[name].value for name in result.parameter_names]))


# ---- K2 every option reaches the kernel -------------------------------------------------------------------------

def _profiles():
    from xrd_tools.analysis.fitting.phase_fitting import _PHASE_PROFILE_SPEC
    return tuple(_PHASE_PROFILE_SPEC)


@pytest.mark.parametrize("profile", ("gaussian", "lorentzian", "pseudovoigt", "voigt", "lorentzian_squared",
                                     "pearson7", "splitlorentzian", "moffat", "pearson4", "studentst",
                                     "skewedgaussian"))
def test_k2_every_notebook_profile_reaches_the_kernel(library, tmp_path, profile):
    from xrd_tools.analysis.fitting.phase_fitting import _PHASE_PROFILE_SPEC
    assert set(_profiles()) == {"gaussian", "lorentzian", "pseudovoigt", "voigt", "lorentzian_squared", "pearson7",
                                "splitlorentzian", "moffat", "pearson4", "studentst", "skewedgaussian"}
    entry = _cscl(library, tmp_path)
    result = _ok(_run(_plan(_trace(*_flat()), (entry,), phase_profile=profile)))
    names = set(result.parameter_names)
    own = {f"p0_{extra}" for extra in _PHASE_PROFILE_SPEC[profile]["extras"]}
    others = {f"p0_{extra}" for spec in _PHASE_PROFILE_SPEC.values() for extra in spec["extras"]} - own
    assert own <= names and not (others & names), (profile, sorted(names))


@pytest.mark.parametrize("model, present, absent", (
    ("caglioti", {"p0_U", "p0_V", "p0_W"}, {"p0_D", "p0_eps", "p0_sigma"}),
    ("scherrer", {"p0_D", "p0_eps"}, {"p0_U", "p0_V", "p0_W", "p0_sigma"}),
    ("fixed", {"p0_sigma"}, {"p0_U", "p0_V", "p0_W", "p0_D", "p0_eps"}),
))
def test_k2_each_width_model_reaches_the_kernel(library, tmp_path, model, present, absent):
    entry = _cscl(library, tmp_path)
    names = set(_ok(_run(_plan(_trace(*_flat()), (entry,), width_model=model))).parameter_names)
    assert present <= names and not (absent & names), (model, sorted(names))


def test_k2_march_dollase_and_free_texture_reach_the_kernel_within_their_range(library, tmp_path):
    entry = _cscl(library, tmp_path)
    q = np.linspace(1., 6., 401)
    trace = _trace(q, _synthetic(entry, q))
    march = _ok(_run(_plan(trace, (entry,), texture="march_dollase", march_axis=(1, 1, 0))))
    assert "p0_march_r" in march.parameter_names
    free = _ok(_run(_plan(trace, (entry,), texture="free", free_height_range=(.2, .5), max_nfev=40)))
    heights = {name: value for name, value in _values(free).items() if name.startswith("p0_pk")}
    assert len(heights) == len(free.marker_positions) > 3
    assert all(.2 <= value <= .5 for value in heights.values()), heights
    assert "p0_march_r" not in free.parameter_names


@pytest.mark.parametrize("choice, degree, present", (
    ("linear", 3, {"bg_c0", "bg_c1"}),
    ("polynomial", 3, {"bg_c0", "bg_c1", "bg_c2", "bg_c3"}),
    ("chebyshev", 4, {"bg_c0", "bg_c1", "bg_c2", "bg_c3", "bg_c4"}),
))
def test_k2_each_fitted_background_reaches_the_kernel(library, tmp_path, choice, degree, present):
    entry = _cscl(library, tmp_path)
    result = _ok(_run(_plan(_trace(*_flat()), (entry,), fit_background=choice, fit_background_degree=degree)))
    names = set(result.parameter_names)
    highest = 1 if choice == "linear" else degree
    assert present <= names, sorted(names)
    assert not any(name.startswith("bg_c") and int(name[4:]) > highest for name in names), sorted(names)
    assert result.fit_background is not None and result.fit_background.shape == result.fit.shape


def test_k2_a_spline_background_has_at_least_four_knots(library, tmp_path):
    entry = _cscl(library, tmp_path)
    result = _ok(_run(_plan(_trace(*_flat()), (entry,), fit_background="spline", fit_background_degree=2)))
    assert sum(name.startswith("bg_s") for name in result.parameter_names) >= 4


@pytest.mark.parametrize("choice, extra", (
    ("template", set()),
    ("template+linear", {"bgx_c0", "bgx_c1"}),
    ("template+polynomial", {"bgx_c0", "bgx_c1", "bgx_c2", "bgx_c3"}),
    ("template+chebyshev", {"bgx_c0", "bgx_c1", "bgx_c2", "bgx_c3"}),
))
def test_k2_the_substrate_template_reaches_the_kernel(library, tmp_path, choice, extra):
    entry = _cscl(library, tmp_path)
    q, y = _flat()
    substrate = _trace(q, 1. + .2*np.sin(q), label="substrate")
    result = _ok(_run(_plan(_trace(q, y), (entry,), fit_background=choice, fit_background_degree=3,
                            background_template=substrate)))
    names = set(result.parameter_names)
    assert "bg_A" in names and extra <= names, sorted(names)
    assert not any(name.startswith("bg_c") for name in names)


@pytest.mark.parametrize("profile", ("gaussian", "pseudovoigt", "lorentzian", "voigt", "lorentzian_squared"))
def test_k2_the_amorphous_peak_reaches_the_kernel(library, tmp_path, profile):
    entry = _cscl(library, tmp_path)
    result = _ok(_run(_plan(_trace(*_flat()), (entry,), amorphous_peak=profile, amorphous_center=1.55,
                            amorphous_sigma=.3, max_nfev=1)))
    values = _values(result)
    assert {"am_amplitude", "am_center", "am_sigma"} <= set(values), sorted(values)
    assert abs(values["am_center"] - 1.55) < .5
    assert result.amorphous is not None and result.amorphous.shape == result.fit.shape


def test_k2_the_q_shift_bound_limits_the_fitted_shift(library, tmp_path):
    entry = _cscl(library, tmp_path)
    q = np.linspace(1., 6., 401)
    trace = _trace(q, _synthetic(entry, q, q_shift=.03))
    tight = _values(_ok(_run(_plan(trace, (entry,), q_shift_bound=.01, width_model="fixed", max_nfev=300))))
    loose = _values(_ok(_run(_plan(trace, (entry,), q_shift_bound=.05, width_model="fixed", max_nfev=300))))
    assert abs(tight["q_shift"]) <= .01 + 1e-12
    assert abs(loose["q_shift"] - .03) < .005, loose["q_shift"]


def test_k2_the_cross_phase_lock_orders_the_same_axis_across_phases(library, tmp_path):
    first, second = _cscl(library, tmp_path, "low", 4.1), _cscl(library, tmp_path, "high", 4.25)
    trace = _trace(*_flat())
    locked = _ok(_run(_plan(trace, (first, second), lock_cross_phase=True)))
    free = _ok(_run(_plan(trace, (first, second), lock_cross_phase=False)))
    assert any(name.startswith("cross_gap_a_") for name in locked.parameter_names), locked.parameter_names
    assert not any(name.startswith("cross_gap_") for name in free.parameter_names)


def test_k2_the_snip_width_reaches_the_prefit_background(library, tmp_path):
    entry = _cscl(library, tmp_path)
    q = np.linspace(1., 6., 401)
    trace = _trace(q, _synthetic(entry, q, background=5.) + 3.*q)
    narrow = _ok(_run(_plan(trace, (entry,), prefit_background="snip", snip_width=3, max_nfev=1)))
    wide = _ok(_run(_plan(trace, (entry,), prefit_background="snip", snip_width=60, max_nfev=1)))
    auto = _ok(_run(_plan(trace, (entry,), prefit_background="snip", max_nfev=1)))
    parent = _ok(_run(_plan(trace, (entry,), prefit_background="snip", snip_width=None, max_nfev=1)))
    assert not np.array_equal(narrow.background, wide.background)
    np.testing.assert_array_equal(auto.background, parent.background)


def test_k2_width_limits_bound_a_fixed_width(library, tmp_path):
    entry = _cscl(library, tmp_path)
    q = np.linspace(1., 6., 401)
    trace = _trace(q, _synthetic(entry, q, sigma=.02))
    capped = _values(_ok(_run(_plan(trace, (entry,), width_model="fixed", width_limits=(None, .005),
                                    max_nfev=200))))
    assert capped["p0_sigma"] <= .005 + 1e-12


# ---- K3 fit range: the full grid is prepared and returned, only the solve is masked -----------------------------

def test_k3_the_fit_range_masks_only_the_solve_as_the_notebook_kernel_does(library, tmp_path):
    from xrd_tools.core.containers import convert_radial_axis
    entry = _cscl(library, tmp_path)
    q = np.linspace(1., 6., 401)
    trace = _trace(q, _synthetic(entry, q, q_shift=.004))
    bounds = (2.0105, 4.4905)
    result = _ok(_run(_plan(trace, (entry,), fit_range=bounds, max_nfev=60)))
    backend = _direct(trace, (entry,), max_nfev=60, q_range=bounds)
    np.testing.assert_array_equal(np.asarray(result.parameter_values),
                                  np.asarray([backend.params[name].value for name in result.parameter_names]))
    assert result.fit.shape == q.shape and result.residual.shape == q.shape
    two_theta = tuple(float(v) for v in convert_radial_axis(np.asarray(bounds), "q_A^-1", "2th_deg", WL))
    converted = _ok(_run(_plan(trace, (entry,), fit_range=two_theta, fit_range_unit="2th_deg", max_nfev=60)))
    assert converted.parameter_values == result.parameter_values
    narrow = _run(_plan(trace, (entry,), fit_range=(2.003, 2.103)))  # eight samples
    assert narrow.code == "INSUFFICIENT_FIT_POINTS", (narrow.code, narrow.diagnostics)


# ---- K4 validation by field --------------------------------------------------------------------------------------

def _bad_cases():
    q, y = _flat()
    substrate = _trace(q, np.ones_like(q), label="substrate")
    in_two_theta = _trace(np.linspace(10., 40., 251), np.ones(251), unit="2th_deg", label="substrate 2θ")
    single = _trace(np.array([2.]), np.array([1.]), label="one point")
    return (
        ("width_model", {"width_model": "voigt"}),
        ("texture", {"texture": "md"}),
        ("march_axis", {"texture": "march_dollase", "march_axis": (0, 0, 0)}),
        ("march_axis", {"texture": "march_dollase", "march_axis": (1, 0)}),
        ("free_height_range", {"texture": "free", "free_height_range": (1., .5)}),
        ("free_height_range", {"texture": "free", "free_height_range": (-.1, 1.)}),
        ("q_shift_bound", {"q_shift_bound": 0.}),
        ("q_shift_bound", {"q_shift_bound": -.01}),
        ("lock_cross_phase", {"lock_cross_phase": 1}),
        ("snip_width", {"prefit_background": "snip", "snip_width": 0}),
        ("snip_width", {"prefit_background": "snip", "snip_width": 1.5}),
        ("fit_background", {"fit_background": "cubic"}),
        ("fit_background_degree", {"fit_background": "polynomial", "fit_background_degree": 1}),
        ("fit_background_degree", {"fit_background": "chebyshev", "fit_background_degree": 16}),
        ("background_template", {"fit_background": "template"}),
        ("background_template", {"fit_background": "none", "background_template": substrate}),
        ("background_template", {"fit_background": "template", "background_template": in_two_theta}),
        ("background_template", {"fit_background": "template", "background_template": single}),
        ("amorphous_peak", {"amorphous_peak": "gauss"}),
        ("amorphous_sigma", {"amorphous_peak": "gaussian", "amorphous_sigma": 0.}),
        ("amorphous_center", {"amorphous_peak": "gaussian", "amorphous_center": 9.}),
        ("fit_range", {"fit_range": (4., 2.)}),
        ("fit_range_unit", {"fit_range": (2., 4.), "fit_range_unit": "deg"}),
        ("width_limits", {"width_model": "fixed", "width_limits": (.05, .01)}),
        ("width_limits", {"width_model": "fixed", "width_limits": (-1., None)}),
        ("initial_values", {"initial_values": (("q_shift", float("nan")),),
                            "initial_topology": (("x", "cif", ((1, 0, 0),)),)}),
        ("initial_topology", {"initial_values": (("q_shift", 0.),)}),
    )


def test_k4_the_case_table_has_twenty_seven_rows():
    assert len(_bad_cases()) == 27


@pytest.mark.parametrize("index", range(27))
def test_k4_each_invalid_new_field_refuses_by_name(library, tmp_path, index):
    field, options = _bad_cases()[index]
    entry = _cscl(library, tmp_path)
    result = _run(_plan(_trace(*_flat()), (entry,), **options))
    assert result.code == "INVALID_PHASE_PLAN", (field, result.code, result.diagnostics)
    assert field in " ".join(result.diagnostics), (field, result.diagnostics)
    assert result.fit is None


def test_k4_the_domain_boundaries_are_admitted(library, tmp_path):
    entry = _cscl(library, tmp_path)
    q, y = _flat()
    trace = _trace(q, y)
    for options in ({"fit_background": "polynomial", "fit_background_degree": 2},
                    {"fit_background": "chebyshev", "fit_background_degree": 15},
                    {"fit_background": "spline", "fit_background_degree": 2},
                    {"texture": "free", "free_height_range": (0., 100.)},
                    {"q_shift_bound": .5},
                    {"prefit_background": "snip", "snip_width": len(q)},
                    {"width_model": "fixed", "width_limits": (None, None)},
                    {"fit_range": (1., 6.), "fit_range_unit": "q_A^-1"}):
        _ok(_run(_plan(trace, (entry,), max_nfev=1, **options)))


# ---- K5 the existing exact 512-parameter cap ---------------------------------------------------------------------

def test_k5_the_exact_cap_refuses_a_model_the_early_estimate_admits(library, tmp_path):
    cell = library.add_lattice("many heights", a=8.9, b=9.5, c=10.1, space_group=1)
    q = np.linspace(1., 6., 2001)
    trace = _trace(q, np.ones_like(q) + q)
    under = _ok(_run(_plan(trace, (cell,), phase_profile="gaussian", max_nfev=1)))
    assert 490 < len(under.parameter_names) <= 512, len(under.parameter_names)
    over = _run(_plan(trace, (cell,), phase_profile="gaussian", fit_background="chebyshev",
                      fit_background_degree=15, max_nfev=1))
    assert over.code == "FIT_PARAMETER_LIMIT_EXCEEDED", (over.code, over.diagnostics)
    assert over.fit is None and over.parameter_names == ()


def test_k5_free_cif_heights_count_toward_the_exact_cap(library, tmp_path):
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    path = tmp_path / "triclinic.cif"
    path.write_text(str(CifWriter(Structure(Lattice.from_parameters(8.9, 9.5, 10.1, 90, 90, 90), ["Si"],
                                            [[0, 0, 0]]))), encoding="utf-8")
    entry = library.add_cif(path, name="many")
    q = np.linspace(1., 6., 2001)
    trace = _trace(q, np.ones_like(q) + q)
    plain = _ok(_run(_plan(trace, (entry,), phase_profile="gaussian", max_nfev=1)))
    free = _run(_plan(trace, (entry,), phase_profile="gaussian", texture="free", fit_background="chebyshev",
                      fit_background_degree=15, max_nfev=1))
    assert len(plain.parameter_names) < 40
    assert free.code == "FIT_PARAMETER_LIMIT_EXCEEDED", (free.code, free.diagnostics)


# ---- K7 sequential seeding --------------------------------------------------------------------------------------

def _inputs(entry, *extents):
    from xrd_tools.analysis.runner import AnalysisInput
    rows = []
    for index, (lo, hi) in enumerate(extents):
        q = np.linspace(lo, hi, 301)
        y = _synthetic(entry, q, q_shift=.006) if hi > lo else np.full(301, np.nan)
        rows.append(AnalysisInput(label=str(index), x=q, y=y, x_unit="1/angstrom"))
    return rows


def _batch(entry, extents, *, sequential):
    from xrd_tools.analysis.runner import DisplayedPhaseFitAnalyzer, run_batch
    settings = {"cif_paths": (), "expected_hashes": (), "phase_names": (entry.name,), "phase_entries": (entry,),
                "max_nfev": 8, "width_model": "fixed"}  # about one iteration: a seed is visible in the values
    analyzer = DisplayedPhaseFitAnalyzer(settings=settings, wavelength_angstrom=WL, sequential=sequential)
    return run_batch(analyzer, _inputs(entry, *extents))


def test_k7_sequential_seeds_each_fit_from_the_previous_and_a_changed_listing_drops_the_seed(library, tmp_path):
    entry = _cscl(library, tmp_path)
    same, wider = (1., 6.), (1.6, 6.)  # 1.6 Å⁻¹ drops CsCl's (100) at 1.53 Å⁻¹: a changed listing
    plain = _batch(entry, (same, same), sequential=False)
    assert all(_completed(outcome) for outcome in plain)
    assert plain[1].result.parameter_values == plain[0].result.parameter_values  # identical input and start
    seeded = _batch(entry, (same, same, wider), sequential=True)
    assert seeded[0].result.parameter_values == plain[0].result.parameter_values  # each batch starts empty
    assert seeded[1].result.parameter_values != seeded[0].result.parameter_values  # started from frame 0's values
    assert seeded[1].result.plan_fingerprint != plain[1].result.plan_fingerprint
    assert "PHASE_SEED_DROPPED" not in " ".join(seeded[1].result.diagnostics)
    assert "PHASE_SEED_DROPPED" in " ".join(seeded[2].result.diagnostics), seeded[2].result.diagnostics


def test_k7_a_refused_frame_keeps_the_previous_seed(library, tmp_path):
    entry = _cscl(library, tmp_path)
    same, empty = (1., 6.), (3., 3.)
    straight = _batch(entry, (same, same), sequential=True)
    through = _batch(entry, (same, empty, same), sequential=True)
    assert not _completed(through[1])
    assert through[2].result.parameter_values == straight[1].result.parameter_values


# ---- K8 fingerprints, JSON and the notebook recipes -------------------------------------------------------------

NON_DEFAULT = (
    {"width_model": "scherrer"}, {"width_limits": (None, .05)}, {"texture": "march_dollase"},
    {"texture": "march_dollase", "march_axis": (1, 1, 1)}, {"texture": "free", "free_height_range": (0., 5.)},
    {"q_shift_bound": .02}, {"lock_cross_phase": True}, {"prefit_background": "snip", "snip_width": 9},
    {"fit_background": "chebyshev"}, {"fit_background": "chebyshev", "fit_background_degree": 5},
    {"amorphous_peak": "gaussian"}, {"amorphous_peak": "gaussian", "amorphous_center": 2.},
    {"amorphous_peak": "gaussian", "amorphous_sigma": .4}, {"fit_range": (1.5, 5.5)},
)
DEFAULTS = {"width_model": "caglioti", "width_limits": (None, None), "texture": "none", "march_axis": (0, 0, 1),
            "free_height_range": (0., 10.), "q_shift_bound": .05, "lock_cross_phase": False, "snip_width": None,
            "fit_background": "none", "fit_background_degree": 3, "background_template": None,
            "amorphous_peak": "none", "amorphous_center": 1.5, "amorphous_sigma": .3, "fit_range": None,
            "fit_range_unit": "q_A^-1", "initial_values": (), "initial_topology": ()}


def test_k8_each_non_default_field_changes_the_plan_fingerprint_and_defaults_do_not(library, tmp_path):
    entry = _cscl(library, tmp_path)
    trace = _trace(*_flat())
    base = _ok(_run(_plan(trace, (entry,), max_nfev=1))).plan_fingerprint
    assert _ok(_run(_plan(trace, (entry,), max_nfev=1, **DEFAULTS))).plan_fingerprint == base
    seen = {base}
    for options in NON_DEFAULT:
        fingerprint = _ok(_run(_plan(trace, (entry,), max_nfev=1, **options))).plan_fingerprint
        assert fingerprint not in seen, options
        seen.add(fingerprint)
    q, _y = _flat()
    one = _trace(q, np.ones_like(q), label="substrate A")
    two = _trace(q, 1. + q, label="substrate B")
    first = _ok(_run(_plan(trace, (entry,), max_nfev=1, fit_background="template", background_template=one)))
    second = _ok(_run(_plan(trace, (entry,), max_nfev=1, fit_background="template", background_template=two)))
    assert len({first.plan_fingerprint, second.plan_fingerprint} | seen) == len(seen) + 2


def test_k8_a_plan_from_json_loaded_settings_equals_the_tuple_plan(library, tmp_path):
    entry = _cscl(library, tmp_path)
    trace = _trace(*_flat())
    options = {"texture": "march_dollase", "march_axis": (1, 1, 0), "fit_range": (1.5, 5.5),
               "width_model": "fixed", "width_limits": (None, .05), "free_height_range": (0., 5.)}
    loaded = json.loads(json.dumps(options))
    assert isinstance(loaded["march_axis"], list)
    tuple_plan = _ok(_run(_plan(trace, (entry,), max_nfev=1, **options)))
    json_plan = _ok(_run(_plan(trace, (entry,), max_nfev=1, **loaded)))
    assert json_plan.plan_fingerprint == tuple_plan.plan_fingerprint
    assert json_plan.parameter_values == tuple_plan.parameter_values


def _recipe(tmp_path, entry, settings, *, batch=1):
    from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
    from xrd_tools.notebooks import FitNotebookRecipe, capture_snapshot
    q = np.linspace(1., 6., 301)
    def snapshot(index):  # each batch input its own key: a compact batch keys outcomes by label
        return FitInputSnapshot(FitInputKey(f"phase analysis 128 #{index}", "derived", "receipt", 4,
                                            selection=(("meaning", "radial"),)),
                                (FitAxis("q", "q_A^-1", q),), _synthetic(entry, q, q_shift=.006),
                                wavelength_angstrom=WL)
    native = snapshot(0)
    captured = capture_snapshot(native, "phase.npz")
    (tmp_path / "phase.npz").write_bytes(captured.content)
    references = [captured.reference]
    for index in range(1, batch):
        more = capture_snapshot(snapshot(index), f"phase{index}.npz")
        (tmp_path / f"phase{index}.npz").write_bytes(more.content)
        references.append(more.reference)
    plain = {"cif_paths": (f"{entry.name}.cif",), "expected_hashes": (None,), "phase_names": (entry.name,),
             **settings}
    recipe = FitNotebookRecipe("phase", captured.reference, settings=plain, batch=tuple(references))
    return recipe, native, {**plain, "cif_paths": (tmp_path / f"{entry.name}.cif",)}


def test_k8_the_notebook_recipe_replays_every_new_option(library, tmp_path):
    from xrd_tools.analysis.fit_sources import phase_fit_snapshot
    from xrd_tools.analysis.runner import DisplayedPhaseFitAnalyzer
    from xrd_tools.notebooks import recipe_from_json, recipe_to_json, render_notebook, run_recipe
    entry = _cscl(library, tmp_path)
    options = {"phase_profile": "pearson7", "width_model": "fixed", "texture": "march_dollase",
               "march_axis": (1, 1, 0), "fit_background": "chebyshev", "fit_background_degree": 4,
               "amorphous_peak": "gaussian", "amorphous_center": 2.2, "amorphous_sigma": .3, "q_shift_bound": .02,
               "lock_cross_phase": False, "snip_width": None, "fit_range": (1.5, 5.5), "fit_range_unit": "q_A^-1",
               "max_nfev": 20}
    recipe, native, direct_settings = _recipe(tmp_path, entry, options)
    assert recipe_from_json(recipe_to_json(recipe)) == recipe
    direct = DisplayedPhaseFitAnalyzer(direct_settings).analyze(phase_fit_snapshot(native))
    assert _completed(direct), direct.message
    replayed = run_recipe(recipe, bundle_root=tmp_path)
    assert _completed(replayed), replayed.message
    assert replayed.params == pytest.approx(direct.params, rel=1e-12)
    document = render_notebook(recipe, format="jupyter", bundle_root=str(tmp_path))
    namespace = {"NOTEBOOK_HEADLESS": True}
    for cell in json.loads(document)["cells"]:
        if cell["cell_type"] == "code":
            exec(cell["source"], namespace)
    assert _completed(namespace["outcome"])
    assert namespace["outcome"].params == pytest.approx(direct.params, rel=1e-12)


def test_k8_a_sequential_recipe_batch_reuses_one_analyzer(library, tmp_path):
    from xrd_tools.notebooks import run_recipe_batch
    entry = _cscl(library, tmp_path)
    plain, _native, _direct_settings = _recipe(tmp_path, entry, {"max_nfev": 8, "width_model": "fixed"}, batch=2)
    seeded, _native, _direct_settings = _recipe(tmp_path, entry, {"max_nfev": 8, "width_model": "fixed",
                                                                  "sequential": True}, batch=2)
    fresh = run_recipe_batch(plain, bundle_root=tmp_path)
    chained = run_recipe_batch(seeded, bundle_root=tmp_path)
    assert len(fresh) == len(chained) == 2
    assert all(outcome.params for outcome in (*fresh, *chained))  # a compact batch keeps result on the last only
    assert fresh[0].params == fresh[1].params
    assert chained[0].params == fresh[0].params and chained[1].params != chained[0].params


def test_k8_a_legacy_fit_bounds_recipe_keeps_its_native_mask_and_both_forms_together_refuse(library, tmp_path):
    from xrd_tools.notebooks import run_recipe
    entry = _cscl(library, tmp_path)
    legacy, _native, _direct = _recipe(tmp_path, entry, {"max_nfev": 3, "fit_bounds": (1.5, 5.5)})
    assert _completed(run_recipe(legacy, bundle_root=tmp_path))
    both, _native, _direct = _recipe(tmp_path, entry, {"max_nfev": 3, "fit_bounds": (1.5, 5.5),
                                                       "fit_range": (1.5, 5.5)})
    with pytest.raises(ValueError, match="fit_range"):
        run_recipe(both, bundle_root=tmp_path)


# ---- K9 the view's resolver --------------------------------------------------------------------------------------

def test_k9_the_resolver_reads_the_library_only_for_ids_it_has_not_resolved(library, tmp_path, monkeypatch):
    from xrd_tools.analysis.phase_references import PhaseLibrary, PhaseSelectionResolver
    first, second = _cscl(library, tmp_path, "first", 4.1), _cscl(library, tmp_path, "second", 4.25)
    lattice = library.add_lattice("cell", a=4.2, space_group="Pm-3m")
    reads = []
    actual = PhaseLibrary.resolve
    monkeypatch.setattr(PhaseLibrary, "resolve",
                        lambda self, identity, _actual=actual: (reads.append(identity), _actual(self, identity))[1])
    resolver = PhaseSelectionResolver(PhaseLibrary(tmp_path / "library"))
    selection = ((second.id, True), ("calibrant:LaB6", False), (first.id, True))
    listed = resolver.listed(selection)
    assert [entry.id for entry, _shown in listed] == [second.id, "calibrant:LaB6", first.id]
    assert [shown for _entry, shown in listed] == [True, False, True]
    assert sorted(reads) == sorted([second.id, "calibrant:LaB6", first.id])
    reads.clear()
    for flags in ((False, True, True), (True, True, False), (True, False, True)):
        again = resolver.listed(tuple((identifier, flag) for (identifier, _old), flag in zip(selection, flags)))
        assert [shown for _entry, shown in again] == list(flags)
    assert reads == [], "an unchanged id set must not read the library"
    grown = resolver.listed(((lattice.id, True),) + selection)
    assert reads == [lattice.id] and grown[0][0].id == lattice.id
    missing = "0" * 64
    reads.clear()
    assert [entry.id for entry, _shown in resolver.listed(((missing, True), (first.id, True)))] == [first.id]
    assert reads == [missing]
    reads.clear()
    resolver.refresh()
    resolver.listed(((missing, True), (first.id, True)))
    assert missing in reads, "an explicit refresh re-reads ids, including a previously missing one"
