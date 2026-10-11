"""Maintainer's frozen P3b lattice-fit decisions; real fitting owners only."""
from dataclasses import replace

import numpy as np

from xrd_tools.analysis.phase import PhaseModel
from xrd_tools.analysis.fitting.phase_fitting import PhaseFitter


def test_lattice_templates_have_equal_heights_free_peaks_and_a_real_cell():
    phase = PhaseModel.from_lattice(
        "positions", a=5.06, b=5.23, c=5.07, space_group="Pca2_1",
    )
    assert len({peak.intensity for peak in phase.peaks}) == 1, (
        "lattice templates must have equal numeric starting heights, not dummy-atom intensities"
    )
    assert phase.structure is None
    q = np.linspace(1., 4., 301)
    fitter = PhaseFitter(q, np.ones_like(q))
    fitter.add_phase(phase, q_range=(1., 4.), min_intensity=100.)
    expected = sum(1. <= peak.q <= 4. for peak in phase.peaks)
    params = fitter.build_parameters(phase_profile="gaussian")
    heights = [name for name in params if name.startswith("p0_pk")]
    assert len(heights) == expected and expected > 1
    assert all(params[name].vary and params[name].value == 1. for name in heights)
    for axis, value in (("a", 5.06), ("b", 5.23), ("c", 5.07)):
        assert params[f"p0_{axis}"].value == value
        assert params[f"p0_{axis}"].vary or params[f"p0_{axis}"].expr
    before = fitter.eval_model(params)
    # Public lattice parameters drive the model despite no atomic structure.
    changed = params.copy()
    independent = next(name for name in ("p0_a", "p0_b", "p0_c") if changed[name].vary)
    changed[independent].value *= 1.001
    assert not np.allclose(fitter.eval_model(changed), before, rtol=1e-8, atol=1e-8)
    result = fitter.fit(params=params, phase_profile="gaussian", max_nfev=1)
    assert result.phase_fractions() == {}
    assert "lattice" in result.phase_fractions_reason.lower()
    assert "fraction" in result.summary().lower()


def _plan(tmp_path, entries, *, floor=5.):
    from xrd_tools.analysis.display_fit_operations import (
        DisplayedPhaseFitPlan, DisplayedTraceInput, DisplayedTraceReceipt,
    )
    q = np.linspace(1., 6., 251)
    trace = DisplayedTraceInput(
        axis=q, intensity=np.ones_like(q), label="lattice", axis_unit="1/angstrom",
        title="lattice", epoch=1,
        receipt=DisplayedTraceReceipt(
            "displayed-trace-v1", 1, "run", "1", tmp_path / "output.nxs", "lattice", 0,
        ),
    )
    return DisplayedPhaseFitPlan(
        trace=trace, cif_paths=(), expected_hashes=(), phase_names=tuple(e.name for e in entries),
        wavelength_angstrom=.729, phase_profile="gaussian", max_nfev=1,
        min_intensity=floor, phase_entries=entries,
    )


def test_displayed_lattice_fit_withholds_fractions_and_cif_keeps_tied_heights(tmp_path):
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from xrd_tools.analysis.display_fit_operations import run_displayed_phase_fit
    library = PhaseLibrary(tmp_path / "library")
    lattice = library.add_lattice("positions", a=4.1, space_group="Pm-3m")
    path = tmp_path / "CsCl.cif"
    path.write_text(str(CifWriter(Structure(
        Lattice.cubic(4.1), ["Cs", "Cl"], [[0, 0, 0], [.5, .5, .5]],
    ))))
    cif = library.add_cif(path, name="structure")
    result = run_displayed_phase_fit(_plan(tmp_path, (lattice, cif)))
    assert result.code == "OK", result.diagnostics
    assert result.phase_names == ("positions", "structure")
    assert result.phase_fractions == () and "lattice" in result.phase_fractions_reason.lower()
    assert any(name.startswith("p0_pk") for name in result.parameter_names)
    assert not any(name.startswith("p1_pk") for name in result.parameter_names)
    alone = run_displayed_phase_fit(_plan(tmp_path, (cif,)))
    assert alone.code == "OK", alone.diagnostics
    assert dict(alone.phase_fractions) == {"structure": 1.}
    assert alone.phase_fractions_reason == ""
    low = run_displayed_phase_fit(_plan(tmp_path, (lattice,), floor=0.))
    high = run_displayed_phase_fit(_plan(tmp_path, (lattice,), floor=100.))
    np.testing.assert_array_equal(low.marker_positions, high.marker_positions)


def test_predicted_parameter_cap_counts_every_lattice_height(tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from xrd_tools.analysis.display_fit_operations import run_displayed_phase_fit
    library = PhaseLibrary(tmp_path / "library")
    lattice = library.add_lattice("many independent heights", a=14.1, b=15.2, c=16.3, space_group=1)
    plan = replace(_plan(tmp_path, (lattice,)), wavelength_angstrom=.251)
    result = run_displayed_phase_fit(plan)
    assert result.code == "FIT_PARAMETER_LIMIT_EXCEEDED", result.diagnostics
    assert result.fit is None and result.parameter_names == ()


def test_displayed_lattice_plan_fingerprint_binds_its_cell(tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from xrd_tools.analysis.display_fit_operations import run_displayed_phase_fit
    library = PhaseLibrary(tmp_path / "library")
    first = library.add_lattice("positions", a=4.1, space_group="Pm-3m")
    second = library.add_lattice("positions", a=4.2, space_group="Pm-3m")
    one = run_displayed_phase_fit(_plan(tmp_path, (first,)))
    two = run_displayed_phase_fit(_plan(tmp_path, (second,)))
    assert one.code == two.code == "OK"
    assert one.plan_fingerprint != two.plan_fingerprint, "the plan identity must bind its lattice cell"
