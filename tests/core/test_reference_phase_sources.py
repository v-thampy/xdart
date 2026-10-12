"""REFERENCE-PHASES-20261011 frozen headless oracle RS1-RS9 (Claude, October 11).

Built-in calibrant sources, per-phase intensity floors in the shared provider, the overlay job and the
displayed phase fit, and the notebook replay of per-phase floors. Real pymatgen listings and real fits run
throughout; the provider seam is only observed, never replaced. Module-level imports exist on the parent
(037ec3e5), so every new API is reached inside a test body and the parent fails per row, not at collection.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import xrd_tools.analysis.phase_references as provider
from xrd_tools.analysis.phase import HAS_PYMATGEN
from xrd_tools.analysis.phase_references import PhaseEntry, PhaseLibrary, build_phase_references

pytestmark = pytest.mark.skipif(not HAS_PYMATGEN, reason="requires pymatgen")

WAVELENGTH = 1.0
Q_RANGE = (1.0, 6.0)
#: NaCl (311): 2.13 % of the strongest reachable line at 1.0 Å over q 1-6 Å⁻¹ (probe p01); isolated from NaCl's
#: own neighbours by more than 0.16 Å⁻¹.
NACL_WEAK_Q = 3.6949
#: CsCl (222) 4.31 % and (320) 4.15 % under the same conditions.
CSCL_WEAK_Q = (5.2829, 5.4986)
CAVEAT = "radial agreement alone does not identify detector mounting"


def _cif_text(name):
    from pymatgen.core import Lattice, Structure
    if name == "NaCl":
        structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"],
                                              [[0, 0, 0], [.5, .5, .5]])
    elif name == "CsCl":
        structure = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.12), ["Cs", "Cl"],
                                              [[0, 0, 0], [.5, .5, .5]])
    else:
        raise ValueError(name)
    return structure.to(fmt="cif")


def _entry(name):
    raw = _cif_text(name).encode()
    return PhaseEntry(hashlib.sha256(raw).hexdigest(), name, "cif", raw)


def _full(*entries):
    references = build_phase_references(*entries, wavelength_angstrom=WAVELENGTH, q_range=Q_RANGE,
                                        min_intensity=0.)
    assert not references.reason, references.reason
    return {phase.id: phase for phase in references.phases}


def _ring_context():
    """The P3a/P3b detector fixture: a bound Run's own geometry snapshot and native shape."""
    from tests.core.test_predicted_ring_overlays import bound_session
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    _, _, views, _ = bound_session()
    snapshot = views[0].geometry.poni
    ai = detector_calibration_to_integrator(DetectorCalibration(
        PONI.from_dict(snapshot), json.loads(snapshot['detector_config_json'])))
    return tuple(snapshot.items()), (195, 487), ai.wavelength * 1e10


def _families(job, *entries, floor=5., floors=(), unit='q', line_range=(1., 4.), ring=None):
    from xrd_tools.analysis.phase_overlays import PhaseOverlayRequest
    snapshot, shape, wavelength = ring if ring is not None else ((), None, WAVELENGTH)
    request = PhaseOverlayRequest(tuple(entries), floor, unit, line_range, wavelength, snapshot, shape,
                                  floors=floors)
    return {family.id: family for family in job(request, threading.Event(), None)}


def _listing_spy(monkeypatch):
    calls, actual = [], provider.build_phase_references

    def observed(*entries, **kwargs):
        calls.append(tuple(entry.id for entry in entries))
        return actual(*entries, **kwargs)
    monkeypatch.setattr(provider, "build_phase_references", observed)
    return calls


def test_rs1_calibrant_entries_resolve_without_files_and_list_pyfai_positions(tmp_path):
    from pyFAI.calibrant import get_calibrant
    calibrant_entry = getattr(provider, "calibrant_entry", None)
    assert calibrant_entry is not None, "the shared provider must offer built-in calibrant entries"
    entry = calibrant_entry("LaB6")
    assert (entry.id, entry.name, entry.source_kind) == ("calibrant:LaB6", "LaB6", "calibrant")
    root = tmp_path / "empty-library"
    lookup = PhaseLibrary(root).resolve("calibrant:LaB6")
    assert lookup.entry == entry and not lookup.reason
    assert not root.exists() or not any(root.iterdir()), "a built-in source writes no library file"
    expected = sorted(2. * np.pi / d for d in get_calibrant("LaB6").dspacing
                      if Q_RANGE[0] <= 2. * np.pi / d <= Q_RANGE[1])
    assert expected
    for floor in (0., 50., 100.):
        references = build_phase_references(entry, wavelength_angstrom=WAVELENGTH, q_range=Q_RANGE,
                                            min_intensity=floor)
        assert not references.reason, references.reason
        phase, = references.phases
        np.testing.assert_allclose([reflection.q for reflection in phase.reflections], expected, rtol=1e-12, atol=0)
        assert all(reflection.intensity is None for reflection in phase.reflections), "calibrants carry positions only"
    with pytest.raises(ValueError):
        calibrant_entry("Unobtainium")
    assert PhaseLibrary(root).resolve("calibrant:Unobtainium").entry is None


_NO_PYMATGEN = textwrap.dedent("""
    import json, sys
    sys.modules["pymatgen"] = None
    from xrd_tools.analysis import phase_references as provider
    report = {"available": provider.pymatgen_available()}
    references = provider.build_phase_references(provider.calibrant_entry("LaB6"), wavelength_angstrom=1.0,
                                                 q_range=(1.0, 6.0))
    report["lines"] = len(references.phases[0].reflections) if references.phases else 0
    report["reason"] = references.reason
    report["phase_imported"] = "xrd_tools.analysis.phase" in sys.modules
    cif = provider.PhaseEntry("c" * 64, "cif", "cif", b"data_x\\n_cell_length_a 4.0\\n")
    report["cif_reason"] = provider.build_phase_references(cif, wavelength_angstrom=1.0, q_range=(1.0, 6.0)).reason
    print(json.dumps(report))
""")


def test_rs2_calibrant_references_need_no_pymatgen(tmp_path):
    env = dict(os.environ, XDART_PHASE_LIBRARY=str(tmp_path / "phases"))
    done = subprocess.run([sys.executable, "-c", _NO_PYMATGEN], capture_output=True, text=True, env=env,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["available"] is False
    assert report["lines"] > 0 and not report["reason"], report
    assert report["phase_imported"] is False, "a calibrant listing must not import the pymatgen phase backend"
    assert "pymatgen" in report["cif_reason"].lower(), report


def test_rs3_provider_floors_are_per_entry_and_none_keeps_today():
    a, b = _entry("NaCl"), _entry("CsCl")
    full = _full(a, b)
    weak_a = {r.q for r in full[a.id].reflections if 1. <= r.intensity < 5.}
    weak_b = {r.q for r in full[b.id].reflections if 1. <= r.intensity < 5.}
    assert any(abs(q - NACL_WEAK_Q) < 1e-3 for q in weak_a)
    assert all(any(abs(q - expected) < 1e-3 for q in weak_b) for expected in CSCL_WEAK_Q)
    references = build_phase_references(a, b, wavelength_angstrom=WAVELENGTH, q_range=Q_RANGE, min_intensity=5.,
                                        floors=(1., 5.))
    assert not references.reason, references.reason
    listed = {phase.id: {r.q for r in phase.reflections} for phase in references.phases}
    assert weak_a <= listed[a.id], "A's own 1 % floor admits its 1-5 % reflections"
    assert not weak_b & listed[b.id], "B's 5 % floor still drops its 1-5 % reflections"
    today = build_phase_references(a, b, wavelength_angstrom=WAVELENGTH, q_range=Q_RANGE, min_intensity=5.)
    assert build_phase_references(a, b, wavelength_angstrom=WAVELENGTH, q_range=Q_RANGE, min_intensity=5.,
                                  floors=None) == today
    models, _ = provider.prepare_phase_models(a, b, wavelength_angstrom=WAVELENGTH, q_range=Q_RANGE,
                                              min_intensity=5., floors=(1., 5.))
    assert weak_a <= {peak.q for peak in models[0].peaks}
    assert not weak_b & {peak.q for peak in models[1].peaks}


def test_rs4_overlay_job_filters_each_phase_by_its_own_floor():
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob
    a, b = _entry("NaCl"), _entry("CsCl")
    full = _full(a, b)

    def expected(entry, floor):
        return tuple(r.q for r in full[entry.id].reflections if r.intensity >= floor)
    job = PhaseOverlayJob()
    families = _families(job, a, b, floors=((a.id, 1.),), line_range=Q_RANGE)
    assert families[a.id].q_lines == expected(a, 1.)
    assert families[b.id].q_lines == expected(b, 5.)
    families = _families(job, a, b, line_range=Q_RANGE)
    assert families[a.id].q_lines == expected(a, 5.) and families[b.id].q_lines == expected(b, 5.)


def test_rs5_one_phase_change_lists_only_that_phase_and_keeps_family_identity(monkeypatch, tmp_path):
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob
    ring = _ring_context()
    calibrants = tuple(provider.calibrant_entry(name)
                       for name in ("LaB6", "CeO2", "Si", "Al2O3", "Cr2O3", "AgBh"))
    library = PhaseLibrary(tmp_path / "library")
    nacl, lattice = _entry("NaCl"), library.add_lattice("cell", a=4.1, space_group=221)
    entries = (*calibrants, nacl, lattice)
    calls = _listing_spy(monkeypatch)
    job = PhaseOverlayJob()

    def listed():
        return {identifier for call in calls for identifier in call}
    first = _families(job, *entries, ring=ring)
    assert listed() == {entry.id for entry in entries}
    assert all(not first[entry.id].reason for entry in entries), {k: f.reason for k, f in first.items()}
    assert first[calibrants[0].id].q_lines and first[calibrants[0].id].rings
    calls.clear()
    recut = _families(job, *entries, floors=((nacl.id, 1.),), ring=ring)
    assert listed() <= {nacl.id}, calls
    assert all(recut[entry.id] is first[entry.id] for entry in entries if entry is not nacl)
    assert recut[nacl.id].q_lines != first[nacl.id].q_lines
    calls.clear()
    hidden = _families(job, *entries[1:], floors=((nacl.id, 1.),), ring=ring)
    assert not calls and calibrants[0].id not in hidden
    assert all(hidden[entry.id] is recut[entry.id] for entry in entries[1:])
    shown = _families(job, *entries, floors=((nacl.id, 1.),), ring=ring)
    assert not calls, "a phase shown again within the eight-slot bound is served without listing"
    assert all(shown[entry.id] is recut[entry.id] for entry in entries[1:])
    # One line context per slot: a 2θ listing replaces the q one, so returning to q lists again.
    calls.clear()
    _families(job, *entries, floors=((nacl.id, 1.),), ring=ring, unit='2theta', line_range=(10., 50.))
    _families(job, *entries, floors=((nacl.id, 1.),), ring=ring)
    assert listed() == {entry.id for entry in entries}
    # At most eight phase slots: eight other phases evict the first selection.
    others = tuple(library.add_lattice(f"other {index}", a=3.0 + .17 * index, space_group=221) for index in range(8))
    _families(job, *others, ring=ring)
    calls.clear()
    _families(job, *entries, floors=((nacl.id, 1.),), ring=ring)
    assert listed() == {entry.id for entry in entries}


def test_rs6a_aggregate_cif_bytes_refuse_the_whole_request():
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob, PhaseOverlayRequest
    base = _cif_text("NaCl")
    entries = []
    for index in range(5):
        raw = (base + "\n" + "#" * (3_500_000 - len(base)) + f"\n# fixture {index}\n").encode()
        assert len(raw) <= provider.MAX_CIF_BYTES
        entries.append(PhaseEntry(hashlib.sha256(raw).hexdigest(), f"big {index}", "cif", raw))
    assert sum(len(entry.cif_bytes) for entry in entries) > provider.MAX_TOTAL_CIF_BYTES
    families = PhaseOverlayJob()(PhaseOverlayRequest(tuple(entries), 5., 'q', Q_RANGE, WAVELENGTH),
                                 threading.Event(), None)
    assert len(families) == 5
    for family in families:
        assert not family.q_lines and not family.rings
        assert "byte limit" in family.reason, family.reason


def test_rs6b_a_phase_over_its_cap_at_its_own_floor_is_refused_alone(monkeypatch):
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob
    cscl, nacl = _entry("CsCl"), _entry("NaCl")
    full = _full(cscl, nacl)
    # Existing cap pattern (test_phase_overlay_job.py:71): CsCl has 13 lines at 1 % and 11 at 5 %; NaCl 7 at 5 %.
    monkeypatch.setattr(provider, "MAX_REFLECTIONS_PER_PHASE", 12)
    assert sum(r.intensity >= 1. for r in full[cscl.id].reflections) > 12
    assert sum(r.intensity >= 5. for r in full[cscl.id].reflections) <= 12
    families = _families(PhaseOverlayJob(), cscl, nacl, floors=((cscl.id, 1.),), line_range=Q_RANGE)
    assert not families[cscl.id].q_lines and "limit" in families[cscl.id].reason
    assert families[nacl.id].q_lines == tuple(r.q for r in full[nacl.id].reflections if r.intensity >= 5.)
    assert not families[nacl.id].reason
    families = _families(PhaseOverlayJob(), cscl, nacl, line_range=Q_RANGE)
    assert families[cscl.id].q_lines and families[nacl.id].q_lines


def test_rs6c_the_filtered_total_refuses_only_the_overflowing_phases(monkeypatch):
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob, PhaseOverlayRequest
    nacl, cscl = _entry("NaCl"), _entry("CsCl")
    full = _full(nacl, cscl)
    kept = tuple(r.q for r in full[nacl.id].reflections if r.intensity >= 5.)
    monkeypatch.setattr(provider, "MAX_REFLECTIONS_TOTAL", 15)
    assert len(kept) <= 15 < len(kept) + sum(r.intensity >= 5. for r in full[cscl.id].reflections)
    families = {family.id: family for family in PhaseOverlayJob()(
        PhaseOverlayRequest((nacl, cscl), 5., 'q', Q_RANGE, WAVELENGTH), threading.Event(), None)}
    assert families[nacl.id].q_lines == kept and not families[nacl.id].reason
    assert not families[cscl.id].q_lines and "limit" in families[cscl.id].reason


def test_rs6d_cached_admission_tracks_lines_and_rings_separately(monkeypatch):
    """A phase can regain rings while its lines still exceed the shared total."""
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob
    cscl, nacl = _entry("CsCl"), _entry("NaCl")
    ring = _ring_context()
    monkeypatch.setattr(provider, "MAX_REFLECTIONS_TOTAL", 32)
    job = PhaseOverlayJob()
    first = _families(job, cscl, nacl, floors=((cscl.id, 0.),), ring=ring, line_range=(1., 8.))
    assert not first[nacl.id].q_lines and not first[nacl.id].rings
    second = _families(job, cscl, nacl, floors=((cscl.id, 2.),), ring=ring, line_range=(1., 8.))
    fresh = _families(PhaseOverlayJob(), cscl, nacl, floors=((cscl.id, 2.),), ring=ring, line_range=(1., 8.))
    assert not fresh[nacl.id].q_lines and fresh[nacl.id].rings
    assert first[nacl.id].reason == fresh[nacl.id].reason == "cumulative reflection limit exceeded"
    assert second[nacl.id].rings, "cached cap refusal must not hide newly admitted rings"
    assert second[nacl.id].q_lines == fresh[nacl.id].q_lines
    assert len(second[nacl.id].rings) == len(fresh[nacl.id].rings)
    for actual, expected in zip(second[nacl.id].rings, fresh[nacl.id].rings):
        np.testing.assert_array_equal(actual.xy, expected.xy)
    third = _families(job, cscl, nacl, floors=((cscl.id, 0.),), ring=ring, line_range=(1., 8.))
    assert not third[nacl.id].q_lines and not third[nacl.id].rings
    assert third[nacl.id].reason == first[nacl.id].reason


def test_rs7_calibrant_rings_match_the_p3a_kernel():
    import xrd_tools.geometry_figures as geometry
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob
    ring = _ring_context()
    snapshot, shape, _ = ring
    families = _families(PhaseOverlayJob(), provider.calibrant_entry("LaB6"), ring=ring)
    family, = families.values()
    expected = geometry.predict_detector_rings(dict(snapshot), "LaB6", shape=shape, step_px=2.)
    assert family.rings and len(family.rings) == len(expected.curves)
    for actual, reference in zip(family.rings, expected.curves):
        np.testing.assert_allclose(actual.xy, reference.xy, rtol=0., atol=1e-9)
        assert CAVEAT in actual.tooltip, actual.tooltip


def _fit_inputs(tmp_path):
    from tests.core.test_p37_display_fit_operations import _trace
    paths = []
    for name in ("NaCl", "CsCl"):
        path = tmp_path / f"{name}.cif"
        path.write_text(_cif_text(name), encoding="utf-8")
        paths.append(path)
    q = np.linspace(1.0, 6.0, 1001)
    y = np.full(q.shape, 3.)
    for name, scale in (("NaCl", 1.0), ("CsCl", .6)):
        for reflection in _full(_entry(name))[_entry(name).id].reflections:
            y += scale * reflection.intensity * np.exp(-.5 * ((q - reflection.q) / .025) ** 2)
    return tuple(paths), _trace(axis=q, intensity=y)


def _fit_plan(trace, paths, **changes):
    from xrd_tools.analysis.display_fit_operations import DisplayedPhaseFitPlan
    values = dict(trace=trace, cif_paths=paths, expected_hashes=(None,) * len(paths),
                  phase_names=tuple(path.stem for path in paths), wavelength_angstrom=WAVELENGTH,
                  prefit_background="none", phase_profile="gaussian", lattice_pct=.01, min_intensity=5.,
                  max_nfev=2000)
    values.update(changes)
    return DisplayedPhaseFitPlan(**values)


def _bump(axis, component, center):
    near = np.abs(axis - center) < .012
    side = (np.abs(axis - center) > .07) & (np.abs(axis - center) < .1)
    return float(component[near].max() - np.median(component[side]))


def test_rs8_the_displayed_fit_applies_each_phase_floor_at_both_filter_sites(tmp_path):
    from xrd_tools.analysis.display_fit_operations import run_displayed_phase_fit
    paths, trace = _fit_inputs(tmp_path)
    plan = _fit_plan(trace, paths, phase_min_intensities=(1., 5.))
    assert plan.phase_min_intensities == (1., 5.)
    assert _fit_plan(trace, paths, phase_min_intensities=[1., 5.]).phase_min_intensities == (1., 5.)
    result = run_displayed_phase_fit(plan)
    assert result.code == "OK", (result.code, result.diagnostics, result.message)
    axis = np.asarray(trace.axis)
    nacl, cscl = result.phase_components
    assert _bump(axis, nacl, NACL_WEAK_Q) > .01 * float(nacl.max()), "A's 1 % floor keeps its weak component peak"
    for weak in CSCL_WEAK_Q:
        assert _bump(axis, cscl, weak) < .005 * float(cscl.max()), "B's 5 % floor drops its weak peaks"
    for bad in ((True, 5.), (1., float("nan")), (1., 101.), (-1., 5.), (1.,), (1., 5., 5.), ("1", 5.)):
        refused = run_displayed_phase_fit(_fit_plan(trace, paths, phase_min_intensities=bad))
        assert refused.code == "INVALID_PHASE_PLAN", (bad, refused.code)
    absent = run_displayed_phase_fit(_fit_plan(trace, paths))
    empty = run_displayed_phase_fit(_fit_plan(trace, paths, phase_min_intensities=()))
    assert absent.code == empty.code == "OK"
    assert absent.plan_fingerprint == empty.plan_fingerprint
    changed = run_displayed_phase_fit(_fit_plan(trace, paths, phase_min_intensities=(1., 6.)))
    assert changed.plan_fingerprint != result.plan_fingerprint != absent.plan_fingerprint


def test_rs9_jupyter_replay_filters_phase_floors_by_index_only_when_present():
    from xrd_tools.gui.widgets import PhaseFitControls
    calls = []
    controls = PhaseFitControls(["first", "second", "third"], on_fit=calls.append)
    settings = {"cif_paths": ["a.cif", "b.cif", "c.cif"], "phase_names": ["first", "second", "third"],
                "expected_hashes": [None, None, None], "phase_min_intensities": [1., 2., 3.],
                "phase_profile": "gaussian", "prefit_background": "none"}
    try:
        controls.configure_displayed(settings)
        controls._phase_toggles["second"].value = False
        controls._fit_button.click()
        assert calls[-1]["phase_names"] == ("first", "third")
        assert calls[-1]["phase_min_intensities"] == (1., 3.)
        old = {key: value for key, value in settings.items() if key != "phase_min_intensities"}
        controls.configure_displayed(old)
        controls._phase_toggles["second"].value = False
        controls._fit_button.click()
        assert "phase_min_intensities" not in calls[-1]
        assert calls[-1]["phase_names"] == ("first", "third")
    finally:
        controls.dispose()


def test_rs9_marimo_replay_filters_phase_floors_by_index_only_when_present(tmp_path):
    import importlib.util
    from pymatgen.core import Lattice, Structure
    from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
    from xrd_tools.analysis.phase import PhaseModel
    from xrd_tools.core.containers import convert_radial_axis
    from xrd_tools.notebooks import FitNotebookRecipe, capture_snapshot, render_notebook
    names = ("Cu", "Ag", "Al")
    lattices = {"Cu": 3.61, "Ag": 4.09, "Al": 4.05}
    q = np.linspace(2.5, 6., 301)
    y = np.full(q.shape, 3.)
    for name in names:
        structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(lattices[name]), [name], [[0, 0, 0]])
        structure.to(filename=str(tmp_path / f"{name}.cif"), fmt="cif")
        if name == "Ag":
            continue  # the replayed selection deselects Ag, so the trace holds Cu and Al only
        phase = PhaseModel(name, structure)
        phase.calculate_peaks(wavelength=1.)
        for peak in phase.peaks:
            y += peak.intensity * np.exp(-.5 * ((q - peak.q) / .025) ** 2)
    axis = convert_radial_axis(q, "q_A^-1", "2th_deg", 1.)
    native = FitInputSnapshot(FitInputKey("phase replay", "derived", "receipt", 4,
                                          selection=(("meaning", "radial"),)),
                              (FitAxis("2θ", "2th_deg", axis),), y, wavelength_angstrom=1.)
    captured = capture_snapshot(native, "phase.npz")
    (tmp_path / "phase.npz").write_bytes(captured.content)
    base = {"cif_paths": tuple(f"{name}.cif" for name in names), "expected_hashes": (None, None, None),
            "phase_names": names, "phase_profile": "gaussian", "prefit_background": "none", "max_nfev": 500}
    for floors, expected in (((1., 2., 3.), (1., 3.)), (None, None)):
        settings = dict(base) if floors is None else {**base, "phase_min_intensities": floors}
        recipe = FitNotebookRecipe("phase", captured.reference, settings=settings, batch=(captured.reference,))
        path = tmp_path / f"phase_{'new' if floors else 'old'}.py"
        path.write_text(render_notebook(recipe, format="marimo", bundle_root=str(tmp_path)))
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, definitions = module.app.run()
        control_names = ("input_selector", "overlay_selector", "comparison_selector", "log_display", "fit_range",
                         "region_ranges", "phase_selector", "settings_editor", "axis_unit_selector", "display_min",
                         "display_max", "slice_axis", "fit_apply", "fit_batch")
        overrides = {name: definitions[name] for name in control_names}
        overrides["phase_selector"] = SimpleNamespace(value=["Cu", "Al"])
        overrides["fit_apply"] = SimpleNamespace(value=True)
        _, applied = module.app.run(defs=overrides)
        recipe_applied, fit = applied["get_committed"]()
        values = dict(recipe_applied.settings)
        assert tuple(values["phase_names"]) == ("Cu", "Al")
        if expected is None:
            assert "phase_min_intensities" not in values
        else:
            assert tuple(values["phase_min_intensities"]) == expected
        assert fit.ok, fit.message
