"""Finite notebook contract: detached inputs, literal documents, real public fits, compact batch."""
from __future__ import annotations

import json
import ast
import sys

import numpy as np
import pytest

from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot


def snapshot(label="selected 'α'\\member", ndim=1):
    x = np.linspace(1., 4., 101) ** 1.03
    y = 3. + 50. * np.exp(-.5 * ((x - 2.5) / .08) ** 2)
    mask = np.ones(x.size, dtype=bool)
    mask[7] = False
    key = FitInputKey(label, "original/result.h5", "admitted", 23,
                      (("frame", 8), ("cut", "q")), "presentation", (("model", "gaussian"),))
    return FitInputSnapshot(key, (FitAxis("q", "A^-1", x),), y, mask, 1.54)


def recipe_bundle(tmp_path, **settings):
    from xrd_tools.notebooks import FitNotebookRecipe, capture_snapshot
    captured = capture_snapshot(snapshot(), "selected.npz")
    (tmp_path / captured.reference.path).write_bytes(captured.content)
    recipe = FitNotebookRecipe("peak", captured.reference,
                               settings={"selection_mode": "count", "n_peaks": 1,
                                         "model": "gaussian", "background": "constant", **settings})
    return recipe, captured


@pytest.mark.parametrize("owner", ["fit", "run"])
@pytest.mark.parametrize("format", ["jupyter", "marimo"])
def test_export_headers_use_installed_launchers(tmp_path, owner, format):
    from xrd_tools import __version__
    from xrd_tools.notebooks import render_notebook
    from xrd_tools.notebooks.workflow import WorkflowRecipe
    from xrd_tools.notebooks.workflow_render import render_workflow_notebook
    if owner == "fit":
        recipe, _ = recipe_bundle(tmp_path)
        document = render_notebook(recipe, format=format)
        description_name = "notebook_description"
    else:
        document = render_workflow_notebook(WorkflowRecipe("int", {}), format=format)
        description_name = "description"
    if format == "jupyter":
        notebook = json.loads(document)
        header = notebook["cells"][0]["source"]
        assert notebook["cells"][0]["cell_type"] == "markdown"
        if owner == "run":
            assert "RUN_PROCESSING = False" in notebook["cells"][1]["source"]
    else:
        tree = ast.parse(document)
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == description_name
                                  for target in node.targets))
        header = ast.literal_eval(assignment.value)
        assert f"mo.md({description_name})" in document
        if owner == "run":
            assert "if run_processing.value:" in document
            assert "mo.ui.run_button(label='Run processing')" in document
    assert "`xdart-notebook exported.ipynb`" in header
    assert "`xdart-notebook --marimo exported.py`" in header
    assert f"xdart/xrd_tools {__version__}" in header
    assert "marimo>=0.25,<0.26" in header


def test_lossless_selected_input_transport_and_named_failure(tmp_path, monkeypatch):
    from xrd_tools.notebooks import NotebookInputError, load_snapshot
    recipe, captured = recipe_bundle(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    restored = load_snapshot(recipe.input, bundle_root=tmp_path)
    original = snapshot()
    np.testing.assert_array_equal(restored.values, original.values)
    np.testing.assert_array_equal(restored.axes[0].values, original.axes[0].values)
    np.testing.assert_array_equal(restored.mask, original.mask)
    assert restored.key.selection == original.key.selection
    assert restored.key.background == "presentation"
    assert restored.wavelength_angstrom == 1.54
    assert getattr(restored, "sigma", None) is None
    assert restored.key.generation == 0  # the scientific receipt excludes GUI generations
    with np.load(tmp_path / "selected.npz", allow_pickle=False) as data:
        assert all(data[name].dtype.kind != "O" for name in data.files)
    (tmp_path / "selected.npz").write_bytes(b"replaced")
    with pytest.raises(NotebookInputError, match="INPUT_CHANGED"):
        load_snapshot(recipe.input, bundle_root=tmp_path)
    (tmp_path / "selected.npz").unlink()
    with pytest.raises(NotebookInputError, match="INPUT_MISSING"):
        load_snapshot(recipe.input, bundle_root=tmp_path)


@pytest.mark.parametrize("format", ["jupyter", "marimo"])
def test_documents_literal_and_ordinary_fit_executes_without_ui(tmp_path, format):
    from xrd_tools.notebooks import render_notebook, run_recipe
    from xrd_tools.analysis.runner import DisplayedPeakFitAnalyzer
    recipe, captured = recipe_bundle(tmp_path, fit_bounds=(1.8, 3.2))
    document = render_notebook(recipe, format=format, bundle_root=str(tmp_path))
    direct = DisplayedPeakFitAnalyzer(dict(recipe.settings)).analyze(snapshot())
    assert direct.ok, direct.message
    if format == "jupyter":
        import nbformat
        nb = nbformat.reads(document, as_version=4)
        nbformat.validate(nb)
        assert all(not cell.get("outputs") for cell in nb.cells)
        namespace = {"NOTEBOOK_HEADLESS": True}
        for cell in nb.cells:
            if cell.cell_type == "code":
                exec(cell.source, namespace)
        actual = namespace["outcome"]
        restored_recipe = namespace["recipe"]
    else:
        import marimo
        path = tmp_path / "recipe.py"
        path.write_text(document, encoding="utf-8")
        import importlib.util
        spec = importlib.util.spec_from_file_location("fit_recipe_marimo", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        outputs, definitions = module.app.run()
        actual = definitions["outcome"]
        restored_recipe = definitions["recipe"]
        assert "ipywidgets" not in document
    assert actual.ok, actual.message
    np.testing.assert_allclose(actual.overlay.traces["fit"], direct.overlay.traces["fit"], rtol=1e-10, atol=1e-10)
    assert actual.params == pytest.approx(direct.params, rel=1e-10)
    assert restored_recipe.input.label == snapshot().key.result_identity  # escaped source literal round-trips
    assert "50.0," not in document  # numerical arrays live only in the sidecar


def test_marimo_comparison_inputs_use_native_controls_and_semantic_coordinates(tmp_path):
    import importlib.util
    from dataclasses import replace
    import marimo as mo
    from xrd_tools.core.containers import convert_radial_axis
    from xrd_tools.notebooks import capture_snapshot, render_notebook
    recipe, _ = recipe_bundle(tmp_path)
    q = np.linspace(1.2, 4.2, 91)
    patterns = (
        FitInputSnapshot(FitInputKey("batch q pattern", "derived", "receipt q", 0),
                         (FitAxis("q", "A^-1", q),), 4. + np.exp(-((q-2.7)/.2)**2),
                         wavelength_angstrom=1.54),
        FitInputSnapshot(FitInputKey("batch angle pattern", "derived", "receipt angle", 0),
                         (FitAxis("2θ", "2th_deg", convert_radial_axis(q, "q_A^-1", "2th_deg", 1.54)),),
                         7. + 3.*np.exp(-((q-3.)/.3)**2), wavelength_angstrom=1.54),
    )
    references = []
    for i, pattern in enumerate(patterns):
        captured = capture_snapshot(pattern, f"batch-{i}.npz")
        (tmp_path / captured.reference.path).write_bytes(captured.content)
        references.append(captured.reference)
    recipe = replace(recipe, batch=tuple(references))
    path = tmp_path / "comparison.py"
    path.write_text(render_notebook(recipe, format="marimo", bundle_root=str(tmp_path)))
    spec = importlib.util.spec_from_file_location("native_comparison_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, definitions = module.app.run()
    assert "comparison_selector" in definitions
    selector = definitions["comparison_selector"]
    assert isinstance(selector, mo.ui.multiselect)
    chosen = [name for name, index in selector.options.items() if index in (1, 2)]
    # Native controls with selected initial values drive the generated production cells. This is App.run
    # execution evidence, not browser DOM/event evidence.
    controls = ("input_selector", "overlay_selector", "log_display", "fit_range", "region_ranges",
                "phase_selector", "settings_editor", "axis_unit_selector", "display_min", "display_max",
                "slice_axis", "slice_coordinate", "fit_apply", "fit_batch")
    overrides = {name: definitions[name] for name in controls}
    overrides.update({name: definitions[name] for name in ("get_committed", "set_committed", "get_batch", "set_batch")})
    overrides["comparison_selector"] = mo.ui.multiselect(options=selector.options, value=chosen, max_selections=4)
    axis = definitions["axis_unit_selector"]
    overrides["axis_unit_selector"] = mo.ui.dropdown(options=axis.options, value="2theta (deg)")
    _, selected = module.app.run(defs=overrides)
    assert selected["get_committed"]()[1] is definitions["outcome"]
    figure = selected["fit_figure"]
    expected_x = convert_radial_axis(q, "q_A^-1", "2th_deg", 1.54)
    for pattern in patterns:
        trace = next(trace for trace in figure.data if trace.name == pattern.key.result_identity)
        np.testing.assert_allclose(trace.x, expected_x)
        np.testing.assert_array_equal(trace.y, pattern.values)
    assert figure.layout.xaxis2.title.text == "2θ (2th_deg)"


def test_trend_column_is_metadata_and_old_recipes_remain_readable(tmp_path):
    from dataclasses import replace
    from xrd_tools.notebooks import recipe_to_json, recipe_from_json, render_notebook
    from xrd_tools.notebooks.recipe import recipe_analyzer
    recipe, _ = recipe_bundle(tmp_path)
    selected = replace(recipe, trend_column="elapsed seconds")
    restored = recipe_from_json(recipe_to_json(selected))
    assert restored.trend_column == "elapsed seconds" and restored.settings == recipe.settings
    assert recipe_analyzer(restored, data_root=tmp_path).settings == dict(recipe.settings)
    old_json = json.loads(recipe_to_json(selected))
    old_json.pop("trend_column")
    assert recipe_from_json(json.dumps(old_json)).trend_column == ""
    jupyter = json.loads(render_notebook(selected, format="jupyter", bundle_root=str(tmp_path)))
    assert "elapsed seconds" in jupyter["cells"][0]["source"]
    namespace = {}
    exec(jupyter["cells"][1]["source"], namespace)
    assert namespace["TREND_COLUMN"] == "elapsed seconds"
    marimo = render_notebook(selected, format="marimo", bundle_root=str(tmp_path))
    assert "TREND_COLUMN = recipe.trend_column" in marimo and "elapsed seconds" in marimo


def test_safe_recipe_literals_settings_and_region_extension(tmp_path):
    from xrd_tools.notebooks import FitNotebookRecipe, recipe_from_json, recipe_to_json, render_notebook
    recipe, _ = recipe_bundle(tmp_path)
    hostile = FitNotebookRecipe(recipe.tool, recipe.input, settings=dict(recipe.settings),
                                title="'; __import__('os').system('false') #\n\\ unicode α")
    assert recipe_from_json(recipe_to_json(hostile)) == hostile
    for format in ("jupyter", "marimo"):
        document = render_notebook(hostile, format=format, bundle_root=str(tmp_path))
        if format == "jupyter":
            code = "\n".join(cell["source"] for cell in json.loads(document)["cells"] if cell["cell_type"] == "code")
        else:
            code = document
        compile(code, "export", "exec")
    with pytest.raises((TypeError, ValueError)):
        FitNotebookRecipe("peak", recipe.input, settings={"bad": object()})


def test_compact_batch_is_lazy_and_cancellable(tmp_path):
    from xrd_tools.notebooks import run_recipe_batch
    recipe, _ = recipe_bundle(tmp_path)
    loads = []
    def load(reference, *, bundle_root):
        loads.append(reference.path)
        return snapshot()
    outcomes = run_recipe_batch(recipe, [recipe.input] * 3, bundle_root=tmp_path,
                                loader=load, should_cancel=lambda: len(loads) == 2)
    assert loads == [recipe.input.path] * 2
    assert len(outcomes) == 2 and all(o.ok for o in outcomes)
    assert outcomes[0].overlay is None and outcomes[0].result is None
    assert outcomes[-1].overlay is not None


def test_result_table_unknown_uncertainties_and_failed_rows(tmp_path):
    from xrd_tools.analysis.fit_table import outcomes_table, table_csv
    from xrd_tools.analysis.runner import AnalysisOutcome
    from xrd_tools.notebooks import run_recipe
    recipe, _ = recipe_bundle(tmp_path)
    actual = run_recipe(recipe, bundle_root=tmp_path)
    table = outcomes_table([actual, AnalysisOutcome("missing,\"row\"", False, message="INPUT_MISSING")])
    assert len(table.rows) >= 2
    assert any(row.status == "failed" and row.message == "INPUT_MISSING" for row in table.rows)
    assert all(row.stderr is None or np.isfinite(row.stderr) for row in table.rows)
    assert "input,component,parameter,value,stderr,unit,status,message" in table_csv(table)
    assert '"missing,""row"""' in table_csv(table)


def test_optional_uncertainty_round_trip(tmp_path):
    from xrd_tools.notebooks import capture_snapshot, load_snapshot
    import dataclasses
    if "sigma" not in {field.name for field in dataclasses.fields(FitInputSnapshot)}:
        pytest.skip("root owner integrates optional snapshot sigma")
    original = dataclasses.replace(snapshot(), sigma=np.linspace(.4, 2., 101))
    captured = capture_snapshot(original, "weighted.npz")
    (tmp_path / "weighted.npz").write_bytes(captured.content)
    restored = load_snapshot(captured.reference, bundle_root=tmp_path)
    np.testing.assert_array_equal(restored.sigma, original.sigma)


@pytest.mark.parametrize("format", ["jupyter", "marimo"])
def test_real_phase_request_native_two_theta_and_compact_batch(tmp_path, format):
    from pymatgen.core import Lattice, Structure
    from xrd_tools.analysis.phase import PhaseModel
    from xrd_tools.analysis.fit_sources import phase_fit_snapshot
    from xrd_tools.analysis.runner import DisplayedPhaseFitAnalyzer
    from xrd_tools.core.containers import convert_radial_axis
    from xrd_tools.notebooks import FitNotebookRecipe, capture_snapshot, render_notebook, run_recipe_batch
    structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.61), ["Cu"], [[0, 0, 0]])
    cif = tmp_path / "copper.cif"
    structure.to(filename=str(cif), fmt="cif")
    phase = PhaseModel("Cu", structure)
    phase.calculate_peaks(wavelength=1.)
    q = np.linspace(2.5, 6., 301)
    y = np.full(q.shape, 3.)
    for peak in phase.peaks:
        y += peak.intensity * np.exp(-.5*((q - peak.q)/.025)**2)
    axis = convert_radial_axis(q, "q_A^-1", "2th_deg", 1.)
    native = FitInputSnapshot(FitInputKey("phase member 9", "derived", "receipt", 4,
                                         selection=(("meaning", "radial"),)),
                              (FitAxis("2θ", "2th_deg", axis),), y, wavelength_angstrom=1.)
    captured = capture_snapshot(native, "phase.npz")
    (tmp_path / "phase.npz").write_bytes(captured.content)
    settings = {"cif_paths": ("copper.cif",), "expected_hashes": (None,), "phase_names": ("Cu",),
                "phase_profile": "gaussian", "prefit_background": "none", "max_nfev": 1000}
    recipe = FitNotebookRecipe("phase", captured.reference, settings=settings, batch=(captured.reference,))
    direct_settings = {**settings, "cif_paths": (cif,)}
    direct = DisplayedPhaseFitAnalyzer(direct_settings).analyze(phase_fit_snapshot(native))
    assert direct.ok, direct.message
    document = render_notebook(recipe, format=format, bundle_root=str(tmp_path))
    if format == "jupyter":
        namespace = {"NOTEBOOK_HEADLESS": True}
        for cell in json.loads(document)["cells"]:
            if cell["cell_type"] == "code":
                exec(cell["source"], namespace)
        actual = namespace["outcome"]
    else:
        import importlib.util
        path = tmp_path / "phase.py"
        path.write_text(document)
        spec = importlib.util.spec_from_file_location("native_phase", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, definitions = module.app.run()
        actual = definitions["outcome"]
        # Public programmatic cell overrides exercise the production Apply/state cells. This is a native
        # marimo DAG check, not browser-event evidence (the root owner covers the actual frontend).
        from types import SimpleNamespace
        control_names = ("input_selector", "overlay_selector", "comparison_selector", "log_display", "fit_range", "region_ranges",
                         "phase_selector", "settings_editor", "axis_unit_selector", "display_min", "display_max",
                         "slice_axis", "fit_apply", "fit_batch")
        overrides = {name: definitions[name] for name in control_names}
        bounds = (float(axis[30]), float(axis[-30]))
        overrides["fit_range"] = SimpleNamespace(value=bounds)
        overrides["fit_apply"] = SimpleNamespace(value=True)
        _, applied_defs = module.app.run(defs=overrides)
        applied_recipe, applied_fit = applied_defs["get_committed"]()
        assert dict(applied_recipe.settings)["fit_bounds"] == bounds
        assert applied_fit.ok, applied_fit.message
        assert min(applied_fit.overlay.x) >= bounds[0] and max(applied_fit.overlay.x) <= bounds[1]
        # A run button resets to False when another control changes. The committed fit stays in the same
        # public state cell even though the Apply cell now does nothing.
        overrides.update({name: applied_defs[name] for name in ("get_committed", "set_committed", "get_batch", "set_batch")})
        overrides["fit_apply"] = SimpleNamespace(value=False)
        _, reset_defs = module.app.run(defs=overrides)
        assert reset_defs["get_committed"]()[0] == applied_recipe
    assert actual.ok, actual.message
    assert actual.params == pytest.approx(direct.params, rel=1e-9)
    np.testing.assert_allclose(actual.overlay.x, axis, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.overlay.traces["fit"], direct.overlay.traces["fit"], rtol=1e-10, atol=1e-10)
    batch = run_recipe_batch(recipe, bundle_root=tmp_path)
    assert len(batch) == 1 and batch[0].ok
    assert batch[0].params == pytest.approx(actual.params, rel=1e-9)
    from xrd_tools.analysis.fit_table import outcomes_table
    rows = outcomes_table(batch, contexts=[native], model="gaussian", settings=recipe.settings).rows
    assert any(row.component == "Cu" and row.parameter == "fraction" and row.value == 1. for row in rows)
    assert any(row.component == "Cu" and row.parameter == "a" and row.unit == "Å" for row in rows)


@pytest.mark.parametrize("ndim", [2, 3])
def test_region_recipe_uses_public_solver_native_axes_and_mask(tmp_path, ndim):
    from xrd_tools.notebooks import FitNotebookRecipe, capture_snapshot, run_recipe
    coordinates = (np.linspace(-1., 1., 19), np.linspace(-1.2, .8, 17), np.linspace(-.9, 1.1, 15))[:ndim]
    grids = np.meshgrid(*coordinates, indexing="ij")
    center = (.1, -.2, .05)[:ndim]
    fwhm = (.7, .5, .4)[:ndim]
    truth = 2. + 8.*np.exp(-np.log(2)*sum(((g-c)/(w/2))**2 for g,c,w in zip(grids,center,fwhm)))
    mask = np.ones(truth.shape, dtype=bool)
    mask[(0,) * ndim] = False
    native = FitInputSnapshot(FitInputKey("selected box", "derived", "receipt", 1),
                              tuple(FitAxis(f"Q{i}", "A^-1", values) for i, values in enumerate(coordinates)),
                              truth, mask)
    captured = capture_snapshot(native, "region.npz")
    (tmp_path / "region.npz").write_bytes(captured.content)
    recipe = FitNotebookRecipe(f"region{ndim}d", captured.reference,
                               settings={"seeds": (("stable seed", center, fwhm, 8.),), "background": "constant"},
                               orthonormal=ndim == 3)
    outcome = run_recipe(recipe, bundle_root=tmp_path)
    assert outcome.ok, outcome.message
    fitted = outcome.result.payload
    assert fitted.components[0].id == "stable seed"
    np.testing.assert_allclose(fitted.model()[mask], truth[mask], rtol=1e-5, atol=1e-5)
    from dataclasses import replace
    cropped = replace(recipe, settings={**dict(recipe.settings),
                                        "region_bounds": tuple((f"Q{i}", (float(c[1]), float(c[-2])))
                                                               for i,c in enumerate(coordinates))})
    narrowed = run_recipe(cropped, bundle_root=tmp_path)
    assert narrowed.ok, narrowed.message
    assert narrowed.result.payload.shape == tuple(len(c)-2 for c in coordinates)
    from xrd_tools.viz.plotly import plot_analysis_outcome
    figure = plot_analysis_outcome(native, narrowed, log_y=True,
                                   plane_axis="Q0" if ndim == 3 else None, plane_index=2)
    assert len(figure.data) == 3  # selected data/model and signed linear residual
    residual = np.asarray(figure.data[-1].z)
    assert np.nanmax(abs(residual)) < 1e-5
    if ndim == 3:
        plane = plot_analysis_outcome(native, narrowed, plane_axis="Q0", plane_coordinate=float(coordinates[0][2]))
        assert f"Q0={coordinates[0][2]:.6g}" in plane.layout.title.text


def test_source_reference_is_primitive_and_lazy(tmp_path):
    from xrd_tools.notebooks import SourceReference, FitNotebookRecipe, recipe_from_json, recipe_to_json
    reference = SourceReference("project/scan.nxs", "rsm_lineout", "scan 17", "result-receipt", "request-receipt",
                                (("axis", "H"), ("centre", (1., 2., 3.)), ("width", (.1, .2, .3))),
                                (123, 456), "/old/project/scan.nxs")
    recipe = FitNotebookRecipe("peak", reference, batch=(reference,))
    assert recipe_from_json(recipe_to_json(recipe)) == recipe
    with pytest.raises(ValueError, match="SOURCE_DERIVED"):
        SourceReference("scan.nxs", "stitch", "scan", "result", "request", background="presentation")


def test_source_reference_reuses_real_stitch_reader_and_named_receipts(tmp_path):
    from xrd_tools.io.analysis_artifact import (read_analysis_artifact, AnalysisArtifactRequest, AnalysisArtifactKind,
                                               AnalysisArtifactOverwrite, admit_analysis_artifact)
    from xrd_tools.io.output_transaction import OutputTransactionCoordinator
    from xrd_tools.io.stitch_record import StitchedPattern, write_stitch_record
    from xrd_tools.notebooks import SourceReference, load_snapshot, NotebookInputError
    from dataclasses import replace
    import shutil
    request = AnalysisArtifactRequest(tmp_path / "stitch.nexus", AnalysisArtifactKind.STITCH_1D,
                                      AnalysisArtifactOverwrite.CREATE_NEW, *("a" * 64,) * 4,
                                      {"kind": "stitch-1d"})
    x = np.linspace(1., 4., 101)
    pattern = StitchedPattern(x, "q_A^-1", 3. + 50.*np.exp(-.5*((x-2.5)/.08)**2), np.ones(x.size),
                               np.ones(x.size), np.ones(x.size))
    admit_analysis_artifact(request, coordinator=OutputTransactionCoordinator()).publish(
        lambda entry: write_stitch_record(entry, request.target, pattern))
    payload = read_analysis_artifact(request.target)
    from pathlib import Path
    target = Path(request.target)
    stat = target.stat()
    reference = SourceReference(target.name, "stitch", "selected stored profile", payload.result_fingerprint,
                                payload.inspection.request_fingerprint, (("coordinate", "Q"), ("meaning", "radial")),
                                (stat.st_size, stat.st_mtime_ns), str(request.target))
    restored = load_snapshot(reference, bundle_root=tmp_path, data_root=tmp_path)
    np.testing.assert_array_equal(restored.axes[0].values, payload.axes[0][1])
    np.testing.assert_array_equal(restored.values, payload.intensity)
    np.testing.assert_array_equal(restored.sigma, payload.sigma)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copy2(target, relocated / target.name)
    # Published artifact admission intentionally binds its recorded file_name. A moved reference refuses by
    # name instead of bypassing that owner; the bounded captured NPZ remains relocatable.
    with pytest.raises(NotebookInputError, match="INPUT_SOURCE_INVALID"):
        load_snapshot(reference, bundle_root=relocated, data_root=relocated)
    from xrd_tools.notebooks import capture_snapshot
    captured = capture_snapshot(restored, "selected.npz")
    (relocated / "selected.npz").write_bytes(captured.content)
    moved = load_snapshot(captured.reference, bundle_root=relocated)
    np.testing.assert_array_equal(moved.values, restored.values)
    with pytest.raises(NotebookInputError, match="INPUT_CHANGED"):
        load_snapshot(replace(reference, request_fingerprint="new request"), bundle_root=tmp_path)
    with pytest.raises(NotebookInputError, match="INPUT_MISSING"):
        load_snapshot(replace(reference, path="missing.nxs"), bundle_root=tmp_path)


def test_f9_full_compact_metadata_and_uncertainties_are_identical(tmp_path):
    from xrd_tools.notebooks import run_recipe
    from xrd_tools.analysis.fit_table import outcomes_table, table_csv
    from xrd_tools.analysis.runner import AnalysisOutcome
    recipe, _ = recipe_bundle(tmp_path)
    # The public owner estimates nonzero errors for a perturbed synthetic trace.
    from dataclasses import replace
    native = snapshot()
    noisy = replace(native, values=native.values + .05*np.sin(np.arange(native.values.size)))
    full = run_recipe(recipe, bundle_root=tmp_path, loader=lambda reference, **kwargs: noisy)
    compact = AnalysisOutcome(full.label, full.ok, full.params, message=full.message,
                              parameter_values=full.parameter_values, parameter_errors=full.parameter_errors,
                              parameter_units=full.parameter_units)
    kwargs = dict(contexts=[native], model="gaussian", settings=recipe.settings)
    full_table = outcomes_table([full], **kwargs)
    assert outcomes_table([compact], **kwargs) == full_table
    assert any(row.stderr is not None and row.stderr > 0 for row in full_table.rows)
    assert all(row.result_identity == native.key.result_identity and row.path == native.key.path
               and row.selection == native.key.selection and row.model == "gaussian" for row in full_table.rows)
    assert "original/result.h5" in table_csv(full_table)
    before = outcomes_table([], **kwargs)
    assert len(before.rows) == 1 and before.rows[0].status == "not fit"
    assert before.rows[0].settings == recipe.settings and before.rows[0].selection == native.key.selection
