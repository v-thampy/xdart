"""Concrete terminal-review regressions, through real phase plans and notebook frontends."""
import importlib.util

import numpy as np
import pytest

from tests.core.test_phase_analysis_options import (
    WL, _cscl, _flat, _plan, _recipe, _run, _trace, library,
)


@pytest.mark.parametrize("field,value", [
    ("width_limits", (1.25, None)), ("width_limits", (None, 1e-5)),
    ("initial_topology", (("phase", "cif", None),)),
    ("initial_topology", (("phase", "cif", ((1, 2),)),)),
    ("initial_topology", (("phase", "cif", ((1, 0, float('nan')),)),)),
    ("initial_values", (("x" * 257, 1.),)),
    ("amorphous_center", float("nan")), ("amorphous_sigma", -1.),
    ("amorphous_sigma", float("nan")),
])
def test_malformed_options_are_named_refusals(library, tmp_path, field, value):
    entry = _cscl(library, tmp_path)
    options = {"width_model": "fixed", field: value}
    if field.startswith("initial_"):
        options.setdefault("initial_values", (("p0_scale", 1.),))
        options.setdefault("initial_topology", ((entry.id, "cif", ((1, 0, 0),)),))
    result = _run(_plan(_trace(*_flat()), (entry,), **options))
    assert result.code == "INVALID_PHASE_PLAN", (result.code, result.diagnostics)
    assert field in " ".join(result.diagnostics)


def test_duplicate_substrate_coordinates_are_refused(library, tmp_path):
    entry = _cscl(library, tmp_path)
    template = _trace([1., 1., 6.], [1., 100., 2.], label="duplicate substrate")
    result = _run(_plan(_trace(*_flat()), (entry,), fit_background="template", background_template=template))
    assert result.code == "INVALID_PHASE_PLAN"
    assert "background_template" in " ".join(result.diagnostics)


@pytest.mark.parametrize("frontend", ["jupyter", "marimo"])
@pytest.mark.parametrize("bounds", [None, (1.5, 5.5)])
def test_interactive_phase_range_keeps_native_units_and_full_trace(library, tmp_path, frontend, bounds):
    from dataclasses import replace
    from xrd_tools.analysis.fit_snapshot import FitAxis
    from xrd_tools.core.containers import convert_radial_axis
    from xrd_tools.notebooks import capture_snapshot, render_notebook
    entry = _cscl(library, tmp_path)
    recipe, native, _ = _recipe(tmp_path, entry, {
        "fit_range": bounds, "fit_range_unit": "q_A^-1", "max_nfev": 5,
        "fit_background": "linear", "amorphous_peak": "gaussian", "amorphous_center": 2.2,
    })
    native = replace(native, axes=(FitAxis("2θ", "2th_deg", convert_radial_axis(
        native.axes[0].values, "q_A^-1", "2th_deg", WL)),))
    captured = capture_snapshot(native, "native.npz")
    (tmp_path / "native.npz").write_bytes(captured.content)
    recipe = replace(recipe, input=captured.reference, batch=())
    expected = native.axes[0].values[[0, -1]] if bounds is None else convert_radial_axis(
        np.asarray(bounds), "q_A^-1", "2th_deg", WL)
    chosen = tuple(native.axes[0].values[[40, -40]])
    if frontend == "jupyter":
        # Execute the rendered viewer cell too: the original K8 deliberately used NOTEBOOK_HEADLESS.
        import json
        namespace = {}
        try:
            for cell in json.loads(render_notebook(recipe, format=frontend, bundle_root=str(tmp_path)))["cells"]:
                if cell["cell_type"] == "code":
                    exec(cell["source"], namespace)
            viewer = namespace["fit_viewer"]
            np.testing.assert_allclose(viewer.ranges.fit_bounds.value, expected)
            viewer.ranges.fit_bounds.value = chosen
            viewer.controls._fit_button.click()
            settings, outcome = viewer.current_settings(), viewer.outcome
            viewer._overlays.value = ("components",)
            names = {trace.name for trace in viewer.figure_widget.data}
            assert {"fit_background", "amorphous"} <= names
        finally:
            if "fit_viewer" in namespace:
                namespace["fit_viewer"].dispose()
    else:
        path = tmp_path / "phase_review.py"
        path.write_text(render_notebook(recipe, format=frontend, bundle_root=str(tmp_path)))
        spec = importlib.util.spec_from_file_location("phase_review", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, definitions = module.app.run()
        np.testing.assert_allclose(definitions["fit_range"].value, expected)
        # Drive the real controls' frontend value-update interface, then the generated Apply/state cells.
        definitions["fit_range"]._update(list(chosen))
        definitions["fit_apply"]._update(True)
        definitions["overlay_selector"]._update(["components"])
        controls = ("input_selector", "overlay_selector", "comparison_selector", "log_display", "fit_range",
                    "region_ranges", "phase_selector", "settings_editor", "axis_unit_selector", "display_min",
                    "display_max", "slice_axis", "fit_apply", "fit_batch")
        _, applied = module.app.run(defs={name: definitions[name] for name in controls})
        applied_recipe, outcome = applied["get_committed"]()
        settings = dict(applied_recipe.settings)
        assert {"fit_background", "amorphous"} <= {trace.name for trace in applied["fit_figure"].data}
    assert "fit_bounds" not in settings
    assert settings["fit_range_unit"] == "2th_deg"
    np.testing.assert_allclose(settings["fit_range"], chosen)
    assert outcome.result is not None and outcome.result.code == "OK", outcome.message
    assert outcome.result.fit.size == native.axes[0].values.size
    np.testing.assert_allclose(outcome.overlay.x, native.axes[0].values)
