"""Two fixed, trusted notebook templates with literal configuration; no optional frontend import to render."""
from __future__ import annotations

import json
import textwrap

from .recipe import recipe_to_json
from .inputs import SourceReference


def _code(source):
    return textwrap.dedent(source).strip() + "\n"


def render_notebook(recipe, *, format: str, bundle_root: str = ".", data_root: str | None = None) -> str:
    """Render a clean Jupyter JSON document or native marimo Python app.

    ``BUNDLE_ROOT`` is explicitly editable: set it to the exported notebook/sidecar folder when opening with
    another kernel working directory. ``DATA_ROOT`` is the explicit external-asset root, never guessed from a
    detector frame. The first scientific cell always executes the captured request; Fit/Apply is optional.
    """
    config = _code(f"""
        from pathlib import Path
        from xrd_tools.notebooks import recipe_from_json, run_recipe, run_recipe_batch, load_snapshot
        from xrd_tools.analysis.fit_table import outcomes_table, table_csv
        BUNDLE_ROOT = Path({bundle_root!r}).expanduser()
        DATA_ROOT = Path({(bundle_root if data_root is None else data_root)!r}).expanduser()
        recipe = recipe_from_json({recipe_to_json(recipe)!r})
        TREND_COLUMN = recipe.trend_column
    """)
    scientific = _code("""
        # Ordinary scientific execution: this runs in scripts/headless kernels without any button click.
        outcome = run_recipe(recipe, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)
        results_table = outcomes_table([outcome], contexts=[load_snapshot(recipe.input, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)],
                                      model=dict(recipe.settings).get('model', dict(recipe.settings).get('phase_profile', dict(recipe.settings).get('profile', 'pseudovoigt' if recipe.tool in ('peak', 'phase') else 'gaussian'))), settings=recipe.settings)
        csv_text = table_csv(results_table)
        print(outcome.label, outcome.ok, outcome.message)
    """)
    transport = ("The selected input is reconstructed by its public artifact reader and explicit extraction request. "
                 "DATA_ROOT resolves relative source paths. Published artifacts retain their reader's recorded-path "
                 "admission; moving one can report INPUT_SOURCE_INVALID. A captured lossless sidecar is portable. "
                 if type(recipe.input) is SourceReference else
                 "The selected derived data are in the adjacent lossless NPZ sidecar. ")
    from xrd_tools import __version__
    header = ("# " + recipe.title + "\n\nCurrent admitted input and settings, captured before fitting. " + transport +
              "Edit BUNDLE_ROOT if your kernel "
              "starts in another directory; edit DATA_ROOT for external CIF assets. Changed captured input is "
              "reported as INPUT_CHANGED, not replayed silently.\n\n"
              f"Created with xdart/xrd_tools {__version__}. Open with the installed xdart environment: "
              "Jupyter: `xdart-notebook exported.ipynb`; marimo: `xdart-notebook --marimo exported.py` "
              "(supported marimo>=0.25,<0.26). Plotly is the interactive default. Fit/Apply commits "
              "scientific settings; pan, zoom and Log change presentation. Optional batch fitting is an explicit "
              "call to run_recipe_batch(recipe, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT).\n")
    if recipe.trend_column:
        header += (f"\nSelected trend column: {recipe.trend_column}. TREND_COLUMN preserves this choice for advanced "
                   "callers of the existing read_trend_values API with DATA_ROOT; no trend is loaded automatically.\n")
    if format == "jupyter":
        viewer = _code("""
            # Re-executing this cell disposes the previous controls and callbacks.
            if not globals().get("NOTEBOOK_HEADLESS", False):
                from xrd_tools.gui.widgets import PeakFitViewer
                from IPython.display import display
                if "fit_viewer" in globals():
                    fit_viewer.dispose()
                fit_viewer = PeakFitViewer.from_recipe(recipe, bundle_root=BUNDLE_ROOT,
                                                       data_root=DATA_ROOT, outcome=outcome)
                display(fit_viewer.widget)
        """)
        def cell(kind, source, index):
            entry = {"cell_type": kind, "metadata": {}, "id": f"fit-{index}", "source": source}
            if kind == "code":
                entry.update(execution_count=None, outputs=[])
            return entry
        return json.dumps({"nbformat": 4, "nbformat_minor": 5,
                           "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}},
                           "cells": [cell("markdown", header, 0), cell("code", config, 1),
                                     cell("code", scientific, 2), cell("code", viewer, 3)]},
                          ensure_ascii=False, indent=1) + "\n"
    if format != "marimo":
        raise ValueError("notebook format is jupyter or marimo")
    # Native marimo cells: controls and .value consumers have separate owners. Persist the applied outcome in
    # mo.state; the run_button resetting on another control edit must not revert the visible successful fit.
    cells = [
        ("", "import marimo as mo\nfrom pathlib import Path\nimport json\nfrom dataclasses import replace\n"
         "from xrd_tools.notebooks import recipe_from_json, run_recipe, run_recipe_batch, load_snapshot\n"
         "from xrd_tools.analysis.fit_table import outcomes_table, table_csv\n"
         "from xrd_tools.viz.plotly import plot_analysis_outcome\n"
         "return mo, Path, json, replace, recipe_from_json, run_recipe, run_recipe_batch, load_snapshot, outcomes_table, table_csv, plot_analysis_outcome"),
        ("Path, recipe_from_json", f"BUNDLE_ROOT = Path({bundle_root!r}).expanduser()\n"
         f"DATA_ROOT = Path({(bundle_root if data_root is None else data_root)!r}).expanduser()\n"
         f"recipe = recipe_from_json({recipe_to_json(recipe)!r})\nnotebook_description = {header!r}\n"
         "TREND_COLUMN = recipe.trend_column\nreturn BUNDLE_ROOT, DATA_ROOT, recipe, notebook_description, TREND_COLUMN"),
        ("mo, notebook_description", "mo.md(notebook_description)\nreturn"),
        ("recipe, BUNDLE_ROOT, DATA_ROOT, run_recipe, outcomes_table, table_csv, load_snapshot",
         "outcome = run_recipe(recipe, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)\n"
         "results_table = outcomes_table([outcome], contexts=[load_snapshot(recipe.input, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)], model=dict(recipe.settings).get('model', dict(recipe.settings).get('phase_profile', dict(recipe.settings).get('profile', 'pseudovoigt' if recipe.tool in ('peak', 'phase') else 'gaussian'))), settings=recipe.settings)\n"
         "csv_text = table_csv(results_table)\nreturn outcome, results_table, csv_text"),
        ("mo, recipe, outcome", "get_committed, set_committed = mo.state((recipe, outcome))\n"
         "get_batch, set_batch = mo.state([])\nreturn get_committed, set_committed, get_batch, set_batch"),
        ("mo, recipe",
         "input_selector = mo.ui.dropdown(options={r.label + ' [' + str(i) + ']': i for i, r in enumerate((recipe.input, *recipe.batch))}, value=recipe.input.label + ' [0]', label='Input')\n"
         "overlay_selector = mo.ui.multiselect(options=['fit', 'background', 'residual', 'components'], value=['fit', 'background', 'residual'], label='Overlays')\n"
         "comparison_selector = mo.ui.multiselect(options={r.label + ' [' + str(i) + ']': i for i, r in enumerate((recipe.input, *recipe.batch))}, value=[], max_selections=4, label='Comparison inputs') if recipe.tool in ('peak', 'phase') else None\n"
         "log_display = mo.ui.switch(value=False, label='Log intensity')\n"
         "mo.hstack([w for w in [input_selector, overlay_selector, comparison_selector, log_display] if w is not None])\nreturn input_selector, overlay_selector, comparison_selector, log_display"),
        ("mo, recipe, BUNDLE_ROOT, DATA_ROOT, load_snapshot, json, input_selector",
         "_reference = (recipe.input, *recipe.batch)[input_selector.value]\n"
         "_snapshot = load_snapshot(_reference, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)\n"
         "_axis = _snapshot.axes[0].values\n"
         "_lo, _hi = float(_axis.min()), float(_axis.max())\n"
         "_settings = dict(recipe.settings)\n"
         "fit_range = mo.ui.range_slider(start=_lo, stop=_hi, step=(_hi-_lo)/500, value=list(_settings.get('fit_bounds') or (_lo, _hi)), label='Fit bounds (native coordinates)') if len(_snapshot.axes) == 1 else None\n"
         "region_ranges = {a.name: mo.ui.range_slider(start=float(a.values.min()), stop=float(a.values.max()), step=float(a.values.max()-a.values.min())/500, value=list(dict(_settings.get('region_bounds', ())).get(a.name, (float(a.values.min()), float(a.values.max())))), label=a.name + ' (' + a.unit + ') fit bounds') for a in _snapshot.axes} if recipe.tool.startswith('region') else {}\n"
         "settings_editor = mo.ui.text_area(value=json.dumps(_settings, indent=2), label='Scientific settings (JSON; initials and bounds)', full_width=True)\n"
         "phase_selector = mo.ui.multiselect(options=list(_settings.get('phase_names', ())), value=list(_settings.get('phase_names', ())), label='Enabled phases') if recipe.tool == 'phase' else None\n"
         "_unit_options = {'Native': _snapshot.axes[0].unit}\n"
         "if len(_snapshot.axes) == 1 and _snapshot.wavelength_angstrom is not None and _snapshot.axes[0].unit in ('A^-1', 'q_A^-1', '2th_deg', '2th_rad'):\n"
         "    _unit_options.update({'q (A^-1)': 'q_A^-1', '2theta (deg)': '2th_deg'})\n"
         "axis_unit_selector = mo.ui.dropdown(options=_unit_options, value='Native', label='Display axis')\n"
         "display_min = mo.ui.number(value=_lo, label='Display min (native coordinate)')\n"
         "display_max = mo.ui.number(value=_hi, label='Display max (native coordinate)')\n"
         "slice_axis = mo.ui.dropdown(options=[a.name for a in _snapshot.axes], value=_snapshot.axes[-1].name, label='Displayed slice axis')\n"
         "fit_apply = mo.ui.run_button(label='Fit / Apply settings')\n"
         "fit_batch = mo.ui.run_button(label='Fit batch with applied settings (compact rows)', disabled=not recipe.batch)\n"
         "mo.vstack([w for w in [fit_range, *region_ranges.values(), phase_selector, settings_editor, axis_unit_selector, mo.hstack([display_min, display_max]), slice_axis if len(_snapshot.axes) == 3 else None, fit_apply, fit_batch] if w is not None])\n"
         "return fit_range, region_ranges, phase_selector, settings_editor, axis_unit_selector, display_min, display_max, slice_axis, fit_apply, fit_batch"),
        ("mo, recipe, BUNDLE_ROOT, DATA_ROOT, load_snapshot, slice_axis, input_selector",
         "_reference = (recipe.input, *recipe.batch)[input_selector.value]\n"
         "_snapshot = load_snapshot(_reference, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)\n"
         "_axis = next(a for a in _snapshot.axes if a.name == slice_axis.value)\n"
         "slice_coordinate = mo.ui.number(value=float(_axis.values[len(_axis.values)//2]), label='Slice coordinate (nearest native bin)')\n"
         "mo.vstack([slice_coordinate]) if len(_snapshot.axes) == 3 else None\nreturn slice_coordinate"),
        ("fit_apply, input_selector, fit_range, region_ranges, phase_selector, settings_editor, json, replace, recipe, run_recipe, BUNDLE_ROOT, DATA_ROOT, set_committed, set_batch",
         "if fit_apply.value:\n"
         "    _settings = json.loads(settings_editor.value)\n"
         "    if recipe.tool in ('peak', 'phase'):\n"
         "        _settings['fit_bounds'] = tuple(fit_range.value)\n"
         "    if recipe.tool.startswith('region'):\n"
         "        _settings['region_bounds'] = tuple((name, tuple(control.value)) for name, control in region_ranges.items())\n"
         "    if recipe.tool == 'phase':\n"
         "        _indices = [i for i, n in enumerate(_settings['phase_names']) if n in phase_selector.value]\n"
         "        for _field in ('phase_names', 'cif_paths', 'expected_hashes', 'phase_min_intensities'):\n"
         "            if _field in _settings:\n"
         "                _settings[_field] = tuple(_settings[_field][i] for i in _indices)\n"
         "    _refs = (recipe.input, *recipe.batch)\n"
         "    _applied = replace(recipe, input=_refs[input_selector.value], settings=_settings)\n"
         "    _fit = run_recipe(_applied, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)\n"
         "    set_committed((_applied, _fit))\n    set_batch([])\nreturn"),
        ("fit_batch, get_committed, run_recipe_batch, BUNDLE_ROOT, DATA_ROOT, set_batch",
         "if fit_batch.value:\n"
         "    _applied, _fit = get_committed()\n"
         "    set_batch(run_recipe_batch(_applied, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT))\nreturn"),
        ("mo, recipe, get_committed, get_batch, BUNDLE_ROOT, DATA_ROOT, load_snapshot, plot_analysis_outcome, overlay_selector, comparison_selector, log_display, axis_unit_selector, display_min, display_max, slice_axis, slice_coordinate, outcomes_table, table_csv",
         "_applied, _fit = get_committed()\n"
         "_snapshot = load_snapshot(_applied.input, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)\n"
         "_held = next(a for a in _snapshot.axes if a.name == slice_axis.value)\n"
         "_index = int(abs(_held.values - slice_coordinate.value).argmin())\n"
         "_comparisons = (load_snapshot((recipe.input, *recipe.batch)[i], bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT) for i in (comparison_selector.value[:4] if comparison_selector is not None else ()))\n"
         "fit_figure = plot_analysis_outcome(_snapshot, _fit, overlays=overlay_selector.value, log_y=log_display.value, axis_unit=axis_unit_selector.value, x_range=(display_min.value, display_max.value), peak_model=dict(_applied.settings).get('model', 'pseudovoigt'), plane_axis=slice_axis.value, plane_coordinate=slice_coordinate.value, comparisons=_comparisons)\n"
         "fit_plot = mo.ui.plotly(fit_figure)\n"
         "_batch = get_batch()\n"
         "_contexts = {ref.label: ref for ref in _applied.batch}\n"
         "_contexts[_snapshot.key.result_identity] = _snapshot\n"
         "_table = outcomes_table(_batch or [_fit], contexts=_contexts, model=dict(_applied.settings).get('model', dict(_applied.settings).get('phase_profile', dict(_applied.settings).get('profile', 'pseudovoigt' if _applied.tool in ('peak', 'phase') else 'gaussian'))), settings=_applied.settings)\n"
         "fit_csv = table_csv(_table)\n"
         "mo.vstack([fit_plot, mo.md(_fit.message or ('Fit complete' if _fit.ok else 'Fit failed')), mo.ui.table([{'input': r.input, 'component': r.component, 'parameter': r.parameter, 'value': r.value, 'stderr': r.stderr, 'unit': r.unit, 'status': r.status, 'message': r.message} for r in _table.rows]), mo.download(fit_csv.encode('utf-8'), filename='fit-results.csv', label='Download CSV')])\nreturn fit_figure, fit_plot, fit_csv"),
    ]
    document = "import marimo\n\n__generated_with = '0.25.0'\napp = marimo.App(width='full')\n"
    for parameters, body in cells:
        document += f"\n\n@app.cell\ndef _({parameters}):\n" + textwrap.indent(body, "    ") + "\n"
    return document + '\n\nif __name__ == "__main__":\n    app.run()\n'
