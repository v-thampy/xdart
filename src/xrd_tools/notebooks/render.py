"""Two fixed, trusted notebook templates with literal configuration; no optional frontend import to render."""
from __future__ import annotations

import json
import textwrap

from .recipe import recipe_to_json


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
    """)
    scientific = _code("""
        # Ordinary scientific execution: this runs in scripts/headless kernels without any button click.
        outcome = run_recipe(recipe, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)
        results_table = outcomes_table([outcome])
        csv_text = table_csv(results_table)
        print(outcome.label, outcome.ok, outcome.message)
    """)
    header = ("# " + recipe.title + "\n\nCurrent admitted input and settings, captured before fitting. "
              "The selected derived data are in the adjacent lossless NPZ sidecar. Edit BUNDLE_ROOT if your kernel "
              "starts in another directory; edit DATA_ROOT for external CIF assets. Changed captured input is "
              "reported as INPUT_CHANGED, not replayed silently.\n\n"
              "Use the same xrd-tools environment as xdart. Jupyter: open this file with JupyterLab. "
              "Marimo 0.25: `marimo edit <export>.py`. Plotly is the interactive default. Fit/Apply commits "
              "scientific settings; pan, zoom and Log change presentation. Optional batch fitting is an explicit "
              "call to run_recipe_batch(recipe, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT).\n")
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
         "from xrd_tools.notebooks import recipe_from_json, run_recipe, load_snapshot\n"
         "from xrd_tools.analysis.fit_table import outcomes_table, table_csv\n"
         "from xrd_tools.viz.plotly import plot_analysis_outcome\n"
         "return mo, Path, json, replace, recipe_from_json, run_recipe, load_snapshot, outcomes_table, table_csv, plot_analysis_outcome"),
        ("Path, recipe_from_json", f"BUNDLE_ROOT = Path({bundle_root!r}).expanduser()\n"
         f"DATA_ROOT = Path({(bundle_root if data_root is None else data_root)!r}).expanduser()\n"
         f"recipe = recipe_from_json({recipe_to_json(recipe)!r})\nnotebook_description = {header!r}\n"
         "return BUNDLE_ROOT, DATA_ROOT, recipe, notebook_description"),
        ("mo, notebook_description", "mo.md(notebook_description)\nreturn"),
        ("recipe, BUNDLE_ROOT, DATA_ROOT, run_recipe, outcomes_table, table_csv",
         "outcome = run_recipe(recipe, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)\n"
         "results_table = outcomes_table([outcome])\ncsv_text = table_csv(results_table)\nreturn outcome, results_table, csv_text"),
        ("mo, recipe, outcome", "get_committed, set_committed = mo.state((recipe, outcome))\nreturn get_committed, set_committed"),
        ("mo, recipe, BUNDLE_ROOT, DATA_ROOT, load_snapshot, json",
         "_snapshot = load_snapshot(recipe.input, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)\n"
         "_axis = _snapshot.axes[0].values\n"
         "_lo, _hi = float(_axis.min()), float(_axis.max())\n"
         "_settings = dict(recipe.settings)\n"
         "input_selector = mo.ui.dropdown(options={r.label + ' [' + str(i) + ']': i for i, r in enumerate((recipe.input, *recipe.batch))}, value=recipe.input.label + ' [0]', label='Input')\n"
         "overlay_selector = mo.ui.multiselect(options=['fit', 'background', 'residual'], value=['fit', 'background', 'residual'], label='Overlays')\n"
         "fit_range = mo.ui.range_slider(start=_lo, stop=_hi, step=(_hi-_lo)/500, value=list(_settings.get('fit_bounds') or (_lo, _hi)), label='Fit bounds (native coordinates)')\n"
         "settings_editor = mo.ui.text_area(value=json.dumps(_settings, indent=2), label='Scientific settings (JSON; initials and bounds)', full_width=True)\n"
         "log_display = mo.ui.switch(value=False, label='Log intensity')\n"
         "fit_apply = mo.ui.run_button(label='Fit / Apply settings')\n"
         "mo.vstack([input_selector, overlay_selector, fit_range, settings_editor, log_display, fit_apply])\n"
         "return input_selector, overlay_selector, fit_range, settings_editor, log_display, fit_apply"),
        ("fit_apply, input_selector, fit_range, settings_editor, json, replace, recipe, run_recipe, BUNDLE_ROOT, DATA_ROOT, set_committed",
         "if fit_apply.value:\n"
         "    _settings = json.loads(settings_editor.value)\n"
         "    if recipe.tool in ('peak', 'phase'):\n"
         "        _settings['fit_bounds'] = tuple(fit_range.value)\n"
         "    _refs = (recipe.input, *recipe.batch)\n"
         "    _applied = replace(recipe, input=_refs[input_selector.value], settings=_settings)\n"
         "    _fit = run_recipe(_applied, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)\n"
         "    set_committed((_applied, _fit))\nreturn"),
        ("mo, get_committed, BUNDLE_ROOT, DATA_ROOT, load_snapshot, plot_analysis_outcome, overlay_selector, log_display, outcomes_table, table_csv",
         "_applied, _fit = get_committed()\n"
         "_snapshot = load_snapshot(_applied.input, bundle_root=BUNDLE_ROOT, data_root=DATA_ROOT)\n"
         "fit_plot = mo.ui.plotly(plot_analysis_outcome(_snapshot, _fit, overlays=overlay_selector.value, log_y=log_display.value))\n"
         "_table = outcomes_table([_fit])\n"
         "fit_csv = table_csv(_table)\n"
         "mo.vstack([fit_plot, mo.md(_fit.message or ('Fit complete' if _fit.ok else 'Fit failed')), mo.ui.table([{'input': r.input, 'component': r.component, 'parameter': r.parameter, 'value': r.value, 'stderr': r.stderr, 'unit': r.unit, 'status': r.status, 'message': r.message} for r in _table.rows]), mo.download(fit_csv.encode('utf-8'), filename='fit-results.csv', label='Download CSV')])\nreturn fit_plot, fit_csv"),
    ]
    document = "import marimo\n\n__generated_with = '0.25.0'\napp = marimo.App(width='full')\n"
    for parameters, body in cells:
        document += f"\n\n@app.cell\ndef _({parameters}):\n" + textwrap.indent(body, "    ") + "\n"
    return document + '\n\nif __name__ == "__main__":\n    app.run()\n'
