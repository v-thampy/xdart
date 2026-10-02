"""Fixed page-Run notebook templates: readable configuration, explicit execution, shared result viewers."""
import json
import textwrap

from .workflow import workflow_to_json


def render_workflow_notebook(recipe, *, format, data_root=".", name="workflow"):
    configuration = textwrap.dedent(f"""
        from pathlib import Path
        from xrd_tools.notebooks.workflow import WorkflowRecipe, workflow_from_json, run_workflow, workflow_result_references, read_workflow_result
        recipe = workflow_from_json({workflow_to_json(recipe)!r})
        CONFIGURATION = recipe.configuration
        DATA_ROOT = Path({str(data_root)!r}).expanduser()
        OUTPUT_ROOT = DATA_ROOT / {str(name + '_outputs')!r}
        RUN_PROCESSING = False
    """).strip() + "\n"
    description = (f"# {recipe.mode.upper()} Run replay\n\nThe current page settings and source references are captured below as editable JSON. "
                   "Opening this notebook or Run all does no processing. Jupyter: explicitly set RUN_PROCESSING=True; "
                   "marimo: press Run processing. OUTPUT_ROOT must be a fresh folder below DATA_ROOT, the source Project; "
                   "existing targets are refused. Export/opening creates no output folder. Live replay is refused by the "
                   "finite processing owner. Original source admission, correction, cancellation and publication rules apply. "
                   "After Run, select a published result/frame and compare native patterns using the shared Plotly owners.\n")
    if format == "jupyter":
        sources = [description, configuration,
                   "# Read/edit CONFIGURATION and the two roots before enabling this processing call.\n"
                   "if type(RUN_PROCESSING) is not bool:\n    raise TypeError('RUN_GATE_BOOLEAN_REQUIRED')\n"
                   "outputs = run_workflow(WorkflowRecipe(recipe.mode, CONFIGURATION), data_root=DATA_ROOT, output_root=OUTPUT_ROOT) if RUN_PROCESSING else ()\n"
                   "outputs\n",
                   "if outputs and not globals().get('NOTEBOOK_HEADLESS', False):\n"
                   "    from xrd_tools.notebooks.workflow_viewer import WorkflowResultViewer\n"
                   "    from IPython.display import display\n"
                   "    if 'result_viewer' in globals():\n        result_viewer.dispose()\n"
                   "    result_viewer = WorkflowResultViewer(outputs, recipe.mode)\n    display(result_viewer.widget)\n"]
        cells = []
        for i, source in enumerate(sources):
            cell = {"cell_type": "markdown" if i == 0 else "code", "metadata": {}, "id": f"run-{i}", "source": source}
            if i:
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)
        return json.dumps({"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                           "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}},
                          ensure_ascii=False, indent=1) + "\n"
    if format != "marimo":
        raise ValueError("workflow format is jupyter or marimo")
    cells = [
        ("", "import marimo as mo\nimport json\nfrom pathlib import Path\n"
         "from xrd_tools.notebooks.workflow import WorkflowRecipe, workflow_from_json, run_workflow, workflow_result_references, read_workflow_result\n"
         "from xrd_tools.viz.plotly import plot_analysis_outcome\nreturn mo, json, Path, WorkflowRecipe, workflow_from_json, run_workflow, workflow_result_references, read_workflow_result, plot_analysis_outcome"),
        ("workflow_from_json, Path", f"recipe = workflow_from_json({workflow_to_json(recipe)!r})\n"
         f"DATA_ROOT = Path({str(data_root)!r}).expanduser()\nOUTPUT_ROOT = DATA_ROOT / {str(name + '_outputs')!r}\n"
         f"description = {description!r}\nreturn recipe, DATA_ROOT, OUTPUT_ROOT, description"),
        ("mo, json, recipe, DATA_ROOT, OUTPUT_ROOT, description", "mo.md(description)\n"
         "configuration_editor = mo.ui.text_area(value=json.dumps(recipe.configuration, indent=2), full_width=True, label='Run configuration (JSON)')\n"
         "data_root_editor = mo.ui.text(value=str(DATA_ROOT), label='DATA_ROOT (Project)')\n"
         "output_root_editor = mo.ui.text(value=str(OUTPUT_ROOT), label='OUTPUT_ROOT (fresh Project descendant)')\n"
         "run_processing = mo.ui.run_button(label='Run processing')\n"
         "mo.vstack([configuration_editor, data_root_editor, output_root_editor, run_processing])\n"
         "return configuration_editor, data_root_editor, output_root_editor, run_processing"),
        ("mo", "get_outputs, set_outputs = mo.state(())\nreturn get_outputs, set_outputs"),
        ("run_processing, recipe, configuration_editor, json, WorkflowRecipe, run_workflow, data_root_editor, output_root_editor, set_outputs",
         "if run_processing.value:\n"
         "    _requested = WorkflowRecipe(recipe.mode, json.loads(configuration_editor.value))\n"
         "    set_outputs(run_workflow(_requested, data_root=data_root_editor.value, output_root=output_root_editor.value))\nreturn"),
        ("mo, recipe, get_outputs, workflow_result_references",
         "references = workflow_result_references(get_outputs(), recipe.mode)\n"
         "result_selector = mo.ui.dropdown(options={r.label: i for i, r in enumerate(references)}, value=references[0].label, label='Published result/frame') if references else None\n"
         "comparison_selector = mo.ui.multiselect(options={r.label: i for i, r in enumerate(references)}, value=[], max_selections=4, label='Comparison patterns') if references and recipe.mode != 'rsm' else None\n"
         "log_display = mo.ui.switch(value=False, label='Log intensity')\n"
         "mo.hstack([w for w in [result_selector, comparison_selector, log_display] if w is not None])\nreturn references, result_selector, comparison_selector, log_display"),
        ("mo, references, result_selector, read_workflow_result",
         "_snapshot = read_workflow_result(references[result_selector.value]) if result_selector is not None else None\n"
         "display_bounds = mo.ui.range_slider(start=float(_snapshot.axes[0].values.min()), stop=float(_snapshot.axes[0].values.max()), value=(float(_snapshot.axes[0].values.min()), float(_snapshot.axes[0].values.max())), label='Display range (native coordinate)') if _snapshot is not None else None\n"
         "mo.vstack([display_bounds]) if display_bounds is not None else None\nreturn display_bounds"),
        ("mo, references, result_selector, comparison_selector, display_bounds, log_display, read_workflow_result, plot_analysis_outcome, get_outputs",
         "if result_selector is not None:\n"
         "    _snapshot = read_workflow_result(references[result_selector.value])\n"
         "    _comparisons = (read_workflow_result(references[i]) for i in (comparison_selector.value[:4] if comparison_selector is not None else ()))\n"
         "    _figure = plot_analysis_outcome(_snapshot, log_y=log_display.value, x_range=display_bounds.value, comparisons=_comparisons)\n"
         "    if len(_snapshot.axes) > 1:\n        _figure.update_xaxes(range=list(display_bounds.value))\n"
         "    mo.ui.plotly(_figure)\n"
         "else:\n    mo.md('Processing has not produced a committed result. Enable Run explicitly; terminal values remain in get_outputs().')\nreturn"),
    ]
    document = "import marimo\n\n__generated_with = '0.25.0'\napp = marimo.App(width='full')\n"
    for parameters, body in cells:
        document += f"\n\n@app.cell\ndef _({parameters}):\n" + textwrap.indent(body, "    ") + "\n"
    return document + '\n\nif __name__ == "__main__":\n    app.run()\n'
