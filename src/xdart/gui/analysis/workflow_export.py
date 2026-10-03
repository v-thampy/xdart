"""Shared page-Run export actions; dialogs and rendering use their existing owners."""
from pathlib import Path

from pyqtgraph.Qt import QtCore
from xdart.gui.widgets import file_dialogs


class WorkflowExportActions(QtCore.QObject):
    def __init__(self, parent, *, capture, source_root, status):
        super().__init__(parent)
        self.capture, self.source_root, self.status = capture, source_root, status

    def mount_help_menu(self, menu):
        """Mount the same Run export action beside a page's existing Help actions."""
        action = menu.addAction("Export Run notebook…")
        action.setToolTip("Export current processing inputs for Jupyter or marimo; Run writes "
                          "to a fresh output folder inside DATA_ROOT.")
        action.triggered.connect(lambda _checked=False: self.choose())
        return action

    def export_to(self, path, *, recipe=None):
        from xrd_tools.notebooks.workflow_render import render_workflow_notebook
        path = Path(path)
        format = {".ipynb": "jupyter", ".py": "marimo"}.get(path.suffix.lower())
        if format is None:
            raise ValueError("NOTEBOOK_FORMAT_REQUIRED: choose .ipynb or .py")
        recipe = self.capture() if recipe is None else recipe
        document = render_workflow_notebook(recipe, format=format, data_root=str(self.source_root()), name=path.stem)
        from xrd_tools.io.export import write_exports_atomically
        write_exports_atomically(((path, document),))
        self.status(f"Run notebook exported: {path}")
        return path

    def choose(self, format=None):
        try:
            recipe = self.capture()
            if recipe.mode == 'int':
                source = recipe.configuration['intent'].get('source_spec') or {}
                result = source.get('uri') or source.get('root') or Path(self.source_root()).name
            else:
                result = recipe.configuration['forms'][0]['fields']['output_path']['$path']
            extension = ".py" if format == "marimo" else ".ipynb"
            file_filter = ("Jupyter notebooks (*.ipynb);;marimo apps (*.py)" if format is None
                           else f"Notebook (*{extension})")
            picks = file_dialogs.browse(self.parent(), "Export Run notebook", save=True, fallback=str(self.source_root()),
                                        default_name=file_dialogs.export_name(result, 'run', extension),
                                        file_filter=file_filter)
            if picks:
                path = Path(picks[0])
                if format is not None or not path.suffix:
                    path = path.with_suffix(extension)
                self.export_to(path, recipe=recipe)
        except Exception as error:
            self.status(str(error))
