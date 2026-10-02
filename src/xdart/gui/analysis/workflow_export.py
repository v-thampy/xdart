"""Shared page-Run export actions; dialogs and rendering use their existing owners."""
from pathlib import Path
import os
import tempfile

from pyqtgraph.Qt import QtCore
from xdart.gui.widgets import file_dialogs


class WorkflowExportActions(QtCore.QObject):
    def __init__(self, parent, *, capture, source_root, status):
        super().__init__(parent)
        self.capture, self.source_root, self.status = capture, source_root, status

    def export_to(self, path):
        from xrd_tools.notebooks.workflow_render import render_workflow_notebook
        path = Path(path)
        format = {".ipynb": "jupyter", ".py": "marimo"}.get(path.suffix.lower())
        if format is None:
            raise ValueError("NOTEBOOK_FORMAT_REQUIRED: choose .ipynb or .py")
        recipe = self.capture()
        document = render_workflow_notebook(recipe, format=format, data_root=str(self.source_root()), name=path.stem)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix="."+path.name, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(document)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.status(f"Run notebook exported: {path}")
        return path

    def choose(self, format=None):
        try:
            extension = ".py" if format == "marimo" else ".ipynb"
            file_filter = ("Jupyter notebooks (*.ipynb);;marimo apps (*.py)" if format is None
                           else f"Notebook (*{extension})")
            picks = file_dialogs.browse(self.parent(), "Export Run notebook", save=True, fallback=str(self.source_root()),
                                        default_name="run"+extension, file_filter=file_filter)
            if picks:
                path = Path(picks[0])
                if format is not None or not path.suffix:
                    path = path.with_suffix(extension)
                self.export_to(path)
        except Exception as error:
            self.status(str(error))
