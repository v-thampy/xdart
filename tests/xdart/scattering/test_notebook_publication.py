"""Real export/readback: notebook formats retain independent inputs and normal file modes."""
import json
import stat

import numpy as np
import pytest

from tests.xdart.scattering.test_snapshot_fit_host import sample


def test_jupyter_input_and_batch_survive_same_stem_marimo_export(tmp_path):
    from xdart.gui.analysis.fit_export import write_fit_notebook
    from xrd_tools.notebooks import load_snapshot
    first, second = sample('first'), sample('second')
    # Different values and masks make an accidental sidecar replacement observable.
    from dataclasses import replace
    second = replace(second, values=second.values * 2)
    path = tmp_path / 'calibrant_peak.ipynb'
    write_fit_notebook(path, first, 'peak', {}, batch_inputs=(first,))
    config = next(cell['source'] for cell in json.loads(path.read_text())['cells']
                  if cell['cell_type'] == 'code' and 'recipe = recipe_from_json' in cell['source'])
    namespace = {}
    exec(config, namespace)
    recipe = namespace['recipe']
    write_fit_notebook(path.with_suffix('.py'), second, 'peak', {}, batch_inputs=(second,))
    for reference in (recipe.input, *recipe.batch):
        restored = load_snapshot(reference, bundle_root=tmp_path)
        np.testing.assert_array_equal(restored.values, first.values)
        np.testing.assert_array_equal(restored.axes[0].values, first.axes[0].values)


@pytest.mark.skipif(__import__('os').name != 'posix', reason='POSIX file permission contract')
@pytest.mark.parametrize('owner', ('fit', 'run', 'profile'))
def test_exports_use_normal_new_file_mode_and_preserve_replacement_mode(tmp_path, owner):
    from pyqtgraph.Qt import QtWidgets
    from xdart.gui.analysis.fit_export import write_fit_notebook
    from xdart.gui.analysis.workflow_export import WorkflowExportActions
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.notebooks.workflow import WorkflowRecipe
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    action = WorkflowExportActions(parent, capture=lambda: WorkflowRecipe('int', {}),
                                   source_root=lambda: tmp_path, status=lambda text: None)
    ordinary = tmp_path / 'ordinary'
    ordinary.write_bytes(b'normal permissions under the current umask')
    expected_new = stat.S_IMODE(ordinary.stat().st_mode)
    path = tmp_path / ('fit.ipynb' if owner == 'fit' else 'run.ipynb' if owner == 'run' else 'profile.json')
    def publish():
        if owner == 'fit':
            write_fit_notebook(path, sample(), 'peak', {}, batch_inputs=(sample(),))
        elif owner == 'run':
            action.export_to(path)
        else:
            ScatteringWorkspace._write_profile_atomically(path, '{"value": 2}')
    try:
        publish()
        assert stat.S_IMODE(path.stat().st_mode) == expected_new
        for sidecar in tmp_path.glob('*.npz'):
            assert stat.S_IMODE(sidecar.stat().st_mode) == expected_new
            sidecar.chmod(0o640)
        path.chmod(0o640)
        before = path.read_bytes()
        path.write_bytes(b'old export')
        publish()
        assert path.read_bytes() == before
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
        assert all(stat.S_IMODE(p.stat().st_mode) == 0o640 for p in tmp_path.glob('*.npz'))
    finally:
        parent.deleteLater()
        app.processEvents()
