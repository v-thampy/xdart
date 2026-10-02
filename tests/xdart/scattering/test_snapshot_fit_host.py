"""Shared host: real fit worker, before-fit export and held Int capabilities."""
from dataclasses import replace
import time
import numpy as np
from pyqtgraph.Qt import QtWidgets
from xrd_tools.analysis.fit_snapshot import FitInputKey, FitInputSnapshot, FitAxis
from xdart.gui.analysis.snapshot_fit_host import SnapshotFitHost, FitTarget


def sample(identity='scan', generation=1):
    x = np.linspace(1., 4., 301)
    y = 20 + 500 * np.exp(-.5 * ((x - 2.5) / .03)**2)
    return FitInputSnapshot(FitInputKey(identity, '/data/' + identity + '.nexus', 'admitted', generation),
                            (FitAxis('q', 'q_A^-1', x),), y, wavelength_angstrom=1.)


def drain(app, host):
    deadline = time.monotonic() + 10
    while host.computing and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert not host.computing


def test_before_fit_export_and_single_batch_use_same_real_owner(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    state = [sample()]
    host = SnapshotFitHost('peak', current=lambda: state[0], scope=lambda: state[0].key,
                           batch=lambda: tuple(FitTarget(str(i), lambda i=i: sample(str(i))) for i in range(3)))
    host.dialog.show()
    host.reload()
    drain(app, host)
    assert host.dialog.batch_btn.isEnabled() and not host.dialog.live_check.isEnabled()
    # Current settings exported even before any fit has succeeded.
    host.export_to(tmp_path / 'fit.ipynb')
    assert (tmp_path / 'fit.ipynb').is_file()
    host.dialog.fit_btn.click()
    drain(app, host)
    assert host.outcomes, host.dialog.status.text()
    assert host.outcomes[0].ok and abs(host.outcomes[0].params['center_0'] - 2.5) < 1e-7
    host.dialog.batch_btn.click()
    drain(app, host)
    assert len(host.outcomes) == 3 and all(o.ok for o in host.outcomes)
    assert sum(o.result is not None for o in host.outcomes) == 1
    host.export_csv(tmp_path / 'fit.csv')
    assert 'result' in (tmp_path / 'fit.csv').read_text()
    host.dialog.close()
    assert host.snapshot is None and host.dialog._x is None and host.dialog._y is None
    assert not host.dialog.fit_view.projection.upper.series
    while host.shutdown().status.value == 'pending':
        app.processEvents()


def test_replaced_or_changed_scope_cannot_adopt_completed_fit():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    state = [sample()]
    host = SnapshotFitHost('peak', current=lambda: state[0], scope=lambda: state[0].key)
    host.reload()
    drain(app, host)
    host.dialog.fit_btn.click()
    state[0] = replace(state[0], key=replace(state[0].key, generation=2))
    drain(app, host)
    assert not host.outcomes
    host.dialog.close()
    host.shutdown()
