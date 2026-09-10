"""The mounted threshold row owns automatic saturation and explicit bands."""
from dataclasses import replace

import numpy as np
import pytest
import tifffile
from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import SourceObservationRequest
from xdart.gui.tabs.scattering.controls_editing import reduce_control_edit, EditNoChange, EditRefusal
from xdart.gui.tabs.scattering.controls_inventory import (
    MASK_SATURATION, THRESHOLD_ENABLED, THRESHOLD_MIN, THRESHOLD_MAX,
)
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.widgets.controls_panel import ControlsPanel, RangeRow, PillRow


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(params=[np.uint8, np.uint16, np.uint32])
def observed_input(request, tmp_path):
    path = tmp_path / "image_0001.tif"
    # The observed maximum is 17, deliberately unrelated to saturation.
    tifffile.imwrite(path, np.full((8, 8), 17, dtype=request.param))
    intent = RunIntent(source_spec=image_series_spec(path))
    observed = FilesystemSourceAdapter().observe(
        SourceObservationRequest(1, 0, intent.source_spec)
    )
    return intent, observed, float(np.iinfo(request.param).max - 1)


def test_single_threshold_toggle_default_manual_and_off(observed_input):
    intent, observed, limit = observed_input
    snapshot = RunIntentStore(intent).snapshot()
    fields = {f.path: f for f in project_controls(snapshot, observed, RunPhase.IDLE).fields}
    assert MASK_SATURATION not in fields
    assert fields[THRESHOLD_ENABLED].value is True
    assert fields[THRESHOLD_MIN].value == 0
    assert fields[THRESHOLD_MAX].value == limit
    assert type(reduce_control_edit(snapshot, THRESHOLD_ENABLED, True)) is EditNoChange

    off = reduce_control_edit(snapshot, THRESHOLD_ENABLED, False)
    assert not off.threshold.apply_threshold and not off.threshold.mask_saturation
    auto = reduce_control_edit(RunIntentStore(off).snapshot(), THRESHOLD_ENABLED, True)
    assert auto.threshold.mask_saturation and not auto.threshold.apply_threshold
    assert auto.threshold.threshold_max is None  # type-derived, never observed max

    manual = reduce_control_edit(snapshot, THRESHOLD_MIN, 1., observation=observed)
    assert manual.threshold.apply_threshold and not manual.threshold.mask_saturation
    assert manual.threshold.threshold_min == 1.
    assert manual.threshold.threshold_max == limit
    assert isinstance(reduce_control_edit(
        snapshot, THRESHOLD_MIN, limit + 1, observation=observed,
    ), EditRefusal)
    assert isinstance(reduce_control_edit(snapshot, THRESHOLD_MAX, -1), EditRefusal)
    intent.threshold.threshold_min, intent.threshold.threshold_max = 50., 100.
    edited = reduce_control_edit(
        RunIntentStore(intent).snapshot(), THRESHOLD_MIN, 1., observation=observed,
    )
    assert (edited.threshold.threshold_min, edited.threshold.threshold_max) == (1., limit)
    off = reduce_control_edit(RunIntentStore(manual).snapshot(), THRESHOLD_ENABLED, False)
    on = reduce_control_edit(RunIntentStore(off).snapshot(), THRESHOLD_ENABLED, True)
    assert on.threshold.apply_threshold and not on.threshold.mask_saturation
    assert (on.threshold.threshold_min, on.threshold.threshold_max) == (1., limit)


def test_explicit_max_replaces_default_and_clearing_restores_native_band(observed_input):
    intent, observed, _ = observed_input
    manual = reduce_control_edit(RunIntentStore(intent).snapshot(), THRESHOLD_MAX, 100.)
    assert manual.threshold.apply_threshold and not manual.threshold.mask_saturation
    assert (manual.threshold.threshold_min, manual.threshold.threshold_max) == (0., 100.)
    auto = reduce_control_edit(RunIntentStore(manual).snapshot(), THRESHOLD_MAX, "")
    assert auto.threshold.mask_saturation and not auto.threshold.apply_threshold
    assert auto.threshold.threshold_min is None and auto.threshold.threshold_max is None
    # A default is shown only for this exact selected input, not a stale observation.
    stale = replace(observed, source=replace(intent.source_spec, uri="/different/input"))
    fields = {f.path: f for f in project_controls(RunIntentStore(auto).snapshot(), stale, RunPhase.IDLE).fields}
    assert fields[THRESHOLD_MAX].value is None


def test_rendered_threshold_has_no_independent_saturation_switch(qapp, observed_input):
    intent, observed, limit = observed_input
    panel = ControlsPanel()
    try:
        panel.reconcile(project_controls(RunIntentStore(intent).snapshot(), observed, RunPhase.IDLE))
        row = next(r for r in panel.findChildren(RangeRow) if tuple(r._low_path) == THRESHOLD_MIN)
        assert row._toggle[1].isChecked()
        assert row._low.isEnabled() and row._high.isEnabled()
        assert float(row._high.text()) == limit
        assert not any(tuple(path) == MASK_SATURATION for r in panel.findChildren(PillRow) for path, _ in r._pills)
        emitted = []
        panel.fieldValueChanged.connect(lambda path, value: emitted.append((tuple(path), value)))
        # Real editor completion and Run harvesting both return displayed
        # defaults. Neither may silently switch the faster automatic policy.
        row._low.editingFinished.emit()
        row._high.editingFinished.emit()
        for path, value in emitted + list(row.current_edits()):
            assert isinstance(reduce_control_edit(
                RunIntentStore(intent).snapshot(), path, value, observation=observed,
            ), EditNoChange)
        emitted.clear()
        row._toggle[1].click()
        assert emitted == [(THRESHOLD_ENABLED, False)]
    finally:
        panel.close()
        panel.deleteLater()
