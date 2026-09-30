"""Int and Stitch use identical threshold transitions through their range rows."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from xdart.gui.pages.services import empty_host_services
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import SourceObservationRequest
from xdart.gui.tabs.scattering.controls_editing import (
    EditNoChange, EditRefusal, reduce_control_edit,
)
from xdart.gui.tabs.scattering.controls_inventory import (
    THRESHOLD_ENABLED, THRESHOLD_MAX, THRESHOLD_MIN,
)
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.stitch.page import StitchPage
from xdart.gui.widgets.controls_panel import ControlsPanel, RangeRow
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent, ThresholdIntent
from xrd_tools.sources.selection import image_series_spec
from tests.xdart.pages.test_stitch_page import _Status, _wait, qapp
from tests.xdart.tools.test_stitch_native_values import _native_form


def _edit(row, field, value):
    if field == "enabled":
        button = row._toggle[1]
        if button.isChecked() == value:
            button.toggled.emit(value)  # also exercise a no-change Run harvest
        else:
            button.setChecked(value)
    else:
        editor = row._low if field == "min" else row._high
        editor.setText(str(value))
        editor.setModified(True)
        editor.editingFinished.emit()


def test_restoring_an_automatic_threshold_clears_a_prior_manual_band(qapp, tmp_path):
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    form = _native_form(tmp_path, "multigeometry", scans="1")
    manual = StitchConditioning("manual", None, 800000.)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(replace(form, conditioning=manual))
        assert page.build_form().conditioning == manual
        page.load_form(replace(form, conditioning=StitchConditioning("auto")))
        assert page.build_form().conditioning.mode == "auto"
    finally:
        page.shutdown()


@pytest.mark.parametrize("dtype", ("int32", "uint32", "uint16", "float32"))
@pytest.mark.parametrize("initial", ("auto", "touch-min", "touch-max", "touch-enabled", "disabled-touch"))
def test_threshold_edit_sequences_match_int(qapp, tmp_path, dtype, initial):
    form = replace(_native_form(tmp_path, "multigeometry", scans="1"), raw_dtype=dtype)
    for path in Path(form.image_dir).glob("*.raw"):
        np.fromfile(path, dtype=np.int32).astype(dtype).tofile(path)
    image = tmp_path / "observed_0001.tif"
    tifffile.imwrite(image, np.full((8, 8), 17, dtype=dtype))
    intent = RunIntent(source_spec=image_series_spec(image))
    observation = FilesystemSourceAdapter().observe(
        SourceObservationRequest(1, 0, intent.source_spec))
    limit = None if dtype == "float32" else float(np.iinfo(dtype).max - 1)
    assert observation.default_threshold_max == limit
    state = {"intent": intent, "result": None}
    panel = ControlsPanel()
    page = StitchPage(empty_host_services(_Status()))
    paths = {"min": THRESHOLD_MIN, "max": THRESHOLD_MAX, "enabled": THRESHOLD_ENABLED}
    try:
        page.load_form(form)
        page.begin_preview()
        _wait(page, qapp)
        assert page._owner.prepared is not None, page.run_bar.readinessLabel.full_text()
        assert page._auto_threshold_max == limit
        if initial != "auto":
            # Legacy/manual plans can have an implicit lower default. A touch
            # must materialize it, even when the displayed value is unchanged.
            state["intent"].threshold = ThresholdIntent(True, None, 100., False)
            page.load_form(replace(form, conditioning=StitchConditioning("manual", None, 100.)))

        def project():
            return project_controls(RunIntentStore(state["intent"]).snapshot(), observation, RunPhase.IDLE)

        def edit_int(path, value):
            result = reduce_control_edit(
                RunIntentStore(state["intent"]).snapshot(), tuple(path), value,
                observation=observation)
            state["result"] = result
            if isinstance(result, RunIntent):
                state["intent"] = result
            panel.reconcile(project())

        panel.reconcile(project())
        panel.fieldValueChanged.connect(edit_int)
        if initial == "auto":
            sequences = [
                ("min", "0", "same"),
                ("max", "" if limit is None else str(limit), "same"),
                ("min", "4", "refuse" if limit is None else "change"),
                ("max", "100", "change"),
                ("min", "101", "refuse"),
                ("min", "", "change"),
                ("max", "", "change"),
                ("max", "-1", "refuse"),
                ("max", "100", "change"),
                ("enabled", False, "change"),
                ("enabled", True, "change"),
                ("max", "100", "same"),
                ("min", "0", "same"),
                ("min", "nan", "refuse"),
                ("max", "inf", "refuse"),
            ]
        elif initial == "disabled-touch":
            sequences = [("enabled", False, "change"), ("min", "0", "change")]
        else:
            field = initial.removeprefix("touch-")
            sequences = [(field, {"min": "0", "max": "100", "enabled": True}[field], "change")]

        for field, value, expected in sequences:
            row = next(row for row in panel.findChildren(RangeRow)
                       if tuple(row._low_path) == THRESHOLD_MIN)
            before_revision = page._revision
            _edit(row, field, value)
            _edit(page.threshold_range, field, value)
            result = state["result"]
            actual = "refuse" if isinstance(result, EditRefusal) else "same" if isinstance(result, EditNoChange) else "change"
            assert actual == expected, (initial, field, value)
            assert page._revision - before_revision == int(expected == "change"), (initial, field, value)
            if expected == "refuse":
                assert page.run_bar.readinessLabel.full_text() == result.reason
            threshold = state["intent"].threshold
            mode = "manual" if threshold.apply_threshold else "auto" if threshold.mask_saturation else "off"
            assert page._threshold_mode == mode
            int_fields = {item.path: item.value for item in project().fields}
            for name, key in (("min", "threshold_min"), ("max", "threshold"), ("enabled", "threshold_enabled")):
                assert page._value(key) == int_fields[paths[name]], (initial, field, value, key)
            actual_policy = page.build_form().conditioning
            assert actual_policy == StitchConditioning(mode,
                threshold.threshold_min if mode == "manual" else None,
                threshold.threshold_max if mode == "manual" else None)
    finally:
        page.shutdown()
        panel.close()
        panel.deleteLater()
