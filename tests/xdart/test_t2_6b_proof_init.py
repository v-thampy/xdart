"""T-2.6b (Correction C item 4, §15.12-C.4): GI-motor proof initializes FALSE.

An UNPROVED empty motor emission (no targeted inspection has run) is UNKNOWN, so
freeze preserves the operator's explicit / session-restored θ-motor instead of
resolving it to Manual; only a TARGETED inspection that proved no motors is
KNOWN_EMPTY.  Red at the T-2.6 tip (fresh-wrangler proof defaulted True); green
once the default is inverted.  §12 stays 9/9 because the amended §12 tripwire now
sets ``_gi_motor_knowledge_proved = True`` explicitly before its known-empty emit.
"""
from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    value = staticWidget()
    value._refresh_controls_v2_profile_now()
    try:
        yield value
    finally:
        value.close()
        value.deleteLater()
        qapp.processEvents()


def test_unproved_empty_emit_is_unknown(widget, qapp):
    """A bare empty emission with no targeted inspection is UNKNOWN (stack #7)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import GIMotorObservation

    wrangler = widget.wrangler
    wrangler.motors = []
    wrangler.set_gi_motor_options()          # no discovery / inspection ran
    qapp.processEvents()

    observation = widget._controls_v2_capture_gi_motor_observation()
    assert observation.state == GIMotorObservation.UNKNOWN
    assert observation.choices_for_freeze() is None


def test_targeted_empty_inspection_is_known_empty(widget, qapp):
    """A targeted inspection that PROVED no motors is KNOWN_EMPTY (the partner
    half — resolves to Manual)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import GIMotorObservation

    wrangler = widget.wrangler
    wrangler.motors = []
    wrangler._gi_motor_knowledge_proved = True   # a targeted inspection ran
    wrangler.set_gi_motor_options()
    qapp.processEvents()

    observation = widget._controls_v2_capture_gi_motor_observation()
    assert observation.state == GIMotorObservation.KNOWN_EMPTY
    assert observation.choices_for_freeze() == ()
