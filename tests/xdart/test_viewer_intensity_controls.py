"""Direct Qt contracts for the standalone-viewer intensity control."""

from importlib import import_module, util

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _control(qapp):
    module = "xdart.gui.widgets.intensity_controls"
    assert util.find_spec(module) is not None, "viewer intensity control is missing"
    control = import_module(module).IntensityControls()
    control.resize(460, 32)
    control.show()
    qapp.processEvents()
    return control


def _click(button, qapp):
    QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()


def test_default_auto_and_sync_are_signal_silent(qapp):
    control = _control(qapp)
    ranges, toggles = [], []
    control.rangeChanged.connect(lambda lo, hi: ranges.append((lo, hi)))
    control.autoToggled.connect(toggles.append)
    assert control.autoscale.isChecked()
    assert not control.slider.isEnabled()
    assert control.findChild(QtWidgets.QLabel).text() == "Intensity"
    control.sync((0, 1000), (100, 900))
    assert control.values() == (100, 900)
    assert control.slider.isEnabled()
    control.sync((0, 1000), (100, 900))
    assert ranges == toggles == []


def test_real_manual_handle_gesture_and_auto_toggle(qapp):
    control = _control(qapp)
    ranges, toggles = [], []
    control.rangeChanged.connect(lambda lo, hi: ranges.append((lo, hi)))
    control.autoToggled.connect(toggles.append)
    control.sync((0, 1000), (100, 900))
    _click(control.autoscale, qapp)
    assert toggles == [False]
    assert control.values() == (100, 900)
    slider = control.slider
    start = QtCore.QPoint(round(slider._x_for_value(100)), slider.height() // 2)
    stop = QtCore.QPoint(round(slider._x_for_value(300)), slider.height() // 2)
    QtTest.QTest.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    QtTest.QTest.mouseMove(slider, stop)
    QtTest.QTest.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=stop)
    qapp.processEvents()
    assert control.values()[0] == pytest.approx(300, abs=5)
    assert control.values()[1] == 900
    assert ranges[-1] == control.values()
    count = len(ranges)
    _click(control.autoscale, qapp)
    assert toggles == [False, True]
    assert len(ranges) == count


@pytest.mark.parametrize("domain", [(0, 2000), (0, 50), (-2000, -1000)])
def test_manual_range_survives_new_frame_domains_including_disjoint(qapp, domain):
    control = _control(qapp)
    control.sync((0, 1000), (100, 500))
    _click(control.autoscale, qapp)
    ranges, toggles = [], []
    control.rangeChanged.connect(lambda lo, hi: ranges.append((lo, hi)))
    control.autoToggled.connect(toggles.append)
    control.sync(domain, domain)
    assert control.values() == (100, 500)
    assert control.slider.domain() == (min(domain[0], 100), max(domain[1], 500))
    assert ranges == toggles == []


def test_empty_domain_retains_manual_limits_and_reset_is_silent(qapp):
    control = _control(qapp)
    control.sync((0, 1000), (100, 500))
    _click(control.autoscale, qapp)
    ranges, toggles = [], []
    control.rangeChanged.connect(lambda lo, hi: ranges.append((lo, hi)))
    control.autoToggled.connect(toggles.append)
    control.sync(None, None)
    assert not control.slider.isEnabled()
    assert control.values() == (100, 500)
    assert not control.autoscale.isChecked()
    control.sync(None, None, reset=True)
    assert control.autoscale.isChecked()
    assert control.values() == (100, 500)
    control.sync((-3, 3), (-1, 1))
    assert control.values() == (-1, 1)
    assert ranges == toggles == []


def test_real_double_click_entry_applies_exact_displayed_limits(qapp):
    control = _control(qapp)
    control.sync((-3, 3), (-1, 1))
    ranges, toggles = [], []
    control.rangeChanged.connect(lambda lo, hi: ranges.append((lo, hi)))
    control.autoToggled.connect(toggles.append)
    QtTest.QTest.mouseDClick(control.slider, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()
    popup = control.findChild(QtWidgets.QFrame, "intensityEntryPopup")
    assert popup is not None and popup.isVisible()
    low = popup.findChild(QtWidgets.QLineEdit, "intensityMinimum")
    high = popup.findChild(QtWidgets.QLineEdit, "intensityMaximum")
    low.selectAll()
    QtTest.QTest.keyClicks(low, "-50")
    high.selectAll()
    QtTest.QTest.keyClicks(high, "5000")
    _click(popup.findChild(QtWidgets.QPushButton, "intensityApply"), qapp)
    assert not popup.isVisible()
    assert not control.autoscale.isChecked()
    assert control.values() == (-50, 5000)
    assert control.slider.domain() == (-50, 5000)
    assert toggles == [False]
    assert ranges[-1] == (-50, 5000)


@pytest.mark.parametrize("low,high", [("bad", "10"), ("nan", "10"), ("20", "10")])
def test_invalid_numeric_entry_keeps_popup_and_previous_range(qapp, low, high):
    control = _control(qapp)
    control.sync((0, 1000), (100, 500))
    QtTest.QTest.mouseDClick(control.slider, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()
    popup = control.findChild(QtWidgets.QFrame, "intensityEntryPopup")
    popup.findChild(QtWidgets.QLineEdit, "intensityMinimum").setText(low)
    edit = popup.findChild(QtWidgets.QLineEdit, "intensityMaximum")
    edit.setText(high)
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()
    assert popup.isVisible()
    assert control.autoscale.isChecked()
    assert control.values() == (100, 500)
