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
    if isinstance(button, QtWidgets.QLabel):
        QtTest.QTest.mouseDClick(button, QtCore.Qt.MouseButton.LeftButton)
    else:
        QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()


def test_percentile_popup_repeated_apply_mode_and_escape(qapp):
    from xdart.gui.widgets.intensity_controls import IntensityControls

    control = IntensityControls(percentiles=(2, 98))
    control.show()
    control.sync((-50, 500), (-20, 400))
    assert control.percentile_values() == (2, 98)
    assert "Absolute display limits: -20 to 400" in control.label.toolTip()
    QtTest.QTest.mouseClick(control.label, QtCore.Qt.MouseButton.LeftButton)
    assert control._entry_popup is None
    QtTest.QTest.mouseDClick(control.label, QtCore.Qt.MouseButton.LeftButton)
    QtTest.QTest.mouseRelease(control.label, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()
    popup = control._entry_popup
    for values in ((10, 90), (20, 80)):
        for editor, value in zip(control._entry_editors(), values):
            editor.setText(str(value))
        _click(popup.findChild(QtWidgets.QPushButton, "intensityApply"), qapp)
        assert popup.isVisible() and control.percentile_values() == values
        assert not control.autoscale.isChecked()
    control.sync((-500, 5000), (-200, 4000))
    assert control.percentile_values() == (20, 80)
    mode = popup.findChild(QtWidgets.QComboBox, "intensityMode")
    mode.setCurrentText("Absolute")
    assert control.percentile_values() is None and control.values() == (-200, 4000)
    mode.setCurrentText("Percentile")
    assert control.percentile_values() == (20, 80)
    QtTest.QTest.keyClick(control._entry_editors()[0], QtCore.Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not popup.isVisible()
    QtTest.QTest.mouseDClick(control.label, QtCore.Qt.MouseButton.LeftButton)
    QtTest.QTest.mouseClick(popup, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(-20, -20))
    qapp.processEvents()
    assert not popup.isVisible()
    control.close()


def test_default_auto_and_sync_are_signal_silent(qapp):
    control = _control(qapp)
    ranges, toggles = [], []
    control.rangeChanged.connect(lambda lo, hi: ranges.append((lo, hi)))
    control.autoToggled.connect(toggles.append)
    assert control.autoscale.isChecked()
    assert not control.slider.isEnabled()
    assert control.label.text() == "Scale"
    assert control.autoscale.text() == "Auto"
    assert 100 <= control.slider.width() <= 113  # ~25% below the former 150px floor, even in a wide host.
    assert control.autoscale.width() <= 56
    assert control.label.toolTip() and control.autoscale.toolTip() and control.slider.toolTip()
    control.sync((0, 1000), (100, 900))
    assert control.values() == (100, 900)
    assert control.slider.isEnabled()
    control.sync((0, 1000), (100, 900))
    assert ranges == toggles == []


def test_unchanged_sync_does_not_repaint_the_slider(qapp):
    control = _control(qapp)
    control.sync((0, 1000), (100, 500))
    qapp.processEvents()

    class PaintCounter(QtCore.QObject):
        paints = 0

        def eventFilter(self, watched, event):
            if event.type() == QtCore.QEvent.Type.Paint:
                self.paints += 1
            return False

    counter = PaintCounter(control)
    control.slider.installEventFilter(counter)
    control.sync((0, 1000), (100, 500))
    qapp.processEvents()
    assert counter.paints == 0
    control.sync((0, 1000), (200, 700))
    qapp.processEvents()
    assert counter.paints > 0


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


def test_real_label_click_entry_applies_exact_displayed_limits(qapp):
    control = _control(qapp)
    control.sync((-3, 3), (-1, 1))
    ranges, toggles = [], []
    control.rangeChanged.connect(lambda lo, hi: ranges.append((lo, hi)))
    control.autoToggled.connect(toggles.append)
    _click(control.findChild(QtWidgets.QLabel), qapp)
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
    assert popup.isVisible()
    assert not control.autoscale.isChecked()
    assert control.values() == (-50, 5000)
    assert control.slider.domain() == (-50, 5000)
    assert toggles == [False]
    assert ranges[-1] == (-50, 5000)


@pytest.mark.parametrize("low,high", [("bad", "10"), ("nan", "10"), ("20", "10")])
def test_invalid_numeric_entry_keeps_popup_and_previous_range(qapp, low, high):
    control = _control(qapp)
    control.sync((0, 1000), (100, 500))
    _click(control.findChild(QtWidgets.QLabel), qapp)
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


@pytest.mark.parametrize("manual", [False, True])
def test_label_entry_opens_without_changing_active_range(qapp, manual):
    control = _control(qapp)
    control.sync((0, 4294967295), (100, 500))
    if manual:
        _click(control.autoscale, qapp)
    before = control.values(), control.autoscale.isChecked()
    ranges, toggles = [], []
    control.rangeChanged.connect(lambda lo, hi: ranges.append((lo, hi)))
    control.autoToggled.connect(toggles.append)
    _click(control.findChild(QtWidgets.QLabel), qapp)
    popup = control.findChild(QtWidgets.QFrame, "intensityEntryPopup")
    assert popup is not None and popup.isVisible()
    assert (control.values(), control.autoscale.isChecked()) == before
    assert ranges == toggles == []
    assert tuple(edit.text() for edit in control._entry_editors()) == ("100", "500")
    popup.close()


def test_autoscale_bar_and_tooltip_use_active_limits_not_detector_ceiling(qapp):
    control = _control(qapp)
    control.sync((0, 4294967295), (100, 500))
    assert control.slider.domain() == control.values() == (100, 500)
    assert "100 to 500" in control.slider.toolTip()
    assert "4294967295" not in control.slider.toolTip()
    _click(control.autoscale, qapp)
    assert control.slider.domain() == (0, 4294967295)
    control.sync((0, 4294967295), (200, 700))
    assert control.values() == (100, 500)
    _click(control.autoscale, qapp)
    assert control.slider.domain() == control.values() == (200, 700)
