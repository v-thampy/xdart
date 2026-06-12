"""Status-label width guard (window blow-out fix).

A plain QLabel's minimum width is its full text width, so a long status
message (e.g. the live-GI clip advisory imageThread emits on Start, ~180
chars) forced the WHOLE window to expand horizontally off-screen.
``wranglerWidget._guard_status_label`` gives the label an Ignored horizontal
size policy (it can never drive layout width) and ``_set_status_text`` elides
overlong text into the label, keeping the full message in the tooltip.
"""
import threading

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets
from pyqtgraph.Qt import QtCore

from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerWidget

LONG_MSG = ('Live GI: output range frozen from the first frame — if this '
            'scan sweeps a range of incidence angles, later frames may be '
            'clipped. Reprocess in batch for the full range.')


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _widget_with_label(qapp):
    w = wranglerWidget('f', threading.Condition())
    layout = QtWidgets.QVBoxLayout(w)
    w.statusLabel = QtWidgets.QLabel()
    layout.addWidget(w.statusLabel)
    return w


def test_guard_makes_label_width_ignored(qapp):
    w = _widget_with_label(qapp)
    w._guard_status_label()
    assert (w.statusLabel.sizePolicy().horizontalPolicy()
            == QtWidgets.QSizePolicy.Policy.Ignored)
    # With an Ignored policy the layout disregards the label's size hint,
    # so even a raw setText of a long message cannot widen the window.
    w.statusLabel.setText(LONG_MSG)
    assert w.layout().minimumSize().width() < 300


def test_set_status_text_elides_and_keeps_full_text_in_tooltip(qapp):
    w = _widget_with_label(qapp)
    w._guard_status_label()
    w.resize(400, 60)
    w.show()                       # offscreen; gives the label a real width
    w._set_status_text(LONG_MSG)
    assert w.statusLabel.toolTip() == LONG_MSG
    shown = w.statusLabel.text()
    assert shown.endswith('…'), "long message should elide"
    assert len(shown) < len(LONG_MSG)
    w.hide()


def test_set_status_text_short_message_unchanged(qapp):
    w = _widget_with_label(qapp)
    w._guard_status_label()
    w.resize(400, 60)
    w.show()
    w._set_status_text('Ready')
    assert w.statusLabel.text() == 'Ready'
    assert w.statusLabel.toolTip() == 'Ready'
    w.hide()


def test_status_label_lookup_prefers_statusLabel_then_ui_specLabel(qapp):
    w = wranglerWidget('f', threading.Condition())
    assert w._status_label() is None      # neither exists -> no-op, no crash
    w._guard_status_label()               # must not raise
    w._set_status_text('x')               # must not raise

    class _UI:
        pass
    w.ui = _UI()
    w.ui.specLabel = QtWidgets.QLabel()
    assert w._status_label() is w.ui.specLabel

    w.statusLabel = QtWidgets.QLabel()
    assert w._status_label() is w.statusLabel
