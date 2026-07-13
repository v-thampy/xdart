# -*- coding: utf-8 -*-
"""Intensity controls (IN-1): manual range is ABSOLUTE across renders, and the
double-click popup types exact min/max.

The live-found "flaky autoscale" symptom: with Autoscale off, every render
re-mapped the manual window PROPORTIONALLY onto the new frame's min/max
(``preserve_fraction=True``), so the user's absolute window silently drifted
on every frame step / live repaint.  Manual is now absolute (the slider span
widens instead of clipping); the proportional carry survives only for
scale re-units (Linear<->Log/Sqrt), where the old window has no overlap with
the new domain."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture()
def image_viewer_frame(qapp):
    """A real staticWidget's displayframe in Image-Viewer mode with real
    displayed data on the real image widget."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    df = widget.displayframe
    df.viewer_mode = 'image'                       # the state _live_mode reads
    df.image_widget.displayed_image = np.linspace(
        0.0, 1000.0, 100).reshape(10, 10)
    yield widget, df
    widget.close()
    widget.deleteLater()


def test_manual_window_is_absolute_across_frame_renders(image_viewer_frame):
    """IN-1 GUARD: with Autoscale off, a new frame's different min/max must
    NOT re-map the manual window — 100..500 stays 100..500."""
    from xdart.gui.tabs.static_scan import display_logic as dl

    _widget, df = image_viewer_frame
    df._intensityAuto.setChecked(False)            # real toggle path
    df._intensitySlider.setDomain(0.0, 1000.0, lower=100.0, upper=500.0,
                                  preserve_fraction=False, emit=False)

    # A new frame renders with double the range; the post-render refresh must
    # keep the user's absolute window (the regression re-mapped it to 200..1000).
    df.image_widget.displayed_image = np.linspace(
        0.0, 2000.0, 100).reshape(10, 10)
    df._refresh_intensity_controls_after_render(dl.Mode.IMAGE_VIEWER)

    assert df._intensitySlider.values() == (100.0, 500.0)
    assert df._image_current_levels(df.image_widget) == (100.0, 500.0)

    # A dimmer frame must not clip-ratchet the window either: the slider span
    # widens to keep 100..500 representable.
    df.image_widget.displayed_image = np.linspace(
        0.0, 300.0, 100).reshape(10, 10)
    df._refresh_intensity_controls_after_render(dl.Mode.IMAGE_VIEWER)
    assert df._intensitySlider.values() == (100.0, 500.0)


def test_manual_window_remaps_proportionally_on_scale_reunit(image_viewer_frame):
    """A Linear->Log-style re-unit (no overlap between the old window and the
    new domain) falls back to the proportional carry instead of a degenerate
    clipped window."""
    from xdart.gui.tabs.static_scan import display_logic as dl

    _widget, df = image_viewer_frame
    df._intensityAuto.setChecked(False)
    df._intensitySlider.setDomain(0.0, 1000.0, lower=250.0, upper=750.0,
                                  preserve_fraction=False, emit=False)

    # log10-like domain far below the old window: proportional carry.
    df.image_widget.displayed_image = np.linspace(
        -3.0, 3.0, 100).reshape(10, 10)
    df._refresh_intensity_controls_after_render(dl.Mode.IMAGE_VIEWER)
    lo, hi = df._intensitySlider.values()
    assert (lo, hi) == (pytest.approx(-1.5), pytest.approx(1.5))
    assert hi > lo                                  # never degenerate


def test_entry_popup_applies_typed_window(image_viewer_frame):
    """Double-click popup: typed min/max switch to manual and apply exactly,
    widening the slider span when the typed window exceeds the data range."""
    _widget, df = image_viewer_frame
    df._intensityAuto.setChecked(True)
    df._intensitySlider.setDomain(0.0, 1000.0, lower=0.0, upper=1000.0,
                                  preserve_fraction=False, emit=False)

    df._open_intensity_entry_popup()               # the double-click target
    popup = df._intensity_entry_popup
    assert popup.isVisible()

    popup._lo_edit.setText("120")
    popup._hi_edit.setText("640")
    df._apply_intensity_entry()

    assert df._intensityAuto.isChecked() is False   # typing IS manual mode
    assert df._intensitySlider.values() == (120.0, 640.0)
    assert df._image_current_levels(df.image_widget) == (120.0, 640.0)
    assert not popup.isVisible()

    # Beyond-domain values widen the span instead of clipping.
    df._open_intensity_entry_popup()
    popup._lo_edit.setText("-50")
    popup._hi_edit.setText("5000")
    df._apply_intensity_entry()
    assert df._intensitySlider.values() == (-50.0, 5000.0)
    assert df._intensitySlider.domain() == (-50.0, 5000.0)

    # Nonsense input is rejected and leaves the popup up for correction.
    df._open_intensity_entry_popup()
    popup._lo_edit.setText("abc")
    popup._hi_edit.setText("10")
    df._apply_intensity_entry()
    assert popup.isVisible()
    assert df._intensitySlider.values() == (-50.0, 5000.0)
