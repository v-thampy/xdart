"""Auto-peak detection must reject peakless/diffuse scatter.

Beamline regression: fitting a 1D with no obvious peak (a grazing-incidence cut
with no rings in range) used to auto-detect up to _MAX_PEAKS noise spikes, then
grind a doomed 12-component synchronous fit and end with a cryptic "Fit failed:".
The noise-floor + width gate in ``_detect_peaks`` makes peakless data detect
NOTHING, so the dialog shows the clean "no peaks auto-detected" hint instead --
immediately, with no grind.  A genuine peak well above the noise is still found.

Dialog import is deferred into the tests (it pulls the static_scan GUI stack;
keep module collection headless-safe, per test_gui_logging / test_fit_dialog_esc).
"""
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_detect_peaks_rejects_peakless_noise(qapp):
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    dlg = PeakFitDialog(analysis_context=None)
    try:
        rng = np.random.default_rng(0)
        x = np.linspace(-3.7, 3.7, 800)
        # Flat baseline + Gaussian noise + a handful of TALL single-sample
        # spikes -- exactly the shape (screenshot) that fooled the old
        # 0.04*range prominence floor into "finding" 12 peaks.
        y = 150.0 + rng.normal(0.0, 40.0, x.size)
        for i in rng.integers(0, x.size, 8):
            y[i] += 300.0
        assert dlg._detect_peaks(x, y) == []
    finally:
        dlg.close()


def test_detect_peaks_finds_real_peak_above_noise(qapp):
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    dlg = PeakFitDialog(analysis_context=None)
    try:
        rng = np.random.default_rng(1)
        x = np.linspace(0.0, 5.0, 800)
        y = 100.0 + rng.normal(0.0, 5.0, x.size)
        # A real, multi-sample, high-SNR peak: the noise gate must NOT suppress it.
        y = y + 2000.0 * np.exp(-0.5 * ((x - 2.5) / 0.05) ** 2)
        centers = dlg._detect_peaks(x, y)
        assert len(centers) >= 1
        assert any(abs(c - 2.5) < 0.1 for c in centers)
    finally:
        dlg.close()
