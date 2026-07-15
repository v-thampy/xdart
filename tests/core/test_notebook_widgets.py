"""Notebook-safe widget contracts remain Qt-free and headless-result based."""

from __future__ import annotations

import numpy as np


def test_basic_notebook_widgets_accept_public_data_without_reanalysis():
    from xrd_tools.core.containers import IntegrationResult1D
    from xrd_tools.gui.widgets import (
        ImageViewer,
        PatternViewer,
        PeakFitControls,
        PhaseFitControls,
    )

    q = np.linspace(1.0, 4.0, 16)
    pattern = IntegrationResult1D(q, np.ones_like(q), unit="q_A^-1")
    viewer = PatternViewer(patterns=[pattern])
    assert len(viewer.figure_widget.data) == 1
    assert viewer._offset.continuous_update is False

    image = np.arange(64, dtype=float).reshape(8, 8)
    image_viewer = ImageViewer(image, log_scale=False)
    assert image_viewer.image is image
    assert image_viewer._vmin_pct.continuous_update is False

    peak_calls = []
    peak = PeakFitControls(on_fit=peak_calls.append)
    peak._fit_button.click()
    assert peak_calls and peak_calls[0]["n_peaks"] == 3
    assert peak.sigma_init.continuous_update is False

    phase_calls = []
    phase = PhaseFitControls(["film"], on_fit=phase_calls.append)
    phase._fit_button.click()
    assert phase_calls and phase_calls[0]["enabled_phases"] == ["film"]
