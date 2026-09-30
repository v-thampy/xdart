"""The direct-beam distance fit, shown (maintainer, 2026-09-29: "for Direct beam distance fit, it will be
good to pop up a window with the fit that also shows the fit distance").

The beam spot's centre along the motion against the motor angle, the fitted model's prediction through
the frames it used, and the residuals below, as the analysis fit dialogs plot data, fit and residuals.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


def direct_beam_fit_text(result, energy_eV=None, held=None):
    """One line for a direct-beam fit: its distance and uncertainty, where the beam hits at the zero, the
    other circle as the scan held it, and, when the scan records one, its energy."""
    text = (f'Direct-beam distance: {result.distance_mm:.6g} mm (regression SE '
            f'{result.distance_std_m * 1000:.3g} mm)')
    beam = result.zero_beam_pixel
    if beam is not None:
        text += f' · beam at the zero: {beam[0]:.2f}, {beam[1]:.2f} px'
    for motor, reading in (held or {}).items():
        text += f' · {motor} held at {reading:g}°'
    if energy_eV is not None:
        text += f' · the scan records {energy_eV:.6g} eV'
    return text


class DirectBeamFitWindow(QtWidgets.QDialog):
    """The fit behind a direct-beam distance: centres, the model through the used frames, residuals."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('directBeamFitWindow')
        self.setWindowTitle('Direct-beam fit')
        self.setModal(False)
        self.resize(640, 520)
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName('directBeamFitSummary')
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(self.summary.textInteractionFlags()
                                             | pg.QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(240)
        self.plot.addLegend(offset=(-10, 10))
        self.plot.setLabel('left', 'Beam centre along the motion (px)')
        self.residuals = pg.PlotWidget()
        self.residuals.setMaximumHeight(140)
        self.residuals.setXLink(self.plot)
        self.residuals.setLabel('left', 'Residual (px)')
        self.residuals.setLabel('bottom', 'Motor angle (°)')
        layout.addWidget(self.plot, 3)
        layout.addWidget(self.residuals, 1)

    def show_result(self, result, energy_eV=None, held=None):
        """Plot ``result`` (a ``DirectBeamResult``) and say its distance."""
        self.summary.setText(direct_beam_fit_text(result, energy_eV, held)
                             + f' · {result.model} model, {int(np.count_nonzero(result.used))} of '
                               f'{len(result.angles_deg)} frames used')
        angles = np.asarray(result.angles_deg, dtype=float)
        centres = np.asarray(result.centres_px, dtype=float)
        used = np.asarray(result.used, dtype=bool)
        order = np.argsort(angles[used])
        for plot in (self.plot, self.residuals):
            plot.clear()
        self.residuals.addLine(y=0, pen=pg.mkPen((130, 130, 140), width=1))
        self.plot.plot(angles[used], centres[used], pen=None, symbol='o', symbolSize=7,
                       symbolBrush=(189, 147, 249), symbolPen=None, name='used frames')
        if (~used).any():
            finite = ~used & np.isfinite(centres)
            self.plot.plot(angles[finite], centres[finite], pen=None, symbol='x', symbolSize=8,
                           symbolPen=pg.mkPen((230, 110, 110)), name='left out')
        self.plot.plot(angles[used][order], np.asarray(result.predictions_px, dtype=float)[used][order],
                       pen=pg.mkPen((80, 200, 160), width=2), name=f'{result.model} fit')
        self.residuals.plot(angles[used], np.asarray(result.residuals_px, dtype=float)[used], pen=None,
                            symbol='o', symbolSize=6, symbolBrush=(189, 147, 249), symbolPen=None)
