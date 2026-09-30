"""The calibration window's fits, each shown in its own window (maintainer, 2026-09-29: "for Direct
beam distance fit, it will be good to pop up a window with the fit that also shows the fit distance";
"the results of the refinement should also be displayed in the message box, and also a popup").

A summary above two plots, as the analysis fit dialogs plot data, fit and residuals. The direct-beam
fit: the beam spot's centre along the motion against the motor angle, the fitted model through the
frames it used, and the residuals below. A ring fit (Fit or Refine): each control point's |q|
residual across the arm's angles, and against its ring's q below.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

_USED = (189, 147, 249)
_LEFT_OUT = (230, 110, 110)
_FIT = (80, 200, 160)
_ZERO = (130, 130, 140)
#: One colour per row of the other circle (nu rows under a del arm), as Stitch's overlay colours.
_ROWS = ((31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
         (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207), (127, 127, 127))

#: The fitted parameters as the window names them, with their units (fit_bounds' public names).
PARAMETER_LABELS = {
    'distance_mm': ('distance', 'mm'), 'poni1_mm': ('PONI 1', 'mm'), 'poni2_mm': ('PONI 2', 'mm'),
    'del_zero_deg': ('del zero', '°'), 'nu_zero_deg': ('nu zero', '°'),
    'del_held_deg': ('held del (fixed pitch)', '°'), 'nu_held_deg': ('held nu (fixed yaw)', '°'),
    'mount_tilt_deg': ('mount tilt', '°'), 'mount_roll_deg': ('mount roll', '°'),
    'sample_offset1_mm': ('sample offset 1', 'mm'), 'sample_offset2_mm': ('sample offset 2', 'mm'),
    'sample_offset3_mm': ('sample offset 3', 'mm'), 'energy_eV': ('energy', 'eV')}


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


def calibration_fit_text(fit, kind='fit'):
    """One line for a ring fit (``kind`` 'fit' or 'refine'): how well it fits, its distance and zeros, its
    energy when refined, the model it kept, and any parameter it left at a limit."""
    provenance = fit.asset.get('provenance', {})
    residuals = fit.point_residuals
    frames = len(residuals['frames']) if residuals else len(provenance.get('frames') or ())
    text = (f"{'Refine' if kind == 'refine' else 'Fit'} complete: RMS |q| {fit.rms_q:.3g} Å⁻¹ over "
            f'{fit.n_points} points on {frames} frames')
    rejected = int(np.count_nonzero(~np.asarray(residuals['kept']))) if residuals else 0
    if rejected:
        text += f' ({rejected} more rejected as outliers)'
    parameters = (provenance.get('fit_bounds') or {}).get('parameters') or {}
    for name in ('distance_mm', 'del_zero_deg', 'nu_zero_deg', 'del_held_deg', 'nu_held_deg'):
        if name in parameters:
            label, unit = PARAMETER_LABELS[name]
            text += f" · {label} {parameters[name]['fitted']:.6g}{'' if unit == '°' else ' '}{unit}"
    energy = provenance.get('energy_refinement')
    if energy:
        text += f" · energy {energy['refined_eV']:.1f} ± {energy['sigma_eV']:.1f} eV"
    if provenance.get('arm_model') == 'stacked_extended':
        text += ' · detector mount and sample offset refined'
    at_limit = [PARAMETER_LABELS.get(name, (name, ''))[0] for name, row in parameters.items() if row['at_bound']]
    if at_limit:
        text += ' · at a limit: ' + ', '.join(at_limit)
    return text


class FitWindow(QtWidgets.QDialog):
    """A fit in its own window: a selectable summary above an upper and a lower plot."""

    def __init__(self, parent, name, title, *, stretch=(1, 1)):
        super().__init__(parent)
        self.setObjectName(name)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(640, 520)
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName(f'{name[:-len("Window")]}Summary')
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(self.summary.textInteractionFlags()
                                             | pg.QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(240)
        self.plot.addLegend(offset=(-10, 10))
        self.residuals = pg.PlotWidget()
        layout.addWidget(self.plot, stretch[0])
        layout.addWidget(self.residuals, stretch[1])

    def _clear(self, *zero):
        """Empty both plots; draw the zero line on the ``zero`` ones (residual plots)."""
        for plot in (self.plot, self.residuals):
            plot.clear()
        for plot in zero:
            plot.addLine(y=0, pen=pg.mkPen(_ZERO, width=1))


class DirectBeamFitWindow(FitWindow):
    """The fit behind a direct-beam distance: centres, the model through the used frames, residuals."""

    def __init__(self, parent=None):
        super().__init__(parent, 'directBeamFitWindow', 'Direct-beam fit', stretch=(3, 1))
        self.residuals.setMaximumHeight(140)
        self.plot.setLabel('left', 'Beam centre along the motion (px)')
        self.residuals.setXLink(self.plot)
        self.residuals.setLabel('left', 'Residual (px)')
        self.residuals.setLabel('bottom', 'Motor angle (°)')

    def show_result(self, result, energy_eV=None, held=None):
        """Plot ``result`` (a ``DirectBeamResult``) and say its distance."""
        self.summary.setText(direct_beam_fit_text(result, energy_eV, held)
                             + f' · {result.model} model, {int(np.count_nonzero(result.used))} of '
                               f'{len(result.angles_deg)} frames used')
        angles = np.asarray(result.angles_deg, dtype=float)
        centres = np.asarray(result.centres_px, dtype=float)
        used = np.asarray(result.used, dtype=bool)
        order = np.argsort(angles[used])
        self._clear(self.residuals)
        self.plot.plot(angles[used], centres[used], pen=None, symbol='o', symbolSize=7,
                       symbolBrush=_USED, symbolPen=None, name='used frames')
        if (~used).any():
            finite = ~used & np.isfinite(centres)
            self.plot.plot(angles[finite], centres[finite], pen=None, symbol='x', symbolSize=8,
                           symbolPen=pg.mkPen(_LEFT_OUT), name='left out')
        self.plot.plot(angles[used][order], np.asarray(result.predictions_px, dtype=float)[used][order],
                       pen=pg.mkPen(_FIT, width=2), name=f'{result.model} fit')
        self.residuals.plot(angles[used], np.asarray(result.residuals_px, dtype=float)[used], pen=None,
                            symbol='o', symbolSize=6, symbolBrush=_USED, symbolPen=None)


class CalibrationFitWindow(FitWindow):
    """A ring fit (Fit or Refine): each control point's |q| residual across the arm's angles, coloured
    by the other circle's row, and against its ring's q; the points the fit rejected are crosses."""

    def __init__(self, parent=None):
        super().__init__(parent, 'calibrationFitWindow', 'Calibration fit')
        self.plot.setLabel('left', '|q| residual (10⁻³ Å⁻¹)')
        self.residuals.setLabel('left', '|q| residual (10⁻³ Å⁻¹)')
        self.residuals.setLabel('bottom', 'Ring q (Å⁻¹)')

    def show_result(self, fit, kind='fit'):
        """Plot ``fit`` (a ``CalibrationFit``; ``kind`` 'fit' or 'refine') and say what it found."""
        self.setWindowTitle('Automatic refinement' if kind == 'refine' else 'Calibration fit')
        provenance = fit.asset.get('provenance', {})
        lines = [calibration_fit_text(fit, kind)]
        validation = (provenance.get('automatic_refinement') or {}).get('validation') or {}
        sigma = validation.get('held_out_sigma_A^-1')
        if sigma:
            lines.append(f"Each nu row predicted from the others: σ {sigma['stacked']:.3g} Å⁻¹ with the "
                         f"five-parameter arm, {sigma['stacked_extended']:.3g} Å⁻¹ with the detector mount "
                         "and sample offset")
        if fit.message:
            lines.append(fit.message)
        self.summary.setText('\n'.join(lines))
        self._clear(self.plot, self.residuals)
        residuals = fit.point_residuals
        if not residuals:
            self.plot.setTitle('This fit reports no control points to plot')
            return
        self.plot.setTitle(None)
        motors = provenance.get('source_motors') or {}  # a pyFAI fit's: nu_angle and del_angle
        along = motors.get('del_angle') or motors.get('nu_angle')  # del, or a single arm's one circle
        other = motors.get('nu_angle') if along == motors.get('del_angle') else None  # its rows
        readings = [frame_motors for _index, frame_motors in residuals['frames']]
        frame = np.asarray(residuals['frame'])
        kept = np.asarray(residuals['kept'], dtype=bool)
        dq = np.asarray(residuals['dq'], dtype=float) * 1e3
        q = np.asarray(residuals['q'], dtype=float)
        x = np.array([readings[i].get(along, 0.) for i in frame]) if along else frame.astype(float)
        self.plot.setLabel('bottom', f'{along} (°)' if along else 'Frame')
        rows = np.array([round(readings[i].get(other, 0.), 6) for i in frame]) if other else np.zeros(len(frame))
        for n, row in enumerate(np.unique(rows)):
            chosen = kept & (rows == row)
            colour = _ROWS[n % len(_ROWS)]
            name = f'{other} {row:g}°' if other else 'control points'
            self.plot.plot(x[chosen], dq[chosen], pen=None, symbol='o', symbolSize=5,
                           symbolBrush=colour, symbolPen=None, name=name)
            self.residuals.plot(q[chosen], dq[chosen], pen=None, symbol='o', symbolSize=5,
                                symbolBrush=colour, symbolPen=None)
        if (~kept).any():
            for plot, values in ((self.plot, x), (self.residuals, q)):
                plot.plot(values[~kept], dq[~kept], pen=None, symbol='x', symbolSize=7,
                          symbolPen=pg.mkPen(_ZERO), name='rejected' if plot is self.plot else None)
