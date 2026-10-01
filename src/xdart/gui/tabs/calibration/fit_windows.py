"""The calibration window's fits, each shown in its own window (maintainer, 2026-09-29: "for Direct
beam distance fit, it will be good to pop up a window with the fit that also shows the fit distance";
"the results of the refinement should also be displayed in the message box, and also a popup").

Each window mounts the shared fit display (``tabs/scattering/fit_view.FitView``; Codex's Stitch/Int reuse
audit R2): a summary above two panels, as the analysis fit dialogs show data, fit and residuals. The
direct-beam fit: the beam spot's centre along the motion against the motor angle, the fitted model through
the frames it used, and the residuals below. A ring fit (Fit or Refine): each control point's |q| residual
across the arm's angles, one group per row of the other circle, and against its ring's q below.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.fit_view import FitPanel, FitProjection, FitSeries, FitView

#: The fitted parameters in plain words, with their units (fit_bounds' public names; maintainer,
#: 2026-09-29: "the zero hold etc are not very descriptive and helpful"). pyFAI's axes: 1 vertical
#: (up), 2 horizontal, 3 along the beam.
PARAMETER_LABELS = {
    'distance_mm': ('distance', 'mm'), 'poni1_mm': ('PONI 1', 'mm'), 'poni2_mm': ('PONI 2', 'mm'),
    'del_zero_deg': ('del zero', '°'), 'nu_zero_deg': ('nu zero', '°'),
    'del_held_deg': ('del angle (held, not scanned)', '°'), 'nu_held_deg': ('nu angle (held, not scanned)', '°'),
    'mount_tilt_deg': ('detector tilt on the arm', '°'), 'mount_roll_deg': ('detector roll on the arm', '°'),
    'sample_offset1_mm': ('sample offset, vertical', 'mm'), 'sample_offset2_mm': ('sample offset, horizontal', 'mm'),
    'sample_offset3_mm': ('sample offset, along the beam', 'mm'), 'energy_eV': ('energy', 'eV')}
#: A calibration record's parameter names in the same words (its angles are pyFAI's offsets, not zeros).
RECORD_LABELS = {
    'dist': 'distance', 'poni1': 'PONI 1', 'poni2': 'PONI 2', 'nu_offset': 'nu offset', 'del_offset': 'del offset',
    'mount_tilt': 'detector tilt on the arm', 'mount_roll': 'detector roll on the arm',
    'sample_offset1': 'sample offset, vertical', 'sample_offset2': 'sample offset, horizontal',
    'sample_offset3': 'sample offset, along the beam', 'energy_eV': 'energy', 'beam_rotation': 'beam rotation'}


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
    """A fit in its own window: the shared fit display, its summary selectable above two panels.
    ``summary``, ``plot`` and ``residuals`` are the display's own (read by the calibration rows)."""

    def __init__(self, parent, name, title, *, stretch=(1, 1), lower_max_height=None, link_x=False):
        super().__init__(parent)
        self.setObjectName(name)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(640, 520)
        layout = QtWidgets.QVBoxLayout(self)
        self.view = FitView(self, name=name[:-len('Window')], stretch=stretch, upper_min_height=240,
                            lower_max_height=lower_max_height, link_x=link_x)
        layout.addWidget(self.view)
        self.summary, self.plot, self.residuals = self.view.summary, self.view.plot, self.view.residuals


class DirectBeamFitWindow(FitWindow):
    """The fit behind a direct-beam distance: centres, the model through the used frames, residuals."""

    _UPPER = FitPanel(y_label='Beam centre along the motion (px)')
    _LOWER = FitPanel(x_label='Motor angle (°)', y_label='Residual (px)', zero_line=True)

    def __init__(self, parent=None):
        super().__init__(parent, 'directBeamFitWindow', 'Direct-beam fit', stretch=(3, 1), lower_max_height=140,
                         link_x=True)
        self.view.reconcile(FitProjection(self._UPPER, self._LOWER))

    def show_result(self, result, energy_eV=None, held=None):
        """Plot ``result`` (a ``DirectBeamResult``) and say its distance."""
        angles = np.asarray(result.angles_deg, dtype=float)
        centres = np.asarray(result.centres_px, dtype=float)
        used = np.asarray(result.used, dtype=bool)
        order = np.argsort(angles[used])
        upper = [FitSeries(angles[used], centres[used], 'data', 'used frames')]
        if (~used).any():
            finite = ~used & np.isfinite(centres)
            upper.append(FitSeries(angles[finite], centres[finite], 'rejected', 'left out'))
        upper.append(FitSeries(angles[used][order], np.asarray(result.predictions_px, dtype=float)[used][order],
                               'fit', f'{result.model} fit'))
        residuals = FitSeries(angles[used], np.asarray(result.residuals_px, dtype=float)[used], 'residual_points')
        summary = (direct_beam_fit_text(result, energy_eV, held)
                   + f' · {result.model} model, {int(np.count_nonzero(used))} of {len(result.angles_deg)} frames used')
        self.view.reconcile(FitProjection(replace(self._UPPER, series=tuple(upper)),
                                          replace(self._LOWER, series=(residuals,)), summary))


class CalibrationFitWindow(FitWindow):
    """A ring fit (Fit or Refine): each control point's |q| residual across the arm's angles, grouped by
    the other circle's row, and against its ring's q; the points the fit rejected are crosses."""

    _UPPER = FitPanel(y_label='|q| residual (10⁻³ Å⁻¹)', zero_line=True)
    _LOWER = FitPanel(x_label='Ring q (Å⁻¹)', y_label='|q| residual (10⁻³ Å⁻¹)', zero_line=True)

    def __init__(self, parent=None):
        super().__init__(parent, 'calibrationFitWindow', 'Calibration fit')
        self.view.reconcile(FitProjection(self._UPPER, self._LOWER))

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
        residuals = fit.point_residuals
        if not residuals:
            self.view.reconcile(FitProjection(
                replace(self._UPPER, title='This fit reports no control points to plot'), self._LOWER,
                '\n'.join(lines)))
            return
        motors = provenance.get('source_motors') or {}  # a pyFAI fit's: nu_angle and del_angle
        along = motors.get('del_angle') or motors.get('nu_angle')  # del, or a single arm's one circle
        other = motors.get('nu_angle') if along == motors.get('del_angle') else None  # its rows
        readings = [frame_motors for _index, frame_motors in residuals['frames']]
        frame = np.asarray(residuals['frame'])
        kept = np.asarray(residuals['kept'], dtype=bool)
        dq = np.asarray(residuals['dq'], dtype=float) * 1e3
        q = np.asarray(residuals['q'], dtype=float)
        x = np.array([readings[i].get(along, 0.) for i in frame]) if along else frame.astype(float)
        rows = np.array([round(readings[i].get(other, 0.), 6) for i in frame]) if other else np.zeros(len(frame))
        upper, below = [], []
        for n, row in enumerate(np.unique(rows)):
            chosen = kept & (rows == row)
            upper.append(FitSeries(x[chosen], dq[chosen], 'group', f'{other} {row:g}°' if other else 'control points', n))
            below.append(FitSeries(q[chosen], dq[chosen], 'group', None, n))
        if (~kept).any():
            upper.append(FitSeries(x[~kept], dq[~kept], 'rejected', 'rejected'))
            below.append(FitSeries(q[~kept], dq[~kept], 'rejected'))
        self.view.reconcile(FitProjection(
            replace(self._UPPER, series=tuple(upper), x_label=f'{along} (°)' if along else 'Frame'),
            replace(self._LOWER, series=tuple(below)), '\n'.join(lines)))
