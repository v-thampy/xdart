"""B-P: native scoring, real staged input, independent rings and display clicks."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize('model,shape', [('Pilatus1M', (1043, 981)), ('Eiger16M', (4371, 4150))])
@pytest.mark.parametrize('rotation', [0, 90])
@pytest.mark.parametrize('tilt,pose', [((0., 0.), (0., 0.)), ((2., -1.), (15., 7.))])
def test_preview_caps_native_scores_scaled_rings_and_feature_click(qapp, tmp_path, monkeypatch,
                                                                   model, shape, rotation, tilt, pose):
    from scipy.spatial import cKDTree
    from pyFAI.detectors import detector_factory
    from xrd_tools import geometry_figures as gf
    from xrd_tools.core.geometry import ImageOrientation
    from xrd_tools.core.geometry.camera import stitch_camera, stitch_orientation
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.integrate.calibration_seed import rough_arm_record
    from xrd_tools.integrate.stitch_calibration import CalibrationFrame
    from tests.core.test_geometry_figures import reference

    orientation = ImageOrientation(rotation=rotation); camera = stitch_camera(orientation)
    value = gf.GeometryInputs(shape=shape, detector=model, mount_deg=tilt,
                              readings=(('del', pose[0]), ('nu', pose[1])))
    ai, image = reference(value, camera)
    gap = orientation.unapply(np.asarray(ai.detector.mask, bool))
    image = image.astype(np.float32)
    image[gap] = -1 if model == 'Pilatus1M' else 4294967295
    image[20, 20] = -25.  # legitimate negative, not a gap heuristic
    image[21, 20] = np.nan
    image[22, 20] = 1e7
    static = np.zeros(shape, bool); static[30:60, 30:60] = True
    staged = orientation.apply(image)
    path = tmp_path/'verified.npy'; np.save(path, staged)
    frame = CalibrationFrame('f', 'source', 'spec', {}, 0, str(path), 0, str(path),
        hashlib.sha256(path.read_bytes()).hexdigest(), staged.shape, rotation,
        dict(value.readings), wavelength_m_to_energy_eV(value.wavelength_m))
    centre = (np.array(staged.shape)[::-1]-1)/2
    record = rough_arm_record(detector=model, distance_m=value.distance_m,
        wavelength_m=value.wavelength_m, beam_px=centre+value.beam_offset_px,
        camera=orientation, mount_deg=tilt, orientation=3)
    projected = gf.calibration_orientation_inputs(record, frame)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    finding, figure, physical = gf.diagnose_calibration_frames((frame,), record,
        mask=orientation.apply(static), saturation=1e6, preview_camera=camera)
    # FIRST assertion: parent fails size, not collection or missing new symbols.
    assert max(figure.image.shape) <= 512
    mapping = figure.display_binning
    b = 3 if model == 'Pilatus1M' else 9
    assert mapping.factor == b and mapping.native_shape == shape
    assert figure.image.shape == tuple((n+b-1)//b for n in shape)
    assert physical.shape == value.shape and physical.detector == value.detector
    assert physical.first_pixel == value.first_pixel and physical.readings == value.readings
    assert physical == projected
    assert physical.wavelength_m == value.wavelength_m
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    native_frame = gf.OrientationFrame(image, physical, static, 1e6)
    native = gf.diagnose_orientation((native_frame,))
    assert finding.exclusions == native.exclusions and finding.policy == native.policy
    assert finding.identifiable == native.identifiable
    np.testing.assert_equal((finding.margin, finding.margin_standard_error),
                            (native.margin, native.margin_standard_error))
    for actual, expected in zip(finding.candidates, native.candidates, strict=True):
        assert actual.camera == expected.camera and actual.senses == expected.senses
        assert actual.prediction == expected.prediction
        np.testing.assert_equal((actual.score, actual.standard_error), (expected.score, expected.standard_error))
        assert actual.ring_scores == expected.ring_scores
    assert [(c.camera, c.senses) for c in finding.tied] == [(c.camera, c.senses) for c in native.tied]
    assert finding.best.camera == native.best.camera and finding.best.senses == native.best.senses
    # R1: compare direct native admitted-mask block means, without a sign heuristic.
    admitted = ~(gap | static | ~np.isfinite(image) | (image >= 1e6))
    assert np.nanmax(figure.image) <= np.max(image[admitted])
    assert np.isnan(figure.image).any()
    for y, x in ((0, 0), (20//b, 20//b), (40//b, 40//b),
                 (figure.image.shape[0]-1, figure.image.shape[1]-1)):
        block = image[y*b:min((y+1)*b, shape[0]), x*b:min((x+1)*b, shape[1])]
        allowed = admitted[y*b:min((y+1)*b, shape[0]), x*b:min((x+1)*b, shape[1])]
        target = np.float32(block[allowed].mean(dtype=np.float64)) if allowed.any() else np.nan
        np.testing.assert_allclose(figure.image[y, x], target, rtol=1e-6, equal_nan=True)
    # Native reference keeps its independent pyFAI 2theta contour oracle.
    prediction = gf.predict_ring_curves(physical, camera)
    for theta in {c.two_theta_deg for c in prediction.curves}:
        expected = np.concatenate([c.xy for c in prediction.curves if c.two_theta_deg == theta])
        actual = np.concatenate([c.xy for c in figure.curves if c.two_theta_deg == theta])
        expected = np.array([mapping.native_to_display(p) for p in expected])
        assert cKDTree(actual).query(expected)[0].max() <= .5
        assert cKDTree(expected).query(actual)[0].max() <= .5
    np.testing.assert_allclose(figure.beam_xy, mapping.native_to_display(prediction.beam_xy), atol=1e-10)
    # Tripwire ONLY display preparation; native synthesis/reference may allocate maps.
    def forbidden(*args, **kwargs):
        pytest.fail('Display preparation allocated a detector-sized angle map')
    monkeypatch.setattr(type(ai), 'center_array', forbidden)
    redrawn = gf.orientation_display(native_frame, camera)
    np.testing.assert_array_equal(redrawn.image, figure.image)
    # Explicit value-mask membership and a legitimate negative survive display means.
    small = np.full(shape, -3., np.float32); value_mask = np.zeros(shape, bool)
    value_mask[30:60, 30:60] = True; small[40, 40] = 1e8
    masked = gf.orientation_display(gf.OrientationFrame(small, value, value_mask=value_mask), camera)
    assert np.nanmax(masked.image) == -3.
    # R2: Eiger's short final4150 axis at identity AND quarter turn, native bound4149.
    if model == 'Eiger16M':
        for edge_rotation in (0, 90):
            transform = ImageOrientation(rotation=edge_rotation)
            edge_shape = shape[::-1] if transform.swaps_axes else shape
            edge_mapping = gf.DisplayBinning(edge_shape, 512)
            axis = 1 if edge_rotation == 0 else 0
            raw = [200., 300.]; raw[1-axis] = edge_mapping.display_shape[axis]-1
            point = edge_mapping.display_to_native(raw)
            assert point[1-axis] == 4149
            native_point = transform.unapply_pixel(point, shape)
            result = gf.feature_azimuth(value, stitch_camera(transform), native_point, expected_lab='x+')
            assert np.isfinite((result.pyfai_chi_deg, result.xrayutilities_chi_deg)).all()
    else:
        from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
        dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
        try:
            dialog.open_orientation_check(); panel = dialog.orientation_panel
            panel.show_finding(finding, figure)
            dialog._orientation_geometry = physical
            panel.step_row.editor.setCurrentIndex(2); panel.feature_row.editor.setCurrentText('x+')
            raw = (200., 300.); click = mapping.native_to_display(raw)
            chosen = finding.candidates[panel.candidate_row.editor.currentData()].camera
            expected = gf.feature_azimuth(value, chosen, raw, expected_lab='x+')
            panel.figure_view.figureImageClicked.emit(None, *click)
            assert panel.finding_pane.toPlainText() == expected.reason
            assert 'binn' in panel.figure_view.toolTip().lower() and 'native' in panel.feature_row.toolTip().lower()
        finally:
            dialog.close(); dialog.deleteLater(); qapp.processEvents()
