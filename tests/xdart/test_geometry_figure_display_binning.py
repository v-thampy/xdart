"""B-H: frozen real help/cache/display oracle, including pre-writer R1–R5."""
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_help_teaching_input_cache_ties_and_source_free_second_view(qapp, monkeypatch):
    from xdart.gui.widgets.context_help import owner_for
    from xrd_tools import geometry_figures as gf
    from xrd_tools.core.geometry.camera import DetectorCamera
    from xdart.gui.tabs.scattering.scientific_view import ScientificView

    def forbidden(*args, **kwargs):
        pytest.fail('Offline teaching crossed a source/job boundary')
    monkeypatch.setattr('xrd_tools.io.image.read_image', forbidden)
    monkeypatch.setattr('xrd_tools.io.read_image', forbidden)
    monkeypatch.setattr('xdart.gui.tabs.calibration.dialog._run_job', forbidden)
    root = QtWidgets.QWidget(); second = QtWidgets.QWidget()
    try:
        owner_for(root).show_topic('geometry.orientation'); window = owner_for(root).window
        first = window.figure_view.keyed_projection.panels[0].figure
        # FIRST assertion: native1043 parent failure, before any new API access.
        assert max(first.image.shape) <= 512
        cache = gf.teaching_cache(); cache.clear()  # R4: independent of H/B-H ordering.
        baseline = cache.stats()
        overrides = []
        original = QtWidgets.QApplication.setOverrideCursor
        def track(cursor):
            overrides.append(QtGui.QCursor(cursor).shape()); return original(cursor)
        monkeypatch.setattr(QtWidgets.QApplication, 'setOverrideCursor', track)
        window.show_orientation_check()
        assert QtCore.Qt.CursorShape.WaitCursor in overrides
        assert QtWidgets.QApplication.overrideCursor() is None
        assert len(window.findChildren(ScientificView)) == 1
        for step in range(4):
            window.step_row.editor.setCurrentIndex(step); qapp.processEvents()
            finding = cache.get_finding(step)
            native = gf.synthetic_orientation_finding(step)
            assert [(c.camera, c.senses) for c in finding.tied] == [(c.camera, c.senses) for c in native.tied]
            for actual, expected in zip(finding.candidates, native.candidates, strict=True):
                assert actual.camera == expected.camera and actual.senses == expected.senses
                assert actual.prediction == expected.prediction
                np.testing.assert_equal((actual.score, actual.standard_error), (expected.score, expected.standard_error))
                assert actual.ring_scores == expected.ring_scores
            assert finding.identifiable == native.identifiable and finding.policy == native.policy
            if step == 0:
                assert finding.identifiable and finding.best.camera == DetectorCamera('x-', 'z+')
            if step == 1:
                pairs = {(c.camera, c.senses[0]) for c in finding.tied}
                assert (DetectorCamera('x+', 'z+'), 1) in pairs
                assert (DetectorCamera('x-', 'z+'), -1) in pairs
                for camera, delta in pairs:
                    pair = [c for c in finding.tied if c.camera == camera and c.senses[0] == delta]
                    assert {c.senses[1] for c in pair} == {-1, 1}
                    assert pair[0].score == pair[1].score and pair[0].prediction == pair[1].prediction
                assert any('ν remains unmeasured' in p.figure.caption
                           for p in window.figure_view.keyed_projection.panels)
            if step == 3:
                assert not finding.identifiable and len(finding.tied) == 8
                assert DetectorCamera('x-', 'z+') in {c.camera for c in finding.tied}
                assert 'diagonal' in finding.reason.lower()
            for panel in window.figure_view.keyed_projection.panels:
                figure = panel.figure
                if figure.image is None:
                    continue
                mapping = figure.display_binning
                assert mapping.native_shape == gf.GeometryInputs().shape
                assert mapping.factor == 3 and figure.image.shape == (348, 327)
                assert not figure.image.flags.writeable and np.isnan(figure.image).any()
                assert 'excluded' in figure.caption.lower()
        physical = gf.GeometryInputs()
        assert physical.distance_m == .2 and physical.wavelength_m == 1e-10
        assert physical.beam_offset_px == (-190., 229.) and physical.shape == (1043, 981)
        # Same native beam in mm: plotting only changes the pixel coordinates.
        _, seed, _ = gf._geometry(physical, DetectorCamera('x-', 'z+'), include_rings=False)
        displayed = cache.get_figures('off-centre', 0)[0]
        beam = displayed.display_binning.display_to_native(displayed.beam_xy)
        native_prediction = gf.predict_ring_curves(physical, gf.camera_candidates()[0].camera)
        np.testing.assert_allclose(beam, native_prediction.beam_xy, atol=1e-12)
        assert seed.integrator.wavelength == physical.wavelength_m
        # R2: help's real signal uses the mapping before the native feature owner.
        window.step_row.editor.setCurrentIndex(2)
        window.feature_row.editor.setCurrentText('x+')
        figure = window.figure_view.keyed_projection.panels[0].figure
        raw = (200., 300.); click = figure.display_binning.native_to_display(raw)
        window.figure_view.figureImageClicked.emit(None, *click)
        expected = gf.feature_azimuth(physical, gf.camera_candidates()[window.candidate_row.editor.currentIndex()].camera,
                                      raw, expected_lab='x+')
        assert expected.reason in window.pane.toPlainText()
        assert 'binn' in window.figure_view.toolTip().lower()
        assert 'native' in window.feature_row.toolTip().lower()
        # 24 entries exceed the finite sixteen-entry LRU.
        for key in ('off-centre', 'mirror-tie', 'centred'):
            for index in range(8):
                window._draw_figure_key(key) if index == 0 else cache.get_figures(key, index)
        stats = cache.stats()
        assert stats['findings'] <= 4 and stats['figures'] == 16
        assert stats['buffers'] <= 16*1024**2 and stats['figure_misses']-baseline['figure_misses'] >= 24
        recent = cache.get_figures('off-centre', 0)
        before = cache.stats()
        owner_for(second).show_topic('geometry.orientation')
        again = owner_for(second).window.figure_view.keyed_projection.panels[0].figure
        after = cache.stats()
        assert after['figure_misses'] == before['figure_misses'] and after['figure_hits'] > before['figure_hits']
        np.testing.assert_array_equal(again.image, recent[0].image)
        assert not hasattr(cache, 'source') and not hasattr(cache, 'integrator')
    finally:
        for widget in (second, root):
            widget.close(); widget.deleteLater()
        qapp.processEvents()
