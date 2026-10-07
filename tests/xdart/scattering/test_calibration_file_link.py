"""A selected committed calibration updates the real shared authoring window."""
import json

import pytest

from xdart.gui.pages.services import empty_host_services
from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.core.geometry.camera import stitch_camera
from xrd_tools.integrate.calibration_seed import rough_arm_record
from xrd_tools.session.instrument import InstrumentGeometry
from tests.xdart.pages.test_stitch_page import _Status, qapp  # noqa: F401


def _geometry(path, rotation, distance):
    record = rough_arm_record(distance_m=distance, detector='Pilatus100k', beam_px=(97.25, 243.5),
                              wavelength_m=1e-10, camera=ImageOrientation(rotation=rotation))
    path.write_text(json.dumps(record))
    path.with_suffix('.json.provenance.json').write_text(json.dumps({
        'image_rotation': rotation, 'image_flip': False,
        'instrument_geometry': InstrumentGeometry().to_record(),
        'source_motors': {'del_angle': 'del', 'nu_angle': 'nu'},
    }))


@pytest.mark.parametrize('page_name', ('RSM', 'Stitch'))
def test_selected_file_updates_new_and_retained_window_and_same_file_keeps_edits(qapp, tmp_path, page_name):
    from xdart.gui.tabs.rsm.page import RSMPage
    from xdart.gui.tabs.stitch.page import StitchPage
    page = (RSMPage if page_name == 'RSM' else StitchPage)(empty_host_services(_Status()))
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    _geometry(first, 90, 1.0147)
    _geometry(second, 270, .9)
    try:
        page.set_field('project_root', str(tmp_path))
        page.set_field('geometry_path', str(first))
        page._calibrate()
        dialog = page._calibration_dialog
        assert dialog.camera.value() == stitch_camera(ImageOrientation(rotation=90))
        assert float(dialog._text('rough_distance')) == pytest.approx(1014.7)
        assert tuple(map(float, dialog._text('beam_pixel').split(','))) == pytest.approx((97.25, 243.5))
        assert dialog._text('beam_detector') == 'Pilatus100k'
        assert dialog._text('refine_seed_path') == str(first)
        assert first.name in dialog.windowTitle()
        dialog._set_field_text('source', str(tmp_path / 'my_calibrant.spec'))
        dialog._set_field_text('rms', '.023')
        page.set_field('geometry_path', str(second))
        assert page._calibration_dialog is dialog
        assert dialog.camera.value() == stitch_camera(ImageOrientation(rotation=270))
        assert float(dialog._text('rough_distance')) == pytest.approx(900)
        assert dialog._text('source').endswith('my_calibrant.spec') and dialog._text('rms') == '.023'
        assert dialog.fit is None and not dialog.save_button.isEnabled()
        dialog._set_field_text('rough_distance', '901')
        dialog.hide()
        page._calibrate()
        assert dialog._text('rough_distance') == '901'
        # Saving new bytes at the same path is a new loaded geometry on reopening.
        _geometry(second, 90, .8)
        dialog.hide()
        page._calibrate()
        assert dialog.camera.value() == stitch_camera(ImageOrientation(rotation=90))
        assert float(dialog._text('rough_distance')) == pytest.approx(800)
        before = dialog.settings()
        bad = tmp_path / 'bad.json'
        bad.write_text('{}')
        page.set_field('geometry_path', str(bad))
        assert dialog.settings() == before
        assert 'load' in dialog.status.text().lower()
    finally:
        close = page.close_page if page_name == 'RSM' else page.shutdown
        close()
        page.deleteLater()
        qapp.processEvents()
