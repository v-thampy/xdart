"""AUD-9 stage 2b item 8 (the maintainer: "Yes for calibration window fix"), in the shared calibration window opened
from either page: a camera that turns the stored image by a quarter (RSM's Pilatus 100K, camera z−, x−) has the
window work in the turned image, so the image-size hint says so, and a rough calibration saved there describes the
detector turned with the images: pyFAI's generic detector of the turned shape (the model and turn beside it), the
xrayutilities partner the model in the turned frame with its gaps turned, RSM's raw frame the model's own. Every other
camera saves the model itself, as before (the byte-for-byte comparison with the parent commit is a probe, in the
handoff). The rows read only what the window writes, so they run on the parent too (its fail-before)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.core.geometry.camera import stitch_camera
from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _page_with_dialog, _save_to, _wait

QUARTER = {"270": ImageOrientation(rotation=270), "90": ImageOrientation(rotation=90),
           "270m": ImageOrientation(rotation=270, flip_horizontal=True),
           "90m": ImageOrientation(rotation=90, flip_horizontal=True)}
KEEPING = {"0": ImageOrientation(), "180": ImageOrientation(rotation=180),
           "0m": ImageOrientation(flip_horizontal=True), "180m": ImageOrientation(rotation=180, flip_horizontal=True)}
#: the goniometer record's turned-detector key, spelled out (the parent commit has no module that names it)
TURNED_KEY = "xdart_turned_detector"
#: (column, row) typed in the window's image
BEAM = (97.25, 243.5)


class _Status:
    def show(self, text, timeout_ms=0):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _native(model):
    from pyFAI.detectors import detector_factory
    return detector_factory(model)


def _gaps(model):
    """The model's own module gaps (None: it has none)."""
    mask = _native(model).calc_mask()
    return None if mask is None or not np.any(mask) else np.asarray(mask, dtype=bool)


def _window(page_name, tmp_path):
    """The shared window as the page opens it, and the page's own close."""
    if page_name == "Stitch":
        page, dialog = _page_with_dialog(tmp_path)
        return dialog, page.shutdown
    from xdart.gui.tabs.rsm.page import RSMPage
    page = RSMPage(empty_host_services(_Status()))
    page.set_field("project_root", str(tmp_path))
    page._calibrate()
    return page._calibration_dialog, page.close_page


def _save_rough(dialog, monkeypatch, tmp_path, model, orientation):
    """Type a rough geometry for ``model`` under the camera of ``orientation`` and save it; the saved record, its
    provenance and its xrayutilities partner."""
    from xrd_tools.integrate.stitch_calibration import xu_partner_path
    dialog.camera.set_value(stitch_camera(orientation))
    dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData("manual"))
    for key, value in (("beam_detector", model), ("rough_distance", "1014.7173"),
                       ("beam_pixel", f"{BEAM[0]}, {BEAM[1]}"), ("rough_wavelength", "0.953724089788569"),
                       ("del_domain", "-10, 30"), ("nu_domain", "-5, 5")):
        dialog._set_field_text(key, value)
    target = (tmp_path / "calibration" / "rough.json").resolve()
    _save_to(monkeypatch, target)
    adopted = []
    dialog.calibrationSaved.connect(lambda path, backend: adopted.append((path, backend)))
    dialog.save_rough_button.click()
    _wait(dialog, timeout=120)
    assert adopted == [(str(target), "multigeometry")], dialog.status.text()
    assert xu_partner_path(target).is_file(), dialog.status.text()
    return (json.loads(target.read_text(encoding="utf-8")),
            json.loads(target.with_suffix(".json.provenance.json").read_text(encoding="utf-8")),
            json.loads(xu_partner_path(target).read_text(encoding="utf-8")), target)


#: the Pilatus 100K's image-size hint under each camera
HINTS = {
    "270": "195 × 487 px (x × y) in the window (the stored image is 487 × 195, turned 270°): x 0–194, y 0–486 · "
           "camera rows z-, columns x-",
    "90": "195 × 487 px (x × y) in the window (the stored image is 487 × 195, turned 90°): x 0–194, y 0–486 · "
          "camera rows z+, columns x+",
    "270m": "195 × 487 px (x × y) in the window (the stored image is 487 × 195, turned 270° and mirrored "
            "left-right): x 0–194, y 0–486 · camera rows z+, columns x-",
    "90m": "195 × 487 px (x × y) in the window (the stored image is 487 × 195, turned 90° and mirrored "
           "left-right): x 0–194, y 0–486 · camera rows z-, columns x+",
    # a camera keeping the axes: the hint as it always read
    "0": "487 × 195 px (x × y): x 0–486, y 0–194 · camera rows x-, columns z+",
    "180": "487 × 195 px (x × y): x 0–486, y 0–194 · camera rows x+, columns z-",
    "0m": "487 × 195 px (x × y): x 0–486, y 0–194 · camera rows x-, columns z-",
    "180m": "487 × 195 px (x × y): x 0–486, y 0–194 · camera rows x+, columns z+",
}


@pytest.mark.parametrize("turn", list(HINTS))
def test_the_image_size_hint_says_the_frame_the_window_works_in(qapp, tmp_path, turn):
    dialog = CalibrationDialog(CalibrationContext("RSM", tmp_path))
    try:
        dialog._set_field_text("beam_detector", "Pilatus100k")
        dialog.camera.set_value(stitch_camera({**QUARTER, **KEEPING}[turn]))
        assert dialog._image_size.text() == HINTS[turn]
    finally:
        dialog.close()


@pytest.mark.parametrize(("page_name", "model", "turn"), [
    ("RSM", "Pilatus100k", "270"), ("RSM", "Pilatus300k", "90m"),
    ("Stitch", "Pilatus300k", "90"), ("Stitch", "Pilatus100k", "270m"),
])
def test_a_rough_calibration_saved_for_a_quarter_turned_camera_is_in_the_window_s_frame(
        qapp, tmp_path, monkeypatch, page_name, model, turn):
    """Both pages, both quarter turns and their mirrors, RSM's 100K and a gapped model (16558 gap pixels)."""
    from pyFAI.goniometer import Goniometer
    from xrd_tools.analysis.rsm_geometry_asset import (
        capture_rsm_calibration_geometry, lower_rsm_effective_geometry, rsm_calibration_detector_mask,
    )
    from xrd_tools.analysis.stitch_native import calibration_detector_mask
    from xrd_tools.analysis.xu_stitch_calibration import XuStitchCalibrationInput, capture_xu_stitch_calibration
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    from xrd_tools.integrate.stitch_calibration import xu_partner_path
    from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
    orientation, native = QUARTER[turn], _native(model)
    rows, columns = native.shape
    gaps = _gaps(model)
    turned_gaps = None if gaps is None else orientation.apply(gaps)
    dialog, close = _window(page_name, tmp_path)
    try:
        record, provenance, partner, target = _save_rough(dialog, monkeypatch, tmp_path, model, orientation)
    finally:
        dialog.close()
        close()
    # the pyFAI record: the generic detector of the window's image (pyFAI loads it), the model and turn beside it
    assert (record["detector"], record["detector_config"]) == ("Detector", {
        "pixel1": native.pixel2, "pixel2": native.pixel1, "max_shape": [columns, rows], "orientation": 3})
    assert record[TURNED_KEY] == {"detector": model, "image_rotation": orientation.rotation,
                                  "image_flip": orientation.flip_horizontal}
    assert tuple(Goniometer.sload(str(target)).detector.shape) == (columns, rows)
    assert (provenance["image_rotation"], provenance.get("image_flip", False)) == (
        orientation.rotation, orientation.flip_horizontal)
    calibration = Diffractometer.from_pyfai_goniometer(record, source_motors=provenance["source_motors"]).calibration
    detector = detector_calibration_to_integrator(calibration).detector
    assert tuple(detector.shape) == (columns, rows)
    # the partner: the model in the window's frame, the typed pixel its beam pixel, its gaps turned
    named = partner["detector"]
    assert (named["detector_factory_name"], named["shape"]) == (model, [columns, rows])
    assert (named["pixel1_m"], named["pixel2_m"]) == (native.pixel2, native.pixel1)
    assert (named["cch1_px"], named["cch2_px"]) == pytest.approx((BEAM[1], BEAM[0]), abs=1e-6)
    assert named["mask_count"] == (0 if turned_gaps is None else int(turned_gaps.sum()))
    assert (partner["acquisition"]["image_rotation_quarter_turns"], partner["acquisition"].get("image_flip", False)) \
        == (orientation.rotation // 90, orientation.flip_horizontal)
    # Stitch's readers: the turned gaps; its XU capture admits the pair and resolves its geometry
    stitch_mask = calibration_detector_mask(partner)
    receipt = capture_xu_stitch_calibration(XuStitchCalibrationInput(xu_partner_path(target)), project_root=tmp_path)
    with xu_runtime_session() as runtime:
        assert resolve_xu_stitch_effective_geometry(receipt, runtime) is not None
    # RSM's: the raw frame is the stored image's (the model's own), its mask the model's own gaps
    rsm = capture_rsm_calibration_geometry("calibration/rough.json", project_root=tmp_path)
    header = lower_rsm_effective_geometry(rsm).detector_header
    assert (header.Nch1, header.Nch2) == (rows, columns)
    raw_mask = rsm_calibration_detector_mask(rsm)
    if turned_gaps is None:
        assert detector.mask is None or not np.any(detector.mask)
        assert stitch_mask is None and raw_mask is None
    else:
        assert int(turned_gaps.sum()) == 16558
        np.testing.assert_array_equal(np.asarray(detector.mask, dtype=bool), turned_gaps)
        np.testing.assert_array_equal(stitch_mask, turned_gaps)
        np.testing.assert_array_equal(raw_mask, gaps)


@pytest.mark.parametrize(("page_name", "model", "turn"), [
    ("RSM", "Pilatus100k", "0"), ("RSM", "Pilatus300k", "180m"),
    ("Stitch", "Pilatus300k", "0m"), ("Stitch", "Pilatus100k", "180"),
])
def test_a_rough_calibration_saved_for_a_camera_keeping_the_axes_names_the_model_itself(
        qapp, tmp_path, monkeypatch, page_name, model, turn):
    orientation, native = KEEPING[turn], _native(model)
    dialog, close = _window(page_name, tmp_path)
    try:
        record, provenance, partner, _target = _save_rough(dialog, monkeypatch, tmp_path, model, orientation)
    finally:
        dialog.close()
        close()
    assert (record["detector"], record["detector_config"], TURNED_KEY in record) == (model, {}, False)
    assert (partner["detector"]["detector_factory_name"], partner["detector"]["shape"]) == (model, list(native.shape))
    assert (provenance["image_rotation"], provenance.get("image_flip", False)) == (
        orientation.rotation, orientation.flip_horizontal)


# --- item 8b: the guided fit's and pyFAI-calib2's records under a quarter turn

@pytest.mark.parametrize("page_name", ["RSM", "Stitch"])
def test_the_guided_fit_saved_for_a_quarter_turned_camera_carries_the_turned_model(
        qapp, tmp_path, monkeypatch, page_name):
    """Frames pyFAI-calib2 calibrated with the window's turned detector: Fit takes the Detector panel's model turned
    (a model they were not calibrated with is refused by name), and Save writes the record and partner with it."""
    from xrd_tools.integrate.stitch_calibration import xu_partner_path
    from tests.core.test_turned_detector import _TURNED, _turned_guided
    model, config, turned, shape, _controls, frames = _turned_guided(tmp_path)
    dialog, close = _window(page_name, tmp_path)
    try:
        dialog.camera.set_value(stitch_camera(_TURNED))
        dialog._set_field_text("beam_detector", "Pilatus100k")  # not the frames' detector
        dialog.frames, dialog._frame_motors = frames, ("del", "nu")
        dialog._choices = [(index, dict(frame.motors), None) for index, frame in enumerate(frames)]
        dialog._render_frames()
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is None and "is not the Pilatus100k turned" in dialog.status.text(), dialog.status.text()
        dialog._set_field_text("beam_detector", model)
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        target = (tmp_path / "calibration" / "guided.json").resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        _wait(dialog, timeout=120)
        assert xu_partner_path(target).is_file(), dialog.status.text()
    finally:
        dialog.close()
        close()
    record = json.loads(target.read_text(encoding="utf-8"))
    assert (record["detector"], record["detector_config"], record[TURNED_KEY]) == ("Detector", config, turned)
    partner = json.loads(xu_partner_path(target).read_text(encoding="utf-8"))["detector"]
    gaps = _TURNED.apply(_gaps(model))
    assert (partner["detector_factory_name"], partner["shape"], partner["mask_count"]) == (
        model, list(shape), int(gaps.sum()))


def test_pyfai_calib2_is_given_the_turned_model_s_gaps_and_never_a_bare_generic_detector(qapp, tmp_path):
    """Under a quarter turn pyFAI-calib2 sees the generic turned detector, which holds no mask: the window's mask
    file for it carries the model's gaps turned (with the Detector panel's mask); a camera keeping the axes adds
    nothing, as before. A saved frame naming the generic detector is never passed on as a bare name."""
    import fabio
    from xdart.gui.tabs.calibration.dialog import _pyfai_arguments
    from tests.core.test_turned_detector import _TURNED
    dialog = CalibrationDialog(CalibrationContext("RSM", tmp_path))
    try:
        dialog._set_field_text("beam_detector", "Pilatus300k")
        assert dialog._pyfai_mask() is None  # the default camera keeps the axes, no mask drawn: none
        dialog.camera.set_value(stitch_camera(_TURNED))
        gaps = _TURNED.apply(_gaps("Pilatus300k"))
        np.testing.assert_array_equal(fabio.open(dialog._pyfai_mask()).data.astype(bool), gaps)
        drawn = np.zeros(_native("Pilatus300k").shape, dtype=np.uint8)  # on the stored image
        drawn[:7, :40] = 1
        np.save(tmp_path / "drawn.npy", drawn)
        dialog._set_field_text("mask", str(tmp_path / "drawn.npy"))
        np.testing.assert_array_equal(fabio.open(dialog._pyfai_mask()).data.astype(bool),
                                      gaps | _TURNED.apply(drawn.astype(bool)))
    finally:
        dialog.close()
    frame = type("Frame", (), {"energy_eV": 12000.0})()
    assert "-D" not in _pyfai_arguments(frame, {"calibrant": "LaB6", "detector": "Detector"})
    assert _pyfai_arguments(frame, {"calibrant": "LaB6", "detector": "Pilatus100k"})[-2:] == ("-D", "Pilatus100k")
