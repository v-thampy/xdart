"""AUD-9 stage 2b item 8 (the maintainer: "Yes for calibration window fix"): the shared calibration window describes the
detector of a camera that turns the stored image by a quarter as that same detector turned, in the window's frame:
shape transposed, pixel sizes swapped, its own module gaps turned by the same ``ImageOrientation.apply`` as the
images; every record of it agrees (the pyFAI goniometer record, the xrayutilities partner, their readers), and every
other camera keeps the model itself, record for record."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.geometry.diffractometer import Diffractometer, ImageOrientation
from xrd_tools.integrate.turned_detector import (
    TURNED_DETECTOR_KEY, frame_detector_record, partner_detector, turned_model_mask, window_detector,
)

QUARTER = (ImageOrientation(rotation=90), ImageOrientation(rotation=270),
           ImageOrientation(rotation=90, flip_horizontal=True), ImageOrientation(rotation=270, flip_horizontal=True))
KEEPING = (ImageOrientation(), ImageOrientation(rotation=180), ImageOrientation(flip_horizontal=True),
           ImageOrientation(rotation=180, flip_horizontal=True))
#: a gapped model (three modules, 16558 gap pixels) and RSM's single-module one
MODELS = ("Pilatus300k", "Pilatus100k")
_MOTORS = {"del_angle": "del", "nu_angle": "nu"}


def _native(model):
    from pyFAI.detectors import detector_factory
    return detector_factory(model)


def _native_gaps(model):
    mask = _native(model).calc_mask()
    return np.zeros(_native(model).shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("orientation", QUARTER, ids=lambda o: f"{o.rotation}{'m' if o.flip_horizontal else ''}")
def test_a_quarter_turned_model_is_the_model_turned_with_the_images(model, orientation):
    native = _native(model)
    turned, name, config, record = window_detector(model, orientation)
    rows, columns = native.shape
    assert (type(turned).__name__, name) == ("Detector", "Detector")
    assert tuple(turned.shape) == (columns, rows) and (turned.pixel1, turned.pixel2) == (native.pixel2, native.pixel1)
    assert config == {"pixel1": native.pixel2, "pixel2": native.pixel1, "max_shape": [columns, rows], "orientation": 3}
    assert record == {"detector": model, "image_rotation": orientation.rotation,
                      "image_flip": orientation.flip_horizontal}
    expected = orientation.apply(_native_gaps(model))
    gaps = turned_model_mask(model, orientation)
    if expected.any():  # the gaps travel turned, every one of them
        assert int(expected.sum()) == int(_native_gaps(model).sum()) == 16558
        np.testing.assert_array_equal(gaps, expected)
        np.testing.assert_array_equal(np.asarray(turned.mask, dtype=bool), expected)
    else:
        assert gaps is None


@pytest.mark.parametrize("orientation", KEEPING, ids=lambda o: f"{o.rotation}{'m' if o.flip_horizontal else ''}")
def test_a_camera_keeping_the_axes_keeps_the_model_itself(orientation):
    for pyfai_orientation, config in ((3, {}), (2, {"orientation": 2})):
        detector, name, written, record = window_detector("Pilatus300k", orientation,
                                                          pyfai_orientation=pyfai_orientation)
        assert (type(detector).__name__, name, written, record) == ("Pilatus300k", "Pilatus300k", config, None)


def _rough(model, orientation=None, **extra):
    from xrd_tools.integrate.calibration_seed import rough_arm_record
    return rough_arm_record(distance_m=1.0147173, detector=model, beam_px=(97.0, 243.0),
                            wavelength_m=0.953724089788569e-10, camera=orientation, **extra)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("orientation", QUARTER, ids=lambda o: f"{o.rotation}{'m' if o.flip_horizontal else ''}")
def test_the_rough_record_of_a_quarter_turned_camera_is_in_the_turned_frame(model, orientation):
    """The goniometer record names pyFAI's generic turned detector (which pyFAI itself loads) and, beside it, the
    model and turn; the beam pixel is the turned frame's; every integrator built from it masks the turned gaps."""
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    native = _native(model)
    record = _rough(model, orientation)
    assert record["detector"] == "Detector" and record["detector_config"]["max_shape"] == list(native.shape)[::-1]
    assert record[TURNED_DETECTOR_KEY] == {"detector": model, "image_rotation": orientation.rotation,
                                           "image_flip": orientation.flip_horizontal}
    poni1, poni2, dist = record["param"][1], record["param"][2], record["param"][0]
    assert (poni1, poni2) == pytest.approx(((243.0 + 0.5) * native.pixel2, (97.0 + 0.5) * native.pixel1),
                                           rel=1e-12, abs=0)
    calibration = Diffractometer.from_pyfai_goniometer(record, source_motors=_MOTORS).calibration
    assert calibration.turned_from == record[TURNED_DETECTOR_KEY] and dist == pytest.approx(1.0147173, abs=0)
    detector = detector_calibration_to_integrator(calibration).detector
    assert tuple(detector.shape) == tuple(native.shape)[::-1]
    expected = orientation.apply(_native_gaps(model))
    if expected.any():
        np.testing.assert_array_equal(np.asarray(detector.mask, dtype=bool), expected)


def test_pyfai_itself_loads_the_turned_record(tmp_path):
    from pyFAI.goniometer import Goniometer
    path = tmp_path / "rough.json"
    path.write_text(json.dumps(_rough("Pilatus300k", ImageOrientation(rotation=90))), encoding="utf-8")
    goniometer = Goniometer.sload(str(path))
    assert type(goniometer.detector).__name__ == "Detector" and tuple(goniometer.detector.shape) == (487, 619)


@pytest.mark.parametrize("orientation", KEEPING, ids=lambda o: f"{o.rotation}{'m' if o.flip_horizontal else ''}")
def test_the_rough_record_of_a_camera_keeping_the_axes_is_byte_for_byte_the_model_s(orientation):
    for model in MODELS:
        assert json.dumps(_rough(model, orientation), sort_keys=True) == json.dumps(_rough(model), sort_keys=True)
        assert TURNED_DETECTOR_KEY not in _rough(model, orientation) and _rough(model, orientation)["detector"] == model


def _pair(root: Path, model, orientation, name="cal"):
    """The calibration pair the window saves for ``model`` under ``orientation`` (rough geometry, three validation
    frames), written beside each other in ``root``; its partner asset."""
    from xrd_tools.analysis.xu_stitch_calibration import _canonical_bytes
    from xrd_tools.integrate.xu_transfer import transfer_arm_to_xu
    record = _rough(model, orientation)
    asset, transfer = transfer_arm_to_xu(record, validation_angles=[(0.0, 0.0), (12.0, 0.0), (25.0, 4.0)],
                                         image_rotation=orientation.rotation, image_flip=orientation.flip_horizontal)
    assert transfer["max_abs_dq_A^-1"] <= 1e-8
    (root / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    (root / f"{name}.xu.json").write_bytes(_canonical_bytes(asset))
    return asset


@pytest.mark.parametrize(("model", "orientation"), [
    ("Pilatus300k", ImageOrientation(rotation=90)),
    ("Pilatus300k", ImageOrientation(rotation=270, flip_horizontal=True)),
    ("Pilatus100k", ImageOrientation(rotation=270)),
], ids=["300k-90", "300k-270m", "100k-270"])
def test_a_quarter_turned_pair_names_the_model_in_the_turned_frame_and_every_reader_rebuilds_its_gaps(
        tmp_path, model, orientation):
    """The partner: the model's name, the turned frame's shape and pitches, the turned gaps' count and digest; its
    readers (Stitch's XU capture and mask, RSM's capture, raw-frame geometry and S3c mask) take it as the turned
    model: RSM's raw frame is the model's own, with the model's own gaps."""
    from xrd_tools.analysis.rsm_geometry_asset import (
        capture_rsm_calibration_geometry, lower_rsm_effective_geometry, rsm_calibration_detector_mask,
    )
    from xrd_tools.analysis.stitch_native import calibration_detector_mask
    from xrd_tools.analysis.xu_stitch_calibration import XuStitchCalibrationInput, capture_xu_stitch_calibration
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
    native = _native(model)
    asset = _pair(tmp_path, model, orientation)
    detector = asset["detector"]
    turned_gaps = orientation.apply(_native_gaps(model))
    assert detector["detector_factory_name"] == model and detector["shape"] == list(native.shape)[::-1]
    assert (detector["pixel1_m"], detector["pixel2_m"]) == (native.pixel2, native.pixel1)
    assert "max_shape" not in detector["detector_config"] and detector["mask_count"] == int(turned_gaps.sum())
    assert (detector["cch1_px"], detector["cch2_px"]) == pytest.approx((243.0, 97.0), abs=1e-9)
    arm = asset["corrections"]["solid_angle_arm"]
    assert (arm["detector"], arm["detector_config"]) == (model, detector["detector_config"])
    assert frame_detector_record(model, detector["detector_config"], detector["shape"])[0] == "Detector"
    rebuilt, mask = partner_detector(detector, asset["acquisition"])
    assert tuple(rebuilt.shape) == tuple(detector["shape"])
    stitch_mask = calibration_detector_mask(asset)
    receipt = capture_xu_stitch_calibration(XuStitchCalibrationInput(tmp_path / "cal.xu.json"), project_root=tmp_path)
    with xu_runtime_session() as runtime:
        geometry = resolve_xu_stitch_effective_geometry(receipt, runtime)
    assert geometry is not None
    rsm = capture_rsm_calibration_geometry("cal.json", project_root=tmp_path)
    lowered = lower_rsm_effective_geometry(rsm)
    header = lowered.detector_header
    assert (header.Nch1, header.Nch2) == tuple(native.shape)  # RSM's raw frame: the model's own
    raw_mask = rsm_calibration_detector_mask(rsm)
    if turned_gaps.any():
        np.testing.assert_array_equal(mask, turned_gaps)
        np.testing.assert_array_equal(stitch_mask, turned_gaps)
        np.testing.assert_array_equal(raw_mask, _native_gaps(model))  # the model's own gaps, where they are
    else:
        assert mask is None and stitch_mask is None and raw_mask is None


def test_a_pair_the_old_window_saved_for_a_quarter_turned_camera_reads_as_it_did(tmp_path):
    """The old window wrote the model in its own shape with the beam pixel turned: read as before (its own shape),
    so RSM's raw frame is still the transposed one it refuses for the model's images."""
    from xrd_tools.analysis.rsm_geometry_asset import capture_rsm_calibration_geometry, lower_rsm_effective_geometry
    from xrd_tools.analysis.xu_stitch_calibration import _canonical_bytes
    from xrd_tools.integrate.xu_transfer import transfer_arm_to_xu
    record = _rough("Pilatus100k")  # the model's own frame, as the old window wrote it
    asset, _transfer = transfer_arm_to_xu(record, validation_angles=[(0.0, 0.0), (12.0, 0.0), (25.0, 4.0)],
                                          image_rotation=270)
    assert asset["detector"]["shape"] == [195, 487] and asset["acquisition"]["image_rotation_quarter_turns"] == 3
    (tmp_path / "old.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "old.xu.json").write_bytes(_canonical_bytes(asset))
    rebuilt, _mask = partner_detector(asset["detector"], asset["acquisition"])
    assert type(rebuilt).__name__ == "Pilatus100k"  # the old reading
    header = lower_rsm_effective_geometry(capture_rsm_calibration_geometry("old.json", project_root=tmp_path)
                                          ).detector_header
    assert (header.Nch1, header.Nch2) == (487, 195)  # not the model's 195 × 487 images: RSM refuses them


def test_stitch_runs_a_quarter_turned_window_calibration_with_the_model_s_gaps(tmp_path):
    """Stitch's pyFAI backend on a gapped model's images, stored in the model's own shape, under a quarter turn,
    from the record the window writes: its parser admits exactly that model turned, its plan's detector mask is the
    model's own gaps in the stored frame (as RSM's is), and the run commits."""
    import tifffile
    from xrd_tools.analysis.module_transaction import (
        MetadataColumnSelector, ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceReceipt,
    )
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, StitchOperationPlan, StitchOperationRefused, capture_stitch_geometry,
        prepare_stitch_operation, run_stitch_operation,
    )
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactOverwrite
    from xrd_tools.sources.selection import image_series_spec
    model, orientation = "Pilatus300k", ImageOrientation(rotation=270)
    root = tmp_path / "source"
    root.mkdir()
    for index in range(2):
        image = root / f"scan_{index + 1:04d}.tif"
        tifffile.imwrite(image, np.full(_native(model).shape, 10 + index, dtype=np.uint16))
        image.with_suffix(".txt").write_text(
            f"# Counters\nI0 = 10\n# Motors\ndel = {5 * index}, nu = 0\n"
            "User: stitch, time: Mon Jan 15 10:30:00 2024  # Temp\n", encoding="utf-8")
    table = run_metadata_table(MetadataTablePlan(image_series_spec(root / "scan_0001.tif", metadata_format="txt")))
    source = ModuleSourceReceipt.from_metadata_table(
        table, kind=ModuleKind.STITCH, selected_labels=(1, 2),
        resolved_selectors=(MetadataColumnSelector("I0"), MetadataColumnSelector("del"), MetadataColumnSelector("nu")))
    record = _rough(model, orientation)
    path = tmp_path / "rough.json"

    def captured(value):
        path.write_text(json.dumps(value), encoding="utf-8")
        return capture_stitch_geometry(StitchGeometryInput(
            path, StitchGeometryKind.PYFAI_GONIOMETER_JSON, source_motors=tuple(sorted(_MOTORS.items())),
            image_rotation=orientation.rotation))
    for forged in ({**record, TURNED_DETECTOR_KEY: dict(record[TURNED_DETECTOR_KEY], detector="Pilatus100k")},
                   {**record, "detector_config": dict(record["detector_config"], pixel1=1e-4)}):
        with pytest.raises(StitchOperationRefused, match="detector configuration is unsafe"):
            captured(forged)  # anything but exactly the model turned stays refused
    plan = StitchOperationPlan(captured(record), mode="1d", npt_1d=96, radial_range=(0.1, 6.0),
                               monitor_selector=MetadataColumnSelector("I0"))
    from xrd_tools.analysis.stitch_native import native_detector_mask
    from xrd_tools.analysis.stitch_operation import _detector_mask, _runtime_geometry
    diffractometer, _orientation = _runtime_geometry(plan.geometry, plan.geometry.content)
    np.testing.assert_array_equal(_detector_mask(diffractometer), orientation.apply(_native_gaps(model)))
    np.testing.assert_array_equal(native_detector_mask(plan), _native_gaps(model))  # the stored frame's
    (tmp_path / "output").mkdir()
    request = prepare_stitch_operation(
        source, ModuleOutputRequest(tmp_path / "output" / "stitched.nexus", AnalysisArtifactKind.STITCH_1D,
                                    AnalysisArtifactOverwrite.CREATE_NEW), plan, project_root=tmp_path)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
