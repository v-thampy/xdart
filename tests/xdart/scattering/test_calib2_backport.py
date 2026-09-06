"""Real upstream calibration widget -> saved PONI -> xdart readback."""
from __future__ import annotations

import io

import pyFAI
import pytest
from PySide6 import QtWidgets
from pyFAI.detectors import detector_factory
from pyFAI.detectors._rayonix import _Rayonix
from pyFAI.gui.CalibrationContext import CalibrationContext
from pyFAI.gui.tasks.IntegrationTask import IntegrationTask
from pyFAI.io._json import PyFAIEncoder

from xdart.calib2_main import apply_calibration_backports
from xrd_tools.integrate.calibration import (
    detector_calibration_to_integrator,
    load_detector_calibration,
)


@pytest.fixture(autouse=True)
def restore_backports(monkeypatch):
    monkeypatch.setattr(_Rayonix, "get_config", _Rayonix.get_config)
    monkeypatch.setattr(PyFAIEncoder, "default", PyFAIEncoder.default)
    monkeypatch.setattr(
        IntegrationTask, "_IntegrationTask__getPoni",
        IntegrationTask._IntegrationTask__getPoni,
    )


@pytest.fixture
def calibration_task():
    # No native settings, raw inputs or calibration workload: use the actual
    # upstream widget and models, stopping at its real save builder.
    context = CalibrationContext(settings=None)
    parent = QtWidgets.QWidget()
    context.setParent(parent)  # CalibrationWindow does this before task creation.
    try:
        task = IntegrationTask()
        task.setParent(parent)
        model = context.getCalibrationModel()
        task.setModel(model)
        geometry = model.fittedGeometry()
        with geometry.lockContext():
            geometry.distance().setValue(0.1)
            geometry.poni1().setValue(0.05)
            geometry.poni2().setValue(0.06)
            geometry.rotation1().setValue(0.01)
            geometry.rotation2().setValue(0.02)
            geometry.rotation3().setValue(0.03)
            geometry.wavelength().setValue(1e-10)
        task._geometryTabs.setGeometryModel(geometry)
        yield task
    finally:
        parent.close()
        parent.deleteLater()
        CalibrationContext._releaseSingleton()


@pytest.mark.parametrize("parallax", [False, True])
@pytest.mark.parametrize("detector_name", ["RayonixMx225", "Eiger4M"])
def test_real_gui_save_preserves_sensor_geometry_and_choice(
    calibration_task, tmp_path, detector_name, parallax,
):
    apply_calibration_backports()
    sensor = (
        {"material": "Gd2O2S", "thickness": 40e-6}
        if detector_name == "RayonixMx225"
        else {"material": "Si", "thickness": 450e-6}
    )
    detector = detector_factory(detector_name, {"sensor": sensor})
    settings = calibration_task.model().experimentSettingsModel()
    settings.detectorModel().setDetector(detector)
    settings.parallaxCorrection().setValue(parallax)
    expected_config = detector.get_config()
    poni = calibration_task._IntegrationTask__getPoni()
    output = io.StringIO()
    poni.write(output)
    text = output.getvalue()
    assert "poni_version: 3" in text
    assert f"Parallax: {parallax}" in text
    path = tmp_path / "gui_saved.poni"
    path.write_text(text, encoding="utf-8")
    loaded = load_detector_calibration(path)
    assert dict(loaded.detector_config) == expected_config
    assert loaded.parallax is parallax
    assert (
        loaded.poni.dist, loaded.poni.poni1, loaded.poni.poni2,
        loaded.poni.rot1, loaded.poni.rot2, loaded.poni.rot3,
        loaded.poni.wavelength,
    ) == (0.1, 0.05, 0.06, 0.01, 0.02, 0.03, 1e-10)
    integrator = detector_calibration_to_integrator(loaded)
    assert (integrator.parallax is not None) is parallax
    assert integrator.detector.get_config() == expected_config


def test_sensor_free_save_stays_legacy(calibration_task):
    apply_calibration_backports()
    detector = detector_factory("Detector", {"pixel1": 1e-4, "pixel2": 1e-4})
    detector.sensor = None
    settings = calibration_task.model().experimentSettingsModel()
    settings.detectorModel().setDetector(detector)
    settings.parallaxCorrection().setValue(False)
    poni = calibration_task._IntegrationTask__getPoni()
    output = io.StringIO()
    poni.write(output)
    assert "poni_version: 2.1" in output.getvalue()
    assert "Parallax:" not in output.getvalue()


def test_gui_patch_is_idempotent_and_version_limited(monkeypatch):
    apply_calibration_backports()
    method = IntegrationTask._IntegrationTask__getPoni
    apply_calibration_backports()
    assert IntegrationTask._IntegrationTask__getPoni is method
    monkeypatch.setattr(pyFAI, "version", "2026.9.0")
    assert apply_calibration_backports() is False
    assert IntegrationTask._IntegrationTask__getPoni is method
