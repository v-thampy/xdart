"""Upstream PR2905 serialization and xdart's normal PONI readback boundary."""
from __future__ import annotations

import io
import json
import tomllib
from pathlib import Path

import numpy as np
import pyFAI
import pytest
from pyFAI import units
from pyFAI.detectors import detector_factory
from pyFAI.detectors._rayonix import _Rayonix
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.io._json import PyFAIEncoder
from pyFAI.io.ponifile import PoniFile


_SENSOR = {"material": "Gd2O2S", "thickness": 40e-6}
_CONFIG = {
    "pixel1": 73.242e-6,
    "pixel2": 73.242e-6,
    "orientation": 3,
    "sensor": _SENSOR,
}


@pytest.fixture(autouse=True)
def restore_upstream_methods(monkeypatch):
    # Applying the backport is process-local. Do not affect later test files.
    monkeypatch.setattr(_Rayonix, "get_config", _Rayonix.get_config)
    monkeypatch.setattr(PyFAIEncoder, "default", PyFAIEncoder.default)


def _install():
    from xrd_tools.integrate._pyfai_rayonix import (
        apply_rayonix_serialization_backport,
    )

    return apply_rayonix_serialization_backport()


def _poni_text(parallax):
    return (
        "poni_version: 3.0\n"
        "Detector: RayonixMx225\n"
        f"Detector_config: {json.dumps(_CONFIG)}\n"
        "Distance: 0.1\nPoni1: 0.05\nPoni2: 0.06\n"
        "Rot1: 0.01\nRot2: 0.02\nRot3: 0.03\n"
        f"Wavelength: 1e-10\nParallax: {parallax}\n"
    )


@pytest.mark.parametrize("parallax", [False, True])
def test_rayonix_poni_v3_normal_load_and_reconstruction(tmp_path, parallax):
    """No explicit installer: the ordinary application readback must work.

    This row also runs against the exact parent source for fail-before evidence.
    """
    from xrd_tools.integrate.calibration import (
        detector_calibration_to_integrator,
        load_detector_calibration,
    )

    path = tmp_path / "rayonix.poni"
    path.write_text(_poni_text(parallax), encoding="utf-8")
    calibration = load_detector_calibration(path)
    assert calibration.parallax is parallax
    assert dict(calibration.detector_config) == _CONFIG
    assert calibration.poni.dist == 0.1
    assert calibration.poni.rot2 == 0.02
    integrator = detector_calibration_to_integrator(calibration)
    assert integrator.detector.get_config() == _CONFIG
    assert (integrator.parallax is not None) is parallax


def test_exact_upstream_rayonix_save_and_encoder_behavior():
    detector = detector_factory("RayonixMx225", config=_CONFIG)
    pixels_before = detector.pixel1, detector.pixel2, detector.orientation
    eiger = detector_factory("Eiger4M")
    eiger_config = eiger.get_config()
    _install()
    assert detector.get_config() == _CONFIG
    assert json.loads(json.dumps(detector.get_config())) == _CONFIG
    assert (detector.pixel1, detector.pixel2, detector.orientation) == pixels_before
    assert eiger.get_config() == eiger_config
    assert json.loads(json.dumps(detector.sensor, cls=PyFAIEncoder)) == _SENSOR
    assert json.loads(json.dumps(np.float64(2.5), cls=PyFAIEncoder)) == 2.5
    assert json.loads(json.dumps(np.int64(7), cls=PyFAIEncoder)) == 7
    assert json.loads(json.dumps(units.Q_A, cls=PyFAIEncoder)) == units.Q_A.name
    with pytest.raises(TypeError):
        json.dumps(object(), cls=PyFAIEncoder)
    integrator = AzimuthalIntegrator(
        dist=0.1, poni1=0.05, poni2=0.06, wavelength=1e-10,
        detector=detector,
    )
    output = io.StringIO()
    PoniFile(integrator).write(output)
    assert '"sensor"' in output.getvalue()


def test_rayonix_without_sensor_and_idempotent_install():
    detector = detector_factory("RayonixMx225")
    detector.sensor = None
    before = detector.get_config()
    _install()
    methods = _Rayonix.get_config, PyFAIEncoder.default
    _install()
    assert (_Rayonix.get_config, PyFAIEncoder.default) == methods
    assert detector.get_config() == before
    assert "sensor" not in detector.get_config()


@pytest.mark.parametrize("version", ["2026.5.1", "2026.9.0"])
def test_other_releases_are_not_patched(monkeypatch, version):
    before = _Rayonix.get_config, PyFAIEncoder.default
    monkeypatch.setattr(pyFAI, "version", version)
    assert _install() is False
    assert (_Rayonix.get_config, PyFAIEncoder.default) == before


def test_legacy_sensor_schema_is_not_silently_reinterpreted(tmp_path):
    from xrd_tools.integrate.calibration import load_detector_calibration

    payload = _poni_text(False).replace("poni_version: 3.0", "poni_version: 2.1")
    payload = payload.replace("Parallax: False\n", "")
    path = tmp_path / "invalid_sensor_schema.poni"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="PONI 2 does not support sensor"):
        load_detector_calibration(path)


def test_calibration_entry_point_is_part_of_the_normal_package():
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    document = tomllib.loads(project.read_text(encoding="utf-8"))
    assert document["project"]["scripts"]["xdart-calib2"] == "xdart.calib2_main:main"
