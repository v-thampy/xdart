"""Supported generic acquisition units must reach Int's real energy binder."""

import h5py
import numpy as np
import pytest

from xrd_tools.integrate.calibration import load_detector_calibration
from xrd_tools.session.readiness import bind_native_int_source_energy
from xrd_tools.sources.descriptor import describe_container
from xrd_tools.sources.nexus import NexusStackSource


@pytest.mark.parametrize("field,value,units", [
    ("energy", 16000.0, "eV"),
    ("wavelength", 1.239841984e-6 / 16000.0, "m"),
])
def test_supported_generic_units_agree_at_int_binding(tmp_path, field, value, units):
    path = tmp_path / "acquisition.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("instrument/detector/data", data=np.ones((2, 195, 487), np.uint16))
        entry.create_dataset("instrument/positioners/eta/value", data=[0.11, 0.37])
        recorded = entry.create_dataset(f"instrument/monochromator/{field}", data=value)
        recorded.attrs["units"] = units
    poni = tmp_path / "cal.poni"
    poni.write_text(
        "poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\n"
        "Distance: 0.1234\nPoni1: 0.05\nPoni2: 0.06\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n"
    )
    descriptor = describe_container(path)
    source = NexusStackSource(path)
    try:
        scan = source.to_scan(output_path=tmp_path / "result.nexus")
        selected = bind_native_int_source_energy(
            scan, load_detector_calibration(poni), metadata_wavelength_A=descriptor.wavelength,
            metadata_energy_eV=source.motors.get("energy", ()),
        )
        assert selected.poni.wavelength == pytest.approx(1.239841984e-6 / 16000.0, rel=1e-12)
        assert [frame.metadata["eta"] for frame in scan.frames] == pytest.approx([0.11, 0.37])
    finally:
        source.close()


@pytest.mark.parametrize("field,value,units", [
    ("energy", 16000., "eV"), ("energy", 16., "keV"),
    ("wavelength", 1.239841984e4 / 16000., "angstrom"),
    ("wavelength", 1.239841984e-6 / 16000., "m"),
])
def test_generic_descriptor_uses_declared_scalar_units_only(tmp_path, field, value, units):
    path = tmp_path / "scalar.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("instrument/detector/data", data=np.zeros((2, 8, 8), np.uint16))
        mono = entry.create_dataset(f"instrument/monochromator/{field}", data=value)
        mono.attrs["units"] = units
    assert describe_container(path).wavelength == pytest.approx(1.239841984e4 / 16000., rel=1e-12)
    with h5py.File(path, "r+") as handle:
        entry = handle["entry"]
        del entry[f"instrument/monochromator/{field}"]
        mono = entry.create_dataset(f"instrument/monochromator/{field}", data=[value, value * 1.01])
        mono.attrs["units"] = units
    assert describe_container(path).wavelength is None
