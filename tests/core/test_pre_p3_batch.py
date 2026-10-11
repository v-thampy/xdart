"""PRE-P3-BATCH-20261010: a bare burst master in Stitch/RSM, and the Int energy-scan refusals.

Maintainer, October 10: "A bare Eiger master will never be the only source in RSM and stitching. Though it can be in
int of course with a poni file"; on an energy scan with parallax, "there should be a notification for that with
advice to turn it off to allow processing"; and to "Should Int refuse an energy scan too long to save (over about
11,000 frames) before it runs?", "Yes".
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.core.test_source_edge_contract import _dangling_canonical, _land_canonical

_PONI = ("poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\nDistance: 0.1234\nPoni1: 0.05\n"
         "Poni2: 0.06\nRot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n")


def _burst_master(root: Path) -> Path:
    """A detector-only DECTRIS master whose one trigger records three images, so grouping would make one point."""
    master, data = _dangling_canonical(root)
    _land_canonical(data)
    with h5py.File(master, "a") as stream:
        specific = stream.require_group("entry/instrument/detector/detectorSpecific")
        specific.create_dataset("nimages", data=np.int64(3))
        specific.create_dataset("ntrigger", data=np.int64(1))
    return master


# ---------------------------------------------------------------- a bare burst master in Stitch and RSM


def test_a_detector_only_burst_master_is_refused_naming_its_metadata_source(tmp_path):
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection, resolve_stitch_sources
    from xrd_tools.sources.point_images import PointImagesRequired
    master = _burst_master(tmp_path)
    with pytest.raises(ValueError) as error:
        resolve_stitch_sources(StitchSourceSelection("NeXus", (str(master),)), {})
    assert not isinstance(error.value, PointImagesRequired)  # no question about grouping its images
    message = str(error.value)
    assert master.name in message and "metadata source" in message and "SPEC or NeXus" in message


def test_a_confirmed_grouping_does_not_admit_a_detector_only_burst_master(tmp_path):
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection, resolve_stitch_sources
    from xrd_tools.sources.point_images import PointImagePolicy
    master = _burst_master(tmp_path)
    choice = (str(master), "entry", PointImagePolicy(3, "average"))
    with pytest.raises(ValueError, match="metadata source"):
        resolve_stitch_sources(StitchSourceSelection("NeXus", (str(master),), point_image_choices=(choice,)), {})


def test_directory_mode_skips_a_detector_only_burst_master_and_keeps_its_spec_scans(tmp_path):
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection, resolve_stitch_sources
    from tests.core.test_spec_source import _spec
    _spec(tmp_path)
    for index in range(3):
        np.ones((2, 2), dtype=np.uint16).tofile(tmp_path / f"myscan_scan5_{index}.raw")
    master = _burst_master(tmp_path)
    resolved = resolve_stitch_sources(StitchSourceSelection("Directory", (str(tmp_path),)),
                                      {"detector_shape": (2, 2), "raw_dtype": "uint16"})
    assert resolved.pending == ()  # the folder's SPEC scans are not held behind a grouping question
    assert [member.options["scan"] for group in resolved.groups for member in group] == ["5.1"]
    skipped = {Path(path).name: (entry, reason) for path, entry, reason in resolved.skipped}
    entry, reason = skipped[master.name]
    assert entry == "entry" and "metadata source" in reason and "SPEC or NeXus" in reason


def test_the_real_detector_only_burst_master_is_refused_in_stitch():
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection, resolve_stitch_sources
    from xrd_tools.sources.point_images import PointImagesRequired
    master = Path(os.environ.get("XDART_TEST_DATA", "")) / "nexus" / "Pt_test_burst_00006_master.h5"
    if not master.is_file():
        pytest.skip("no real burst master in XDART_TEST_DATA")
    with pytest.raises(ValueError) as error:  # one trigger, 1000 images
        resolve_stitch_sources(StitchSourceSelection("NeXus", (str(master),)), {})
    assert not isinstance(error.value, PointImagesRequired)
    assert master.name in str(error.value) and "metadata source" in str(error.value)


# ---------------------------------------------------------------- energy scans Int refuses at run start


def _calibration(root: Path):
    from xrd_tools.integrate.calibration import load_detector_calibration
    poni = root / "cal.poni"
    poni.write_text(_PONI)
    return load_detector_calibration(poni)


def _scan_of(energies_keV):
    from xrd_tools.core.scan import Scan, ScanFrame
    return Scan("escan", [ScanFrame(index, metadata={"energy": float(value)})
                          for index, value in enumerate(energies_keV)])


def test_config_entry_overflow_is_the_writers_own_ceiling(tmp_path):
    """The binder measures an energy selection the way the writer stores it, so both refuse the same payloads."""
    from xrd_tools.core.provenance import _write_json, config_entry_overflow
    from xrd_tools.io.append import _MAX_REPLACEMENT_CONFIG_UTF8_BYTES as ceiling
    fits = "x" * (ceiling - 2)  # a JSON string adds its two quotes
    over = fits + "x"
    assert config_entry_overflow("energy_selection", fits) is None
    assert config_entry_overflow("energy_selection", over) == (ceiling + 1, ceiling)
    with h5py.File(tmp_path / "config.h5", "w") as stream:
        _write_json(stream, "energy_selection", fits)
        with pytest.raises(ValueError, match="exceeds the persisted UTF-8 byte ceiling"):
            _write_json(stream, "longer", over)
        assert "longer" not in stream


def test_an_energy_scan_too_long_to_save_is_refused_when_it_is_bound(tmp_path):
    """12,000 full-precision energies need about 275 KB of saved settings, over the 256 KiB the writer accepts; the
    binder refuses before any frame is reduced. 1000 frames bind."""
    from xrd_tools.session.readiness import EnergyScanRefused, bind_native_int_source_energy
    calibration = _calibration(tmp_path)
    long_scan = _scan_of(np.linspace(8.0, 9.0, 12000))
    with pytest.raises(EnergyScanRefused) as error:
        bind_native_int_source_energy(long_scan, calibration)
    message = str(error.value)
    assert "12,000 frames" in message and "262,144" in message
    assert "energy_selection" not in long_scan.extra
    short_scan = _scan_of(np.linspace(8.0, 9.0, 1000))
    bind_native_int_source_energy(short_scan, calibration)
    assert len(short_scan.extra["energy_selection"]["frame_energies_eV"]) == 1000


def test_the_parallax_refusal_is_an_energy_scan_refusal_naming_the_control(tmp_path):
    from xrd_tools.integrate.calibration import apply_sensor_parallax
    from xrd_tools.session.readiness import EnergyScanRefused, bind_native_int_source_energy
    calibration = apply_sensor_parallax(_calibration(tmp_path), material="Si", thickness_m=450e-6, parallax=True)
    with pytest.raises(EnergyScanRefused) as error:
        bind_native_int_source_energy(_scan_of([16.0, 17.0]), calibration)
    message = str(error.value)
    assert "parallax" in message and "Detector sensor / parallax" in message and "Enable parallax" in message
