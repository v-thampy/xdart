"""PROCESSING-ENERGY-BATCH-20261010 (Claude implements, Codex reviews): the frozen behavioral rows.

Maintainer, October 10: "For energies, the rule should be 0-150 read are keV, above 150 in eV. This accounts for soft
xrays as well". October 9 ("Yes"): an impossible recorded energy (zero, negative or not a number) counts as none in
processing too, and a single glitched frame in an otherwise valid series takes the series' energy. October 10 ("Yes
and yes"): row energies (SPEC, txt, Bluesky) are read by size, and energy scans ("What if there is an energy scan?")
are reduced: 2θ never needs one energy; in q each frame is reduced at its own energy on a common grid.
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.core.energy import (
    energy_eV_to_wavelength_m,
    metadata_energy_values,
    resolve_acquisition_energy,
    xray_energy_eV,
    xray_energy_keV,
)
from xrd_tools.core.provenance import read_provenance
from xrd_tools.io import open_scan
from tests.core.test_acquisition_energy import _native_energy_request, _record_nexus_energy, _replay
from tests.core.test_int_workflow_replay import _fixture

_CALIBRATION_M = 1e-10  # the PONI's wavelength: 12398.4 eV
_SHAPE = (195, 487)


# ---------------------------------------------------------------- the size rule


@pytest.mark.parametrize("recorded,eV", [
    (0.25, 250.), (1., 1000.), (8., 8000.), (16., 16000.), (149.5, 149500.), (150., 150000.),
    (150.5, 150.5), (151., 151.), (284., 284.), (500., 500.), (999., 999.), (1000., 1000.), (16000., 16000.),
])
def test_an_energy_up_to_150_is_kev_and_above_150_is_ev(recorded, eV):
    assert xray_energy_eV(recorded) == pytest.approx(eV, rel=1e-15, abs=0)
    assert xray_energy_keV(recorded) == pytest.approx(eV / 1000., rel=1e-15, abs=0)
    np.testing.assert_allclose(xray_energy_eV(np.array([recorded, 16.])), [eV, 16000.], rtol=1e-15, atol=0)
    np.testing.assert_allclose(xray_energy_keV(np.array([recorded, 16000.])), [eV / 1000., 16.], rtol=1e-15, atol=0)


def test_a_value_already_on_the_requested_scale_is_returned_exactly():
    assert xray_energy_eV(16000.123) == 16000.123 and xray_energy_keV(16.123) == 16.123
    assert xray_energy_eV(150.000001) == 150.000001 and xray_energy_keV(150.) == 150.


@pytest.mark.parametrize("value", [0., -1., -2165.0213, float("nan"), float("inf"), -float("inf")])
def test_the_size_rule_leaves_an_impossible_value_to_its_consumer(value):
    for convert in (xray_energy_eV, xray_energy_keV):
        result = convert(value)
        assert math.isnan(result) if math.isnan(value) else result == value


@pytest.mark.parametrize("row,eV", [
    ({"energy": 16}, 16000.), ({"Energy": 16000.}, 16000.), ({"ENERGY": 0.5}, 500.), ({"energy": 500.}, 500.),
    ({"energy": 150.}, 150000.), ({"energy": 151.}, 151.), ({"energy": np.float32(16.)}, 16000.),
])
def test_a_rows_energy_is_ev_or_kev_by_its_size(row, eV):
    assert metadata_energy_values(row) == pytest.approx((eV,), rel=1e-15, abs=0)


# ---------------------------------------------------------------- impossible energies and glitched frames


@pytest.mark.parametrize("recorded", [0., -1., -2165.0213, float("nan"), float("inf"), None, "n/a", True])
def test_an_impossible_recorded_energy_counts_as_none(recorded):
    for values in ((recorded,), (recorded, recorded)):
        selection = resolve_acquisition_energy(_CALIBRATION_M, values)
        assert selection["source"] == "calibration"
        assert selection["wavelength_m"] == _CALIBRATION_M


@pytest.mark.parametrize("recorded", [(16000., -1.), (float("nan"), 16000.), (16000., 0., 16000.4)])
def test_a_glitched_frame_takes_the_series_energy(recorded):
    selection = resolve_acquisition_energy(_CALIBRATION_M, recorded)
    assert (selection["source"], selection["energy_eV"]) == ("metadata", 16000.)
    assert selection["wavelength_m"] == energy_eV_to_wavelength_m(16000.)
    assert "frame_energies_eV" not in selection


def test_a_varying_series_still_refuses_where_one_energy_is_required():
    """Average and Stitch add or stitch frames into one pattern; they keep one energy."""
    with pytest.raises(ValueError, match="metadata energy must be common across selected frames"):
        resolve_acquisition_energy(_CALIBRATION_M, (16000., 17000.))


def test_an_energy_scan_resolves_each_frames_own_energy():
    """A glitched frame in a scan takes the nearest earlier frame's energy (the first valid one when it leads)."""
    selection = resolve_acquisition_energy(_CALIBRATION_M, (0., 16000., -1., 17000., float("nan")),
                                           allow_energy_scan=True)
    assert selection["frame_energies_eV"] == (16000., 16000., 16000., 17000., 17000.)
    assert (selection["source"], selection["energy_eV"]) == ("metadata", 16000.)
    assert selection["wavelength_m"] == energy_eV_to_wavelength_m(16000.)
    common = resolve_acquisition_energy(_CALIBRATION_M, (16000., 16000.4), allow_energy_scan=True)
    assert "frame_energies_eV" not in common and common["energy_eV"] == 16000.
    assert resolve_acquisition_energy(_CALIBRATION_M, (0., float("nan")), allow_energy_scan=True)[
        "source"] == "calibration"


# ---------------------------------------------------------------- Int (the replay notebook shares Run's binder)


def _identical_frames(intent):
    """Both frames the same non-uniform image, so only each frame's energy can tell their profiles apart."""
    rows, columns = np.mgrid[:_SHAPE[0], :_SHAPE[1]]
    with h5py.File(intent.source_spec.uri, "r+") as handle:
        handle["entry/instrument/detector/data"][...] = (100 + (rows + 2 * columns) % 97).astype(np.uint16)


def _int(root, energies, unit, mode="Int 1D", radial_range=None, gi=None):
    intent = _fixture(root)
    intent.processing_mode = mode
    for args in (intent.bai_1d_args, intent.bai_2d_args):
        args["unit"] = unit
        if radial_range is not None:
            args["radial_range"] = tuple(radial_range)
    if gi is not None:
        intent.gi = gi
    _identical_frames(intent)
    _record_nexus_energy(intent, energies)
    return intent


def _edges(axis):
    """The integration range whose bin centres are *axis* (uniform bins)."""
    axis = np.asarray(axis, dtype=float)
    half = (axis[-1] - axis[0]) / (len(axis) - 1) / 2
    return float(axis[0] - half), float(axis[-1] + half)


def _q_grid(poni_file, low_eV, high_eV):
    """The declared common grid, computed directly with pyFAI: from the lowest energy's lowest q to the highest
    energy's highest q over every pixel's extent (centre ± delta), never below 0."""
    from xrd_tools.integrate.calibration import load_detector_calibration, detector_calibration_to_integrator
    extents = []
    for energy in (low_eV, high_eV):
        integrator = detector_calibration_to_integrator(load_detector_calibration(poni_file))
        integrator.wavelength = energy_eV_to_wavelength_m(energy)
        centre = integrator.array_from_unit(_SHAPE, "center", "q_A^-1", scale=True)
        delta = integrator.array_from_unit(_SHAPE, "delta", "q_A^-1", scale=True)
        extents.append((float(np.min(centre - delta)), float(np.max(centre + delta))))
    return max(0.0, extents[0][0]), extents[1][1]


@pytest.mark.parametrize("recorded", [[16000., 17000.], [16., 17.]])
def test_int_reduces_an_energy_scan_in_q_at_each_frames_energy_on_one_grid(tmp_path, recorded):
    intent = _int(tmp_path / "scan", recorded, "q_A^-1")
    path = _replay(intent, "scan")
    selection = read_provenance(path)["config"]["energy_selection"]
    assert selection["frame_labels"] == [0, 1]
    assert selection["frame_energies_eV"] == pytest.approx([16000., 17000.], rel=1e-15, abs=0)
    assert selection["energy_eV"] == pytest.approx(16000., rel=1e-15, abs=0)
    scan = open_scan(path)
    q0, q1 = scan.get_1d(0).q, scan.get_1d(1).q
    np.testing.assert_array_equal(q0, q1)
    assert not np.allclose(scan.get_1d(0).intensity, scan.get_1d(1).intensity)
    grid = _q_grid(intent.poni_file, 16000., 17000.)
    for label, energy in ((0, 16000.), (1, 17000.)):
        # Each frame equals an ordinary one-energy run at its own energy on the declared common grid.
        reference = open_scan(_replay(_int(tmp_path / f"reference-{label}", [energy, energy], "q_A^-1",
                                           radial_range=grid), "reference"))
        np.testing.assert_array_equal(reference.get_1d(0).q, q0)
        np.testing.assert_allclose(scan.get_1d(label).intensity, reference.get_1d(0).intensity, rtol=1e-12, atol=0)


def test_int_reduces_a_2d_energy_scan_in_q_on_one_grid(tmp_path):
    intent = _int(tmp_path / "scan", [16000., 17000.], "q_A^-1", mode="Int 2D")
    path = _replay(intent, "scan")
    scan = open_scan(path)
    first, second = scan.get_2d(0), scan.get_2d(1)
    np.testing.assert_array_equal(first.q, second.q)
    np.testing.assert_array_equal(first.chi, second.chi)
    reference = open_scan(_replay(_int(tmp_path / "reference", [17000., 17000.], "q_A^-1", mode="Int 2D",
                                       radial_range=_q_grid(intent.poni_file, 16000., 17000.)), "reference"))
    np.testing.assert_array_equal(reference.get_2d(0).q, first.q)
    np.testing.assert_allclose(second.intensity, reference.get_2d(0).intensity, rtol=1e-12, atol=0, equal_nan=True)


def test_an_energy_scan_in_two_theta_needs_no_common_energy(tmp_path):
    scan = open_scan(_replay(_int(tmp_path / "scan", [16000., 17000.], "2th_deg"), "scan"))
    reference = open_scan(_replay(_int(tmp_path / "reference", [16000., 16000.], "2th_deg"), "reference"))
    for label in (0, 1):
        np.testing.assert_array_equal(scan.get_1d(label).q, reference.get_1d(label).q)
        np.testing.assert_array_equal(scan.get_1d(label).intensity, reference.get_1d(label).intensity)


def test_a_gi_energy_scan_in_q_shares_one_grid(tmp_path):
    from xrd_tools.session.run_configuration import GIIntent
    gi = GIIntent(enabled=True, incidence_motor="Manual", th_val=0.3)
    scan = open_scan(_replay(_int(tmp_path / "scan", [16000., 17000.], "q_A^-1", gi=gi), "scan"))
    np.testing.assert_array_equal(scan.get_1d(0).q, scan.get_1d(1).q)
    assert not np.allclose(scan.get_1d(0).intensity, scan.get_1d(1).intensity, equal_nan=True)
    # The common grid spans each frame's own one-energy grid.
    low, high = _edges(scan.get_1d(0).q)
    for energy in (16000., 17000.):
        reference = open_scan(_replay(_int(tmp_path / f"reference-{energy:g}", [energy, energy], "q_A^-1", gi=gi),
                                      "reference"))
        own_low, own_high = _edges(reference.get_1d(0).q)
        assert low <= own_low + 1e-6 and high >= own_high - 1e-6


def test_reintegrate_names_a_gi_energy_scan_it_does_not_repeat():
    """A GI energy scan's common grid comes from its Run's scouts, which Reintegrate does not repeat yet. (Added after
    implementation: a GI Run output is refused earlier today by the open GI-REINTEGRATE-2D-REFUSED row, so the guard
    is reached directly.)"""
    from xrd_tools.reduction.reintegrate import _prepare_gi_scouts
    shared = {"gi": {"enabled": True, "resolved_motor": "Manual"},
              "energy_selection": {"frame_labels": [0, 1], "frame_energies_eV": [16000., 17000.]}}
    with pytest.raises(ValueError, match="does not yet repeat a GI energy scan"):
        _prepare_gi_scouts(None, "entry", None, None, shared, None, None)


def test_int_replay_glitched_frame_takes_the_series_energy(tmp_path):
    """Superseded the [16000, -1] refusal row of test_int_replay_invalid_or_varying_metadata_energy_refuses."""
    glitched = _replay(_int(tmp_path / "glitched", [16000., -1.], "q_A^-1"), "run")
    steady = open_scan(_replay(_int(tmp_path / "steady", [16000., 16000.], "q_A^-1"), "run"))
    selection = read_provenance(glitched)["config"]["energy_selection"]
    assert (selection["source"], selection["energy_eV"]) == ("metadata", 16000.)
    assert "frame_energies_eV" not in selection
    for label in (0, 1):
        np.testing.assert_array_equal(open_scan(glitched).get_1d(label).intensity, steady.get_1d(label).intensity)


def test_an_energy_scan_with_sensor_parallax_refuses_by_name(tmp_path):
    """Parallax is computed at one energy, so an energy scan cannot take it on."""
    from xrd_tools.integrate.calibration import apply_sensor_parallax, load_detector_calibration
    from xrd_tools.session.readiness import bind_native_int_source_energy
    from xrd_tools.sources import open_source
    intent = _int(tmp_path / "scan", [16000., 17000.], "q_A^-1")
    calibration = apply_sensor_parallax(load_detector_calibration(intent.poni_file),
                                        material="Si", thickness_m=450e-6, parallax=True)
    source = open_source(intent.source_spec)
    try:
        with pytest.raises(ValueError, match="parallax"):
            bind_native_int_source_energy(source.to_scan(), calibration)
    finally:
        source.close()


# ---------------------------------------------------------------- Average and Stitch keep one energy


def test_average_glitched_frame_takes_the_series_energy(tmp_path):
    """Superseded the -1 rows of test_average_refuses_invalid_or_mixed_energy_before_replacing_output."""
    from tests.core.test_average_energy import _average
    intent = _fixture(tmp_path / "project")
    intent.processing_mode = "Int 1D"
    _record_nexus_energy(intent, [16000., -1.])
    result, _plan = _average(intent, tmp_path / "average.nexus")
    assert result.disposition == "COMMITTED", result
    selection = read_provenance(result.target)["config"]["energy_selection"]
    assert (selection["source"], selection["energy_eV"]) == ("metadata", 16000.)


def test_average_reads_txt_energies_by_size(tmp_path):
    from tests.core.test_average_energy import _average
    from tests.core.test_p36_average import _series, _txt
    from xrd_tools.core.scan import SourceSpec
    intent = _fixture(tmp_path / "project")
    intent.processing_mode = "Int 1D"
    source = _series(Path(intent.project_root), (np.full(_SHAPE, 100, np.uint16),) * 2)
    intent.source_spec = SourceSpec(source.uri, source.kind, options={**source.options, "metadata_format": "txt"})
    for file in source.options["files"]:
        _txt(file, motors=(("energy", 16.),))
    result, _plan = _average(intent, tmp_path / "average.nexus")
    assert result.disposition == "COMMITTED", result
    assert read_provenance(result.target)["config"]["energy_selection"]["energy_eV"] == pytest.approx(16000.)


@pytest.mark.parametrize("recorded,source,energy", [(16., "metadata", 16000.), (0., "calibration", 12398.41984),
                                                    (-2165.0213, "calibration", 12398.41984)])
def test_native_stitch_reads_txt_energy_by_size_and_ignores_an_impossible_one(tmp_path, recorded, source, energy):
    request = _native_energy_request(tmp_path, "q_A^-1", recorded)
    selection = request.provenance["plan"]["energy_selection"]
    assert (selection["source"], selection["energy_eV"]) == (source, pytest.approx(energy))


# ---------------------------------------------------------------- calibration and Bluesky readers


def _spec_with_energy(root, energy):
    import tifffile
    root.mkdir(parents=True, exist_ok=True)
    spec = root / "scan.spec"
    spec.write_text(f"#F scan.spec\n#O0 energy\n#S 1 ascan del 0 1 1 1\n#P0 {energy}\n#N 3\n#L del  nu  I0\n"
                    "0 0 1\n1 0 1\n")
    for index in range(2):
        tifffile.imwrite(root / f"scan_scan1_{index:04d}.tif", np.full(_SHAPE, 100, np.uint16))
    from xrd_tools.core.scan import SourceKind, SourceSpec
    return SourceSpec(spec, SourceKind.SPEC, options={"scan": "1.1", "image_dir": str(root)})


@pytest.mark.parametrize("recorded", [16, 16000])
def test_calibration_reads_a_rows_energy_by_its_size(tmp_path, recorded):
    from xrd_tools.integrate.stitch_calibration import calibration_source_energy_eV, source_frame_choices
    spec = _spec_with_energy(tmp_path / "data", recorded)
    assert calibration_source_energy_eV(spec) == pytest.approx(16000., rel=1e-15, abs=0)
    assert [energy for _index, _motors, energy in source_frame_choices(spec)] == pytest.approx([16000., 16000.])


@pytest.mark.parametrize("recorded", [16., 16000.])
def test_a_bluesky_tiled_energy_is_kev_by_its_size(recorded):
    from xrd_tools.io.tiled import _extract_energy
    assert _extract_energy(None, {"energy": recorded}, "primary") == pytest.approx(16., rel=1e-15, abs=0)
