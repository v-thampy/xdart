"""PROCESSING-ENERGY-BATCH-20261010: an energy scan through the production Int executor, then Reintegrate.

Maintainer, October 10 ("Yes and yes"): energy scans are reduced (q at each frame's own energy on a common grid), and a
row's energy is read as eV or keV by its size ("0-150 read are keV, above 150 in eV"). The txt rows record 16 and 17
(keV); identical images make each frame's energy the only difference between their profiles.
"""
from __future__ import annotations

from pathlib import Path

import fabio
import numpy as np
import pytest

from tests.xdart.scattering._output_slots import written
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _intent, _run_to_terminal, write_poni
from xdart.gui.tabs.scattering.adapters.run_executor import StandardEventKind


def _series(root: Path, energies_keV) -> Path:
    rows, columns = np.mgrid[:195, :487]
    image = (100 + (rows + 2 * columns) % 97).astype(np.uint16)
    for index, energy in enumerate(energies_keV, 1):
        raw = root / f"escan_{index:04d}.tif"
        fabio.tifimage.TifImage(data=image).write(str(raw))
        raw.with_suffix(".txt").write_text(f"# Counters\nI0 = 1\n# Motors\nenergy = {energy}\n"
                                           "User: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n")
    return root / "escan_0001.tif"


def _run(root: Path, energies_keV, unit: str, radial_range=None):
    from xrd_tools.sources import image_series_spec
    root.mkdir(parents=True, exist_ok=True)
    raw = _series(root, energies_keV)
    poni = root / "cal.poni"
    write_poni(poni)
    intent = _intent(raw, root / "escan.nexus", poni)
    intent.source_spec = image_series_spec(raw, metadata_format="txt")
    intent.bai_1d_args["unit"] = unit
    if radial_range is not None:
        intent.bai_1d_args["radial_range"] = tuple(radial_range)
    executor, identity, events = _run_to_terminal(intent, request_value=9131)
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert terminal.kind is StandardEventKind.FINISHED, terminal.primary
    executor.close(identity)
    return intent, written(root / "escan.nexus", "Int 1D")


def _edges(axis):
    axis = np.asarray(axis, dtype=float)
    half = (axis[-1] - axis[0]) / (len(axis) - 1) / 2
    return float(axis[0] - half), float(axis[-1] + half)


def test_a_real_energy_scan_run_reduces_each_frame_at_its_energy_and_reintegrates_the_same(tmp_path):
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.io import open_scan
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor
    intent, artifact = _run(tmp_path / "scan", (16, 17), "q_A^-1")
    selection = read_provenance(artifact)["config"]["energy_selection"]
    scan = open_scan(artifact)
    labels = [int(label) for label in scan.frame_indices]
    assert selection["frame_labels"] == labels
    assert selection["frame_energies_eV"] == pytest.approx([16000., 17000.], rel=1e-15, abs=0)
    first, second = (scan.get_1d(label) for label in labels)
    np.testing.assert_array_equal(first.q, second.q)
    assert not np.allclose(first.intensity, second.intensity)
    low, high = _edges(first.q)
    for label, energy in zip(labels, (16, 17)):
        _intent_unused, reference = _run(tmp_path / f"reference-{energy}", (energy, energy), "q_A^-1",
                                         radial_range=(low, high))
        expected = open_scan(reference).get_1d(int(open_scan(reference).frame_indices[0]))
        np.testing.assert_allclose(expected.q, first.q, rtol=1e-12, atol=0)
        np.testing.assert_allclose(scan.get_1d(label).intensity, expected.intensity, rtol=1e-12, atol=0)
    # Reintegrate repeats each frame at its persisted energy; no current source-energy inference.
    plan = ReintegrateSuccessorPlan.from_artifact(
        artifact, entry="entry", dimension="1d", source_root=str(tmp_path / "scan"),
        expected_target_snapshot=capture_target_snapshot(artifact), expected_labels=tuple(labels),
        preparation=ScatteringWorkspace._reintegrate_preparation(intent, "1d"))
    result = run_reintegrate_successor(plan)
    assert result.disposition == "COMMITTED", result
    output = open_scan(Path(result.output_artifact))
    for label, before in zip(labels, (first, second)):
        np.testing.assert_array_equal(output.get_1d(label).q, before.q)
        np.testing.assert_array_equal(output.get_1d(label).intensity, before.intensity)
    assert read_provenance(Path(result.output_artifact))["config"]["energy_selection"] == selection


def test_a_real_energy_scan_run_in_two_theta_matches_a_one_energy_run(tmp_path):
    from xrd_tools.io import open_scan
    _intent_scan, scan_path = _run(tmp_path / "scan", (16, 17), "2th_deg")
    _intent_reference, reference_path = _run(tmp_path / "reference", (16, 16), "2th_deg")
    scan, reference = open_scan(scan_path), open_scan(reference_path)
    for label, other in zip(scan.frame_indices, reference.frame_indices):
        np.testing.assert_array_equal(scan.get_1d(int(label)).q, reference.get_1d(int(other)).q)
        np.testing.assert_array_equal(scan.get_1d(int(label)).intensity, reference.get_1d(int(other)).intensity)
