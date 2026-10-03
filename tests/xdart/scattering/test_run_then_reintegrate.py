# -*- coding: utf-8 -*-
"""What a REAL Run output carries, driven through the production executor.

NAME CORRECTED after Codex review of `cca7d537`: this was called
`..._can_be_reintegrated`, which overstated it -- the body runs the executor and
checks provenance but never calls a reintegration method. The round trip is
proved by the p34 NexusSink-seeded row, and independently by Codex on real
651-frame Eiger data through the actual page preparation builder.

This exists because Fable F3 -- the family stamp making every real Run output
un-reintegratable -- survived my own testing and two independent reviews, and it
did so for one reason: no test ever built the production predecessor shape.
Every predecessor fixture writes either an unstamped artifact or a full finite
lineage node, and a real Run output is neither.

So this drives the actual `StandardRunExecutor`, on a real TIFF series, and
hands its published artifact to `ReintegrateSuccessorPlan.from_artifact`. No
hand-built fixture stands in for the thing under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.xdart.scattering._output_slots import written
from tests.xdart.scattering.test_p1b_output_graph import (
    _TERMINAL,
    _intent,
    _run_to_terminal,
    _write_tiff,
    write_poni,
)
from xdart.gui.tabs.scattering.adapters.run_executor import StandardEventKind


@pytest.mark.parametrize('unit', ['q_A^-1', '2th_deg'])
def test_real_run_uses_metadata_energy_with_effective_persisted_calibration(tmp_path, unit):
    from xrd_tools.sources import image_series_spec
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    from xrd_tools.io import open_scan
    from xrd_tools.core.provenance import read_provenance
    raw = tmp_path / 'energy_0001.tif'
    _write_tiff(raw, 1)
    raw.with_suffix('.txt').write_text('# Counters\nI0 = 1\n# Motors\nenergy = 16000\n'
        'User: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n')
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    intent = _intent(raw, tmp_path / 'energy.nexus', poni)
    intent.source_spec = image_series_spec(raw, metadata_format='txt')
    intent.bai_1d_args['unit'] = unit
    executor, identity, events = _run_to_terminal(intent, request_value=9102)
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert terminal.kind is StandardEventKind.FINISHED, terminal.primary
    executor.close(identity)
    artifact = written(tmp_path / 'energy.nexus', 'Int 1D')
    assert open_scan(artifact).metadata['wavelength_A'] == pytest.approx(
        energy_eV_to_wavelength_m(16000.) * 1e10)
    selection = read_provenance(artifact)['config']['energy_selection']
    assert selection['source'] == 'metadata'
    assert selection['energy_eV'] == pytest.approx(16000.)

    # Use the page's real preparation builder and the successor execution owner.
    # Repeat from the persisted Run: no current source-energy inference is allowed.
    import numpy as np
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor
    before = open_scan(artifact).get_1d(1)
    axis, intensity = before.q.copy(), before.intensity.copy()
    for _ in range(2):
        labels = tuple(int(v) for v in open_scan(artifact).frame_indices)
        plan = ReintegrateSuccessorPlan.from_artifact(
            artifact, entry='entry', dimension='1d', source_root=str(tmp_path),
            expected_target_snapshot=capture_target_snapshot(artifact), expected_labels=labels,
            preparation=ScatteringWorkspace._reintegrate_preparation(intent, '1d'))
        result = run_reintegrate_successor(plan)
        assert result.disposition == 'COMMITTED', result
        output = Path(result.output_artifact)
        scan = open_scan(output)
        assert scan.metadata['wavelength_A'] == pytest.approx(selection['wavelength_m'] * 1e10)
        np.testing.assert_array_equal(scan.get_1d(1).q, axis)
        np.testing.assert_array_equal(scan.get_1d(1).intensity, intensity)
        assert read_provenance(output)['config']['energy_selection'] == selection


def test_wavelengthless_two_theta_run_reintegrates_with_admitted_calibration(tmp_path):
    import numpy as np
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.io import open_scan
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor

    raw = tmp_path / 'no-energy_0001.tif'
    _write_tiff(raw, 1)
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    poni.write_text('\n'.join(line for line in poni.read_text().splitlines()
                              if not line.startswith('Wavelength:')) + '\n')
    intent = _intent(raw, tmp_path / 'no-energy.nexus', poni)
    intent.bai_1d_args['unit'] = '2th_deg'
    executor, identity, events = _run_to_terminal(intent, request_value=9103)
    try:
        terminal = next(event for event in events if event.kind in _TERMINAL)
        assert terminal.kind is StandardEventKind.FINISHED, terminal.primary
    finally:
        executor.close(identity)

    artifact = written(tmp_path / 'no-energy.nexus', 'Int 1D')
    predecessor = open_scan(artifact)
    before = predecessor.get_1d(1)
    axis, intensity = before.q.copy(), before.intensity.copy()
    plan = ReintegrateSuccessorPlan.from_artifact(
        artifact, entry='entry', dimension='1d', source_root=str(tmp_path),
        expected_target_snapshot=capture_target_snapshot(artifact),
        expected_labels=tuple(int(label) for label in predecessor.frame_indices),
        preparation=ScatteringWorkspace._reintegrate_preparation(intent, '1d'))
    result = run_reintegrate_successor(plan)
    assert result.disposition == 'COMMITTED', result
    after = open_scan(result.output_artifact).get_1d(1)
    np.testing.assert_array_equal(after.q, axis)
    np.testing.assert_array_equal(after.intensity, intensity)


def test_a_real_run_output_carries_what_reintegrate_requires(tmp_path: Path) -> None:
    from xrd_tools.io.schema import ARTIFACT_FAMILY_ATTR
    from xrd_tools.reduction import ReintegrateSuccessorPlan

    import h5py

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = raw_root / "scan12_0001.tif"
    poni = tmp_path / "cal.poni"
    _write_tiff(raw, 1)
    write_poni(poni)

    executor, identity, events = _run_to_terminal(
        _intent(raw, tmp_path / "scan12.nexus", poni), request_value=9101,
    )
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert terminal.kind is StandardEventKind.FINISHED, terminal.primary
    executor.close(identity)

    artifact = written(tmp_path / "scan12.nexus", "Int 1D")
    assert artifact.is_file(), sorted(p.name for p in tmp_path.iterdir())

    # It really is the production shape: family stamped, no finite lineage.
    with h5py.File(artifact, "r") as document:
        assert document["entry"].attrs[ARTIFACT_FAMILY_ATTR] == "scan12"

    # A REAL Run persists a `scientific_signature` -- verified below -- so the
    # gap Codex recorded as F5 is AVERAGE-only and does not block this path.
    from xrd_tools.core.provenance import read_provenance

    provenance = read_provenance(artifact, entry="entry")
    run = provenance["config"]["run_configuration"]
    assert "scientific_signature" in run

    # WHAT THIS PROVES, and what it does not.
    #
    # PROVEN here: a real Run output carries BOTH the family stamp and a
    # persisted `scientific_signature`. The second fact is the load-bearing one.
    # `_validated_shared_science` (reintegrate.py:1979) refuses a predecessor
    # without a signature, and that refusal is what Codex recorded as F5 for
    # AVERAGE. Run persists one, so F5 is AVERAGE-ONLY and does not block the
    # Run -> Reintegrate workflow.
    #
    # NOT PROVEN here: the full `from_artifact` -> `run_reintegrate_successor`
    # round trip. It needs a canonical `preparation`, which the GUI derives from
    # a loaded browse capture (`page.py:2419`); hand-building one in a test
    # reaches "selected GI mode/BAI differs", which is the product validating
    # its inputs rather than refusing the predecessor. Driving the browse path
    # is a larger harness and its own slice.
    assert "scientific_signature" in run
    assert run["scientific_signature"]
