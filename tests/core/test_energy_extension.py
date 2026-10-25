"""A grown scan's energy from its output's saved selection (extend_acquisition_energy) is exactly the selection
resolve_acquisition_energy gives the whole scan, or None when the earlier frames' own values are needed."""
from __future__ import annotations

import json

import pytest

from xrd_tools.core.energy import extend_acquisition_energy, resolve_acquisition_energy

WAVELENGTH_M = 1.0e-10


def _resolved(values):
    """As bind_native_int_source_energy resolves a TIFF scan's recorded energies (eV), then saves it (JSON)."""
    per_frame = any(value is not None for value in values)
    selection = resolve_acquisition_energy(WAVELENGTH_M, values if per_frame else [], allow_energy_scan=per_frame)
    if "frame_energies_eV" in selection:
        selection = {**selection, "frame_energies_eV": [float(value) for value in selection["frame_energies_eV"]]}
    return json.loads(json.dumps(selection))


def _normalized(selection):
    return None if selection is None else json.loads(json.dumps(
        {**selection, **({"frame_energies_eV": [float(value) for value in selection["frame_energies_eV"]]}
                         if "frame_energies_eV" in selection else {})}))


@pytest.mark.parametrize("before, after", [
    ([16000.0] * 4, [16000.0, 16010.0]),            # common, later within 0.1 %
    ([None, 16000.0, 16005.0], [15995.0]),           # a glitched first frame; common
    ([None, None], [None]),                           # none recorded: the calibration's
    ([0.0, None], [None, 0.0]),                       # only impossible ones: none
    ([16000.0, 16100.0, 16200.0], [16300.0]),       # an energy scan grows
    ([16000.0, 16100.0, 16200.0], [None, 0.0, 16300.0]),  # glitched later frames carry the previous one
    ([16000.0, 16100.0], [16100.0]),                 # a scan whose new frame repeats
])
def test_settled_extensions_equal_the_whole_scans_selection(before, after):
    assert _normalized(extend_acquisition_energy(_resolved(before), after, WAVELENGTH_M)) == _resolved(before + after)


@pytest.mark.parametrize("before, after", [
    ([16000.0] * 3, [16200.0]),          # a common energy that now varies: needs each earlier frame's own value
    ([None, None], [16000.0]),           # a first valid energy only now recorded
])
def test_unsettled_extensions_need_the_earlier_frames(before, after):
    assert extend_acquisition_energy(_resolved(before), after, WAVELENGTH_M) is None


def test_another_calibration_or_a_fitted_offset_is_never_extended():
    saved = _resolved([16000.0] * 3)
    assert extend_acquisition_energy(saved, [16000.0], 1.1e-10) is None
    assert extend_acquisition_energy({**saved, "fitted_energy_offset_eV": 1.0}, [16000.0], WAVELENGTH_M) is None
