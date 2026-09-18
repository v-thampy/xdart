"""Automatic GI names keep their XYE curves separate from Standard output."""

import pytest

from tests.xdart.scattering.test_p1b_output_graph import (
    _TERMINAL,
    _intent,
    _run_to_terminal,
    _write_tiff,
    write_poni,
)
from xdart.gui.tabs.scattering.display_values import StandardEventKind


@pytest.mark.parametrize("processing_mode", ("Int 1D", "Int 1D (XYE)"))
@pytest.mark.parametrize("folder_name", ("processed", "2026.09.18", "beamtime.v2", "processed.nexus"))
def test_standard_and_gi_xye_families_coexist(tmp_path, processing_mode, folder_name):
    raw, poni = tmp_path / "sample_0001.tif", tmp_path / "cal.poni"
    _write_tiff(raw, 50)
    write_poni(poni)
    output = tmp_path / folder_name
    output.mkdir()
    standard_xye = output / "sample" / "iq_sample_0001.xye"
    standard_bytes = None

    for gi, family in ((False, "sample"), (True, "sample_gi")):
        intent = _intent(raw, output, poni, processing_mode=processing_mode)
        intent.gi.enabled = gi
        intent.gi.incidence_motor = "Manual"
        intent.gi.th_val = 0.2
        executor, identity, events = _run_to_terminal(
            intent, request_value=9261 + int(gi),
        )
        try:
            terminal = next(event for event in events if event.kind in _TERMINAL)
            assert terminal.kind is StandardEventKind.FINISHED, terminal.primary
            assert (output / family / f"iq_{family}_0001.xye").is_file()
            if gi:
                assert standard_xye.read_bytes() == standard_bytes
            else:
                standard_bytes = standard_xye.read_bytes()
            if processing_mode == "Int 1D":
                assert (output / f"{family}_int1d.nexus").is_file()
            frame = next(event.frame_key for event in events
                         if event.kind is StandardEventKind.FRAME_READY)
            assert frame.source_scan == "sample"
        finally:
            executor.close(identity)


def test_explicit_gi_output_family_is_used_verbatim_for_xye(tmp_path):
    raw, poni = tmp_path / "sample_0001.tif", tmp_path / "cal.poni"
    _write_tiff(raw, 50)
    write_poni(poni)
    intent = _intent(raw, tmp_path / "chosen.nexus", poni)
    intent.gi.enabled = True
    intent.gi.incidence_motor = "Manual"
    intent.gi.th_val = 0.2
    executor, identity, events = _run_to_terminal(intent, request_value=9263)
    try:
        terminal = next(event for event in events if event.kind in _TERMINAL)
        assert terminal.kind is StandardEventKind.FINISHED, terminal.primary
        assert (tmp_path / "chosen_int1d.nexus").is_file()
        assert (tmp_path / "chosen" / "iq_chosen_0001.xye").is_file()
        assert not (tmp_path / "chosen_gi").exists()
        assert not (tmp_path / "sample").exists()
    finally:
        executor.close(identity)
