"""The Flat Field control edits the run intent, projects as a browse field, freezes
into the run configuration, and survives a profile save/load."""
from __future__ import annotations

from pathlib import Path

from xdart.gui.tabs.scattering.controls_editing import EditNoChange, reduce_control_edit
from xdart.gui.tabs.scattering.controls_inventory import FLAT_FILE, MASK_FILE
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.session.run_intent_profile import dump_run_intent_profile, load_run_intent_profile
from xrd_tools.sources.selection import image_series_spec


def test_flat_field_control_edits_projects_freezes_and_round_trips(tmp_path: Path) -> None:
    source = tmp_path / "scan_1.tif"; source.write_bytes(b"source")
    snapshot = RunIntentStore(RunIntent(source_spec=image_series_spec(source))).snapshot()
    flat = str(tmp_path / "flat.tif")

    candidate = reduce_control_edit(snapshot, FLAT_FILE, flat)
    frozen = candidate.freeze()
    assert frozen.flat_file == flat
    assert frozen.as_provenance()["flat_file"] == flat
    assert frozen.fingerprint != snapshot.thaw().freeze().fingerprint
    assert "flat_file" not in snapshot.thaw().freeze().as_provenance()

    edited = RunIntentStore(candidate).snapshot()
    fields = {field.path: field for field in project_controls(edited, None, RunPhase.IDLE).fields}
    assert fields[FLAT_FILE].label == "Flat Field" and fields[FLAT_FILE].browse
    assert fields[FLAT_FILE].enabled and fields[FLAT_FILE].value == flat
    assert fields[FLAT_FILE].section == fields[MASK_FILE].section
    assert type(reduce_control_edit(edited, FLAT_FILE, flat)) is EditNoChange
    assert reduce_control_edit(edited, FLAT_FILE, "").freeze().flat_file == ""

    assert load_run_intent_profile(dump_run_intent_profile(candidate)).flat_file == flat
