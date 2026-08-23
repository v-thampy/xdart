"""Focused Controls capture for the P3-3B headless Background policy."""
from __future__ import annotations

from pathlib import Path

from xdart.gui.tabs.scattering.controls_editing import EditRefusal, reduce_control_edit
from xdart.gui.tabs.scattering.controls_inventory import BACKGROUND_FILE, BACKGROUND_TYPE
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec
from xrd_tools.reduction import FrameBackgroundPlan


def _snapshot(tmp_path: Path):
    source = tmp_path / "scan_1.tif"; source.write_bytes(b"source")
    return RunIntentStore(RunIntent(source_spec=image_series_spec(source))).snapshot()


def test_parent_red_background_mode_edit_becomes_frozen_science_policy(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    try:
        candidate = reduce_control_edit(snapshot, BACKGROUND_TYPE, "Single BG File")
        plan = getattr(candidate, "background", None)
    except (KeyError, TypeError, ValueError):
        plan = None
    assert plan is not None and plan.mode == "Single BG File"
    background = tmp_path / "background.tif"; background.write_bytes(b"background")
    complete = reduce_control_edit(RunIntentStore(candidate).snapshot(),
                                   BACKGROUND_FILE, str(background))
    scaled = reduce_control_edit(RunIntentStore(complete).snapshot(),
                                 ("BG", "Scale"), -1.5)
    normalized = reduce_control_edit(RunIntentStore(scaled).snapshot(),
                                     ("BG", "Normalize"), "monitor")
    frozen = normalized.freeze()
    assert frozen.background.mode == "Single BG File"
    assert frozen.as_provenance()["background"]["locator"] == str(background)
    assert frozen.background.scale == -1.5 and frozen.background.normalization_key == "monitor"


def test_background_control_grammar_projection_and_dormant_fields(tmp_path: Path) -> None:
    from xdart.gui.tabs.scattering.controls_inventory import (
        BACKGROUND_DIRECTORY, BACKGROUND_FILE, BACKGROUND_FILTER,
        BACKGROUND_MATCH, BACKGROUND_METADATA_KEY, BACKGROUND_NORMALIZE,
        BACKGROUND_SCALE,
    )
    snapshot = _snapshot(tmp_path)
    none = project_controls(snapshot, None, RunPhase.IDLE)
    paths = {field.path for field in none.bound_controls.fields}
    assert BACKGROUND_TYPE in paths
    assert not paths & {BACKGROUND_FILE, BACKGROUND_DIRECTORY, BACKGROUND_MATCH,
                        BACKGROUND_METADATA_KEY, BACKGROUND_FILTER,
                        BACKGROUND_SCALE, BACKGROUND_NORMALIZE}
    mode_field = next(field for field in none.bound_controls.fields if field.path == BACKGROUND_TYPE)
    assert mode_field.enabled and mode_field.choices == (
        "None", "Single BG File", "Series Average", "BG Directory")
    dormant = RunIntent(background=FrameBackgroundPlan(scale=-9.0)).freeze()
    assert dormant.fingerprint == RunIntent().freeze().fingerprint
    for mode, label in (("Single BG File", "Source File"),
                        ("Series Average", "Series Member")):
        selected = reduce_control_edit(snapshot, BACKGROUND_TYPE, mode)
        fields = {field.path: field for field in project_controls(
            RunIntentStore(selected).snapshot(), None, RunPhase.IDLE).bound_controls.fields}
        assert fields[BACKGROUND_FILE].browse and fields[BACKGROUND_FILE].label == label
        assert fields[BACKGROUND_FILE].enabled and BACKGROUND_DIRECTORY not in fields
    candidate = reduce_control_edit(snapshot, BACKGROUND_TYPE, "BG Directory")
    directory = project_controls(RunIntentStore(candidate).snapshot(), None, RunPhase.IDLE)
    fields = {field.path: field for field in directory.bound_controls.fields}
    assert {BACKGROUND_DIRECTORY, BACKGROUND_MATCH, BACKGROUND_FILTER,
            BACKGROUND_SCALE, BACKGROUND_NORMALIZE} <= set(fields)
    assert BACKGROUND_FILE not in fields and BACKGROUND_METADATA_KEY not in fields
    metadata = reduce_control_edit(RunIntentStore(candidate).snapshot(),
                                   BACKGROUND_MATCH, "Metadata Key")
    fields = {field.path: field for field in project_controls(
        RunIntentStore(metadata).snapshot(), None, RunPhase.IDLE).bound_controls.fields}
    assert BACKGROUND_METADATA_KEY in fields
    assert fields[BACKGROUND_METADATA_KEY].enabled


def test_background_edit_and_freeze_refusals(tmp_path: Path) -> None:
    from xdart.gui.tabs.scattering.controls_inventory import (
        BACKGROUND_DIRECTORY, BACKGROUND_FILE, BACKGROUND_FILTER,
        BACKGROUND_MATCH, BACKGROUND_METADATA_KEY, BACKGROUND_SCALE,
    )
    snapshot = _snapshot(tmp_path)
    assert isinstance(reduce_control_edit(snapshot, BACKGROUND_SCALE, float("nan")), EditRefusal)
    assert isinstance(reduce_control_edit(snapshot, BACKGROUND_SCALE, float("inf")), EditRefusal)
    assert isinstance(reduce_control_edit(snapshot, BACKGROUND_FILTER, "|"), EditRefusal)
    assert isinstance(reduce_control_edit(snapshot, BACKGROUND_MATCH, "first"), EditRefusal)
    single = reduce_control_edit(snapshot, BACKGROUND_TYPE, "Single BG File")
    assert isinstance(reduce_control_edit(RunIntentStore(single).snapshot(),
                                          BACKGROUND_DIRECTORY, str(tmp_path)), EditRefusal)
    container = tmp_path / "background.h5"; container.write_bytes(b"container")
    assert isinstance(reduce_control_edit(RunIntentStore(single).snapshot(),
                                          BACKGROUND_FILE, str(container)), EditRefusal)
    directory = reduce_control_edit(snapshot, BACKGROUND_TYPE, "BG Directory")
    metadata = reduce_control_edit(RunIntentStore(directory).snapshot(),
                                   BACKGROUND_MATCH, "Metadata Key")
    assert isinstance(reduce_control_edit(RunIntentStore(metadata).snapshot(),
                                          BACKGROUND_METADATA_KEY, ""), EditRefusal)
    accepted_key = reduce_control_edit(RunIntentStore(metadata).snapshot(),
                                       BACKGROUND_METADATA_KEY, "k" * 256)
    assert not isinstance(accepted_key, EditRefusal)
    assert isinstance(reduce_control_edit(RunIntentStore(metadata).snapshot(),
                                          BACKGROUND_METADATA_KEY, "k" * 257), EditRefusal)
    assert not isinstance(reduce_control_edit(snapshot, BACKGROUND_SCALE, -2.0), EditRefusal)
    for plan in (
        FrameBackgroundPlan(mode="Single BG File", locator=str(tmp_path / "one.tif")),
        FrameBackgroundPlan(mode="Series Average", locator=str(tmp_path / "scan_1.tif")),
        FrameBackgroundPlan(mode="BG Directory", locator=str(tmp_path),
                            match_rule="Scan Root + Frame Number"),
    ):
        assert RunIntent(background=plan).freeze().background == plan
    for incomplete in (
        FrameBackgroundPlan(mode="Single BG File"),
        FrameBackgroundPlan(mode="Series Average"),
        FrameBackgroundPlan(mode="BG Directory", locator=str(tmp_path)),
    ):
        try: RunIntent(background=incomplete).freeze()
        except ValueError: pass
        else: raise AssertionError("incomplete active Background policy froze")
    try: FrameBackgroundPlan(mode="Single BG File", locator="/" + "x" * 4096)
    except ValueError: pass
    else: raise AssertionError("Background locator cap+1 was accepted")
