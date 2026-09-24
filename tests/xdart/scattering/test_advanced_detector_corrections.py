"""Dark and flat field live in the Advanced dialog's Detector corrections group:
they edit the run intent, freeze into the run configuration and survive a profile
save/load. Controls no longer lists a flat field."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.advanced_editor import AdvancedSettingsDialog
from xdart.gui.tabs.scattering.controls_editing import (
    EditNoChange, EditRefusal, advanced_settings_values, reduce_advanced_settings,
)
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.session.run_intent_profile import dump_run_intent_profile, load_run_intent_profile
from xrd_tools.sources.selection import image_series_spec


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _snapshot(tmp_path: Path):
    source = tmp_path / "scan_1.tif"
    source.write_bytes(b"source")
    return RunIntentStore(RunIntent(source_spec=image_series_spec(source))).snapshot()


def test_dark_and_flat_edit_freeze_and_round_trip(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    dark, flat = str(tmp_path / "dark.tif"), str(tmp_path / "flat.tif")
    values = advanced_settings_values(snapshot)
    assert (values.dark_file, values.flat_file) == ("", "")

    candidate = reduce_advanced_settings(snapshot, replace(values, dark_file=f" {dark} ", flat_file=flat))
    frozen = candidate.freeze()
    assert (frozen.dark_file, frozen.flat_file) == (dark, flat)
    assert frozen.as_provenance()["dark_file"] == dark
    assert frozen.as_provenance()["flat_file"] == flat
    assert frozen.fingerprint != snapshot.thaw().freeze().fingerprint
    assert not {"dark_file", "flat_file"} & set(snapshot.thaw().freeze().as_provenance())

    edited = RunIntentStore(candidate).snapshot()
    shown = advanced_settings_values(edited)
    assert (shown.dark_file, shown.flat_file) == (dark, flat)
    assert type(reduce_advanced_settings(edited, shown)) is EditNoChange
    cleared = reduce_advanced_settings(edited, replace(shown, dark_file="", flat_file=""))
    assert (cleared.freeze().dark_file, cleared.freeze().flat_file) == ("", "")
    assert type(reduce_advanced_settings(edited, replace(shown, dark_file=None))) is EditRefusal

    restored = load_run_intent_profile(dump_run_intent_profile(candidate))
    assert (restored.dark_file, restored.flat_file) == (dark, flat)


def test_the_dialog_shows_browses_and_returns_both_paths(qapp, tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    requests = []

    def choose(role, current):
        requests.append((role, current))
        return str(tmp_path / f"{role}.npy")

    dialog = AdvancedSettingsDialog(path_chooser=choose)
    try:
        dialog.load_values(advanced_settings_values(snapshot))
        assert dialog.corrections_group.title() == "Detector corrections"
        assert (dialog.dark_file.text(), dialog.flat_file.text()) == ("", "")
        dialog.findChild(QtWidgets.QPushButton, "vnextDarkFileBrowse").click()
        dialog.findChild(QtWidgets.QPushButton, "vnextFlatFileBrowse").click()
        assert requests == [("dark", ""), ("flat", "")]
        values = dialog.values()
        assert (values.dark_file, values.flat_file) == (
            str(tmp_path / "dark.npy"), str(tmp_path / "flat.npy"))
        candidate = reduce_advanced_settings(snapshot, values)
        assert candidate.freeze().dark_file == str(tmp_path / "dark.npy")
    finally:
        dialog.close()


def test_controls_no_longer_list_a_flat_field(tmp_path: Path) -> None:
    fields = project_controls(_snapshot(tmp_path), None, RunPhase.IDLE).fields
    assert not any(field.path[-1] in {"flat_file", "dark_file"} for field in fields)
    assert not any(field.label in {"Flat Field", "Dark"} for field in fields)
