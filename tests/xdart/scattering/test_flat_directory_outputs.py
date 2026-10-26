"""A directory Run writes its outputs flat into the Save Path (maintainer, 2026-10-25: "in the early avatars xdart would
save all processed files in the parent save path and not create subdirs under the save path for subdirs in directory
mode ... it makes it much easier to navigate through processed outputs"; his pick "Prefix only on a clash").

A scan name found in more than one folder names each subfolder's output after its folder; a root-level scan keeps
its own. A Live Run never renames a scan it has already written: a same-named scan that appears later in another
folder is the one named after its folder. Names are compared ignoring case, as a Windows Save Path compares them, and
no two scans share an output. Real Runs over real TIFF + .txt scans."""
from __future__ import annotations

from dataclasses import replace

from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _labels, _settle, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.output_preflight import _name_output_groups
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.run_plan import RunCandidatePlan
from xrd_tools.sources.selection import DirectorySourceSpec


def _recursive(tmp_path, raw, *, live):
    return replace(_intent(tmp_path, raw, live=live),
                   source_spec=DirectorySourceSpec(raw, recursive=True, suffixes=(".tif",), metadata_format="txt"))


def _written(save):
    return sorted(path.relative_to(save).as_posix() for path in save.rglob("*.nexus"))


def test_a_directory_run_writes_flat_and_names_clashing_scans_after_their_folders(tmp_path) -> None:
    raw = tmp_path / "Baniecki_Oct2026"
    for folder in ("a", "b", "b/nested"):
        (raw / folder).mkdir(parents=True)
    _frame(raw, "lab6", 1, "txt", 1.0)            # the root's own: keeps its name
    for index in (1, 2):
        _frame(raw / "a", "lab6", index, "txt", float(index))
    _frame(raw / "b", "lab6", 1, "txt", 1.0)
    _frame(raw / "b", "other", 1, "txt", 1.0)     # no clash: its own name, flat
    executor, identity, events = _run_to_terminal(_recursive(tmp_path, raw, live=False), request_value=9961)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    assert _written(tmp_path / "processed") == [
        "a_lab6_int1d.nexus", "b_lab6_int1d.nexus", "lab6_int1d.nexus", "other_int1d.nexus"]


def test_live_keeps_a_written_scans_name_when_a_same_named_scan_appears_elsewhere(tmp_path) -> None:
    raw = tmp_path / "Baniecki_Oct2026"
    (raw / "calibration").mkdir(parents=True)
    _frame(raw / "calibration", "lab6", 1, "txt", 1.0)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _recursive(tmp_path, raw, live=True), request_value=9962)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values, "lab6") or any(
            event.kind in _TERMINAL for event in values))
        assert _written(tmp_path / "processed") == ["lab6_int1d.nexus"]
        (raw / "sampleA").mkdir()
        _frame(raw / "sampleA", "lab6", 1, "txt", 5.0)   # the same scan name in another folder
        _frame(raw / "calibration", "lab6", 2, "txt", 2.0)  # the first scan grows on
        _wait(executor, events, lambda values: 2 in _labels(values, "lab6") or any(
            event.kind in _TERMINAL for event in values))
        _settle(executor, events, 1.)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert _written(tmp_path / "processed") == ["lab6_int1d.nexus", "sampleA_lab6_int1d.nexus"]


def test_names_differing_only_in_case_clash_and_no_two_scans_share_an_output(tmp_path) -> None:
    raw = tmp_path / "Baniecki_Oct2026"
    (raw / "a").mkdir(parents=True)
    _frame(raw, "lab6", 1, "txt", 1.0)
    _frame(raw, "a_lab6", 1, "txt", 1.0)          # a real scan with the name a's LaB6 would be given
    _frame(raw / "a", "LaB6", 1, "txt", 1.0)      # lab6 ignoring case: named after its folder, then numbered
    executor, identity, events = _run_to_terminal(_recursive(tmp_path, raw, live=False), request_value=9963)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    assert _written(tmp_path / "processed") == ["a_LaB6_2_int1d.nexus", "a_lab6_int1d.nexus", "lab6_int1d.nexus"]


def test_a_live_newcomer_whose_name_differs_only_in_case_is_named_after_its_folder(tmp_path) -> None:
    raw = tmp_path / "Baniecki_Oct2026"
    for folder in ("calibration", "sampleA"):
        (raw / folder).mkdir(parents=True)
    _frame(raw / "calibration", "lab6", 1, "txt", 1.0)
    _frame(raw / "sampleA", "LAB6", 1, "txt", 1.0)
    _frame(raw, "Lab6", 1, "txt", 1.0)
    session = DirectoryIndexSession(probe_candidates=False)
    try:
        session.configure(raw, recursive=True, suffixes=(".tif",))
        plan = RunCandidatePlan.from_snapshot(session.observe(refresh=True).discovered_snapshot)
    finally:
        session.close()
    assigned = {("calibration", "lab6"): "lab6"}  # written by this Live Run before the others appeared
    named = _name_output_groups(plan, assigned)
    assert assigned == {("calibration", "lab6"): "lab6", ("sampleA", "LAB6"): "sampleA_LAB6",
                        (".", "Lab6"): "Baniecki_Oct2026_Lab6"}
    assert named.output_names == ((".", "Lab6", "Baniecki_Oct2026_Lab6"), ("sampleA", "LAB6", "sampleA_LAB6"))
    assert _name_output_groups(named, dict(assigned)).output_names == named.output_names  # the next poll agrees
