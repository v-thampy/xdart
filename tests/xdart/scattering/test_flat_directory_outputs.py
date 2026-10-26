"""A directory Run writes its outputs flat into the Save Path (maintainer, 2026-10-25: "in the early avatars xdart would
save all processed files in the parent save path and not create subdirs under the save path for subdirs in directory
mode ... it makes it much easier to navigate through processed outputs"; his pick "Prefix only on a clash").

A scan name found in more than one folder names each subfolder's output after its folder; a root-level scan keeps
its own. A Live Run never renames a scan it has already written: a same-named scan that appears later in another
folder is the one named after its folder. Names are compared ignoring case, as a Windows Save Path compares them, and
no two scans share an output, and a folder whose name is no legal start of an output name, or is long, still names a
legal one (Codex N-978a0cde FLAT-R1). The XYE viewer at a Run's end opens the file the Run wrote for a renamed
container (FLAT-R2). Real Runs over real TIFF + .txt scans and real containers."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _pump, _run
from tests.xdart.scattering.test_e2sd_directory_motor_exact import _recursive_container_start
from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _labels, _settle, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.output_preflight import _name_output_groups
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.run_plan import RunCandidatePlan
from xrd_tools.sources.selection import DirectorySourceSpec


@pytest.fixture
def qapp(_xdart_qt_harness):
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance()


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


@pytest.mark.parametrize(("folder", "scan", "expected"), [
    ("_calibration", "lab6", ["calibration_lab6_int1d.nexus", "sample_lab6_int1d.nexus"]),
    ("calibration" * 6, "sample" * 6, [f"{('calibration' * 6)[:40]}_{'sample' * 6}_int1d.nexus",
                                       f"sample_{'sample' * 6}_int1d.nexus"]),
])
def test_legal_folder_names_never_make_clashing_scans_unprocessable(tmp_path, folder, scan, expected) -> None:
    # A leading "_" is dropped; a long folder name gives way so the name stays within a family's 80 characters with
    # room for a GI Run's "_gi" (Codex N-978a0cde FLAT-R1 discriminator).
    raw = tmp_path / "raw"
    for name in (folder, "sample"):
        (raw / name).mkdir(parents=True)
        _frame(raw / name, scan, 1, "txt", 1.0)
    executor, identity, events = _run_to_terminal(_recursive(tmp_path, raw, live=False), request_value=10301)
    executor.close(identity)
    terminal = [event for event in events if event.kind in _TERMINAL]
    assert [event.kind.name for event in terminal] == ["FINISHED"] and terminal[0].files_processed == 2
    assert _written(tmp_path / "processed") == expected


def test_renamed_containers_open_the_xye_file_the_run_actually_wrote(qapp, tmp_path) -> None:
    # Two scan_0001 containers under data/ and live_test/: their scans keep their own names while their outputs are
    # named after their folders (Codex N-978a0cde FLAT-R2 discriminator, the real page Run-click path).
    start, _raw, output = _recursive_container_start(tmp_path)
    intent = replace(start.intent_snapshot.thaw(), processing_mode="Int 1D (XYE)", max_cores=1,
                     bai_1d_args={"npt": 8, "method": "numpy"})
    with _page(qapp, intent) as (page, _scientific, events):
        terminal = _run(qapp, page, events)
        assert terminal.completed == 2
        generated = set(output.rglob("*.xye"))
        assert len(generated) == 2
        controller = page._context_controller
        ready = _pump(qapp, lambda: controller.viewer_1d_context is not None
                      and controller.viewer_1d_context.state.value == "ready", 15.)
        assert ready, (sorted(map(str, generated)), page._notice_text)
        assert Path(controller.viewer_1d_context.current_path) in generated


def test_a_folder_named_in_many_byte_characters_still_names_an_output_one_component_holds(tmp_path) -> None:
    # 60 rockets: 240 UTF-8 bytes, a legal folder and family start, but with "_lab6_int1d.nexus" 257 bytes, more than
    # one component holds on ext4. The folder part gives way in bytes too (Codex N-40897109 FLAT-R3 discriminator).
    rocket = "\U0001F680"
    raw = tmp_path / "raw"
    for name in (rocket * 60, "sample"):
        (raw / name).mkdir(parents=True)
        _frame(raw / name, "lab6", 1, "txt", 1.0)
    executor, identity, events = _run_to_terminal(_recursive(tmp_path, raw, live=False), request_value=10302)
    executor.close(identity)
    terminal = [event for event in events if event.kind in _TERMINAL]
    assert [event.kind.name for event in terminal] == ["FINISHED"] and terminal[0].files_processed == 2
    written = _written(tmp_path / "processed")
    assert written == ["sample_lab6_int1d.nexus", f"{rocket * 40}_lab6_int1d.nexus"]
    assert all(len(name.encode("utf-8")) <= 255 for name in written)
