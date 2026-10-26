"""A directory Run leaves out the frames that have no sidecar of their own (maintainer, 2026-10-26: "If I have chosen
txt as the metadata type, then any frame without the txt sidecar should be skipped"; "in a live or directory run, it
can be assumed that the moment a metadata sidecar is found, that type applies universally for all data in all folders
for that run"; "The only time the 150s wait should come in is when the frame in question is the last or only frame to
be processed in a directory or series", his pick "Only if recent": the last frame waits only while its image is newer
than the sidecar limit). Real directory Runs over real TIFF + .txt files."""
from __future__ import annotations

import logging
import os
import threading
import time

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _output
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent(tmp_path, fmt="txt"):
    raw, poni = tmp_path / "raw", tmp_path / "cal.poni"
    write_poni(poni)
    return RunIntent(source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format=fmt), poni_file=str(poni),
                     project_root=str(raw), save_path=str(tmp_path / "processed"), output_mode="Overwrite",
                     processing_mode="Int 1D", live_mode=False, max_cores=1, bai_1d_args={"npt": 8, "method": "numpy"})


def _run(tmp_path, request_value, fmt="txt"):
    started = time.monotonic()
    executor, identity, events = _run_to_terminal(_intent(tmp_path, fmt), request_value=request_value)
    terminal = next(event for event in events if event.kind in _TERMINAL)
    executor.close(identity)
    assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
    return time.monotonic() - started


def _aged(path, seconds=3600.):
    """As an image written long before the Run."""
    then = time.time() - seconds
    os.utime(path, (then, then))


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


def test_a_frame_without_its_txt_is_left_out_and_its_scan_goes_on(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    for index, sidecar in ((1, True), (2, False), (3, True)):
        _frame(raw, "grow", index, "txt", float(index), sidecar=sidecar)
    _run(tmp_path, 7801)
    assert _output(tmp_path, "grow")[:2] == ((1, 2), (1.0, 3.0))
    messages = _messages(caplog)
    assert "grow_int1d.nexus: grow_0002.tif is left out; it has no metadata sidecar of its own [SCAN]" in messages
    assert "Left out without their metadata (1): grow_0002.tif (grow_int1d.nexus) [LIVE]" in messages


def test_an_old_last_frame_without_its_txt_is_left_out_at_once(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in (1, 2):
        _frame(raw, "grow", index, "txt", float(index))
    _aged(_frame(raw, "grow", 3, "txt", 3.0, sidecar=False))
    took = _run(tmp_path, 7802)
    assert took < 20., f"an old last frame waited: {took:.1f} s"
    assert _output(tmp_path, "grow")[:2] == ((1, 2), (1.0, 2.0))


def test_a_recent_last_frame_waits_for_its_txt_and_is_processed_with_it(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _frame(raw, "grow", 1, "txt", 1.0)
    _frame(raw, "grow", 2, "txt", 2.0, sidecar=False)  # just written: its .txt lands a second later
    landing = threading.Timer(1.0, lambda: _frame(raw, "grow", 2, "txt", 2.0, image=False))
    landing.start()
    try:
        _run(tmp_path, 7803)
    finally:
        landing.cancel()
    assert _output(tmp_path, "grow")[:2] == ((1, 2), (1.0, 2.0))


def test_a_recent_last_frame_whose_txt_never_comes_is_left_out_at_the_limit(tmp_path, monkeypatch):
    monkeypatch.setitem(run_executor._METADATA_WAIT_LIMIT_S, "sidecar", 2.)  # 150 s by default
    raw = tmp_path / "raw"
    raw.mkdir()
    _frame(raw, "grow", 1, "txt", 1.0)
    _frame(raw, "grow", 2, "txt", 2.0, sidecar=False)
    took = _run(tmp_path, 7804)
    assert took >= 1.5, f"the recent last frame did not wait: {took:.1f} s"
    assert _output(tmp_path, "grow")[:2] == ((1,), (1.0,))


def test_under_auto_a_folder_without_any_sidecar_is_processed_without_metadata(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in (1, 2, 3):
        _aged(_frame(raw, "grow", index, "txt", float(index), sidecar=False))
    _run(tmp_path, 7805, fmt="auto")
    labels, i0, _epoch = _output(tmp_path, "grow")
    assert labels == (1, 2, 3) and i0 is None, (labels, i0)


def test_under_auto_the_first_sidecar_found_makes_every_frame_need_its_own(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _frame(raw, "with", 1, "txt", 1.0)
    _aged(_frame(raw, "grow", 1, "txt", 2.0, sidecar=False))
    _frame(raw, "grow", 2, "txt", 3.0)
    _run(tmp_path, 7806, fmt="auto")
    assert _output(tmp_path, "with")[:2] == ((1,), (1.0,))
    assert _output(tmp_path, "grow")[:2] == ((1,), (3.0,))
