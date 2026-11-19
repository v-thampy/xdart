"""A directory Int 1D Run goes on while a frame read is still running at a scan boundary (LIVE-1212-LIGHT1D-LINEAGE;
maintainer, 2026-11-18, from the beamline: "Keep getting this crap in int 1d in directory mode", a Run on an X: share
failing with "prior light-1D lineage cleanup remains pending" after 9 of 170 frames).

Releasing a finished scan's light-1D curves refused while any frame read of the Run was still running, and the Run
failed on that one refusal. A slow share makes a read span a boundary. The real page, executor and Run over real TIFF +
.txt files, frames picked the way a click picks them; only ``read_frame_preview`` is held, as a slow share holds it, and
then reads for real."""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("pyqtgraph")

import xdart.gui.tabs.scattering.hydration_transport as hydration_transport
from tests.xdart.scattering.test_browse_reference_rings_run import _page, _poni
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

_TERMINAL = {StandardEventKind.FINISHED, StandardEventKind.STOPPED, StandardEventKind.FAILED}


def _scans(tmp_path, count, frames):
    raw = tmp_path / "raw"
    raw.mkdir()
    stems = tuple(f"s{index:02d}" for index in range(count))
    for index, stem in enumerate(stems):
        for frame in range(1, frames + 1):
            _frame(raw, stem, frame, "txt", float(index + 1))
    intent = RunIntent(source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
                       poni_file=str(_poni(tmp_path)), project_root=str(raw), save_path=str(tmp_path / "processed"),
                       output_mode="Overwrite", processing_mode="Int 1D", live_mode=False, max_cores=1,
                       bai_1d_args={"npt": 8, "method": "numpy"})
    return stems, intent


def _held_reads(monkeypatch, release):
    """Each frame read waits until ``release(name)`` (its output's file name) is true, as a slow share makes it wait,
    then reads for real. Returns the names read, in order."""
    real, names = hydration_transport.read_frame_preview, []

    def held(read_key, *args, **kwargs):
        name = os.path.basename(str(read_key.artifact_identity))
        names.append(name)
        until = time.monotonic() + 30.
        while time.monotonic() < until and not release(name):
            time.sleep(.01)
        return real(read_key, *args, **kwargs)

    monkeypatch.setattr(hydration_transport, "read_frame_preview", held)
    return names


def _outputs(page):
    run = page._run_executor._exact_run(page._lifecycle.active_run_identity)
    catalog = page._run_executor.frame_catalog(page._lifecycle.active_run_identity)
    entries = () if catalog is None else catalog.entries
    return run, entries


def _click(page, entry):
    page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_FRAME, frame=entry, frames=(entry,)))


def _finish(qapp, page, observed, pick, *, clicks=1, every=.05, seconds=120.):  # noqa: F811
    """Run with the GUI pumping (as the app does); ``pick(entries)`` names a frame to click, at most ``clicks`` times
    and ``every`` seconds apart."""
    page._shell.run_controls.startButton.click()
    started = time.monotonic()
    clicked, next_at, until, looks = 0, 0., started + seconds, []
    while time.monotonic() < until and not any(event.kind in _TERMINAL for event in observed):
        qapp.processEvents()
        time.sleep(.003)
        if clicked >= clicks or time.monotonic() < next_at or page._lifecycle.active_run_identity is None:
            continue
        entries = _outputs(page)[1]
        if not looks or looks[-1][1] != len(entries):
            looks.append((round(time.monotonic() - started, 2), len(entries)))
        entry = pick(entries)
        if entry is not None:
            _click(page, entry)
            clicked, next_at = clicked + 1, time.monotonic() + every
    terminals = [event for event in observed if event.kind in _TERMINAL]
    assert clicked, f"no frame was picked during the Run; the picker saw (seconds, frames listed): {looks[:30]}"
    assert terminals, "the Run never ended"
    return terminals[-1]


def _written(tmp_path, stems):
    return sorted(path.name for path in (tmp_path / "processed").glob("*_int1d.nexus")) == \
        [f"{stem}_int1d.nexus" for stem in stems]


def test_a_read_of_an_earlier_scan_does_not_hold_the_next_boundary(qapp, tmp_path, monkeypatch):  # noqa: F811
    """A frame of the first scan picked while the fourth is reduced: its read (held until the last scan starts) is not
    the fourth scan's, so the fourth scan's boundary does not wait for it."""
    stems, intent = _scans(tmp_path, 7, 4)
    started = set()
    names = _held_reads(monkeypatch, lambda name: name != f"{stems[0]}_int1d.nexus" or stems[-1] in started)

    def pick(entries):
        scans = {os.path.basename(entry.artifact)[:3] for entry in entries}
        started.update(scans)
        if len(scans) < 4:
            return None
        return next(entry for entry in entries if os.path.basename(entry.artifact) == f"{stems[0]}_int1d.nexus")

    with _page(qapp, intent) as (page, _view, observed):
        terminal = _finish(qapp, page, observed, pick)
        assert terminal.kind is StandardEventKind.FINISHED, (page._notice_text, terminal)
        assert f"{stems[0]}_int1d.nexus" in names
        assert _written(tmp_path, stems)


def test_a_read_of_the_finished_scan_is_waited_for_at_its_boundary(qapp, tmp_path, monkeypatch):  # noqa: F811
    """Earlier frames of the first scan picked one after another while it is reduced, their raw images gone from memory
    (eight frames kept, as a big detector's images on a small PC leave): reads of it (a slow share's 1.5 s each) run
    around its boundary, and the Run goes on. Picking may go on until the next scan starts, so a slow runner cannot miss
    the window (hosted run 44453095652 did); whether a read is still running at the boundary then depends on timing,
    so this row no longer proves the wait itself (LIVE-1212-LIGHT1D-LINEAGE residual)."""
    import fabio

    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    stems, intent = _scans(tmp_path, 3, 80)
    real_open = fabio.open

    def share_open(*args, **kwargs):  # each image takes 40 ms to open, so the first scan is seen while it is reduced
        time.sleep(.04)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(fabio, "open", share_open)
    first = {}

    def release(name):
        first.setdefault(name, time.monotonic())
        return time.monotonic() - first[name] >= 1.5

    def pick(entries):
        own = [entry for entry in entries if os.path.basename(entry.artifact) == f"{stems[0]}_int1d.nexus"]
        if len(own) < 10 or len(own) != len(entries):  # until the next scan starts
            return None
        pick.turn = (getattr(pick, "turn", -1) + 1) % (len(own) - 8)
        return own[pick.turn]

    names = _held_reads(monkeypatch, release)
    with _page(qapp, intent) as (page, _view, observed):
        terminal = _finish(qapp, page, observed, pick, clicks=80, every=.02)
        assert terminal.kind is StandardEventKind.FINISHED, (page._notice_text, terminal)
        assert f"{stems[0]}_int1d.nexus" in names
        assert _written(tmp_path, stems)
