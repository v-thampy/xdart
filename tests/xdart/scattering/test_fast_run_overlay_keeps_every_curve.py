"""A fast Run ends with every frame it processed drawn in Overlay (maintainer, 2026-10-26: "1d traces are so cheap that
they can easily be kept in memory for an entire run"; "the painting can lag if a norm channel is chosen, but it
shouldn't skip frames at the end of the run"). Before, 16 one-frame scans in Overlay ended with 2 curves in a directory
Run and 5 in a Live burst: each scan's curves were released when the next began, before the lagging view drew them, and
the end's Browse re-reads only the last file. The real page, executor and Runs over real TIFF + .txt files; the GUI is
starved while the Run works (no Qt events), as a slow GUI is."""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _poni, _pump
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

#: Below sixteen, Overlay draws curves; from the sixteenth trace on it shows them as a waterfall image
#: (scientific_waterfall_policy), its rows the same history.
COUNTS = (12, 16)


def _intent(root, *, live):
    raw = root / "raw"
    return RunIntent(source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
                     poni_file=str(_poni(root)), project_root=str(raw), save_path=str(root / "processed"),
                     output_mode="Overwrite", processing_mode="Int 1D", live_mode=live, max_cores=1,
                     bai_1d_args={"npt": 8, "method": "numpy"})


def _starved_until(predicate, seconds=120.):
    """Wait with the GUI starved: no Qt events, so the page drains nothing meanwhile."""
    until = time.monotonic() + seconds
    while time.monotonic() < until and not predicate():
        time.sleep(.02)
    return predicate()


def _drawn(page):
    """The frames the view shows, each as its scan's output: a Run and a Browse of one scan name it alike."""
    return sorted(os.path.basename(str(frame.artifact)) for frame in page._shell.scientific.trace_history_keys)


def _expected(page, stems):
    assert _drawn(page) == [f"{stem}_int1d.nexus" for stem in stems], _drawn(page)
    if len(stems) < 16:
        assert len(page._shell.scientific.curve.listDataItems()) == len(stems)


def _ended_on_browse(qapp, page, last):  # noqa: F811
    """The end-of-Run Browse of the last scan's output has painted: its row is Browse's, nothing is pending."""
    controller, view = page._context_controller, page._shell.scientific
    return _pump(qapp, lambda: page._lifecycle.phase is RunPhase.IDLE and not controller.browse_pending
                 and controller.browse_context is not None and controller.browse_context.loaded
                 and not page._scientific_repaint_pending
                 and any(frame.source_scan == f"{last}_int1d" for frame in view.trace_history_keys), 60.)


def _overlay(page):
    page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))


@pytest.mark.parametrize("count", COUNTS)
def test_a_fast_directory_run_ends_with_every_scan_drawn_in_overlay(qapp, tmp_path, count):  # noqa: F811
    stems = tuple(f"s{index:02d}" for index in range(count))
    raw = tmp_path / "raw"
    raw.mkdir()
    for index, stem in enumerate(stems):
        _frame(raw, stem, 1, "txt", float(index + 1))
    with _page(qapp, _intent(tmp_path, live=False)) as (page, _view, _observed):
        _overlay(page)
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: page._lifecycle.active_run_identity is not None, 30.)
        run = page._run_executor._exact_run(page._lifecycle.active_run_identity)
        assert _starved_until(lambda: run.terminal_emitted), "the Run never ended"
        assert _ended_on_browse(qapp, page, stems[-1]), (page._notice_text, _drawn(page))
        _expected(page, stems)


@pytest.mark.parametrize("count", COUNTS)
def test_a_fast_live_burst_ends_with_every_scan_drawn_in_overlay(qapp, tmp_path, count):  # noqa: F811
    stems = tuple(f"s{index:02d}" for index in range(count))
    raw = tmp_path / "raw"
    raw.mkdir()
    with _page(qapp, _intent(tmp_path, live=True)) as (page, _view, _observed):
        _overlay(page)
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: page._lifecycle.active_run_identity is not None, 30.)
        run = page._run_executor._exact_run(page._lifecycle.active_run_identity)
        for index, stem in enumerate(stems):
            _frame(raw, stem, 1, "txt", float(index + 1))
        assert _starved_until(lambda: run.files_processed >= len(stems)), run.files_processed
        page._shell.run_controls.stopButton.click()
        assert _starved_until(lambda: run.terminal_emitted), "the Run never stopped"
        assert _ended_on_browse(qapp, page, stems[-1]), (page._notice_text, _drawn(page))
        _expected(page, stems)


def test_a_fast_directory_run_with_a_norm_channel_ends_with_every_scan_drawn_each_divided_by_its_i0(
        qapp, tmp_path):  # noqa: F811
    """The kept curves carry each frame's own metadata, so a selected Norm Channel divides each by its own I0 and its
    coverage spans every scan: the view's history is not cleared at the scan boundaries."""
    from tests.xdart.scattering.test_overlay_persistence_run import _divided

    stems = tuple(f"s{index:02d}" for index in range(12))
    raw = tmp_path / "raw"
    raw.mkdir()
    for index, stem in enumerate(stems):
        _frame(raw, stem, 1, "txt", float(index + 1))
    with _page(qapp, _intent(tmp_path, live=False)) as (page, _view, _observed):
        _overlay(page)
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, 0.0, ("overlay", "offset")))
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_NORM_CHANNEL, "i0"))
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: page._lifecycle.active_run_identity is not None, 30.)
        run = page._run_executor._exact_run(page._lifecycle.active_run_identity)
        assert _starved_until(lambda: run.terminal_emitted), "the Run never ended"
        assert _ended_on_browse(qapp, page, stems[-1]), (page._notice_text, _drawn(page))
        _expected(page, stems)
        assert page._last_scientific_projection.norm_channel == "i0"
        _divided(page, "int1d")
