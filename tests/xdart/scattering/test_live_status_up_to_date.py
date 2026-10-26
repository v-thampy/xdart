"""A running Live Run that has processed everything it found says so (maintainer, 2026-10-26: "make it clear in the
status when all scans/frames have been processed in a live run with a brief message implying that xdart is waiting for
new frames", his pick "Lead with it"), compacted the same day ("this status is very long and can be shrunk. maybe
remove "up to date"", his pick "Fraction", keeping the new-frame count, and "yes for the total visible in all
states"): "Running · Waiting for new images · 561/561 processed (561 new)". The real page, executor and Live Run over
real TIFF + .txt files."""
from __future__ import annotations

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _poni, _pump, _terminal
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.shell_values import DirectoryFileProgress
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

UP_TO_DATE = "Waiting for new images"


def test_the_status_leads_with_waiting_only_while_a_running_live_run_has_nothing_pending():
    caught_up, behind = DirectoryFileProgress(561, 0, 0, 561, 561), DirectoryFileProgress(537, 2, 22, 561, 537)
    assert caught_up.text("Running", watching=True) == (
        "Running · Waiting for new images · 561/561 processed (561 new)")
    assert behind.text("Running", watching=True) == behind.text("Running") == (
        "Running · 537/561 processed (537 new) · 2 skipped")  # skipped only when some were
    assert UP_TO_DATE not in caught_up.text("Paused")  # only a running Live Run watches
    assert caught_up.text("Paused") == "Paused · 561/561 processed (561 new)"


def test_zero_directory_totals_stay_visible_in_every_status():
    empty = DirectoryFileProgress(0, 0, 0, 0)
    for state in ("Starting", "Running", "Paused", "Failed"):
        assert empty.text(state) == f"{state} · 0/0 processed"


@pytest.mark.parametrize("output_mode", ("Overwrite", "Append"))
def test_an_empty_live_directory_uses_the_compact_waiting_status(qapp, tmp_path, output_mode):
    from tests.xdart.scattering.test_browse_reference_rings_run import _directory

    with _page(qapp, _directory(tmp_path, live=True, output_mode=output_mode)) as (page, _view, observed):
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: any(event.kind is StandardEventKind.DISCOVERY for event in observed), 30.)
        new = " (0 new)" if output_mode == "Append" else ""
        assert page._shell.run_controls.readinessLabel.full_text() == (
            f"Running · Waiting for new images · 0/0 processed{new}")
        page._shell.run_controls.stopButton.click()
        _terminal(qapp, page, observed, StandardEventKind.STOPPED)


def test_a_live_run_that_processed_all_it_found_says_it_waits_for_new_images(qapp, tmp_path):  # noqa: F811
    raw = tmp_path / "raw"
    raw.mkdir()
    _frame(raw, "scan", 1, "txt", 1.0)
    intent = RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
        poni_file=str(_poni(tmp_path)), project_root=str(raw), save_path=str(tmp_path / "processed"),
        output_mode="Overwrite", processing_mode="Int 1D", live_mode=True, max_cores=1,
        bai_1d_args={"npt": 8, "method": "numpy"})
    with _page(qapp, intent) as (page, _view, observed):
        label = page._shell.run_controls.readinessLabel
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: UP_TO_DATE in label.full_text(), 30.), label.full_text()
        assert label.full_text() == "Running · Waiting for new images · 1/1 processed", label.full_text()
        # It leads, however it elides: the label keeps the head of the text, cut where its width ends (hosted 40320041462
        # showed the longer earlier text cut as "Running · Up to date, waiting for new imag…").
        shown = label.text()
        assert shown.startswith("Running · Waiting") and label.full_text().startswith(shown.rstrip("…")), shown
        _frame(raw, "scan", 2, "txt", 2.0, sidecar=False)  # a new image, its .txt not yet there: pending again
        assert _pump(qapp, lambda: "1/2 processed" in label.full_text(), 30.), label.full_text()
        assert UP_TO_DATE not in label.full_text()
        page._shell.run_controls.stopButton.click()
        _terminal(qapp, page, observed, StandardEventKind.STOPPED)
