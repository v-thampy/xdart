"""A running Live Run that has processed everything it found says so (maintainer, 2026-10-26: "make it clear in the
status when all scans/frames have been processed in a live run with a brief message implying that xdart is waiting for
new frames", his pick "Lead with it": "Running · Up to date, waiting for new images · 561 processed (561 new) · 0
skipped · 561 discovered"). The real page, executor and Live Run over real TIFF + .txt files."""
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

UP_TO_DATE = "Up to date, waiting for new images"


def test_the_status_leads_with_up_to_date_only_while_a_running_live_run_has_nothing_pending():
    caught_up, behind = DirectoryFileProgress(561, 0, 0, 561, 561), DirectoryFileProgress(537, 0, 24, 561, 537)
    assert caught_up.text("Running", watching=True) == (
        "Running · Up to date, waiting for new images · 561 processed (561 new) · 0 skipped · 561 discovered")
    assert behind.text("Running", watching=True) == behind.text("Running")
    assert UP_TO_DATE not in caught_up.text("Paused")  # only a running Live Run watches


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
        assert label.full_text() == ("Running · Up to date, waiting for new images · 1 processed · 0 skipped · "
                                     "1 discovered"), label.full_text()
        assert label.text().startswith("Running · Up to date, waiting for new images")  # leads, however it elides
        _frame(raw, "scan", 2, "txt", 2.0, sidecar=False)  # a new image, its .txt not yet there: pending again
        assert _pump(qapp, lambda: "1 pending" in label.full_text(), 30.), label.full_text()
        assert UP_TO_DATE not in label.full_text()
        page._shell.run_controls.stopButton.click()
        _terminal(qapp, page, observed, StandardEventKind.STOPPED)
