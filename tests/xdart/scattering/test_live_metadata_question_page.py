"""Live Auto's one question per Run, on the page (maintainer, 2026-10-26: his pick "Once per Run"): a box that leaves
the page usable while the Run goes on, whose answer reaches the Run, and that closes itself once the Run settles
without it. The real page, executor and Live Run over real TIFF + .txt files."""
from __future__ import annotations

import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _poni, _pump, _terminal
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


@pytest.fixture
def asked_page(qapp, tmp_path, monkeypatch):  # noqa: F811
    """A Live Run under Auto on an empty folder, one image without its sidecar, and the page's question open."""
    monkeypatch.setattr(run_executor, "_METADATA_QUESTION_AFTER_S", .3)  # 10 s by default
    raw = tmp_path / "raw"
    raw.mkdir()
    intent = RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="auto"),
        poni_file=str(_poni(tmp_path)), project_root=str(raw), save_path=str(tmp_path / "processed"),
        output_mode="Overwrite", processing_mode="Int 1D", live_mode=True, max_cores=1,
        bai_1d_args={"npt": 8, "method": "numpy"})
    with _page(qapp, intent) as (page, _view, observed):
        page._shell.run_controls.startButton.click()
        _frame(raw, "scan", 1, "txt", 1.0, sidecar=False)
        assert _pump(qapp, lambda: page._metadata_question_dialog is not None, 30.), "the page never asked"
        yield page, raw, observed
        if not any(event.kind is StandardEventKind.STOPPED for event in observed):
            page._shell.run_controls.stopButton.click()
            _terminal(qapp, page, observed, StandardEventKind.STOPPED)


def test_the_question_leaves_the_page_usable_and_its_yes_reaches_the_run(qapp, asked_page):  # noqa: F811
    page, _raw, _observed = asked_page
    box = page._metadata_question_dialog
    assert box.isVisible() and not box.isModal(), "the question blocks the page"
    assert "scan_0001.tif" in box.text() and page._shell.run_controls.stopButton.isEnabled()
    box.button(QtWidgets.QMessageBox.StandardButton.Yes).click()
    assert _pump(qapp, lambda: page._metadata_question_dialog is None), "the box outlived its answer"
    run = page._run_executor._exact_run(page._lifecycle.active_run_identity)
    assert _pump(qapp, lambda: run.metadata_kind == "sidecar"), run.metadata_kind
    qapp.processEvents()
    assert page._metadata_question_dialog is None, "the page asked again after the answer"


def test_the_question_closes_itself_once_a_sidecar_is_found(qapp, asked_page):  # noqa: F811
    page, raw, _observed = asked_page
    box = page._metadata_question_dialog
    closed = []
    box.finished.connect(lambda _result: closed.append(box.clickedButton()))
    _frame(raw, "scan", 1, "txt", 1.0, image=False)
    assert _pump(qapp, lambda: page._metadata_question_dialog is None), "the box outlived the question"
    assert closed == [None], closed  # closed by the Run's settling, not by a click
    run = page._run_executor._exact_run(page._lifecycle.active_run_identity)
    assert run.metadata_kind == "sidecar" and run.metadata_answer is None, (run.metadata_kind, run.metadata_answer)
