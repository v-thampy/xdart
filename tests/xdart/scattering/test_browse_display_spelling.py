"""Browse names a scan in its own spelling while it compares paths case-folded, as Windows folds them (maintainer,
2026-10-25: after a Run on his share the legend and Browse read ``effect_of_rocking_…`` for the file
``Effect_of_Rocking_…``). Real Runs and the real page; ``os.path.normcase`` folds case here as it does on Windows,
which needs a case-insensitive file system (macOS, Windows)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _browsing, _page, _run, _series
from xdart.gui.tabs.scattering import shell_projection, shell_values
from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest


@pytest.fixture
def qapp(_xdart_qt_harness):
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance()


@pytest.fixture
def folded(monkeypatch, tmp_path):
    """``os.path.normcase`` as on Windows: it folds case (each module that imported it by name too)."""
    probe = tmp_path / "CaseProbe"
    probe.write_text("")
    insensitive = (tmp_path / "caseprobe").exists()
    probe.unlink()
    if not insensitive:
        pytest.skip("folding paths needs a case-insensitive file system")
    real = os.path.normcase

    def fold(value):
        return real(value).lower()

    for owner in (os.path, shell_projection, shell_values):
        monkeypatch.setattr(owner, "normcase", fold)
    return fold


def test_after_a_run_browse_names_the_scan_as_its_file_is_spelled(qapp, tmp_path, folded) -> None:
    intent = _series(tmp_path / "Baniecki_Oct2026", (16.0, 16.0), mode="Int 1D")
    intent.save_path = str(tmp_path / "Baniecki_Oct2026" / "Effect_of_Rocking.nexus")
    with _page(qapp, intent) as (page, _scientific, events):
        terminal = _run(qapp, page, events)
        written = Path(terminal.artifact)
        assert written.name == "Effect_of_Rocking_int1d.nexus" and written.is_file()
        browse = _browsing(qapp, page)
        assert browse.requested_path == folded(str(written))  # compared folded, as before
        assert browse.scan_key == "Effect_of_Rocking_int1d"  # named as the file is spelled
        projection = page._last_scientific_projection
        titles = [trace.title for trace in projection.traces]
        assert titles and all(title.startswith("Effect_of_Rocking_int1d") for title in titles), titles


def test_a_browse_request_spelling_must_fold_to_its_path() -> None:
    BrowseLoadRequest("browse", 1, "/data/scan.nexus", display_path="/data/scan.nexus")
    with pytest.raises(TypeError):
        BrowseLoadRequest("browse", 1, "/data/scan.nexus", display_path="/data/other.nexus")
