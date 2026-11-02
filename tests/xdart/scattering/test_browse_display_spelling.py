"""Browse keeps file spelling; its legend keeps family spelling without the operation suffix.

Real Runs exercise both native paths (all platforms) and Windows-like case folding (on a case-insensitive
filesystem). The maintainer separately asked on 2026-10-26 that ``int1d`` never appear in a 1-D legend.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _browsing, _page, _pump, _run, _series
from xdart.gui.tabs.scattering import shell_projection, shell_values
from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest


@pytest.fixture
def qapp(_xdart_qt_harness):
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance()


@pytest.fixture
def folded(monkeypatch, tmp_path, request):
    """``os.path.normcase`` as on Windows: it folds case (each module that imported it by name too)."""
    if not request.param:
        return os.path.normcase
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


@pytest.mark.parametrize("folded", (False, True), indirect=True, ids=("native-path", "folded-path"))
def test_after_a_run_browse_names_the_scan_as_its_file_is_spelled(qapp, tmp_path, folded) -> None:
    intent = _series(tmp_path / "Baniecki_Oct2026", (16.0, 16.0), mode="Int 1D")
    intent.save_path = str(tmp_path / "Baniecki_Oct2026" / "Effect_of_Rocking.nexus")
    with _page(qapp, intent) as (page, _scientific, events):
        terminal = _run(qapp, page, events)
        written = Path(terminal.artifact)
        assert written.name == "Effect_of_Rocking_int1d.nexus" and written.is_file()
        browse = _browsing(qapp, page)
        assert browse.requested_path == folded(str(written))  # normalized for comparison only
        assert browse.scan_key == "Effect_of_Rocking_int1d"  # named as the file is spelled
        def drawn():  # the trace hydrates after Browse adopts the file; reading it at once raced on Linux CI
            page._refresh_shell()
            return page._last_scientific_projection.traces
        assert _pump(qapp, drawn, seconds=30.), "Browse never drew the trace"
        titles = [trace.title for trace in page._last_scientific_projection.traces]
        # A Browse scan is named by its file stem, while a curve uses the persisted family, as during Run.
        assert titles == ["Effect_of_Rocking_2"], titles


def test_a_browse_request_spelling_must_fold_to_its_path() -> None:
    BrowseLoadRequest("browse", 1, "/data/scan.nexus", display_path="/data/scan.nexus")
    with pytest.raises(TypeError):
        BrowseLoadRequest("browse", 1, "/data/scan.nexus", display_path="/data/other.nexus")
