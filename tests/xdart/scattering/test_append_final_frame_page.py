"""The end of a finite directory Append shows its last scan (maintainer, typed in the Codex thread 2026-10-23: "...
and non-live append mode not showing the last frame processed at the end of the processing"; Codex review
N-0243a1bb, third priority).

Beamline 2026-10-23 (dev 305d77fb and 4c28033b, Windows share, no typed filter): an Append that showed saved scans,
reduced new ones, then showed a last saved scan ended with every plot empty and "Refused: Browse 1-D navigation is
incomplete", Auto Last on that last scan. Real page, real Run click, real files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.xdart.scattering.test_append_already_processed import _intent, _run as _seed, _scan
from tests.xdart.scattering.test_browse_reference_rings_run import _browsing, _page, _pump, _run
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xrd_tools.io import FrameViewReader

_SAVED = {"saved-new-saved": ("scana", "scanc"), "only-saved": ("scana", "scanb", "scanc"), "only-new": ()}


@pytest.mark.parametrize("shape", tuple(_SAVED))
def test_a_finite_append_ends_showing_its_last_scan(qapp, tmp_path, shape):
    raw = tmp_path / "raw"
    raw.mkdir()
    for stem in _SAVED[shape]:
        _scan(raw, stem, 2)
    if _SAVED[shape]:
        events, _catalog = _seed(tmp_path, raw, 9401)
        assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    for stem in ("scana", "scanb", "scanc"):
        if stem not in _SAVED[shape]:
            _scan(raw, stem, 2)
    with _page(qapp, _intent(tmp_path, raw)) as (page, scientific, observed):
        terminal = _run(qapp, page, observed)
        controller = page._context_controller

        def facts():
            context, navigation = controller.browse_context, controller.navigation
            return (shape, Path(str(terminal.artifact)).name, controller.browse_pending,
                    None if context is None else (Path(str(context.requested_path)).name, context.loaded,
                                                  context.frame_ids),
                    None if navigation is None else (navigation.current, navigation.selected),
                    page._processed_browser.terminal_handoff, page._notice_text)
        _pump(qapp, lambda: not controller.browse_pending and controller.browse_context is not None
              and controller.browse_context.loaded, 30.)
        assert not controller.browse_pending and controller.browse_context is not None \
            and controller.browse_context.loaded, facts()
        browse = _browsing(qapp, page)
        navigation = controller.navigation
        assert Path(str(browse.requested_path)).name == "scanc_int1d.nexus", facts()
        assert navigation.current is not None and navigation.selected, facts()
        current = navigation.current
        with FrameViewReader(Path(str(browse.requested_path))) as reader:
            saved = reader.read(current.local_frame_label)
        assert _pump(qapp, lambda: (
            bool(scientific.curve.listDataItems())
            and page._last_scientific_projection is not None
            and bool(page._last_scientific_projection.traces)
            and page._last_scientific_projection.traces[0].frame is current
        ), 20.), facts()
        curve = scientific.curve.listDataItems()[0]
        np.testing.assert_allclose(curve.getData()[1], saved.intensity_1d, rtol=1e-6)
        assert "navigation is incomplete" not in str(page._notice_text), facts()
