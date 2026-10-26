"""The footer under the plots names the output NeXus file of the frame it shows (maintainer, 2026-10-26: "in the
bottom status bar where the file name is shown as it is processed also add [Output: <nexus file name>]"; "it doesn't
have to have the full path to the output file. just the name of the nexus file")."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.e3_shell_support import make_shell_projection
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey, StandardDisplayPayload
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection, ProgressProjection
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.core import FrameView
from xrd_tools.session.run_configuration import RunIntent

IMAGE = "LaB6_4deg_300mm_15s_10261256_0001.tif"


def _status(artifact, *, processing_mode="Int 2D", notice=""):
    base = make_shell_projection(frame_count=1, selected_index=0, plot_mode="Single")
    frame = DisplayFrameKey(RunIdentity(7, "footer"), "LaB6_4deg_300mm_15s_10261256", artifact, 1, 1)
    navigation = FrameNavigationProjection(frames=(frame,), current=frame, selected=(frame,))
    payload = StandardDisplayPayload(
        0, frame, "Standard · LaB6_4deg_300mm_15s_10261256 · frame 1",
        FrameView(1, raw=np.ones((2, 3)), source_path=f"/raw/{IMAGE}", source_frame_index=0), "running")
    projected = ContextProjection().build_shell(
        revision=2, controls=base.controls, controls_readiness=base.controls_readiness, phase=RunPhase.RUNNING,
        intent=RunIntent(processing_mode=processing_mode), contexts=(), selection=None, navigation=navigation,
        payloads=(payload,), resident_frames=frozenset({frame}), progress=ProgressProjection(1, 1, ""),
        preferences=ScientificPreferences(plot_mode="Single"), browser_directory="", date_sorted=False,
        auto_last=True, executor_available=True, start_permitted=True, start_blocker="", notice=notice)
    return projected.scientific


def test_the_footer_names_the_output_nexus_file_by_its_name_only():
    shown = _status("/beamline/xdart_processed_data/LaB6_4deg_300mm_15s_10261256_int2d.nexus")
    assert shown.status == f"{IMAGE} [Output: LaB6_4deg_300mm_15s_10261256_int2d.nexus]"
    assert "xdart_processed_data" not in shown.status
    assert shown.title == IMAGE  # the plot title keeps the image alone


@pytest.mark.parametrize("artifact, mode", [
    ("/beamline/xdart_processed_data/scan_int1d.nexus", "Int 1D (XYE)"),  # writes .xye files, no NeXus output
    ("viewer-2d", "Int 2D"),  # not an output file
    ("/beamline/xdart_processed_data/scan_0001.xye", "Int 1D"),
])
def test_the_footer_names_no_output_for_a_frame_without_a_nexus_file(artifact, mode):
    assert _status(artifact, processing_mode=mode).status == IMAGE


def test_a_notice_still_takes_the_footer():
    notice = "Run not started: the Save Path folder name may not contain a dot"
    assert _status("/out/scan_int2d.nexus", notice=notice).status == notice


def test_a_long_footer_status_never_widens_the_view(qapp):  # noqa: F811
    """Measured before the size setting: an 88-character status raised the view's minimum width from 420 to 767 px."""
    from xdart.gui.tabs.scattering.scientific_view import ScientificView

    view = ScientificView()
    try:
        view.resize(900, 600)
        view.show()
        widths = []
        for text in ("", IMAGE, f"{IMAGE} [Output: LaB6_4deg_300mm_15s_10261256_int2d.nexus]"):
            view.status.setText(text)
            qapp.processEvents()
            widths.append(view.minimumSizeHint().width())
        assert widths[0] == widths[1] == widths[2], widths
        assert view.status.text().endswith("_int2d.nexus]")  # the text itself is kept whole
    finally:
        view.close()
        view.deleteLater()


def test_a_clipped_footer_exposes_its_current_full_text_in_the_tooltip(qapp):
    from xdart.gui.tabs.scattering.scientific_view import ScientificView

    view = ScientificView()
    try:
        text = f"{IMAGE} [Output: LaB6_4deg_300mm_15s_10261256_int2d.nexus]"
        assert view.reconcile_operation_status(text)
        assert text in view.status.toolTip()
        assert view.reconcile_operation_status("Replacement committed")
        assert "Replacement committed" in view.status.toolTip() and IMAGE not in view.status.toolTip()
    finally:
        view.close()
        view.deleteLater()


def test_the_2d_viewer_clear_also_replaces_the_footer_tooltip(qapp):  # noqa: F811
    """The ninth footer write, in clear_viewer_2d's chrome reset, goes through the same setter (Codex N-9a2596d5 routed
    the other eight): a failed clear says so in the text and the tooltip, and the old file name leaves both."""
    from xdart.gui.tabs.scattering.scientific_view import ScientificView

    view = ScientificView()
    try:
        assert view.reconcile_operation_status(f"{IMAGE} [Output: LaB6_4deg_300mm_15s_10261256_int2d.nexus]")
        view.clear_viewer_2d(None, failure=True)
        assert view.status.text() == "2D Viewer · Render failed; retry available."
        assert "2D Viewer · Render failed" in view.status.toolTip() and IMAGE not in view.status.toolTip()
    finally:
        view.close()
        view.deleteLater()

