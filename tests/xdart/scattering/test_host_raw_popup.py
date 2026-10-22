"""A host page's image in the view's own raw pop-up (Codex's Stitch/Int reuse audit R2a: Stitch's contributing
frame opened in a page-built dialog of its own). Int's own pop-up path (its Raw button, Int 1D) is pinned by
``test_e4_scientific_interactions.py`` and stays as it was."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
from tests.xdart.scattering.e3_shell_support import make_shell_projection


def test_a_host_raw_popup_keeps_its_image_through_the_views_own_reconciles_and_layouts() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = ScientificView()
    commands, closed = [], []
    view.commandRequested.connect(commands.append)
    view.rawPopupClosed.connect(lambda: closed.append(True))
    try:
        view.show_raw_popup("Stitch contributing frame")
        assert view.raw_popup_shown() and view.raw_popup_dialog.windowTitle() == "Stitch contributing frame"
        assert view.raw_popup_mode() == "thumbnail" and view.raw_popup_thumbnail.isChecked()
        image = np.arange(6, dtype=np.float32).reshape(2, 3)
        view.show_raw_image(image, status="2-0 · Thumbnail", color_map="Default", log_scale=False)
        np.testing.assert_array_equal(view.raw_popup_image.image.image, image.T[:, ::-1])
        assert view.raw_popup_status.text() == "2-0 · Thumbnail"
        # the host drives the view in Int 1D with no heavy frame: Int's reconcile leaves the host's image
        projection = make_shell_projection(frame_count=1, plot_mode="Single")
        view.reconcile(replace(projection.scientific, processing_mode="Int 1D", heavy=None),
                       projection.navigation, completed=1, total=1, detail="Ready")
        np.testing.assert_array_equal(view.raw_popup_image.image.image, image.T[:, ::-1])
        # a 2-D result's layout does not close it (Int's pop-up follows Int 1D only)
        view._apply_processing_layout("Int 2D")
        assert view.raw_popup_shown()
        view.raw_popup_full.click()  # the host reads its frame again in that mode
        assert view.raw_popup_mode() == "full"
        assert (commands[-1].kind, commands[-1].value, commands[-1].path) == (
            ShellCommandKind.SET_DETECTOR_MODE, "full", ("popup",))
        count = len(commands)
        view.show_raw_status("Loading selected frame…", clear_image=True)
        assert view.raw_popup_image.image.image is None
        view.raw_popup_dialog.close()
        app.processEvents()
        assert closed == [True] and len(commands) == count  # the host is told; Int's thumbnail is not asked
        assert view.raw_popup_status.text() == "" and view.raw_popup_image.canvas.raw_image.size == 0
        view.show_raw_image(image, status="ignored", color_map="Default", log_scale=False)  # closed: nothing
        assert view.raw_popup_image.image.image is None
    finally:
        view.close()
