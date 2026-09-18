"""GI-COMPANION-20260918 — the three Processing 2D Axis choices, through the page.

A real Run from the real workspace page (Start button, real executor, real
writer), then the written file and the loaded Browse state are inspected.
"""

from __future__ import annotations

import time

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets
import tifffile

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_output_graph import _intent, _written
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.controls_inventory import INT_2D_AXIS
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.scientific_axes import DERIVED_Q_CHI
from xrd_tools.io import read_frame_record
from xrd_tools.session.intent_store import RunIntentStore

FRAMES = 4


def _page(tmp_path, choice: str):
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:195, :487]
    for label in range(1, FRAMES + 1):
        image = 200 + 80 * np.sin(xx / 11.0 + label) + 60 * np.cos(yy / 7.0)
        tifffile.imwrite(
            tmp_path / f"scan_{label:04d}.tif",
            (image + rng.poisson(5, image.shape)).astype(np.uint16),
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    target = tmp_path / "result.nexus"
    intent = _intent(tmp_path / "scan_0001.tif", target, poni, processing_mode="Int 2D")
    intent.bai_2d_args = {"npt_rad": 40, "npt_azim": 30, "method": "no"}
    intent.gi.enabled = True
    intent.gi.incidence_motor = "Manual"
    intent.gi.th_val = 0.2
    page = ScatteringWorkspace(
        intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(join_timeout=2.0),
    )
    page._on_field_value(INT_2D_AXIS, choice)
    return page, _written(target, "Int 2D")


def _wait(page, predicate, timeout=60.0):
    app = QtWidgets.QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail(page._notice_text or "the run did not settle")


def _run(page):
    _wait(page, page._shell.run_controls.startButton.isEnabled)
    page._shell.run_controls.startButton.click()
    _wait(page, lambda: page._capture_current_loaded_browse() is not None
          and len(page._capture_current_loaded_browse().labels) == FRAMES
          and page._start_permitted()[0])


def _close(page):
    _wait(page, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
    page.deleteLater()
    QtWidgets.QApplication.instance().processEvents()


@pytest.mark.parametrize(("choice", "primary", "stored", "offered"), (
    ("Qip-Qoop", "qip_qoop", ("qip_qoop",), ("qip_qoop", DERIVED_Q_CHI)),
    ("Q-χ", "q_chi", ("q_chi",), ("q_chi",)),
    ("Q-χ + Qip-Qoop", "qip_qoop", ("qip_qoop", "q_chi"), ("qip_qoop", "q_chi")),
))
def test_each_processing_choice_stores_and_offers_exactly_its_maps(
    tmp_path, choice, primary, stored, offered,
):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page, artifact = _page(tmp_path, choice)
    try:
        _run(page)
        with h5py.File(artifact, "r") as handle:
            top = handle["entry/integrated_2d"]
            assert top.attrs["primary_mode"] == primary
            assert [name for name in top if isinstance(top[name], h5py.Group)] == list(stored[1:])
            for group in (top, *(top[name] for name in stored[1:])):
                # One physical frame, once, in every map.
                assert list(group["frame_index"][()]) == list(range(1, FRAMES + 1))
            assert len(handle["entry/frames"]) == FRAMES
        record = read_frame_record(artifact, FRAMES)
        assert record.modes_2d == stored and record.active_mode_2d == primary

        # The pane offers exactly the maps this frame has, showing the primary.
        pane = page._shell.scientific
        combo = pane.image_axis
        _wait(page, lambda: tuple(
            combo.itemData(i) for i in range(combo.count())
        ) == offered)
        assert combo.currentData() == primary
        assert combo.isEnabled() is (len(offered) > 1)
        if len(offered) == 1:
            return

        # Choosing another map is display only: same file bytes, same intent,
        # no new run -- and the choice survives without the raw images.
        before = artifact.read_bytes()
        revision = page._intents.snapshot().revision
        for raw in tmp_path.glob("scan_*.tif"):
            raw.unlink()
        other = offered[1]
        combo.setCurrentIndex(1)
        combo.activated.emit(1)
        _wait(page, lambda: pane.rendered_image_axis == other)
        assert combo.currentData() == other
        # I–χ comes from the shown q–χ map although the 1-D result is q_total;
        # plain I–Q stays the direct 1-D integration and is not relabelled.
        labels = [pane.plot_axis.itemText(i) for i in range(pane.plot_axis.count())]
        assert labels == [
            "Q (Å⁻¹)", "2θ (°)",
            "χGI (°, derived)" if other == DERIVED_Q_CHI else "χGI (°)",
        ]
        assert page._intents.snapshot().revision == revision
        assert page._start_permitted()[0]
        assert artifact.read_bytes() == before
    finally:
        _close(page)
