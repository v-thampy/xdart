"""R4-D source-mode ownership reproducer and desired count contract."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _write_real_tiff_series(directory, *, count=5):
    """Write a genuine single-page TIFF series without detector-sized payloads."""
    tifffile = pytest.importorskip("tifffile")
    directory.mkdir()
    paths = []
    for index in range(1, count + 1):
        path = directory / f"series_{index:04d}.tif"
        tifffile.imwrite(
            path,
            np.full((8, 8), index, dtype=np.uint16),
        )
        paths.append(path)
    return tuple(paths)


def test_r4d_nxs_directory_to_tiff_series_uses_selected_file_source_spec(
    qapp, monkeypatch, tmp_path,
):
    """A previous source mode's hidden extension must not define TIFF count."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))

    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    nxs_dir = tmp_path / "nxs"
    nxs_dir.mkdir()
    _write_bluesky_nxwriter(nxs_dir / "scan_00001.nxs", n=2)
    tiff_paths = _write_real_tiff_series(tmp_path / "tiff", count=5)

    widget = staticWidget()
    try:
        signal = widget.wrangler.parameters.child("Signal")

        # Match the production cross-mode sequence: configure an NXS directory,
        # then choose the fourth member of a TIFF Image Series. The mode switch
        # hides img_ext but intentionally leaves this captured stale value in
        # place so the desired typed-SourceSpec contract is exercised.
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_dir").setValue(str(nxs_dir))
        signal.child("img_ext").setValue("nxs")
        assert signal.child("img_ext").value() == "nxs"

        signal.child("inp_type").setValue("Image Series")
        signal.child("File").setValue(str(tiff_paths[3]))
        qapp.processEvents()

        # Production setup has understood the selected file, while the hidden
        # compatibility field still describes the previous mode.
        assert widget.wrangler.img_ext == "tif"
        assert signal.child("img_ext").value() == "nxs"

        count = widget._controls_v2_source_frame_count()
        source_spec = widget._controls_v2_freeze_source_spec()
        state = widget._controls_v2_state()
        widget._refresh_controls_v2_profile_now()
        source_status = widget.controls_v2.source_card.status.text()

        # Current captured result at 41292079 is
        # (1, 1, "1 frames · Image Series").
        assert (count, state.frame_count, source_status) == (
            5,
            5,
            "5 frames · Image Series",
        )
        assert tuple(source_spec.options["files"]) == tuple(
            str(path) for path in tiff_paths)
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()
