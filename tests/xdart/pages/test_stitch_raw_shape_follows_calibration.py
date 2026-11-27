"""Stitch's Advanced Rows/Columns follow the Calibration's detector, as RSM's do. Maintainer, 2026-11-27, at the
beamline (RSM): "the detector was initially set to eiger2 500k. i changed it to pilatus 100k but it still didn't change
the number of rows and columns" and "i thought the rows and columns were supposed to update automatically based on the
detector and images". Stitch decodes the same headerless RAW with its own Rows/Columns (default 195 x 1475), and its
source resolver refused his Pilatus 100K frames with them the same way."""
import time

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.stitch.page import StitchPage
from tests.xdart.pages.test_rsm_eiger_source import _pilatus_raw_source


class _Status:
    def show(self, *_args):
        pass


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dispose(page, qapp, timeout=30):
    until = time.monotonic() + timeout
    receipt = page.shutdown()
    while receipt.status is PageCleanup.PENDING and time.monotonic() < until:
        qapp.processEvents()
        time.sleep(.005)
        receipt = page.shutdown()
    assert receipt.status is PageCleanup.CLEAN, receipt
    page.deleteLater()
    qapp.processEvents()


def _shape(page):
    return page._value("detector_rows"), page._value("detector_cols")


@pytest.mark.parametrize("partner", [False, True], ids=["pyfai-file-alone", "with-partner"])
def test_stitch_rows_columns_follow_the_calibration_chosen_saved_and_at_preview(qapp, tmp_path, partner):
    locator = _pilatus_raw_source(tmp_path, partner=partner)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.set_field("project_root", str(tmp_path))
        page.set_field("detector_rows", "512"); page.set_field("detector_cols", "1028")  # an Eiger's, left over
        page.set_field("geometry_path", str(tmp_path / locator))
        assert _shape(page) == ("195", "487")
        page.set_field("detector_rows", "512"); page.set_field("detector_cols", "1028")
        page._calibration_project = str(page._value("project_root"))
        page._adopt_calibration(str(tmp_path / locator), None)  # the window saves under the name the row holds
        assert _shape(page) == ("195", "487")
        page.set_field("detector_rows", "512"); page.set_field("detector_cols", "1028")
        page.begin_preview()  # a partner saved later under the same name: Preview reads it first
        assert _shape(page) == ("195", "487")
    finally:
        _dispose(page, qapp)
