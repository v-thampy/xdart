"""Reviewer discriminator for Quick's real image-grouping cancellation boundary."""
import shutil

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog
from xrd_tools.core.geometry.camera import UPRIGHT_CAMERA
from xrd_tools.sources.point_images import inspect_point_images
from tests.xdart.scattering.test_calibration_quick_chain import (
    RING_ANGLES, _close, _configure, _page_with_dialog, _quick_card, _run,
    _starts, _stitch_pair, qapp,
)


def test_cancelled_quick_never_opens_a_grouping_confirmation(qapp, tmp_path, monkeypatch):
    """Q6: Cancel on the metadata job also stops its pixel-free grouping return.

    The real source gains two images per point after setup; the worker inspects
    those files itself. A watcher dismisses any actual modal so a regression
    fails at a visible popup assertion rather than hanging in its event loop.
    """
    pair = _stitch_pair(tmp_path / 'data')
    page, dialog = _page_with_dialog(tmp_path / 'project')
    watcher = QtCore.QTimer(dialog)
    shown = []

    def dismiss_grouping():
        popup = QtWidgets.QApplication.activeModalWidget()
        if isinstance(popup, PointImagesDialog):
            shown.append(popup.windowTitle())
            popup.reject()

    watcher.timeout.connect(dismiss_grouping)
    watcher.setInterval(5)
    try:
        card = _quick_card(dialog)
        _configure(dialog, card, pair, camera=UPRIGHT_CAMERA)
        count = len(RING_ANGLES)
        for index in range(count):
            shutil.copyfile(pair.ring.parent / f'ring_scan1_{index:04d}.raw',
                            pair.ring.parent / f'ring_scan1_{index + count:04d}.raw')
        detection = inspect_point_images(dialog._spec())
        assert (detection.point_count, detection.image_count) == (count, 2 * count)

        def cancel_metadata(kind):
            if kind == 'enumerate':
                dialog.cancel_button.click()

        started = _starts(dialog, monkeypatch, cancel_metadata)
        watcher.start()
        status = _run(dialog, card)
        assert shown == [], f'Cancel still opened grouping confirmation: {shown}'
        assert started.count('enumerate') == 1 and 'refine' not in started
        assert 'Cancelled' in status and 'Calibrant metadata' in card.status.text()
        assert dialog._distance_result is not None and dialog.fit is None
        assert dialog.point_images.choices == ()
    finally:
        watcher.stop()
        _close(qapp, page, dialog)
