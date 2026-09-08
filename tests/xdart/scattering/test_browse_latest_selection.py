"""Rapid scan gestures use the real page, loader and persisted readers."""

from pathlib import Path
from shutil import copy2
from threading import Event

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.xdart.scattering.test_p3_experiment_operation_composition import _close
from tests.xdart.scattering.test_p34_reintegrate_operation import _loaded_page, _wait
from xdart.gui.tabs.scattering.events import CleanupStatus
from xrd_tools.io import FrameViewReader


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def burst(tmp_path, monkeypatch, qapp):
    page, _store, seed, original = _loaded_page(tmp_path, monkeypatch, qapp)
    paths = tuple(
        seed.target.with_name(f"browse-{name}.nexus")
        for name in ("active", "skipped", "latest", "newest")
    )
    for path in paths:
        copy2(seed.target, path)
    entered, release_read, closing, release_close = (Event() for _ in range(4))
    opened, closed = [], []

    class GatedReader:
        # Delay a real reader at its I/O boundary; no loader/controller fakes.
        def __init__(self, source, **kwargs):
            self.path = Path(source)
            self.reader = FrameViewReader(source, **kwargs)

        def __enter__(self):
            assert self.reader.__enter__() is self.reader
            opened.append(self.path)
            return self

        def read_scalar_catalog(self, *, cancelled):
            if self.path == paths[0]:
                entered.set()
                assert release_read.wait(10)
            return self.reader.read_scalar_catalog(cancelled=cancelled)

        def __exit__(self, *args):
            if self.path == paths[0]:
                closing.set()
                assert release_close.wait(10)
            self.reader.__exit__(*args)
            closed.append(self.path)

    loader = page._context_controller._browse_loader
    monkeypatch.setattr(loader, "_open_reader", GatedReader)
    page.resize(1200, 800)
    page.show()
    page._set_browser_directory(str(seed.target.parent), explicit=True)
    page._refresh_shell()
    scans = page._shell.browser.scans

    def initial_painted():
        page._drain_executor()
        state = page._last_scientific_projection
        owner = page._context_controller._browse_hydration_owner
        return state if (state is not None
                         and state.browse_trace_snapshot is not None
                         and not page._scientific_repaint_pending
                         and not owner.polling_needed()) else None

    _wait(initial_painted)

    def item_for(path):
        return next((scans.item(i) for i in range(scans.count())
                     if scans.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                     == str(path)), None)

    def select(path):
        item = _wait(lambda: item_for(path))
        scans.scrollToItem(item)
        QtTest.QTest.mouseClick(
            scans.viewport(), QtCore.Qt.MouseButton.LeftButton,
            pos=scans.visualItemRect(item).center(),
        )
        qapp.processEvents()
        request = page._context_controller._browse_request
        assert request is not None and request.source_path == str(path)
        return request

    try:
        yield (page, loader, original, paths, select, entered, release_read,
               closing, release_close, opened, closed)
    finally:
        release_read.set()
        release_close.set()

        def cleanup():
            receipt = page.close_workspace()
            return receipt if receipt.cleanup_status is CleanupStatus.CLEANED else None

        _wait(cleanup)
        _close(page, qapp)


@pytest.mark.parametrize("terminal", ("adopt", "cancel", "close"))
def test_rapid_browse_keeps_only_latest_after_real_cleanup(burst, terminal, caplog):
    (page, loader, original, paths, select, entered, release_read,
     closing, release_close, opened, closed) = burst
    controller = page._context_controller
    old_selection = controller.selection
    old_navigation = controller.navigation
    active = select(paths[0])
    assert entered.wait(5)
    worker = loader._worker
    assert worker is not None and worker.is_alive()
    skipped = select(paths[1])
    latest = select(paths[2])

    assert loader.owns_request(active)
    assert not loader.owns_request(skipped)
    assert loader.owns_request(latest)
    assert loader.poll(skipped) is None
    # A stale cancellation must not cancel the new queued selection.
    assert loader.cancel(skipped).cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert not loader._queued.cancelled.is_set()
    assert opened == [paths[0]] and closed == []
    assert original.released
    assert controller.selection is old_selection
    assert controller.navigation is old_navigation

    release_read.set()
    assert closing.wait(5)
    # Selections may continue while the active reader is actually closing.
    newest = select(paths[3])
    assert not loader.owns_request(latest)
    assert controller._runtime._pending_replacement.request is newest
    assert controller.selection is old_selection
    assert controller.navigation is old_navigation
    assert opened == [paths[0]] and closed == []

    if terminal == "cancel":
        receipt = loader.cancel(newest)
    elif terminal == "close":
        receipt = page.close_workspace()
    if terminal != "adopt":
        browse_receipt = receipt if terminal == "cancel" else controller._close
        assert browse_receipt.request is newest
        assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
        # Explicit cleanup still blocks admission; it is not superseded.
        with pytest.raises(RuntimeError, match="closing|cleanup|lifecycle"):
            controller.begin_browse(str(paths[1]))

    release_close.set()
    worker.join(5)
    assert not worker.is_alive()
    assert closed == [paths[0]]
    if terminal == "adopt":
        def settled():
            page._drain_executor()
            context = controller.browse_context
            return context if (context is not None
                               and context.load_request is newest
                               and not page._scientific_repaint_pending) else None

        context = _wait(settled)
        assert context.loaded_labels == (2, 5, 9)
        assert controller.selection.names(context)
        assert controller.navigation.current.artifact == str(paths[3])
        assert (page._shell.scientific.frame_selector.currentData()
                is controller.navigation.current)
        selected = page._shell.browser.scans.selectedItems()
        assert [item.data(QtCore.Qt.ItemDataRole.UserRole)
                for item in selected] == [str(paths[3])]
        assert page._shell.scientific.curve.listDataItems()
        assert opened == closed == [paths[0], paths[3]]
    else:
        receipt = (loader.cancel(newest) if terminal == "cancel"
                   else page.close_workspace())
        browse_receipt = receipt if terminal == "cancel" else controller._close
        assert browse_receipt.request is newest
        assert receipt.cleanup_status is CleanupStatus.CLEANED
        assert loader._active is None and loader._queued is None
        assert opened == closed == [paths[0]]
    assert "Browse refused" not in caplog.text
