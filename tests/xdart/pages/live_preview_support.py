"""Ports for the mounted Live preview and the three v3 preview-off controls."""

from importlib.util import find_spec

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.pages.live_per_scan_support import start_live, wait_until


class MissingPreviewCapability(AssertionError):
    """A running native Live page has no mounted per-frame preview owner."""


PREVIEW_OWNER_MODULE = "xdart.gui.widgets.live_preview"

unimplemented_preview = pytest.mark.xfail(
    find_spec(PREVIEW_OWNER_MODULE) is None,
    reason="the implementation base mounts per-scan Live without a per-frame preview owner",
    raises=MissingPreviewCapability,
    strict=True,
)


def preview_checkbox(page):
    checkbox = page.advanced.findChild(QtWidgets.QCheckBox, "livePreviewCheck")
    assert checkbox is not None, "Advanced does not mount the shared Live preview setting"
    return checkbox


def disable_preview(page):
    if find_spec(PREVIEW_OWNER_MODULE) is None:
        assert page.advanced.findChild(QtWidgets.QCheckBox, "livePreviewCheck") is None
        return  # The untouched base has only the v3 path.
    assert page.advanced_button.isEnabled()
    page.advanced_button.click()
    assert page.advanced.isVisible()
    checkbox = preview_checkbox(page)
    assert checkbox.isEnabled() and checkbox.toolTip()
    if checkbox.isChecked():
        checkbox.click()
    assert not checkbox.isChecked()
    page.advanced.close()


def start_preview(page, qapp):
    watcher = start_live(page)
    wait_until(qapp, lambda: not watcher.polling)
    assert watcher.watching and page.run_bar.stopButton.isEnabled()
    if find_spec(PREVIEW_OWNER_MODULE) is None and not hasattr(page, "_live_preview"):
        raise MissingPreviewCapability(f"{type(page).__name__} runs Live without a preview owner")
    owner = page._live_preview
    assert owner.watching and owner.clock is watcher.clock
    return owner


def advance_preview(page, owner, qapp, seconds=2.0, *, drain=True):
    """Offer one controlled discovery/display tick through the production owners."""
    watcher = page._live_watcher
    wait_until(qapp, lambda: not watcher.polling)
    watcher.clock.advance(seconds)
    watcher.poll_due()
    wait_until(qapp, lambda: not watcher.polling)
    owner.poll_due()
    if drain:
        wait_until(qapp, lambda: not owner.working)
        owner.poll_due()
        qapp.processEvents()
