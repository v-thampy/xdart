"""Committed SPEC sources qualify their real images before showing readiness."""
from pathlib import Path
from threading import get_ident
import time

import pytest

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.stitch.page import StitchPage
from tests.xdart.pages.test_stitch_page import _Status, _as_native, _form, qapp  # noqa: F401


def _settle(page, qapp):
    until = time.monotonic() + 5
    observer = getattr(page, "_source_observer", None)
    while observer is not None and observer.computing and time.monotonic() < until:
        qapp.processEvents()
        time.sleep(.005)
    qapp.processEvents()
    assert observer is None or not observer.computing


def _close(page, qapp):
    _settle(page, qapp)
    assert page.shutdown().status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize("problem", ["parent_folder", "frame_mismatch"])
def test_spec_readiness_admits_nested_images_and_names_refusals_off_thread(qapp, tmp_path, monkeypatch, problem):
    from xrd_tools.io import image
    form = _form(tmp_path, "multigeometry")
    missing = Path(form.image_dir) / "sample_scan1_0001.raw"
    original = missing.read_bytes()
    if problem == "frame_mismatch":
        missing.unlink()
    calls, gui_thread = [], get_ident()
    read_layout = image.read_detector_image_layout

    def observed_layout(*args, **kwargs):
        calls.append(get_ident())
        return read_layout(*args, **kwargs)

    monkeypatch.setattr(image, "read_detector_image_layout", observed_layout)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(_as_native(form))
        if problem == "parent_folder":
            page.set_field("image_dir", str(Path(form.spec_path).parent))
        assert page.controls.source_card.valid_marker.isHidden()
        assert "2 SPEC points" in page.controls.source_card.status.text()
        _settle(page, qapp)
        assert page.controls.source_card.valid_marker.isHidden()
        assert not page.run_bar.startButton.isEnabled()
        status = page.run_bar.readinessLabel.full_text()
        if problem == "parent_folder":
            assert "No detector images named" in status and str(Path(form.spec_path).parent) in status
            page.set_field("image_dir", str(form.image_dir))
        else:
            assert "2 metadata points but 1 image frames" in status
            missing.write_bytes(original)
            page.set_field("image_dir", str(form.image_dir))
        _settle(page, qapp)
        assert not page.controls.source_card.valid_marker.isHidden()
        assert page.run_bar.startButton.isEnabled(), page.run_bar.readinessLabel.full_text()
        assert "2 frames" in page.controls.source_card.status.text()
        assert calls and gui_thread not in calls
        before = len(calls)
        page.set_field("npt_1d", 80)
        page.set_field("image_dir", str(form.image_dir))
        _settle(page, qapp)
        assert len(calls) == before  # processing and unchanged source edits reuse admission
    finally:
        _close(page, qapp)


@pytest.mark.parametrize("change", ["browse", "typed"])
def test_spec_file_change_resets_scan_and_auto_stem_but_folder_change_preserves_them(qapp, tmp_path, change):
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, second = _form(first_root, "multigeometry"), _form(second_root, "multigeometry")
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(_as_native(first))
        page.set_field("scan", "2")
        page.set_field("image_stem", "custom_{scan}_")
        page.set_field("image_dir", str(second.image_dir))
        page.set_sources((first.spec_path,))
        assert page._value("scan") == "2" and page._value("image_stem") == "custom_{scan}_"
        if change == "browse":
            page.set_sources((second.spec_path,))
        else:
            page.set_field("spec_path", str(second.spec_path))
        assert page._value("scan") == "1"
        assert page._value("image_stem") == ""
        assert page._value("image_dir") == str(second.image_dir)
        _settle(page, qapp)
        assert page.run_bar.startButton.isEnabled(), page.run_bar.readinessLabel.full_text()
    finally:
        _close(page, qapp)
