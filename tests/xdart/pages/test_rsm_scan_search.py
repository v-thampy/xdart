"""C-16 of NEXT-1210-RESPONSIVENESS: RSM's scan search lists each image folder once, shows its count, and stops.

The maintainer, 2026-11-13 (verbatim), running RSM over every scan of a SPEC file of about 20,700 points: "when i
clicked "stop" it took a good 20s or more for the run to stop". Diagnosed (read-only, measured on a 20,700-image
folder): every Run searches the Source's scans again; with a blank Image Stem the search listed the image folder twice
per scan (about 0.94 s a listing: about 1.9 s a scan) and checked for Stop only once, before it began.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tests.xdart.pages.test_rsm_page import (  # noqa: F401
    _close, _page, _spec_scan, _synthetic_source, _wait, _with_source, qapp,
)


def _scans(root, count):
    """The RSM page's synthetic SPEC file with ``count`` two-point scans, two images each (a blank Image Stem)."""
    import tifffile
    spec = _synthetic_source(root)  # scans 1 and 2
    spec.write_text(spec.read_text() + "\n" + "\n".join(_spec_scan(scan, 9 + scan) for scan in range(3, count + 1)),
                    encoding="utf-8")
    images = root / "source" / "images"
    for scan in range(3, count + 1):
        for label in range(2):
            tifffile.imwrite(images / f"RSM_synth_scan{scan}_{label:04d}.tif",
                             np.arange(20, dtype=np.uint16).reshape(4, 5) + label)
    return spec, images


def test_the_scan_search_lists_the_image_folder_once(tmp_path, monkeypatch):
    from xrd_tools.analysis.module_transaction import MetadataColumnSelector
    from xrd_tools.analysis.rsm_values import RSMFrameSelector, rsm_member_groups
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection
    spec, images = _scans(tmp_path, 6)
    listed = []
    real = Path.glob

    def glob(self, pattern, *args, **kwargs):
        if Path(self) == images:
            listed.append(pattern)
        return real(self, pattern, *args, **kwargs)
    monkeypatch.setattr(Path, "glob", glob)
    motors = tuple((role, MetadataColumnSelector(role)) for role in ("mu", "eta", "chi", "phi", "nu", "del"))
    groups, skipped = rsm_member_groups(
        StitchSourceSelection("Spec", (str(spec),), stitch_all=True, image_dir=str(images)),
        frame_selector=RSMFrameSelector(0, 1, 1), detector_shape=(4, 5), raw_dtype="uint16", raw_header_skip=0,
        motor_selectors=motors)
    assert [[member.scan for member in group] for group in groups] == [[f"{scan}.1"] for scan in range(1, 7)]
    assert skipped == ()
    assert len(listed) == 1, f"the image folder was listed {len(listed)} times for 6 scans"


def _slow_scans(monkeypatch, seconds):
    """Each SPEC scan opened by the search takes ``seconds`` (as a slow share would)."""
    from xrd_tools.sources import spec as spec_module
    real = spec_module.SpecSource.__init__

    def slow(self, *args, **kwargs):
        time.sleep(seconds)
        real(self, *args, **kwargs)
    monkeypatch.setattr(spec_module.SpecSource, "__init__", slow)


def _searching(page, qapp, seconds):
    texts, until = [], time.monotonic() + seconds
    while time.monotonic() < until:
        qapp.processEvents()
        page._poll()
        texts.append(page.run_bar.readinessLabel.full_text())
        time.sleep(.01)
    return texts


def test_the_scan_search_shows_its_count(qapp, tmp_path, monkeypatch):
    page = _page(qapp)
    try:
        _scans(tmp_path, 10)
        _with_source(page, tmp_path)
        _wait(page, qapp, lambda: page._resolving is None and not page._owner.busy)
        _slow_scans(monkeypatch, .2)
        page.set_field("scan_mode", "All Scans")
        page.begin_preview()
        texts = _searching(page, qapp, 1.5)
        assert any(" of 10" in text for text in texts), texts[-1]
    finally:
        page.stop()
        _wait(page, qapp, lambda: page._resolving is None, seconds=30)
        _close(page, qapp)


def test_stop_during_the_scan_search_returns_promptly(qapp, tmp_path, monkeypatch):
    page = _page(qapp)
    try:
        _scans(tmp_path, 10)
        _with_source(page, tmp_path)
        _wait(page, qapp, lambda: page._resolving is None and not page._owner.busy)
        _slow_scans(monkeypatch, .3)
        page.set_field("scan_mode", "All Scans")
        page.begin_preview()
        _searching(page, qapp, .5)
        assert page._resolving is not None, "the search had already ended"
        started = time.monotonic()
        page.stop()
        _wait(page, qapp, lambda: page._resolving is None, seconds=30)
        took = time.monotonic() - started
        assert took < 1.5, f"Stop took {took:.1f} s during the scan search"
        assert "stopped" in page.run_bar.readinessLabel.full_text().lower()
        assert page.build_forms() == () or page.build_forms() == [], page.build_forms()
    finally:
        _close(page, qapp)
