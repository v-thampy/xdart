"""NEXT-1217-EIGER-SOURCES in the RSM page. Maintainer, 2026-11-20: "with scan 2 i get this" (Preview refused:
RSM_MEMBER_IDENTITY_MISMATCH), his picks "Read shape from file (Recommended)" and "Source card flags imageless scans",
and "the Advanced Rows/columns should update as well when the image source changes". A SPEC + Eiger member takes the
Eiger's own frame shape and the Advanced Rows/Columns follow it; a real mismatch with the calibration still refuses;
the Source card names a scan without images. Real page, real resolver and Preview, synthetic DECTRIS masters."""
from pathlib import Path
import time

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_container_descriptor import _vendor_eiger_master
from xdart.gui.tabs.scattering.controls_inventory import THRESHOLD_ENABLED
from tests.xdart.pages.test_rsm_page import (
    _canonical_partner, _close, _page, _spec_file, _spec_scan, _wait,
)
from tests.xdart.scattering.test_lt_data_format import _choose


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _eiger_source(root, shape):
    """Three SPEC scans; scans 1 and 2 have DECTRIS masters named as the detector names them (zero padded), scan 3
    none. The calibration is the canonical partner (a 195 x 487 detector)."""
    _canonical_partner(root)
    images = root / "source" / "images"
    images.mkdir(parents=True)
    _spec_file(root / "source" / "RSM_synth", [_spec_scan(scan, 9 + scan) for scan in (1, 2, 3)])
    for scan in (1, 2):
        _vendor_eiger_master(images / f"RSM_synth_scan{scan:03d}_master.h5",
                             [images / f"RSM_synth_scan{scan:03d}_data_000001.h5"],
                             per_seg=(2, *shape), chunks=(1, *shape))


def _on_source(page, qapp, root, scans, *, rows=None):
    _choose(qapp, page._rows["data_format"], "spec_container")
    page.set_field("project_root", str(root))
    page.set_field("geometry_path", "calibration/rsm_canonical.json")
    page.set_sources([str(root / "source" / "RSM_synth")])
    page.set_field("image_dir", "source/images")
    page.set_field("normalization", "None")
    page.set_field("offset", "0")
    page._threshold_edited(THRESHOLD_ENABLED, False)
    page.set_field("static_hot", "")
    if rows is not None:  # Rows/Columns left from another detector
        page.set_field("adv_rows", str(rows[0])); page.set_field("adv_columns", str(rows[1]))
    if scans == "all":
        page.set_field("scan_mode", "All Scans")
    else:
        page.set_field("scan", scans)


def _preview(page, qapp, seconds=60.0):
    page.begin_preview()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents(); page._poll()
        text = page.run_bar.readinessLabel.full_text()
        if "Preview ready" in text or "refused" in text.lower():
            return text
        time.sleep(0.005)
    raise AssertionError(page.run_bar.readinessLabel.full_text())


def test_a_spec_eiger_member_takes_the_eigers_shape_and_rows_columns_follow(qapp, tmp_path):
    _eiger_source(tmp_path, (195, 487))  # the calibrated detector's shape
    page = _page(qapp)
    try:
        _on_source(page, qapp, tmp_path, "2", rows=(4, 5))
        groups, _skipped = page.resolve_source()
        assert [member.detector_shape for group in groups for member in group] == [(195, 487)]
        assert (page._value("adv_rows"), page._value("adv_columns")) == ("195", "487")
        assert "Preview ready" in _preview(page, qapp)
    finally:
        _close(page, qapp)


def test_an_eiger_that_differs_from_the_calibration_still_refuses(qapp, tmp_path):
    _eiger_source(tmp_path, (4, 5))  # Rows/Columns keep the preset's 195 x 487, the calibration's shape
    page = _page(qapp)
    try:
        _on_source(page, qapp, tmp_path, "2")
        refusal = _preview(page, qapp)
        assert "RSM_MEMBER_IDENTITY_MISMATCH" in refusal and "4×5" in refusal and "195×487" in refusal, refusal
        assert (page._value("adv_rows"), page._value("adv_columns")) == ("4", "5")
    finally:
        _close(page, qapp)


@pytest.mark.parametrize("scans, missing_ub", [("3", False), ("all", False), ("all", True)])
def test_the_source_card_names_a_scan_without_images(qapp, tmp_path, scans, missing_ub):
    _eiger_source(tmp_path, (195, 487))
    if missing_ub:
        spec = tmp_path / "source" / "RSM_synth"
        spec.write_text(spec.read_text(encoding="utf-8").replace("#G3 1 0 0 0 1 0 0 0 1\n", ""), encoding="utf-8")
    page = _page(qapp)
    try:
        _on_source(page, qapp, tmp_path, scans)
        groups, skipped = page.resolve_source()
        assert [key for _path, key, _why in skipped] == ["3.1"]
        assert len(groups) == (0 if scans == "3" else 2)
        header = page._headers["source"]
        assert not header.ready and "no images" in header.text and "scan 3" in header.text, header
        assert "RSM_synth_scan3_" in header.detail
        if missing_ub:
            assert "Scans 1 and 2 record no UB matrix" in header.detail
        assert page.controls.source_card.status.text() == header.text
    finally:
        _close(page, qapp)


def test_rows_columns_still_drive_a_raw_free_tiff_source(qapp, tmp_path):
    """Images without their own container (the synthetic TIFFs here) keep the Advanced Rows/Columns as before."""
    import tifffile
    _canonical_partner(tmp_path)
    images = tmp_path / "source" / "images"
    images.mkdir(parents=True)
    _spec_file(tmp_path / "source" / "RSM_synth", [_spec_scan(1, 10)])
    for label in range(2):
        tifffile.imwrite(images / f"RSM_synth_scan1_{label:04d}.tif", np.zeros((4, 5), dtype=np.uint16))
    page = _page(qapp)
    try:
        _on_source(page, qapp, tmp_path, "1")
        groups, _skipped = page.resolve_source()
        assert [member.detector_shape for group in groups for member in group] == [(195, 487)]
        assert (page._value("adv_rows"), page._value("adv_columns")) == ("195", "487")
    finally:
        _close(page, qapp)


def test_a_queued_run_finds_its_outputs_however_the_save_path_is_spelled(qapp, tmp_path):
    """Maintainer, 2026-11-20, at the beamline: "when I try to process all scans in a spec file, I get the error that
    "the output changed while the scans were found. Run again." Running again just gives the same error". The form
    keeps its output normalized (on Windows normcase lowers it and turns / into \\) while the queue check kept the
    Save Path's spelling, so the lookup never matched. A Save Path spelled with '..' differs the same way anywhere."""
    _eiger_source(tmp_path, (195, 487))
    (tmp_path / "sub").mkdir()
    page = _page(qapp)
    try:
        _on_source(page, qapp, tmp_path, "1,2")
        page.set_field("output_path", str(tmp_path / "sub" / ".." / "Out"))
        page.begin_run()
        deadline = time.monotonic() + 120
        while page.active() and time.monotonic() < deadline:
            qapp.processEvents(); page._poll(); time.sleep(0.005)
        text = page.run_bar.readinessLabel.full_text()
        assert "Run again" not in text, text
        assert len(list((tmp_path / "Out").glob("*_rsm.nexus"))) == 2, text
    finally:
        _close(page, qapp)


def test_a_queued_run_checks_the_actual_normalized_output_before_gridding(qapp, tmp_path):
    _eiger_source(tmp_path, (195, 487))
    page = _page(qapp)
    try:
        _on_source(page, qapp, tmp_path, "1,2")
        page.set_field("output_path", str(tmp_path / "absent" / ".." / "Out"))
        page.resolve_source()
        first, second = (Path(form.output_path) for form in page.build_forms())
        second.parent.mkdir()
        second.write_bytes(b"operator data")
        page.begin_run()
        _wait(page, qapp, lambda: not page.active(), seconds=120)
        assert not first.exists(), page.run_bar.readinessLabel.full_text()
        assert "nothing was gridded" in page.run_bar.readinessLabel.full_text()
        assert second.read_bytes() == b"operator data"
    finally:
        _close(page, qapp)


def test_run_resolves_new_container_shape_before_applying_chunk_cap(qapp, tmp_path):
    _eiger_source(tmp_path, (195, 487))
    page = _page(qapp)
    try:
        _on_source(page, qapp, tmp_path, "2", rows=(2048, 2048))
        page.set_field("adv_max_chunk_mib", "32")
        assert page.run_bar.startButton.isEnabled(), page.run_bar.readinessLabel.full_text()
        page.run_bar.startButton.click()
        _wait(page, qapp, lambda: not page.active(), seconds=120)
        assert (page._value("adv_rows"), page._value("adv_columns")) == ("195", "487")
        assert len(page.browser.listed_results()) == 1, page.run_bar.readinessLabel.full_text()
    finally:
        _close(page, qapp)


def test_switching_spec_format_resolves_the_newly_enabled_entry_before_run(qapp, tmp_path):
    _canonical_partner(tmp_path)
    images = tmp_path / "source" / "images"
    images.mkdir(parents=True)
    _spec_file(tmp_path / "source" / "RSM_synth", [_spec_scan(2, 11)])
    with h5py.File(images / "RSM_synth_scan002_master.h5", "w") as handle:
        for name, shape in (("entry", (1024, 1024)), ("experiment", (195, 487))):
            entry = handle.create_group(name)
            entry.attrs["NX_class"] = "NXentry"
            entry.create_dataset("data/data", shape=(2, *shape), dtype=np.uint16, fillvalue=1)
    page = _page(qapp)
    try:
        _on_source(page, qapp, tmp_path, "2")
        page.set_field("nexus_entry", "experiment")
        _choose(qapp, page._rows["data_format"], "spec_series")
        page.resolve_source()
        page.set_field("adv_max_chunk_mib", "32")
        assert not page.run_bar.startButton.isEnabled()
        _choose(qapp, page._rows["data_format"], "spec_container")
        assert page.run_bar.startButton.isEnabled(), page.run_bar.readinessLabel.full_text()
        page.run_bar.startButton.click()
        _wait(page, qapp, lambda: not page.active(), seconds=120)
        assert (page._value("adv_rows"), page._value("adv_columns")) == ("195", "487")
        assert len(page.browser.listed_results()) == 1, page.run_bar.readinessLabel.full_text()
    finally:
        _close(page, qapp)
