# -*- coding: utf-8 -*-
"""The shared ScanSourceWidget: kind classification, the scan selector, the
images affordance + raw-reachable dot, and directory mode."""
import numpy as np
import pytest
import time

pytest.importorskip("silx")

_SPEC = """#F myscan
#E 1
#O0 th  chi

#S 5 ascan th 0 2 2 1
#P0 0 5
#N 3
#L th  i0  det
0 100 10
1 110 20
2 120 30

#S 6 ascan chi 0 1 1 1
#P0 7 0
#N 2
#L chi  i0
0 300
1 310
"""


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _spec_with_images(tmp_path):
    spec = tmp_path / "myscan"
    spec.write_text(_SPEC)
    for i in range(3):                       # scan-5 raw frames
        np.full((6, 6), i + 1, dtype="int32").tofile(
            tmp_path / f"myscan_scan5_{i:04d}.raw")
    return spec


def _wait_for(qapp, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_widget_spec_metadata_then_images(qapp, tmp_path):
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget
    from xrd_tools.core.scan import SourceKind

    spec = _spec_with_images(tmp_path)
    w = ScanSourceWidget(mode="roi")
    emitted = []
    w.sigSourceChanged.connect(lambda sel: emitted.append(sel))
    try:
        w.set_uri(str(spec))
        sel = emitted[-1]
        assert sel is not None and sel.spec.kind is SourceKind.SPEC
        assert sel.reachable is False                 # no images yet → metadata only
        assert sel.source.frame_indices == [0, 1, 2]  # scan 5 (default, first)
        # the multi-scan selector lists both scans
        assert [w.scan_combo.itemText(i) for i in range(w.scan_combo.count())] == \
            ["myscan [5.1]", "myscan [6.1]"]

        # point Images at the raw folder + give raw read params → reachable
        w.image_dir_edit.setText(str(tmp_path))
        w.det_rows.setText("6")
        w.det_cols.setText("6")
        w.dtype_combo.setCurrentText("int32")
        w._emit_selection()
        sel2 = emitted[-1]
        assert sel2.reachable is True
        assert "● raw" in w.raw_dot.text()
        np.testing.assert_allclose(sel2.source.load_frame(0), 1.0)
        np.testing.assert_allclose(sel2.source.load_frame(2), 3.0)
    finally:
        w.deleteLater()


def test_widget_spec_auto_image_folder(qapp, tmp_path):
    """When the image folder is left blank, images sitting next to the spec file
    are auto-found (design §3), so raw becomes reachable without a folder pick."""
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget

    spec = _spec_with_images(tmp_path)
    w = ScanSourceWidget(mode="roi")
    emitted = []
    w.sigSourceChanged.connect(lambda sel: emitted.append(sel))
    try:
        w.set_uri(str(spec))
        # raw read params (needed to decode the .raw) but NOT the image folder
        w.det_rows.setText("6")
        w.det_cols.setText("6")
        w.dtype_combo.setCurrentText("int32")
        w._emit_selection()
        sel = emitted[-1]
        assert sel.reachable is True
        assert sel.spec.options["image_dir"] == str(tmp_path)   # auto-derived
    finally:
        w.deleteLater()


def test_widget_caches_unchanged_spec(qapp, tmp_path):
    """Re-emitting with no field change re-uses the cached, already-opened
    selection (no redundant open_source / frame decode)."""
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget

    spec = _spec_with_images(tmp_path)
    w = ScanSourceWidget(mode="roi")
    emitted = []
    w.sigSourceChanged.connect(lambda sel: emitted.append(sel))
    try:
        w.set_uri(str(spec))
        first = emitted[-1]
        w._emit_selection()                       # nothing changed
        assert emitted[-1] is first               # same cached object, not re-opened
    finally:
        w.deleteLater()


def test_widget_scan_switch_reloads(qapp, tmp_path):
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget

    spec = _spec_with_images(tmp_path)
    w = ScanSourceWidget(mode="roi")
    emitted = []
    w.sigSourceChanged.connect(lambda sel: emitted.append(sel))
    try:
        w.set_uri(str(spec))
        assert emitted[-1].source.frame_indices == [0, 1, 2]   # scan 5
        w.scan_combo.setCurrentIndex(1)                        # → scan 6 (2 pts)
        assert emitted[-1].source.frame_indices == [0, 1]
        assert "6.1" in emitted[-1].spec.options["scan"]
    finally:
        w.deleteLater()


def test_widget_directory_mode_discovers_scans(qapp, tmp_path):
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget

    _spec_with_images(tmp_path)
    w = ScanSourceWidget(mode="roi")
    emitted = []
    w.sigSourceChanged.connect(lambda sel: emitted.append(sel))
    try:
        w.dir_check.setChecked(True)
        w.dir_kind_combo.setCurrentIndex(0)        # SPEC
        w.path_edit.setText(str(tmp_path))
        w._refresh_candidates()
        # both SPEC scans discovered in the folder
        assert [w.scan_combo.itemText(i) for i in range(w.scan_combo.count())] == \
            ["myscan [5.1]", "myscan [6.1]"]
        assert emitted[-1].source.frame_indices == [0, 1, 2]
    finally:
        w.deleteLater()


def test_widget_async_probe_emits_latest_selection(qapp, tmp_path):
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget

    spec = _spec_with_images(tmp_path)
    w = ScanSourceWidget(mode="roi", async_probe=True)
    emitted = []
    w.sigSourceChanged.connect(lambda sel: emitted.append(sel))
    try:
        w.set_uri(str(spec))
        assert "probing" in w.raw_dot.text()
        assert _wait_for(qapp, lambda: emitted and emitted[-1] is not None)
        first = emitted[-1]
        assert first.reachable is False

        w.image_dir_edit.setText(str(tmp_path))
        w.det_rows.setText("6")
        w.det_cols.setText("6")
        w.dtype_combo.setCurrentText("int32")
        w._emit_selection()

        assert _wait_for(qapp, lambda: emitted[-1] is not first)
        sel = emitted[-1]
        assert sel.reachable is True
        np.testing.assert_allclose(sel.source.load_frame(0), 1.0)
    finally:
        w.shutdown_probe_worker()
        w.deleteLater()


def test_widget_async_probe_ignores_stale_generation(qapp, tmp_path):
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget
    from xrd_tools.core.scan import SourceKind, SourceSpec

    w = ScanSourceWidget(mode="roi", async_probe=True)
    emitted = []
    w.sigSourceChanged.connect(lambda sel: emitted.append(sel))
    try:
        spec = SourceSpec(str(tmp_path), SourceKind.TIFF_SERIES)
        sig = w._spec_signature(spec)
        w._probe_generation = 2
        w._pending_sig = sig
        w._on_probe_done((1, sig, spec, object(), True, None, None))
        assert emitted == []
        assert w._last_selection is None
    finally:
        w.shutdown_probe_worker()
        w.deleteLater()


def test_file_candidates_tiff_filters_to_scan_stem(qapp, tmp_path):
    """A picked TIFF resolves to a TIFF_SERIES filtered to its OWN scan stem, so a
    folder holding several scans isn't concatenated into one series."""
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget
    from xrd_tools.core.scan import SourceKind, SourceSpec

    for name in ("scanA_0001.tif", "scanA_0002.tif",
                 "scanB_0001.tif", "scanB_0002.tif", "scanB_0003.tif"):
        (tmp_path / name).write_bytes(b"")
    picked = str(tmp_path / "scanA_0002.tif")
    specs = ScanSourceWidget._file_candidates(
        picked, SourceKind.IMAGE_FILE, SourceKind, SourceSpec)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.kind is SourceKind.TIFF_SERIES
    assert str(spec.uri) == str(tmp_path)
    assert dict(spec.options).get("pattern") == "scanA_*"   # scan A only, not B


def test_source_widget_ui_tweaks(qapp):
    """Folder label reserves room (no clip), and Raw-params shares the images row."""
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget

    w = ScanSourceWidget(mode="roi")
    try:
        fm = w.dir_check.fontMetrics()
        assert w.dir_check.minimumWidth() >= fm.horizontalAdvance("Folder")
        assert w.adv_btn.parentWidget() is w.images_row     # combined onto row 2
    finally:
        w.deleteLater()
