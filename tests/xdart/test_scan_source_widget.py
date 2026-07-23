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


def _open(sel):
    """Open the source a VALUE-ONLY selection points at.  H19 §3: the widget no
    longer hands a live FrameSource across its async-probe boundary — a consumer
    opens its own from ``sel.spec``."""
    from xrd_tools.sources import open_source
    return open_source(sel.spec)


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
        assert not hasattr(sel, "source")             # value-only: no live source (H19 §3)
        assert _open(sel).frame_indices == [0, 1, 2]  # scan 5 (default, first)
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
        src2 = _open(sel2)
        np.testing.assert_allclose(src2.load_frame(0), 1.0)
        np.testing.assert_allclose(src2.load_frame(2), 3.0)
        # the value-only probe still carried the decoded first frame as a COPY
        assert sel2.first_image is not None
        assert isinstance(sel2.first_image.data, bytes)
        np.testing.assert_allclose(sel2.first_image.to_array(), 1.0)
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
        assert _open(emitted[-1]).frame_indices == [0, 1, 2]   # scan 5
        w.scan_combo.setCurrentIndex(1)                        # → scan 6 (2 pts)
        assert _open(emitted[-1]).frame_indices == [0, 1]
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
        assert _open(emitted[-1]).frame_indices == [0, 1, 2]
    finally:
        w.deleteLater()


def test_widget_async_probe_emits_latest_selection(qapp, tmp_path):
    from xdart.gui.tabs.static_scan.scan_source_widget import (
        ImagePreview, ScanSourceWidget)

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
        np.testing.assert_allclose(_open(sel).load_frame(0), 1.0)
        assert isinstance(sel.first_image, ImagePreview)
        assert isinstance(sel.first_image.data, bytes)
        np.testing.assert_allclose(sel.first_image.to_array(), 1.0)
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
        # value-only result tuple: (gen, sig, spec, reachable, first_image, exc)
        w._on_probe_done((1, sig, spec, True, None, None))
        assert emitted == []
        assert w._last_selection is None
    finally:
        w.shutdown_probe_worker()
        w.deleteLater()


def test_controls_source_widget_emits_latest_directory_generation(
    qapp, tmp_path, monkeypatch,
):
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget
    from xrd_tools.sources.directory_index import DirectoryIndex

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "old.nxs").write_bytes(b"old")
    (second / "scan_10.nxs").write_bytes(b"ten")
    (second / "scan_2.nxs").write_bytes(b"two")

    monkeypatch.setattr(
        DirectoryIndex,
        "probe_candidate",
        lambda *_args, **_kwargs: pytest.fail(
            "Source status must not inspect container contents"),
    )
    w = ScanSourceWidget(mode="controls_source", async_probe=True)
    emitted = []
    w.sigDirectoryChanged.connect(lambda observation: emitted.append(observation))
    try:
        w.configure_directory(first, suffixes=(".nxs",))
        w.configure_directory(second, suffixes=(".nxs",))
        assert _wait_for(
            qapp,
            lambda: emitted
            and emitted[-1] is not None
            and emitted[-1].discovered_snapshot.root == second,
        )
        assert [
            item.path.name
            for item in emitted[-1].discovered_snapshot.candidates
        ] == ["scan_2.nxs", "scan_10.nxs"]
        assert emitted[-1].ready_snapshot.candidates == ()
        assert emitted[-1].content_opens == 0
        assert emitted[-1].request_generation == (
            w.directory_session.request_generation)
        assert w.directory_status.text() == "2 matching files in this folder"

        # Routine watch polls must leave the last completed observation on
        # screen.  Replacing it with a one-frame "Checking directory..."
        # message once per second makes the Source card visibly flicker.
        w.request_directory_poll()
        assert w.directory_status.text() == "2 matching files in this folder"
        assert _wait_for(qapp, lambda: w._directory_future is None)
        assert w.directory_status.text() == "2 matching files in this folder"
    finally:
        w.shutdown_probe_worker()
        w.deleteLater()


def test_controls_source_counts_direct_names_without_content_or_subdir_walk(
    qapp, monkeypatch, tmp_path,
):
    """Source status is a cheap direct-child name projection.

    Subdirs is frozen as Run intent, but selecting it must neither recurse nor
    open/probe every candidate merely to paint the Source card.
    """
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget
    from xrd_tools.sources.directory_index import DirectoryIndex

    for index in range(17):
        (tmp_path / f"scan_{index}.nxs").write_bytes(b"x")
    nested = tmp_path / "nested"
    nested.mkdir()
    for index in range(5):
        (nested / f"nested_{index}.nxs").write_bytes(b"x")

    probe_calls = []
    monkeypatch.setattr(
        DirectoryIndex,
        "probe_candidate",
        lambda *_args, **_kwargs: probe_calls.append(True),
    )

    w = ScanSourceWidget(mode="controls_source", async_probe=True)
    w._directory_timer.stop()
    try:
        w.configure_directory(
            tmp_path,
            recursive=True,
            suffixes=(".nxs",),
            subdirs_lazy=True,
        )
        assert _wait_for(
            qapp,
            lambda: w.directory_observation is not None,
        )
        observation = w.directory_observation
        assert len(observation.discovered_snapshot.candidates) == 17
        assert observation.ready_snapshot.candidates == ()
        assert observation.content_opens == 0
        assert probe_calls == []
        assert w.directory_session.configured.recursive is False
        assert w.directory_subdirs_lazy is True
        assert w.directory_status.text() == (
            "17 matching files in this folder · "
            "subfolders processed during Run"
        )
    finally:
        w.shutdown_probe_worker()
        w.deleteLater()


def test_image_preview_is_read_only_and_rejects_unsafe_payloads():
    from xdart.gui.tabs.static_scan.scan_source_widget import ImagePreview

    preview = ImagePreview.from_array(np.arange(6, dtype=np.uint16).reshape(2, 3))
    restored = preview.to_array()
    assert restored.flags.writeable is False
    np.testing.assert_array_equal(restored, [[0, 1, 2], [3, 4, 5]])

    with pytest.raises(TypeError, match="object-dtype"):
        ImagePreview.from_array(np.asarray([[object()]], dtype=object))
    with pytest.raises(ValueError, match="byte count"):
        ImagePreview((2, 2), "<u2", b"short")


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


def test_widget_numbered_raw_file_resolves_series_and_enables_roi(qapp, tmp_path):
    """Picking one numbered RAW frame represents its whole scan series, and
    the widget's binary-read parameters reach the source used by the ROI gate."""
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget
    from xrd_tools.core.scan import SourceKind

    for index in (0, 1, 2):
        np.full((3, 4), index + 1, dtype=np.uint16).tofile(
            tmp_path / f"scanA_{index:04d}.raw")
    np.full((3, 4), 99, dtype=np.uint16).tofile(
        tmp_path / "scanB_0000.raw")

    w = ScanSourceWidget(mode="roi")
    emitted = []
    w.sigSourceChanged.connect(lambda sel: emitted.append(sel))
    try:
        w.set_uri(str(tmp_path / "scanA_0001.raw"))
        assert w._current_candidate().kind is SourceKind.TIFF_SERIES
        assert dict(w._current_candidate().options)["pattern"] == "scanA_*"
        assert w.adv_btn.isChecked()
        assert "enter raw shape" in w.raw_dot.text().lower()

        w.det_rows.setText("3")
        w.det_cols.setText("4")
        w.dtype_combo.setCurrentText("uint16")
        w._emit_selection()

        selection = emitted[-1]
        assert selection is not None and selection.reachable
        assert "raw ready" in w.raw_dot.text()
        source = _open(selection)
        assert source.frame_indices == [1, 2, 3]
        np.testing.assert_array_equal(source.load_frame(1), np.ones((3, 4)))
        np.testing.assert_array_equal(source.load_frame(3), np.full((3, 4), 3))
        assert w.image_dir_edit.isHidden()
        assert w.image_stem_edit.isHidden()
    finally:
        w.deleteLater()


def test_widget_spec_raw_controls_use_operator_facing_labels(qapp, tmp_path):
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget

    spec = _spec_with_images(tmp_path)
    w = ScanSourceWidget(mode="roi")
    try:
        w.set_uri(str(spec))
        assert not w.image_dir_edit.isHidden()
        assert not w.image_stem_edit.isHidden()
        assert w.images_label.text() == "Raw image folder"
        assert w.image_stem_label.text() == "Filename contains"
        assert "automatic" in w.image_stem_edit.toolTip().lower()
    finally:
        w.deleteLater()


def test_widget_spec_raw_status_explains_shape_and_scan_match(qapp, tmp_path):
    """A disabled ROI gate tells the operator which SPEC/raw precondition is
    missing instead of collapsing every failure to ``raw unavailable``."""
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget

    spec = _spec_with_images(tmp_path)
    w = ScanSourceWidget(mode="roi")
    try:
        w.set_uri(str(spec))
        w.image_dir_edit.setText(str(tmp_path))
        w._emit_selection()
        assert "enter raw shape" in w.raw_dot.text().lower()

        w.det_rows.setText("6")
        w.det_cols.setText("6")
        w.dtype_combo.setCurrentText("int32")
        w.scan_combo.setCurrentIndex(1)  # scan 6 has metadata but no images
        assert "no matching images" in w.raw_dot.text().lower()
        assert "myscan_scan6_" in w.raw_dot.toolTip()
        assert str(tmp_path) in w.raw_dot.toolTip()
    finally:
        w.deleteLater()


def test_widget_spec_common_raw_shape_enables_roi_without_manual_shape(
        qapp, tmp_path):
    """A known headerless detector shape is a headless reader capability, not
    a private Image Viewer fallback or an operator requirement."""
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget

    spec = tmp_path / "known"
    spec.write_text(_SPEC.replace("#F myscan", "#F known"))
    image = np.arange(195 * 487, dtype=np.int32).reshape(195, 487)
    for index in range(3):
        (image + index).tofile(tmp_path / f"known_scan5_{index:04d}.raw")

    w = ScanSourceWidget(mode="roi")
    emitted = []
    w.sigSourceChanged.connect(lambda selection: emitted.append(selection))
    try:
        w.set_uri(str(spec))
        assert emitted[-1] is not None and emitted[-1].reachable
        assert "raw ready" in w.raw_dot.text().lower()
        assert emitted[-1].first_image.shape == (195, 487)
        assert "detector_shape" not in dict(
            emitted[-1].spec.options["read_image_kwargs"])
    finally:
        w.deleteLater()


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
