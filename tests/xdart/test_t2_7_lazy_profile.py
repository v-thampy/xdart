"""T-2.7 (Correction D, §15.12-D): extension-agnostic lazy directory/profile.

The GUI profile path is direct-child-only and never recurses, opens a container,
or reads metadata for ANY Image Directory extension; the legacy source-tree
signal reconciles only for a genuine source-selection change.  Red at 6ea1483a
(TIFF + metadata preview still rglob(); the tree signal reconciled on any edit).
Production-wired: real staticWidget / wrangler / parameter tree.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    value = staticWidget()
    value._refresh_controls_v2_profile_now()
    try:
        yield value
    finally:
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _tiff_dir_with_nested(widget, tmp_path):
    root = tmp_path / "root"
    child = root / "nested"
    child.mkdir(parents=True)
    (root / "direct.tif").write_bytes(b"x")
    (child / "nested.tif").write_bytes(b"x")
    signal = widget.wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_ext").setValue("tif")
    signal.child("img_dir").setValue(str(root))
    signal.child("include_subdir").setValue(True)
    widget._v2_frame_count_cache = None
    return root


# -- item 1 : direct-child-only count for every extension -------------------

def test_tiff_directory_with_subdirs_counts_direct_children_only(widget, tmp_path):
    _tiff_dir_with_nested(widget, tmp_path)
    assert widget._controls_v2_source_frame_count() == 1


# -- item 3 : no rglob / count_frames / HDF5 open / metadata read -----------

def test_profile_source_inspection_never_recurses_or_opens(
    widget, tmp_path, monkeypatch
):
    _tiff_dir_with_nested(widget, tmp_path)

    rglob_calls = []
    original_rglob = Path.rglob

    def tracked_rglob(self, pattern, *args, **kwargs):
        rglob_calls.append((self, pattern))
        return original_rglob(self, pattern, *args, **kwargs)

    from xrd_tools.io import image as image_io

    count_calls = []
    original_count = image_io.count_frames

    def tracked_count(path, *args, **kwargs):
        count_calls.append(path)
        return original_count(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rglob", tracked_rglob)
    monkeypatch.setattr(image_io, "count_frames", tracked_count)

    widget._v2_frame_count_cache = None
    widget._controls_v2_source_frame_count()
    widget._controls_v2_first_metadata_file()

    assert rglob_calls == []
    assert count_calls == []


def test_deep_recursive_root_profile_count_stays_bounded(widget, tmp_path):
    """A recursive root with matching data only deep in subdirectories reports the
    direct-child count (here zero), never a recursive total."""
    root = tmp_path / "deep"
    leaf = root / "a" / "b" / "c"
    leaf.mkdir(parents=True)
    (leaf / "buried.tif").write_bytes(b"x")
    signal = widget.wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_ext").setValue("tif")
    signal.child("img_dir").setValue(str(root))
    signal.child("include_subdir").setValue(True)
    widget._v2_frame_count_cache = None

    assert widget._controls_v2_source_frame_count() == 0


# -- item 4 : legacy tree signal filtered to source-selection paths ---------

@pytest.mark.parametrize("path", [
    ("Signal", "mask_file"),
    ("Signal", "poni_file"),
    ("BG", "File"),
])
def test_direct_non_source_tree_edit_performs_zero_reconcile(
    widget, monkeypatch, path
):
    calls = []
    monkeypatch.setattr(
        widget, "_sync_controls_v2_source_index", lambda: calls.append(1))
    param = widget._controls_v2_param(path)
    if param is None:
        pytest.skip(f"no {path} parameter in this profile")
    param.setValue("/tmp/unrelated-direct-tree.edf")

    assert calls == []


def test_same_value_source_edit_performs_zero_reconcile(widget, monkeypatch, tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    signal = widget.wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_ext").setValue("tif")
    signal.child("img_dir").setValue(str(a))
    # prime the last-reconciled token so a same-value re-set is a no-op
    widget._controls_v2_last_reconciled_source_token = (
        widget._controls_v2_source_token())
    calls = []
    monkeypatch.setattr(
        widget, "_sync_controls_v2_source_index", lambda: calls.append(1))
    signal.child("img_dir").setValue(str(a))     # same value

    assert calls == []


def test_one_real_source_edit_reconciles_exactly_once(widget, monkeypatch, tmp_path):
    """A single genuine source edit reconciles exactly once; a redundant
    same-value re-trigger of the tree signal adds no further reconcile."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    signal = widget.wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_ext").setValue("tif")
    signal.child("img_dir").setValue(str(a))
    widget._controls_v2_last_reconciled_source_token = (
        widget._controls_v2_source_token())

    calls = []
    monkeypatch.setattr(
        widget, "_sync_controls_v2_source_index", lambda: calls.append(1))

    signal.child("img_dir").setValue(str(b))     # one real change -> reconcile
    # a redundant tree re-trigger at the SAME (new) value must not reconcile again
    widget._on_controls_v2_source_tree_changed(
        None,
        [(signal.child("img_dir"), "value", str(b))],
    )

    assert calls == [1]
