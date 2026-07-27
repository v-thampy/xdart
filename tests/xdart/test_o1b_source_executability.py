"""O-1b S2 — capability truth is not execution eligibility.

Review §54.4 S2.  ``describe_source_readiness`` correctly reports that an
extensionless SPEC file contains scan-table frames, metadata, motors and psi
columns, and after H18 the panel agrees with it.  But the Image-Series worker
has no reader for that file, so "the source describes frames" was being taken as
"a Run can consume it" and ``RunTarget.SOURCE`` was set on something unrunnable.

These rows pin both directions: the unrunnable selections that must NOT become
``RunTarget.SOURCE``, and the valid ones whose accepted eligibility must not be
collateral damage.  Every row drives the real ``staticWidget`` through the
production Source-card parameters.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")

h5py = pytest.importorskip("h5py")


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _controls_v2(monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))


def _state(qapp, configure):
    """Real widget -> production parameter edits -> real control state."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        configure(widget)
        return widget._controls_v2_state()
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def _select(source_type, path):
    def configure(widget):
        widget._controls_v2_param(("Signal", "inp_type")).setValue(source_type)
        widget._controls_v2_param(("Signal", "File")).setValue(str(path))
    return configure


def _tiff_series(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    frames = []
    for index in range(3):
        path = tmp_path / f"series_{index:04d}.tif"
        tifffile.imwrite(path, np.full((6, 6), index + 1, dtype=np.int32))
        frames.append(path)
    return frames[0]


def _eiger_master(tmp_path):
    master = tmp_path / "scan_master.h5"
    raw = np.arange(2 * 8 * 8, dtype=np.uint32).reshape(2, 8, 8)
    with h5py.File(master, "w") as handle:
        handle.create_dataset("entry/data/data", data=raw)
    return master


# --------------------------------------------------------------------------- #
# Not executable — must not claim RunTarget.SOURCE.
# --------------------------------------------------------------------------- #

def test_unsupported_extension_image_member_is_not_runnable(qapp, tmp_path):
    """§54.4 S2 item 4.  No accepted adapter claims this filename at all, so
    there is no reader that could consume it and it must not gate a Run."""
    from xrd_tools.session.readiness import RunTarget

    stray = tmp_path / "measurement_0001.qqq"
    stray.write_bytes(b"not an image")

    state = _state(qapp, _select("Image Series", stray))

    assert state.run_target is not RunTarget.SOURCE, (
        "a file no accepted reader owns was offered as a runnable source")


def test_spec_file_selected_as_the_image_member_is_not_runnable(
        qapp, tmp_path):
    """§54.4 S2 item 4, the reported case.  The SPEC adapter DOES own an
    extensionless file -- that is why its capabilities are honestly true -- but
    ``SourceKind.SPEC`` is not a family the image worker can execute."""
    from xrd_tools.session.readiness import RunTarget

    spec = tmp_path / "myscan"
    spec.write_text("#F myscan\n#S 5 ascan hy 0 2 2 1\n#N 3\n#L hy  chi  I0\n"
                    "0 1 100\n1 2 100\n2 3 100\n")

    state = _state(qapp, _select("Image Series", spec))

    assert state.run_target is not RunTarget.SOURCE


def test_capability_truth_is_untouched_for_a_spec_source(qapp, tmp_path):
    """§54.4 S2 item 2.  Only EXECUTABILITY moved.  The panel must still report
    the SPEC scan table's frames, metadata, motors and psi columns, because
    that is what the source genuinely serves."""
    spec = tmp_path / "myscan"
    spec.write_text("#F myscan\n#S 5 ascan hy 0 2 2 1\n#N 3\n#L hy  chi  I0\n"
                    "0 1 100\n1 2 100\n2 3 100\n")

    caps = _state(qapp, _select("Image Series", spec)).source_caps

    assert caps.has_frames is True
    assert caps.has_raw is True
    assert caps.has_metadata is True
    assert caps.has_motors is True
    assert caps.has_psi_metadata is True


# --------------------------------------------------------------------------- #
# Executable — accepted eligibility must survive.
# --------------------------------------------------------------------------- #

def test_valid_tiff_series_remains_runnable(qapp, tmp_path):
    """§54.4 S2 item 6."""
    from xrd_tools.session.readiness import RunTarget

    first = _tiff_series(tmp_path)

    state = _state(qapp, _select("Image Series", first))

    assert state.run_target is RunTarget.SOURCE, (
        "a valid TIFF series lost its accepted Run eligibility")


def test_eiger_master_remains_runnable(qapp, tmp_path):
    """§54.4 S2 item 6."""
    from xrd_tools.session.readiness import RunTarget

    master = _eiger_master(tmp_path)

    def configure(widget):
        widget._controls_v2_param(("Signal", "inp_type")).setValue("Image Series")
        widget._controls_v2_param(("Signal", "img_ext")).setValue("h5")
        widget._controls_v2_param(("Signal", "File")).setValue(str(master))

    state = _state(qapp, configure)

    assert state.run_target is RunTarget.SOURCE, (
        "a valid Eiger/HDF5 source lost its accepted Run eligibility")


def test_single_image_remains_runnable(qapp, tmp_path):
    """A Single Image selection is the plainest executable source there is."""
    from xrd_tools.session.readiness import RunTarget

    tifffile = pytest.importorskip("tifffile")
    frame = tmp_path / "single_0001.tif"
    tifffile.imwrite(frame, np.full((6, 6), 7, dtype=np.int32))

    state = _state(qapp, _select("Single Image", frame))

    assert state.run_target is RunTarget.SOURCE


def test_configured_directory_remains_runnable(qapp, tmp_path):
    """§54.4 S2 item 6 + R4A-6.  A lazy Directory has no selected member to
    classify -- membership is discovered by the worker after Run -- so its
    configured-intent eligibility must be untouched by this correction."""
    from xrd_tools.session.readiness import RunTarget

    raw = tmp_path / "raw"
    raw.mkdir()

    def configure(widget):
        signal = widget.wrangler.parameters.child("Signal")
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_dir").setValue(str(raw))
        signal.child("img_ext").setValue("nxs")
        signal.child("include_subdir").setValue(True)
        signal.child("File").setValue("")

    state = _state(qapp, configure)

    assert state.run_target is RunTarget.SOURCE, (
        "a configured lazy container directory lost Run eligibility")


def test_spec_metadata_beside_a_valid_image_source_stays_runnable(
        qapp, tmp_path):
    """§54.4 S2 item 5.  SPEC as the METADATA format is a different field from
    the image selection, and it must keep working: this correction refuses SPEC
    as an image member, not SPEC as a sidecar."""
    from xrd_tools.session.readiness import RunTarget

    first = _tiff_series(tmp_path)
    (tmp_path / "series").write_text(
        "#F series\n#S 1 ascan hy 0 2 2 1\n#N 3\n#L hy  chi  I0\n"
        "0 1 100\n1 2 100\n2 3 100\n")

    def configure(widget):
        widget._controls_v2_param(("Signal", "inp_type")).setValue("Image Series")
        widget._controls_v2_param(("Signal", "File")).setValue(str(first))
        widget._controls_v2_param(("Signal", "meta_ext")).setValue("spec")

    state = _state(qapp, configure)

    assert state.run_target is RunTarget.SOURCE, (
        "selecting SPEC as the metadata format disabled a valid image Run")
