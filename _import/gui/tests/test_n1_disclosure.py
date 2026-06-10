"""N1 progressive disclosure (design §2) for the image wrangler:

Project Folder (always visible) -> Calibration appears once a folder is set ->
the rest (Signal/GI/Mask/BG) appears once a valid PONI also loads.  A folder
change INVALIDATES the dependent (folder-relative) paths (Decision 2), and the
folder-change handler is inert during a session restore so it doesn't wipe the
values the restore is setting.

Headless: drives the real wrangler disclosure methods against a real param tree
via a light holder (no heavy widget __init__).
"""
import os
import types
from types import MethodType

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets
from pyqtgraph.parametertree import Parameter

from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler, params


@pytest.fixture(scope="module")
def qapp():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Lbl:
    def setText(self, t):
        self.t = t


def _holder():
    root = Parameter.create(name="p", type="group", children=params)
    h = types.SimpleNamespace(
        parameters=root, poni=None, poni_file="", thread=None, _restoring=False,
        ui=types.SimpleNamespace(specLabel=_Lbl()),
        _sync_h5_dir_from_parameters=lambda: None,
    )
    for m in ("_compute_source_base", "_apply_disclosure",
              "_on_project_folder_changed", "_default_h5_under_project"):
        setattr(h, m, MethodType(getattr(imageWrangler, m), h))
    h._DISCLOSURE_REST = imageWrangler._DISCLOSURE_REST
    h._DISCLOSURE_TOPLEVEL = imageWrangler._DISCLOSURE_TOPLEVEL
    return h, root


def _hidden(root, group):
    return root.child(group).opts.get("visible") is False


def test_disclosure_three_stages(qapp):
    h, root = _holder()

    # Fresh: ONLY Project visible (incl. the Save-Path row hidden).
    h._apply_disclosure()
    assert not _hidden(root, "Project")
    assert _hidden(root, "Calibration") and _hidden(root, "Signal")
    assert root.child("h5_dir").opts.get("visible") is False
    assert root.child("h5_dir_browse").opts.get("visible") is False
    assert "Project Folder" in h.ui.specLabel.t

    # Folder set, no PONI -> Calibration appears, rest still hidden.
    root.child("Project").child("project_folder").setValue("/tmp")
    h._apply_disclosure()
    assert not _hidden(root, "Calibration")
    assert _hidden(root, "Signal") and _hidden(root, "GI")
    assert "PONI" in h.ui.specLabel.t

    # Folder + valid PONI -> everything revealed.
    h.poni = object()
    h._apply_disclosure()
    for g in ("Project", "Calibration", "Signal", "GI", "Mask", "BG"):
        assert not _hidden(root, g)
    assert h.ui.specLabel.t == ""


def test_folder_change_resets_dependent_paths_and_clears_stale_poni(qapp, tmp_path):
    h, root = _holder()
    # Seed a prior folder's config, incl. a "loaded" PONI (instance attr + param).
    root.child("Calibration").child("poni_file").setValue("/old/cal.poni")
    h.poni_file = "/old/cal.poni"
    h.poni = object()                              # a loaded calibration
    root.child("Signal").child("File").setValue("/old/img_0001.tif")
    root.child("Signal").child("mask_file").setValue("/old/mask.edf")

    # User picks a new Project Folder -> Decision 2 invalidation.
    root.child("Project").child("project_folder").setValue(str(tmp_path))
    h._on_project_folder_changed()

    assert root.child("Calibration").child("poni_file").value() == ""
    # The INSTANCE attr is resynced too (else get_poni_dict reloads the stale PONI
    # and _inputs_valid stays True -> Start runs the new images vs the old cal).
    assert h.poni_file == ""
    assert root.child("Signal").child("File").value() == ""
    assert root.child("Signal").child("mask_file").value() == ""
    assert root.child("h5_dir").value() == os.path.join(
        str(tmp_path), "xdart_processed_data")
    assert h.source_base == os.path.abspath(str(tmp_path))


def test_folder_change_inert_during_restore(qapp):
    """Drive the REAL sigValueChanged wiring: while _restoring, the handler's
    destructive body is short-circuited (poni_file kept); after restore, a
    genuine folder change clears it."""
    h, root = _holder()
    root.child("Project").child("project_folder").sigValueChanged.connect(
        lambda *a: h._on_project_folder_changed())
    root.child("Calibration").child("poni_file").setValue("/keep/cal.poni")
    h.poni_file = "/keep/cal.poni"

    h._restoring = True
    root.child("Project").child("project_folder").setValue("/restored/root")
    assert root.child("Calibration").child("poni_file").value() == "/keep/cal.poni"
    assert h.poni_file == "/keep/cal.poni"          # guard held

    h._restoring = False
    root.child("Project").child("project_folder").setValue("/new/root")
    assert root.child("Calibration").child("poni_file").value() == ""   # now reset
    assert h.poni_file == ""
