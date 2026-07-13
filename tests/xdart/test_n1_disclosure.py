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
import xdart.gui.gui_utils  # noqa: F401  # registers the 'str_browse' param type (the live GUI imports gui_utils at startup; the wrangler-only import path here does not)
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
    # The real _set_status_text elides into a QLabel; the fake label just
    # records the text (the disclosure messages are what's under test).
    h._set_status_text = h.ui.specLabel.setText
    h._DISCLOSURE_REST = imageWrangler._DISCLOSURE_REST
    h._DISCLOSURE_CARRIERS = imageWrangler._DISCLOSURE_CARRIERS
    h._DISCLOSURE_TOPLEVEL = imageWrangler._DISCLOSURE_TOPLEVEL
    return h, root


def _hidden(root, group):
    return root.child(group).opts.get("visible") is False


def test_disclosure_two_stages(qapp):
    # Two-stage now: the PONI picker lives inside DATA (first row), so the
    # separate CALIBRATION stage is gone — DATA reveals once a Project Folder is
    # set, and PONI validity is enforced at run time (status nudge only).
    h, root = _holder()

    # Fresh: ONLY Project visible (Folder + Save Path live in it); DATA hidden.
    h._apply_disclosure()
    assert not _hidden(root, "Project")
    assert _hidden(root, "Signal")
    assert root.child("Project").child("project_folder").value() == ""
    assert root.child("Project").child("h5_dir").value() == ""
    assert "Project Folder" in h.ui.specLabel.t

    # Invalid folder text is still a fresh-start state: keep setup hidden.
    root.child("Project").child("project_folder").setValue(
        "/definitely/not/an/xdart/project")
    h._apply_disclosure()
    assert _hidden(root, "Signal")
    assert "Project Folder" in h.ui.specLabel.t

    # Folder set -> the whole DATA group reveals (Poni is its first row), even
    # without a PONI yet — but the status nudges to load one.
    root.child("Project").child("project_folder").setValue("/tmp")
    h._apply_disclosure()
    assert not _hidden(root, "Signal") and not _hidden(root, "BG")
    assert "PONI" in h.ui.specLabel.t

    # Folder + valid PONI -> status clears.
    h.poni = object()
    h._apply_disclosure()
    for g in ("Project", "Signal", "BG"):
        assert not _hidden(root, g)
    assert h.ui.specLabel.t == ""


def test_folder_change_resets_dependent_paths_and_clears_stale_poni(qapp, tmp_path):
    h, root = _holder()
    # Seed a prior folder's config, incl. a "loaded" PONI (instance attr + param).
    root.child("Signal").child("poni_file").setValue("/old/cal.poni")
    h.poni_file = "/old/cal.poni"
    h.poni = object()                              # a loaded calibration
    root.child("Signal").child("File").setValue("/old/img_0001.tif")
    root.child("Signal").child("mask_file").setValue("/old/mask.edf")

    # User picks a new Project Folder -> Decision 2 invalidation.
    root.child("Project").child("project_folder").setValue(str(tmp_path))
    h._on_project_folder_changed()

    assert root.child("Signal").child("poni_file").value() == ""
    # The INSTANCE attr is resynced too (else get_poni_dict reloads the stale PONI
    # and _inputs_valid stays True -> Start runs the new images vs the old cal).
    assert h.poni_file == ""
    assert root.child("Signal").child("File").value() == ""
    assert root.child("Signal").child("mask_file").value() == ""
    assert root.child("Project").child("h5_dir").value() == os.path.join(
        str(tmp_path), "xdart_processed_data")
    assert h.source_base == os.path.abspath(str(tmp_path))


def test_folder_change_inert_during_restore(qapp):
    """Drive the REAL sigValueChanged wiring: while _restoring, the handler's
    destructive body is short-circuited (poni_file kept); after restore, a
    genuine folder change clears it."""
    h, root = _holder()
    root.child("Project").child("project_folder").sigValueChanged.connect(
        lambda *a: h._on_project_folder_changed())
    root.child("Signal").child("poni_file").setValue("/keep/cal.poni")
    h.poni_file = "/keep/cal.poni"

    h._restoring = True
    root.child("Project").child("project_folder").setValue("/restored/root")
    assert root.child("Signal").child("poni_file").value() == "/keep/cal.poni"
    assert h.poni_file == "/keep/cal.poni"          # guard held

    h._restoring = False
    root.child("Project").child("project_folder").setValue("/new/root")
    assert root.child("Signal").child("poni_file").value() == ""   # now reset
    assert h.poni_file == ""


def test_meta_ext_hidden_for_nxs_image_type(qapp):
    """Scan taxonomy (Vivek, Jun 2026): a .nxs embeds its own metadata, so the
    Meta File field is irrelevant for nxs and must be HIDDEN (it was
    previously only made readonly)."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import params as wparams

    root = Parameter.create(name="p", type="group", children=wparams)
    h = types.SimpleNamespace(parameters=root, img_ext="nxs")
    sync = MethodType(imageWrangler._sync_meta_ext_to_img_ext, h)

    sync()
    meta = root.child("Signal").child("meta_ext")
    assert meta.opts.get("visible") is False
    assert meta.value() == "none"

    h.img_ext = "tif"
    sync()
    assert meta.opts.get("visible", True) is True


def test_meta_ext_fresh_default_is_auto(qapp):
    """Fresh sessions default to metadata auto-discovery, with explicit off."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import params as wparams

    root = Parameter.create(name="p", type="group", children=wparams)
    meta = root.child("Signal").child("meta_ext")

    assert meta.value() == "auto"
    assert set(meta.opts["limits"]) >= {"auto", "none", "txt", "pdi", "metadata", "spec"}


@pytest.mark.parametrize(
    ("saved", "param_value", "attr_value"),
    [
        ("txt", "txt", "txt"),
        ("SPEC", "spec", "spec"),
        # A saved literal 'none' is a DELIBERATE post-MD-2 off — honored.
        ("none", "none", None),
        # Legacy sessions (pre-'auto') encoded "never set" as None/'' — the
        # old metadata-off default, not a choice.  Restore migrates them to
        # the modern default instead of cementing a permanent 'none'
        # (maintainer, 2026-07-13: Meta Type should be auto by default).
        (None, "auto", "auto"),
        ("", "auto", "auto"),
    ],
)
def test_meta_ext_session_restore_keeps_saved_value(qapp, monkeypatch, saved, param_value, attr_value):
    """Restored sessions keep a deliberately saved Meta Type; legacy-unset
    encodings (None/'') migrate to the modern 'auto' default."""
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler as iw

    class _Combo:
        def findText(self, _text):
            return -1
        def setCurrentIndex(self, _idx):
            pass

    class _Check:
        def setChecked(self, value):
            self.value = value

    root = Parameter.create(name="p", type="group", children=params)
    h = types.SimpleNamespace(
        parameters=root,
        _restoring=False,
        _SESSION_PARAMS=imageWrangler._SESSION_PARAMS,
        ui=types.SimpleNamespace(
            processingModeCombo=_Combo(),
            batchCheckBox=_Check(),
        ),
        _compute_source_base=lambda: None,
        get_poni_dict=lambda: None,
    )
    monkeypatch.setattr(iw, "load_session", lambda: {"meta_ext": saved})

    MethodType(imageWrangler._restore_from_session, h)()

    assert root.child("Signal").child("meta_ext").value() == param_value
    assert h.meta_ext == attr_value


def test_sync_meta_ext_is_idempotent_no_reemit(qapp):
    """Jun 10 live-testing regression: pyqtgraph show()/hide() emit
    sigOptionsChanged UNCONDITIONALLY, and _sync_meta_ext_to_img_ext runs
    inside setup() (wired to sigTreeStateChanged) — an unconditional emit
    there is an infinite setup() recursion (RecursionError at app start).
    A second sync with unchanged state must emit NOTHING."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import params as wparams

    root = Parameter.create(name="p", type="group", children=wparams)
    h = types.SimpleNamespace(parameters=root, img_ext="tif")
    sync = MethodType(imageWrangler._sync_meta_ext_to_img_ext, h)
    meta = root.child("Signal").child("meta_ext")

    sync()                                   # settle to the tif state
    events = []
    meta.sigOptionsChanged.connect(lambda *a: events.append(a))
    sync()                                   # unchanged -> must not emit
    assert events == []

    h.img_ext = "nxs"
    sync()                                   # real change -> emits (hide)
    assert events
    n = len(events)
    sync()                                   # unchanged nxs state -> silent
    assert len(events) == n


def test_apply_disclosure_is_idempotent_no_reemit(qapp):
    """A settled disclosure pass must not re-emit visibility options changes."""
    h, root = _holder()
    root.child("Project").child("project_folder").setValue("/tmp")
    h.poni = object()
    h._apply_disclosure()

    events = []
    for child in root.children():
        child.sigOptionsChanged.connect(lambda *a: events.append(a))

    h._apply_disclosure()
    assert events == []


def test_viewer_modes_hide_all_processing_groups(qapp):
    """Image/XYE viewer modes hide every processing group, leaving only Project
    Folder + Save Path — even when a folder + PONI are set (which would normally
    reveal everything via progressive disclosure)."""
    for vm in ("image", "xye"):
        h, root = _holder()
        h.viewer_mode = vm
        root.child("Project").child("project_folder").setValue("/tmp")
        h.poni = object()                          # would normally reveal all
        h._apply_disclosure()
        assert not _hidden(root, "Project"), vm    # Project Folder: visible
        assert root.child("Project").child("h5_dir").opts.get("visible") is not False, vm        # Save Path (in Project)
        for g in imageWrangler._DISCLOSURE_REST:   # Signal/BG (DATA + background)
            assert _hidden(root, g), (vm, g)


def test_leaving_viewer_mode_restores_disclosure(qapp):
    """Switching back from a viewer mode to a processing mode restores the
    normal Project->Calibration->rest disclosure (folder + PONI set -> all)."""
    h, root = _holder()
    root.child("Project").child("project_folder").setValue("/tmp")
    h.poni = object()
    h.viewer_mode = "image"
    h._apply_disclosure()
    assert _hidden(root, "Signal")                 # hidden in viewer mode
    h.viewer_mode = None                           # back to a processing mode
    h._apply_disclosure()
    for g in ("Project", "Signal", "BG"):
        assert not _hidden(root, g), g             # full disclosure restored
