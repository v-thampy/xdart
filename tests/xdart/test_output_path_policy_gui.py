# -*- coding: utf-8 -*-
"""P4/OUT-1 — frozen acceptance oracle for the canonical GUI output-path consumers.

The GUI (PP-C1) half of the finite oracle.  The headless owner and its
headless consumers are pinned in ``tests/core/test_output_path_policy.py``.

Pinned here (handoff §5): the Image and NeXus wranglers resolve generated
targets through the shared owner, New/Overwrite and absent-target Append
generate ``.nexus``, an explicit existing legacy ``.nxs`` Append is retained,
H5Viewer lists/loads/restores/follows both suffixes, the scratch/default and
run-end fallback names come from the shared policy, and a function-scoped
census proves every authorized GUI generated-output constructor consumes the
owner.

Production-wired: real ``imageThread`` objects built by the real constructor,
the real unbound ``H5Viewer``/``staticWidget`` methods, and a real
``QListWidget`` — no fake stands on the seam under test (HARD RULE 2).
"""

from __future__ import annotations

import ast
import os
import threading
from pathlib import Path
from queue import Queue
from types import MethodType, SimpleNamespace

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

from xrd_tools.io import LEGACY_OUTPUT_SUFFIX, NEW_OUTPUT_SUFFIX
from xdart.modules.ewald.scan import LiveScan
from xrd_tools.core.containers import PONI
from xdart.gui.tabs.static_scan.h5viewer import H5Viewer
from xdart.gui.tabs.static_scan.static_scan_widget import (
    _finished_output_file, staticWidget,
)
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
from tests.xdart._accepted_run import (
    accepted_run,
    admitted_worker,
    directory_source,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

_LIVE_THREADS: list = []


@pytest.fixture(autouse=True)
def _teardown_live_threads():
    yield
    for t in _LIVE_THREADS:
        try:
            t.command = "stop"
            t._prefetch_stop_prior()
            t._eiger_close_master()
        except Exception:
            pass
    _LIVE_THREADS.clear()


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _make_thread(watch_dir, out_dir, *, scan_name="scan", write_mode="Overwrite",
                 include_subdir=False):
    """A REAL imageThread watching *watch_dir* and writing into *out_dir*."""
    scan = LiveScan("scan", data_file=str(Path(out_dir) / "scan.nexus"),
                    static=True)
    # R4-G retired the policy constructor arguments: the watched directory,
    # extension, recursion and output mode reach the worker only through the
    # accepted configuration admitted below — the same seam production
    # admission uses (E6/PM reconciliation of this canonical oracle).
    t = imageThread(
        Queue(), threading.RLock(), "",
        scan_name,                   # scan_name
        PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10),
        "",                          # img_file
        "None",                      # bg_type
        "", "", None, "", "",        # bg_*
        1.0, None,                   # bg_scale, bg_norm_channel
        "q_total", "qip_qoop",       # gi modes
        "start", scan,
    )
    admitted_worker(
        t,
        frozen=accepted_run(
            processing_mode="Int 2D",
            output_mode=write_mode,
            save_path=str(out_dir),
            source_spec=directory_source(
                watch_dir, ext="nxs", recursive=include_subdir),
        ),
    )
    _LIVE_THREADS.append(t)
    return t


def _write_processed(path: Path, *, n_frames: int = 2) -> Path:
    q = np.linspace(1.0, 5.0, 9, dtype=np.float32)
    frames = np.arange(n_frames, dtype=np.int64)
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["ssrl_schema_version"] = 2
        g = entry.create_group("integrated_1d")
        g.create_dataset("frame_index", data=frames)
        g.create_dataset("q", data=q)
        g.create_dataset("intensity",
                         data=np.ones((n_frames, len(q)), dtype=np.float32))
        fg = entry.create_group("frames")
        for frame in frames:
            grp = fg.create_group(f"frame_{frame:04d}")
            thumb = grp.create_dataset(
                "thumbnail", data=np.arange(12, dtype=np.uint8).reshape(3, 4))
            thumb.attrs["vmin"] = 10.0
            thumb.attrs["vmax"] = 265.0
            thumb.attrs["dtype"] = "uint8"
    return path


def _list_host(tmp_path, *, scan_name=None, viewer_mode="normal"):
    """A host driving the REAL update_scans over a REAL QListWidget."""
    lw = QtWidgets.QListWidget()
    host = SimpleNamespace(
        dirname=str(tmp_path),
        viewer_mode=viewer_mode,
        scan_name=scan_name,
        ui=SimpleNamespace(listScans=lw, dateSort=None),
        _update_scans_header=lambda: None,
        _date_sort_dir_cache={},
        _IMAGE_EXTS=H5Viewer._IMAGE_EXTS,
        _XYE_EXTS=H5Viewer._XYE_EXTS,
        _NEXUS_EXTS=H5Viewer._NEXUS_EXTS,
    )
    host._natural_sort_key = H5Viewer._natural_sort_key
    host._date_sort_mtime = MethodType(H5Viewer._date_sort_mtime, host)
    host.update_scans = MethodType(H5Viewer.update_scans, host)
    return host, lw


def _entries(lw):
    return [lw.item(i).text() for i in range(lw.count())]


# ---------------------------------------------------------------------------
# Row 2 (GUI) — New / Overwrite and absent-target Append generate ``.nexus``
# ---------------------------------------------------------------------------

def test_row2_gui_overwrite_generates_nexus(tmp_path):
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Overwrite")

    assert t._append_output_path(t.run_configuration, "scan_042") == os.fspath(
        out / "scan_042.nexus")


def test_row2_gui_append_with_no_existing_target_generates_nexus(tmp_path):
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Append")

    assert t._append_output_path(t.run_configuration, "scan_042") == os.fspath(
        out / "scan_042.nexus")


def test_row2_gui_initialize_scan_targets_nexus(tmp_path):
    """The real ``initialize_scan`` writes its output to ``<scan>.nexus``."""
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Overwrite")

    t.initialize_scan()

    assert Path(t.fname) == out / "scan_042.nexus"
    assert (out / "scan_042.nexus").exists()
    assert not (out / "scan_042.nxs").exists()


def test_row2_gui_overwrite_never_replaces_a_sibling_legacy_file(tmp_path):
    """§2 rule 2 at the GUI seam: an existing ``.nxs`` is left untouched."""
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    legacy = _write_processed(out / "scan_042.nxs")
    before = legacy.read_bytes()

    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Overwrite")
    t.initialize_scan()

    assert Path(t.fname) == out / "scan_042.nexus"
    assert legacy.read_bytes() == before


# ---------------------------------------------------------------------------
# Row 9 (GUI) — an explicit existing legacy ``.nxs`` Append is retained
# ---------------------------------------------------------------------------

def test_row9_gui_append_reuses_an_existing_legacy_target(tmp_path):
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    legacy = _write_processed(out / "scan_042.nxs")

    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Append")

    assert t._append_output_path(t.run_configuration, "scan_042") == os.fspath(legacy)


def test_row9_gui_append_run_end_fallback_finds_the_legacy_target(tmp_path):
    """``_finished_output_file`` reports the legacy file an Append reused."""
    legacy = _write_processed(tmp_path / "scan_042.nxs")
    thread = admitted_worker(
        SimpleNamespace(
            _append_skip_frames_by_scan={"scan_042": {1, 2}},
            _append_output_path=None,
        ),
        processing_mode="Int 2D",
        save_path=str(tmp_path),
        source_spec=directory_source(tmp_path),
    )
    wrangler = SimpleNamespace(thread=thread, fname=None)

    found = _finished_output_file(thread, wrangler, all_skipped_append=True)

    assert found is not None
    assert Path(found) == legacy


# ---------------------------------------------------------------------------
# Row 10 (GUI) — both siblings exist: ``.nexus`` wins
# ---------------------------------------------------------------------------

def test_row10_gui_append_prefers_nexus_when_both_siblings_exist(tmp_path):
    watch, out = tmp_path / "raw", tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    _write_processed(out / "scan_042.nxs")
    _write_processed(out / "scan_042.nexus")

    t = _make_thread(watch, out, scan_name="scan_042", write_mode="Append")

    assert t._append_output_path(t.run_configuration, "scan_042") == os.fspath(
        out / "scan_042.nexus")


# ---------------------------------------------------------------------------
# Row 6 (GUI) — H5Viewer explicitly opens ``.nexus``
# ---------------------------------------------------------------------------

def test_row6_gui_viewer_extension_sets_accept_nexus():
    assert NEW_OUTPUT_SUFFIX in H5Viewer._NEXUS_EXTS
    assert NEW_OUTPUT_SUFFIX in H5Viewer._IMAGE_EXTS
    assert LEGACY_OUTPUT_SUFFIX in H5Viewer._NEXUS_EXTS
    assert LEGACY_OUTPUT_SUFFIX in H5Viewer._IMAGE_EXTS


def test_row6_gui_nexus_viewer_mode_lists_both_suffixes(qapp, tmp_path):
    _write_processed(tmp_path / "a.nexus")
    _write_processed(tmp_path / "b.nxs")
    (tmp_path / "c.txt").write_text("x")

    host, lw = _list_host(tmp_path, viewer_mode="nexus")
    host.update_scans()

    assert _entries(lw) == ["..", "a.nexus", "b.nxs"]


def test_row6_gui_image_viewer_mode_lists_both_suffixes(qapp, tmp_path):
    _write_processed(tmp_path / "a.nexus")
    _write_processed(tmp_path / "b.nxs")

    host, lw = _list_host(tmp_path, viewer_mode="image")
    host.update_scans()

    assert _entries(lw) == ["..", "a.nexus", "b.nxs"]


# ---------------------------------------------------------------------------
# Row 7 — GUI run-end listing / follow / restore works for ``.nexus``
# ---------------------------------------------------------------------------

def test_row7_normal_mode_lists_nexus_and_legacy(qapp, tmp_path):
    _write_processed(tmp_path / "scan_1.nexus")
    _write_processed(tmp_path / "scan_2.nxs")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()

    host, lw = _list_host(tmp_path)
    host.update_scans()

    assert _entries(lw) == ["..", "scan_1.nexus", "scan_2.nxs", "sub/"]


def test_row7_follow_selects_the_current_nexus_scan(qapp, tmp_path):
    _write_processed(tmp_path / "scan_1.nexus")
    _write_processed(tmp_path / "scan_2.nexus")

    host, lw = _list_host(tmp_path, scan_name="scan_2")
    host.update_scans()

    assert lw.currentItem() is not None
    assert lw.currentItem().text() == "scan_2.nexus"


def test_row7_follow_still_selects_a_legacy_scan(qapp, tmp_path):
    _write_processed(tmp_path / "scan_1.nxs")
    _write_processed(tmp_path / "scan_2.nxs")

    host, lw = _list_host(tmp_path, scan_name="scan_2")
    host.update_scans()

    assert lw.currentItem() is not None
    assert lw.currentItem().text() == "scan_2.nxs"


def test_row7_follow_tolerates_a_frame_count_suffix_on_a_nexus_scan(
        qapp, tmp_path):
    """The fuzzy ``<scan>_N`` form must still resolve to ``<scan>.nexus``."""
    _write_processed(tmp_path / "scan.nexus")

    host, lw = _list_host(tmp_path, scan_name="scan_5")
    host.update_scans()

    assert lw.currentItem() is not None
    assert lw.currentItem().text() == "scan.nexus"


def test_row7_restore_selection_targets_a_nexus_scan(qapp, tmp_path):
    _write_processed(tmp_path / "scan_1.nexus")
    _write_processed(tmp_path / "scan_2.nexus")

    host, lw = _list_host(tmp_path)
    host.update_scans()
    host.scan_name = "scan_2"
    host.file_thread = None
    host._restore_loaded_scan_selection = MethodType(
        H5Viewer._restore_loaded_scan_selection, host)

    assert host._restore_loaded_scan_selection() is True
    assert lw.currentItem().text() == "scan_2.nexus"


def test_row7_restore_selection_still_targets_a_legacy_scan(qapp, tmp_path):
    _write_processed(tmp_path / "scan_1.nxs")

    host, lw = _list_host(tmp_path)
    host.update_scans()
    host.scan_name = "scan_1"
    host.file_thread = None
    host._restore_loaded_scan_selection = MethodType(
        H5Viewer._restore_loaded_scan_selection, host)

    assert host._restore_loaded_scan_selection() is True
    assert lw.currentItem().text() == "scan_1.nxs"


def test_row7_run_end_fallback_finds_a_nexus_output(tmp_path):
    output = _write_processed(tmp_path / "scan_042.nexus")
    thread = admitted_worker(
        SimpleNamespace(
            _append_skip_frames_by_scan={"scan_042": {1}},
            _append_output_path=None,
        ),
        processing_mode="Int 2D",
        save_path=str(tmp_path),
        source_spec=directory_source(tmp_path),
    )
    wrangler = SimpleNamespace(thread=thread, fname=None)

    found = _finished_output_file(thread, wrangler, all_skipped_append=True)

    assert found is not None
    assert Path(found) == output


def test_row7_canonical_run_token_strips_the_new_suffix():
    assert staticWidget._canonical_run_token("scan_042.nexus") == "scan_042"
    assert staticWidget._canonical_run_token("scan_042.nxs") == "scan_042"


# ---------------------------------------------------------------------------
# Row 15 (GUI) — function-scoped owner census
# ---------------------------------------------------------------------------

#: (relative source path, qualified function) for every authorized GUI
#: generated-output constructor.
GUI_PRODUCERS = (
    ("xdart/gui/tabs/static_scan/h5viewer.py", "H5Viewer.update_scans"),
    ("xdart/gui/tabs/static_scan/h5viewer.py",
     "H5Viewer._restore_loaded_scan_selection"),
    ("xdart/gui/tabs/static_scan/static_scan_widget.py",
     "_finished_output_file"),
    ("xdart/gui/tabs/static_scan/static_scan_widget.py",
     "staticWidget._init_data_objects"),
    ("xdart/gui/tabs/static_scan/wranglers/image_wrangler.py",
     "imageWrangler._candidate_append_target_file"),
    ("xdart/gui/tabs/static_scan/wranglers/image_wrangler.py",
     "imageWrangler.setup"),
    ("xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py",
     "imageThread._append_output_path"),
    ("xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py",
     "imageThread.initialize_scan"),
    ("xdart/gui/tabs/static_scan/wranglers/nexus_wrangler.py",
     "nexusWrangler.setup"),
)

#: Only the owner's real API names count as "consumes the owner".  A bare
#: ``output_path`` substring is deliberately NOT here: it matches unrelated
#: local names such as ``_append_output_path``, which would pass the census
#: without consuming anything.
OWNER_NAMES = ("resolve_output_target", "default_output_path",
               "NEW_OUTPUT_SUFFIX", "READABLE_OUTPUT_SUFFIXES",
               "is_readable_output_path")


def _function_node(path: Path, qualname: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node: ast.AST = tree
    for part in qualname.split("."):
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef))
                    and child.name == part):
                node = child
                break
        else:                                       # pragma: no cover
            raise AssertionError(f"{qualname} not found in {path}")
    return node


def _legacy_literals(path: Path, qualname: str) -> list[str]:
    """Non-docstring string constants inside *qualname* naming ``.nxs``.

    AST-based, so a comment or docstring mentioning the legacy suffix is not a
    false positive while every shape of hard-coded legacy target is caught.
    """
    node = _function_node(path, qualname)
    docstrings = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.ClassDef, ast.Module)):
            body = getattr(sub, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    return [sub.value for sub in ast.walk(node)
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            and id(sub) not in docstrings and ".nxs" in sub.value.lower()]


@pytest.mark.parametrize("rel_path,qualname", GUI_PRODUCERS)
def test_row15_gui_producer_consumes_the_shared_owner(rel_path, qualname):
    source = ast.get_source_segment(
        (SRC / rel_path).read_text(encoding="utf-8"),
        _function_node(SRC / rel_path, qualname)) or ""
    assert any(name in source for name in OWNER_NAMES), (
        f"{rel_path}::{qualname} does not consume the shared path owner")


@pytest.mark.parametrize("rel_path,qualname", GUI_PRODUCERS)
def test_row15_gui_producer_has_no_hard_coded_legacy_target(rel_path, qualname):
    found = _legacy_literals(SRC / rel_path, qualname)
    assert found == [], (
        f"{rel_path}::{qualname} still builds a hard-coded legacy target: {found}")


def test_row15_raw_source_pickers_keep_their_literal_nxs():
    """``.nexus`` must NOT leak into a RAW source picker or a raw watch config.

    These are raw *source* checks, explicitly allowed to remain literal (§5
    row 15) and required to stay literal by §2 rule 7 — ``.nexus`` is
    output-only and excluded from raw discovery.
    """
    picker = (SRC / "xdart/gui/tabs/static_scan/scan_source_widget.py"
              ).read_text(encoding="utf-8")
    assert "*.nxs" in picker
    # The glob filter specifically — the file legitimately mentions the word
    # "nexus" (source-kind labels, the ``xrd_tools.io.nexus`` import).
    assert "*.nexus" not in picker

    ssw = SRC / "xdart/gui/tabs/static_scan/static_scan_widget.py"
    watch = ast.get_source_segment(
        ssw.read_text(encoding="utf-8"),
        _function_node(ssw, "staticWidget._controls_v2_container_index_config"),
    ) or ""
    assert 'suffixes = (".nxs",)' in watch
    assert NEW_OUTPUT_SUFFIX not in watch
