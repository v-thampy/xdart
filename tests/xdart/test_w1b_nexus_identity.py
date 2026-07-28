"""O-1a-W1B — NeXus admission parity and durable run identity.

Frozen acceptance oracle for W-1.2 cases 6, 7, 8, 9, 10 and 12 (the W-1 ratified
immediate-start packet).  Written BEFORE any production edit and red at the
accepted parent ``e754c379``.

Production-wired throughout (CLAUDE.md rule 2): the real ``staticWidget``, the
real ``nexusWrangler`` taken from the real wrangler stack, the real
``imageThread``, the real Controls freeze owner, and — for case 10 — a real
NeXus write and reload.  Case 12 is an architecture guard that asserts a simple
fact about the actual production tree (rule 9), not a taint analyzer: it
enumerates every mutable-display-scan access left in the two worker modules and
requires each one to appear in the justified-survivor table embedded beside it.

Case map (W-1.2):

6  a second NeXus Start while active/stopping is a zero-delta typed refusal
7  NeXus prepare/adopt publishes the exact accepted object and clears pending
   ownership exactly once
8  absent / foreign / stale adoption refuses visibly and emits a structured event
9  container-versus-series selection follows the FROZEN source kind
10 one ``(generation, fingerprint)`` through click -> wrapper -> thread -> plan
   -> worker scan -> persisted provenance -> reload
12 zero execution-time reads of the mutable display scan, with the explicit
   justified-survivor list embedded beside the guard
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pyqtgraph.Qt import QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    value = staticWidget()
    try:
        yield value
    finally:
        try:
            value._exit_run_state(value._new_projection_receipt())
        except Exception:
            pass
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _write_poni(path, dist=0.10):
    path.write_text(
        f"Distance: {dist}\nPoni1: 0.01\nPoni2: 0.02\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n"
    )
    return str(path)


def _select_nexus_wrangler(widget):
    """Switch the REAL wrangler stack to the NeXus wrangler (production path)."""
    from xdart.gui.tabs.static_scan.wranglers import nexusWrangler

    stack = widget.ui.wranglerStack
    for index in range(stack.count()):
        if isinstance(stack.widget(index), nexusWrangler):
            stack.setCurrentIndex(index)
            widget.set_wrangler(index)
            assert widget.wrangler is stack.widget(index)
            return widget.wrangler
    raise AssertionError("no nexusWrangler in the production wrangler stack")


def _prepare_nexus_inputs(widget, tmp_path):
    """The minimum a NeXus run needs: a calibration, a source, an output dir."""
    wrangler = widget.wrangler
    poni_path = _write_poni(tmp_path / "cal.poni")
    source = tmp_path / "raw_scan.nxs"
    source.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    wrangler.parameters.child("Calibration", "poni_file").setValue(poni_path)
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(source))
    wrangler.parameters.child("Output", "h5_dir").setValue(str(out))
    return source, out


def _nexus_snapshot(widget):
    """Everything a refused NeXus Start must leave untouched."""
    wrangler = widget.wrangler
    thread = getattr(wrangler, "thread", None)
    return {
        "command": getattr(wrangler, "command", None),
        "thread_command": getattr(thread, "command", None),
        "thread": thread,
        "thread_max_cores": getattr(thread, "max_cores", None),
        "start_enabled": wrangler.startButton.isEnabled(),
        "stop_enabled": wrangler.stopButton.isEnabled(),
        "run_configuration": getattr(wrangler, "run_configuration", None),
        "thread_run_configuration": getattr(thread, "run_configuration", None),
        "pending": getattr(
            widget, "_pending_controls_v2_run_configuration", None),
        "intent_generation": int(
            widget._controls_v2_ensure_run_intent().generation),
    }


# --------------------------------------------------------------------------- #
# Case 6 — a second Start while active/stopping is a zero-delta typed refusal.
# --------------------------------------------------------------------------- #

def test_second_nexus_start_while_active_is_a_zero_delta_refusal(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 6.  ``nexusWrangler.start()`` sets ``command='start'`` on its
    FIRST line today, so a Start during the stopping window mutates the command,
    the worker's cores, the mode flags, both buttons and the session before
    anything can refuse.  The image path refuses first; NeXus must too."""
    _select_nexus_wrangler(widget)
    _prepare_nexus_inputs(widget, tmp_path)

    saves = []
    real_save = widget.wrangler._save_to_session
    monkeypatch.setattr(widget.wrangler, "_save_to_session",
                        lambda *a, **k: (saves.append(True), real_save(*a, **k))[1])
    emitted = []
    widget.wrangler.sigStart.connect(lambda: emitted.append(True))
    widget.wrangler.command = "stop"
    widget.wrangler.thread.command = "stop"

    # the fast-Start window: the worker is still unwinding
    monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: True)
    assert widget._controls_v2_active_run_owner() is not None

    before = _nexus_snapshot(widget)
    widget.wrangler.start()
    after = _nexus_snapshot(widget)

    assert after == before, "a refused NeXus Start changed run state"
    assert emitted == [], "a refused NeXus Start emitted sigStart"
    assert saves == [], "a refused NeXus Start wrote the session"


def test_refused_nexus_start_reports_the_owner(widget, tmp_path, monkeypatch):
    """W-1.2 case 6 (typed half).  The refusal is reported through the same
    owner-labelled status seam the image path uses."""
    _select_nexus_wrangler(widget)
    _prepare_nexus_inputs(widget, tmp_path)
    seen = []
    monkeypatch.setattr(widget.wrangler, "_set_status_text",
                        lambda text: seen.append(text), raising=False)
    monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: True)

    widget.wrangler.start()

    assert seen, "the refusal was silent"
    assert "still stopping" in seen[-1].lower() or "refused" in seen[-1].lower()


# --------------------------------------------------------------------------- #
# Case 7 — NeXus owns prepare/adopt of the exact accepted object.
# --------------------------------------------------------------------------- #

def test_nexus_start_publishes_the_exact_frozen_object_to_every_owner(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 7.  One freeze, then THAT object on the wrangler and on the
    worker thread that will actually run -- ``nexusWrangler.setup()`` REPLACES
    the thread, so the newly built worker must carry the accepted identity."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    _select_nexus_wrangler(widget)
    _prepare_nexus_inputs(widget, tmp_path)
    started = []
    monkeypatch.setattr(nexusThread, "start", lambda self: started.append(True))

    widget.wrangler.start()

    frozen = widget.wrangler.run_configuration
    assert frozen is not None, "the NeXus Start published no frozen configuration"
    assert started == [True]
    assert widget.wrangler.thread.run_configuration is frozen
    assert widget._pending_controls_v2_run_configuration is None


def test_nexus_pending_ownership_is_consumed_exactly_once(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 7 (pending half).  The pending slot is cleared on adoption, so
    a later consumer cannot inherit the earlier click's configuration."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    _select_nexus_wrangler(widget)
    _prepare_nexus_inputs(widget, tmp_path)
    monkeypatch.setattr(nexusThread, "start", lambda self: None)

    widget.wrangler.start()
    first = widget.wrangler.run_configuration
    assert widget._pending_controls_v2_run_configuration is None

    # the first run finishes through the production run-end owner; a second Start
    # while it is still live is the case-6 refusal, not a second adoption.
    widget._exit_run_state(widget._new_projection_receipt())
    widget.wrangler.start()
    second = widget.wrangler.run_configuration

    assert second is not first
    assert int(second.generation) > int(first.generation)
    assert widget.wrangler.thread.run_configuration is second


# --------------------------------------------------------------------------- #
# Case 8 — absent / foreign / stale adoption refuses visibly + structurally.
# --------------------------------------------------------------------------- #

def test_nexus_absent_configuration_refuses_visibly_and_structurally(
        widget, tmp_path, monkeypatch, caplog):
    """W-1.2 case 8.  No accepted configuration -> a visible refusal and a
    structured refusal event; never a legacy fallback run."""
    import logging

    _select_nexus_wrangler(widget)
    _prepare_nexus_inputs(widget, tmp_path)
    seen = []
    monkeypatch.setattr(widget.wrangler, "_set_status_text",
                        lambda text: seen.append(text), raising=False)
    emitted = []
    widget.wrangler.sigStart.connect(lambda: emitted.append(True))
    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "0")

    with caplog.at_level(logging.WARNING):
        widget.wrangler.start()

    assert emitted == [], "a refused NeXus Start started the run"
    assert seen, "the refusal was not visible"
    assert any("run_configuration_refused" in record.getMessage()
               for record in caplog.records), "no structured refusal event"


def test_nexus_worker_refuses_a_foreign_configuration(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 8 (foreign).  A duck-typed carrier is refused by the typed
    admission owner, not consumed."""
    from xrd_tools.session import RunConfigurationRefused, require_run_configuration

    _select_nexus_wrangler(widget)
    _prepare_nexus_inputs(widget, tmp_path)
    thread = widget.wrangler.thread
    thread.run_configuration = SimpleNamespace(generation=3, fingerprint="x")

    with pytest.raises(RunConfigurationRefused) as excinfo:
        require_run_configuration(thread.run_configuration, stage="nexus-worker")

    assert excinfo.value.reason == "foreign"


def test_nexus_worker_refuses_a_stale_generation(widget, tmp_path, monkeypatch):
    """W-1.2 case 8 (stale).  A configuration frozen for an earlier click is
    refused against the object this run admitted.

    RULE-9 CORRECTION (W-1R, review §39.2 W1R-P1-1).  This case originally
    expressed the refusal as ``generation < floor`` against a bare integer
    watermark.  §39.2 W1R-P1-1 rules that comparison out for CONSUMPTION -- a
    floor cannot tell the object this click published from a different genuine
    object at the same generation -- and §39.5 Phase 1 item 3 replaces it with an
    exact expected-object comparison.  The INTENT is unchanged and still
    asserted: an earlier click's configuration is refused ``stale``, and the
    admitted object passes.  Only the API expressing it moved."""
    from xrd_tools.session import RunConfigurationRefused, require_run_configuration
    from xrd_tools.session import RunIntent

    intent = RunIntent(processing_mode="Int 2D", output_mode="Append")
    stale = intent.freeze()
    fresh = intent.freeze()
    assert int(fresh.generation) > int(stale.generation)

    with pytest.raises(RunConfigurationRefused) as excinfo:
        require_run_configuration(
            stale, stage="nexus-worker", expected=fresh)

    assert excinfo.value.reason == "stale"
    assert require_run_configuration(
        fresh, stage="nexus-worker", expected=fresh) is fresh


def test_nexus_worker_run_refuses_absent_configuration_before_any_read(
        qapp, tmp_path, monkeypatch):
    """W-1.2 case 8 (worker entry).  ADDED AFTER THE FREEZE and disclosed as
    such: it completes case 8's worker leg (mutation N5 showed the frozen nine
    never drove ``nexusThread.run()`` itself).  A worker without the accepted
    configuration reads no source, opens no output and starts no session."""
    import threading
    from queue import Queue

    from xrd_tools.core.containers import PONI
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )
    from xdart.modules.live import LiveScan

    out = tmp_path / "out"
    out.mkdir()
    source = tmp_path / "raw.nxs"
    source.write_bytes(b"")
    scan = LiveScan("scan", data_file=str(out / "scan.nxs"), static=True)
    # R4-G: mask + GI arguments retired from this signature (frozen-owned).
    thread = nexusThread(
        Queue(), threading.RLock(), str(out / "scan.nxs"),
        str(source),
        PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10),
        "q_total", "qip_qoop",
        "start", scan,
    )
    assert thread.run_configuration is None
    reached = []
    monkeypatch.setattr(thread, "_run_impl", lambda: reached.append(True))

    thread.run()

    assert reached == [], "the NeXus worker ran without an accepted configuration"
    assert thread.command == "stop"
    assert not (out / "scan.nxs").exists()


# --------------------------------------------------------------------------- #
# Case 9 — container-versus-series follows the frozen source kind.
# --------------------------------------------------------------------------- #

def _frozen_with_source(source_spec):
    """Freeze through the PRODUCTION owner with one typed source selection."""
    from xrd_tools.session import RunIntent

    return RunIntent(source_spec=source_spec, processing_mode="Int 2D").freeze()


def test_container_selection_follows_the_frozen_directory_source(
        widget, monkeypatch, tmp_path):
    """W-1.2 case 9.  A frozen container-directory source takes the container
    reader even though the mutable ``img_ext`` panel mirror says ``tif``
    (R4B-14: the format authority is the frozen source kind, not that mirror)."""
    from xrd_tools.sources import DirectorySourceSpec

    thread = widget.wrangler.thread
    frozen = _frozen_with_source(DirectorySourceSpec(
        root=Path(tmp_path), recursive=False, suffixes=(".h5",)))
    thread.run_configuration = frozen
    thread.img_ext = "tif"
    thread.img_file = ""
    thread.inp_type = "Image Directory"
    thread.single_img = False
    thread.img_fnames = []
    thread.processed = []
    container = []
    monkeypatch.setattr(thread, "_get_next_eiger_frame",
                        lambda _frozen: container.append(True) or (None, "s", 1, None, {}))

    thread.get_next_image(thread.run_configuration)

    assert container == [True], "the frozen container source did not select the "\
        "container reader"


def test_series_selection_follows_the_frozen_series_source(
        widget, monkeypatch, tmp_path):
    """W-1.2 case 9 (other polarity).  A frozen image-series source takes the
    series reader even though the mutable ``img_ext`` mirror says ``h5``."""
    from xrd_tools.sources import image_series_spec

    paths = [tmp_path / f"scan_{idx:04d}.tif" for idx in (1, 2)]
    for path in paths:
        path.touch()
    thread = widget.wrangler.thread
    frozen = _frozen_with_source(image_series_spec(paths[0]))
    thread.run_configuration = frozen
    thread.source_spec = frozen.thaw_source_spec()
    thread.img_ext = "h5"
    thread.img_file = str(paths[0])
    thread.inp_type = "Image Series"
    thread.single_img = False
    thread.img_fnames = []
    thread.processed = []
    thread.scan_name = "scan"
    thread.meta_ext = None
    container = []
    monkeypatch.setattr(thread, "_get_next_eiger_frame",
                        lambda _frozen: container.append(True) or (None, "s", 1, None, {}))
    import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as iwt
    import numpy as np
    monkeypatch.setattr(iwt, "read_image",
                        lambda path: np.ones((2, 2), dtype=float))

    img_file, scan_name, img_number, img_data, _meta = thread.get_next_image(thread.run_configuration)

    assert container == [], "the frozen series source selected the container reader"
    assert Path(img_file).name == paths[0].name


# --------------------------------------------------------------------------- #
# Case 9 (W-1C) — the SAME decision, driven by the REAL freeze owner.
#
# The two rows above hand-build ``DirectorySourceSpec(suffixes=(".h5",))``.  That
# spelling is one production NEVER emits: ``_controls_v2_container_index_config``
# emits MATCH SUFFIXES (``("_master.h5",)`` for File Type h5, ``(".nxs",)`` for
# nxs), so a consumer that compares whole suffix strings against bare extensions
# passes a literal-driven test and still refuses every real Eiger-master
# directory.  These rows consume the emitter's actual output instead.
# --------------------------------------------------------------------------- #

def _configure_container_directory(widget, root, ext):
    """Configure the REAL Source panel for a container-directory run."""
    signal = widget.wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(root))
    signal.child("img_ext").setValue(ext)
    assert signal.child("img_ext").value() == ext, (
        f"File Type {ext!r} is not selectable in this build")



def _publish_admitted(widget, frozen):
    """O-1a-W1R-D1 (review §40.3 D1 items 5-6): ``_prepare_...`` now only STAGES
    the one tentative object -- publishing there is what made the zero-delta
    refusal claim false (§40.1 P1-D).  Bind through the same production step
    admission uses, so these cases still drive real code to reach the worker.
    """
    if frozen is None:
        return None
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        wranglerWidget,
    )
    wranglerWidget._bind_admitted_run_configuration(widget.wrangler, frozen)
    thread = getattr(widget.wrangler, "thread", None)
    if thread is not None:
        wranglerWidget._bind_admitted_run_configuration(thread, frozen)
    return frozen

@pytest.mark.parametrize("ext", ["h5", "nxs"])
def test_frozen_container_directory_from_the_real_freeze_owner(
        widget, tmp_path, monkeypatch, ext):
    """W-1.2 case 9, end to end through production: the File Type the operator
    picks -> the emitter's match suffixes -> the frozen source -> the container
    reader.  No literal suffix appears anywhere in this test."""
    _configure_container_directory(widget, tmp_path, ext)
    emitted = widget._controls_v2_container_index_config()
    assert emitted is not None, "the production emitter refused this File Type"
    suffixes = emitted[3]

    frozen = _publish_admitted(
        widget, widget._prepare_controls_v2_run_configuration())
    assert frozen is not None
    assert frozen.source is not None and frozen.source.family == "directory"
    assert frozen.source.suffixes == tuple(suffixes), (
        "the frozen source did not carry the emitter's own suffixes")

    thread = widget.wrangler.thread
    assert thread.run_configuration is frozen
    assert imageThread._frozen_source_is_container(thread, thread.run_configuration) is True, (
        f"a frozen {ext} container directory was not recognised as a container; "
        f"emitter suffixes were {suffixes!r}")


@pytest.mark.parametrize("ext", ["h5", "nxs"])
def test_live_directory_watch_reaches_the_container_reader(
        widget, tmp_path, monkeypatch, ext):
    """W-1.2 case 9, the consequence that matters.  A Live directory run
    intentionally starts with an EMPTY ``img_file``
    (``imageWrangler._directory_run_without_seed_ok``), so ``is_master`` is False
    and the container decision rests entirely on the frozen source.  If it says
    "series", every ``*_master.h5`` container is handed to the single-image
    reader and its frames are silently lost."""
    _configure_container_directory(widget, tmp_path, ext)
    frozen = _publish_admitted(
        widget, widget._prepare_controls_v2_run_configuration())
    assert frozen is not None

    thread = widget.wrangler.thread
    thread.img_file = ""
    thread.single_img = False
    thread.inp_type = "Image Directory"
    thread.img_fnames = []
    thread.processed = []
    container = []
    monkeypatch.setattr(
        thread, "_get_next_eiger_frame",
        lambda _frozen: container.append(True) or (None, "s", 1, None, {}))

    thread.get_next_image(thread.run_configuration)

    assert container == [True], (
        "the Live container watch fell through to the plain-file branch")


def test_every_emitter_branch_is_accepted_as_a_container(widget, tmp_path):
    """W-1.2 case 9 (complete emitter coverage, including the ``hdf5`` branch).

    ``_controls_v2_container_index_config`` has three branches; the visible File
    Type list is ``['tif','raw','h5','nxs','mar3450']``, so its ``hdf5`` branch —
    which emits the TWO-suffix tuple ``("_master.hdf5", "_master.h5")`` — cannot
    currently be selected through the panel.  Rather than assert a literal, this
    row extends the list parameter's limits so the PRODUCTION emitter runs its own
    hdf5 branch, and then requires the consumer to accept whatever it emits.
    Disclosed: extending the limits is a test-side widening of the visible
    choices, not a substitute for the emitter.
    """
    signal = widget.wrangler.parameters.child("Signal")
    limits = list(signal.child("img_ext").opts.get("limits") or [])
    signal.child("img_ext").setLimits(limits + ["hdf5"])

    observed = {}
    for ext in ("h5", "hdf5", "nxs"):
        _configure_container_directory(widget, tmp_path, ext)
        emitted = widget._controls_v2_container_index_config()
        assert emitted is not None, ext
        suffixes = emitted[3]
        observed[ext] = suffixes
        frozen = _publish_admitted(
        widget, widget._prepare_controls_v2_run_configuration())
        thread = widget.wrangler.thread
        assert imageThread._frozen_source_is_container(thread, thread.run_configuration) is True, (
            f"emitter branch {ext} -> {suffixes!r} was refused")

    # the branch shapes this row actually exercised, recorded as evidence
    assert observed["h5"] == ("_master.h5",)
    assert observed["hdf5"] == ("_master.hdf5", "_master.h5")
    assert observed["nxs"] == (".nxs",)


def test_a_non_container_directory_is_still_a_series(widget, tmp_path):
    """W-1.2 case 9 (the other polarity, through production).  A tif directory
    freezes a TYPED directory source and must stay on the non-container path.

    O-1a-W1R-D1 (review §41.4 item 3): this case asserted ``source is None``,
    which was the behavior §40.3 D0 item 8 ordered replaced -- a plain-image
    Directory was selectable in the production File Type list yet froze no source,
    so every source-shape consumer fell back to legacy inference.  The contract is
    now a typed ``DirectorySourceSpec`` whose format keeps it off the container
    reader.
    """
    signal = widget.wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(tmp_path))
    signal.child("img_ext").setValue("tif")
    # The CONTAINER emitter still refuses it; that is what makes it non-container.
    assert widget._controls_v2_container_index_config() is None

    frozen = _publish_admitted(
        widget, widget._prepare_controls_v2_run_configuration())
    assert frozen is not None
    assert frozen.source is not None
    assert frozen.source.family == "directory"
    assert frozen.source.suffixes == (".tif",)
    thread = widget.wrangler.thread
    thread.img_file = ""
    assert imageThread._frozen_source_is_container(thread, thread.run_configuration) is False


# --------------------------------------------------------------------------- #
# Case 10 — the one-identity trace, click to reload.
# --------------------------------------------------------------------------- #

def test_one_identity_from_the_run_click_through_reload(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 10.  ONE ``(generation, fingerprint)`` at every hop: the Run
    click, the wrapper, the worker thread, the reduction plan's input, the
    per-run scan, the written provenance and a real reload."""
    from xrd_tools.core.provenance import read_provenance

    poni_path = _write_poni(tmp_path / "cal.poni")
    raw_path = tmp_path / "scan_0001.tif"
    raw_path.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()
    widget._set_poni_field(poni_path)
    widget.wrangler.parameters.child("Project", "h5_dir").setValue(str(out))
    # O-1a-W1R-D1 (review §40.1 P1-A, §40.3 D1 items 1-2): arm the
    # AUTHORITATIVE Source card.  Setting only ``img_file`` is the shorthand
    # shape §40.1 named -- a truthful ``get_img_fname`` sync clears a cursor the
    # Source card does not back, and admission now requires a typed source.
    _signal = widget.wrangler.parameters.child("Signal")
    _signal.child("inp_type").setValue("Image Series")
    _signal.child("File").setValue(str(raw_path))
    widget.wrangler.img_file = str(raw_path)
    widget.controls.set_write_mode("Overwrite")
    widget._on_controls_v2_field_changed(("Int1D", "points"), 404)

    started = []
    monkeypatch.setattr(widget.wrangler.thread, "start",
                        lambda: started.append(True))
    widget.wrangler.start()
    assert started == [True]

    hops = {}
    frozen = widget.wrangler.run_configuration
    hops["wrapper"] = frozen
    thread = widget.wrangler.thread
    hops["thread"] = thread.run_configuration
    thread.h5_dir = str(out)
    thread.scan_name = "scan"

    scan = thread.initialize_scan()
    hops["worker_scan"] = scan.run_configuration
    plan = thread._plan_cache.get(scan, integrate_2d=not frozen.skip_2d)
    assert plan is not None, "the production plan builder was never reached"
    hops["plan"] = getattr(thread._plan_cache.plan_builder,
                           "run_configuration", None)

    for name, value in hops.items():
        assert value is frozen, f"hop {name} carried a different object"

    identity = frozen.identity
    assert (int(scan.run_configuration_generation),
            scan.run_configuration_fingerprint) == identity

    stored = (read_provenance(scan.data_file).get("config") or {}).get(
        "run_configuration")
    assert stored is not None, "the run identity was not persisted"
    assert (int(stored["generation"]), stored["fingerprint"]) == identity


# --------------------------------------------------------------------------- #
# Case 12 — zero execution-time display-scan reads + the justified survivors.
# --------------------------------------------------------------------------- #

#: The COMPLETE justified-survivor table.  Every remaining ``self.scan`` access in
#: the two worker modules must appear here with its W-1.2 disposition.  A new
#: access fails this guard until it is dispositioned in the boundary report.
#:
#: key   -> (module, enclosing function, attribute or ``<self.scan>`` for a bare
#:           use, "read"/"write")
#: value -> the W-1.2 disposition
JUSTIFIED_DISPLAY_SCAN_SURVIVORS = {
    # the display handle itself: retained ONLY so the backward acquisition
    # projection below has a target.
    ("image_wrangler_thread.py", "__init__", "<self.scan>", "write"):
        "ALLOW_BACKWARD_DISPLAY_WRITE",
    ("nexus_wrangler_thread.py", "__init__", "<self.scan>", "write"):
        "ALLOW_BACKWARD_DISPLAY_WRITE",
    # the retained backward GI-mode projection (display/acquisition only).
    ("image_wrangler_thread.py", "_project_gi_modes_onto_display_scan",
     "<self.scan>", "read"): "ALLOW_BACKWARD_DISPLAY_WRITE",
    ("nexus_wrangler_thread.py", "_project_gi_modes_onto_display_scan",
     "<self.scan>", "read"): "ALLOW_BACKWARD_DISPLAY_WRITE",
    ("image_wrangler_thread.py", "_project_gi_modes_onto_display_scan",
     "bai_1d_args", "read"): "ALLOW_BACKWARD_DISPLAY_WRITE",
    ("image_wrangler_thread.py", "_project_gi_modes_onto_display_scan",
     "bai_2d_args", "read"): "ALLOW_BACKWARD_DISPLAY_WRITE",
    ("nexus_wrangler_thread.py", "_project_gi_modes_onto_display_scan",
     "bai_1d_args", "read"): "ALLOW_BACKWARD_DISPLAY_WRITE",
    ("nexus_wrangler_thread.py", "_project_gi_modes_onto_display_scan",
     "bai_2d_args", "read"): "ALLOW_BACKWARD_DISPLAY_WRITE",
    # O-3N.R.1 §16.1: the NeXus worker no longer executes on the display scan at
    # all -- ``_initialize_scan`` builds a FRESH per-run ``LiveScan``, as the
    # image worker does, so every ``self.scan`` write it used to make is gone.
    # The residual that carried them is closed, not re-dispositioned.
}


def _display_scan_accesses(path: Path):
    """Every ``self.scan`` access in one worker module, by enclosing function."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]

    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                functions[id(child)] = node.name

    found = []
    for node in ast.walk(tree):
        # ``getattr(self, "scan", ...)`` is the same access wearing a hat; count
        # it, or the guard could be satisfied by spelling the read differently.
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "scan"):
            found.append((path.name, functions.get(id(node), "<module>"),
                          "<self.scan>", "read"))
            continue
        if not isinstance(node, ast.Attribute) or node.attr != "scan":
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            continue
        func = functions.get(id(node), "<module>")
        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.Attribute):
            mode = "write" if isinstance(parent.ctx, ast.Store) else "read"
            found.append((path.name, func, parent.attr, mode))
        else:
            mode = "write" if isinstance(node.ctx, ast.Store) else "read"
            found.append((path.name, func, "<self.scan>", mode))
    return sorted(set(found))


def test_no_execution_time_display_scan_reads_outside_the_survivor_list():
    """W-1.2 case 12.  The acceptance grep, as an assertion about the production
    tree: ZERO worker reads of the mutable display scan for run configuration.
    Every remaining access is a named, dispositioned survivor."""
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread

    root = Path(image_wrangler_thread.__file__).parent
    observed = set()
    for name in ("image_wrangler_thread.py", "nexus_wrangler_thread.py"):
        observed.update(_display_scan_accesses(root / name))

    unjustified = sorted(
        item for item in observed
        if item not in JUSTIFIED_DISPLAY_SCAN_SURVIVORS)
    assert unjustified == [], (
        "undispositioned mutable-display-scan access in the execution path: "
        f"{unjustified}")


def test_the_survivor_list_has_no_stale_entries():
    """W-1.2 case 12 (the other direction).  The survivor table may not carry
    entries that no longer exist -- an unmaintained allow-list is not evidence."""
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread

    root = Path(image_wrangler_thread.__file__).parent
    observed = set()
    for name in ("image_wrangler_thread.py", "nexus_wrangler_thread.py"):
        observed.update(_display_scan_accesses(root / name))

    stale = sorted(set(JUSTIFIED_DISPLAY_SCAN_SURVIVORS) - observed)
    assert stale == [], f"survivor list names accesses that are gone: {stale}"
