# -*- coding: utf-8 -*-
"""O-2 / O-2.1 — acquisition-vs-browse context reproducers and diagnostics pins.

Every case here drives the REAL seams end to end: a real offscreen
``staticWidget``, a real Run click over the real Eiger Standard container, real
frames through the real worker, the real Pause/Resume action seam, a real
browser gesture whose load goes through ``fileHandlerThread.set_datafile``, the
real ``itemSelectionChanged`` frame selection, and the real
``FrameHydrationWorker`` round trip.  R4T-4 is the anti-pattern this file exists
to avoid: no ``SimpleNamespace`` host and no ``lambda`` ever stands in for a seam
under test.  The only test-owned objects are OBSERVERS (recorders that also call
the production implementation) and ONE deterministic latch that suspends a real
hydrator mid-read so a late completion can be delivered after Resume on purpose
rather than by luck.

The structural cases are **strict-xfail**: they state the contract O-3 will
implement, so the suite stays green while the reproducers stay red for exactly
the reason O-3 must fix.

**O-2.1 (review §61) — what the oracle now points at.**  The first cut of these
tests could report success without proving the context-qualified property.  Four
corrections, all here:

* Ownership is asserted against **objects captured at Pause**, never re-resolved
  through ``widget.scan`` or "whichever store is currently selected".  A correct
  pointer split must leave these rows green; a mutation of A must redden them.
  At this parent every pointer is the same object, so the red polarity is
  unchanged — the assertions simply stop being satisfiable by accident.
* The browsed scan is resolved through the **display/viewer selection seam**
  (``displayframe.scan`` / ``H5Viewer.scan``), which is what O-3 swaps.
* "Servable" means ``CapabilityDisposition.RESIDENT`` / ``CapabilityState.
  AVAILABLE``.  A thumbnail, a source fallback, a dropped or errored fact is not
  the payload the panel is supposed to be showing.
* The collision case asserts **both** owners simultaneously — A's captured store
  still serving A's publication AND B's selected store serving B's — so
  suppressing B's publication can no longer pass it.

Fixture roles (real files, via ``$XDART_TEST_DATA``):

* **A** — acquisition, Eiger Standard: ``nexus/bluesky_17_2_00090.nxs`` reduced
  live into a temporary project with its own PONI and detector mask.
* **B** — browse target, Rayonix GI:
  ``xdart_processed_data/Combi4_Angledependence_samz_4p9_03271005.nxs``, a
  processed result with its own geometry, mask and GI identity.

A and B disagree on every carrier that matters (scan key, GI on/off, detector,
PONI, mask shape, integration args) and their frame labels COLLIDE, which is what
makes them the right pair: no assertion here can pass by coincidence.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets

from xdart.gui.tabs.static_scan.frame_projection_adapter import (
    STORE_SCAN_KEY_ATTR,
    FrameProjectionAdapter,
    ProjectionRequest,
    publication_serves_scan,
)
from xdart.gui.tabs.static_scan.run_config_debug import (
    DECISION_CAPABILITY_FORCES_CLEAR,
    DECISION_HYDRATION_CONTEXT_MISMATCH,
    DECISION_PROJECTION_STORE_ABSENT,
    DECISION_PROJECTION_SUPERSEDED,
    DECISION_PUBLICATION_ABSENT,
    DECISION_PUBLICATION_OWNER_MISMATCH,
    DECISION_RECORD_STORE_SKIPPED,
    DISPLAY_CONTEXT_PHASES,
    DISPLAY_CONTEXT_TRANSITION_EVENT,
    FAIL_CLOSED_DECISIONS,
    FAIL_CLOSED_REJECTION_EVENT,
)
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
    scan_name_from_source,
)
from xdart.modules.frame_publication import (
    PublicationStore,
    publication_from_frame_view,
)
from xrd_tools.core import FrameView, IntegrationResult1D
from xrd_tools.session import (
    CapabilityDisposition,
    CapabilityState,
    FrameRecordStore,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Real fixtures
# --------------------------------------------------------------------------- #

_DATA = Path(os.environ.get(
    "XDART_TEST_DATA", Path(__file__).resolve().parents[3] / "test_data"))

#: A — the Eiger Standard acquisition source, its calibration and its mask.
_A_SOURCE = _DATA / "nexus" / "bluesky_17_2_00090.nxs"
_A_PONI = _DATA / "nexus" / "LaB6_0710_1025pm_00005.poni"
_A_MASK = _DATA / "nexus" / "mask.edf"
#: B — the Rayonix GI processed result the operator browses while paused.
_B_RESULT = (_DATA / "xdart_processed_data"
             / "Combi4_Angledependence_samz_4p9_03271005.nxs")

_REQUIRED = (_A_SOURCE, _A_PONI, _A_MASK, _B_RESULT)

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in _REQUIRED),
    reason=(f"O-2 acceptance fixtures absent under {_DATA} "
            "(set XDART_TEST_DATA)"),
)

#: Frames A must have reduced before the run is paused.  Small on purpose: the
#: contract is about ownership at a boundary, not about throughput, and every
#: extra Eiger frame is a 2167x2070 read plus an integration.
_A_FRAMES_BEFORE_PAUSE = 4

#: Wall-clock ceilings.  Generous enough for a cold page cache on the 141 MB
#: container, tight enough that a genuine hang fails the test instead of the
#: session.
_RUN_TIMEOUT_S = 240.0
_BOUNDARY_TIMEOUT_S = 90.0

#: The frame labels A and B both use — the collision the fixtures were chosen
#: for.  B has sixteen frames, so these are always in range for both.
_COLLIDING_LABELS = (1, 2, 3)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# --------------------------------------------------------------------------- #
# Event-loop helpers
# --------------------------------------------------------------------------- #

def _pump(qapp, seconds):
    """Service the real event loop for ``seconds`` (queued signals, timers)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)


def _wait_until(qapp, predicate, timeout, what):
    """Pump until ``predicate`` holds; a timeout is an explicit failure.

    Never returns False: a boundary that does not arrive is a broken seam, and
    silently continuing would turn it into an unrelated assertion failure ten
    lines later.
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        qapp.processEvents()
        time.sleep(0.01)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")


class _RunRecorder:
    """Observer on the real wrangler lifecycle signals.

    Pure observation — it adds connections and never replaces one, so the
    production handlers still run exactly as they do in the app.
    """

    def __init__(self, widget):
        self.frames = []
        self.paused = []
        self.resuming = []
        wrangler = widget.wrangler
        wrangler.sigUpdateData.connect(self._on_frame)
        wrangler.sigPaused.connect(self._on_paused)
        wrangler.sigResuming.connect(self._on_resuming)

    def _on_frame(self, idx):
        self.frames.append(idx)

    def _on_paused(self):
        self.paused.append(time.monotonic())

    def _on_resuming(self):
        self.resuming.append(time.monotonic())


# --------------------------------------------------------------------------- #
# The real Run / Pause / browse / Resume seams
# --------------------------------------------------------------------------- #

def _make_widget(monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    return staticWidget()


def _configure_eiger_standard(widget, out_dir):
    """Configure acquisition A through the production Controls/parameter seam.

    Deliberately the same field writes the Controls panel performs, so the Run
    click that follows is the ordinary admitted-run path — not a hand-built
    ``FrozenRunConfiguration`` slipped past the freeze owner.
    """
    out_dir = str(out_dir)
    widget._controls_v2_param(("Project", "project_folder")).setValue(out_dir)
    widget._controls_v2_param(("Project", "h5_dir")).setValue(out_dir)
    widget._controls_v2_param(("Signal", "inp_type")).setValue("Image Series")
    widget._controls_v2_param(("Signal", "File")).setValue(str(_A_SOURCE))
    widget._controls_v2_param(("Signal", "mask_file")).setValue(str(_A_MASK))
    widget.wrangler.project_folder = out_dir
    widget.wrangler.h5_dir = out_dir
    widget.wrangler.inp_type = "Image Series"
    widget.wrangler.img_file = str(_A_SOURCE)
    widget._set_poni_field(str(_A_PONI))
    widget._refresh_controls_v2_profile_now()


def _start_acquisition(qapp, widget, recorder, frames=_A_FRAMES_BEFORE_PAUSE):
    """Click the REAL Run button and let the REAL worker reduce ``frames``."""
    assert widget.controls.startButton.isEnabled(), (
        "the configured Eiger Standard source did not become runnable")
    widget.controls.startButton.click()
    _wait_until(qapp, lambda: len(recorder.frames) >= frames,
                _RUN_TIMEOUT_S, f"{frames} reduced frames from acquisition A")
    # Let the coalesced publish/render timers drain so the stores hold what the
    # operator would actually be looking at when they hit Pause.
    _pump(qapp, 2.0)
    assert widget._run_active is True


def _pause_acquisition(qapp, widget, recorder):
    """Pause through the REAL action seam (Run morphs to Pause mid-run)."""
    before = len(recorder.paused)
    widget.controls.startButton.click()
    _wait_until(qapp, lambda: len(recorder.paused) > before,
                _BOUNDARY_TIMEOUT_S, "the worker's sigPaused")
    _pump(qapp, 0.5)
    # Pause keeps the run OWNED while lifting the disk-read freeze; that pair is
    # precisely what makes a paused browse legal.
    assert widget._run_active is True
    assert widget.h5viewer._run_writing is False


def _resume_acquisition(qapp, widget, recorder):
    """Resume through the REAL action seam."""
    before = len(recorder.resuming)
    widget.controls.startButton.click()
    _wait_until(qapp, lambda: len(recorder.resuming) > before,
                _BOUNDARY_TIMEOUT_S, "the worker's sigResuming")
    _pump(qapp, 0.5)


def _browse_load(qapp, widget, path):
    """Load ``path`` through the REAL browser gesture.

    ``scans_clicked`` -> ``H5Viewer.set_file`` -> the file thread's queued
    ``set_datafile`` -> ``H5Viewer.thread_finished``: the exact chain the
    contract names, with no shortcut into ``scan.set_datafile`` and no synthetic
    file thread.
    """
    path = Path(path)
    viewer = widget.h5viewer
    viewer.dirname = str(path.parent)
    viewer.update_scans()
    _pump(qapp, 0.3)
    rows = [viewer.ui.listScans.item(r)
            for r in range(viewer.ui.listScans.count())]
    item = next((row for row in rows if row.text() == path.name), None)
    assert item is not None, (
        f"{path.name} is not offered by the real browser listing of "
        f"{path.parent}")
    viewer.scans_clicked(item)
    _wait_until(
        qapp,
        lambda: (viewer.file_thread.fname == str(path)
                 and not viewer.file_thread.running),
        _RUN_TIMEOUT_S, f"the file thread's set_datafile for {path.name}")
    _pump(qapp, 1.5)


def _select_frame_row(qapp, widget, row):
    """Select a hydrated frame through the REAL list-selection seam."""
    listing = widget.h5viewer.ui.listData
    assert listing.count() > row, (
        f"the browsed scan exposed {listing.count()} frames; row {row} "
        "is not selectable")
    listing.setCurrentRow(row)
    _pump(qapp, 2.5)
    return listing.item(row).text()


def _teardown(qapp, widget):
    """Stop the real worker, then close the widget.

    Order matters: a still-running native QThread destroyed with the widget is a
    Qt qFatal, and the hydration worker may be mid-read, so both are stopped
    before the Qt object graph goes away.
    """
    try:
        thread = widget.wrangler.thread
        thread.command = "stop"
        end = time.monotonic() + 60
        while thread.isRunning() and time.monotonic() < end:
            qapp.processEvents()
            time.sleep(0.02)
    except Exception:
        logger.debug("worker stop failed in teardown", exc_info=True)
    try:
        if widget._run_active:
            widget._exit_run_state(widget._new_projection_receipt())
    except Exception:
        logger.debug("run-state exit failed in teardown", exc_info=True)
    try:
        widget.displayframe.stop_hydration_worker()
    except Exception:
        logger.debug("hydration worker stop failed in teardown", exc_info=True)
    try:
        widget._controls_v2_refresh_timer.cancel()
    except Exception:
        logger.debug("refresh timer cancel failed in teardown", exc_info=True)
    # Stop the coalescers BEFORE close(): a throttled flush that fires after the
    # widget has torn its state down re-enters _drain_pending_frames against a
    # widget with no `scan`, which prints a traceback from a thread that is no
    # longer anybody's business.
    for name in ("_update_timer", "_list_timer", "_reint_update_timer"):
        timer = getattr(widget, name, None)
        stop = getattr(timer, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                logger.debug("%s stop failed in teardown", name, exc_info=True)
    _pump(qapp, 0.3)
    try:
        widget.close()
        widget.deleteLater()
    except Exception:
        logger.debug("widget close failed in teardown", exc_info=True)
    for _ in range(3):
        qapp.processEvents()


@pytest.fixture
def acquisition(qapp, monkeypatch, tmp_path):
    """Factory for a real, running, paused Eiger Standard acquisition A."""
    built = []

    def _build(*, frames=_A_FRAMES_BEFORE_PAUSE, pause=True):
        out_dir = tmp_path / f"project_{len(built)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        widget = _make_widget(monkeypatch, tmp_path)
        built.append(widget)
        _configure_eiger_standard(widget, out_dir)
        recorder = _RunRecorder(widget)
        _start_acquisition(qapp, widget, recorder, frames=frames)
        if pause:
            _pause_acquisition(qapp, widget, recorder)
        return widget, recorder

    yield _build

    for widget in built:
        _teardown(qapp, widget)


# --------------------------------------------------------------------------- #
# Ownership capture — EXACT objects, taken once, never re-resolved
# --------------------------------------------------------------------------- #

def _store_labels(store):
    snapshot = getattr(store, "snapshot", None)
    if callable(snapshot):
        return tuple(sorted(snapshot().keys(),
                            key=lambda k: (str(type(k)), str(k))))
    records = getattr(store, "_records", None)
    if isinstance(records, dict):
        return tuple(sorted(records.keys(),
                            key=lambda k: (str(type(k)), str(k))))
    return ()


def _publication_owner(store, label):
    """The owner a publication DECLARES, or ``None`` when the label is unheld."""
    if store is None:
        return None
    publication = store.get(label)
    if publication is None:
        return None
    owner = getattr(publication, "scan_key", None)
    if owner:
        return str(owner)
    source = getattr(publication, "source_identity", None)
    return None if source in (None, "") else str(source)


def _capture_acquisition_owners(widget):
    """Freeze the EXACT objects acquisition A owns, at Pause.

    §61.4 D.  Every later "unchanged" question is asked of THESE references, not
    of ``widget.scan`` or "the currently selected store".  Re-resolving through
    the singleton is what let the first cut's rows stay green under a correct
    pointer split and, worse, inspect the wrong store under a partial one.

    The values captured alongside each object (key, GI flag, argument dicts) are
    what the object held at Pause, so a later comparison detects mutation of the
    captured object itself — not merely substitution of a different one.
    """
    scan = getattr(widget, "_x1_run_scan_capture", None) or widget.scan
    return {
        # -- objects (compared by identity) --------------------------------- #
        "scan": scan,
        "global_mask": getattr(scan, "global_mask", None),
        "cached_data_mask": getattr(scan, "_cached_data_mask", None),
        "poni": getattr(scan, "_cached_poni", None),
        "record_store": widget._active_frame_record_store(),
        "publication_store": widget.publication_store,
        "integrator_scan": getattr(
            getattr(widget, "integratorTree", None), "scan", None),
        "stitch_scan": getattr(
            getattr(widget, "stitch_thread", None), "scan", None),
        "wrangler_poni": getattr(widget.wrangler, "poni", None),
        # -- values the captured objects held at Pause ----------------------- #
        "scan_key": str(getattr(scan, "name", "")),
        "data_file": str(getattr(scan, "data_file", "") or ""),
        "gi": bool(getattr(scan, "gi", False)),
        "bai_1d_args": dict(getattr(scan, "bai_1d_args", {}) or {}),
        "bai_2d_args": dict(getattr(scan, "bai_2d_args", {}) or {}),
        "record_store_owner": getattr(
            widget._active_frame_record_store(), STORE_SCAN_KEY_ATTR, None),
        "record_store_labels": _store_labels(
            widget._active_frame_record_store()),
        "publication_generation": widget.publication_store.generation,
        "publication_labels": _store_labels(widget.publication_store),
        "publication_owners": {
            label: _publication_owner(widget.publication_store, label)
            for label in _COLLIDING_LABELS
        },
    }


def _selected_display_owners(widget):
    """Resolve the CURRENTLY SELECTED display context through its own seam.

    §61.4 D.  At this parent ``displayframe.scan``, ``H5Viewer.scan`` and
    ``widget.scan`` are one object, so this changes no polarity; once O-3 swaps
    the display selection, these are the pointers that must move — and the
    acquisition capture above is what must not.
    """
    display = widget.displayframe
    viewer = widget.h5viewer
    return {
        "display_scan": getattr(display, "scan", None),
        "viewer_scan": getattr(viewer, "scan", None),
        "publication_store": getattr(display, "publication_store", None),
        "record_store": widget._active_frame_record_store(),
        "projection": getattr(display, "_current_frame_projection", None),
        "projection_scan_key": str(
            getattr(display, "_current_frame_projection_scan_key", "") or ""),
        "title": display.ui.labelCurrent.text(),
    }


def _capability(projection, name):
    return getattr(getattr(projection, "capabilities", None), name, None)


def _capability_is_rendered_payload(projection, name):
    """§61.4 D: only a RESIDENT / AVAILABLE fact is the payload itself.

    ``HYDRATABLE`` plus a stored thumbnail, a ``SOURCE_FALLBACK`` that could be
    re-read, a ``DROPPED`` or ``ERROR`` fact — none of those is the tier the
    panel is supposed to be displaying, and treating "anything but ABSENT" as
    servable is what let the first cut accept a thumbnail for the raw row.
    """
    fact = _capability(projection, name)
    return (getattr(fact, "disposition", None) is CapabilityDisposition.RESIDENT
            and getattr(fact, "state", None) is CapabilityState.AVAILABLE)


def _panel_has_image(pair):
    return bool(pair is not None and pair[0] is not None
                and getattr(pair[0], "size", 0))


def _panel_has_trace(plot_data):
    if plot_data is None or len(plot_data) < 2:
        return False
    y = plot_data[1]
    return bool(getattr(y, "size", 0) and np.isfinite(y).any())


# --------------------------------------------------------------------------- #
# Structured-log decoding
# --------------------------------------------------------------------------- #

def _debug_events(caplog):
    """Every structured run-config record in the captured log, decoded."""
    events = []
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith("RUN_CONFIG_DEBUG "):
            continue
        try:
            events.append(json.loads(message.removeprefix("RUN_CONFIG_DEBUG ")))
        except ValueError:
            continue
    return events


def _transition_events(caplog):
    return [event for event in _debug_events(caplog)
            if event.get("event") == DISPLAY_CONTEXT_TRANSITION_EVENT]


def _rejection_events(caplog):
    return [event for event in _debug_events(caplog)
            if event.get("event") == FAIL_CLOSED_REJECTION_EVENT]


def _user_browse_pair(caplog, requested_path):
    """The start/finish records for the OPERATOR's browse of ``requested_path``.

    Internal run-output reloads are real transitions and are traced, but they are
    a different ``kind`` and must never satisfy the user-browse acceptance
    assertion (§61.4 B).
    """
    wanted = str(requested_path)
    starts, finishes = [], []
    for event in _transition_events(caplog):
        operation = event.get("operation") or {}
        if operation.get("kind") != "user_browse":
            continue
        if operation.get("requested_path") != wanted:
            continue
        if event["phase"] == "browse_load_start":
            starts.append(event)
        elif event["phase"] == "browse_load_finish":
            finishes.append(event)
    return starts, finishes


# --------------------------------------------------------------------------- #
# 1. A paused browse must render the BROWSED scan (R4T-5's gap)
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason=(
        "O-2 reproducer (O-3 flips it): a paused browse of the real Rayonix GI "
        "result cannot serve its own raw, cake or integrated tiers as RESIDENT "
        "payload — the browse never gets a context of its own"
    ),
)
def test_paused_browse_renders_browsed_scan(qapp, acquisition):
    """Acceptance path 1 — B must be FULLY servable while A is paused.

    Servability is asserted at BOTH tiers on purpose: the rendered panels (what
    the operator sees) and the typed projection capabilities (what the display
    layer was actually allowed to consult).  A browse that paints something
    while its projection reports ABSENT is not "working" — it is showing
    whatever the singleton happened to be carrying.

    Geometry and mask are read from the SELECTED display scan, not from
    ``widget.scan``: after O-3 those differ, and only the selected one is the
    browse's own.
    """
    widget, _recorder = acquisition()
    b_key = _B_RESULT.stem

    _browse_load(qapp, widget, _B_RESULT)
    label = _select_frame_row(qapp, widget, 2)
    assert label, "the real selection seam produced no current frame"

    display = widget.displayframe
    selected = _selected_display_owners(widget)
    projection = selected["projection"]
    browse_scan = selected["display_scan"]
    poni = getattr(browse_scan, "_cached_poni", None)

    observed = {
        "projection_present": bool(getattr(projection, "present", False)),
        "projection_scan_key": selected["projection_scan_key"],
        "raw_image_rendered": _panel_has_image(display.image_data),
        "cake_rendered": _panel_has_image(display.binned_data),
        "one_d_trace_rendered": _panel_has_trace(display.plot_data),
        "raw_is_resident_payload": _capability_is_rendered_payload(
            projection, "raw"),
        "cake_is_resident_payload": _capability_is_rendered_payload(
            projection, "integrated_2d"),
        "one_d_is_resident_payload": _capability_is_rendered_payload(
            projection, "integrated_1d"),
        "title_names_browsed_scan": b_key in selected["title"],
        "selected_scan_key_is_browsed": str(
            getattr(browse_scan, "name", "")) == b_key,
        "geometry_is_browsed_detector": str(
            getattr(poni, "detector", "")).startswith("Rayonix"),
        "mask_is_browsed_scan": getattr(
            browse_scan, "global_mask", None) is not None,
    }
    expected = {
        "projection_present": True,
        "projection_scan_key": b_key,
        "raw_image_rendered": True,
        "cake_rendered": True,
        "one_d_trace_rendered": True,
        "raw_is_resident_payload": True,
        "cake_is_resident_payload": True,
        "one_d_is_resident_payload": True,
        "title_names_browsed_scan": True,
        "selected_scan_key_is_browsed": True,
        "geometry_is_browsed_detector": True,
        "mask_is_browsed_scan": True,
    }
    # Captured at this parent: cake_rendered False, and all three
    # *_is_resident_payload rows False — the projection falls through to a
    # publication view that owns nothing of B's, so both integrated tiers report
    # ABSENT while the raw row is only a SOURCE_FALLBACK thumbnail.
    assert observed == expected


# --------------------------------------------------------------------------- #
# 2. Browsing B must leave acquisition A untouched (R4T-6, first half)
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason=(
        "O-2 reproducer (O-3 flips it): the browse load repoints the CAPTURED "
        "acquisition scan in place — key, GI, mask, PONI and both BAI argument "
        "sets move on the object A still owns, and B's publications overwrite "
        "A's on the colliding labels"
    ),
)
def test_paused_browse_leaves_acquisition_untouched(qapp, acquisition):
    """Acceptance path 2 — nothing A owns may move while B is browsed.

    Every row interrogates the object captured at Pause.  ``scan_object`` is not
    asked "is ``widget.scan`` still the same?" — it is asked "does the object A
    captured still hold A's key, A's GI flag, A's calibration?".  That is the
    question a pointer split answers green and a singleton mutation answers red,
    and it cannot be satisfied by re-resolving whatever pointer is current.
    """
    widget, _recorder = acquisition()
    captured = _capture_acquisition_owners(widget)
    a_scan = captured["scan"]
    a_store = captured["publication_store"]
    held = tuple(label for label in _COLLIDING_LABELS
                 if captured["publication_owners"].get(label))
    assert held, (
        "acquisition A published none of the colliding low labels; the "
        "collision this test needs did not occur")

    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 2)

    record_store = captured["record_store"]
    observed = {
        "captured_scan_keeps_its_key":
            str(getattr(a_scan, "name", "")) == captured["scan_key"],
        "captured_scan_keeps_its_file":
            str(getattr(a_scan, "data_file", "") or "")
            == captured["data_file"],
        "captured_scan_keeps_its_gi":
            bool(getattr(a_scan, "gi", False)) == captured["gi"],
        "captured_scan_keeps_its_mask":
            getattr(a_scan, "global_mask", None) is captured["global_mask"],
        "captured_scan_keeps_its_data_mask":
            getattr(a_scan, "_cached_data_mask", None)
            is captured["cached_data_mask"],
        "captured_scan_keeps_its_poni":
            getattr(a_scan, "_cached_poni", None) is captured["poni"],
        "captured_scan_keeps_its_1d_args":
            dict(getattr(a_scan, "bai_1d_args", {}) or {})
            == captured["bai_1d_args"],
        "captured_scan_keeps_its_2d_args":
            dict(getattr(a_scan, "bai_2d_args", {}) or {})
            == captured["bai_2d_args"],
        "captured_record_store_keeps_its_owner":
            getattr(record_store, STORE_SCAN_KEY_ATTR, None)
            == captured["record_store_owner"],
        "captured_record_store_keeps_its_frames":
            set(captured["record_store_labels"])
            <= set(_store_labels(record_store)),
        "captured_publication_store_keeps_its_generation":
            a_store.generation == captured["publication_generation"],
        "captured_publications_survive": all(
            _publication_owner(a_store, label)
            == captured["publication_owners"][label] for label in held),
        "integrator_stays_bound_to_acquisition":
            captured["integrator_scan"] is a_scan,
        "stitch_stays_bound_to_acquisition":
            captured["stitch_scan"] is a_scan,
    }
    expected = dict.fromkeys(observed, True)
    # Captured at this parent: the key, file, GI flag, mask, PONI and both
    # argument sets all move on A's own object, and the colliding publications
    # are overwritten.  The integrator/stitch rows are green here and must STAY
    # green — O-3 swaps only the display selection, never their binding.
    assert observed == expected


# --------------------------------------------------------------------------- #
# 3. Resume must be a SELECTION, not a restoration
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason=(
        "O-2 reproducer (O-3 flips it): Resume leaves B's GI flag and PONI on "
        "the captured acquisition scan, the first resumed frame's rescope "
        "clears the publication store, and the resumed payload is not owner-"
        "qualified to A"
    ),
)
def test_resume_restores_acquisition_coherently(qapp, acquisition):
    """Acceptance path 3 — the next A frame renders as A, and nothing is wiped.

    §61.4 D: an A title over stale B panels is not a pass.  The rows below ask
    whether the display SELECTION points back at the captured acquisition object
    by identity, whether the acquisition's own carriers were ever touched, and
    whether the resumed frame's projection is qualified to A — not merely
    whether some label reads "A" again after a restoration procedure ran.
    """
    widget, recorder = acquisition()
    captured = _capture_acquisition_owners(widget)
    a_scan = captured["scan"]
    a_key = captured["scan_key"]

    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 2)

    frames_at_resume = len(recorder.frames)
    _resume_acquisition(qapp, widget, recorder)
    _wait_until(qapp, lambda: len(recorder.frames) > frames_at_resume + 1,
                _RUN_TIMEOUT_S, "the next reduced frame from acquisition A")
    _pump(qapp, 2.0)

    selected = _selected_display_owners(widget)
    projection = selected["projection"]
    observed = {
        "run_still_owned": widget._run_active is True,
        "selection_points_at_captured_acquisition":
            selected["display_scan"] is a_scan,
        "captured_scan_keeps_its_key":
            str(getattr(a_scan, "name", "")) == a_key,
        "captured_scan_keeps_its_gi":
            bool(getattr(a_scan, "gi", False)) == captured["gi"],
        "captured_scan_keeps_its_poni":
            getattr(a_scan, "_cached_poni", None) is captured["poni"],
        "captured_scan_keeps_its_mask":
            getattr(a_scan, "global_mask", None) is captured["global_mask"],
        "captured_scan_keeps_its_1d_args":
            dict(getattr(a_scan, "bai_1d_args", {}) or {})
            == captured["bai_1d_args"],
        "captured_scan_keeps_its_2d_args":
            dict(getattr(a_scan, "bai_2d_args", {}) or {})
            == captured["bai_2d_args"],
        "integrator_never_left_the_acquisition":
            captured["integrator_scan"] is a_scan,
        "title_names_acquisition": a_key in selected["title"],
        "resumed_projection_is_owner_qualified":
            selected["projection_scan_key"] == a_key
            and bool(getattr(projection, "present", False)),
        "captured_publication_store_not_recycled":
            selected["publication_store"] is captured["publication_store"],
        "captured_publication_store_not_cleared":
            captured["publication_store"].generation
            == captured["publication_generation"],
    }
    expected = dict.fromkeys(observed, True)
    # Captured at this parent: the GI flag, PONI and both argument sets are
    # still B's on A's own object, and the publication store's generation has
    # been bumped by the first resumed frame's rescope.  The scan KEY and title
    # do come back — but only because that same rescope renamed the singleton,
    # which is the restoration this row set exists to reject.
    assert observed == expected


# --------------------------------------------------------------------------- #
# 4. A delayed browse hydration must be rejected BY CONTEXT after Resume
# --------------------------------------------------------------------------- #

class _HydrationLatch:
    """Suspend ONE real hydration inside the production hydrator.

    This is not a stand-in for the seam under test: the registered production
    hydrator still performs the read, and the real
    ``FrameHydrationWorker`` -> ``sigHydrated`` -> ``_on_frame_hydrated`` round
    trip is untouched.  All the latch does is make "the completion lands after
    Resume" a fact rather than a race.

    The wrapper is stored ONCE (``self._wrapper``) so ``restore`` can compare the
    exact registered callable; the first cut compared two separately-created
    bound methods with ``is``, which can never match.
    """

    def __init__(self, store):
        self._store = store
        self._inner = getattr(store, "_hydrator", None)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.labels = []
        assert callable(self._inner), (
            "the browse load registered no publication hydrator; there is no "
            "real completion to hold")
        self._wrapper = self._hold
        store.set_hydrator(self._wrapper)

    def _hold(self, label):
        self.labels.append(label)
        self.entered.set()
        # Bounded: a latch that never releases must fail the test, not wedge
        # the session's worker thread forever.
        self.release.wait(_BOUNDARY_TIMEOUT_S)
        return self._inner(label)

    def restore(self):
        self.release.set()
        if getattr(self._store, "_hydrator", None) is self._wrapper:
            self._store.set_hydrator(self._inner)


def _record_hydration_requests(monkeypatch, sink):
    """Spy the REAL worker request seam without replacing it."""
    from xdart.gui.tabs.static_scan.frame_hydration_worker import (
        FrameHydrationWorker,
    )
    original = FrameHydrationWorker.request

    def recording_request(self, label, generation, **kwargs):
        sink.append((label, generation, dict(kwargs)))
        return original(self, label, generation, **kwargs)

    monkeypatch.setattr(FrameHydrationWorker, "request", recording_request)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "O-2 reproducer (O-3 flips it): a hydration round trip carries a "
        "display generation and no owner, so the browse completion held past "
        "Resume produces no hydration_context_mismatch and is not rejected by "
        "context identity"
    ),
)
def test_delayed_browse_hydration_rejected_after_resume(
        qapp, acquisition, monkeypatch, caplog):
    """Acceptance path 4 — the HELD B completion must be refused by identity.

    §61.4 E: the held request is captured the moment the latch is entered, so a
    later A request cannot stand in for it; the rejection log offset is taken
    before release, so only NEW records count; and every consequence row is an
    independent all-true fact rather than an ``or`` that a changed PONI could
    satisfy.
    """
    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    # A narrow heavy-staging window is what makes an evicted browse frame — and
    # therefore a REAL background hydration — reachable at all.  It is the
    # production ``XDART_HEAVY_WINDOW`` knob, read by the store at construction,
    # so it must be set before the widget is built.  With the default window
    # every freshly browsed frame stays resident and the production request
    # short-circuits on the resident-tier guard, which would leave nothing for
    # the latch to hold.
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    caplog.set_level(logging.INFO)
    requests = []
    _record_hydration_requests(monkeypatch, requests)

    widget, recorder = acquisition()
    display = widget.displayframe
    captured = _capture_acquisition_owners(widget)
    a_scan = captured["scan"]
    a_key = captured["scan_key"]

    _browse_load(qapp, widget, _B_RESULT)
    selected_label = _select_frame_row(qapp, widget, 2)
    browse_owners = _selected_display_owners(widget)
    browse_key = str(getattr(browse_owners["display_scan"], "name", ""))
    browse_store = browse_owners["publication_store"]

    # The production enable-once seam the live app calls; headless tests leave
    # hydration synchronous, so this is what puts the REAL worker in the path.
    display.enable_async_hydration()
    latch = _HydrationLatch(widget.publication_store)
    held_request = None
    try:
        # Ask for a B frame OTHER than the selected one whose heavy tier the
        # store has actually evicted — the production guards refuse to request a
        # resident tier, and rightly so.
        listing = widget.h5viewer.ui.listData
        target_label = next(
            (int(listing.item(row).text())
             for row in range(listing.count())
             if listing.item(row).text() != selected_label
             and not display._hydration_purpose_resident(
                 int(listing.item(row).text()), "full")),
            None)
        assert target_label is not None, (
            "no browsed frame had an evicted heavy tier, so no real hydration "
            "could be put in flight")
        requests_before = len(requests)
        display._request_frame_hydration(target_label, purpose="full")
        _wait_until(qapp, latch.entered.is_set, _BOUNDARY_TIMEOUT_S,
                    "the real hydrator to enter the latch")
        # Capture the HELD request NOW: after Resume the worker may enqueue A
        # requests, and `requests[-1]` would then describe one of those instead.
        assert len(requests) > requests_before, (
            "the latch was entered without the production request seam "
            "recording anything")
        held_request = requests[requests_before]
        assert held_request[0] == target_label

        hydrated = []
        display._hydration_worker.sigHydrated.connect(
            lambda label, generation: hydrated.append((label, generation)))

        frames_at_resume = len(recorder.frames)
        _resume_acquisition(qapp, widget, recorder)
        _wait_until(qapp, lambda: len(recorder.frames) > frames_at_resume,
                    _RUN_TIMEOUT_S, "acquisition A to produce a frame again")

        # Everything the stale completion must not be able to change, sampled
        # after Resume and before release.
        before_release = {
            "poni": getattr(a_scan, "_cached_poni", None),
            "mask": getattr(a_scan, "global_mask", None),
            "key": str(getattr(a_scan, "name", "")),
            "display_scan": widget.displayframe.scan,
            "a_labels": _store_labels(captured["publication_store"]),
        }
        rejections_before = len(_rejection_events(caplog))

        # Now let the stale browse completion land.
        latch.release.set()
        _wait_until(qapp, lambda: bool(hydrated), _BOUNDARY_TIMEOUT_S,
                    "the held hydration completion")
        _pump(qapp, 1.5)
    finally:
        latch.restore()

    new_rejections = _rejection_events(caplog)[rejections_before:]
    held_label, held_generation, held_kwargs = held_request
    owner_keys = {"owner", "owner_token", "request_owner", "context_id"}
    correlated = [
        event for event in new_rejections
        if event.get("decision") == DECISION_HYDRATION_CONTEXT_MISMATCH
        and event.get("label") == held_label
        and event.get("generation") == held_generation
        and event.get("expected_owner") == a_key
        and browse_key in str(event.get("found_owner", ""))
    ]
    observed = {
        "request_carries_owner": bool(
            owner_keys & set(held_kwargs)
            or {"context_id", "scan_key"} <= set(held_kwargs)),
        "exactly_one_correlated_rejection": len(correlated) == 1,
        "acquisition_poni_untouched":
            getattr(a_scan, "_cached_poni", None) is before_release["poni"],
        "acquisition_mask_untouched":
            getattr(a_scan, "global_mask", None) is before_release["mask"],
        "acquisition_key_untouched":
            str(getattr(a_scan, "name", "")) == before_release["key"],
        "display_selection_untouched":
            widget.displayframe.scan is before_release["display_scan"],
        "acquisition_store_unchanged":
            _store_labels(captured["publication_store"])
            == before_release["a_labels"],
        "browse_store_kept_its_own_payload":
            _publication_owner(browse_store, held_label) is not None,
    }
    expected = dict.fromkeys(observed, True)
    # Captured at this parent: request_carries_owner is False (the request seam
    # takes label/generation/purpose/consumer and nothing that identifies the
    # context) and exactly_one_correlated_rejection is False (nothing emits
    # hydration_context_mismatch at all, so the held completion is never
    # rejected by context).
    assert observed == expected


# --------------------------------------------------------------------------- #
# 5. Same-label collisions across contexts
# --------------------------------------------------------------------------- #

def test_same_label_publications_never_cross_serve(qapp, acquisition):
    """GREEN PIN — the ownership predicate itself is already correct.

    Both real scans publish a frame ``1``.  ``publication_serves_scan`` is the
    production gate every scan-qualified lookup runs through, and it must refuse
    each scan's publication for the other's key.  O-3 must not "fix" the
    collision by loosening this rule; the store, not the predicate, is what
    needs an owner.
    """
    widget, _recorder = acquisition()
    a_key = widget.scan.name
    a_publication = widget.publication_store.get(1)
    assert a_publication is not None, "acquisition A published no frame 1"

    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 0)
    selected = _selected_display_owners(widget)
    b_key = str(getattr(selected["display_scan"], "name", ""))
    b_publication = selected["publication_store"].get(1)
    assert b_publication is not None, "the browsed scan published no frame 1"
    assert a_key != b_key

    assert publication_serves_scan(a_publication, a_key) is True
    assert publication_serves_scan(b_publication, b_key) is True
    assert publication_serves_scan(a_publication, b_key) is False
    assert publication_serves_scan(b_publication, a_key) is False


@pytest.mark.xfail(
    strict=True,
    reason=(
        "O-2 reproducer (O-3 flips it): A and B share frame labels and share "
        "ONE publication store, so A's captured store no longer serves A's "
        "publication for a colliding label and the two contexts have no "
        "distinct stores at all"
    ),
)
def test_same_label_collision_preserves_both_owners(qapp, acquisition):
    """Acceptance path 5a — BOTH owners must hold the shared label at once.

    §61.4 D: asserting only that A's labels survive in one dynamically selected
    store could be satisfied by suppressing B's publication entirely.  The four
    rows below are simultaneous: A's captured store serves A's frame, B's
    selected store serves B's frame, the two stores are distinct owners, and
    neither publication will serve the other's key.
    """
    widget, _recorder = acquisition()
    captured = _capture_acquisition_owners(widget)
    a_store = captured["publication_store"]
    a_key = captured["scan_key"]
    label = next(
        (one for one in _COLLIDING_LABELS
         if captured["publication_owners"].get(one)), None)
    assert label is not None, (
        "acquisition A retained none of the colliding low labels")

    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 0)

    selected = _selected_display_owners(widget)
    b_store = selected["publication_store"]
    b_key = str(getattr(selected["display_scan"], "name", ""))
    a_publication = a_store.get(label)
    b_publication = b_store.get(label)

    observed = {
        "captured_store_serves_acquisition":
            a_publication is not None
            and publication_serves_scan(a_publication, a_key),
        "selected_store_serves_browse":
            b_publication is not None
            and publication_serves_scan(b_publication, b_key),
        "the_two_stores_are_distinct": a_store is not b_store,
        "acquisition_publication_refuses_browse_key":
            a_publication is not None
            and not publication_serves_scan(a_publication, b_key),
        "browse_publication_refuses_acquisition_key":
            b_publication is not None
            and not publication_serves_scan(b_publication, a_key),
    }
    expected = dict.fromkeys(observed, True)
    # Captured at this parent: there is only ONE store, so
    # ``the_two_stores_are_distinct`` is False and the single entry under the
    # shared label is B's — which makes ``captured_store_serves_acquisition``
    # and ``acquisition_publication_refuses_browse_key`` False too.
    assert observed == expected


# --------------------------------------------------------------------------- #
# 6. Dotted scan stems — the two-parser divergence
# --------------------------------------------------------------------------- #

def _dotted_stem_link(tmp_path, source):
    """A REAL dotted-stem container: the real B result under a dotted name.

    A link, not a copy: the bytes under test are the shipped fixture's, so the
    load is a genuine load and the only thing that changed is the name the two
    parsers disagree about.
    """
    target = tmp_path / "Combi4.v2_03271005.nxs"
    if not target.exists():
        try:
            os.link(source, target)
        except OSError:
            os.symlink(source, target)
    return target


def test_canonical_parser_keeps_the_full_dotted_stem(tmp_path):
    """GREEN PIN — the canonical parser is the one that is already right.

    ``scan_name_from_source`` keeps a container's FULL stem, because the numeric
    suffix is part of the scan identity (the F2 rule).  A dotted stem is the
    same case: everything before the container suffix belongs to the name.
    """
    dotted = _dotted_stem_link(tmp_path, _B_RESULT)
    assert scan_name_from_source(str(dotted)) == "Combi4.v2_03271005"
    assert scan_name_from_source(str(_B_RESULT)) == _B_RESULT.stem


@pytest.mark.xfail(
    strict=True,
    reason=(
        "O-2 reproducer (O-3 flips it): the browse path names the selected scan "
        "with set_datafile's first-dot truncation instead of the canonical "
        "scan_name_from_source, so a dotted stem loses everything after the "
        "first dot"
    ),
)
def test_browse_load_names_a_dotted_stem_canonically(
        qapp, monkeypatch, tmp_path):
    """Acceptance path 5b — ONE parser must name the browsed scan.

    Driven through the real browser gesture and the real file thread, and read
    back off the SELECTED display/viewer scan rather than ``widget.scan``: the
    divergence only matters because the browse path picks the wrong parser for
    the object the display is about to show.
    """
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        dotted = _dotted_stem_link(tmp_path, _B_RESULT)
        canonical = scan_name_from_source(str(dotted))
        _browse_load(qapp, widget, dotted)

        selected = _selected_display_owners(widget)
        observed = {
            "display_scan_key": str(
                getattr(selected["display_scan"], "name", "")),
            "viewer_scan_key": str(
                getattr(selected["viewer_scan"], "name", "")),
            "data_file": os.path.basename(
                getattr(selected["display_scan"], "data_file", "") or ""),
        }
        # Captured at this parent: both keys are "Combi4" — set_datafile splits
        # on the FIRST dot, so "v2_03271005" is silently discarded while the
        # data file (and every title derived from the canonical parser) keeps it.
        assert observed == {
            "display_scan_key": canonical,
            "viewer_scan_key": canonical,
            "data_file": dotted.name,
        }
    finally:
        _teardown(qapp, widget)


# --------------------------------------------------------------------------- #
# 7. The forbidden-design discriminator (§61.4 F)
# --------------------------------------------------------------------------- #

_O3_CONTEXT_MODULE = "xdart.modules.display_context"
_O3_CONTEXT_TYPES = ("AcquisitionContext", "BrowseContext", "DisplaySelection")


def _o3_context_module_facts():
    """Whether O-3's Qt-free context owners exist, without importing Qt."""
    try:
        module = importlib.import_module(_O3_CONTEXT_MODULE)
    except Exception:
        return {"module_present": False, "types_present": False,
                "qt_free": False}
    types_present = all(
        isinstance(getattr(module, name, None), type)
        for name in _O3_CONTEXT_TYPES)
    qt_free = True
    try:
        source = inspect.getsource(module)
        qt_free = not any(
            token in source
            for token in ("pyqtgraph", "PySide6", "QtCore", "QtWidgets"))
    except Exception:
        qt_free = False
    return {"module_present": True, "types_present": types_present,
            "qt_free": qt_free}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "O-2.1 forbidden-design pin (O-3 supplies the shape and removes this "
        "marker): there is no xdart.modules.display_context, the "
        "_x1_run_scan_capture alias still exists, and the browse targets the "
        "acquisition scan itself"
    ),
)
def test_o3_context_owner_shape_is_absent(qapp, acquisition):
    """The O-3 shape, pinned as a discriminator BEFORE it exists.

    §61.4 F.  This is not authorization to add the types during O-2.1 — it is
    the assertion that makes the forbidden designs fail loudly.  A deep-copied
    ``LiveScan`` passes "the browse target is not the acquisition object" but
    fails the module rows.  A snapshot/restore Resume passes "the key came
    back" but fails ``acquisition_never_mutated``, which is sampled DURING the
    browse — the window a restoration procedure cannot hide.
    """
    widget, recorder = acquisition()
    captured = _capture_acquisition_owners(widget)
    a_scan = captured["scan"]

    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 2)

    # Sampled while B is displayed: this is the window a restoration-style
    # Resume would mutate and then paper over.
    during_browse = {
        "key": str(getattr(a_scan, "name", "")),
        "gi": bool(getattr(a_scan, "gi", False)),
        "poni": getattr(a_scan, "_cached_poni", None),
        "browse_target": getattr(widget.h5viewer.file_thread, "scan", None),
        "display_scan": widget.displayframe.scan,
        "integrator_scan": getattr(widget.integratorTree, "scan", None),
    }

    frames_at_resume = len(recorder.frames)
    _resume_acquisition(qapp, widget, recorder)
    _wait_until(qapp, lambda: len(recorder.frames) > frames_at_resume + 1,
                _RUN_TIMEOUT_S, "the next reduced frame from acquisition A")
    _pump(qapp, 2.0)

    module_facts = _o3_context_module_facts()
    observed = {
        "context_module_present": module_facts["module_present"],
        "context_types_present": module_facts["types_present"],
        "context_owners_are_qt_free": module_facts["qt_free"],
        "run_scan_alias_deleted": not hasattr(widget, "_x1_run_scan_capture"),
        "browse_target_is_not_the_acquisition_scan":
            during_browse["browse_target"] is not a_scan,
        "acquisition_never_mutated": (
            during_browse["key"] == captured["scan_key"]
            and during_browse["gi"] == captured["gi"]
            and during_browse["poni"] is captured["poni"]),
        "only_display_selection_swapped": (
            during_browse["display_scan"] is not a_scan
            and during_browse["integrator_scan"] is a_scan),
        "resume_is_a_pointer_selection": (
            widget.displayframe.scan is a_scan
            and getattr(a_scan, "_cached_poni", None) is captured["poni"]),
    }
    expected = dict.fromkeys(observed, True)
    # Captured at this parent: every module row is False (the module does not
    # exist), the alias is present, the browse target IS the acquisition scan,
    # and the acquisition is mutated in place during the browse.
    assert observed == expected


# --------------------------------------------------------------------------- #
# Diagnostics discriminators (§61.4 A) — green once O-2.1's instrumentation
# lands, red at its parent.
# --------------------------------------------------------------------------- #

def test_user_browse_transition_pair_is_request_qualified(
        qapp, acquisition, monkeypatch, caplog):
    """§61.4 A.1 — start and finish must be ONE identified operation.

    The pair selected here is the OPERATOR's browse of B; the run's own internal
    output reloads are traced under a different ``kind`` and must not be able to
    satisfy this.  Every correlation field is required to be non-null: a record
    whose generations are all ``None`` cannot qualify anything, which is exactly
    what the first cut emitted from the file thread.
    """
    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    caplog.set_level(logging.INFO)

    widget, _recorder = acquisition()
    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 2)

    starts, finishes = _user_browse_pair(caplog, _B_RESULT)
    assert len(starts) == 1, f"expected one user browse start, got {len(starts)}"
    assert len(finishes) == 1, (
        f"expected one user browse finish, got {len(finishes)}")
    start_op = starts[0]["operation"]
    finish_op = finishes[0]["operation"]

    assert start_op["token"], "the browse start minted no operation token"
    assert start_op == finish_op, (
        "the finish did not echo the exact start operation identity")
    assert start_op["requested_path"] == str(_B_RESULT)
    assert start_op["load_generation"] is not None

    identity = start_op["identity"]
    assert identity is not None, (
        "the browse pair carries no accepted run/config identity")
    assert identity["run_generation"] is not None
    assert identity["runend_generation"] is not None
    assert identity["display_generation"] is not None
    assert identity["run_active"] is True
    assert identity["acquisition_key"], (
        "the browse pair names no acquisition owner")
    assert identity["acquisition_object_id"]

    # The internal run-output reload is classified separately and is NOT this
    # pair — the acceptance assertion must not be satisfiable by it.
    kinds = {
        (event.get("operation") or {}).get("kind")
        for event in _transition_events(caplog)
        if event["phase"].startswith("browse_load")
    }
    assert "user_browse" in kinds
    assert "internal_output_reload" in kinds


def test_projection_adapter_records_each_fail_closed_decision(
        monkeypatch, caplog):
    """§61.4 A.2 — one discriminator per store-side fail-closed decision.

    Drives the REAL ``FrameProjectionAdapter`` over REAL stores and REAL
    publications, and requires the exact decision, expected/found owner,
    outcome, blanking flag and label for each.  Deleting any one emission site
    reddens this test; so does collapsing "no publication" back into "wrong
    owner", which is the untruthful taxonomy §61.3 rejected.
    """
    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    caplog.set_level(logging.INFO)

    def _view(label, source):
        return FrameView.from_results(
            label=label,
            result_1d=IntegrationResult1D(
                radial=np.linspace(0.5, 3.5, 4),
                intensity=np.array([2.0, 4.0, 8.0, 16.0]),
                sigma=np.ones(4), unit="q_A^-1"),
            metadata_raw={"i0": 1.0},
            source_path=source, source_frame_index=label)

    publications = PublicationStore()
    publications.upsert(publication_from_frame_view(
        _view(1, "/data/owner_a.nxs"), scan_key="owner_a"))
    record_store = FrameRecordStore()
    setattr(record_store, STORE_SCAN_KEY_ATTR, "owner_a")

    def decisions(events):
        return {event["decision"]: event for event in events}

    # (a) a publication exists, under the WRONG owner -> panel blanks.
    caplog.clear()
    adapter = FrameProjectionAdapter(lambda: None, lambda: publications)
    adapter.project(ProjectionRequest(
        scan_key="owner_b", frame_index=1, generation=1))
    found = decisions(_rejection_events(caplog))
    mismatch = found.get(DECISION_PUBLICATION_OWNER_MISMATCH)
    assert mismatch is not None, sorted(found)
    assert mismatch["expected_owner"] == "owner_b"
    assert "owner_a" in mismatch["found_owner"]
    assert mismatch["outcome"] == "record_withheld"
    assert mismatch["blanks_panel"] is True
    assert mismatch["label"] == 1

    # (b) NO publication exists for the label -> a different decision.
    caplog.clear()
    adapter = FrameProjectionAdapter(lambda: None, lambda: publications)
    adapter.project(ProjectionRequest(
        scan_key="owner_a", frame_index=99, generation=1))
    found = decisions(_rejection_events(caplog))
    absent = found.get(DECISION_PUBLICATION_ABSENT)
    assert absent is not None, sorted(found)
    assert absent["expected_owner"] == "owner_a"
    assert absent["found_owner"] == ""
    assert absent["outcome"] == "record_absent"
    assert absent["blanks_panel"] is True
    assert DECISION_PUBLICATION_OWNER_MISMATCH not in found

    # (c) the record store is SKIPPED but the publication fallback serves — a
    #     rejection that blanks nothing.
    caplog.clear()
    adapter = FrameProjectionAdapter(
        lambda: record_store, lambda: publications)
    adapter.project(ProjectionRequest(
        scan_key="owner_b", frame_index=1, generation=1))
    found = decisions(_rejection_events(caplog))
    skipped = found.get(DECISION_RECORD_STORE_SKIPPED)
    assert skipped is not None, sorted(found)
    assert skipped["expected_owner"] == "owner_b"
    assert skipped["found_owner"] == "owner_a"
    assert skipped["outcome"] == "fallthrough_to_publication_store"
    assert skipped["blanks_panel"] is False

    # (d) every qualified store failed -> the panel really is blanked.
    caplog.clear()
    adapter = FrameProjectionAdapter(lambda: record_store, lambda: None)
    adapter.project(ProjectionRequest(
        scan_key="owner_b", frame_index=1, generation=1))
    found = decisions(_rejection_events(caplog))
    blanked = found.get(DECISION_PROJECTION_STORE_ABSENT)
    assert blanked is not None, sorted(found)
    assert blanked["expected_owner"] == "owner_b"
    assert blanked["found_owner"] == "owner_a"
    assert blanked["outcome"] == "panel_blanked"
    assert blanked["blanks_panel"] is True
    assert found[DECISION_RECORD_STORE_SKIPPED]["outcome"] == "no_store"
    assert found[DECISION_RECORD_STORE_SKIPPED]["blanks_panel"] is True

    # (e) a superseded request is dropped rather than allowed to overwrite.
    caplog.clear()
    adapter = FrameProjectionAdapter(lambda: None, lambda: publications)
    adapter.project(ProjectionRequest(
        scan_key="owner_a", frame_index=1, generation=7))
    caplog.clear()
    adapter.project(ProjectionRequest(
        scan_key="owner_a", frame_index=1, generation=3))
    found = decisions(_rejection_events(caplog))
    superseded = found.get(DECISION_PROJECTION_SUPERSEDED)
    assert superseded is not None, sorted(found)
    assert superseded["expected_owner"] == "7"
    assert superseded["found_owner"] == "3"
    assert superseded["outcome"] == "request_dropped"
    assert superseded["blanks_panel"] is True

    for event in _rejection_events(caplog):
        assert event["decision_known"] is True
        assert event["decision"] in FAIL_CLOSED_DECISIONS


def test_capability_blanking_is_recorded_on_both_paths(
        qapp, acquisition, monkeypatch, caplog):
    """§61.4 A.2 — availability blanking, on BOTH clear paths, with identity.

    The first cut emitted only from the processing-persistence override, so an
    ordinary idle/paused capability blank went unreported, and it read identity
    off one ``Capability`` (which has none), so every record's ``found_owner``
    was empty.  Both are load-bearing rows here.
    """
    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    caplog.set_level(logging.INFO)

    widget, _recorder = acquisition()
    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 2)

    records = [event for event in _rejection_events(caplog)
               if event["decision"] == DECISION_CAPABILITY_FORCES_CLEAR]
    assert records, "no capability blanking was recorded at all"
    outcomes = {event["outcome"] for event in records}
    assert {"normal_clear", "persistence_override"} <= outcomes, (
        f"capability blanking was only observed on {sorted(outcomes)}; both "
        "the ordinary clear delegate and the persistence override must report")
    assert any(event["found_owner"] for event in records), (
        "every capability record carried an empty found_owner — identity must "
        "come from the pinned projection's DisplayCapabilities")
    for event in records:
        assert event["blanks_panel"] is True
        assert event["role"]
        assert event["reason"]
        assert event["capability_state"]
        assert "pin_scan_key" in event


def test_disabled_channel_installs_nothing_and_evaluates_nothing(
        qapp, monkeypatch, tmp_path):
    """§61.4 A.3 — the disabled channel is a true no-op at construction and load.

    Two things are proved: the file worker carries no live store or reference
    (the §61.3 residual — the first cut installed the live ``PublicationStore``
    on it unconditionally), and a real browse load with the channel off mints no
    operation object, so none of the transition arguments were evaluated.
    """
    monkeypatch.delenv("XDART_RUN_CONFIG_DEBUG", raising=False)
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        file_thread = widget.h5viewer.file_thread
        assert not hasattr(file_thread, "diagnostic_publication_store"), (
            "the file worker carries a live publication store with the "
            "channel disabled")
        for name in vars(file_thread):
            assert "publication" not in name, (
                f"file worker attribute {name!r} looks like a live store "
                "carrier")
        assert widget.h5viewer._display_context_operation is None
        assert widget.h5viewer.diagnostic_run_identity is None

        _browse_load(qapp, widget, _B_RESULT)

        assert widget.h5viewer._display_context_operation is None, (
            "a browse load with the channel disabled built an operation "
            "identity; the transition arguments were evaluated")
        assert widget.h5viewer.diagnostic_run_identity is None
    finally:
        _teardown(qapp, widget)


def test_paused_browse_resume_emits_the_context_transition_trace(
        qapp, acquisition, monkeypatch, caplog):
    """GREEN PIN — the O-2 instrumentation covers the whole boundary sequence.

    Drives the full paused A -> browse B -> Resume A sequence with the channel
    on and asserts that every one of the five boundaries reports, with the
    ownership fields a reviewer needs to attribute a mutation: origin, the
    run/context/display generations, scan key and path, GI identity, PONI and
    mask identity, both stores' owner and generation, and the TARGET object.

    Deliberately asserts COVERAGE, not today's wrong answer: the same assertions
    must still hold once O-3 gives the browse its own context, so this pin
    protects the diagnostics across the fix instead of having to be deleted with
    it.  The trace itself — where today's singleton mutation is visible as one
    ``object_id`` under every role — is written out for the handback when
    ``XDART_O2_TRACE_OUT`` names a file.
    """
    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    caplog.set_level(logging.INFO)

    widget, recorder = acquisition()
    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 2)
    frames_at_resume = len(recorder.frames)
    _resume_acquisition(qapp, widget, recorder)
    _wait_until(qapp, lambda: len(recorder.frames) > frames_at_resume + 1,
                _RUN_TIMEOUT_S, "the next reduced frame from acquisition A")
    _pump(qapp, 2.0)

    events = _transition_events(caplog)
    trace_out = os.environ.get("XDART_O2_TRACE_OUT")
    if trace_out:
        Path(trace_out).write_text(
            "\n".join(json.dumps(event, sort_keys=True)
                      for event in events + _rejection_events(caplog)),
            encoding="utf-8")

    phases = {event["phase"] for event in events}
    assert set(DISPLAY_CONTEXT_PHASES) <= phases, (
        f"missing display-context transitions: "
        f"{sorted(set(DISPLAY_CONTEXT_PHASES) - phases)}")
    assert all(event["phase_known"] for event in events)

    for event in events:
        context = event["context"]
        assert event["origin"], f"{event['phase']} reported no origin"
        for role in ("acquisition", "shared", "target"):
            assert role in context, f"{event['phase']} omitted the {role} role"
        for key in ("run_generation", "config_generation",
                    "display_generation", "load_generation"):
            assert key in context, f"{event['phase']} omitted {key}"
        assert "record_store" in context
        # The publication store is the carrier the rescope boundary destroys, so
        # every transition must be able to name its owner and generation —
        # including the two the browse chain emits without a static widget.
        publications = context["publication_store"]
        assert publications["present"] is True, (
            f"{event['phase']} reported no publication store")
        assert {"owner", "generation", "count"} <= set(publications)

    # The target object identity is what makes the trace load-bearing: a browse
    # boundary must always name the object it is about to write to.
    browse_finish = [event for event in events
                     if event["phase"] == "browse_load_finish"]
    assert browse_finish, "no browse_load_finish was emitted"
    for event in browse_finish:
        target = event["context"]["target"]
        assert target["present"] is True
        assert target["object_id"]
        assert target["scan_key"]
        assert "poni" in target and "global_mask" in target
        assert event["origin"] == "H5Viewer.thread_finished", (
            "the finish must come from the GUI-thread completion seam, not "
            "the file worker")

    # No array, no live Qt object: the whole trace round-trips through JSON and
    # every payload stays a bounded identity record.
    for event in events:
        assert len(json.dumps(event)) < 20000


def test_fail_closed_rejection_is_recorded_for_a_cross_owner_publication(
        qapp, acquisition, monkeypatch, caplog):
    """GREEN PIN — the previously-silent blanking decision now reports.

    A paused browse asks the projection for a frame while the acquisition's
    scan-qualified record store is still attached; the store is skipped and the
    publication view rejects what does not belong to the requested scan.  Both
    were invisible before O-2, which is why "the panel blanked" and "the frame
    has not arrived" were indistinguishable in every captured live log.
    """
    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    caplog.set_level(logging.INFO)

    widget, _recorder = acquisition()
    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 2)

    rejections = _rejection_events(caplog)
    assert rejections, "no fail-closed rejection was recorded for the browse"
    decisions = {event["decision"] for event in rejections}
    assert DECISION_RECORD_STORE_SKIPPED in decisions, (
        "the acquisition's scan-qualified record store was skipped without a "
        "record")
    for event in rejections:
        assert event["decision_known"] is True
        assert event["reason"], f"{event['decision']} recorded no reason"
        assert event["outcome"], f"{event['decision']} recorded no outcome"
        assert isinstance(event["blanks_panel"], bool)
        assert "expected_owner" in event and "found_owner" in event
        assert event["origin"]
        # A rejection that blanks a panel must always name what it wanted.
        if event["blanks_panel"]:
            assert event["expected_owner"], (
                f"{event['decision']} blanked a panel without naming the "
                "expected owner")
