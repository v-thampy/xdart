# -*- coding: utf-8 -*-
"""O-2 — R4-D remainder: production-wired acquisition/browse context reproducers.

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
the reason O-3 must fix.  Each one ends in a single composite assertion of an
all-correct expectation, with the tuple captured at this tip recorded beside it —
so a partial O-3 that fixes one carrier cannot silently flip the test.  The green
pins beside them fix the behaviour that is ALREADY right, so O-3 cannot "fix" a
reproducer by regressing a neighbour.

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
    publication_serves_scan,
)
from xdart.gui.tabs.static_scan.run_config_debug import (
    DISPLAY_CONTEXT_PHASES,
    DISPLAY_CONTEXT_TRANSITION_EVENT,
    FAIL_CLOSED_REJECTION_EVENT,
)
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
    scan_name_from_source,
)
from xrd_tools.session import CapabilityDisposition

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
    ``set_datafile``: the exact chain the contract names, with no shortcut into
    ``scan.set_datafile`` and no synthetic file thread.
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
# Ownership snapshots
# --------------------------------------------------------------------------- #

def _object_id(value):
    return None if value is None else id(value)


def _publication_owner(store, label):
    """The owner a publication DECLARES, or ``None`` when the label is unheld."""
    publication = store.get(label)
    if publication is None:
        return None
    owner = getattr(publication, "scan_key", None)
    if owner:
        return str(owner)
    source = getattr(publication, "source_identity", None)
    return None if source in (None, "") else str(source)


def _store_labels(store):
    snapshot = getattr(store, "snapshot", None)
    if callable(snapshot):
        return tuple(sorted(snapshot().keys(), key=lambda k: (str(type(k)), str(k))))
    return ()


def _acquisition_carriers(widget):
    """Every carrier the acquisition owns that a browse must not be able to move.

    Identities, not values: the question O-3 answers is "is this still A's
    object?", and comparing array CONTENT would let a browse swap a mask for an
    equal-valued one and still pass.
    """
    scan = getattr(widget, "_x1_run_scan_capture", None) or widget.scan
    store = widget.publication_store
    record_store = widget._active_frame_record_store()
    return {
        "scan_object": _object_id(scan),
        "scan_key": str(getattr(scan, "name", "")),
        "data_file": str(getattr(scan, "data_file", "") or ""),
        "gi": bool(getattr(scan, "gi", False)),
        "global_mask": _object_id(getattr(scan, "global_mask", None)),
        "cached_data_mask": _object_id(
            getattr(scan, "_cached_data_mask", None)),
        "poni": _object_id(getattr(scan, "_cached_poni", None)),
        "bai_1d_args": dict(getattr(scan, "bai_1d_args", {}) or {}),
        "bai_2d_args": dict(getattr(scan, "bai_2d_args", {}) or {}),
        "record_store_object": _object_id(record_store),
        "record_store_owner": getattr(record_store, STORE_SCAN_KEY_ATTR, None),
        "record_store_labels": tuple(sorted(
            getattr(record_store, "_records", {}) or {})),
        "publication_store_object": _object_id(store),
        "publication_store_generation": store.generation,
        "publication_labels": _store_labels(store),
    }


def _capability_disposition(projection, name):
    caps = getattr(projection, "capabilities", None)
    fact = getattr(caps, name, None)
    return getattr(fact, "disposition", None)


#: A capability whose disposition is ABSENT has nothing to serve — the frame is
#: not in the store the projection was allowed to consult.  Everything else is
#: some flavour of "there is data or a way to get it".
def _capability_servable(projection, name):
    disposition = _capability_disposition(projection, name)
    return disposition is not None and disposition is not (
        CapabilityDisposition.ABSENT)


def _panel_has_image(pair):
    return bool(pair is not None and pair[0] is not None
                and getattr(pair[0], "size", 0))


def _panel_has_trace(plot_data):
    if plot_data is None or len(plot_data) < 2:
        return False
    y = plot_data[1]
    return bool(getattr(y, "size", 0) and np.isfinite(y).any())


def _browsed_scan_servability(widget, expected_key):
    """Is the BROWSED scan fully servable in the paused display?"""
    display = widget.displayframe
    projection = getattr(display, "_current_frame_projection", None)
    scan = widget.scan
    poni = getattr(scan, "_cached_poni", None)
    return {
        "projection_present": bool(getattr(projection, "present", False)),
        "projection_scan_key": str(
            getattr(display, "_current_frame_projection_scan_key", "") or ""),
        "raw_image_rendered": _panel_has_image(display.image_data),
        "cake_rendered": _panel_has_image(display.binned_data),
        "one_d_trace_rendered": _panel_has_trace(display.plot_data),
        "integrated_1d_servable": _capability_servable(
            projection, "integrated_1d"),
        "integrated_2d_servable": _capability_servable(
            projection, "integrated_2d"),
        "raw_servable": _capability_servable(projection, "raw"),
        "title_names_browsed_scan": expected_key in display.ui.labelCurrent.text(),
        "geometry_is_browsed_detector": str(
            getattr(poni, "detector", "")).startswith("Rayonix"),
        "mask_is_browsed_scan": _object_id(
            getattr(scan, "global_mask", None)) is not None,
    }


# --------------------------------------------------------------------------- #
# 1. A paused browse must render the BROWSED scan (R4T-5's gap)
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason=(
        "O-2 reproducer (O-3 flips it): a paused browse of the real Rayonix GI "
        "result cannot serve its own cake or integrated results — the "
        "projection reports both ABSENT because the browse never gets a "
        "context of its own"
    ),
)
def test_paused_browse_renders_browsed_scan(qapp, acquisition):
    """Acceptance path 1 — B must be FULLY servable while A is paused.

    Real Run over the Eiger Standard container, real frames, real Pause, real
    browser gesture into ``fileHandlerThread.set_datafile``, real frame
    selection.  Servability is asserted at BOTH tiers on purpose: the rendered
    panels (what the operator sees) and the typed projection capabilities (what
    the display layer was actually allowed to consult).  A browse that paints
    something while its projection reports ABSENT is not "working" — it is
    showing whatever the singleton happened to be carrying.
    """
    widget, _recorder = acquisition()
    b_key = _B_RESULT.stem

    _browse_load(qapp, widget, _B_RESULT)
    label = _select_frame_row(qapp, widget, 2)
    assert label, "the real selection seam produced no current frame"

    servable = _browsed_scan_servability(widget, b_key)
    expected = {
        "projection_present": True,
        "projection_scan_key": b_key,
        "raw_image_rendered": True,
        "cake_rendered": True,
        "one_d_trace_rendered": True,
        "integrated_1d_servable": True,
        "integrated_2d_servable": True,
        "raw_servable": True,
        "title_names_browsed_scan": True,
        "geometry_is_browsed_detector": True,
        "mask_is_browsed_scan": True,
    }
    # Captured at this tip: cake_rendered / integrated_1d_servable /
    # integrated_2d_servable are all False — the projection falls through to a
    # publication view that owns nothing of B's, so the cake and both integrated
    # tiers report CapabilityDisposition.ABSENT while the raw panel paints B's
    # thumbnail.  Everything else already holds.
    assert servable == expected


# --------------------------------------------------------------------------- #
# 2. Browsing B must leave acquisition A untouched (R4T-6, first half)
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason=(
        "O-2 reproducer (O-3 flips it): the browse load repoints the "
        "ACQUISITION scan object in place — key, GI, mask, PONI and both BAI "
        "argument sets move, and B's publications overwrite A's on colliding "
        "labels"
    ),
)
def test_paused_browse_leaves_acquisition_untouched(qapp, acquisition):
    """Acceptance path 2 — nothing A owns may move while B is browsed.

    Snapshots the acquisition's carriers at Pause, browses B through the real
    file thread, and compares.  Object IDENTITY is the comparison for every
    object-valued carrier: a reference to a mutable object is not a snapshot,
    so only "is it still the same object, still carrying the same declared
    owner?" is a real answer.
    """
    widget, _recorder = acquisition()
    before = _acquisition_carriers(widget)
    shared_label = next(
        (label for label in before["publication_labels"]
         if str(label) in {"1", "2", "3", "4"}), None)
    assert shared_label is not None, (
        "acquisition A published none of the low frame labels B also uses; "
        "the collision this test needs did not occur")
    owner_before = _publication_owner(widget.publication_store, shared_label)
    assert owner_before, "A's publication declared no owner to protect"

    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 2)

    after = _acquisition_carriers(widget)
    observed = {
        "scan_object_unchanged": after["scan_object"] == before["scan_object"],
        "scan_key_unchanged": after["scan_key"] == before["scan_key"],
        "data_file_unchanged": after["data_file"] == before["data_file"],
        "gi_unchanged": after["gi"] == before["gi"],
        "global_mask_unchanged":
            after["global_mask"] == before["global_mask"],
        "cached_data_mask_unchanged":
            after["cached_data_mask"] == before["cached_data_mask"],
        "poni_unchanged": after["poni"] == before["poni"],
        "bai_1d_args_unchanged":
            after["bai_1d_args"] == before["bai_1d_args"],
        "bai_2d_args_unchanged":
            after["bai_2d_args"] == before["bai_2d_args"],
        "record_store_object_unchanged":
            after["record_store_object"] == before["record_store_object"],
        "record_store_owner_unchanged":
            after["record_store_owner"] == before["record_store_owner"],
        "record_store_keeps_its_frames":
            set(before["record_store_labels"])
            <= set(after["record_store_labels"]),
        "publication_store_object_unchanged":
            after["publication_store_object"]
            == before["publication_store_object"],
        "acquisition_publication_survives":
            _publication_owner(widget.publication_store, shared_label)
            == owner_before,
    }
    expected = dict.fromkeys(observed, True)
    # Captured at this tip: scan_key / data_file / gi / global_mask / poni /
    # bai_1d_args / bai_2d_args / acquisition_publication_survives are all
    # False.  ``scan_object_unchanged`` is True for the WRONG reason — the
    # object is the same because it is the same singleton, mutated in place —
    # which is exactly why the key/GI/calibration rows are asserted beside it.
    assert observed == expected


# --------------------------------------------------------------------------- #
# 3. Resume must return coherently to A (the ssw publication-clear boundary)
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason=(
        "O-2 reproducer (O-3 flips it): Resume leaves B's GI flag and PONI on "
        "the acquisition scan, and the first resumed frame drives a scan-"
        "boundary rescope that CLEARS the publication store"
    ),
)
def test_resume_restores_acquisition_coherently(qapp, acquisition):
    """Acceptance path 3 — the next A frame renders as A, and nothing is wiped.

    Resume is driven through the real action seam and the assertions are taken
    only AFTER the worker has delivered another real A frame: the question is
    not "did the flags get restored at the resume signal?" but "did the frame
    that followed actually render in A's geometry, mask, PONI, mode and title?".
    """
    widget, recorder = acquisition()
    before = _acquisition_carriers(widget)
    a_key = before["scan_key"]

    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 2)

    frames_at_resume = len(recorder.frames)
    _resume_acquisition(qapp, widget, recorder)
    _wait_until(qapp, lambda: len(recorder.frames) > frames_at_resume + 1,
                _RUN_TIMEOUT_S, "the next reduced frame from acquisition A")
    _pump(qapp, 2.0)

    after = _acquisition_carriers(widget)
    display = widget.displayframe
    observed = {
        "run_still_owned": widget._run_active is True,
        "scan_key_is_acquisition": after["scan_key"] == a_key,
        "gi_is_acquisition": after["gi"] == before["gi"],
        "poni_is_acquisition": after["poni"] == before["poni"],
        "global_mask_is_acquisition":
            after["global_mask"] == before["global_mask"],
        "bai_1d_args_are_acquisition":
            after["bai_1d_args"] == before["bai_1d_args"],
        "bai_2d_args_are_acquisition":
            after["bai_2d_args"] == before["bai_2d_args"],
        "title_names_acquisition": a_key in display.ui.labelCurrent.text(),
        "publication_store_not_recycled":
            after["publication_store_object"]
            == before["publication_store_object"],
        "publication_store_not_cleared":
            after["publication_store_generation"]
            == before["publication_store_generation"],
        "record_store_owner_is_acquisition":
            after["record_store_owner"] == before["record_store_owner"],
    }
    expected = dict.fromkeys(observed, True)
    # Captured at this tip: gi_is_acquisition, poni_is_acquisition,
    # bai_1d_args_are_acquisition, bai_2d_args_are_acquisition and
    # publication_store_not_cleared are False.  The scan KEY and the title do
    # come back — but only because the first resumed frame drives a
    # frame-boundary rescope, and that same rescope is what bumps the
    # publication store's generation and drops A's retained frames.  Resume is
    # a context SELECTION; nothing about it should be a restoration procedure.
    assert observed == expected


# --------------------------------------------------------------------------- #
# 4. A delayed browse hydration must be rejected after Resume
# --------------------------------------------------------------------------- #

class _HydrationLatch:
    """Suspend ONE real hydration inside the production hydrator.

    This is not a stand-in for the seam under test: the registered production
    hydrator still performs the read, and the real
    ``FrameHydrationWorker`` -> ``sigHydrated`` -> ``_on_frame_hydrated`` round
    trip is untouched.  All the latch does is make "the completion lands after
    Resume" a fact rather than a race.
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
        store.set_hydrator(self._hold)

    def _hold(self, label):
        self.labels.append(label)
        self.entered.set()
        # Bounded: a latch that never releases must fail the test, not wedge
        # the session's worker thread forever.
        self.release.wait(_BOUNDARY_TIMEOUT_S)
        return self._inner(label)

    def restore(self):
        self.release.set()
        # Only un-wrap if the latch is still the registered hydrator: a resume
        # that re-scoped the store may already have installed a fresh one, and
        # restoring over it would leave the store pointing at a stale closure.
        if getattr(self._store, "_hydrator", None) is self._hold:
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
        "display generation and no owner, so a browse completion released "
        "after Resume is never rejected by CONTEXT identity and no fail-closed "
        "rejection is recorded for it"
    ),
)
def test_delayed_browse_hydration_rejected_after_resume(
        qapp, acquisition, monkeypatch, caplog):
    """Acceptance path 4 — a stale B completion must be refused by identity.

    One real B hydration is held inside the production hydrator, A is resumed
    through the real action seam, and only then is the completion released.  A
    correct round trip knows the request belonged to the browse context and
    refuses it; a generation number cannot answer that question, which is the
    contract this pins.
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
    a_carriers = _acquisition_carriers(widget)

    _browse_load(qapp, widget, _B_RESULT)
    selected = _select_frame_row(qapp, widget, 2)
    browse_key = widget.scan.name

    # The production enable-once seam the live app calls; headless tests leave
    # hydration synchronous, so this is what puts the REAL worker in the path.
    display.enable_async_hydration()
    latch = _HydrationLatch(widget.publication_store)
    try:
        # Ask for a B frame OTHER than the selected one whose heavy tier the
        # store has actually evicted — the production guards refuse to request a
        # resident tier, and rightly so.
        listing = widget.h5viewer.ui.listData
        target_label = next(
            (int(listing.item(row).text())
             for row in range(listing.count())
             if listing.item(row).text() != selected
             and not display._hydration_purpose_resident(
                 int(listing.item(row).text()), "full")),
            None)
        assert target_label is not None, (
            "no browsed frame had an evicted heavy tier, so no real hydration "
            "could be put in flight")
        display._request_frame_hydration(target_label, purpose="full")
        _wait_until(qapp, latch.entered.is_set, _BOUNDARY_TIMEOUT_S,
                    "the real hydrator to enter the latch")

        hydrated = []
        display._hydration_worker.sigHydrated.connect(
            lambda label, generation: hydrated.append((label, generation)))

        frames_at_resume = len(recorder.frames)
        _resume_acquisition(qapp, widget, recorder)
        _wait_until(qapp, lambda: len(recorder.frames) > frames_at_resume,
                    _RUN_TIMEOUT_S, "acquisition A to produce a frame again")

        # Now let the stale browse completion land.
        latch.release.set()
        _wait_until(qapp, lambda: bool(hydrated), _BOUNDARY_TIMEOUT_S,
                    "the held hydration completion")
        _pump(qapp, 1.5)
    finally:
        latch.restore()

    assert requests, "the production request seam was never reached"
    request_kwargs = requests[-1][2]
    owner_keys = {"owner", "owner_token", "request_owner", "context_id"}
    rejections = _rejection_events(caplog)
    after = _acquisition_carriers(widget)
    observed = {
        "request_carries_owner": bool(
            owner_keys & set(request_kwargs)
            or {"context_id", "scan_key"} <= set(request_kwargs)),
        "completion_rejected_by_context": any(
            event.get("decision", "").startswith("hydration")
            or browse_key in str(event.get("found_owner", ""))
            for event in rejections),
        "no_browse_payload_adopted": _publication_owner(
            widget.publication_store, target_label) not in (browse_key,),
        "calibration_not_adopted":
            after["poni"] == a_carriers["poni"]
            or after["scan_key"] == a_carriers["scan_key"],
        "mask_not_adopted":
            after["global_mask"] == a_carriers["global_mask"],
    }
    expected = dict.fromkeys(observed, True)
    # Captured at this tip: request_carries_owner is False (the request seam
    # takes label/generation/purpose/consumer and nothing that identifies the
    # context), and completion_rejected_by_context is False (nothing rejects
    # the completion by owner, so no fail-closed rejection is recorded for it).
    assert observed == expected


# --------------------------------------------------------------------------- #
# 5a. Same-label collisions across contexts
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
    b_key = widget.scan.name
    b_publication = widget.publication_store.get(1)
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
        "ONE publication store, so B's browse publications overwrite A's "
        "retained frames label for label"
    ),
)
def test_same_label_collision_preserves_both_owners(qapp, acquisition):
    """Acceptance path 5a — a shared label must not destroy the other owner.

    The predicate above proves a cross-owner publication cannot be SERVED; this
    proves the deeper problem, which is that A's publication no longer exists to
    be refused.  A store keyed by label alone cannot hold two contexts at once.
    """
    widget, _recorder = acquisition()
    a_key = widget.scan.name
    shared = tuple(
        label for label in _store_labels(widget.publication_store)
        if str(label) in {"1", "2", "3"})
    assert shared, "acquisition A retained none of the colliding low labels"
    owners_before = {
        label: _publication_owner(widget.publication_store, label)
        for label in shared
    }

    _browse_load(qapp, widget, _B_RESULT)
    _select_frame_row(qapp, widget, 0)

    owners_after = {
        label: _publication_owner(widget.publication_store, label)
        for label in shared
    }
    observed = {
        "acquisition_owners_survive": owners_after == owners_before,
        "acquisition_key_still_owns_them": all(
            owner == a_key for owner in owners_after.values()),
    }
    # Captured at this tip: both are False — every shared label now reports the
    # browsed Rayonix source as its owner.
    assert observed == {
        "acquisition_owners_survive": True,
        "acquisition_key_still_owns_them": True,
    }


# --------------------------------------------------------------------------- #
# 5b. Dotted scan stems — the two-parser divergence
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
        "O-2 reproducer (O-3 flips it): the browse path names the scan with "
        "set_datafile's first-dot truncation instead of the canonical "
        "scan_name_from_source, so a dotted stem loses everything after the "
        "first dot"
    ),
)
def test_browse_load_names_a_dotted_stem_canonically(
        qapp, monkeypatch, tmp_path):
    """Acceptance path 5b — ONE parser must name the browsed scan.

    Driven through the real browser gesture and the real file thread, not by
    calling either parser directly: the divergence only matters because the
    browse path picks the wrong one.
    """
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        dotted = _dotted_stem_link(tmp_path, _B_RESULT)
        canonical = scan_name_from_source(str(dotted))
        _browse_load(qapp, widget, dotted)

        observed = {
            "scan_key": widget.scan.name,
            "data_file": os.path.basename(widget.scan.data_file or ""),
        }
        # Captured at this tip: scan_key is "Combi4" — set_datafile splits on
        # the FIRST dot, so "v2_03271005" is silently discarded while the data
        # file (and every title derived from the canonical parser) keeps it.
        assert observed == {
            "scan_key": canonical,
            "data_file": dotted.name,
        }
    finally:
        _teardown(qapp, widget)


# --------------------------------------------------------------------------- #
# The O-2 diagnostics contract (and the captured trace artifact)
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
    assert browse_finish, "the file thread emitted no browse_load_finish"
    for event in browse_finish:
        target = event["context"]["target"]
        assert target["present"] is True
        assert target["object_id"]
        assert target["scan_key"]
        assert "poni" in target and "global_mask" in target

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
    assert decisions & {
        "record_store_owner_mismatch",
        "publication_owner_mismatch",
        "projection_store_absent",
        "projection_superseded",
        "capability_forces_clear",
    }, f"unexpected rejection decisions: {sorted(decisions)}"
    for event in rejections:
        assert event["reason"], f"{event['decision']} recorded no reason"
        assert "expected_owner" in event and "found_owner" in event
        assert event["origin"]
