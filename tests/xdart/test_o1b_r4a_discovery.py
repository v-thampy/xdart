"""O-1b frozen oracle — R4A-4 walk order, R4A-5 IN_PROGRESS deferral, R4A-6
honest eligibility.

Red at `c4ff402a`, driven through the real production helpers.  The preview rows
live in ``test_o1b_r4a_lazy_preview.py``; these are the discovery and eligibility
halves of the same lazy-discovery contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------- #
# R4A-4 — the name-only listing is natural-sorted GLOBALLY, not per scandir
# batch, so `scan_2` precedes `scan_10` in a directory wider than one batch.
# --------------------------------------------------------------------------- #

def _walk_raw(root, *, recursive=False, suffixes=(".nxs",)):
    """The production generator itself, unfiltered."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )

    host = SimpleNamespace()
    host._directory_walk_items = MethodType(
        imageThread._directory_walk_items, host)
    return host._directory_walk_items(
        root, recursive=recursive, suffixes=suffixes, match=lambda _stem: True)


def _walk_matches(root, *, recursive=False, suffixes=(".nxs",)):
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )

    host = SimpleNamespace()
    host._directory_walk_items = MethodType(
        imageThread._directory_walk_items, host)
    yielded = host._directory_walk_items(
        root, recursive=recursive, suffixes=suffixes, match=lambda _stem: True)
    return [path for path in yielded if path is not None]


def test_global_natural_order_survives_a_directory_wider_than_one_batch(
        tmp_path):
    """R4A-4.  The scanner natural-sorted each 64-entry batch in isolation, so a
    wide directory yielded `scan_100` before `scan_2`.

    The files are created in REVERSE natural order so batch-local sorting cannot
    accidentally produce the global order.
    """
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        natural_sort_ints,
    )

    total = 130
    for index in range(total, 0, -1):
        (tmp_path / f"scan_{index}.nxs").touch()

    got = [path.name for path in _walk_matches(tmp_path)]
    expected = [Path(name).name for name in natural_sort_ints(
        [str(tmp_path / f"scan_{index}.nxs") for index in range(1, total + 1)])]

    assert got[:6] == expected[:6], (
        f"batch-local ordering leaked: first yields {got[:6]}")
    assert got == expected, "the whole directory must be globally ordered"


def test_wide_directory_still_yields_incrementally(tmp_path):
    """The lazy contract: completing the NAME listing must not materialize
    content work for the whole directory before the first match is available."""
    for index in range(200):
        (tmp_path / f"scan_{index:04d}.nxs").touch()

    # The name listing may complete (it is cheap and inside the lazy contract);
    # what must stay incremental is CONSUMPTION.  Pulling one token may not
    # exhaust the walker, and the walker must be a generator rather than a
    # materialized list -- `hasattr(..., "__iter__")` would pass for a list and
    # prove nothing.
    import inspect

    walker = _walk_raw(tmp_path)
    assert inspect.isgenerator(walker), type(walker)
    first = next(walker)
    assert first is None or first.name.endswith(".nxs")
    remaining = sum(1 for _ in walker)
    assert remaining == 199, (
        f"one pull consumed more than one entry; {remaining} left of 200")


# --------------------------------------------------------------------------- #
# R4A-6 — a merely-configured directory must report a typed discovery-deferred
# state, not fabricated counted/has_frames/has_raw/raw_reachable facts, while Run
# stays enabled on configured-intent validity.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    value = staticWidget()
    try:
        yield value
    finally:
        value._controls_v2_refresh_timer.cancel()
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _configure_directory_only(widget, root, *, ext="nxs", recursive=False):
    """Configure an Image Directory source WITHOUT probing anything."""
    wrangler = widget.wrangler
    signal = wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(root))
    signal.child("img_ext").setValue(ext)
    signal.child("include_subdir").setValue(recursive)
    signal.child("File").setValue("")
    return wrangler


def test_configured_directory_does_not_fabricate_frame_or_raw_facts(
        widget, tmp_path):
    """R4A-6.  `directory_intent_ready` reported counted/has_frames/has_raw/
    raw_reachable TRUE for a directory nobody had verified, which is dishonest
    eligibility labelling: the publication gates fail closed, but the readiness
    rows and launchers claim evidence that does not exist."""
    raw = tmp_path / "raw"
    raw.mkdir()
    # Subdirs ON is the configuration that makes `directory_intent_ready` true
    # on configuration ALONE, which is where the fabricated facts come from.
    _configure_directory_only(widget, raw, recursive=True)

    caps, _source_ready = widget._controls_v2_source_caps(
        source_label=str(raw), frame_count=0,
        has_metadata=False, has_motors=False, has_geometry=False,
        has_psi_metadata=False, has_energy=True, live_unknown=False)

    assert caps.has_frames is False, "claimed frames for an unverified directory"
    assert caps.has_raw is False, "claimed raw data for an unverified directory"
    assert caps.raw_reachable is False, (
        "claimed reachable raw data for an unverified directory")
    assert getattr(caps, "discovery_deferred", False) is True, (
        "the typed capability result does not distinguish deferred discovery "
        "from a proved-empty source"
    )


def test_configured_directory_still_enables_run(widget, tmp_path):
    """R4A-6's other half: honesty must not disable a legitimate lazy Run.

    Run enablement keys on configured-intent validity, not on fabricated counts.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    _configure_directory_only(widget, raw)

    # The other direction of the same dishonesty: a VALID configured intent with
    # discovery merely deferred must not lose Run eligibility just because no
    # count has been fabricated for it.
    caps, source_ready = widget._controls_v2_source_caps(
        source_label=str(raw), frame_count=0,
        has_metadata=False, has_motors=False, has_geometry=False,
        has_psi_metadata=False, has_energy=True, live_unknown=False)
    assert source_ready is True, (
        "a configured directory with deferred discovery lost Run eligibility")
    assert getattr(caps, "discovery_deferred", False) is True
    assert widget.wrangler._directory_run_without_seed_ok() is True


# --------------------------------------------------------------------------- #
# R4A-5 — batch mode must DEFER an in-progress container un-retired and
# re-check it before end-of-run, not retire it at first sight.
#
# The live-watch half of this property is already pinned by
# `test_bluesky_image_wrangler.py::test_f5_unfinalized_nxs_deferred_then_consumed_in_full`
# (F5/DIR-2).  The R4-A finding names BATCH mode, which has no watch loop to
# re-poll it, so this is that missing half.
# --------------------------------------------------------------------------- #

def _batch_dir_thread(watch_dir, out_dir):
    """A real imageThread over a directory of .nxs containers, in BATCH mode."""
    import threading
    from queue import Queue

    from tests.xdart._accepted_run import admitted_worker, directory_source
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )
    from xdart.modules.live import LiveScan
    from xrd_tools.core.containers import PONI

    scan = LiveScan("scan", data_file=str(out_dir / "scan.nxs"), static=True)
    # R4-G: retired policy arguments removed; the watched directory and every
    # other run policy arrive through `admitted_worker` below.
    worker = imageThread(
        Queue(), threading.RLock(), "",
        "scan",
        PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10),
        "",
        "None", "", "", None, "", "", 1.0, None,
        "q_total", "qip_qoop", "start", scan,
    )
    return admitted_worker(
        worker,
        batch_mode=True,
        save_path=str(out_dir),
        source_spec=directory_source(watch_dir, ext="nxs"))


def test_batch_prefetch_generation_consumes_a_container_finalizing_midrun(
        tmp_path):
    """R4A-5, production-shaped (review §49.6 correction A).

    The parent row called ``_get_next_eiger_frame_sync()`` a second time after
    an end-of-stream tuple.  Production never does that: ``_prefetch_worker``
    queues the terminal tuple and EXITS, so that row could pass while a real
    batch still dropped a container that changed from ``IN_PROGRESS`` to
    complete.

    This drives ONE real prefetch generation through the production entry.  A
    deterministic latch pauses the worker inside the provisional classification
    (no sleep, no spin loop), the fixture finalizes while that generation is
    still live, and the whole label range must then arrive exactly once from
    that same generation -- no second Run, no restarted reader, no post-EOS
    manual call.
    """
    import threading

    import h5py

    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    growing = watch / "grow_00001.nxs"
    _write_bluesky_nxwriter(growing, n=2)
    with h5py.File(growing, "r+") as handle:
        del handle["entry/end_time"]

    worker = _batch_dir_thread(watch, out)
    frozen = worker.run_configuration

    observed = threading.Event()
    resume = threading.Event()
    real_open = worker._eiger_open_master

    def latching_open(*args, **kwargs):
        result = real_open(*args, **kwargs)
        if getattr(worker, "_eiger_open_state", None) == "not ready":
            # Pause the LIVE generation here so the finalize below lands after
            # the IN_PROGRESS observation and before terminal retirement.
            observed.set()
            if not resume.wait(10.0):
                raise AssertionError(
                    "the test owner did not finalize/release the provisional "
                    "container before the deterministic timeout")
        return result

    worker._eiger_open_master = latching_open

    # Enter through the consumer-facing production route.  Draining a private
    # queue after producer termination would be a false green: a broken worker
    # could enqueue EOS and only then enqueue frames, while production stops at
    # the first EOS and never consumes those later frames.
    start_calls = []
    real_start = worker._start_prefetcher

    def recording_start(config):
        result = real_start(config)
        start_calls.append(worker._prefetch_thread)
        return result

    worker._start_prefetcher = recording_start
    delivered = []

    def consume_one_generation():
        for _ in range(16):
            item = worker.get_next_image(frozen)
            delivered.append(item)
            if item[3] is None:
                return
        raise AssertionError("one prefetch generation never delivered EOS")

    consumer = threading.Thread(
        target=consume_one_generation,
        name="o1b-r4a5-consumer", daemon=True)
    consumer.start()
    try:
        assert observed.wait(10.0), (
            "the provisional container was never classified IN_PROGRESS, so "
            "this case cannot speak to the deferral at all")
        assert str(growing) not in worker._eiger_done_masters, (
            "batch RETIRED a provisional container at first sight, so the "
            "frames it flushes later in the SAME run can never be read")
        _write_bluesky_nxwriter(growing, n=5)
    finally:
        resume.set()
    consumer.join(timeout=30.0)
    assert not consumer.is_alive(), "the production consumer never terminated"
    assert len(start_calls) == 1, (
        f"the same batch Run started {len(start_calls)} prefetch generations")
    producer = start_calls[0]
    assert producer is worker._prefetch_thread
    producer.join(timeout=30.0)
    assert not producer.is_alive(), "the prefetch generation never terminated"
    worker._prefetch_stop_evt.set()

    terminal = [index for index, item in enumerate(delivered)
                if item[3] is None]
    assert terminal == [len(delivered) - 1], (
        "one terminal sentinel must follow every frame exactly once; "
        f"terminal positions={terminal}, items={delivered!r}")
    assert worker._prefetch_queue.empty(), (
        "the producer queued data after the terminal sentinel; production "
        "would stop at EOS and drop it")
    labels = [item[2] for item in delivered if item[3] is not None]

    assert labels == [1, 2, 3, 4, 5], (
        "one batch prefetch generation must emit every label exactly once "
        f"after a mid-run finalization; got {labels}")
    assert str(growing) in worker._eiger_done_masters, (
        "retired only after a finalized drain")
