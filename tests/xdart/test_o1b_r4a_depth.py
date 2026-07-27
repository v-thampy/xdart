"""O-1b R4A implementation-depth and mutation rows.

The frozen acceptance oracle lives in ``test_o1b_r4a_lazy_preview.py`` and
``test_o1b_r4a_discovery.py`` and is unchanged.  These rows are the
§49.3/§49.4 depth coverage the review requires ALONGSIDE it: each one is named
by a required mutation (``P1``..``P8`` / ``D1``..``D7``) and must go red when
that mutation is applied to production.

They are deliberately production-wired: real ``imageWrangler`` /
``imageThread`` objects, the real Directory ``get_next_image`` route, real
on-disk NXWriter containers, and the production admission helper -- no stub sits
on a seam under test (CLAUDE.md rule 2).
"""

from __future__ import annotations

import os
import threading
import types
from pathlib import Path
from types import MethodType

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")

from tests.core.test_bluesky_nexus import (  # noqa: E402
    _write_bluesky_nxwriter,
)


class _FakeSignal:
    def __init__(self):
        self.emissions = []

    def emit(self, *values):
        self.emissions.append(values)

    def connect(self, *_a, **_k):
        pass


def _holder(tmp_path, *, recursive=False, ext="nxs", meta_ext="auto"):
    """A real ``imageWrangler`` method host over a real parameter tree."""
    import xdart.gui.gui_utils  # noqa: F401  # registers 'str_browse'
    from pyqtgraph.parametertree import Parameter

    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
        params,
    )

    root = Parameter.create(
        name="image_wrangler", type="group", children=params)
    signal = root.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(tmp_path))
    signal.child("img_ext").setValue(ext)
    signal.child("include_subdir").setValue(recursive)
    signal.child("File").setValue("")

    host = types.SimpleNamespace(
        parameters=root,
        img_file="", img_dir=str(tmp_path), img_ext=ext,
        inp_type="Image Directory", single_img=False,
        include_subdir=recursive, meta_ext=meta_ext, meta_dir="",
        file_filter="", scan_parameters=[], motors=[], counters=[],
        incidence_motor="th", poni=None, _bluesky_cols_cache=None,
        source_spec=None, _gi_motor_knowledge_proved=False,
        sigGIMotorOptions=_FakeSignal(),
        showLabel=_FakeSignal(),
    )
    for name in (
        "_read_bluesky_source_columns", "get_scan_parameters",
        "set_pars_from_meta", "set_gi_motor_options", "set_gi_th_motor",
        "set_bg_norm_options", "set_bg_matching_options", "exists_meta_file",
        "_sync_meta_ext_to_img_ext", "_directory_metadata_preview_suffixes",
        "_directory_metadata_preview_file", "_adopt_directory_metadata_preview",
        "get_img_fname", "_gi_source_fingerprint",
        "_next_gi_hydration_generation", "_gi_hydration_registry",
        "_retire_gi_hydration_token", "_emit_gi_hydration",
        "_announce_gi_hydration",
    ):
        setattr(host, name, MethodType(getattr(imageWrangler, name), host))
    return host, root


def _motor_choices(root):
    return list(root.child("GI").child("th_motor").opts["limits"])


# --------------------------------------------------------------------------- #
# P2 — ONE usability definition: discovery and adoption may not each open the
# same candidate.
# --------------------------------------------------------------------------- #

def test_discovery_and_adoption_open_each_candidate_exactly_once(tmp_path):
    """P2.  Discovery selecting a candidate and adoption independently deciding
    usability is the §50.1 root cause: two predicates drift, and the second one
    re-opens what the first already read.  One shared immutable result means a
    usable candidate is content-opened EXACTLY once for the whole selection."""
    _write_bluesky_nxwriter(tmp_path / "scan_0001.nxs")

    host, root = _holder(tmp_path)
    opens = []
    real_reader = host._read_bluesky_source_columns
    host._read_bluesky_source_columns = (
        lambda path: opens.append(str(path)) or real_reader(path))

    host.get_img_fname()

    assert "hy" in _motor_choices(root), _motor_choices(root)
    assert opens == [str(tmp_path / "scan_0001.nxs")], (
        "discovery and adoption did not share ONE inspection result; "
        f"opens={opens}")


def test_unusable_candidates_are_opened_once_each_before_the_usable_one(
        tmp_path):
    """P1/P2 together: the fallback inspects each candidate once, in global
    natural order, and stops at the first usable one -- it neither re-opens a
    rejected candidate nor keeps walking past a usable result."""
    (tmp_path / "aaa_scan.nxs").write_bytes(b"")
    (tmp_path / "bbb_scan.nxs").write_bytes(b"")
    _write_bluesky_nxwriter(tmp_path / "ccc_scan.nxs")
    _write_bluesky_nxwriter(tmp_path / "ddd_scan.nxs")

    host, root = _holder(tmp_path)
    opens = []
    real_reader = host._read_bluesky_source_columns
    host._read_bluesky_source_columns = (
        lambda path: opens.append(Path(path).name) or real_reader(path))

    host.get_img_fname()

    assert "hy" in _motor_choices(root)
    assert opens == ["aaa_scan.nxs", "bbb_scan.nxs", "ccc_scan.nxs"], opens


# --------------------------------------------------------------------------- #
# R4A-1 — Subdirs is an INPUT to preview discovery, so toggling it must
# re-discover rather than re-serve a cached "nothing usable".
# --------------------------------------------------------------------------- #

def test_toggling_subdirs_rediscovers_instead_of_reusing_the_cached_answer(
        tmp_path):
    """The real GUI ordering.  The operator picks a root (Subdirs OFF, nothing
    usable at the top level) and THEN ticks Subdirs.  The directory's own mtime
    has not changed, so a discovery key that omits the recursive flag re-serves
    the cached empty answer and the dropdown never fills -- which is precisely
    the R4A-1 symptom, just reached through the panel instead of the fixture."""
    child = tmp_path / "day1"
    child.mkdir()
    _write_bluesky_nxwriter(child / "scan_0001.nxs")

    host, root = _holder(tmp_path, recursive=False)
    host.get_img_fname()
    assert _motor_choices(root) == ["Manual"]

    root.child("Signal").child("include_subdir").setValue(True)
    host.get_img_fname()

    assert "hy" in _motor_choices(root), (
        "ticking Subdirs re-served the cached no-preview answer; "
        f"dropdown={_motor_choices(root)}")


# --------------------------------------------------------------------------- #
# P5/P6/P7 — mid-run JIT hydration: exact-run qualification, once per run, and
# no write-back into the frozen execution policy.
# --------------------------------------------------------------------------- #

def _jit_rig(tmp_path):
    """A real wrapper + worker over a directory whose data has NOT landed yet.

    This is the production R4A-1(iii) scenario: the pre-Run preview genuinely
    finds nothing, so anything that later fills the dropdown can only be the
    mid-run delivery.
    """
    from pyqtgraph.Qt import QtWidgets

    import xdart.gui.gui_utils  # noqa: F401  # registers str_browse
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
    )
    from xdart.modules.live import LiveScan

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    raw = tmp_path / "raw"
    (raw / "day1").mkdir(parents=True)
    output = tmp_path / "processed"
    output.mkdir()
    scan = LiveScan(
        "preview", data_file=str(output / "preview.nxs"), static=True)
    wrapper = imageWrangler("", threading.RLock(), scan)
    signal = wrapper.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(raw))
    signal.child("img_ext").setValue("nxs")
    signal.child("include_subdir").setValue(True)
    signal.child("File").setValue("")
    wrapper.get_img_fname()
    return app, wrapper, raw, output


def _drive_run_route(worker, frozen, *, limit=8):
    frames = []
    for _ in range(limit):
        item = worker.get_next_image(frozen)
        if item[3] is None:
            break
        frames.append(item)
    return frames


def test_jit_hydration_fills_the_dropdown_when_data_lands_during_the_run(
        tmp_path):
    """R4A-1(iii), end to end.

    Nothing is discoverable when the operator selects the directory, so the
    dropdown is ``['Manual']`` with the operator hint.  The container lands
    during the run; the FIRST classified container publishes exactly one
    immutable value; the wrapper translates it onto the existing hydration
    signal; a duplicate frame republishes nothing (P7); and the accepted
    configuration plus its effective motor are untouched (P6).
    """
    from tests.xdart._accepted_run import (
        accepted_run,
        admitted_worker,
        directory_source,
        gi_intent,
    )

    app, wrapper, raw, output = _jit_rig(tmp_path)
    worker = wrapper.thread
    try:
        assert _motor_choices(wrapper.parameters) == ["Manual"], (
            "the preview found something, so this case cannot attribute a "
            "filled dropdown to the mid-run delivery")

        frozen = accepted_run(
            save_path=str(output),
            source_spec=directory_source(raw, ext="nxs", recursive=True),
            gi=gi_intent(enabled=True, incidence_motor="halpha"),
        )
        admitted_worker(wrapper, frozen=frozen)
        admitted_worker(worker, frozen=frozen)
        effective_before = frozen.gi.effective_motor

        emitted = []
        wrapper.sigGIMotorOptions.connect(emitted.append)

        # The acquisition writes the container AFTER Run started.
        _write_bluesky_nxwriter(raw / "day1" / "scan_0001.nxs")

        frames = _drive_run_route(worker, frozen)
        assert len(frames) >= 2, (
            f"too few frames to test republication: {[f[2] for f in frames]}")
        app.processEvents()

        assert len(emitted) == 1, (
            "exactly ONE immutable hydration value is published per run; "
            f"{len(emitted)} were emitted across {len(frames)} frames")
        published = emitted[0]
        assert "hy" in tuple(published.motors)
        assert "hy" in _motor_choices(wrapper.parameters)

        # P7 — a duplicate frame must not republish.
        worker.get_next_image(frozen)
        app.processEvents()
        assert len(emitted) == 1, "a duplicate frame republished the hydration"

        # A stale/foreign completion stays inert on the existing chain.
        foreign = published._replace(source_fingerprint="foreign-source")
        assert wrapper.gi_hydration_is_current(foreign) is False
        assert "hy" in _motor_choices(wrapper.parameters)

        # P6 — nothing was written back into the frozen execution policy.
        assert wrapper.run_configuration is frozen
        assert worker.run_configuration is frozen
        assert frozen.gi.effective_motor == effective_before == "halpha"
    finally:
        worker._eiger_close_master()
        wrapper.close()
        wrapper.deleteLater()
        app.processEvents()


def test_jit_delivery_without_the_exact_admitted_object_is_inert(tmp_path):
    """P5.  Qualification is ``is`` against the exact accepted object, so an
    EQUAL-VALUED configuration -- the shape a "close enough" equality check
    would accept -- publishes nothing.  A replayed delivery carrying the right
    object is inert too, because publication is once per run."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        GISourceMotorDiscovery,
    )
    from tests.xdart._accepted_run import (
        accepted_run,
        admitted_worker,
        directory_source,
        gi_intent,
    )

    app, wrapper, raw, output = _jit_rig(tmp_path)
    worker = wrapper.thread
    try:
        def _config():
            return accepted_run(
                save_path=str(output),
                source_spec=directory_source(raw, ext="nxs", recursive=True),
                gi=gi_intent(enabled=True, incidence_motor="halpha"),
            )

        frozen = _config()
        twin = _config()
        assert twin is not frozen
        admitted_worker(wrapper, frozen=frozen)
        admitted_worker(worker, frozen=frozen)

        emitted = []
        wrapper.sigGIMotorOptions.connect(emitted.append)

        def _deliver(config):
            worker.sigGISourceMotors.emit(GISourceMotorDiscovery(
                run_configuration=config,
                source_path=str(raw / "day1" / "scan_0001.nxs"),
                motors=("hy",), counters=("i0",)))
            app.processEvents()

        _deliver(twin)
        assert emitted == [], (
            "a foreign but equal-valued configuration was accepted; "
            "qualification is not identity-based")
        assert _motor_choices(wrapper.parameters) == ["Manual"]

        _deliver(frozen)
        assert len(emitted) == 1
        assert "hy" in _motor_choices(wrapper.parameters)

        _deliver(frozen)
        assert len(emitted) == 1, "a replayed delivery republished the value"
    finally:
        wrapper.close()
        wrapper.deleteLater()
        app.processEvents()


def test_worker_publishes_at_most_once_per_run_across_containers(tmp_path):
    """P7, worker side.  The publication latch is keyed on the accepted
    configuration OBJECT, so a SECOND classified container in the same run is
    silent, and a genuinely new run re-arms with no reset step to forget."""
    from tests.xdart._accepted_run import (
        accepted_run,
        admitted_worker,
        directory_source,
        gi_intent,
    )

    app, wrapper, raw, output = _jit_rig(tmp_path)
    worker = wrapper.thread
    try:
        frozen = accepted_run(
            save_path=str(output),
            source_spec=directory_source(raw, ext="nxs", recursive=True),
            gi=gi_intent(enabled=True, incidence_motor="halpha"),
        )
        admitted_worker(wrapper, frozen=frozen)
        admitted_worker(worker, frozen=frozen)

        published = []
        worker.sigGISourceMotors.connect(published.append)

        _write_bluesky_nxwriter(raw / "day1" / "scan_0001.nxs")
        _write_bluesky_nxwriter(raw / "day1" / "scan_0002.nxs")

        _drive_run_route(worker, frozen, limit=24)
        app.processEvents()

        assert len(published) == 1, (
            "two containers in ONE run published twice; "
            f"{[value.source_path for value in published]}")
        assert published[0].run_configuration is frozen
        assert isinstance(published[0].motors, tuple)
        assert isinstance(published[0].counters, tuple)
    finally:
        worker._eiger_close_master()
        wrapper.close()
        wrapper.deleteLater()
        app.processEvents()


# --------------------------------------------------------------------------- #
# D3/D4 — the provisional recheck is BOUNDED, rebuilds the reader, and never
# retires a container it did not consume.
# --------------------------------------------------------------------------- #

def _batch_dir_thread(watch_dir, out_dir):
    """A real ``imageThread`` over a directory of .nxs containers, BATCH mode."""
    from queue import Queue

    from tests.xdart._accepted_run import admitted_worker, directory_source
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )
    from xdart.modules.live import LiveScan
    from xrd_tools.core.containers import PONI

    scan = LiveScan("scan", data_file=str(out_dir / "scan.nxs"), static=True)
    worker = imageThread(
        Queue(), {}, threading.RLock(), "",
        str(out_dir), "scan", False,
        PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10),
        "Image Directory", "", str(watch_dir), False, "nxs", False, None, "",
        None, "Full", "None", "", "", None, "", "", 1.0, None,
        False, None, 1, 0.0, "q_total", "qip_qoop", "start", scan,
        live_mode=False, max_cores=1,
    )
    return admitted_worker(
        worker,
        batch_mode=True,
        save_path=str(out_dir),
        source_spec=directory_source(watch_dir, ext="nxs"))


def _unfinalized(path, *, n=2):
    import h5py

    _write_bluesky_nxwriter(path, n=n)
    with h5py.File(path, "r+") as handle:
        del handle["entry/end_time"]
    return path


def _drain(worker, frozen, *, limit=24, timeout=30.0):
    """Consume ONE generation to end-of-stream, under a deterministic watchdog.

    The consumer runs on its own thread with a bounded join, because the
    failure this must report -- a reader that spins on an unchanged provisional
    file -- would otherwise WEDGE the gate instead of failing it.  A wedged gate
    is not a red.
    """
    delivered = []
    done = threading.Event()

    def consume():
        try:
            for _ in range(limit):
                item = worker.get_next_image(frozen)
                delivered.append(item)
                if item[3] is None:
                    return
        finally:
            done.set()

    consumer = threading.Thread(target=consume, name="o1b-r4a-drain",
                                daemon=True)
    consumer.start()
    finished = done.wait(timeout)
    if not finished:
        worker.command = 'stop'
        stop_evt = getattr(worker, '_prefetch_stop_evt', None)
        if stop_evt is not None:
            stop_evt.set()
        consumer.join(timeout=10.0)
        raise AssertionError(
            "the directory reader never terminated: it is spinning on a "
            f"container that never finalized (delivered={len(delivered)})")
    consumer.join(timeout=10.0)
    assert delivered and delivered[-1][3] is None, (
        f"one generation must end with exactly one EOS; got {delivered!r}")
    return delivered


def test_permanently_provisional_container_finishes_without_spinning(tmp_path):
    """D4.  A container that never finalizes must end the run TRUTHFULLY.

    The recheck is bounded at ONE, so the sweep neither spins on an unchanged
    file nor reopens it without limit -- and because the container was never
    consumed it is never recorded as retired, so the next Run picks it up.
    """
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    growing = _unfinalized(watch / "grow_00001.nxs")

    worker = _batch_dir_thread(watch, out)
    frozen = worker.run_configuration

    opens = []
    real_open = worker._eiger_open_master

    def counting_open(frozen_arg, path):
        opens.append(str(path))
        return real_open(frozen_arg, path)

    worker._eiger_open_master = counting_open
    try:
        delivered = _drain(worker, frozen)
    finally:
        worker._prefetch_stop_evt.set()
        worker._eiger_close_master()

    labels = [item[2] for item in delivered if item[3] is not None]
    assert labels == [], f"an unfinalized container produced frames: {labels}"
    assert opens == [str(growing), str(growing)], (
        "the container must be opened exactly twice -- once in the sweep and "
        f"once for its single bounded recheck; got {len(opens)} opens")
    assert str(growing) not in worker._eiger_done_masters, (
        "a container that was never consumed was recorded as retired, so the "
        "next Run would skip it")
    assert imageThread._eiger_provisional_hold(worker, str(growing)) is True


def test_end_of_sweep_recheck_rebuilds_the_reader(tmp_path):
    """The recheck must go back through the normal open/adapter binding.

    A cheap frame-count re-read would look cheaper and then read frames with the
    stale binding that saw the container provisional -- which is exactly the
    ``HDF5 file does not contain an Eiger-like structure`` failure the parent
    produced.  Proving a fresh cursor was BOUND is what makes the delivered
    frames attributable to a rebuilt reader.
    """
    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    growing = _unfinalized(watch / "grow_00001.nxs")

    worker = _batch_dir_thread(watch, out)
    frozen = worker.run_configuration

    binds = []
    real_bind = worker._eiger_bind_cursor

    def counting_bind(cursor):
        binds.append(cursor)
        return real_bind(cursor)

    worker._eiger_bind_cursor = counting_bind

    real_open = worker._eiger_open_master

    def finalizing_open(frozen_arg, path):
        result = real_open(frozen_arg, path)
        if getattr(worker, "_eiger_open_state", None) == "not ready":
            # The acquisition completes between the sweep and the recheck.
            _write_bluesky_nxwriter(growing, n=5)
        return result

    worker._eiger_open_master = finalizing_open
    try:
        delivered = _drain(worker, frozen)
    finally:
        worker._prefetch_stop_evt.set()
        worker._eiger_close_master()

    labels = [item[2] for item in delivered if item[3] is not None]
    assert labels == [1, 2, 3, 4, 5], labels
    assert len(binds) == 1, (
        "the finalized container was read without binding a fresh cursor; "
        f"{len(binds)} bindings")
    assert delivered[-1][3] is None, "end-of-stream must come last"
    assert str(growing) in worker._eiger_done_masters, (
        "a fully drained container must be retired")


def test_jit_delivery_after_teardown_is_inert(tmp_path):
    """A worker discovery that arrives after teardown must be INERT.

    The hydration chain's existing inertness comes from invalidating outstanding
    TOKENS, and a value delivery carries its own identity instead of a token --
    so without an explicit closed state it would sail past teardown and touch a
    wrangler the host has already torn down.
    """
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        GISourceMotorDiscovery,
    )
    from tests.xdart._accepted_run import (
        accepted_run,
        admitted_worker,
        directory_source,
        gi_intent,
    )

    app, wrapper, raw, output = _jit_rig(tmp_path)
    worker = wrapper.thread
    try:
        frozen = accepted_run(
            save_path=str(output),
            source_spec=directory_source(raw, ext="nxs", recursive=True),
            gi=gi_intent(enabled=True, incidence_motor="halpha"),
        )
        admitted_worker(wrapper, frozen=frozen)
        admitted_worker(worker, frozen=frozen)

        emitted = []
        wrapper.sigGIMotorOptions.connect(emitted.append)

        # The production teardown owner, not a private poke.
        wrapper._invalidate_gi_hydration_requests()

        wrapper._on_gi_source_motors(GISourceMotorDiscovery(
            run_configuration=frozen,
            source_path=str(raw / "day1" / "scan_0001.nxs"),
            motors=("hy",), counters=("i0",)))
        app.processEvents()

        assert emitted == [], "a post-teardown delivery was applied"
        assert _motor_choices(wrapper.parameters) == ["Manual"]
    finally:
        wrapper.close()
        wrapper.deleteLater()
        app.processEvents()


# --------------------------------------------------------------------------- #
# R4A-III (review §52.2) — the 0.75 s deadline bounds NAME ENUMERATION too, not
# only content opens.
# --------------------------------------------------------------------------- #

class _CountingEntry:
    """A ``DirEntry`` proxy that records CLASSIFICATION of a listed name.

    ``is_file``/``is_dir`` are the first thing done to an entry once a listing
    is accepted, so counting them separates "the deadline stopped enumeration"
    from "the deadline stopped enumeration and then the partial head was
    classified and sorted anyway".
    """

    def __init__(self, inner, on_classify):
        self._inner = inner
        self._on_classify = on_classify

    @property
    def name(self):
        return self._inner.name

    @property
    def path(self):
        return self._inner.path

    def is_file(self, *args, **kwargs):
        self._on_classify(self._inner.name)
        return self._inner.is_file(*args, **kwargs)

    def is_dir(self, *args, **kwargs):
        self._on_classify(self._inner.name)
        return self._inner.is_dir(*args, **kwargs)


class _TickingScandir:
    """A real ``os.scandir`` result whose iteration advances an injected clock.

    The entries are REAL ``DirEntry`` objects from a REAL directory -- only the
    passage of time is synthetic, so this pins the production consumption loop
    rather than a stubbed listing.
    """

    def __init__(self, inner, on_yield, wrap=None):
        self._inner = inner
        self._on_yield = on_yield
        self._wrap = wrap

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def __iter__(self):
        for entry in self._inner:
            self._on_yield(entry)
            yield self._wrap(entry) if self._wrap is not None else entry


def test_preview_deadline_bounds_name_enumeration_not_only_content_opens(
        tmp_path, monkeypatch):
    """R4A-III.  Bounding only content opens left the user-facing freeze open.

    ``_directory_preview_entries`` used to drain the whole ``scandir`` iterator
    and sort it before any budget check, so a wide accidental root could spend
    arbitrarily longer than 0.75 s enumerating names with ZERO opens -- the
    eight-open cap never fires on that path, and the existing fake-clock row
    only delays ``_read_bluesky_source_columns``, so neither could see it.

    Here the clock advances per ENTRY YIELDED and nothing else moves it.  The
    directory holds a genuinely usable container, so an implementation that
    inspected the partial head would find ``hy`` and fill the dropdown; a
    correct one abandons truthfully.
    """
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler as iw

    root = tmp_path / "wide"
    root.mkdir()
    _write_bluesky_nxwriter(root / "aaa_usable.nxs")
    for index in range(200):
        (root / f"scan_{index:04d}.nxs").touch()
    # A subdirectory the descent would list if it ever got that far.
    (root / "day1").mkdir()
    _write_bluesky_nxwriter(root / "day1" / "nested_0001.nxs")

    clock = {"now": 1000.0}
    monkeypatch.setattr(iw.time, "monotonic", lambda: clock["now"])

    yielded = []
    listed = []
    classified = []
    real_scandir = iw.os.scandir

    def ticking_scandir(path):
        inner = real_scandir(path)
        if Path(path) == root or Path(path).parent == root:
            listed.append(str(path))

            def tick(entry):
                yielded.append(entry.name)
                # 0.25 is exactly representable, so three entries reach the
                # 0.75 s deadline with no floating-point drift to argue about.
                clock["now"] += 0.25

            return _TickingScandir(
                inner, tick,
                wrap=lambda entry: _CountingEntry(entry, classified.append))
        return inner

    monkeypatch.setattr(iw.os, "scandir", ticking_scandir)

    host, params = _holder(root, recursive=True)
    opens = []
    real_reader = host._read_bluesky_source_columns
    host._read_bluesky_source_columns = (
        lambda path: opens.append(str(path)) or real_reader(path))

    host.get_img_fname()

    elapsed = clock["now"] - 1000.0
    # The deadline really expired, and it expired DURING enumeration.
    assert elapsed >= 0.75, (
        f"the injected clock never reached the deadline: {elapsed}")
    assert len(yielded) == 3, (
        "a 0.75 s deadline with 0.25 s per entry must stop consuming after "
        f"exactly three entries, not {len(yielded)}")
    # No complete materialization: the directory is far wider than what was read.
    assert len(yielded) < 201, (
        f"the whole directory was materialized before any budget check: "
        f"{len(yielded)} entries")
    # The partial head is ABANDONED, not classified and sorted anyway.  A
    # partial listing is not a globally ordered candidate set, so treating it
    # as one would silently reintroduce the candidates[0]-order bug even when
    # the shared budget happens to block the opens that would follow.
    assert classified == [], (
        f"the partial listing was classified after the deadline: {classified}")
    # No content open, and therefore no partial, non-globally-ordered inspection.
    assert opens == [], (
        f"a candidate was opened after the deadline expired: {opens}")
    # No recursive descent past expiry -- only the root was ever listed.
    assert listed == [str(root)], (
        f"the descent listed further directories after expiry: {listed}")
    # Truthful projection: UNKNOWN, Manual, and the real operator hint.
    assert _motor_choices(params) == ["Manual"], _motor_choices(params)
    assert host._gi_motor_knowledge_proved is False, (
        "an abandoned preview must leave motor knowledge UNKNOWN, not "
        "known-empty")
    hints = [values[0] for values in host.showLabel.emissions if values]
    assert any("motor" in str(hint).lower() for hint in hints), (
        f"the operator got no hint that motors will fill during Run; {hints}")
