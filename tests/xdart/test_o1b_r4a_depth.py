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
