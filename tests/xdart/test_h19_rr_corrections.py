"""Production-wired regression tests for the two H19 review corrections.

RR-1 — empty-directory Live arming must be reachable through the REAL Live and
Run buttons.  The wrangler input gate and the Controls V2 readiness gate must
both permit the arm-then-acquire workflow: watch an empty directory, select
Live, click Run, and let the first container land afterward.

RR-2 — the run-end Source-card handoff clear must be exception-safe.  It used to
be the final statement of ``wrangler_finished``; a raise anywhere in the run-end
tail leaked the frozen plan/session into the next between-runs ``setup()``
seeding.  The clear now runs in a ``finally``; the test raises in the tail and
proves both wrangler and worker-thread references are dropped and that a later
seed cannot inherit the stale plan.
"""

import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets


def _wait_until(qapp, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _drain_qt_events_after_test(qapp):
    yield
    for _ in range(3):
        qapp.processEvents()


def _poni():
    from xrd_tools.core.containers import PONI

    return PONI(
        dist=0.1794,
        poni1=0.0,
        poni2=0.0,
        detector="RayonixMx225",
        wavelength=0.7293e-10,
    )


def _install_status_spy(monkeypatch):
    """Capture every ``_safe_status_text`` message so a test can prove the
    "Choose an image source to run" rejection did (or did not) fire."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    recorded = []
    original = imageWrangler._safe_status_text

    def _spy(obj, text):
        recorded.append("" if text is None else str(text))
        try:
            original(obj, text)
        except Exception:
            pass

    monkeypatch.setattr(imageWrangler, "_safe_status_text", staticmethod(_spy))
    return recorded


def _make_widget(monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    return staticWidget()


def _configure_container_directory(widget, directory, *, ext="nxs"):
    signal = widget.wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(directory))
    signal.child("img_ext").setValue(ext)


# --------------------------------------------------------------------------- #
# RR-1
# --------------------------------------------------------------------------- #

def test_rr1_empty_live_directory_arms_through_real_run_button(
    qapp, monkeypatch, tmp_path,
):
    """The decisive RR-1 test: a genuinely empty watched directory arms a Live
    run through the real Run-button path, and the SAME armed run then discovers
    and consumes the first container that lands — with no second Run click.

    Fail-before (``fa7f453a``): ``_inputs_valid`` rejects ``img_file == ''`` with
    "Choose an image source to run", ``start()`` returns early, and
    ``source_run_plan`` stays ``None`` (never arms).
    """
    from xrd_tools.core.scan import SourceKind
    from xrd_tools.sources.directory_index import DirectoryIndex
    from xrd_tools.sources.probe import ProbeResult, ProbeState

    def _probe_ready(_index, candidate):
        return ProbeResult(ProbeState.READY, kind=SourceKind.NEXUS_STACK)

    monkeypatch.setattr(DirectoryIndex, "probe_candidate", _probe_ready)

    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()  # genuinely EMPTY at Run time

    widget = _make_widget(monkeypatch, tmp_path)
    started = []
    recorded = _install_status_spy(monkeypatch)
    try:
        _configure_container_directory(widget, watch_dir)
        # The Source card mounts and produces its (empty) observation.
        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_current_directory_observation() is not None,
        )
        observation = widget._controls_v2_current_directory_observation()
        assert observation.discovered_snapshot.candidates == ()

        # Zero READY candidates must not disable the Live affordance.  Run is
        # still gated until Live is selected, avoiding a circular state where
        # the operator cannot make the empty directory runnable.
        assert widget.controls.actionRow.isEnabled() is True
        assert widget.controls.liveButton.isEnabled() is True
        assert widget.controls.startButton.isEnabled() is False
        assert "Enable Live" in widget.controls.readinessLabel.text()

        poni = _poni()
        widget.wrangler.poni = poni
        widget.scan._cached_poni = poni
        widget.wrangler.project_folder = str(tmp_path)
        widget.wrangler.h5_dir = str(tmp_path)
        widget._refresh_controls_v2_profile(immediate=True)

        widget.controls.liveButton.click()
        assert _wait_until(qapp, widget.controls.startButton.isEnabled)
        assert widget.wrangler.live_mode is True

        # The empty directory leaves the between-runs seed at img_file=''.
        widget.wrangler.img_file = ""
        # Stub the worker so start() arms without launching the watch loop.
        monkeypatch.setattr(
            widget.wrangler.thread, "start", lambda: started.append(True))

        # Precondition: the config IS the authoritative directory path, so the
        # only thing standing between an empty img_file and a run is the gate
        # RR-1 fixes.  ``_inputs_valid`` is the exact gate — it returned False on
        # ``img_file == ''`` before the fix (the clean fail-before boundary).
        assert widget._controls_v2_container_index_config() is not None
        assert widget.wrangler._inputs_valid() is True

        # Drive the REAL Qt Run button (not wrangler.start/start_wrangler).
        widget.controls.startButton.click()
        qapp.processEvents()

        # Armed: reached start_wrangler -> thread.start (stub), no rejection.
        assert started == [True]
        assert not any("Choose an image source" in m for m in recorded)
        plan = widget.wrangler.source_run_plan
        assert plan is not None
        assert plan.paths == ()  # honest empty baseline
        assert widget.wrangler.img_file == ""
        # The frozen handoff propagated to the worker thread.
        assert widget.wrangler.thread.source_run_plan is plan
        assert (widget.wrangler.thread.source_index_session
                is widget.wrangler.source_index_session)
        assert widget.wrangler.thread._h19_live_directory_armed() is True
        # The gate helper agrees the empty source is an armed authoritative run.
        assert widget.wrangler._h19_empty_directory_live_run_ok() is True

        # --- add the first candidate AFTER arming ---
        (watch_dir / "scan_0.nxs").write_bytes(b"frame")

        # The SAME armed run's worker discovers + consumes it — no new Run click,
        # no re-freeze: one refill poll against the frozen plan + shared session.
        widget.wrangler.thread._eiger_refill_master_queue()
        queued = list(widget.wrangler.thread._eiger_master_queue)
        assert any(str(item).endswith("scan_0.nxs") for item in queued), queued
    finally:
        widget._exit_run_state()
        widget.close()
        widget.deleteLater()


def test_rr1_empty_non_live_directory_does_not_start(qapp, monkeypatch, tmp_path):
    """Negative: an empty container directory in BATCH (non-Live) mode is still
    rejected — the exception is Live-only.  The config is otherwise eligible, so
    Live-off is the sole reason for rejection."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()

    widget = _make_widget(monkeypatch, tmp_path)
    started = []
    recorded = _install_status_spy(monkeypatch)
    try:
        _configure_container_directory(widget, watch_dir)
        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_current_directory_observation() is not None,
        )
        # Live NOT enabled (default off).
        assert widget.wrangler.live_mode is False
        widget.wrangler.poni = _poni()
        widget.wrangler.img_file = ""
        monkeypatch.setattr(
            widget.wrangler.thread, "start", lambda: started.append(True))

        # Config is eligible, but the Live gate blocks the empty source.
        assert widget._controls_v2_container_index_config() is not None
        assert widget.wrangler._h19_empty_directory_live_run_ok() is False
        assert widget.wrangler._inputs_valid() is False

        widget.wrangler.start()
        assert started == []
        assert widget.wrangler.source_run_plan is None
        assert any("Choose an image source" in m for m in recorded)
    finally:
        widget.close()
        widget.deleteLater()


def test_rr1_nonexistent_directory_does_not_start(qapp, monkeypatch, tmp_path):
    """Negative: a Live Image-Directory run whose configured directory does not
    exist is rejected — the exception requires a real directory on disk."""
    missing_dir = tmp_path / "does_not_exist"  # never created

    widget = _make_widget(monkeypatch, tmp_path)
    started = []
    recorded = _install_status_spy(monkeypatch)
    try:
        _configure_container_directory(widget, missing_dir)
        widget.controls.liveButton.setChecked(True)
        assert widget.wrangler.live_mode is True
        widget.wrangler.poni = _poni()
        widget.wrangler.img_file = ""
        monkeypatch.setattr(
            widget.wrangler.thread, "start", lambda: started.append(True))

        assert widget.wrangler._h19_empty_directory_live_run_ok() is False
        assert widget.wrangler._inputs_valid() is False

        widget.wrangler.start()
        assert started == []
        assert widget.wrangler.source_run_plan is None
        assert any("Choose an image source" in m for m in recorded)
    finally:
        widget.close()
        widget.deleteLater()


def test_rr1_empty_single_image_source_does_not_inherit_exception(
    qapp, monkeypatch, tmp_path,
):
    """Negative: a non-directory source (Single Image) with an empty file and
    Live enabled does NOT inherit the directory exception — the empty-source
    rejection still fires."""
    widget = _make_widget(monkeypatch, tmp_path)
    started = []
    recorded = _install_status_spy(monkeypatch)
    try:
        signal = widget.wrangler.parameters.child("Signal")
        signal.child("inp_type").setValue("Single Image")
        widget.controls.liveButton.setChecked(True)
        assert widget.wrangler.live_mode is True
        widget.wrangler.poni = _poni()
        widget.wrangler.img_file = ""
        monkeypatch.setattr(
            widget.wrangler.thread, "start", lambda: started.append(True))

        assert widget.wrangler._h19_empty_directory_live_run_ok() is False
        assert widget.wrangler._inputs_valid() is False

        widget.wrangler.start()
        assert started == []
        assert widget.wrangler.source_run_plan is None
        assert any("Choose an image source" in m for m in recorded)
    finally:
        widget.close()
        widget.deleteLater()


# --------------------------------------------------------------------------- #
# RR-2
# --------------------------------------------------------------------------- #

def test_rr2_run_end_source_authority_cleared_when_tail_raises(
    qapp, monkeypatch, tmp_path,
):
    """RR-2: a raise inside the run-end tail must NOT leak the frozen Source-card
    handoff.  The clear now runs in a ``finally``; afterwards both wrangler and
    worker-thread plan/session are dropped and a later between-runs seed cannot
    inherit the stale plan.

    Fail-before (``fa7f453a``): the clear is the final statement of
    ``wrangler_finished``, so the raise skips it — ``source_run_plan`` survives
    on both layers and ``get_img_fname`` seeds ``img_file`` from the stale
    frozen candidate.
    """
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter

    # A real container the stale plan would seed from (proves the leak
    # behaviorally, not just structurally).
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    stale_file = stale_dir / "scan_00000.nxs"
    _write_bluesky_nxwriter(stale_file, n=2)

    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()  # empty: the legacy walk finds nothing -> img_file=''

    widget = _make_widget(monkeypatch, tmp_path)
    try:
        wrangler = widget.wrangler

        # Armed run-end state: the frozen handoff lives on BOTH layers.
        plan = SimpleNamespace(paths=(str(stale_file),))
        session = SimpleNamespace()
        wrangler.source_run_plan = plan
        wrangler.source_index_session = session
        wrangler.thread.source_run_plan = plan
        wrangler.thread.source_index_session = session

        # Configure the between-runs seed to walk the EMPTY watched directory.
        _configure_container_directory(widget, watch_dir)
        wrangler.inp_type = "Image Directory"
        wrangler.poni = _poni()
        wrangler.img_file = ""
        wrangler.file_filter = ""

        # Deliberately raise in an early run-end finalization step.
        boom = RuntimeError("run-end tail failure (test)")

        def _raise(*args, **kwargs):
            raise boom

        monkeypatch.setattr(widget, "_exit_run_state", _raise)

        with pytest.raises(RuntimeError):
            widget.wrangler_finished()

        # The finally cleared BOTH layers despite the raise.
        assert wrangler.source_run_plan is None
        assert wrangler.source_index_session is None
        assert wrangler.thread.source_run_plan is None
        assert wrangler.thread.source_index_session is None

        # A later between-runs seed cannot inherit the frozen (stale) plan:
        # with the plan cleared, get_img_fname falls to the legacy walk of the
        # empty directory and leaves img_file empty rather than seeding the
        # stale candidate.
        wrangler.get_img_fname()
        assert wrangler.img_file != str(stale_file)
        assert wrangler.img_file == ""
    finally:
        widget.close()
        widget.deleteLater()
