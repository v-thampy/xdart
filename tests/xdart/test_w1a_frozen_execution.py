"""O-1a-W1A — the image execution path consumes ONE frozen configuration.

Frozen acceptance oracle for W-1.2 cases 1, 2, 3, 4, 5 and 11 (the W-1 ratified
immediate-start packet).  Written BEFORE any production edit and red at the
accepted parent ``e754c379``.

Everything here is production-wired (CLAUDE.md rule 2): the real ``staticWidget``
with its real ``imageWrangler``, the real ``imageThread`` that wrangler built, the
real Controls transaction/freeze owner, a real ``PONI``, the real
``initialize_scan`` writer path and a real NeXus round trip.  No ``__new__``
worker, no dummy scan, no identity rebuilt from equal values.

The discriminator throughout is POISON: after the production Run click has frozen
and published one configuration, the mutable display scan (and the panel-derived
thread mirrors that ``imageWrangler.setup()`` re-reads) are set to values the run
must NOT use.  A run that still agrees with the frozen object read no display
state; a run that follows the poison did.

Case map (W-1.2):

1  poisoned display scan -> ``initialize_scan()`` builds the worker scan
   entirely from the accepted frozen configuration
2  the Append comparison uses ``frozen.processing_mapping()`` -- at the
   pre-read cursor AND at the in-run comparison
3  the reduction-plan builder receives the exact frozen object and reads no
   display/GUI state
4  the per-run scan carries the exact generation/fingerprint and a detached
   JSON-native provenance projection that survives write and reload
5  wrapper and worker refuse absent / foreign / stale configuration before any
   command, button, source, output or worker side effect
11 backward GI display writes stay, but cannot reach the frozen object or the
   run
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------- #
# Fixtures — the real widget, the real wrangler, the real worker thread.
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


class _Run:
    """The state a production Run click leaves behind, with the worker blocked."""

    def __init__(self, widget, thread, frozen, out_dir, started):
        self.widget = widget
        self.wrangler = widget.wrangler
        self.thread = thread
        self.frozen = frozen
        self.out_dir = out_dir
        self.started = started

    @property
    def output_path(self):
        return Path(self.out_dir) / f"{self.thread.scan_name}.nxs"


def _click_run(widget, tmp_path, monkeypatch, *, write_mode="Overwrite",
               points=321, gi=False, scan_name="scan"):
    """Drive the REAL production Run click; block only ``thread.start()``.

    ``imageWrangler.start()`` refuses/validates/freezes/publishes and emits
    ``sigStart``, which is connected to ``staticWidget.start_wrangler`` -- so
    ``_apply_controls_v2_run_state`` (adoption) and ``imageWrangler.setup()``
    (which re-reads the panel into the thread mirrors) both run for real.
    """
    poni_path = _write_poni(tmp_path / "cal.poni")
    raw_path = tmp_path / f"{scan_name}_0001.tif"
    raw_path.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)

    widget._set_poni_field(poni_path)
    widget.wrangler.parameters.child("Project", "h5_dir").setValue(str(out))
    widget.wrangler.img_file = str(raw_path)
    widget.controls.set_write_mode(write_mode)
    if gi:
        widget.wrangler.parameters.child("GI", "Grazing").setValue(True)
    widget._on_controls_v2_field_changed(("Int1D", "points"), points)

    started = []
    monkeypatch.setattr(widget.wrangler.thread, "start",
                        lambda: started.append(True))
    widget.wrangler.start()
    assert started == [True], "the production Run click did not reach the worker"

    thread = widget.wrangler.thread
    frozen = widget.wrangler.run_configuration
    assert frozen is not None, "the Run click published no frozen configuration"
    assert thread.run_configuration is frozen
    thread.h5_dir = str(out)
    thread.scan_name = scan_name
    thread.fname = str(out / f"{scan_name}.nxs")
    return _Run(widget, thread, frozen, out, started)


def _poison_display_state(run, *, unit="POISON", numpoints=7):
    """Make every mutable display/panel surface disagree with the frozen run.

    This is the whole discriminator: after the freeze, the display scan is still
    the browser's scan and can be replaced, reloaded or edited at any moment, and
    ``imageWrangler.setup()`` has already re-read the panel into the thread
    mirrors.  Nothing here may reach the run.
    """
    scan = run.widget.scan
    frozen = run.frozen
    scan.bai_1d_args = {"unit": unit, "numpoints": numpoints}
    scan.bai_2d_args = {"unit": unit, "npt_rad": numpoints, "npt_azim": numpoints}
    scan.gi_config = {"gi_mode_1d": "POISON_MODE"}
    scan.gi = not bool(frozen.gi.enabled)
    scan.skip_2d = not bool(frozen.skip_2d)
    scan.apply_threshold = not bool(frozen.threshold.apply_threshold)
    scan.threshold_min = -12345.0
    scan.threshold_max = -12344.0
    scan.mask_sentinel = not bool(frozen.threshold.mask_saturation)
    # the panel-derived thread mirrors imageWrangler.setup() re-read AFTER the
    # frozen projection was applied
    thread = run.thread
    thread.scan_args = {"bai_1d_args": {"unit": unit, "numpoints": numpoints},
                        "bai_2d_args": {"unit": unit}}
    thread.gi = not bool(frozen.gi.enabled)
    thread.incidence_motor = "POISON_MOTOR"
    thread.apply_threshold = not bool(frozen.threshold.apply_threshold)
    thread.threshold_min = -12345.0
    thread.threshold_max = -12344.0
    thread.mask_sentinel = not bool(frozen.threshold.mask_saturation)


def _write_processed_target(path, config, *, labels=(1, 2), labels_2d=None):
    """A real processed .nxs whose stored reduction config is ``config``."""
    import h5py
    from xrd_tools.core.provenance import write_provenance

    labels = np.asarray(labels, dtype=np.int64)
    q = np.linspace(0.1, 1.0, 4, dtype=np.float32)
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        g1 = entry.create_group("integrated_1d")
        g1.attrs["NX_class"] = "NXdata"
        g1.attrs["signal"] = "intensity"
        g1.attrs["axes"] = ["frame_index", "q"]
        g1.create_dataset("frame_index", data=labels)
        qds = g1.create_dataset("q", data=q)
        qds.attrs["units"] = "q_A^-1"
        g1.create_dataset("intensity", data=np.zeros(
            (labels.size, q.size), dtype=np.float32))
        if labels_2d is not None:
            g2 = entry.create_group("integrated_2d")
            g2.create_dataset("frame_index",
                              data=np.asarray(labels_2d, dtype=np.int64))
        write_provenance(h5, config=config, host="")
    return path


# --------------------------------------------------------------------------- #
# Case 1 — the per-run scan is built entirely from the frozen configuration.
# --------------------------------------------------------------------------- #

def test_poisoned_display_scan_does_not_reach_the_per_run_scan(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 1.  ``initialize_scan()`` builds the worker ``LiveScan`` from
    ``frozen.scan_kwargs()``; the poisoned display scan and the panel-derived
    thread mirrors reach none of it."""
    run = _click_run(widget, tmp_path, monkeypatch, points=321)
    _poison_display_state(run)
    frozen = run.frozen

    scan = run.thread.initialize_scan()

    assert scan is not widget.scan, "the run must not execute on the display scan"
    assert scan.bai_1d_args == frozen.bai_1d_args
    assert scan.bai_2d_args == frozen.bai_2d_args
    assert bool(scan.skip_2d) == bool(frozen.skip_2d)
    assert bool(scan.gi) == bool(frozen.gi.enabled)
    assert scan.incidence_motor == frozen.gi.scan_incidence_motor
    assert bool(scan.apply_threshold) == bool(frozen.threshold.apply_threshold)
    assert scan.threshold_min == frozen.threshold.threshold_min
    assert scan.threshold_max == frozen.threshold.threshold_max
    assert bool(scan.mask_sentinel) == bool(frozen.threshold.mask_saturation)
    assert scan.bai_1d_args["numpoints"] == 321


def test_run_scan_integration_args_are_not_aliased_to_the_frozen_value(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 1 (deep-immutability half).  The worker gets a fresh mutable
    copy; mutating the run scan can never write back into the frozen object."""
    run = _click_run(widget, tmp_path, monkeypatch, points=222)
    scan = run.thread.initialize_scan()

    scan.bai_1d_args["numpoints"] = 999
    assert run.frozen.bai_1d_args["numpoints"] == 222
    assert run.thread.run_configuration.bai_1d_args["numpoints"] == 222


# --------------------------------------------------------------------------- #
# Case 2 — the Append comparison compares against the frozen mapping.
# --------------------------------------------------------------------------- #

def test_append_cursor_compares_the_frozen_mapping_not_the_display_scan(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 2 (pre-read cursor).  A processed target stored with exactly
    ``frozen.processing_mapping()`` must be accepted even though the display scan
    now says something else."""
    run = _click_run(widget, tmp_path, monkeypatch,
                     write_mode="Append", points=321)
    _write_processed_target(run.output_path, run.frozen.processing_mapping(),
                            labels=(1, 2), labels_2d=(1, 2))
    _poison_display_state(run)

    existing = run.thread._load_append_skip_snapshot(run.thread.scan_name)

    assert existing == {1, 2}


def test_append_cursor_refuses_when_the_frozen_mapping_differs(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 2 (the other polarity).  A target stored with the POISONED
    display configuration must MISMATCH the frozen run -- proving the comparison
    is not merely tolerant."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        AppendConfigMismatchError,
    )

    run = _click_run(widget, tmp_path, monkeypatch,
                     write_mode="Append", points=321)
    poisoned_config = {
        "bai_1d_args": {"unit": "POISON", "numpoints": 7},
        "bai_2d_args": {"unit": "POISON"},
        "gi": False,
        "gi_config": {},
    }
    _write_processed_target(run.output_path, poisoned_config)
    _poison_display_state(run)

    with pytest.raises(AppendConfigMismatchError):
        run.thread._load_append_skip_snapshot(run.thread.scan_name)


def test_append_cursor_require_2d_follows_the_frozen_skip_2d(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 2 (``require_2d``).  The cursor's completion rule is the frozen
    run's ``skip_2d``, not the display scan's."""
    run = _click_run(widget, tmp_path, monkeypatch,
                     write_mode="Append", points=321)
    assert bool(run.frozen.skip_2d) is False, "Int 2D freezes skip_2d False"
    _write_processed_target(
        run.output_path, run.frozen.processing_mapping(),
        labels=(1, 2), labels_2d=(1,))
    # the display scan claims 1D-only, which would call frame 2 complete
    _poison_display_state(run)
    widget.scan.skip_2d = True

    existing = run.thread._load_append_skip_snapshot(run.thread.scan_name)

    assert existing == {1}, "require_2d must come from the frozen configuration"


def test_in_run_append_comparison_uses_the_frozen_mapping(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 2 (in-run comparison inside ``initialize_scan``).  The mid-run
    guard compares the processed record against the frozen run."""
    run = _click_run(widget, tmp_path, monkeypatch,
                     write_mode="Append", points=321)
    _write_processed_target(run.output_path, run.frozen.processing_mapping())
    _poison_display_state(run)

    scan = run.thread.initialize_scan()

    assert scan.bai_1d_args == run.frozen.bai_1d_args


# --------------------------------------------------------------------------- #
# Case 3 — the plan builder receives the exact frozen object.
# --------------------------------------------------------------------------- #

def test_run_plan_builder_carries_the_exact_frozen_object(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 3.  The run's plan builder -- the production object installed on
    the worker's plan cache -- carries the EXACT accepted configuration (``is``),
    and the plan it produces follows that object even after the per-run scan's
    own dicts are mutated."""
    run = _click_run(widget, tmp_path, monkeypatch, points=321)
    _poison_display_state(run)
    scan = run.thread.initialize_scan()

    builder = run.thread._plan_cache.plan_builder
    assert builder is not None, "no run plan builder was installed for the run"
    assert getattr(builder, "run_configuration", None) is run.frozen, (
        "the run plan builder does not carry the exact frozen object")

    scan.bai_1d_args["numpoints"] = 11
    plan = run.thread._plan_cache.get(scan, integrate_2d=not run.frozen.skip_2d)

    assert plan.integration_1d.npt == run.frozen.bai_1d_args["numpoints"]
    assert (plan.integration_2d is not None) is (not run.frozen.skip_2d)
    assert (plan.gi is not None) is bool(run.frozen.gi.enabled)


def test_run_plan_configuration_never_reads_the_live_controls_snapshot(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 3 (no GUI snapshot/read).  Re-configuring the run plan while
    the accepted run is live must not consult the live Controls snapshot: the
    frozen configuration is the only authority once a run is admitted."""
    run = _click_run(widget, tmp_path, monkeypatch, points=321)
    assert widget._run_active is True

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "the run plan re-read the live Controls snapshot")

    monkeypatch.setattr(widget, "_controls_v2_native_int_snapshot", _forbidden)

    widget._configure_controls_v2_native_run_plan()

    builder = run.thread._plan_cache.plan_builder
    assert getattr(builder, "run_configuration", None) is run.frozen


# --------------------------------------------------------------------------- #
# Case 4 — durable identity and detached provenance.
# --------------------------------------------------------------------------- #

def test_per_run_scan_carries_the_exact_identity_and_a_detached_projection(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 4 (in memory).  Exact ``(generation, fingerprint)``, the exact
    frozen object, and a JSON-native provenance mapping that is DETACHED (a
    fresh mapping whose mutation cannot reach the frozen run)."""
    run = _click_run(widget, tmp_path, monkeypatch, points=321)
    frozen = run.frozen

    scan = run.thread.initialize_scan()

    assert scan.run_configuration is frozen
    assert int(scan.run_configuration_generation) == int(frozen.generation)
    assert scan.run_configuration_fingerprint == frozen.fingerprint
    provenance = scan.run_configuration_provenance
    assert isinstance(provenance, dict)
    assert provenance == frozen.as_provenance()
    assert json.loads(json.dumps(provenance)) == provenance, (
        "the writer projection must be JSON-native before the output opens")
    provenance["fingerprint"] = "TAMPERED"
    assert frozen.fingerprint != "TAMPERED"
    assert run.thread.run_configuration.fingerprint == frozen.fingerprint


def test_run_identity_survives_the_write_and_a_real_reload(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 4 (durable half).  The projection is attached BEFORE the output
    is opened, so it is already in the FIRST written file, and a real reload
    returns the same ``(generation, fingerprint)``."""
    from xrd_tools.core.provenance import read_provenance

    run = _click_run(widget, tmp_path, monkeypatch, points=321)
    frozen = run.frozen

    scan = run.thread.initialize_scan()
    assert os.path.exists(scan.data_file)

    reloaded = read_provenance(scan.data_file)
    stored = (reloaded.get("config") or {}).get("run_configuration")

    assert stored is not None, "the run configuration was not persisted"
    assert int(stored["generation"]) == int(frozen.generation)
    assert stored["fingerprint"] == frozen.fingerprint
    assert stored == frozen.as_provenance()


# --------------------------------------------------------------------------- #
# Case 5 — absent / foreign / stale is a typed refusal before any side effect.
# --------------------------------------------------------------------------- #

def test_start_refuses_absent_configuration_before_any_side_effect(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 5 (wrapper).  With the Controls owner unable to produce a
    configuration the Run click must refuse BEFORE command, button, session,
    source, output or worker side effects -- never fall back to display state."""
    poni_path = _write_poni(tmp_path / "cal.poni")
    raw_path = tmp_path / "scan_0001.tif"
    raw_path.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()
    widget._set_poni_field(poni_path)
    widget.wrangler.parameters.child("Project", "h5_dir").setValue(str(out))
    widget.wrangler.img_file = str(raw_path)

    started = []
    monkeypatch.setattr(widget.wrangler.thread, "start",
                        lambda: started.append(True))
    emitted = []
    widget.wrangler.sigStart.connect(lambda: emitted.append(True))
    widget.wrangler.command = "stop"
    widget.wrangler.thread.command = "stop"
    stop_enabled = widget.wrangler.ui.stopButton.isEnabled()

    # the supported production opt-out: no Controls V2 owner, so no frozen
    # configuration exists for this click.
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "0")
    widget.wrangler.start()

    assert started == []
    assert emitted == []
    assert widget.wrangler.command == "stop"
    assert widget.wrangler.thread.command == "stop"
    assert widget.wrangler.ui.stopButton.isEnabled() == stop_enabled
    assert widget.wrangler.run_configuration is None
    assert widget.wrangler.thread.run_configuration is None


def test_worker_refuses_absent_configuration_before_opening_output(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 5 (worker, absent).  ``initialize_scan`` is the last checkpoint
    before a writer opens; without the accepted configuration it must raise a
    typed refusal and create no output."""
    from xrd_tools.session import RunConfigurationRefused

    run = _click_run(widget, tmp_path, monkeypatch, points=321)
    run.thread.run_configuration = None

    with pytest.raises(RunConfigurationRefused) as excinfo:
        run.thread.initialize_scan()

    assert excinfo.value.reason == "absent"
    assert not run.output_path.exists()


def test_worker_refuses_foreign_configuration_before_opening_output(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 5 (worker, foreign).  A non-``FrozenRunConfiguration`` carrier
    is a typed refusal, never a duck-typed fallback."""
    from types import SimpleNamespace

    from xrd_tools.session import RunConfigurationRefused

    run = _click_run(widget, tmp_path, monkeypatch, points=321)
    run.thread.run_configuration = SimpleNamespace(
        generation=1,
        fingerprint=run.frozen.fingerprint,
        bai_1d_args={"unit": "q_A^-1", "numpoints": 7},
        bai_2d_args={},
        skip_2d=False,
    )

    with pytest.raises(RunConfigurationRefused) as excinfo:
        run.thread.initialize_scan()

    assert excinfo.value.reason == "foreign"
    assert not run.output_path.exists()


def test_worker_refuses_a_stale_generation_before_opening_output(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 5 (worker, stale).  A configuration frozen for an EARLIER click
    that is still sitting on the worker must be refused: the accepted generation
    only ever moves forward."""
    from xrd_tools.session import RunConfigurationRefused

    run = _click_run(widget, tmp_path, monkeypatch, points=321)
    stale = run.frozen

    # the first run finishes through the production run-end owner, then a second
    # production Run click accepts a newer generation
    widget._exit_run_state(widget._new_projection_receipt())
    widget._on_controls_v2_field_changed(("Int1D", "points"), 456)
    monkeypatch.setattr(widget.wrangler.thread, "start", lambda: None)
    widget.wrangler.start()
    fresh = widget.wrangler.run_configuration
    assert int(fresh.generation) > int(stale.generation)

    run.thread.run_configuration = stale

    with pytest.raises(RunConfigurationRefused) as excinfo:
        run.thread.initialize_scan()

    assert excinfo.value.reason == "stale"


def test_run_returns_without_side_effects_when_configuration_is_absent(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 5 (worker entry).  ``run()`` refuses at the top: no output, no
    source read, no reduction session."""
    run = _click_run(widget, tmp_path, monkeypatch, points=321)
    run.thread.run_configuration = None
    reached = []
    monkeypatch.setattr(run.thread, "process_scan",
                        lambda *a, **k: reached.append(True))

    run.thread.run()

    assert reached == []
    assert not run.output_path.exists()


# --------------------------------------------------------------------------- #
# Case 11 — the backward GI display write stays, but cannot reach the run.
# --------------------------------------------------------------------------- #

def test_backward_gi_display_write_cannot_reach_the_frozen_run(
        widget, tmp_path, monkeypatch):
    """W-1.2 case 11.  The backward GI-mode write onto the DISPLAY scan is
    retained as an acquisition/display projection; it may not change the frozen
    object, and the per-run scan must ignore it."""
    run = _click_run(widget, tmp_path, monkeypatch, points=321, gi=True)
    frozen = run.frozen
    before_fingerprint = frozen.fingerprint
    before_modes = (frozen.bai_1d_args.get("gi_mode_1d"),
                    frozen.bai_2d_args.get("gi_mode_2d"))

    run.thread.gi = True
    run.thread.gi_mode_1d = "POISON_MODE_1D"
    run.thread.gi_mode_2d = "POISON_MODE_2D"
    run.thread._project_gi_modes_onto_display_scan()

    # the backward write STAYS (display/acquisition projection)
    assert widget.scan.bai_1d_args["gi_mode_1d"] == "POISON_MODE_1D"
    assert widget.scan.bai_2d_args["gi_mode_2d"] == "POISON_MODE_2D"
    # ... and reaches neither the frozen object nor the run
    assert frozen.fingerprint == before_fingerprint
    assert (frozen.bai_1d_args.get("gi_mode_1d"),
            frozen.bai_2d_args.get("gi_mode_2d")) == before_modes
    scan = run.thread.initialize_scan()
    assert scan.bai_1d_args.get("gi_mode_1d") == before_modes[0]
    assert scan.bai_2d_args.get("gi_mode_2d") == before_modes[1]


def test_the_frozen_configuration_refuses_mutation(widget, tmp_path, monkeypatch):
    """W-1.2 case 11 (preservation shape -- green at the parent, disclosed).
    The accepted object is deeply immutable, so no consumer can revise it."""
    import dataclasses

    run = _click_run(widget, tmp_path, monkeypatch, points=321)

    with pytest.raises(dataclasses.FrozenInstanceError):
        run.frozen.processing_mode = "Int 1D"
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.frozen.gi.enabled = True
