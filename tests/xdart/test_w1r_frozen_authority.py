"""O-1a-W1R — the accepted frozen configuration is the SOLE execution authority.

Frozen acceptance oracle for the ratified W-1R structural correction (review
§39.2 blocking defects W1R-P1-1..P1-8, §39.3 oracle corrections, §39.4 policy
rulings, §39.5 phase contract, §39.6 mutation tier).  Written BEFORE any
production edit of this packet and demonstrated RED at the accepted parent
``eaa941ac``.

Every case here promotes a preserved §39 exact-tip reproducer into the tree and
repairs the three defects §39.3 named in the previous round's green oracle:

* canonical Controls V2 GI intent is written and ``frozen.gi.enabled`` is
  asserted BEFORE any mirror is poisoned (§39.3 item 1) -- the W-1A helper's
  hidden legacy ``GI/Grazing`` parameter froze a STANDARD configuration, so the
  old GI case could not discriminate;
* identity evidence uses two GENUINE ``FrozenRunConfiguration`` objects, never a
  ``SimpleNamespace`` (§39.3 item 2);
* every poison case asserts the accepted frozen value first and then proves the
  real downstream object/output/frame retained it (§39.3 item 7).

Production-wired per CLAUDE.md rule 2: the real ``staticWidget``, the real
``imageWrangler``/``nexusWrangler`` it built, the real worker threads, the real
Controls transaction/freeze owner, the real ``initialize_scan`` writer path and
the real production File Type selector.  ``thread.start`` is the ONLY blocked
seam -- the Run click itself, ``start_wrangler``, ``_apply_controls_v2_run_state``
and ``imageWrangler.setup()`` all run for real.

Mutation-row map (§39.5 Phase 4):

 1 same-generation foreign frozen object          test_same_generation_foreign_*
 2 future-generation foreign frozen object        test_future_generation_foreign_*
 3 equal-valued reconstructed object              test_equal_valued_reconstructed_*
 4 missing handoff causing a second freeze        test_missing_pending_handoff_*
 5 post-freeze source recapture                   test_execution_source_*
 6 mutable source family/suffix/single-image       test_frozen_series_*, test_frozen_container_*
 7 mutable output path / output mode               test_output_target_*, test_output_mode_*
 8 mutable threshold / mask                        test_threshold_policy_*
 9 mutable GI/motor/orientation/tilt -> frame       test_real_live_frame_*
10 mutable live/batch/core/XYE/series-average      test_mode_and_parallelism_*
11 NeXus local-alias ``skip_2d`` read              test_nexus_skip_2d_*
12 non-JSON-native provenance                     test_supported_frozen_values_*
13 refused Start mutating PONI carriers            test_absent_refusal_does_not_adopt_*
14 ``.hdf5`` only via a test-side selector change  test_hdf5_is_reachable_*
15 wrangler reaching source/output work without
   the exact admitted object                      test_worker_without_the_admitted_*
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.xdart.test_w1a_frozen_execution import (  # noqa: E402
    _click_run,
    _poison_display_state,
    _write_poni,
)

from xrd_tools.session import (  # noqa: E402
    RunConfigurationRefused,
    RunIntent,
    require_run_configuration,
)
from xrd_tools.session.run_configuration import (  # noqa: E402
    FrozenRunConfiguration,
)


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


def _configure_image_run(widget, tmp_path, *, inp_type="Image Series"):
    """Arm the production panel for a real image Run and return the raw path."""
    poni_path = _write_poni(tmp_path / "cal.poni")
    raw = tmp_path / "scan_0001.tif"
    raw.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    widget._set_poni_field(poni_path)
    signal = widget.wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue(inp_type)
    signal.child("File").setValue(str(raw))
    widget.wrangler.parameters.child("Project", "h5_dir").setValue(str(out))
    widget.wrangler.img_file = str(raw)
    return raw, out


# --------------------------------------------------------------------------- #
# W1R-P1-1 / rows 1, 2, 3, 15 — exact-object identity, not a generation floor.
# --------------------------------------------------------------------------- #

def test_same_generation_foreign_frozen_object_is_refused_by_the_shared_owner():
    """Row 1.  A GENUINE frozen object at the accepted generation is not the
    accepted object.  The parent accepted it: ``require_run_configuration``
    only checked absence, type, and ``generation < floor``."""
    accepted = RunIntent(
        processing_mode="Int 2D", output_mode="Append").freeze(generation=7)
    foreign = RunIntent(
        processing_mode="Int 1D", output_mode="Overwrite").freeze(generation=7)
    assert foreign is not accepted
    assert foreign.generation == accepted.generation
    assert foreign.fingerprint != accepted.fingerprint

    with pytest.raises(RunConfigurationRefused) as excinfo:
        require_run_configuration(
            foreign, stage="worker", floor=accepted.generation)
    assert excinfo.value.reason == "foreign"

    # The accepted object itself passes when it is the declared expectation.
    assert require_run_configuration(
        accepted, stage="worker", expected=accepted) is accepted


def test_future_generation_foreign_frozen_object_is_refused_at_consumption():
    """Row 2.  A LATER genuine generation is still not this run's object."""
    accepted = RunIntent(processing_mode="Int 2D").freeze(generation=3)
    future = RunIntent(processing_mode="Int 2D").freeze(generation=99)
    with pytest.raises(RunConfigurationRefused) as excinfo:
        require_run_configuration(
            future, stage="worker", expected=accepted)
    assert excinfo.value.reason == "foreign"


def test_equal_valued_reconstructed_object_is_refused_at_consumption():
    """Row 3.  Equal VALUES are not identity: same generation, same fingerprint,
    different object."""
    intent = RunIntent(processing_mode="Int 2D", output_mode="Append")
    accepted = intent.freeze(generation=5)
    rebuilt = RunIntent.from_frozen(accepted).freeze(generation=5 + 1)
    rebuilt_same_gen = RunIntent(
        processing_mode="Int 2D", output_mode="Append").freeze(generation=5)
    assert rebuilt_same_gen.fingerprint == accepted.fingerprint
    assert rebuilt_same_gen is not accepted

    for candidate in (rebuilt, rebuilt_same_gen):
        with pytest.raises(RunConfigurationRefused) as excinfo:
            require_run_configuration(
                candidate, stage="worker", expected=accepted)
        assert excinfo.value.reason == "foreign"


def test_wrapper_refuses_a_same_generation_foreign_object_without_mutating():
    """Row 1 (wrapper half).  Admission may not replace an already-bound
    object with a different genuine one, and must leave both carriers alone."""
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        wranglerWidget,
    )

    accepted = RunIntent(
        processing_mode="Int 2D", output_mode="Append").freeze(generation=4)
    foreign = RunIntent(
        processing_mode="Int 1D", output_mode="Overwrite").freeze(generation=4)
    thread = SimpleNamespace(
        run_configuration=accepted, run_configuration_floor=accepted.generation)
    holder = SimpleNamespace(
        run_configuration=accepted,
        run_configuration_floor=accepted.generation,
        thread=thread,
        _h19_host=SimpleNamespace(
            _prepare_controls_v2_run_configuration=lambda: foreign),
    )

    with pytest.raises(RunConfigurationRefused) as excinfo:
        wranglerWidget._admit_run_configuration(holder, "wrapper")
    assert excinfo.value.reason == "foreign"
    assert holder.run_configuration is accepted
    assert thread.run_configuration is accepted


def test_worker_without_the_admitted_object_refuses_before_source_or_output():
    """Row 15.  A worker whose carrier was replaced after admission refuses at
    the consumption gate rather than executing the substitution."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )

    accepted = RunIntent(processing_mode="Int 2D").freeze()
    foreign = RunIntent(processing_mode="Int 1D").freeze()
    assert foreign.generation == accepted.generation
    assert foreign.fingerprint != accepted.fingerprint

    worker = SimpleNamespace(
        run_configuration=foreign,
        run_configuration_floor=accepted.generation,
    )
    with pytest.raises(RunConfigurationRefused) as excinfo:
        imageThread._require_run_configuration(worker, "policy-probe")
    assert excinfo.value.reason == "foreign"


def test_real_worker_refuses_a_same_generation_foreign_configuration(
        widget, tmp_path, monkeypatch):
    """Row 1 / row 15 (real-worker half).  A real production Run click, then a
    genuine same-generation substitution on the real worker: the real
    ``initialize_scan`` must refuse before it opens output."""
    _configure_image_run(widget, tmp_path)
    monkeypatch.setattr(widget.wrangler.thread, "start", lambda: None)
    widget.wrangler.start()
    thread = widget.wrangler.thread
    accepted = thread.run_configuration
    assert accepted is widget.wrangler.run_configuration

    foreign = RunIntent(
        source_spec=accepted.thaw_source_spec(),
        processing_mode="Int 1D",
        output_mode="Overwrite",
        poni_file=accepted.poni_file,
        poni_values=accepted.poni_values,
    ).freeze(generation=accepted.generation)
    assert foreign is not accepted
    assert foreign.fingerprint != accepted.fingerprint
    thread.run_configuration = foreign

    with pytest.raises(RunConfigurationRefused) as excinfo:
        thread.initialize_scan()
    assert excinfo.value.reason == "foreign"


def test_post_entry_substitution_cannot_change_this_runs_policy(
        widget, tmp_path, monkeypatch):
    """Row 16 (added at the orchestrator's request, review §39.5 Phase 2 item 1).

    ``frozen_run_policy`` re-reads the carrier per access and checks presence and
    type only -- identity lives at the worker-entry gate.  The consequence that
    must hold is therefore: once an entry gate has QUALIFIED the accepted object,
    reassigning ``thread.run_configuration`` to a GENUINE same-generation object
    with a different fingerprint cannot change what the run executes.

    Both acceptable outcomes are asserted: the qualified reference is what
    execution uses, AND the next entry gate refuses the substituted carrier.
    """
    run = _click_run(widget, tmp_path, monkeypatch, write_mode="Append")
    thread = run.thread
    accepted = run.frozen

    # Enter and qualify, exactly as the worker does.
    assert imageThreadEntryGate(thread, "row16-entry") is accepted

    substitute = RunIntent(
        source_spec=accepted.thaw_source_spec(),
        processing_mode="Int 1D",
        output_mode="Overwrite",
        save_path=str(tmp_path / "substituted-output"),
        poni_file=accepted.poni_file,
        poni_values=accepted.poni_values,
    ).freeze(generation=accepted.generation)
    assert substitute is not accepted
    assert substitute.generation == accepted.generation
    assert substitute.fingerprint != accepted.fingerprint
    thread.run_configuration = substitute

    # 1) the qualified reference still drives every policy read
    assert thread.write_mode == accepted.output_mode == "Append"
    assert Path(thread.h5_dir) == Path(accepted.save_path)
    assert thread._append_skip_enabled() is True
    assert bool(thread.xye_only) is bool(
        accepted.run_options.get("xye_only", False))

    # 2) and the next entry gate refuses the substituted carrier outright
    with pytest.raises(RunConfigurationRefused) as excinfo:
        imageThreadEntryGate(thread, "row16-reentry")
    assert excinfo.value.reason == "foreign"


def imageThreadEntryGate(thread, stage):
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )

    return imageThread._require_run_configuration(thread, stage)


# --------------------------------------------------------------------------- #
# W1R-P1-3 / row 4 — a lost handoff refuses; it never freezes again.
# --------------------------------------------------------------------------- #

def test_missing_pending_handoff_refuses_instead_of_refreezing(
        widget, tmp_path, monkeypatch):
    """Row 4.  Clearing the pending slot between admission and host delivery
    made the parent freeze a SECOND generation and proceed."""
    _configure_image_run(widget, tmp_path)
    started = []
    monkeypatch.setattr(
        widget.wrangler.thread, "start", lambda: started.append(True))
    original = widget._prepare_controls_v2_run_configuration
    prepared = []

    def prepare():
        frozen = original()
        prepared.append(frozen)
        if len(prepared) == 1:
            widget._pending_controls_v2_run_configuration = None
        return frozen

    monkeypatch.setattr(
        widget, "_prepare_controls_v2_run_configuration", prepare)
    widget.wrangler.start()

    assert len(prepared) == 1, "the lost handoff produced a second freeze"
    assert widget.wrangler.run_configuration is prepared[0]
    assert started == [], "the run started on an unverified handoff"


# --------------------------------------------------------------------------- #
# W1R-P1-2 / row 5 — the execution source is captured once.
# --------------------------------------------------------------------------- #

def test_execution_source_is_thawed_from_the_accepted_frozen_object(
        widget, tmp_path, monkeypatch):
    """Row 5.  ``start_wrangler`` re-called the GUI source capture and installed
    capture B while the accepted configuration held capture A."""
    import inspect

    from xrd_tools.core.scan import SourceKind, SourceSpec

    raw, _out = _configure_image_run(widget, tmp_path)
    other = tmp_path / "other_0001.tif"
    other.write_bytes(b"")
    first = SourceSpec(raw, SourceKind.IMAGE_FILE)
    second = SourceSpec(other, SourceKind.IMAGE_FILE)
    seen = []

    def alternating():
        caller = inspect.stack()[1].function
        value = second if caller == "start_wrangler" else first
        seen.append((caller, value))
        return value

    monkeypatch.setattr(
        widget, "_controls_v2_freeze_source_spec", alternating)
    monkeypatch.setattr(widget.wrangler.thread, "start", lambda: None)

    widget.wrangler.start()

    frozen = widget.wrangler.run_configuration
    assert frozen is not None
    assert frozen.thaw_source_spec() == first
    assert widget.wrangler.thread.source_spec == first
    assert all(caller != "start_wrangler" for caller, _v in seen), seen


def test_start_wrangler_does_not_recapture_the_source_after_acceptance(
        widget, tmp_path, monkeypatch):
    """Row 5 (call-site half).  Profile/observation captures may remain; the
    EXECUTION path must not ask the GUI for the source again."""
    import inspect

    _configure_image_run(widget, tmp_path)
    original = widget._controls_v2_freeze_source_spec
    stacks = []

    def counted():
        value = original()
        stacks.append(
            tuple(frame.function for frame in inspect.stack()[1:6]))
        return value

    monkeypatch.setattr(widget, "_controls_v2_freeze_source_spec", counted)
    monkeypatch.setattr(widget.wrangler.thread, "start", lambda: None)

    widget.wrangler.start()

    assert all("start_wrangler" not in stack for stack in stacks), stacks


# --------------------------------------------------------------------------- #
# W1R-P1-5 / row 6 — mutable format/mode fields cannot re-decide the source.
# --------------------------------------------------------------------------- #

def test_frozen_series_is_not_overridden_by_a_master_shaped_mutable_path(
        widget, tmp_path, monkeypatch):
    """Row 6.  A frozen TIFF series plus a master-shaped mutable ``img_file``
    routed to the Eiger reader on the parent."""
    from collections import deque

    import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as iwt
    from xrd_tools.sources import image_series_spec

    tif = tmp_path / "scan_0001.tif"
    tif.write_bytes(b"")
    frozen = RunIntent(
        source_spec=image_series_spec(tif), processing_mode="Int 2D").freeze()
    thread = widget.wrangler.thread
    thread.run_configuration = frozen
    thread.run_configuration_floor = frozen.generation
    thread.source_spec = frozen.thaw_source_spec()
    thread.img_file = str(tmp_path / "rogue_master.h5")
    thread.img_ext = "tif"
    thread.inp_type = "Image Series"
    thread.single_img = False
    thread.img_fnames = deque()
    thread.processed = []
    thread.scan_name = "scan"
    thread.meta_ext = None
    eiger = []
    monkeypatch.setattr(
        thread, "_get_next_eiger_frame",
        lambda: eiger.append(True) or (None, "s", 1, None, {}))
    monkeypatch.setattr(
        iwt, "read_image", lambda path: np.ones((2, 2), dtype=float))

    result = thread.get_next_image()

    assert eiger == [], "a mutable master-shaped path reached the Eiger reader"
    assert Path(result[0]).name == tif.name


def test_frozen_container_directory_is_not_overridden_by_single_image_flag(
        widget, tmp_path, monkeypatch):
    """Row 6.  A frozen container directory plus mutable ``single_img=True``
    routed to plain-image handling on the parent."""
    from collections import deque

    import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as iwt
    from xrd_tools.sources import DirectorySourceSpec

    frozen = RunIntent(
        source_spec=DirectorySourceSpec(
            root=tmp_path, suffixes=("_master.h5",)),
        processing_mode="Int 2D",
    ).freeze()
    ordinary = tmp_path / "ordinary.tif"
    ordinary.write_bytes(b"")
    thread = widget.wrangler.thread
    thread.run_configuration = frozen
    thread.run_configuration_floor = frozen.generation
    thread.source_spec = frozen.thaw_source_spec()
    thread.img_file = str(ordinary)
    thread.img_ext = "tif"
    thread.inp_type = "Single Image"
    thread.single_img = True
    thread.img_fnames = deque()
    thread.processed = []
    thread.scan_name = "ordinary"
    thread.meta_ext = None
    eiger = []
    monkeypatch.setattr(
        thread, "_get_next_eiger_frame",
        lambda: eiger.append(True) or (None, "s", 1, None, {}))
    monkeypatch.setattr(
        iwt, "read_image", lambda path: np.ones((2, 2), dtype=float))

    thread.get_next_image()

    assert eiger == [True], "the frozen container source lost the reader choice"


def test_output_safety_uses_the_frozen_container_source(
        widget, tmp_path, monkeypatch):
    """Row 6 / row 7.  Poisoning ``img_ext='tif'`` disabled container-directory
    collision protection for a frozen H5 source on the parent."""
    from xrd_tools.io.output_safety import (
        OutputCollisionError,
        check_output_not_source,
    )

    signal = widget.wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(tmp_path))
    signal.child("img_ext").setValue("h5")

    frozen = widget._prepare_controls_v2_run_configuration()
    assert frozen is not None
    assert frozen.source is not None
    assert frozen.source.family == "directory"

    thread = widget.wrangler.thread
    thread.run_configuration = frozen
    thread.run_configuration_floor = frozen.generation
    thread.inp_type = "Image Directory"
    thread.img_dir = str(tmp_path)
    thread.h5_dir = str(tmp_path)
    thread.img_file = ""
    thread.img_ext = "tif"

    with pytest.raises(OutputCollisionError):
        check_output_not_source(
            str(tmp_path / "raw_master.nxs"), **thread._output_safety_args())


# --------------------------------------------------------------------------- #
# W1R-P1-4 / rows 7, 8, 10 — output, threshold, mode/parallelism.
# --------------------------------------------------------------------------- #

def test_output_mode_is_consumed_from_the_frozen_configuration(
        widget, tmp_path, monkeypatch):
    """Row 7.  A post-admission ``thread.write_mode`` write turned frozen
    Append into non-Append on the parent."""
    _configure_image_run(widget, tmp_path)
    run = _click_run(widget, tmp_path, monkeypatch, write_mode="Append")
    assert run.frozen.output_mode == "Append"
    assert run.thread.run_configuration is run.frozen

    run.thread.write_mode = "Overwrite"
    assert run.thread._append_skip_enabled() is True


def test_output_target_is_consumed_from_the_frozen_configuration(
        widget, tmp_path, monkeypatch):
    """Row 7.  A post-admission ``thread.h5_dir`` write moved the real writer
    target on the parent."""
    run = _click_run(widget, tmp_path, monkeypatch)
    accepted_output = Path(run.frozen.save_path)
    assert accepted_output == Path(run.out_dir)

    run.thread.h5_dir = str(tmp_path / "late-output")
    run.thread.scan_name = "authority"
    scan = run.thread.initialize_scan()

    assert Path(scan.data_file).parent == accepted_output


def test_threshold_policy_is_consumed_from_the_frozen_configuration(
        widget, tmp_path, monkeypatch):
    """Row 8.  Post-admission threshold mirrors turned ``[1.0]`` into NaN on
    the parent -- a scientific pixel-rejection change."""
    run = _click_run(widget, tmp_path, monkeypatch)
    assert run.frozen.threshold.apply_threshold is False

    run.thread.apply_threshold = True
    run.thread.threshold_min = 0.0
    run.thread.threshold_max = 0.0
    observed = run.thread._apply_threshold_inline(np.asarray([1.0]))

    assert np.array_equal(observed, np.asarray([1.0]))


def test_mode_and_parallelism_are_consumed_from_the_frozen_configuration(
        widget, tmp_path, monkeypatch):
    """Row 10.  live/batch/cores/XYE/series-average must all come from the
    accepted object, not from post-admission worker mirrors."""
    run = _click_run(widget, tmp_path, monkeypatch)
    frozen = run.frozen
    thread = run.thread

    thread.batch_mode = not bool(frozen.batch_mode)
    thread.live_mode = not bool(frozen.live_mode)
    thread.max_cores = int(frozen.max_cores) + 7
    thread.xye_only = not bool(frozen.run_options.get("xye_only", False))
    thread.series_average = True

    assert bool(thread.batch_mode) is bool(frozen.batch_mode)
    assert bool(thread.live_mode) is bool(frozen.live_mode)
    assert int(thread.max_cores) == int(frozen.max_cores)
    assert bool(thread.xye_only) is bool(
        frozen.run_options.get("xye_only", False))
    assert bool(thread.series_average) is bool(
        frozen.run_options.get("series_average", False))


# --------------------------------------------------------------------------- #
# W1R-P1-4 / row 9 — canonical GI intent reaches a REAL LiveFrame.
# --------------------------------------------------------------------------- #

def test_real_live_frame_consumes_the_frozen_gi_policy(
        widget, tmp_path, monkeypatch):
    """Row 9, and the §39.3 item-1 oracle repair.

    Canonical Controls V2 GI intent is written and ``frozen.gi.enabled`` is
    asserted BEFORE poisoning, so the case genuinely discriminates.  The parent
    produced ``LiveFrame.gi is False`` and ``th_mtr == 'POISON_MOTOR'``.
    """
    widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
    run = _click_run(widget, tmp_path, monkeypatch, points=321, gi=False)
    assert run.frozen.gi.enabled is True, (
        "canonical Controls V2 GI intent did not reach the freeze")

    _poison_display_state(run)
    scan = run.thread.initialize_scan()
    frame = run.thread._build_batch_frames(
        scan,
        [(str(tmp_path / "scan_0001.tif"), 1,
          np.zeros((4, 4), dtype=np.float32), {}, None, 0.0)],
    )[0]

    assert frame.gi is run.frozen.gi.enabled
    assert frame.th_mtr == run.frozen.gi.scan_incidence_motor
    assert int(frame.sample_orientation) == int(
        run.frozen.gi.sample_orientation)
    assert float(frame.tilt_angle) == float(run.frozen.gi.tilt_angle)


# --------------------------------------------------------------------------- #
# W1R-P1-6 / row 11 — the NeXus local-alias skip_2d execution read.
# --------------------------------------------------------------------------- #

def test_nexus_skip_2d_execution_read_uses_frozen_policy_not_the_scan_alias():
    """Row 11.  ``nexusThread._run_impl`` aliases the initialized scan locally
    and read ``scan.skip_2d``; the committed AST census only recognized literal
    ``self.scan`` shapes, so it missed this.

    Targeted discriminator (§39.2 W1R-P1-6: no general taint analyzer): the
    execution decision in ``_run_impl`` must not read ``skip_2d`` off ANY local
    alias of the initialized scan.
    """
    import ast
    import inspect
    import textwrap

    from xdart.gui.tabs.static_scan.wranglers import nexus_wrangler_thread

    source = inspect.getsource(nexus_wrangler_thread.nexusThread._run_impl)
    tree = ast.parse(textwrap.dedent(source))

    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Call) \
                    and isinstance(value.func, ast.Attribute) \
                    and value.func.attr in {
                        "_initialize_scan", "initialize_scan"}:
                aliases.add(node.targets[0].id)
            if isinstance(value, ast.Attribute) and value.attr == "scan":
                aliases.add(node.targets[0].id)

    offenders = [
        (node.lineno, ast.unparse(node))
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "skip_2d"
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    ]
    assert offenders == [], (
        "nexusThread._run_impl still decides 2D integration from a mutable "
        f"display-scan alias: {offenders}")


# --------------------------------------------------------------------------- #
# W1R-P1-7 / row 12 — detached, JSON-native provenance before output opens.
# --------------------------------------------------------------------------- #

def test_supported_frozen_values_have_json_native_provenance():
    """Row 12.  A supported pyFAI method TUPLE survived into provenance, so
    ``json.loads(json.dumps(p)) != p``."""
    frozen = RunIntent(
        bai_1d_args={"method": ("bbox", "csr", "cython")}).freeze()
    provenance = frozen.as_provenance()
    assert json.loads(json.dumps(provenance)) == provenance


def test_nested_supported_values_are_normalized_recursively():
    """Row 12 (recursion half)."""
    frozen = RunIntent(
        bai_2d_args={
            "method": ("bbox", "csr", "cython"),
            "radial_range": (0.1, 2.0),
            "nested": {"pair": (1, 2), "deep": [(3, 4)]},
        },
        run_options={"tuple_option": ("a", "b")},
    ).freeze()
    provenance = frozen.as_provenance()
    assert json.loads(json.dumps(provenance)) == provenance
    assert provenance["bai_2d_args"]["method"] == ["bbox", "csr", "cython"]
    assert provenance["bai_2d_args"]["nested"]["deep"] == [[3, 4]]


def test_path_provenance_values_are_normalized_not_left_live():
    """Row 12 (supported-but-not-native half).  ``Path`` is an accepted frozen
    value; provenance must carry its canonical string, not a live object."""
    frozen = RunIntent(
        bai_1d_args={"lut_file": Path("/tmp/lut.h5")}).freeze()
    provenance = frozen.as_provenance()
    assert json.loads(json.dumps(provenance)) == provenance
    assert provenance["bai_1d_args"]["lut_file"] == "/tmp/lut.h5"


def test_unsupported_provenance_value_refuses_before_output_opens():
    """Row 12 (fail-closed half).  A value with no canonical JSON form must
    REFUSE before output creation, never be stringified into the record."""
    frozen = RunIntent(bai_1d_args={"blob": b"\x00\x01"}).freeze()
    with pytest.raises((TypeError, ValueError)):
        frozen.as_provenance()


# --------------------------------------------------------------------------- #
# W1R-P1-8 / row 13 — a refused Start is zero-delta on PONI carriers.
# --------------------------------------------------------------------------- #

def test_absent_refusal_does_not_adopt_loaded_scan_calibration(
        widget, tmp_path, monkeypatch):
    """Row 13.  ``_inputs_valid`` adopted loaded-scan PONI/integrator values
    into the wrapper and worker carriers BEFORE admission refused."""
    raw, _out = _configure_image_run(widget, tmp_path)
    wrangler = widget.wrangler
    adopted = object()
    integrator = object()
    wrangler.poni = None
    wrangler.thread._adopted_poni = None
    wrangler.scan._cached_poni = adopted
    wrangler.scan._cached_integrator = integrator
    monkeypatch.setattr(
        type(wrangler.scan.frames), "index",
        property(lambda self: [1]), raising=False)
    wrangler.img_file = str(raw)
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "0")

    wrangler.start()

    assert wrangler.poni is None
    assert wrangler.thread._adopted_poni is None


def test_absent_controls_owner_refuses_at_the_production_seam(
        widget, tmp_path, monkeypatch):
    """§39.4.  Absence is injected at the production seam, not only through the
    retiring env flag: an absent Controls V2 owner fails CLOSED and starts
    nothing."""
    _configure_image_run(widget, tmp_path)
    started = []
    monkeypatch.setattr(
        widget.wrangler.thread, "start", lambda: started.append(True))
    monkeypatch.setattr(
        widget, "_prepare_controls_v2_run_configuration", lambda: None)

    widget.wrangler.start()

    assert started == []
    assert widget.wrangler.run_configuration is None


# --------------------------------------------------------------------------- #
# §39.4 / row 14 — .hdf5 through the UNMODIFIED production selector.
# --------------------------------------------------------------------------- #

def test_hdf5_is_reachable_through_the_real_production_file_type_selector(
        widget, tmp_path):
    """Row 14.  ``hdf5`` was absent from the production File Type selector, so
    the supported backend branch was unreachable without a test-side override.
    """
    signal = widget.wrangler.parameters.child("Signal")
    values = list(signal.child("img_ext").opts.get("limits", ()))
    assert "hdf5" in values, (
        f"the real File Type selector still cannot select hdf5: {values}")

    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(tmp_path))
    signal.child("img_ext").setValue("hdf5")
    assert signal.child("img_ext").value() == "hdf5"

    frozen = widget._prepare_controls_v2_run_configuration()
    assert frozen is not None and frozen.source is not None
    assert frozen.source.family == "directory"
    assert any(
        str(suffix).lower().endswith("hdf5") for suffix in frozen.source.suffixes
    ), frozen.source.suffixes
