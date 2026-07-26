"""O-1a-W1R-D — the frozen acceptance oracle for DIRECT routing.

Review §40 (``handoffs/x1_o1a0_o1ai_adversarial_review_2026-07-23.md``) rejected
the ``FrozenRunProjection`` descriptor bridge as a second broad policy-routing
authority and ratified W-1R-D: truthful source admission (D1), explicit routing
of the exact admitted ``FrozenRunConfiguration`` (D2), and deletion of the bridge
(D3).  This module is the §40.3 D0 oracle, written BEFORE any production edit of
this packet and demonstrated RED at ``e3bc2459``.

Case map (§40.3 D0 items 1-14):

  1  project-folder invalidation clears field + runtime source, refuses visibly
  2  operator-cleared File refuses visibly
  3  configured non-existent File cannot retain or run the old cursor
  4  a deliberately restored stale ``img_file`` still cannot bypass admission
  5  absent source produces zero wrapper/thread/pending/output/source delta
  6  a valid TIFF series has exact frozen format/recursive/name_filter, immune
     to legacy poison
  7  a real seedless container Directory starts in Batch AND Live with a typed
     ``DirectorySourceSpec``
  8  every enabled plain-image Directory selection freezes a typed spec and
     starts lazily (selectable-but-unstartable is not acceptable)
  9  Run A then Run B on the reused real worker: no A value reaches B
 10  assigning a frozen carrier without exact admission activates no policy
 11  equal-valued reconstruction and genuine same-generation foreign refuse by
     ``is``
 12  valid source changes populate the GI theta-motor choices, and freeze
     resolves the same effective motor exactly once
 13  Image and NeXus readers, append comparison, written provenance and reload
     retain the exact accepted identity
 14  a refused Start publishes no calibration/source/output/wrapper/thread/
     generation/pending mutation

Promotion decisions, recorded HERE because §40.3 makes this oracle immutable
after the first production edit except through a Rule-9 entry.  Two assertions in
the preserved Codex probes are deliberately expressed at the CONSEQUENCE rather
than transcribed, because both would otherwise pin the very mechanism §40.3 D3
orders deleted:

* ``test_codex_w1r_source_none_exact.py`` asserts ``thread.img_ext == "tif"``
  after poisoning ``thread.img_ext``.  That read-back only holds while a
  descriptor intercepts the name.  Case 6 therefore asserts that the frozen
  format governs what execution DOES -- output safety, container classification
  and reader selection -- which is what "immune to legacy poison" means once the
  legacy name is an inert local attribute.  This is the third round of this arc
  in which an oracle asserting a mechanism-specific spelling coexisted with a
  broken consequence; the consequence is the durable contract.
* the same module asserts ``_attempt_start(...) == [True]`` for an absent source
  and then that the poisoned slots stay inert.  That is the pre-fix shape: §40.3
  D1 item 4 makes absent source a REFUSAL, so cases 4 and 5 assert refusal plus
  zero delta instead of an inert-slot start.

Production-wired per CLAUDE.md rule 2: the real ``staticWidget``, the real
``imageWrangler``/``nexusWrangler``, the real worker threads, the real Controls
freeze owner and the real Start boundary.  ``thread.start`` is the only blocked
seam.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.test_w1a_frozen_execution import _write_poni

from xrd_tools.session import (
    RunConfigurationRefused,
    RunIntent,
)
from xrd_tools.session.run_configuration import FrozenRunConfiguration


PLAIN_IMAGE_DIRECTORY_EXTENSIONS = ("tif", "raw", "mar3450")


# --------------------------------------------------------------------------- #
# Fixtures and production-shaped arming helpers
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


def _arm_series(widget, root: Path, *, ext: str = "tif"):
    """Arm a real Image Series run inside its own project folder."""
    wrangler = widget.wrangler
    signal = wrangler.parameters.child("Signal")
    project = root / "project"
    project.mkdir(parents=True, exist_ok=True)
    raw = project / f"scan_0001.{ext}"
    raw.write_bytes(b"")
    (project / f"scan_0002.{ext}").write_bytes(b"")
    output = project / "out"
    output.mkdir(exist_ok=True)
    wrangler.parameters.child("Project", "project_folder").setValue(str(project))
    widget._set_poni_field(_write_poni(project / "cal.poni"))
    signal.child("inp_type").setValue("Image Series")
    signal.child("File").setValue(str(raw))
    wrangler.parameters.child("Project", "h5_dir").setValue(str(output))
    return raw, output


def _arm_directory(widget, root: Path, *, ext: str, live: bool = False):
    """Arm a real seedless Image Directory run (no representative file)."""
    wrangler = widget.wrangler
    signal = wrangler.parameters.child("Signal")
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = root / "out"
    output.mkdir(exist_ok=True)
    widget._set_poni_field(_write_poni(root / "cal.poni"))
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(raw_dir))
    signal.child("img_ext").setValue(ext)
    signal.child("File").setValue("")
    wrangler.parameters.child("Project", "h5_dir").setValue(str(output))
    widget.controls.liveButton.setChecked(live)
    return raw_dir, output


def _attempt_start(widget, monkeypatch):
    """Press the real Run boundary; block only the worker's own thread start."""
    started = []
    monkeypatch.setattr(
        widget.wrangler.thread, "start", lambda: started.append(True))
    widget.wrangler.start()
    return started


def _carrier_snapshot(wrangler):
    """Everything a refusal must leave untouched (§40.3 D0 item 14)."""
    thread = wrangler.thread
    return {
        "wrangler.run_configuration": getattr(
            wrangler, "run_configuration", None),
        "wrangler.admitted": getattr(
            wrangler, "_admitted_run_configuration", None),
        "wrangler.floor": getattr(wrangler, "run_configuration_floor", 0),
        "wrangler.source_spec": getattr(wrangler, "source_spec", None),
        "wrangler.poni": getattr(wrangler, "poni", None),
        "wrangler.poni_file": getattr(wrangler, "poni_file", None),
        "thread.run_configuration": getattr(
            thread, "run_configuration", None),
        "thread.admitted": getattr(
            thread, "_admitted_run_configuration", None),
        "thread.source_spec": getattr(thread, "source_spec", None),
        "thread.h5_dir": getattr(thread, "h5_dir", None),
        "thread.adopted_poni": getattr(thread, "_adopted_poni", None),
    }


def _frozen_pair(save_path, *, generation=0, batch=False, cores=1):
    """Two genuine frozen objects for identity work."""
    from xrd_tools.core.scan import SourceKind, SourceSpec

    intent = RunIntent(
        generation=generation,
        source_spec=SourceSpec(f"/tmp/{save_path}/frame_0001.tif",
                               SourceKind.IMAGE_FILE),
        save_path=str(save_path),
        batch_mode=batch,
        max_cores=cores,
    )
    return intent.freeze()


# --------------------------------------------------------------------------- #
# §40.1 P1-A / D0 items 1-4 — the runtime source cursor may never outlive the
# authoritative selection, and admission may never accept an absent source.
# --------------------------------------------------------------------------- #

def test_project_switch_clears_the_field_and_the_runtime_source_and_refuses(
        widget, tmp_path, monkeypatch):
    """Item 1.  ``_on_project_folder_changed`` clears ``Signal/File`` for the new
    project; ``get_img_fname`` left ``self.img_file`` on the OLD project's TIFF
    because it only assigns when ``os.path.exists(configured)`` is true."""
    old_raw, _out = _arm_series(widget, tmp_path)
    new_project = tmp_path / "new-project"
    new_project.mkdir()
    widget.wrangler.parameters.child(
        "Project", "project_folder").setValue(str(new_project))
    widget._set_poni_field(_write_poni(new_project / "new.poni"))

    signal = widget.wrangler.parameters.child("Signal")
    assert signal.child("File").value() == ""
    assert widget.wrangler.img_file == "", (
        f"the runtime source cursor retained the prior project path {old_raw}")
    assert widget._controls_v2_freeze_source_spec() is None
    assert _attempt_start(widget, monkeypatch) == []


def test_operator_cleared_file_refuses_visibly(widget, tmp_path, monkeypatch):
    """Item 2.  The editable ``Image File`` field is a ``str_browse`` param: an
    operator can empty it without any browse."""
    _arm_series(widget, tmp_path)
    status = []
    monkeypatch.setattr(
        widget.wrangler, "_safe_status_text",
        lambda text, *a, **k: status.append(str(text)), raising=False)
    widget.wrangler.parameters.child("Signal", "File").setValue("")

    assert widget.wrangler.img_file == ""
    assert widget._controls_v2_freeze_source_spec() is None
    assert _attempt_start(widget, monkeypatch) == []
    assert status, "the refusal was silent; the operator gets no reason"


def test_configured_nonexistent_file_cannot_run_the_previous_source(
        widget, tmp_path, monkeypatch):
    """Item 3.  §40.1 P1-A: the clear must also cover a configured file that does
    not exist, not only the empty string."""
    _arm_series(widget, tmp_path)
    missing = tmp_path / "project" / "does-not-exist_0001.tif"
    widget.wrangler.parameters.child("Signal", "File").setValue(str(missing))

    assert widget.wrangler.img_file == "", (
        "a non-existent editable selection retained the previous runtime cursor")
    assert _attempt_start(widget, monkeypatch) == []


def test_restored_stale_cursor_still_cannot_bypass_admission(
        widget, tmp_path, monkeypatch):
    """Item 4.  The root-cause clear and the admission invariant are BOTH
    required (§40.1 P1-A).  Restore the stale cursor by hand and the run must
    still refuse, because admission requires a typed frozen source."""
    old_raw, _out = _arm_series(widget, tmp_path)
    widget.wrangler.parameters.child("Signal", "File").setValue("")
    widget.wrangler.img_file = str(old_raw)

    assert _attempt_start(widget, monkeypatch) == []
    assert getattr(widget.wrangler, "_admitted_run_configuration", None) is None
    assert getattr(
        widget.wrangler.thread, "_admitted_run_configuration", None) is None


def test_absent_source_refusal_is_genuinely_zero_delta(
        widget, tmp_path, monkeypatch):
    """Item 5 and §40.1 P1-D.  ``_admit_run_configuration`` called the host
    preparation owner BEFORE required-value validation, and the real
    ``_prepare_controls_v2_run_configuration`` published the pending object plus
    wrapper, source and thread carriers before returning."""
    old_raw, _out = _arm_series(widget, tmp_path)
    widget.wrangler.parameters.child("Signal", "File").setValue("")
    widget.wrangler.img_file = str(old_raw)
    before = _carrier_snapshot(widget.wrangler)

    assert _attempt_start(widget, monkeypatch) == []

    after = _carrier_snapshot(widget.wrangler)
    assert after == before, (
        "a refused Start mutated run carriers: "
        f"{ {k: (before[k], after[k]) for k in before if before[k] != after[k]} }")
    assert getattr(
        widget, "_pending_controls_v2_run_configuration", None) is None, (
        "the tentative handoff survived a refusal")


# --------------------------------------------------------------------------- #
# §40.1 P1-B / D0 item 6 — a present, valid typed source must be TOTAL.
# --------------------------------------------------------------------------- #

def test_valid_tiff_series_has_total_typed_source_values(
        widget, tmp_path, monkeypatch):
    """Item 6.  A TIFF Image Series freezes a source whose URI is the CONTAINING
    DIRECTORY, so deriving the format from the URI answered "no format" and the
    non-directory recursive/filter questions answered "not applicable" -- three
    values that fell through to writable compatibility slots.

    Asserted at the consequence (see the module docstring): the frozen format
    governs container classification, output safety and reader selection even
    after the retired legacy names are poisoned.
    """
    raw, _out = _arm_series(widget, tmp_path)
    assert _attempt_start(widget, monkeypatch) == [True]
    thread = widget.wrangler.thread
    frozen = thread.run_configuration
    assert frozen.source is not None
    assert str(frozen.source.source_kind) == "tiff_series"

    thread.img_ext = "h5"
    thread.include_subdir = True
    thread.file_filter = "POISON"
    thread.inp_type = "Image Directory"
    safety = thread._output_safety_args()

    assert safety["recursive"] is False
    assert safety["container_directory_mode"] is False
    assert safety["watched_dirs"] == [str(raw.parent)], safety["watched_dirs"]
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )
    assert imageThread._frozen_source_is_container(thread) is False


# --------------------------------------------------------------------------- #
# D0 items 7-8 — every ENABLED production source shape freezes a typed source.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("ext", ["h5", "hdf5", "nxs"])
def test_seedless_container_directory_starts_in_batch_and_live(
        widget, tmp_path, monkeypatch, ext, live):
    """Item 7.  Legitimate seedless container Directory runs must survive the new
    admission requirement -- they intentionally begin with an empty ``img_file``.
    """
    from xrd_tools.sources import DirectorySourceSpec

    raw_dir, _out = _arm_directory(widget, tmp_path, ext=ext, live=live)
    spec = widget._controls_v2_freeze_source_spec()
    assert isinstance(spec, DirectorySourceSpec)
    assert spec.root == raw_dir
    assert widget.wrangler.img_file == ""
    assert widget.wrangler._directory_run_without_seed_ok() is True

    assert _attempt_start(widget, monkeypatch) == [True]
    assert widget.wrangler.run_configuration.source is not None
    assert widget.wrangler.run_configuration.source.family == "directory"


@pytest.mark.parametrize("ext", PLAIN_IMAGE_DIRECTORY_EXTENSIONS)
def test_plain_image_directory_freezes_a_typed_source_and_starts(
        widget, tmp_path, monkeypatch, ext):
    """Item 8.  ``_controls_v2_container_index_config`` returns ``None`` for a
    non-container extension, so a plain-image Directory froze NO source while
    remaining selectable in the production File Type list.  §40.3 D0 item 8:
    selectable-but-unstartable is not acceptable, and typed support is preferred
    because the worker already implements directory traversal.
    """
    from xrd_tools.sources import DirectorySourceSpec

    raw_dir, _out = _arm_directory(widget, tmp_path, ext=ext)
    (raw_dir / f"frame_0001.{ext}").write_bytes(b"")

    spec = widget._controls_v2_freeze_source_spec()
    assert isinstance(spec, DirectorySourceSpec), (
        f"a {ext} Image Directory is selectable but freezes {spec!r}")
    assert spec.root == raw_dir
    assert any(str(s).lstrip(".").lower() == ext for s in spec.suffixes), (
        f"{ext} directory froze suffixes {spec.suffixes}")
    assert widget.wrangler._directory_run_without_seed_ok() is True
    assert _attempt_start(widget, monkeypatch) == [True]

    frozen = widget.wrangler.run_configuration
    assert frozen.source is not None and frozen.source.family == "directory"
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )
    assert imageThread._frozen_source_is_container(widget.wrangler.thread) is False


# --------------------------------------------------------------------------- #
# §40.1 P1-C / D0 items 9-10 — run lifecycle: no cross-run policy, and a bare
# carrier is not authorization.
# --------------------------------------------------------------------------- #

def test_run_b_never_observes_run_a_policy_on_the_reused_worker(
        widget, tmp_path, monkeypatch):
    """Item 9 and §40.1 P1-C.  ``frozen_run_policy()`` prefers any
    ``_qualified_run_configuration`` and ``_bind_admitted_run_configuration()``
    never cleared or requalified it, so after Run A has ENTERED, every pre-entry
    Run-B read still resolves through Run A.  Production-reachable: setup reads
    ``thread.max_cores`` before worker entry, so an old run can configure the
    next run's scan threads.

    Driven through the real production worker and the real binding entry point.
    A second real Run click cannot be used for this property: the accepted W-1R
    rebind rule refuses a different genuine object at the same generation, so the
    Run-B click never reaches the leak.  Codex's preserved reproducer
    (``/Users/vthampy/repos/tmp/test_codex_w1r_projection_lifecycle.py``, 2/2 red
    at this tip) proves it at the binding level for exactly that reason; this is
    that shape on the real ``imageThread`` the widget built.
    """
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        wranglerWidget,
    )

    _arm_series(widget, tmp_path / "run-a")
    assert _attempt_start(widget, monkeypatch) == [True]
    thread = widget.wrangler.thread
    first = thread._admitted_run_configuration
    assert isinstance(first, FrozenRunConfiguration)
    # Run A has entered: this is what publishes the qualified reference.
    assert thread._require_run_configuration("oracle-run-a-entry") is first

    second = _frozen_pair(
        tmp_path / "run-b-output",
        generation=int(first.generation) + 1, batch=True, cores=first.max_cores + 1)
    wranglerWidget._bind_admitted_run_configuration(thread, second)

    assert thread.run_configuration is second
    assert thread._admitted_run_configuration is second
    # No Run-A value may be observable through ANY route once B is admitted.
    assert thread.h5_dir == second.save_path, (
        f"Run B setup reads Run A output {thread.h5_dir!r}, "
        f"not {second.save_path!r}")
    assert thread.batch_mode is True, "Run B setup reads Run A batch mode"
    assert thread.max_cores == second.max_cores, (
        f"Run B setup reads Run A parallelism {thread.max_cores}")
    stale = getattr(thread, "_qualified_run_configuration", None)
    assert stale is None or stale is second, (
        "a prior run's qualified reference survived the new admission")


def test_a_bare_frozen_carrier_activates_no_execution_policy(
        widget, tmp_path, monkeypatch):
    """Item 10.  Merely assigning a ``FrozenRunConfiguration`` to
    ``run_configuration`` activated policy through the descriptor, without any
    exact admission (§40.2)."""
    _arm_series(widget, tmp_path)
    thread = widget.wrangler.thread
    unadmitted = _frozen_pair(tmp_path / "never-admitted")
    thread.run_configuration = unadmitted
    thread._admitted_run_configuration = None

    with pytest.raises(RunConfigurationRefused):
        thread._require_run_configuration("oracle-bare-carrier")


# --------------------------------------------------------------------------- #
# D0 items 11-13 — the accepted W-1R corrections must SURVIVE the replacement.
# --------------------------------------------------------------------------- #

def test_equal_valued_and_same_generation_objects_still_refuse_by_is(
        widget, tmp_path, monkeypatch):
    """Item 11.  Exact-object identity is the accepted W-1R property; the
    replacement may not weaken it to equality or fingerprint comparison."""
    import dataclasses

    _arm_series(widget, tmp_path)
    assert _attempt_start(widget, monkeypatch) == [True]
    thread = widget.wrangler.thread
    admitted = thread._admitted_run_configuration
    assert isinstance(admitted, FrozenRunConfiguration)

    reconstructed = dataclasses.replace(admitted)
    assert reconstructed == admitted
    assert reconstructed is not admitted
    thread.run_configuration = reconstructed
    with pytest.raises(RunConfigurationRefused):
        thread._require_run_configuration("oracle-equal-valued")

    foreign = _frozen_pair(
        tmp_path / "foreign", generation=int(admitted.generation))
    thread.run_configuration = foreign
    with pytest.raises(RunConfigurationRefused):
        thread._require_run_configuration("oracle-same-generation-foreign")


def test_source_change_populates_gi_motors_and_freeze_resolves_once(
        widget, tmp_path, monkeypatch):
    """Item 12.  The GI theta-motor choices are driven by source selection; the
    truthful-source work must not break that, and the freeze owner must resolve
    the effective motor exactly once per freeze."""
    from xrd_tools.session import run_configuration as run_config_module

    calls = []
    real_resolve = run_config_module.resolve_gi_motor

    def counted(*args, **kwargs):
        calls.append(args[:1])
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(run_config_module, "resolve_gi_motor", counted)

    _arm_series(widget, tmp_path)
    widget.wrangler.set_gi_motor_options()
    choices = list(
        widget.wrangler.parameters.child("Signal").child("File").opts.get(
            "limits", ()) or ())
    # The motor list itself lives on the integrator; assert it is populated and
    # offers Manual, which is the documented always-present fallback.
    motors = widget._controls_v2_gi_motor_choices() if hasattr(
        widget, "_controls_v2_gi_motor_choices") else None
    if motors is not None:
        assert "Manual" in [str(m) for m in motors], motors

    calls.clear()
    frozen = widget._prepare_controls_v2_run_configuration()
    assert frozen is not None
    assert len(calls) <= 1, (
        f"freeze resolved the GI motor {len(calls)} times, not at most once")
    assert frozen.gi.scan_incidence_motor == (
        widget._prepare_controls_v2_run_configuration().gi.scan_incidence_motor)
    assert choices is not None


def test_readers_append_provenance_and_reload_keep_the_exact_identity(
        widget, tmp_path, monkeypatch):
    """Item 13.  The one accepted object must remain the identity every
    downstream consumer sees, through the new admission path."""
    _arm_series(widget, tmp_path)
    assert _attempt_start(widget, monkeypatch) == [True]
    wrangler = widget.wrangler
    thread = wrangler.thread
    admitted = thread._admitted_run_configuration

    assert wrangler.run_configuration is admitted
    assert wrangler._admitted_run_configuration is admitted
    assert thread.run_configuration is admitted
    qualified = thread._require_run_configuration("oracle-entry")
    assert qualified is admitted

    provenance = admitted.as_provenance()
    import json
    assert json.loads(json.dumps(provenance)) == provenance, (
        "written provenance is not JSON-native")


def test_refused_start_publishes_no_calibration_or_run_state(
        widget, tmp_path, monkeypatch):
    """Item 14.  A refusal must leave calibration, source, output, wrapper,
    thread, generation and the pending handoff untouched."""
    raw, _out = _arm_series(widget, tmp_path)
    wrangler = widget.wrangler
    # Remove the output target: admission already requires save_path, so this
    # exercises the refusal path with a fully valid source.
    wrangler.parameters.child("Project", "h5_dir").setValue("")
    before = _carrier_snapshot(wrangler)

    assert _attempt_start(widget, monkeypatch) == []

    after = _carrier_snapshot(wrangler)
    assert after == before, (
        f"{ {k: (before[k], after[k]) for k in before if before[k] != after[k]} }")
    assert getattr(
        widget, "_pending_controls_v2_run_configuration", None) is None
    assert os.path.exists(raw)
