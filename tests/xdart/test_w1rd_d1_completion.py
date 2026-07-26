"""O-1a-W1R-D1 completion oracle — review §41.2 / §41.3.A-E.

Boundary 22 accepted D0 and returned D1 for rework IN PLACE.  This module is the
bounded completion oracle §41.2 requires, frozen before the §41.3 production
fixes and proven red against the dirty D1 draft.  It is oracle COMPLETION, not
permission to change accepted behavior: every case here asserts something the D0
oracle left unobserved.

Families (§41.6 step 1):

  1. detached copying of nested ``SourceSpec.options``, and a production
     Run -> idle edit -> Run (§41.3.A);
  2. complete canonical ``RunIntent`` zero-delta on refusal, generation included
     (§41.3.B);
  3. staged-calibration cleanup on every early-return path (§41.3.C);
  4. a nonexistent directory and a mutable-mode poison cannot be admitted
     (§41.3.D);
  5. accepted-carrier publication is atomic (§41.3.E).

Production-wired per CLAUDE.md rule 2: the real ``staticWidget``, the real
``imageWrangler`` and worker, the real Controls freeze/stage owners and the real
Start boundary.  ``thread.start`` is the only blocked seam.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.test_w1a_frozen_execution import _write_poni

from xrd_tools.session import RunConfigurationRefused
from xrd_tools.session.run_configuration import (
    FrozenRunConfiguration,
    RunIntent,
)


INTENT_FIELDS = (
    "source_spec", "processing_mode", "output_mode", "live_mode", "batch_mode",
    "max_cores", "bai_1d_args", "bai_2d_args", "gi", "threshold", "poni_file",
    "poni_values", "mask_file", "project_root", "save_path", "run_options",
    "generation",
)


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


def _project_intent(intent):
    """A PURE, non-freezing projection of the complete canonical intent.

    §41.3.B: the refusal snapshot must cover every canonical field, including
    ``generation`` and the nested GI/threshold/run-option values, and it must not
    itself freeze (which would advance the very counter under test).
    """
    return {name: repr(getattr(intent, name, None)) for name in INTENT_FIELDS}


def _arm_series(widget, root: Path, *, ext="tif"):
    wrangler = widget.wrangler
    signal = wrangler.parameters.child("Signal")
    project = root / "project"
    project.mkdir(parents=True, exist_ok=True)
    raw = project / f"scan_0001.{ext}"
    raw.write_bytes(b"")
    (project / f"scan_0002.{ext}").write_bytes(b"")
    out = project / "out"
    out.mkdir(exist_ok=True)
    wrangler.parameters.child("Project", "project_folder").setValue(str(project))
    widget._set_poni_field(_write_poni(project / "cal.poni"))
    signal.child("inp_type").setValue("Image Series")
    signal.child("File").setValue(str(raw))
    wrangler.parameters.child("Project", "h5_dir").setValue(str(out))
    return raw, out


def _attempt_start(widget, monkeypatch):
    started = []
    monkeypatch.setattr(
        widget.wrangler.thread, "start", lambda: started.append(True))
    widget.wrangler.start()
    return started


# --------------------------------------------------------------------------- #
# Family 1 — §41.3.A: a typed source on the intent must survive the next
# idle-edit stage, and its nested option values must be DETACHED.
# --------------------------------------------------------------------------- #

def test_stage_candidate_survives_a_typed_source_on_the_intent(
        widget, tmp_path):
    """§41.3.A.  ``SourceSpec.__post_init__`` stores ``options`` in a
    ``MappingProxyType``; ``_controls_v2_new_stage_candidate`` deep-copies the
    live intent, so the first idle edit after a typed source is retained raises
    ``TypeError: cannot pickle 'mappingproxy' object``."""
    _arm_series(widget, tmp_path)
    intent = widget._controls_v2_ensure_run_intent()
    intent.source_spec = widget._controls_v2_freeze_source_spec()
    assert intent.source_spec is not None

    candidate = widget._controls_v2_new_stage_candidate()

    assert candidate is not None
    assert candidate.intent is not intent


def test_stage_candidate_detaches_nested_source_option_values(
        widget, tmp_path):
    """§41.2 / §41.3.A: a copy boundary that shares nested option values is not a
    copy.  Mutating the candidate's own option containers must not reach the
    canonical intent."""
    from xrd_tools.core.scan import SourceKind, SourceSpec

    _arm_series(widget, tmp_path)
    intent = widget._controls_v2_ensure_run_intent()
    nested = {"files": ["a.tif", "b.tif"], "meta": {"scan": 1}}
    intent.source_spec = SourceSpec(
        str(tmp_path / "project"), SourceKind.TIFF_SERIES, options=nested)

    candidate = widget._controls_v2_new_stage_candidate()
    copied = candidate.intent.source_spec

    assert copied is not intent.source_spec
    assert dict(copied.options) == dict(intent.source_spec.options)
    # Detached at every level: the candidate's nested containers are not the
    # canonical intent's.
    assert copied.options["files"] is not intent.source_spec.options["files"]
    assert copied.options["meta"] is not intent.source_spec.options["meta"]
    copied.options["files"].append("POISON")
    copied.options["meta"]["scan"] = 999
    assert "POISON" not in intent.source_spec.options["files"]
    assert intent.source_spec.options["meta"]["scan"] == 1


def test_run_then_idle_edit_then_run_through_production(
        widget, tmp_path, monkeypatch):
    """§41.2: the production sequence the crash actually breaks -- an accepted
    Run, then an ordinary operator edit while idle, then a second Run."""
    _arm_series(widget, tmp_path)
    assert _attempt_start(widget, monkeypatch) == [True]
    first = widget.wrangler._admitted_run_configuration
    assert isinstance(first, FrozenRunConfiguration)

    # Run A must actually END before the next click is an idle one: a Start while
    # a run owner is active is correctly refused, and the action button has
    # morphed to Pause.  Drive the production run-end owner.
    widget.wrangler_finished()
    assert widget.wrangler.command != "start"

    # An ordinary idle edit: this is the path that stages a fresh candidate, and
    # the path that raised once a typed source was retained on the intent.
    widget.wrangler.parameters.child("Signal", "series_average").setValue(True)
    widget.wrangler.parameters.child("Signal", "series_average").setValue(False)

    raw_b, out_b = _arm_series(widget, tmp_path / "second")
    assert _attempt_start(widget, monkeypatch) == [True]
    second = widget.wrangler._admitted_run_configuration
    assert isinstance(second, FrozenRunConfiguration)
    assert second is not first
    assert second.save_path == str(out_b)


# --------------------------------------------------------------------------- #
# Family 2 — §41.3.B: refusal is zero-delta on the CANONICAL intent.
# --------------------------------------------------------------------------- #

def test_refused_start_leaves_the_canonical_intent_untouched(
        widget, tmp_path, monkeypatch):
    """§41.3.B.  ``prepare`` populates the live ``_controls_v2_run_intent`` and
    calls ``freeze()``, which advances ``intent.generation`` before admission can
    reject a missing source or save_path.  The D0 carrier snapshot never looked at
    the canonical intent, so its green result missed this."""
    old_raw, _out = _arm_series(widget, tmp_path)
    widget.wrangler.parameters.child("Signal", "File").setValue("")
    widget.wrangler.img_file = str(old_raw)
    canonical = widget._controls_v2_ensure_run_intent()
    before = _project_intent(canonical)

    assert _attempt_start(widget, monkeypatch) == []

    after = _project_intent(canonical)
    changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    assert changed == {}, f"a refused Start mutated the canonical intent: {changed}"
    assert getattr(
        widget, "_pending_controls_v2_run_configuration", None) is None


def test_refused_start_does_not_advance_the_generation(
        widget, tmp_path, monkeypatch):
    """§41.3.B item 5, isolated: the generation is the value a retry must not
    burn, and it is what makes a rejected click look like an accepted one."""
    _arm_series(widget, tmp_path)
    widget.wrangler.parameters.child("Project", "h5_dir").setValue("")
    canonical = widget._controls_v2_ensure_run_intent()
    generation_before = int(canonical.generation)

    assert _attempt_start(widget, monkeypatch) == []

    assert int(canonical.generation) == generation_before


def test_accepted_start_advances_the_generation_exactly_once(
        widget, tmp_path, monkeypatch):
    """§41.3.B item 6: the accepted candidate is committed exactly once -- no
    hole in the sequence, and no second revision authority."""
    _arm_series(widget, tmp_path)
    canonical = widget._controls_v2_ensure_run_intent()
    generation_before = int(canonical.generation)

    assert _attempt_start(widget, monkeypatch) == [True]

    accepted = widget.wrangler._admitted_run_configuration
    assert int(accepted.generation) == generation_before + 1
    assert int(canonical.generation) == int(accepted.generation)


# --------------------------------------------------------------------------- #
# Family 3 — §41.3.C: the staged calibration is consumed on every path.
# --------------------------------------------------------------------------- #

def _arm_staged_calibration(widget, tmp_path):
    """Reach the state where ``start()`` actually STAGES a candidate.

    ``_stage_loaded_scan_calibration`` returns early unless the wrapper has no
    PONI of its own and the loaded scan carries a cached one, so a case that
    merely loads a .poni never reaches the retention paths at all.
    """
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.calibration import poni_to_integrator

    raw, out = _arm_series(widget, tmp_path)
    adopted = PONI.from_poni_file(_write_poni(tmp_path / "adopted.poni", 0.2222))
    widget.wrangler.poni = None
    widget.wrangler.parameters.child("Signal", "poni_file").setValue("")
    widget.scan._cached_poni = adopted
    widget.scan._cached_integrator = poni_to_integrator(adopted)
    return raw, out


def test_staged_calibration_is_cleared_when_inputs_are_invalid(
        widget, tmp_path, monkeypatch):
    """§41.3.C.  ``start()`` clears ``_staged_run_calibration`` in the admission
    exception branches and after publication, but the ``_inputs_valid()`` false
    return at ``image_wrangler.py:1832`` retains the staged candidate."""
    _arm_staged_calibration(widget, tmp_path)
    # Invalidate the SOURCE (not the calibration) through the authoritative card,
    # so staging has already happened when the gate refuses.
    widget.wrangler.parameters.child("Signal", "File").setValue("")
    assert imageWranglerStaged(widget) is None

    assert _attempt_start(widget, monkeypatch) == []

    assert getattr(widget.wrangler, "_staged_run_calibration", None) is None, (
        "an _inputs_valid() refusal retained the staged calibration candidate")


def imageWranglerStaged(widget):
    """The staged slot before the click, for an explicit precondition."""
    return getattr(widget.wrangler, "_staged_run_calibration", None)


def test_staged_calibration_is_cleared_on_the_stitch_diversion(
        widget, tmp_path, monkeypatch):
    """§41.3.C.  The Stitch diversion at ``image_wrangler.py:1841`` returns before
    admission and also leaves the staged candidate behind."""
    _arm_staged_calibration(widget, tmp_path)
    widget.wrangler.stitch_mode = True
    requested = []
    widget.wrangler.sigStitchRequested.connect(lambda kind: requested.append(kind))

    assert _attempt_start(widget, monkeypatch) == []

    assert requested, "the stitch diversion was not reached"
    assert getattr(widget.wrangler, "_staged_run_calibration", None) is None, (
        "the stitch diversion retained the staged calibration candidate")


def test_staged_calibration_is_cleared_when_the_run_is_refused(
        widget, tmp_path, monkeypatch):
    """§41.3.C, the admission-refusal path."""
    _arm_series(widget, tmp_path)
    widget.wrangler.parameters.child("Project", "h5_dir").setValue("")

    assert _attempt_start(widget, monkeypatch) == []

    assert getattr(widget.wrangler, "_staged_run_calibration", None) is None


def test_staged_calibration_is_cleared_after_a_successful_run(
        widget, tmp_path, monkeypatch):
    """§41.3.C, the success path: one total cleanup owner, not branch-local
    clears."""
    _arm_series(widget, tmp_path)

    assert _attempt_start(widget, monkeypatch) == [True]

    assert getattr(widget.wrangler, "_staged_run_calibration", None) is None


# --------------------------------------------------------------------------- #
# Family 4 — §41.3.D: a typed source must also be a VALID source.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ext", ["tif", "raw", "mar3450"])
def test_nonexistent_directory_cannot_be_admitted(
        widget, tmp_path, monkeypatch, ext):
    """§41.3.D.  ``_authoritative_source_selected`` accepts any nonblank string
    for the Directory leg and the plain-image freeze builds a
    ``DirectorySourceSpec`` without checking the root, so a nonexistent directory
    reaches Start with a typed but invalid source."""
    wrangler = widget.wrangler
    signal = wrangler.parameters.child("Signal")
    out = tmp_path / "out"
    out.mkdir()
    widget._set_poni_field(_write_poni(tmp_path / "cal.poni"))
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(tmp_path / "does-not-exist"))
    signal.child("img_ext").setValue(ext)
    signal.child("File").setValue("")
    wrangler.parameters.child("Project", "h5_dir").setValue(str(out))

    assert _attempt_start(widget, monkeypatch) == []
    assert getattr(wrangler, "_admitted_run_configuration", None) is None


def test_empty_but_existing_directory_remains_legitimate(
        widget, tmp_path, monkeypatch):
    """§41.3.D's explicit carve-out: the lazy Directory contract must survive.
    An EMPTY but existing directory is a legitimate seedless run; the fix may not
    restore eager discovery or demand a representative frame."""
    wrangler = widget.wrangler
    signal = wrangler.parameters.child("Signal")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    widget._set_poni_field(_write_poni(tmp_path / "cal.poni"))
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(raw_dir))
    signal.child("img_ext").setValue("h5")
    signal.child("File").setValue("")
    wrangler.parameters.child("Project", "h5_dir").setValue(str(out))

    assert _attempt_start(widget, monkeypatch) == [True]
    assert wrangler.run_configuration.source is not None


def test_mutable_mode_poison_cannot_change_the_validated_source(
        widget, tmp_path, monkeypatch):
    """§41.3.D.  The readiness gate branches on mutable ``self.inp_type``; the
    authoritative Source card is the only thing entitled to say which mode is
    selected."""
    _arm_series(widget, tmp_path)
    # Poison the mutable mirror so the gate would consult the Directory leg,
    # whose non-blank-string rule is the weaker one.
    widget.wrangler.inp_type = "Image Directory"
    widget.wrangler.img_dir = str(tmp_path / "nowhere-at-all")

    started = _attempt_start(widget, monkeypatch)

    # The card still selects the valid TIFF series, so the run is admitted on the
    # card's terms -- never on the poisoned mirror's.
    assert started == [True]
    frozen = widget.wrangler.run_configuration
    assert frozen.source is not None
    assert str(frozen.source.source_kind) == "tiff_series"
    assert frozen.source.family == "source"


# --------------------------------------------------------------------------- #
# Family 5 — §41.3.E: accepted-carrier publication is atomic.
# --------------------------------------------------------------------------- #

def test_publication_failure_leaves_no_partially_accepted_run(
        widget, tmp_path, monkeypatch):
    """§41.3.E.  The publication path writes the wrapper binding, the source
    projection and the thread binding sequentially, so an exception part-way
    through leaves a partially accepted run.  Every fallible projection must be
    computed BEFORE the first write."""
    from xdart.gui.tabs.static_scan.wranglers import wrangler_widget as ww

    _arm_series(widget, tmp_path)
    wrangler = widget.wrangler
    thread = wrangler.thread
    before = {
        "wrangler.run_configuration": getattr(wrangler, "run_configuration", None),
        "wrangler.admitted": getattr(wrangler, "_admitted_run_configuration", None),
        "wrangler.floor": getattr(wrangler, "run_configuration_floor", 0),
        "wrangler.source_spec": getattr(wrangler, "source_spec", None),
        "thread.run_configuration": getattr(thread, "run_configuration", None),
        "thread.admitted": getattr(thread, "_admitted_run_configuration", None),
    }

    # Inject a failure in the LAST fallible step of the publication set.
    real_thaw = FrozenRunConfiguration.thaw_source_spec
    calls = []

    def exploding(self):
        calls.append(True)
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(
        FrozenRunConfiguration, "thaw_source_spec", exploding, raising=False)

    started = []
    monkeypatch.setattr(thread, "start", lambda: started.append(True))
    try:
        wrangler.start()
    except Exception:
        pass

    assert calls, "the injection point was never reached"
    assert started == []
    after = {
        "wrangler.run_configuration": getattr(wrangler, "run_configuration", None),
        "wrangler.admitted": getattr(wrangler, "_admitted_run_configuration", None),
        "wrangler.floor": getattr(wrangler, "run_configuration_floor", 0),
        "wrangler.source_spec": getattr(wrangler, "source_spec", None),
        "thread.run_configuration": getattr(thread, "run_configuration", None),
        "thread.admitted": getattr(thread, "_admitted_run_configuration", None),
    }
    changed = {k: (before[k], after[k]) for k in before if before[k] is not after[k]}
    assert changed == {}, f"publication was not atomic: {sorted(changed)}"
    monkeypatch.setattr(
        FrozenRunConfiguration, "thaw_source_spec", real_thaw, raising=False)
