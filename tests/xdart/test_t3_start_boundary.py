"""O-1a-T3 (§9.10 steps 2 + 5) — the composed Start boundary.

ONE production active/stopping predicate consulted at the TOP of
``imageWrangler.start()`` and defensively in
``_prepare_controls_v2_run_configuration``; on refusal the complete §31.3-item-4
preservation set is untouched and NO frozen object is published.  The successful
order is::

    refuse active/stopping
    -> harvest revision winners
    -> pure stage
    -> checked commit
    -> ONE freeze
    -> publish the SAME frozen object to every owner
    -> start

This module is the FROZEN acceptance oracle for the six named T-3 families
(a)-(f) plus the GI one-policy deletion.  It deliberately does NOT re-test the
landed transaction engine: engine behaviour is owned by the retained
Boundary 6-11 modules (``test_t2_5r1_bound_carriers``,
``test_t2_5r2_strict_writes``, ``test_t2_5r3_residuals``,
``test_t2_5r4_cleanup_totality``, ``test_t2_5r5_registry_ownership``,
``test_t2_5r6_diagnostic_reachability``, ``test_t2_5_atomic_engine``,
``test_t2_5r_atomicity_depth``) and the real-PONI compound carrier is owned by
``test_controls_panel::test_t2_1_compound_poni_*``.  Every widget here is the
REAL ``staticWidget`` with its real wrangler, real parameter tree, real
``PONI``, and the real transaction engine (rule 2: no fakes on the seam).
"""

from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtWidgets


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


def _write_poni(path, dist):
    path.write_text(
        f"Distance: {dist}\nPoni1: 0.01\nPoni2: 0.02\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n"
    )
    return str(path)


def _preservation_snapshot(widget):
    """The §31.3 item-4 refusal preservation set, by identity + value.

    RunIntent, display projection, legacy parameters, journal, wrangler/thread
    frozen carriers, source owner/index, generation, and fingerprint.
    """
    import copy

    wrangler = widget.wrangler
    thread = getattr(wrangler, "thread", None)
    intent = widget._controls_v2_ensure_run_intent()
    scan = getattr(widget, "scan", None)
    return {
        "intent_generation": int(intent.generation),
        "intent_output_mode": str(intent.output_mode),
        "intent_poni_file": str(intent.poni_file),
        "intent_poni_values": copy.deepcopy(intent.poni_values),
        "intent_bai_1d": copy.deepcopy(intent.bai_1d_args),
        "intent_gi": copy.deepcopy(intent.gi),
        "intent_source_spec": copy.deepcopy(intent.source_spec),
        "legacy": dict(widget._controls_v2_committed_legacy_values()),
        "journal": copy.deepcopy(widget._controls_v2_edit_journal_dict()),
        "wrangler_run_configuration": getattr(
            wrangler, "run_configuration", None),
        "thread_run_configuration": getattr(thread, "run_configuration", None),
        "pending": getattr(
            widget, "_pending_controls_v2_run_configuration", None),
        "wrangler_source_spec": getattr(wrangler, "source_spec", None),
        "wrangler_poni": getattr(wrangler, "poni", None),
        "thread_poni": getattr(thread, "poni", None),
        "wrangler_command": getattr(wrangler, "command", None),
        "thread_command": getattr(thread, "command", None),
        "source_index": getattr(
            widget, "_controls_v2_source_index_session", None),
        "scan_bai_1d": copy.deepcopy(getattr(scan, "bai_1d_args", None)),
        "scan_gi_config": copy.deepcopy(getattr(scan, "gi_config", None)),
    }


def _assert_preserved(before, after):
    identity_keys = (
        "wrangler_run_configuration",
        "thread_run_configuration",
        "pending",
        "wrangler_poni",
        "thread_poni",
        "source_index",
    )
    for key in identity_keys:
        assert before[key] is after[key], f"{key} was replaced on refusal"
    for key, value in before.items():
        if key in identity_keys:
            continue
        assert after[key] == value, f"{key} changed on refusal"


def _establish_prior_frozen(widget):
    """Stage one prior frozen object.

    O-1a-W1R-D1 (review §41.3.B / §41.4 item 2): preparation now only STAGES the
    tentative candidate -- publishing there is what made the zero-delta refusal
    claim false -- so this returns the staged object and the carriers stay
    unpublished until exact admission.
    """
    frozen = widget._prepare_controls_v2_run_configuration()
    assert frozen is not None
    return frozen


def _make_stopping_wrangler(widget, monkeypatch):
    """The exact fast-Start window: the worker thread is still running while the
    host latch and the wrangler run phase already read idle."""
    monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: True)
    monkeypatch.setattr(widget.wrangler, "_run_phase", "idle", raising=False)
    assert widget._run_active is False


def _block_thread_start(widget, monkeypatch):
    started = []
    monkeypatch.setattr(
        widget.wrangler.thread, "start", lambda: started.append(True))
    return started


# --------------------------------------------------------------------------- #
# Family (d) — ONE active/stopping predicate; §9.10 tests 8 and 9.
# --------------------------------------------------------------------------- #

def test_predicate_reports_active_for_the_wrangler_stopping_window(
        widget, monkeypatch):
    """(d) The predicate must see a still-running worker even when the run phase
    and the host latch already read idle — the r1 fast-Start window."""
    assert widget._controls_v2_active_run_owner() is None
    _make_stopping_wrangler(widget, monkeypatch)
    assert widget._controls_v2_active_run_owner() == "wrangler"


def test_predicate_reports_active_for_reintegration(widget, monkeypatch):
    """(d) Integration/reintegration is one of the four required owners."""
    thread = widget.integratorTree.integrator_thread
    monkeypatch.setattr(thread, "isRunning", lambda: True)
    assert widget._controls_v2_active_run_owner() == "reintegration"


def test_predicate_reports_active_for_stitch(widget, monkeypatch):
    """(d) Stitch is one of the four required owners."""
    monkeypatch.setattr(widget.stitch_thread, "isRunning", lambda: True)
    assert widget._controls_v2_active_run_owner() == "stitch"


def test_predicate_reports_active_for_the_host_run_state_latch(widget):
    """(d) The host run-state latch is one of the four required owners."""
    widget._enter_run_state()
    assert widget._controls_v2_active_run_owner() == "run"


def test_predicate_reports_idle_when_every_owner_is_idle(widget):
    """(d) The predicate is not simply always-on."""
    assert widget._controls_v2_active_run_owner() is None
    assert widget._controls_v2_run_active() is False


def test_fast_start_with_an_empty_journal_preserves_everything(
        widget, monkeypatch):
    """§9.10 test 8 — fast Start, EMPTY journal: every sentinel in the §31.3
    item-4 preservation set is unchanged and no worker starts."""
    _establish_prior_frozen(widget)
    assert not widget._controls_v2_edit_journal_dict()
    _make_stopping_wrangler(widget, monkeypatch)
    started = _block_thread_start(widget, monkeypatch)

    before = _preservation_snapshot(widget)
    widget.wrangler.start()
    after = _preservation_snapshot(widget)

    _assert_preserved(before, after)
    assert started == []


def test_fast_start_with_a_non_empty_journal_preserves_sentinels_and_journal(
        widget, monkeypatch, tmp_path):
    """§9.10 test 9 — fast Start, NON-EMPTY journal: sentinels AND the complete
    revisioned journal survive; no edit is consumed and nothing is published."""
    _establish_prior_frozen(widget)
    widget._enter_run_state()
    widget._on_controls_v2_field_changed(("Int1D", "points"), 1234)
    journal = dict(widget._controls_v2_edit_journal_dict())
    assert journal, "the run-active edit must be journaled as deferred"
    widget._exit_run_state(widget._new_projection_receipt())

    _make_stopping_wrangler(widget, monkeypatch)
    started = _block_thread_start(widget, monkeypatch)

    before = _preservation_snapshot(widget)
    widget.wrangler.start()
    after = _preservation_snapshot(widget)

    _assert_preserved(before, after)
    assert widget._controls_v2_edit_journal_dict() == journal
    assert started == []


def test_fast_start_publishes_no_frozen_object_and_never_sets_command(
        widget, monkeypatch):
    """(d) The r1 defect was publication BEFORE refusal.  A refused Start may
    not touch the wrangler/thread frozen carriers, the pending configuration,
    the intent generation, or the run command."""
    prior = _establish_prior_frozen(widget)
    widget.wrangler.command = "stop"
    widget.wrangler.thread.command = "stop"
    generation = int(widget._controls_v2_ensure_run_intent().generation)
    _make_stopping_wrangler(widget, monkeypatch)
    _block_thread_start(widget, monkeypatch)

    widget.wrangler.start()

    # §41.4 item 2: preparation publishes nothing, so the refusal must leave the
    # carriers UNPUBLISHED and the staged object intact.
    assert widget.wrangler.run_configuration is None
    assert widget.wrangler.thread.run_configuration is None
    assert widget._pending_controls_v2_run_configuration is prior
    assert int(widget._controls_v2_ensure_run_intent().generation) == generation
    assert widget.wrangler.command == "stop"
    assert widget.wrangler.thread.command == "stop"


def test_prepare_defensively_refuses_while_a_run_owner_is_active(
        widget, monkeypatch):
    """(d) The predicate is consulted defensively inside preparation too: a
    direct call while an owner is active is a typed refusal that freezes and
    publishes NOTHING."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        RunOwnerActiveError,
    )

    prior = _establish_prior_frozen(widget)
    _make_stopping_wrangler(widget, monkeypatch)
    before = _preservation_snapshot(widget)

    with pytest.raises(RunOwnerActiveError):
        widget._prepare_controls_v2_run_configuration()

    after = _preservation_snapshot(widget)
    _assert_preserved(before, after)
    # §41.4 item 2: nothing was published before the refusal, so nothing changed.
    assert widget.wrangler.run_configuration is None
    assert widget._pending_controls_v2_run_configuration is prior


# --------------------------------------------------------------------------- #
# Family (a) — sticky Overwrite latch (regression guard).
# --------------------------------------------------------------------------- #

def test_output_mode_is_re_read_from_the_control_every_run(widget):
    """(a) ``output_mode`` is a per-Run read of the visible control."""
    widget.controls.set_write_mode("Append")
    assert widget._prepare_controls_v2_run_configuration().output_mode == "Append"
    widget.controls.set_write_mode("Overwrite")
    assert (widget._prepare_controls_v2_run_configuration().output_mode
            == "Overwrite")


def test_accepted_overwrite_does_not_survive_a_flip_back_to_append(widget):
    """(a) The r1 destructive-output regression: a modal-accepted Overwrite must
    NOT latch once the user returns the visible control to Append."""
    widget.controls.set_write_mode("Append")
    widget._prepare_controls_v2_run_configuration()
    # An accepted Append-mismatch modal flips the ACTIVE write mode.
    widget.wrangler._set_active_write_mode("Overwrite")
    assert widget.controls.write_mode() == "Overwrite"
    assert (widget._prepare_controls_v2_run_configuration().output_mode
            == "Overwrite")

    # The user then clicks the visible control back to Append.
    widget.controls.set_write_mode("Append")
    frozen = widget._prepare_controls_v2_run_configuration()

    assert widget.controls.write_mode() == "Append"
    assert frozen.output_mode == "Append"
    assert widget._controls_v2_ensure_run_intent().output_mode == "Append"


# --------------------------------------------------------------------------- #
# Family (b) — the Append comparison uses the CANDIDATE, not the previous run.
# --------------------------------------------------------------------------- #

def test_append_mismatch_compares_the_candidate_not_the_previous_run(
        widget, monkeypatch, tmp_path):
    """(b) ``_append_config_mismatch_details`` must build ``current`` from the
    staged CANDIDATE configuration.  With a PREVIOUS run's frozen object still
    on the wrangler (1D points 100) and the live Controls now at 200, the
    comparison against a processed target stored at 100 must MISMATCH."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
    )

    poni_path = _write_poni(tmp_path / "cal.poni", 0.10)
    raw_path = tmp_path / "scan_0001.tif"
    raw_path.write_bytes(b"")
    target = tmp_path / "scan.nxs"
    target.write_bytes(b"sentinel")

    widget._set_poni_field(poni_path)
    widget.wrangler.parameters.child("Project", "h5_dir").setValue(str(tmp_path))
    # O-1a-W1R-D1 (review §40.1 P1-A, §40.3 D1 items 1-2): arm the
    # AUTHORITATIVE Source card.  Setting only ``img_file`` is the shorthand
    # shape §40.1 named -- a truthful ``get_img_fname`` sync clears a cursor the
    # Source card does not back, and admission now requires a typed source.
    _signal = widget.wrangler.parameters.child("Signal")
    _signal.child("inp_type").setValue("Image Series")
    _signal.child("File").setValue(str(raw_path))
    widget.wrangler.img_file = str(raw_path)
    widget.controls.set_write_mode("Append")

    # The processed append target reports 1D points 100 (stored config).
    widget.scan.data_file = str(target)
    widget.scan.reduction_config = {
        "gi": False,
        "bai_1d_args": {"unit": "q_A^-1", "numpoints": 100},
        "bai_2d_args": {"unit": "q_A^-1"},
    }
    widget.scan._display_reduction_config = dict(widget.scan.reduction_config)

    # Freeze a PREVIOUS run at 100, then move the live Controls to 200.
    widget._on_controls_v2_field_changed(("Int1D", "points"), 100)
    # §41.4 item 2: establish the previous ACCEPTED object through the real
    # admission seam.  Preparation stages; only admission publishes.
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        wranglerWidget,
    )
    stale = wranglerWidget._admit_run_configuration(
        widget.wrangler, "t3-prior-accepted")
    assert stale is widget.wrangler.run_configuration
    assert stale is widget.wrangler._admitted_run_configuration
    widget._on_controls_v2_field_changed(("Int1D", "points"), 200)

    check, processed, current = imageWrangler._append_config_mismatch_details(
        widget.wrangler)

    assert processed is not None and current is not None
    assert processed.npt_1d == 100
    assert current.npt_1d == 200, (
        "the comparison used the previous run's frozen configuration")
    assert check is not None and check.ok is False


def test_accepted_append_modal_overwrite_reaches_the_frozen_configuration(
        widget, monkeypatch, tmp_path):
    """(b) The modal decision must be applied to the candidate BEFORE the freeze:
    an accepted Replace/Overwrite has to be visible on the frozen object the
    writer consumes, not only on the mutable thread carrier."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
    )

    poni_path = _write_poni(tmp_path / "cal.poni", 0.10)
    raw_path = tmp_path / "scan_0001.tif"
    raw_path.write_bytes(b"")
    target = tmp_path / "scan.nxs"
    target.write_bytes(b"sentinel")

    widget._set_poni_field(poni_path)
    widget.wrangler.parameters.child("Project", "h5_dir").setValue(str(tmp_path))
    # O-1a-W1R-D1 (review §40.1 P1-A, §40.3 D1 items 1-2): arm the
    # AUTHORITATIVE Source card.  Setting only ``img_file`` is the shorthand
    # shape §40.1 named -- a truthful ``get_img_fname`` sync clears a cursor the
    # Source card does not back, and admission now requires a typed source.
    _signal = widget.wrangler.parameters.child("Signal")
    _signal.child("inp_type").setValue("Image Series")
    _signal.child("File").setValue(str(raw_path))
    widget.wrangler.img_file = str(raw_path)
    widget.controls.set_write_mode("Append")
    widget.scan.data_file = str(target)
    widget.scan.reduction_config = {
        "gi": False,
        "bai_1d_args": {"unit": "q_A^-1", "numpoints": 100},
        "bai_2d_args": {"unit": "q_A^-1"},
    }
    widget.scan._display_reduction_config = dict(widget.scan.reduction_config)
    widget._on_controls_v2_field_changed(("Int1D", "points"), 200)

    accepted = []

    def _accept(check, processed, current):
        accepted.append(check)
        return True

    monkeypatch.setattr(
        widget.wrangler, "_confirm_append_config_replace", _accept,
        raising=False)
    started = _block_thread_start(widget, monkeypatch)
    monkeypatch.setattr(
        imageWrangler, "_set_action_button", lambda *a, **k: None)

    widget.wrangler.start()

    assert accepted, "the pre-run Append mismatch modal did not fire"
    frozen = widget.wrangler.run_configuration
    assert frozen is not None
    assert frozen.output_mode == "Overwrite", (
        "the accepted modal decision did not reach the frozen configuration")
    assert widget.wrangler.thread.run_configuration is frozen
    assert started


# --------------------------------------------------------------------------- #
# Family (c) — pending-configuration staleness.
# --------------------------------------------------------------------------- #

def test_pending_run_configuration_is_cleared_on_adoption(widget):
    """(c) The pending frozen configuration is consumed exactly once: once a run
    owner adopts it, the slot is empty so no later consumer can inherit it."""
    frozen = widget._prepare_controls_v2_run_configuration()
    assert widget._pending_controls_v2_run_configuration is frozen

    widget._apply_controls_v2_run_state()

    assert widget._pending_controls_v2_run_configuration is None
    assert widget.wrangler.run_configuration is frozen
    assert widget.wrangler.thread.run_configuration is frozen


def test_run_state_apply_after_a_refused_start_does_not_reuse_the_stale_pending(
        widget, monkeypatch):
    """(c) A frozen object left pending by an earlier click must never be the
    configuration a later run adopts — the later run re-prepares from the LIVE
    intent."""
    stale = widget._prepare_controls_v2_run_configuration()
    widget._apply_controls_v2_run_state()
    assert widget._pending_controls_v2_run_configuration is None

    # A later idle edit, then a run-state apply with no fresh prepare.
    #
    # O-1a-W1R-D1 (review §41.4 item 4): this case was written for the
    # second-freeze fallback, which §39 DELETED -- a lost handoff is a typed,
    # visible refusal, never permission to freeze a second generation and run it.
    # The retired expectation (a fresh object carrying the later edit) is what the
    # r1 stale-pending defect looked like from the outside.
    from xrd_tools.session import RunConfigurationRefused

    widget._on_controls_v2_field_changed(("Int1D", "points"), 777)
    generation = int(widget._controls_v2_ensure_run_intent().generation)
    published = widget.wrangler.run_configuration

    with pytest.raises(RunConfigurationRefused) as refusal:
        widget._apply_controls_v2_run_state()

    assert refusal.value.reason == "absent"
    # No new generation, and nothing published.
    assert int(widget._controls_v2_ensure_run_intent().generation) == generation
    assert widget.wrangler.run_configuration is published
    assert widget._pending_controls_v2_run_configuration is None
    assert stale is not None


# --------------------------------------------------------------------------- #
# Family (e) — the REAL production PONI type (regression guard + adoption).
# --------------------------------------------------------------------------- #

def test_real_poni_to_dict_populates_the_frozen_configuration(
        widget, tmp_path):
    """(e) R4B-11 against the production type: ``PONI.to_dict()`` (there is no
    ``as_dict``) fills ``poni_values`` on the frozen object and on every carrier.

    Engine-side compound motion is owned by
    ``test_controls_panel::test_t2_1_compound_poni_*``; this is the freeze
    boundary's own guard."""
    from xrd_tools.core.containers import PONI

    poni_path = _write_poni(tmp_path / "real.poni", 0.1794)
    expected = PONI.from_poni_file(poni_path).to_dict()
    assert not hasattr(PONI, "as_dict")

    widget._on_controls_v2_field_changed(("Signal", "poni_file"), poni_path)
    frozen = widget._prepare_controls_v2_run_configuration()

    assert frozen.poni_file == poni_path
    assert frozen.poni_values == expected
    assert widget.wrangler.poni.to_dict() == expected
    assert widget.wrangler.thread.poni.to_dict() == expected


def test_adopted_scan_calibration_reaches_the_frozen_configuration(
        widget, monkeypatch, tmp_path):
    """(e) §10.5 second half: a calibration ADOPTED from a loaded processed scan
    must be in the frozen provenance before any consumer starts — adoption
    happens before the freeze, not after it.

    A processed scan carries no .poni PATH, so the adopted identity is its
    VALUES; ``poni_file`` legitimately stays empty."""
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.calibration import poni_to_integrator

    poni_path = _write_poni(tmp_path / "adopted.poni", 0.2222)
    adopted = PONI.from_poni_file(poni_path)
    expected = adopted.to_dict()

    # A loaded processed scan carrying its own restored calibration + detector
    # (the GENERIC-DETECTOR guard requires the restored integrator); the
    # wrangler has no calibration of its own.
    widget.scan._cached_poni = adopted
    widget.scan._cached_integrator = poni_to_integrator(adopted)
    widget.wrangler.poni = None
    widget.wrangler.parameters.child("Signal", "poni_file").setValue("")
    # O-1a-W1R-D1 (review §40.1 P1-A, §40.3 D1 items 1-2): arm the
    # AUTHORITATIVE Source card.  Setting only ``img_file`` is the shorthand
    # shape §40.1 named -- a truthful ``get_img_fname`` sync clears a cursor the
    # Source card does not back, and admission now requires a typed source.
    _signal = widget.wrangler.parameters.child("Signal")
    _signal.child("inp_type").setValue("Image Series")
    _signal.child("File").setValue(str(tmp_path / "scan_0001.tif"))
    widget.wrangler.img_file = str(tmp_path / "scan_0001.tif")
    (tmp_path / "scan_0001.tif").write_bytes(b"")
    widget.wrangler.parameters.child("Project", "h5_dir").setValue(str(tmp_path))
    started = _block_thread_start(widget, monkeypatch)
    monkeypatch.setattr(
        type(widget.wrangler), "_set_action_button", lambda *a, **k: None)

    widget.wrangler.start()

    assert started, "the adopted calibration should permit the run to start"
    frozen = widget.wrangler.run_configuration
    assert frozen is not None
    assert frozen.poni_values == expected


# --------------------------------------------------------------------------- #
# Family (f) — §9.10 test 10: ONE identity, ONE freeze.
# --------------------------------------------------------------------------- #

def test_one_frozen_identity_reaches_every_owner_t3_publishes(
        widget, monkeypatch, tmp_path):
    """§9.10 test 10 (T-3 scope): the owners T-3 publishes to — wrangler,
    worker thread, and the pending slot — observe the SAME frozen object with
    ONE ``(generation, fingerprint)``.  The reduction-plan / per-run-scan /
    written-provenance hops are W-1 scope (no new carrier publication here)."""
    poni_path = _write_poni(tmp_path / "cal.poni", 0.10)
    raw_path = tmp_path / "scan_0001.tif"
    raw_path.write_bytes(b"")
    widget._set_poni_field(poni_path)
    # O-1a-W1R-D1 (review §40.1 P1-A, §40.3 D1 items 1-2): arm the
    # AUTHORITATIVE Source card.  Setting only ``img_file`` is the shorthand
    # shape §40.1 named -- a truthful ``get_img_fname`` sync clears a cursor the
    # Source card does not back, and admission now requires a typed source.
    _signal = widget.wrangler.parameters.child("Signal")
    _signal.child("inp_type").setValue("Image Series")
    _signal.child("File").setValue(str(raw_path))
    widget.wrangler.img_file = str(raw_path)
    widget.wrangler.parameters.child("Project", "h5_dir").setValue(str(tmp_path))
    started = _block_thread_start(widget, monkeypatch)
    monkeypatch.setattr(
        type(widget.wrangler), "_set_action_button", lambda *a, **k: None)

    widget.wrangler.start()
    assert started

    frozen = widget.wrangler.run_configuration
    assert frozen is not None
    assert widget.wrangler.thread.run_configuration is frozen
    identities = {frozen.identity, widget.wrangler.thread.run_configuration.identity}
    assert len(identities) == 1


def test_a_single_run_click_freezes_exactly_once(
        widget, monkeypatch, tmp_path):
    """§9.10 step 5: no consumer refreezes.  One Run click advances the
    Controls-owned generation exactly once."""
    from xrd_tools.session.run_configuration import RunIntent

    poni_path = _write_poni(tmp_path / "cal.poni", 0.10)
    raw_path = tmp_path / "scan_0001.tif"
    raw_path.write_bytes(b"")
    widget._set_poni_field(poni_path)
    # O-1a-W1R-D1 (review §40.1 P1-A, §40.3 D1 items 1-2): arm the
    # AUTHORITATIVE Source card.  Setting only ``img_file`` is the shorthand
    # shape §40.1 named -- a truthful ``get_img_fname`` sync clears a cursor the
    # Source card does not back, and admission now requires a typed source.
    _signal = widget.wrangler.parameters.child("Signal")
    _signal.child("inp_type").setValue("Image Series")
    _signal.child("File").setValue(str(raw_path))
    widget.wrangler.img_file = str(raw_path)
    widget.wrangler.parameters.child("Project", "h5_dir").setValue(str(tmp_path))
    _block_thread_start(widget, monkeypatch)
    monkeypatch.setattr(
        type(widget.wrangler), "_set_action_button", lambda *a, **k: None)

    live = widget._controls_v2_ensure_run_intent()
    before = int(live.generation)
    calls = []
    real_freeze = RunIntent.freeze

    canonical_freezes = []

    def _counting_freeze(self, **kwargs):
        # §41.3.B: the click freezes a DETACHED candidate cloned from the
        # canonical intent.  Freezing the canonical instance is what burned a
        # generation on a refused click, so that is the thing to forbid; the idle
        # validation path legitimately freezes its own throwaway candidates.
        if self is live:
            canonical_freezes.append(kwargs)
        calls.append(kwargs)
        return real_freeze(self, **kwargs)

    monkeypatch.setattr(RunIntent, "freeze", _counting_freeze)

    widget.wrangler.start()

    assert canonical_freezes == [], (
        "the canonical intent was frozen directly; §41.3.B requires a detached "
        "candidate so a refused click cannot burn a generation")
    assert calls, "the run click never froze a candidate"
    # The accepted candidate's generation is committed back to the canonical
    # intent exactly once (§41.3.B item 6).
    assert int(live.generation) == before + 1


# --------------------------------------------------------------------------- #
# GI motor default substitution — ONE policy, imported from xrd_tools.session.
# --------------------------------------------------------------------------- #

def test_gi_config_has_no_second_default_substitution_policy(widget):
    """The duplicate default-substitution policy inside
    ``_controls_v2_gi_config`` is DELETED: the surviving resolution is the one
    ``xrd_tools.session.resolve_gi_motor`` policy."""
    import inspect

    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    source = inspect.getsource(staticWidget._controls_v2_gi_config)
    assert "resolve_gi_motor" in source, (
        "_controls_v2_gi_config must delegate to the ONE policy")
    # Exactly ONE surviving default pick: the empty-motor branch, which calls
    # straight into the shared `pick_default_gi_motor`.  The stale-motor branch
    # must no longer re-derive the substitution rule locally.
    assert source.count("_controls_v2_default_gi_motor") == 1, (
        "the stale-motor branch still re-implements the default substitution")


def test_r4b15_unverifiable_stale_motor_still_resolves_to_manual(widget):
    """R4B-15 behaviour preservation: a non-explicit leftover motor ('th') with
    NO real motors offered by the source must resolve to Manual — known-empty is
    ``()``, so the one policy repicks over an empty list."""
    intent = widget._controls_v2_ensure_run_intent()
    intent.gi.enabled = True
    intent.gi.incidence_motor = "th"
    widget._controls_v2_gi_selection_explicit = False

    cfg = widget._controls_v2_gi_config()

    assert cfg["incidence_motor"] == "Manual"


def test_explicit_motor_survives_an_unpopulated_choice_list(widget):
    """The ``()``-vs-``None`` distinction: an EXPLICIT user pick with no
    discovered choices passes ``None`` (knowledge unknown) and is honored, never
    degraded to Manual."""
    intent = widget._controls_v2_ensure_run_intent()
    intent.gi.enabled = True
    intent.gi.incidence_motor = "halpha"
    widget._controls_v2_gi_selection_explicit = True

    cfg = widget._controls_v2_gi_config()

    assert cfg["incidence_motor"] == "halpha"


def test_gi_config_resolution_matches_the_one_policy_over_real_choices(
        widget, monkeypatch):
    """When the source DOES offer real motors, the resolved motor is exactly
    what ``resolve_gi_motor`` returns — no second preference list."""
    from xrd_tools.session import resolve_gi_motor

    choices = ("Manual", "samz", "halpha")
    monkeypatch.setattr(
        widget, "_controls_v2_native_int_choices",
        lambda: {("GI", "th_motor"): choices})
    intent = widget._controls_v2_ensure_run_intent()
    intent.gi.enabled = True
    intent.gi.incidence_motor = "not_a_motor"
    widget._controls_v2_gi_selection_explicit = True

    cfg = widget._controls_v2_gi_config()

    assert cfg["incidence_motor"] == resolve_gi_motor(
        "not_a_motor", ("samz", "halpha"))
    assert cfg["incidence_motor"] == "halpha"
