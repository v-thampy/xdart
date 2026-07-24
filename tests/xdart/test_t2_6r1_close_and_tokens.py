"""O-1a-T2.6R.1 (§21.3/§21.4/§21.5) — production close ownership, single-use
hydration tokens, honest source-recovery receipt.

§21.3: the PRODUCTION teardown path is ``staticWidget.close()``.  Qt does not
deliver a child's ``closeEvent`` when the parent closes, so the child guard alone
left outstanding GI requests alive across a real tab/application close.

§21.4: an explicit token is a SINGLE-COMPLETION CAPABILITY — proved outstanding
and retired before anything is emitted — so a duplicate completion carrying the
still-current token cannot overwrite the first result.  Completion and direct
re-announcement are structurally separate APIs.

§21.5: the receipt's proof fields (configured, lazy, explicit visibility) are all
verified; generation and the by-value observation are diagnostic only.

Production-wired: the real ``staticWidget``, the real wrangler stack, real Qt
signals, and the real commit engine.
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
    value._refresh_controls_v2_profile_now()
    try:
        yield value
    finally:
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _collector(wrangler):
    emitted = []
    wrangler.sigGIMotorOptions.connect(emitted.append)
    wrangler.inp_type = "Image Directory"
    return emitted


def _stack_wranglers(widget):
    stack = widget.ui.wranglerStack
    return [stack.widget(i) for i in range(stack.count())]


# ---------------------------------------------------------------------------
# §21.3 — production close ownership
# ---------------------------------------------------------------------------

def test_host_close_invalidates_requests_on_every_wrangler(widget, qapp):
    """§21.3 req 5: requests opened on the ACTIVE and on an INACTIVE wrangler are
    all invalidated by the real ``staticWidget.close()``; retained callbacks are
    inert afterwards and cannot touch the visible GI choices."""
    wranglers = _stack_wranglers(widget)
    assert len(wranglers) >= 2, "need an active and an inactive wrangler"

    retained = []
    emissions = []
    for wrangler in wranglers:
        wrangler.sigGIMotorOptions.connect(emissions.append)
        retained.append((wrangler, wrangler._begin_gi_hydration_request()))
    epochs = {id(w): int(w._gi_hydration_generation) for w, _ in retained}
    assert all(w._gi_hydration_token_outstanding(t) for w, t in retained)

    widget.close()
    qapp.processEvents()

    for wrangler, token in retained:
        assert not wrangler._gi_hydration_outstanding      # registry empty
        assert int(wrangler._gi_hydration_generation) > epochs[id(wrangler)]
        # the retained callback fires late: inert, no accepted emission
        before = len(emissions)
        outcome = wrangler._emit_gi_hydration(
            ["late"], proved=True, token=token)
        assert outcome.accepted is False
        assert outcome.hydration is None
        assert len(emissions) == before
        assert wrangler._gi_hydration_token_outstanding(token) is False


def test_host_close_is_idempotent_and_opens_no_request(widget, qapp):
    """§21.3 req 3: a second close — and a child close after host invalidation —
    must not create a request or raise."""
    wrangler = widget.wrangler
    wrangler._begin_gi_hydration_request()

    first = widget._invalidate_all_gi_hydration()
    assert first >= 1
    epoch = int(wrangler._gi_hydration_generation)

    second = widget._invalidate_all_gi_hydration()
    assert second == first
    assert not wrangler._gi_hydration_outstanding

    wrangler.close()                       # child close AFTER host invalidation
    qapp.processEvents()
    assert not wrangler._gi_hydration_outstanding
    assert int(wrangler._gi_hydration_generation) >= epoch

    widget.close()                         # second host close
    qapp.processEvents()
    assert not wrangler._gi_hydration_outstanding


def test_host_teardown_owner_runs_before_thread_shutdown(widget, monkeypatch):
    """§21.3 req 1: the invalidation owner runs at the START of close(), before
    any thread shutdown or queued-signal unwind."""
    order = []
    real_invalidate = widget._invalidate_all_gi_hydration
    monkeypatch.setattr(
        widget, "_invalidate_all_gi_hydration",
        lambda: (order.append("invalidate"), real_invalidate())[1])
    monkeypatch.setattr(
        widget, "_stop_wrangler_thread_on_close",
        lambda *_a, **_k: order.append("threads"))

    widget.close()

    assert order[:2] == ["invalidate", "threads"]


# ---------------------------------------------------------------------------
# §21.4 — single-use tokens and API separation
# ---------------------------------------------------------------------------

def test_current_token_completes_once_and_duplicate_is_inert(widget, qapp):
    """§21.4 req 1/8: the duplicate carries the STILL-CURRENT token; exactly one
    emission is accepted and the observed motors are unchanged afterwards."""
    wrangler = widget.wrangler
    emitted = _collector(wrangler)
    wrangler.img_dir = "/tmp/t26r1-dup"

    token = wrangler._begin_gi_hydration_request()
    first = wrangler._emit_gi_hydration(["accepted"], proved=True, token=token)
    assert first.accepted is True
    assert wrangler.gi_hydration_is_current(emitted[-1]) is True
    count_after_first = len(emitted)

    duplicate = wrangler._emit_gi_hydration(
        ["duplicate_overwrite"], proved=True, token=token)

    assert duplicate.accepted is False
    assert duplicate.hydration is None
    assert len(emitted) == count_after_first          # ONE accepted emission
    assert emitted[-1].motors == ("accepted",)        # motors unchanged
    qapp.processEvents()
    assert emitted[-1].motors == ("accepted",)


def test_duplicate_after_cancellation_is_inert(widget, qapp):
    wrangler = widget.wrangler
    emitted = _collector(wrangler)
    wrangler.img_dir = "/tmp/t26r1-cancel"

    token = wrangler._begin_gi_hydration_request()
    assert wrangler._cancel_gi_hydration_request(token) is True
    before = len(emitted)

    outcome = wrangler._emit_gi_hydration(["late"], proved=True, token=token)

    assert outcome.accepted is False
    assert len(emitted) == before


def test_duplicate_after_host_close_is_inert(widget, qapp):
    """A completion arriving after the real host close is never applicable."""
    wrangler = widget.wrangler
    emitted = _collector(wrangler)
    token = wrangler._begin_gi_hydration_request()

    widget.close()
    qapp.processEvents()
    before = len(emitted)

    outcome = wrangler._emit_gi_hydration(["after_close"], proved=True,
                                          token=token)

    assert outcome.accepted is False
    assert len(emitted) == before


def test_reverse_ab_completion_each_carries_its_own_token(widget, qapp):
    wrangler = widget.wrangler
    emitted = _collector(wrangler)

    wrangler.img_dir = "/tmp/t26r1-a"
    token_a = wrangler._begin_gi_hydration_request()
    wrangler.img_dir = "/tmp/t26r1-b"
    token_b = wrangler._begin_gi_hydration_request()

    assert wrangler._emit_gi_hydration(
        ["motor_b"], proved=True, token=token_b).accepted is True
    assert wrangler._emit_gi_hydration(
        ["motor_a"], proved=True, token=token_a).accepted is True
    qapp.processEvents()

    current_b, stale_a = emitted[-2:]
    assert (current_b.generation, current_b.source_fingerprint) == token_b
    assert wrangler.gi_hydration_is_current(current_b) is True
    assert (stale_a.generation, stale_a.source_fingerprint) == token_a
    assert wrangler.gi_hydration_is_current(stale_a) is False


def test_beyond_former_capacity_an_exact_token_still_completes(widget, qapp):
    """§21.4 req 3: the identity-keyed registry is the authority, so correctness
    does not depend on the bounded diagnostic history's capacity."""
    wrangler = widget.wrangler
    emitted = _collector(wrangler)
    wrangler.img_dir = "/tmp/t26r1-capacity"

    capacity = wrangler._gi_hydration_pending.maxlen or 16
    tokens = [wrangler._begin_gi_hydration_request()
              for _ in range(capacity + 4)]

    # evicted from the DIAGNOSTIC history ...
    assert tokens[0] not in wrangler._gi_hydration_pending
    # ... but still outstanding in the AUTHORITY, so it completes exactly once.
    assert wrangler._gi_hydration_token_outstanding(tokens[0]) is True
    assert wrangler._emit_gi_hydration(
        ["oldest"], proved=True, token=tokens[0]).accepted is True
    assert (emitted[-1].generation, emitted[-1].source_fingerprint) == tokens[0]
    assert wrangler._emit_gi_hydration(
        ["oldest_again"], proved=True, token=tokens[0]).accepted is False


def test_announcement_never_consumes_an_outstanding_request(widget, qapp):
    """§21.4 req 4: the direct current-source re-announcement is a separate API
    that neither consumes nor infers a request token."""
    wrangler = widget.wrangler
    emitted = _collector(wrangler)
    wrangler.img_dir = "/tmp/t26r1-announce"

    token = wrangler._begin_gi_hydration_request()
    outcome = wrangler._announce_gi_hydration(["announced"], proved=True)
    qapp.processEvents()

    assert outcome.accepted is True
    assert emitted[-1].motors == ("announced",)
    # the outstanding request is untouched and still completes for its owner
    assert wrangler._gi_hydration_token_outstanding(token) is True
    assert wrangler._emit_gi_hydration(
        ["completed"], proved=True, token=token).accepted is True


def test_completion_requires_a_token(widget):
    """§21.4 req 4: completion is structurally token-bearing."""
    wrangler = widget.wrangler
    with pytest.raises(TypeError):
        wrangler._emit_gi_hydration(["no_token"], proved=True)


def test_synchronous_image_path_completes_its_own_request(widget, qapp):
    """§21.4 req 5: the synchronous image discovery path threads the token opened
    by its request rather than relying on 'most recent pending'."""
    wrangler = widget.wrangler
    emitted = _collector(wrangler)
    wrangler.img_dir = "/tmp/t26r1-sync"
    wrangler.motors = ["halpha"]
    wrangler._gi_motor_knowledge_proved = True

    token = wrangler._begin_gi_hydration_request()
    wrangler.set_gi_motor_options()
    qapp.processEvents()

    assert (emitted[-1].generation, emitted[-1].source_fingerprint) == token
    assert wrangler._gi_hydration_token_outstanding(token) is False
    # A second bare call is a re-announcement, not a second completion.
    before = len(emitted)
    wrangler.set_gi_motor_options()
    qapp.processEvents()
    assert len(emitted) == before + 1
    assert emitted[-1].generation == int(wrangler._gi_hydration_generation)


# ---------------------------------------------------------------------------
# §21.5 — the receipt proves what it claims
# ---------------------------------------------------------------------------

def test_receipt_proof_fields_verify_on_success(widget):
    receipt = widget._controls_v2_capture_source_receipt()
    assert widget._controls_v2_source_restore_verified(receipt) is True


def test_silent_visibility_no_op_is_a_source_recovery_failure(widget, monkeypatch):
    """§21.5 req 3: a visibility setter that silently does nothing must fail the
    proof rather than pass unchecked."""
    receipt = widget._controls_v2_capture_source_receipt()
    monkeypatch.setattr(
        widget, "_controls_v2_source_widget_visible",
        lambda: not bool(receipt.visible))

    assert widget._controls_v2_source_restore_verified(receipt) is False


def test_raising_visibility_reader_does_not_escape(widget, monkeypatch):
    panel = widget.controls_v2
    monkeypatch.setattr(
        panel, "source_widget_visible",
        lambda: (_ for _ in ()).throw(RuntimeError("injected")))

    assert widget._controls_v2_source_widget_visible() is False


def test_receipt_diagnostic_fields_are_by_value_and_not_proof(widget):
    """§21.5 req 1/4: generation and observation are diagnostic — not
    equality-checked — and the observation is a snapshot, not a live reference."""
    observation = {"motors": ["halpha"]}
    widget._controls_v2_directory_observation = observation
    receipt = widget._controls_v2_capture_source_receipt()

    observation["motors"].append("mutated")
    assert receipt.observation == {"motors": ["halpha"]}

    bumped = receipt._replace(generation=receipt.generation + 5)
    assert widget._controls_v2_source_restore_verified(bumped) is True


def test_receipt_is_frozen(widget):
    receipt = widget._controls_v2_capture_source_receipt()
    with pytest.raises((AttributeError, TypeError)):
        receipt.configured = "other"
    with pytest.raises((AttributeError, TypeError)):
        receipt.visible = not receipt.visible
