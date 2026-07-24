"""O-1a-T2.6R (§19.4) — request-carried GI hydration token + frozen source receipt.

The GI-motor hydration completion must carry the request-local token captured
when THAT request started, so ownership is decided by request identity rather
than completion order.  The old FIFO-oldest pop mis-stamped a reverse-order
completion (A-starts, B-starts, B-completes) with the wrong request's identity.

Production-wired: the real ``imageWrangler`` created inside ``staticWidget``, its
real ``sigGIMotorOptions`` Qt signal, and the real request-token bookkeeping.
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


def _emitted_collector(wrangler):
    emitted = []
    wrangler.sigGIMotorOptions.connect(emitted.append)
    wrangler.inp_type = "Image Directory"
    wrangler._invalidate_gi_hydration_requests()
    return emitted


def test_reverse_order_completion_keeps_each_requests_token(widget, qapp):
    wrangler = widget.wrangler
    emitted = _emitted_collector(wrangler)

    wrangler.img_dir = "/tmp/t26r-a"
    token_a = wrangler._begin_gi_hydration_request()
    wrangler.img_dir = "/tmp/t26r-b"
    token_b = wrangler._begin_gi_hydration_request()

    # B completes first, then stale A — each carrying ITS OWN token (§21.4 req 6:
    # a no-token emit is a re-announcement, never a simulated async completion).
    wrangler._emit_gi_hydration(["motor_b"], proved=True, token=token_b)
    wrangler._emit_gi_hydration(["motor_a"], proved=True, token=token_a)
    qapp.processEvents()

    first, second = emitted[-2:]
    assert (first.generation, first.source_fingerprint) == token_b
    assert first.motors == ("motor_b",)
    assert wrangler.gi_hydration_is_current(first) is True
    assert (second.generation, second.source_fingerprint) == token_a
    assert wrangler.gi_hydration_is_current(second) is False


def test_forward_order_completion_also_keeps_each_requests_token(widget, qapp):
    wrangler = widget.wrangler
    emitted = _emitted_collector(wrangler)

    wrangler.img_dir = "/tmp/t26r-a"
    token_a = wrangler._begin_gi_hydration_request()
    wrangler.img_dir = "/tmp/t26r-b"
    token_b = wrangler._begin_gi_hydration_request()

    # An async owner that retains its own token completes A first, then B; each
    # emit carries its OWN captured token regardless of the pending order.
    wrangler._emit_gi_hydration(["motor_a"], proved=True, token=token_a)
    wrangler._emit_gi_hydration(["motor_b"], proved=True, token=token_b)
    qapp.processEvents()

    first, second = emitted[-2:]
    assert (first.generation, first.source_fingerprint) == token_a
    assert first.motors == ("motor_a",)
    assert (second.generation, second.source_fingerprint) == token_b
    assert wrangler.gi_hydration_is_current(second) is True


def test_source_change_between_request_and_completion_is_rejected(widget, qapp):
    wrangler = widget.wrangler
    emitted = _emitted_collector(wrangler)

    wrangler.img_dir = "/tmp/t26r-old"
    token_old = wrangler._begin_gi_hydration_request()
    # The source changes before the (delayed) completion arrives.
    wrangler.img_dir = "/tmp/t26r-new"
    wrangler._begin_gi_hydration_request()

    wrangler._emit_gi_hydration(["stale_motor"], proved=True, token=token_old)
    qapp.processEvents()

    stale = emitted[-1]
    assert (stale.generation, stale.source_fingerprint) == token_old
    assert wrangler.gi_hydration_is_current(stale) is False


def test_same_source_repeated_requests_keep_their_own_generations(widget, qapp):
    """Two requests for the SAME source: identity is the request generation, so
    the older completion is rejected even though the fingerprint still matches."""
    wrangler = widget.wrangler
    emitted = _emitted_collector(wrangler)
    wrangler.img_dir = "/tmp/t26r-same"

    token_old = wrangler._begin_gi_hydration_request()
    token_new = wrangler._begin_gi_hydration_request()
    assert token_old.source_fingerprint == token_new.source_fingerprint
    assert token_old.generation < token_new.generation

    wrangler._emit_gi_hydration(["older"], proved=True, token=token_old)
    wrangler._emit_gi_hydration(["newer"], proved=True, token=token_new)
    qapp.processEvents()

    older, newer = emitted[-2:]
    assert (older.generation, older.source_fingerprint) == token_old
    assert wrangler.gi_hydration_is_current(older) is False
    assert (newer.generation, newer.source_fingerprint) == token_new
    assert wrangler.gi_hydration_is_current(newer) is True


def test_duplicate_completion_never_steals_another_requests_token(widget, qapp):
    """A duplicated completion re-emits its OWN identity; it never consumes the
    outstanding token of a different request."""
    wrangler = widget.wrangler
    emitted = _emitted_collector(wrangler)

    wrangler.img_dir = "/tmp/t26r-dup-a"
    token_a = wrangler._begin_gi_hydration_request()
    wrangler.img_dir = "/tmp/t26r-dup-b"
    token_b = wrangler._begin_gi_hydration_request()

    assert wrangler._emit_gi_hydration(
        ["motor_a"], proved=True, token=token_a).accepted is True
    accepted_count = len(emitted)
    # §21.4 req 1: the token was a SINGLE-COMPLETION capability — the duplicate is
    # inert and cannot emit again under A's (or anyone's) identity.
    assert wrangler._emit_gi_hydration(
        ["motor_a_again"], proved=True, token=token_a).accepted is False
    qapp.processEvents()
    assert len(emitted) == accepted_count
    assert (emitted[-1].generation, emitted[-1].source_fingerprint) == token_a
    assert wrangler.gi_hydration_is_current(emitted[-1]) is False
    # B is untouched by A's replay and still completes for its own owner.
    assert wrangler._emit_gi_hydration(
        ["motor_b"], proved=True, token=token_b).accepted is True
    qapp.processEvents()
    assert wrangler.gi_hydration_is_current(emitted[-1]) is True


def test_requests_beyond_the_former_deque_capacity_keep_their_tokens(widget, qapp):
    """Correlation rides on the request-carried token, so more outstanding
    requests than the diagnostic registry's bound cannot lose identity."""
    wrangler = widget.wrangler
    emitted = _emitted_collector(wrangler)
    wrangler.img_dir = "/tmp/t26r-capacity"

    capacity = wrangler._gi_hydration_pending.maxlen or 16
    tokens = [
        wrangler._begin_gi_hydration_request() for _ in range(capacity + 4)
    ]
    # The oldest tokens are shed from the bounded DIAGNOSTIC history ...
    assert tokens[0] not in wrangler._gi_hydration_pending
    # ... but the identity-keyed AUTHORITY still holds them (§21.4 req 3), so the
    # request that holds one still completes under its own identity.
    assert wrangler._gi_hydration_token_outstanding(tokens[0]) is True
    wrangler._emit_gi_hydration(["oldest"], proved=True, token=tokens[0])
    wrangler._emit_gi_hydration(["newest"], proved=True, token=tokens[-1])
    qapp.processEvents()

    oldest, newest = emitted[-2:]
    assert (oldest.generation, oldest.source_fingerprint) == tokens[0]
    assert wrangler.gi_hydration_is_current(oldest) is False
    assert (newest.generation, newest.source_fingerprint) == tokens[-1]
    assert wrangler.gi_hydration_is_current(newest) is True


def test_cancelled_request_can_never_complete_as_current(widget, qapp):
    """A request that is cancelled (discovery failed, nothing to emit) is
    invalidated: a late completion carrying its token is never current, and the
    cancelled token is not left behind for the next synchronous emit."""
    wrangler = widget.wrangler
    emitted = _emitted_collector(wrangler)
    wrangler.img_dir = "/tmp/t26r-cancel"

    token = wrangler._begin_gi_hydration_request()
    assert wrangler._cancel_gi_hydration_request(token) is True
    assert wrangler._gi_hydration_token_outstanding(token) is False

    # §21.4 req 2: a cancelled token yields NO owner-applicable emission.
    before = len(emitted)
    assert wrangler._emit_gi_hydration(
        ["late"], proved=True, token=token).accepted is False
    qapp.processEvents()
    assert len(emitted) == before

    # A following completion must not adopt the cancelled identity.
    fresh = wrangler._begin_gi_hydration_request()
    wrangler._emit_gi_hydration(["fresh"], proved=True, token=fresh)
    qapp.processEvents()
    assert (emitted[-1].generation, emitted[-1].source_fingerprint) == fresh
    assert wrangler.gi_hydration_is_current(emitted[-1]) is True


def test_close_invalidates_every_outstanding_request(widget, qapp):
    """Closing the wrangler cancels all outstanding requests (§19.4 req 6)."""
    wrangler = widget.wrangler
    emitted = _emitted_collector(wrangler)
    wrangler.img_dir = "/tmp/t26r-close"

    token_a = wrangler._begin_gi_hydration_request()
    token_b = wrangler._begin_gi_hydration_request()

    wrangler.close()
    qapp.processEvents()
    assert not wrangler._gi_hydration_outstanding

    # §21.4 req 2: an invalidated token yields NO owner-applicable emission.
    for token in (token_a, token_b):
        before = len(emitted)
        assert wrangler._emit_gi_hydration(
            ["after_close"], proved=True, token=token).accepted is False
        qapp.processEvents()
        assert len(emitted) == before


def test_nexus_read_failure_cancels_its_own_request(widget, qapp, monkeypatch, tmp_path):
    """Production seam: the NeXus GI-motor read opens a request and, when the
    read raises, CANCELS it instead of leaking an outstanding token."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler import nexusWrangler

    wrangler = next(iter(widget.findChildren(nexusWrangler)), None)
    assert wrangler is not None, "no real nexusWrangler in the static widget"

    target = tmp_path / "broken.nxs"
    target.write_bytes(b"not-a-nexus-file")
    wrangler.nexus_file = str(target)
    wrangler._invalidate_gi_hydration_requests()
    before = int(wrangler._gi_hydration_generation)

    wrangler._emit_gi_motor_options()          # the read raises -> cancel
    qapp.processEvents()

    assert not wrangler._gi_hydration_outstanding      # no leaked token
    assert int(wrangler._gi_hydration_generation) > before


def test_request_token_is_immutable():
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        GIHydrationRequestToken,
    )

    token = GIHydrationRequestToken(3, ("image", "/tmp/x"))
    assert (token.generation, token.source_fingerprint) == (3, ("image", "/tmp/x"))
    with pytest.raises((AttributeError, TypeError)):
        token.generation = 9


def test_source_recovery_receipt_is_frozen():
    from xdart.gui.tabs.static_scan.static_scan_widget import SourceRecoveryReceipt

    receipt = SourceRecoveryReceipt(
        configured="sel", generation=2, observation=None, lazy=True, visible=True)
    assert receipt.configured == "sel"
    assert receipt.lazy is True
    with pytest.raises((AttributeError, TypeError)):
        receipt.configured = "other"
