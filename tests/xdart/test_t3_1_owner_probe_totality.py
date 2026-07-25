"""O-1a-T3.1 (§32.2) — the run-owner admission decision must be TOTAL.

The composed Start boundary of T-3 describes its active/stopping decision as
fail-closed, but both production levels erased an inability to determine owner
activity into "idle": `isRunning()` raising was caught and reported False, and a
host-predicate failure fell back to a probe whose own failure returned None.  A
deleted or broken Qt owner can raise while its worker is active or still
unwinding, and T-3 then permitted harvest/commit/freeze/publication onto exactly
the lifecycle it exists to protect.

The three-owner reproducer here is PROMOTED from Codex's preserved exact-object
adversary `~/repos/tmp/test_codex_t3_owner_probe_totality.py` (3 failed at
`6ec8c336`); the original is Codex-owned and was run, never edited.  Per §32.3
item 1 each owner case now asserts the COMPLETE refusal preservation snapshot
rather than only that the worker did not start (closing N6), and two composition
cases plus one green-only Stitch guard (closing N2's test gap) are added.

The preservation helpers are imported from the committed T-3 module so both
oracles assert an identical preservation set by construction — there is no second
copy to drift.
"""

from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtWidgets

from .test_t3_start_boundary import (
    _assert_preserved,
    _establish_prior_frozen,
    _preservation_snapshot,
)

#: Owner label -> the real Qt owner that label observes on a live staticWidget.
_OWNERS = ("wrangler", "reintegration", "stitch")


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
            value._exit_run_state()
        except Exception:
            pass
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _owner_threads(widget):
    return {
        "wrangler": widget.wrangler.thread,
        "reintegration": widget.integratorTree.integrator_thread,
        "stitch": widget.stitch_thread,
    }


def _boom_probe():
    raise RuntimeError("simulated deleted/broken Qt owner probe")


def _break_one_owner_probe(widget, monkeypatch, owner):
    """Make exactly ONE owner's probe raise; every other owner reads idle."""
    for label, thread in _owner_threads(widget).items():
        monkeypatch.setattr(
            thread, "isRunning",
            _boom_probe if label == owner else (lambda: False))


def _install_status_spy(monkeypatch):
    """Capture every `_safe_status_text` message the refusal renders."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
    )

    recorded = []
    original = imageWrangler._safe_status_text

    def _spy(obj, text):
        recorded.append(text)
        return original(obj, text)

    monkeypatch.setattr(imageWrangler, "_safe_status_text", staticmethod(_spy))
    return recorded


# --------------------------------------------------------------------------- #
# Promoted Codex adversary — one case per owner, full preservation set (N6).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("owner", _OWNERS)
def test_owner_probe_exception_refuses_before_freeze(widget, monkeypatch, owner):
    """§32.2: an owner whose `isRunning()` RAISES is unknown, never idle.

    Preparation must raise the typed refusal and leave the complete §31.3
    item-4 preservation set untouched — no harvest, no freeze, no publication.
    """
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        RunOwnerActiveError,
    )

    prior = _establish_prior_frozen(widget)
    _break_one_owner_probe(widget, monkeypatch, owner)
    before = _preservation_snapshot(widget)

    with pytest.raises(RunOwnerActiveError):
        widget._prepare_controls_v2_run_configuration()

    after = _preservation_snapshot(widget)
    _assert_preserved(before, after)
    assert widget.wrangler.run_configuration is prior


@pytest.mark.parametrize("owner", _OWNERS)
def test_owner_probe_exception_is_labelled_not_silently_idle(
        widget, monkeypatch, owner):
    """The predicate reports a STABLE label naming the owner it could not
    observe, so the refusal is diagnosable rather than anonymous."""
    _break_one_owner_probe(widget, monkeypatch, owner)

    decision = widget._controls_v2_active_run_owner()

    assert decision is not None
    assert owner in str(decision)


@pytest.mark.parametrize("owner", _OWNERS)
def test_present_owner_without_a_callable_probe_is_unknown(
        widget, monkeypatch, owner):
    """§32.3 item 2: a PRESENT owner that offers no callable activity probe is
    also unknown.  (An owner genuinely ABSENT during partial construction stays
    idle — that carve-out is pinned separately below.)"""
    threads = _owner_threads(widget)
    for label, thread in threads.items():
        if label != owner:
            monkeypatch.setattr(thread, "isRunning", lambda: False)
    # Shadow the bound method on the INSTANCE with a non-callable.  (Deleting it
    # from the QThread class instead would break every other owner's teardown.)
    monkeypatch.setattr(threads[owner], "isRunning", None, raising=False)

    decision = widget._controls_v2_active_run_owner()

    assert decision is not None
    assert owner in str(decision)


@pytest.mark.parametrize("owner", _OWNERS)
def test_absent_owner_during_partial_construction_stays_idle(
        widget, monkeypatch, owner):
    """§32.3 item 2 carve-out: an owner attribute that is genuinely absent/None
    must NOT refuse — partial construction is not an active run."""
    for label, thread in _owner_threads(widget).items():
        monkeypatch.setattr(thread, "isRunning", lambda: False)
    if owner == "wrangler":
        monkeypatch.setattr(widget.wrangler, "thread", None)
    elif owner == "reintegration":
        monkeypatch.setattr(widget.integratorTree, "integrator_thread", None)
    else:
        monkeypatch.setattr(widget, "stitch_thread", None)

    assert widget._controls_v2_active_run_owner() is None


# --------------------------------------------------------------------------- #
# Composition case 1 — the run-session probe (§32.3 item 1, second bullet).
# --------------------------------------------------------------------------- #

def test_run_session_is_running_failure_refuses_before_preparation(
        widget, monkeypatch):
    """A streaming session whose `is_running` RAISES is an unknown/unsafe owner
    state: preparation refuses and preserves everything.

    `is_running` is a property, so a broken/torn-down session raises on ATTRIBUTE
    ACCESS — the exact shape a real half-destroyed session takes.
    """
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        RunOwnerActiveError,
    )

    prior = _establish_prior_frozen(widget)
    for thread in _owner_threads(widget).values():
        monkeypatch.setattr(thread, "isRunning", lambda: False)

    class _BrokenSession:
        @property
        def is_running(self):
            raise RuntimeError("session torn down mid-query")

    monkeypatch.setattr(
        widget.wrangler, "scan_session", _BrokenSession(), raising=False)
    before = _preservation_snapshot(widget)

    decision = widget._controls_v2_active_run_owner()
    with pytest.raises(RunOwnerActiveError):
        widget._prepare_controls_v2_run_configuration()

    assert decision is not None
    after = _preservation_snapshot(widget)
    _assert_preserved(before, after)
    assert widget.wrangler.run_configuration is prior


def test_absent_run_session_still_reads_idle(widget, monkeypatch):
    """No session open is the ordinary idle case, not an unknown one."""
    for thread in _owner_threads(widget).values():
        monkeypatch.setattr(thread, "isRunning", lambda: False)
    monkeypatch.setattr(widget.wrangler, "scan_session", None, raising=False)

    assert widget._controls_v2_active_run_owner() is None


# --------------------------------------------------------------------------- #
# Composition case 2 — the wrapper must not erase a host-predicate failure.
# --------------------------------------------------------------------------- #

def test_host_predicate_failure_refuses_start_before_any_mutation(
        widget, monkeypatch):
    """§32.3 items 1 and 3: when the host predicate EXISTS but FAILS,
    `imageWrangler.start()` refuses VISIBLY before `_inputs_valid`, command
    mutation, the action-button morph, freeze, or `sigStart`."""
    prior = _establish_prior_frozen(widget)

    def _broken_predicate():
        raise RuntimeError("admission predicate itself failed")

    monkeypatch.setattr(
        widget, "_controls_v2_active_run_owner", _broken_predicate)

    validated = []
    monkeypatch.setattr(
        widget.wrangler, "_inputs_valid",
        lambda: validated.append(True) or True)
    emitted = []
    widget.wrangler.sigStart.connect(lambda: emitted.append(True))
    started = []
    monkeypatch.setattr(
        widget.wrangler.thread, "start", lambda: started.append(True))
    recorded = _install_status_spy(monkeypatch)
    phase = getattr(widget.wrangler, "_run_phase", None)
    before = _preservation_snapshot(widget)

    widget.wrangler.start()

    assert recorded, "the refusal was not surfaced to the operator"
    assert validated == [], "_inputs_valid ran despite the refusal"
    assert emitted == [] and started == []
    assert getattr(widget.wrangler, "_run_phase", None) == phase
    after = _preservation_snapshot(widget)
    _assert_preserved(before, after)
    assert widget.wrangler.run_configuration is prior


def test_fallback_probe_failure_also_refuses(widget, monkeypatch):
    """§32.3 item 3: a holder WITHOUT the shared host predicate may use the
    wrangler-only fallback, but a fallback probe that RAISES still refuses."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
    )

    monkeypatch.setattr(widget.wrangler, "_h19_host", None, raising=False)
    monkeypatch.setattr(widget.wrangler.thread, "isRunning", _boom_probe)

    assert imageWrangler._active_run_owner(widget.wrangler) is not None


def test_fallback_without_a_probe_stays_idle(widget, monkeypatch):
    """The duck-typed/partial holders the sentinels use expose no `isRunning` at
    all.  Absence is the item-2 carve-out, NOT a probe failure: those holders
    must keep starting, or the Start sentinels would regress."""
    import types

    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
    )

    holder = types.SimpleNamespace(thread=types.SimpleNamespace(command=None))

    assert imageWrangler._active_run_owner(holder) is None


# --------------------------------------------------------------------------- #
# N2 — green-only regression guard for the accepted Stitch behaviour.
# --------------------------------------------------------------------------- #

def test_stitch_click_emits_its_request_without_preparing_a_run(
        widget, monkeypatch, tmp_path):
    """N2 (accepted, no production change): Stitch reduces an ALREADY-LOADED
    scan, so a Stitch click must emit its request while making zero
    prepare/freeze/journal/pending-carrier changes."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    poni_path = tmp_path / "cal.poni"
    poni_path.write_text(
        "Distance: 0.1\nPoni1: 0.01\nPoni2: 0.02\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n")
    widget._set_poni_field(str(poni_path))
    prior = _establish_prior_frozen(widget)

    prepared = []
    real_prepare = staticWidget._prepare_controls_v2_run_configuration
    monkeypatch.setattr(
        staticWidget, "_prepare_controls_v2_run_configuration",
        lambda self: (prepared.append(True), real_prepare(self))[1])

    requested = []
    widget.wrangler.sigStitchRequested.connect(requested.append)
    started = []
    monkeypatch.setattr(
        widget.wrangler.thread, "start", lambda: started.append(True))
    monkeypatch.setattr(widget.wrangler, "stitch_mode", True, raising=False)
    before = _preservation_snapshot(widget)

    widget.wrangler.start()

    assert requested, "the Stitch request was not emitted"
    assert prepared == [], "a Stitch click prepared/froze a run configuration"
    assert started == []
    after = _preservation_snapshot(widget)
    _assert_preserved(before, after)
    assert widget.wrangler.run_configuration is prior
