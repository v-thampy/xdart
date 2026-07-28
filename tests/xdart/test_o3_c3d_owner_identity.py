# -*- coding: utf-8 -*-
"""X1 O-3 c3D — one hydration owner, one rescope, one action algebra.

FROZEN BEFORE the c3D-1 production edits (§12.7), red at `9a3b166d`.

Section 12's root cause is that "one hydration owner travels with the request"
was implemented as PARALLEL SCALARS in four places — the context's admitted
versus current source, the selection's copied fields, the public request's four
fields, and the worker request's four fields — so every reconstruction was
another chance to pick the admitted source instead of the current one, or to
apply a different normalization rule.  Alongside that, two boundaries collapsed
states the policy distinguishes: selected-context resolution folded "could not
observe" into "idle", and the run-action callback folded "malformed" into
"allowed" by truthiness.

The families pinned here are exactly those:

* **A** — the owner has ONE construction contract.  The frozen dataclass
  normalizes, and ``of()`` delegates to it; a bool, string, float or hostile
  epoch can never become an epoch, and nothing raises out of ``qualified``.
* **B** — the request and the selection are minted from the context's CURRENT
  identity, so a request built after a member rescope is admitted by the very
  context that built it.
* **C** — a rescope moves a complete ``(key, source)`` pair or nothing at all;
  the signal-only ``new_scan`` path may not advance the hydration identity.
* **D** — resolution is IDLE, ACTIVE or ERROR.  Only IDLE is compatibility.
* **E** — the action callback allows on exact ``None`` and refuses on anything
  else, and the public-action matrix walks the REAL ``ui.wranglerStack``.

RULE 10: every invocation of this module must set ``XDART_SESSION_FILE`` to a
unique path under ``/Users/vthampy/repos/tmp``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
from xdart.gui.tabs.static_scan.static_scan_widget import (
    RUN_ORIGIN_REINTEGRATE,
    staticWidget,
)
from xdart.modules.display_context import (
    AcquisitionContext,
    ContextKind,
    DisplaySelection,
    HydrationOwner,
)

_ADMITTED = "/raw/admitted-root"
_MEMBER = "/raw/member-0007.h5"


class _Store:
    pass


def _context(*, source=_ADMITTED):
    return AcquisitionContext(
        context_token="context-A",
        run_configuration=object(),
        config_generation=1,
        config_fingerprint="fp",
        run_scan_key="scan-A",
        source_path=source,
        scan=object(),
        frame=object(),
        frame_ids=[],
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=_Store(),
    )


def _display(context, *, mirror="context-A"):
    return SimpleNamespace(
        selected_display_context=lambda: context,
        display_context_token=mirror,
        display_generation=7,
        frame_record_store=None,
        publication_store=_Store(),
    )


# --------------------------------------------------------------------------- #
# A — one construction contract
# --------------------------------------------------------------------------- #

class _ExplosiveEpoch:
    def __gt__(self, other):
        raise RuntimeError("epoch comparison escaped")

    def __bool__(self):
        raise RuntimeError("epoch truthiness escaped")


class _ExplosiveStr(str):
    def __bool__(self):
        raise RuntimeError("text truthiness escaped")


class _ExplosiveInt(int):
    def __gt__(self, other):
        raise RuntimeError("integer comparison escaped")


class _ExplosiveList(list):
    def __len__(self):
        raise RuntimeError("container length escaped")


class _ExplosiveOwner(HydrationOwner):
    @property
    def qualified(self):
        raise RuntimeError("owner subclass escaped")


class _ExplosiveActionStr(str):
    def __bool__(self):
        raise RuntimeError("action truthiness escaped")


class _ExplosiveClass:
    @property
    def __class__(self):
        raise RuntimeError("class lookup escaped")


@pytest.mark.parametrize(
    "epoch", [True, False, "7", "not-an-int", 1.5, object(), None, [],
              _ExplosiveEpoch()])
def test_the_direct_constructor_normalizes_exactly_like_the_factory(epoch):
    """§12.6 A.1 — one invariant, whichever path a caller picks.

    ``of()`` normalized and the public dataclass constructor did not, so a
    caller could choose which contract the value object enforced.  That is the
    same authority split in a smaller spelling.
    """
    direct = HydrationOwner("tok", "key", "/data/a.nxs", epoch)
    factory = HydrationOwner.of("tok", "key", "/data/a.nxs", epoch)
    assert direct == factory
    assert direct.epoch == 0
    assert direct.qualified is False


def test_a_direct_bool_epoch_never_equals_a_real_epoch_of_one():
    """A bool is not an epoch — and must not compare equal to integer 1."""
    forged = HydrationOwner("tok", "key", "/data/a.nxs", True)
    real = HydrationOwner.of("tok", "key", "/data/a.nxs", 1)
    assert forged != real
    assert forged.qualified is False
    assert real.qualified is True


def test_qualified_is_total_on_a_directly_constructed_owner():
    """§12.6 A.3 — ``qualified`` reads normalized fields and cannot raise."""
    owner = HydrationOwner("tok", "key", "/data/a.nxs", _ExplosiveEpoch())
    assert owner.qualified is False


def test_accepted_type_subclasses_cannot_escape_owner_normalization():
    """§12.6 A.2/A.3 applies to hostile subclasses, not only other types."""
    owner = HydrationOwner(
        _ExplosiveStr("tok"), "key", "/data/a.nxs", _ExplosiveInt(7))
    assert owner.context_token == ""
    assert owner.epoch == 0
    assert owner.qualified is False


def test_foreign_class_protocol_cannot_escape_text_normalization():
    owner = HydrationOwner(_ExplosiveClass(), "key", "/data/a.nxs", 7)
    assert type(owner.context_token) is str
    assert isinstance(owner.qualified, bool)


def test_a_directly_constructed_malformed_owner_is_refused_without_raising():
    context = _context()
    display = _display(context)
    for epoch in (True, "7", object(), _ExplosiveEpoch()):
        malformed = HydrationOwner(
            context.context_token, context.scan_key, context.source, epoch)
        assert displayFrameWidget._admit_hydration_owner(
            display, malformed, 1, 7) is False, epoch


def test_foreign_decoder_subclasses_are_inert_nonthrowing_refusals():
    """§11.1.2/§12.6 A — accepted-looking subclasses are still foreign."""
    context = _context()
    display = _display(context)
    malformed = (
        _ExplosiveList(["tok", "key"]),
        _ExplosiveOwner("tok", "key", "/data/a.nxs", 7),
    )
    for value in malformed:
        assert displayFrameWidget._admit_hydration_owner(
            display, value, 1, 7) is False


# --------------------------------------------------------------------------- #
# B — one projection, carried unchanged
# --------------------------------------------------------------------------- #

def test_a_context_projects_exactly_one_hydration_owner():
    """§12.6 B.1 — the ONE production mint, from the CURRENT identity."""
    context = _context()
    owner = context.hydration_owner
    assert owner == HydrationOwner.of(
        context.context_token, context.scan_key, context.source,
        context.commit_epoch)
    assert owner.qualified is True

    context.rescope_to("scan-A-member-7", source=_MEMBER)
    moved = context.hydration_owner
    assert moved.scan_key == "scan-A-member-7"
    assert moved.source == _MEMBER
    assert moved.epoch > owner.epoch


def test_the_request_is_minted_from_the_current_source(tmp_path):
    """§12.2 — the functional loss: scroll-back after a member transition."""
    context = _context()
    display = _display(context)
    context.rescope_to("scan-A-member-7", source=_MEMBER)

    request = displayFrameWidget._build_hydration_request(
        display, 1, purpose="2d")

    assert request.owner == context.hydration_owner
    assert request.context_scan_key == context.scan_key
    assert request.context_source == context.source == _MEMBER


def test_a_request_built_after_a_rescope_is_admitted_by_its_own_context():
    """The builder and the admission boundary must not disagree."""
    context = _context()
    display = _display(context)
    context.rescope_to("scan-A-member-7", source=_MEMBER)
    request = displayFrameWidget._build_hydration_request(
        display, 1, purpose="2d")

    assert displayFrameWidget._admit_hydration_owner(
        display, request.owner, request.label, request.generation) is True


def test_active_context_never_reconstructs_a_missing_owner_projection():
    """§12.6 B.1 — the context projection is the only active mint."""
    context = SimpleNamespace(
        hydration_owner=object(),
        context_token="forged-token",
        scan_key="forged-key",
        source="/forged/source",
        commit_epoch=9,
        commit_gate=object(),
    )
    display = _display(context)
    request = displayFrameWidget._build_hydration_request(
        display, 1, purpose="2d")
    assert request is not None
    assert request.owner == HydrationOwner()
    assert request.enqueueable is False
    assert displayFrameWidget._admit_hydration_owner(
        display,
        HydrationOwner("forged-token", "forged-key", "/forged/source", 9),
        1,
        request.generation,
    ) is False


def test_the_selection_is_minted_from_the_current_source():
    """§12.6 B.4 — the selection cannot independently choose the admitted one."""
    context = _context()
    context.rescope_to("scan-B", source="/raw/member-B.h5")
    selection = DisplaySelection.for_context(context, 3)

    assert selection.kind is ContextKind.ACQUISITION
    assert selection.scan_key == "scan-B"
    assert selection.source_path == "/raw/member-B.h5"
    assert selection.owner == context.hydration_owner


def test_the_admitted_source_stays_available_as_provenance():
    """§12.6 B.5 — provenance is kept, just not under the current accessor."""
    context = _context()
    context.rescope_to("scan-A-member-7", source=_MEMBER)
    assert context.admitted_source == _ADMITTED
    assert context.source == _MEMBER


# --------------------------------------------------------------------------- #
# C — one unambiguous rescope
# --------------------------------------------------------------------------- #

def test_a_rescope_moves_a_complete_pair_or_nothing():
    """§12.6 C.1/C.4 — omission may not mean "reuse the old source"."""
    context = _context()
    before = (context.scan_key, context.source, context.commit_epoch)

    for bad in ("", None):
        with pytest.raises(Exception):
            context.rescope_to("scan-B", source=bad)
        assert (context.scan_key, context.source,
                context.commit_epoch) == before, bad

    with pytest.raises(Exception):
        context.rescope_to("", source=_MEMBER)
    assert (context.scan_key, context.source, context.commit_epoch) == before


def test_a_signal_only_new_scan_does_not_advance_the_hydration_identity(
        widget):
    """§12.6 C.3 / §12.2 — the signal-before-frame race.

    ``new_scan()`` can rescope with no frame at all.  Advancing the key and the
    epoch there left the previous SOURCE in place, and the later frame then saw
    an already-matching key and never performed the member-source restamp — a
    mixed identity reachable purely by signal ordering.
    """
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    context = widget._acquisition_context
    before = (context.scan_key, context.source, context.commit_epoch)

    # The signal arrives first, carrying no authoritative member source.
    widget._rescope_frame_panel_to("member-7")
    assert (context.scan_key, context.source, context.commit_epoch) == before, (
        "a signal-only rescope advanced the context hydration identity")

    # The frame then arrives and IS authoritative.
    widget._rescope_frame_panel_to(
        "member-7", first_frame=SimpleNamespace(source_file=_MEMBER))
    assert context.scan_key == "member-7"
    assert context.source == _MEMBER
    assert context.commit_epoch > before[2]


# --------------------------------------------------------------------------- #
# D — three-state resolution
# --------------------------------------------------------------------------- #

def _broken_display(context):
    display = _display(context, mirror="")

    def _broken():
        raise RuntimeError("selected context unavailable")

    display.selected_display_context = _broken
    return display


def test_a_broken_resolver_is_an_error_not_idle_permission():
    """§12.3/§12.6 D — a broken authority is never permission."""
    display = _broken_display(_context())
    assert displayFrameWidget._admit_hydration_owner(
        display, None, 1, 7) is False
    owner = HydrationOwner.of("old-context", "old-scan", "/old/source", 2)
    assert displayFrameWidget._admit_hydration_owner(
        display, owner, 1, 7) is False


def test_a_broken_resolver_does_not_mint_an_ownerless_request():
    """ERROR may not be downgraded to the explicit idle adapter."""
    display = _broken_display(_context())
    request = displayFrameWidget._build_hydration_request(
        display, 1, purpose="2d")
    assert request is not None
    assert request.enqueueable is False, (
        "an unresolvable owner produced an enqueueable request")


def test_an_installed_but_unresolved_selection_is_error_not_idle(widget):
    """The real host must distinguish stale selection from no selection."""
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    context = widget._acquisition_context
    display = widget.displayframe
    try:
        # Keep the installed selection but withdraw the context it names.
        widget._acquisition_context = None
        request = displayFrameWidget._build_hydration_request(
            display, 1, purpose="2d")
        assert request is not None
        assert request.enqueueable is False
        assert request.owner == HydrationOwner()
        assert request.stores == ()
    finally:
        widget._acquisition_context = context


def test_an_ownerful_late_completion_is_not_the_idle_compatibility_call():
    """A complete owner from a RETIRED context is not an idle completion."""
    display = SimpleNamespace(
        display_context_token="",
        selected_display_context=lambda: None,
    )
    owner = HydrationOwner.of("old-context", "old-scan", "/old/source", 2)
    assert displayFrameWidget._admit_hydration_owner(
        display, owner, 1, 1) is False


def test_the_genuine_idle_path_still_admits_an_ownerless_completion():
    """IDLE — and only IDLE — keeps the pinned legacy compatibility shape."""
    display = SimpleNamespace(
        display_context_token="",
        selected_display_context=lambda: None,
    )
    assert displayFrameWidget._admit_hydration_owner(
        display, None, 1, 1) is True
    assert displayFrameWidget._build_hydration_request(
        display, 1, purpose="2d") is None


# --------------------------------------------------------------------------- #
# E — exact action algebra and the real wrangler topology
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    value = staticWidget()
    try:
        yield value
    finally:
        value._acquisition_context = None
        value._browse_context = None
        value._display_selection = None
        value._run_active = False
        try:
            value._controls_v2_refresh_timer.cancel()
        except Exception:
            pass
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _arm(tree, monkeypatch, events):
    monkeypatch.setattr(tree, "_block_if_no_frames", lambda _dim: False)
    monkeypatch.setattr(tree, "_block_if_reload_only_frames",
                        lambda _dim: False)
    monkeypatch.setattr(tree, "_ensure_reintegration_calibration",
                        lambda _dim: events.append("calibration") or True)
    monkeypatch.setattr(tree, "_apply_gi_config_to_scan",
                        lambda: events.append("gi"))
    monkeypatch.setattr(tree, "_apply_threshold_config_to_thread",
                        lambda: events.append("threshold"))
    monkeypatch.setattr(tree.integrator_thread, "isRunning", lambda: False)
    monkeypatch.setattr(tree.integrator_thread, "start",
                        lambda: events.append("start"))


@pytest.mark.parametrize(
    "malformed",
    [False, 0, "", [], {}, 0.0, object(), _ExplosiveActionStr("owner")])
@pytest.mark.parametrize("action", ["bai_1d", "bai_2d"])
def test_a_malformed_action_result_refuses_rather_than_permits(
        widget, monkeypatch, malformed, action):
    """§12.4/§12.6 E.2 — truthiness is not the algebra.

    The callback contract is ``str | None``.  ``bool(result)`` made every
    FALSEY malformed answer permission, so both entries then loaded
    calibration, applied GI/threshold state and started the thread.
    """
    tree = widget.integratorTree
    events = []
    _arm(tree, monkeypatch, events)
    monkeypatch.setattr(tree, "_refuse_run_action",
                        lambda _action: malformed, raising=False)

    getattr(tree, action)(None)
    assert events == [], (
        f"{action} treated malformed result {malformed!r} as permission")


@pytest.mark.parametrize("action", ["bai_1d", "bai_2d"])
def test_only_exact_none_allows_a_reintegration_action(
        widget, monkeypatch, action):
    """§12.6 E.1/E.2 — ``None`` allows; a non-empty label refuses."""
    tree = widget.integratorTree
    events = []
    _arm(tree, monkeypatch, events)

    monkeypatch.setattr(tree, "_refuse_run_action",
                        lambda _action: "context-cleanup", raising=False)
    getattr(tree, action)(None)
    assert events == []

    monkeypatch.setattr(tree, "_refuse_run_action",
                        lambda _action: None, raising=False)
    getattr(tree, action)(None)
    assert "start" in events, "exact None did not allow the action"


def _stack_wranglers(widget):
    """Every wrangler in the REAL production stack (§12.6 E.3)."""
    stack = widget.ui.wranglerStack
    return [stack.widget(index) for index in range(stack.count())]


def test_the_action_matrix_walks_the_real_wrangler_stack(widget):
    """§12.4 — the committed helper claimed image + NeXus and reached one.

    It enumerated ``widget.wrangler`` and a nonexistent ``widget._wranglers``;
    the real second owner lives in ``ui.wranglerStack``.
    """
    enumerated = staticWidget._public_start_entries(widget)
    assert {type(one).__name__ for one in enumerated} == {
        "imageWrangler", "nexusWrangler"}
    assert {type(one).__name__ for one in _stack_wranglers(widget)} == {
        type(one).__name__ for one in enumerated}
    assert not hasattr(widget, "_wranglers"), (
        "an invented wrangler collection is forbidden")


@pytest.mark.parametrize("owner_result", ["raise", "falsey-malformed"])
def test_both_real_public_starts_fail_closed_before_side_effects(
        qapp, widget, monkeypatch, owner_result):
    """§12.6 E.3 — drive each real stack entry's own public ``start()``."""
    def probe():
        if owner_result == "raise":
            raise RuntimeError("owner probe failed")
        return False

    def button(one, name):
        direct = getattr(one, name, None)
        return direct if direct is not None else getattr(one.ui, name)

    monkeypatch.setattr(widget, "_controls_v2_active_run_owner", probe)
    stack = widget.ui.wranglerStack
    for index, wrangler in enumerate(_stack_wranglers(widget)):
        events = []
        stack.setCurrentIndex(index)
        qapp.processEvents()
        assert widget.wrangler is wrangler
        assert wrangler._h19_host is widget
        wrangler.sigStart.connect(lambda e=events: e.append("sigStart"))
        before = (
            getattr(wrangler, "command", None),
            getattr(getattr(wrangler, "thread", None), "command", None),
            bool(button(wrangler, "startButton").isEnabled()),
            bool(button(wrangler, "stopButton").isEnabled()),
            getattr(wrangler, "_run_phase", None),
        )

        wrangler.start()

        after = (
            getattr(wrangler, "command", None),
            getattr(getattr(wrangler, "thread", None), "command", None),
            bool(button(wrangler, "startButton").isEnabled()),
            bool(button(wrangler, "stopButton").isEnabled()),
            getattr(wrangler, "_run_phase", None),
        )
        assert events == [], f"{type(wrangler).__name__} emitted {events}"
        assert after == before, type(wrangler).__name__
