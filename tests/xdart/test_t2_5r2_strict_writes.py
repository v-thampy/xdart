"""O-1a-T2.5R.2 (§22) — strict bound writes + member-total recovery.

The §22 exact-object review of T-2.5R.1 (`4d7065ac`) found the bound-carrier and
recovery contracts unfinished in seven places.  This module is the fail-before /
mutation evidence for the correction:

* §22.2 — the transaction delegated its forward write to the PERMISSIVE
  compatibility mirror, which discards every child setter exception.  A setter
  that installed the exact coerced value and then raised produced ``ok=True`` on
  a partially-executed setter.  The transaction now has a STRICT writer: the
  required root+child signal blocking is kept, but ANY setter exception reaches
  the transaction boundary and is a forward failure even when the final readback
  equals ``expected``.
* §22.3 — ``_controls_v2_bound_carriers`` was a second MUTABLE target authority: the
  writer re-selected its target by ``path -> registry -> param``, so an entry
  replaced after the pre-write identity guard redirected the write to an object the
  outer loop never rolled back.  The prepared carrier is now passed DIRECTLY; the
  registry is an immutable diagnostic record that never selects a target.
* §22.4 — the identity resolver and the energy-cache invalidation ran OUTSIDE the
  guarded boundary, so each could leave an earlier carrier installed and escape
  untyped.  Every post-first-write step is now behind ONE funnel that recovers
  exactly once and returns a typed ``ControlsCommitResult``.
* §22.5 — recovery was class-continuing but not MEMBER-continuing: shared ``try``
  blocks meant the first raising member left every later member mutated.  Each
  class is now ordered member descriptors.
* §22.6 — ``collect()`` only wrapped ``restore(*args)``; iterating/normalizing the
  result happened outside the ``try``, so a wrapper result that raised mid-iteration
  skipped every later recovery class.
* §22.7 — the prepared plan was attribute-frozen but not recursively immutable
  (an open-ended ``deepcopy`` fallback).  The carrier now shares the ONE closed
  journal value algebra.
* §22.8 — the self-state comparator had no declared per-field policy, and
  ``_controls_v2_source_restore_verified(None)`` was FAIL-OPEN.

Production-wired: real ``staticWidget``, real pyqtgraph ``Parameter`` handles, the
real staging/commit engine.
"""

from __future__ import annotations

import copy
from types import MappingProxyType, SimpleNamespace

import pytest
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import Parameter


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


MASK = ("Signal", "mask_file")
BG = ("BG", "File")
#: a SOURCE-SELECTION legacy carrier (§22.10.B.5)
INCLUDE_SUBDIR = ("Signal", "include_subdir")
#: the PONI-file legacy carrier (§22.10.B.5)
PONI = ("Signal", "poni_file")


def _boom(*_args, **_kwargs):
    raise RuntimeError("injected")


def _exact_then_raise(original, requested):
    """A setter that installs the EXACT requested value and THEN raises."""
    real_set = type(original).setValue

    def setter(self, value, *args, **kwargs):
        if self is original and value == requested:
            real_set(self, value, *args, **kwargs)
            raise RuntimeError("setter failed after exact mutation")
        return real_set(self, value, *args, **kwargs)

    return setter


# ---------------------------------------------------------------------------
# §22.2 / §22.10.B — strict bound writes
# ---------------------------------------------------------------------------

def test_exact_expected_mutation_then_setter_raise_is_not_success(
        widget, monkeypatch):
    """§22.11 case 1 / §22.10.B.3-4.  A setter that installs the exact coerced
    value and then RAISES is a forward failure even though the readback would
    match: readback proves the final value, never that the setter completed its
    side effects.  The carrier is rolled back and the setter exception is
    preserved as the forward diagnostic."""
    original = widget._controls_v2_param(MASK)
    prior = original.value()
    requested = "/tmp/t25r2-exact-then-raise.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])

    monkeypatch.setattr(
        type(original), "setValue", _exact_then_raise(original, requested))
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert result.failed_path == MASK
    # B.4: the setter's own exception is the forward diagnostic, not a generic
    # "readback failed" (which would be a false explanation of what went wrong).
    assert "setter failed after exact mutation" in result.reason
    assert original.value() == prior


def test_source_selection_carrier_uses_the_strict_bound_writer(
        widget, monkeypatch):
    """§22.11 case 13 / §22.10.B.5.  A SOURCE-SELECTION legacy carrier gets the
    same strict path — no permissive variant survives for it."""
    original = widget._controls_v2_param(INCLUDE_SUBDIR)
    prior = bool(original.value())
    requested = not prior
    staged = widget.stage_controls_transaction([(INCLUDE_SUBDIR, requested)])

    monkeypatch.setattr(
        type(original), "setValue", _exact_then_raise(original, requested))
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert result.failed_path == INCLUDE_SUBDIR
    assert "setter failed after exact mutation" in result.reason
    assert bool(original.value()) == prior


def test_poni_file_carrier_uses_the_strict_bound_writer(widget, monkeypatch, tmp_path):
    """§22.11 case 13 / §22.10.B.5.  The PONI-file legacy carrier gets the same
    strict path."""
    original = widget._controls_v2_param(PONI)
    prior = original.value()
    requested = str(tmp_path / "t25r2-strict.poni")
    staged = widget.stage_controls_transaction([(PONI, requested)])

    monkeypatch.setattr(
        type(original), "setValue", _exact_then_raise(original, requested))
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert result.failed_path == PONI
    assert "setter failed after exact mutation" in result.reason
    assert original.value() == prior


def test_strict_writer_blocks_root_and_child_signals(widget):
    """§22.10.B.1: the strict writer is NOT the permissive mirror, but it keeps
    the mirror's required signal discipline — the ROOT (whose signal owns
    ``wrangler.setup()``) and the bound CHILD are both blocked for the write."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _PreparedLegacyCarrier)

    root = widget.wrangler.parameters
    param = widget._controls_v2_param(MASK)
    prior = param.value()
    carrier = _PreparedLegacyCarrier(MASK, param, prior, prior, prior)
    seen = {"root": 0, "child": 0}
    root.sigTreeStateChanged.connect(
        lambda *_a: seen.__setitem__("root", seen["root"] + 1))
    param.sigValueChanged.connect(
        lambda *_a: seen.__setitem__("child", seen["child"] + 1))

    widget._controls_v2_write_legacy_carrier(
        root, carrier, "/tmp/t25r2-blocked.edf")

    assert seen == {"root": 0, "child": 0}
    assert param.value() == "/tmp/t25r2-blocked.edf"


def test_non_transactional_mirror_callers_keep_skip_missing(widget):
    """§22.10.B.2 (PRESERVE): the two NON-transactional pre-run mirror callers
    keep the permissive skip-missing behavior — a heterogeneous wrangler schema
    without a given group must not raise out of a pre-run push."""
    root = widget.wrangler.parameters
    # A path that does not exist in this wrangler's schema is SKIPPED, not raised.
    widget._mirror_wrangler_parameter_values(
        root, ((("NoSuchGroup", "no_such_field"), 1),))
    # ... and a real path in the same batch is still written.
    widget._mirror_wrangler_parameter_values(
        root,
        ((("NoSuchGroup", "no_such_field"), 1),
         (MASK, "/tmp/t25r2-mirror-survivor.edf")))
    assert widget._controls_v2_param(MASK).value() == (
        "/tmp/t25r2-mirror-survivor.edf")


# ---------------------------------------------------------------------------
# §22.3 / §22.10.A — one target authority
# ---------------------------------------------------------------------------

def test_bound_carrier_registry_is_immutable_and_never_selects_a_target(
        widget):
    """§22.10.A.1/A.3.  The published plan registry is IMMUTABLE (a second
    authority cannot rewrite an entry at all) and is retained for diagnostics
    only."""
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r2-registry.edf")])
    widget.commit_controls_transaction(staged)

    registry = widget._controls_v2_bound_carriers
    assert isinstance(registry, MappingProxyType)
    with pytest.raises(TypeError):
        registry[MASK] = SimpleNamespace(param=None)


def test_registry_redirect_after_the_identity_guard_cannot_move_the_write(
        widget, monkeypatch):
    """§22.11 case 2 / §22.10.A.  A second authority that swaps the widget-level
    registry entry after the pre-write identity guard must NOT redirect the
    forward write: the replacement is never touched and the original is restored.

    At `4d7065ac` the writer re-selected its target through the registry, so the
    replacement was mutated and left mutated (the outer loop rolled back only the
    original, which it had read back and refused)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _PreparedLegacyCarrier)

    original = widget._controls_v2_param(MASK)
    original_prior = original.value()
    replacement = Parameter.create(
        name="replacement-mask", type="str", value="/tmp/replacement-prior.edf")
    replacement_prior = replacement.value()
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r2-redirect.edf")])

    real_lookup = widget._controls_v2_param
    calls = {"n": 0}

    def lookup(candidate):
        if tuple(candidate) != MASK:
            return real_lookup(candidate)
        calls["n"] += 1
        if calls["n"] == 2:            # after the pre-write identity guard
            registry = dict(widget._controls_v2_bound_carriers)
            registry[MASK] = _PreparedLegacyCarrier(
                MASK, replacement, "/tmp/t25r2-redirect.edf",
                replacement_prior, "/tmp/t25r2-redirect.edf")
            # Even a whole-attribute swap (not merely an item write, which the
            # immutable proxy already refuses) must not move the target: the
            # commit loop holds its own carrier objects.
            object.__setattr__(widget, "_controls_v2_bound_carriers",
                               MappingProxyType(registry))
        return original

    monkeypatch.setattr(widget, "_controls_v2_param", lookup)
    result = widget.commit_controls_transaction(staged)

    assert calls["n"] >= 2, "the identity guard did not run after preflight"
    # The redirect is a COMPLETE no-op: the write went to the prepared original
    # and the replacement was never touched, so there is nothing to roll back.
    assert replacement.value() == replacement_prior
    assert original.value() == "/tmp/t25r2-redirect.edf"
    assert original.value() != original_prior
    assert result.ok


# ---------------------------------------------------------------------------
# §22.4 / §22.10.C — the post-first-write funnel is exception-total
# ---------------------------------------------------------------------------

def test_raising_later_carrier_guard_is_typed_and_rolls_back_the_earlier_one(
        widget, monkeypatch):
    """§22.11 case 3 / §22.10.C.1.  A raising identity guard while preparing a
    LATER carrier used to escape as an untyped ``RuntimeError`` with an EARLIER
    carrier still installed."""
    first = widget._controls_v2_param(MASK)
    second = widget._controls_v2_param(BG)
    first_prior = first.value()
    second_prior = second.value()
    staged = widget.stage_controls_transaction(
        [(MASK, "/tmp/t25r2-first.edf"), (BG, "/tmp/t25r2-second.edf")])

    real_lookup = widget._controls_v2_param
    calls = {"second": 0}

    def lookup(path):
        if tuple(path) == BG:
            calls["second"] += 1
            if calls["second"] == 2:      # the second carrier's pre-write guard
                raise RuntimeError("injected identity-guard failure")
        return real_lookup(path)

    monkeypatch.setattr(widget, "_controls_v2_param", lookup)
    try:
        result = widget.commit_controls_transaction(staged)
    except Exception as exc:                      # pragma: no cover - the defect
        pytest.fail(f"untyped exception escaped the transaction: {exc!r}")

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert first.value() == first_prior            # earlier carrier rolled back
    assert second.value() == second_prior          # never written


def test_raising_energy_cache_invalidation_is_typed_and_recovers(
        widget, monkeypatch):
    """§22.11 case 4 / §22.10.C.1-2.  Energy-cache invalidation ran outside the
    guarded boundary: a raise there left the REQUESTED preference installed and
    escaped untyped.  It is now inside the one funnel — typed result, recovery
    ran exactly once, and the prior preference is restored."""
    prior = getattr(widget, "_controls_v2_source_energy_preference", "poni")
    requested = "metadata" if prior != "metadata" else "poni"
    staged = widget.stage_controls_transaction(
        [(("Source", "energy_preference"), requested)])
    cache_state = {"value": getattr(
        widget, "_controls_v2_source_energy_cache", None)}

    def get_cache(_self):
        return cache_state["value"]

    def set_cache(_self, value):
        if value is None:
            raise RuntimeError("injected cache invalidation failure")
        cache_state["value"] = value

    with monkeypatch.context() as patch:
        patch.setattr(
            type(widget), "_controls_v2_source_energy_cache",
            property(get_cache, set_cache), raising=False)
        try:
            result = widget.commit_controls_transaction(staged)
        except Exception as exc:                  # pragma: no cover - the defect
            pytest.fail(f"untyped exception escaped the transaction: {exc!r}")

    assert not result.ok
    assert result.phase == "install"
    assert "energy cache invalidation" in result.reason
    # The requested preference must NOT survive an aborted transaction.
    assert widget._controls_v2_source_energy_preference == prior


def test_recovery_runs_exactly_once_and_preserves_the_forward_identity(
        widget, monkeypatch):
    """§22.10.C.2-3.  Every post-first-write failure invokes
    ``_controls_v2_recover_all`` EXACTLY ONCE, and the original forward
    phase/path/reason survives independently of recovery failures."""
    original = widget._controls_v2_param(MASK)
    staged = widget.stage_controls_transaction(
        [(MASK, "/tmp/t25r2-once.edf"), (("Int1D", "points"), "444")])
    calls = {"n": 0}
    real_recover = widget._controls_v2_recover_all

    def counting_recover(*args, **kwargs):
        calls["n"] += 1
        return real_recover(*args, **kwargs)

    monkeypatch.setattr(widget, "_controls_v2_recover_all", counting_recover)
    monkeypatch.setattr(widget, "_controls_v2_apply_snapshot_to_scan", _boom)
    # Recovery ITSELF fails for a class; the forward identity must not change.
    monkeypatch.setattr(widget, "_controls_v2_restore_display_scan", _boom)
    result = widget.commit_controls_transaction(staged)

    assert calls["n"] == 1
    assert not result.ok
    assert result.phase == "install"
    assert result.reason == "intent install failed"
    assert ("Display",) in result.recovery_failed_paths
    assert original.value() != "/tmp/t25r2-once.edf"   # legacy class still ran


# ---------------------------------------------------------------------------
# §22.5 / §22.10.D — member-total recovery
# ---------------------------------------------------------------------------

def test_first_source_cache_member_raises_later_members_still_restore(widget):
    """§22.11 case 5 / §22.10.D.1-3.  A raising directory-observation setter used
    to leave BOTH the energy cache and the probe cache at the new values."""
    prior_observation = object()
    widget._controls_v2_source_energy_cache = ("new", 1)
    widget._controls_v2_metadata_probe_cache = ("new", 2)
    ctx = {
        "prior_observation": prior_observation,
        "prior_energy_cache": ("prior", 1),
        "prior_probe_cache": ("prior", 2),
    }

    class FailingObservation:
        def __get__(self, obj, owner=None):
            return None

        def __set__(self, obj, value):
            raise RuntimeError("injected observation restore failure")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(widget), "_controls_v2_directory_observation",
                      FailingObservation(), raising=False)
        failures = widget._controls_v2_restore_source_caches_verified(ctx)

    # the later members really restored
    assert widget._controls_v2_source_energy_cache == ("prior", 1)
    assert widget._controls_v2_metadata_probe_cache == ("prior", 2)
    # the class label AND the unrecoverable member's own stable path
    assert ("Source", "cache") in failures
    assert ("Source", "observation") in failures
    assert ("Source", "energy_cache") not in failures
    assert ("Source", "probe_cache") not in failures


def test_first_poni_member_raises_later_members_still_restore(widget):
    """§22.11 case 6 / §22.10.D.4-6.  A raising ``wrangler.poni`` restore used to
    leave ``wrangler.poni_file`` AND ``thread.poni`` at the new values."""
    prior_wrangler_poni = object()
    prior_thread_poni = object()

    class FailingWrangler:
        def __init__(self):
            object.__setattr__(self, "poni", object())
            object.__setattr__(self, "poni_file", "new.poni")

        def __setattr__(self, name, value):
            if name == "poni" and value is prior_wrangler_poni:
                raise RuntimeError("injected wrangler PONI restore failure")
            object.__setattr__(self, name, value)

    wrangler = FailingWrangler()
    thread = SimpleNamespace(poni=object())
    ctx = {
        "wrangler": wrangler, "thread": thread,
        "prior_wrangler_poni": prior_wrangler_poni,
        "prior_wrangler_poni_file": "prior.poni",
        "prior_thread_poni": prior_thread_poni,
    }

    failures = widget._controls_v2_restore_poni_carriers_verified(ctx)

    assert wrangler.poni_file == "prior.poni"          # later member restored
    assert thread.poni is prior_thread_poni            # later member restored
    assert ("Signal", "poni_file") in failures         # class label
    assert ("Signal", "wrangler_poni") in failures     # unrecoverable member
    assert ("Signal", "wrangler_poni_file") not in failures
    assert ("Signal", "thread_poni") not in failures


def test_first_intent_member_raises_later_members_still_restore(widget):
    """§22.11 case 7 / §22.10.D.7.  Intent fields restore INDEPENDENTLY — the
    first field whose restore raises no longer leaves the later fields holding
    the transaction's objects."""
    live = widget._controls_v2_ensure_run_intent()
    snapshot = widget._controls_v2_snapshot_intent_values(live)
    names = list(snapshot)
    first, second = names[0], names[1]
    sentinel_first, sentinel_second = object(), object()
    setattr(live, first, sentinel_first)
    setattr(live, second, sentinel_second)

    real_apply = type(widget)._controls_v2_apply_intent_snapshot

    def failing_apply(target, values):
        if first in values:
            raise RuntimeError("injected intent member failure")
        return real_apply(target, values)

    with pytest.MonkeyPatch.context() as patch:
        # the class-level primary is a silent no-op, so every member falls to its
        # own authoritative backstop; the FIRST member's backstop then raises.
        patch.setattr(widget, "_controls_v2_restore_intent_values",
                      lambda *_a, **_k: None)
        patch.setattr(type(widget), "_controls_v2_apply_intent_snapshot",
                      staticmethod(failing_apply))
        failures = widget._controls_v2_restore_intent_values_verified(
            live, snapshot)

    assert getattr(live, first) is sentinel_first          # unrecoverable
    assert getattr(live, second) is snapshot[second]       # later member restored
    assert ("Intent",) in failures
    assert ("Intent", first) in failures
    assert ("Intent", second) not in failures


def test_first_display_member_raises_later_members_still_restore(widget):
    """§22.11 case 8 / §22.10.D.8.  Display-scan fields restore INDEPENDENTLY."""
    scan = widget.scan
    snapshot = widget._controls_v2_snapshot_display_scan(scan)
    values = widget._controls_v2_display_snapshot_values(snapshot)
    names = [n for n in values if not isinstance(values[n], dict)]
    assert len(names) >= 2, "need two scalar display fields for this case"
    first, second = names[0], names[1]
    setattr(scan, first, "t25r2-mutated-first")
    setattr(scan, second, "t25r2-mutated-second")

    real_apply = type(widget)._controls_v2_apply_display_snapshot

    def failing_apply(target, vals):
        if first in vals:
            raise RuntimeError("injected display member failure")
        return real_apply(target, vals)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(widget, "_controls_v2_restore_display_scan",
                      lambda *_a, **_k: None)
        patch.setattr(type(widget), "_controls_v2_apply_display_snapshot",
                      staticmethod(failing_apply))
        failures = widget._controls_v2_restore_display_scan_verified(
            scan, snapshot)

    assert getattr(scan, first) == "t25r2-mutated-first"   # unrecoverable
    assert getattr(scan, second) == values[second]         # later member restored
    assert ("Display",) in failures
    assert ("Display", first) in failures
    assert ("Display", second) not in failures


def test_member_backstop_success_does_not_erase_primary_failure_evidence(
        widget):
    """§22.10.D closing rule.  A self-state member whose OWN primary failed is
    still NAMED even though the authoritative instance-dict backstop recovered
    the value."""
    prior = getattr(widget, "_controls_v2_source_energy_preference", "poni")
    ctx = {"prior_gi_explicit": False, "prior_threshold_state": None,
           "prior_energy_pref": prior}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            type(widget), "_controls_v2_source_energy_preference",
            property(lambda _s: prior, _boom), raising=False)
        failures = widget._controls_v2_restore_self_state_verified(ctx)

    # the value reads back correctly (the getter returns prior) ...
    assert widget._controls_v2_source_energy_preference == prior
    # ... but the primary restore contract FAILED and that evidence survives.
    assert ("Intent", "source_energy_preference") in failures


# ---------------------------------------------------------------------------
# §22.6 / §22.10.F — the collector contains result iteration
# ---------------------------------------------------------------------------

def test_wrapper_result_iterator_raising_does_not_skip_later_classes(
        widget, monkeypatch):
    """§22.11 case 9 / §22.10.F.1.  A wrapper returning a generator that yields a
    failure and THEN raises used to escape ``_controls_v2_recover_all`` and skip
    every later recovery class."""
    def later_failure():
        yield ("Display",)
        raise RuntimeError("injected during failure iteration")

    monkeypatch.setattr(
        widget, "_controls_v2_restore_display_scan_verified",
        lambda *_a, **_k: later_failure())
    live = widget._controls_v2_ensure_run_intent()
    ctx = {
        "staged": SimpleNamespace(poni_touched=False),
        "scan": widget.scan, "display_scan_snapshot": {},
        "params": widget.wrangler.parameters,
        "live": live,
        "intent_snapshot": widget._controls_v2_snapshot_intent_values(live),
        "prior_gi_explicit": False, "prior_threshold_state": None,
        "prior_energy_pref": getattr(
            widget, "_controls_v2_source_energy_preference", "poni"),
    }

    try:
        failures = widget._controls_v2_recover_all(
            ctx, [], touched={"display", "intent"})
    except Exception as exc:                      # pragma: no cover - the defect
        pytest.fail(f"recovery collector let an exception escape: {exc!r}")

    # the malformed/raising class becomes its stable fallback label ...
    assert ("Display",) in failures
    # ... and the LATER classes still ran (self-state is attempted after intent).
    assert widget._controls_v2_gi_selection_explicit is False


@pytest.mark.parametrize("bad", ["not-a-path", 5, [object()]])
def test_malformed_wrapper_result_becomes_the_class_fallback_label(
        widget, monkeypatch, bad):
    """§22.10.F.2-3.  Only ``None`` or a concrete sequence of path tuples is a
    valid wrapper result; anything malformed becomes the class's stable label
    and later classes still run."""
    monkeypatch.setattr(
        widget, "_controls_v2_restore_display_scan_verified",
        lambda *_a, **_k: bad)
    live = widget._controls_v2_ensure_run_intent()
    ctx = {
        "staged": SimpleNamespace(poni_touched=False),
        "scan": widget.scan, "display_scan_snapshot": {},
        "params": widget.wrangler.parameters, "live": live,
        "intent_snapshot": widget._controls_v2_snapshot_intent_values(live),
        "prior_gi_explicit": False, "prior_threshold_state": None,
        "prior_energy_pref": getattr(
            widget, "_controls_v2_source_energy_preference", "poni"),
    }

    failures = widget._controls_v2_recover_all(
        ctx, [], touched={"display", "intent"})

    assert ("Display",) in failures
    for path in failures:
        assert isinstance(path, tuple)


# ---------------------------------------------------------------------------
# §22.7 / §22.10.E — ONE closed immutable value algebra
# ---------------------------------------------------------------------------

def test_prepared_payload_reads_are_fresh_reconstructions():
    """§22.11 case 12 / §22.10.E.5.  A mutable payload read out of the prepared
    plan is a FRESH reconstruction — mutating it cannot reach transaction
    authority.  At `4d7065ac` ``carrier.value`` handed out the stored object."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _PreparedLegacyCarrier)

    carrier = _PreparedLegacyCarrier(
        MASK, object(), ["a"], bytearray(b"p"), {"k": ["e"]})

    carrier.value.append("mutated")
    carrier.prior.extend(b"mutated")
    carrier.expected["k"].append("mutated")
    carrier.expected["injected"] = 1

    assert carrier.value == ["a"]
    assert carrier.prior == bytearray(b"p")
    assert carrier.expected == {"k": ["e"]}
    # ... and each read is a distinct object, never the stored representation.
    assert carrier.value is not carrier.value


def test_unsupported_mutable_prepared_payload_is_refused_at_staging():
    """§22.11 case 12 / §22.10.E.3-4.  The carrier/journal INVENTORY is scalar and
    small-container only, so an array-like or mutable custom payload has no
    reviewed immutable encoding and is REJECTED at staging with the existing typed
    refusal — the open-ended ``deepcopy`` fallback that kept it mutable is gone."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        ControlsTransactionError, _PreparedLegacyCarrier)

    np = pytest.importorskip("numpy")

    class MutableCustom:
        pass

    for payload in (np.array([1, 2]), MutableCustom(), {"nested": [object()]}):
        with pytest.raises(ControlsTransactionError):
            _PreparedLegacyCarrier(MASK, object(), payload, None, None)


def test_unsupported_carrier_payload_fails_preflight_with_zero_writes(
        widget, monkeypatch):
    """§22.10.E.4: the refusal is mapped to a typed PREFLIGHT failure, before the
    first write — not an exception escaping the commit."""
    original = widget._controls_v2_param(MASK)
    prior = original.value()
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r2-refuse.edf")])
    np = pytest.importorskip("numpy")

    monkeypatch.setattr(
        "xdart.gui.tabs.static_scan.static_scan_widget."
        "coerce_control_edit_value",
        lambda _current, _value: np.array([1, 2]))
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "preflight"
    assert result.failed_path == MASK
    assert original.value() == prior            # ZERO writes


def test_carrier_and_journal_share_one_algebra():
    """§22.7 / §22.10.E.2: literally the same closed encoding in both places."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        JournalEntry, _PreparedLegacyCarrier)

    payload = {"a": [1, bytearray(b"x")], "b": (True, None)}
    carrier = _PreparedLegacyCarrier(MASK, object(), payload, payload, payload)
    entry = JournalEntry(payload, 1, "ui")
    assert carrier._value == entry._frozen


def test_zero_d_numpy_array_is_normalized_and_higher_rank_is_refused():
    """§22.10.E.3 DECLARED 0-d POLICY.  A 0-d array carries exactly one immutable
    scalar, so it is NORMALIZED to that scalar (only the scalar is ever stored);
    rank >= 1 has no reviewed immutable encoding here and is REFUSED."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        ControlsTransactionError, JournalEntry)

    np = pytest.importorskip("numpy")

    zero_d = JournalEntry(np.array(5), 1, "ui")
    assert zero_d.value == 5
    assert type(zero_d.value) is int
    with pytest.raises(ControlsTransactionError):
        JournalEntry(np.array([5]), 1, "ui")          # rank 1
    with pytest.raises(ControlsTransactionError):
        JournalEntry(np.array([[5]]), 1, "ui")        # rank 2


def test_journal_entry_equality_is_mapping_order_independent():
    """§21.6 req 5 (orchestrator finding): entry-level equality must be
    value-correct for mappings.  Dict pairs are frozen in INSERTION order so a
    thaw round-trips it, so equality canonicalises instead."""
    from xdart.gui.tabs.static_scan.static_scan_widget import JournalEntry

    a = JournalEntry({"a": 1, "b": 2}, 3, "ui")
    b = JournalEntry({"b": 2, "a": 1}, 3, "ui")
    assert a == b
    assert JournalEntry({"a": 1}, 3, "ui") != JournalEntry({"a": 2}, 3, "ui")
    # the thaw still round-trips the caller's original insertion order
    assert list(JournalEntry({"b": 1, "a": 2}, 3, "ui").value) == ["b", "a"]
    # tags are retained, so container KINDS stay distinguishable
    assert JournalEntry({1}, 3, "ui") != JournalEntry(frozenset({1}), 3, "ui")
    assert JournalEntry([1], 3, "ui") != JournalEntry((1,), 3, "ui")


# ---------------------------------------------------------------------------
# §22.8 / §22.10.E.6 — receipt proof and declared comparator policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "receipt", [None, "not-a-receipt", ("configured", 0, None, False, True)])
def test_missing_or_malformed_source_receipt_cannot_certify_recovery(
        widget, receipt):
    """§22.11 case 10 / §22.10.E.6.  ``receipt is None -> True`` was FAIL-OPEN:
    when source reconciliation was attempted, the absence (or wrong type) of the
    preflight receipt certified restoration unconditionally.  It now fails
    CLOSED."""
    assert widget._controls_v2_source_restore_verified(receipt) is False


def test_source_recovery_without_a_receipt_is_a_named_failure(
        widget, monkeypatch):
    """§22.10.E.6 wired through the recovery collector: an uncertifiable source
    restore is the stable ``("Source",)`` recovery failure, never silent success."""
    ctx = {"source_receipt": None}
    monkeypatch.setattr(
        widget, "_controls_v2_restore_source_selection_carriers",
        lambda *_a, **_k: None)
    monkeypatch.setattr(widget, "_sync_controls_v2_source_index",
                        lambda *_a, **_k: None)

    assert widget._controls_v2_restore_source_owner_verified(
        ctx, []) == [("Source",)]


def test_equal_but_distinct_self_state_follows_the_declared_field_policy(
        widget):
    """§22.11 case 11 / §22.8.  The per-field policy is DECLARED (see
    ``_CONTROLS_V2_SELF_STATE_FIELDS``) and the comparator follows it: all three
    fields are VALUE semantics, so an equal-but-distinct object IS a correct
    restoration and must NOT be reported as a failure."""
    prior_threshold = {"apply_threshold": False, "threshold_min": 0.0,
                       "threshold_max": 0.0, "mask_saturation": True}
    prior_pref = "".join(["meta", "data"])
    widget._controls_v2_gi_selection_explicit = True
    widget._controls_v2_threshold_state = copy.deepcopy(prior_threshold)
    widget._controls_v2_source_energy_preference = "".join(["meta", "data"])
    ctx = {"prior_gi_explicit": True,
           "prior_threshold_state": prior_threshold,
           "prior_energy_pref": prior_pref}

    # equal-but-DISTINCT objects are already installed
    assert widget._controls_v2_threshold_state is not prior_threshold
    assert widget._controls_v2_source_energy_preference is not prior_pref

    assert widget._controls_v2_restore_self_state_verified(ctx) == []
    # every field's DECLARED policy is recorded in the descriptor table
    policies = {name: policy for name, _attr, _key, policy
                in type(widget)._CONTROLS_V2_SELF_STATE_FIELDS}
    assert policies == {
        "gi_selection_explicit": "scalar-value",
        "threshold_state": "value",
        "source_energy_preference": "scalar-value",
    }


# ---------------------------------------------------------------------------
# PRESERVE list (§22 "must be preserved" + Codex-explicit)
# ---------------------------------------------------------------------------

def test_preserved_push_before_setter_and_bound_original_readback(
        widget, monkeypatch):
    """PRESERVE: the carrier is pushed onto the rollback stack BEFORE its setter,
    and the readback reads the BOUND ORIGINAL — so a setter that half-writes and
    raises is still rolled back."""
    original = widget._controls_v2_param(MASK)
    prior = original.value()
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r2-push.edf")])
    real_set = type(original).setValue

    def half_write_then_raise(self, value, *args, **kwargs):
        # Only the FORWARD write misbehaves; the rollback setter is real, so the
        # test proves the carrier reached the rollback stack before its setter.
        if self is original and value == "/tmp/t25r2-push.edf":
            real_set(self, "/tmp/t25r2-half-written.edf")
            raise RuntimeError("setter failed after a partial write")
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(original), "setValue", half_write_then_raise)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert original.value() == prior


def test_preserved_replacement_before_write_is_untouched(widget, monkeypatch):
    """PRESERVE: replacement detection + non-mutation of the replacement."""
    original = widget._controls_v2_param(MASK)
    original_prior = original.value()
    replacement = Parameter.create(
        name="replacement-mask", type="str", value="/tmp/replacement-prior.edf")
    replacement_prior = replacement.value()
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r2-replaced.edf")])

    real_lookup = widget._controls_v2_param
    calls = {"n": 0}

    def switching(p):
        if tuple(p) != MASK:
            return real_lookup(p)
        calls["n"] += 1
        # preflight binds the ORIGINAL; the pre-write identity guard then sees a
        # replacement that took over the path.
        return original if calls["n"] == 1 else replacement

    monkeypatch.setattr(widget, "_controls_v2_param", switching)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert original.value() == original_prior
    assert replacement.value() == replacement_prior
