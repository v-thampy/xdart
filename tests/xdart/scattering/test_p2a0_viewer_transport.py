from __future__ import annotations

import importlib
import inspect
import os
import threading
import time
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from xrd_tools.session.hydration import (
    HydrationOutcome, HydrationPurpose, HydrationReadKey, HydrationScope,
    HydrationToken,
)

_EXPECTED_R = 11_546_624

try:
    _PHYSICAL_RAM = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
except (ValueError, OSError, AttributeError):
    _PHYSICAL_RAM = 0
_EXPECTED_B = 884_736_000 if _PHYSICAL_RAM <= 0 else min(1024**3, _PHYSICAL_RAM // 20)


@pytest.fixture(autouse=True)
def _load_modules():
    global dc, ht, reader
    dc = importlib.import_module("xdart.modules.display_context")
    ht = importlib.import_module("xdart.gui.tabs.scattering.hydration_transport")
    reader = importlib.import_module("xrd_tools.io.viewer_2d")


def _viewer_request(
    port, gate, generation=1, *, path="unresolved.csv", label="catalog",
    catalog=None, receipt=None, policy=None,
):
    scope = HydrationScope("viewer-token", "viewer-2d", "viewer-2d", gate.epoch)
    key = HydrationReadKey(scope, "viewer-2d", label, HydrationPurpose.PREVIEW)
    token = HydrationToken(key, generation)
    if policy is None:
        policy = reader.Viewer2DFormatPolicy()
    if label == "catalog":
        return dc.Viewer2DCatalogHydrationRequest(path, policy, generation, gate, port, key, token)
    return dc.Viewer2DFrameHydrationRequest(
        label, generation, gate, port, catalog, receipt, policy, key, token)


def _selected_admission(catalog, label):
    fact = next((value for value in catalog.frame_facts if value.label == label), None)
    shape = fact.shape if fact is not None else catalog.source_shape[-2:]
    encoded = (catalog.primary_revision.size if catalog.format_name in
               {"edf", "cbf", "img", "mar3450"} else 0)
    canonical = 8 * shape[0] * shape[1]
    peak = max(3 * canonical, encoded + 4 * canonical)
    return canonical + _EXPECTED_R + max(peak, 6 * canonical)


class _Port:
    def __init__(self, *, commit_outcome=None, cleanup_states=()):
        self.commit_outcome = commit_outcome or HydrationOutcome.HYDRATED
        self.cleanup_states = list(cleanup_states)
        self.calls = []
        self.activations = []
        self.completions = []
        self.transport = None
        self.gate = None
        self.reentrant_result = "unset"

    def _outside(self):
        assert not getattr(self.transport._lock, "_is_owned", lambda: False)()

    def _return_activation(self, request, activation):
        self.calls.append(("activate", request))
        self.activations.append(activation)
        return activation

    def activate(self, request):
        self._outside()
        phase = (dc.Viewer2DReceiptPhase.CATALOG_R
                 if type(request) is dc.Viewer2DCatalogHydrationRequest
                 else dc.Viewer2DReceiptPhase.FRAME_A)
        identity = (object() if phase is dc.Viewer2DReceiptPhase.CATALOG_R
                    else request.receipt_identity)
        reserved = (_EXPECTED_R if phase is dc.Viewer2DReceiptPhase.CATALOG_R
                    else _selected_admission(request.catalog, request.label))
        return self._return_activation(
            request, _activation(request.token, phase, identity, reserved))

    def dispose(self, disposal):
        self._outside()
        self.calls.append(("dispose", disposal))
        state = self.cleanup_states.pop(0) if self.cleanup_states else dc.Viewer2DCleanupState.CLEANED
        return dc.Viewer2DCleanupReceipt(disposal.token, state)

    def commit(self, prepared):
        self._outside()
        self.calls.append(("commit", prepared))
        return self.commit_outcome

    def complete(self, completion):
        self._outside()
        self.calls.append(("complete", completion))
        self.completions.append(completion)


def _wait(ticket, timeout=3):
    deadline = time.monotonic() + timeout
    while ticket.result() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert ticket.result() is not None
    return ticket.result()


def _activation(token, phase, identity, reserved, *, capacity=_EXPECTED_B):
    return dc.Viewer2DReadActivation(token, dc.Viewer2DReadBudgetReceipt(
        identity, capacity, reserved, phase, token), phase)


def _bypass_forge(value, **changes):
    forged = object.__new__(type(value))
    for field in fields(value):
        member = changes.get(field.name, getattr(value, field.name))
        object.__setattr__(forged, field.name, member)
    return forged


def _viewer_case(tmp_path, cleanup_states=(), value=None):
    path = tmp_path / "matrix.npy"
    np.save(path, np.arange(6).reshape(2, 3) if value is None else value)
    catalog = reader.catalog_viewer_2d(path)
    gate, identity = dc.Viewer2DCommitGate(), object()
    port = _Port(cleanup_states=cleanup_states)
    transport = ht.HydrationTransport(lambda value: None, lambda value: (None, None))
    port.transport = transport
    return path, catalog, gate, identity, port, transport


def _await_blocked(transport, token, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        outer = transport.blocked_cleanup_token(token)
        if outer is not None:
            return outer
        time.sleep(0.005)
    pytest.fail("transport never entered cleanup-blocked custody")


def _gated_thread_factory(real_thread, entered, release):
    def build(*, target, **kwargs):
        def gated():
            entered.set()
            assert release.wait(2)
            target()
        return real_thread(target=gated, **kwargs)
    return build


def _disposal_case(tmp_path):
    path, catalog, gate, identity, port, _ = _viewer_case(tmp_path)
    catalog_request = _viewer_request(port, gate, path=str(path))
    catalog_activation = _activation(
        catalog_request.token, dc.Viewer2DReceiptPhase.CATALOG_R, object(), _EXPECTED_R)
    catalog_prepared = dc.Prepared2DCatalogCommit(catalog_request, catalog_activation, catalog)
    frame_request = _viewer_request(
        port, gate, label=0, catalog=catalog, receipt=identity)
    admission = _selected_admission(catalog, 0)
    frame_activation = _activation(
        frame_request.token, dc.Viewer2DReceiptPhase.FRAME_A, identity, admission)
    frame = reader.read_viewer_2d_frame(catalog, 0)
    frame_prepared = dc.Prepared2DFrameCommit(
        frame_request, frame_activation, frame, identity)
    return SimpleNamespace(
        catalog=catalog, catalog_request=catalog_request,
        catalog_activation=catalog_activation, catalog_prepared=catalog_prepared,
        frame_request=frame_request, frame_activation=frame_activation,
        frame_prepared=frame_prepared, admission=admission)


def _assert_history(transport, observed, expected):
    actual = [(item.token, item.outcome) for item in observed]
    assert actual == expected
    assert [(item.token, item.outcome) for item in transport.completions()] == expected
    counters = transport.counters()
    assert all(counters[outcome] == sum(item is outcome for _, item in expected)
               for outcome in HydrationOutcome)
    assert sum(counters.values()) == len(expected)


def _assert_released(transport, gate):
    deadline = time.monotonic() + 2
    while ((transport.active_token is not None or transport.queued_token is not None
            or transport.worker is not None) and time.monotonic() < deadline):
        time.sleep(0.005)
    assert transport.active_token is transport.queued_token is None
    assert transport.worker is None and not transport.retains_gate(gate)
    assert transport.retire(join_timeout=1)


def _forbid_viewer_io(monkeypatch):
    monkeypatch.setattr(ht, "catalog_viewer_2d", pytest.fail)
    monkeypatch.setattr(ht, "read_viewer_2d_frame", pytest.fail)


def _start_gated_thread(entered, release, failures, *, fail=False):
    class GatedThread(threading.Thread):
        def start(self):
            entered.set()
            if not release.wait(3):
                failures.append("start barrier timed out")
                return
            if fail:
                raise RuntimeError("controlled start failure")
            super().start()
    return GatedThread


def _without_slot(value, name):
    forged = object.__new__(type(value))
    for field in fields(value):
        if field.name != name:
            object.__setattr__(forged, field.name, getattr(value, field.name))
    assert not hasattr(forged, name)
    return forged


def _raises(expected, operation, *args, **kwargs):
    with pytest.raises(expected) as caught:
        operation(*args, **kwargs)
    return caught.value


def _activation_variant(request, activation, axis):
    if axis == "refused":
        return dc.Viewer2DReadActivation(request.token, None, None, False, "budget refused")
    if axis in {"foreign", "foreign-activation"}:
        return object()
    if axis == "foreign-receipt":
        return _bypass_forge(activation, receipt=object())
    if axis == "malformed-scalar":
        axis = "malformed-capacity"
    receipt = activation.receipt
    if axis in {"negative-capacity", "negative-reserved"}:
        field = axis.removeprefix("negative-")
        return _bypass_forge(activation, receipt=_bypass_forge(receipt, **{field: -1}))
    if axis in {"foreign-token", "token-disagreement"}:
        other = _viewer_request(request.port, dc.Viewer2DCommitGate(), 99).token
        receipt = _bypass_forge(receipt, request_token=other)
        token = other if axis == "foreign-token" else activation.token
        return _bypass_forge(activation, token=token, receipt=receipt)
    if axis in {"unreadable-phase", "phase-disagreement"}:
        receipt_phase = ("catalog_r" if axis == "unreadable-phase" else
                         dc.Viewer2DReceiptPhase.FRAME_A)
        receipt = _bypass_forge(receipt, phase=receipt_phase)
        phase = receipt_phase if axis == "unreadable-phase" else activation.phase
        return _bypass_forge(activation, receipt=receipt, phase=phase)
    if axis.startswith("malformed-"):
        field = axis.removeprefix("malformed-")
        if field in {"accepted", "diagnostic"}:
            return _bypass_forge(activation, **{field: object()})
        receipt = _bypass_forge(receipt, **{field: True})
        return _bypass_forge(activation, receipt=receipt)
    if axis == "identity-none":
        receipt = _bypass_forge(receipt, identity=None)
        return _bypass_forge(activation, receipt=receipt)
    if axis.endswith("wrong-phase"):
        phase = (dc.Viewer2DReceiptPhase.CATALOG_R
                 if activation.phase is dc.Viewer2DReceiptPhase.FRAME_A
                 else dc.Viewer2DReceiptPhase.FRAME_A)
        receipt = _bypass_forge(receipt, phase=phase)
        return _bypass_forge(activation, receipt=receipt, phase=phase)
    reserved = receipt.reserved + int(axis.endswith(("wrong-r", "wrong-a")))
    receipt = _bypass_forge(receipt,
        capacity=_EXPECTED_B - int(axis.endswith("wrong-b")), reserved=reserved)
    return _bypass_forge(activation, receipt=receipt)


def test_viewer_values_are_qt_free_exact_and_port_has_only_four_operations():
    assert dc.ContextKind.VIEWER_2D.value == "viewer_2d"
    assert {state.value for state in dc.Viewer2DState} == {
        "empty", "catalog_loading", "frame_loading", "ready", "cleanup_pending", "closed"
    }
    assert {phase.value for phase in dc.Viewer2DReceiptPhase} == {
        "catalog_r", "frame_a", "frame_ready_a", "cleanup_pending", "released"
    }
    operations = {
        name for name, value in dc.TwoDViewerCommitPort.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    assert operations == {"activate", "dispose", "commit", "complete"}
    source = inspect.getsource(importlib.import_module("xdart.modules.display_context"))
    assert "numpy" not in source and "pyqt" not in source.lower()

    assert dc.Viewer2DCommitGate.__bases__ == (dc.CommitGate,)
    assert dc.Viewer2DCommitGate.__slots__ == ()
    gate = dc.Viewer2DCommitGate()
    port = _Port()
    request = _viewer_request(port, gate)
    assert request.path == "unresolved.csv"
    assert request.read_key.frame_identity == "catalog"
    assert request.read_key.artifact_identity == "viewer-2d"
    assert request.scope.source == request.scope.scan_key == "viewer-2d"
    assert all("path" not in name for name in ("context_token", "scan_key", "source"))
    _raises(TypeError, dc.Viewer2DContext, "viewer", 1, "input.csv", dc.CommitGate())
    _raises(ValueError, _viewer_request, port, dc.CommitGate())


def test_viewer_request_local_types_and_cleanup_values_are_strict(tmp_path):
    case = _disposal_case(tmp_path)
    request = case.catalog_request
    _raises(TypeError, dc.Viewer2DReadBudgetReceipt, object(), 100, 0,
            dc.Viewer2DReceiptPhase.CATALOG_R, request.token)
    invalid_catalog = SimpleNamespace(catalog_identity="x")
    _raises(TypeError, dc.Prepared2DCatalogCommit,
            request, case.catalog_activation, invalid_catalog)
    _raises(TypeError, _viewer_request, request.port, request.commit_gate, label=0,
            catalog=SimpleNamespace(frame_labels=(0,), catalog_identity="x",
                                    policy_identity=request.policy.identity), receipt=object())
    _raises(TypeError, dc.Viewer2DDisposal, request, case.catalog_activation,
            SimpleNamespace(request=request), dc.Viewer2DDisposalToken())
    disposal = dc.Viewer2DDisposal(
        request, case.catalog_activation, case.catalog_prepared, dc.Viewer2DDisposalToken())
    assert dc.Viewer2DCleanupReceipt(
        disposal.token, dc.Viewer2DCleanupState.CLEANED).token is disposal.token
    _raises(TypeError, dc.Viewer2DRendererClearRequest,
            "viewer", 1, case.catalog.catalog_identity, -1)


@pytest.mark.parametrize(
    "axis", ["normal-phase", "normal-frame-receipt", "normal-frame-a",
             "prepared-activation", "prepared-request", "catalog-matching-none",
             "catalog-phase-mismatch", "catalog-b-mismatch", "catalog-r-mismatch",
             "frame-phase-mismatch", "frame-b-mismatch", "frame-a-mismatch",
             "frame-matching-none", "frame-foreign-none", "foreign-token",
             "unreadable-phase", "phase-disagreement", "token-disagreement",
             "malformed-capacity", "malformed-reserved", "negative-capacity",
             "negative-reserved", "identity-none",
             "refused-activation", "foreign-activation", "foreign-receipt",
             "malformed-accepted", "malformed-diagnostic"],
    ids=lambda axis: f"r30-f06-{axis}",
)
def test_r30_disposal_binds_normal_custody_or_quarantines_one_mismatch(tmp_path, axis):
    case = _disposal_case(tmp_path)
    token = dc.Viewer2DDisposalToken()
    phase = dc.Viewer2DReceiptPhase.FRAME_A
    accepted = axis.endswith("-mismatch") or axis in {
        "catalog-matching-none", "frame-matching-none"}
    if axis in {"normal-phase", "catalog-phase-mismatch"}:
        receipt = _bypass_forge(case.catalog_activation.receipt, phase=phase)
        target_activation = _bypass_forge(case.catalog_activation,
                                          receipt=receipt, phase=phase)
        target = (case.catalog_request, target_activation,
            _bypass_forge(case.catalog_prepared, activation=target_activation)
            if axis == "normal-phase" else None)
    elif axis == "normal-frame-receipt":
        target = (case.frame_request, case.frame_activation,
                  _bypass_forge(case.frame_prepared, receipt_identity=object()))
    elif axis in {"normal-frame-a", "frame-a-mismatch"}:
        target_activation = _activation(case.frame_request.token, phase,
            case.frame_request.receipt_identity, case.admission + 1)
        target = (case.frame_request, target_activation, _bypass_forge(
            case.frame_prepared, activation=target_activation)
            if axis == "normal-frame-a" else None)
    elif axis.endswith("-mismatch"):
        frame_case = axis.startswith("frame-")
        request = case.frame_request if frame_case else case.catalog_request
        normal = case.frame_activation if frame_case else case.catalog_activation
        owner, field, _ = axis.split("-")
        activation = _activation_variant(request, normal, f"{owner}-wrong-{field}")
        target = request, activation, None
    elif axis == "prepared-activation":
        target = (case.frame_request, case.frame_activation,
            _bypass_forge(case.frame_prepared, activation=_activation(
                case.frame_request.token, phase,
                case.frame_request.receipt_identity, case.admission)))
    elif axis == "prepared-request":
        target = (case.catalog_request, case.catalog_activation,
                  _bypass_forge(case.catalog_prepared,
                                request=replace(case.catalog_request)))
    elif axis == "catalog-matching-none":
        target = (case.catalog_request, case.catalog_activation, None)
    elif axis == "frame-matching-none":
        target = (case.frame_request, case.frame_activation, None)
    elif axis == "frame-foreign-none":
        target = (case.frame_request, _activation(
            case.frame_request.token, phase, object(), case.admission), None)
    else:
        variant = (axis if axis == "foreign-activation" else
                   axis.removesuffix("-activation"))
        activation = _activation_variant(
            case.catalog_request, case.catalog_activation, variant)
        target = case.catalog_request, activation, None
    if accepted:
        disposal = dc.Viewer2DDisposal(*target, token)
        assert disposal.request is target[0] and disposal.activation is target[1]
        assert disposal.prepared is target[2] and disposal.token is token
        return
    _raises((TypeError, ValueError), dc.Viewer2DDisposal, *target, token)


@pytest.mark.parametrize(
    "axis", ["frame-receipt", "catalog-foreign-policy", "catalog-wrong-b",
             "catalog-wrong-r", "foreign-selected-provenance",
             "frame-foreign-selected-provenance", "frame-wrong-b", "frame-wrong-a"],
    ids=lambda axis: f"existing-prepared-{axis}",
)
def test_prepared_viewer_cross_fields_remain_isolated_positive_controls(tmp_path, axis):
    case = _disposal_case(tmp_path)
    catalog, frame = case.catalog, case.frame_prepared.frame
    dependency = reader.Viewer2DDependency(catalog.canonical_path, None,
                                           catalog.primary_revision)
    if axis == "foreign-selected-provenance":
        _raises(TypeError, replace, frame.provenance, dependencies=(dependency,))
        return
    if axis.startswith("catalog-"):
        policy = (reader.Viewer2DFormatPolicy(source_root=str(tmp_path))
                  if axis == "catalog-foreign-policy" else None)
        request, activation = case.catalog_request, case.catalog_activation
        if policy is not None:
            request = _bypass_forge(request, policy=policy)
        if axis in {"catalog-wrong-b", "catalog-wrong-r"}:
            receipt = _bypass_forge(
                activation.receipt,
                capacity=_EXPECTED_B - int(axis == "catalog-wrong-b"),
                reserved=_EXPECTED_R + int(axis == "catalog-wrong-r"))
            activation = _bypass_forge(activation, receipt=receipt)
        _raises((TypeError, ValueError), dc.Prepared2DCatalogCommit,
                request, activation, catalog)
        return
    request, activation = case.frame_request, case.frame_activation
    receipt_identity = request.receipt_identity
    if axis == "frame-receipt":
        receipt_identity = object()
    elif axis == "frame-foreign-selected-provenance":
        provenance = _bypass_forge(
            frame.provenance, dependencies=(dependency,),
        )
        frame = _bypass_forge(frame, provenance=provenance)
    else:
        receipt = _bypass_forge(
            activation.receipt,
            capacity=_EXPECTED_B - int(axis == "frame-wrong-b"),
            reserved=case.admission + int(axis == "frame-wrong-a"),
        )
        activation = _bypass_forge(activation, receipt=receipt)
    _raises((TypeError, ValueError), dc.Prepared2DFrameCommit,
            request, activation, frame, receipt_identity)


def test_catalog_activation_precedes_path_work_and_every_port_call_is_detached(
    tmp_path, monkeypatch
):
    path = tmp_path / "matrix.csv"
    path.write_text("1,2\n3,4\n")
    order = []
    original = reader.catalog_viewer_2d

    def catalog(value, **kwargs):
        order.append("path-work")
        return original(value, **kwargs)

    monkeypatch.setattr(ht, "catalog_viewer_2d", catalog)
    gate = dc.Viewer2DCommitGate()
    port = _Port()
    transport = ht.HydrationTransport(lambda value: pytest.fail("detector crossover"), lambda value: (None, None))
    port.transport = transport
    original_activate = port.activate

    def activate(request):
        order.append("activate")
        return original_activate(request)

    port.activate = activate
    request = _viewer_request(port, gate, path=str(path))
    ticket = transport._submit_admission(request)
    assert ticket is not None
    result = _wait(ticket)
    assert result.outcome is HydrationOutcome.HYDRATED
    assert order == ["activate", "path-work"]
    assert [name for name, _ in port.calls] == ["activate", "commit", "complete"]
    assert type(port.calls[1][1]) is dc.Prepared2DCatalogCommit
    assert port.calls[1][1].catalog.frame_labels == (0,)
    assert transport.retire(join_timeout=1)


def test_frame_dispatch_uses_exact_catalog_receipt_identity_and_no_cross_route(tmp_path):
    value = np.arange(24).reshape(2, 3, 4)
    _, catalog, gate, receipt, port, transport = _viewer_case(tmp_path, value=value)
    request = _viewer_request(
        port, gate, generation=4, label=1, catalog=catalog, receipt=receipt)
    result = _wait(transport._submit_admission(request))
    assert type(port.calls[1][1]) is dc.Prepared2DFrameCommit
    prepared = port.calls[1][1]
    assert prepared.request is request and prepared.receipt_identity is receipt
    assert np.array_equal(prepared.frame.array, value[1])
    assert result.token is request.token
    assert transport.retire(join_timeout=1)


def test_viewer_requests_never_same_read_coalesce_and_latest_slot_stays_bounded(
    tmp_path, monkeypatch
):
    path, _, gate, _, port, transport = _viewer_case(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = reader.catalog_viewer_2d
    calls = 0

    def blocked(value, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(2)
        return original(value, **kwargs)

    monkeypatch.setattr(ht, "catalog_viewer_2d", blocked)
    first = transport._submit_admission(_viewer_request(port, gate, 1, path=str(path)))
    assert entered.wait(2)
    second = transport._submit_admission(_viewer_request(port, gate, 2, path=str(path)))
    third = transport._submit_admission(_viewer_request(port, gate, 3, path=str(path)))
    assert first is not second and second is not third
    release.set()
    assert _wait(second).outcome.value == "superseded"
    _wait(first)
    _wait(third)
    assert calls == 2
    assert transport.queued_token is None
    assert transport.retire(join_timeout=1)


@pytest.mark.parametrize(
    "axis", ["worker-dequeue", "stale-finalized"],
    ids=lambda axis: f"r30-f08-{axis}",
)
def test_r30_detached_replacement_cannot_be_dequeued_while_delivery_is_pending(
    tmp_path, monkeypatch, axis,
):
    path, _, gate, _, port, transport = _viewer_case(tmp_path)
    first = transport.submit_detached(_viewer_request(port, gate, 1, path=str(path)))
    latest = transport.submit_detached(_viewer_request(port, gate, 2, path=str(path)))
    delivery = latest.deliveries[0]
    if axis == "worker-dequeue":
        real_thread = threading.Thread
        entered, release = threading.Event(), threading.Event()
        worker = None
        monkeypatch.setattr(ht, "Thread", _gated_thread_factory(
            real_thread, entered, release))
        try:
            transport.dispatch_detached(first)
            assert entered.wait(2)
            worker = transport.worker
            with transport._lock:
                assert transport._queued is delivery.entry
                assert delivery.entry.state is ht._EntryState.DELIVERY_PENDING
                assert delivery.entry.delivery_guard is delivery.guard
        finally:
            release.set()
            if worker is not None:
                worker.join(2)
                assert not worker.is_alive()
        monkeypatch.setattr(ht, "Thread", real_thread)
        with transport._lock:
            assert transport._queued is delivery.entry
            assert delivery.entry.state is ht._EntryState.DELIVERY_PENDING
            assert delivery.entry.delivery_guard is delivery.guard
        assert port.completions == []
    else:
        transport.dispatch_detached(ht.DetachedHydrationMutation(
            deliveries=latest.deliveries))
        with transport._lock:
            assert transport._queued is None
            assert delivery.entry.delivery_guard is None
        assert first.ticket.result().token is first.token
        assert first.ticket.result().outcome is HydrationOutcome.SUPERSEDED
        assert latest.ticket.result() is None
        assert not [value for name, value in port.calls
                    if name in {"activate", "commit"}]
    transport.dispatch_detached(latest)
    completion = _wait(latest.ticket)
    assert completion.token is latest.token
    assert completion.outcome is HydrationOutcome.HYDRATED
    expected = [(first.token, HydrationOutcome.SUPERSEDED),
                (latest.token, HydrationOutcome.HYDRATED)]
    assert first.ticket.result().token is first.token
    assert first.ticket.result().outcome is HydrationOutcome.SUPERSEDED
    _assert_history(transport, port.completions, expected)
    activates = [value.token for name, value in port.calls if name == "activate"]
    commits = [value.request.token for name, value in port.calls if name == "commit"]
    assert activates == [latest.token]
    assert commits == [latest.token]
    assert transport.retire(join_timeout=1)


def test_detector_same_read_coalescing_and_constructor_bound_route_are_preserved(monkeypatch):
    owner = dc.HydrationOwner("ctx", "scan", "source", 1)
    gate = dc.CommitGate()
    scope = HydrationScope(*owner.as_tuple())
    key = HydrationReadKey(scope, "artifact", 1, HydrationPurpose.PREVIEW)
    entered, release = threading.Event(), threading.Event()

    def read(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(detector_diagnostic=None)

    monkeypatch.setattr(ht, "read_frame_preview", read)
    committed, completed = [], []

    def commit(prepared):
        committed.append(prepared)
        return HydrationOutcome.HYDRATED

    transport = ht.HydrationTransport(
        commit, lambda request: (None, None), completion_sink=completed.append)
    stores = (object(), object())

    def request(generation):
        token = HydrationToken(key, generation)
        return dc.HydrationRequest(
            1, HydrationPurpose.PREVIEW, generation, owner, stores, gate,
            read_key=key, token=token)

    first = transport._submit_admission(request(1))
    assert entered.wait(2)
    worker = transport.worker
    mutation = transport.submit_detached(request(2))
    second = mutation.ticket
    delivery = mutation.deliveries[0]
    entry = transport._active
    assert first.result().token is first.token
    assert first.result().outcome.value == "superseded"
    assert entry.state is ht._EntryState.DELIVERY_PENDING
    assert entry.delivery_guard is delivery.guard
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    with transport._lock:
        assert transport._active is entry
        assert entry.token is mutation.token
        assert entry.ticket is mutation.ticket
        assert entry.state is ht._EntryState.DELIVERY_PENDING
        assert entry.delivery_guard is delivery.guard
    assert mutation.ticket.result() is None
    assert committed == []
    transport.dispatch_detached(mutation)
    result = _wait(second)
    assert result.token is second.token
    assert result.outcome.value == "hydrated"
    assert len(committed) == 1
    assert committed[0].token is second.token
    expected = [(first.token, HydrationOutcome.SUPERSEDED),
                (second.token, HydrationOutcome.HYDRATED)]
    _assert_history(transport, completed, expected)
    assert transport.retire(join_timeout=1)


def test_completion_reentry_cannot_overwrite_delivery_pending_entry(tmp_path):
    path, _, gate, _, port, transport = _viewer_case(tmp_path)
    first_request = _viewer_request(port, gate, 1, path=str(path))
    reentrant = _viewer_request(port, gate, 2, path=str(path))
    original_complete = port.complete

    def complete(completion):
        port.reentrant_result = transport.submit(reentrant)
        original_complete(completion)

    port.complete = complete
    _wait(transport._submit_admission(first_request))
    assert port.reentrant_result is None
    assert transport.retire(join_timeout=1)


def test_failed_unadopted_disposal_retains_exact_cleanup_custody_until_monotonic_retry(tmp_path):
    path, _, gate, _, port, transport = _viewer_case(
        tmp_path, cleanup_states=(dc.Viewer2DCleanupState.CLEANUP_PENDING,
                                  dc.Viewer2DCleanupState.CLEANUP_PENDING,
                                  dc.Viewer2DCleanupState.CLEANED))
    port.commit_outcome = HydrationOutcome.OWNER_MISMATCH
    request = _viewer_request(port, gate, 1, path=str(path))
    ticket = transport._submit_admission(request)
    outer = _await_blocked(transport, request.token)
    latest_request = _viewer_request(port, gate, 2, path=str(path))
    latest = transport._submit_admission(latest_request)
    assert latest is not None and transport.queued_token is latest_request.token
    assert ticket.result() is None and transport.retains_gate(gate)
    assert not transport.retire(join_timeout=0)
    assert transport.blocked_cleanup_token(request.token) is outer

    first = transport.retry_blocked_cleanup(outer)
    assert first.state is dc.Viewer2DCleanupState.CLEANUP_PENDING
    assert transport.blocked_cleanup_token(request.token) is outer
    stale = transport.retry_blocked_cleanup(object())
    assert stale.state is dc.Viewer2DCleanupState.STALE
    final = transport.retry_blocked_cleanup(outer)
    assert final.state is dc.Viewer2DCleanupState.CLEANED
    assert _wait(ticket).outcome is HydrationOutcome.OWNER_MISMATCH
    assert _wait(latest).outcome is HydrationOutcome.CANCELLED
    assert len(port.completions) == 2 and not transport.retains_gate(gate)
    assert transport.retire(join_timeout=1)


def test_r30_activation_mismatch_cleanup_blocked_retries_exact_custody(
    tmp_path, monkeypatch,
):
    path, _, gate, _, port, transport = _viewer_case(
        tmp_path, cleanup_states=(dc.Viewer2DCleanupState.CLEANUP_PENDING,
                                  dc.Viewer2DCleanupState.CLEANED))
    request = _viewer_request(port, gate, path=str(path))
    _forbid_viewer_io(monkeypatch)
    original = port.activate

    def activate(value):
        activation = original(value)
        receipt = _bypass_forge(activation.receipt, capacity=_EXPECTED_B - 1)
        forged = _bypass_forge(activation, receipt=receipt)
        port.activations[-1] = forged
        return forged

    port.activate = activate
    ticket = transport._submit_admission(request)
    outer = _await_blocked(transport, request.token)
    activation, = port.activations
    disposal = next(value for name, value in port.calls if name == "dispose")
    assert ticket.result() is None and transport.retains_gate(gate)
    assert disposal.request is request and disposal.activation is activation
    assert disposal.prepared is None
    assert [name for name, _ in port.calls] == ["activate", "dispose"]
    assert port.completions == [] and transport.completions() == ()
    assert not any(transport.counters().values())
    assert transport.active_token is request.token and transport.queued_token is None
    assert transport.worker is None
    stale = transport.retry_blocked_cleanup(object())
    assert stale.state is dc.Viewer2DCleanupState.STALE
    assert transport.blocked_cleanup_token(request.token) is outer
    assert ticket.result() is None and transport.retains_gate(gate)
    assert port.completions == [] and not any(transport.counters().values())
    receipt = transport.retry_blocked_cleanup(outer)
    assert receipt.state is dc.Viewer2DCleanupState.CLEANED
    assert all(value is disposal for name, value in port.calls if name == "dispose")
    assert _wait(ticket).outcome is HydrationOutcome.FAILED
    assert [name for name, _ in port.calls] == ["activate", "dispose", "dispose", "complete"]
    _assert_history(transport, port.completions, [(request.token, HydrationOutcome.FAILED)])
    _assert_released(transport, gate)


def test_runtime_facade_keeps_viewer_out_of_light_graph_and_detector_target_is_two_stores(monkeypatch):
    runtime = importlib.import_module("xdart.gui.tabs.scattering.display_runtime")
    events = importlib.import_module("xdart.gui.tabs.scattering.events")
    values = importlib.import_module("xdart.gui.tabs.scattering.display_values")
    state = runtime.RunDisplayState(events.RunIdentity(1, "run"), max_payload_items=1)

    class FakeTransport:
        def __init__(self):
            self.requests = []
            self.dispatched = []
            self.gate_calls = []

        def submit_detached(self, request, *, closed=False):
            assert not state._light_admission_lock.acquire(blocking=False)
            self.requests.append((request, closed))
            return type("Mutation", (), {"token": request.token})()

        def submit(self, request, *, closed=False):
            pytest.fail("synchronous submit entered the admission lock")

        def dispatch_detached(self, mutation):
            assert state._light_admission_lock.acquire(blocking=False)
            state._light_admission_lock.release()
            self.dispatched.append(mutation)

        def cancel_gate_detached(self, gate):
            assert not state._light_admission_lock.acquire(blocking=False)
            self.gate_calls.append(("cancel", gate))
            return type("Mutation", (), {"token": None})()

        def cancel_gate(self, gate):
            pytest.fail("synchronous cancellation entered the admission lock")

        def retains_gate(self, gate):
            self.gate_calls.append(("retains", gate))
            return True

    fake = FakeTransport()
    state._transport = fake
    records, publications = object(), object()
    art = runtime.DisplayArtifact(Path("artifact.nxs"), "scan", records, publications)
    owner = dc.HydrationOwner("ctx", "scan", "source", 1)
    gate = dc.CommitGate()
    key = values.DisplayFrameKey(state.identity, "scan", "artifact.nxs", 7, 1)
    state._request_preview(art, key, 3, False, owner, gate)
    request = fake.requests[0][0]
    assert request.stores == (records, publications) and len(request.stores) == 2
    assert fake.dispatched

    port = _Port()
    viewer_gate = dc.Viewer2DCommitGate()
    viewer = _viewer_request(port, viewer_gate)
    state.submit_viewer_2d(viewer)
    assert fake.requests[-1][0] is viewer
    base_gate = dc.CommitGate()
    forged = object.__new__(dc.Viewer2DCatalogHydrationRequest)
    for field in fields(dc.Viewer2DCatalogHydrationRequest):
        object.__setattr__(
            forged, field.name,
            base_gate if field.name == "commit_gate" else getattr(viewer, field.name),
        )
    before = len(fake.requests), len(fake.dispatched), len(fake.gate_calls)
    assert state.submit_viewer_2d(forged) is None
    assert (len(fake.requests), len(fake.dispatched), len(fake.gate_calls)) == before
    state.cancel_viewer_2d(base_gate)
    assert not state.viewer_2d_retains_gate(base_gate)
    assert fake.gate_calls == []
    state.cancel_viewer_2d(viewer_gate)
    assert state.viewer_2d_retains_gate(viewer_gate)
    assert fake.gate_calls == [("cancel", viewer_gate), ("retains", viewer_gate)]
    assert art.light_lease is art.light_slot is art.light_hooks is None
    assert not hasattr(art, "light_records")


def test_runtime_light_cancel_unsubscribes_then_captures_and_dispatches_outside_lock():
    runtime = importlib.import_module("xdart.gui.tabs.scattering.display_runtime")
    events = importlib.import_module("xdart.gui.tabs.scattering.events")
    state = runtime.RunDisplayState(events.RunIdentity(2, "run"), max_payload_items=1)
    order = []

    class FakeTransport:
        def cancel_gate_detached(self, gate):
            assert not state._light_admission_lock.acquire(blocking=False)
            order.append(("capture", gate))
            return object()

        def dispatch_detached(self, mutation):
            assert state._light_admission_lock.acquire(blocking=False)
            state._light_admission_lock.release()
            order.append(("dispatch", mutation))

        def cancel_gate(self, gate):
            pytest.fail("synchronous cancellation entered the admission lock")

    state._transport = FakeTransport()
    artifact = runtime.DisplayArtifact(Path("artifact.nxs"), "scan", object(), object())
    gate = dc.CommitGate()
    artifact.light_unsubscribe = lambda: order.append(("unsubscribe", None))
    state.cancel_light_1d(artifact, gate)
    assert [name for name, _ in order] == ["unsubscribe", "capture", "dispatch"]
    assert order[1][1] is gate and artifact.light_unsubscribe is None


@pytest.mark.parametrize("worker_mode", ["start-failure", "inline-fast", "start-handoff"])
def test_real_runtime_worker_start_and_fast_completion_are_post_admission(
    monkeypatch, worker_mode,
):
    runtime = importlib.import_module("xdart.gui.tabs.scattering.display_runtime")
    events = importlib.import_module("xdart.gui.tabs.scattering.events")
    values = importlib.import_module("xdart.gui.tabs.scattering.display_values")
    entered, release, terminal = threading.Event(), threading.Event(), threading.Event()
    starts, late_starts = [], []

    class Worker:
        ident = None

        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            if worker_mode == "start-failure":
                raise RuntimeError("no worker")
            if worker_mode == "start-handoff":
                entered.set()
                assert release.wait(2)
                (late_starts if terminal.is_set() else starts).append(1)
            self.target()

        def is_alive(self):
            return False

    def fail_read(*args, **kwargs):
        raise RuntimeError("fast read failure")

    monkeypatch.setattr(ht, "Thread", Worker)
    monkeypatch.setattr(ht, "read_frame_preview", fail_read)
    state = runtime.RunDisplayState(events.RunIdentity(3, "run"), max_payload_items=1)
    completed = []

    def completion(value):
        assert state._light_admission_lock.acquire(blocking=False)
        state._light_admission_lock.release()
        completed.append(value)

    state._transport._completion_sink = completion
    artifact = runtime.DisplayArtifact(Path("artifact.nxs"), "scan", object(), object())
    owner = dc.HydrationOwner("ctx", "scan", "source", 1)
    gate = dc.CommitGate()
    key = values.DisplayFrameKey(state.identity, "scan", "artifact.nxs", 0, 1)
    if worker_mode == "start-handoff":
        expected = HydrationToken(HydrationReadKey(
            HydrationScope(*owner.as_tuple()), key.artifact, key.local_frame_label,
            HydrationPurpose.PREVIEW), 1)
        launcher = threading.Thread(
            target=lambda: state._request_preview(artifact, key, 1, False, owner, gate)
        )
        launcher.start()
        assert entered.wait(2)
        try:
            first_retire = state._transport.retire(join_timeout=0)
        finally:
            release.set()
            launcher.join(2)
        assert not launcher.is_alive()
        later_retire = state._transport.retire(join_timeout=1)
        assert first_retire is False and later_retire is True
        assert starts == [1] and late_starts == []
        _assert_history(state._transport, completed,
                        [(expected, HydrationOutcome.CANCELLED)])
        assert (state._transport.active_token, state._transport.queued_token,
                state._transport.worker) == (None, None, None)
        assert not state._transport.retains_gate(gate)
        snapshot = tuple(completed), state._transport.completions(), state._transport.counters()
        terminal.set()
        state._request_preview(artifact, key, 2, False, owner, gate)
        assert (tuple(completed), state._transport.completions(),
                state._transport.counters()) == snapshot
        assert (state._transport.active_token, state._transport.queued_token,
                state._transport.worker) == (None, None, None)
        assert starts == [1] and late_starts == [] and not state._transport.retains_gate(gate)
        return
    state._request_preview(artifact, key, 1, False, owner, gate)
    assert len(completed) == 1 and completed[0].outcome.value == "failed"
    assert not state._transport.retains_gate(gate)


@pytest.mark.parametrize(
    "mode", ["exception", "none", "foreign", "refused", "foreign-token",
             "frame-foreign-identity", "unreadable-phase", "token-disagreement",
             "phase-disagreement", "malformed-scalar", "malformed-reserved",
             "negative-capacity", "negative-reserved",
             "malformed-accepted", "malformed-diagnostic", "foreign-receipt",
             "identity-none"],
    ids=lambda mode: f"r30-f09-inert-{mode}",
)
def test_invalid_or_refused_activation_is_total_and_releases_ticket_and_gate(
    tmp_path, monkeypatch, mode
):
    path, catalog, gate, identity, port, transport = _viewer_case(tmp_path)
    frame_case = mode == "frame-foreign-identity"
    request = (_viewer_request(port, gate, label=0, catalog=catalog, receipt=identity)
               if frame_case else _viewer_request(port, gate, path=str(path)))
    _forbid_viewer_io(monkeypatch)

    returned = []

    def activate(value):
        port._outside()
        if mode == "exception":
            port.calls.append(("activate", value))
            raise RuntimeError("activation failed")
        if mode == "none":
            activation = None
            returned.append(activation)
            return port._return_activation(value, activation)
        phase = (dc.Viewer2DReceiptPhase.FRAME_A if frame_case
                 else dc.Viewer2DReceiptPhase.CATALOG_R)
        reserved = (_selected_admission(catalog, 0) if frame_case else _EXPECTED_R)
        activation = _activation(value.token, phase, object(), reserved)
        if frame_case:
            activation = _bypass_forge(
                activation, receipt=_bypass_forge(activation.receipt, identity=object()))
        activation = _activation_variant(value, activation, mode)
        returned.append(activation)
        return port._return_activation(value, activation)

    port.activate = activate
    result = _wait(transport._submit_admission(request), timeout=1)
    assert result.token is request.token and result.outcome is HydrationOutcome.FAILED
    assert len(port.activations) == len(returned) and all(left is right for left, right in zip(port.activations, returned))
    assert [name for name, _ in port.calls] == ["activate", "complete"]
    assert not [value for name, value in port.calls if name == "dispose"]
    _assert_released(transport, gate)
    _assert_history(transport, port.completions, [(request.token, HydrationOutcome.FAILED)])


@pytest.mark.parametrize(
    "axis", ["catalog-wrong-phase", "frame-wrong-phase", "catalog-wrong-b",
             "catalog-wrong-r", "frame-wrong-b", "frame-wrong-a",
             "catalog-read-failure"],
    ids=lambda axis: f"r30-f09-owning-{axis}",
)
def test_owning_activation_mismatch_disposes_once_before_any_io(
    tmp_path, monkeypatch, axis,
):
    path, catalog, gate, identity, port, transport = _viewer_case(tmp_path)
    frame_case = axis.startswith("frame-")
    request = (_viewer_request(port, gate, path=str(path)) if not frame_case else
               _viewer_request(port, gate, label=0,
                   catalog=catalog, receipt=identity))
    if axis == "catalog-read-failure":
        monkeypatch.setattr(ht, "catalog_viewer_2d",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("read failed")))
        monkeypatch.setattr(ht, "read_viewer_2d_frame", pytest.fail)
    else:
        _forbid_viewer_io(monkeypatch)
    original = port.activate

    def activate(value):
        returned = original(value)
        if axis == "catalog-read-failure":
            return returned
        forged = _activation_variant(value, returned, axis)
        port.activations[-1] = forged
        return forged

    port.activate = activate
    result = _wait(transport._submit_admission(request), timeout=1)
    disposal = next(value for name, value in port.calls if name == "dispose")
    activation, = port.activations
    assert result.token is request.token and result.outcome is HydrationOutcome.FAILED
    assert disposal.request is request and disposal.activation is activation
    assert disposal.prepared is None
    assert [name for name, _ in port.calls] == ["activate", "dispose", "complete"]
    _assert_history(transport, port.completions, [(request.token, HydrationOutcome.FAILED)])
    _assert_released(transport, gate)


def test_delivery_pending_retains_custody_and_refuses_reentrant_mutation(tmp_path):
    path, _, gate, _, port, transport = _viewer_case(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = port.complete

    def complete(value):
        entered.set()
        assert release.wait(2)
        original(value)

    port.complete = complete
    ticket = transport._submit_admission(_viewer_request(port, gate, path=str(path)))
    assert entered.wait(2)
    assert ticket.result() is not None and transport.retains_gate(gate)
    assert transport.submit(_viewer_request(port, gate, 2, path=str(path))) is None
    transport.cancel_gate(gate)
    assert transport.retains_gate(gate) and not transport.retire(join_timeout=0)
    release.set()
    deadline = time.monotonic() + 2
    while transport.retains_gate(gate) and time.monotonic() < deadline:
        time.sleep(0.005)
    _assert_released(transport, gate)


def test_worker_start_failure_settles_once_without_gate_custody(tmp_path, monkeypatch):

    class BrokenThread:
        ident = None

        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("no worker")

        def is_alive(self):
            return False

    path, _, gate, _, port, transport = _viewer_case(tmp_path)
    monkeypatch.setattr(ht, "Thread", BrokenThread)
    result = _wait(transport._submit_admission(_viewer_request(port, gate, path=str(path))))
    assert result.outcome is HydrationOutcome.FAILED
    assert len(port.completions) == 1 and not transport.retains_gate(gate)
    assert transport.retire(join_timeout=0)


def test_cleanup_retry_worker_slot_handoff_survives_stale_worker_unwind(
    tmp_path, monkeypatch,
):
    path, _, gate, _, port, transport = _viewer_case(
        tmp_path, cleanup_states=(dc.Viewer2DCleanupState.CLEANUP_PENDING,
                                  dc.Viewer2DCleanupState.CLEANED))
    port.commit_outcome = HydrationOutcome.OWNER_MISMATCH
    stale_ready, release_stale = threading.Event(), threading.Event()
    start_entered, release_start = threading.Event(), threading.Event()
    failures, retry_receipts, stale_workers = [], [], []
    original_execute = transport._execute
    def execute(*args):
        result = original_execute(*args)
        stale_workers.append(threading.current_thread())
        stale_ready.set()
        if not release_stale.wait(3): failures.append("stale barrier timed out")
        return result

    transport._execute = execute
    first_request = _viewer_request(port, gate, 1, path=str(path))
    first = transport._submit_admission(first_request)
    assert stale_ready.wait(2)
    outer = _await_blocked(transport, first_request.token)
    second_request = _viewer_request(port, gate, 2, path=str(path))
    second = transport._submit_admission(second_request)
    monkeypatch.setattr(ht, "Thread", _start_gated_thread(
        start_entered, release_start, failures))
    retry = threading.Thread(target=lambda: retry_receipts.append(
        transport.retry_blocked_cleanup(outer)), daemon=True)
    retry.start()
    assert start_entered.wait(2)
    published = transport.worker
    try:
        release_stale.set()
        stale_workers[0].join(3)
        slot_preserved = transport.worker is published
        retired_early = transport.retire(join_timeout=0)
    finally:
        release_start.set()
        retry.join(3)
        if published is not None and published.ident is not None:
            published.join(3)
    assert (failures == [] and not stale_workers[0].is_alive()
            and retry_receipts[0].state is dc.Viewer2DCleanupState.CLEANED)
    assert slot_preserved and retired_early is False
    assert first.result().outcome is HydrationOutcome.OWNER_MISMATCH
    assert second.result().outcome is HydrationOutcome.CANCELLED
    _assert_history(transport, port.completions, [
        (first.token, HydrationOutcome.OWNER_MISMATCH),
        (second.token, HydrationOutcome.CANCELLED)])
    _assert_released(transport, gate)


def test_retire_cancellation_wins_over_published_start_failure(tmp_path, monkeypatch):
    path, _, gate, _, port, transport = _viewer_case(tmp_path)
    start_entered, release_start = threading.Event(), threading.Event()
    cancel_entered, release_cancel = threading.Event(), threading.Event()
    failures, retire_results = [], []
    monkeypatch.setattr(ht, "Thread", _start_gated_thread(
        start_entered, release_start, failures, fail=True))
    mutation = transport.submit_detached(_viewer_request(port, gate, path=str(path)))
    entry = transport._queued
    launcher = threading.Thread(target=lambda: transport.dispatch_detached(mutation), daemon=True)
    launcher.start()
    assert start_entered.wait(2)
    original_complete = port.complete
    def complete(value):
        if value.outcome is HydrationOutcome.CANCELLED:
            cancel_entered.set()
            if not release_cancel.wait(3): failures.append("cancel barrier timed out")
        original_complete(value)

    port.complete = complete
    retiring = threading.Thread(
        target=lambda: retire_results.append(transport.retire(join_timeout=0)), daemon=True)
    retiring.start()
    assert cancel_entered.wait(2)
    with transport._lock:
        guard = entry.delivery_guard
    try:
        release_start.set()
        launcher.join(3)
        with transport._lock:
            guard_preserved = (transport._queued is entry
                and entry.state is ht._EntryState.DELIVERY_PENDING
                and entry.delivery_guard is guard)
    finally:
        release_cancel.set()
        launcher.join(3)
        retiring.join(3)
    assert failures == [] and not launcher.is_alive() and not retiring.is_alive()
    assert guard_preserved and retire_results == [False]
    assert mutation.ticket.result().outcome is HydrationOutcome.CANCELLED
    _assert_history(transport, port.completions,
                    [(mutation.token, HydrationOutcome.CANCELLED)])
    _assert_released(transport, gate)


@pytest.mark.parametrize("slot", ["activation", "receipt"],
                         ids=lambda slot: f"r45-t3-missing-exact-slot-{slot}")
def test_missing_exact_activation_slot_is_total_and_releases_custody(
    tmp_path, monkeypatch, slot,
):
    path, _, gate, _, port, transport = _viewer_case(tmp_path)
    _forbid_viewer_io(monkeypatch)
    crashes, workers = [], []
    real_thread = threading.Thread
    def guarded_thread(*, target, **kwargs):
        def guarded():
            try: target()
            except BaseException as error: crashes.append(error)
        worker = real_thread(target=guarded, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(ht, "Thread", guarded_thread)
    original = port.activate
    def activate(request):
        normal = original(request)
        forged = (_without_slot(normal, "receipt") if slot == "activation" else
                  _bypass_forge(normal, receipt=_without_slot(normal.receipt, "capacity")))
        port.activations[-1] = forged
        return forged

    port.activate = activate
    request = _viewer_request(port, gate, path=str(path))
    ticket = transport._submit_admission(request)
    workers[0].join(2)
    assert not workers[0].is_alive() and crashes == []
    assert ticket.result().outcome is HydrationOutcome.FAILED
    assert [name for name, _ in port.calls] == ["activate", "complete"]
    _assert_history(transport, port.completions,
                    [(request.token, HydrationOutcome.FAILED)])
    _assert_released(transport, gate)


def test_stage_census_has_one_port_one_lane_and_no_new_queue_cache_or_store():
    entry_names = {item.name for item in fields(ht._TransportEntry)}
    assert {"request", "ticket", "state"} <= entry_names
    transport = ht.HydrationTransport(lambda value: None, lambda value: (None, None))
    allowed = {
        "_commit", "_derive", "_completion_sink", "_lock", "_active", "_queued",
        "_worker", "_retired", "_counters", "_completions",
    }
    assert set(vars(transport)) == allowed
    assert [name for name, value in vars(transport).items()
            if type(value).__name__ == "deque"] == ["_completions"]
    assert not any(word in name.lower() for name in vars(transport)
                   for word in ("cache", "store", "provider", "registry", "scheduler"))
    source = inspect.getsource(ht)
    assert source.count("class TwoDViewerCommitPort") == 0
    assert source.count("deque(") == 1
    assert not any(word in source for word in ("Queue(", "SimpleQueue(", "ThreadPool", "LRU"))
    runtime_source = inspect.getsource(importlib.import_module("xdart.gui.tabs.scattering.display_runtime"))
    assert "light_records" not in runtime_source
    assert "Viewer2D" in "".join(dc.__all__)
