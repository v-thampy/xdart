from __future__ import annotations

import inspect
import threading
import time

import pytest

from xrd_tools.session.hydration import (
    HydrationOutcome,
    HydrationPurpose,
    HydrationReadKey,
    HydrationScope,
    HydrationToken,
)


def _request(dc, reader, transport, port, path, generation):
    gate = port.gate
    scope = HydrationScope("viewer-1d-owner", "viewer-1d", "viewer-1d", gate.epoch)
    key = HydrationReadKey(scope, "viewer-1d", "batch", HydrationPurpose.ONE_D)
    token = HydrationToken(key, generation)
    return dc.Viewer1DBatchHydrationRequest(
        (str(path),), reader.Viewer1DFormatPolicy(), generation, gate, port,
        key, token, port.gui_thread_id, port.owner_identity, port.owner_claim,
        transport,
    )

class _Port:
    def __init__(self, dc, *, adopt=True):
        self.dc, self.adopt = dc, adopt
        self.gate = dc.Viewer1DCommitGate()
        self.gui_thread_id = threading.get_ident()
        self.owner_identity, self.owner_claim = object(), object()
        self.calls, self.completions, self.notices, self.holders = [], [], [], []

    def commit(self, prepared):
        import xrd_tools.session.viewer_1d as reader
        self.calls.append(("commit", prepared.request.generation))
        if not self.adopt:
            return None
        receipt = reader.adopt_prepared_viewer_1d(
            prepared, owner_identity=self.owner_identity,
            owner_request_claim=self.owner_claim, port=self,
            owner_generation=prepared.request.generation,
            commit_gate=prepared.request.commit_gate,
            owner_state=self.dc.Viewer1DState.LOADING)
        self.holders.append(prepared.transfer.owner_holder(receipt))
        return receipt

    def cleanup_pending(self, notice):
        self.calls.append(("pending", notice.request.generation))
        self.notices.append(notice)
        return self._ack(notice)

    def _ack(self, notice):
        request = notice.request
        return self.dc.acknowledge_viewer_1d_cleanup_pending(
            notice, port=self, owner_identity=self.owner_identity,
            owner_request_claim=self.owner_claim, commit_gate=request.commit_gate,
            admitted_provider=request.admitted_provider_identity,
            owner_generation=request.generation,
            owner_state=self.dc.Viewer1DState.CLEANUP_PENDING)

    def complete(self, completion):
        self.calls.append(("complete", completion.token.presentation_generation,
                           completion.outcome))
        self.completions.append(completion)

def _modules():
    import xdart.modules.display_context as dc
    import xdart.gui.tabs.scattering.hydration_transport as ht
    import xrd_tools.session.viewer_1d as reader
    return dc, ht, reader

def _transport(ht):
    return ht.HydrationTransport(lambda value: None, lambda value: (None, None))

def _wait(ticket, timeout=5):
    deadline = time.monotonic() + timeout
    while ticket.result() is None and time.monotonic() < deadline:
        time.sleep(.005)
    assert ticket.result() is not None
    return ticket.result()

def _force_real_cleanup_pending(reader, monkeypatch):
    complete, close = reader.Viewer1DReadOperation.complete, reader._BuildingGraph.close_streams
    failed = []
    def stop_first(operation, permit):
        if operation.request.generation == 1:
            return reader._failure(operation.transfer,
                reader.Viewer1DReadFailureStage.PASS_TWO, "forced pass-two refusal")
        return complete(operation, permit)
    def fail_drain_once(graph):
        if not failed:
            failed.append(True)
            raise RuntimeError("forced drain pending")
        return close(graph)
    monkeypatch.setattr(reader.Viewer1DReadOperation, "complete", stop_first)
    monkeypatch.setattr(reader._BuildingGraph, "close_streams", fail_drain_once)

class _FatalControl(BaseException):
    pass

def _capture_thread_controls(monkeypatch):
    seen, arrived = [], threading.Event()
    def capture(args):
        seen.append(args.exc_value)
        arrived.set()
    monkeypatch.setattr(threading, "excepthook", capture)
    return seen, arrived

def _wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.005)
    assert predicate()


@pytest.mark.parametrize("forgery", ["wrong_generation", "foreign_read_key"])
def test_canonical_request_is_revalidated_before_admission_mutation(tmp_path, forgery):
    dc, ht, reader = _modules()
    transport, port = _transport(ht), _Port(dc)
    request = _request(dc, reader, transport, port, tmp_path / "unused.xye", 1)
    if forgery == "wrong_generation":
        token = HydrationToken(request.read_key, request.generation + 1)
    else:
        scope = HydrationScope("foreign", "viewer-1d", "viewer-1d", port.gate.epoch)
        token = HydrationToken(HydrationReadKey(
            scope, "viewer-1d", "batch", HydrationPurpose.ONE_D), request.generation)
    object.__setattr__(request, "token", token)
    gate, epoch = request.commit_gate, request.commit_gate.epoch
    def snapshot():
        return (request.commit_gate, port.gate, gate, gate.epoch, gate.cancelled, gate._reserved_epoch,
            transport._active, transport._queued, transport._worker, transport._retired,
            transport.completions(), transport.counters(), tuple(port.calls),
            tuple(port.completions), tuple(port.notices), tuple(port.holders))
    before = snapshot()
    assert before == (gate, gate, gate, epoch, False, 0, None, None, None, False, (),
        {outcome: 0 for outcome in HydrationOutcome}, (), (), (), ())
    mutation = transport.submit_detached(request)
    assert type(mutation) is ht.DetachedHydrationMutation
    assert mutation.ticket is None and mutation.deliveries == () and mutation.replacement is None
    assert request.commit_gate is port.gate is gate
    assert snapshot() == before
    assert gate is port.gate and gate.enter(epoch)
    gate.leave()
    assert request.commit_gate is port.gate is gate
    assert snapshot() == before
    assert transport.retire(join_timeout=1)

def test_qt_free_values_port_and_path_independent_identity(tmp_path):
    dc, ht, reader = _modules()
    assert dc.ContextKind.VIEWER_1D.value == "viewer_1d"
    assert {state.value for state in dc.Viewer1DState} == {
        "empty", "loading", "ready", "cleanup_pending", "closed"}
    operations = {name for name, value in dc.OneDViewerCommitPort.__dict__.items()
                  if callable(value) and not name.startswith("_")}
    assert operations == {"commit", "cleanup_pending", "complete"}
    source = inspect.getsource(dc).lower()
    assert "pyqt" not in source and "pyside" not in source
    transport, port = _transport(ht), _Port(dc)
    one = _request(dc, reader, transport, port, tmp_path / "a.xye", 1)
    two = _request(dc, reader, transport, port, tmp_path / "b.xye", 2)
    assert one.paths != two.paths
    assert one.read_key.artifact_identity == two.read_key.artifact_identity == "viewer-1d"
    assert one.read_key.frame_identity == two.read_key.frame_identity == "batch"
    assert one.token != two.token
    assert transport.retire(join_timeout=1)

def test_one_d_never_coalesces_and_latest_cross_generation_wins(tmp_path, monkeypatch):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    transport, port = _transport(ht), _Port(dc)
    entered, release = threading.Event(), threading.Event()
    original = ht.begin_viewer_1d_read

    def gated(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(ht, "begin_viewer_1d_read", gated)
    a = _request(dc, reader, transport, port, path, 1)
    ta = transport._submit_admission(a)
    assert ta is not None and entered.wait(2)
    b = _request(dc, reader, transport, port, path, 2)
    c = _request(dc, reader, transport, port, path, 3)
    tb = transport._submit_admission(b)
    tc = transport._submit_admission(c)
    assert tb is not None and tc is not None
    assert _wait(tb).outcome is HydrationOutcome.SUPERSEDED
    release.set()
    assert _wait(ta).outcome is HydrationOutcome.HYDRATED
    assert _wait(tc).outcome is HydrationOutcome.HYDRATED
    assert [call[1] for call in port.calls if call[0] == "commit"] == [1, 3]
    assert [item.token.presentation_generation for item in port.completions] == [2, 1, 3]
    for holder in port.holders:
        assert holder.release("done")
    assert transport.retire(join_timeout=1)

def test_adoption_cell_decides_lost_commit_return(tmp_path):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    transport = _transport(ht)

    class LostReturn(_Port):
        def commit(self, prepared):
            receipt = super().commit(prepared)
            assert prepared.transfer.inspect().receipt is receipt
            raise RuntimeError("lost after adoption")

    port = LostReturn(dc)
    ticket = transport._submit_admission(_request(dc, reader, transport, port, path, 1))
    completion = _wait(ticket)
    assert completion.outcome is HydrationOutcome.HYDRATED
    assert len(port.holders) == 1 and len(port.completions) == 1
    assert port.holders[0].release("done")
    assert transport.retire(join_timeout=1)

def test_cleanup_pending_acknowledges_before_token_and_retry(tmp_path, monkeypatch):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    transport, port = _transport(ht), _Port(dc, adopt=False)
    _force_real_cleanup_pending(reader, monkeypatch)
    request = _request(dc, reader, transport, port, path, 1)
    ticket = transport._submit_admission(request)
    deadline = time.monotonic() + 3
    outer = None
    while outer is None and time.monotonic() < deadline:
        outer = transport.blocked_cleanup_token(request.token)
        time.sleep(.005)
    assert outer is not None and ticket.result() is None
    assert len(port.notices) == 1 and port.notices[0].acknowledgement is not None
    receipt = transport.retry_blocked_cleanup(outer)
    assert receipt.state is dc.Viewer2DCleanupState.CLEANED
    assert _wait(ticket).outcome is HydrationOutcome.FAILED
    assert len(port.completions) == 1
    assert transport.retire(join_timeout=1)

@pytest.mark.parametrize("ack_first", [False, True])
def test_cleanup_blocked_pre_and_post_ack_keep_only_latest(tmp_path, monkeypatch, ack_first):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    transport = _transport(ht)

    class HeldAck(_Port):
        def __init__(self, dc):
            super().__init__(dc, adopt=False)
            self.entered, self.allow_ack = threading.Event(), threading.Event()
        def cleanup_pending(self, notice):
            self.notices.append(notice)
            self.entered.set()
            assert self.allow_ack.wait(3)
            return self._ack(notice)

    port = HeldAck(dc)
    _force_real_cleanup_pending(reader, monkeypatch)
    reads, real_begin = [], ht.begin_viewer_1d_read
    monkeypatch.setattr(ht, "begin_viewer_1d_read", lambda request, *args: (
        reads.append(request.generation) or real_begin(request, *args)))
    a = _request(dc, reader, transport, port, path, 1)
    ta = transport._submit_admission(a)
    assert port.entered.wait(2) and port.notices and ta.result() is None
    assert transport.blocked_cleanup_token(a.token) is None
    with transport._lock:
        exact_a = (transport._active.disposal, transport._active.terminal,
                   transport._active.terminal.notice, transport._active.delivery_guard)
    b = _request(dc, reader, transport, port, path, 2)
    tb = transport._submit_admission(b)
    c = _request(dc, reader, transport, port, path, 3)
    if ack_first:
        port.allow_ack.set()
        deadline = time.monotonic() + 2
        while transport.blocked_cleanup_token(a.token) is None and time.monotonic() < deadline:
            time.sleep(.005)
    tc = transport._submit_admission(c)
    assert _wait(tb).outcome is HydrationOutcome.SUPERSEDED
    if not ack_first:
        port.allow_ack.set()
        deadline = time.monotonic() + 2
        while transport.blocked_cleanup_token(a.token) is None and time.monotonic() < deadline:
            time.sleep(.005)
    d = _request(dc, reader, transport, port, path, 4)
    td = transport._submit_admission(d)
    assert _wait(tc).outcome is HydrationOutcome.SUPERSEDED
    outer = transport.blocked_cleanup_token(a.token)
    assert outer is not None
    with transport._lock:
        assert exact_a[:3] == (transport._active.disposal, transport._active.terminal,
                              transport._active.terminal.notice)
        assert transport._active.delivery_guard is None
    assert transport.retry_blocked_cleanup(outer).state is dc.Viewer2DCleanupState.CLEANED
    assert _wait(ta).outcome is HydrationOutcome.FAILED
    assert _wait(td).outcome is HydrationOutcome.FAILED
    assert reads == [1, 4]
    assert [item.token.presentation_generation for item in port.completions] == [2, 3, 1, 4]
    assert transport.retire(join_timeout=1)

@pytest.mark.parametrize("control_type", [MemoryError, _FatalControl])
@pytest.mark.parametrize("seam, expected", [
    ("before_descriptor", HydrationOutcome.FAILED),
    ("after_reservation", HydrationOutcome.FAILED),
    ("receipt_retirement", HydrationOutcome.FAILED),
    ("permit", HydrationOutcome.FAILED),
    ("pass_two", HydrationOutcome.FAILED),
    ("prepared", HydrationOutcome.FAILED),
    ("before_commit", HydrationOutcome.FAILED),
    ("after_adoption", HydrationOutcome.HYDRATED),
    ("completion", HydrationOutcome.HYDRATED),
])
def test_control_identity_and_terminalization_at_every_custody_seam(
        tmp_path, monkeypatch, control_type, seam, expected):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    control = control_type(f"control at {seam}")
    seen, arrived = _capture_thread_controls(monkeypatch)
    original_begin = ht.begin_viewer_1d_read
    original_retire = ht._retire_viewer_1d_budget_receipt
    original_permit = ht.mint_viewer_1d_pass_two_permit
    original_complete = reader.Viewer1DReadOperation.complete
    original_prepared = ht._new_prepared_viewer_1d_commit
    original_commit, original_delivery = _Port.commit, _Port.complete

    if seam == "before_descriptor":
        monkeypatch.setattr(ht, "begin_viewer_1d_read",
                            lambda *args, **kwargs: (_ for _ in ()).throw(control))
    elif seam == "after_reservation":
        def stop_after_reservation(*args, **kwargs):
            operation = original_begin(*args, **kwargs)
            assert type(operation) is reader.Viewer1DReadOperation
            raise control
        monkeypatch.setattr(ht, "begin_viewer_1d_read", stop_after_reservation)
    elif seam == "receipt_retirement":
        def stop_after_retirement(*args, **kwargs):
            original_retire(*args, **kwargs)
            raise control
        monkeypatch.setattr(ht, "_retire_viewer_1d_budget_receipt",
                            stop_after_retirement)
    elif seam == "permit":
        def stop_after_permit(*args, **kwargs):
            original_permit(*args, **kwargs)
            raise control
        monkeypatch.setattr(ht, "mint_viewer_1d_pass_two_permit", stop_after_permit)
    elif seam == "pass_two":
        monkeypatch.setattr(reader.Viewer1DReadOperation, "complete",
                            lambda *args, **kwargs: (_ for _ in ()).throw(control))
    elif seam == "prepared":
        def stop_after_prepared(*args, **kwargs):
            original_prepared(*args, **kwargs)
            raise control
        monkeypatch.setattr(ht, "_new_prepared_viewer_1d_commit", stop_after_prepared)
    elif seam == "before_commit":
        monkeypatch.setattr(_Port, "commit",
                            lambda *args, **kwargs: (_ for _ in ()).throw(control))
    elif seam == "after_adoption":
        def stop_after_adoption(port, prepared):
            original_commit(port, prepared)
            raise control
        monkeypatch.setattr(_Port, "commit", stop_after_adoption)
    else:
        def stop_after_completion(port, completion):
            original_delivery(port, completion)
            raise control
        monkeypatch.setattr(_Port, "complete", stop_after_completion)

    transport, port = _transport(ht), _Port(dc)
    ticket = transport._submit_admission(
        _request(dc, reader, transport, port, path, 1))
    assert ticket is not None
    completion = _wait(ticket)
    assert completion.outcome is expected
    assert arrived.wait(3) and seen == [control]
    assert port.completions == [completion]
    assert transport.completions() == (completion,)
    assert transport.counters()[expected] == 1
    assert transport.active_token is transport.queued_token is transport.worker is None
    for holder in port.holders:
        assert holder.release("done")
    assert transport.retire(join_timeout=1)

@pytest.mark.parametrize("control_type", [MemoryError, _FatalControl])
def test_control_after_successful_positive_cleanup_is_terminal(tmp_path, monkeypatch,
                                                               control_type):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    control = control_type("after cleanup")
    seen, arrived = _capture_thread_controls(monkeypatch)
    original_release = reader.Viewer1DDisposal.release
    released = []
    def stop_after_release(disposal):
        assert original_release(disposal)
        released.append(disposal.transfer)
        raise control
    monkeypatch.setattr(reader.Viewer1DDisposal, "release", stop_after_release)
    transport, port = _transport(ht), _Port(dc, adopt=False)
    request = _request(dc, reader, transport, port, path, 1)
    ticket = transport._submit_admission(request)
    completion = _wait(ticket)
    assert completion.outcome is HydrationOutcome.FAILED
    assert arrived.wait(3) and seen == [control]
    assert len(released) == 1 and reader.viewer_1d_transfer_is_released(released[0])
    assert transport.blocked_cleanup_token(request.token) is None
    assert transport.active_token is transport.queued_token is transport.worker is None
    assert port.completions == [completion]
    assert transport.retire(join_timeout=1)

@pytest.mark.parametrize("control_type", [MemoryError, _FatalControl])
def test_control_cleanup_pending_ack_then_token_then_retry(tmp_path, monkeypatch,
                                                          control_type):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    control = control_type("cleanup drain")
    seen, arrived = _capture_thread_controls(monkeypatch)
    complete = reader.Viewer1DReadOperation.complete
    close = reader._BuildingGraph.close_streams
    failed = []
    def stop_pass_two(operation, permit):
        return reader._failure(operation.transfer,
            reader.Viewer1DReadFailureStage.PASS_TWO, "forced refusal")
    def control_drain_once(graph):
        if not failed:
            failed.append(True)
            raise control
        return close(graph)
    monkeypatch.setattr(reader.Viewer1DReadOperation, "complete", stop_pass_two)
    monkeypatch.setattr(reader._BuildingGraph, "close_streams", control_drain_once)
    transport, port = _transport(ht), _Port(dc, adopt=False)
    request = _request(dc, reader, transport, port, path, 1)
    ticket = transport._submit_admission(request)
    assert arrived.wait(3) and seen == [control]
    _wait_for(lambda: transport.blocked_cleanup_token(request.token) is not None)
    outer = transport.blocked_cleanup_token(request.token)
    assert outer is not None and ticket.result() is None
    notice = port.notices[0]
    assert notice.request is request and notice.transport_token is request.token
    assert notice.owner_identity is port.owner_identity
    assert notice.owner_request_claim is port.owner_claim
    assert notice.commit_gate_identity is port.gate
    assert notice.admitted_provider_identity is transport
    assert notice.issuer_transport_identity is transport
    assert notice.issuer_claim is transport._viewer_1d_claim
    assert notice.acknowledgement is not None
    assert transport.retry_blocked_cleanup(outer).state is dc.Viewer2DCleanupState.CLEANED
    completion = _wait(ticket)
    assert completion.outcome is HydrationOutcome.FAILED
    assert port.completions == [completion]
    assert transport.retire(join_timeout=1)

@pytest.mark.parametrize("control_type", [MemoryError, _FatalControl])
def test_retry_control_preserves_exact_blocked_custody(tmp_path, monkeypatch,
                                                       control_type):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    transport, port = _transport(ht), _Port(dc, adopt=False)
    _force_real_cleanup_pending(reader, monkeypatch)
    request = _request(dc, reader, transport, port, path, 1)
    ticket = transport._submit_admission(request)
    _wait_for(lambda: transport.blocked_cleanup_token(request.token) is not None)
    outer = transport.blocked_cleanup_token(request.token)
    with transport._lock:
        disposal, entry = transport._active.disposal, transport._active
    control = control_type("retry control")
    original_retry = reader.Viewer1DDisposal.retry
    monkeypatch.setattr(reader.Viewer1DDisposal, "retry",
                        lambda *args, **kwargs: (_ for _ in ()).throw(control))
    caught = None
    try:
        transport.retry_blocked_cleanup(outer)
    except BaseException as error:
        caught = error
    assert caught is control
    with transport._lock:
        assert transport._active is entry
        assert entry.disposal is disposal and entry.cleanup_token is outer
        assert not entry.retrying and transport._worker is None
    assert ticket.result() is None
    assert transport.blocked_cleanup_token(request.token) is outer
    monkeypatch.setattr(reader.Viewer1DDisposal, "retry", original_retry)
    assert transport.retry_blocked_cleanup(outer).state is dc.Viewer2DCleanupState.CLEANED
    assert _wait(ticket).outcome is HydrationOutcome.FAILED
    assert len(port.completions) == 1
    assert transport.retire(join_timeout=1)

def test_pre_ack_displaced_completion_control_preserves_replacement_order(
        tmp_path, monkeypatch):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    control = MemoryError("B completion control")
    transport = _transport(ht)

    class HeldAck(_Port):
        def __init__(self, dc):
            super().__init__(dc)
            self.entered, self.allow_ack = threading.Event(), threading.Event()
        def cleanup_pending(self, notice):
            self.notices.append(notice)
            self.entered.set()
            assert self.allow_ack.wait(3)
            return self._ack(notice)
        def complete(self, completion):
            super().complete(completion)
            if completion.token.presentation_generation == 2:
                raise control

    port = HeldAck(dc)
    _force_real_cleanup_pending(reader, monkeypatch)
    a = _request(dc, reader, transport, port, path, 1)
    ta = transport._submit_admission(a)
    assert port.entered.wait(2) and ta.result() is None
    with transport._lock:
        guard = transport._active.delivery_guard
        notice = transport._active.terminal.notice
    b = _request(dc, reader, transport, port, path, 2)
    tb = transport._submit_admission(b)
    c = _request(dc, reader, transport, port, path, 3)
    mutation_c = transport.submit_detached(c)
    tc = mutation_c.ticket
    caught = None
    try:
        transport.dispatch_detached(mutation_c)
    except BaseException as error:
        caught = error
    assert caught is control and _wait(tb).outcome is HydrationOutcome.SUPERSEDED
    with transport._lock:
        assert transport._active.request is a
        assert transport._active.delivery_guard is guard
        assert transport._active.terminal.notice is notice
        assert transport._queued.request is c
    port.allow_ack.set()
    _wait_for(lambda: transport.blocked_cleanup_token(a.token) is not None)
    d = _request(dc, reader, transport, port, path, 4)
    td = transport._submit_admission(d)
    assert _wait(tc).outcome is HydrationOutcome.SUPERSEDED
    outer = transport.blocked_cleanup_token(a.token)
    assert transport.retry_blocked_cleanup(outer).state is dc.Viewer2DCleanupState.CLEANED
    assert _wait(ta).outcome is HydrationOutcome.FAILED
    assert _wait(td).outcome is HydrationOutcome.HYDRATED
    assert [call[1] for call in port.calls if call[0] == "commit"] == [4]
    assert [item.token.presentation_generation for item in port.completions] == [2, 3, 1, 4]
    assert [item.outcome for item in port.completions] == [
        HydrationOutcome.SUPERSEDED, HydrationOutcome.SUPERSEDED,
        HydrationOutcome.FAILED, HydrationOutcome.HYDRATED]
    for holder in port.holders:
        assert holder.release("done")
    assert transport.retire(join_timeout=1)

def test_blocked_lineage_refusals_and_retirement_cancel_only_queued(
        tmp_path, monkeypatch):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    transport = _transport(ht)

    class HeldAck(_Port):
        def __init__(self, dc):
            super().__init__(dc)
            self.entered, self.allow_ack = threading.Event(), threading.Event()
        def cleanup_pending(self, notice):
            self.notices.append(notice)
            self.entered.set()
            assert self.allow_ack.wait(3)
            return self._ack(notice)

    port = HeldAck(dc)
    _force_real_cleanup_pending(reader, monkeypatch)
    a = _request(dc, reader, transport, port, path, 1)
    ta = transport._submit_admission(a)
    assert port.entered.wait(2) and ta.result() is None
    b = _request(dc, reader, transport, port, path, 2)
    tb = transport._submit_admission(b)
    assert tb is not None and transport.queued_token is b.token
    assert transport._submit_admission(
        _request(dc, reader, transport, port, path, 1)) is None
    foreign_port = _Port(dc)
    assert transport._submit_admission(
        _request(dc, reader, transport, foreign_port, path, 3)) is None
    base = _request(dc, reader, transport, port, path, 3)
    foreign_owner = dc.Viewer1DBatchHydrationRequest(
        base.paths, base.policy, base.generation, base.commit_gate, base.port,
        base.read_key, base.token, base.gui_thread_id, object(),
        base.owner_request_claim, base.admitted_provider_identity)
    assert transport._submit_admission(foreign_owner) is None
    for field in ("owner_request_claim", "commit_gate", "admitted_provider_identity",
                  "token", "port"):
        forged = _request(dc, reader, transport, port, path, 3)
        value = (dc.Viewer1DCommitGate() if field == "commit_gate" else
                 HydrationToken(forged.read_key, forged.generation + 1)
                 if field == "token" else object())
        object.__setattr__(forged, field, value)
        assert transport._submit_admission(forged) is None, field
    notice = port.notices[0]
    object.__setattr__(notice, "_acknowledgement", object())
    assert notice.acknowledgement is None and transport.blocked_cleanup_token(a.token) is None
    object.__setattr__(notice, "_acknowledgement", None)
    assert not transport.retire(join_timeout=.01)
    assert _wait(tb).outcome is HydrationOutcome.CANCELLED
    assert transport.active_token is a.token and transport.queued_token is None
    assert ta.result() is None and port.notices[0].acknowledgement is None
    port.allow_ack.set()
    _wait_for(lambda: transport.blocked_cleanup_token(a.token) is not None)
    outer = transport.blocked_cleanup_token(a.token)
    assert transport.retry_blocked_cleanup(outer).state is dc.Viewer2DCleanupState.CLEANED
    assert _wait(ta).outcome is HydrationOutcome.FAILED
    assert [call for call in port.calls if call[0] == "commit"] == []
    assert transport._submit_admission(
        _request(dc, reader, transport, port, path, 4)) is None
    assert transport.retire(join_timeout=1)

def test_factory_only_custody_records_refuse_direct_construction():
    from dataclasses import fields
    dc, _, reader = _modules()
    with pytest.raises(TypeError, match="foreign viewer 1-D transfer"):
        reader.Viewer1DAdoptionTransfer(object(), object(), object())
    with pytest.raises(TypeError, match="foreign viewer 1-D disposal"):
        reader.Viewer1DDisposal(object(), "foreign")
    with pytest.raises(TypeError, match="prepared viewer 1-D batch"):
        dc.Prepared1DBatchCommit(object(), object(), object(), "0" * 64, object())
    assert [field.name for field in fields(dc.Prepared1DBatchCommit)] == [
        "request", "retired_receipt_identity", "identity", "batch_identity", "transfer"]
    assert reader.Viewer1DDisposal.__slots__ == (
        "transfer", "reason", "_inner_token", "_hooks", "_no_lease_step")
    assert set(reader._BuildingCustody.__slots__) == {"graph", "claim"}
    assert set(reader._OwnerAdoption.__slots__) == {"holder", "receipt", "claim"}
    assert set(reader._DisposalCustody.__slots__) == {"graph", "disposal", "claim"}
    assert set(reader._Released.__slots__) == {"identity", "claim"}
    assert reader._BatchCustody.__slots__ == (
        "batch", "batch_identity", "prepared_identity", "retired_receipt_identity",
        "request_identity", "transport_token_identity", "issuer_transport_identity",
        "port_identity", "claim")
    _, ht, _ = _modules()
    assert ht.HydrationTransport.__slots__ == ("_viewer_1d_claim", "__dict__")
    assert [field.name for field in fields(ht._TransportEntry)] == [
        "request", "projection", "key", "closed", "token", "ticket",
        "committed_token", "committed_ticket", "state", "disposal",
        "viewer_1d_receipt", "viewer_1d_transfer", "cleanup_token", "terminal",
        "retrying", "delivery_guard"]

def test_r12_retry_retirement_control_keeps_visible_usable_token(
        tmp_path, monkeypatch):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"
    path.write_text("0 1\n1 2\n")
    control, retire_calls, drain_calls = MemoryError("retry retirement"), [], []
    real_retire = ht._retire_viewer_1d_budget_receipt
    real_close = reader._BuildingGraph.close_streams
    def retire(receipt, reason):
        retire_calls.append((receipt, reason))
        if len(retire_calls) == 1: raise RuntimeError("ordinary pre-retirement failure")
        real_retire(receipt, reason)
        if len(retire_calls) == 2: raise control
    def close(graph):
        if not drain_calls:
            drain_calls.append(True)
            raise RuntimeError("ordinary cleanup pending")
        return real_close(graph)
    monkeypatch.setattr(ht, "_retire_viewer_1d_budget_receipt", retire)
    monkeypatch.setattr(reader._BuildingGraph, "close_streams", close)
    transport, port = _transport(ht), _Port(dc, adopt=False)
    request = _request(dc, reader, transport, port, path, 1)
    ticket = transport._submit_admission(request)
    _wait_for(lambda: transport.blocked_cleanup_token(request.token) is not None)
    outer = transport.blocked_cleanup_token(request.token)
    caught = None
    try: transport.retry_blocked_cleanup(outer)
    except BaseException as error: caught = error
    assert caught is control and ticket.result() is None
    assert transport.blocked_cleanup_token(request.token) is outer
    with transport._lock:
        disposal = transport._active.disposal
        assert disposal is not None and transport._active.cleanup_token is outer
    assert transport.retry_blocked_cleanup(object()).state is dc.Viewer2DCleanupState.STALE
    real_retry = reader.Viewer1DDisposal.retry
    def retry(value):
        assert transport.retry_blocked_cleanup(outer).state is dc.Viewer2DCleanupState.STALE
        return real_retry(value)
    monkeypatch.setattr(reader.Viewer1DDisposal, "retry", retry)
    assert transport.retry_blocked_cleanup(outer).state is dc.Viewer2DCleanupState.CLEANED
    assert _wait(ticket).outcome is HydrationOutcome.FAILED
    assert disposal._hooks is disposal._inner_token is None
    assert transport.retry_blocked_cleanup(outer).state is dc.Viewer2DCleanupState.STALE
    assert len(port.completions) == 1 and transport.retire(join_timeout=1)


def test_post_capture_retry_control_delivers_finalizes_and_wakes_latest(
        tmp_path, monkeypatch):
    dc, ht, reader = _modules()
    path = tmp_path / "values.xye"; path.write_text("0 1\n1 2\n")
    control, injected, reads = MemoryError("post-capture"), [], []
    transport, port = _transport(ht), _Port(dc, adopt=False)
    _force_real_cleanup_pending(reader, monkeypatch)
    real_begin, real_capture = ht.begin_viewer_1d_read, ht.HydrationTransport._capture_locked
    monkeypatch.setattr(ht, "begin_viewer_1d_read", lambda request, *args: (
        reads.append(request.generation) or real_begin(request, *args)))
    a = _request(dc, reader, transport, port, path, 1)
    ta = transport._submit_admission(a)
    _wait_for(lambda: transport.blocked_cleanup_token(a.token) is not None)
    outer = transport.blocked_cleanup_token(a.token)
    b = _request(dc, reader, transport, port, path, 2)
    tb = transport._submit_admission(b)
    def capture_then_control(self, entry, *args, **kwargs):
        delivery = real_capture(self, entry, *args, **kwargs)
        if entry.request is a and not injected:
            injected.append(delivery)
            raise control
        return delivery
    monkeypatch.setattr(ht.HydrationTransport, "_capture_locked", capture_then_control)
    caught = None
    try: transport.retry_blocked_cleanup(outer)
    except BaseException as error: caught = error
    assert caught is control
    assert _wait(ta).outcome is HydrationOutcome.FAILED
    assert _wait(tb).outcome is HydrationOutcome.FAILED
    assert reads == [1, 2]
    assert [item.token for item in port.completions] == [a.token, b.token]
    assert tuple(item.token for item in transport.completions()).count(a.token) == 1
    assert transport.active_token is transport.queued_token is transport.worker is None
    assert transport.retry_blocked_cleanup(outer).state is dc.Viewer2DCleanupState.STALE
    assert transport.retire(join_timeout=1)
