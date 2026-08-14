from __future__ import annotations

import io
import os
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.session.hydration import (
    HydrationOutcome,
    HydrationPurpose,
    HydrationReadKey,
    HydrationScope,
    HydrationToken,
)


def _npy(value, *, version=(1, 0)) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(value), version=version,
                              allow_pickle=False)
    return stream.getvalue()


def _request(dc, reader, transport, port, paths, generation=1):
    gate = dc.Viewer1DCommitGate()
    scope = HydrationScope("viewer-1d-test", "viewer-1d", "viewer-1d", gate.epoch)
    key = HydrationReadKey(scope, "viewer-1d", "batch", HydrationPurpose.ONE_D)
    token = HydrationToken(key, generation)
    return dc.Viewer1DBatchHydrationRequest(
        tuple(str(path) for path in paths), reader.Viewer1DFormatPolicy(), generation,
        gate, port, key, token, threading.get_ident(), port.owner_identity,
        port.owner_claim, transport,
    )


class _Port:
    def __init__(self):
        self.owner_identity, self.owner_claim = object(), object()
        self.prepared = self.receipt = self.holder = None
        self.completions = []

    def commit(self, prepared):
        import xrd_tools.io.viewer_1d as reader
        self.prepared = prepared
        self.receipt = reader.adopt_prepared_viewer_1d(
            prepared, owner_identity=self.owner_identity,
            owner_request_claim=self.owner_claim, port=self,
            owner_generation=prepared.request.generation,
            commit_gate=prepared.request.commit_gate,
            owner_state=__import__("xdart.modules.display_context",
                fromlist=["Viewer1DState"]).Viewer1DState.LOADING)
        self.holder = prepared.transfer.owner_holder(self.receipt)
        return self.receipt

    def cleanup_pending(self, notice):
        from xdart.modules.display_context import acknowledge_viewer_1d_cleanup_pending
        request = notice.request
        return acknowledge_viewer_1d_cleanup_pending(notice, port=self,
            owner_identity=self.owner_identity, owner_request_claim=self.owner_claim,
            commit_gate=request.commit_gate,
            admitted_provider=request.admitted_provider_identity,
            owner_generation=request.generation, owner_state=__import__(
                "xdart.modules.display_context", fromlist=["Viewer1DState"]
            ).Viewer1DState.CLEANUP_PENDING)

    def complete(self, completion):
        self.completions.append(completion)


def _load(paths, generation=1):
    import xdart.modules.display_context as dc
    import xdart.gui.tabs.scattering.hydration_transport as ht
    import xrd_tools.io.viewer_1d as reader
    port = _Port()
    transport = ht.HydrationTransport(lambda value: None, lambda value: (None, None))
    request = _request(dc, reader, transport, port, paths, generation)
    ticket = transport._submit_admission(request)
    assert ticket is not None
    deadline = time.monotonic() + 5
    while ticket.result() is None and time.monotonic() < deadline:
        time.sleep(.005)
    assert ticket.result() is not None
    return reader, transport, port, ticket.result()


def _mode_arrays(port):
    record = port.holder.borrow
    return tuple((value.coordinate, value.intensity, value.uncertainty)
                 for value in record.modes.values())


def test_closed_formats_build_one_exact_lease_row_and_manifest(tmp_path):
    xye = tmp_path / "first.xye"
    xye.write_text("# ignored\n" + "\n".join(f"{i} {i + .5}" for i in range(7)) + "\n")
    csv = tmp_path / "second.csv"
    csv.write_text("angle,intensity,sigma\n" +
                   "\n".join(f"{i},{i + 2},{i / 10}" for i in range(11)) + "\n")
    npy = tmp_path / "third.npy"
    np.save(npy, np.column_stack((np.arange(5), np.arange(5) ** 2)))
    npz = tmp_path / "fourth.npz"
    with zipfile.ZipFile(npz, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("x.npy", _npy(np.arange(3, dtype=np.float32)))
        archive.writestr("y.npy", _npy(np.array([1, np.nan, 3], dtype=np.float64)))
        archive.writestr("sigma.npy", _npy(np.array([.1, .2, .3])))
        archive.writestr("x_label.npy", _npy(np.array("q")))
        archive.writestr("x_unit.npy", _npy(np.array("1/A")))
        archive.writestr("y_label.npy", _npy(np.array("I")))

    reader, transport, port, completion = _load((xye, csv, npy, npz))
    assert completion.outcome is HydrationOutcome.HYDRATED
    assert port.completions == [completion]
    arrays = _mode_arrays(port)
    assert [len(values[0]) for values in arrays] == [7, 11, 5, 3]
    assert arrays[0][2] is None and arrays[1][2] is not None
    assert np.isnan(arrays[3][1][1])
    assert all(array.dtype == np.dtype("float64") and not array.flags.writeable
               for values in arrays for array in values if array is not None)
    batch = port.prepared.transfer.inspect().holder.batch
    ledger = batch.manifest.ledger
    assert ledger.C == 8 * (7 * 2 + 11 * 3 + 5 * 2 + 3 * 3)
    assert ledger.T == ledger.R + max(3 * ledger.C, ledger.N)
    assert ledger.A == ledger.C + ledger.T <= ledger.B
    snapshot = batch.authority.snapshot()
    assert snapshot.committed_bytes == {"viewer_1d_transient": ledger.T}
    assert snapshot.categories == {"viewer_1d_transient": ledger.T,
                                   "light_1d": ledger.C}
    assert snapshot.available_bytes == 0 and snapshot.reservation_count == 1
    assert batch.lease.keys() == (batch.row_identity,)
    assert batch.lease.active_borrow_count == 1
    assert len(batch.manifest.sources) == 4
    assert all(len(source.scalar_record) == 64 for source in batch.manifest.sources)
    assert batch.manifest.sources[-1].x_label == "q"
    del arrays
    assert port.holder.release("test complete")
    assert batch.authority.snapshot().reservation_count == 0
    assert transport.retire(join_timeout=1)


def test_xye_compatibility_mixed_lengths_and_independent_roots(tmp_path):
    from xrd_tools.io.export import read_xye
    paths = []
    for index, count in enumerate((7, 7, 11)):
        path = tmp_path / f"compat-{index}.xye"
        path.write_text("\n".join(
            f"{point} {point + index + .25}" +
            (f" {point / 10}" if index else "") for point in range(count)) + "\n")
        paths.append(path)
    _, transport, port, completion = _load(paths)
    assert completion.outcome is HydrationOutcome.HYDRATED
    observed = _mode_arrays(port)
    for path, actual in zip(paths, observed):
        expected = read_xye(path)
        for left, right in zip(actual, expected):
            if right is None:
                assert left is None
            else:
                assert np.array_equal(left, right, equal_nan=True)
    roots = [array for mode in observed for array in mode if array is not None]
    assert all(not np.shares_memory(left, right) for index, left in enumerate(roots)
               for right in roots[index + 1:])
    del observed, roots, actual, expected, left, right
    assert port.holder.release("done") and transport.retire(join_timeout=1)


@pytest.mark.parametrize("kind", ["ragged", "object", "transpose", "ambiguous", "negative_sigma"])
def test_closed_grammar_refuses_without_adoption(tmp_path, kind):
    path = tmp_path / ("bad.npz" if kind == "ambiguous" else
                       "bad.npy" if kind in {"object", "transpose"} else "bad.csv")
    if kind == "ragged":
        path.write_text("1,2\n3,4,5\n")
    elif kind == "object":
        np.save(path, np.array([object()], dtype=object), allow_pickle=True)
    elif kind == "transpose":
        np.save(path, np.arange(8).reshape(2, 4))
    elif kind == "negative_sigma":
        path.write_text("1,2,-1\n")
    else:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("x.npy", _npy([1., 2.]))
            archive.writestr("y.npy", _npy([3., 4.]))
            archive.writestr("data.npy", _npy([[1., 3.], [2., 4.]]))
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.FAILED
    assert completion.diagnostic
    assert port.prepared is port.receipt is port.holder is None
    assert transport.active_token is transport.queued_token is None
    assert transport.retire(join_timeout=1)


def test_explicit_reload_is_fresh_and_resident_graph_is_source_independent(tmp_path):
    path = tmp_path / "fresh.xye"
    path.write_text("0 1\n1 2\n")
    _, first_transport, first, one = _load((path,), 1)
    first_y = _mode_arrays(first)[0][1].copy()
    path.write_text("0 10\n1 20\n")
    _, second_transport, second, two = _load((path,), 2)
    assert one.outcome is two.outcome is HydrationOutcome.HYDRATED
    assert np.array_equal(first_y, [1, 2])
    assert np.array_equal(_mode_arrays(first)[0][1], [1, 2])
    assert np.array_equal(_mode_arrays(second)[0][1], [10, 20])
    assert first.prepared.batch_identity != second.prepared.batch_identity
    assert first.holder.release("done") and second.holder.release("done")
    assert first_transport.retire(join_timeout=1)
    assert second_transport.retire(join_timeout=1)


def test_receipt_precedes_path_io_and_pass_one_is_array_free(tmp_path, monkeypatch):
    import xdart.modules.display_context as dc
    import xdart.gui.tabs.scattering.hydration_transport as ht
    import xrd_tools.io.viewer_1d as reader
    path = tmp_path / "ordered.xye"
    path.write_text("0 1\n1 2\n")
    events, real_begin = [], ht.begin_viewer_1d_read
    real_path, real_open, real_empty = os.path.realpath, os.open, np.empty

    def begin(request, receipt, transfer):
        assert receipt.state is dc.Viewer1DReadBudgetState.ACTIVE
        assert receipt.request is request and transfer.active_receipt_identity is receipt.identity
        monkeypatch.setattr(os.path, "realpath", lambda value: (
            events.append("path") or real_path(value)))
        monkeypatch.setattr(os, "open", lambda *args, **kwargs: (
            events.append("open") or real_open(*args, **kwargs)))
        monkeypatch.setattr(np, "empty", lambda *a, **k: pytest.fail("pass one allocated an array"))
        try:
            operation = real_begin(request, receipt, transfer)
        finally:
            monkeypatch.setattr(os.path, "realpath", real_path)
            monkeypatch.setattr(os, "open", real_open)
            monkeypatch.setattr(np, "empty", real_empty)
        assert type(operation) is reader.Viewer1DReadOperation
        assert transfer.inspect().graph.authority.snapshot().available_bytes == 0
        return operation

    monkeypatch.setattr(ht, "begin_viewer_1d_read", begin)
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.HYDRATED
    assert events[0] == "path" and "open" in events
    assert port.prepared.request.paths == (str(path),)
    assert port.prepared.request.paths[0] is not port.prepared.transfer.inspect().holder.batch.manifest.sources[0].canonical_path
    assert port.holder.release("done") and transport.retire(join_timeout=1)


def test_descriptor_headroom_refuses_before_placeholder_or_source(tmp_path, monkeypatch):
    import xdart.gui.tabs.scattering.hydration_transport as ht
    import xrd_tools.io.viewer_1d as reader
    path = tmp_path / "headroom.xye"
    path.write_text("0 1\n")
    opened = []
    monkeypatch.setattr(reader, "_descriptor_limit_and_count", lambda: (34, 1))
    monkeypatch.setattr(reader.os, "open", lambda *a, **k: opened.append(a) or pytest.fail("opened"))
    monkeypatch.setattr(ht, "viewer_1d_budget", reader.viewer_1d_budget)
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.FAILED
    assert "descriptor headroom" in completion.diagnostic and not opened
    assert port.prepared is None and transport.retire(join_timeout=1)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("shape", [(5,), (5, 2), (5, 3)])
def test_npy_versions_and_closed_shapes(tmp_path, version, shape):
    path = tmp_path / "matrix.npy"
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    path.write_bytes(_npy(values, version=version))
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.HYDRATED
    x, y, sigma = _mode_arrays(port)[0]
    assert len(x) == len(y) == shape[0]
    assert (sigma is not None) is (len(shape) == 2 and shape[1] == 3)
    del x, y, sigma
    assert port.holder.release("done") and transport.retire(join_timeout=1)


@pytest.mark.parametrize("axis", ["fortran", "trailing", "complex", "infinite_x"])
def test_npy_dtype_order_eof_and_axis_refusals(tmp_path, axis):
    path = tmp_path / "bad.npy"
    value = (np.asfortranarray(np.arange(12.).reshape(4, 3)) if axis == "fortran"
             else np.array([1 + 2j]) if axis == "complex"
             else np.array([[np.inf, 1.], [2., 3.]]) if axis == "infinite_x"
             else np.arange(6.).reshape(3, 2))
    path.write_bytes(_npy(value) + (b"tail" if axis == "trailing" else b""))
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.FAILED and port.holder is None
    assert transport.retire(join_timeout=1)


@pytest.mark.parametrize("kind", ["structured", "datetime", "string", "squeezed", "negative_sigma"])
def test_npy_closed_dtype_and_rank_matrix(tmp_path, kind):
    path = tmp_path / "closed.npy"
    values = {
        "structured": np.array([(1., 2.)], dtype=[("x", "f8"), ("y", "f8")]),
        "datetime": np.array(["2026-01-01"], dtype="datetime64[D]"),
        "string": np.array(["1", "2"]),
        "squeezed": np.arange(4.).reshape(1, 4),
        "negative_sigma": np.array([[0., 1., -1.], [1., 2., .1]]),
    }[kind]
    np.save(path, values)
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.FAILED
    assert port.holder is None and transport.retire(join_timeout=1)


@pytest.mark.parametrize("schema", ["data", "payload"])
def test_npz_data_and_single_payload_schemas(tmp_path, schema):
    path = tmp_path / "schema.npz"
    name = "data.npy" if schema == "data" else "arbitrary.npy"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(name, _npy(np.column_stack((np.arange(4), np.arange(4) + 1))))
        archive.writestr("y_label.npy", _npy(np.array("counts")))
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.HYDRATED
    source = port.prepared.transfer.inspect().holder.batch.manifest.sources[0]
    assert source.y_label == "counts" and source.role_sha256 != bytes(32)
    assert port.holder.release("done") and transport.retire(join_timeout=1)


@pytest.mark.parametrize("kind", [
    "incomplete", "extra", "traversal", "absolute", "unsupported_compression",
    "member_count", "archive_comment", "member_trailing",
])
def test_npz_closed_archive_and_schema_refusals(tmp_path, kind):
    path = tmp_path / "closed.npz"
    compression = zipfile.ZIP_BZIP2 if kind == "unsupported_compression" else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        if kind == "incomplete":
            archive.writestr("x.npy", _npy([1., 2.]))
        elif kind == "extra":
            archive.writestr("x.npy", _npy([1., 2.]))
            archive.writestr("y.npy", _npy([3., 4.]))
            archive.writestr("extra.npy", _npy([5., 6.]))
        elif kind in {"traversal", "absolute"}:
            archive.writestr("../payload.npy" if kind == "traversal" else "/payload.npy",
                             _npy([[0., 1.], [1., 2.]]))
        elif kind == "member_count":
            for index in range(17): archive.writestr(f"value-{index}.npy", _npy([index]))
        else:
            archive.writestr("data.npy", _npy([[0., 1.], [1., 2.]]) +
                             (b"tail" if kind == "member_trailing" else b""))
        if kind == "archive_comment": archive.comment = b"x" * 4097
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.FAILED
    assert port.holder is None and transport.retire(join_timeout=1)


@pytest.mark.parametrize("count", [1, 32, 256])
def test_descriptor_positive_envelope_keeps_same_open_sources(tmp_path, count):
    path = tmp_path / "same-open.xye"
    path.write_text("0 1\n")
    _, transport, port, completion = _load((path,) * count)
    assert completion.outcome is HydrationOutcome.HYDRATED
    batch = port.prepared.transfer.inspect().holder.batch
    assert len(batch.manifest.sources) == count
    assert batch.manifest.ledger.counts == (1,) * count
    assert port.holder.release("done") and transport.retire(join_timeout=1)


def test_descriptor_source_open_failure_closes_every_placeholder(tmp_path, monkeypatch):
    import xrd_tools.io.viewer_1d as reader
    path = tmp_path / "source-open.xye"
    path.write_text("0 1\n")
    real_open, placeholders = reader.os.open, []
    def fail_source(name, flags):
        if name == os.devnull:
            descriptor = real_open(name, flags)
            placeholders.append(descriptor)
            return descriptor
        raise OSError("source open refused")
    monkeypatch.setattr(reader.os, "open", fail_source)
    _, transport, port, completion = _load((path, path))
    assert completion.outcome is HydrationOutcome.FAILED and port.holder is None
    assert placeholders
    for descriptor in placeholders:
        with pytest.raises(OSError): os.fstat(descriptor)
    assert transport.retire(join_timeout=1)


@pytest.mark.parametrize("change", ["growth", "truncation", "inplace", "replacement"])
def test_source_drift_between_passes_refuses_and_releases(tmp_path, monkeypatch, change):
    import xrd_tools.io.viewer_1d as reader
    path = tmp_path / "aba.xye"
    path.write_text("0 1\n1 2\n")
    original = reader.Viewer1DReadOperation.complete
    changed = []
    def replace(operation, permit):
        if not changed:
            if change == "growth": path.write_text("0 1\n1 2\n2 3\n")
            elif change == "truncation": path.write_text("0 1\n")
            elif change == "inplace": path.write_text("0 9\n1 8\n")
            else:
                replacement = path.with_suffix(".new")
                replacement.write_text("0 9\n1 8\n")
                os.replace(replacement, path)
            changed.append(True)
        return original(operation, permit)
    monkeypatch.setattr(reader.Viewer1DReadOperation, "complete", replace)
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.FAILED
    assert "changed" in completion.diagnostic
    assert port.holder is None and transport.retire(join_timeout=1)


def test_pass_two_recertifies_npz_without_retaining_member_names(tmp_path, monkeypatch):
    import xdart.gui.tabs.scattering.hydration_transport as ht
    import xrd_tools.io.viewer_1d as reader
    path = tmp_path / "recertify.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.npy", _npy([[0., 1.], [1., 2.]]))
    inspections, operation_seen = [], []
    real_inspect, real_begin = reader._inspect_npz, ht.begin_viewer_1d_read
    def inspect_npz(*args, **kwargs):
        inspections.append(True)
        return real_inspect(*args, **kwargs)
    def begin(*args, **kwargs):
        operation = real_begin(*args, **kwargs)
        operation_seen.append(operation)
        return operation
    monkeypatch.setattr(reader, "_inspect_npz", inspect_npz)
    monkeypatch.setattr(ht, "begin_viewer_1d_read", begin)
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.HYDRATED
    assert len(inspections) == 2 and operation_seen[0].facts[0].roles is None
    source = port.holder.batch.manifest.sources[0]
    assert not hasattr(source, "roles") and not hasattr(source, "member_names")
    assert port.holder.release("done") and transport.retire(join_timeout=1)


def test_pass_two_permit_binds_full_retired_receipt_and_funding(tmp_path, monkeypatch):
    import xdart.gui.tabs.scattering.hydration_transport as ht
    import xrd_tools.io.viewer_1d as reader
    path = tmp_path / "permit.xye"
    path.write_text("0 1\n1 2\n")
    permits, real_mint = [], ht.mint_viewer_1d_pass_two_permit
    def mint(operation, receipt, issuer):
        permit = real_mint(operation, receipt, issuer)
        assert receipt.retirement_reason == "transferred"
        assert permit.operation is operation and permit.receipt_identity is receipt.identity
        assert permit.issuer is issuer and permit.capacity_bytes == operation.ledger.B
        assert permit.reserved_bytes == operation.ledger.R
        assert permit.gui_thread_id == operation.gui_thread_id
        assert permit.worker_thread_id == operation.worker_thread_id
        permits.append(permit)
        return permit
    monkeypatch.setattr(ht, "mint_viewer_1d_pass_two_permit", mint)
    _, transport, port, completion = _load((path,))
    assert completion.outcome is HydrationOutcome.HYDRATED
    assert len(permits) == 1 and permits[0].operation._consumed is True
    assert port.holder.release("done") and transport.retire(join_timeout=1)


def test_request_envelope_refuses_257_paths_before_transport_io(tmp_path):
    import xdart.modules.display_context as dc
    import xdart.gui.tabs.scattering.hydration_transport as ht
    import xrd_tools.io.viewer_1d as reader
    transport, port = ht.HydrationTransport(lambda value: None, lambda value: (None, None)), _Port()
    with pytest.raises(TypeError, match="request is malformed"):
        _request(dc, reader, transport, port, [tmp_path / f"{i}.xye" for i in range(257)])
    assert transport.retire(join_timeout=1)
