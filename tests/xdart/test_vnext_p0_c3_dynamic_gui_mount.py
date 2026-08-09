"""Finite fourteen-node C3 dynamic-GUI cutover oracle."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import inspect
import os
from pathlib import Path
from queue import Queue
import threading
from threading import RLock
from types import MethodType, SimpleNamespace

import h5py
import numpy as np
import pytest

def _image_thread():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
    return imageThread

def _adapter():
    from xdart.gui.tabs.static_scan.wranglers.scan_session import ScanSessionAdapter
    return ScanSessionAdapter

def _source(function) -> str: return inspect.getsource(function)

def _c3_source_observations(value):
    found, seen = [], set()
    def visit(item):
        identity = id(item)
        if identity in seen: return
        seen.add(identity)
        if isinstance(item, np.ndarray) or item is None: return
        if all(hasattr(item, name) for name in ("path", "size", "mtime_ns")):
            found.append(item); return
        if getattr(item, "facts", None) is not None: visit(item.facts)
        if isinstance(item, dict):
            for key, child in item.items():
                visit(key); visit(child)
            return
        if isinstance(item, (tuple, list, set, frozenset)):
            for child in item: visit(child)
            return
        if is_dataclass(item) and not isinstance(item, type):
            for descriptor in fields(item): visit(getattr(item, descriptor.name))
    visit(value)
    return tuple(found)

def _c3_observation_stamp(observation): return int(observation.size), int(observation.mtime_ns)

def _c3_real_source_case(
    tmp_path, kind, *, nframes=None, live_mode=False,
    processing_mode="Int 1D",
):
    from tests.core._vnext_p0_c2_bridge_support import DeterministicIntegrator
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    from tests.xdart._accepted_run import accepted_run, directory_source
    from tests.xdart.test_frame_read_partial_write import _write_valid_tif
    from tests.xdart.test_nxs_directory_safety import _make_thread
    from xdart.modules.frame_publication import PublicationStore
    root, watch, output = tmp_path / kind, tmp_path / kind / "watch", tmp_path / kind / "output"
    watch.mkdir(parents=True)
    output.mkdir()
    if kind == "tiff":
        source = watch / "scan_00001.tif"
        _write_valid_tif(source)
        ext, dependency = "tif", None
    elif kind == "external-eiger":
        from tests.xdart.scattering.test_e6_live_directory_wait import _write_external_hdf
        source = watch / "scan_master.h5"
        dependency = watch / "scan_data_000001.h5"
        _write_external_hdf(source, dependency)
        with h5py.File(dependency, "a") as handle:
            initial = handle["entry/data/data"][0:1]
            del handle["entry/data/data"]
            handle.create_dataset("entry/data/data", data=initial, maxshape=(None, *initial.shape[1:]), chunks=True)
        ext = "h5"
    else:
        source = watch / "scan_00001.nxs"
        _write_bluesky_nxwriter(source, n=int(nframes or 1))
        ext, dependency = "nxs", None
    worker = _make_thread(watch, output, img_ext=ext, scan_name="scan", live_mode=live_mode)
    worker.publication_store = PublicationStore(max_items=16)
    worker.run_configuration = worker._admitted_run_configuration = accepted_run(
        source_spec=directory_source(watch, ext=ext), processing_mode=processing_mode,
        output_mode="Replace", live_mode=live_mode, batch_mode=False, save_path=str(output),
        bai_1d_args={"npt": 8, "unit": "q_A^-1"}, bai_2d_args={"npt_rad": 500, "npt_azim": 500, "unit": "q_A^-1"},
    )
    worker._adopted_poni = worker.poni
    class Integrator(DeterministicIntegrator):
        def integrate2d(self, image, npt_rad, npt_azim, *, unit, **_kwargs):
            return SimpleNamespace(
                radial=np.linspace(0.0, 1.0, int(npt_rad)), azimuthal=np.linspace(-180.0, 180.0, int(npt_azim)),
                intensity=np.full((int(npt_azim), int(npt_rad)), float(np.asarray(image).sum())),
                sigma=None, unit=unit, azimuthal_unit="chi_deg")
    worker._adopted_integrator = Integrator()
    worker._adopted_fiber_integrator = None
    worker.gui_thread_id = threading.get_ident()
    worker._gui_thread_id = worker.gui_thread_id
    return SimpleNamespace(kind=kind, worker=worker, source=source, dependency=dependency, output=output,
                           stamp=(int(source.stat().st_size), int(source.stat().st_mtime_ns)))

def _c3_install_adapter_trace(monkeypatch, active):
    adapter_type = _adapter()
    for name in (
        "discover", "begin_attempt", "record_enqueued", "record_failed",
        "record_cancelled", "submit", "commit_epoch", "extend_live",
        "stop", "finish", "release_retained_custody",
    ):
        original = getattr(adapter_type, name, None)
        if not callable(original):
            continue
        def traced(owner, *args, __name=name, __original=original, **kwargs):
            trace = active.get("trace")
            if trace is not None: trace.append((f"adapter.{__name}.enter", owner, args, dict(kwargs)))
            active["adapter_call"] = __name
            try:
                result = __original(owner, *args, **kwargs)
            finally:
                active.pop("adapter_call", None)
            if trace is not None: trace.append((f"adapter.{__name}.return", owner, result))
            if (__name == "submit" and result
                    and active.get("stop_after") is not None):
                active["accepted_submits"] += 1
                if active["accepted_submits"] == active["stop_after"]:
                    active["worker"].command = "stop"
            return result
        monkeypatch.setattr(adapter_type, name, traced)

def _c3_wrap_worker_source_seams(case, trace):
    worker = case.worker
    def wrap(name, *, result_observations=False):
        original = getattr(worker, name)
        def traced(*args, **kwargs):
            captured = (*args[:2], tuple(args[2])) if name == "_dispatch_batch" and len(args) > 2 else args
            trace.append((f"{name}.enter", captured, dict(kwargs)))
            result = original(*args, **kwargs)
            trace.append((f"{name}.return", result,
                          _c3_source_observations(result) if result_observations else ()))
            return result
        setattr(worker, name, traced)
    wrap("get_next_image", result_observations=True)
    wrap("_get_next_eiger_frame_sync", result_observations=True)
    wrap("_prefetch_worker")
    wrap("_dispatch_batch")
    wrap("_record_discovered_frame")
    wrap("_commit_frame")
    original_push = worker._push_frame_to_queue
    def traced_push(item, **kwargs):
        trace.append(("prefetch.queue", item, _c3_source_observations(item)))
        return original_push(item, **kwargs)
    worker._push_frame_to_queue = traced_push
    real_mount = getattr(type(worker), "_mount_dynamic_reduction_session", None)
    def traced_mount(key, *, frozen, scan, plan, pending_frame,
                     output_path, gui_thread_id):
        trace.append(("mount.enter", key, frozen, scan, plan, pending_frame, output_path,
                      getattr(scan, "_same_run_intent", None), gui_thread_id))
        if real_mount is None: pytest.fail("real process path did not have the C3 dynamic mount")
        adapter = real_mount(worker, key, frozen=frozen, scan=scan, plan=plan, pending_frame=pending_frame,
                             output_path=output_path, gui_thread_id=gui_thread_id)
        trace.append(("mount.return", adapter))
        return adapter
    worker._mount_dynamic_reduction_session = traced_mount

@contextmanager
def _c3_worker_lifetime(worker):
    try:
        yield worker
    finally:
        worker.command = "stop"
        try:
            worker._prefetch_stop_prior()
        finally:
            worker._eiger_close_master()

def _c3_run(case, monkeypatch, *, settle=False):
    trace = []; _c3_install_adapter_trace(monkeypatch, {"trace": trace})
    _c3_wrap_worker_source_seams(case, trace)
    with _c3_worker_lifetime(case.worker):
        case.worker.process_scan(case.worker.run_configuration)
        adapters = [row[1] for row in trace if row[0] == "mount.return"]
        if settle: assert adapters and adapters[-1].quiesce(timeout=5.0); adapters[-1].resume()
    return trace, adapters

def test_empty_live_waits_without_output_owner_until_stop(tmp_path, monkeypatch):
    """An armed empty Live directory waits without acquiring output owners."""
    from tests.xdart.test_bluesky_image_wrangler import _real_dir_watch_thread
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread as mod
    watch = tmp_path / "watch"
    output = tmp_path / "output"
    watch.mkdir()
    output.mkdir()
    worker = _real_dir_watch_thread(watch, output)
    initializations, mounts = [], []
    real_initialize = worker.initialize_scan
    real_mount = worker._mount_dynamic_reduction_session
    def initialize():
        initializations.append(True)
        return real_initialize()
    def mount(*args, **kwargs):
        mounts.append((args, kwargs))
        return real_mount(*args, **kwargs)
    worker.initialize_scan = initialize
    worker._mount_dynamic_reduction_session = mount
    sleeps = []
    def stop_after_first_idle(delay):
        sleeps.append(float(delay))
        worker.command = "stop"
    monkeypatch.setattr(mod.time, "sleep", stop_after_first_idle)
    worker.process_scan(worker.run_configuration)
    assert sleeps == [0.1]
    assert initializations == mounts == []
    assert worker._scan_session_adapter is None
    assert not (output / "scan.nxs").exists()
    assert {
        "_streaming_session",
        "_streaming_sink",
        "_streaming_record_store",
        "_streaming_scan_id",
    }.isdisjoint(worker.__dict__)

def test_first_standard_frame_binds_dynamic_graph_and_light_custody_before_submit(
    tmp_path, monkeypatch,
):
    """Real TIFF/sync-Eiger/bulk-Eiger reads reach one tokenized mount."""
    active = {"trace": None}
    _c3_install_adapter_trace(monkeypatch, active)
    cases = (
        _c3_real_source_case(tmp_path, "tiff"),
        _c3_real_source_case(tmp_path, "sync-eiger", nframes=1),
        _c3_real_source_case(tmp_path, "bulk-eiger", nframes=3),
    )
    failures = []
    for case in cases:
        trace = []; active["trace"] = trace
        _c3_wrap_worker_source_seams(case, trace)
        with _c3_worker_lifetime(case.worker): case.worker.process_scan(case.worker.run_configuration); closed = case.worker._close_reduction_session()
        try:
            names = [row[0] for row in trace]
            events = lambda name: [row for row in trace if row[0] == name]
            assert "_dispatch_batch.enter" in names and "mount.enter" in names, names
            submit_returns, submit_enters = events("adapter.submit.return"), events("adapter.submit.enter")
            assert submit_returns and all(row[2] is True for row in submit_returns)
            assert len(submit_enters) == len(submit_returns)
            tokens = [row[3].get("attempt_token") for row in submit_enters]
            assert all(token is not None for token in tokens) and len({id(token) for token in tokens}) == len(tokens)
            mount = next(row for row in trace if row[0] == "mount.enter")
            assert mount[-1] == case.worker.gui_thread_id
            mounted_adapter = events("mount.return")[0][1]
            first_submit, mount_return = names.index("adapter.submit.enter"), names.index("mount.return")
            assert mount_return < first_submit
            for role in (
                "_session", "_accounting", "_sink_graph", "_observer",
                "_publication_store", "_policy", "_light_authority",
                "_light_lease", "_light_hooks", "_light_slot",
            ):
                assert getattr(mounted_adapter, role, None) is not None, role
            assert mounted_adapter._publication_store is case.worker.publication_store
            assert mounted_adapter._publication_store.allocation is mounted_adapter._policy.allocation
            assert mounted_adapter._publication_store._light_1d is mounted_adapter._light_lease
            assert mounted_adapter._light_lease.layout.active_mode in {
                mode.mode for mode in mounted_adapter._light_lease.layout.modes
            }
            begin_calls = events("adapter.begin_attempt.enter")
            assert len(begin_calls) == len(submit_enters)
            assert all(int(row[3]["source_revision"]) == max(1, case.stamp[1]) for row in begin_calls)
            positive_reads = [row for row in events("get_next_image.return") if row[1][3] is not None]
            assert positive_reads
            assert all(row[2] for row in positive_reads)
            assert all(_c3_observation_stamp(observation) == case.stamp for row in positive_reads for observation in row[2])
            if case.kind == "tiff":
                discoveries = [i for i, row in enumerate(trace) if row[0] == "_record_discovered_frame.enter"]
                commits = [i for i, row in enumerate(trace) if row[0] == "_commit_frame.enter"]
                submits = [i for i, row in enumerate(trace) if row[0] == "adapter.submit.enter"]
                accepts = [i for i, row in enumerate(trace) if row[0] == "adapter.submit.return" and row[2] is True]
                assert len(discoveries) == len(submits) == len(commits) == len(accepts) == 1
                assert discoveries[0] < submits[0] < accepts[0] < commits[0]
                assert case.worker._discovered_frame_count == 1
                assert case.source in tuple(Path(path) for path in case.worker.processed)
            elif case.kind == "sync-eiger":
                sync = [row for row in events("_get_next_eiger_frame_sync.return") if row[1][3] is not None]
                assert sync and all(row[2] for row in sync)
            else:
                assert "_prefetch_worker.enter" in names
                queued = [row for row in events("prefetch.queue") if row[1] is not None and row[1][3] is not None]
                assert len(queued) == 3 and all(row[2] for row in queued)
            assert closed is True and case.worker._scan_session_adapter is None and case.worker._retained_scan_session_adapter is mounted_adapter
        except AssertionError as exc:
            failures.append((case.kind, str(exc), [row[0] for row in trace]))
        finally:
            if getattr(case.worker, "_scan_session_adapter", None) is not None:
                case.worker._close_reduction_session()
    assert not failures, "\n" + "\n".join(f"{label}: {detail}; trace={names}" for label, detail, names in failures)

def test_first_auto_gi_frame_freezes_once_before_session_and_submits_same_frame(
    tmp_path, monkeypatch,
):
    from tests.xdart._accepted_run import accepted_run, directory_source, gi_intent
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread as mod
    from xrd_tools.reduction import core as reduction_core
    import xrd_tools.core as core_mod
    import xrd_tools.session as session_mod
    case = _c3_real_source_case(tmp_path, "tiff")
    frozen = accepted_run(
        source_spec=directory_source(case.source.parent, ext="tif"),
        processing_mode="Int 1D", output_mode="Replace", live_mode=False,
        batch_mode=False, save_path=str(case.output),
        bai_1d_args={"unit": "q_A^-1"},
        gi=gi_intent(enabled=True, incidence_motor="Manual", th_val=0.1),
    )
    case.worker.run_configuration = case.worker._admitted_run_configuration = frozen
    trace = []
    _c3_install_adapter_trace(monkeypatch, {"trace": trace})
    _c3_wrap_worker_source_seams(case, trace)
    freezes, resources = [], {"total": [], "rows": [], "acquire": []}
    original_freeze, original_rows = mod.freeze_live_scan_gi_ranges, core_mod.browse_publication_max_items
    original_acquire = session_mod.acquire_light_1d_retention
    total_ram = 8 * 1024 ** 3
    monkeypatch.setattr(core_mod, "total_physical_ram_bytes", lambda: resources["total"].append(True) or total_ram)
    def rows(npt, *, total_ram_bytes):
        resources["rows"].append((int(npt), int(total_ram_bytes)))
        return original_rows(npt, total_ram_bytes=total_ram_bytes)
    def acquire(authority, **kwargs):
        resources["acquire"].append((authority, dict(kwargs)))
        return original_acquire(authority, **kwargs)
    monkeypatch.setattr(core_mod, "browse_publication_max_items", rows)
    monkeypatch.setattr(session_mod, "acquire_light_1d_retention", acquire)
    class DeterministicFiber:
        detector = None
        def integrate1d(self, data, npt, *, unit="q_A^-1", **_kwargs):
            return SimpleNamespace(radial=np.linspace(0.1, 1.0, int(npt)),
                intensity=np.full(int(npt), float(np.asarray(data).sum())), sigma=None, unit=unit)
    monkeypatch.setattr(reduction_core, "poni_to_fiber_integrator", lambda *_args, **_kwargs: DeterministicFiber())
    def traced_freeze(scan, frames, **kwargs):
        frame_values = tuple(frames)
        freezes.append((scan, frame_values, dict(kwargs)))
        return original_freeze(scan, frame_values, **kwargs)
    monkeypatch.setattr(mod, "freeze_live_scan_gi_ranges", traced_freeze)
    with _c3_worker_lifetime(case.worker): case.worker.process_scan(frozen)
    assert len(freezes) == 1
    scout = freezes[0][1]
    assert len(scout) == 1 and freezes[0][2]["gi_freeze_mode"] == "first_frame"
    mount = next(row for row in trace if row[0] == "mount.enter")
    submitted = next(row for row in trace if row[0] == "adapter.submit.enter")
    assert mount[5] is scout[0] and submitted[2][0] is scout[0]
    adapter = next(row[1] for row in trace if row[0] == "mount.return")
    layout = adapter._light_lease.layout
    assert len(layout.modes) == 1
    assert layout.active_mode == frozen.gi.mode_1d
    mode = layout.modes[0]
    assert mode.mode == frozen.gi.mode_1d
    for buffer in (mode.coordinate, mode.intensity):
        assert np.dtype(buffer.dtype) == np.dtype(np.float64)
        assert buffer.itemsize == 8 and buffer.shared is False
    assert adapter._publication_store._light_1d is adapter._light_lease
    assert resources["total"] == [True]
    assert resources["rows"] == [(mode.coordinate.length, total_ram)]
    authority, request = resources["acquire"][0]
    assert authority is adapter._light_authority
    assert request["layout"] is adapter._light_lease.layout
    assert request["requested_rows"] == adapter._light_lease.row_cap
    assert request["compatibility_byte_ceiling"] == min(1024 ** 3,
                                                         int(0.05 * total_ram))
    assert request["funding_mode"].value == "replace-publication-a1"
    assert request["current_lineage_rows"] is None
    assert request["gui_thread_id"] == case.worker.gui_thread_id
    assert adapter._publication_store.allocation is adapter._policy.allocation
    assert adapter._accounting._light_lease is adapter._light_lease
    assert adapter._accounting._light_cleanup_hooks is adapter._light_hooks
    assert adapter._accounting._light_custody_slot is adapter._light_slot
    case.worker._close_reduction_session()

def test_first_frame_bind_failure_unwinds_without_source_advance(
    tmp_path, monkeypatch,
):
    """Each real source retains one popped value through bind/submit refusal."""
    from xdart.modules.frame_publication import PublicationStore; aggregate = []
    for phase in ("bind", "submit"):
        for kind, nframes in (("tiff", None), ("sync-eiger", 1), ("bulk-eiger", 3)):
            with monkeypatch.context() as scoped:
                case = _c3_real_source_case(tmp_path / phase, kind, nframes=nframes); trace = []
                _c3_install_adapter_trace(scoped, {"trace": trace})
                _c3_wrap_worker_source_seams(case, trace)
                failed_leases = []
                if phase == "bind":
                    original_bind = PublicationStore.bind_light_1d
                    def fail_store_bind(owner, lease):
                        if owner is case.worker.publication_store:
                            if not failed_leases:
                                failed_leases.append(lease)
                                raise RuntimeError("C3 injected store-bind fault")
                            assert all(prior.state.value == "released" for prior in failed_leases)
                        return original_bind(owner, lease)
                    scoped.setattr(PublicationStore, "bind_light_1d", fail_store_bind)
                else:
                    adapter_type = _adapter()
                    traced_submit = adapter_type.submit
                    refusals = [True]
                    def refuse_once(owner, frame, *args, **kwargs):
                        if refusals:
                            refusals.pop()
                            trace.append(("adapter.submit.injected-refusal", owner, frame, dict(kwargs)))
                            return False
                        return traced_submit(owner, frame, *args, **kwargs)
                    scoped.setattr(adapter_type, "submit", refuse_once)
                with _c3_worker_lifetime(case.worker): case.worker.process_scan(case.worker.run_configuration)
                try:
                    dispatches = [row for row in trace if row[0] == "_dispatch_batch.enter"]
                    assert len(dispatches) >= 2, len(dispatches)
                    first_pending, retry_pending = (dispatches[index][1][2] for index in (0, 1))
                    assert first_pending and retry_pending and retry_pending[0] is first_pending[0]
                    reads = [row for row in trace if row[0] == "get_next_image.return" and row[1][3] is not None]
                    assert len(reads) == (3 if kind == "bulk-eiger" else 1)
                    if phase == "bind":
                        assert len(failed_leases) == sum(row[0] == "mount.return" for row in trace) == 1
                        assert case.worker._scan_session_adapter is not None
                        assert case.worker._retained_scan_session_adapter is None
                        assert all(lease.state.value == "released" for lease in failed_leases)
                    else:
                        refusals_seen = [row for row in trace if row[0] == "adapter.submit.injected-refusal"]
                        accepts = [row for row in trace if row[0] == "adapter.submit.return" and row[2] is True]
                        settled = [row for row in trace if row[0] == "adapter.record_failed.enter"]
                        assert len(refusals_seen) == len(settled) == 1
                        assert accepts
                        settled_at, accepted_at = trace.index(settled[0]), trace.index(accepts[0])
                        assert settled_at < accepted_at
                        refused_token = refusals_seen[0][3].get("attempt_token")
                        assert refused_token is not None and settled[0][2][0] is refused_token
                        if kind == "tiff":
                            commits = [i for i, row in enumerate(trace) if row[0] == "_commit_frame.enter"]
                            assert commits == [accepted_at + 2] and trace[accepted_at + 1][0] == "_dispatch_batch.return" and str(case.source) in case.worker.processed
                except AssertionError as exc:
                    aggregate.append((phase, kind, str(exc), tuple(row[0] for row in trace)))
                finally:
                    if getattr(case.worker, "_scan_session_adapter", None) is not None:
                        case.worker._close_reduction_session()
    with monkeypatch.context() as scoped:
        terminal = _c3_real_source_case(tmp_path / "terminal-submit", "tiff"); trace = []
        _c3_install_adapter_trace(scoped, {"trace": trace})
        _c3_wrap_worker_source_seams(terminal, trace)
        adapter_type = _adapter()
        def terminal_refusal(owner, frame, *args, **kwargs):
            trace.append(("adapter.submit.terminal-refusal", owner, frame, dict(kwargs)))
            terminal.worker.command = "stop"
            return False
        scoped.setattr(adapter_type, "submit", terminal_refusal)
        with _c3_worker_lifetime(terminal.worker):
            terminal.worker.process_scan(terminal.worker.run_configuration)
        assert terminal.worker._discovered_frame_count == 1
        assert terminal.worker.files_processed == 0
        assert not any(row[0] == "_commit_frame.enter" for row in trace)
        assert str(terminal.source) not in terminal.worker.processed
        assert terminal.worker.img_fnames and terminal.worker.img_fnames[0] == str(terminal.source)
        assert terminal.worker._close_reduction_session() is True
    import xrd_tools.session as session_mod
    prelease = _c3_real_source_case(tmp_path / "prelease", "tiff")
    primary = ValueError("C3 injected pre-lease refusal")
    with monkeypatch.context() as scoped:
        scoped.setattr(session_mod, "acquire_light_1d_retention",
                       lambda *_a, **_k: (_ for _ in ()).throw(primary))
        with _c3_worker_lifetime(prelease.worker), pytest.raises(ValueError) as caught:
            prelease.worker.process_scan(prelease.worker.run_configuration)
    assert caught.value is primary and caught.value.__cause__ is None
    assert prelease.worker._scan_session_adapter is None and prelease.worker.publication_store.allocation is None
    assert prelease.worker.release_retained_custody() is True
    partial = _c3_real_source_case(tmp_path / "partial", "tiff")
    hook_primary, hook_cleanup = (ValueError("C3 injected post-session bind refusal"),
                                  RuntimeError("C3 injected partial cleanup refusal"))
    original_hooks, hook_failures = PublicationStore.light_1d_cleanup_hooks, [hook_primary, hook_cleanup]
    def fail_hooks(owner, lease, **kwargs):
        if owner is partial.worker.publication_store and hook_failures:
            raise hook_failures.pop(0)
        return original_hooks(owner, lease, **kwargs)
    monkeypatch.setattr(PublicationStore, "light_1d_cleanup_hooks", fail_hooks)
    with _c3_worker_lifetime(partial.worker), pytest.raises(ValueError) as caught:
        partial.worker.process_scan(partial.worker.run_configuration)
    retained = partial.worker._scan_session_adapter
    assert caught.value is hook_primary and caught.value.__cause__ is hook_cleanup
    assert retained is not None and partial.worker.release_retained_custody() is False
    assert partial.worker._close_reduction_session() is True
    assert partial.worker._scan_session_adapter is None and partial.worker.publication_store.allocation is None
    # Unregister precedes accounting; later settlement failure preserves primary.
    from xdart.gui.tabs.static_scan.wranglers import scan_session as adapter_mod
    scoped_live = SimpleNamespace(idx=1); monkeypatch.setattr(adapter_mod, "frame_from_live_frame", lambda live: live)
    for failure_site in ("snapshot", "settlement"):
        pinned, submit_order, token = [], [], object()
        primary, cleanup = OSError("C3 submit fault"), RuntimeError(f"C3 {failure_site} fault")
        observer = SimpleNamespace(register=lambda live, attempt: pinned.append((live, attempt)) or True,
            unregister=lambda live, attempt: (submit_order.append(("unregister", live, attempt)),
                                                pinned.remove((live, attempt)), True)[-1])
        session = SimpleNamespace(submit=lambda *_a, **_k: (submit_order.append(("submit", tuple(pinned))),
                                                             (_ for _ in ()).throw(primary))[-1])
        class _Accounting:
            def snapshot(self):
                submit_order.append(("snapshot", tuple(pinned)))
                if failure_site == "snapshot": raise cleanup
                return SimpleNamespace(attempt_states={token: SimpleNamespace(value="provisional")})
            def record_failed(self, *_a, **_k):
                submit_order.append(("settlement", tuple(pinned))); raise cleanup
        direct = _adapter()(session=session, accounting=_Accounting(), sink_graph=None, observer=observer,
            publication_store=None, policy=None, light_authority=None, light_lease=None,
            light_hooks=None, light_slot=None, generation=1)
        with pytest.raises(OSError) as caught:
            direct.submit(scoped_live, attempt_token=token)
        assert caught.value is primary and caught.value.__cause__ is cleanup and pinned == []
        assert submit_order[:3] == [("submit", ((scoped_live, token),)),
                                    ("unregister", scoped_live, token), ("snapshot", ())]
        assert [row[0] for row in submit_order[3:]] == ([] if failure_site == "snapshot" else ["settlement"])
    # Partial mounts delegate only to the sink's public abort capability.
    for failure in (None, RuntimeError("integrity_hold")):
        calls, releases = [], []
        class _Sink:
            def abort(self, result):
                calls.append(result)
                if failure is not None: raise failure
        partial_adapter = _adapter()(session=None, accounting=None, sink_graph=_Sink(),
            observer=None, publication_store=None, policy=None, light_authority=None,
            light_lease=None, light_hooks=None, light_slot=None, generation=1)
        partial_adapter._release_partial_light = lambda: releases.append(True)
        if failure is not None:
            with pytest.raises(RuntimeError, match="integrity_hold"): partial_adapter.finish()
            assert releases == []
        else:
            partial_adapter.finish(); assert releases == [True]
        assert calls == [None]
    from xrd_tools.sources.discover import Candidate
    topology = tmp_path / "topology"; topology.mkdir()
    attempt_effects, mount_effects = [], []
    real_begin = _adapter().begin_attempt
    def topology_begin(owner, *args, **kwargs):
        attempt_effects.append((owner, args, kwargs))
        return real_begin(owner, *args, **kwargs)
    monkeypatch.setattr(_adapter(), "begin_attempt", topology_begin)
    worker, real_mount = partial.worker, partial.worker._mount_dynamic_reduction_session
    def topology_mount(*args, **kwargs):
        mount_effects.append((args, kwargs))
        return real_mount(*args, **kwargs)
    worker._mount_dynamic_reduction_session = topology_mount
    for kind in ("leaf-soft", "ancestor-soft", "ancestor-external", "vds", "external-storage"):
        master = topology / f"{kind}.h5"
        dependency = topology / f"{kind}-dependency.h5"
        with h5py.File(master, "w") as handle:
            if kind == "leaf-soft":
                handle.create_dataset("real", data=np.ones((1, 2, 2)))
                handle.require_group("entry/data")
                handle["entry/data/data"] = h5py.SoftLink("/real")
            elif kind == "ancestor-soft":
                handle.create_dataset("real/data", data=np.ones((1, 2, 2)))
                handle.require_group("entry")
                handle["entry/data"] = h5py.SoftLink("/real")
            elif kind == "ancestor-external":
                with h5py.File(dependency, "w") as dep:
                    dep.create_dataset("entry/data/data", data=np.ones((1, 2, 2)))
                handle["entry"] = h5py.ExternalLink(dependency.name, "/entry")
            elif kind == "vds":
                handle.create_dataset("raw", data=np.ones((1, 2, 2)))
                layout = h5py.VirtualLayout((1, 2, 2), dtype=np.float64)
                layout[:] = h5py.VirtualSource(master.name, "/raw", shape=(1, 2, 2))
                handle.create_virtual_dataset("entry/data/data", layout)
            else:
                handle.create_dataset("entry/data/data", shape=(1, 2, 2), dtype=np.float64,
                                      external=[(f"{kind}.raw", 0, h5py.h5f.UNLIMITED)])
        stat = master.stat()
        worker._eiger_master_path = str(master)
        worker._eiger_master_candidate = Candidate(master, "nexus_hdf5", int(stat.st_size), int(stat.st_mtime_ns))
        worker._eiger_descriptor = SimpleNamespace(segment_paths=("/entry/data/data",), dataset_path=None)
        with h5py.File(master, "r") as handle:
            worker._eiger_cursor = SimpleNamespace(_h5=handle)
            output_before = tuple(partial.output.iterdir())
            with pytest.raises(ValueError, match="unsupported Eiger"):
                worker._eiger_source_facts(0)
            assert tuple(partial.output.iterdir()) == output_before
        assert worker._scan_session_adapter is None
        assert attempt_effects == mount_effects == []
    assert not aggregate, aggregate

def test_image_controller_uses_one_worker_owned_dynamic_nexus_session(
    tmp_path, monkeypatch,
):
    import xdart.modules.reduction as reduction_mod
    case = _c3_real_source_case(tmp_path, "tiff"); opened = []; real_open = reduction_mod.open_live_scan_session
    def open_session(*args, **kwargs):
        opened.append(dict(kwargs))
        assert type(kwargs["sink"]).__name__ != "QtFrameObserver"
        return real_open(*args, **kwargs)
    monkeypatch.setattr(reduction_mod, "open_live_scan_session", open_session)
    trace, adapters = _c3_run(case, monkeypatch)
    try:
        assert len(adapters) == 1
        adapter = adapters[0]
        submits = [row[1] for row in trace if row[0] == "adapter.submit.enter"]
        assert submits and all(owner is adapter for owner in submits)
        assert adapter._sink_graph is not adapter._observer
        assert len(opened) == 1
        assert opened[0]["sink"] is adapter._sink_graph
        assert opened[0]["sink"] is not adapter._observer
        assert opened[0]["record_store"] is None
        assert opened[0]["record_store_persisted_on_write"] is False
        source = _source(_image_thread())
        assert "_get_reduction_session(" not in source
        assert "_streaming_session" not in source
        assert "QtNexusSink" not in source
    finally:
        case.worker._close_reduction_session()

def test_nexus_controller_uses_the_same_dynamic_mount_without_legacy_output_owner(
    tmp_path, monkeypatch,
):
    from tests.core._vnext_p0_c2_bridge_support import DeterministicIntegrator
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    from tests.xdart._accepted_run import accepted_run, admitted_worker, gi_intent
    from xdart.gui.tabs.static_scan.wranglers import nexus_wrangler as wrapper_mod
    from xdart.gui.tabs.static_scan.wranglers import nexus_wrangler_thread as mod
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler import nexusWrangler
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import nexusThread
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread, wranglerWidget
    from xdart.modules.frame_publication import PublicationStore
    from xdart.modules.live import LiveScan
    from xdart.modules import reduction as reduction_mod
    import xdart.gui.gui_utils  # noqa: F401
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.io.nexus import NexusImageStack
    from xrd_tools.reduction import core as reduction_core
    from xrd_tools.session.run_configuration import RunConfigurationRefused
    source = tmp_path / "fixed.nxs"
    member = tmp_path / "fixed-data.h5"
    def write_source(nframes):
        _write_bluesky_nxwriter(source, n=nframes)
        with h5py.File(source, "r") as handle:
            images = np.asarray(handle["entry/data/eiger_image"])
        with h5py.File(member, "w") as handle:
            handle.create_dataset("entry/data/data", data=images)
        with h5py.File(source, "a") as handle:
            del handle["entry/instrument/detectors/eiger/data"]
            del handle["entry/data/eiger_image"]
            handle["entry/data/data_000001"] = h5py.ExternalLink(
                member.name, "/entry/data/data")
        return source
    write_source(9)
    output = tmp_path / "out"
    output.mkdir()
    target = output / "fixed.nexus"
    poni = PONI(dist=.2, poni1=.1, poni2=.1, wavelength=1e-10)
    active = dict(
        trace=None, reads=None, opened=None, mounts=None, freezes=None,
        worker=None, stop_after=None, accepted_submits=0,
        source_close_failures=0, candidate_calls=None,
        revise_final_restat=False,
    )
    source_candidate = mod._source_candidate
    def output_snapshot():
        return tuple((str(path.relative_to(output)), path.read_bytes())
                     for path in sorted(output.rglob("*")) if path.is_file())
    @contextmanager
    def source_graph_fixture(label, topology=None):
        topology = topology or label
        root = tmp_path / "source-graphs" / label
        root.mkdir(parents=True)
        master_path, member_path = root / "master.nxs", root / "member.h5"
        data = np.ones((1, 2, 2), dtype=np.uint16)
        selectors = (("/entry/data/hard", "/entry/data/external")
                     if topology == "mixed" else ("/entry/data/data",))
        if topology in {"external", "indirect", "mixed", "missing"}:
            with h5py.File(member_path, "w") as member_handle:
                member_handle.create_dataset("/entry/data/data", data=data)
        with h5py.File(master_path, "w") as master:
            if topology == "indirect":
                master["entry"] = h5py.ExternalLink(member_path.name, "/entry")
            else:
                master.require_group("/entry/data")
                if topology == "hard":
                    master.create_dataset(selectors[0], data=data)
                elif topology in {"external", "missing"}:
                    master[selectors[0]] = h5py.ExternalLink(
                        member_path.name, "/entry/data/data")
                elif topology == "soft":
                    master.create_dataset("/real", data=data)
                    master[selectors[0]] = h5py.SoftLink("/real")
                elif topology == "vds":
                    master.create_dataset("/raw", data=data)
                    layout = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
                    layout[:] = h5py.VirtualSource(
                        str(master_path), "/raw", shape=data.shape)
                    master.create_virtual_dataset(selectors[0], layout)
                elif topology == "external-storage":
                    master.create_dataset(
                        selectors[0], shape=data.shape, dtype=data.dtype,
                        external=[("payload.raw", 0, h5py.h5f.UNLIMITED)])
                elif topology == "mixed":
                    master.create_dataset(selectors[0], data=data)
                    master[selectors[1]] = h5py.ExternalLink(
                        member_path.name, "/entry/data/data")
                elif topology == "invalid-rank":
                    master.create_dataset(selectors[0], data=np.ones(2))
        handle = h5py.File(master_path, "r")
        datasets = []
        try:
            if topology == "invalid-rank":
                datasets = [handle[path] for path in selectors]
                stack = SimpleNamespace(
                    _h5=handle, _paths=selectors, _dsets=datasets,
                    _offsets=[0, 2], shape=(2, 2, 2))
            else:
                stack = NexusImageStack(handle, list(selectors))
                datasets = list(stack._dsets)
            yield SimpleNamespace(
                stack=stack, master=source_candidate(master_path)
            ), master_path, member_path
        finally:
            for dataset in datasets:
                if dataset.id.valid:
                    dataset.id.close()
            handle.close()

    topology_generation = 37
    for topology in ("hard", "external"):
        with source_graph_fixture(topology) as (prepared, master_path, member_path):
            graph, observations = nexusThread._source_graph(
                prepared, topology_generation)
            assert (graph.dataset_paths, graph.extent, graph.generation) == (
                tuple(prepared.stack._paths), prepared.stack.shape[0],
                topology_generation)
            if topology == "hard":
                assert graph.external_members == ()
                assert observations == (prepared.master,)
            else:
                assert observations[0] == prepared.master and tuple(item.path for item in observations) == (
                    master_path, member_path)
                assert len(graph.external_members) == 1
                external = graph.external_members[0]
                assert (Path(external.path), external.dataset_path,
                        external.source_start, external.source_stop,
                        external.ordinal) == (
                    member_path, "/entry/data/data", 0, 1, 0)
            nexusThread._require_source_unchanged(
                observations, topology_generation)
    negatives = {
        "soft": "indirect detector leaf",
        "indirect": "indirect ancestor",
        "vds": "non-owned or invalid-rank storage",
        "external-storage": "non-owned or invalid-rank storage",
        "mixed": "mixed storage owners",
        "invalid-rank": "non-owned or invalid-rank storage",
        "missing": "member observation failed",
    }
    for topology, reason in negatives.items():
        before_output, before_mounts = output_snapshot(), active["mounts"]
        with source_graph_fixture(topology) as (prepared, _master, _member):
            with monkeypatch.context() as scoped:
                if topology == "missing":
                    def missing_member_candidate(path):
                        if Path(path) == _member:
                            raise OSError("injected missing member")
                        return source_candidate(path)
                    scoped.setattr(mod, "_source_candidate", missing_member_candidate)
                with pytest.raises(mod.AppendSourceGraphRefused) as caught:
                    nexusThread._source_graph(prepared, topology_generation)
        assert reason in str(caught.value)
        assert caught.value.decision.source_generation == topology_generation
        assert output_snapshot() == before_output and active["mounts"] is before_mounts
    for label, ordinal, action, reason in (
        ("changed-master", 0, "touch", "source changed before mount"),
        ("changed-member", 1, "touch", "source changed before mount"),
        ("disappeared-master", 0, "unlink", "source member disappeared"),
        ("disappeared-member", 1, "unlink", "source member disappeared"),
    ):
        with source_graph_fixture(label, "external") as (prepared, _master, _member):
            _graph, observations = nexusThread._source_graph(
                prepared, topology_generation)
        changed = Path(observations[ordinal].path)
        if action == "touch":
            with changed.open("ab") as stream:
                stream.write(b"\0")
        else:
            changed.unlink()
        before_output, before_mounts = output_snapshot(), active["mounts"]
        with pytest.raises(mod.AppendSourceGraphRefused) as caught:
            nexusThread._require_source_unchanged(
                observations, topology_generation)
        assert reason in str(caught.value)
        assert output_snapshot() == before_output and active["mounts"] is before_mounts
    class Integrator(DeterministicIntegrator):
        def integrate2d(self, image, npt_rad, npt_azim, *, unit, **_kwargs):
            intensity = np.full((npt_azim, npt_rad), np.asarray(image).sum())
            return SimpleNamespace(radial=np.arange(npt_rad),
                azimuthal=np.arange(npt_azim), intensity=intensity, sigma=None,
                unit=unit, azimuthal_unit="chi_deg")
    def gi_1d(image, _fiber, *, npt, unit="q_A^-1", **_kwargs):
        axis = np.linspace(.1, 1., int(npt))
        return SimpleNamespace(radial=axis,
            intensity=np.full_like(axis, np.asarray(image).sum()), sigma=None,
            unit=unit)
    def gi_2d(image, _fiber, *, npt_rad, npt_azim,
              unit="qip_A^-1", **_kwargs):
        radial = np.linspace(.1, 1., int(npt_rad))
        azimuthal = np.linspace(-1., 1., int(npt_azim))
        intensity = np.full((len(radial), len(azimuthal)), np.asarray(image).sum())
        return SimpleNamespace(radial=radial, azimuthal=azimuthal,
            intensity=intensity, sigma=None, unit=unit,
            azimuthal_unit="qoop_A^-1")
    monkeypatch.setattr(mod, "poni_to_integrator", lambda _poni: Integrator())
    monkeypatch.setattr(reduction_core, "poni_to_fiber_integrator",
                        lambda *_a, **_k: object())
    monkeypatch.setattr(reduction_core, "integrate_gi_polar_1d", gi_1d)
    monkeypatch.setattr(reduction_core, "integrate_gi_2d", gi_2d)
    _c3_install_adapter_trace(monkeypatch, active)
    real_get, real_open = NexusImageStack.__getitem__, reduction_mod.open_live_scan_session
    def traced_get(stack, key):
        active["reads"].append(key)
        return real_get(stack, key)
    def traced_open(*args, **kwargs):
        active["opened"].append(kwargs["sink"])
        return real_open(*args, **kwargs)
    def traced_candidate(path):
        candidate = source_candidate(path)
        if active["revise_final_restat"] and len(active["candidate_calls"]) == 5:
            candidate = type(candidate)(candidate.path, candidate.adapter_id,
                                        candidate.size + 1, candidate.mtime_ns)
        active["candidate_calls"].append(candidate)
        return candidate
    real_freeze = reduction_mod.freeze_live_scan_gi_ranges
    def traced_freeze(scan, frames, **kwargs):
        values = tuple(frames)
        assert active["worker"]._scan_session_adapter is None
        active["freezes"].append(values)
        return real_freeze(scan, values, **kwargs)
    monkeypatch.setattr(NexusImageStack, "__getitem__", traced_get)
    monkeypatch.setattr(reduction_mod, "open_live_scan_session", traced_open)
    monkeypatch.setattr(mod, "_source_candidate", traced_candidate)
    monkeypatch.setattr(mod, "freeze_live_scan_gi_ranges", traced_freeze,
                        raising=False)
    real_execution_close = mod.PreparedNexusExecution.close
    def traced_execution_close(envelope):
        active["trace"].append(("execution.close.enter", envelope))
        if active["source_close_failures"]:
            active["source_close_failures"] -= 1
            raise OSError("injected retained source close")
        result = real_execution_close(envelope)
        active["trace"].append(("execution.close.return", envelope, result))
        return result
    monkeypatch.setattr(
        mod.PreparedNexusExecution, "close", traced_execution_close)
    def run(
        mode="Int 1D + 2D", *, cleanup_pending=False, stop_after=None,
        source_close_failures=0, revise_final_restat=False,
    ):
        active.update(
            trace=[], reads=[], opened=[], mounts=[], freezes=[],
            stop_after=stop_after, accepted_submits=0,
            source_close_failures=source_close_failures, candidate_calls=[],
            revise_final_restat=bool(revise_final_restat),
        )
        display = LiveScan("display", static=True)
        worker = nexusThread(Queue(), RLock(), str(target), str(source), poni,
            "q_total", "qip_qoop", "start", display)
        active["worker"] = worker
        frozen = accepted_run(
            source_spec=SourceSpec(str(source), SourceKind.NEXUS_STACK, entry="entry"),
            processing_mode=mode, output_mode="Append", save_path=str(output), run_options={"xye_only": mode == "Int 1D (XYE)"},
            max_cores=1, poni_values=poni.to_dict(),
            bai_1d_args={"npt": 8, "unit": "q_A^-1"},
            bai_2d_args={"npt_rad": 4, "npt_azim": 4, "unit": "q_A^-1"},
            gi=gi_intent(enabled=True, incidence_motor="Manual", th_val=.1),
        )
        admitted_worker(worker, frozen=frozen)
        worker.publication_store = PublicationStore(max_items=16)
        worker.gui_thread_id = worker._gui_thread_id = threading.get_ident()
        statuses = []
        worker.showLabel.connect(statuses.append)
        real_mount = worker._mount_dynamic_reduction_session
        def traced_mount(*args, **kwargs):
            adapter = real_mount(*args, **kwargs)
            intent = worker._active_scan._same_run_intent
            active["mounts"].append((adapter, intent, kwargs.get("policy")))
            return adapter
        worker._mount_dynamic_reduction_session = traced_mount
        if cleanup_pending:
            worker._test_real_close = worker._close_reduction_session
            worker._close_reduction_session = lambda: False
        worker.run()
        return worker, statuses

    expected_candidate_paths = [os.path.abspath(source), os.path.abspath(member)] * 3
    fresh, fresh_statuses = run()
    fresh_candidates = tuple(active["candidate_calls"])
    assert [os.path.abspath(item.path) for item in fresh_candidates] == expected_candidate_paths
    assert fresh_candidates[0] == fresh_candidates[2] == fresh_candidates[4] and fresh_candidates[1] == fresh_candidates[3] == fresh_candidates[5]
    trace = list(active["trace"])
    fresh_freezes = list(active["freezes"])
    mounts = active["mounts"]
    opened = active["opened"]
    enters = [row[0] for row in trace if row[0].endswith(".enter")]
    assert len(mounts) == 1, "Nexus must mount one shared dynamic adapter"
    assert len(opened) == 1, "Nexus must open one headless sink graph"
    adapter, initial, policy = mounts[0]
    assert adapter._sink_graph is opened[0]
    assert adapter._sink_graph is not adapter._observer
    assert policy is adapter._policy
    final_plan = fresh._plan_cache.get(fresh._active_scan, integrate_2d=True)
    provisional = copy.deepcopy(final_plan); provisional.integration_1d.npt += 1
    bad_policy = fresh._resolve_dynamic_session_policy(frozen=fresh.run_configuration,
        plan=provisional, frame_shape=(2, 2), dtype=np.uint16)
    policy_effects = []
    fresh._scan_session_adapter = object()
    fresh._close_reduction_session = lambda: policy_effects.append("close") or True
    fresh.release_retained_custody = lambda: policy_effects.append("release") or True
    pending = SimpleNamespace(map_raw=np.ones((2, 2), dtype=np.uint16), bg_raw=None)
    with pytest.raises(RunConfigurationRefused) as policy_refusal:
        fresh._mount_dynamic_reduction_session((fresh.run_configuration.generation, os.path.abspath(target)),
            frozen=fresh.run_configuration, scan=fresh._active_scan, plan=final_plan, pending_frame=pending,
            output_path=target, gui_thread_id=fresh.gui_thread_id, policy=bad_policy)
    assert (policy_refusal.value.stage, policy_refusal.value.generation, policy_effects) == (
        "dynamic-policy", fresh.run_configuration.generation, [])
    fresh._scan_session_adapter = None
    width = min(policy.flush.interval, policy.flush.hard_threshold())
    assert width == len(initial.labels) == 8
    assert len(fresh_freezes) == 1
    assert len(fresh_freezes[0]) == 1
    assert fresh_freezes[0][0].idx == 0
    submits = [row for row in trace if row[0] == "adapter.submit.enter"]
    assert len(submits) == 9
    assert all(row[1] is adapter for row in submits)
    submit_tokens = [row[3]["attempt_token"] for row in submits]
    begin_tokens = [row[2] for row in trace
                    if row[0] == "adapter.begin_attempt.return"]
    assert submit_tokens == begin_tokens
    assert len({id(token) for token in submit_tokens}) == 9
    attempt_events = {"adapter.discover.enter", "adapter.begin_attempt.enter",
                      "adapter.record_enqueued.enter", "adapter.submit.enter"}
    per_frame = [name for name in enters if name in attempt_events]
    expected_attempt = ["adapter.discover.enter", "adapter.begin_attempt.enter",
                        "adapter.record_enqueued.enter", "adapter.submit.enter"]
    assert all(per_frame[pos:pos + 4] == expected_attempt
               for pos in range(0, len(per_frame), 4))
    commit_positions = [index for index, name in enumerate(enters)
                        if name == "adapter.commit_epoch.enter"]
    extend_positions = [index for index, name in enumerate(enters)
                        if name == "adapter.extend_live.enter"]
    submit_positions = [index for index, name in enumerate(enters)
                        if name == "adapter.submit.enter"]
    assert len(commit_positions) == 1
    assert len(extend_positions) == 1
    commit = commit_positions[0]
    extend = extend_positions[0]
    assert enters[:commit].count("adapter.submit.enter") == 8
    assert commit < extend < submit_positions[8]
    finish = enters.index("adapter.finish.enter")
    source_close = enters.index("execution.close.enter")
    assert submit_positions[8] < finish
    assert finish < source_close
    assert "adapter.stop.enter" not in enters
    assert any("done" in text.lower() for text in fresh_statuses)
    successor = next(row[2][0] for row in trace
                     if row[0] == "adapter.extend_live.enter")
    assert initial.labels == tuple(range(8))
    assert initial.source.extent == 8
    assert successor.labels == tuple(range(9))
    assert successor.source.extent == 9
    assert successor.source.generation == initial.source.generation + 1
    assert successor.source.path == initial.source.path
    assert successor.source.adapter_id == initial.source.adapter_id
    assert successor.source.dataset_paths == initial.source.dataset_paths
    assert successor.source.size == initial.source.size
    assert successor.source.mtime_ns == initial.source.mtime_ns
    assert successor.source.digest == initial.source.digest
    assert successor.entry == initial.entry
    assert successor.source_base == initial.source_base
    assert successor.source_identity == initial.source_identity
    assert successor.science_fingerprint == initial.science_fingerprint
    assert successor.modes == initial.modes
    assert len(initial.source.external_members) == 1
    assert len(successor.source.external_members) == 1
    initial_member = initial.source.external_members[0]
    successor_member = successor.source.external_members[0]
    assert (Path(initial_member.path), initial_member.dataset_path, initial_member.source_start, initial_member.source_stop, initial_member.ordinal) == (
        member, "/entry/data/data", 0, 8, 0)
    assert successor_member.path == initial_member.path
    assert successor_member.dataset_path == initial_member.dataset_path
    assert successor_member.size == initial_member.size
    assert successor_member.mtime_ns == initial_member.mtime_ns
    assert successor_member.source_start == initial_member.source_start
    assert successor_member.ordinal == initial_member.ordinal
    assert initial_member.source_stop == 8
    assert successor_member.source_stop == 9
    with h5py.File(target) as handle:
        written_1d = tuple(map(int, handle["entry/integrated_1d/frame_index"][()]))
        written_2d = tuple(map(int, handle["entry/integrated_2d/frame_index"][()]))
    assert written_1d == written_2d == tuple(range(9))

    write_source(10)
    before_output = output_snapshot()
    _refused, refused_statuses = run(revise_final_restat=True)
    revised = tuple(active["candidate_calls"])
    assert [os.path.abspath(item.path) for item in revised] == expected_candidate_paths
    assert revised[0] == revised[2] == revised[4] and revised[1] == revised[3] and revised[-1].size == revised[1].size + 1
    assert _refused.command == "stop" and active["mounts"] == active["opened"] == [] and output_snapshot() == before_output and any("changed before mount" in text for text in refused_statuses)
    resumed, _ = run()
    resumed_freezes = list(active["freezes"])
    resumed_submits = [row for row in active["trace"]
                       if row[0] == "adapter.submit.enter"]
    resumed_discovers = [row for row in active["trace"]
                         if row[0] == "adapter.discover.enter"]
    assert len(active["mounts"]) == 1
    assert len(active["opened"]) == 1
    assert len(resumed_submits) == 1
    assert len(resumed_discovers) == 1
    assert resumed_discovers[0][3]["output_label"] == 9
    assert active["mounts"][0][1].labels == tuple(range(10))
    assert resumed._active_scan._committed_append_prefix.intent.labels \
        == tuple(range(9))
    assert len(resumed_freezes) == 1
    assert len(resumed_freezes[0]) == 1
    assert resumed_freezes[0][0].idx == 9
    for key in active["reads"]:
        first = int(key) if isinstance(key, (int, np.integer)) \
            else int(key.start or 0)
        assert first >= 9, f"committed detector pixel was reread: {key!r}"
    fully_skipped, skipped_statuses = run()
    assert active["reads"] == []
    assert active["mounts"] == []
    assert active["opened"] == []
    assert not [
        row for row in active["trace"]
        if row[0] == "adapter.submit.enter"]
    assert any("already processed" in text.lower()
               for text in skipped_statuses)
    assert int(fully_skipped.files_processed) == 0

    with monkeypatch.context() as scoped:
        xye_output = tmp_path / "xye-controller-output"
        display = LiveScan("xye-controller-display", static=True)
        controller = nexusWrangler(str(tmp_path / "unused.nexus"), RLock(), display)
        controller.parameters.child("NeXus File").child("nexus_file").setValue(
            str(source))
        controller.parameters.child("Output").child("h5_dir").setValue(
            str(xye_output))
        controller.processingModeCombo.setCurrentText("Int 1D (XYE)")
        controller_statuses = []
        controller._set_status_text = controller_statuses.append
        admissions, releases, starts, constructions, source_opens, effects = [], [], [], [], [], []
        real_h5_file = h5py.File
        def traced_h5_file(*args, **kwargs):
            source_opens.append((args, dict(kwargs)))
            return real_h5_file(*args, **kwargs)
        controller.thread.release_retained_custody = (
            lambda: releases.append(True) or True)
        controller.sigStart.connect(lambda: (starts.append(True), effects.append(("start",))))
        scoped.setattr(
            wranglerWidget, "_admit_run_configuration",
            staticmethod(lambda *_args, **_kwargs: (admissions.append(True), effects.append(("admit",)))))
        scoped.setattr(
            wrapper_mod, "nexusThread",
            lambda *_args, **_kwargs: constructions.append(True))
        scoped.setattr(h5py, "File", traced_h5_file)
        state = lambda: (controller.thread, controller.fname, controller.scan.data_file,
            controller.command, controller.run_configuration, controller._frozen_setup_pending,
            controller.startButton.isEnabled(), controller.stopButton.isEnabled(),
            getattr(controller.thread, "run_configuration", None))
        baseline = state()
        controller.start()
        assert admissions == []
        assert releases == []
        assert starts == []
        assert constructions == []
        assert source_opens == []
        assert state() == baseline
        assert not xye_output.exists()
        assert controller_statuses
        assert "xye" in controller_statuses[-1].lower()
        assert "refus" in controller_statuses[-1].lower()
        controller.processingModeCombo.setCurrentText("Int 1D")
        owners = []
        def owner_probe(_owner):
            value = owners.pop(0); effects.append(("probe", value)); return value
        scoped.setattr(wranglerWidget, "_active_run_owner", owner_probe)
        controller.thread.release_retained_custody = lambda: effects.append(("release",)) or False
        owners[:] = [None]; effects.clear(); controller.start()
        assert effects == [("probe", None), ("release",)] and state() == baseline
        assert "cleanup" in controller_statuses[-1].lower() and admissions == starts == constructions == source_opens == []
        owners[:] = ["wrangler"]; effects.clear(); controller.start()
        assert effects == [("probe", "wrangler")] and state() == baseline
        is_running = controller.thread.isRunning; controller.thread.isRunning = lambda: True
        owners[:] = [None]; effects.clear(); controller.start()
        assert effects == [("probe", None)] and state() == baseline
        controller.thread.isRunning = is_running
        controller.thread.release_retained_custody = lambda: effects.append(("release",)) or True
        owners[:] = ["output-cleanup", None]; effects.clear(); controller.start()
        assert effects == [("probe", "output-cleanup"), ("release",), ("probe", None), ("admit",), ("start",)]
        assert admissions == starts == [True] and constructions == source_opens == []

    xye_worker, xye_statuses = run("Int 1D (XYE)")
    assert active["reads"] == []
    assert active["mounts"] == []
    assert active["opened"] == []
    assert xye_worker.command == "stop"
    assert any("xye" in text.lower() and "refus" in text.lower()
               for text in xye_statuses)

    write_source(11)
    pending_worker, pending_statuses = run(cleanup_pending=True)
    pending_trace = active["trace"]
    assert pending_worker._scan_session_adapter is not None
    assert pending_worker._execution is not None
    assert pending_worker._execution.stack._h5.id.valid
    assert "execution.close.enter" not in [row[0] for row in pending_trace]
    assert not any("done" in text.lower() for text in pending_statuses)
    assert pending_worker.dynamic_cleanup_pending() is True
    statuses = []
    finished = []
    owner = SimpleNamespace(thread=pending_worker, _set_status_text=statuses.append,
                            finished=SimpleNamespace(emit=lambda: finished.append(True)))
    nexusWrangler._on_worker_thread_finished(owner)
    assert statuses
    assert "pending" in statuses[-1].lower()
    assert finished == [True]
    pending_worker._close_reduction_session = pending_worker._test_real_close
    assert pending_worker.release_retained_custody() is True
    assert pending_worker._execution is None
    pending_names = [row[0] for row in pending_trace]
    assert pending_names.index("adapter.finish.return") \
        < pending_names.index("execution.close.enter")

    write_source(14)
    with monkeypatch.context() as scoped:
        def stop_refusal(owner, frame, **kwargs):
            active["trace"].append(("adapter.submit.stop-refusal", owner, frame, kwargs))
            active["worker"].command = "stop"; return False
        scoped.setattr(_adapter(), "submit", stop_refusal)
        stopped_worker, stopped_statuses = run()
    stopped_trace = active["trace"]
    stopped_names = [row[0] for row in stopped_trace]
    refusal = next(row for row in stopped_trace if row[0] == "adapter.submit.stop-refusal")
    settlement = next(row for row in stopped_trace if row[0] == "adapter.record_failed.enter")
    assert settlement[2][0] is refusal[3]["attempt_token"]
    assert stopped_names.index("adapter.stop.enter") \
        < stopped_names.index("adapter.finish.enter")
    assert stopped_names.index("adapter.finish.return") \
        < stopped_names.index("execution.close.enter")
    assert stopped_worker._execution is None
    assert stopped_worker._reduction_write_error is None
    assert not any("dynamic submit refused" in text.lower() for text in stopped_statuses)
    assert not any("done" in text.lower() for text in stopped_statuses)

    # A source-close fault after graph success retains the exact envelope.
    source_pending, source_pending_statuses = run(source_close_failures=1)
    source_pending_trace = active["trace"]
    source_pending_names = [row[0] for row in source_pending_trace]
    retained_envelope = source_pending._execution
    assert retained_envelope is not None
    assert retained_envelope.stack._h5.id.valid
    assert source_pending._scan_session_adapter is None
    assert source_pending_names.index("adapter.finish.return") \
        < source_pending_names.index("execution.close.enter")
    assert source_pending.dynamic_cleanup_pending() is True
    assert not any("done" in text.lower()
                   for text in source_pending_statuses)
    statuses = []
    owner = SimpleNamespace(thread=source_pending,
                            _set_status_text=statuses.append,
                            finished=SimpleNamespace(emit=lambda: None))
    nexusWrangler._on_worker_thread_finished(owner)
    assert statuses
    assert "pending" in statuses[-1].lower()

    controller = nexusWrangler(str(target), RLock(),
                               LiveScan("controller-display", static=True))
    controller.parameters.child("NeXus File").child("nexus_file").setValue(str(source))
    controller.parameters.child("Output").child("h5_dir").setValue(str(output))
    controller.poni = poni
    controller.publication_store = PublicationStore(max_items=16)
    controller_gui_thread_id = controller._gui_thread_id
    source_pending.gui_thread_id, source_pending._gui_thread_id = 41, -41
    controller.thread = source_pending
    real_public_release = source_pending.release_retained_custody
    real_constructor = wrapper_mod.nexusThread
    def public_release():
        source_pending_trace.append(("worker.release.enter", source_pending))
        result = real_public_release()
        source_pending_trace.append(("worker.release.return", source_pending, result))
        return result
    def replacement_constructor(*args, **kwargs):
        source_pending_trace.append(("replacement.construct",))
        return real_constructor(*args, **kwargs)
    source_pending.release_retained_custody = public_release
    with monkeypatch.context() as scoped:
        scoped.setattr(wrapper_mod, "nexusThread", replacement_constructor)
        controller.setup()
    replacement_names = [row[0] for row in source_pending_trace]
    close_owners = [row[1] for row in source_pending_trace
                    if row[0] == "execution.close.enter"]
    assert close_owners == [retained_envelope, retained_envelope]
    assert replacement_names.index("execution.close.return") \
        < replacement_names.index("worker.release.return")
    assert replacement_names.index("worker.release.return") \
        < replacement_names.index("replacement.construct")
    assert source_pending._execution is None
    assert controller.thread is not source_pending
    assert controller.thread.publication_store is controller.publication_store
    assert controller.thread.gui_thread_id == controller_gui_thread_id != 41
    assert "_gui_thread_id" not in controller.thread.__dict__
    controller.thread.sigRetainedCustody.emit(41, str(target))
    expected_custody = (41, os.path.normcase(os.path.abspath(target)))
    assert controller._viewer_preserve_token[:2] == expected_custody

    old = controller.thread
    baseline = (controller.thread, controller.fname, controller.scan.data_file)
    releases = []
    old.release_retained_custody = lambda: releases.append(True) or False
    controller.setup()
    assert releases == [True]
    assert (controller.thread, controller.fname, controller.scan.data_file) == baseline

    body = _source(nexusThread._run_body)
    legacy = (
        "_get_reduction_session", "open_live_reduction_session",
        "reduce_live_frames", "_publish(", "_save_to_disk",
        "_flush_xye_buffer", "_final_save_to_nexus", "scan.add_frame",
        "scan.save_to_nexus", "scan._save_to_nexus",
        "PreparedOverwriteTransaction", "PreparedXyeOutput",
    )
    for name in legacy:
        assert name not in body, f"legacy Nexus owner remains: {name}"
    retired_module_owners = {
        "OverwritePhase", "PreparedOverwriteTransaction", "PreparedXyeOutput",
        "open_live_reduction_session", "reduce_live_frames", "write_xye",
        "read_provenance", "_get_h5pool",
    }
    for name in retired_module_owners:
        assert not hasattr(mod, name), f"retired Nexus module owner remains: {name}"
    retired_nexus_methods = {
        "_final_save_to_nexus", "_append_identity", "_prepare_output_for_run",
        "_emit_overwrite_refusal", "_repair_overwrite_locked",
        "_rollback_overwrite_locked", "_execute_overwrite_locked",
        "_write_run_result", "_finish_overwrite_cleanup", "_save_to_disk",
        "_flush_xye_buffer", "_clear_stale_xye_tail", "_finish_xye_output",
        "_publish", "_get_reduction_session", "_reduction_session_key_for",
    }
    for name in retired_nexus_methods:
        assert not hasattr(nexusThread, name), f"retired Nexus method remains: {name}"
    retired_base_methods = {
        "_get_reduction_session", "_reduction_session_key_for",
        "_flush_xye_buffer", "_write_xye_entries",
        "_reset_xye_output_notifications", "save_1d", "_save_to_disk",
    }
    for name in retired_base_methods:
        assert not hasattr(wranglerThread, name), f"retired base owner remains: {name}"
    retired_instance_state = {
        "_reduction_session", "_reduction_session_key", "_xye_buffer",
        "_xye_lock", "_xye_ready_dirs",
    }
    assert retired_instance_state.isdisjoint(fresh.__dict__)
    retired_envelope_state = {"append_qualified", "overwrite", "xye"}
    assert retired_envelope_state.isdisjoint(
        set(mod.PreparedNexusExecution.__slots__))
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread as image_mod
    assert not hasattr(image_mod, "write_xye")
    assert "_reset_xye_output_notifications" not in _source(image_mod.imageThread.run)

def test_canonical_finish_and_prefix_stop_adopt_light_bank_for_browse(
    tmp_path, monkeypatch,
):
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerWidget
    failures = []
    for terminal in ("finish", "prefix-stop"):
        with monkeypatch.context() as scoped:
            case = _c3_real_source_case(tmp_path / terminal, "tiff")
            while case.worker.publication_store.generation == int(
                    case.worker.run_configuration.generation):
                case.worker.publication_store.clear()
            retained_signals = []
            controller = SimpleNamespace(_viewer_preserve_token=None)
            case.worker.sigRetainedCustody.connect(
                lambda generation, path: (
                    retained_signals.append((int(generation), os.path.abspath(path))),
                    wranglerWidget._adopt_retained_custody(
                        controller, generation, path)))
            trace, adapters = _c3_run(case, scoped)
            case.worker.command = "start" if terminal == "finish" else "stop"
            closed = case.worker._close_reduction_session()
            try:
                assert closed is True and len(adapters) == 1
                adapter = adapters[0]
                assert case.worker._scan_session_adapter is None
                assert getattr(case.worker, "_retained_scan_session_adapter", None) is adapter
                assert adapter._light_slot.state.value == "retained"
                assert adapter._light_slot.custody_receipt.retained_rows == 1
                assert adapter._publication_store._light_1d is adapter._light_lease
                assert adapter._light_authority.snapshot().reserved_bytes > 0
                assert retained_signals == [(
                    adapter._publication_store.generation,
                    os.path.abspath(os.fspath(adapter._sink_graph.path)))]
                assert retained_signals[0][0] != adapter._generation
                assert controller._viewer_preserve_token[:2] == (
                    adapter._publication_store.generation,
                    os.path.normcase(os.path.abspath(
                        os.fspath(adapter._sink_graph.path))))
            except AssertionError as exc:
                failures.append((terminal, str(exc), [row[0] for row in trace]))
            finally:
                retained = getattr(case.worker, "_retained_scan_session_adapter", None)
                if retained is not None:
                    retained.release_retained_custody()
    assert not failures, failures

def test_noncanonical_stop_zero_prefix_and_abort_release_light_bank(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import NexusSink; from xrd_tools.session import Light1DModeData, Light1DRecord
    failures = []
    for terminal in ("zero-prefix", "noncanonical-stop", "abort"):
        with monkeypatch.context() as scoped:
            case = _c3_real_source_case(tmp_path / terminal, "tiff")
            if terminal == "zero-prefix":
                def refuse(owner, _frame, **_kwargs):
                    case.worker.command = "stop"
                    return False
                scoped.setattr(_adapter(), "submit", refuse)
            elif terminal == "abort":
                scoped.setattr(
                    NexusSink, "write",
                    lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        OSError("C3 injected abort")),
                )
            trace, adapters = _c3_run(case, scoped, settle=terminal == "noncanonical-stop"); assert len(adapters) == 1; adapter = adapters[0]
            if terminal == "noncanonical-stop":
                lease = adapter._light_lease; mode = lease.layout.modes[0]
                values = Light1DModeData(*(np.zeros(spec.length, dtype=spec.dtype) for spec in (mode.coordinate, mode.intensity)))
                lease.retain(Light1DRecord(99, lease.generation, mode.mode, {mode.mode: values}),
                             grant_id=lease.grant_id, generation=lease.generation)
                assert set(lease.keys()) - adapter._accounting.ledger.snapshot().accepted == {99}
            case.worker.command = "stop"
            case.worker._close_reduction_session()
            try:
                assert adapter._light_slot.state.value == "cancelled"
                assert adapter._light_lease.state.value == "released"
                assert adapter._light_authority.snapshot().reservation_count == 0
                assert case.worker._retained_scan_session_adapter is None
            except AssertionError as exc:
                failures.append((terminal, str(exc), [row[0] for row in trace]))
    assert not failures, failures

def test_replacement_and_close_retry_exact_retained_custody_before_successor(
    tmp_path, monkeypatch,
):
    """All replacement families share one retryable gate and one-shot reload."""
    from tests.xdart.test_live_refresh import _FakeAction, _FakeListWidget, _FakeSignal, _wrangler_host
    from tests.xdart.test_multi_scan_frame_boundary import _boundary_host
    from xdart.gui.tabs.static_scan import h5viewer as h5viewer_mod
    from xdart.gui.tabs.static_scan.scan_threads import FileTask
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerWidget
    from xrd_tools.io import ImageSourceKind
    H5Viewer = h5viewer_mod.H5Viewer
    assert callable(getattr(H5Viewer, "_set_pre_transition_callback", None))
    assert callable(getattr(H5Viewer, "_before_transition", None))
    class _C3Store:
        def __init__(self, generation=41):
            self.generation, self.light_root = int(generation), object()
            self.entries, self.clear_count = {"old": object()}, 0
        def clear(self):
            self.entries.clear(); self.clear_count += 1
    class _C3ReleaseGate:
        def __init__(self):
            self.allow_release, self.retry_token, self.calls = False, object(), []
        def __call__(self, transition, *args, **facts):
            self.calls.append((str(transition), self.retry_token, dict(facts)))
            return "released" if self.allow_release else "cleanup-pending"
    def _controller(generation, output_path):
        releases = []
        controller = SimpleNamespace(_viewer_preserve_token=None,
            thread=SimpleNamespace(release_retained_custody=lambda: releases.append(True) or True))
        controller._viewer_transition = MethodType(wranglerWidget._viewer_transition, controller)
        wranglerWidget._adopt_retained_custody(controller, generation, os.fspath(output_path))
        return controller, releases
    def _make_viewer(*, selected="one.xye", generation=41):
        list_data = _FakeListWidget(["1"])
        list_data.itemSelectionChanged = SimpleNamespace(disconnect=lambda *_a: None, connect=lambda *_a: None)
        list_scans = _FakeListWidget([selected])
        list_scans.selectAll()
        viewer = SimpleNamespace(
            data_lock=RLock(), viewer_rows_1d={1: object()},
            viewer_rows_2d={1: {"map_raw": np.ones((2, 2)), "thumbnail": None}}, frames={1: object()},
            frame_ids=["1"], publication_store=_C3Store(generation), dirname=str(tmp_path / "old"),
            latest_idx=1, new_scan_loaded=True, live_run_active=False, auto_last=False, new_scan=False,
            update_2d=True, _run_writing=False, _pre_transition_callback=None,
            _accepted_preserve_operation=None, _browser_restore_in_progress=False,
            _auto_select_last_on_finish=False, _raw_cache_order=[1], _displayed_list_count=1,
            _displayed_last_label="1", _load_generation=0, _load_worker=None, _load_thread=None,
            _pending_load_ids=None, _update_coalesce_timer=None, _load_coalesce_timer=None,
            scan=SimpleNamespace(data_file=str(tmp_path / "old.nxs"), frames=SimpleNamespace(index=[1]), gi=False),
            file_thread=SimpleNamespace(fname=str(tmp_path / "old.nxs"), queue=Queue(),
                                        scan=SimpleNamespace(), live_run=False, no_nxs=False),
            ui=SimpleNamespace(listData=list_data, listScans=list_scans,
                               labelCurrent=SimpleNamespace(setText=lambda *_a: None)),
            _h5pool=SimpleNamespace(close=lambda *_a: None), _cancel_count=0,
            sigUpdate=_FakeSignal(), sigThreadFinished=_FakeSignal())
        def cancel_pending_loads(): viewer._cancel_count += 1
        viewer.cancel_pending_loads = cancel_pending_loads
        for name in (
            "_ensure_file_thread_running", "_remember_displayed_frames",
            "_clear_raw_cache", "_populate_image_viewer_rows",
            "_refresh_nexus_selected_preview", "update_scans", "update",
            "set_current_frame", "data_changed",
        ):
            setattr(viewer, name, lambda *_a, **_k: None)
        viewer._load_single_frame, viewer._nexus_summary_rows = (lambda *_a, **_k: True), (lambda _summary: [])
        for name in (
            "_set_pre_transition_callback", "_before_transition", "set_file",
            "_load_xye_files", "_load_nexus_file", "_load_image_file",
            "data_reset", "open_folder", "enter_viewer_mode_cleanup",
            "thread_finished",
        ):
            setattr(viewer, name, MethodType(getattr(H5Viewer, name), viewer))
        return viewer
    def _snapshot(viewer):
        rows = lambda mapping: tuple((key, id(value)) for key, value in mapping.items())
        return (
            id(viewer.publication_store), id(viewer.publication_store.light_root),
            rows(viewer.publication_store.entries), viewer.publication_store.clear_count,
            rows(viewer.viewer_rows_1d), rows(viewer.viewer_rows_2d),
            rows(viewer.frames), tuple(viewer.frame_ids),
            tuple(viewer.ui.listData.item(i).text() for i in range(viewer.ui.listData.count())),
            tuple(item.text() for item in viewer.ui.listScans.selectedItems()),
            viewer.dirname, viewer.file_thread.fname,
            tuple(id(item) for item in tuple(viewer.file_thread.queue.queue)), viewer._cancel_count)
    monkeypatch.setattr(h5viewer_mod, "read_xye", lambda _p: (np.arange(2.0), np.arange(2.0), np.ones(2)))
    import xrd_tools.io as io_mod
    monkeypatch.setattr(io_mod, "inspect_nexus", lambda *_a, **_k: object())
    unknown = SimpleNamespace(kind=ImageSourceKind.UNKNOWN, frame_labels=())
    monkeypatch.setattr(h5viewer_mod.ImageViewerController, "classify", staticmethod(lambda _path: unknown))
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    class _C3Dialog:
        ShowDirsOnly = object()
        def getExistingDirectory(self, **_kwargs): return str(chosen)
    monkeypatch.setattr(h5viewer_mod, "QFileDialog", _C3Dialog)
    monkeypatch.setattr(h5viewer_mod, "save_session", lambda *_a, **_k: None)
    import xdart.utils.browse as browse_mod
    monkeypatch.setattr(browse_mod, "remember_browse_path", lambda *_a: None)
    transitions = (
        ("set_file", "scan.nxs", (str(tmp_path / "next.nxs"),)),
        ("_load_xye_files", "one.xye", ()),
        ("_load_nexus_file", "scan.nxs", (str(tmp_path / "scan.nxs"),)),
        ("_load_image_file", "image.edf", (str(tmp_path / "image.edf"),)),
        ("data_reset", "scan.nxs", ()), ("open_folder", "scan.nxs", ()),
        ("enter_viewer_mode_cleanup", "scan.nxs", ()))
    for family, selected, arguments in transitions:
        viewer, gate = _make_viewer(selected=selected), _C3ReleaseGate()
        viewer._set_pre_transition_callback(gate)
        before = _snapshot(viewer)
        transition = getattr(viewer, family)
        transition(*arguments)
        assert _snapshot(viewer) == before, family
        assert [call[0] for call in gate.calls] == [family]
        assert viewer._pre_transition_callback is gate
        gate.allow_release = True
        transition(*arguments)
        assert len(gate.calls) == 2 and gate.calls[0][1] is gate.calls[1][1]
        assert _snapshot(viewer) != before, family
    output = tmp_path / "output" / "scan.nxs"; output.parent.mkdir()
    alias = output.parent / "nested" / ".." / output.name
    viewer = _make_viewer(generation=41); controller, releases = _controller(41, output)
    viewer._set_pre_transition_callback(controller._viewer_transition)
    viewer.set_file(str(alias), internal=True)
    queued = viewer.file_thread.queue.queue[-1] if viewer.file_thread.queue.qsize() else None
    assert isinstance(queued, FileTask)
    token = controller._viewer_preserve_token
    assert token is not None and token[0] == 41 and token[2] is queued
    store = viewer.publication_store
    store_identity = id(store), id(store.light_root), dict(store.entries), store.clear_count
    viewer.data_reset()
    assert (id(store), id(store.light_root), dict(store.entries), store.clear_count) == store_identity
    assert viewer.viewer_rows_1d == {} and viewer.viewer_rows_2d == {}
    assert controller._viewer_preserve_token is not None; viewer.thread_finished(queued)
    assert controller._viewer_preserve_token is None and releases == []
    before_queue = viewer.file_thread.queue.qsize()
    viewer.set_file(str(output), internal=True); assert len(releases) == 1
    assert viewer.file_thread.queue.qsize() == before_queue + 1

    # Roll back before enqueue; after Queue.put the exact task remains bound.
    failed = _make_viewer(generation=41); failed_owner, failed_releases = _controller(41, output)
    failed._set_pre_transition_callback(failed_owner._viewer_transition)
    failed._ensure_file_thread_running = lambda: (_ for _ in ()).throw(RuntimeError("pre-enqueue refusal"))
    failed.set_file(str(output), internal=True)
    assert failed.file_thread.queue.empty() and failed_releases == []
    assert failed_owner._viewer_preserve_token == (41, os.path.normcase(os.path.abspath(output)), None)
    accepted = _make_viewer(generation=41); accepted_owner, accepted_releases = _controller(41, output)
    accepted._set_pre_transition_callback(accepted_owner._viewer_transition)
    with monkeypatch.context() as scoped:
        scoped.setattr(h5viewer_mod, "run_config_debug_enabled", lambda: True)
        scoped.setattr(h5viewer_mod, "new_display_context_operation", lambda **_facts: object())
        scoped.setattr(h5viewer_mod, "display_context_transition_log", lambda *_a, **_k:
                       (_ for _ in ()).throw(RuntimeError("post-enqueue failure")))
        accepted.set_file(str(output), internal=True)
    accepted_task = accepted.file_thread.queue.queue[-1]
    assert accepted_owner._viewer_preserve_token[2] is accepted_task and accepted_releases == []
    accepted.thread_finished(accepted_task)
    assert accepted_owner._viewer_preserve_token is None

    foreign_viewer = _make_viewer(generation=41); foreign_controller, foreign_releases = _controller(41, output)
    foreign_viewer._set_pre_transition_callback(foreign_controller._viewer_transition)
    foreign_viewer.set_file(str(output), internal=True)
    original_task = foreign_viewer.file_thread.queue.queue[-1]
    reconstructed = FileTask(original_task.method, original_task.operation)
    foreign_viewer.thread_finished(reconstructed)
    assert reconstructed == original_task and reconstructed is not original_task
    assert foreign_controller._viewer_preserve_token is None and len(foreign_releases) == 1
    mismatch_rows = (
        (42, str(output), True, "set_file"),
        (41, str(tmp_path / "different.nxs"), True, "set_file"),
        (41, str(output), False, "set_file"),
        (41, str(output), True, "data_reset"),
    )
    for generation, path, internal, family in mismatch_rows:
        mismatch = _make_viewer(generation=generation); authority, mismatch_releases = _controller(41, output)
        mismatch._set_pre_transition_callback(authority._viewer_transition)
        (mismatch.set_file(path, internal=internal)
         if family == "set_file" else mismatch.data_reset())
        assert len(mismatch_releases) == 1 and authority._viewer_preserve_token is None

    # Successor releases A; B abort cannot leave A's scalar token reusable.
    chain, chain_releases = _controller(41, output)
    token_a = chain._viewer_preserve_token
    assert chain._viewer_transition("successor") == "released"
    assert chain._viewer_preserve_token is None and chain_releases == [True]
    output_b = tmp_path / "output" / "scan_b.nxs"
    wranglerWidget._adopt_retained_custody(chain, 42, output_b)
    assert chain._viewer_preserve_token != token_a
    assert chain._viewer_transition("abort") == "released"
    assert chain._viewer_preserve_token is None and chain_releases == [True, True]
    with monkeypatch.context() as scoped:
        scoped.setattr(imageWrangler, "_stage_loaded_scan_calibration", lambda _owner: {})
        for stitch in (False, True):
            starter = _wrangler_host("Int 2D"); starter.stitch_mode = stitch
            starter.sigStitchRequested = _FakeSignal()
            starter._inputs_valid = lambda _staged: True
            starter._viewer_preserve_token = (41, os.path.normcase(os.path.abspath(output)), None)
            starter.thread.release_retained_custody = lambda: False
            imageWrangler.start(starter)
            assert starter._viewer_preserve_token is not None
            assert starter.sigStart.emitted == [] and starter.sigStitchRequested.emitted == []

    class _Stack:
        def __init__(self, widgets, current):
            self.widgets, self.current, self.blocked = widgets, current, False
        def widget(self, index): return self.widgets[index]
        def indexOf(self, widget): return self.widgets.index(widget)
        def currentIndex(self): return self.current
        def setCurrentIndex(self, index): self.current = int(index)
        def blockSignals(self, blocked):
            prior, self.blocked = self.blocked, bool(blocked)
            return prior
    static_viewer, static_gate = _make_viewer(), _C3ReleaseGate(); static_viewer._set_pre_transition_callback(static_gate)
    incoming, old_thread = SimpleNamespace(), SimpleNamespace(command="start", isRunning=lambda: False)
    old_wrangler = SimpleNamespace(thread=old_thread, command="start")
    stack = _Stack([incoming, old_wrangler], 0)
    host = SimpleNamespace(h5viewer=static_viewer, wrangler=old_wrangler, ui=SimpleNamespace(wranglerStack=stack))
    staticWidget.set_wrangler(host, 0)
    assert stack.currentIndex() == 1 and stack.widget(1) is host.wrangler
    assert [call[0] for call in static_gate.calls] == ["wrangler_swap"]
    close_gate, stops, close_viewer = _C3ReleaseGate(), [], _make_viewer()
    close_viewer._set_pre_transition_callback(close_gate)
    close_thread = SimpleNamespace(command="start", isRunning=lambda: True,
                                   wait=lambda timeout: stops.append(timeout) or True)
    close_wrangler = SimpleNamespace(thread=close_thread, command="start")
    close_host = SimpleNamespace(h5viewer=close_viewer, wrangler=close_wrangler)
    staticWidget.close(close_host)
    assert not getattr(close_host, "_tearing_down", False)
    assert stops == [30000] and close_thread.command == close_wrangler.command == "stop"
    assert [call[0] for call in close_gate.calls] == ["page_close"]

    # A pending caller restores the last accepted text before flag mutation.
    class _Combo:
        def __init__(self, text): self.text, self.blocked = text, False
        def currentText(self): return self.text
        def setCurrentText(self, text): self.text = str(text)
        def blockSignals(self, blocked):
            prior, self.blocked = self.blocked, bool(blocked)
            return prior
    combo, cleanup_calls = _Combo("Image Viewer"), []
    mode_viewer = _make_viewer()
    mode_viewer.enter_viewer_mode_cleanup = lambda: cleanup_calls.append(True) or False
    mode_host = SimpleNamespace(
        h5viewer=mode_viewer, controls=SimpleNamespace(modeCombo=combo),
        _accepted_processing_mode="Int 2D", _accepted_viewer_mode_transition=None)
    for name in ("_sync_processing_mode_to_scan", "_apply_integration_control_state",
                 "_fit_controls_height", "_refresh_controls_v2_profile"):
        setattr(mode_host, name, lambda *_a, **_k: None)
    mode_host._preflight_processing_viewer_mode = MethodType(staticWidget._preflight_processing_viewer_mode, mode_host)
    caller = SimpleNamespace(ui=SimpleNamespace(processingModeCombo=combo), _h19_host=mode_host, _prev_viewer_mode="")
    imageWrangler._on_mode_changed(caller)
    assert combo.currentText() == "Int 2D" and caller._prev_viewer_mode == ""
    assert mode_host._accepted_viewer_mode_transition is None
    mode_viewer.enter_viewer_mode_cleanup = lambda: cleanup_calls.append(True) or True
    combo.setCurrentText("Image Viewer")
    assert mode_host._preflight_processing_viewer_mode("Image Viewer") is True
    mode_viewer.ui.listScans, mode_viewer.viewer_mode = _FakeListWidget(["scan.nxs"]), None
    mode_viewer._apply_frames_panel_width = lambda *_a: None
    mode_viewer.actionNewFile, mode_viewer.actionSaveDataAs = _FakeAction(), _FakeAction()
    mode_viewer.update_scans = lambda: None
    mode_host.controls.current_mode = combo.currentText
    mode_host.wrangler = SimpleNamespace(tree=SimpleNamespace(setEnabled=lambda *_a: None), h5_dir=None)
    mode_host.displayframe = SimpleNamespace(_viewer_is_xdart=False,
        set_viewer_display_mode=lambda *_a: None, clear_display_state=lambda: None)
    assert staticWidget._on_viewer_mode_changed(mode_host, "image") is True
    assert mode_host._accepted_viewer_mode_transition is None
    assert len(cleanup_calls) == 2 and mode_viewer.viewer_mode == "image"
    assert staticWidget._on_viewer_mode_changed(mode_host, "") is True
    assert mode_viewer.viewer_mode is None and len(cleanup_calls) == 2

    with monkeypatch.context() as scoped:
        dynamic = _c3_real_source_case(tmp_path / "dynamic-rescope", "tiff")
        dynamic_host, dynamic_scan = _boundary_host()
        dynamic_store, dynamic_gate = dynamic.worker.publication_store, _C3ReleaseGate()
        dynamic_host.publication_store = dynamic_store
        dynamic_host.h5viewer.publication_store = dynamic_store
        dynamic_host.h5viewer.live_run_active = True
        for name in ("_set_pre_transition_callback", "_before_transition"):
            setattr(dynamic_host.h5viewer, name, MethodType(getattr(H5Viewer, name), dynamic_host.h5viewer))
        dynamic_host.h5viewer._pre_transition_callback = None
        dynamic_host.h5viewer._set_pre_transition_callback(dynamic_gate)
        queued_before = object()
        dynamic_host.h5viewer.file_thread = SimpleNamespace(fname="prior.nxs", queue=Queue())
        dynamic_host.h5viewer.file_thread.queue.put(queued_before)
        dynamic_scan.frames.data_file = "prior.nxs"
        dynamic_host._scan_fname_cache = {}
        delivered = []
        def rescope_from_signal(name, path, *_args):
            dynamic_host._scan_fname_cache[name] = path
            delivered.append((name, dynamic_store.generation))
            staticWidget._rescope_frame_panel_to(dynamic_host, name)
        dynamic.worker.sigUpdateFile.connect(rescope_from_signal)
        _, adapters = _c3_run(dynamic, scoped, settle=True)
        adapter = adapters[0]
        output_path = os.path.abspath(os.fspath(adapter._sink_graph.path))
        assert len(delivered) == 1 and dynamic_gate.calls == []
        assert dynamic_store.generation == delivered[0][1]
        assert dynamic_store.allocation is adapter._policy.allocation and dynamic_store._light_1d is adapter._light_lease
        assert dynamic_store._items and dynamic_store._light_1d_items
        assert dynamic_host.h5viewer.file_thread.fname == output_path
        assert dynamic_scan.data_file == dynamic_scan.frames.data_file == output_path
        assert tuple(dynamic_host.h5viewer.file_thread.queue.queue) == (queued_before,)
        dynamic.worker.command = "start"
        dynamic.worker._close_reduction_session()
        dynamic.worker.release_retained_custody()
    rescope_host, scan = _boundary_host(); rescope_gate = _C3ReleaseGate()
    for name in ("_set_pre_transition_callback", "_before_transition"):
        setattr(rescope_host.h5viewer, name, MethodType(getattr(H5Viewer, name), rescope_host.h5viewer))
    rescope_host.h5viewer._pre_transition_callback = None
    rescope_host.h5viewer._set_pre_transition_callback(rescope_gate)
    before_rescope = (scan.name, tuple(scan.frames.index))
    rescope_host._rescope_frame_panel_to("new")
    assert (scan.name, tuple(scan.frames.index)) == before_rescope
    assert [call[0] for call in rescope_gate.calls] == ["display_rescope"]
    rescope_gate.allow_release = True
    rescope_host._rescope_frame_panel_to("newer")
    assert scan.name == "newer"
    assert rescope_gate.calls[0][1] is rescope_gate.calls[1][1]

def test_same_run_extension_commits_before_source_cursor_advance(
    tmp_path, monkeypatch,
):
    from tests.xdart.test_frame_read_partial_write import _write_valid_tif
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread as worker_api
    from xrd_tools.io import AppendDisposition, AppendRefused
    from xrd_tools.session import FlushPolicy, RunIntent
    import xrd_tools.io as io_api
    import xrd_tools.reduction.core as reduction_core
    case = _c3_real_source_case(tmp_path, "tiff", processing_mode="Int 2D")
    for label in range(2, 10): _write_valid_tif(case.source.with_name(f"scan_{label:05d}.tif"))
    trace = []
    real_should_flush = FlushPolicy.should_flush
    def should_flush(owner, *args, **kwargs):
        result = real_should_flush(owner, *args, **kwargs)
        trace.append(("policy.should_flush", owner, args, dict(kwargs), result))
        return result
    monkeypatch.setattr(FlushPolicy, "should_flush", should_flush)
    real_cursor, cursors = worker_api._nexus_append_cursor, []
    def cursor(path, **kwargs):
        trace.append(("cursor.enter", os.path.abspath(os.fspath(path))))
        result = real_cursor(path, **kwargs)
        cursors.append(result[2]); trace.append(("cursor.return", result[2]))
        return result
    monkeypatch.setattr(worker_api, "_nexus_append_cursor", cursor)
    real_preflight, preflights, drop = io_api.prepare_append_preflight, [], False
    def preflight(path, intent, **kwargs):
        prefix = kwargs.get("committed_prefix")
        preflights.append((intent, prefix)); trace.append(("preflight", intent, prefix))
        if drop: Path(path).unlink()
        return real_preflight(path, intent, **kwargs)
    monkeypatch.setattr(reduction_core, "prepare_append_preflight", preflight)
    _c3_install_adapter_trace(monkeypatch, {"trace": trace})
    _c3_wrap_worker_source_seams(case, trace)
    with _c3_worker_lifetime(case.worker):
        case.worker.run()
        first_trace, frozen = tuple(trace), case.worker.run_configuration
        real_observe = worker_api._source_observation
        def observe(path):
            trace.append(("source.observe", Path(path).name))
            return real_observe(path)
        monkeypatch.setattr(worker_api, "_source_observation", observe)
        _write_valid_tif(case.source.with_name("scan_00010.tif"))
        candidate = RunIntent.from_frozen(frozen); candidate.output_mode = "Append"
        case.worker.run_configuration = case.worker._admitted_run_configuration = (
            candidate.freeze(generation=frozen.generation + 1))
        case.worker.command = "start"; case.worker.run()
        second_trace = tuple(trace[len(first_trace):])
        assert case.worker._append_skip_without_reading == 9
        assert case.worker.files_processed == case.worker._last_files_processed == 1
        assert sum(case.worker.files_processed_by_output.values()) == 1
        assert case.worker.files_processed_by_output == case.worker._last_files_processed_by_output
        assert Path(case.worker.fname).exists()
        _write_valid_tif(case.source.with_name("scan_00011.tif"))
        candidate = RunIntent.from_frozen(case.worker.run_configuration)
        case.worker.run_configuration = case.worker._admitted_run_configuration = (
            candidate.freeze(generation=candidate.generation + 1))
        third_start, preflight_start, cursor_start = len(trace), len(preflights), len(cursors)
        statuses = []; case.worker.showLabel.connect(statuses.append)
        drop, case.worker.command = True, "start"
        case.worker.run()
        third_trace = tuple(trace[third_start:])
        third_preflights, third_cursors = preflights[preflight_start:], cursors[cursor_start:]
    adapters = [row[1] for row in first_trace if row[0] == "mount.return"]
    try:
        names = [row[0] for row in first_trace]
        assert len(adapters) == 1
        commits = [i for i, name in enumerate(names) if name == "adapter.commit_epoch.enter"]
        extends = [i for i, name in enumerate(names) if name == "adapter.extend_live.return"]; extend_intents = list(dict.fromkeys(row[2][0] for row in first_trace if row[0] == "adapter.extend_live.enter"))
        submits = [i for i, name in enumerate(names) if name == "adapter.submit.enter"]
        accepts = [i for i, row in enumerate(first_trace) if row[0] == "adapter.submit.return" and row[2] is True]
        source_commits = [i for i, name in enumerate(names) if name == "_commit_frame.enter"]
        due = [i for i, row in enumerate(first_trace) if row[0] == "policy.should_flush" and row[4] is True]
        assert len(submits) == len(accepts) == len(source_commits) == 9
        assert len(commits) == 1 and len(extends) == 8 and extends[0] < due[0]
        assert len(due) == 1 and accepts[7] < due[0] < commits[0]
        post_commit_extend = next(i for i in extends if i > commits[0])
        assert commits[0] < post_commit_extend < submits[8]
        first_mount = next(row for row in first_trace if row[0] == "mount.enter")
        assert first_mount[-2].labels == (1,)
        assert [intent.labels for intent in extend_intents] == [
            tuple(range(1, end)) for end in range(3, 11)]
        assert all(row[1] is adapters[0]._policy.flush for row in first_trace if row[0] == "policy.should_flush")
        assert all(accepted < committed for accepted, committed in zip(accepts, source_commits))
        second_names = [row[0] for row in second_trace]
        assert (second_names.count("adapter.submit.enter"), second_names.count("_commit_frame.enter"),
                second_names.count("adapter.commit_epoch.enter"), second_names.count("adapter.extend_live.enter")) == (1, 10, 0, 0)
        assert second_names.count("cursor.enter") == second_names.count("cursor.return") == 1
        assert [row[1] for row in second_trace if row[0] == "source.observe"] == [
            "scan_00010.tif", "scan_00010.tif"]
        assert second_names.index("get_next_image.enter") < second_names.index("cursor.enter") < second_names.index("source.observe")
        second_preflights = preflights[:preflight_start]
        assert len(second_preflights) == 1
        successor, prefix = second_preflights[0]
        second_prefix = next(row[1] for row in second_trace if row[0] == "cursor.return")
        assert prefix is second_prefix
        assert prefix.intent.labels == tuple(range(1, 10)) and successor.labels == tuple(range(1, 11))
        assert len(prefix.intent.source.image_members) == 9 and len(successor.source.image_members) == 10
        assert successor.source.generation == prefix.intent.source.generation + 1
        second_mount = next(row for row in second_trace if row[0] == "mount.enter")
        assert second_mount[-2] is successor

        # Positive predecessor, one cursor, then one durable terminal refusal.
        third_names = [row[0] for row in third_trace]
        assert len(third_preflights) == len(third_cursors) == 1
        refused_intent, refused_prefix = third_preflights[0]
        assert refused_prefix is third_cursors[0]
        assert refused_prefix.intent.labels == tuple(range(1, 11)) and refused_intent.labels == tuple(range(1, 12))
        error = case.worker._reduction_write_error
        assert type(error) is AppendRefused and error.decision.disposition is AppendDisposition.REFUSE
        assert error.decision.reason == "committed Append prefix target disappeared"
        assert statuses == ["Append refused: committed Append prefix target disappeared"]
        assert third_names.count("preflight") == 1
        assert "adapter.submit.enter" not in third_names and [Path(row[1][0]).name for row in third_trace if row[0] == "_commit_frame.enter"] == [f"scan_{label:05d}.tif" for label in range(1, 11)]
        assert case.worker._scan_session_adapter is None and case.worker._retained_scan_session_adapter is None
        assert case.worker.command == "stop"
        assert case.worker.files_processed == case.worker._last_files_processed == 0
        assert case.worker.files_processed_by_output == case.worker._last_files_processed_by_output == {}
        assert not Path(case.worker.fname).exists()
    finally:
        case.worker.command = "start"
        case.worker._close_reduction_session()

def test_dynamic_xye_and_series_average_refuse_before_worker_or_target(tmp_path):
    """Dormant dynamic products refuse before source or output effects."""
    from tests.xdart._accepted_run import series_source
    from tests.xdart.test_bluesky_image_wrangler import _real_dir_watch_thread
    from xdart.modules.frame_publication import PublicationStore
    from xrd_tools.session import RunIntent
    raw = tmp_path / "raw" / "frame_0001.tif"; raw.parent.mkdir(); failures = []
    for label, processing_mode, run_options in (
        ("xye", "Int 1D", {"xye_only": True}),
        ("series-average", "Int 2D", {"series_average": True}),
    ):
        output = tmp_path / label; output.mkdir()
        worker = _real_dir_watch_thread(raw.parent, output)
        worker.publication_store = PublicationStore(max_items=16)
        worker.run_configuration = worker._admitted_run_configuration = RunIntent(
            processing_mode=processing_mode, source_spec=series_source(raw),
            save_path=str(output), run_options=run_options).freeze()
        reads, mounts = [], []
        def forbidden_read(_frozen): reads.append(True); return None, None, 1, None, {}
        worker.get_next_image = forbidden_read
        worker._mount_dynamic_reduction_session = lambda *args, **kwargs: mounts.append((args, kwargs))
        try:
            worker.process_scan(worker.run_configuration)
            error = None
        except BaseException as caught:  # assertion below keeps the red semantic
            error = caught
        try:
            assert isinstance(error, TypeError), error
            assert label.replace("-", " ") in str(error).lower() and reads == mounts == []
            assert list(output.iterdir()) == [] and worker._scan_session_adapter is None
            assert worker.publication_store.allocation is None and worker.publication_store._light_1d is None
        except AssertionError as exc:
            failures.append((label, str(exc), repr(error)))
    assert not failures, failures

def test_cleanup_pending_retains_worker_owner_and_blocks_gui_success(
    tmp_path, monkeypatch,
):
    from tests.xdart.test_batch_finish_select_last import _finish_host
    from xdart.gui.tabs.static_scan.h5viewer import H5Viewer
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerWidget
    from xrd_tools.session import Light1DCustodySlot
    case = _c3_real_source_case(tmp_path, "tiff")
    trace, adapters = _c3_run(case, monkeypatch)
    assert len(adapters) == 1
    adapter = adapters[0]; slot, session = adapter._light_slot, adapter._session
    failures = [RuntimeError("C3 injected adoption fault")]
    original = Light1DCustodySlot.adopt
    def fail_once(owner, *args, **kwargs):
        if owner is slot and failures:
            raise failures.pop()
        return original(owner, *args, **kwargs)
    monkeypatch.setattr(Light1DCustodySlot, "adopt", fail_once)
    case.worker.command = "start"
    assert case.worker._close_reduction_session() is False
    assert case.worker._scan_session_adapter is adapter and case.worker._retained_scan_session_adapter is None
    assert adapter._session is session and slot.state.value == "pending"
    assert case.worker.release_retained_custody() is False
    statuses, finishes = [], []
    wrapper = SimpleNamespace(
        thread=case.worker, finished=SimpleNamespace(emit=lambda: finishes.append(True)),
        _set_status_text=lambda text: statuses.append(str(text)))
    imageWrangler._on_worker_thread_finished(wrapper)
    assert finishes == [True] and statuses and "cleanup" in statuses[-1].lower()
    controller = SimpleNamespace(thread=case.worker, _viewer_preserve_token=None)
    controller._viewer_transition = MethodType(wranglerWidget._viewer_transition, controller)
    viewer = SimpleNamespace(_pre_transition_callback=controller._viewer_transition)
    assert H5Viewer._before_transition(viewer, "data_reset") == "cleanup-pending"
    host = SimpleNamespace(_run_active=False, wrangler=controller,
        _CONTROLS_V2_OWNER_PROBE_ERROR_SUFFIX="-probe-error")
    host._controls_v2_session_activity = MethodType(staticWidget._controls_v2_session_activity, host); host._controls_v2_owner_activity = MethodType(staticWidget._controls_v2_owner_activity, host)
    host._controls_v2_active_run_owner = MethodType(staticWidget._controls_v2_active_run_owner, host)
    assert host._controls_v2_active_run_owner() == "output-cleanup"
    normal = SimpleNamespace(_h19_host=host, thread=case.worker, stitch_mode=False); assert wranglerWidget._active_run_owner(normal) == "output-cleanup"
    normal.stitch_mode = True; assert wranglerWidget._active_run_owner(normal) == "output-cleanup"
    mutations = []
    monkeypatch.setattr(staticWidget, "_arm_runend_overlay_catchup", lambda _host: mutations.append("catchup"))
    for label, batch, xye, failed, retained in (("batch", True, False, False, True), ("xye", True, True, False, False),
              ("zero", False, False, False, False), ("error", True, False, True, False)):
        tail_root = tmp_path / f"run-end-{label}"; tail_root.mkdir(); (tail_root / "scanA").mkdir() if xye else None
        tail, viewer, _nxs, loads = _finish_host(tail_root, batch=batch, saw_frame=label != "zero", xye_only=xye, write_mode="Append" if label == "zero" else "Overwrite", files_processed=0 if label == "zero" else None, append_skipped=1 if label == "zero" else 0, indexed_count=1 if label == "zero" else 0)
        tail.wrangler.dynamic_cleanup_pending = (lambda: False) if (failed or retained or label == "zero") else case.worker.dynamic_cleanup_pending
        if failed: tail.wrangler.thread._reduction_write_error = RuntimeError("writer failed")
        if retained:
            blocker = SimpleNamespace(thread=SimpleNamespace(release_retained_custody=lambda: False), _viewer_preserve_token=(1, os.path.normcase(os.path.abspath(tail_root / "foreign.nxs")), None)); blocker._viewer_transition = MethodType(wranglerWidget._viewer_transition, blocker)
            viewer._pre_transition_callback = blocker._viewer_transition; viewer._before_transition = MethodType(H5Viewer._before_transition, viewer); viewer.publication_store = SimpleNamespace(generation=1)
        if label == "zero":
            load_task = object(); load_owner = SimpleNamespace(thread=SimpleNamespace(release_retained_custody=lambda: mutations.append("premature-release") or False), _viewer_preserve_token=(1, os.path.normcase(os.path.abspath(_nxs)), None)); load_owner._viewer_transition = MethodType(wranglerWidget._viewer_transition, load_owner)
            viewer._pre_transition_callback = load_owner._viewer_transition; viewer._before_transition = MethodType(H5Viewer._before_transition, viewer); viewer.publication_store = SimpleNamespace(generation=1)
            viewer.set_file = lambda path, internal=False: load_owner._viewer_transition("set_file", path=path, generation=1, internal=internal, operation=load_task, accepted=True)
        note = lambda name: None if label == "zero" else mutations.append(name)
        viewer.update_scans = lambda: note("update_scans"); tail._reconcile_h5viewer_frame_list_after_run = lambda *_a: note("reconcile") or 0
        tail._select_finished_scan_row = lambda *_a: note("select"); tail._on_viewer_mode_changed = lambda *_a: note("xye") or True
        tail.wrangler_finished()
        assert loads == [] and viewer._auto_select_last_on_finish is (label == "zero") and (not retained or blocker._viewer_preserve_token is not None) and (label != "zero" or load_owner._viewer_preserve_token[2] is load_task)
    assert mutations == []
    assert load_owner._viewer_transition("thread_finished", operation=load_task) == "preserve" and load_owner._viewer_preserve_token is None
    order = []
    real_close = case.worker._close_reduction_session
    case.worker._prefetch_stop_prior = lambda: order.append("prefetch-dead") or True
    case.worker._eiger_close_master = lambda: order.append("detector-reset")
    real_initialize, real_mount = case.worker.initialize_scan, case.worker._mount_dynamic_reduction_session
    case.worker.initialize_scan = lambda *a, **k: (order.append("initialize"), real_initialize(*a, **k))[1]
    case.worker._mount_dynamic_reduction_session = lambda *a, **k: (order.append("mount"), real_mount(*a, **k))[1]
    def close():
        order.append(("cleanup-retry", case.worker._scan_session_adapter))
        return real_close()
    case.worker._close_reduction_session = close
    case.worker.process_scan = lambda _frozen: order.append("process")
    failures.append(RuntimeError("C3 injected retry remains pending")); case.worker.command = "start"; case.worker.run()
    assert order == ["prefetch-dead", ("cleanup-retry", adapter)]
    case.worker.command = "start"; case.worker.run()
    assert order[2:6] == [
        "prefetch-dead", ("cleanup-retry", adapter), "detector-reset", "process"]
    assert case.worker._scan_session_adapter is case.worker._retained_scan_session_adapter is None
    assert slot.state.value == "released"
    assert "initialize" not in order and "mount" not in order
    assert case.worker.release_retained_custody() is True and slot.state.value == adapter._light_lease.state.value == "released"
    case.worker._scan_session_adapter = adapter
    monkeypatch.setattr(adapter, "finish", lambda **_k: SimpleNamespace(failed=True, error="settled writer fault"))
    assert case.worker._close_reduction_session() is False
    assert case.worker._scan_session_adapter is case.worker._retained_scan_session_adapter is None
    statuses = []
    relay = SimpleNamespace(thread=case.worker, _set_status_text=statuses.append,
        _set_action_button=lambda *_a: None,
        ui=SimpleNamespace(stopButton=SimpleNamespace(setEnabled=lambda *_a: None),
            liveCheckBox=SimpleNamespace(isChecked=lambda: False)), finished=None)
    relay.finished = SimpleNamespace(emit=lambda: imageWrangler.stop(relay))
    imageWrangler._on_worker_thread_finished(relay)
    assert statuses == ["Dynamic output failed: settled writer fault"]

def test_transient_later_read_settles_attempt_before_new_revision(tmp_path, monkeypatch):
    case = _c3_real_source_case(tmp_path, "external-eiger", live_mode=True)
    state = case.dependency.stat(); dependency_stamps = [(int(state.st_size), int(state.st_mtime_ns))]; normalized = lambda path: os.path.abspath(os.fspath(path)); source_path, dependency_path = map(normalized, (case.source, case.dependency))
    trace, sync_calls, read_gate, read_done = [], [], threading.Event(), threading.Event(); active = {"trace": trace}
    _c3_install_adapter_trace(monkeypatch, active); _c3_wrap_worker_source_seams(case, trace)
    traced_mount = case.worker._mount_dynamic_reduction_session
    def guarded_mount(*args, **kwargs):
        active["adapter_call"] = "mount"
        try:
            return traced_mount(*args, **kwargs)
        finally:
            active.pop("adapter_call", None)
    case.worker._mount_dynamic_reduction_session = guarded_mount; original_stat = os.stat
    def guarded_stat(path, *args, **kwargs):
        if normalized(path) in {source_path, dependency_path}:
            assert "adapter_call" not in active, active["adapter_call"]
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(os, "stat", guarded_stat)
    traced_read, traced_sync, misses = case.worker.get_next_image, case.worker._get_next_eiger_frame_sync, []
    def gated_sync(frozen):
        sync_calls.append(True)
        if len(sync_calls) == 2:
            path, candidate = case.worker._eiger_master_path, case.worker._eiger_master_candidate; case.worker._eiger_close_master(); assert read_gate.wait(5)
            case.worker._eiger_master_candidate = candidate; case.worker._eiger_open_master(frozen, path)
        result = traced_sync(frozen); read_done.set() if len(sync_calls) == 2 else None; return result
    case.worker._get_next_eiger_frame_sync = gated_sync
    def finite_read(frozen):
        item = traced_read(frozen)
        if item[3] is None:
            misses.append(True)
            if len(misses) >= 8: case.worker.command = "stop"
        else:
            misses.clear()
        return item
    case.worker.get_next_image = finite_read
    adapter_type, refusal = _adapter(), []; traced_submit, traced_failed = adapter_type.submit, adapter_type.record_failed
    monkeypatch.setattr(adapter_type, "record_failed", lambda owner, *a, **k: (traced_failed(owner, *a, **k), read_gate.set(), read_done.wait(5) or pytest.fail("later read did not finish"))[0])
    def refuse_once(owner, frame, *args, **kwargs):
        if not refusal:
            with h5py.File(case.dependency, "a") as handle:
                data = handle["entry/data/data"]; data.resize((2, *data.shape[1:])); data[1] = 2
            dependency = original_stat(case.dependency)
            dependency_stamps.append((int(dependency.st_size), int(dependency.st_mtime_ns)))
            master = original_stat(case.source)
            assert (int(master.st_size), int(master.st_mtime_ns)) == case.stamp
            refusal.append(kwargs.get("attempt_token"))
            return False
        return traced_submit(owner, frame, *args, **kwargs)
    monkeypatch.setattr(adapter_type, "submit", refuse_once)
    failure = None
    with _c3_worker_lifetime(case.worker):
        try: case.worker.process_scan(case.worker.run_configuration)
        except BaseException as caught: failure = caught; read_gate.set()
    assert failure is None, failure
    try:
        begins = [row for row in trace if row[0] == "adapter.begin_attempt.enter"]
        failed = [row for row in trace if row[0] == "adapter.record_failed.enter"]
        assert len(refusal) == len(failed) == 1 and len(begins) >= 3
        revisions = [int(row[3]["source_revision"]) for row in begins]
        assert revisions == [max(1, case.stamp[1])] * len(begins)
        assert failed[0][2][0] is refusal[0] and failed[0][3].get("retryable") is True
        assert trace.index(failed[0]) < next(i for i, row in enumerate(trace) if row is begins[1])
        extend = [i for i, row in enumerate(trace) if row[0] == "adapter.extend_live.return"]
        assert len(extend) == 1 and trace.index(failed[0]) < extend[0]
        extend = max(i for i in range(extend[0]) if trace[i][0] == "adapter.extend_live.enter"); intent = trace[extend][2][0]
        assert normalized(intent.source.path) == source_path and intent.source.adapter_id == "nexus_hdf5"
        assert (intent.source.size, intent.source.mtime_ns) == case.stamp
        members = [m for m in intent.source.external_members if normalized(m.path) == dependency_path]
        assert len(members) == 1
        assert (members[0].size, members[0].mtime_ns) == dependency_stamps[1]
        successor_begin = next(i for i in range(extend + 1, len(trace)) if trace[i][0] == "adapter.begin_attempt.enter")
        successor_submit = next(i for i in range(successor_begin + 1, len(trace)) if trace[i][0] == "adapter.submit.enter")
        assert extend < successor_begin < successor_submit
        queued = [value for row in trace if row[0] == "prefetch.queue" for value in row[2]]
        carried = {}
        for value in queued:
            carried.setdefault(normalized(value.path), set()).add(_c3_observation_stamp(value))
        assert dependency_stamps[0] != dependency_stamps[1] and carried[source_path] == {case.stamp}
        assert set(dependency_stamps).issubset(carried[dependency_path])
    finally:
        case.worker._close_reduction_session()

def test_delayed_prior_generation_event_cannot_mutate_settled_or_successor_run(
    tmp_path, monkeypatch,
):
    from xdart.modules.frame_publication import PublicationStore
    from xrd_tools.core import IntegrationResult1D
    from xrd_tools.session import FrameEvent
    case = _c3_real_source_case(tmp_path, "tiff")
    trace, adapters = _c3_run(case, monkeypatch, settle=True)
    try:
        assert len(adapters) == 1
        adapter = adapters[0]; observer, store = adapter._observer, adapter._publication_store
        live = next(row[2][0] for row in trace if row[0] == "adapter.submit.enter")
        token_a, token_b = object(), object()
        assert observer.register(live, token_a) is True
        calls = []
        original_publish = PublicationStore.publish_gui_light_1d
        def counted_publish(owner, publication, record):
            if owner is store:
                calls.append((publication, record))
            return original_publish(owner, publication, record)
        monkeypatch.setattr(PublicationStore, "publish_gui_light_1d", counted_publish)
        def snapshot():
            return (store.generation,
                    tuple((key, id(value)) for key, value in store._items.items()),
                    tuple((key, id(value)) for key, value in store._light_1d_items.items()))
        before = snapshot()
        npt = adapter._light_lease.layout.modes[0].coordinate.length
        result = IntegrationResult1D(np.linspace(0.0, 1.0, npt),
                                     np.ones(npt), unit="q_A^-1")
        def event(generation):
            return FrameEvent(int(live.idx), None, result, None, {}, generation, 1.0)
        assert observer.on_frame_completed(event(adapter._generation - 1)) is False
        assert snapshot() == before
        successor = copy.copy(live)
        session_calls = []
        monkeypatch.setattr(adapter._session, "submit", lambda *_a, **_k: session_calls.append(True) or True)
        assert adapter.submit(successor, attempt_token=token_b) is False
        assert session_calls == []
        settled = []
        adapter._accounting = SimpleNamespace(record_failed=lambda token, **facts: settled.append((token, facts)))
        adapter.record_failed(token_b, retryable=True)
        assert settled[0][0] is token_b
        assert observer.unregister(live, token_b) is False
        assert observer.on_frame_completed(event(adapter._generation)) is True
        assert len(calls) == 1 and live.int_1d is None
        assert observer.register(successor, token_b) is True
        assert observer.on_frame_completed(event(adapter._generation)) is True
        assert len(calls) == 2 and successor.int_1d is None
        assert observer.unregister(successor, token_b) is False
    finally:
        case.worker.command = "start"
        case.worker._close_reduction_session()
