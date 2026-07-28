"""O-3N.D — ratified prepared-source and output-owner design (§21).

Frozen before the O-3N design correction.  These rows observe durable
resources, transaction isolation, and public worker outcomes rather than
private predecessor latch names.
"""

from __future__ import annotations

import ast
import inspect
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")
h5py = pytest.importorskip("h5py")

from tests.xdart.test_o3n_execution_owner import (  # noqa: E402,F401
    _select_nexus,
    _start_recorder,
    qapp,
    widget,
)
from tests.xdart.test_o3nr2_prepared_execution import _envelope  # noqa: E402
from tests.xdart.test_o3nr3_execution_transaction import (  # noqa: E402
    _arm_pilatus,
)
from xdart.gui.tabs.static_scan.wranglers import (  # noqa: E402
    nexus_wrangler_thread as nwt,
)
from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (  # noqa: E402
    FrozenSourceTarget,
    nexusThread,
)
from xdart.utils.h5pool import H5FilePool  # noqa: E402
from xrd_tools.io.nexus import open_nexus_execution_source  # noqa: E402


class _Signal:
    def __init__(self):
        self.values: list[str] = []

    def emit(self, value):
        self.values.append(str(value))


def _dataset_ids(stack):
    """The exact dataset IDs production already owns; creates no file handle."""
    return tuple(stack._dsets)


def _write_internal_source(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        detector = entry.create_group("instrument/detector")
        detector.create_dataset(
            "data", data=np.arange(32, dtype=np.uint16).reshape(2, 4, 4))
    return path


def _write_external_source(
    root: Path,
    *,
    same_file: bool,
) -> Path:
    master_path = root / "scan_master.h5"
    target_paths = ([root / "scan_data.h5"] * 2 if same_file else
                    [root / "scan_data_1.h5", root / "scan_data_2.h5"])
    dataset_paths = [
        "/entry/data/data_a",
        "/entry/data/data_b",
    ]
    for index, target_path in enumerate(dict.fromkeys(target_paths)):
        with h5py.File(target_path, "w") as target:
            data = target.create_group("entry/data")
            if same_file:
                data.create_dataset(
                    "data_a",
                    data=np.full((1, 4, 4), 10, dtype=np.uint16),
                )
                data.create_dataset(
                    "data_b",
                    data=np.full((1, 4, 4), 20, dtype=np.uint16),
                )
            else:
                name = f"data_{'a' if index == 0 else 'b'}"
                data.create_dataset(
                    name,
                    data=np.full((1, 4, 4), 10 + index, dtype=np.uint16),
                )
    with h5py.File(master_path, "w") as master:
        entry = master.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        links = entry.create_group("data")
        for index, (target_path, dataset_path) in enumerate(
                zip(target_paths, dataset_paths, strict=True), start=1):
            links[f"data_{index:06d}"] = h5py.ExternalLink(
                str(target_path),
                dataset_path,
            )
    return master_path


@pytest.mark.parametrize("layout", ["internal", "same-file", "different-files"])
def test_prepared_source_close_releases_every_owned_dataset_id(
    tmp_path: Path,
    layout: str,
) -> None:
    if layout == "internal":
        path = _write_internal_source(tmp_path / "internal.nxs")
    else:
        path = _write_external_source(
            tmp_path,
            same_file=(layout == "same-file"),
        )
    source = open_nexus_execution_source(path, "entry")
    stack = source.stack
    datasets = _dataset_ids(stack)
    assert datasets and all(dataset.id.valid for dataset in datasets)
    assert int(np.asarray(stack[0]).flat[0]) >= 0

    stack.close()

    assert all(not dataset.id.valid for dataset in datasets)
    assert stack._dsets == []
    assert stack._h5 is None
    with pytest.raises(Exception):
        _ = datasets[0][()]
    stack.close()  # repeat close is inert


def test_failed_source_close_retains_exact_cleanup_owner_for_retry() -> None:
    calls: list[str] = []

    class _Stack:
        def close(self):
            calls.append("close")
            if len(calls) == 1:
                raise OSError("fail once")

    prepared = SimpleNamespace(
        xye=SimpleNamespace(retire=lambda: 0),
        close=_Stack().close,
    )
    worker = SimpleNamespace(_execution=prepared)

    with pytest.raises(OSError, match="fail once"):
        nexusThread._release_execution(worker)
    assert worker._execution is prepared

    nexusThread._release_execution(worker)
    assert worker._execution is None
    assert calls == ["close", "close"]


def test_failed_adoption_close_retains_non_executable_cleanup_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = SimpleNamespace(output_mode="Overwrite", generation=7)
    target = FrozenSourceTarget(
        uri=str(tmp_path / "source.nxs"),
        entry="entry",
        scan_name="scan",
        output_path=str(tmp_path / "output.nxs"),
    )
    closes: list[str] = []

    class _Stack:
        def close(self):
            closes.append("close")
            if len(closes) == 1:
                raise OSError("cleanup failed once")

    source = SimpleNamespace(stack=_Stack(), scan_metadata=None)
    worker = SimpleNamespace(
        _execution=None,
        file_lock=threading.RLock(),
        showLabel=_Signal(),
    )
    monkeypatch.setattr(
        nexusThread,
        "_require_run_configuration",
        lambda _self, _stage, candidate=None: candidate,
    )
    monkeypatch.setattr(
        nexusThread, "_frozen_source_target", lambda _frozen: target)
    monkeypatch.setattr(
        nexusThread, "_preflight_execution_target",
        lambda _self, _target: source,
    )
    monkeypatch.setattr(
        nexusThread, "_adopt_frozen_source_target",
        lambda _self, _prepared: (_ for _ in ()).throw(
            RuntimeError("adoption failed")),
    )

    with pytest.raises(OSError, match="cleanup failed once"):
        nexusThread._prepare_execution(worker, frozen)

    prepared = worker._execution
    assert prepared is not None
    assert not prepared.adopted
    with pytest.raises(Exception, match="no prepared execution"):
        nexusThread._require_execution(worker, frozen)

    nexusThread._release_execution(worker)

    assert closes == ["close", "close"]
    assert worker._execution is None


def _overwrite_host(target: Path):
    frozen = SimpleNamespace(output_mode="Overwrite")
    prepared = _envelope(frozen, target)
    worker = SimpleNamespace(
        file_lock=threading.RLock(),
        showLabel=_Signal(),
    )
    scan = SimpleNamespace(data_file=str(target))
    return prepared, worker, scan


def test_failed_first_overwrite_removes_partial_when_no_prior_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "new-result.nxs"
    prepared, worker, scan = _overwrite_host(target)
    pool = SimpleNamespace(pause=lambda _path: None, resume=lambda _path: None)
    monkeypatch.setattr(nwt, "_get_h5pool", lambda: pool)

    def fail_writer():
        target.write_bytes(b"partial")
        raise OSError("writer failed")

    with pytest.raises(OSError, match="writer failed"):
        nexusThread._write_run_result(worker, prepared, scan, fail_writer)

    assert not target.exists()
    assert prepared.overwrite.ready_for_retry


def test_failed_no_prior_cleanup_is_retained_and_retried_on_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "partial-result.nxs"
    prepared, worker, scan = _overwrite_host(target)
    worker._execution = prepared
    pool = SimpleNamespace(pause=lambda _path: None, resume=lambda _path: None)
    monkeypatch.setattr(nwt, "_get_h5pool", lambda: pool)
    real_unlink = Path.unlink
    attempts: list[str] = []
    writes: list[str] = []

    def fail_once(path, *args, **kwargs):
        if Path(path) == target and not attempts:
            attempts.append("failed")
            raise OSError("unlink failed once")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)

    def fail_writer():
        writes.append("write")
        target.write_bytes(b"partial")
        raise OSError("writer failed")

    with pytest.raises(OSError, match="writer failed"):
        nexusThread._write_run_result(worker, prepared, scan, fail_writer)

    assert target.read_bytes() == b"partial"
    assert prepared.overwrite.phase is nwt.OverwritePhase.ROLLBACK_PENDING

    nexusThread._release_execution(worker)

    assert writes == ["write"]
    assert not target.exists()
    assert prepared.overwrite.ready_for_retry
    assert worker._execution is None


def test_failed_pool_resume_is_retained_without_replaying_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "resume-result.nxs"
    prepared, worker, scan = _overwrite_host(target)
    worker._execution = prepared
    resumes: list[str] = []
    writes: list[str] = []

    class _Pool:
        @staticmethod
        def pause(_path):
            return None

        @staticmethod
        def resume(_path):
            resumes.append("resume")
            if len(resumes) == 1:
                raise OSError("resume failed once")

    pool = _Pool()
    monkeypatch.setattr(nwt, "_get_h5pool", lambda: pool)

    def writer():
        writes.append("write")
        target.write_bytes(b"complete")

    with pytest.raises(OSError, match="resume failed once"):
        nexusThread._write_run_result(worker, prepared, scan, writer)

    assert target.read_bytes() == b"complete"
    assert prepared.overwrite.committed
    assert writes == ["write"]

    nexusThread._release_execution(worker)

    assert resumes == ["resume", "resume"]
    assert writes == ["write"]
    assert worker._execution is None


def test_output_cleanup_failure_still_closes_source_and_retains_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "cleanup-result.nxs"
    prepared, worker, _scan = _overwrite_host(target)
    worker._execution = prepared
    backup = prepared.overwrite.backup
    backup.write_bytes(b"superseded prior")
    prepared.overwrite.phase = nwt.OverwritePhase.CLEANUP_PENDING
    closes: list[str] = []

    class _Stack:
        @staticmethod
        def close():
            closes.append("close")

    prepared.stack = _Stack()
    pool = SimpleNamespace(pause=lambda _path: None, resume=lambda _path: None)
    monkeypatch.setattr(nwt, "_get_h5pool", lambda: pool)
    real_unlink = Path.unlink

    def refuse_backup(path, *args, **kwargs):
        if Path(path) == backup:
            raise OSError("backup cleanup blocked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_backup)

    with pytest.raises(OSError, match="backup cleanup blocked"):
        nexusThread._release_execution(worker)

    assert closes == ["close"]
    assert prepared._closed
    assert worker._execution is prepared
    assert backup.exists()

    monkeypatch.setattr(Path, "unlink", real_unlink)
    nexusThread._release_execution(worker)

    assert closes == ["close"]
    assert worker._execution is None
    assert not backup.exists()


def test_unowned_backup_refuses_before_absent_target_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "crash-window.nxs"
    backup = target.with_name(target.name + nwt._REPLACING_SUFFIX)
    backup.write_bytes(b"only durable prior")
    prepared, worker, scan = _overwrite_host(target)
    pool = SimpleNamespace(pause=lambda _path: None, resume=lambda _path: None)
    monkeypatch.setattr(nwt, "_get_h5pool", lambda: pool)
    writes: list[str] = []

    with pytest.raises(OSError, match="unresolved prior-result backup"):
        nexusThread._write_run_result(
            worker,
            prepared,
            scan,
            lambda: writes.append("write"),
        )

    assert writes == []
    assert not target.exists()
    assert backup.read_bytes() == b"only durable prior"


def test_overwrite_transaction_holds_lock_and_pool_through_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "scan.nxs"
    with h5py.File(target, "w") as handle:
        handle.create_dataset("prior", data=[1])
    prepared, worker, scan = _overwrite_host(target)
    shared_lock = threading.RLock()
    worker.file_lock = shared_lock
    pool = H5FilePool(max_open=2)
    monkeypatch.setattr(nwt, "_get_h5pool", lambda: pool)
    assert int(pool.get(target)["prior"][0]) == 1

    writer_entered = threading.Event()
    probe_done = threading.Event()
    observations: list[tuple[bool, bool, str]] = []

    def reader():
        assert writer_entered.wait(2)
        acquired = shared_lock.acquire(blocking=False)
        try:
            if not acquired:
                observations.append((False, target.exists(), "blocked"))
                return
            try:
                pooled = pool.get(target)
            except (OSError, FileNotFoundError) as exc:
                observations.append((True, target.exists(), type(exc).__name__))
            else:
                observations.append((
                    True,
                    target.exists(),
                    "none" if pooled is None else "opened",
                ))
        finally:
            if acquired:
                shared_lock.release()
            probe_done.set()

    thread = threading.Thread(target=reader)
    thread.start()

    def writer():
        writer_entered.set()
        assert probe_done.wait(2)
        assert pool.get(target) is None
        with h5py.File(target, "w") as handle:
            handle.create_dataset("new", data=[2])

    nexusThread._write_run_result(worker, prepared, scan, writer)
    thread.join(2)
    try:
        assert observations == [(False, False, "blocked")]
        assert int(pool.get(target)["new"][0]) == 2
    finally:
        pool.close_all()


def test_overwrite_target_is_the_accepted_frozen_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = tmp_path / "accepted.nxs"
    foreign = tmp_path / "foreign.nxs"
    accepted.write_bytes(b"accepted prior")
    foreign.write_bytes(b"foreign prior")
    prepared, worker, scan = _overwrite_host(accepted)
    scan.data_file = str(foreign)
    pool = SimpleNamespace(pause=lambda _path: None, resume=lambda _path: None)
    monkeypatch.setattr(nwt, "_get_h5pool", lambda: pool)

    with pytest.raises(Exception, match="accepted output"):
        nexusThread._write_run_result(
            worker,
            prepared,
            scan,
            lambda: foreign.write_bytes(b"wrong"),
        )

    assert accepted.read_bytes() == b"accepted prior"
    assert foreign.read_bytes() == b"foreign prior"


def test_empty_nexus_xye_flush_never_discovers_or_sweeps_prior_tail(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scan"
    root.mkdir()
    prior = root / "iq_scan_0009.xye"
    prior.write_text("prior durable run")
    frozen = SimpleNamespace(output_mode="Overwrite")
    prepared = _envelope(frozen, tmp_path / "scan.nxs")
    worker = SimpleNamespace(_execution=prepared)
    scan = SimpleNamespace(data_file=str(tmp_path / "scan.nxs"), name="scan")

    nexusThread._flush_xye_buffer(worker, scan, published_idxs=set())

    assert prior.exists()
    assert prepared.xye.tail_pending is None


def test_release_retires_preflush_xye_entries_even_if_source_close_fails() -> None:
    calls: list[str] = []

    class _Stack:
        def close(self):
            calls.append("close")
            raise OSError("source cleanup failed")

    prepared = SimpleNamespace(
        xye=SimpleNamespace(
            entries=[(7, object())],
            retire=lambda: prepared.xye.entries.clear(),
        ),
        close=_Stack().close,
    )
    worker = SimpleNamespace(_execution=prepared)

    with pytest.raises(OSError, match="source cleanup failed"):
        nexusThread._release_execution(worker)

    assert prepared.xye.entries == []
    assert worker._execution is prepared


def test_real_empty_result_run_preserves_prior_xye_tail(
    widget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    source, output = _arm_pilatus(wrangler, tmp_path)
    root = output / Path(source).stem
    root.mkdir(parents=True)
    prior = root / f"iq_{Path(source).stem}_0009.xye"
    prior.write_text("prior durable run")
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    monkeypatch.setattr(
        nexusThread,
        "_get_reduction_session",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(nwt, "reduce_live_frames", lambda *_a, **_k: [])

    thread.run()

    assert prior.exists()


def test_real_publish_failure_retires_run_owned_xye_entries(
    widget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    _arm_pilatus(wrangler, tmp_path)
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    monkeypatch.setattr(
        nexusThread,
        "_get_reduction_session",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        nwt,
        "reduce_live_frames",
        lambda frames, *_a, **_k: list(frames),
    )
    captured = []

    class _PublishFault(RuntimeError):
        pass

    def fail_publish(self, *_args, **_kwargs):
        captured.append(self._execution.xye)
        raise _PublishFault

    monkeypatch.setattr(nexusThread, "_publish", fail_publish)

    with pytest.raises(_PublishFault):
        thread.run()

    assert captured
    assert captured[0].entries == []
    assert thread._execution is None


def test_nexus_execution_has_no_worker_global_xye_buffer_reference() -> None:
    tree = ast.parse(inspect.getsource(nwt))
    offenders = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and node.attr == "_xye_buffer"
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            offenders.append(node.lineno)
    assert offenders == []
