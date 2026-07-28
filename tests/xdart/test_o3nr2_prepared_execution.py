"""O-3N.R.2 — the ONE prepared NeXus execution envelope (handoff §17).

Frozen acceptance oracle for the §17.8 ratified structural correction, written
BEFORE any production edit and red at parent ``cbbd2e4d``.

§17 named one root-cause family: the O-3N.R.1 owner recorded
"prepared/replaced/cleared" identities *before* the corresponding resource or
transaction reached a successful terminal state, and it kept five parallel
latches (``_prepared_for``, ``_prepared_stack``, ``_run_output_prepared``,
``_run_target_replaced``, ``_xye_tail_cleared``) that no single object owned.
The correction is one run-owned :class:`PreparedNexusExecution` envelope.

Row map — the §17.8 required discriminator set:

1  foreign equal-valued direct ``_run_impl()`` refusal before source open  (§17.1)
2  exact-entry binding: qualification and stack are ONE resource            (§17.3)
3  stack cleanup on a pre-context exception and on incompatible Append      (§17.2)
4  a later worker refusal is CONTAINED, visible and returns to idle         (§17.2)
5  same URI/entry with a changed accepted PONI fingerprint refuses          (§17.4)
6  a failed Append qualification stays failed on an exact retry             (§17.4)
7  a fail-once replacement does not consume the replace-once identity       (§17.5)
8  a writer failure preserves the prior durable target byte-for-byte        (§17.5)
9  XYE carrier poisoning, and fail-once stale-tail deletion is retryable    (§17.6)
10 the superseded parallel latches and the dead helper are gone             (§17.8/7)

Production-wired (CLAUDE.md rule 2): the real ``staticWidget``, the real NeXus
page from the real wrangler stack, the real replacement ``nexusThread``, real
h5py fixtures, and the real writer/provenance reader.  The three unit rows that
drive the transaction algebra directly build a REAL ``PreparedNexusExecution``
over a REAL ``FrozenSourceTarget`` — the envelope is the object under test, so
it is never doubled.
"""

from __future__ import annotations

import os
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")
h5py = pytest.importorskip("h5py")

from tests.xdart.test_o3n_execution_owner import (  # noqa: E402,F401
    _arm_real,
    _select_nexus,
    _start_recorder,
    _started,
    _write_nexus,
    _write_poni,
    qapp,
    widget,
)
from tests.xdart.test_o3nr1_run_scan import (  # noqa: E402
    _fake_frame,
    _write_real_target,
)

from xdart.gui.tabs.static_scan.wranglers import (  # noqa: E402
    nexus_wrangler_thread as nwt,
)
from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (  # noqa: E402,E501
    FrozenSourceTarget,
    PreparedNexusExecution,
    nexusThread,
)
from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (  # noqa: E402
    wranglerThread,
)
from xrd_tools.io.nexus import (  # noqa: E402
    open_nexus_image_stack,
    open_nexus_image_stack_exact,
)
from xrd_tools.session.run_configuration import (  # noqa: E402
    RunConfigurationRefused,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _collect_source_handles(monkeypatch, source):
    """Collect every real ``h5py.File`` this process opens on *source*.

    Opener-name independent: it observes the actual OS handle, so it stays a
    true statement about leakage no matter which function opens the container.
    """
    wanted = os.path.abspath(str(source))
    handles: list = []
    real_file = h5py.File

    def collecting(name, mode="r", *args, **kwargs):
        handle = real_file(name, mode, *args, **kwargs)
        try:
            if os.path.abspath(str(name)) == wanted:
                handles.append(handle)
        except (TypeError, ValueError):       # non-path-like (in-memory driver)
            pass
        return handle

    monkeypatch.setattr(h5py, "File", collecting)
    return handles


def _envelope(frozen, output_path, **target_kw):
    """A REAL envelope over a REAL target — no double of the object under test."""
    target = FrozenSourceTarget(
        uri=str(target_kw.pop("uri", "")),
        entry=str(target_kw.pop("entry", "")),
        scan_name=str(target_kw.pop("scan_name", "scan")),
        output_path=str(output_path),
        **target_kw,
    )
    return PreparedNexusExecution(frozen, target)


# --------------------------------------------------------------------------- #
# 1 — §17.1: exact admission owns preparation; direct entry traverses it too
# --------------------------------------------------------------------------- #

def test_direct_impl_refuses_a_foreign_frozen_object_before_source_open(
        widget, tmp_path, monkeypatch):
    """The supported direct ``_run_impl(frozen)`` path accepted an equal-valued
    but non-identical object and opened its source (§17.1)."""
    wrangler = _select_nexus(widget)
    _src, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    accepted = thread.run_configuration
    foreign = replace(accepted)
    assert foreign == accepted and foreign is not accepted

    reached = []
    monkeypatch.setattr(
        nexusThread, "_preflight_execution_target",
        lambda _self, target: reached.append(target))

    with pytest.raises(RunConfigurationRefused):
        nexusThread._prepare_execution(thread, foreign)

    assert reached == [], "a foreign frozen object reached the source open"
    assert thread._execution is None


def test_preparation_is_one_owner_and_one_derivation(
        widget, tmp_path, monkeypatch):
    """§17.1: ``run()``/``_run_impl()``/``_initialize_scan()`` each derived the
    target and rebuilt a different PONI.  One envelope now answers for all."""
    wrangler = _select_nexus(widget)
    src, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    accepted = thread.run_configuration

    first = nexusThread._prepare_execution(thread, accepted)
    again = nexusThread._prepare_execution(thread, accepted)

    assert again is first, "preparation is not idempotent"
    assert first.frozen is accepted, "the envelope holds the exact admitted object"
    scan = thread._initialize_scan(Path(src).stem)
    assert thread._execution is first, "_initialize_scan re-derived the target"
    assert first.scan is scan
    assert scan.data_file == first.target.output_path
    assert thread.poni is first.poni, "a second PONI object reached execution"


# --------------------------------------------------------------------------- #
# 2 — §17.3: Group qualification and stack binding are ONE resource
# --------------------------------------------------------------------------- #

def test_the_exact_opener_refuses_where_the_shared_opener_falls_back(tmp_path):
    """The shared opener backstops a Dataset/missing hint with the first
    runnable NXentry.  The execution opener may never do that (§17.3)."""
    source = tmp_path / "dataset-hint-with-runnable-fallback.nxs"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("selected", data=[1])
        actual = handle.create_group("actual")
        actual.attrs["NX_class"] = "NXentry"
        detector = actual.create_group("instrument/detector")
        detector.create_dataset("data", data=np.zeros((2, 4, 4), dtype=np.float32))

    with open_nexus_image_stack(source, "selected") as lenient:
        assert lenient.paths == ("/actual/instrument/detector/data",)

    for hint in ("selected", "missing", ""):
        with pytest.raises((KeyError, ValueError)):
            open_nexus_image_stack_exact(source, hint)


def test_a_non_group_entry_refuses_precisely_with_no_fallback_to_shadow_it(
        widget, tmp_path, monkeypatch):
    """The Group check is not subsumed by the outside-the-group check.

    Mutation M-2b (delete the Group check) survived the first oracle, because a
    Dataset hint beside a runnable NXentry is still caught by the
    resolved-outside-the-selected-entry check.  It is NOT equivalent: with no
    other entry to fall back to, the dataset finder walks a Dataset as if it
    were a group and leaks whatever numpy/h5py raises — an internal
    ``ValueError`` about array truth values, or a ``TypeError`` the preflight
    does not even catch.  The Group check is what makes the refusal precise and
    typed, so it gets its own row.
    """
    source = tmp_path / "three-d-dataset-hint-alone.nxs"
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "selected", data=np.zeros((2, 4, 4), dtype=np.float32))

    with pytest.raises(KeyError, match="not an HDF5 group"):
        open_nexus_image_stack_exact(source, "selected")

    # And through the real worker preflight: one typed refusal, no output.
    wrangler = _select_nexus(widget)
    output = tmp_path / "out"
    output.mkdir()
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        _write_poni(tmp_path / "cal.poni"))
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(source))
    wrangler.parameters.child("NeXus File", "entry").setValue("selected")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(output))
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    target = nexusThread._frozen_source_target(thread.run_configuration)

    with pytest.raises(RunConfigurationRefused):
        nexusThread._preflight_execution_target(thread, target)
    assert list(output.iterdir()) == []


def test_the_exact_opener_binds_only_within_the_selected_group(tmp_path):
    source = tmp_path / "two-entry.nxs"
    with h5py.File(source, "w") as handle:
        for name, value in (("selected", 1.0), ("fallback", 9.0)):
            entry = handle.create_group(name)
            entry.attrs["NX_class"] = "NXentry"
            detector = entry.create_group("instrument/detector")
            detector.create_dataset(
                "data", data=np.full((1, 4, 4), value, dtype=np.float32))

    with open_nexus_image_stack_exact(source, "selected") as stack:
        assert all(path.startswith("/selected/") for path in stack.paths)
        assert float(np.asarray(stack[0]).ravel()[0]) == pytest.approx(1.0)


def test_preflight_proves_and_binds_the_source_in_one_open(
        widget, tmp_path, monkeypatch):
    """§17.3: the parent proved the Group with one open, CLOSED it, and opened
    the file again to bind the stack.  Replacing the selected Group between the
    two opens bound ``/fallback/...`` under ``selected``'s provenance.  One open
    makes that window structurally unreachable.

    O-3N.R.3 reshape (declared): §19.1 extended the strict prepared-source
    operation, so preflight now returns the execution-source PAIR — the same
    one open additionally detaches the scan metadata.  The row's fact is
    unchanged and strictly stronger: still exactly one open of the source.
    """
    wrangler = _select_nexus(widget)
    src, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    target = nexusThread._frozen_source_target(thread.run_configuration)
    opens = _collect_source_handles(monkeypatch, src)

    source = nexusThread._preflight_execution_target(thread, target)
    try:
        assert all(path.startswith(f"/{target.entry}/")
                   for path in source.stack.paths)
    finally:
        source.stack.close()

    assert len(opens) == 1, (
        f"qualification, binding and metadata used {len(opens)} separate "
        "opens of the source; they must describe the same resource")


# --------------------------------------------------------------------------- #
# 3 — §17.2: total worker lifetime for the proved raw stack
# --------------------------------------------------------------------------- #

def test_prepared_stack_is_closed_when_failure_precedes_context_entry(
        widget, tmp_path, monkeypatch):
    """The stack was opened during preparation but only entered a ``with``
    block after detector/mask/scan/Append work.  Anything failing in between
    leaked the open HDF5 handle (§17.2)."""
    wrangler = _select_nexus(widget)
    src, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    accepted = thread.run_configuration
    handles = _collect_source_handles(monkeypatch, src)

    class StopProbe(BaseException):
        pass

    monkeypatch.setattr(
        nexusThread, "_initialize_scan",
        lambda _self, _scan_name: (_ for _ in ()).throw(StopProbe()))

    with pytest.raises(StopProbe):
        nexusThread._run_impl(thread, accepted)

    assert handles, "the row never opened the source"
    assert all(not handle.id.valid for handle in handles), (
        "the proved raw-stack file is still open after a pre-context failure")
    assert thread._execution is None, "the lifecycle did not return to idle"


def test_incompatible_append_closes_the_proved_stack(
        widget, tmp_path, monkeypatch):
    """An Append refusal raised AFTER preparation must not leak the source."""
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Append")
    src, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    frozen = thread.run_configuration
    target = nexusThread._frozen_source_target(frozen)
    Path(target.output_path).write_bytes(b"not an admitted prior result")

    prepared = nexusThread._prepare_execution(thread, frozen)
    handle = prepared.stack._h5
    assert handle.id.valid

    with pytest.raises(RunConfigurationRefused):
        thread._initialize_scan(Path(src).stem)

    assert not handle.id.valid, "the incompatible Append leaked the raw stack"
    assert thread._execution is None


# --------------------------------------------------------------------------- #
# 4 — §17.2: a later worker refusal is contained, visible and idle-returning
# --------------------------------------------------------------------------- #

def test_public_worker_contains_a_late_append_refusal(
        widget, tmp_path, monkeypatch):
    """``run()``'s handler covered only the initial preparation call, so a
    production-shaped incompatible Append escaped the QThread entry (§17.2)."""
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Append")
    src, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    target = nexusThread._frozen_source_target(thread.run_configuration)
    prior = b"FOREIGN APPEND TARGET"
    Path(target.output_path).write_bytes(prior)
    messages: list[str] = []
    thread.showLabel.connect(messages.append)
    handles = _collect_source_handles(monkeypatch, src)

    # This is exactly what QThread invokes.  A worker-boundary refusal is an
    # expected typed terminal outcome, not an uncaught exception.
    thread.run()

    assert any("refused" in message.lower() for message in messages), messages
    assert Path(target.output_path).read_bytes() == prior
    assert thread._execution is None
    assert all(not handle.id.valid for handle in handles)
    assert thread.command == "stop", "the lifecycle did not return to idle"


# --------------------------------------------------------------------------- #
# 5 — §17.4: Append compares the accepted CONTENT identity
# --------------------------------------------------------------------------- #

def test_append_refuses_same_source_with_changed_frozen_fingerprint(
        widget, tmp_path, monkeypatch):
    """Same URI/entry plus a changed accepted PONI distance is a foreign
    content identity.  Source/entry/processing alone admitted it (§17.4)."""
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Append")
    source, _out, first_thread = _started(wrangler, tmp_path, monkeypatch)
    _write_real_target(first_thread, source, frames=(0,))
    first_fingerprint = first_thread.run_configuration.fingerprint

    # Finish run A through the production run-end owner, then change ONE
    # scientific value that ``processing_mapping()`` deliberately omits.
    widget._exit_run_state(widget._new_projection_receipt())
    second_poni = tmp_path / "second.poni"
    second_poni.write_text(
        (tmp_path / "cal.poni").read_text().replace(
            "Distance: 0.1", "Distance: 0.2"))
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        str(second_poni))
    wrangler.start()
    second_thread = wrangler.thread
    assert second_thread is not first_thread

    assert second_thread.run_configuration.fingerprint != first_fingerprint
    with pytest.raises(RunConfigurationRefused):
        second_thread._initialize_scan(Path(source).stem)


# --------------------------------------------------------------------------- #
# 6 — §17.4: the preparation identity is committed only after proof
# --------------------------------------------------------------------------- #

def test_failed_append_qualification_remains_failed_on_exact_retry(
        tmp_path, monkeypatch):
    """``_run_output_prepared`` was assigned before provenance was read, so an
    exact retry after a refusal returned early and bypassed qualification."""
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.session import RunIntent

    source = tmp_path / "source.nxs"
    source.write_bytes(b"source")
    frozen = RunIntent(
        output_mode="Append",
        save_path=str(tmp_path),
        source_spec=SourceSpec(source, SourceKind.NEXUS_STACK, entry="entry"),
    ).freeze()
    target = tmp_path / "target.nxs"
    target.write_bytes(b"prior")
    scan = SimpleNamespace(data_file=str(target))
    prepared = _envelope(frozen, target, uri=str(source), entry="entry")
    worker = SimpleNamespace(_execution=prepared, file_lock=threading.RLock())
    monkeypatch.setattr(
        nwt, "read_provenance", lambda _path: {"config": {"run_configuration": {}}})

    with pytest.raises(RunConfigurationRefused):
        nexusThread._prepare_output_for_run(worker, prepared, scan)
    assert prepared.append_qualified is False
    with pytest.raises(RunConfigurationRefused):
        nexusThread._prepare_output_for_run(worker, prepared, scan)
    assert target.read_bytes() == b"prior"


# --------------------------------------------------------------------------- #
# 7/8 — §17.5: Overwrite consumes the once-identity only after a commit
# --------------------------------------------------------------------------- #

def test_a_failed_replacement_does_not_consume_the_once_identity(
        widget, tmp_path, monkeypatch):
    """``_replace_target_once`` recorded the replacement before pause/unlink
    succeeded, so a fail-once replacement burned the retry identity (§17.5)."""
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    src, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    frozen = thread.run_configuration
    scan = thread._initialize_scan(Path(src).stem)
    prepared = thread._execution
    prior = b"prior-durable-result"
    Path(scan.data_file).write_bytes(prior)
    scan.add_frame(frame=_fake_frame(0), calculate=False, update=True,
                   get_sd=True, set_mg=False, static=True, batch_save=True)

    attempts: list[str] = []
    real_pool = nwt._get_h5pool

    def fail_once():
        pool = real_pool()

        def pause(path):
            attempts.append(str(path))
            if len(attempts) == 1:
                raise OSError("injected replacement failure")
            return pool.pause(path)

        return SimpleNamespace(pause=pause, resume=pool.resume)

    monkeypatch.setattr(nwt, "_get_h5pool", fail_once)

    with pytest.raises(OSError, match="injected replacement failure"):
        thread._save_to_disk(frozen, scan)

    assert Path(scan.data_file).read_bytes() == prior
    assert prepared.overwrite.committed is False, (
        "a failed replacement consumed the replace-once identity")

    thread._save_to_disk(frozen, scan)            # one exact retry, and it works

    assert Path(scan.data_file).read_bytes() != prior
    assert prepared.overwrite.committed is True
    assert len(attempts) >= 2


def test_writer_failure_preserves_the_prior_durable_target(
        widget, tmp_path, monkeypatch):
    """``_save_to_disk`` unlinked the prior target and THEN called the writer;
    an injected writer failure left no old target and no new result (§17.5)."""
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    src, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    first = _write_real_target(thread, src, frames=(0, 1))
    target = Path(first.data_file)
    prior = target.read_bytes()
    assert prior

    widget._exit_run_state(widget._new_projection_receipt())
    wrangler.start()
    thread_b = wrangler.thread
    assert thread_b is not thread
    scan_b = thread_b._initialize_scan(Path(src).stem)
    prepared = thread_b._execution
    scan_b.add_frame(frame=_fake_frame(0), calculate=False, update=True,
                     get_sd=True, set_mg=False, static=True, batch_save=True)

    monkeypatch.setattr(
        scan_b,
        "_save_to_nexus",
        lambda: (_ for _ in ()).throw(
            OSError("injected writer failure")),
    )
    with pytest.raises(OSError, match="injected writer failure"):
        thread_b._save_to_disk(thread_b.run_configuration, scan_b)

    assert target.read_bytes() == prior, (
        "a failed replacement writer destroyed the prior durable target")
    assert prepared.overwrite.committed is False
    assert sorted(p.name for p in target.parent.glob("*")
                  if p.suffix != ".nxs") == [], (
        "the rollback left staging debris beside the accepted target")

    monkeypatch.undo()                             # one exact retry, and it works
    thread_b._save_to_disk(thread_b.run_configuration, scan_b)
    assert target.read_bytes() != prior
    assert prepared.overwrite.committed is True


# --------------------------------------------------------------------------- #
# 9 — §17.6: the XYE tail transaction
# --------------------------------------------------------------------------- #

def test_xye_tail_policy_uses_the_envelope_not_the_mutable_carrier(
        tmp_path, monkeypatch):
    """Replacing ``self.run_configuration`` after admission with a foreign
    Append object suppressed Overwrite stale-tail cleanup (§17.6)."""
    root = tmp_path / "scan"
    root.mkdir()
    stale = root / "iq_scan_0009.xye"
    stale.write_text("stale")
    accepted = SimpleNamespace(output_mode="Overwrite")
    foreign = SimpleNamespace(output_mode="Append")
    prepared = _envelope(accepted, tmp_path / "scan.nxs")
    worker = SimpleNamespace(run_configuration=foreign, _execution=prepared)
    scan = SimpleNamespace(data_file=str(tmp_path / "scan.nxs"), name="scan")
    prepared.xye.stage(0, object())
    monkeypatch.setattr(
        wranglerThread, "_write_xye_entries",
        lambda self, scan, entries, **kwargs: None)

    nexusThread._flush_xye_buffer(worker, scan, published_idxs={0})

    assert not stale.exists()


def test_xye_tail_cleanup_retries_a_transient_unlink_failure(
        tmp_path, monkeypatch):
    """A swallowed delete failure consumed the once-only cleanup latch and left
    the stale higher-frame file permanently (§17.6)."""
    root = tmp_path / "scan"
    root.mkdir()
    stale = root / "iq_scan_0009.xye"
    stale.write_text("stale")
    frozen = SimpleNamespace(output_mode="Overwrite")
    prepared = _envelope(frozen, tmp_path / "scan.nxs")
    worker = SimpleNamespace(run_configuration=frozen, _execution=prepared)
    scan = SimpleNamespace(data_file=str(tmp_path / "scan.nxs"), name="scan")
    real_unlink = Path.unlink
    attempts: list[str] = []

    def fail_once(path, *args, **kwargs):
        if path == stale:
            attempts.append(str(path))
            if len(attempts) == 1:
                raise OSError("injected transient")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    prepared.xye.stage(0, object())
    monkeypatch.setattr(
        wranglerThread, "_write_xye_entries",
        lambda self, scan, entries, **kwargs: None)

    nexusThread._flush_xye_buffer(worker, scan, published_idxs={0})
    assert stale.exists(), "a failed deletion was reported as done"
    assert prepared.xye.tail_pending, "the pending cleanup was forgotten"

    nexusThread._flush_xye_buffer(worker, scan, published_idxs=set())

    assert not stale.exists()
    assert prepared.xye.tail_pending == []
    assert len(attempts) == 2


def test_xye_tail_discovery_never_deletes_this_run_s_own_output(
        tmp_path, monkeypatch):
    """The stale set is discovered ONCE, before this run writes; a retry may
    not sweep the files the first flush produced."""
    root = tmp_path / "scan"
    root.mkdir()
    stale = root / "iq_scan_0009.xye"
    stale.write_text("stale")
    frozen = SimpleNamespace(output_mode="Overwrite")
    prepared = _envelope(frozen, tmp_path / "scan.nxs")
    worker = SimpleNamespace(run_configuration=frozen, _execution=prepared)
    scan = SimpleNamespace(data_file=str(tmp_path / "scan.nxs"), name="scan")
    ours = root / "iq_scan_0000.xye"

    def write_ours(self, scan, entries, **_kwargs):
        ours.write_text("this run")

    monkeypatch.setattr(wranglerThread, "_write_xye_entries", write_ours)
    prepared.xye.stage(0, object())

    nexusThread._flush_xye_buffer(worker, scan, published_idxs={0})
    nexusThread._flush_xye_buffer(worker, scan, published_idxs=set())

    assert not stale.exists()
    assert ours.exists(), "the second flush swept this run's own output"


# --------------------------------------------------------------------------- #
# 10 — §17.8 item 7: the superseded parallel state is gone
# --------------------------------------------------------------------------- #

def test_the_superseded_parallel_latches_and_dead_helper_are_gone():
    """A bounded fact about the actual production owner graph (rule 9), not a
    taint analyzer: the five parallel latches and the dead ``_execution_poni``
    helper are deleted once the envelope owns those phases.

    Asserted over the AST's attribute/name/def references, NOT over raw module
    text — the class docstring names the deleted latches deliberately, because
    "what this object replaced, and why" is the durable record of the §17 root
    cause.  A guard that cannot tell prose from a reference would force that
    history out of the code.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(nwt))
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            referenced.add(node.name)

    superseded = {"_prepared_for", "_prepared_stack", "_run_output_prepared",
                  "_run_target_replaced", "_xye_tail_cleared",
                  "_execution_poni"}
    found = sorted(superseded & referenced)

    assert found == [], f"superseded parallel state survived: {found}"


def test_the_execution_target_holds_values_not_a_mutable_science_object():
    """§17.7: ``FrozenSourceTarget`` called itself immutable while embedding a
    mutable ``PONI``.  It carries values; each caller gets its own object."""
    values = {"dist": 0.1, "poni1": 0.02, "poni2": 0.02, "wavelength": 1e-10}
    target = FrozenSourceTarget(
        uri="/x.nxs", entry="entry", scan_name="x", output_path="/out/x.nxs",
        poni_values=tuple(values.items()))

    first, second = target.poni(), target.poni()

    assert first is not second, "two callers share one mutable PONI"
    assert first == second
    first.dist = 9.9
    assert target.poni().dist == pytest.approx(0.1), (
        "mutating one caller's PONI changed the frozen execution target")
