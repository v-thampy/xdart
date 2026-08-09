"""Retained prepared-NeXus execution and exact-source ownership oracles.

C3 successor coverage owns the retired private Append/save/XYE shadow.  This
module keeps the still-current exact-admission, exact-entry binding, public
late-refusal, negative-owner-census, and immutable-target rows.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

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
from xdart.gui.tabs.static_scan.wranglers import (  # noqa: E402
    nexus_wrangler_thread as nwt,
)
from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (  # noqa: E402,E501
    FrozenSourceTarget,
    nexusThread,
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
    assert not hasattr(first, "scan")
    assert thread._active_scan is scan
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
