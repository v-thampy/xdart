"""R4-G — the cumulative deletion boundary of the accepted Slice-3/O-3 stack.

Frozen BEFORE the R4-G production sweep (parent `ff5380a7`).  The rows below
are the acceptance oracle for six measured deletions and, in the same module,
the retained-behaviour contracts each deletion must not disturb.

Deletion rows (RED at the parent, GREEN at the R4-G tip):

* **G1** ``_ControlsStrictWriteError.unrestored`` and its structural twin
  ``.cause`` — zero production reads at the accepted tip.  The single raise
  site seeds both; the single catch site reads only ``reason``.  Ledgered for
  this sweep by review §29.4 / §30.5 row 2 and the deferred-ledger arc-residual
  rows.
* **G2** the R.5 dead test scaffolding (``MappingProxyType`` import,
  ``_find_live_registry``, ``_one_shot_unblock``) — ledgered by §30.5 row 1.
* **G3** the 22 retired worker constructor parameters that W-1R-D emptied of
  their slots (17 on ``imageThread``, 5 on ``nexusThread``) plus the dead
  ``_DEFAULT_MAX_CORES``.  Review §42 deferred the API retirement itself to
  R4-G; every value these once seeded is now read only from the accepted
  ``FrozenRunConfiguration``.
* **G4** the ``scan_args`` legacy thread carrier.  The wrangler widget's slot
  is LIVE (populated at admission, read at the readiness/metadata seam), but
  the constructor threaded an always-empty snapshot into a parameter no worker
  ever stored.
* **G5** ``displayFrameWidget._wrangler`` — a dead cross-widget reference whose
  mask/threshold consumer is gone, and which retained a wrangler (and through
  it a worker thread) on the display widget.
* **G6** ``displayFrameWidget._wavelength_run_scan_key`` — a duplicate of the
  live ``_run_wavelength_scan_key`` authority with zero readers.

Retained-behaviour rows are GREEN at the parent as well: they are the mutation
targets that prove each deletion was not load-bearing.

Production-wired: a real ``staticWidget``, real pyqtgraph ``Parameter``
handles, the real staging/commit engine, and the real production wranglers.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "src"
STATIC_SCAN = SRC / "xdart" / "gui" / "tabs" / "static_scan"
WRANGLERS = STATIC_SCAN / "wranglers"

#: The exact names W-1R-D emptied of their worker slots (review §44.2).
RETIRED_IMAGE_PARAMS = (
    "h5_dir", "single_img", "inp_type", "img_dir", "include_subdir", "img_ext",
    "series_average", "meta_ext", "file_filter", "mask_file", "write_mode",
    "gi", "th_mtr", "sample_orientation", "tilt_angle", "live_mode",
    "max_cores",
)
RETIRED_NEXUS_PARAMS = (
    "mask_file", "gi", "th_mtr", "sample_orientation", "tilt_angle",
)


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


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _param_names(func: ast.FunctionDef) -> list[str]:
    args = func.args
    out: list[str] = []
    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        out += [a.arg for a in group]
    if args.vararg:
        out.append(args.vararg.arg)
    if args.kwarg:
        out.append(args.kwarg.arg)
    return [n for n in out if n != "self"]


def _init_of(path: Path, cls: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                    return sub
    raise AssertionError(f"{cls}.__init__ not found in {path}")


def _attribute_stores(root: Path, name: str) -> list[str]:
    """Every ``<expr>.name = ...`` site under *root* (a deletion guard, not a
    taint analysis — rule 9)."""
    sites: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = _tree(path)
        except SyntaxError:  # pragma: no cover - source tree is importable
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr == name
                    and isinstance(node.ctx, ast.Store)):
                sites.append(f"{path.relative_to(SRC)}:{node.lineno}")
    return sites


# ---------------------------------------------------------------------------
# G1 — the dead strict-write failure fields
# ---------------------------------------------------------------------------

def test_g1_strict_write_error_carries_only_its_reason():
    """The typed strict-write failure exposes the forward diagnostic and
    nothing else.  ``unrestored`` and ``cause`` were seeded at the single raise
    site and read by no shipped consumer; the transaction's one signal registry
    has owned stranded-owner recovery since §25.4."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _ControlsStrictWriteError,
    )

    assert _param_names(
        _init_of(STATIC_SCAN / "static_scan_widget.py",
                 "_ControlsStrictWriteError")) == ["reason"]

    exc = _ControlsStrictWriteError("boom")
    assert exc.reason == "boom"
    assert str(exc) == "boom"
    assert not hasattr(exc, "unrestored")
    assert not hasattr(exc, "cause")


def test_g1_the_single_raise_site_seeds_no_dead_field():
    """Exactly one production raise site exists and it passes only a reason."""
    raises = [
        node
        for node in ast.walk(_tree(STATIC_SCAN / "static_scan_widget.py"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_ControlsStrictWriteError"
    ]
    assert len(raises) == 1, [ast.dump(r) for r in raises]
    assert len(raises[0].args) == 1
    assert not raises[0].keywords


def test_g1_no_production_text_still_promises_the_deleted_field():
    """Documentation debt travels with its subject: no docstring may still
    tell a reader that a stranded owner 'rides out on' the deleted field."""
    text = (STATIC_SCAN / "static_scan_widget.py").read_text(encoding="utf-8")
    assert "_ControlsStrictWriteError.unrestored" not in text
    assert "``unrestored``" not in text


# ---------------------------------------------------------------------------
# G3 / G4 — the retired worker constructor API
# ---------------------------------------------------------------------------

def test_g3_image_worker_accepts_no_retired_policy_parameter():
    names = set(_param_names(
        _init_of(WRANGLERS / "image_wrangler_thread.py", "imageThread")))
    assert names.isdisjoint(RETIRED_IMAGE_PARAMS), sorted(
        names & set(RETIRED_IMAGE_PARAMS))


def test_g3_nexus_worker_accepts_no_retired_policy_parameter():
    names = set(_param_names(
        _init_of(WRANGLERS / "nexus_wrangler_thread.py", "nexusThread")))
    assert names.isdisjoint(RETIRED_NEXUS_PARAMS), sorted(
        names & set(RETIRED_NEXUS_PARAMS))


def test_g3_dead_default_core_count_is_absent():
    from xdart.gui.tabs.static_scan.wranglers import nexus_wrangler_thread

    assert not hasattr(nexus_wrangler_thread, "_DEFAULT_MAX_CORES")


def test_g3_no_stale_prose_promises_a_worker_core_cache():
    """Three comment blocks promised the Cores selection was 'pushed down' to
    ``self.max_cores`` on the worker.  That slot is gone; the value is read
    from the accepted frozen configuration."""
    for path in (WRANGLERS / "nexus_wrangler.py",
                 WRANGLERS / "nexus_wrangler_thread.py"):
        text = path.read_text(encoding="utf-8")
        assert "self.max_cores" not in text, path


def test_g4_no_worker_constructor_takes_the_scan_args_carrier():
    for path, cls in (
            (WRANGLERS / "wrangler_widget.py", "wranglerThread"),
            (WRANGLERS / "image_wrangler_thread.py", "imageThread"),
            (WRANGLERS / "nexus_wrangler_thread.py", "nexusThread"),
    ):
        assert "scan_args" not in _param_names(_init_of(path, cls)), cls


def test_g4_no_production_construction_passes_a_retired_argument():
    """Every in-tree worker construction is keyword-clean of the retired names
    and positionally consistent with the reduced signatures."""
    retired = set(RETIRED_IMAGE_PARAMS) | set(RETIRED_NEXUS_PARAMS) | {
        "scan_args"}
    expected = {
        "imageThread": len(_param_names(
            _init_of(WRANGLERS / "image_wrangler_thread.py", "imageThread"))),
        "nexusThread": len(_param_names(
            _init_of(WRANGLERS / "nexus_wrangler_thread.py", "nexusThread"))),
        "wranglerThread": len(_param_names(
            _init_of(WRANGLERS / "wrangler_widget.py", "wranglerThread"))),
    }
    seen = 0
    for path in sorted((SRC / "xdart").rglob("*.py")):
        for node in ast.walk(_tree(path)):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in expected):
                continue
            # the class statement itself is a ClassDef, not a Call
            seen += 1
            where = f"{path.relative_to(SRC)}:{node.lineno}"
            assert retired.isdisjoint(
                {k.arg for k in node.keywords if k.arg}), where
            assert len(node.args) <= expected[node.func.id], where
    assert seen >= 5, f"expected every production construction, saw {seen}"


# ---------------------------------------------------------------------------
# G5 / G6 — the dead display-widget slots
# ---------------------------------------------------------------------------

def test_g5_display_widget_retains_no_dead_wrangler_reference(widget):
    """The display widget must not hold a wrangler — and through it a live
    worker thread — for a mask/threshold consumer that no longer exists."""
    assert _attribute_stores(SRC, "_wrangler") == []
    assert not hasattr(widget.displayframe, "_wrangler")


def test_g6_display_widget_keeps_one_wavelength_scan_key(widget):
    """``_run_wavelength_scan_key`` is the authority the finalize path reads;
    the parallel ``_wavelength_run_scan_key`` had no reader at all."""
    assert _attribute_stores(SRC, "_wavelength_run_scan_key") == []
    frame = widget.displayframe
    assert not hasattr(frame, "_wavelength_run_scan_key")
    assert hasattr(frame, "_run_wavelength_scan_key")
    assert hasattr(frame, "_wavelength_run_scan")


# ---------------------------------------------------------------------------
# Retained behaviour — the mutation targets
# ---------------------------------------------------------------------------

def test_retained_strict_write_failure_still_aborts_with_the_setter_reason(
        widget, monkeypatch):
    """G1 deletes reporting slots, never the failure contract: a legacy-carrier
    setter that raises still fails the commit at ``legacy_apply`` with the
    setter as the primary reason and the carrier rolled back."""
    mask = ("Signal", "mask_file")
    original = widget._controls_v2_param(mask)
    prior = original.value()
    staged = widget.stage_controls_transaction(
        [(mask, "/tmp/r4g-strict-write.edf")])

    real_set = type(original).setValue

    def boom(self, value, *args, **kwargs):
        if self is original:
            raise RuntimeError("injected setter failure")
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(original), "setValue", boom)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert result.failed_path == mask
    assert "injected setter failure" in result.reason
    monkeypatch.undo()
    assert widget._controls_v2_param(mask).value() == prior


def test_retained_wrangler_scan_args_slot_is_still_the_live_carrier(widget):
    """G4 deletes the constructor thread-through, not the widget slot: the
    frozen configuration still publishes ``scan_args`` onto the wrangler, and
    the readiness/metadata seam still reads it."""
    wrangler = widget.wrangler
    assert hasattr(wrangler, "scan_args")
    wrangler.scan_args = {"bai_1d_args": {"numpoints": 4242}}
    assert wrangler.scan_args["bai_1d_args"]["numpoints"] == 4242

    source = inspect.getsource(type(widget))
    assert 'getattr(wrangler, "scan_args", None)' in source


def test_retained_workers_still_construct_through_their_wranglers(widget):
    """Both production wranglers still own a constructed worker after the
    signature reduction — the reduced call sites are real, not theoretical."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )

    assert isinstance(widget.wrangler.thread, imageThread)
    assert widget.wrangler.thread.run_configuration is None
    assert widget.wrangler.thread._admitted_run_configuration is None
    # the reduced signature keeps every slot the worker actually owns
    for slot in ("scan_name", "poni", "img_file", "command", "scan",
                 "source_base", "run_configuration_floor"):
        assert hasattr(widget.wrangler.thread, slot), slot


def test_retained_worker_policy_comes_only_from_the_frozen_configuration():
    """G3's premise, stated as a contract: the worker owns no ``live_mode`` or
    ``max_cores`` slot at all — it neither writes nor reads one — so retiring
    the constructor parameters cannot change what a run executes.  Both values
    reach execution only as ``frozen.live_mode`` / ``frozen.max_cores``.

    Scoped to the two worker modules on purpose: the wrangler widget and the
    frozen value model legitimately own attributes of the same NAME, and a
    tree-wide ban would assert something production never promised.
    """
    for path in (WRANGLERS / "image_wrangler_thread.py",
                 WRANGLERS / "nexus_wrangler_thread.py"):
        selfish = {
            f"{node.attr}:{node.lineno}"
            for node in ast.walk(_tree(path))
            if isinstance(node, ast.Attribute)
            and node.attr in {"live_mode", "max_cores"}
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }
        assert selfish == set(), f"{path.name}: {sorted(selfish)}"

    frozen_reads = 0
    for path in (WRANGLERS / "image_wrangler_thread.py",
                 WRANGLERS / "nexus_wrangler_thread.py"):
        for node in ast.walk(_tree(path)):
            if (isinstance(node, ast.Attribute)
                    and node.attr in {"live_mode", "max_cores"}
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "frozen"):
                frozen_reads += 1
    assert frozen_reads >= 10, frozen_reads


def test_retained_wavelength_stamp_stays_scan_qualified(widget):
    """G6 removes a duplicate key, not the fail-closed rule: a run-end stamp
    whose cached scan identity does not match the finalized key writes
    nothing."""
    frame = widget.displayframe

    class _Scan:
        name = "r4g-scan"
        mg_args: dict = {}
        _persisted_wavelength_m = None

    scan = _Scan()
    frame.begin_processing(scan, "r4g-scan")
    frame._run_wavelength_m = 1.2345e-10
    frame._run_wavelength_scan_key = "a-different-scan"

    frame.finish_processing(scan, "r4g-scan")

    assert scan._persisted_wavelength_m is None
    assert frame._run_wavelength_m is None
    assert frame._run_wavelength_scan_key is None
