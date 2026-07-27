"""O-1a-W1R-D3 — the bridge is GONE and every policy read is explicitly routed.

Simple facts about the ACTUAL production tree and owner graph (CLAUDE.md rule 9
— explicitly NOT a general Python taint analyzer).  This replaces the W-1R
descriptor-choreography census (``test_w1r_execution_read_census.py``): that file
asserted the projection bridge behaved correctly, and D3 deletes the bridge, so
the successor asserts the stronger property the routing establishes:

1. the bridge's symbols do not exist anywhere in the production tree —
   ``FrozenRunProjection``, ``FROZEN_RUN_PROJECTIONS``,
   ``install_frozen_run_projections``, the ``_UNSET`` policy sentinel, the
   ``frozen_run_policy``/``_frozen_run_policy`` accessors and the retired source
   wrappers;
2. NO worker class carries any of the 27 retired policy names — not as a
   descriptor, not as a constructor slot, not as an instance attribute — so a
   read of one cannot silently resolve to a mutable mirror;
3. no worker method READS a retired name off ``self``, and no host reads one off
   a worker: every execution read takes the accepted configuration as an
   explicit argument;
4. no execution helper re-reads a policy HOLDER (``run_configuration``,
   ``_qualified_run_configuration``) after worker entry — only the three named
   entry gates consult the carrier;
5. every worker scope that consumes policy declares ``frozen`` in its signature.

This file is Qt-light: it imports the worker classes but constructs no widget.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from xdart.gui.tabs.static_scan.wranglers import (
    image_wrangler_thread,
    nexus_wrangler_thread,
    wrangler_widget,
)
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import nexusThread
from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread

#: The 27 run-owned policy names the projection bridge used to carry, and where
#: execution reads each of them now.  This is the frozen retirement ledger.
RETIRED_POLICY_NAMES = {
    # output
    "write_mode": "frozen.output_mode",
    "h5_dir": "frozen.save_path",
    "project_folder": "frozen.project_root",
    # scientific policy
    "apply_threshold": "frozen.threshold.apply_threshold",
    "threshold_min": "frozen.threshold.threshold_min",
    "threshold_max": "frozen.threshold.threshold_max",
    "mask_sentinel": "frozen.threshold.mask_saturation",
    "mask_file": "frozen.mask_file",
    "poni_file": "frozen.poni_file",
    # GI
    "gi": "frozen.gi.enabled",
    "incidence_motor": "frozen.gi.scan_incidence_motor",
    "sample_orientation": "frozen.gi.sample_orientation",
    "tilt_angle": "frozen.gi.tilt_angle",
    # mode / parallelism
    "live_mode": "frozen.live_mode",
    "batch_mode": "frozen.batch_mode",
    "max_cores": "frozen.max_cores",
    "xye_only": "frozen.run_options['xye_only']",
    "series_average": "frozen.run_options['series_average']",
    "meta_ext": "frozen.run_options['meta_ext']",
    # source family / traversal (never the frame cursor)
    "inp_type": "frozen.source.family / .source_kind",
    "img_ext": "frozen.source.format_tokens",
    "img_dir": "frozen.source.filesystem_root",
    "single_img": "frozen.source.source_kind",
    "include_subdir": "frozen.source.recursive",
    "file_filter": "frozen.source.name_filter",
    "source_spec": "frozen.thaw_source_spec()",
    "scan_args": "frozen.scan_args()",
}

#: Runtime cursor state stays writable: it is real per-frame data, not policy.
RUNTIME_SOURCE_CURSOR = (
    "img_file", "img_fnames", "scan_name", "processed", "poni", "detector",
    "mask", "meta_dir", "source_base",
)

#: The ONLY places allowed to consult the published carrier: the worker-entry
#: identity gates (review §39.5 Phase 1 item 5) and the admission owner.
ENTRY_GATE_SCOPES = {
    ("imageThread", "_require_run_configuration"),
    ("imageThread", "run"),
    ("imageThread", "initialize_scan"),
    ("nexusThread", "_require_run_configuration"),
    ("nexusThread", "run"),
    ("nexusThread", "_initialize_scan"),
    ("wranglerThread", "__init__"),
    ("wranglerWidget", "_admit_run_configuration"),
    ("wranglerWidget", "_bind_admitted_run_configuration"),
    ("wranglerWidget", "_publish_run_configuration_to_thread"),
}

WORKER_CLASSES = ("imageThread", "nexusThread", "wranglerThread")
_WRANGLER_DIR = Path(wrangler_widget.__file__).parent
_WORKER_FILES = (
    _WRANGLER_DIR / "image_wrangler_thread.py",
    _WRANGLER_DIR / "nexus_wrangler_thread.py",
    _WRANGLER_DIR / "wrangler_widget.py",
)
_SINK_FILE = _WRANGLER_DIR / "qt_nexus_sink.py"
_PRODUCTION_ROOT = Path(image_wrangler_thread.__file__).parents[4]


def _worker_scopes(tree, *, methods_only=False):
    """Yield (class, FunctionDef) for every scope of a worker class.

    ``methods_only`` drops nested closures: they see their enclosing method's
    stack local, so they neither need nor could declare the parameter.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in WORKER_CLASSES:
            continue
        methods = [n for n in node.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for method in methods:
            yield node.name, method
            if methods_only:
                continue
            for child in ast.walk(method):
                if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child is not method):
                    yield node.name, child


def test_the_projection_bridge_symbols_do_not_exist():
    """Item 1.  Nothing may re-import or re-create the retired bridge."""
    for module in (wrangler_widget, image_wrangler_thread, nexus_wrangler_thread):
        for symbol in ("FrozenRunProjection", "FROZEN_RUN_PROJECTIONS",
                       "install_frozen_run_projections", "frozen_run_policy",
                       "_frozen_source_root", "_frozen_source_format",
                       "_frozen_source_family", "_frozen_single_image",
                       "_frozen_recursive", "_frozen_name_filter",
                       "_frozen_run_option", "_frozen_thawed_source",
                       "_source_format_token"):
            assert not hasattr(module, symbol), (
                f"{module.__name__}.{symbol} survived D3")
    for cls in (imageThread, nexusThread, wranglerThread):
        assert not hasattr(cls, "_frozen_run_policy"), (
            f"{cls.__name__}._frozen_run_policy survived D3")

    # The admission-time filesystem check is deliberately RETAINED (§43.1): it
    # is not a value projection, it performs I/O at the fallible GUI boundary.
    assert callable(wrangler_widget._frozen_source_is_admissible)


@pytest.mark.parametrize("cls", [imageThread, nexusThread, wranglerThread],
                         ids=lambda c: c.__name__)
def test_no_worker_class_carries_a_retired_policy_name(cls):
    """Item 2.  Not as a descriptor, a slot, or a class attribute."""
    for name in RETIRED_POLICY_NAMES:
        assert not hasattr(cls, name), (
            f"{cls.__name__}.{name} still exists; execution could resolve "
            f"policy from it instead of {RETIRED_POLICY_NAMES[name]}")


def test_no_worker_reads_a_retired_policy_name_off_self():
    """Item 3.  Every execution read takes the accepted object as an argument."""
    offenders = []
    for path in _WORKER_FILES:
        tree = ast.parse(path.read_text())
        for cls, fn in _worker_scopes(tree):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "self"
                        and node.attr in RETIRED_POLICY_NAMES):
                    offenders.append(
                        f"{path.name}:{node.lineno} {cls}.{fn.name} "
                        f"reads self.{node.attr}")
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and len(node.args) >= 2
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "self"
                        and isinstance(node.args[1], ast.Constant)
                        and node.args[1].value in RETIRED_POLICY_NAMES):
                    offenders.append(
                        f"{path.name}:{node.lineno} {cls}.{fn.name} "
                        f"reads getattr(self, {node.args[1].value!r})")
    assert offenders == [], offenders


def test_no_host_reads_or_writes_a_retired_name_on_a_worker():
    """Item 3 (host half).  The GUI reads the accepted object, not a mirror."""
    offenders = []
    for path in sorted(_PRODUCTION_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            target = receiver = None
            if isinstance(node, ast.Attribute) and node.attr in RETIRED_POLICY_NAMES:
                receiver, target = ast.unparse(node.value), node.attr
            elif (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in RETIRED_POLICY_NAMES):
                receiver, target = ast.unparse(node.args[0]), node.args[1].value
            if receiver is None or receiver == "self":
                continue
            lowered = receiver.lower()
            if ("thread" in lowered or "worker" in lowered) and (
                    "_accepted_run_policy" not in receiver):
                offenders.append(f"{path.name}:{node.lineno} {receiver}.{target}")
    assert offenders == [], offenders


def test_streaming_sink_does_not_read_retired_policy_from_its_host():
    """The transitive writer consumer receives policy from the frozen run."""
    tree = ast.parse(_SINK_FILE.read_text())
    offenders = []
    sink_policy = {
        "xye_only",
        "batch_mode",
        "gi",
        "incidence_motor",
        "series_average",
    }
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and node.attr in sink_policy
                and ast.unparse(node.value) == "self._host"):
            offenders.append(
                f"{_SINK_FILE.name}:{node.lineno} self._host.{node.attr}")
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and ast.unparse(node.args[0]) == "self._host"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in sink_policy):
            offenders.append(
                f"{_SINK_FILE.name}:{node.lineno} "
                f"getattr(self._host, {node.args[1].value!r})")
    assert offenders == [], offenders


def test_only_the_entry_gates_consult_the_published_carrier():
    """Item 4.  No execution helper re-reads a holder after worker entry."""
    holders = {"run_configuration", "_qualified_run_configuration"}
    offenders = []
    for path in _WORKER_FILES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for fn in [n for n in ast.walk(node)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                if (node.name, fn.name) in ENTRY_GATE_SCOPES:
                    continue
                for inner in ast.walk(fn):
                    name = None
                    if (isinstance(inner, ast.Attribute)
                            and isinstance(inner.ctx, ast.Load)
                            and inner.attr in holders):
                        name = inner.attr
                    elif (isinstance(inner, ast.Call)
                            and isinstance(inner.func, ast.Name)
                            and inner.func.id == "getattr"
                            and len(inner.args) >= 2
                            and isinstance(inner.args[1], ast.Constant)
                            and inner.args[1].value in holders):
                        name = inner.args[1].value
                    if name is not None:
                        offenders.append(
                            f"{path.name}:{inner.lineno} "
                            f"{node.name}.{fn.name} re-reads {name}")
    assert offenders == [], offenders

    # And the stored active reference itself is gone (§43.4): the graph is
    # parameter-threaded, so there is no lifecycle to clear.
    assert "_qualified_run_configuration" not in "".join(
        path.read_text() for path in _WORKER_FILES)


def test_every_policy_consuming_worker_scope_declares_frozen():
    """Item 5.  The routed value arrives by required argument, uniformly."""
    missing = []
    for path in _WORKER_FILES:
        tree = ast.parse(path.read_text())
        for cls, fn in _worker_scopes(tree, methods_only=True):
            source = ast.unparse(fn)
            if "frozen." not in source:
                continue
            args = [a.arg for a in fn.args.args]
            if "frozen" in args:
                # and it must be REQUIRED: an optional policy argument is a
                # legacy fallback in disguise (§42.5 mutation 12).
                first_default = len(args) - len(fn.args.defaults)
                assert args.index("frozen") < first_default, (
                    f"{cls}.{fn.name} makes the accepted configuration "
                    "optional")
                continue
            kwonly = {a.arg for a in fn.args.kwonlyargs}
            if "frozen" in kwonly:
                index = [a.arg for a in fn.args.kwonlyargs].index("frozen")
                assert fn.args.kw_defaults[index] is None, (
                    f"{cls}.{fn.name} makes the accepted configuration "
                    "optional")
                continue
            # an entry gate creates the local itself (nested closures are
            # excluded above: they see their enclosing method's local)
            if (cls, fn.name) in ENTRY_GATE_SCOPES:
                continue
            if any(isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "frozen"
                           for t in node.targets)
                   for node in ast.walk(fn)):
                continue
            missing.append(f"{path.name}:{fn.lineno} {cls}.{fn.name}")
    assert missing == [], missing


def test_runtime_cursor_state_is_still_writable():
    """The retirement must not have swept away real per-frame data."""
    worker = imageThread.__new__(imageThread)
    for name in RUNTIME_SOURCE_CURSOR:
        setattr(worker, name, "value")
        assert getattr(worker, name) == "value"


def test_the_worker_entry_gates_are_the_documented_three():
    """The gate LOCATION stays knowable: exactly the named entry points."""
    for cls, name in (
            (imageThread, "run"), (imageThread, "initialize_scan"),
            (nexusThread, "run"), (nexusThread, "_initialize_scan")):
        source = inspect.getsource(getattr(cls, name))
        assert "_require_run_configuration" in source, (
            f"{cls.__name__}.{name} no longer gates its entry")
