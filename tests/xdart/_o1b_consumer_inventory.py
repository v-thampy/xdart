"""O-1b — the frozen transitive execution-consumer inventory (acceptance artifact).

The `QtNexusSink._host` escape (review §45.1) was invisible to the W-1R-D3 guard
because that guard keyed on receiver NAMES (`thread`, `worker`) and on an
enumerated file list.  The O-1b rule replaces both: the guard DERIVES the set of
objects that are handed a worker reference, follows simple receiver aliases, and
compares the derivation against this inventory.  A new downstream consumer
therefore fails the guard until it is declared here with its classification.

This module is data plus one bounded derivation helper.  It is deliberately NOT a
dataflow analyzer: it tracks a worker passed as a call argument and stored under a
simple attribute alias, nothing more.

Classification vocabulary
-------------------------
``OPERATIONAL``
    The consumer touches only run-mechanics: Qt signals, locks, the cancellation
    command, frame buffers, queues.  Explicitly allowlisted.
``POLICY_BY_ARGUMENT``
    The consumer needs run/source/GI policy and receives the accepted
    ``FrozenRunConfiguration`` (or a value derived from it) as a REQUIRED typed
    argument, identity-qualified against the worker's admission ledger.
``VALUE_ONLY``
    The consumer is handed an immutable value object and holds no worker
    reference at all.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

#: The run workers whose references are tracked outward.
WORKER_CLASSES = ("imageThread", "nexusThread", "wranglerThread")

#: Run/source/GI policy that may NEVER be read through a worker reference.
#: Finite by construction: this is the retired W-1R carrier set plus the O-1b
#: source-shape values.  Operational names are not in it and never should be.
FORBIDDEN_POLICY_FIELDS = frozenset({
    # output
    "write_mode", "h5_dir", "project_folder",
    # scientific policy
    "apply_threshold", "threshold_min", "threshold_max", "mask_sentinel",
    "mask_file", "poni_file",
    # grazing incidence
    "gi", "incidence_motor", "sample_orientation", "tilt_angle",
    # mode / parallelism
    "live_mode", "batch_mode", "max_cores", "xye_only", "series_average",
    "meta_ext",
    # source shape / traversal
    "inp_type", "img_ext", "img_dir", "single_img", "include_subdir",
    "file_filter", "source_spec", "scan_args",
})

#: Run mechanics a downstream consumer MAY reach through its worker reference.
#: Signals, locks, cancellation, buffers and queues -- never policy.
OPERATIONAL_ALLOWLIST = frozenset({
    "showLabel", "sigUpdate", "sigUpdateFile", "sigUpdateData", "sigUpdateGI",
    "sigPaused", "sigResuming", "sigXyeOutputReady", "sigGIMotorOptions",
    "sigAppendMismatch", "command", "command_lock", "command_q", "input_q",
    "signal_q", "file_lock", "_xye_lock", "_xye_buffer", "_published_frames",
    "_prefetch_queue", "_prefetch_stop_evt", "LIVE_SAVE_INTERVAL",
    "_middle_truncate", "_frames_since_save", "scan", "fname",
})

#: THE INVENTORY.  Every object handed a worker reference, with its alias, the
#: worker site that hands it over, and its classification.
#:
#: consumer -> {aliases, handed_by, classification, note}
DECLARED_CONSUMERS: dict[str, dict[str, Any]] = {
    "QtNexusSink": {
        "aliases": frozenset({"self._host"}),
        "handed_by": ("imageThread._get_streaming_session",),
        "classification": "POLICY_BY_ARGUMENT",
        "note": (
            "The streaming writer. Takes `run_configuration` as a REQUIRED "
            "keyword argument, refuses a non-frozen value, and refuses any "
            "object that is not the one its host admitted (identity). Its "
            "`_host` reads are operational only (review §46.2, `a62b6c3b`)."
        ),
    },
    "ScanSessionAdapter": {
        "aliases": frozenset({"self._host"}),
        "handed_by": ("imageThread._get_streaming_session",),
        "classification": "OPERATIONAL",
        "note": (
            "Pause/resume/stop plumbing for the streaming session. Reads only "
            "`showLabel`, `command` and `command_lock`."
        ),
    },
    "_CommandCancelToken": {
        "aliases": frozenset({"self._owner"}),
        "handed_by": ("wranglerThread._cancel_token",),
        "classification": "OPERATIONAL",
        "note": "Duck-typed CancelToken; reads `command` and nothing else.",
    },
}

#: Worker bound methods handed outward as callbacks.  Each must receive its run
#: policy as an argument at handoff, not read it from the worker afterwards.
DECLARED_CALLBACKS: dict[str, dict[str, Any]] = {
    "imageThread._prefetch_worker": {
        "handed_to": "threading.Thread(target=...) in _start_prefetcher",
        "classification": "POLICY_BY_ARGUMENT",
        "note": (
            "The Eiger prefetch thread receives the accepted configuration as "
            "the first spawn argument, so a concurrent reader never consults a "
            "carrier from another thread (review §43.4)."
        ),
    },
}


# --------------------------------------------------------------------------- #
# The bounded derivation the guard runs against the real tree.
# --------------------------------------------------------------------------- #

def _class_index(src_root: Path):
    index, trees = {}, {}
    for path in sorted(src_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        trees[path] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                index.setdefault(node.name, (path, node))
    return index, trees


def _methods(cls_node):
    return [n for n in cls_node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _init_param(cls_node, key):
    for m in _methods(cls_node):
        if m.name != "__init__":
            continue
        positional = [a.arg for a in m.args.args]
        if isinstance(key, int):
            return positional[key + 1] if key + 1 < len(positional) else None
        kwonly = [a.arg for a in m.args.kwonlyargs]
        return key if (key in positional or key in kwonly) else None
    return None


def _aliases_for(cls_node, params):
    found = set()
    for m in _methods(cls_node):
        if m.name != "__init__":
            continue
        for node in ast.walk(m):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Attribute)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id == "self"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in params):
                found.add(f"self.{node.targets[0].attr}")
    return found


def derive_worker_alias_attributes(src_root: Path):
    """Attribute names the tree actually stores a worker under.

    DERIVED, never assumed: an attribute counts only because some class assigns
    a worker CONSTRUCTION to it (``self.thread = wranglerThread(...)``).  This is
    what lets the guard follow a host's worker reference without keying on the
    names "thread" or "worker", which is exactly how the `QtNexusSink._host`
    escape survived the W-1R-D3 guard.
    """
    _index, trees = _class_index(src_root)
    aliases = set()
    for _path, tree in trees.items():
        # O-1b-0.1: a local alias for the worker CLASS still constructs a worker,
        # so resolve `builder = imageThread` before reading constructions.
        class_aliases = set(WORKER_CLASSES)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in class_aliases):
                class_aliases.add(node.targets[0].id)
        for node in ast.walk(tree):
            # O-1b-0.1: annotated storage (`self.thread: imageThread = ...`) is
            # an AnnAssign, which the first version missed entirely.
            if isinstance(node, ast.AnnAssign):
                target, value = node.target, node.value
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            else:
                continue
            if not isinstance(target, ast.Attribute):
                continue
            if isinstance(node, ast.AnnAssign) and isinstance(
                    node.annotation, ast.Name):
                if node.annotation.id in class_aliases:
                    aliases.add(target.attr)
            builder = None
            if isinstance(value, ast.Call):
                builder = (value.func.id if isinstance(value.func, ast.Name)
                           else getattr(value.func, "attr", None))
            if builder in class_aliases:
                aliases.add(target.attr)
    return frozenset(aliases)


def derive_callback_handoffs(src_root: Path):
    """Real ``Thread(target=self.<method>, args=(...))`` handoffs from a worker.

    Returns ``{f"{cls}.{method}": {"args": [...], "site": "...",
    "target_params": [...]}}``.  Derived from the tree so the guard can prove the
    accepted configuration is actually PASSED, rather than trusting a declaration
    (the first version's callback row was tautological).
    """
    index, _trees = _class_index(src_root)
    handoffs = {}
    for worker in WORKER_CLASSES:
        if worker not in index:
            continue
        path, cls_node = index[worker]
        by_name = {m.name: m for m in _methods(cls_node)}
        for method in _methods(cls_node):
            for node in ast.walk(method):
                if not isinstance(node, ast.Call):
                    continue
                builder = (node.func.id if isinstance(node.func, ast.Name)
                           else getattr(node.func, "attr", None))
                if builder != "Thread":
                    continue
                target = args = None
                for kw in node.keywords:
                    if kw.arg == "target":
                        target = ast.unparse(kw.value)
                    elif kw.arg == "args":
                        args = kw.value
                if not target or not target.startswith("self."):
                    continue
                callee = target.split(".", 1)[1]
                arg_texts = ([ast.unparse(e) for e in args.elts]
                             if isinstance(args, ast.Tuple) else [])
                params = ([a.arg for a in by_name[callee].args.args]
                          if callee in by_name else [])
                handoffs[f"{worker}.{callee}"] = {
                    "args": arg_texts,
                    "site": f"{path.name}:{node.lineno} {worker}.{method.name}",
                    "target_params": params,
                }
    return handoffs


def derive_consumer_closure(src_root: Path, *, max_depth: int = 6):
    """Objects handed a worker reference, keyed by class name.

    Returns ``{class_name: {"aliases": {...}, "handed_by": {...},
    "path": Path}}``.  Recurses: a consumer that passes its own alias onward
    makes the receiver a consumer too.
    """
    index, _trees = _class_index(src_root)
    closure: dict[str, dict[str, Any]] = {}
    frontier = {name: {"self"} for name in WORKER_CLASSES if name in index}
    depth = 0
    while frontier and depth < max_depth:
        nxt: dict[str, set[str]] = {}
        for holder, holder_aliases in frontier.items():
            _hpath, holder_node = index[holder]
            for method in _methods(holder_node):
                # O-1b-0.1: a local bound to an alias (`me = self`) hands the
                # same reference onward, so watch it for this method too.
                local_aliases = set()
                for node in ast.walk(method):
                    if (isinstance(node, ast.Assign) and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Name)
                            and ast.unparse(node.value) in holder_aliases):
                        local_aliases.add(node.targets[0].id)
                watched = set(holder_aliases) | local_aliases
                for node in ast.walk(method):
                    if not isinstance(node, ast.Call):
                        continue
                    target = (node.func.id if isinstance(node.func, ast.Name)
                              else getattr(node.func, "attr", None))
                    if target not in index or target in WORKER_CLASSES:
                        continue
                    keys = []
                    for position, arg in enumerate(node.args):
                        if ast.unparse(arg) in watched:
                            keys.append(position)
                    for kw in node.keywords:
                        if kw.arg and ast.unparse(kw.value) in watched:
                            keys.append(kw.arg)
                    if not keys:
                        continue
                    tpath, tnode = index[target]
                    params = {p for p in (_init_param(tnode, k) for k in keys)
                              if p}
                    aliases = _aliases_for(tnode, params)
                    entry = closure.setdefault(target, {
                        "aliases": set(), "handed_by": set(), "path": tpath})
                    entry["aliases"] |= aliases
                    entry["handed_by"].add(f"{holder}.{method.name}")
                    if aliases:
                        nxt.setdefault(target, set()).update(aliases)
        frontier = nxt
        depth += 1
    return closure


def _policy_access(node):
    """(receiver_expr, field) for an attribute or getattr access, else None."""
    if isinstance(node, ast.Attribute) and isinstance(
            node.ctx, (ast.Load, ast.Store)):
        return ast.unparse(node.value), node.attr
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "getattr" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)):
        return ast.unparse(node.args[0]), node.args[1].value
    return None


def forbidden_reads_through_aliases(src_root: Path, closure):
    """Every forbidden policy access reached through a worker reference.

    Covers both holder shapes and is scoped by DERIVATION, not by a file list:

    * a consumer handed a worker (its declared alias, e.g. ``self._host``);
    * any expression whose final attribute is one the tree stores a worker under
      (derived by :func:`derive_worker_alias_attributes`), e.g.
      ``self.wrangler.thread``;
    * simple locals bound to either of those (``host = self._host``,
      ``thread = getattr(wrangler, "thread", None)``).
    """
    offenders = []
    index, trees = _class_index(src_root)
    worker_attrs = derive_worker_alias_attributes(src_root)
    watched_globally = {f"self.{attr}" for attr in worker_attrs}

    consumer_aliases = {}
    for consumer, entry in closure.items():
        if consumer in index:
            consumer_aliases[index[consumer][0]] = (
                consumer, set(entry["aliases"]))

    for path, tree in trees.items():
        declared = consumer_aliases.get(path)
        base = set(watched_globally)
        if declared is not None:
            base |= declared[1]
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            watched = set(base)
            # a local bound to an alias, or to any <expr>.<worker-attr>
            for node in ast.walk(scope):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)):
                    text = ast.unparse(node.value)
                    access = _policy_access(node.value)
                    if (text in watched
                            or (access is not None
                                and access[1] in worker_attrs)):
                        watched.add(node.targets[0].id)
            for node in ast.walk(scope):
                access = _policy_access(node)
                if access is None:
                    continue
                receiver, field = access
                if field not in FORBIDDEN_POLICY_FIELDS:
                    continue
                if field in OPERATIONAL_ALLOWLIST:
                    continue
                tail = receiver.rsplit(".", 1)[-1]
                if receiver not in watched and tail not in worker_attrs:
                    continue
                offenders.append(
                    f"{path.name}:{node.lineno} {scope.name} "
                    f"accesses {receiver}.{field}")
    return offenders
