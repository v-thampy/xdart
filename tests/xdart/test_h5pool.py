# -*- coding: utf-8 -*-
"""H5FilePool pause/resume refcount (review_2026-06-15 §5).

A plain-set ``_paused`` let the FIRST resume reopen a file that a SECOND,
still-active writer had paused — a torn read mid-write once the 7+8 flush
fan-out makes overlapping pauses routine.  The Counter refcount keeps a path
paused until every pause is matched.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from xdart.utils.h5pool import H5FilePool


def _make_file(tmp_path, name="pool.h5"):
    p = tmp_path / name
    with h5py.File(p, "w") as f:
        f.create_dataset("x", data=np.arange(4))
    return str(p)


def test_pause_blocks_get_resume_unblocks(tmp_path):
    pool = H5FilePool(max_open=2)
    path = _make_file(tmp_path)
    assert pool.get(path) is not None
    pool.pause(path)
    assert pool.get(path) is None        # paused: must not reopen
    pool.resume(path)
    assert pool.get(path) is not None     # un-paused
    pool.close_all()


def test_nested_pause_needs_matching_resumes(tmp_path):
    # The §5 regression: two concurrent writers pause the same file.
    pool = H5FilePool(max_open=2)
    path = _make_file(tmp_path)
    pool.pause(path)                      # writer A
    pool.pause(path)                      # writer B
    pool.resume(path)                     # A done — B still writing
    assert pool.get(path) is None         # MUST stay paused (the bug = not None)
    pool.resume(path)                     # B done
    assert pool.get(path) is not None     # now safe to reopen
    pool.close_all()


def test_unbalanced_resume_is_safe(tmp_path):
    # A stray resume on a never-paused path must not drive the count negative
    # (which would make a later single pause fail to block).
    pool = H5FilePool(max_open=2)
    path = _make_file(tmp_path)
    pool.resume(path)                     # never paused
    pool.resume(path)
    pool.pause(path)
    assert pool.get(path) is None         # one pause still blocks
    pool.resume(path)
    assert pool.get(path) is not None
    pool.close_all()


def test_c3_headless_contract_owns_pool_and_compatibility_is_adapter():
    import xrd_tools.session as session

    getter = getattr(session, "get_pool", None)
    assert callable(getter)
    pool_type = getattr(session, "H5FilePool", None)
    assert isinstance(pool_type, type)
    assert pool_type.__module__ == "xrd_tools.session.io_coordination"

    adapter = importlib.import_module("xdart.utils.h5pool")
    utils = importlib.import_module("xdart.utils")
    assert adapter.H5FilePool is pool_type
    assert utils.H5FilePool is pool_type
    assert adapter.get_pool is getter
    assert adapter.get_pool() is getter() is getter()

    adapter_tree = ast.parse(Path(adapter.__file__).read_text(encoding="utf-8"))
    definitions = [node for node in adapter_tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    assignments = {
        target.id
        for node in adapter_tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    assert definitions == []
    assert not ({"_pool", "_pool_key"} & assignments)

    utils_tree = ast.parse(
        (Path(adapter.__file__).parent / "__init__.py").read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "xrd_tools.session"
        and any(alias.name == "H5FilePool" for alias in node.names)
        for node in utils_tree.body
    )


def test_c3_lazy_qt_free_concurrent_first_access_has_one_singleton():
    probe = (
        "import concurrent.futures, sys, threading\n"
        "import xrd_tools.session as session\n"
        "io_name='xrd_tools.session.io_coordination'\n"
        "assert io_name not in sys.modules, 'plain session import was not light'\n"
        "barrier=threading.Barrier(12)\n"
        "def first(_):\n"
        " barrier.wait()\n"
        " getter=getattr(session, 'get_pool', None)\n"
        " assert callable(getter), 'H10-C3 callable session.get_pool is absent'\n"
        " return id(getter())\n"
        "with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:\n"
        " identities=list(ex.map(first, range(12)))\n"
        "assert len(set(identities)) == 1, identities\n"
        "assert io_name in sys.modules\n"
        "forbidden=('PySide6','PyQt5','PyQt6','qtpy','pyqtgraph','xdart')\n"
        "leaked=sorted(m for m in sys.modules if m.split('.')[0] in forbidden)\n"
        "assert not leaked, leaked\n"
        "print('c3-singleton-ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "c3-singleton-ok" in proc.stdout


def test_lru_eviction_and_stale_handle_reopen(tmp_path):
    pool = H5FilePool(max_open=2)
    first = _make_file(tmp_path, "first.h5")
    second = _make_file(tmp_path, "second.h5")
    third = _make_file(tmp_path, "third.h5")

    first_handle = pool.get(first)
    second_handle = pool.get(second)
    assert pool.get(first) is first_handle  # make first the most recent
    pool.get(third)
    assert first_handle.id.valid and not second_handle.id.valid

    first_handle.close()  # stale cached handle must be replaced on demand
    reopened = pool.get(first)
    assert reopened is not first_handle and reopened.id.valid
    pool.close_all()
