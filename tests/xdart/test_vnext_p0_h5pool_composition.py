"""C1 composition contract for the headless HDF5 pool owner."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys


def test_headless_pool_module_owns_compatibility_adapter():
    headless = importlib.import_module("xrd_tools.session.io_coordination")
    adapter = importlib.import_module("xdart.utils.h5pool")

    assert headless.H5FilePool.__module__ == (
        "xrd_tools.session.io_coordination"
    )
    assert adapter.H5FilePool is headless.H5FilePool
    assert adapter.get_pool is headless.get_pool
    assert adapter.get_pool() is headless.get_pool()

    adapter_tree = ast.parse(Path(adapter.__file__).read_text(encoding="utf-8"))
    definitions = [
        node for node in adapter_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    assignments = {
        target.id
        for node in adapter_tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    assert definitions == [], "xdart.utils.h5pool must be a pure adapter"
    assert not ({"_pool", "_pool_key"} & assignments)


def test_direct_headless_pool_singleton_is_qt_free_and_concurrent():
    probe = (
        "import concurrent.futures, importlib, sys, threading\n"
        "io_name='xrd_tools.session.io_coordination'\n"
        "assert io_name not in sys.modules\n"
        "barrier=threading.Barrier(12)\n"
        "def first(_):\n"
        " barrier.wait()\n"
        " module=importlib.import_module(io_name)\n"
        " return id(module.get_pool()), id(module.H5FilePool)\n"
        "with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:\n"
        " identities=list(ex.map(first, range(12)))\n"
        "assert len(set(identities)) == 1, identities\n"
        "module=importlib.import_module(io_name)\n"
        "assert module.H5FilePool.__module__ == io_name\n"
        "forbidden=('PySide6','PyQt5','PyQt6','qtpy','pyqtgraph','xdart')\n"
        "leaked=sorted(m for m in sys.modules if m.split('.')[0] in forbidden)\n"
        "assert not leaked, leaked\n"
        "print('direct-headless-singleton-ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "direct-headless-singleton-ok" in proc.stdout
