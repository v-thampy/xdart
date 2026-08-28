"""Direct contract tests for the headless HDF5 read-handle pool."""

from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np

from xrd_tools.session import H5FilePool


def _make_file(tmp_path, name="pool.h5"):
    path = tmp_path / name
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.arange(4))
    return str(path)


def test_pause_blocks_get_until_matching_resume(tmp_path):
    pool = H5FilePool(max_open=2)
    path = _make_file(tmp_path)
    assert pool.get(path) is not None
    pool.pause(path)
    pool.pause(path)
    assert pool.get(path) is None
    pool.resume(path)
    assert pool.get(path) is None
    pool.resume(path)
    assert pool.get(path) is not None
    pool.close_all()


def test_unbalanced_resume_is_safe(tmp_path):
    pool = H5FilePool(max_open=2)
    path = _make_file(tmp_path)
    pool.resume(path)
    pool.resume(path)
    pool.pause(path)
    assert pool.get(path) is None
    pool.resume(path)
    assert pool.get(path) is not None
    pool.close_all()


def test_lru_eviction_and_stale_handle_reopen(tmp_path):
    pool = H5FilePool(max_open=2)
    first = _make_file(tmp_path, "first.h5")
    second = _make_file(tmp_path, "second.h5")
    third = _make_file(tmp_path, "third.h5")

    first_handle = pool.get(first)
    second_handle = pool.get(second)
    assert pool.get(first) is first_handle
    pool.get(third)
    assert first_handle.id.valid and not second_handle.id.valid

    first_handle.close()
    reopened = pool.get(first)
    assert reopened is not first_handle and reopened.id.valid
    pool.close_all()


def test_public_lazy_export_has_one_qt_free_concurrent_singleton():
    probe = (
        "import concurrent.futures, sys, threading\n"
        "import xrd_tools.session as session\n"
        "io_name='xrd_tools.session.io_coordination'\n"
        "assert io_name not in sys.modules\n"
        "barrier=threading.Barrier(12)\n"
        "def first(_):\n"
        " barrier.wait()\n"
        " return id(session.get_pool())\n"
        "with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:\n"
        " identities=list(ex.map(first, range(12)))\n"
        "assert len(set(identities)) == 1, identities\n"
        "assert session.H5FilePool.__module__ == io_name\n"
        "forbidden=('PySide6','PyQt5','PyQt6','qtpy','pyqtgraph','xdart')\n"
        "leaked=sorted(m for m in sys.modules if m.split('.')[0] in forbidden)\n"
        "assert not leaked, leaked\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
