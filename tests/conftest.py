# -*- coding: utf-8 -*-
"""Top-level test guard: pin the Qt binding to PySide6 BEFORE anything imports
pyFAI.

pyFAI loads a Qt binding at import time and defaults to PyQt5 unless ``QT_API``
is already set.  The combined ``tests/core`` + ``tests/xdart`` run imports
``xrd_tools.integrate.calibration`` (via tests/core/conftest.py) -> ``import
pyFAI`` BEFORE ``xdart/__init__`` runs its own ``QT_API``/``PYQTGRAPH_QT_LIB``
pin, so PyQt5 and PySide6 could both end up loaded in one process -> a SIGSEGV
when the PySide6 GUI widgets are constructed (codex P1, a monorepo import-order
trap).

This is the rootdir conftest, imported before any test module is collected, so
setting the binding here makes it deterministic regardless of whether core or
xdart tests are collected first.  ``setdefault`` so an explicit ``QT_API`` in
the environment still wins.
"""
import os

os.environ.setdefault("QT_API", "PySide6")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")


# ---------------------------------------------------------------------------
# The Windows stat shape, reproducible on any host
# ---------------------------------------------------------------------------
#
# CPython on Windows fills ``st_ctime`` from the change time for a handle
# ``fstat`` but from the creation time for a pathname ``stat``/``lstat``, so a
# descriptor view and a pathname view of one untouched file disagree on ctime
# alone (374 ms on the windows-latest runner, PR #1 2026-09-11).  Every
# identity compare between the two views routes its ctime slot through
# ``xrd_tools.io.stat_identity.identity_ctime_ns``; the fixture below gives
# one file that shape here so each compare can be exercised under both seam
# settings without a Windows host.

import pytest  # noqa: E402  (after the Qt pin above, by design)

_WIN32_CTIME_GAP_NS = 374_000_000


class _CreationTimeStat:
    """A pathname stat whose ``st_ctime_ns`` is the (earlier) creation time."""

    def __init__(self, real, ctime_ns: int) -> None:
        self._real = real
        self.st_ctime_ns = ctime_ns

    def __getattr__(self, name: str):
        return getattr(self._real, name)


@pytest.fixture
def ctime_seam():
    """The tree-wide win32 ctime seam module."""
    import importlib

    return importlib.import_module("xrd_tools.io.stat_identity")


@pytest.fixture
def win32_pathname_ctime(monkeypatch):
    """``install(path)``: pathname ``os.stat``/``os.lstat`` of *path* report a
    ctime ``_WIN32_CTIME_GAP_NS`` earlier than ``os.fstat`` does, and only
    that.  Descriptor views are untouched, as on Windows."""
    real_stat, real_lstat = os.stat, os.lstat

    def install(path) -> int:
        spelled = {os.path.abspath(os.fspath(path)), os.path.realpath(path)}

        def shifted(real):
            def call(target, *args, **kwargs):
                result = real(target, *args, **kwargs)
                if isinstance(target, (str, bytes, os.PathLike)) and (
                    os.path.abspath(os.fsdecode(target)) in spelled
                ):
                    return _CreationTimeStat(
                        result, result.st_ctime_ns - _WIN32_CTIME_GAP_NS,
                    )
                return result

            return call

        monkeypatch.setattr(os, "stat", shifted(real_stat))
        monkeypatch.setattr(os, "lstat", shifted(real_lstat))
        return _WIN32_CTIME_GAP_NS

    return install
