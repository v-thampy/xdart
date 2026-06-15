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
