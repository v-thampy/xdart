# -*- coding: utf-8 -*-
"""Run pytest, then hard-exit with its verdict — skipping interpreter teardown.

Why this exists: on linux CI the ``tests/xdart`` offscreen suite SIGSEGVs
(exit 139) during *Python interpreter shutdown* — deterministically since
2026-07-08, ~30s AFTER pytest has printed its summary and written a clean
JUnit report.  The crash is in the teardown of ~1600 tests' worth of
accumulated PySide6/pyqtgraph object graphs at Py_Finalize, not in any test:
a mid-run crash makes ``pytest.main`` never return, so the process still dies
with the raw signal exit and CI still fails loud.

``os._exit(rc)`` after ``pytest.main`` returns keeps pytest's own verdict as
the process exit code while skipping GC/atexit/Py_Finalize where the segfault
lives.  This is the standard practice for Qt suites in CI — pyqtgraph ships
``pg.exit()`` (the same ``os._exit`` trick) precisely for this crash class.

The JUnit report (``--junitxml``) is written inside ``pytest.main`` (during
``pytest_sessionfinish``), so it is complete before the hard exit.

Usage (CI):  python scripts/ci_pytest.py tests/xdart -q --timeout=600 ...
Local runs don't need it — plain ``pytest`` keeps full teardown semantics.
"""
import os
import sys

import pytest


def main() -> None:
    rc = pytest.main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(rc))


if __name__ == "__main__":
    main()
