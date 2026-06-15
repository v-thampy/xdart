"""Import-order guard: importing ``xrd_tools.integrate.calibration`` must NOT
load pyFAI (and therefore must not load a Qt binding) at import time.

pyFAI loads a Qt binding when imported (defaulting to PyQt5 when ``QT_API`` is
unset), so an eager ``import pyFAI`` in calibration would let a notebook/headless
process pin the wrong binding merely by importing this module.  xrd_tools is
headless-first: pyFAI is imported lazily, inside the functions that use it.

Runs in a clean subprocess with ``QT_API`` unset so the contract is verified
without the test-suite's ``QT_API`` pin (tests/conftest.py) or an already-imported
pyFAI masking the leak.
"""
from __future__ import annotations

import os
import subprocess
import sys

_CHILD = """
import sys
import xrd_tools.integrate.calibration  # noqa: F401

leaked = sorted(
    m for m in sys.modules
    if m == "pyFAI" or m.startswith("pyFAI.")
    or m == "PyQt5" or m.startswith("PyQt5.")
)
if leaked:
    print("LEAKED:" + ",".join(leaked))
    sys.exit(1)
sys.exit(0)
"""


def test_importing_calibration_does_not_load_pyfai_or_pyqt5():
    env = dict(os.environ)
    # Unset the binding pins so the import is proven clean on its own, not
    # because a Qt binding was already chosen for us.
    for var in ("QT_API", "PYQTGRAPH_QT_LIB"):
        env.pop(var, None)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, (
        "importing xrd_tools.integrate.calibration loaded pyFAI/PyQt5 at import "
        f"time.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
