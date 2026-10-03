"""Preserve current page-service isolation and lazy imports."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from xdart.gui.pages.services import (
    DiagnosticIdentity,
    ExecutionProfile,
    HostServices,
)
from xdart.gui.pages.values import PageKey
SELECTED = PageKey("scattering-workspace")
FOREIGN = PageKey("static-scan")


class _Status:
    def show(self, text: str, timeout_ms: int = 0) -> None:
        return None


def test_existing_provider_isolation():
    calls = []

    class _Intents:
        def store_for(self, key):
            calls.append(("intent", key))
            return ("intent", key)

    class _Execution:
        def executor_for(self, key):
            calls.append(("executor", key))
            return ("executor", key)

    class _Sources:
        def source_port_for(self, key):
            calls.append(("source", key))
            return ("source", key)

    bound = HostServices(
        status=_Status(),
        run_intents=_Intents(),
        execution=_Execution(),
        sources=_Sources(),
        execution_profile=ExecutionProfile.TEST,
        diagnostics=DiagnosticIdentity("tests.j1"),
    ).for_page(SELECTED)
    assert bound.run_intents.store_for(SELECTED) == ("intent", SELECTED)
    assert bound.execution.executor_for(SELECTED) == ("executor", SELECTED)
    assert bound.sources.source_port_for(SELECTED) == ("source", SELECTED)
    assert bound.run_intents.store_for(FOREIGN) is None
    assert bound.execution.executor_for(FOREIGN) is None
    assert bound.sources.source_port_for(FOREIGN) is None
    assert calls == [
        ("intent", SELECTED), ("executor", SELECTED), ("source", SELECTED)]


def test_importing_services_does_not_eagerly_load_experiment_values():
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    code = (
        "import sys\n"
        "import xdart.gui.pages.services\n"
        "assert 'xrd_tools.session.experiment_state' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
