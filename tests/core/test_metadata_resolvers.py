from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.metadata import (
    INCIDENCE_MOTOR_SEARCH_ORDER,
    IncidenceAngleUnresolved,
    resolve_incident_angle,
    resolve_monitor_norm,
)
from xrd_tools.core.strictness import StrictnessError


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def test_resolve_incident_angle_uses_manual_numeric_before_metadata() -> None:
    assert resolve_incident_angle({"th": 9.0}, "0.25") == pytest.approx(0.25)


def test_resolve_incident_angle_uses_explicit_motor_case_insensitively() -> None:
    assert resolve_incident_angle({"TH": 1.0, "Eta": 2.0}, "eta") == pytest.approx(2.0)


def test_resolve_incident_angle_default_search_order() -> None:
    assert INCIDENCE_MOTOR_SEARCH_ORDER == (
        "th",
        "theta",
        "eta",
        "halpha",
        "gth",
        "gonth",
    )
    for offset, expected_motor in enumerate(INCIDENCE_MOTOR_SEARCH_ORDER):
        metadata = {
            motor.upper(): float(index)
            for index, motor in enumerate(INCIDENCE_MOTOR_SEARCH_ORDER[offset:], start=offset)
        }
        assert resolve_incident_angle(metadata, None) == pytest.approx(float(offset))
        assert expected_motor.upper() in metadata


def test_resolve_incident_angle_unresolved_is_strictness_error() -> None:
    assert issubclass(IncidenceAngleUnresolved, StrictnessError)
    with pytest.raises(IncidenceAngleUnresolved):
        resolve_incident_angle({"i0": 100.0}, "th")


def test_resolve_monitor_norm_uses_case_insensitive_positive_value() -> None:
    assert resolve_monitor_norm({"I0": "12.5"}, "i0") == pytest.approx(12.5)


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, np.nan, np.inf, -np.inf, "open", None],
)
def test_resolve_monitor_norm_returns_none_for_unusable_values(value: object) -> None:
    assert resolve_monitor_norm({"I0": value}, "i0") is None


def test_resolve_monitor_norm_returns_none_without_configured_monitor() -> None:
    assert resolve_monitor_norm({"I0": 12.5}, None) is None
    assert resolve_monitor_norm({"I0": 12.5}, "") is None
    assert resolve_monitor_norm({"I0": 12.5}, "missing") is None


def test_metadata_resolvers_import_without_gui_or_heavy_reader_stack() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SRC)
        if not env.get("PYTHONPATH")
        else os.pathsep.join([str(SRC), env["PYTHONPATH"]])
    )
    code = textwrap.dedent(
        """
        import importlib
        import sys

        before = set(sys.modules)
        module = importlib.import_module("xrd_tools.core.metadata")
        assert module.resolve_incident_angle({"th": 0.1}, None) == 0.1
        loaded = set(sys.modules) - before
        forbidden_roots = (
            "PySide",
            "PyQt",
            "qtpy",
            "pyqtgraph",
            "pyFAI",
            "h5py",
            "fabio",
            "xdart",
        )
        forbidden = sorted(
            name
            for name in loaded
            if any(name == root or name.startswith(root + ".") for root in forbidden_roots)
        )
        if forbidden:
            raise SystemExit("\\n".join(forbidden))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
