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


def test_public_finite_motor_metadata_keeps_image_policy_without_mutating_values():
    from xrd_tools.core.metadata import finite_motor_metadata

    vector = np.array([1., 2.])
    raw = {"eta": "0.25", "th": np.float64(.1), "ROI1": 9, "sample_PD": 8,
           "bool": True, "npbool": np.bool_(False), "text": "missing", "vector": vector,
           "nan": np.nan, "inf": np.inf, "": 1, "zero": 0, "negative": -3}
    assert finite_motor_metadata(raw) == {"eta": .25, "th": .1, "zero": 0., "negative": -3.}
    assert raw["vector"] is vector and raw["eta"] == "0.25"


@pytest.mark.parametrize("catalogs,expected", [
    ((), None), ((None, ("th",)), None), (((), ("th",)), ()),
    ((("eta", "th", "Manual", ""), ("th", "eta", "Manual")), ("eta", "th")),
    ((("manual", "Manual", "th"),), ("manual", "th")),
])
def test_public_motor_intersection_keeps_order_and_unknown_empty_distinction(catalogs, expected):
    from xrd_tools.core.metadata import ordered_motor_intersection

    assert ordered_motor_intersection(iter(catalogs)) == expected


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
        assert module.finite_motor_metadata({"th": "0.1", "ROI": 4}) == {"th": 0.1}
        assert module.ordered_motor_intersection((("th", "eta"), ("eta",))) == ("eta",)
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


def test_monitor_candidates_list_likely_monitors_first_and_drop_the_never_monitors():
    """Stitch's Monitor and Int's Norm channel (maintainer, 2026-09-30): \"if in doubt, list it. if it
    clearly not a monitor ... it can be dropped\"; a choice already made is kept."""
    from xrd_tools.core.metadata import monitor_candidates
    names = ("Two Theta", "H", "K", "L", "Epoch", "Seconds", "Seconds", "mon", "det", "I0", "pd1",
             "bstop", "Beamstop", "ic1", "temp")
    assert monitor_candidates(names) == ("mon", "I0", "pd1", "bstop", "Beamstop", "ic1", "Two Theta",
                                         "Seconds", "det", "temp")
    assert monitor_candidates(("epoch", "i1"), keep=("epoch",)) == ("i1", "epoch")
