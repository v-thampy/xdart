from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.xdart._accepted_run import accepted_run, admitted_worker
from xdart.gui.tabs.static_scan.static_scan_widget import _accepted_run_policy


def test_admitted_worker_binds_exact_identity_floor_and_policy():
    frozen = accepted_run(batch_mode=True)
    target = admitted_worker(
        SimpleNamespace(operational_field="preserved"),
        frozen=frozen,
    )

    assert target.run_configuration is frozen
    assert target._admitted_run_configuration is frozen
    assert target.run_configuration_floor == frozen.generation
    assert _accepted_run_policy(target) is frozen
    assert target.operational_field == "preserved"


@pytest.mark.parametrize(
    "reserved",
    (
        "run_configuration",
        "_admitted_run_configuration",
        "run_configuration_floor",
    ),
)
def test_admitted_worker_rejects_reserved_keyword_overrides(reserved):
    with pytest.raises(TypeError, match="owns admission fields"):
        admitted_worker(**{reserved: object()})


def test_equal_but_distinct_ledger_is_not_accepted_policy():
    frozen = accepted_run(batch_mode=True)
    target = admitted_worker(frozen=frozen)
    foreign = replace(frozen)
    assert foreign == frozen
    assert foreign is not frozen

    target._admitted_run_configuration = foreign

    assert _accepted_run_policy(target) is None


def test_carrier_without_admission_ledger_is_not_accepted_policy():
    frozen = accepted_run(batch_mode=True)
    target = SimpleNamespace(run_configuration=frozen)

    assert _accepted_run_policy(target) is None


def test_accepted_run_helper_import_remains_qt_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import tests.xdart._accepted_run; "
                "assert not any(name.startswith('xdart.gui') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
