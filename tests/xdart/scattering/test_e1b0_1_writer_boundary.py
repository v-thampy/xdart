from __future__ import annotations

import numpy as np
import pytest

import xrd_tools.reduction.core as reduction_core
from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.core.provenance import read_provenance
from xrd_tools.reduction import (
    Frame,
    Integration1DPlan,
    NexusSink,
    ReductionPlan,
    Scan,
    run_reduction,
)
from xrd_tools.session.run_configuration import RunIntent


class _Authority:
    """A legal Python object whose deepcopy deliberately preserves identity."""

    def __deepcopy__(self, memo):
        return self


def _identity(**extra):
    return {
        "schema_version": 1,
        "generation": 1,
        "fingerprint": "f",
        **extra,
    }


def _assert_json_native(value):
    if value is None or type(value) in (bool, int, float, str):
        return
    if type(value) is list:
        for item in value:
            _assert_json_native(item)
        return
    assert type(value) is dict
    for key, item in value.items():
        assert type(key) is str
        _assert_json_native(item)


def _write_provenance(tmp_path, monkeypatch, sink):
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: IntegrationResult1D(
            radial=np.array([0.0, 1.0]),
            intensity=np.array([1.0, 2.0]),
            sigma=np.array([0.1, 0.2]),
            unit="q_A^-1",
        ),
    )
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=2,
            unit="q_A^-1",
            method="csr",
            radial_range=(0.0, 1.0),
        ),
        integration_2d=None,
    )
    result = run_reduction(
        plan,
        Scan("provenance", [Frame(0, image=np.ones((2, 2)))], integrator=object()),
        sink,
    )
    assert result.n_processed == 1
    return sink.path


def test_sink_rejects_callback_in_provenance_value_tree(tmp_path):
    callback = lambda: None

    try:
        sink = NexusSink(
            tmp_path / "callback.nxs",
            run_configuration_provenance=_identity(callback=callback),
        )
    except (TypeError, ValueError):
        return

    assert sink._run_configuration_provenance["callback"] is not callback


def test_sink_rejects_live_authority_that_survives_deepcopy(tmp_path):
    authority = _Authority()

    try:
        sink = NexusSink(
            tmp_path / "authority.nxs",
            run_configuration_provenance=_identity(authority=authority),
        )
    except (TypeError, ValueError):
        return

    assert sink._run_configuration_provenance["authority"] is not authority


def test_full_accepted_configuration_value_algebra_round_trips(
    tmp_path,
    monkeypatch,
):
    configuration = RunIntent(
        bai_1d_args={
            "radial_range": (0.0, 5.0),
            "method": ("full", "histogram", "cython"),
        },
    ).freeze()
    projected = configuration.as_provenance()
    _assert_json_native(projected)
    path = _write_provenance(
        tmp_path,
        monkeypatch,
        NexusSink(
            tmp_path / "tuple-values.nxs",
            overwrite=True,
            run_configuration_provenance=projected,
        ),
    )

    assert (
        read_provenance(path)["config"]["run_configuration"]
        == projected
    )


@pytest.mark.parametrize("value", [lambda: None, _Authority(), ("not", "json-native")])
def test_sink_refuses_non_json_native_authority_values_before_output(tmp_path, value):
    with pytest.raises(ValueError):
        NexusSink(
            tmp_path / "refused.nxs",
            run_configuration_provenance=_identity(value=value),
        )
