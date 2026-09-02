"""Shared oracle for bounded JSON admission used by finite operations."""

from __future__ import annotations

import json

import pytest

from xrd_tools.io.bounded_json import (
    BoundedJsonError,
    bounded_json_snapshot,
)


def _snapshot(value, **limits):
    return bounded_json_snapshot(
        value,
        role="test value",
        max_encoded_bytes=limits.get("bytes", 4096),
        max_key_bytes=limits.get("key", 64),
        max_string_bytes=limits.get("string", 64),
        max_depth=limits.get("depth", 8),
        max_nodes=limits.get("nodes", 128),
        max_children=limits.get("children", 16),
    )


def test_bounded_json_detaches_aliases_and_reports_exact_size() -> None:
    shared = {"value": [1, 2]}
    offered = [shared, shared]
    snapshot, size = _snapshot(offered)
    shared["value"][0] = 99

    assert snapshot == [{"value": [1, 2]}, {"value": [1, 2]}]
    assert snapshot[0] is not snapshot[1]
    assert size == len(json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8"))
    assert _snapshot([[0]], depth=3)[0] == [[0]]
    assert _snapshot({"éé": "éé"}, key=4, string=4)[0] == {"éé": "éé"}
    cycle = []
    cycle.append(cycle)
    with pytest.raises(BoundedJsonError, match="cycle"):
        _snapshot(cycle)


class _BombDict(dict):
    def items(self):
        raise AssertionError("untrusted hook ran")


class _BombList(list):
    def __iter__(self):
        raise AssertionError("untrusted hook ran")


@pytest.mark.parametrize(
    ("value", "limits", "reason", "message"),
    (
        ([[[0]]], {"depth": 3}, "cardinality", "depth"),
        ([1, 2, 3], {"children": 2}, "cardinality", "child"),
        ({"ééa": 1}, {"key": 4}, "bytes", "ceiling"),
        ("ééa", {"string": 4}, "bytes", "ceiling"),
        ("\ud800", {}, "schema", "surrogate"),
        (2**63, {}, "schema", "64-bit"),
        (float("nan"), {}, "schema", "finite"),
        (_BombDict(a=1), {}, "schema", "non-JSON"),
        (_BombList([1]), {}, "schema", "non-JSON"),
    ),
)
def test_bounded_json_refuses_adversarial_values(
    value, limits, reason, message,
) -> None:
    with pytest.raises(BoundedJsonError, match=message) as captured:
        _snapshot(value, **limits)
    assert captured.value.reason == reason


@pytest.mark.parametrize(
    ("value", "required_charge"),
    ((0, 20), ("\n", 8)),
    ids=("signed-integer", "escaped-control"),
)
def test_bounded_json_conservative_scalar_charge_is_an_exact_boundary(
    value, required_charge,
) -> None:
    with pytest.raises(BoundedJsonError) as captured:
        _snapshot(value, bytes=required_charge - 1)
    assert captured.value.reason == "bytes"
    assert _snapshot(value, bytes=required_charge)[0] == value


def test_prepared_json_key_ceiling_matches_the_frozen_untrusted_bound() -> None:
    import xrd_tools.reduction.reintegrate_prepared as prepared
    import xrd_tools.reduction.reintegrate_successor as successor

    assert prepared.MAX_PREPARED_KEY_BYTES == 8 << 20
    assert successor._MAX_RECIPE_KEY_BYTES == 8 << 20
