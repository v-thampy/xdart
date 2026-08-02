# -*- coding: utf-8 -*-
"""Frozen suite for the Slice-5 ``ScanNormAggregate`` canonical value.

Covers the E5-V §3.2 contract: exact-identity frozen value, incremental
fold with the one ABSENT rule (``resolve_monitor_norm``), fold does not
advance the revision, monotonic consumer acceptance with idempotent
equal-revision replay, loud validation, and Qt/h5py/pandas import purity.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from xrd_tools.core.metadata import resolve_monitor_norm
from xrd_tools.session.scan_norm import (
    ScanNormAggregate,
    accepts_norm_aggregate,
    channel_is_partial,
    empty_norm_aggregate,
    fold_norm_metadata,
    next_norm_revision,
)

ACQ_IDENTITY = (7, "sig-1f2e", "/data/run7.nxs", "scan001")
BROWSE_IDENTITY = ("ctx-01", "scan001", "/data/old.nxs")


def test_empty_aggregate_shape():
    agg = empty_norm_aggregate(ACQ_IDENTITY)
    assert agg.identity == ACQ_IDENTITY
    assert agg.revision == 0
    assert agg.row_count == 0
    assert dict(agg.channels) == {}


def test_empty_normalizes_identity_sequence_to_tuple():
    agg = empty_norm_aggregate(list(ACQ_IDENTITY))
    assert agg.identity == ACQ_IDENTITY
    assert isinstance(agg.identity, tuple)


def test_fold_accumulates_finite_positive_channels():
    agg = empty_norm_aggregate(ACQ_IDENTITY)
    agg = fold_norm_metadata(agg, {"Monitor": 2.0, "i0": 4.5})
    agg = fold_norm_metadata(agg, {"Monitor": 3.0, "i0": 0.5})
    assert agg.row_count == 2
    assert agg.channels["monitor"] == (5.0, 2)
    assert agg.channels["i0"] == (5.0, 2)


def test_fold_canonicalizes_case_aliases_across_rows():
    agg = empty_norm_aggregate(ACQ_IDENTITY)
    agg = fold_norm_metadata(agg, {"Monitor": 2.0})
    agg = fold_norm_metadata(agg, {"monitor": 3.0})

    assert dict(agg.channels) == {"monitor": (5.0, 2)}
    assert channel_is_partial(agg, "monitor") is False


def test_fold_counts_same_row_case_alias_once_via_the_existing_kernel():
    metadata = {"Monitor": 2.0, "monitor": 3.0}
    expected = resolve_monitor_norm(metadata, "monitor")
    assert expected == 2.0  # the shared kernel's deterministic first match

    agg = fold_norm_metadata(empty_norm_aggregate(ACQ_IDENTITY), metadata)

    assert dict(agg.channels) == {"monitor": (expected, 1)}


def test_canonical_channel_order_is_deterministic():
    left = fold_norm_metadata(
        empty_norm_aggregate(ACQ_IDENTITY), {"Z": 1.0, "a": 2.0}
    )
    right = fold_norm_metadata(
        empty_norm_aggregate(ACQ_IDENTITY), {"a": 2.0, "Z": 1.0}
    )

    assert tuple(left.channels) == ("a", "z")
    assert tuple(right.channels) == tuple(left.channels)


def test_fold_applies_the_absent_rule_per_kernel():
    # zero, negative, nonnumeric and non-finite values contribute to
    # row_count but never to a channel (resolve_monitor_norm returns None).
    agg = empty_norm_aggregate(ACQ_IDENTITY)
    agg = fold_norm_metadata(
        agg,
        {"zero": 0.0, "neg": -3.0, "text": "abc", "inf": float("inf")},
    )
    assert agg.row_count == 1
    assert dict(agg.channels) == {}


def test_fold_with_empty_or_none_metadata_counts_the_row_only():
    agg = empty_norm_aggregate(ACQ_IDENTITY)
    agg = fold_norm_metadata(agg, {})
    agg = fold_norm_metadata(agg, None)
    assert agg.row_count == 2
    assert dict(agg.channels) == {}


def test_channel_appears_only_after_first_valid_row_and_partial_rule():
    agg = empty_norm_aggregate(ACQ_IDENTITY)
    agg = fold_norm_metadata(agg, {"Monitor": 0.0})  # invalid -> no channel
    assert "monitor" not in agg.channels
    agg = fold_norm_metadata(agg, {"Monitor": 1.5})
    assert agg.channels["monitor"] == (1.5, 1)
    assert agg.row_count == 2
    assert channel_is_partial(agg, "monitor") is True
    agg2 = fold_norm_metadata(
        fold_norm_metadata(empty_norm_aggregate(ACQ_IDENTITY), {"m": 1.0}),
        {"m": 2.0},
    )
    assert channel_is_partial(agg2, "m") is False


def test_channel_is_partial_is_loud_for_absent_channels():
    agg = empty_norm_aggregate(ACQ_IDENTITY)
    with pytest.raises(KeyError):
        channel_is_partial(agg, "Monitor")


def test_fold_does_not_advance_the_revision():
    agg = empty_norm_aggregate(ACQ_IDENTITY)
    for i in range(5):
        agg = fold_norm_metadata(agg, {"Monitor": 1.0 + i})
    assert agg.revision == 0


def test_next_revision_bumps_and_preserves_content():
    agg = fold_norm_metadata(empty_norm_aggregate(ACQ_IDENTITY), {"m": 2.0})
    bumped = next_norm_revision(agg)
    assert bumped.revision == agg.revision + 1
    assert bumped.identity == agg.identity
    assert bumped.row_count == agg.row_count
    assert dict(bumped.channels) == dict(agg.channels)
    assert agg.revision == 0  # original unchanged


def test_value_is_frozen_and_channels_immutable():
    agg = fold_norm_metadata(empty_norm_aggregate(ACQ_IDENTITY), {"m": 2.0})
    with pytest.raises(AttributeError):
        agg.revision = 3
    with pytest.raises(TypeError):
        agg.channels["m"] = (1.0, 1)


def test_construction_copies_channels_defensively():
    src = {"m": (2.0, 1)}
    agg = ScanNormAggregate(ACQ_IDENTITY, 1, 1, src)
    src["rogue"] = (1.0, 1)
    assert "rogue" not in agg.channels


def test_acceptance_gate_monotonic_per_exact_identity():
    current = next_norm_revision(
        fold_norm_metadata(empty_norm_aggregate(ACQ_IDENTITY), {"m": 1.0})
    )  # revision 1
    newer = next_norm_revision(current)  # revision 2
    assert accepts_norm_aggregate(None, current, ACQ_IDENTITY) is True
    assert accepts_norm_aggregate(current, newer, ACQ_IDENTITY) is True
    # equal-revision replay is an idempotent no-op
    assert accepts_norm_aggregate(current, current, ACQ_IDENTITY) is False
    # older revision rejected
    assert accepts_norm_aggregate(newer, current, ACQ_IDENTITY) is False


def test_acceptance_rejects_foreign_identity_regardless_of_revision():
    foreign = next_norm_revision(
        next_norm_revision(empty_norm_aggregate(BROWSE_IDENTITY))
    )
    current = empty_norm_aggregate(ACQ_IDENTITY)
    assert accepts_norm_aggregate(current, foreign, ACQ_IDENTITY) is False
    assert accepts_norm_aggregate(None, foreign, ACQ_IDENTITY) is False


def test_acceptance_treats_identity_switch_as_fresh():
    stale = next_norm_revision(
        next_norm_revision(next_norm_revision(empty_norm_aggregate(ACQ_IDENTITY)))
    )  # revision 3 of the OLD context
    candidate = next_norm_revision(empty_norm_aggregate(BROWSE_IDENTITY))  # rev 1
    assert accepts_norm_aggregate(stale, candidate, BROWSE_IDENTITY) is True


def test_validation_is_loud():
    with pytest.raises(ValueError):
        ScanNormAggregate(ACQ_IDENTITY, -1, 0, {})
    with pytest.raises(ValueError):
        ScanNormAggregate(ACQ_IDENTITY, 0, -1, {})
    with pytest.raises(TypeError):
        ScanNormAggregate((object(),), 0, 0, {})
    with pytest.raises(TypeError):
        empty_norm_aggregate("not-a-sequence-of-fields")
    with pytest.raises(ValueError):
        ScanNormAggregate(ACQ_IDENTITY, 0, 1, {"m": (float("nan"), 1)})
    with pytest.raises(ValueError):
        ScanNormAggregate(ACQ_IDENTITY, 0, 1, {"m": (2.0, 0)})
    with pytest.raises(ValueError):
        # finite_count may not exceed row_count
        ScanNormAggregate(ACQ_IDENTITY, 0, 1, {"m": (2.0, 2)})
    with pytest.raises(TypeError):
        ScanNormAggregate(ACQ_IDENTITY, 0, 1, {3: (2.0, 1)})


def test_fold_rejects_foreign_row_types_loudly():
    agg = empty_norm_aggregate(ACQ_IDENTITY)
    with pytest.raises(TypeError):
        fold_norm_metadata(agg, ["not", "a", "mapping"])


def test_case_insensitive_kernel_reuse():
    # Aggregate keys use the same lowercase equivalence as the shared kernel.
    agg = fold_norm_metadata(empty_norm_aggregate(ACQ_IDENTITY), {"MONITOR": 2.0})
    assert agg.channels["monitor"] == (2.0, 1)


def test_import_purity_no_qt_h5py_pandas():
    code = (
        "import sys; import xrd_tools.session.scan_norm; "
        "bad = [m for m in ('PySide6', 'PyQt5', 'h5py', 'pandas', 'fabio', 'pyFAI') "
        "if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()


def test_lazy_session_export():
    import xrd_tools.session as session

    assert session.ScanNormAggregate is ScanNormAggregate
    assert "ScanNormAggregate" in session.__all__
