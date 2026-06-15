"""Unit tests for the multi-result :class:`FrameRecord` (ADR-0003 / ADR-0005).

Pure, GUI-free.  Covers the record shape, dimension-pure storage, the
``from_view`` bridge, immutable upserts, projection/merge round-trips, and the
multi-mode equivalence atom :func:`assert_framerecord_equivalent`.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core import (
    DEFAULT_MODE_KEY,
    Axis,
    FrameRecord,
    FrameView,
    TwoDKind,
    assert_framerecord_equivalent,
    assert_frameview_equivalent,
)


def _view(label=0, *, with_1d=True, with_2d=True, scale=1.0):
    kw = {}
    if with_1d:
        kw.update(
            axis_1d=Axis("Q", "q_A^-1", values=np.array([1.0, 2.0, 3.0])),
            intensity_1d=np.array([10.0, 20.0, 30.0]) * scale,
        )
    if with_2d:
        kw.update(
            axis_2d_x=Axis("Q", "q_A^-1", values=np.array([1.0, 2.0])),
            axis_2d_y=Axis("χ", "chi_deg", values=np.array([0.0, 1.0, 2.0])),
            intensity_2d=np.arange(6.0).reshape(3, 2) * scale,
            two_d_kind=TwoDKind.Q_CHI,
        )
    return FrameView(
        label=label,
        raw=np.ones((4, 4)),
        incident_angle=0.2,
        metadata_raw={"motor": 1.0},
        **kw,
    )


# --------------------------------------------------------------------------- #
# from_view bridge + projection round-trip
# --------------------------------------------------------------------------- #

def test_from_view_default_keys_and_project_roundtrip():
    view = _view()
    rec = FrameRecord.from_view(view)

    assert rec.label == 0
    assert rec.modes_1d == (DEFAULT_MODE_KEY,)
    assert rec.modes_2d == (DEFAULT_MODE_KEY,)
    assert rec.active_mode_1d == DEFAULT_MODE_KEY
    assert rec.active_mode_2d == DEFAULT_MODE_KEY

    # project() merges the active 1D + 2D modes back to an equivalent view.
    assert_frameview_equivalent(rec.project(), view)
    assert_frameview_equivalent(rec.active_view(), view)


def test_from_view_stores_dimension_pure_entries():
    rec = FrameRecord.from_view(_view())
    v1 = rec.view_1d()
    v2 = rec.view_2d()
    # 1D entry carries no 2D arrays; 2D entry carries no 1D arrays.
    assert v1.has_1d and not v1.has_2d
    assert v2.has_2d and not v2.has_1d
    # Shared per-frame fields survive on both pure views.
    assert v1.raw is not None and v2.raw is not None
    assert v1.incident_angle == pytest.approx(0.2)


def test_from_view_one_dimensional_only():
    rec = FrameRecord.from_view(_view(with_2d=False))
    assert rec.modes_1d == (DEFAULT_MODE_KEY,)
    assert rec.modes_2d == ()
    assert rec.active_mode_2d == DEFAULT_MODE_KEY
    assert rec.view_2d() is None
    # projecting a 1D-only record yields the 1D view.
    proj = rec.project()
    assert proj.has_1d and not proj.has_2d


# --------------------------------------------------------------------------- #
# multi-mode upserts
# --------------------------------------------------------------------------- #

def test_with_result_upsert_accumulates_modes_and_tracks_active():
    rec = FrameRecord.from_view(_view(), mode_1d="q_total", mode_2d="q_chi")
    rec2 = rec.with_result_1d("q_ip", _view(scale=2.0))

    # original is unchanged (immutable upsert)
    assert rec.modes_1d == ("q_total",)
    assert set(rec2.modes_1d) == {"q_total", "q_ip"}
    assert rec2.active_mode_1d == "q_ip"
    # 2D map untouched by a 1D upsert
    assert rec2.modes_2d == ("q_chi",)

    # the new 1D mode renders its own intensity
    proj_ip = rec2.project(mode_1d="q_ip")
    np.testing.assert_allclose(proj_ip.intensity_1d, np.array([20.0, 40.0, 60.0]))
    # switching back to q_total is the cached original
    proj_tot = rec2.project(mode_1d="q_total")
    np.testing.assert_allclose(proj_tot.intensity_1d, np.array([10.0, 20.0, 30.0]))


def test_with_result_make_active_false_keeps_active():
    rec = FrameRecord.from_view(_view(), mode_1d="q_total")
    rec2 = rec.with_result_1d("q_ip", _view(scale=3.0), make_active=False)
    assert rec2.active_mode_1d == "q_total"
    assert set(rec2.modes_1d) == {"q_total", "q_ip"}


def test_with_result_2d_is_dimension_pure():
    rec = FrameRecord.from_view(_view(), mode_2d="q_chi")
    rec2 = rec.with_result_2d("qip_qoop", _view(scale=2.0))
    stored = rec2.view_2d("qip_qoop")
    assert stored.has_2d and not stored.has_1d


# --------------------------------------------------------------------------- #
# validation + immutability
# --------------------------------------------------------------------------- #

def test_active_mode_must_exist_in_results():
    v = FrameRecord.from_view(_view(with_2d=False)).view_1d()
    with pytest.raises(ValueError, match="active_mode_1d"):
        FrameRecord(label=0, results_1d={"q_total": v}, active_mode_1d="missing")


def test_direct_construction_enforces_dimension_purity():
    # The reload path constructs records directly from disk-read views; an
    # impure (combined) view must be made dimension-pure by the constructor.
    impure = _view()  # carries both 1D and 2D
    rec = FrameRecord(
        label=0,
        results_1d={"q_total": impure},
        results_2d={"q_chi": impure},
        active_mode_1d="q_total",
        active_mode_2d="q_chi",
    )
    assert rec.view_1d("q_total").has_1d and not rec.view_1d("q_total").has_2d
    assert rec.view_2d("q_chi").has_2d and not rec.view_2d("q_chi").has_1d
    # ...and projection merges them back to the original combined view.
    assert_frameview_equivalent(rec.project(), impure)


def test_project_single_dimension_does_not_leak_other_dimension():
    # A 2D-only record built directly from a combined view must not leak 1D.
    rec = FrameRecord(label=0, results_2d={"q_chi": _view()}, active_mode_2d="q_chi")
    proj = rec.project()
    assert proj.has_2d and not proj.has_1d


def test_non_frameview_value_rejected():
    with pytest.raises(TypeError):
        FrameRecord(label=0, results_1d={"q_total": object()})  # type: ignore[dict-item]


def test_empty_record_project_raises():
    rec = FrameRecord(label=0)
    assert rec.is_empty
    with pytest.raises(ValueError, match="empty FrameRecord"):
        rec.project()


def test_record_is_frozen():
    rec = FrameRecord.from_view(_view())
    with pytest.raises(Exception):
        rec.active_mode_1d = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# equivalence atom
# --------------------------------------------------------------------------- #

def test_assert_framerecord_equivalent_passes_for_equal_records():
    a = FrameRecord.from_view(_view(), mode_1d="q_total").with_result_1d(
        "q_ip", _view(scale=2.0)
    )
    b = FrameRecord.from_view(_view(), mode_1d="q_total").with_result_1d(
        "q_ip", _view(scale=2.0)
    )
    assert_framerecord_equivalent(a, b)


def test_assert_framerecord_equivalent_detects_mode_set_mismatch():
    a = FrameRecord.from_view(_view(), mode_1d="q_total")
    # keep the active mode equal so the mode-SET check is what fires
    b = a.with_result_1d("q_ip", _view(scale=2.0), make_active=False)
    with pytest.raises(AssertionError, match="mode sets differ"):
        assert_framerecord_equivalent(a, b)


def test_assert_framerecord_equivalent_detects_active_mode_mismatch():
    a = FrameRecord.from_view(_view(), mode_1d="q_total")
    b = FrameRecord.from_view(_view(), mode_1d="q_ip")
    with pytest.raises(AssertionError, match="active_mode_1d"):
        assert_framerecord_equivalent(a, b)


def test_assert_framerecord_equivalent_detects_value_mismatch():
    a = FrameRecord.from_view(_view(), mode_1d="q_total")
    b = FrameRecord.from_view(_view(scale=5.0), mode_1d="q_total")
    with pytest.raises(AssertionError, match="results_1d"):
        assert_framerecord_equivalent(a, b)
