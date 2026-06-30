"""A-prep1: pin the store-first ``get -> project(mode_1d, mode_2d) -> FrameView``
contract that the later D3 one-store GUI accessor will be a thin wrapper over.

These are headless, Qt-free, synthetic-data tests.  They lock two things the
GUI store-first read will rely on:

1.  A multi-mode :class:`FrameRecord` (1D and 2D results across the canonical GI
    sub-modes) projects, for EVERY ``(mode_1d, mode_2d)`` combination, to a
    :class:`FrameView` that is byte-equivalent (via
    :func:`assert_frameview_equivalent`) to the view built directly from the
    source pyFAI results — i.e. the dimension-pure per-mode split + merge in
    :meth:`FrameRecord.project` is lossless.
2.  A ``store.upsert`` -> ``store.get`` -> ``project`` round-trip preserves the
    full per-mode dict (every 1D mode crossed with every 2D mode), so collapsing
    the GUI onto :class:`FrameRecordStore` does not lose a GI sub-mode.

The single-mode projection round-trip already lives in ``test_frame_record.py``;
this file is the multi-mode / store-fronted superset that the GUI wrapper pins
against.
"""

from __future__ import annotations

import itertools

import numpy as np

from xrd_tools.core import (
    FrameRecord,
    FrameView,
    IntegrationResult1D,
    IntegrationResult2D,
    TwoDKind,
    assert_frameview_equivalent,
)
from xrd_tools.session import FrameRecordStore

# Canonical GI sub-mode keys (mirror xrd_tools.io.schema.GI_MODE_KEYS_*; the
# FrameEvent.mode_key vocabulary).  Each maps to a distinct pyFAI unit pair so
# the projected FrameView carries a distinguishable axis identity + TwoDKind.
_MODES_1D: dict[str, str] = {
    "q_total": "qtot_A^-1",
    "q_ip": "qip_A^-1",
    "q_oop": "qoop_A^-1",
    "exit_angle": "exit_angle_vert_deg",
}
_MODES_2D: dict[str, tuple[str, str, TwoDKind]] = {
    "q_chi": ("q_A^-1", "chi_deg", TwoDKind.Q_CHI),
    "qip_qoop": ("qip_A^-1", "qoop_A^-1", TwoDKind.QIP_QOOP),
    "exit_angles": ("exit_angle_horz_deg", "exit_angle_vert_deg", TwoDKind.EXIT_ANGLES),
}

_SOURCE = "/data/gi_scan_0001.h5"
_FRAME_INDEX = 4
_LABEL = 4

_METADATA = {"monitor": 12.0, "incident_angle": 0.18, "sample": "thin_film"}
_INCIDENT_ANGLE = 0.18


def _result_1d(unit: str, *, scale: float) -> IntegrationResult1D:
    radial = np.linspace(0.5, 5.0, 6)
    intensity = scale * np.linspace(10.0, 60.0, 6)
    return IntegrationResult1D(
        radial=radial,
        intensity=intensity,
        sigma=np.sqrt(intensity),
        unit=unit,
    )


def _result_2d(x_unit: str, y_unit: str, *, scale: float) -> IntegrationResult2D:
    radial = np.linspace(-1.0, 2.0, 4)
    azimuthal = np.linspace(0.0, 3.0, 3)
    # IntegrationResult2D.intensity is (radial, azimuthal) = (len(x), len(y)).
    intensity = scale * np.arange(12, dtype=float).reshape(4, 3)
    return IntegrationResult2D(
        radial=radial,
        azimuthal=azimuthal,
        intensity=intensity,
        sigma=np.sqrt(intensity + 1.0),
        unit=x_unit,
        azimuthal_unit=y_unit,
    )


def _view_for_mode_1d(mode: str) -> FrameView:
    scale = 1.0 + 0.5 * list(_MODES_1D).index(mode)
    return FrameView.from_results(
        label=_LABEL,
        result_1d=_result_1d(_MODES_1D[mode], scale=scale),
        metadata_raw=_METADATA,
        incident_angle=_INCIDENT_ANGLE,
        source_path=_SOURCE,
        source_frame_index=_FRAME_INDEX,
    )


def _view_for_mode_2d(mode: str) -> FrameView:
    x_unit, y_unit, _kind = _MODES_2D[mode]
    scale = 1.0 + 0.25 * list(_MODES_2D).index(mode)
    return FrameView.from_results(
        label=_LABEL,
        result_2d=_result_2d(x_unit, y_unit, scale=scale),
        metadata_raw=_METADATA,
        incident_angle=_INCIDENT_ANGLE,
        source_path=_SOURCE,
        source_frame_index=_FRAME_INDEX,
    )


def _build_multimode_record() -> FrameRecord:
    """A frame record carrying every GI 1D mode AND every GI 2D mode."""
    modes_1d = list(_MODES_1D)
    modes_2d = list(_MODES_2D)
    rec = FrameRecord.from_view(
        _view_for_mode_1d(modes_1d[0]),
        mode_1d=modes_1d[0],
    )
    for mode in modes_1d[1:]:
        rec = rec.with_result_1d(mode, _view_for_mode_1d(mode), make_active=False)
    for i, mode in enumerate(modes_2d):
        rec = rec.with_result_2d(
            mode, _view_for_mode_2d(mode), make_active=(i == 0)
        )
    return rec


def _expected_combined_view(mode_1d: str, mode_2d: str) -> FrameView:
    """The combined view ``project(mode_1d, mode_2d)`` should reproduce.

    Built directly from the source pyFAI results via ``from_results`` — the
    single source of truth ``assert_frameview_equivalent`` compares against.
    ``assert_frameview_equivalent`` compares the FULL view (both legs at once),
    so the expected view must carry both the 1D and the 2D source legs.
    """
    scale_1d = 1.0 + 0.5 * list(_MODES_1D).index(mode_1d)
    x_unit, y_unit, _kind = _MODES_2D[mode_2d]
    scale_2d = 1.0 + 0.25 * list(_MODES_2D).index(mode_2d)
    return FrameView.from_results(
        label=_LABEL,
        result_1d=_result_1d(_MODES_1D[mode_1d], scale=scale_1d),
        result_2d=_result_2d(x_unit, y_unit, scale=scale_2d),
        metadata_raw=_METADATA,
        incident_angle=_INCIDENT_ANGLE,
        source_path=_SOURCE,
        source_frame_index=_FRAME_INDEX,
    )


# --------------------------------------------------------------------------- #
# project(mode_1d, mode_2d) cross product is byte-equivalent to source results
# --------------------------------------------------------------------------- #

def test_record_carries_every_gi_submode():
    rec = _build_multimode_record()
    assert set(rec.modes_1d) == set(_MODES_1D)
    assert set(rec.modes_2d) == set(_MODES_2D)


def test_project_cross_product_is_byte_equivalent_to_source_results():
    rec = _build_multimode_record()

    for mode_1d, mode_2d in itertools.product(_MODES_1D, _MODES_2D):
        projected = rec.project(mode_1d=mode_1d, mode_2d=mode_2d)

        # The 1D leg carries the selected 1D mode's axis identity...
        assert projected.axis_1d is not None
        assert projected.axis_1d.unit == _MODES_1D[mode_1d]
        # ...and the 2D leg carries the selected 2D mode's axes + kind (incl.
        # orientation: FrameView.intensity_2d is the .T of result.intensity).
        x_unit, y_unit, kind = _MODES_2D[mode_2d]
        assert projected.has_2d
        assert projected.axis_2d_x.unit == x_unit
        assert projected.axis_2d_y.unit == y_unit
        assert projected.two_d_kind is kind

        # Byte-equivalent to the combined view built straight from the sources.
        assert_frameview_equivalent(
            projected, _expected_combined_view(mode_1d, mode_2d)
        )


def test_project_preserves_2d_orientation_against_pyfai_transpose():
    # Pin the (radial, azimuthal) -> (y, x) orientation through project(): the
    # projected intensity_2d must be the transpose of the pyFAI result.intensity.
    rec = _build_multimode_record()
    for mode_2d in _MODES_2D:
        x_unit, y_unit, _kind = _MODES_2D[mode_2d]
        scale = 1.0 + 0.25 * list(_MODES_2D).index(mode_2d)
        src = _result_2d(x_unit, y_unit, scale=scale)
        projected = rec.project(mode_1d="q_total", mode_2d=mode_2d)
        np.testing.assert_allclose(projected.intensity_2d, src.intensity.T)
        np.testing.assert_allclose(projected.sigma_2d, src.sigma.T)


# --------------------------------------------------------------------------- #
# store-first get -> project round-trip preserves the per-mode dict
# --------------------------------------------------------------------------- #

def test_store_get_project_roundtrip_preserves_every_submode():
    # This is the contract the later GUI store-first accessor wraps: upsert a
    # multi-mode record, get it back, and assert EVERY (mode_1d, mode_2d) still
    # projects byte-equivalent to its source result.  No mode is lost or merged.
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(_build_multimode_record())

    rec = store.get(_LABEL)
    assert rec is not None
    assert set(rec.modes_1d) == set(_MODES_1D)
    assert set(rec.modes_2d) == set(_MODES_2D)

    for mode_1d, mode_2d in itertools.product(_MODES_1D, _MODES_2D):
        projected = rec.project(mode_1d=mode_1d, mode_2d=mode_2d)
        assert_frameview_equivalent(
            projected, _expected_combined_view(mode_1d, mode_2d)
        )


def test_store_accumulated_upserts_match_one_shot_record():
    # The GUI accumulates modes via repeated single-mode upserts (one per
    # completed FrameEvent) for the same source.  The store-merged record must
    # project identically to a one-shot record carrying all modes at once.
    store = FrameRecordStore(max_heavy_items=None)
    for mode in _MODES_1D:
        store.upsert(
            FrameRecord.from_view(_view_for_mode_1d(mode), mode_1d=mode)
        )
    for mode in _MODES_2D:
        store.upsert(
            FrameRecord.from_view(_view_for_mode_2d(mode), mode_2d=mode)
        )

    rec = store.get(_LABEL)
    assert rec is not None
    assert set(rec.modes_1d) == set(_MODES_1D)
    assert set(rec.modes_2d) == set(_MODES_2D)
    for mode_1d, mode_2d in itertools.product(_MODES_1D, _MODES_2D):
        projected = rec.project(mode_1d=mode_1d, mode_2d=mode_2d)
        assert_frameview_equivalent(
            projected, _expected_combined_view(mode_1d, mode_2d)
        )


def test_store_active_view_projects_active_modes():
    # get(...).active_view() == project() of the active 1D + 2D modes: the
    # default the GUI store-first accessor renders without an explicit selection.
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(_build_multimode_record())
    rec = store.get(_LABEL)

    active = rec.active_view()
    expected_active = rec.project(rec.active_mode_1d, rec.active_mode_2d)
    assert_frameview_equivalent(active, expected_active)
    # The active 2D mode was set to the first ("q_chi") in _build_multimode_record.
    assert rec.active_mode_2d == "q_chi"
    assert active.two_d_kind is TwoDKind.Q_CHI
