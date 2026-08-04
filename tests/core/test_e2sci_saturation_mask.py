"""E2-SCI: exact saturation-mask policy and real-data science parity.

The value-mask owner is ``xrd_tools.reduction.core._apply_saturation_mask``.
These tests deliberately calculate the accepted bad-pixel set independently,
then compare persisted Standard and GI products against direct pyFAI results.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest

import xrd_tools.reduction.core as reduction_core
from xrd_tools.core.containers import (
    IntegrationResult1D,
    IntegrationResult2D,
    PONI,
)
from xrd_tools.core.invalid import (
    UINT32_CEILING,
    integer_saturation_ceiling,
    saturation_pixels,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.calibration import (
    poni_to_fiber_integrator,
    poni_to_integrator,
)
from xrd_tools.integrate.gid import integrate_gi_2d, integrate_gi_polar_1d
from xrd_tools.integrate.single import integrate_1d, integrate_2d
from xrd_tools.io.read import get_1d, get_2d
from xrd_tools.io.frame_view import read_frame_record
from xrd_tools.reduction import (
    Frame,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    NexusSink,
    ReductionPlan,
    run_reduction,
)
from xrd_tools.sources.registry import open_source


_DATA_ROOT = os.environ.get("XDART_TEST_DATA")
DATA = (
    Path(_DATA_ROOT).expanduser().resolve()
    if _DATA_ROOT
    else Path("__xdart_test_data_unset__")
)
EIGER_ROOT = DATA / "eiger"
EIGER_PONI = DATA / "eiger" / "LaB6_detxn26_detyn6p5_eta4p5.poni"
GI_DIRECTORY = DATA / "nexus" / "bluesky_data"
GI_PONI = GI_DIRECTORY / "LaB6_align61_0003_SR.poni"

SHORT_MASTERS = (
    "Eiger_NbN_1_thin_test__200mdeg_scan001_master.h5",
    "Eiger_NbN_2_thin_test__200mdeg_scan001_master.h5",
    "eiger_w2s3_test_2_scan001_master.h5",
    "eiger_w2s4_1_eta_0p118_scan002_master.h5",
)
GI_FILES = (
    "num_2_716V_5ms_sfpx_70p28_halpha_0p04_00001.nxs",
    "num_2_716V_5ms_sfpx_70p62_halpha_0p04_00001.nxs",
    "num_2_716V_5ms_sfpx_70p97_halpha_0p04_00001.nxs",
    "num_2_716V_5ms_sfpx_71p31_halpha_0p04_00001.nxs",
    "num_2_716V_5ms_sfpx_71p66_halpha_0p04_00001.nxs",
)

_REAL_DATA_UNAVAILABLE = not (
    _DATA_ROOT
    and EIGER_PONI.exists()
    and GI_PONI.exists()
)
requires_real_data = pytest.mark.skipif(
    _REAL_DATA_UNAVAILABLE,
    reason="set XDART_TEST_DATA to the explicit real-data corpus root",
)


def _plan(*, gi: bool = False) -> ReductionPlan:
    return ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=32,
            unit="q_A^-1",
            method="no",
        ),
        integration_2d=Integration2DPlan(
            npt_rad=32,
            npt_azim=16,
            unit="q_A^-1",
            method="no",
        ),
        gi=(
            GIMode(
                incidence_motor="halpha",
                mode_1d="q_total",
                mode_2d="qip_qoop",
                method="no",
                sample_orientation=4,
            )
            if gi
            else None
        ),
        mask_saturation=True,
    )


def _independent_value_mask(raw: np.ndarray) -> np.ndarray:
    """Canonical GUI parity, derived without the reduction helper."""
    values = np.asarray(raw)
    return (
        (values < 0)
        | (values >= UINT32_CEILING)
        | saturation_pixels(
            values,
            ceiling=integer_saturation_ceiling(values),
        )
    )


def _source(path: Path, poni: PONI):
    source = open_source(SourceSpec(path, SourceKind.NEXUS_STACK))
    source.poni = poni
    return source


def _assert_1d(actual, expected: IntegrationResult1D) -> None:
    np.testing.assert_allclose(actual.q, expected.radial, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        actual.intensity,
        expected.intensity,
        rtol=2e-6,
        atol=2e-6,
        equal_nan=True,
    )


def _assert_2d(actual, expected: IntegrationResult2D) -> None:
    np.testing.assert_allclose(actual.q, expected.radial, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        actual.chi,
        expected.azimuthal,
        rtol=1e-7,
        atol=1e-9,
    )
    # IntegrationResult2D is radial × azimuthal; persistence is chi × q.
    np.testing.assert_allclose(
        actual.intensity,
        expected.intensity.T,
        rtol=2e-6,
        atol=2e-6,
        equal_nan=True,
    )


def _r1() -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([1.0, 2.0]),
        unit="q_A^-1",
    )


def _r2() -> IntegrationResult2D:
    return IntegrationResult2D(
        radial=np.array([0.0, 1.0]),
        azimuthal=np.array([-1.0, 1.0]),
        intensity=np.ones((2, 2)),
        unit="q_A^-1",
        azimuthal_unit="chi_deg",
    )


class _Integrators:
    def standard(self):
        return object()

    def fiber(self):
        return object()


def test_value_mask_is_toggle_qualified_fraction_guarded_and_unioning() -> None:
    enabled = _plan()
    disabled = ReductionPlan(
        integration_1d=Integration1DPlan(npt=2),
        integration_2d=None,
        mask_saturation=False,
    )
    existing = np.zeros((100, 100), dtype=bool)
    existing[4, 4] = True

    uint32 = np.zeros(existing.shape, dtype=np.uint32)
    uint32[0, 0] = np.iinfo(np.uint32).max
    resolved = reduction_core._apply_saturation_mask(existing, uint32, enabled)
    assert resolved[0, 0] and resolved[4, 4] and resolved.sum() == 2

    dense_uint32 = np.full(existing.shape, np.iinfo(np.uint32).max, dtype=np.uint32)
    assert reduction_core._apply_saturation_mask(
        None, dense_uint32, enabled
    ).all()

    uint16_sparse = np.zeros(existing.shape, dtype=np.uint16)
    uint16_sparse[0, 0] = np.iinfo(np.uint16).max
    assert reduction_core._apply_saturation_mask(
        None, uint16_sparse, enabled
    ) is None

    uint16_dense = uint16_sparse.copy()
    uint16_dense[0, 1] = np.iinfo(np.uint16).max
    dense_mask = reduction_core._apply_saturation_mask(
        existing, uint16_dense, enabled
    )
    assert dense_mask[0, :2].all() and dense_mask[4, 4]
    assert dense_mask.sum() == 3

    signed = np.zeros(existing.shape, dtype=np.int32)
    signed[2, 2] = -1
    negative_mask = reduction_core._apply_saturation_mask(
        existing, signed, enabled
    )
    assert negative_mask[2, 2] and negative_mask[4, 4]

    # The operator toggle remains authoritative for dummies and negatives.
    assert reduction_core._apply_saturation_mask(
        existing, dense_uint32, disabled
    ) is existing
    assert reduction_core._apply_saturation_mask(
        None, signed, disabled
    ) is None


def test_value_mask_is_resolved_once_from_first_native_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = replace(_plan(), integration_2d=None)
    first = np.zeros((100, 100), dtype=np.uint16)
    first[0, :2] = np.iinfo(np.uint16).max
    later = np.zeros_like(first)
    later[1, :3] = np.iinfo(np.uint16).max
    real = reduction_core.detector_value_mask
    calls: list[np.ndarray] = []
    masks: list[np.ndarray | None] = []

    def counted(mask, raw, *, enabled):
        calls.append(np.asarray(raw))
        return real(mask, raw, enabled=enabled)

    monkeypatch.setattr(reduction_core, "detector_value_mask", counted)
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda _image, _ai, **kwargs: (
            masks.append(kwargs["mask"]),
            _r1(),
        )[1],
    )
    run_reduction(
        plan,
        reduction_core.Scan(
            "first",
            [
                Frame(index=0, image=first),
                Frame(index=1, image=later),
            ],
            integrator=object(),
        ),
        chunk_size=1,
    )

    assert len(calls) == 1
    assert calls[0] is first
    assert masks[0] is not None and masks[1] is not None
    assert masks[0] is masks[1]
    assert masks[0][0, :2].all()
    assert not masks[0][1, :3].any()
    assert masks[1][0, :2].all()
    assert not masks[1][1, :3].any()
    assert not hasattr(plan, "resolved_value_mask")
    assert not hasattr(plan, "value_mask_resolved")


def test_streaming_source_uses_one_first_frame_mask_for_integration_and_thumbnail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime mask is shared without reopening the sustained source."""

    first = np.zeros((100, 100), dtype=np.uint16)
    first[0, :2] = np.iinfo(np.uint16).max
    first[5, 5] = 10
    later = np.zeros_like(first)
    later[1, :3] = np.iinfo(np.uint16).max
    later[5, 5] = 20
    real = reduction_core.detector_value_mask
    value_mask_calls: list[np.ndarray] = []
    integration_masks: dict[int, np.ndarray | None] = {}

    def counted(mask, raw, *, enabled):
        value_mask_calls.append(np.asarray(raw))
        return real(mask, raw, enabled=enabled)

    def capture(image, _ai, **kwargs):
        integration_masks[int(image[5, 5])] = kwargs["mask"]
        return _r1()

    monkeypatch.setattr(reduction_core, "detector_value_mask", counted)
    monkeypatch.setattr(reduction_core, "integrate_1d", capture)

    class SustainedSource:
        name = "stable-mask-source"
        frame_indices = [0, 1]
        integrator = object()
        output_path = None

        def __init__(self) -> None:
            self.loaded: list[int] = []
            self.iterated = 0

        def load_frame(self, index):
            self.loaded.append(int(index))
            raise AssertionError("streaming reduction reopened the source")

        def frame_for(self, index):
            return Frame(
                int(index),
                loader=lambda frame: self.load_frame(frame.index),
            )

        def iter_chunks(self, chunk_size):
            assert chunk_size == 2
            self.iterated += 1
            yield np.stack((first, later)), [0, 1]

        def to_scan(self, **_kwargs):
            return reduction_core.Scan(
                self.name,
                [self.frame_for(index) for index in self.frame_indices],
                integrator=self.integrator,
            )

    source = SustainedSource()
    output = tmp_path / "stable-mask.nxs"
    result = run_reduction(
        replace(_plan(), integration_2d=None),
        source,
        NexusSink(output, overwrite=True, thumbnail_max=128),
        chunk_size=2,
        executor=1,
        execution="streaming",
    )

    assert result.n_processed == 2
    assert source.iterated == 1
    assert source.loaded == []
    assert len(value_mask_calls) == 1
    np.testing.assert_array_equal(value_mask_calls[0], first)
    assert integration_masks[10] is integration_masks[20]
    stable = integration_masks[10]
    assert stable is not None
    assert stable[0, :2].all()
    assert not stable[1, :3].any()

    first_view = read_frame_record(output, 0).active_view()
    later_view = read_frame_record(output, 1).active_view()
    assert first_view.mask_baked and later_view.mask_baked
    assert first_view.thumbnail is not None
    assert later_view.thumbnail is not None
    assert np.isnan(first_view.thumbnail[0, :2]).all()
    assert np.isnan(later_view.thumbnail[0, :2]).all()
    assert np.isfinite(later_view.thumbnail[1, :3]).all()


def test_threshold_membership_is_recomputed_for_each_native_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production parity: thresholding remains dynamic for every frame."""

    first = np.array([[1, 20], [2, 3]], dtype=np.uint16)
    later = np.array([[30, 4], [5, 6]], dtype=np.uint16)
    integrated: list[np.ndarray] = []

    def capture(image, _ai, **_kwargs):
        integrated.append(np.asarray(image).copy())
        return _r1()

    monkeypatch.setattr(reduction_core, "integrate_1d", capture)
    result = run_reduction(
        ReductionPlan(
            integration_1d=Integration1DPlan(npt=2),
            integration_2d=None,
            threshold_max=10.0,
            mask_saturation=True,
        ),
        reduction_core.Scan(
            "dynamic-threshold",
            [
                Frame(index=0, image=first),
                Frame(index=1, image=later),
            ],
            integrator=object(),
        ),
        chunk_size=1,
    )

    assert result.failed is False
    assert len(integrated) == 2
    np.testing.assert_array_equal(
        np.isnan(integrated[0]),
        np.array([[False, True], [False, False]]),
    )
    np.testing.assert_array_equal(
        np.isnan(integrated[1]),
        np.array([[True, False], [False, False]]),
    )


def test_resolved_mask_reaches_all_four_integration_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, np.ndarray | None] = {}

    def standard_1d(_image, _ai, **kwargs):
        calls["standard_1d"] = kwargs["mask"]
        return _r1()

    def standard_2d(_image, _ai, **kwargs):
        calls["standard_2d"] = kwargs["mask"]
        return _r2()

    def gi_1d(_image, _fi, _plan, _gi, **kwargs):
        calls["gi_1d"] = kwargs["mask"]
        return _r1()

    def gi_2d(_image, _fi, _plan, _gi, **kwargs):
        calls["gi_2d"] = kwargs["mask"]
        return _r2()

    monkeypatch.setattr(reduction_core, "integrate_1d", standard_1d)
    monkeypatch.setattr(reduction_core, "integrate_2d", standard_2d)
    monkeypatch.setattr(reduction_core, "_run_gi_1d", gi_1d)
    monkeypatch.setattr(reduction_core, "_run_gi_2d", gi_2d)

    raw = np.zeros((100, 100), dtype=np.uint32)
    raw[0, 0] = np.iinfo(np.uint32).max
    existing = np.zeros(raw.shape, dtype=bool)
    existing[1, 1] = True
    for gi in (False, True):
        plan = _plan(gi=gi)
        if plan.gi is not None:
            plan.gi = replace(
                plan.gi,
                incident_angle=0.2,
                incidence_motor=None,
            )
        plan.mask = existing
        reduction_core._reduce_frame(
            Frame(index=0, image=raw.copy()),
            None,
            plan,
            _Integrators(),
            {},
            run_saturation_mask=reduction_core._RunSaturationMask(True),
        )

    assert set(calls) == {
        "standard_1d",
        "standard_2d",
        "gi_1d",
        "gi_2d",
    }
    for mask in calls.values():
        assert mask is not None
        assert mask[0, 0] and mask[1, 1] and mask.sum() == 2


@pytest.mark.parametrize("master_name", SHORT_MASTERS)
@requires_real_data
def test_all_short_eiger_standard_outputs_match_explicit_mask(
    tmp_path: Path,
    master_name: str,
) -> None:
    master = EIGER_ROOT / master_name
    poni = PONI.from_poni_file(EIGER_PONI)
    plan = _plan()
    output = tmp_path / f"{master.stem}.nxs"
    source = _source(master, poni)
    result = run_reduction(
        plan,
        source,
        NexusSink(output, overwrite=True),
        executor=False,
    )

    reference_source = _source(master, poni)
    ai = poni_to_integrator(poni)
    try:
        labels = tuple(reference_source.frame_indices)
        assert result.n_processed == len(labels)
        # Production intentionally resolves detector-value membership once
        # from the first native frame.  This preserves the established
        # scan-stable policy and avoids rebuilding pyFAI's mask-qualified LUT.
        first_mask = _independent_value_mask(
            np.asarray(reference_source.load_frame(labels[0]))
        )
        for label in labels:
            raw = np.asarray(reference_source.load_frame(label))
            expected_1d = integrate_1d(
                raw.astype(float),
                ai,
                npt=32,
                unit="q_A^-1",
                method="no",
                mask=first_mask,
            )
            expected_2d = integrate_2d(
                raw.astype(float),
                ai,
                npt_rad=32,
                npt_azim=16,
                unit="q_A^-1",
                method="no",
                mask=first_mask,
            )
            _assert_1d(get_1d(output, label), expected_1d)
            _assert_2d(get_2d(output, label), expected_2d)

            if master_name == SHORT_MASTERS[0] and label == labels[0]:
                unmasked_1d = integrate_1d(
                    raw.astype(float),
                    ai,
                    npt=32,
                    unit="q_A^-1",
                    method="no",
                    mask=None,
                )
                unmasked_2d = integrate_2d(
                    raw.astype(float),
                    ai,
                    npt_rad=32,
                    npt_azim=16,
                    unit="q_A^-1",
                    method="no",
                    mask=None,
                )
                with pytest.raises(AssertionError):
                    _assert_1d(get_1d(output, label), unmasked_1d)
                with pytest.raises(AssertionError):
                    _assert_2d(get_2d(output, label), unmasked_2d)
    finally:
        reference_source.close()


@pytest.mark.parametrize("source_name", GI_FILES)
@requires_real_data
def test_five_file_gi_directory_outputs_match_explicit_mask(
    tmp_path: Path,
    source_name: str,
) -> None:
    source_path = GI_DIRECTORY / source_name
    poni = PONI.from_poni_file(GI_PONI)
    plan = _plan(gi=True)
    output = tmp_path / f"{source_path.stem}.processed.nxs"
    source = _source(source_path, poni)
    result = run_reduction(
        plan,
        source,
        NexusSink(output, overwrite=True),
        executor=False,
    )

    reference_source = _source(source_path, poni)
    try:
        labels = tuple(reference_source.frame_indices)
        assert len(labels) == 5
        assert result.n_processed == 5
        # GI uses the same first-native-frame runtime mask as Standard; the
        # incidence-dependent geometry remains per-frame.
        first_mask = _independent_value_mask(
            np.asarray(reference_source.load_frame(labels[0]))
        )
        for label in labels:
            raw = np.asarray(reference_source.load_frame(label))
            incidence = float(reference_source.metadata_for(label)["halpha"])
            fi = poni_to_fiber_integrator(
                poni,
                incident_angle=incidence,
                sample_orientation=4,
            )
            expected_1d = integrate_gi_polar_1d(
                raw.astype(float),
                fi,
                npt=32,
                unit="q_A^-1",
                method="no",
                mask=first_mask,
                incident_angle=incidence,
                sample_orientation=4,
            )
            expected_2d = integrate_gi_2d(
                raw.astype(float),
                fi,
                npt_rad=32,
                npt_azim=16,
                unit="qip_A^-1",
                method="no",
                mask=first_mask,
                incident_angle=incidence,
                sample_orientation=4,
            )
            _assert_1d(get_1d(output, label), expected_1d)
            _assert_2d(get_2d(output, label), expected_2d)
    finally:
        reference_source.close()
