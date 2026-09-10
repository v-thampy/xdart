"""E2-SCI: exact saturation-mask policy and real-data science parity.

The value-mask owner is ``xrd_tools.core.invalid.detector_value_mask``.
These tests deliberately calculate the accepted bad-pixel set independently,
then compare persisted Standard and GI products against direct pyFAI results.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.containers import (
    IntegrationResult1D,
    IntegrationResult2D,
    PONI,
)
from xrd_tools.core.invalid import (
    UINT32_CEILING,
    detector_value_mask,
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


def test_value_mask_is_toggle_qualified_fraction_guarded_and_unioning() -> None:
    existing = np.zeros((100, 100), dtype=bool)
    existing[4, 4] = True

    uint32 = np.zeros(existing.shape, dtype=np.uint32)
    uint32[0, 0] = np.iinfo(np.uint32).max
    resolved = detector_value_mask(existing, uint32, enabled=True)
    assert resolved[0, 0] and resolved[4, 4] and resolved.sum() == 2

    dense_uint32 = np.full(existing.shape, np.iinfo(np.uint32).max, dtype=np.uint32)
    assert detector_value_mask(
        None, dense_uint32, enabled=True
    ).all()

    uint16_sparse = np.zeros(existing.shape, dtype=np.uint16)
    uint16_sparse[0, 0] = np.iinfo(np.uint16).max
    assert detector_value_mask(
        None, uint16_sparse, enabled=True
    ) is None

    uint16_dense = uint16_sparse.copy()
    uint16_dense[0, 1] = np.iinfo(np.uint16).max
    dense_mask = detector_value_mask(
        existing, uint16_dense, enabled=True
    )
    assert dense_mask[0, :2].all() and dense_mask[4, 4]
    assert dense_mask.sum() == 3

    signed = np.zeros(existing.shape, dtype=np.int32)
    signed[2, 2] = -1
    negative_mask = detector_value_mask(
        existing, signed, enabled=True
    )
    assert negative_mask[2, 2] and negative_mask[4, 4]

    # The operator toggle remains authoritative for dummies and negatives.
    assert detector_value_mask(
        existing, dense_uint32, enabled=False
    ) is existing
    assert detector_value_mask(
        None, signed, enabled=False
    ) is None


@pytest.mark.parametrize("master_name", SHORT_MASTERS)
@requires_real_data
def test_all_short_eiger_standard_outputs_match_local_value_exclusions(
    tmp_path: Path,
    master_name: str,
) -> None:
    master = EIGER_ROOT / master_name
    poni = PONI.from_poni_file(EIGER_PONI)
    plan = _plan()
    output = tmp_path / f"{master.stem}.nexus"
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
        for label in labels:
            raw = np.asarray(reference_source.load_frame(label))
            current_mask = _independent_value_mask(raw)
            working = raw.astype(float)
            working[current_mask] = np.nan
            expected_1d = integrate_1d(
                working,
                ai,
                npt=32,
                unit="q_A^-1",
                method="no",
                mask=None,
            )
            expected_2d = integrate_2d(
                working,
                ai,
                npt_rad=32,
                npt_azim=16,
                unit="q_A^-1",
                method="no",
                mask=None,
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
def test_five_file_gi_directory_outputs_match_local_value_exclusions(
    tmp_path: Path,
    source_name: str,
) -> None:
    source_path = GI_DIRECTORY / source_name
    poni = PONI.from_poni_file(GI_PONI)
    plan = _plan(gi=True)
    output = tmp_path / f"{source_path.stem}.processed.nexus"
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
        for label in labels:
            raw = np.asarray(reference_source.load_frame(label))
            current_mask = _independent_value_mask(raw)
            working = raw.astype(float)
            working[current_mask] = np.nan
            incidence = float(reference_source.metadata_for(label)["halpha"])
            fi = poni_to_fiber_integrator(
                poni,
                incident_angle=incidence,
                sample_orientation=4,
            )
            expected_1d = integrate_gi_polar_1d(
                working,
                fi,
                npt=32,
                unit="q_A^-1",
                method="no",
                mask=None,
                incident_angle=incidence,
                sample_orientation=4,
            )
            expected_2d = integrate_gi_2d(
                working,
                fi,
                npt_rad=32,
                npt_azim=16,
                unit="qip_A^-1",
                method="no",
                mask=None,
                incident_angle=incidence,
                sample_orientation=4,
            )
            _assert_1d(get_1d(output, label), expected_1d)
            _assert_2d(get_2d(output, label), expected_2d)
    finally:
        reference_source.close()
