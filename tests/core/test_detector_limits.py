# -*- coding: utf-8 -*-
"""Detector raw-dtype saturation-ceiling policy (LV-UI-11 owner)."""

import pytest

from xrd_tools.session.detector_limits import (
    DETECTOR_FAMILY_RAW_DTYPES,
    detector_saturation_ceiling,
)


@pytest.mark.parametrize(
    ("detector", "ceiling"),
    [
        ("Eiger1M", 4294967295.0),
        ("Eiger2CdTe1M", 4294967295.0),
        ("eiger 500k", 4294967295.0),
        ("RayonixMx225", 65535.0),
        ("rayonix mx300-hs", 65535.0),
        ("Perkin", 65535.0),
        ("PerkinElmer XRD1621", 65535.0),
    ],
)
def test_known_families_report_their_raw_dtype_ceiling(detector, ceiling):
    value = detector_saturation_ceiling(detector)
    assert type(value) is float and value == ceiling


@pytest.mark.parametrize(
    "detector",
    [
        "Pilatus1M",          # family not verified -> blank, never a guess
        "Detector",
        "",
        None,
        "   ",
        123,
    ],
)
def test_unknown_or_absent_detectors_report_none(detector):
    assert detector_saturation_ceiling(detector) is None


def test_family_table_holds_real_integer_dtypes():
    import numpy as np

    for family, dtype in DETECTOR_FAMILY_RAW_DTYPES.items():
        assert family == family.lower().strip()
        assert np.issubdtype(np.dtype(dtype), np.integer)
