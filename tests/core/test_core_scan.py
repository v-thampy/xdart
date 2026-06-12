from __future__ import annotations

import numpy as np
import pytest


def test_reduction_reexports_core_scan_contracts():
    from xrd_tools.core.scan import FrameSource, MaskSpec, Scan, ScanFrame
    from xrd_tools.reduction import (
        Frame as ReductionFrame,
        FrameSource as ReductionFrameSource,
        MaskSpec as ReductionMaskSpec,
        Scan as ReductionScan,
    )

    assert ReductionFrame is ScanFrame
    assert ReductionScan is Scan
    assert ReductionMaskSpec is MaskSpec
    assert ReductionFrameSource is FrameSource


def test_scan_frame_preserves_heterogeneous_metadata_with_numeric_view():
    from xrd_tools.core.scan import ScanFrame

    frame = ScanFrame(
        index=5,
        image=np.zeros((2, 2)),
        metadata={"i0": "12.5", "sample": "LaB6", "flag": object()},
    )

    assert frame.metadata_raw["sample"] == "LaB6"
    assert frame.metadata_numeric["i0"] == 12.5
    assert "sample" not in frame.metadata_numeric
    assert "flag" not in frame.metadata_numeric


def test_canonical_scan_is_frame_source_and_preserves_strings_in_scan_data():
    from xrd_tools.core.scan import FrameSource, Scan, ScanFrame

    scan = Scan(
        "mixed",
        [
            ScanFrame(2, image=np.full((2, 2), 2), metadata={"i0": 2, "tag": "b"}),
            ScanFrame(1, image=np.ones((2, 2)), metadata={"i0": 1, "tag": "a"}),
        ],
        motors={"th": np.array([0.1, 0.2])},
    )

    assert isinstance(scan, FrameSource)
    assert scan.frame_indices == [1, 2]
    assert np.array_equal(scan.load_frame(1), np.ones((2, 2)))
    df = scan.to_scan_data()
    assert list(df.index) == [1, 2]
    assert list(df["tag"]) == ["a", "b"]
    assert np.allclose(df["th"], [0.1, 0.2])


def test_canonical_scan_rejects_duplicate_indices():
    from xrd_tools.core.scan import Scan, ScanFrame

    with pytest.raises(ValueError, match="duplicate frame indices"):
        Scan("dupes", [ScanFrame(1, image=np.zeros((1, 1))), ScanFrame(1, image=np.ones((1, 1)))])
