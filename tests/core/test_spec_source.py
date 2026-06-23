"""Metadata-only SPEC FrameSource: classification, per-point + constant-motor
columns, scan selection, and the no-images contract (load_frame raises)."""
import numpy as np
import pytest

pytest.importorskip("silx")

from xrd_tools.sources import (SourceKind, SpecSource, guess_source_kind,
                               open_source)

_SPEC = """#F t.spec
#E 1
#D today
#O0 th  chi  phi

#S 1 ascan th 0 2 2 1
#D today
#P0 0 5 10
#N 4
#L th  Epoch  i0  det
0 1 100 10
1 2 110 20
2 3 120 30

#S 2 ascan chi 0 1 1 1
#D today
#P0 7 0 10
#N 2
#L chi  i0
0 300
1 310
"""


def _spec(tmp_path):
    p = tmp_path / "t.spec"
    p.write_text(_SPEC)
    return p


def test_spec_is_classified_and_opened(tmp_path):
    p = _spec(tmp_path)
    assert guess_source_kind(p) is SourceKind.SPEC
    src = open_source(p)
    assert isinstance(src, SpecSource)
    assert src.scan_key == "1.1"                 # default = first scan
    assert src.frame_indices == [0, 1, 2]


def test_spec_columns_and_constant_motors(tmp_path):
    src = SpecSource(_spec(tmp_path))
    # per-point scanned column + counters (whole arrays via .motors)
    np.testing.assert_allclose(src.motors["th"], [0, 1, 2])
    np.testing.assert_allclose(src.motors["i0"], [100, 110, 120])
    # metadata_for merges per-point (winning) with constant scan-start motors
    md = dict(src.metadata_for(1))
    assert md["th"] == 1.0 and md["i0"] == 110.0          # per-point
    assert md["chi"] == 5.0 and md["phi"] == 10.0          # constant


def test_spec_is_metadata_only(tmp_path):
    src = SpecSource(_spec(tmp_path))
    assert src.capabilities.has_metadata is True
    assert src.capabilities.has_raw_references is False
    with pytest.raises(NotImplementedError):
        src.load_frame(0)


def test_spec_scan_selection(tmp_path):
    p = _spec(tmp_path)
    second = SpecSource(p, scan=2)
    assert second.scan_key == "2.1"
    assert second.frame_indices == [0, 1]
    np.testing.assert_allclose(second.motors["chi"], [0, 1])
    # open_source threads the scan option through.
    via_opts = open_source(p, scan="2.1")
    assert via_opts.frame_indices == [0, 1]
