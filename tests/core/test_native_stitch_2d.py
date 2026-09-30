"""Native Stitch 2D: the (χ, q) map and I(q) as an Int 2D record (STITCH-2D-20260930).

Stage A0's frozen oracle, written before the product code: the native record carries the science the
existing kernels make, bit for bit. The old layout's 2-D result is the reference while it exists; I(q)
is native Stitch 1D's for the same inputs. The map is non-square (48 q × 32 χ), so a transposed map
fails on its shape and its values.
"""
from dataclasses import replace

import numpy as np

import h5py
import pytest

from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleOutputRequest
from xrd_tools.analysis.stitch_operation import (
    prepare_stitch_operation, run_stitch_operation, stitch_output_current,
)
from xrd_tools.io.analysis_artifact import AnalysisArtifactInvalid, AnalysisArtifactKind, read_analysis_artifact
from xrd_tools.io.frame_view import read_frame_view
from xrd_tools.io.output_path import resolve_finite_output_target
from tests.core.test_stitch_operation import _prepared


def _native(tmp_path, original, *, mode):
    """``original``'s plan as a native Int record of ``mode`` ('1d' or '2d'), in its stable slot."""
    plan = replace(original.plan, native_records=True, mode=mode,
                   azimuth_range=original.plan.azimuth_range if mode == "2d" else None)
    kind = AnalysisArtifactKind.STITCH_2D if mode == "2d" else AnalysisArtifactKind.STITCH_1D
    family = f"native-{mode}"
    target = resolve_finite_output_target(tmp_path, family, operation_token=f"stitch-{mode}")
    output = ModuleOutputRequest(target, kind, artifact_family=family)
    return prepare_stitch_operation(original.module.source, output, plan, project_root=tmp_path)


def test_a_native_stitch_2d_carries_the_old_layouts_map_and_stitch_1ds_pattern(tmp_path):
    old = _prepared(tmp_path, mode="2d")
    reference = run_stitch_operation(old)
    assert reference.terminal.disposition is ModuleDisposition.COMMITTED, reference.terminal.diagnostic
    native = _native(tmp_path, old, mode="2d")
    result = run_stitch_operation(native)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal.diagnostic
    payload = read_analysis_artifact(native.module.output.target, expected_kind=AnalysisArtifactKind.STITCH_2D)
    # the map in the container's orientation, (q, χ) = (radial, azimuthal), bit for bit the old layout's
    assert payload.intensity.shape == reference.payload.intensity.shape == (48, 32)
    np.testing.assert_array_equal(payload.intensity, reference.payload.intensity)
    for name in ("q", "chi"):
        np.testing.assert_array_equal(payload.axis(name), reference.payload.axis(name))
    np.testing.assert_array_equal(payload.coverage, reference.payload.coverage)
    np.testing.assert_array_equal(payload.normalization, reference.payload.normalization)
    # I(q) beside it, as Int 2D writes one: native Stitch 1D's for the same inputs
    one = _native(tmp_path, old, mode="1d")
    assert run_stitch_operation(one).terminal.disposition is ModuleDisposition.COMMITTED
    pattern = read_analysis_artifact(one.module.output.target, expected_kind=AnalysisArtifactKind.STITCH_1D)
    np.testing.assert_array_equal(payload.profile.intensity, pattern.intensity)
    np.testing.assert_array_equal(payload.profile.axes[0][1], pattern.axis("q"))
    np.testing.assert_array_equal(payload.profile.coverage, pattern.coverage)
    np.testing.assert_array_equal(payload.profile.normalization, pattern.normalization)


def test_a_native_stitch_2d_record_is_an_int_2d_record(tmp_path):
    """A2/A3: Int's reader reads the record as an Int 2D row (I(q) in the 1-D mode, the map as (χ, q)),
    Append knows it, and a map without its coverage and normalization is refused."""
    native = _native(tmp_path, _prepared(tmp_path, mode="2d"), mode="2d")
    assert run_stitch_operation(native).terminal.disposition is ModuleDisposition.COMMITTED
    target = native.module.output.target
    payload = read_analysis_artifact(target, expected_kind=AnalysisArtifactKind.STITCH_2D)
    view = read_frame_view(target, 1, include_thumbnail=False)
    np.testing.assert_array_equal(view.intensity_2d, payload.intensity.T)
    assert (view.axis_2d_x.unit, view.axis_2d_y.unit) == ("q_A^-1", "chi_deg")
    np.testing.assert_array_equal(view.axis_2d_x.values, payload.axis("q"))
    np.testing.assert_array_equal(view.axis_2d_y.values, payload.axis("chi"))
    np.testing.assert_array_equal(view.intensity_1d, payload.profile.intensity)
    np.testing.assert_array_equal(view.axis_1d.values, payload.profile.axes[0][1])
    assert stitch_output_current(native)  # Append skips it
    with h5py.File(target, "r+") as handle:
        del handle["entry/stitch/map"]
    with pytest.raises(AnalysisArtifactInvalid):
        read_analysis_artifact(target, expected_kind=AnalysisArtifactKind.STITCH_2D)
