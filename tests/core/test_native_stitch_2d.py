"""Native Stitch 2D: the (χ, q) map and I(q) as an Int 2D record (STITCH-2D-20260930).

Stage A0's frozen oracle, written before the product code: the native record carries the science the
existing kernels make, bit for bit. The map's reference is the 2-D kernel alone (``stitch_2d_streaming``,
the old layout's) run on the very frames and integrators the native one-pass kernel receives, stored by
the artifact projection; I(q) is native Stitch 1D's for the same inputs. The map is non-square
(48 q × 32 χ), so a transposed map fails on its shape and its values.
"""
import copy
from dataclasses import replace
import os

import numpy as np

import h5py
import pytest

from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleOutputRequest
from xrd_tools.analysis.stitch_operation import (
    prepare_stitch_operation, run_stitch_operation, stitch_output_current,
)
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactInvalid, AnalysisArtifactKind, _project_analysis_artifact_result_v1_compat,
    read_analysis_artifact,
)
from xrd_tools.io.frame_view import read_frame_view
from xrd_tools.io.output_path import resolve_finite_output_target
from tests.core.test_stitch_operation import _prepared


def _native(tmp_path, original, *, mode):
    """``original``'s plan (a Stitch 1D's) as a native Int record of ``mode`` ('1d' or '2d'), in its stable
    slot; a map over χ −90…90 on I(q)'s axis (its own, as Int 2D's, is the plan's ``map_unit`` and
    ``map_radial_range``)."""
    plan = replace(original.plan, native_records=True, mode=mode,
                   azimuth_range=(-90.0, 90.0) if mode == "2d" else None,
                   **(dict(map_unit=original.plan.unit, map_radial_range=original.plan.radial_range)
                      if mode == "2d" else {}))
    kind = AnalysisArtifactKind.STITCH_2D if mode == "2d" else AnalysisArtifactKind.STITCH_1D
    family = f"native-{mode}"
    target = resolve_finite_output_target(tmp_path, family, operation_token=f"stitch-{mode}")
    output = ModuleOutputRequest(target, kind, artifact_family=family)
    return prepare_stitch_operation(original.module.source, output, plan, project_root=tmp_path)


def _kernel_map(monkeypatch):
    """The map the 2-D kernel alone (``stitch_2d_streaming``) makes from the frames and integrators the
    native one-pass kernel (the map and I(q)) receives: an independent run of the old layout's kernel, so the
    native map is checked against it rather than against itself."""
    from xrd_tools.integrate import multi
    captured = {}
    one_pass = multi.stitch_2d_with_1d_streaming

    def both(images, image_count, integrator_factory, **kwargs):
        frames = [np.array(image, copy=True) for image in images]
        alone = {key: value for key, value in kwargs.items()
                 if key not in {"npt_1d", "gi", "gi_angles", "unit_1d", "radial_range_1d"}}
        alone["frame_screen"] = copy.deepcopy(kwargs.get("frame_screen"))  # a fresh screen, as the pass's
        captured["map"] = multi.stitch_2d_streaming(iter(frames), image_count, integrator_factory, **alone)
        return one_pass(iter(frames), image_count, integrator_factory, **kwargs)
    monkeypatch.setattr(multi, "stitch_2d_with_1d_streaming", both)
    return captured


def test_a_native_stitch_2d_carries_the_2d_kernels_map_and_stitch_1ds_pattern(tmp_path, monkeypatch):
    base = _prepared(tmp_path)  # a Stitch 1D's plan and sources
    kernel = _kernel_map(monkeypatch)
    native = _native(tmp_path, base, mode="2d")
    result = run_stitch_operation(native)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal.diagnostic
    payload = read_analysis_artifact(native.module.output.target, expected_kind=AnalysisArtifactKind.STITCH_2D)
    value, diagnostics = kernel["map"]
    # stored as the operation stores a MultiGeometry map (pixel splitting: fractional coverage)
    reference = _project_analysis_artifact_result_v1_compat(
        kind=AnalysisArtifactKind.STITCH_2D, axes=(("q", value.radial), ("chi", value.azimuthal)),
        axis_units=(("q", value.unit), ("chi", value.azimuthal_unit)), intensity=value.intensity,
        sigma=value.sigma, coverage=diagnostics.coverage, normalization=diagnostics.normalization)
    # the map in the container's orientation, (q, χ) = (radial, azimuthal), bit for bit the 2-D kernel's
    assert payload.intensity.shape == reference.intensity.shape == (48, 32)
    np.testing.assert_array_equal(payload.intensity, reference.intensity)
    for (name, values), (_name, expected) in zip(payload.axes, reference.axes, strict=True):
        np.testing.assert_array_equal(values, expected, err_msg=name)
    np.testing.assert_array_equal(payload.coverage, reference.coverage)
    np.testing.assert_array_equal(payload.normalization, reference.normalization)
    # I(q) beside it, as Int 2D writes one: native Stitch 1D's for the same inputs
    one = _native(tmp_path, base, mode="1d")
    assert run_stitch_operation(one).terminal.disposition is ModuleDisposition.COMMITTED
    pattern = read_analysis_artifact(one.module.output.target, expected_kind=AnalysisArtifactKind.STITCH_1D)
    np.testing.assert_array_equal(payload.profile.intensity, pattern.intensity)
    np.testing.assert_array_equal(payload.profile.axes[0][1], pattern.axis("q"))
    np.testing.assert_array_equal(payload.profile.coverage, pattern.coverage)
    np.testing.assert_array_equal(payload.profile.normalization, pattern.normalization)


def test_a_native_stitch_2d_record_is_an_int_2d_record(tmp_path):
    """A2/A3: Int's reader reads the record as an Int 2D row (I(q) in the 1-D mode, the map as (χ, q)),
    Append knows it, and a map without its coverage and normalization is refused."""
    native = _native(tmp_path, _prepared(tmp_path), mode="2d")
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
    image = tmp_path / "source" / "scan_0001.tif"
    before = image.stat()
    image.write_bytes(image.read_bytes() + b"revision drift")
    os.utime(image, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    changed = image.stat()
    assert changed.st_size != before.st_size and changed.st_mtime_ns != before.st_mtime_ns
    from xrd_tools.analysis.module_transaction import ModuleSourceReceipt
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    table = run_metadata_table(MetadataTablePlan(native.module.source.analysis.source_spec))
    source = ModuleSourceReceipt.from_metadata_table(
        table, kind=native.module.source.kind, selected_labels=native.module.source.selected_labels,
        resolved_selectors=native.module.source.resolved_selectors,
    )
    refreshed = prepare_stitch_operation(source, native.module.output, native.plan, project_root=tmp_path)
    assert not stitch_output_current(refreshed)
    with h5py.File(target, "r+") as handle:
        del handle["entry/stitch/map"]
    with pytest.raises(AnalysisArtifactInvalid):
        read_analysis_artifact(target, expected_kind=AnalysisArtifactKind.STITCH_2D)


@pytest.mark.parametrize("mode", ("1d", "2d"))
def test_native_stitch_save_reload_preserves_missing_pattern_bins(tmp_path, mode):
    """Actual native artifacts and Int's reader retain missing-data NaNs in I(q)."""
    native = _native(tmp_path, _prepared(tmp_path), mode=mode)
    native = prepare_stitch_operation(
        native.module.source, native.module.output, replace(native.plan, error_model="poisson"),
        project_root=tmp_path,
    )
    result = run_stitch_operation(native)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal.diagnostic
    payload = read_analysis_artifact(native.module.output.target)
    pattern = payload if mode == "1d" else payload.profile
    empty = pattern.coverage == 0
    assert np.any(empty) and np.any(~empty)
    assert np.isnan(pattern.intensity[empty]).all()
    assert pattern.sigma is not None
    assert np.isnan(pattern.sigma[empty]).all()
    view = read_frame_view(native.module.output.target, 1, include_thumbnail=False)
    np.testing.assert_array_equal(view.intensity_1d, pattern.intensity)
    from xrd_tools.io.export import read_xye, write_stitch_xye
    from xrd_tools.io.read import resolve_source_master
    import os
    exported = write_stitch_xye(payload)
    for actual, expected in zip(read_xye(exported), (dict(pattern.axes)["q"], pattern.intensity, pattern.sigma)):
        np.testing.assert_allclose(actual, expected, rtol=5e-9, atol=0, equal_nan=True)
    record = payload.contributing_frames[0]
    source = record.get("source_constituents", (record,))[0]
    raw = resolve_source_master(source["source_path"], scan_file=native.module.output.target,
                                source_base=str(tmp_path))
    assert raw is not None
    before = raw.read_bytes()
    exported.unlink()
    os.link(raw, exported)
    with pytest.raises(ValueError, match="source"):
        write_stitch_xye(payload)
    assert raw.read_bytes() == before and os.path.samefile(raw, exported)
    exported.unlink()
