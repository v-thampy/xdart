"""Headless ROI-stats core: RoiSpec geometry, masked reducers, and the
scan-level run_roi_stats driver (synthetic image stacks — no GUI)."""
import numpy as np
import pytest

from xrd_tools.analysis.plans import RoiStatsPlan, run_roi_stats
from xrd_tools.core.invalid import UINT32_CEILING
from xrd_tools.core.roi import RoiSpec, invalid_pixel_mask, roi_reduce


def test_roi_spec_pixel_slice_and_full_frame():
    roi = RoiSpec(center_x=2, center_y=2, width_x=2, width_y=2)
    rs, cs = roi.pixel_slice((6, 6))
    assert (rs.start, rs.stop) == (1, 3) and (cs.start, cs.stop) == (1, 3)
    # clamps to bounds
    edge = RoiSpec(center_x=0, center_y=0, width_x=4, width_y=4)
    rs, cs = edge.pixel_slice((6, 6))
    assert rs.start == 0 and cs.start == 0
    # full frame covers everything
    rs, cs = RoiSpec.full_frame().pixel_slice((6, 8))
    assert (rs.start, rs.stop, cs.start, cs.stop) == (0, 6, 0, 8)


def test_roi_reduce_reducers_and_mask():
    img = np.arange(36, dtype=float).reshape(6, 6)
    roi = RoiSpec(center_x=2, center_y=2, width_x=3, width_y=3)
    rs, cs = roi.pixel_slice(img.shape)
    patch = img[rs, cs]
    assert roi_reduce(img, roi, reducer="sum")[0] == patch.sum()
    assert roi_reduce(img, roi, reducer="mean")[0] == patch.mean()
    assert roi_reduce(img, roi, reducer="max")[0] == patch.max()
    assert roi_reduce(img, roi, reducer="min")[0] == patch.min()
    assert roi_reduce(img, roi, reducer="mean")[1] == patch.size
    # mask excludes pixels from both value and count
    mask = np.zeros_like(img, dtype=bool)
    mask[rs.start, cs.start] = True
    val, n = roi_reduce(img, roi, mask=mask, reducer="mean")
    assert n == patch.size - 1
    expected = np.nanmean(np.where(mask[rs, cs], np.nan, patch))
    assert val == pytest.approx(expected)
    # all-invalid -> NaN, 0
    allbad = np.ones_like(img, dtype=bool)
    v, n0 = roi_reduce(img, roi, mask=allbad)
    assert np.isnan(v) and n0 == 0


def test_invalid_pixel_mask_policy():
    img = np.array([[1.0, UINT32_CEILING], [np.nan, 5.0]])
    m = invalid_pixel_mask(img)
    assert m[0, 1] and m[1, 0]            # uint32 dummy + NaN always excluded
    assert not m[0, 0] and not m[1, 1]
    # dtype-ceiling saturation: gated by mask_saturation + the fraction guard
    sat = np.full((10, 10), 65535, dtype=np.uint16)
    sat[0, 0] = 1
    assert not invalid_pixel_mask(sat, mask_saturation=False).any()
    on = invalid_pixel_mask(sat, mask_saturation=True)
    assert on.sum() == 99 and not on[0, 0]


def _stack():
    """3 frames: a signal block [1:3,1:3] = (f+1)*100 and a const bg [4:6,4:6]=10."""
    frames = []
    for f in range(3):
        im = np.zeros((6, 6), dtype=float)
        im[1:3, 1:3] = (f + 1) * 100.0
        im[4:6, 4:6] = 10.0
        frames.append(im)
    return frames


def test_run_roi_stats_series_and_background():
    from xrd_tools.sources import MemoryFrameSource

    src = MemoryFrameSource(_stack())
    sig = RoiSpec(center_x=2, center_y=2, width_x=2, width_y=2, name="sig")
    bg = RoiSpec(center_x=4.5, center_y=4.5, width_x=2, width_y=2, name="bg")

    # plain mean series
    res = run_roi_stats(RoiStatsPlan(rois=(sig,), reducer="mean"), src).payload
    np.testing.assert_allclose(res.series["sig"], [100, 200, 300])
    np.testing.assert_array_equal(res.frames, [0, 1, 2])
    assert res.x_label == "frame"
    np.testing.assert_array_equal(res.valid_counts["sig"], [4, 4, 4])

    # subtract: mean - background density
    sub = run_roi_stats(
        RoiStatsPlan(rois=(sig,), background=bg, background_op="subtract",
                     reducer="mean"), src).payload
    np.testing.assert_allclose(sub.series["sig"], [90, 190, 290])

    # subtract: sum stays area-scaled (sum - density*n_valid)
    ssum = run_roi_stats(
        RoiStatsPlan(rois=(sig,), background=bg, background_op="subtract",
                     reducer="sum"), src).payload
    np.testing.assert_allclose(ssum.series["sig"], [360, 760, 1160])

    # divide: same-reducer ratio
    div = run_roi_stats(
        RoiStatsPlan(rois=(sig,), background=bg, background_op="divide",
                     reducer="mean"), src).payload
    np.testing.assert_allclose(div.series["sig"], [10, 20, 30])


def test_run_roi_stats_no_raw_frame_is_nan_not_crash():
    from xrd_tools.sources import MemoryFrameSource

    class _Raising(MemoryFrameSource):
        def load_frame(self, index):
            if int(index) == 1:
                raise IOError("raw unreachable")
            return super().load_frame(index)

    src = _Raising([np.full((4, 4), 5.0) for _ in range(3)])
    res = run_roi_stats(RoiStatsPlan(rois=(RoiSpec.full_frame(),)), src).payload
    assert res.diagnostics["no_raw_frames"] == [1]
    assert res.series["full"][0] == 5.0
    assert np.isnan(res.series["full"][1])       # unreadable frame -> NaN, no crash
    assert res.series["full"][2] == 5.0
