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


def test_run_roi_signals_per_roi_reducer_and_background():
    """Each RoiSignal carries its OWN reducer + background — two signals with
    different reducers/ops reduce in a single pass over the frames."""
    from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
    from xrd_tools.sources import MemoryFrameSource

    src = MemoryFrameSource(_stack())
    sig_roi = RoiSpec(center_x=2, center_y=2, width_x=2, width_y=2)
    bg = RoiSpec(center_x=4.5, center_y=4.5, width_x=2, width_y=2)
    signals = (
        RoiSignal(roi=sig_roi, reducer="sum", background=bg,
                  background_op="subtract", name="A"),       # 360, 760, 1160
        RoiSignal(roi=sig_roi, reducer="mean", background=bg,
                  background_op="divide", name="B"),         # 10, 20, 30
    )
    res = run_roi_signals(signals, src).payload
    np.testing.assert_allclose(res.series["A"], [360, 760, 1160])
    np.testing.assert_allclose(res.series["B"], [10, 20, 30])
    np.testing.assert_array_equal(res.frames, [0, 1, 2])
    assert res.diagnostics["cancelled"] is False


def test_run_roi_signals_matches_run_roi_stats_for_shared_config():
    """The shared-reducer plan is the special case of per-ROI signals — the
    wrapper and the driver must agree element-for-element (the mini spine)."""
    from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
    from xrd_tools.sources import MemoryFrameSource

    src = MemoryFrameSource(_stack())
    a = RoiSpec(center_x=2, center_y=2, width_x=2, width_y=2, name="a")
    b = RoiSpec(center_x=4.5, center_y=4.5, width_x=2, width_y=2, name="b")
    plan = RoiStatsPlan(rois=(a, b), reducer="mean")
    via_plan = run_roi_stats(plan, src).payload
    via_signals = run_roi_signals(
        (RoiSignal(roi=a, reducer="mean", name="a"),
         RoiSignal(roi=b, reducer="mean", name="b")), src).payload
    for name in ("a", "b"):
        np.testing.assert_allclose(via_plan.series[name], via_signals.series[name])


def test_run_roi_signals_divide_is_area_scaled():
    """divide uses the background DENSITY (like subtract), so sum+divide and
    mean+divide give the same ratio regardless of the ROI's pixel area."""
    from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
    from xrd_tools.sources import MemoryFrameSource

    src = MemoryFrameSource(_stack())
    sig = RoiSpec(center_x=2, center_y=2, width_x=2, width_y=2)
    bg = RoiSpec(center_x=4.5, center_y=4.5, width_x=2, width_y=2)   # density 10
    mean = run_roi_signals(
        (RoiSignal(roi=sig, reducer="mean", background=bg,
                   background_op="divide", name="m"),), src).payload
    summ = run_roi_signals(
        (RoiSignal(roi=sig, reducer="sum", background=bg,
                   background_op="divide", name="s"),), src).payload
    np.testing.assert_allclose(mean.series["m"], [10, 20, 30])
    np.testing.assert_allclose(summ.series["s"], [10, 20, 30])    # area cancels


def test_run_roi_signals_applies_static_mask():
    """A static detector mask (MaskSpec or bool array) excludes pixels from ROI
    stats — the parity §6.3 requires with the reducer."""
    from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
    from xrd_tools.core.scan import MaskSpec
    from xrd_tools.sources import MemoryFrameSource

    img = np.arange(36, dtype=float).reshape(6, 6)
    src = MemoryFrameSource([img])
    sig = RoiSignal(roi=RoiSpec.full_frame(), reducer="mean", name="f")
    base = run_roi_signals((sig,), src).payload
    mask = np.zeros((6, 6), dtype=bool)
    mask[:, :3] = True                       # exclude the (lower-valued) left half
    masked = run_roi_signals((sig,), src, mask=mask).payload
    assert masked.valid_counts["f"][0] == 18
    assert masked.series["f"][0] == pytest.approx(img[:, 3:].mean())
    assert masked.series["f"][0] > base.series["f"][0]
    # the MaskSpec wrapper resolves to the same exclusion.
    via_spec = run_roi_signals((sig,), src, mask=MaskSpec(mask)).payload
    np.testing.assert_allclose(via_spec.series["f"], masked.series["f"])


def test_run_roi_signals_dedups_colliding_names():
    """Two signals resolving to the SAME name must not collapse into one
    interleaved, wrong-length column (silent corruption for headless callers)."""
    from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
    from xrd_tools.sources import MemoryFrameSource

    src = MemoryFrameSource(_stack())                 # 3 frames
    a = RoiSpec(center_x=2, center_y=2, width_x=2, width_y=2)
    b = RoiSpec(center_x=4.5, center_y=4.5, width_x=2, width_y=2)
    res = run_roi_signals(
        (RoiSignal(roi=a, name="roi"), RoiSignal(roi=b, name="roi")), src).payload
    assert set(res.series) == {"roi", "roi_2"}        # disambiguated
    for name in ("roi", "roi_2"):
        assert len(res.series[name]) == len(res.frames) == 3
    # the two columns are distinct (signal vs background block), not interleaved.
    assert not np.array_equal(res.series["roi"], res.series["roi_2"])


def test_run_roi_signals_streams_and_cancels():
    """on_frame/on_progress stream per frame; should_cancel stops early and the
    result covers only the processed frames (diagnostics['cancelled'])."""
    from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
    from xrd_tools.sources import MemoryFrameSource

    src = MemoryFrameSource(_stack())                 # 3 frames
    sig = RoiSignal(roi=RoiSpec.full_frame(), reducer="mean", name="full")

    streamed, progress = [], []
    res = run_roi_signals(
        (sig,), src,
        on_frame=lambda f, row: streamed.append((f, row["full"])),
        on_progress=lambda d, t: progress.append((d, t))).payload
    assert [f for f, _ in streamed] == [0, 1, 2]
    assert progress[-1] == (3, 3)
    assert res.diagnostics["cancelled"] is False

    # cancel before the 2nd frame -> only frame 0 processed.
    seen = []

    def stop_after_one():
        return len(seen) >= 1

    res2 = run_roi_signals(
        (sig,), src, on_frame=lambda f, _row: seen.append(f),
        should_cancel=stop_after_one).payload
    assert list(res2.frames) == [0]
    assert res2.diagnostics["cancelled"] is True
