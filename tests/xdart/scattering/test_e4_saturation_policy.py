import numpy as np
import pytest

from xdart.gui.tabs.scattering.display_runtime import (
    project_detector_values,
    project_frame_detector_values,
)


def test_live_projection_uses_each_frame_values_with_static_and_frame_masks() -> None:
    static = np.zeros((100, 100), dtype=bool)
    static[1, 1] = True
    frame = np.zeros(static.shape, dtype=bool)
    frame[2, 2] = True
    previous = None
    for row in (3, 4):
        raw = np.zeros(static.shape, dtype=np.uint16)
        raw[row, :2] = np.iinfo(np.uint16).max
        before = raw.copy()
        projected, mask_baked = project_frame_detector_values(
            raw, static, frame, value_mask_enabled=True,
        )
        assert mask_baked
        assert np.isnan(projected[row, :2]).all()
        assert np.isnan(projected[1, 1])
        assert np.isnan(projected[2, 2])
        assert np.isfinite(projected[7 - row, :2]).all()
        if previous is not None:
            assert np.isnan(previous[3, :2]).all()
        np.testing.assert_array_equal(raw, before)
        previous = projected


def test_masked_integer_projection_uses_one_float_output_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.arange(16, dtype=np.uint16).reshape(4, 4)
    mask = np.zeros(raw.shape, dtype=bool)
    mask[1, 2] = True
    real_array = np.array
    calls: list[tuple[object, object, object]] = []

    def traced_array(value, *args, **kwargs):
        calls.append(
            (kwargs.get("dtype"), kwargs.get("copy"), np.asarray(value).dtype)
        )
        return real_array(value, *args, **kwargs)

    monkeypatch.setattr(
        "xdart.gui.tabs.scattering.display_runtime.np.array",
        traced_array,
    )

    projected, mask_baked = project_detector_values(
        raw,
        mask,
        value_mask_enabled=False,
    )

    assert mask_baked
    assert calls == [(float, True, raw.dtype)]
    assert projected.dtype == float
    assert np.isnan(projected[1, 2])
    assert not projected.flags.writeable
