from __future__ import annotations

import numpy as np

from xrd_tools.io import nexus_record


def test_make_thumbnail_array_masks_union_without_mutating_input():
    image = np.arange(16, dtype=np.uint16).reshape(4, 4)
    original = image.copy()

    thumbnail = nexus_record.make_thumbnail_array(
        image,
        mask_flat=np.array([-1, 0, 0, 16]),
        global_mask_flat=np.array([5, 5]),
        max_size=4,
    )

    np.testing.assert_array_equal(image, original)
    assert thumbnail.dtype == np.float32
    assert np.isnan(thumbnail.flat[0])
    assert np.isnan(thumbnail.flat[5])
    assert np.isfinite(thumbnail.flat[-1])
    assert np.count_nonzero(np.isnan(thumbnail)) == 2


def test_make_thumbnail_array_single_mask_skips_sort_and_concatenate(
    monkeypatch,
):
    def fail(*_args, **_kwargs):
        raise AssertionError("single-mask thumbnail path must not sort or concatenate")

    monkeypatch.setattr(nexus_record.np, "unique", fail)
    monkeypatch.setattr(nexus_record.np, "concatenate", fail)

    thumbnail = nexus_record.make_thumbnail_array(
        np.arange(9, dtype=np.float32).reshape(3, 3),
        mask_flat=np.array([8, 0, 8]),
        max_size=3,
    )

    assert np.isnan(thumbnail.flat[0])
    assert np.isnan(thumbnail.flat[8])
    assert np.count_nonzero(np.isnan(thumbnail)) == 2
