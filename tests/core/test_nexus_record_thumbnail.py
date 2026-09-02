from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize("max_size", [8, 4])
def test_make_thumbnail_array_boolean_mask_matches_legacy_without_aliasing(
    max_size,
):
    image = np.arange(48, dtype=np.float32).reshape(6, 8)
    original = image.copy()
    mask = np.zeros(image.shape, dtype=bool)
    mask[1, 2] = True
    mask[4, 6] = True
    legacy = nexus_record.make_thumbnail_array(
        image, mask_flat=np.flatnonzero(mask), max_size=max_size,
    )

    safe = nexus_record.make_thumbnail_array(
        image, mask=mask, max_size=max_size,
    )
    scratch = image.copy()
    owned = nexus_record.make_thumbnail_array(
        scratch, mask=mask, max_size=max_size, _owned=True,
    )

    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(safe, legacy)
    np.testing.assert_array_equal(owned, legacy)
    if max_size == 8:
        assert owned is scratch


def test_make_thumbnail_array_owned_contract_is_narrow_and_fail_closed():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = np.zeros(image.shape, dtype=bool)

    with pytest.raises(TypeError, match="exact bool"):
        nexus_record.make_thumbnail_array(image, mask=mask, _owned=1)
    with pytest.raises(ValueError, match="writable owning float32"):
        nexus_record.make_thumbnail_array(image[:, :2], mask=mask[:, :2], _owned=True)
    with pytest.raises(ValueError, match="thumbnail mask"):
        nexus_record.make_thumbnail_array(image, mask=np.zeros((2, 2), bool))
    with pytest.raises(ValueError, match="mutually exclusive"):
        nexus_record.make_thumbnail_array(image, mask=mask, mask_flat=[0])
