"""Real viewer interfaces: display projections, inclusive crops and retained science."""
import numpy as np
import pytest

from tests.core.test_rsm_viewer_model import _values, _q_projection
from xrd_tools.session.rsm_viewer_model import RSMViewerModel, RSMViewerRefused, make_rsm_viewer_values


@pytest.mark.parametrize("cartesian", [False, True])
def test_mean_maps_crop_existing_centres_keep_zero_nan_and_source(cartesian):
    data = np.arange(60.0).reshape(3, 4, 5)
    data[1, 1, 2] = np.nan
    data[1, 2, 2] = 0
    values = make_rsm_viewer_values(_q_projection(intensity=data)) if cartesian else _values(intensity=data)
    axes = tuple(x for _, x in values.axes)
    bounds = tuple((float(x[1]), float(x[-1])) for x in axes)
    before = values.intensity.tobytes()
    model = RSMViewerModel()
    projected = model.snapshot(values, projection=True, coordinate_ranges=bounds)
    crop = values.intensity[1:, 1:, 1:]
    for product, normal in zip(projected.products[:3], (2, 1, 0), strict=True):
        np.testing.assert_array_equal(product.values, np.nanmean(crop, axis=normal, dtype=np.float64).astype('f4').T)
        assert product.source_indices == (None, None, None)
        assert not product.values.flags.writeable
    for product, retained in zip(projected.products[3:], range(3), strict=True):
        other = tuple(i for i in range(3) if i != retained)
        np.testing.assert_array_equal(product.values, np.nanmean(crop, axis=other, dtype=np.float64).astype('f4'))
        np.testing.assert_array_equal(product.x_axis, axes[retained][1:])
    assert model.snapshot(values, projection=True, coordinate_ranges=bounds) is projected
    assert values.intensity.tobytes() == before
    thin = model.snapshot(values)
    np.testing.assert_array_equal(thin.products[0].values, values.intensity[:, :, 2].T)
    assert thin is not projected


def test_empty_or_invalid_crop_keeps_last_snapshot_and_single_bin_is_allowed():
    values = _values()
    model = RSMViewerModel()
    prior = model.snapshot(values, projection=True)
    for bounds in (((9., 10.), (1., 2.), (2., 3.)), ((0., float('nan')), (1., 2.), (2., 3.))):
        with pytest.raises(RSMViewerRefused):
            model.snapshot(values, projection=True, coordinate_ranges=bounds)
        assert model.current_snapshot is prior
    axes = tuple(x for _, x in values.axes)
    bounds = tuple((float(x[1]), float(x[1])) for x in axes)
    single = model.snapshot(values, projection=True, coordinate_ranges=bounds)
    assert [x.values.shape for x in single.products] == [(1, 1)]*3 + [(1,)]*3
