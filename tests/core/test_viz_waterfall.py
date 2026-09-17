"""Coordinate-driven waterfall plots preserve the scientific row/column mapping."""

import numpy as np
import pytest
import xarray as xr

from xrd_tools.viz import plot_waterfall


@pytest.mark.parametrize(
    "y_coord,values,label,units",
    [
        ("frame", [4, 9], "Frame", ""),
        ("time", [0.0, 0.25], "Elapsed time", "s"),
        ("temperature", [295.0, 310.0], "Temperature", "K"),
    ],
)
def test_waterfall_uses_coordinate_labels_and_dimension_order(y_coord, values, label, units):
    # Store intensity in (bin, exposure) order; the heatmap must use (y, x).
    intensity = np.array([[1., 2., 3.], [4., 5., 6.]])
    dataset = xr.Dataset(
        {"intensity": (("bin", "exposure"), intensity.T)},
        coords={
            "angle": ("bin", [10., 20., 30.], {"long_name": "Two theta", "units": "deg"}),
            y_coord: ("exposure", values, {"long_name": label, "units": units}),
        },
    )
    original = dataset.copy(deep=True)
    figure = plot_waterfall(dataset, x_coord="angle", y_coord=y_coord)
    np.testing.assert_array_equal(figure.data[0].z, intensity)
    np.testing.assert_array_equal(figure.data[0].x, [10., 20., 30.])
    np.testing.assert_array_equal(figure.data[0].y, values)
    assert figure.layout.xaxis.title.text == "Two theta (deg)"
    expected_y = f"{label} ({units})" if units else label
    assert figure.layout.yaxis.title.text == expected_y
    assert expected_y in figure.data[0].hovertemplate
    assert "Two theta (deg)" in figure.data[0].hovertemplate
    assert dataset.identical(original)


def test_waterfall_defaults_to_q_and_frame_and_keeps_log_display_separate():
    dataset = xr.Dataset(
        {"intensity": (("frame", "q"), [[1., 10.], [100., 1000.]])},
        coords={"q": [1., 2.], "frame": [7, 11]},
    )
    original = dataset.copy(deep=True)
    figure = plot_waterfall(dataset, log_intensity=True, color_percentiles=(0, 100))
    # Retain the existing positive-floor policy at the low end.
    assert figure.data[0].z[1].tolist() == [2., 3.]
    assert figure.layout.xaxis.title.text == "q"
    assert figure.layout.yaxis.title.text == "frame"
    assert figure.data[0].colorbar.title.text == "log10(intensity)"
    assert dataset.identical(original)
