"""Owner-approved neutral dataset names; scientific values and orientation stay fixed."""

import h5py
import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.nexus import write_nexus, read_scan, read_scan_metadata
from xrd_tools.io.processed_scan_id import is_current_processed_xdart_path
from xrd_tools.io.frame_view import read_frame_view
from xrd_tools.io.nexus import write_integrated_stack


@pytest.mark.parametrize(
    "x_unit,y_unit,x_label,y_label",
    [
        ("q_A^-1", "chi_deg", "Q", "χ"),
        ("qip_A^-1", "qoop_A^-1", "Q_ip", "Q_oop"),
        ("2th_deg", "chi_deg", "2θ", "χ"),
        ("qtot_A^-1", "chigi_deg", "Q_total", "χ_GI"),
        ("exit_angle_horz_deg", "exit_angle_vert_deg",
         "exit angle horizontal", "exit angle vertical"),
    ],
)
def test_neutral_axes_roundtrip_values_labels_and_nonsquare_orientation(
    tmp_path, x_unit, y_unit, x_label, y_label,
):
    path = tmp_path / "neutral.nexus"
    x = np.linspace(0.1, 2.0, 5, dtype=np.float32)
    y = np.linspace(-2.0, 3.0, 3, dtype=np.float32)
    intensity = np.arange(15, dtype=np.float32).reshape(5, 3)
    write_nexus(
        path,
        results_1d={1: IntegrationResult1D(x, x * 3, unit=x_unit)},
        results_2d={1: IntegrationResult2D(
            x, y, intensity, unit=x_unit, azimuthal_unit=y_unit,
        )},
        overwrite=True,
        compression=None,
    )
    with h5py.File(path, "r") as handle:
        assert handle["entry"].attrs["ssrl_schema_version"] == 3
        for name, expected in (
            ("integrated_1d", ("frame_index", "axis_x")),
            ("integrated_2d", ("frame_index", "axis_y", "axis_x")),
        ):
            group = handle[f"entry/{name}"]
            assert tuple(group.attrs["axes"]) == expected
            assert "q" not in group and "chi" not in group
            np.testing.assert_array_equal(group["axis_x"][()], x)
            assert group["axis_x"].attrs["units"] == x_unit
            assert group["axis_x"].attrs["long_name"].startswith(x_label + " (")
        group = handle["entry/integrated_2d"]
        np.testing.assert_array_equal(group["axis_y"][()], y)
        np.testing.assert_array_equal(group["intensity"][0], intensity.T)
        assert group["axis_y"].attrs["units"] == y_unit
        assert group["axis_y"].attrs["long_name"].startswith(y_label + " (")
    assert is_current_processed_xdart_path(path)
    # Existing Python coordinate API stays stable; physical disk names do not.
    data = read_scan(path)
    np.testing.assert_array_equal(data.intensity_2d.values[0], intensity.T)
    np.testing.assert_array_equal(data.coords["q"].values, x)
    np.testing.assert_array_equal(read_scan_metadata(path).coords["chi"].values, y)
    view = read_frame_view(path, 1, include_thumbnail=False)
    np.testing.assert_array_equal(view.intensity_2d, intensity.T)
    assert view.axis_2d_x.unit == x_unit and view.axis_2d_y.unit == y_unit


@pytest.mark.parametrize("bulk", [False, True])
def test_named_gi_modes_append_and_reload_neutral_axes(tmp_path, bulk):
    from tests.core.v2_fixture_factory import current_entry

    path = tmp_path / "gi_modes.nexus"
    x = np.linspace(-2, 2, 5, dtype=np.float32)
    y = np.linspace(0, 3, 3, dtype=np.float32)
    image = np.arange(15, dtype=np.float32).reshape(5, 3)
    with h5py.File(path, "w") as handle:
        entry = current_entry(handle)
        for label in (1, 4):
            write_integrated_stack(
                entry, frame_indices=[label],
                results_1d=[IntegrationResult1D(x, x + label, unit="qip_A^-1")],
                results_2d=[IntegrationResult2D(
                    x, y, image + label, unit="qip_A^-1", azimuthal_unit="qoop_A^-1",
                )],
                primary_mode_1d="q_ip", primary_mode_2d="qip_qoop",
                extra_modes_1d={"q_oop": [IntegrationResult1D(
                    y, y + label, unit="qoop_A^-1",
                )]}, bulk_new_rows=bulk,
            )
        for name in ("integrated_1d", "integrated_1d/q_oop", "integrated_2d"):
            group = entry[name]
            assert "axis_x" in group and "q" not in group
            assert "long_name" in group["axis_x"].attrs
            np.testing.assert_array_equal(group["frame_index"], [1, 4])
    assert is_current_processed_xdart_path(path)
    view = read_frame_view(path, 4, mode_1d="q_oop", include_thumbnail=False)
    np.testing.assert_array_equal(view.axis_1d.values, y)
    np.testing.assert_array_equal(view.intensity_1d, y + 4)
    np.testing.assert_array_equal(view.intensity_2d, (image + 4).T)
