"""Owner-approved neutral dataset names; scientific values and orientation stay fixed."""

import h5py
import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.nexus import write_nexus, read_scan, read_scan_metadata
from xrd_tools.io.processed_scan_id import is_current_processed_xdart_path
from xrd_tools.io.frame_view import read_frame_view
from xrd_tools.io.nexus import write_integrated_stack


def _as_v2(path):
    """Build the previous exact layout, independently of the read adapter."""
    with h5py.File(path, "r+") as handle:
        entry = handle["entry"]
        groups = []
        entry.visititems(lambda _name, node: groups.append(node)
                         if isinstance(node, h5py.Group) and "axis_x" in node else None)
        for group in groups:
            group.move("axis_x", "q")
            del group["q"].attrs["long_name"]
            if "axis_y" in group:
                group.move("axis_y", "chi")
                del group["chi"].attrs["long_name"]
                group.attrs["axes"] = ["frame_index", "chi", "q"]
            else:
                group.attrs["axes"] = ["frame_index", "q"]
        entry.attrs["ssrl_schema_version"] = 2


@pytest.mark.parametrize("gi", [False, True])
def test_previous_axes_read_without_mutation_and_append_refuses(tmp_path, gi):
    from xrd_tools.io.read import get_1d, get_2d, open_scan
    from xrd_tools.io.nexus import open_nexus_writer
    from xrd_tools.io.nexus_inspect import inspect_nexus
    from xrd_tools.io.processed_scan_id import require_current_writable_processed_groups
    from tests.core.v2_fixture_factory import current_entry

    path = tmp_path / "previous.nexus"
    x, y = np.arange(5, dtype=np.float32), np.arange(3, dtype=np.float32)
    cake = np.arange(15, dtype=np.float32).reshape(5, 3)
    unit, azimuthal_unit = ("qip_A^-1", "qoop_A^-1") if gi else ("q_A^-1", "chi_deg")
    with h5py.File(path, "w") as handle:
        write_integrated_stack(
            current_entry(handle), frame_indices=[2, 5],
            results_1d=[IntegrationResult1D(x, x + n, unit=unit) for n in (2, 5)],
            results_2d=[IntegrationResult2D(x, y, cake + n, unit=unit,
                       azimuthal_unit=azimuthal_unit) for n in (2, 5)],
            primary_mode_1d="q_ip" if gi else "default",
            primary_mode_2d="qip_qoop" if gi else "default",
            extra_modes_1d={"q_oop": [IntegrationResult1D(y, y + n, unit="qoop_A^-1")
                                     for n in (2, 5)]} if gi else None,
        )
    _as_v2(path)
    before = path.read_bytes()
    assert is_current_processed_xdart_path(path)
    np.testing.assert_array_equal(get_1d(path, 5).intensity, x + 5)
    np.testing.assert_array_equal(get_2d(path, 2).intensity, (cake + 2).T)
    np.testing.assert_array_equal(read_scan(path).coords["q"], x)
    np.testing.assert_array_equal(read_scan_metadata(path).coords["chi"], y)
    view = read_frame_view(path, 5, mode_1d="q_oop" if gi else "default",
                           include_thumbnail=False)
    np.testing.assert_array_equal(view.intensity_1d, (y if gi else x) + 5)
    np.testing.assert_array_equal(view.intensity_2d, (cake + 5).T)
    scan = open_scan(path)
    np.testing.assert_array_equal(scan.get_1d(2).intensity, x + 2)
    inspected = inspect_nexus(path).xdart
    assert [axis.name for axis in inspected.integrated_2d.axes] == ["q", "chi"]
    with h5py.File(path, "r") as handle:
        with pytest.raises(ValueError, match="read-only.*v2"):
            require_current_writable_processed_groups(handle)
    with pytest.raises(ValueError, match="read-only.*v2"):
        open_nexus_writer(path)
    with pytest.raises(ValueError, match="read-only.*v2"):
        write_nexus(path, results_1d={6: IntegrationResult1D(x, x, unit=unit)})
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [1, 4, True, "2", 2.0])
def test_old_axis_support_does_not_guess_unknown_schema(tmp_path, version):
    path = tmp_path / "unknown.nexus"
    write_nexus(path, results_1d={1: IntegrationResult1D(np.arange(5), np.arange(5))})
    _as_v2(path)
    with h5py.File(path, "r+") as handle:
        handle["entry"].attrs["ssrl_schema_version"] = version
    assert not is_current_processed_xdart_path(path)
    with pytest.raises(ValueError, match="not a current"):
        read_scan(path)


@pytest.mark.parametrize("dimension", ["1d", "2d"])
@pytest.mark.parametrize("prepared", [False, True])
def test_v2_reintegrate_only_normalizes_private_candidate(tmp_path, monkeypatch, dimension, prepared):
    from tests.core.test_reintegrate_immutable_successor import (
        _seed_existing, _stub_integrators, _plan, _prepared_plan, _prepared_eligible_source,
    )
    from xrd_tools.io.finite_artifact import capture_finite_source
    from xrd_tools.reduction import prepare_reintegrate_bundle, run_reintegrate_successor

    seeded = _seed_existing(tmp_path, labels=(2, 5))
    if prepared:
        _prepared_eligible_source(seeded)
    _as_v2(seeded.target)
    before, raw_before = seeded.target.read_bytes(), seeded.source.read_bytes()
    other = "2d" if dimension == "1d" else "1d"
    with h5py.File(seeded.target, "r") as handle:
        preserved = handle[f"entry/integrated_{other}/intensity"][()]
    _stub_integrators(monkeypatch)
    if prepared:
        offer = prepare_reintegrate_bundle(capture_finite_source(seeded.target),
                  entry="entry", labels=seeded.labels, source_root=str(seeded.target.parent))
        _, plan = _prepared_plan(seeded, dimension=dimension, offer=offer)
    else:
        plan = _plan(seeded, dimension=dimension, expected_terminal=None)
    result = run_reintegrate_successor(plan)
    assert result.disposition == "COMMITTED", result
    assert seeded.target.read_bytes() == before
    assert seeded.source.read_bytes() == raw_before
    with h5py.File(result.output_artifact, "r") as handle:
        assert handle["entry"].attrs["ssrl_schema_version"] == 3
        for name in ("integrated_1d", "integrated_2d"):
            group = handle[f"entry/{name}"]
            assert "axis_x" in group and "q" not in group and "chi" not in group
            assert "long_name" in group["axis_x"].attrs
        np.testing.assert_array_equal(handle[f"entry/integrated_{other}/intensity"], preserved)
    assert read_frame_view(result.output_artifact, 5, include_thumbnail=False).intensity_1d is not None


@pytest.mark.parametrize("damage", ["intensity", "units", "other_metadata"])
def test_v2_preservation_normalizes_only_approved_deltas(tmp_path, damage):
    import shutil
    from xrd_tools.io.record_writer import _replacement_manifest_digest_for
    from xrd_tools.io.processed_scan_id import upgrade_private_integrated_axes

    source, candidate = tmp_path / "source.nexus", tmp_path / "candidate.nexus"
    x, y = np.arange(5), np.arange(3)
    write_nexus(source, results_1d={2: IntegrationResult1D(x, x)},
                results_2d={2: IntegrationResult2D(x, y, np.arange(15).reshape(5, 3))})
    _as_v2(source)
    before = source.read_bytes()
    def signature(handle):
        return _replacement_manifest_digest_for(handle, "entry", (),
                    ignore_source_base=False, ignore_file_name=False,
                    normalize_integrated_axes=True)
    with h5py.File(source, "r") as handle:
        expected = signature(handle)
    shutil.copyfile(source, candidate)
    with h5py.File(candidate, "r+") as handle:
        upgrade_private_integrated_axes(handle, "entry", container=candidate)
        assert signature(handle) == expected
        group = handle["entry/integrated_2d"]
        if damage == "intensity":
            group["intensity"][0, 0, 0] += 1
        elif damage == "units":
            group["axis_x"].attrs["units"] = "qoop_A^-1"
        else:
            handle["entry"].attrs["unrelated"] = "changed"
        assert signature(handle) != expected
    assert source.read_bytes() == before


@pytest.mark.parametrize("damage", ["mixed_names", "wrong_axes", "missing_axis"])
def test_v2_requires_its_exact_axis_layout(tmp_path, damage):
    path = tmp_path / "malformed.nexus"
    write_nexus(path, results_1d={1: IntegrationResult1D(np.arange(5), np.arange(5))})
    _as_v2(path)
    with h5py.File(path, "r+") as handle:
        group = handle["entry/integrated_1d"]
        if damage == "mixed_names":
            group["axis_x"] = group["q"]
        elif damage == "wrong_axes":
            group.attrs["axes"] = ["frame_index", "axis_x"]
        else:
            del group["q"]
    assert not is_current_processed_xdart_path(path)


def test_v2_replace_recognition_does_not_grant_inplace_reintegration(tmp_path):
    from tests.core.test_reintegrate_immutable_successor import _seed_existing
    from xrd_tools.io.processed_scan_id import require_raw_input, ProcessedXdartInputError
    from xrd_tools.io.record_writer import NexusRecordWriter
    from xrd_tools.reduction import NexusSink

    seeded = _seed_existing(tmp_path, labels=(2,))
    _as_v2(seeded.target)
    before = seeded.target.read_bytes()
    NexusSink(seeded.target, overwrite=True)._require_current_append_target()
    with pytest.raises(ValueError, match="read-only.*v2"):
        NexusSink(seeded.target)._require_current_append_target()
    # Exercise the shared existing-replacement admission before the writer opens.
    writer = NexusRecordWriter(seeded.target)
    writer._replacement_configuration = ("1d",)
    with pytest.raises(ValueError, match="read-only.*v2"):
        writer._validate_existing_contract()
    with pytest.raises(ProcessedXdartInputError):
        require_raw_input(seeded.target)
    assert seeded.target.read_bytes() == before


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
