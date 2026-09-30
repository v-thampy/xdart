"""Native finite selections: stable slots, multiple exact jobs and raw reads."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tools.stitch_owner import StitchOwnerOutcomeKind, StitchToolOwner
from xdart.gui.tools.stitch_values import StitchFrameSelector, prepare_stitch_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_sources import StitchSourceSelection
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from tests.xdart.pages.test_stitch_page import _form
from tests.xdart.tools.test_stitch_owner import _poll_terminal


def test_native_family_preserves_readable_source_and_frame_subset_slot(tmp_path):
    from xdart.gui.tools.stitch_values import StitchFrameSelector
    form = _native_form(tmp_path, "multigeometry", scans="1")
    first = prepare_stitch_tool(form).request
    subset = prepare_stitch_tool(replace(form,
        frame_selector=StitchFrameSelector(0, 0))).request
    assert Path(first.module.output.target).name == "sample_scan1_stitch1d.nexus"
    assert subset.module.output.target == first.module.output.target


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_a_preflight_marks_the_outputs_already_stitched_from_the_same_frames_with_the_same_settings(
        tmp_path, backend, monkeypatch):
    """What a Run in Append mode skips (maintainer, 2026-09-29: "skip if something has already been
    processed with the same settings as before, else overwrite"): the same settings over the same
    frames, whatever the source files' revisions."""
    import os
    import time
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    form = _native_form(tmp_path, backend, scans="1,[1,2]")
    prepared = prepare_stitch_tool(form)
    assert prepared.unchanged == ()  # nothing stitched yet
    first, grouped = prepared.requests
    assert run_stitch_operation(first).terminal.disposition is ModuleDisposition.COMMITTED
    assert prepare_stitch_tool(form).unchanged == (0,)
    # the envelope alone: the stored result is neither re-read nor re-hashed (maintainer, 2026-09-30:
    # a processed file can always be regenerated, so its full check is not worth the time)
    from xrd_tools.io import analysis_artifact
    monkeypatch.setattr(analysis_artifact, "_stitch_record_projection",
                        lambda *args: pytest.fail("the check re-read the stored result"))
    assert prepare_stitch_tool(form).unchanged == (0,)
    monkeypatch.undo()
    spec = Path(form.spec_path)  # the SPEC file grows by a later scan: its revision changes
    spec.write_text(spec.read_text() + "\n#S 9 ascan del 14 15 1 1\n#D today\n#P0 14 -10\n#N 2\n#L del  nu\n14 -10\n15 -10\n")
    image = next(Path(form.image_dir).glob("sample_scan1_0000*"))
    data = image.read_bytes()
    time.sleep(.01)
    image.write_bytes(data)  # the same bytes under a new revision
    os.utime(image, ns=(time.time_ns() + 10**9,) * 2)
    again = prepare_stitch_tool(form)
    assert again.requests[0].module.fingerprint != first.module.fingerprint  # the revisions differ
    assert again.unchanged == (0,)
    assert run_stitch_operation(again.requests[1]).terminal.disposition is ModuleDisposition.COMMITTED
    assert prepare_stitch_tool(form).unchanged == (0, 1)
    assert prepare_stitch_tool(replace(form, npt_1d=80)).unchanged == ()  # another setting
    subset = replace(form, frame_selector=StitchFrameSelector(0, 0))
    assert prepare_stitch_tool(subset).unchanged == ()  # other frames
    text = spec.read_text()
    start, end = text.index("#S 2 "), text.index("#S 3 ")
    spec.write_text(text[:start] + text[start:end].replace("\n15 -10", "\n15.5 -10", 1) + text[end:])
    assert prepare_stitch_tool(form).unchanged == (0,)  # scan 2 read another angle: its output changed
    Path(first.module.output.target).write_bytes(b"not a stitch")
    assert prepare_stitch_tool(form).unchanged == ()  # an unreadable file is stitched again


@pytest.mark.parametrize("rows, first_only, expected", [
    (("14 -10", "15 -10"), False, True),
    (("14 0", "0 -10"), False, False),
    (("14 0", "15 1e-12"), False, True),
    (("14 0", "15 -10"), True, False),
])
def test_native_affine_warning_uses_selected_coupled_angles(tmp_path, rows, first_only, expected):
    from xdart.gui.tools.stitch_values import StitchFrameSelector
    form = _native_form(tmp_path, "multigeometry", scans="1")
    spec = Path(form.spec_path)
    spec.write_text(spec.read_text().replace("14 -10\n15 -10\n", "\n".join(rows) + "\n"))
    if first_only:
        form = replace(form, frame_selector=StitchFrameSelector(0, 0))
    original = Path(form.geometry_path).read_bytes()
    prepared = prepare_stitch_tool(form)
    assert bool(prepared.warnings) is expected
    if expected:
        assert len(prepared.warnings) == 1
        warning = prepared.warnings[0]
        assert "sample_scan1_stitch1d.nexus" in warning
        assert "independent Euler" in warning
        assert "If del is mounted on nu" in warning
        assert "stacked calibration" in warning
    assert Path(form.geometry_path).read_bytes() == original
    assert prepared.request.plan.geometry.content == original


def test_native_affine_warning_does_not_assume_other_instrument_axes(tmp_path):
    import json
    form = _native_form(tmp_path, "multigeometry", scans="1")
    path = Path(form.geometry_path)
    path.write_text(path.read_text().replace("nu_angle", "tilt_angle").replace("del_angle", "two_theta"))
    form = replace(form, source_motors=(("tilt_angle", "nu"), ("two_theta", "del")))
    prepared = prepare_stitch_tool(form)
    assert prepared.warnings == ()
    assert json.loads(prepared.request.plan.geometry.content)["pos_names"] == ["two_theta", "tilt_angle"]


def test_native_affine_warning_does_not_apply_to_xu(tmp_path):
    prepared = prepare_stitch_tool(_native_form(tmp_path, "xu_hist", scans="1"))
    assert prepared.warnings == ()


def test_native_stacked_calibration_previews_runs_and_reloads(tmp_path):
    import json
    from xrd_tools.core.geometry import Diffractometer, StackedDetectorArm
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    form = _native_form(tmp_path, 'multigeometry', scans='1')
    path = Path(form.geometry_path)
    calibration = Diffractometer.from_pyfai_goniometer(path).calibration
    path.write_text(json.dumps(StackedDetectorArm().to_pyfai_goniometer(calibration)))
    owner = StitchToolOwner()
    try:
        owner.set_form(form)
        assert owner.begin_preflight()
        preview = _poll_terminal(owner)
        assert preview.terminal_status is OperationTerminalStatus.RETURNED, preview.failure_message
        prepared = owner.prepared
        assert prepared.warnings == ()
        assert owner.begin_run()
        terminal = _poll_terminal(owner, timeout=30)
        assert terminal.terminal_status is OperationTerminalStatus.RETURNED, terminal.failure_message
        result = terminal.outcome.result
        assert result.terminal.disposition is ModuleDisposition.COMMITTED
        reloaded = read_analysis_artifact(prepared.request.module.output.target)
        np.testing.assert_array_equal(reloaded.intensity, result.payload.intensity)
        assert np.isfinite(reloaded.intensity).any()
        assert len(reloaded.contributing_frames) == 2
        assert prepared.request.plan.geometry.sha256 in reloaded.provenance_json
        assert reloaded.radial_axis('q_A^-1').shape == (64,)
        assert reloaded.radial_axis('2th_deg').shape == (64,)
    finally:
        assert owner.close().status is PageCleanup.CLEAN


def test_native_group_is_named_automatically_or_as_entered(tmp_path):
    form = replace(_native_form(tmp_path, "multigeometry", scans="[1,2]"), output_family=None)
    prepared = prepare_stitch_tool(form)
    # as Int names a result (maintainer, 2026-09-29): the images' scan name, then the slot
    assert Path(prepared.request.module.output.target).name == "sample_scan1-2_stitch1d.nexus"
    prepared = prepare_stitch_tool(replace(form, output_family="Sample A é"))
    assert Path(prepared.request.module.output.target).name == "Sample A é_stitch1d.nexus"


def test_native_automatic_name_fills_the_stem_with_the_scans_occurrence_key(tmp_path):
    """Codex re-review (P3), now Preview's alone (Codex audit G-F2: the page lists no image folder):
    a stem with {occurrence} is filled with SPEC's N.M key, and the automatic name is Int's name of
    the images it finds."""
    form = replace(_native_form(tmp_path, "multigeometry", scans="1"), output_family=None)
    for source in Path(form.image_dir).glob("*scan1_*"):
        source.rename(source.with_name("sample_occ1.1_" + source.name.rsplit("_", 1)[1]))
    selection = replace(form.source_selection, image_stem="sample_occ{occurrence}_")
    prepared = prepare_stitch_tool(replace(form, source_selection=selection))
    assert Path(prepared.request.module.output.target).name == "sample_occ1.1_stitch1d.nexus"


def test_an_automatic_name_over_80_characters_asks_for_one(tmp_path):
    from types import SimpleNamespace
    from xdart.gui.tools.stitch_native_values import _families
    from xdart.gui.tools.stitch_values import StitchToolPreflightRefused
    spec = tmp_path / "data" / "sample"
    odd = tuple(SimpleNamespace(uri=str(spec), options={"scan": f"{n}.1"}) for n in range(1, 80, 2))
    resolution = SimpleNamespace(groups=(odd,), skipped=(),
                                 names=(tuple(f"sample_scan{n}" for n in range(1, 80, 2)),))
    with pytest.raises(StitchToolPreflightRefused, match="OUTPUT_FAMILY_REQUIRED"):
        _families(SimpleNamespace(output_families=(), output_family=None, gi=None), resolution, tmp_path.resolve())
    assert _families(SimpleNamespace(output_families=(), output_family="odd scans", gi=None), resolution,
                     tmp_path.resolve()) == ("odd scans",)


def test_native_repeated_output_refuses_without_counter(tmp_path):
    from xdart.gui.tools.stitch_values import StitchToolPreflightRefused
    form = _native_form(tmp_path, "multigeometry", scans="1,1")
    with pytest.raises(StitchToolPreflightRefused, match="OUTPUT_FAMILY_COLLISION"):
        prepare_stitch_tool(form)


def test_native_named_multiple_groups_and_invalid_names(tmp_path):
    from xdart.gui.tools.stitch_values import StitchToolPreflightRefused
    form = replace(_native_form(tmp_path, "multigeometry", scans="1,[1,2],[2,1]"),
                   output_family=None, output_families=("", "sample-up", "sample-down"))
    prepared = prepare_stitch_tool(form)
    assert [Path(item.module.output.target).name for item in prepared.requests] == [
        "sample_scan1_stitch1d.nexus", "sample-up_stitch1d.nexus", "sample-down_stitch1d.nexus"]
    for names, code in [(("", "x" * 81, "fine"), "OUTPUT_FAMILY_INVALID"),
                        (("", "../bad", "fine"), "OUTPUT_FAMILY_INVALID"),
                        (("", "same", "SAME"), "OUTPUT_FAMILY_COLLISION"),
                        (("only", "two"), "OUTPUT_FAMILY_COUNT_MISMATCH")]:
        with pytest.raises(StitchToolPreflightRefused, match=code):
            prepare_stitch_tool(replace(form, output_families=names))


@pytest.mark.parametrize("name, code", [(None, "OUTPUT_FAMILY_COLLISION"),
                                       ("combined", "OUTPUT_FAMILY_COUNT_MISMATCH")])
def test_native_naming_refusal_explains_skipped_group_members(tmp_path, name, code):
    from xdart.gui.tools.stitch_values import StitchToolPreflightRefused
    form = replace(_native_form(tmp_path, "multigeometry", scans="1,[1,3]"), output_family=name)
    with pytest.raises(StitchToolPreflightRefused, match=code) as refused:
        prepare_stitch_tool(form)
    assert "Skipped" in str(refused.value)
    assert "source/sample · scan 3.1" in str(refused.value)
    assert "No detector images" in str(refused.value)
    prepared = prepare_stitch_tool(replace(form, output_family=None,
                                          output_families=("single", "combined")))
    assert len(prepared.requests) == 2
    # the reason names what was searched: the scan's automatic stem in the images folder
    assert prepared.skipped[0][1:] == (
        "3.1", f"No detector images named *{Path(form.spec_path).stem}_scan3_* in {form.image_dir}")


def test_native_refused_preview_does_not_create_save_path(tmp_path):
    from xdart.gui.tools.stitch_values import StitchToolPreflightRefused
    form = replace(_native_form(tmp_path, "multigeometry", scans="1,1"),
                   output_family=None, output_path=tmp_path / "new" / "result.nexus")
    with pytest.raises(StitchToolPreflightRefused, match="OUTPUT_FAMILY_COLLISION"):
        prepare_stitch_tool(form)
    assert not (tmp_path / "new").exists()


def test_native_preview_missing_save_path_is_read_only_and_run_creates_it(tmp_path):
    form = replace(_native_form(tmp_path, "multigeometry", scans="1"),
                   output_path=tmp_path / "new" / "nested" / "result_stitch1d.nexus")
    owner = StitchToolOwner()
    try:
        owner.set_form(form)
        assert owner.begin_preflight()
        preview = _poll_terminal(owner, timeout=20)
        assert preview.terminal_status is OperationTerminalStatus.RETURNED, preview.failure_message
        assert not (tmp_path / "new").exists()
        assert owner.begin_run()
        terminal = _poll_terminal(owner, timeout=30)
        assert terminal.terminal_status is OperationTerminalStatus.RETURNED, terminal.failure_message
        assert terminal.outcome.result.terminal.disposition is ModuleDisposition.COMMITTED
        assert Path(terminal.outcome.result.request.module.output.target).is_file()
    finally:
        assert owner.close().status is PageCleanup.CLEAN


def test_native_run_rechecks_save_directory_stays_in_project(tmp_path):
    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    form = replace(_native_form(project, "multigeometry", scans="1"),
                   output_path=project / "new" / "nested" / "result.nexus")
    owner = StitchToolOwner()
    try:
        owner.set_form(form)
        assert owner.begin_preflight()
        assert _poll_terminal(owner).terminal_status is OperationTerminalStatus.RETURNED
        (project / "new").symlink_to(outside, target_is_directory=True)
        assert owner.begin_run()
        terminal = _poll_terminal(owner)
        assert terminal.terminal_status is OperationTerminalStatus.FAILED
        assert "OUTPUT_OUTSIDE_PROJECT" in terminal.failure_message
        assert list(outside.iterdir()) == []
    finally:
        assert owner.close().status is PageCleanup.CLEAN


def test_native_eiger_missing_angles_names_source_and_entry(tmp_path):
    from tests.core.test_source_edge_contract import _dangling_canonical, _land_canonical
    from xdart.gui.tools.stitch_values import StitchToolPreflightRefused
    form = _native_form(tmp_path, "multigeometry", scans="1")
    master, sidecar = _dangling_canonical(tmp_path)
    _land_canonical(sidecar)
    form = replace(form, source_selection=StitchSourceSelection("NeXus", (str(master),)))
    with pytest.raises(StitchToolPreflightRefused, match="MISSING_REQUIRED_METADATA") as error:
        prepare_stitch_tool(form)
    assert str(master) in str(error.value)
    assert "entry entry" in str(error.value)


def _native_form(tmp_path, backend, *, scans="1,2,[1,2]", unit="q_A^-1"):
    form = _form(tmp_path, backend)
    path = Path(form.spec_path)
    text = path.read_text()
    # The MG test calibration is 1 Angstrom; no independent energy was measured
    # for this synthetic source. XU uses its canonical recorded energy.
    if backend == "multigeometry":
        text = text.replace("  energy", "").replace(" 17000.018", "").replace("#N 3", "#N 2")
    second = text[text.index("#S 1"):].replace("#S 1", "#S 2", 1)
    empty = second.replace("#S 2", "#S 3", 1)
    path.write_text(text + "\n" + second + "\n" + empty)
    for source in Path(form.image_dir).glob("*scan1_*"):
        destination = source.with_name(source.name.replace("scan1_", "scan2_"))
        destination.write_bytes(source.read_bytes())
    return replace(form, source_selection=StitchSourceSelection(
        "Spec", (str(path),), scans=scans, image_dir=str(form.image_dir)),
        unit=unit, q_range=None, conditioning=StitchConditioning("auto"),
        output_family="combined" if "[" in scans and "," in scans else None)


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
@pytest.mark.parametrize("unit", ("q_A^-1", "2th_deg"))
def test_native_preflight_run_jobs_raw_and_stable_replacement(tmp_path, backend, unit):
    form = replace(_native_form(tmp_path, backend, unit=unit),
                   output_path=tmp_path / "new-save-folder" / "result_stitch1d.nexus")
    prepared = prepare_stitch_tool(form)
    assert len(prepared.requests) == 3
    assert prepared.summary is None and prepared.request is prepared.requests[0]
    assert [len(item.manifest.contributions) for item in prepared.requests] == [2, 2, 4]
    assert [len(members) for members in prepared.preview_members] == [2, 2, 4]
    assert [member["identity"]["scan_id"] for member in prepared.preview_members[-1]] == ["1.1", "1.1", "2.1", "2.1"]
    targets = [item.module.output.target for item in prepared.requests]
    assert len(set(targets)) == 3
    assert all(Path(target).name.endswith("_stitch1d.nexus") for target in targets)
    repeated = prepare_stitch_tool(replace(form, npt_1d=80))
    assert targets == [item.module.output.target for item in repeated.requests]
    owner = StitchToolOwner()
    try:
        owner.set_form(form)
        assert owner.begin_preflight() is not None
        assert _poll_terminal(owner, timeout=20).outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY
        exact = owner.prepared
        for index, request in enumerate(exact.requests):
            assert owner.begin_run(index) is not None
            terminal = _poll_terminal(owner, timeout=30)
            assert terminal.terminal_status is OperationTerminalStatus.RETURNED, terminal.failure_message
            result = terminal.outcome.result
            assert result.request is request
            assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal.diagnostic
            payload = result.payload
            assert payload.native_unit == unit
            assert payload.radial_axis("q_A^-1").shape == (64,)
            assert payload.radial_axis("2th_deg").shape == (64,)
            assert len(payload.contributing_frames) == (2 if index < 2 else 4)
        record = payload.contributing_frames[0]
        # The shared saved record key defines frame identity for raw navigation.
        key = record["key"].split("/")
        scan = key[0].removeprefix("scan_") if len(key) == 2 else None
        frame = int(key[-1].removeprefix("frame_"))
        assert owner.begin_frame_read(payload, frame, scan=scan, source_root=tmp_path, full_raw=True)
        raw = _poll_terminal(owner, timeout=20)
        assert raw.terminal_status is OperationTerminalStatus.RETURNED, raw.failure_message
        assert raw.outcome.kind is StitchOwnerOutcomeKind.FRAME_READ
        assert raw.outcome.frame_view.raw is not None
        assert owner.prepared is exact
        from xrd_tools.analysis.stitch_operation import run_stitch_operation
        replacement = run_stitch_operation(repeated.requests[0])
        assert replacement.terminal.disposition is ModuleDisposition.COMMITTED, replacement.terminal.diagnostic
        assert replacement.payload.intensity.shape == (80,)
        assert replacement.request.module.output.target == targets[0]
    finally:
        assert owner.close().status is PageCleanup.CLEAN


def test_no_image_scans_skip_without_execution_authority(tmp_path):
    form = _native_form(tmp_path, "multigeometry", scans="3")
    prepared = prepare_stitch_tool(form)
    assert prepared.requests == () and prepared.request is None
    assert prepared.skipped == ((str(form.spec_path), "3.1",
                                 f"No detector images named *{Path(form.spec_path).stem}_scan3_* "
                                 f"in {form.image_dir}"),)
    owner = StitchToolOwner()
    try:
        owner.set_form(form)
        assert owner.begin_preflight()
        assert _poll_terminal(owner).outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY
        assert owner.begin_run() is None
    finally:
        assert owner.close().status is PageCleanup.CLEAN


def test_native_preview_never_decodes_and_partial_images_refuse(tmp_path, monkeypatch):
    from xrd_tools.sources.spec import SpecSource
    from xrd_tools.io import image as image_api
    form = _native_form(tmp_path, "multigeometry")
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview decoded a detector image")
    monkeypatch.setattr(SpecSource, "load_frame", forbidden)
    monkeypatch.setattr(SpecSource, "load_native_frame", forbidden)
    monkeypatch.setattr(image_api, "read_image", forbidden)
    assert len(prepare_stitch_tool(form).requests) == 3
    next(Path(form.image_dir).glob("*scan2_*")).unlink()
    with pytest.raises(ValueError, match=r"scan 2\.1: 2 metadata points but 1 image frames in 1 files matching "
                                         r"\*sample_scan2_\*"):
        prepare_stitch_tool(form)


def test_native_nexus_group_uses_metadata_and_exact_raw_dataset(tmp_path):
    import h5py
    import json
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    from tests.core.test_stitch_operation import _goniometer_record
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.io.frame_view import read_contributing_frame_view
    form = _native_form(tmp_path, "multigeometry", scans="1")
    files = tuple(_write_bluesky_nxwriter(tmp_path / f"source-{index}.nxs") for index in (1, 2))
    wavelength = 1.2398419843320026
    for path in files:
        with h5py.File(path, "r+") as stream:
            entry = stream["entry"]
            data = entry["data"]
            del data["eiger_image"]
            del entry["instrument/detectors/eiger/data"]
            image = data.create_dataset("eiger_image", data=np.arange(5 * 195 * 487,
                dtype=np.uint32).reshape(5, 195, 487))
            image.attrs["signal_type"] = "detector"
            entry["instrument/detectors/eiger/data"] = image
            data.move("hy", "del")
            data.create_dataset("nu", data=np.zeros(5))
            positions = entry["instrument/positioners"]
            positions.move("hy", "del")
            positions.create_group("nu").create_dataset("value", data=np.zeros(5))
            md = entry["instrument/bluesky/metadata"]
            del md["motors"]
            md.create_dataset("motors", data=b"!!python/tuple\n- del\n- nu\n")
            md["configuration/eiger/data/eiger_cam_wavelength"][()] = wavelength
    geometry = _goniometer_record()
    geometry["wavelength"] = wavelength * 1e-10
    Path(form.geometry_path).write_text(json.dumps(geometry))
    form = replace(form, source_selection=StitchSourceSelection("NeXus", tuple(map(str, files))),
                   detector_shape=(195, 487), raw_dtype="uint32", output_family="joined-nexus")
    prepared = prepare_stitch_tool(form)
    assert len(prepared.requests) == 1
    result = run_stitch_operation(prepared.request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal.diagnostic
    payload = result.payload
    assert len(payload.contributing_frames) == 10
    first = payload.contributing_frames[0]
    scan, frame = first["key"].split("/")
    view = read_contributing_frame_view(result.request.module.output.target,
        int(frame.removeprefix("frame_")), scan=scan.removeprefix("scan_"), source_root=tmp_path,
        full_raw=True, artifact_payload=payload)
    np.testing.assert_array_equal(view.raw, np.arange(195 * 487).reshape(195, 487))


@pytest.mark.parametrize("held", ("nu", "del"))
def test_a_single_arm_xu_calibration_reads_only_its_2theta_motor(tmp_path, held):
    """Preview and Run of an xrayutilities calibration for a fourc/twoc arm: vertical (nu held
    at zero) or horizontal (del held at zero)."""
    import json
    from pyFAI.detectors import detector_factory
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.analysis.xu_stitch_calibration import (
        _canonical_bytes, authored_surface_calibration, canonical_surface_resource_bytes,
    )
    form = _native_form(tmp_path, "xu_hist", scans="1")
    canonical = json.loads(canonical_surface_resource_bytes())
    asset = authored_surface_calibration(
        detector=detector_factory("Pilatus300kw", config={"orientation": 3}),
        cch1=canonical["detector"]["cch1_px"], cch2=canonical["detector"]["cch2_px"],
        distance_m=canonical["detector"]["distance_m"], del_offset_deg=3., nu_offset_deg=1.2,
        energy_eV=canonical["acquisition"]["energy_eV"],
        xu_wavelength_A=canonical["acquisition"]["xu_mapping_wavelength_A"],
        source_motors={"del": "del", "nu": None} if held == "nu" else {"del": None, "nu": "del"}, rotation=0,
        control_domain={"del": [14., 15.], "nu": [0., 0.]} if held == "nu" else {"del": [0., 0.], "nu": [14., 15.]},
        domain={"del": [0., 50.], "nu": [0., 0.]} if held == "nu" else {"del": [0., 0.], "nu": [0., 50.]},
        rms_q=1e-3, condition=10., n_frames=3, n_points=100)
    path = tmp_path / "xu-single.json"
    path.write_bytes(_canonical_bytes(asset))
    mapping = (("del", "del"),) if held == "nu" else (("nu", "del"),)  # the scan's del column is the 2θ
    request = prepare_stitch_tool(replace(form, geometry_path=path, source_motors=mapping)).request
    assert {name for item in request.manifest.contributions for name, _o, _v in item.values} == {"del"}
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal


def test_an_energy_refined_calibration_checks_the_scans_against_their_recorded_energy(tmp_path):
    """Maintainer, 2026-09-29: "add a refine energy toggle". A calibration refined to 17 100 eV keeps the
    17 000.018 eV the scans recorded (source_energy_eV, in the pyFAI file and its xrayutilities partner):
    both backends check the scans against it, and Stitch runs; without it, the scans are refused."""
    import json
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.integrate.stitch_calibration import CalibrationFit, save_calibration_pair
    from xdart.gui.tools.stitch_values import StitchToolPreflightRefused
    form = _native_form(tmp_path, "xu_hist", scans="1")
    calibration = DetectorCalibration(PONI(dist=.39, poni1=.02, poni2=.04, detector="Pilatus300kw",
                                           wavelength=energy_eV_to_wavelength_m(17100.)),
                                      detector_config={"orientation": 3})
    record = StackedDetectorArm(nu_motor="nu", del_motor="del", nu_offset=.01,
                                del_offset=-.02).to_pyfai_goniometer(calibration)
    record["detector_config"] = {"orientation": 3}

    def pair(name, **extra):
        fit = CalibrationFit("multigeometry", {"calibration": {**record, **extra}, "provenance": {
            "source_motors": {"del_angle": "del", "nu_angle": "nu"}, "image_rotation": 0, "frames": []}},
            1e-3, 10., 100, "")
        return save_calibration_pair(fit, tmp_path / "calibration" / f"{name}.json",
                                     validation_angles=[(12., -12.), (14., -10.), (17., -8.)])

    refined = pair("refined", source_energy_eV=17000.018)
    partner = json.loads(Path(refined.xu_path).read_text())["acquisition"]
    assert partner["source_energy_eV"] == 17000.018 and partner["energy_eV"] == pytest.approx(17100.)
    mg = replace(form, backend="multigeometry", source_motors=(("del_angle", "del"), ("nu_angle", "nu")))
    for backend_form in (form, mg):
        prepared = prepare_stitch_tool(replace(backend_form, geometry_path=Path(refined.pyfai_path)))
        assert prepared.requests[0].provenance["source_energy_checks"]  # the scans' energies were checked
        result = run_stitch_operation(prepared.requests[0])
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    unrecorded = pair("unrecorded")
    for backend_form in (form, mg):
        with pytest.raises(StitchToolPreflightRefused, match="ENERGY"):
            prepare_stitch_tool(replace(backend_form, geometry_path=Path(unrecorded.pyfai_path)))
    # Codex re-review R2-F1: a recorded energy no comparison can fail (NaN, infinity, a string) or one
    # far from the calibration's own is refused at Preview, never taken as the reference
    for index, bad in enumerate(("nan", "inf", "17000", 16000.)):
        saved = pair(f"bad{index}", source_energy_eV=bad)
        assert saved.xu_refusal  # its xrayutilities partner is refused too
        with pytest.raises(StitchToolPreflightRefused, match="STITCH_CALIBRATION_ENERGY_INVALID"):
            prepare_stitch_tool(replace(mg, geometry_path=Path(saved.pyfai_path)))


def test_an_xu_stitch_reads_the_partner_saved_beside_a_pyfai_calibration(tmp_path):
    """Always both (maintainer, 2026-09-29): Backend xrayutilities with the Calibration's pyFAI file
    reads the xrayutilities file saved with it; a pyFAI file saved alone (as the fits before, e.g. the
    −0.87° test calibration) is refused with what to do."""
    import hashlib
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.integrate.stitch_calibration import CalibrationFit, save_calibration_fit, save_calibration_pair
    form = _native_form(tmp_path, "xu_hist", scans="1")
    calibration = DetectorCalibration(PONI(dist=.39, poni1=.02, poni2=.04, detector="Pilatus300kw",
                                           wavelength=energy_eV_to_wavelength_m(17000.018)),
                                      detector_config={"orientation": 3})
    record = StackedDetectorArm(nu_motor="nu", del_motor="del", nu_offset=.01,
                                del_offset=-.02).to_pyfai_goniometer(calibration)
    record["detector_config"] = {"orientation": 3}
    fit = CalibrationFit("multigeometry", {"calibration": record, "provenance": {
        "source_motors": {"del_angle": "del", "nu_angle": "nu"}, "image_rotation": 0, "frames": []}},
        1e-3, 10., 100, "")
    saved = save_calibration_pair(fit, tmp_path / "calibration" / "arm.json",
                                  validation_angles=[(12., -12.), (14., -10.), (17., -8.)])
    prepared = prepare_stitch_tool(replace(form, geometry_path=Path(saved.pyfai_path)))
    receipt = prepared.requests[0].plan.calibration
    assert receipt.raw_sha256 == hashlib.sha256(Path(saved.xu_path).read_bytes()).hexdigest()
    result = run_stitch_operation(prepared.requests[0])
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    alone = save_calibration_fit(fit, tmp_path / "calibration" / "alone.json")
    with pytest.raises(ValueError, match=r"alone.json has no xrayutilities partner \(alone.xu.json\): save"):
        prepare_stitch_tool(replace(form, geometry_path=Path(alone)))
