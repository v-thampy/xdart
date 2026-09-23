"""Native finite selections: stable slots, multiple exact jobs and raw reads."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tools.stitch_owner import StitchOwnerOutcomeKind, StitchToolOwner
from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_sources import StitchSourceSelection
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from tests.xdart.pages.test_stitch_page import _form
from tests.xdart.tools.test_stitch_owner import _poll_terminal


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
        unit=unit, q_range=None, conditioning=StitchConditioning("auto"))


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
@pytest.mark.parametrize("unit", ("q_A^-1", "2th_deg"))
def test_native_preflight_run_jobs_raw_and_stable_replacement(tmp_path, backend, unit):
    form = replace(_native_form(tmp_path, backend, unit=unit),
                   output_path=tmp_path / "new-save-folder" / "result_stitch1d.nexus")
    prepared = prepare_stitch_tool(form)
    assert len(prepared.requests) == 3
    assert prepared.summary is None and prepared.request is prepared.requests[0]
    assert [len(item.manifest.contributions) for item in prepared.requests] == [2, 2, 4]
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
    assert prepared.skipped == ((str(form.spec_path), "3.1", "No detector images"),)
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
    with pytest.raises(ValueError, match="image count"):
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
                   detector_shape=(195, 487), raw_dtype="uint32")
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
