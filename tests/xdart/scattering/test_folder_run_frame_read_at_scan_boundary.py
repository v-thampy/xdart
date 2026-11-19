"""A directory Int 1D Run goes on while a frame read is still running at a scan boundary (LIVE-1212-LIGHT1D-LINEAGE;
maintainer, 2026-11-18, from the beamline: "Keep getting this crap in int 1d in directory mode", a Run on an X: share
failing with "prior light-1D lineage cleanup remains pending" after 9 of 170 frames).

Releasing a finished scan's light-1D curves refused while any frame read of the Run was still running, and the Run
failed on that one refusal. These Folder Runs use a real page, real TIFF and .txt inputs, and the acquisition
display's request methods. Events hold the worker at a durable boundary and control when the real read resumes."""
from __future__ import annotations

import os
from threading import Event

import pytest

pytest.importorskip("pyqtgraph")

import xdart.gui.tabs.scattering.hydration_transport as hydration_transport
from tests.xdart.scattering.test_browse_reference_rings_run import _page, _poni, _pump, _terminal
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _scans(tmp_path, count, frames):
    raw = tmp_path / "raw"
    raw.mkdir()
    stems = tuple(f"s{index:02d}" for index in range(count))
    for index, stem in enumerate(stems):
        for frame in range(1, frames + 1):
            _frame(raw, stem, frame, "txt", float(index + 1))
    intent = RunIntent(source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
                       poni_file=str(_poni(tmp_path)), project_root=str(raw), save_path=str(tmp_path / "processed"),
                       output_mode="Overwrite", processing_mode="Int 1D", live_mode=False, max_cores=1,
                       bai_1d_args={"npt": 8, "method": "numpy"})
    return stems, intent


def _outputs(page):
    run = page._run_executor._exact_run(page._lifecycle.active_run_identity)
    catalog = page._run_executor.frame_catalog(page._lifecycle.active_run_identity)
    entries = () if catalog is None else catalog.entries
    return run, entries


def _written(tmp_path, stems):
    return sorted(path.name for path in (tmp_path / "processed").glob("*_int1d.nexus")) == \
        [f"{stem}_int1d.nexus" for stem in stems]


@pytest.mark.parametrize("same_output", (False, True), ids=("earlier-scan", "finishing-scan"))
def test_a_folder_run_at_a_boundary_with_a_saved_frame_read(qapp, tmp_path, monkeypatch, same_output):  # noqa: F811
    """Hold a real physical read of a durable frame when the worker releases the second scan."""
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    stems, intent = _scans(tmp_path, 3, 24)
    retiring = f"{stems[1]}_int1d.nexus"
    selected = f"{stems[int(same_output)]}_int1d.nexus"
    boundary, start_release, reading, finish_read = (Event() for _ in range(4))
    attempts = []
    keep = RunDisplayState.keep_released_1d
    release = RunDisplayState.release_light_1d
    read = hydration_transport.read_frame_preview

    def hold_boundary(self, owner):
        count = keep(self, owner)
        if owner.artifact.name == retiring:
            boundary.set()
            assert start_release.wait(10), "the test never released the boundary"
        return count

    def hold_read(key, *args, **kwargs):
        if os.path.basename(key.artifact_identity) == selected and key.frame_identity == 1:
            reading.set()
            assert finish_read.wait(10), "cleanup never attempted to release the scan"
        return read(key, *args, **kwargs)

    def record_release(self, owner, **kwargs):
        clean = release(self, owner, **kwargs)
        if owner.artifact.name == retiring and reading.is_set():
            receipt = None if clean else owner.light_lease.cleanup_receipt
            attempts.append((clean, None if receipt is None else receipt.failed_step))
            finish_read.set()
        return clean

    monkeypatch.setattr(RunDisplayState, "keep_released_1d", hold_boundary)
    monkeypatch.setattr(RunDisplayState, "release_light_1d", record_release)
    monkeypatch.setattr(hydration_transport, "read_frame_preview", hold_read)
    with _page(qapp, intent) as (page, _view, observed):
        try:
            page._shell.run_controls.startButton.click()
            assert _pump(qapp, boundary.is_set, seconds=15)
            run, entries = _outputs(page)
            frame = next(key for key in entries if os.path.basename(key.artifact) == selected
                         and key.local_frame_label == 1)
            owner = run.display.artifacts[frame.artifact]
            context = run.context_runtime.context
            run.display._request_preview(owner, frame, 101, True, context.hydration_owner, context.commit_gate)
            assert _pump(qapp, reading.is_set, seconds=6)
            start_release.set()
            _terminal(qapp, page, observed)
        finally:
            start_release.set()
            finish_read.set()
        assert attempts[0] == ((False, "drain") if same_output else (True, None)), attempts
        assert attempts[-1] == (True, None), attempts
        assert _written(tmp_path, stems)


def test_full_raw_cannot_enter_after_the_finishing_scan_has_drained(qapp, tmp_path, monkeypatch):  # noqa: F811
    """Hold the real release after drain has accepted idle stores, then try an exact Full Raw request."""
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    stems, intent = _scans(tmp_path, 2, 24)
    first = f"{stems[0]}_int1d.nexus"
    drained, finish_release = Event(), Event()
    drain = StandardRunExecutor._drain_light_lineage

    def hold_after_drain(self, run, owner):
        drain(self, run, owner)
        if owner.artifact.name == first:
            drained.set()
            assert finish_release.wait(10), "the test never released the drain"

    monkeypatch.setattr(StandardRunExecutor, "_drain_light_lineage", hold_after_drain)
    with _page(qapp, intent) as (page, _view, observed):
        try:
            page._shell.run_controls.startButton.click()
            assert _pump(qapp, drained.is_set, seconds=15)
            run, entries = _outputs(page)
            frame = entries[0]
            assert os.path.basename(frame.artifact) == first and frame.local_frame_label == 1
            assert not run.display.full_raw_status(frame)[0]
            context = run.context_runtime.context
            token = run.display.request_full(frame, 1, owner=context.hydration_owner, commit_gate=context.commit_gate)
            assert token is None, "Full Raw was admitted into a store whose lease is fenced for release"
            assert run.display.full_raw_status(frame) == (False, False, None)
        finally:
            finish_release.set()
        _terminal(qapp, page, observed)
        assert _written(tmp_path, stems)
