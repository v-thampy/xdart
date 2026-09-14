"""Named GI modes survive first Browse hydration on the retained run transport."""

from dataclasses import replace
import time

import h5py
import numpy as np
import pytest

from tests.core.v2_fixture_factory import current_entry
from tests.xdart.scattering.test_e3_context_contract import _running_controller
from tests.xdart.scattering.test_e4_preview_transport import _wait_transport_idle
from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
from xdart.gui.tabs.scattering.browse_1d_display import prepare_browse_1d_display
from xdart.gui.tabs.scattering.browse_1d_projection import Browse1DProjectionStatus
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.events import CleanupStatus, DurableFinal, ExecutionEnded
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xrd_tools.core import FrameRecord, FrameView, TwoDKind, axis_from_unit
from xrd_tools.io import FrameViewReader, write_frame_records
from xrd_tools.io.nexus_record import ensure_frames_container, write_frame_record


def _write_named_gi(path):
    records = []
    for label in range(1, 17):
        view = FrameView(
            label,
            axis_1d=axis_from_unit("qip_A^-1", np.array([0.1, 0.2, 0.3])),
            intensity_1d=np.array([1.0, 2.0, 3.0]) + label,
            sigma_1d=np.array([0.1, 0.2, 0.3]),
            axis_2d_x=axis_from_unit("qip_A^-1", np.array([0.1, 0.2, 0.3])),
            axis_2d_y=axis_from_unit("qoop_A^-1", np.array([0.4, 0.8])),
            intensity_2d=np.arange(6.0).reshape(2, 3) + label,
            two_d_kind=TwoDKind.QIP_QOOP,
        )
        record = FrameRecord.from_view(view, mode_1d="q_ip", mode_2d="qip_qoop")
        record = record.with_result_1d(
            "q_oop",
            replace(
                view,
                axis_1d=axis_from_unit("qoop_A^-1", np.array([1.0, 2.0, 3.0])),
                intensity_1d=np.array([100.0, 200.0, 300.0]) + label,
            ),
            make_active=False,
        )
        records.append(record)
    with h5py.File(path, "w") as handle:
        entry = current_entry(handle)
        write_frame_records(entry, records)
        frames = ensure_frames_container(entry)
        for label in range(1, 17):
            write_frame_record(
                frames, f"frame_{label:04d}",
                thumbnail=np.array([[0, 20], [40, 80]], dtype=np.uint8) + label,
            )


@pytest.mark.parametrize("first_mode", ("Waterfall", "Single"))
@pytest.mark.parametrize("schema_version", (2, 3))
def test_first_borrowed_gi_preview_and_frame9_preserve_persisted_modes(
    tmp_path, first_mode, schema_version,
):
    path = tmp_path / "named_gi.nexus"
    _write_named_gi(path)
    if schema_version == 2:
        from tests.core.test_neutral_axis_contract import _as_v2
        _as_v2(path)
    with FrameViewReader(path) as reader:
        expected = {label: reader.read(label) for label in (16, 9)}

    _, lifecycle, executor, _, acquisition = _running_controller(gi=True)
    display = acquisition.publication_store
    loader = BrowseLoader(max_items=16)
    controller = ContextController(
        lifecycle=lifecycle, executor=executor, browse_loader=loader,
        projection=ContextProjection(),
    )
    controller.adopt_acquisition(executor.identity)
    lifecycle.execution_ended(ExecutionEnded(executor.identity))
    assert lifecycle.durable_final(DurableFinal(executor.identity)).phase.value == "idle"
    browse = None
    runtime = None
    try:
        request = controller.begin_browse(str(path))
        deadline = time.monotonic() + 10.0
        outcome = None
        while outcome is None and time.monotonic() < deadline:
            outcome = controller.poll_browse()
            if outcome is None:
                time.sleep(0.005)
        assert outcome is not None and outcome.request is request
        browse = controller.browse_context
        assert browse is not None and browse.loaded
        owner = controller._browse_hydration_owner
        assert not owner.owns_transport
        assert owner.transport is display.transport
        assert owner.transport._commit.__self__ is display
        assert browse.publication_store.get(16) is None
        assert browse.publication_store.get(9) is None
        frames = controller.frame_keys
        other_mode = "Single" if first_mode == "Waterfall" else "Waterfall"

        for label in (16, 9):
            current = next(frame for frame in frames if frame.local_frame_label == label)
            assert browse.publication_store.get(label) is None
            assert controller.select_navigation(current, frames)
            assert controller.request_current_browse_preview() is None
            assert _wait_transport_idle(display)
            controller.poll_browse_preview()
            publication = browse.publication_store.get(label)
            assert publication is not None
            assert publication.record.active_mode_1d == "q_ip"
            assert publication.record.active_mode_2d == "qip_qoop"
            assert tuple(publication.record.results_1d) == ("q_ip",)
            assert tuple(publication.record.results_2d) == ("qip_qoop",)

            for mode in (first_mode, other_mode):
                # Single honors explicit multi-selection too. Model the
                # ordinary single click when requesting its one-row oracle.
                selected = (current,) if mode == "Single" else frames
                assert controller.select_navigation(current, selected)
                preferences = ScientificPreferences(plot_mode=mode)
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    runtime = controller.project_browse_1d_cache(
                        preferences=preferences, was_waterfall_active=mode == "Waterfall",
                    )
                    assert runtime is not None
                    if runtime.status is not Browse1DProjectionStatus.INCOMPLETE:
                        break
                    assert runtime.borrow_bundle is None
                    time.sleep(0.005)
                assert runtime.status is Browse1DProjectionStatus.COMPLETE, runtime.diagnostic
                assert runtime.plan.plot_mode == mode
                assert len(runtime.payloads) == (16 if mode == "Waterfall" else 1)
                payload = next(item for item in runtime.payloads if item.frame_key is current)
                assert payload.gi_mode_1d == "q_ip"
                assert payload.gi_mode_2d == "qip_qoop"
                for name in ("axis_1d", "axis_2d_x", "axis_2d_y"):
                    actual_axis, expected_axis = getattr(payload.view, name), getattr(expected[label], name)
                    assert (actual_axis.label, actual_axis.unit) == (expected_axis.label, expected_axis.unit)
                    np.testing.assert_array_equal(actual_axis.values, expected_axis.values)
                np.testing.assert_array_equal(payload.view.intensity_1d, expected[label].intensity_1d)
                np.testing.assert_array_equal(payload.view.sigma_1d, expected[label].sigma_1d)
                assert payload.view.intensity_2d is publication.view.intensity_2d
                assert payload.view.thumbnail is publication.view.thumbnail
                np.testing.assert_array_equal(payload.view.intensity_2d, expected[label].intensity_2d)
                np.testing.assert_array_equal(payload.view.thumbnail, expected[label].thumbnail)
                adopted = prepare_browse_1d_display(runtime)
                assert adopted is not None
                assert runtime.borrow_bundle.released
                assert runtime.borrow_bundle.remaining == 0
                # Hydration protects cached rows with its own borrows until
                # worker cleanup; display release does not retire those pins.
                deadline = time.monotonic() + 10.0
                while owner.polling_needed() and time.monotonic() < deadline:
                    controller.poll_browse_preview()
                    time.sleep(0.005)
                assert not owner.polling_needed()
                assert browse.browse_1d_cache.outstanding_borrows == 0
                runtime = None

        publication = browse.publication_store.get(9)
        incompatible = replace(
            publication,
            record=FrameRecord.from_view(
                publication.view, mode_1d="q_oop", mode_2d="qip_qoop",
            ),
        )
        browse.publication_store.upsert(incompatible)
        runtime = controller.project_browse_1d_cache(
            preferences=ScientificPreferences(plot_mode="Single"),
            was_waterfall_active=False,
        )
        assert runtime.status is Browse1DProjectionStatus.REFUSED
        assert runtime.diagnostic == "_Refused: Browse sparse active 1d mode changed"
        assert not runtime.payloads
        assert browse.browse_1d_cache.outstanding_borrows == 0
    finally:
        if runtime is not None and runtime.borrow_bundle is not None:
            runtime.borrow_bundle.release()
        assert display.retire(join_timeout=10.0)
        if browse is not None:
            receipt = controller._browse_hydration_owner.release(loader, browse)
            assert receipt.cleanup_status is CleanupStatus.CLEANED
