"""Frozen E4-R oracle: one typed bounded preview transport for vNext.

Freezes the §6 rows E4-R0 owns (handoff
``x1_scattering_workspace_e4_bounded_preview_handoff_2026-07-29.md``): the
one-active/one-latest transport, terminal-total completion accounting, the
atomic target commit, and the acquisition/Browse/display coherence rows.  The
E4-S shared rows (``tests/core/test_hydration_contract.py`` /
``test_frame_preview.py`` / ``test_import_purity.py``) remain retained
sentinels and are not duplicated here.

Frozen production surface (implemented by E4-R1/E4-R2):

* ``xdart.gui.tabs.scattering.hydration_transport.HydrationTransport`` —
  ``submit(request, *, closed=False) -> HydrationToken | None`` (the frozen
  detector projection is derived ONCE at submit from the exact request-carried
  target; a Browse target without persisted mask provenance derives the
  unavailable projection), ``active_token``/``queued_token``/``worker``,
  ``completions()``, ``counters()``, ``cancel_gate(gate)``,
  ``retains_gate(gate)``, ``retire(join_timeout)``; module value
  ``PreparedHydrationCommit``; the queued entry (``_queued``) carries exactly
  ``request``/``projection``/presentation facts.
* ``RunDisplayState.transport`` (one owner, composed in), ``commit_preview``
  as the ONE target-port operation, ``bind_transport(event_sink=...)``,
  ``project(frame, generation, closed=..., owner=..., commit_gate=...)``
  as the acquisition issuer, and ``DisplayArtifact.saturation_ceiling``
  stamped once by ``stamp_saturation_ceiling``.

Every Qt process obeys repository rule 10 (unique ``XDART_SESSION_FILE``).
"""

from __future__ import annotations

import threading
import time
from importlib import import_module
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io import read_frame_record
from xrd_tools.io.nexus import write_integrated_stack
from xrd_tools.io.nexus_record import (
    ensure_frames_container,
    stamp_source_base,
    write_frame_record,
)
from xrd_tools.session.hydration import (
    HydrationCompletion,
    HydrationOutcome,
    HydrationPurpose,
    HydrationReadKey,
    HydrationScope,
    HydrationToken,
)
from xdart.modules.display_context import (
    AcquisitionContext,
    CommitGate,
    ContextKind,
    HydrationOwner,
    HydrationRequest,
    new_context_token,
)
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.context_runtime import _ContextRuntime
from xdart.gui.tabs.scattering import display_runtime
from xdart.gui.tabs.scattering.display_runtime import (
    DetectorHydrationOutcome,
    RunDisplayState,
)
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)


def _transport_api():
    """The E4-R transport module (absent at the frozen parent — honest red)."""
    return import_module("xdart.gui.tabs.scattering.hydration_transport")


# --------------------------------------------------------------------------- #
# Real processed artifacts
# --------------------------------------------------------------------------- #

def _write_processed(
    root: Path,
    *,
    labels: tuple[int, ...] = (1, 2, 3),
    thumbnails: bool = True,
    two_d: bool = True,
    raw_dtype=np.uint16,
    schema_version=2,
) -> tuple[Path, Path]:
    """One real processed container + one real raw master, portable layout."""
    raw = np.arange(16, dtype=raw_dtype).reshape(4, 4)
    raw_path = root / "raw" / "image.tif"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(raw_path, raw)
    processed = root / "xdart_processed_data" / "scan.nexus"
    processed.parent.mkdir(exist_ok=True)
    one_d = [
        IntegrationResult1D(
            radial=np.array([0.1, 0.2, 0.3]),
            intensity=np.array([1.0, 2.0, 3.0]) + label,
            unit="q_A^-1",
        )
        for label in labels
    ]
    two_d_results = [
        IntegrationResult2D(
            radial=np.array([0.1, 0.2, 0.3]),
            azimuthal=np.array([-1.0, 1.0]),
            intensity=np.arange(6, dtype=float).reshape(3, 2) + label,
            unit="q_A^-1",
            azimuthal_unit="chi_deg",
        )
        for label in labels
    ]
    with h5py.File(processed, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["ssrl_schema"] = "xrd_tools.processed_scan"
        entry.attrs["ssrl_schema_version"] = schema_version
        write_integrated_stack(
            entry,
            frame_indices=list(labels),
            results_1d=one_d,
            results_2d=two_d_results if two_d else None,
        )
        base = stamp_source_base(entry, root)
        for label in labels:
            write_frame_record(
                ensure_frames_container(entry),
                f"frame_{label:04d}",
                thumbnail=(raw[::2, ::2] if thumbnails else None),
                source_path=raw_path,
                source_frame_index=0,
                source_base=base,
            )
    return processed, raw_path


def _instrument_reads(monkeypatch, processed: Path):
    """Count exact processed-container opens and detector-source reads."""
    frame_view = import_module("xrd_tools.io.frame_view")
    preview = import_module("xrd_tools.io.frame_preview")
    original_open = frame_view.h5py.File
    original_read = preview.read_image
    counts = {"processed": 0, "detector": 0, "read_threads": []}

    def counted_open(path, *args, **kwargs):
        if Path(path) == Path(processed):
            counts["processed"] += 1
            counts["read_threads"].append(threading.current_thread().name)
        return original_open(path, *args, **kwargs)

    def counted_read(path, *args, **kwargs):
        counts["detector"] += 1
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(frame_view.h5py, "File", counted_open)
    monkeypatch.setattr(preview, "read_image", counted_read)
    return counts


# --------------------------------------------------------------------------- #
# Production-owner harness (real RunDisplayState + real files)
# --------------------------------------------------------------------------- #

def _state(
    processed: Path,
    *,
    identity=None,
    mask_saturation=False,
    catalog_max_items=None,
):
    state = RunDisplayState(
        identity or RunIdentity(1, "e4-preview"),
        max_payload_items=4,
        catalog_max_items=catalog_max_items,
    )
    state.configure(partition_count=1, npt=8, frame_bytes=64)
    owner = state.add_artifact(
        processed,
        "scan",
        mask=None,
        mask_saturation=mask_saturation,
        measurement_mode="Standard",
    )
    return state, owner


def _catalog(state, owner, labels):
    keys = {}
    for label in labels:
        if owner.records.get(label) is None and Path(owner.artifact).is_file():
            record = read_frame_record(owner.artifact, label)
            view = record.active_view()
            source_identity = (
                f"{view.source_path or ''}#{view.source_frame_index}"
            )
            owner.records.upsert(
                record,
                source_identity=source_identity,
            )
            modes = tuple(
                [("1d", mode) for mode in record.results_1d]
                + [("2d", mode) for mode in record.results_2d]
            )
            owner.records.replace_projection(
                label,
                hydratable=modes,
                durable=modes,
            )
        delta = state.append_navigation(
            owner.source_scan, str(owner.artifact), label
        )
        keys[label] = delta.appended
    return keys


def _acquisition_identity(state):
    """A qualified production owner + gate for acquisition-shaped rows."""
    gate = CommitGate()
    owner = HydrationOwner("ctx-acq", "scan", "/raw/source", gate.epoch)
    return owner, gate


def _typed_request(
    state,
    owner_value: HydrationOwner,
    gate,
    artifact: Path,
    label: int,
    purpose: HydrationPurpose,
    generation: int,
    *,
    stores: tuple | None = None,
    checkpoint: bool = False,
):
    scope = HydrationScope(*owner_value.as_tuple())
    read_key = HydrationReadKey(scope, str(artifact), label, purpose)
    token = HydrationToken(read_key, generation)
    art = state.artifacts[str(artifact)]
    checkpoint_token = checkpoint_gate = None
    if checkpoint:
        checkpoint_token, checkpoint_gate = (
            art.records._checkpoint_hydration_authority()
        )
        assert checkpoint_token is not None and checkpoint_gate is not None
    return HydrationRequest(
        label,
        purpose,
        generation,
        owner_value,
        stores
        if stores is not None
        else (art.records, art.publications),
        gate,
        read_key=read_key,
        token=token,
        checkpoint_token=checkpoint_token,
        checkpoint_gate=checkpoint_gate,
    )


def _bound_state(monkeypatch, tmp_path, **kwargs):
    processed, raw_path = _write_processed(tmp_path, **kwargs)
    state, owner = _state(processed)
    events = []
    state.bind_transport(event_sink=events.append)
    return state, owner, processed, raw_path, events


def _wait_transport_idle(state, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        transport = state.transport
        # Read QUEUED before ACTIVE: an entry only ever moves
        # queued -> active -> done, so this observation order is monotone and
        # (None, None) proves quiescence without a wider lock.
        if transport.queued_token is None and transport.active_token is None:
            return True
        time.sleep(0.005)
    return False


def _completion_outcomes(state):
    return tuple(
        (completion.token.read_key.frame_identity, completion.outcome)
        for completion in state.transport.completions()
    )


# --------------------------------------------------------------------------- #
# Row 1/2 — thumbnail/cake path and exact evicted-frame identity
# --------------------------------------------------------------------------- #

def test_thumbnail_preview_one_open_zero_detector_reads_and_no_full_raw(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, events = _bound_state(monkeypatch, tmp_path)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    counts = _instrument_reads(monkeypatch, processed)

    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 5
    )
    token = state.transport.submit(request)
    assert type(token) is HydrationToken
    assert _wait_transport_idle(state)

    assert counts["processed"] == 1
    assert counts["detector"] == 0
    assert counts["read_threads"]  # the read happened…
    assert all(  # …and never on the submitting/GUI thread
        name != threading.main_thread().name
        for name in counts["read_threads"]
    )
    publication = owner.publications.get(2)
    assert publication is not None
    assert publication.view.thumbnail is not None
    assert publication.view.intensity_1d is not None
    assert publication.view.intensity_2d is not None
    assert publication.view.raw is None  # PREVIEW retains no full raw
    assert (2, HydrationOutcome.HYDRATED) in _completion_outcomes(state)
    assert keys[2] in state.payloads


def test_absent_frame_fails_typed_without_alias_or_invention(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)

    request = _typed_request(
        state, owner_value, gate, processed, 9, HydrationPurpose.PREVIEW, 3
    )
    token = state.transport.submit(request)
    assert type(token) is HydrationToken
    assert _wait_transport_idle(state)

    assert owner.publications.get(9) is None
    outcomes = dict(_completion_outcomes(state))
    assert outcomes.get(9) is HydrationOutcome.FAILED
    completion = next(
        item
        for item in state.transport.completions()
        if item.token.read_key.frame_identity == 9
    )
    assert type(completion) is HydrationCompletion
    assert completion.diagnostic


# --------------------------------------------------------------------------- #
# Row 3 — no-thumbnail fallback truth and fail-closed
# --------------------------------------------------------------------------- #

def test_no_thumbnail_preview_keeps_science_without_detector_fallback(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(
        monkeypatch, tmp_path, thumbnails=False
    )
    _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    counts = _instrument_reads(monkeypatch, processed)

    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 4
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)

    assert counts["processed"] == 1
    assert counts["detector"] == 0
    publication = owner.publications.get(2)
    assert publication is not None
    assert publication.view.raw is publication.view.thumbnail is None
    assert publication.view.has_1d and publication.view.has_2d
    assert (2, HydrationOutcome.HYDRATED) in _completion_outcomes(state)


@pytest.mark.parametrize("with_2d", (False, True), ids=("Int1D", "Int2D"))
def test_live_thumbnail_carrier_is_exact_complete_and_snapshot_ordered(
    monkeypatch, tmp_path, with_2d
):
    from queue import Queue; from types import SimpleNamespace; import xrd_tools.reduction.core as reduction_core
    from xrd_tools.reduction import Frame, Integration1DPlan, Integration2DPlan, NexusSink, ReductionPlan, Scan
    from xrd_tools.session import ScanSession
    from xdart.gui.tabs.scattering.adapters import dynamic_output; from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor, _StandardRun

    one = IntegrationResult1D(np.array([0., 1.]), np.array([2., 3.]), unit="q_A^-1")
    two = IntegrationResult2D(np.array([0., 1.]), np.array([-1., 1.]),
                              np.arange(4.).reshape(2, 2), unit="q_A^-1",
                              azimuthal_unit="chi_deg")
    monkeypatch.setattr(reduction_core, "integrate_1d", lambda *_a, **_k: one)
    monkeypatch.setattr(reduction_core, "integrate_2d", lambda *_a, **_k: two)
    image = np.arange(48, dtype=np.uint16).reshape(8, 6)
    mask = np.zeros(image.shape, dtype=bool); mask[:2, :2] = True
    frame = Frame(1, image=image, mask=mask)
    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=2), integration_2d=Integration2DPlan() if with_2d else None); scan = Scan("live", [frame], integrator=object())
    target = tmp_path / "live.nexus"
    run = _StandardRun(None, RunIdentity(1, "perf-c5"), scan, None, None, None, target)
    owner = run.display.add_artifact(target, "live", mask=None, mask_saturation=False, measurement_mode="Standard")
    policy, layout, rows, ceiling, _ = dynamic_output._light_policy_layout(SimpleNamespace(max_cores=1, live_mode=False, gi=SimpleNamespace(enabled=False)), plan, SimpleNamespace(descriptor=None), scan, (1,))
    lease = dynamic_output.acquire_light_1d_retention(dynamic_output.SessionResourceAuthority.from_allocation(policy.allocation), owner="scattering-light-1d:perf-c5", generation=1, layout=layout, requested_rows=rows, compatibility_byte_ceiling=ceiling, gui_thread_id=run.gui_thread_id, funding_mode=dynamic_output.Light1DFundingMode.REPLACE_PUBLICATION_A1, current_lineage_rows=1)
    owner.publications.bind_allocation(policy.allocation); owner.publications.bind_light_1d(lease)
    hooks = owner.publications.light_1d_cleanup_hooks(lease)
    run.display.stage_light_1d(owner, lease, hooks=hooks); run.display.bind_light_1d(owner, lease)
    prepared, writes = [], []
    original_prepare, original_write = (NexusSink._prepare_frame_thumbnail, NexusSink._write_frame_record)

    def observe_prepare(self, owned_frame, **kwargs):
        prepared.append(owned_frame.index); return original_prepare(self, owned_frame, **kwargs)

    def observe_write(self, owned_frame, reduction, **kwargs):
        value = original_write(self, owned_frame, reduction, **kwargs)
        writes.append((reduction, value)); return value

    monkeypatch.setattr(NexusSink, "_prepare_frame_thumbnail", observe_prepare)
    monkeypatch.setattr(NexusSink, "_write_frame_record", observe_write)
    session = ScanSession(plan, scan, sink=NexusSink(target, overwrite=True, thumbnail_max=4),
                          executor=1, record_store=owner.records)
    run.session, run.records, run.frames_by_label[1] = session, owner.records, frame
    session.submit(frame)
    session.finish()
    reduction, record_write = writes[0]; view = owner.records.get(1).active_view(); thumbnail = reduction.thumbnail
    assert prepared == [1] and record_write.thumbnail is thumbnail is view.thumbnail
    assert thumbnail.dtype == np.float32 and not thumbnail.flags.writeable
    assert reduction._thumbnail_mask_baked is record_write.thumbnail_mask_baked is view.mask_baked is True
    assert np.isnan(thumbnail[0, 0]) and np.isfinite(thumbnail[-1, -1])
    assert view.raw is None and view.has_1d and view.has_2d is with_2d
    assert view.extra["detector_shape"] == image.shape

    run.total = run.current_total = 1
    executor = StandardRunExecutor(); executor._active = run
    pending = run.display_projection_queue = Queue()
    executor._frame_ready(run, SimpleNamespace(frame_index=1)); queued = pending.get_nowait()
    assert queued.frame_index == 1 and queued.record is owner.records.get(1)
    executor._frame_ready_owned(run, queued, None, None)
    event = executor.drain_events()[-1]; payload = run.display.payloads[event.frame_key]
    assert payload.view.thumbnail is thumbnail and payload.view.raw is None
    assert run.display.project(event.frame_key, 0, closed=False) is not None
    assert sum(run.display.transport.counters().values()) == 0
    run.display_projection_queue = None
    order, entered, release = [], threading.Event(), threading.Event()
    project, mark = executor._frame_ready_owned, run.display.mark_checkpoint_recoverable
    def delayed(*args):
        order.append("frame"); entered.set(); assert release.wait(2); return project(*args)
    def checkpoint(*args):
        order.append("checkpoint"); return mark(*args)
    monkeypatch.setattr(executor, "_frame_ready_owned", delayed)
    monkeypatch.setattr(run.display, "mark_checkpoint_recoverable", checkpoint)
    executor._start_display_projection(run)
    executor._frame_ready(run, SimpleNamespace(frame_index=1))
    executor._checkpoint_ready(run, SimpleNamespace(labels=(1,)))
    assert entered.wait(2) and order == ["frame"]
    release.set(); assert executor._drain_display_projection(run, 2)
    executor._finish_display_projection(run)
    assert order == ["frame", "checkpoint"]
    monkeypatch.setattr(executor, "_frame_ready_owned", project)
    monkeypatch.setattr(run.display, "mark_checkpoint_recoverable", mark)
    pending = run.display_projection_queue = Queue()
    error = RuntimeError("stamp failed"); monkeypatch.setattr(run.display, "stamp_saturation_ceiling",
                        lambda *_a, **_k: (_ for _ in ()).throw(error))
    executor._frame_ready(run, SimpleNamespace(frame_index=1))
    assert run.display_projection_errors == [error] and pending.empty()
    run.display.artifacts.pop(str(target))
    with pytest.raises(RuntimeError, match="exact owner"):
        executor._frame_ready_owned(run, queued, None, None)

    if not with_2d:
        supplied = np.arange(6, dtype=np.float32).reshape(2, 3)
        direct = reduction_core.FrameReduction(2, result_1d=one, thumbnail=supplied,
                                                _thumbnail_mask_baked=True)
        direct_frame = Frame(2, image=np.ones((2, 3)))
        direct_sink = NexusSink(tmp_path / "direct.nexus", overwrite=True)
        direct_sink.begin(Scan("direct", [direct_frame]), plan)
        before = len(prepared)
        direct_sink.write_batch(((direct_frame, direct),))
        direct_sink.finish(reduction_core.ReductionResult("direct", {}, 1))
        assert len(prepared) == before
        assert writes[-1][1].thumbnail is supplied
        assert writes[-1][1].thumbnail_mask_baked is True


def test_thumbnail_semilight_missing_science_rehydrates_once(monkeypatch, tmp_path):
    monkeypatch.setattr(display_runtime, "heavy_window", lambda _bytes: 1)
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    keys = _catalog(state, owner, (1, 2))
    owner_value, gate = _acquisition_identity(state)
    _hydrate_ok(state, owner_value, gate, processed, 1, 1)
    _hydrate_ok(state, owner_value, gate, processed, 2, 2)
    semilight = owner.publications.get(1)
    assert semilight.view.thumbnail is not None and semilight.record.is_empty
    counts = _instrument_reads(monkeypatch, processed)
    before = dict(state.transport.counters())
    assert state.project(keys[1], 3, closed=False, owner=owner_value,
                         commit_gate=gate) is None
    assert _wait_transport_idle(state)
    restored = owner.publications.get(1)
    assert restored.view.has_1d and restored.view.has_2d and restored.view.raw is None
    assert counts["processed"] == 1 and counts["detector"] == 0
    assert state.project(keys[1], 4, closed=False) is not None
    assert state.transport.counters()[HydrationOutcome.HYDRATED] == before[HydrationOutcome.HYDRATED] + 1


def test_non_square_thumbnail_uses_true_extent_without_remask_or_cake_drift(tmp_path):
    from types import SimpleNamespace; from unittest.mock import Mock
    from xdart.gui.tabs.scattering.scientific_axes import heavy_projection
    from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
    from xdart.gui.tabs.scattering.shell_widgets import ScientificImagePane
    thumbnail = np.array([[np.nan, 1., 2.], [3., 4., 5.]], dtype=np.float32)
    cake = np.arange(6., dtype=float).reshape(2, 3)
    state, owner = _state(tmp_path / "render.nexus")
    key = state.append_navigation(owner.source_scan, str(owner.artifact), 1).appended
    view = FrameView(label=1, thumbnail=thumbnail, axis_2d_x=Axis("q", "1/angstrom", values=np.array([.1, .2, .3])), axis_2d_y=Axis("chi", "degree", values=np.array([-1., 1.])), intensity_2d=cake, extra={"detector_shape": (8, 12)})
    heavy = heavy_projection(StandardDisplayPayload(0, key, "frame", view))
    assert heavy.detector_shape == (8, 12) and heavy.raw is thumbnail
    pane = SimpleNamespace(canvas=Mock(), plot=Mock()); pane.canvas.imageViewBox = Mock()
    ScientificImagePane.render(pane, heavy.raw, detector_shape=heavy.detector_shape)
    image, options = pane.canvas.setImage.call_args.args[0], pane.canvas.setImage.call_args.kwargs
    np.testing.assert_allclose(image, thumbnail.T[:, ::-1], equal_nan=True); rect = options["rect"]
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (0., 0., 11., 7.)
    pane.canvas.setImage.reset_mock(); ScientificImagePane.render(pane, heavy.cake, x_axis=heavy.cake_x, y_axis=heavy.cake_y)
    np.testing.assert_array_equal(pane.canvas.setImage.call_args.args[0], cake.T)
    assert pane.canvas.setImage.call_args.kwargs["linear_percentiles"] == (0.5, 99.5)


@pytest.mark.parametrize(
    ("mode", "persisted_shape", "expected_source", "expected_shape"),
    (("thumbnail", (8, 12), "thumbnail", (8, 12)),
     ("thumbnail", (0, 12), "thumbnail", None),
     ("full", (8, 12), "full", None)),
)
def test_detector_projection_is_thumbnail_first_identity_preserving_and_extent_exact(
    tmp_path, mode, persisted_shape, expected_source, expected_shape,
):
    from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
    from xdart.gui.tabs.scattering.scientific_axes import heavy_projection
    thumbnail = np.array([[np.nan, 1., 2.], [3., 4., 5.]], dtype=np.float32)
    raw = np.arange(20, dtype=np.float32).reshape(4, 5)
    cake = np.arange(6, dtype=np.float32).reshape(2, 3)
    state, owner = _state(tmp_path / "b2-projection.nexus")
    key = state.append_navigation(owner.source_scan, str(owner.artifact), 1).appended
    view = FrameView(
        label=1, raw=raw, thumbnail=thumbnail,
        axis_2d_x=Axis("q", "1/angstrom", values=np.arange(3.)),
        axis_2d_y=Axis("chi", "degree", values=np.arange(2.)),
        intensity_2d=cake, extra={"detector_shape": persisted_shape},
    )
    heavy = heavy_projection(
        StandardDisplayPayload(0, key, "frame", view), detector_mode=mode,
    )
    assert heavy.raw is (view.thumbnail if mode == "thumbnail" else view.raw)
    assert heavy.cake is view.intensity_2d and heavy.detector_source == expected_source
    assert heavy.detector_shape == expected_shape
    assert heavy.cake_x.values is view.axis_2d_x.values
    assert heavy.cake_y.values is view.axis_2d_y.values


def test_b2_full_demand_is_zero_by_default_and_one_per_exact_current() -> None:
    from types import SimpleNamespace
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    first, second, requests = object(), object(), []
    navigation = SimpleNamespace(current=first)
    selection = SimpleNamespace(kind=ContextKind.ACQUISITION, owner=object())
    controller = SimpleNamespace(
        navigation=navigation, selection=selection,
        full_raw_availability=lambda: (True, ""),
        full_raw_status=lambda: (False, bool(requests), None),
        request_full_current=lambda: requests.append(navigation.current) or object(),
        clear_full_raw=lambda: True,
    )
    page = SimpleNamespace(
        _preferences=ScientificPreferences(), _context_controller=controller,
        _detector_scope_owner=selection.owner, _detector_demand_frame=None,
        _ensure_timer=lambda: None, _notice=lambda _text: None,
    )
    ScatteringWorkspace._sync_detector_demand(page)
    assert requests == []
    assert ScatteringWorkspace._set_detector_mode(page, "full")
    ScatteringWorkspace._sync_detector_demand(page)
    ScatteringWorkspace._sync_detector_demand(page)
    navigation.current = second
    ScatteringWorkspace._sync_detector_demand(page)
    assert requests == [first, second]


def test_b2_failure_thumbnail_reset_and_context_replacement_are_fail_closed(
    tmp_path,
) -> None:
    from types import SimpleNamespace
    from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xdart.gui.tabs.scattering.scientific_axes import heavy_projection
    thumbnail = np.arange(6, dtype=np.float32).reshape(2, 3)
    state, owner = _state(tmp_path / "b2-failure.nexus")
    key = state.append_navigation(owner.source_scan, str(owner.artifact), 1).appended
    heavy = heavy_projection(
        StandardDisplayPayload(0, key, "frame", FrameView(
            label=1, thumbnail=thumbnail, extra={"detector_shape": (8, 12)},
        )), detector_mode="full",
    )
    assert heavy.raw is thumbnail and heavy.detector_source == "thumbnail"

    clears, requests = [], []
    old_owner, new_owner = object(), object()
    selection = SimpleNamespace(kind=ContextKind.ACQUISITION, owner=new_owner)
    controller = SimpleNamespace(
        navigation=SimpleNamespace(current=key), selection=selection,
        full_raw_availability=lambda: (True, ""),
        full_raw_status=lambda: (False, False, "detector read failed"),
        request_full_current=lambda: requests.append(key) or object(),
        clear_full_raw=lambda: clears.append(None) or True,
    )
    page = SimpleNamespace(
        _preferences=ScientificPreferences(detector_mode="full"),
        _context_controller=controller, _detector_scope_owner=new_owner,
        _detector_demand_frame=key, _ensure_timer=lambda: None,
        _notice=lambda _text: None,
    )
    ScatteringWorkspace._sync_detector_demand(page)
    assert page._preferences.detector_mode == "full"
    assert page._preferences.detector_diagnostic == "detector read failed"
    selection.owner = old_owner
    ScatteringWorkspace._sync_detector_demand(page)
    assert page._preferences.detector_mode == "thumbnail"
    assert clears == [None] and requests == []
    controller.full_raw_availability = lambda: (
        False, "Full Raw is available only for an exact acquisition frame."
    )
    assert not ScatteringWorkspace._set_detector_mode(page, "full")
    assert page._preferences.detector_mode == "thumbnail"


def test_frame_mask_qualified_no_thumbnail_fails_closed_with_usable_cake(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(
        monkeypatch, tmp_path, thumbnails=False
    )
    keys = _catalog(state, owner, (1, 2, 3))
    with state._lock:
        state._frame_mask_qualified.add(keys[2])
    owner_value, gate = _acquisition_identity(state)
    counts = _instrument_reads(monkeypatch, processed)

    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 4
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)

    assert counts["detector"] == 0  # unmasked raw never appears
    assert state.detector_outcome(keys[2]) is (
        DetectorHydrationOutcome.FRAME_MASK_UNAVAILABLE
    )
    publication = owner.publications.get(2)
    assert publication is not None
    assert publication.view.raw is None
    assert publication.view.intensity_1d is not None  # cake/1-D remain usable
    assert publication.view.intensity_2d is not None


# --------------------------------------------------------------------------- #
# Row 4/5 — FULL distinct; coalescing order
# --------------------------------------------------------------------------- #

def test_full_after_thumbnail_preview_performs_distinct_detector_read(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    counts = _instrument_reads(monkeypatch, processed)

    preview = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 6
    )
    assert state.transport.submit(preview) is not None
    assert _wait_transport_idle(state)
    assert counts["detector"] == 0

    assert state.request_full(
        keys[2], 7, owner=owner_value, commit_gate=gate) is not None
    assert _wait_transport_idle(state)

    assert counts["detector"] == 1  # thumbnail-backed PREVIEW never satisfies FULL
    publication = owner.publications.get(2)
    assert publication.view.raw is not None
    outcomes = _completion_outcomes(state)
    assert outcomes.count((2, HydrationOutcome.HYDRATED)) == 2


def test_same_read_coalesces_and_every_admitted_token_terminates(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    preview_module = import_module("xrd_tools.io.frame_preview")
    entered = threading.Event()
    hold = threading.Event()
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        entered.set()
        hold.wait(timeout=10.0)
        return original(read_key, **kwargs)

    transport_module = _transport_api()
    monkeypatch.setattr(
        transport_module, "read_frame_preview", holding_read
    )
    counts = _instrument_reads(monkeypatch, processed)

    first = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 10
    )
    second = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 11
    )
    third = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 12
    )
    token_a = state.transport.submit(first)
    assert entered.wait(timeout=10.0)  # the read is ACTIVE before the repeats
    token_b = state.transport.submit(second)  # same exact read: reuse, move token
    token_c = state.transport.submit(third)
    assert token_a is not None and token_b is not None and token_c is not None
    hold.set()
    assert _wait_transport_idle(state)

    assert counts["processed"] == 1  # one aggregate read
    outcomes = _completion_outcomes(state)
    assert outcomes.count((2, HydrationOutcome.HYDRATED)) == 1
    assert outcomes.count((2, HydrationOutcome.SUPERSEDED)) == 2
    counters = state.transport.counters()
    assert counters[HydrationOutcome.HYDRATED] == 1
    assert counters[HydrationOutcome.SUPERSEDED] == 2


def test_full_never_coalesces_with_an_active_preview(
    monkeypatch, tmp_path
):
    """Mutation-detectability row (§7.6): a FULL submitted while the same
    frame's PREVIEW read is active is a DISTINCT latest request."""
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    preview_module = import_module("xrd_tools.io.frame_preview")
    hold = threading.Event()
    release = threading.Event()
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        hold.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)
    preview = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 6
    )
    assert state.transport.submit(preview) is not None
    assert hold.wait(timeout=10.0)
    token = state.request_full(
        keys[2], 7, owner=owner_value, commit_gate=gate)
    assert token is not None
    queued = state.transport.queued_token
    assert queued is not None
    assert queued.read_key.purpose is HydrationPurpose.FULL
    release.set()
    assert _wait_transport_idle(state)
    publication = owner.publications.get(2)
    assert publication is not None and publication.view.raw is not None
    outcomes = _completion_outcomes(state)
    assert outcomes.count((2, HydrationOutcome.HYDRATED)) == 2


def test_gate_cancel_waits_for_the_bounded_insert(monkeypatch, tmp_path):
    """Mutation-detectability row (§7.8): the exact carried CommitGate is held
    around the insert, so cancel() returning proves no further insert."""
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    inside = threading.Event()
    release = threading.Event()
    original_upsert = owner.publications.upsert

    def holding_upsert(publication):
        inside.set()
        assert release.wait(timeout=10.0)
        return original_upsert(publication)

    monkeypatch.setattr(owner.publications, "upsert", holding_upsert)
    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request) is not None
    assert inside.wait(timeout=10.0)

    cancel_done = threading.Event()
    canceller = threading.Thread(
        target=lambda: (gate.cancel(), cancel_done.set()), daemon=True
    )
    canceller.start()
    time.sleep(0.15)
    assert not cancel_done.is_set()  # cancel waits for the bounded insert
    release.set()
    canceller.join(timeout=10.0)
    assert cancel_done.is_set()
    assert _wait_transport_idle(state)


# --------------------------------------------------------------------------- #
# Row 6 — rapid latest wins, bounded queue, no stranded token
# --------------------------------------------------------------------------- #

def test_rapid_navigation_latest_selection_wins_with_bounded_queue(
    monkeypatch, tmp_path
):
    labels = tuple(range(1, 21))  # A…T
    state, owner, processed, _raw, _events = _bound_state(
        monkeypatch, tmp_path, labels=labels
    )
    _catalog(state, owner, labels)
    owner_value, gate = _acquisition_identity(state)
    preview_module = import_module("xrd_tools.io.frame_preview")
    hold = threading.Event()
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        if int(read_key.frame_identity) == 1:
            hold.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)
    counts = _instrument_reads(monkeypatch, processed)

    tokens = []
    for index, label in enumerate(labels):
        request = _typed_request(
            state,
            owner_value,
            gate,
            processed,
            label,
            HydrationPurpose.PREVIEW,
            100 + index,
        )
        tokens.append(state.transport.submit(request))
    assert all(token is not None for token in tokens)
    hold.set()
    assert _wait_transport_idle(state)

    # Final exact selection wins; intermediate queued tokens superseded.
    assert owner.publications.get(20) is not None
    outcomes = _completion_outcomes(state)
    assert (20, HydrationOutcome.HYDRATED) in outcomes
    superseded = [item for item in outcomes if item[1] is HydrationOutcome.SUPERSEDED]
    assert len(superseded) == 18  # every displaced intermediate, none stranded
    assert len(outcomes) == len(labels)
    assert counts["processed"] <= 2  # held first read + final read only


# --------------------------------------------------------------------------- #
# Row 7 — atomic stale rejection before any public mutation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "mutate",
    [
        "owner_epoch",
        "gate_cancelled",
        "catalog_retired",
        "state_retired",
    ],
)
def test_stale_mutations_reject_before_upsert_or_repaint(
    monkeypatch, tmp_path, mutate
):
    processed, _raw = _write_processed(tmp_path, labels=(1, 2, 3, 4, 5))
    state, owner = _state(processed, catalog_max_items=3)
    events = []
    state.bind_transport(event_sink=events.append)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    preview_module = import_module("xrd_tools.io.frame_preview")
    hold = threading.Event()
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        hold.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)

    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request) is not None

    if mutate == "owner_epoch":
        gate.advance()
    elif mutate == "gate_cancelled":
        gate.cancel()
    elif mutate == "catalog_retired":
        # Real retirement route: appending past the explicit catalog capacity
        # retires the oldest keys, including the requested frame 2.
        state.append_navigation(owner.source_scan, str(owner.artifact), 4)
        state.append_navigation(owner.source_scan, str(owner.artifact), 5)
        assert state.resolve_frame(keys[2]) is not keys[2]
    elif mutate == "state_retired":
        state.retire(join_timeout=0.0)
    hold.set()
    assert _wait_transport_idle(state)

    assert owner.publications.get(2) is None
    assert keys[2] not in state.payloads
    assert not any(
        getattr(event, "frame_key", None) is keys[2] for event in events
    )
    outcomes = dict(_completion_outcomes(state))
    assert outcomes.get(2) in {
        HydrationOutcome.CANCELLED,
        HydrationOutcome.OWNER_MISMATCH,
        HydrationOutcome.SUPERSEDED,
    }


# --------------------------------------------------------------------------- #
# E4-R4a — §20.2 corrections: true latest queue and atomic publication
# --------------------------------------------------------------------------- #

def test_newest_active_selection_supersedes_older_queued_read(
    monkeypatch, tmp_path
):
    """P1-E4R-2 (§20.2): active A -> queued B -> newest A under ONE display
    generation leaves no live B: B terminalizes SUPERSEDED exactly once, the
    queued slot empties, only A's already-active read opens, B never
    publishes or emits DISPLAY_READY, and the final payload/key is exact A."""
    state, owner, processed, _raw, events = _bound_state(monkeypatch, tmp_path)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    preview_module = import_module("xrd_tools.io.frame_preview")
    entered = threading.Event()
    release = threading.Event()
    original = preview_module.read_frame_preview

    def hold_first(read_key, **kwargs):
        if int(read_key.frame_identity) == 1:
            entered.set()
            assert release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", hold_first)
    counts = _instrument_reads(monkeypatch, processed)

    generation = 5  # one display generation across all three submissions
    active_a = _typed_request(
        state, owner_value, gate, processed, 1,
        HydrationPurpose.PREVIEW, generation,
    )
    older_b = _typed_request(
        state, owner_value, gate, processed, 2,
        HydrationPurpose.PREVIEW, generation,
    )
    newest_a = _typed_request(
        state, owner_value, gate, processed, 1,
        HydrationPurpose.PREVIEW, generation,
    )
    assert state.transport.submit(active_a) is not None
    assert entered.wait(timeout=10.0)
    assert state.transport.submit(older_b) is not None
    assert state.transport.submit(newest_a) is not None

    # The newest active selection displaces the older queued read NOW.
    assert state.transport.queued_token is None
    release.set()
    assert _wait_transport_idle(state)

    outcomes = _completion_outcomes(state)
    assert outcomes.count((2, HydrationOutcome.SUPERSEDED)) == 1
    assert outcomes.count((1, HydrationOutcome.HYDRATED)) == 1
    assert len(outcomes) == 2  # one terminal completion per admitted token
    assert counts["processed"] == 1  # only A's already-active read opened
    assert owner.publications.get(2) is None  # B never published
    assert all(
        getattr(event, "frame_key", None) is not keys[2] for event in events
    )
    payload = state.payloads.get(keys[1])
    assert payload is not None and payload.frame_key is keys[1]
    assert keys[2] not in state.payloads


def _install_seam_failure(monkeypatch, state, owner, seam, *, permanent):
    """Inject one exact commit-seam failure (fail-once or persistent)."""
    target, name = {
        "light": (owner.light_records, "exchange_releasable_record"),
        "residency": (state._residency, "observe"),
        "publication": (owner.publications, "upsert"),
    }[seam]
    original = getattr(target, name)
    calls = {"n": 0}

    def failing(*args, **kwargs):
        calls["n"] += 1
        if permanent or calls["n"] == 1:
            raise RuntimeError(f"injected {seam} failure #{calls['n']}")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, failing)
    return calls


def _hydrate_ok(state, owner_value, gate, processed, label, generation):
    request = _typed_request(
        state, owner_value, gate, processed, label,
        HydrationPurpose.PREVIEW, generation,
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)
    return request


@pytest.mark.parametrize("seam", ["light", "residency", "publication"])
def test_persistent_seam_failure_rejects_candidate_and_preserves_display(
    monkeypatch, tmp_path, seam
):
    """P1-E4R-1 (§20.2): a persistent failure at ANY commit seam terminalizes
    FAILED exactly once, leaves the rejected frame publicly unprojectable
    (no publication, payload or event), and preserves the prior coherent
    public display."""
    state, owner, processed, _raw, events = _bound_state(monkeypatch, tmp_path)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    _hydrate_ok(state, owner_value, gate, processed, 2, 4)
    prior_publication = owner.publications.get(2)
    assert prior_publication is not None
    prior_payload = state.payloads.get(keys[2])
    assert prior_payload is not None
    events.clear()

    _install_seam_failure(monkeypatch, state, owner, seam, permanent=True)
    request = _typed_request(
        state, owner_value, gate, processed, 3, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)

    outcomes = _completion_outcomes(state)
    assert outcomes.count((3, HydrationOutcome.FAILED)) == 1
    assert owner.publications.get(3) is None
    assert keys[3] not in state.payloads
    assert state.project(keys[3], 6, closed=False) is None  # no issuer identity
    assert not events  # no payload/event for the rejected candidate
    # The prior coherent public display is untouched.
    assert owner.publications.get(2) is prior_publication
    assert state.payloads.get(keys[2]) is prior_payload


@pytest.mark.parametrize("seam", ["light", "residency", "publication"])
def test_fail_once_seam_retries_exact_prepared_commit_once(
    monkeypatch, tmp_path, seam
):
    """§20.4(2): fail-once at each seam retries the exact retained prepared
    commit once and succeeds once, with exactly one public signal."""
    state, owner, processed, _raw, events = _bound_state(monkeypatch, tmp_path)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    calls = _install_seam_failure(
        monkeypatch, state, owner, seam, permanent=False
    )
    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)

    outcomes = _completion_outcomes(state)
    assert outcomes.count((2, HydrationOutcome.HYDRATED)) == 1
    assert calls["n"] >= 2  # the exact prepared commit was retried once
    publication = owner.publications.get(2)
    assert publication is not None and publication.view.thumbnail is not None
    ready = [
        event for event in events
        if getattr(event, "frame_key", None) is keys[2]
    ]
    assert len(ready) == 1  # exactly one public signal


def test_failed_rehydration_preserves_the_exact_existing_publication(
    monkeypatch, tmp_path
):
    """A FAILED re-hydration of an already-public frame must leave the exact
    prior publication object public (frozen §3.5 rejection semantics)."""
    state, owner, processed, _raw, events = _bound_state(monkeypatch, tmp_path)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    _hydrate_ok(state, owner_value, gate, processed, 2, 4)
    prior_publication = owner.publications.get(2)
    assert prior_publication is not None
    events.clear()

    _install_seam_failure(
        monkeypatch, state, owner, "residency", permanent=True
    )
    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 6
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)

    outcomes = _completion_outcomes(state)
    assert outcomes.count((2, HydrationOutcome.FAILED)) == 1
    assert owner.publications.get(2) is prior_publication
    assert not events


def test_owed_multimode_light_row_refuses_hydration_before_mutation(
    tmp_path,
):
    """P1-C L1: hydration cannot merge onto an unreleasable GUI-light row."""
    state, owner, processed, _raw, events = _bound_state(None, tmp_path)
    keys = _catalog(state, owner, (1,))
    owner_value, gate = _acquisition_identity(state)
    view = FrameView(
        label=1,
        axis_1d=Axis("q", "1/angstrom", np.array([0.1, 0.2, 0.3])),
        intensity_1d=np.array([4.0, 5.0, 6.0]),
        source_path="/owed/source.tif",
        source_frame_index=1,
    )
    owed = FrameRecord.from_view(view, mode_1d="owed-a").with_result_1d(
        "owed-b", view, make_active=False,
    )
    source_identity = "/owed/source.tif#1"
    owner.light_records.upsert(
        owed,
        source_identity=source_identity,
        persisted=False,
    )
    before = (
        owner.light_records.get(1),
        owner.light_records.source_identity(1),
        owner.light_records.is_persisted(1),
        owner.publications.get(1),
        state.payloads.get(keys[1]),
        state.residency_snapshot(),
    )

    request = _typed_request(
        state,
        owner_value,
        gate,
        processed,
        1,
        HydrationPurpose.PREVIEW,
        1,
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)

    assert _completion_outcomes(state).count(
        (1, HydrationOutcome.FAILED)
    ) == 1
    assert (
        owner.light_records.get(1),
        owner.light_records.source_identity(1),
        owner.light_records.is_persisted(1),
        owner.publications.get(1),
        state.payloads.get(keys[1]),
        state.residency_snapshot(),
    ) == before
    assert not events


# --------------------------------------------------------------------------- #
# E4-R5a — §22 complete failed-commit isolation (five ported review rows
# plus the §22.3 exact-state freezes)
# --------------------------------------------------------------------------- #

def _tier_orders(state):
    residency = state._residency
    return {
        "heavy": tuple(residency._heavy),
        "thumbnails": tuple(residency._thumbnails),
        "browse": tuple(residency._browse),
        "live": tuple(residency._live),
    }


def test_rejected_candidate_never_occupies_a_heavy_residency_slot(
    monkeypatch, tmp_path
):
    """§22.2 row 1: a never-published candidate must not later displace
    public heavy data."""
    monkeypatch.setattr(display_runtime, "heavy_window", lambda _bytes: 2)
    monkeypatch.setattr(display_runtime, "THUMBNAIL_MAX_ITEMS", 8)
    processed, _raw = _write_processed(tmp_path, labels=(1, 2, 3, 4))
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    _catalog(state, owner, (1, 2, 3, 4))
    owner_value, gate = _acquisition_identity(state)

    _hydrate_ok(state, owner_value, gate, processed, 1, 1)
    _hydrate_ok(state, owner_value, gate, processed, 2, 2)
    assert state.residency_snapshot().heavy == 2
    assert owner.publications.has_heavy_payload(1)
    assert owner.publications.has_heavy_payload(2)

    _install_seam_failure(monkeypatch, state, owner, "publication",
                          permanent=True)
    rejected = _typed_request(
        state, owner_value, gate, processed, 3, HydrationPurpose.PREVIEW, 3
    )
    assert state.transport.submit(rejected) is not None
    assert _wait_transport_idle(state)
    assert _completion_outcomes(state).count(
        (3, HydrationOutcome.FAILED)
    ) == 1
    assert owner.publications.get(3) is None
    # The failed candidate cannot retain a phantom heavy slot.
    assert state.residency_snapshot().heavy == 2

    # A later success may demote only the true oldest public heavy frame.
    monkeypatch.undo()
    monkeypatch.setattr(display_runtime, "heavy_window", lambda _bytes: 2)
    _hydrate_ok(state, owner_value, gate, processed, 4, 4)
    assert state.residency_snapshot().heavy == 2
    # Ported-row correction (recorded): publications.has_heavy_payload stays
    # True for ANY thumbnail-backed frame because semilight eviction KEEPS the
    # thumbnail (accepted D2 policy) and the heavy predicate counts it.  The
    # reviewer's intent — only the TRUE oldest demotes — is pinned exactly via
    # the records-side predicate and the semilight publication status.
    assert not owner.records.has_heavy_payload(1)  # the true oldest demoted
    assert owner.publications.get(1).raw_status == "thumbnail"  # semilight
    assert owner.records.has_heavy_payload(2)  # untouched public frame
    assert owner.publications.get(3) is None
    assert owner.records.has_heavy_payload(4)


def test_failed_rehydration_does_not_change_future_heavy_eviction_order(
    monkeypatch, tmp_path
):
    """§22.2 row 2: a failed refresh of frame 1 cannot make frame 2 look
    older than frame 1."""
    monkeypatch.setattr(display_runtime, "heavy_window", lambda _bytes: 2)
    monkeypatch.setattr(display_runtime, "THUMBNAIL_MAX_ITEMS", 8)
    processed, _raw = _write_processed(tmp_path, labels=(1, 2, 3))
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    _hydrate_ok(state, owner_value, gate, processed, 1, 1)
    _hydrate_ok(state, owner_value, gate, processed, 2, 2)

    _install_seam_failure(monkeypatch, state, owner, "publication",
                          permanent=True)
    rejected = _typed_request(
        state, owner_value, gate, processed, 1, HydrationPurpose.PREVIEW, 3
    )
    assert state.transport.submit(rejected) is not None
    assert _wait_transport_idle(state)
    assert _completion_outcomes(state).count(
        (1, HydrationOutcome.FAILED)
    ) == 1

    monkeypatch.undo()
    monkeypatch.setattr(display_runtime, "heavy_window", lambda _bytes: 2)
    _hydrate_ok(state, owner_value, gate, processed, 3, 4)
    # Same recorded correction as the heavy-slot row: pin the demotion on the
    # records side (semilight keeps the thumbnail by accepted D2 policy).
    assert not owner.records.has_heavy_payload(1)  # true oldest, not frame 2
    assert owner.publications.get(1).raw_status == "thumbnail"
    assert owner.records.has_heavy_payload(2)
    assert owner.records.has_heavy_payload(3)


def test_failed_rehydration_does_not_leak_light_record_after_cache_eviction(
    monkeypatch, tmp_path
):
    """§22.2 row 3: a rejected candidate cannot later surface through the
    bounded-cache light-record reconstruction path."""
    processed, _raw = _write_processed(tmp_path, labels=(1, 2))
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    keys = _catalog(state, owner, (1, 2))
    owner_value, gate = _acquisition_identity(state)
    _hydrate_ok(state, owner_value, gate, processed, 1, 1)
    prior = state.payloads[keys[1]].view.intensity_1d.copy()

    transport_api = _transport_api()
    original_read = transport_api.read_frame_preview
    rejected_values = np.array([101.0, 102.0, 103.0])
    rejected_values.setflags(write=False)

    def changed_preview(*args, **kwargs):
        preview = original_read(*args, **kwargs)
        from dataclasses import replace as _replace
        changed_view = _replace(preview.view, intensity_1d=rejected_values)
        return _replace(preview, view=changed_view)

    monkeypatch.setattr(transport_api, "read_frame_preview", changed_preview)
    _install_seam_failure(monkeypatch, state, owner, "publication",
                          permanent=True)
    rejected = _typed_request(
        state, owner_value, gate, processed, 1, HydrationPurpose.PREVIEW, 2
    )
    assert state.transport.submit(rejected) is not None
    assert _wait_transport_idle(state)
    assert _completion_outcomes(state).count(
        (1, HydrationOutcome.FAILED)
    ) == 1

    # Cache-HIT projection equality after rejection.
    cached = state.project(keys[1], 3, closed=True)
    assert cached is not None
    np.testing.assert_array_equal(cached.view.intensity_1d, prior)

    # Force the normal bounded-cache reconstruction path (cache MISS).
    state.payloads.pop(keys[1])
    rebuilt = state.project(keys[1], 4, closed=True)
    assert rebuilt is not None
    np.testing.assert_array_equal(rebuilt.view.intensity_1d, prior)


def test_rejected_candidate_never_occupies_browse_or_live_slots(
    monkeypatch, tmp_path
):
    """§22.2 row 4: injected small caps expose the same phantom in the
    cheap tiers."""
    from xdart.gui.tabs.scattering.display_residency import (
        DisplayResidencyLimits,
    )

    processed, _raw = _write_processed(tmp_path, labels=(1, 2, 3, 4))
    state, owner = _state(processed)
    state._residency.limits = DisplayResidencyLimits(8, 8, 2, 2)
    state.bind_transport(event_sink=lambda event: None)
    _catalog(state, owner, (1, 2, 3, 4))
    owner_value, gate = _acquisition_identity(state)
    _hydrate_ok(state, owner_value, gate, processed, 1, 1)
    _hydrate_ok(state, owner_value, gate, processed, 2, 2)

    _install_seam_failure(monkeypatch, state, owner, "publication",
                          permanent=True)
    rejected = _typed_request(
        state, owner_value, gate, processed, 3, HydrationPurpose.PREVIEW, 3
    )
    assert state.transport.submit(rejected) is not None
    assert _wait_transport_idle(state)
    snapshot = state.residency_snapshot()
    assert snapshot.browse == 2
    assert snapshot.live == 2

    monkeypatch.undo()
    _hydrate_ok(state, owner_value, gate, processed, 4, 4)
    assert owner.publications.get(1) is None
    assert owner.publications.get(2) is not None
    assert owner.publications.get(3) is None
    assert owner.publications.get(4) is not None


def test_failed_commit_does_not_publish_detector_outcome(
    monkeypatch, tmp_path
):
    """§22.2 row 5: a FAILED hydration cannot leave a terminal scientific
    result behind."""
    processed, raw_path = _write_processed(
        tmp_path, labels=(1,), thumbnails=False
    )
    raw_path.unlink()
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    keys = _catalog(state, owner, (1,))
    owner_value, gate = _acquisition_identity(state)
    _install_seam_failure(monkeypatch, state, owner, "light",
                          permanent=True)
    rejected = _typed_request(
        state, owner_value, gate, processed, 1, HydrationPurpose.PREVIEW, 1
    )
    assert state.transport.submit(rejected, closed=True) is not None
    assert _wait_transport_idle(state)
    assert _completion_outcomes(state).count(
        (1, HydrationOutcome.FAILED)
    ) == 1
    assert state.detector_outcome(keys[1]) is None


@pytest.mark.parametrize("seam", ["light", "residency", "publication"])
@pytest.mark.parametrize("shape", ["fresh", "rehydrated"])
def test_failed_attempt_restores_every_public_surface_exactly(
    monkeypatch, tmp_path, seam, shape
):
    """§22.3 freeze: after every thrown attempt, every public-consumed
    surface — records, light records, detector outcome, all four residency
    tiers WITH FIFO order, publication, payload — equals its exact
    pre-attempt state, for both a new key and an already-public key."""
    labels = (1, 2, 3)
    processed, _raw = _write_processed(tmp_path, labels=labels)
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    keys = _catalog(state, owner, labels)
    owner_value, gate = _acquisition_identity(state)
    _hydrate_ok(state, owner_value, gate, processed, 1, 1)
    _hydrate_ok(state, owner_value, gate, processed, 2, 2)
    target = 1 if shape == "rehydrated" else 3

    before = dict(
        record=owner.records.get(target),
        light=owner.light_records.get(target),
        outcome=state.detector_outcome(keys[target]),
        publication=owner.publications.get(target),
        payload=state.payloads.get(keys[target]),
        orders=_tier_orders(state),
        snapshot=state.residency_snapshot(),
    )
    _install_seam_failure(monkeypatch, state, owner, seam, permanent=True)
    rejected = _typed_request(
        state, owner_value, gate, processed, target,
        HydrationPurpose.PREVIEW, 9,
    )
    assert state.transport.submit(rejected) is not None
    assert _wait_transport_idle(state)
    assert _completion_outcomes(state).count(
        (target, HydrationOutcome.FAILED)
    ) == 1

    assert owner.records.get(target) is before["record"]
    assert owner.light_records.get(target) is before["light"]
    assert state.detector_outcome(keys[target]) is before["outcome"]
    assert owner.publications.get(target) is before["publication"]
    assert state.payloads.get(keys[target]) is before["payload"]
    assert _tier_orders(state) == before["orders"]
    assert state.residency_snapshot() == before["snapshot"]


def test_failed_reattempt_preserves_a_prior_terminal_detector_outcome(
    monkeypatch, tmp_path
):
    """§22.3 freeze: a key holding a terminal scientific outcome keeps that
    EXACT outcome through a later rejected direct re-attempt."""
    processed, raw_path = _write_processed(
        tmp_path, labels=(1, 2), thumbnails=False
    )
    raw_path.unlink()
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    keys = _catalog(state, owner, (1, 2))
    owner_value, gate = _acquisition_identity(state)
    first = _typed_request(
        state, owner_value, gate, processed, 1, HydrationPurpose.PREVIEW, 1
    )
    assert state.transport.submit(first, closed=True) is not None
    assert _wait_transport_idle(state)
    prior_outcome = state.detector_outcome(keys[1])
    assert prior_outcome is DetectorHydrationOutcome.DETECTOR_UNAVAILABLE

    _install_seam_failure(monkeypatch, state, owner, "publication",
                          permanent=True)
    again = _typed_request(
        state, owner_value, gate, processed, 1, HydrationPurpose.PREVIEW, 2
    )
    assert state.transport.submit(again, closed=True) is not None
    assert _wait_transport_idle(state)
    assert state.detector_outcome(keys[1]) is prior_outcome


def test_failed_hydration_restore_preserves_concurrent_live_residency(
    monkeypatch, tmp_path
):
    """Reviewer correction: exact rollback cannot erase a live frame whose
    producer already passed the display lock before hydration captured the
    global tier order."""
    from xdart.modules.frame_publication import FramePublication
    from xrd_tools.core import FrameRecord
    from xrd_tools.io.frame_preview import (
        DetectorPreviewProjection,
        read_frame_preview,
    )

    processed, _raw = _write_processed(tmp_path, labels=(1, 2))
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    keys = _catalog(state, owner, (1, 2))
    owner_value, gate = _acquisition_identity(state)
    _hydrate_ok(state, owner_value, gate, processed, 1, 1)

    scope = HydrationScope(*owner_value.as_tuple())
    read_key = HydrationReadKey(
        scope, str(processed), 2, HydrationPurpose.PREVIEW
    )
    preview = read_frame_preview(
        read_key,
        detector_projection=DetectorPreviewProjection.without_static_mask(
            mask_saturation=False
        ),
    )
    record = FrameRecord.from_view(preview.view)
    source_identity = (
        f"{preview.view.source_path or ''}#{preview.view.source_frame_index}"
    )
    publication = FramePublication(
        preview.view,
        record=record,
        source_identity=source_identity,
        scan_key=owner.source_scan,
    )

    live_passed_display_lock = threading.Event()
    release_live_store_write = threading.Event()
    hydration_reached_publication = threading.Event()
    live_complete = threading.Event()
    live_errors = []
    canonical_record = owner.records.get(2)
    original_light_upsert = owner.light_records.upsert
    original_publication_upsert = owner.publications.upsert

    def coordinated_light_upsert(candidate, *args, **kwargs):
        if (
            candidate.label == 2
            and threading.current_thread().name == "live-producer"
        ):
            # retain_frame has already left RunDisplayState._lock here on the
            # rejected parent.
            live_passed_display_lock.set()
            assert release_live_store_write.wait(timeout=10)
        return original_light_upsert(candidate, *args, **kwargs)

    def coordinated_publication_upsert(candidate):
        if (
            candidate.label == 1
            and threading.current_thread().name != "live-producer"
        ):
            # R5 has captured/touched frame 1. Let the already-admitted live
            # producer finish frame 2, then fail hydration so restore follows.
            hydration_reached_publication.set()
            release_live_store_write.set()
            assert live_complete.wait(timeout=10)
            raise RuntimeError("injected hydration publication failure")
        return original_publication_upsert(candidate)

    monkeypatch.setattr(
        owner.light_records, "upsert", coordinated_light_upsert
    )
    monkeypatch.setattr(
        owner.publications, "upsert", coordinated_publication_upsert
    )

    def publish_live_frame():
        try:
            state.retain_frame(
                owner,
                keys[2],
                record,
                publication,
                source_identity=source_identity,
                frame_mask_qualified=False,
            )
        except BaseException as error:
            live_errors.append(error)
        finally:
            live_complete.set()

    live_thread = threading.Thread(
        target=publish_live_frame, name="live-producer"
    )
    live_thread.start()
    assert live_passed_display_lock.wait(timeout=10)
    rejected = _typed_request(
        state,
        owner_value,
        gate,
        processed,
        1,
        HydrationPurpose.PREVIEW,
        2,
    )
    submit_results = []
    submit_errors = []

    def submit_hydration():
        try:
            submit_results.append(state.transport.submit(rejected))
        except BaseException as error:
            submit_errors.append(error)

    submit_thread = threading.Thread(
        target=submit_hydration, name="hydration-submitter"
    )
    submit_thread.start()
    # On the rejected parent hydration overtakes the blocked live producer.
    # With the correction even target derivation cannot pass the display lock
    # until that producer finishes.
    hydration_reached_publication.wait(timeout=0.2)
    release_live_store_write.set()
    submit_thread.join(timeout=10)
    assert not submit_thread.is_alive()
    assert not submit_errors
    assert submit_results and submit_results[0] is not None
    assert hydration_reached_publication.wait(timeout=10)
    assert _wait_transport_idle(state)
    live_thread.join(timeout=10)

    assert not live_thread.is_alive()
    assert not live_errors
    assert _completion_outcomes(state).count(
        (1, HydrationOutcome.FAILED)
    ) == 1
    assert owner.records.get(2) is canonical_record
    assert owner.publications.get(2) is publication
    residency = state._residency
    assert keys[2] in residency._stores
    assert keys[2] in residency._heavy
    assert keys[2] in residency._thumbnails
    assert keys[2] in residency._browse
    assert keys[2] in residency._live


def test_enforce_failure_never_unpublishes_a_sealed_commit(
    monkeypatch, tmp_path
):
    """Adversarial-pass row: cap enforcement runs OUTSIDE the sealed public
    boundary. Once the authoritative publication/payload landed coherently,
    a failing trim pass must not flip the outcome to FAILED, un-publish
    anything, or suppress the one public signal; the next successful commit
    re-enforces the caps exactly."""
    monkeypatch.setattr(display_runtime, "THUMBNAIL_MAX_ITEMS", 2)
    labels = (1, 2, 3)
    processed, _raw = _write_processed(tmp_path, labels=labels)
    state, owner = _state(processed)
    events = []
    state.bind_transport(event_sink=events.append)
    keys = _catalog(state, owner, labels)
    owner_value, gate = _acquisition_identity(state)
    _hydrate_ok(state, owner_value, gate, processed, 1, 3)

    original_enforce = state._residency.enforce
    broken = {"active": True}

    def failing_enforce(*, protected=(), heavy_victim=None):
        if broken["active"]:
            raise RuntimeError("injected persistent enforce failure")
        return original_enforce(
            protected=protected,
            heavy_victim=heavy_victim,
        )

    monkeypatch.setattr(state._residency, "enforce", failing_enforce)
    events.clear()
    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)

    outcomes = _completion_outcomes(state)
    assert outcomes.count((2, HydrationOutcome.HYDRATED)) == 1  # sealed
    assert (2, HydrationOutcome.FAILED) not in outcomes
    publication = owner.publications.get(2)
    assert publication is not None and publication.view.thumbnail is not None
    assert keys[2] in state.payloads
    assert len(events) == 1  # the one public signal still fired

    # Self-healing: the next commit with a working trim pass re-enforces the
    # caps exactly (three thumbnails collapse to the cap, oldest demoted).
    broken["active"] = False
    _hydrate_ok(state, owner_value, gate, processed, 3, 6)
    assert state.residency_snapshot().thumbnails == 2
    assert owner.publications.has_thumbnail(1) is False
    assert owner.publications.get(1) is not None


def test_cap_pressure_rejected_candidate_neither_evicts_nor_goes_stale(
    monkeypatch, tmp_path
):
    """§20.4(3): under thumbnail cap pressure, a rejected candidate must not
    evict prior public state or leave stale residency accounting; the next
    successful commit then evicts exactly the oldest and stays exact."""
    monkeypatch.setattr(display_runtime, "THUMBNAIL_MAX_ITEMS", 2)
    labels = (1, 2, 3)
    processed, _raw = _write_processed(tmp_path, labels=labels)
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    _catalog(state, owner, labels)
    owner_value, gate = _acquisition_identity(state)
    _hydrate_ok(state, owner_value, gate, processed, 1, 3)
    _hydrate_ok(state, owner_value, gate, processed, 2, 4)
    assert state.residency_snapshot().thumbnails == 2

    calls = _install_seam_failure(
        monkeypatch, state, owner, "publication", permanent=True
    )
    request = _typed_request(
        state, owner_value, gate, processed, 3, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)
    assert calls["n"] >= 2

    outcomes = _completion_outcomes(state)
    assert outcomes.count((3, HydrationOutcome.FAILED)) == 1
    # Nothing was evicted or exposed for the rejected candidate...
    assert owner.publications.get(1).view.thumbnail is not None
    assert owner.publications.get(2).view.thumbnail is not None
    assert owner.publications.get(3) is None
    # ...and residency accounting is not stale (no phantom third entry).
    assert state.residency_snapshot().thumbnails == 2

    # The next SUCCESSFUL commit enforces the cap exactly: the oldest
    # thumbnail demotes, the entry stays projectable, accounting stays exact.
    monkeypatch.undo()
    monkeypatch.setattr(display_runtime, "THUMBNAIL_MAX_ITEMS", 2)
    _hydrate_ok(state, owner_value, gate, processed, 3, 6)
    assert owner.publications.get(3).view.thumbnail is not None
    assert state.residency_snapshot().thumbnails == 2
    assert owner.publications.has_thumbnail(1) is False  # oldest demoted
    assert owner.publications.get(1) is not None  # still projectable (light)
    assert owner.publications.get(2).view.thumbnail is not None


# --------------------------------------------------------------------------- #
# Row 12 — terminal absence versus retryable running miss
# --------------------------------------------------------------------------- #

def test_closed_integrated_only_record_reaches_detector_unavailable_once(
    monkeypatch, tmp_path
):
    processed, raw_path = _write_processed(tmp_path, thumbnails=False)
    raw_path.unlink()  # detector source genuinely absent
    state, owner = _state(processed)
    events = []
    state.bind_transport(event_sink=events.append)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    counts = _instrument_reads(monkeypatch, processed)

    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request, closed=True) is not None
    assert _wait_transport_idle(state)

    assert state.detector_outcome(keys[2]) is (
        DetectorHydrationOutcome.DETECTOR_UNAVAILABLE
    )
    publication = owner.publications.get(2)
    assert publication is not None and publication.view.intensity_1d is not None
    first_reads = counts["processed"]

    # Terminal: a repeated projection performs no further read.
    payload = state.project(
        keys[2], 6, closed=True, owner=owner_value, commit_gate=gate
    )
    assert payload is not None
    assert counts["processed"] == first_reads


def test_empty_integrated_arrays_never_terminalize_detector_absence(
    monkeypatch, tmp_path
):
    """A closed record whose integrated arrays are EMPTY has no scientific
    basis for a terminal detector verdict (superseded E2-LV-D claim retained
    against the transport seam)."""
    processed = tmp_path / "xdart_processed_data" / "empty.nexus"
    processed.parent.mkdir(parents=True)
    with h5py.File(processed, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        # A reachable degenerate persisted shape: the frame record exists but
        # carries NO integrated arrays, thumbnail or source provenance.
        record = entry.create_group("frames/frame_0002")
        record.create_dataset("metadata/placeholder", data=1)
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    keys = _catalog(state, owner, (2,))
    owner_value, gate = _acquisition_identity(state)

    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request, closed=True) is not None
    assert _wait_transport_idle(state)
    assert state.detector_outcome(keys[2]) is None  # never terminal


def test_running_detector_miss_remains_retryable_with_truthful_cake(
    monkeypatch, tmp_path
):
    processed, raw_path = _write_processed(tmp_path, thumbnails=False)
    raw_path.unlink()
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)

    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request, closed=False) is not None
    assert _wait_transport_idle(state)

    assert state.detector_outcome(keys[2]) is None  # not terminal while running
    publication = owner.publications.get(2)
    assert publication is not None
    assert publication.view.intensity_1d is not None
    assert publication.view.raw is None and publication.view.thumbnail is None


# --------------------------------------------------------------------------- #
# Row 13 — injected bounds independent of host RAM
# --------------------------------------------------------------------------- #

def test_injected_store_caps_bound_residency_independent_of_host(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(display_runtime, "heavy_window", lambda _bytes: 2)
    monkeypatch.setattr(display_runtime, "THUMBNAIL_MAX_ITEMS", 2)
    labels = tuple(range(1, 9))
    processed, _raw = _write_processed(tmp_path, labels=labels)
    state, owner = _state(processed)
    state.bind_transport(event_sink=lambda event: None)
    _catalog(state, owner, labels)
    owner_value, gate = _acquisition_identity(state)

    for index, label in enumerate(labels):
        request = _typed_request(
            state, owner_value, gate, processed, label,
            HydrationPurpose.PREVIEW, 50 + index,
        )
        assert state.transport.submit(request) is not None
        assert _wait_transport_idle(state)

    snapshot = state.residency_snapshot()
    assert snapshot.heavy <= 2
    assert snapshot.thumbnails <= 2
    assert len(state.payloads) <= state.max_payload_items


# --------------------------------------------------------------------------- #
# Row 16 — retirement/Close: one cancellation, one join, no late commit
# --------------------------------------------------------------------------- #

def test_retire_during_held_read_cancels_joins_and_refuses_late_commit(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    preview_module = import_module("xrd_tools.io.frame_preview")
    hold = threading.Event()
    release = threading.Event()
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        hold.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)

    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request) is not None
    assert hold.wait(timeout=10.0)
    worker = state.transport.worker
    assert worker is not None and worker.is_alive()

    release.set()
    assert state.retire(join_timeout=10.0) is True
    assert state.transport.worker is None or not state.transport.worker.is_alive()
    assert owner.publications.get(2) is None
    outcomes = dict(_completion_outcomes(state))
    assert outcomes.get(2) in {
        HydrationOutcome.CANCELLED,
        HydrationOutcome.OWNER_MISMATCH,
    }


def test_cancel_gate_drops_queued_reference_and_reports_retention(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    browse_gate = CommitGate()
    browse_owner = HydrationOwner(
        "ctx-browse", "scan", str(processed), browse_gate.epoch
    )
    preview_module = import_module("xrd_tools.io.frame_preview")
    hold = threading.Event()
    release = threading.Event()
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        hold.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)

    active = _typed_request(
        state, owner_value, gate, processed, 1, HydrationPurpose.PREVIEW, 5
    )
    queued = _typed_request(
        state,
        browse_owner,
        browse_gate,
        processed,
        3,
        HydrationPurpose.PREVIEW,
        6,
        stores=(owner.publications,),
    )
    assert state.transport.submit(active) is not None
    assert hold.wait(timeout=10.0)
    assert state.transport.submit(queued) is not None
    assert state.transport.retains_gate(browse_gate) is True

    browse_gate.cancel()
    state.transport.cancel_gate(browse_gate)
    assert state.transport.retains_gate(browse_gate) is False
    outcomes = dict(_completion_outcomes(state))
    assert outcomes.get(3) is HydrationOutcome.CANCELLED

    # The unrelated active read still retains its own gate until terminal.
    assert state.transport.retains_gate(gate) is True
    release.set()
    assert _wait_transport_idle(state)
    assert state.transport.retains_gate(gate) is False


# --------------------------------------------------------------------------- #
# Row 19 — failure-total commit with one retained retry
# --------------------------------------------------------------------------- #

def test_injected_commit_failure_retains_exact_prepared_commit_once(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, events = _bound_state(monkeypatch, tmp_path)
    keys = _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)

    original_upsert = owner.publications.upsert
    failures = {"remaining": 1}

    def failing_upsert(publication):
        if failures["remaining"]:
            failures["remaining"] -= 1
            raise RuntimeError("injected publication failure")
        return original_upsert(publication)

    monkeypatch.setattr(owner.publications, "upsert", failing_upsert)

    request = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 5
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)

    # Retry consumed the exact retained prepared commit once and succeeded.
    publication = owner.publications.get(2)
    assert publication is not None
    assert (2, HydrationOutcome.HYDRATED) in _completion_outcomes(state)
    assert keys[2] in state.payloads
    ready = [
        event for event in events
        if getattr(event, "frame_key", None) is keys[2]
    ]
    assert len(ready) == 1  # publication-last, exactly one public signal


def test_stale_active_checkpoint_read_publishes_nothing_but_closed_browse_runs(
    monkeypatch, tmp_path,
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    keys = _catalog(state, owner, (1, 2, 3))
    state.bind_checkpoint_hydration(owner)
    owner.records._authorize_checkpoint_hydration(object())
    owner_value, context_gate = _acquisition_identity(state)
    entered, release = _hold_reads(monkeypatch)
    request = _typed_request(
        state, owner_value, context_gate, processed, 2,
        HydrationPurpose.PREVIEW, 5, checkpoint=True,
    )
    assert state.transport.submit(request) is not None
    assert entered.wait(2.0)
    owner.records._revoke_checkpoint_recovery()
    release.set()
    assert _wait_transport_idle(state)
    assert owner.publications.get(2) is None and keys[2] not in state.payloads
    assert dict(_completion_outcomes(state))[2] is HydrationOutcome.OWNER_MISMATCH

    assert state.project(
        keys[3], 6, closed=True, owner=owner_value, commit_gate=context_gate,
    ) is None
    assert _wait_transport_idle(state) and owner.publications.get(3) is not None


def test_active_commit_holds_checkpoint_gate_until_bounded_publish_finishes(
    monkeypatch, tmp_path,
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    _catalog(state, owner, (1, 2, 3))
    state.bind_checkpoint_hydration(owner)
    owner.records._authorize_checkpoint_hydration(object())
    owner_value, context_gate = _acquisition_identity(state)
    entered, release, revoked = (
        threading.Event(), threading.Event(), threading.Event()
    )
    original = state._commit_acquisition_locked

    def held_commit(*args):
        entered.set()
        assert release.wait(2.0)
        return original(*args)

    monkeypatch.setattr(state, "_commit_acquisition_locked", held_commit)
    request = _typed_request(
        state, owner_value, context_gate, processed, 2,
        HydrationPurpose.PREVIEW, 5, checkpoint=True,
    )
    assert state.transport.submit(request) is not None
    assert entered.wait(2.0)
    worker = threading.Thread(target=lambda: (
        owner.records._revoke_checkpoint_recovery(), revoked.set()
    ))
    worker.start()
    assert not revoked.wait(0.05)
    release.set()
    assert revoked.wait(2.0)
    worker.join(2.0)
    assert _wait_transport_idle(state)


# --------------------------------------------------------------------------- #
# Row 17 — architecture census
# --------------------------------------------------------------------------- #

def test_private_flight_protocol_and_string_purpose_are_deleted():
    package_root = Path(
        str(import_module("xdart.gui.tabs.scattering").__file__)
    ).parent
    assert not (package_root / "hydration_flight.py").exists()
    sources = {
        path: path.read_text()
        for path in package_root.rglob("*.py")
    }
    for path, text in sources.items():
        assert "hydration_flight" not in text, path
        assert "HydrationFlightOwner" not in text, path
    projection_module = import_module(
        "xdart.gui.tabs.scattering.context_projection"
    )
    fields = getattr(
        projection_module.ProjectionRequest, "__dataclass_fields__", {}
    )
    assert "purpose" not in fields
    for token in ('"raw", "2d", "1d", "all"', "'raw', '2d', '1d', 'all'"):
        assert all(token not in text for text in sources.values())


def test_queued_reference_carries_no_arrays_handles_or_callbacks(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    preview_module = import_module("xrd_tools.io.frame_preview")
    hold = threading.Event()
    release = threading.Event()
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        hold.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)
    first = _typed_request(
        state, owner_value, gate, processed, 1, HydrationPurpose.PREVIEW, 5
    )
    second = _typed_request(
        state, owner_value, gate, processed, 2, HydrationPurpose.PREVIEW, 6
    )
    assert state.transport.submit(first) is not None
    assert hold.wait(timeout=10.0)
    assert state.transport.submit(second) is not None

    entry = state.transport._queued
    assert entry is not None
    request = entry.request
    assert type(request) is HydrationRequest
    for value in (request.label, request.generation, request.owner,
                  request.read_key, request.token):
        assert not isinstance(value, np.ndarray)
        assert not callable(value)
    for store in request.stores:
        assert not isinstance(store, np.ndarray)
    projection = entry.projection
    assert projection is None or type(projection).__name__ == (
        "DetectorPreviewProjection"
    )
    completion_fields = HydrationCompletion.__dataclass_fields__
    assert set(completion_fields) == {"token", "outcome", "diagnostic"}
    release.set()
    assert _wait_transport_idle(state)


def test_legacy_optional_identity_shape_is_refused_by_transport(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    _catalog(state, owner, (1, 2, 3))
    owner_value, gate = _acquisition_identity(state)
    legacy = HydrationRequest(
        2,
        HydrationPurpose.PREVIEW,
        5,
        owner_value,
        (owner.records, owner.light_records, owner.publications),
        gate,
    )
    assert legacy.read_key is None
    assert state.transport.submit(legacy) is None
    assert _wait_transport_idle(state)
    assert owner.publications.get(2) is None


# --------------------------------------------------------------------------- #
# Row 14 — accepted lightweight 1-D history survives preview churn
# --------------------------------------------------------------------------- #

def test_light_one_d_history_order_survives_preview_churn(
    monkeypatch, tmp_path
):
    labels = (1, 2, 3, 4, 5)
    state, owner, processed, _raw, _events = _bound_state(
        monkeypatch, tmp_path, labels=labels
    )
    _catalog(state, owner, labels)
    owner_value, gate = _acquisition_identity(state)

    for index, label in enumerate(labels):
        request = _typed_request(
            state, owner_value, gate, processed, label,
            HydrationPurpose.PREVIEW, 70 + index,
        )
        assert state.transport.submit(request) is not None
        assert _wait_transport_idle(state)
    history = tuple(sorted(owner.light_records.labels()))
    assert history == labels  # complete 1-D history for Single/Overlay/Waterfall
    for label in labels:
        light = owner.light_records.get(label)
        assert light is not None
        assert light.active_view().intensity_1d is not None

    # More preview churn on one frame neither reorders nor drops the history.
    request = _typed_request(
        state, owner_value, gate, processed, 3, HydrationPurpose.PREVIEW, 90
    )
    assert state.transport.submit(request) is not None
    assert _wait_transport_idle(state)
    assert tuple(sorted(owner.light_records.labels())) == labels


def test_acquisition_residency_does_not_compose_historical_publications(
    monkeypatch, tmp_path
) -> None:
    from tests.xdart.scattering.test_e3_context_contract import _configuration

    monkeypatch.setattr(display_runtime, "heavy_window", lambda _bytes: 2)
    labels = (1, 2, 3, 4)
    processed, _raw = _write_processed(tmp_path, labels=labels)
    configuration = _configuration()
    identity = RunIdentity.from_configuration(configuration)
    state, owner = _state(processed, identity=identity)
    state.bind_transport(event_sink=lambda event: None)
    keys_by_label = _catalog(state, owner, labels)
    owner_value, gate = _acquisition_identity(state)
    for label in labels:
        _hydrate_ok(state, owner_value, gate, processed, label, label)
    context = AcquisitionContext(
        context_token=new_context_token(ContextKind.ACQUISITION),
        run_configuration=configuration,
        config_generation=configuration.generation,
        config_fingerprint=configuration.fingerprint,
        run_scan_key="scan",
        source_path=str(processed),
        scan=object(),
        frame=None,
        frame_ids=state.catalog,
        frames=state.artifacts,
        viewer_rows_1d=(),
        viewer_rows_2d=(),
        publication_store=state,
        origin="scattering-standard",
        poni_identity=configuration.poni_file,
    )
    context.adopt_record_store(state)
    keys = tuple(keys_by_label[label] for label in labels)
    projection_owner = ContextProjection()
    expected = projection_owner.resident_frame_keys(context, keys)
    assert expected

    def fail_compose(*_args, **_kwargs):
        raise AssertionError("residency projection materialized a publication")

    monkeypatch.setattr(owner.publications, "_compose_locked", fail_compose)
    assert projection_owner.resident_frame_keys(context, keys) == expected


def test_sixteen_trace_members_project_from_light_history_with_heavy_window_eight(
    monkeypatch, tmp_path
) -> None:
    """Accumulator membership is not bounded by detector/cake residency."""

    from tests.xdart.scattering.test_e3_context_contract import _configuration

    monkeypatch.setattr(display_runtime, "heavy_window", lambda _bytes: 8)
    labels = tuple(range(1, 17))
    processed, _raw = _write_processed(tmp_path, labels=labels)
    configuration = _configuration()
    identity = RunIdentity.from_configuration(configuration)
    state, owner = _state(processed, identity=identity)
    state.bind_transport(event_sink=lambda event: None)
    keys_by_label = _catalog(state, owner, labels)
    owner_value, gate = _acquisition_identity(state)
    for label in labels:
        _hydrate_ok(
            state,
            owner_value,
            gate,
            processed,
            label,
            label,
        )

    context = AcquisitionContext(
        context_token=new_context_token(ContextKind.ACQUISITION),
        run_configuration=configuration,
        config_generation=configuration.generation,
        config_fingerprint=configuration.fingerprint,
        run_scan_key="scan",
        source_path=str(processed),
        scan=object(),
        frame=None,
        frame_ids=state.catalog,
        frames=state.artifacts,
        viewer_rows_1d=(),
        viewer_rows_2d=(),
        publication_store=state,
        origin="scattering-standard",
        poni_identity=configuration.poni_file,
    )
    context.adopt_record_store(state)
    runtime = _ContextRuntime()
    runtime.adopt_acquisition(identity, context)
    keys = tuple(keys_by_label[label] for label in labels)
    assert runtime.select_navigation(keys[-1], keys)

    projection_owner = ContextProjection()
    payloads = runtime.project_navigation(projection_owner)
    resident = projection_owner.resident_frame_keys(context, keys)
    scientific = build_scientific_projection(
        payloads,
        runtime.navigation,
        resident,
        ScientificPreferences(plot_mode="Overlay"),
        "",
    )

    assert len(resident) == 8
    assert len(payloads) == 16
    assert len(scientific.traces) == 16
    assert tuple(trace.frame for trace in scientific.traces) == keys
    assert scientific.heavy is not None
    assert scientific.heavy.frame is keys[-1]


# --------------------------------------------------------------------------- #
# Part 2 — Browse context rows through the production controller
# --------------------------------------------------------------------------- #

def _adopted_browse(tmp_path, *, labels=(1, 2, 3), loader_max=32):
    from tests.xdart.scattering.test_e3_context_contract import (
        _running_controller,
    )
    from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
    from xdart.gui.tabs.scattering.context_controller import ContextController
    from xdart.gui.tabs.scattering.context_projection import ContextProjection

    processed, _raw = _write_processed(tmp_path, labels=labels)
    _, lifecycle, executor, _, acquisition = _running_controller()
    loader = BrowseLoader(max_items=loader_max)
    controller = ContextController(
        lifecycle=lifecycle,
        executor=executor,
        browse_loader=loader,
        projection=ContextProjection(),
    )
    controller.adopt_acquisition(executor.identity)
    controller.pause()
    request = controller.begin_browse(str(processed))
    deadline = time.monotonic() + 15.0
    outcome = None
    while outcome is None and time.monotonic() < deadline:
        outcome = controller.poll_browse()
        if outcome is None:
            time.sleep(0.005)
    assert outcome is not None and outcome.request is request
    browse = controller.browse_context
    assert browse is not None and browse.loaded
    return controller, acquisition, browse, processed


def _adopted_cold_browse(
    tmp_path,
    *,
    labels=(1, 2, 3),
    loader_max=1,
    thumbnails=True,
    two_d=True,
):
    from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
    from xdart.gui.tabs.scattering.context_controller import ContextController
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator

    processed, _raw = _write_processed(
        tmp_path,
        labels=labels,
        thumbnails=thumbnails,
        two_d=two_d,
    )
    loader = BrowseLoader(max_items=loader_max)
    controller = ContextController(
        lifecycle=ScatteringCoordinator(),
        executor=object(),
        browse_loader=loader,
        projection=ContextProjection(),
    )
    request = controller.begin_browse(str(processed))
    deadline = time.monotonic() + 15.0
    outcome = None
    while outcome is None and time.monotonic() < deadline:
        outcome = controller.poll_browse()
        if outcome is None:
            time.sleep(0.005)
    assert outcome is not None and outcome.request is request
    browse = controller.browse_context
    assert browse is not None and browse.loaded
    assert controller.acquisition_context is None
    assert controller.run_identity is None
    return controller, browse, processed


def _browse_key(controller, label):
    key = next(
        frame
        for frame in controller.frame_keys
        if frame.local_frame_label == label
    )
    assert controller.select_navigation(key, (key,))
    return key


def _demote_browse_publication(browse, label):
    """Evict one B publication through its own store's demotion seam."""
    store = browse.publication_store
    publication = store.get(label)
    assert publication is not None
    assert store.evict_heavy(label)
    assert store.evict_thumbnail(label)
    view = store.get(label).view
    assert view.raw is None and view.thumbnail is None


def test_browse_evicted_frame_rehydrates_b_store_without_second_context(
    tmp_path,
):
    controller, acquisition, browse, processed = _adopted_browse(tmp_path)
    transport = acquisition.publication_store.transport
    key = _browse_key(controller, 2)
    reference = browse.publication_store.get(2)
    assert reference is not None
    reference_source = (reference.view.source_path, reference.view.source_frame_index)
    _demote_browse_publication(browse, 2)

    light = controller.resolve_projection(
        controller.project_request(key, require_complete=False)
    )  # truthful retained trace/cake history, no detector
    assert light is not None
    assert light.view.raw is None and light.view.thumbnail is None
    assert controller.project(key) is None  # full current-anchor hydration starts
    deadline = time.monotonic() + 10.0
    payload = None
    while time.monotonic() < deadline:
        if (
            transport.queued_token is None
            and transport.active_token is None
        ):
            candidate = controller.project(key)
            if (
                candidate is not None
                and candidate.view.thumbnail is not None
            ):
                payload = candidate
                break
        time.sleep(0.005)
    assert payload is not None
    assert payload.frame_key is key
    assert (payload.view.source_path, payload.view.source_frame_index) == reference_source
    restored = browse.publication_store.get(2)
    assert restored is not None
    assert restored.scan_key == browse.scan_key
    assert restored.view.thumbnail is not None
    assert restored.view.intensity_1d is not None
    assert (restored.view.source_path, restored.view.source_frame_index) == reference_source
    # The request landed in B's own store through B's exact context: the
    # acquisition display gained no browse frame.
    assert acquisition.publication_store.artifacts.get(str(processed)) is None
    outcomes = transport.counters()
    assert outcomes[HydrationOutcome.HYDRATED] >= 1


def test_cold_browse_evicted_frame_rehydrates_its_exact_store(
    tmp_path,
):
    controller, browse, processed = _adopted_cold_browse(tmp_path)
    key = _browse_key(controller, 1)
    reference = read_frame_record(processed, 1).active_view()
    reference_source = (reference.source_path, reference.source_frame_index)
    assert browse.publication_store.get(1) is None

    assert controller.project(key) is None
    transport = controller._browse_hydration_owner.transport
    assert transport is not None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if (
            transport.queued_token is None
            and transport.active_token is None
        ):
            break
        time.sleep(0.005)

    payload = controller.project(key)
    assert payload is not None
    assert payload.frame_key is key
    assert (payload.view.source_path, payload.view.source_frame_index) == reference_source
    restored = browse.publication_store.get(1)
    assert restored is not None
    assert restored.scan_key == browse.scan_key
    assert restored.view.thumbnail is not None
    assert (restored.view.source_path, restored.view.source_frame_index) == reference_source
    assert browse.requested_path == str(processed)
    assert controller.browse_context is browse
    assert controller.acquisition_context is None
    assert controller.run_identity is None
    assert transport.counters()[HydrationOutcome.HYDRATED] >= 1

    receipt = controller.close()
    assert receipt.cleanup_status.value == "cleaned"
    assert browse.released is True
    assert transport.retains_gate(browse.commit_gate) is False
    assert transport.worker is None or not transport.worker.is_alive()


def test_cold_browse_raw_absent_thumbnail_rehydrates(monkeypatch, tmp_path):
    controller, browse, processed = _adopted_cold_browse(tmp_path)
    key = _browse_key(controller, 1)
    raw_path = tmp_path / "raw" / "image.tif"
    raw_path.unlink()
    assert browse.publication_store.get(1) is None
    counts = _instrument_reads(monkeypatch, processed)

    assert controller.project(key) is None
    transport = controller._browse_hydration_owner.transport
    assert transport is not None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if transport.queued_token is None and transport.active_token is None:
            break
        time.sleep(0.005)

    payload = controller.project(key)
    assert payload is not None and payload.frame_key is key
    assert payload.view.thumbnail is not None
    assert (payload.view.source_path, payload.view.source_frame_index) == (
        "raw/image.tif", 0
    )
    restored = browse.publication_store.get(1)
    assert restored is not None and restored.view.thumbnail is not None
    assert restored.view.intensity_1d is not None
    assert (restored.view.source_path, restored.view.source_frame_index) == (
        "raw/image.tif", 0
    )
    assert counts["processed"] == 1 and counts["detector"] == 0
    assert transport.counters()[HydrationOutcome.HYDRATED] >= 1

    receipt = controller.close()
    assert receipt.cleanup_status.value == "cleaned"
    assert transport.retains_gate(browse.commit_gate) is False


def test_cold_browse_no_thumbnail_exact_read_terminalizes_once(
    monkeypatch, tmp_path
):
    controller, browse, processed = _adopted_cold_browse(
        tmp_path,
        thumbnails=False,
    )
    key = _browse_key(controller, 1)
    assert browse.publication_store.get(1) is None
    counts = _instrument_reads(monkeypatch, processed)
    assert controller.project(key) is None
    transport = controller._browse_hydration_owner.transport
    assert transport is not None
    deadline = time.monotonic() + 10.0
    while (
        transport.queued_token is not None
        or transport.active_token is not None
    ) and time.monotonic() < deadline:
        time.sleep(0.005)

    restored = browse.publication_store.get(1)
    assert restored is not None
    assert restored.view.thumbnail is None
    assert restored.view.raw is None
    light = controller.resolve_projection(
        controller.project_request(key, require_complete=False)
    )
    assert light is not None
    np.testing.assert_allclose(
        light.view.intensity_1d,
        np.array([2.0, 3.0, 4.0]),
    )
    assert counts == {
        "processed": 1,
        "detector": 0,
        "read_threads": ["scattering-preview-transport"],
    }
    assert transport.counters()[HydrationOutcome.HYDRATED] == 1
    assert controller.poll_browse_preview() is True
    assert controller.poll_browse_preview() is False

    # The repaint projects the truthful retained 1D payload but cannot invent
    # a detector mask/value policy.  That exact completed read is terminal for
    # this immutable Browse context, so repaint must not enqueue it again.
    complete = controller.project(key)
    assert complete is not None
    assert complete.view.raw is None
    np.testing.assert_allclose(
        complete.view.intensity_1d,
        np.array([2.0, 3.0, 4.0]),
    )
    time.sleep(0.05)
    assert transport.queued_token is None
    assert transport.active_token is None
    assert transport.counters()[HydrationOutcome.HYDRATED] == 1
    assert controller.browse_preview_polling_needed is False

    receipt = controller.close()
    assert receipt.cleanup_status.value == "cleaned"
    assert browse.released is True
    assert transport.worker is None or not transport.worker.is_alive()


def test_cold_browse_close_retries_its_exact_active_transport(
    monkeypatch, tmp_path
):
    from xdart.gui.tabs.scattering.events import CleanupStatus

    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    key = _browse_key(controller, 1)
    assert browse.publication_store.get(1) is None
    hold = threading.Event()
    release = threading.Event()
    preview_module = import_module("xrd_tools.io.frame_preview")
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        hold.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)
    assert controller.project(key) is None
    transport = controller._browse_hydration_owner.transport
    assert hold.wait(timeout=10.0)

    receipt = controller.close()

    assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert transport.retains_gate(browse.commit_gate) is True
    assert browse.released is False

    release.set()
    deadline = time.monotonic() + 10.0
    while (
        receipt.cleanup_status is not CleanupStatus.CLEANED
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
        receipt = controller.close()
    assert receipt.cleanup_status is CleanupStatus.CLEANED
    assert transport.retains_gate(browse.commit_gate) is False
    assert transport.worker is None or not transport.worker.is_alive()
    assert browse.released is True


def test_resume_makes_late_browse_completion_terminal_bookkeeping_only(
    monkeypatch, tmp_path
):
    controller, acquisition, browse, _processed = _adopted_browse(tmp_path)
    transport = acquisition.publication_store.transport
    key = _browse_key(controller, 2)
    _demote_browse_publication(browse, 2)

    preview_module = import_module("xrd_tools.io.frame_preview")
    hold = threading.Event()
    release = threading.Event()
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        hold.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)
    controller.project(key)  # light payload; submits the exact B preview
    assert hold.wait(timeout=10.0)

    selection = controller.resume()
    assert selection.names(acquisition)
    assert browse.invalidated
    release.set()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if transport.retains_gate(browse.commit_gate) is False:
            break
        time.sleep(0.005)
    assert transport.retains_gate(browse.commit_gate) is False

    late = browse.publication_store.get(2)
    assert late is not None and late.view.thumbnail is None  # nothing landed
    counters = transport.counters()
    assert (
        counters[HydrationOutcome.CANCELLED]
        + counters[HydrationOutcome.OWNER_MISMATCH]
    ) >= 1
    # Invalidated browse can no longer submit: the accepted foreign-frame
    # refusal fires first (case_10 contract) and no read is admitted.
    with pytest.raises(RuntimeError, match="no display selection"):
        controller.project(key)
    assert transport.queued_token is None and transport.active_token is None


def test_close_withholds_cleaned_until_transport_drops_browse_request(
    monkeypatch, tmp_path
):
    from xdart.gui.tabs.scattering.events import CleanupStatus

    controller, acquisition, browse, _processed = _adopted_browse(tmp_path)
    transport = acquisition.publication_store.transport
    key = _browse_key(controller, 2)
    _demote_browse_publication(browse, 2)

    preview_module = import_module("xrd_tools.io.frame_preview")
    hold = threading.Event()
    release = threading.Event()
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        hold.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)
    controller.project(key)  # light payload; submits the exact B preview
    assert hold.wait(timeout=10.0)

    started = time.monotonic()
    receipt = controller.close()
    elapsed = time.monotonic() - started
    assert elapsed < 2.0  # no GUI-thread join while the read is held
    assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert transport.retains_gate(browse.commit_gate) is True
    assert browse.released is False  # B stores not released early

    release.set()
    deadline = time.monotonic() + 10.0
    final = receipt
    while (
        final.cleanup_status is not CleanupStatus.CLEANED
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
        final = controller.close()
    assert final.cleanup_status is CleanupStatus.CLEANED
    assert transport.retains_gate(browse.commit_gate) is False
    assert browse.released is True


# --------------------------------------------------------------------------- #
# Part 3 — mounted ScatteringWorkspace acceptance rows
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("thumbnails", (True, False))
def test_mounted_cold_browse_completion_wakes_and_repaints(
    monkeypatch, tmp_path, thumbnails
):
    from pyqtgraph.Qt import QtWidgets

    from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
    from xdart.gui.tabs.scattering.adapters.source import (
        FilesystemSourceAdapter,
    )
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.workspace_shell import (
        ScatteringWorkspaceShell,
    )
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent

    processed, _raw = _write_processed(
        tmp_path,
        labels=(1, 2, 3),
        thumbnails=thumbnails,
    )
    loader = BrowseLoader(max_items=1)
    page_module = import_module("xdart.gui.tabs.scattering.page")
    monkeypatch.setattr(page_module, "BrowseLoader", lambda: loader)
    hold = threading.Event()
    release = threading.Event()
    preview_module = import_module("xrd_tools.io.frame_preview")
    original = preview_module.read_frame_preview
    reads = []

    def holding_read(read_key, **kwargs):
        reads.append(read_key)
        hold.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    lifecycle = ScatteringCoordinator()
    page = page_module.ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(save_path=str(processed.parent))
        ),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
    )
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    controller = page._context_controller
    close_receipt = None

    def wait_for(predicate, *, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.002)
        raise AssertionError("cold Browse repaint did not settle")

    try:
        page._select_scan(str(processed))
        wait_for(
            lambda: (
                controller.browse_context is not None
                and hold.is_set()
            )
        )
        browse = controller.browse_context
        assert browse is not None
        assert controller.acquisition_context is None
        assert controller.run_identity is None
        assert browse.publication_store.get(1) is None
        assert shell.scientific.raw.image.image is None
        assert page._run_timer.isActive()

        release.set()
        wait_for(
            lambda: (
                shell.scientific.raw.image.image is not None
                if thumbnails
                else bool(shell.scientific.curve.listDataItems())
            )
        )
        restored = browse.publication_store.get(1)
        assert restored is not None
        assert (restored.view.thumbnail is not None) is thumbnails
        assert restored.view.raw is None
        assert bool(shell.scientific.curve.listDataItems())
        assert controller.browse_context is browse
        assert lifecycle.phase.value == "idle"
        wait_for(lambda: not page._run_timer.isActive())
        assert len(reads) == 1
        assert controller.poll_browse_preview() is False
        assert controller.browse_preview_polling_needed is False
    finally:
        release.set()
        close_receipt = page.close_workspace()
        page.deleteLater()
        app.processEvents()
    assert close_receipt.cleanup_status.value == "cleaned"


def _lv():
    return import_module("tests.xdart.scattering.test_e2lv_live_display")


def test_mounted_selection_retains_presentation_then_restores_thumbnail(
    monkeypatch, tmp_path
):
    lv_support = _lv()
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "2")
    monkeypatch.setattr(display_runtime, "THUMBNAIL_MAX_ITEMS", 2)
    qapp, page, lifecycle, _executor, output = lv_support._standard_page(
        monkeypatch, tmp_path, labels=tuple(range(1, 11))
    )
    shell, controller = lv_support._mounted(page)
    try:
        def run_diagnostic():
            active = _executor._active
            worker = None if active is None else active.worker
            return (
                f"phase={lifecycle.phase.value}; "
                f"notice={page._notice_text!r}; "
                f"timer={page._run_timer.isActive()}; "
                f"admission={page._admission is not None}; "
                f"context_identity={controller.run_identity!r}; "
                f"executor_active={active is not None}; "
                f"worker_alive={bool(worker and worker.is_alive())}"
            )

        lv_support._wait(
            qapp, lambda: shell.run_controls.startButton.isEnabled()
        )
        shell.run_controls.startButton.click()
        lv_support._wait(
            qapp,
            lambda: lifecycle.phase.value == "idle",
            timeout=60.0,
            diagnostic=run_diagnostic,
        )
        display = controller.acquisition_context.publication_store
        artifact_owner = next(iter(display.artifacts.values()))
        demoted = artifact_owner.publications.get(1)
        assert demoted is None or (
            demoted.view.raw is None and demoted.view.thumbnail is None
        )
        held_title = shell.scientific.title.text()
        assert held_title == "tiny_0001.tif"
        held_image = shell.scientific.raw.image.image
        assert held_image is not None

        preview_module = import_module("xrd_tools.io.frame_preview")
        hold = threading.Event()
        release = threading.Event()
        original = preview_module.read_frame_preview

        def holding_read(read_key, **kwargs):
            hold.set()
            release.wait(timeout=15.0)
            return original(read_key, **kwargs)

        monkeypatch.setattr(
            _transport_api(), "read_frame_preview", holding_read
        )
        counts = _instrument_reads(monkeypatch, output)

        first = next(
            frame
            for frame in controller.frame_keys
            if frame.local_frame_label == 1
        )
        lv_support._select_exact_frame(shell, first)
        assert hold.wait(timeout=15.0)
        qapp.processEvents()
        # Exact same-selection pending: the last coherent presentation stays
        # visible — no blank, no stale compatible-looking replacement.
        assert shell.scientific.title.text() == held_title
        assert shell.scientific.raw.image.image is not None

        release.set()
        transport = display.transport
        lv_support._wait(
            qapp,
            lambda: (
                shell.scientific.frame_selector.currentData() is first
                and shell.scientific.title.text() == "tiny_0001.tif"
                and shell.scientific.raw.image.image is not None
                and transport.counters()[HydrationOutcome.HYDRATED] >= 1
            ),
            timeout=30.0,
        )
        assert counts["processed"] == 1
        assert counts["detector"] == 0
        assert transport.counters()[HydrationOutcome.HYDRATED] >= 1
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_651_frame_run_hydrates_oldest_frame_with_bounded_opens(
    monkeypatch, tmp_path
):
    lv_support = _lv()
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    monkeypatch.setattr(display_runtime, "THUMBNAIL_MAX_ITEMS", 8)
    qapp, page, lifecycle, _executor, output = lv_support._standard_page(
        monkeypatch, tmp_path, labels=tuple(range(1, 652))
    )
    shell, controller = lv_support._mounted(page)
    try:
        lv_support._wait(
            qapp, lambda: shell.run_controls.startButton.isEnabled()
        )
        shell.run_controls.startButton.click()
        lv_support._wait(
            qapp, lambda: lifecycle.phase.value == "idle", timeout=300.0
        )
        display = controller.acquisition_context.publication_store
        keys = controller.frame_keys
        assert keys and keys[-1].local_frame_label == 651
        oldest = keys[0]
        counts = _instrument_reads(monkeypatch, output)

        lv_support._select_exact_frame(shell, oldest)
        lv_support._wait(
            qapp,
            lambda: (
                shell.scientific.frame_selector.currentData() is oldest
                and shell.scientific.title.text() == "tiny_0001.tif"
                and shell.scientific.raw.image.image is not None
            ),
            timeout=60.0,
        )
        # One historical selection: one processed open, zero detector reads,
        # no full-history reconstruction, bounded caches, quiescent transport.
        assert counts["processed"] == 1
        assert counts["detector"] == 0
        assert len(display.payloads) <= display.max_payload_items
        assert _wait_transport_idle(display)
        snapshot = display.residency_snapshot()
        assert snapshot.heavy <= snapshot.limits.heavy
        assert snapshot.thumbnails <= snapshot.limits.thumbnails
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_post_run_overlay_to_single_browser_selection_repaints_hydrated_frame(
    monkeypatch, tmp_path
):
    from pyqtgraph.Qt import QtCore

    lv_support = _lv()
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    monkeypatch.setattr(display_runtime, "THUMBNAIL_MAX_ITEMS", 8)
    qapp, page, lifecycle, _executor, _output = lv_support._standard_page(
        monkeypatch, tmp_path, labels=tuple(range(1, 33))
    )
    shell, controller = lv_support._mounted(page)
    try:
        shell.scientific.plot_mode.setCurrentText("Overlay")
        lv_support._wait(
            qapp, lambda: shell.run_controls.startButton.isEnabled()
        )
        shell.run_controls.startButton.click()
        lv_support._wait(
            qapp, lambda: lifecycle.phase.value == "idle", timeout=90.0
        )
        reconciled_heavy: list[object] = []
        original_reconcile = shell.scientific.reconcile

        def capture_reconcile(state, navigation, **kwargs):
            if state.heavy is not None:
                reconciled_heavy.append(state.heavy.frame)
            return original_reconcile(state, navigation, **kwargs)

        monkeypatch.setattr(
            shell.scientific, "reconcile", capture_reconcile
        )
        shell.scientific.plot_mode.setCurrentText("Single")
        lv_support._wait(
            qapp,
            lambda: shell.scientific.plot_mode.currentText() == "Single",
        )

        oldest = controller.frame_keys[0]
        row = shell.browser.frame_model.row_for(oldest)
        assert row == 0
        index = shell.browser.frame_model.index(row, 0)
        shell.browser.frames.selectionModel().setCurrentIndex(
            index,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        lv_support._wait(
            qapp,
            lambda: (
                controller.navigation.current is oldest
                and shell.scientific.frame_selector.currentData() is oldest
            ),
            timeout=15.0,
        )
        lv_support._wait(
            qapp,
            lambda: any(frame is oldest for frame in reconciled_heavy),
            timeout=30.0,
        )
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_batch_run_click_projects_phase_and_control_lock_without_display_repaint(
    monkeypatch, tmp_path
):
    lv_support = _lv()
    qapp, page, lifecycle, _executor, _output = lv_support._standard_page(
        monkeypatch, tmp_path, labels=tuple(range(1, 65))
    )
    shell, _controller = lv_support._mounted(page)
    captures: list[tuple[str, bool, str, bool, bool, bool]] = []
    original_apply = shell.apply_state

    def capture_apply(state, *, preserve_display=False):
        original_apply(state, preserve_display=preserve_display)
        fields = state.controls.fields
        captures.append(
            (
                state.run.phase.value,
                bool(preserve_display),
                shell.run_controls.startButton.text(),
                shell.run_controls.startButton.isEnabled(),
                shell.run_controls.modeCombo.isEnabled(),
                any(field.enabled for field in fields),
            )
        )

    monkeypatch.setattr(shell, "apply_state", capture_apply)
    try:
        shell.run_controls.batchButton.click()
        lv_support._wait(
            qapp, lambda: shell.run_controls.batchButton.isChecked()
        )
        captures.clear()

        shell.run_controls.startButton.click()

        # The command handler owns this first projection synchronously.  It
        # locks the run/control surface but deliberately retains the outgoing
        # browser and scientific paint while admission is pending.
        assert captures
        preparing = captures[-1]
        assert preparing[:2] == ("preparing", True)
        assert "Pause" in preparing[2]
        assert preparing[3:] == (False, False, False)

        lv_support._wait(
            qapp,
            lambda: any(phase == "running" for phase, *_ in captures),
            timeout=30.0,
        )
        running = next(
            capture for capture in captures if capture[0] == "running"
        )
        assert running[1] is True
        assert "Pause" in running[2]
        assert running[3:] == (True, False, False)
        lv_support._wait(
            qapp, lambda: lifecycle.phase.value == "idle", timeout=60.0
        )
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


# E6-PM2 admission-instance tickets (ratified 2026-08-04): HydrationToken is a
# VALUE, so a re-admitted equal-generation read yields an equal token while the
# first completion still sits in the bounded diagnostic deque.  The transport
# mints one private ticket per admission; ticket OBJECT IDENTITY is the
# admission identity.  The deque and counters stay raw.


def _ticket_type():
    return getattr(_transport_api(), "_HydrationTicket", None)


def _admit(transport, request, *, closed=False):
    seam = getattr(transport, "_submit_admission", None)
    assert seam is not None, "transport must expose the private ticket seam"
    return seam(request, closed=closed)


def _hold_reads(monkeypatch):
    """Block the worker inside the real read; returns (entered, release)."""

    entered = threading.Event()
    release = threading.Event()
    original = _transport_api().read_frame_preview

    def held(*args, **kwargs):
        entered.set()
        release.wait(timeout=10.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", held)
    return entered, release


def test_e6pm2_ticket_contract_refusal_and_distinct_admissions(
    monkeypatch, tmp_path
):
    api = _transport_api()
    ticket_type = _ticket_type()
    assert ticket_type is not None, "hydration_transport needs _HydrationTicket"
    assert api.__all__ == ["HydrationTransport", "PreparedHydrationCommit"]

    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    art_owner, gate = _acquisition_identity(state)
    transport = state.transport
    request = _typed_request(
        state, art_owner, gate, processed, 2, HydrationPurpose.PREVIEW, 1
    )

    first = _admit(transport, request)
    assert type(first) is ticket_type
    assert first.token == request.token
    assert first.result() is None and first.result() is None
    assert _wait_transport_idle(state)
    assert first.result() is not None

    # Equal-valued token, a brand-new admission: a DIFFERENT ticket object.
    second = _admit(transport, request)
    assert second is not first
    assert second.token == first.token
    assert _wait_transport_idle(state)

    # The public seam still speaks tokens only, never a ticket.
    third = transport.submit(request)
    assert type(third) is HydrationToken
    assert _wait_transport_idle(state)

    # A refused admission (retired transport) mints nothing.
    transport.retire(join_timeout=1.0)
    assert _admit(transport, request) is None
    assert transport.submit(request) is None


def test_e6pm2_same_active_same_token_resubmit_returns_one_ticket(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    art_owner, gate = _acquisition_identity(state)
    transport = state.transport
    request = _typed_request(
        state, art_owner, gate, processed, 2, HydrationPurpose.PREVIEW, 1
    )
    entered, release = _hold_reads(monkeypatch)

    first = _admit(transport, request)
    assert entered.wait(timeout=10.0)
    assert _admit(transport, request) is first  # one outstanding receipt

    release.set()
    assert _wait_transport_idle(state)
    assert first.result() is not None
    assert first.result().outcome is not HydrationOutcome.SUPERSEDED


def test_e6pm2_moved_presentation_supersedes_the_displaced_ticket(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    art_owner, gate = _acquisition_identity(state)
    transport = state.transport
    entered, release = _hold_reads(monkeypatch)

    ticket_a = _admit(
        transport,
        _typed_request(
            state, art_owner, gate, processed, 2, HydrationPurpose.PREVIEW, 1
        ),
    )
    assert entered.wait(timeout=10.0)
    before = len(transport.completions())
    ticket_b = _admit(
        transport,
        _typed_request(
            state, art_owner, gate, processed, 2, HydrationPurpose.PREVIEW, 2
        ),
    )
    assert ticket_b is not ticket_a
    # The displaced presentation settles at once; the raw deque still records it.
    assert ticket_a.result().outcome is HydrationOutcome.SUPERSEDED
    assert len(transport.completions()) == before + 1
    assert transport.counters()[HydrationOutcome.SUPERSEDED] >= 1
    assert ticket_b.result() is None

    release.set()
    assert _wait_transport_idle(state)
    assert ticket_b.result() is not None
    # First-wins keeps the superseded fact immutable through the final split.
    assert ticket_a.result().outcome is HydrationOutcome.SUPERSEDED


def test_e6pm2_settlement_validates_token_and_is_first_wins(
    monkeypatch, tmp_path
):
    ticket_type = _ticket_type()
    assert ticket_type is not None
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    art_owner, gate = _acquisition_identity(state)
    mine = _typed_request(
        state, art_owner, gate, processed, 2, HydrationPurpose.PREVIEW, 1
    )
    other = _typed_request(
        state, art_owner, gate, processed, 3, HydrationPurpose.PREVIEW, 1
    )
    ticket = ticket_type(mine.token)

    assert ticket._settle(
        HydrationCompletion(other.token, HydrationOutcome.HYDRATED)
    ) is False
    assert ticket.result() is None

    exact = HydrationCompletion(mine.token, HydrationOutcome.FAILED, "one")
    assert ticket._settle(exact) is True
    assert ticket.result() is exact
    assert ticket._settle(
        HydrationCompletion(mine.token, HydrationOutcome.HYDRATED, "two")
    ) is False
    assert ticket.result() is exact


def test_e6pm2_queued_displacement_cancel_and_worker_failure_settle_exactly(
    monkeypatch, tmp_path
):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    art_owner, gate = _acquisition_identity(state)
    transport = state.transport
    entered, release = _hold_reads(monkeypatch)

    active = _admit(
        transport,
        _typed_request(
            state, art_owner, gate, processed, 2, HydrationPurpose.PREVIEW, 1
        ),
    )
    assert entered.wait(timeout=10.0)
    queued_first = _admit(
        transport,
        _typed_request(
            state, art_owner, gate, processed, 3, HydrationPurpose.PREVIEW, 1
        ),
    )
    queued_second = _admit(
        transport,
        _typed_request(
            state, art_owner, gate, processed, 4, HydrationPurpose.PREVIEW, 1
        ),
    )
    assert queued_first.result().outcome is HydrationOutcome.SUPERSEDED
    assert queued_second.result() is None

    transport.cancel_gate(gate)
    assert queued_second.result().outcome is HydrationOutcome.CANCELLED
    # An ACTIVE ticket settles only through its own execution/commit path.
    assert active.result() is None

    release.set()
    assert _wait_transport_idle(state)
    assert active.result() is not None
    assert active.result().outcome is not HydrationOutcome.CANCELLED


def test_e6pm2_worker_start_failure_settles_failed(monkeypatch, tmp_path):
    state, owner, processed, _raw, _events = _bound_state(monkeypatch, tmp_path)
    art_owner, gate = _acquisition_identity(state)

    def refuse_start(self):
        raise RuntimeError("no thread")

    monkeypatch.setattr(threading.Thread, "start", refuse_start)
    ticket = _admit(
        state.transport,
        _typed_request(
            state, art_owner, gate, processed, 2, HydrationPurpose.PREVIEW, 1
        ),
    )
    assert ticket is not None
    assert ticket.result().outcome is HydrationOutcome.FAILED


def _b1_acquisition(tmp_path, labels):
    from tests.xdart.scattering.test_e3_context_contract import _configuration

    processed, raw_path = _write_processed(tmp_path, labels=labels)
    configuration = _configuration()
    identity = RunIdentity.from_configuration(configuration)
    state, artifact = _state(processed, identity=identity)
    state.max_payload_items = 16
    if len(labels) > 2:
        from xrd_tools.session import Light1DBufferLayout, Light1DLayout, Light1DModeLayout, SessionResourceAuthority, SessionResourceRequirements, acquire_light_1d_retention, resolve_session_policy
        allocation = resolve_session_policy(SessionResourceRequirements(4, 4, 2, modes_1d=1, modes_2d=1, npt_1d=3, npt_rad=3, npt_azim=2), envelope_bytes=4 * 1024 ** 3, requests={"record_heavy_items": 1, "publication_heavy_items": 1}, env={}).allocation
        artifact.publications.bind_allocation(allocation); state.bind_heavy_allocation(allocation); artifact.mask = np.eye(4, dtype=bool)
        buffer = lambda owner: Light1DBufferLayout(3, 8, owner, np.dtype(np.float64).str)
        layout = Light1DLayout((Light1DModeLayout("default", buffer("q"), buffer("i")),), "default")
        lease = acquire_light_1d_retention(SessionResourceAuthority.from_allocation(allocation), owner="b1", generation=1, layout=layout, requested_rows=len(labels), compatibility_byte_ceiling=layout.shared_bytes + len(labels) * layout.per_row_unique_ndarray_bytes, gui_thread_id=threading.get_ident())
        artifact.publications.bind_light_1d(lease); hooks = artifact.publications.light_1d_cleanup_hooks(lease); state.stage_light_1d(artifact, lease, hooks=hooks); state.bind_light_1d(artifact, lease)
    events = []; state.bind_transport(event_sink=events.append)
    keys = _catalog(state, artifact, labels)
    context = AcquisitionContext(
        new_context_token(ContextKind.ACQUISITION), configuration,
        configuration.generation, configuration.fingerprint, "scan",
        str(processed), object(), None, state.catalog, state.artifacts, (), (),
        state, origin="scattering-standard", poni_identity=configuration.poni_file,
    )
    context.adopt_record_store(state)
    runtime = _ContextRuntime(); runtime.adopt_acquisition(identity, context)
    for generation, label in enumerate(labels, 1):
        _hydrate_ok(state, context.hydration_owner, context.commit_gate,
                    processed, label, generation)
    events.clear()
    return runtime, state, artifact, context, keys, raw_path, events


def test_b1_light_publish_and_full_admission_share_one_lock_order(
    monkeypatch, tmp_path,
):
    """A writer publication cannot deadlock full-resolution admission.

    Full admission owns the light-admission and transport locks while deriving
    under the display lock.  The writer must therefore wait for light
    admission *before* taking the display lock, leaving derivation able to
    finish and release the shared admission seam.
    """
    _runtime, state, artifact, context, keys, _raw, _events = _b1_acquisition(
        tmp_path, (1, 2, 3),
    )
    key = keys[3]
    record = artifact.records.get(3)
    before = artifact.publications.get(3)
    assert record is not None and before is not None
    assert not artifact.publications.has_raw(3)

    derive_entered = threading.Event()
    release_derive = threading.Event()
    writer_attempted_admission = threading.Event()
    original_derive = state.transport._derive
    original_admission_lock = state._light_admission_lock

    class ObservedAdmissionLock:
        def acquire(self, *args, **kwargs):
            if threading.current_thread().name == "b1-light-writer":
                writer_attempted_admission.set()
            return original_admission_lock.acquire(*args, **kwargs)

        def release(self):
            return original_admission_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            self.release()

    def held_derive(request):
        assert request.read_key.frame_identity == 3
        assert request.read_key.purpose is HydrationPurpose.FULL
        derive_entered.set()
        assert release_derive.wait(timeout=10.0)
        return original_derive(request)

    monkeypatch.setattr(
        state, "_light_admission_lock", ObservedAdmissionLock(),
    )
    monkeypatch.setattr(state.transport, "_derive", held_derive)

    admitted = []
    published = []
    errors = []

    def admit_full():
        try:
            admitted.append(state.request_full(
                key,
                1,
                owner=context.hydration_owner,
                commit_gate=context.commit_gate,
            ))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(("admission", error))

    def publish_light():
        try:
            published.append(state.publish_light_1d(
                artifact, record, source_identity=before.source_identity,
            ))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(("publication", error))

    gui = threading.Thread(
        target=admit_full, name="b1-full-admission", daemon=True,
    )
    writer = threading.Thread(
        target=publish_light, name="b1-light-writer", daemon=True,
    )
    gui.start()
    assert derive_entered.wait(timeout=5.0)
    writer.start()
    assert writer_attempted_admission.wait(timeout=5.0)
    release_derive.set()

    deadline = time.monotonic() + 5.0
    for thread in (gui, writer):
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not gui.is_alive() and not writer.is_alive()
    assert errors == []

    assert len(admitted) == 1 and admitted[0] is not None
    token = admitted[0]
    assert token.read_key.artifact_identity == str(artifact.artifact)
    assert token.read_key.frame_identity == 3
    assert token.read_key.purpose is HydrationPurpose.FULL
    assert token.presentation_generation == 1
    assert len(published) == 1
    assert published[0].source_identity == before.source_identity
    assert published[0].scan_key == artifact.source_scan

    assert _wait_transport_idle(state)
    completions = tuple(
        completion
        for completion in state.transport.completions()
        if completion.token is token
    )
    assert len(completions) == 1
    assert completions[0].outcome is HydrationOutcome.HYDRATED
    current = artifact.publications.get(3)
    assert current is not None and artifact.publications.has_raw(3)
    assert current.source_identity == before.source_identity
    assert current.scan_key == artifact.source_scan
    assert np.array_equal(
        current.view.axis_1d.values, record.active_view().axis_1d.values,
    )
    assert np.array_equal(
        current.view.intensity_1d, record.active_view().intensity_1d,
    )


def test_b1_full_demand_latest_current_lru8_and_thumbnail_clear(monkeypatch, tmp_path):
    labels = tuple(range(1, 10))
    runtime, state, artifact, context, keys, _raw, events = _b1_acquisition(
        tmp_path, labels,
    )
    before_publication = artifact.publications.get(9); before_payload = state.payloads[keys[9]]
    nonraw = ("axis_2d_x", "axis_2d_y", "intensity_2d", "thumbnail", "metadata_raw", "extra")
    entered, release = threading.Event(), threading.Event()
    original = _transport_api().read_frame_preview

    def hold_first_full(read_key, **kwargs):
        if read_key.purpose is HydrationPurpose.FULL and read_key.frame_identity == 1:
            entered.set(); release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", hold_first_full)
    assert runtime.select_navigation(keys[1], (keys[1],))
    token_a = runtime.request_full_current()
    assert token_a.read_key.frame_identity == 1 and entered.wait(timeout=10.0)
    assert runtime.clear_full_raw()
    token_same = runtime.request_full_current()
    assert token_same is not token_a
    assert token_same.presentation_generation != token_a.presentation_generation
    assert runtime.select_navigation(keys[2], (keys[2],))
    token_b = runtime.request_full_current()
    assert token_b.read_key.frame_identity == 2
    release.set(); assert _wait_transport_idle(state)
    assert not artifact.publications.has_raw(1)
    assert artifact.publications.has_raw(2)
    assert artifact.records.get(2).active_view().raw is None
    assert not any(event.frame_key is keys[1] for event in events)
    assert next(event for event in events if event.frame_key is keys[2]).selection_generation == runtime.selection.display_generation

    monkeypatch.setattr(_transport_api(), "read_frame_preview", original)
    for label in labels:
        assert runtime.select_navigation(keys[label], (keys[label],))
        assert runtime.request_full_current().read_key.frame_identity == label
        assert _wait_transport_idle(state)
    after_publication = artifact.publications.get(9); after_payload = state.payloads[keys[9]]
    state.publish_light_1d(artifact, artifact.records.get(9), source_identity=after_publication.source_identity); light_only = artifact.publications.get(9)
    light_preserved = light_only.view.raw is after_publication.view.raw and light_only.view.mask_baked is True and all(getattr(light_only.view, name) is getattr(after_publication.view, name) for name in nonraw[:4]) and all(getattr(light_only.view, name) == getattr(after_publication.view, name) for name in nonraw[4:])
    state.retain_frame(artifact, keys[9], artifact.records.get(9), before_publication, source_identity=before_publication.source_identity, frame_mask_qualified=False); retained = artifact.publications.get(9)
    dataclasses = import_module("dataclasses"); raw_free = dataclasses.replace(after_payload, view=dataclasses.replace(after_payload.view, raw=None, mask_baked=False))
    state.put_payload(raw_free); republished = state.payloads[keys[9]]
    assert light_preserved
    assert retained.view.raw is after_publication.view.raw and retained.view.mask_baked is True
    assert all(getattr(retained.view, name) is getattr(before_publication.view, name) for name in nonraw[:4]) and all(getattr(retained.view, name) == getattr(before_publication.view, name) for name in nonraw[4:])
    assert republished.view.raw is retained.view.raw and republished.view.mask_baked is True
    assert all(getattr(republished.view, name) is getattr(raw_free.view, name) for name in nonraw[:4])
    _hydrate_ok(state, context.hydration_owner, context.commit_gate, Path(artifact.artifact), 9, 90)
    _hydrate_ok(state, context.hydration_owner, context.commit_gate, Path(artifact.artifact), 1, 91)
    limits = state._residency.limits
    state._residency.limits = type(limits)(1, 1, 1, 1)
    state.mark_durable(artifact, labels)
    assert not artifact.publications.has_raw(1)
    assert all(artifact.publications.has_raw(label) for label in labels[1:])
    assert all(state.payloads[keys[label]].view.raw is artifact.publications.get(label).view.raw for label in labels[1:])
    assert keys[1] not in state.payloads or state.payloads[keys[1]].view.raw is None
    assert runtime.clear_full_raw()
    assert all(not artifact.publications.has_raw(label) for label in labels)
    assert all(payload.view.raw is None for payload in state.payloads.values())
    state.mark_durable(artifact, labels); snapshot = state.residency_snapshot(); assert max(snapshot.heavy, snapshot.thumbnails, snapshot.browse, snapshot.live) <= 1


def test_b1_full_failure_keeps_thumbnail_and_retirement_clears_raw(tmp_path):
    runtime, state, artifact, context, keys, raw_path, events = _b1_acquisition(
        tmp_path, (1, 2),
    )
    assert runtime.select_navigation(keys[2], (keys[2],))
    assert runtime.request_full_current() is not None and _wait_transport_idle(state)
    assert artifact.publications.has_raw(2)
    before = artifact.publications.get(1); before_payload = state.payloads[keys[1]]
    raw_path.unlink()
    assert runtime.select_navigation(keys[1], (keys[1],))
    assert runtime.request_full_current() is not None and _wait_transport_idle(state)
    after = artifact.publications.get(1)
    resident, pending, diagnostic = runtime.full_raw_status()
    assert not resident and not pending and diagnostic
    assert diagnostic == state.transport.completions()[-1].diagnostic
    assert after is before
    assert state.payloads[keys[1]] is before_payload
    assert any(event.frame_key is keys[1] for event in events)
    module = import_module("xdart.modules.display_context")
    viewer = module.Viewer1DContext("b1-viewer", 1, ("scan.xye",), module.Viewer1DCommitGate())
    viewer_runtime = _ContextRuntime()
    viewer_runtime._selection = viewer_runtime.prepare_viewer_1d_begin(viewer)[2]
    assert viewer_runtime.full_raw_availability()[0] is False
    assert viewer_runtime.request_full_current() is None
    assert artifact.publications.has_raw(2)
    assert runtime.release_acquisition(state.identity)
    assert not artifact.publications.has_raw(2)
    assert runtime.full_raw_availability()[0] is False
    assert "acquisition" in runtime.full_raw_availability()[1].lower()
    browse, _context, _processed = _adopted_cold_browse(tmp_path / "browse")
    assert browse.full_raw_availability()[0] is False
    assert browse.request_full_current() is None
