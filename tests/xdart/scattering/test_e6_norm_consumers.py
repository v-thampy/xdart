"""Frozen E6-NORM-N2 consumer oracle (handoff §§25.3/26, prompt §4.4).

Frozen RED on the exact accepted E6-NORM-N1 parent
``46e68a91a423679fffefe976e557563b04d69412`` for missing consumer behavior
while the complete E6-NORM-N1 producer oracle, the protected 22-row Q2 suite
and the scattering architecture file stay green.  The two frozen §25.2
identity shapes are the acquisition tuple
``(run_identity.generation, run_identity.fingerprint, str(artifact),
source_scan)`` and the Browse tuple
``(context_token, scan_key, requested_path)``.

The consumer contract under test: ``_ContextRuntime`` captures at most ONE
candidate per refresh through the Q2 ``accepts_norm_aggregate`` gate;
``build_shell``/``build_scientific_projection`` consume that exact captured
object; ``trace_projection`` divides each admitted payload's new 1-D array
by its OWN ``resolve_monitor_norm`` value before scaling and Sum/Average;
identity and the effective channel enter both trace-cache scopes; revision
remains truthful aggregate provenance but cannot invalidate traces whose
per-frame divisor regime is unchanged.

E6-NORM-N2 correction 1 (handoff §27): the cross-token refused-switch rows
and the reserved-sentinel row were frozen RED on the exact held candidate
``19aeb62a345e79d415b2fff068347aa723ba6b96``; the fail-closed divisor
matrix, the alias-hardened census and the real four-route consumer parity
node are §27.3 oracle completion.

E6-NORM-N2 correction 2 (handoff §28): the acquisition rescope-refusal
foreign-hold rows and the lowercase-only whitespace-equivalence rows were
frozen RED on the exact rejected correction-1 descendant
``e4c5eea2520c35731a72a5c8bf4926f493ea087e``.  Channel equivalence is the
frozen kernel rule — ``str.lower()`` only, whitespace preserved — and an
owner-mismatch capture refusal preserves exactly a hold whose identity
equals the selected acquisition frame's four-part identity.
"""

from __future__ import annotations

import ast
import os
import re
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets
from xrd_tools.io import Browse1DCache, FrameScalarCatalog, FrameScalarRow

from xdart.gui.tabs.scattering.browse_values import (
    BrowseLoadRequest,
    BrowseLoadStatus,
)
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.context_runtime import _ContextRuntime
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xdart.gui.tabs.scattering.display_values import (
    DisplayFrameKey,
    StandardDisplayPayload,
)
from xdart.gui.tabs.scattering.events import (
    ExecutorAccepted,
    PreflightAccepted,
    RunIdentity,
)
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xdart.gui.tabs.scattering.shell_values import (
    FrameNavigationProjection,
    ProgressProjection,
)
from xdart.gui.tabs.scattering.shell_widgets import aggregate_traces
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.modules.display_context import (
    AcquisitionContext,
    BrowseContext,
    ContextKind,
    DisplaySelection,
    new_context_token,
)
from xdart.modules.frame_publication import (
    FramePublication,
    PublicationStore,
    canonical_frame_source_identity,
)
from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.core.metadata import resolve_monitor_norm
from xrd_tools.io.output_path import READABLE_OUTPUT_SUFFIXES
from xrd_tools.session.display_logic import nanmean_slice
from xrd_tools.session.frame_record_store import FrameRecordStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.session.scan_norm import (
    channel_is_partial,
    empty_norm_aggregate,
    fold_norm_metadata,
    next_norm_revision,
)
from xrd_tools.sources.selection import image_series_spec

from tests.xdart.scattering.e3_shell_support import make_shell_projection

_SCATTERING = (
    Path(__file__).parents[3] / "src" / "xdart" / "gui" / "tabs" / "scattering"
)
PRODUCTION = {
    "context_runtime": _SCATTERING / "context_runtime.py",
    "context_controller": _SCATTERING / "context_controller.py",
    "page": _SCATTERING / "page.py",
    "context_projection": _SCATTERING / "context_projection.py",
    "shell_projection": _SCATTERING / "shell_projection.py",
    "scientific_axes": _SCATTERING / "scientific_axes.py",
    "shell_values": _SCATTERING / "shell_values.py",
    "scientific_view": _SCATTERING / "scientific_view.py",
}
PLACEHOLDER = "Norm Channel"
_DECOY = {"raw_only_monitor": 999.0}
ARTIFACT = "/out/a.nxs"
SCAN = "run.a"


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _configuration(*, gi: bool = False):
    values = {
        "source_spec": image_series_spec(Path("/data/a_0001.tif")),
        "poni_file": "/data/a.poni",
        "save_path": ARTIFACT,
        "output_mode": "Overwrite",
    }
    if gi:
        values["gi"] = {
            "enabled": True,
            "incidence_motor": "samth",
            "mode_1d": "q_total",
            "mode_2d": "q_chi",
        }
    return RunIntent(**values).freeze()


def _view(
    label: int,
    value: float,
    metadata: dict,
    *,
    one_d: bool = True,
) -> FrameView:
    raw = np.full((2, 3), value)
    return FrameView(
        label,
        axis_1d=(
            Axis("q", "A^-1", values=np.array([0.0, 1.0])) if one_d else None
        ),
        intensity_1d=(
            np.array([value, value + 1.0]) if one_d else None
        ),
        axis_2d_x=Axis("q", "A^-1", values=np.arange(3.0)),
        axis_2d_y=Axis("chi", "deg", values=np.arange(2.0)),
        intensity_2d=raw + 10.0,
        raw=raw,
        thumbnail=raw,
        source_path=f"/data/scan_{label}.tif",
        source_frame_index=label,
        metadata_numeric=dict(metadata),
        metadata_raw=dict(_DECOY),
    )


def _retain(display, owner, scan_key, artifact, label, value, metadata):
    view = _view(label, value, metadata)
    record = FrameRecord.from_view(view)
    publication = FramePublication(
        view,
        record=record,
        source_identity=f"{view.source_path}#{label}",
        scan_key=scan_key,
    )
    delta = display.append_navigation(scan_key, artifact, label)
    display.retain_frame(
        owner,
        delta.appended,
        record,
        publication,
        source_identity=publication.source_identity,
        frame_mask_qualified=False,
    )
    display.publish_light_1d(owner, record, source_identity=publication.source_identity)
    display.put_payload(
        StandardDisplayPayload(
            0, delta.appended, f"Standard · {scan_key} · {label}", view
        )
    )
    return delta.appended, delta


def _light_1d_layout(record: FrameRecord):
    """One light-1D layout shaped exactly by a real multi-mode record."""
    from xrd_tools.session import (
        Light1DBufferLayout, Light1DLayout, Light1DModeLayout,
    )

    f64 = np.dtype(np.float64).str
    modes = []
    for mode, view in record.results_1d.items():
        modes.append(Light1DModeLayout(
            mode,
            Light1DBufferLayout(len(view.axis_1d.values), 8, f"{mode}-axis", f64),
            Light1DBufferLayout(
                len(view.intensity_1d), 8, f"{mode}-intensity", f64,
            ),
            None if view.sigma_1d is None else Light1DBufferLayout(
                len(view.sigma_1d), 8, f"{mode}-sigma", f64,
            ),
        ))
    return Light1DLayout(tuple(modes), record.active_mode_1d)


def _add_artifact(
    display, artifact: str, scan_key: str, *, layout=None, gi: bool = False,
):
    from xrd_tools.core.frame_view import DEFAULT_MODE_KEY
    from xrd_tools.session import (
        Light1DBufferLayout, Light1DLayout, Light1DModeLayout,
        SessionResourceAuthority, SessionResourceRequirements,
        acquire_light_1d_retention, resolve_session_policy,
    )

    owner = display.add_artifact(
        Path(artifact),
        scan_key,
        mask=None,
        mask_saturation=True,
        measurement_mode="GI" if gi else "Standard",
        gi_incidence_motor="th" if gi else "",
        gi_resolved_motor="th" if gi else "",
        gi_mode_1d="q_total" if gi else "",
        gi_mode_2d="qip_qoop" if gi else "",
    )
    if layout is None:
        layout = Light1DLayout((Light1DModeLayout(
            DEFAULT_MODE_KEY,
            Light1DBufferLayout(2, 8, "norm-axis", np.dtype(np.float64).str),
            Light1DBufferLayout(2, 8, "norm-intensity", np.dtype(np.float64).str),
        ),), DEFAULT_MODE_KEY)
    npt_1d = max(mode.intensity.length for mode in layout.modes)
    row_capacity = 1024
    ceiling = layout.shared_bytes + row_capacity * layout.per_row_unique_ndarray_bytes
    allocation = resolve_session_policy(
        SessionResourceRequirements(
            2, 3, 8, modes_1d=len(layout.modes), modes_2d=1,
            npt_1d=npt_1d, npt_rad=3, npt_azim=2,
        ),
        envelope_bytes=2 << 30,
        requests={"record_heavy_items": 2, "publication_heavy_items": 2},
        env={},
    ).allocation
    owner.publications.bind_allocation(allocation)
    lease = acquire_light_1d_retention(
        SessionResourceAuthority.from_allocation(allocation),
        owner=f"norm-consumer:{artifact}", generation=1, layout=layout,
        requested_rows=row_capacity, compatibility_byte_ceiling=ceiling,
        gui_thread_id=threading.get_ident(),
    )
    owner.publications.bind_light_1d(lease)
    display.stage_light_1d(
        owner, lease, hooks=owner.publications.light_1d_cleanup_hooks(lease),
    )
    display.bind_light_1d(owner, lease)
    return owner


def _acquisition_context(configuration, display) -> AcquisitionContext:
    context = AcquisitionContext(
        context_token=new_context_token(ContextKind.ACQUISITION),
        run_configuration=configuration,
        config_generation=configuration.generation,
        config_fingerprint=configuration.fingerprint,
        run_scan_key=SCAN,
        source_path="/data/a_0001.tif",
        scan=object(),
        frame=None,
        frame_ids=display.catalog,
        frames=display.artifacts,
        viewer_rows_1d=(),
        viewer_rows_2d=(),
        publication_store=display,
        origin="scattering-standard",
        poni_identity=configuration.poni_file,
    )
    context.adopt_record_store(display)
    return context


def _acquisition_parts(
    rows,
    *,
    plot_mode: str = "Overlay",
    partitions=1,
    max_payload_items: int = 16,
):
    """Real display + runtime: retained rows, adopted, latest-all selected."""
    configuration = _configuration()
    identity = RunIdentity.from_configuration(configuration)
    display = RunDisplayState(
        identity,
        max_payload_items=max_payload_items,
    )
    display.set_factories(FrameRecordStore, PublicationStore)
    display.configure(partition_count=partitions, npt=2, frame_bytes=48)
    owner = _add_artifact(display, ARTIFACT, SCAN)
    keys = [
        _retain(display, owner, SCAN, ARTIFACT, label, value, metadata)[0]
        for label, value, metadata in rows
    ]
    context = _acquisition_context(configuration, display)
    runtime = _ContextRuntime()
    runtime.adopt_acquisition(identity, context)
    runtime.select_latest_navigation(plot_mode=plot_mode)
    if plot_mode == "Overlay":
        frames = runtime.navigation.frames
        assert runtime.select_navigation(frames[-1], frames)
    return SimpleNamespace(
        configuration=configuration,
        identity=identity,
        display=display,
        owner=owner,
        context=context,
        runtime=runtime,
        projection=ContextProjection(),
        keys=keys,
        expected_identity=(
            identity.generation,
            identity.fingerprint,
            ARTIFACT,
            SCAN,
        ),
    )


def _project(parts, prefs, *, live_update: bool = False):
    return parts.runtime.project_navigation(
        parts.projection,
        preferences=prefs,
        processing_mode="Int 2D",
        live_update=live_update,
    )


def _browse_parts(
    *,
    scan_key: str = "browse.b",
    path: str = "/processed/browse.b.nxs",
    generation: int = 3,
    rows=({"mon": 5.0},),
    values=(20.0,),
    aggregate_identity=None,
    with_aggregate: bool = True,
):
    token = new_context_token(ContextKind.BROWSE)
    request = BrowseLoadRequest(token, generation, path)
    catalog = FrameScalarCatalog(
        path, "entry", tuple(
            FrameScalarRow(label, metadata_raw=metadata)
            for label, metadata in enumerate(rows, 1)
        ),
    )
    records = FrameRecordStore(max_items=8)
    publications = PublicationStore(max_items=8)
    for label, (metadata, value) in enumerate(
        zip(rows, values, strict=True), 1
    ):
        view = _view(label, value, metadata)
        record = FrameRecord.from_view(view)
        records.upsert(
            record, source_identity=f"{path}#{label}", persisted=True
        )
        publications.upsert(
            FramePublication(
                view,
                record=record,
                source_identity=f"{path}#{label}",
                scan_key=scan_key,
            )
        )
    aggregate = None
    if with_aggregate:
        draft = empty_norm_aggregate(
            aggregate_identity or (token, scan_key, path)
        )
        for metadata in rows:
            draft = fold_norm_metadata(draft, metadata)
        aggregate = next_norm_revision(draft)
    context = BrowseContext(
        context_token=token,
        load_generation=generation,
        operation=request,
        requested_path=path,
        scan_key=scan_key,
        scan=object(),
        frame=None,
        frame_ids=catalog.labels,
        loaded_labels=catalog.labels,
        scalar_catalog=catalog,
        browse_1d_cache=Browse1DCache(budget_bytes=1024),
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=publications,
        record_store=records,
        norm_aggregate=aggregate,
    )
    context.adopt_load_request(request)
    context.mark_loaded()
    runtime = _ContextRuntime()
    runtime.adopt_browse(context, request)
    return SimpleNamespace(
        token=token,
        request=request,
        context=context,
        runtime=runtime,
        projection=ContextProjection(),
        aggregate=aggregate,
    )


def _release_browse_value(context):
    """Retire the scalar fixture's actual cache before its Browse payload."""
    context.invalidate()
    cache = context.browse_1d_cache
    cache.close()
    context.detach_browse_1d_cache(cache)
    context.release()


def _acq_identity(identity: RunIdentity, artifact: str, scan: str):
    return (identity.generation, identity.fingerprint, artifact, scan)


def _kernel_aggregate(identity_tuple, rows):
    draft = empty_norm_aggregate(identity_tuple)
    for metadata in rows:
        draft = fold_norm_metadata(draft, metadata)
    return next_norm_revision(draft)


def _fabricated(rows, *, identity=None, scan=SCAN, artifact=ARTIFACT,
                one_d=None):
    """Direct-build fixtures: frames, payloads, navigation, aggregate."""
    identity = identity or RunIdentity(7, "n2-fp")
    one_d = one_d or (True,) * len(rows)
    frames = tuple(
        DisplayFrameKey(identity, scan, artifact, label, ordinal)
        for ordinal, (label, _, _) in enumerate(rows, 1)
    )
    payloads = tuple(
        StandardDisplayPayload(
            0,
            frame,
            f"t {frame.local_frame_label}",
            _view(
                frame.local_frame_label,
                value,
                metadata,
                one_d=one_d[index],
            ),
        )
        for index, (frame, (_, value, metadata)) in enumerate(
            zip(frames, rows, strict=True)
        )
    )
    navigation = FrameNavigationProjection(frames, frames[-1], frames)
    aggregate = _kernel_aggregate(
        _acq_identity(identity, artifact, scan),
        [metadata for _, _, metadata in rows],
    )
    return SimpleNamespace(
        identity=identity,
        frames=frames,
        payloads=payloads,
        navigation=navigation,
        aggregate=aggregate,
    )


def _build(payloads, navigation, prefs, aggregate, *, mode="Int 2D"):
    return build_scientific_projection(
        tuple(payloads),
        navigation,
        frozenset(navigation.frames),
        prefs,
        "",
        processing_mode=mode,
        norm_aggregate=aggregate,
    )


def _prefs(**kw) -> ScientificPreferences:
    return ScientificPreferences(**kw)


def _trace_by_label(state):
    return {
        trace.frame.local_frame_label: trace.intensity
        for trace in state.traces
    }


def test_absent_and_revision_zero_aggregates_give_only_the_placeholder():
    parts = _fabricated([(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})])
    zero = empty_norm_aggregate(
        _acq_identity(parts.identity, ARTIFACT, SCAN)
    )
    for aggregate in (None, zero):
        state = _build(
            parts.payloads, parts.navigation, _prefs(norm_channel="mon"),
            aggregate,
        )
        assert state.norm_channels == (PLACEHOLDER,)
        assert state.norm_channel == PLACEHOLDER
        assert state.norm_identity is None
        assert state.norm_revision == 0
        values = _trace_by_label(state)
        np.testing.assert_array_equal(values[1], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(values[2], np.array([2.0, 3.0]))


def test_complete_channels_appear_and_partial_channels_do_not():
    parts = _fabricated(
        [(1, 1.0, {"mon": 2.0, "bstop": 3.0}), (2, 2.0, {"mon": 4.0})]
    )
    assert dict(parts.aggregate.channels) == {
        "mon": (6.0, 2),
        "bstop": (3.0, 1),
    }
    state = _build(
        parts.payloads, parts.navigation, _prefs(), parts.aggregate
    )
    assert state.norm_channels == (PLACEHOLDER, "mon")


def test_choices_selection_divisor_label_identity_and_revision_are_one_call():
    parts = _fabricated([(1, 1.0, {"MON": 2.0}), (2, 2.0, {"MON": 4.0})])
    state = _build(
        parts.payloads, parts.navigation, _prefs(norm_channel="Mon"),
        parts.aggregate,
    )
    assert state.norm_channels == (PLACEHOLDER, "mon")
    assert state.norm_channel == "mon"
    assert state.norm_identity == _acq_identity(
        parts.identity, ARTIFACT, SCAN
    )
    assert state.norm_revision == parts.aggregate.revision == 1
    values = _trace_by_label(state)
    np.testing.assert_allclose(values[1], np.array([1.0, 2.0]) / 2.0)
    np.testing.assert_allclose(values[2], np.array([2.0, 3.0]) / 4.0)


def test_producer_swap_after_capture_stays_on_r_until_next_refresh():
    parts = _acquisition_parts(
        [(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})]
    )
    prefs = _prefs(plot_mode="Overlay", norm_channel="mon")
    payloads = _project(parts, prefs)
    captured = parts.runtime.norm_aggregate
    assert captured is not None and captured.revision == 2
    _, delta = _retain(
        parts.display, parts.owner, SCAN, ARTIFACT, 3, 3.0, {"mon": 8.0}
    )
    assert parts.owner.norm_aggregate.revision == 3
    # The borrow is the runtime-held object: the racing producer advance is
    # NOT visible to this refresh's shell construction.
    assert parts.runtime.norm_aggregate is captured
    state = _build(
        payloads, parts.runtime.navigation, prefs,
        parts.runtime.norm_aggregate,
    )
    assert state.norm_revision == 2
    assert parts.runtime.accept_navigation(delta, plot_mode="Overlay")
    _project(parts, prefs)
    assert parts.runtime.norm_aggregate.revision == 3


def test_case_alias_selection_divides_native_and_cake_traces_per_payload():
    parts = _fabricated(
        [(1, 1.0, {"MON": 2.0}), (2, 2.0, {"Mon": 4.0})],
        one_d=(True, False),
    )
    state = _build(
        parts.payloads, parts.navigation, _prefs(norm_channel="mON"),
        parts.aggregate,
    )
    values = _trace_by_label(state)
    np.testing.assert_allclose(values[1], np.array([1.0, 2.0]) / 2.0)
    cake = parts.payloads[1].view.intensity_2d
    np.testing.assert_allclose(values[2], nanmean_slice(cake, 0) / 4.0)


@pytest.mark.parametrize(
    "bad",
    (
        pytest.param({}, id="missing"),
        pytest.param({"mon": None}, id="none"),
        pytest.param({"mon": "wat"}, id="nonnumeric"),
        pytest.param({"mon": float("nan")}, id="nan"),
        pytest.param({"mon": 0.0}, id="zero"),
        pytest.param({"mon": -3.0}, id="negative"),
    ),
)
def test_complete_channel_with_bad_payload_divisor_fails_closed(bad) -> None:
    # §27.3 completion: the accepted aggregate says ``mon`` is complete,
    # but THIS admitted payload's own row cannot divide.  The effective
    # ``mon`` selection stays, and that one trace fails CLOSED — never a
    # substituted divisor 1.0, never an unnormalized trace under an
    # ``I / mon`` label.
    parts = _fabricated([(1, 1.0, {"mon": 2.0}), (2, 2.0, bad)])
    complete = _kernel_aggregate(
        _acq_identity(parts.identity, ARTIFACT, SCAN),
        [{"mon": 2.0}, {"mon": 4.0}],
    )
    state = _build(
        parts.payloads, parts.navigation, _prefs(norm_channel="mon"),
        complete,
    )
    assert state.norm_channel == "mon"
    assert state.norm_revision == 1
    values = _trace_by_label(state)
    assert set(values) == {1}
    np.testing.assert_allclose(values[1], np.array([1.0, 2.0]) / 2.0)


def test_unknown_and_partial_selection_are_the_global_unnormalized_fallback():
    parts = _fabricated(
        [(1, 1.0, {"mon": 2.0, "bstop": 3.0}), (2, 2.0, {"mon": 4.0})]
    )
    for saved in ("wat", "bstop"):
        state = _build(
            parts.payloads, parts.navigation, _prefs(norm_channel=saved),
            parts.aggregate,
        )
        assert state.norm_channel == PLACEHOLDER
        values = _trace_by_label(state)
        np.testing.assert_array_equal(values[1], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(values[2], np.array([2.0, 3.0]))


def test_reserved_sentinel_channel_is_neither_offered_nor_effective():
    # §27.2 blocker 2 (§25.3 amended): a complete metadata channel that
    # canonicalizes to ``norm channel`` collides with the display
    # placeholder.  It is reserved case-insensitively — never offered,
    # never effective — and DEFAULT preferences must not silently divide.
    # With the colliding channel ALONE, choices are EXACTLY the
    # placeholder: no duplicate-looking entry survives in any form.
    alone = _fabricated(
        [(1, 1.0, {"Norm Channel": 2.0}), (2, 2.0, {"norm channel": 4.0})]
    )
    parts = _fabricated(
        [(1, 1.0, {"Norm Channel": 2.0, "mon": 3.0}),
         (2, 2.0, {"norm channel": 4.0, "mon": 6.0})]
    )
    for prefs in (_prefs(), _prefs(norm_channel="NORM CHANNEL")):
        state = _build(
            alone.payloads, alone.navigation, prefs, alone.aggregate
        )
        assert state.norm_channels == (PLACEHOLDER,)
        assert state.norm_channel == PLACEHOLDER
        values = _trace_by_label(state)
        np.testing.assert_array_equal(values[1], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(values[2], np.array([2.0, 3.0]))
        state = _build(
            parts.payloads, parts.navigation, prefs, parts.aggregate
        )
        assert state.norm_channels == (PLACEHOLDER, "mon")
        assert state.norm_channel == PLACEHOLDER
        values = _trace_by_label(state)
        np.testing.assert_array_equal(values[1], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(values[2], np.array([2.0, 3.0]))
    # The reservation is surgical: a genuine channel beside the colliding
    # key still divides each payload by its own row.
    state = _build(
        parts.payloads, parts.navigation, _prefs(norm_channel="mon"),
        parts.aggregate,
    )
    assert all(not trace.intensity.flags.writeable for trace in state.traces)
    assert state.norm_channel == "mon"
    values = _trace_by_label(state)
    np.testing.assert_allclose(values[1], np.array([1.0, 2.0]) / 3.0)
    np.testing.assert_allclose(values[2], np.array([2.0, 3.0]) / 6.0)


def test_whitespace_channel_matches_by_kernel_lowercase_equivalence():
    # §28.2 blocker 2 (§25.3): Q2 and ``resolve_monitor_norm`` canonicalize
    # CASE only — whitespace is preserved.  The consumer matches the saved
    # selection by that same lowercase-only equivalence: the exact offered
    # key " mon " becomes effective, a case alias normalizes to the
    # canonical lowercase key, and every trace divides by its own
    # payload's row.
    parts = _fabricated(
        [(1, 1.0, {" mon ": 2.0}), (2, 2.0, {" MON ": 4.0})]
    )
    # The EXACT offered choice becomes effective and divides per payload.
    state = _build(
        parts.payloads, parts.navigation, _prefs(norm_channel=" mon "),
        parts.aggregate,
    )
    assert " mon " in state.norm_channels
    assert state.norm_channel == " mon "
    values = _trace_by_label(state)
    np.testing.assert_allclose(values[1], np.array([1.0, 2.0]) / 2.0)
    np.testing.assert_allclose(values[2], np.array([2.0, 3.0]) / 4.0)
    # A saved case alias resolves and normalizes to the canonical key.
    state = _build(
        parts.payloads, parts.navigation, _prefs(norm_channel=" MON "),
        parts.aggregate,
    )
    assert state.norm_channel == " mon "
    values = _trace_by_label(state)
    np.testing.assert_allclose(values[1], np.array([1.0, 2.0]) / 2.0)
    np.testing.assert_allclose(values[2], np.array([2.0, 3.0]) / 4.0)


def test_only_the_exact_lowercase_canonical_placeholder_is_reserved():
    # §28.2 blocker 2 (§25.3): only a key whose lowercase-only canonical
    # spelling is exactly ``norm channel`` collides with the display
    # placeholder.  " norm channel " is whitespace-distinct under the
    # frozen kernel equivalence — a genuine, offerable, effective channel.
    parts = _fabricated(
        [(1, 1.0, {" norm channel ": 4.0, "mon": 3.0}),
         (2, 2.0, {" norm channel ": 8.0, "mon": 6.0})]
    )
    state = _build(
        parts.payloads, parts.navigation,
        _prefs(norm_channel=" norm channel "), parts.aggregate,
    )
    assert state.norm_channels == (PLACEHOLDER, " norm channel ", "mon")
    assert state.norm_channel == " norm channel "
    values = _trace_by_label(state)
    np.testing.assert_allclose(values[1], np.array([1.0, 2.0]) / 4.0)
    np.testing.assert_allclose(values[2], np.array([2.0, 3.0]) / 8.0)
    # Exact case aliases of the placeholder remain reserved.
    state = _build(
        parts.payloads, parts.navigation,
        _prefs(norm_channel="NORM CHANNEL"), parts.aggregate,
    )
    assert state.norm_channel == PLACEHOLDER
    np.testing.assert_array_equal(
        _trace_by_label(state)[1], np.array([1.0, 2.0])
    )


@pytest.mark.parametrize(
    "refusal",
    (
        "wrong_token",
        "wrong_scan_key",
        "wrong_requested_path",
        "invalidated",
        "released",
        "cancelled_gate",
        "stale_selection",
    ),
)
def test_browse_refusals_are_independent_and_cannot_change_presentation(
    refusal: str,
) -> None:
    prefs = _prefs(norm_channel="mon")
    if refusal in {"wrong_token", "wrong_scan_key", "wrong_requested_path"}:
        # A doctored aggregate wrong in exactly ONE identity element is
        # never admitted: the presentation stays the placeholder.
        element = {
            "wrong_token": ("browse-foreign", "browse.b",
                            "/processed/browse.b.nxs"),
            "wrong_scan_key": (None, "browse.z",
                               "/processed/browse.b.nxs"),
            "wrong_requested_path": (None, "browse.b",
                                     "/processed/OTHER.nxs"),
        }[refusal]
        parts = _browse_parts()
        wrong = (
            element[0] or parts.token,
            element[1],
            element[2],
        )
        object.__setattr__(
            parts.context, "norm_aggregate",
            _kernel_aggregate(wrong, [{"mon": 5.0}]),
        )
        _project(parts, prefs)
        assert parts.runtime.norm_aggregate is None
        state = _build(
            (), parts.runtime.navigation, prefs,
            parts.runtime.norm_aggregate,
        )
        assert state.norm_channel == PLACEHOLDER
        return
    parts = _browse_parts()
    _project(parts, prefs)
    admitted = parts.runtime.norm_aggregate
    assert admitted is parts.aggregate
    # A newer revision now sits behind the context: only a live, exactly
    # named scope may admit it, so a refusal is observable as "still rev 1".
    newer = next_norm_revision(
        fold_norm_metadata(parts.aggregate, {"mon": 7.0})
    )
    object.__setattr__(parts.context, "norm_aggregate", newer)
    if refusal == "invalidated":
        parts.context.invalidate()
    elif refusal == "released":
        _release_browse_value(parts.context)
    elif refusal == "cancelled_gate":
        parts.context.commit_gate.cancel()
    else:
        foreign = _browse_parts(scan_key="browse.z",
                                path="/processed/z.nxs")
        parts.runtime._selection = DisplaySelection.for_context(
            foreign.context, 99
        )
    # A dead or stale scope is a capture no-op: the refusal cannot change
    # what is presented, in either direction.
    _project(parts, prefs)
    assert parts.runtime.norm_aggregate is admitted


@pytest.mark.parametrize(
    "refusal",
    ("invalidated", "released", "cancelled_gate", "stale_selection"),
)
def test_refused_new_browse_context_cannot_expose_the_prior_token(
    refusal: str,
) -> None:
    # §27.2 blocker 1: ``adopt_browse`` installs a NEW token over the same
    # scan/path.  If the new context is refused before its FIRST capture,
    # the held prior-token aggregate is foreign to the currently owned
    # context and must clear — Browse frame coverage compares only
    # scan/path, so a stale hold would divide the new context's frames.
    prefs = _prefs(norm_channel="mon")
    b1 = _browse_parts()
    _project(b1, prefs)
    assert b1.runtime.norm_aggregate is b1.aggregate
    b2 = _browse_parts(with_aggregate=False)
    b1.runtime.adopt_browse(b2.context, b2.request)
    if refusal == "invalidated":
        b2.context.invalidate()
    elif refusal == "released":
        _release_browse_value(b2.context)
    elif refusal == "cancelled_gate":
        b2.context.commit_gate.cancel()
    else:
        third = _browse_parts(
            scan_key="browse.z", path="/processed/z.nxs",
            with_aggregate=False,
        )
        b1.runtime._selection = DisplaySelection.for_context(
            third.context, 99
        )
    payloads = _project(b1, prefs)
    assert b1.runtime.norm_aggregate is None
    state = _build(
        payloads, b1.runtime.navigation, prefs,
        b1.runtime.norm_aggregate,
    )
    assert state.norm_channel == PLACEHOLDER


def test_pending_replacement_retains_the_outgoing_presentation():
    # §25.3, preserved by correction 1: during _PendingBrowseReplacement
    # the OUTGOING presentation — navigation AND the held aggregate — is
    # deliberately retained.  Capture is a strict no-op there, never a
    # clear.
    prefs = _prefs(norm_channel="mon")
    b1 = _browse_parts()
    _project(b1, prefs)
    assert b1.runtime.norm_aggregate is b1.aggregate
    _release_browse_value(b1.context)
    replacement = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE), 4, "/processed/next.nxs"
    )
    b1.runtime.begin_replacement(replacement)
    _project(b1, prefs)
    assert b1.runtime.norm_aggregate is b1.aggregate
    state = _build(
        (), b1.runtime.navigation, prefs, b1.runtime.norm_aggregate
    )
    assert state.norm_channel == "mon"


def test_acquisition_rescope_refusal_cannot_expose_a_browse_hold():
    # §28.2 blocker 1: the page synchronizes acquisition scope, then the
    # worker may ``rescope_to`` BEFORE this refresh captures.  The stale
    # selection owner is a capture refusal — but a held BROWSE aggregate
    # for the same scan/path is foreign to the selected acquisition
    # frame's four-part identity and must clear: Browse frame coverage
    # compares only scan/path, so the prior token would divide the
    # acquisition frames even while payload projection fails closed.
    prefs = _prefs(norm_channel="mon")
    acquisition = _acquisition_parts(
        [(1, 10.0, {"mon": 2.0}), (2, 20.0, {"mon": 4.0})]
    )
    browse = _browse_parts(
        scan_key=SCAN, path=ARTIFACT,
        rows=({"mon": 2.0}, {"mon": 4.0}), values=(10.0, 20.0),
    )
    acquisition.runtime.adopt_browse(browse.context, browse.request)
    acquisition.runtime.select_latest_navigation(plot_mode="Overlay")
    _project(acquisition, prefs)
    assert acquisition.runtime.norm_aggregate is browse.aggregate
    acquisition.runtime.select_acquisition()
    acquisition.context.rescope_to("run.next", "/data/next.tif", object())
    payloads = _project(acquisition, prefs)
    # Payload projection fails closed on the stale owner; normalization
    # fails closed WITH it instead of retaining the foreign token.
    assert payloads == ()
    assert acquisition.runtime.norm_aggregate is None
    state = _build(
        payloads, acquisition.runtime.navigation, prefs,
        acquisition.runtime.norm_aggregate,
    )
    assert state.norm_identity is None
    assert state.norm_channel == PLACEHOLDER
    assert state.norm_channels == (PLACEHOLDER,)
    assert state.traces == ()


def test_acquisition_rescope_refusal_keeps_the_owned_acquisition_hold():
    # The §28 clear is surgical, never blind: an aggregate held FOR the
    # still-selected old acquisition identity survives the same
    # owner-mismatch refusal untouched, exactly like a same-context
    # Browse refusal (§27.2).
    prefs = _prefs(norm_channel="mon")
    parts = _acquisition_parts(
        [(1, 10.0, {"mon": 2.0}), (2, 20.0, {"mon": 4.0})]
    )
    _project(parts, prefs)
    held = parts.runtime.norm_aggregate
    assert held is not None
    assert held.identity == parts.expected_identity
    parts.context.rescope_to("run.next", "/data/next.tif", object())
    payloads = _project(parts, prefs)
    assert payloads == ()
    assert parts.runtime.norm_aggregate is held
    state = _build(
        payloads, parts.runtime.navigation, prefs,
        parts.runtime.norm_aggregate,
    )
    assert state.norm_channel == "mon"


def test_same_channel_foreign_artifact_cannot_cross_and_span_falls_back():
    parts = _acquisition_parts([(1, 1.0, {"mon": 2.0})], partitions=2)
    owner_b = _add_artifact(parts.display, "/out/b.nxs", "run.b")
    _, delta_b = _retain(
        parts.display, parts.owner, SCAN, ARTIFACT, 2, 2.0, {"mon": 4.0}
    )
    view_b = _view(9, 5.0, {"mon": 8.0})
    record_b = FrameRecord.from_view(view_b)
    publication_b = FramePublication(
        view_b,
        record=record_b,
        source_identity="/data/scan_9.tif#9",
        scan_key="run.b",
    )
    delta_foreign = parts.display.append_navigation("run.b", "/out/b.nxs", 9)
    parts.display.retain_frame(
        owner_b,
        delta_foreign.appended,
        record_b,
        publication_b,
        source_identity=publication_b.source_identity,
        frame_mask_qualified=False,
    )
    parts.display.publish_light_1d(
        owner_b, record_b, source_identity=publication_b.source_identity,
    )
    parts.display.put_payload(
        StandardDisplayPayload(
            0, delta_foreign.appended, "Standard · run.b · 9", view_b
        )
    )
    assert parts.runtime.accept_navigation(delta_b, plot_mode="Overlay")
    assert parts.runtime.accept_navigation(
        delta_foreign, plot_mode="Overlay"
    )
    frames = parts.runtime.navigation.frames
    prefs = _prefs(plot_mode="Overlay", norm_channel="mon")
    # Same-identity selection with the foreign frame as the CURRENT anchor:
    # the captured aggregate is the current frame's, the selected set is
    # foreign to it, and the same channel key buys no division.
    assert parts.runtime.select_navigation(frames[2], frames[:2])
    payloads = _project(parts, prefs)
    captured = parts.runtime.norm_aggregate
    assert captured is not None
    assert captured.identity == (
        parts.identity.generation,
        parts.identity.fingerprint,
        "/out/b.nxs",
        "run.b",
    )
    state = _build(payloads, parts.runtime.navigation, prefs, captured)
    assert state.norm_channel == PLACEHOLDER
    values = _trace_by_label(state)
    np.testing.assert_array_equal(values[1], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(values[2], np.array([2.0, 3.0]))
    # A multi-artifact selected set is the same global fallback.
    assert parts.runtime.select_navigation(frames[2], frames)
    payloads = _project(parts, prefs)
    state = _build(
        payloads, parts.runtime.navigation, prefs,
        parts.runtime.norm_aggregate,
    )
    assert state.norm_channel == PLACEHOLDER
    values = _trace_by_label(state)
    np.testing.assert_array_equal(values[1], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(values[2], np.array([2.0, 3.0]))
    np.testing.assert_array_equal(values[9], np.array([5.0, 6.0]))
    # An exact same-identity selection normalizes by each payload's OWN row.
    assert parts.runtime.select_navigation(frames[1], frames[:2])
    payloads = _project(parts, prefs)
    state = _build(
        payloads, parts.runtime.navigation, prefs,
        parts.runtime.norm_aggregate,
    )
    assert state.norm_channel == "mon"
    values = _trace_by_label(state)
    np.testing.assert_allclose(values[1], np.array([1.0, 2.0]) / 2.0)
    np.testing.assert_allclose(values[2], np.array([2.0, 3.0]) / 4.0)


def test_new_revision_at_stable_prefix_projects_only_the_appended_suffix():
    parts = _acquisition_parts(
        [(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})]
    )
    prefs = _prefs(plot_mode="Overlay", norm_channel="mon")
    payloads = _project(parts, prefs)
    assert parts.runtime.commit_navigation_projection(
        tuple(payload.frame_key for payload in payloads)
    )
    _, delta = _retain(
        parts.display, parts.owner, SCAN, ARTIFACT, 3, 3.0, {"mon": 8.0}
    )
    assert parts.runtime.accept_navigation(delta, plot_mode="Overlay")
    frames = parts.runtime.navigation.frames
    assert parts.runtime.select_navigation(frames[-1], frames)
    payloads = _project(parts, prefs)
    # Each trace divides by that frame's own immutable metadata value.  A
    # newer aggregate revision therefore changes provenance/choices, not the
    # already-rendered prefix's numeric regime.
    assert [
        payload.frame_key.local_frame_label for payload in payloads
    ] == [3]
    assert parts.runtime.norm_aggregate.revision == 3
    assert parts.runtime.commit_navigation_projection(
        parts.runtime.navigation.selected
    )
    replay = _project(parts, prefs)
    assert [
        payload.frame_key.local_frame_label for payload in replay
    ] == [3]


def test_revision_growth_keeps_trace_projection_work_linear():
    """A retained prefix is projected once, not once per aggregate revision."""

    frame_count = 651
    parts = _acquisition_parts(
        [(1, 1.0, {"mon": 2.0})],
        max_payload_items=frame_count + 1,
    )
    prefs = _prefs(plot_mode="Overlay", norm_channel="mon")
    projected = _project(parts, prefs)
    projection_count = len(projected)
    assert parts.runtime.commit_navigation_projection(
        parts.runtime.navigation.selected
    )

    for label in range(2, frame_count + 1):
        _, delta = _retain(
            parts.display,
            parts.owner,
            SCAN,
            ARTIFACT,
            label,
            float(label),
            {"mon": float(label + 1)},
        )
        assert parts.runtime.accept_navigation(delta, plot_mode="Overlay")
        frames = parts.runtime.navigation.frames
        assert parts.runtime.select_navigation(frames[-1], frames)
        projected = _project(parts, prefs, live_update=True)
        projection_count += len(projected)
        assert [
            payload.frame_key.local_frame_label for payload in projected
        ] == [label]
        assert parts.runtime.commit_navigation_projection(
            parts.runtime.navigation.selected
        )

    assert projection_count == frame_count


def test_channel_change_at_stable_prefix_reseeds_the_runtime_ledger():
    parts = _acquisition_parts(
        [(1, 1.0, {"mon": 2.0, "i0": 3.0}), (2, 2.0, {"mon": 4.0, "i0": 6.0})]
    )
    payloads = _project(parts, _prefs(plot_mode="Overlay", norm_channel="mon"))
    assert parts.runtime.commit_navigation_projection(
        tuple(payload.frame_key for payload in payloads)
    )
    changed = _project(
        parts, _prefs(plot_mode="Overlay", norm_channel="i0")
    )
    assert {
        payload.frame_key.local_frame_label for payload in changed
    } == {1, 2}


def test_equal_replay_does_not_churn_the_runtime_scope():
    parts = _acquisition_parts(
        [(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})]
    )
    prefs = _prefs(plot_mode="Overlay", norm_channel="mon")
    payloads = _project(parts, prefs)
    captured = parts.runtime.norm_aggregate
    assert parts.runtime.commit_navigation_projection(
        tuple(payload.frame_key for payload in payloads)
    )
    replay = _project(parts, prefs)
    assert parts.runtime.norm_aggregate is captured
    assert [
        payload.frame_key.local_frame_label for payload in replay
    ] == [2]


def test_sum_and_average_operate_on_per_frame_normalized_traces():
    parts = _fabricated([(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})])
    state = _build(
        parts.payloads, parts.navigation, _prefs(norm_channel="mon"),
        parts.aggregate,
    )
    per_frame = [
        np.array([1.0, 2.0]) / 2.0,
        np.array([2.0, 3.0]) / 4.0,
    ]
    summed = aggregate_traces(state.traces, "Sum")[0].intensity
    averaged = aggregate_traces(state.traces, "Average")[0].intensity
    np.testing.assert_allclose(summed, per_frame[0] + per_frame[1])
    np.testing.assert_allclose(
        averaged, (per_frame[0] + per_frame[1]) / 2.0
    )
    # A post-aggregation divisor is a DIFFERENT number: the mutation that
    # normalizes after Sum/Average cannot reproduce the per-frame result.
    assert not np.allclose(
        summed, (np.array([1.0, 2.0]) + np.array([2.0, 3.0])) / 2.0
    )


def test_standard_gi_browse_and_reload_agree_on_normalized_traces():
    rows = [(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})]
    standard = _fabricated(rows)
    gi = _fabricated(rows, identity=RunIdentity(8, "n2-gi"))
    gi_payloads = tuple(
        replace(
            payload,
            measurement_mode="GI",
            gi_incidence_motor="samth",
            gi_resolved_motor="samth",
            gi_mode_1d="q_total",
            gi_mode_2d="q_chi",
        )
        for payload in gi.payloads
    )
    browse_identity = RunIdentity(3, "browse-token-one")
    browse = _fabricated(
        rows,
        identity=browse_identity,
        scan="browse.b",
        artifact="/processed/browse.b.nxs",
    )
    browse_aggregate = _kernel_aggregate(
        ("browse-token-one", "browse.b", "/processed/browse.b.nxs"),
        [metadata for _, _, metadata in rows],
    )
    reload_identity = RunIdentity(4, "browse-token-two")
    reloaded = _fabricated(
        rows,
        identity=reload_identity,
        scan="browse.b",
        artifact="/processed/browse.b.nxs",
    )
    reload_aggregate = _kernel_aggregate(
        ("browse-token-two", "browse.b", "/processed/browse.b.nxs"),
        [metadata for _, _, metadata in rows],
    )
    prefs = _prefs(norm_channel="mon")
    states = [
        _build(standard.payloads, standard.navigation, prefs,
               standard.aggregate),
        _build(gi_payloads, gi.navigation, prefs, gi.aggregate),
        _build(browse.payloads, browse.navigation, prefs, browse_aggregate),
        _build(reloaded.payloads, reloaded.navigation, prefs,
               reload_aggregate),
    ]
    for state in states:
        assert state.norm_channel == "mon"
    for label in (1, 2):
        reference = _trace_by_label(states[0])[label]
        for state in states[1:]:
            np.testing.assert_allclose(
                _trace_by_label(state)[label], reference
            )


def test_revision_zero_older_and_foreign_candidates_are_refused():
    parts = _acquisition_parts(
        [(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})]
    )
    prefs = _prefs(plot_mode="Overlay", norm_channel="mon")
    _project(parts, prefs)
    held = parts.runtime.norm_aggregate
    assert held is not None and held.revision == 2
    produced = parts.owner.norm_aggregate
    expected = parts.expected_identity
    older = next_norm_revision(
        fold_norm_metadata(empty_norm_aggregate(expected), {"mon": 99.0})
    )
    foreign = _kernel_aggregate(
        (parts.identity.generation, parts.identity.fingerprint,
         "/out/OTHER.nxs", SCAN),
        [{"mon": 7.0}],
    )
    for doctored in (
        empty_norm_aggregate(expected),
        older,
        foreign,
    ):
        parts.owner.norm_aggregate = doctored
        _project(parts, prefs)
        assert parts.runtime.norm_aggregate is held
    parts.owner.norm_aggregate = produced
    # A fresh runtime over a revision-0 producer captures nothing at all.
    parts.owner.norm_aggregate = empty_norm_aggregate(expected)
    fresh = _ContextRuntime()
    fresh.adopt_acquisition(parts.identity, parts.context)
    fresh.select_latest_navigation(plot_mode="Overlay")
    fresh.project_navigation(
        parts.projection, preferences=prefs, processing_mode="Int 2D"
    )
    assert fresh.norm_aggregate is None
    parts.owner.norm_aggregate = produced


def test_context_switch_never_exposes_the_prior_identity_aggregate():
    parts = _acquisition_parts([(1, 1.0, {"mon": 2.0})])
    prefs = _prefs(norm_channel="mon")
    _project(parts, prefs)
    assert parts.runtime.norm_aggregate is not None
    browse = _browse_parts(with_aggregate=False)
    parts.runtime._browse = browse.context
    parts.runtime._selection = DisplaySelection.for_context(
        browse.context, 5
    )
    _project(parts, prefs)
    assert parts.runtime.norm_aggregate is None


def test_runtime_delta_scope_uses_identity_and_channel_not_revision():
    parts = _acquisition_parts(
        [(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})]
    )
    prefs = _prefs(plot_mode="Overlay", norm_channel="Mon")
    _project(parts, prefs)
    scope = parts.runtime._trace_projection_scope(prefs, "Int 2D")
    assert parts.runtime.norm_aggregate.revision == 2
    assert scope[-2:] == (parts.expected_identity, "mon")
    placeholder = _prefs(plot_mode="Overlay", norm_channel="wat")
    scope = parts.runtime._trace_projection_scope(placeholder, "Int 2D")
    assert scope[-2:] == (parts.expected_identity, "")


@pytest.mark.parametrize("changed", ("identity", "channel"))
def test_view_history_scope_reseeds_on_identity_and_channel(
    qapp: QtWidgets.QApplication,
    changed: str,
) -> None:
    parts = _fabricated([(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})])
    first = _build(
        parts.payloads, parts.navigation, _prefs(
            plot_mode="Overlay", norm_channel="mon"
        ),
        parts.aggregate,
    )
    assert len(first.traces) == 2
    delta_trace = tuple(
        trace for trace in first.traces
        if trace.frame is parts.frames[1]
    )
    doctored = {
        "identity": {"norm_identity": ("doctored", "identity", "tuple")},
        "channel": {"norm_channel": "i0"},
    }[changed]
    second = replace(first, traces=delta_trace, **doctored)
    view = ScientificView()
    try:
        view.reconcile(
            first, parts.navigation, completed=0, total=0, detail=""
        )
        assert view.trace_history_keys == parts.frames
        # The delta names only the second frame: a changed normalization
        # fact must clear detached history rather than mix regimes.
        view.reconcile(
            second, parts.navigation, completed=0, total=0, detail=""
        )
        assert view.trace_history_keys == (parts.frames[1],)
        # An equal replay retains the reseeded row without churn.
        view.reconcile(
            second, parts.navigation, completed=0, total=0, detail=""
        )
        assert view.trace_history_keys == (parts.frames[1],)
    finally:
        view.close()


def test_view_history_preserves_prefix_across_revision_only(
    qapp: QtWidgets.QApplication,
) -> None:
    parts = _fabricated(
        [(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})]
    )
    first = _build(
        parts.payloads,
        parts.navigation,
        _prefs(plot_mode="Overlay", norm_channel="mon"),
        parts.aggregate,
    )
    delta_trace = tuple(
        trace for trace in first.traces if trace.frame is parts.frames[1]
    )
    second = replace(
        first,
        traces=delta_trace,
        norm_revision=first.norm_revision + 1,
    )
    view = ScientificView()
    try:
        view.reconcile(
            first, parts.navigation, completed=0, total=0, detail=""
        )
        view.reconcile(
            second, parts.navigation, completed=0, total=0, detail=""
        )
        assert view.trace_history_keys == parts.frames
    finally:
        view.close()


def test_transport_borrows_the_exact_captured_object_into_build_shell():
    configuration = _configuration()
    lifecycle = ScatteringCoordinator()
    request = lifecycle.begin_start().request_id
    accepted = lifecycle.preflight_accepted(
        PreflightAccepted(request, configuration)
    )
    identity = accepted.run_identity
    assert identity is not None
    display = RunDisplayState(identity, max_payload_items=16)
    display.set_factories(FrameRecordStore, PublicationStore)
    display.configure(partition_count=1, npt=2, frame_bytes=48)
    owner = _add_artifact(display, ARTIFACT, SCAN)
    _retain(display, owner, SCAN, ARTIFACT, 1, 1.0, {"mon": 2.0})
    context = _acquisition_context(configuration, display)
    assert lifecycle.executor_accepted(ExecutorAccepted(identity))

    class _ExecutorPort:
        def acquisition_context(self, requested):
            return context if requested is identity else None

    from xdart.gui.tabs.scattering.context_controller import (
        ContextController,
    )

    controller = ContextController(
        lifecycle=lifecycle,
        executor=_ExecutorPort(),
        browse_loader=object(),
        projection=ContextProjection(),
    )
    controller.adopt_acquisition(identity)
    controller.select_latest_navigation(plot_mode="Overlay")
    prefs = _prefs(plot_mode="Overlay", norm_channel="mon")
    payloads = controller.project_navigation(
        preferences=prefs, processing_mode="Int 2D"
    )
    borrow = controller.norm_aggregate
    assert borrow is controller._runtime._norm_aggregate
    assert borrow is owner.norm_aggregate
    assert borrow.revision == 1
    # A producer advance AFTER the runtime capture: shell construction must
    # consume the exact borrowed snapshot, never a reread of the producer.
    _retain(display, owner, SCAN, ARTIFACT, 2, 2.0, {"mon": 4.0})
    assert owner.norm_aggregate.revision == 2
    support = make_shell_projection()
    shell = ContextProjection().build_shell(
        revision=1,
        controls=support.controls,
        controls_readiness=support.controls_readiness,
        phase=RunPhase.IDLE,
        intent=RunIntent(
            source_spec=image_series_spec(Path("/data/a_0001.tif")),
            poni_file="/data/a.poni",
            save_path=ARTIFACT,
            output_mode="Overwrite",
        ),
        contexts=controller.projectable_contexts,
        selection=controller.selection,
        navigation=controller.navigation,
        payloads=payloads,
        resident_frames=frozenset(controller.navigation.frames),
        progress=ProgressProjection(),
        preferences=prefs,
        browser_directory="",
        date_sorted=False,
        auto_last=True,
        executor_available=True,
        start_permitted=False,
        start_blocker="",
        notice="",
        norm_aggregate=borrow,
    )
    assert shell.scientific.norm_revision == borrow.revision == 1
    assert shell.scientific.norm_identity == borrow.identity
    assert shell.scientific.norm_channel == "mon"
    values = _trace_by_label(shell.scientific)
    np.testing.assert_allclose(values[1], np.array([1.0, 2.0]) / 2.0)


def test_real_shell_combo_traces_and_label_agree_without_main_mount(
    qapp: QtWidgets.QApplication,
) -> None:
    from xdart.gui.tabs.scattering.shell_values import (
        ScientificPlotOptions,
    )

    parts = _fabricated([(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})])
    scientific = _build(
        parts.payloads, parts.navigation, _prefs(
            plot_mode="Overlay",
            norm_channel="MON",
            plot_options=ScientificPlotOptions(overlay_offset=0.0),
        ),
        parts.aggregate,
    )
    base = make_shell_projection(plot_mode="Overlay")
    state = replace(
        base,
        scientific=scientific,
        navigation=parts.navigation,
        browser=replace(base.browser, frames=parts.frames),
    )
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(state)
        combo = shell.scientific.norm
        assert tuple(
            combo.itemText(index) for index in range(combo.count())
        ) == (PLACEHOLDER, "mon")
        assert combo.currentText() == "mon"
        label = shell.scientific.curve.getPlotItem().getAxis(
            "left"
        ).labelText
        assert "I / mon" in label
        items = shell.scientific.curve.listDataItems()
        assert len(items) == 2
        rendered = {
            tuple(round(float(value), 6) for value in item.yData)
            for item in items
        }
        assert rendered == {(0.5, 1.0), (0.5, 0.75)}
    finally:
        shell.close()


def test_census_one_capture_one_borrow_no_second_read_no_forbidden_routes():
    parts = _fabricated([(1, 1.0, {"mon": 2.0}), (2, 2.0, {"mon": 4.0})])
    before = [
        np.array(payload.view.intensity_1d) for payload in parts.payloads
    ]
    state = _build(
        parts.payloads, parts.navigation, _prefs(norm_channel="mon"),
        parts.aggregate,
    )
    for payload, original in zip(parts.payloads, before, strict=True):
        np.testing.assert_array_equal(
            payload.view.intensity_1d, original
        )
    for trace, payload in zip(state.traces, parts.payloads, strict=True):
        assert trace.intensity is not payload.view.intensity_1d

    sources = {
        name: path.read_text() for name, path in PRODUCTION.items()
    }
    trees = {
        name: ast.parse(source) for name, source in sources.items()
    }
    for name, source in sources.items():
        for token in (
            "pandas",
            "scan_data",
            "read_scan_data",
            "scan_aggregate",
            "DataFrame",
            "import h5py",
        ):
            assert token not in source, (name, token)
        assert "metadata_raw" not in source, name
        if name != "context_runtime":
            # Token-exact, not attribute-exact: a getattr-string reread
            # of the producer must not evade the census (the J1 lesson).
            assert "frame_norm_aggregate" not in source, name

    def _attribute_loads(tree, attribute):
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == attribute
            and isinstance(node.ctx, ast.Load)
        ]

    def _name_loads(tree, symbol):
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == symbol
            and isinstance(node.ctx, ast.Load)
        ]

    def _enclosing_functions(tree, targets):
        owners = []

        def visit(node, stack):
            for child in ast.iter_child_nodes(node):
                entered = stack
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    entered = stack + [child.name]
                if child in targets:
                    owners.append(entered[-1] if entered else None)
                visit(child, entered)

        visit(tree, [])
        return owners

    # Exactly ONE producer read per route, in the capture owner only.
    getter_calls = {
        name: _attribute_loads(tree, "frame_norm_aggregate")
        for name, tree in trees.items()
    }
    assert {
        name: len(nodes) for name, nodes in getter_calls.items()
    } == {name: (1 if name == "context_runtime" else 0)
          for name in PRODUCTION}
    assert _enclosing_functions(
        trees["context_runtime"], set(getter_calls["context_runtime"])
    ) == ["_capture_norm_aggregate"]
    borrow_counts = {
        name: len(_attribute_loads(tree, "norm_aggregate"))
        for name, tree in trees.items()
    }
    assert borrow_counts == {
        "context_runtime": 1,
        "context_controller": 1,
        "page": 3,
        "context_projection": 0,
        "shell_projection": 0,
        "scientific_axes": 0,
        "shell_values": 0,
        "scientific_view": 0,
    }
    assert _enclosing_functions(
        trees["page"], set(_attribute_loads(trees["page"], "norm_aggregate")),
    ) == ["_background_action", "_consume_background_update", "_refresh_shell"]
    capture_calls = [
        node
        for node in ast.walk(trees["context_runtime"])
        if isinstance(node, ast.Attribute)
        and node.attr == "_capture_norm_aggregate"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(capture_calls) == 1
    assert _enclosing_functions(
        trees["context_runtime"], set(capture_calls)
    ) == ["capture_norm_aggregate_for_refresh"]
    gate_calls = _name_loads(
        trees["context_runtime"], "accepts_norm_aggregate"
    )
    assert len(gate_calls) == 1
    assert _enclosing_functions(
        trees["context_runtime"], set(gate_calls)
    ) == ["_capture_norm_aggregate"]
    divisor_calls = {
        name: _name_loads(tree, "resolve_monitor_norm")
        for name, tree in trees.items()
    }
    assert {
        name: len(nodes) for name, nodes in divisor_calls.items()
    } == {name: (1 if name == "scientific_axes" else 0)
          for name in PRODUCTION}
    assert _enclosing_functions(
        trees["scientific_axes"], set(divisor_calls["scientific_axes"])
    ) == ["trace_projection"]
    resolver_calls = {
        name: len(_name_loads(tree, "resolve_norm_presentation"))
        for name, tree in trees.items()
    }
    assert resolver_calls["shell_projection"] == 1
    assert resolver_calls["context_runtime"] == 1
    assert resolver_calls["page"] == 2
    assert _enclosing_functions(
        trees["page"], set(_name_loads(trees["page"], "resolve_norm_presentation")),
    ) == ["_background_action", "_consume_background_update"]
    assert resolver_calls["context_projection"] == 0
    assert resolver_calls["scientific_view"] == 0

    # §27.3 hardening: import-alias, module-attribute and string-getattr
    # routes must not evade the one-resolver/one-divisor census.  Token
    # counts pin EVERY textual occurrence of the symbols — imports, calls,
    # strings — so an aliased or getattr-string second route changes the
    # count even when the AST name census cannot see it.
    def _token_count(source, symbol):
        return len(
            re.findall(
                rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])",
                source,
            )
        )

    assert {
        name: _token_count(source, "resolve_norm_presentation")
        for name, source in sources.items()
    } == {
        "context_runtime": 2,      # one import, one scope call
        "shell_projection": 2,     # one import, one projection call
        "scientific_axes": 2,      # the def and its __all__ export
        "context_controller": 0,
        "page": 3,                # import and two display-background borrows
        "context_projection": 0,
        "shell_values": 0,
        "scientific_view": 0,
    }
    assert {
        name: _token_count(source, "resolve_monitor_norm")
        for name, source in sources.items()
    } == {name: (2 if name == "scientific_axes" else 0)
          for name in PRODUCTION}
    assert {
        name: _token_count(source, "frame_norm_aggregate")
        for name, source in sources.items()
    } == {name: (1 if name == "context_runtime" else 0)
          for name in PRODUCTION}
    for symbol in ("resolve_norm_presentation", "resolve_monitor_norm"):
        for name, tree in trees.items():
            assert not _attribute_loads(tree, symbol), (name, symbol)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name.split(".")[-1] == symbol:
                            assert alias.asname is None, (name, symbol)


def _browse_load(path: Path, generation: int):
    """Load one real artifact through the production loader.

    Returns ``(context, request, loader)``; the caller owns all three and
    must settle them through :func:`_release_browse` (rejected candidates
    included).  The loader publishes an EMPTY record store plus its scalar
    catalog — intensity rows are hydrated on demand, never here.
    """
    from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader

    loader = BrowseLoader()
    request = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE), generation, str(path)
    )
    try:
        assert loader.begin(request) is request
        deadline = time.monotonic() + 120.0
        outcome = None
        while time.monotonic() < deadline:
            outcome = loader.poll(request)
            if outcome is not None:
                break
            time.sleep(0.05)
        assert outcome is not None and (
            outcome.status is BrowseLoadStatus.READY
        ), f"browse load did not become ready for {path}"
        context = loader.consume(outcome)
        assert type(context) is BrowseContext
    except BaseException:
        loader.close()
        raise
    return context, request, loader


def _release_browse(loader, context, owner=None) -> None:
    """Settle one real Browse ownership chain: owner, context, loader."""
    if owner is not None:
        receipt = owner.release(loader, context)
    else:
        receipt = loader.release_context(context)
    assert receipt.cleanup_status.value == "cleaned", receipt
    assert loader.close().cleanup_status.value == "cleaned"


def _independent_scalar_catalog(path: str) -> FrameScalarCatalog:
    """One independent array-free read of the artifact's scalar metadata."""
    from xrd_tools.io import FrameViewReader

    with FrameViewReader(path, resolve_source=False) as reader:
        return reader.read_scalar_catalog()


def _readable_output_candidates(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(
        (
            path
            for suffix in READABLE_OUTPUT_SUFFIXES
            for path in root.rglob(f"*{suffix}")
        ),
        key=lambda path: path.as_posix(),
    ))


@pytest.mark.skipif(
    not os.environ.get("XDART_TEST_DATA"),
    reason=(
        "XDART_TEST_DATA not set: real Browse/reload aggregate fold "
        "parity needs the real data root"
    ),
)
def test_real_browse_reload_aggregate_fold_parity():
    # Renamed by §27.3: this node loads Browse twice and compares kernel
    # aggregate FACTS against an independent fold.  It does NOT drive
    # Standard/GI consumer traces — that claim belongs to
    # test_real_standard_gi_browse_reload_consumer_trace_parity below.
    root = Path(os.environ["XDART_TEST_DATA"])
    candidates = _readable_output_candidates(root)
    assert candidates, f"no readable processed artifact under {root}"

    context = loader = None
    for candidate in candidates:
        try:
            context, _, loader = _browse_load(candidate, 1)
            break
        except AssertionError:
            continue
    assert context is not None, "no loadable real artifact"
    reloaded, _, reload_loader = _browse_load(Path(context.requested_path), 2)
    try:
        first = context.norm_aggregate
        second = reloaded.norm_aggregate
        assert first is not None and second is not None
        assert first.revision == second.revision == 1
        assert first.row_count == second.row_count
        assert dict(first.channels) == dict(second.channels)
        assert first.identity != second.identity
        # Browse hydrates intensity rows on demand: the record store is empty
        # after admission.  Fold an INDEPENDENT scalar-catalog read instead.
        assert len(context.record_store) == 0
        catalog = _independent_scalar_catalog(context.requested_path)
        expected = empty_norm_aggregate(first.identity)
        for label in context.frame_ids:
            row = catalog.row(int(label))
            assert row is not None, label
            expected = fold_norm_metadata(expected, row.metadata_numeric)
        expected = next_norm_revision(expected)
        assert first.row_count == expected.row_count
        assert dict(first.channels) == dict(expected.channels)
    finally:
        _release_browse(reload_loader, reloaded)
        _release_browse(loader, context)


@pytest.mark.skipif(
    not os.environ.get("XDART_TEST_DATA"),
    reason=(
        "XDART_TEST_DATA not set: real Standard/GI/Browse/reload consumer "
        "trace parity needs the real data root"
    ),
)
def test_real_standard_gi_browse_reload_consumer_trace_parity():
    # §27.3: drive the PRODUCTION N2 consumer — runtime capture through
    # the Q2 gate, the one shared resolution, per-payload division — over
    # ONE real processed artifact on all four routes (acquisition
    # Standard, acquisition GI, Browse, Browse reload) and compare every
    # normalized trace against an independent kernel division of the same
    # real records.
    from xdart.gui.tabs.scattering.browse_1d_display import (
        prepare_browse_1d_display,
    )
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_hydration import (
        _BrowseHydrationOwner,
    )
    from xrd_tools.io import FrameViewReader

    root = Path(os.environ["XDART_TEST_DATA"])
    candidates = _readable_output_candidates(root)
    assert candidates, f"no readable processed artifact under {root}"

    def _usable_channel(context):
        # Browse admits an EMPTY record store: eligibility is decided from
        # the loader's own array-free scalar catalog, never from residency.
        aggregate = context.norm_aggregate
        if aggregate is None or aggregate.revision != 1:
            return None
        if not 2 <= len(context.frame_ids) <= 16:
            return None
        catalog = context.scalar_catalog
        if type(catalog) is not FrameScalarCatalog:
            return None
        rows = [catalog.row(int(label)) for label in context.frame_ids]
        if any(row is None or not row.modes_1d for row in rows):
            return None
        for key in ("mon", *aggregate.channels):
            if key not in aggregate.channels:
                continue
            if channel_is_partial(aggregate, key):
                continue
            if all(
                resolve_monitor_norm(row.metadata_numeric, key)
                is not None
                for row in rows
            ):
                return key
        return None

    # Every admitted Browse object is settled: [loader, context, owner].
    owned: list[list] = []
    try:
        browse = browse_request = browse_loader = channel = None
        for candidate in candidates:
            try:
                context, request, loader = _browse_load(candidate, 1)
            except AssertionError:
                continue
            key = _usable_channel(context)
            if key is None:
                _release_browse(loader, context)
                continue
            browse, browse_request, browse_loader, channel = (
                context, request, loader, key,
            )
            owned.append([loader, context, None])
            break
        assert browse is not None and channel is not None, (
            "four-route real parity is UNVERIFIED: no loadable artifact with "
            "a complete, positively-resolvable channel and 2..16 frames"
        )
        # Independent 1-D reference views: complete 1-D modes plus metadata,
        # no detector/cake payloads (the sparse projection reads none).
        with FrameViewReader(
            browse.requested_path, resolve_source=False,
        ) as reader:
            views = {
                int(label): reader.read_record(
                    int(label), include_heavy=False,
                ).active_view()
                for label in browse.frame_ids
            }
        assert all(view.intensity_1d is not None for view in views.values())
        expected_traces = {}
        for label, view in views.items():
            divisor = resolve_monitor_norm(view.metadata_numeric, channel)
            expected_traces[label] = np.asarray(view.intensity_1d) / divisor
        prefs = _prefs(plot_mode="Overlay", norm_channel=channel.upper())

        def _browse_route(context, request, loader):
            runtime = _ContextRuntime()
            runtime.adopt_browse(context, request)
            frames = runtime.navigation.frames
            assert runtime.select_navigation(frames[-1], frames)
            owner = _BrowseHydrationOwner(context)
            for entry in owned:
                if entry[0] is loader:
                    entry[2] = owner
            # The production sparse 1-D path: plan → hydrate on demand →
            # borrow → detach.  INCOMPLETE submits the read; poll it.
            deadline = time.monotonic() + 60.0
            while True:
                projected = runtime.project_browse_1d_cache(
                    owner, preferences=prefs, was_waterfall_active=False,
                )
                if projected.status is not Browse1DProjectionStatus.INCOMPLETE:
                    break
                assert time.monotonic() < deadline, (
                    f"Browse 1-D hydration timed out: {projected.diagnostic}"
                )
                owner.consume_repaint()
                time.sleep(0.005)
            assert projected.status is Browse1DProjectionStatus.COMPLETE, (
                projected.diagnostic
            )
            captured = runtime.norm_aggregate
            assert captured is context.norm_aggregate
            detached = prepare_browse_1d_display(projected)
            assert detached is not None
            assert len(detached.payloads) == len(views)
            state = _build(
                detached.payloads, runtime.navigation, prefs, captured,
            )
            deadline = time.monotonic() + 10.0
            while owner.polling_needed() and time.monotonic() < deadline:
                owner.consume_repaint()
                time.sleep(0.005)
            assert not owner.polling_needed()
            return state

        def _acquisition_route(gi: bool):
            configuration = _configuration(gi=gi)
            identity = RunIdentity.from_configuration(configuration)
            display = RunDisplayState(identity, max_payload_items=16)
            display.set_factories(FrameRecordStore, PublicationStore)
            display.configure(
                partition_count=1, npt=1000, frame_bytes=8000
            )
            artifact = str(browse.requested_path)
            scan_key = browse.scan_key
            mode = "GI" if gi else "Standard"
            # Acquisition keeps its 1-D rows in the light tier (the retained
            # publication is stripped of them): lease it, shaped by the real
            # records, exactly as the fabricated rows above do.
            records = {
                label: FrameRecord.from_view(view)
                for label, view in sorted(views.items())
            }
            owner = _add_artifact(
                display, artifact, scan_key,
                layout=_light_1d_layout(next(iter(records.values()))),
                gi=gi,
            )
            for label, view in sorted(views.items()):
                record = records[label]
                publication = FramePublication(
                    view,
                    record=record,
                    source_identity=canonical_frame_source_identity(
                        view,
                        source_base=owner.source_base,
                        fallback_path=owner.artifact,
                    ),
                    scan_key=scan_key,
                )
                delta = display.append_navigation(scan_key, artifact, label)
                display.retain_frame(
                    owner,
                    delta.appended,
                    record,
                    publication,
                    source_identity=publication.source_identity,
                    frame_mask_qualified=False,
                )
                display.publish_light_1d(
                    owner, record, source_identity=publication.source_identity,
                )
                display.put_payload(
                    StandardDisplayPayload(
                        0,
                        delta.appended,
                        f"{mode} · {scan_key} · {label}",
                        view,
                        measurement_mode=mode,
                        gi_incidence_motor="th" if gi else "",
                        gi_resolved_motor="th" if gi else "",
                        gi_mode_1d="q_total" if gi else "",
                        gi_mode_2d="qip_qoop" if gi else "",
                    )
                )
            context = AcquisitionContext(
                context_token=new_context_token(ContextKind.ACQUISITION),
                run_configuration=configuration,
                config_generation=configuration.generation,
                config_fingerprint=configuration.fingerprint,
                run_scan_key=scan_key,
                source_path=artifact,
                scan=object(),
                frame=None,
                frame_ids=display.catalog,
                frames=display.artifacts,
                viewer_rows_1d=(),
                viewer_rows_2d=(),
                publication_store=display,
                origin="scattering-standard",
                poni_identity=configuration.poni_file,
            )
            context.adopt_record_store(display)
            runtime = _ContextRuntime()
            runtime.adopt_acquisition(identity, context)
            # Overlay preserves the explicit selection (01a73ab4): select
            # every frame, as the Browse routes do.
            frames = runtime.navigation.frames
            assert runtime.select_navigation(frames[-1], frames)
            payloads = runtime.project_navigation(
                ContextProjection(), preferences=prefs,
                processing_mode="Int 2D",
            )
            captured = runtime.norm_aggregate
            assert captured is not None
            assert captured.revision == len(views)
            assert len(payloads) == len(views)
            return _build(payloads, runtime.navigation, prefs, captured)

        reloaded, reload_request, reload_loader = _browse_load(
            Path(browse.requested_path), 2
        )
        owned.append([reload_loader, reloaded, None])
        states = {
            "standard": _acquisition_route(gi=False),
            "gi": _acquisition_route(gi=True),
            "browse": _browse_route(browse, browse_request, browse_loader),
            "reload": _browse_route(reloaded, reload_request, reload_loader),
        }
        for name, state in states.items():
            assert state.norm_channel == channel, name
            assert channel in state.norm_channels, name
            values = _trace_by_label(state)
            assert set(values) == set(views), name
            for label, expected in expected_traces.items():
                np.testing.assert_allclose(
                    values[label], expected, err_msg=f"{name}:{label}"
                )
        assert states["browse"].norm_identity == (
            browse.context_token, browse.scan_key, browse.requested_path,
        )
        assert states["reload"].norm_identity == (
            reloaded.context_token, reloaded.scan_key, reloaded.requested_path,
        )
        assert states["browse"].norm_identity != states["reload"].norm_identity
        assert states["browse"].norm_revision == 1
        assert states["reload"].norm_revision == 1
    finally:
        unsettled = []
        for loader, context, owner in reversed(owned):
            try:
                _release_browse(loader, context, owner)
            except AssertionError as error:
                unsettled.append(error)
        assert not unsettled, unsettled
