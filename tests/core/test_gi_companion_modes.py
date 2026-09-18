# -*- coding: utf-8 -*-
"""GI-COMPANION-20260918 — a GI run may declare a second DIRECT 2-D map.

``plan.extra["enabled_modes_2d"]`` lists the primary GI 2-D mode first and any
companion after it.  Both maps are integrated by pyFAI from the same conditioned
frame, each on its own frozen grid, cross the completion together (one frame
counts once), and are stored as the primary group plus a nested extra mode.

Everything here drives the real engine, session, writer and reader with real
pyFAI on a small real detector geometry; nothing on those seams is faked.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from xrd_tools.session.stage_accounting import ResultMode
from xrd_tools.core.containers import PONI
from xrd_tools.integrate.calibration import poni_to_integrator
from xrd_tools.io import read_frame_record
from xrd_tools.reduction import (
    Frame,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    NexusSink,
    ReductionPlan,
    Scan,
    run_reduction,
)
from xrd_tools.reduction.core import COMPANION_RANGES_2D, companion_modes_2d
from xrd_tools.session.frame_record_store import FrameRecordStore
from xrd_tools.session.scan_session import ScanSession, required_result_modes

SHAPE = (195, 487)          # Pilatus 100k
ANGLES = (0.15, 0.20, 0.25)
NPT_RAD, NPT_AZIM = 30, 24


def _scan() -> Scan:
    poni = PONI(dist=0.15, poni1=0.002, poni2=0.04, wavelength=1.0e-10,
                detector="Pilatus100k")
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    frames = [
        Frame(
            index,
            image=50 + 40 * np.sin(xx / 9.0 + index) + 30 * np.cos(yy / 7.0)
            + rng.poisson(5, SHAPE),
            metadata={"th": angle},
        )
        for index, angle in enumerate(ANGLES)
    ]
    return Scan("gi", frames, poni=poni, integrator=poni_to_integrator(poni))


def _plan(*modes: str, ranges: dict | None = None, one_d: bool = True) -> ReductionPlan:
    extra: dict = {"enabled_modes_2d": tuple(modes)}
    p2d = Integration2DPlan(npt_rad=NPT_RAD, npt_azim=NPT_AZIM, unit="qip_A^-1")
    if ranges:
        if modes[0] == "qip_qoop":
            p2d = Integration2DPlan(
                npt_rad=NPT_RAD, npt_azim=NPT_AZIM, unit="qip_A^-1", extra=dict(ranges),
            )
        else:
            p2d = Integration2DPlan(
                npt_rad=NPT_RAD, npt_azim=NPT_AZIM, unit="q_A^-1", **ranges,
            )
    return ReductionPlan(
        integration_1d=Integration1DPlan(npt=40) if one_d else None,
        integration_2d=p2d,
        gi=GIMode(incidence_motor="th", mode_2d=modes[0], method="no"),
        extra=extra,
    )


def _run(tmp_path, name: str, plan: ReductionPlan):
    target = tmp_path / f"{name}.nexus"
    sink = NexusSink(path=target, overwrite=True)
    result = run_reduction(plan, _scan(), sink=sink, executor=1,
                           gi_freeze_mode="scout_union")
    assert not result.failed and result.n_processed == len(ANGLES)
    return target, sink


def test_a_single_declared_mode_has_no_companion():
    assert companion_modes_2d(_plan("qip_qoop")) == ()
    assert companion_modes_2d(_plan("q_chi")) == ()
    assert companion_modes_2d(ReductionPlan()) == ()


@pytest.mark.parametrize("modes", (
    ("q_chi", "qip_qoop", "q_chi"),      # repeat
    ("qip_qoop", "exit_angles"),         # not a q-space companion
))
def test_a_malformed_declaration_is_refused(modes):
    plan = _plan(*modes)
    with pytest.raises(ValueError, match="enabled_modes_2d|limited to"):
        companion_modes_2d(plan)


def test_the_declaration_must_lead_with_the_primary_mode():
    plan = _plan("qip_qoop", "q_chi")
    plan.extra["enabled_modes_2d"] = ("q_chi", "qip_qoop")
    with pytest.raises(ValueError, match="primary GI 2-D mode first"):
        companion_modes_2d(plan)


def test_both_direct_maps_are_stored_with_their_own_frozen_axes(tmp_path):
    target, _sink = _run(tmp_path, "both", _plan("qip_qoop", "q_chi"))

    with h5py.File(target, "r") as handle:
        top = handle["entry/integrated_2d"]
        assert top.attrs["primary_mode"] == "qip_qoop"
        assert list(top.attrs["multi_result_modes"]) == ["qip_qoop", "q_chi"]
        # One physical frame, once, in each map.
        labels = list(range(len(ANGLES)))
        assert list(top["frame_index"][()]) == labels
        assert list(top["q_chi/frame_index"][()]) == labels
        assert list(handle["entry/integrated_1d/frame_index"][()]) == labels
        assert len(handle["entry/frames"]) == len(ANGLES)

    records = [read_frame_record(target, label) for label in range(len(ANGLES))]
    for record in records:
        assert record.modes_2d == ("qip_qoop", "q_chi")
        assert record.active_mode_2d == "qip_qoop"
    cart, polar = records[0].view_2d("qip_qoop"), records[0].view_2d("q_chi")
    assert (cart.axis_2d_x.unit, cart.axis_2d_y.unit) == ("qip_A^-1", "qoop_A^-1")
    assert (polar.axis_2d_x.unit, polar.axis_2d_y.unit) == ("qtot_A^-1", "chigi_deg")
    # q/chi limits are their own: a magnitude from ~0 and an angle in degrees,
    # never the signed qip window or the qoop window.
    assert polar.axis_2d_x.values[0] >= 0.0
    assert cart.axis_2d_x.values[0] < 0.0
    assert -180.0 <= polar.axis_2d_y.values[0] < polar.axis_2d_y.values[-1] <= 180.0
    assert polar.axis_2d_y.values[-1] > 10 * cart.axis_2d_y.values[-1]
    # Frozen across the scan, per map.
    for record in records[1:]:
        for mode, first in (("qip_qoop", cart), ("q_chi", polar)):
            view = record.view_2d(mode)
            np.testing.assert_array_equal(view.axis_2d_x.values, first.axis_2d_x.values)
            np.testing.assert_array_equal(view.axis_2d_y.values, first.axis_2d_y.values)


def test_each_direct_map_equals_its_own_single_mode_run_on_the_same_grid(tmp_path):
    """The numerical contract: a companion is the ordinary direct integration."""
    from xrd_tools.reduction.core import _apply_gi_freeze_policy

    frozen = _apply_gi_freeze_policy(
        _plan("qip_qoop", "q_chi"), _scan(), freeze_policy="scout_union",
        fi=None, initial_incident_angle=None,
    )
    both, _sink = _run(tmp_path, "both", frozen)
    cart_ranges = {key: frozen.integration_2d.extra[key] for key in ("x_range", "y_range")}
    polar_ranges = dict(frozen.extra[COMPANION_RANGES_2D]["q_chi"])
    assert set(polar_ranges) == {"radial_range", "azimuth_range"}

    cart_only, _ = _run(tmp_path, "cart", _plan("qip_qoop", ranges=cart_ranges))
    polar_only, _ = _run(tmp_path, "polar", _plan("q_chi", ranges=polar_ranges))

    for label in range(len(ANGLES)):
        record = read_frame_record(both, label)
        for mode, single in (("qip_qoop", cart_only), ("q_chi", polar_only)):
            ours = record.view_2d(mode)
            theirs = read_frame_record(single, label).view_2d(mode)
            np.testing.assert_array_equal(ours.axis_2d_x.values, theirs.axis_2d_x.values)
            np.testing.assert_array_equal(ours.axis_2d_y.values, theirs.axis_2d_y.values)
            np.testing.assert_array_equal(ours.intensity_2d, theirs.intensity_2d)
            assert np.isfinite(ours.intensity_2d).any()


def test_q_chi_alone_is_stored_once_as_the_primary_group(tmp_path):
    target, _sink = _run(tmp_path, "polar", _plan("q_chi"))
    with h5py.File(target, "r") as handle:
        top = handle["entry/integrated_2d"]
        assert top.attrs["primary_mode"] == "q_chi"
        assert "q_chi" not in top and "qip_qoop" not in top
    record = read_frame_record(target, 0)
    assert record.modes_2d == ("q_chi",)


def test_plain_qip_qoop_runs_one_2d_integration_per_frame(tmp_path, monkeypatch):
    """No hidden second detector integration, and no companion bookkeeping."""
    from xrd_tools.reduction import core

    calls: list[str] = []
    real = core._run_gi_2d

    def counted(image, fi, plan, gi, **kwargs):
        calls.append(gi.mode_2d.value)
        return real(image, fi, plan, gi, **kwargs)

    monkeypatch.setattr(core, "_run_gi_2d", counted)
    target, sink = _run(tmp_path, "cart", _plan("qip_qoop"))

    # Two scout integrations (first + last frame) freeze the grid, then exactly
    # one per frame.
    assert calls == ["qip_qoop"] * (2 + len(ANGLES))
    assert not sink._companion_record_writes
    with h5py.File(target, "r") as handle:
        assert "q_chi" not in handle["entry/integrated_2d"]

    calls.clear()
    _run(tmp_path, "both", _plan("qip_qoop", "q_chi"))
    assert calls.count("qip_qoop") == 2 + len(ANGLES)
    assert calls.count("q_chi") == 2 + len(ANGLES)


def test_the_session_counts_a_two_map_frame_once_and_keeps_both_results(tmp_path):
    plan = _plan("qip_qoop", "q_chi")
    assert required_result_modes(plan) == (
        ResultMode.one_d("q_total"),
        ResultMode.two_d("qip_qoop"),
        ResultMode.two_d("q_chi"),
    )
    scan = _scan()
    store = FrameRecordStore(max_heavy_items=None)
    target = tmp_path / "session.nexus"
    session = ScanSession(
        plan, scan, sink=NexusSink(path=target, overwrite=True), executor=1,
        gi_freeze_mode="scout_union", record_store=store,
    )
    events = []
    session.on_frame_completed(events.append)
    for frame in scan.frames:
        session.submit(frame)
    session.finish()

    assert [event.frame_index for event in events] == [0, 1, 2]
    assert session.frames_completed == len(ANGLES)
    for event in events:
        assert event.result_2d is not None
        assert tuple(event.extra_results_2d) == ("q_chi",)
        assert not event.extra_results_2d["q_chi"].intensity.flags.writeable
    for label in range(len(ANGLES)):
        live = store.get(label)
        assert live.modes_2d == ("qip_qoop", "q_chi")
        assert live.active_mode_2d == "qip_qoop"
        reloaded = read_frame_record(target, label)
        for mode in live.modes_2d:
            np.testing.assert_allclose(
                live.view_2d(mode).intensity_2d,
                reloaded.view_2d(mode).intensity_2d,
                rtol=1e-6, equal_nan=True,
            )
            np.testing.assert_allclose(
                live.view_2d(mode).axis_2d_x.values,
                reloaded.view_2d(mode).axis_2d_x.values,
            )
