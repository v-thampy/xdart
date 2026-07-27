"""W-1R-D4B: the streaming writer consumes the exact admitted run policy."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from tests.xdart._accepted_run import accepted_run, gi_intent, series_source
from xdart.gui.tabs.static_scan.static_scan_widget import _accepted_run_policy
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink


def _real_worker_with_policy(
    tmp_path,
    *,
    batch_mode=False,
    xye_only=False,
    gi=False,
    incidence_motor="Manual",
    series_average=False,
):
    worker = imageThread.__new__(imageThread)
    frozen = accepted_run(
        source_spec=series_source(tmp_path / "raw_0001.tif"),
        save_path=str(tmp_path),
        processing_mode="Int 1D" if xye_only else "Int 2D",
        batch_mode=batch_mode,
        live_mode=not batch_mode,
        gi=gi_intent(
            enabled=gi,
            incidence_motor=incidence_motor,
            sample_orientation=4,
            tilt_angle=0.0,
        ),
        run_options={
            "xye_only": xye_only,
            "series_average": series_average,
            "meta_ext": "txt",
        },
    )
    worker.run_configuration = worker._admitted_run_configuration = frozen
    return worker, frozen


def test_streaming_sink_xye_policy_needs_no_retired_worker_fields(tmp_path):
    worker, frozen = _real_worker_with_policy(
        tmp_path,
        batch_mode=False,
        xye_only=True,
    )
    assert worker._require_run_configuration("probe") is frozen
    for retired in (
        "batch_mode",
        "xye_only",
        "gi",
        "incidence_motor",
        "series_average",
    ):
        assert not hasattr(worker, retired)

    scan = SimpleNamespace(
        frames=SimpleNamespace(_in_memory_cap=64),
        skip_2d=True,
    )
    sink = QtNexusSink(
        worker,
        scan,
        SimpleNamespace(gi=None),
        run_configuration=frozen,
    )
    assert sink._run_configuration is frozen
    live = SimpleNamespace(idx=1)
    sink.register(live)

    sink.worker_process(
        SimpleNamespace(index=1),
        SimpleNamespace(corrected_image=None),
    )


def test_streaming_sink_live_publication_uses_frozen_batch_mode(tmp_path):
    worker, frozen = _real_worker_with_policy(
        tmp_path,
        batch_mode=False,
    )
    worker._published_frames = {}
    emitted = []
    worker.sigUpdate = SimpleNamespace(emit=emitted.append)
    sink = QtNexusSink(
        worker,
        SimpleNamespace(frames=SimpleNamespace(_in_memory_cap=64)),
        SimpleNamespace(gi=None),
        run_configuration=frozen,
    )

    sink._publish_display(SimpleNamespace(idx=7))

    assert emitted == [7]
    assert worker._published_frames[7].idx == 7


def test_streaming_sink_gi_motor_and_series_policy_ignore_host_poison(tmp_path):
    worker, frozen = _real_worker_with_policy(
        tmp_path,
        batch_mode=True,
        gi=True,
        incidence_motor="th",
        series_average=True,
    )
    # If the sink consults a legacy host field, these deliberately disagree.
    worker.gi = False
    worker.incidence_motor = "POISON"
    worker.series_average = False
    calls = []
    scan = SimpleNamespace(add_frame=lambda **kwargs: calls.append(kwargs))
    sink = QtNexusSink(
        worker,
        scan,
        SimpleNamespace(gi=None),
        run_configuration=frozen,
    )

    sink._add_frame(SimpleNamespace())

    assert len(calls) == 1
    assert calls[0]["gi"] is True
    assert calls[0]["th_mtr"] == "th"
    assert calls[0]["series_average"] is True


def test_host_policy_requires_the_exact_admitted_identity(tmp_path):
    _worker, frozen = _real_worker_with_policy(tmp_path)

    assert _accepted_run_policy(
        SimpleNamespace(
            run_configuration=frozen,
            _admitted_run_configuration=frozen,
        )
    ) is frozen
    assert _accepted_run_policy(
        SimpleNamespace(run_configuration=frozen)
    ) is None
    assert _accepted_run_policy(
        SimpleNamespace(
            run_configuration=replace(frozen),
            _admitted_run_configuration=frozen,
        )
    ) is None


def test_streaming_sink_requires_the_exact_admitted_identity(tmp_path):
    worker, frozen = _real_worker_with_policy(tmp_path)
    scan = SimpleNamespace(frames=SimpleNamespace(_in_memory_cap=64))
    plan = SimpleNamespace(gi=None)

    QtNexusSink(worker, scan, plan, run_configuration=frozen)

    del worker._admitted_run_configuration
    try:
        QtNexusSink(worker, scan, plan, run_configuration=frozen)
    except ValueError:
        pass
    else:
        raise AssertionError("a bare frozen carrier must not authorize a sink")

    worker._admitted_run_configuration = frozen
    try:
        QtNexusSink(worker, scan, plan, run_configuration=replace(frozen))
    except ValueError:
        pass
    else:
        raise AssertionError(
            "an equal-valued reconstruction must not authorize a sink")
