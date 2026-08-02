from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
from xdart.gui.tabs.scattering.display_runtime import (
    RunDisplayState,
    project_detector_values,
    project_frame_detector_values,
)
from xdart.gui.tabs.scattering.events import RunIdentity
from xrd_tools.reduction.core import _RunSaturationMask
from xrd_tools.session.scan_session import _EventSink


def test_live_projection_uses_scan_stable_value_mask_not_current_frame_values() -> None:
    raw = np.zeros((100, 100), dtype=np.uint16)
    raw[3, :2] = np.iinfo(np.uint16).max
    static = np.zeros(raw.shape, dtype=bool)
    static[1, 1] = True
    frame = np.zeros(raw.shape, dtype=bool)
    frame[2, 2] = True
    stable = np.zeros(raw.shape, dtype=bool)
    stable[0, :2] = True

    projected, mask_baked = project_frame_detector_values(
        raw,
        static,
        frame,
        value_mask_enabled=False,
        stable_value_mask=stable,
    )

    assert mask_baked
    assert np.isnan(projected[0, :2]).all()
    assert np.isnan(projected[1, 1])
    assert np.isnan(projected[2, 2])
    assert np.isfinite(projected[3, :2]).all()


def test_masked_integer_projection_uses_one_float_output_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.arange(16, dtype=np.uint16).reshape(4, 4)
    mask = np.zeros(raw.shape, dtype=bool)
    mask[1, 2] = True
    real_array = np.array
    calls: list[tuple[object, object, object]] = []

    def traced_array(value, *args, **kwargs):
        calls.append(
            (kwargs.get("dtype"), kwargs.get("copy"), np.asarray(value).dtype)
        )
        return real_array(value, *args, **kwargs)

    monkeypatch.setattr(
        "xdart.gui.tabs.scattering.display_runtime.np.array",
        traced_array,
    )

    projected, mask_baked = project_detector_values(
        raw,
        mask,
        value_mask_enabled=False,
    )

    assert mask_baked
    assert calls == [(float, True, raw.dtype)]
    assert projected.dtype == float
    assert np.isnan(projected[1, 2])
    assert not projected.flags.writeable


def test_display_owner_copies_one_runtime_mask_and_rejects_divergence() -> None:
    state = RunDisplayState(
        RunIdentity(1, "stable-saturation"),
        max_payload_items=2,
    )
    owner = state.add_artifact(
        Path("/tmp/stable-saturation.nxs"),
        "scan",
        mask=None,
        mask_saturation=True,
        measurement_mode="Standard",
    )
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True

    state.stamp_saturation_mask(owner, mask)
    mask[0, 0] = False

    assert owner.saturation_mask_seeded
    assert owner.saturation_mask is not None
    assert owner.saturation_mask[0, 0]
    assert not owner.saturation_mask.flags.writeable
    state.stamp_saturation_mask(owner, owner.saturation_mask)

    divergent = np.zeros((4, 4), dtype=bool)
    divergent[1, 1] = True
    with pytest.raises(RuntimeError, match="changed within one run"):
        state.stamp_saturation_mask(owner, divergent)


def test_scan_session_forwards_runtime_mask_to_qt_thumbnail_owner() -> None:
    runtime = _RunSaturationMask(True)
    raw = np.zeros((4, 4), dtype=np.int16)
    raw[0, 0] = -1
    runtime.seed(raw)
    sink = QtNexusSink(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        mask=np.array([3], dtype=np.intp),
    )

    _EventSink(sink, lambda _frame, _reduction: None)._bind_run_saturation_mask(
        runtime
    )

    assert sink._run_saturation_mask is runtime
    assert set(map(int, sink._thumbnail_global_mask())) == {0, 3}
