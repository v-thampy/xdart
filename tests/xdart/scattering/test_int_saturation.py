"""Decision 39 in Int: admission resolves the run's automatic ceiling (the value the source records, else
the calibration detector's documented counter limit, else the data type's maximum), signs it into the run's
threshold policy, masks with it, and records it only when it is not the data type's rule."""
import json

import h5py
import numpy as np
import pytest
import tifffile

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _intent, _run_to_terminal, _written
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus


def _run(tmp_path, dtype, name, threshold):
    raw_root = tmp_path / f"raw-{dtype}"
    if not raw_root.exists():
        raw_root.mkdir()
        for label in range(1, 5):
            data = np.full((195, 487), 100, dtype=dtype)
            if label == 2 and (np.dtype(dtype).kind == "f" or np.iinfo(dtype).max > 2_000_000):
                data[90:110, 150:350] = 2_000_000  # at or above the PILATUS counter limit
            tifffile.imwrite(raw_root / f"scan_{label:04}.tif", data)
    poni = tmp_path / "pilatus.poni"
    write_poni(poni)  # Detector: Pilatus100k
    target = tmp_path / name / "output.nexus"
    target.parent.mkdir()
    intent = _intent(raw_root / "scan_0001.tif", target, poni, npt=32)
    for key, value in threshold.items():
        setattr(intent.threshold, key, value)
    executor, identity, events = _run_to_terminal(intent, request_value=abs(hash((dtype, name))) % 90000 + 1000)
    try:
        assert next(event for event in events if event.kind in _TERMINAL).kind is StandardEventKind.FINISHED
    finally:
        assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED
    with h5py.File(_written(target), "r") as handle:
        configuration = json.loads(handle["entry/reduction/config/run_configuration"][()])
        return configuration, handle["entry/integrated_1d/intensity"][()]


def test_int_leaves_out_counts_at_the_pilatus_counter_limit_and_records_it(tmp_path):
    configuration, auto = _run(tmp_path, np.int32, "auto", {})
    for mapping in (configuration["threshold"], configuration["scientific_signature"]["threshold"]):
        assert mapping == {"apply_threshold": False, "threshold_min": None, "threshold_max": None,
                           "mask_saturation": True, "saturation": 1048575}
    # the same science as keeping 0..1048574 (a manual maximum keeps its own value)
    manual_configuration, manual = _run(tmp_path, np.int32, "manual", {
        "apply_threshold": True, "threshold_min": 0.0, "threshold_max": 1048574.0, "mask_saturation": False})
    assert "saturation" not in manual_configuration["threshold"]
    np.testing.assert_array_equal(auto, manual)
    _off, off = _run(tmp_path, np.int32, "off", {"mask_saturation": False})
    assert not np.array_equal(np.nan_to_num(off[1]), np.nan_to_num(auto[1]))  # frame 2's block, kept when off
    np.testing.assert_array_equal(off[0], auto[0])


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_a_ceiling_the_data_cannot_reach_records_nothing(tmp_path, dtype):
    # uint16 never reaches 1048575 (65535 is always left out); float data takes no counter limit
    configuration, _intensity = _run(tmp_path, dtype, "auto", {})
    assert "saturation" not in configuration["threshold"]
    assert "saturation" not in configuration["scientific_signature"]["threshold"]


def test_the_int_hint_is_the_run_ceiling_and_never_the_value(tmp_path):
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xrd_tools.sources.selection import DirectorySourceSpec, image_series_spec
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.contracts import SourceObservationRequest
    from xdart.gui.tabs.scattering.controls_editing import observed_threshold_max
    from xdart.gui.tabs.scattering.controls_inventory import THRESHOLD_MAX
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    image = tmp_path / "frame_0001.tif"
    tifffile.imwrite(image, np.full((8, 8), 17, dtype=np.int32))
    poni = tmp_path / "pilatus.poni"
    write_poni(poni)
    for source in (image_series_spec(image), DirectorySourceSpec(str(tmp_path))):
        intent = RunIntent(source_spec=source, poni_file=str(poni))
        observed = FilesystemSourceAdapter().observe(SourceObservationRequest(1, 0, intent.source_spec))
        field = {item.path: item for item in project_controls(
            RunIntentStore(intent).snapshot(), observed, RunPhase.IDLE).fields}[THRESHOLD_MAX]
        assert field.value is None and field.placeholder == "Auto · 1048575 (saturation)"
        assert observed_threshold_max(intent, observed) == 1048574.0


def test_a_float_image_with_a_recorded_cutoff_keeps_its_raw_preview(tmp_path):
    """Codex terminal review F1: float PILATUS TIFFs recording Count_cutoff 400. The run resolves and signs 400;
    the FULL raw preview takes that ceiling (a float frame has no dtype one) and masks at and above it, rather
    than reporting its detector projection unavailable."""
    from tests.xdart.scattering.test_e4_preview_transport import _acquisition_identity, _typed_request
    from xrd_tools.io.frame_preview import read_frame_preview
    from xrd_tools.session.hydration import HydrationPurpose
    for label in (1, 2):
        raw = np.full((195, 487), 100, np.float32)
        raw[90:110, 150:350] = 425
        tifffile.imwrite(tmp_path / f"scan_{label:04}.tif", raw, description="# Count_cutoff 400 counts\n")
    poni = tmp_path / "pilatus.poni"
    write_poni(poni)
    target = tmp_path / "out" / "output.nexus"
    target.parent.mkdir()
    executor, identity, events = _run_to_terminal(
        _intent(tmp_path / "scan_0001.tif", target, poni, npt=32), request_value=9251)
    try:
        assert next(event for event in events if event.kind in _TERMINAL).kind is StandardEventKind.FINISHED
        path = _written(target)
        with h5py.File(path) as handle:
            recorded = json.loads(handle["entry/reduction/config/run_configuration"][()])
        assert recorded["threshold"]["saturation"] == 400
        state = executor._exact_run(identity).display
        owner = state.artifacts[str(path)]
        assert (owner.saturation, owner.saturation_ceiling) == (400, 400.0)
        live, gate = _acquisition_identity(state)
        request = _typed_request(state, live, gate, path, 2, HydrationPurpose.FULL, 1)
        projection, _key = state._derive_target(request)
        preview = read_frame_preview(request.read_key, detector_projection=projection)
        assert projection.mask_available and preview.raw is not None
    finally:
        assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED


def test_the_live_preview_leaves_out_values_at_the_run_ceiling():
    from xrd_tools.io.frame_preview import DetectorPreviewProjection, _masked_detector
    from xdart.gui.tabs.scattering.display_runtime import DisplayArtifact, RunDisplayState
    owner = DisplayArtifact.__new__(DisplayArtifact)
    owner.saturation, owner.saturation_ceiling = 1048575, None
    RunDisplayState.stamp_saturation_ceiling(None, owner, np.zeros((2, 2), np.int32))
    assert owner.saturation_ceiling == 1048575.0  # below int32's own
    projection = DetectorPreviewProjection.from_mask(
        np.zeros((1, 4), bool), mask_saturation=True, saturation_ceiling=owner.saturation_ceiling)
    shown = _masked_detector(np.array([[1048574, 1048575, 1048576, -1]], np.int32), projection)
    assert np.isfinite(shown).tolist() == [[True, False, False, False]]
    native = DisplayArtifact.__new__(DisplayArtifact)
    native.saturation, native.saturation_ceiling = None, None
    RunDisplayState.stamp_saturation_ceiling(None, native, np.zeros((2, 2), np.uint16))
    assert native.saturation_ceiling == 65535.0  # the data type's, as before


def test_reintegrate_reads_the_saved_ceiling_and_absence_is_the_data_type_rule():
    from xrd_tools.reduction import reintegrate as module
    selected = {"version": 1, "dimension": "1d", "gi_mode": None,
                "bai_args": {"npt": 100, "unit": "q_A^-1", "method": "csr", "radial_range": None,
                             "azimuth_range": None}}
    shared = {"version": 1, "poni_values": None, "geometry": None, "background": None,
              "accepted_scientific_assets": {"poni_values": None, "poni_detector_config_json": None,
                                             "poni_sha256": None, "mask_sha256": None},
              "gi": {"enabled": False, "incidence_motor": "Manual", "resolved_motor": "Manual", "th_val": 0.0,
                     "sample_orientation": 4, "tilt_angle": 0.0},
              "threshold": {"apply_threshold": False, "threshold_min": None, "threshold_max": None,
                            "mask_saturation": True}}
    module._validate_science(selected, shared, "1d")
    assert module._core_plan(selected, shared).saturation is None  # an older output: the data type's rule
    saved = {**shared, "threshold": {**shared["threshold"], "saturation": 1048575}}
    module._validate_science(selected, saved, "1d")
    assert module._core_plan(selected, saved).saturation == 1048575
    for bad in (0, 1048575.0, True):
        with pytest.raises(ValueError, match="threshold science is malformed"):
            module._validate_science(selected, {**shared, "threshold": {**shared["threshold"], "saturation": bad}}, "1d")
    with pytest.raises(ValueError, match="threshold science is malformed"):
        module._validate_science(selected, {**shared, "threshold": {
            **shared["threshold"], "mask_saturation": False, "saturation": 1048575}}, "1d")
