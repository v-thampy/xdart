from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.reduction.background import FrameBackgroundPlan
from xrd_tools.session import (
    PROFILE_SCHEMA,
    PROFILE_VERSION,
    RunIntentProfileError,
    dump_run_intent_profile,
    load_run_intent_profile,
)
from xrd_tools.session.run_configuration import (
    GIIntent,
    RunIntent,
    ThresholdIntent,
)
import xrd_tools.session.run_configuration as run_configuration_module
from xrd_tools.sources.selection import DirectorySourceSpec


def _full_intent(tmp_path: Path) -> RunIntent:
    selected = tmp_path / "sample_0001.tif"
    source = SourceSpec(
        tmp_path,
        SourceKind.TIFF_SERIES,
        metadata_uri=tmp_path / "sample.json",
        entry="entry",
        options={
            "selected_file": str(selected),
            "files": (str(selected),),
            "pattern": "sample_*.tif",
            "scan_name": "sample",
            "metadata_format": "auto",
        },
    )
    return RunIntent(
        source_spec=source,
        processing_mode="Int 1D + 2D",
        output_mode="Overwrite",
        live_mode=False,
        batch_mode=True,
        max_cores=3,
        bai_1d_args={
            "npt": 711,
            "unit": "q_A^-1",
            "radial_range": (0.1, 4.2),
        },
        bai_2d_args={
            "npt_rad": 401,
            "npt_azim": 91,
            "lookup": tmp_path / "lookup.dat",
        },
        gi=GIIntent(
            enabled=True,
            incidence_motor="theta",
            th_val=0.25,
            sample_orientation=3,
            tilt_angle=-0.5,
            mode_1d="q_xy",
            mode_2d="qip_qoop",
        ),
        threshold=ThresholdIntent(
            apply_threshold=True,
            threshold_min=-2.0,
            threshold_max=65000.0,
            mask_saturation=False,
        ),
        poni_file=str(tmp_path / "calibration.poni"),
        poni_values={
            "dist": 0.2,
            "detector_config": {"shape": (2048, 2048), "pixel1": 1e-4},
        },
        mask_file=str(tmp_path / "mask.edf"),
        background=FrameBackgroundPlan(
            mode="Single BG File",
            locator=str(tmp_path / "background.tif"),
            scale=-0.75,
            normalization_key="monitor",
        ),
        project_root=str(tmp_path / "project"),
        save_path=str(tmp_path / "processed"),
        run_options={"heavy_window": 32, "writer": {"flush": 8}},
        generation=17,
    )


def test_versioned_profile_round_trips_full_next_run_intent(tmp_path):
    original = _full_intent(tmp_path)

    text = dump_run_intent_profile(original)
    document = json.loads(text)

    assert document["schema"] == PROFILE_SCHEMA
    assert document["version"] == PROFILE_VERSION
    assert "generation" not in document["intent"]
    assert "fingerprint" not in document["intent"]
    assert "generation" not in document["intent"]["source_spec"]
    assert "resolved_motor" not in document["intent"]["gi"]
    assert original.generation == 17

    loaded = load_run_intent_profile(text)
    assert loaded.generation == 0
    assert type(loaded.source_spec) is SourceSpec
    assert isinstance(loaded.source_spec.uri, Path)
    assert isinstance(loaded.source_spec.metadata_uri, Path)
    assert loaded.source_spec.entry == "entry"
    assert loaded.source_spec.options["files"] == (
        str(tmp_path / "sample_0001.tif"),
    )
    assert loaded.processing_mode == "Int 1D + 2D"
    assert loaded.output_mode == "Overwrite"
    assert loaded.batch_mode and not loaded.live_mode
    assert loaded.max_cores == 3
    assert loaded.bai_1d_args == {
        "npt": 711,
        "unit": "q_A^-1",
        "radial_range": (0.1, 4.2),
    }
    assert loaded.bai_2d_args == {
        "npt_rad": 401,
        "npt_azim": 91,
        "lookup": tmp_path / "lookup.dat",
    }
    assert loaded.gi == original.gi
    assert loaded.threshold == original.threshold
    assert loaded.poni_file == original.poni_file
    assert loaded.poni_values == original.poni_values
    assert loaded.mask_file == original.mask_file
    assert loaded.background == original.background
    assert loaded.project_root == original.project_root
    assert loaded.save_path == original.save_path
    assert loaded.run_options == original.run_options

    original_frozen = original.clone_candidate().freeze()
    loaded_frozen = loaded.clone_candidate().freeze()
    assert loaded_frozen.fingerprint == original_frozen.fingerprint


def test_directory_profile_drops_observer_and_intent_generations(tmp_path):
    original = RunIntent(
        source_spec=DirectorySourceSpec(
            tmp_path / "images",
            recursive=True,
            suffixes=(".tif", ".cbf"),
            name_filter="sample*",
            generation=91,
            metadata_format=None,
        ),
        generation=23,
    )

    document = json.loads(dump_run_intent_profile(original))
    assert set(document["intent"]["source_spec"]) == {
        "family",
        "root",
        "recursive",
        "suffixes",
        "name_filter",
        "metadata_format",
    }

    loaded = load_run_intent_profile(json.dumps(document))
    assert loaded.generation == 0
    assert type(loaded.source_spec) is DirectorySourceSpec
    assert loaded.source_spec.generation == 0
    assert loaded.source_spec.root == tmp_path / "images"
    assert loaded.source_spec.recursive is True
    assert loaded.source_spec.suffixes == (".tif", ".cbf")
    assert loaded.source_spec.name_filter == "sample*"
    assert loaded.source_spec.metadata_format is None
    assert original.generation == 23
    assert original.source_spec.generation == 91


def test_profile_load_rejects_partial_or_runtime_state_before_return(tmp_path):
    document = json.loads(dump_run_intent_profile(_full_intent(tmp_path)))

    extra_runtime = copy.deepcopy(document)
    extra_runtime["intent"]["generation"] = 9
    with pytest.raises(RunIntentProfileError, match="invalid keyset"):
        load_run_intent_profile(json.dumps(extra_runtime))

    both_modes = copy.deepcopy(document)
    both_modes["intent"]["live_mode"] = True
    both_modes["intent"]["batch_mode"] = True
    with pytest.raises(RunIntentProfileError, match="cannot both"):
        load_run_intent_profile(json.dumps(both_modes))

    bad_threshold = copy.deepcopy(document)
    bad_threshold["intent"]["threshold"]["threshold_min"] = 20.0
    bad_threshold["intent"]["threshold"]["threshold_max"] = 10.0
    with pytest.raises(RunIntentProfileError, match="threshold_min"):
        load_run_intent_profile(json.dumps(bad_threshold))

    future = copy.deepcopy(document)
    future["version"] = PROFILE_VERSION + 1
    with pytest.raises(RunIntentProfileError, match="version"):
        load_run_intent_profile(json.dumps(future))

    with pytest.raises(RunIntentProfileError, match="duplicate"):
        load_run_intent_profile('{"schema":"a","schema":"b"}')
    with pytest.raises(RunIntentProfileError, match="non-finite"):
        load_run_intent_profile('{"value": NaN}')


def test_unversioned_static_scan_profile_is_refused():
    document = {
        "image_wrangler": {
            "image_wrangler": {
                "Signal": {
                    "inp_type": "Image Series",
                },
            }
        },
    }
    with pytest.raises(RunIntentProfileError, match="profile.*keyset"):
        load_run_intent_profile(json.dumps(document))


def test_versioned_profile_round_trips_enabled_poni_v3_override_and_omits_disabled_legacy_bytes():
    override_type = getattr(
        run_configuration_module, "PoniV3OverrideIntent"
    )
    legacy = RunIntent(poni_file="/calibration/legacy.poni")
    legacy_text = dump_run_intent_profile(legacy)
    assert "poni_v3_override" not in json.loads(legacy_text)["intent"]

    enabled = RunIntent(
        poni_file="/calibration/v3.poni",
        poni_v3_override=override_type("BaFBr0.85I0.15", 0.01, True),
    )
    text = dump_run_intent_profile(enabled)
    document = json.loads(text)
    assert document["intent"]["poni_v3_override"] == {
        "material": "BaFBr0.85I0.15",
        "thickness_m": 0.01,
        "parallax": True,
    }
    loaded = load_run_intent_profile(text)
    assert loaded.poni_v3_override == enabled.poni_v3_override
    assert loaded.clone_candidate().freeze().fingerprint == (
        enabled.clone_candidate().freeze().fingerprint
    )

    malformed = copy.deepcopy(document)
    malformed["intent"]["poni_v3_override"]["parallax"] = 1
    with pytest.raises(RunIntentProfileError):
        load_run_intent_profile(json.dumps(malformed))
