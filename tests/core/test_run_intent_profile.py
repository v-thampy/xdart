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


def test_historical_static_scan_translation_is_host_independent(
    tmp_path, monkeypatch
):
    first = tmp_path / "scan_0001.tif"
    second = tmp_path / "scan_0002.tif"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    project = tmp_path / "project"
    save_path = tmp_path / "historical-output"
    monkeypatch.setenv("SAVE_PATH", str(tmp_path / "environment-output"))
    monkeypatch.setenv("VNEXT_MAX_CORES", "47")
    document = {
        "image_wrangler": {
            "image_wrangler": {
                "Project": {
                    "project_folder": str(project),
                    "h5_dir": str(save_path),
                },
                "Signal": {
                    "inp_type": "Image Series",
                    "File": str(second),
                    "meta_ext": "none",
                    "mask_file": str(tmp_path / "legacy-mask.edf"),
                    "write_mode": "Replace",
                },
                "GI": {
                    "Grazing": False,
                    "th_motor": "legacy-theta",
                    "th_val": 0.1,
                    "sample_orientation": 4,
                    "tilt_angle": 0.0,
                    "gi_mode_1d": "q_total",
                    "gi_mode_2d": "qip_qoop",
                },
            }
        },
        "_xdart_static_controls": {
            "schema_version": 1,
            "poni_file": str(tmp_path / "legacy.poni"),
            "processing_mode": "Int 1D",
            "controls_v2_int": {
                "bai_1d_args": {"numpoints": 333},
                "bai_2d_args": {"npt_rad": 444, "npt_azim": 55},
                "gi": True,
                "gi_config": {
                    "incidence_motor": "theta",
                    "th_val": 0.2,
                    "sample_orientation": 6,
                    "tilt_angle": 1.25,
                    "gi_mode_1d": "q_xy",
                    "gi_mode_2d": "qip_qoop",
                },
                "threshold_config": {
                    "apply_threshold": True,
                    "threshold_min": 2.0,
                    "threshold_max": 9.0,
                    "mask_saturation": False,
                },
            },
        },
    }

    loaded = load_run_intent_profile(json.dumps(document))

    assert loaded.generation == 0
    assert type(loaded.source_spec) is SourceSpec
    assert loaded.source_spec.kind is SourceKind.TIFF_SERIES
    assert loaded.source_spec.options["selected_file"] == str(second)
    assert loaded.source_spec.options["files"] == (str(first), str(second))
    assert loaded.source_spec.options["metadata_format"] == "auto"
    assert loaded.project_root == str(project)
    assert loaded.save_path == str(save_path)
    assert loaded.poni_file == str(tmp_path / "legacy.poni")
    assert loaded.mask_file == str(tmp_path / "legacy-mask.edf")
    assert loaded.processing_mode == "Int 1D"
    assert loaded.output_mode == "Overwrite"
    assert loaded.live_mode is loaded.batch_mode is False
    assert loaded.max_cores == 1
    assert loaded.bai_1d_args == {"numpoints": 333}
    assert loaded.bai_2d_args == {"npt_rad": 444, "npt_azim": 55}
    assert loaded.gi == GIIntent(True, "theta", 0.2, 6, 1.25, "q_xy", "qip_qoop")
    assert loaded.threshold == ThresholdIntent(True, 2.0, 9.0, False)
    assert loaded.background == FrameBackgroundPlan()
    assert loaded.run_options == {}


def test_historical_directory_uses_fixed_missing_field_defaults(tmp_path):
    document = {
        "image_wrangler": {
            "image_wrangler": {
                "Signal": {
                    "inp_type": "Image Directory",
                    "img_dir": str(tmp_path / "incoming"),
                    "img_ext": "CBF",
                    "include_subdir": True,
                    "Filter": "sample*",
                    "meta_ext": None,
                }
            }
        }
    }

    loaded = load_run_intent_profile(json.dumps(document))

    assert type(loaded.source_spec) is DirectorySourceSpec
    assert loaded.source_spec.generation == 0
    assert loaded.source_spec.root == tmp_path / "incoming"
    assert loaded.source_spec.recursive is True
    assert loaded.source_spec.suffixes == (".cbf",)
    assert loaded.source_spec.name_filter == "sample*"
    assert loaded.source_spec.metadata_format == "auto"
    assert loaded.project_root == str(tmp_path / "incoming")
    assert loaded.save_path == str(tmp_path / "incoming" / "xdart_processed_data")
    assert loaded.max_cores == 1
    assert loaded.bai_1d_args == {"npt": 128}
    assert loaded.bai_2d_args == {"npt_rad": 128, "npt_azim": 64}


def test_historical_active_background_is_rejected_without_science_loss(
    tmp_path,
):
    document = {
        "image_wrangler": {
            "image_wrangler": {
                "Signal": {"inp_type": "Image Series"},
                "BG": {
                    "bg_type": "Single BG File",
                    "File": str(tmp_path / "background.tif"),
                    "Scale": 0.5,
                    "norm_channel": "monitor",
                },
            }
        }
    }

    with pytest.raises(RunIntentProfileError, match="background"):
        load_run_intent_profile(json.dumps(document))
