"""Headless contracts for generation-stamped run configuration values."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.readiness import processing_config_from_mapping
from xrd_tools.session.run_configuration import (
    FrozenRunConfiguration,
    GIIntent,
    RunIntent,
    ThresholdIntent,
)
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent() -> RunIntent:
    nested_files = ["/data/scan_0001.tif", "/data/scan_0002.tif"]
    return RunIntent(
        source_spec=SourceSpec(
            Path("/data"),
            SourceKind.TIFF_SERIES,
            metadata_uri=Path("/data/scan.txt"),
            entry="entry",
            options={
                "files": nested_files,
                "selection": {"indices": [1, 2]},
            },
        ),
        processing_mode="Int 2D",
        output_mode="Replace",
        max_cores=4,
        bai_1d_args={
            "unit": "q_A^-1",
            "npt": 1000,
            "radial_range": [0.1, 5.0],
            "method": ("bbox", "csr", "cython"),
        },
        bai_2d_args={
            "unit": "q_A^-1",
            "npt_rad": 500,
            "npt_azim": 360,
            "nested": {"weights": [1.0, 2.0]},
        },
        gi=GIIntent(
            enabled=True,
            incidence_motor="halpha",
            th_val=0.12,
            sample_orientation=3,
            tilt_angle=0.25,
            mode_1d="q_ip",
            mode_2d="qip_qoop",
        ),
        threshold=ThresholdIntent(
            apply_threshold=True,
            threshold_min=10,
            threshold_max=60_000,
            mask_saturation=False,
        ),
        poni_file="/calibration/eiger.poni",
        poni_values={
            "dist": 0.2,
            "detector_config": {"pixel1": 75e-6},
        },
        mask_file="/calibration/eiger-mask.npy",
        project_root="/data",
        save_path="/processed",
        run_options={"writer": {"flush_every": 5}},
    )


def test_freeze_is_deeply_immutable_and_generation_is_content_independent():
    intent = _intent()
    frozen = intent.freeze()

    # Mutating every mutable owner-side input after Run cannot change this run.
    intent.bai_1d_args["radial_range"][0] = 99
    intent.bai_2d_args["nested"]["weights"].append(3.0)
    intent.gi.incidence_motor = "theta"
    intent.threshold.threshold_max = 20
    intent.poni_values["detector_config"]["pixel1"] = 1.0
    intent.run_options["writer"]["flush_every"] = 99
    intent.source_spec.options["files"].append("/data/scan_0003.tif")
    intent.source_spec.options["selection"]["indices"].append(3)

    assert frozen.generation == 1
    assert frozen.output_mode == "Overwrite"
    assert frozen.gi.incidence_motor == "halpha"
    assert frozen.threshold.threshold_max == 60_000
    assert frozen.bai_1d_args["radial_range"] == [0.1, 5.0]
    assert frozen.bai_2d_args["nested"]["weights"] == [1.0, 2.0]
    assert frozen.poni_values["detector_config"]["pixel1"] == 75e-6
    assert frozen.run_options["writer"]["flush_every"] == 5
    assert frozen.thaw_source_spec().options["files"] == [
        "/data/scan_0001.tif",
        "/data/scan_0002.tif",
    ]

    # Accessors return independent copies rather than exposing frozen storage.
    args = frozen.bai_1d_args
    args["radial_range"].append(12)
    provenance = frozen.as_provenance()
    provenance["bai_2d_args"]["nested"]["weights"].clear()
    assert frozen.bai_1d_args["radial_range"] == [0.1, 5.0]
    assert frozen.bai_2d_args["nested"]["weights"] == [1.0, 2.0]

    # Generation distinguishes accepted runs; fingerprint describes content.
    same_content = RunIntent.from_frozen(frozen).freeze()
    assert same_content.generation == 2
    assert same_content.fingerprint == frozen.fingerprint
    assert same_content.identity != frozen.identity


def test_accessors_preserve_gi_motor_and_processing_signature():
    frozen = _intent().freeze()

    assert frozen.scan_args() == {
        "bai_1d_args": frozen.bai_1d_args,
        "bai_2d_args": frozen.bai_2d_args,
    }
    kwargs = frozen.scan_kwargs()
    assert kwargs["gi"] is True
    assert kwargs["incidence_motor"] == "halpha"
    assert kwargs["skip_2d"] is False
    assert kwargs["apply_threshold"] is True
    assert kwargs["threshold_min"] == 10
    assert kwargs["threshold_max"] == 60_000
    assert kwargs["mask_sentinel"] is False

    mapping = frozen.processing_mapping()
    assert mapping["gi_config"] == {
        "gi_mode_1d": "q_ip",
        "gi_mode_2d": "qip_qoop",
        "incidence_motor": "halpha",
        "th_val": 0.12,
        "sample_orientation": 3,
        "tilt_angle": 0.25,
    }
    signature = processing_config_from_mapping(mapping)
    assert signature is not None
    assert signature.mode.value == "gi"
    assert signature.axis_1d == "q_ip"
    assert signature.axis_2d == "qip_qoop"
    assert signature.npt_1d == 1000
    assert signature.npt_rad_2d == 500
    assert signature.npt_azim_2d == 360


def test_manual_gi_motor_thaws_to_numeric_live_scan_value():
    intent = _intent()
    intent.gi.incidence_motor = "Manual"
    intent.gi.th_val = 0.375

    frozen = intent.freeze()

    assert frozen.gi.incidence_motor == "Manual"
    assert frozen.gi.scan_incidence_motor == "0.375"
    assert frozen.scan_kwargs()["incidence_motor"] == "0.375"
    assert frozen.processing_mapping()["gi_config"]["incidence_motor"] == "Manual"


def test_directory_source_round_trip_is_typed_and_independent(tmp_path):
    source = DirectorySourceSpec(
        root=tmp_path / "raw",
        recursive=True,
        suffixes=(".NXS", ".H5"),
        name_filter="sample",
        generation=8,
    )
    intent = RunIntent(source_spec=source)

    frozen = intent.freeze()
    intent.source_spec = None
    thawed = frozen.thaw_source_spec()

    assert isinstance(thawed, DirectorySourceSpec)
    assert thawed is not source
    assert thawed.root == tmp_path / "raw"
    assert thawed.recursive is True
    assert thawed.suffixes == (".nxs", ".h5")
    assert thawed.name_filter == "sample"
    assert thawed.generation == 8


def test_fingerprint_is_mapping_order_independent_and_value_sensitive():
    first = _intent()
    second = _intent()
    second.bai_1d_args = dict(reversed(tuple(second.bai_1d_args.items())))
    second.bai_2d_args = dict(reversed(tuple(second.bai_2d_args.items())))

    first_frozen = first.freeze()
    second_frozen = second.freeze()

    assert first_frozen.fingerprint == second_frozen.fingerprint
    second.gi.incidence_motor = "theta"
    changed = second.freeze()
    assert changed.fingerprint != second_frozen.fingerprint


def test_failed_freeze_does_not_consume_generation():
    intent = RunIntent(live_mode=True, batch_mode=True)

    with pytest.raises(
        ValueError,
        match="live_mode and batch_mode cannot both be enabled",
    ):
        intent.freeze()

    assert intent.generation == 0
    intent.batch_mode = False
    assert intent.freeze().generation == 1


def test_session_package_exposes_values_lazily_without_qt():
    from xrd_tools.session import RunIntent as PublicRunIntent

    assert PublicRunIntent is RunIntent
    assert issubclass(FrozenRunConfiguration, object)

    # R4B-13: import purity ("keeps xdart thin") must be proven in a FRESH
    # interpreter.  Asserting ``"pyqtgraph" not in sys.modules`` in THIS process
    # was order-fragile — any earlier Qt-importing test in the same pytest
    # process left pyqtgraph resident and failed the assertion vacuously.  A
    # subprocess isolates the fact under test: importing ``xrd_tools.session``
    # must not drag in Qt/pyqtgraph.
    import subprocess

    code = (
        "import sys\n"
        "import xrd_tools.session as session\n"
        "assert session.RunIntent is not None\n"
        "leaked = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name == 'qtpy' or name == 'pyqtgraph'\n"
        "    or name.startswith('qtpy.') or name.startswith('pyqtgraph.')\n"
        ")\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
