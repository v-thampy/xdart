from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from xdart.gui.tools.stitch_values import StitchFrameSelector, StitchToolForm
from xrd_tools.analysis.module_transaction import MetadataColumnSelector
from xrd_tools.analysis.stitch_operation import StitchGeometryKind
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite


def _geometry_record() -> dict[str, object]:
    names = [
        "dist",
        "poni1",
        "poni2",
        "rot1_offset",
        "rot1_scale",
        "rot2_offset",
        "rot2_scale",
    ]
    positions = ["del_angle", "nu_angle"]
    return {
        "content": "Goniometer calibration v2",
        "detector": "Pilatus100k",
        "detector_config": {"orientation": 3},
        "wavelength": 1.0e-10,
        "param": [0.2, 0.01, 0.02, 0.0, 0.01, 0.0, 0.01],
        "param_names": names,
        "pos_names": positions,
        "trans_function": {
            "content": "GeometryTransformation",
            "param_names": names,
            "pos_names": positions,
            "dist_expr": "dist",
            "poni1_expr": "poni1",
            "poni2_expr": "poni2",
            "rot1_expr": "rot1_scale * nu_angle + rot1_offset",
            "rot2_expr": "rot2_scale * del_angle + rot2_offset",
            "rot3_expr": "0.0",
            "constants": {"pi": float(np.pi)},
        },
    }


@pytest.fixture
def stitch_form(tmp_path):
    spec = tmp_path / "myscan"
    spec.write_text(
        "#F myscan\n"
        "#E 1\n"
        "#D today\n"
        "#O0 del  nu\n"
        "\n"
        "#S 5 ascan del 0 2 2 1\n"
        "#D today\n"
        "#P0 0 0\n"
        "#N 4\n"
        "#L del  nu  I0  det\n"
        "0 0 10 1\n"
        "1 0.1 11 2\n"
        "2 0.2 12 3\n",
        encoding="utf-8",
    )
    images = tmp_path / "images"
    images.mkdir()
    for label in range(3):
        np.full((4, 5), label + 1, dtype=np.int32).tofile(
            images / f"myscan_scan5_{label:04d}.raw"
        )
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps(_geometry_record()), encoding="utf-8")
    digest = hashlib.sha256(geometry.read_bytes()).hexdigest()
    output = tmp_path / "output"
    output.mkdir()
    return StitchToolForm(
        project_root=tmp_path,
        spec_path=spec,
        scan="5",
        image_dir=images,
        image_stem="myscan_scan5_",
        frame_selector=StitchFrameSelector(0, 2, 2),
        detector_shape=(4, 5),
        raw_dtype="int32",
        raw_header_skip=0,
        threshold=800_000,
        geometry_path=geometry,
        geometry_kind=StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        expected_geometry_sha256=digest,
        source_motors=(("del_angle", "del"), ("nu_angle", "nu")),
        q_range=(0.1, 6.0),
        npt_1d=96,
        monitor_selector=MetadataColumnSelector("I0"),
        use_detector_mask=True,
        output_path=output / "stitched.nexus",
        overwrite=AnalysisArtifactOverwrite.CREATE_NEW,
        max_frame_bytes=1 << 20,
    )
