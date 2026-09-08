from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from xdart.gui.tools.rsm_values import (
    RSMFrameSelector,
    RSMToolForm,
    prepare_rsm_tool,
)
from xdart.gui.tools.stitch_values import StitchFrameSelector, StitchToolForm
from xrd_tools.analysis.module_transaction import MetadataColumnSelector
from xrd_tools.analysis.rsm_operation import (
    RSMDetectorGeometry,
    RSMImageConditioning,
    RSMNormalizationMode,
    RSMNormalizationPolicy,
    RSMOperationPlan,
)
from xrd_tools.analysis.stitch_operation import StitchGeometryKind
from xrd_tools.core.geometry import DetectorHeader, ImageOrientation
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
    output.mkdir(exist_ok=True)
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
        max_frame_bytes=1 << 20,
    )


@pytest.fixture
def rsm_tool_form(tmp_path: Path) -> RSMToolForm:
    source_root = tmp_path / "source"
    image_root = source_root / "images"
    output_root = tmp_path / "output"
    image_root.mkdir(parents=True)
    output_root.mkdir(exist_ok=True)
    spec_path = source_root / "RSM_synth"
    spec_path.write_text(
        """#F RSM_synth
#E 1
#D Mon Jan 15 10:30:00 2024
#O0 energy  mu  chi  phi  nu  del

#S 1 ascan eta 0 2 2 1
#D Mon Jan 15 10:30:00 2024
#P0 13000.007 10 20 30 40 50
#G3 1 0 0 0 1 0 0 0 1
#N 4
#L eta  Seconds  Seconds  foil status
0 1 10 101
1 2 20 110
2 3 40 1010
""",
        encoding="utf-8",
    )
    frames = (
        np.arange(20, dtype=np.uint16).reshape(4, 5) + 1,
        np.arange(20, dtype=np.uint16).reshape(4, 5) + 21,
        np.arange(20, dtype=np.uint16).reshape(4, 5) + 41,
    )
    for index, frame in enumerate(frames):
        tifffile.imwrite(image_root / f"frame_{index:04d}.tif", frame)
    geometry = RSMDetectorGeometry(
        DetectorHeader(1.5, 2.0, 0.172, 0.172, 1014.7173, 4, 5),
        tuple(
            (role, MetadataColumnSelector(role, 0))
            for role in ("mu", "eta", "chi", "phi", "nu", "del")
        ),
        ImageOrientation(),
        (0, -1, 0, -1),
    )
    plan = RSMOperationPlan(
        geometry,
        RSMImageConditioning(1.0e-6, 2.0e10, 100.0),
        RSMNormalizationPolicy(
            RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
            MetadataColumnSelector("foil status", 0),
            MetadataColumnSelector("Seconds", 1),
            (1.06, 3.04, 4.65, 9.5),
        ),
        bins=(4, 5, 6),
        chunk_size=1,
        max_frame_bytes=1024 * 1024,
        max_chunk_bytes=1024 * 1024,
    )
    return RSMToolForm(
        tmp_path,
        spec_path,
        "1.1",
        image_root,
        "frame_",
        RSMFrameSelector(0, 2, 2),
        (4, 5),
        "uint16",
        plan,
        output_root / "rsm.nexus",
        0,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )


@pytest.fixture
def prepared_rsm_tool(rsm_tool_form):
    return prepare_rsm_tool(rsm_tool_form)
