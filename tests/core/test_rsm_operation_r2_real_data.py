from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from xdart.gui.tools.rsm_values import (
    RSMFrameSelector,
    RSMScanMemberForm,
    RSMToolFormV2,
    prepare_rsm_tool_v2,
    rsm_tool_preset,
)
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.rsm_geometry_asset import (
    CANONICAL_RSM_GEOMETRY_LOCATOR,
    install_canonical_rsm_geometry_asset,
    rsm_geometry_asset_input,
)
from xrd_tools.analysis.rsm_operation import run_rsm_operation_v2


_SPEC_SHA256 = "12e0d05219cb0309f7fb5f170c8fb12c507e7603dcd71b3954721d1b6e0c90b1"
_ASSET_SHA256 = "0f22b00363ff93fec7b97c7b5c31f8b8e389aac3aae334cbadbc213b629d9f84"
_FULL_MASK_SHA256 = "049be822051c9a992313cf7aa467161527cf72febfd83bf6934db5b3aacab718"
_CROPPED_MASK_SHA256 = "e8b86a058cd9d5137dda258e7e187dd402d6b1d4d1ff4f002678198f649260b9"
_UB = (
    (0.01216395927, -0.006505909096, -1.608423764),
    (1.60834664, 0.01390876234, 0.01210757889),
    (0.01385901527, -1.608409624, 0.006610666882),
)
_MEMBER_Q_BOUNDS = {
    42: (
        (0.9908804064994762, 1.058930535724079),
        (0.848903291069077, 1.068207580136568),
        (2.82039243020986, 3.1864330312948455),
    ),
    43: (
        (0.9991454998430791, 1.0507535379970891),
        (0.8498770429375965, 1.0673826805071531),
        (2.8230899655686454, 3.18407314269762),
    ),
}
_UNION_Q_BOUNDS = _MEMBER_Q_BOUNDS[42]
_SOURCE_FACTS = {
    42: (
        161,
        162,
        61_157_460,
        "af26ecce4cf2d9f2713ad16b20624b439e3d41b8106f89c642d40d6c7129828f",
    ),
    43: (
        61,
        61,
        23_171_460,
        "6a7d8b37cf8e2391ee9195888676acb8f97a3d7d1d8b9f867cae9cbb68d648be",
    ),
}
_SCIENCE = {
    ((43,), 40): {
        "axes": (
            "da4737e621a0f8624048f2599e9b9bda8abf06a4f2ad6c10c62eecd13c2e2255",
            "6a04b8310bf3ac8403ed23f137d6df7478afb92b1d35d9348313876cb642f4dd",
            "cd7a6d27a5525c9ca87644cb4137095ab5a2df9d216a49ef5d251491aecdabc8",
        ),
        "finite": 5_531,
        "intensity": "1f5c827c333a9b919db9491c5155353e0b4a0adf13ccb619652c36443febf9e6",
        "result": "8a88826f85eda95bf9fb4eb9cd043b7b822e0827e4ec3d285203f9256f43b324",
    },
    ((43,), 200): {
        "axes": (
            "fe508d5d7a8cb7689d96a055c551c05d90406aded84390e9a07d10b3b038294b",
            "5585548dfa6fed3b5d0e12cd9d7f4f692ccc6c8c3a69c7566772f6be79333557",
            "9600883671d49f607bcd9595660d34d35497d8d0c1fc35f7d5ce32aa1da4a607",
        ),
        "finite": 502_759,
        "intensity": "33dec932699ebe41e92deec57c658259b9f369778730a18374859afa221749ad",
        "result": "a958981652ad4806354cef0fd9c08a72915cafb039392e828be6793557624653",
    },
    ((42, 43), 40): {
        "axes": (
            "dfcd0ddf2e947f5c483c8371628258d2c02fd5bb2d10413666be211332aab5f7",
            "102192fdd44239f50d31a1fa7ee04ee041c060b0be1fd8275d67d3887595726c",
            "6690352299ea70b49b757e4a5df510fdeb949548d3521b97b0e71160b0d7badb",
        ),
        "finite": 12_777,
        "intensity": "82769a2e5976d6f7201a3f277a6aa0004d2138e92278563f015ccf7140014f39",
        "result": "b80913bcfecac367f90a5319bfa2c6e525110694f67822651c5e9c9830cc5e9c",
    },
    ((42, 43), 200): {
        "axes": (
            "88002ddc692445bf659e040e2b22b845c7b8c06e3bdf944a2b1948ae13a9d40c",
            "0e58e6173754569caf7948c71474d1e54a456c4628c685a67b6bc3cac965809e",
            "13d4d397f39a6cdf67d419a5fb60fed4cc873876131704c951c06f69603d47b0",
        ),
        "finite": 1_420_551,
        "intensity": "f4c98405bc33769eab5d22261b0356385d948ba546d7280956dbf0a372e3bac7",
        "result": "b14a87579a838f3716b4c5428001de929ac0b216b943fe17690ddbf094ea2a77",
    },
}


def _data_root() -> Path:
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured:
        pytest.skip("XDART_TEST_DATA is not configured")
    root = Path(configured) / "RSM"
    if not root.is_dir():
        pytest.skip("authenticated RSM test data is unavailable")
    return root


def _raw_manifest(paths: tuple[Path, ...]) -> str:
    rows = (
        f"{path.name}\0{path.stat().st_size}\0"
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}"
        for path in paths
    )
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def rsm_r2_project(tmp_path_factory) -> Path:
    source = _data_root()
    spec = source / "STO_align"
    assert spec.stat().st_size == 23_411_880
    assert hashlib.sha256(spec.read_bytes()).hexdigest() == _SPEC_SHA256
    project = tmp_path_factory.mktemp("rsm-r2-real-project")
    images = project / "images"
    images.mkdir()
    shutil.copy2(spec, project / spec.name)
    for scan_number, facts in _SOURCE_FACTS.items():
        frame_count, pdi_count, raw_bytes, manifest = facts
        stem = f"b_thampy_STO_align_scan{scan_number}_"
        raws = tuple(sorted((source / "images").glob(f"{stem}*.raw")))
        pdis = tuple(sorted((source / "images").glob(f"{stem}*.raw.pdi")))
        assert len(raws) == frame_count
        assert len(pdis) == pdi_count
        assert sum(path.stat().st_size for path in raws) == raw_bytes
        assert _raw_manifest(raws) == manifest
        for path in (*raws, *pdis):
            shutil.copy2(path, images / path.name)
    asset = install_canonical_rsm_geometry_asset(project_root=project)
    assert asset.raw_sha256 == _ASSET_SHA256
    assert not any(project.glob("*.nexus"))
    return project


def _form(
    project: Path,
    scans: tuple[int, ...],
    bins: int,
) -> RSMToolFormV2:
    preset = rsm_tool_preset()
    members = tuple(
        RSMScanMemberForm(
            project / "STO_align",
            f"{scan}.1",
            project / "images",
            f"b_thampy_STO_align_scan{scan}_",
            RSMFrameSelector(0, _SOURCE_FACTS[scan][0] - 1, 1),
            preset.detector_shape,
            preset.raw_dtype,
            preset.raw_header_skip,
            preset.motor_selectors,
        )
        for scan in scans
    )
    return RSMToolFormV2(
        project,
        rsm_geometry_asset_input(CANONICAL_RSM_GEOMETRY_LOCATOR),
        members,
        preset.conditioning,
        preset.normalization,
        (bins, bins, bins),
        preset.chunk_size,
        preset.max_frame_bytes,
        preset.max_chunk_bytes,
        project / f"rsm-r2-{'-'.join(str(item) for item in scans)}-{bins}.nexus",
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("scans", "bins"),
    tuple(_SCIENCE),
    ids=("scan43-40", "scan43-200", "scan42-43-40", "scan42-43-200"),
)
def test_rsm_v2_matches_independent_authenticated_oracle(
    rsm_r2_project,
    scans,
    bins,
):
    expected = _SCIENCE[(scans, bins)]
    form = _form(rsm_r2_project, scans, bins)
    output = Path(form.output_path)
    assert not output.exists()
    try:
        prepared = prepare_rsm_tool_v2(form)
        request = prepared.request
        expected_frames = tuple(_SOURCE_FACTS[item][0] for item in scans)
        assert tuple(
            item.source_scan for item in request.preflight.members
        ) == tuple(f"{item}.1" for item in scans)
        assert tuple(
            len(item.contributions) for item in request.preflight.members
        ) == expected_frames
        assert tuple(
            item.energy_eV for item in request.preflight.members
        ) == (13_000.007,) * len(scans)
        assert tuple(item.ub for item in request.preflight.members) == (
            _UB,
        ) * len(scans)
        assert tuple(
            item.member_q_bounds for item in request.preflight.members
        ) == tuple(_MEMBER_Q_BOUNDS[item] for item in scans)
        expected_union = (
            _MEMBER_Q_BOUNDS[43] if scans == (43,) else _UNION_Q_BOUNDS
        )
        assert request.plan.common_grid.bounds == expected_union
        result = run_rsm_operation_v2(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED
        assert result.payload is not None
        payload = result.payload
        assert payload.schema_version == 2
        assert payload.inspection.shape == (bins, bins, bins)
        assert payload.result_fingerprint == expected["result"]
        assert tuple(
            _array_sha256(axis) for _name, axis in payload.axes
        ) == expected["axes"]
        assert int(np.isfinite(payload.intensity).sum()) == expected["finite"]
        assert _array_sha256(payload.intensity) == expected["intensity"]
        assert payload.sigma is None
        assert payload.coverage is None
        assert payload.normalization is None
        attestation = json.loads(payload.execution_attestation_json)
        expected_chunks = 8 if scans == (43,) else 29
        assert attestation["selected_scan_count"] == len(scans)
        assert attestation["selected_frame_count"] == sum(expected_frames)
        assert attestation["science_chunk_count"] == expected_chunks
        assert attestation["q_release_check_chunk_count"] == expected_chunks
        assert attestation["frame_release_check_frame_count"] == sum(
            expected_frames
        )
        assert attestation["release_check_passed"] is True
        assert attestation["xu_runtime"]["restore_passed"] is True
        assert [item["ordinal"] for item in attestation["member_masks"]] == list(
            range(len(scans))
        )
        for item in attestation["member_masks"]:
            assert item["mask_policy"] == (
                "exact-all-selected-frames-static-hot-v1"
            )
            assert item["full_shape"] == [195, 487]
            assert item["full_masked_pixel_count"] == 0
            assert item["full_raw_digest"] == _FULL_MASK_SHA256
            assert item["cropped_shape"] == [194, 486]
            assert item["cropped_masked_pixel_count"] == 0
            assert item["cropped_raw_digest"] == _CROPPED_MASK_SHA256
    finally:
        if output.exists():
            output.unlink()
