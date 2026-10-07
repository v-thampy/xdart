"""Actual Stanton intensity oracle, independent of cached notebook volumes.

Frozen before science edits: compare the complete published common-grid volume
with explicit notebook HXRD/Gridder3D for both raw and foil/exposure modes.
"""
from pathlib import Path
import hashlib
import os
import shutil

import numpy as np
import pytest
from silx.io.specfile import SpecFile

from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_geometry_input
from xrd_tools.analysis.rsm_operation import RSMNormalizationPolicy, run_rsm_operation_v2
from xrd_tools.analysis.rsm_values import (
    RSMFrameSelector, RSMScanMemberForm, RSMToolFormV2,
    prepare_rsm_tool_v2, rsm_tool_preset,
)
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session


def test_stanton_raw_to_published_volume_matches_explicit_notebook_grid(tmp_path):
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured:
        pytest.skip("XDART_TEST_DATA is not configured")
    source = Path(configured) / "RSM/2026_05_Stanton"
    pins = {
        "STO_align": "12e0d05219cb0309f7fb5f170c8fb12c507e7603dcd71b3954721d1b6e0c90b1",
        "stanton_calib_270_pphi.json": "4ed8ecec1dc95683c616d336724d6bd93bf7fde31de79164c3049cacf2fbadae",
        "stanton_calib_270_pphi.xu.json": "1bdb31daf6395845ba11e768b8efc1661f8a9f27d6f8d86a30f85ae7ab18b6a6",
    }
    if not all((source / name).is_file() for name in pins):
        pytest.skip("Stanton source and saved calibration pair are unavailable")
    for name, digest in pins.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
        shutil.copy2(source / name, tmp_path / name)
    images = tmp_path / "images"
    images.mkdir()
    preset = rsm_tool_preset()
    members, reference = [], []
    spec = SpecFile(str(tmp_path / "STO_align"))
    for number, count in ((1194, 61), (1195, 101), (1196, 61)):
        stem = f"b_thampy_STO_align_scan{number}_"
        paths = sorted((source / "images").glob(f"{stem}*.raw"))
        assert len(paths) == count
        raw = []
        for path in paths:
            assert path.stat().st_size == 195 * 487 * 4
            shutil.copy2(path, images / path.name)
            raw.append(np.fromfile(path, dtype="<i4").reshape(195, 487))
        data = np.asarray(raw, dtype=np.float64) + 1e-6
        data[data > 2e10] = np.nan
        hot = np.all(data > 100, axis=0) & np.all(data == data[0], axis=0)
        data[:, hot] = np.nan
        scan = spec[f"{number}.1"]
        angles = tuple(scan.data_column_by_name(role) if role in scan.labels else
                       np.full(count, scan.motor_position_by_name(role))
                       for role in ("mu", "eta", "chi", "phi", "nu", "del"))
        foil = scan.data_column_by_name("foil status")
        digits = np.array([[int(c) for c in str(int(v)).zfill(4)] for v in foil])
        divisor = np.exp(-digits @ np.array([1.06, 3.04, 4.65, 9.5]))
        divisor *= scan.data_column_by_name("Seconds")
        ub = np.array(scan.scan_header_dict["G3"].split(), dtype=float).reshape(3, 3)
        reference.append((data[:, :-1, :-1], angles, scan.motor_position_by_name("energy"), ub, divisor))
        print(number, "hot pixels", np.argwhere(hot).tolist(), "divisors", np.unique(divisor))
        members.append(RSMScanMemberForm(tmp_path / "STO_align", f"{number}.1", images,
            stem, RSMFrameSelector(0, count - 1, 1), (195, 487), "<i4", 0,
            preset.motor_selectors))
    for name, normalization in (("none", RSMNormalizationPolicy.identity()),
                                 ("foil-exposure", preset.normalization)):
        form = RSMToolFormV2(tmp_path, rsm_calibration_geometry_input("stanton_calib_270_pphi.json"),
            tuple(members), preset.conditioning, normalization, (200, 200, 200),
            preset.chunk_size, preset.max_frame_bytes, None, tmp_path / f"{name}.nexus",
            corrections=None)
        request = prepare_rsm_tool_v2(form).request
        for member, ref in zip(request.preflight.members, reference, strict=True):
            expected_divisors = np.ones(len(ref[4])) if name == "none" else ref[4]
            np.testing.assert_allclose([c.normalization_divisor for c in member.contributions],
                                       expected_divisors, rtol=1e-14, atol=0)
        with xu_runtime_session(request.plan.effective_geometry.runtime_requirements) as session:
            xu, _ = session.active_modules()
            grid = xu.Gridder3D(200, 200, 200)
            grid.KeepData(True)
            grid.dataRange(*(v for pair in request.plan.common_grid.bounds for v in pair), fixed=True)
            for data, angles, energy, ub, divisors in reference:
                conversion = xu.QConversion(["x+", "z-", "y+", "z-"], ["x+", "z-"], [0, 1, 0])
                hxrd = xu.HXRD([0, 1, 0], [0, 0, 1], geometry="real", en=energy, qconv=conversion)
                hxrd.Ang2Q.init_area("z-", "x-", cch1=97, cch2=243, pwidth1=.172,
                                    pwidth2=.172, distance=1014.7173, Nch1=194, Nch2=486)
                q = hxrd.Ang2Q.area(*angles, UB=ub)
                signal = data if name == "none" else data / divisors[:, None, None]
                grid(*q, signal)
            expected = grid.data.copy()
            # XU's empty bins are zero; all valid native counts have a nonzero offset.
            expected[expected == 0] = np.nan
        result = run_rsm_operation_v2(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        actual = result.payload.intensity
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1e-6, atol=0, equal_nan=True)
        hk = np.nanmean(actual, axis=2)
        axes = tuple(axis for _, axis in result.payload.axes)
        h, k = axes[:2]
        central = hk[np.ix_(abs(h - 1.505) < .006, abs(k + 2.50) < .012)]
        print(name, "finite", int(np.isfinite(actual).sum()), "central HK median",
              float(np.nanmedian(central)), "HK percentiles50/99.6",
              np.nanpercentile(hk, [50, 99.6]), "full-volume max error",
              float(np.nanmax(abs(actual - expected.astype(np.float32)))))
