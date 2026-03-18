from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssrl_xrd_tools.rsm import (
    DiffractometerConfig,
    ExperimentConfig,
    RSMVolume,
    ScanInfo,
    combine_grids,
    extract_2d_slice,
    extract_line_cut,
    get_common_grid,
    grid_img_data,
    mask_data,
    save_vtk,
)

try:
    import xrayutilities as xu

    _HAS_XU = True
except ImportError:
    xu = None
    _HAS_XU = False

try:
    from ssrl_xrd_tools.rsm import volume as volume_module

    _HAS_VTK = bool(volume_module._VTK_AVAILABLE)
except Exception:
    _HAS_VTK = False


@pytest.fixture
def rsm_volume() -> RSMVolume:
    """Small 10x10x10 RSMVolume for testing."""
    h = np.linspace(-1, 1, 10)
    k = np.linspace(-1, 1, 10)
    l = np.linspace(0, 2, 10)
    intensity = np.random.default_rng(42).random((10, 10, 10))
    return RSMVolume(h=h, k=k, l=l, intensity=intensity)


class TestRSMVolume:
    def test_shape(self, rsm_volume: RSMVolume) -> None:
        assert rsm_volume.shape == (10, 10, 10)

    def test_get_bounds(self, rsm_volume: RSMVolume) -> None:
        bounds = rsm_volume.get_bounds()
        assert len(bounds) == 3
        for b in bounds:
            assert len(b) == 3
            assert b[0] <= b[1]

    def test_crop(self, rsm_volume: RSMVolume) -> None:
        cropped = rsm_volume.crop(hrange=(-0.5, 0.5), krange=(-0.5, 0.5), lrange=(0.5, 1.5))
        assert cropped.shape[0] < rsm_volume.shape[0]
        assert cropped.shape[1] < rsm_volume.shape[1]
        assert cropped.shape[2] < rsm_volume.shape[2]
        assert float(np.nanmin(cropped.h)) >= -0.5
        assert float(np.nanmax(cropped.h)) <= 0.5
        assert float(np.nanmin(cropped.k)) >= -0.5
        assert float(np.nanmax(cropped.k)) <= 0.5
        assert float(np.nanmin(cropped.l)) >= 0.5
        assert float(np.nanmax(cropped.l)) <= 1.5

    def test_crop_no_change(self, rsm_volume: RSMVolume) -> None:
        cropped = rsm_volume.crop(
            hrange=(-np.inf, np.inf),
            krange=(-np.inf, np.inf),
            lrange=(-np.inf, np.inf),
        )
        assert cropped.shape == rsm_volume.shape

    def test_line_cut_h(self, rsm_volume: RSMVolume) -> None:
        axis_vals, intensity_1d = rsm_volume.line_cut("h")
        assert len(axis_vals) == 10
        assert intensity_1d.shape == (10,)

    def test_line_cut_with_ranges(self, rsm_volume: RSMVolume) -> None:
        axis_vals, intensity_1d = rsm_volume.line_cut(
            "h",
            fixed_ranges={"k": (-0.2, 0.2), "l": (0.8, 1.2)},
        )
        assert axis_vals.shape == (10,)
        assert intensity_1d.shape == (10,)

    def test_get_slice_k(self, rsm_volume: RSMVolume) -> None:
        axis1, axis2, slice_2d, integrated_vals = rsm_volume.get_slice("k")
        assert axis1.ndim == 1
        assert axis2.ndim == 1
        assert slice_2d.ndim == 2
        assert integrated_vals.ndim == 1

    def test_get_slice_with_range(self, rsm_volume: RSMVolume) -> None:
        _, _, _, full_vals = rsm_volume.get_slice("k")
        _, _, _, ranged_vals = rsm_volume.get_slice("k", val_range=(-0.3, 0.3))
        assert len(ranged_vals) < len(full_vals)

    def test_invalid_axis(self, rsm_volume: RSMVolume) -> None:
        with pytest.raises(ValueError):
            rsm_volume.line_cut("x")
        with pytest.raises(ValueError):
            rsm_volume.get_slice("x")

    def test_intensity_shape_mismatch(self) -> None:
        h = np.linspace(0, 1, 5)
        k = np.linspace(0, 1, 6)
        l = np.linspace(0, 1, 7)
        bad = np.zeros((5, 6, 8), dtype=float)
        with pytest.raises(ValueError):
            RSMVolume(h=h, k=k, l=l, intensity=bad)


class TestMaskData:
    def test_crop_range(self, rsm_volume: RSMVolume) -> None:
        axes, data = mask_data(
            rsm_volume.h,
            rsm_volume.k,
            rsm_volume.l,
            rsm_volume.intensity,
            HRange=(-0.5, 0.5),
            KRange=(-0.5, 0.5),
            LRange=(0.5, 1.5),
        )
        assert data.shape[0] < rsm_volume.shape[0]
        assert data.shape[1] < rsm_volume.shape[1]
        assert data.shape[2] < rsm_volume.shape[2]
        assert axes[0].ndim == axes[1].ndim == axes[2].ndim == 1

    def test_full_range(self, rsm_volume: RSMVolume) -> None:
        axes, data = mask_data(
            rsm_volume.h,
            rsm_volume.k,
            rsm_volume.l,
            rsm_volume.intensity,
            HRange=(-np.inf, np.inf),
            KRange=(-np.inf, np.inf),
            LRange=(-np.inf, np.inf),
        )
        assert data.shape == rsm_volume.shape
        np.testing.assert_allclose(axes[0], rsm_volume.h)
        np.testing.assert_allclose(axes[1], rsm_volume.k)
        np.testing.assert_allclose(axes[2], rsm_volume.l)


class TestExtractLineCut:
    def test_axis_0(self, rsm_volume: RSMVolume) -> None:
        axis_vals, intensity_1d = extract_line_cut(
            rsm_volume.h,
            rsm_volume.k,
            rsm_volume.l,
            rsm_volume.intensity,
            axis=0,
        )
        assert axis_vals.shape == (10,)
        assert intensity_1d.shape == (10,)

    def test_axis_invalid(self, rsm_volume: RSMVolume) -> None:
        with pytest.raises(ValueError):
            extract_line_cut(
                rsm_volume.h,
                rsm_volume.k,
                rsm_volume.l,
                rsm_volume.intensity,
                axis=3,
            )


class TestExtract2dSlice:
    def test_integrate_axis_0(self, rsm_volume: RSMVolume) -> None:
        axis1, axis2, sl, integrated_vals = extract_2d_slice(
            rsm_volume.h,
            rsm_volume.k,
            rsm_volume.l,
            rsm_volume.intensity,
            integrate_axis=0,
        )
        assert axis1.shape == (10,)
        assert axis2.shape == (10,)
        assert sl.shape == (10, 10)
        assert integrated_vals.shape == (10,)

    def test_with_range(self, rsm_volume: RSMVolume) -> None:
        _, _, _, full_vals = extract_2d_slice(
            rsm_volume.h,
            rsm_volume.k,
            rsm_volume.l,
            rsm_volume.intensity,
            integrate_axis=2,
        )
        _, _, _, ranged_vals = extract_2d_slice(
            rsm_volume.h,
            rsm_volume.k,
            rsm_volume.l,
            rsm_volume.intensity,
            integrate_axis=2,
            axis_range=(0.8, 1.2),
        )
        assert len(ranged_vals) < len(full_vals)


class TestSaveVtk:
    def test_save_vtk_not_available(self, rsm_volume: RSMVolume, monkeypatch: pytest.MonkeyPatch) -> None:
        from ssrl_xrd_tools.rsm import volume as volume_module

        monkeypatch.setattr(volume_module, "_VTK_AVAILABLE", False)
        monkeypatch.setattr(volume_module, "gridToVTK", None)

        with pytest.raises(ImportError):
            save_vtk(rsm_volume.intensity, (rsm_volume.h, rsm_volume.k, rsm_volume.l), "dummy")

    @pytest.mark.skipif(not _HAS_VTK, reason="pyevtk not installed")
    def test_save_vtk_real(self, tmp_path: Path, rsm_volume: RSMVolume) -> None:
        out = tmp_path / "vol_test"
        save_vtk(rsm_volume.intensity, (rsm_volume.h, rsm_volume.k, rsm_volume.l), out)
        # pyevtk extension can vary by backend/version; assert any output exists
        matches = list(tmp_path.glob("vol_test*"))
        assert matches


class TestDiffractometerConfig:
    def test_defaults(self) -> None:
        cfg = DiffractometerConfig()
        assert cfg.sample_rot == ("z-", "y+", "z-")
        assert cfg.detector_rot == ("z-",)
        np.testing.assert_allclose(cfg.r_i, (0.0, 1.0, 0.0))

    @pytest.mark.skipif(not _HAS_XU, reason="xrayutilities not installed")
    def test_make_hxrd(self) -> None:
        cfg = DiffractometerConfig()
        hxrd = cfg.make_hxrd(12000.0)
        assert isinstance(hxrd, xu.HXRD)


@pytest.mark.skipif(not _HAS_XU, reason="xrayutilities not installed")
class TestGridImgData:
    def test_returns_rsm_volume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeAng2Q:
            def init_area(self, *args, **kwargs):
                return None

            def area(self, *args, **kwargs):
                shape = (3, 5, 6)
                return (
                    np.zeros(shape, dtype=float),
                    np.zeros(shape, dtype=float),
                    np.zeros(shape, dtype=float),
                )

        class _FakeHXRD:
            Ang2Q = _FakeAng2Q()

        class _FakeGridder3D:
            def __init__(self, nx: int, ny: int, nz: int):
                self.xaxis = np.linspace(-1, 1, nx)
                self.yaxis = np.linspace(-2, 2, ny)
                self.zaxis = np.linspace(0, 3, nz)
                self.data = np.zeros((nx, ny, nz), dtype=float)

            def __call__(self, qx, qy, qz, img):
                self.data = np.full(self.data.shape, float(np.nanmean(img)))

        import ssrl_xrd_tools.rsm.gridding as gridding_module

        monkeypatch.setattr(gridding_module.xu, "Gridder3D", _FakeGridder3D)

        cfg = DiffractometerConfig()
        monkeypatch.setattr(DiffractometerConfig, "make_hxrd", lambda self, energy: _FakeHXRD())

        img = np.random.default_rng(0).random((3, 5, 6))
        UB = np.eye(3)
        angles = [[0.0, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]
        header = {"cch1": "2", "cch2": "3"}

        out = grid_img_data(
            img=img,
            energy=12000.0,
            UB=UB,
            angles=angles,
            header=header,
            diff_config=cfg,
            bins=(8, 9, 10),
        )
        assert isinstance(out, RSMVolume)
        assert out.shape == (8, 9, 10)

    def test_roi_crop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeAng2Q:
            def init_area(self, *args, **kwargs):
                captured["Nch1"] = kwargs.get("Nch1")
                captured["Nch2"] = kwargs.get("Nch2")
                return None

            def area(self, *args, **kwargs):
                shape = (3, captured["Nch1"], captured["Nch2"])
                return (
                    np.zeros(shape, dtype=float),
                    np.zeros(shape, dtype=float),
                    np.zeros(shape, dtype=float),
                )

        class _FakeHXRD:
            Ang2Q = _FakeAng2Q()

        class _FakeGridder3D:
            def __init__(self, nx: int, ny: int, nz: int):
                self.xaxis = np.linspace(-1, 1, nx)
                self.yaxis = np.linspace(-1, 1, ny)
                self.zaxis = np.linspace(-1, 1, nz)
                self.data = np.zeros((nx, ny, nz), dtype=float)

            def __call__(self, qx, qy, qz, img):
                self.data = np.full(self.data.shape, float(np.nanmean(img)))

        import ssrl_xrd_tools.rsm.gridding as gridding_module

        monkeypatch.setattr(gridding_module.xu, "Gridder3D", _FakeGridder3D)

        cfg = DiffractometerConfig()
        monkeypatch.setattr(DiffractometerConfig, "make_hxrd", lambda self, energy: _FakeHXRD())

        img = np.random.default_rng(1).random((3, 10, 10))
        UB = np.eye(3)
        angles = [[0.0, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]
        header = {"cch1": "5", "cch2": "5"}
        roi = (2, 8, 1, 9)  # 6 x 8 cropped image

        out = grid_img_data(
            img=img,
            energy=12000.0,
            UB=UB,
            angles=angles,
            header=header,
            diff_config=cfg,
            bins=(5, 5, 5),
            roi=roi,
        )
        assert isinstance(out, RSMVolume)
        assert captured["Nch1"] == 6
        assert captured["Nch2"] == 8


class TestCombineGrids:
    def test_combine_two_volumes(self) -> None:
        rng = np.random.default_rng(10)
        v1 = RSMVolume(
            h=np.linspace(-1, 0.5, 6),
            k=np.linspace(-1, 1, 6),
            l=np.linspace(0, 2, 6),
            intensity=rng.random((6, 6, 6)),
        )
        v2 = RSMVolume(
            h=np.linspace(-0.5, 1, 5),
            k=np.linspace(-1, 1, 5),
            l=np.linspace(0, 2, 5),
            intensity=rng.random((5, 5, 5)),
        )
        out = combine_grids([v1, v2], bins=(8, 9, 10))
        assert isinstance(out, RSMVolume)
        assert out.shape == (8, 9, 10)

    def test_empty_list(self) -> None:
        with pytest.raises(ValueError):
            combine_grids([], bins=(5, 5, 5))


class TestGetCommonGrid:
    def test_common_grid_bounds(self) -> None:
        v1 = RSMVolume(
            h=np.array([-1.0, 0.0]),
            k=np.array([-2.0, 0.0]),
            l=np.array([0.0, 1.0]),
            intensity=np.ones((2, 2, 2)),
        )
        v2 = RSMVolume(
            h=np.array([0.5, 2.0]),
            k=np.array([-1.0, 3.0]),
            l=np.array([-0.5, 2.5]),
            intensity=np.ones((2, 2, 2)),
        )
        h, k, l = get_common_grid([v1, v2], bins=(4, 5, 6))
        np.testing.assert_allclose(h[[0, -1]], [-1.0, 2.0])
        np.testing.assert_allclose(k[[0, -1]], [-2.0, 3.0])
        np.testing.assert_allclose(l[[0, -1]], [-0.5, 2.5])

    def test_empty_list(self) -> None:
        with pytest.raises(ValueError):
            get_common_grid([], bins=(3, 3, 3))


class TestScanInfo:
    def test_fields(self, tmp_path: Path) -> None:
        spec = tmp_path / "spec.dat"
        img_dir = tmp_path / "images"
        h5 = tmp_path / "scan_master.h5"
        s = ScanInfo(spec_path=spec, img_dir=img_dir, h5_path=h5)
        assert s.spec_path == spec
        assert s.img_dir == img_dir
        assert s.h5_path == h5


class TestExperimentConfig:
    def test_post_init_creates_dir(self, tmp_path: Path) -> None:
        pdir = tmp_path / "pickles"
        assert not pdir.exists()
        ExperimentConfig(base_path=tmp_path, pickle_dir=pdir, header={"cch1": "0", "cch2": "0"})
        assert pdir.exists()

    def test_find_h5(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(base_path=tmp_path, pickle_dir=tmp_path / "pk", header={"cch1": "0", "cch2": "0"})
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()
        expected = spec_dir / "sample_scan12_master.h5"
        expected.write_text("")

        found = cfg.find_h5(spec_dir, "sample", 12)
        assert found == expected


class TestProcessScanData:
    @pytest.mark.skip(reason="requires SPEC data and detector images")
    def test_process_scan_data(self) -> None:
        pass


class TestProcessScan:
    @pytest.mark.skip(reason="requires SPEC data and detector images")
    def test_process_scan(self) -> None:
        pass
