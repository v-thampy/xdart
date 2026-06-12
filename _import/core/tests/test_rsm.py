from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssrl_xrd_tools.rsm import (
    DetectorHeader,
    DiffractometerConfig,
    ExperimentConfig,
    PixelQMap,
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


def _default_header(Nch1: int = 0, Nch2: int = 0) -> DetectorHeader:
    """Tiny test header — geometry values are arbitrary placeholders."""
    return DetectorHeader(
        cch1=5.0, cch2=5.0,
        pwidth1=0.075, pwidth2=0.075,
        distance=830.0,
        Nch1=Nch1, Nch2=Nch2,
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
        monkeypatch.setattr(DiffractometerConfig, "make_hxrd", lambda self, energy: _FakeHXRD())

        mapper = PixelQMap(
            diff_config=DiffractometerConfig(),
            header=_default_header(Nch1=5, Nch2=6),
        )
        img = np.random.default_rng(0).random((3, 5, 6))
        UB = np.eye(3)
        angles = [[0.0, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]

        out = grid_img_data(
            mapper,
            img,
            angles,
            energy=12000.0,
            UB=UB,
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
                captured["cch1"] = kwargs.get("cch1")
                captured["cch2"] = kwargs.get("cch2")
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
        monkeypatch.setattr(DiffractometerConfig, "make_hxrd", lambda self, energy: _FakeHXRD())

        mapper = PixelQMap(
            diff_config=DiffractometerConfig(),
            header=_default_header(Nch1=10, Nch2=10),
        )
        img = np.random.default_rng(1).random((3, 10, 10))
        UB = np.eye(3)
        angles = [[0.0, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]
        roi = (2, 8, 1, 9)  # 6 x 8 cropped image

        out = grid_img_data(
            mapper,
            img,
            angles,
            energy=12000.0,
            UB=UB,
            bins=(5, 5, 5),
            roi=roi,
        )
        assert isinstance(out, RSMVolume)
        assert captured["Nch1"] == 6
        assert captured["Nch2"] == 8
        # ROI also shifts the beam centre: cch1 - r0, cch2 - c0
        assert captured["cch1"] == pytest.approx(5.0 - 2)
        assert captured["cch2"] == pytest.approx(5.0 - 1)


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
        ExperimentConfig(
            base_path=str(tmp_path),
            pickle_dir=str(pdir),
            header=_default_header(),
        )
        assert pdir.exists()

    def test_find_h5(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            base_path=str(tmp_path),
            pickle_dir=str(tmp_path / "pk"),
            header=_default_header(),
        )
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()
        expected = spec_dir / "sample_scan12_master.h5"
        expected.write_text("")

        found = cfg.find_h5(spec_dir, "sample", 12)
        assert found == expected

    def test_mapper_property_bundles_diff_and_header(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            base_path=str(tmp_path),
            pickle_dir=str(tmp_path / "pk"),
            header=_default_header(Nch1=100, Nch2=200),
            diff_config=DiffractometerConfig(),
        )
        mapper = cfg.mapper
        assert isinstance(mapper, PixelQMap)
        assert mapper.header.Nch1 == 100
        assert mapper.header.Nch2 == 200
        assert mapper.diff_config is cfg.diff_config

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            base_path=str(tmp_path),
            pickle_dir=str(tmp_path / "pk"),
            header=DetectorHeader(
                cch1=257.5, cch2=515.0,
                pwidth1=0.075, pwidth2=0.075,
                distance=830.0,
                Nch1=514, Nch2=1030,
            ),
        )
        path = tmp_path / "exp.json"
        cfg.to_file(path)
        loaded = ExperimentConfig.from_file(path)
        assert loaded.header == cfg.header
        assert loaded.diff_config == cfg.diff_config
        assert loaded.bins == cfg.bins


class TestProcessScanData:
    @pytest.mark.skip(reason="requires SPEC data and detector images")
    def test_process_scan_data(self) -> None:
        pass

    def test_p3_streaming_default_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P3 regression: process_scan_data defaults streaming=True so
        the public SPEC/HDF5 path inherits the memory-bounded gridder."""
        from ssrl_xrd_tools.rsm import pipeline as pipeline_module

        calls: list[str] = []

        def _fake_streaming(*args: Any, **kwargs: Any) -> RSMVolume:
            calls.append("streaming")
            return RSMVolume(
                h=np.zeros(2), k=np.zeros(2), l=np.zeros(2),
                intensity=np.zeros((2, 2, 2)),
            )

        def _fake_single_shot(*args: Any, **kwargs: Any) -> RSMVolume:
            calls.append("single_shot")
            return RSMVolume(
                h=np.zeros(2), k=np.zeros(2), l=np.zeros(2),
                intensity=np.zeros((2, 2, 2)),
            )

        monkeypatch.setattr(pipeline_module, "grid_img_data_streaming",
                            _fake_streaming)
        monkeypatch.setattr(pipeline_module, "grid_img_data", _fake_single_shot)
        # Short-circuit the SPEC + image-load chain
        monkeypatch.setattr(pipeline_module, "get_energy_and_UB",
                            lambda *a, **k: (12000.0, np.eye(3)))
        monkeypatch.setattr(pipeline_module, "get_angles",
                            lambda *a, **k: [np.array([0.0, 0.1])])
        monkeypatch.setattr(pipeline_module, "load_images",
                            lambda *a, **k: np.zeros((2, 4, 4), dtype=float))
        monkeypatch.setattr(pipeline_module, "get_scan_path_info",
                            lambda name: (name, "1.1"))

        mapper = PixelQMap(
            diff_config=DiffractometerConfig(),
            header=DetectorHeader(
                cch1=2.0, cch2=2.0, pwidth1=0.075, pwidth2=0.075,
                distance=830.0, Nch1=4, Nch2=4,
            ),
        )
        scan_info = ScanInfo(spec_path=Path("/tmp/x"), img_dir=Path("/tmp/y"))

        # Default call — should pick streaming.
        pipeline_module.process_scan_data(
            "test_scan", scan_info, mapper,
            diff_motors=("tth",), bins=(2, 2, 2),
        )
        assert calls == ["streaming"]

        # Explicit streaming=False — should fall back to single-shot.
        calls.clear()
        pipeline_module.process_scan_data(
            "test_scan", scan_info, mapper,
            diff_motors=("tth",), bins=(2, 2, 2),
            streaming=False,
        )
        assert calls == ["single_shot"]


class TestProcessScan:
    @pytest.mark.skip(reason="requires SPEC data and detector images")
    def test_process_scan(self) -> None:
        pass

    def test_p3_chunk_size_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``chunk_size=`` on process_scan must reach grid_img_data_streaming."""
        from ssrl_xrd_tools.rsm import pipeline as pipeline_module

        captured: dict[str, Any] = {}

        def _fake_streaming(*args: Any, **kwargs: Any) -> RSMVolume:
            captured.update(kwargs)
            return RSMVolume(
                h=np.zeros(2), k=np.zeros(2), l=np.zeros(2),
                intensity=np.zeros((2, 2, 2)),
            )

        monkeypatch.setattr(pipeline_module, "grid_img_data_streaming",
                            _fake_streaming)
        monkeypatch.setattr(pipeline_module, "get_energy_and_UB",
                            lambda *a, **k: (12000.0, np.eye(3)))
        monkeypatch.setattr(pipeline_module, "get_angles",
                            lambda *a, **k: [np.array([0.0, 0.1])])
        monkeypatch.setattr(pipeline_module, "load_images",
                            lambda *a, **k: np.zeros((2, 4, 4), dtype=float))
        monkeypatch.setattr(pipeline_module, "get_scan_path_info",
                            lambda name: (name, "1.1"))

        mapper = PixelQMap(
            diff_config=DiffractometerConfig(),
            header=DetectorHeader(
                cch1=2.0, cch2=2.0, pwidth1=0.075, pwidth2=0.075,
                distance=830.0, Nch1=4, Nch2=4,
            ),
        )
        scan_info = ScanInfo(spec_path=Path("/tmp/x"), img_dir=Path("/tmp/y"))

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pipeline_module.process_scan(
                "test_scan", scan_info,
                bins=(2, 2, 2), mapper=mapper,
                diff_motors=("tth",),
                pickle_dir=Path(tmp),
                chunk_size=4,
                reprocess=True,
            )
        assert captured.get("chunk_size") == 4
