"""Import-purity and package-API oracles for reciprocal-space mapping."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

EXPECTED_EXPORTS = {
    "DetectorHeader",
    "DiffractometerConfig",
    "PixelQMap",
    "RSMVolume",
    "extract_2d_slice",
    "extract_slice",
    "extract_line_cut",
    "mask_data",
    "save_vtk",
    "StreamingGridder",
    "StreamingScan",
    "combine_grids",
    "get_common_grid",
    "grid_img_data",
    "grid_img_data_streaming",
    "ExperimentConfig",
    "ScanInfo",
    "ScanInput",
    "grid_scans_streaming",
    "load_images",
    "process_scan",
    "process_scan_data",
    "process_scan_from_nexus",
}

EXPORT_OWNERS = {
    "DetectorHeader": ("xrd_tools.core.geometry", "DetectorHeader"),
    "DiffractometerConfig": (
        "xrd_tools.core.geometry",
        "DiffractometerConfig",
    ),
    "PixelQMap": ("xrd_tools.core.geometry", "PixelQMap"),
    "RSMVolume": ("xrd_tools.rsm.volume", "RSMVolume"),
    "extract_2d_slice": ("xrd_tools.rsm.volume", "extract_2d_slice"),
    "extract_slice": ("xrd_tools.rsm.volume", "extract_2d_slice"),
    "extract_line_cut": ("xrd_tools.rsm.volume", "extract_line_cut"),
    "mask_data": ("xrd_tools.rsm.volume", "mask_data"),
    "save_vtk": ("xrd_tools.rsm.volume", "save_vtk"),
    "StreamingGridder": ("xrd_tools.rsm.gridding", "StreamingGridder"),
    "StreamingScan": ("xrd_tools.rsm.gridding", "StreamingScan"),
    "combine_grids": ("xrd_tools.rsm.gridding", "combine_grids"),
    "get_common_grid": ("xrd_tools.rsm.gridding", "get_common_grid"),
    "grid_img_data": ("xrd_tools.rsm.gridding", "grid_img_data"),
    "grid_img_data_streaming": (
        "xrd_tools.rsm.gridding",
        "grid_img_data_streaming",
    ),
    "ExperimentConfig": ("xrd_tools.core.config", "ExperimentConfig"),
    "ScanInfo": ("xrd_tools.rsm.pipeline", "ScanInfo"),
    "ScanInput": ("xrd_tools.rsm.pipeline", "ScanInput"),
    "grid_scans_streaming": (
        "xrd_tools.rsm.pipeline",
        "grid_scans_streaming",
    ),
    "load_images": ("xrd_tools.rsm.pipeline", "load_images"),
    "process_scan": ("xrd_tools.rsm.pipeline", "process_scan"),
    "process_scan_data": ("xrd_tools.rsm.pipeline", "process_scan_data"),
    "process_scan_from_nexus": (
        "xrd_tools.rsm.pipeline",
        "process_scan_from_nexus",
    ),
}

FORBIDDEN_ROOTS = (
    "numpy",
    "scipy",
    "pandas",
    "h5py",
    "fabio",
    "lmfit",
    "matplotlib",
    "xrayutilities",
    "pyevtk",
    "silx",
    "pyFAI",
    "xdart",
    "PySide6",
    "PyQt5",
    "PyQt6",
    "pyqtgraph",
)


def _probe(script: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SRC)
        if not inherited
        else os.pathsep.join((str(SRC), inherited))
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_rsm_package_import_is_dependency_light_and_submodules_are_lazy(
) -> None:
    result = _probe(
        f"""
        import sys
        import xrd_tools.rsm

        forbidden = {FORBIDDEN_ROOTS!r}
        bad = sorted(
            name for name in sys.modules
            if any(
                name == root or name.startswith(root + ".")
                for root in forbidden
            )
        )
        eager = sorted(
            name for name in (
                "xrd_tools.rsm.volume",
                "xrd_tools.rsm.gridding",
                "xrd_tools.rsm.pipeline",
            )
            if name in sys.modules
        )
        assert not bad, bad
        assert not eager, eager
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gridding_import_does_not_load_scipy() -> None:
    result = _probe(
        """
        import sys
        import xrd_tools.rsm.gridding

        bad = sorted(
            name for name in sys.modules
            if name == "scipy" or name.startswith("scipy.")
        )
        assert not bad, bad
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rsm_lazy_api_preserves_public_identity_and_duplicate_resolution(
) -> None:
    import xrd_tools.rsm as rsm

    assert set(rsm.__all__) == EXPECTED_EXPORTS
    assert len(rsm.__all__) == len(set(rsm.__all__)) == 23
    assert not hasattr(rsm, "Any")
    assert not hasattr(rsm, "import_module")
    for name, (module_name, attribute_name) in EXPORT_OWNERS.items():
        assert getattr(rsm, name) is getattr(
            import_module(module_name), attribute_name,
        )
    assert rsm.grid_scans_streaming is not (
        rsm.gridding.grid_scans_streaming
    )


def test_rsm_lazy_submodules_and_concurrent_symbol_cache_are_stable() -> None:
    result = _probe(
        """
        from concurrent.futures import ThreadPoolExecutor
        import sys
        import xrd_tools.rsm as rsm

        assert {"volume", "gridding", "pipeline"}.issubset(dir(rsm))
        assert "xrd_tools.rsm.volume" not in sys.modules
        with ThreadPoolExecutor(max_workers=8) as pool:
            volumes = tuple(
                pool.map(lambda _index: rsm.volume, range(32))
            )
        volume = sys.modules["xrd_tools.rsm.volume"]
        assert all(value is volume for value in volumes)
        assert rsm.__dict__["volume"] is volume

        assert "RSMVolume" not in rsm.__dict__
        with ThreadPoolExecutor(max_workers=8) as pool:
            symbols = tuple(
                pool.map(lambda _index: rsm.RSMVolume, range(32))
            )
        expected = volume.RSMVolume
        assert all(value is expected for value in symbols)
        assert rsm.__dict__["RSMVolume"] is expected

        for name in ("gridding", "pipeline"):
            assert f"xrd_tools.rsm.{name}" not in sys.modules
            module = getattr(rsm, name)
            assert module is sys.modules[f"xrd_tools.rsm.{name}"]
            assert rsm.__dict__[name] is module
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
