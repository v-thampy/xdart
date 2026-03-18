from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ssrl_xrd_tools.io.image import (
    read_image,
    read_image_stack,
    read_images_parallel,
    find_image_files,
    get_detector_mask,
    apply_rotation,
)
from ssrl_xrd_tools.io.spec import (
    get_scan_path_info,
    get_energy_and_UB,
    get_angles,
)
from ssrl_xrd_tools.rsm.volume import RSMVolume
from ssrl_xrd_tools.rsm.geometry import DiffractometerConfig
from ssrl_xrd_tools.rsm.gridding import grid_img_data

logger = logging.getLogger(__name__)


def _as_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _load_pickle(path: Path) -> Any | None:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        logger.exception("Failed to load pickle: %s", path)
        return None


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


@dataclass(slots=True)
class ScanInfo:
    spec_path: Path
    img_dir: Path
    h5_path: Path | None = None


@dataclass(slots=True)
class ExperimentConfig:
    """Session-level configuration for an RSM experiment."""

    base_path: Path
    pickle_dir: Path
    header: dict[str, Any]

    img_rel_path: str = "images"
    diff_motors: tuple[str, ...] = ("th", "chi", "phi", "tth")
    diff_config: DiffractometerConfig = field(default_factory=DiffractometerConfig)
    bins: tuple[int, int, int] = (80, 80, 100)
    rotation: int = 0
    h5_glob: str = "{sample}_scan*{scan_num}_master.h5"
    detector: str = "Pilatus300k"

    def __post_init__(self) -> None:
        self.pickle_dir.mkdir(parents=True, exist_ok=True)

    def find_h5(self, spec_dir: Path, sample: str, scan_num: int) -> Path | None:
        pattern = self.h5_glob.format(sample=sample, scan_num=scan_num)
        try:
            return next(spec_dir.glob(pattern))
        except StopIteration:
            return None

    def build_scans(self, scan_dict: dict[str, dict[str, Any]]) -> dict[str, ScanInfo]:
        scans: dict[str, ScanInfo] = {}
        for sample, sdict in scan_dict.items():
            spec_dir = self.base_path / sdict["spec_rel_path"]
            spec_path = spec_dir / sample

            for scan_num in sdict["scan_nums"]:
                h5_file = self.find_h5(spec_dir, sample, scan_num)
                img_dir = spec_dir if h5_file else self.base_path / self.img_rel_path
                scans[f"{sample}_{scan_num}"] = ScanInfo(
                    spec_path=spec_path,
                    h5_path=h5_file,
                    img_dir=img_dir,
                )
        return scans

    def process(
        self,
        scan_name: str,
        scan_info: ScanInfo,
        bins: tuple[int, int, int] | None = None,
        rotation: int | None = None,
        reprocess: bool = False,
        roi: tuple[int, int, int, int] | None = None,
        parallel: bool = True,
        strict: bool = False,
    ) -> RSMVolume | None:
        return process_scan(
            scan_name=scan_name,
            scan_info=scan_info,
            bins=self.bins if bins is None else bins,
            header=self.header,
            diff_motors=self.diff_motors,
            diff_config=self.diff_config,
            pickle_dir=self.pickle_dir,
            rotation=self.rotation if rotation is None else rotation,
            reprocess=reprocess,
            roi=roi,
            parallel=parallel,
            strict=strict,
            detector=self.detector,
        )


def load_images(
    scan_name: str,
    scan_info: ScanInfo,
    rotation: int = 0,
    parallel: bool = True,
    detector: str = "Pilatus300k",
) -> np.ndarray | None:
    """Load image stack for a scan (HDF5 or individual image files)."""
    mask = get_detector_mask(detector)
    if scan_info.h5_path:
        return read_image_stack(
            scan_info.h5_path,
            mask=mask,
            rotation=rotation,
        )
    spec_name, scan_num = get_scan_path_info(scan_name)
    img_files = find_image_files(
        scan_info.img_dir,
        stem=f"_{spec_name}_scan{scan_num[:-2]}_",
    )
    if not img_files:
        logger.warning(
            "No image files found for scan %s in %s",
            scan_name, scan_info.img_dir,
        )
        return None
    if parallel and len(img_files) > 1:
        return read_images_parallel(
            img_files, rotation=rotation, mask=mask,
        )
    return np.stack([
        read_image(f, mask=mask, rotation=rotation)
        for f in img_files
    ])


def process_scan_data(
    scan_name: str,
    scan_info: ScanInfo,
    header: dict[str, Any],
    diff_motors: list[str] | tuple[str, ...],
    diff_config: DiffractometerConfig,
    bins: tuple[int, int, int],
    rotation: int = 0,
    roi: tuple[int, int, int, int] | None = None,
    parallel: bool = True,
    strict: bool = False,
    detector: str = "Pilatus300k",
) -> RSMVolume | None:
    """
    Pure processing path without cache I/O.
    """
    spec_file = scan_info.spec_path
    _, scan_num = get_scan_path_info(scan_name)

    try:
        energy, UB = get_energy_and_UB(spec_file, scan_num)
        angles = get_angles(spec_file, scan_num, diff_motors)
        img_arr = load_images(
            scan_name, scan_info,
            rotation=rotation,
            parallel=parallel,
            detector=detector,
        )

        if img_arr is None or img_arr.size == 0:
            logger.warning("No image data for scan %s", scan_name)
            return None

        return grid_img_data(
            img_arr,
            energy,
            UB,
            angles,
            header,
            diff_config,
            bins=bins,
            roi=roi,
        )
    except Exception:
        logger.exception("Error processing scan %s", scan_name)
        if strict:
            raise
        return None


def process_scan(
    scan_name: str,
    scan_info: ScanInfo,
    bins: tuple[int, int, int],
    header: dict[str, Any],
    diff_motors: list[str] | tuple[str, ...],
    diff_config: DiffractometerConfig,
    pickle_dir: Path,
    rotation: int = 0,
    reprocess: bool = False,
    roi: tuple[int, int, int, int] | None = None,
    parallel: bool = True,
    strict: bool = False,
    detector: str = "Pilatus300k",
) -> RSMVolume | None:
    """
    Process one scan to an RSMVolume, using cache if available.
    """
    sample_name, scan_num = get_scan_path_info(scan_name)
    pickle_file = pickle_dir / f"{sample_name}_{scan_num[:-2]}.pkl"

    if pickle_file.is_file() and not reprocess:
        cached = _load_pickle(pickle_file)
        if isinstance(cached, RSMVolume):
            return cached
        if isinstance(cached, tuple) and len(cached) == 4:
            return RSMVolume(
                h=np.asarray(cached[0]),
                k=np.asarray(cached[1]),
                l=np.asarray(cached[2]),
                intensity=np.asarray(cached[3]),
            )

    volume = process_scan_data(
        scan_name=scan_name,
        scan_info=scan_info,
        header=header,
        diff_motors=diff_motors,
        diff_config=diff_config,
        bins=bins,
        rotation=rotation,
        roi=roi,
        parallel=parallel,
        strict=strict,
        detector=detector,
    )

    if volume is not None:
        _save_pickle(pickle_file, volume)

    return volume
