# ssrl_xrd_tools/core/config.py
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ssrl_xrd_tools.rsm.pipeline import ScanInfo

from ssrl_xrd_tools.core.geometry import DiffractometerConfig

@dataclass
class ExperimentConfig:
    """Standardized JSON-loadable configuration template for experiments."""
    base_path: str = "."
    pickle_dir: str = "pickles"
    header: dict[str, Any] = field(default_factory=dict)
    
    img_rel_path: str = "images"
    diff_motors: tuple[str, ...] = ("th", "chi", "phi", "tth")
    diff_config: DiffractometerConfig = field(default_factory=DiffractometerConfig)
    bins: tuple[int, int, int] = (80, 80, 100)
    rotation: int = 0
    h5_glob: str = "{sample}_scan*{scan_num}_master.h5"
    detector: str = "Pilatus300k"

    def __post_init__(self) -> None:
        # Ensure pickle directory exists (matches original ExperimentConfig contract)
        Path(self.pickle_dir).mkdir(parents=True, exist_ok=True)

    @property
    def base_path_p(self) -> Path:
        return Path(self.base_path)
        
    @property
    def pickle_dir_p(self) -> Path:
        return Path(self.pickle_dir)

    @classmethod
    def from_file(cls, path: Path | str) -> "ExperimentConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Parse nested dataclasses natively
        if "diff_config" in data and isinstance(data["diff_config"], dict):
            data["diff_config"] = DiffractometerConfig(**data["diff_config"])
        else:
            data.pop("diff_config", None) # Fallback to default if not properly formed

        # Handle tuple conversions
        if "diff_motors" in data and isinstance(data["diff_motors"], list):
            data["diff_motors"] = tuple(data["diff_motors"])
        if "bins" in data and isinstance(data["bins"], list):
            data["bins"] = tuple(data["bins"])
            
        return cls(**data)
        
    def to_file(self, path: Path | str) -> None:
        """Dump configuration to a JSON file."""
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def find_h5(self, spec_dir: Path, sample: str, scan_num: int) -> Path | None:
        pattern = self.h5_glob.format(sample=sample, scan_num=scan_num)
        try:
            return next(spec_dir.glob(pattern))
        except StopIteration:
            return None

    def build_scans(self, scan_dict: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """
        Builds ScanInfo mappings. 
        Returns dict[str, ScanInfo] but relies on rsm pipeline natively.
        """
        from ssrl_xrd_tools.rsm.pipeline import ScanInfo
        
        scans: dict[str, ScanInfo] = {}
        for sample, sdict in scan_dict.items():
            spec_dir = self.base_path_p / sdict["spec_rel_path"]
            spec_path = spec_dir / sample

            for scan_num in sdict["scan_nums"]:
                h5_file = self.find_h5(spec_dir, sample, scan_num)
                img_dir = spec_dir if h5_file else self.base_path_p / self.img_rel_path
                scans[f"{sample}_{scan_num}"] = ScanInfo(
                    spec_path=spec_path,
                    h5_path=h5_file,
                    img_dir=img_dir,
                )
        return scans

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """
        Delegates processing to the RSM pipeline.
        """
        from ssrl_xrd_tools.rsm.pipeline import process_scan
        
        # Pull defaults from config that aren't supplied
        if "bins" not in kwargs: kwargs["bins"] = self.bins
        if "rotation" not in kwargs: kwargs["rotation"] = self.rotation
        
        return process_scan(
            header=self.header,
            diff_motors=self.diff_motors,
            diff_config=self.diff_config,
            pickle_dir=self.pickle_dir_p,
            detector=self.detector,
            *args, 
            **kwargs
        )
