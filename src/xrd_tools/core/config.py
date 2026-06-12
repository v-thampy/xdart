# xrd_tools/core/config.py
"""JSON-loadable experiment configuration bundle.

Holds the per-experiment knobs (paths, detector geometry, diffractometer
convention, binning, output detector) and provides a convenience
``process()`` method that delegates to the RSM pipeline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from xrd_tools.core.geometry import (
    DetectorHeader,
    DiffractometerConfig,
    PixelQMap,
)

if TYPE_CHECKING:
    from xrd_tools.rsm.pipeline import ScanInfo


def _default_header() -> DetectorHeader:
    """Placeholder header — must be overridden before processing."""
    return DetectorHeader(
        cch1=0.0, cch2=0.0,
        pwidth1=0.0, pwidth2=0.0,
        distance=0.0,
        Nch1=0, Nch2=0,
    )


@dataclass
class ExperimentConfig:
    """Standardised JSON-loadable configuration template for experiments."""
    base_path: str = "."
    pickle_dir: str = "pickles"
    header: DetectorHeader = field(default_factory=_default_header)

    img_rel_path: str = "images"
    diff_motors: tuple[str, ...] = ("th", "chi", "phi", "tth")
    diff_config: DiffractometerConfig = field(default_factory=DiffractometerConfig)
    bins: tuple[int, int, int] = (80, 80, 100)
    rotation: int = 0
    h5_glob: str = "{sample}_scan*{scan_num}_master.h5"
    detector: str = "Pilatus300k"

    def __post_init__(self) -> None:
        # Normalise tuple-typed fields after construction.  JSON load
        # (``from_file``) delivers ``diff_motors`` / ``bins`` as lists;
        # without this normalisation a round-tripped config compares
        # unequal to the original (the same trap ``DiffractometerConfig``
        # solves in its own ``__post_init__`` for its five tuple fields).
        if not isinstance(self.diff_motors, tuple):
            self.diff_motors = tuple(self.diff_motors)
        if not isinstance(self.bins, tuple):
            self.bins = tuple(self.bins)
        # Ensure pickle directory exists.
        Path(self.pickle_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def base_path_p(self) -> Path:
        return Path(self.base_path)

    @property
    def pickle_dir_p(self) -> Path:
        return Path(self.pickle_dir)

    @property
    def mapper(self) -> PixelQMap:
        """The :class:`PixelQMap` derived from ``diff_config`` + ``header``."""
        return PixelQMap(self.diff_config, self.header)

    # ------------------------------------------------------------------
    # JSON serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Path | str) -> "ExperimentConfig":
        """Load configuration from a JSON file.

        Nested dataclasses are reconstructed from their JSON dict form.
        Tuple-typed scalar fields (``diff_motors``, ``bins``) are
        normalised by :meth:`__post_init__`; nested
        :class:`DiffractometerConfig` tuple fields are normalised by
        *its* ``__post_init__``.  No tuple-conversion logic lives here.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Parse nested dataclasses.
        if "diff_config" in data and isinstance(data["diff_config"], dict):
            data["diff_config"] = DiffractometerConfig(**data["diff_config"])
        else:
            data.pop("diff_config", None)

        if "header" in data and isinstance(data["header"], dict):
            data["header"] = DetectorHeader(**data["header"])
        else:
            data.pop("header", None)

        return cls(**data)

    def to_file(self, path: Path | str) -> None:
        """Dump configuration to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)

    # ------------------------------------------------------------------
    # Scan-discovery and processing helpers
    # ------------------------------------------------------------------

    def find_h5(self, spec_dir: Path, sample: str, scan_num: int) -> Path | None:
        pattern = self.h5_glob.format(sample=sample, scan_num=scan_num)
        try:
            return next(spec_dir.glob(pattern))
        except StopIteration:
            return None

    def build_scans(self, scan_dict: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Build ``{scan_name: ScanInfo}`` from a nested scan description."""
        from xrd_tools.rsm.pipeline import ScanInfo  # noqa: PLC0415

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
        """Delegate to :func:`xrd_tools.rsm.pipeline.process_scan`."""
        from xrd_tools.rsm.pipeline import process_scan  # noqa: PLC0415

        # Pull defaults from config that aren't supplied.
        kwargs.setdefault("bins", self.bins)
        kwargs.setdefault("rotation", self.rotation)
        kwargs.setdefault("mapper", self.mapper)
        kwargs.setdefault("diff_motors", self.diff_motors)
        kwargs.setdefault("pickle_dir", self.pickle_dir_p)
        kwargs.setdefault("detector", self.detector)

        return process_scan(*args, **kwargs)
