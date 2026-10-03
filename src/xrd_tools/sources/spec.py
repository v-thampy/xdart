# -*- coding: utf-8 -*-
"""SPEC FrameSource — the primary scan-definition source.

A scan is ``(SPEC file, scan number)``: the SPEC file gives the per-point
metadata (the ``#L`` columns + the constant ``#O``/``#P`` motors), and — when an
image directory is supplied — the matching detector image files give the raw
frames (the RSM ``ScanInfo(spec_path, img_dir)`` pattern, stitching design §5.4).

* **Metadata only** (no ``image_dir``): ``load_frame`` raises, so ROI stats stay
  disabled — but every column plots.
* **With images** (``image_dir`` + the scan's filename stem): ``load_frame``
  reads the raw frame, so ROI / stitch / RSM can reduce it.

All SPEC parsing + image I/O is reused from the headless ``xrd_tools.io`` layer
(``io.spec`` / ``io.image``); this is the thin :class:`FrameSource` wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from xrd_tools.core.scan import (
    ScanFrame,
    SourceCapabilities,
    SourceKind,
    SourceSpec,
)
from xrd_tools.sources.base import BaseFrameSource


def find_spec_image_files(directory: str | Path, spec_stem: str, scan: Any,
                          image_stem: str | None = None) -> list[Path]:
    """Discover SPEC detector inputs, allowing zero-padded automatic scan tokens.

    Explicit stems retain ``find_image_files`` substring/glob semantics. Eiger
    data shards are omitted when their same-prefix master is in the directory.
    Discovery uses filenames only; the existing image reader owns content.
    """
    import re
    from xrd_tools.io.image import find_image_files

    files = find_image_files(directory, stem=image_stem or None)
    if not image_stem:
        files = [path for path in files if _spec_scan_match(path, spec_stem, scan) is not None]
    files = [path for path in files if path.is_file()]
    return [path for path in files if not (
        (match := re.fullmatch(r"(.+)_data_\d+", path.stem))
        and path.suffix.lower() == ".h5"
        and path.with_name(match[1] + "_master.h5").is_file())]


def _spec_scan_match(path: str | Path, spec_stem: str, scan: Any):
    import re
    pattern = re.compile(re.escape(spec_stem) + r"_scan(\d+)_")
    scan_number = int(str(scan).split(".")[0])
    return next((match for match in pattern.finditer(Path(path).name)
                 if int(match[1]) == scan_number), None)


def spec_image_stem(path: str | Path, spec_stem: str, scan: Any) -> str:
    """Return a discovered input's automatic SPEC prefix, including its padding."""
    match = _spec_scan_match(path, spec_stem, scan)
    if match is None:
        raise ValueError("input filename does not match the SPEC scan")
    return match[0]


def resolve_spec_image_scan(path: str | Path, *, meta_dir: str | Path | None = None) -> tuple[Path, str] | None:
    """Resolve a numbered image's actual SPEC scan through the metadata reader.

    Auto sidecars remain independent metadata; only a successfully read SPEC
    point can establish grouping. The reader owns its existing folder search.
    """
    from xrd_tools.io.metadata import _extract_scan_info, read_image_metadata_observed
    from xrd_tools.io.spec import is_spec_file

    _stem, scan, _image = _extract_scan_info(path)
    if scan is None:
        return None
    observed = read_image_metadata_observed(path, "spec", meta_dir=meta_dir)
    if not observed.values or observed.source_path is None or not is_spec_file(observed.source_path):
        return None
    return observed.source_path, f"{scan}.1"


class SpecSource(BaseFrameSource):
    """FrameSource over ONE scan of a SPEC file (metadata; raw images optional).

    Parameters
    ----------
    path
        The (typically extensionless) SPEC file.
    scan
        Scan selector — ``"1.1"`` / ``1`` / ``"1"``; default = the first scan.
    image_dir
        Directory of the detector image files.  When given, the scan's frames
        become loadable raw images (ROI/stitch/RSM); omitted ⇒ metadata only.
    image_stem
        Filename substring selecting this scan's images (default the SSRL
        convention ``_{spec_stem}_scan{N}_``).
    read_image_kwargs
        Passed to :func:`xrd_tools.io.image.read_image` (e.g. ``detector_shape``
        / ``raw_dtype`` / ``raw_header_skip`` for headerless raw binaries).
    """

    kind = SourceKind.SPEC

    def __init__(self, path: str | Path, *, scan: Any = None,
                 name: str | None = None, image_dir: str | Path | None = None,
                 image_stem: str | None = None,
                 require_complete_images: bool = False,
                 read_image_kwargs: Mapping[str, Any] | None = None,
                 point_images: Mapping[str, Any] | None = None) -> None:
        from xrd_tools.io.spec import (
            get_spec_scanned_axes,
            list_spec_scans,
            read_spec_scan_columns,
        )

        self.path = Path(path)
        self.available_scans = list_spec_scans(self.path)
        self.scan_key = self._resolve_scan_key(self.available_scans, scan)
        if require_complete_images and str(scan) not in {self.scan_key, str(self.scan_key).removesuffix(".1")}:
            raise ValueError(f"SPEC scan {scan!r} is not present")

        columns: dict[str, np.ndarray] = {}
        motors: dict[str, float] = {}
        physical_columns: tuple[tuple[str, np.ndarray], ...] = ()
        npts = 0
        if self.scan_key is not None:
            physical_columns, motor_positions, npts = read_spec_scan_columns(
                self.path, self.scan_key
            )
            # Compatibility mapping: a Mapping cannot carry repeated physical
            # names, so retain the first occurrence just as silx's
            # data_column_by_name historically did here.
            for column_name, values in physical_columns:
                columns.setdefault(column_name, values)
            motors = dict(motor_positions)
        self._columns = columns
        self._motors = motors
        self._physical_columns = physical_columns
        self._scanned_axes = (
            get_spec_scanned_axes(self.path, self.scan_key)
            if self.scan_key is not None else ()
        )

        # Optional raw images — located once by stem in image_dir.  The matched
        # files may be one-file-per-frame (raw / tiff / edf / cbf) OR a single
        # multi-frame container (Eiger master h5), so build an explicit
        # (file, frame-in-file) map and read via io.image.read_image (format-
        # agnostic).
        self.image_dir = Path(image_dir) if image_dir else None
        self._read_image_kwargs = dict(read_image_kwargs or {})
        from xrd_tools.sources.point_images import PointImagePolicy
        self.point_image_policy = (None if point_images is None else
                                   PointImagePolicy.from_dict(point_images))
        source_options = dict(scan=self.scan_key, image_dir=str(self.image_dir) if self.image_dir else None,
                              image_stem=image_stem, require_complete_images=require_complete_images,
                              read_image_kwargs=self._read_image_kwargs)
        if self.point_image_policy is not None:
            source_options["point_images"] = self.point_image_policy.to_dict()
        source_spec = SourceSpec(self.path, SourceKind.SPEC, options=source_options)
        self._frame_map: list[tuple[Path, int]] = []
        if self.image_dir is not None and self.scan_key is not None:
            stem = image_stem or self._default_image_stem(self.scan_key)
            files = find_spec_image_files(self.image_dir, self.path.stem, self.scan_key, image_stem)
            if (require_complete_images or self.point_image_policy is not None) and files:
                from xrd_tools.io.image import read_detector_image_layout
                layouts = [read_detector_image_layout(path, **{
                    key: value for key, value in self._read_image_kwargs.items()
                    if key in {"detector_shape", "raw_dtype", "raw_header_skip"}
                }) for path in files]
                frames = sum(layout.frame_count for layout in layouts)
                if self.point_image_policy is not None:
                    if npts <= 0 or frames != npts * self.point_image_policy.images_per_point:
                        raise ValueError("confirmed images per point does not match image and point counts")
                elif frames != npts:  # a ratio is a suggestion, never an implicit mapping
                    from xrd_tools.sources.point_images import detect_point_images, PointImagesRequired
                    detection = detect_point_images(source_spec, self.scan_key, npts, frames)
                    if detection is not None:
                        raise PointImagesRequired(detection)
                    raise ValueError(
                        f"SPEC scan {self.scan_key}: {npts} metadata points but {frames} image frames in "
                        f"{len(files)} files matching *{stem}* in {self.image_dir}; check the image stem and folder")
                if len(files) > 1 and all(layout.frame_count == 1 for layout in layouts):
                    import re
                    positions = [re.search(r"(\d+)(?:\.[^.]+)?$", path.name) for path in files]
                    if all(positions):
                        numbers = [int(match[1]) for match in positions]
                        if numbers not in [list(range(frames)), list(range(1, frames + 1))]:
                            raise ValueError(f"SPEC scan {self.scan_key}: image point numbers contain gaps or duplicates")
            if (require_complete_images or self.point_image_policy is not None) and files:
                self._frame_map = [(path, i) for path, layout in zip(files, layouts)
                                   for i in range(layout.frame_count)]
            elif len(files) == 1 and npts and npts > 1:
                # one multi-frame container -> frame index within it
                self._frame_map = [(files[0], i) for i in range(npts)]
            else:
                self._frame_map = [(f, 0) for f in files]
        if self.point_image_policy is not None and not self._frame_map:
            raise ValueError("confirmed point image grouping requires detector images and scan points")
        has_raw = bool(self._frame_map)

        # Frame count: the metadata point count is authoritative; with one file
        # per frame, clamp to the images actually found.
        if has_raw and npts:
            n_frames = (npts if len(self._frame_map) >= npts
                        else len(self._frame_map))
        elif has_raw:
            n_frames = len(self._frame_map)
        else:
            n_frames = npts

        super().__init__(
            name=name or self._default_name(),
            frame_indices=list(range(n_frames)),
            spec=source_spec,
            capabilities=SourceCapabilities(
                supports_random_access=True,
                supports_chunks=has_raw,
                has_metadata=True,
                has_raw_references=has_raw,
                has_scan_manifest=True,
            ),
        )

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _resolve_scan_key(keys, scan):
        """Pick the scan key — ``scan`` matched exactly or as ``N`` → ``N.1``;
        default the first scan.  ``None`` when the file has no scans."""
        if not keys:
            return None
        if scan is None:
            return keys[0]
        s = str(scan)
        if s in keys:
            return s
        if f"{s}.1" in keys:
            return f"{s}.1"
        return keys[0]

    def _default_name(self):
        return (f"{self.path.stem} [{self.scan_key}]"
                if self.scan_key else self.path.stem)

    def _default_image_stem(self, scan_key):
        """The image-filename substring for this scan: ``{spec_stem}_scan{N}_``.
        The trailing ``_`` anchors the scan number so ``scan5`` does not also
        match ``scan50``; override via ``image_stem`` for other conventions."""
        scan_n = str(scan_key).split(".")[0]
        return f"{self.path.stem}_scan{scan_n}_"

    # ---- FrameSource API ------------------------------------------------
    @property
    def motors(self) -> dict[str, np.ndarray]:
        """The legacy name-keyed ``#L`` mapping view."""

        return dict(self._columns)

    @property
    def physical_metadata_columns(self) -> tuple[tuple[str, np.ndarray], ...]:
        """Ordered physical ``#L`` columns, including repeated names.

        Fixed motors remain scalar in :attr:`fixed_motor_positions`; exact
        table projection broadcasts only the requested ones, avoiding an eager
        all-motors allocation for large scans.
        """

        size = len(self._frame_indices)
        return tuple(
            (name, np.asarray(values[:size]))
            for name, values in self._physical_columns
        )

    @property
    def fixed_motor_positions(self) -> tuple[tuple[str, float], ...]:
        """Ordered scalar ``#O``/``#P`` motor positions."""

        return tuple(self._motors.items())

    def motor_series(self, name: str) -> np.ndarray:
        """Resolve one motor lazily, preferring its per-point ``#L`` values."""

        if type(name) is not str or not name:
            raise KeyError(name)
        size = len(self._frame_indices)
        if name in self._columns:
            return np.asarray(self._columns[name][:size])
        if name in self._motors:
            return np.full(size, self._motors[name], dtype=float)
        raise KeyError(name)

    @property
    def scanned_axes(self) -> tuple[str, ...]:
        """Per-point axes declared by ``#S``, in command order.

        Unlike :attr:`motors`, this excludes counters and is therefore the
        authoritative default-axis hint for plotting, stitching, and RSM.
        """
        return self._scanned_axes

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        i = int(index)
        md: dict[str, Any] = dict(self._motors)        # constant scan-start motors
        md.update({k: v[i] for k, v in self._columns.items() if i < len(v)})
        if self.point_image_policy is not None:
            from xrd_tools.sources.point_images import point_image_metadata
            md["point_images"] = point_image_metadata(self.point_image_policy, self.point_raw_locators(i))
        return md

    def load_frame(self, index: int) -> np.ndarray:
        if self.point_image_policy is not None:
            return self.load_point_frame(index)
        if not self._frame_map:
            raise NotImplementedError(
                "SPEC source has no image directory — metadata only; pass "
                "image_dir to enable raw frames (ROI/stitch/RSM)")
        path, frame = self._frame_map[int(index)]
        from xrd_tools.io.image import read_image
        return np.asarray(
            read_image(path, frame=frame, **self._read_image_kwargs),
            dtype=float)

    def load_native_frame(self, index: int) -> np.ndarray:
        """Decode once with native dtype for the shared conditioning owner."""
        if self.point_image_policy is not None:
            return self.load_point_frame(index)
        path, frame = self.raw_locator_for(index)
        if path is None:
            raise NotImplementedError("SPEC source has no detector images")
        from xrd_tools.io.image import read_image
        options = {key: value for key, value in self._read_image_kwargs.items()
                   if key not in {"threshold", "mask", "preserve_dtype", "exact_frame"}}
        return read_image(path, frame=frame, preserve_dtype=True, exact_frame=True, **options)

    def point_raw_locators(self, index: int) -> tuple[tuple[Path, int], ...]:
        """The exact raw contributors to one logical point, in acquisition order."""
        if type(index) is not int or index not in range(len(self._frame_indices)):
            raise IndexError("scan point index is outside the source")
        if not self._frame_map:
            return ()
        policy = self.point_image_policy
        labels = policy.raw_indices(index) if policy is not None else (index,)
        return tuple(self._frame_map[raw] for raw in labels)

    def load_point_statistics(self, index: int, conditioning=None, saturation=None, *,
                              poisson=False, native_transform=None):
        """Apply value rules to each native input before the confirmed operation."""
        from xrd_tools.sources.point_images import combine_point_statistics, PointImagePolicy
        from xrd_tools.io.image import read_image
        policy = self.point_image_policy
        if policy is None:
            return combine_point_statistics(lambda _: self.load_native_frame(index), (0,),
                PointImagePolicy(1, 'select'), conditioning, saturation,
                poisson=poisson, native_transform=native_transform)
        locators = self.point_raw_locators(index)
        options = {key: value for key, value in self._read_image_kwargs.items()
                   if key not in {"threshold", "mask", "preserve_dtype", "exact_frame"}}
        def native(position):
            path, frame = locators[position]
            return read_image(path, frame=frame, preserve_dtype=True, exact_frame=True, **options)
        return combine_point_statistics(native, range(len(locators)), policy, conditioning, saturation,
                                        poisson=poisson, native_transform=native_transform)

    def load_point_frame(self, index: int, conditioning=None, saturation=None, *,
                         native_transform=None) -> np.ndarray:
        return self.load_point_statistics(index, conditioning, saturation,
                                         native_transform=native_transform).signal

    def raw_locator_for(self, index: int) -> tuple[Path | None, int | None]:
        """Return one raw member locator without constructing frame metadata."""

        i = int(index)
        if not self._frame_map:
            return None, None
        if self.point_image_policy is not None:
            return self.point_raw_locators(i)[0]
        path, frame_in_file = self._frame_map[i]
        return path, frame_in_file

    def frame_for(self, index: int) -> ScanFrame:
        """Attach the raw-image source pointer (from the in-memory ``_frame_map``)
        so a stitch/RSM built from this source can persist resolvable
        contributing-frame records (the raw popup).  A metadata-only SPEC source
        (no ``image_dir``) has no ``_frame_map`` → ``source_path`` stays ``None``
        (correctly — there is no raw image to point at)."""
        i = int(index)
        path, frame_in_file = self.raw_locator_for(i)
        return ScanFrame(
            index=i,
            metadata=dict(self.metadata_for(i)),
            source_path=path,
            source_frame_index=frame_in_file,
            loader=lambda fr: self.load_frame(fr.index),
            source_identity=self.name,
            point_statistics_loader=(self.load_point_statistics if self.point_image_policy is not None else None),
        )


__all__ = ["SpecSource", "find_spec_image_files", "spec_image_stem"]
