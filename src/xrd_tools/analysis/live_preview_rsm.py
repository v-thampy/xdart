"""In-memory RSM preview science using the native frame and gridding rules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json

import numpy as np

from xrd_tools.analysis import rsm_operation as native
from xrd_tools.analysis.live_preview_grid import preview_rsm_bins
from xrd_tools.analysis.rsm_geometry_asset import (
    bind_rsm_member_geometry,
    capture_rsm_calibration_geometry,
    lower_rsm_effective_geometry,
    revalidate_rsm_geometry_source,
    rsm_calibration_detector,
    rsm_calibration_detector_mask,
    rsm_effective_pixel_q_map,
)
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate.stitch_conditioning import condition_stitch_frame
from xrd_tools.io.spec import get_energy, get_energy_and_UB, read_spec_preview_prefix
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame, rsm_coordinate_matrix
from xrd_tools.rsm.gridding import RSMGridChunkLease, StreamingGridder
from xrd_tools.sources.point_images import combine_point_images


@dataclass(frozen=True, slots=True)
class RSMPreviewOptions:
    project_root: str
    geometry_asset: object
    coordinate_frame: RSMCoordinateFrame
    conditioning: native.RSMImageConditioning
    normalization: native.RSMNormalizationPolicy
    bins: tuple[int, int, int]
    max_frame_bytes: int
    max_chunk_bytes: int | None
    corrections: native.RSMFrameCorrections
    ub: native.RSMUBOverride | None
    motor_selectors: tuple
    detector_shape: tuple[int, int]
    raw_dtype: str


@dataclass(frozen=True, slots=True)
class _CorrectionInputs:
    corrections: native.RSMBoundCorrections | None
    max_frame_bytes: int


@dataclass(frozen=True, slots=True)
class _DetectorInput:
    detector_shape: tuple[int, int]
    source_options_json: str


@dataclass(frozen=True, slots=True)
class _MemberCoordinates:
    energy: float
    ub: np.ndarray


def _verify_preview_assets(geometry, ub, bound, max_frame_bytes):
    revalidate_rsm_geometry_source(geometry)
    native._revalidate_rsm_ub(ub)
    if bound is None:
        return
    from xrd_tools.analysis.stitch_native import capture_detector_image, revalidate_mask

    revalidate_mask(bound.mask_receipt)
    for role in ("dark", "flat"):
        receipt = getattr(bound, f"{role}_receipt")
        if receipt is None:
            continue
        current, _ = capture_detector_image(
            receipt.lexical_path, role, max_frame_bytes,
            bound.corrections.flat_min_fraction, detector_mask=bound.detector_mask,
        )
        if current != receipt:
            raise ValueError(f"Live preview {role} changed")


class RSMPreviewKernel:
    """Own calibration and correction state independently of a native output."""

    def __init__(self, options: RSMPreviewOptions, *, difference: bool):
        self.options = options
        self.bins = preview_rsm_bins(options.bins)
        calibration = options.geometry_asset
        self.receipt = capture_rsm_calibration_geometry(
            calibration.locator, project_root=options.project_root, refinement=calibration.refinement,
        )
        self.geometry = lower_rsm_effective_geometry(self.receipt)
        self.mapper = rsm_effective_pixel_q_map(self.geometry)
        header = self.geometry.detector_header
        shape = int(header.Nch1), int(header.Nch2)
        if shape != options.detector_shape:
            raise ValueError("Live preview calibration and detector shapes differ")
        self.detector_mask = rsm_calibration_detector_mask(self.receipt)
        self.bound_ub = native._bind_rsm_ub(options.ub, options.coordinate_frame)
        binding = bind_rsm_member_geometry(self.geometry, member_ordinal=0, motor_selectors=options.motor_selectors)
        self.selectors = binding.motor_selectors
        detector = _DetectorInput(shape, json.dumps({"read_image_kwargs": {"raw_dtype": options.raw_dtype}}))
        self.corrections = native._bind_rsm_corrections(
            options.corrections, (detector,), options.max_frame_bytes,
            detector=rsm_calibration_detector(self.receipt), detector_mask=self.detector_mask,
        )
        self.correction_state = native._rsm_correction_state(
            _CorrectionInputs(self.corrections, options.max_frame_bytes), header,
            self.geometry.diffractometer_projection, self.receipt, separate_dark_invalid=difference,
        )
        cropped = header if self.geometry.roi is None else header.with_roi(self.geometry.roi)
        roi_shape = int(cropped.Nch1), int(cropped.Nch2)
        memory = native.rsm_v2_chunk_bytes(
            shape, roi_shape, 1, static_hot=self.hot_threshold is not None,
            corrections=options.corrections, dark=options.corrections.dark_path is not None,
            flat=options.corrections.flat_path is not None,
            detector_mask=self.detector_mask is not None, point_images=difference,
        )
        self.max_chunk_bytes = options.max_chunk_bytes or memory
        if memory > self.max_chunk_bytes or shape[0] * shape[1] * 8 > options.max_frame_bytes:
            raise ValueError("A detector point exceeds the preview science memory limit")

    @property
    def hot_threshold(self):
        return self.options.conditioning.static_hot_threshold

    def verify_assets(self):
        self.asset_witness()()

    def asset_witness(self):
        return partial(_verify_preview_assets, self.receipt, self.bound_ub, self.corrections,
                       self.options.max_frame_bytes)

    @property
    def witness_bytes(self):
        bound = self.corrections
        if bound is None:
            return 0
        mask = 0 if bound.mask_receipt is None else len(bound.mask_receipt.pixels)
        return mask + (0 if bound.detector_mask is None else bound.detector_mask.nbytes)

    def runtime(self):
        return xu_runtime_session(self.geometry.runtime_requirements)

    def coordinates(self, member, point, session):
        # Energy and orientation must be read from a capture that agrees with
        # the detached point even if the original was edited and restored.
        current = read_spec_preview_prefix(member.path, member.key)
        if (current is None or current[0].identity != member.prefix.identity
                or not current[0].extent.extends(member.prefix.extent.through(point.label + 1))):
            raise ValueError(f"SPEC geometry context changed for scan {member.key}")
        frame = self.options.coordinate_frame
        ub = None
        if frame is RSMCoordinateFrame.HKL and self.bound_ub is None:
            energy, g3 = get_energy_and_UB(member.path, member.key)
            ub, _ = native.rsm_spec_orientation_ub(member.path, member.key, g3, self.mapper, session)
        else:
            energy = get_energy(member.path, member.key)
            if self.bound_ub is not None:
                ub = self.bound_ub.array()
        return _MemberCoordinates(energy, rsm_coordinate_matrix(frame, ub))

    @staticmethod
    def _column(point, selector):
        if selector is None:
            return None
        values = tuple(value for name, value in point.columns if name == selector.name)
        if not values and selector.occurrence == 0 and selector.name in point.metadata:
            values = (point.metadata[selector.name],)
        if selector.occurrence >= len(values):
            raise ValueError(f"Preview point lacks SPEC column {selector.name}")
        return np.asarray([values[selector.occurrence]], dtype=float)

    def angles(self, point):
        return tuple(self._column(point, selector) for _, selector in self.selectors)

    def footprint(self, angles, coordinates, session):
        ranges = native._resolve_exact_rsm_q_bounds_active(
            self.mapper, angles, coordinates.energy, coordinates.ub, roi=self.geometry.roi, chunk_size=1,
            max_frame_bytes=self.options.max_frame_bytes, max_chunk_bytes=self.max_chunk_bytes,
            runtime_session=session,
        )
        return dict(zip(self.options.coordinate_frame.axis_names, ranges, strict=True))

    def condition(self, point, planes):
        conditioning = self.options.conditioning
        policy = point.policy
        if policy is None:
            raw = np.array(planes[0][None], dtype=float, order="C")
            signal = native.condition_rsm_images(raw, conditioning)
        else:
            bound = self.corrections
            threshold = None if bound is None else bound.corrections.threshold
            saturation = None if bound is None or bound.saturation is None else bound.saturation[0]

            def transform(plane):
                return native.condition_rsm_images(condition_stitch_frame(plane, threshold, saturation), conditioning)[0]

            combined = combine_point_images(planes.__getitem__, range(len(planes)), policy, native_transform=transform)
            raw = np.array(combined[None], dtype=float, order="C")
            signal = raw.copy()
        return raw, signal

    def new_grid(self, bounds, session):
        grid = StreamingGridder(self.mapper, self.bins, coordinate_frame=self.options.coordinate_frame)
        with grid.runtime_slice(session):
            grid.set_bounds(*(bounds[name] for name in self.options.coordinate_frame.axis_names))
        return grid

    def add(self, grid, point, raw_and_signal, angles, coordinates, mask, session):
        raw, signal = raw_and_signal
        # The gridding lease consumes the only references to its roots. The caller
        # can keep its own input or conditioned hot-pixel evidence separately.
        raw = raw.copy()
        signal = signal.copy()
        if self.detector_mask is not None:
            signal[:, self.detector_mask] = np.nan
        if mask is not None:
            signal[:, mask] = np.nan
        norm = self.options.normalization
        divisors = native.rsm_normalization_divisors(
            norm, 1, foil_status=self._column(point, norm.foil_selector),
            exposure_seconds=self._column(point, norm.exposure_selector),
            monitor_counts=self._column(point, norm.monitor_selector),
        )
        state = self.correction_state
        if state is None:
            signal /= divisors[:, None, None]
        else:
            saturation = None if state.bound.saturation is None else state.bound.saturation[0]
            native._apply_rsm_frame_corrections(raw, signal, state, divisors, saturation, point.policy)
        roi = self.geometry.roi
        weights = native._rsm_chunk_weights(state, self.mapper, session, angles, coordinates.energy, roi)
        lease = RSMGridChunkLease.from_arrays(raw, signal, 1)
        del raw, signal
        with grid.runtime_slice(session):
            grid.add_leased(lease, angles, coordinates.energy, UB=coordinates.ub, roi=roi,
                            weight=1.0 if weights is None else weights)

    def volume(self, grid, session):
        with grid.runtime_slice(session):
            return grid.to_volume()
