"""Ordered provisional Stitch science using the finite native integrations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import math
from pathlib import Path
from threading import Event

import numpy as np

from xrd_tools.analysis import stitch_operation as native
from xrd_tools.analysis.live_preview import (
    MAX_PREVIEW_POINT_BYTES, PreviewGroup, PreviewProgress, PreviewReplay, PreviewSnapshot,
    largest_preview_point_planes, preview_compute_slot,
)
from xrd_tools.analysis.live_preview_grid import enclosing_preview_bounds
from xrd_tools.analysis.stitch_native import (
    calibration_detector, geometry_facts, native_detector_image, native_energy_selection,
    native_mask, revalidate_mask,
)
from xrd_tools.analysis.stitch_native_values import capture_native_stitch_plan, resolve_stitch_gi
from xrd_tools.analysis.xu_stitch_calibration import revalidate_xu_stitch_calibration
from xrd_tools.core.energy import xray_energy_eV
from xrd_tools.core.geometry.xu_runtime import XuRuntimeUnsupported
from xrd_tools.core.invalid import resolve_detector_saturation
from xrd_tools.core.scan import ScanFrame
from xrd_tools.integrate.frame_screen import FrameScreen, FrameScreenPolicy
from xrd_tools.integrate.stitch_stream import StitchPreviewStream
from xrd_tools.integrate.xu_stitch import XuStitchCancelled, XuStitchScienceRefused, run_xu_hist_stitch_1d
from xrd_tools.sources.live_preview import (
    PreviewPointPending, preview_prefixes_hold, read_preview_point,
)
from xrd_tools.sources.memory import MemoryFrameSource
from xrd_tools.sources.point_images import combine_point_statistics

_SLICE_POINTS = 8
_MAX_PREVIEW_PROFILE_BINS = 2048
_MAX_PREVIEW_RADIAL_BINS = 512
_MAX_PREVIEW_AZIMUTHAL_BINS = 180


@dataclass(slots=True)
class _StitchReplay:
    stream: StitchPreviewStream
    screen: FrameScreen
    target: tuple
    cursor: int
    receipt: PreviewReplay


def _verify_preview_assets(plan):
    if type(plan) is native.StitchOperationPlan:
        native._revalidate_geometry(plan.geometry)
    else:
        revalidate_xu_stitch_calibration(plan.calibration)
    revalidate_mask(plan._mask_receipt)
    for receipt in (plan._dark_receipt, plan._flat_receipt):
        if receipt is not None:
            native_detector_image(plan, receipt)


class _StitchPreviewKernel:
    """Bind the finite form's science and project verified point data into it."""

    def __init__(self, form):
        form = replace(form, npt_1d=min(form.npt_1d, _MAX_PREVIEW_PROFILE_BINS),
                       npt_rad_2d=min(form.npt_rad_2d, _MAX_PREVIEW_RADIAL_BINS),
                       npt_azim_2d=min(form.npt_azim_2d, _MAX_PREVIEW_AZIMUTHAL_BINS))
        self.form = form
        self.base, self.motors, self.energy_name = capture_native_stitch_plan(
            form, Path(form.project_root).resolve(strict=True), form.gi)
        self.plan = None

    def context(self, group):
        members = group.members
        gi, _unrecorded = resolve_stitch_gi(self.form.gi, (
            (member.path, member.key, {*member.columns, *(name for name, _ in member.motors)})
            for member in members if member.points))
        # The native group binds each member's observed energy extrema in
        # member order, with its own scale conversion and common-energy rule.
        energies = {id(member): [] for member in members}
        for member, point in group.points:
            values = [value for name, value in point.columns if name == "energy"]
            if len(values) > 1:
                raise ValueError("Energy must be unique in each selected member")
            if values and math.isfinite(values[0]) and values[0] > 0:
                energies[id(member)].append(xray_energy_eV(values[0]))
        terms = [value for values in energies.values() if values for value in (min(values), max(values))]
        selection = native_energy_selection(self.base, terms)
        energy = selection["source_energy_eV"] if selection["source"] == "metadata" else None
        xu = type(self.base) is native.XuStitchOperationPlan
        if xu and math.isclose(selection["wavelength_m"] * 1e10, self.base.effective_geometry.runtime_xdart_wavelength_A,
                               rel_tol=1e-14):
            energy = None
        old = self.plan
        if old is not None and (old.gi, old.metadata_energy_eV) == (gi, energy):
            return False
        self.plan = replace(self.base, gi=gi, metadata_energy_eV=energy)
        self.row_motor = native._screen_row_motor(self.plan)
        self.detector = calibration_detector(self.plan)
        _wavelength, self.shape, self.orientation = geometry_facts(self.plan)
        expected = self.form.detector_shape
        if self.orientation.rotation in {90, 270}:
            expected = expected[::-1]
        if tuple(expected) != self.shape:
            raise ValueError("The preview detector shape differs from the calibrated detector")
        if xu:
            self.mask = native_mask(self.plan)
            self.dark = native_detector_image(self.plan, self.plan._dark_receipt)
            self.flat = native_detector_image(self.plan, self.plan._flat_receipt)
        else:
            receipt = self.plan.geometry
            diff, _orientation = native._runtime_geometry(receipt, receipt.content, metadata_energy_eV=energy)
            self.runtime_plan = native._runtime_plan(self.plan, diff)
        return old is not None

    def verify_assets(self):
        _verify_preview_assets(self.base)

    def asset_witness(self):
        return partial(_verify_preview_assets, self.base)

    def _metadata(self, point):
        row = dict(point.metadata)
        if point.policy is not None:
            row["point_images"] = point.policy.to_dict()
        return row

    def scout(self, entries, cancel_token):
        stream = StitchPreviewStream()
        source = MemoryFrameSource([ScanFrame(i, metadata=self._metadata(point))
                                    for i, (_member, point) in enumerate(entries)])
        self.compute(source, stream, None, None, cancel_token)
        return stream

    def prepare_frame(self, point, planes, recorded, label):
        if any(plane.shape != self.form.detector_shape for plane in planes):
            raise ValueError("The preview detector point differs from the configured frame shape")
        if len(set(recorded)) > 1:
            raise ValueError("Detector constituents record different saturation cutoffs")
        dtypes = {plane.dtype for plane in planes}
        ceiling = resolve_detector_saturation(recorded=recorded[0], detector=self.detector,
                                              dtype=dtypes.pop() if len(dtypes) == 1 else None)
        saturation = None if ceiling is None or ceiling.source == "dtype" else {label: ceiling.value}
        oriented = tuple(self.orientation.apply(plane) for plane in planes)
        frame = ScanFrame(label, image=oriented[0], metadata=self._metadata(point))
        if point.policy is not None:
            policy = point.policy

            def statistics(_index=None, conditioning=None, saturation=None, *, poisson=False):
                return combine_point_statistics(oriented.__getitem__, range(len(oriented)), policy,
                                               conditioning, saturation, poisson=poisson)

            frame.point_statistics_loader = statistics
        return frame, saturation

    def compute(self, source, stream, screen, saturation, cancel_token):
        plan = self.plan
        if type(plan) is native.StitchOperationPlan:
            runtime = self.runtime_plan if saturation is None else replace(self.runtime_plan, saturation=saturation)
            native.run_stitch(runtime, source, frame_indices=source.frame_indices, frame_screen=screen,
                              collect_frame_records=False, cancel_token=cancel_token, preview=stream)
            return
        run_xu_hist_stitch_1d(
            plan.calibration, source, frame_indices=source.frame_indices, frame_screen=screen,
            monitor_key=None if plan.monitor_selector is None else plan.monitor_selector.name,
            **native._foil_backend_kwargs(plan), npt=plan.npt_1d, max_frame_bytes=plan.max_frame_bytes,
            unit=plan.unit, radial_range=plan.radial_range, conditioning=plan.conditioning,
            saturation=saturation, mask=self.mask, polarization_factor=plan.polarization_factor,
            dark=self.dark, flat=self.flat, flat_min_fraction=plan.flat_min_fraction, gi=plan.gi,
            **({} if plan.mode == "1d" else dict(
                map_grid=native._xu_map_grid(plan), map_unit=plan.map_unit,
                map_radial_range=plan.map_radial_range)),
            azimuth_range_1d=plan.azimuth_range_1d, allow_extrapolation=plan.allow_extrapolation,
            metadata_energy_eV=plan.metadata_energy_eV, cancel_token=cancel_token, preview=stream,
        )

    def screen_row(self, member, point):
        value = round(float(point.metadata.get(self.row_motor, 0.0)), 6) if self.row_motor else 0.0
        return (member.path, member.key), value

    @property
    def grid_bytes(self):
        plan = self.plan
        profile = plan.npt_1d * (1 + (0 if plan.gi is None else len(plan.gi.cuts)))
        pixels = 0 if plan.mode == "1d" else plan.npt_rad_2d * plan.npt_azim_2d * (1 + (plan.gi is not None))
        # Both backends retain at most six float64 channels per native part.
        return (profile + pixels) * 6 * 8

    @property
    def asset_bytes(self):
        if type(self.plan) is native.StitchOperationPlan:
            images = self.runtime_plan.mask, self.runtime_plan.dark, self.runtime_plan.flat
        else:
            images = self.mask, self.dark, self.flat
        return sum(array.nbytes for array in images if array is not None)


class StitchPreviewWork:
    """Own one growing verified prefix, its staged replay and native screen."""

    def __init__(self, form, *, image_axis="q_chi"):
        self.form = form
        self.image_axis = image_axis
        self.progress = PreviewProgress()
        self.group = self.kernel = self.stream = self.screen = self.replay = self.snapshot = None
        self.points = ()
        self.saturations = {}
        self.decoded = None

    def _update(self, **changes):
        self.progress = replace(self.progress, **changes)

    def staging_bytes(self, group):
        form = self.form
        pixels = math.prod(form.detector_shape)
        planes = min(pixels * 8 * largest_preview_point_planes(group), form.max_frame_bytes,
                     MAX_PREVIEW_POINT_BYTES)
        profile = min(form.npt_1d, _MAX_PREVIEW_PROFILE_BINS) * (1 + (0 if form.gi is None else len(form.gi.cuts)))
        image = min(form.npt_rad_2d, _MAX_PREVIEW_RADIAL_BINS) * min(form.npt_azim_2d, _MAX_PREVIEW_AZIMUTHAL_BINS)
        if form.mode == "2d":
            profile += image * (1 + (form.gi is not None))
        # Both native grids and their held frame parts can coexist during replay.
        window = FrameScreenPolicy(exclude=form.frame_screen).window
        return planes + pixels * 128 + profile * 6 * 8 * (4 + 2 * window)

    def _clear_grid(self):
        self.stream = self.screen = self.replay = self.snapshot = None
        self.points = ()
        self.saturations.clear()
        self.decoded = None
        self._update(included=0, held=0, excluded=0, waiting=False, end_of_stream=False,
                     replay_target=None, pending_watermark=None, resident_bytes=0, admitted_bytes=0)

    def invalidate(self, group):
        self.group = group
        self.kernel = None
        self._clear_grid()
        self._update(invalidated=self.progress.invalidated + 1)

    def retire_staging(self):
        if self.kernel is not None:
            self.kernel = None
            self._clear_grid()
            self._update(resource_restarts=self.progress.resource_restarts + 1)

    def verify_assets(self):
        if self.kernel is not None:
            self.kernel.verify_assets()

    def _feed(self, member, point, stream, screen, position, cancelled, cancel_token, yield_priority, *, replay):
        if self.decoded is None or self.decoded[0] != point.identity:
            count = 1 if point.policy is None else len(point.policy.raw_indices(0))
            plane_bytes = math.prod(self.form.detector_shape) * 8
            if plane_bytes * count > min(MAX_PREVIEW_POINT_BYTES, self.form.max_frame_bytes):
                raise ValueError("The grouped detector point exceeds the preview memory bound")
            planes, recorded = read_preview_point(point, cancelled=cancelled, include_saturation=True)
            self.decoded = point.identity, planes, recorded
            self._update(decodes=self.progress.decodes + len(planes),
                         forward_decodes=self.progress.forward_decodes + (0 if replay else len(planes)))
        if cancelled():
            return False
        with preview_compute_slot(yield_priority, uses_xu=self._uses_xu) as admitted:
            if not admitted:
                return False
            self._integrate_point(member, point, stream, screen, position, cancel_token)
        return True

    @property
    def _uses_xu(self):
        return type(self.kernel.plan) is not native.StitchOperationPlan

    def _integrate_point(self, member, point, stream, screen, position, cancel_token):
        _, planes, recorded = self.decoded
        self.decoded = None
        member_key = member.path, member.key
        if member_key in self.saturations and self.saturations[member_key] != recorded[0]:
            raise ValueError(f"The detector saturation changed in scan {member.key}")
        self.saturations[member_key] = recorded[0]
        frame, saturation = self.kernel.prepare_frame(point, planes, recorded, position)
        screen.extend((position,), (self.kernel.screen_row(member, point),))
        self.kernel.compute(MemoryFrameSource([frame]), stream, screen, saturation, cancel_token)

    def _grow(self, group, offset, end, cancel_token, progress_callback, yield_priority):
        with preview_compute_slot(yield_priority, uses_xu=self._uses_xu) as admitted:
            if not admitted:
                return False
            scout = self.kernel.scout(group.points[offset:end], cancel_token)
        prior = None if self.stream is None else self.stream.bounds
        bounds = enclosing_preview_bounds(scout.footprint, scout.automatic, prior)
        if self.stream is None:
            self.stream = StitchPreviewStream(bounds)
            self.screen = FrameScreen(self.kernel.plan.frame_screen, incremental=True)
        elif bounds != prior:
            def spans(grid):
                return tuple(high - low for low, high in grid.values())

            self.replay = _StitchReplay(
                StitchPreviewStream(bounds), FrameScreen(self.kernel.plan.frame_screen, incremental=True),
                group.points[:end], 0, PreviewReplay(True, False, end, spans(prior), spans(bounds)),
            )
            self._update(replay_target=end)
            progress_callback("Rebuilding preview", 0, 1)
        return True

    def _step(self, group, cancelled, yield_priority, progress_callback, cancel_token):
        remaining = _SLICE_POINTS
        while remaining and not cancelled() and not yield_priority():
            stage = self.replay
            if stage is not None:
                position = stage.cursor
                member, point = stage.target[position]
                if not self._feed(member, point, stage.stream, stage.screen, position,
                                  cancelled, cancel_token, yield_priority, replay=True):
                    return
                stage.cursor += 1
                remaining -= 1
                if stage.cursor == len(stage.target):
                    self.stream, self.screen, self.points = stage.stream, stage.screen, stage.target
                    self.replay = None
                    self._update(replay_target=None, replays=(*self.progress.replays, stage.receipt),
                                 replayed_points=self.progress.replayed_points + stage.cursor)
                continue
            offset = len(self.points)
            if offset == len(group.points):
                break
            end = min(len(group.points), offset + remaining)
            if not self._grow(group, offset, end, cancel_token, progress_callback, yield_priority):
                return
            if self.replay is not None:
                continue
            for position in range(offset, end):
                if cancelled() or yield_priority():
                    return
                member, point = group.points[position]
                if not self._feed(member, point, self.stream, self.screen, position,
                                  cancelled, cancel_token, yield_priority, replay=False):
                    return
                self.points = (*self.points, (member, point))
                remaining -= 1

    def _set_snapshot(self, group):
        complete = group.ended and self.replay is None and len(self.points) == len(group.points)
        if complete and not self.progress.end_of_stream:
            self.stream.finish(self.screen)
        keys = tuple((self.points[item.label][0].key, self.points[item.label][1].label)
                     for item in self.screen.verdicts if not item.excluded)
        self._update(included=len(keys), held=self.screen.held,
                     excluded=len(self.screen.verdicts) - len(keys), end_of_stream=complete,
                     waiting=False, pending_watermark=len(group.points) if len(self.points) < len(group.points) else None)
        if self.replay is not None or not keys:
            return
        bounds = tuple(self.stream.bounds.values())
        if self.snapshot is not None and (self.snapshot.points, self.snapshot.bounds) == (keys, bounds):
            return
        product = self.stream.snapshot()
        gi = self.image_axis == "qip_qoop" and product.gi_map is not None
        if gi:
            payload, diagnostics = product.gi_map
            radial, second = "q_ip", "q_oop"
        else:
            payload, diagnostics = product.payload, product.diagnostics
            radial, second = "q", "chi"
        axes = ((radial, payload.radial),)
        intensity, sigma = payload.intensity, payload.sigma
        coverage, normalization = diagnostics.coverage, diagnostics.normalization
        if self.form.mode == "2d":
            axes += ((second, payload.azimuthal),)
            intensity, coverage, normalization = intensity.T, coverage.T, normalization.T
            sigma = None if sigma is None else sigma.T
        ranked = getattr(group, "member_ranks", {})
        represented = {(self.points[item.label][0].path, self.points[item.label][0].key)
                       for item in self.screen.verdicts if not item.excluded}
        rank = max((ranked.get(key, group.rank) for key in represented), default=group.rank)
        mask = self.kernel.base._mask_receipt
        self.snapshot = PreviewSnapshot(group.identity, rank, keys, bounds, axes, intensity,
                                        group.covered(len(self.points)), normalization, coverage, sigma, product=product,
                                        display_bytes=3 * self.kernel.grid_bytes
                                        + (0 if mask is None else len(mask.pixels)), metadata_prefix=group)

    def _memory(self):
        grid = self.kernel.grid_bytes
        held = (0 if self.screen is None else self.screen.held)
        stages = 1 if self.replay is None else 2
        held += 0 if self.replay is None else self.replay.screen.held
        visible = 0 if self.snapshot is None else grid
        self._update(resident_bytes=self.kernel.asset_bytes + grid * (stages + held) + visible,
                     admitted_bytes=self.kernel.asset_bytes + grid * (4 + 2 * self.kernel.plan.frame_screen.window)
                     + min(MAX_PREVIEW_POINT_BYTES, self.form.max_frame_bytes))

    def run(self, group, cancelled, yield_priority, progress_callback):
        if self.group != group.identity:
            self.group, self.kernel = group.identity, None
            self._clear_grid()
        if not group.points:
            self._update(waiting=True)
            return None
        targets = self.points if self.replay is None else self.replay.target
        if tuple(point.identity for _, point in targets) != tuple(point.identity for _, point in group.points[:len(targets)]):
            self.invalidate(group.identity)
        cancel_token = getattr(cancelled, "__self__", None)
        if type(cancel_token) is not Event:
            cancel_token = None
        try:
            if cancelled() or yield_priority():
                return self.snapshot
            if not group.verifies(cancelled=cancelled):
                old = self.snapshot
                if old is not None and old.input_prefix.verifies(metadata=old.metadata_prefix, cancelled=cancelled):
                    self._update(waiting=True)
                    return self.snapshot
                self.invalidate(group.identity)
                return None
            if self.kernel is None:
                with preview_compute_slot(yield_priority, uses_xu=self.form.backend != "multigeometry") as admitted:
                    if not admitted:
                        return self.snapshot
                    self.kernel = _StitchPreviewKernel(self.form)
            else:
                self.verify_assets()
            if self.kernel.context(group):
                self._clear_grid()
                self._update(invalidated=self.progress.invalidated + 1)
            self._step(group, cancelled, yield_priority, progress_callback, cancel_token)
            if cancelled() or yield_priority():
                return self.snapshot
            self.verify_assets()
            if not group.covered(len(self.points)).verifies(metadata=group, cancelled=cancelled):
                self.invalidate(group.identity)
                return None
            if self.stream is not None:
                self._set_snapshot(group)
                self._memory()
            return self.snapshot
        except (InterruptedError, XuStitchCancelled):
            return None
        except PreviewPointPending:
            # A temporarily unreadable newer detector slot does not revoke
            # settled input. The ordered stream cannot advance past this slot.
            if not preview_prefixes_hold(PreviewGroup.requirements(self.points), cancelled=cancelled):
                self.invalidate(group.identity)
            self._update(waiting=True)
            return self.snapshot
        except (OSError, ValueError, XuStitchScienceRefused, XuRuntimeUnsupported, MemoryError):
            self.invalidate(group.identity)
            self._update(waiting=True)
            return None
