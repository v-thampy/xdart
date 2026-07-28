# -*- coding: utf-8 -*-
"""
nexusThread — worker thread for NeXus/Tiled wrangler.

Reads frames from a NeXus HDF5 file (Bluesky suitcase-nexus format)
and integrates each one using the same LiveFrame pipeline as imageThread.

Performance shape (post-P3A refactor 2026-05-13):

* **Bulk HDF5 reads** — frames are read in ``_READ_CHUNK``-sized
  slices (``ds[a:b]``), so HDF5 chunk decompression happens once per
  N frames rather than N times for the same chunk.  Reads go through
  :class:`xrd_tools.io.nexus.NexusImageStack`, which exposes a
  single (N, H, W) logical view across either a single 3D dataset or
  an Eiger master's sibling ``data_NNNNNN`` external links — chunked
  reads cross file boundaries seamlessly.
* **Parallel integration** — within each chunk, xdart builds frame shells and
  delegates the worker pool to ``xrd_tools.reduction.run_reduction``.
* **Periodic saves** — disk writes are batched every
  ``_LIVE_SAVE_INTERVAL`` frames so the v2 NeXus writer's per-flush
  cost amortises across the scan.  Skipped entirely under
  ``xye_only`` mode (Int 1D (XYE)).
* **Per-chunk XYE flush** — XYE files are buffered inside the worker
  and flushed once per chunk by ``_flush_xye_buffer`` (inherited from
  wranglerThread).  Buffering keeps the worker thread cheap and
  groups disk traffic so it doesn't interleave with the next chunk's
  integration.  Per-frame XYE export happens in **every** mode
  (Int 1D + 2D, Int 1D, Int 1D (XYE)).
* **GI mode safe** — incident-angle resolution and fiber-integrator ownership
  live inside the headless reduction spine.
* **1D-only mode** — set ``scan.skip_2d = True`` to bypass 2D
  integration entirely (faster on large detectors).  Set
  ``self.xye_only = True`` (in addition) to also bypass the .nxs
  writer and produce XYE files only.

@author: thampy
"""

# Standard library imports
import logging
import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Qt imports
from pyqtgraph import Qt

# Project imports
from xdart.modules.live import LiveFrame
from xrd_tools.core.containers import PONI
from xrd_tools.io.output_safety import (
    OutputCollisionError,
    check_output_not_source,
)
from xrd_tools.integrate.calibration import poni_to_integrator, get_detector
from xrd_tools.reduction import GIFreezeError
from xrd_tools.io.nexus import open_nexus_image_stack, read_nexus
from xrd_tools.io.image import read_image
from xrd_tools.io.processed_scan_id import ProcessedXdartInputError
from xrd_tools.io.export import write_xye
from xdart.utils.h5pool import get_pool as _get_h5pool
from xrd_tools.session.run_configuration import (
    RunConfigurationRefused,
    require_run_configuration,
)
from xdart.modules.reduction import (
    apply_frozen_run_configuration as _apply_frozen_run_configuration,
    open_live_reduction_session,
    StandardPlanCache,
    reduce_live_frames,
    sync_live_scan_gi_settings,
)
from .wrangler_widget import (
    wranglerThread,
)

logger = logging.getLogger(__name__)


class FrozenSourceTarget(NamedTuple):
    """The whole NeXus execution target, from the accepted object.

    O-3N (§14.2 items 4-5) established the first four values.  O-3N.R (§15.4)
    completes it: the newly reachable NeXus worker was still taking its
    calibration, portable source base and output policy from post-admission
    mutable state, so one run could execute with panel science while its
    provenance claimed the accepted identity.  Everything an executing NeXus run
    decides is derived HERE, once, from one ``FrozenRunConfiguration``.
    """

    uri: str
    entry: str
    scan_name: str
    output_path: str
    output_dir: str = ""
    source_base: str = ""
    output_mode: str = "Append"
    poni_values: dict | None = None
    generation: int = 0
    fingerprint: str = ""


# How many frames to bulk-read from the source HDF5 per iteration.
# HDF5 chunks for typical detectors hold a handful of frames each;
# reading in 16-frame slices avoids paying per-chunk decompression
# multiple times for the same chunk.
_READ_CHUNK = 16

# Save cadence inherited from wranglerThread.LIVE_SAVE_INTERVAL.
# Subclass-level override would go here as ``LIVE_SAVE_INTERVAL = N``.

# Default number of parallel integration workers when the GUI
# doesn't expose a Cores spinbox to the NeXus wrangler (it currently
# doesn't — wiring it up is a small UI follow-up).  Caller can set
# ``self.max_cores`` before starting the thread to override.
_DEFAULT_MAX_CORES = 4


class nexusThread(wranglerThread):
    """Thread for processing NeXus/HDF5 image stacks.

    Reads an image dataset from a NeXus file, integrates each frame
    through the headless reduction spine, and emits ``sigUpdate`` after
    each one.

    signals:
        showLabel: str, status text for the UI label
    """
    showLabel = Qt.QtCore.Signal(str)

    def __init__(
            self,
            command_queue,
            scan_args,
            file_lock,
            fname,
            nexus_file,
            poni,
            mask_file,
            gi,
            th_mtr,
            sample_orientation,
            tilt_angle,
            gi_mode_1d,
            gi_mode_2d,
            command,
            scan,
            entry='entry',
            parent=None):

        super().__init__(command_queue, scan_args, fname, file_lock, parent)

        self.nexus_file = nexus_file
        self.poni = poni
        self.gi_mode_1d = gi_mode_1d
        self.gi_mode_2d = gi_mode_2d
        self.command = command
        self.scan = scan
        self.entry = entry
        # N1: project root for portable @source_base; set from the wrangler in
        # setup() (None -> absolute raw paths, back-compat).
        self.source_base = None

        self.detector = None
        self.mask = None
        # O-3N.R: the immutable execution target for the run in progress, and
        # the one-shot latch that makes Overwrite replace exactly once.
        self._execution_target = None
        self._run_output_prepared = None

        # NeXus processing is always batch-mode equivalent — surface the
        # same flags the GUI's wrangler_finished handler checks so it
        # auto-reloads the generated file and selects the last frame
        # once processing is done. Without these, the display is left
        # showing stale state from before the run.
        # Settable from outside (e.g. before .start()) when the GUI
        # eventually exposes a Cores spinbox for the NeXus wrangler.
        # C1: cached standard ReductionPlan, rebuilt only when scan
        # settings change.  Lives on the thread so it survives across
        # chunks within a single run.
        self._plan_cache = StandardPlanCache()

    def _final_save_to_nexus(self, frozen, scan, files_processed):
        """Persist the final NeXus tail with writer-lock-before-pool-pause."""
        if files_processed <= 0 or frozen.run_options.get("xye_only", False):
            return
        is_finalize = (self.command != 'stop')
        # O-3N.R §15.4 item 4: Overwrite already replaced the target once, at
        # the FIRST writer action of this run (``_prepare_output_for_run``), so
        # every save inside the run appends into what this run created and the
        # final save can never mix a previous identity's rows under the new
        # provenance.  ``replace`` stays False here for exactly that reason.
        with self.file_lock:
            pool = _get_h5pool()
            pool.pause(scan.data_file)
            try:
                scan.default_geometry()
                scan.save_to_nexus(
                    replace=False, finalize=is_finalize,
                )
            finally:
                pool.resume(scan.data_file)
        if not is_finalize:
            logger.info(
                '[NEXUS] Stop tail-flushed %d frames; .nxs is '
                'a partial result (no finalize stamp).',
                files_processed,
            )

    # ── Main entry point ─────────────────────────────────────────────────

    def _require_run_configuration(self, stage):
        """Return the EXACT admitted frozen configuration, or refuse (typed).

        Parity with ``imageThread``: the WORKER-ENTRY identity gate.  O-1a-W1R
        (review §39.2 W1R-P1-1) makes the comparison exact object identity
        against the admission ledger, so a genuine but different
        ``FrozenRunConfiguration`` -- same generation, future generation, or an
        equal-valued reconstruction -- refuses here rather than executing the
        substitution.
        """

        frozen = require_run_configuration(
            getattr(self, "run_configuration", None),
            stage=stage,
            floor=int(getattr(self, "run_configuration_floor", 0) or 0),
            expected=getattr(self, "_admitted_run_configuration", None),
        )
        return frozen

    @staticmethod
    def _frozen_source_target(frozen):
        """The accepted source URI, entry, scan name and output path.

        The ONE place a NeXus run answers "what do I open?" and "where does the
        output go?".  Both come from the accepted ``FrozenRunConfiguration``, so
        a poisoned Qt field, wrapper mirror or worker cursor cannot move either
        (§14.2 items 4-5).  Admission already refuses a configuration missing
        ``source`` or ``save_path``; this raises the same typed refusal rather
        than inventing a fallback, because there is no safe value to guess.
        """
        source = getattr(frozen, "source", None)
        uri = str(getattr(source, "uri", "") or "").strip()
        save_path = str(getattr(frozen, "save_path", "") or "").strip()
        for name, value in (("source", uri), ("save_path", save_path)):
            if not value:
                raise RunConfigurationRefused(
                    "absent",
                    stage="nexus-source-target",
                    detail=(
                        f"the accepted run configuration carries no {name}; "
                        "the NeXus run would have to infer it"),
                    generation=int(getattr(frozen, "generation", 0) or 0),
                )
        # O-3N.R (§15.2): NO worker-only fallback.  A cleared Entry editor is
        # refused at admission; inventing "entry" here is exactly what made one
        # accepted identity mean two things -- provenance claiming nothing while
        # execution opened a group.
        entry = str(getattr(source, "entry", "") or "").strip()
        scan_name = Path(uri).stem or "nexus_scan"
        return FrozenSourceTarget(
            uri, entry, scan_name,
            os.path.join(save_path, f"{scan_name}.nxs"),
            output_dir=save_path,
            source_base=str(getattr(frozen, "project_root", "") or ""),
            output_mode=str(getattr(frozen, "output_mode", "") or "Append"),
            poni_values=(dict(frozen.poni_values)
                         if getattr(frozen, "poni_values", None) else None),
            generation=int(getattr(frozen, "generation", 0) or 0),
            fingerprint=str(getattr(frozen, "fingerprint", "") or ""))

    def _preflight_execution_target(self, target):
        """Validate the filesystem and HDF5 facts, in the WORKER, before I/O.

        O-3N.R §15.2/§15.3.  Runs after exact frozen admission and before any
        source-content read or writer open, off the GUI thread:

        * the source is an existing file;
        * the output directory exists (or can be created) and is a directory;
        * the SELECTED entry exists and is the group that will be opened -- the
          shared reader silently falls back to the first ``NXentry``, which let
          a run reduce one group while its provenance named another;
        * the output does not collide with the source, through the existing
          headless :func:`check_output_not_source` owner, so symlink/hard-link
          identity is covered by the same policy the image path uses.
        """
        def refuse(detail):
            raise RunConfigurationRefused(
                "absent", stage="nexus-preflight", detail=detail,
                generation=int(target.generation))

        if not target.entry:
            refuse("the accepted NeXus configuration names no entry")
        source = Path(target.uri)
        if not source.is_file():
            refuse(f"the accepted NeXus source is not a file: {target.uri}")
        output_dir = Path(target.output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            refuse(f"the accepted NeXus output is not a directory: "
                   f"{target.output_dir}")
        try:
            check_output_not_source(target.output_path,
                                    input_files=[target.uri])
        except OutputCollisionError as exc:
            refuse(str(exc))
        # Strict entry: open ONLY the group index, never frame content.
        try:
            import h5py

            with h5py.File(target.uri, "r") as handle:
                present = target.entry in handle
        except OSError as exc:
            refuse(f"the accepted NeXus source could not be opened: {exc}")
        else:
            if not present:
                refuse(f"the selected entry {target.entry!r} does not exist in "
                       f"{target.uri}; refusing rather than reducing another "
                       "NXentry under its provenance")
        return target

    def _execution_poni(self, frozen):
        """The calibration this run integrates with, from the accepted values.

        O-3N.R §15.4 item 1: ``setup()`` rebuilt ``self.poni`` from the PONI
        editor after admission, so a post-admission calibration edit changed the
        science while the frozen provenance claimed the accepted one.
        """
        values = getattr(frozen, "poni_values", None)
        if not values:
            return None
        try:
            return PONI.from_dict(dict(values))
        except Exception:
            logger.debug("accepted PONI values are not constructible",
                         exc_info=True)
            return None

    def _prepare_output_for_run(self, frozen, scan):
        """Honour the accepted output mode on the FIRST writer action.

        O-3N.R §15.4 item 4: the periodic/final saves ignored
        ``frozen.output_mode`` and the final save hard-coded ``replace=False``,
        so two same-stem identities could mix rows in one file while provenance
        moved to the newer one.  Overwrite replaces once, at the start of the
        run; later saves in the SAME run append into what this run created.
        """
        Path(os.path.dirname(scan.data_file)).mkdir(parents=True,
                                                    exist_ok=True)
        if self._run_output_prepared is frozen:
            return False
        self._run_output_prepared = frozen
        if str(getattr(frozen, "output_mode", "Append")) != "Overwrite":
            return False
        target = Path(scan.data_file)
        if target.exists():
            with self.file_lock:
                pool = _get_h5pool()
                pool.pause(str(target))
                try:
                    target.unlink()
                finally:
                    pool.resume(str(target))
            logger.info('[NEXUS] Overwrite: replaced %s', target)
        return True

    def _adopt_frozen_source_target(self, frozen):
        """Initialize this worker's runtime cursors FROM the accepted values.

        Cursor/handle state stays mutable — the reader needs it — but it is
        (re)initialized here from the accepted object before anything opens, so
        a mirror poisoned between admission and execution is overwritten rather
        than obeyed (§14.2 item 4).
        """
        target = nexusThread._frozen_source_target(frozen)
        self.nexus_file = target.uri
        self.entry = target.entry
        self.scan_name = target.scan_name
        self.fname = target.output_path
        # O-3N.R §15.4 items 1-2: the calibration and the portable source base
        # are accepted values too, not panel rereads.
        self.source_base = target.source_base or None
        accepted_poni = nexusThread._execution_poni(self, frozen)
        if accepted_poni is not None:
            self.poni = accepted_poni
        self._execution_target = target
        return target

    def _project_gi_modes_onto_display_scan(self, frozen):
        """Backward GI-mode write onto the mutable DISPLAY scan (retained).

        The twin of ``imageThread._project_gi_modes_onto_display_scan``: an
        acquisition/display projection only.  The values come from the accepted
        frozen configuration, so this can never change what the run integrates.
        """

        if not frozen.gi.enabled or self.scan is None:
            return
        self.scan.bai_1d_args['gi_mode_1d'] = frozen.gi.mode_1d
        self.scan.bai_2d_args['gi_mode_2d'] = frozen.gi.mode_2d

    def run(self):
        """QThread entry: run the integration body."""
        # W-1.2 case 5 parity: refuse before any source read, output open or
        # reduction session — a worker without the accepted configuration does
        # nothing at all.
        try:
            frozen = nexusThread._require_run_configuration(
                self, "nexus-worker-run")
            # O-3N (§14.2 item 4): re-derive what this run opens and where it
            # writes from the accepted object BEFORE any source read or output
            # open, so the panel/wrapper/worker mirrors cannot decide either.
            target = nexusThread._adopt_frozen_source_target(self, frozen)
            # O-3N.R (§15.2/§15.3): the filesystem and HDF5 facts are validated
            # HERE -- in the worker, off the GUI thread, before any source
            # content read or writer open.
            nexusThread._preflight_execution_target(self, target)
        except RunConfigurationRefused as exc:
            logger.error("run refused: %s", exc)
            self.command = 'stop'
            try:
                self.showLabel.emit(f"Run refused: {exc}")
            except Exception:
                logger.debug("showLabel emit failed for refusal", exc_info=True)
            return
        self._reset_xye_output_notifications()
        try:
            self._run_impl(frozen)
        finally:
            self._close_reduction_session()

    def _run_impl(self, frozen):
        """Read frames from a NeXus file and integrate them in parallel."""
        # O-3N.R: derive the execution target from the accepted object HERE too.
        # ``run()`` already adopted it (and preflighted the filesystem/HDF5
        # facts), but the reduction body must be self-sufficient: sibling
        # callers drive ``_run_impl`` directly, and a body that depended on an
        # earlier call having set its cursors would break on them.  The
        # derivation is pure and idempotent.
        target = nexusThread._adopt_frozen_source_target(self, frozen)
        xye_only = frozen.run_options.get("xye_only", False)
        t0 = time.time()
        # O-1a-W1R (review §39.5 Phase 2 item 1): capture the accepted frozen
        # reference ONCE at worker entry.  Identity was gated in ``run()``; this
        # is the object every execution decision below consumes.
        if self.poni is None or not self.nexus_file:
            return

        # Setup detector and global mask
        self.detector = (get_detector(self.poni.detector)
                         if self.poni.detector else None)
        det_mask = self.detector.mask if self.detector is not None else None
        if frozen.mask_file and os.path.exists(frozen.mask_file):
            custom_mask = np.asarray(read_image(frozen.mask_file), dtype=bool)
            det_mask = (det_mask | custom_mask if det_mask is not None
                        else custom_mask)
        self.mask = np.flatnonzero(det_mask) if det_mask is not None else None

        nexusThread._project_gi_modes_onto_display_scan(self, frozen)

        # Read scan-level metadata once (counters/angles per-frame
        # arrays).  Per-frame slicing happens later.
        try:
            scan_meta = read_nexus(self.nexus_file, self.entry)
            base_meta = {}
            for k, v in scan_meta.counters.items():
                if len(v) > 0:
                    base_meta[k] = float(v[0])
            for k, v in scan_meta.angles.items():
                if len(v) > 0:
                    base_meta[k] = float(v[0])
        except Exception:
            scan_meta = None
            base_meta = {}

        # O-3N: one derivation — the name comes from the accepted source, never
        # a second stem computed off a mutable mirror.
        scan_name = target.scan_name
        scan = self._initialize_scan(scan_name)
        scan._cached_integrator = poni_to_integrator(self.poni)
        scan._cached_poni = self.poni
        scan._cached_fiber_integrator = None

        # Notify the GUI that a new scan is being processed.
        self.sigUpdateFile.emit(
            scan_name, self.fname,
            frozen.gi.enabled, frozen.gi.scan_incidence_motor,
            False, False,  # single_img=False, series_average=False
        )

        files_processed = 0
        # ``open_nexus_image_stack`` transparently handles two layouts:
        #   • single 3D dataset (e.g. /entry/instrument/detector/data)
        #   • Eiger master with sibling external links
        #     /entry/data/data_NNNNNN → individual _data_*.h5 files.
        # The proxy exposes the full scan as one (N, H, W) slice-able
        # object, so chunked reads can cross file boundaries.
        try:
            ds_cm = open_nexus_image_stack(self.nexus_file, self.entry)
        except ProcessedXdartInputError:
            # F-NXS-2: the selected file is a processed xdart output (integrated
            # results, no raw detector frames).  The shared finder now rejects it
            # before the largest-3D fallback could return an integrated cake as a
            # raw stack; surface a clear message and stop cleanly instead of
            # letting the ValueError escape run()'s try/finally as an uncaught
            # QThread exception (mirrors the image wrangler's processed-skip).
            self.showLabel.emit(
                'Selected file is a processed xdart output, not a raw NeXus '
                'acquisition; open it in the viewer instead of reducing it.')
            return
        except (KeyError, FileNotFoundError) as exc:
            self.showLabel.emit(f'No image dataset found in NeXus file: {exc}')
            return
        with ds_cm as ds:
            nframes = ds.shape[0]
            n_segments = ds.n_segments
            self.showLabel.emit(
                f'Found {nframes} frames in {Path(self.nexus_file).name}'
                + (f' ({n_segments} data files)' if n_segments > 1 else '')
            )

            # F3: prewarm the stable bad-pixel mask cache on the main
            # thread before any worker runs.  Without this, the first
            # N workers all race to compute and write
            # scan._cached_data_mask (same value, but the invariant
            # isn't enforced).  Cheap: one frame read + a flatten.
            if getattr(scan, '_cached_data_mask', None) is None:
                first_frame = np.asarray(ds[0], dtype=np.float32)
                self._prewarm_frame_mask(frozen, scan, first_frame)

            n_workers = min(frozen.max_cores, nframes)
            # C1: cached per-scan plan — rebuilt only when scan
            # integration settings or mask change between chunks.
            sync_live_scan_gi_settings(
                scan,
                incidence_motor=frozen.gi.scan_incidence_motor,
                sample_orientation=frozen.gi.sample_orientation,
                tilt_angle=frozen.gi.tilt_angle,
            )
            standard_plan = self._plan_cache.get(
                # O-1a-W1R (review §39.2 W1R-P1-6): this 2D-integration
                # decision used to read ``skip_2d`` off the locally
                # aliased DISPLAY scan, which the committed AST census
                # could not see.  It now comes from frozen policy.
                # O-3N: the name is ``frozen`` in this scope -- ``_frozen`` was
                # a NameError that no NeXus run could reach while the O-3 source
                # guard refused them all.  Making NeXus runs executable again
                # makes this line reachable, so it is corrected here.
                scan, integrate_2d=not frozen.skip_2d,
            )
            frames_since_save = 0
            for chunk_start in range(0, nframes, _READ_CHUNK):
                if self.command == 'stop':
                    break
                chunk_end = min(chunk_start + _READ_CHUNK, nframes)
                chunk_size = chunk_end - chunk_start

                # Bulk-read the chunk — one HDF5 decompression pass.
                _t_read = time.time()
                block = np.asarray(ds[chunk_start:chunk_end],
                                   dtype=np.float32)
                _t_read = time.time() - _t_read

                # Build per-frame live shells.  The headless reducer owns the
                # worker pool; xdart keeps source provenance and later GUI
                # publication.
                frames = []
                for i, frame_idx in enumerate(range(chunk_start, chunk_end)):
                    frames.append(self._build_frame(
                        frozen, scan,
                        frame_idx,
                        block[i],
                        self._frame_meta(scan_meta, base_meta, frame_idx),
                    ))

                # ── Headless parallel integration ───────────────────
                self.showLabel.emit(
                    f'Integrating frames {chunk_start+1}-{chunk_end}'
                    f'/{nframes} ({n_workers} workers)'
                )
                _t_phase1 = time.time()
                executor = n_workers if n_workers > 1 else None
                try:
                    session = self._get_reduction_session(
                        self._reduction_session_key_for(scan, standard_plan, n_workers),
                        lambda: open_live_reduction_session(
                            frames,
                            standard_plan,
                            scan_name=str(getattr(scan, "name", "scan")),
                            global_mask=self.mask,
                            integrator=scan._cached_integrator,
                            poni=self.poni,
                            executor=executor,
                            cancel_token=self._cancel_token(),
                            chunk_size=len(frames) if frames else 1,
                            gi_freeze_mode="scout_union" if frozen.gi.enabled else None,
                        ),
                    )
                except GIFreezeError as exc:
                    # GI freeze scout (run when the session is built) found a
                    # blank/degenerate grid.  The whole scan shares the GI
                    # geometry, so retrying later chunks won't help -- surface
                    # the fix and stop.
                    self.showLabel.emit(
                        'GI 2D scout frame is blank or the grid is degenerate: '
                        'set Theta Motor to Manual and enter the incident '
                        'angle, or check the mask / threshold.'
                    )
                    logger.warning('GI freeze scout failed: %s', exc)
                    break
                frames = reduce_live_frames(
                    frames,
                    standard_plan,
                    scan_name=str(getattr(scan, "name", "scan")),
                    global_mask=self.mask,
                    integrator=scan._cached_integrator,
                    poni=self.poni,
                    session=session,
                    cancel_token=self._cancel_token(),
                    chunk_size=len(frames) if frames else 1,
                    gi_freeze_mode="scout_union" if frozen.gi.enabled else None,
                )
                _t_phase1 = time.time() - _t_phase1

                for frame in frames:
                    if frame is None:
                        continue
                    with self._xye_lock:
                        self._xye_buffer.append((frame.idx, frame))

                # ── Serial accumulation into the scan ─────────────
                # scan.add_frame and the sigUpdate emit happen serially; scan
                # isn't thread-safe for concurrent writes, and the GUI widgets
                # it feeds aren't either.
                for frame in frames:
                    if frame is None:
                        continue
                    self._publish(frozen, scan, frame)
                    self.sigUpdate.emit(frame.idx)
                    files_processed += 1
                    frames_since_save += 1

                # ── Per-chunk XYE flush ─────────────────────────────
                # Drain the XYE buffer once per chunk — keeps disk I/O
                # batched and prevents the buffer from growing without
                # bound on long scans.  Inherited ``_flush_xye_buffer``
                # is a no-op when the buffer is empty (e.g. on Int 2D
                # mode would be — but we always populate it).
                # P3: pass the set of frame.idx values that survived the
                # headless reduction call so a Stop-aborted batch doesn't
                # leave orphan XYE files for frames that never landed in .nxs.
                _t_xye = time.time()
                published_idxs = {a.idx for a in frames if a is not None}
                self._flush_xye_buffer(scan, published_idxs=published_idxs)
                _t_xye = time.time() - _t_xye

                logger.info(
                    '[NEXUS-BATCH] frames %d-%d  read=%.3fs  '
                    'integrate=%.3fs  xye=%.3fs  total=%.3fs',
                    chunk_start, chunk_end - 1, _t_read, _t_phase1,
                    _t_xye, _t_read + _t_phase1 + _t_xye,
                )

                # ── Periodic .nxs save ──────────────────────────────
                # ``LIVE_SAVE_INTERVAL`` (inherited from
                # wranglerThread) is checked at chunk boundaries —
                # not frame boundaries; the v2 writer's per-flush
                # cost is ~30 ms regardless, so the granularity is
                # close enough.  Skipped entirely in xye_only mode
                # (the inherited ``_save_to_disk`` is also a no-op
                # under xye_only, but we short-circuit here too so
                # the chunk loop reads clean).
                _due = frames_since_save >= self.LIVE_SAVE_INTERVAL
                if not _due and not xye_only and frames_since_save > 0:
                    # Cap-aware bound (mirrors imageThread._save_due): the
                    # 1D interval is 1000, but stash() cannot evict unsaved
                    # frames -- without this, up to 1000 frames each pinning
                    # an ~18 MB raw chunk view accumulated between saves.
                    _cap = getattr(scan.frames, "_in_memory_cap", 64)
                    _counter = getattr(scan.frames,
                                       "unsaved_in_memory_count", None)
                    _unsaved = (_counter() if callable(_counter)
                                else frames_since_save)
                    _due = _unsaved >= max(1, _cap - 8)
                if not xye_only and _due:
                    self._save_to_disk(frozen, scan)
                    frames_since_save = 0

        # Final save: write everything coherent + provenance + finalize.
        #
        # N4 — Stop tail flush.  Pre-N4 this was gated on
        # ``self.command != 'stop'``, which meant that if the user
        # hit Stop after some frames had been processed and
        # published but before the next periodic save kicked in,
        # those tail frames remained in memory only and were lost.
        # Now we always do a non-finalize save on Stop so the
        # processed prefix lands on disk — only ``finalize=True``
        # (provenance + write-once items) is skipped on Stop, since
        # the scan didn't actually complete and the file should be
        # marked as a partial result.
        #
        # H30/RN-2: take the writer file_lock before pausing the pooled reader.
        # ``scan.save_to_nexus`` also takes this same reentrant lock internally;
        # the outer hold is the ordering guard that prevents pause/close from
        # racing a load worker that borrowed a pooled read handle under file_lock.
        self._final_save_to_nexus(frozen, scan, files_processed)

        self.showLabel.emit(f'Done — {files_processed} frames processed')
        logger.info(
            'NeXus total time: %.2fs, %d frames', time.time() - t0,
            files_processed,
        )
        # Pool reclamation is handled by run()'s finally (covers normal AND
        # exception-aborted runs).

    # ── Helpers ─────────────────────────────────────────────────────────

    def _initialize_scan(self, scan_name):
        """Create or reset the LiveScan for this scan.

        RESIDUAL (ledgered): the NeXus worker still executes ON the display scan
        object rather than a per-run scan of its own -- that restructuring is
        outside the W-1 packet.  What W-1B closes is the AUTHORITY: every
        configuration value written here now comes from the accepted frozen
        configuration, and the run identity + detached writer projection are
        attached BEFORE the writer opens, exactly as on the image path.
        """
        frozen = nexusThread._require_run_configuration(
            self, "nexus-initialize-scan")
        target = nexusThread._adopt_frozen_source_target(self, frozen)
        self.scan.name = scan_name
        self.scan.gi = bool(frozen.gi.enabled)
        self.scan.static = True
        # O-3N.R §15.1 — THE writer target.  Both the periodic save and
        # ``_final_save_to_nexus`` write through ``scan.data_file``, and so does
        # every XYE file (``save_1d`` derives its directory from it).  Until
        # this assignment the only repoint was the ASYNCHRONOUS GUI
        # ``sigUpdateFile`` chain, so a previous/default/browsed file could
        # receive the run if the GUI had not processed the signal before the
        # first flush.  The run owns its output before anything can write.
        self.scan.data_file = target.output_path
        _apply_frozen_run_configuration(self.scan, frozen)
        # N1: the project root -> entry/@source_base + relative raw source paths
        # in the writer (portable .nxs).  None -> absolute paths (back-compat).
        # The abspath at frame.source_file stays as-is; the writer relativizes it
        # against source_base at write time.  O-3N.R §15.4 item 2: the value is
        # the ACCEPTED project root, not a post-admission panel reread.
        self.scan.source_base = target.source_base or None
        self._active_scan = self.scan
        nexusThread._prepare_output_for_run(self, frozen, self.scan)
        return self.scan

    def _frame_meta(self, scan_meta, base_meta, frame_idx):
        """Build a per-frame metadata dict from scan-level arrays.

        Falls back to ``base_meta`` (frame 0's slice) when the scan
        arrays are shorter than expected — keeps the call cheap and
        avoids exceptions in the parallel section.
        """
        meta = dict(base_meta)
        if scan_meta is None:
            return meta
        try:
            for k, v in scan_meta.counters.items():
                if frame_idx < len(v):
                    meta[k] = float(v[frame_idx])
            for k, v in scan_meta.angles.items():
                if frame_idx < len(v):
                    meta[k] = float(v[frame_idx])
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            # AttributeError: scan_meta missing counters/angles attr.
            # TypeError/ValueError: counter array contains non-numeric.
            # KeyError: shouldn't happen but be defensive on dict-like.
            # Any of these falls back to base_meta — already populated.
            logger.debug(
                "frame_meta lookup failed for frame %s: %s", frame_idx, e,
            )
        return meta

    def _build_frame(self, frozen, scan, frame_idx, img_data, img_meta):
        """Build a LiveFrame shell for the headless reducer."""
        frame_mask = self._resolve_frame_mask(frozen, scan, img_data)
        frame = LiveFrame(
            frame_idx, img_data, poni=self.poni,
            scan_info=img_meta, static=True, gi=frozen.gi.enabled,
            th_mtr=frozen.gi.scan_incidence_motor,
            sample_orientation=frozen.gi.sample_orientation,
            tilt_angle=frozen.gi.tilt_angle,
            series_average=False,
            integrator=scan._cached_integrator,
            mask=frame_mask,
        )

        # Set source file reference for the v2 NeXus per-frame group.
        # The source_frame_idx is the *global* frame index across all
        # external-link data files (matches NexusImageStack's flattened
        # view), so a lazy raw loader can do
        # ``NexusImageStack(source_file)[source_frame_idx]`` directly
        # without needing to know which data_NNNNNN segment to open.
        frame.source_file = os.path.abspath(str(self.nexus_file))
        frame.source_frame_idx = int(frame_idx)
        # NeXus frames already live in the source — don't double-store
        # them in the output .nxs.
        frame.skip_map_raw = True

        return frame

    def _publish(self, frozen, scan, frame):
        """Push the integrated frame into scan + the publish slot.

        Runs on the main thread after the parallel section so
        ``scan.add_frame`` is serialised — scan isn't thread-safe
        for concurrent writes.

        After D1 (unified handoff): we leave the frame in
        ``self._published_frames[frame.idx]`` and emit ``sigUpdate``;
        ``static_scan_widget.update_data`` publishes it into the shared store.
        """
        # In-memory accumulate only — the chunked flush at the end of
        # the dispatch loop (and the final ``save_to_nexus(finalize=True)``
        # at the bottom of ``run()``) handle persistence.
        # xye_only: SKIP the series stash (mirrors imageThread).  Both saves
        # are gated off in this mode, so mark_persisted never runs and
        # stash() could never evict — every frame's raw (a view pinning the
        # whole bulk read chunk) accumulated for the entire run.
        if not frozen.run_options.get("xye_only", False):
            scan.add_frame(
                frame=frame, calculate=False, update=True,
                get_sd=True, set_mg=False, static=True, gi=frozen.gi.enabled,
                th_mtr=frozen.gi.scan_incidence_motor, series_average=False,
                batch_save=True,
            )
        # Publish for the GUI's update_data slot to consume.  Single
        # write site for the dict round-trip.
        self._published_frames[frame.idx] = frame

    # ``_save_to_disk`` is inherited from wranglerThread.  Called
    # from the chunk loop every LIVE_SAVE_INTERVAL frames so the
    # on-disk file stays close to in-memory state even if the user
    # kills the process mid-scan.


