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
from xdart.modules.live import LiveFrame, LiveScan
from xrd_tools.core.containers import PONI
from xrd_tools.core.provenance import read_provenance
from xrd_tools.io.output_safety import (
    OutputCollisionError,
    check_output_not_source,
)
from xrd_tools.session.readiness import (
    append_config_mismatch_check,
    processing_config_from_mapping,
)
from xrd_tools.integrate.calibration import poni_to_integrator, get_detector
from xrd_tools.reduction import GIFreezeError
from xrd_tools.io.nexus import open_nexus_image_stack_exact, read_nexus
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

#: Sentinel: "judge the object on my carrier", as distinct from an explicitly
#: supplied candidate that happens to be ``None`` (which must still refuse).
_CARRIER = object()

#: Suffix of the run-owned staging copy an Overwrite replacement renames the
#: prior target to.  It lives beside the target so the rename is atomic, and it
#: is dropped on commit / restored on rollback (§17.5).
_REPLACING_SUFFIX = ".xdart-replacing"


class FrozenSourceTarget(NamedTuple):
    """The whole NeXus execution target, from the accepted object.

    O-3N (§14.2 items 4-5) established the first four values.  O-3N.R (§15.4)
    completes it: the newly reachable NeXus worker was still taking its
    calibration, portable source base and output policy from post-admission
    mutable state, so one run could execute with panel science while its
    provenance claimed the accepted identity.  Everything an executing NeXus run
    decides is derived HERE, once, from one ``FrozenRunConfiguration``.

    O-3N.R.2 §17.7: the target holds calibration VALUES.  It used to embed a
    constructed ``PONI`` -- a ``@dataclass(slots=True)``, so mutable -- while
    calling itself immutable, which meant every consumer shared one object that
    any of them could edit.  :meth:`poni` hands each caller its own.
    """

    uri: str
    entry: str
    scan_name: str
    output_path: str
    output_dir: str = ""
    source_base: str = ""
    output_mode: str = "Append"
    #: The accepted calibration as an immutable ``(key, value)`` tuple.  Empty
    #: means the accepted run genuinely carries no calibration; execution then
    #: refuses rather than reusing whatever object the worker happened to hold.
    poni_values: tuple[tuple[str, object], ...] = ()
    generation: int = 0
    fingerprint: str = ""

    def poni(self):
        """A FRESH :class:`PONI` for this caller, or ``None``.

        Constructibility was already proved in
        :meth:`nexusThread._frozen_source_target`, so this cannot raise for a
        target that exists.
        """
        if not self.poni_values:
            return None
        return PONI.from_dict(dict(self.poni_values))


class PreparedNexusExecution:
    """The ONE prepared execution envelope for a NeXus run (§17.8 item 1).

    O-3N.R.2's design checkpoint.  The parent kept five parallel latches --
    ``_prepared_for``, ``_prepared_stack``, ``_run_output_prepared``,
    ``_run_target_replaced`` and ``_xye_tail_cleared`` -- and each of them
    recorded "prepared / replaced / cleared" *before* the resource or
    transaction it named had reached a successful terminal state.  That is one
    root cause with six symptoms (§17.1-§17.6): a foreign object could be
    prepared, a proved stack could leak, a refused Append could be retried into
    silence, a failed replacement could burn its retry identity, and a swallowed
    delete could strand a stale XYE tail forever.

    So this object owns all of it, for exactly one run:

    * ``frozen`` -- the EXACT admitted ``FrozenRunConfiguration``, held by
      reference.  Every later decision compares against it by ``is``;
    * ``target`` -- the one derived :class:`FrozenSourceTarget`;
    * ``stack`` -- the strict exact-entry raw stack, closed exactly once;
    * ``scan`` -- this run's own ``LiveScan``;
    * the output-transaction state, each flag consumed only AFTER the operation
      it describes has succeeded.
    """

    __slots__ = ("frozen", "target", "stack", "scan", "poni",
                 "output_committed", "target_replaced", "xye_tail_pending",
                 "_closed")

    def __init__(self, frozen, target, stack=None):
        self.frozen = frozen
        self.target = target
        self.stack = stack
        self.scan = None
        self.poni = target.poni() if target is not None else None
        #: Append qualification finished successfully (never set on refusal).
        self.output_committed = False
        #: An Overwrite replacement has been COMMITTED by a successful writer.
        self.target_replaced = False
        #: ``None`` until the stale XYE set is discovered (once, before this run
        #: writes); then the list of artifacts still awaiting deletion.
        self.xye_tail_pending = None
        self._closed = False

    def owns(self, frozen) -> bool:
        """True only for the exact object this envelope was prepared for."""
        return frozen is self.frozen

    def close(self) -> None:
        """Close the owned raw stack exactly once.  Idempotent by contract.

        §17.8 item 3: every success, typed refusal, exception, Stop and Close
        path routes here, so the proved HDF5 handle cannot outlive the run.
        """
        if self._closed:
            return
        self._closed = True
        stack = self.stack
        if stack is None:
            return
        try:
            stack.close()
        except Exception:                              # noqa: BLE001
            logger.debug("prepared stack close failed", exc_info=True)


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
        # O-3N.R.2 §17.8 item 1: the ONE prepared execution envelope for the run
        # in progress -- exact admitted object, derived target, proved raw
        # stack, run scan and output-transaction state.  It replaces the five
        # parallel latches whose early "prepared/replaced/cleared" writes were
        # the §17 root cause; ``None`` means this worker is idle.
        self._execution = None

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
        prepared = nexusThread._require_execution(self, frozen)

        def write():
            # O-3N.R §15.4 item 4: Overwrite replaces the target exactly once,
            # at the first SUCCESSFUL writer action of this run, so every save
            # inside the run appends into what this run created and the final
            # save can never mix a previous identity's rows under the new
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

        # §16.4/§17.5: if no periodic save happened, THIS is the run's first
        # writer action, so the atomic Overwrite replacement belongs here -- the
        # same success-after-operation rule as every other save.
        nexusThread._write_run_result(self, prepared, scan, write)
        if not is_finalize:
            logger.info(
                '[NEXUS] Stop tail-flushed %d frames; .nxs is '
                'a partial result (no finalize stamp).',
                files_processed,
            )

    # ── Main entry point ─────────────────────────────────────────────────

    def _require_run_configuration(self, stage, candidate=_CARRIER):
        """Return the EXACT admitted frozen configuration, or refuse (typed).

        Parity with ``imageThread``: the WORKER-ENTRY identity gate.  O-1a-W1R
        (review §39.2 W1R-P1-1) makes the comparison exact object identity
        against the admission ledger, so a genuine but different
        ``FrozenRunConfiguration`` -- same generation, future generation, or an
        equal-valued reconstruction -- refuses here rather than executing the
        substitution.

        O-3N.R.2 §17.1: ``candidate`` lets the gate judge an object that was
        HANDED to the worker rather than read off its carrier.  The supported
        direct ``_run_impl(frozen)`` entry point took that argument on trust,
        so an equal-valued but non-identical object opened its source and wrote
        accepted output; now that entry traverses this same gate.
        """

        value = (getattr(self, "run_configuration", None)
                 if candidate is _CARRIER else candidate)
        frozen = require_run_configuration(
            value,
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
        # O-3N.R.1 §16.5: the accepted calibration is assigned or REFUSED; a
        # run whose identity claims no constructible calibration may not fall
        # back to whatever object the worker happened to hold.
        # O-3N.R.2 §17.7: constructibility is proved HERE, once, and the target
        # then carries VALUES -- ``PONI`` is a mutable dataclass, so keeping one
        # inside a self-described immutable target handed every consumer the
        # same editable object.
        values = getattr(frozen, "poni_values", None)
        poni_values: tuple[tuple[str, object], ...] = ()
        if values:
            values = dict(values)
            try:
                PONI.from_dict(dict(values))
            except Exception as exc:                   # noqa: BLE001
                raise RunConfigurationRefused(
                    "absent", stage="nexus-source-target",
                    detail=("the accepted calibration values are not "
                            f"constructible: {exc}"),
                    generation=int(getattr(frozen, "generation", 0) or 0))
            poni_values = tuple(values.items())
        return FrozenSourceTarget(
            uri, entry, scan_name,
            os.path.join(save_path, f"{scan_name}.nxs"),
            output_dir=save_path,
            source_base=str(getattr(frozen, "project_root", "") or ""),
            output_mode=str(getattr(frozen, "output_mode", "") or "Append"),
            poni_values=poni_values,
            generation=int(getattr(frozen, "generation", 0) or 0),
            fingerprint=str(getattr(frozen, "fingerprint", "") or ""))

    def _preflight_execution_target(self, target):
        """Prove every filesystem and HDF5 fact, in the WORKER, before any I/O.

        O-3N.R.1 §16.3/§16.6.  One idempotent worker operation; nothing that
        reads source content or touches output may precede it.  It proves:

        * the source is an existing file and the output directory is a
          directory (or creatable) -- these are filesystem facts, so they live
          here, not on the GUI thread (§16.6);
        * the output does not collide with the source, through the shared
          headless :func:`check_output_not_source` owner, so symlink/hard-link
          identity is one policy;
        * the SELECTED entry is an HDF5 **Group** carrying a RUNNABLE raw
          stack, proved by :func:`open_nexus_image_stack_exact` -- ONE open
          that both qualifies the group and binds the stack.

          O-3N.R.2 §17.3: this used to be two facts.  The group was proved with
          one ``h5py.File`` open, that handle was closed, and the shared opener
          opened the file again to bind the stack -- with its legacy fallback
          to the first ``NXentry`` still live.  Replacing the selected group
          between the two opens bound ``/fallback/instrument/detector/data``
          while the accepted source and written provenance still named
          ``selected``.  Membership alone had already been insufficient (§16.3:
          a Dataset at that path passed); the remaining gap was that
          qualification and binding described different resources.  One strict
          open closes both.

          §16.4: output ownership must not become destructive before this
          holds, or a valid-but-frameless container destroys a durable prior
          result and creates nothing.

        Returns the proved ``NexusImageStack``; the ENVELOPE owns closing it.
        """
        def refuse(detail):
            raise RunConfigurationRefused(
                "absent", stage="nexus-preflight", detail=detail,
                generation=int(target.generation))

        if not target.entry:
            refuse("the accepted NeXus configuration names no entry")
        if not target.poni_values:
            refuse("the accepted NeXus configuration carries no calibration")
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
        # Strict source open: the selected group IS the bound stack.  This is
        # the last precondition for output ownership becoming destructive.
        try:
            stack = open_nexus_image_stack_exact(target.uri, target.entry)
        except ProcessedXdartInputError:
            # F-NXS-2 keeps its actionable operator message; the STOP is now the
            # shared typed refusal, raised before any output action.
            try:
                self.showLabel.emit(
                    'Selected file is a processed xdart output, not a raw '
                    'NeXus acquisition; open it in the viewer instead of '
                    'reducing it.')
            except Exception:
                logger.debug("showLabel emit failed", exc_info=True)
            refuse("the selected file is a processed xdart output, not a raw "
                   "NeXus acquisition")
        except (KeyError, FileNotFoundError, OSError, ValueError) as exc:
            refuse(f"the selected entry carries no runnable detector stack: "
                   f"{exc}")
            raise                                      # unreachable; refuse()
        return stack

    @staticmethod
    def _append_identity(mapping):
        """The stored (uri, entry, processing signature, fingerprint) identity.

        O-3N.R.2 §17.4: the fingerprint is part of it.  Comparing only URI,
        entry and the narrower processing mapping admitted a second Start on
        the same container whose accepted PONI distance had changed -- a
        different accepted CONTENT identity appending into the first run's rows.
        """
        config = (mapping or {}).get("config") or {}
        run = config.get("run_configuration") or {}
        source = run.get("source") or {}
        return (str(source.get("uri") or ""), str(source.get("entry") or ""),
                run.get("processing_mapping") or {
                    "bai_1d_args": run.get("bai_1d_args"),
                    "bai_2d_args": run.get("bai_2d_args"),
                    "gi": run.get("gi"),
                },
                str(run.get("fingerprint") or ""))

    def _prepare_output_for_run(self, prepared, scan):
        """Own the output transaction for this run, before any writer action.

        O-3N.R.1 §16.2/§16.4, closed by O-3N.R.2 §17.4.  Two modes, two
        contracts:

        * **Append** is PROVENANCE-QUALIFIED.  An existing target is inspected
          before reduction or any writer mutation; only a target whose stored
          source URI, exact entry, processing signature AND accepted content
          fingerprint match this run is admitted.  Missing, malformed,
          foreign-source, foreign-entry, science-incompatible or
          foreign-content provenance is a visible typed refusal that leaves the
          target byte-for-byte.
        * **Overwrite** is NOT destructive here.  The prior result survives
          until the raw stack is proved and a writer action COMMITS a
          replacement, exactly once (see :meth:`_write_run_result`).

        The envelope's ``output_committed`` flag is set only after every proof
        and the reapply have succeeded.  The parent assigned its equivalent
        latch on entry, so one malformed or foreign target refused once and
        then let an exact retry return early, bypassing qualification entirely.

        XYE-only never touches the ``.nxs`` at all.
        """
        frozen = prepared.frozen
        if frozen.run_options.get("xye_only", False):
            return False
        Path(os.path.dirname(scan.data_file)).mkdir(parents=True,
                                                    exist_ok=True)
        if prepared.output_committed:
            return False
        if str(getattr(frozen, "output_mode", "Append")) == "Overwrite":
            # §16.4: retain the old target.  The first successful writer action
            # replaces it; nothing is destroyed during preparation.
            prepared.output_committed = True
            return False

        target = Path(scan.data_file)
        if not target.exists():
            prepared.output_committed = True
            return False

        def refuse(detail):
            raise RunConfigurationRefused(
                "foreign", stage="nexus-append-qualification", detail=detail,
                generation=int(getattr(frozen, "generation", 0) or 0))

        try:
            stored = read_provenance(str(target))
        except Exception as exc:                       # noqa: BLE001
            refuse(f"the Append target carries no readable provenance "
                   f"({exc}); refusing rather than mixing rows under a new "
                   "identity")
        stored_uri, stored_entry, stored_processing, stored_fingerprint = (
            nexusThread._append_identity(stored))
        accepted = frozen.source
        if not stored_uri:
            refuse("the Append target names no source; it was not written by "
                   "an admitted run")
        if stored_uri != str(accepted.uri):
            refuse(f"the Append target was written from {stored_uri!r}, not "
                   f"{str(accepted.uri)!r}")
        if stored_entry != str(accepted.entry or ""):
            refuse(f"the Append target was written from entry "
                   f"{stored_entry!r}, not {str(accepted.entry or '')!r}")
        current = processing_config_from_mapping(frozen.processing_mapping())
        check = append_config_mismatch_check(
            "Append", processing_config_from_mapping(stored_processing),
            current)
        if not check.ok:
            refuse(f"the Append target's processing configuration is "
                   f"incompatible: {check.reason}")
        # §17.4: the accepted CONTENT identity.  ``processing_mapping()`` is
        # deliberately narrower than the frozen fingerprint -- the calibration
        # lives outside it -- so this is the comparison that catches a changed
        # accepted science value on an otherwise identical source/entry.
        accepted_fingerprint = str(getattr(frozen, "fingerprint", "") or "")
        if not stored_fingerprint:
            refuse("the Append target records no content fingerprint; it was "
                   "not written by an admitted run")
        if stored_fingerprint != accepted_fingerprint:
            refuse(f"the Append target carries content identity "
                   f"{stored_fingerprint!r}, not this run's "
                   f"{accepted_fingerprint!r}")
        # Compatible: load the existing rows into THIS run's scan so new rows
        # are added once and existing durability state is preserved honestly.
        # The identity is committed only once this has SUCCEEDED.
        loader = getattr(scan, "load_from_h5", None)
        if callable(loader):
            with self.file_lock:
                loader(replace=False, mode='r')
            _apply_frozen_run_configuration(scan, frozen)
            for key, value in frozen.scan_args().items():
                setattr(scan, key, value)
        prepared.output_committed = True
        return True

    def _begin_target_replacement(self, prepared, scan):
        """Move a prior Overwrite target aside so a writer failure can undo it.

        O-3N.R.2 §17.5.  The parent unlinked the prior target and only then
        called the writer, so an injected writer failure left no old target and
        no new result.  A rename within the output directory is atomic, so
        there is no instant at which neither file exists; the backup is dropped
        on commit and restored on rollback.

        Returns the backup path, or ``None`` when there is nothing to protect.
        """
        frozen = prepared.frozen
        if str(getattr(frozen, "output_mode", "Append")) != "Overwrite":
            return None
        if prepared.target_replaced:
            return None
        target = Path(scan.data_file)
        if not target.exists():
            return None
        backup = target.with_name(target.name + _REPLACING_SUFFIX)
        with self.file_lock:
            pool = _get_h5pool()
            pool.pause(str(target))
            try:
                if backup.exists():
                    backup.unlink()
                os.replace(target, backup)
            finally:
                pool.resume(str(target))
        return backup

    def _rollback_target_replacement(self, scan, backup):
        """Restore the prior Overwrite target byte-for-byte after a failure."""
        target = Path(scan.data_file)
        with self.file_lock:
            pool = _get_h5pool()
            pool.pause(str(target))
            try:
                if target.exists():
                    target.unlink()            # discard the failed partial
                os.replace(backup, target)
            except OSError:
                # Loud, and RECOVERABLE: the prior result is still on disk
                # under the staging name, so an operator can restore it.
                logger.error(
                    '[NEXUS] Overwrite rollback failed; the prior result is '
                    'preserved at %s', backup, exc_info=True)
            finally:
                pool.resume(str(target))

    def _write_run_result(self, prepared, scan, write):
        """ONE atomic writer transaction for this run (§17.5).

        Every ``.nxs`` writer action of a NeXus run goes through here --
        periodic and final alike -- so "replace at the first SUCCESSFUL writer
        action" is a property of the transaction rather than of one call site.
        The once-only replacement identity is consumed after the commit, never
        before: a failed writer restores the prior target and leaves one exact
        retry available.
        """
        backup = nexusThread._begin_target_replacement(self, prepared, scan)
        try:
            result = write()
        except BaseException:
            if backup is not None:
                nexusThread._rollback_target_replacement(self, scan, backup)
            raise
        if str(getattr(prepared.frozen, "output_mode", "Append")) == "Overwrite":
            if not prepared.target_replaced:
                logger.info(
                    '[NEXUS] Overwrite: replaced %s at the first successful '
                    'writer action', scan.data_file)
            prepared.target_replaced = True
        if backup is not None:
            try:
                backup.unlink()
            except OSError:
                logger.warning(
                    '[NEXUS] the replaced prior target could not be removed: '
                    '%s', backup, exc_info=True)
        return result

    def _adopt_frozen_source_target(self, prepared):
        """Initialize this worker's runtime cursors FROM the accepted values.

        Cursor/handle state stays mutable — the reader needs it — but it is
        (re)initialized here from the ENVELOPE before anything opens, so a
        mirror poisoned between admission and execution is overwritten rather
        than obeyed (§14.2 item 4).

        O-3N.R.2 §17.1: it takes the prepared envelope, not a raw frozen
        object.  ``run()``, ``_run_impl()`` and ``_initialize_scan()`` each used
        to re-derive the target and rebuild a different ``PONI``; the envelope
        derives both exactly once.
        """
        target = prepared.target
        self.nexus_file = target.uri
        self.entry = target.entry
        self.scan_name = target.scan_name
        self.fname = target.output_path
        # O-3N.R §15.4 items 1-2: the calibration and the portable source base
        # are accepted values too, not panel rereads.
        self.source_base = target.source_base or None
        # O-3N.R.1 §16.5: ASSIGN the accepted calibration -- never preserve an
        # older object when the accepted run carries none.  A nonconstructible
        # mapping already refused inside ``_frozen_source_target``.
        self.poni = prepared.poni
        return target

    def _save_to_disk(self, frozen, scan):
        """The run's periodic writer action, inside the output transaction.

        §16.4/§17.5: Overwrite is destructive exactly ONCE per run, and only
        once a writer action has SUCCEEDED -- after the raw stack was proved
        and a result exists.  Later saves in the same run append into what this
        run created.  XYE-only never reaches the ``.nxs`` writer at all.
        """
        if frozen.run_options.get("xye_only", False):
            return
        prepared = nexusThread._require_execution(self, frozen)
        return nexusThread._write_run_result(
            self, prepared, scan,
            lambda: wranglerThread._save_to_disk(self, frozen, scan))

    def _flush_xye_buffer(self, scan, published_idxs=None):
        """Flush XYE, clearing an earlier longer run's tail under Overwrite.

        §16.4: an Overwrite XYE run must not leave higher-frame files from a
        previous, longer run beside this run's output.
        """
        prepared = getattr(self, "_execution", None)
        if prepared is not None:
            nexusThread._clear_stale_xye_tail(self, prepared, scan)
        return wranglerThread._flush_xye_buffer(
            self, scan, published_idxs=published_idxs)

    def _clear_stale_xye_tail(self, prepared, scan):
        """All-or-pending stale-XYE deletion for an Overwrite run (§17.6).

        Two parent defects, one owner:

        * the policy was read off mutable ``self.run_configuration``, so
          replacing that carrier after admission with a foreign Append object
          suppressed cleanup entirely.  It now comes from the ENVELOPE'S
          accepted output policy, which nothing can poison;
        * the completion latch was consumed before unlinking and ``OSError``
          was swallowed, so one transient delete failure stranded a stale
          higher-frame file permanently.  Cleanup is now all-or-pending and
          every later flush retries what is left.

        The stale set is discovered ONCE, before this run has written anything,
        so a retry can never sweep the run's own output.
        """
        if str(getattr(prepared.frozen, "output_mode", "Append")) != "Overwrite":
            return
        if prepared.xye_tail_pending is None:
            root = Path(os.path.dirname(scan.data_file)) / str(scan.name)
            prepared.xye_tail_pending = (sorted(root.glob("*.xye"))
                                         if root.is_dir() else [])
        remaining = []
        for stale in prepared.xye_tail_pending:
            try:
                stale.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                remaining.append(stale)
        prepared.xye_tail_pending = remaining
        if remaining:
            logger.warning(
                '[NEXUS] Overwrite: %d stale XYE file(s) could not be removed; '
                'cleanup stays PENDING and the next flush retries: %s',
                len(remaining), [str(p) for p in remaining])

    def _prepare_execution(self, frozen):
        """THE worker execution-preparation owner (§16.7 R.1-1, §17.8 item 1).

        One idempotent operation, reached by both ``run()`` and a supported
        direct ``_run_impl()`` entry.  In order: it admits the EXACT frozen
        object, derives the immutable target once, and proves/binds the exact
        raw stack in one strict open.  No content read and no output action
        precedes it.

        O-3N.R.2 §17.1: admission comes FIRST.  The parent adopted the supplied
        object's target and opened its source before any identity check, so the
        direct path accepted an equal-valued but non-identical configuration and
        one invocation could read a foreign source while writing accepted
        output and provenance.

        Returns the :class:`PreparedNexusExecution` envelope; the WORKER owns
        releasing it.
        """
        prepared = getattr(self, "_execution", None)
        if prepared is not None and prepared.owns(frozen):
            return prepared
        accepted = nexusThread._require_run_configuration(
            self, "nexus-execution-preparation", candidate=frozen)
        # A superseded envelope never lingers beside a new one.
        nexusThread._release_execution(self)
        stack = None
        try:
            target = nexusThread._frozen_source_target(accepted)
            stack = nexusThread._preflight_execution_target(self, target)
            prepared = PreparedNexusExecution(accepted, target, stack)
        except BaseException:
            if stack is not None:
                try:
                    stack.close()
                except Exception:                      # noqa: BLE001
                    logger.debug("stack close failed", exc_info=True)
            raise
        self._execution = prepared
        nexusThread._adopt_frozen_source_target(self, prepared)
        return prepared

    def _require_execution(self, frozen):
        """The prepared envelope for the EXACT object, or a typed refusal."""
        prepared = getattr(self, "_execution", None)
        if prepared is None or not prepared.owns(frozen):
            raise RunConfigurationRefused(
                "foreign", stage="nexus-execution",
                detail=("no prepared execution envelope owns this frozen "
                        "configuration; the run was never prepared, or the "
                        "object is not the one it was prepared for"),
                generation=int(getattr(frozen, "generation", 0) or 0))
        return prepared

    def _release_execution(self):
        """Close and clear the prepared envelope exactly once (§17.8 item 3).

        Every terminal path -- success, typed refusal, exception, Stop and
        Close -- routes here, so a proved raw-stack handle cannot outlive its
        run and the worker returns to idle.
        """
        prepared = getattr(self, "_execution", None)
        self._execution = None
        if prepared is not None:
            prepared.close()

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
        """QThread entry: run the integration body.

        O-3N.R.2 §17.2/§17.8 item 3: EVERY worker refusal is contained here --
        not just one raised by the initial preparation call.  A
        production-shaped incompatible Append refuses well after preparation,
        and the parent let that escape the QThread entry while the prepared
        stack stayed open.  A worker-boundary refusal is an expected typed
        terminal outcome: it is surfaced visibly, the envelope is released, and
        the lifecycle returns to idle.
        """
        # W-1.2 case 5 parity: refuse before any source read, output open or
        # reduction session — a worker without the accepted configuration does
        # nothing at all.
        try:
            try:
                frozen = nexusThread._require_run_configuration(
                    self, "nexus-worker-run")
                # O-3N (§14.2 item 4): derive what this run opens and where it
                # writes from the accepted object BEFORE any source read or
                # output open, so no mirror can decide either.
                nexusThread._prepare_execution(self, frozen)
                self._reset_xye_output_notifications()
                self._run_impl(frozen)
            except RunConfigurationRefused as exc:
                logger.error("run refused: %s", exc)
                self.command = 'stop'
                try:
                    self.showLabel.emit(f"Run refused: {exc}")
                except Exception:
                    logger.debug("showLabel emit failed for refusal",
                                 exc_info=True)
        finally:
            # The reduction session drains first (its finish() is the streaming
            # batch's end-of-scan write), then the source stack is released.
            try:
                self._close_reduction_session()
            finally:
                nexusThread._release_execution(self)

    def _run_impl(self, frozen):
        """Read frames from a NeXus file and integrate them in parallel."""
        # O-3N.R.1 §16.7 R.1-1: ONE worker preparation owner, reached by both
        # ``run()`` and a supported direct ``_run_impl()`` entry.  It is
        # idempotent, admits the exact object first, and nothing that reads
        # source content or touches output may precede it.
        prepared = nexusThread._prepare_execution(self, frozen)
        try:
            return nexusThread._run_body(self, prepared)
        finally:
            # §17.2: the proved raw stack was opened during preparation but only
            # entered a ``with`` block after detector/mask/scan/Append work.
            # Anything failing in between leaked the open HDF5 handle; the
            # release is unconditional, and idempotent.
            nexusThread._release_execution(self)

    def _run_body(self, prepared):
        """The integration body, executing ONLY on the prepared envelope."""
        frozen = prepared.frozen
        target = prepared.target
        ds_cm = prepared.stack
        xye_only = frozen.run_options.get("xye_only", False)
        t0 = time.time()
        # §17.1: source, entry, output and calibration are the envelope's --
        # preflight already refused an absent calibration or unreadable source,
        # so there is no silent early return to fall through here.

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
        # §16.4: the raw stack was PROVED by the preparation owner before any
        # output ownership became destructive; this is that exact resource, not
        # a second open.  A processed/frameless/dangling source refused there.
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
        """Build THIS run's own ``LiveScan`` (§16.1/§16.7 R.1-2).

        The design checkpoint was resolved in favour of a genuine fresh per-run
        scan, following the image worker, rather than a larger reset protocol on
        the display singleton.  The previous revision renamed ``self.scan`` and
        changed some fields; it cleared neither prior frame labels and resident
        frames, nor ``scan_data``, nor writer cursor/durability marks, nor
        cached integrator/mask/geometry state.  Under Overwrite those rows could
        be written into a new target under a new identity; under Append the
        whole prior display/browse identity participated in the write.

        The display scan is neither executed on nor reset here.  Publication
        still reaches the GUI through the existing per-frame handoff.

        O-3N.R.2 §17.1: the target is the ENVELOPE'S, not a third derivation.
        §17.2/§17.8 item 3: an Append refusal raised from here releases the
        envelope, so the proved raw stack cannot outlive the refusal.
        """
        frozen = nexusThread._require_run_configuration(
            self, "nexus-initialize-scan")
        prepared = nexusThread._prepare_execution(self, frozen)
        try:
            target = prepared.target
            scan_kwargs = frozen.scan_kwargs()
            scan = LiveScan(
                scan_name,
                # §15.1: the run owns its output BEFORE anything can write.
                # Both saves and every XYE file resolve through ``data_file``.
                data_file=target.output_path,
                static=True,
                gi=bool(scan_kwargs["gi"]),
                incidence_motor=scan_kwargs["incidence_motor"],
                series_average=False,
                global_mask=self.mask,
                # J2: share the wrangler's writer lock, as the image worker does.
                file_lock=self.file_lock,
                bai_1d_args=scan_kwargs["bai_1d_args"],
                bai_2d_args=scan_kwargs["bai_2d_args"],
            )
            _apply_frozen_run_configuration(scan, frozen)
            # N1: the accepted project root -> entry/@source_base + relative raw
            # source paths in the writer (portable .nxs).
            scan.source_base = target.source_base or None
            prepared.scan = scan
            self._active_scan = scan
            nexusThread._prepare_output_for_run(self, prepared, scan)
        except BaseException:
            nexusThread._release_execution(self)
            raise
        return scan

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


