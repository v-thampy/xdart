# -*- coding: utf-8 -*-
"""
nexusThread — worker thread for NeXus/Tiled wrangler.

Reads frames from a NeXus HDF5 file (Bluesky suitcase-nexus format)
and integrates each one using the same LiveFrame pipeline as imageThread.

@author: thampy
"""

# Standard library imports
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np

# Qt imports
from pyqtgraph import Qt

# Project imports
from xdart.modules.live import LiveFrame, LiveScan
from xrd_tools.core.containers import PONI
from xrd_tools.io.output_safety import (
    OutputCollisionError,
    check_output_not_source,
)
from xrd_tools.integrate.calibration import poni_to_integrator, get_detector
from xrd_tools.io import (
    AppendDisposition, AppendExternalMember, AppendIntent, AppendRefused,
    AppendSource, AppendSourceGraphRefused, decode_committed_append_prefix,
    qualify_append, resolve_output_target, science_fingerprint,
    truncate_append_source,
)
from xrd_tools.io.nexus import open_nexus_execution_source
from xrd_tools.io.image import read_image
from xrd_tools.io.processed_scan_id import ProcessedXdartInputError
from xrd_tools.session import DynamicFrameIdentity, required_result_modes
from xrd_tools.session.run_configuration import (
    RunConfigurationRefused,
    require_run_configuration,
)
from xdart.modules.reduction import (
    apply_frozen_run_configuration as _apply_frozen_run_configuration,
    freeze_live_scan_gi_ranges,
    StandardPlanCache,
    sync_live_scan_gi_settings,
)
from xrd_tools.sources.adapters import candidate_owner
from xrd_tools.sources.discover import Candidate
import xrd_tools.sources.registry  # noqa: F401
from .wrangler_widget import (
    wranglerThread,
)

logger = logging.getLogger(__name__)

#: Sentinel: "judge the object on my carrier", as distinct from an explicitly
#: supplied candidate that happens to be ``None`` (which must still refuse).
_CARRIER = object()


def _source_candidate(path):
    path = Path(path)
    stat = path.stat()
    owner = candidate_owner(path)
    if owner is None:
        raise ValueError(f"no source adapter owns {path}")
    return Candidate(path, owner.id, int(stat.st_size), int(stat.st_mtime_ns))

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
    """Retryable custody for one admitted source handle and detached metadata."""

    __slots__ = (
        "frozen",
        "target",
        "stack",
        "scan_metadata",
        "poni",
        "master",
        "source",
        "observations",
        "_adopted",
        "_closed",
    )

    def __init__(self, frozen, target, master, poni):
        self.frozen = frozen
        self.target = target
        self.master = master
        self.stack = None
        self.scan_metadata = None
        self.poni = poni
        self.source = None
        self.observations = ()
        self._adopted = False
        self._closed = False

    def owns(self, frozen) -> bool:
        """True only for the exact object this envelope was prepared for."""
        return frozen is self.frozen

    @property
    def adopted(self) -> bool:
        """Whether every fallible worker-cursor adoption step completed."""
        return self._adopted

    def mark_adopted(self) -> None:
        self._adopted = True

    def close(self) -> None:
        """Close the raw stack exactly once, retaining a failed close."""
        if self._closed:
            return
        stack = self.stack
        if stack is not None:
            stack.close()
            self.stack = None
        self._closed = True


# The session policy resolves its worker grant from frozen.max_cores; no worker
# fallback or cached duplicate remains.


class nexusThread(wranglerThread):
    """Thread for processing NeXus/HDF5 image stacks.

    Reads an image dataset from a NeXus file, integrates each frame
    through the headless reduction spine, and emits ``sigUpdate`` after
    each one.

    signals:
        showLabel: str, status text for the UI label
    """
    showLabel = Qt.QtCore.Signal(str)
    sigRetainedCustody = Qt.QtCore.Signal(int, str)

    def __init__(
            self,
            command_queue,
            file_lock,
            fname,
            nexus_file,
            poni,
            gi_mode_1d,
            gi_mode_2d,
            command,
            scan,
            entry='entry',
            parent=None):
        """R4-G: ``scan_args``, ``mask_file``, ``gi``, ``th_mtr``,
        ``sample_orientation`` and ``tilt_angle`` were retired here.

        W-1R-D deleted the worker slots they seeded (review §44.2); the mask
        and the GI geometry are read from the accepted
        ``FrozenRunConfiguration``, and ``scan_args`` was an always-empty
        construction-time snapshot the base class never stored.
        """

        super().__init__(command_queue, fname, file_lock, parent)

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
        # The one prepared execution envelope for the run: exact admitted
        # object, derived target, proved raw stack, source graph, and scan.
        self._execution = None

        # Settable from outside (e.g. before .start()) when the GUI
        # eventually exposes a Cores spinbox for the NeXus wrangler.
        # C1: cached standard ReductionPlan, rebuilt only when scan
        # settings change.  Lives on the thread so it survives across
        # chunks within a single run.
        self._plan_cache = StandardPlanCache()

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
            os.fspath(resolve_output_target(
                save_path, scan_name, mode=frozen.output_mode)),
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
          stack, proved by :func:`open_nexus_execution_source` -- ONE open
          that qualifies the group, binds the stack AND detaches the scan
          metadata.

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

          O-3N.R.3 §19.1 completes the closure: scan METADATA is part of the
          same accepted resource.  The body's later ``read_nexus(path, entry)``
          reopened the pathname, so a replacement after preparation fed the
          run counters and motor angles from a different inode than its raw
          pixels -- a scientific TOCTOU.  The metadata is detached here, on
          the stack's own handle, and nothing later re-resolves the pathname.

          §16.4: output ownership must not become destructive before this
          holds, or a valid-but-frameless container destroys a durable prior
          result and creates nothing.

        Returns the proved ``NexusExecutionSource`` (stack + detached
        metadata); the ENVELOPE owns closing the stack.
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
        # Strict source open: the selected group IS the bound stack, and the
        # detached scan metadata rides the same held handle (§19.1).  This is
        # the last precondition for output ownership becoming destructive.
        try:
            source = open_nexus_execution_source(target.uri, target.entry)
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
        return source

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

        O-3N.R.3 §19.2: publication comes LAST, and the whole interval is
        ``BaseException``-total.  The parent assigned ``self._execution`` and
        only then adopted the worker cursors; an adoption fault (including an
        interrupt) left the envelope PUBLISHED and its exact HDF5 handle live,
        with the direct entry's ``finally`` not yet armed.  Nothing may
        survive a failed transfer: every fallible step runs first, and any
        throwable closes the envelope before it can be observed.

        Returns the :class:`PreparedNexusExecution` envelope; the WORKER owns
        releasing it.
        """
        prepared = getattr(self, "_execution", None)
        if prepared is not None and prepared.owns(frozen):
            if prepared.adopted:
                return prepared
            # A prior adoption fault retained this envelope solely as a
            # cleanup owner.  Complete that cleanup before preparing anew.
            nexusThread._release_execution(self)
        accepted = nexusThread._require_run_configuration(
            self, "nexus-execution-preparation", candidate=frozen)
        # A superseded envelope never lingers beside a new one.
        nexusThread._release_execution(self)
        prepared = None
        try:
            target = nexusThread._frozen_source_target(accepted)
            try:
                master = _source_candidate(target.uri)
            except (OSError, ValueError) as exc:
                raise AppendSourceGraphRefused(
                    f"master observation failed: {exc}",
                    source_generation=int(accepted.generation)) from exc
            prepared = PreparedNexusExecution(
                accepted, target, master, target.poni())
            source = nexusThread._preflight_execution_target(self, target)
            prepared.stack = source.stack
            prepared.scan_metadata = source.scan_metadata
            prepared.source, prepared.observations = nexusThread._source_graph(
                prepared, accepted.generation)
            nexusThread._adopt_frozen_source_target(self, prepared)
            prepared.mark_adopted()
        except BaseException as primary:
            if prepared is not None:
                try:
                    prepared.close()
                except BaseException as cleanup:
                    # The envelope is not executable, but remains the exact
                    # retry owner for its source.  A later public release or
                    # preparation attempt completes only this cleanup.
                    self._execution = prepared
                    raise cleanup from primary
            raise
        self._execution = prepared
        return prepared

    def _require_execution(self, frozen):
        """The prepared envelope for the EXACT object, or a typed refusal."""
        prepared = getattr(self, "_execution", None)
        if (prepared is None or not prepared.owns(frozen)
                or not prepared.adopted):
            raise RunConfigurationRefused(
                "foreign", stage="nexus-execution",
                detail=("no prepared execution envelope owns this frozen "
                        "configuration; the run was never prepared, or the "
                        "object is not the one it was prepared for"),
                generation=int(getattr(frozen, "generation", 0) or 0))
        return prepared

    def _release_execution(self):
        """Close and clear the exact source envelope after graph cleanup."""
        prepared = getattr(self, "_execution", None)
        if prepared is not None:
            prepared.close()
            self._execution = None

    def release_retained_custody(self):
        if (self._scan_session_adapter is not None
                and not self._close_reduction_session()):
            return False
        try:
            nexusThread._release_execution(self)
        except BaseException as exc:
            self._retain_dynamic_failure(exc, "Output cleanup remains pending")
            return False
        return super().release_retained_custody()

    def dynamic_cleanup_pending(self):
        return self._execution is not None or super().dynamic_cleanup_pending()

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
        """Run one admitted fixed snapshot through the dynamic session."""
        outcome = None
        succeeded = False
        try:
            try:
                frozen = nexusThread._require_run_configuration(
                    self, "nexus-worker-run")
                if frozen.run_options.get("xye_only", False):
                    raise RunConfigurationRefused(
                        "absent", stage="nexus-dynamic-output",
                        detail="dynamic XYE output is not mounted",
                        generation=int(frozen.generation))
                self._reduction_write_error = None
                outcome = self._run_impl(frozen)
                succeeded = True
            except (RunConfigurationRefused, AppendRefused) as exc:
                logger.error("run refused: %s", exc)
                self.command = 'stop'
                try:
                    self.showLabel.emit(f"Run refused: {exc}")
                except Exception:
                    logger.debug("showLabel emit failed for refusal",
                                 exc_info=True)
        finally:
            clean = self._close_reduction_session()
            if clean:
                try:
                    nexusThread._release_execution(self)
                except BaseException as exc:
                    clean = False
                    self._retain_dynamic_failure(
                        exc, "Output cleanup remains pending")
            if (succeeded and clean and outcome != "skip"
                    and self.command != "stop"):
                self.showLabel.emit(
                    f"Done — {int(outcome or 0)} frames processed")

    def _run_impl(self, frozen):
        """Submit the retained NeXus stack through one dynamic session."""
        prepared = nexusThread._prepare_execution(self, frozen)
        return nexusThread._run_body(self, prepared)

    @staticmethod
    def _source_graph(prepared, generation):
        stack, master = prepared.stack, prepared.master

        def refuse(reason):
            raise AppendSourceGraphRefused(
                f"unsupported NeXus detector topology: {reason}",
                source_generation=int(generation))

        paths = tuple(stack._paths)
        datasets = tuple(stack._dsets)
        offsets = tuple(stack._offsets)
        if (not paths or len(paths) != len(datasets)
                or len(offsets) != len(paths) + 1):
            refuse("incomplete retained stack")
        members, observations, kinds = [], [master], set()
        for ordinal, (selector, dataset) in enumerate(zip(paths, datasets)):
            owner = stack._h5
            parts = [part for part in str(selector).split("/") if part]
            if not parts:
                refuse("empty detector path")
            for part in parts[:-1]:
                link = owner.get(part, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    refuse("indirect ancestor")
                owner = owner.get(part)
                if not isinstance(owner, h5py.Group):
                    refuse("non-group ancestor")
            link = owner.get(parts[-1], getlink=True)
            if not isinstance(link, (h5py.HardLink, h5py.ExternalLink)):
                refuse("indirect detector leaf")
            if (not isinstance(dataset, h5py.Dataset)
                    or dataset.ndim not in (2, 3) or dataset.is_virtual
                    or dataset.id.get_create_plist().get_external_count()):
                refuse("non-owned or invalid-rank storage")
            kind = "external" if isinstance(link, h5py.ExternalLink) else "hard"
            kinds.add(kind)
            if kind == "hard":
                if os.path.abspath(dataset.file.filename) != os.path.abspath(master.path):
                    refuse("hard link escaped the master")
                continue
            member_path = Path(link.filename)
            if not member_path.is_absolute():
                member_path = Path(master.path).parent / member_path
            member_path = Path(os.path.abspath(member_path))
            if (os.path.abspath(dataset.file.filename) != os.path.abspath(member_path)
                    or dataset.name != str(link.path)):
                refuse("external link is not a direct leaf")
            try:
                observed = _source_candidate(member_path)
            except (OSError, ValueError) as exc:
                refuse(f"member observation failed: {exc}")
            observations.append(observed)
            members.append(AppendExternalMember(
                str(member_path), str(link.path), observed.size,
                observed.mtime_ns, int(offsets[ordinal]),
                int(offsets[ordinal + 1]), ordinal))
        if len(kinds) != 1:
            refuse("mixed storage owners")
        source = AppendSource(
            os.path.abspath(master.path), master.adapter_id, master.size,
            master.mtime_ns, int(stack.shape[0]), dataset_paths=paths,
            external_members=tuple(members), generation=int(generation))
        return source, tuple(observations)

    @staticmethod
    def _require_source_unchanged(observations, generation):
        for observed in observations:
            try:
                current = _source_candidate(observed.path)
            except (OSError, ValueError) as exc:
                raise AppendSourceGraphRefused(
                    f"source member disappeared: {exc}",
                    source_generation=int(generation)) from exc
            if current != observed:
                raise AppendSourceGraphRefused(
                    f"source changed before mount: {observed.path}",
                    source_generation=int(generation))

    @staticmethod
    def _frontier_intent(full, extent, generation):
        source = replace(full.source, generation=int(generation))
        return replace(
            full, source=truncate_append_source(source, int(extent)),
            labels=tuple(range(int(extent))))

    def _run_body(self, prepared):
        """Submit one exact fixed source graph through one dynamic adapter."""
        frozen, target, stack = prepared.frozen, prepared.target, prepared.stack
        scan_meta, base_meta = prepared.scan_metadata, {}
        try:
            for values in (scan_meta.counters, scan_meta.angles):
                for name, column in values.items():
                    if len(column):
                        base_meta[name] = float(column[0])
        except Exception:
            scan_meta, base_meta = None, {}

        scan = self._initialize_scan(target.scan_name)
        provisional = self._plan_cache.get(
            scan, integrate_2d=not frozen.skip_2d)
        policy = self._resolve_dynamic_session_policy(
            frozen=frozen, plan=provisional, frame_shape=stack.shape[1:],
            dtype=stack.dtype)
        modes = tuple(f"{mode.kind}:{mode.key}"
                      for mode in required_result_modes(provisional))
        full = AppendIntent(
            target.entry, target.source_base, os.path.abspath(target.uri),
            science_fingerprint(frozen.processing_mapping()), modes,
            prepared.source, tuple(range(int(stack.shape[0]))))
        observations = prepared.observations
        nexusThread._require_source_unchanged(
            observations, full.source.generation)

        prefix, decision = None, None
        if target.output_mode == "Append":
            with self.file_lock:
                if Path(target.output_path).exists():
                    try:
                        with h5py.File(target.output_path, "r") as handle:
                            prefix = decode_committed_append_prefix(
                                handle, entry=target.entry)
                    except (OSError, ValueError, TypeError, KeyError):
                        prefix = None
                if prefix is not None:
                    k0 = len(prefix.intent.labels)
                    if prefix.intent.labels != tuple(range(k0)):
                        raise AppendSourceGraphRefused(
                            "committed Nexus labels are not a zero-based prefix",
                            source_generation=int(frozen.generation))
                    generation = max(
                        int(frozen.generation),
                        int(prefix.intent.source.generation) + 1)
                    full = nexusThread._frontier_intent(
                        full, stack.shape[0], generation)
                decision = qualify_append(
                    target.output_path, full, committed_prefix=prefix)
            if decision.disposition is AppendDisposition.REFUSE:
                raise AppendRefused(decision)
            k0 = len(prefix.intent.labels) if prefix is not None else 0
            scan._committed_append_prefix = prefix
            if decision.disposition is AppendDisposition.SKIP:
                self.files_processed = self._last_files_processed = 0
                self.showLabel.emit("Already processed — exact source is committed")
                return "skip"
        else:
            k0 = 0
            scan._committed_append_prefix = None

        self.detector = (get_detector(self.poni.detector)
                         if self.poni.detector else None)
        det_mask = self.detector.mask if self.detector is not None else None
        if frozen.mask_file and os.path.exists(frozen.mask_file):
            custom = np.asarray(read_image(frozen.mask_file), dtype=bool)
            det_mask = det_mask | custom if det_mask is not None else custom
        self.mask = np.flatnonzero(det_mask) if det_mask is not None else None
        nexusThread._project_gi_modes_onto_display_scan(self, frozen)
        scan._cached_integrator = poni_to_integrator(self.poni)
        scan._cached_poni = self.poni
        scan._cached_fiber_integrator = None
        sync_live_scan_gi_settings(
            scan, incidence_motor=frozen.gi.scan_incidence_motor,
            sample_orientation=frozen.gi.sample_orientation,
            tilt_angle=frozen.gi.tilt_angle)
        width = min(policy.flush.interval, policy.flush.hard_threshold())
        frontier = min(int(stack.shape[0]), k0 + width)
        intent = nexusThread._frontier_intent(
            full, frontier, full.source.generation)
        first_data = np.asarray(stack[k0])
        first = self._build_frame(
            frozen, scan, k0, first_data,
            self._frame_meta(scan_meta, base_meta, k0))
        if frozen.gi.enabled:
            freeze_live_scan_gi_ranges(
                scan, (first,), scan_name=str(scan.name),
                global_mask=self.mask, integrator=scan._cached_integrator,
                poni=self.poni, integrate_2d=not frozen.skip_2d,
                gi_freeze_mode="first_frame")
        final_plan = self._plan_cache.get(
            scan, integrate_2d=not frozen.skip_2d)
        final_modes = tuple(f"{mode.kind}:{mode.key}"
                            for mode in required_result_modes(final_plan))
        if final_modes != modes:
            raise AppendSourceGraphRefused(
                "GI freeze changed the qualified result modes",
                source_generation=int(intent.source.generation))
        nexusThread._require_source_unchanged(
            observations, intent.source.generation)
        scan._same_run_intent = intent
        adapter = self._mount_dynamic_reduction_session(
            (int(frozen.generation), os.path.abspath(target.output_path)),
            frozen=frozen, scan=scan, plan=final_plan, pending_frame=first,
            output_path=os.path.abspath(target.output_path),
            gui_thread_id=self.gui_thread_id, policy=policy)
        self.sigUpdateFile.emit(
            str(scan.name), os.path.abspath(target.output_path),
            bool(frozen.gi.enabled), str(frozen.gi.scan_incidence_motor),
            False, False)

        accepted = 0
        revision = max(1, *(int(item.mtime_ns) for item in observations))
        for frame_idx in range(k0, int(stack.shape[0])):
            if self.command == "stop":
                break
            live = first if frame_idx == k0 else self._build_frame(
                frozen, scan, frame_idx, np.asarray(stack[frame_idx]),
                self._frame_meta(scan_meta, base_meta, frame_idx))
            key = DynamicFrameIdentity(
                os.path.abspath(target.uri), int(frame_idx))
            adapter.discover(
                key, group=os.path.abspath(target.output_path),
                ordinal=int(frame_idx), output_label=int(frame_idx))
            submitted = False
            for _attempt in range(32):
                token = adapter.begin_attempt(key, source_revision=revision)
                adapter.record_enqueued(token)
                if adapter.submit(live, attempt_token=token):
                    submitted = True
                    break
                adapter.record_failed(token, retryable=True)
                if (self.command == "stop"
                        or not adapter.quiesce(timeout=60.0)):
                    break
                adapter.resume()
            if not submitted:
                if self.command == "stop":
                    break
                self._retain_dynamic_failure(
                    RuntimeError(f"dynamic submit refused frame {frame_idx}"),
                    "Dynamic submit refused")
                break
            accepted += 1
            self._frames_since_save += 1
            if adapter.should_flush(
                    self._frames_since_save, unsaved_in_memory=None):
                adapter.commit_epoch()
                self._frames_since_save = 0
                if frontier < int(stack.shape[0]):
                    frontier = min(int(stack.shape[0]), frontier + width)
                    intent = nexusThread._frontier_intent(
                        full, frontier, intent.source.generation + 1)
                    adapter.extend_live(intent)
                    scan._same_run_intent = intent
        self.files_processed = self._last_files_processed = accepted
        return accepted

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
            self._active_scan = scan
        except BaseException:
            nexusThread._release_execution(self)
            raise
        return scan

    def _frame_meta(self, scan_meta, base_meta, frame_idx):
        """Build a per-frame metadata dict from scan-level arrays.

        Falls back to ``base_meta`` (frame 0's slice) when the scan
        arrays are shorter than expected.
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
