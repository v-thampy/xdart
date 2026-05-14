import logging
import os
from threading import Condition, _PyRLock

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .arch import EwaldArch
from .arch_series import ArchSeries
from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xdart import utils


class EwaldSphere:
    """Class for storing multiple arch objects in v2 NeXus-formatted HDF5 files.

    Output file structure (xdart v2 schema, written by
    :func:`xdart.modules.ewald.nexus_writer.save_sphere_to_nexus`)::

        scan.nxs
        └── entry/                       (NXentry)
            ├── calibration/             (PONI geometry)
            ├── instrument/detector/     (incl. global mask, flat pixel indices)
            ├── sample/positioners/      (per-frame motor positions)
            ├── frames/                  (per-frame NXcollection — thumbnails)
            │   └── frame_NNNN/thumbnail
            ├── integrated_1d/           (stacked NXdata, (N, nq))
            ├── integrated_2d/           (stacked NXdata, (N, nchi, nq_2d))
            ├── per_frame_geometry/      (rot1/rot2/rot3/incident_angle)
            └── reduction/               (NXprocess provenance)
    """

    def __init__(self, name='scan0', arches=None, data_file=None,
                 scan_data=None, mg_args=None,
                 bai_1d_args=None, bai_2d_args=None,
                 static=False, gi=False, th_mtr=None,
                 incidence_motor=None, geometry=None,
                 series_average=False,
                 single_img=False,
                 global_mask=None,
                 file_lock=None,
                 **_unused,
                 ):
        super().__init__()
        # None-sentinel pattern: mutable defaults (lists, dicts, DataFrames)
        # in function signatures are shared across all calls that omit them,
        # so any mutation leaks between instances. Resolve them here instead.
        if arches is None:
            arches = []
        if scan_data is None:
            scan_data = pd.DataFrame()
        if mg_args is None:
            mg_args = {'wavelength': 1e-10}
        if bai_1d_args is None:
            bai_1d_args = {}
        if bai_2d_args is None:
            bai_2d_args = {}

        # J2: file_lock unification.  Callers can pass in their own
        # ``threading.Condition`` so the wrangler-level file_lock
        # (used by save paths under ``self.file_lock``) is the same
        # lock that ``ArchSeries.__getitem__`` uses for lazy loads.
        # Pre-J2 each sphere created its own Condition() and the
        # wrangler's GUI file_lock was a *different* lock — direct
        # ``ArchSeries.__getitem__`` reads could happen mid-save
        # while the wrangler thought the file was quiescent.  With a
        # single shared lock the read just waits.  Existing callers
        # that don't pass file_lock still work — they just get the
        # legacy private-lock behavior.
        self.file_lock = file_lock if file_lock is not None else Condition()
        if name is None:
            self.name = os.path.split(data_file)[-1].split('.')[0]
        else:
            self.name = name
        if data_file is None:
            self.data_file = name + ".nxs"
        else:
            self.data_file = data_file

        self.static = static
        self.gi = gi
        # th_mtr is the legacy name; incidence_motor is the canonical name
        # going forward.  th_mtr kwarg defaults to None so older callers
        # passing th_mtr='th' still work, and incidence_motor mirrors it.
        if incidence_motor is None:
            incidence_motor = th_mtr if th_mtr is not None else "th"
        if th_mtr is None:
            th_mtr = incidence_motor
        self.th_mtr = th_mtr               # deprecated alias
        self.incidence_motor = incidence_motor
        # Flexible diffractometer geometry — used by the v2 NeXus writer
        # to derive per-frame pyFAI rotations + incidence-angle arrays.
        self.geometry = geometry            # ssrl_xrd_tools.core.geometry.DiffractometerGeometry
        # Optional stitched-output containers (populated by run_stitch).
        self.stitched_1d = None
        self.stitched_2d = None
        self.single_img = single_img
        self.series_average = series_average
        self.skip_2d = False
        self._cached_integrator = None
        self._cached_fiber_integrator = None

        if arches:
            self.arches = ArchSeries(self.data_file, self.file_lock, arches,
                                     static=self.static, gi=self.gi)
        else:
            self.arches = ArchSeries(self.data_file, self.file_lock,
                                     static=self.static, gi=self.gi)
        self.scan_data = scan_data

        # ``mg_args`` retained: nexus_writer reads
        # ``mg_args["wavelength"]`` for the NXsource stamp.  The
        # ``_mg_integrators`` list that powered the deleted
        # ``multigeometry_integrate_*`` API is gone — stitching now
        # goes through :mod:`xdart.modules.ewald.stitch` instead.
        self.mg_args = mg_args

        self.bai_1d_args = bai_1d_args
        self.bai_2d_args = bai_2d_args
        self.sphere_lock = Condition(_PyRLock())

        self.bai_1d: IntegrationResult1D | None = None
        self.bai_2d: IntegrationResult2D | None = None

        # G2: ``overall_raw`` was a sum-of-raw-frames accumulator
        # consumed only by display_data.get_sphere_map_raw (now
        # deleted).  Drifted from disk under R1 replace-frames and
        # was never repopulated on v2 reload.
        self.global_mask = global_mask

    def reset(self):
        """Resets all held data objects to blank state."""
        with self.sphere_lock:
            self.scan_data = pd.DataFrame()
            self.arches = ArchSeries(self.data_file, self.file_lock,
                                     static=self.static, gi=self.gi)
            self.global_mask = None
            self.bai_1d = None
            self.bai_2d = None

    def has_reload_only_frames(self) -> bool:
        """Return True iff any arch can't recover its raw image.

        Used by the GUI reintegrate buttons (the R3 guardrail).  After
        L1 wired lazy raw load, an arch's ``is_reload_only`` is True
        only when neither ``arch.map_raw`` nor a resolvable
        ``arch.source_file`` is available — i.e. the original raw
        frames have been moved/deleted relative to the .nxs.

        Path A — in-memory cache has entries: definitive answer from
        the cache.  Re-integration buttons get the right answer
        instantly.

        Path B — empty cache (freshly opened .nxs, before any arch
        has been materialised): probe the first frame's
        ``/entry/frames/frame_NNNN/source/path`` directly from h5py
        WITHOUT materialising an :class:`EwaldArch`.  This is the C2
        fix — the pre-C2 code returned ``True`` whenever the cache
        was empty + the index had rows, which falsely blocked
        re-integration on a freshly opened .nxs even when every
        source file was present and lazy load would have worked.
        Probing one frame is a single tiny h5 read (~ms).
        """
        in_mem = getattr(self.arches, "_in_memory", None)
        if in_mem:
            return any(getattr(a, "is_reload_only", False)
                       for a in in_mem.values())
        if not self.arches.index:
            return False
        return self._probe_reload_only_via_h5()

    def _probe_reload_only_via_h5(self) -> bool:
        """Sample one frame's source ref from the .nxs to gauge the flag.

        Reads ``/entry/frames/frame_NNNN/source/path`` for the first
        index in ``self.arches.index`` and checks whether it resolves
        to an existing file.  We assume the wrangler stamped sources
        uniformly across the scan, so one probe represents the whole
        scan — if not, the per-arch :attr:`is_reload_only` flag
        (set by :func:`_load_arch_v2` on each lazy load) catches
        outliers when they're actually materialised.

        Returns ``True`` (conservative — block re-integration) on any
        read error: missing group, decoding failure, h5py error.
        ``False`` only when the source path is on disk.
        """
        try:
            from xdart.utils import catch_h5py_file as _catch
        except Exception:
            return True
        try:
            first_idx = int(self.arches.index[0])
        except (IndexError, ValueError, TypeError):
            return True
        try:
            with _catch(self.data_file, 'r') as f:
                grp_path = f"entry/frames/frame_{first_idx:04d}/source"
                grp = f.get(grp_path)
                if grp is None or "path" not in grp:
                    return True  # no source ref at all → not recoverable
                raw = grp["path"][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                full = str(raw)
                if not os.path.isabs(full):
                    full = os.path.normpath(
                        os.path.join(
                            os.path.dirname(self.data_file), full,
                        )
                    )
                return not os.path.exists(full)
        except (OSError, KeyError, ValueError, TypeError, AttributeError):
            return True

    def add_arch(self, arch=None, calculate=True, update=True, get_sd=True,
                 h5file=None, batch_save=False, **kwargs):
        """Adds a new arch to the sphere.

        In-memory state (``arches.index``, ``scan_data``,
        ``bai_1d``/``bai_2d`` accumulators) is always updated.
        Persistence policy:

        * ``batch_save=True`` — skip the per-frame ``_save_to_nexus`` call;
          the caller is responsible for invoking the v2 writer once at end
          of batch.
        * ``batch_save=False`` (default, used by live mode) — call
          ``_save_to_nexus(h5file)`` once per arch.  The v2 writer is
          idempotent and uses stacked slice-assigns, so per-frame calls
          are O(1) in the dataset shape rather than O(N) like the old v1
          delete-and-recreate pattern.

        Pre-F6 this method also took ``set_mg=True`` which
        rebuilt ``self._mg_integrators`` (used only by the
        deleted ``multigeometry_integrate_*`` API).  Callers that
        still pass ``set_mg`` are silently absorbed by ``**kwargs``;
        new callers should drop the kwarg.
        """
        # Eat any stale ``set_mg`` kwarg for backwards compat with
        # one or two release-old call sites.  Other unknown kwargs
        # land in **kwargs and are forwarded to EwaldArch below.
        kwargs.pop("set_mg", None)
        with self.sphere_lock:
            if arch is None:
                arch = EwaldArch(**kwargs)
            if calculate:
                arch.integrate_1d(global_mask=self.global_mask, **self.bai_1d_args)
                arch.integrate_2d(global_mask=self.global_mask, **self.bai_2d_args)
            arch.file_lock = self.file_lock
            # In-memory append only; ArchSeries.__setitem__ does no disk I/O.
            if arch.idx not in self.arches.index:
                self.arches.index.append(arch.idx)
                self.arches.index.sort()
            # Stash the live arch object so the v2 writer can read its
            # int_1d/int_2d/thumbnail without going back to disk.
            self.arches.stash(arch)

            if arch.scan_info and get_sd:
                ser = pd.Series(arch.scan_info, dtype='float64')
                if list(self.scan_data.columns):
                    try:
                        self.scan_data.loc[arch.idx] = ser
                    except ValueError:
                        print('Mismatched columns')
                else:
                    self.scan_data = pd.DataFrame(
                        arch.scan_info, index=[arch.idx], dtype='float64'
                    )
                self.scan_data.sort_index(inplace=True)

            if update:
                self._accumulate_bai_1d(arch)
                if not self.skip_2d:
                    self._accumulate_bai_2d(arch)

            # G2: ``self.overall_raw += (arch.map_raw - arch.bg_raw)``
            # removed.  The accumulator's only consumer was
            # ``display_data.get_sphere_map_raw``; the Overall view
            # now aggregates over per-arch ``data_2d['map_raw']``,
            # which stays correct after R1 replace-frames and v2
            # reload.

            # Persist via the v2 writer.  Idempotent; one slice-assign per
            # stacked dataset.  Skipped in batch mode (caller flushes once
            # at end of batch).  The writer opens its own file — ignore
            # any ``h5file`` argument (kept for signature compatibility).
            if not batch_save:
                self._save_to_nexus()

    def _accumulate_bai_1d(self, arch):
        """In-memory running sum of 1D integration results (no HDF5 write)."""
        with self.sphere_lock:
            if arch.int_1d is None:
                return
            try:
                if self.bai_1d is None:
                    self.bai_1d = arch.int_1d
                else:
                    self.bai_1d = self.bai_1d + arch.int_1d
            except (ValueError, AttributeError):
                self.bai_1d = arch.int_1d

    def _accumulate_bai_2d(self, arch):
        """In-memory running sum of 2D integration results (no HDF5 write)."""
        with self.sphere_lock:
            if arch.int_2d is None:
                return
            try:
                if self.bai_2d is None:
                    self.bai_2d = arch.int_2d
                else:
                    self.bai_2d = self.bai_2d + arch.int_2d
            except (ValueError, AttributeError):
                self.bai_2d = arch.int_2d

    # ------------------------------------------------------------------
    # v2 NeXus persistence
    # ------------------------------------------------------------------

    def save_to_nexus(self, *, entry: str = "entry", finalize: bool = False,
                      replace: bool = False,
                      replace_frame_indices=None) -> None:
        """Save sphere state into a v2 NeXus file.  Idempotent across calls.

        Two modes — see :func:`nexus_writer.save_sphere_to_nexus` for
        the full contract:

        * ``replace_frame_indices=None`` (default): append-only.  Stacked
          datasets grow by however many new frames have been added
          since the last save.
        * ``replace_frame_indices=[...]``: slice-assign recomputed
          integrated_1d/2d rows over their existing on-disk positions.
          Used by GUI reintegration (``sphere_threads.bai_1d_all``).

        The writer owns its own file handle (with NFS-retry semantics)
        so the caller only needs to hold ``self.file_lock``.
        """
        mode = 'w' if replace else 'a'
        with self.file_lock:
            self._save_to_nexus(
                mode=mode, entry=entry, finalize=finalize,
                replace_frame_indices=replace_frame_indices,
            )

    def _save_to_nexus(self, *, mode: str = "a", entry: str = "entry",
                       finalize: bool = False,
                       replace_frame_indices=None) -> None:
        """Inner v2 writer; delegates to ``nexus_writer.save_sphere_to_nexus``.

        Opens the file at ``self.data_file`` internally.  Callers must
        NOT hold an open h5py.File on the same path during this call —
        HDF5 single-writer semantics will reject the second open.
        """
        from xdart.modules.ewald.nexus_writer import save_sphere_to_nexus
        with self.sphere_lock:
            save_sphere_to_nexus(
                self, self.data_file, mode=mode,
                entry=entry, finalize=finalize,
                replace_frame_indices=replace_frame_indices,
            )

    def load_from_h5(self, replace=True, mode='r', *args, **kwargs):
        """Load sphere state from a v2 NeXus file.

        Kept the legacy method name (``load_from_h5``) so callers across
        the GUI don't need updating, but the implementation is v2-only.
        """
        with self.file_lock:
            if replace:
                self.reset()
            with utils.catch_h5py_file(self.data_file, mode=mode) as file:
                self._load_from_h5(file, *args, **kwargs)

    def _load_from_h5(self, grp, data_only=False, **_unused):
        """Load sphere state from a v2 NeXus file (no-op dispatcher).

        ``**_unused`` absorbs any historical kwargs (notably
        ``set_mg`` from before F6 removed the multigeometry API).
        """
        with self.sphere_lock:
            self._load_from_nexus_v2(grp, data_only=data_only)

    def set_datafile(self, fname, name=None, keep_current_data=False,
                     save_args={}, load_args={}):
        """Sets the data_file."""
        with self.sphere_lock:
            self.data_file = fname
            if name is None:
                self.name = os.path.split(fname)[-1].split('.')[0]
            else:
                self.name = name
            if keep_current_data:
                self.save_to_nexus(replace=True, **save_args)
            else:
                if os.path.exists(fname):
                    self.load_from_h5(replace=True, **load_args)
                else:
                    self.reset()
                    self.save_to_nexus(replace=True, **save_args)

    def _set_args(self, args):
        """Ensures any range args are lists."""
        for arg in args:
            if 'range' in arg:
                if args[arg] is not None:
                    args[arg] = list(args[arg])

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def default_geometry(self):
        """Return ``self.geometry``, constructing a sensible default if unset.

        The default is a two-circle convention with detector arm named
        ``tth`` and sample tilt named ``self.incidence_motor`` (which
        falls back to ``"th"``).  Suitable for standard transmission
        and 2-circle GI experiments.  Override by assigning a different
        :class:`DiffractometerGeometry` to ``self.geometry`` before
        saving in v2 mode.
        """
        if self.geometry is not None:
            return self.geometry
        from ssrl_xrd_tools.core.geometry import DiffractometerGeometry
        self.geometry = DiffractometerGeometry.two_circle(
            tth="tth", th=self.incidence_motor or "th",
        )
        return self.geometry

    # ------------------------------------------------------------------
    # v2 NeXus loader (xdart 0.37+ schema)
    # ------------------------------------------------------------------

    def _load_from_nexus_v2(self, grp, *, data_only: bool = False) -> None:
        """Populate sphere state from a v2 NXroot.

        C5: uses :func:`read_sphere_metadata` (not the full
        ``read_sphere``) — we only need frame_index, axes, motor
        columns, and the reduction provenance attrs.  The heavy
        ``intensity_1d`` / ``intensity_2d`` stacks stay on disk and
        :class:`ArchSeries.__getitem__` lazy-loads each frame's
        slices on demand via :func:`_load_arch_v2`.  For a 10k-frame
        Eiger scan this is the difference between ~seconds (full
        materialisation) and ~tens of ms (frame index + a few KB of
        coords + motor columns).
        """
        # Prefer the metadata-only loader — it skips the heavy stacks.
        # Fall back to the full ``read_sphere`` if the metadata loader
        # isn't available on older ssrl_xrd_tools installs.
        try:
            from ssrl_xrd_tools.io.nexus import read_sphere_metadata as _read
        except ImportError:
            from ssrl_xrd_tools.io.nexus import read_sphere as _read

        try:
            ds = _read(self.data_file)
        except Exception as exc:
            # Surface the failure: silent debug-logging meant users saw an
            # empty viewer with no hint why.  exc_info=True dumps the
            # stacktrace into the xdart log so the bug is diagnosable.
            logger.exception(
                "v2 NeXus load failed for %s; viewer will be empty: %s",
                self.data_file, exc,
            )
            return
        logger.debug(
            "v2 NeXus load: %s — %d frames, %d motor cols [data_only=%s]",
            self.data_file,
            ds.sizes.get("frame", 0),
            sum(1 for v in ds.data_vars if ds[v].dims == ("frame",)),
            data_only,
        )

        # ── scan_data DataFrame from motor variables ─────────────────
        reserved_vars = {
            "intensity_1d", "sigma_1d", "intensity_2d",
            "thumbnail", "rot1", "rot2", "rot3", "incident_angle",
        }
        motor_cols: dict[str, np.ndarray] = {
            name: np.asarray(ds[name].values)
            for name in ds.data_vars
            if name not in reserved_vars and ds[name].dims == ("frame",)
        }
        if motor_cols:
            self.scan_data = pd.DataFrame(motor_cols)

        # ── wavelength + bai args from reduction config (if present) ─
        reduction = ds.attrs.get("reduction", {}) or {}
        config = reduction.get("config", {}) or {}
        if isinstance(config.get("bai_1d_args"), dict):
            self.bai_1d_args = dict(config["bai_1d_args"])
        if isinstance(config.get("bai_2d_args"), dict):
            self.bai_2d_args = dict(config["bai_2d_args"])
        # geometry config → DiffractometerGeometry instance
        geom_cfg = config.get("geometry")
        if isinstance(geom_cfg, dict) and geom_cfg.get("mapping_json"):
            try:
                from ssrl_xrd_tools.core.geometry import (
                    DiffractometerGeometry,
                )
                mj = geom_cfg["mapping_json"]
                if isinstance(mj, dict):  # already parsed by read_provenance
                    import json as _json
                    mj = _json.dumps(mj)
                self.geometry = DiffractometerGeometry.from_json(mj)
            except Exception:
                logger.debug("Failed to restore geometry from reduction config",
                             exc_info=True)

        # ── global_mask: persisted under /entry/instrument/detector/mask ─
        # Restored here so the displayframe can overlay it on raw images
        # without depending on the wrangler having loaded the mask file.
        try:
            import h5py
            with h5py.File(self.data_file, "r") as _f:
                _det = _f.get("entry/instrument/detector")
                if _det is not None and "mask" in _det:
                    self.global_mask = np.asarray(_det["mask"][()], dtype=np.int64)
        except Exception:
            logger.debug("Failed to read global_mask from %s",
                         self.data_file, exc_info=True)

        # ── populate arches.index (always — required even in data_only
        # mode so the wrangler's per-frame `update_sphere` refresh has
        # something for the GUI's listData to render).  ArchSeries lazy-
        # loads each arch on demand from the stacked v2 datasets via
        # _load_arch_v2; nothing to materialize up-front.
        #
        # An empty file (e.g. the wrangler has created the .nxs but
        # hasn't flushed the first batch yet) returns a Dataset with
        # no ``frame`` dim — start with an empty index in that case.
        # Don't raise: the wrangler immediately follows with new arches.
        try:
            if "frame" in ds.dims:
                frame_indices = (
                    np.asarray(ds["frame"].values).astype(int).tolist()
                )
            else:
                frame_indices = []
            empty_series = ArchSeries(
                self.data_file, self.file_lock,
                static=self.static, gi=self.gi,
            )
            for idx in frame_indices:
                if idx not in empty_series.index:
                    empty_series.index.append(idx)
            empty_series.index.sort()
            self.arches = empty_series
        except Exception:
            logger.exception(
                "v2 NeXus load: failed to populate arches.index from %s",
                self.data_file,
            )
            return
        logger.debug(
            "v2 NeXus load: populated arches.index (%d frames) on %r",
            len(self.arches.index),
            self.name,
        )

        # ── data_only short-circuit ──────────────────────────────────
        # In data_only mode (live-mode per-frame refresh from
        # file_thread.update_sphere) we stop here.  bai_args / geometry
        # were already restored above; no need to redo them per frame.
        if data_only:
            return
        if "intensity_1d" not in ds.data_vars:
            return


