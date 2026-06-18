import logging
import os
from threading import Condition, _PyRLock

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .frame import LiveFrame
from .frame_series import LiveFrameSeries
from xdart import utils
from xdart.modules.live_compat import normalize_live_class_names
from xdart.modules.wavelength import (
    DEFAULT_WAVELENGTH_SENTINEL_M,
    wavelength_angstrom_to_m,
)


def _coerce_scan_info(scan_info):
    """Return ``scan_info`` with numeric-looking values coerced to float and
    genuinely non-numeric values (tags, status, comments, timestamps) kept as
    strings.  Every key is preserved.

    The numeric-vs-string decision for each *column* belongs at the NeXus
    writer (numeric -> float32, non-numeric -> vlen UTF-8 string), not here:
    dropping a whole column because one value won't parse silently loses
    provenance (N2 -- e.g. a SPEC counter reported as ``"0V"``).  Numeric
    strings (SPEC often stores numbers as text) still coerce, so motors --
    including the GI ``th`` incidence motor and monitor counts -- stay numeric.
    """
    out = {}
    for key, value in scan_info.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value if isinstance(value, str) else str(value)
    return out


class LiveScan:
    """Stateful xdart live scan in v2 NeXus-formatted HDF5 files.

    Output file structure (xdart v2 schema, written by
    :func:`xdart.modules.ewald.nexus_writer.save_scan_to_nexus`)::

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

    def __init__(self, name='scan0', frames=None, data_file=None,
                 scan_data=None, mg_args=None,
                 bai_1d_args=None, bai_2d_args=None,
                 static=False, gi=False, th_mtr=None,
                 incidence_motor=None, geometry=None,
                 series_average=False,
                 single_img=False,
                 global_mask=None,
                 detector_shape=None,
                 file_lock=None,
                 **_unused,
                 ):
        super().__init__()
        # None-sentinel pattern: mutable defaults (lists, dicts, DataFrames)
        # in function signatures are shared across all calls that omit them,
        # so any mutation leaks between instances. Resolve them here instead.
        if frames is None:
            frames = []
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
        # lock that ``LiveFrameSeries.__getitem__`` uses for lazy loads.
        # Pre-J2 each scan created its own Condition() and the
        # wrangler's GUI file_lock was a *different* lock — direct
        # ``LiveFrameSeries.__getitem__`` reads could happen mid-save
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
        self.geometry = geometry            # xrd_tools.core.geometry.DiffractometerGeometry
        # Optional stitched-output containers (populated by run_stitch).
        self.stitched_1d = None
        self.stitched_2d = None
        self.single_img = single_img
        self.series_average = series_average
        self.skip_2d = False
        # Real wavelength restored from a loaded v2 .nxs (G1).  Authoritative
        # when set (display/adapters may trust even an exactly-1.0 Å value);
        # MUST be cleared whenever the scan's data identity changes without a
        # v2 load -- see _clear_persisted_wavelength.
        self._persisted_wavelength_m = None
        self._cached_integrator = None
        # PONI used to build ``_cached_integrator`` -- stashed so the
        # reintegration path can pass it to the headless GI session (which
        # requires ``scan.poni`` to build the fiber integrator).
        self._cached_poni = None
        self._cached_fiber_integrator = None

        if frames:
            self.frames = LiveFrameSeries(self.data_file, self.file_lock, frames,
                                          static=self.static, gi=self.gi)
        else:
            self.frames = LiveFrameSeries(self.data_file, self.file_lock,
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
        self.scan_lock = Condition(_PyRLock())

        # G2: ``overall_raw`` was a sum-of-raw-frames accumulator
        # consumed only by display_data.get_scan_map_raw (now
        # deleted).  Drifted from disk under R1 replace-frames and
        # was never repopulated on v2 reload.
        self.global_mask = global_mask
        # Full-resolution detector (raw image) shape (H, W) — the shape the flat
        # ``global_mask`` indices index into.  Persisted to the .nxs so a reloaded
        # thumbnail-only scan can map the detector gap mask into thumbnail
        # coordinates without needing a resident full-res frame this session.
        self.detector_shape = (
            tuple(detector_shape) if detector_shape is not None else None)

    def reset(self):
        """Resets all held data objects to blank state."""
        with self.scan_lock:
            self.scan_data = pd.DataFrame()
            self.frames = LiveFrameSeries(self.data_file, self.file_lock,
                                          static=self.static, gi=self.gi)
            self.global_mask = None
            self.detector_shape = None
            self._clear_persisted_wavelength()

    def _clear_persisted_wavelength(self):
        """Drop the wavelength restored from a previously loaded .nxs (G1).

        The persisted value short-circuits ``_get_wavelength`` AHEAD of the
        current file's own stamp, so it must be cleared at every data-identity
        change that does not run ``_load_from_nexus_v2`` (``reset()``, the
        live/XYE ``set_datafile`` repoints, ``new_scan``) -- otherwise a run
        into file B keeps converting Q↔2θ with file A's wavelength."""
        self._persisted_wavelength_m = None
        if isinstance(self.mg_args, dict):
            self.mg_args["wavelength"] = DEFAULT_WAVELENGTH_SENTINEL_M

    def has_reload_only_frames(self) -> bool:
        """Return True iff any frame can't recover its raw image.

        Used by the GUI reintegrate buttons (the R3 guardrail).  After
        L1 wired lazy raw load, an frame's ``is_reload_only`` is True
        only when neither ``frame.map_raw`` nor a resolvable
        ``frame.source_file`` is available — i.e. the original raw
        frames have been moved/deleted relative to the .nxs.

        Path A — in-memory cache has entries: definitive answer from
        the cache.  Re-integration buttons get the right answer
        instantly.

        Path B — empty cache (freshly opened .nxs, before any frame
        has been materialised): probe the first frame's
        ``/entry/frames/frame_NNNN/source/path`` directly from h5py
        WITHOUT materialising a :class:`LiveFrame`.  This is the C2
        fix — the pre-C2 code returned ``True`` whenever the cache
        was empty + the index had rows, which falsely blocked
        re-integration on a freshly opened .nxs even when every
        source file was present and lazy load would have worked.
        Probing one frame is a single tiny h5 read (~ms).
        """
        in_mem = getattr(self.frames, "_in_memory", None)
        if in_mem:
            return any(getattr(a, "is_reload_only", False)
                       for a in in_mem.values())
        if not self.frames.index:
            return False
        return self._probe_reload_only_via_h5()

    @property
    def frame_indices(self) -> list[int]:
        """Ordered labels for the headless ``FrameSource`` boundary."""
        return [int(idx) for idx in self.frames.index]

    def load_frame(self, index: int) -> np.ndarray:
        """Load one detector frame without retaining newly hydrated raw data."""
        frame = self.frames[int(index)]
        loaded_here = frame.map_raw is None
        if loaded_here and not frame._lazy_load_raw():
            raise RuntimeError(f"could not lazy-load raw frame {index}")
        try:
            return np.asarray(frame.map_raw)
        finally:
            if loaded_here:
                frame.map_raw = None

    def iter_chunks(self, chunk_size: int):
        """Yield bounded raw-image chunks for headless RSM consumers."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")
        for start in range(0, len(self.frames.index), chunk_size):
            indices = [int(idx) for idx in self.frames.index[start:start + chunk_size]]
            yield np.stack([self.load_frame(idx) for idx in indices]), indices

    def _probe_reload_only_via_h5(self) -> bool:
        """Probe the .nxs file's per-frame source refs to gauge the flag.

        K3: scans **all distinct** ``/entry/frames/frame_NNNN/source/path``
        values in the file (deduplicated) and checks that each
        resolves to an existing file.  For SPEC scans this is
        typically a 1000-element loop over the same TIF-pattern
        directory; for Eiger this is at most a handful of master
        files.  Either way the cost is dominated by the h5 reads
        (~few KB) — file existence checks come from the kernel's
        dentry cache after the first miss.

        Returns ``True`` (block re-integration) on any of:
        - read error
        - any frame missing a ``source/path`` field
        - any referenced source file missing from disk

        Returns ``False`` only when every distinct source path
        resolves.  This is stricter than the C2-style single-probe
        but cheap enough to do at GUI-button click time, and
        catches the mixed-source case where a scan was assembled
        from multiple data sources (rare but possible during
        live-mode acquisition from multiple SPEC sessions, or for
        scans manually edited post-hoc).
        """
        try:
            from xdart.utils import catch_h5py_file as _catch
        except Exception:
            return True
        try:
            indices = list(self.frames.index)
            if not indices:
                return True
        except (TypeError, AttributeError):
            return True

        # Cap the scan at a reasonable size — past ~2000 frames the
        # per-h5-read cost adds up, and a well-behaved scan with
        # 10k frames almost certainly has uniform source refs.
        # Sample first + last + every Nth in between to keep the
        # probe bounded.
        _MAX_PROBE = 256
        if len(indices) > _MAX_PROBE:
            step = max(1, len(indices) // _MAX_PROBE)
            probe_ids = indices[::step]
            if indices[-1] not in probe_ids:
                probe_ids.append(indices[-1])
        else:
            probe_ids = indices

        try:
            with _catch(self.data_file, 'r') as f:
                seen_paths: set = set()
                for idx in probe_ids:
                    grp_path = f"entry/frames/frame_{int(idx):04d}/source"
                    grp = f.get(grp_path)
                    if grp is None or "path" not in grp:
                        return True
                    raw = grp["path"][()]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    seen_paths.add(str(raw))
        except (OSError, KeyError, ValueError, TypeError, AttributeError):
            return True

        # Now verify each distinct source path actually exists.
        # File-existence checks are cheap relative to the h5 reads
        # we already paid for above.
        data_dir = os.path.dirname(self.data_file)
        for raw in seen_paths:
            try:
                full = raw
                if not os.path.isabs(full):
                    full = os.path.normpath(os.path.join(data_dir, full))
                if not os.path.exists(full):
                    return True
            except (OSError, TypeError):
                return True
        return False

    def _probe_reload_only_via_h5_legacy_single(self) -> bool:
        """Pre-K3 single-frame probe.  Kept for tests + emergency
        fallback if the multi-probe path turns out to be too slow on
        some pathological file.
        """
        try:
            from xdart.utils import catch_h5py_file as _catch
        except Exception:
            return True
        try:
            first_idx = int(self.frames.index[0])
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

    def add_frame(self, frame=None, calculate=True, update=True, get_sd=True,
                 h5file=None, batch_save=False, **kwargs):
        """Adds a new frame to the scan.

        In-memory state (``frames.index``, ``scan_data``) and the persisted
        scan state are always updated.  The legacy ``update`` argument is
        accepted for old call sites but no longer changes this behaviour.
        Persistence policy:

        * ``batch_save=True`` — skip the per-frame ``_save_to_nexus`` call;
          the caller is responsible for invoking the v2 writer once at end
          of batch.
        * ``batch_save=False`` (default, used by live mode) — call
          ``_save_to_nexus(h5file)`` once per frame.  The v2 writer is
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
        # land in **kwargs and are forwarded to LiveFrame below.
        kwargs.pop("set_mg", None)
        with self.scan_lock:
            if frame is None:
                frame = LiveFrame(**kwargs)
            if calculate:
                frame.integrate_1d(global_mask=self.global_mask, **self.bai_1d_args)
                frame.integrate_2d(global_mask=self.global_mask, **self.bai_2d_args)
            frame.file_lock = self.file_lock
            # In-memory append only; LiveFrameSeries.__setitem__ does no disk I/O.
            # Fast path: frames arrive in order, so append without the
            # O(N log N) re-sort.  Only an out-of-order index (rare:
            # reintegrate/reload) pays for a sort — keeps long live scans
            # O(1) per frame instead of O(N log N).
            index = self.frames.index
            if frame.idx not in index:
                if not index or frame.idx > index[-1]:
                    index.append(frame.idx)
                else:
                    index.append(frame.idx)
                    index.sort()
            # Stash the live frame object so the v2 writer can read its
            # int_1d/int_2d/thumbnail without going back to disk.
            self.frames.stash(frame)

            if frame.scan_info and get_sd:
                # Keep every metadata field heterogeneous (numeric coerced to
                # float, non-numeric kept as strings) so non-numeric provenance
                # (sample tag, status, "0V" counter) survives to the writer,
                # which persists numeric columns as float32 and non-numeric as
                # vlen UTF-8 strings (N2).  No forced float64 dtype: pandas
                # infers per column.
                coerced_info = _coerce_scan_info(frame.scan_info)
                if coerced_info:
                    ser = pd.Series(coerced_info)
                    if list(self.scan_data.columns):
                        try:
                            self.scan_data.loc[frame.idx] = ser
                            # In-order fast path: frames usually arrive
                            # ascending, so the row just added is already last
                            # — only sort when it landed out of order (rare:
                            # late/reordered frame).  Avoids an O(N log N) sort
                            # every frame.
                            sidx = self.scan_data.index
                            if len(sidx) >= 2 and sidx[-1] < sidx[-2]:
                                self.scan_data.sort_index(inplace=True)
                        except (ValueError, TypeError):
                            logger.debug(
                                'scan_data column mismatch for frame %s '
                                '(have %s, got %s)',
                                frame.idx, list(self.scan_data.columns),
                                list(coerced_info),
                            )
                    else:
                        self.scan_data = pd.DataFrame(
                            [coerced_info], index=[frame.idx]
                        )

            # G2: ``self.overall_raw += (frame.map_raw - frame.bg_raw)``
            # removed.  The accumulator's only consumer was
            # ``display_data.get_scan_map_raw``; the Overall view
            # now aggregates over per-frame ``data_2d['map_raw']``,
            # which stays correct after R1 replace-frames and v2
            # reload.

            # Persist via the v2 writer.  Idempotent; one slice-assign per
            # stacked dataset.  Skipped in batch mode (caller flushes once
            # at end of batch).  The writer opens its own file — ignore
            # any ``h5file`` argument (kept for signature compatibility).
            if not batch_save:
                self._save_to_nexus()

    # ------------------------------------------------------------------
    # v2 NeXus persistence
    # ------------------------------------------------------------------

    def save_to_nexus(self, *, entry: str = "entry", finalize: bool = False,
                      replace: bool = False,
                      replace_frame_indices=None) -> None:
        """Save scan state into a v2 NeXus file.  Idempotent across calls.

        Two modes — see :func:`nexus_writer.save_scan_to_nexus` for
        the full contract:

        * ``replace_frame_indices=None`` (default): append-only.  Stacked
          datasets grow by however many new frames have been added
          since the last save.
        * ``replace_frame_indices=[...]``: slice-assign recomputed
          integrated_1d/2d rows over their existing on-disk positions.
          Used by GUI reintegration (``scan_threads.bai_1d_all``).

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
        """Inner v2 writer; delegates to ``nexus_writer.save_scan_to_nexus``.

        Opens the file at ``self.data_file`` internally.  Callers must
        NOT hold an open h5py.File on the same path during this call —
        HDF5 single-writer semantics will reject the second open.
        """
        from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
        with self.scan_lock:
            save_scan_to_nexus(
                self, self.data_file, mode=mode,
                entry=entry, finalize=finalize,
                replace_frame_indices=replace_frame_indices,
            )
            # Persist-before-evict (data-loss guard): every frame now in the
            # index is written to disk, so the in-memory cache may evict them.
            # Until this mark, ``LiveFrameSeries.stash`` refuses to drop them
            # (their int_1d/int_2d live only on the in-memory LiveFrame). Only
            # reached on a successful save — a raising writer leaves the frames
            # unmarked (and therefore un-evictable).
            mark = getattr(self.frames, "mark_persisted", None)
            if callable(mark):
                mark(list(self.frames.index))

    def load_from_h5(self, replace=True, mode='r', *args, **kwargs):
        """Load scan state from a v2 NeXus file.

        Kept the legacy method name (``load_from_h5``) so callers across
        the GUI don't need updating, but the implementation is v2-only.
        """
        with self.file_lock:
            if replace:
                self.reset()
            with utils.catch_h5py_file(self.data_file, mode=mode) as file:
                self._load_from_h5(file, *args, **kwargs)

    def _load_from_h5(self, grp, data_only=False, **_unused):
        """Load scan state from a v2 NeXus file (no-op dispatcher).

        ``**_unused`` absorbs any historical kwargs (notably
        ``set_mg`` from before F6 removed the multigeometry API).
        """
        with self.scan_lock:
            self._load_from_nexus_v2(grp, data_only=data_only)

    def set_datafile(self, fname, name=None, keep_current_data=False,
                     save_args=None, load_args=None):
        """Sets the data_file.

        N5: save_args / load_args switched to None-sentinels (was
        ``{}`` mutable defaults — shared across all callers who
        omitted the kwarg).  Same trap F4 / F5 / H3 fixed elsewhere.
        """
        save_args = {} if save_args is None else save_args
        load_args = {} if load_args is None else load_args
        with self.scan_lock:
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
        from xrd_tools.core.geometry import DiffractometerGeometry
        self.geometry = DiffractometerGeometry.two_circle(
            tth="tth", th=self.incidence_motor or "th",
        )
        return self.geometry

    # ------------------------------------------------------------------
    # v2 NeXus loader (xdart 0.37+ schema)
    # ------------------------------------------------------------------

    def _load_from_nexus_v2(self, grp, *, data_only: bool = False) -> None:
        """Populate scan state from a v2 NXroot.

        C5: uses :func:`read_scan_metadata` (not the full
        ``read_scan``) — we only need frame_index, axes, motor
        columns, and the reduction provenance attrs.  The heavy
        ``intensity_1d`` / ``intensity_2d`` stacks stay on disk and
        :class:`LiveFrameSeries.__getitem__` lazy-loads each frame's
        slices on demand via :func:`_load_frame_v2`.  For a 10k-frame
        Eiger scan this is the difference between ~seconds (full
        materialisation) and ~tens of ms (frame index + a few KB of
        coords + motor columns).
        """
        self._clear_persisted_wavelength()

        # Prefer the metadata-only loader — it skips the heavy stacks;
        # fall back to the full reader on older builds that lack it.
        try:
            from xrd_tools.io.nexus import read_scan_metadata as _read
        except ImportError:
            from xrd_tools.io.nexus import read_scan as _read

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
        loaded_scan_data = False
        if motor_cols:
            # N2: index scan_data by the actual frame IDs (from the
            # ``frame`` coord), not 0..N-1 default range index.
            # Live acquisition writes rows by ``frame.idx`` via
            # ``self.scan_data.loc[frame.idx] = ser`` — and the v2
            # writer stamps ``integrated_*/frame_index`` from the
            # same frame IDs.  If we left scan_data on the default
            # range index, ``loc[frame.idx]`` on a reload would
            # silently misalign for 1-based SPEC, gapped IDs, or
            # any non-zero-based scheme.
            frame_index = None
            if "frame" in ds.coords:
                try:
                    frame_index = np.asarray(
                        ds["frame"].values, dtype=int,
                    )
                except (TypeError, ValueError):
                    frame_index = None
            if frame_index is not None and len(frame_index) == len(
                next(iter(motor_cols.values()))
            ):
                self.scan_data = pd.DataFrame(motor_cols, index=frame_index)
            else:
                # Fall back to default range index if ds lacks a
                # ``frame`` coord (very old files / partial writes).
                self.scan_data = pd.DataFrame(motor_cols)
            loaded_scan_data = True

        # ── wavelength + bai args from reduction config (if present) ─
        reduction = normalize_live_class_names(ds.attrs.get("reduction", {}) or {})
        config = reduction.get("config", {}) or {}
        if isinstance(config.get("bai_1d_args"), dict):
            self.bai_1d_args = dict(config["bai_1d_args"])
        if isinstance(config.get("bai_2d_args"), dict):
            self.bai_2d_args = dict(config["bai_2d_args"])
        if isinstance(config.get("gi_config"), dict):
            self.gi_config = dict(config["gi_config"])
        # geometry config → DiffractometerGeometry instance
        geom_cfg = config.get("geometry")
        if isinstance(geom_cfg, dict) and geom_cfg.get("mapping_json"):
            try:
                from xrd_tools.core.geometry import (
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

        # v2 writer stores real calibration wavelength at
        # /entry/instrument/source/wavelength_A.  Restore it into mg_args so a
        # reloaded scan can do Q↔2θ display conversion without falling back to
        # the LiveScan constructor's 1e-10 m placeholder.  Read through the
        # ALREADY-OPEN handle (``grp``) — the caller holds the file open
        # (Append loads use mode='a'), so a second ``h5py.File`` open of the
        # same path is fragile under HDF5 file locking.
        try:
            wl_m = wavelength_angstrom_to_m(
                grp["entry/instrument/source/wavelength_A"][()]
            )
            if wl_m is not None:
                self._persisted_wavelength_m = wl_m
                self.mg_args["wavelength"] = wl_m
        except KeyError:
            logger.debug("No persisted instrument/source wavelength in %s",
                         self.data_file)
        except Exception:
            logger.debug("Failed reading persisted wavelength from %s",
                         self.data_file, exc_info=True)

        # ── global_mask: persisted under /entry/instrument/detector/mask ─
        # Restored here so the displayframe can overlay it on raw images
        # without depending on the wrangler having loaded the mask file.
        # Read through the already-open handle (same rationale as the
        # wavelength restore above — no double-open of a file the caller
        # holds open in mode 'a').
        try:
            _det = grp.get("entry/instrument/detector")
            if _det is not None and "mask" in _det:
                self.global_mask = np.asarray(_det["mask"][()], dtype=np.int64)
            # Full-res detector shape (H, W) the flat mask indices index into —
            # lets the display map the gap mask onto a reloaded thumbnail without
            # a resident full-res frame.  Absent in old files (falls back).
            if _det is not None and "detector_shape" in _det:
                _ds = np.asarray(_det["detector_shape"][()]).ravel()
                if _ds.size >= 2:
                    self.detector_shape = (int(_ds[0]), int(_ds[1]))
        except Exception:
            logger.debug("Failed to read global_mask from %s",
                         self.data_file, exc_info=True)

        # ── populate frames.index (always — required even in data_only
        # mode so the wrangler's per-frame `update_scan` refresh has
        # something for the GUI's listData to render).  LiveFrameSeries lazy-
        # loads each frame on demand from the stacked v2 datasets via
        # _load_frame_v2; nothing to materialize up-front.
        #
        # An empty file (e.g. the wrangler has created the .nxs but
        # hasn't flushed the first batch yet) returns a Dataset with
        # no ``frame`` dim — start with an empty index in that case.
        # Don't raise: the wrangler immediately follows with new frames.
        try:
            frame_indices = []
            for coord_name in ("frame", "frame_2d"):
                if coord_name in ds.coords:
                    frame_indices.extend(
                        np.asarray(ds[coord_name].values).astype(int).tolist()
                    )
            frame_indices = sorted(set(frame_indices))
            empty_series = LiveFrameSeries(
                self.data_file, self.file_lock,
                static=self.static, gi=self.gi,
            )
            for idx in frame_indices:
                if idx not in empty_series.index:
                    empty_series.index.append(idx)
            empty_series.index.sort()
            self.frames = empty_series
            if frame_indices:
                if loaded_scan_data:
                    self.scan_data = self.scan_data.reindex(frame_indices)
                elif not data_only and len(self.scan_data) == 0:
                    self.scan_data = pd.DataFrame(index=frame_indices)
        except Exception:
            logger.exception(
                "v2 NeXus load: failed to populate frames.index from %s",
                self.data_file,
            )
            return
        logger.debug(
            "v2 NeXus load: populated frames.index (%d frames) on %r",
            len(self.frames.index),
            self.name,
        )

        # ── data_only short-circuit ──────────────────────────────────
        # In data_only mode (live-mode per-frame refresh from
        # file_thread.update_scan) we stop here.  bai_args / geometry
        # were already restored above; no need to redo them per frame.
        if data_only:
            return
        if "intensity_1d" not in ds.data_vars:
            return

    def load_frame_index_only(self, fname: "str | None" = None) -> int:
        """Populate ONLY ``self.frames`` (the lazy frame index) from a v2 .nxs,
        leaving ``scan_data`` / ``bai_*_args`` / ``global_mask`` / geometry — and
        the GUI's separate display caches — untouched.

        The streaming live path writes the .nxs but never adds to ``self.frames``
        (the lazy series re-integration iterates), so after a live run the
        Reintegrate buttons have nothing to iterate.  Once the run has finished
        and the writer has closed the file, this rebuilds the lazy index from it
        so reintegration works immediately — matching the post-batch behavior —
        WITHOUT a full ``set_datafile`` reload (which ``reset()``s the scan and
        reloads every field).  Frames stay lazy: raw images materialize on demand
        from the stacked datasets / source files when a re-integration reads them.

        Returns the number of frames indexed (0 on empty file / failure).
        """
        with self.file_lock, self.scan_lock:
            if fname is not None:
                self.data_file = fname
            if not self.data_file or not os.path.exists(self.data_file):
                return 0
            try:  # metadata-only reader skips the heavy stacks; fall back if absent
                from xrd_tools.io.nexus import read_scan_metadata as _read
            except ImportError:
                from xrd_tools.io.nexus import read_scan as _read
            try:
                ds = _read(self.data_file)
                frame_indices = []
                for coord_name in ("frame", "frame_2d"):
                    if coord_name in ds.coords:
                        frame_indices.extend(
                            np.asarray(ds[coord_name].values).astype(int).tolist())
                frame_indices = sorted(set(frame_indices))
                series = LiveFrameSeries(
                    self.data_file, self.file_lock,
                    static=self.static, gi=self.gi)
                for idx in frame_indices:
                    if idx not in series.index:
                        series.index.append(idx)
                series.index.sort()
                self.frames = series
                return len(series.index)
            except Exception:
                logger.exception(
                    "load_frame_index_only failed for %s", self.data_file)
                return 0


__all__ = ["LiveScan"]
